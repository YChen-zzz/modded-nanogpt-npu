import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Muon optimizer (NPU port: no torch.compile, npu device)

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="npu")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model (NPU port)

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = False

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    _shared_mask_cache: dict = {}

    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

    @classmethod
    def _get_window_causal_mask(cls, size: int, window: int, device):
        key = (size, window)
        if key not in cls._shared_mask_cache:
            row = torch.arange(size, device=device).unsqueeze(1)
            col = torch.arange(size, device=device).unsqueeze(0)
            cls._shared_mask_cache[key] = (col > row) | (col < row - window + 1)
        return cls._shared_mask_cache[key]

    def forward(self, x: Tensor, ve: Tensor | None, window_size: int, actual_seq_qlen: list):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v

        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))
        max_doc_len = min(max_doc_len, 4096)
        attn_mask = self._get_window_causal_mask(max_doc_len, window_size, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.attn_scale,
            atten_mask=attn_mask,
            sparse_mode=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, max_seq_len: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, window_size: int, actual_seq_qlen: list):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, window_size, actual_seq_qlen)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_layers: int, num_embeddings: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_embeddings = num_embeddings
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for _ in range(num_embeddings)])

    def forward(self, input_seq: Tensor) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        ve = [ve[0], ve[1], ve[2]] + [None] * (self.num_layers - 2 * self.num_embeddings) + [ve[0], ve[1], ve[2]]
        return ve

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx, max_seq_len) for layer_idx in range(num_layers)])
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_()

    def forward(self, input_seq: Tensor, target_seq: Tensor, window_size: int):
        assert input_seq.ndim == 1
        T = len(input_seq)
        MAX_DOC_LEN = 4096

        bos_positions = (input_seq == 50256).nonzero(as_tuple=True)[0]
        if len(bos_positions) == 0:
            boundaries = list(range(0, T, MAX_DOC_LEN))
        else:
            boundaries = bos_positions.tolist()
            if boundaries[0] != 0:
                boundaries = [0] + boundaries
        actual_seq_qlen = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else T
            while start < end:
                chunk_end = min(start + MAX_DOC_LEN, end)
                actual_seq_qlen.append(chunk_end)
                start = chunk_end

        x = x0 = norm(self.embed(input_seq)[None])
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        skip_connections = []
        long_ws = window_size
        short_ws = max(window_size // 2, 128)
        block_masks_enc = [long_ws, short_ws, short_ws, short_ws, long_ws, short_ws]
        assert len(block_masks_enc) == self.num_encoder_layers
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_masks_enc[i], actual_seq_qlen)
            skip_connections.append(x)
        block_masks_dec = list(reversed(block_masks_enc))
        assert len(block_masks_dec) == self.num_decoder_layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_masks_dec[i], actual_seq_qlen)
        x = norm(x)
        x_2d = x.view(-1, x.size(-1))
        chunk_size = 4096
        num_tokens = x_2d.size(0)
        total_loss = x_2d.new_zeros((), dtype=torch.float32)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            logits_chunk = self.lm_head(x_2d[start:end])
            logits_chunk = 30 * torch.sigmoid(logits_chunk.float() / 7.5)
            total_loss = total_loss + F.cross_entropy(logits_chunk, target_seq[start:end], reduction="sum")
        loss = total_loss / num_tokens
        return loss
# -----------------------------------------------------------------------------
# Distributed Data Loader (NPU port: no pin_memory)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    files = sorted(Path(filename_pattern).parent.glob(Path(filename_pattern).name))
    if not files:
        files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="npu", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="npu", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets
# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 10485760
    num_iterations = 1770
    cooldown_frac = 0.4
    val_loss_every = 125
    seq_len = 48 * 1024
    val_seq_len = 64 * 1024
    save_checkpoint = False
args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group(backend="hccl")
dist.barrier()
master_process = (rank == 0)

logfile = None
run_dir = None
if master_process:
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    run_dir = os.environ.get("RUN_DIR", f"logs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    logfile = f"{run_dir}/train.log"
    print(run_dir)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")

def npu_smi():
    import subprocess
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(npu_smi())
print0("=" * 100)

train_batch_size = world_size * args.seq_len
train_loader = distributed_data_generator(args.train_files, train_batch_size, rank, world_size)

model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(args.seq_len, args.val_seq_len)).npu()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=False)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]

def get_lr(step: int):
    t = 1 - step / args.num_iterations
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0)
    return w * 1.0 + (1 - w) * 0.1
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float("nan") if step <= 11 else (step - 10) + 1

    new_window_size = next_multiple_of_n(1728 * step / train_steps, n=128)
    if step == 0:
        window_size = new_window_size
    elif new_window_size != window_size:
        window_size = new_window_size
        CausalSelfAttention._shared_mask_cache.clear()
        torch.npu.empty_cache()

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps_count = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps_count):
                x, y = next(val_loader)
                val_loss += model(x, y, window_size)
        val_loss /= val_steps_count
        del val_loader
        torch.npu.empty_cache()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms", console=True)
        model.train()
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"{run_dir}", exist_ok=True)
            torch.save(log, f"{run_dir}/state_step{step:06d}.pt")
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    for input_seq, target_seq in zip(inputs.split(args.seq_len), targets.split(args.seq_len)):
        model(input_seq, target_seq, window_size).backward()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    frac = min(step / 300, 1)
    for group in optimizer2.param_groups:
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms", console=True)

print0(
    f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB",
    console=True,
)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
