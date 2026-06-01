import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu

from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = y / math.sqrt(24)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.npu(), y.npu()

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, flush=True, **kwargs)

if __name__ == "__main__":
    import time
    import json
    print0(f"Running pytorch {torch.version.__version__}")

    data_path = os.environ.get("DATA_PATH", "/models/modded-nanogpt_record50_cautious_wd")
    input_bin = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin")
    input_val_bin = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin")
    output_dir = os.environ.get("OUTPUT_DIR", "")
    seed = int(os.environ.get("TRAIN_SEED", "42"))

    B = 32
    T = 1024
    total_batch_size = 524288
    num_iterations = 19560
    learning_rate = 6e-4
    warmup_iters = 700
    weight_decay = 0.1
    grad_clip = 1.0
    val_loss_every = 250
    val_max_steps = 20


    assert torch.npu.is_available(), "NPU not available"
    init_process_group(backend='hccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'npu:{ddp_local_rank}'
    torch.npu.set_device(device)
    master_process = ddp_rank == 0
    print(f"rank {ddp_rank}: using device: {device}", flush=True)

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    tokens_per_fwdbwd = B * T * ddp_world_size
    assert tokens_per_fwdbwd == total_batch_size, f"tokens_per_fwdbwd={tokens_per_fwdbwd} != total_batch_size={total_batch_size}"

    ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

    model_config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(model_config)
    model = model.train().npu()
    if master_process:
        print("Model created and moved to NPU", flush=True)

    train_loader = DistributedDataLoader(input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if input_val_bin:
        val_loader = DistributedDataLoader(input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay, learning_rate=learning_rate,
        betas=(0.9, 0.95), device_type=device
    )

    def get_lr(it):
        assert it <= num_iterations
        if it < warmup_iters:
            return learning_rate * (it+1) / warmup_iters
        decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * (0.1 + coeff * 0.9)

    run_id = str(uuid.uuid4())

    logfile = None
    if master_process and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logfile = os.path.join(output_dir, "train.log")
        with open(logfile, "w") as f:
            pass
        with open(os.path.join(output_dir, "command.txt"), "w") as f:
            f.write(f"cwd: {os.getcwd()}\nargv: {' '.join(sys.argv)}\nworld_size: {ddp_world_size}\nseed: {seed}\n")

    timings = []
    norm_val = -1.0
    final_val_loss = None
    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = (step == num_iterations)

        if (val_loss_every > 0 and (step % val_loss_every == 0 or last_step)) and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    with ctx:
                        _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= val_max_steps
            print0(f"step {step} val_loss {val_loss:.6f}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
            if last_step:
                final_val_loss = val_loss

        if last_step:
            break

        model.train()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        x, y = train_loader.next_batch()
        loss.backward()
        norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.npu.synchronize()
        t1 = time.time()
        tokens_per_second = ddp_world_size * B * T / (t1-t0)
        lossf = loss.item()
        if step % 100 == 0 or step < 5:
            print0(f"step {step+1:5d}/{num_iterations} | train loss {lossf:.6f} | norm {norm_val:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        if step > 0 and step > num_iterations - 20:
            timings.append(t1-t0)

    timings = timings[-20:]
    if timings:
        print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")

    if master_process and output_dir:
        metrics = {
            "seed": seed,
            "final_val_loss": final_val_loss,
            "num_iterations": num_iterations,
            "total_batch_size": total_batch_size,
            "learning_rate": learning_rate,
            "world_size": ddp_world_size,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print0(f"Final val loss: {final_val_loss}")

    destroy_process_group()
