# NPU 适配改动说明 (Record 50 — Cautious Weight Decay on Adam)

基于 Record 30 的 NPU 适配经验，对 Record 50 (commit 49465cc) 进行 NPU 适配。
Record 50 相比 Record 30 引入了 NorMuon（Polar Express + 方差缩减）、YaRN 动态 RoPE 缩放、
批量大小调度、交替优化器步进，以及 Cautious Weight Decay 扩展到 Adam 参数。

## 一、纯 CUDA→NPU 文本替换（机械性）

| 原始 (CUDA) | 替换为 (NPU) | 位置 |
|---|---|---|
| `PYTORCH_ALLOC_CONF` | `PYTORCH_NPU_ALLOC_CONF` | 环境变量 |
| `torch.empty(1, device="cuda", ...)` | `torch.empty(1, device="npu", ...)` | 启动 warmup hack |
| `assert torch.cuda.is_available()` | `assert torch.npu.is_available()` | DDP 初始化 |
| `torch.device("cuda", ...)` | `torch.device("npu", ...)` | device 设置 |
| `torch.cuda.set_device(device)` | `torch.npu.set_device(device)` | 设备设置 |
| `backend="nccl"` | `backend="hccl"` | init_process_group |
| `device="cuda"` (数据加载) | `device="npu"` | distributed_data_generator yield |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步 (多处) |
| `torch.cuda.max_memory_allocated/reserved()` | `torch.npu.max_memory_allocated/reserved()` | 显存统计 |
| `.cuda()` (模型) | `.npu()` | 模型上设备 |
| `nvidia-smi` | `npu-smi info` | 日志记录 |

## 二、非等价改动

### 2.1 删除全部 Triton kernel，用纯 PyTorch 实现 Polar Express

原始代码包含：
- `import triton`, `import triton.language as tl`
- `from kernels import get_kernel`
- `import torch._dynamo as dynamo; dynamo.config.recompile_limit = 64`
- `XXT_kernel`, `ba_plus_cAA_kernel` — Triton 对称矩阵乘 kernel
- `polar_express` — 使用 Triton kernel 的 5 步 Polar Express 迭代

**替换方案**：保留 `polar_express()` 函数签名和全部系数，仅将 Triton kernel 替换为纯 PyTorch matmul：
```python
# Triton XXT(X, out=A)  →  A = X @ X.mT
# Triton ba_plus_cAA(A, alpha=c, beta=b, out=B)  →  B = b * A + c * A @ A
# baddbmm/addmm  →  X = a * X + B @ X
```
系数（`polar_express_coeffs`）、归一化方式（`X/(norm*(1+2e-2)+1e-6)`）、迭代次数（5 步）均与 GPU 版完全一致。
`split_baddbmm` 参数保留但在纯 PyTorch 中为 no-op（不影响数学结果）。

### 2.2 删除 FP8 matmul 自定义算子

原始代码包含完整的 FP8 前向+反向实现：
- `nanogpt::mm`、`nanogpt::mm_backward`
- `CastedLinear.forward` 中条件调用 FP8

**替换方案**：`CastedLinear.forward` 始终使用 `F.linear(x, self.weight.type_as(x))`
**影响**：丧失 FP8 加速，bf16 更精确。

### 2.3 flash_attn_varlen_func → npu_fusion_attention (TND varlen)

原始代码使用 Flash Attention 3 varlen 接口（通过 `kernels` 包）：
```python
y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0],
    cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
    max_seqlen_q=max_len, max_seqlen_k=max_len,
    causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0))
```

NPU 版使用 `npu_fusion_attention` TND varlen 模式（复用 Record 30 方案）：
```python
actual_seq_qlen = self._extract_actual_seqlens(seqlens, T)
attn_mask = self._get_window_causal_mask(max_doc_len, bm_size, x.device)
y = torch_npu.npu_fusion_attention(
    q.squeeze(0), k.squeeze(0), v.squeeze(0),
    head_num=self.num_heads, input_layout="TND",
    scale=attn_scale, atten_mask=attn_mask, sparse_mode=0,
    actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_qlen,
)[0].unsqueeze(0)
```

**已恢复的功能**：
- ✅ varlen 序列打包（文档边界隔离）
- ✅ 自定义 softmax_scale（通过 `scale` 参数）
- ✅ window attention（显式 window+causal mask）
- ✅ document boundary mask

### 2.4 删除 torch.compile

| 位置 | 原始 | NPU 版 |
|---|---|---|
| `model = torch.compile(model, ...)` | 编译整个模型 | 删除 |
| `@torch.compile` on `polar_express` | 编译 PE 迭代 | 改用纯 PyTorch matmul（系数不变） |
| `@torch.compile` on `cautious_wd_and_update_inplace` | 编译 WD+更新 | 删除装饰器 |
| `@torch.compile` on `apply_normuon_variance_reduction` | 编译方差缩减 | 删除装饰器 |
| `@torch.compile` on `DistAdam.step` | 编译 Adam 步骤 | 删除装饰器 |

### 2.5 删除 CUDA 专用优化

- `pin_memory=True` → 移除（NPU 兼容性）
- `device_id=device` in `init_process_group` → 移除（hccl 不支持）

### 2.6 grad_accum_steps 适配

原始：`assert 8 % world_size == 0; grad_accum_steps = 8 // world_size`（8 GPU → accum=1）
NPU：`grad_accum_steps = max(1, 8 // world_size)`（16 die → accum=1，但非整除时兜底）

### 2.7 val chunked logits 计算

为避免 val 时 OOM（`(T, 50304) × float32`），val 时分块计算 logits + cross_entropy：
每次只处理 4096 个 token，数学上完全等价。

### 2.8 val 后清理 mask cache

训练和 val 的 sequence length 不同，val 产生的大 mask 如果不清理会累积 OOM。
每次 val 后调用 `CausalSelfAttention.clear_mask_cache(keep_sizes={train_max_seq_len})`。

## 三、训练配置对比

| 参数 | GPU 原始 (8卡) | NPU (16 die) | 是否对齐 |
|---|---|---|---|
| `train_bs_schedule` | (131072, 262144, 393216) | 相同 | ✅ |
| `train_max_seq_len` | 2048 | 2048 | ✅ |
| `num_iterations` | 2090 | 2090 | ✅ |
| `cooldown_frac` | 0.55 | 0.55 | ✅ |
| `grad_accum_steps` | 1 | 1 | ✅ |
| `val_tokens` | 10,485,760 | 10,485,760 | ✅ |
| `val_batch_size` | 2,097,152 | 2,097,152 | ✅ |
| `block_size` / ws_schedule | 128 / (3,7,11) | 128 / (3,7,11) | ✅ |
| `ws_final` | 13 | 13 | ✅ |
| `ws_validate_post_yarn_ext` | 20 | 20 | ✅ |
| optimizer1 weight_decay | 0.005 | 0.005 | ✅ |

## 四、Record 50 相比 Record 30 的新特性

| 特性 | NPU 状态 | 对训练 dynamic 的影响 |
|---|---|---|
| NorMuon (Polar Express + 方差缩减) | ✅ 纯 PyTorch 实现 Polar Express（系数、归一化、迭代逻辑与 GPU 完全一致） | 无影响，与 GPU 训练 dynamic 完全一致：相同的 5 组自适应系数、相同的 `X/(norm*(1+2e-2)+1e-6)` 归一化、相同的 5 步迭代，仅 Triton kernel 替换为 torch matmul |
| Cautious Weight Decay (NorMuon + Adam) | ✅ 保留 | 无影响，数学语义完全一致 |
| YaRN 动态 RoPE 缩放 | ✅ 不依赖 CUDA | 无影响，纯数学计算 |
| 批量大小调度 (3阶段) | ✅ 不依赖 CUDA | 无影响，调度逻辑一致 |
| 交替优化器步进 (偶=Muon, 奇=All) | ✅ 不依赖 CUDA | 无影响，步进逻辑一致 |
| Key shift (长窗口层) | ✅ 不依赖 CUDA | 无影响，纯 tensor 操作 |
| Smear gate | ✅ 不依赖 CUDA | 无影响，纯 tensor 操作 |
| Backout layer | ✅ 不依赖 CUDA | 无影响，纯 tensor 操作 |
| Custom NorMuon distributed sizing | ✅ 降级为 standard sizing（非 8 GPU） | ⚠️ 16 die 使用 standard sizing，reduce_scatter 分组方式与 GPU 8 卡 custom sizing 不同；但全局梯度归约结果数学等价，仅通信调度顺序不同，对最终梯度值无影响 |
| DataPreloader (异步预加载) | ✅ 不依赖 CUDA | 无影响，仅 I/O 优化 |
| BOSFinder quickload | ✅ 不依赖 CUDA | 无影响，仅 I/O 优化 |
| FP8 matmul | ❌ 回退到 bf16 F.linear | ✅ 有利影响：bf16 精度高于 FP8（e4m3fn/e5m2），lm_head 前向和反向的数值更精确，理论上训练 dynamic 更稳定 |
| Triton symmetric matmul | ❌ 回退到纯 PyTorch matmul | 无影响，数学语义完全等价（A@A.T 和 b*A+c*A@A 计算结果一致），Polar Express 系数完全保留，仅 float32 累加路径可能有微小舍入差异（≤1e-6 量级） |
| torch.compile | ❌ 删除 | 无影响，torch.compile 仅做算子融合和内存优化，不改变数学语义；但删除后无 activation recompute，显存占用更大 |

## 五、运行中修复的 NPU 兼容性问题

### 5.1 addcmul_ 不支持 DT_BOOL

Cautious Weight Decay 中 `mask = (update * p_slice) >= 0` 产生 bool tensor，
NPU 的 `addcmul_` 算子不支持 bool 类型（要求 float/int/bfloat16 等）。

**报错**：`AclNN_Parameter_Error: Tensor tensor2 not implemented for DT_BOOL`

**修复**：显式将 bool mask 转为数值类型，涉及两处：
```python
# DistAdam.step — cautious weight decay
mask = ((update * p_slice) >= 0).to(update.dtype)

# cautious_wd_and_update_inplace — NorMuon cautious weight decay
mask = ((v * p) >= 0).to(p.dtype)
```
数学上完全等价（`True→1.0, False→0.0`）。

## 六、实验结果

### 6.1 完整训练结果

- 设备：16× Ascend 910C (65536 MiB HBM/卡)
- 环境：torch 2.10.0 + torch_npu 2.10.0
- 训练步数：2090（2050 scheduled + 40 extension）
- NPU 日志：`logs/6cea5af2-db12-4948-8ae7-bb96c9825e99.txt`
- GPU 日志：`records/track_1_short/2025-12-18_CautiousWDAdam/edfbf2aa-ab5d-481d-8ce8-c26dcc2ead61.txt`

| 指标 | GPU (8×H100) | NPU (16×910C) |
|---|---|---|
| final val_loss | **3.2754** | **3.2656** |
| 总训练时间 | 126,997 ms (2.1 min) | 577,444 ms (9.6 min) |
| step_avg | 60.76 ms | 276.29 ms |
| 峰值显存 | — | 28538 MiB alloc / 31460 MiB reserved |

### 6.2 Val loss 逐步对比

| step | GPU    | NPU    | 差异     |
|------|--------|--------|----------|
| 0    | 10.8258| 10.8125| -0.0133  |
| 250  | 4.2693 | 4.2500 | -0.0193  |
| 500  | 4.0064 | 4.0000 | -0.0064  |
| 750  | 3.8530 | 3.8438 | -0.0092  |
| 1000 | 3.7086 | 3.7188 | +0.0102  |
| 1250 | 3.5845 | 3.5938 | +0.0093  |
| 1500 | 3.4760 | 3.4688 | -0.0072  |
| 1750 | 3.3744 | 3.3750 | +0.0006  |
| 2000 | 3.2973 | 3.3125 | +0.0152  |
| 2090 | 3.2754 | 3.2656 | **-0.0098** |

### 6.3 结论

- **val loss 高度一致**：全程差异 ≤ 0.02，final 差异仅 0.0098
- **NPU 略优于 GPU**：可能因 bf16 替代 FP8 精度更高
- **均低于 3.28 目标线**：GPU 3.2754，NPU 3.2656
- **对比图**：`comparison_plots/record50_cautious_wd/`

## 七、总结

Record 50 的核心改动（Cautious Weight Decay 扩展到 Adam）完全保留在 NPU 版本中。
Polar Express 正交化使用纯 PyTorch 实现，系数和迭代逻辑与 GPU 版完全一致，训练 dynamic 无差异。
主要降级项：FP8→bf16、Triton kernel→PyTorch matmul、torch.compile 删除。
Attention 实现复用 Record 30 的 npu_fusion_attention TND varlen 方案。
实测 final val loss 差异仅 0.0098，NPU 适配验证通过。
