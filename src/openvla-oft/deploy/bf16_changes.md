# BF16 适配变更清单

## 概述

本地（local）使用 INT4（bitsandbytes）量化以在 12GB VRAM 内运行 7B 模型。
服务器（4090，24GB）使用 BF16（原生精度），需要去掉量化相关配置。

**模型内存占用**：
- INT4：7B × 0.5 bytes ≈ 3.5 GB（+量化 overhead ≈ 4.5 GB total）
- BF16：7B × 2 bytes ≈ 14 GB（4090 24GB 完全可用）

---

## 变更点详细说明

### 1. 运行命令（最简单的变更）

**本地（INT4）**：
```bash
python run_libero_eval.py \
    --load_in_4bit True \
    ...
```

**服务器（BF16）**：
```bash
python run_libero_eval.py \
    # 不传 --load_in_4bit（默认 False）
    # 不传 --load_in_8bit（默认 False）
    ...
```

无需修改任何代码，仅去掉 `--load_in_4bit True` flag 即可。

---

### 2. `experiments/robot/openvla_utils.py` — 代码行为分析

#### 2.1 模型加载（L288-312）

```python
# L292：BF16 dtype 已经正确设置
vla = AutoModelForVision2Seq.from_pretrained(
    ...
    torch_dtype=torch.bfloat16,   # ✅ 已正确，BF16 和 INT4 都走这行
    load_in_8bit=cfg.load_in_8bit,  # BF16 时 = False
    load_in_4bit=cfg.load_in_4bit,  # BF16 时 = False
    ...
)

# L309：BF16 时 load_in_4bit=False, load_in_8bit=False → 条件为 True → 执行 .to(DEVICE)
if not cfg.load_in_8bit and not cfg.load_in_4bit:
    vla = vla.to(DEVICE)   # ✅ BF16 模型正常移到 GPU
```

**结论**：无需修改，逻辑自动适配。

#### 2.2 KV Cache CPU 卸载（L1032-1034, L1121-1123）

```python
# L1032-1034 注释：
# "It was offloaded to CPU after the previous query to make room for the
# bitsandbytes INT4 dequantisation workspace ~86 MiB per FFN layer."

# 实际卸载代码（L1033-1035）：
_kvcache_to_device(prompt_cache, DEVICE)   # 推理前移回 GPU
# ...推理...
_kvcache_to_device(last_caches['past_key_values'], 'cpu')  # 推理后移到 CPU
```

**INT4 需要卸载的原因**：bitsandbytes 在 FFN forward 时需要 ~86 MB/层的 dequant 工作空间，
32层累计 ~2.7 GB 峰值，必须提前释放 KV cache 占用的空间。

**BF16 下是否需要？**：
- BF16 无 dequant 工作空间需求
- 但 BF16 模型本身占用 14 GB，4090 还剩 ~10 GB
- KV cache（512 token × 32层 × 2(K+V) × 2 bytes ≈ 134 MB）很小
- **建议**：BF16 下可以保留 CPU 卸载（无害，减少峰值占用），也可以去掉（加快速度）
- **不去掉也没问题**，CPU 卸载只增加约 2ms 延迟

**如果要去掉 CPU 卸载以提速**（可选优化）：
在 `openvla_utils.py` 中注释掉以下两行：
```python
# _kvcache_to_device(last_caches['past_key_values'], 'cpu')   # BF16 可注释
# gc.collect(); torch.cuda.empty_cache()                       # BF16 可注释
```

---

### 3. `run_libero_eval.py` — 无需代码修改

`load_in_4bit: bool = False` 是 `GenerateConfig` 的默认值，
不传 `--load_in_4bit True` 时自动为 False，模型以 BF16 加载。

```python
# GenerateConfig (L115-116)：
load_in_8bit: bool = False   # 默认 False
load_in_4bit: bool = False   # 默认 False ← 不传 flag 即为 BF16
```

---

### 4. `modeling_llama.py`（transformers site-packages）— 无需修改

SDPA 路径选择依赖 `output_attentions` flag，与精度无关：
- `modeling_prismatic.py` L912 强制 `output_attentions=True`
- → `LlamaSdpaAttention` fallback 到 `LlamaAttention`（eager mode）
- → 行为与 INT4 完全相同

Layer Skip（T11）的 `continue` 语句与精度无关，BF16 下正常工作。

---

### 5. 不需要修改的文件汇总

| 文件 | 结论 |
|------|------|
| `openvla_utils.py` | 无需修改（load_in_4bit=False 路径已有） |
| `run_libero_eval.py` | 无需修改（仅去掉 --load_in_4bit flag） |
| `modeling_llama.py` | 无需修改（Layer Skip 与精度无关） |
| `cache_utils.py` | 无需修改（已修复的 from_legacy_cache 与精度无关） |
| `vla_cache_utils.py` | 无需修改 |
| `modeling_prismatic.py` | 无需修改（checkpoint 内的文件，与精度无关） |

---

### 6. MUJOCO 渲染配置

服务器（无显示器）必须：
```bash
export MUJOCO_GL=egl
```

本地可用 `osmesa` 或 `glfw`，服务器仅支持 `egl`（headless GPU 渲染）。

---

### 7. 预期性能差异（BF16 vs INT4）

| 指标 | INT4（本地，12GB）| BF16（服务器，4090）| 说明 |
|------|------------------|---------------------|------|
| 模型加载 VRAM | ~4.5 GB | ~14 GB | BF16 占用更多 |
| 前向推理 CUDA 延迟 | ~551ms | **~200-350ms（预期）** | BF16 在 4090 上更快 |
| KV cache 卸载开销 | 必须（BNB 工作空间） | 可选 | BF16 可省去 |
| 数值精度 | INT4（量化噪声） | BF16（更精确） | SR 可能略有差异 |
| 输出确定性 | 较好（已量化） | 好（更高精度） | — |

**注**：BF16 在 4090（SM 8.9，支持 BF16 Tensor Core）上推理速度应显著快于本地 INT4，
预期 baseline 延迟 ≤ 350ms（本地 INT4 ~551ms）。

---

### 8. 服务器 checkpoint 获取方案

checkpoint 大小约 15 GB，有两种方案：

**方案 A：SCP 从本地传输**（推荐，最可靠）
```bash
scp -r checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    user@server:/path/to/vla-cache/src/openvla-oft/checkpoints/
```

**方案 B：HuggingFace 下载**（需要网络访问）
```bash
# checkpoint 来自 OpenVLA-OFT 官方 HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10',
    local_dir='checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10'
)
"
```
