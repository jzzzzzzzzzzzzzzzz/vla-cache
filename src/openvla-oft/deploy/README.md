# VLA-Cache 服务器部署指南

## 目录结构

```
deploy/
├── README.md                        # 本文件
├── setup_server.sh                  # 一键安装脚本（conda + 依赖 + patch）
├── run_all_bf16.sh                  # 批量实验运行脚本（E0/E1/E5f，BF16）
├── bf16_changes.md                  # INT4 → BF16 适配变更清单
└── modified_site_packages/
    ├── modeling_llama.py            # 已修改：Layer Skip（T11）
    └── cache_utils.py               # 已修改：from_legacy_cache 跳过空 KV slot（Bug3修复）
```

---

## 快速部署步骤

### 步骤 1：上传代码到服务器

```bash
# 本地：push 代码到 GitHub
git add -A && git commit -m "add server deploy package"
git push origin main

# 服务器：clone
git clone https://github.com/jzzzzzzzzzzzzzzzz/vla-cache.git
cd vla-cache/src/openvla-oft
```

### 步骤 2：运行安装脚本

```bash
# 服务器上：
bash deploy/setup_server.sh https://github.com/jzzzzzzzzzzzzzzzz/vla-cache.git
```

脚本会自动：
1. 创建 `openvla-oft` conda 环境（Python 3.10）
2. 安装 PyTorch 2.2.0+cu121
3. 安装 openvla-oft（editable）+ 自定义 transformers fork（VLA-Cache 支持）
4. 安装 LIBERO
5. 安装其余依赖（无 bitsandbytes）
6. 应用两个 site-packages 补丁（modeling_llama.py + cache_utils.py）

### 步骤 3：上传 checkpoint

```bash
# 方案 A：从本地 SCP（推荐）
scp -r checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    user@server:/path/to/vla-cache/src/openvla-oft/checkpoints/

# 方案 B：服务器直接从 HuggingFace 下载（见 bf16_changes.md §8）
```

### 步骤 4：配置环境变量

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 5：运行实验

```bash
conda activate openvla-oft
cd /path/to/vla-cache/src/openvla-oft

# 修改 run_all_bf16.sh 中的 PYTHON 路径（第一次运行前检查）
bash deploy/run_all_bf16.sh
```

---

## 补丁说明

### `modeling_llama.py`（T11 Layer Skip）

在 `LlamaModel.forward()` 的 decoder loop 中插入 layer skip：
```python
for layer_idx, decoder_layer in enumerate(self.layers):
    # ─── Layer Skip (T11) ─────────────────────────────────
    if layer_idx in getattr(self.config, 'skip_layers', set()):
        continue
    # ──────────────────────────────────────────────────────
    ...  # 原有逻辑不变
```

### `cache_utils.py`（Bug3 修复）

`DynamicCache.from_legacy_cache()` 中跳过 skip 层的空 slot：
```python
for layer_idx in range(len(past_key_values)):
    key_states, value_states = past_key_values[layer_idx]
    if not isinstance(key_states, torch.Tensor):  # skipped layer — no KV
        continue
    cache.update(key_states, value_states, layer_idx)
```

---

## 关键实验命令

所有实验均在 BF16 下运行（去掉 `--load_in_4bit True`）：

```bash
PYTHONPATH=/path/to/LIBERO CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 20 \
    [--use_vla_cache True/False] \
    [--skip_layers "3,4,5,6"] \
    --run_id_note "EXPERIMENT_ID"
```

详见 `experiments/research_log.md` 第六节的完整命令模板。
