# VLA-Cache + Prune 实验进展总结报告

**日期**：2026-03-12
**项目**：VLA-Cache + Prune B 联合框架
**目标**：在 OpenVLA-OFT 上实现视觉 token 剪枝，减少推理计算量，同时保持机器人操作成功率

---

## 一、背景与系统架构

### 1.1 OpenVLA-OFT 推理流程

每个时间步 t，模型执行如下操作：

```
观测图像 → ViT 提取 patch 特征 → LLaMA Decoder（32 层）→ Action Head（L1回归）→ 7-DOF 动作
```

- 双目输入：Primary 相机（256 patches）+ Wrist 相机（256 patches）
- 完整 LLM 序列：`[BOS(1), Primary(256), Wrist(256), Text(~34), Action(56)] ≈ 603 tokens`
- 使用 bitsandbytes INT4 量化，单步推理 ~415ms，约 3.18 TFLOPs

### 1.2 VLA-Cache 机制

VLA-Cache 每步重新注入全部 visual+text tokens，但对"静态"（A类）patch 复用上一步的 KV cache，省去其 Q/K/V 计算。

**Token 分类**（基于像素相似度 × 文本注意力）：

| 类别 | 像素相似度 | 文本注意力 | 处理方式 |
|------|-----------|-----------|---------|
| A (Cache) | 高（静止）| 低 | KV 复用上一步缓存 |
| B (Prune 候选) | 低（动态）| 低 | 目标：跳过计算 |
| C (Recompute) | 高（静止）| 高 | 每步重新计算 |
| D (Recompute) | 低（动态）| 高 | 每步重新计算 |

### 1.3 关键代码文件

| 文件 | 作用 |
|------|------|
| `experiments/robot/openvla_utils.py` | `get_vla_action()` —— VLA-Cache + Prune 主逻辑 |
| `experiments/robot/vla_cache_utils.py` | `task_relevant_selection()`，`find_static_patches()` |
| `experiments/robot/analysis_utils.py` | `FrameStats`，`AttentionHookCapture`，可视化工具 |
| `experiments/robot/libero/run_libero_eval.py` | 成功率评估驱动脚本 |
| `experiments/robot/libero/run_attention_analysis.py` | Phase 1.1 注意力分析驱动脚本 |
| `transformers/.../modeling_llama.py` | LlamaModel forward，含 VLA-Cache 和 Prune B 逻辑 |

---

## 二、Phase 1.1：B 类 Token 可行性分析

### 2.1 目标

在真实推理过程中，统计每一帧各类 token 的比例，验证 B 类 token（低相似度 + 低注意力）是否足够多，值得剪枝。

### 2.2 遇到的工程问题

**问题 1：显存 OOM（torch.no_grad() 缺失）**

- **现象**：运行分析脚本时 GPU OOM 崩溃
- **根本原因**：`predict_action` 未包裹 `torch.no_grad()`，autograd 为 32 层 softmax 保存了大量中间张量（约 693 MB）
- **修复**：在 `openvla_utils.py` 的 `predict_action` 调用处添加 `with torch.no_grad()`
- **位置**：`openvla_utils.py` 第 939 行附近

**问题 2：注意力张量 GPU 显存堆积**

- **现象**：`output_attentions=True` 模式下，32 层注意力张量同时驻留 GPU，约 1.5 GB
- **根本原因**：模型等待所有 32 层 forward 完毕才返回，所有注意力张量同时在 GPU 上
- **修复**：实现 `AttentionHookCapture` 类，为每个 `LlamaDecoderLayer` 注册 forward hook，每层注意力计算完立即转移到 CPU，GPU 峰值从 1.5 GB 降至 ~47 MB（单层）
- **位置**：`analysis_utils.py`，`class AttentionHookCapture`

### 2.3 Phase 1.1 结果

在 LIBERO-Spatial（10 tasks × 10 trials = 1205 frames）和 LIBERO-10（10 tasks × 10 trials = 3217 frames）上统计 B 类 token 比例：

| Suite | 帧数 | Primary B 均值 | Wrist B 均值 | 结论 |
|-------|------|--------------|------------|------|
| libero_spatial | 1205 | **18.31% ± 3.07%** | **20.44% ± 4.10%** | ✓ 可行 |
| libero_10 | 3217 | **18.75% ± 3.24%** | **20.32% ± 5.01%** | ✓ 可行 |

完整类别分布（Primary 相机，libero_spatial 均值）：
- A（Cache）= 42.63%，B（Prune 候选）= 18.31%，C（Recompute，静止+相关）= 15.97%，D（Recompute，动态+相关）= 23.10%

**结论**：B 类 token 约占 18-20%，值得实现剪枝以节省约 18% 额外计算量（在 VLA-Cache 已节省 43% 的基础上）。

---

## 三、Phase 1.2：实现与调试

### 3.1 Phase 1.2 目标

实现 Prune B，并在 LIBERO-Spatial（10 tasks × 20 trials/task）上进行三组对照实验：
- **Baseline**：无 VLA-Cache，无 Prune
- **VLA-Cache**：仅 A 类复用
- **VLA-Cache + Prune**：A 类复用 + B 类物理删除

### 3.2 初步实现

#### 代码变更

**`vla_cache_utils.py`**：`task_relevant_selection()` 添加 `return_prune` 参数，返回 B 类 token 在 LLM 序列空间中的位置索引。

**`openvla_utils.py`** 中 `get_vla_action()` 添加 Prune B 步骤：
1. 调用 `task_relevant_selection(..., return_prune=True)` 获取 B 类位置
2. 将 B 位置 tensor 存入 `vla.language_model.config.prune_patches`
3. 对 prompt_cache 中 B 位置的 KV 执行 zero（清零）
4. 前向传播前 zero 操作完成

**`modeling_llama.py`**（LlamaModel.forward）：在 decoder loop 之前插入 Prune B block：
```python
if prune_patches is not None and inputs_embeds.size(1) != 1:
    _keep = ~torch.isin(cache_position, _prune_dev)
    hidden_states  = hidden_states[:, _keep, :]   # 物理删除 B
    cache_position = cache_position[_keep]         # 更新位置索引
    position_ids   = cache_position.unsqueeze(0)   # 更新 RoPE 位置
    pruned_seq_len = hidden_states.shape[1]
    # ... 掩码处理（见下方调试过程）
```

**`run_libero_eval.py`**：添加 `use_prune: bool = False` 配置项，添加 JSON 格式结果保存。

---

### 3.3 失败 #1：VLA-Cache + Prune 成功率仅 5.5%（灾难性失败）

**Phase 1.2 正式评估结果（200 episodes，20 trials/task）**：

| 条件 | 总成功率 | Per-task 成功率 |
|------|---------|----------------|
| Baseline | **95.5%** (191/200) | [1.0, 1.0, 1.0, 1.0, 0.75, 0.95, 1.0, 0.95, 1.0, 0.9] |
| VLA-Cache | **95.0%** (190/200) | [1.0, 0.95, 1.0, 1.0, 0.75, 1.0, 1.0, 0.9, 1.0, 0.9] |
| VLA-Cache + Prune | **5.5%** (11/200) | [0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |

VLA-Cache + Prune 仅任务 3 有 55% 成功率，其余任务全部失败。

#### 调试分析过程

**关键架构知识点（逐步排查）**：

1. **VLA-Cache 的 `past_seen_tokens=0` 强制覆盖**

   `modeling_llama.py` 第 988 行：
   ```python
   past_seen_tokens = past_key_values.get_seq_length() if self.config.proportion_attn_var is None else 0
   ```
   VLA-Cache 激活时强制 `past_seen_tokens=0`，使得每步都像全新序列一样处理。

2. **SDPA 因果掩码快捷路径**

   `_update_causal_mask` 中（第 1174-1181 行）：
   ```python
   output_attentions = False  # 强制覆盖为 False！
   if self.config._attn_implementation == "sdpa" and not output_attentions:
       if AttentionMaskConverter._ignore_causal_mask_sdpa(
           attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
       ):
           return None   # ← 关键：在 VLA-Cache 下始终返回 None
   ```

   **重点**：第 1174 行将 `output_attentions` **硬编码覆盖为 False**，即使调用者传入 `output_attentions=True`，这里也被忽略。因此，即使 `_regression_or_discrete_prediction` 硬编码了 `output_attentions=True`（第 912 行），`_update_causal_mask` 也仍然走 SDPA 快捷路径。

   `_ignore_causal_mask_sdpa` 的触发条件：`key_value_length == query_length`。VLA-Cache 下 `past_seen_tokens=0`，所以 `key_value_length = 0 + N_orig = N_orig = query_length`，**始终触发**，`causal_mask` 返回 `None`。

3. **`output_attentions=True` 触发 LlamaAttention（eager）回退**

   虽然 `_update_causal_mask` 返回 `None`，但 `LlamaSdpaAttention.forward` 检测到 `output_attentions=True` 时：
   ```python
   if output_attentions:
       return super().forward(...)  # 回退到 LlamaAttention（eager）
   ```
   LlamaAttention（eager）直接使用 `attention_mask`。若 `attention_mask=None`，则执行**全注意力**（无任何掩码）。

4. **Prune B 后 zero KV 导致 softmax 污染（真正根因）**

   最初以为问题出在因果掩码被跳过，实则根因更底层：

   - B 位置的 KV 在前向传播前被**清零**（`key_cache[:, :, B_positions, :] = 0.0`）
   - zero K 使得 `Q · K_B = Q · 0 = 0`，所有 B 位置的注意力 logit 恰好等于 **0**
   - 经 softmax 时，每个 B 位置贡献 `exp(0) = 1`
   - 共约 92 个 B 位置 × `exp(0)=1` = **92 个单位**进入 softmax 分母
   - 当非 B 位置的 logit 多为**负值**（层归一化后的典型情况），`exp(负值) << 1`
   - B 位置的 `exp(0)=1` 远大于非 B 位置的 `exp(负值)`，主导了 softmax 分母
   - **结果**：注意力分布被 B 位置稀释，模型的注意力输出趋近于零（V_B=0），有效信息几乎全部丢失

   **数值估算**：若 510 个非 B 位置平均 logit = -1（exp(-1) ≈ 0.37），则：
   - 非 B 分母贡献：510 × 0.37 ≈ 189
   - B 分母贡献：92 × 1 = 92
   - B 占比：92 / (189 + 92) ≈ **33%**

   33% 的注意力权重流向 V=0 的 B 位置，等效于注意力输出被缩减约 1/3，连续 32 层后效果灾难性。

---

### 3.4 修复尝试 #1：显式因果掩码 → 成功率 20%（不够好）

**思路**：在 Prune B block 的 `else:` 分支（`causal_mask=None` 时）构建显式位置因果掩码，同时屏蔽 B 列：

```python
_kv_pos = torch.arange(_N_orig, device=_dev)
_cm = ((_kv_pos[None, :] > cache_position[:, None])   # 因果约束
       | torch.isin(_kv_pos, _prune_dev)[None, :])      # 屏蔽 B
causal_mask = _cm.to(hidden_states.dtype) * _min_dt
causal_mask = causal_mask[None, None, :, :]
```

**快速验证（3 trials/task，30 episodes）**：**20%** (6/30)

**分析**：
- B 屏蔽（-inf）消除了 exp(0) 的 softmax 污染 → 从 5.5% 提升到 20%
- 但引入了严格因果掩码（primary patch 位置 p 只能注意到位置 0..p 的 K）
- VLA-Cache 本身使用**全注意力**（`causal_mask=None`，无任何限制）
- 严格因果掩码破坏了模型依赖的全注意力模式（primary patch 无法看 wrist patch 等跨相机信息）
- 结论：屏蔽 B 有效，但引入因果限制有害

---

### 3.5 修复尝试 #2：仅屏蔽 B 列，不加因果约束 → 成功率 66.7%（有改善但不足）

**思路**：只屏蔽 B 位置（-inf），不添加因果约束，与 VLA-Cache 的全注意力模式一致：

```python
_b_row = torch.isin(_kv_pos, _prune_dev)
causal_mask = _b_row[None, :].expand(hidden_states.shape[1], -1)
causal_mask = causal_mask.to(hidden_states.dtype) * _min_dt
causal_mask = causal_mask[None, None, :, :]
```

**快速验证（3 trials/task，30 episodes）**：**66.7%** (20/30)

**分析**：
- B-only 屏蔽正确消除了 softmax 污染 → 大幅提升
- 但显式掩码改变了 softmax 的归一化范围（从 N_orig 个位置归一化变为 N_orig - |B| 个位置）
- 同时，B 位置 zero 的 V 值意味着即便注意力漏过去也没有贡献
- 问题：VLA-Cache 允许注意到所有位置（包括 A 类的 stale KV），这种全注意力格局是模型工作的基础
- 显式掩码（哪怕只屏蔽 B）仍改变了 softmax 归一化结构，与 VLA-Cache 不完全一致

---

### 3.6 修复成功 #3：保留 B 的 stale KV，不清零，不加掩码 → 成功率 93.3%

**核心洞察**：

VLA-Cache 的核心设计是：A 类 token 的 KV 被"复用"（stale），其他 token 可以正常 attend 到 stale A KV，模型仍能工作（95%）。这是因为 A 类 patch 视觉上稳定，stale KV 是 good approximation。

类比 B 类：B 类 patch 虽然视觉上动态（低像素相似度），但文本注意力低。即使 KV 过时，对模型决策的影响也有限。**正确做法**：不清零 B 的 KV，让 B 的 stale KV 像 A 类一样参与 attention，只是不更新 B 的 hidden state（不做 Q 计算）。

**代码变更**：

1. `openvla_utils.py`：**删除**对 B 位置 KV 的清零操作
2. `modeling_llama.py` Prune B block 的 `else:` 分支：不添加任何掩码（`causal_mask` 保持 `None`）

等效于把 B 类 token 当成"彻底不更新的 A 类"——既不计算新的 hidden state，也不清零其缓存，保持与 VLA-Cache 完全一致的全注意力模式。

**快速验证（3 trials/task，30 episodes）**：**93.3%** (28/30)

Per-task 对比：

| Task | Baseline | VLA-Cache | Cache+Prune(fix) |
|------|---------|-----------|-----------------|
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 0.95 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 |
| 5 | 0.75 | 0.75 | 0.67 |
| 6 | 0.95 | 1.00 | 1.00 |
| 7 | 1.00 | 1.00 | 1.00 |
| 8 | 0.95 | 0.90 | 1.00 |
| 9 | 1.00 | 1.00 | 1.00 |
| 10 | 0.90 | 0.90 | 0.67 |

Task 5 和 Task 10 略低，但 3 trials 的方差很大，需要 20-trial 数据确认。

---

## 四、调试经验与关键知识点

### 4.1 关键知识点汇总

**知识点 1：`_update_causal_mask` 中的 `output_attentions` 硬编码覆盖**

```python
# modeling_llama.py line 1174
output_attentions = False  # ← 强制覆盖！caller 的 output_attentions=True 被忽略
```
即使 `_regression_or_discrete_prediction` 传入 `output_attentions=True`，`_update_causal_mask` 内部仍按 `False` 处理，始终走 SDPA 快捷路径。

**知识点 2：VLA-Cache 导致 causal_mask 始终为 None**

VLA-Cache 强制 `past_seen_tokens=0` → `key_value_length = query_length` → `_ignore_causal_mask_sdpa` 返回 True → `causal_mask=None`。

结果：LlamaAttention（eager，因 output_attentions=True）在 VLA-Cache 下**始终使用全注意力**，即所有 token 可以 attend 到所有 KV 位置（无因果限制）。

**知识点 3：Zero KV 会污染 softmax**

对缓存中的 KV 清零（`K=0, V=0`）不等于"屏蔽"该位置。`Q·0=0` 给出 logit=0，在 softmax 中产生 `exp(0)=1`。当正常位置的 logit 普遍为负时，零化位置会主导 softmax 分母，导致 attention 输出接近零。

**正确"屏蔽"方式**：用 `-inf` 掩码（而非清零 KV）。或者如本项目的正确做法：**根本不清零，让 stale KV 正常参与**。

**知识点 4：Prune B 的正确语义**

"Prune B"的含义应该是：**不在当前步更新 B token 的 hidden state（省去 Q 计算的计算量）**，而非"删除 B 的全部贡献"。B 的历史 KV 仍然有效参与 attention，只是不再更新。这与 VLA-Cache 对 A 类 token 的处理是完全对称的。

### 4.2 调试方法总结

1. **控制变量**：快速 3-trial 测试（30 episodes）验证每次修改，避免浪费 3-4 小时跑完整 200 episodes
2. **分层排查**：先确认 baseline/VLA-Cache 正常（200 ep 可靠数据），再调试 Prune 问题
3. **理论分析先行**：通过代码静态分析（trace call stack）找到可能的根因，再用实验验证
4. **数值推理**：用 exp(0)=1 vs exp(负值)<<1 的数量级比较快速定位 softmax 污染

---

## 五、当前状态与下一步

### 5.1 已完成实验结果

| 条件 | Episodes | 成功率 | 状态 |
|------|---------|--------|------|
| Baseline | 200 | **95.5%** | ✓ 最终结果 |
| VLA-Cache | 200 | **95.0%** | ✓ 最终结果 |
| VLA-Cache+Prune (v1, zero+无掩码) | 200 | 5.5% | ✗ 失败（已废弃） |
| VLA-Cache+Prune (v2, zero+因果掩码) | 30 | 20% | ✗ 不足（已废弃） |
| VLA-Cache+Prune (v2b, zero+B屏蔽) | 30 | 66.7% | ✗ 不足（已废弃） |
| **VLA-Cache+Prune (v3, stale KV)** | **30** | **93.3%** | ✓ 待 20-trial 确认 |

### 5.2 待完成

1. **运行 20-trial 正式评估**（Cache+Prune v3，libero_spatial）：
   ```bash
   cd /home/jzzz/vla-ws/vla-cache/src/openvla-oft/
   PYTHONPATH=/home/jzzz/vla-ws/LIBERO CUDA_VISIBLE_DEVICES=0 \
   python experiments/robot/libero/run_libero_eval.py \
       --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
       --task_suite_name libero_spatial --num_trials_per_task 20 \
       --load_in_4bit True --use_vla_cache True --use_prune True
   ```

2. **推理速度分析**：记录 3 组条件的平均推理时间和 TFLOPs，量化计算节省

3. **扩展到 libero_10**：在更难的套件上验证泛化性

4. **更新 MEMORY.md**：记录 Phase 1.2 最终结果

---

## 六、Prune B 最终正确实现方案

### 6.1 核心设计原则

**Prune B = 扩展版 VLA-Cache A-class 复用**

- A 类（原 VLA-Cache）：视觉静态 + 低注意力 → KV 复用，H 在特定层更新
- B 类（Prune B 扩展）：视觉动态 + 低注意力 → KV 永久复用（step-0 值），H 从不更新

两者共同构成"不计算 Q 的 token 集合"，差别仅在 KV 的新鲜度。

### 6.2 最终代码逻辑

**`openvla_utils.py`**：
- 调用 `task_relevant_selection(..., return_prune=True)` 获取 B 类位置
- 设置 `vla.language_model.config.prune_patches = B_positions`
- **不**对 prompt_cache 中的 B 位置执行 zeroing

**`modeling_llama.py`**（Prune B block，decoder loop 之前）：
```python
if prune_patches is not None and inputs_embeds.size(1) != 1:
    _keep = ~torch.isin(cache_position, _prune_dev)
    hidden_states  = hidden_states[:, _keep, :]   # 物理删除 B（节省计算）
    cache_position = cache_position[_keep]         # 位置索引同步更新
    position_ids   = cache_position.unsqueeze(0)   # RoPE 位置同步更新
    pruned_seq_len = hidden_states.shape[1]
    # causal_mask 不作修改（保持 None 或现有值）
    # B 的 stale KV 在 cache 中保留，其他 token 正常 attend
```

**VLA-Cache 每层剪枝循环**（保持不变）：
- 从 `hidden_states` 中移除 A 类（选定的 reusable_patches）
- `causal_mask` 若非 None 则同步去掉对应行（当 `_update_causal_mask` 非 SDPA shortcut 时有效）
