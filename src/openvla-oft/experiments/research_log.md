# VLA-Cache + Token/Layer Pruning 研究日志
**项目**: OpenVLA-OFT 推理加速（Training-Free）
**Benchmark**: LIBERO-Spatial（libero_spatial，10任务×20-trial = 200 episodes/条件）
**更新日期**: 2026-03-15

---

## 一、研究背景与动机

**目标**：在保持 SR ≥ 93% 的前提下，降低 OpenVLA-OFT 的推理延迟。
**基准**：
- Baseline 单步耗时 ~551ms（LLM 7B参数，INT4量化）
- VLA-Cache（原方法）已实现 KV 跨帧复用，但总延迟仍 ~551ms（Python/IO 开销抵消 LLM 节省）

**探索的两个正交冗余维度**：
| 维度 | 冗余类型 | 方法 | 论文来源 |
|------|---------|------|---------|
| 时间冗余 | 相邻帧视觉变化小 → KV 可复用 | VLA-Cache（KV 跨帧缓存） | VLA-Cache [22] |
| 空间冗余 | 背景/无关区域 token 可删 | Token Pruning（三集合并集） | SpecPrune-VLA 思想 |
| 深度冗余 | LLM 相邻层表示高度相似 → 层可跳过 | Layer Skip | EfficientVLA [本次新引入] |

---

## 二、完整实验结果对照表

### 2.1 基础对照组

| 实验ID | 方法 | Token筛选机制 | Primary保留 | Wrist保留 | 总Step耗时 | LLM计算耗时 | Trials | Episodes | SR |
|--------|------|-------------|------------|----------|-----------|------------|--------|----------|-----|
| **E0** | Baseline（无优化） | 全量 512 token | 256/256 | 256/256 | ~551ms | ~551ms | 20 | 200 | **95.5%** |
| **E1** | VLA-Cache only | ABCD分类：A类(~42%×512≈215个)KV复用，不删token | 256 | 256 | ~551ms总 | 减少（未单测） | 20 | 200 | **95.0%** |

### 2.2 B类剪枝路线（已废弃）

> **筛选机制**：`task_relevant_selection()`，ABCD分类：pixel-sim(threshold=0.996) × text-attn分位。
> B类 ≈ 静态 + 高文本注意力，占 ~18.3%（≈47个/相机）。被剪枝后其 KV 在不同版本中被不同处理。

| 实验ID | 版本 | B类KV处理 | token数 | 总耗时 | Trials | Episodes | SR | 失败原因 |
|--------|------|----------|---------|--------|--------|----------|-----|---------|
| — | Cache+Prune v1 | 置零（zero KV） | 512-94=418 | — | 20 | 200 | **5.5%** ✗ | Q·K_B=0→exp(0)=1，B主导softmax |
| — | Cache+Prune v2 | 置零+causal mask | 418 | — | 3 | 30 | **20%** ✗ | mask未完全解决 |
| — | Cache+Prune v2b | B类整行block | 418 | — | 3 | 30 | **66.7%** ✗ | 部分缓解但不足 |
| — | **Cache+Prune v3** | Stale KV保留（不清零，不mask）| 418 | — | 3 | 30 | **93.3%** ⚠️ | 3-trial不可靠，待20-trial |
| — | Cascade v1 | Stale KV + step-0冻结attn做B分类 | 418 | — | 3 | 30 | **76.7%** ✗ | B分类冻结→step-0 attn已过时 |
| — | **Cascade v2（3-trial）** | Stale KV + per-step B分类（token_attention_merge） | 418 | ~584ms | 3 | 30 | **83.3%** | — |
| — | **Cascade v2（20-trial）** | 同上 | 418 | ~561ms（LLM ~318ms） | 20 | 200 | **79.5%** ✓正式 | 详见2.3 |

**Cascade v2 per-task 成绩（20-trial）**：
| 任务 | Baseline | Cache | Cascade v2 | 掉点 |
|------|----------|-------|-----------|------|
| T1 between plate & ramekin | 100% | 100% | 90% | -10% |
| T2 next to ramekin | 100% | 95% | **55%** | **-45%** |
| T3 table center | 100% | 100% | 100% | 0% ✓ |
| T4 on cookie box | 100% | 100% | 100% | 0% ✓ |
| T5 top drawer cabinet | 75% | 75% | **50%** | **-25%** |
| T6 on ramekin | 95% | 100% | 80% | -15% |
| T7 next to cookie box | 100% | 100% | 100% | 0% ✓ |
| T8 on stove | 95% | 90% | 95% | 0% ✓ |
| T9 next to plate | 100% | 100% | **65%** | **-35%** |
| T10 on wooden cabinet | 90% | 90% | **60%** | **-30%** |

**根本原因**：B类 ≈ 18.3% token，单次加速明显但每步 B 集合随 compact-attn 动态更新，导致 N_compact 变化，A 类 KV 复用不稳定（已禁用）→ 实际无 KV 复用加速，但仍有剪枝误伤。

### 2.3 三集合并集路线（Cascade v3 / E2_fixed）

> **筛选机制**：`compute_preprune_mask()`，保留集 = V_global ∪ V_dynamic ∪ V_local
> - V_global：上一步 attention top-60（跨步时序保护）
> - V_dynamic：pixel-sim < 0.99 的patch（当前帧变化区域）
> - V_local：step-0 全序列 attention top-80（frozen，消除compact bias）
> - 剪除集 = {0..255} − 保留集

| 实验ID | 版本 | 相机策略 | V_local来源 | Primary保留 | Wrist保留 | 总耗时 | Trials | Episodes | SR | 关键问题 |
|--------|------|---------|-----------|------------|----------|--------|--------|----------|-----|---------|
| **E2** | v3 首跑 | Primary+Wrist均剪 | per-step compact attn (有bias) | ~110/256 | ~110/256 | — | 3 | 30 | **76.7%** | prev_images bug：V_dynamic全空；compact bias累积 |
| **E2'** | v3 K_local=130 | Primary+Wrist均剪 | per-step compact attn | ~140/256 | ~140/256 | — | 3 | 30 | **66.7%** | 更差，排除K_local是问题根源 |
| **E2_fixed** | v3 fixed（当前最优）| 仅Primary剪，Wrist全量 | **step-0 frozen全序列attn** | ~130/256(约剪49%) | 256/256 | ~568ms | 3 | 30 | **93.3%** ✓ | T4(drawer)仍33%，其余100% |

**E2_fixed 设计要点**：
1. **Wrist 不剪枝**：wrist视野随机械臂变化，V_dynamic收缩时frozen V_local仍指向旧视野 → "死区"问题。wrist只有256 token，冗余度低，收益不抵风险。
2. **frozen V_local**：用 step-0 完整（未compact）attention scores，存入 `last_caches['init_attn_scores_primary']`，episode 内不更新。解决雪球效应（compact attn逐步偏离全序列）。
3. **V_global 每步更新**：仍从 per-step compact attn 取 top-60，保持时序保护有效。
4. **无 VLA-Cache KV 复用**：E2_fixed 阶段未叠加 KV 复用，故总耗时 568ms ≈ baseline 551ms，无实质加速。

**E2_fixed 内部 Token 数学**：
```
保留 token 数 ≈ |V_global ∪ V_dynamic ∪ V_local|
  V_global = 60
  V_dynamic ≈ 30-60（取决于当前帧变化量）
  V_local = 80
  三集合有重叠 → 实际保留 ~110-140/256（约43-55%）
  加上 wrist 256 → LLM 输入 ~366-396 tokens（对比 baseline 512）
```

---

## 三、关键架构发现（对实现有影响）

| 发现 | 影响 |
|------|------|
| `_update_causal_mask` 第1174行硬编码 `output_attentions=False` → SDPA shortcut always fires | 实际用 eager LlamaAttention（因 `output_attentions=True` in modeling_prismatic.py L912） |
| VLA-Cache 强制 `past_seen_tokens=0` → `key_value_length==query_length` → `causal_mask=None` | 所有推理使用全注意力（无mask） |
| B类置零 KV → `exp(0)=1` 主导 softmax（当其他 logit 为负时） | v1/v2 失败的直接原因 |
| compact attention 偏差（雪球效应）| V_local 必须从完整序列 attn 获取，不能从 compact attn 更新 |
| prev_images bug | eval 脚本的 prev_img 是当前帧而非上一查询帧，导致 V_dynamic 全空 |

---

## 四、EfficientVLA 论文解读（arXiv:2506.10100v1）

### 4.1 论文概述

**针对对象**：CogACT（diffusion-based VLA，Llama2-7B + DiT action head）
**三个组件**：
1. **Layer Pruning**：基于层间 cosine similarity 的重要性评分，非连续剪枝最不重要的 n 层
2. **Visual Token Pruning**：V_pruned = V_key ∪ V_task ∪ V_div（task-relevance + diversity）
3. **Action Cache**：DiT 去噪步骤中跨 timestep 缓存 attention/MLP 中间特征（**仅适用于 diffusion action head，不适用于 OpenVLA-OFT**）

**总体结果（CogACT，SIMPLER benchmark）**：
- EfficientVLA(L=22, T=56)：SR 93.3%，推理 0.1213s，**1.93× speedup**，FLOPs 降至 28.9%
- CogACT baseline：SR 91.3%，0.2342s，1.00×

### 4.2 Layer Pruning 详解

**重要性评分公式**：
$$I^{(\ell)} = 1 - \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \left( \frac{1}{L} \sum_{j=1}^{L} \frac{\mathbf{x}_{i,j}^{(\ell)} \cdot \mathbf{x}_{i,j}^{(\ell+1)}}{\|\mathbf{x}_{i,j}^{(\ell)}\|_2 \|\mathbf{x}_{i,j}^{(\ell+1)}\|_2} \right)$$

- 输入与输出 cosine similarity 越高 → 该层变换越小 → 重要性 $I^{(\ell)}$ 越低 → 越应被跳过
- 需要校准数据集 $\mathcal{D}$（代表性训练样本）
- 排序后取最低 n 个层跳过（**非连续**，按重要性排序而非按层号顺序）

**消融实验（Table 5，CogACT pick-coke-can任务）**：
| 实验 | Layer剪枝 | MLP稀疏化 | Token剪枝 | Action Cache | SR | 推理时间 | Speedup |
|------|--------|---------|---------|------------|-----|---------|---------|
| Ex0（baseline） | ✗ | ✗ | ✗ | ✗ | 91.3% | 0.2342s | 1.00× |
| **Ex3（Layer only）** | **✓** | ✗ | ✗ | ✗ | **85.7%** | **0.1604s** | **1.46×** |
| Ex6（Layer+Token） | ✓ | ✗ | ✓ | ✗ | **95.3%** | 0.1592s | **1.47×** |
| Ex7（全部） | ✓ | ✓ | ✓ | ✓ | 93.3% | 0.1213s | 1.93× |

**关键结论**：
- Layer pruning 单独 → 1.46× speedup，但 SR 从 91.3% 降至 85.7%（-5.6%）
- Layer pruning + Token pruning（Ex6）→ SR 反而升至 95.3%（Token pruning 过滤噪声token，弥补层剪枝的精度损失）
- **Layer pruning 是加速的主要来源**（token pruning 受限于 memory bound，加速效果有限）

**为什么 Token Pruning 加速有限**（见 Figure 1(a) 和论文 Section 4.3）：
- 当 token 数较少时，系统从 computation-bound 切换到 memory-bound
- memory-bound 状态下继续削减 token 数加速效果边际递减
- Token pruning 最大 speedup ≈ 1.25×（Ex1），而 layer pruning 达 1.46×

### 4.3 与我们系统的对应关系

| EfficientVLA 组件 | 是否适用 OpenVLA-OFT | 我们对应方案 |
|-----------------|-------------------|------------|
| Layer Pruning | ✅ 完全适用（同为 Llama2-7B 32层） | **新引入：Layer Skip** |
| Visual Token Pruning（V_key∪V_task∪V_div） | ✅ 适用（概念相近） | 已有：三集合并集（V_global∪V_dynamic∪V_local），差异在于我们加了时序保护V_global和frozen V_local |
| Action Cache（DiT去噪步骤缓存） | ❌ 不适用 | OpenVLA-OFT 用 L1回归头+parallel decoding，无扩散去噪过程 |
| MLP 稀疏化（PruneNet，25% sparsity） | ⚠️ 需校准数据，接近 training-aware | 暂不实现 |

**我们的组合的独特性**：
- VLA-Cache 利用**时间冗余**（跨帧 KV 复用）
- Token Pruning（三集合）利用**空间冗余**（删背景 token）+ 时序保护（V_global）
- Layer Skip 利用**深度冗余**（跳过高 cosine-sim 层）
- 三者组合在 OpenVLA-OFT + LIBERO 上的效果**从未在任何论文中报告过**

---

## 五、Layer Skip 实现计划（Phase 1.3）

### 5.1 核心思路

在 `modeling_llama.py` 的 decoder loop（L1051）中，对指定层直接 `continue`，hidden_state 原封不动传入下一层（等效于 residual 短路）：

```python
# 实现位置：modeling_llama.py L1051，for loop 开头
skip_layers = getattr(self.config, 'skip_layers', set())
for layer_idx, decoder_layer in enumerate(self.layers):
    # ─── Layer Skip ───────────────────────────────────────
    if layer_idx in skip_layers:
        continue   # hidden_states 原样传递，不执行 attn+FFN，不更新 KV
    # ─────────────────────────────────────────────────────
    # （原有 VLA-Cache reuse 逻辑、FLOPs 计数等保持不变）
    ...
```

**注意**：`continue` 必须在 FLOPs 累加、`output_hidden_states` 收集、`all_self_attns` 收集之前，避免对跳过层误计。

### 5.2 与 VLA-Cache / E2_fixed 的兼容性

| 潜在问题 | 分析 | 结论 |
|---------|------|------|
| 跳过层的 KV 不写入 cache | E2_fixed 每步 fresh `DynamicCache()`，跳过层的 slot 为空，不影响其他层 | ✅ 安全 |
| VLA-Cache A 类 KV 复用（`reusable_patches`） | 复用发生在 `pruning_loc` 层，与 skip_layers 独立配置，不冲突 | ✅ 安全 |
| AttentionHookCapture 收集 attn | hook 绑定在每层 self_attn 上，跳过层不触发 hook | ✅ 安全（不需要跳过层的 attn） |
| Cache+Prune v3 stale KV（per-layer slot） | 跳过层的 slot 恒为空；其他层的 stale KV 正常保留 | ✅ 安全 |
| FLOPs 计数准确性 | `continue` 前不执行 `self.all_FLOPs +=`，跳过层不计入 | ✅ 准确 |

**结论：Layer Skip 与所有现有机制完全正交，无任何索引对齐问题。**

### 5.3 跳过哪些层？—— 层选择策略

**背景**：EfficientVLA 使用重要性评分 $I^{(\ell)} = 1 - \cos(h_{in}, h_{out})$ 选层（式1）。我们在 OpenVLA-OFT（Llama2-7B backbone）上实测校准。

### 5.3.1 T14 校准结果（2026-03-15，5 episodes × 28 queries/ep = 140 forward passes）

| 排名 | 层 | I^(ℓ) | 区域 | 备注 |
|------|----|--------|------|------|
| 1  | **3**  | 0.0168 | 早期 | ← 最冗余 |
| 2  | **5**  | 0.0178 | 早期 | |
| 3  | **4**  | 0.0200 | 早期 | |
| 4  | **6**  | 0.0236 | 早期 | |
| 5  | **23** | 0.0324 | 深层 | |
| 6  | **28** | 0.0326 | 深层 | |
| 7  | **25** | 0.0369 | 深层 | |
| 8  | **21** | 0.0370 | 中深 | |
| 9  | **20** | 0.0374 | 中深 | |
| 10 | **27** | 0.0377 | 深层 | |
| 11 | **26** | 0.0381 | 深层 | |
| 12 | 19 | 0.0408 | 中深 | E5a range |
| 13 | 17 | 0.0427 | 中深 | E5a range |
| 14 | 18 | 0.0429 | 中深 | E5a range |
| 17 | 24 | 0.0451 | 深层 | E5b range |
| 21 | 16 | 0.0488 | 中深 | E5a range |
| … | … | … | … | |
| 29 | 15 | 0.0622 | 中层 | |
| 30 | 30 | 0.0688 | 末层 | |
| 31 | **0**  | 0.1213 | 首层 | ← 第2重要 |
| 32 | **31** | 0.1394 | 末层 | ← 最重要 |

**关键发现**：
- 最重要层：31（末层，映射到 action），0（首层，嵌入投影），30，15，22
- 最冗余层：{3,4,5,6}（早期层，残差主导，变化极小）→ 真正的数据驱动最优选择
- E5a 跳过的 {16-19}：排名 21/14/13/12（中等冗余，I≈0.041-0.049），已被证明损伤严重
- E5b 跳过的 {24-27}：排名 17/7/11/10（较冗余，I≈0.037-0.045），仍损伤 25%
- **早期层 {3-6} 的 I 值是 {16-19} 的 40%**，有望损伤更小

**备选层组（按重要性数据驱动）**：

| 组名 | Skip 层（0-indexed） | I范围 | 选择依据 |
|------|-------------------|--------|---------|
| ~~**Mid-4**~~ | ~~{16,17,18,19}~~ | 0.041-0.049 | ✗ 已测，43.3% SR |
| ~~**Deep-4**~~ | ~~{24,25,26,27}~~ | 0.037-0.045 | ✗ 已测，70.0% SR |
| **Calib-Early-4**（首选） | {3,4,5,6} | 0.017-0.024 | ✓ 最低 I，最冗余 |
| **Calib-Deep-4** | {23,25,26,27} | 0.032-0.039 | ✓ 深层中最冗余 |
| **Calib-Spread-4** | {3,5,23,28} | 0.017-0.033 | 不同深度最冗余各选1 |

**实验优先级**：E5f（Calib-Early-4）→ E5g（Calib-Deep-4）→ 决定最终 layer skip 组

### 5.4 实现任务清单

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| **T11** | `modeling_llama.py` decoder loop L1051 新增 layer skip | modeling_llama.py | ✅ done |
| **T12** | `openvla_utils.py` L812 追加 skip_layers（含 str→set 转换） | openvla_utils.py | ✅ done |
| **T13** | `run_libero_eval.py` GenerateConfig 新增 `skip_layers: str = ""` | run_libero_eval.py | ✅ done |
| **T14** | `compute_layer_importance.py`（5-episode 校准，输出每层 I^(ℓ)） | experiments/robot/ | ✅ done (2026-03-15) |
| **T15** | 最小测试：VLA-Cache + Skip{16-19}，1-trial | — | ✅ done (2026-03-15) |

**T15 执行过程中发现并修复了 3 个 Bug**：

| Bug # | 报错信息 | 根因 | 修复位置 | 修复方式 |
|-------|---------|------|---------|---------|
| Bug 1 | `'list' object has no attribute 'to'` | `_kvcache_to_device()` 遍历 `cache.key_cache`，跳过层的 slot 是 `[]`（Python 列表），不是 Tensor | `openvla_utils.py` L727 | `isinstance(t, torch.Tensor)` 判断，非 Tensor 直接透传 |
| Bug 2 | `'NoneType' object has no attribute 'mean'` | `AttentionHookCapture` 构建 32 项元组；跳过层 hook 不触发 → 对应位置为 `None`；`get_layer_mask_schedule()` 调用 `None.mean()` 崩溃 | `vla_cache_utils.py` L11 | 重写函数：跳过 `None` 项，记录有效层索引，最终填回 full-length 向量（跳过层填 0.0） |
| Bug 3 | `'list' object has no attribute 'size'` | `LlamaModel.forward()` L987 始终调用 `DynamicCache.from_legacy_cache(past_key_values)`；该函数迭代所有 32 层，对跳过层调用 `cache.update([], [], layer_idx)`，在 `update()` 内 `key_states.size(-2)` 崩溃 | `cache_utils.py` L507-509 | `from_legacy_cache` 循环内增加 `if not isinstance(key_states, torch.Tensor): continue` |

**T15 结果**（VLA-Cache + Skip{16-19}，1-trial×10任务）：
- 整体 SR: **50%**（10/10任务中5个成功，符合1-trial高方差预期）
- 步均耗时: ~562ms（与baseline ~551ms相当）
- FLOPs: 3.512 TFLOPs（vs baseline ~4.01 TFLOPs = 87.5%，符合28/32层预期）
- **无 crash，T15 PASS → 继续 E5a**

### 5.5 各任务精确改动（逐行）

**T11 — `modeling_llama.py` L1051**（1处，2行新增）：
```python
# 在 for layer_idx, decoder_layer in enumerate(self.layers): 的第一行后插入
for layer_idx, decoder_layer in enumerate(self.layers):
    # ─── Layer Skip ───────────────────────────────────────
    if layer_idx in getattr(self.config, 'skip_layers', set()):
        continue
    # ──────────────────────────────────────────────────────
    # （L1053 起的原有逻辑不变）
```
`continue` 在 FLOPs计数(L1081)、output_hidden_states(L1088)、all_self_attns(L1118) 之前，三处自动跳过，无需额外处理。

**T12 — `openvla_utils.py` L812 后**（1行新增）：
```python
# 原有：
vla.language_model.config.proportion_attn_var = None   # L810
vla.language_model.config.prune_patches = None          # L811
vla.language_model.config.prepruning_B_indices = None   # L812
# 新增一行：
vla.language_model.config.skip_layers = getattr(cfg, 'skip_layers', set())
```

**T13 — `run_libero_eval.py` 两处**（3行新增）：
```python
# 位置1：GenerateConfig dataclass L91后，新增字段：
preprune_k_local: int = 80
skip_layers: str = ""       # 逗号分隔层号，如 "16,17,18,19"；空串=不跳过

# 位置2：main() 中 cfg = tyro.cli(GenerateConfig) 之后，解析字符串：
if cfg.skip_layers:
    cfg.skip_layers = set(int(x) for x in cfg.skip_layers.split(","))
else:
    cfg.skip_layers = set()
```

**T12 完整代码（openvla_utils.py L812 后）**：
```python
# 注意：T12 最终版本增加了 str→set 转换（2026-03-15 bugfix），避免 `int in str` 错误
_skip = getattr(cfg, 'skip_layers', set())
if isinstance(_skip, str):
    _skip = set(int(x) for x in _skip.split(',') if x.strip()) if _skip else set()
vla.language_model.config.skip_layers = _skip
```

**T14 校准脚本** `experiments/robot/compute_layer_importance.py`（约380行，2026-03-15新增）：
- 注册 `register_forward_hook` 在每个 `LlamaDecoderLayer` 上
- Hook 捕获 `input[0]`（层输入 hidden_states）和 `output[0]`（层输出 hidden_states）
- 对每次 forward 调用，计算 `F.cosine_similarity(h_in, h_out, dim=-1).mean()` 并累积
- 最终输出 `I^(ℓ) = 1 - mean_cos_sim`
- 输出文件：`experiments/analysis/layer_importance/layer_importance.json`, `.txt`, `.png`
- 运行命令（5 episodes × 40 queries/ep，实际跑了 5×28=140 forward passes）：
```bash
cd /home/jzzz/vla-ws/vla-cache/src/openvla-oft/
PYTHONPATH=/home/jzzz/vla-ws/LIBERO CUDA_VISIBLE_DEVICES=0 \
python experiments/robot/compute_layer_importance.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial \
    --num_calibration_episodes 5 --max_queries_per_episode 40 --load_in_4bit True
```

**T15 遇到的3个Bug及修复（2026-03-15）**：
1. **`openvla_utils.py _kvcache_to_device`（L727）**：`[]` 空列表（跳过层 KV slot）调用 `.to(device)` 失败
   - 修复：`[t.to(device) if isinstance(t, torch.Tensor) else t for t in cache.key_cache]`
2. **`vla_cache_utils.py get_layer_mask_schedule`**：`AttentionHookCapture` 对跳过层返回 `None`，`None.mean()` 崩溃
   - 修复：重写函数，跳过 `None` 项，建立 `valid_indices` 列表，结果填回 full-length tensor（跳过层=0.0）
3. **`transformers/cache_utils.py from_legacy_cache`（L507）**：迭代所有32层时对跳过层调用 `cache.update([], [], idx)`，`[].size(-2)` 崩溃
   - 修复：`if not isinstance(key_states, torch.Tensor): continue`（跳过非 Tensor 项）

**合计最终改动**：
- `modeling_llama.py` +4行（含注释）：layer skip `continue`
- `openvla_utils.py` +4行：skip_layers 赋值 + str→set 转换；+2行：_kvcache_to_device isinstance 守卫
- `run_libero_eval.py` +3行：GenerateConfig 字段 + 解析
- `vla_cache_utils.py` 重写 `get_layer_mask_schedule`：+35行
- `cache_utils.py`（transformers 库）+1行：from_legacy_cache 跳过非 Tensor
- `compute_layer_importance.py` 新增 ~380行

---

## 六、Phase 1.3 实验矩阵（含运行命令）

### 6.1 执行路线（决策树）

```
Step 1: 实现 T11-T13
Step 2: 最小测试 T15（1-trial，验证不 crash）
         │
Step 3: E5a（VLA-Cache + Skip{16-19}，3-trial）
         ├── SR ≥ 93%? → 继续 E6a（Cache+Prune v3 + Skip{16-19}，3-trial）
         │                    ├── SR ≥ 91%? → 20-trial 正式 eval（E7 最终）
         │                    └── SR < 91%? → 尝试其他层组 or 减少 skip 数
         └── SR < 91%? → 尝试 E5c（Skip{24-27}，3-trial）或 Calibrated-4

Step 4: 并行跑消融对照组（用于论文佐证）
         E5b（Skip{24-27}），E5d（Spread-4），E5e（Mid-6）
```

### 6.2 全部待跑实验（3-trial，按优先级排序）

| 实验ID | 配置 | Skip 层 | Token策略 | 优先级 | 预期 SR | 预期耗时 | 状态 |
|--------|------|---------|---------|--------|--------|---------|------|
| **E5a** | VLA-Cache + Layer Skip | {16,17,18,19} Mid-4 | A类KV复用，无删token | P0（首跑） | ~88-93%? | 302ms CUDA/553ms总 | ✅ **43.3%** ✗ 层16-19太关键 |
| **E5b** | VLA-Cache + Layer Skip | {24,25,26,27} Deep-4 | 同上 | P1（对照） | ~88-93%? | 300ms CUDA/554ms总 | ✅ **70.0%** ✗ 仍偏低，T2/T10系统性失败 |
| **E5f (3-trial)** | VLA-Cache + Layer Skip | {3,4,5,6} Calib-Early-4 | A类KV复用，无删token | — | — | 303ms CUDA/555ms总 | ✅ **90.0%** (3t噪声，非真实) |
| **E5g** | VLA-Cache + Layer Skip | {23,25,26,27} Calib-Deep-4 | 同上 | — | — | 302ms CUDA/544ms总 | ✅ **70.0%** ✗ 深层仍重要 |
| **E5h** | VLA-Cache + Layer Skip | {3,5,23,28} Calib-Spread-4 | 同上 | — | — | 300ms CUDA/560ms总 | ✅ **90.0%** (3t，与E5f并列) |
| **E5f (20-trial)** ★ | VLA-Cache + Layer Skip | {3,4,5,6} Calib-Early-4 | 同上 | P0 | — | 303ms CUDA/570ms总 | ✅ **88.5%** ✗ 低于93%目标 |
| **E3_20t** ★ | Cache+Prune v3 only（E2_fixed，无layer skip）| — | Primary三集合剪49%+frozen V_local，Wrist全量 | P0 | ≥91% | 309ms CUDA/555ms总 | ✅ **93.0% (186/200)** ✓ 3-trial结论稳定 |
| **E5i** | VLA-Cache + Layer Skip 2层 | {3,4} 最冗余2层 | A类KV复用，无删token | P0 | ≥91% | 327ms CUDA/586ms总 | ✅ **90.0% (27/30)** ⚠️ 3-trial，待20-trial确认 |
| E5c | VLA-Cache + Layer Skip | {8,9,10,11} Early-4 | 同上 | P3（消融对照） | ~75-88%? | — | ⬜ |
| E5d | VLA-Cache + Layer Skip | {8,16,24,28} Spread-4 | 同上 | P3 | ~85-93%? | — | ⬜ |
| E5e | VLA-Cache + Layer Skip | {14,15,16,17,18,19} Mid-6 | 同上 | P3（激进） | ~78-88%? | — | ⬜ |
| **E6a** | Cache+Prune v3 + Layer Skip | {3,4,5,6} | A+B类stale KV + Layer Skip | P0（决策后） | ~88-93%? | ~280-350ms? | ⬜ |
| **E6b** | E2_fixed + Layer Skip | {3,4,5,6} | Primary三集合剪49% + Layer Skip | P0（决策后） | ~88-93%? | ~300-370ms? | ⬜ |
| **E7** | 最优组合 20-trial 正式 eval | TBD | TBD | P0（最终） | ≥91% | ≤420ms | ⬜ |

**实验定义**：
- **E3_20t**：`--use_preprune_v3 True --use_vla_cache False`，20-trial，确认 E2_fixed 3-trial 93.3% 在大样本下是否稳定
- **E5i**：`--use_vla_cache True --skip_layers "3,4"`，3-trial，验证更保守的 2-layer skip（最冗余两层 I≈0.017/0.020）是否维持 ≥91% SR

**E3_20t 逐任务结果（20-trial，2026-03-15）—— 正式确认数据**：
| 任务 | E3_20t SR | E0 Baseline | 备注 |
|------|-----------|------------|------|
| T1 between plate&ramekin | **100%** | 100% | ✓ |
| T2 next to ramekin | **95%** | 100% | -5% |
| T3 table center | **100%** | 100% | ✓ |
| T4 on cookie box | **100%** | 100% | ✓ |
| **T5 top drawer cabinet** | **50%** | 75% | **-25%** 主要失分点 |
| T6 on ramekin | **100%** | 95% | ✓ |
| T7 next to cookie box | **100%** | 100% | ✓ |
| T8 on stove | **95%** | 95% | ✓ |
| T9 next to plate | **100%** | 100% | ✓ |
| T10 on wooden cabinet | **90%** | 90% | ✓ |
| **总计** | **186/200 = 93.0%** | **95.5%** | **-2.5% vs E0** |
- CUDA: 309ms，TFLOPs: 2.947，总耗时: 555ms（ Token Reuse Ratio=0%，无KV复用）
- **结论**：3-trial 93.3% 确实稳定，20-trial 93.0% ✓。主要弱点 T5（drawer 50%），与 E2_fixed 3-trial 一致（T4 drawer 33% → 20-trial 变 T5 drawer 50%，测量稳定）

**E5i 逐任务结果（3-trial，Skip{3,4} 2层，2026-03-15）**：
| 任务 | E5i {3,4} | E5f {3-6} | E1 Cache |
|------|-----------|-----------|---------|
| T1 between plate&ramekin | 67% | 67% | 100% |
| T2 next to ramekin | 100% | 100% | 95% |
| T3 table center | 100% | 100% | 100% |
| T4 on cookie box | 100% | 100% | 100% |
| T5 top drawer cabinet | 67% | 67% | 75% |
| T6 on ramekin | 100% | 100% | 100% |
| T7 next to cookie box | 100% | 100% | 100% |
| T8 on stove | 100% | 100% | 90% |
| T9 next to plate | 100% | 100% | 100% |
| T10 on wooden cabinet | 67% | 67% | 90% |
| **总计** | **90.0% (27/30)** | **90.0% (27/30)** | **95.0%** |
- CUDA: 327ms，TFLOPs: 3.007，总耗时: 586ms
- **关键发现**：2层skip（{3,4}）的3-trial SR与4层skip（{3,4,5,6}）**完全相同（90%）**
  - 节省CUDA：2层=327ms vs 4层=303ms（多节省了24ms），但3-trial无法区分
  - E5f 20-trial真实SR=88.5%；E5i若跑20-trial，预期≈88-90%（接近但未必更好）
  - 失败任务完全相同（T1 67%、T5 67%、T10 67%），表明加速收益来自早期层的残差冗余，但T1/T5/T10的精细操作对早期特征提取仍有轻微依赖

**E5f 逐任务结果（3-trial，2026-03-15）**：
| 任务 | E5f {3-6} | E5b {24-27} | E5a {16-19} | E1 Cache |
|------|-----------|------------|------------|---------|
| T1 between plate&ramekin | 67% | 100% | 0% | 100% |
| T2 next to ramekin | **100%** | 0% | 0% | 95% |
| T3 table center | 100% | 100% | 100% | 100% |
| T4 on cookie box | 100% | 100% | 100% | 100% |
| T5 top drawer cabinet | 67% | 67% | 0% | 75% |
| T6 on ramekin | 100% | 33% | 67% | 100% |
| T7 next to cookie box | 100% | 67% | 100% | 100% |
| T8 on stove | 100% | 100% | 33% | 90% |
| T9 next to plate | **100%** | 100% | 0% | 100% |
| T10 on wooden cabinet | 67% | 33% | 33% | 90% |
| **总计** | **90.0%** | **70.0%** | **43.3%** | **95.0%** |
- CUDA 延迟（3-trial）：~303ms（vs 551ms = 45% LLM 计算减少）；TFLOPs：2.897
- 失败模式（3-trial）：T1（between 精细操作），T5（抽屉把手），T10（木柜位置）

**E5f 逐任务结果（20-trial，2026-03-15）—— 最终正式数据**：
| 任务 | E5f {3-6} 20t | E5f 3t（参考） | E1 Cache 20t | E0 Baseline 20t |
|------|---------------|--------------|-------------|----------------|
| T1 between plate&ramekin | **95%** | 67% | 100% | 100% |
| T2 next to ramekin | **100%** | 100% | 95% | 100% |
| T3 table center | **100%** | 100% | 100% | 100% |
| T4 on cookie box | **100%** | 100% | 100% | 100% |
| T5 top drawer cabinet | **80%** | 67% | 75% | 75% |
| T6 on ramekin | **85%** | 100% | 100% | 95% |
| T7 next to cookie box | **100%** | 100% | 100% | 100% |
| **T8 on stove** | **65%** | 100% | 90% | 95% |
| T9 next to plate | **100%** | 100% | 100% | 100% |
| **T10 on wooden cabinet** | **60%** | 67% | 90% | 90% |
| **总计** | **88.5% (177/200)** | 90.0% | **95.0%** | **95.5%** |
- CUDA: 303ms，TFLOPs: 2.894，总耗时: 570ms（vs baseline 551ms，Python 开销略增）
- **新失分点**（20-trial揭露）：T8（65%，3-trial噪声掩盖）和 T10（60%）是主要拖累
- 3-trial 高估了真实 SR：T6、T8 在3-trial各抽到了好局（100%），20-trial回归均值
- **结论**：VLA-Cache + Layer Skip{3-6} 单独无法达到 ≥93% 目标，需叠加 Token Pruning（EfficientVLA 路线）

**E5g 逐任务结果（3-trial，Skip{23,25,26,27}，2026-03-15）**：
| 任务 | E5g {23,25-27} | E5f {3-6} | E1 Cache |
|------|----------------|-----------|---------|
| T1 between plate&ramekin | 100% | 67% | 100% |
| T2 next to ramekin | 33% | 100% | 95% |
| T3 table center | 100% | 100% | 100% |
| T4 on cookie box | 100% | 100% | 100% |
| T5 top drawer cabinet | 0% | 67% | 75% |
| T6 on ramekin | 67% | 100% | 100% |
| T7 next to cookie box | 100% | 100% | 100% |
| T8 on stove | 67% | 100% | 90% |
| T9 next to plate | 100% | 100% | 100% |
| T10 on wooden cabinet | 33% | 67% | 90% |
| **总计** | **70.0%** | **90.0%** | **95.0%** |
- CUDA: 302ms，TFLOPs: 2.732，总耗时: 544ms
- T2/T5/T10 失败，与 E5b（相近层组）行为一致，证明深层 {20-28} 对精细/空间任务有重要贡献

**E5h 逐任务结果（3-trial，Skip{3,5,23,28}，2026-03-15）**：
| 任务 | E5h {3,5,23,28} | E5f {3-6} | E1 Cache |
|------|-----------------|-----------|---------|
| T1 between plate&ramekin | 100% | 67% | 100% |
| T2 next to ramekin | 100% | 100% | 95% |
| T3 table center | 100% | 100% | 100% |
| T4 on cookie box | 100% | 100% | 100% |
| T5 top drawer cabinet | 33% | 67% | 75% |
| T6 on ramekin | 100% | 100% | 100% |
| T7 next to cookie box | 100% | 100% | 100% |
| T8 on stove | 100% | 100% | 90% |
| T9 next to plate | 100% | 100% | 100% |
| T10 on wooden cabinet | 67% | 67% | 90% |
| **总计** | **90.0%** | **90.0%** | **95.0%** |
- CUDA: 300ms，TFLOPs: 2.796，总耗时: 560ms
- 与 E5f 完全并列（90%）；T5/T10 仍是共同弱点；T1 比 E5f 好（100% vs 67%）
- **跨组对比**：{3,5,23,28} 避免了跳连续层 {3,4,5,6}，T1 表现更好，但 T5 下降更多

**Layer Skip 系列 3-trial 汇总对比**：
| 层组 | Skip层 | 最冗余 I（平均） | SR | CUDA | T1 | T2 | T5 | T10 |
|------|--------|---------------|-----|------|-----|-----|-----|-----|
| Mid-4 | {16-19} | 0.044 | 43.3% | 302ms | 0% | 0% | 0% | 33% |
| Deep-4 | {24-27} | 0.039 | 70.0% | 300ms | 100% | 0% | 67% | 33% |
| Calib-Deep | {23,25-27} | 0.036 | 70.0% | 302ms | 100% | 33% | 0% | 33% |
| **Calib-Early** | **{3-6}** | **0.020** | **90.0%** | **303ms** | 67% | 100% | 67% | 67% |
| **Calib-Spread** | **{3,5,23,28}** | **0.024** | **90.0%** | **300ms** | 100% | 100% | 33% | 67% |
- 规律：I 越低的层组 SR 越高（正如 EfficientVLA 方法论预期）
- E5f 与 E5h 并列 90%，但 failure task 不同：E5f 在 T1 弱，E5h 在 T5 弱

### 6.3 各实验运行命令模板

**通用路径与前缀**：
```bash
cd /home/jzzz/vla-ws/vla-cache/src/openvla-oft/
PYTHONPATH=/home/jzzz/vla-ws/LIBERO CUDA_VISIBLE_DEVICES=0 \
/home/jzzz/miniconda3/envs/openvla-oft/bin/python \
    experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial \
    --load_in_4bit True
```

**E5a（VLA-Cache + Skip Mid-4，3-trial）**：
```bash
# ... 通用前缀 ...
    --use_vla_cache True \
    --skip_layers "16,17,18,19" \
    --num_trials_per_task 3 \
    --run_id_note "E5a_vlacache_skip1619"
```

**E5b（VLA-Cache + Skip Deep-4，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "24,25,26,27" \
    --num_trials_per_task 3 \
    --run_id_note "E5b_vlacache_skip2427"
```

**E5c（VLA-Cache + Skip Early-4，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "8,9,10,11" \
    --num_trials_per_task 3 \
    --run_id_note "E5c_vlacache_skip0811"
```

**E5d（VLA-Cache + Spread-4，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "8,16,24,28" \
    --num_trials_per_task 3 \
    --run_id_note "E5d_vlacache_spread4"
```

**E5e（VLA-Cache + Skip Mid-6，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "14,15,16,17,18,19" \
    --num_trials_per_task 3 \
    --run_id_note "E5e_vlacache_skip1419"
```

**E5f（VLA-Cache + Skip Calib-Early-4，3-trial / 20-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "3,4,5,6" \
    --num_trials_per_task 3 \   # 或 20
    --run_id_note "E5f_cache_skip3-6"
```

**E5g（VLA-Cache + Skip Calib-Deep-4，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "23,25,26,27" \
    --num_trials_per_task 3 \
    --run_id_note "E5g_cache_skip23252627"
```

**E5h（VLA-Cache + Skip Calib-Spread-4，3-trial）**：
```bash
    --use_vla_cache True \
    --skip_layers "3,5,23,28" \
    --num_trials_per_task 3 \
    --run_id_note "E5h_cache_skip_spread3_5_23_28"
```

**E6a（Cache+Prune v3 + Skip，3-trial）**：
```bash
    --use_vla_cache True \
    --use_preprune True \          # Cache+Prune v3 = stale KV path
    --skip_layers "X,X,X,X" \     # 用 E5 最优层填入
    --num_trials_per_task 3 \
    --run_id_note "E6a_v3_skip_best"
```

**E6b（E2_fixed + Skip，3-trial）**：
```bash
    --use_preprune_v3 True \       # E2_fixed = 三集合 + frozen V_local
    --use_vla_cache False \
    --skip_layers "X,X,X,X" \     # 用 E5 最优层填入
    --num_trials_per_task 3 \
    --run_id_note "E6b_e2fixed_skip_best"
```

**E7（20-trial 正式 eval，最终配置）**：
```bash
    # [最优 token 策略 flags] \
    --skip_layers "X,X,X,X" \
    --num_trials_per_task 20 \
    --run_id_note "E7_final_20trial"
```

### 6.4 论文消融表结构（目标）

最终论文中的实验表应覆盖以下维度：

| 实验 | VLA-Cache (时间冗余) | Token Prune (空间冗余) | Layer Skip (深度冗余) | Skip层数 | Skip层组 | SR | Speedup | 总耗时 |
|------|--------------------|--------------------|---------------------|---------|---------|-----|---------|--------|
| E0 Baseline | ✗ | ✗ | ✗ | 0 | — | 95.5% | 1.00× | ~551ms |
| E1 VLA-Cache | ✓ | ✗ | ✗ | 0 | — | 95.0% | — | ~551ms |
| E2_fixed(3t) | ✗ | ✓(Primary) | ✗ | 0 | — | 93.3% (3t) | — | ~568ms |
| **E3_20t** ★ | ✗ | ✓(Primary) | ✗ | 0 | — | **93.0%** ✓ | — | 555ms |
| **E5i** | ✓ | ✗ | ✓ | 2 | Top-2 {3,4} | **90.0%** (3t) | ~40% CUDA | 586ms |
| E5a | ✓ | ✗ | ✓ | 4 | Mid-4 {16-19} | **43.3%** ✗ | ~45% CUDA | 553ms |
| E5b | ✓ | ✗ | ✓ | 4 | Deep-4 {24-27} | **70.0%** ✗ | ~45% CUDA | 554ms |
| E5f(3t) | ✓ | ✗ | ✓ | 4 | Calib-Early {3-6} | **90.0%** ⚠️ | ~45% CUDA | 555ms |
| E5g | ✓ | ✗ | ✓ | 4 | Calib-Deep {23,25-27} | **70.0%** ✗ | ~45% CUDA | 544ms |
| E5h | ✓ | ✗ | ✓ | 4 | Calib-Spread {3,5,23,28} | **90.0%** ⚠️ | ~45% CUDA | 560ms |
| **E5f(20t)** ★ | ✓ | ✗ | ✓ | 4 | Calib-Early {3-6} | **88.5%** ✗ | ~45% CUDA | 570ms |
| E6a | ✓(+B stale) | ✗ | ✓ | 4 | Best-E5 | ⬜ | — | — |
| E6b | ✗ | ✓(Primary) | ✓ | 4 | Best-E5 | ⬜ | — | — |
| **E7** | **TBD** | **TBD** | **✓** | **4** | **Best** | **⬜** | **—** | **—** |

---

## 七、实验路线图（完整版）

```
E0 Baseline 95.5% ✓ (20-trial)
    │
    ├── E1 VLA-Cache 95.0% ✓ (20-trial)         ← 时间冗余
    │
    ├── Cascade v2 79.5% ✗ (20-trial)           ← B类剪枝失败
    │
    ├── E2_fixed 93.3% ✓ (3-trial)               ← 空间冗余（无加速）
    │   └── E3_20t（E2_fixed 20-trial）────── 93.0% ✓ (186/200，3-trial结论稳定)
    │
    ├── [Phase 1.3] Layer Skip 系列 ★
    │   ├── E5a VLA-Cache+Skip{16-19} ── 43.3% ✗  （层16-19太关键）
    │   ├── E5b VLA-Cache+Skip{24-27} ── 70.0% ✗  （消融：深层，仍不足）
    │   ├── T14 层重要性校准 ──────────── ✅ done   （{3-6}最冗余，{0,31}最重要）
    │   ├── E5f(3t) VLA-Cache+Skip{3-6} ─ 90.0% ⚠️ （校准早期层，最佳3trial）
    │   ├── E5g VLA-Cache+Skip{23,25-27} ─ 70.0% ✗  （消融：校准深层，与E5b相同）
    │   ├── E5h VLA-Cache+Skip{3,5,23,28} ─ 90.0% ⚠️（消融：校准spread，与E5f并列）
    │   ├── E5f(20t) VLA-Cache+Skip{3-6} ─ 88.5% ✗  （真实SR，低于93%目标）
    │   └── E5i VLA-Cache+Skip{3,4} 2层 ─── 90.0% ⚠️ (3t，与E5f相同，327ms CUDA)
    │   ├── E5c VLA-Cache+Skip{8-11}  ── ⬜          （消融：对照）
    │   └── E5e VLA-Cache+Skip{14-19} ── ⬜          （消融：激进）
    │
    ├── [Phase 1.3] 组合系列（条件于E5结果）
    │   ├── E6a Cache+Prune v3+Skip{best} ── TBD （时间+深度）
    │   └── E6b E2_fixed+Skip{best}       ── TBD （空间+深度）★论文主结果候选
    │
    └── E7 最优配置 20-trial 正式 eval ── TBD   ← 论文最终数字
```

**论文叙事逻辑**：
1. 单一冗余维度优化（E0→E1→E2_fixed）：各维度单独均有效但效果有限
2. 层跳过的消融（E5系列）：不同层组/数量的 SR-speedup 曲线
3. 组合的正交性验证（E6系列）：多维度叠加接近效果叠加，互不干扰
4. 最优组合（E7）：在 OpenVLA-OFT + LIBERO 上首次报告三维冗余联合利用

---

## 七、关键文件索引

| 文件 | 内容 | 行数参考 |
|------|------|---------|
| `experiments/robot/openvla_utils.py` | `get_vla_action()`，KV-cache+Prune逻辑 | 977行 |
| `experiments/robot/vla_cache_utils.py` | `task_relevant_selection()`，`compute_preprune_mask()` | 201行 |
| `experiments/robot/libero/run_libero_eval.py` | eval 驱动，flag 定义，per-task SR 计算 | 591行 |
| `modeling_llama.py`（site-packages） | decoder loop（L1051），layer skip 在此添加 | L1051-1113 |
| `experiments/cascade_v3_plan.md` | Cascade v3 原始任务清单（本文件的前身） | — |

**注**：`modeling_llama.py` 路径为：
`/home/jzzz/miniconda3/envs/openvla-oft/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`

---

## 八、当前代码状态确认（2026-03-14）

| 配置 | 实际行为 | 已验证SR |
|------|---------|---------|
| `use_preprune_v3=False, use_vla_cache=False` | Baseline | 95.5% |
| `use_preprune_v3=False, use_vla_cache=True` | VLA-Cache only | 95.0% |
| `use_preprune=True, use_vla_cache=True` | Cascade v2 | 79.5% |
| `use_preprune_v3=True, use_vla_cache=False` | **E2_fixed（当前推荐）** | **93.3%** |
| `use_preprune_v3=True, use_vla_cache=True` | E3残留（等同E2_fixed，fixed_prune_p_primary未写入cache）| 未测 |
