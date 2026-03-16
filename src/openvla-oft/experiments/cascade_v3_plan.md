# Cascade v3 实现计划
**日期**: 2026-03-13
**目标**: 用SpecPrune-VLA风格的三集合token selection（V_global ∪ V_dynamic ∪ V_local）替换当前错误的B类剪枝逻辑，修复79.5% → 目标≥93%

---

## 核心思路

- **Pre-prune职责**: 删空间冗余（信息荒漠 = 静态+历史不重要+当前不重要）
- **VLA-Cache职责**: 删时间冗余（stable的patch复用KV）
- **两者不竞争**: pre-prune保留的token中，VLA-Cache进一步对stable ones做KV复用

保留集合 = V_global ∪ V_dynamic ∪ V_local
剪除集合 = 256 - 保留集合

| 集合 | 含义 | 来源 | 参数 |
|------|------|------|------|
| V_global | 上一步attn top-K（跨步时序保护） | last_attn_scores.topk(K_global).indices | K_global=60 |
| V_dynamic | 当前帧变化的patch（pixel-sim低） | ~find_static_patches(threshold=0.99)的补集 | dynamic_threshold=0.99 |
| V_local | 当前步text-attn top-K | last_attn_scores_primary/wrist.topk(K_local) | K_local=80 |

---

## 任务清单

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| T1 | 新增 `compute_preprune_mask()` 函数 | vla_cache_utils.py | ✅ done |
| T2 | 单元测试 T1（shape/edge cases） | - | ✅ done |
| T3 | last_caches 新增 v_global 字段（读取） | openvla_utils.py | ✅ done |
| T4 | last_caches 新增 v_global 字段（写入） | openvla_utils.py | ✅ done |
| T5 | 替换 Step 4 use_preprune 分支逻辑 | openvla_utils.py | ✅ done |
| T6 | run_libero_eval.py 新增 use_preprune_v3/use_adaptive flag | run_libero_eval.py | ✅ done |
| T7 | 跑 E2（preprune_v3 only, 3-trial）验证SR | - | ✅ done: 76.7% |
| T7b | 跑 E2'（K_local=130, 3-trial）验证是否过度剪枝 | - | ✅ done: 66.7%（更差，不是K_local问题） |
| T7c | E2_fixed: 仅Primary做pre-prune + frozen V_local（3-trial） | openvla_utils.py | ✅ done: 93.3% |
| T8 | 跑 E3（cascade v3 = preprune_v3_fixed + VLA-Cache, 3-trial） | - | ⬜ pending |
| T9 | （视E2/E3结果）自适应开关 ADP velocity（E4） | openvla_utils.py | ⬜ pending |
| T10 | 20-trial 正式eval（E3/E4最优配置） | - | ⬜ pending |

---

## 超参数

```python
K_GLOBAL = 60           # V_global大小（每相机）
K_LOCAL  = 80           # V_local大小（每相机）
DYNAMIC_THRESHOLD = 0.99  # V_dynamic pixel-sim阈值（<0.99为动态，宽松于VLA-Cache的0.996）
# 自适应开关（E4）
VELOCITY_TAU   = 8      # adjacent-extrema窗口大小
MAX_CONSEC_PRUNE = 3    # 最大连续prune window数
```

---

## 关键文件

- `experiments/robot/vla_cache_utils.py` — T1/T2
- `experiments/robot/openvla_utils.py` — T3/T4/T5/T9
- `experiments/robot/libero/run_libero_eval.py` — T6

---

## 实验结果记录

| 条件 | Episodes | SR | 备注 |
|------|----------|-----|------|
| E0 Baseline | 200 | 95.5% | ✓ 已有 |
| E1 VLA-Cache only | 200 | 95.0% | ✓ 已有 |
| E2 preprune_v3 (K_local=80) | 30 | 76.7% | T1↑(55%→100%) T4/T9↓(33%) |
| E2' preprune_v3 (K_local=130) | 30 | 66.7% | 更差，说明问题不是K_local而是wrist雪球+V_local偏差 |
| E2_fixed primary-only + frozen V_local | 30 | **93.3%** | T4(drawer)33% 仍失败，其余全100% |
| E3 cascade v3_fixed + VLA-Cache | - | - | 待跑 |
| E4 cascade v3 + ADP | - | - | 待跑 |

## E2 诊断笔记（2026-03-13）

**关键发现**：
- V_local 和 V_global 来自同一源（last_attn_scores），V_global ⊆ V_local → 3集合实际只有 V_dynamic ∪ V_local
- K_local=80 → 每相机保留~80-110 tokens，剪除57%+ → 对精密任务(T4 drawer, T9)过激
- prev_images bug：eval脚本的prev_img跟踪是当前帧而非上一查询帧 → 已修复，用prev_query_images_raw
- V_global有效：T1(ramekin) 55%→100%，跨步时序保护工作

---

## E2_fixed 设计（2026-03-13）

### 核心修改：仅Primary做pre-prune，Wrist全量保留

**Wrist不做pre-prune的原因**：
- wrist视野随机械臂持续变化
- 视野刚稳定时出现"死区"：V_dynamic收缩但frozen V_local指向旧视野
  → 新视野中的重要token三集合都不覆盖 → 被错误剪除
- wrist只有256 token，冗余度低，代价收益不对等

**Primary可以安全pre-prune**：
- 固定视角，大量背景冗余
- Frozen V_local（step 0全序列attention）能准确识别常驻任务区域

**V_local改为frozen**：
- 用step 0的完整attention scores（未被compact bias污染）
- 存于`last_caches['init_attn_scores_primary']`，episode内不更新
- V_global仍每步从compact attention更新（保持时序保护）

**Token数学**：
- 旧方案（E2）：primary ~110 + wrist ~110 = ~220 kept（prune 57%）
- 新方案：primary ~130 + wrist 256 = ~386 kept（prune 25%）

**雪球效应消除**：
- wrist不剪 → 无偏差积累
- primary V_local来自step 0全序列 → 无compact bias

**VLA-Cache集成改善**：
- wrist始终256 token → 索引对齐trivial，A-class KV复用可正常运行
- primary变长 → 仍需处理，但只影响一半的vision tokens

### 实现细节

**openvla_utils.py改动**：
1. 读取 `init_attn_scores_primary`（从function argument的last_caches，step 0时为None）
2. cascade v3 pre-prune：primary用init_attn_scores，wrist设prune_w=[]
3. last_caches写入：新增 `init_attn_scores_primary`（仅step 0写入，后续透传）

**新字段**：
- `last_caches['init_attn_scores_primary']`: step 0全序列attention scores [256]，episode内不变
