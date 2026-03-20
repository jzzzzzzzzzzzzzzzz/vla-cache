# 论文写作框架：VLA推理加速的三维冗余分析

> **论文定位**：实证分析型论文（Empirical Analysis Paper），不是方法提出型论文（Method Paper）
> **核心卖点**：在OpenVLA-OFT上首次系统分析时间/空间/深度三个冗余维度的加速效果及交互关系，特别是揭示了KV cache框架内token物理删除失败的机制性原因
> **目标期刊**：Neurocomputing (SCI Q1/Q2) 或 EAAI (SCI Q1)
> **与竞争论文的差异化**：AC²-VLA/EfficientVLA/MoLe-VLA都是在CogACT（扩散头）上做的；我们在OpenVLA-OFT（L1回归+parallel decoding）上做，且是training-free的

---

## 暂定标题

**英文**：Understanding Multi-Dimensional Redundancy in VLA Inference: An Empirical Study of Temporal Caching, Spatial Pruning, and Depth Skipping on OpenVLA-OFT

**中文工作标题**：VLA推理多维冗余分析：OpenVLA-OFT上时间缓存、空间剪枝与深度跳层的实证研究

---

## 全文逻辑主线（一句话概括每节的核心论点）

1. **引言**：VLA模型推理延迟是部署瓶颈，现有加速方法各自针对单一冗余维度，缺乏对多维冗余交互效应的系统理解
2. **相关工作**：梳理三个维度的现有方法，指出缺少在同一平台上的对比分析
3. **冗余分析**：通过实验量化OpenVLA-OFT中三个维度的冗余程度和分布特征
4. **方法**：提出基于VLA-Cache的扩展缓存框架 + 校准式Layer Skip的组合方案
5. **关键发现：KV Cache内Token Deletion的失败机制**：这是论文最独特的贡献
6. **实验**：在LIBERO四个suite上的完整消融实验
7. **讨论与结论**：三维冗余的互补性与限制，对后续研究的指导

---

## 第1节 Introduction（约1.5页）

### 1.1 背景与问题（第1-2段）
**要说的内容**：VLA模型（RT-2, OpenVLA, CogACT, π0等）在机器人操作中展现了强大的泛化能力，但推理延迟高（OpenVLA-OFT在RTX 4090上单步~79ms），限制了实时闭环控制的控制频率。
**引用**：OpenVLA [Kim et al., 2024], OpenVLA-OFT [Kim et al., 2025], CogACT [Li et al., 2024]

### 1.2 现有加速方法的局限（第3段）
**要说的内容**：近期的加速方法分别利用了不同维度的冗余——VLA-Cache利用帧间时间冗余做KV复用（时间维度），SpecPrune-VLA/ADP利用背景token的空间冗余做剪枝（空间维度），EfficientVLA/MoLe-VLA利用层间表示相似性做层跳过（深度维度）。但这些方法各自独立工作，缺少对"这三个维度如何在同一模型中交互"的系统理解。
**关键问题**：(a) 三个维度的加速是否可以安全叠加？(b) 叠加后的收益是否可加？(c) 在哪些场景下哪种维度更有效？

### 1.3 我们的工作（第4段）
**要说的内容**：我们在OpenVLA-OFT + LIBERO上系统研究了三个冗余维度的加速效果。主要发现包括：
- (发现1) VLA-Cache的KV复用可以从静态+低注意力token扩展到所有低注意力token（Extended Cache），额外节省FFN计算，SR仅降~2%
- (发现2) 在VLA-Cache的KV cache框架内做token物理删除会导致严重的成功率下降，原因是softmax分布偏移——这是首次被系统分析的机制性障碍
- (发现3) 校准式Layer Skip（基于层间cosine similarity选择跳过层）在OpenVLA-OFT上的最优层组是早期冗余层{3-6}，与CogACT上的深层剪枝不同
- (发现4) VLA-Cache + Layer Skip的组合可以在BF16精度下实现1.2-1.4×的实际wall-clock加速（待4090验证）

### 1.4 贡献总结（第5段）
**三点贡献**（标准格式）：
1. 在OpenVLA-OFT上首次进行了三维冗余（时间/空间/深度）的系统定量分析，包括ABCD四类token分布统计和32层层间重要性校准
2. 揭示了KV cache框架内token物理删除导致SR下降的三个机制性原因（softmax污染、分布偏移、全注意力假设破坏），解释了为什么现有pruning方法都选择在LLM输入端而非KV cache内部操作
3. 提出了VLA-Cache + 校准式Layer Skip的training-free组合方案，在LIBERO benchmark上验证了其效果（具体数字待BF16实验确认）

---

## 第2节 Related Work（约1.5页）

### 2.1 VLA模型与推理瓶颈（0.3页）
**要说的内容**：简述VLA模型的发展脉络（RT-2→OpenVLA→CogACT→OpenVLA-OFT→π0），指出LLM backbone占推理延迟的70%+（引用SpecPrune-VLA Figure 1b的数据），parallel decoding缩短了action token生成延迟但LLM backbone仍是瓶颈。
**引用**：Survey [arXiv:2510.24795], VLA-Perf [arXiv:2602.18397]

### 2.2 时间维度：跨帧KV缓存（0.4页）
**要说的内容**：VLA-Cache首创的跨帧KV复用思路——利用机器人操作场景中帧间视觉变化小的特性，对静态且非任务关键的visual token复用上一帧的KV值。LAC将其升级为可学习策略。TTF-VLA扩展了融合机制。
**必须讨论**：VLA-Cache [NeurIPS 2025], LAC [arXiv:2602.00686], TTF-VLA [AAAI 2026]
**与我们的关系**：我们的Extended Cache扩展了VLA-Cache的复用范围，但在cache框架内做token deletion遇到了根本性障碍

### 2.3 空间维度：Visual Token剪枝/压缩（0.4页）
**要说的内容**：SpecPrune-VLA的两级剪枝（action-level + layer-level）+ action-aware控制器。ADP的text-driven剪枝 + 末端执行器速度门控。EfficientVLA的task-relevance + diversity双准则选择。SP-VLA的时空联合。
**必须讨论**：SpecPrune-VLA, ADP [ICLR 2026], EfficientVLA [NeurIPS 2025], SP-VLA, VLA-Pruner
**与我们的关系**：我们尝试了多种token selection策略（ABCD分类、三集合并集），发现在VLA-Cache框架内物理删除token会失败

### 2.4 深度维度：层跳过与动态路由（0.3页）
**要说的内容**：MoLe-VLA的mixture-of-layers router + CogKD蒸馏。DeeR-VLA的early exit。EfficientVLA的层间cosine similarity校准。DySL-VLA的动态-静态层跳过。
**必须讨论**：MoLe-VLA [AAAI 2026], EfficientVLA [NeurIPS 2025], DySL-VLA
**与我们的关系**：我们采用类似EfficientVLA的校准方法选择跳过层，但在OpenVLA-OFT上验证了不同的最优层组

### 2.5 多维联合加速（0.2页）
**要说的内容**：AC²-VLA提出action-prior router统一协调缓存/剪枝/跳层三个维度，但它需要训练且基于CogACT。EfficientVLA组合了层剪枝/token剪枝/去噪步缓存，但第三个维度针对扩散头。我们的工作是首个在OpenVLA-OFT（L1回归头）上做training-free三维分析的。
**必须讨论**：AC²-VLA [arXiv:2601.19634], EfficientVLA
**关键区别**：AC²-VLA是trained + CogACT；我们是training-free + OpenVLA-OFT

---

## 第3节 冗余分析（Redundancy Analysis，约2页）

> **这一节是论文的分析基础，相当于"Preliminary + Motivation"**

### 3.1 OpenVLA-OFT推理流程（0.3页）
**要说的内容**：描述OpenVLA-OFT的推理pipeline——双目ViT编码（primary 256 patches + wrist 256 patches）→ 512 visual tokens + ~34 text tokens + 56 action tokens → LLaMA-2-7B (32层) → L1回归MLP动作头 → 7-DOF连续动作。
**放公式**：序列长度 L = 1 + N_primary + N_wrist + N_text + N_action = 603。每层FLOPs = 4LD² + 2L²D + 2LDM。

### 3.2 时间冗余量化（0.5页）
**要说的内容**：基于你的Phase 1.1实验数据——在LIBERO-Spatial的1205帧中，统计帧间patch cosine similarity分布。展示：平均约42.6%的primary patches在帧间高度相似（A类，similarity > 0.996且text-attention低），可以安全复用KV。
**放图表**：
- Figure 3a：帧间cosine similarity的直方图分布（所有帧的所有patch的similarity值）
- Table 1：ABCD四类token的平均占比（A=42.6%, B=18.3%, C=16.0%, D=23.1%）

### 3.3 空间冗余量化（0.5页）
**要说的内容**：基于你的attention分析——B类token（动态+低注意力）占18.3%，它们是"发生了视觉变化但模型不关注"的区域。加上C类中的"静态但高注意力"token（模型需要重新计算以跟踪任务状态），空间上真正冗余的token约占总量的18-20%。
**放图表**：
- Figure 3b：token四类分布的可视化（在原始图像上用颜色标注ABCD四类patch）
- 对比primary vs wrist相机的冗余结构差异

### 3.4 深度冗余量化（0.5页）
**要说的内容**：基于你的层重要性校准实验——计算OpenVLA-OFT 32层的层间cosine similarity，发现Layer 3-6的重要性分数I最低（0.020），Layer 0和31最高。这与EfficientVLA在CogACT上的发现不同（CogACT的最冗余层偏深）。
**放图表**：
- Figure 3c：32层的重要性分数I^(l)柱状图（参考EfficientVLA Figure 1b风格）
- 标注：OpenVLA-OFT的冗余层分布与CogACT不同，说明层冗余是architecture-specific的

### 3.5 关键观察：三维冗余的独立性（0.2页）
**要说的内容**：时间冗余取决于帧间场景变化速度，空间冗余取决于token与任务的相关性，深度冗余取决于层间表示的冗余度——三者的决定因素不同，暗示了它们可能是正交的、可以安全叠加的。这个假设在后续实验中验证。

---

## 第4节 方法（Method，约2.5页）

### 4.1 VLA-Cache回顾（0.5页）
**要说的内容**：简要复述VLA-Cache的三个核心机制——Static Token Selection（公式4-5）、Task-Relevant Eviction（公式6-8）、Layer-Adaptive Caching（公式9）。这是我们框架的基础。
**放公式**：复述VLA-Cache的核心公式（Sim, P_static, P_task-relevant, P_final, α^l）

### 4.2 Extended Cache：将KV复用扩展到B类Token（0.5页）
**要说的内容**：我们观察到B类token（动态+低注意力）虽然视觉上发生了变化，但由于其text-attention低，stale KV对模型决策的影响很小。因此我们将VLA-Cache的KV复用范围从A类（静态+低注意力）扩展到A+B类（所有低注意力token），不清零KV、不加mask，让stale KV像A类一样正常参与attention。
**放公式**：
- 定义B类token集合：P_B = {P^{i,j}_t | Sim(P^{i,j}_t, P^{i,j}_{t-1}) < τ AND S_task-relevance[i,j] < τ_task}
- Extended复用集合：P_extended = P_final ∪ P_B（VLA-Cache原始的P_final加上B类）
- hidden_states中移除P_extended位置的token，但保留其在KV cache中的stale值

### 4.3 校准式Layer Skip（0.5页）
**要说的内容**：采用EfficientVLA的层间cosine similarity方法计算层重要性，选择最低的n层跳过。与EfficientVLA的区别是我们在OpenVLA-OFT（非CogACT）上校准，发现最优层组不同。
**放公式**：
- 层重要性分数 I^(l) = 1 - (1/|D|)∑_i (1/L)∑_j cos(x^(l)_{i,j}, x^(l+1)_{i,j})（EfficientVLA Eq.1）
- 选择I^(l)最低的n层组成skip set S = {l | I^(l) ∈ bottom-n}
- 跳过层的处理：h_k = h_{k-1}（恒等映射，等价于残差连接中f(x,θ)=0）

### 4.4 组合框架（0.5页）
**要说的内容**：VLA-Cache + Extended Cache + Layer Skip的完整推理流程。
**放Algorithm 1（类似VLA-Cache的Algorithm 1风格）**：
```
Input: 帧序列{I_t}, 指令u, skip层集合S, 阈值τ, τ_task
Output: 动作序列{a_t}

1. Step 0: 完整forward（无加速），初始化KV cache和attention scores
2. Step t>0:
   a. 计算帧间patch similarity → 识别A类（static+low-attn）和B类（dynamic+low-attn）
   b. 构建Extended复用集合 P_extended = P_final ∪ P_B
   c. 从hidden_states中移除P_extended位置的token（保留stale KV）
   d. 对剩余token执行LLM forward，跳过S中的层
   e. 在非跳过层中，VLA-Cache对A类token做layer-adaptive KV复用
   f. 输出动作a_t
```

### 4.5 复杂度分析（0.5页）
**放公式**：
- Baseline FLOPs = N_layers × (4LD² + 2L²D + 2LDM)
- Extended Cache节省 = B类token数 × (Q/K/V projection + FFN) per non-skipped layer
- Layer Skip节省 = |S|/N_layers × 总FLOPs
- 总理论FLOPs降低比例

---

## 第5节 关键发现：KV Cache内Token Deletion的失败机制（约1.5页）

> **这一节是论文最独特的贡献。写作参考ADP论文的Appendix "Case Discussions"风格，但更深入。**

### 5.1 问题定义（0.3页）
**要说的内容**：直觉上，既然VLA-Cache已经在KV cache中标记了"可复用"的token，那么进一步把一些token从序列中完全删除（物理移除hidden_states，减小attention matrix的L维度）应该带来额外的加速（O(L²)减小）。我们系统地尝试了四种deletion策略，发现它们全部导致SR显著下降。

### 5.2 四种策略的实验结果（0.3页）
**放Table**：四种策略的SR对比表（v1 zero KV=5.5%, v2 causal mask=20%, v2b B-only mask=66.7%, v3 stale KV=93.3%）
- 每种策略一行，列出：策略名称、B类KV处理方式、SR、失败原因

### 5.3 机制分析（0.6页）
**5.3.1 Softmax污染（Zero KV策略的失败原因）**
**放公式**：Q·K_B = Q·0 = 0 → exp(0) = 1。当非B位置的logit普遍为负时（层归一化后的典型情况），B位置的exp(0)=1远大于非B位置的exp(负值)<<1，导致B位置主导softmax分母。数值估算：92个B位置 × exp(0)=1 = 92，510个非B位置 × exp(-1)≈0.37 = 189，B占比 = 92/(92+189) ≈ 33%。注意力输出被缩减~1/3，32层累积后效果灾难性。

**5.3.2 分布偏移（-inf掩码策略的失败原因）**
**放公式**：用-inf掩码屏蔽B位置后，softmax的归一化分母从∑_{i=1}^{603} exp(s_i)变为∑_{i∉B} exp(s_i)，归一化范围缩小。模型训练时始终在603个位置上做softmax，从未见过"部分KV缺失"的分布。32层累积后，每层的轻微分布偏移放大为严重的表示偏离。

**5.3.3 VLA-Cache全注意力假设的发现**
**要说的内容**：通过代码分析发现VLA-Cache强制past_seen_tokens=0 → causal_mask始终为None → 模型使用全注意力（所有token可以attend到所有KV位置，无因果限制）。这意味着添加任何形式的attention mask都会破坏模型依赖的全注意力格局，即使只是屏蔽B类token也是如此。

### 5.4 为什么Stale KV策略有效（0.3页）
**要说的内容**：v3（不清零、不加mask、保留stale KV）之所以能维持93.3% SR，是因为它完全保留了VLA-Cache的全注意力模式。B类token的stale KV虽然不精确（因为视觉上发生了变化），但由于B类token的attention score低，stale KV对attention输出的影响很小。本质上，Extended Cache把B类token当成了"永久版的A类"——不更新hidden states，不清零KV，让模型自行决定给它们多少注意力。

### 5.5 与现有工作的关系（0.2页）
**要说的内容**：这个发现解释了为什么SpecPrune-VLA、ADP等成功的pruning方法都选择在LLM输入端做token物理删除（从一开始就不让这些token进入LLM），而不是在KV cache内部操作。在LLM输入端删除token构建的是一个"更短但自洽的序列"，模型从Layer 0开始就在短序列上建立注意力分布；而在KV cache内部删除token改变的是"已经建立的注意力格局"，32层累积的分布偏移是不可控的。

---

## 第6节 实验（约3页）

### 6.1 实验设置（0.5页）
**要说的内容**：基座模型（OpenVLA-OFT, LLaMA-2-7B, parallel decoding）、Benchmark（LIBERO四个suite）、评估指标（Success Rate, FLOPs, CUDA Latency, Control Frequency）、硬件（RTX 4090, BF16精度）、每个suite 10 tasks × 20 trials = 200 episodes/条件。
**对比方法**：OpenVLA-OFT baseline, VLA-Cache [Xu et al., 2025], SpecPrune-VLA（引用论文数字，注明硬件差异）

### 6.2 主结果（Main Results，0.8页）
**放Table（论文核心表格）**：

| 方法 | 训练 | Spatial | Object | Goal | Long | Avg | FLOPs↓ | Latency↓ | Speedup |
|------|------|---------|--------|------|------|-----|--------|----------|---------|
| OpenVLA-OFT (baseline) | - | X% | X% | X% | X% | X% | 1.00 | Xms | 1.00× |
| VLA-Cache | Free | X% | X% | X% | X% | X% | X | Xms | X× |
| VLA-Cache + Extended Cache | Free | X% | ... | ... | ... | X% | X | Xms | X× |
| VLA-Cache + Skip{3-6} | Free | X% | ... | ... | ... | X% | X | Xms | X× |
| VLA-Cache + Extended + Skip | Free | X% | ... | ... | ... | X% | X | Xms | X× |
| SpecPrune-VLA† | Free | 98.2% | 96.3% | 95.2% | 83.7% | 93.4% | - | 72.4ms | 1.46× |
| ADP† | Free | 99.4% | 98.0% | 96.4% | 91.2% | 96.3% | - | - | 1.35× |

†表示引用论文数字，非本文复现，硬件/设置可能不同

### 6.3 消融实验（Ablation Study，0.8页）
**6.3.1 Layer Skip层组选择消融**
**放Table**：不同层组的SR对比（{16-19}=43.3%, {24-27}=70%, {3-6}=88.5%, {3,5,23,28}=90%），验证校准方法的有效性

**6.3.2 Extended Cache的消融**
**放Table**：VLA-Cache alone vs VLA-Cache + Extended Cache的SR和FLOPs对比

**6.3.3 Token Deletion策略的消融（连接到第5节的分析）**
**放Table**：四种deletion策略的完整SR对比表（这个表在第5节已经出现过，这里可以引用）

### 6.4 Per-task分析（0.5页）
**放Table**：每个task的per-task SR对比（如你现有的per-task表格），分析哪些task对加速更敏感
**要说的内容**：Layer Skip对T8(on stove)和T10(on wooden cabinet)影响较大，分析原因——这些任务需要精确的空间定位，跳过早期特征提取层影响了空间编码

### 6.5 延迟分析（0.4页）
**放Table/Figure**：wall-clock延迟breakdown（tokenizer/LLM/action head各占多少），对比baseline和加速后的latency分布

---

## 第7节 Discussion（约0.8页）

### 7.1 三维冗余的互补性
**要说的内容**：VLA-Cache（时间）+ Layer Skip（深度）的组合加速接近各自独立加速之和，验证了正交性假设。但空间维度（token deletion）与时间维度（KV caching）之间存在冲突（如第5节分析），不能简单叠加。

### 7.2 与CogACT平台的差异
**要说的内容**：EfficientVLA在CogACT上发现token pruning能提升SR（扩散头的纠错能力），但在OpenVLA-OFT上token deletion会降低SR（parallel decoding的单次forward无纠错机制）。这说明加速策略的效果是architecture-dependent的。

### 7.3 Limitations
- 实验仅在仿真环境（LIBERO）中进行，未在真实机器人上验证
- Layer Skip的层选择需要校准数据，虽然是training-free但不是zero-shot的
- 未探索量化与caching/skip的联合效果

### 7.4 对后续研究的指导
- 在KV cache框架内做token deletion需要training-based方案（如LAC）来学习对分布偏移的适应
- OpenVLA-OFT的层冗余分布（早期层最冗余）与CogACT不同，说明加速策略需要per-architecture校准
- 未来工作可以探索AC²-VLA风格的trained router在OpenVLA-OFT上的效果

---

## 第8节 Conclusion（约0.3页）
**要说的内容**：总结三维冗余分析的主要发现，强调KV cache内token deletion失败机制的实证价值，以及VLA-Cache + Layer Skip组合方案的实用性。

---

## 你在4090上需要跑的完整实验清单

### 必须跑的（论文核心数据）：
1. E0 Baseline（BF16，4个suite × 20-trial）
2. E1 VLA-Cache only（BF16，4个suite × 20-trial）
3. VLA-Cache + Skip{3-6}（BF16，4个suite × 20-trial）
4. VLA-Cache + Extended Cache（BF16，Spatial 20-trial，确认v3数字）
5. VLA-Cache + Extended Cache + Skip{3-6}（BF16，Spatial 20-trial）

### 消融实验（至少Spatial suite）：
6. Layer Skip alone（无VLA-Cache，Skip{3-6}，Spatial 3-trial）
7. Skip{16-19}（Spatial 3-trial，对照组）
8. Skip{24-27}（Spatial 3-trial，对照组）
9. Skip{3,5,23,28}（Spatial 3-trial，spread对照组）
10. VLA-Cache + Skip 2层{3,4}（Spatial 3-trial）

### 如果时间允许：
11. Token deletion系列的BF16复现（v1/v2b/v3各一组Spatial 3-trial，验证INT4结论在BF16下是否一致）

### 预计GPU时间：
- 4个suite × 20-trial × 5个配置 × ~2小时/suite = ~40小时
- 消融实验 ~10-15小时
- 总计约50-60 GPU小时，按¥1.87/h计算约¥95-112

---

## 图表清单

### 必须有的图：
- **Figure 1（方法概览）**：类似VLA-Cache Figure 2或AC²-VLA Figure 2的风格，展示三维冗余 + 我们的加速框架
- **Figure 2（冗余量化）**：三个子图——(a)帧间similarity分布 (b)ABCD token分布可视化 (c)层重要性分数柱状图
- **Figure 3（Token Deletion失败机制）**：示意图展示softmax污染的数值过程
- **Figure 4（Per-task结果可视化）**：雷达图或热力图展示不同配置在10个task上的SR对比

### 必须有的表：
- **Table 1**：ABCD四类token统计（来自Phase 1.1数据）
- **Table 2**：主结果对比表（所有方法的SR+FLOPs+Latency）
- **Table 3**：Layer Skip消融表
- **Table 4**：Token Deletion策略对比表
- **Table 5**：Per-task SR详细表

---

## 可以现在就开始写的部分

1. **Abstract**（等4090数据出来后最后写）
2. **Section 1 Introduction**（除了具体数字，逻辑框架可以先写）
3. **Section 2 Related Work**（完全可以现在写，不依赖实验数据）
4. **Section 3.1-3.2 冗余分析的前两小节**（基于你已有的Phase 1.1数据）
5. **Section 4.1-4.3 方法描述**（公式和算法描述不依赖具体SR数字）
6. **Section 5 Token Deletion失败机制**（这是你最完整的分析，现在就可以写）
