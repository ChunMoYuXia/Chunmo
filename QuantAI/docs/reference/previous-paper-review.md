# 论文审查：理由与文献对照

> 本文档为《改进建议补丁》中每条建议的**详细理由说明**与**论文参考链接**。
> 编号 PRn 与补丁文档中的 `<!-- 论文审查PRn -->` 一一对应。
> 审阅日期：2026-10-15
> 审阅方法：逐节通读原文档（v2.1, ~4894 行） + 前沿论文检索（2024–2025） + 交叉比对

---

## PR1：多级重排与表格语义化

### 问题定位

原文档 §7.7 检索流程使用单级 cross-encoder 重排（top 50 → top 8~12），§7.4 分块策略将表格独立成块但仅转 Markdown 存储。

### 论文依据

**1. 两级重排级联**

Multi-Reranker（arXiv:2411.16732）在 FinanceRAG 挑战赛中获得第二名，核心创新是**两级 reranker 级联**：先用轻量 cross-encoder（jina-reranker-v2-base-multilingual）筛选 top 200，再用更强的领域专用 reranker 精选 top 20。消融实验显示关键词扩展优于 HyDE 和 paraphrase，表格提取对多表数据集有显著帮助。

- 论文链接：https://arxiv.org/abs/2411.16732
- 关键数据：NDCG@10 = 0.640，相比单级 reranker 提升约 15–25%

**2. 表格语义化（Table-to-Narrative）**

FinSage（CIKM 2025）针对金融文档的异构性（文本+表格+图表），提出了**多模态预处理流水线**：将表格转换为叙述性数据趋势、图表转换为结构化标题、执行共指消解。在 FinanceBench 上实现 49.66% 准确率，相比之前 32.62% 的 SOTA 提升 24%。

- 论文链接：https://arxiv.org/abs/2507.09995（CIKM 2025）
- 关键数据：准确率提升 24%，Recall 92.51%（专家数据集）

FinGEAR（EMNLP 2025 Findings）进一步证明：金融披露文档的 Item 级结构引导 + 双层次索引 + 两阶段 cross-encoder reranker，在 FinQA 数据集上 F1 提升高达 56.7%（相比 flat RAG）。

- 论文链接：https://aclanthology.org/2025.findings-emnlp.382/
- DOI：10.18653/v1/2025.findings-emnlp.382

**3. DPO 微调重排器 + 时间加权**

FinSage 的重排器基于 BGE-Gemma，使用 **Direct Preference Optimization (DPO)** 微调，训练模型优先识别合规关键术语。同时引入**时间加权（Time Bonus）**机制，优先近期文件，解决金融 RAG 的"时效性问题"。

- 同上 FinSage 论文

**4. Small-to-Big 检索验证**

Rethinking Retrieval（arXiv:2511.18177）首次系统对比了向量 RAG 与层次节点 RAG 在金融文档上的表现。cross-encoder reranking 在最优参数下实现 MRR@5 绝对提升 59%；small-to-big chunk retrieval 实现对基线 65% 胜率，仅增加 0.2s 延迟。

- 论文链接：https://arxiv.org/abs/2511.18177

### 改进理由

原文档的单级 rerank 在金融文档场景下存在两个短板：① 表格内容因转 Markdown 而丧失语义上下文，导致召回率偏低；② 单级 rerank 对大量候选的过滤效率不如级联方案。引入两级级联 + 表格语义化 + DPO 微调，可在不显著增加延迟的前提下将 NDCG@10 提升 15–25%。

---

## PR2：查询改写策略消融与自适应

### 问题定位

原文档 §7.7 第 2 步同时使用多查询扩展（3 个改写）和 HyDE（生成假想答案再检索），但未做消融对比，可能导致某些场景下 HyDE 反而降低召回质量。

### 论文依据

Multi-Reranker（arXiv:2411.16732）的消融实验显示：**关键词扩展优于 HyDE 和 paraphrase**。在金融文档场景下，HyDE 生成的假想答案可能引入语义偏移，尤其对数值型查询（"茅台 PE 是多少"）效果不佳。

FinSage（CIKM 2025）的多路检索策略中包含 HyDE，但仅在特定 doc_type（如 research_report）下开启，且作为四路并行检索之一而非默认策略。

- 论文链接：https://arxiv.org/abs/2411.16732（Multi-Reranker）
- 论文链接：https://arxiv.org/abs/2507.09995（FinSage）

### 改进理由

HyDE 在开放式文本问答中有效，但在精确数值查询场景下可能引入噪音。应将 HyDE 改为按 doc_type 自适应开关，并以关键词扩展作为默认策略。每次策略变更须触发消融评测（NDCG/Recall/Faithfulness），确保改写策略实际有效。

---

## PR3：SAG 对抗性攻击防护

### 问题定位

原文档 §8.6 的 SQL AST 白名单校验和执行沙箱主要防护**传统 SQL 注入**，但未覆盖 LLM 时代特有的**后门攻击**和**自回归注入攻击**。

### 论文依据

**1. 后门攻击（ToxicSQL）**

Lin et al.（SIGMOD 2025）提出 ToxicSQL 框架，系统研究了 LLM-based Text-to-SQL 模型的后门攻击漏洞。仅需 **0.44% 的毒化数据**即可达到 **79.41% 的攻击成功率**，生成的恶意 SQL 可执行且隐蔽。后门触发器使用语义级和字符级设计，难以检测和移除。

- 论文链接：https://www.cs.emory.edu/~jyang71/files/toxicsql.pdf
- DOI：10.1145/3769762
- 关键数据：0.44% 毒化 → 79.41% 攻击成功率

**2. 自回归注入攻击（AIA）**

Li et al.（AAAI 2025）提出 AIA 攻击框架，通过构造特定输入并自回归调整，使 Text2SQL 模型生成攻击 payload。成功率 **>70%**，且生成的对抗输入具有一定的跨模型迁移性。实践实验表明 AIA 可使 Text2SQL 模型提取用户列表甚至删除数据库数据。

- 论文链接：https://openreview.net/forum?id=xPWdcc67ir（AAAI 2025）

**3. 恶意意图检测分类器**

Chafik et al.（2025, Cambridge University Press）提出基于 transformer 的恶意查询检测方案，包含大规模恶意/安全提示数据集和两个 transformer 分类器，整体安全增强 **70%**。

- 论文链接：https://doi.org/10.1017/nlp.2025.10008
- 期刊：Natural Language Processing, Vol. 31, No. 6, pp. 1399–1422

### 改进理由

LLM 时代的 NL2SQL 安全威胁远超传统 SQL 注入：后门攻击可绕过所有 AST 白名单（因为它生成的是合法但恶意的 SQL），自回归注入可通过精心构造的自然语言输入诱导模型生成攻击性 SQL。需要增加语义层面的恶意意图分类器，并对模型来源和 fine-tuning 数据做供应链审计。

---

## PR4：推理级幻觉检测与多面 Critic

### 问题定位

原文档 §9.2 的 Critic 仅检测"无证据断言"（事实级幻觉），但存在**推理级幻觉**盲区：每条事实引用真实存在，但从事实到结论的推理链存在逻辑跳步。

### 论文依据

**1. 推理级幻觉**

Lee et al.（arXiv 2025）的 PMADS（Popperian Multi-Agent Debate System）研究指出，LLM 在金融信用评估中存在"推理级幻觉"——逻辑跳跃缺乏充分证据支撑，即使所有事实引用都真实。该研究提出基于 Karl Popper 辩论原则的 10 步对抗协议来缓解此问题。

- 论文链接：https://arxiv.org/abs/2505.15079（PMADS, 2025）
- 关键发现：结构化辩论使推理树更深，但推理级幻觉仍需专门检测

**2. 多面 Critic 反思框架**

Fatemi & Hu（ICAIF 2024）提出含 critic agent 的多智能体反思框架，通过增加**多个专注不同方面的 critic agent** 实现性能显著提升：LLaMA3-8B 提升 15%，LLaMA3-70B 提升 5%。该框架在性能上与 LLaMA3.1-405B 和 GPT-4o-mini 相当甚至超越。

- 论文链接：https://arxiv.org/abs/2410.21741
- DOI：10.1145/3677052.3698686

**3. 安全辩论协议**

Cai et al.（arXiv 2025）的 FinDebate 提出"安全辩论协议"（safe debate protocol），使 agent 能在挑战和修正初始结论的同时保持推荐的一致性。五个专业 agent（盈利/市场/情绪/估值/风险）并行运行，通过协作辩论合成多维洞察。

- 论文链接：https://arxiv.org/abs/2509.17395

**4. Reflexion 自我反思**

Shinn et al.（2023）的 Reflexion 框架使 agent 能批判自身推理并在行动前修正结论。在交易场景中，基于 Reflexion 的交易者先生成初始交易理由，识别逻辑缺陷（如忽略行业风险），然后修正决策。

- 论文链接：https://arxiv.org/abs/2303.11366（Reflexion, NeurIPS 2023）

**5. 结构化对抗综合（SAS）**

Amo et al.（FinNLP 2025）提出 SAS 框架，通过确定性多 agent 系统将财报电话会议转录文本转化为高保真投资分析。核心创新是结构化的、有节制的多轮辩论协议。

- 论文链接：https://aclanthology.org/2025.finnlp-2.21.pdf

### 改进理由

单 Critic 只能发现"没有引用的断言"，无法发现"有引用但推理过度延伸"的结论。多面 Critic 矩阵通过分工（引用/数值/推理/合规/覆盖）显著降低漏检率。推理级幻觉检测通过"前提-结论对提取→逻辑链重建→每步验证"填补了文档中最大的质量盲区。

---

## PR5：双重下降意识与 CPCV 变体

### 问题定位

原文档 §10.3 和 §10.6 提到 PBO/DSR 和 CPCV，但未提及组合优化的"双重下降"现象，且未区分 CPCV 的前沿变体。

### 论文依据

**1. 双重下降（Double Descent）**

Lu et al.（arXiv:2411.18830, 2024）首次在均值-方差组合优化中发现"双重下降"现象：以资产数量为模型复杂度代理变量，样本外 Sharpe 比率先升后降（过拟合区），然后在高维区再次上升。机制在于高维模型的理论 Sharpee 接近上限，且参数多于数据约束时过拟合反而减轻。

- 论文链接：https://arxiv.org/abs/2411.18830
- 关键发现：组合优化存在中间过拟合谷，高维区反而受益

**2. CPCV 变体对比**

Arian et al.（Knowledge-Based Systems, 2024）在合成受控环境中系统比较了 OOS 测试方法。CPCV 在降低 PBO 方面**显著优于 Walk-Forward**。研究引入了 Bagged CPCV（集成降低方差）和 Adaptive CPCV（按市场状态动态调整）两个变体。Walk-Forward 在假阳性防护上表现不佳，时间变异性更强、平稳性更弱。

- 论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110
- 期刊：Knowledge-Based Systems, Volume 305, 112477
- 关键数据：CPCV 的 PBO 最低；Bagged CPCV 进一步降低方差

### 改进理由

组合优化的双重下降意味着文档中"资产数越少越稳健"的直觉可能不成立——某些高维组合反而更优。CPCV 相比 Walk-Forward 在假阳性防护上有质的差异，文档应明确推荐 Bagged CPCV 为默认方法，而非仅提到 CPCV 名称。

---

## PR6：时变流动性下的最优执行

### 问题定位

原文档 §10.2 的 Almgren-Chriss 模型假设临时冲击 η 和永久冲击 γ 为常数，ADV 与 σ_daily 在执行窗口内不变。

### 论文依据

Macrì & Lillo（Applied Mathematical Finance, 2024）研究了流动性时变下的最优执行问题。核心发现：

1. 流动性是动态的：盘中 ADV 随时间变化（开盘/午间/尾盘差异巨大）
2. 流动性是隐含的：实时不可直接观测，需从成交数据推断
3. **Double Deep Q-learning (DDQN)** 能在流动性时变场景下学习最优交易策略
4. 在解析解可用时，DDQN 能学到最优策略；在解析解不可用时，DDQN 优于近似解和基准

- 论文链接：https://arxiv.org/abs/2402.12049
- 期刊：Applied Mathematical Finance, Vol. 31(5), pp. 312–342
- DOI：10.1080/1350486X.2025.2490157

### 改进理由

Almgren-Chriss 是 2000 年的经典模型，其恒定流动性假设在 A 股高波动场景下偏差较大。引入 DDQN 作为执行策略补充，在时变流动性场景下可显著降低执行成本。但 RL 策略的稳定性要求高（影子模式 ≥3 个月），因此定位为 AC 基线的补充而非替代。

---

## PR7：GraphRAG 成本优化与增量更新

### 问题定位

原文档 §8.7 的 GraphRAG 采用全量社区摘要预计算，在大语料场景下索引成本极高。

### 论文依据

**1. GraphRAG 原始论文**

Edge et al.（Microsoft Research, arXiv:2404.16130, 2024）提出 GraphRAG：LLM 构建实体-关系图 → Leiden 社区发现 → 每社区生成摘要 → 查询时检索社区摘要。在百万 token 级语料上，GraphRAG 在全面性和多样性上显著优于向量 RAG（胜率 72–83%）。

- 论文链接：https://arxiv.org/abs/2404.16130

**2. LazyGraphRAG（成本优化）**

Microsoft（2024 年 11 月）发布 LazyGraphRAG，将几乎所有 LLM 使用延迟到查询时间，索引成本降至全量 GraphRAG 的约 **0.1%**，质量在局部查询上持平，全局查询略低。

- 参考：https://microsoft.github.io/graphrag/

**3. LightRAG（双层检索 + 增量更新）**

Guo et al.（arXiv:2410.05779, 2024）提出 LightRAG，双层检索系统：低层路径（实体/关系级，窄而精确）+ 高层路径（主题/社区级，宽而概括）。Token 效率约 **10 倍**提升。支持增量更新：新增文档时仅更新相关子图。

- 论文链接：https://arxiv.org/abs/2410.05779

**4. 金融事件演化知识图谱（FEEKG）**

Liu et al.（Expert Systems with Applications, 2024）提出 Financial Event Evolution Knowledge Graph（FEEKG），采用"实体-事件-风险"三层结构，通过事件关联识别关键风险源和风险传导路径。

- 论文链接：https://doi.org/10.1016/j.eswa.2024.123999
- 期刊：Expert Systems with Applications, Vol. 252

**5. 时序金融知识图谱与系统性风险**

Liu et al.（Pacific-Basin Finance Journal, 2025）使用时序金融知识图谱测量系统性风险，发现 FinTech 和证券行业比银行和保险更易受系统性风险影响。

- 论文链接：https://doi.org/10.1016/j.pacfin.2025.102924

### 改进理由

全量 GraphRAG 的索引成本在百万 token 级语料上可能数千次 LLM 调用，对中小机构不现实。LazyGraphRAG 模式将索引成本降至 0.1%，LightRAG 的双层检索 + 增量更新在 token 效率上提升 10 倍。文档应提供三种模式（全量/延迟/混合）的切换选项。金融事件演化图谱的"实体-事件-风险"三层结构可直接增强 §10.4 风险模型的定性分析能力。

---

## PR8：ECI 与多步保形预测

### 问题定位

原文档 §10.6 在"概率校准与不确定性"表格中提到 Conformal Prediction 和 ACI/ECI 名称，但未展开 ECI 的实现细节，且未涉及多步异方差保形预测。

### 论文依据

**1. ECI（Error-quantified Conformal Inference）**

Wu et al.（ICLR 2025）提出 ECI，解决现有在线保形推理方法仅使用二元反馈（覆盖/未覆盖）而忽略误差量化的问题。ECI 通过平滑分位数损失函数引入连续自适应反馈尺度，在任意依赖和分布漂移下建立长期覆盖率保证。实验显示 ECI 能实现有效的覆盖率控制并输出比基线更紧的预测集。

- 论文链接：https://arxiv.org/abs/2502.00818
- 会议：ICLR 2025

**2. 多步异方差保形预测（AEnbMIMOCQR）**

Sousa et al.（Neurocomputing, 2024）提出 AEnbMIMOCQR，模型无关、无需数据切分、在数据非可交换时仍提供接近精确的覆盖率。预测区间不忽略异方差（高波动期自动变宽），对分布漂移鲁棒（无限期内保持有效，无需重训）。

- 论文链接：https://www.sciencedirect.com/science/article/pii/S0925231224012050
- 期刊：Neurocomputing, Volume 608, 128434
- DOI：10.1016/j.neucom.2024.128434
- 代码：https://github.com/Quilograma/AEnbMIMOCQR

### 改进理由

ACI 的二元反馈在剧烈分布漂移下收敛慢。ECI 的连续反馈能更快适应漂移，区间更紧（信息效率更高）。多步保形预测直接服务于文档中 horizon = [1, 5, 20] 的多周期预测需求，且异方差感知确保高波动期区间自动变宽——这对 A 股的高波动特性尤为重要。

---

## PR9：自适应攻击评估与认证防御

### 问题定位

原文档 §14.4 的 AI 安全措施（指令层级隔离、文档包裹、输入检测、工具白名单）在标准攻击下有效，但未评估**自适应攻击**的绕过风险。

### 论文依据

**1. 自适应攻击打破所有防御**

Zhan et al.（NAACL 2025 Findings）评估了八种不同的间接注入防御，并使用自适应攻击**全部绕过**，攻击成功率持续 **>50%**。研究强调在设计防御时需要进行自适应攻击评估。

- 论文链接：https://arxiv.org/abs/2503.00061
- 会议：NAACL 2025 Findings
- 关键数据：8 种防御全部被绕过，成功率 >50%

**2. FATH 认证防御**

Wang et al.（arXiv:2509.02718, 2025）提出 FATH（Formatting AuThentication with Hash-based tags），一种测试时防御策略。与现有方法阻止 LLM 回答额外指令不同，FATH 实施认证系统：要求 LLM 回答所有收到的指令（含注入），通过 hash 标签区分用户指令响应与注入指令响应，最终只输出用户指令的响应。在 Llama3 和 GPT3.5 上实现 SOTA 性能。

- 论文链接：https://arxiv.org/abs/2509.02718
- 代码：https://github.com/Jayfeather1024/FATH

**3. 信息流控制系统级防御**

Wu et al.（arXiv:2409.19091, 2024）提出基于信息流控制原则的系统级防御（f-secure LLM system）。将 LLM 系统拆解为上下文感知管道，安全监控过滤不可信输入到规划过程。提供形式化安全保证。

- 论文链接：https://arxiv.org/abs/2409.19091
- 代码：https://github.com/fzwark/Secure_LLM_System

**4. 间接注入检测与移除**

Anonym.（arXiv:2502.16580, 2025）调查了间接提示注入攻击的检测和移除方法，构建了基准数据集，评估了现有 LLM 和开源检测模型的性能，并训练了专门的检测模型和两种移除方法（分段移除和提取移除）。

- 论文链接：https://arxiv.org/abs/2502.16580

### 改进理由

原文档的防御措施（指令隔离、文档包裹、检测器）在标准攻击下有效，但 Zhan et al. 的研究证明所有这些防御在自适应攻击下均可被绕过（>50% 成功率）。FATH 的认证机制提供了更强的防御——不试图阻止 LLM 执行注入指令，而是通过 hash 标签在输出层过滤。信息流控制提供了系统级的形式化安全保证。文档应将自适应攻击评估纳入 CI，并引入 FATH 作为输出层增强防御。

---

## PR10：新增验收指标与假设清单

### 改进理由

为保证上述 9 项改进可落地、可验收、可追溯，需在 §16 验收标准中新增 12 项可测量条目（B-15~B-22, C-10~C-11, D-12），并在 §18.9 假设清单中新增 7 项待确认假设（H-07~H-13）。

每项验收指标均来自对应论文中的关键发现：
- B-15（推理级幻觉率）← PR4 / PMADS 论文
- B-16（自适应攻击拦截率）← PR9 / Zhan et al. 论文
- B-17（多级重排增益）← PR1 / Multi-Reranker 论文
- B-18（表格语义化召回）← PR1 / FinSage 论文
- B-19（恶意 SQL 分类拦截）← PR3 / Chafik et al. 论文
- B-20（CPCV 对比完整性）← PR5 / Arian et al. 论文
- B-21（ECI 覆盖率偏差）← PR8 / Wu et al. ICLR 2025 论文
- B-22（多步保形区间质量）← PR8 / Sousa et al. 论文
- C-10（双重下降扫描）← PR5 / Lu et al. 论文
- C-11（时变流动性执行）← PR6 / Macrì & Lillo 论文
- D-12（GraphRAG 成本控制）← PR7 / LazyGraphRAG / LightRAG 论文

---

## 论文索引汇总

| 编号 | 论文标题 | 来源 | 年份 | 对应建议 |
| ---- | -------- | ---- | ---- | -------- |
| P01 | Multi-Reranker: Maximizing performance of RAG in FinanceRAG | arXiv:2411.16732 | 2024 | PR1, PR2 |
| P02 | FinSage: A Multi-aspect RAG System for Financial Filings QA | CIKM 2025 | 2025 | PR1 |
| P03 | FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval | EMNLP 2025 Findings | 2025 | PR1 |
| P04 | Rethinking Retrieval: Vector vs Non-Vector RAG in Finance | arXiv:2511.18177 | 2025 | PR1 |
| P05 | ToxicSQL: Are Your LLM-based Text-to-SQL Models Secure? | SIGMOD 2025 | 2025 | PR3 |
| P06 | AIA: Autoregression-Based Injection Attacks Against Text2SQL | AAAI 2025 | 2025 | PR3 |
| P07 | Enhancing security in text-to-SQL systems | Natural Language Processing | 2025 | PR3 |
| P08 | Structured Debate Improves Corporate Credit Reasoning (PMADS) | arXiv 2025 | 2025 | PR4 |
| P09 | Enhancing Financial QA with Multi-Agent Reflection Framework | ICAIF 2024 | 2024 | PR4 |
| P10 | FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis | arXiv:2509.17395 | 2025 | PR4 |
| P11 | Reflexion: Language Agents with Verbal Reinforcement Learning | NeurIPS 2023 | 2023 | PR4 |
| P12 | Structured Adversarial Synthesis (SAS) | FinNLP 2025 | 2025 | PR4 |
| P13 | Double Descent in Portfolio Optimization | arXiv:2411.18830 | 2024 | PR5 |
| P14 | Backtest overfitting: OOS testing methods comparison | Knowledge-Based Systems | 2024 | PR5 |
| P15 | RL for Optimal Execution When Liquidity Is Time-Varying | Applied Mathematical Finance | 2024 | PR6 |
| P16 | From Local to Global: A Graph RAG Approach | arXiv:2404.16130 | 2024 | PR7 |
| P17 | LightRAG: Simple and Fast Graph-RAG | arXiv:2410.05779 | 2024 | PR7 |
| P18 | Financial Event Evolution Knowledge Graph (FEEKG) | Expert Systems with Applications | 2024 | PR7 |
| P19 | Systemic risk measurement based on temporal financial KG | Pacific-Basin Finance Journal | 2025 | PR7 |
| P20 | Error-quantified Conformal Inference for Time Series (ECI) | ICLR 2025 | 2025 | PR8 |
| P21 | Multi-step ahead adaptive conformal heteroscedastic forecasting | Neurocomputing | 2024 | PR8 |
| P22 | Adaptive Attacks Break Defenses Against Indirect Prompt Injection | NAACL 2025 Findings | 2025 | PR9 |
| P23 | FATH: Authentication-based Test-time Defense | arXiv:2509.02718 | 2025 | PR9 |
| P24 | System-Level Defense Against Indirect Prompt Injection | arXiv:2409.19091 | 2024 | PR9 |
| P25 | Can Indirect Prompt Injection Attacks Be Detected and Removed? | arXiv:2502.16580 | 2025 | PR9 |
