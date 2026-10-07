# 十一数据集低分子任务与改进次序（2026-10-07）

按最新已跑的 A13 / v1.2 快照排序，并保留上一轮 v1.1 同题成绩。这是既有结果分析，没有重新调用 Add/Search、答案或裁判模型。当前工作分支为 main，v1.2 已发布；表中成绩来自发布前冻结的 A13 运行，不是当前 main 或 HEAD 的重新运行结果。

后续已完成临床多跳的本地作答格式修复与冻结子集重答，详见
[临床多跳推进记录](clinical-multihop-20261007.md)。以下排序保留原 A13 口径，
不将单项 QA 修复结果混入其他未重跑的版本成绩。

范围是十一份 native pipeline 的冻结子集：3,001 题中 2,999 题已判。十份完整，FEVEROUS 缺两题；已有 ANSWER_ERROR / JUDGE_ERROR 仍按零分计入已判分母。只用二值完全正确率排序，不将 F1、部分分或平均 rubric 分混入。各评分器的严格程度不同，排序供仓内定位短板使用，不推算 AML Full 或官方维度分。

已核对冻结 manifest、逐题输入和标签：题号无重复，评分字段与上一轮相同，完整 run 的正确数和题数均与 run record 一致。LongMemEval / LoCoMo 的原生分类使用冻结 run record，已核对分类计数合计；其余新增细分由逐题输入标签和 labels 重新聚合。

## 数据集覆盖与总体成绩

| 数据集 | v1.2 已判完全正确率 | v1.1 同题完全正确率 | 状态 |
| --- | ---: | ---: | --- |
| FEVEROUS | 32.23%（39/121） | 30.08%（37/123） | 仅 121/123 题有判分；ANSWER_ERROR 8 |
| MedMemoryBench | 43.81%（170/388） | 44.33%（172/388） | 完整 |
| MuSiQue | 44.17%（53/120） | 45.83%（55/120） | 完整 |
| MemTrapBench | 44.40%（111/250） | 44.00%（110/250） | 完整；JUDGE_ERROR 3 |
| CorporateBench | 49.60%（124/250） | 49.20%（123/250） | 完整 |
| LongMemEval | 52.48%（53/101） | 52.48%（53/101） | 完整 |
| HybridQA | 60.83%（73/120） | 60.83%（73/120） | 完整 |
| MQuAKE | 71.61%（550/768） | 71.35%（548/768） | 完整 |
| HaluMem | 73.40%（149/203） | 73.89%（150/203） | 完整 |
| LoCoMo | 78.32%（271/346） | 77.46%（268/346） | 完整 |
| TempReason | 94.88%（315/332） | 94.58%（314/332） | 完整 |

FEVEROUS 的 39/121 = 32.23% 是已判题精度；冻结 123 题的最终完全正确率尚未形成。按总分母的范围为 31.71%–33.33%。缺失题为 feverous-4053、feverous-31905，分别属于 SUPPORTS/Other、REFUTES/Other；以下多跳、数值、表格与正文、NEI 等组均无缺题。

## 子任务按分数升序

共 55 项。CorporateBench 按原有 answer_type 细分，MemTrapBench 按六个源场景文件细分，MuSiQue 同时保留跳数与可答性，FEVEROUS 按 challenge 跨标签聚合；额外列出 NEI 标签组，它与 challenge 行重叠，不能把这些行相加。其余保留 run record 的原生题类。小于 10 题的组仅作为错误案例线索。

| 数据集 | 子任务 | v1.2 完全正确率 | v1.1 同题 | 主要考察能力 | 状态 |
| --- | --- | ---: | ---: | --- | --- |
| FEVEROUS | 信息不足（NEI） | 0.00%（0/10） | 0.00%（0/10） | 识别无法核实的 claim；本地严格分还要求相应完整证据组 | 完整 |
| MedMemoryBench | 临床多跳推演 | 0.00%（0/39） | 0.00%（0/39） | 结合多次问诊的患者特定病史、用药、数值和事件，形成完整因果链 | 完整 |
| MemTrapBench | 数字游戏（number_game） | 4.26%（2/47） | 4.26%（2/47） | 摆脱历史无解结论和解题定势，按当前题目重新推理 | 完整 |
| FEVEROUS | 多跳证据 | 5.00%（1/20） | 5.00%（1/20） | 跨句子、页面或表格组合证据；标签正确且完整证据组命中 | 完整 |
| FEVEROUS | 数值证据推理 | 14.29%（2/14） | 7.14%（1/14） | 比较、计算或解释数值与单位，同时返回完整证据组 | 完整 |
| LongMemEval | 单会话偏好应用 | 16.67%（1/6） | 16.67%（1/6） | 把已知用户偏好用于新建议或选择，而不只复述偏好 | 小样本 |
| MuSiQue | 可答：2 跳 | 19.35%（6/31） | 19.35%（6/31） | 在候选段落间连接两步实体/关系证据，得出短答案 | 完整 |
| MuSiQue | 可答：4 跳 | 20.00%（2/10） | 20.00%（2/10） | 维持更长的证据链，避免只找到首尾段落 | 完整 |
| MuSiQue | 可答：3 跳 | 21.05%（4/19） | 21.05%（4/19） | 找齐并连接三步证据，防止中间实体或关系断链 | 完整 |
| LongMemEval | 长历史时间推理 | 23.08%（6/26） | 19.23%（5/26） | 跨会话处理事件先后、时距、时间范围和时间约束 | 完整 |
| FEVEROUS | 表格与正文联合取证 | 23.53%（4/17） | 29.41%（5/17） | 联结表格单元格和正文证据，保留可引用的元素 ID | 完整 |
| MedMemoryBench | 医学多选 | 25.32%（20/79） | 25.32%（20/79） | 根据个人病史选齐正确选项，避免漏选或多选 | 完整 |
| HaluMem | 泛化与应用 | 25.71%（9/35） | 34.29%（12/35） | 把兴趣、偏好和经历用于新情境，控制推断边界 | 完整 |
| CorporateBench | 数量/计数（int） | 32.18%（28/87） | 32.18%（28/87） | 根据条件找齐事实、去重，再得到正确数量 | 完整 |
| FEVEROUS | 实体消歧 | 33.33%（1/3） | 33.33%（1/3） | 区分同名实体或相近标题，找到对应事实及证据 | 小样本 |
| FEVEROUS | 隐式检索词 | 33.33%（1/3） | 0.00%（0/3） | 处理 claim 中未直接出现的证据检索词或桥接实体 | 小样本 |
| MedMemoryBench | 个性化推断生成 | 35.44%（28/79） | 34.18%（27/79） | 利用该患者的历史信息进行推断，避免只给通用结论 | 完整 |
| HaluMem | 多跳推断 | 40.00%（2/5） | 60.00%（3/5） | 组合多条人物关系、经历与目标信息 | 小样本 |
| LongMemEval | 跨会话整合 | 40.00%（10/25） | 44.00%（11/25） | 找齐分散在多个会话中的证据，再关联、枚举或计数 | 完整 |
| MemTrapBench | 任务/解释框架切换（hallucination） | 41.67%（15/36） | 44.44%（16/36） | 依据当前材料解释，避免旧领域框架和臆造内容污染答案 | 完整 |
| MemTrapBench | 任务与格式边界（unclear） | 42.55%（20/47） | 42.55%（20/47） | 遵从当前任务和输出格式，避免旧系统/项目规则越界 | 完整 |
| FEVEROUS | 其他挑战 | 46.88%（30/64） | 43.94%（29/66） | 其余事实核查；仍需标签与完整证据组 | 缺 2 题，已判分母 |
| CorporateBench | 完整名单（List[str]） | 49.18%（60/122） | 48.36%（59/122） | 完整枚举实体或事件；名单漏项、多项都影响完全正确率 | 完整 |
| MemTrapBench | 安全与毒记忆（poison） | 56.25%（27/48） | 52.08%（25/48） | 识别伪标准、错误历史断言，避免将其当成可信事实 | 完整 |
| MQuAKE | CF9k | 56.77%（109/192） | 56.25%（108/192） | 反事实知识更新后，采用新事实并完成多跳关系合成 | 完整 |
| LoCoMo | 多跳（category 1） | 57.14%（32/56） | 57.14%（32/56） | 跨对话轮次和会话拼接关系、汇总事实 | 完整 |
| MQuAKE | CF6334 | 57.29%（110/192） | 56.25%（108/192） | 反事实知识更新后，采用新事实并完成多跳关系合成 | 完整 |
| HybridQA | 答案在表格 | 59.57%（28/47） | 61.70%（29/47） | 保持行、列、标题对应关系，在表格中定位答案 | 完整 |
| LongMemEval | 拒答 | 60.00%（3/5） | 60.00%（3/5） | 证据不足时不编造，不把相似历史当成答案 | 小样本 |
| HybridQA | 答案在关联段落 | 61.43%（43/70） | 60.00%（42/70） | 从表格中的实体连接到关联正文，跨表格与段落取证 | 完整 |
| MedMemoryBench | 医学实体回忆 | 62.50%（50/80） | 62.50%（50/80） | 完整回忆明确提及的病名、药名等实体 | 完整 |
| MuSiQue | 不可答：3 跳 | 63.16%（12/19） | 73.68%（14/19） | 识别三跳链条缺失，避免用常识补齐 | 完整 |
| MedMemoryBench | 医学事件时间定位 | 63.89%（46/72） | 66.67%（48/72） | 把症状、检查和治疗事件定位到正确日期或时段 | 完整 |
| MemTrapBench | 情绪/创伤干扰（hurt） | 63.89%（23/36） | 63.89%（23/36） | 避免过去批评和负反馈压制当前正确方法或回答 | 完整 |
| HybridQA | 其他/未分类 | 66.67%（2/3） | 66.67%（2/3） | 表格和关联正文的综合问答；该组只有少量样本 | 小样本 |
| MedMemoryBench | 当前状态更新 | 66.67%（26/39） | 69.23%（27/39） | 追踪截至提问检查点的用药、症状或治疗状态 | 完整 |
| MemTrapBench | 流程惯性（Inertia） | 66.67%（24/36） | 66.67%（24/36） | 重新判断适用方法，避免沿用历史冗长或不适合的流程 | 完整 |
| MuSiQue | 不可答：4 跳 | 70.00%（7/10） | 70.00%（7/10） | 识别四跳链条缺失，避免猜测终点答案 | 完整 |
| MuSiQue | 不可答：2 跳 | 70.97%（22/31） | 70.97%（22/31） | 识别两跳链条缺失，按本地协议拒答 | 完整 |
| CorporateBench | 存在性判断（bool） | 75.00%（15/20） | 75.00%（15/20） | 判断满足给定条件的事件或关系是否存在 | 完整 |
| HaluMem | 记忆冲突 | 75.00%（45/60） | 73.33%（44/60） | 区分冲突陈述的来源、时效和当前有效内容 | 完整 |
| MQuAKE | CF3k | 75.00%（144/192） | 74.48%（143/192） | 反事实知识更新后，采用新事实并完成多跳关系合成 | 完整 |
| LongMemEval | 知识更新 | 80.00%（12/15） | 80.00%（12/15） | 区分旧值与新值，采用问题所需时点的有效事实 | 完整 |
| LoCoMo | 单跳（category 4） | 81.35%（157/193） | 80.31%（155/193） | 从一条或少量直接证据中找出明确事实 | 完整 |
| LongMemEval | 助手侧直接事实 | 81.82%（9/11） | 81.82%（9/11） | 回忆助手在单次会话中的回答和建议 | 完整 |
| LoCoMo | 开放域（category 3） | 83.33%（10/12） | 75.00%（9/12） | 把对话事实与相关背景联系起来回答 | 完整 |
| LoCoMo | 时间（category 2） | 84.71%（72/85） | 84.71%（72/85） | 将事件与会话时间锚点关联，保留合适的时间粒度 | 完整 |
| HaluMem | 基础事实回忆 | 84.78%（39/46） | 80.43%（37/46） | 准确回忆人物明确说过的事实 | 完整 |
| LongMemEval | 用户侧直接事实 | 92.31%（12/13） | 92.31%（12/13） | 找出用户在单次会话中明确说过的信息 | 完整 |
| TempReason | L2：时点定位 | 94.59%（175/185） | 94.59%（175/185） | 把问题日期落入正确的事实有效区间，选中当时的实体或属性 | 完整 |
| HaluMem | 记忆边界 | 94.64%（53/56） | 94.64%（53/56） | 认识哪些内容记忆中没有，避免越界编造 | 完整 |
| TempReason | L3：前后顺序 | 95.24%（140/147） | 94.56%（139/147） | 根据 before/after 等关系定位相邻事件或时段 | 完整 |
| MQuAKE | T | 97.40%（187/192） | 98.44%（189/192） | 现实事实随时间更新后的多跳知识问答 | 完整 |
| CorporateBench | 单值事实（str） | 100.00%（21/21） | 100.00%（21/21） | 准确抽取一个名称或属性值 | 完整 |
| HaluMem | 动态更新 | 100.00%（1/1） | 100.00%（1/1） | 新陈述出现后正确更新对人物事实或偏好的理解 | 小样本 |

## 被细分掩盖的原生父类

| 原生父类 | v1.2 完全正确率 | 说明 |
| --- | ---: | --- |
| CorporateBench / kb_qa | 49.60%（124/250） | 本轮只覆盖 kb_qa；topic_qa、integrated_qa 未测 |
| MemTrapBench / cognitive_bias | 31.33%（26/83） | number_game 2/47 与 Inertia 24/36 合并 |
| MemTrapBench / task_boundary | 42.17%（35/83） | hallucination 15/36 与 unclear 20/47 合并 |
| MuSiQue / 可答题合计 | 20.00%（12/60） | 三种跳数各组都低，不是仅四跳困难 |
| MuSiQue / abstention | 68.33%（41/60） | 拒答较好将总体分抬至 44.17% |

## FEVEROUS 原始标签 / challenge 明细

原始交叉分类中很多组只有 1–5 题；零分仍列出，但改进顺序优先参考上面跨标签聚合。

| 原始类别 | v1.2 严格分 | v1.1 严格分 | 状态 |
| --- | ---: | ---: | --- |
| NOT ENOUGH INFO/Combining Tables and Text | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| NOT ENOUGH INFO/Entity Disambiguation | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| NOT ENOUGH INFO/Multi-hop Reasoning | 0.00%（0/2） | 0.00%（0/2） | 小样本 |
| NOT ENOUGH INFO/Numerical Reasoning | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| NOT ENOUGH INFO/Other | 0.00%（0/4） | 0.00%（0/4） | 小样本 |
| NOT ENOUGH INFO/Search terms not in claim | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| REFUTES/Combining Tables and Text | 0.00%（0/3） | 0.00%（0/3） | 小样本 |
| REFUTES/Multi-hop Reasoning | 0.00%（0/3） | 0.00%（0/3） | 小样本 |
| SUPPORTS/Entity Disambiguation | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| SUPPORTS/Numerical Reasoning | 0.00%（0/5） | 0.00%（0/5） | 小样本 |
| SUPPORTS/Search terms not in claim | 0.00%（0/1） | 0.00%（0/1） | 小样本 |
| SUPPORTS/Multi-hop Reasoning | 6.67%（1/15） | 6.67%（1/15） | 完整 |
| REFUTES/Numerical Reasoning | 25.00%（2/8） | 12.50%（1/8） | 小样本 |
| SUPPORTS/Combining Tables and Text | 30.77%（4/13） | 38.46%（5/13） | 完整 |
| REFUTES/Other | 41.67%（15/36） | 37.84%（14/37） | 缺 1 题，已判分母 |
| SUPPORTS/Other | 62.50%（15/24） | 60.00%（15/25） | 缺 1 题，已判分母 |
| REFUTES/Entity Disambiguation | 100.00%（1/1） | 100.00%（1/1） | 小样本 |
| REFUTES/Search terms not in claim | 100.00%（1/1） | 0.00%（0/1） | 小样本 |

## 已观察到的错误现象与建议

1. **MedMemoryBench 临床多跳：先核查本地作答口径，再定位证据链。**39 道全为 multi_hop_clinical_deduction:WRONG，零 JUDGE_ERROR。本地通用 ANSWER_PROMPT 要求只用记忆、用一句话简短作答，而上游多跳裁判要求覆盖患者特定的多节点信息和因果链。抽查三题中，答案给出泛化结论或极短解释，裁判指出缺少具体病史、药物、日期与数值节点。这是已观察到的口径差异和输出缺项，尚不能证明单纯由检索或答案模型哪一方造成。下一步应核对每个所需节点是否已进入 Search 文本，再做固定案例的完整证据对照。

2. **MemTrapBench 数字游戏：检查旧记忆是否把本轮推理锁死。**2/47 比 cognitive_bias 父类的 26/83 更值得先定位。抽查出现与 expected_failure_output 一致的无解回答；任务/格式边界题也出现当前要求 CSV，回答却沿用历史 XML 规则并拒绝转换。建议固定相同题目比较当前检索与无记忆/去除不相关历史的诊断对照，区分记忆污染与作答模型本身的推理能力。

3. **FEVEROUS 与 MuSiQue：优先查完整证据链及其可见性。**FEVEROUS 多跳标签正确 10/20，但严格正确只有 1/20；应逐级核查本地候选池、Add/切块、Search 排序/截断、证据 ID 与整组引用。NEI 有两道标签判对却返回空证据，仍未通过此严格评分。MuSiQue 可答题 12/60，其中 22/60 明确输出 INSUFFICIENT_EVIDENCE；需要区分真正缺证据与已给证据但未连接成功。拒答分不能替代可答推理分。

4. **LongMemEval 时间与偏好：分别处理，保留小样本边界。**时间为 6/26，跨会话为 10/25，应检查事件时间、会话日期、相对时间锚点与多证据整合。偏好为 1/6，值得检查从用户偏好到建议的应用，但仅六题，不宜据此断言整体个性化能力。LoCoMo 时间 72/85、TempReason 两类均超过 94%，不能据它们认定长对话时间推理已解决。

5. **HaluMem 泛化与医学多选：检查来源、时效和过度扩展。**HaluMem 泛化从 v1.1 的 12/35 降至 9/35，抽查答案出现额外兴趣或扩展表述。后续应核对有效偏好、用户自述与助手建议的来源，以及输出是否多加未被支持的内容；医学多选则需检查多条患者信息是否齐全，并区分漏选、多选和输出字母解析问题。

6. **CorporateBench：数量题和名单题一起检查穷举覆盖。**计数 28/87、名单 60/122，建议核查相关事实是否找齐、是否去重、时间窗口和关系条件是否全部满足。只提高单条证据相关性不能保证集合完整。

建议顺序：先审查临床多跳的评分/作答口径，再处理数字游戏的记忆污染，随后处理 FEVEROUS / MuSiQue 的完整证据链、LongMemEval 的时间/偏好、HaluMem 的应用边界及 CorporateBench 的穷举聚合。前后只有几题变化的细分分数含模型和裁判波动，不将差值全归因于代码。

服务改进仍以 Add/Search 为边界：提供有来源且完整的证据，不在 Search 中计算或生成最终答案。诊断本地答案/评分口径属于评测工作，不能将本地答案提示词收益直接当成 AML 服务收益。

## 来源

- [三阶段成绩与候选状态](version-comparison-20261007.md)
- [上一轮完整十一份成绩](eleven-datasets-scores-20261006.md)
- [本轮完成与缺失核对](runs/optimization-12h-20261007/completion-a13.json)
- [本轮逐题审计](runs/optimization-12h-20261007/audit-full-a13.json)
- [本报告重聚合摘要](runs/subtask-priorities-20261007/summary.json)
- 分类与能力依据：eval/datasets 各加载器、MemTrapBench / MQuAKE / HaluMem 的本地上游 README，以及 eval/harness/run_record.py、extra_pipeline.py、corpusqa_pipeline.py。

| 数据集 | v1.2 原始目录 | v1.1 run record |
| --- | --- | --- |
| CorporateBench | [optimization-12h-20261007-a13-full-corporatebench](runs/optimization-12h-20261007-a13-full-corporatebench/) | [facts-benchmark-20261005-on-corporatebench](runs/facts-benchmark-20261005-on-corporatebench.json) |
| MQuAKE | [optimization-12h-20261007-a13-full-mquake-remastered](runs/optimization-12h-20261007-a13-full-mquake-remastered/) | [facts-benchmark-20261005-on-mquake-remastered](runs/facts-benchmark-20261005-on-mquake-remastered.json) |
| TempReason | [optimization-12h-20261007-a13-full-tempreason](runs/optimization-12h-20261007-a13-full-tempreason/) | [facts-benchmark-20261005-on-tempreason](runs/facts-benchmark-20261005-on-tempreason.json) |
| MemTrapBench | [optimization-12h-20261007-a13-full-memtrapbench](runs/optimization-12h-20261007-a13-full-memtrapbench/) | [facts-benchmark-20261005-on-memtrapbench](runs/facts-benchmark-20261005-on-memtrapbench.json) |
| LoCoMo | [optimization-12h-20261007-a13-full-locomo-refined](runs/optimization-12h-20261007-a13-full-locomo-refined/) | [facts-benchmark-20261005-on-locomo-refined](runs/facts-benchmark-20261005-on-locomo-refined.json) |
| LongMemEval | [optimization-12h-20261007-a13-full-longmemeval-s](runs/optimization-12h-20261007-a13-full-longmemeval-s/) | [facts-benchmark-20261005-on-longmemeval-s](runs/facts-benchmark-20261005-on-longmemeval-s.json) |
| MedMemoryBench | [optimization-12h-20261007-a13-full-medmemorybench](runs/optimization-12h-20261007-a13-full-medmemorybench/) | [facts-benchmark-20261005-on-medmemorybench](runs/facts-benchmark-20261005-on-medmemorybench.json) |
| HaluMem | [optimization-12h-20261007-a13-full-halumem](runs/optimization-12h-20261007-a13-full-halumem/) | [new-datasets-benchmark-20261006-subset-halumem](runs/new-datasets-benchmark-20261006-subset-halumem.json) |
| MuSiQue | [optimization-12h-20261007-a13-full-musique](runs/optimization-12h-20261007-a13-full-musique/) | [new-datasets-benchmark-20261006-subset-musique](runs/new-datasets-benchmark-20261006-subset-musique.json) |
| HybridQA | [optimization-12h-20261007-a13-full-hybridqa](runs/optimization-12h-20261007-a13-full-hybridqa/) | [new-datasets-benchmark-20261006-subset-hybridqa](runs/new-datasets-benchmark-20261006-subset-hybridqa.json) |
| FEVEROUS | [optimization-12h-20261007-a13-full-feverous](runs/optimization-12h-20261007-a13-full-feverous/) | [new-datasets-benchmark-20261006-subset-feverous](runs/new-datasets-benchmark-20261006-subset-feverous.json) |
