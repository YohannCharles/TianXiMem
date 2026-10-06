# 十一个数据集的完整子集成绩与历史参照（2026-10-06）

截至 2026-10-06 21:47:38（北京时间），新增四数据集队列为 `subset_complete`。
原七数据集与新增四数据集共 **3,001 题**，本次逐份核对冻结 manifest、输入、答案、判分：
题号集合一致，无重复或缺题；正确数与各 run record 的 `scores.overall` 一致。
另有 CorporateBench 当前提示词下的关闭共同取证对照 250 题，单独列出，不计入 3,001 题。

这里的“完整”指本轮固定子集完整评分。范围是本地原始数据集 pipeline，
公开数据集全量、AML 比赛 Full 及新接入的 `aml-v1` 输入口径尚无本轮成绩。

## 本轮主分与最近完整历史

主分采用每个数据集固定裁判的二值 `overall`；历史列采用原七数据集 manifest 中选定的同题集参照。
本次再次核对六份历史输入与本轮的题号集合；共同原始文件指纹的核对见
[七数据集重跑报告](facts-benchmark-20261005.md)。变化单位为百分点，按 run record 原值计算，再保留两位小数。

| 数据集与范围 | 题数 | 本轮正确数 | 历史主分 | 本轮主分 | 变化（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorporateBench | 250 | 123 | 26.40% | 49.20% | +22.80 |
| MQuAKE-Remastered | 768 | 548 | 54.04% | 71.35% | +17.32 |
| TempReason | 332 | 314 | 79.52% | 94.58% | +15.06 |
| MemTrapBench | 250 | 110 | 42.40% | 44.00% | +1.60 |
| LoCoMo-Refined | 346 | 268 | 76.59% | 77.46% | +0.87 |
| LongMemEval-S | 101 | 53 | 53.47% | 52.48% | -0.99 |
| MedMemoryBench zh | 388 | 172 | — | 44.33% | — |
| HaluMem-Medium QA | 203 | 150 | — | 73.89% | — |
| MuSiQue-Full dev | 120 | 55 | — | 45.83% | — |
| HybridQA dev | 120 | 73 | — | 60.83% | — |
| FEVEROUS dev challenges | 123 | 37 | — | 30.08% | — |

MedMemoryBench 的早期 `base-mmb` 三次因网关 524 中断，残留 161 条答案，未形成完整基线。
新增四份是本仓库首次完成的正式子集基线；十道冒烟仅验证链路，不作为历史精度参照。

## 附加指标与评分范围

| 数据集 | 附加指标 | 解释 |
| --- | --- | --- |
| CorporateBench | QA 均分 **51.40%**；历史原答案重判 **28.27%** | 标量 exact match / 列表 set-F1；与主表二值完全正确率分别记录 |
| HaluMem | 正确 150/203；Hallucination **36/203 = 17.73%**；Omission **17/203 = 8.37%** | 仅测试 QA；未包含上游记忆抽取与更新任务的独立指标 |
| MuSiQue | 可答题 **12/60 = 20.00%**；不可答题 **43/60 = 71.67%** | 前者按答案别名匹配，后者按本地拒答协议判分；45.83% 是两组共同计入的主分 |
| HybridQA | EM **60.83%**；token-F1 **69.47%** | 将完整表格与链接段落构造为文本 Add，经 Search 后作答的本地范围 |
| FEVEROUS | 严格分 **30.08%**；标签精度 **65.04%**；证据 precision **48.25%**、recall **33.33%**、F1 **39.43%** | claim-only 候选整页检索；需证据的题同时要求标签正确及完整证据组命中；F1 是证据宏平均 precision/recall 的调和均值 |

FEVEROUS 候选由 claim 的标题/引言检索生成，不使用 gold 选择候选页面。
候选语料与固定评分细节见 [HybridQA/FEVEROUS 接入报告](hybridqa-feverous-pipelines-20261006.md)，
HaluMem/MuSiQue 范围见 [接入报告](halumem-musique-pipelines-20261006.md)。

## CorporateBench 同轮开关对照

历史 CorporateBench 使用较早的通用答案提示词，历史到本轮的 +22.80 个百分点不能全部归因于共同取证。
同轮的开关对照使用当前相同提示词、同一原文库与向量集合：

| 共同取证 | 完全正确数 / 题数 | 二值主分 | 标量 EM / 列表 set-F1 均分 |
| --- | ---: | ---: | ---: |
| 关闭 | 110/250 | 44.00% | 48.39% |
| 开启 | 123/250 | 49.20% | 51.40% |
| 变化（百分点） | — | +5.20 | +3.01 |

关闭对照来源为 [facts-benchmark-20261005-off-corporatebench](runs/facts-benchmark-20261005-off-corporatebench.json)。
其余历史比较还包含独立索引及答案/裁判模型波动，LoCoMo、LongMemEval 等小幅变化需要复测才能定论。

## 错误与完成状态

| 数据集 | 回答错误 | 裁判错误 | 计分方式 |
| --- | ---: | ---: | --- |
| MemTrapBench | 0 | 3 | `JUDGE_ERROR` 按零分计入 250 题 |
| FEVEROUS | 11 | 0 | `ANSWER_ERROR` 按零分计入 123 题 |
| 其余九份 | 0 | 0 | 普通错答按各自裁判记零 |

本轮共 14 道回答/裁判错误，均保留在原分母内。FEVEROUS 的超上下文、JSON 和不可见证据 ID
错误及续跑策略保留在 [新增四数据集报告](new-datasets-benchmark-20261006.md)，原失败阶段没有覆盖。
原七数据集旧 `status.json` 顶层仍为 `running`，其自动核对告警保留；
这里的完成核对依据各 run record、逐题覆盖和 `audit-summary.json`，未改写历史运行状态。

## 来源与复现

本轮回答/裁判模型为 `Qwen/Qwen3.5-9B`，embedding 为 `qwen3-embedding-8b`；
HTTP Add/Search、top_k=100、official Add、local profile、邻接半径 0、rerank 关闭，共同取证开启。
七数据集冻结产品为 `67fb106e77d1dec2e02bdf726623bcb587309193`；新增四份源码快照与后续
FEVEROUS 协议修订固定在自己的 manifest 中。本表没有把这些已存答案标为当前 HEAD 的重新运行结果。

| 数据集 | 本轮 run record | 历史 run record |
| --- | --- | --- |
| CorporateBench | [facts-benchmark-20261005-on-corporatebench](runs/facts-benchmark-20261005-on-corporatebench.json) | [base-cb-rescore-20261004](runs/base-cb-rescore-20261004.json) |
| MQuAKE-Remastered | [facts-benchmark-20261005-on-mquake-remastered](runs/facts-benchmark-20261005-on-mquake-remastered.json) | [mqk-fixtest](runs/mqk-fixtest.json) |
| TempReason | [facts-benchmark-20261005-on-tempreason](runs/facts-benchmark-20261005-on-tempreason.json) | [tr-promptfix](runs/tr-promptfix.json) |
| MemTrapBench | [facts-benchmark-20261005-on-memtrapbench](runs/facts-benchmark-20261005-on-memtrapbench.json) | [mtb-fixtest](runs/mtb-fixtest.json) |
| LoCoMo-Refined | [facts-benchmark-20261005-on-locomo-refined](runs/facts-benchmark-20261005-on-locomo-refined.json) | [base-locomo0](runs/base-locomo0.json) |
| LongMemEval-S | [facts-benchmark-20261005-on-longmemeval-s](runs/facts-benchmark-20261005-on-longmemeval-s.json) | [base-lme2](runs/base-lme2.json) |
| MedMemoryBench zh | [facts-benchmark-20261005-on-medmemorybench](runs/facts-benchmark-20261005-on-medmemorybench.json) | 尚无完整历史 |
| HaluMem | [new-datasets-benchmark-20261006-subset-halumem](runs/new-datasets-benchmark-20261006-subset-halumem.json) | 首轮正式子集 |
| MuSiQue | [new-datasets-benchmark-20261006-subset-musique](runs/new-datasets-benchmark-20261006-subset-musique.json) | 首轮正式子集 |
| HybridQA | [new-datasets-benchmark-20261006-subset-hybridqa](runs/new-datasets-benchmark-20261006-subset-hybridqa.json) | 首轮正式子集 |
| FEVEROUS | [new-datasets-benchmark-20261006-subset-feverous](runs/new-datasets-benchmark-20261006-subset-feverous.json) | 首轮正式子集 |

冻结选题与源码/配置指纹：[七数据集 manifest](runs/facts-benchmark-20261005/manifest.json)、
[四数据集 manifest](runs/new-datasets-benchmark-20261006/manifest.json)。逐题输入、答案、判分在对应 run 目录内。
原始 JSON/JSONL 按仓库约定不入 Git，本报告与台账保存可复核的派生结论。

## 更早的历史记录

以下用于追踪从 2026-10-02 起的口径修复，已作废或语料不同的记录不充当主表的历史参照。

| 数据集 | 较早记录 | 状态与原因 |
| --- | --- | --- |
| CorporateBench | [base-cb](runs/base-cb.json) **26.00%** → 原答案重判 **26.40%** | 修正旧裁判的数值/日期、空集合、列表等处理；未用新提示词重答 |
| MQuAKE | [base-mqk](runs/base-mqk.json) **39.97%** → [mqk-fixtest](runs/mqk-fixtest.json) **54.04%** | 39.97% 已作废：通用答案提示词的拒答条款与禁世界知识不适合任务 |
| MemTrapBench | [base-mtb](runs/base-mtb.json) **14.40%** → [mtb-fixtest](runs/mtb-fixtest.json) **42.40%** | 14.40% 已作废：答案提示词接错、裁判输出截断 |
| TempReason | [base-tr](runs/base-tr.json) **6.63%** → [tr-fixtest](runs/tr-fixtest.json) **55.12%** → [tr-splitfix](runs/tr-splitfix.json) **63.86%** → [tr-promptfix](runs/tr-promptfix.json) **79.52%** | 前两代已作废；依次修复丢失 fact_context、区间切句、通用提示词拒答问题 |
| LongMemEval | [base-lme](runs/base-lme.json) **52.48%** → [base-lme2](runs/base-lme2.json) **53.47%** | 前者使用 D32 前的语料，主表采用 D32 后干净库基线 |
| MedMemoryBench | `base-mmb` 中断，未形成完整分数 | 不以残留部分答案估计历史完整精度 |

历史修复详情见 [台账](ledger.md)。各数据集裁判指标不同，不计算跨数据集平均分；
原生题类与拒答分也不补写为 AML 七维分数，该七维成绩在本轮不可得。
