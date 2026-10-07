# v1.0 / v1.1 / v1.2 十一数据集对照（2026-10-07）

本表按用户指定的三个功能阶段命名：v1.0 为加入共同事实取证前的历史参照，
v1.1 为事实抽取/共同取证后的上一轮十一份成绩，v1.2 为本轮冻结 A13 候选。
这些是阶段成绩，未按三个 Git tag 重新同轮跑分；v1.2 尚未通过完整回归验收，也未发布标签。
相关代码按用户要求分类归档在 `candidate/v1.2-grounded-evidence` 分支，
提交分类见[优化记录](optimization-12h-20261007.md)；候选归档不代表发布验收。

范围为同一批本地 native pipeline 冻结子集，共 3,001 题；主分为各数据集的二值 overall。
每题固定评分字段和 v1.1 的题号集合逐一核对相同；错误按零分保留。不是 AML Full 榜分，
各数据集指标不同，不计算跨数据集综合分。— 表示尚无完整历史成绩。

| 数据集 | 冻结题数 | v1.0 历史 | v1.1 事实抽取 | v1.2 当前候选 | v1.2 − v1.1（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorporateBench | 250 | 26.40% | 49.20% | 49.60% | +0.40 |
| MQuAKE-Remastered | 768 | 54.04% | 71.35% | 71.61% | +0.26 |
| TempReason | 332 | 79.52% | 94.58% | 94.88% | +0.30 |
| MemTrapBench | 250 | 42.40% | 44.00% | 44.40% | +0.40 |
| LoCoMo-Refined | 346 | 76.59% | 77.46% | 78.32% | +0.87 |
| LongMemEval-S | 101 | 53.47% | 52.48% | 52.48% | +0.00 |
| MedMemoryBench zh | 388 | — | 44.33% | 43.81% | -0.52 |
| HaluMem QA | 203 | — | 73.89% | 73.40% | -0.49 |
| MuSiQue | 120 | — | 45.83% | 44.17% | -1.67 |
| HybridQA | 120 | — | 60.83% | 60.83% | +0.00 |
| FEVEROUS（严格分） | 123 | — | 30.08% | 未完整：39/121（32.23%）* | — |

*FEVEROUS 32.23% 仅为已判 121 题的精度，不能直接和 123 题的完整历史成绩比较。
已知正确数为 39；以冻结总分母 123 计算的当前下界为 31.71%，两道缺失题全部正确时
上界为 33.33%，这一区间不是已完成的 overall。另有 8 道 ANSWER_ERROR 已计入零分；
两道 HTTP 400 上下文超限题未生成判分。用户选择不调整服务上限，保留缺失，不补造答案。

v1.2 共 2,999/3,001 题有判分记录，其中 MemTrapBench 有 3 道 JUDGE_ERROR，
FEVEROUS 有 8 道 ANSWER_ERROR。完整回归未通过，不能将当前候选写成已正式接受的改进。

历史 v1.0 的 CorporateBench 提示词不同，26.40% → 49.20% 不能全归因于事实抽取；
相同提示词的 v1.1 同轮开关对照为 44.00% → 49.20%。v1.2 的 QA 均分另为 50.49%，
v1.1 为 51.40%，因此二值精度小幅提高不等于所有指标都提高。

v1.2 后五份使用 memory2/memory3 两个独立 Qwen 服务，逻辑模型名相同，部署输出不保证
完全一致；前六份使用 memory2。多数主分仅差 1–3 题，历史比较包含生成/裁判波动，
不能将全部差值归因于代码。具体输入变化与同输入翻转见逐题审计。

AML Full 目前只有用户提供的 v1.0 历史综合分 **46.61**；v1.1、v1.2 未跑对应官方 Full，
不得从本表推算平台综合分或九维分数。

来源：[上一轮十一份汇总](eleven-datasets-scores-20261006.md)、
[优化与失败尝试记录](optimization-12h-20261007.md)、
[本轮逐题审计](runs/optimization-12h-20261007/audit-full-a13.json)、
[完成与缺失核对](runs/optimization-12h-20261007/completion-a13.json)。

| 数据集 | v1.0 历史 run | v1.1 run | v1.2 run |
| --- | --- | --- | --- |
| CorporateBench | [base-cb-rescore-20261004](runs/base-cb-rescore-20261004.json) | [facts-benchmark-20261005-on-corporatebench](runs/facts-benchmark-20261005-on-corporatebench.json) | [optimization-12h-20261007-a13-full-corporatebench](runs/optimization-12h-20261007-a13-full-corporatebench.json) |
| MQuAKE-Remastered | [mqk-fixtest](runs/mqk-fixtest.json) | [facts-benchmark-20261005-on-mquake-remastered](runs/facts-benchmark-20261005-on-mquake-remastered.json) | [optimization-12h-20261007-a13-full-mquake-remastered](runs/optimization-12h-20261007-a13-full-mquake-remastered.json) |
| TempReason | [tr-promptfix](runs/tr-promptfix.json) | [facts-benchmark-20261005-on-tempreason](runs/facts-benchmark-20261005-on-tempreason.json) | [optimization-12h-20261007-a13-full-tempreason](runs/optimization-12h-20261007-a13-full-tempreason.json) |
| MemTrapBench | [mtb-fixtest](runs/mtb-fixtest.json) | [facts-benchmark-20261005-on-memtrapbench](runs/facts-benchmark-20261005-on-memtrapbench.json) | [optimization-12h-20261007-a13-full-memtrapbench](runs/optimization-12h-20261007-a13-full-memtrapbench.json) |
| LoCoMo-Refined | [base-locomo0](runs/base-locomo0.json) | [facts-benchmark-20261005-on-locomo-refined](runs/facts-benchmark-20261005-on-locomo-refined.json) | [optimization-12h-20261007-a13-full-locomo-refined](runs/optimization-12h-20261007-a13-full-locomo-refined.json) |
| LongMemEval-S | [base-lme2](runs/base-lme2.json) | [facts-benchmark-20261005-on-longmemeval-s](runs/facts-benchmark-20261005-on-longmemeval-s.json) | [optimization-12h-20261007-a13-full-longmemeval-s](runs/optimization-12h-20261007-a13-full-longmemeval-s.json) |
| MedMemoryBench zh | — | [facts-benchmark-20261005-on-medmemorybench](runs/facts-benchmark-20261005-on-medmemorybench.json) | [optimization-12h-20261007-a13-full-medmemorybench](runs/optimization-12h-20261007-a13-full-medmemorybench.json) |
| HaluMem QA | — | [new-datasets-benchmark-20261006-subset-halumem](runs/new-datasets-benchmark-20261006-subset-halumem.json) | [optimization-12h-20261007-a13-full-halumem](runs/optimization-12h-20261007-a13-full-halumem.json) |
| MuSiQue | — | [new-datasets-benchmark-20261006-subset-musique](runs/new-datasets-benchmark-20261006-subset-musique.json) | [optimization-12h-20261007-a13-full-musique](runs/optimization-12h-20261007-a13-full-musique.json) |
| HybridQA | — | [new-datasets-benchmark-20261006-subset-hybridqa](runs/new-datasets-benchmark-20261006-subset-hybridqa.json) | [optimization-12h-20261007-a13-full-hybridqa](runs/optimization-12h-20261007-a13-full-hybridqa.json) |
| FEVEROUS（严格分） | — | [new-datasets-benchmark-20261006-subset-feverous](runs/new-datasets-benchmark-20261006-subset-feverous.json) | [optimization-12h-20261007-a13-full-feverous 原始目录](runs/optimization-12h-20261007-a13-full-feverous/) |
