# 四个新增数据集的本地冒烟与子集基准（2026-10-06）

用户授权：先完成四份真实 Add/Search、回答、判分冒烟，全部通过后后台串行运行子集。
本轮沿用七数据集评测的本地模型和运行设置，不调用 AML Smoke/Full。
当前状态以 [`status.json`](runs/new-datasets-benchmark-20261006/status.json) 为准。

## 配置与选题

回答/裁判为 `Qwen/Qwen3.5-9B`，embedding 为 `qwen3-embedding-8b`；
HTTP Add/Search、top_k=100、official Add、local profile、邻接半径 0、rerank 关闭，
共同事实取证开启。原始语料与固定评分代码不改动。

当前工作区的产品与评测源码复制到 `var/new-datasets-benchmark-20261006/source/`，
独立配置快照在 `configs/runs/new-datasets-benchmark-20261006/`；
只更换向量集合和运行时路径。既有服务、真源库与结果保留。
向量集合为 `memories_new_datasets_20261006`，独立真源库在该轮 `var/` 目录，
embedding 缓存通过 SQLite backup 作一致快照。模型凭据不写入 manifest 或日志。

| 数据集 | 正式子集 | 冒烟范围 |
| --- | ---: | --- |
| HaluMem-Medium QA | 203 题 / 60 个检查点 | 最早与最晚检查点各一题，检查大小不同的完整历史投喂 |
| MuSiQue-Full dev | 120 题 | 可答与不可答各一题 |
| HybridQA dev | 120 题 | table、passage、other 各一题 |
| FEVEROUS dev challenges | 123 题 | SUPPORTS、REFUTES、NOT ENOUGH INFO 各一题 |

正式子集总计 566 题，配置的唯一声明处是 `eval/experiments/recipes.py`。
冒烟共 10 题，只用于验证链路，不能据其小样本分数估计精度。
每题或检查点仍投喂完整可见语料，题目选择不会裁掉 Add 的语料。
正式运行不复用冒烟答案，但相同 corpus 的 Add 可以通过幂等守卫及 embedding 缓存复用。

准备阶段离线核验源数据、加载冻结样本，记录 qid、类别、实际 Add 批次数与数据指纹。
运行所用不可变 Sample 计划暂存在本轮 `var/plan-*.pickle`，只加载本轮自己生成且哈希固定的计划，
避免每次 smoke/正式运行重复做 FEVEROUS 页面候选检索；它是运行缓存，不是新的数据集。

## 运行与审计

执行脚本：[run_suite.py](runs/new-datasets-benchmark-20261006/run_suite.py)。
每个数据集起一个单 worker 的独立服务，串行驱动 HTTP，然后停止该服务再进入下一份。
冒烟要求题号集合完整、无重复、无 answer/judge 错误；四份未全部通过时不允许正式队列启动。
正式子集中的模型回答错误按原评分记零并单独落审计文件，完整题号覆盖后仍继续队列，
不会通过删除错题或重选题来获得通过。
源码、配置与 Sample 计划哈希在每轮和判分前核对，完整输入变化时不得续用旧答案。
逐题 Search 原文、输入、回答、判分及 HTTP/模型阶段耗时均保留。

原始运行产物：

- `runs/new-datasets-benchmark-20261006/manifest.json`：选题、原始数据与源码/配置/计划指纹。
- `runs/new-datasets-benchmark-20261006/status.json`：队列、每份开始/结束时间与 owned PID。
- `runs/new-datasets-benchmark-20261006/{http,searches}-<phase>-<dataset>.jsonl`：真实请求耗时与检索响应。
- `runs/new-datasets-benchmark-20261006-{smoke,subset}-<dataset>/`：逐题输入、答案、判分。
- `var/new-datasets-benchmark-20261006/run-<phase>-<dataset>.log` 与 `service-*.log`：执行与服务日志。
- `var/new-datasets-benchmark-20261006/snapshot-<phase>.db`：阶段结束的原文库一致快照。

FEVEROUS 仍是 claim-only 候选整页上的本地检索与证据评分；HaluMem 只测 QA。
具体输入/评分范围见 [HaluMem/MuSiQue 接入报告](halumem-musique-pipelines-20261006.md) 和
[HybridQA/FEVEROUS 接入报告](hybridqa-feverous-pipelines-20261006.md)。
各数据集指标不同，不计算跨数据集平均值，不将本地成绩当作 AML 榜分。

## 冒烟、后台启动和最初预计时间

首轮 HaluMem、MuSiQue、HybridQA 冒烟已通过，分别覆盖 2、2、3 题，
墙钟耗时分别为 169.081、20.883、105.633 秒。FEVEROUS 原三题首轮耗时 320.407 秒，
其中两题模型引用提示中的占位 ID 或猜测的不可见元素 ID，严格校验记 `ANSWER_ERROR`，
另一个普通错答记 `WRONG`。该轮失败产物保留，不能将其标为已通过。

修复只发生在本地 FEVEROUS 作答协议：版本升为 `feverous-claim-pages-evidence-v2`，
移除假 ID 示例、要求完整逐字复制可见 ID，并对无效 JSON/标签/ID 最多追加一次模型校正。
校正使用相同 Search 文本和格式错误，无 gold、评分反馈或补检索；两次输出均留档。
评分源码、候选语料、产品代码、题号和 frozen Sample 计划不变。
修改后的评测源码建立快照修订并记录新旧哈希，用 `smoke2-feverous` 新 run-id 复跑原三题，
其余三份已通过的 smoke 保留。

`smoke2-feverous` 原三题全部完成，耗时 451.019 秒，其中 Aramais 的一题仍有
`header_cell` 被写成 `cell` 的不可见 ID；大上下文题在一次作答子进程失败后重试完成。
该轮单次模型请求限额为 180 秒，但重试成功时未保留失败 stderr，不能确认失败是超时。
该修订仍不算通过。源码及 manifest 再归档后，建立 `feverous-claim-pages-evidence-v3`：
在一次校正中提供从同一 Search 文本收集的少量相近合法 ID，包含
`header_cell`/`cell` 的可见拼写变体；相近 ID 不是相关性证明，不自动替换预测证据。
单次本地 FEVEROUS 作答超时扩为 360 秒，其余管线沿用原限额。
新 run-id 为 `smoke3-feverous`，仍使用原三题与相同完整候选语料。
两次修订记录在 `revision-smoke{2,3}.json`，旧 manifest 与源码均保留。

本轮校正逻辑相关测试 29 项通过，runner/其他数据集回归 166 项通过，`make lint` 通过。

最终 FEVEROUS 原三题复验通过，墙钟 512.484 秒，无 `ANSWER_ERROR`/`JUDGE_ERROR`。
Aramais 题经过一次 ID 校正通过；最大上下文题经历两次作答子进程失败后，
标准 harness 第三次幂等重试成功。已确认三轮该题的 Search 文本逐字相同，
均为 281,961 字符（sha256 `fe4e396f62286b5fa004f0ea85a1114a1f46d697fe86f5101b69ac5f8d21527e`）。
重试成功时 harness 未保留前两次的完整 stderr，因此不把两次失败都断言为超时。

| 数据集 | 最终冒烟题数 | 回答/评分错误 | 最终采用的运行 |
| --- | ---: | ---: | --- |
| HaluMem | 2 | 0 | `new-datasets-benchmark-20261006-smoke-halumem` |
| MuSiQue | 2 | 0 | `new-datasets-benchmark-20261006-smoke-musique` |
| HybridQA | 3 | 0 | `new-datasets-benchmark-20261006-smoke-hybridqa` |
| FEVEROUS | 3 | 0 | `new-datasets-benchmark-20261006-smoke3-feverous` |

冒烟通过表示真实 HTTP、模型调用和判分链路完成，普通 `WRONG` 不属于运行故障。
通过后立即在 **2026-10-06 17:57:48（北京时间）**启动 detached 后台子集队列，
supervisor PID/session/group 均为 **1955592**，首份 HaluMem 已实际完成 Add 与 Search，
开始新的正式作答；正式子集答案不复用 smoke。
启动记录见 [`background-launch.json`](runs/new-datasets-benchmark-20261006/background-launch.json)。

启动时预计全部完成需 **6–10 小时**，即 **10 月 7 日约 00:00–04:00（北京时间）**。
估算用冷启动 smoke 的实际 Add 批次耗时，以及最终通过 smoke 的 Search/回答/评分耗时。
均匀按题数外推约 8.83 小时；FEVEROUS 冒烟中稀少 NEI 长上下文题占比更高，
按正式 label 数量加权后外推约 6 小时，故保留该范围及网关重试余量。
embedding 缓存、Add 幂等复用可能加快，网关持续重试可能拉长。
完整估算明细见 [`estimate.json`](runs/new-datasets-benchmark-20261006/estimate.json)。

正式队列后台串行执行 HaluMem → MuSiQue → HybridQA → FEVEROUS。
当前进度见 `status.json`；总调度日志在 `var/new-datasets-benchmark-20261006/background.log`，
逐检查点/题目进度在各 `run-subset-<dataset>.log` 和逐题答案/判分文件。
正式子集已完成，最终成绩见下文；冒烟仍仅用于链路验证。

## 正式子集的上下文超限与续跑（20:24 更新）

HaluMem 203 题、MuSiQue 120 题、HybridQA 120 题已完成。
FEVEROUS 前 4 题已有判分，第 5 题 `feverous-4053` 在原作答子进程的三次尝试后仍返回 HTTP 400，
队列于 19:57:23（北京时间）中断。原状态和日志保留。

对同一冻结提示的诊断请求确认了网关错误正文：Qwen 的上下文上限为 131,072 token，
提示至少 130,049 输入 token，再请求 1,024 输出 token，合计至少 131,073，因此被拒绝。
该完整提示用平台 o200k_base 计数是 117,672 token；这反映本地模型分词器与平台口径的差异。
原始错误响应、提示长度和指纹见
[`gateway-error-feverous-4053.json`](runs/new-datasets-benchmark-20261006/gateway-error-feverous-4053.json)。

续跑使用 [resume_terminal_errors.py](runs/new-datasets-benchmark-20261006/resume_terminal_errors.py)。
产品、loader、原始计划、Search 响应、完整答案提示、模型、输出额度和评分代码保持冻结。
原生作答子进程完成既有有界重试后仍返回 HTTP 400 的题，记录传输错误和提示指纹，
写入显式失败答案；未修改的裁判将其判为 `ANSWER_ERROR`，按零分保留原分母。
因此剩余题继续运行，同时错误原因保留在 `terminal-answer-errors.jsonl`，不删除失败题或裁上下文。
非该类失败仍终止，不吞掉未知错误；旧的成功答案继续按输入指纹校验并复用。

首次旁路仅替换了模块内函数，未接入 `run.py` 使用的包导出引用，同一 400 仍使该阶段失败。
该阶段保留在 status 历史中，第一版脚本保存在本轮 `var/revisions/`。
修正导出路径后，通过实际 `run.py` 导出入口验证：输入指纹不变、400 记 `ANSWER_ERROR`、
其他错误继续抛出。已确认实际 400 响应且提示指纹完全一致的题复用已确认的错误记录，
不再重请求该超限提示；重复续跑不会重复记该题。Ruff 检查通过。
策略和代码哈希见 `continuation-manifest.json`，旧 manifest/launch 以哈希后缀另存。
2026-10-06 **20:29:54**（北京时间）启动最新 detached supervisor **2017274**。
启动信息见 `continuation-launch.json`，调度日志在 `var/new-datasets-benchmark-20261006/continuation-background.log`。
原失败阶段仍在 status 历史中；最新阶段单独追加。最终汇总必须附作答/裁判错误数。

## 正式子集最终成绩（21:47 完成）

队列于 **2026-10-06 21:47:38（北京时间）**更新为 `subset_complete`，
FEVEROUS 最后阶段 exit code 为 0。四份输入、答案与判分题号集合均与冻结 manifest 一致，
无重复或缺题，正式子集共 **566 题**。从 17:57:48 启动到完成实际约 **3 小时 50 分钟**。

| 数据集 | 正确数 / 题数 | 主分 overall | 回答错误 | 裁判错误 |
| --- | ---: | ---: | ---: | ---: |
| HaluMem-Medium QA | 150/203 | 73.89% | 0 | 0 |
| MuSiQue-Full dev | 55/120 | 45.83% | 0 | 0 |
| HybridQA dev | 73/120 | 60.83% | 0 | 0 |
| FEVEROUS dev challenges | 37/123 | 30.08% | 11 | 0 |

HaluMem 另有 Hallucination 36 题、Omission 17 题。
MuSiQue 可答题 12/60（20.00%），不可答题 43/60（71.67%），两组均计入主分。
HybridQA token-F1 为 69.47%；FEVEROUS 标签精度 65.04%，证据 precision 48.25%、
recall 33.33%、调和 F1 39.43%。FEVEROUS 的 11 道 `ANSWER_ERROR` 均按零分保留在 123 题内。

四份 run record 为 `runs/new-datasets-benchmark-20261006-subset-{halumem,musique,hybridqa,feverous}.json`。
完整十一数据集成绩、历史参照与指标解释见 [汇总报告](eleven-datasets-scores-20261006.md)。
