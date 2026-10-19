# 官方采集的数据集分类与版本对比

> 最后核对：2026-10-19，对应 PRD §2.2、§6.4、§12.3、§13。

入口为 `uv run python -m eval.experiments.official_dataset`，依次提供 `prepare`、
`run`、`summarize`、`compare`。分类统计见
[当前分类报告](../eval/reports/runs/official-dataset-20260929/coverage.md)。

## 一键入口（2026-10-19）

全量 **8,783 条可评分题**（1,361 条无公开金标的不进入分母）用
[`tools/run_official_eval.py`](../tools/run_official_eval.py) 驱动；它只做
**预检 → 调用 `official_dataset.run/summarize/compare`**，重放逻辑仍只有
`official_dataset.py` 一份：

```bash
# 跑前检查：清单/timeline 覆盖、14 个家族的判分分派、裁判依赖、端点、/health
uv run --env-file .env python tools/run_official_eval.py check --base-url http://127.0.0.1:8000

# 正式跑；中断后用完全相同的命令续
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-v1 --service-revision v1 --base-url http://127.0.0.1:8000

# 离线看进度
uv run python tools/run_official_eval.py status --run-id official-v1
```

**不要按数据集重新整理一份 add/search 再拼回去**：采集的 Add 是 AML 原文，user 内
add/search 按真实时间交错（全量 41% 的题在"该 user 最后一条 add"之前就问），
按家族拆开会同时丢掉**跨语料干扰项**与**防未来泄漏**。逐题分派已经在
[`official_capture_pipeline.py`](../eval/harness/official_capture_pipeline.py) 里完成。

长跑韧性（2026-10-19 加）：

| 失败 | 处置 |
| --- | --- |
| `/add` 5xx / 网络层（已重试 3 次） | **跳过该批并记 `skipped.jsonl`**，游标继续——线上那一批同样会丢；4xx 仍响亮失败 |
| `/search` 5xx / 网络层 | `ServiceClient` **有界重试 3 次**（只读，无副作用）；4xx 不重试 |
| 某个用户的 answer/judge 子进程失败 | 记 `errors.jsonl`、**继续下一个用户**；同 run-id 重跑从该用户的检查点续 |
| 某题判成 `JUDGE_ERROR`（网关抖动） | 默认保留并在 `summary.md` 的「技术错误」列可见；加 `--retry-errors` 重跑会先清掉这些行再重判（`CORRECT` / `WRONG` 等真实判定不动） |
| 中断后重跑 | 已完成的 Add/Search 不重发；run record 的 `overall` 从**磁盘上的全部** `labels.jsonl` 重建，不会只统计最后一轮 |

### 两版比较（v1.0 vs 最新）

用**同一份当前 harness** 分别驱动两个服务；只有服务侧版本、配置目录、端口和
`run-id` 不同（`--memory-date` / `--annotate-mark` / `--max-tokens` 与答案/裁判
端点必须一致，否则 `compare` 会拒绝）：

```bash
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-v1.0 --service-revision v1.0 \
  --base-url http://127.0.0.1:8010 --configs-dir /path/to/v1.0/configs --profile local
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-latest --service-revision main \
  --base-url http://127.0.0.1:8020 --configs-dir /path/to/latest/configs --profile local

uv run python tools/run_official_eval.py compare \
  --left-run-id official-v1.0 --right-run-id official-latest \
  --left-name v1.0 --right-name latest
```

⛔ **不要**用 v1.0 的 harness checkout 跑一臂、最新 checkout 跑另一臂：
`compare` 核对 `evaluation` 指纹（答案/裁判模型、端点、预算、评分源码），
源码不同会拒绝比较。产品版本与产品配置可以不同——那正是要比较的东西。

### 金标审计（只读）

```bash
uv run python tools/verify_official_gold.py --sample-per-family 3
```

它**只证明来源与搬运**：8,783 条里每条都能追到 `official-gold-extra*` 或
`public_gold`，且 `gold_answers` / `gold_rubric` / `gold_native` 与来源行逐字段相同；
审计报告按族给出「extra 来源 / public_gold 来源 / 无来源 / 不一致」计数，并抽 N 条
列出 `source_locator` 供人工复核。报告默认写到 `eval/reports/runs/official-gold-audit.md`
（`runs/` 已 gitignore——里面有题目与金标原文，不随仓库分发）。

⛔ **它不证明上游金标语义正确**：公开题库/上游 rubric 可能本身有歧义，LLM 裁判也可能
与人工判断不一致。已声明的口径偏离（PersonaMem 的 0.5 二值化阈值、FEVEROUS 只判
label、BEAM/MemTrapBench 的阈值等）在报告的「已声明口径偏离」一节逐条列出。
⛔ 需要逐条语义确认时，必须回到 `source_locator` 指的上游行人工复核——那不是这个
脚本能替代的。


## 分类与固定题目

```bash
uv run python -m eval.experiments.official_dataset prepare \
  --capture-dir official-dataset-2026-09-29 \
  --report-dir eval/reports/runs/official-dataset-20260929
```

按原始 Search 的 `seq` 与 `official-eval-kit.jsonl` 联接。只采用题级可靠归属
（采集标签、官方回写、公开金标或补金标），未知归属统一为 `other`，不凭语料形态猜测。
同一道题的重复 Search 记录均保留，原始文件不改写。

产物在采集目录的 `grouped/`：`searches/<dataset>.jsonl` 包含该类所有原始 Search 字段、
金标、`source_dataset` 和 `scorable`；`manifest.json` 包含全部源文件 SHA-256、覆盖统计
及固定评分题目的 seq、用户和数据集。没有可靠金标的记录保留供分析，不进入正确率分母。

全量清单默认选中所有可评分记录。先做小规模验证时，可以生成独立清单：

```bash
uv run python -m eval.experiments.official_dataset prepare \
  --output official-dataset-2026-09-29/grouped-probe \
  --report-dir eval/reports/runs/official-dataset-probe \
  --per-dataset 2
```

`--per-dataset` 是每个数据集的题数上限，按用户确定性轮转取题。
可用 `--datasets mquake-remastered halumem` 限定评分家族；分类导出仍保留所有记录。
已有分类目录不覆盖；两次版本运行必须复用同一份清单。

## 跑一个版本

先启动要测的服务，并确认服务的 Git SHA/标签和实际配置目录。答案/裁判模型由
`.env` 的现有 `AML_*` / `JUDGE_*` 配置控制，参见
[harness 说明](../eval/harness/CLAUDE.md)。`--service-revision` 是调用者声明，
pipeline 会保存它，但无法通过 HTTP 验证服务的真实 Git 版本。

```bash
uv run --env-file .env python -m eval.experiments.official_dataset run \
  --manifest official-dataset-2026-09-29/grouped/manifest.json \
  --run-id official-v1.0 \
  --service-revision v1.0 \
  --base-url http://127.0.0.1:8000 \
  --configs-dir /path/to/v1.0/configs \
  --profile local
```

使用新 run-id 跑另一版本，例如 `official-v1.1.1`；指定另一服务的端口、版本与配置目录。
两版依次运行，保持答案/裁判模型、输出预算、评分源码和固定清单一致。
产品版本与产品配置可以不同，配置字节会保存到运行目录。
`--offline` 禁止下载裁判依赖；缺失依赖或端点配置在发 Add/Search 前报错。

语料按原始时间线交错重放，正文、会话和批界保留；只改服务侧用户 ID 为本次 run 的
隔离命名空间，防止版本之间串用记忆。所选最后一道题之后的 Add 不再发送。
沿用 Search 的 top_k 契约及平台上下文前缀预算（PRD §2.2、§6.4）。

执行复用逐事件检查点：历史 Search 快照落盘后才继续 Add。**服务端/网络层的 Add 失败**
按 `--on-add-error` 处理（缺省 `skip`：记入 `skipped.jsonl` 并继续，等价于"线上那一批
也丢了"；**4xx 永远不跳过**，因为那是系统性我方 bug）；Search 有界重试后仍失败、
或答案/裁判子进程失败，则记入 `errors.jsonl` 并继续下一个用户。
修复服务后重跑同一命令从检查点续上，不在已经灌入未来记忆的用户中重问历史题；
缺题与跳过批在报告中保留（`summary.md` 的缺题列 + 技术错误列）。
续跑须保留原服务存储和隔离用户；服务存储丢失时使用新 run-id 完整重放。
**wire 身份 / 答案·裁判模型、端点、预算 / 服务版本**变化必须使用新 run-id；
**代码哈希、配置目录、base URL** 漂移会写进 `execution-drift.jsonl` 后继续
（与 `plan_driver.freeze_run` 同口径）——否则改一行注释就会锁死一条几十小时的长跑。

每题按现有 `official_capture_pipeline.py` 的家族规则生成答案和判分。
该输入属于采集重放，不能与截图中先前的 native 抽样成绩直接比较。
各家族评分范围及本地偏离见 [采集说明](../official-dataset-2026-09-29/README.md) §9。

## 进度与对比表

运行产物在 `eval/reports/runs/<run-id>/`，其中 `summary.md`、`summary.json` 在每个用户
完成后刷新。用户目录保留检查点、`input.jsonl`、`answers.jsonl` 与 `labels.jsonl`。
中断后可独立离线汇总：

```bash
uv run python -m eval.experiments.official_dataset summarize \
  --run-dir eval/reports/runs/official-v1.0

uv run python -m eval.experiments.official_dataset compare \
  --left eval/reports/runs/official-v1.0 \
  --right eval/reports/runs/official-v1.1.1 \
  --left-name v1.0 --right-name v1.1.1 \
  --output eval/reports/runs/official-version-comparison
```

输出 `comparison.md`、`comparison.csv`（Excel 可读）和 `comparison.json`。
主表列为「数据集、两版正确/题数（正确率）、变化（百分点）」，按变化降序排列，
另有加权总计、缺题和技术错误统计。

分母始终为固定清单题数，技术错误保留并单列，缺题不会静默缩小分母。
某数据集未判完时显示完成数，不给出该行正确率或百分点变化；`other` 显示无评分。
源文件、题目清单或答案/裁判口径不同的两次运行会拒绝比较（PRD §13）。
