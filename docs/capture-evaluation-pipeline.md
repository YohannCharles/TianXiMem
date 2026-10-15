# 官方采集的数据集分类与版本对比

> 最后核对：2026-10-13，对应 PRD §2.2、§6.4、§12.3、§13。

入口为 `uv run python -m eval.experiments.official_dataset`，依次提供 `prepare`、
`run`、`summarize`、`compare`。分类统计见
[当前分类报告](../eval/reports/runs/official-dataset-20260929/coverage.md)。

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

执行复用逐事件检查点：历史 Search 快照落盘后才继续 Add；Add 或 Search 失败会阻断
该用户的后续事件。修复服务后重跑同一命令续上，不在已经灌入未来记忆的用户中重问历史题。
这与旧 `replay_official.py` 的失败批次跳过策略不同，缺题会在报告中保留。
续跑须保留原服务存储和隔离用户；服务存储丢失时使用新 run-id 完整重放。
题目、模型、版本或配置变化后必须使用新 run-id。

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
