# 共享固定题单与现有原生评测

> 最后核对：2026-10-14，对应 PRD §12.3、§13。题量、去重和验证结果见
> [数据集报告](../eval/reports/datasets-20261006.md#共享固定题单memory600--官方采集分层抽样)。

共享入口是 `uv run python -m eval.experiments.selection_suite`。
固定题单为 [memory600-official10-v1.tsv](../eval/experiments/selections/memory600-official10-v1.tsv)。
Git 只保存两列：数据集名称、原生题号，以及必要的审计注释。
详细 JSON 和输入指纹在校验或运行时生成，默认放在被 Git 忽略的 `eval/reports/runs/`。
GitHub 协作者运行时只需项目、这份题单和公开数据集；原采集与 Memory600 包仅用于
重新生成题单或额外核验出处。

## 选题与去重

题池为 Memory600 与官方采集可评分 Search 的分层抽样并集，不额外加入旧冻结配方题。
抽样按数据集分层，以固定 seed 对 seq 做哈希排序；总量向上取整，各层以最大余数分配。
先固定官方抽样，再跳过不支持的题目，**不补抽**，因此剩余题不再代表完整官方分布。
执行直接读取已选题号，不会重新抽样；抽样过程见上述数据集报告。

同源匹配依据来源地址：公开题号、persona 内题号、检查点或 case/问法序号。
MuSiQue 的可答/不可答变体与 MQuAKE 更新前/后世界分别识别；题面相同不直接合并。
生成题单时跳过来源映射不可靠、原生题号不存在或加载器不支持的记录，原因见报告。
MQuAKE 原生加载器只支持更新后 QA，采集中的更新前题因此跳过。

按用户最新约定，**只复用已实现的原生加载器与 pipeline**。同源重复只执行一次，
共享文件只保留执行所需的原生题号；来源成员和原始指纹可以在重新生成时留存于本地 JSON。
原生加载器的语料、原生题面与评分契约优先；例如 MuSiQue 转用逐题候选范围，
MQuAKE 使用既有分组及原始/更新顺序。此套件不是 Memory600 原空间评测或官方逐请求重放。

已知审计失败项排除；待审多跳、拒答边界和新增补充题的审计状态保留。
审计记录不进入 Add 或答案 prompt。原数据与金标不会在生成题单时改写。

## 协作者校验与运行

先安装项目依赖并准备所需公开数据：

```bash
make sync
make fetch-data DATASET=<题单中的数据集>
uv run --env-file .env python -m eval.experiments.selection_suite verify
```

数据根目录可通过 `--data-root` 或现有 `TIANXIMEM_BENCHMARK_DIR` 指定。
所需数据集从 TSV 第一列的去重集合读取；已准备的数据可直接复用。
新机器缺哪些数据就分别执行 `make fetch-data DATASET=<名称>`，准备流程见
[数据准备说明](benchmark-data.md)。`--env-file .env` 会将配置的数据根路径传给评测进程。
`verify` 加载被选问题和完整原生语料，生成金标与语料指纹并编译切批计划，
不访问服务或模型。选中题目缺失时直接失败，不临时替换来源。
TSV 本身不携带逐题内容指纹；首次运行冻结本机数据、配置和模型，续跑时拒绝输入变化。

启动要测的 Add/Search 服务、填好既有 AML/JUDGE 环境配置后执行：

```bash
uv run --env-file .env python -m eval.experiments.selection_suite run \
  --run-id shared-v1 \
  --service-revision <服务实际Git-SHA> \
  --base-url http://127.0.0.1:8000 \
  --configs-dir configs --profile local --offline
```

运行保持原生完整语料，只筛选问题；沿用既有 `shape_batch`、切批、HTTP 驱动、
答案输入构造与 `run_judge`，不重写各数据集评分逻辑。每个 run 派生独立用户命名空间。
逐事件检查点保留已成功 Search，重复执行同一命令可以续跑。
服务存储丢失、配置/模型/题单变化时用新的 run-id，不能拿旧检查点继续。
`--service-revision` 是调用者声明，须自行核对运行服务的实际版本。

运行产物位于 `eval/reports/runs/<run-id>/`，默认被 Git 忽略。
保存题单、配置快照、运行指纹、逐用户输入/答案/标签和结果汇总。
只有 Search 返回片段进入答案上下文；原始语料只用于 Add，金标仅用于判分。

## 汇总与版本比较

```bash
uv run python -m eval.experiments.selection_suite summarize \
  --run-dir eval/reports/runs/shared-v1

uv run python -m eval.experiments.selection_suite compare \
  --left eval/reports/runs/shared-v1 \
  --right eval/reports/runs/shared-v2 \
  --output eval/reports/runs/shared-comparison
```

汇总按固定去重题数报告完成数、缺题、正确数、准确率和技术错误；未完成时不发布准确率。
技术错误保留在分母。默认按总体和原生数据集汇总；精简题单不再保存来源成员映射，
因此不输出 Memory600/官方成员组分数。需要该分析时可显式使用本地详细 JSON。
比较要求同题单、同答案/裁判模型与端点、同预算和评分代码；服务版本与产品配置可以不同。

## 重新生成与出处核验

准备完整公开数据、原采集元数据和原 Memory600 包后：

```bash
uv run python -m eval.experiments.selection_suite prepare \
  --bundle /path/to/memory600 \
  --capture-dir /path/to/official-capture \
  --selection /path/to/new-selection.tsv \
  --details eval/reports/runs/selection-suite/source-details.json

uv run python -m eval.experiments.selection_suite verify \
  --selection /path/to/new-selection.tsv \
  --details eval/reports/runs/selection-suite/source-details.json \
  --verify-origins --bundle /path/to/memory600 \
  --capture-dir /path/to/official-capture
```

已有题单不会覆盖。重新选题或切换数据版本应生成新题单，经复核后共享。
原采集请求、Memory600 压缩语料、模型凭据与原始逐题运行产物不随题单提交。

## 交给 AI 运行的简要提示词

可直接将下面一句交给仓库中的 AI 助手，运行细节按本文和已有配置执行：

```text
请运行 eval.experiments.selection_suite 的固定题单评测，沿用现有配置，完整跑完并汇总结果。
```

只检查是否可跑时：请执行 `eval.experiments.selection_suite verify`，报告校验结果。
