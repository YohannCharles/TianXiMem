# HaluMem 与 MuSiQue 独立本地评测接入

核对日期：2026-10-06。范围：PRD §12 的本地评测加载、输入适配与评分；服务仍只接收
Add/Search 契约。数据来源和固定版本继续以 `eval/datasets/manifest.py` 为唯一清单。

## 已接入的入口

```bash
make baseline DATASET=halumem
make baseline DATASET=musique
# 服务需先启动；首次运行自动准备缺失数据及评分依赖。
# 禁止下载时加 ARGS='--offline'。
make data-check DATASET=halumem
make data-check DATASET=musique
```

`make eval DATASET=<名称>` 也可运行，默认加载全量；`make baseline` 使用
`eval/experiments/recipes.py` 声明的冻结抽样。两份都沿用当前数据目录配置与临时下载流程。

## 数据及冻结抽样的实测计数

| 数据集 | 全量独立样本 | 全量题目 | 冻结参数 | 冻结实测 |
|---|---:|---:|---|---|
| HaluMem-Medium QA | 896 个提问检查点 | 3,467 | `--limit 60 --spread` | 60 个检查点、203 题 |
| MuSiQue-Full dev | 4,834 个题目实例 | 4,834 | `--limit 120 --spread` | 120 题，可答/不可答各 60 |

HaluMem 原始文件包含 20 个用户、1,387 个会话。491 个会话没有 QA，但仍作为后续
检查点的历史保留。每个检查点的用户 ID 独立，只包含当前及此前会话，题目 ID 由用户、
会话位置、题目位置共同构造。`persona_info`、`memory_points`、题目及证据标注均不进入 Add。

检查点之间重复投喂历史是这一实现的明确成本：全量独立检查点合计投喂 1,342,018 条
消息；原始对话为 60,146 条消息。加载时共用不可变 Session/Message 对象，重复的是
实际投喂。冻结抽样覆盖全部 QA 类型，但 Dynamic Update 只有 1 题，Multi-hop Inference
只有 5 题，不能凭这个小样本判断这些类型的稳定表现。

MuSiQue-Full 同一原始 ID 出现两次：2,417 个可答变体与 2,417 个不可答变体。
两个变体分别生成用户 ID 和题目 ID，避免结果覆盖和语料混用。Add 只包含段落标题与正文，
不包含答案、`is_supporting`、问题分解链。段落顺序沿用公开文件，并非官方采集里的重采样顺序。

候选段落数并非全部为 20：20 段有 4,803 题，19 段有 17 题，18 段有 9 题，17/16 段
各 2 题，15 段有 1 题。加载器按原文逐题保留。冻结抽样按跳数及可答性分层，每种可答性
分别取 2-hop 31 题、3-hop 19 题、4-hop 10 题。

## 评分范围与兼容性

HaluMem 复用 `official_capture_pipeline.py` 中的作答入口与上游 QA 裁判：从归档的
`eval_tools.py` 读取原始 prompt，保留 Correct/Hallucination/Omission 标签，仅 Correct
计为正确；裁判解析失败保留 JUDGE_ERROR。参考回答和关键记忆点只用于裁判。
本入口不覆盖上游记忆抽取、完整性、更新指标，也不声称复现 AML 的输入适配。

MuSiQue 在 `extra_pipeline.py` 中增加本地短答案作答与纯函数评分：可答题按现有
NFKC、小写、空白及首尾包装标点归一化后精确匹配参考答案或别名；不可答题仅接受
`INSUFFICIENT_EVIDENCE`，允许首尾空白。混合答案、附带猜测或空回答不能通过拒答判分。
拒答题另通过既有 abstention 汇总记录。

MuSiQue 的独立作答入口显式带 `musique-local-qa-v1` 契约标记，旧采集重放继续使用原有
答案 prompt 与评分。这里没有实现上游 answer/support F1 或成组 sufficiency 指标，
成绩只用于相同本地口径下比较。

两份都登记了数据指纹、形状说明、正文标签和冻结抽样。HaluMem 指纹覆盖实际读取的
Medium JSONL，MuSiQue 覆盖实际读取的 Full dev JSONL；未用作 Add 的标注不生成额外数据副本。

## 验证

完整题库加载已核对用户 ID、题目 ID 和样本内会话 ID 唯一性；原始材料已离线校验固定哈希。
回归测试覆盖检查点前缀、变体隔离、分层抽样、标注不泄入、空检索不回填历史/答案、
三分类裁判、精确别名匹配、拒答及旧采集 prompt 兼容性。

HTTP 驱动测试使用 MockTransport。真实 pipeline 子进程的模型调用使用回环地址上的
模型桩，核对配置注入、检索进入实际答案请求和裁判 JSON 读取，不调用付费模型。

- `uv run pytest -q`：1,110 passed，65 warnings，89.10 秒。警告来自依赖弃用与
  Qdrant 客户端版本探测，不影响通过结果。
- `make lint`：mypy 检查 42 个服务源文件通过，`ruff check .` 通过。
- `make data-check DATASET=halumem` / `DATASET=musique`：各 3 项材料与固定清单一致。
- 标准库模式 `python3 -S -m eval.datasets.prepare --list` 通过。
- `make -n baseline DATASET=halumem` / `DATASET=musique`：均进入通用 runner 的对应
  `--dataset` 与 `--frozen` 分支。
- `git diff --check` 通过。

本次不产出真实模型基线成绩；本地模型桩的正确回答仅用于验证接线和评分协议。
