# HybridQA / FEVEROUS 独立 pipeline 接入（2026-10-06）

两份已接入按需下载、独立加载器、HTTP Add/Search、回答及上游函数评分、冻结抽样和 run record。
本次验证使用真实上游语料和本地 HTTP/LLM 桩，没有运行付费模型基线，也没有调用 AML Smoke/Full。

## 来源与下载

来源、固定版本及全部 sha256 的唯一清单为
[`../datasets/manifest.py`](../datasets/manifest.py)，本地路径由 `layout.py` 映射。

| 数据集 | 固定来源 | 本地材料 |
| --- | --- | --- |
| HybridQA | [HybridQA](https://github.com/wenhuchen/HybridQA/tree/db22fda8c5951438fade3c69d75b350335ba93b3) dev/reference 和评分脚本；其 README 指向的 [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks/tree/dc066e1a6d5281511d8b73a6107d5ad2824cc2b2) 固定提交 | `dataset/hybridqa/` 的题库、原始 tar.gz、dev 引用的 `tables_tok/` 和 `request_tok/`；评分脚本与语料 README 在 `.upstream/hybridqa/` |
| FEVEROUS | [官方下载脚本](https://github.com/Raldir/FEVEROUS/blob/32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6/download_data.sh) 指向的 [Wikipedia ZIP](https://fever.ai/download/feverous/feverous-wiki-pages-db.zip)，以及同一提交的评分脚本 | `dataset/feverous/` 的 dev annotations、`feverous_wikiv1.db` 及 `.cache/title-intro-v1.sqlite`；评分源码与下载脚本在 `.upstream/feverous/` |

HybridQA README 的旧 S3 预处理包本次返回 `404 NoSuchBucket`，使用其另行公开的
WikiTables-WithLinks 固定仓库包。FEVEROUS 的脚本位于仓库根目录，README 中的
`scripts/download_data.sh` 路径与该提交不符；准备器下载语料，不直接执行 shell 脚本。

FEVEROUS ZIP 实测 MD5 为 `b01271df477efdfca99441c7dc6fe859`，与
[官方 Zenodo 归档](https://zenodo.org/records/4911508) 一致；进一步固定 ZIP 和数据库各自的
sha256。下载在 `dataset/.tmp/` 暂存，核对 ZIP 长度/sha256、成员长度/CRC 和数据库
sha256 后原子发布数据库，清除临时压缩包。HybridQA 白名单提取完整 table/request 对，
`corpus-manifest.json` 记录逐成员哈希，加载所选表时再核对成员字节。
已有原文件损坏时失败并保留，不能静默覆盖。

| 本机核验项 | 结果 |
| --- | --- |
| HybridQA 下载包 | 158,882,547 B；下载约 22.5 s |
| HybridQA dev 引用语料 | 3,053 张表，6,106 个 table/request 文件，合计 113,876,533 B |
| HybridQA 全量加载 | 3,053 个 Sample、3,466 题、153,290 条归一化消息，约 2.67 s |
| FEVEROUS ZIP | 10,353,775,701 B；本次下载约 570.4 s |
| FEVEROUS 数据库 | 53,486,538,752 B；流式解压及哈希约 163.3 s |
| FEVEROUS 完整页面索引 | 5,421,405 页；派生索引 5,740,314,624 B |
| FEVEROUS dev | 7,890 条有效 claim；跳过文件开头的全空模板行 |
| FEVEROUS label 分布 | SUPPORTS 3,908、REFUTES 3,481、NOT ENOUGH INFO 501 |

全量 annotations 的 33,581 个证据 ID 均通过类型/位置后缀解析。

首次 FEVEROUS 下载/解压的压缩包与数据库同时占用约 63.84 GB；保留数据库与最终索引
约 59.23 GB，另需索引构建及 SQLite 临时空间，建议准备至少 70 GB 可用空间。
下载器先检查压缩包加数据库所需空间，派生构建失败会清理本次暂存文件。
重复准备和 `data-check` 仍会读取原文件校验哈希，数据库校验有本地 I/O 成本。
索引可删除后由准备入口重建，loader 不自动下载或建索引。

## 输入范围

### HybridQA

[`hybridqa.py`](../datasets/hybridqa.py) 抽题后按 `table_id` 合并，共享同一用户语料。
Add 包含整张表的标题、上下文、所有行、链接和 `request_tok` 的所有关联段落，
不会依据 `answer-node` 挑答案行/段落，不注入题目、参考答案或类别。
不合成不存在的时间戳。表格行与段落文本化是本地适配，不能假设等同 AML 序列化方式。

### FEVEROUS

[`feverous.py`](../datasets/feverous.py) 从完整数据库构建标题及前五句 FTS5 索引。
claim 经 NFC 归一化、固定停用词过滤后 OR 查询，BM25 标题权重 8、intro 权重 1，
固定取前五页；同分按原页面 ID 排序。使用 FTS5 rank 游标提前停止，并读完最后一名的
同分行，保持与 `ORDER BY bm25(...), page` 相同的确定性边界。
[SQLite 官方文档](https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results) 说明了 rank 游标及提前停止的排序行为。

候选生成函数只接收 claim，不接收 label、gold evidence 或 annotator 操作，漏召页面不补齐。
候选整页句子、表格、表头、caption、列表和原始元素 ID 都进入 Add；每题隔离用户，逐页分 session。
数据库原始页面/元素 ID 保留 Unicode 字节，NFC 只作用于检索 claim。

**这是有限候选整页上的本地检索评测**：完整 Wikipedia 用于固定的候选生成器，
被测 Add/Search 服务接收候选页，而非整个数据库。它不能表示被测服务在全 Wikipedia 上
的端到端检索成绩，也不复刻 AML 候选池。该范围记录在加载器版本、数据指纹和
`dataset_score.scope` 中，引用分数时必须保留。

两份作答都只读取逐题 Search 返回的 `retrieved_context`；空检索显式渲染 `(no memories)`，
不会回填原语料或 gold。默认 Add 正文使用 `Corpus:` 前缀，并沿用现有消息/词数切批。

## 评分与产物

[`corpusqa_pipeline.py`](../harness/corpusqa_pipeline.py) 下载后验证上游评分源码 sha256，
执行未改动的 import/函数节点，跳过脚本 CLI 顶层入口。回答 prompt 为本地 Search 适配。

| 数据集 | 逐题函数与结果 | `scores.dataset_score` 聚合 |
| --- | --- | --- |
| HybridQA | 上游 `compute_exact`、`compute_f1`；`is_correct` 对应 EM | `exact_match`、token `f1` 的逐题均值，`mean` 为 EM |
| FEVEROUS | 上游 `truncate_evidence`、`is_correct_label`、`is_strictly_correct`、`evidence_macro_precision`、`evidence_macro_recall` | `strict_score`、`label_accuracy`、证据 macro precision/recall 和二者的调和 `evidence_f1`；`mean` 为严格分 |

FEVEROUS 回答 JSON 为 `label` 加元素 ID 列表，只接受 Search 文本中可见的 ID。
元素 ID 从结尾的类型/位置解析，页面名可以含下划线；转换为上游 `[page, type, position]`。
按上游最多五条句子及二十五条其他元素截断。
标签正确且至少一个金标证据组被完整覆盖才得严格分，**NOT ENOUGH INFO 同样执行该上游规则**。
非法 JSON/标签/证据或引用不可见 ID 记 `ANSWER_ERROR`，所有逐题指标为零。

上游聚合函数在平均 precision 和 recall 同为零时会除零；本仓复用逐题函数，聚合时将
该调和 F1 定义为零，不修改上游源码。证据 F1 是**先平均 precision/recall 再取调和均值**，
不是平均逐题 F1。`overall` 保持二值 EM 或严格正确率，额外指标经
`JudgeResult.metrics` 汇总并打印在 CLI 和 run record 中。

答案文件包含完整输入指纹；同一 run-id 的 Search 结果或其他输入变化时，在发起模型请求前
拒绝复用旧答案。evaluate 先核对全部 ID 和输入指纹，再打开判分产物。
官方采集重放继续执行原有 label-only FEVEROUS 口径，不能与本次严格证据组分混用。

## 冻结抽样与复现

参数的唯一声明处为 [`recipes.py`](../experiments/recipes.py)，`--frozen` 不与显式 limit/spread 混用。

| 数据集 | 冻结参数 | 真实加载结果 |
| --- | --- | --- |
| HybridQA | limit=120，spread，按 table/passage/other 分层 | 120 个 Sample、120 题、6,568 条消息；passage 70、table 47、other 3 |
| FEVEROUS | limit=120，spread，按 label/challenge 分层 | 四舍五入后 123 个 Sample、123 题，合计 615 个候选页面 session、23,495 条消息 |

FEVEROUS 冻结样本包括 SUPPORTS 60、REFUTES 53、NOT ENOUGH INFO 10，覆盖所有十八个
label/challenge 组合。候选页面 session 的计数包含不同题目间重复出现的页面。
本次完成固定样本的整页解析，也另行核验了原始数据库前 300 页的句子、section、表格和列表结构。

```bash
make fetch-data DATASET=hybridqa
make fetch-data DATASET=feverous
make data-check DATASET=hybridqa
make data-check DATASET=feverous
make baseline DATASET=hybridqa
make baseline DATASET=feverous
# 服务及模型配置沿用 README；已有语料可加 ARGS='--offline'
```

`--offline` 禁止联网，允许由本地固定原文件重建派生语料/索引；`--check` 只校验，不创建或修复文件。
`purpose='judge'` 只准备小型上游评分依赖，采集重放评分不会下载这两份大语料。

## 验证

新增 [`test_corpusqa_datasets.py`](../../tests/test_corpusqa_datasets.py) 覆盖整表与全部段落、
候选不依赖金标、NFC 检索、表格/列表 ID、边界同分排序、空 Search 无回填、上游评分、
压缩包失败及原子发布、变更输入不能续用旧答案。
HTTP MockTransport 驱动真实 Add/Search harness，loopback HTTP LLM 桩接真实 answer/evaluate
子进程及 run record，验证从检索到指标的接线；未调用外部付费模型。
未下载两份语料时，真实冻结样本检查会明确 skip，合成 fixture 测试照常运行，
不会在测试期间自动下载 Wikipedia。

真实材料的 `make data-check DATASET=hybridqa` 和 `DATASET=feverous` 均通过，
HybridQA 的 6,106 个提取语料文件逐项校验通过。

- `uv run pytest -q`：**1,135 passed**，约 347.85 s；包含新增的 23 项 corpus QA 测试，
  以及真实冻结样本两次加载的题数、qid 确定性与 session 唯一性检查。
- `make lint`：mypy 的 42 个源文件及全仓 ruff 通过。
- 本次涉及的 Python 文件 `ruff format --check` 通过；`git diff --check` 通过。
- 标准库 Python 的 `python3 -m eval.datasets.prepare --list` 正常，准备器不依赖模型栈。
- `make -n baseline DATASET=hybridqa` / `DATASET=feverous` 均指向通用 runner 的 `--frozen` 入口。

本次未产生真实模型分数；以上数字是材料规模和实现验证，不是新的基线成绩。
