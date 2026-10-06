# 公开数据的 AML 输入适配（2026-10-06）

已实现 `aml-v1`，显式接入九个家族的公开数据输入计划。
`eval/datasets/` 保留名称与原生加载器；默认 `native` 的 Add/Search 输入和本地评分逻辑保持原口径。
使用说明统一见 [benchmark-data](../../docs/benchmark-data.md#公开数据的-aml-输入适配)，
前置依据为 [Full 输入审计](aml-input-alignment-20261006.md)。

## 实现

- [`datasets/aml/`](../datasets/aml/)：`plan.py` 定义 Add/Search 事件和独立评分题；
  `registry.py` 管理支持范围、参数预检、按需分派与源版本声明；`common.py` 处理通用对话、
  CorporateBench 文档和持续检查点。其余五个文件分别处理 MQuAKE、LoCoMo、MuSiQue、HybridQA、FEVEROUS。
- [`harness/plan_driver.py`](../harness/plan_driver.py)：先包装后切片/切批，再有序执行 HTTP，
  派生与实际输入和 run-id 绑定的用户/request ID。HTTP 身份指纹不包含金标；评分指纹单独包含问题、
  金标及类别。`input-manifest.json` 同时记录规则代码、源版本声明、执行配置与实际事件指纹。
- [`experiments/run.py`](../experiments/run.py)：新增 `--input-contract native|aml-v1`、
  `--aml-time-style`、`--aml-pool`，并按契约选择 [`recipes.py`](../experiments/recipes.py) 的冻结配方。
  LoCoMo 会同时准备 LongMemEval 材料，FEVEROUS 缺少声明页面池时在下载及模型预检前退出。
- [`harness/corpusqa_pipeline.py`](../harness/corpusqa_pipeline.py)：新增显式的 FEVEROUS JSON
  证据契约，仅从 Search 文本中的完整可解析页面或显式括号 ID 验证引用，复用原有固定上游评分函数。
  采集重放仅复用公共 HTTP 分派，不调用输入编译器；其 payload、批界、query 和原有失败策略保持不变。

MQuAKE 的原始读取/抽样、MuSiQue 的原始 row 与问题验证、HybridQA 的经 receipt 校验的原始整表和段落
已抽成可复用函数。原生序列化逻辑保留。MuSiQue 旧说明中的“采集段落重采样”已依据输入审计更正，
该说明修订不改变原生加载器实际返回的语料。

## 历史时点与续跑

MQuAKE 按 `原事实 → 原答案 Search → UPDATE → 新答案 Search` 执行。
来自更新后链的非编辑边作为普通背景事实；不在原始链上的编辑边先放 `target_true`，再通过 UPDATE
放新值。真实冻结 case 曾出现新分支主体，不能假定每条编辑边都能从原始答案链里找到。

HaluMem、MedMemoryBench 按稳定的源 persona 检查点前缀合并，每次只 Add 新增会话，再 Search 当期题。
检索失败会中止后续 Add。每个成功 Search 原子保存历史快照后才继续执行下一事件；
答案生成在随后读取对应快照，不会在完整记忆已写完的用户上重新检索早期问题。
所有用户的初始检查点在首次 HTTP 前建立；删除用户目录、丢失历史快照或改变输入/执行配置会明确失败。
续跑保留相同 run-id、存储和输出目录；变更输入使用新 run-id 与独立存储快照。

## 冻结口径的真实材料检查

只读取本地已准备材料并构造/编译计划，未调用服务、模型或下载器，未产生新的付费 benchmark。
以下计数来自 [离线复核脚本](runs/aml-input-adapters-20261006/check_plans.py)，
输出为同目录 gitignored 的 `plan-checks.json`。准备层负责源文件的完整 SHA 校验；
本次计划检查没有再次扫描大型 Wikipedia 数据库的全部字节。

| 数据集 | 编译后用户数 | 评分题数 | Add 请求数 | 消息数 |
| --- | ---: | ---: | ---: | ---: |
| CorporateBench | 1 | 250 | 353 | 353 |
| FEVEROUS | 1 | 116 | 993 | 2,267 |
| HaluMem | 20 | 203 | 2,359 | 40,292 |
| HybridQA | 120 | 120 | 481 | 4,677 |
| LoCoMo-Refined | 3 | 346 | 305 | 3,015 |
| MedMemoryBench | 20 | 388 | 1,200 | 18,932 |
| MemTrapBench | 250 | 250 | 809 | 14,298 |
| MQuAKE-Remastered | 40 | 240 | 80 | 202 |
| MuSiQue | 120 | 120 | 156 | 2,399 |
| 合计 | 575 | 2,033 | 6,736 | 86,435 |

全部配方的实际题数与声明一致。复核同时验证 user/request ID 唯一、HTTP 字段白名单、
消息字符上限、批次消息上限，以及每个评分题与有序 Search 的唯一对应关系。

FEVEROUS 的缺省选题限定到同一捕获用户的原始 Search：249 请求对应 240 个不同 claim，
全部唯一匹配到公开 dev。按该候选集分层抽样后为 116 题，其中 SUPPORTS/REFUTES 共 76 题，
其至少一组完整金标证据全部存在于声明的 Add 池，余下 40 题为 NOT ENOUGH INFO。
该覆盖率是评分侧诊断，**没有用于挑页面或过滤题目**，也不代表 Search 已召回证据。
不能沿用同一捕获页面池却对公开全集直接抽题：初始检查中 113 道可判定题只有 7 道有完整证据在池内，
这会把语料/题库不匹配当成检索失败。当前按原始 query 对齐解决该问题；金标改变不会改变 Add/选题。
显式自定义页面池的题目来自公开 dev，该范围不同于缺省捕获范围，实际题数由 manifest 记录。

```bash
.venv/bin/python eval/reports/runs/aml-input-adapters-20261006/check_plans.py
.venv/bin/pytest tests/test_aml_inputs.py -q
```

## 测试与兼容性

新增 [`test_aml_inputs.py`](../../tests/test_aml_inputs.py) 最终 **51 项通过**，其中包括九份真实材料
的冻结题数检查；合成 fixture 覆盖各适配器的结构和标注隔离，不需要下载或模型调用。
HTTP 使用真实 `ServiceClient` 与 `httpx.MockTransport`，验证失败阻断未来 Add、恢复时不重新 Search
已完成的早期题、相同 ID/相同 payload、缓存损坏/参数变更失败，以及 runner 到评分输入的接线。
FEVEROUS 证据测试调用固定版本上游 scorer，区分正确标签与完整证据组严格分。

原有相关回归连同当时新增的 36 项一起运行：**399 passed，322.52 秒**。
之后补入页面池预检、九份冻结题数、同一采集 query 匹配与失配失败测试，再单独运行新增文件：
**51 passed，9.89 秒**。两组有重叠，不能相加作为测试总数。
FEVEROUS 查询范围修正后，另复核 runner 的相关回归：**44 passed，14 deselected，6.58 秒**；
未重复运行已通过的大型原生材料检查，九份 AML 冻结材料由新增测试覆盖。

```bash
.venv/bin/pytest tests/test_aml_inputs.py tests/test_datasets.py tests/test_datasets_extra.py \
  tests/test_memoryqa_datasets.py tests/test_corpusqa_datasets.py tests/test_experiments.py \
  tests/test_harness.py tests/test_official_capture.py tests/test_dataset_prepare.py -q --tb=short
```

改动 Python 文件的 Ruff 检查通过，`git diff --check` 通过。
另外对修订前快照和当前实现的原生 FEVEROUS 初始作答/格式校正 prompt 做了逐字比较，
prompt、两次输出及返回结果均一致。加载器抽取函数和原生 CLI 默认值由上述回归覆盖。

## 与 Full 的实际差距

- **时间与批界**：合成时钟冻结了形态，不猜测平台用户偏移。8,000 字符切片与 20 条/2,000
  空白词切批是本地近似；逐请求字节一致仍依靠原始采集回放。
- **MQuAKE**：一公开 case 一用户，是有序更新测试的小语料范围；不是捕获中按用户聚集的大事实池。
- **LoCoMo**：每个完整对话配一个确定性公共 LME haystack。公开题库没有逐题到达时刻和平台反馈，
  因此不制造真实交错或反馈写回；精确时间线需捕获请求。
- **HaluMem/MedMemoryBench**：保证公开已选检查点的历史可见性，但不能声称检查点到达顺序与具体平台
  调度完全一致。`max_questions` 在合并前按源检查点/QA 子集裁题，不代表每个合并用户的总题数上限。
- **FEVEROUS**：本地捕获 Add 恢复出 416 个页面标题，再从公开数据库加载原始页面。
  DB 中存在 NFD 标题、清单中存在 NFC 标题，查询兼容二者，实际 JSON/引用保留源页面拼写。
  重新序列化与切批后为 993 Add/2,267 消息，与捕获的 1,112 Add/2,500 消息不逐字相同。
  默认 claim 子集按同一捕获的原始 query 唯一匹配到公开 dev，保留原始 Search query；
  重复请求合并，公开标注用于评分。显式自定义池的公开全集选题则是另一来源范围。
  严格证据分依赖 Search 实际返回完整可解析页面或显式 ID；
  残片无法补全，可能产生标签正确但严格分为零的结果。
- **评分**：其余适配器继续复用各自已有本地答案/评分入口，FEVEROUS 仍为标签与完整证据组评分；
  它不同于采集重放的 label-only，不等同 AML 平台最终分数。
- **未支持**：TempReason 与 LongMemEval 独立 QA 缺少可靠的捕获实例映射，`aml-v1` 明确拒绝。
  原生入口继续可用；LongMemEval 语料作为 LoCoMo 干扰历史使用。BEAM、CL-Bench、PersonaMem 等
  未登记此版适配器的家族同样明确拒绝，不静默回落。
