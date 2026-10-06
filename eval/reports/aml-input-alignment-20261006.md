# 11 份本地数据集输入与 AML Full 采集的对照

审计日期：2026-10-06。范围是上一轮七份数据集及新接入的 HaluMem、MuSiQue、HybridQA、FEVEROUS。
这里的「Full」指 `official-dataset-2026-09-29/` 中已采集的请求，不能推断下一次比赛的数据构造。

结论：HTTP 字段和通用包装已接近 AML；语料选择、正文序列化、用户/会话边界及提问时序仍有实质差异。
`--add-shape official` 不能视为「这个数据集的输入已经复刻 Full」。最应补的是 MQuAKE 的更新前后时序、
FEVEROUS 的共享语料池，以及 LoCoMo 的干扰语料和输入角色变体。MuSiQue 与 HybridQA 的源语料已经对上，
应改的是具体包装，不需要另找一套语料。

本次没有修改产品或数据集 pipeline。四份子集仍使用冻结的原生适配口径；
其四个 loader、`add_shape.py` 和 `driver.py` 与本次读取的工作区版本哈希相同。
核查状态时发现 FEVEROUS 的正式作答被网关 HTTP 400 中断，已另加显式记错的续跑旁路，见下文。

## 证据与复现

复现入口：[audit.py](runs/aml-input-alignment-20261006/audit.py)。它只读取本地文件，
不下载数据、不调用模型或服务；结果写入同目录的 `audit.json`（忽略的原始产物）。

```bash
.venv/bin/python eval/reports/runs/aml-input-alignment-20261006/audit.py
```

审计绑定的采集文件指纹：

| 文件 | SHA-256 |
| --- | --- |
| `official-adds.jsonl` | `38b242f3b54a372162785e1d7732d2f5c9d157271853e4a250b3c7cf4a3028d4` |
| `official-searches.jsonl` | `850c167572acd2f009d698e446bf6a5330a187e94d2b7e6b8eeba364c6f76812` |
| `official-eval-kit.jsonl` | `ba4692ceb0b39f5d295e612fcf337f6b396c0212b7450c40f22eb0edf70131d4` |

题目归属采用套件已有归属；TempReason 另用原文标签或 `source:` 正文中的年月区间识别，明确不当作金标。
Query 比较只做首尾 `strip()`，不做大小写、标点或改写归一化。统计包括采集到的重复 Search。
采集的覆盖空档、时间记录口径和重试历史见 [采集 README](../../official-dataset-2026-09-29/README.md) §2/§7/§8。
以下 Add 数是唯一请求数，不是原始重投次数。

## 当前已对齐的部分

- [ServiceClient](../harness/driver.py) 通过 HTTP 发 Add 的 `request_id/user_id/session_id/messages`，
  Search 的 `user_id/query/top_k`。这十一组已标注查询的采集 `top_k` 均为 100，本地基线也使用 100。
- [add_shape.py](../harness/add_shape.py) 加逐数据集正文标签，将非 assistant 角色折入 user，
  按包含标签的 8,000 字符上限切片；保留 loader 提供的可选时间戳。
- [batching.py](../harness/batching.py) 在单会话内按 20 条消息或约 2,000 个空白词合批。
  官方 Adapter 的词数算法仍未知，不能据此保证批界相同。

本地 ID 用 `mqk-*`、`halumem-*` 等命名，而采集使用哈希 ID，本身不是需要修复的问题。
需要对齐的是「哪些语料归到同一个 user、哪些消息共用 session、何时向这个 user 提问」。
本地 [run.py](../experiments/run.py) 对每个 Sample 先 `ingest` 全部语料，再将
`Question.question` 直接送 Search；Add 包装不会自动改变 Search query。

## 逐数据集比较

| 数据集 | Add 输入差异 | Search 输入比较 | 调整建议 |
| --- | --- | --- | --- |
| **CorporateBench** | 353 条有序消息的正文、role、timestamp **完全一致**。Full：1 个 user、每封邮件独立 session/独立 Add。Native：三种 QA 各一个 user，全部邮件共用一个 session 后合批。 | 783/783 条查询匹配公开题面。 | 补每文档独立 session 和共享公司语料 user 的兼容口径；不用改邮件正文。 |
| **MQuAKE-Remastered** | Full 用含实体/关系 ID 的 JSON 三元组及 `UPDATE` 请求；native 用可读三元组和自然语言改写，每 user 默认混合 64 个 case，并总注入改写。 | 1,124/1,124 题面匹配，但 Full 有更新前/后两个阶段，native 只问更新后的状态。 | **优先调整**：JSON 输入臂、按 case 的语料范围、同 user 原始 Add → Search → UPDATE Add → Search。 |
| **TempReason** | 可识别 Full 例子直接给整段时序事实 `source: ... from ... to ...`；native 每页将事实与维基正文一起注入，并逐句拆成消息。 | 公开 L2/L3 的 9,789 个唯一题面在全体采集 query 中零精确匹配。已识别的时间事实族也零匹配。 | 原生测试保留时间推理用途；Full 原文回放可检查输入/检索，缺公开金标的实例不能补出可信端到端分数。 |
| **MemTrapBench** | 对话角色及正文标签接近。Full 含合成 timestamp 和少量 canary 包装；native 原始对话不带 timestamp。批界仍是近似。 | 308/308 题面匹配 `final_trigger`。 | 不需要重写 QA 任务；补时间字段和批次形态的兼容验证。 |
| **LoCoMo-Refined** | Native 只注入本身的完整对话；Full 的相关 user 含大量 LongMemEval 干扰对话、部分全 user 的真人对话，以及平台反馈写回，且存在交错提问。 | 1,476/1,476 题面匹配；题面相同不代表可见记忆或检索难度相同。 | **优先补对照**：角色变体、干扰语料和按提问时刻可见的历史；优先用采集回放验证。 |
| **LongMemEval-s** | Native 每题独立 haystack，合成同会话时间戳。Full 中查到它的对话被用于 LoCoMo 相关 user 的语料。 | 公开 500 个题面在这份采集 query 中零精确匹配。 | 保留独立长历史基线；不能声称已与 Full 的独立 LongMemEval QA 输入对齐。 |
| **MedMemoryBench** | Native 是独立检查点 user 的历史前缀；Full 存在同 user 持续 Add/Search，并有原文无前缀、角色前缀及合成时间戳等形态。 | 500/500 题面匹配。 | 保留「只到当前检查点」；需要补持续同 user 的 timeline 验证与时间/切批兼容。 |
| **HaluMem** | 对话来源一致。Native 使用独立检查点 user、原始时间；Full 同时有 `[Time: ...]` 无 timestamp 和角色前缀加合成 timestamp 两种输入，还有持续同 user 的检查点。 | 1,481/1,481 题面匹配。 | 补时间包装变体与持续更新验证；题干不需要重写。 |
| **MuSiQue** | 240/240 个可答/不可答实例的有序 `(idx,title,段落正文)` 与 Full 对上。Native 少了 `[Paragraph N]` 包装、标题后的空行及 Full 合成时间字段。 | Full 在题干后追加「仅用段落、证据不足回答 `INSUFFICIENT_EVIDENCE`」。Native Search 只有题干：0/240 全串匹配，去该尾指令后 240/240 匹配。 | **调整成本较低**：补段落包装和 Search 原指令；继续使用 Full-dev 两变体。 |
| **HybridQA** | 100/100 张 Full 表格结构及全部链接段落能对回公开语料。Full 给整表 JSON 和 `Passage ID`，native 改成逐行/逐列文本及 `Page/Link` 文本。 | 113/113 题面匹配。 | 补整表 JSON 序列化输入臂。语料下载及「整表 + 全部链接」选择已正确。 |
| **FEVEROUS** | Full 的 249 道题共享一个 user/一个 session 的 416 页 JSON 语料。Native 为每个 claim 从全库选 5 页，独立 user，将页面改成可读元素文本并显式写证据 ID。 | Full 是完整核查指令 + `Claim:`；native Search 只有 claim。全串 0/249 匹配，抽 claim 后 249/249 匹配。 | **优先调整**：共享固定语料池、原始页面 JSON、完整 query。原生 claim-only 选页测试作为另一种输入口径保留。 |

来源分别是对应的 [loader](../datasets/)、[通用 Add 包装](../harness/add_shape.py)、
[native runner](../experiments/run.py) 和本次指纹绑定的请求原文。

## 最值得先改的几项

### MQuAKE：现在没有测到同一份更新任务

Native [mquake.py](../datasets/mquake.py) 的 `_memory_messages` 会同时注入原始事实与自然语言改写，
`_questions` 一律取更新后答案。在已恢复金标的 Full 查询中，386 条对应原始状态，738 条对应更新后状态；
249 个相关 user 中 126 个在首次 Search 后仍有 Add。不能把本地全部更新后的结果当成两阶段测试结果。

兼容臂应从相同公开 case 构造带 ID 的事实 JSON，保留明确的 UPDATE 包装，并在同一 user 上分两阶段提问。
若只把 natural language 前面加 `Corpus:`，这几个差异都还存在。

### FEVEROUS：语料池、检索范围和 query 同时不同

把 2,500 条采集消息的 `Corpus:` 包装去掉，按原顺序拼接后，可以完整解析出 **416 个 JSON 页面**，
其标题均不重复，总正文 16,442,322 字符。1112 次 Add 后才开始该 user 的 249 次 Search。
这与 [feverous.py](../datasets/feverous.py) 的每题 top-5 页输入有不同的干扰范围及召回前提。

下载了完整 Wikipedia SQLite 并不能使二者自动相同：当前 native 先用 claim-only FTS/BM25 选页，
被测服务仅看选中的五页。应增加共享语料池/原始 JSON 的测试口径，或者直接回放捕获的 Full 输入。
共享池应从声明的输入来源生成，不能从问题的 gold evidence 补页。

Search 的核查指令也需要保留：Full 明确要求三类标签及原生证据 ID；
native 虽在答案 prompt 中提出标签/证据要求，但该 prompt 没有被送入 Search。

### LoCoMo：同题不同记忆库

Full 中归到 LoCoMo 的 user 内，36,566 条消息使用 `User:`/`Assistant:` 标签；
其中 **36,510 条正文**能在公开 LongMemEval-S 的消息里精确匹配。
这证明了源语料交叉，不能只用数据集名称或题面匹配来判断 Add 已对齐。

该族还有全 user 的真人对话输入，当前 native `official` 保留 assistant。
此外，8 个有 LoCoMo 查询且有 Add 的 user 在首次 Search 后仍有 Add。
单纯「完整原始对话先 Add、然后问全部题」无法复现上述状态，适合用真实请求的 timeline 回放作对照。

### 时间、会话边界：不应通过删掉历史来修

Full 的 HaluMem、MedMemoryBench 分别有 7、3 个相关 user 出现交错。
Native 的独立检查点方式避免使用后续会话，原则合理；与持续 user 的索引演化仍不等价。
应补持续更新的验证，不应改成先投喂整个未来历史再问早期问题。

CorporateBench 则是一个可以低成本对齐的例子：消息本身完全一致，Full 的 353 次单消息 Add/353 个 session
与本地单 session 合批不同。批界和 session 会影响配对、邻接及段合并；ID 字面值可以不同，作用域应该明确。

## 建议的实现方式

1. **保留当前原生 pipeline 作为基线，增加显式的输入契约版本。** 对照运行的 manifest 记录语料构造、
   query 包装、timestamp 形态与批次口径；输入变更使用新的 run ID、存储及配置快照。
2. **将检索 query 与答题题干分开。** 当前只有 `Question.question`，同时服务于 Search 和作答。
   可新增可选 `search_query`，缺省仍用题干；MuSiQue 和 FEVEROUS 的兼容口径填完整 Full query。
3. **分别增加 JSON/时序输入适配。** 优先处理 MQuAKE 两阶段更新、FEVEROUS 共享页面池和 LoCoMo 干扰/timeline；
   然后对齐 HybridQA JSON、CorporateBench 会话边界及对话数据的时间包装。MuSiQue 包装可作为较小改动先落地。
4. **用已有 [official-capture replay](../experiments/replay_official.py) 检验原始 Full 输入。**
   它绕过 `add_shape`/重新切批，在 user 内按采集 timeline 投递原文 Add 和完整 query，适合作为输入对齐主线。
   抽样应保留所选题目需要的历史；不能先喂完未来 Add 再搜索。

回放仍有边界：采集不是全部平台内部状态，失败/跳过请求需查看 `skipped.jsonl`，
全局并发和平台实际重试不能由当前串行重放完整还原。它复现的是已有请求输入及已记录的时序。

**输入对齐也不代表评分对齐。** 多份本地 answer/judge 是自写适配。
当前原生 FEVEROUS 评标签及完整证据组，采集回放的本地裁判只评标签；这只是两个本地评测口径的区别，
不能推断 AML Full 实际只评标签。此次报告没有给它们的分数建立换算关系。

## 验证

审计脚本在上述真实文件上执行成功；结构化表格/页面重建和消息比较均完成。
`ruff check` 和 `ruff format --check` 通过。输入审计不调用模型，冻结输入快照未更改。

## 审计期间查到的后台中断

前三份正式子集已经完成：HaluMem 203 题、MuSiQue 120 题、HybridQA 120 题。
FEVEROUS 在第 5 题 `feverous-4053` 的作答 API 上中断，前 4 题已有判分。
同一个冻结提示再次请求，拿到网关实际响应（[诊断原始产物](runs/new-datasets-benchmark-20261006/gateway-error-feverous-4053.json)）：

- Qwen 最大上下文为 131,072 token；本次请求输出额度为 1,024 token。
- 网关报告提示至少 130,049 输入 token，合计至少 131,073，因超限返回 400。
- 相同提示共 329,773 字符，用平台 `o200k_base` 计数为 117,672 token。

这不是数据集缺语料或等待重试能解决的问题，而是本地模型与平台 tokenizer 的长度口径不同。
它也不能说明旧 smoke 的每一次重试都由同一原因造成；旧成功重试未保存 stderr 的边界不变。

[续跑入口](runs/new-datasets-benchmark-20261006/resume_terminal_errors.py) 保留全部冻结输入、原答案及原始评分函数。
原生作答子进程经过既有有界重试后，如果仍返回 HTTP 400，就把该题写为带传输错误元数据的空答案；
原裁判将其标成 `ANSWER_ERROR`，计零且保留在原始题号集合内，然后继续剩余题。
其它运行异常仍响亮失败。失败 stderr 和实际 prompt 指纹写入 `terminal-answer-errors.jsonl`。
它不缩短记忆、降低输出额度或重选题目，不把技术失败当成普通 `WRONG`。

续跑策略及脚本哈希独立归档到 `continuation-manifest.json`，原来的失败阶段保留在 `status.json`。
首次续跑的旁路未接上 runner 使用的包导出引用，仍因同一题中断；该失败阶段与脚本修订均保留。
修正导出路径后，2026-10-06 **20:29:54**（北京时间）以 detached supervisor **2017274** 再次启动。
对已有实际 400 响应且提示指纹完全一致的失败题复用错误记录，避免重新请求已经确认超限的提示。
已通过实际 `run.py` 导出函数做无模型调用检查，确认输入指纹不变、400 得到 `ANSWER_ERROR`、
其它错误继续抛出，已有错误记录的复用幂等。
本轮成绩汇总时必须同时给出作答错误数；对本地模型增加长度适配应该另立评测口径。
