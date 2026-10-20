# CorporateBench 失败 50 题：共同事实取证为什么没命中（2026-10-16）

**口径**：诊断集 = `eval/reports/runs/answer-targeted-50-20261016/manifest.json` 的 50 道
v1.1.1 CorporateBench 历史错题；事实库 = 同一目录下 LLM 抽取版
（`llm-add-qwen-20261016.sqlite`，353 篇 → 637 条事实）；正式基线是
`results-qwen-planner-collection-20261016`（**0/50**，45 题走 100 条兜底）。

**免责**：这是**定向错误集**，不是全量分数；数字只在仓内前后对比。

## 一、结论

这 50 题的失败**大多不在"抽取质量"**。逐题过一遍离线链路后的分布：

| 桶 | 题数 | 卡在哪 |
| --- | ---: | --- |
| 员工入职 / 开始工作 | **17** | **文档里没有证据**：gold 由 `graph.nq` 的 `worksAt` + `beginning` 算出（12 人里只有 2 人有日期）；文档只给了 2 处间接线索 ⇒ 见 §八 |
| "meetings about X" 主题题 | 18 | 10 题的**主题短语在语料里没有任何字面落点**（gold 多为 `[]`/`False`）；8 题有落点但题面是**改写**（"Quality Review and Standards Alignment" ↔ 语料 "Quality and standards alignment"） |
| 缺席 / 没出席 | 7 | 语料有 32 篇带 `Attendees:`/`Organizer:` 的邀请，但**抽取器一条出席事实都没产出**（提示词把 `To:` 头禁掉后模型连标注字段也不抽了） |
| 计数（部门/团队/任务） | 3 | 需要实体归类与去重后计数；抽取把 `Facilities Team` 归成 department、把签名行切成 `Facilities Team - Sales & Marketing` |
| 标题计数题（"出现几次"） | **5** | **精确 gold 已经在事实库里**，被两处机械缺陷挡掉 ⇒ 见下 |

所以"换成 LLM 抽取"没救回来，**主要不是模型不行**：17 题无解、18 题要么无落点要么需要语义匹配、
7 题缺一整个关系层，只剩下 5 题是能立刻拿回的。

## 二、5 道计数题：gold 一直在库里（机械缺陷）

用修正后的模式离线读同一个事实库，逐题与 gold 相等：

| 题 | 事实 | gold |
| --- | ---: | ---: |
| kb_qa-38 Cross-platform compatibility testing - Work Session（Q1） | 1 | 1 |
| kb_qa-51 Jurre Price <> Emilia Anglada（2 月） | 5 | 5 |
| kb_qa-63 Charlotte Dowerg <> Jared Chartier 1:1（3/16–28） | 0 | 0 |
| kb_qa-87 Optimization results synthesis Review | 5 | 5 |
| kb_qa-88 Emilia Anglada & Maria Richards Check-in（Q1） | 11 | 11 |

两处缺陷（修前全部 0 命中）：

1. **标题归一化不一致**：查询侧用 `title_key()`（抹标点），匹配时事实侧用 `key()`（保留标点）
   ⇒ 只要标题里有 `-`、`<>`、`&`、`1:1` 就永不命中。
2. **按三元组去重吃掉计数**：`fetch_evidence` 的 `ROW_NUMBER() … PARTITION BY
   subject_key,relation_key,object_key,…` 把跨文档的同名标题压成 1 条 ⇒ "出现几次"数不出来。

**修法**（`src/tianximem/facts/query.py` / `retrieve/evidence.py` / `service/search_stages.py`）：

* 标题模式改用 `key()`（与事实侧同一归一化），另加"跨过量化短语"的捕获过滤；
* `QueryPlan.per_occurrence`：标题/主题/出席这类"哪几次"的计划**关掉三元组去重**，
  且**不先用时间窗口裁事实**——每一次出现带着自己的来源日期进证据，窗口比较留给答案模型
  （裁掉之后"窗口内 0 次"和"没有这条会议"在证据里长得一模一样）。

## 三、LLM planner：被自家护栏挡死，且 schema 表达不了时间（已修）

原始指标（`llm-add-metrics-planner-collection.json` 之前的 planner 臂）：planner
`unsupported_scope 18 / abstained 4 / planned 1`。

* `_UNSUPPORTED_SCOPE` 里含 **`\b\d{4}\b`（任何年份）** 与 `during|after|before|between|...`
  ⇒ 这批题几乎全带年份/季度，**整条 LLM 通路在本语料上等于关着**；
* 提示词又写着"时间限制一律返回 null"，而 schema 里**没有字段**能表达时间——
  确定性执行器其实早就支持 `date_bounds`（`time_bounds()` 能解析
  "after 23 February 2024"/"between … and …"/"the week of Jan 22nd, 2024"）。

**修法**（`src/tianximem/service/grounded_planner.py`，`prompt_version` 升 **v2**）：
filter schema 加 `period`（模型**逐字**抄题面时间片段，本地解析成区间）；
解析不出、或题面有时间措辞而计划没有区间 ⇒ 弃权（`invalid_period` / `missing_period`），
**限制条件不许被悄悄丢掉**；同时补了 meeting topic / attendee 的绑定示例。
护栏里只保留真正不可表达的（否定、比较、多实体、相对今天的时间、频率词）。

修后同一批 26 道 `no_plan` 题：planner 给出计划 9–11 道（其余 `unsupported_scope`/弃权/
非法时间段），**但全部 `no_match`**——因为事实层里没有这些主题的事实（见第一节的 18 题）。

## 四、抽取器：缺一整个关系层（已改，重跑中）

353 篇只出 618 条事实（`document title` 332 + `employee of` 212 + department 34 + team 16 +
task 9 + meeting topic 8 + meeting type 4 + role commencement 3），其中：

* **`attendee` / `organizer` / `absent from meeting` 一条都没有**——提示词第 4 条为了防
  "把 `To:` 头当出席人"，把模型吓得连 32 篇邀请里明写的 `Attendees:` 字段块也不抽了；
* 抽出的 3 条 `role commencement` 是**错关系**（"I've commenced work on <task>" 被当成入职），
  真正的入职句全语料只有 1 条；
* `supported_fact` 里藏着一个数据集特判 `or "zenith labs" in lower`，它正是
  "把 profile 职业当 employee"那类误报的门。

改动（`tools/run_llm_add_corporatebench.py`，`PROMPT_VERSION` → `corporatebench-qwen-add-v2`）：
标注字段（`Attendees:` 块 / `Organizer:` 行）**明确允许**并给出三件套（人、会议名、引句）；
补 `meeting topic` 的语料内许可（"our X meeting"、"For our upcoming work session … focus on X"）；
删掉 `zenith labs` 特判。

## 五、诊断集自校准：可答性标注（新增）

`eval/corporatebench-answerability-failed-50.jsonl`（50 行，只存 qid / 判定 / 依据，不含题目原文）：

| 判定 | 题数 | 含义 |
| --- | ---: | --- |
| `absent` | 27 | 语料里没有证据（17 入职 + 10 主题无落点）⇒ **任何抽取器都答不了** |
| `paraphrase` | 8 | 主题短语有落点但题面是改写 ⇒ 精确事实匹配不可靠，属混合检索的活 |
| `open` | 10 | 需要跨来源集合差（7 缺席）或实体归类计数（3 计数） |
| `exact` | 5 | 题面即标题，事实层逐条可证 ⇒ 已在第二节修回 |

**这条校准的用处**：这批题上"再涨 X 题"的上限是 **15/50**（exact 5 + open 10），
其余 35 题要么无证据、要么本来就该走混合链。之前把它当成"LLM 抽取不好"的证据会走偏。

## 六、复跑结果（2026-10-16，`results-qwen-planner-v2-20261016`）

同一 manifest、同一 answer/judge 模型（Qwen3.5-9B）、同一切批口径，只换两件事：
事实库改成 v2 LLM 抽取（353 篇 → 538 条事实：document title 353 · **attendee 102** ·
organizer 32 · department 24 · team 7 · task 7 · role commencement 5 · meeting topic 5 ·
meeting type 3）、planner 提示词 v2。

| 指标 | 上一臂（v1 事实库 + planner v1） | 本轮 |
| --- | ---: | ---: |
| 完全正确 | 0/50 | **1/50**（kb_qa-63） |
| 返回共同取证证据（非 100 条兜底） | 5 | **17** |
| 走 100 条兜底 | 45 | 33 |

**检索这一侧已经对了**：5 道计数题全部返回逐次出现的证据（1 / 10 / 9 / 5 / 11 条，
含每条的来源日期）。**但答案这一侧没跟上**：

| 题 | gold | 证据里的事实 | 模型答 |
| --- | ---: | ---: | ---: |
| kb_qa-38 | 1 | 1 | 0 ✗ |
| kb_qa-51 | 5 | 10（2 月 5 条） | 2 ✗ |
| kb_qa-63 | 0 | 9（窗口内 0） | 0 ✓ |
| kb_qa-87 | 5 | 5 | 2 ✗ |
| kb_qa-88 | 11 | 11 | 7 ✗ |

**离线 A/B（同一答案 prompt、temperature=0）**：把证据换成**事实投影**（逐行带
`Source record date`，而不是整封邮件）后 87 答对（5）、38 答错（0）——两臂分别是 **1/5 与 2/5，
且方向相反** ⇒ 这是答案模型的计数能力问题，不是证据形状问题。注意 harness 的答案模型是本地
替身，换 `gpt-4o-mini` 后必须重测（与 D22 同一条口径）。

其余两处也不在检索：

* **入职家族**：现在返回 1 条**合法但不足**的证据——"I've commenced my role at Zenith Labs"
  （Jennifer，2024-03-30）。"after 23 Feb 2024" 的 gold=2 里，另一个人的入职在语料里
  **没有陈述**（只有 "Going through Zenith Labs's orientation schedule" 这种暗示）
  ⇒ 这 7 题仍答不对，但证据本身是对的；
* **缺席家族**：返回 22–60 条逐次出席事实，模型**直接列出参会人**而不是做集合差（gold 全是 `[]`）。

⇒ **下一轮的瓶颈已经从"取证"移到"答案"**：要么等 Step 5 换答案模型，要么在打包侧
给出更适合计数/集合差的证据形态（本轮 A/B 说明"换成事实投影"不是那个形态）。

## 七、已知遗留

* `evidence_limit = 12` 对"出现 20 次的会议"仍会回退（kb_qa-88 的 11 次刚好压线）；
* 缺席系列需要**同一系列全部出现的出席清单**（22 次 × 数人），当前上限装不下，
  且答案模型要自己做集合差——本轮不动它；
* planner 的词表护栏仍会误伤"标题里带 `plan`"的题（如 "Design feedback integration plan
  Review"）——保守方向，宁可弃权。

## 八、KB 侧核对：gold 是怎么算出来的（2026-10-16 补测）

数据集自己的 README（`dataset/corporatebench/README.md`；数据目录不进 git）写明：**先建一张时间知识库**
（`.kb` 里的 `graph.nq`），文档是"**围绕图里的证据串**生成的"（*Generate corporate
documents around graph-derived evidence strings*），gold 则是"**手工核验的 SPARQL**"
（*Derive QA answers through manually verified SPARQL queries*）。官方评测本身就有两条臂：
**RAG（k=10）或直接 SQL 查这张 KB**。

把 `dataset/corporatebench/data/kb/zenith.kb` 打开逐条复算（RDF-star：`<<e worksAt c>> core#beginning "…"`）：

| 题面条件 | `graph.nq` 复算 | gold |
| --- | ---: | ---: |
| 员工总数 | 12 | 12 |
| `worksAt` 带 `beginning` 的人 | **2**（Andre Chauvin / Jennifer Andersson，均 2024-03-30） | — |
| 入职 after 2024-02-23 / after 2024-03-12 / after 2024-01-10 | 2 / 2 / 2 | 2 |
| 入职 before 2024-01-09 / before 2024-03-14 | 10 / 10（= 12 − 2：其余 10 人在窗口开始前就在职，图里没有 `beginning`） | 10 |
| 2024 年 1 月入职 / 2 月 5 日那周入职 | 0 / 0 | `[]` |

会议题同理：每个 meeting 实体带 `meetingType`（task 35 · direct_report 30 · project 5 ·
team 3 · department 3）与 `topic`，题目问的正是这两者 + 日期窗口的组合：

| 题 | 图里 (type × topic) 命中 | gold |
| --- | ---: | --- |
| kb_qa-39 / 127 direct_report / project × "testing and validation checkpoint" | 0 / 0 | False |
| kb_qa-161 task × "Quality Review and Standards Alignment"（Q1） | 2（2024-02-22、2024-03-08） | True |
| kb_qa-36 / 65 / 138 task × "Task Documentation and Handover" | 4（1/4、1/12、2/21…）——落在题面窗口内的是 1 / 0 / 0 | 与 gold 一致 |
| kb_qa-240 team × "Timeline and Deliverable Planning" | 0 | False |

**文档侧为什么没有**：

* **`meetingType` 在正文里根本不出现**（"direct report" 全语料 0 命中）；
* `topic` 多数只在正文里顺带一提（"…for our testing and validation checkpoint meeting"），
  或者**只印在我们看不到的文件名上**（`agenda_Quality_Review_and_Standards_Alignment_8e01.txt`）；
* 两名"晚入职"的日期恰好等于他们第一封邮件的日期，所以文档里只剩间接线索
  （"I've commenced my role at Zenith Labs" / "Going through Zenith Labs's orientation schedule"），
  其余 10 人在图里**没有入职日期**记录。

⇒ 这一族（2026-10-04 的审计记作 **"员工入职时间筛选后计数 42 题"**，见
[`corporatebench-20261004.md`](./corporatebench-20261004.md)）**在数据集设计上就属于
"文档臂答不出、KB 臂才答得出"的一类**。AML 只发文档、不给 KB ⇒ 在记忆系统里结构性不可解，
继续投优化预算没有意义（这正是 `answerability` 里 `absent` 的定义）。
