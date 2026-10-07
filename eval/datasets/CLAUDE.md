# eval/datasets/ — 数据加载与 schema 落差预处理

**PRD**：§12.2（代理数据集）、§12.3（已知陷阱）、§12.4（契约不统一）、§12.5（法律约束）

## 要写什么

```text
longmemeval.py    lme_s_cleaned.json 加载
locomo.py         questions.jsonl + conversations.jsonl 加载与合并
clbench.py        clbench.jsonl 加载（契约与上面两份差得最多的一份）
beam.py           BEAM 100K（**公开 6 管线之一**；语料存在归档的 `official-extra` 档里，
                  **裁判走官方 `pipeline_beam.py`**，不是我们自写的那一类）
mquake.py         official-extra：MQuAKE-Remastered（**一条事实 = 一条消息**，整样本一个 session）
memtrapbench.py   official-extra：MemTrapBench（真对话，陷阱题）
corporatebench.py official-extra：CorporateBench（邮件/KB 文档）
medmemorybench.py official-extra：MedMemoryBench zh（**检查点式**：记忆只喂到该次问诊）
tempreason.py     official-extra：TempReason **L2/L3**（L1 无上下文，不是记忆任务）
personamem.py     公开六份之一：PersonaMem-v2（对话用于 Add，逐题 Search 输入由 harness 适配）
halumem.py        HaluMem-Medium QA（独立检查点，只加载当时可见对话；不注入标注）
musique.py        MuSiQue-Full dev（可答/不可答变体隔离，候选段落作为 Add 语料）
hybridqa.py       HybridQA dev（整张表格及全部链接段落，同表题目共享用户语料）
feverous.py       FEVEROUS dev（claim-only 候选页检索、整页元素 ID；不由 gold 补页面）
corpus_prepare.py CLI 的语料解压：固定包成员白名单、校验与原子发布
official_capture.py **官方采集流量**（2026-09-29 那轮，不进 git）——按 timeline 交错重放、
                  payload 用采集原文；**与独立加载器结构不同**（见 D34）
aml/              公开数据的版本化 AML 输入计划；进入前读 aml/CLAUDE.md，原生加载器不改名
preprocess.py     **schema 落差预处理层**（§12.3 第 9 条）
sampling.py       确定性分层抽样（`stratified_sample`，跨类不随机）
manifest.py       数据清单：URL / 固定 commit / sha256 / **已声明补丁**的唯一来源（D35）
layout.py         清单标识 → 本地路径的**唯一映射**（D35 的目录布局）
prepare.py        按需下载 / 校验 / 打补丁 / 原子发布 / 旧目录迁移（评测 CLI 的入口，D35）
registry.py       数据指纹（版本 + 切批口径）——§13 的记录要它
```

> ⚠ **ScriptMem 与 Doc-PP 不做**（各自的理由见 [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md)）：
> 前者的**剧本正文没有任何公开来源**（上游 `data/raw/*.json` 的 `conversation` 只有 381 字符的格式示例），
> 后者的作答端**只认 PDF 图像**、而我们的服务返回文本。
>
> ⚠ `personamem.py` 走**适配器**（[`../harness/`](../harness/) 的
> `*_pipeline.py`）——官方 pipeline 的子命令形状与通用两子命令不同，适配器里**直接调官方那几个函数**。
>
> **MQuAKE / MemTrapBench / CorporateBench / MedMemoryBench / TempReason 没有官方 AML pipeline**
> ⇒ 它们的 answer（以及大部分 judge）
> 是我们自写的（[`../harness/extra_pipeline.py`](../harness/extra_pipeline.py)），
> **分数只在仓内前后对比**。⚠ 一个例外：**MedMemoryBench 的判分口径是上游发布的**
> （`dataset/.upstream/medmemorybench/metrics/`，我们照它实现、prompt 直接从它读）——
> 但作答侧仍是我们的。
> 逐份的构造约定写在各自模块的 docstring 顶部——**读 loader 之前先读它**。

---

> **⚠ 一条边界（D16）**：**本目录是唯一允许知道具体数据集文件名的地方**（`conversations.jsonl` / `questions.jsonl` / `locomo_refined.json` / `lme_s_cleaned.json` …）。
>
> **`pairing/` 与 `store/` 只能看见归一化后的 AML 契约形状**——`role` / `content` / 可选 `timestamp`。
>
> **理由**：同一套 `Add` / `Search` 最终要接 **LongMemEval / PersonaMem / BEAM 以及未来的真实 API 请求**。**任何数据集的便利格式都不得向上渗透**——`conversations.jsonl` 的"本身即 JSONL / 自带 `role`"止步于此。

> **⚠ 一条路径口径（D16）**：**数据路径一律通过 `TIANXIMEM_BENCHMARK_DIR` 读取，代码中不得硬编码 `benchmark_data/` 或 `eval/datasets/`。** 默认值为 `dataset/`（D35）；本地开发通过 `.env` 指向实际数据目录。
>
> **边界**：`eval/datasets/LoCoMo-Refined/data/` 是**开发与自测用**的数据，**不是最终要跑的数据集**——归档才是。两者不可混为一谈，**也不得让代码依赖任何一边**。

---

## ⚠ 本目录是**两头**的，不是一个加载器

| 头 | 干什么 | 为什么 |
| --- | --- | --- |
| **(a) 加载器** | 各数据集的加载（见上「要写什么」） | 其中只有 LoCoMo-Refined + LongMemEval 共用同一套代理契约（§12.4）；其余各带自己的落差 |
| **(b) 契约抽取** | 从**其余各份**的 pipeline 源码里抽出记忆注入字段与裁判规则 | §11.3 的打包规则**是从它们推导出来的**——BEAM 的时间规则**正好相反**、CL-Bench 读 `text` + `created_at`、PersonaMem 归档原版只读历史（本地输入适配见 [`../harness/CLAUDE.md`](../harness/CLAUDE.md)） |

**只做 (a) 会漏掉一整类错误**：在 LoCoMo 上调好的注入格式，到 CL-Bench 可能被**整批丢弃**——而 (b) 就是提前知道这件事的唯一手段。

**各数据集的契约对照表在 [`../CLAUDE.md`](../CLAUDE.md)，本目录不重复。**

---

## §12.3 第 9 条：预处理层不是可选的

> **归档里的数据集与 pipeline 之间存在 schema 落差，Step 0 必须写一层预处理。**

**已核实的落差（真源是 pipeline 代码，不是 readme）：**

### LoCoMo-Refined

1. **`questions.jsonl` 的键与 pipeline 读的键对不上**：
   - 文件里有 `qa_id`，pipeline 读 `item["id"]` → **`KeyError`**
   - 文件里有 `answer`（list[str]），pipeline 读四个 gold 键之一 → **`ValueError`**
   - 记忆字段**为空**（文件里根本没有）
2. **记忆注入字段在数据集里根本不存在**——`speaker_1_memories` / `retrieved_context` / `memories` 的命中数都是 0，**只能由检索方运行时注入，但用什么键名注入没有明文**（§11.3 / [`../../docs/open-questions.md`](../../docs/open-questions.md) V3）。

**因此加载器必须产出一份转换后的 JSONL**，每行含：`id`（取自 `qa_id`）、`question`、四个 gold 键之一（取自 `answer`；`memory_text` 会把 list `"\n"` 拼起来）、以及**由系统注入的记忆字段**。

### 喂给 Add 的对话全文取自哪个文件

**`questions.jsonl` 只有 `evidence_messages`（证据轮），不是整段对话**——所以：

| 用途 | 取哪个文件 |
| --- | --- |
| **喂给 Add 的对话全文** | `data/public/conversations.jsonl`（开发 clone 才有）**优先**；归档退回 `locomo_refined.json` 的 `conversation`（D16） |
| 判断/评分用的题目与 gold | `questions.jsonl`（或 `locomo_refined.json` 的 `qa`） |

**二者逐题对齐**（仅 6 处答案 int/str 差异）——所以两边都要加载、按 `qa_id` 对齐。**别假设用一个就够。**

> **两条语义后果：**
>
> 1. **Add 源的两种布局（`conversations.jsonl` 优先 / 归档退回 `locomo_refined.json`）与 `strip()` 口径见 D16。** [`../../tests/CLAUDE.md`](../../tests/CLAUDE.md) 要求 content **首尾无空白**（AML 只做 `"\n".join(...)`、不插分隔符）——归一化层对两条路径都保证这一点。
> 2. **LoCoMo 是两个真人在对话**，此处的 `user`/`assistant` 是**数据集给 `speaker_a`/`speaker_b` 的约定标签，不是"用户 vs 助手"**。在 LoCoMo 上配对规则实际是"speaker_a 的一轮 + 对方回应，直到 speaker_a 的下一轮"。**LongMemEval 的 `role: user` 才是真 user**——两边语义不同，**别混**。

### CL-Bench

raw `clbench.jsonl` 的顶层键只有 `messages` / `rubrics` / `metadata`，而 `build_answer_prompt` 读的是 `system_prompt` / `question` / `qa_type` / `options` / `*_retrieval`——**这些顶层键在 raw 文件里一个都不存在**。

**不做转换的后果**：答案 prompt 塌成空的 system/question + 字面量 `"(no memories)"`——**不报错，只是分数没了**。

---

## ⚠ 两个数据集的 turn schema **不一样**（已核实）

**这是加载层最容易踩的坑**，因为 §6.2 的配对判据只依赖一个字段——"这条是不是 `user`"。

| | LongMemEval | LoCoMo-Refined |
| --- | --- | --- |
| turn 的键 | **`role`** + `content` | `locomo_refined.json`：**`speaker`** + `dia_id` + `text`<br>`conversations.jsonl`：**`role`** + `text` + `dia_id` + `speaker` + `session_date_time`（**Add 源，已带 `role`**） |
| role 取值 | `user` / `assistant`——**真 user** | 也是 `user` / `assistant`，**但那是 `speaker_a`/`speaker_b` 的约定标签** |
| 证据标记 | turn 上的 **`has_answer`**（bool，可选，只有证据轮带） | `evidence`（dia-id 列表，如 `"D1:3"`） |
| **每 turn 有 timestamp 吗** | ❌ **没有**（时间在 session 级 `haystack_dates`） | ❌ 没有——是 **session 级**；但 `conversations.jsonl` 把 `session_date_time` **复制到了每条 message 上**（值相同，仍无 session 内区分度） |

**⇒ 加载层必须把两边都归一化成 AML 的形状（`role` / `content` / 可选 `timestamp`）**，否则 `pairing/` 的判据看不见 `role`，**会把整个 session 归成一个对，且不报错**。

> **LoCoMo 侧不用"归化出 role"**（`conversations.jsonl` 自带）——**但归一化层仍要写**，因为 LongMemEval 走的是 `role`+`content` 那条路，而**两边的输出必须先统一，`pairing/` 才只依赖一个字段**。
>
> ⚠ **两个数据集都没有 per-turn 时间戳** ⇒ `event_time` 在 session 内**必然没有区分度**（不是"可能"）。

**`locomo_refined.json` 的 `conversation` 结构**：一个 dict，键是 `speaker_a` / `speaker_b`，然后成对出现 `session_N_date_time` / `session_N`；每个 `session_N` 是 turn 的列表。**对话全文在这里**（不在 `questions.jsonl`）。

---

## ⚠ LongMemEval 没有 per-turn 时间戳——`event_time` 会退化

**已核实**：LongMemEval 的 turn dict **只有 `role` + `content`（和可选的 `has_answer`）**，**时间在 session 级**，存在与 `haystack_sessions` 平行的 `haystack_dates` 里（形如 `"2023/05/20 (Sat) 02:21"`）。

**两个直接后果：**

1. **harness 必须为每条消息合成 `timestamp`**——否则 §2.1 的可选字段为空，§6.1 的 `event_time` 全为 NULL，`created_at` 只能发 `""`（§11.3 的有定义降级路径）。
2. **同一个 session 内所有消息拿到同一个日期** ⇒ `event_time` 在 session 内没有区分度 ⇒ **位置是唯一能保证邻域稳定的东西**（§6.1；D28 起 = `(request_id, local_index)` + **Add 内显式链** `prev` / `next`：邻接不再跨 Add）。

---

## 两个计分数据集的形状

### LongMemEval

**500 条**，顶层是 JSON **数组**。每条 9 个键：`question_id` · `question_type` · `question` · `question_date` · `answer` · `answer_session_ids` · `haystack_dates` · `haystack_session_ids` · `haystack_sessions`。

| 字段 | 注意 |
| --- | --- |
| `question` / `answer` | **`answer` 是单个字符串**（不是列表——与 LoCoMo 相反） |
| `answer_session_ids` | 证据的 **session 级** id 列表，形如 `["answer_280352e9"]`——做 session 级 recall 用这个 |
| `haystack_sessions` | `list[list]`——外层每个元素是一个 session，内层是 turn dict 的列表 |
| `haystack_session_ids` / `haystack_dates` | **与 `haystack_sessions` 按下标平行**（三条数组按 index 对齐） |

- **题量分布**（Step 0 建 harness 用）：`multi-session` 133 / `temporal-reasoning` 133 / `knowledge-update` 78 / `single-session-user` 70 / `single-session-assistant` 56 / `single-session-preference` 30
- **64.8% 的问题需要 ≥2 个 session 的证据**（324/500，平均 1.896 个），其中 multi-session 类 133 题 **100%** 跨 session——**这是本项目唯一的靶子**

> ⚠ **拒答题是横切标记，不是第 7 类**：`question_id` 以 `_abs` 结尾的共 **30 道**（multi-session 12 / single-session-user 6 / temporal-reasoning 6 / knowledge-update 6）。**§13 的 T2 要人工给 133 道 multi-session 题分三类，其中 12 道是拒答题，行为与其他题不同，必须单独拎出来。**

#### `lme_s_cleaned.json` vs `lme_test.json` —— 已复算

> **⚠ `lme_test.json` 已不在归档里**（它是个等着被误用的坑，见 [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md)）。
> **⇒ 下表那些数字无法在本地复算**——要复核"1,230"，得先从上游 LongMemEval 取回那一份（线索见 [`../../tools/fetch_benchmark_data.py`](../../tools/fetch_benchmark_data.py) 的 `DELETED` 段）。

| 断言 | 核实结果 |
| --- | --- |
| 同题、同证据 | ✅ **500/500 的 `answer` 与 `answer_session_ids` 完全一致**，`question_id` 集合相等 |
| 只在 haystack 上不同 | ✅ `s_cleaned` 的 session id 集合在 **500/500** 条里都是 `test` 的**子集**；两者共有 session 的 turn 数完全一致 |
| 空 session | ✅ **`test` 有 1,230 个 0-turn session，`s_cleaned` 有 0 个** |

**一条细化（PRD 说"1,230 个空 session 与 15 个干扰 session"，措辞可以更准）**：在 120 条的抽样里，`test` 多出的 309 个 session 中 **302 个是空的、7 个有内容**。所以"多出来的都是空的"**过强**——**是"绝大多数空 + 极少数有内容"**，而那些有内容的正是"干扰"。**结论不变：用 `lme_s_cleaned.json`。**

**为什么不用 `lme_test.json`**：**空 session 会让本地切批与线上对不上**——一个 0-turn session 会凭空多出一个（空的）批次边界，而 D24 之后**批界落在哪直接决定组合结果**。

### LoCoMo-Refined

- CC BY-NC 4.0，未饱和，**1,382 题**
- **计数类问题集中在多跳类**：21/213 = 9.9%，单跳类 27/802 = 3.4%

> ### ⚠ "计数类"的口径必须先在 harness 里固定
>
> 上列数字对应 `how many|how much|how often|number of|count|how long`。**只算 `how many` 时单跳类是 0.25%，多跳类 9.39%**——**换口径数字就变**，而它正是附录 A"实体层做不做"的依据。

> ### ⚠ LoCoMo 发布版的分类 ID 与论文顺序不一致（§12.3 第 1 条）
>
> **实测映射**：`1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial`
>
> **依据不是猜题面**，而是 **evidence 跨度**：ID 1 有 95% 跨 ≥2 个 session，ID 4 有 94.5% 只有单条 evidence，ID 5 则是 446/446 全带 `adversarial_answer` 且 `answer=null`。
>
> **论文 §4.1 的顺序是** `1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial`——**按论文顺序映射会让 5 类里的 4 类被错标**（只有 adversarial 恰好对上），**且不会报错**。
>
> **实测分布可用来交叉验证这个映射**：`"4"` 802 · `"2"` 299 · `"1"` 213 · `"3"` 68（合计 1,382）。**ID 1 只有 213 条却占 9.9% 的计数类问题、ID 4 有 802 条却只占 3.4%**——与"多跳类少而集中、单跳类多而稀"一致，**与上面的映射自洽**。

> ### ⚠ `category` 的类型在两个文件里不一样
>
> **`questions.jsonl` 里是【字符串】**（`"4"` / `"2"` / `"1"` / `"3"`），**`locomo_refined.json` 里是【整数】**（1–4）。
>
> **harness 不要硬编码单一类型**——`int(category)` 归一化，否则按 `category == 1` 筛选会**静默筛出 0 条**。

---

## 另外三条陷阱（§12.3）

| # | 陷阱 |
| --- | --- |
| **2** | **ScriptMem 做不了代理评测**——对话原文因版权原因未发布 |
| **3** | **不要用 MemoryAgentBench 当代理**——它不在 AML 的数据集清单里，在其上调优未必迁移 |
| **8** | **BEAM 的数据现在在归档里**（`official-extra` 的 `beam/`，**100K 档**，2026-09-30 取回）——那三份失败残留（`beam.json` / `beam_rows.json` / `beam_100k.json`）仍在清单的 `DELETED` 段里。⚠ 但**它仍然不能当代理数据集**（§12.4：契约与 LoCoMo/LME 不同），而且**只覆盖 100K 档**（500K / 1M 未取） |

**第 7 条（PersonaMem）**：三个 split 的 schema 不统一——`question_type` 的词表在 32k / 128k / 1M 之间互不相同；`correct_answer` 在 32k 里是 `(c)` 这类选项字母、在 128k/1M 里是整段选项文本。**"MCQ 精确文本匹配"必须先做归一化**，harness **不要硬编码单一词表或单一答案格式**。

**并且**：PersonaMem 的 CSV **没有 `chat_history` 列、也没有 `incorrect_answers` 列**，而 `pipeline_v2_personamem.py` 缺 `incorrect_answers` 直接 `raise TypeError`，且用整段选项文本匹配——**该 pipeline 不能直接吃归档 CSV**。

---

## 法律约束（§12.5）—— **许可表在本节**

**数据不得用于训练。「微调模型」不是暂不实现，是不允许。**

| 数据集 | 许可 | 出处 |
| ---- | ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** | 归档 readme |
| ScriptMem | **CC BY-NC 4.0** | 归档 readme |
| PersonaMem v2 | CC BY 4.0 | 归档 readme |
| LongMemEval | **MIT**——⚠ **归档内仍无依据**；出处已从上游取回（HF 数据集卡 + LongMemEval-V2 仓 `LICENSE`），**细节见** [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md) 的许可表 | —— |
| CL-Bench | 归档内无许可证文本 | —— |
| BEAM | **CC BY-SA 4.0** | 归档 `beam/README.md` 的 `## 📄 License` 段（**100K 档那一份卡**） |

> **NC（非商业）这一列值得注意**：LoCoMo-Refined 与 ScriptMem 都是 CC BY-NC 4.0。**不影响参赛**，但**它意味着这两份数据不能进任何商业用途的产物**——如果后续想把系统或其中组件开源/商用，**这两份数据的评测结果是引用不了的**。

> **"不得训练"这条的出处提醒**：归档里**没有**这些条款的出处（全库检索 `only for the evaluation` 等措辞零命中），属**单边来源**，引用前须回原始页面/许可证文件复核。**结论不变，但不要把它当成已归档的实证。**


## 按需准备（D35）

`native` 与 `aml-v1` 的边界、支持范围和续跑纪律统一见
[`../../docs/benchmark-data.md`](../../docs/benchmark-data.md) 的「公开数据的 AML 输入适配」。
适配器只构造事件和独立评分题；HTTP、切片及切批交给 harness，不从金标挑语料。

`manifest.py` 只记录来源、版本与哈希，`layout.py` 只负责本地目录映射，
`prepare.py` 负责暂存下载、校验、已声明补丁与原子发布。
HybridQA 的固定语料包提取出逐表文件并记录成员哈希；FEVEROUS 从官方 ZIP 提取
原始 SQLite，再构建可重建的 title/intro FTS5 缓存。候选生成范围只在
[`feverous.py`](./feverous.py) 声明，数据规模和验证见 [接入报告](../reports/hybridqa-feverous-pipelines-20261006.md)。
评测 CLI 在执行前显式调用 `ensure_dataset`；loader 和模块 import 不联网。
`__init__.py` 按需导出加载器；仅下载/校验时不要求安装 pyarrow。
目录纪律见 [`../../dataset/CLAUDE.md`](../../dataset/CLAUDE.md)。

HaluMem 和 MuSiQue 的独立评测口径、复用的评分来源、未覆盖的上游指标见各自
加载器 docstring；使用方法与验证见
[`../reports/halumem-musique-pipelines-20261006.md`](../reports/halumem-musique-pipelines-20261006.md)。
