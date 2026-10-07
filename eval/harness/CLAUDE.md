# eval/harness/ — 一轮评测怎么跑

**PRD**：§12.1（三层结构）、§12.3（第 5 / 9 条）、§6.5（切批）、§13（记录）

## 要写什么

```text
driver.py                  喂 Add、调 Search（**走 HTTP**）、收集返回
plan_driver.py             AML 事件编译、HTTP 分派与逐事件检查点；历史 Search 落盘后才继续 Add
batching.py                模拟 AML 切批（20 条消息 + 2,000 词的**空白分词近似**）
add_shape.py               模拟 AML **正文形态**（`<标签>: …` 前缀 + `system` 折叠 + 单条 8,000 字符切分）——线上就是这么发的
judge.py                   包住 pipeline 的裁判（`run_judge` 跑 `answer`→`evaluate` 两步）
extra_pipeline.py          本地自写的 `official-extra` answer / evaluate（含 MuSiQue）
corpusqa_pipeline.py       HybridQA / FEVEROUS：Search-only 作答，固定上游函数判分与逐题指标
personamem_pipeline.py     PersonaMem-v2 适配器——Search 片段替代原始历史，选项与 MCQ 判分复用归档
scriptmem_pipeline.py      ScriptMem 适配器——**到不了裁判**（加载层与 pipeline 表里都没有它），保留作参考
corporatebench_pipeline.py CorporateBench 的答案 prompt 与纯函数判分——被 `extra_pipeline.py` 调用
official_capture_pipeline.py 官方采集重放的逐题分派层（D34）；HaluMem 独立 QA 也复用其
                        上游三分类裁判与通用作答入口。评分范围见对应加载器与验证报告
api_config.py              归档 pipeline 要 `import` 的端点配置——`run_judge` 经 `PYTHONPATH` 注入本目录这份（见下文）
annotate.py                相对时间就地注解——`--memory-date annotate` 那一档的 harness 侧实现
run_record.py              每次 run 的配置指纹 + 数据指纹 + 结果
```

> **`batching.py` 与 `add_shape.py` 是同一类东西**：都在回答"本地发的像不像线上"。
> 切批决定**批界落在哪**，形态决定**正文长什么样**——两者都进数据指纹，
> **两者都不能拿不同口径的分数互比**。

---

## ⚠ `Add` 的 payload 不是我们造的——它由 AML 造

这是 `add_shape.py` 存在的全部理由。**全量实测**官方 43,272 条 add / 454,937 条消息：
**正文 93.9% 带 `<标签>: ` 前缀**（其中 82.9% 的标签就是 `role`），
**`role` 的取值域只有 `user` / `assistant`，零 `system`**——
逐项数字（含绝对条数）在 [`add_shape.py`](./add_shape.py) 的模块 docstring。

⇒ 缺省口径是 **`official`**：逐数据集加实测到的标签、`system` 折进 user 侧。
`native`（数据集原始形状）**只作对照**，`alluser`（把 role 全折成 user）是**判分池那一簇**
的形态——三个取值都进数据指纹。

**`system` 折叠是逐字节证明的**：`dataset/clbench/clbench.jsonl:24` 的 raw `messages[0]`
是 `role="system"`，官方发的是 `role="user"` + `user: …`。它**不是排版**——D24 下
"一段连续 user + 紧随的非 user"才合并（D29），而**折叠会把 `system` 变成那段 user 里的
一条**、从而改变**哪一条是"最后一条"**（D32 只配最后一条）⇒ 块的个数与内容都会变。

> 逐数据集的标签表、三个取值的边界、`alluser` 为什么进不了缺省值，都在
> [`add_shape.py`](./add_shape.py) 的模块 docstring 里——**本文件不复制那张表**。


> **"打 HTTP、不 import `src/`"这条边界在 [`../CLAUDE.md`](../CLAUDE.md)，本目录不重复。**

---

## ⚠ 与归档 pipeline 有关的三件事

### 1. `api_config.py` —— ✅ 已实现，见 [`api_config.py`](./api_config.py)

**7 个 pipeline 都做这两件事**：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                        JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)
```

D35 后官方脚本位于 `dataset/.upstream/aml/`，`parents[2]` 指向数据根目录。
旧归档在 `benchmark_data/` 时它指向仓库外；两种布局都继续靠 harness 注入的
`PYTHONPATH` 找到本目录的配置，不依赖那条脆弱路径。

**⚠ 不要在仓库外创建它。** 那份就是本目录里的 [`api_config.py`](./api_config.py)（AML 公开的那份是 520 字节、无凭据），由 `judge.run_judge()` 起 subprocess 时把**本目录**放进 `PYTHONPATH` 找到（`judge._subprocess_env`）——`sys.path.insert(0, <不存在路径>)` 只是塞进一个没有该模块的条目，**import 会继续往后找到 `PYTHONPATH` 里的那份**，于是归档保持只读、`parents[2]` 那条脆弱路径被绕开、配置只有 `.env` 一份。

**两边的名字不一样**——上游读 `ANSWER_*` / `JUDGE_*`，而我们 `.env` 里是 `AML_*` 那一组：**适配器负责接上**（`JUDGE_*` 留空即回落 `ANSWER_*`——网关只有一个对话模型）。**那七个名字里不含 embedding**：归档 pipeline 不向量化，Qwen3-Embedding-8B 只属于 `src/tianximem`。

> **回归用例**：`tests/test_harness.py` 里那个桩 pipeline **真的 `import api_config`**——这条路径断了会立刻红。

### 2. 那些 CLI 参数是死的

`--model` / `--base-url` / `--api-key-env` 看着像配置入口，**实际上被覆盖**：`answer()` 与 `evaluate()` 在协程开头**用 `api_config` 的常量重写 `args.*`**。`--api-key-env` 的默认值（`OPENAI_API_KEY` / `SILICONFLOW_API_KEY`）**从未被读取**。

**所以模型控制只能走 `api_config.py`。** 别在 `Makefile` 或 runner 里传 `--model` 然后困惑于它没生效。

> `JUDGE_VERSION` 被 **7 个** pipeline 全部 import，但**没有任何一处使用它**。

### 3. pipeline 不可 import，只能当脚本跑

`pipeline_locomo-refined.py` 的**文件名里带连字符**，**无法 `import`**。它和 `clb_pipeline.py` 都是带 argparse 子命令的独立脚本：

```bash
python pipeline_locomo-refined.py answer   --input ... --output ...
python pipeline_locomo-refined.py evaluate --input ... --answers ... --output ...
```

**harness 要么 `subprocess` 调它，要么复制它的逻辑**（后者会让"契约以 pipeline 代码为准"这条失去意义）。**推荐 subprocess。**

---

## 契约：字段名以 **pipeline 代码**为准，不要照 readme

> ⚠ 归档的 7 个 pipeline 有一处**已声明的本地修订**（`Path.open` 当异步上下文用的语法问题，
> `contextlib.nullcontext` 包一层）——**prompt 与判分逻辑与上游逐字相同**，差的是文件句柄那一处。
> 理由、范围与复现方式见 [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md) 的"本地修订"一节，
> 机器可查的部分在 [`../../eval/datasets/manifest.py`](../../eval/datasets/manifest.py) 的 `local_patch`。
> **"以 pipeline 代码为准"照旧。**

**readme 与代码不一致，已核实**：readme 写 `predicted_answer`（LoCoMo）/ `hypothesis`（LME），但**这些 pipeline 实际读写的是 `generated_answer`**。

| pipeline | answer 步写入 | evaluate 步读取 | question 键 | gold 键 | id 键 |
|---|---|---|---|---|---|
| `pipeline_locomo-refined.py` | `generated_answer` | `item["generated_answer"]` | `item["question"]` | 四选一：`gold_answer`/`golden_answer`/`reference_answer`/`correct_answer` | `item["id"]` |
| `pipeline_longmemeval-s.py` | `generated_answer` | `item["generated_answer"]` | `item["question"]` | 同上（四选一，同一元组） | `item["id"]` |
| `clb_pipeline.py` | **`model_output`** | `model_output`，**回退** `generated_answer` | `item.get("question")` | n/a（rubric 判分） | `row_id()` 多级回退 |
| `pipeline_beam.py` | `generated_answer` | `item["generated_answer"]` | —— | —— | `item["id"]` |
| `pipeline_v2_personamem.py` | `generated_answer` | `row.get("generated_answer")` | —— | `mapping[row["correct_letter"]]` | —— |
| `pipeline_v1_personamem.py` | `generated_answer` | `generated_answer` → `prediction` | `item["question"]` + `item["all_options"]` | **`item["correct_answer"]`**（⚠ **与 v2 的 `correct_letter` 不同**） | `row_id()` 多级回退 |
| `pipeline_scriptmem.py` | `generated_answer` | `generated_answer` → `predicted_answer` → `prediction` | —— | `record["answer"]` | —— |

**三条可直接照做的结论：**

1. **stage 之间的规范字段是 `generated_answer`**（七个里六个如此；CL-Bench 写 `model_output`，读时兼容两者）
2. **`hypothesis` 没有任何 pipeline 读它**——尽管 `lme_readme.md` 明文文档化了它
3. **`predicted_answer` 从来不是 inter-stage 的键**（只在 CL-Bench 里是个**函数参数名**，在 PersonaMem / ScriptMem 里是 evaluate 步**写出**的字段）

### PersonaMem-v2 的本地检索输入（2026-10-06）

归档原版只读 `chat_history/messages`。本地选择题适配器把每题 `retrieved_context`
（兼容旧 `speaker_1_memories`）转成一条 user 历史消息，再调用官方选项构造与 MCQ 判分。
保留 Search 顺序；空检索不回填原始对话，缺字段直接失败。
⚠ **平台 token 前缀截断在这条路径上还没接**（[`judge.py`](./judge.py) 的 `_build_personamem_items`
直接 `render_memories`；其余适配器都截）——**待补**。
这属于本地答案输入口径的显式改动，不能认定 AML 线上也是同一输入方式。
旧完整历史答案与成绩不能直接复用；适配器校验输入版本和指纹，重跑使用新 run-id。
验证与复现见 [`../reports/personamem-pipeline-fix-20261006.md`](../reports/personamem-pipeline-fix-20261006.md)。

### HybridQA / FEVEROUS 的本地文本检索适配

两份走 [`corpusqa_pipeline.py`](./corpusqa_pipeline.py)，要求显式 `answer_contract`，
只有逐题 Search 文本进入回答 prompt；FEVEROUS 只接受其中可见的元素 ID。
FEVEROUS 的 `feverous-claim-pages-evidence-v3` 对无效 JSON、标签或不可见 ID
最多做一次模型校正：仍只用相同 Search 文本、上次输出及格式错误，保留每次输出，
从同一文本列出与非法 ID 拼写相近的可见 ID，不自动替换引用，不读 gold 或评分反馈；
校正后仍无效则按原规则记 `ANSWER_ERROR`。
整页长上下文的单次作答超时为 360 秒；该设置仅属于本地 FEVEROUS 管线。
答案记录完整输入指纹，输入变更后不能继续复用旧答案。
上游评分源码保持原字节并校验哈希，通过函数节点调用避免执行其命令行入口。
逐题 `metrics` 经 `run_record.py` 汇总到 `scores.dataset_score`；
FEVEROUS 的严格分包含完整证据组要求，采集重放仍按其原有 label-only 口径。
输入范围、聚合指标及验证见 [接入报告](../reports/hybridqa-feverous-pipelines-20261006.md)。
官方采集重放入口已经把 Search 结果注入开放题 prompt，不受选择题入口的修复影响。

`aml-v1` 使用独立的 FEVEROUS JSON 证据契约；完整可解析的 Search 页面可提供原生 ID，
不完整页面不回填，原生括号 ID 契约保持原逻辑。公共事件分派不重塑采集请求。
支持范围、页面池来源和续跑纪律见
[`../../docs/benchmark-data.md`](../../docs/benchmark-data.md) 的「公开数据的 AML 输入适配」。

### MedMemoryBench native 临床多跳作答

`extra_pipeline.py` 对输入 `category=multi_hop_clinical_deduction` 使用病史依据与
推理格式，仍只将问题和 Search 文本放入提示词。格式路由不读 gold；没有该公开
分类字段的采集请求保留原有入口。诊断、适用范围和原始就诊对照见
[临床多跳报告](../reports/clinical-multihop-20261007.md)。

这类答案记录 `answer_contract`、`input_fingerprint` 和实际答案输出预算。
提示词、问题、Search 文本、答案端点/模型或输出预算变化后，`answer` 拒绝续用旧答案，
应使用新 run-id；`evaluate` 也核验答案来源，答案预算从答案记录读取，和裁判预算分开。
CLI 的 `--max-tokens` 仍按请求执行，其他题类的作答与旧答案续跑方式保留。

MedMemoryBench 的 LLM 裁判对混用 note/reason 字符串引号做语法修复，仍只接受
显式布尔裁决；原响应与重新解析的结果分别归档。具体来源与校验同见上述报告。

### 两个不对称，会改变结果

| pipeline | 行为 |
| --- | --- |
| `pipeline_locomo-refined.py` | `evaluate` 对 input/answer 的 **ID 集合不一致直接 `raise SystemExit`**——**硬失败** |
| `clb_pipeline.py` | `evaluate` 用 `answers.get(ident, {})`，**缺答案一律判 0** |

**同一个 harness 缺陷（漏跑几题）在两份数据集上表现完全不同**：一份崩掉，一份静默掉分。**别把后者当成"模型变差了"。**

### 其它已核实的调用细节

| 项 | 值 |
| --- | --- |
| HTTP 形状 | `POST {base_url}/chat/completions`，`Authorization: Bearer {key}` |
| `temperature` | **硬编码 `0`**（即使 `clb_pipeline.py` 收 `--temperature` 也不生效） |
| `messages` | 恒为单条 `{"role": "user", ...}` |
| timeout | locomo 硬编码 `120`；clb 默认 `180.0` |
| 输出目录 | **clb 会 `mkdir(parents=True)`，locomo 不会**——locomo 需要 harness 先建目录 |
| 续跑 | 两者 `answer` 步都会跳过已完成的 id 并以**追加**模式打开输出；`evaluate` 步以 `"w"` **覆盖**打开 |

### ⚠ 续跑的陷阱：**被强杀过的 run 会在同一样本上永久卡住**（2026-09-26，实测耗掉一小时）

链条是这样的，**每一步都不报错到最后一步**：

1. 判分侧**一次瞬时网络抖动**（实测经本地代理时 TLS 握手偶发失败）把 runner 打死；
2. 死的那一刻 pipeline 正以**追加**模式写 `answers.jsonl`（逐行 `write` + `flush`）
   ⇒ 文件末尾留下**半行 JSON**；
3. 而 `answer` 步开头**要读它**来跳过已完成的题
   （`done = {item["id"] for item in rows(output)}`）⇒ `JSONDecodeError`；
4. ⇒ **之后每次续跑都在同一个样本上确定性失败**——现象是"**卡在同一道题**"，
   而它**看起来仍像网络问题**（第一次确实是），于是排查方向整个跑偏。

**处置（已落地，都在 [`judge.py`](./judge.py)）**：

| 症状 | 修法 |
| --- | --- |
| 半行 JSON 让续跑卡死 | `_sanitize_jsonl()`——每次 `answer` 之前**丢掉末尾不完整的行**（⚠ **只在末尾删**：中途的坏行说明别的问题，不该被静默吞掉），并**就地转义**会被 `splitlines()` 劈开的行分隔符 |
| 一次抖动打死整轮 | `_run(..., attempts=3)`——**子命令级退避重试**。安全，因为两个子命令都幂等（`answer` 追加 + 跳过已完成、`evaluate` 覆盖）。**重试有界**，失败信息原样带出去 |

> **另一条同源的运维事实**：`HTTPS_PROXY` 指向本地代理时，**打网关的那条路也在走代理**
> （只把 `localhost` 放进 `NO_PROXY` 是不够的）⇒ 把网关域名也加进 `NO_PROXY`。
> 实测直连可用且小包更快。⚠ 这条属于**环境**，不是代码。

---

## 切批模拟（§6.5 / §12.3 第 5 条）

**本地两条预算都实现**（20 条消息 + 2,000 词）——词数用**空白分词近似**（线上按"冻结 Adapter 计的词"，那个组件本仓没有）⇒ **本地测出的批界与线上不保证一致**（口径与后果见 [`batching.py`](./batching.py) 与 §12.3 第 5 条）。

**为什么这件事 D24 之后仍然重要**：组合的边界是**一次 Add**，所以**批界落在哪里直接决定有多少 QA 对被切成两个半块**（实测 LoCoMo 全量 63/3,075）。而批界**线上是 AML 造的、本地是我们造的** ⇒ "切批会不会打断对"**只能在 Smoke 上用真实的 20 条复现**，本地测不出来。

---

## 裁判：两个 prompt 是**故意互相矛盾**的

**这是这两份文件的真实属性**，不是 bug，但会让结果反直觉：

| | 说法 |
| --- | --- |
| **答案 prompt 第 7 条** | Convert relative times like `"yesterday"`, `"last month"`, and `"last year"` into dates, months, or years **when the memory timestamp makes it clear**. **Keep week-based expressions relative.** |
| **裁判 prompt TIME 块** | **Do NOT convert relative ↔ absolute.** If the gold uses a relative time expression, the generated answer **must also use a relative form**…not a computed date/range |

**答案阶段被要求做的事，正是裁判阶段被判负的事**——除非 gold 本身也是绝对时间。**这就是根 `CLAUDE.md` 那条"不要在 content 里注入绝对时间戳"的实证来源**，也是 §13 的 T1 实验要测的东西。

**判分输出契约**：裁判返回一句话解释 + `CORRECT`/`WRONG`，包成 JSON `{"label": ...}`；解析器正则取**第一个 `{...}`** 并要求 label ∈ {CORRECT, WRONG}。**JSON 解析失败时要看 `clb_pipeline.py` 的纪律——API/JSON 失败一律记 0**（§12.4）——那是 CL-Bench 的严格性；**本仓的处置已定并落档**：`JUDGE_ERROR` 与 `WRONG` 分开写、分数口径不变（[`../../docs/decisions.md`](../../docs/decisions.md) 的 **D34**）。

**裁判模型**：readme 记为 `Qwen/Qwen3-14B`，temperature `0.0`。
