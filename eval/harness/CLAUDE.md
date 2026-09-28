# eval/harness/ — 一轮评测怎么跑

**PRD**：§12.1（三层结构）、§12.3（第 5 / 9 条）、§6.5（切批）、§13（记录）

## 要写什么

```text
driver.py      喂 Add、调 Search（**走 HTTP**）、收集返回
batching.py    模拟 AML 切批（20 条那一路）
judge.py       包住 AML pipeline 的裁判
run_record.py  每次 run 的配置指纹 + 数据指纹 + 结果
```

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

`__file__` 在 `benchmark_data/` 下，所以 `parents[2]` 解析到 **`/home/buptc/project`**——**不是 `TianXi_AM/`，是它的上一级**。

**⚠ 不要在仓库外创建它。** 那份就是本目录里的 [`api_config.py`](./api_config.py)（AML 公开的那份是 520 字节、无凭据），由 `judge.run_judge()` 起 subprocess 时把**本目录**放进 `PYTHONPATH` 找到（`judge._subprocess_env`）——`sys.path.insert(0, <不存在路径>)` 只是塞进一个没有该模块的条目，**import 会继续往后找到 `PYTHONPATH` 里的那份**，于是归档保持只读、`parents[2]` 那条脆弱路径被绕开、配置只有 `.env` 一份。

**两边的名字不一样**——上游读 `ANSWER_*` / `JUDGE_*`，而我们 `.env` 里是 `AML_*` 那一组：**适配器负责接上**（`JUDGE_*` 留空即回落 `ANSWER_*`——网关只有一个对话模型）。**那七个名字里不含 embedding**：归档 pipeline 不向量化，Qwen3-Embedding-8B 只属于 `src/tianxi_am`。

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
> 机器可查的部分在 [`../../tools/fetch_benchmark_data.py`](../../tools/fetch_benchmark_data.py) 的 `local_patch`。
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
| 半行 JSON 让续跑卡死 | `_drop_trailing_partial_line()`——每次 `answer` 之前**丢掉末尾不完整的行**（⚠ **只在末尾删**：中途的坏行说明别的问题，不该被静默吞掉） |
| 一次抖动打死整轮 | `_run(..., attempts=3)`——**子命令级退避重试**。安全，因为两个子命令都幂等（`answer` 追加 + 跳过已完成、`evaluate` 覆盖）。**重试有界**，失败信息原样带出去 |

> **另一条同源的运维事实**：`HTTPS_PROXY` 指向本地代理时，**打网关的那条路也在走代理**
> （只把 `localhost` 放进 `NO_PROXY` 是不够的）⇒ 把网关域名也加进 `NO_PROXY`。
> 实测直连可用且小包更快。⚠ 这条属于**环境**，不是代码。

---

## 切批模拟（§6.5 / §12.3 第 5 条）

**本地只能按 20 条复现**——原因见 [`../datasets/CLAUDE.md`](../datasets/CLAUDE.md) 与 §12.3 第 5 条（本目录不重复）。

**为什么这件事 D24 之后仍然重要**：组合的边界是**一次 Add**，所以**批界落在哪里直接决定有多少 QA 对被切成两个半块**（实测 LoCoMo 全量 63/3,075）。而批界**线上是 AML 造的、本地是我们造的** ⇒ "切批会不会打断对"**只能在 Smoke 上用真实的 20 条复现**，本地测不出来。

---

## 裁判：两个 prompt 是**故意互相矛盾**的

**这是这两份文件的真实属性**，不是 bug，但会让结果反直觉：

| | 说法 |
| --- | --- |
| **答案 prompt 第 7 条** | Convert relative times like `"yesterday"`, `"last month"`, `"last year"` into dates, months, or years **when the memory timestamp makes it clear**. **Keep week-based expressions relative.** |
| **裁判 prompt TIME 块** | **Do NOT convert relative ↔ absolute.** If the gold uses a relative time expression, the generated answer **must also use a relative form**…not a computed date/range |

**答案阶段被要求做的事，正是裁判阶段被判负的事**——除非 gold 本身也是绝对时间。**这就是根 `CLAUDE.md` 那条"不要在 content 里注入绝对时间戳"的实证来源**，也是 §13 的 T1 实验要测的东西。

**判分输出契约**：裁判返回一句话解释 + `CORRECT`/`WRONG`，包成 JSON `{"label": ...}`；解析器正则取**第一个 `{...}`** 并要求 label ∈ {CORRECT, WRONG}。**JSON 解析失败时要看 `clb_pipeline.py` 的纪律——API/JSON 失败一律记 0**（§12.4）——那是 CL-Bench 的严格性，**代理评测里要显式决定是否照此办理，并记进 `docs/decisions.md`。**

**裁判模型**：readme 记为 `Qwen/Qwen3-14B`，temperature `0.0`。
