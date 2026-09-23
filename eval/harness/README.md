# eval/harness/ — 一轮评测怎么跑

**PRD**：§12.1（三层结构）、§12.3（第 5 / 9 条）、§6.5（切批）、§13（记录）

## 要写什么

```text
driver.py      喂 Add、调 Search（**走 HTTP**）、收集返回
batching.py    模拟 AML 切批（20 条那一路）
judge.py       包住 AML pipeline 的裁判
run_record.py  每次 run 的配置指纹 + 数据指纹 + 结果
```

---

## ⚠ 三个必须先解决的阻塞项

### 1. `api_config.py` 不存在，而且它在仓库**外面**

五个 pipeline 都做这两件事：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL,
                        JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)
```

`__file__` 在 `benchmark_data/` 下，所以 `parents[2]` 解析到 **`/home/buptc/project`**——**不是 `TianXi_AM/`，是它的上一级**。

**该文件当前不存在**（已在 `/home/buptc` 下全盘查找确认），所以**五个 pipeline 今天都 import 失败**。

**处置**：Step 0 必须创建 `/home/buptc/project/api_config.py`，导出上述七个名字。它是**仓库外的环境依赖**，写 README 里不够——**要进 `docs/roadmap.md` 的 Step 0 清单，并在 `docs/decisions.md` 记一条**（因为它意味着"clone 下来就能跑"不成立）。

### 2. 那些 CLI 参数是死的

`--model` / `--base-url` / `--api-key-env` 看着像配置入口，**实际上被覆盖**：`answer()` 与 `evaluate()` 在协程开头**用 `api_config` 的常量重写 `args.*`**。`--api-key-env` 的默认值（`OPENAI_API_KEY` / `SILICONFLOW_API_KEY`）**从未被读取**。

**所以模型控制只能走 `api_config.py`。** 别在 `Makefile` 或 runner 里传 `--model` 然后困惑于它没生效。

> `JUDGE_VERSION` 被五个 pipeline 全部 import，但**没有任何一处使用它**。

### 3. pipeline 不可 import，只能当脚本跑

`pipeline_locomo-refined.py` 的**文件名里带连字符**，**无法 `import`**。它和 `clb_pipeline.py` 都是带 argparse 子命令的独立脚本：

```bash
python pipeline_locomo-refined.py answer   --input ... --output ...
python pipeline_locomo-refined.py evaluate --input ... --answers ... --output ...
```

**harness 要么 `subprocess` 调它，要么复制它的逻辑**（后者会让"契约以 pipeline 代码为准"这条失去意义）。**推荐 subprocess。**

---

## 契约：字段名以 **pipeline 代码**为准，不要照 readme

**readme 与代码不一致，已核实**：readme 写 `predicted_answer`（LoCoMo）/ `hypothesis`（LME），但**这四个 pipeline 实际读写的是 `generated_answer`**。

| pipeline | answer 步写入 | evaluate 步读取 | question 键 | gold 键 | id 键 |
|---|---|---|---|---|---|
| `pipeline_locomo-refined.py` | `generated_answer` | `item["generated_answer"]` | `item["question"]` | 四选一：`gold_answer`/`golden_answer`/`reference_answer`/`correct_answer` | `item["id"]` |
| `clb_pipeline.py` | **`model_output`** | `model_output`，**回退** `generated_answer` | `item.get("question")` | n/a（rubric 判分） | `row_id()` 多级回退 |
| `pipeline_beam.py` | `generated_answer` | `item["generated_answer"]` | —— | —— | `item["id"]` |
| `pipeline_v2_personamem.py` | `generated_answer` | `row.get("generated_answer")` | —— | `mapping[row["correct_letter"]]` | —— |
| `pipeline_scriptmem.py` | `generated_answer` | `generated_answer` → `predicted_answer` → `prediction` | —— | `record["answer"]` | —— |

**三条可直接照做的结论：**

1. **stage 之间的规范字段是 `generated_answer`**（五个里四个如此；CL-Bench 写 `model_output`，读时兼容两者）
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

---

## 切批模拟（§6.5 / §12.3 第 5 条）

**本地只能按 20 条复现。**

AML 按"**20 条消息 或 2,000 个 Adapter 计数的词**"确定性切分，而 **"Adapter" 官方从未定义**。

> **因此本地测出的 `pending` 埋点数与线上必然对不上**，解读那三个计数器时必须记住（`docs/open-questions.md` S2，Step 6 才能清掉）。

**harness 的切批必须与 §6.5 的续接逻辑配套测**：喂进去的批次边界是**我们自己造的**，而线上是 AML 造的——**所以"续接逻辑自洽"能在本地验证，"切批会不会打断对"不能。**

---

## 裁判：两个 prompt 是**故意互相矛盾**的

**这是这两份文件的真实属性**，不是 bug，但会让结果反直觉：

| | 说法 |
| --- | --- |
| **答案 prompt 第 7 条** | Convert relative times like `"yesterday"`, `"last month"`, `"last year"` into dates, months, or years **when the memory timestamp makes it clear**. **Keep week-based expressions relative.** |
| **裁判 prompt TIME 块** | **Do NOT convert relative ↔ absolute.** If the gold uses a relative time expression, the generated answer **must also use a relative form**…not a computed date/range |

**答案阶段被要求做的事，正是裁判阶段被判负的事**——除非 gold 本身也是绝对时间。**这就是 §11.3 那条"不要在 content 里注入绝对时间戳"的实证来源**，也是 T1 实验要测的东西。

**判分输出契约**：裁判返回一句话解释 + `CORRECT`/`WRONG`，包成 JSON `{"label": ...}`；解析器正则取**第一个 `{...}`** 并要求 label ∈ {CORRECT, WRONG}。**JSON 解析失败时要看 `clb_pipeline.py` 的纪律——API/JSON 失败一律记 0**（§12.4）——那是 CL-Bench 的严格性，**代理评测里要显式决定是否照此办理，并记进 `docs/decisions.md`。**

**裁判模型**：readme 记为 `Qwen/Qwen3-14B`，temperature `0.0`。

---

## 一条纪律

**harness 打 HTTP，不 import `src/`。** 理由见 [`../README.md`](../README.md)——B1 只能是"另一个服务"，A1 必须验证契约层，两者都只有在进程间调用下才成立。

**顺带**：这也让"契约违规"在本地就被抓到（比如返回超过 `top_k`），**而不是等 Smoke 那 30 次配额**。
