# benchmark_data/ — 归档说明

> 最后核对：2026-09-24，对应 PRD §12.2 / §12.3 / §12.5 与附录 B。**冲突以 PRD 为准。**

## 这个目录是什么

**AML 官方 pipeline 源码 + 数据集 + 各数据集 readme 的只读归档。**PRD 附录 B 的"各数据集问题类型分布、跨 session 比例、许可证"一行的来源就是这里。

**⚠ 本目录整目录被 `.gitignore` 排除**（见 `.gitignore` 的"其他"段）。因此：

- **不要在这里放 README 或笔记**——不会被提交，别人看不到
- **新成员 clone 仓库后拿不到这个目录**，需要单独取回 —— **取回办法见下节**

**它是只读的**：harness 只读它，**不写回**（§12.2）。

---

## 出处链：字节从哪来（2026-09-24 核对）

**除了 `rh.md` / `rh2.md` / `rh3.md` 三份，每一份都有确定的出处，且都能按 commit / revision 钉死。**
（2026-09-24 全部逐字节核对通过：**24 份里 21 份有出处，3 份取不回来**。）机器可查的清单（出处 + sha256）在
[`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py)——**哈希只写那一处，本文不重复**。
取回 `make fetch-data`，校验 `make data-check`。

| 内容 | 出处（在哪） | 钉法 |
| --- | --- | --- |
| —— | **具体 commit / revision 值与 sha256 一律不在此处重复**，见 [`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py) | —— |
| 7 个 `pipeline_*.py` / `clb_pipeline.py`、`aml_readme.md` | [github.com/AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard) 的 `data/*/pipeline.py` + 根 `README.md` | 按 commit 钉死 |
| `locomo_refined.json`、`questions.jsonl`、`locomo_refined_readme.md` | [github.com/mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined) | 按 commit 钉死 |
| `scriptmem_q.jsonl`、`scriptmem_readme.md` | [github.com/memorax-ai/ScriptMem](https://github.com/memorax-ai/ScriptMem) | 按 commit 钉死 |
| `lme_s_cleaned.json` | HF [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 的 `longmemeval_s_cleaned.json` | 按 revision 钉死。**sha256 = HF 的 LFS oid**（2026-09-24 实测相等） |
| `clbench.jsonl` | HF [`tencent/CL-bench`](https://huggingface.co/datasets/tencent/CL-bench) 的 `CL-bench.jsonl` | 按 revision 钉死。sha256 = LFS oid |
| `locomo10.json` | [github.com/snap-research/locomo](https://github.com/snap-research/locomo) 的 `data/locomo10.json` | 按 commit 钉死 |
| `pm_32k.csv`、`pm_questions_128k.csv`、`pm_questions_1M.csv` | HF [`bowen-upenn/PersonaMem-v1`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) 的 `questions_{32k,128k,1M}.csv` | 按 revision 钉死。⚠ **不是 v2 仓**——v2 仓里只有 `benchmark/*.csv`，本地文件名是被改过的 |
| `pmv2.md`、`lme_readme.md` | HF `bowen-upenn/PersonaMem-v2` 的 `README.md`；[github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) 的 `README.md` | 分别按 revision / commit 钉死 |
| `rh.md`、`rh2.md`、`rh3.md` | ❌ 抓取时原始 URL 已 404，是搜索引擎索引副本 | **取不回来**，只作存档 |

**⇒ 因此"所有人都需要的那份"不需要住在任何人的电脑上，也不需要任何共享盘。**
字节各自从公开源取，清单进版本库——这同时满足"不打搅同事的机器"与"不外传"。

### 两条不外传的理由（引用前必读）

1. **AML 仓没有 LICENSE**，而且它的 README 有一节 *Deliberately not included*：不发布语料与金标，并写明
   **"Do not submit any of these materials in an issue or pull request."**
2. **第三方数据带 CC BY-NC 等约束**（见下方许可表），分发范围未确认前同样不提交。

**⇒ `benchmark_data/` 保持 `.gitignore` 排除，这是刻意状态，不要改成提交进去。**
这两条不构成"不能各自取回"：数据从上游直接下载，**不经过我们**。

---

## 哪些文件是真数据，哪些不是

**这一节是为了防止有人把残留文件当成数据集反复排查。**

| 文件 | 是不是数据 |
| --- | --- |
| `lme_s_cleaned.json` | ✅ **用这个**（LongMemEval） |
| `lme_test.json` | 🗑 **已于 2026-09-24 删除**（266MB）——它是"用得上但**明令别用**"的陷阱：多出 1,230 个空 session + 15 个干扰 session，**会污染按 20 条切批的埋点**（§6.5）。要复核那 1,230 这个数字时，从上游 LongMemEval 取回（**出处未核**，见 `tools/fetch_benchmark_data.py` 的 `DELETED`） |
| `questions.jsonl` | ✅ LoCoMo-Refined 题目（1,382 题），**含 `evidence_messages`（只有证据轮）** |
| `locomo_refined.json` | ✅ LoCoMo-Refined 全文（10 个 conversation），**含 `conversation` 整段对话** |
| `locomo10.json` | 原始 LoCoMo |
| `clbench.jsonl` | ✅ CL-Bench（真 JSONL） |
| `pm_32k.csv` / `pm_questions_128k.csv` / `pm_questions_1M.csv` | PersonaMem 三个 split 的问题表。**来源是 v1 仓的 `questions_{32k,128k,1M}.csv`**（2026-09-24 逐字节核对）——⚠ 原先记作"v2 三个 split"，**v2 仓里没有这三份** |
| `scriptmem_q.jsonl` | ScriptMem **题目**（457 道 MCQ：Single Choice / Multi-Select / Ordering，4 个 script / 6 种题型）。**对话原文因版权未发布** |
| `beam.json` / `beam_rows.json` | 🗑 **已于 2026-09-24 删除**——**失败下载的残留**（15 字节 `Entry not found` / 29 字节 `{"error":"Unexpected error."}`）。留着只会让下一个人重排一遍 |
| `beam_100k.json` | 🗑 **已于 2026-09-24 删除**——**不是数据集**，是 HuggingFace datasets-server 的**分页响应**（顶层键 `features`/`rows`/`num_rows_total`，且 `num_rows_total=20`、**实际只取到 1 行**） |
| `rh.md` | ❌ **不是数据**——14 字节的桩文件，全文只有 `404: Not Found`（失败下载的占位） |
| `rh2.md` / `rh3.md` | ⚠️ **不是 AML 材料**——它们是第三方系统 **MemoryHub** 自己的说明与**跑分结果**（R@k / P@k / MRR / NDCG / latency 与各次 run 的清单）。**与 AML 的数据集 schema 无关，不要当成数据集的读取格式依据** |

**结论（§12.3 第 8 条）**：**BEAM 的数据实际上不在归档里**——§12.4 的 BEAM 一行**只有 `pipeline_beam.py` 单方依据，没有数据可交叉核对**。真要覆盖 BEAM，**须先把数据取回来**。

---

## pipeline 源码

```text
pipeline_locomo-refined.py     LoCoMo-Refined / LongMemEval 共用契约的代表
pipeline_longmemeval-s.py      LongMemEval（**2026-09-24 补入**，原归档缺这一份）
clb_pipeline.py                CL-Bench
pipeline_beam.py               BEAM
pipeline_scriptmem.py          ScriptMem
pipeline_v1_personamem.py      PersonaMem v1（**2026-09-24 补入**，原归档只有 v2）
pipeline_v2_personamem.py      PersonaMem v2
```

**harness 的 I/O 以这些代码为准，不要照 readme**（§12.3 第 9 条）：readme 写 `predicted_answer` / `hypothesis`，而代码实际读写的是 **`generated_answer`**。

**踩坑要点已整理到**：[`../eval/datasets/CLAUDE.md`](../eval/datasets/CLAUDE.md)（schema 落差、分类 ID 映射、切批口径）与 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md)（`api_config` 依赖、字段名表、调用细节）。

> **`api_config` 这条依赖**：**7 个 pipeline 都** `from api_config import (...)`（七个名字：`ANSWER_API_BASE` / `_KEY` / `_MODEL`、`JUDGE_API_BASE` / `_KEY` / `_MODEL` / `_VERSION`；最后一个是死引用，import 了但从不使用）。
> **AML 自己就发布了这个文件**——公开仓根目录的 [`api_config.py`](https://github.com/AML-memory/agent-memory-leaderboard/blob/main/api_config.py)，
> **520 字节、无凭据**，只是 `os.environ.get(...)` 的适配器（README 原话："Credentials and service endpoints must be supplied externally; no secret is bundled with this repository."）。
>
> **处置（D12 后已简化）**：**不要在仓库外创建它**。把它放在仓库内、由 harness 在 subprocess 里注入 `PYTHONPATH`——
> `sys.path.insert(0, <不存在路径>)` 只是塞进一个没有该模块的条目，import 会继续往后找到 `PYTHONPATH` 里的那份。
> 好处：归档保持只读、`parents[2]` 那条脆弱路径被绕开、配置只有 `.env` 一份。细则见 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md)。

---

## 许可证（2026-09-22 核对，2026-09-23 回原文验证）

| 数据集 | 许可 | 归档内的原文依据 |
| ---- | ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** | `locomo_refined_readme.md:9` 徽章 + `:296` 正文："LoCoMo-Refined is released under **CC BY-NC 4.0**. This benchmark modifies the original LoCoMo benchmark; see `NOTICE` for attribution and modification details." |
| ScriptMem | **CC BY-NC 4.0** | `scriptmem_readme.md:9` + `:183` |
| PersonaMem v2 | **CC BY 4.0** | `pmv2.md:2` 的 YAML frontmatter：`license: cc-by-4.0` |
| LongMemEval | **MIT** | ⚠ 归档内**仍**无依据（`lme_readme.md` 无 License 章节，`grep -ni licen` 零命中）。**2026-09-24 从上游取到**：HF `xiaowu0162/longmemeval-cleaned` 数据集卡 `license: mit`，且 `github.com/xiaowu0162/LongMemEval-V2` 仓内有 `LICENSE` 文件——**两条独立出处，但都不在归档里** |
| BEAM / CL-Bench | **归档内无许可证文本** | ❌ |

### 两条必须记住的推论

1. **NC（非商业）这一列**：LoCoMo-Refined 与 ScriptMem 都是 CC BY-NC 4.0。**不影响参赛**，但**意味着这两份数据不能进任何商业用途的产物**——如果后续想把系统或其中组件开源/商用，**这两份数据的评测结果是引用不了的**。
2. **"数据不得用于训练"这条在归档里没有出处。** 全库检索 `only for the evaluation` 等措辞**零命中**；`aml_readme.md` 只说到各数据集"remain subject to their respective upstream licenses and usage terms"，**没有任何"不得训练"的措辞**；归档里也没有 CLBench 的许可证文本。**它们来自归档之外的来源，属单边来源，引用前须回原始页面/许可证文件复核。**
   > **结论不变**——即便只按 AML 的通用条款，也不该拿这些数据训练——**但不要把它当成已归档的实证**（§12.5）。

---

## 归档没覆盖到的东西（引用前注意）

**PRD 附录 B 已标出两处"单边来源"**，本目录的归档里确实找不到依据：

| 断言 | 状态 |
| --- | --- |
| LongMemEval 的 MIT 许可 | ✅ **2026-09-24 取到出处**（上游 HF 数据集卡 + LongMemEval-V2 仓 `LICENSE`），但**归档内仍无** |
| AML 数据条款原文（禁止训练） | ❌ **归档内无依据**——**2026-09-24 复核过 AML 仓 README（= `aml_readme.md`，逐字节一致）：全文无 "train" 措辞**，只有"remain subject to their respective upstream licenses and usage terms" |
| CLBench 许可证文本 | 归档内无 |
| **"`speaker_1_memories` 等字段的使用方式"** | ✅ 有依据，且**现在有两份**：`pipeline_locomo-refined.py` 与 `pipeline_longmemeval-s.py` 的渲染函数（后者确认了 `{{speaker_1_memories}}` / `{{speaker_2_memories}}` 的 `<memories>` 块形状） |
| **检索结果 → `memories` 字段的映射** | ❌ **在 AML 那一侧，归档里看不到**（§11.3）——这正是 S1 只能靠 Smoke 消除的原因 |

**另外**：LongMemEval 的 pipeline 文件**原归档缺失，2026-09-24 已从上游补入**（`pipeline_longmemeval-s.py`）——PRD 里"LoCoMo-Refined / LongMemEval 契约相同"这条断言，**因此不再只靠 LoCoMo 那一份文件作证**。
