# benchmark_data/ — 归档说明

> 最后核对：2026-09-23，对应 PRD §12.2 / §12.3 / §12.5 与附录 B。**冲突以 PRD 为准。**

## 这个目录是什么

**AML 官方 pipeline 源码 + 数据集 + 各数据集 readme 的只读归档。**PRD 附录 B 的"各数据集问题类型分布、跨 session 比例、许可证"一行的来源就是这里。

**⚠ 本目录整目录被 `.gitignore` 排除**（见 `.gitignore` 的"其他"段）。因此：

- **不要在这里放 README 或笔记**——不会被提交，别人看不到
- **新成员 clone 仓库后拿不到这个目录**，需要单独取回

**它是只读的**：harness 只读它，**不写回**（§12.2）。

---

## 哪些文件是真数据，哪些不是

**这一节是为了防止有人把残留文件当成数据集反复排查。**

| 文件 | 是不是数据 |
| --- | --- |
| `lme_s_cleaned.json` | ✅ **用这个**（LongMemEval） |
| `lme_test.json` | ⚠️ 是数据但**不用**——多出 1,230 个空 session + 15 个干扰 session，**会污染按 20 条切批的埋点**（§6.5） |
| `questions.jsonl` | ✅ LoCoMo-Refined 题目（1,382 题），**含 `evidence_messages`（只有证据轮）** |
| `locomo_refined.json` | ✅ LoCoMo-Refined 全文（10 个 conversation），**含 `conversation` 整段对话** |
| `locomo10.json` | 原始 LoCoMo |
| `clbench.jsonl` | ✅ CL-Bench（真 JSONL） |
| `pm_32k.csv` / `pm_questions_128k.csv` / `pm_questions_1M.csv` | PersonaMem v2 三个 split |
| `scriptmem_q.jsonl` | ScriptMem **题目**（457 道 MCQ：Single Choice / Multi-Select / Ordering，4 个 script / 6 种题型）。**对话原文因版权未发布** |
| `beam.json` / `beam_rows.json` | ❌ **失败下载的残留**（`Entry not found` / `{"error":"Unexpected error."}`） |
| `beam_100k.json` | ❌ **不是数据集**——是 HuggingFace datasets-server 的**分页响应**（顶层键 `features`/`rows`/`num_rows_total`，且 `num_rows_total=20`、**实际只取到 1 行**） |
| `rh.md` | ❌ **不是数据**——14 字节的桩文件，全文只有 `404: Not Found`（失败下载的占位） |
| `rh2.md` / `rh3.md` | ⚠️ **不是 AML 材料**——它们是第三方系统 **MemoryHub** 自己的说明与**跑分结果**（R@k / P@k / MRR / NDCG / latency 与各次 run 的清单）。**与 AML 的数据集 schema 无关，不要当成数据集的读取格式依据** |

**结论（§12.3 第 8 条）**：**BEAM 的数据实际上不在归档里**——§12.4 的 BEAM 一行**只有 `pipeline_beam.py` 单方依据，没有数据可交叉核对**。真要覆盖 BEAM，**须先把数据取回来**。

---

## pipeline 源码

```text
pipeline_locomo-refined.py     LoCoMo-Refined / LongMemEval 共用契约的代表
clb_pipeline.py                CL-Bench
pipeline_beam.py               BEAM
pipeline_scriptmem.py          ScriptMem
pipeline_v2_personamem.py      PersonaMem v2
```

**harness 的 I/O 以这些代码为准，不要照 readme**（§12.3 第 9 条）：readme 写 `predicted_answer` / `hypothesis`，而代码实际读写的是 **`generated_answer`**。

**踩坑要点已整理到**：[`../eval/datasets/CLAUDE.md`](../eval/datasets/CLAUDE.md)（schema 落差、分类 ID 映射、切批口径）与 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md)（`api_config` 依赖、字段名表、调用细节）。

> **一个容易漏的依赖**：五个 pipeline 都从 `Path(__file__).resolve().parents[2]`（即 **`/home/buptc/project/`**，**仓库外**）import `api_config`，而**该文件目前不存在**。详见 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md) §"三个必须先解决的阻塞项"。

---

## 许可证（2026-09-22 核对，2026-09-23 回原文验证）

| 数据集 | 许可 | 归档内的原文依据 |
| ---- | ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** | `locomo_refined_readme.md:9` 徽章 + `:296` 正文："LoCoMo-Refined is released under **CC BY-NC 4.0**. This benchmark modifies the original LoCoMo benchmark; see `NOTICE` for attribution and modification details." |
| ScriptMem | **CC BY-NC 4.0** | `scriptmem_readme.md:9` + `:183` |
| PersonaMem v2 | **CC BY 4.0** | `pmv2.md:2` 的 YAML frontmatter：`license: cc-by-4.0` |
| LongMemEval | **待确认** | ❌ **`lme_readme.md` 无 License 章节**（`grep -ni licen` 零命中）——**只有 Citation/bibtex 块**。归档内无依据 |
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
| LongMemEval 的 MIT 许可 | 归档内无依据 |
| AML 数据条款原文（禁止训练） | 归档内无依据 |
| CLBench 许可证文本 | 归档内无 |
| **"`speaker_1_memories` 等字段的使用方式"** | ✅ 这部分有依据——见 `pipeline_locomo-refined.py` 的渲染函数 |
| **检索结果 → `memories` 字段的映射** | ❌ **在 AML 那一侧，归档里看不到**（§11.3）——这正是 S1 只能靠 Smoke 消除的原因 |

**另外**：LongMemEval 的对应 pipeline 文件**没有归档**——PRD 里"LoCoMo-Refined / LongMemEval 契约相同"这条断言，**本地只有 LoCoMo 那一份文件作证，属单边来源**（§11.3）。
