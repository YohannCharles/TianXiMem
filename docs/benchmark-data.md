# benchmark_data/ — 归档说明

> 最后核对：2026-09-30，对应 PRD §12.2 / §12.3 / §12.5 与附录 B。**冲突以 PRD 为准。**

## 这个目录是什么

**AML 官方 pipeline 源码 + 数据集 + 各数据集 readme 的只读归档。**PRD 附录 B 的"各数据集问题类型分布、跨 session 比例、许可证"一行的来源就是这里。

**归档里有两组东西**：公开仓库那 6 个管线的材料（本文件下面各节），与 **`official-extra`**——
官方真实流量里出现过、但**不在**公开 6 管线里的 8 个数据集（见下面的专节）。

**⚠ 本目录整目录被 `.gitignore` 排除**（见 `.gitignore` 的"其他"段）。因此：

- **不要在这里放 README 或笔记**——不会被提交，别人看不到
- **新成员 clone 仓库后拿不到这个目录**，需要单独取回 —— **取回办法见下节**

**它是只读的**：harness 只读它，**不写回**（§12.2）。

---

## 出处链：字节从哪来

**除了 `rh.md` / `rh2.md` / `rh3.md` 三份，每一份都有确定的出处，且都能按 commit / revision 钉死。**
（逐字节核对通过；只有那 3 份取不回来。）机器可查的清单（出处 + sha256）在
[`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py)——**哈希只写那一处，本文不重复**。
取回 `make fetch-data`，校验 `make data-check`。

| 内容 | 出处（在哪） | 钉法 |
| --- | --- | --- |
| —— | **具体 commit / revision 值与 sha256 一律不在此处重复**，见 [`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py) | —— |
| 7 个 `pipeline_*.py` / `clb_pipeline.py`、`aml_readme.md` | [github.com/AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard) 的 `data/*/pipeline.py` + 根 `README.md` | 按 commit 钉死 |
| `locomo_refined.json`、`questions.jsonl`、`locomo_refined_readme.md` | [github.com/mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined) | 按 commit 钉死 |
| `scriptmem_q.jsonl`、`scriptmem_readme.md` | [github.com/memorax-ai/ScriptMem](https://github.com/memorax-ai/ScriptMem) | 按 commit 钉死 |
| `lme_s_cleaned.json` | HF [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 的 `longmemeval_s_cleaned.json` | 按 revision 钉死。**sha256 = HF 的 LFS oid**（实测相等） |
| `clbench.jsonl` | HF [`tencent/CL-bench`](https://huggingface.co/datasets/tencent/CL-bench) 的 `CL-bench.jsonl` | 按 revision 钉死。sha256 = LFS oid |
| `locomo10.json` | [github.com/snap-research/locomo](https://github.com/snap-research/locomo) 的 `data/locomo10.json` | 按 commit 钉死 |
| `pm_32k.csv`、`pm_questions_128k.csv`、`pm_questions_1M.csv` | HF [`bowen-upenn/PersonaMem-v1`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) 的 `questions_{32k,128k,1M}.csv` | 按 revision 钉死。⚠ **不是 v2 仓**——v2 仓里只有 `benchmark/*.csv`，本地文件名是被改过的 |
| `pmv2.md`、`lme_readme.md` | HF `bowen-upenn/PersonaMem-v2` 的 `README.md`；[github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) 的 `README.md` | 分别按 revision / commit 钉死 |
| **`official-extra` 的 8 个数据集** | 6 个 HF 数据集 + 2 个 GitHub 仓（`zjunlp/MemTrapBench`、`hwanchang00/doc-pp`），**路径清单见本文件下面的专节** | 按 revision / commit 钉死；HF 侧 `sha256 = LFS oid`（实测相等） |
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
| `lme_test.json` | 🗑 **不入档**（266MB）——它是"用得上但**明令别用**"的陷阱：多出 1,230 个空 session + 15 个干扰 session，**会污染按 20 条切批的埋点**（§6.5）。要复核那 1,230 这个数字时，从上游 LongMemEval 取回（**出处未核**，见 `tools/fetch_benchmark_data.py` 的 `DELETED`） |
| `questions.jsonl` | ✅ LoCoMo-Refined 题目（1,382 题），**含 `evidence_messages`（只有证据轮）** |
| `locomo_refined.json` | ✅ LoCoMo-Refined 全文（10 个 conversation），**含 `conversation` 整段对话** |
| `locomo10.json` | 原始 LoCoMo |
| `clbench.jsonl` | ✅ CL-Bench（真 JSONL） |
| `pm_32k.csv` / `pm_questions_128k.csv` / `pm_questions_1M.csv` | PersonaMem 三个 split 的问题表。**来源是 v1 仓的 `questions_{32k,128k,1M}.csv`**——⚠ **来源是 v1 仓**：v2 仓里没有这三份 |
| `scriptmem_q.jsonl` | ScriptMem **题目**（457 道 MCQ：Single Choice / Multi-Select / Ordering，4 个 script / 6 种题型）。**对话原文因版权未发布** |
| `beam.json` / `beam_rows.json` | 🗑 **不入档**——**失败下载的残留**（15 字节 `Entry not found` / 29 字节 `{"error":"Unexpected error."}`）。留着只会让下一个人重排一遍 |
| `beam_100k.json` | 🗑 **不入档**——**不是数据集**，是 HuggingFace datasets-server 的**分页响应**（顶层键 `features`/`rows`/`num_rows_total`，且 `num_rows_total=20`、**实际只取到 1 行**） |
| `rh.md` | ❌ **不是数据**——14 字节的桩文件，全文只有 `404: Not Found`（失败下载的占位） |
| `rh2.md` / `rh3.md` | ⚠️ **不是 AML 材料**——它们是第三方系统 **MemoryHub** 自己的说明与**跑分结果**（R@k / P@k / MRR / NDCG / latency 与各次 run 的清单）。**与 AML 的数据集 schema 无关，不要当成数据集的读取格式依据** |

**结论（§12.3 第 8 条）**：**BEAM 的数据现在在归档里**——`official-extra` 的 `beam/`（**100K 档**，2026-09-30 取回），§12.4 的 BEAM 一行因此有数据可交叉核对。
⚠ **只覆盖 100K**：500K / 1M 两档未取（官方流量里只出现过 `beam_100k`；上游的档位命名就是 `100K` 而不是论文散文里的 128K）。

---

## `official-extra`：官方流量里出现过、公开 6 管线之外的数据集

**判据与出处**：官方 2026-09-29 那轮真实流量的语料普查——见
[`../official-dataset-2026-09-29/README.md`](../official-dataset-2026-09-29/README.md) §4.2.1 / §4.2.2。
那份报告是**唯一说明**，这里只记"归档里存了什么、缺什么"。

**⚠ 三条边界，引用前必读**：

1. **它不是新的评测基线**——「代理评测只覆盖 LoCoMo-Refined + LongMemEval」那条边界（§12.4）不变；
   这批数据的用途是**复盘官方实际在跑什么**与**给采集到的官方查询补金标**。
2. **只取流量里出现过的那一档**（下表"取的档"一列）。上游还有别的档，**不要**当成已归档。
3. **能逐字匹配回官方 query 的只有一部分**：实测 TempReason 与 Doc-PP 的题面在 10,144 条官方
   search 里 **0 命中**（官方问法被改写过），**不代表这些数据集没被跑过**。

| 归档目录 | 取的档 | 是什么 |
| --- | --- | --- |
| `mquake-remastered/` | 全量（4 份） | 多跳知识更新；流量里 `Corpus: {…}` / `Corpus: UPDATE:` 两族 |
| `beam/` | **100K** | 长对话记忆 |
| `personamem-v2/` | text/benchmark 分片 + `data/chat_history_32k/` 的 200 份 | 隐式 persona；5,000 题 |
| `tempreason/` | **test_*** | 时间推理 |
| `corporatebench/` | **只 zenith** | 邮件线程 + KB 问答 |
| `medmemorybench/` | **data/zh 干净档** | 中文医患对话 |
| `memtrapbench/` | 整仓 | 记忆陷阱题（含它自己的评测代码） |
| `doc-pp/` | 整仓 | 文档披露政策（PDF + 1,141 题） |

许可依据**不在这张表里**——见下面的「许可证」一节。

**归档里没有的**（不要把这份表读成"官方只跑这些"）：

- **ScriptMem 那 4 部剧本的对话正文**——上游因版权明确不发布，**只存在于我们采集到的流量里**；
- **中文法条**（`source: 中华人民共和国未成年人保护法`）那一族——**出处至今未识别**；
- **Arknights 剧情**——不是评测数据集，不归档；
- PersonaMem-v2 的 **其余 800 份** `chat_history_32k`（上游共 1,000 份 / 约 167 MB）
  ——**我们只取了 `benchmark/text/benchmark.csv` 引用到的那 200 份**，另外 800 份没有任何题引用；
  `chat_history_128k`（约 519 MB）与 1M 档整体未取。
  > ⚠ **别用 HuggingFace 的 `?recursive=true` 判断"某个档在不在"**：它**静默截断在 1,000 条**，
  > 而 PersonaMem-v2 光 `chat_history_128k` 就有 1,000 个文件 ⇒ 那次查询看不到 `32k`，
  > 一度把它误记成"已撤下"。查子目录要单独打 `tree/main/<路径>`；
  > 要精确查某几个路径，用 `POST /api/datasets/{repo}/paths-info/{rev}`（它不受 1,000 条截断影响）。
- MedMemoryBench 的 `data/zh` 两个带噪变体（`dialogues_with_noise` / `noise_sessions`）

### ⚠ `doc-pp/data/` 是派生物，且上游的分卷 zip 不能直接解

`doc-pp/` 里注册进清单的是**仓库原文件**，其中 `data.zip` + `data.z01`–`z06` 是一个**分卷 zip**。
解出来的 `data/`（`03_final_data.json` + 635 份 PDF）**不在清单里、也不留在磁盘上**——它是派生物
（**doc-pp 这个数据集不做了**，理由见上一节）。要复核时才解、解完即弃：

```bash
cd benchmark_data/doc-pp
zip -s 0 data.zip --out data_combined.zip && unzip data_combined.zip && rm data_combined.zip
```

⚠ 本机（WSL，2026-09-30）**没有 `zip`/`unzip`/`7z` 任何一个**，Python 的 `zipfile` 直接解也会在
条目偏移上失败（`Bad magic number`）。当时的解法是「按 `z01…z06, zip` 顺序拼接 + 扫本地头签名」。
⇒ 换机器取回时先确认有 `zip` 系工具；没有就走拼接那条路。

### 这批题库的用途之一：给采集到的官方查询补金标

脚本 [`../tools/recover_official_gold.py`](../tools/recover_official_gold.py) 按归一化题面把它们对回
官方流量的 query，产物 `<采集目录>/official-gold-extra.jsonl`（**gitignored**，随采集目录一起），
按 `seq` 与 `official-eval-questions.jsonl` join。

⚠ **每一行的判分口径都不一样，而且都写在那一行的 `gold_judging` 里**——引用金标前先读它。
两个最容易踩的：MQuAKE 有更新前/后两套答案（取哪套**逐条按记忆里有没有注入对应改写**判），
BEAM 得用 `rubric`（10 个 probing 组的答案字段各不相同，只有 `rubric` 每组都有）。

### 用途之二：这些数据集已经能当**本地评测数据集**跑

`mquake-remastered` / `memtrapbench` / `corporatebench` / `medmemorybench` / `tempreason`
现在有加载器（[`../eval/datasets/`](../eval/datasets/)）与自写的 pipeline
（[`../eval/harness/extra_pipeline.py`](../eval/harness/extra_pipeline.py)）：

```bash
make eval DATASET=mquake-remastered ARGS='--limit 1 --spread'
```

⚠ 官方**没有发布**这几份的 pipeline ⇒ answer/judge 的 prompt 是**我们写的**，
**分数只在仓内前后对比**（其余三份走归档里的官方 pipeline，见
[`../eval/experiments/CLAUDE.md`](../eval/experiments/CLAUDE.md)）。

**BEAM 也已经是本地评测数据集了**（`make eval DATASET=beam`）——它的记忆语料来自本目录的
`beam/`，但**裁判走归档里的 `pipeline_beam.py`**（官方那份），所以它不属于"我们自写"那一类。

**MedMemoryBench 同理**（`make eval DATASET=medmemorybench`）：语料在 `medmemorybench/data/zh`，
**判分口径在上游发布的代码里**——本目录多了一份 `medmemorybench-code/`（只取 `metrics/` +
`utils/prompts_judge.py` + 骨架，不取 `methods/` 那 2545 个文件）：
`entity_exact_match → string_contain`、`multiple_choice → option_match`、其余四类 → LLM 裁判
（多跳那条是 `llm_judge_mcd`）。`extra_pipeline.py` **照它实现，prompt 直接从它读**。
⚠ 它的**形状是照上游评测循环复刻的**：一个 Sample = 一个（persona, 检查点），记忆只喂到该检查点
——**不读未来**，代价是投喂量 ≈ 5.4×（见 `eval/datasets/medmemorybench.py` 的 docstring）。

---

## pipeline 源码

```text
pipeline_locomo-refined.py     LoCoMo-Refined / LongMemEval 共用契约的代表
pipeline_longmemeval-s.py      LongMemEval
clb_pipeline.py                CL-Bench
pipeline_beam.py               BEAM
pipeline_scriptmem.py          ScriptMem
pipeline_v1_personamem.py      PersonaMem v1
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
> 好处：`parents[2]` 那条脆弱路径被绕开、配置只有 `.env` 一份。细则见 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md)。

### ⚠ 本地修订：这 7 份**不再逐字节等于上游**

**已核实**：上游钉住的那个 revision 上，7 个 pipeline 的 `answer` / `evaluate` 第一步就崩——

```python
async with httpx.AsyncClient(timeout=120) as client, output.open("a", encoding="utf-8") as handle:
```

`pathlib.Path.open()` 给的 `TextIOWrapper` **没有异步上下文协议** ⇒ `TypeError`，
**7 个文件、12 处**，全都跑不起来。拉上游原始字节比对过：**不是我们取错了文件**。

**处置（团队决定）**：**就地修订 + 记成已声明的偏离**。修法只有一处机械替换——
`contextlib.nullcontext(...)` 包住那个 `open(...)`（该函数同时支持同步与异步协议），
外加补一行 `import contextlib`。**不碰任何 prompt、不碰任何判分逻辑。**

| 事实 | 记在哪 |
| --- | --- |
| 修了哪 7 个文件、各几处、上游哈希是多少 | [`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py) 清单的 `local_patch` / `upstream_sha256` |
| 补丁**恰好只做那一处替换**（排除了"顺手改了别的"） | [`../tests/test_benchmark_archive.py`](../tests/test_benchmark_archive.py) |
| 重取时自动重打补丁 | `make fetch-data`（下载 → 校上游哈希 → 打补丁 → 校本地哈希）；网络不通时 `--patch` 就地打 |
| 校验 | `make data-check`（校的是**打完补丁**的本地哈希） |

⇒ **「契约以 pipeline 代码为准」这条仍然成立，但以后要带着这个星号读**：prompt 与判分逻辑与上游逐字相同，
差的是文件句柄那处语法。

---

## 许可证（已回原文验证）

| 数据集 | 许可 | 归档内的原文依据 |
| ---- | ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** | `locomo_refined_readme.md:9` 徽章 + `:296` 正文："LoCoMo-Refined is released under **CC BY-NC 4.0**. This benchmark modifies the original LoCoMo benchmark; see `NOTICE` for attribution and modification details." |
| ScriptMem | **CC BY-NC 4.0** | `scriptmem_readme.md:9` + `:183` |
| PersonaMem v2 | **CC BY 4.0** | `pmv2.md:2` 的 YAML frontmatter：`license: cc-by-4.0` |
| LongMemEval | **MIT** | ⚠ 归档内**仍**无依据（`lme_readme.md` 无 License 章节，`grep -ni licen` 零命中）。**从上游取到出处**：HF `xiaowu0162/longmemeval-cleaned` 数据集卡 `license: mit`，且 `github.com/xiaowu0162/LongMemEval-V2` 仓内有 `LICENSE` 文件——**两条独立出处，但都不在归档里** |
| CL-Bench | **归档内无许可证文本** | ❌（BEAM 那份见下面的 `official-extra` 表） |

**`official-extra` 那 8 个**（依据同样在本目录里逐份核过）：

| 数据集 | 许可 | 归档内的原文依据 |
| ---- | ---- | ---- |
| MQuAKE-Remastered | **CC-BY-4.0** | `mquake-remastered/README.md` frontmatter：`license: cc-by-4.0` |
| BEAM（100K 档） | **CC BY-SA 4.0** | `beam/README.md:73` frontmatter + `:166` `## 📄 License` 段（**上一节那份 BEAM 的"归档内无依据"到此为止**） |
| PersonaMem-v2（text/benchmark） | **CC BY 4.0** | `personamem-v2/README.md` frontmatter（与 `pmv2.md` 同一份卡） |
| TempReason | **CC BY-SA 3.0** | `tempreason/README.md` frontmatter：`license: cc-by-sa-3.0` |
| CorporateBench（zenith） | **Apache-2.0** | `corporatebench/LICENSE` 文件 + README frontmatter：`license: apache-2.0` |
| MedMemoryBench（zh） | ⚠ **两处冲突** | `medmemorybench/README.md` frontmatter 写 **CC BY-NC-SA 4.0**；流量报告记 GitHub 侧写 **CC BY 4.0**（该侧依据未落进归档） |
| MemTrapBench | ⚠ **unknown** | ❌ 整仓**没有** LICENSE 文件，README 也无许可字样（GitHub API 的 `license` 字段同为 `None`） |
| Doc-PP | ⚠ **归档内无依据** | ❌ README 与仓内文件都没有许可文本；流量报告记为 CC BY 4.0（**单边来源**） |

### 两条必须记住的推论

1. **NC（非商业）这一列**：LoCoMo-Refined 与 ScriptMem 都是 CC BY-NC 4.0。**不影响参赛**，但**意味着这两份数据不能进任何商业用途的产物**——如果后续想把系统或其中组件开源/商用，**这两份数据的评测结果是引用不了的**。
2. **"数据不得用于训练"这条在归档里没有出处。** 全库检索 `only for the evaluation` 等措辞**零命中**；`aml_readme.md` 只说到各数据集"remain subject to their respective upstream licenses and usage terms"，**没有任何"不得训练"的措辞**；归档里也没有 CLBench 的许可证文本。**它们来自归档之外的来源，属单边来源，引用前须回原始页面/许可证文件复核。**
   > **结论不变**——即便只按 AML 的通用条款，也不该拿这些数据训练——**但不要把它当成已归档的实证**（§12.5）。

---

## 归档没覆盖到的东西（引用前注意）

**PRD 附录 B 已标出两处"单边来源"**，本目录的归档里确实找不到依据：

| 断言 | 状态 |
| --- | --- |
| LongMemEval 的 MIT 许可 | ✅ **已取到出处**（上游 HF 数据集卡 + LongMemEval-V2 仓 `LICENSE`），但**归档内仍无** |
| AML 数据条款原文（禁止训练） | ❌ **归档内无依据**——**复核过 AML 仓 README（= `aml_readme.md`，逐字节一致）：全文无 "train" 措辞**，只有"remain subject to their respective upstream licenses and usage terms" |
| CLBench 许可证文本 | 归档内无 |
| **"`speaker_1_memories` 等字段的使用方式"** | ✅ 有依据，且**现在有两份**：`pipeline_locomo-refined.py` 与 `pipeline_longmemeval-s.py` 的渲染函数（后者确认了 `{{speaker_1_memories}}` / `{{speaker_2_memories}}` 的 `<memories>` 块形状） |
| **检索结果 → `memories` 字段的映射** | ❌ **在 AML 那一侧，归档里看不到**（§11.3）——这正是 S1 只能靠 Smoke 消除的原因 |

**另外**：LongMemEval 的 pipeline 文件**原归档缺失，已从上游补入**（`pipeline_longmemeval-s.py`）——PRD 里"LoCoMo-Refined / LongMemEval 契约相同"这条断言，**因此不再只靠 LoCoMo 那一份文件作证**。


### ⛔ 两个数据集**不做本地评测**（2026-09-30，逐条查过）

| 数据集 | 为什么做不了 | 查到什么程度 |
| --- | --- | --- |
| **ScriptMem** | **剧本正文没有任何公开来源**：上游 `README` 明写"因版权不发布对话正文"，**实测**它 `data/raw/{angry,enemy,friends,man_earth}.json` 的 `conversation` 字段只有 **252–381 字符的 `format_example`**（四份逐份核过）；HF 上搜 `scriptmem` **零命中**；仓里只有题（`data/public/questions.jsonl`，已归档） | ⛔ **唯一副本在请求采集里**——但那不是"从原链接下载"，按团队口径不采用 |
| **Doc-PP** | 官方作答端**只认 PDF 图像**（`--doc-mode pdf\|image`），而我们的服务返回**文本证据** ⇒ 接了也测不了自己 | 数据保留在归档（`doc-pp/`，清单里 36 项），**只是不接进评测**；它解出来的 `data/` 是派生物，**不留在磁盘上** |

> ⚠ 两条都**保留归档**（题目 / PDF 都在，出处与哈希照旧）——不做评测不等于把它删掉。
