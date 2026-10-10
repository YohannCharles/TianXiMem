# dataset/ — 评测材料准备与归档

> 对应 PRD §12.2 / §12.3 / §12.5、附录 B 与 D35。冲突以 PRD 为准。

`dataset/` 保存本地评测所需的公开数据；项目脚本放在 `eval/`。
旧 `benchmark_data/` 已迁移。现有数据字节与评分口径不因目录整理而改变。

```text
dataset/
├── CLAUDE.md              # 项目说明，进入 Git
├── locomo-refined/        # questions.jsonl、locomo_refined.json、上游 README
├── longmemeval-s/         # lme_s_cleaned.json、上游 README
├── clbench/               # clbench.jsonl
├── personamem-v2/         # benchmark/ 与 data/
├── …                      # 每个数据集一个目录
├── .upstream/             # 下载的官方 pipeline、裁判代码、参考材料
├── .legacy/               # 无法重取的 rh*.md 与旧归档中的未登记文件
└── .tmp/                  # 下载与补丁暂存，不供评测读取

eval/datasets/
├── manifest.py            # URL、固定 commit/revision、sha256 的唯一清单
├── layout.py              # 清单标识到本地路径的唯一映射
├── prepare.py             # 按需下载、校验、补丁、发布与旧目录迁移
├── aml/                   # AML 输入契约：有序 Add/Search 计划，保留原生加载器
└── …                      # 加载器与 schema 归一化
```

只有 `dataset/CLAUDE.md` 进入 Git；数据、上游材料、临时文件与历史存档都被忽略。
上游 README 可随数据下载，项目规范只写在可提交的说明里。

## 准备与使用

```bash
make fetch-data DATASET=locomo-refined  # 只准备该数据集和评分依赖
make data-check DATASET=locomo-refined  # 离线校验
make eval DATASET=locomo-refined        # 缺失材料自动下载，准备完才发 Add/Search
make eval DATASET=locomo-refined ARGS='--offline'  # 禁止下载，缺失就失败
make replay-official ARGS='--offline'  # 重放只准备所选家族的评分依赖
make fetch-data DATASET=all            # 显式取全档
make data-check DATASET=all            # 校验全档
python -m eval.datasets.prepare --list # 可准备的数据集
```

下载器本身只依赖 Python 标准库；`make` 入口会读取存在的 `.env`。
数据根目录仍由 `TIANXIMEM_BENCHMARK_DIR` 指定，缺省为 `dataset`。

流程是：**选本次数据集 → 缺失文件下载到独立 `.tmp/` 子目录 → 校验固定上游 sha256
→ 应用清单声明的机械补丁 → 校验本地 sha256 → 原子移入目标路径**。
失败时不发布半文件，本次临时文件会清理；重跑复用已校验文件。
已有文件校验不符会明确失败并保留原件，不静默换版本。
`--offline` 只读已有材料；import 和 loader 从不自动联网。

“处理”分为目录整理与 schema 归一化：下载阶段保留上游数据字节，
加载阶段复用现有 `preprocess.py` 和各数据集 loader 转成 AML 消息/问题形状。
不重复存一份通用 JSONL，避免内存、磁盘与数据口径分叉。分卷归档仍按需读取目标成员。

官方采集重放只准备所选家族的裁判依赖，不下载公开语料替代采集原文。
`official-dataset-*` 是我们自己采集的线上流量，无法由此下载器重建。

旧平铺目录与测试 fixture 仍可只读加载。新机器不必建立旧目录。
迁移命令为 `python -m eval.datasets.prepare --migrate-from <旧目录> --dir <数据根目录>`：
先核对所有文件与目标冲突，再移动，未登记文件也保留在 `.legacy/unregistered/`。
验证记录见 [`../eval/reports/datasets-20261006.md`](../eval/reports/datasets-20261006.md)。

## 公开数据的 AML 输入适配

`eval/datasets/` 保留现名：它负责材料准备、原始数据读取和输入计划；
`eval/harness/` 负责 HTTP 执行与评分。新增
[`eval/datasets/aml/`](../eval/datasets/aml/) 子包，入口为 `registry.py`，
公共事件类型为 `plan.py`。通用对话、文档与检查点转换放在 `common.py`；
MQuAKE、LoCoMo、MuSiQue、HybridQA、FEVEROUS 各有专用适配器。

runner 缺省仍为 `--input-contract native`。显式使用 `aml-v1` 才启用新的输入规则：

```bash
make eval DATASET=musique ARGS='--frozen --input-contract aml-v1'
make eval DATASET=halumem ARGS='--frozen --input-contract aml-v1 --aml-time-style inline'
make eval DATASET=locomo-refined ARGS='--frozen --input-contract aml-v1 --add-shape alluser'
make eval DATASET=feverous ARGS='--frozen --input-contract aml-v1 --aml-pool /path/to/pages.json'
```

| 数据集 | `aml-v1` 的 Add/Search 规则 |
| --- | --- |
| CorporateBench | QA 子集合并到同一个公司用户；文档正文加 `document:`，逐篇独立 session，保留源日期 |
| MQuAKE-Remastered | 每个公开 case 一个用户；JSON 原始事实 → 原答案提问 → JSON `UPDATE` → 新答案提问；更新后链中的普通事实作为背景，不提前应用编辑 |
| MemTrapBench | 保留对话角色，正文加角色前缀，使用合成时钟；Search 使用原始 final trigger |
| LoCoMo-Refined | 完整对话加确定性分配的公开 LongMemEval haystack；真人用姓名前缀，干扰历史用 `User/Assistant`；`alluser` 仅改变真人对话角色 |
| MedMemoryBench | 同一 persona 的已选源检查点合并成持续用户；仅 Add 尚未出现的会话，再 Search 当前题目 |
| HaluMem | 同上；默认角色前缀与合成时间，也支持捕获中的 `[Time: …]` 内联源时间形态 |
| MuSiQue | 保留 Full-dev 候选段落源序，添加 `Corpus: [Paragraph idx]` 和标题；Search 追加仅用给定段落作答与拒答指令 |
| HybridQA | 同表题目共用用户；Add 完整原始表格 JSON 和所有链接段落；Search 为原题 |
| FEVEROUS | 所有已选 claim 共用声明页面池；默认选同一捕获用户的 Search 对应公开 claim，Add 完整上游页面 JSON，Search 保留原始核查指令；自定义池使用公开 dev 选题 |

TempReason、LongMemEval 的独立 QA 与采集实例缺少可靠映射，`aml-v1` 明确拒绝；
它们继续使用原生模式。LongMemEval 在此作为 LoCoMo 的干扰历史，CLI 会一起准备材料。
其余未登记的家族也不会自动回落成声称 AML 对齐的输入。

FEVEROUS 的 `pages.json` 仅接受 `{"titles": ["Page A", "Page B"], "provenance": "来源说明"}`。
清单应独立于题目金标声明，不从 gold evidence 补页。不传清单时，只从本地采集 Add
恢复标题，再从公开 Wikipedia 数据库读原始页面。默认题目来自同一用户的原始 Search query，
唯一匹配到公开 dev claim 后再抽样；重复 query 合并，匹配缺失或歧义时失败。
页面选择不读取 query 的 gold，答案与证据只取公开标注用于评分。
自定义标题池使用公开 dev 选题，题数与来源范围另记在 manifest；不能与默认捕获范围混比。
没有采集时必须提供清单，该条件在自动下载前检查。采集目录配置沿用 `TIANXIMEM_CAPTURE_DIR`。
标题查询兼容 NFC/NFD，但 Add 与证据引用保留原始页面拼写。

这是一套**可复现的本地输入近似**。合成时钟、字符切片和按空白计词的批界记录在
`input-manifest.json`，不保证与平台 tokenizer 或捕获请求字节相同。
LoCoMo 的公开题库没有逐题到达时刻和反馈写回，当前先投喂所选完整历史再检索，
不能恢复捕获中的真实交错；逐请求原文回放继续使用 `make replay-official`。

[`plan_driver.py`](../eval/harness/plan_driver.py) 先编译消息与批次，再按事件顺序发 HTTP。
版本、实际请求内容和 run-id 共同隔离用户；金标只用于独立的评分指纹，不进入 Add 或 query。
`--frozen` 按输入契约选择独立配方，`--max-questions` 只裁 Search，保留 Add 的原位置；
通用转换合并前按每个源检查点/QA 子集裁题。
每次 Search 成功后先保存结果，再允许后续 Add；失败时停止当前用户的后续事件。
重跑同一 run-id 复用该历史快照，不重新检索已经走过的时点。
输出目录、配置或输入变化、检查点丢失时停止；变更输入需要新的 run-id 与独立存储快照，
不能通过删除检查点或 `--skip-ingest` 重置历史状态。

答案仍只读取 Search 返回文本。FEVEROUS 的新
`feverous-aml-json-evidence-v1` 契约从完整可解析的检索 JSON 页面或显式可见的括号 ID
验证证据；不完整、缺失或乱序片段不补全，也不从 Add 池取证。
该保守限制可能降低严格证据分。它仍使用固定上游标签与完整证据组评分，
与采集重放的 label-only 口径分开。
实现与本地验证见 [接入报告](../eval/reports/datasets-20261006.md)。

---

## 出处链：字节从哪来

**除了 `rh.md` / `rh2.md` / `rh3.md` 三份，每一份都有确定的出处，且都能按 commit / revision 钉死。**
（逐字节核对通过；只有那 3 份取不回来。）机器可查的清单（出处 + sha256）在
[`../eval/datasets/manifest.py`](../eval/datasets/manifest.py)——**哈希只写那一处，本文不重复**。
取回 `make fetch-data`，校验 `make data-check`。

| 内容 | 出处（在哪） | 钉法 |
| --- | --- | --- |
| —— | **具体 commit / revision 值与 sha256 一律不在此处重复**，见 [`../eval/datasets/manifest.py`](../eval/datasets/manifest.py) | —— |
| `pipeline_*.py` / `clb_pipeline.py`、`aml_readme.md` | [github.com/AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard) 的 `data/*/pipeline.py` + 根 `README.md` | 按 commit 钉死 |
| `locomo_refined.json`、`questions.jsonl`、`locomo_refined_readme.md` | [github.com/mem-eval-suite/LoCoMo_refined](https://github.com/mem-eval-suite/LoCoMo_refined) | 按 commit 钉死 |
| `scriptmem_q.jsonl`、`scriptmem_readme.md` | [github.com/memorax-ai/ScriptMem](https://github.com/memorax-ai/ScriptMem) | 按 commit 钉死 |
| `lme_s_cleaned.json` | HF [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 的 `longmemeval_s_cleaned.json` | 按 revision 钉死。**sha256 = HF 的 LFS oid**（实测相等） |
| `clbench.jsonl` | HF [`tencent/CL-bench`](https://huggingface.co/datasets/tencent/CL-bench) 的 `CL-bench.jsonl` | 按 revision 钉死。sha256 = LFS oid |
| `locomo10.json` | [github.com/snap-research/locomo](https://github.com/snap-research/locomo) 的 `data/locomo10.json` | 按 commit 钉死 |
| `pm_32k.csv`、`pm_questions_128k.csv`、`pm_questions_1M.csv` | HF [`bowen-upenn/PersonaMem-v1`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) 的 `questions_{32k,128k,1M}.csv` | 按 revision 钉死。⚠ **不是 v2 仓**——v2 仓里只有 `benchmark/*.csv`，本地文件名是被改过的 |
| `pmv2.md`、`lme_readme.md` | HF `bowen-upenn/PersonaMem-v2` 的 `README.md`；[github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) 的 `README.md` | 分别按 revision / commit 钉死 |
| **`official-extra` 的数据集** | 6 个 HF 数据集 + 2 个 GitHub 仓（`zjunlp/MemTrapBench`、`hwanchang00/doc-pp`），**路径清单见本文件下面的专节** | 按 revision / commit 钉死；HF 侧 `sha256 = LFS oid`（实测相等） |
| `rh.md`、`rh2.md`、`rh3.md` | ❌ 抓取时原始 URL 已 404，是搜索引擎索引副本 | **取不回来**，只作存档 |

**⇒ 因此"所有人都需要的那份"不需要住在任何人的电脑上，也不需要任何共享盘。**
字节各自从公开源取，清单进版本库——这同时满足"不打搅同事的机器"与"不外传"。

### 两条不外传的理由（引用前必读）

1. **AML 仓没有 LICENSE**，而且它的 README 有一节 *Deliberately not included*：不发布语料与金标，并写明
   **"Do not submit any of these materials in an issue or pull request."**
2. **第三方数据带 CC BY-NC 等约束**（见下方许可表），分发范围未确认前同样不提交。

**⇒ 下载的材料保持 `.gitignore` 排除，这是刻意状态；只提交项目说明和清单。**
这两条不构成"不能各自取回"：数据从上游直接下载，**不经过我们**。

---

## 哪些文件是真数据，哪些不是

以下文件名沿用 `manifest.py` 的 `name` 标识；实际本地路径由 `layout.py` 映射。

**这一节是为了防止有人把残留文件当成数据集反复排查。**

| 文件 | 是不是数据 |
| --- | --- |
| `lme_s_cleaned.json` | ✅ **用这个**（LongMemEval） |
| `lme_test.json` | 🗑 **不入档**（266MB）——它是"用得上但**明令别用**"的陷阱：多出 1,230 个空 session + 15 个干扰 session，**会污染按 20 条切批的埋点**（§6.5）。要复核那 1,230 这个数字时，从上游 LongMemEval 取回（**出处未核**，见 `eval/datasets/manifest.py` 的 `DELETED`） |
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
3. **能逐字匹配回官方 query 的只有一部分**：TempReason 的题面在 10,144 条官方 search 里
   **0 命中**（官方实例集不在归档里，问法也被改写过），**不代表它没被跑过**。
   ⚠ 2026-10-06 更新：**Doc-PP 早就对上了**——它的 316 条 query 走 `(user question, policy)`
   精确 join（见 [`../tools/recover_docpp.py`](../tools/recover_docpp.py)），当时的"0 命中"
   是**拿整段 preamble 当题面**去比的假阴性。

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
| `halumem/` | **Medium** + 上游 `eval/eval_tools.py` | persona 访谈（**流量里那一族 1,481 题就是它**） |
| `hybridqa/` | released_data/dev* + 固定 WikiTables-WithLinks 语料包 | 表格 + 全部链接段落的多跳问答 |
| `musique/` | ans + full 的 dev | 多跳问答（Full-dev 候选段落源序；变体 join 同时核对语料） |
| `feverous/` | dev challenges + 官方 Wikipedia SQLite | 事实核查（claim → 标签与元素证据） |

许可依据**不在这张表里**——见下面的「许可证」一节。

**归档里没有的**（不要把这份表读成"官方只跑这些"）：

- **ScriptMem 那 4 部剧本的对话正文**——上游因版权明确不发布，**只存在于我们采集到的流量里**；
- **中文法条**（`source: 中华人民共和国未成年人保护法`）那一族——**出处至今未识别**；
- **Arknights 剧情**——不是评测数据集，不归档；
- PersonaMem-v2 的 **其余 800 份** `chat_history_32k`（上游共 1,000 份 / 约 167 MB）
  ——**我们只取了 `benchmark/text/benchmark.csv` 引用到的那 200 份**，另外 800 份没有任何题引用；
  `chat_history_128k`（约 519 MB）与 1M 档整体未取。
  > ⚠ **别用 HuggingFace 的 `?recursive=true` 判断"某个档在不在"**：它**静默截断在 1,000 条**，
  > 而 PersonaMem-v2 光 `chat_history_128k` 就有 1,000 个文件 ⇒ 那次查询看不到 `32k`。
  > 查子目录要单独打 `tree/main/<路径>`；
  > 要精确查某几个路径，用 `POST /api/datasets/{repo}/paths-info/{rev}`（它不受 1,000 条截断影响）。
- MedMemoryBench 的 `data/zh` 两个带噪变体（`dialogues_with_noise` / `noise_sessions`）

### ⚠ `doc-pp/data/` 是派生物，且上游的分卷 zip 不能直接解

`doc-pp/` 里注册进清单的是**仓库原文件**，其中 `data.zip` + `data.z01`–`z06` 是一个**分卷 zip**。
解出来的 `data/`（`03_final_data.json` + 635 份 PDF）**不在清单里、也不留在磁盘上**——它是派生物
（**不做成评测数据集**：它的作答端只认 PDF 图像）。要复核时才解、解完即弃：

```bash
cd dataset/doc-pp
zip -s 0 data.zip --out data_combined.zip && unzip data_combined.zip && rm data_combined.zip
```

⚠ 本机（WSL，2026-09-30）**没有 `zip`/`unzip`/`7z` 任何一个**，Python 的 `zipfile` 直接解也会在
条目偏移上失败（`Bad magic number`）。
**2026-10-06 落成了确定性的读法**：[`../tools/recover_docpp.py`](../tools/recover_docpp.py) 里那段
**disk-aware 读取**（自解析中央目录 + 按 `94371840 × 盘号 + 局部头偏移` 定位成员 + 校验
CRC/长度），**只读它要的那一个成员、不落盘**——`data.zip` 的 sha256 也在那条脚本里钉着。
⇒ 换机器取回时不必先装 `zip`。

### 这批题库的用途之一：给采集到的官方查询补金标

脚本 [`../tools/recover_official_gold.py`](../tools/recover_official_gold.py) 按归一化题面把它们对回
官方流量的 query，产物 `<采集目录>/official-gold-extra.jsonl`（**gitignored**，随采集目录一起），
按 `seq` 与 `official-eval-questions.jsonl` join。

⚠ **每一行的判分口径都不一样，而且都写在那一行的 `gold_judging` 里**——引用金标前先读它。
两个最容易踩的：MQuAKE 有更新前/后两套答案（取哪套**逐条按记忆里有没有注入对应改写**判），
BEAM 得用 `rubric`（10 个 probing 组的答案字段各不相同，只有 `rubric` 每组都有）。

### 逐族补金标（2026-10-06）：一族一个脚本、一份产物

`tools/recover_{official_gold,docpp,clbench,halumem,corpusqa}.py`，产物都是
`<采集目录>/official-gold-extra*.jsonl`（**一族一份**，按 `seq` join）。
它们把 10,144 条官方 search 里 **8,783 条（86.6%）** 变成了**有金标、可判分**的题
（逐族覆盖率与判分口径见采集目录 README §9，口径的决定见
[`decisions.md`](decisions.md) **D34**）。合成套件是
[`../tools/build_official_kit.py`](../tools/build_official_kit.py)。

⚠ **新增的四份归档各自有一条坑**，都写在脚本的文件头：
HaluMem 是 **CC-BY-NC-ND-4.0**（禁再分发）；MuSiQue 的采集语料已与 Full-dev 候选段落源序对齐，
原生包装与采集有差异，核对依据见 [输入审计](../eval/reports/datasets-20261006.md)；
FEVEROUS 的金标是 **label + evidence id**，
采集重放只判 label；独立 FEVEROUS pipeline 判标签与完整证据组，见下文；
HybridQA 的 dev 够用（test 的答案封存）。

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
**判分口径在上游发布的代码里**——`.upstream/medmemorybench/` 保存上游评分实现（只取 `metrics/` +
`utils/prompts_judge.py` + 骨架，不取 `methods/` 那 2545 个文件）：
`entity_exact_match → string_contain`、`multiple_choice → option_match`、其余四类 → LLM 裁判
（多跳那条是 `llm_judge_mcd`）。`extra_pipeline.py` **照它实现，prompt 直接从它读**。
⚠ 它的**形状是照上游评测循环复刻的**：一个 Sample = 一个（persona, 检查点），记忆只喂到该检查点
——**不读未来**，代价是投喂量 ≈ 5.4×（见 `eval/datasets/medmemorybench.py` 的 docstring）。
临床多跳的 native 作答格式与旧答案复用纪律见
[`eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md) 的对应小节；
本地格式对照与证据缺口诊断见 [临床多跳报告](../eval/reports/optimization-20261007.md)。

**HaluMem 和 MuSiQue 也已接入独立本地评测**：

```bash
make baseline DATASET=halumem
make baseline DATASET=musique
# 数据缺失自动准备；离线可加 ARGS='--offline'
```

输入构造与评分口径的唯一说明在
[`halumem.py`](../eval/datasets/halumem.py) / [`musique.py`](../eval/datasets/musique.py) 的
docstring；冻结抽样由 [`recipes.py`](../eval/experiments/recipes.py) 声明。
实际数据计数、覆盖范围与验证结果见
[接入报告](../eval/reports/datasets-20261006.md)。

**HybridQA 和 FEVEROUS 也已接入独立本地评测**：

```bash
make fetch-data DATASET=hybridqa       # 缺失时 eval/baseline 也会自动准备
make fetch-data DATASET=feverous
make data-check DATASET=hybridqa
make data-check DATASET=feverous
make baseline DATASET=hybridqa
make baseline DATASET=feverous
# 已有源数据时，ARGS='--offline' 禁止联网，可重建派生文件
```

HybridQA 语料来自上游指定的 [WikiTables-WithLinks](https://github.com/wenhuchen/WikiTables-WithLinks)，
按固定提交取整表及全部链接段落。FEVEROUS 语料来自
[上游下载脚本](https://github.com/Raldir/FEVEROUS/blob/32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6/download_data.sh)
指定的 Wikipedia ZIP；准备入口校验压缩包和数据库后构建本地候选索引。
两份只把 Search 返回的文本送入作答，评分复用清单中固定版本的上游函数。

FEVEROUS 按 claim 从完整数据库生成有限候选页，再把候选整页喂 Add；
不使用金标或标注者操作来选页。它是本地候选适配，不能作为被测服务对完整
Wikipedia 的检索成绩，也不复刻 AML 候选池。输入构造的唯一说明在
[`hybridqa.py`](../eval/datasets/hybridqa.py) / [`feverous.py`](../eval/datasets/feverous.py)，
评分入口在 [`corpusqa_pipeline.py`](../eval/harness/corpusqa_pipeline.py)。
下载规模、空间要求、冻结样本及验证结果见
[接入报告](../eval/reports/datasets-20261006.md)。

---

## pipeline 源码

官方 AML 脚本统一放在 `.upstream/aml/`；下表保留清单标识。

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
| 修了哪 7 个文件、各几处、上游哈希是多少 | [`../eval/datasets/manifest.py`](../eval/datasets/manifest.py) 清单的 `local_patch` / `upstream_sha256` |
| 补丁**恰好只做那一处替换**（排除了"顺手改了别的"） | [`../tests/test_benchmark_archive.py`](../tests/test_benchmark_archive.py) |
| 重取时自动重打补丁 | `make fetch-data DATASET=<名称>`（暂存 → 校上游哈希 → 打补丁 → 校本地哈希 → 发布）；网络不通时 `--patch` 就地打 |
| 校验 | `make data-check DATASET=<名称>`（校的是**打完补丁**的本地哈希） |

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

**`official-extra` 的材料**（依据同样在本目录里逐份核过）：

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
