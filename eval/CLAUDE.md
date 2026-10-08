# eval/ — 代理评测

**PRD**：§12（评测方案）、§13（对照实验）、§2.4（机会成本）

**这是本项目全部迭代发生的地方**——AML 不提供本地评测，所以所有消融、调参、验证都在这里完成。配额与三层结构见根 `CLAUDE.md`；**本节只管本目录的分工与边界**。

---

## 目录分工

| 目录 | 是什么 |
| --- | --- |
| [`datasets/`](./datasets/) | **数据怎么进来**：独立评测加载器 + **schema 落差预处理层**；`aml/` 构造版本化输入事件 |
| [`harness/`](./harness/) | **怎么跑一轮**：模拟 AML 切批喂 Add、调 Search、接裁判 |
| [`experiments/`](./experiments/) | **怎么跑一个对照**：A3 / T1 / T2 的 arm 定义 + T2 的标注产物（A4 已移出 v1，**D26**；A0 未开始） |
| [`baselines/`](./baselines/) | B1 ReFind（**Vendor，禁止被 `src/` import**）· 源码参考 `memmachine/`（[学习入口](../docs/memmachine-reference.md)）· 只读参考 `invmem-candidate/` 与 `serve_invmem_qwen.py`（**不是基线**） |
| [`smoke/`](./smoke/) | **只打真 AML 的那一层**：契约预检（✅ `preflight.py`）；⬜ 未写：配额/节流（`quota.py`）、S1/S2/S3 判别实验 |
| [`reports/`](./reports/) | **数字的唯一落点**：结果台账 + 每次 run 的归档 |

> **三处都叫"实验"是刻意的，但边界不能糊**（否则必然漂移）：
> `docs/experiments.md` = **协议登记**（哪个对照回答什么问题、决定什么）·
> `experiments/` = **怎么跑**（arm 定义与 runner）·
> `reports/` = **数字**。

---

## ⚠ 一条贯穿全目录的边界：**harness 打 HTTP，不 import `src/`**

**`harness/` 必须通过 HTTP 驱动服务，不能 `import tianximem` 的内部模块。**（各子目录不重复这条。）

理由（§13）：**B1 是 ReFind 的原版实现，只以"另一个 Add/Search 服务"的形式存在**——进程内调用会让它成为特例、与主路径不可比；且**主路径（混合检索）过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）**只有打 HTTP 才验证得了**，进程内调用根本碰不到契约层。

**顺带的好处**：契约违规（例如返回超过 `top_k`）会在本地就被抓到，而不是等 Smoke 那 30 次配额。

---

## 代理评测的边界：不可线性外推（§12.4 / P3）

**代理评测只覆盖 LoCoMo-Refined + LongMemEval，而这两份恰好是全部数据集里唯一共用同一套契约的。**

> **本节与 PRD §12.4 的分工（单数来源）**：**原则与"不可外推"的结论以 §12.4 为准**；
> 下表只登记**评测侧的操作口径**：既有各份的注入/裁判口径从 pipeline 代码核实而来，
> **新接入的份**（HaluMem / MuSiQue / HybridQA / FEVEROUS / `official-extra`）的本地适配
> **只登记在这里**——它们没有官方 AML pipeline，PRD 也没有对应行。

| 数据集 | 记忆注入字段 | 裁判 |
| ---- | ---- | ---- |
| LoCoMo-Refined / LongMemEval | 主字段 `speaker_1_memories` / `speaker_2_memories`。**退化不对称**：`speaker_1_memories` → `retrieved_context` → `memories`，而 `speaker_2_memories` **无退化，缺即空串** | 二元 CORRECT/WRONG，含严格时间粒度规则 |
| BEAM | `context` / `retrieved_context` / `memories` **优先**，speaker 块只是兜底（**优先级与上面相反**） | 逐条 rubric 三点制；**时间规则与上面相反** |
| ScriptMem | **只用** `speaker_*_memories`，缺失时**静默渲染成空串，不报错** | **无 LLM 裁判**，选项精确匹配 |
| CL-Bench | 嵌套的 `retrieval.selected` / `msp_retrieval.selected`，每项含 `created_at` + `text` | 严格全有全无 rubric，**且 API/JSON 失败一律记 0** |
| PersonaMem v2 | 归档原版读 `chat_history/messages`；本地适配器用逐题 Search 片段替代历史（边界见 [`harness/CLAUDE.md`](./harness/CLAUDE.md)） | 官方 MCQ 选项字母判分 |
| HaluMem / MuSiQue | 本地定义的 `retrieved_context`；输入和评分范围见 [`datasets/halumem.py`](./datasets/halumem.py) / [`datasets/musique.py`](./datasets/musique.py) | HaluMem 复用上游 QA 三分类；MuSiQue 用本地答案别名精确匹配及拒答评分 |
| HybridQA / FEVEROUS | 本地定义的 `retrieved_context`；分别从整表及链接段落、claim-only 候选整页构造 Add，范围见对应加载器 | 复用固定版本上游 EM/F1 或标签与完整证据组评分；本地输入范围和验证见 [接入报告](./reports/hybridqa-feverous-pipelines-20261006.md) |
| **MQuAKE / MemTrapBench / CorporateBench / MedMemoryBench / TempReason**（`official-extra`） | **我们自定**：平铺的 `retrieved_context`（这五份没有官方 pipeline，见 [`harness/extra_pipeline.py`](./harness/extra_pipeline.py)） | MQuAKE / CorporateBench：纯函数（别名表 / 标量·集合）；MemTrapBench：LLM 四维均分（阈值我们定）；MedMemoryBench：照上游发布的 `metrics/`；TempReason：纯函数（可接受答案串命中，口径我们定） |

表中的公共语料构造描述属于 `native`；`aml-v1` 的事件顺序、共享页面池和近似范围统一见
[`../docs/benchmark-data.md`](../docs/benchmark-data.md) 的「公开数据的 AML 输入适配」。

> **最后一行的边界**：其余归档评分与本地适配范围见上表；PersonaMem 的本地答案历史
> 经过显式检索适配，并非原版完整历史口径。最后一行**没有官方 AML pipeline**——answer 侧多为自写
> （**MemTrapBench 例外**：逐字用官方 prompt），判分口径多为自定（**MedMemoryBench 例外**：
> 照上游发布的 `metrics/`；MemTrapBench 按官方四维改写）。
> 所以这五家的分数**只能在仓内前后对比**，不能与官方分数对齐（§12.4 的理由照旧）。

> ### ⚠ 还有一边不止"注入字段"：**送进去的 `Add` 本身**
>
> 上表说的是**我们返回什么**。反过来的那一半是**AML 送什么进来**——`Add` 的 payload
> 由 AML 造，本地由 harness 造 ⇒ **本地发的像不像线上，决定分数预不预测得了线上**。
> 实测官方 add 的正文**绝大多数带 `<标签>: ` 前缀**（`Caroline:` / `Corpus:` / `user:` …），
> 且**`role` 取值域零 `system`**——**数字与样本量见 [`reports/ledger.md`](./reports/ledger.md)**。
> 缺省按这个渲染（`--add-shape official`），依据与逐数据集表在
> [`harness/add_shape.py`](./harness/add_shape.py)——**本文件不复制那张表**。

**三条推论：**

1. **分数不能线性外推。** 各家注入格式、时间处理和判分不同；PersonaMem 还存在归档完整历史与本地检索适配的口径差别。
2. **`created_at` 不要省**——CL-Bench 靠它渲染时间戳。
3. **"给更多上下文"在不同数据集上风险方向相反。** ScriptMem 与 CL-Bench 是精确匹配 / 全有全无，**多给的直接判 0**；BEAM 与 PersonaMem 的裁判则宽容。**没有一个"更丰富总是更好"的统一策略。**

**因此**：只在代理上做**相对**比较，**不做绝对外推**（§17.3 P3）。

---

## 两条判读纪律（本目录内不重复）

| 纪律 | 出处 |
| --- | --- |
| **重点指标是端到端，不是 Recall@K** | §12.2 / §14——缺口在"留哪些、按什么顺序留" |
| **不要相信自报数字** | §12.3 第 4 条：**Mem0 自报 93.4%、第三方复现 29.07% 是常态。只有自己 harness 里跑出来的数才算数** |
