# eval/ — 代理评测

**PRD**：§12（评测方案）、§13（对照实验）、§2.4（机会成本）

**这是本项目全部迭代发生的地方**——AML 不提供本地评测，所以所有消融、调参、验证都在这里完成。配额与三层结构见根 `CLAUDE.md`；**本节只管本目录的分工与边界**。

---

## 目录分工

| 目录 | 是什么 |
| --- | --- |
| [`datasets/`](./datasets/) | **数据怎么进来**：两个计分数据集的加载器 + **schema 落差预处理层** |
| [`harness/`](./harness/) | **怎么跑一轮**：模拟 AML 切批喂 Add、调 Search、接裁判 |
| [`experiments/`](./experiments/) | **怎么跑一个对照**：A3 / A4 / T1 / T2 的 arm 定义 + T2 的标注产物 |
| [`baselines/`](./baselines/) | A0 recency sanity · B1 ReFind（**Vendor，禁止被 `src/` import**） |
| [`smoke/`](./smoke/) | **只打真 AML 的那一层**：配额/节流、契约预检（✅ `preflight.py`）、S1/S2/S3 判别实验 |
| [`reports/`](./reports/) | **数字的唯一落点**：结果台账 + 每次 run 的归档 |

> **三处都叫"实验"是刻意的，但边界不能糊**（否则必然漂移）：
> `docs/experiments.md` = **协议登记**（哪个对照回答什么问题、决定什么）·
> `experiments/` = **怎么跑**（arm 定义与 runner）·
> `reports/` = **数字**。

---

## ⚠ 一条贯穿全目录的边界：**harness 打 HTTP，不 import `src/`**

**`harness/` 必须通过 HTTP 驱动服务，不能 `import tianxi_am` 的内部模块。**（各子目录不重复这条。）

理由（§13）：**B1 是 ReFind 的原版实现，只以"另一个 Add/Search 服务"的形式存在**——进程内调用会让它成为特例、两条基线不可比；且**主路径（混合检索）过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）**只有打 HTTP 才验证得了**，进程内调用根本碰不到契约层。

**顺带的好处**：契约违规（例如返回超过 `top_k`）会在本地就被抓到，而不是等 Smoke 那 30 次配额。

---

## 代理评测的边界：不可线性外推（§12.4 / P3）

**代理评测只覆盖 LoCoMo-Refined + LongMemEval，而这两份恰好是全部六份里唯一共用同一套契约的。**

| 数据集 | 记忆注入字段 | 裁判 |
| ---- | ---- | ---- |
| LoCoMo-Refined / LongMemEval | 主字段 `speaker_1_memories` / `speaker_2_memories`。**退化不对称**：`speaker_1_memories` → `retrieved_context` → `memories`，而 `speaker_2_memories` **无退化，缺即空串** | 二元 CORRECT/WRONG，含严格时间粒度规则 |
| BEAM | `context` / `retrieved_context` / `memories` **优先**，speaker 块只是兜底（**优先级与上面相反**） | 逐条 rubric 三点制；**时间规则与上面相反** |
| ScriptMem | **只用** `speaker_*_memories`，缺失时**静默渲染成空串，不报错** | **无 LLM 裁判**，选项精确匹配 |
| CL-Bench | 嵌套的 `retrieval.selected` / `msp_retrieval.selected`，每项含 `created_at` + `text` | 严格全有全无 rubric，**且 API/JSON 失败一律记 0** |
| PersonaMem v2 | **没有记忆注入**——消费 `chat_history` / `messages`，**忽略全部检索字段** | MCQ 精确文本匹配 |

**三条推论：**

1. **分数不能线性外推。** 你在 LoCoMo-Refined 上调出的注入格式、时间处理、排序偏好，到了 PersonaMem **可能完全不生效**——那份 pipeline **根本不读检索字段**。
2. **`created_at` 不要省**——CL-Bench 靠它渲染时间戳。
3. **"给更多上下文"在不同数据集上风险方向相反。** ScriptMem 与 CL-Bench 是精确匹配 / 全有全无，**多给的直接判 0**；BEAM 与 PersonaMem 的裁判则宽容。**没有一个"更丰富总是更好"的统一策略。**

**因此**：只在代理上做**相对**比较，**不做绝对外推**（§17.3 P3）。

---

## 两条判读纪律（本目录内不重复）

| 纪律 | 出处 |
| --- | --- |
| **重点指标是端到端，不是 Recall@K** | §12.2 / §14——缺口在"留哪些、按什么顺序留" |
| **不要相信自报数字** | §12.3 第 4 条：**Mem0 自报 93.4%、第三方复现 29.07% 是常态。只有自己 harness 里跑出来的数才算数** |
