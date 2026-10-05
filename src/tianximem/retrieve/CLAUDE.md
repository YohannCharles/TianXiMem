# retrieve/ — 检索与判据（rerank 之前的那一半）

**PRD**：§7.1（BM25）、§7.2（Dense）、§7.3（Weighted RRF）、§8（Evidence Checker）

## 要写什么

```text
bm25.py      词法那一路：查询侧原样送（不含改写）
dense.py     语义那一路：Embedder 调用（**每查询恰好 1 次**）
fusion.py    **检索策略与参数所有权**（prefetch_limit / weights / k / top_k）
checker.py   §8 名次一致性判据
evidence.py  共同证据执行器：执行 facts/query 的共同计划（筛选/连接/区间/当前输入）+ 来源覆盖审查
```

> ### ⚠ 分工：**参数归本目录，执行归 `store/`**
>
> | | 谁 | 内容 |
> | --- | --- | --- |
> | **策略与参数** | **本目录** | `prefetch_limit` / `weights` / `k` / `top_k` 的**取值、校验、标定**；两路怎么组、权重多少 |
> | **请求构造与执行** | [`../store/qdrant_store.py`](../store/qdrant_store.py) | 把参数翻成 `prefetch` + `rrf` 的 Qdrant 调用，结果还原成领域对象 |
>
> **为什么这样切**：`k=61` / `prefetch` 每路带 `using` / 根级 `limit` 的**写法**是 Qdrant 的语法知识，而"用哪个模型、权重要不要调"是检索策略——策略**要能被配置驱动、能被 ablation 检验**，所以所有权上移到本目录（执行代码不搬）。
>
> ⚠ 因此 **`k=61` 的校验写在本目录**：它是正确性常量，配错要**拒绝启动**，不是"顺手调"（D5）。

**本目录只负责"拿到候选 + 判断够不够"。** 混合链的候选与共同取证链的证据都在这里产生；共同取证不适用或不齐全时，由 `service` 回落到混合链（见下文「共同取证」）。rerank 之后的链路（**Neighbor Expansion、Context Packaging**）在 [`../rank/`](../rank/)——§10 明确它的顺序是"**排序 → 扩窗 → 打包**"，扩窗跑在 rerank **之后**。

> ⚠ **不要把 Neighbor Expansion 挪回来。** 它在 §5 的流程图里位置很靠上（容易误读成检索的一环），但 §10 开头的顺序说明是权威的。

---

## §7.1 / §7.2 — 两路

| | 词法 | 语义 |
| --- | --- | --- |
| 实现 | **Qdrant 原生 `qdrant/bm25`**（服务端推理，真 BM25，sparse + `modifier: idf`） | `text-embedding-v4`，经 `embed/` 的 `Embedder` 接口 |
| 查询 | —— | **原样送进去，不做任何改写**；每查询**恰好 1 次**调用 |

**不使用任何学出来的稀疏权重**（learned sparse）：它既**不是 BM25**；且提交时的 `text-embedding-v4` **不提供该能力**（§7.1）。

### ⚠ 一条必须实测的假设

**BM25 的分数对分词器高度敏感**，所以"真 BM25"这个选择成立，但理由**不是**"BM25 天然更强"，而是"**配正经分词器的 BM25** 更强"。

> **实现要求**：选 `qdrant/bm25` 时，**它的分词行为要实测一次**（拿一两个多语言长文档的 case 对照），**别默认它等价于 Lucene Analyzer**——这是 [`../../../docs/open-questions.md`](../../../docs/open-questions.md) 的 **V1**。

---

## §7.3 — Weighted RRF

```json
{"prefetch": [{"query": <bm25>,  "using": "bm25",  "limit": N},
              {"query": <dense>, "using": "dense", "limit": N}],
 "query": {"rrf": {"weights": [w_bm25, w_dense], "k": 61}},
 "limit": 2 × N}
```

| 要点 | 说明 |
| --- | --- |
| **`k` 必须显式设 `61`** | **不是默认值（2），也不是文献里的 60**。是**正确性常量，不是调参项**——由来见根 `CLAUDE.md` 与 D5 |
| **每个 `prefetch` 都要带 `using`** | 命名向量场景下不写 `using`，Qdrant 无法确定用哪一路 |
| **根级 `limit` 取 `2 × prefetch_limit`**（**要全部融合结果**） | **不在 Qdrant 里按 `top_k` 截**——截断在 [`fusion.py`](./fusion.py) 里、确定性排序**之后**执行（否则并列分数卡在截断线上时"谁进谁出"由 Qdrant 随机决定）。契约 `len(data) <= top_k` 仍成立，但**任何一层都不要写死 100**。⚠ PRD §7.3 的片段写的是 `<top_k>`——实现有意把截断点后移，依据见 ledger 的「V13 补完」 |
| **每路 `prefetch` 的 `limit`（`N`）必须显式给定** | 初值 **200**，是"进入 RRF 融合的候选池"大小 |
| 权重初值 `[0.5, 0.5]` | **任何调整都必须有 ablation 数据支撑** |

**版本门槛**：`k` 自 v1.16.0、`weights` 自 v1.17.0 起可用 → 要求 **≥ v1.17.0**，部署时**钉死**（[`../../deploy/`](../../../deploy/)）。

### 三个不同的量，不要混用

`prefetch_limit`（**200**）· 种子数（`neighbor.expansion_seed_limit`，实际 **1000**）· Top-K（**100**，§2.2）

> ⚠ **别把 §10 的"20 种子 / 60 槽位"读进这里**——那是 PRD 用来演示预算怎么算的
> **示例算术**。实际取值与它的差异见 [`../rank/CLAUDE.md`](../rank/CLAUDE.md) §2。

**融合池上界**：200 × 2 路 = 至多 400 条，要给 100 个名额留出**邻域折叠与去重的余量**。

### 一个简化，以及它的代价

**RRF 只用排名不用分数**，因此 **BM25 分数与余弦相似度无需归一化对齐**——这绕开了 hybrid 检索最麻烦的一步。**代价**：权重代表的是"**名次话语权**"而非"分数重要性"（§7.3）。

---

## 共同取证（`evidence.py`）—— 混合检索之前的一条短路

启用 `retrieval.grounded_evidence` 时，`service` 先让
[`../facts/query.py`](../facts/query.py) 把问题编译成共同计划，再交给本目录的
`select_evidence(...)` 执行——命中则直接产出证据片段（不生成答案，§2.1 照旧），
不适用或判"不齐"才回原混合检索。

> ⚠ **问法识别不归本目录**：关系同义词与问法句法集中在 [`../facts/`](../facts/)
> （`query.py` 编译计划、`grammar.py` 抽取来源句法、`input.py` 检查当前字面输入）。
> **新增问法改那里，不要在本目录加对应分支**——本目录只执行计划。

| 计划 | 执行 |
| --- | --- |
| `filter` 条件筛选 | 按主体/关系/客体与原文限定条件取独立事实；**名单与计数共用同一份计划** |
| `walk` 有限关系连接 | 只沿各条原文边行走、深度由 `hop_limit` 封顶；不给推导结果造新记忆 |
| `interval` 显式区间比较 | 只用原文明确的起止月份；来源日期不冒充事件日期 |
| `current` 当前输入 | 计划判定输入已完备 ⇒ 空选择，由 `service` 返回合法空结果 |

外加两类把关：**数量消歧**（`_quantities`，在 `filter` 分支里**无条件跑**——同一项的多个数量观察若单位冲突、或没有可信来源时间可分新旧，选择失败）与**来源覆盖审查**（仅当计划带字面范围守卫时：同一字面范围的原文里若仍有未解释的相关正向/变更声明，选择失败）——**物理扫过不等于语义齐全**。

**两条上限由调用方注入**（`source_limit` / `hop_limit` 实参，来自
`retrieval.evidence_limit` / `evidence_hop_limit`；键与口径见
[`../../../docs/config-reference.md`](../../../docs/config-reference.md)）——
**本目录不读配置**。

### 不齐全 ⇒ `None`，回原混合检索

以下任一情况 `select_evidence` 返回 `None`（或 `service` 不进这条路径）⇒ 原混合
检索链原样接管；**不得把部分索引当完整集合**：

* 没有适用的计划（`compile_query` 返回 `None`）；
* 当前用户的事实索引未扫完（版本覆盖不全）；
* 相关原文仍有未解释声明或数量冲突；
* 选择超过 `evidence_limit`。

设计与验证记录见 [`../../../eval/reports/unified-evidence-refactor-20261005.md`](../../../eval/reports/unified-evidence-refactor-20261005.md)。

---

## §8 — Evidence Checker

判断当前候选是否已足够回答 Query；足够则直接返回，不足才进入 Agentic Search。

> **v1 里它是恒返回"充足"的带日志空实现**（D13）——下游 `agent/` 归 v2。**判据与接入点要在 v1 就写对**；⚠ **D26 之后 A4 已移出 v1** ⇒ `arm_rankings` 没人传、`criterion_would_say` 恒为 `None`（要补上那份分布得先给 `store/` 加一个单路查询——**已登记、未实现**）。

### 判据不能用融合分数

> **RRF 融合后的分数不是校准量**——Qdrant 官方明确警告不要把单路阈值用到根级 `score_threshold`，**"照搬 dense-only 的阈值会静默截断结果"**。

改用**名次一致性判据**：分别跑一次 `bm25-only` 与一次 `dense-only`，用两者返回的**排名**做判断：

```text
足够（Direct Return）：
  两路 top1 相同
  或 两路 top-5 重叠 ≥ 3
  或 bm25 top1 的名次在 dense 结果中位于前 3

否则：
  → Agentic Search
```

⚠ **v1 的调用方不跑这两次查询**（D26 之后 A4 已移出 v1）：`decide()` 接受可选的 `arm_rankings`，但唯一的调用方不传 ⇒ `criterion_would_say` 恒记 `None`。判据本身写全并测到了，见 [`checker.py`](./checker.py)。

### 阈值要标定，且**开关必须是配置项**

判据里的三个初值需在代理评测上标定。**`checker.enabled` 必须是配置项**（§15 / D15）——§13 的 A4 要关它做对照。四个键的落点见 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) §4。

---

## 边界：本目录到 Rerank 为止

下游是 [`../rank/`](../rank/) 的三连环——**排序 → 扩窗 → 打包**（§10 + §11.2 + §11.3）。本目录**不**做扩窗、**不**做打包、**不**排序。

**混合链要交接的信息**：Checker 的判定（v1 恒为"充足"）与候选名次（`Candidate`，交给 `rank/` 精排）。共同取证链交回 `service` 的是 `EvidenceSelection`（独立事实或原文来源）；**打包（含原文投影的段合并）仍由 `service` 调 [`../rank/`](../rank/) 的打包器完成**——本目录不越界。

**`top_k` 的最终计数不在本目录**：本目录返回的是候选（或 `EvidenceSelection`），**精确 ≤ `top_k` 的计数责任在打包完成之后**（§2.2）——因为扩窗会往里加槽位（§10），而扩窗发生在 `rank/`。
