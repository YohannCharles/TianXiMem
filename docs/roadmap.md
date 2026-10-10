# 路线图与阶段门

> 对应 PRD §16（+ §1 / §6.4 / §11.2 / §12.1 的阶段要求）。**冲突以 PRD 为准。**

**状态以根 [`CLAUDE.md`](../CLAUDE.md) 的模块表为准**；本文件记的是分步交付与**剩余项**。

---

## 总览

| 阶段 | 交付 | 状态 |
| ---- | ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval） | 🟡 加载层 + harness + 契约预检 + `experiments/run.py` 已就位，**链路已跑通多轮**；剩 S1–S3 三个真-Smoke 探针 |
| Step 1 | 存储层 + Add/Search 服务 + **混合检索**（BM25 + Dense + RRF，含 T2 实验） | 🟡 `src/` 已落；**T2 脚手架就位、人工标注未做** |
| Step 2 | **Neighbor Expansion + 双预算截断** | ✅ 已完成 |
| Step 3 | Rerank + Context Packaging（含 T1 实验） | 🟡 rerank 已接 + 打包已落地 + 渲染模板定稿；**T1 的问题已回答**（`t1-dated` 0.633），**干净的两臂 A/B 未跑** |
| Step 4 | Conditional Agentic Search | ⬜ |
| **Step 5** | **切换到提交模型**，重标定全部阈值，重跑 T2 | ⬜ |
| Step 6 | 对照实验（§13）+ Smoke 验证 + Full 定稿 | ⬜ |

---

## 两条不可协商的阶段约束

> ### Step 0 不可跳过
> **没有它，后面每一步都是盲调，而 Full 只有 2 次。**

> ### Step 5 不可与任何设计改动合并
> 见 §12.1 风险 R1 对冲 4。**否则分数变化无法归因。**

---

## Step 0 — 代理评测 harness

**这是唯一的"没有它后面全是盲调"的阶段。** 交付物不是 harness 本身，而是**"能产生可信数字的能力"**。

### 环境阻塞项

- [x] **`SqliteStore` 单连接 × 多线程服务 —— 短生命周期连接**（D17）
      长期持有一个在启动线程里建的 `sqlite3.Connection` 时，FastAPI 的 `def` 路由（跑在**线程池**里）**每个请求**都抛 `ProgrammingError`。
      修法是"连接限定在一次逻辑操作 / 一个事务内，用完即关"，不用 thread-local、也不用单一共享连接。
      回归用例：3 个 HTTP 往返 + 3 个并发用例 + `test_store.py` 的 3 个连接生命周期用例与 2 个**真实并发写**压力用例。
- [x] **`api_config.py`** —— [`../eval/harness/api_config.py`](../eval/harness/api_config.py)
      就是那个 `os.environ.get` 适配器（`AML_*` → 归档要的七个名字，`JUDGE_*` 留空即回落 `ANSWER_*`）。
      **归档侧不需要放在仓库外**：`judge.run_judge()` 起 subprocess 时把 `eval/harness/` 放进 `PYTHONPATH`，归档保持只读。
      回归用例：`tests/test_harness.py` 的桩 pipeline **真的 `import api_config`**。
- [x] **`docker` 已就位**——Docker Desktop / `desktop-linux` context / WSL2 后端。`qdrant/qdrant:v1.17.0` 的 tag 已在 registry 核实存在（§7.3 的版本门槛成立）。
      **拓扑已定**（D12）：harness 与检索服务 + Qdrant 都在**本机**、三段模型经自建网关远程访问、**reranker 端点待部署**。
- [x] **另一台机器（WSL）上的两个环境阻塞项 —— 均已解决**
      ① `make` 没装 ⇒ `sudo apt install make`。② Qdrant 容器没起 ⇒
      `sudo docker compose -f deploy/compose.yaml up -d`（`sudo` 绕过 docker 组，不必重开 WSL）。
      ⚠ 那台机器的 **WSL 里 `docker` CLI 用不了**（`/var/run/docker.sock` 的属主是 `root:docker`）——
      这与 `deploy/CLAUDE.md` 说的"daemon 走 Windows 命名管道、从 Git Bash 用"一致，**不是新问题**。
      **症状速查**：`make check` 会一次把这两件事都报出来（Qdrant 那一项会说清 502 与解法：**502 = 转发在、容器不在**，不是"拒绝连接"）。
- [ ] ⚠ **Windows 那台机器（非 WSL）的 `tmp_path` 故障 —— 与代码无关，是机器状态**
      `%TEMP%\pytest-of-r0304` 的 ACL 已损坏，于是**所有用 `tmp_path` 的用例**在 fixture setup 阶段就报 `PermissionError`
      （`uv run pytest` 表现为 **128 passed, 111 errors**）。绕法（**不写进 `Makefile`**——那是机器专属路径）：

      ```bash
      mkdir -p "$TEMP/tianxi-tmp"
      PYTEST_DEBUG_TEMPROOT="$TEMP/tianxi-tmp" uv run pytest
      ```

      根因修复要管理员权限（`takeown` + `icacls /reset`），或等系统重启后清理 `%TEMP%`。

### 数据侧

- [x] **数据加载层 + schema 落差预处理**（§12.3 第 9 条）——[`../eval/datasets/`](../eval/datasets/)：
      `locomo.py` / `longmemeval.py` / `preprocess.py` / `registry.py` + `tests/test_datasets.py`。
      **这不是"读个 JSON 就能跑"**，落定的落差：
      - **喂给 Add 的对话全文取 `data/public/conversations.jsonl`**（D16）——它本身即 JSONL、**每条 message 自带 `role`**（用 `locomo_refined.json` 会引入 209 条契约违规）
      - **`questions.jsonl` 只有 `evidence_messages`（证据轮，不是整段对话）**，但含 1,382 题与 gold ⇒ **与上一条按 `qa_id` 逐题对齐使用**
      - **`questions.jsonl` 的键与 pipeline 读的键对不上**（有 `qa_id` 无 `id`；有 `answer` 无 `gold_answer`）⇒ **直接喂会 `KeyError` + `ValueError`**
      - **两个数据集的 turn schema 不同**：LongMemEval 是 `role`+`content`，**LoCoMo 的 `conversations.jsonl` 是 `role`+`text`+`dia_id`** ⇒ 必须归一化，否则 **`pairing/` 看不见 `role`，整个 session 归成一个对且不报错**
      - **`category` 类型不一致**：`questions.jsonl` 里是字符串 `"4"`，`locomo_refined.json` 里是整数 `4` ⇒ 不归一化会**静默筛出 0 条**
      - PersonaMem 的 CSV **没有 `chat_history` 列、也没有 `incorrect_answers` 列**，而 pipeline 缺后者直接 `raise TypeError`
      - **答案字段名以 pipeline 代码为准**：规范字段是 **`generated_answer`**（CL-Bench 写 `model_output`）。**readme 写的 `predicted_answer` / `hypothesis` 没有 pipeline 读**
- [x] **LongMemEval 用 `lme_s_cleaned.json`**（加载器只认这一个文件名），**不要用 `lme_test.json`**——`test` 有 1,230 个 0-turn session，`s_cleaned` 有 0 个。**空 session 会污染按"20 条消息"切批的埋点逻辑**（§6.5）
- [x] **给 LongMemEval 合成 per-message `timestamp`**（`longmemeval._sessions`）——它的 turn **只有 `role`+`content`**，时间在 **session 级**的 `haystack_dates` 里。不合成则 `event_time` 全 NULL、`created_at` 只能发 `""`
      > **副作用是有价值的**：同一 session 内所有消息拿到同一日期 ⇒ **实证了 §6.1 的判断**——`event_time` 保证不了 session 内顺序，**位置才是唯一能保证邻域稳定的东西**（D28 起 = Add 内显式链）
- [x] **切批模拟**：本地只能按 20 条复现词数那一路（§12.3 第 5 条）——[`../eval/harness/batching.py`](../eval/harness/batching.py)
      **量级已测（全量两份数据）**：LoCoMo 单条消息**最长 87 词**（中位 20），**从不触到 2,000 词上限** ⇒ 两条路径在它上面**完全重合**；
      LongMemEval 中位 75 / 均值 159 / **最长 11,661 词**，**60 条消息超 2,000 词**，且**40%（9,528/23,867）的 session 首批是被词数上限切开的**。
      ⇒ **词数那一路只在 LongMemEval 上有分量**，而它正是 S2 未清的那一半。**别用 LoCoMo 的"完全重合"去推断 LongMemEval。**
- [x] 题量分布核对（§12.2）——`multi-session` 133 / `temporal-reasoning` 133 / `knowledge-update` 78 / `single-session-user` 70 / `single-session-assistant` 56 / `single-session-preference` 30；`_abs` 拒答题 30 道
- [ ] **固定"计数类"问题的口径**（§12.2）——**换口径数字就变**，而它正是附录 A"实体层做不做"的依据。上列数字对应 `how many|how much|how often|number of|count|how long`
- [ ] **重点指标是端到端，不是 Recall@K**（§12.2 / §14）

---

## Step 1 — 存储 + 服务 + 混合检索

> **检索只有一种模式：混合**（BM25 + Dense 两路 `prefetch` → Weighted RRF）。**没有裸 BM25 模式**（D15）。

- [x] SQLite 真源：`qa_pairs` + **`applied_batches`**（§6.1）
- [x] Qdrant server 模式（**Docker**），单分片，payload 索引**在写入前**建（§6.3）
- [x] `Add` 路径六步，**幂等分两层**（§15 / §6.5）
- [x] **HTTP 层**：`POST /add` / `POST /search` + Add/Search 编排（③-c）——契约形状钉在 pydantic 模型上
- [x] `Embedder` 协议 + 两个实现；**架构保持 embedder-agnostic**（§7.4 / §2.3）
- [x] **落盘的向量缓存**，键 = 渲染文本哈希（§7.2）
- [x] **`common/config.py`：配置的唯一入口**（③-d）——`service/` 等**五个目录都不再读 `os.environ`**（静态测试钉住）；`rrf_k != 61` 与 `workers != 1` 都**拒绝启动**
- [x] `Search` 路径：混合检索，**精确 ≤ `top_k`**（§2.2）
- [x] Weighted RRF：**`k=61` 显式设**、`prefetch` 每路带 `using`、根级 `limit` 取请求 `top_k`（§7.3）
      [`../src/tianximem/retrieve/fusion.py`](../src/tianximem/retrieve/fusion.py) + [`../src/tianximem/store/qdrant_store.py`](../src/tianximem/store/qdrant_store.py)。
      ⚠ 参数（`prefetch_limit` / `weights` / `rrf_k`）由**装配处注入 `QdrantStore`**——放在编排者手里会**静默无效**（它不执行那些参数）
- [x] **主路径必须能通过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）——**它是所有对照的参照点**（§13）
      **本地自动化**（③-e）：`make contract-check` → [`../eval/smoke/preflight.py`](../eval/smoke/preflight.py)，
      **14 条检查全过**（打真 HTTP、真 embedding、真 Qdrant；自启隔离实例，跑完 drop 集合）。
      ⚠ **"本地过" ≠ "Smoke 过"**：只剩**"相邻项拼接"**本地做不到（AML 侧怎么拼 `content`）——那要 harness。
- [ ] **跑 T2 实验**（§13，半天工作量）：133 道 multi-session 题（12 道拒答题单列）人工分三类
      🟡 **脚手架已就位**：`make t2-dump` 出待填表（纯 BM25 直查 Qdrant），**人填 `label`**，`make t2` 汇总分布与判读。
      **这一项没做完就等于没做**——表填不完，结论不许出（半张表的分布看起来像个结果）

### 独立后续切片：Checker 的两路分离查询（**已登记，未实现；不阻塞 Step 1**）

**链条**：`Qdrant arm_search`（单路查询）→ `service` 提供 BM25 / Dense 两路排名 → `Checker` 记录 `criterion_would_say` / **A4 分布**。

**它解决什么**：[`../src/tianximem/retrieve/checker.py`](../src/tianximem/retrieve/checker.py) 的判据**已经写全并测到**（§8 的三条规则），但 v1 的调用方**不传**两路排名，于是 `criterion_would_say` 恒为 `None` ⇒ **D13 想要的那份 A4 反事实分布拿不到**。

**为什么现在不做**：要跑那两次查询，得给 [`../src/tianximem/store/qdrant_store.py`](../src/tianximem/store/qdrant_store.py) 加一个**单路查询**（现有的 `hybrid_search` 只做融合）。它也**不是通过 Smoke 契约校验所必需的**（Smoke 只看响应形状）。

---

## Step 2 — Neighbor Expansion + 双预算  ✅ **已完成**

> **Dense 与 RRF 归 Step 1**（检索一次到位，D15），本阶段只剩扩窗与预算。

- [x] Neighbor Expansion：种子 `neighbor.expansion_seed_limit` + 窗口 `neighbor.radius`（§10）
      ——**具体取值不在本文件复述，以 [`config-reference.md`](./config-reference.md) §6 为准**
      （⚠ 那两个数改过两次，且 **D31 起半径为 0 = 默认关**；在这里抄一份必然漂移）
- [x] **全部 rerank 候选一条不删**，只对前 N 条扩窗；新扩出来的邻居 `rerank_rank = None`
- [x] 同 `(user_id, session_id)` 才扩；**禁止跨 session**
- [x] **Context Segment Merge**：连续块合成段，段内 `local_index` 序、段间 `best_rank` 序（§11.2）——连续性判据 = **Add 内显式链**（D28）
- [x] 双预算截断：**段数（`top_k`）+ token 数**（§6.4）——段是**原子单位**，装不下就停
- [x] `common/tokens.py`：`o200k_base` 计数，对**最终拼好的字符串**数
- [x] `rank/reranker.py` 的**接缝**（协议 + 降级）——远端实现已接，见 Step 3

> ⚠ **`top_k` 约束的是段数，不是 raw memory 数**——所以它只能在合并**之后**生效。
> 在扩窗阶段按 raw 数截断会把本该成段的邻居砍掉，而**返回的每一段看起来都合法**。

---

## Step 3 — Rerank + Packaging

> **本阶段定稿两件"贵"东西**：渲染模板与 reranker（**选型已定：`Qwen3-Reranker-4B`**）。

- [x] **reranker 接入**——**提交时不得更换**（D12）
      落点：`rank/reranker.RemoteReranker` → `POST {TIANXIMEM_RERANKER_BASE_URL}/rerank`。
      **线格式是实测的**（`top_n` 会静默截断、`model` 被忽略、响应按分数降序——三条都写在该文件顶部）。
      连通性与"它在链上真的起作用"用 `make probe-reranker` 验（真网关，**不消耗 Smoke 配额**）。
      ⚠ 端点**部署**仍不在本项目范围内（D12）；端点挂了 ⇒ 降级回 RRF 顺序并记 `rerank_degraded`。
- [x] **渲染模板定稿为 `v1`**：`Q: {q}` / `A: {a}`，**正文带日粒度日期锚点**（D21，见下一条）。
      声明处是 [`../src/tianximem/rank/CLAUDE.md`](../src/tianximem/rank/CLAUDE.md) §4；实现是 [`../src/tianximem/common/render.py`](../src/tianximem/common/render.py)。
      ⚠ **「定稿」不等于「不可推翻」**：推翻它要付「重建索引」的钱，而 T1 就是那笔钱的用途
- [x] **跑 T1 实验**（§13），与 `created_at` 粒度那条同批测（§11.3）—— ✅ **2026-09-25 已跑**
      结论：日期写**段首**无效、写**每一对旁边**有效（multi-hop +8.9pt / temporal +3.5pt，整体 0.601 → **0.633**）
      ⇒ **`packaging.inject_abs_time` 默认为 `true`**（**D21**），代价是**索引重建**（`tools/reindex.py`）。
      明细见 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)；旧两臂快照已清理，恢复方式见 [`配置快照说明`](../configs/runs/README.md)
      （`make t1`：不加参数只打印计划，`--freeze` 冻结、`--execute` 开跑、`--compare` 比结果）
- [x] `created_at` 只给日粒度；`event_time` 为 NULL 时发 `""`（§11.3）——固定 UTC，无旋钮
      ⚠ **粒度变细 / 相对↔绝对那两条规则仍属 T1 的待验证项**，落地的只是"发日期、不发秒"

> ⚠ **"接上了"不等于"有效"**：本阶段只保证**链路通**。精排值不值要在代理评测上跑对照（A3）才回答得了
> ——**不要拿探针里的几个样例下结论**。

---

## Step 4 — Conditional Agentic Search

- [ ] 最多 4 轮、每轮 `top_k=5`、三个工具（§9）
- [ ] **产出与初始候选合并，不是替换**，按 `id` 去重（§9）
- [ ] Evidence Checker 的名次一致性判据（§8）
- [ ] 监控 Agent Trigger Rate（§14）——太高 → Checker 太保守；接近 0 → agent 没起作用

---

## Step 5 — 切换到提交模型

> **本阶段只做一件事：换模型 + 重标定。不与任何设计改动合并。**

- [ ] 切到 `text-embedding-v4` + `gpt-4o-mini`（§2.3）—— 🟡 **embedder 一侧已落地**（2026-09-28：
      [`../src/tianximem/embed/text_embedding_v4.py`](../src/tianximem/embed/text_embedding_v4.py) +
      [`../configs/submit.yaml`](../configs/submit.yaml) + 装配按模型名挑实现），**剩下的只是把
      `AML_EMB_*` 指到真端点**（端点尚未到位）。⚠ **`gpt-4o-mini` 在 v1 里没有调用点**
      （`Add` 侧不调用、`Search` 只在 `agent/` 里调用，而 v1 不做 agentic，D13）⇒ 无需接线。
- [ ] **按新维度重建向量集合**（§2.3 / §16）——embedding 缓存整体失效（§7.2）
- [ ] **重标定全部阈值与权重**（§12.1 R1）
- [ ] **重跑 T2**，确认切换没引入系统性偏移——**它对外部模型依赖最小（纯 BM25 检索 + 人工判读，不调用任何模型），先用它确认，再去信其他实验**（§12.1 R1 对冲 2）
- [ ] **重新量一次单请求实际返回的对数**（[`open-questions.md`](./open-questions.md) **E7**）——本地分词器与 `o200k_base` 不同，**本地量的不能直接搬**
- [ ] 确认无硬编码残留：扫一遍 `docs/config-reference.md` §11

---

## Step 6 — 对照实验 + Smoke + Full 定稿

- [ ] 跑完 §13 全部对照（**B1 已跑**：agent 模式 3 段 346 题，我们领先 +29.5pt；**v1 还剩 A0/T2**——**A4 不在 v1**（**D26**）：它要 agent 真的存在，而 v1 不做 agentic（D13），arm 定义留在 [`experiments.md`](./experiments.md)）
- [ ] **Smoke 验证契约合规**
- [ ] **Smoke 跑通后第一件事：设计 S1 的判别实验**（§17.1）
- [ ] **Full 定稿**——每 Key 每轨道 **2 次**，第二次隔 30 天；**一旦接受即版本冻结**

---

## 工作量提醒（容易被当成零成本的两项）

| 项 | 说明 |
| --- | --- |
| **B1 包装 ReFind** | ✅ **不用包装**——ReFind 自带 AML 兼容的 `/add` `/search` ⇒ 起它、把 `--base-url` 指过去即可。**实际成本是它的 agentic 检索**（每 query ≈ 6 秒） |
| **Step 0 的 schema 预处理层** | 数据集与 pipeline 之间存在落差，**不是"读个 JSON 就能跑"**（§12.3 第 9 条） |
