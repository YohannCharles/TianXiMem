# 路线图与阶段门

> 最后核对：2026-09-24，对应 PRD §16（+ §1 / §6.4 / §11.2 / §12.1 的阶段要求）。**冲突以 PRD 为准。**

**状态**：Step 1 的实现**已开工**（`store/` `pairing/` `embed/` `common/render.py` 与 8 个测试文件已存在，见根 `CLAUDE.md`）；下方勾选状态尚未逐项回填，**以根 `CLAUDE.md` 为准**。

---

## 总览

| 阶段 | 交付 | 状态 |
| ---- | ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval） | ⬜ |
| Step 1 | 存储层 + Add/Search 服务 + **混合检索**（BM25 + Dense + RRF，含 T2 实验） | ⬜ |
| Step 2 | **Neighbor Expansion + 双预算截断** | ⬜ |
| Step 3 | Rerank + Context Packaging（含 T1 实验） | ⬜ |
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

### 环境阻塞项（不解决则本步无法完成）

- [x] **`SqliteStore` 单连接 × 多线程服务 —— 已修**（2026-09-24，**结案**）
      `SqliteStore` 曾持有**单个** `sqlite3.Connection`（在启动线程里建），而 FastAPI 的
      `def` 路由跑在**线程池**里 ⇒ **每个请求**都抛
      `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.`
      （`TestClient` 也在另一个线程里跑 app，所以 HTTP 级用例同样跑不了。）
      ⇒ **修法是短生命周期连接**（连接限定在"一次逻辑操作 / 一个事务"内，用完即关），
      不用 thread-local、也不用单一共享连接。**决策与三条被排除的路见 [D17](./decisions.md)**
      （原待决事项 8 已结案）。原先记着的 6 个 `xfail(strict=True)` **已全部摘掉**，
      它们现在是回归用例：3 个 HTTP 往返 + 3 个并发用例，另加
      `test_store.py` 的 3 个连接生命周期用例与 2 个**真实并发写**压力用例。
- [ ] **`api_config.py` —— 7 个 pipeline 今天都 import 失败**（**处置已定，2026-09-24 更新**）
      它们做 `sys.path.insert(0, Path(__file__).resolve().parents[2])` 再 `from api_config import (...)`，而 `parents[2]` 解析到**仓库外一层**。
      它要导出七个名字：`ANSWER_API_BASE` / `ANSWER_API_KEY` / `ANSWER_MODEL` / `JUDGE_API_BASE` / `JUDGE_API_KEY` / `JUDGE_MODEL` / `JUDGE_VERSION`（最后一个是**死引用**，全部 import 但无一使用）。
      ✅ **不用从零写**：AML 自己就发布了它——公开仓根目录的 `api_config.py`，520 字节、无凭据，只是 `os.environ.get(...)` 的适配器。
      ✅ **也不用放在仓库外**：放仓库内，由 harness 在 subprocess 里注入 `PYTHONPATH`（`sys.path.insert` 塞进一个不存在的路径不会中断 import，会继续往后找）。
      细则与理由见 [`../eval/harness/CLAUDE.md`](../eval/harness/CLAUDE.md) 与 [`.env.example`](../.env.example) 末尾。
- [x] **`docker` 已就位**（2026-09-23 晚）——Docker Desktop 29.8.0 / `desktop-linux` context / WSL2 后端。`qdrant/qdrant:v1.17.0` 的 tag **已在 registry 核实存在**（§7.3 的版本门槛成立）。
      **拓扑已定**（D12）：harness 与检索服务 + Qdrant 都在**本机**、三段模型经自建网关远程访问、**reranker 端点待部署**。
- [ ] ⚠ **本机 `tmp_path` 故障**（2026-09-24）——`C:\Users\r0304\AppData\Local\Temp\pytest-of-r0304`
      的 ACL 已损坏：`ls` / `icacls` / `Remove-Item` **全部拒绝访问**，于是**所有用 `tmp_path` 的用例**
      在 fixture setup 阶段就报 `PermissionError`（`uv run pytest` 表现为 **128 passed, 111 errors**）。
      **与代码无关，是机器状态。** 绕法（已验证可用，**不写进 `Makefile`**——那是机器专属路径）：

      ```bash
      mkdir -p "$TEMP/tianxi-tmp"
      PYTEST_DEBUG_TEMPROOT="$TEMP/tianxi-tmp" uv run pytest
      ```

      根因修复要管理员权限（`takeown` + `icacls /reset`），或等系统重启后清理 `%TEMP%`。

### 数据侧

- [ ] **数据加载层 + schema 落差预处理**（§12.3 第 9 条）——**不是"读个 JSON 就能跑"**。已核实的落差：
      - **喂给 Add 的对话全文取 `data/public/conversations.jsonl`**（D16）——它本身即 JSONL、**每条 message 自带 `role`**，天然满足 content 首尾无空白的要求（用 `locomo_refined.json` 会引入 209 条契约违规）
      - **`questions.jsonl` 只有 `evidence_messages`（证据轮，不是整段对话）**，但含 1,382 题与 gold ⇒ **与上一条按 `qa_id` 逐题对齐使用**（仅 6 处答案 int/str 差异）
      - **`questions.jsonl` 的键与 pipeline 读的键对不上**（有 `qa_id` 无 `id`；有 `answer` 无 `gold_answer`）⇒ **直接喂会 `KeyError` + `ValueError`**
      - **两个数据集的 turn schema 不同**：LongMemEval 是 `role`+`content`，**LoCoMo 的 `conversations.jsonl` 是 `role`+`text`+`dia_id`** ⇒ 必须归一化，否则 **`pairing/` 看不见 `role`，整个 session 归成一个对且不报错**
      - **`category` 类型不一致**：`questions.jsonl` 里是字符串 `"4"`，`locomo_refined.json` 里是整数 `4` ⇒ 不归一化会**静默筛出 0 条**
      - PersonaMem 的 CSV **没有 `chat_history` 列、也没有 `incorrect_answers` 列**，而 pipeline 缺后者直接 `raise TypeError`
      - **答案字段名以 pipeline 代码为准**：规范字段是 **`generated_answer`**（CL-Bench 写 `model_output`）。**readme 写的 `predicted_answer` / `hypothesis` 没有 pipeline 读**
- [ ] **LongMemEval 用 `lme_s_cleaned.json`**，**不要用 `lme_test.json`**——已复算：`test` 有 **1,230 个 0-turn session**，`s_cleaned` 有 0 个。**空 session 会污染按"20 条消息"切批的埋点逻辑**（§6.5）
- [ ] **给 LongMemEval 合成 per-message `timestamp`**——它的 turn **只有 `role`+`content`**，时间在 **session 级**的 `haystack_dates` 里（形如 `"2023/05/20 (Sat) 02:21"`）。不合成则 `event_time` 全 NULL、`created_at` 只能发 `""`
      > **副作用是有价值的**：同一 session 内所有消息拿到同一日期 ⇒ **实证了 §6.1 的判断**——`event_time` 保证不了 session 内顺序，**`pair_idx` 是唯一能保证邻域稳定的东西**
- [ ] **固定"计数类"问题的口径**（§12.2）——**换口径数字就变**，而它正是附录 A"实体层做不做"的依据。上列数字对应 `how many|how much|how often|number of|count|how long`
- [ ] **切批模拟**：本地只能按 20 条复现词数那一路（§12.3 第 5 条）
- [ ] 题量分布核对（§12.2）：single-session-user 70 / single-session-assistant 56 / single-session-preference 30 / temporal-reasoning 133 / knowledge-update 78 / multi-session 133
- [ ] **重点指标是端到端，不是 Recall@K**（§12.2 / §14）

---

## Step 1 — 存储 + 服务 + 混合检索

> **检索只有一种模式：混合**（BM25 + Dense 两路 `prefetch` → Weighted RRF）。**没有裸 BM25 模式**（D15）。

- [x] SQLite 真源：`qa_pairs` + **`applied_batches`**（§6.1）
- [x] Qdrant server 模式（**Docker**），单分片，payload 索引**在写入前**建（§6.3）
- [x] `Add` 路径六步，**幂等分两层**（§15 / §6.5）
- [x] **HTTP 层**：`POST /add` / `POST /search` + Add/Search 编排（③-c，2026-09-24）——契约形状钉在 pydantic 模型上
- [x] `Embedder` 协议 + 两个实现；**架构保持 embedder-agnostic**（§7.4 / §2.3）
- [x] **落盘的向量缓存**，键 = 渲染文本哈希（§7.2）
- [x] **`common/config.py`：配置的唯一入口**（③-d，2026-09-24）——`service/` 等**五个目录都不再读 `os.environ`**（静态测试钉住）；`configs/default.yaml` + `local.yaml` 已建；`rrf_k != 61` 与 `workers != 1` 都**拒绝启动**
- [x] `Search` 路径：混合检索，**精确 ≤ `top_k`**（§2.2）
- [ ] Weighted RRF：**`k=61` 显式设**、`prefetch` 每路带 `using`、根级 `limit` 取请求 `top_k`（§7.3）
- [ ] **主路径必须能通过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）——**它是所有对照的参照点**（§13）
- [ ] **跑 T2 实验**（§13，半天工作量）：133 道 multi-session 题（12 道拒答题单列）人工分三类
- [ ] 三个 `pending` 计数器埋点（§6.5）——**发射已在 `pairing/instrument.py`**，聚合在 `observability/`（未接）

### 独立后续切片：Checker 的两路分离查询（**已登记，未实现；不阻塞 Step 1**）

**链条**：`Qdrant arm_search`（单路查询）→ `service` 提供 BM25 / Dense 两路排名 → `Checker` 记录 `criterion_would_say` / **A4 分布**。

**它解决什么**：[`../src/tianxi_am/retrieve/checker.py`](../src/tianxi_am/retrieve/checker.py) 的判据**已经写全并测到**（§8 的三条规则），但 v1 的调用方**不传**两路排名，于是 `criterion_would_say` 恒为 `None` ⇒ **D13 想要的那份 A4 反事实分布拿不到**。

**为什么现在不做**：要跑那两次查询，得给 [`../src/tianxi_am/store/qdrant_store.py`](../src/tianxi_am/store/qdrant_store.py) 加一个**单路查询**（现有的 `hybrid_search` 只做融合）——而 ③-b 的口径是"不改 ①② 已稳定代码"。它也**不是通过 Smoke 契约校验所必需的**（Smoke 只看响应形状）。

---

## Step 2 — Neighbor Expansion + 双预算

> **Dense 与 RRF 归 Step 1**（检索一次到位，D15），本阶段只剩扩窗与预算。

- [ ] Neighbor Expansion：种子 20、窗口 ±1、**槽位占 `top_k` 名额**（§10）
- [ ] 双预算截断（槽位数 + token 数）（§6.4）

---

## Step 3 — Rerank + Packaging

> **本阶段定稿两件"贵"东西**：渲染模板与 reranker（**选型已定：`Qwen3-Reranker-4B`**）。

- [ ] **reranker 接入**——**选型已定：`Qwen3-Reranker-4B`（2026-09-24）**。剩下的是端点部署（不在本项目范围）+ `rank/reranker.py` 的调用。**提交时不得更换**（D12）
- [ ] 顺序与预算配合：按名次依次扩窗（§11.2）
- [ ] 组内按 `pair_idx` 时间序；组间按种子名次（§11.2）
- [ ] **渲染模板定稿**——**改模板 = 重建索引**，别拖到 Step 5 之后（§11.3 / E6）
- [ ] **跑 T1 实验**（§13），与 `created_at` 粒度那条同批测（§11.3）
- [ ] `created_at` 只给日粒度；`event_time` 为 NULL 时发 `""`（§11.3）

---

## Step 4 — Conditional Agentic Search

- [ ] 最多 4 轮、每轮 `top_k=5`、三个工具（§9）
- [ ] **产出与初始候选合并，不是替换**，按 `id` 去重（§9）
- [ ] Evidence Checker 的名次一致性判据（§8）
- [ ] 监控 Agent Trigger Rate（§14）——太高 → Checker 太保守；接近 0 → agent 没起作用

---

## Step 5 — 切换到提交模型

> **本阶段只做一件事：换模型 + 重标定。不与任何设计改动合并。**

- [ ] 切到 `text-embedding-v4` + `gpt-4o-mini`（§2.3）
- [ ] **按新维度重建向量集合**（§2.3 / §16）——embedding 缓存整体失效（§7.2）
- [ ] **重标定全部阈值与权重**（§12.1 R1）
- [ ] **重跑 T2**，确认切换没引入系统性偏移——**它对外部模型依赖最小（纯 BM25 检索 + 人工判读，不调用任何模型），先用它确认，再去信其他实验**（§12.1 R1 对冲 2）
- [ ] **重新量一次单请求实际返回的对数**（§6.4 补录的对冲 / `docs/open-questions.md` E7）——本地分词器与 `o200k_base` 不同，**本地量的不能直接搬**
- [ ] 确认无硬编码残留：扫一遍 `docs/config-reference.md` §11

---

## Step 6 — 对照实验 + Smoke + Full 定稿

- [ ] 跑完 §13 全部对照（含 **B1**——**先估包装 ReFind 的工作量**，它未计入任何 Step）
- [ ] **Smoke 验证契约合规**
- [ ] **Smoke 跑通后第一件事：设计 S1 的判别实验**（§17.1）
- [ ] **Full 定稿**——每 Key 每轨道 **2 次**，第二次隔 30 天；**一旦接受即版本冻结**

---

## 工作量提醒（容易被当成零成本的两项）

| 项 | 说明 |
| --- | --- |
| **B1 包装 ReFind** | ReFind 是**方法实现**，要进我们的 harness，得**给它包一层 Add/Search 服务**（或把它的检索器接到我们的 harness 接口上）。**这部分工作量目前未计入任何 Step**（§13） |
| **Step 0 的 schema 预处理层** | 数据集与 pipeline 之间存在落差，**不是"读个 JSON 就能跑"**（§12.3 第 9 条） |
