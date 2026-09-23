# 路线图与阶段门

> 最后核对：2026-09-23，对应 PRD §16（+ §1 / §6.4 / §11.2 / §12.1 的阶段要求）。**冲突以 PRD 为准。**

**状态**：Step 1 的实现**已开工**（`store/` `pairing/` `embed/` `common/render.py` 与 8 个测试文件已存在，见根 `CLAUDE.md`）；下方勾选状态尚未逐项回填，**以根 `CLAUDE.md` 为准**。

---

## 总览

| 阶段 | 交付 | 状态 |
| ---- | ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval）+ **确认显存预算** | ⬜ |
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

- [ ] **`api_config.py` —— 五个 pipeline 今天都 import 失败**
      它们做 `sys.path.insert(0, Path(__file__).resolve().parents[2])` 再 `from api_config import (...)`，而 `parents[2]` 解析到 **`/home/buptc/project/`（仓库外一层）**。**该文件不存在。**
      要导出七个名字：`ANSWER_API_BASE` / `ANSWER_API_KEY` / `ANSWER_MODEL` / `JUDGE_API_BASE` / `JUDGE_API_KEY` / `JUDGE_MODEL` / `JUDGE_VERSION`（最后一个是**死引用**，五个 pipeline 全部 import 但无一使用）。
      ⚠ 它是**环境依赖，不是仓库内容** ⇒ 在 [`decisions.md`](./decisions.md) 记一条，并写进提交包说明（"clone 下来不能直接跑"）。
- [x] **`docker` 已就位**（2026-09-23 晚）——Docker Desktop 29.8.0 / `desktop-linux` context / WSL2 后端。`qdrant/qdrant:v1.17.0` 的 tag **已在 registry 核实存在**（§7.3 的版本门槛成立）。`nvidia-smi` **已不需要**（模型不在本机）。
      **拓扑已定**（D12）：harness 与检索服务 + Qdrant 都在**本机**、三段模型经自建网关远程访问、**reranker 端点待部署**。
- [ ] **确认显存预算**（§1 / §11.2）——⏸ **本项目延后，不阻塞开发**（D14）。
      qwen3.5-9b（128K）+ BGE-M3 + reranker 需**共存于一张 L20（48GB）**；而**模型部署不在本项目范围内**（三段模型在自建网关，reranker 待部署）。
      **⚠ 这是有意偏离，不是已满足**：该前提**未被核对**，若挤不下则 R1 的四条对冲需要重写——**风险已接受**（见 [`decisions.md`](./decisions.md) D14）。

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

- [ ] SQLite 真源：`qa_pairs` + **`applied_batches`**（§6.1）
- [ ] Qdrant server 模式（**Docker**），单分片，payload 索引**在写入前**建（§6.3）
- [ ] `Add` 路径六步，**幂等分两层**（§15 / §6.5）
- [ ] `Embedder` 协议 + 两个实现；**架构保持 embedder-agnostic**（§7.4 / §2.3）
- [ ] **落盘的向量缓存**，键 = 渲染文本哈希（§7.2）
- [ ] `Search` 路径：混合检索，**精确 ≤ `top_k`**（§2.2）
- [ ] Weighted RRF：**`k=61` 显式设**、`prefetch` 每路带 `using`、根级 `limit` 取请求 `top_k`（§7.3）
- [ ] **主路径必须能通过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）——**它是所有对照的参照点**（§13）
- [ ] **跑 T2 实验**（§13，半天工作量）：133 道 multi-session 题（12 道拒答题单列）人工分三类
- [ ] 三个 `pending` 计数器埋点（§6.5）

---

## Step 2 — Neighbor Expansion + 双预算

> **Dense 与 RRF 归 Step 1**（检索一次到位，D15），本阶段只剩扩窗与预算。

- [ ] Neighbor Expansion：种子 20、窗口 ±1、**槽位占 `top_k` 名额**（§10）
- [ ] 双预算截断（槽位数 + token 数）（§6.4）

---

## Step 3 — Rerank + Packaging

> **本阶段定稿两件"贵"东西**：渲染模板与 reranker 选型。

- [ ] **reranker 选型**（§11.2，2026-09-23 决定：Step 3 前再选）——约束：与另外两个模型共存于一张 L20；数据集以英文为主
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
