<!--
维护者注释（HTML 注释在注入 context 前会被剥离，不花 token）：
  本文件是 Claude Code 每次会话启动时唯一必然加载的文档。
  内容准则：只放「任何动作都要守」的约束 + 路由指针。
    · 模块专属的约束 → 该模块的 CLAUDE.md（Claude 读该目录下文件时按需加载）
    · 论证、依据、实验数据 → PRD 与 docs/（不要搬进这里）
    · 数字 → 只住 eval/reports/（见 docs/README.md 维护约定）
  本文件已按官方建议控制篇幅（<200 行）——超了会降低遵循度。
-->

# TianXi_AM — AML 参赛系统

**只提供 `Add` / `Search` 两个 HTTP 端点**（外加一个探活用、不碰下游的 `GET /health`——平台默认探它，S4）；答案生成、评判、聚合全部由 AML 完成。本系统的输出是**按名次排列的证据**，不是答案。

**权威规格**：[`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md)。本文件与各模块 `CLAUDE.md` 只做导航与速查；**冲突时一律以 PRD 为准**。

**当前状态**：**实现已开工，不是空脚手架。** 本表是模块级状态的**唯一权威**（别处一律指回这里）。

| 状态 | 模块 |
| --- | --- |
| ✅ **已实现** | [`store/`](src/tianxi_am/store/)（SQLite 真源 + Qdrant + schema）· [`pairing/`](src/tianxi_am/pairing/)（配对 / 续接三步 / 计数器）· [`embed/`](src/tianxi_am/embed/)（`Embedder` 协议 + Qwen3-Embedding-8B + 落盘缓存 + 查询侧 instruction 兼容层）· [`common/`](src/tianxi_am/common/) 的 **`render.py`**（渲染唯一实现）、**`config.py`**（**全包唯一读环境变量的地方**，③-d）与 **`tokens.py`**（`o200k_base` 计数，§6.4）· [`retrieve/`](src/tianxi_am/retrieve/)（**策略与参数所有权** + §8 判据 + `dedup_candidates`）· [`rank/`](src/tianxi_am/rank/) 的 **`reranker.py`**（**远端精排接入**）、**`neighbor.py`**（**扩窗 + 段合并**，§10/§11.2）与 **`packaging.py`**（**段级打包 + 双预算**）· [`service/`](src/tianxi_am/service/)（HTTP 层 + **Add/Search 端到端编排**，③-c + **`GET /health` 探活**，S4）· [`configs/`](configs/) 的 `default.yaml` + `local.yaml` · [`eval/smoke/preflight.py`](eval/smoke/preflight.py)（**契约预检，③-e**）· [`eval/datasets/`](eval/datasets/)（**加载层 + schema 落差预处理**）· [`eval/harness/`](eval/harness/)（**HTTP 驱动 / 切批 / 裁判包装 / run record + `api_config.py`**）· [`eval/reports/schema.py`](eval/reports/schema.py)（**run record 的形状**）· [`tools/check_env.py`](tools/check_env.py)（**环境自检，含 V7 探针**）、[`tools/probe_reranker.py`](tools/probe_reranker.py)（**精排探针**）与 [`tools/t2_retrieval_dump.py`](tools/t2_retrieval_dump.py)（**T2 的纯 BM25 转储**）· [`eval/experiments/`](eval/experiments/)（**通用 runner `run.py` + T1/T2/A3 三个 arm**）· [`eval/reports/ledger.md`](eval/reports/ledger.md)（**结果台账**）· [`configs/runs/`](configs/runs/)（**各 arm 的冻结配置**）——测试在 [`tests/`](tests/) |
| ⬜ **未实现** | [`llm/`](src/tianxi_am/llm/) · [`observability/`](src/tianxi_am/observability/) · `embed/` 的 `text_embedding_v4.py` · `configs/submit.yaml` · **`eval/` 还没写的**：`datasets/contracts.py`、`experiments/` 的 **A4/A0 两个 arm**、`baselines/`（**含 B1 包装 ReFind**）、`smoke/` 的 S1/S2/S3 探针与 `quota.py` · 尚未接线的消融开关（**逐项状态以 [`docs/config-reference.md`](docs/config-reference.md) §2 的表为准**——`checker.*` / `agent.*` / `rrf` 未接，`neighbor` 只有 `radius` 可关） |
| ⛔ **v1 不做** | [`agent/`](src/tianxi_am/agent/)（D13） |

> **⚠ 状态描述是本仓最易过期的东西**：改动状态时，**连同搜一遍所有声称"未实现 / 未开始"的地方**——本表是唯一权威，别处只应指回这里。

**目标**：AML 榜分优于 ReFind 的 44.97。核心判断是**检索不是瓶颈（召回已 96–99%）、选择与排序才是**，所以主线是 Rerank + Context Packaging（§4 / §11）。

---

## 路由：动 X 之前先读 Y

| 要动的东西 | 先读 |
| --- | --- |
| `src/tianxi_am/<模块>/` 下任何代码 | 该目录的 `CLAUDE.md`（要写什么、边界在哪、本层的坑） |
| 契约层（service、Add/Search 形状） | [`docs/contract.md`](docs/contract.md) |
| 任何阈值 / 权重 / 开关 | [`docs/config-reference.md`](docs/config-reference.md)（**开关的唯一声明处**）+ [`configs/CLAUDE.md`](configs/CLAUDE.md)（**每个键住在 `.env` 还是 yaml**）。**全包只有 [`common/config.py`](src/tianxi_am/common/config.py) 读环境变量**——有静态测试钉住 |
| 一个"已锁定"的决定 | [`docs/decisions.md`](docs/decisions.md)（D1–D20 + 待决事项） |
| 跑对照实验 | [`docs/experiments.md`](docs/experiments.md)（协议）+ [`eval/experiments/CLAUDE.md`](eval/experiments/CLAUDE.md)（怎么跑） |
| 数据集加载 / harness | [`docs/benchmark-data.md`](docs/benchmark-data.md) + [`eval/datasets/CLAUDE.md`](eval/datasets/CLAUDE.md) |
| **取回 / 校验 `benchmark_data/`**（新机器、数据缺了、要确认手上的是不是同一份） | [`docs/benchmark-data.md`](docs/benchmark-data.md) 的"出处链" + **"本地修订"**（7 个 pipeline 有一处已声明的偏离：上游那份跑不起来）+ [`tools/fetch_benchmark_data.py`](tools/fetch_benchmark_data.py)。`make fetch-data` / `make data-check` / `make data-patch` |
| 提交周期与截止日 | [`docs/submission.md`](docs/submission.md) §0（**第二期 09-20 已开，材料截止 10-31**） |
| 发 Smoke / Full | [`docs/submission.md`](docs/submission.md)（配额与版本冻结） |
| 模块边界 / 谁依赖谁 / 七个评分维度各由谁回应 | [`docs/architecture.md`](docs/architecture.md) |
| 每个 Step 的交付与进度 | [`docs/roadmap.md`](docs/roadmap.md) |
| 还有哪些未知没清掉 | [`docs/open-questions.md`](docs/open-questions.md) |

---

## 硬约束（AML 契约，违反通常**不报错**）

| 约束 | 后果 | 出处 |
| --- | --- | --- |
| **`top_k` 由 AML 固定为 100** | 返回超过它是**契约错误**，**不会被静默截断**——必须**精确计数** | §2.2 |
| **答案阶段按返回顺序取 117,760 token 前缀** | 排在后面的证据**整段作废**，白白占名额 | §2.2 / §6.4 |
| **`user_id` 是唯一的检索隔离字段** | 跨 user 检索被禁止。`session_id` 只是分组字段，**不是 Search 的过滤器** | §2.2 |
| **Add 最多被重试 32 次**（`request_id` 与 payload 不变） | 必须幂等 | §2.2 |
| **响应前必须持久化完成且立即可搜索** | 不允许异步建索引 | §2.1 |
| **`Search` 不得生成最终答案**，也不得把答案伪装成记忆记录 | reranker 只重排证据 | §2.1 / §11.2 |
| **Add 按 `(user_id, session_id)` 串行化，且必须 `--workers 1`** | 锁是进程内的，多 worker **静默失效** | §15 |
| **单请求最长 30 分钟；Full run 连续跑 0.5–2 天** | 阻塞调用会互相饿死，问题直到 Full 才炸 | §2.2 / §15 |

### 三个静默出错的重灾区

| 陷阱 | 为什么静默 | 在本文件之外 |
| --- | --- | --- |
| **幂等分两层**：内容层靠"填空 + 追加、绝不覆盖"；但**位置分配不幂等**——`pair_idx` 是读-改-写，事务提交后崩溃再重试会落到**新的 `pair_idx`**，产生重复记录。**批次级必须查 `applied_batches` 旁表** | 写入是 upsert，重复落库不报错 | [`pairing/CLAUDE.md`](src/tianxi_am/pairing/CLAUDE.md) |
| **Qdrant RRF 的 `k` 默认是 `2`**，不是文献里的 60。**必须显式设 `k=61`**（Qdrant 秩 0-based：`1/(0+61) = 1/(1+60)`） | 不设会得到一个与所有参考实现都不同的融合行为，极难排查 | [`docs/decisions.md`](docs/decisions.md) D5 |
| **Qdrant local 模式会静默丢弃 payload 索引**（`create_payload_index` 只打一行警告就返回） | 而 `user_id` / `session_id` / `event_time` 三个筛选**全依赖**它 | [`deploy/CLAUDE.md`](deploy/CLAUDE.md) |

### 两个 ID / 缓存键，方向正好相反

| 用途 | 键 | 理由 |
| --- | --- | --- |
| 记录 `id` | **位置派生** `hash(user_id, session_id, pair_idx)` | 用内容哈希会在补全时变 `id`，留下**孤儿 point** |
| embedding 缓存键 | **渲染后文本的哈希**，**不能用 `id`** | 补全时内容变了而 `id` 不变，用 `id` 会拿到**陈旧向量** |

**两者互换都会静默出错。** 缓存**必须落盘**，且要能在 Step 5 切模型时整体失效（§7.2 / §12.1 R1）。

### 时间处理：不要在 content 里注入绝对时间戳前缀

裁判 prompt 的 TIME 块有**两条独立规则**都会因此判负：**粒度变细**（DAY → Second，与 gold 是否相对无关）、**相对↔绝对互转**。而答案 prompt 第 7 条**却要求**转换相对时间——所以加绝对时间戳**可能反而有害**（§11.3）。

推论：**筛选走 `event_time` 列；正文只保留原始时间表述。** 但 **`created_at` 必须带，且只给到日粒度**（如 `2026-07-26`）——CL-Bench 那条路径靠它渲染时间，而秒级粒度会诱发模型按秒级回答。

> 这条来自 LoCoMo-Refined / LongMemEval 共用的契约，**不是全赛道规则**：BEAM 的裁判正好相反（允许等价形式），CL-Bench 由 AML 侧主动注入时间戳。**代理评测按"不加"执行，但不要外推。**

### 模型规定（§2.3，不可协商）

| 组件 | 规定 |
| --- | --- |
| **Embedding** | **只能用 `text-embedding-v4`** |
| **LLM 相关组件** | **只能用 `gpt-4o-mini`** |
| **Reranker** | **不作规定**——整份规则里唯一不限模型的组件 |

由此推出一条设计约束：**架构必须 embedder-agnostic**。开发期的 Qwen3-Embedding-8B 与提交期的 `text-embedding-v4` **都不提供** sparse 或 ColBERT 输出，因此**任何依赖多向量能力的代码都是死重**。向量维度**由接口提供、不能写死**。

**开发期用自建网关上的 Qwen3-Embedding-8B + qwen3.5-9b 替代**（§12.1 风险 R1，团队已接受）——代价是**所有阈值、权重、排序策略在切换后都不保证成立**。四条对冲见 D2；与写代码直接相关的两条：**阈值一律配置化**、**切换单独占一个阶段**。

### 法律约束（§12.5 / §17.3 P2）

**数据不得用于训练。「微调模型」不是暂不实现，是不允许**——AML 数据条款 + CLBench 许可证双重封死，**无对冲余地**。

两处最容易伸手的地方：`gpt-4o-mini` 不能微调，但**本地 qwen3.5-9b 可以**；**reranker 也是自托管开放权重的**。**都不允许。**

> 归档里没有这两条条款的出处（属单边来源，引用前须回原始页面复核），**但结论不变**。数据集许可表见 [`eval/datasets/CLAUDE.md`](eval/datasets/CLAUDE.md)。

---

## 评测机会成本（§2.4）——本项目最大的约束

**AML 不提供本地评测**（不给 gold answer、不给评分标准、不提供批量数据下载）。真实信号只有两条路：

| 层 | 用途 | 次数限制 |
| --- | --- | --- |
| **代理评测**（本地，`eval/`） | **全部迭代、消融、调参** | 无限 |
| **Smoke** | 验证契约合规、端到端连通 | 每轨道 **≤30 次**，每小时 1 次，**不进榜** |
| **Full** | 最终定稿 | 每 Key 每轨道 **2 次**，第二次隔 30 天；**一旦接受即版本冻结** |

**没有"跑一遍看看"的余地。** 任何能在代理评测上回答的问题，都不该花 Smoke 的额度。Smoke 只做两件事：验证契约合规、消除本地无从验证的未知（§17.1 的 S1–S3）。

**代理评测的边界**：只覆盖 LoCoMo-Refined + LongMemEval——这两份恰好是六份数据集里**唯一共用同一套契约**的。其余四份的记忆注入字段与裁判规则各不相同（PersonaMem 甚至**根本不读检索字段**），所以**代理分数不能线性外推到全赛道**（§12.4）。只在代理上做**相对**比较。

---

## 文档与代码纪律

- **引用 PRD 一律带 § 号**（如 `§6.5`）。不带 § 号的断言视为未经验证。
- **"待验证"必须显式标注**，不要写成已知。
- **数字只住在 [`eval/reports/`](eval/reports/)**——其余文档引用数字时**指回去**，不要复制。
- **单数来源**：一条约束只有一个家。写了第二遍就是漂移的开始——发现重复，改成指针。
- **配置化 ≠ 可调**。`rrf.k = 61` 与 `top_k = 100` 都是**正确性常量**，写进配置是为了追溯与切换，**不是为了调**。
- **编号纪律**：`S1`–`S3` 属 §17.1、`P1`–`P3` 属 §17.3、`R1` 属 §12.1、`E1`–`E7` 属 §17.2（**E2 已随 D15 删除，该号不再启用**），**不要挪作他用**；Step 5 专属事项用 `S5-*`。
- **任何"顺手在写入时抽个摘要 / 抽事实 / 归并实体"的想法都属于 v2**——它会同时破坏 Add 侧的成本属性与 §11.3 的"原文优先"（v1 不产生任何合成文本）。

```bash
cp .env.example .env      # 填密钥与路径
make sync                 # uv sync --all-extras
make help                 # 看全部目标
```

> `Makefile` 里**尚未实现**的模块对应的目标会**明确失败**——这是有意的，**避免误以为某一步已经实现**。
>
> ⚠ **Windows 那台机器上 `uv run pytest` 会报 111 个 `PermissionError`**：`%TEMP%\pytest-of-r0304`
> 的 ACL 损坏，**与代码无关**。绕法见 [`docs/roadmap.md`](docs/roadmap.md) Step 0 的环境项
> （`PYTEST_DEBUG_TEMPROOT=<一个新建目录>`）。
