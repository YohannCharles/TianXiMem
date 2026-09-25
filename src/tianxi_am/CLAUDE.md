# src/tianxi_am/ — 检索服务本体

实现 AML 要求的两个 HTTP 端点（`Add` / `Search`）。**答案生成、评判、聚合都不在这里**——全部由 AML 完成（§1）。本包的输出是**按名次排列的证据**。

> **红线**：`Search` **不得生成最终答案，也不得把答案伪装成记忆记录**（§2.1）。reranker 只重排证据（§11.2）。

---

## 模块地图

| 目录 | 负责什么 | PRD |
| --- | --- | --- |
| [`service/`](./service/) | HTTP 层：两个业务端点 + `/health` 探活、请求/响应模型、按 `(user_id, session_id)` 串行化 | §2.1、§15 |
| [`pairing/`](./pairing/) | QA 对配对判据；批次续接三步；`pending` 判定 | §6.2、§6.5、§15 |
| [`store/`](./store/) | SQLite 真源 + Qdrant 派生索引 + 邻域查询的 SQL | §6.1、§6.3 |
| [`embed/`](./embed/) | `Embedder` 协议 + 两个实现 + **落盘向量缓存** | §7.4、§7.2 |
| [`retrieve/`](./retrieve/) | BM25；dense；**混合检索的策略与参数所有权**；Evidence Checker（**到 rerank 为止**） | §7.1–§7.3、§8 |
| [`rank/`](./rank/) | **远端 rerank（`RemoteReranker`）** → **Neighbor Expansion（§10）** → Context Segment Merge → Context Packaging | §10、§11 |
| [`agent/`](./agent/) | Conditional Agentic Search 循环与工具（**v1 不实现**，D13） | §9 |
| [`llm/`](./llm/) | LLM 后端抽象 | §2.3、§12.1 |
| [`common/`](./common/) | **渲染模板的唯一实现**；token 计数；配置加载 + **开关校验** | §11.3、§6.4、§15 |
| [`observability/`](./observability/) | §14 指标 + §6.5 三个 `pending` 计数器的**聚合**（发射在各自层） | §14、§6.5 |

**逐目录的依赖方向与不变式见 [`../../docs/architecture.md`](../../docs/architecture.md)。**

**每个目录下有自己的 `CLAUDE.md`**——写该目录的代码前先读它。

---

## 三条先读再动手的边界

1. **`store/` 是唯一接触 SQLite 与 Qdrant 的目录。** 上层拿到领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`。
2. **`common/render` 是渲染的唯一实现**（§7.2 / §11.3）：embedding 的输入与返回给 AML 的 `content` 必须是同一份渲染，不一致会让"检索命中的"与"模型读到的"**静默漂移**。`embed/` 与 `rank/` 都不自己拼字符串。
3. **`Search` 路径不调用任何生成式 LLM**，除非走了 `agent/`。**`Add` 路径完全不调用 LLM**（§7.2）——embedding 是唯一的 Add 侧成本。

---

## 命名与结构约定

- **src 布局**：包根在 `src/tianxi_am/`——`[tool.uv] package = true` + `[build-system]`（hatchling）+ `[tool.hatch.build.targets.wheel] packages = ["src/tianxi_am"]`（**必须显式声明**，否则 hatchling 找不到包）让 `uv sync` 以 editable 方式装上本仓库，`uvicorn` / `python -m` 不需要 `PYTHONPATH`。
  > ⚠ `make sync` 走的是 `--all-extras`，会把 `[local]`（torch / transformers 栈，数 GB）一起装上——**提交链路不需要它**。
- **一个模块一件事**：目录名对应 PRD 的一节或一条链路。**若发现某个模块需要同时满足两节的互斥要求，那是切分错了，先回去改 `docs/architecture.md`。**
- **不在本包内读环境变量**——统一走 `common/` 的配置加载。
