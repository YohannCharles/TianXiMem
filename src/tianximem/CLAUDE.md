# src/tianximem/ — 检索服务本体

实现 AML 要求的两个 HTTP 端点（`Add` / `Search`）。**答案生成、评判、聚合都不在这里**——全部由 AML 完成（§1）。本包的输出是**按名次排列的证据**。

> **红线**：`Search` **不得生成最终答案，也不得把答案伪装成记忆记录**（§2.1）。reranker 只重排证据（§11.2）。

---

## 模块地图

**每个目录负责什么与 PRD 归属一处声明在
[`../../docs/architecture.md`](../../docs/architecture.md) §2，依赖方向在 §3，不变式清单在 §4**——本文件不复制那些表。
本包共 11 个目录：

[`service/`](./service/) · [`pairing/`](./pairing/) · [`facts/`](./facts/) · [`store/`](./store/) ·
[`embed/`](./embed/) · [`retrieve/`](./retrieve/) · [`rank/`](./rank/) ·
[`agent/`](./agent/)（v1 不实现）· [`llm/`](./llm/)（未实现）· [`common/`](./common/) ·
[`observability/`](./observability/)

**每个模块目录下有自己的 `CLAUDE.md`**——写该目录的代码前先读它。

---

## 三条先读再动手的边界

**第 1、2 条的理由与限定词一处声明在 [`../../docs/architecture.md`](../../docs/architecture.md) §3**——本文件只留规则本身与本地增量：

1. **`store/` 是唯一接触「业务真源」（SQLite）与 Qdrant 的目录。** 上层拿到领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`。（本地增量：限定词与 `DiskVectorCache` 那个例外见 [`./store/CLAUDE.md`](./store/CLAUDE.md)。）
2. **`common/render` 是渲染的唯一实现**（§7.2 / §11.3）：embedding 的输入与返回给 AML 的 `content` 必须是同一份渲染，不一致会让"检索命中的"与"模型读到的"**静默漂移**。`embed/`、`rank/` 与共同事实链都不自己拼字符串——事实片段的 `content` 走 `render_evidence`，打包直接用它。
   ⚠ **唯一一个例外（I1 的松动）声明在 [`../../docs/architecture.md`](../../docs/architecture.md) §4**：`packaging.annotate_relatives` 在 `content` 上叠一层**纯函数注解**——**只走 `content`**、剥掉注解后逐字相同，所以**不碰索引**；别把它当成"可以再往 content 里随手加东西"的先例。
3. **`Search` 路径不调用任何生成式 LLM**，除非走了 `agent/`。**`Add` 路径完全不调用 LLM**（§7.2）——embedding 是唯一的 Add 侧成本。共同事实链同样不调 LLM：来源句法抽取与问题计划编译都是确定性解析（`facts/`），执行只筛选/投影来源事实（`retrieve/evidence`），不生成答案。

---

## 命名与结构约定

- **src 布局**：包根在 `src/tianximem/`——`[tool.uv] package = true` + `[build-system]`（hatchling）+ `[tool.hatch.build.targets.wheel] packages = ["src/tianximem"]`（**必须显式声明**，否则 hatchling 找不到包）让 `uv sync` 以 editable 方式装上本仓库，`uvicorn` / `python -m` 不需要 `PYTHONPATH`。
  > ⚠ `make sync` 走的是 `--all-extras`，会把 `[local]`（torch / transformers 栈，数 GB）一起装上——**提交链路不需要它**。
- **一个模块一件事**：目录名对应 PRD 的一节或一条链路。**若发现某个模块需要同时满足两节的互斥要求，那是切分错了，先回去改 `docs/architecture.md`。**
- **不在本包内读环境变量**——统一走 `common/` 的配置加载。
