# service/ — HTTP 层

**PRD**：§2.1（接口形状）、§2.2（硬性规则）、§15（工程与部署）

## 要写什么

```text
app.py        FastAPI 实例 + 对象图装配（`--factory` 入口）、lifespan
routes.py     POST /add、POST /search、GET /health（薄路由，只做形状映射）
schemas.py    请求与响应模型（pydantic）= §2.1 的字面翻译
pipeline.py   Add / Search 的**编排**（"按什么顺序调"；Search 先试共同取证，不适用则回退原链）
errors.py     异常 → 保持"可重试"的边界处理
capture.py    请求**原文**采集（诊断旁路，默认关）—— **S6** 的唯一直接观察口
```

> **`capture.py` 的边界**：它把官方发来的 `/add` / `/search` **原样**抄一份落盘
> （`capture.enabled`，默认关），用来核验官方真实的请求形状——尤其 `request_id`。
> **四条纪律与文件格式一处声明在** [`capture.py`](./capture.py) 的模块 docstring
> （在解析之前抄 / 不改下游 body / 不吞异常 / 写盘失败不影响响应），本文件不复制。
> ⚠ 它是本层**唯一**允许"吞掉自己内部错误"的地方（只留 WARNING），理由见那条纪律 4：
> **诊断不该有能力把一次 Add 变成 500**。

> **配置不在本目录**：由 [`../common/config.py`](../common/config.py) 统一提供，**注入** `build_services()`。
> **`service/` 自己不读 `os.environ`**——这一条有静态测试钉住
> （`tests/test_config.py::test_only_config_reads_the_environment`）。
>
> ⇒ 本目录拿到的是一个 [`AppConfig`](../common/config.py) 对象；**它不知道那些值从哪来**。

> **为什么有 `pipeline.py`**：本层"不做检索、不做配对、不碰存储"指的是**不重新实现**那些逻辑（全部往下调用）。而 Search 的链横跨 `facts/`、`retrieve/` 与 `rank/`、Add 的链横跨 `pairing/`、`facts/`、`store/`、`embed/`——**没有任何单个下层模块能拥有整条链**，所以"顺序"必须有人拥有，就在这里（路由仍然是薄的，`routes.py` 只有形状映射）。

## 这一层只做两件事

1. **把契约形状钉死**——请求/响应模型就是 §2.1 的字面翻译。
2. **保持可重试**——内部异常时**不要吞掉、不要返回部分成功**。

它**不**做检索、不做配对、不碰存储。所有业务都往下走。

> **契约硬约束在根 `CLAUDE.md`（已加载）**——本节只列本层要落地的那几条，不重述 AML 侧的规则原文。

---

## 本层必须守住

| # | 要求 | 出处 |
| --- | --- | --- |
| 1 | **精确计数**：`len(data) <= top_k`，含邻域扩展加进来的槽位 | §2.2 / §10 |
| 2 | **原样回显** `request_id` / `user_id` / `session_id` | §2.1 |
| 3 | **响应前必须持久化完成且立即可搜索**——不允许异步建索引 | §2.1 |
| 4 | **位置 = `(request_id, local_index)`**（D28）——`request_id` 是 **opaque string**：**不许解析它**，也不许回退到"按到达顺序分配"。同 session 并发与乱序到达都安全 | §15 → **D28** |
| 5 | **非 200 的行为未定义，必须假设 AML 会重试**——内部异常应让本批保持"可重试"（事务未提交），**而不是返回一个"部分成功"** | §15 |
| 6 | **`user_id` 是唯一隔离字段**；`session_id` 不是 Search 的过滤条件 | §2.2 |

---

## 共同取证分支（`retrieval.grounded_evidence`）

Search 在开关打开时**先试共同取证**，不适用就**整体回退**到原混合检索链，
**不半途混用**。本层拥有的是"先试哪条、何时退回"这个顺序决策；**回退判据的唯一
声明处**在 [`../retrieve/CLAUDE.md`](../retrieve/CLAUDE.md) 的「不齐全 ⇒ `None`」节
（键语义在 [`../../../docs/config-reference.md`](../../../docs/config-reference.md)）——
注意两者的走法不同：**计划编译不出来、抓取行数超限**时本层**根本不进取证路径**；
而**索引未扫完**（覆盖不全）时本层以**空证据**进入，由 `select_evidence` 返回 `None` 回退——
不是"执行器返回了 `None`"这一句话能概括的。

命中时**只做取证与打包**：原文或有来源的独立事实片段复用既有的段结构、
槽位与 token 双预算（此路径半径恒为 0），**不调用 rerank**。

Add 侧：`apply_batch` 在**同一个写事务**里逐行索引事实（开关打开时）；
**幂等重放**（守卫命中）也经同一入口补写事实索引——真源不变。

四个配置键由 `build_services()` 从 `AppConfig` 注入两条 pipeline，**本层不读环境变量**：
`retrieval.grounded_evidence`（两条都收）、`retrieval.evidence_limit` /
`retrieval.evidence_hop_limit` / `retrieval.fact_backfill_limit`（只进 Search）。

---

## 两个容易写错的地方

### 必须 `--workers 1`

**理由**：见根 `CLAUDE.md` 的《硬约束》表（`Add` 仍必须 `--workers 1`，出处 §15）——**本文件不重复**。
⇒ 在那之前保持这条约束，**不要用"位置已经是请求的纯函数了"当理由把它去掉**。

**这一条有两条防线，都不是文档警告**（③-d）：

| 防线 | 在哪 | 拦住什么 |
| --- | --- | --- |
| **配置阶段** | `common/config.py` 的 `validate()` | `TIANXIMEM_WORKERS != 1` ⇒ `ConfigError` |
| **进程阶段** | `common/config.py` 的 `assert_single_process()`，由 `create_app_from_env()` 调用 | **命令行**给的 `--workers 4`——配置层看不见它 |

进程阶段的判据是 `multiprocessing.parent_process() is not None`：uvicorn 的 `--workers N`（N>1）会用
`multiprocessing` 派生子进程来跑 app，而单进程启动时它是 `None`。
⇒ **`--reload` 也会被一并拦下**（两者在子进程里形状相同，父进程只留下 pid 与名字）；开发期请用 `make serve`。

> **`BEGIN IMMEDIATE` 是并发唯一的排队点**（`store/` 层）：所有写请求都在 SQLite 的
> 写锁上串行，抢不到的一方**等待**（`busy_timeout`）而不是抛 `SQLITE_BUSY`。
> 它是**数据库级**的，与"哪个进程"无关。

### FastAPI 是 async，但这一层的下游是阻塞的

**这一层的下游全部是阻塞调用**：SQLite 是阻塞 IO，而 embedding 与 rerank 是
**同步 HTTP**（`httpx.Client`）——**同步客户端在事件循环里会堵住整个循环**。
用 FastAPI 不代表要写 `async def` 业务函数：把阻塞部分显式下到线程池
（`run_in_threadpool`），**或者干脆把路由写成 `def`**（FastAPI 会自动下线程池）。

**不要**在 `async def` 里直接调阻塞的存储或模型调用——那会让并发请求互相饿死，
而 30 分钟的单请求预算会掩盖这个问题，**直到 Full run 跑 0.5–2 天时才炸**（§15）。

> ⚠ **模型不在本机**（D12/D14）：三者都经自建网关远程访问，所以"阻塞"的形状是
> **等待网络**，不是等 GPU。这不会让它变轻——一个 30 秒的 rerank 超时
> （`rerank.timeout_seconds`）在事件循环里就是 30 秒的停摆。

---

## 契约自查

跑 Smoke 之前逐条过 [`../../../docs/contract.md`](../../../docs/contract.md) §4 的清单。**Smoke 次数有限（每轨道 ≤30 次），不要拿它当调试器。**

`make contract-check` 指 [`../../../eval/smoke/preflight.py`](../../../eval/smoke/preflight.py)——**尤其是"精确计数 ≤ `top_k`"与"`created_at` 始终存在"这两条检查**，它们最容易在加了邻域扩展之后悄悄破掉。
