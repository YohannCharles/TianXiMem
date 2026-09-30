# common/ — 渲染、token 计数、配置

**PRD**：§11.3（渲染）、§6.4（token 预算）、§15（配置开关）

## 本模块的构成

```text
render.py    ✅ 已实现——渲染模板的**唯一实现**
annotate.py  ✅ 已实现——正文里的相对时间**就地注解**（只改 `content`、不碰索引）
config.py    ✅ 已实现——**全包唯一**读环境变量的地方（③-d）
tokens.py    ✅ 已实现——o200k_base 计数（§6.4）
```

**这不是一个工具箱。** 它存在的理由是**一条不变式**，外加两个"必须只有一份实现"的量。

---

## 1. `render` —— 唯一实现（不变式 I1）

> **§7.2 与 §11.3 要求：embedding 的输入 与 返回给 AML 的 `content`，必须是同一份渲染。**

两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会**漂移**——**而且这种漂移不会报错**：检索照常返回值，分数照常算，只是**命中的是另一个版本**。这类 bug 的排查成本极高（要同时对两处做 diff 才能发现）。

**因此**：

- `render.py` 是**唯一**拼 QA 对文本的地方
- [`../embed/`](../embed/) **不自己拼字符串**
- [`../rank/`](../rank/) 的 packaging **也不**自己拼
- 两处都调 `render.py`，传同一个 QA 对

**渲染规则（模板、role 标记、自定界）的唯一声明处是 [`../rank/CLAUDE.md`](../rank/CLAUDE.md) §4**——本文件不重复，避免两处规则描述漂移（**同样的理由**）。

**⚠ 「贵」消融项**：改渲染模板 = 改变 embedding 输入 = **整个向量索引要重建**（[`../../../tools/reindex.py`](../../../tools/reindex.py)）。模板现为 `v1` + **日粒度日期前缀**（**D21**，2026-09-25）——**推翻它要付重建索引的钱**。

**三个入口都在本模块**（一个对 → 它的文本只该有一条路）：

| 名字 | 用途 |
| --- | --- |
| `render(q, a, *, date="")` | 逐对的唯一拼装（`date` 非空 ⇒ 前缀 `[YYYY-MM-DD] `） |
| `render_pair(pair, *, inject_abs_time)` | **三个调用点的入口**（索引 / 精排输入 / `content`）——日期口径只此一处 |
| `render_date(...)` · `day_granularity(...)` · `event_day(...)` | T1 的开关落点、日期格式（`created_at` 与 `content` **共用同一个格式**）与**时间换算的唯一一处** |

⇒ **T1 的"带日期"臂**（`packaging.inject_abs_time`）不是三处各改一遍，而是这一条路上的一个开关。
⛔ 反过来：**不要在别处再拼一次日期**——那会同时踩中"两处渲染漂移不报错"与"两臂同时在两个维度上不同"。

> ### ⚠ `annotate.py` 是 I1 的**唯一例外**，边界写得死
>
> `packaging.annotate_relatives=true` ⇒ 正文里的相对时间就地注成绝对日期
> （`last Tues (July 18, 2023)`），原文一字不动。**它只接在 `content` 这一条路上**
> （[`../rank/neighbor.py`](../rank/neighbor.py) 的 `_text()`），索引侧与精排输入
> 仍然只认 `render_pair` ⇒ **开它不用重建索引**（与 T1 正相反）。
>
> **例外到哪为止**：`content` = 被索引的文本 **+ 一层纯函数注解**，剥掉后**逐字相同**。
> 别把它读成"以后可以往 `content` 里加任何东西"——每加一层都要重新回答
> "它进不进索引、缓存坐标要不要动"（[`../../../docs/architecture.md`](../../../docs/architecture.md) §4）。
>
> ⚠ **[`eval/harness/annotate.py`](../../../eval/harness/annotate.py) 是同一套逻辑的第二份**
> （harness 不许 import `src/`，而那次实验是在 harness 侧跑的）——**两份都对**，
> 由 [`../../../tests/test_annotate.py`](../../../tests/test_annotate.py) 的逐例等价断言钉住。

---

## 2. `tokens` —— 必须用答案模型自己的分词器（§6.4）

| 要求 | 说明 |
| --- | --- |
| **`o200k_base`** | `gpt-4o-mini` 的分词器 |
| **不要用字符数或空格切分近似** | **单次近似偏差会在 100 个对上被放大到几千 token** |
| 上限 **117,760** | 答案窗口 128k 扣掉输出与安全余量 |

> ### ⚠ R1 类风险（§6.4）
>
> 本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` **不同**，**本地量出的"能装多少对"不能直接搬到线上**。§12.1 的四条对冲里**没有这一条**，PRD §6.4 明确"应补上"。
>
> **Step 5 切换后必须重新量一次单请求实际返回的对数**，且 token 预算做成配置项。见 [`../../../docs/open-questions.md`](../../../docs/open-questions.md) **E7**。

**双预算**：本模块提供 token 计数，但**"槽位数 + token 数双截断"的决策在 [`../rank/`](../rank/)**——本模块只回答"这段文本是多少 token"，不决定截谁。

---

## 3. `config` —— 加载与校验（§15）

配置加载不只是 `yaml.safe_load`。它要承担两件事：

**一条不变的配置纪律**：**不得硬编码**（§12.1 R1 对冲 3）。凡是 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) 里列出的量，代码里只应有读配置的语句。**开关之间的依赖与"关掉时不得改变什么"也在那份文档**——**它是一处声明**，本文件不另列一份。

### 三条结构性质（③-d）

`config.py` 现在是**全包唯一**读 `os.environ` 的地方。三条结构性质，各自有测试钉住：

| 性质 | 由谁保证 |
| --- | --- |
| **只有 `config.py` 访问 `os.environ` / `os.getenv`** | `tests/test_config.py::test_only_config_reads_the_environment`——**AST 扫描** `src/tianximem/**/*.py`，不是靠自觉 |
| **只有 `config.py` 解析 yaml** | 同上的 `test_no_module_reads_config_files_directly` |
| **每个键只有一个家**（env 或 yaml，不重叠） | `test_thresholds_are_not_overridable_from_env` + `test_env_owned_keys_are_rejected_in_yaml` |

**两层分工**（一处声明在 [`../../../configs/CLAUDE.md`](../../../configs/CLAUDE.md)）：

| 层 | 拥有 | 例子 |
| --- | --- | --- |
| **环境变量**（`.env`） | 密钥、端点、**路径**、worker 数 | `AML_EMB_BASE_URL` / `TIANXIMEM_SQLITE_PATH` / `TIANXIMEM_WORKERS` |
| **`configs/<profile>.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.rrf.k` / `models.embedder` / `storage.qdrant.collection` |

**`.env` 也由本模块读取**——它排在真实环境变量**之下**（`export` 过的值压过它），
且**只在 `env=None`（生产路径）时读**：测试传 `env` dict 就完全不碰磁盘。
换位置用 `TIANXIMEM_ENV_FILE`。

> ⚠ **不认识的键一律报错**（env 的那几项写进 yaml 也算）：拼错的键被静默忽略 ⇒ 跑的是默认值，
> 而**没有任何人会发现**——这正是本项目反复要避免的那一类失败。

**⚠ 配置化 ≠ 可调**。`rrf.k = 61` 与 `top_k = 100` 都是**正确性常量**，写进配置是为了追溯与切换，**不是为了调**——分组见 config-reference 的 A / B / C 三分类。**不合法的值在启动阶段就拒绝**（`rrf_k != 61`、`workers != 1` 都是硬错误，不是警告）。

**一条待补的开关**：**`checker.enabled` 与它的三个判据阈值都必须是配置项**（§15 / D15）——§8 要求阈值在代理评测上标定，§13 的 A4 要关它做对照。落点见 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) §4。

> **只收"今天有代码消费方"的键**——**收一个没有消费方的键等于预留字段**（§6.1 对 DDL 的同一条纪律）；**哪些开关还没接线，逐项以 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) §2 的表为准**。
> **但已接线的键仍不可随手调**（`neighbor.*` / `budget.*`，消费方 `rank/neighbor.py` + `service/pipeline.py`）——哪些能调见 `configs/default.yaml` 的 A / B / C 类注释。

---

## 依赖方向

**本目录不依赖任何业务层**——它被所有层依赖。若你发现 `common/` 里 import 了 `store/` 或 `retrieve/`，那是分层错了。
