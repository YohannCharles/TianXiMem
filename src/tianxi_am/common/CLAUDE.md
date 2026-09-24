# common/ — 渲染、token 计数、配置

**PRD**：§11.3（渲染）、§6.4（token 预算）、§15（配置开关）

## 本模块的构成

```text
render.py    ✅ 已实现——渲染模板的**唯一实现**
config.py    ✅ 已实现——**全包唯一**读环境变量的地方（③-d）
tokens.py    ⬜ 待写——o200k_base 计数（§6.4）
```

**这不是一个工具箱。** 它存在的理由是**一条不变式**，外加两个"必须只有一份实现"的量。

---

## 1. `render` —— 唯一实现（不变式 I1）

> **§7.2 与 §11.3 要求：embedding 的输入 与 返回给 AML 的 `content`，必须是同一份渲染。**

两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会**漂移**——**而且这种漂移不会报错**。检索照常返回值，分数照常算，只是**命中的是另一个版本**。这类 bug 的排查成本极高（要同时对两处做 diff 才能发现）。

**因此**：

- `render.py` 是**唯一**拼 QA 对文本的地方
- [`../embed/`](../embed/) **不自己拼字符串**
- [`../rank/`](../rank/) 的 packaging **也不**自己拼
- 两处都调 `render.py`，传同一个 QA 对

**渲染规则（模板、role 标记、自定界）的唯一声明处是 [`../rank/CLAUDE.md`](../rank/CLAUDE.md) §4**——本文件不重复，避免两处规则描述漂移（**同样的理由**）。

**⚠ "贵"消融项**：改渲染模板 = 改变 embedding 输入 = **整个向量索引要重建**。**Step 3 定稿，别拖到 Step 5 之后。**

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

### ✅ 已实现（2026-09-24，③-d）

`config.py` 现在是**全包唯一**读 `os.environ` 的地方。三条结构性质，各自有测试钉住：

| 性质 | 由谁保证 |
| --- | --- |
| **只有 `config.py` 访问 `os.environ` / `os.getenv`** | `tests/test_config.py::test_only_config_reads_the_environment`——**AST 扫描** `src/tianxi_am/**/*.py`，不是靠自觉 |
| **只有 `config.py` 解析 yaml** | 同上的 `test_no_module_reads_config_files_directly` |
| **每个键只有一个家**（env 或 yaml，不重叠） | `test_thresholds_are_not_overridable_from_env` + `test_env_owned_keys_are_rejected_in_yaml` |

**两层分工**（一处声明在 [`../../../configs/CLAUDE.md`](../../../configs/CLAUDE.md)）：

| 层 | 拥有 | 例子 |
| --- | --- | --- |
| **环境变量**（`.env`） | 密钥、端点、**路径**、worker 数 | `AML_EMB_BASE_URL` / `TIANXI_SQLITE_PATH` / `TIANXI_WORKERS` |
| **`configs/<profile>.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.rrf.k` / `models.embedder` / `storage.qdrant.collection` |

**`.env` 也由本模块读取**——它排在真实环境变量**之下**（`export` 过的值压过它），
且**只在 `env=None`（生产路径）时读**：测试传 `env` dict 就完全不碰磁盘。
换位置用 `TIANXI_ENV_FILE`。

> ⚠ **不认识的键一律报错**（env 的那几项写进 yaml 也算）：拼错的键被静默忽略 ⇒ 跑的是默认值，
> 而**没有任何人会发现**——这正是本项目反复要避免的那一类失败。

**⚠ 配置化 ≠ 可调**。`rrf.k = 61` 与 `top_k = 100` 都是**正确性常量**，写进配置是为了追溯与切换，**不是为了调**——分组见 config-reference 的 A / B / C 三分类。**不合法的值在启动阶段就拒绝**（`rrf_k != 61`、`workers != 1` 都是硬错误，不是警告）。

**一条待补的开关**：**Checker 的开关必须是配置项**（§15 的开关清单里原本漏了它），因为 §13 的 A4 要关它做对照。**§8 要求它的三个判据阈值在代理评测上标定**，所以阈值也是配置项。

> **③-d 只收了"今天有代码消费方"的键**。`checker.*` / `neighbor.*` / `rerank.*` / `agent.*` /
> `budget.*` 都还没接——那些模块有的未实现、有的还没接线。**收一个没有消费方的键等于预留字段**
> （§6.1 对 DDL 的同一条纪律）。它们的落点已经写在 [`../../../docs/config-reference.md`](../../../docs/config-reference.md)。

---

## 依赖方向

**本目录不依赖任何业务层**——它被所有层依赖。若你发现 `common/` 里 import 了 `store/` 或 `retrieve/`，那是分层错了。
