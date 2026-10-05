# docs/ — 文档索引

**权威规格是仓库根目录的 [`AML Agentic Memory 增强框架 PRD.md`](../AML%20Agentic%20Memory%20增强框架%20PRD.md)。**
本目录的文档**不是**规格的第二份副本——它们只做三件 PRD 没做的事：

1. **导航**：把 PRD 的 § 映射到代码目录（`architecture.md`）
2. **清单化**：把散落在各节的"必须做/必须不能做"压成可勾选、可自查的形式（`contract.md`、`config-reference.md`、`open-questions.md`、`roadmap.md`）
3. **记录**：写下 PRD 没有的**状态**与**决策变更**（`experiments.md`、`decisions.md`）

> **冲突时一律以 PRD 为准。** 本目录若与 PRD 不一致，那是一个 bug，请改本目录。

## 四层文档，各管一段

| 层 | 位置 | 回答什么 | 何时加载 |
| --- | --- | --- | --- |
| **规格** | 根 PRD | 唯一的权威定义 | 按需（指针指向它） |
| **约束速查** | 根 `CLAUDE.md` | 任何动作都要守的红线 + 文档路由 | **每次会话自动** |
| **模块约束** | 各目录 `CLAUDE.md` | 该目录要写什么、边界、本层的坑 | 读该目录文件时按需 |
| **为什么 / 状态** | 本目录 + 各目录 `CLAUDE.md` | 依据、决策变更、进度、未知 | 按需（由上述两层指过来） |

**一条硬性质**：**本目录不重复模块级约束**。哪个模块要遵守什么写在它的 `CLAUDE.md` 里；本目录只写**跨模块的协议与状态**。同一条约束有两个家，就是漂移的开始——见下。

## 文档清单

| 文档 | 回答什么问题 | 什么时候看 |
| --- | --- | --- |
| [architecture.md](./architecture.md) | 数据怎么流、模块边界在哪、谁依赖谁、**七个评分维度各由谁回应** | 动手写任何模块之前 |
| [contract.md](./contract.md) | Add/Search 的精确形状是什么、怎么自查合规 | 写 service 层、发 Smoke 之前 |
| [config-reference.md](./config-reference.md) | 有哪些配置项、默认值多少、**哪些是常量哪些是阈值**、开关之间的依赖是什么（**开关的唯一声明处**） | 加任何阈值/权重/开关时 |
| [experiments.md](./experiments.md) | 要做哪些对照、每个对照决定什么、现在什么状态 | 每跑完一组对照 |
| [open-questions.md](./open-questions.md) | 还有哪些没清掉的未知、谁负责消除、什么时候会阻塞 | 每个 Step 开始前 |
| [roadmap.md](./roadmap.md) | Step 0–6 各自要交付什么、当前进度 | 每个 Step 开始/结束时 |
| [decisions.md](./decisions.md) | 哪些决策被推翻过、为什么、影响哪些节（**逐条 `Dn` + 待决事项**） | 想改一个"已锁定"的决定之前 |
| [submission.md](./submission.md) | Smoke/Full 还剩几次、版本冻结在哪、S 未知清了没有 | 每次发 Smoke / Full 之前 |
| [benchmark-data.md](./benchmark-data.md) | 归档里哪些文件是真数据、schema 落差在哪、许可证什么情况 | 写 harness / 加载器时 |
| [reference-implementations.md](./reference-implementations.md) | 榜单前两名（InvMem / ReFind）的代码里有什么可学、什么别学、代码在哪（含 commit） | 想"看别人怎么解同一道题"时 |
| [add-search-improvement-plan.md](./add-search-improvement-plan.md) | 如何在 Add/Search 内增加可查证的事实、条件检索及对象／事件证据选择（**已实施，2026-10-05**；正文为方案底稿，最终形态见共同取证整理） | 评估文档类列表与计数的改进时 |

> **[`benchmark-data.md`](./benchmark-data.md) 尤其要看**：归档目录 `benchmark_data/` **整目录被 `.gitignore` 排除**，所以**不能在那里放任何文档**（不会被提交）——关于归档的一切说明都在那份文档里。

## 维护约定

- **引用 PRD 一律带 § 号**（如 `§6.5`），不带 § 号的断言视为未经验证。
- **单数来源（最重要的一条）**：一条约束只允许有一个"家"。发现自己在写第二遍时，**改成指针**。**改动一条约束时，先搜一遍它还出现在哪几个文件里。**
- **每份文档顶部写"最后核对日期 + 对应 PRD 版本"**——PRD 仍在演进。
  > 对 `CLAUDE.md` 这类每次会话都加载的文件，把它写成 **HTML 注释**（`<!-- ... -->`）：注释在注入模型上下文前会被剥离，**不花 token，人还能看到**。
- **记录待验证的判断时，必须显式标注"待验证"**，不要写成已知（这是 PRD 本身的书写约定，见其文档定位段）。
- **编号纪律**：见根 [`CLAUDE.md`](../CLAUDE.md) 的那一条（**不要挪作他用**）。
- **数字只住在 [`../eval/reports/`](../eval/reports/)**——其余文档引用数字时**指回去**，不要复制。
