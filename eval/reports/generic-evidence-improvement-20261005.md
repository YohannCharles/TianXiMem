# 共用关系与物品取证改进（2026-10-05）

本轮将 Add/Search 中的一部分取证能力抽象成共用事实表示与查询路径，已保留实现并默认
开启。产品提交为 **`f9548a3`**，基线为 `e63fecd`。先手工验证建议片段，再实现、通过
HTTP Add/Search 验证；接口、答案模型、提示词和裁判保持不变，邻接半径仍为 0。

固定的六领域诊断题，旧版 **19/24**、改进版 **24/24**；代码冻结后新增两个场景，
旧版 **6/8**、改进版 **8/8**。这些是合成的小样本迁移诊断，**不是完整数据集精度、
模型精度上限或 AML 榜分**，也不能说明任意表达都已支持。

## 具体改了什么

以前一些人物信息按公司、志愿活动等分别抽取，有的场景已经抽到事实，但查询没有对应
路径；背包等场景直接返回混杂原文，答案模型还需自行辨别所有者、计划、重复及物品数量。

现在共用两个原子结构：

| 结构 | 字段 | 来源例子 |
| --- | --- | --- |
| 关系事实 | 主体、关系、对象、原文引文、父记忆、来源日期、版本 | 某人是某机构的员工、学生、患者或成员 |
| 物品事实 | 上述字段，另有容器和原文单项数量 | 某人的某箱子有若干杯子 |

角色与容器名称是来源字段，不为公司、学校、诊所分别建表。抽取器处理有限的共同句法，
如 `Name: I am a <role> at <organization>.`，以及
`Name: My <container> contains <quantity> <item>, ...`。
因此新增 `patient` 或新箱子名称不需要新增领域分支。少量 `work→employee`、
`study→student`、`belong→member` 等关系同义映射集中维护；未知同义关系不自行推断。

Search 把支持的“有哪些/多少”问法归到同一关系与范围，返回同一组独立证据。它仍不
执行最终名单拼接或 SUM/COUNT。持久化与渲染在 Add/旧记忆补索引时完成，Search 筛选
已有事实；不是收到问题后把答案重新包装为记忆。已有专用取证路径保留优先级。

## Search 返回的仍是记忆事实

原始记录：

> Rosa Quinn: My 10-liter gray crate currently contains four cups, two plates, and one kettle.

真实 HTTP Search 返回三个独立片段，其中之一为：

```text
Memory fact
Subject: Rosa Quinn
Relation: contains
Object: cups
Container: 10-liter gray crate
Quantity: 4
Source record date: 2025-09-01
Source quotation: My 10-liter gray crate currently contains four cups, two plates, and one kettle.
```

另外两条是 plates / 2、kettle / 1。Quantity 仅转写原文单项数量，容量 `10-liter`
仍属于容器名称；片段中没有新增总数 `7` 或针对问题生成的最终物品数组。名单和求和由
原来的 Qwen 答案链完成。日期仅表示原始记录日期，没有推断物品购入或患者登记日期。

患者例子的实际片段使用 `Subject: Rosa Quinn`、`Relation: patient at`、
`Object: Aspen Clinic` 和对应原句，不将患者转成公司员工。

## 固定协议与实际结果

1. 从前一轮四领域验证复用原始数据、题目和金标，增加俱乐部与工具箱，固定 24 题。
   每领域 7 条原文，包含重复、未来计划、非成员/错误所有者或容器、虚构引用。
2. 人工提出每领域 3 个有来源的原子事实，复用原有答案流水线预检，**24/24 通过**后
   才修改产品代码。手工事实没有进入 Add。
3. 两个独立服务只接收相同原文，分别经 HTTP Add → HTTP Search → 原有答案链评分。
   旧版服务在修改产品前启动，固定为 `e63fecd` 的进程；两臂独立 SQLite/向量集合。
4. 修正未解释相关原文的保守回退检查后，重查 24 题，最终上下文与已评分上下文逐字一致，
   没有为了改进数字重复采样答案。
5. 冻结产品文件指纹后，新增“诊所患者”和“带容量箱子”两场景，共 8 题。先手工预检
   **8/8**，随后仅原文 HTTP Add，旧版与改进版各回答一次。产品指纹在新增检查结束后
   保持一致，没有针对这两个领域增加产品规则。

模型为 `Qwen/Qwen3.5-9B`，temperature=0，max_tokens=1024，top_k=100；列表按
原有 set-F1、整数按 exact-match 判分。本轮两臂所有部分分均为 0 或 1，均分因此恰好
等于完全正确率；一般情况下两者不能混用。答案/裁判调用串行，没有运行完整基准。

| 场景 | 手工事实预检 | 旧版真实 Add/Search | 改进版真实 Add/Search |
| --- | --- | --- | --- |
| 公司员工 | 4/4 | 4/4 | 4/4 |
| 学校学生 | 4/4 | 4/4 | 4/4 |
| 收容所志愿者 | 4/4 | 3/4 | 4/4 |
| 背包物品 | 4/4 | 3/4 | 4/4 |
| 俱乐部成员 | 4/4 | 4/4 | 4/4 |
| 工具箱物品 | 4/4 | 1/4 | 4/4 |
| 首批合计 | 24/24 | 19/24（79.2%） | 24/24（100%） |
| 新增：诊所患者 | 4/4 | 3/4 | 4/4 |
| 新增：带容量箱子 | 4/4 | 3/4 | 4/4 |
| 新增检查合计 | 8/8 | 6/8（75%） | 8/8（100%） |

每场景均有列表、计数原问法及改述，四题共享来源，不能当作四个完全独立的语料样本。
前一轮 16 题的旧成绩与本轮重新跑出的分组数字存在差异；即使 temperature=0，也不能
假定不同运行完全一致。本轮收益按本轮固定的两臂对照计算，不拼接前一轮成绩。

32 题每题实际返回 3 条独立事实。其中公司两道原问法继续使用已有任职事实路径，
其余 **30 题使用新增共用事实路径**。这次学校、患者等的成功已核对到实际抽取/取证，
不会把原文回退路径的答对误报为通用抽取成功。

| 上下文量 | 旧版 | 改进版 |
| --- | --- | --- |
| 首批 24 题平均 token | 207.8 | 140.7 |
| 新增 8 题平均 token | 205.5 | 155.0 |

这里没有扩大返回窗口或塞入更多邻接片段。原文回退在这些小语料里通常返回 7 条；
新路径每题选择 3 条事实，减少答案模型需要同时处理的干扰。

## 几个真实失败前后例子

| 问题/场景 | 旧版输出 | 改进版输出 | 相关变化 |
| --- | --- | --- | --- |
| 志愿者人数改述 | `4` | `3` | 使用同一关系下去重的人物事实 |
| 背包物品总数改述 | `5` | `6` | 明确区分物品种类与单项数量 |
| 工具箱物品种类 | `["four bolts", "two nuts", "three washers"]` | `["bolts", "nuts", "washers"]` | Item 与 Quantity 分开表示 |
| 工具箱物品总数 | `11` | `9` | 绑定 Noah 的绿色工具箱，排除他人及另一容器 |
| 患者人数 | `2` | `3` | 同一通用关系结构处理 patient |
| 带容量箱子物品种类改述 | `["four cups", "two plates", "one kettle"]` | `["cups", "plates", "kettle"]` | 容量、物品名称和件数分别保留 |

这些是取证与表示变化之后的答案链结果。没有逐一删除每个干扰项，不能断言某个干扰
就是全部错误的唯一原因，也没有用更大的模型改判。

## 边界、回退与数据核对

- 新 SQLite 索引保存父记忆、user_id、原文逐字引文与版本；写入时校验原文来源和用户。
  索引与覆盖标记在同一 Add 事务内保存，重放可补索引，不改原始记忆和批次指纹。
- 名单按同一主体/关系/对象去重。物品按所有者/容器/种类/原文数量绑定，同一种物品出现
  不同数量时保留原文处理，不自行选择一个版本、累加多次快照或猜测当前总数。
- 否定、计划和虚构引文不抽成正向事实。同范围仍有未解释的正向陈述或离开状态，
  事实分支回退；同段存在计划/引用和真实陈述的混合案例也有单元验证。
- 旧记忆补索引有 `fact_backfill_limit`，来源与事实读取有上限。未扫完或超限时回退，
  不以已知子集声称全部成员/物品已经取齐。仍受 top_k 和原来的 token 预算限制。
- 两臂 **56 条原文、56 个向量、8 个批次**核对完成；各自的原文/日期/批次指纹一致。
  改进版共存储 **41 条通用事实**，含重复来源及其他所有者的真实事实，不按金标从存储中
  删除它们。所有引文均逐字来自同用户父原文；查询再按范围筛选和去重。
- 32 个实际 HTTP 返回上下文逐条匹配持久化事实，来源用户、日期及名次分数核对通过；
  每对列表/计数返回相同正文，同题重查稳定，top_k=1 有效，未知用户返回空数组。

这套表示还不是任意语义的通用抽取器。当前支持明确具名的第一人称英语句式、有限问法
和字面机构/容器范围；未知同义词、隐含所有者、纯代词指代、跨时状态更新及多语言仍需
验证。范围审查依赖原文字面名称，不能保证捕获没有复述名称的隐含更新。
已有的否定/离开与数量冲突回退只验证了显式范围案例；**覆盖扫描完成不等于语义完整**。

## 旧案例回归与失败检查记录

从六小时改进轮选择 **24 个旧诊断查询**，涉及 CorporateBench、MQuAKE、TempReason、
LongMemEval、LoCoMo 和 MemTrapBench。使用同一旧语料、同一向量集合，分别开启/关闭
新功能：**24/24 上下文逐字一致**，且均与先前保留版本的历史上下文一致。
其中 8 个此前已修复的成功案例实际重新回答，**8/8 正确**。上下文相同与重新答对分别
记录；没有把另外 16 个未重答案例宣称为正确。

初次回归配置误选 `memories_dev`，实际历史集合是 `memories_goal_20261005`，导致
6 个上下文与历史不同，其中 5 个 LongMemEval 回退查询为空。同一错误配置下开关对照
已是 24/24 一致；改为正确历史集合后，开关对照和历史对照均 24/24 一致。
因此这次差异属于实验配置错误，未作为产品退化结论。原始失败检查、原因和修正后结果
均保留，没有覆盖初次失败产物，也没有通过改答案或金标消除差异。

本轮未出现需要 Git 回退的产品候选；已有六小时实验的失败/回退记录继续保留。
两个回归库的 **6,237 条原文和 757 个批次**与改动前快照一致。回归服务只查询旧集合，
没有对它执行 Add 或改写向量。

验证通过：全量单元测试 **1,042 passed**；Ruff 检查及格式检查通过；mypy 的 52 个产品
模块通过；本地真实 HTTP 契约预检 **14/14** 通过。单元测试增加了任意角色/容器、
说话人绑定、用户隔离、部分库存、冲突/离开回退、重放/补索引、来源校验及配置检查。
完整单元测试和本地契约预检不属于完整数据集实验，本轮没有调用 AML Smoke/Full。

## 本轮覆盖的七个维度

| 维度 | 实际覆盖与限制 |
| --- | --- |
| Explicit fact recall | 人物关系与单项物品事实；仅有限句式及小样本 |
| Relational and multi-hop reasoning | 主体/关系/对象及所有者/容器绑定；未新增多跳语义推理 |
| Temporal and event understanding | 计划排除、显式离开/数量冲突回退；未实现一般状态演变 |
| Memory governance | 原文与批次不变，来源/版本留存、幂等重放、补索引及重复事实去重 |
| Personalization and care | 所有者范围绑定；未测照护与偏好任务，不报子分 |
| Rules and process execution | 保持列表/整数规则和 HTTP 契约；未测一般流程执行 |
| Epistemic safety and privacy | 同用户逐字溯源、虚构排除、未知用户隔离与未解释来源回退；仅已测范围 |

## 实现、复现与归档

- 产品核心：[evidence.py](../../src/tianximem/facts/evidence.py)、
  [Search 路由](../../src/tianximem/service/pipeline.py)、
  [SQLite 索引](../../src/tianximem/store/sqlite_store.py)。
  Add 与旧来源补索引复用 [index_pair_facts](../../src/tianximem/pairing/apply.py)。
- 配置项 `retrieval.grounded_evidence: true`，可按 YAML 关闭，见
  [配置说明](../../docs/config-reference.md)。没有新 LLM、答案调用或环境变量。
- 运行脚本在 [runs/generic-evidence-20261005/](runs/generic-evidence-20261005/)，
  配置快照在 `configs/runs/generic-evidence-20261005/{baseline,candidate,regression,regression-off}/`。
  基线/改进服务端口为 8096/8097，旧语料开关对照为 8098/8099，均使用隔离库。
- 首批题目 SHA-256 为 `9fb4487c70d487abb07701bf57522a90d7b42e84cf9815f2b3c3a73c49093a69`；
  新增题目为 `13d428191bc7a097fdf33129d51f2aea7b454df9363488f9b4d85abee561648a`。
  输入、模型、提示词与判分指纹在 `frozen.json` / `heldout-frozen.json`；新增检查产品
  指纹在 `heldout-code.json`，最终归档时再次核对，保持一致。
- 主要产物：`manual-results.json`、`baseline/summary.json`、`candidate/summary.json`、
  `heldout-*/summary.json`、`returned-evidence-audit.json`、`final-source-audit.json`、
  `regression-results.json`、`regression-controlled-parity.json`。
  初次配置错误检查保存在 `regression-contexts.json`、
  `regression-initial-wrong-collection-parity.json` 和 `regression-initial-config-error.json`。
- 四臂一致性 SQLite 快照保存在 `var/generic-evidence-20261005/<arm>/tianxi-snapshot.db`，
  SQLite/配置/产品指纹在 `archive.json`。原始产物与数据库按仓库惯例 gitignored，
  报告、脚本及配置快照提交 Git，实验记录和向量集合均保留。

复现顺序为 `prepare.py` 手工验证 → 原文 `ingest.py` → 两臂 `tools/targeted_eval.py` →
`heldout.py prepare/manual/ingest` → 新增两臂评测 → `audit.py` → `archive.py`。
含模型/冻结检查的脚本使用 `uv run --env-file .env python`；模型调用串行。
重跑前应确认目标库与向量集合一致，不复用已有答案缓存来代表新的一次模型评分。

```bash
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/prepare.py
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/ingest.py --arm candidate --base-url http://127.0.0.1:8097
uv run --env-file .env python tools/targeted_eval.py --manifest eval/reports/runs/generic-evidence-20261005/manifest.json --base-url http://127.0.0.1:8097 --output eval/reports/runs/generic-evidence-20261005/candidate --freeze eval/reports/runs/generic-evidence-20261005/frozen.json --deadline <未来UTC秒数>
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/heldout.py manual
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/audit.py regression
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/audit.py returns
uv run --env-file .env python eval/reports/runs/generic-evidence-20261005/archive.py
```

验证结束后关闭本轮四个隔离服务。用户原有服务未重启；默认配置将在下次启动时加载。
