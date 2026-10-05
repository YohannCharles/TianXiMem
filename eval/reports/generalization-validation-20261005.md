# 跨领域、换问法的取证验证（2026-10-05）

本轮是验证记录，没有修改产品代码或开始新的六小时 goal。结论是：**手工提供清楚的、
有来源的独立事实时，16 道题全部答对；现有 Add/Search 实际答对 14/16。现有事实
抽取与查询路由仍然依赖领域及表达，不能据此称为通用事实索引。**

## Search 与最终回答的边界

遵循本仓 [契约 §1](../../docs/contract.md)：Search 不生成最终答案，也不把答案伪装
成记忆记录。用户允许 Add/Search 内部存储、检索及打包模块调整，接口与答案流水线保持不变。

语义判断可用于确认说话人、区分现实陈述/计划/否定/虚构引用、筛选相关来源，以及同一事实
的去重。这些步骤的输出仍应是独立记忆或有原文支持的事实。去重不等于把查询结果写成答案。

例如，源记录为：

> Maya Chen: My blue backpack currently has two notebooks, three pens, and one ruler.

允许返回该原文，或返回分别可追溯到它的片段：

```text
Owner: Maya Chen
Container: blue backpack
Item: notebooks
Quantity: 2
Source record date: 2025-09-01
Source quotation: Maya Chen: My blue backpack currently has two notebooks, three pens, and one ruler.
```

另外两条事实分别为 pens / 3、ruler / 1。Quantity 是原文已有的单项数量。
本例中，Search 不应新增“总共有 6 件”，也不应新增针对查询的最终数组
`["notebooks", "pens", "ruler"]`。总数及最终回答格式由原有答案模型生成。
若原文自身记载名单或总数，它仍可作为原记忆返回；禁止的是在 Search 中新生成最终答案。

## 固定协议与来源

- 四个合成领域：公司员工、学校学生、收容所志愿者、背包物品。每个领域 7 条原始记录，
  每个领域一个独立 user_id。前三组的实际成员都是 Maya Chen、Noah Reed、Inez Patel。
- 每个领域设“有哪些/多少”两种任务，各有原问法和改述，共 16 道题。背包列表金标为
  notebooks / pens / ruler，计数金标为 6。
- 原始记录包含重复陈述、尚未发生的计划、非成员/错误所有者或容器、虚构引用；背包还含
  未购买的物品。题目与金标在手工预检前冻结，没有按实际答案改金标。
- 手工预检每领域提供 3 个独立事实，各自包含来源引文。它们只用于预检，**没有进入 Add**。
  之后实际服务只接收原文，经 HTTP Add → HTTP Search → 既有答案链评分。
- 答案模型 Qwen/Qwen3.5-9B，temperature=0，max_tokens=1024。复用既有 CorporateBench
  类型提示词与判分：列表 set-F1、标量 exact-match，不调用另一个 LLM 改判。
- 这些是自建的小样本迁移诊断，**不是四个官方数据集的评测，也不是 CorporateBench/AML 榜分**。
- 产品基线 Git `3eee19a`，配置快照 `configs/runs/generalization-20261005/`；邻接半径 0。
  独立服务端口 8093；SQLite `var/generalization-20261005/tianxi.db`；Qdrant 集合
  `memories_generalization_20261005`。rerank 关闭，capture 关闭。

完整语料、事实预检与题目由 [prepare.py](runs/generalization-20261005/prepare.py) 生成。
冻结输入 SHA-256：`de3f6d921a732767f9020db6c8cd629951044525865e05262f24648ecd2d9819`。
模型、提示词、判分代码及输入指纹在 `runs/generalization-20261005/frozen.json`。

## 实际结果与取证路径

| 场景 | 手工事实：原问法/改述 | 实际服务：原问法/改述 | Add 实际抽取 | Search 实际返回 |
| --- | --- | --- | --- | --- |
| 公司员工 | 2/2、2/2 | 2/2、2/2 | 4 条任职事实，含 Maya 的重复来源 | 原问法 3 条去重后的任职事实；改述 7 条原文 |
| 学校学生 | 2/2、2/2 | 2/2、2/2 | 没有学生关系事实 | 两种问法均 7 条原文 |
| 收容所志愿者 | 2/2、2/2 | 2/2、2/2 | 4 条 volunteer-place 事实，含重复来源 | 两种问法均 7 条原文，未使用这批志愿地点事实 |
| 背包物品 | 2/2、2/2 | 2/2、0/2 | 没有背包物品事实 | 两种问法均 7 条原文 |
| 合计 | 16/16 | 14/16（87.5%） | 只有两种已有领域抽取器产出事实 | 2 题走事实取证，14 题走原文检索 |

实际服务的原问法为 8/8，改述为 6/8；列表、计数分别为 7/8。这些数字来自相同的
16 道固定题，只是不同分组方式，不能相加为更多独立样本。

公司原问法是 `Who are the employees of Arden Labs?`、
`What is the total number of employees mentioned in the corpus?`；改述为
`Who works at Arden Labs?`、`How many people currently work for Arden Labs?`。
原问法进入事实分支，改述回退原文。**答对没有消除表达识别的缺口。**

学校和志愿者的实际答案正确，但都依赖这批仅 7 条的原文上下文。学校没有抽取到学生
关系；志愿地点已经抽取，名单/人数问题却没有对应的查询路由。因此不能把这两组的
正确答案当作“跨领域事实抽取与取证已经通过”。回退路径包含全部原文与干扰记录；
不是本轮提出的新筛选方案，也不能外推到大语料。

## 两个失败案例与补充对照

| 固定改述问题 | 期望 | 实际答案 | 现象 |
| --- | --- | --- | --- |
| List the kinds of belongings Maya Chen currently keeps in the blue backpack. | `["notebooks", "pens", "ruler"]` | `["two notebooks", "three pens", "one ruler"]` | 类型与数量未分离，严格列表 set-F1 为 0 |
| What is the total number of items currently inside the blue backpack owned by Maya Chen? | `6` | `5` | 2/3/1 的真实原文都已返回，计数仍错 |

从模型最终文本无法确定它为什么算成 5。可以确认的是：不是正确证据缺失；原文检索
未排除虚构引用、他人的背包、计划与重复记录。换问法还改变了原文排名。

为避免把排序变化直接归因为问法或干扰，补做两种固定上下文对照，只测这两道失败题。
问题、金标、模型、提示词及判分保持不变，所有片段仍为原文：

| 上下文条件 | 列表改述 | 计数改述 |
| --- | --- | --- |
| 仅上述一条真实原文 | 正确：`["notebooks", "pens", "ruler"]` | 正确：`6` |
| 保留全部 7 条原文，固定为原计数问法的返回顺序 | 仍错：带数量的名称 | 仍错：`5` |
| 初始手工原子事实预检（非补跑） | 正确 | 正确 |

补充对照共 4 次模型调用；不计入 14/16 的实际服务成绩。结果支持优先验证**证据减噪与
物品/数量拆分**，仅固定排序在这两个样本上无效。没有逐个删除干扰项，不能确定某一项
是唯一原因；temperature=0 也不是普遍稳定性的证明。

补充脚本：[probe.py](runs/generalization-20261005/probe.py)。协议、内容指纹与逐题结果
保留在 `probe-protocol.json`、`probe-results.json` 及 `probes/`。

## 内容与隔离核对

[audit.py](runs/generalization-20261005/audit.py) 已通过，输出 `audit.json`：

- 28 条原文、28 个向量、4 个成功 Add 批次均核对完成；库中 question 与 Add 原文逐字一致。
  answer 均为空，原文及批次在 Search 后保持不变。
- 8 条持久化事实都有同 user_id 的真实父记录；本轮任职/志愿事实只来自成员记录，
  没有把计划、非成员或虚构引用抽成事实。学校和背包的“没抽到”单独记录，未视为成功。
- 16 次 Search 的每条 content 都逐字对应原文渲染或持久化原子事实；没有新增跨记录名单
  或计数结论。8 对“有哪些/多少”查询返回同一组记忆正文，而不是各自生成不同答案。
- 同题 HTTP 重查的 id/content/日期/score 完全一致；top_k=1 的数量约束通过；不存在的
  user_id 在四个领域问题上均返回空数组；未发现跨 user 来源。
- 重查前后原文、批次、所有事实表与覆盖表的整体指纹一致。答案链冻结指纹一致。

这证明的是本轮本地返回内容及隔离检查通过，不替代官方 AML 判定。源记录数量或覆盖标记
只能证明来源已扫描，不能证明语义事实已完整抽取。

首次服务启动因缺少 SQLite 父目录失败；当时 Add 未完成，没有据此评分。创建目录并
确认全部原文/向量已写入后才运行实际评分。一次审计调用漏载 `.env`，冻结检查拒绝了
不同模型默认值；改用本轮相同环境后审计通过，没有接受或覆盖错误指纹。

## 对更高层抽象的结论

共用“主体—关系—对象—限定条件—状态—来源”的表达是合理方向；**现有代码还没有把
所有事实类型统一为这套通用机制**。本轮没有给新领域追加正则或修改 Search。

下一步可在 Add/Search 内部验证三项能力：

1. **共用事实表示**：任职、就读、志愿关系都保留人物、关系、机构及来源；库存保留
   所有者、容器、物品、单项数量及来源。关系本身来自原文，不能把所有机构关系归为任职。
2. **共用条件检索**：同义问法归到同一关系及范围；“有哪些/多少”使用同一证据集合。
   两者在 Search 内都不执行最终名单拼接或 SUM/COUNT。
3. **有依据的语义约束**：否定、计划、虚构引用、所有者/容器错配与重复陈述分别处理。
   同一事实多条来源可去重，真正多次事件必须保留；不能按金标删掉额外真实记录。

改进门槛仍是先手工提出有出处的片段，再以同一固定答案链快速验证；通过后才实现，
并用 HTTP 验证原文到事实的实际转换及回退情况。不得用相邻扩窗或塞满上下文替代取证。

## 本轮覆盖的七个维度

| 维度 | 实际覆盖与限制 |
| --- | --- |
| Explicit fact recall | 四领域关系及物品；未测大语料覆盖 |
| Relational and multi-hop reasoning | 单层人员关系、所有者/容器绑定；未测多跳链 |
| Temporal and event understanding | 当前陈述与未来计划干扰；未测真实跨时更新或日期计算 |
| Memory governance | 原文/批次不变，重复事实去重及查询重放检查；未测删除接口 |
| Personalization and care | 仅所有者范围绑定；没有照护或偏好任务，不报子分 |
| Rules and process execution | 复用既有列表/整数格式规则，未测工作流程任务 |
| Epistemic safety and privacy | 真实/虚构来源区分、同用户溯源、未知用户空结果；仅本地样本 |

## 复现与归档

运行序列为 `prepare.py` → `run_inputs.py manual` → 启动隔离服务 →
`run_inputs.py ingest` → `tools/targeted_eval.py` → `probe.py` → `audit.py`。
答案调用一律串行。调用含模型配置的脚本均使用 `uv run --env-file .env python`。

```bash
uv run python eval/reports/runs/generalization-20261005/prepare.py
uv run --env-file .env python eval/reports/runs/generalization-20261005/run_inputs.py manual
TIANXIMEM_PROFILE=local TIANXIMEM_CONFIG_DIR=configs/runs/generalization-20261005 TIANXIMEM_SQLITE_PATH=var/generalization-20261005/tianxi.db uv run uvicorn tianximem.service.app:create_app_from_env --factory --host 127.0.0.1 --port 8093
uv run --env-file .env python eval/reports/runs/generalization-20261005/run_inputs.py ingest
uv run --env-file .env python tools/targeted_eval.py --manifest eval/reports/runs/generalization-20261005/manifest.json --base-url http://127.0.0.1:8093 --output eval/reports/runs/generalization-20261005/http --freeze eval/reports/runs/generalization-20261005/frozen.json --deadline <未来UTC秒数>
uv run --env-file .env python eval/reports/runs/generalization-20261005/probe.py
uv run --env-file .env python eval/reports/runs/generalization-20261005/audit.py
```

`prepare.py` 拒绝覆盖已冻结的题目和配置。`run_inputs.py ingest` 使用同批次 id 重放时
保持幂等，重放核对已通过。原始产物按仓库惯例 gitignored，报告、脚本及配置快照入库；
没有删除实验数据库或向量集合。新增脚本 Ruff 检查通过；本轮未改产品，不重复跑完整基准
或全量产品单元测试。

一致性 SQLite 快照另外保存在 `var/generalization-20261005/tianxi-snapshot.db`，其指纹及
配置指纹在 `runs/generalization-20261005/archive.json`。验证结束后已关闭本轮隔离服务。
