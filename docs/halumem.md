# HaluMem-Medium 数据集统计报告

**数据源**：`dataset/halumem/HaluMem-Medium.jsonl`（Halu-Medium 档，20 用户）
**加载口径**：[`eval/datasets/halumem.py`](../eval/datasets/halumem.py)——一个评测 Sample = 一个"提问检查点"（该 session 有 `questions` 的才算），只投喂到该检查点为止的对话前缀。
**Tokenizer**：`o200k_base`（tiktoken，§6.4）。**已交叉验证**：本数据集自带的 `dialogue_token_length` 与我用 `o200k_base` 逐 session 重算的结果 **1,387/1,387 完全相等**，即该字段本身就是 o200k 计数。

---

## 一、整体规模统计表

### 原始规模 vs 评测展开规模

| 指标 | 原始规模 | 评测展开规模 |
| --- | ---: | ---: |
| 评测问题总数 | **3,467** | 3,467（一问一 Sample，不重复） |
| 用户数 | **20** | 20 |
| Session 总数 | **1,387** | 1,387（投喂时前缀重复） |
| 唯一 Session 数 | **1,387**（无任何重复） | — |
| QA 总数 | **3,467** | 3,467 |
| 唯一 QA 数（question 文本去重） | **3,189** | — |
| Token 总数 | **3,198,219** | 投喂 **70,411,560**（≈ **22.0×**） |
| Token 去重后 | **2,661,107**（按消息 `content` 去重） | — |

> **重复来源**：①展开侧——加载器对每个检查点重复投喂其**前缀历史**（22×）；②原始侧——278 条 question 文本在不同检查点/用户间重复（8.0%），另有 537,112 token 的**逐字重复消息**（10,470 条 assistant/user 问候与套话）。

### 分布统计（均值 / 中位数 / 最小 / 最大）

| 维度 | n | 均值 | 中位数 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每用户 Session 数 | 20 | 69.4 | 69 | 58 | 82 |
| 每用户 QA 数 | 20 | 173.4 | 173.5 | 137 | 206 |
| 每用户对话 Token | 20 | 159,911 | 158,844 | 131,377 | 187,778 |
| 每 Session Token | 1,387 | 2,305.9 | 2,283 | 323 | 5,066 |
| **每道题关联 Session 数**（前缀） | 3,467 | 35.9 | 36 | 1 | 82 |
| **每道题关联 QA 数** | 3,467 | 1 | 1 | 1 | 1 |
| **每道题关联 Token 数**（前缀） | 3,467 | 78,584 | 76,681 | 323 | 187,778 |
| 每道题证据条数（`evidence`） | 3,467 | 1.34 | 1 | 0 | 9 |

> ⚠ `每道题关联 Session 数` 的中位数 36 来自检查点分布——896 个检查点集中在中等偏后位置；**这不等于"每道题需要 36 个 session 的证据"**，只是加载器投喂的前缀长度。

---

## 二、平均内容规模统计表

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| 每 Session 平均 QA 组数 | **2.50**（中位 3，0–15） | 仅 896/1,387 个 session 带题；无题 session 记 0 |
| 每条 User 消息平均 Token | **22.9**（中位 21，5–194） | 30,073 条 |
| 每条 Assistant 消息平均 Token | **83.4**（中位 83，6–181） | 30,073 条 |
| 每**组完整 QA**（user+assistant 一轮）平均 Token | **106.3**（中位 105，24–223） | 30,073 组 |
| 每 Session 平均 Token | **2,305.9**（中位 2,283） | 每 session 平均 21.7 组 QA |
| 每道评测题关联的完整对话 Token | **78,584**（中位 76,681） | 检查点前缀（1–82 个 session） |

**数量级校验**：20 用户 × 69.35 session/用户 × 2,305.9 token/session = **3,198,300 ≈ 3,198,219**（实测，误差 < 0.01%）；21.7 组 QA/session × 106.3 token/组 = 2,307 ≈ 2,305.9。两条独立路径自洽。Assistant 占全部 token 的 **78.4%**——对话 token 由助手长回复主导，用户话很短（22.9 vs 83.4）。

> 数据集自带统计（`total_dialogue_token_length` 累计、`total_question_count`、`memory_points`）已与重算值对齐：memory points 合计 **14,948**（Persona 9,116 / Event 4,550 / Relationship 1,282）。

---

## 三、记忆问题类型及数量、占比统计表

**官方标签**：数据集自带 `question_type`（6 类）与 `difficulty`（3 档），下表**原样采用，未做任何改写**。

| 官方 question_type | 数量 | 占比 | 难度分布（easy/medium/hard） | 考察能力（官方标签语义） |
| --- | ---: | ---: | --- | --- |
| Memory Boundary | **828** | 23.9% | 588 / 196 / 44 | 记忆边界：问**用户从未提供过**的信息，正确行为是拒答"未知"，考察幻觉抑制 |
| Memory Conflict | **769** | 22.2% | 431 / 319 / 19 | 记忆冲突：对话中存在**相互矛盾**的记忆点，需判别并给出正确结论 |
| Basic Fact Recall | **746** | 21.5% | 692 / 54 / 0 | 基础事实召回：单条记忆点的直接检索 |
| Generalization & Application | **746** | 21.5% | 123 / 612 / 11 | 泛化与应用：从多条记忆归纳出未逐字出现的新结论 |
| Multi-hop Inference | **198** | 5.7% | 49 / 131 / 18 | 多跳推理：串联 ≥2 个记忆点得出答案 |
| Dynamic Update | **180** | 5.2% | 87 / 87 / 6 | 动态更新：记忆被后续对话覆盖，须返回**最新**状态 |
| **合计** | **3,467** | 100% | 1,970 / 1,399 / 98 | easy 56.8% · medium 40.4% · hard 2.8% |

---

## 四、不同记忆能力的特点与代表性问题

⚠ **本节为推断**（依据题面 + `evidence` 字段的条数与跨度，非官方标签）。**多标签口径**：一题若同时满足多条则重复计数，故各比例之和 > 100%；官方的 6 类标签见上表。

| 记忆能力（推断） | 对应官方类型 | 涉及题数 | 证据跨度特征 | 代表性问题（逐字取自数据） |
| --- | --- | ---: | --- | --- |
| **幻觉抑制 / 记忆边界** | Memory Boundary | 828 | `evidence` **全部为空**（828/828）；答案形如 "Unknown; not provided by the user." | *"What is Martin Mark's middle name?"* → *"Unknown; not provided by the user."* |
| **单条事实检索** | Basic Fact Recall | 746 | 证据 1.07 条，恒为 1 条记忆点 | *"What is Martin Mark's birth date?"* → `1996-08-02`（evidence: Persona Memory） |
| **跨记忆归纳与应用** | Generalization & Application | 746 | 证据 **2.42 条**（6 类中最高），须跨 ≥2 记忆点 | 由多条 Persona/Event 记忆推出"之后可能尝试的极限运动类型" |
| **知识冲突处理** | Memory Conflict | 769 | 证据 **1.72 条**，含对抗性干扰记忆（`memory_source: interference`） | 需在相互矛盾的两条记忆里判定哪条成立 |
| **知识更新 / 时序一致性** | Dynamic Update | 180 | 证据 1.42 条，含 `is_update: True` + `original_memories` 链 | 用更新后的记忆回答，忽略被覆盖的旧值 |
| **多跳关联推理** | Multi-hop Inference | 198 | 证据 **2.36 条**、前缀 **40.5 个 session**（最长，均值 90,667 token） | 串联 ≥2 个记忆点做推理 |
| （横切）**时间理解** | 上述各类 | — | 全部记忆点与对话轮均带 `timestamp`；session 带 `start_time`/`end_time` | 如 *"...after April 10, 2026?"* |

**各类型难度特征（按官方类型，量化）**

| question_type | n | 前缀 token 均值 | 前缀 Session 均值/中位 | 证据条数均值 | easy 占比 |
| --- | ---: | ---: | --- | ---: | ---: |
| Memory Boundary | 828 | 80,099 | 35.9 / 36 | **0.00** | 71.0% |
| Basic Fact Recall | 746 | 79,963 | 35.9 / 36 | 1.07 | 92.8% |
| Generalization & Application | 746 | 80,408 | 36.2 / 36 | **2.42** | 16.5% |
| Memory Conflict | 769 | 76,282 | 34.2 / 32 | 1.72 | 56.0% |
| Dynamic Update | 180 | 80,790 | 36.1 / 31 | 1.42 | 48.3% |
| Multi-hop Inference | 198 | **90,667** | **40.5 / 44** | 2.36 | 24.7% |

> **可读出的结论**：证据条数（Generalization 2.42 / Multi-hop 2.36 vs Boundary 0.00）比前缀 token 更能区分类型；**"跨 Session"在本数据集几乎是常量**（36.2 vs 34.2，差别很小）——**不能把跨 session 直接等同于多跳**：Multi-hop 只占 5.7%，而几乎所有题都跨了几十个 session。

---

## 五、数据集结构及字段说明

**格式**：JSONL，**一行 = 一个用户**（`HaluMem-Medium.jsonl`，20 行）。

```
uuid ─┬─ persona_info (str)          # 用户画像文本，[Recorded on ...] 开头，均值 140 token
      ├─ sessions[]  ← 一行一个 uuid，含全部 58–82 个 session
      │    ├─ start_time / end_time     # "Sep 04, 2025, 18:42:18"（session 级，含秒）
      │    ├─ dialogue_turn_num          # 轮对数量
      │    ├─ dialogue[]                 # [{role, content, timestamp, dialogue_turn}]
      │    ├─ dialogue_token_length      # = o200k(dialogue 各 content) —— 已验证
      │    ├─ memory_points[]            # 该 session 抽取的记忆点（数据集标注，非检索产物）
      │    │    └─ {index, memory_content, memory_type, memory_source,
      │    │        is_update, original_memories[], importance, timestamp}
      │    └─ questions[]                # 该检查点的评测题（0–15 条）
      │         └─ {question, answer, evidence[], difficulty, question_type}
      ├─ total_dialogue_token_length
      ├─ total_question_count
      └─ token_cost
```

**组织关系与关联**

- **用户 → Session → Message**：`uuid` 下 `sessions` 数组有序，每个 session 的 `dialogue` 按 `dialogue_turn` 成对（user + assistant）。
- **时间戳**：message 级（`timestamp`，“Sep 04, 2025, 18:42:18”，**秒级**）、session 级（`start_time`/`end_time`）、记忆点级（`timestamp`）、数据集级（`persona_info` 里的 `[Recorded on Oct 04, 2025]`）。
- **标准答案**：`questions[].answer` 是**自由文本**（含 "Unknown; not provided by the user." 这类拒答）。HaluMem 的裁判走 **QA 三分类**（见 [`eval/datasets/CLAUDE.md`](../eval/datasets/CLAUDE.md)）。
- **证据定位**：`questions[].evidence[]` 列 `{memory_content, memory_type}`——**指向记忆点，不是指向对话轮**；Memory Boundary 的 evidence 恒为空数组。
- **记忆点标注**：`memory_source` ∈ {primary, secondary, interference, system}；`is_update: "True"` 的记忆点用 `original_memories` 指回被覆盖的旧记忆（**字符串 "True"/"False"**，不是布尔）。

**重复与去重**

| 层面 | 是否重复 | 去重方式 |
| --- | --- | --- |
| Session（含正文） | **无**（1,387 唯一） | 不需要 |
| 消息 content | 有（60,146 条中 10,545 条重复） | 按 `content` 去重 → token 3,198,219 → **2,661,107** |
| 评测 question 文本 | 有（278 条重复） | 按 question 文本去重 → 3,467 → **3,189** |
| 投喂前缀 | **大量重复**（22×） | 加载器设计如此；embedding 缓存按内容哈希 ⇒ 不重付钱 |

---

## 六、简短总结

- **规模**：20 用户 / 1,387 session / 3,467 道评测题 / **320 万 token**；单用户上下文约 **16 万 token**（与官方 "~160k" 一致）。
- **主要考察的记忆能力**：以 **记忆边界 / 幻觉抑制（23.9%）** 与 **冲突、事实召回、泛化（各约 21.5–22.2%）** 为主，真正的多跳推理与动态更新合计仅 **10.9%**。
- **最突出的评测特点**：
  1. **操作级幻觉评测**——题目直接对应记忆系统的三步操作（抽取 / 更新 / 问答），每道题都挂 `evidence` 记忆点，可定位到"错在哪一步"；
  2. **拒答是最大单一题型**（828 题，evidence 全空），考的是**不乱答**；
  3. **干扰记忆内建**——`memory_source: interference` 的记忆点是数据集主动埋的对抗噪声（2,648 条）；
  4. **检查点式投喂**——同一段历史会在后续检查点被反复喂入（22×），评测展开规模远大于原始正文规模，**统计时必须区分两者**。
