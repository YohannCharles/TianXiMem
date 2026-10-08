# MedMemoryBench（中文档）数据集统计报告

**数据源**：`dataset/medmemorybench/data/zh/`——`dialogues.parquet`（医学会话）、`queries.parquet`（评测题）、`personas.parquet`、`events.parquet`、`trap_events.parquet`、`clinical_reports.parquet`。
⚠ **归档的是 zh 干净档**：`dialogues_with_noise.parquet` / `noise_sessions.parquet`（噪声与家族问诊）**未取**，故本报告不含噪声会话。
**加载口径**：[`eval/datasets/medmemorybench.py`](../eval/datasets/medmemorybench.py)——一个评测 Sample = 一个 (persona, 检查点)，`session_id` 是 10 的倍数（10…100），**记忆只喂到该检查点为止**（不读未来）。
**Tokenizer**：`o200k_base`（tiktoken，§6.4）。⚠ 本数据集**不提供** token 字段，全部为本次用 o200k 重算；中文按 o200k 计约 0.6–0.7 token/字。

---

## 一、整体规模统计表

### 原始规模 vs 评测展开规模

| 指标 | 原始规模 | 评测展开规模 |
| --- | ---: | ---: |
| 评测问题总数 | **1,939** | 1,939（一题一 Sample，不重复） |
| 用户数（persona） | **20** | 20 |
| Session 总数 | **2,020**（101 / persona） | 2,020（投喂时前缀重复） |
| 唯一 Session 数 | **2,020**（无重复） | — |
| QA 总数 | **1,939** | 1,939 |
| 唯一 QA 数（question 文本去重） | **1,938**（仅 1 条重复） | — |
| Token 总数 | **13,860,137** | 投喂 **73,942,112**（≈ **5.33×**） |
| Token 去重后 | **13,860,137**（**零重复**，逐字无一相同） | — |
| 评测检查点数 | — | **200**（20 persona × 10 检查点） |

> **重复来源只有一个**：展开侧的前缀重复（5.33×，加载器 `SHAPE_NOTE` 记录为 ≈5.4×）。**原始正文没有任何逐字重复** —— `dialogues.parquet` 的 31,976 条 `content` 去重后仍是 31,976 条。

### 分布统计（均值 / 中位数 / 最小 / 最大）

| 维度 | n | 均值 | 中位数 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每用户 Session 数 | 20 | **101** | 101 | 101 | 101 |
| 每用户 QA 数 | 20 | 96.95 | 97 | 89 | 100 |
| 每用户对话 Token | 20 | 693,007 | — | 604,956 | 795,316 |
| 每 Session 消息数（user+assistant） | 2,020 | 15.83 | 16 | 12 | 16 |
| 每 Session Token | 2,020 | 6,861 | 6,625 | 2,269 | 15,315 |
| **每道题关联 Session 数**（前缀） | 1,939 | **54.3** | 50 | 10 | 100 |
| **每道题关联 QA 数** | 1,939 | 1 | 1 | 1 | 1 |
| **每道题关联 Token 数**（前缀） | 1,939 | **369,332** | — | 67,842 | 693,007 |
| 每道题证据知识点数（`source_key_points`） | 1,939 | 5.28 | 2 | 1 | 96 |
| 每道题 gold 答案条数（`answers`） | 1,939 | 1.62 | 1 | 1 | 4 |

> ⚠ "每道题关联 Session 数 54.3"是**加载器投喂的前缀长度**（检查点 10 → 10 个 session，检查点 100 → 100 个），**不是证据所需 session 数**。真实的证据跨度见第四节。

---

## 二、平均内容规模统计表

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| 每 Session 平均 QA 组数 | **0** | ⚠ 本数据集的对话是**医患问诊**，**不含 QA 对**；评测题单独存在 `queries.parquet` 里 |
| （替代口径）每检查点平均评测题数 | **9.70**（中位 9，7–11） | 200 个检查点合计 1,939 题 |
| 每条 User 消息平均 Token | **195.9**（中位 179，2–974） | 15,988 条（患者） |
| 每条 Assistant 消息平均 Token | **671.0**（中位 642，36–1,842） | 15,988 条（医生） |
| 每**组完整 QA**（patient+doctor 一轮）平均 Token | **866.9**（中位 ~830） | 15,988 组 |
| 每 Session 平均 Token | **6,861**（中位 6,625） | 每 session 7.9 组 QA（15.8 条消息） |
| 每道评测题关联的完整对话 Token | **369,332** | 检查点前缀（10–100 个 session） |

**数量级校验**：20 persona × 101 session × 6,861 token/session = **13,859,220 ≈ 13,860,137**（误差 < 0.01%）；7.9 组 QA/session × 866.9 token/组 = 6,848 ≈ 6,861。两条独立路径自洽。医生回复占 token 的 **77.4%**（671.0 vs 195.9），与 HaluMem 同一条形。

---

## 三、记忆问题类型及数量、占比统计表

**官方标签**：`queries.parquet` 自带 `query_type`（6 类），下表**原样采用**。⚠ `metadata.inference_type` 是**自由文本**（191 类 inference_generation 题里出现 250+ 种互不相同的写法），**不是**可控分类，故不作统计口径。

| 官方 query_type | 数量 | 占比 | 难度（hard/easy/medium） | 考察能力（官方定义） |
| --- | ---: | ---: | --- | --- |
| Entity Exact Match | **400** | 20.6% | 266 / 31 / 103 | 精确实体召回：直接命中单个医学事实 |
| Multiple Choice | **398** | 20.5% | 398 / 0 / 0 | 选择题：从选项中挑出正确答案（内建 trap 干扰） |
| Inference & Generation | **381** | 19.6% | 380 / 0 / 1 | 推理与生成：结合患者特殊情况推导结论 |
| Temporal Localization | **374** | 19.3% | 21 / 282 / 71 | 时间定位：确定事件"发生在什么时候" |
| State Update Tracking | **195** | 10.1% | 0 / 163 / 32 | 状态更新追踪：追踪用药/症状/检查值的变化 |
| Multi-hop Clinical Deduction | **191** | 9.8% | 191 / 0 / 0 | 多跳临床推理：串联多个事实做因果链推导 |
| **合计** | **1,939** | 100% | 1,256 / 476 / 206（+1 缺失） | hard 64.8% · easy 24.5% · medium 10.6% |

---

## 四、不同记忆能力的特点与代表性问题

⚠ **本节为推断**（依据题面 + `source_key_points`/`metadata`，**非官方能力标签**）。**多标签口径**：一题可归多条能力，故比例之和 > 100%；官方 6 类标签见上表。

| 记忆能力（推断） | 对应官方类型 | 题数 | 证据跨度（唯一 `session_id` 数） | 代表性问题（逐字取自数据） |
| --- | --- | ---: | --- | --- |
| **单条事实检索** | Entity Exact Match、Temporal Localization | 774 | **1.00 / 1.00**（恒为单 session 单知识点） | *"患者既往病史中提到的慢性代谢性疾病是什么？"* → `糖尿病` |
| **时间信息理解与推理** | Temporal Localization、Multi-hop | 374+ | 1.00；`time_type` = absolute_time 253 / event_at_time 117 / event_time 4 | *"患者近期裤腰变松并再次下降约1-2斤的体重变化被记录在什么时候？"* → `2024-01-15` |
| **知识更新与冲突处理** | State Update Tracking | 195 | **18.04**（中位 10，最大 **96**）——需回看整条用药史 | *"患者此前一直规律服用恩格列净，请问他目前对恩格列净的用药状态是什么？"* → `已停用` |
| **跨 Session 信息整合** | Inference & Generation | 381 | 1.71（1–3） | 需把分散在 2–3 次问诊里的指标与病程拼起来 |
| **多跳关联推理** | Multi-hop Clinical Deduction | 191 | **2.60**（2–4；`sessions_involved` 实测 3 节点 131 / 2 节点 28 / 4 节点 28） | *"…明明药一直按时吃，是不是说明现在的药已经压不住血糖了？"*（hop_count 恒为 4） |
| **用户偏好与个性化记忆** | Multiple Choice、Inference | ~579 | 1.74 | 选项由**患者特殊禁忌/偏好**设计陷阱（如"对头孢呋辛过敏"） |
| **医疗安全 / 陷阱规避**（数据集特有） | Multiple Choice、Inference | 589 | — | `trap_events` 120 条 + MCQ 的 `trap_design`（allergy_cross / temporal_state 等） |

**各类型难度特征（按官方类型，量化）**

| query_type | n | 前缀 token 均值 | 证据知识点均值 | 证据唯一 session 均值 | 前缀 Session 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| entity_exact_match | 400 | 369,711 | 1.00 | **1.00** | 55.0 |
| multiple_choice | 398 | 368,534 | 1.90 | 1.74 | 54.9 |
| inference_generation | 381 | 368,814 | 1.82 | 1.71 | 54.6 |
| temporal_localization | 374 | 368,773 | 1.00 | **1.00** | 54.7 |
| state_update | 195 | 368,510 | **29.41** | **18.04** | 54.8 |
| multi_hop_clinical_deduction | 191 | **396,665** | 2.75 | **2.60** | **59.2** |

> **可读出的结论**：①前缀 token 在所有类型上几乎相同（36.9 万，差 <1%），**前缀长度不区分难度**——真正区分的是**证据跨度**；② State Update 的关键知识点多达 29 条、跨 18 个 session，是"要翻遍用药史才答得对"的一类；③ Multi-hop 的 `sessions_involved` 实测 **2–4 个节点**，`hop_count` 恒为 4，时间跨度均值 **50.5 天**。**"跨 Session"（几乎所有题都跨 50+ 个 session）≠ 多跳（191 题，仅 9.8%）**。

---

## 五、数据集结构及字段说明

**格式**：6 个 Parquet 文件，均为**长表**（一行一记录），通过 `persona_id` / `session_id` / `event_id` / `query_id` 关联。

```
personas.parquet     20 行 —— persona_id, type_name, gender, category, core_feature,
                                 health_goals, age_range, occupation_detail, background_story,
                                 sleep_pattern, diet_habits, exercise_frequency, stress_level,
                                 medical_history, disease_progression
      │
      ├─ events.parquet        2,074 行 [persona_id, event_id, event{text}, type, event_date, triggered_by]
      ├─ trap_events.parquet     120 行 —— 120 = 20 persona × 6 类安全事件（allergy / medication_history /
      │                                      disease_history / medication_preference / diet_preference /
      │                                      lifestyle_economic）
      ├─ dialogues.parquet    31,976 行 —— 一行一消息
      │      [persona_id, session_id(1–101), turn, role(user=患者/assistant=医生),
      │       content, agent_type(user_agent/doctor_agent), event_id,
      │       event_info(JSON:{event,type,date}), knowledge_points(JSON[]), kp_count, turn_count]
      ├─ clinical_reports.parquet 20 行 —— phase_1..phase_5 五阶段临床报告（长文本）
      └─ queries.parquet       1,939 行 —— 一行一评测题
             [persona_id, query_id, session_id(检查点,10 的倍数), query_type,
              question, answers(JSON[]), source_key_points(JSON[]), metadata(JSON)]
```

**组织关系与关联**

- **用户 → Session → Message**：`(persona_id, session_id)` 唯一确定一次问诊；`turn` 是 session 内轮次。**一次问诊共享一个 `event_info`**（同一天、同一事件），故 **session 内没有时间区分度**（与 LoCoMo / LongMemEval 同形，§6.1）。
- **时间戳**：`event_info.date`（`YYYY-MM-DD`，session 级）；`trap_events.event_date`；知识点 `time`（可空，本数据里出现过字符串 `"None"`）。
- **标准答案**：`answers` 是 JSON 数组，每条 `{content, is_correct, explanation}`——**`is_correct` 是 gold 的标记，不是模型答案**。全库 1,939 题共 **3,133 条** gold 答案，其中 **2,263 条** `is_correct: true`。⚠ **选择题（`multiple_choice`）的 `answers` 把全部 4 个选项都列出**（该类型 1,592 条答案里只有 **722 条**为 true，即平均每题 1.8 条"正确"文本），判分须按 `metadata.correct_count` 区分，**不能用"答案条数 = 1"假设**。
- **证据定位**：`source_key_points` 是知识点数组 `{category, name, content, trap_score, time, session_id}`——**`session_id` 指回具体问诊**，这是 `state_update`（均值 29 条）与 `multi_hop` 的关键。
- **难度与陷阱**：`metadata.difficulty`；MCQ 的 `metadata.trap_design`（`trap_type` / `trap_mechanism` / `why_others_fail` / `required_patient_info`）；Multi-hop 的 `metadata.reasoning_chain` / `sessions_involved` / `time_span_days`。

**重复与去重**

| 层面 | 是否重复 | 说明 |
| --- | --- | --- |
| 消息 `content` | **无**（31,976 唯一） | token 去重前后同为 **13,860,137** |
| 评测 question 文本 | 1 条重复 | 1,939 → 1,938 |
| `knowledge_points` | **高度累积** | ⚠ `kp_count` 随 `session_id` **单调递增**（persona 1：session 1 = 8 → session 101 = 256），即每行携带的是**到该次问诊为止的累计知识点**，不是本次新增。统计时按 session 取一次，勿逐 turn 累加 |
| 投喂前缀 | **重复**（5.33×） | 加载器检查点设计的固有代价 |

---

## 六、简短总结

- **规模**：20 患者 × 101 次问诊 = 2,020 session / 31,976 条消息 / 1,939 道题 / **1,386 万 token**；单患者 12 个月病程约 **69 万 token**，单题前缀约 **37 万 token**。
- **主要考察的记忆能力**：**单条事实检索与时间定位合计占 39.9%**（774 题），是最轻的一端；**状态更新追踪（10.1%）与多跳临床推理（9.8%）** 是最重的两端，前者要跨 18 个 session 翻用药史，后者要串 2–4 个病例节点。难度自报 **hard 占 64.8%**。
- **最突出的评测特点**：
  1. **检查点式时序评测**——每次问诊后提问"到此刻为止"的记忆，`session_id` 是 10 的倍数，**天然禁止读未来**；
  2. **医疗安全陷阱内建**——120 条 `trap_events` + 每题 `trap_design`，考的是"记得住 + 不出危险建议"；
  3. **原始正文零重复、展开 5.33×**——统计原始规模与投喂量必须分开；
  4. **对照强**——中英双语对齐、`events.parquet` 提供 2,074 条因果事件时间线（`triggered_by` 建因果图），可做"证据在不在"的 oracle 诊断。
