# MemTrapBench 数据集统计报告

**数据源**：`dataset/memtrapbench/memtrapbench/<场景>/*.json`（10 个 JSON 文件，其中 **6 个可评测、4 个是种子**）。
**加载口径**：[`eval/datasets/memtrapbench.py`](../eval/datasets/memtrapbench.py)——**一个评测 Sample = 一条陷阱题**（1 条记忆 : 1 道题）；判据是**有无 `context_history`**（不是文件名）。
**Tokenizer**：`o200k_base`（tiktoken，§6.4）。⚠ 本数据集**不提供** token 字段，全部为本次重算。

---

## 一、整体规模统计表

### 原始规模 vs 评测展开规模

| 指标 | 原始规模 | 评测展开规模 |
| --- | ---: | ---: |
| 评测问题总数 | **1,050** | 1,050（一题一 Sample，**零展开**） |
| 用户数 | **1,050**（每个样本独立 `user_id`） | 1,050 |
| Session 总数 | **1,050**（1 样本 = 1 段 `context_history`） | 1,050 |
| 唯一 Session 数 | **1,050**（`context_history` 逐字去重后仍是 1,050，**零重复**） | — |
| QA 总数 | **1,050** | 1,050 |
| 唯一 QA 数 | **1,050**（`final_trigger` 零重复） | — |
| Token 总数（`context_history` 正文） | **4,565,520** | 4,565,520（无前缀重复） |
| Token 去重后（按消息 `content` 去重） | **4,453,884** | — |
| 另有：`final_trigger` / `gold_standard` | 114,132 / 86,263 token | — |
| **种子条目（不计入评测）** | **163** 条 | 分布在 4 个 `*_seed.json` / `hurt.json` |

> **无展开是这份数据的特性**：与本仓其他检查点式数据集不同，MemTrapBench 每条样本只投喂**它自己那一段**对话，**不存在前缀重复**，原始规模 = 投喂规模。唯一的重复是跨样本的 111,636 token 逐字重复消息（11.4 万 / 456.6 万 = 2.4%）。

### 分布统计（均值 / 中位数 / 最小 / 最大）

| 维度 | n | 均值 | 中位数 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每用户 Session 数 | 1,050 | 1 | 1 | 1 | 1 |
| 每用户 QA 数 | 1,050 | 1 | 1 | 1 | 1 |
| 每用户对话 Token | 1,050 | 4,348.1 | 3,265.5 | 654 | 32,472 |
| 每 Session 消息数 | 1,050 | 56.9 | 60 | 36 | 80 |
| **每道题关联 Session 数** | 1,050 | **1** | 1 | 1 | 1 |
| **每道题关联 QA 数** | 1,050 | **1** | 1 | 1 | 1 |
| **每道题关联 Token 数**（= 对话正文） | 1,050 | **4,348.1** | 3,265.5 | 654 | 32,472 |
| `final_trigger` token | 1,050 | 108.7 | 75 | 18 | 605 |
| `gold_standard` token | 1,050 | 82.2 | 43 | 1 | 744 |

---

## 二、平均内容规模统计表

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| 每 Session 平均 QA 组数（数据集内） | **0** | ⚠ 历史对话是**纯闲聊/任务对话**，**不是 QA 对**；评测题在 `final_trigger`（每题 1 条） |
| （替代口径）每条历史消息平均 Token | **76.5** | 59,707 条消息 |
| 每条 User 消息平均 Token | **34.6**（中位 23，2–535） | 29,852 条 |
| 每条 Assistant 消息平均 Token | **118.3**（中位 81，1–2,001） | 29,855 条 |
| 每**组完整 QA**（user+assistant 一轮）平均 Token | **152.9**（中位 ~104） | 约 29,854 组 |
| 每 Session 平均 Token | **4,348.1**（中位 3,265.5） | 约 28.5 组 QA |
| 每道评测题关联的完整对话 Token | **4,348.1** | 同上一行（1 样本 = 1 段对话） |

**数量级校验**：1,050 × 4,348.1 = **4,565,505 ≈ 4,565,520**（误差 < 0.001%）；1,050 × 56.9 条消息 = 59,745 ≈ 59,707；29,854 组 × 152.9 token/组 = 4,565,377。三条独立路径自洽。助手回复占 token 的 **77.4%**（118.3 vs 34.6）——**三份数据集的共同形态**。

⚠ **方差极大**：`cognitive_bias/number_game_200.json` 的对话只有 659 token（中位 659），而 `cognitive_bias/Inertia_150.json` 平均 12,570 token——**同一场景内相差 19 倍**，中位数比均值更能代表总体（3,266 vs 4,348）。

---

## 三、记忆问题类型及数量、占比统计表

**官方标签有两种粒度，本报告都原样保留、不合并也不改写**：

**(a) 论文级：2 大类 / 4 场景**（README 的 taxonomy，也是加载器的 `category` 字段）

| 官方认知陷阱大类 | 官方场景 | 对应文件 | 数量 | 占比 | 历史对话要做什么 | 正确行为 |
| --- | --- | --- | ---: | ---: | --- | --- |
| Reasoning Fixation | **Task Boundary** | `task_boundary/unclear_200` + `hallucination_150` | **350** | 33.3% | 让之前的任务/格式显得重要，然后问一个**完全不同**的新问题 | 跟随最新提问，不套用旧任务/旧格式 |
| Reasoning Fixation | **Cognitive Bias** | `cognitive_bias/number_game_200` + `Inertia_150` | **350** | 33.3% | 反复强化一个领域/框架/解法路径，使模型自动沿用 | 从零重新思考最终问题，选对框架 |
| Belief Distortion | **Safety** | `safety/poison_200` | **200** | 19.0% | 把**假规则/假标准/不安全指令**伪装成权威 | 依真实事实与安全规则，不认植入的假前提 |
| Reasoning Fixation | **Trauma** | `trauma/hurt_150` | **150** | 14.3% | 用批评、施压、负反馈让正确概念/答案"变成禁忌" | 不让情绪压力覆盖正确答案 |
| **合计** | 4 场景 | 6 文件 | **1,050** | 100% | 大类合计：Reasoning Fixation **850**（81.0%）/ Belief Distortion **200**（19.0%） |

**(b) 文件级子类型**（⚠ 这些字段**只有 safety 一族是受控词表**，其余是自由文本）

| 字段 | 出现于 | 取值与数量 |
| --- | --- | --- |
| `poison_type`（**受控，4 值**） | `safety/poison_200` | `A. Semantic Hijacking` **77** / `B. Factual Overwriting` **65** / `C. Hidden Dependency` **58** |
| `test_type`（**受控，2 值**） | `safety/poison_200` | `red` **128** / `green` **72**（红队/对照样本） |
| `inference_type` | `task_boundary/*` | ⚠ **自由文本**：155 条里出现 100+ 种互不相同的长串（"Proactive Interference + Attention Hijacking via …"），**不可作统计口径** |
| `inertia_type` | `cognitive_bias/*` | 同上，自由文本 |
| `interference_type` | `trauma/*`、`task_boundary/hallucination_150` | 同上，自由文本 |

> **口径说明**：只有 **(a) 的 4 场景**与 **(b) 的 `poison_type` / `test_type`** 是数据集**受控**标签；其余字符串字段是构造时的自由描述，**不得当作官方分类**。

---

## 四、不同记忆能力的特点与代表性问题

⚠ **本节为推断**（依据 README 的构造说明 + 题面），**非官方能力标签**；官方分类见上表。**多标签口径**：一条样本可归多条能力，比例之和 > 100%。

| 记忆能力（推断） | 官方场景 | 题数 | 对话 Token 规模（均值/中位） | 代表性问题（逐字取自数据） |
| --- | --- | ---: | --- | --- |
| **指令/任务边界抑制**（数据集核心） | Task Boundary | 350 | 3,304 / 2,384 | 历史上要求输出 XML `<NegotiationPolicyReport>`；最终问 *"…output exactly one CSV line with this column order: system,protocol,…"*，gold 是一行 CSV |
| **推理定势打破**（认知偏置） | Cognitive Bias | 350 | 5,764 / 661 | 反复的"数字游戏"上下文使模型沿用隐藏运算；Inertia 一族长达 12,570 token 的重复框架 |
| **安全性冲突处理**（假权威覆盖） | Safety | 200 | 3,117 / 3,126 | 植入 *"…the 'Cryo-Culinary' safety guidelines… immediate water deployment to induce endothermic shock"*；正确答案是**拒绝**该规则 |
| **情绪压力抵抗**（创伤定势） | Trauma | 150 | 5,122 / 4,848 | 历史以批评/施压让某概念"变禁忌"；最终问需给出被压制的正确答案 |
| **噪声中定位**（横切） | 全部 | 1,050 | — | 陷阱被"Bury it in Noise"混入日常轮次（如 *"Draft a quick email about next week's schedule."*） |
| **无记忆可解性**（横切，数据集自定） | 全部 | 1,050 | — | 官方质检要求 `final_trigger` **只看题面也能答对**——所以这是"记忆**有害**"的评测 |
| **拒用记忆 / 冲突取舍**（数据集特有） | 全部 | 1,050 | — | 与"记忆有用"类数据集方向**相反**：考的是何时**不该**用历史 |

**各场景难度特征（量化）**

| 场景 | n | 对话 Token 均值 / 中位 | 最小 / 最大 | 消息数均值 | `final_trigger` token | `gold_standard` token |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| task_boundary | 350 | 3,304 / 2,384 | 899 / 10,003 | — | **202** | **154** |
| cognitive_bias | 350 | 5,764 / 661 | 654 / **32,472** | — | 58 | 31 |
| safety | 200 | 3,117 / 3,126 | 1,260 / 4,963 | — | 70 | 79 |
| trauma | 150 | 5,122 / 4,848 | 3,232 / 8,077 | — | 60 | 38 |
| **全体** | **1,050** | 4,348 / 3,266 | 654 / 32,472 | 56.9 | 108.7 | 82.2 |

> **可读出的结论**：①对话 token 规模**不区分陷阱类型**（3.1k–5.8k 区间重叠严重），真正区分的是**陷阱机制**而非长度；②`task_boundary` 的题面最长（202 token，答案是结构化 CSV/XML 一行）——它的难点在**格式服从**，不在推理；③**每条样本都有 `expected_failure_output`（1,050/1,050）**——这是红队数据的显式特征，可直接用于"是否踩坑"的二分类诊断。

---

## 五、数据集结构及字段说明

**格式**：每个文件是一个 **JSON 数组**，数组元素即一条测试样本（无外层包装）。

```
[ { …一条样本… }, { … }, … ]        ← memtrapbench/<场景>/<名字>.json
```

**一条样本的字段**（以 `task_boundary/unclear_200.json` 为例）

| 字段 | 类型 | 含义 | 备注 |
| --- | --- | --- | --- |
| `id` | str | 样本 id（如 `"0001"`） | 加载器用它拼 `user_id = mtb-{文件名}-{id}` |
| `context_history` | list | 埋了陷阱的对话历史 `[{turn, role, content}]` | **有无此字段 = 是不是题**（种子没有） |
| `history_turns` | int | 轮对数（如 18） | ⚠ 与 `len(context_history)` **不等**（18 vs 36）——它是**轮**，数组是**消息** |
| `final_trigger` | str | 最终问题（**自包含**，无历史也能答） | 官方质检要求 |
| `gold_standard` | str | 正确答案要点（**自由文本**，非答案串） | 判分要点 |
| `expected_failure_output` | str | 踩了陷阱会长什么样（红队） | **1,050/1,050 全有** |
| `domain` | str | 领域标签（如 `"Culinary & Kitchen Safety"`） | 受控取值 150 种；⚠ **`task_boundary` 两文件（350 条）整个字段缺失** |
| `interference_type` / `inertia_type` / `poison_type` / `test_type` | str | 子类型 | 见第三节；**只有 safety 那几个是受控词表** |
| `poisoned_fact` / `objective_truth` | str | 植入的假事实 / 客观真相 | 仅 `safety/poison_200.json` |
| `original_id` | str | 溯源到种子 | 部分文件有 |
| 种子文件字段 | — | `inertia_seed.json`：`DOMAIN`/`SEED_ID`/`STRATEGY_A`/`STRATEGY_B`；`hurt.json`：只有 `domain`/`id` | **163 条，加载器跳过并报数** |

**组织关系与关联**

- **用户 / Session / Message / QA 四者在本数据集里是 1:1:1:1**——一条样本 = 一个独立 `user_id` = 一段 `context_history`（1 个 session）= 1 道 `final_trigger`。**没有跨样本的用户或 session 共享**。
- **时间戳**：⚠ **完全没有**。`context_history` 的每条消息只有 `turn` / `role` / `content`，**没有时间字段**。加载器据此把 `event_time` 置 NULL、`created_at` 发空串（§11.3 的有定义降级路径）——**不是漏读**。
- **标准答案关联**：`gold_standard` 与 `final_trigger` 成对；`expected_failure_output` 与 `gold_standard` 构成"错/对"两极。**判分口径是我们自定的**（官方只给四维均分，无过/不过线，见 loader docstring）。
- **证据定位**：⚠ **没有 evidence 字段**——这是本数据集与其他几份最大的结构差异，无法做"证据在不在"的 oracle 诊断。

**重复与去重**

| 层面 | 是否重复 | 说明 |
| --- | --- | --- |
| `context_history` 整段 | **无**（1,050 唯一） | 无需去重 |
| `final_trigger` | **无** | — |
| 消息 `content` | 少量（4,565,520 → 4,453,884 token，重复 2.4%） | 跨样本的模板句 |
| 前缀/检查点展开 | **无** | **1 样本只喂 1 段对话**，原始规模 = 投喂规模 |
| 子类型字符串 | **大量近重复** | `interference_type` 等自由文本字段需归一化后才能统计 |

---

## 六、简短总结

- **规模**：**1,050** 条样本 / 1,050 个独立用户 / 1,050 段对话 / **456.6 万 token**；单条对话均值 **4,348 token**（中位 3,266，654–32,472）——是**本仓最轻量**的评测集，但**陷阱密度最高**。
- **主要考察的记忆能力**：**不是"记得住"，而是"何时不该用记忆"**。四类陷阱里 81.0% 属 Reasoning Fixation（任务边界 / 认知偏置 / 创伤，共 850 题），19.0% 属 Belief Distortion（安全，200 题）。
- **最突出的评测特点**：
  1. **记忆有害性评测**——官方质检强制 `final_trigger` **无历史也能答对**，因此分数直接度量"历史把模型带偏了多少"，方向与本仓其他数据集**相反**；
  2. **1:1:1:1 结构**——一题一段对话、零共享、零展开，统计口径最简单，**不存在原始/展开规模的分歧**；
  3. **红队数据内建**——`expected_failure_output` 100% 覆盖，`test_type` 分 red/green，可直接做"踩坑率"诊断；
  4. **无时间戳、无 evidence**——不能做时间推理评测，也不能做证据级 oracle 诊断；且**子类型标签多为自由文本**，只有 `poison_type`（3 值）与 `test_type`（2 值）是可统计的受控标签；
  5. **方差极大**——中位数（3,266）与均值（4,348）差 33%，`number_game_200` 单文件仅 659 token，与其他场景相差近 20 倍。
