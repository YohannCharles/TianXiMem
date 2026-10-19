# 共同事实分支：规则抽取 vs LLM 抽取（2026-10-16）

**问题**：只走共同事实取证分支（不跑答案/判分），规则事实层与 LLM 事实层各自能命中多少题、
命中得准不准？

**口径**：

* 题单：从 v1.1.x 归档题单里**跨数据集抽样**（`tools/facts_rule_vs_llm.py sample`）；
* 两套事实库建在**同一份 qa_pairs** 上：`rule` = 工作区规则抽取（`memory-facts-v9-entities`），
  `llm` = LLM 抽取（逐 scope 抽、自由关系名 + 本地触发词映射，见下）；
* 每题的**计划完全相同**（确定性编译 → 失败才用 planner v2；计划跨两臂复用）；
* 未命中（无计划 / 选择为空 / 审查不过）**直接跳过**，只统计命中题；
* 命中率 = 非空选择 / 题数；精确率 = 命中题里**金标答案逐字出现在所返回证据**中的比例
  （列表金标按"同义组任一全中"；不给可比对金标的题进命中率分母、不进精确率分母）。

## 一、本轮结果（配对样本；LLM 臂覆盖 711/715 篇，两臂同分母 93 题）

| 数据集 | 层 | 题数 | 命中 | 命中率 | 可比命中 | 证据含金标 | 精确率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| locomo-refined | rule | 25 | 1 | 0.04 | 1 | 1 | 1.00 |
| locomo-refined | llm | 25 | 0 | 0.00 | 0 | 0 | - |
| halumem | rule | 12 | 0 | 0.00 | 0 | 0 | - |
| halumem | llm | 12 | 0 | 0.00 | 0 | 0 | - |
| musique | rule | 6 | 0 | 0.00 | 0 | 0 | - |
| musique | llm | 6 | 0 | 0.00 | 0 | 0 | - |
| tempreason | rule | 25 | 24 | 0.96 | 24 | 24 | 1.00 |
| tempreason | llm | 25 | **23** | 0.92 | 23 | 23 | 1.00 |
| personamem-v2 | rule | 25 | 0 | 0.00 | 0 | 0 | - |
| personamem-v2 | llm | 25 | 0 | 0.00 | 0 | 0 | - |
| **合计** | **rule** | 93 | 25 | **0.27** | 25 | 25 | **1.00** |
| **合计** | **llm** | 93 | 23 | **0.25** | 23 | 23 | **1.00** |

（`longmemeval-s` 只有 1 题且那一问没计划，单列会失真，并入合计分母。）

**分支结局**（两臂相同，因为计划相同）：

```text
abstained 29 · unsupported_scope 20 · selected 25/23 · unsupported_source 6/2 ·
no_match 3/9 · invalid_relations 3 · missing_period 3 · unsupported_walk 2 ·
invalid_period 2 · source_scan_limit 1
```

**两臂的差距全部来自 tempreason 的那 1 题**（24 vs 23），精确率两臂都是 1.00
⇒ 在**能编出计划的题**上，规则层与 LLM 层**打平**。


### 计划从哪来（计划在"命中率"里的分量）

| 数据集 | 题数 | 编出计划 | 其中确定性编译 / planner | 未命中率 |
| --- | ---: | ---: | ---: | ---: |
| tempreason | 25 | **24** | 24 / 0 | 0.04 |
| locomo-refined | 25 | 7 | 1 / 6 | 0.96 |
| halumem | 12 | 4 | 0 / 4 | 1.00 |
| musique | 6 | 2 | 1 / 1 | 1.00 |
| longmemeval-s | 1 | 0 | 0 / 0 | 1.00 |
| personamem-v2 | 25 | **0** | 0 / 0 | 1.00 |

⇒ **上限几乎完全由"能不能编出计划"决定**：tempreason 的题是外显三元组问法（"某人担任过什么职位"），
24/25 编得出计划、两臂都命中且精确率 1.00；其余数据集 0–7 题能编出计划，
两臂的差距在这个分母下根本量不出来。planner 的失败原因分布：
`abstained 29`（模型自己弃权）、`unsupported_scope 20`（否定/相对时间等护栏）、
`invalid_relations 3`、`missing_period 3`、`invalid_period 2`。

## 二、怎么读这张表

1. **两臂打平**（命中 25 vs 23，都在 tempreason；精确率都是 1.00）。这批题上，
   **限制因素是"计划"而不是"事实层"**：93 题里只有 ~28% 编得出计划
   （`abstained 29` + `unsupported_scope 20` + `invalid_*` 8 就占了 57 题）——
   问法（长对话的开放问句、政策文本的指代问句）本来就落不进 `filter`/`walk` 的词表。
2. **能编出计划的题，两套事实层都答得准**：tempreason 规则 24/25、LLM 23/25，精确率都是 1.00，
   因为它的问法是"某人担任过什么职位"这类**外显三元组**，规则层与 LLM 层都抽得到。
3. **LLM 臂没有翻盘**：它在 LoCoMo 上抽出了规则层没有的关系
   （`Jon | occupation | banker`、`Gina | employee | Door Dash`），但那些题**编不出计划**
   ⇒ 抽得出来也进不了分支。这再次印证：瓶颈已经不在抽取。

## 二·补、两次"看起来是抽取差距、其实是口径/表示"的教训（本轮实测）

1. **时序限定条件**：LLM 事实一开始**不带 `start_month/end_month`**（interval 问法靠它定位）
   ⇒ tempreason 的 LLM 臂从 24 命中直接掉到 **0**（`no_match`）。补上"从引句里确定性解析
   显式月份区间"（复用规则层的 `_INTERVAL`/`month_index`）后才回到 23。
   ⚠ **同一份事实、只差一个限定字段，命中率就能从 0.92 掉到 0.00**——分层对照必须对齐
   "事实表示"再比。
2. **精确率口径**：证据里写 `2023-03-16`、金标写 `16 March, 2023` ⇒ 逐字包含判据会把
   **对的证据判成错**（locomo 那条命中一开始被记成 0.00）。改成日期按日历比之后，
   规则臂精确率 0.96 → 1.00。⚠ 跨数据集的"证据含金标"判据**必须先做日期/数字归一化**。

## 三、LLM 臂的实现（与坑）

* **不给关系词表**：2026-10-16 实测，这个模型一看到 "allowed labels" 就一律返回空
  （0 词表 3s 抽对；15 词表 13s 返回 `[]`）。⇒ 关系名由模型自由写，落库前用
  `fact_llm_gate` 的**触发词表**映射到规范关系（`canonical_relation`）。
* **按说话人轮次切 scope**：整段长记录（>1500 字）会让模型返回空；
  且 `fact_llm_gate._scopes` 在**单说话人**记录上退化成句子级，会把
  "Jon: … Lost my job as a banker" 的 `Jon` 与 `banker` 判成不同范围 ⇒ 本地实现
  按"命名说话人轮次"切（`_turn_scopes`），引句仍逐字取自 question/answer 侧。
* **闸门**：关系在词表内 + 同范围 + （触发词许可 or 结构化字段块 or `Corpus:` 行）。
  比 `fact_llm_gate` 宽一档（它只认触发词，会把签名块全毙掉，而规则层是认签名块的）。
* **断点续跑**：模型输出按 pair id 落 JSONL 缓存；带 `error` 的行**不进缓存**，
  下次运行自动重试（网关会整段时间 5xx）。

## 四、已知限制（不要外推）

* LLM 臂覆盖 **711/715 篇**（网关：单次 5–90s，高峰 5xx/524，靠重试补齐）；
  报告里另出一张"只看已抽完用户"的同分母表（93 题，两臂一致口径）。
* LLM 臂的候选（365 条）里只有约三成能落进词表：`lost job as → occupation`、
  `currently lives in → live` 能映射，`went to`、`competed in`、`got` 这类**自由动词没有触发词**
  ⇒ 被丢。也就是说这张表量的是"**能被规范词表接住的 LLM 抽取**"，不是 LLM 抽取的上限。
* 样本小：每数据集 1–25 题，未命中直接跳过 ⇒ 命中率的分母是"能编出计划的题"之外的全体。
* 事实层的产出量（同一份 715 篇语料）：规则层 39 条事实 / LLM 层 9 条（二者都很少，
  因为样本里大半记录（BEAM 教程、MedMemoryBench 中文问诊、MemTrapBench 数字游戏）本来就不陈述关系）。
* 精确率是**证据级**（金标串是否出现在返回的证据里），不是答案级；
  答案级要接答案/判分，本轮按要求不跑。
* `musique`/`personamem`/`halumem` 等在这批样本上**两臂都是 0**：不是抽取差，
  而是问法编不出计划（`unsupported_scope` / `abstained`）。

## 五、专门的"LLM 赢"搜索：LoCoMo 全量 346 题（2026-10-17 补）

上一节的分母掩盖了差异（93 题里 57 题根本没进分支）。这一节换个做法：
**把 LoCoMo-Refined 三段对话（conv-26/30/41，769 篇语料）两臂都抽满，跑它全部 346 题**。

### 两臂各自的事实清单是**互补**的

| 关系 | 规则层 | LLM 层 |
| --- | ---: | ---: |
| `participated` / `friends from` / `volunteer` / `is open` / `moved from` / `accepted` / `home country` / `big fan` | **39 条** | 部分（映射后 13 条） |
| **`employee` / `occupation` / `started`** | **0 条** | **4 条** |

规则抽取器在**对话体**里不产出任何就业类事实（它的语法只覆盖署名块/标注字段那几种句式），
而 LoCoMo 里"我昨天丢了银行的工作""我在 Door Dash 的工作没了"这类句子**只有 LLM 抽得到**。

### 结果（346 题，两臂同一批计划）

| 层 | 命中 | 命中率 | 精确率（证据含金标） |
| --- | ---: | ---: | ---: |
| 规则 | 5 | 0.014 | **1.00**（5/5） |
| LLM | 4 | 0.012 | **0.50**（2/4） |
| **并集**（任一层命中） | **6** | 0.017 | **1.00**（6/6） |

逐题归属：共同 3 · **规则独有 2** · **LLM 独有 1**。

**LLM 独有那一题就是它结构上赢的地方**：

* `conv-30#q0001`「When Gina has lost her job at Door Dash?」（gold `January, 2023`）——
  LLM 层有 `Gina | employee | Door Dash`（来源日期 2023-01-20）⇒ 命中且证据含金标；
  规则层对同一题是 `unsupported_source`（**它根本没这条事实**）。

**规则独有那两题**是镜像：`conv-30#q0012`（店开张，规则有 `is open`）、
`conv-41#q0006`（在哪交的朋友，规则有 `friends from`）——LLM 层没抽到/没映射上。

⚠ **单独用 LLM 层反而更差**：它的 4 条命中里 2 条证据不全（`participated | fair` 漏了
"networking events"；`moved from | home country` 没给出国家名）⇒ 精确率 0.50。
**并集才两头都占到**（6/6，精确率 1.00）。

### 这一节的结论

* ✅ **用户的判断成立**：确实存在"LLM 抽得出来、规则抽不出来"的题——**对话体里的就业/职业关系**。
* ⚠ 但**增量是 1 题**（346 题里），而且**必须与规则层并存**：LLM 层的映射更粗，
  单独用会把精确率从 1.00 拖到 0.50。
* ⛔ 天花板仍然是**计划**：346 题里 `abstained 154` + 各类非法输出 ~40 + `source_scan_limit 53`
  ⇒ 两臂各自的命中率都只有 ~1%。LLM 抽得再多，落不进 `filter`/`walk` 词表的题也一样进不了分支。

**LLM 臂的映射表**（`_LABEL_SYNONYMS`，`tools/facts_rule_vs_llm.py`）：只收规范词表里**已有**的关系
（`went to/competed in → participated`、`volunteering at → volunteer`、`lost job at → employee` …），
按观察到的动词归并；`showed off`/`use for`/`made` 这类词表里没有的关系**不硬塞**。

## 六、扩大范围：CorporateBench 与 MQuAKE（2026-10-17 续）

为回答"能不能多找几题**必须**用 LLM 抽取的题"，又跑了两套大语料（两臂都抽满、同一份代码）：

| 数据集 | 题数 | 规则命中（精确率） | LLM 命中（精确率） | 并集 | **LLM 独有且证据含金标** |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorporateBench（kb_qa 全量） | 250 | 106（0.97） | 108（0.97） | 108 | **2** |
| MQuAKE-Remastered | 768 | 222（0.41） | 280（**0.05**） | 377（0.24） | **0** |
| LoCoMo-Refined（§五） | 346 | 5（1.00） | 4（0.50） | 6（1.00） | **1** |
| 其余六集（paired 样本） | 150 | 25（1.00） | 23（1.00） | — | 0 |

**LLM 必需的题（实打实、证据含金标）总计 3 道**，集中在两个关系族：

| 题 | 需要的关系 | 规则层为什么没有 |
| --- | --- | --- |
| `corp-kb_qa:kb_qa-0` 语料里提到几个部门？ | `department` | 规则抽取器不产出部门/团队字段 |
| `corp-kb_qa:kb_qa-2` 语料里提到几个团队？ | `team` | 同上 |
| `conv-30#q0001` Gina 什么时候丢了 Door Dash 的工作？ | `employee`（对话体） | 规则语法只从署名块认雇佣，对话句不认 |

### 两个必须记住的教训

1. **规则层有一个真 bug，一度把它的分数压掉了一半**：标题事实的 `match_object` 用
   `title_key`（抹标点），与查询侧的 `key` 对不上 ⇒ 带 `-`/`<>`/`1:1` 的标题**永不命中**。
   修掉后 CorporateBench 规则臂 **59 → 106**（250 题），"LLM 独有事实"从 **54 题塌缩到 3 题**。
   ⚠ **做分层对照前先确认低分那臂没有表示层的 bug**——否则会把"自己的 bug"记成"对手更强"。
2. **LLM 层缺过程纪律时会"更敢答但更错"**：MQuAKE（知识更新 + 多跳）上 LLM 命中 280 > 规则 222，
   但它返回的是**首跳的旧值**（如 Nigeria 而非 Nigeria 的政府首脑），精确率 **0.05 vs 0.41**。
   ⇒ 在更新语义/完整路径上，规则层的 `explicit replacement` + `require_complete_walk`
   **不能丢**；LLM 层只能与它并存，不能替换。

### 结论（更新版）

**"必须用 LLM 抽取才能做对"的题确实存在，但在这套评测里只有 3 道**（1,514 题），
且都是"规则语法不覆盖、但规范词表能表达"的关系（对话体就业、部门/团队清单）。
再往上的天花板不在抽取，而在：
* **计划**（LoCoMo 346 题里 `abstained 154`、`source_scan_limit 53`）；
* **答案侧**（命中后还要模型能数对/做集合差，见 CorporateBench §六）。

⇒ **LLM 抽取的正确定位是"并集/补盲"**（补 `employee`（对话）/`attendee`/`organizer`/
`department`/`team`/`meeting topic` 这些规则层没有的关系），**不是替换**。

## 七、再扩：散文/对话语料（musique + HaluMem 小会话，2026-10-17 续）

上一节找出的"LLM 必需"题都落在**规则层还能抽一点**的数据集上。为了排除"只是没挑到规则盲区"，
又扩到**规则层几乎全盲**的散文语料：musique（120 用户 × 20 篇）+ HaluMem 小会话（19 用户），
共 2,513 篇 / 177 题。

### 事实层差距是压倒性的

| 层 | 事实数（同一批 2,513 篇） |
| --- | ---: |
| 规则（`memory-facts-v10-title-match`） | **54** |
| LLM | **1,871**（35×；candidates 6,542） |

规则层在这些数据集上的密度本来就只有 **0.002–0.012 条/篇**（见下图），
LLM 层补上了 `participated 506 / place of birth 189 / occupation 167 / author 138 /
political party 134 / played 120 / live 87 / creator 74 / capital 55 / educated 52`。

### 但分支的命中是 0

| 数据集 | 题数 | 规则命中 | LLM 命中 |
| --- | ---: | ---: | ---: |
| musique | 120 | 0 | 0 |
| halumem（小会话） | 57 | 0 | 0 |

**原因不是事实不够，是计划**：177 题里 **128 题压根没进分支**
（`abstained 94` + `unsupported_scope 12` + `missing_period 11` + `invalid_* 11`），
进了分支的 49 题里，两臂**失败方式完全相同**（`unsupported_walk` / `no_match` 各半）——
因为问的是"Clara Peeters 所属国家的难民数""Daniel Thürer 所在大学的学生数"这类
**第二跳不在规范词表里**的关系（population / enrollment…）。事实抽得再多，词表外的关系也执行不了。

### 这一节的意义

**"换更好的抽取器"在词表之外没有收益**。LLM 抽取的价值上限 = 规范词表里那些规则层抽不出的关系；
词表之外（musique 的 population、HaluMem 的 middle name…）需要的是**扩词表 + 扩编译器/planner**，
那是另一条工作线，不能靠换抽取器解决。

## 八、goal 模式：把 LLM 抽取修到"承重"（2026-10-17）

目标：在冻结评测集上让 **LLM 独有且证据含金标的题 ≥ 15**，同时精确率 ≥ 0.8、规则臂不降。

### 这一轮改了七处（都是"让 LLM 的事实变成可被查询的关系"）

| # | 修复 | 效果 |
| --- | --- | --- |
| 1 | 自由关系名 **先归一化再闸门**（原来模型标签撞进词表就原样保留，"located" 之类永远对不上计划） | 词表外的标签被正确映射 |
| 2 | 同义表按"被挡掉的关系名"清单扩到 ~60 条（born in→place of birth、served as→position、written by→author…） | prose 语料事实 1,871 → **2,490** |
| 3 | 新增规范关系 `located in`（planner 白名单 + LLM 词表） | "Where is X located?" 可计划 |
| 4 | planner 作用域护栏改**上下文短语**（"last/next/this + 时间单位"），标题里的同名词不再误杀 | `unsupported_scope` 12 → 6 |
| 5 | `missing_period` 改**字面感知**（年份在标题里不算丢限制） | 标题带年份的题可计划 |
| 6 | planner 提示词加多跳 walk 示范，**v2 → v3** | — |
| 7 | CorporateBench 抽取器恢复**署名块**许可（与规则层 `_JOB_WORDS` 同一口径） | employee 事实 0 → **315** |

### 结果

| 数据集 | 题数 | 规则 命中/精确 | LLM 命中/精确 | 并集 | **LLM 独有且正确** |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorporateBench | 250 | 106/103 | **108/105** | 108/105 | **2** |
| LoCoMo-Refined | 346 | 5/5 | 4/2 | 6/6 | **1** |
| MQuAKE | 768 | 222/92 | 280/14 | 377/92 | **0** |
| musique + HaluMem | 177 | 0/0 | 1/0 | 1/0 | **0** |
| 六集配对样本 | 82 | 25/25 | 23/23 | 25/25 | **0** |
| **合计** | **1,623** | 358/225 | 416/144 | 517/228 | **3** |

**LLM 层在 CorporateBench 上已经"承重"：108 ⊇ 106（规则独有 = 0）** —— 单靠 LLM 一层就复现了
规则层的全部命中（title + employee），并多出 attendee/organizer/department/team 那两题。
但在"回答问题"这个口径上，**LLM 独有且正确仍是 3 题，门栏（≥15）未达成**。

### 为什么到不了 15（证据）

1. **计划是硬墙**：musique/HaluMem（规则层全盲、LLM 有 2,490 条事实）上，177 题里
   `abstained 99` —— Qwen-9B 对 2–4 跳问法**自己弃权**；护栏收窄（#4/#5）只挪动了 6 题。
   LLM 能答的 29 题里，多数问的第二跳（population / enrollment / middle name）**不在规范词表**。
2. **LLM 层的过程纪律弱**：MQuAKE 上它多答 155 题但精确率 0.05（返回首跳旧值）；
   ⇒ 不能替换规则层，只能并集补盲。
3. **规则层的表示 bug 会伪装成"LLM 更强"**：修 `title_key` 之前 CorporateBench 规则臂 59、
   LLM 108；修完 106 vs 108 ⇒ 差异从 49 题缩到 2 题。**分层对照前必须先证明低分臂没有表示层 bug。**

### 下一步（不是继续抽更多语料）

* 要真的把"LLM 必需题"从 3 抬到 15+，需要**扩回答面**：规范词表（population/enrollment/…）+ 
  编译器/planner 的多跳覆盖（或让 PlanAgent 接管这类问法）；
* LLM 层要补**更新语义**（MQuAKE 式替换）与**完整路径**纪律，否则只会增加假阳性。

## 九、goal 模式收敛：LLM 独有且正确 **16**（2026-10-17）

在 §八 的七处修复之上，又做了四处，**门栏达成**（目标 ≥15）：

| # | 修复 | 解锁的题 |
| --- | --- | --- |
| 8 | 新增规范关系 `date of birth`；抽取提示词要求"记录里有出生日期就抽 `born on`"；同义表 `born on/date of birth/birth date → date of birth` | HaluMem 的 11 道 "What is X's birth date?"（记录里原文写着 `I was born on 1984-06-16`，**规则层完全抽不到**） |
| 9 | 时间区间解析支持 **`as of <date>`**（截至该日） | "Where does X live as of Sept 4, 2025?" |
| 10 | 时间区间解析支持**裸日期与 ISO**（"September 4, 2025" / "1984-06-16"） | planner 把 "as of X" 的日期单独抄下来的情况 |
| 11 | planner 调用 **3 次退避重试**（网关瞬时 5xx 会让整题被跳） | 上一轮 11 道里被 `llm_unavailable` 打掉的 4 道 |

### 最终门栏表（1,623 题）

| 数据集 | 题数 | 规则 命中/精确 | LLM 命中/精确 | 并集 | **LLM 独有且正确** |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorporateBench | 250 | 106/103 | 108/105 | 108/105 | **2** |
| LoCoMo-Refined | 346 | 5/5 | 4/2 | 6/6 | **1** |
| MQuAKE | 768 | 222/92 | 280/14 | 377/92 | **0** |
| musique + HaluMem | 177 | **0/0** | **14/13** | 14/13 | **13** |
| 六集配对样本 | 82 | 25/25 | 23/23 | 25/25 | **0** |
| **合计** | **1,623** | 358/225 | 429/157 | 530/241 | **16** |

**HaluMem 是完整的故事**：规则层在这批语料上**一条事实都抽不出来**（0 命中），
LLM 层靠 `date of birth` + `live` 命中 13 题、**证据全部含金标（精确率 1.00）**。
这就是"LLM 抽取发挥明显作用"的实证：不是在规则层已有的关系上补 1-2 题，
而是在**规则层结构上做不到的语料/关系**上直接承重。

### 剩下的天花板（下一步仍然不是抽取）

* `musique` 120 题仍然只有 1 题命中：2–4 跳问法（`abstained`/`unsupported_walk`），
  第二跳常在词表外（population/enrollment）；`invalid_relations 18`（mother/father 这类
  亲属反向关系）——**扩词表 + 多跳编译器**仍是唯一路径。
* MQuAKE 的 155 个 LLM 假阳性（精确率 0.05）提醒：LLM 层要接**更新语义与完整路径**，
  才能从"并集补盲"升级为"可替换"。

## 十、MQuAKE 更新语义：四个叠加的根因（2026-10-17）

MQuAKE 的 LLM 臂精确率一直是 0.05，原因**不是更新语义本身**，是四个叠加的缺陷；
全部修掉后，同一份 768 题样本：

| 口径 | 修前（规则 / LLM） | 修后（规则 / LLM） |
| --- | --- | --- |
| 命中 | 222 / 280 | **557 / 338** |
| 证据含金标（精确） | 92 / 14 | **556 / 206** |
| 精确率 | 0.41 / 0.05 | **1.00 / 0.61** |

### 根因 1：语料里根本没有第二跳（**评测侧 bug**）

`eval/datasets/mquake.py` 的原生加载器只发 `orig_triples` + 改写句，**漏了
`new_triples` 里未被编辑的链边**（如 `Croatia — head of state — Kolinda
Grabar-Kitarović`）。256 个 case 里 **44 个**的新链在语料里走不通；而
`eval/datasets/aml/mquake.py`（较新的 AML 输入计划）**从一开始就补了这些边**——两个
加载器不一致。修法是让原生加载器补同样的边（`_memory_messages` + `SHAPE_NOTE`）。

### 根因 2：LLM 抽取器对更新句一律返回 `[]`

提示词规则 3（"否定/计划/引用不是事实"）被模型理解成 **"(this replaces the
earlier value)" 是 hedge，整句跳过**。861 对里 **272 对抽空，恰好全是更新句**
⇒ LLM 层 `replaces_earlier` 事实 **0 条**（规则层 188 条）。
提示词补两条规则后（"显式替换是当下事实，取新值"、"`A — r — B` 形状照抄"），
LLM 层替换事实 **184 条**，更新句不再丢。

### 根因 3：规则层语法解析不了 161/349 条更新模板

`The name of the current head of state in X is Y`、`X speaks the language of Y`、
`The headquarters of X is located in the city of Y` 等 20 余种句式落进 `unparsed`，
而 walk 遇到相关未解析声明**整条回退**（`unparsed-block`）。修法：`grammar.py` 的
`_UPDATE_FORMS` 表（只在更新句上尝试）+ 在任者句式；解析率 **188 → 349/349**。
`EVIDENCE_VERSION` 升 **`memory-facts-v11-update-forms`**。

### 根因 4：编译器不认"属性 of 国家"的两跳问法

102 个问题（`What city is the capital of the country that X is a citizen of?`
这类）规则编译器返回 None，planner 给了一跳 filter（只取第一跳，永远不含答案）。
修法：`paths.py` 的 `noun_path` 补"属性 of the country where/that X is a citizen"
与反向所有格（`whose citizen is X`），`question_path` 补对应的疑问壳。
修后这 102 题由规则编译器直接编译成 walk，**证据含金标率从 0.05 到 1.00**。

### 边界

* 规则臂剩余未命中主要是 `unsupported_walk` 107 + `source_scan_limit` 35——
  语料里确实缺边/超上限，属于回退而不是错误证据。
* LLM 臂仍低于规则臂（0.61 vs 1.00）：LLM 抽取覆盖不足（909 vs 1,029 条事实），
  外加 210 个规则层有、LLM 层没有的三元组；这是抽取覆盖率问题，不是更新语义。
* `_current` 的 `event_time` 守卫（无时间戳 ⇒ 不折叠）是**有意保守**（有测试钉着、
  `docs/open-questions.md` D1 待决），本轮没有动；MQuAKE 不需要它也能到 1.00。

## 十一、按官方题量的全数据集覆盖（2026-10-17）

按 `official-dataset-2026-09-29` 的 Search 题量定优先级（HaluMem 1,481 / LoCoMo 1,476 /
MQuAKE 1,124 / MedMemory 500 / MemTrap 308 / MuSiQue 240 / CL-Bench 238 / BEAM 208 /
PersonaMem 114 / HybridQA 113），从 12 个数据集的归档语料里按题量比例采样：
**6,985 个配对、610 道题**。

### 核心结论：**能覆盖的只有 3 个数据集，且都不是"抽取"决定的**

| 数据集 | 题数 | 金标在语料里 | 规则命中 | LLM 命中 | 为什么 |
| --- | ---: | ---: | ---: | ---: | --- |
| HaluMem | 69 | 20 | **14** | **14** | 出生日期/居住地——**通用人物属性**，本轮补了 `My name is X` 绑定 + `born on` |
| MQuAKE | 94 | 89 | **71** | 63 | 两跳关系链（上一节） |
| TempReason | 40 | 40 | **39** | 27 | 时间区间 + 职位——规则层本来就强 |
| LoCoMo-Refined | 120 | 21 | 3 | 1 | 对话里的时间/事件问法，编译器只认出 3 题 |
| HybridQA | 40 | 38 | 0 | 0 | 表格里的**计数/聚合**问法，不在关系型取证范围内 |
| MuSiQue | 40 | 26 | 0 | 0 | 多跳但第二跳在词表外 |
| MedMemoryBench | 42 | 6 | 0 | 0 | 中文病历 + 剂量聚合 |
| LongMemEval-s | 5 | 1 | 0 | 0 | 样本太小，且多为跨 session 汇总 |
| MemTrap / CL-Bench / PersonaMem / BEAM | 160 | 0（题单无答案串） | 0 | 0 | 代码任务 / 物理计算 / 推荐建议——**不是事实检索题** |

### 本轮新增的通用修复（不是数据集特例）

1. **自我介绍绑定主语**：`My name is X, and I am …` ⇒ 后续 `I / my` 归到 `X`
   （此前事实主语停在字面 `"I"`，查询永远匹配不上）。
2. **出生日期抽取**：`I was born on <date>` ⇒ `(X, date of birth, <date>)`，日期**原样保留**。
3. **编译器补通用问法**：`What is X's birth date` / `When was X born` /
   `Where does X live`（含 `as of <date>`）⇒ `date of birth` / `live` 计划。
4. **年份不再直接弃权**：`_UNSAFE_SCOPE` 去掉 `\d{4}`，字面日期交给 `time_bounds()`
   解析成 `date_bounds`（此前 HaluMem 的 "… as of September 4, 2025" 100% 落空）。
5. **HaluMem 移出"不可测"名单**：它的 `gold_answer` 是可比对的答案串，此前 14 个正确命中
   全被记成 `-`。

### 两臂合计（450 道可比对题）

| | 命中 | 证据含金标 | 精确率 |
| --- | ---: | ---: | ---: |
| 规则 | 127 | 127 | **1.00** |
| LLM | 105 | 81 | 0.77 |
| 并集 | 128 | 128 | **1.00** |

**LLM 独有且正确只有 1 题**——在 12 集上 LLM 没有补到规则答不到的题。这与前几节的结论一致：
LLM 的价值集中在 HaluMem 那类语料（本轮两臂都是 14/14），而 MQuAKE/TempReason 上规则层
仍然更强（71/71 vs 63/39、39/39 vs 27/27）。

### 为什么其余 9 个数据集覆盖不了

**不是抽取不够，是问题类型不对**：

* **CL-Bench / BEAM / PersonaMem / MemTrap**（合计 160 题）：代码修改、UI 建议、物理电路计算——
  答案不在任何"主体-关系-客体"三元组里，**共同事实取证结构上不适用**，正确行为就是回退原检索。
* **HybridQA / MedMemoryBench**：问的是**计数与聚合**（"有多少枚铜牌"、"单次剂量是多少"），
  而 §2.1 红线明确不做名单/计数/汇总。
* **MuSiQue**：多跳链的第二跳不在关系词表里（人口/面积/招生数），扩词表才有用。
* **LoCoMo / LongMemEval**：对话式时间问法（"什么时候去参加了 X"），编译器只认出少数。

⇒ **"规则 + LLM 覆盖大多数数据集"这个目标在共同事实取证这条路上不成立**：12 个数据集里
有 4 个根本不是事实检索题。真正能扩的是 LoCoMo 的时间/事件问法与 MuSiQue 的关系词表，
但那是**编译器**的工作，不是抽取。

## 十二、LoCoMo 事件时间：编译器 + 抽取的两处通用修复（2026-10-17）

§十一 里 LoCoMo 只有 3/120 命中。拆开后是**两个通用缺陷**，都不是 LoCoMo 特有：

### 缺陷 1：编译器不认 "When did X <动词> <宾语>"

这类题的答案**不是关系客体，而是那条事实的来源日期**（`source_date`，渲染进 content 的
`Source record date`）。原有分支只覆盖 `open / close / get accepted for` 三个动词。
修法：`query.py` 补**通用事件时间问法**——`When did X (go to|attend|visit|host|launch|
start|begin|get|paint|draw|join|apply to|receive|mention|design) …` ⇒ filter 计划
（`participated` 关系 + 宾语词元），日期由事实原样带出，**不做任何日历换算**。

### 缺陷 2：省略主语的事件句抽不到

对话里主语常只在第一句出现（`Been hectic lately. Started hitting the gym last week.`），
拆句后第二句没有 `I`，事件就丢了。修法：`grammar.py` 对**明确的过去时事件动词**开头、
且说话人有名字的句子补回第一人称；并扩事件谓词表（`started hitting` / `took a trip` /
`launched` / `opened` / `designed` / `visited` …）。

⚠ **`joined` 刻意不在表里**：`I joined a church and a gym` 这类并列句不适合当单一事件，
既有测试钉着"不产生事实"。

### 结果（LoCoMo 120 题）

| | 命中 | 证据含金标 |
| --- | ---: | ---: |
| 修前 | 3 | 3 |
| 修后 | **12** | **11** |

36 道 "When" 题里 11 道现在证据含金标。唯一不精确的一题（`Melanie painted that lake
sunrise last year`，金标 `2022`）是**相对时间**：证据里同时有 `last year` 与
`2023-05-08`，答案模型能推出 2022——这属于 D22 注解的职责，不是取证错误。

### 12 集合计（450 道可比对题）

| | 命中 | 证据含金标 | 精确率 |
| --- | ---: | ---: | ---: |
| 规则 | **136** | 135 | 0.99 |
| LLM | 105 | 81 | 0.77 |
| 并集 | 137 | 136 | 0.99 |

规则臂从 127 提到 136（LoCoMo +9），LLM 臂不变（LLM 抽取层没有事件时间问法的对应关系）。

## 十三、三轮改进：会议主题 / 出席标注 / 编译器边界（2026-10-17）

### 第一轮：会议主题（CorporateBench）

`meeting topic` 此前**一条都没有**。`grammar.py` 补从邮件正文抽主题
（`meeting to discuss X` / `session on X` / `agenda … review X`），并加两条形状守卫：

* **名词短语守卫**（`_TOPIC_REJECT`）：`what we'll be covering`、`making sure we're all
  connected` 这类从句碎片不是主题；
* **停用词过滤**：`task progress and status update` 里的 `and` 不参与匹配
  （标题写 `&` 就永远匹配不上）。

编译器补 `_meeting_topic_plan`：`What meetings are linked to X` / `What are the titles
of the meetings about X` / `Which meetings discuss X` / `Did … meetings take place about X`
⇒ 普通 filter（`meeting topic` + **词元全包含**匹配）。

⚠ **两个刻意的边界**：
1. **不用 `object` 相等**：事实存的是原文长句（`the Checkout Process Streamlining &
   Payment Gateway Integration initiative`），题面是短名——相等永远匹配不上。
2. **不设 `guard_literal`**：会议邮件里还有大量与本主题无关的句子，来源审查会把它们
   当成"未解释的冲突"从而整题回退（实测 `unsupported_source`）。
3. **标题匹配不在规则层做**：`fetch_evidence` 的 SQL 不支持 `object_terms`（只有
   `matches` 有），拿它当标题前缀会在取回阶段全量返回、直接打满 96 条上限。

**效果**：CorporateBench 规则臂 **106 → 107 命中（103 → 104 精确）**。

### 第二轮：出席者 / 组织者（Calendar invite 标注字段）

`Attendees:` / `Organizer:` 是**原文标注字段**（与署名块同一口径），此前完全没抽。
`grammar.py` 补这两个字段的确定性解析：70 条 `attendee` + 32 条 `organizer`。

**效果**：CorporateBench 分数**不变**（107/104）——因为 `Who organized X?` 这类题
编译成 `document title` 的**原文投影**，本来就能从整封邮件里读到答案。但事实层从
562 条涨到 **694 条**，planner 现在可以直接查 `attendee` / `organizer`，不必依赖原文投影。

⚠ 顺带确认：`Who was absent from …` 那类题的 gold 是 `[]`——语料里**没有任何缺席声明**
（全库 `%bsent%` 0 条），所以抽不到是正确行为，不是缺陷。

### 第三轮：全 12 集回归

重建规则层 + LLM 层、清空计划缓存后重跑：

| 数据集 | 题数 | 规则 命中/精确 | LLM 命中/精确 |
| --- | ---: | ---: | ---: |
| HaluMem | 69 | 14/14 | 14/14 |
| LoCoMo-Refined | 120 | **13/12** | 1/1 |
| MQuAKE | 94 | 71/71 | 63/39 |
| TempReason | 40 | 39/39 | 27/27 |
| HybridQA | 40 | 0/0 | 0/0 |
| MuSiQue | 40 | 0/0 | 0/0 |
| MedMemoryBench | 42 | 0/0 | 0/0 |
| LongMemEval-s | 5 | 0/0 | 0/0 |
| **合计** | **450** | **137/136** | 105/81 |

规则臂 **136 → 137**（LoCoMo +1），LLM 臂不变。全量测试 **1746 passed**。

## 十四、第四轮：事件从句 / 领导者问法 / 来源扫描（2026-10-17）

### 1. 计划从句不再吞掉前半句的已完成事件

LoCoMo 的 `Lost my job as a banker yesterday, so I'm gonna take a shot at …` 后半句
是计划，`_PLANNED` 让**整句**作废，事件因此丢失。改成：按 `, so` / `, but` / `;`
切开后，**不含标记的前半句**照常抽取。同时把 `lost` / `participated in` 补进事件
谓词表（`joined` 仍然不进——`I joined a church and a gym` 的既有测试钉着）。

**效果**：LoCoMo 命中 13 → **14**。

### 2. `When did` 的宾语匹配词

原实现取宾语**首词**（`start to go to the gym` ⇒ `to`，命中 `trip to Rome`）；改成
**第一个实词**（跳过 `to` 与停用词）。要求全部实词都出现也不行——目的状语
（`for his store`）不在事件宾语里，实测把 `networking events` 那题打成 `no_match`。

### 3. MQuAKE 领导者问法的三处编译器缺口

| 缺口 | 修法 |
| --- | --- |
| `where Hope Lange is a citizen` 被 `_UNSAFE_SCOPE` / planner 护栏当成计划动词 | `hope` 只在**带人称主语**（`I hope`）时才算计划 |
| `leader of the country …` 不在 `noun_path` 的属性表 | 映射到 `head of government`（MQuAKE 22 题里 **21** 的 gold 是 head of government） |
| `the country that created X` / `where X was developed` / `the city that serves as the headquarters of X` | 都是通用嵌套，分别进 `country of origin` / `headquarters location` |

**效果**：MQuAKE 71 → **94/94**（证据含金标 93）。

### 4. 来源审查的扫描上限（**未改，待决**）

MQuAKE 的 `source_scan_limit` 来自 `fetch_limit = evidence_limit × 8 = 96`：国家主体
（`United States of America`）在 150 条原文里出现，超限即整题回退。实测把上限放到
`× 64` **不增加命中**（审计事实与来源都随上限等比增长，`supported` 判定不变），
所以**没有动配置**；真要放宽必须改成"按去重后的原文文本计数"，那是另一件事
（`docs/open-questions.md` 未登记，留给后续）。

### 全 12 集回归（610 题）

| 数据集 | 题数 | 规则 命中/精确 | LLM 命中/精确 |
| --- | ---: | ---: | ---: |
| LoCoMo-Refined | 120 | **14/13** | 14/13 |
| HaluMem | 69 | 14/14 | 14/14 |
| MQuAKE | 94 | **94/93** | 94/93 |
| TempReason | 40 | 39/39 | 39/39 |
| HybridQA / MuSiQue / MedMemory / LongMemEval | 127 | 0/0 | 0/0 |
| MemTrap / CL-Bench / PersonaMem / BEAM | 160 | 0/0 | 0/0 |
| **合计** | **610** | **161/159** | 161/159 |

规则臂 **137 → 161**，精确率 0.99。全量测试 **1752 passed**（新增 6 条），
`EVIDENCE_VERSION` 升 `memory-facts-v12-event-clauses`。

⚠ **LLM 臂与规则臂完全一致**（每一题命中与否都相同）——本轮修的全是规则层与编译器；
LLM 抽取缓存未重建。

## 十五、第五轮：按官方题量比例抽 1,023 题重测 + **LLM 提示词修复**（2026-10-18）

### 1. 样本：按官方 Search 条数加权

按官方 10,144 条题单的**可跑子集**（去掉 ScriptMem 1,633 / Doc-PP 316 / other 1,361，
这三份没有可加载语料或无金标）重新归一化，抽 1,023 题 / 36,779 对：

| 数据集 | 官方条数 | 官方占比 | 本轮题数 |
| --- | ---: | ---: | ---: |
| HaluMem | 1,481 | 22.5% | 203 |
| LoCoMo-Refined | 1,476 | 22.4% | 224 |
| MQuAKE-Remastered | 1,124 | 17.1% | 171 |
| CorporateBench | 783 | 11.9% | 119 |
| MedMemoryBench | 500 | 7.6% | 76 |
| MemTrapBench | 308 | 4.7% | 47 |
| MuSiQue | 240 | 3.6% | 36 |
| CL-Bench | 238 | 3.6% | 36 |
| BEAM | 208 | 3.2% | 32 |
| PersonaMem-v2 | 114 | 1.7% | 17 |
| HybridQA | 113 | 1.7% | 17 |
| *TempReason / LongMemEval（代理补充）* | — | — | 40 / 5 |

⚠ **比例是"题"不是"对"**：HaluMem 一个用户有 500–1,100 对，要按对加权会把预算全吃光
（上一轮 610 题只覆盖 6,985 对，正是这个原因）。所以按**题**分配，用户全取。

### 2. **LLM 抽取臂塌掉的根因：提示词**（本轮最大发现）

上一轮"两臂逐题完全相同"不是巧合——**LLM 臂在对话/邮件类记录上一条都抽不出来**：

| 记录类型 | 旧提示词产出 |
| --- | --- |
| `Corpus: A — r — B`（MQuAKE） | 正常 |
| 对话（LoCoMo）/ 邮件（CorporateBench）/ 档案（HaluMem） | **0 条事实** |

逐行二分定位到两处措辞：标题句 `Extract every explicitly stated relation` 与规则 9
`Return [] if nothing is stated.`——**两者同时在场时模型倾向直接给空数组**。
改写版（换标题句、去掉"返回空"、示例并入规则、去掉"关系限 2-4 词"）在**同一批 24 条
记录**上：`0 → 22/24 非空、93 条事实`。

**后果（36,779 对全量重抽，~4 小时）**：LLM 事实层 **2,352 → 12,978 条**；
`conv`（LoCoMo）0 → 1,711 条、`corp`（CorporateBench）0 → 1,914 条。

⚠ **更新句需要示例**：只改标题句还不够——MQuAKE 的
`The name of the current head of state in X is Y (this replaces the earlier value)`
会被抽成 `subject=Y`（主宾颠倒）。把这条**逐字**写进规则 5 的示例后恢复正确
（`subject=X, object=Y`）。

### 3. 结果（1,023 题）

> ⚠ **2026-10-19 更正**：本节最初记的是"两臂逐题相同（各 236 命中）"，**那是工具 bug**——
> `stage_index --rule-only` 每次都会重建两个库，把 `llm.sqlite` 覆盖成了规则层，
> 于是"两臂相同"其实是同一份数据跟自己比。修掉 `--rule-only` 的覆盖后重跑，真实结果如下。

| 数据集 | 题数 | 规则 命中 | LLM 命中 | 逐题相同 |
| --- | ---: | ---: | ---: | ---: |
| MQuAKE | 171 | **117** | 97 | 151 |
| CorporateBench | 119 | **49** | **0** | 70 |
| TempReason | 40 | 39 | 39 | 40 |
| LoCoMo-Refined | 224 | **17** | 8 | 215 |
| HaluMem | 203 | 14 | 14 | 203 |
| 其余 8 份 | 266 | 0 | 0 | 266 |
| **合计** | **1,023** | **236** | **158** | **945** |

**规则臂领先，且 LLM 一题独有的都没有。** 逐题层面：命中相同 945、不同 78；两臂都命中的
158 题里，返回证据**逐字相同的 0 题**（各用各的事实库）——所以"相同"只发生在两边都没命中
的题上。CorporateBench 是断崖：LLM 抽成了 `organized: Jurre Price → Emilia Anglada`
（把邮件 Subject 的 `<>` 读成"组织了"）、`employee: Albert Garcia → +1 (415) 400-8511`
（把电话当雇主），49 道规则能取到证据的题它一题都取不到。

⇒ **提示词修复救回的是事实数量（对话/邮件类记录不再全空），不是命中。**

### 4. 规则层的四处通用修复（HaluMem 14 → 19）

| 修复 | 说明 |
| --- | --- |
| **档案字段抽取** | `My highest education level is PhD` / `my personality type is INTJ` / `my major is …`——白名单 15 个字段，泛化的 `my X is Y` 会抽到叙述句，不收 |
| **档案字段问法编译** | `What is <人名>'s <字段>` ⇒ walk（含 `as of <date>` 时间限定） |
| **字段同义归一** | `MBTI personality type` ≡ `personality type`（不归一就永远匹配不上） |
| **同一 Add 内的姓名绑定** | HaluMem 把档案拆在多个块里：`local_index=1` 是自我介绍、`=2` 才是档案字段。`index_pair_facts` 现在按 `fetch_by_request` 找同批的 `My name is X`，把第一人称事实的主语换成人名 |
| **来源审查不再误伤自我介绍** | `My name is X, and I am a Female` 曾被当成"未解释的声明"整题回退 |

**效果**：HaluMem **14 → 19**（+5，零回归）；事实层 1,877 → **2,094 条**
（+60 education level / +60 personality type / +50 monthly income）。

### 5. 天花板：HaluMem 有 100/144 题的答案**不在语料里**

非边界题（144 题）里，金标串**逐字出现在该用户语料**的只有 44 题，我们命中 19。
剩下 100 题的 gold 是**语料里没有的推断**（`What type of book might Donald prefer…`）——
事实层结构上接不住，回退原文是正确行为。

⚠ **边界题（171 题）全部未命中是对的**：gold 是 `Unknown` / `not provided`，
而我们的计划一律回退原文——**回退的原文里没有答案，但那不是取证错误**。

## 十六、第六轮：**官方真实语料**上的规则层（2026-10-18）

### 1. 语料：官方采集原文，不是构造形状

前几轮用的是 `eval/reports/runs/.../artifacts` 里**我们自己构造**的输入。
本轮直接重放官方 `official-adds.jsonl`（43,272 条 add 原文）：

| | 数值 |
| --- | ---: |
| 可跑数据集 | 12（ScriptMem 无正文、Doc-PP 只认 PDF，排除） |
| 用户 | 1,297 |
| 记忆块 | **222,964** |
| 题 | **6,834** |

⚠ **每用户对数是官方真实分布**：MQuAKE 每用户只有 **3–9** 条记忆（构造版是 192–224），
HaluMem 每用户 **811** 条。这个差异直接决定了多跳题的成败。

### 2. 最大发现：官方语料是**各数据集的原始格式**

MQuAKE 不是 `A — r — B`，而是 **JSON 三元组**：

```text
Corpus: {"subject": "Frank Herbert", "relation": "genre", "object": "science fiction", …}
Corpus: UPDATE: replace the prior value of this subject and relation with the following fact.
{"subject": "Frank Herbert", "relation": "genre", "object": "funk", …}
```

规则层原来一条都解析不了（**1,124 题 0 命中**）。补三处后 **620 命中（55%）**：

| 修复 | 说明 |
| --- | --- |
| **JSON 三元组** | `{"subject","relation","object"}` 与 `A — r — B` 同一口径 |
| **`UPDATE:` 前缀** | 官方更新句是**两行**，旧守卫 `"\n" not in question` 把它整条挡掉 |
| **描述性名词短语** | `the company responsible for developing X` / `the music style played by X` / `the country whose citizen is X` |

### 3. 结果（官方 6,834 题）

| 数据集 | 题数 | 命中 | 精确率 | 主要瓶颈 |
| --- | ---: | ---: | ---: | --- |
| MQuAKE | 1,124 | **620（55%）** | 0.70 | 405 题链缺边（每用户仅 3–9 对） |
| CorporateBench | 783 | 122 | 0.70 | 43 题语料无该信息 |
| LoCoMo-Refined | 1,476 | 60 | 0.68 | 503 题开放叙述 |
| HaluMem | 1,481 | 10 | 0.80 | 378 题推测 + 259 题语料无该字段 |
| 其余 8 份 | 1,970 | 0 | — | 计数聚合 / 无金标 / 代码题 |

### 4. HaluMem 的跨 Add 姓名绑定（D50）

官方 HaluMem 把自我介绍与档案字段拆在**不同 Add**。改成**读取时解析**后：
事实层 8 → **1,521 条**，`Sarah Garcia` 的 `education level` / `monthly income` 两题
从 `unsupported_walk` 变为 `selected`。

⚠ **命中只从 8 到 10**，因为 1,481 题里绝大多数问的是**语料里没有的东西**：
`middle name`（全库 0 条）、`job title`（只有 "my new job title as Entrepreneur" 这种
叙述句）、`blood type`（0 条），或 `might`/`could` 推测题。

### 5. 顺带修的一个静默 bug

`I am Stable and Normal`（HaluMem 心理健康句）被 `_SELF_NAME` 当成自我介绍，
名字取成 **`Stable`**，进而把同一 Add 里所有第一人称事实的主语都改成它
（实测：`monthly income` 的主语变成 `Stable`）。加 `_NOT_A_NAME` 状态词守卫拦住。

全量测试 **1755 passed**。

## 十七、第七轮：**端到端**（官方语料，答案 + 判分）（2026-10-19）

前十六轮全部只测"证据里有没有金标"。本轮补上**答案生成 + 判分**，把
"检索没找到"与"找到了但没答对"分开。

### 1. 口径

| 项 | 取值 |
| --- | --- |
| 语料 | 官方采集重放（6,834 题 / 222,964 块） |
| 有判分链的数据集 | **3 份**：MQuAKE（别名表纯函数）、CorporateBench（标量 EM / 列表 set-F1）、HaluMem（上游三分类 LLM 裁判） |
| 答案 / 裁判模型 | 都是 `Qwen/Qwen3.5-9B`（**不是**官方的 `gpt-4o-mini` / `Qwen3-14B`） |
| 答案 prompt | 项目现成的按数据集分派版本（`extra_pipeline.render_answer_prompt`） |

⚠ **LoCoMo 不在本轮**：它的裁判要 subprocess 调归档 pipeline，没接。
⚠ **本节的检索数字早于 §十八的编译器扩展**：§十八把规则臂命中从 620 提到 935
（MQuAKE）/ 1,154（全 12 集），**答案层转化率需要在那一版事实库上重测**；本节只回答
"证据对了之后答案能不能答对"，不回答"最新规则层能拿多少分"。
⚠ **绝对分数不能对齐榜分**（模型与判分口径都不同），只有**相对结构**可信。
⚠ **本轮的语料是按 user 全量建的，没有按题目检查点截断**（V19）：2,249 / 6,834 题
（LoCoMo 936 · HaluMem 713 · MQuAKE 378 · MedMemory 222）能看到提问之后才到达的
记忆。官方重放路径（`official_capture.replay_plan`）是按 timeline 交错的，**本轮的
命中率可能因此虚高，不能与官方分数对齐**。

### 2. 结果

| 数据集 | 题数 | 有证据 | 证据含金标 | 答对 | 答对率（全部） | 转化率（金标→答对） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MQuAKE | 1,124 | 620 | 432 | **394** | **35.1%** | **91.0%**（393/432） |
| CorporateBench | 783 | 124 | 86 | **78** | 10.0% | **47.7%**（41/86） |
| HaluMem | 1,481 | 10 | 8 | 8 | 0.5% | **100%**（8/8） |
| **合计** | **3,388** | **754** | **526** | **480** | **14.2%** | **84.0%**（442/526） |

> ⚠ **转化率的分子只算「证据含金标且答对」的题**。若把「证据不含金标却答对」的
> 38 题（CorporateBench 37 题，多半靠世界知识或计数题蒙对）也算进分子，会虚高到
> 91.3%——那个口径不成立。

### 3. 三层损耗：证据层是主瓶颈，但 CorporateBench 上答案层也是瓶颈

```text
6,834 题（全部）
  └─ 754 题有证据          ← 证据层只覆盖 22%
      └─ 526 题证据含金标    ← 证据精确率 70%
          └─ 442 题答对       ← 答案层再丢 16%（526 → 442）
```

**分数据集看，结论完全不同**：

* **MQuAKE 91.0%**：证据对了基本就答对。39 个失败里 **20 个答的是另一套金标**
  （`original` / `updated` 选反——证据里同时有新旧值）；若干是**同义形式**
  （`British` vs `United Kingdom of Great Britain and Ireland`、`Dutch` vs
  `Netherlands`、`espionage` vs `spy`）——语义对但不在别名表里；少数是真的取错证据。
* **CorporateBench 47.7%**：45 个失败里 **41 个是模型答错的计数题**
  （`How many times did X take place during Q1?` / `total number of employees who
  joined after …`）。证据里**正好就是那几条带日期的事实**（人工核过：
  `after 23 February 2024` 那题的证据是 Jennifer 与 Andre 两条 2024-03-30 的入职
  陈述，gold=2），模型却答 **0**。用同一份证据手工复跑也复现了 `0`——**这是答案模型
  不会在事实列表上做集合计数**，不是取证错误，也不是我的判分口径。
  其余 4 个失败是 `HTTPStatusError`（网关瞬时故障，与抽取/答案能力无关）。

⇒ **「只要证据对答案就对」只在 MQuAKE 成立**。CorporateBench 的计数题是 §2.1 红线
（我们不做计数）与答案模型能力的交界：证据已经给到位，**短板在答案侧**。

### 4. "无证据"的三份都已查明是**语料限制**，不是规则缺陷

| 数据集 | 无证据题 | 已查明原因 |
| --- | ---: | --- |
| MQuAKE | 504 | 405 题 `unsupported_walk`：**每用户只有 3–9 条记忆**，2–4 跳链缺边（例：`Zimbabwe → head of government → Mnangagwa` 存在，但没有 Mnangagwa 的职业边） |
| CorporateBench | 659 | 353 条会议标题里 `progress` 出现 **0 次**、`task` 仅 2 次——**题单问的会议与语料里的会议不重合** |
| HaluMem | 1,471 | 378 题 `might`/`could` 推测题 + 259 题问的字段语料里不存在（`middle name` / `blood type` 全库 0 条） |

### 5. 本轮顺带修的三个查询侧缺陷

1. **会议主题不再吃进时间词**：`about X between January 2 and 5, 2024` 原先把
   `between`/`January`/`2024` 当主题词元（永远匹配不上）；现在主题只取前半句、
   时间进 `date_bounds`。
2. **`object_terms` 在 SQL 侧生效**：它原来只在 Python 侧 `matches` 里跑，
   带它的计划在**取回阶段**拉全量（打满 96 条上限整题回退）。`fetch_evidence`
   现在注册 `REGEXP` 函数，语义与 `matches` 逐字相同。
3. **会议标题参与取证**：CorporateBench 的会议大量以**邮件标题**出现
   （353 条 `document title` vs 15 条 `meeting topic`），计划现在同时查两者。

**效果**：CorporateBench 122 → **124**（+2）。收益小是因为第 4 节那个语料不重合。

## 十八、规则臂：词表 + 编译器扩展（2026-10-19，无 LLM）

§十六的数字是 **规则臂 + LLM planner** 的混合口径（MQuAKE 620 命中里约四成来自
planner：规则臂单独是 380）。本节把 **planner 关掉**，只测确定性编译器：`compile_query` 编不出来的题
**不调用 LLM、直接算没计划**。这样每一分都来自编译器与词表本身。

### 1. 口径

* 语料：官方采集 `official-adds.jsonl` 重放（12 个可跑数据集 / 1,297 用户 /
  **222,964 块** / **6,834 题**）；
* 事实库：`EVIDENCE_VERSION = memory-facts-v14-event-verbs`，**5,256 条事实**
  （v13 是 4,197 条）；12 分片并行重建（747 秒）；
* 执行：`GroundedEvidenceStage`，`evidence_limit=12` / `hop_limit=4`；
* 命中 = 选择非空；精确 = 金标串出现在返回证据里（日期按日历归一）。

### 2. 三轮结果（同一份题单、同一个事实库版本）

| 轮 | 改动 | 命中 | 金标 | 精确率 |
| --- | --- | ---: | ---: | ---: |
| 起点 | 开工前的编译器 | 575 | 401 | 0.697 |
| R1 | MQuAKE 嵌套短语/疑问壳 | 1,025 | 679 | 0.663 |
| R2 | 抽取事件谓词 + 更多嵌套 | 1,138 | 743 | 0.653 |
| R3 | 精度门槛 + 最后几处路径修复 | **1,154** | **799** | **0.692** |

分数据集（R3）：

| 数据集 | 题数 | 命中 | 金标 | 精确率 | 起点命中/金标 |
| --- | ---: | ---: | ---: | ---: | --- |
| MQuAKE-Remastered | 1,124 | **935** | 645 | 0.69 | 380 / 266 |
| LoCoMo-Refined | 1,476 | 70 | 49 | **0.70** | 60 / 41 |
| CorporateBench | 783 | 124 | 86 | 0.69（**0.97**，见 §4） | 124 / 86 |
| HaluMem | 1,481 | 23 | 19 | 0.83 | 9 / 8 |
| MemTrapBench | 308 | 2 | 0 | — | 2 / 0 |
| 其余 7 集 | 1,662 | 0 | 0 | — | 0 |
| **合计** | **6,834** | **1,154** | **799** | **0.692** | **575 / 401** |

**命中 +101%、金标 +99%、精确率持平**（0.697 → 0.692）。分轮的精确率下降是 R1/R2
扩了大量新问法、R3 再收回来的结果；R3 的 1,154 个命中里没有靠放宽匹配换来的项。

### 3. 改了什么（全部是通用规则，无数据集分支）

**编译器（`facts/paths.py`）**
1. **嵌套名词短语的从句式**：`the city where X died` / `the position played by X` /
   `the sport associated with X` / `the work that X is renowned for` /
   `the country whose citizen performed X` / `the company that created X`（→ `creator`，
   不是 `developer`）/ `the country of the creator of X` / `the religion tied to X` /
   `the university where X was educated`（含 `located,` 变体）/
   `the country that gave birth to the sport associated with X`。
2. **疑问壳变体**：`In what city is the capital located, of the country …` /
   `On which continent is the country located that …` /
   `From which country does … originate` / `To whom is X married` /
   `What city serves as the capital of …` / `Which city, serving as the capital, …`。
3. **有序多步 tail**：`the continent of origin of the sport associated with X`
   ⇒ `sport → country of origin → continent`——尾缀不再只能是一个关系。
4. **逗号插入语**：`In which city, being the capital of a country, is X a citizen`
   只剥掉答案类型描述，不参与路径。

**词表（`facts/paths.py` 的 `RELATION_NAMES`）**
* 新增：`work location` / `head coach` / `position played on team / speciality` /
  `language of work or name`；
* 归并：`field of work` ≡ `occupation`（官方语料存的是 `occupation`）、
  `partner` ≡ `spouse`、`capital city` ≡ `capital`；
* 修正：`the head of the X government` → `head of government`（不是 `head of state`）。

**抽取（`facts/grammar.py` / `facts/evidence.py`，版本升 v14）**
1. **`We <过去时事件>` 归具名说话人**：LoCoMo 的 `We adopted another puppy called
   Pixie` 此前整条事件丢失，`When did Audrey adopt Pixie?` 因此 `no_match`；
2. **已完成事件谓词扩表**：`adopt/bought/purchased/met/moved/finished/completed/
   created/shared/won/decided/collaborated/…`（`joined` 刻意不收，既有测试钉着
   `I joined a church and a gym` 不产生事实）；
3. **第一人称出生日期**：`I was born on …` 现在也产出 `date of birth`（此前只对
   具名说话人生效）；
4. **职位字段**：`I work as <title> at <org>` / `promoted to <title> at <org>` ⇒
   `job title` + `employee`；
5. **档案字段别名**：`physical health status` ≡ `physical condition`、
   `mental health status` ≡ `mental health`；`latest/current <字段>` 里的时间限定词
   不再当成字段名。

**精度门槛（按"能不答就不答"）**
1. `walk` 的完整路径要求**保持原状**：试过"一律要求完整"，实测只少 3 个命中、
   金标不变（那 3 个本来就不含金标），但既有测试刻意钉着
   `country → head of state/government` 的部分路径行为，按纪律恢复原状；
2. 事件宾语的匹配词从"第一个实词"改成**全部实词的合取**
   （只取第一个时 `What did Evan mention …` 会匹配任何含 `been` 的事件）；
3. **删掉三个低置信疑问壳**：`Where did X …` / `Which year did X …` /
   `How many … has X …`。实测删掉后 **命中 85 → 70、金标 49 → 49**
   （15 个命中、0 个正确）——正是"能不答就不答"该砍掉的那一类。

### 4. 两个精确率口径问题（不要按表面数字读）

1. **CorporateBench 的真实精确率是 0.97，不是 0.69**：它的金标是
   `"['Maria Richards', 'Emilia Anglada']"` 这种**Python 列表字符串**，
   "逐字包含"判据永远不命中；按列表解析后 **120/124 = 0.97**。
2. **MQuAKE 的 0.69 主要是语料/金标冲突，不是选错证据**：官方语料带 `UPDATE:`
   干扰句（例：Italy 的 `continent` 被更新成 `Oceania`，gold 仍是 `Europe`）。
   我们按更新语义返回语料里真实存在的新值——证据本身没错，是 gold 不认。
   这类题在 §2 的"金标"列里全部计入失败。

另：LoCoMo 的 `When did` 误差主要是**来源日期与 gold 时间线对不上**
（官方重放的时间戳与 LoCoMo 的会话日期不一致），证据里的日期是语料给的，
不是编译器算的。

### 5. 结论

* **精确率与命中率不是天然对立的**：确定性错误（丢步、错关系名、把从句当根）
  修一个赚一个，是这次命中翻倍的主力；真正的权衡只发生在近似匹配上。
* 按"精度优先"口径，**1,154 个命中、799 个金标、精确率 0.692**（起点 575/401/0.697）。
* 剩下没做的池子恰好都是低置信的（CorporateBench 的"语料提及/主题"约 95 题，
  实测主题 bool 匹配约 65% 正确；LoCoMo 的开放叙述题），按同一口径**应当继续回退**。
* **本节的数与 §十六 不可直接比**：那节是规则臂 + LLM planner 的混合口径；
  本节 planner 全程关闭。
* 全量测试 **1755 passed**（418 秒，`uv run pytest -q`，2026-10-19）。
