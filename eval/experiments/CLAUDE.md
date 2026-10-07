# eval/experiments/ — 对照实验的 arm 定义与 runner

**PRD**：§13（对照实验）、§17.2（E1–E7；**E2 已随 D15 删除**）

## 要写什么

```text
run.py            通用 runner：给定配置 → 跑一轮 → 落一份报告；native / aml-v1 输入契约 ✅
replay_official.py **官方采集重放** runner：按 timeline 交错、`--per-family N` 保族覆盖 ✅
recipes.py        **冻结的抽样口径**：按输入契约区分原生与 AML 配方的唯一声明处 ✅
arms.py           两臂脚手架的**唯一实现**：冻结快照 / 核对 / 驱动 / 比较  ✅
a0_recency.py         A0 sanity：不检索、按时间倒序返回最近 N 对      ⬜ 未开始（判据：若逼近全系统，§4 核心判断要重写）
t1_timestamp.py       T1 时间戳前缀 带/不带（两臂冻结快照）        ✅ 问题已答（`t1-dated` 0.633）；**干净的两臂 A/B 未跑**
t2_cross_session.py   T2 跨 session 失败归因（含**人工标注产物**）  🟡 汇总半边已就位；标注待人工
a3_rerank.py          A3 Rerank 开/关                            ✅（首组对照见 ../reports/ledger.md）
a4_agent.py           A4a 门控 / A4b always-on                    ⛔ **不在 v1**（**D26**；arm 定义留在 [`../../docs/experiments.md`](../../docs/experiments.md)）
```

> ## ⚠ 跑任何一个数据集，都用 **`--frozen`**
>
> ```bash
> uv run python eval/experiments/run.py --dataset clbench --frozen --base-url ...
> ```
>
> 它按 [`recipes.py`](./recipes.py) 取 `--limit` / `--spread`——**"这个数据集跑多少题"的唯一声明处**。
> **不要手抄 `--limit 3` 这类数字**：各数据集的全量题数差着量级，
> 每个的 `limit` 数的**还不是同一个东西**（对话数 / 题数 / persona 数 / QA 子集 …），
> 而**抄错一个数的后果是分数看起来完全正常、却不可比**。
>
> `tests/test_experiments.py` 把表里每个 `n_questions` 都钉住（逐个加载、断言实际题数），
> **改 k 而忘了改表，测试立刻红**。`--frozen` 与显式 `--limit` / `--spread` **互斥**。
>
> ⚠ **下面那几条 `--spread` 的告诫仍然成立，但不再是手抄的理由**——
> 逐数据集的轴、分组键、覆盖度与"为什么是这个 k"，都在 `recipes.py` 里，本文件不复制。

> ## ⛔ 建基线**串行跑，不要开两条链**（2026-10-02 实测）
>
> 开发网关在 **Cloudflare 后面**，而 CF 的源站时限是 ~125 秒：**源站超过它就回 524**，
> 而**客户端超时调多大都没用**（实测 300 s / 900 s 同样是 524）。
> 一道题要不要超过 125 秒，取决于**prompt 大小 × 网关当时有多忙**：
> 同样一份 medmemorybench（每题 ~54k token），**单链下 20 题全过，双链下当场 524**。
>
> ⇒ **串行**。别被"本机 CPU 只用了 2%"误导——**瓶颈在远端网关，不在本机**。
> 并发不会让总工作量变小，只会**把 524 从"不会发生"变成"随机发生"**。
>
> ⚠ 而且失败是**连锁**的：超大的请求会反复重试（`pipeline_beam.py` 是 6 次 ×120 s），
> 期间把网关占满 ⇒ **同时跑的别的链**会被排到超过 harness 的 30 分钟客户端超时后死掉。
>
> ⚠ **`beam` 的基线本地拿不到**（每题 ~109k token，空闲网关上照样超 125 秒）。
> 详见 [`../reports/ledger.md`](../reports/ledger.md) 的「跑一轮要多久」。

> **加一个新对照 = 加一个 arm 脚本，不用碰脚手架**：定义一条 `arms.Arm` 子类
> （字段 + `overrides()` + `switches()`）、一个 `arms.Spec`（标签、快照头部那句人话、
> CLI 文字），再 `ARMS = (...)`。**`verify()` 的判据从 `overrides()` 现算**，
> 所以"这个臂该改什么"与"怎么核对它"不可能漂移——那正是这条检查要防的东西。

**跑一轮**（runner 打 HTTP，服务得先起着）：

```bash
make serve                    # 另一个终端；TIANXIMEM_PROFILE=local 时用 memories_dev 集合
make eval                     # 自动准备缺失材料，默认 locomo-refined；ARGS='--limit 3 --skip-ingest' 可冒烟
make eval DATASET=longmemeval-s
make eval DATASET=mquake-remastered ARGS='--limit 1 --max-questions 2'   # official-extra 之一
# ⚠ LongMemEval **部分跑必须加 `--spread`**：它的文件**按 `question_type` 分块**
#   （70 个 single-session-user → 62 个 multi-session → …），`--limit 60` 不加它
#   只会拿到**一类**题——而分数看起来完全正常。`--spread` 按比例跨类取（确定性、可复现）。
#   例：ARGS='--limit 60 --spread'
# ⚠ 同样要 `--spread` 的还有：**clbench**（按 context_category）、**mquake**（按 4 份 parquet）、
#   **memtrapbench**（按 6 个场景）、**beam**（按 conversation_seed.category）。
#   **corporatebench 不需要**——它的 `limit` 选的是 QA 子集。
# ⚠ **beam 还有一个 `--limit` 之外的坑**：它的题**按 probing 组排序**（`abstention` 在最前），
#   所以 `--max-questions 2` 取到的是**同两组拒答题**——看着像"跑了两题"，其实只覆盖一类。
#   要小规模冒烟就配 `--limit 1 --max-questions 20`（整格 20 题），别用小 `--max-questions`。
```

### `--add-shape`：本地发的 add 像不像线上

本节的通用形态开关属于 `native`。`--input-contract aml-v1` 由专用适配器包装，
仅 LoCoMo 允许 `alluser`，仅 HaluMem 允许内联时间；按事件检查点续跑，不支持 `--skip-ingest`。
支持范围、CLI 用法与输出指纹统一见
[`../../docs/benchmark-data.md`](../../docs/benchmark-data.md) 的「公开数据的 AML 输入适配」。

`Add` 的 payload **由 AML 造，不由我们造** ⇒ 本地复现的形态决定**分数预不预测得了线上**。
缺省 `official`（线上那个：逐数据集加 `<标签>: ` 前缀、`system` 折成 `user`）；
`native` 是改之前的形状，**只作对照**；`alluser` 是**判分池那一簇**的形态。

⚠ **它的分辨率与切批同级**：换形态 = 换输入。三个取值都写进 run record 的
`data_fingerprint.add_shape`，**不同形态的分数不可互比**。
细节与实测依据在 [`../harness/add_shape.py`](../harness/add_shape.py)。

**独立评测按评分来源分派**：`locomo-refined` / `longmemeval-s` / `clbench` / **`beam`** 走
**归档里的官方 pipeline**；`personamem-v2` 走**本仓的适配器**
（[`../harness/personamem_pipeline.py`](../harness/personamem_pipeline.py)——**Search 片段替代历史，选项与判分复用官方函数**）；
`mquake-remastered` / `memtrapbench` / `corporatebench` / `medmemorybench` / **`tempreason`**
（来自 `official-extra`）**没有官方 AML pipeline**，走我们自写的
[`../harness/extra_pipeline.py`](../harness/extra_pipeline.py)
⇒ **这五家的分数只在仓内前后对比，别与官方分数对齐**。
⚠ `medmemorybench` 例外一半：**判分口径是上游发布的**（`metrics/`，见 [`../../docs/benchmark-data.md`](../../docs/benchmark-data.md)），我们照它实现——但**作答侧**仍是我们自己的 prompt。

`halumem` 的 QA 裁判复用上游三分类 prompt；`musique` 的本地评分为答案别名
归一化后精确匹配及拒答标记匹配。两份的输入范围与未覆盖指标见加载器 docstring。
HaluMem 的 `limit` 数提问检查点，MuSiQue 数题目；都已登记 `--frozen` 配置。

`hybridqa` / `feverous` 走 [`corpusqa_pipeline.py`](../harness/corpusqa_pipeline.py)，
分别复用固定上游的 EM/F1 与标签、证据组函数。⚠ **两份的 `limit` 轴不同**（hybridqa 数题、
feverous 数 claim——见 `recipes.py`），`--spread` 的分组轴和冻结配置由 `recipes.py` 声明。
FEVEROUS 有本地 claim-only 候选生成步骤；它与完整 Wikipedia 检索及 AML 候选池
的范围区别见 [接入报告](../reports/hybridqa-feverous-pipelines-20261006.md)。

⚠ **`beam` 的判分产物与那三份形状不同**：官方 `pipeline_beam.py` 写的是 `llm_judge_score`
（逐条 rubric 三点制的均分，**没有 `is_correct`**）⇒ 二值化口径由 harness 定
（均分 == 1.0 才算对），真分留在 `label`/`judge_response` 里。

> **纯 BM25 检索是本目录的 T2 手段**（见下），**不是一条被评分的 arm**——检索只有混合一种形态，参照点由**混合主路径自身**承担（§13）。

**协议登记在 [`../../docs/experiments.md`](../../docs/experiments.md)**——那个文件回答"这个对照决定什么"，本目录只回答"**怎么跑**"。数字落 [`../reports/`](../reports/)。

---

## 三条执行纪律

### 1. 开关必须只影响它命名的那一件事（§13）

若一个开关的"关"分支顺带改变了别的量，这个对照**就不成立**——**而结果看起来完全正常，只是结论错了**。

**跑之前先在 [`../../docs/config-reference.md`](../../docs/config-reference.md) §2 核对**——**开关的依赖图与"关掉时不得改变什么"一处声明在那里**。对应测试见 [`../../tests/CLAUDE.md`](../../tests/CLAUDE.md)。

### 2. 每个 arm 都要冻结配置

`configs/runs/` 存每次实验的**配置快照**——`docs/experiments.md` 要求记录**配置指纹**。**"当时的 `local.yaml` 大概是这样"在 Step 5 之后就重建不了了。**

### 3. 记录端到端总分 + 各维度子分

**latency / 成本只在明显变差时才追**（§13）。**不看 Recall@K**（§14）。字段清单见 [`../reports/CLAUDE.md`](../reports/CLAUDE.md)。

---

## 验证一个改动：**最短路径**（2026-10-03）

**目标**：手上有一个改动（换渲染 / 调打包 / 动精排），想知道**它值不值**。

```bash
make serve                                    # 另一终端，服务得先起着
make baseline DATASET=locomo-refined          # 44 分钟 ⇒ 与 ledger 的「当前缺省口径下的基线」比
```

**三条判读纪律**（都吃过亏，逐条都有实测依据）：

| # | 纪律 | 为什么 |
| --- | --- | --- |
| 1 | **比之前先看 `config_fingerprint.snapshot_hashes.default.yaml`** | 改任何**默认值**都会让一批老数字**静默过期**——屏幕上什么都看不出来（D31 改半径那次就是靠它抓出来的） |
| 2 | **一次只跑一条**（不并发） | 网关在 Cloudflare 后面（源站 ~125 秒 ⇒ **524**）；并发把 524 从"不会发生"变成"随机发生"，而**客户端超时调多大都没用** |
| 3 | **1pt 以内的差读不出来** | 346 题上的噪声底 ~1pt，**而它只有单跑的样本**（要做结论得重复跑） |

**选哪个数据集跑**：数字与状态在 [`../reports/ledger.md`](../reports/ledger.md) 的
「当前缺省口径下的基线」表 + 「本机拿不到基线的数据集」表——**本文件不复制**。
一句话：**locomo 是唯一适合每次改动都跑的哨兵**（44 分钟）；其余按改动涉及的面挑，
`clbench` / `beam` 本机跑不了，`medmemorybench` 靠重试碰运气。

> ⚠ **代理评测只覆盖 LoCoMo + LongMemEval**（§12.4 / P3）：其余数据集的契约与裁判各不相同，
> 它们的分数**只能同数据集前后比**，**不能外推**。

## 各 arm 的定义（容易糊的地方）

| 对照 | arm | 注意 |
| --- | --- | --- |
| **A3** | Rerank 开 / 关 | §11 主线的验证。若不值，把算力挪去别处 |
| **A4** | **A4a 门控** / **A4b always-on** | ⛔ **不在 v1**（**D26**）——两臂的 arm 定义与判据见协议登记表，本目录不重复 |
| **T1** | 时间戳前缀 带 / 不带 | 测出差异时**归因不要默认只来自一条机制**（§11.3 的两条独立规则） |
| **T2** | 跨 session 失败归因 | **人工标注**，见下 |

### A4 的两臂定义在协议登记表里

> **A4a / A4b 的 arm 定义、为什么两个都要、以及失败判据一处声明在
> [`../../docs/experiments.md`](../../docs/experiments.md)**（本目录不重复，连速查表也不留）。

> ⚠ 两臂都**不在 v1**（**D26**，2026-09-28）——它们要 agent 真的存在。

---

## T2：本目录里唯一需要**人工产出**的实验

**半天工作量的前置实验**（§13）。

```text
输入：lme_s_cleaned.json 的 133 道 multi-session 题
检索：**纯 BM25**
动作：**人工**把失败样本分三类
  (i)   没召回
  (ii)  召回了但被 top-100 截断
  (iii) 在里面但排序靠后
```

### 两条硬性要求

1. **12 道拒答题必须单独拎出来。** LongMemEval 的拒答题是**横切标记，不是第 7 类**——`question_id` 以 `_abs` 结尾的共 **30 道**，其中 multi-session 类占 **12 道**。**它们的行为与其他题不同**（§12.3 第 6 条）。
2. **标注产物要提交进 git。** 它不是运行时产物，**不在 `var/`、也不在 `../reports/` 的 gitignore 范围内**（后者只忽略原始 JSON/JSONL/CSV 结果）。133 行的人工标注是**一次性劳动**，丢了要重做。

### 它决定什么（§13 / 附录 A）

| T2 结果 | 决定 |
| --- | --- |
| 主因是"**措辞不同导致漏召**" | **别名归并值得做**（附录 A 的实体层） |
| 主因是"**召回但被截断 / 排序靠后**" | **实体层解决的不是本项目的瓶颈**——v1 不做 |

**Step 5 后要重跑 T2**（§12.1 R1 对冲 2）——它是**对外部模型依赖最小**的一组（纯 BM25 检索 + 人工判读，**不调用任何模型**），先用它确认切换没引入系统性偏移，**再去信其他实验**。

---

## 一条提醒：B1 不在这里

**B1（ReFind 原版）的 runner 在 [`../baselines/`](../baselines/) 的 `refind/`**（Vendor 代码已就位）——它是"另一个服务"，不是本系统的一个 arm。


## 数据准备（D35）

`run.py` 在发请求前准备所选数据集；`replay_official.py` 只准备所选题目家族的
裁判源码，不下载公开语料替代采集原文。两者的 `--offline` 均禁止下载。
准备与校验失败要在付费 Add/Search 前退出，不能等到首题裁判时才发现依赖缺失。
