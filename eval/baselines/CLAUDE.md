# eval/baselines/ — 基线

**PRD**：§13（对照实验）、§1（从零搭建）

## 要写什么

```text
recency_only/         A0 —— 不检索，直接按时间倒序返回最近 N 对（⬜ **未实现**，成本约等于零）
refind/               B1 —— ReFind 原版（MIT，**Vendor，不改动**）
memmachine/           MemMachine 参考源码（Apache-2.0，固定版本，**只研读，不改动**）
invmem-candidate/     InvMem 的**候选映射**（无 LICENSE，**只研读**，不是基线）
```

**出处、commit SHA 与逐项对照见 [`../../docs/reference-implementations.md`](../../docs/reference-implementations.md)。** 各克隆目录分别 **gitignored**（[`.gitignore`](../../.gitignore) 的"参考实现"一节）——上游副本不入库，来源与学习笔记入库。

> **裸 BM25 不是这里的一条基线**——检索只有混合一种形态，参照点由**混合主路径自身**承担（§13）。
> **纯 BM25 检索的代码仍然存在**，但身份是 [T2 的检索手段](../experiments/)与 [Checker 判据的一路](../../src/tianximem/retrieve/CLAUDE.md)。

---

## A0 — recency-only sanity check（⬜ **未实现**，成本约等于零）

把**主路径**换成"**不检索，直接按时间倒序返回最近 N 对**"。

> ⚠ 目录里还没有 `recency_only/`；根 `CLAUDE.md` 的状态表把这一个 arm 记在
> [`../experiments/`](../experiments/) 的待写清单下——**实现时把落点收敛到一处**。

> **如果它逼近全系统，说明 §4 那条核心判断要重写。**（§13）

**做了不亏**——这是整个实验计划里最便宜、可能推翻最多的一条。

---

## B1 — ReFind 原版

| 属性 | 值 |
| --- | --- |
| 许可 | **MIT**（三个参考系统里唯一有论文且 MIT 许可的，§1） |
| 论文 | arXiv:2608.12888 |
| 代码 | `github.com/imlrz/ReFind` |

### ✅ 工作量**已估**（2026-09-26）：**不需要"包一层服务"**，因为它本身就是兼容服务

> **不需要给它包一层 Add/Search 服务**：[`refind/app/main.py`](./refind/app/main.py)
> 本身就是一个 **AML 兼容的 Add/Search 服务**（`/add` `/search` 短别名 + `/v1/memories/*`，
> `description="Agent Memory Leaderboard-compatible Add/Search service."`），
> 请求/响应形状与我们的 driver **逐字段对得上** ⇒ **包装量 ≈ 0**。怎么跑：

```bash
# 1) 独立 venv（钉死它声明的版本；**放在 vendor 目录之外**，vendor 保持原样）
uv venv /tmp/refind-venv --python 3.12
uv pip install --python /tmp/refind-venv/bin/python -r eval/baselines/refind/requirements.txt

# 2) 起它（LLM 用我们网关的同一个模型——同期同模型才算对照）
cd eval/baselines/refind
LLM_API_KEY=$AML_API_KEY LLM_BASE_URL=$AML_BASE_URL LLM_MODEL=$AML_MODEL \
RETRIEVAL_MODE=agent DATABASE_PATH=/tmp/refind.sqlite3 \
/tmp/refind-venv/bin/python -m uvicorn app.main:app --port 8001

# 3) 我们的 harness 直接打它（**不 import 它的代码**，隔离边界不变）
uv run --env-file .env python -m eval.experiments.run --dataset locomo-refined --limit 3 \
  --base-url http://127.0.0.1:8001 --run-id refind-3conv \
  --configs-dir configs/runs/refind --profile local
```

> **成本在别处**：它是 **agentic**（`AGENT_MAX_ITERATIONS=4`、`SEARCH_TOP_K=5`，都未改），
> 实测**每 query ≈ 6 秒**（本地网关）⇒ 346 题 ≈ **35 分钟检索 + 约 35 分钟判分**。
> 它的 `Add` 按 `request_id + payload` **精确幂等**（重复投喂返回 200 no-op），重跑安全。
>
> ⚠ **模型口径**：用我们网关的模型（与主系统同期同模型）；它自报的 58.2 / 93.2
> **是它自己 harness 的数，不作对照**（§13）。数字落 [`../reports/ledger.md`](../reports/ledger.md)。

### ⚠ 必须在我们自己的 harness 里重跑（§13）

**ReFind 公开的 58.2 / 93.2 是它自己的 harness、可能不同的问题子集和 prompt 下得到的。直接抄来当基线，等于拿它的 harness 和我们的比。**

### 隔离边界（两条，别破）

1. **`refind/` 只能被 [`../harness/`](../harness/) 当"另一个 Add/Search 服务"来调**——**任何 `src/tianximem/` 里的代码都不得 import 它**。§1 明确"本项目从零搭建，不复用任何既有代码"；B1 是**外部基线**，不是代码来源。
2. **Vendor 进来时保持原样**，不修改。改了就不再是 B1 了——**而"我们赢了吗"这个问题的答案依赖于它没被改过**。

**⚠ 一条归因纪律**：ReFind 的消融数据出自 **GPT-5-mini backbone 的 matched 子集**——**它证明的是"检索后端之间 BM25 不输"，不能直接推及 §9 的 `gpt-4o-mini` 场景；引用时别把两者混在一起**（§4 第 3 条）。数字与完整的纪律表见 [`../../docs/experiments.md`](../../docs/experiments.md) 的「基线数字的引用纪律」——**本文件不复制那张表**。

> **⚠ 这条数据无法在本地检验**（D15）——**若日后分数不及预期，第一个该复检的就是它**。

---

## MemMachine — 源码参考

项目负责人于 2026-10-08 明确限定为取得源码，供后续学习与改进 TianXiMem。
当前只研读，不安装依赖、不新增运行适配或正式评测臂。
上游保持原样，主系统 `src/tianximem/` 不得 import 参考源码（PRD §1）。
固定来源在 [参考实现清单](../../docs/reference-implementations.md)；源码获取与研读顺序
见 [MemMachine 学习入口](../../docs/memmachine-reference.md)。
摘要、LLM 画像抽取与 retrieval agent 的设计仍须按本仓 v2 边界评估。

## InvMem（榜上 45.06）：**只读，不是基线**

| 事实 | 出处 |
| --- | --- |
| **不是论文，也没有第一方文档。** 它是榜单上的**显示名** | PRD §1 |
| 其对应的 repo **全库无 LICENSE**，且代码里**从不出现 "InvMem" 一词**——**名称到代码的映射是第三方推断** | PRD 附录 B |
| ⇒ **不可作为参考实现依赖**（§1）；它的分数只能作**目标参照**，不能作**本地对照** | PRD §1、§13 |

**2026-09-25 变更**：候选仓库已克隆到 `invmem-candidate/`，**唯一的用途是读代码**（对照笔记见 [`../../docs/reference-implementations.md`](../../docs/reference-implementations.md)）。

> **它不因此变成一条基线。** 上表四条事实一条都没变——**无 LICENSE ⇒ 不能分发、不能声称、不能依赖**；从它读出来的结论要写成"某候选实现"，**不能写成"InvMem 的做法"**（PRD §1 的"不可作为参考实现依赖"仍然有效）。
>
> 也**不要因为它进了这个目录就给它包 Add/Search 服务**——B1 的隔离边界（见上）是给 MIT 许可、映射确凿的 `refind/` 定的。

### ✅ 2026-09-26：**用户明确授权跑它作参考**（上面那条规则没有失效，是**被显式覆盖一次**）

**决策人：项目负责人**（原话："请你继续跑 InvMem 吧，毕竟也是作为一个参考"）。
上面四条事实**一条都没变**——它**仍然不是基线**、**仍然不可声称**：台账里它的身份照旧是
"**候选映射仓库**（第三方推断的映射）"，不是"InvMem"。**引用它时要照旧写"某候选实现"。**

**怎么跑的**（[`../reports/ledger.md`](../reports/ledger.md) 有数字）：

```bash
# 1) 独立 venv（依赖重：torch CPU + sentence-transformers + faiss）
uv venv /tmp/invmem-venv --python 3.12
uv pip install --python /tmp/invmem-venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/invmem-venv/bin/python -r eval/baselines/invmem-candidate/requirements.txt

# 2) 起它 —— **用我们的启动器**（见下），vendor 代码一行不改
mkdir -p /tmp/invmem-qwen
NO_PROXY=127.0.0.1,localhost \
MEMORY_DB_PATH=/tmp/invmem-qwen/memory.db ALLOW_UNAUTHENTICATED=true PORT=8002 \
/tmp/invmem-venv/bin/python eval/baselines/serve_invmem_qwen.py

# 3) harness 打它
uv run --env-file .env python -m eval.experiments.run --dataset locomo-refined --limit 3 \
  --base-url http://127.0.0.1:8002 --run-id invmem-qwen-3conv \
  --configs-dir configs/runs/invmem-qwen --profile local
```

**⚠ 关键：跑的是 [`serve_invmem_qwen.py`](./serve_invmem_qwen.py)，不是它的 `uvicorn`。**
那个启动器在进程内只换掉 `build_embedder` **一个函数**，把 embedding 换成我们的
`Qwen3-Embedding-8B`（用户要求：它自带的 `bge-small-en-v1.5` 只有 33M，直接比会把
"embedding 强弱"混进"管线设计"）——**vendor 依旧保持原样**。
⇒ 台账里那一行的口径是"**它的管线 + 我们的 embedding**"。

> ⚠ **两条别踩**：① `MEMORY_DB_PATH` **必须指向 /tmp**——它的默认值是仓库内的
> `artifacts/memory.db`，会用运行时文件污染 vendor 目录；② 别在 vendor 目录里敲
> `uv run`——那会按**它的** pyproject 解析环境，往 vendor 里拖一个 `.venv` 并开始下 CUDA 轮子。

---

## 引用数字的纪律

**引用数字的总纪律（"不要相信自报数字"）在 [`../CLAUDE.md`](../CLAUDE.md)；四类数字的读法（ReFind 58.2/93.2、榜单分、ActiveMemoryIndex、自报数字）一处声明在 [`../../docs/experiments.md`](../../docs/experiments.md) 的「基线数字的引用纪律」。本目录只登记真正独有的一行：**

| 数字 | 能否当基线 |
| --- | --- |
| **候选映射仓库**（`invmem-candidate/`）的公开数据回归：0.6.0 在 LoCoMo-Refined **1,382 题、Top K=100**：Evidence Recall **0.9277**；同数据用官方 Answer/Judge 模板 + `gpt-4o-mini`：Judge Accuracy **76.92%**（1063/1382）。0.5.0：**扩窗 vs 无窗口** 54.0% vs 49.5%（200 题分层样本，TopK=90） | ⚠️ **自报 + 公开数据 + 自建流程**，非平台分（它自己也这么声明）。**但它是我们手上唯一与我们代理评测同场景的第三方数字**（LoCoMo-Refined + `gpt-4o-mini`），可作**量级校准**的参照——**仍不可当基线** |

> **候选仓库那条的读法**：它的作用是回答"**我们离一个能上榜的系统有多远**"这类量级问题（**当前同场景的数：0.6387 / 0.7948，见 [`../reports/ledger.md`](../reports/ledger.md)**；对照口径见 [`../../docs/reference-implementations.md`](../../docs/reference-implementations.md)），**不是**"它 76.9% 所以我们也该有 76.9%"。**两边的题集、注入字段、裁判实现都可能不同。**

> **注意 AML 自己的两篇技术解读末位差 0.04**（45.06/44.97/44.84 vs 45.10/45.00/44.80）。**本 PRD 取前者**（附录 B）。
