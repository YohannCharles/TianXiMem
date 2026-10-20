# embed/ — Embedding 后端

**PRD**：§7.4（接口）、§7.2（缓存与索引/查询两侧）

## 本模块的构成

```text
base.py               ✅ Embedder 协议（含 `OpenAICompatEmbedder` HTTP 基类）
text_embedding_v4.py  ✅ 提交链路用（§2.3 唯一允许的 embedding 模型，2026-09-28 建）
qwen3_embedding.py    ✅ 开发期对照实验用（**不进提交链路**）
query_instruction.py  ✅ 查询侧 instruction 兼容层（**默认关**，见下"两侧的输入"）
```

> **两个实现只差 `base_url` / `api_key` / `model`，外加一处 API 事实**：`text-embedding-v4`
> 的端点**单请求至多 10 条输入**（超了 HTTP 400），所以它的批大小是 10、不是基类默认的 64。
> **批大小跟着模型走**——装配按 `models.embedder` 挑实现（[`../service/app.py`](../service/app.py)
> 的 `build_embedder`），判据是模型名本身，不是另开一个配置键。
> ⚠ 端点与密钥仍走 `.env` 的 `AML_EMB_*`，**值随部署而变**（开发期 = 自建主网关，
> 提交期 = DashScope）——**别拿开发网关跑提交口径**，理由见 `docs/decisions.md` **D27**。

## 接口（§7.4）

```python
class Embedder(Protocol):
    dim: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...
```

> **代码中不得出现任何依赖具体模型输出结构的逻辑。**

`dim` **由接口提供、不能写死**。Step 5 换模型时按新维度**重建集合**（§2.3 / §16）。

---

## 为什么这个抽象是硬要求，不是好品味

**架构必须 embedder-agnostic**——§2.3 的规定（原文见根 `CLAUDE.md`）推出的设计约束：开发期的 Qwen3-Embedding-4B 与提交期的 `text-embedding-v4` **都不提供 sparse 或 ColBERT 输出**，**任何依赖多向量能力的代码都是死重**。若代码里渗入了多向量能力，Step 5 就不是"换一个实现"，而是"删掉一批代码并重写调用方"（§12.1 R1）。

**同理**：词法那一路**不要用任何学出来的稀疏权重**——理由与替代方案见 [`../retrieve/CLAUDE.md`](../retrieve/CLAUDE.md)（§7.1）。

---

## 两侧的输入必须不同，且各有约束

| | 索引侧 | 查询侧 |
| --- | --- | --- |
| 输入 | 一个 QA 对的**渲染文本**（§11.3），**一对一个向量** | **query 原样送进去** |
| 改写 | —— | **不做任何改写**——v1 没有 Query Analyzer，查询改写属 v2 |
| 调用次数 | 每个对 1 次（缓存后趋近 0） | **每查询恰好 1 次** |

> **⚠ 查询侧另有一层"输入格式"兼容层**（[`query_instruction.py`](./query_instruction.py)）：Qwen3-Embedding-8B 的模型卡推荐**查询侧加 `Instruct: …\nQuery:…`、文档侧不加**，并称不加会让检索掉约 1%–5%。
> 默认 `instruction=""` ⇒ **与"原样送"逐字节相同**；要不要开是**待定的规格问题**，不是实现细节。**唯一不可违反的顺序：前缀必须加在缓存之上**（理由见该模块 docstring）。
>
> **查询侧"不做改写"是 v1 的规格，不是省略。** §5 明确区分了两件事：图里的"关键词重写"是 **Agent 每轮自己产出的检索关键词**（属于 agent 循环）；而**检索之前那一次独立的 query 改写属 v2，v1 没有**。**v1 两者都没有**（D13）。

### 查询侧超长输入：**发送前截断**（2026-10-20）

> 官方采集（`official-dataset-2026-09-29`）里 CL-Bench 有一组 `query` 远超
> `text-embedding-v4` 的单条窗口（`EMBED_MAX_TOKENS = 8,192`）。长度分布、延迟与
> 问句位置的逐项数字见 [`eval/reports/embedding-20261020.md`](../../../eval/reports/embedding-20261020.md)。
> 适配层此前**原样 POST**：网关要先收完整请求、再 tokenize，然后才拒绝或静默截断——
> 这段时间它占着连接与推理槽位，同批的短查询一起变慢。
>
> **因此 `OpenAICompatEmbedder.encode_query()` 在发送前截断**（[`truncate.py`](./truncate.py)）：

| 规则 | 落在哪 |
| --- | --- |
| **只截查询侧**：`encode()`（文档侧）**原样发送** | 索引侧正文由打包预算与 `tests/test_datasets.py` 钉在窗口内；那里超窗是**数据 bug**，必须让本地网关响亮 400，不能靠截断掩盖 |
| **前缀先加、截断后做** | `QueryInstructionEmbedder.encode_query()` → 内层 `CachingEmbedder.encode_query()` → `OpenAICompatEmbedder.encode_query()`。反过来会把前缀截掉 |
| **缓存身份两侧分开** | 文档侧仍用原文（已有 Add 缓存不失效）；查询侧用 `QUERY_CACHE_PREFIX + 原文`。两侧共用原文键时，先文档后查询（或反之）会静默复用另一侧的向量——原文相同**不再**意味着模型输入相同 |
| 预算 = `EMBED_MAX_TOKENS × 0.56`，头尾各一半 | 0.56 的依据是 Qwen 官方 tokenizer 的实测最坏比例 1.70（`1/1.70` 再留余量），见报告 §7.3/§8.1；**它不保证 `text-embedding-v4` 的 tokenizer 同样安全**，只是可验证的保守近似。头尾各半的依据是 CL-Bench 问句位置分布 |

> ⚠ **它没有配置键，是有意的**：窗口是模型的 API 事实（与 `rrf.k = 61` 同类），
> 写进配置只会多一个"配得出、跑不通"的旋钮。截断不动 `content`、不动索引、不改 §7.2 的
> "查询原样送"——它只是**在模型窗口处**做的输入适配，和 instruction 前缀同一层。
>
> 用例：[`tests/test_embed_truncation.py`](../../../tests/test_embed_truncation.py)
> 钉住"文档侧不被截""两侧缓存身份分开""前缀先于截断""结果严格不超限"。

**索引侧的渲染必须是 `common/render` 的那一份**（§7.2 / §11.3 要求它与返回给 AML 的 `content` 是同一份渲染）。**`embed/` 不自己拼字符串。**

---

## 缓存（§7.2）

> **缓存键的规则（= 渲染后文本的哈希，不能用 `id`）在根 `CLAUDE.md`。本节写本目录要落地的三条。**

| 要求 | 说明 |
| --- | --- |
| **必须落盘** | 否则每次重启重付一遍 |
| **能整体失效** | Step 5 切模型时要能一次性作废（§12.1 R1 对冲 1）。作废动作是 [`../../../deploy/CLAUDE.md`](../../../deploy/CLAUDE.md) 的 Step 5 runbook 第 3 步 |
| **缓存键的坐标系要写进缓存文件本身** | 模型名 + **渲染模板版本** + **期望维度**（`models.embed_dim`）。⚠ 放进去的是**声明值**、不是探测值——探测值第一次调用之后才知道、进不了文件名；而声明值换一个就换一个文件（2026-10-09 那次「模型名没变、网关侧改了维度」正是靠这条防的） |
| **一个坐标系一个缓存文件** | 落点是 `<cache_dir>/<coordinate_key>.db`（[`base.py`](./base.py)）——换坐标系自然换文件，别把所有坐标系塞进同一个库 |

**为什么落盘是硬要求**：v1 的 Add 阶段**不调用任何 LLM**，embedding 是**唯一的 Add 侧成本**，且**只与内容有关**——**缓存后即成为一次性成本，与迭代次数无关**。若不落盘，每次改检索逻辑重跑 harness 都要把整个数据集重新 embed 一遍，**成本随迭代次数线性增长**。

**为什么坐标系必须写进缓存文件**：否则 Step 5 切换后**旧缓存会静默命中**，拿开发期模型的向量去查 `text-embedding-v4` 的集合，而**同维度时没有任何一层会拒绝**（Qdrant 也不拒绝），**只表现为"检索结果很差"**；**维度不同时多半会被缓存与集合两处的维度自检响亮拦下——但那是运气，不是保障**。
