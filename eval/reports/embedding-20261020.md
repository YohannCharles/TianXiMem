# Embedding 输入适配 —— CL-Bench 超长查询与发送前截断（2026-10-20）

> **后续复核**：下面 §7 补入按 Search 时点恢复记忆与 Qwen 官方 tokenizer 的实测。
> 其中 tokenizer 安全性、缓存身份和排队因果的限定优先于前面各节；
> `o200k_base × 0.9` 不能保证 embedding 实际输入不超窗，短请求慢也不足以单独证明队首阻塞。

**结论**：官方采集里有一组 CL-Bench `query` 远超 `text-embedding-v4` 的单条窗口
（8,192 token）。适配层此前**原样 POST**，超长请求在网关侧排队/拒绝，同批短查询变慢。
`OpenAICompatEmbedder.encode_query()` 现在**发送前截断**（头尾各一半，预算按 Qwen
官方 tokenizer 的实测比例保守折算为窗口的 0.56）；索引侧不截。**它不是窗口的精确保证**，
边界见 §7.3 与 §8。

---

## 1. 数据与口径

- 来源：[`official-dataset-2026-09-29/grouped/searches/clbench.jsonl`](../../official-dataset-2026-09-29/grouped/searches/clbench.jsonl)
  （该目录不进 git；快照哈希见其 `manifest.json`）。
- 238 条 CL-Bench `/search` 记录（`grouped/searches/clbench.jsonl` 全量）。
- token 数用本仓的 `o200k_base` 计数（`common/tokens.py`）；它不是 embedding 模型的分词器。
  初版预算取窗口的 0.9；§7.3 的 Qwen 官方 tokenizer 对拍否定了它（截后仍有 9 条超窗），
  ⇒ 最终预算改为 **0.56**（`1 / 1.70` 再留余量，依据见 §8）。

## 2. 长度分布

| 指标 | 字符数 | o200k token（该条） |
| --- | ---: | ---: |
| min | 36 | — |
| p50 | 9,050 | 约 2,100 |
| p90 | 29,845 | — |
| p99 | 133,904 | — |
| max | 185,627 | 40,036 |

按原文的 **o200k** 计数，超窗（8,192 token）为 **19 条**；按最终预算
（`8,192 × 0.56 ≈ 4,587`）触发 **32 / 238 条**。其余 206 条逐字不变。
⇒ 这次修复改变的是**最长的约 13%**，不是全部 CL-Bench。

## 3. 延迟观察：与"长请求占住网关"一致，但不能单独归因

17:42 那一分钟是网关重启窗口（313 条 `EmbeddingError` 的根因见
[数据集 README §8.1](../../official-dataset-2026-09-29/README.md)，是 `Connection refused`，
**与超长无关**，延迟中位 3 ms，不能计入）。同一窗口里**成功**的请求是另一条线索：

| 证据 | 数值 |
| --- | --- |
| 一条 **1,274 字符**的查询（17:43:26） | **25,233 ms** |
| 一条 **5,980 字符**的查询（17:42:28） | **25,535 ms** |
| 同批 **>30,000 字符**的成功查询 | 3,370 – 30,786 ms |

短查询自己不需要 25 秒——它与长请求同批变慢，**与"长 embedding 占住网关/队首阻塞"
一致**。但这一分钟同时发生过网关重启，**没有 queue/prefill 指标**，所以不能把
"短请求慢"单独断言为某一台网关的队首阻塞（§7.5）。

## 4. 问句位置：为什么头尾各留一半

对成功且 >10,000 字符的 31 条查询，取最后一个 `?` 的相对位置：

| 分位 | 位置 |
| --- | ---: |
| p10 | 0.08 |
| p50 | 0.85 |
| p90 | 1.00 |

- **38.7%** 的问号落在全文最后 10%；
- **12.9%** 落在最前 10%。

⇒ 只留头部会丢掉多数问句，只留尾部会丢掉开场设定。截断保留**头部 50% + 尾部 50%**。

## 5. 截断验证（真实数据）

对 238 条逐条跑 `truncate_for_embedding()`：

| 断言 | 结果 |
| --- | --- |
| 触发条数 | **32 / 238** |
| 截断后最大 o200k token | **4,571**（预算 4,587） |
| 截断后最大 Qwen token | **7,714**（窗口 8,192） |
| 原文 Qwen > 8,192 的 21 条，截断后仍超窗 | **0 条** |
| 带问号的触发查询（24 条）里，问号落在保留区 | **21 / 24** |
| 未触发的 206 条 | 逐字不变 |

抽样：185,627 字符 → 21,603 字符 / 4,571 o200k / 4,647 Qwen，开头文档与结尾问句都在；
133,904 字符 → 19,462 字符 / 4,571 o200k / 5,022 Qwen，开头的 10-Q 问题与结尾签名都在。

**3 条问号落在被切掉的中段**（去重后 2 个不同 query）：

| seq | 字符 | 原文 Qwen | 问号 token 位置 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 41157 / 47275 | 28,420 | 6,358 | 0.608 | 原文**不超窗**，是保守预算的过度截断；问题在正中，头尾都够不到 |
| 41271 | 73,370 | 18,794 | 0.765 | 原文必须截断；问题在 76.5%，头尾各半够不到 |

⇒ 头尾各半的代价是**问题在中间 50%** 的查询会丢问题。这两例登记在案；
若后续确认这类问法常见，再评估"按问句定位保留下界"之类的策略（属 §7.6 的方案 3）。

## 6. 复现

```bash
uv run python - <<'PY'
import json
from tianximem.common.tokens import load_counter
from tianximem.embed.truncate import truncate_for_embedding

rows = [json.loads(l) for l in open(
    "official-dataset-2026-09-29/grouped/searches/clbench.jsonl") if l.strip()]
c = load_counter()
triggered = [r for r in rows if truncate_for_embedding(r["query"]) != r["query"]]
print(len(rows), len(triggered), max(c.count(truncate_for_embedding(r["query"])) for r in triggered))
PY
```

用例：[`tests/test_embed_truncation.py`](../../tests/test_embed_truncation.py)（14 条，不碰网络）。

## 7. 后续调查：先判断是否需要检索，再决定如何表示长查询

本节是原因分析与方案比较，尚未上线检索策略变更。
保留已有截断补丁；没有保留调查中的焦点抽取草稿。

### 7.1 数据身份与复核方法

读取采集套件中 `dataset=clbench` 的全部 Search，而非 native 加载器构造的代理题。
逐条取 `ts` 之前到达的该用户 Add，用当前 `compose_memory_blocks()` 恢复块，
经 `common.render.render()` 渲染并计数；没有使用提问之后的 Add。
记忆 token 统计统一额外带日粒度日期前缀，以免低估，未计平台外层 prompt 的开销。

| 输入 | SHA-256 |
| --- | --- |
| `official-eval-kit.jsonl` | `43d605bdc02f2ac626a7feebf6c8d6436e26a8481c6d44292223b755e7f879ed` |
| `official-adds.jsonl` | `38b242f3b54a372162785e1d7732d2f5c9d157271853e4a250b3c7cf4a3028d4` |
| `official-searches.jsonl` | `850c167572acd2f009d698e446bf6a5330a187e94d2b7e6b8eeba364c6f76812` |

238 条 Search 涉及 56 个用户、141 份不同 query。37 条 query 超过 14,000 字符，
涉及 28 个用户及 28 份不同 query。所有 238 条的 `question` 都与 `query` 逐字相同，
不存在直接换字段就能取得短问题的路径。

按排序位置 `floor((n-1)*p)` 取分位，字符长度中位为 9,017、p99 为 133,904、
最大为 185,627；这与前文采用其他分位定义得出的中位 9,050 不混用。
o200k token 中位为 2,282、p95 为 12,399、最大为 40,036；
19 条超过 8,192 token，21 条超过当前截断预算。
另有 29 条 `unknown` 长 query，其 `prior_adds_for_user=0`，不应误归为 CL-Bench。

### 7.2 根本错位：当前请求中的文档，不等于需要召回的历史记忆

采集直接显示：长驾驶手册、财报、卫星数据表、库文档都在当前 Search query 里。
多数对应用户的历史只有一条角色、格式或行为要求。
例如 seq 41123 的完整驾驶手册在 query 中；该用户只有一条驾驶助手行为规则。
seq 41161 的财报在 query 中；历史是一条 MBA 教授角色和回答格式要求。

| 范围 | Search 数 | 当时只有 1 块记忆 | 单块渲染 token 最小 / 中位 / 最大 |
| --- | ---: | ---: | --- |
| 全部 CL-Bench | 238 | 227 | 40 / 372 / 1,343 |
| query 超过 14,000 字符 | 37 | 36 | 40 / 150 / 347 |
| query 超过 8,192 个 o200k token | 19 | 18 | 121 / 170.5 / 347 |

唯一的长 query 多块例外是 seq 41283：当时 10 块，合计 18,578 个 o200k token，
含此前论文与对话。不能根据长度就假定当前输入已完备、直接返回空记忆。
全部 238 条在当时的记忆块数均不超过 18，渲染 token 总数最大 27,797；
这些数只说明候选集合小，不证明全量回传一定优于排序。

native [CL-Bench 加载器](../../eval/datasets/clbench.py) 则把文档放在记忆里，
问题只取末条 user 的尾部窗口，这是它明示的代理假设。
因此 native 测试能跑通不能证明真实采集的长 query 安全，也不能据它解释这批数据的检索需求。

### 7.3 已验证的截断缺陷：答案 tokenizer 不是 embedding tokenizer

官方模型卡说明 Qwen3-Embedding-4B 原生支持 32K 上下文，不能把示例中的
`max_length=8192` 当成模型原生上限；实际网关可以部署更短窗口。
当前配置端点的只读 `/models` 返回模型 ID，但未提供 `max_model_len`，
因此本次没有重新证明现役网关的实际窗口。
提交期 `text-embedding-v4` 的单文本上限仍是 8,192，不能通过扩大开发窗口代替提交适配。

来源：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-4B)、
[百炼 embedding 文档](https://www.alibabacloud.com/help/en/model-studio/embedding)。

本次通过 `tokenizers.Tokenizer.from_str()` 加载官方 tokenizer JSON，
`add_special_tokens=False`，对完整输入与当前截断输出分别计数。
tokenizer revision 为 `5cf2132abc99cad020ac570b19d031efec650f2b`，文件 SHA-256 为
`83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d`。

238 条中，Qwen token 超过 8,192 的为 21 条，超过 32,768 的为 5 条。
应用当前截断后，仍有 9 条超过 8,192 个 Qwen token：

| seq | 原文 o200k | 原文 Qwen | 截后 o200k | 截后 Qwen |
| --- | ---: | ---: | ---: | ---: |
| 41106 | 15,701 | 25,949 | 7,356 | 11,290 |
| 41124 | 22,774 | 24,542 | 7,355 | 8,283 |
| 41127 | 37,139 | 62,647 | 7,356 | 12,509 |
| 41129 | 9,016 | 13,703 | 7,356 | 11,441 |
| 41130 | 9,336 | 10,597 | 7,356 | 8,352 |
| 41161 / 41215 / 47397 | 31,941 | 37,205 | 7,355 | 8,498 |
| 41283 | 12,399 | 13,463 | 7,356 | 8,210 |

这是同系列官方 tokenizer 的离线验证，不是网关内部 tokenizer 的逐字节核验，
也不是 `text-embedding-v4` tokenizer 的精确计数。
但已经足以否定“o200k 留 10% 就能保证所有后端窗口安全”。
其中 seq 41129 只有 13,816 字符，说明按 14,000 字符设门槛还会漏掉密集数值内容。

### 7.4 已复现的缓存身份问题

当前 [CachingEmbedder](../../src/tianximem/embed/base.py) 的文档与查询路径都调用
`cache.get(text)`，但文档侧对长文本原样编码，查询侧先截断；原文相同不再意味着模型输入相同。
用记录输入并按输入长度返回向量的假 HTTP 客户端可复现：

- 先 `encode([long_text])` 再 `encode_query([long_text])`：只发一次 HTTP，查询复用完整文档向量。
- 先 `encode_query([long_text])` 再 `encode([long_text])`：只发一次 HTTP，文档复用截断查询向量。

此外，查询适配策略、预算变化后，原文哈希未变，旧向量仍会命中。
保留原文作为唯一查询缓存键并不能防止这种情况。
可选修正是缓存实际模型输入，或用“角色 + 适配版本 + 参数 + 原文”构造查询身份；
多个原文经确定性适配得到同一份实际输入时，共用向量本身不是错误。
修改查询身份不应无谓失效 Add 的全部文档缓存。

### 7.5 排队机制与证据边界

[HybridRetriever.search](../../src/tianximem/retrieve/fusion.py:167) 先等 dense embedding，
成功后才调用同时含词法与 dense 的 Qdrant hybrid 查询。
所以 embedding 阻塞时，BM25 也没有机会先返回。
Search 的 LLM planner、router 在 embedding 之前；reranker 后续也收到原始全文 query。
仅保护 embedding 后，长输入仍可能在其他阶段放大计算和排队。

长输入的 tokenize、prefill、连接占用及共享推理调度，是短请求也慢的合理解释；
拆开 Add/Search/答案/判分只消除阶段间竞争，不消除 embedding 阶段内部的长短请求竞争。
但本次未取得刚才那轮运行的网关 queue/prefill 指标、拒绝发生时点或取消日志，
不能把短请求慢直接断言为特定网关的队首阻塞，也不能断言所有超长请求都会“先推理再拒绝”。
历史 17:42 附近还同时发生过网关重启，`status=null / EmbeddingError` 本身不证明超窗。
HTTP 客户端超时是否取消服务端计算也待验证；若未取消，外层重试可能叠加仍在运行的任务。

### 7.6 建议顺序与方案取舍

1. **先验证单块记忆直接返回。** 在 Search 编排入口、planner/router/embedding 之前，
   从同一只读快照按 `user_id` 有界读取候选；只读到唯一一个可返回记忆、且完整打包不超
   `top_k` 和预算时，复用既有渲染、稳定 ID 和日期返回。单块无排序问题，
   不需要压缩长 query，也不丢记忆内容。当前采集可覆盖 36/37 条长 query。
   这项依据记忆结构而非数据集名；空用户可以直接返回合法空数组。
   必须先审清 deleted/pending 等有效性与 Add 未完成索引的可见性，不能绕过存储状态语义。
   此策略是对 PRD §7.2“普通 dense 链每查询一次”的显式新增旁路，实施前应同步规格。

2. **再试有预算的小集合回传。** 只有整个候选集可完整取出并满足槽位、token 预算时才考虑，
   其中单块以外仍有证据排序、无关历史干扰和治理语义需要验证。
   “不超过 top_k”不是独自充分的条件，不能对所有小用户直接泛化。

3. **大集合使用检索用的短表示。** 保留全文给最终答案输入，把本轮任务、指代对象、
   必要时间和约束作为检索视图；先用显式边界与逐字片段，复杂情况再评估受限 LLM 提取。
   多视图检索后按记忆 ID 融合，比将整份文档等权平均成一个向量更有针对性。
   但文档内部也会有问号、`Task:`、`Question:`；问号位置不等于真实任务边界。
   离线草稿已在 Kant 正文问句、模板内部问题、论文段落上误选，因此未保留默认实现。
   BM25 和 reranker 同样需要审查输入表示与各自的预算，不能只缩 dense 后声称链路已经解决。
   查询改写及多视图调用涉及 PRD §7.2 的策略变化，需要独立实验和缓存版本。

4. **截断保留为模型适配的末道防线。** 窗口跟随后端实际能力，计数尽量使用对应 tokenizer；
   无法取得 tokenizer 时应明确近似范围与可验证的保守策略。
   在发请求前完成限长；策略版本进入查询缓存身份，记录触发与退化原因。

5. **token 预算式调度补上运行保护。** 同时限制在飞请求数和估计 token 总量，
   将长短任务分队列并设置每请求工作预算，避免多个长请求同时占满后端。
   同机多进程或不同 URL 不自动等于 GPU 与调度资源隔离；需要后端取消与队列指标验证。
   这项保护调度公平性，不能代替上面减少不必要计算的策略。

**不优先选择的方案**：整篇分块后平均 embedding 会增加 token 工作总量并稀释任务焦点；
所有分块同时发送还可能加重排队。它最多解决单条输入上限，不自动改善检索或短探针延迟。
先用 LLM 总结整篇文档也要处理全部长输入，可能把瓶颈移到 LLM，且可能丢掉数字与否定约束。
扩大开发网关窗口可暂时减少拒绝，但不能满足提交模型的限制。

### 7.7 验证状态

已完成全量采集结构统计、按时点恢复 CL-Bench 记忆、官方 Qwen tokenizer 对拍、
缓存调用顺序复现及现有 embedding 用例回归：
`uv run pytest -q tests/test_embed_truncation.py tests/test_embed_cache.py tests/test_query_instruction.py`
为 **49 passed**。这不证明截断后的 Qwen 输入安全，上述反例正是测试尚未覆盖的边界。

下一项最小验证应是单块分支：核对它与既有检索的记忆 ID/content，验证无远程模型调用、
跨用户隔离与完整预算；再将唯一多块长题和普通用户作为对照。
本次没有向网关重发超长 embedding，也没有重跑全量答案/判分或宣称准确率提升。

## 8. 最终预算与缓存身份（2026-10-20，§7 的修正落地）

### 8.1 预算从 0.9 改为 0.56

§7.3 用官方 Qwen tokenizer 对拍后，初版 `o200k × 0.9` 在 21 条超窗查询里截完仍有
**9 条**超过 8,192 Qwen token（最高 12,509）。实测最大 Qwen/o200k 比例为
`12,509 / 7,356 = 1.70` ⇒ 预算取 `1 / 1.70 ≈ 0.588`，再留余量定为 **0.56**
（`_BUDGET_RATIO`，默认预算 `8,192 × 0.56 = 4,587` o200k token）。

用同一份官方 tokenizer（revision `5cf2132…`，文件 SHA-256
`83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d`）复验：

| 指标 | 原文 | 截断后 |
| --- | ---: | ---: |
| Qwen > 8,192 的条数 | 21 | **0** |
| 最大 Qwen token | 40,509 | **7,714** |

⚠ **它仍然只是近似**：`text-embedding-v4` 的 tokenizer 与本仓的 `o200k_base`
不是同一套，Qwen 的 1.70 只是当前 238 条上的**实测最坏比例**，不是对所有文本的保证。
拿到 v4 的 tokenizer 后应重新标定；在那之前这是"可验证的保守策略"，不是精确窗口。

### 8.2 缓存身份：查询侧与文档侧分开

§7.4 的两个复现顺序都源于同一个事实：**文档侧原样编码、查询侧截断后，
原文相同不再意味着模型输入相同**。修复落在 `CachingEmbedder`：

| 侧 | 缓存身份 |
| --- | --- |
| 文档 | 原文（**不变**——已有的 Add 缓存不失效） |
| 查询 | `QUERY_CACHE_PREFIX + 原文`（`"query-adapter-v1:"`） |

截断策略/预算一变，抬前缀里的版本号即可整体失效查询向量；文档缓存不受影响。
新增用例：

- `test_document_then_query_never_shares_a_vector`
- `test_query_then_document_never_shares_a_vector`
- `test_cache_key_is_the_original_text_with_a_query_role_prefix`

### 8.3 验证状态

```bash
uv run pytest -q tests/test_embed_truncation.py tests/test_embed_cache.py \
  tests/test_query_instruction.py tests/test_retrieve.py
```

**90 passed**（§7.7 的 49 条是修正前的数字，新增用例后为 90）。
全量 `uv run pytest -q` 为 **1806 passed**（缓存身份与预算修正后）。

**仍未做**：`text-embedding-v4` tokenizer 的对拍；长查询截断对 CL-Bench 检索质量的
A/B（报告 §7.6 的"单块记忆直接返回"等方案尚未实现）。本次没有向网关重发超长 embedding。
