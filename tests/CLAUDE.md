# tests/ — 单元测试

**PRD**：§2.2（硬性规则）、§6.2 / **D24**（记忆块组合）、§6.5（幂等）、§7.2 / §11.3（同一份渲染）、§13（开关纯度）

## 要写什么

```text
test_pairing.py         §6.2 配对判据（三种真实情况）
test_apply.py           一次 Add 的落库边界（D24）+ 幂等协作
test_idempotency.py     §6.5 / §15 批次级守卫
test_contract.py        §2.1 / §2.2 契约形状与计数（含真 HTTP 往返）
test_contract_preflight.py  `eval/smoke/preflight.py` 的静态一致性与**检查是否真会失败**
test_isolation.py       §2.2 user_id 隔离
test_render.py          §7.2 / §11.3 同一份渲染
test_annotate.py        §11.3 / **D21** 相对时间就地注解：**只加不改**、推不出不动、**不碰索引**
test_store.py           §6.1 / §6.3 DDL、连续性与索引、**连接生命周期与并发**
test_config.py          §15 配置唯一入口（**AST 静态断言** + 校验 + `.env` 读取）
test_neighbor.py        §10 / §11.2 扩窗 + 段合并（**24 条清单的主体**）
test_packaging.py       §11.3 段级打包 + 双预算（段进来之后的事）
test_reranker.py        §11.2 远端精排：线格式、降级、两个计数器、装配（**不发真请求**）
test_observability.py   §14 指标出口：聚合、**计数是增量不是累计**、装配与 runner 读取（**V12**）
test_capture.py         请求**原文**采集（**S6**）：开关归 yaml / 路径归 env、一行的形状、
                        **以及"它什么都没改变"**（不改响应、不改真源、不吞异常）
```

> ⚠ **§13 的开关纯度没有独立文件**。断言住在**各自开关所在的模块**里：
> `test_reranker.py` 的 `test_candidate_set_is_unchanged_by_rerank`（关掉精排后 **id 集合一个不多一个不少**）
> 与 `test_rerank_does_not_touch_the_token_budget`，
> `test_neighbor.py` 的 `test_neighbors_do_not_change_segment_priority`（`radius=0` 与 `=1` 对比，段的 `best_rank` 与锚点一字不变）。
> **要另立一个横跨各模块的纯度文件，先解决"`neighbor.enabled` 还没接线"**——断言只能写在
> **真能关的开关**上（清单见本文件 §五末）。断言的规矩见 §五。

---

## 一、记忆块组合（§6.2 / **D24**）

**组合规则一处声明在 [`../src/tianxi_am/pairing/CLAUDE.md`](../src/tianxi_am/pairing/CLAUDE.md)**，
本节只列**要覆盖的用例**。

```text
test_pairing.py    组合规则本身（纯函数）
test_apply.py      一次 Add 的落库边界（含跨 Add 的三条）
test_idempotency.py  批次级守卫
```

| 用例 | 覆盖 | 住在哪 |
| --- | --- | --- |
| `U A U A` / `U U A A A U A` / `A U A` / `U A U` / `A` / `U` | 六个形状逐字对照 | `test_pairing.py`（**Test 1–6**） |
| 连续同 role 合并、相邻 U+A 配对、落单独立成块 | 同上，含工具调用序列**不劈开** | `test_pairing.py` |
| 未知 role（`system` / `tool` …） | **不得丢消息**——判据只依赖"是不是 `user`"，**不要枚举白名单** | `test_pairing.py` |
| 来源元数据 | `source_idxs` 合起来是 `0..n-1` 的**排列**（无丢失、无重复） | `test_pairing.py` |
| **跨 Add 不拼接** | `Add0=[U A U]` + `Add1=[A U A]` ⇒ **不出现 U(Add0)+A(Add1)** | `test_apply.py`（**Test 7**） |
| **连续 assistant 跨 Add** | 不合并成 `AAA` | `test_apply.py`（**Test 8**） |
| **乱序 Add / 同 session 并发** | **性质级断言**（D28）：乱序 + 多线程投喂的最终 `{位置集合, 每块正文, id 集合, prev/next 指针}` 与顺序投喂**逐字一致**（不是"每个 Add 各自对"） | `test_apply.py::test_arrival_order_does_not_change_the_truth_source` |
| 同一 `request_id` 重试 | 不重复落库 / 不重复 embedding | `test_idempotency.py`（**Test 10**） |
| `(request_id, local_index)` 不重号 | 不同 Add 并发时**互不相交**（D28）；重放撞 `UNIQUE` 而**不是**静默重复 | `test_apply.py`、`test_store.py` |
| 空批次 | **响亮失败**，不静默 no-op | `test_apply.py` |
| **`request_id` 是 opaque string** | 任意形状（`abc` / `foo:bar` / `xxx:chunk-0-extra` / UUID）都必须能正常 Add（D28）——**不许有格式要求** | `test_apply.py::test_request_id_is_opaque_and_any_shape_works` |
| **邻接只在一次 Add 内** | 同 user / 同 session 的两次 Add **绝不互为邻居**（`M2.next is None`、`M3.prev is None`） | `test_apply.py`、`test_neighbor.py` |

> ⛔ **不要再写这几类的用例**（**实现不存在，写了也无处落**）：
> 跨批续接三步 `3a′/3a/3b/3d`、`pending` 判定与三个计数器、`open_pair` /
> `append_question` / `append_answer` / `mark_complete`。（均为 **D24** 取消的东西。）

## 二、幂等（§6.5 / §15）—— 唯一能抓到"守卫被绕开"的用例

**核心用例必须模拟这个时序**：

```text
1. 批次 A 应用成功（事务已提交，位置已分配）
2. 响应【未发出】或进程崩溃
3. AML 重试批次 A（request_id 与 payload 不变）
4. 断言：库里【没有新增行】，且既有行的内容不变
```

⚠ **它抓的是这一对**：位置是**请求的纯函数** ⇒ 重放必然算出**同一位置** ⇒ 守卫被绕开就会撞
`UNIQUE`、表现为**整批 500**；而用例要证明的正是**守卫把"正常重试"从 500 里救回来**。

**只测"重复 POST 两次"仍然太弱**——它测不到"提交后、响应前"那个中间态。
必须显式构造（这正是 ≥1 次真 HTTP 往返的用例存在的理由）。

**另外两条**：

- 断言守卫查的是 **`applied_batches`**，**不是 `qa_pairs.request_id`**（后者会被后一批覆盖，指纹丢失）
- 断言 `answer` 的**追加**语义：同一批至多被应用一次 ⇒ 追加安全

---

## 三、契约（§2.1 / §2.2）

| 断言 | 说明 |
| --- | --- |
| **`len(data) <= top_k` 精确成立** | 但**数的单位是段，不是 raw memory**——见下面的"段模型改了四件事" |
| 空结果是 `[]` 不是 `null` | §2.1 |
| 每项含 `id` / `content` / **`created_at`** | `created_at` **始终存在**：日粒度或 `""`（§11.3） |
| `score` 单调递减 | 且**不是**原始 RRF 分数（§11.3） |
| 200 响应原样回显三个字段 | `request_id` / `user_id` / `session_id` |
| **根级 `limit` 取自请求的 `top_k`** | **没写死 100**——写死会在 AML 传更小值时变成"返回超限"，那是**契约错误**（§7.3） |

### ⚠ 段模型改了四件事——写用例前必须知道

**四条都会让"照着旧假设写的断言"变成空过**，而它们**全都不会报错**：

| 陷阱 | 会空过的假设 | 现在怎么造用例 |
| --- | --- | --- |
| **一个候选 ≠ 一项** | "落 N 条记忆 ⇒ 返回 N 项" | 想要 N 项就让它们**各自成一次 Add**（`conftest.seed_pair_in`，D28 起不同 Add 永不合并），或者**分属 N 个 session**。⚠ 反过来想要"一段"就得用 `conftest.seed_line`（**同一次 Add 内**连着落几条） |
| **相邻会被扩进来** | "落了 1 条 ⇒ 只返回那 1 条" | 它会把 `prev` / `next` 指着的邻居**一起带回来**（`test_neighbor` 的那几条整链用例） |
| **重复写入 ≠ 多一项** | "重复 POST ⇒ 检索里多一条" | 同一批重放**根本不写**（`applied_batches` 守卫），所以要比的是"行数没变"而不是"结果条数"。**必须比 `content`、不能只比 `id` 与条数**（`preflight.check_replay_does_not_write_again`） |
| **`score` 数的是段的位置** | "`score = 1/(rerank 名次+1)`" | 是 `1/(**输出位置**+1)`——预算跳段时照抄名次会出现空洞，而"还是单调递减" |

> **这就是"一个永远不会 FAIL 的检查等于没有检查"的又一次具体教训**：
> 预检里"`top_k` 真的会截断"那条一度**静默空过**（3 条相邻记忆 = 1 段，
> `top_k=1` 返回 1 条成了必然）。⇒ **凡是"要多条才能验"的检查都要回头看一眼**。

### 隔离（§2.2）

| 断言 | 说明 |
| --- | --- |
| 任何路径都不返回其他 `user_id` 的记录 | **含 Agent 工具调用、邻域扩展、缓存命中**——后三条是漏得最多的地方 |
| `session_id` **没有**被当成 Search 的过滤条件 | 它是分组字段，不是过滤器 |

**隔离要按"路径"逐个测**，不能只测主检索路径——**邻域扩展是 SQL 查询，很容易忘记带 `user_id` 条件**。
⇒ ✅ 已覆盖：`test_neighbor.py::test_expansion_never_crosses_sessions`（**同位置**、不同 session / 不同 user
三种行同时摆在库里，只有同 `(user_id, session_id)` 的那条能进）。

---

## 四、同一份渲染（§7.2 / §11.3）

| 断言 | 说明 |
| --- | --- |
| **embedding 的输入 == 返回的 `content`** | 同一个 QA 对、同一次渲染调用 |
| `question` 为空 → 只有 `A:` 行 | §11.3 |
| `answer` 为空（块里没有非 user 消息） → 只有 `Q:` 行 | §11.3 |
| 多非 user 消息**每条带 role 标记** | §11.3 |
| **首尾无空白** | AML 只做 `"\n".join(...)`，**不插分隔符** |
| **绝对时间只到日粒度**（秒级绝不出现）、**相对表述原样保留** | **D21**（2026-09-25，有 346 题的反例） |

**第一条是最重要的**：两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"会**漂移**，**而且这种漂移不会报错**。

---

## 五、开关纯度（§13）—— 最便宜、保护面最大的一类

> **开关必须只影响它命名的那一件事。** 如果关掉 rerank 顺带改变了候选数量、或关掉 agent 顺带改变了打包顺序，这个对照**就不成立**——**而结果看起来完全正常，只是结论错了**。

**做法**：对每个开关，**断言"关"分支下不变量不变**：

| 开关关掉时 | 必须不变 |
| --- | --- |
| `rerank: false` | **候选数量**不变（只是顺序变了） |
| `agent: false` | **打包顺序**不变（只是候选少了 agent 补的那部分） |
| `neighbor: false` | 种子集合不变（只是没有扩窗） |
| `packaging.annotate_relatives: false` | **名次 / `id` / `created_at` / `score` / 段数 / token 口径全不变**，而且**被索引的文本逐字相同**（只是 `content` 少了那层括号注） |
| `dense: false` | ——（**无下游依赖**：D15 删掉了裸 BM25 模式与"验证后再加 dense"的分阶段，所以没有任何东西依赖它） |

> ✅ **`rerank` 的纯度断言**：
> `test_reranker.py::test_candidate_set_is_unchanged_by_rerank` 断言"精排把顺序整个倒过来之后，
> **id 集合一个不多一个不少**"；`test_rerank_does_not_touch_the_token_budget` 断言
> "开/关精排**不改变 token 计数的次数**"（段数变则计数变，而段数由数据决定）。
>
> > ⚠ **这条纯度是结构性保证，不是"我们记得别改集合"**：`RemoteReranker.score()` 返回的是
> > **按输入位置对齐**的分数，重排由 `pipeline._maybe_rerank` 对**同一个列表**做——
> > 那条链上**没有第二条路径**能改动候选集合。若哪天有人让 reranker 返回"排好序的 id 列表"，
> > 这条纯度就只剩一句注释了。

> ✅ **`neighbor` 的纯度断言**：扩窗的旋钮是 `neighbor.radius`，
> `test_neighbor.py::test_neighbors_do_not_change_segment_priority` 把 `radius=0` 与 `=1` 对比，
> 断言**段的 `best_rank` 与锚点一字不变**（只有段的长度变了）。
> ⚠ 但 **`neighbor.enabled` 这个开关本身还没接线**（见 `config-reference.md` §2）——
> 现在能关的只有 `radius`，**别把"能关半径"当成"开关已落地"**。

### 精排测试的三层（`test_reranker.py`）—— **不发真请求**

| 层 | 怎么隔离 | 为什么这么切 |
| --- | --- | --- |
| `RemoteReranker` | `httpx.MockTransport` | 让"超时 / 5xx / 坏 JSON / NaN"变成**可精确构造**的输入 |
| `SearchPipeline._maybe_rerank` | `FakeReranker`（协议级） | 恰好一次、降级、两个计数器——**不需要 HTTP** |
| `build_reranker` / `build_services` | 真 `AppConfig` + tmp 路径 | 装配漏了 `None` 是**静默**的：Search 照常工作，只是没精排 |

> ⚠ **超时要用"抛 `httpx.ReadTimeout`"来模拟，不要让假传输层真的睡**：
> `MockTransport` 不参与超时计时，睡多久都不会触发超时——那样写的用例是**空过的**。
> 真超时与它是同一个 `httpx.HTTPError` 分支。
>
> ⚠ **`NaN` / `Infinity` 只能手工构造原始文本**（`httpx` 的 `json=` 走 `allow_nan=False`），
> 而 Python 的 `json.loads` **默认接受**这三个非标准字面量——所以"字段是数字"的检查挡不住它们，
> 必须显式 `isfinite`。
>
> ⚠ 真端点那一路在 [`../tools/probe_reranker.py`](../tools/probe_reranker.py)（`make probe-reranker`）：
> **单元测试用假传输层，真连通性用探针**——两者都要，不能互相替代。

**这组测试很便宜，而它保护的是整个 §13 实验计划。** 没有它，所有消融结论都不可信。

> ⚠ `checker` 与 `dense` **之间没有依赖边**（D15）——
> 见 [`../docs/config-reference.md`](../docs/config-reference.md) §2 的"三条**不要实现**的东西"。

---

## 六、存储（§6.1 / §6.3）

| 断言 | 说明 |
| --- | --- |
| `UNIQUE(user_id, session_id, request_id, local_index)` 是「取一次 Add」的排序键 | **D28**：既保唯一，又给 `fetch_by_request` 提供顺序（`ORDER BY local_index`） |
| **链上缺一块就断开** | `prev` / `next` 指着的是一个**不存在的 id** ⇒ 扩窗走到头就停、合并各自成段（**不许**按 `local_index` 硬推） |
| **`id` 位置派生**；同一位置重放得到**同一个 `id`** | 内容哈希会留下**孤儿 point**；D28 之后 `id` 由 `(request_id, local_index)` 决定（`request_id` 不解析） |
| **`index_pairs` 失败会留下"SQLite 有、Qdrant 没有"的行** | 检索命中的是**不存在的那个版本**而**不报错**——要测的是这条路径能被修复 |
| **配的向量维度来自接口，不是常量** | §2.3 / §7.4——写死会在 Step 5 静默错 |
| **能仅凭 SQLite 全量重建 Qdrant** | §6.3 的"派生读存储"就是这条的意思 |
| **代码中没有硬编码 `benchmark_data/` 或 `eval/datasets/`** | D16：路径一律走 `TIANXI_BENCHMARK_DIR` |

### 连接生命周期与并发（D17）

| 断言 | 说明 |
| --- | --- |
| **连接不跨线程复用**：在另一个线程里读 + 写都正常 | 长期持有一个连接会在另一个线程里抛 `ProgrammingError`——而 FastAPI 的 `def` 路由**就在线程池里** |
| 同一操作**各拿一个连接**（两次 `read()` 不是同一个对象） | 短生命周期模型最直接的可观测性质 |
| **异常后：事务已回滚 + 连接已关** | 只断言回滚会漏掉连接泄漏；只断言关闭会漏掉脏数据。**两条都要** |
| **并发写不丢不串**：多 session 同时 `BEGIN IMMEDIATE` ⇒ 无 `SQLITE_BUSY`、**每 session 的位置不重号**、无跨 session 污染 | 见 `test_store.py` 与 `test_service_add.py` 的**压力**用例 |
| **同 session 并发不再串行**（**D25**） | 位置互不相交 ⇒ 断言**两端都成功**且位置集合不重 |
| **跨 session 不互相阻塞** | 用 `Barrier` **证明**（而不是靠 sleep 赌时间）——若被串行化，barrier 会超时 |

> **测试里读一律写 `rd(store, store.<方法>, ...)`**（[`conftest.py`](./conftest.py)）：
> 它对应生产代码的 `with store.read() as conn:`——**一次逻辑操作一个连接**（D17）。
> **写必须走 `store.transaction()`**，绝不自己开连接。
>
> ⚠ **别把 `xfail` 当成"记录缺口"的长久手段**：缺口修好后它们会 XPASS，而
> `strict=True` 会提醒摘标记——**但它也可能顺手盖住用例里的另一个真 bug**。摘标记时要
> 确认它是因为"该过的过了"而 XPASS，不是"换了个失败理由"。

---

## 跑之前

```bash
make test      # uv run pytest
```

**契约相关的断言也要能被 `make contract-check` 在**服务**上跑一遍**——单元测试验证函数，[`../eval/smoke/preflight.py`](../eval/smoke/) 验证**真的 HTTP 响应**。两者都要（§13 要求主路径能通过 Smoke 契约校验）。

⚠ **"门禁必须能失败"本身也要有测试**：[`test_contract_preflight.py`](./test_contract_preflight.py) 塞一个**故意违规的假服务**，逐条确认对应的检查报 FAIL，再用合规的假服务做**阳性对照**。
**一个永远不会 FAIL 的检查等于没有检查**——而"没有检查"与"检查通过"在屏幕上是同一个样子。
