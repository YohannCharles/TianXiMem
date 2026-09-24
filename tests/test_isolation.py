"""**隔离**：`user_id` 是唯一的检索隔离字段（§2.2）。

> `session_id` 只是分组字段，**不是 Search 的过滤器**。

⚠ Qdrant 侧真正的隔离（payload filter 在**候选阶段**生效）已由 ②
`tests/test_qdrant_store.py::test_user_filter_applies_at_candidate_stage_not_after_fusion`
证明。本文件证明的是**服务层**：请求里的 `user_id` 被原样传到检索、且不跨用户串结果。

⚠ 走 `SearchPipeline.run()` 而不是 HTTP：本文件用**假 Qdrant**（`FakeQdrantSearch.by_user`）
精确指定"每个用户的检索会返回什么"，而 `/search` 的路由那一跳已由
`test_contract.py::test_http_search_round_trip` 覆盖。**两条路径都要有，但不是每处都重复。**
"""

from __future__ import annotations

from collections.abc import Callable

from tests.conftest import Wired


def test_search_never_returns_another_users_memories(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """**跨 user 检索被禁止**（§2.2）。"""
    alice = seed_pair(wired.store, 0, "alice 的秘密", "a", user_id="alice")
    bob = seed_pair(wired.store, 0, "bob 的秘密", "b", user_id="bob")
    wired.qdrant.by_user = {"alice": [alice], "bob": [bob]}

    got = wired.search(user_id="alice", query="秘密", top_k=10)

    assert [item.id for item in got.items] == [alice]
    assert bob not in {item.id for item in got.items}


def test_user_id_is_forwarded_to_the_retriever(wired: Wired) -> None:
    """请求里的 `user_id` 必须**原样**到达检索层——否则隔离就断在中间。"""
    wired.qdrant.by_user = {"alice": []}

    wired.search(user_id="alice", top_k=5)

    assert wired.qdrant.calls[0]["user_id"] == "alice"


def test_same_session_id_does_not_merge_users(wired: Wired, seed_pair: Callable[..., str]) -> None:
    """两个 user 用**同一个 `session_id`**（本 fixture 固定为 `s1`）也不得互相看见。"""
    alice = seed_pair(wired.store, 0, "a", "a", user_id="alice")
    bob = seed_pair(wired.store, 0, "b", "b", user_id="bob")
    wired.qdrant.by_user = {"alice": [alice], "bob": [bob]}

    got = wired.search(user_id="alice", top_k=10)

    assert {item.id for item in got.items} == {alice}


def test_isolation_is_owned_by_the_candidate_stage_not_the_service_layer(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """⚠ **边界归属**：服务层**不做**二次过滤。

    这里让假 Qdrant 故意"越界"返回别人的 id，服务层会如实打包两条 ⇒
    说明**隔离完全依赖候选阶段的 payload filter**（真 Qdrant 已由 ② 证明生效）。

    **别把这条当成"服务层会兜底"**——它记录的是相反的结论。
    """
    alice = seed_pair(wired.store, 0, "a", "a", user_id="alice")
    bob = seed_pair(wired.store, 0, "b", "b", user_id="bob")
    wired.qdrant.by_user = {"alice": [alice, bob]}  # ← 故意越界

    got = wired.search(user_id="alice", top_k=10)

    assert {item.id for item in got.items} == {alice, bob}
