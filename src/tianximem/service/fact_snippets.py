from __future__ import annotations

import logging
from collections.abc import Sequence

from tianximem.facts.evidence import EvidenceFact, key
from tianximem.facts.query_entities import EntityQuery, query_entities
from tianximem.llm import LlmClient
from tianximem.store.sqlite_store import SqliteStore

__all__ = ["FactSuffix", "collect_fact_snippets"]

logger = logging.getLogger(__name__)


class FactSuffix:
    __slots__ = ("memory_ids", "snippets", "status")

    def __init__(
        self,
        snippets: tuple[str, ...] = (),
        memory_ids: tuple[str, ...] = (),
        status: str = "skipped",
    ) -> None:
        self.snippets = snippets
        self.memory_ids = memory_ids
        self.status = status

    def __bool__(self) -> bool:
        return bool(self.snippets)


_MAX_FACTS: int = 24


def collect_fact_snippets(
    *,
    store: SqliteStore,
    llm: LlmClient | None,
    user_id: str,
    query: str,
    max_output_tokens: int,
    temperature: float = 0.0,
    limit: int = _MAX_FACTS,
) -> FactSuffix:

    if llm is None or not query.strip():
        return FactSuffix()

    try:
        entities = query_entities(
            query,
            llm=llm,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询实体抽取失败，跳过事实补充：%s: %s", type(exc).__name__, exc)
        return FactSuffix(status="entity_error")

    if isinstance(entities, EntityQuery) and entities.skip:
        return FactSuffix(status=entities.status)

    try:
        facts = _fetch(store, user_id, entities.entities, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("事实检索失败，跳过事实补充：%s: %s", type(exc).__name__, exc)
        return FactSuffix(status="store_error")

    snippets: list[str] = []
    memory_ids: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        text = fact.content.strip()
        if not text:
            continue
        marker = key(text)
        if marker in seen:
            continue
        seen.add(marker)
        snippets.append(text)
        if fact.parent_memory_id not in memory_ids:
            memory_ids.append(fact.parent_memory_id)
        if len(snippets) >= limit:
            break
    if not snippets:
        return FactSuffix(status="empty")
    return FactSuffix(tuple(snippets), tuple(memory_ids), "ok")


def _fetch(
    store: SqliteStore, user_id: str, entities: Sequence[str], limit: int
) -> list[EvidenceFact]:
    with store.read() as conn:
        return store.fetch_facts_for_entities(conn, user_id, entities, limit=limit)
