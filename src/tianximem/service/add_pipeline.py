from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

from tianximem.common.render import render_pair
from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.llm_extract import ExtractionRecord, ModelExtractor
from tianximem.pairing import AddBatch, ApplyBatchResult, apply_batch
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = ["AddOutcome", "AddPipeline"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AddOutcome:
    applied: bool

    new_pair_count: int

    repaired: bool


class AddPipeline:
    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        embedder: object,
        inject_abs_time: bool = False,
        extractor: ModelExtractor | None = None,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._embedder = embedder
        self._inject_abs_time = inject_abs_time
        self._extractor = extractor

    @property
    def extractor(self) -> ModelExtractor | None:
        return self._extractor

    def apply(self, batch: AddBatch) -> AddOutcome:
        result = apply_batch(self._store, batch)
        if result.applied:
            self._extract_facts(self._pairs_by_ids(result.new_pair_ids))
        else:
            self._extract_facts(self._uncovered_pairs(batch))
        pairs, repaired = self._pairs_to_index(batch, result)
        if pairs:
            self._qdrant.index_pairs(
                pairs,
                self._embedder,
                renderer=partial(render_pair, inject_abs_time=self._inject_abs_time),
            )
        return AddOutcome(
            applied=result.applied,
            new_pair_count=result.new_pair_count,
            repaired=repaired,
        )

    def _extract_facts(self, pairs: Sequence[QaPair]) -> None:

        if self._extractor is None or not pairs:
            return
        try:
            records = self._records(pairs)
            if not records:
                return
            facts = self._extractor.extract(records)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "事实抽取失败，本次写入不含事实层：原因=%s: %s",
                type(exc).__name__,
                exc,
            )
            return
        if not facts:
            return
        try:
            with self._store.transaction() as conn:
                self._store.insert_evidence(conn, facts)
                indexed = {fact.parent_memory_id for fact in facts}
                for pair in pairs:
                    if pair.id in indexed:
                        self._store.mark_fact_index_coverage(conn, pair, version=EVIDENCE_VERSION)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "事实写入失败，本次写入不含事实层：原因=%s: %s",
                type(exc).__name__,
                exc,
            )

    def _pairs_by_ids(self, pair_ids: Sequence[str]) -> list[QaPair]:
        if not pair_ids:
            return []
        with self._store.read() as conn:
            return self._store.fetch_pairs_by_ids(conn, list(pair_ids))

    def _uncovered_pairs(self, batch: AddBatch) -> list[QaPair]:
        with self._store.read() as conn:
            siblings = self._store.fetch_by_request(
                conn, batch.user_id, batch.session_id, batch.request_id
            )
            if not siblings:
                return []
            pending = self._store.fetch_unindexed_pairs(
                conn, batch.user_id, version=EVIDENCE_VERSION, limit=len(siblings) * 8
            )
            pending_ids = {pair.id for pair in pending}
            return [pair for pair in siblings if pair.id in pending_ids]

    def _records(self, pairs: Sequence[QaPair]) -> tuple[ExtractionRecord, ...]:

        records: list[ExtractionRecord] = []
        with self._store.read() as conn:
            for pair in pairs:
                messages = self._store.fetch_pair_messages(conn, pair.id)
                if messages:
                    records.extend(
                        ExtractionRecord(
                            parent_memory_id=pair.id,
                            user_id=pair.user_id,
                            source_index=message.source_index,
                            role=message.role,
                            content=message.content,
                            timestamp=message.timestamp,
                        )
                        for message in messages
                    )
                    continue
                if pair.question:
                    records.append(
                        ExtractionRecord(
                            parent_memory_id=pair.id,
                            user_id=pair.user_id,
                            source_index=0,
                            role="user",
                            content=pair.question,
                            timestamp=pair.event_time,
                        )
                    )
                if pair.answer:
                    records.append(
                        ExtractionRecord(
                            parent_memory_id=pair.id,
                            user_id=pair.user_id,
                            source_index=1,
                            role="assistant",
                            content=pair.answer,
                            timestamp=pair.event_time,
                        )
                    )
        return tuple(records)

    def _pairs_to_index(
        self, batch: AddBatch, result: ApplyBatchResult
    ) -> tuple[list[QaPair], bool]:
        if not result.applied:
            return self._batch_pairs(batch), True

        ids = list(result.new_pair_ids)
        if not ids:
            return [], False
        with self._store.read() as conn:
            return self._store.fetch_pairs_by_ids(conn, ids), False

    def _batch_pairs(self, batch: AddBatch) -> list[QaPair]:
        with self._store.read() as conn:
            return self._store.fetch_by_request(
                conn, batch.user_id, batch.session_id, batch.request_id
            )
