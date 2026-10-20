"""Search stages that are independent from HTTP and candidate packaging.

The service pipeline owns ordering.  This module owns the bounded, SQLite-backed
grounded-evidence operation so it can be tested and evolved without making the
main orchestration class larger.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.query import FactPattern, QueryPlan
from tianximem.pairing.apply import index_pair_facts
from tianximem.retrieve.evidence import (
    EvidenceDecision,
    evaluate_evidence,
    walk_source_literals,
    walk_sources_supported,
    walk_subjects,
)
from tianximem.store.sqlite_store import QaPair, SqliteStore


class GroundedEvidenceStage:
    """Select complete, source-backed facts from one consistent SQLite snapshot."""

    def __init__(
        self,
        *,
        store: SqliteStore,
        evidence_limit: int,
        evidence_hop_limit: int,
        fact_backfill_limit: int,
    ) -> None:
        if fact_backfill_limit <= 0:
            raise ValueError("fact_backfill_limit 必须为正")
        self._store = store
        self._evidence_limit = evidence_limit
        self._evidence_hop_limit = evidence_hop_limit
        self._fact_backfill_limit = fact_backfill_limit

    @contextmanager
    def snapshot(self, user_id: str) -> Iterator[sqlite3.Connection | None]:
        """Backfill this evidence version once, then expose a complete snapshot."""
        with self._store.read_snapshot() as conn:
            if self._store.has_fact_index_coverage(conn, user_id, version=EVIDENCE_VERSION):
                yield conn
                return
        with self._store.transaction() as conn:
            for pair in self._store.fetch_unindexed_pairs(
                conn,
                user_id,
                version=EVIDENCE_VERSION,
                limit=self._fact_backfill_limit,
            ):
                index_pair_facts(self._store, conn, pair)
        with self._store.read_snapshot() as conn:
            yield (
                conn
                if self._store.has_fact_index_coverage(conn, user_id, version=EVIDENCE_VERSION)
                else None
            )

    def decide(
        self,
        user_id: str,
        plan: QueryPlan,
        *,
        evaluator=evaluate_evidence,
    ) -> EvidenceDecision:
        """Run fact selection and source auditing without partial-result fallbacks."""
        fetch_limit = self._evidence_limit * 8
        patterns = plan.patterns
        if plan.resolve_references:
            patterns = tuple(FactPattern(subject=p.subject) for p in patterns)
        if plan.operator == "walk":
            patterns = (*patterns, FactPattern(("explicit replacement",)))
        with self.snapshot(user_id) as conn:
            if conn is None:
                return EvidenceDecision(None, "index_incomplete")
            evidence = self._store.fetch_evidence(
                conn,
                user_id,
                patterns,
                limit=fetch_limit + 1,
                # 逐次取证的计划（"出现几次"、标题清单）靠**重复的行**计数；
                # 按三元组去重会把跨来源的多次出现压成一条。
                deduplicate=not plan.per_occurrence,
            )
            if len(evidence) > fetch_limit:
                return EvidenceDecision(None, "fetch_limit")
            # 读取期已把第一人称事实的主语换成真名，计划侧仍是字面 `"i"`。
            # 同一个读快照里取一次真名键，交给 `matches` 做等价比较。
            owner = self._store.owner_name(conn, user_id)
            sources: list[QaPair] = []
            audit: list = []
            if plan.guard_literal:
                source_scan_limit = (
                    fetch_limit if plan.guard_assertion_pattern else self._evidence_limit
                )
                by_id: dict[str, QaPair] = {}
                for literal in (plan.guard_literal, *plan.guard_source_literals):
                    for source in self._store.fetch_evidence_sources(
                        conn,
                        user_id,
                        literal,
                        limit=source_scan_limit + 1,
                    ):
                        by_id.setdefault(source.id, source)
                    if len(by_id) > source_scan_limit:
                        return EvidenceDecision(None, "source_scan_limit")
                sources = list(by_id.values())
                if sources:
                    audit = self._store.fetch_evidence(
                        conn,
                        user_id,
                        (FactPattern(source_ids=tuple(p.id for p in sources)),),
                        limit=fetch_limit + 1,
                        deduplicate=False,
                    )
                    if len(audit) > fetch_limit:
                        return EvidenceDecision(None, "audit_limit")
            decision = evaluator(
                evidence,
                plan,
                source_limit=self._evidence_limit,
                hop_limit=self._evidence_hop_limit,
                sources=sources,
                audit_facts=audit,
                owner=owner,
            )
            if (
                plan.operator != "walk"
                or not plan.require_complete_walk
                or decision.selection is None
            ):
                return decision
            subjects = walk_subjects(evidence, decision.selection, plan, owner=owner)
            related: dict[str, QaPair] = {}
            for literal in walk_source_literals(subjects):
                for source in self._store.fetch_evidence_sources(
                    conn,
                    user_id,
                    literal,
                    limit=fetch_limit + 1,
                ):
                    related[source.id] = source
                if len(related) > fetch_limit:
                    return EvidenceDecision(None, "source_scan_limit")
            audit = (
                self._store.fetch_evidence(
                    conn,
                    user_id,
                    (FactPattern(source_ids=tuple(related)),),
                    limit=fetch_limit + 1,
                    deduplicate=False,
                )
                if related
                else []
            )
            if len(audit) > fetch_limit:
                return EvidenceDecision(None, "audit_limit")
            if not walk_sources_supported(
                tuple(related.values()), decision.selection, audit, subjects
            ):
                return EvidenceDecision(None, "unsupported_source")
            return decision
