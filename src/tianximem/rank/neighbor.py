from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from tianximem.common.annotate import annotate
from tianximem.common.config import DEFAULT_EXPANSION_SEED_LIMIT, DEFAULT_RADIUS
from tianximem.common.render import event_day, render, render_date, render_segment
from tianximem.common.tokens import TokenCounter
from tianximem.retrieve.fusion import Candidate
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "DEFAULT_EXPANSION_SEED_LIMIT",
    "DEFAULT_RADIUS",
    "ContextSegment",
    "ExpansionResult",
    "SelectedMemory",
    "expand_neighbors",
    "merge_segments",
]


@dataclass(frozen=True, slots=True)
class SelectedMemory:
    memory_id: str
    pair: QaPair
    rerank_rank: int | None
    is_neighbor: bool

    @property
    def local_index(self) -> int:
        return self.pair.local_index

    @property
    def session_key(self) -> tuple[str, str]:
        return (self.pair.user_id, self.pair.session_id)

    @property
    def chain_key(self) -> tuple[str, str, str]:
        return (self.pair.user_id, self.pair.session_id, self.pair.request_id)


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    selected: tuple[SelectedMemory, ...]

    neighbors_added: int

    missing_rows: int


def expand_neighbors(
    ranked: list[Candidate],
    *,
    store: SqliteStore,
    seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT,
    radius: int = DEFAULT_RADIUS,
) -> ExpansionResult:
    if seed_limit < 0:
        raise ValueError(f"seed_limit 不得为负：{seed_limit}")
    if radius < 0:
        raise ValueError(f"radius 不得为负：{radius}")

    by_id: dict[str, Candidate] = {}
    for candidate in ranked:
        by_id.setdefault(candidate.memory_id, candidate)

    if not by_id:
        return ExpansionResult(selected=(), neighbors_added=0, missing_rows=0)

    with store.read() as conn:
        pairs = store.fetch_pairs_by_ids(conn, list(by_id))
        pair_by_id: dict[str, QaPair] = {p.id: p for p in pairs}

        known: dict[str, QaPair] = dict(pair_by_id)
        selected: dict[str, SelectedMemory] = {}
        for memory_id, candidate in by_id.items():
            pair = pair_by_id.get(memory_id)
            if pair is None:
                continue
            selected[memory_id] = SelectedMemory(
                memory_id=memory_id,
                pair=pair,
                rerank_rank=candidate.rank,
                is_neighbor=False,
            )

        seeds = sorted((c for c in by_id.values() if c.rank < seed_limit), key=lambda c: c.rank)
        added = 0
        frontier: set[str] = {c.memory_id for c in seeds if c.memory_id in selected}
        for _ in range(radius):
            wanted: dict[str, None] = {}
            for memory_id in sorted(frontier):
                pair = known.get(memory_id)
                if pair is None:
                    continue
                for neighbour in (pair.prev_memory_id, pair.next_memory_id):
                    if neighbour is not None and neighbour not in selected:
                        wanted.setdefault(neighbour, None)
            if not wanted:
                break

            rows = store.fetch_pairs_by_ids(conn, list(wanted))
            for pair in rows:
                known[pair.id] = pair
                if pair.id in selected:
                    continue
                selected[pair.id] = SelectedMemory(
                    memory_id=pair.id,
                    pair=pair,
                    rerank_rank=None,
                    is_neighbor=True,
                )
                added += 1
            frontier = {pair.id for pair in rows}

    return ExpansionResult(
        selected=tuple(selected.values()),
        neighbors_added=added,
        missing_rows=len(by_id) - len(pair_by_id),
    )


@dataclass(frozen=True, slots=True)
class ContextSegment:
    source_memory_ids: tuple[str, ...]

    user_id: str
    session_id: str
    request_id: str

    start_local_index: int
    end_local_index: int

    anchor_memory_id: str

    best_rank: int

    anchor_event_time: int | None

    content: str

    token_count: int

    rerank_member_count: int

    @property
    def length(self) -> int:
        return len(self.source_memory_ids)


def merge_segments(
    selected: tuple[SelectedMemory, ...] | list[SelectedMemory],
    *,
    counter: TokenCounter,
    inject_abs_time: bool = False,
    seed_placement: str = "keep",
    annotate_relatives: bool = False,
) -> list[ContextSegment]:
    groups: dict[tuple[str, str, str], list[SelectedMemory]] = {}
    for memory in selected:
        groups.setdefault(memory.chain_key, []).append(memory)

    segments: list[ContextSegment] = []
    for (user_id, session_id, request_id), members in groups.items():
        members.sort(key=lambda m: m.local_index)

        runs: list[list[SelectedMemory]] = []
        for memory in members:
            previous = runs[-1][-1] if runs else None
            if previous is not None and previous.pair.next_memory_id == memory.memory_id:
                runs[-1].append(memory)
            else:
                runs.append([memory])

        for run in runs:
            segments.append(
                _build_segment(
                    run,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    counter=counter,
                    inject_abs_time=inject_abs_time,
                    seed_placement=seed_placement,
                    annotate_relatives=annotate_relatives,
                )
            )

    segments.sort(key=lambda s: s.best_rank)
    return segments


def _build_segment(
    run: list[SelectedMemory],
    *,
    user_id: str,
    session_id: str,
    request_id: str,
    counter: TokenCounter,
    inject_abs_time: bool = False,
    seed_placement: str = "keep",
    annotate_relatives: bool = False,
) -> ContextSegment:
    ranked_members = [m for m in run if m.rerank_rank is not None]
    anchor = (
        min(ranked_members, key=lambda m: cast(int, m.rerank_rank)) if ranked_members else run[0]
    )
    best_rank = anchor.rerank_rank if anchor.rerank_rank is not None else _NO_RANK

    ordered = run
    echoed: list[SelectedMemory] = []
    if seed_placement == "front" and anchor in run:
        ordered = [anchor, *(m for m in run if m is not anchor)]
    elif seed_placement == "echo" and anchor in run:
        echoed = [anchor]

    def _text(member: SelectedMemory) -> str:
        text = render(
            member.pair.question,
            member.pair.answer,
            date=render_date(member.pair.event_time, inject_abs_time=inject_abs_time),
        )
        if not annotate_relatives:
            return text
        anchor_day = event_day(member.pair.event_time)
        return text if anchor_day is None else annotate(text, anchor_day)

    content = render_segment([*(_text(m) for m in echoed), *(_text(m) for m in ordered)])

    return ContextSegment(
        source_memory_ids=tuple(m.memory_id for m in run),
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        start_local_index=run[0].local_index,
        end_local_index=run[-1].local_index,
        anchor_memory_id=anchor.memory_id,
        best_rank=best_rank,
        anchor_event_time=anchor.pair.event_time,
        content=content,
        token_count=counter.count(content),
        rerank_member_count=len(ranked_members),
    )


_NO_RANK: Final[int] = 1 << 60
