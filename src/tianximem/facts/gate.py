from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Final

from tianximem.facts.domains import DOMAIN_TERMS, normalize

__all__ = ["DEFAULT_MIN_SCORE", "Score", "score_record"]

DEFAULT_MIN_SCORE: Final[int] = 3

_HAN: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class Score:
    value: int

    terms: tuple[str, ...]

    threshold: int = DEFAULT_MIN_SCORE

    @property
    def admitted(self) -> bool:
        return self.value >= self.threshold


@cache
def _latin_matcher(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![0-9a-z])" + re.escape(term) + r"(?![0-9a-z])")


def score_record(text: str, *, threshold: int = DEFAULT_MIN_SCORE) -> Score:

    folded = normalize(text or "")
    if not folded:
        return Score(0, (), threshold)
    hits: list[str] = []
    for term in sorted(DOMAIN_TERMS, key=len, reverse=True):
        if _HAN.search(term):
            if term in folded:
                hits.append(term)
        elif _latin_matcher(term).search(folded):
            hits.append(term)
    return Score(len(hits), tuple(hits), threshold)
