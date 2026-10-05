"""切分合并块内明确具名的发言，保留来源位置并拒绝明显的引文续行。"""

import re
from collections.abc import Iterator

NAME = r"(?:[A-Z][A-Za-z'-]*(?: [A-Z][A-Za-z'-]*){0,3}|[\u4e00-\u9fff·]{2,16})"
SPEAKER = re.compile(rf"^(?:\[assistant\]\s*)?(?P<actor>{NAME})[:：]\s*(?P<body>.+)$", re.S)
_BOUNDARY = re.compile(rf"\n(?=(?:\[assistant\]\s*)?{NAME}[:：])")
_QUOTE_INTRO = re.compile(
    r"(?:\b(?:said|says|wrote|quoted|reads|read|text|example)|说|说道|写道|引用|示例|例子)"
    r"\s*[:：]\s*$",
    re.I,
)


def sentences(text: str) -> list[str]:
    """仅切显式句末；返回的句子仍是原文子串。"""
    return re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", text)


def named_utterances(text: str | None) -> Iterator[tuple[int, str]]:
    ascii_quoted = False
    quote_depth = {("“", "”"): 0, ("‘", "’"): 0}
    previous = ""
    for position, utterance in enumerate(_BOUNDARY.split(text or "")):
        continued_quote = (
            ascii_quoted or any(quote_depth.values()) or bool(_QUOTE_INTRO.search(previous))
        )
        ascii_quoted ^= utterance.count('"') % 2 == 1
        for opening, closing in quote_depth:
            quote_depth[(opening, closing)] = max(
                0,
                quote_depth[(opening, closing)]
                + utterance.count(opening)
                - utterance.count(closing),
            )
        if not continued_quote:
            yield position, utterance
        previous = utterance
