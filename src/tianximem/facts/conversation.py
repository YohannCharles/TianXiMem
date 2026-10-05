"""切分合并块内明确具名的发言，保留来源位置并拒绝明显的引文续行。"""

import re
from collections.abc import Iterator

_NAME = r"[A-Z][A-Za-z'-]*(?: [A-Z][A-Za-z'-]*){0,3}"
_BOUNDARY = re.compile(rf"\n(?=(?:\[assistant\]\s*)?{_NAME}:\s)")
_QUOTE_INTRO = re.compile(r"\b(?:said|says|wrote|quoted|reads|read|text|example)\s*:\s*$", re.I)


def named_utterances(text: str | None) -> Iterator[tuple[int, str]]:
    quoted = False
    previous = ""
    for position, utterance in enumerate(_BOUNDARY.split(text or "")):
        continued_quote = quoted or bool(_QUOTE_INTRO.search(previous))
        quoted ^= utterance.count('"') % 2 == 1
        if utterance.count("“") > utterance.count("”"):
            quoted = True
        elif "”" in utterance and utterance.count("”") >= utterance.count("“"):
            quoted = False
        if not continued_quote:
            yield position, utterance
        previous = utterance
