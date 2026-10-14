"""Compile supported literal question forms into ordered evidence dependencies.

Entity names remain verbatim; uncertain or unparsed forms do not short-circuit
Search. This module does not extract source facts or call a model.
"""

import re

RELATION_NAMES = {
    "author": ("author", "writer"),
    "capital": ("capital",),
    "country of citizenship": ("country of citizenship", "nationality", "citizenship"),
    "country of origin": ("country of origin",),
    "creator": ("creator",),
    "spouse": ("spouse", "husband", "wife"),
    "founded by": ("founder", "founded by"),
    "educated at": ("educated at", "alma mater"),
    "employee": ("employer", "workplace"),
    "head of state": ("head of state", "head of the state"),
    "head of government": ("head of government", "head of the government"),
    "chief executive officer": ("chief executive officer", "CEO"),
    "headquarters location": ("headquarters location", "headquarters", "main office"),
    "original broadcaster": ("original broadcaster",),
    "developer": ("developer",),
    "manufacturer": ("manufacturer", "producer"),
    "director / manager": ("director", "manager", "director / manager"),
    "chairperson": ("chairperson",),
    "place of birth": ("place of birth", "birthplace", "city of birth"),
    "place of death": ("place of death", "location of death"),
    "location of formation": ("location of formation",),
    "languages spoken, written or signed": ("language spoken", "native language", "native tongue"),
    "official language": ("official language",),
    "notable work": (
        "notable work",
        "notable achievement",
        "claim to fame",
        "notable work or achievement",
    ),
    "religion or worldview": ("religion", "faith", "religious affiliation", "worldview"),
    "continent": ("continent",),
    "sport": ("sport",),
    "performer": ("performer",),
    "genre": ("genre", "type of music"),
    "occupation": ("occupation", "field of work", "profession"),
    "child": ("child",),
    "officeholder": ("officeholder",),
}


_ALIASES = sorted(
    ((a, rel) for rel, names in RELATION_NAMES.items() for a in names), key=lambda p: -len(p[0])
)


def noun_path(text: str, depth: int = 0) -> tuple[str, tuple[str, ...]] | None:
    """A nested noun phrase is a literal root followed by ordered dependencies."""
    # Parsing recognizes longer paths so the executor can reject them as a
    # whole at its configured hop limit. A parser miss must not degrade into
    # legacy relation-keyword retrieval of only the first hop.
    if depth > 32:
        return None
    literal = text.strip()
    value = re.sub(r"^(?:(?:the|a|an|current)\s+)+", "", literal, flags=re.I)
    for alias, rel in _ALIASES:
        m = re.fullmatch(re.escape(alias) + r" (?:of|in) (.+)", value, re.I)
        if m:
            nested = noun_path(m[1], depth + 1)
            return (nested[0], (*nested[1], rel)) if nested else None
    citizen = re.fullmatch(
        r"country (?:where|that) (.+?) (?:is a citizen(?: of)?|holds citizenship)", value, re.I
    )
    if citizen:
        nested = noun_path(citizen[1], depth + 1)
        return (nested[0], (*nested[1], "country of citizenship")) if nested else None
    relative_frames = (
        ("author", r"(?:person|individual) who (?:wrote|authored) (.+)"),
        ("creator", r"(?:person|individual) who created (.+)"),
        ("developer", r"(?:company|organization) that developed (.+)"),
        ("employee", r"(?:company|organization) that employs (.+)"),
        ("educated at", r"(?:university|college|school) that educated (.+)"),
        ("educated at", r"(?:university|college|school) where (.+?) was educated"),
        ("educated at", r"(?:university|college|school) that (.+?) attended"),
        ("sport", r"sport associated with (.+)"),
        ("spouse", r"person (?:who is )?married to (.+)"),
    )
    for rel, pattern in relative_frames:
        relative = re.fullmatch(pattern, value, re.I)
        if relative:
            nested = noun_path(relative[1], depth + 1)
            return (nested[0], (*nested[1], rel)) if nested else None
    producer = re.fullmatch(r"company that (?:produced|manufactured) (.+)", value, re.I)
    if producer:
        nested = noun_path(producer[1], depth + 1)
        return (nested[0], (*nested[1], "manufacturer")) if nested else None
    origin = re.fullmatch(
        r"country where (.+?) (?:originated|was created|was produced)", value, re.I
    )
    if origin:
        nested = noun_path(origin[1], depth + 1)
        return (nested[0], (*nested[1], "country of origin")) if nested else None
    named_office = re.fullmatch(r"entity known as (.+)", value, re.I)
    if named_office and re.search(
        r"\b(?:secretary[- ]general|president|governor|chairperson|CEO|chief .+ officer)$",
        named_office[1],
        re.I,
    ):
        return named_office[1], ("officeholder",)
    if not value or len(value) > 200 or re.search(r"\b(?:who|where|that|which)\b", value, re.I):
        return None
    return literal, ()


def question_path(text: str) -> tuple[str, tuple[str, ...]] | None:
    frames = (
        (None, r"(?:Who|What) is (.+)"),
        (None, r"What is the name of (.+)"),
        ("country of citizenship", r"What country is (.+?) a citizen of"),
        ("employee", r"(?:Who|What) does (.+?) work for"),
        ("employee", r"Where (?:does|do) (.+?) work"),
        ("educated at", r"Where was (.+?) educated"),
        (
            "educated at",
            r"(?:At which|What) university did (.+?) (?:receive their education|attend)",
        ),
        ("place of birth", r"Where was (.+?) born"),
        ("place of birth", r"(?:In which|In what|What) city was (.+?) born"),
        ("notable work", r"What (?:notable achievement is|is) (.+?) (?:known for|renowned for)"),
        (None, r"Where is (.+)"),
        (None, r"(?:In what|What) city (?:can you find|houses) (.+)"),
        (None, r"What is the (?:location|name) of (.+)"),
        ("continent", r"Which continent (?:houses|contains) (.+)"),
        ("languages spoken, written or signed", r"(?:Which|What) language does (.+?) speak"),
        (
            "languages spoken, written or signed",
            r"(?:Which|What) language is the native tongue of (.+)",
        ),
        ("religion or worldview", r"To which religion is (.+?) connected"),
        ("religion or worldview", r"What faith is linked to (.+)"),
        ("place of death", r"Where did (.+?) (?:die|pass away)"),
        ("place of death", r"In (?:which|what) city did (.+?) (?:die|meet their demise)"),
        ("educated at", r"Where did (.+?) receive their education"),
        ("educated at", r"Which university educated (.+)"),
        ("employee", r"What organization employs (.+)"),
        ("occupation", r"In what field is (.+?) employed"),
        ("country of origin", r"(?:Which|What) country is credited with creating (.+)"),
        ("country of origin", r"In which country was (.+?) (?:created|originated)"),
        ("country of origin", r"What country is the birthplace of (.+)"),
        ("country of origin", r"Where was (.+?) (?:created|originated)"),
        ("genre", r"What (?:type|style) of music is (?:played by|associated with) (.+)"),
        ("genre", r"What genre of music does (.+?) play"),
        (None, r"(?:Which|What) city (?:is|was) (.+)"),
        (None, r"(?:Which|What) city is home to (.+)"),
        (None, r"In (?:which|what) city is (.+?) located"),
        (None, r"Where is (.+?) located"),
    )
    # Specific wrappers must precede every unconstrained wrapper: otherwise
    # "Where is ... located" binds "located" as part of the root entity.
    general = {r"(?:Who|What) is (.+)", r"Where is (.+)", r"(?:Which|What) city (?:is|was) (.+)"}
    ordered = [frame for frame in frames if frame[1] not in general]
    ordered.extend(frame for frame in frames if frame[1] in general)
    for tail, pattern in ordered:
        m = re.fullmatch(pattern, text, re.I)
        if m:
            parsed = noun_path(m[1])
            if parsed:
                root, steps = parsed
                if pattern in {
                    r"What country is the birthplace of (.+)",
                    r"Where was (.+?) (?:created|originated)",
                } and (not steps or steps[-1] != "sport"):
                    continue
                if tail:
                    steps = (*steps, tail)
                if (
                    pattern
                    in {
                        r"Where is (.+)",
                        r"(?:In what|What) city (?:can you find|houses) (.+)",
                        r"What is the (?:location|name) of (.+)",
                        r"(?:Which|What) city (?:is|was) (.+)",
                        r"(?:Which|What) city is home to (.+)",
                        r"In (?:which|what) city is (.+?) located",
                        r"Where is (.+?) located",
                    }
                    and not text.lower().startswith("what is the name")
                    and (
                        not steps
                        or steps[-1]
                        not in {
                            "headquarters location",
                            "place of birth",
                            "place of death",
                            "location of formation",
                        }
                    )
                ):
                    continue
                if steps:
                    return root, steps
    return None
