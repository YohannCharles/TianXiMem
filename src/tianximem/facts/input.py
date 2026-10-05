"""当前输入的字面依赖检查。只判定输入是否齐全，不计算或返回结果。"""

from __future__ import annotations

import json
import re
from collections import defaultdict

_HISTORY = re.compile(
    (
        r"\b(?:remember|earlier|previous(?:ly)?|stored|saved|history|conversation|compare|"
        r"reconcile)\b|\b(?:we (?:agreed|discussed)|as discussed|last time|as before)\b|\b"
        r"(?:usual|existing) (?:policy|rules|schema|format)\b"
    ),
    re.I,
)
_UNDEFINED = re.compile(r"^(?:unknown|redacted|todo|null|none|\?|n/a)(?:\s|$)", re.I)
_VARIABLE = re.compile(r"\$(?:[A-Za-z_]\w*|\{)|\{\{")
_ASSIGNMENT = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*[:=][ \t]*(\S[^\n]*)$", re.M
)
_COLUMN_NAMES = re.compile(
    r"\b(?:columns?|column order)\b[^:\n]{0,80}:\s*([A-Za-z_]\w*(?:[ \t]*,[ \t]*[A-Za-z_]\w*)+)",
    re.I,
)
_RAW = re.compile(r"\braw (?:[A-Za-z-]+\s+){0,4}(?:data|values)\b", re.I)
_JSON_REQUEST = re.compile(
    (
        "\\bwhat is the (?P<property>version(?: number)?) of (?:the )?['\\\""
        "](?P<entity>[A-Za-z0-9_@/.-]+)['\\\"](?: library| package| depende"
        "ncy)? in (?:this|the) (?P<file>[\\w.-]+\\.json)\\s*[?.!]*$"
    ),
    re.I,
)
_FIELD_REQUEST = re.compile(
    (
        "\\b(?:what is|what's|extract|return|output|give)\\b[^\\n]{0,80}['\\\""
        "]([A-Za-z_][A-Za-z0-9_.-]*)['\\\"][^\\n]{0,100}\\b(?:value|host|stri"
        "ng)\\b"
    ),
    re.I,
)
_LABEL_REQUEST = re.compile(
    r"\blist (?:the )?([A-Za-z_]\w*) names(?: found)? in (?:a )?simple bulleted list\s*[?.!]*$",
    re.I,
)
_LABEL = re.compile(r'[ \t]*([A-Za-z_]\w*)[ \t]*=[ \t]*("(?:[^"\\]|\\.)*")[ \t]*(?:,|$)')
_ROW = re.compile(
    r"^[A-Za-z_:][A-Za-z0-9_:]*\{([^{}]*)\}[ \t]+[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:[ \t]+\d+)?$"
)


def _literal(values: set[str]) -> bool:
    if len(values) != 1:
        return False
    value = next(iter(values))
    return bool(value) and not _UNDEFINED.search(value) and not _VARIABLE.search(value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate JSON key")
        value[name] = item
    return value


def _json_values(value: object, entity: str) -> list[object]:
    if not isinstance(value, dict):
        return []
    return [item for name, item in value.items() if name == entity] + [
        item for nested in value.values() for item in _json_values(nested, entity)
    ]


def input_complete(query: str) -> bool:
    """缺项、变量、历史依赖、重复冲突都必须继续检索。"""
    if _HISTORY.search(query):
        return False
    request = _JSON_REQUEST.search(query)
    if request:
        markers = list(
            re.finditer(
                r"\b"
                + re.escape(request["file"])
                + r"(?:[ \t]+(?:fragment|snippet|contents?))?[ \t]*:[ \t]*",
                query,
                re.I,
            )
        )
        if len(markers) != 1:
            return False
        text = query[markers[0].end() :].lstrip()
        if text.startswith("```"):
            text = text.partition("\n")[2]
        try:
            value, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(text)
        except (ValueError, RecursionError):
            return False
        json_values = _json_values(value, request["entity"])
        return (
            not any(c in text[end:] for c in "{[")
            and len(json_values) == 1
            and isinstance(json_values[0], str)
            and _literal({json_values[0]})
            and bool(
                re.fullmatch(
                    r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", json_values[0]
                )
            )
        )
    label_request = _LABEL_REQUEST.search(query)
    if label_request:
        marker = re.search(r"^[ \t]*Raw [A-Za-z ]{1,50}:\s*$", query, re.M | re.I)
        if not marker:
            return False
        text = query[marker.end() :].strip()
        block, separator, rest = text.partition("\n\n")
        if not separator or any(_ROW.fullmatch(line.strip()) for line in rest.splitlines()):
            return False
        selected = False
        target = label_request[1].casefold()
        for line in block.splitlines():
            row = _ROW.fullmatch(line.strip())
            if row is None:
                return False
            labels, position, seen = row[1], 0, set()
            while position < len(labels):
                label = _LABEL.match(labels, position)
                if label is None or label[1] in seen:
                    return False
                seen.add(label[1])
                position = label.end()
                try:
                    value = json.loads(label[2])
                except ValueError:
                    return False
                if label[1].casefold() in {target, target + "_name"}:
                    if not _literal({value}):
                        return False
                    selected = True
        return selected
    raw, columns = list(_RAW.finditer(query)), _COLUMN_NAMES.search(query)
    if (
        raw
        and columns
        and re.search(r"\bcsv\b", query, re.I)
        and re.search(r"\b(?:only|nothing else|no explanation|exactly)\b", query, re.I)
    ):
        values: dict[str, set[str]] = defaultdict(set)
        for name, value in _ASSIGNMENT.findall(query[raw[-1].end() :]):
            values[name].add(value.strip())
        if len(values) < 2 or not all(_literal(items) for items in values.values()):
            return False
        names = [name.strip() for name in columns[1].split(",")]
        return all(name in values for name in names) or (
            len(names) == 2
            and names[0].casefold()
            in {"field", "field_name", "key", "name", "parameter", "parameter_name", "variable"}
            and names[1].casefold() in {"value", "field_value", "parameter_value", "variable_value"}
        )
    field = _FIELD_REQUEST.search(query)
    if field and re.search(r"\b(?:config(?:uration)?|snippet|payload|file)\b", query, re.I):
        values = defaultdict(set)
        for name, value in _ASSIGNMENT.findall(query):
            values[name].add(value.strip())
        return field[1] in values and _literal(values[field[1]])
    return False
