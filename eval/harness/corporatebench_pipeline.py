"""CorporateBench 的答案格式与纯函数判分；不读取端点、不调用模型。

标量给 exact-match 分，列表给真正的 set-F1。兼容旧运行的自然语言答案，
新答案要求 JSON 值；空集合与证据不足的拒答是不同结果。题目类型可以进入
答案 prompt，金标答案绝不进入。归档 README 的协议见 benchmark_data/corporatebench/。
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

__all__ = ["judge_corporatebench", "score_corporatebench"]


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"\s+", " ", text).strip().strip("。.,;:!?\"'`()")


def _json_value(text: str) -> tuple[bool, object]:
    raw = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return False, text
    if isinstance(value, dict):
        if set(value) != {"answer"}:
            return True, None
        value = value["answer"]
    return True, value


def _refuses(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    text = _norm(value)
    return (
        not text
        or any(
            phrase in text
            for phrase in (
                "cannot determine",
                "can't determine",
                "unable to determine",
                "insufficient information",
                "not enough information",
                "cannot be determined",
            )
        )
        or text in {"unknown", "null", "not known", "not provided"}
    )


def _list_item(text: str) -> str:
    text = text.strip()
    bold = re.match(r"\*\*(.+?)\*\*", text)
    if bold:
        return bold.group(1).strip()
    # 旧答案会在实体后补日期、职位或解释；JSON 数组不经过这个兼容步骤。
    text = re.sub(r"\s+\([^()]*\)(?:\s*[-–—]\s*.*)?\s*$", "", text)
    return text.strip().rstrip(".")


def _list_value(value: object) -> set[str] | None:
    if isinstance(value, list):
        if any(not isinstance(x, str) or not x.strip() for x in value):
            return None
        return {_norm(x) for x in value}
    if not isinstance(value, str) or _refuses(value):
        return None
    text = value.strip()
    if _norm(text) in {
        "none",
        "no matches",
        "no matching items",
        "no matching meetings",
        "no such meetings",
        "no meetings",
        "no employees",
        "no one",
        "nobody",
    }:
        return set()
    bullets = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", text, re.MULTILINE)
    if bullets:
        return {_norm(_list_item(x)) for x in bullets}
    # 去掉旧答案的引导语和谓语，保留实际列出的对象，而不是只在答案里搜金标。
    text = re.sub(
        r"^(?:the\s+)?(?:employees|meetings|participants|attendees|names|items)\b.*?"
        r"\b(?:are|include)\s*:?[ \t]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.split(
        r"\s+(?:work(?:s)?\s+(?:at|for)|attended|participated\s+in|were\s+present\s+at|"
        r"started\s+working\s+at)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    # & 属于会议标题，不是列表分隔符；含 and 的单个会议标题也保持完整。
    if not re.search(r"[,;\n]", text) and re.search(
        r"\b(?:meeting|review|discussion|session|huddle|check-in)\.?$",
        text,
        re.IGNORECASE,
    ):
        return {_norm(_list_item(text))}
    pieces = re.split(r"\s*[,;\n]\s*|\s+and\s+", text)
    cleaned = [re.sub(r"^and\s+|^only\s+", "", x.strip(), flags=re.IGNORECASE) for x in pieces]
    return {_norm(_list_item(x)) for x in cleaned if x.strip()}


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_COUNT_WORD = rf"(?:{_NUMBER}|{'|'.join(_NUMBER_WORDS)})"
_COUNT_NOUN = r"(?:occurrences?|times?|meetings?|events?|departments?|employees?|tasks?|teams?)"


def _number_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str) or _refuses(value):
        return None
    text = _norm(value)
    if re.fullmatch(_NUMBER, text):
        return float(text.replace(",", ""))
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    counts: set[float] = set()
    for token in re.findall(rf"(?<![\w.])({_COUNT_WORD})\s+(?:distinct\s+)?{_COUNT_NOUN}\b", text):
        counts.add(
            float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else float(token.replace(",", ""))
        )
    if re.search(r"\bonce\b", text):
        counts.add(1.0)
    if re.search(r"\btwice\b", text):
        counts.add(2.0)
    if re.search(rf"\bno\s+{_COUNT_NOUN}\b", text):
        counts.add(0.0)
    # 没有明确次数就拒绝解析，不能把日期或会议名里的数字当作答案。
    return next(iter(counts)) if len(counts) == 1 else None


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str) or _refuses(value):
        return None
    text = _norm(value)
    if re.search(r"\bnot\s+(?:true|false|yes|no)\b", text):
        return None
    yes = re.search(r"\b(?:yes|true)\b", text) is not None
    no = re.search(r"\b(?:no|false)\b", text) is not None
    return yes if yes != no else None


def score_corporatebench(generated: str, gold: object) -> tuple[float, str]:
    """标量 EM / 列表 set-F1；兼容文本答案，拒答永远不等于空集合。"""
    payload = gold if isinstance(gold, dict) else {"answer": gold}
    expected = payload.get("answer")
    kind = str(payload.get("answer_type") or "").lower()
    structured, value = _json_value(generated)
    if _refuses(value):
        return 0.0, "CorporateBench：拒答或答案为空"
    if isinstance(expected, list) or kind.startswith("list"):
        if not isinstance(expected, list) or any(not isinstance(x, str) for x in expected):
            return 0.0, "CorporateBench：列表金标类型无效"
        wanted = {_norm(x) for x in expected}
        found = _list_value(value)
        if found is None:
            return 0.0, "CorporateBench：列表答案类型无效"
        matched = len(wanted & found)
        f1 = 1.0 if not wanted and not found else 2 * matched / (len(wanted) + len(found))
        return f1, (
            f"CorporateBench：set-F1={f1:.6f}（命中 {matched}/{len(wanted)}，预测 {len(found)} 项）"
        )
    if isinstance(expected, bool) or kind == "bool":
        truth = expected if isinstance(expected, bool) else _bool_value(expected)
        said = _bool_value(value)
        return float(truth is not None and said is not None and truth == said), (
            f"CorporateBench：期望 {truth}，解析到 {said}"
        )
    if isinstance(expected, (int, float)) or kind in {"int", "float"}:
        wanted_number, found_number = _number_value(expected), _number_value(value)
        ok = (
            wanted_number is not None and found_number is not None and wanted_number == found_number
        )
        return float(ok), f"CorporateBench：期望 {wanted_number}，解析到 {found_number}"
    wanted_text = _norm(expected) if expected is not None else ""
    if not isinstance(value, str) or not wanted_text:
        return 0.0, "CorporateBench：标量答案类型无效"
    answer = _norm(value)
    # 新的 JSON 字符串精确匹配；旧的短句保留带边界的实体匹配。
    ok = answer == wanted_text
    if not structured and not re.search(r"\b(?:not|no)\b", answer):
        ok = ok or re.search(rf"(?<!\w){re.escape(wanted_text)}(?!\w)", answer) is not None
    return float(ok), f"CorporateBench：期望 {wanted_text!r}"


def judge_corporatebench(generated: str, gold: object) -> tuple[bool, str]:
    """保留原二值接口；连续 QA 分由 score_corporatebench 提供。"""
    score, reason = score_corporatebench(generated, gold)
    return score == 1.0, reason


