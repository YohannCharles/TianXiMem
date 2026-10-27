"""CoQ 输出的**两阶段解析 + 严格校验**。

模型回的不一定是干净的 JSON。里它经常先吐一段思考，再吐对象：

```text
<think>
...
</think>

{
  "is_sufficient": false,
  "evidence_indices": [1, 3],
  "new_query": "Where did Alice work?",
  "confidence_score": 0.93
}
```

⇒ **不能**只对整串做一次 `json.loads()`，也**不能**用贪婪正则抠 JSON
（会在字符串里的大括号、转义、嵌套上出错）。本模块因此分两步：

| 阶段 | 做什么 |
| --- | --- |
| ① 取对象 | 先试整串；失败则扫出**括号平衡**的对象，**从后往前**取第一个能解析成 `dict` 的 |
| ② 校验 | 字段**恰好**四个、类型**逐项**严格、索引**不越界**；任何一项不满足 ⇒ 抛异常 |

> ⛔ **"解析出来"与"能用来做决策"是两件事。** 缺字段不是"用默认值补上"，
> 而是**这条输出不可用**——补默认值等于**让模型的一次胡话变成一个看似正常的决策**，
> 而它会静默地改变检索。

⚠ **`extraction` 这个字段要留着**（`direct` / `recovered`）：模型输出的稳定性是
提示词健康的探针，而"要抠一次才拿到"正是它开始退化的第一个信号。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "COQ_DECISION_KEYS",
    "CoqDecision",
    "InvalidCoqOutput",
    "extract_json_object",
    "parse_coq_decision",
]

#: 提示词写死的四个字段。**恰好四个**——多一个少一个都是不可用的输出。
COQ_DECISION_KEYS: Final[frozenset[str]] = frozenset(
    {"is_sufficient", "evidence_indices", "new_query", "confidence_score"}
)


class InvalidCoqOutput(ValueError):
    """模型的输出不能被当作一条决策使用。**调用方必须降级**。

    ⚠ 异常消息里**只有原因，没有原文**：原文是模型读着不可信输入写出来的东西，
    把它塞进日志等于让 Query 里的内容进日志（与 `LlmUnavailable` 同一条纪律）。
    """


@dataclass(frozen=True, slots=True)
class CoqDecision:
    """一条**通过全部校验**的 CoQ 决策。"""

    is_sufficient: bool
    evidence_indices: tuple[int, ...]
    new_query: str
    confidence_score: float
    #: `direct` = 整串就是 JSON；`recovered` = 从文本中提取得到。
    extraction: str
    #: 模型的原始输出。**只用于诊断**，不进响应。
    raw: str


def _balanced_objects(text: str) -> list[tuple[int, int]]:
    """扫出所有**括号平衡**的 `{...}` 区间（含嵌套），按出现顺序返回。

    正确处理两件容易写错的事：

    * **字符串里的大括号**（`{"new_query": "the {x} thing"}`）——进入字符串后
      只找结束引号，`{}` 一律不计数；
    * **转义**（`"a \\" }"`）——`\\` 之后的那个字符不参与定界。
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, index))
    return spans


def extract_json_object(text: str) -> tuple[dict[str, Any], str] | None:
    """阶段①：拿到一个 JSON 对象，以及**是怎么拿到的**（`direct` / `recovered`）。

    先试整串，再从后往前试各个平衡区间。**从后往前**是因为提示词要求
    "只输出 JSON"，但模型常把思考写在前面——最后那个对象才是它的结论。
    """
    stripped = text.strip()
    if stripped:
        try:
            direct = json.loads(stripped)
        except ValueError:
            direct = None
        if isinstance(direct, dict):
            return direct, "direct"

    spans = _balanced_objects(text)
    for start, end in reversed(spans):
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed, "recovered"
    return None


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    # ⚠ `bool` 是 `int` 的子类，所以**必须先查 bool**——否则 `1` / `0` 会被当成
    #   合法的布尔值收下。反过来，索引那边必须**排除** bool（见 `_require_indices`）。
    value = payload[key]
    if not isinstance(value, bool):
        raise InvalidCoqOutput(f"`{key}` 必须是布尔，收到 {type(value).__name__}")
    return value


def _require_indices(payload: dict[str, Any], key: str, *, document_count: int) -> tuple[int, ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise InvalidCoqOutput(f"`{key}` 必须是数组，收到 {type(value).__name__}")
    indices: list[int] = []
    for item in value:
        # ⚠ **排除 bool**：`True` 在 Python 里是 `int` 的实例，不排除的话
        #   `[true]` 会被读成索引 `[1]`——**布尔值不该通过整数索引的校验**。
        if isinstance(item, bool) or not isinstance(item, int):
            raise InvalidCoqOutput(f"`{key}` 里必须是整数，收到 {type(item).__name__}")
        if item < 0 or item >= document_count:
            # **越界即异常**，不是"忽略这一条"：忽略会让"模型指着不存在的证据说充分"
            # 变成一个看起来正常的决策。
            raise InvalidCoqOutput(f"`{key}` 越界：{item}（本轮只有 {document_count} 篇文档）")
        if item not in indices:
            indices.append(item)
    return tuple(indices)


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidCoqOutput(f"`{key}` 必须是字符串，收到 {type(value).__name__}")
    if "\n" in value or "\r" in value:
        # 提示词要求 single-line。多行不是"格式小瑕疵"——它会被当成**要执行的检索词**，
        # 而检索侧的每一次调用都只接受一条查询。
        raise InvalidCoqOutput(f"`{key}` 必须是单行")
    return value


def _require_score(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCoqOutput(f"`{key}` 必须是数字，收到 {type(value).__name__}")
    score = float(value)
    if not math.isfinite(score):
        raise InvalidCoqOutput(f"`{key}` 必须是有限数字")
    if not 0.0 <= score <= 1.0:
        raise InvalidCoqOutput(f"`{key}` 必须在 0..1 之间，收到 {score}")
    return score


def parse_coq_decision(text: str, *, document_count: int) -> CoqDecision:
    """阶段①②：从模型输出里拿到一条**可用的**决策，否则抛 `InvalidCoqOutput`。

    `document_count` 是**本轮真正发给模型的文档数**——`evidence_indices` 的合法上界
    由它决定。⚠ 这个参数刻意是必填的：把"上界"写成常量或省略，就等于放弃了
    "索引必须落在我发出去的那份列表里"这条不变式。
    """
    found = extract_json_object(text)
    if found is None:
        raise InvalidCoqOutput("响应里找不到 JSON 对象")
    payload, extraction = found

    keys = set(payload)
    if keys != COQ_DECISION_KEYS:
        missing = sorted(COQ_DECISION_KEYS - keys)
        extra = sorted(keys - COQ_DECISION_KEYS)
        raise InvalidCoqOutput(f"字段不匹配：缺少 {missing}，多出 {extra}")

    return CoqDecision(
        is_sufficient=_require_bool(payload, "is_sufficient"),
        evidence_indices=_require_indices(
            payload, "evidence_indices", document_count=document_count
        ),
        new_query=_require_str(payload, "new_query"),
        confidence_score=_require_score(payload, "confidence_score"),
        extraction=extraction,
        raw=text,
    )
