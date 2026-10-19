"""`official-extra` 各家族的 answer / evaluate **共用底座**（`extra_pipeline.py` 拆出来的第一层）。

这里是**跨家族复用**的东西：所有精确判分共用的归一化（`normalize` / `_contains`）、
判词 JSON 的修复与布尔归一（`_escape_stray_quotes` / `_repair_memtrap_string_escapes` /
`_as_bool`）、模型调用的两个工具与两个超时（`_chat` / `_config` / `ANSWER_TIMEOUT` /
`JUDGE_TIMEOUT`）、通用答案 prompt（`ANSWER_PROMPT`）与判分行形状（`_verdict`）。

## 三层与依赖方向（拆自 `extra_pipeline.py`，2026-10-15）

    extra_base ← extra_<家族>（六个）← extra_pipeline（façade）

家族模块**不许** import façade；façade 把搬走的名字全部重新导出，
`from eval.harness.extra_pipeline import X` 全线照旧。

⚠ **`_chat` / `_config` 定义在这一层，但调用它们的是 façade**（`cmd_answer` / `cmd_evaluate` /
`_call_judge`）——这是刻意的：消费方直接 `monkeypatch.setattr(extra_pipeline, "_chat" / "_config", …)`
把模型调用换成桩，而 bare-name 在**调用处所在模块**的字典里解析 ⇒ 调用留在 façade、补丁才生效。

⚠ **"打 HTTP、不 import `src/`"这条边界在本目录的 [`CLAUDE.md`](./CLAUDE.md)，本文件不重复。**
"""

from __future__ import annotations

import re
import time
import unicodedata

ANSWER_TIMEOUT = 600.0
JUDGE_TIMEOUT = 180.0
_CHAT_RETRY_ATTEMPTS = 3
_CHAT_RETRY_BACKOFF = 5.0


# ── 归一化：所有精确判分共用一份（改它等于改分，别在别处再写一个）───────────────
def normalize(text: object) -> str:
    """NFKC + 转小写 + 空白折叠 + 去掉常见包装标点。"""
    raw = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    return re.sub(r"\s+", " ", raw).strip().lower().strip("。.,;:!?\"'`()[]{}")


def _contains(haystack: str, needle: str) -> bool:
    """`needle` 是否作为**一个词**出现在 `haystack` 里。

    ASCII 词用 `\\b` 边界——否则 MQuAKE 那种长度 2 的别名（`de` = German）会满屏假阳性。
    非 ASCII（中文等）没有词边界，退回子串判断。
    """
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9 .'/-]+", needle):
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


# ── 判词 JSON 的修复与布尔归一：**只在严格解析失败之后**才配用 ─────────────────


def _escape_stray_quotes(text: str) -> str:
    """把**字符串值里**的裸 `"` 转义掉。**只在严格解析已经失败之后才配用它。**

    ## 为什么需要它（2026-10-03 实测）

    裁判写中文理由时会**混用全角开引号与半角闭引号**：

    ```text
    …且未明确引用或体现患者“终身禁用 NSAIDs"这一关键禁忌症信息…
    ```

    那个半角 `"` 把 JSON 字符串**提前截断** ⇒ 整份 JSON 非法 ⇒ 记 `JUDGE_ERROR`
    ——而**判词本身是完整、可读、判断正确的**。上游 `_extract_json_from_text` 的
    括号计数法在这里**同样失效**（它靠 `"` 翻转 `in_string`，撞上这个引号后
    后面的 `}` 被当成了字符串内容），所以这不是"照上游实现"能解决的事。

    ## 判据：这个 `"` 后面是不是结构字符

    在字符串里，`"` 只有在**后面（跳过空白）跟着 `,` `}` `]` `:` 或已到结尾**时
    才是真的收尾引号。`…NSAIDs"这一…` 后面是 `这` ⇒ 它是内容，转义；
    `…推理。"}` 后面是 `}` ⇒ 收尾，照旧。
    """
    out: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and in_string:  # 已转义的序列整个跳过，别二次转义
            out.append(text[index : index + 2])
            index += 2
            continue
        if char == "”" and in_string:
            look = index + 1
            while look < len(text) and text[look].isspace():
                look += 1
            if look < len(text) and text[look] == "}":
                char = '"'
        if char != '"':
            out.append(char)
            index += 1
            continue
        if not in_string:
            in_string = True
        else:
            look = index + 1
            while look < len(text) and text[look] in " \t\r\n":
                look += 1
            if look >= len(text) or text[look] in ",}]:":
                in_string = False
                out.append(char)
            else:
                out.append('\\"')
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _repair_memtrap_string_escapes(text: str) -> str:
    text = _escape_stray_quotes(text)
    out = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            in_string = not in_string
        if char == "\\" and in_string and index + 1 < len(text):
            following = text[index + 1]
            if following not in '"\\/bfnrtu':
                out.append("\\\\")
            else:
                out.append(char)
            out.append(following)
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _as_bool(value: object) -> bool | None:
    """`is_correct` 归一化。**读不出布尔就交回 `None`（⇒ `JUDGE_ERROR`），不要瞎猜。**

    ⚠ 不归一化会让 `bool("false")` **为真**——裁判偶尔把布尔写成字符串，而那样
    一条**判负**的回答会被静默记成**判对**。宁可响亮地记一条 ERROR。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


# ── 判分行形状：所有家族的判分出口共用这一份（`_verdict`） ─────────────────────


def _verdict(qid: str, ok: bool, why: str) -> dict:
    return {
        "id": qid,
        "is_correct": ok,
        "label": "CORRECT" if ok else "WRONG",
        "judge_response": why,
    }


# ── 模型调用 ────────────────────────────────────────────────────────────────
def _config(which: str) -> tuple[str, str, str]:
    """取端点配置。**与归档 pipeline 同一个来源**：`eval/harness/api_config.py`。

    它在 subprocess 里由 `run_judge` 经 `PYTHONPATH` 注入；单元测试 import 本模块时
    不会走到这里（`answer` / `evaluate` 才需要）。
    """
    try:
        import api_config
    except ImportError as exc:  # pragma: no cover — 只在没按 harness 的方式跑时发生
        raise SystemExit(
            "找不到 api_config（应当由 eval/harness/judge.py 的 run_judge 经 PYTHONPATH 注入）"
        ) from exc
    if which == "judge":
        return api_config.JUDGE_API_BASE, api_config.JUDGE_API_KEY, api_config.JUDGE_MODEL
    return api_config.ANSWER_API_BASE, api_config.ANSWER_API_KEY, api_config.ANSWER_MODEL


def _chat(base: str, key: str, model: str, prompt: str, *, max_tokens: int, timeout: float) -> str:
    """Retry transient transport and server errors; client errors fail immediately."""
    import httpx

    if not base:
        raise SystemExit("端点未配置（AML_BASE_URL）——本脚本不能离线跑模型调用")
    for attempt in range(_CHAT_RETRY_ATTEMPTS):
        try:
            response = httpx.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            if attempt + 1 == _CHAT_RETRY_ATTEMPTS:
                raise
            time.sleep(_CHAT_RETRY_BACKOFF)
    raise AssertionError("chat retry attempts must be positive")


#: 答案 prompt。**我们写的**（官方那 6 份的模板在 `dataset/.upstream/aml/pipeline_*.py`）。
ANSWER_PROMPT = """
You are an assistant that answers a question using ONLY the memories provided below.

Rules:
1. Use only the information in the memories. Do not use outside knowledge.
2. If the memories do not contain the answer, reply exactly: Cannot determine from the memories.
3. Answer concisely — one short sentence or the value itself.

Memories:
{memories}

Question: {question}

Answer:"""
