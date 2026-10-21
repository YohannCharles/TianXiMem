"""查询侧超长输入的**前置截断**——在发送给 embedding 网关之前压进模型窗口。

## 为什么需要

部分输入查询超过 `text-embedding-v4` 的单条窗口（8,192 token）。在本地先截断可避免
网关接收完整请求后再拒绝或静默截断，也避免长请求占用推理槽位。

## 截断形状：头尾各一半

长查询可能是“长文档 + 问题”，问题也可能出现在开头。只保留头部会丢失结尾问题，
只保留尾部会丢失开场设定，因此头尾各保留一半，中间用 `\\n...\\n` 标记。

## 为什么用 `o200k_base` 计数

embedding 模型自己的分词器不在本仓库中；`o200k_base` 是当前统一的分词器实现。
它是近似而非窗口保证：与 Qwen tokenizer 的最坏比例约为 1.70，因此预算取
`EMBED_MAX_TOKENS ÷ 1.70` 并留出余量。在拿到提交模型精确 tokenizer 前，这是保守近似。

## 只给查询侧用

索引侧的正文由打包预算与 `tests/test_datasets.py` 钉在窗口内；那里一旦超窗是**数据 bug**，
应当响亮失败（本地网关 400），**不能靠截断掩盖**。所以 `encode()`（文档侧）不经过这里，
只有 `encode_query()` 会。

## 它不是配置项

窗口是模型的 API 事实，不是调优参数。头尾比例用于覆盖问题位于两端的情况，
也不作为普通调优旋钮暴露。
"""

from __future__ import annotations

from typing import Final

from tianximem.common.tokens import (
    EMBED_MAX_TOKENS,
    O200kCounter,
    TokenCounterError,
    load_counter,
)

__all__ = ["TRUNCATION_MARKER", "truncate_for_embedding"]

#: 被切掉的那一段的占位标记。它本身要花 token，预算里先扣掉。
TRUNCATION_MARKER: Final[str] = "\n...\n"

#: 截断预算占窗口的比例。
#:
#: 不是普通调优值：Qwen tokenizer 与 o200k 的最坏比例约为 1.70，因此取
#: `1 / 1.70 ≈ 0.588` 并留出余量后使用 0.56。本仓库没有提交模型 tokenizer 的精确计数，
#: 所以这是保守近似，不是绝对窗口保证。
_BUDGET_RATIO: Final[float] = 0.56

#: `decode` 的切点可能落在 UTF-8 多字节中间，还原成替换字符后会多花几个 token；
#: 预算里留这点余量，保证结果**严格**不超过上限。
_DECODE_SLACK: Final[int] = 16

#: 默认预算（token）。`truncate_for_embedding(max_tokens=...)` 只给测试与探针用。
_DEFAULT_LIMIT: Final[int] = int(EMBED_MAX_TOKENS * _BUDGET_RATIO)


def _o200k() -> O200kCounter:
    counter = load_counter()
    if not isinstance(counter, O200kCounter):
        raise TokenCounterError(
            "truncate_for_embedding 需要能切分 token id 的 O200kCounter，"
            f"当前拿到的是 {type(counter).__name__}"
        )
    return counter


def truncate_for_embedding(text: str, *, max_tokens: int | None = None) -> str:
    """把一条查询压进 embedding 模型的单条窗口。**未超限时逐字返回**。

    `max_tokens=None` ⇒ 用 `EMBED_MAX_TOKENS × 0.56`（见模块 docstring）。
    """
    limit = _DEFAULT_LIMIT if max_tokens is None else max_tokens
    if limit <= 0:
        raise ValueError(f"max_tokens 必须为正，收到 {limit}")
    counter = _o200k()
    ids = counter.encode_ids(text)
    if len(ids) <= limit:
        return text

    marker_ids = counter.encode_ids(TRUNCATION_MARKER)
    budget = limit - len(marker_ids) - _DECODE_SLACK
    if budget < 2:  # 窗口小到连标记都放不下：只留头部
        return counter.decode_ids(ids[:limit])

    head_n = budget // 2
    tail_n = budget - head_n
    head = counter.decode_ids(ids[:head_n])
    tail = counter.decode_ids(ids[len(ids) - tail_n :])
    return f"{head}{TRUNCATION_MARKER}{tail}"
