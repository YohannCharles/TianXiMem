"""查询侧超长输入的**前置截断**——在发送给 embedding 网关之前压进模型窗口。

## 为什么需要（2026-10-20）

官方采集里 CL-Bench 有一组 `query` 远超 `text-embedding-v4` 的单条窗口（8,192 token）。
长度分布、延迟与问句位置的逐项数字见
[`eval/reports/embedding-20261020.md`](../../../eval/reports/embedding-20261020.md)。
适配层此前**原样 POST**：网关要先收完整请求、再 tokenize，然后才拒绝或静默截断——
这段时间里它占着连接与推理槽位，同批的短查询一起排队。

⇒ **在本仓先截断**，不把"等网关拒绝"当成截断点。网关侧的窗口是 API 事实，
不是我们可以协商的超时。

## 截断形状：头尾各一半

CL-Bench 的 query 是"长文档 + 问题"：问号既可能在前 10%（开场即是问题），
也可能在最后 10%（文档 + 结尾提问）。只留头部会丢掉多数问句，只留尾部会丢掉开场设定
⇒ **头尾各保留一半**，中间用 `\\n...\\n` 标记。位置分布见上面那份报告。

## 为什么用 `o200k_base` 计数

embedding 模型自己的分词器不在本仓里（提交链路也不装 `tokenizers`）；
`o200k_base` 是 §6.4 的既定口径，也是本仓唯一的分词器实现。
**它只是近似，不是窗口保证**：报告 §7.3 用官方 Qwen tokenizer 对拍，同一段文本的
token 数最高到 o200k 的 1.70 倍；初版的 0.9 预算截断后仍有 9 条超窗。
⇒ 预算取 `EMBED_MAX_TOKENS ÷ 1.70` 再留一点余量（约 0.56 倍），让**实测的最坏比例**
也落进窗口。拿到 `text-embedding-v4` 自己的 tokenizer 之前，这是可验证的保守近似，不是精确保证。

## 只给查询侧用

索引侧的正文由打包预算与 `tests/test_datasets.py` 钉在窗口内；那里一旦超窗是**数据 bug**，
应当响亮失败（本地网关 400），**不能靠截断掩盖**。所以 `encode()`（文档侧）不经过这里，
只有 `encode_query()` 会。

## 它不是配置项

窗口是模型的 API 事实（与 `rrf.k = 61` 同类），不是调优参数；写进配置只会多一个
"配得出、跑不通"的旋钮。截断策略的头尾比例同理——它的依据是上面那份问句位置分布，
不是可以随手调的权重。
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
#: ⚠ **不是调优值**：报告 §7.3 用官方 Qwen tokenizer 实测，Qwen/o200k 的最坏比例
#: 是 1.70 ⇒ `1 / 1.70 ≈ 0.588`，再留一点余量取 0.56。
#: 它**不保证** `text-embedding-v4` 的 tokenizer 同样安全——那是另一个分词器，
#: 本仓没有它的精确计数；这是"可验证的保守近似"，不是窗口保证。
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
