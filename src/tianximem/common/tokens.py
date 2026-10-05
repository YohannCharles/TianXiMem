"""Token 计数 —— **必须用答案模型自己的分词器**（§6.4）。

> §6.4 逐字：token 用答案模型自己的分词器（`gpt-4o-mini` 的 `o200k_base`），
> **不要用字符数或空格切分近似**——**单次近似偏差会在 100 个对上被放大到几千 token**。

## 这个模块只回答一个问题

**"这段文本是多少 token"** —— 它**不决定截谁**。"槽位数 + token 数"的**双预算决策**在
[`../rank/`](../rank/)（§10 / §11.3）——那样切分才不需要知道分词器是什么。

## ⚠ 一条 R1 类风险（§6.4 / E7）

本地开发期用的是 qwen3.5-9b，**它的分词器与 `gpt-4o-mini` 不同** ⇒
**本地量出的"能装多少对"不能直接搬到线上**。§12.1 的四条对冲里**没有这一条**，
PRD §6.4 明确"应补上"：**Step 5 切换后必须重新量一次单请求实际返回的对数**
（见 `docs/open-questions.md` E7）。本模块的**实现**不会因此变——变的只是"预算够装几条"这个结论。

## 为什么计数发生在**拼接之后**

预算必须对**真实送出去的那个字符串**计数：段内各对之间的连接符、`Q:` / `A:` 行首标记，
**都是要花 token 的字符**。先算各对的 token 再相加会漏掉连接符——
在 100 个对上，一个连接符的偏差就是 100 个 token。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Protocol

__all__ = [
    "DEFAULT_TOKENIZER",
    "MAX_INPUT_TOKENS",
    "O200kCounter",
    "TokenCounter",
    "TokenCounterError",
    "load_counter",
]

#: `gpt-4o-mini` 的分词器（§2.3 规定提交期只能用 `gpt-4o-mini`）。
DEFAULT_TOKENIZER: Final[str] = "o200k_base"

#: 答案窗口 128k 扣掉输出与安全余量后剩下的 input 预算（§2.2 / §6.4）。
#: ⚠ **A 类外部契约常量**：AML 定的，**我们无权改**；写进配置只为追溯，不是为了调。
#: 见 config-reference §7（`budget.max_tokens`）。
MAX_INPUT_TOKENS: Final[int] = 117_760


class TokenCounterError(RuntimeError):
    """分词器加载失败。

    ⚠ **必须响亮**：分词器拿不到就没有预算可言，而"没有预算"会退化成
    "把所有证据都塞进去"——那会让排在后面的证据**整段作废**（§2.2），而**不报错**。
    """


class TokenCounter(Protocol):
    """数 token 的东西。**协议只有这一个方法**——本层不决定截谁。"""

    @property
    def name(self) -> str:
        """分词器标识（进 run record 的配置指纹——换分词器会改变能装几条）。"""
        ...

    def count(self, text: str) -> int:
        """`text` 有多少个 token。**空串是 0**。"""
        ...


class O200kCounter:
    """`o200k_base`——用 `tiktoken` 的真分词器，**不是近似**。"""

    def __init__(self, encoding_name: str = DEFAULT_TOKENIZER) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover — 依赖缺失才走到
            raise TokenCounterError(
                "tiktoken 没装上，无法数 token。它是 `[project] dependencies` 里的一项。"
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # noqa: BLE001 — tiktoken 首次使用要联网取 BPE 文件
            raise TokenCounterError(
                f"加载分词器 {encoding_name!r} 失败：{type(exc).__name__}: {exc}\n"
                "  ⚠ `tiktoken` **第一次使用会联网下载** BPE 文件并缓存到本地；"
                "之后离线可用。离线机器上请先在有网时预热一次。"
            ) from exc
        self._name = encoding_name

    @property
    def name(self) -> str:
        return self._name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))


@lru_cache(maxsize=4)
def _cached(encoding_name: str) -> O200kCounter:
    # 加载一次约 0.1–1s（解析 BPE 文件）；一次请求里会数很多次，所以必须缓存。
    return O200kCounter(encoding_name)


def load_counter(encoding_name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    """取一个（**进程内共享的**）分词器。同名多次调用拿到同一个实例。"""
    return _cached(encoding_name)
