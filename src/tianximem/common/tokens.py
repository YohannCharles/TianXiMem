"""Token 计数，使用答案模型自己的分词器（`o200k_base`）。

不要用字符数或空格切分近似；单次近似误差会在长上下文中累积。

## 职责

本模块只回答“这段文本有多少 token”，不决定截断哪些内容。槽位与 token 的双预算
由 [`../rank/`](../rank/) 负责。

## Model-Switch Risk

Development and submission models may use different tokenizers. A context budget measured
with one model is not guaranteed to fit another. Re-measure the effective budget whenever
the answer model or tokenizer changes; this module's implementation remains unchanged.

## 为什么计数发生在**拼接之后**

预算必须对**真实送出去的那个字符串**计数：段内各对之间的连接符、`Q:` / `A:` 行首标记，
**都是要花 token 的字符**。先算各对的 token 再相加会漏掉连接符——
在 100 个对上，一个连接符的偏差就是 100 个 token。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final, Protocol

__all__ = [
    "DEFAULT_TOKENIZER",
    "EMBED_MAX_TOKENS",
    "LLM_MAX_INPUT_TOKENS",
    "MAX_INPUT_TOKENS",
    "O200kCounter",
    "TokenCounter",
    "TokenCounterError",
    "load_counter",
]

#: 提交用答案模型 `gpt-4o-mini` 的 `o200k_base` 分词器。
DEFAULT_TOKENIZER: Final[str] = "o200k_base"

#: 答案窗口 128k 扣除输出与安全余量后的输入预算。
#: 这是 AML 外部契约常量，不是调优参数；见 `docs/config-reference.md` 的
#: `budget.max_tokens`。
MAX_INPUT_TOKENS: Final[int] = 117_760

#: 内部 LLM 提示词的发送上限。
#:
#: 端点总窗口为 128k，输入还要与调用方的输出预算、协议计数和 tokenizer
#: 差异共享空间。110k 为这些差异留出约 18k 余量；超过时客户端只发送
#: 不超此上限的最长 token 前缀，而不是发起一个已知会失败的请求。
LLM_MAX_INPUT_TOKENS: Final[int] = 110_000

#: **提交口径的嵌入窗口**：`text-embedding-v4` 单条文本最多 **8,192 token**
#: （官方 FAQ 原文：`Each text can contain at most 8,192 tokens`）。
#:
#: ⚠ **它比窗口本身更值得记住的是超限行为**：官方原文是
#: `Content exceeding this limit is truncated before embedding`——
#: **静默截断，不报错**。本地那个开发网关（vLLM）同窗口但会**响亮 400**，
#: 所以本地这一侧的响亮失败是**门禁**：线上不会有这个错。
#:
#: ⇒ **"一个记忆块不能超这个数"这条线只能由服务自己守**：摄入批大小、打包预算
#: 与网关的超窗错误共同约束它。
#: `embed/truncate.py` 在查询发送前使用该窗口截断超长查询；部分官方查询会超过它。
EMBED_MAX_TOKENS: Final[int] = 8_192


class TokenCounterError(RuntimeError):
    """分词器加载失败。

    分词器加载失败必须响亮报错，否则预算无法执行，排在后面的证据可能被静默截断。
    """


class TokenCounter(Protocol):
    """数 token 的东西。**协议只有这一个方法**——本层不决定截谁。"""

    @property
    def name(self) -> str:
        """分词器标识（换分词器会改变"能装几条"）。

        ⚠ **当前无消费方**——两个实现都提供它，但没有任何地方读。
        要拿它做 run record 的配置指纹时得先真的接上（`budget.tokenizer` 是另一个家）。
        """
        ...

    def count(self, text: str) -> int:
        """`text` 有多少个 token。**空串是 0**。"""
        ...

    def encode_ids(self, text: str) -> list[int]:
        """把文本编码为 token ID；供发送前按精确 token 边界取前缀。"""
        ...

    def decode_ids(self, ids: Sequence[int]) -> str:
        """把 token ID 解码为文本。"""
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
        return len(self.encode_ids(text))

    def encode_ids(self, text: str) -> list[int]:
        """把文本编成 token id 序列。

        ⚠ 这是给**输入适配**用的底层能力（[`../embed/truncate.py`](../embed/truncate.py)
        在发送前把超长查询压进模型窗口），**不是**预算决策——"截谁"仍归 `rank/`。
        """
        if not text:
            return []
        return self._encoding.encode(text, disallowed_special=())

    def decode_ids(self, ids: Sequence[int]) -> str:
        """把 token id 序列还原成文本。切点落在 UTF-8 多字节中间时会出现替换字符。"""
        return self._encoding.decode(list(ids))


@lru_cache(maxsize=4)
def _cached(encoding_name: str) -> O200kCounter:
    # 加载一次约 0.1–1s（解析 BPE 文件）；一次请求里会数很多次，所以必须缓存。
    return O200kCounter(encoding_name)


def load_counter(encoding_name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    """取一个（**进程内共享的**）分词器。同名多次调用拿到同一个实例。"""
    return _cached(encoding_name)
