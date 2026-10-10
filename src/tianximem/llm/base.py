"""LLM 协议与"可降级"信号（§12.1 R1 的接缝本体）。

```text
【本模块在这】  Reranker / Embedder 都有各自的可降级信号，LLM 这一半是 LlmUnavailable
```

* [`LlmClient`][LlmClient] —— 协议。**只收一段提示词、只回一段文本**，不知道它被用来做什么
* [`LlmUnavailable`][LlmUnavailable] —— "可降级"的唯一信号（与 `RerankUnavailable` 同形）

## 为什么协议只有 `complete()` 一个方法

`llm/CLAUDE.md` 写的是"一次调用 / 工具调用"两个方法。**工具调用没有实现**，因为
**它没有消费者**：v1 不做 agentic（D13），而本目录当前的唯一调用点是查询路由器——
它只需要"给一段文本、拿回一段文本"。**只收有代码消费方的入口**（`config-reference.md`
那条对键的纪律，对方法同样成立）；等 `agent/` 真的落地时再加，那时的形状由它决定。

## 为什么是"文本进、文本出"，而不是"消息列表进、结构化对象出"

与 `Reranker` 刻意"只收文本、只回分数"**同一条理由**：把**解析**留在调用方。
解析策略（严格匹配三个枚举值、非法即降级）是**路由器**的契约，不是 LLM 后端的契约——
把它塞进这里，换一个后端就会连带换掉解析行为，而那些行为是**测过的**。

## 模型名不进协议

**`name` 是只读属性、不是一个构造参数的名字**：它进 run record 的配置指纹
（与 `Reranker.name` 同一个用途）。协议里**没有**"用哪个模型"这个方法——
模型由构造实现的那一方决定，调用方**无从指定**。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["DEFAULT_MAX_OUTPUT_TOKENS", "LlmClient", "LlmUnavailable"]

#: 输出上限的**兜底默认值**（真正的家是 `configs/*.yaml` 的 `router.max_output_tokens`）。
#: 路由器只该回一个词，但**不能靠"它答应只回一个词"来限制成本**——生成是逐 token 的。
DEFAULT_MAX_OUTPUT_TOKENS: int = 8


class LlmUnavailable(RuntimeError):
    """LLM 端点不可用（超时、连接失败、非 2xx、响应形状不符、内容为空）。

    ⚠ **它是"可降级"，不是"可忽略"**：调用方**必须**捕获它、走那条已声明的备用路径、
    **并把这件事记下来**——而不是让它冒泡成 5xx。

    与 `RerankUnavailable` **同一条理由**：LLM 是**位于提交链路关键路径上、又不被规则
    保证可用**的远程组件，而 Full 只有 2 次机会。**一次没分好类 vs 整个 Search 500**
    ——后者的代价是把真机会烧在一个跟答案无关的故障上。

    ⚠ **本异常绝不携带 Query 原文**：`.env` 里的密钥不许被打印（见根 `CLAUDE.md` 的
    安全口径），而 Query 是**不可信输入**——把它裹进异常消息再进日志，等于给
    "日志注入"开一个口子。调用方要记 Query，自己按需记。
    """


@runtime_checkable
class LlmClient(Protocol):
    """一次问答。**同步、无状态、不重试。**

    ⚠ **不重试**：与 `embed/` / `rerank/` 同一条理由——重试会与 AML 的 32 次重试叠加成
    两层，失败路径变得无法归因。这里失败就是失败，由调用方降级。
    """

    @property
    def name(self) -> str:
        """模型标识（进 run record 的配置指纹）。**提交时不得更换**（§2.3）。"""
        ...

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """把 `prompt` 发给模型，返回它回的那段文本（**不做任何解析**）。

        * `max_tokens` 限制输出长度——**不是可选的礼貌建议**：真实网关按 token 计费，
          而"它答应只回一个词"不是一条能兜住成本的保证。
        * `temperature = 0` 是**调用方声明"这一步要可复现"**的方式（§6 的可复现性要求）。
        * 端点不可用、非 2xx、响应形状不符、内容是空串 ⇒ 抛 [`LlmUnavailable`][LlmUnavailable]
        """
        ...
