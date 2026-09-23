"""渲染 —— **唯一实现**（不变式 I1）。

> §7.2 与 §11.3 要求：embedding 的输入 与 返回给 AML 的 `content`，必须是**同一份渲染**。

两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会**漂移**——**而且这种漂移不会报错**。
检索照常返回值，分数照常算，只是**命中的是另一个版本**。

**因此本模块是唯一拼 QA 对文本的地方**：
[`../embed/`](../embed/) 不自己拼字符串，[`../rank/`](../rank/) 的 packaging 也不自己拼。

**本模块不依赖任何业务层**（common/README.md 的分层要求）——它只是
`(question, answer) -> str` 的纯函数，**不认识 SQLite、不认识 Qdrant、不认识任何数据集**。
调用方自己取 `pair.question` / `pair.answer` 传进来。
"""

from __future__ import annotations

# ── 模板版本 ────────────────────────────────────────────────────────────
# ⚠ 改模板【必须】同时改这个版本号：它会进 embedding 缓存的坐标系
#   （embed/base.py 的 EmbeddingCoordinate），而 §11.3 明确"改模板 = 改变 embedding
#   输入 = 整个向量索引要重建"。版本号是让"旧缓存静默命中"变成不可能的那把锁。
TEMPLATE_VERSION: str = "v1"

QUESTION_PREFIX: str = "Q: "
ANSWER_PREFIX: str = "A: "

# Q 与 A 之间的分隔。rank/README.md 与 PRD §11.3 的【模板块】都是一个换行；
# 同节的示例里画成了空行（两个换行）——两处文档都有这个矛盾。
# 这里取模板块的字面（规范表述优先于示意），并把"改它"集中到这一行。
QUESTION_ANSWER_SEP: str = "\n"


def render(question: str | None, answer: str | None) -> str:
    """把一个 QA 对渲染成最终文本。**这是唯一一处拼装。**

    规则（rank/README.md §4 = 渲染规则的唯一出处）：

    | 输入 | 输出 |
    | --- | --- |
    | 都有 | `Q: {question}\\nA: {answer}` |
    | `question` 为空（§6.2 的"无问的对"） | 只输出 `A:` 那一行 |
    | `answer` 为空（`pending` 对） | 只输出 `Q:` 那一行 |

    返回值**首尾无空白**：AML 只做 `"\\n".join(...)` 拼接、不插分隔符，
    任何一项首尾留白都会让拼接处粘连（§11.3"content 必须自定界"）。

    ⚠ **不注入任何时间戳**。§11.3 有两条**互相独立**的机制都指向"不要注入绝对时间"，
    而答案 prompt 第 7 条**却要求**转换相对时间——所以加绝对时间戳可能反而有害。
    时间信息由 `event_time` 列负责筛选，正文只保留原始表述。
    """
    lines: list[str] = []
    if question:
        lines.append(f"{QUESTION_PREFIX}{question}")
    if answer:
        lines.append(f"{ANSWER_PREFIX}{answer}")
    return QUESTION_ANSWER_SEP.join(lines).strip()


__all__ = [
    "ANSWER_PREFIX",
    "QUESTION_ANSWER_SEP",
    "QUESTION_PREFIX",
    "TEMPLATE_VERSION",
    "render",
]
