"""渲染、token 计数、配置。

**这不是一个工具箱。** 它存在的理由是**一条不变式**（渲染只有一份实现），
外加两个"必须只有一份实现"的量（token 计数、配置校验）。

**本目录不依赖任何业务层**——它被所有层依赖。
"""

from tianxi_am.common.render import (
    ANSWER_PREFIX,
    QUESTION_ANSWER_SEP,
    QUESTION_PREFIX,
    TEMPLATE_VERSION,
    render,
)

__all__ = [
    "ANSWER_PREFIX",
    "QUESTION_ANSWER_SEP",
    "QUESTION_PREFIX",
    "TEMPLATE_VERSION",
    "render",
]
