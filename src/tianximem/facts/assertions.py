"""疑问判据：一句话在问，就不是在主张事实。

只服务规则层的逐句抽取（`grammar.extract_evidence`）——判据一律保守：
只有明确的疑问形式（问号、句末语气词、句首疑问词、中文的是否类词）才算疑问。
"""

from __future__ import annotations

import re

# 句末疑问标记（含中文语气词）。`么` 同时覆盖「…什么」这类句末疑问词。
_ZH_SUFFIX = re.compile(r"(?:吗|呢|么|吧|没有)[。！!？?…~\s]*$")
# 句首疑问词；**不匹配句中出现**——"我不知道当时发生了什么" 是陈述，不能被整句吞掉。
_ZH_LEAD = re.compile(
    r"^\s*(?:请问|请告诉我|问一下|是否|为什么|怎么|如何|什么|哪(?:个|些|里|儿)|谁|多少|"
    r"几(?:个|位|名|次|种|人)|能不能|可不可以)"
)
_ZH_ANYWHERE = re.compile(r"是否|是不是|能否|可否")
_EN_LEAD = re.compile(r"^\s*(?:who|whom|whose|what|which|where|when|why|how)\b", re.I)
_EN_INVERSION = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|have|has|had|can|could|will|would|should|may|might|"
    r"must|am)\s+(?:you|i|we|they|he|she|it|there|the|a|an|any|this|that|these|those)\b",
    re.I,
)
_MARK = re.compile(r"[?？]")


def interrogative(text: str) -> bool:
    """疑问句：句末/句首标记，或中文的是否类词。"""
    if _MARK.search(text) or _ZH_SUFFIX.search(text) or _ZH_ANYWHERE.search(text):
        return True
    return bool(_ZH_LEAD.match(text) or _EN_LEAD.match(text) or _EN_INVERSION.match(text))
