"""官方 add 正文的形态（2026-10-01 实测）——**"本地发的 add 长什么样"的唯一实现**。

## 为什么它属于 harness

`Add` 的 payload **不是我们造的**：线上由 AML 造，本地由 harness 造。所以"本地发的
像不像线上"直接决定**本地分数预不预测得了线上**——这与切批（[`batching.py`](./batching.py)）
是同一类问题，也因此**与切批一样要进数据指纹**。

## 全量实测（官方 43,272 条 add / 454,937 条消息）

| 观测 | 数 |
| --- | --- |
| 正文带 `<标签>: ` 前缀 | **426,971 / 454,937 = 93.9%** |
| ├ 标签就是 `role`（`user:` / `assistant:` 小写） | 377,278（82.9%） |
| ├ 标签是说话人名或固定词（`Caroline:` / `Corpus:` / `source:` …） | 49,693（10.9%） |
| 完全没有前缀 | 27,966（6.1%） |
| `role` 的取值域 | **只有 `user` / `assistant`，零 `system`** |

⇒ 两条规则：**正文一律 `<标签>: ` 开头**；**`system` 不存在**。

`system` 那一条是**逐字节证明**的：`dataset/clbench/clbench.jsonl:24` 的 raw
`messages[0]` 是 `role="system"`、正文 `You are a Chemical Engineer researcher…`，
而官方那条 canary 发的是 `role="user"`、正文 `user: You are a Chemical Engineer researcher…`
（该 canary 的正文去转义后在官方流量里逐字命中，同一条也命中 `:302`）。
⇒ 官方**把 `system` 折进 user 侧的前缀、role 改 `user`**。

## 标签怎么来——逐数据集，全部取自官方流量原文

| 数据集 | user 侧 | assistant 侧 | 依据 |
| --- | --- | --- | --- |
| `locomo-refined` | 真说话人名 | 真说话人名 | canary `Caroline:` / `Melanie:` |
| `longmemeval-s` | `User` | `Assistant` | 正文逐字命中 |
| `clbench` | `user` | `assistant` | canary；raw `system` 折成 `user:` |
| `beam` | persona 名 | `Assistant` | canary `Christina Baker:` / `Assistant:` |
| `personamem-v2` | `User` | `Assistant` | canary |
| `mquake-remastered` | `Corpus` | `Corpus` | canary `Corpus: {json}` |
| `memtrapbench` | `user` | `assistant` | canary |
| `corporatebench` | `document` | `document` | 内容归属（zenithlabs 邮件） |
| `medmemorybench` | `user` | `assistant` | 内容归属（中文医患，**占全量 73%**） |
| `tempreason` | `source` | `source` | canary `source: …` |

⚠ **`mquake` 只有标签能对齐**：官方正文是 `Corpus: {json}`，我们发的是自然句
（[`../datasets/mquake.py`](../datasets/mquake.py) 的实测决定——`UPDATE: …` 那种形式让
模型答 "Cannot determine"）。**这是有意的偏离，加个标签不代表对齐了。**

## 它还负责**切**

官方单条消息有**硬字符上限**（实测 ≈8,000，见 [`MAX_MESSAGE_CHARS`][...]），超了就切成
**连续的多条同 role 消息**——每片重新带标签，**拼回去是全文**（不是截断）。
本模块把它和渲染放在一起，因为**上限是连标签一起算的**：先切后加标签会算出另一个长度。

⇒ **返回值可能比输入长。** 调用方必须**先 shape、再切批**（否则批界会落在错的条数上）——
[`driver.ingest`](./driver.py) 就是这个顺序。

## 三个取值

| `--add-shape` | 是什么 |
| --- | --- |
| **`official`**（缺省） | 上表那两条规则 **+ 切到 8,000**。**线上就是这个**，除非有理由，别改 |
| `native` | 改之前的样子（数据集给什么 role 就发什么、正文不加前缀、**也不切**）。**只作对照** |
| `alluser` | `official` 之上再**把 role 全折成 `user`**——见下 |

**关于 `alluser`**：官方**判分池**（23 个有官方判分反馈的 user）实测是
**100% `role: user`、零 assistant**，说话人只活在正文标签里（`Gina:` / `Audrey:` /
`FOREMAN:`）。按 **旧规则**推下去：连续同 role 会合并 ⇒ 一次 Add（≤20 条）塌成
**1 个 MemoryBlock**（只有 `Q:` 行）⇒ **扩窗 / 段合并 / QA 配对在那个池子上是结构性失效的**。

⚠ **边界**：这个形态**只在 LoCoMo / ScriptMem 那一簇上实测到**（判分池里正好是这两份），
**不是全赛道的形态**（全量 82.9% 仍是 role 保留 + 标签）。它进不了缺省值，用途是
"量一量我们的 D24 在那种输入下还剩多少"，**不是"线上就长这样"**。
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.datasets.preprocess import Message

__all__ = [
    "ADD_SHAPES",
    "DEFAULT_ADD_SHAPE",
    "DEFAULT_LABEL_RULE",
    "LABEL_RULES",
    "MAX_MESSAGE_CHARS",
    "shape_batch",
    "split_prefixed_payload",
]

#: 渲染口径。**进数据指纹**（[`../datasets/registry.py`](../datasets/registry.py)）——
#: 与切批同一条纪律：**别拿不同形态的分数互比**。
ADD_SHAPES: tuple[str, ...] = ("official", "alluser", "native")

#: 缺省。理由见模块 docstring 的表格——线上就是它。
DEFAULT_ADD_SHAPE: str = "official"

#: **官方单条消息的字符上限**（2026-10-01 实测，全量 454,937 条消息）：
#: **2,514 条正文正好 8,000 字符，`0` 条超过**，而 7,900–7,999 之间只有约 30 条
#: ⇒ 这是**硬上限**，不是分布的自然末端。
#:
#: ⚠ **它连标签一起算**：切出来的每一片都**重新带 `<标签>: ` 前缀**——主干流量里
#: 逐字见过切点落在半个 token 上：`…"row_span": "` → 下一片 `Corpus: 1"…`；
#: `…lost to [[Se` → `Corpus: rena_Williams|Serena]]…`。
#: ⇒ **先渲染、后切**，切完每片再盖一次前缀（`_split_labelled` 就是干这个的）。
#:
#: ⚠ **与 S2 的"2,000 个 Adapter 计的词"对不上**：两次实测（这个字符上限、和 canary 上
#: 约 2,000 词一片的切法）**落在不同的数上**，换算关系没查清。这里跟的是**主干**那一个，
#: 因为主干占 99.7% 的流量。依据与待查项见
#: [`../../docs/open-questions.md`](../../docs/open-questions.md) 的 S2 / S5。
MAX_MESSAGE_CHARS: int = 8_000

#: 表里没登记的数据集用哪条规则。取**全量占比最高的那一条**（82.9% 的正文就是
#: `user: ` / `assistant: `）——它不是"兜底"，是缺省。
DEFAULT_LABEL_RULE: tuple[str, str] = ("user", "assistant")

#: 数据集 → `(user 侧标签, assistant 侧标签)`。`"@speaker"` = 取 `Sample.speaker_names`
#: 里对应的那个（user → `[0]`、assistant → `[1]`）。
LABEL_RULES: dict[str, tuple[str, str]] = {
    "locomo-refined": ("@speaker", "@speaker"),
    "longmemeval-s": ("User", "Assistant"),
    "clbench": ("user", "assistant"),
    "beam": ("@speaker", "Assistant"),
    "personamem-v2": ("User", "Assistant"),
    "mquake-remastered": ("Corpus", "Corpus"),
    "memtrapbench": ("user", "assistant"),
    "corporatebench": ("document", "document"),
    "medmemorybench": ("user", "assistant"),
    "tempreason": ("source", "source"),
    "halumem": ("user", "assistant"),
    "musique": ("Corpus", "Corpus"),
    "hybridqa": ("Corpus", "Corpus"),
    "feverous": ("Corpus", "Corpus"),
}


def _side(role: str) -> int:
    """`assistant` → 1，**其余一律 → 0**（包括 `user` 与任何未知 role）。

    **不枚举白名单**（与 [`pairing.is_user`](../../src/tianximem/pairing/pairing.py) 同一条理由）：
    AML 的 role 取值域没有文档。⚠ 本模块比 `pairing` **多一步**——`pairing` 只把非 user
    归到"非 user 侧"，本模块还要**改写 `role` 字段**：官方零 `system`，它把 `system`
    折进了 **user 侧**（逐字节证据见模块 docstring）⇒ 这里 `system` 必须落到 0。
    """
    return 1 if role == "assistant" else 0


def _label(rule: str, *, side: int, speaker_names: tuple[str, str]) -> str:
    if rule != "@speaker":
        return rule
    # `@speaker` 只有真名可用的数据集登记（LoCoMo / BEAM）。名字缺了**不许静默**——
    # 少一个说话人名会让检索看不见"谁说的"，而屏幕上看不出任何异常。
    if side >= len(speaker_names) or not speaker_names[side]:
        raise ValueError(f"`@speaker` 标签缺名字：speaker_names={speaker_names!r}，side={side}")
    return speaker_names[side]


def _split_labelled(label: str, text: str, cap: int = MAX_MESSAGE_CHARS) -> list[str]:
    """`<标签>: <正文>` 按 `cap` 字符硬切，**每一片都重新盖前缀**。

    这是官方那一步的还原：切点**不在句边界**（实测落在半个 token 上），切完的片
    是**独立的、同 role 的消息**（不是一条消息里的字段）。

    ⚠ **`cap` 是连前缀一起算的** ⇒ 先减去 `len("<标签>: ")` 再切正文，否则每片会超出
    `len(前缀)` 个字符——而超出的那几字符正好会撞上嵌入上限。
    """
    prefix = f"{label}: "
    room = cap - len(prefix)
    if room < 1:
        raise ValueError(f"标签 {label!r} 太长，{cap} 字符里塞不下它的前缀")
    return [prefix + text[start : start + room] for start in range(0, len(text), room)]


def split_prefixed_payload(message: dict) -> list[dict]:
    """AML 适配器已完成包装，只切长消息，不再次加标签或修改角色/时间。

    无标签的 inline-time 消息保留原文，按相同字符上限切片。
    原请求回放不调用此函数。
    """
    content = message["content"]
    if len(content) <= MAX_MESSAGE_CHARS:
        return [dict(message)]
    label, separator, body = content.partition(": ")
    if separator and "\n" not in label and len(label) < 100 and not label.startswith("["):
        parts = _split_labelled(label, body)
    else:
        parts = [
            content[start : start + MAX_MESSAGE_CHARS]
            for start in range(0, len(content), MAX_MESSAGE_CHARS)
        ]
    return [dict(message, content=part) for part in parts]


def shape_batch(
    messages: Sequence[Message],
    *,
    dataset: str,
    speaker_names: tuple[str, str],
    shape: str = DEFAULT_ADD_SHAPE,
) -> list[dict]:
    """把一批消息渲染成 `Add` 的 `messages` 数组。

    `native` 直接走 [`Message.to_add_payload`](../datasets/preprocess.py)——
    那条路是"数据集给的形状"，**不是**"线上会来的形状"（也因此**不切**：切是平台行为）。

    ⚠ **返回值可能比输入长**——超长消息会被切成多条（[`MAX_MESSAGE_CHARS`][...]）。
    ⇒ **调用方必须先 shape 再切批**（见 [`driver.ingest`](./driver.py)）。
    """
    if shape not in ADD_SHAPES:
        raise ValueError(f"未知的 add 形态 {shape!r}——只有 {' / '.join(ADD_SHAPES)}")
    if shape == "native":
        return [m.to_add_payload() for m in messages]

    rules = LABEL_RULES.get(dataset, DEFAULT_LABEL_RULE)
    payloads: list[dict] = []
    for m in messages:
        side = _side(m.role)
        label = _label(rules[side], side=side, speaker_names=speaker_names)
        # `official`：只有 `assistant` 保住自己的 role，其余折成 `user`（官方零 system）。
        # `alluser`：连 `assistant` 也折——说话人只活在正文标签里（判分池的形态）。
        role = "user" if (shape == "alluser" or side == 0) else "assistant"
        for content in _split_labelled(label, m.content):
            item: dict = {"role": role, "content": content}
            if m.timestamp_ms is not None:
                item["timestamp"] = m.timestamp_ms
            payloads.append(item)
    return payloads
