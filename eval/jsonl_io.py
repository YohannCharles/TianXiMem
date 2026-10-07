"""JSONL 的**唯一读写实现**——转义掉 `splitlines()` 会断行的字符。

## 为什么单独一个模块（2026-09-26，一次真实事故）

`json.dumps(..., ensure_ascii=False)` **不转义** NEL / LS / PS 三个字符，
而 `str.splitlines()` **会把它们当换行**。于是一条**完整**的记录会被劈成两半，
前半段的字符串没有闭合 ⇒ `JSONDecodeError: Unterminated string`。

归档 pipeline 的 `rows()` 正是这么读的：

```python
[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
```

⇒ 现象是"**卡在同一道题**"（实测 47 个席位里只有 1 个含 U+2028 ⇒ **数据相关**，
却看起来像网络问题，排查方向整个跑偏）。归档那侧不许改，**写入侧转义是唯一的修法**。

**本模块的作用**：本仓自己的每一个 JSONL 读写点都走这里，
不再各写各的 `json.dumps(..., ensure_ascii=False) + 换行` 与 `splitlines()`。

⚠ **不要在本仓别处再写一份**：写方要转义、读方要按换行切，两者必须成对，
分开写必然漂移——而漂移的表现就是上面那条"看起来像网络问题"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Final

#: `json.dumps(ensure_ascii=False)` **不转义**、而 `str.splitlines()` **照断**的三个字符。
#:
#: NEL（U+0085）/ LS（U+2028）/ PS（U+2029）。`\\n` / `\\r` / `\\x0b` / `\\x0c` / `\\x1c-\\x1e`
#: 都由 `json.dumps` 自己转义掉了，所以不必列；这三个才是会咬人的。
#:
#: ⚠ **故意写成 `chr()` 而不是字符串字面量**：这三个字符**在编辑器里看不见**，
#: 写成字面量只会得到三处"看起来是对的"的空白——而本文件恰恰是讲这件事的。
LINE_BREAKS: Final[tuple[str, ...]] = (chr(0x85), chr(0x2028), chr(0x2029))


def jsonl_line(item: dict) -> str:
    """把一项写成**一行** JSONL——**转义掉 `splitlines()` 会断行的字符**。

    转义是安全的：`json.loads` 会解回同一个字符，**语义一字不变**。
    """
    line = json.dumps(item, ensure_ascii=False)
    for char in LINE_BREAKS:
        if char in line:
            # 换成 JSON 转义序列：`json.loads` 会解回同一个字符，语义一字不变。
            line = line.replace(char, f"\\u{ord(char):04x}")
    return line + "\n"


def write_line(handle: IO[str], row: dict) -> None:
    """写一行并**立刻落盘**：被强杀时最多丢最后一行（`judge._sanitize_jsonl` 会先修半行）。"""
    handle.write(jsonl_line(row))
    handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    """读一个 JSONL 文件；不存在则返回空表。

    ⚠ 按 `"\\n"` 切、**不用 `splitlines()`**（理由见模块 docstring）。
    """
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
