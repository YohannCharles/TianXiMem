"""归档的**本地修订**—— 补丁必须是**声明出来的**，不是悄悄改的。

## 背景

AML 公开的 7 个 pipeline **在 `answer` / `evaluate` 第一步就崩**：

```python
async with httpx.AsyncClient(timeout=120) as client, output.open("a", encoding="utf-8") as handle:
```

`pathlib.Path.open()` 给的 `TextIOWrapper` **没有异步上下文协议** ⇒ 一个 `TypeError`，
7 个 pipeline、12 处，全都跑不起来。已核对上游钉住的 revision：**逐字节就是这样**
（`upstream_sha256` 记着那份的哈希）。

⇒ 处置是**就地修订 + 记成已声明的偏离**（见
[`../tools/fetch_benchmark_data.py`](../tools/fetch_benchmark_data.py) 的 `local_patch` 一节）。
本文件把那条声明**钉住**：修了几处、只修了什么、以及它确实**没碰 prompt 与判分逻辑**。

⚠ **归档不在场时这些用例 skip**（与 `qdrant` 夹具同一条纪律）：测试不该依赖
`benchmark_data/` 是否取回过。但它们**在场时必须真的验**——"一个永远不会 FAIL 的检查
等于没有检查"。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tools.fetch_benchmark_data import MANIFEST, _patch_async_open

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = _ROOT / "benchmark_data"

#: 修订前的形状。**它不该在任何归档文件里再出现。**
_RAW_ASYNC_OPEN = re.compile(
    r"async with httpx\.AsyncClient\([^)]*\) as client, "
    r'\w+\.open\("[aw]", encoding="utf-8"\) as \w+:'
)

#: 反向替换（只在本文件的"补丁恰好是声明的那一处"用例里用）。**与 `_patch_async_open`
#: 的产物逐字对应**——两者的正则若不配对，下面那条用例会立刻红。
_PATCHED_ASYNC_OPEN = re.compile(
    r"async with httpx\.AsyncClient\((?P<client>[^)]*)\) as client, "
    r'contextlib\.nullcontext\((?P<open>\w+\.open\("[aw]", encoding="utf-8"\))\) as (?P<var>\w+):'
)

_PATCHED = [entry for entry in MANIFEST if entry.get("local_patch")]


def _read(entry: dict) -> str:
    path = _ARCHIVE / entry["name"]
    if not path.exists():
        pytest.skip(f"{entry['name']} 不在归档里（`make fetch-data` 才取回）")
    return path.read_text(encoding="utf-8")


def test_the_manifest_declares_exactly_the_deviation() -> None:
    """**修订只许声明在清单里**：每个 `local_patch` 都要有上游哈希，且两者不同。

    `upstream_sha256` 是"我们从上游拿到的是什么"，`sha256` 是"本地归档是什么"——
    两个都记着，才可能回答"这份文件与上游差在哪、差的是不是我们打算差的那一处"。
    """
    names = {entry["name"] for entry in _PATCHED}
    assert names == {
        "pipeline_locomo-refined.py",
        "pipeline_longmemeval-s.py",
        "clb_pipeline.py",
        "pipeline_beam.py",
        "pipeline_scriptmem.py",
        "pipeline_v1_personamem.py",
        "pipeline_v2_personamem.py",
    }, f"被修订的文件集合变了：{sorted(names)}"

    for entry in _PATCHED:
        assert entry["local_patch"] == "async-open"
        upstream = entry["upstream_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", upstream), entry["name"]
        assert upstream != entry["sha256"], f"{entry['name']}：两个哈希相同 ⇒ 修订其实没落地"


def test_no_unpatched_async_open_remains_in_the_archive() -> None:
    """修订之后，**12 处** `Path.open` 当异步上下文用的写法一处都不该剩。"""
    total = 0
    for entry in _PATCHED:
        text = _read(entry)
        offenders = _RAW_ASYNC_OPEN.findall(text)
        assert offenders == [], f"{entry['name']} 还有没打补丁的 async open：{offenders}"
        assert "contextlib.nullcontext(" in text, f"{entry['name']} 里没有 nullcontext"
        total += text.count("contextlib.nullcontext(")
    assert total == 12, f"修订处数应为 12，实测 {total}——文件被换过？"


def test_patch_is_exactly_the_declared_deviation() -> None:
    """**补丁恰好是那一处替换**：把磁盘上的文件反推回上游形状，再打一次补丁，必须逐字节相同。

    这条比"有没有 nullcontext"强得多：它排除了"顺手还改了别的"——包括任何 prompt。
    这才是「契约以 pipeline 代码为准」那句话在本地修订之后仍然成立的原因。
    """
    for entry in _PATCHED:
        text = _read(entry)

        def revert(match: re.Match) -> str:
            return (
                f"async with httpx.AsyncClient({match['client']}) as client, "
                f"{match['open']} as {match['var']}:"
            )

        pristine, reverted = _PATCHED_ASYNC_OPEN.subn(revert, text)
        assert reverted > 0, entry["name"]
        # 反向替换也要把那行 `import contextlib` 拿掉，才回到上游的样子
        pristine = pristine.replace("import argparse\nimport contextlib\n", "import argparse\n", 1)

        repatched, count = _patch_async_open(pristine)
        assert count == reverted, f"{entry['name']}：补丁处数 {count} ≠ 反向替换处数 {reverted}"
        assert repatched == text, f"{entry['name']}：补丁不只是那处替换（有人顺手改了别的）"


def test_patch_is_idempotent() -> None:
    """已经打过的文本再打一次 ⇒ 零处改动（`--patch` 重复跑是安全的）。"""
    for entry in _PATCHED:
        _, count = _patch_async_open(_read(entry))
        assert count == 0, f"{entry['name']}：补丁不幂等"
