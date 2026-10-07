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

## 另一半：`official-extra` 那一档的清单形状

它的文件表是**紧凑格式**（一行路径、一行 sha256，见 `tools/fetch_benchmark_data.py`），
解析在 import 时就会对哈希格式断言。这里再钉住形状本身——**格式坏了要么响亮失败，要么别写**。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from eval.datasets.layout import archive_file
from eval.datasets.manifest import MANIFEST
from eval.datasets.prepare import _patch_async_open
from eval.datasets.registry import benchmark_dir

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = benchmark_dir()

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
    path = archive_file(_ARCHIVE, entry["name"])
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


def test_official_extra_tier_is_well_formed() -> None:
    """`official-extra` 那一档：条目齐、名字不重、每个都有 URL 与**格式合法的 sha256**。

    ⚠ 这条**不依赖归档是否取回过**——它钉的是清单本身。紧凑文件表一旦被谁改坏
    （少一行哈希、路径里混进空格），`_parse_file_table` 会在 import 时就抛；
    而"整档被悄悄删空"这类不会抛的，由下面这三条兜住。
    """
    extra = [entry for entry in MANIFEST if entry["tier"] == "official-extra"]
    assert extra, "official-extra 那一档空了"
    assert len({entry["name"] for entry in extra}) == len(extra), "有重名条目"
    for entry in extra:
        assert entry["url"].startswith("https://"), entry["name"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["name"]
    # 8 个数据集各有一条 README——档还在，至少它们要在
    readmes = {
        entry["name"].split("/")[0] for entry in extra if entry["name"].endswith("README.md")
    }
    assert readmes == {entry["name"].split("/")[0] for entry in extra}, (
        f"有数据集一条 README 都没登记：{ {e['name'].split('/')[0] for e in extra} - readmes }"
    )


# ── 取回工具的**入口行为**（2026-10-04）─────────────────────────────────────
def test_fetch_does_not_require_the_directory_to_exist_beforehand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚠ **`--fetch` 不能被"目录得先存在"那道门挡住。**

    那两个建议的解法（`--dir` / `TIANXIMEM_BENCHMARK_DIR`）**指向不存在的路径同样过不了
    那道判断** ⇒ "取回数据"这件**唯一能建出目录**的事，反倒被一道"目录得先存在"的门挡了。

    ⇒ `--fetch` 现在不受它拦（目录由 `_download` 在连接成功后建）。
    本用例把 `_download` 换成一个**立刻抛网络错**的桩：**只要它被调到，就说明越过了守卫**。
    """
    import urllib.error

    from eval.datasets import prepare as F

    called: list[str] = []

    def boom(url: str, dest: Path) -> None:
        called.append(url)
        raise urllib.error.URLError("本用例不打网络")

    monkeypatch.setattr(F, "_download", boom)
    target = tmp_path / "还没建过的归档"
    monkeypatch.setattr(
        "sys.argv",
        ["fetch_benchmark_data.py", "--fetch", "--tier", "required", "--dir", str(target)],
    )
    assert not target.exists()
    assert F.main() == 1, "下载全失败 ⇒ 以 1 退出"
    assert called, "没走到下载 ⇒ 被那道守卫拦住了"


def test_download_creates_the_directory_only_after_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⚠ **连不上时不许留下空目录。**

    `tests/test_datasets.py` 的 `needs_archive` 守卫判的是【**目录在不在**】而不是
    【归档在不在】⇒ 网络不通时先在磁盘上留一个空 `benchmark_data/`，会让那两条归档用例
    **从 skip 变 fail**，而屏幕上看不出与网络有关（`.gitignore` 里记着这条实测教训）。

    ⇒ `mkdir` 排在 `urlopen` **成功之后**。本用例让 `urlopen` 抛，断言目录**没被建**。
    """
    import urllib.error
    import urllib.request

    from eval.datasets import prepare as F

    def boom(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("连不上")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    dest = tmp_path / "归档" / "某个文件.json"
    with pytest.raises(urllib.error.URLError):
        F._download("https://example.invalid/x.json", dest)
    assert not dest.parent.exists(), "连不上却把目录建出来了"


def test_check_on_a_missing_archive_points_at_the_fetch_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--check` 撞上不存在的目录时，**提示必须指向真正能解决它的那一步**。"""
    from eval.datasets import prepare as F

    monkeypatch.setattr(
        "sys.argv", ["fetch_benchmark_data.py", "--check", "--dir", str(tmp_path / "没有")]
    )
    assert F.main() == 1
    text = capsys.readouterr().out
    assert "make fetch-data" in text, text
    assert "--dir 或 TIANXIMEM_BENCHMARK_DIR 指定" not in text, "那句建议无效（同样要求目录已存在）"
