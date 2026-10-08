"""按数据集准备评测材料：暂存 → 固定版本校验 → 声明的补丁 → 原子发布。

加载器和 import 从不联网。评测 CLI 在发 Add/Search 之前调用 ensure_dataset；
--offline 禁止联网但允许生成派生语料/索引；--check 只校验。
源文件清单与目录映射分别只有 manifest.py / layout.py 一处。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .layout import PIPELINES, archive_file, local_path
from .manifest import MANIFEST


class PreparationError(RuntimeError):
    """材料缺失、校验失败或下载失败；必须在付费评测之前退出。"""


def sha256_file(path: Path) -> str:
    """分块读——LME 那份 277 MB、HaluMem 那份 33 MB，整个读进来只为算哈希不划算。

    ⚠ **本仓算文件哈希只用这一份**（`eval/datasets/**` 与 `tools/**` 都从这里 import）：
    各写一份的话，`prepare.py` 的分块版与别处的 `read_bytes()` 版会**给出一致的值但吃不同的内存**
    ——于是"哪个能跑得动"只在最大的那份文件上才暴露。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ASYNC_OPEN = re.compile(
    r"async with httpx\.AsyncClient\((?P<client>timeout=[^)]*)\) as client, "
    r'(?P<open>\w+\.open\("(?P<mode>[aw])", encoding="utf-8"\)) as (?P<var>\w+):'
)


def _patch_async_open(text: str) -> tuple[str, int]:
    """只修上游把同步文件当异步上下文的问题，保持已锁定的本地哈希。"""
    patched, count = _ASYNC_OPEN.subn(
        lambda m: (
            f"async with httpx.AsyncClient({m['client']}) as client, "
            f"contextlib.nullcontext({m['open']}) as {m['var']}:"
        ),
        text,
    )
    if count and "import contextlib" not in patched:
        if "import argparse" not in patched:
            raise PreparationError("上游形状变了，无法应用 async-open 补丁")
        patched = patched.replace("import argparse", "import argparse\nimport contextlib", 1)
    return patched, count


LOCAL_PATCHES = {"async-open": _patch_async_open}


def apply_local_patch(entry: dict, path: Path) -> tuple[bool, str]:
    patch_id = entry.get("local_patch")
    if not patch_id:
        return False, "无需修订"
    patched, count = LOCAL_PATCHES[patch_id](path.read_text(encoding="utf-8"))
    if count:
        # ⚠ `newline=""`：禁止 Windows 把写出的 '\n' 翻译成 '\r\n'。清单里的本地 sha256 是在
        #    Linux（读写都不翻译）上算的；翻译会让补丁后的字节对不上本地哈希（响亮失败）。
        path.write_text(patched, encoding="utf-8", newline="")
    return bool(count), f"修订 {count} 处（{patch_id}）"


def _group(entry: dict) -> str:
    path = local_path(entry["name"])
    if path.parts[0] == ".upstream":
        return path.parts[1]
    return path.parts[0]


DATASETS = tuple(sorted({_group(e) for e in MANIFEST} - {".legacy"}))


def entries_for(dataset: str, *, purpose: str = "eval") -> list[dict]:
    """只选该数据集的数据与评分依赖；judge 只取上游评分材料。"""
    if dataset not in DATASETS and dataset != "official-capture":
        raise ValueError(f"未知数据集 {dataset!r}，可选：{', '.join(DATASETS)}")
    if purpose not in {"eval", "judge"}:
        raise ValueError(f"未知准备用途：{purpose}")
    if dataset == "official-capture":
        # 实际重放按已选问题的家族调用 judge；本项是预取所有裁判依赖的便利入口。
        return [e for e in MANIFEST if local_path(e["name"]).parts[0] == ".upstream"]
    names = {PIPELINES[dataset], "aml_readme.md"} if dataset in PIPELINES else set()
    return [
        e
        for e in MANIFEST
        if e["name"] in names
        or (
            _group(e) == dataset
            and (purpose == "eval" or local_path(e["name"]).parts[0] == ".upstream")
        )
    ]


def _download(url: str, dest: Path) -> None:
    """仅下载到调用者的临时文件，最终路径由校验后的发布步骤写入。"""
    # ⚠ 先做百分号编码：清单里有真实带空格的路径（doc-pp 的 `ACL 2025_*` 资产），
    #    `urllib` 对它们直接抛 `InvalidURL`（httpx 会自动编码，所以旧工具从没暴露过）。
    request = urllib.request.Request(
        urllib.parse.quote(url, safe=":/%#?=@[]!$&'()*+,;"),
        headers={"User-Agent": "tianximem/prepare-dataset"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            size = 0
            last = time.monotonic()
            for chunk in iter(lambda: response.read(1 << 20), b""):
                handle.write(chunk)
                size += len(chunk)
                if time.monotonic() - last >= 20:
                    print(f"  {dest.name}: downloaded {size / (1 << 20):,.0f} MiB", flush=True)
                    last = time.monotonic()


def _prepare_entry(entry: dict, root: Path, *, offline: bool, patch: bool = False) -> bool:
    """返回是否新发布文件。任何不符的已有文件都不能被自动覆盖。"""
    path = archive_file(root, entry["name"])
    actual = sha256_file(path) if path.is_file() else None
    if actual == entry["sha256"]:
        return False
    upstream = entry.get("upstream_sha256", entry["sha256"])
    if actual is not None and not (patch and entry.get("local_patch") and actual == upstream):
        raise PreparationError(f"{path} sha256 不符；保留原文件，请核对来源后再移走重取")
    if actual is None and (offline or patch):
        raise PreparationError(f"缺 {path}；用 make fetch-data DATASET={_group(entry)} 准备")
    if not entry["url"] and actual is None:
        raise PreparationError(f"{entry['name']} 无可恢复来源，只能保留本地存档")
    if entry.get("archive_member") and actual is None:
        required = int(entry["bytes"]) + int(entry["download_bytes"])
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < required:
            raise PreparationError(f"{entry['name']} 下载/解压需要至少 {required:,} 字节可用空间")

    staging_root = root / ".tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    # 每次使用独立目录：并发下载不争用 .part，失败退出会移除本次临时字节。
    with tempfile.TemporaryDirectory(prefix="download-", dir=staging_root) as tmp:
        staged = Path(tmp) / Path(entry["name"]).name
        if actual is not None:
            shutil.copyfile(path, staged)
        else:
            print(f"  ↓ {local_path(entry['name'])}", flush=True)
            try:
                if entry.get("archive_member"):
                    from .corpus_prepare import extract_zip_member

                    compressed = Path(tmp) / "source.zip"
                    _download(entry["url"], compressed)
                    if (
                        compressed.stat().st_size != int(entry["download_bytes"])
                        or sha256_file(compressed) != entry["download_sha256"]
                    ):
                        raise PreparationError(
                            f"{entry['name']} 压缩包与固定版本的 sha256/大小不符"
                        )
                    extract_zip_member(compressed, staged, entry)
                else:
                    _download(entry["url"], staged)
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                ValueError,
                zipfile.BadZipFile,
            ) as error:
                raise PreparationError(f"下载 {entry['name']} 失败：{error}") from error
        if sha256_file(staged) != upstream:
            raise PreparationError(f"{entry['name']} 下载字节与固定版本的上游 sha256 不符")
        if entry.get("local_patch"):
            apply_local_patch(entry, staged)
        if sha256_file(staged) != entry["sha256"]:
            raise PreparationError(f"{entry['name']} 准备后 sha256 不符，不发布")
        destination = path if actual is not None else root / local_path(entry["name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(destination)
    return True


def _prepare_corpus(dataset: str, root: Path, *, check_only: bool = False) -> None:
    """原始包解出语料及建立派生检索索引；只有 CLI 调用，裁判准备不调用。"""
    try:
        if dataset == "hybridqa":
            from .corpus_prepare import prepare_hybridqa_corpus
            from .hybridqa import ARCHIVE, DATA_DIR

            entry = next(e for e in MANIFEST if e["name"] == f"{DATA_DIR}/{ARCHIVE}")
            prepare_hybridqa_corpus(root, entry, check_only=check_only)
        elif dataset == "feverous":
            from .feverous import prepare_index, source_sha256

            prepare_index(root, source_sha256(), check_only=check_only)
    except (ValueError, OSError, sqlite3.Error, tarfile.TarError) as error:
        raise PreparationError(str(error)) from error


def ensure_dataset(
    dataset: str, root: str | Path, *, offline: bool = False, purpose: str = "eval"
) -> None:
    """CLI 的按需准备入口；禁止从 loader/import 自动调用。"""
    entries = entries_for(dataset, purpose=purpose)
    for entry in entries:
        _prepare_entry(entry, Path(root), offline=offline)
    if purpose == "eval":
        _prepare_corpus(dataset, Path(root))
    if entries:
        print(f"  数据准备完成：{dataset}（{len(entries)} 个文件已校验）", flush=True)


def migrate_archive(source: Path, root: Path) -> int:
    """先校验整份迁移计划，再移动；未知文件也保留，碰到冲突不覆盖。"""
    source, root = source.resolve(), root.resolve()
    if not source.is_dir() or source == root or source in root.parents or root in source.parents:
        raise PreparationError("迁移源和目标必须是不同的、互不包含的目录")
    known = {entry["name"]: entry for entry in MANIFEST}
    moves = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise PreparationError(f"归档里有符号链接，需人工核对：{path}")
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        entry = known.get(name)
        if entry and sha256_file(path) != entry["sha256"]:
            raise PreparationError(f"迁移前校验不符：{path}；未移动任何文件")
        relative = local_path(name) if entry else Path(".legacy/unregistered") / name
        destination = root / relative
        if destination.exists():
            raise PreparationError(f"迁移目标已存在：{destination}；未移动任何文件")
        moves.append((path, destination))
    completed = []
    try:
        for path, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            completed.append((path, destination))
    except OSError:
        for path, destination in reversed(completed):
            path.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(path)
        raise
    for directory in sorted(source.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir():
            directory.rmdir()
    source.rmdir()
    return len(moves)


def main(argv: list[str] | None = None) -> int:
    from .registry import benchmark_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--fetch", action="store_true", help="准备缺失文件（默认）")
    action.add_argument("--check", action="store_true", help="只校验，不联网")
    action.add_argument("--patch", action="store_true", help="仅应用清单声明的本地补丁")
    parser.add_argument("--offline", action="store_true", help="只使用已有文件，禁止下载")
    parser.add_argument("--dataset", action="append", choices=(*DATASETS, "official-capture"))
    parser.add_argument("--tier", choices=("required", "optional", "official-extra", "all"))
    parser.add_argument("--dir", type=Path, default=benchmark_dir(), help="数据根目录")
    parser.add_argument("--list", action="store_true", help="列出可准备的数据集")
    parser.add_argument("--migrate-from", type=Path, help="校验并迁移旧的 benchmark_data")
    args = parser.parse_args(argv)
    if args.dataset and args.tier:
        parser.error("--dataset 与 --tier 不能混用")
    if args.list:
        for name in DATASETS:
            print(f"{name}: {len(entries_for(name))} 个材料文件")
        return 0
    if args.migrate_from:
        try:
            count = migrate_archive(args.migrate_from, args.dir)
        except (PreparationError, OSError) as error:
            print(f"✗ {error}")
            return 1
        print(f"✓ 已迁移 {count} 个文件到 {args.dir}")
        return 0
    if args.check and not args.dir.is_dir():
        print(f"✗ 数据目录不存在：{args.dir}；用 make fetch-data 准备")
        return 1
    if args.dataset:
        wanted = {e["name"] for name in args.dataset for e in entries_for(name)}
        entries = [e for e in MANIFEST if e["name"] in wanted]
    else:
        entries = [e for e in MANIFEST if args.tier in (None, "all") or e["tier"] == args.tier]
    if args.patch:
        entries = [entry for entry in entries if entry.get("local_patch")]
    bad, checked = 0, 0
    for entry in entries:
        if entry["tier"] == "archive-only" and not archive_file(args.dir, entry["name"]).exists():
            print(f"  · {entry['name']} 无可恢复来源，不参与评测")
            continue
        try:
            _prepare_entry(entry, args.dir, offline=args.check or args.offline, patch=args.patch)
            checked += 1
        except (PreparationError, OSError) as error:
            print(f"✗ {error}")
            bad += 1
    if not bad and not args.patch:
        groups = set(args.dataset or [_group(e) for e in entries])
        for dataset in sorted(groups & {"hybridqa", "feverous"}):
            try:
                _prepare_corpus(dataset, args.dir, check_only=args.check)
            except PreparationError as error:
                print(f"✗ {error}")
                bad += 1
    print(f"✗ {bad} 项有问题" if bad else f"✓ {checked} 项全部与清单一致")
    return int(bool(bad))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
