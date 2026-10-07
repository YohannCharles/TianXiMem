"""语料包的白名单提取：只在准备入口运行，保持上游成员字节。"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from .prepare import sha256_file


def extract_zip_member(archive: Path, destination: Path, entry: dict) -> None:
    """不使用 extractall；ZIP CRC、成员大小及发布后的 sha256 分别校验。"""
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.getinfo(entry["archive_member"])
        if member.is_dir() or member.file_size != int(entry["bytes"]):
            raise ValueError("ZIP member type/size differs from pinned source")
        with bundle.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)


def prepare_hybridqa_corpus(root: Path, entry: dict, *, check_only: bool = False) -> None:
    """从固定 tar 包选择 dev 引用的完整 table/request 对，逐文件原子发布。"""
    from .hybridqa import ARCHIVE, CORPUS_RECEIPT, DATA_DIR, required_corpus_files

    wanted = required_corpus_files(root)
    archive = root / DATA_DIR / ARCHIVE
    prefix = entry["archive_root"] + "/"
    staging_root = root / ".tmp"
    if not check_only:
        staging_root.mkdir(parents=True, exist_ok=True)
    # check 不建临时目录、不改任何文件；流式解压只在内存校验选中的 JSON。
    temporary = (
        tempfile.TemporaryDirectory(prefix="hybridqa-", dir=staging_root)
        if not check_only
        else None
    )
    staged_root = Path(temporary.name) if temporary else None
    hashes: dict[str, str] = {}
    pending: list[str] = []
    try:
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                name = member.name.removeprefix(prefix)
                if name not in wanted:
                    continue
                if not member.isfile() or name in hashes:
                    raise ValueError(f"HybridQA: invalid/duplicate archive member {name}")
                handle = bundle.extractfile(member)
                assert handle is not None
                with handle:
                    content = handle.read()
                digest = hashlib.sha256(content).hexdigest()
                hashes[name] = digest
                target = root / DATA_DIR / name
                if target.is_file():
                    if sha256_file(target) != digest:
                        raise ValueError(
                            f"HybridQA corpus sha256 mismatch: {target}; existing file preserved"
                        )
                elif check_only:
                    raise ValueError(f"Missing {target}; run make fetch-data DATASET=hybridqa")
                else:
                    assert staged_root is not None
                    staged = staged_root / name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(content)
                    pending.append(name)
        if set(hashes) != wanted:
            raise ValueError(
                f"HybridQA archive missing {len(wanted - set(hashes))} required corpus files"
            )
        receipt = {"archive_sha256": entry["sha256"], "files": dict(sorted(hashes.items()))}
        receipt_path = root / DATA_DIR / CORPUS_RECEIPT
        if check_only:
            if (
                not receipt_path.is_file()
                or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt
            ):
                raise ValueError(
                    "HybridQA corpus receipt missing/changed; run make fetch-data DATASET=hybridqa"
                )
        else:
            assert staged_root is not None
            # 整份计划先校验再发布，已有的损坏文件不覆盖。
            for name in pending:
                destination = root / DATA_DIR / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staged_root / name).replace(destination)
            staged_receipt = staged_root / CORPUS_RECEIPT
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
            )
            staged_receipt.replace(receipt_path)
        print(f"  HybridQA: {len(hashes)} corpus files verified", flush=True)
    finally:
        if temporary:
            temporary.cleanup()
