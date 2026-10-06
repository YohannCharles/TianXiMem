"""评测材料的本地目录映射（D35）；旧归档标识与字节保持不变。

所有路径相对调用者提供的数据根目录。旧的平铺归档与合成 fixture 仍可只读加载；
新下载一律写入按数据集组织的路径。
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

DEFAULT_DATASET_DIR = "dataset"

_FLAT = {
    "aml_readme.md": ".upstream/aml/README.md",
    "locomo_refined.json": "locomo-refined/locomo_refined.json",
    "questions.jsonl": "locomo-refined/questions.jsonl",
    "conversations.jsonl": "locomo-refined/conversations.jsonl",
    "locomo_refined_readme.md": "locomo-refined/README.md",
    "lme_s_cleaned.json": "longmemeval-s/lme_s_cleaned.json",
    "lme_readme.md": "longmemeval-s/README.md",
    "clbench.jsonl": "clbench/clbench.jsonl",
    "locomo10.json": "locomo/locomo10.json",
    "scriptmem_q.jsonl": "scriptmem/scriptmem_q.jsonl",
    "scriptmem_readme.md": "scriptmem/README.md",
    "pm_32k.csv": "personamem-v1/pm_32k.csv",
    "pm_questions_128k.csv": "personamem-v1/pm_questions_128k.csv",
    "pm_questions_1M.csv": "personamem-v1/pm_questions_1M.csv",
    "pmv2.md": ".upstream/personamem-v2/hf_README.md",
    "rh.md": ".legacy/rh.md",
    "rh2.md": ".legacy/rh2.md",
    "rh3.md": ".legacy/rh3.md",
}

PIPELINES = {
    "locomo-refined": "pipeline_locomo-refined.py",
    "longmemeval-s": "pipeline_longmemeval-s.py",
    "clbench": "clb_pipeline.py",
    "beam": "pipeline_beam.py",
    "scriptmem": "pipeline_scriptmem.py",
    "personamem-v1": "pipeline_v1_personamem.py",
    "personamem-v2": "pipeline_v2_personamem.py",
}


def local_path(name: str) -> Path:
    """清单中的旧标识 → 新的相对路径；禁止绝对路径和向上逃逸。"""
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError(f"非法归档路径：{name!r}")
    if name in _FLAT:
        return Path(_FLAT[name])
    if name in PIPELINES.values():
        return Path(".upstream/aml") / name
    slug, _, rest = name.partition("/")
    if slug == "medmemorybench-code":
        return Path(".upstream/medmemorybench") / rest
    if name == "halumem/eval_tools.py":
        return Path(".upstream/halumem/eval_tools.py")
    if name in {
        "hybridqa/evaluate_script.py",
        "hybridqa/WikiTables-WithLinks.README.md",
        "feverous/feverous_scorer.py",
        "feverous/download_data.sh",
    }:
        return Path(".upstream") / name
    if slug == "doc-pp" and rest != "README.md" and not rest.startswith("data.z"):
        return Path(".upstream/doc-pp") / rest
    if slug == "memtrapbench" and rest != "README.md" and not rest.startswith("memtrapbench/"):
        return Path(".upstream/memtrapbench") / rest
    return Path(name)


def archive_file(root: str | Path, name: str) -> Path:
    """解析一个文件，不联网、不写盘；新目录优先，其次兼容旧平铺目录。"""
    root = Path(root)
    current = root / local_path(name)
    legacy = root / name
    return current if current.exists() or not legacy.exists() else legacy


def upstream_aml_dir(root: str | Path, name: str = "pipeline_v2_personamem.py") -> Path:
    """上游 pipeline 模块的 import 目录；兼容旧归档。"""
    return archive_file(root, name).parent
