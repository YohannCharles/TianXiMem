"""输入契约的支持范围、按需分派和代码版本指纹。"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

from .plan import INPUT_CONTRACT, InputPlan, digest

SUPPORTED = frozenset(
    {
        "corporatebench",
        "mquake-remastered",
        "memtrapbench",
        "locomo-refined",
        "medmemorybench",
        "halumem",
        "musique",
        "hybridqa",
        "feverous",
    }
)
UNSUPPORTED = {
    "tempreason": "captured instance questions/gold do not align with public L2/L3",
    "longmemeval-s": (
        "independent QA is not aligned in this capture; corpus is used as LoCoMo distractors"
    ),
}
_SPECIAL = {
    "mquake-remastered": "mquake",
    "locomo-refined": "locomo",
    "musique": "musique",
    "hybridqa": "hybridqa",
    "feverous": "feverous",
}


def validate_options(
    dataset: str,
    *,
    add_shape: str = "official",
    time_style: str = "synthetic",
    pool: Path | None = None,
) -> None:
    if dataset not in SUPPORTED:
        raise ValueError(
            f"{INPUT_CONTRACT} unsupported for {dataset}: "
            + UNSUPPORTED.get(dataset, "no audited input adapter")
        )
    if add_shape not in {"official", "alluser"} or (
        add_shape == "alluser" and dataset != "locomo-refined"
    ):
        raise ValueError("AML input uses its own wrappers; alluser is supported only for LoCoMo")
    if time_style not in {"synthetic", "inline"} or (
        time_style == "inline" and dataset != "halumem"
    ):
        raise ValueError("AML inline time is supported only for HaluMem")
    if pool is not None and dataset != "feverous":
        raise ValueError("--aml-pool applies only to FEVEROUS")
    if dataset == "feverous":
        from .feverous import check_pool_available

        check_pool_available(pool)


def required_datasets(dataset: str) -> tuple[str, ...]:
    return (dataset, "longmemeval-s") if dataset == "locomo-refined" else (dataset,)


def policy_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


def source_pins(dataset: str) -> dict[str, str]:
    """声明的源版本；CLI 的准备层负责校验实际字节，不在加载时重复扫描大型数据库。"""
    from ..prepare import entries_for

    return {
        entry["name"]: entry["sha256"]
        for dependency in required_datasets(dataset)
        for entry in entries_for(dependency)
    }


def load_plans(
    root: Path,
    dataset: str,
    *,
    limit: int | None = None,
    spread: bool = False,
    max_questions: int | None = None,
    add_shape: str = "official",
    time_style: str = "synthetic",
    pool: Path | None = None,
) -> list[InputPlan]:
    validate_options(dataset, add_shape=add_shape, time_style=time_style, pool=pool)
    if limit is not None and limit < 0:
        raise ValueError("AML limit must be nonnegative")
    if max_questions is not None and max_questions < 1:
        raise ValueError("AML max_questions must be positive")
    kwargs = {"limit": limit, "spread": spread}
    if dataset in _SPECIAL:
        module = import_module(f".{_SPECIAL[dataset]}", __package__)
        if dataset == "locomo-refined":
            kwargs["alluser"] = add_shape == "alluser"
        if dataset == "feverous":
            kwargs["pool"] = pool
        plans = module.load(root, **kwargs)
    else:
        from .common import checkpoint_plans, corporate_plans, dialogue_plans

        module = import_module(f"..{dataset}", __package__)
        samples = getattr(module, f"load_{dataset}")(root, **kwargs)
        if max_questions is not None:
            from dataclasses import replace

            samples = [replace(s, questions=s.questions[:max_questions]) for s in samples]
        if dataset == "corporatebench":
            plans = corporate_plans(samples)
        elif dataset in {"halumem", "medmemorybench"}:
            plans = checkpoint_plans(samples, time_style=time_style)
        else:
            plans = dialogue_plans(samples)
        # 每个原始检查点/QA 子集分别裁题；合并后不能再裁一次。
        max_questions = None
    policy = policy_fingerprint()
    pins = source_pins(dataset)
    for plan in plans:
        plan.metadata["adapter_sha256"] = policy
        plan.metadata["source_pins"] = pins
    plans = [p.limit_questions(max_questions) for p in plans]
    for plan in plans:
        plan.validate()
    return plans
