"""T1 / A3（以及将来的 A0 / A4）共用的**两臂对照脚手架**。

**为什么单独一层**：每个对照实验唯一的真正差别是「改哪几个配置键」；而
「把当前 `configs/` 冻成一份可追溯的快照 → 跑之前核对它没掉队 → 按同一套姿势驱动
两臂 → 逐项比」这件事**一模一样**。

各写一份的代价不是行数，而是**两份 `verify` 会各自漂移**：快照掉队时两边看起来都
"配置齐全"，你会**认真地跑完一次对照、得到"两臂没有差别"**——而那个结论是假的
（§13 的开关纯度规则）。那正是这条检查本身要防的东西。

⇒ **`verify()` 的判据从 `arm.overrides()` 现算**：`overrides()` 里的每个叶子都必须
原样落在冻结的 `local.yaml` 里。于是"这个臂该改什么"与"怎么核对它"**不可能漂移**。

**怎么加一个新对照**：定义一条 `Arm` 子类（字段 + `overrides()` + `switches()`），
再给几个类常量（标签、快照头部那句人话、CLI 文字），最后 `ARMS = (...)`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

import yaml
from eval.experiments.run import main as run_main

EXIT_OK: Final[int] = 0
EXIT_PRECONDITION_FAILED: Final[int] = 2

#: 冻结快照的根目录。**它是 `TIANXI_CONFIG_DIR` 要指向的地方**（configs/CLAUDE.md）。
RUNS_DIR: Final[Path] = Path("configs/runs")

#: 基线配置（两份都要**原样**进快照，否则臂跑的是内置默认值而不是今天的值）。
BASE_FILES: Final[tuple[str, ...]] = ("default.yaml", "local.yaml")


@dataclass(frozen=True, slots=True)
class Spec:
    """一个对照实验**除「改哪些配置键」之外**的全部差异——都是文字与 CLI 形状。

    ⚠ 往这里加一个会改变**行为**的字段之前先想清楚：那说明这条差异属于
    `Arm.overrides()`，不属于脚手架。
    """

    #: 实验编号，出现在快照头部（`T1` / `A3` / …）
    label: str
    #: 生成快照的脚本名——写进头部，让人知道该重跑谁
    generator: str
    #: 快照头部那句人话：**这几件事为什么必须动**（取值本身由 `overrides()` 自动列出）。
    #: 每行自带 `# ` 前缀，因为它直接进 YAML 注释。
    note: str

    prog: str
    description: str

    #: `--compare` 两个参数的名字，兼作打印时的左右标签
    compare_labels: tuple[str, str]
    #: 两条臂的 `--<x>-url` 旗标与其默认值
    url_flags: tuple[tuple[str, str], ...]

    #: 不带 `--execute` 时那句"要跑的话怎么跑"
    plan_hint: str
    #: 不带 `--execute` 时那句"跑之前必须知道的事"
    execute_hint: str
    #: `--compare` 结尾那句读法警告
    compare_warning: str


class Arm:
    """一条臂 = 一份冻结配置（+ 它的集合）。**子类用 `@dataclass(frozen=True, slots=True)`。**

    子类至少要提供：`name` 字段、`overrides()`、`switches()`。
    """

    __slots__ = ()

    #: 该实验的**全部文字与 CLI 形状**——两个臂共享同一份
    SPEC: ClassVar[Spec]

    name: str

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.name

    def overrides(self) -> dict:
        """这份快照相对基线**必动的几项**（其余全部继承 `default.yaml`）。"""
        raise NotImplementedError

    def switches(self) -> dict:
        """写进 run record 的消融声明（`config_fingerprint` 的 `switches`）。

        ⚠ **它不一定等于 `overrides()`**：`overrides()` 里可能夹着"必须分集合"这类
        非开关项，而 `switches` 只记**开关**（§13 的对照内容）。
        """
        raise NotImplementedError


# ── 叶子路径（`overrides()` 与校验共用同一套遍历）────────────────────────


def flatten(node: dict, prefix: str = "") -> dict[str, Any]:
    """`{"a": {"b": 1}}` → `{"a.b": 1}`。"""
    out: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        out.update(flatten(value, f"{path}.") if isinstance(value, dict) else {path: value})
    return out


def deep_merge(base: dict, overlay: dict) -> dict:
    """`overlay` 覆盖 `base`，字典**逐层合并**而不是整块替换。"""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _dig(node: dict, path: str) -> Any:
    """按 `flatten()` 的路径取值；中途缺失返回 `None`。"""
    current: Any = node
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


# ── 冻结 / 核对 ────────────────────────────────────────────────────────


def freeze(arm: Arm, *, configs_dir: Path = Path("configs")) -> list[Path]:
    """把当前 `configs/` 冻结成这个臂的**独立快照**（§13）。

    * `default.yaml`：**逐字复制**（保住那些写满理由的注释——它们是配置的一部分）
    * `local.yaml`：解析后套上本臂的覆盖项再写回，**带头注释**记录冻结时刻与基线哈希

    ⚠ 覆盖项**写在 `local.yaml` 里而不是 `default.yaml`**：profile 文件是"与基线的差异"
    该待的地方（`configs/CLAUDE.md`），而"这条臂改了哪几件事"正是这次对照的全部内容。
    """
    spec = arm.SPEC
    arm.dir.mkdir(parents=True, exist_ok=True)

    default_text = (configs_dir / "default.yaml").read_text(encoding="utf-8")
    (arm.dir / "default.yaml").write_text(default_text, encoding="utf-8")

    local = yaml.safe_load((configs_dir / "local.yaml").read_text(encoding="utf-8")) or {}
    merged = deep_merge(local, arm.overrides())
    digest = hashlib.sha256(default_text.encode("utf-8")).hexdigest()[:16]
    frozen_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = "".join(
        f"#   {path} = {value!r}\n" for path, value in flatten(arm.overrides()).items()
    )
    header = (
        f"# {spec.label} 臂 `{arm.name}` 的冻结配置（§13 的配置指纹）"
        f"——**由 {spec.generator} --freeze 生成**\n"
        f"# 冻结时刻：{frozen_at}（UTC）· 基线 `configs/default.yaml` 的 sha256[:16] = {digest}\n"
        f"# 本臂相对基线只动这几件事：\n"
        f"{changed}"
        f"{spec.note}"
        f"# ⚠ 冻结之后 `configs/*.yaml` 再改**不影响本臂**——这正是 §13 要的可追溯性。\n"
    )
    body = yaml.safe_dump(merged, allow_unicode=True, sort_keys=True)
    (arm.dir / "local.yaml").write_text(header + body, encoding="utf-8")
    return sorted(arm.dir.iterdir())


def verify(arm: Arm) -> list[str]:
    """核对快照确实声明**这个臂 `overrides()` 里的每一项**，返回问题清单（空 = 通过）。

    ⚠ 判据**从 `overrides()` 现算**，不另写一份：快照掉队（比如两份都成了 `false`）
    会让人**认真地跑完一次对照并得到"两臂没有差别"**——而那个结论是假的。
    """
    problems: list[str] = []
    if not arm.dir.exists():
        return [f"{arm.dir} 不存在——先跑 `--freeze`"]
    for name in BASE_FILES:
        if not (arm.dir / name).exists():
            problems.append(f"{arm.dir / name} 缺失（基线的那份必须原样在场）")
    local_path = arm.dir / "local.yaml"
    if not local_path.exists():
        return problems
    local = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
    for path, expected in flatten(arm.overrides()).items():
        actual = _dig(local, path)
        if actual != expected:
            problems.append(f"{local_path} 的 {path}={actual!r}，本臂应为 {expected!r}")
    return problems


def base_drift(arm: Arm, *, configs_dir: Path = Path("configs")) -> str | None:
    """快照的基线**是否还等于当前 `configs/default.yaml`**——只提示，不算错。

    漂移是**正常的**（快照本来就该冻住当时的基线），但你要知道自己跑的是哪一份：
    在两臂之间改过 `configs/*.yaml` 而没重新冻结，两臂比的就不是同一套参数，
    **而两边看起来都"配置齐全"**。
    """
    snapshot = arm.dir / "default.yaml"
    current = configs_dir / "default.yaml"
    if not snapshot.exists() or not current.exists():
        return None
    if snapshot.read_bytes() == current.read_bytes():
        return None
    return (
        f"{arm.name} 的基线快照与当前 `{current}` **不一致**——"
        "要按当前配置跑就重新 `--freeze`，否则你比的是冻结那一刻的参数"
    )


# ── 计划与执行 ─────────────────────────────────────────────────────────


def commands(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> list[str]:
    """不带 `--execute` 时打印的两条命令（起服务 / 跑一轮）。**它计划、不代管进程。**"""
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return [
        f"TIANXI_CONFIG_DIR={arm.dir} TIANXI_PROFILE=local make serve",
        (
            "uv run --env-file .env python -m eval.experiments.run"
            f" --dataset {dataset} --base-url {base_url}"
            f" --configs-dir {arm.dir} --profile local"
            f" --switches '{json.dumps(arm.switches())}'"
            f" --step step-3 --run-id {arm.name}{limit_arg}"
        ),
    ]


def print_plan(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> None:
    print(f"\n【{arm.name}】{_describe(arm)}")
    for line in commands(arm, dataset=dataset, base_url=base_url, limit=limit):
        print(f"  {line}")


def _describe(arm: Arm) -> str:
    """`overrides()` 的叶子，一行印出来——"这条臂到底开了什么"一眼可见。"""
    return "  ".join(f"{path}={value!r}" for path, value in flatten(arm.overrides()).items())


def compare(left_id: str, right_id: str, *, spec: Spec, reports_dir: Path) -> int:
    """读两份 run record，逐项打印差异。**数字只在 `eval/reports/` 里算。**"""
    left_label, right_label = spec.compare_labels
    records: dict[str, dict] = {}
    for label, run_id in ((left_label, left_id), (right_label, right_id)):
        path = reports_dir / "runs" / f"{run_id}.json"
        if not path.exists():
            print(f"找不到 {path}", file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        records[label] = json.loads(path.read_text(encoding="utf-8"))

    left, right = records[left_label], records[right_label]
    print(
        f"overall：{left_label}={left['scores']['overall']}"
        f"  {right_label}={right['scores']['overall']}"
    )
    for category in sorted(set(left["breakdown"]) | set(right["breakdown"])):
        print(
            f"  {category}: {left_label}={(left['breakdown'].get(category) or {}).get('accuracy')}"
            f"  {right_label}={(right['breakdown'].get(category) or {}).get('accuracy')}"
        )
    print(f"\n⚠ {spec.compare_warning}")
    return EXIT_OK


def main(arms: tuple[Arm, ...], argv: list[str] | None = None) -> int:
    """两臂脚手架的**唯一一份** `main`：冻结 → 核对 → 计划 / 跑 / 比。"""
    spec = arms[0].SPEC
    parser = argparse.ArgumentParser(prog=spec.prog, description=spec.description)
    parser.add_argument("--freeze", action="store_true", help="按当前 configs/ 重新冻结两臂快照")
    parser.add_argument("--execute", action="store_true", help="两臂各跑一轮（服务需已就绪）")
    parser.add_argument("--compare", nargs=2, metavar=spec.compare_labels)
    parser.add_argument("--dataset", default="locomo-refined")
    parser.add_argument("--limit", type=int, default=None, help="冒烟用；会在数据指纹里留警示")
    for flag, default in spec.url_flags:
        parser.add_argument(flag, default=default)
    parser.add_argument("--reports-dir", default="eval/reports")
    args = parser.parse_args(argv)

    if args.freeze:
        for arm in arms:
            written = freeze(arm)
            print(f"冻结 {arm.name}：{[p.name for p in written]}")

    problems = [(arm, issue) for arm in arms for issue in verify(arm)]
    if problems:
        for arm, issue in problems:
            print(f"✗ {arm.name}：{issue}", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED

    for arm in arms:
        notice = base_drift(arm)
        if notice:
            print(f"⚠ {notice}")

    if args.compare:
        return compare(
            args.compare[0], args.compare[1], spec=spec, reports_dir=Path(args.reports_dir)
        )

    urls = {
        arm.name: getattr(args, flag.removeprefix("--").replace("-", "_"))
        for arm, (flag, _) in zip(arms, spec.url_flags, strict=True)
    }
    if not args.execute:
        print(spec.plan_hint)
        for arm in arms:
            print_plan(arm, dataset=args.dataset, base_url=urls[arm.name], limit=args.limit)
        print(f"\n⚠ {spec.execute_hint}")
        return EXIT_OK

    for arm in arms:
        print_plan(arm, dataset=args.dataset, base_url=urls[arm.name], limit=args.limit)
        code = run_main(
            [
                "--dataset",
                args.dataset,
                "--base-url",
                urls[arm.name],
                "--configs-dir",
                str(arm.dir),
                "--profile",
                "local",
                "--step",
                "step-3",
                "--run-id",
                arm.name,
                "--switches",
                json.dumps(arm.switches()),
                "--reports-dir",
                args.reports_dir,
                *(("--limit", str(args.limit)) if args.limit is not None else ()),
            ]
        )
        if code != EXIT_OK:
            print(
                f"{arm.name} 那一轮没跑成（退出码 {code}）——两臂缺一臂，这个对照不成立",
                file=sys.stderr,
            )
            return code

    print(f"\n两臂跑完：{' / '.join(a.name for a in arms)}。用 `--compare` 看差异。")
    return EXIT_OK
