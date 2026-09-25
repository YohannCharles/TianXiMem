"""A3 —— **Rerank 开 / 关**（§13 / §11.2）。

```text
on    rerank.enabled = true    ← 精排（`Qwen3-Reranker-4B`，100 篇/题 5–12s）
off   rerank.enabled = false   ← 直接用融合名次（**2026-09-25 起的默认值**）
```

**它决定什么**：排序值不值这份算力（§11 主线的验证）。若不值，把算力挪去别处。

## ⚠ 两臂**共用一套向量**——这是 A3 与 T1 最大的不同

`rerank.enabled` 只改**排名**，不改正文 ⇒ 索引、embedding 输入、缓存全都一样。
所以两臂**可以而且应该**用同一个集合（`memories_dev`）；只有 A3 与 T1 之间才需要分集合
（T1 改的是正文）。⇒ **跑 A3 不需要重建索引**，开关一翻即可，代价只是再跑一轮评测。

## 用法

```bash
# 1) 冻结两臂配置（跑之前做；之后改 configs/*.yaml 不影响已冻结的臂）
uv run python -m eval.experiments.a3_rerank --freeze

# 2) 两个服务，各带一份快照（**同一个集合**，端口不同）
TIANXI_CONFIG_DIR=configs/runs/a3-on  TIANXI_PROFILE=local make serve   # :8000
TIANXI_CONFIG_DIR=configs/runs/a3-off TIANXI_PROFILE=local make serve   # :8001

# 3) 两臂各跑一轮（同一份数据、同一批题）
uv run python -m eval.experiments.a3_rerank --execute \
    --on-url http://127.0.0.1:8000 --off-url http://127.0.0.1:8001

# 4) 比一比（都是相对比较，别外推）
uv run python -m eval.experiments.a3_rerank --compare <run-id-on> <run-id-off>
```

⚠ **`--limit` 只给冒烟用**：截断跑会在数据指纹里留警示，两臂的数字不能拿它定论。
⚠ 两臂若跑在**不同时刻**，注意 `rerank.enabled` 之外的配置必须一样——`--freeze` 保证这一点。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml
from eval.experiments.run import main as run_main

EXIT_OK: Final[int] = 0
EXIT_PRECONDITION_FAILED: Final[int] = 2

RUNS_DIR: Final[Path] = Path("configs/runs")
BASE_FILES: Final[tuple[str, ...]] = ("default.yaml", "local.yaml")


@dataclass(frozen=True, slots=True)
class Arm:
    """一条臂 = 一份冻结配置。**两臂只差 `rerank.enabled`。**"""

    name: str
    rerank_enabled: bool

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.name

    def overrides(self) -> dict:
        """相对基线**只动这一件事**——A3 的对照内容就是它。"""
        return {"rerank": {"enabled": self.rerank_enabled}}


ARM_ON: Final[Arm] = Arm(name="a3-on", rerank_enabled=True)
ARM_OFF: Final[Arm] = Arm(name="a3-off", rerank_enabled=False)
ARMS: Final[tuple[Arm, ...]] = (ARM_OFF, ARM_ON)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def freeze(arm: Arm, *, configs_dir: Path = Path("configs")) -> list[Path]:
    """把当前 `configs/` 冻结成这个臂的独立快照（§13）。**与 T1 那份同一套做法。**"""
    arm.dir.mkdir(parents=True, exist_ok=True)
    default_text = (configs_dir / "default.yaml").read_text(encoding="utf-8")
    (arm.dir / "default.yaml").write_text(default_text, encoding="utf-8")

    local = yaml.safe_load((configs_dir / "local.yaml").read_text(encoding="utf-8")) or {}
    merged = _deep_merge(local, arm.overrides())
    digest = hashlib.sha256(default_text.encode("utf-8")).hexdigest()[:16]
    frozen_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# A3 臂 `{arm.name}` 的冻结配置（§13 的配置指纹）——**由 a3_rerank.py --freeze 生成**\n"
        f"# 冻结时刻：{frozen_at}（UTC）· 基线 `configs/default.yaml` 的 sha256[:16] = {digest}\n"
        f"# 本臂相对基线**只动一件事**：\n"
        f"#   rerank.enabled = {str(arm.rerank_enabled).lower()}"
        f"（{'精排开' if arm.rerank_enabled else '精排关，直接用融合名次'}）\n"
        f"# ⚠ 两臂**共用一套向量**（rerank 只改排名）⇒ 集合不用换；变的是排序与耗时。\n"
    )
    body = yaml.safe_dump(merged, allow_unicode=True, sort_keys=True)
    (arm.dir / "local.yaml").write_text(header + body, encoding="utf-8")
    return sorted(arm.dir.iterdir())


def verify(arm: Arm) -> list[str]:
    """核对快照确实声明了这个臂该有的取值（空列表 = 通过）。

    ⚠ 快照掉队（比如两份都成了 `false`）会让人**认真地跑完一次 A3 并得到"两臂没差别"**——
    而那个结论是假的。这条检查是这套脚手架里最便宜、最要紧的一条。
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
    actual = (local.get("rerank") or {}).get("enabled")
    if actual != arm.rerank_enabled:
        problems.append(
            f"{local_path} 的 rerank.enabled={actual!r}，本臂应为 {arm.rerank_enabled!r}"
        )
    return problems


def _commands(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> list[str]:
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return [
        f"TIANXI_CONFIG_DIR={arm.dir} TIANXI_PROFILE=local make serve",
        (
            "uv run --env-file .env python -m eval.experiments.run"
            f" --dataset {dataset} --base-url {base_url}"
            f" --configs-dir {arm.dir} --profile local"
            f" --switches '{json.dumps({'rerank.enabled': arm.rerank_enabled})}'"
            f" --step step-3 --run-id {arm.name}{limit_arg}"
        ),
    ]


def _print_plan(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> None:
    print(f"\n【{arm.name}】rerank.enabled={str(arm.rerank_enabled).lower()}")
    for line in _commands(arm, dataset=dataset, base_url=base_url, limit=limit):
        print(f"  {line}")


def compare(on_id: str, off_id: str, *, reports_dir: Path) -> int:
    """两臂逐项比。**数字只在 `eval/reports/` 里算**，这里只做相对比较。"""
    records = {}
    for label, run_id in (("on", on_id), ("off", off_id)):
        path = reports_dir / "runs" / f"{run_id}.json"
        if not path.exists():
            print(f"找不到 {path}", file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        records[label] = json.loads(path.read_text(encoding="utf-8"))

    on, off = records["on"], records["off"]
    print(f"overall：on={on['scores']['overall']}  off={off['scores']['overall']}")
    for category in sorted(set(on["breakdown"]) | set(off["breakdown"])):
        print(
            f"  {category}: on={(on['breakdown'].get(category) or {}).get('accuracy')}"
            f"  off={(off['breakdown'].get(category) or {}).get('accuracy')}"
        )
    print(
        "\n⚠ 这是**代理评测上的相对比较**（§12.4 / P3）：两份数据集共用契约，"
        "其余四份不适用。结论写进 eval/reports/ledger.md 与 docs/experiments.md 的结论列。"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3_rerank.py", description="§13 的 A3 实验脚手架")
    parser.add_argument("--freeze", action="store_true", help="按当前 configs/ 重新冻结两臂快照")
    parser.add_argument("--execute", action="store_true", help="两臂各跑一轮（服务需已就绪）")
    parser.add_argument("--compare", nargs=2, metavar=("ON_RUN_ID", "OFF_RUN_ID"))
    parser.add_argument("--dataset", default="locomo-refined")
    parser.add_argument("--limit", type=int, default=None, help="冒烟用；会在数据指纹里留警示")
    parser.add_argument("--on-url", default="http://127.0.0.1:8000")
    parser.add_argument("--off-url", default="http://127.0.0.1:8001")
    parser.add_argument("--reports-dir", default="eval/reports")
    args = parser.parse_args(argv)

    if args.freeze:
        for arm in ARMS:
            written = freeze(arm)
            print(f"冻结 {arm.name}：{[p.name for p in written]}")

    problems = [(arm, issue) for arm in ARMS for issue in verify(arm)]
    if problems:
        for arm, issue in problems:
            print(f"✗ {arm.name}：{issue}", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED

    if args.compare:
        return compare(args.compare[0], args.compare[1], reports_dir=Path(args.reports_dir))

    urls = {ARM_ON.name: args.on_url, ARM_OFF.name: args.off_url}
    if not args.execute:
        print("两臂快照齐备。要跑的话（**同一个集合**，两个服务）：")
        for arm in ARMS:
            _print_plan(arm, dataset=args.dataset, base_url=urls[arm.name], limit=args.limit)
        print("\n⚠ 两臂共用一套向量（rerank 只改排名）⇒ 集合不用重建，开关一翻即可。")
        return EXIT_OK

    for arm in ARMS:
        _print_plan(arm, dataset=args.dataset, base_url=urls[arm.name], limit=args.limit)
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
                json.dumps({"rerank.enabled": arm.rerank_enabled}),
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

    print(f"\n两臂跑完：{ARM_ON.name} / {ARM_OFF.name}。用 `--compare` 看差异。")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
