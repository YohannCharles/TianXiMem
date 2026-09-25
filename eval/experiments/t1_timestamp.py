"""T1 —— 时间戳前缀 **带 / 不带**（§13 / §11.3）。

```text
不带（对照）  packaging.inject_abs_time = false   ← **v1 定稿口径**
带            packaging.inject_abs_time = true    ← 只为这个对照存在
```

**它决定什么**：`content` 的渲染方式（§13 的"决定什么"列）。§11.3 的结论是"不加"，
两条**互相独立**的裁判规则指向同一个动作——粒度变细、相对↔绝对互转。
T1 是**把那条结论在代理评测上验一遍**，不是重新决定它。

## ⚠ 这是"贵"消融项：两臂必须跑在**两个集合**上

正文一改，**embedding 输入就改**（不变式 I1：同一份渲染既是索引输入也是 `content`），
所以：

| 后果 | 为什么不能省 |
| --- | --- |
| 向量要**重算** | 缓存键是渲染后文本的哈希 ⇒ 自动失效（§7.2 的设计） |
| 两臂要**分集合** | 共用一个集合时，"检索到的是哪一臂的向量"**看不出来**——而两臂的向量不一样 |
| 两臂各有一份**冻结配置** | §13：`configs/runs/<arm>/`。
"当时的 local.yaml 大概是这样"在 Step 5 之后就重建不了了 |

⇒ 本脚本把两臂声明成 `ARM_*`，其余交给 [`run.py`](./run.py)。

## 用法

```bash
# 1) 冻结两臂的配置快照（**跑之前**做，之后改 configs/*.yaml 不影响已冻结的臂）
uv run python eval/experiments/t1_timestamp.py --freeze

# 2) 两个服务（各带一份快照）——**服务自己读配置，脚本不代管进程**
TIANXI_CONFIG_DIR=configs/runs/t1-plain TIANXI_PROFILE=local make serve   # :8000
TIANXI_CONFIG_DIR=configs/runs/t1-dated TIANXI_PROFILE=local make serve   # 另一个端口

# 3) 跑两臂（同一份数据、同一批题）
uv run python eval/experiments/t1_timestamp.py --execute \
    --plain-url http://127.0.0.1:8000 --dated-url http://127.0.0.1:8001

# 4) 比一比
uv run python eval/experiments/t1_timestamp.py --compare <run-id-plain> <run-id-dated>
```

⚠ **`--limit` 只给冒烟用**：截断跑会在数据指纹里留警示，**两臂的数字不能拿它定论**。
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
EXIT_FAILED: Final[int] = 1
EXIT_PRECONDITION_FAILED: Final[int] = 2

#: 冻结快照的根目录。**它是 `TIANXI_CONFIG_DIR` 要指向的地方**（configs/CLAUDE.md）。
RUNS_DIR: Final[Path] = Path("configs/runs")

#: 基线配置（两份都要**原样**进快照，否则臂跑的是内置默认值而不是今天的值）。
BASE_FILES: Final[tuple[str, ...]] = ("default.yaml", "local.yaml")


@dataclass(frozen=True, slots=True)
class Arm:
    """一条臂 = 一份冻结配置 + 一个集合。**两臂只差 `inject_abs_time`。**"""

    name: str
    inject_abs_time: bool
    collection: str

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.name

    def overrides(self) -> dict:
        """这份快照相对基线**必动的两项**（其余全部继承 `default.yaml`）。

        ⚠ `storage.qdrant.collection` 是**必须**动的：两臂的向量不同，混在一个集合里
        检索到的是哪一臂**看不出来**（`configs/CLAUDE.md` 的 V9 同一条风险的形状）。
        """
        return {
            "storage": {"qdrant": {"collection": self.collection}},
            "packaging": {"inject_abs_time": self.inject_abs_time},
        }


#: 两臂。**顺序即"对照在前、变体在后"**。
ARM_PLAIN: Final[Arm] = Arm(name="t1-plain", inject_abs_time=False, collection="memories_t1_plain")
ARM_DATED: Final[Arm] = Arm(name="t1-dated", inject_abs_time=True, collection="memories_t1_dated")
ARMS: Final[tuple[Arm, ...]] = (ARM_PLAIN, ARM_DATED)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def freeze(arm: Arm, *, configs_dir: Path = Path("configs")) -> list[Path]:
    """把当前 `configs/` 冻结成这个臂的**独立快照**（§13）。

    * `default.yaml`：**逐字复制**（保住那些写满理由的注释——它们是配置的一部分）
    * `local.yaml`：解析后套上本臂的覆盖项再写回，**带头注释**记录冻结时刻与基线哈希

    ⚠ 覆盖项**写在 `local.yaml` 里而不是 `default.yaml`**：profile 文件是"与基线的差异"
    该待的地方（`configs/CLAUDE.md`），而"这条臂改了哪两件事"正是这次对照的全部内容。
    """
    arm.dir.mkdir(parents=True, exist_ok=True)

    default_text = (configs_dir / "default.yaml").read_text(encoding="utf-8")
    (arm.dir / "default.yaml").write_text(default_text, encoding="utf-8")

    local = yaml.safe_load((configs_dir / "local.yaml").read_text(encoding="utf-8")) or {}
    merged = _deep_merge(local, arm.overrides())
    digest = hashlib.sha256(default_text.encode("utf-8")).hexdigest()[:16]
    frozen_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    prefix_word = "带" if arm.inject_abs_time else "不带"
    header = (
        f"# T1 臂 `{arm.name}` 的冻结配置（§13 的配置指纹）——**由 t1_timestamp.py --freeze 生成**\n"
        f"# 冻结时刻：{frozen_at}（UTC）· 基线 `configs/default.yaml` 的 sha256[:16] = {digest}\n"
        f"# 本臂相对基线**只动两件事**：\n"
        f"#   packaging.inject_abs_time = {str(arm.inject_abs_time).lower()}"
        f"   （正文{prefix_word}日期前缀）\n"
        f"#   storage.qdrant.collection = {arm.collection}   （两臂的向量不同，必须分集合）\n"
        f"# ⚠ 冻结之后 `configs/*.yaml` 再改**不影响本臂**——这正是 §13 要的可追溯性。\n"
    )
    body = yaml.safe_dump(merged, allow_unicode=True, sort_keys=True)
    (arm.dir / "local.yaml").write_text(header + body, encoding="utf-8")
    return sorted(arm.dir.iterdir())


def verify(arm: Arm) -> list[str]:
    """核对快照**确实声明了这个臂该有的两件事**，返回问题清单（空 = 通过）。

    ⚠ 这条检查是这套脚手架里最便宜、也最要紧的一条：快照一旦掉队（比如两份都成了
    `false`），你会**认真地跑完一次 T1 并得到"两臂没有差别"**——而那个结论是假的。
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
    actual = (local.get("packaging") or {}).get("inject_abs_time")
    if actual != arm.inject_abs_time:
        problems.append(
            f"{local_path} 的 inject_abs_time={actual!r}，本臂应为 {arm.inject_abs_time!r}"
        )
    collection = ((local.get("storage") or {}).get("qdrant") or {}).get("collection")
    if collection != arm.collection:
        problems.append(f"{local_path} 的 collection={collection!r}，本臂应为 {arm.collection!r}")
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


def _commands(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> list[str]:
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return [
        f"TIANXI_CONFIG_DIR={arm.dir} TIANXI_PROFILE=local make serve",
        (
            "uv run --env-file .env python -m eval.experiments.run"
            f" --dataset {dataset} --base-url {base_url}"
            f" --configs-dir {arm.dir} --profile local"
            f" --switches '{json.dumps(_switches(arm))}'"
            f" --step step-3 --run-id {arm.name}{limit_arg}"
        ),
    ]


def _switches(arm: Arm) -> dict:
    """写进 run record 的消融声明（`config_fingerprint` 的 `switches`）。"""
    return {"packaging.inject_abs_time": arm.inject_abs_time}


def _print_plan(arm: Arm, *, dataset: str, base_url: str, limit: int | None) -> None:
    print(f"\n【{arm.name}】inject_abs_time={str(arm.inject_abs_time).lower()} → {arm.collection}")
    for line in _commands(arm, dataset=dataset, base_url=base_url, limit=limit):
        print(f"  {line}")


def compare(plain_id: str, dated_id: str, *, reports_dir: Path) -> int:
    """读两份 run record，逐项打印差异。**数字只在 `eval/reports/` 里算。**"""
    records = {}
    for label, run_id in (("plain", plain_id), ("dated", dated_id)):
        path = reports_dir / "runs" / f"{run_id}.json"
        if not path.exists():
            print(f"找不到 {path}", file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        records[label] = json.loads(path.read_text(encoding="utf-8"))

    plain, dated = records["plain"], records["dated"]
    print(f"overall：plain={plain['scores']['overall']}  dated={dated['scores']['overall']}")
    categories = sorted(set(plain["breakdown"]) | set(dated["breakdown"]))
    for category in categories:
        left = (plain["breakdown"].get(category) or {}).get("accuracy")
        right = (dated["breakdown"].get(category) or {}).get("accuracy")
        print(f"  {category}: plain={left}  dated={right}")
    print(
        "拒答：plain={}  dated={}".format(
            plain["breakdown"].get("abstention"), dated["breakdown"].get("abstention")
        )
    )
    print(
        "\n⚠ 这只是**一次**两臂之差。§11.3 的两条机制（粒度变细 / 相对↔绝对互转）"
        "无法从总分上分开——差异要归因，得看逐题判语（`runs/<run_id>/*/labels.jsonl`）。"
    )
    print("⚠ 结论写进 eval/reports/ledger.md 与 docs/experiments.md 的结论列，别只留在终端。")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t1_timestamp.py", description="§13 的 T1 实验脚手架")
    parser.add_argument("--freeze", action="store_true", help="按当前 configs/ 重新冻结两臂快照")
    parser.add_argument("--execute", action="store_true", help="两臂各跑一轮（服务需已就绪）")
    parser.add_argument("--compare", nargs=2, metavar=("PLAIN_RUN_ID", "DATED_RUN_ID"))
    parser.add_argument("--dataset", default="locomo-refined")
    parser.add_argument("--limit", type=int, default=None, help="冒烟用；会在数据指纹里留警示")
    parser.add_argument("--plain-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dated-url", default="http://127.0.0.1:8001")
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

    for arm in ARMS:
        notice = base_drift(arm)
        if notice:
            print(f"⚠ {notice}")

    if args.compare:
        return compare(args.compare[0], args.compare[1], reports_dir=Path(args.reports_dir))

    urls = {ARM_PLAIN.name: args.plain_url, ARM_DATED.name: args.dated_url}
    if not args.execute:
        print("两臂快照齐备。要跑的话（服务各带一份快照）：")
        for arm in ARMS:
            _print_plan(arm, dataset=args.dataset, base_url=urls[arm.name], limit=args.limit)
        print("\n⚠ 两臂**必须分集合**（已写进各自快照），且都跑完整数据集。")
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
                json.dumps(_switches(arm)),
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

    print(f"\n两臂跑完：{ARM_PLAIN.name} / {ARM_DATED.name}。用 `--compare` 看差异。")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
