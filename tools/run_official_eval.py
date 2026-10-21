#!/usr/bin/env python3
"""官方采集 8,783 条可评分题的一键入口：**预检 → 重放 → 汇总**。

## 为什么不是"按数据集重新整理一份 add/search/评分集"

官方采集与其余数据集的结构**故意不同**（[`official_capture.py`](../eval/datasets/official_capture.py)
的模块 docstring）：一次 Add 是 AML 造的原文，user 内 add/search **按真实时间交错**。
按数据集重排会同时破坏两件事：

1. **干扰项**：同一个 user 的记忆里本来就有别的家族的 add（以及 `Corpus:` / `Scenario:`
   这类前缀下的其它语料）。拆开重排等于把它们丢掉；
2. **防未来泄漏**：全量里 **41%** 的题在"该 user 的最后一条 add"之前就问。
   先灌完再检索会让这些题读到未来。

⇒ 本入口复用 [`official_dataset`](../eval/experiments/official_dataset.py) 的固定清单
重放：**原始 timeline、原始 payload、逐事件检查点**；answer/judge 由
[`official_capture_pipeline.py`](../eval/harness/official_capture_pipeline.py)
**逐题按家族分派**（14 个家族，见套件里的 `gold_judging` / `judge_ref`）。
本文件只做三件它不管的事：**跑前把"会不会白跑"查掉、给长跑加韧性、把续跑命令说清楚**。

## 用法

```bash
# 1) 只检查，不发任何 Add/Search（先跑这个）
uv run --env-file .env python tools/run_official_eval.py check --base-url http://127.0.0.1:8000

# 2) 正式跑；中断后用**同一条命令**续
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-v1 --service-revision v1 \
  --base-url http://127.0.0.1:8000

# 3) 离线看进度/成绩
uv run python tools/run_official_eval.py status --run-id official-v1

# 4) 对比两次运行
uv run python tools/run_official_eval.py compare \
  --left-run-id official-baseline --right-run-id official-candidate \
  --left-name baseline --right-name candidate
```

**运行比较**：用同一份当前 harness（本仓库）分别请求两个服务：

```bash
# 基线服务：另起端口/库/collection，并分别记录 run-id 与 service-revision
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-baseline --service-revision baseline \
  --base-url http://127.0.0.1:8010 --configs-dir /path/to/v1.0/configs --profile local
# 最新服务：同一个清单、同样的答案/裁判模型与预算
uv run --env-file .env python tools/run_official_eval.py run \
  --run-id official-latest --service-revision main \
  --base-url http://127.0.0.1:8020 --configs-dir /path/to/latest/configs --profile local
```

**不要**用 v1.0 的 harness checkout 跑一臂、最新 checkout 跑另一臂——`compare` 会
核对 `evaluation` 指纹（答案/裁判模型、端点、预算、评分源码），源码不同会拒绝比较。
产品版本与产品配置可以不同，它们只影响服务侧；这正是要比较的东西。

⚠ **服务要用干净的库与集合**：服务侧 `TIANXIMEM_SQLITE_PATH` / Qdrant collection 决定
记忆落在哪；换一次 run 至少换一个 `run-id`（隔离 user_id 命名空间），正式比较建议
同时换 collection。本脚本**不启动服务**，也不会替你选端口。

⚠ 缺省 `--offline`：数据依赖必须已经准备好；缺了会在付费的 Add/Search **之前**退出。
要允许下载就传 `--allow-download`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets.official_capture import replay_plan  # noqa: E402
from eval.datasets.official_groups import data_identity, load_manifest  # noqa: E402
from eval.datasets.prepare import PreparationError, ensure_dataset  # noqa: E402
from eval.datasets.registry import benchmark_dir, capture_dir  # noqa: E402
from eval.harness import api_config, pipeline_for  # noqa: E402
from eval.harness.official_capture_pipeline import supports  # noqa: E402

#: 采集族名 → 归档清单里的 slug。
_ARCHIVE_SLUG = {"docpp": "doc-pp"}

DEFAULT_REPORTS = "eval/reports"


def _manifest_path(capture: Path, override: str | None) -> Path:
    return Path(override) if override else capture / "grouped" / "manifest.json"


def _git_revision() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _check_endpoints() -> list[str]:
    required = {
        "AML_BASE_URL": api_config.ANSWER_API_BASE,
        "AML_API_KEY": api_config.ANSWER_API_KEY,
        "AML_MODEL": api_config.ANSWER_MODEL,
        "JUDGE_BASE_URL": api_config.JUDGE_API_BASE,
        "JUDGE_API_KEY": api_config.JUDGE_API_KEY,
        "JUDGE_MODEL": api_config.JUDGE_MODEL,
    }
    return [name for name, value in required.items() if not value]


def _check_service(base_url: str) -> str | None:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        return f"{base_url}/health 不可达：{type(error).__name__}: {error}"
    try:
        payload = response.json()
    except ValueError:
        return f"{base_url}/health 返回的不是 JSON"
    if payload.get("status") != "ok":
        return f"{base_url}/health 返回 {payload!r}（预期 {{'status': 'ok'}}）"
    return None


def _clear_technical_errors(run_dir: Path) -> int:
    """删掉 `labels.jsonl` 里 `label` 以 `_ERROR` 结尾的行，让下一轮重判它们。

    只删**技术错误**：`WRONG` / `CORRECT` / `LETTERS=…` 这些是真实判定，不动。
    网关抖动造成的 `JUDGE_ERROR` 若一直留在磁盘上，续跑会把它当成"已判"永久跳过——
    这正是"跑完了但有一片题没真正判过"的来源。只解析不了的行（被强杀留下的半行）
    也一并丢弃，它们本来就不是一条完整判定。
    """
    cleared = 0
    for labels in sorted(run_dir.glob("*/labels.jsonl")):
        rows: list[dict] = []
        removed = 0
        malformed = 0
        for line in labels.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if str(row.get("label", "")).endswith("_ERROR"):
                removed += 1
            else:
                rows.append(row)
        if removed == 0 and malformed == 0:
            continue
        cleared += removed
        temporary = labels.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(labels)
    return cleared


def preflight(args: argparse.Namespace) -> tuple[dict, Path]:
    """把"跑到一半才发现"的几类问题全部前置；返回 (manifest, capture)。"""
    capture = Path(args.capture_dir) if args.capture_dir else capture_dir()
    manifest_path = _manifest_path(capture, args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(
            f"✗ 固定清单不存在：{manifest_path}\n"
            "  ⇒ 先生成分类与清单：\n"
            f"    uv run python -m eval.experiments.official_dataset prepare "
            f"--capture-dir {capture}"
        )
    manifest = load_manifest(manifest_path, verify_sources=True)
    questions = manifest.get("questions") or []
    if not questions:
        raise SystemExit("✗ 固定清单没有可评分题目")
    identity = data_identity(manifest)
    plans = replay_plan(capture, only_seqs={int(row["seq"]) for row in questions})
    covered = {int(row["seq"]) for plan in plans for row in plan.questions}
    if covered != {int(row["seq"]) for row in questions}:
        raise SystemExit("✗ timeline 未覆盖固定清单的全部题目")

    # `manifest.questions` 只冻结 seq/dataset/user（防重排），**没有 judge_kind**；
    # 判分语义在套件那一行里，由 `replay_plan` 带进 `plan.questions`。
    scorable = [row for plan in plans for row in plan.questions]
    if len(scorable) != len(questions):
        raise SystemExit(f"✗ 套件可评分行数 {len(scorable)} 与固定清单 {len(questions)} 不一致")
    missing_gold = [
        row["seq"]
        for row in scorable
        if not (
            any(str(value).strip() for value in row.get("gold_answers") or [])
            or row.get("gold_rubric")
        )
    ]
    if missing_gold:
        raise SystemExit(
            "✗ 以下题目标成可评分但没有金标（会全部落成 JUDGE_ERROR）："
            + ", ".join(str(seq) for seq in missing_gold[:20])
        )
    counts = Counter((row["dataset"], str(row.get("judge_kind") or "")) for row in scorable)
    unsupported = sorted({(d, k) for (d, k) in counts if not supports(d, k)})
    replay_only = getattr(args, "replay_only", False)
    if unsupported and not replay_only:
        raise SystemExit(
            "✗ 以下家族 / judge_kind 没有判分实现（会全部落成 JUDGE_ERROR）：\n  "
            + "\n  ".join(f"{d} / {k}" for d, k in unsupported)
        )

    family_errors = []
    for family in ([] if replay_only else sorted({row["dataset"] for row in scorable})):
        try:
            ensure_dataset(
                _ARCHIVE_SLUG.get(family, family),
                benchmark_dir(),
                offline=not args.allow_download,
                purpose="judge",
            )
        except (PreparationError, ValueError) as error:
            family_errors.append(f"{family}: {error}")
    if family_errors:
        raise SystemExit(
            "✗ 裁判依赖没准备好（Add/Search 是付费的，所以在这里退出）：\n  "
            + "\n  ".join(family_errors)
        )

    pipeline = pipeline_for(benchmark_dir(), "official-capture")
    if not replay_only and not pipeline.is_file():
        raise SystemExit(f"✗ 分派 pipeline 缺失：{pipeline}")
    if not replay_only and (missing := _check_endpoints()):
        raise SystemExit(
            "✗ 答案/裁判端点未配置："
            + ", ".join(missing)
            + "\n  ⇒ 用 `uv run --env-file .env` 运行；配置项见 .env.example"
        )
    if not args.skip_service_check and (error := _check_service(args.base_url)):
        raise SystemExit(
            f"✗ 服务预检失败：{error}\n"
            "  ⇒ 先在另一个终端启动服务（见 deploy/CLAUDE.md / make serve）"
        )

    print(
        f"✓ 固定清单 {len(questions)} 题 / {len(plans)} 用户 / "
        f"{sum(plan.add_count for plan in plans)} 条 add"
    )
    print(f"  数据指纹 {identity[:16]}；capture {capture}")
    print("  逐族：")
    for (dataset, kind), count in sorted(counts.items()):
        print(f"    {dataset:<18} {kind:<8} {count:>5}")
    print(f"  分派 pipeline：{pipeline}")
    print(f"  服务：{args.base_url} {'（未检查）' if args.skip_service_check else '✓'}")
    return manifest, capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--capture-dir", default=None, help="采集目录（缺省 TIANXIMEM_CAPTURE_DIR）")
        p.add_argument(
            "--manifest", default=None, help="固定清单（缺省 <capture>/grouped/manifest.json）"
        )
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--allow-download", action="store_true", help="允许下载缺失的裁判依赖")
        p.add_argument("--skip-service-check", action="store_true", help="不检查 /health")

    check = sub.add_parser("check", help="只做预检，不发 Add/Search")
    common(check)

    run = sub.add_parser("run", help="按固定清单重放（中断后同命令续跑）")
    common(run)
    run.add_argument("--run-id", default=None, help="缺省 official-capture-<UTC 时间>")
    run.add_argument("--service-revision", default=None, help="所运行服务的 Git SHA/标签（记录用）")
    run.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    run.add_argument("--configs-dir", default="configs")
    run.add_argument("--profile", default="local")
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument(
        "--replay-only", action="store_true",
        help="只执行 Add/Search 并落盘；等提示词和评分标准确定后再作答、判分",
    )
    run.add_argument("--ingest-workers", type=int, default=1, help="跨用户 Add/Search 并发数")
    run.add_argument("--rerank-local-workers", type=int, default=0)
    run.add_argument("--rerank-memory3-workers", type=int, default=0)
    run.add_argument("--serial-prefix", default=None, help="两臂复用的串行前缀边界 JSON")
    run.add_argument("--prefix-scheduling", choices=("staged", "independent"), default="staged")
    run.add_argument(
        "--judge-workers", type=int, default=0,
        help="正数启用逐题作答/判分与有序 Search 重叠；两臂使用相同值",
    )
    run.add_argument("--memory-date", default="none")
    run.add_argument("--judge-scheduling", choices=("overlap", "after-replay"), default="overlap")
    run.add_argument("--annotate-mark", default="paren")
    run.add_argument("--on-add-error", choices=("skip", "abort"), default="skip")
    run.add_argument("--stop-on-error", action="store_true")
    run.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "先把已有 labels 里 label 以 _ERROR 结尾的题删掉再跑（只删技术错误，"
            "真实判定不动）——网关抖动留下的 JUDGE_ERROR 默认会被当成已判永久跳过"
        ),
    )

    judge = sub.add_parser("judge", help="只从已完成 Search checkpoint 执行 answer → judge")
    judge.add_argument("--source-run-dir", required=True)
    judge.add_argument("--run-id", required=True)
    judge.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    judge.add_argument("--configs-dir", default="configs")
    judge.add_argument("--profile", default="compare")
    judge.add_argument("--max-tokens", type=int, default=1024)
    judge.add_argument("--judge-workers", type=int, default=16)
    judge.add_argument("--memory-date", default="none")
    judge.add_argument("--annotate-mark", default="paren")

    status = sub.add_parser("status", help="离线汇总一个 run（不联网）")
    status.add_argument("--run-id", required=True)
    status.add_argument("--reports-dir", default=DEFAULT_REPORTS)

    compare = sub.add_parser(
        "compare", help="比较两个 run（例如 v1.0 vs 最新）；同题、同答案/裁判口径才允许"
    )
    compare.add_argument("--left-run-id", required=True)
    compare.add_argument("--right-run-id", required=True)
    compare.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS,
        help="两臂共用的根；两臂不在同一根时直接用 official_dataset compare --left/--right",
    )
    compare.add_argument(
        "--output", default=None, help="缺省 <reports>/runs/official-comparison-<left>-vs-<right>"
    )
    compare.add_argument("--left-name", default=None, help="表头名字（缺省用 run-id）")
    compare.add_argument("--right-name", default=None, help="表头名字（缺省用 run-id）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        run_dir = Path(args.reports_dir) / "runs" / args.run_id
        if not (run_dir / "run-manifest.json").is_file():
            print(f"✗ 不是一次已开始的 run：{run_dir}", file=sys.stderr)
            return 2
        from eval.experiments.official_dataset import main as official_main

        code = official_main(["summarize", "--run-dir", str(run_dir)])
        if code == 0:
            print(f"→ {run_dir / 'summary.md'}")
        return code

    if args.command == "compare":
        runs = Path(args.reports_dir) / "runs"
        left, right = runs / args.left_run_id, runs / args.right_run_id
        missing = [path for path in (left, right) if not (path / "run-manifest.json").is_file()]
        if missing:
            print(
                "✗ 以下 run 还没开始或 run-id 写错了：\n  "
                + "\n  ".join(str(path) for path in missing),
                file=sys.stderr,
            )
            return 2
        output = (
            Path(args.output)
            if args.output
            else runs / f"official-comparison-{args.left_run_id}-vs-{args.right_run_id}"
        )
        from eval.experiments.official_dataset import main as official_main

        code = official_main(
            [
                "compare",
                "--left",
                str(left),
                "--right",
                str(right),
                "--left-name",
                args.left_name or args.left_run_id,
                "--right-name",
                args.right_name or args.right_run_id,
                "--output",
                str(output),
            ]
        )
        if code == 0:
            print(f"→ {output}/comparison.md / comparison.csv / comparison.json")
        return code

    if args.command == "judge":
        if args.judge_workers < 1 or args.max_tokens < 1:
            print("✗ judge-workers/max-tokens 必须为正", file=sys.stderr)
            return 2
        from eval.experiments.official_dataset import main as official_main

        return official_main([
            "judge",
            "--source-run-dir", args.source_run_dir,
            "--run-id", args.run_id,
            "--reports-dir", args.reports_dir,
            "--configs-dir", args.configs_dir,
            "--profile", args.profile,
            "--max-tokens", str(args.max_tokens),
            "--judge-workers", str(args.judge_workers),
            "--memory-date", args.memory_date,
            "--annotate-mark", args.annotate_mark,
        ])

    manifest, capture = preflight(args)
    if args.command == "check":
        print("✓ 预检通过（没有发任何 Add/Search）")
        return 0

    run_id = args.run_id or f"official-capture-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}"
    revision = args.service_revision or _git_revision()
    run_dir = Path(args.reports_dir) / "runs" / run_id
    if args.retry_errors and run_dir.is_dir():
        cleared = _clear_technical_errors(run_dir)
        print(f"  重判技术错误：清掉 {cleared} 条 _ERROR 行（下一轮会重新判）", flush=True)
    from eval.experiments.official_dataset import main as official_main

    argv_out = [
        "run",
        "--manifest",
        str(_manifest_path(capture, args.manifest)),
        "--run-id",
        run_id,
        "--service-revision",
        revision,
        "--base-url",
        args.base_url,
        "--reports-dir",
        args.reports_dir,
        "--configs-dir",
        args.configs_dir,
        "--profile",
        args.profile,
        "--max-tokens",
        str(args.max_tokens),
        "--judge-workers",
        str(args.judge_workers),
        "--judge-scheduling",
        args.judge_scheduling,
        "--ingest-workers",
        str(args.ingest_workers),
        "--memory-date",
        args.memory_date,
        "--annotate-mark",
        args.annotate_mark,
        "--on-add-error",
        args.on_add_error,
    ]
    if not args.allow_download:
        argv_out.append("--offline")
    if args.replay_only:
        argv_out.append("--replay-only")
    if args.serial_prefix:
        argv_out.extend(["--serial-prefix", args.serial_prefix])
    argv_out.extend(["--prefix-scheduling", args.prefix_scheduling])
    if args.rerank_local_workers or args.rerank_memory3_workers:
        argv_out.extend(["--rerank-local-workers", str(args.rerank_local_workers),
                         "--rerank-memory3-workers", str(args.rerank_memory3_workers)])
    if args.stop_on_error:
        argv_out.append("--stop-on-error")

    print(
        f"\n开始：run-id={run_id} / {len(manifest['questions'])} 题 / "
        f"服务={args.base_url}（中断后用完全相同的命令续跑）",
        flush=True,
    )
    code = official_main(argv_out)
    if code != 0:
        print(
            "\n✗ 本轮退出码非 0。修好服务后**用同一条命令**重跑会从检查点续；"
            "已有用户的 add/search 不会重发。",
            file=sys.stderr,
        )
    else:
        print(f"\n✓ 完成。离线看进度：python tools/run_official_eval.py status --run-id {run_id}")
        print(f"  产物：{Path(args.reports_dir) / 'runs' / run_id}/summary.md")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
