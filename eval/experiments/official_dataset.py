"""官方采集按数据集分类、固定清单重放评测与两版成绩对比。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from eval.datasets.aml.plan import AddEvent, InputPlan, SearchEvent
from eval.datasets.official_capture import UserPlan, replay_plan
from eval.datasets.official_groups import (
    classified_rows,
    data_identity,
    load_manifest,
    prepare_groups,
    sha256,
    write_json,
)
from eval.datasets.prepare import PreparationError, ensure_dataset
from eval.datasets.registry import benchmark_dir, capture_dir
from eval.harness import ServiceClient, api_config, build_official_items, pipeline_for, run_judge
from eval.harness.annotate import MARKS
from eval.harness.judge import DATE_MODES
from eval.harness.plan_driver import compile_plan, execute_plan, freeze_run
from eval.harness.run_record import build_record, config_fingerprint, write_record
from eval.reports.official_comparison import compare_runs, coverage_report, write_summary


def checkpoint_plan(plan: UserPlan, namespace: str) -> InputPlan:
    """使用既有逐事件检查点驱动；所有 Add 已成批，不再改正文或批界。"""
    events = tuple(
        AddEvent(event.session_id, event.messages, event.request_id, batch_ready=True)
        if event.kind == "add"
        else SearchEvent(str(event.seq), event.question["query"])
        for event in plan.events
    )
    compiled = compile_plan(
        InputPlan(
            sample=plan.to_sample(),
            events=events,
            contract="official-capture-v1",
            metadata={"batching": "verbatim", "add_shape": "verbatim"},
        ),
        namespace=namespace,
    )
    return replace(
        compiled, metadata=compiled.metadata | {"batching": "verbatim", "add_shape": "verbatim"}
    )


def evaluation_fingerprint(args: argparse.Namespace) -> dict:
    """冻结答案/裁判口径；不保存 API 密钥。产品版本与配置允许在两臂间不同。"""
    root = Path(__file__).resolve().parents[2]
    code = sorted((root / "eval" / "harness").glob("*.py"))
    code += [Path(__file__), root / "eval" / "datasets" / "official_capture.py"]
    archive = benchmark_dir()
    upstream = sorted((archive / ".upstream").rglob("*.py"))
    return {
        "answer_model": api_config.ANSWER_MODEL,
        "judge_model": api_config.JUDGE_MODEL,
        "answer_base": api_config.ANSWER_API_BASE,
        "judge_base": api_config.JUDGE_API_BASE,
        "max_tokens": args.max_tokens,
        "top_k": 100,
        "memory_date": args.memory_date,
        "annotate_mark": args.annotate_mark,
        "scheduling": "serial users; ordered checkpoints",
        "code_sha256": {str(p.relative_to(root)): sha256(p) for p in code},
        "upstream_sha256": {str(p.relative_to(archive)): sha256(p) for p in upstream},
    }


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path, verify_sources=True)
    if not manifest["questions"]:
        raise ValueError("固定清单没有可评分题目")
    identity = data_identity(manifest)
    capture = Path(manifest["capture_dir"])
    questions = {int(row["seq"]): row for row in manifest["questions"]}
    classified = {int(row["seq"]): row for row in classified_rows(capture)}
    for seq, expected in questions.items():
        actual = classified.get(seq)
        if (
            not actual
            or not actual["scorable"]
            or any(actual[key] != expected[key] for key in ("dataset", "user_id"))
        ):
            raise ValueError(f"固定清单与源数据的归属/金标不符：seq={seq}")
    plans = replay_plan(capture, only_seqs=set(questions))
    if {int(r["seq"]) for p in plans for r in p.questions} != set(questions):
        raise ValueError("timeline 未覆盖所选全部题目")
    compiled = [checkpoint_plan(p, namespace=f"{args.run_id}:{identity}") for p in plans]
    for family in sorted({row["dataset"] for row in questions.values()}):
        ensure_dataset(
            "doc-pp" if family == "docpp" else family,
            benchmark_dir(),
            offline=args.offline,
            purpose="judge",
        )
    required = {
        "AML_BASE_URL": api_config.ANSWER_API_BASE,
        "AML_API_KEY": api_config.ANSWER_API_KEY,
        "AML_MODEL": api_config.ANSWER_MODEL,
        "JUDGE_API_BASE": api_config.JUDGE_API_BASE,
        "JUDGE_API_KEY": api_config.JUDGE_API_KEY,
        "JUDGE_MODEL": api_config.JUDGE_MODEL,
    }
    if missing := [key for key, value in required.items() if not value]:
        raise ValueError(f"未配置答案/裁判端点：{missing}；使用 uv run --env-file .env")
    pipeline = pipeline_for(benchmark_dir(), "official-capture")
    if not pipeline.is_file():
        raise ValueError(f"评分 pipeline 缺失：{pipeline}")
    output = Path(args.reports_dir) / "runs" / args.run_id
    runtime = {
        "run_id": args.run_id,
        "service_revision": args.service_revision,
        "service_revision_note": "caller-declared; verify against the running service checkout",
        "service_base": args.base_url,
        "top_k": 100,
        "data_identity": identity,
        "evaluation": evaluation_fingerprint(args),
        "config_fingerprint": config_fingerprint(args.profile, configs_dir=Path(args.configs_dir)),
        "users": {p.user_id: c.sample.user_id for p, c in zip(plans, compiled, strict=True)},
    }
    runtime_path = output / "run-manifest.json"
    if runtime_path.exists():
        if json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
            raise ValueError("运行的版本、输入、模型或配置变化；请使用新的 run-id")
        snapshot = load_manifest(output / "selection-manifest.json")
        if data_identity(snapshot) != identity:
            raise ValueError("已保存的题目清单变化；请使用新的 run-id")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("运行目录含有未冻结的旧产物；请使用新的 run-id")
    fingerprint = freeze_run(output, compiled, execution=runtime)
    fingerprint |= {"batching": "verbatim", "add_shape": "verbatim"}
    if not runtime_path.exists():
        shutil.copyfile(manifest_path, output / "selection-manifest.json")
        write_json(runtime_path, runtime)
        for name in runtime["config_fingerprint"]["snapshot_files"]:
            target = output / "config-snapshot" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(args.configs_dir) / name, target)
    results = []
    samples = [p.sample for p in compiled]
    client = ServiceClient(args.base_url)
    print(f"{args.run_id}：固定 {len(questions)} 题 / {len(plans)} 个用户 → {output}", flush=True)
    try:
        for index, (source, wire) in enumerate(zip(plans, compiled, strict=True), 1):
            print(f"[{index}/{len(plans)}] {len(source.questions)} 题", flush=True)
            user_dir = output / wire.sample.user_id
            hits = execute_plan(wire, client=client, out_dir=user_dir)
            items = build_official_items(
                source.questions, hits, date_mode=args.memory_date, annotate_mark=args.annotate_mark
            )
            results.extend(
                run_judge(
                    pipeline,
                    items,
                    user_dir,
                    dataset="official-capture",
                    max_tokens=args.max_tokens,
                )
            )
            summary = write_summary(output)
            print(f"已判 {summary['completed']}/{summary['expected']}", flush=True)
    finally:
        client.close()
        write_summary(output)
        record = build_record(
            run_id=args.run_id,
            step="official-capture-comparison",
            profile=args.profile,
            bench_dir=capture,
            samples=samples,
            results=results,
            data_fingerprint=fingerprint,
            models={"llm": api_config.ANSWER_MODEL, "judge": api_config.JUDGE_MODEL},
            configs_dir=Path(args.configs_dir),
            notes="固定采集 seq；原文有序重放；失败阻断后续 Add，逐事件检查点续跑。",
        )
        write_record(record, Path(args.reports_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="分类所有 Search，生成固定可评分清单")
    prepare.add_argument("--capture-dir", default=None)
    prepare.add_argument("--output", default=None)
    prepare.add_argument("--report-dir", default="eval/reports/runs/official-dataset-groups")
    prepare.add_argument("--per-dataset", type=int, default=None)
    prepare.add_argument("--datasets", nargs="+", default=None)
    execute = sub.add_parser("run", help="按固定 manifest 跑 Add → Search → answer → judge")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--service-revision", required=True, help="所运行服务的 Git SHA/版本标签")
    execute.add_argument("--base-url", default="http://127.0.0.1:8000")
    execute.add_argument("--reports-dir", default="eval/reports")
    execute.add_argument("--configs-dir", default="configs")
    execute.add_argument("--profile", default="local")
    execute.add_argument("--max-tokens", type=int, default=1024)
    execute.add_argument("--memory-date", default="none", choices=DATE_MODES)
    execute.add_argument("--annotate-mark", default="paren", choices=MARKS)
    execute.add_argument("--offline", action="store_true")
    summarize = sub.add_parser("summarize", help="离线汇总单次进度和成绩")
    summarize.add_argument("--run-dir", required=True)
    compare = sub.add_parser("compare", help="核验同题同模型后输出 Markdown/CSV/JSON 对比表")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--left-name", default="v1.0")
    compare.add_argument("--right-name", default="v1.1.1")
    compare.add_argument("--output", default="eval/reports/runs/official-comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            capture = Path(args.capture_dir) if args.capture_dir else capture_dir()
            output = Path(args.output) if args.output else capture / "grouped"
            manifest = prepare_groups(
                capture, output, per_dataset=args.per_dataset, datasets=args.datasets
            )
            coverage_report(manifest, Path(args.report_dir))
            print(f"分类文件 → {output / 'searches'}\n固定清单 → {output / 'manifest.json'}")
            print(f"分类统计 → {Path(args.report_dir) / 'coverage.md'}")
        elif args.command == "run":
            if (
                args.max_tokens < 1
                or args.run_id in ("", ".", "..")
                or Path(args.run_id).name != args.run_id
            ):
                raise ValueError("max-tokens 必须为正，run-id 必须是单层目录名")
            return run(args)
        elif args.command == "summarize":
            summary = write_summary(Path(args.run_dir))
            print(f"完成 {summary['completed']}/{summary['expected']} → {args.run_dir}/summary.md")
        else:
            compare_runs(
                Path(args.left),
                Path(args.right),
                Path(args.output),
                args.left_name,
                args.right_name,
            )
            print(f"对比表 → {args.output}/comparison.md / comparison.csv / comparison.json")
    except (ValueError, OSError, PreparationError) as error:
        print(f"✗ {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
