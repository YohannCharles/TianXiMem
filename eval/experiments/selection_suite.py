"""Prepare, verify and execute a shared ID-only Memory600/captured-10% suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from eval.datasets.aml.plan import AddEvent, InputPlan, SearchEvent, digest
from eval.datasets.registry import benchmark_dir, capture_dir
from eval.datasets.selection_suite import (
    load_native,
    load_selection,
    prepare,
    question_identity,
    row_audits,
    sample_scope,
    sha256,
    verify_sources,
    write_ids,
    write_json,
)
from eval.harness import (
    ServiceClient,
    build_input_items,
    pipeline_for,
    run_judge,
    shape_batch,
)
from eval.harness.judge import DATE_MODES
from eval.harness.plan_driver import compile_plan, execute_plan, freeze_run
from eval.harness.run_record import config_fingerprint


def evaluation_fingerprint(args) -> dict:
    from eval.harness import api_config

    root = Path(__file__).resolve().parents[2]
    code = list((root / "eval/harness").glob("*.py"))
    code += list((root / "eval/datasets").glob("*.py")) + [Path(__file__)]
    return {
        "answer_model": api_config.ANSWER_MODEL,
        "judge_model": api_config.JUDGE_MODEL,
        "answer_base": api_config.ANSWER_API_BASE,
        "judge_base": api_config.JUDGE_API_BASE,
        "max_tokens": args.max_tokens,
        "top_k": 100,
        "memory_date": args.memory_date,
        "scheduling": "serial isolated users; ordered checkpoints",
        "code_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(code)},
        "upstream_sha256": {
            str(p.relative_to(args.data_root)): sha256(p)
            for p in sorted((args.data_root / ".upstream").rglob("*.py"))
        },
    }


def build_plans(manifest: dict, root: Path, capture: Path, bundle: Path, namespace: str):
    """Resolve all inputs before the first HTTP call; never substitute at run time."""
    wanted = defaultdict(set)
    native_rows = {}
    for row in manifest["questions"]:
        wanted[row["dataset"]].add(row["native_qid"])
        native_rows[(row["dataset"], row["native_qid"])] = row
    groups, found = [], set()
    for sample in load_native(root, wanted):
        scope = sample_scope(sample)
        entries = []
        for question in sample.questions:
            row = native_rows[(sample.dataset, question.qid)]
            identity = question_identity(question)
            if scope != row.get("scope_sha256", scope) or identity != row.get(
                "evaluation_sha256", identity
            ):
                raise ValueError(f"native source/gold changed: {sample.dataset}/{question.qid}")
            row.update(scope_sha256=scope, evaluation_sha256=identity)
            entries.append(row)
            found.add((sample.dataset, question.qid))
        groups.append((sample, entries, "native"))
    if found != set(native_rows):
        raise ValueError(f"selected native questions missing: {sorted(set(native_rows) - found)}")
    jobs = []
    for sample, entries, route in groups:
        events = []
        for session in sample.sessions:
            messages = shape_batch(
                session.messages,
                dataset=sample.dataset,
                speaker_names=sample.speaker_names,
                shape="official",
            )
            events.append(AddEvent(session.session_id, tuple(messages)))
        events.extend(SearchEvent(q.qid, q.question) for q in sample.questions)
        plan = compile_plan(
            InputPlan(
                sample,
                tuple(events),
                contract=f"selection-{route}-v1",
                metadata={
                    "route": route,
                    "add_shape": "official",
                    "source_scope_sha256": sample_scope(sample),
                },
            ),
            namespace=namespace,
        )
        jobs.append((plan, entries))
    if {row["id"] for _, rows in jobs for row in rows} != {r["id"] for r in manifest["questions"]}:
        raise ValueError("execution plans do not cover the fixed selection")
    return jobs


def selection_report(manifest: dict, path: Path, jobs=None) -> None:
    counts = manifest["counts"]
    lines = [
        "# 共享固定题单",
        "",
        f"版本：{manifest['version']}；清单 SHA-256：`{digest(manifest)}`。",
        "",
        "题池是 Memory600 与可评分官方采集 Search 的分层 10% 并集；不加入旧冻结评测题。",
        "仅采用已实现的 native 加载器与 pipeline；未支持/无法映射的题目跳过并列明原因。",
        "该混合套件不代表逐请求官方复现，来源组分数也不能当成 AML 榜分。",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in counts.items() if isinstance(value, int))
    lines += ["", "| 执行来源 | 最终题数 |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(counts["routes"].items()))
    if "official_strata" in counts:
        lines += ["", "| 官方抽样数据集 | 采集记录数 |", "|---|---:|"]
        lines.extend(
            f"| {key} | {value} |" for key, value in sorted(counts["official_strata"].items())
        )
    replaced = [r for r in manifest["questions"] if r["route"] == "native"]
    lines += [
        "",
        f"采用现有原生评测：{len(replaced)}。",
        "Git 题单只保留原生题号和少量审计注释；语料/评分指纹在本地运行时生成。",
    ]
    if any(r["members"] for r in manifest["questions"]):
        duplicates = sum(len(r["members"]) > 1 for r in manifest["questions"])
        lines += ["", f"同源合并组：{duplicates}。"]
    if manifest["excluded"]:
        lines += ["", "跳过原因（不补抽，不改原始数据）：", ""]
        lines.extend(
            f"- {reason}：{count}"
            for reason, count in sorted(
                Counter(r.get("reason", "audit_failed") for r in manifest["excluded"]).items()
            )
        )
        lines.extend(
            f"- 审计失败：{r['origin']}/{r['origin_id']}"
            for r in manifest["excluded"]
            if not r.get("reason")
        )
    pending = sum(bool(row_audits(r)) for r in manifest["questions"])
    lines += ["", f"带待审/审计标记的最终题目：{pending}。标记保留，不把未独立审计当成合格。"]
    if jobs is not None:
        stats = [p.fingerprint() for p, _ in jobs]
        lines += [
            "",
            f"离线计划：{len(jobs)} 个隔离用户、{sum(s['n_adds'] for s in stats)} 次 Add、"
            f"{sum(s['n_messages'] for s in stats)} 条投喂消息。",
            "离线 verify 不调用服务、embedding、答案模型或裁判。",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(output: Path) -> dict:
    manifest = load_selection(output / "selection.json")
    expected = {r["id"]: r for r in manifest["questions"]}
    results = {}
    for path in sorted(output.glob("*/task-results.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["id"] not in expected or row["id"] in results:
                raise ValueError("unexpected or duplicate scored task")
            results[row["id"]] = row

    def stats(ids):
        rows = [results[key] for key in ids if key in results]
        correct = sum(r["result"]["is_correct"] for r in rows)
        return {
            "expected": len(ids),
            "completed": len(rows),
            "missing": len(ids) - len(rows),
            "correct": correct,
            "accuracy": correct / len(ids) if rows and len(rows) == len(ids) else None,
            "technical_errors": sum(
                r["result"]["label"] in {"ANSWER_ERROR", "JUDGE_ERROR"} for r in rows
            ),
        }

    groups = {"overall": stats(set(expected))}
    for field in ("dataset", "route"):
        for value in sorted({r[field] for r in expected.values()}):
            groups[f"{field}:{value}"] = stats(
                {key for key, r in expected.items() if r[field] == value}
            )
    for origin in ("memory600", "official"):
        count_key = "memory600" if origin == "memory600" else "official_sample"
        if count_key not in manifest["counts"]:
            continue
        ids = {
            key for key, r in expected.items() if any(m["origin"] == origin for m in r["members"])
        }
        groups[f"membership:{origin}"] = stats(ids) | {
            "original_selected_records": manifest["counts"][count_key],
            "excluded_records": sum(r["origin"] == origin for r in manifest["excluded"]),
            "scope_note": (
                "unique surviving source memberships, including native replacements; "
                "not original-scope accuracy"
            ),
        }
    result = {"selection_sha256": digest(manifest), "groups": groups}
    write_json(output / "summary.json", result)
    lines = [
        "# 固定套件运行进度",
        "",
        "来源成员组含原生替代，不是原始官方/Memory600 条件下的准确率。",
        "",
        "| 范围 | 完成/固定题数 | 正确 | 正确率 | 技术错误 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in groups.items():
        accuracy = "未完成" if value["accuracy"] is None else f"{value['accuracy']:.2%}"
        lines.append(
            f"| {key} | {value['completed']}/{value['expected']} | {value['correct']} "
            f"| {accuracy} | {value['technical_errors']} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def compare(left: Path, right: Path, output: Path) -> dict:
    a = json.loads((left / "run-manifest.json").read_text(encoding="utf-8"))
    b = json.loads((right / "run-manifest.json").read_text(encoding="utf-8"))
    for key in ("selection_sha256", "evaluation", "suite_code_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"runs are not comparable: {key}")
    left_groups, right_groups = summarize(left)["groups"], summarize(right)["groups"]
    rows = {}
    for key in left_groups:
        if left_groups[key]["expected"] != right_groups[key]["expected"]:
            raise ValueError("fixed denominators differ")
        difference = (
            100 * (right_groups[key]["accuracy"] - left_groups[key]["accuracy"])
            if left_groups[key]["accuracy"] is not None
            and right_groups[key]["accuracy"] is not None
            else None
        )
        rows[key] = {"left": left_groups[key], "right": right_groups[key], "change_pp": difference}
    result = {"selection_sha256": a["selection_sha256"], "groups": rows}
    write_json(output / "comparison.json", result)
    lines = [
        "# 固定套件版本对比",
        "",
        "混合来源原生评测，不是 AML 官方成绩。",
        "",
        "| 范围 | 左版正确/题数 | 右版正确/题数 | 变化百分点 |",
        "|---|---:|---:|---:|",
    ]
    for key, row in rows.items():
        delta = "未完成" if row["change_pp"] is None else f"{row['change_pp']:+.2f}"
        lstat, rstat = row["left"], row["right"]
        lines.append(
            f"| {key} | {lstat['correct']}/{lstat['expected']} "
            f"| {rstat['correct']}/{rstat['expected']} | {delta} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run(args, manifest: dict, jobs) -> int:
    from eval.datasets.prepare import ensure_dataset
    from eval.harness import api_config

    required = {
        "AML_BASE_URL": api_config.ANSWER_API_BASE,
        "AML_API_KEY": api_config.ANSWER_API_KEY,
        "AML_MODEL": api_config.ANSWER_MODEL,
        "JUDGE_API_BASE": api_config.JUDGE_API_BASE,
        "JUDGE_API_KEY": api_config.JUDGE_API_KEY,
        "JUDGE_MODEL": api_config.JUDGE_MODEL,
    }
    if missing := [key for key, value in required.items() if not value]:
        raise ValueError(f"missing model configuration: {missing}; use uv run --env-file .env")
    families = {r["dataset"] for r in manifest["questions"]}
    for family in sorted(families):
        ensure_dataset(
            "doc-pp" if family == "docpp" else family,
            args.data_root,
            offline=args.offline,
            purpose="judge",
        )
    output = Path(args.reports_dir) / "runs" / args.run_id
    runtime = {
        "selection_sha256": digest(manifest),
        "run_id": args.run_id,
        "service_revision": args.service_revision,
        "service_revision_note": "caller-declared",
        "service_base": args.base_url,
        "top_k": 100,
        "evaluation": evaluation_fingerprint(args),
        "suite_code_sha256": digest(
            [
                Path(__file__).read_text(),
                Path(__file__).parents[1].joinpath("datasets/selection_suite.py").read_text(),
            ]
        ),
        "config_fingerprint": config_fingerprint(args.profile, configs_dir=args.configs_dir),
    }
    freeze_run(output, [p for p, _ in jobs], execution=runtime)
    selection_path = output / "selection.json"
    if selection_path.exists() and load_selection(selection_path) != manifest:
        raise ValueError("saved selection changed; use a new run-id")
    write_json(selection_path, manifest)
    write_json(output / "run-manifest.json", runtime)
    snapshot = output / "config-snapshot"
    for name in runtime["config_fingerprint"]["snapshot_files"]:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((args.configs_dir / name).read_bytes())
    client = ServiceClient(args.base_url)
    try:
        for index, (plan, entries) in enumerate(jobs, 1):
            directory = output / plan.sample.user_id
            print(f"[{index}/{len(jobs)}] {plan.sample.dataset}: {len(entries)} tasks", flush=True)
            hits = execute_plan(plan, client=client, out_dir=directory)
            items = build_input_items(plan.sample, hits, date_mode=args.memory_date)
            pipeline = pipeline_for(args.data_root, plan.sample.dataset)
            results = run_judge(
                pipeline, items, directory, dataset=plan.sample.dataset, max_tokens=args.max_tokens
            )
            ids = {q.qid: row["id"] for q, row in zip(plan.sample.questions, entries, strict=True)}
            if {r.qid for r in results} != set(ids) or len(results) != len(ids):
                raise ValueError("judge output does not cover fixed questions uniquely")
            write_json(
                directory / "task-results.json",
                [{"id": ids[r.qid], "result": asdict(r)} for r in results],
            )
            summary = summarize(output)["groups"]["overall"]
            print(f"completed {summary['completed']}/{summary['expected']}", flush=True)
    finally:
        client.close()
        summarize(output)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path, default=benchmark_dir())
        p.add_argument("--capture-dir", type=Path, default=capture_dir())
        p.add_argument("--bundle", type=Path, required=name == "prepare")
        if name != "prepare":
            p.add_argument(
                "--verify-origins",
                action="store_true",
                help="check capture/Memory600 sources too; execution needs only public data",
            )
        p.add_argument(
            "--manifest",
            "--selection",
            type=Path,
            default=Path("eval/experiments/selections/memory600-official10-v1.tsv"),
        )
        p.add_argument(
            "--details",
            type=Path,
            help="local detailed JSON; generated by prepare, optional for origin checks",
        )
        if name == "prepare":
            p.add_argument("--seed", default="tianximem-20261014")
        if name != "run":
            p.add_argument(
                "--report",
                type=Path,
                default=Path("eval/reports/runs/selection-suite/selection.md"),
            )
        if name == "run":
            p.add_argument("--run-id", required=True)
            p.add_argument("--service-revision", required=True)
            p.add_argument("--base-url", default="http://127.0.0.1:8000")
            p.add_argument("--reports-dir", default="eval/reports")
            p.add_argument("--configs-dir", type=Path, default=Path("configs"))
            p.add_argument("--profile", default="local")
            p.add_argument("--max-tokens", type=int, default=1024)
            p.add_argument("--memory-date", choices=DATE_MODES, default="none")
            p.add_argument("--offline", action="store_true")
            p.add_argument("--annotate-mark", default="paren", choices=("paren",))
    sub.add_parser("summarize").add_argument("--run-dir", type=Path, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--left", type=Path, required=True)
    comparison.add_argument("--right", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            print(json.dumps(summarize(args.run_dir)["groups"]["overall"], ensure_ascii=False))
            return 0
        if args.command == "compare":
            compare(args.left, args.right, args.output)
            print(f"comparison → {args.output / 'comparison.md'}")
            return 0
        if args.command == "prepare":
            if args.manifest.exists():
                raise ValueError("selection exists; reuse it or choose a new manifest path")
            manifest = prepare(args.data_root, args.capture_dir, args.bundle, seed=args.seed)
            if args.manifest.suffix == ".tsv":
                write_ids(args.manifest, manifest)
                write_json(args.details or args.report.with_suffix(".details.json"), manifest)
            else:
                write_json(args.manifest, manifest)
            selection_report(manifest, args.report)
            print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
            return 0
        manifest = load_selection(args.manifest)
        if args.command == "run" and (
            args.max_tokens < 1
            or args.run_id in ("", ".", "..")
            or Path(args.run_id).name != args.run_id
        ):
            raise ValueError("invalid run-id or max-tokens")
        if args.verify_origins:
            if args.bundle is None:
                raise ValueError("--verify-origins requires --bundle")
            if "sources" not in manifest:
                if args.details is None:
                    raise ValueError(
                        "--verify-origins requires local --details JSON for a TSV selection"
                    )
                details = load_selection(args.details)
                if {(r["dataset"], r["native_qid"]) for r in details["questions"]} != {
                    (r["dataset"], r["native_qid"]) for r in manifest["questions"]
                }:
                    raise ValueError("local details and compact question IDs differ")
                manifest = details
            verify_sources(manifest, args.capture_dir, args.bundle)
        namespace = f"{getattr(args, 'run_id', 'offline-verify')}:{digest(manifest)}"
        jobs = build_plans(manifest, args.data_root, args.capture_dir, args.bundle, namespace)
        if args.command == "verify":
            selection_report(manifest, args.report, jobs)
            write_json(args.report.with_suffix(".json"), manifest)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "tasks": len(manifest["questions"]),
                        "users": len(jobs),
                        "api_calls": 0,
                    },
                    indent=2,
                )
            )
            return 0
        return run(args, manifest, jobs)
    except (OSError, ValueError) as error:
        print(f"✗ {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
