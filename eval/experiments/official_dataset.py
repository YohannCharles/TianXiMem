"""官方采集按数据集分类、固定清单重放评测与两版成绩对比。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import httpx
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
from eval.harness.judge_results import results_from_labels
from eval.harness.plan_driver import compile_plan, execute_plan, freeze_run
from eval.harness.run_record import build_record, config_fingerprint, write_record
from eval.reports.official_comparison import compare_runs, coverage_report, write_summary


class FatalAddError(ValueError):
    """`Add` 返回 4xx——我方 bug（尤其 409"同 request_id 不同 payload"）。

    ⛔ 不能按"线上那一批也丢了"跳过：4xx 会**对每个用户**重复出现，跳过只会把
    系统性问题藏进 `skipped.jsonl`，最后得到一份"跑完了但什么都不对"的成绩。
    """


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


def _append_jsonl(path: Path, row: dict) -> None:
    """逐条追加并 flush——长跑被打断时，"已经跳过了哪一批"必须留在磁盘上。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())


def _results_from_disk(output: Path, compiled: list[InputPlan]) -> list:
    """从**所有**用户的 `labels.jsonl` 读回结果——续跑时 `run_judge` 只返回本轮处理过的题，
    只拿本轮返回值会让 run record 的 `overall` 偷偷缩小分母。"""
    results: list = []
    for wire in compiled:
        results.extend(results_from_labels(output / wire.sample.user_id))
    return results


def _runtime_drift(saved: dict, current: dict) -> tuple[list[str], list[str]]:
    """续跑时哪些变化**必须换 run-id**、哪些只记账继续。

    `freeze_run` 已经把同一套口径用在通用 runner 上（`execution-drift.jsonl`）：
    **wire 身份 / 模型 / 端点 / 预算 / 服务版本**变了 ⇒ 分数不可比，必须换 run-id；
    **代码哈希、配置目录、base URL** 变了 ⇒ 大声记一笔、继续跑——否则改一行注释
    就把一条跑了 30 小时的采集重放锁死，只能从头再来。
    """
    fatal: list[str] = []
    if saved.get("data_identity") != current.get("data_identity"):
        fatal.append("data_identity")
    if saved.get("users") != current.get("users"):
        fatal.append("users")
    if saved.get("top_k") != current.get("top_k"):
        fatal.append("top_k")
    if saved.get("service_revision") != current.get("service_revision"):
        fatal.append("service_revision")
    saved_eval = saved.get("evaluation") or {}
    current_eval = current.get("evaluation") or {}
    for key in (
        "answer_model",
        "judge_model",
        "answer_base",
        "judge_base",
        "max_tokens",
        "memory_date",
        "annotate_mark",
    ):
        if saved_eval.get(key) != current_eval.get(key):
            fatal.append(f"evaluation.{key}")

    drifted: list[str] = []
    for key in sorted(set(saved) | set(current)):
        if key in {"data_identity", "users", "top_k", "service_revision"}:
            continue
        if key == "evaluation":
            for sub in sorted(set(saved_eval) | set(current_eval)):
                if sub not in {
                    "answer_model",
                    "judge_model",
                    "answer_base",
                    "judge_base",
                    "max_tokens",
                    "memory_date",
                    "annotate_mark",
                } and saved_eval.get(sub) != current_eval.get(sub):
                    drifted.append(f"evaluation.{sub}")
        elif saved.get(key) != current.get(key):
            drifted.append(key)
    return fatal, drifted


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
        saved_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        snapshot = load_manifest(output / "selection-manifest.json")
        if data_identity(snapshot) != identity:
            raise ValueError("已保存的题目清单变化；请使用新的 run-id")
        fatal, drifted = _runtime_drift(saved_runtime, runtime)
        if fatal:
            raise ValueError(
                "运行的版本/输入/模型变化（" + ", ".join(fatal) + "）；请使用新的 run-id"
            )
        if drifted:
            print(
                f"  ⚠ 运行执行指纹漂移（{', '.join(drifted)}）；记入 execution-drift.jsonl 后继续",
                flush=True,
            )
            _append_jsonl(
                output / "execution-drift.jsonl",
                {"drifted_keys": drifted, "prior_execution": saved_runtime},
            )
            write_json(runtime_path, runtime)
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
    skipped_path = output / "skipped.jsonl"
    errors_path = output / "errors.jsonl"

    def make_on_add_error(user_id: str, seq_by_request: dict[str, int]):
        """服务端/网络层失败 ⇒ **像线上那样跳过这一批并记账**；4xx 立刻抛。

        线上那一批 add 若注定 500（例如单块超过 embedding 上限），AML 重试 32 次
        之后同样会丢——本地继续重投只会让整轮永远停在同一个用户上。4xx（尤其 409
        "同 request_id 不同 payload"）是我方 bug，必须响亮失败，不能吞。

        ⚠ `AddEvent` 本身**没有 `seq`**（那是采集 timeline 的字段，编译后只剩
        `request_id`）⇒ 用编译前的 `source.events` 建映射，记账才能对回原始 seq。
        """

        def on_add_error(event: AddEvent, error: Exception) -> None:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500:
                raise FatalAddError(
                    f"/add 返回 {error.response.status_code}（4xx）——这是我方 bug，"
                    "不能当作'线上也丢了'跳过"
                ) from error
            _append_jsonl(
                skipped_path,
                {
                    "kind": "add",
                    "user_id": user_id,
                    "seq": seq_by_request.get(event.request_id),
                    "request_id": event.request_id,
                    "session_id": event.session_id,
                    "n_messages": len(event.messages),
                    "chars": sum(len(m.get("content") or "") for m in event.messages),
                    "error": type(error).__name__,
                    "detail": (str(error) or "")[:300],
                },
            )
            print(
                f"    ⚠ /add 失败（{type(error).__name__}）⇒ **跳过这一批**"
                f"（request_id={event.request_id}）并继续——见 skipped.jsonl",
                flush=True,
            )

        return on_add_error

    print(f"{args.run_id}：固定 {len(questions)} 题 / {len(plans)} 个用户 → {output}", flush=True)
    try:
        for index, (source, wire) in enumerate(zip(plans, compiled, strict=True), 1):
            print(f"[{index}/{len(plans)}] {len(source.questions)} 题", flush=True)
            user_dir = output / wire.sample.user_id
            add_seq = {
                event.request_id: event.seq for event in source.events if event.kind == "add"
            }
            try:
                hits = execute_plan(
                    wire,
                    client=client,
                    out_dir=user_dir,
                    on_add_error=(
                        make_on_add_error(source.user_id, add_seq)
                        if args.on_add_error == "skip"
                        else None
                    ),
                )
                items = build_official_items(
                    source.questions,
                    hits,
                    date_mode=args.memory_date,
                    annotate_mark=args.annotate_mark,
                )
                run_judge(
                    pipeline,
                    items,
                    user_dir,
                    dataset="official-capture",
                    max_tokens=args.max_tokens,
                )
            except Exception as error:
                # ⚠ 长跑不能因为一个用户的一步失败就打死整轮。该用户的检查点仍然
                #   停在失败处 ⇒ 同一个 run-id 重跑会**从这里续**；其余用户先跑完。
                # ⛔ 4xx（`FatalAddError`）是系统性我方 bug，必须整轮停。
                if args.stop_on_error or isinstance(error, FatalAddError):
                    raise
                _append_jsonl(
                    errors_path,
                    {
                        "user_id": source.user_id,
                        "wire_user_id": wire.sample.user_id,
                        "n_questions": len(source.questions),
                        "error": type(error).__name__,
                        "detail": (str(error) or "")[:1000],
                    },
                )
                print(
                    f"    ✗ 用户失败（{type(error).__name__}: {str(error)[:180]}）"
                    "⇒ 记入 errors.jsonl，继续下一个用户",
                    flush=True,
                )
            try:
                summary = write_summary(output)
                print(
                    f"已判 {summary['completed']}/{summary['expected']}"
                    f"；跳过 add 批 {_count_lines(skipped_path)}"
                    f"；用户错误 {_count_lines(errors_path)}",
                    flush=True,
                )
            except (ValueError, OSError) as error:
                # 汇总失败不该反过来打死正在跑的 Add/Search——留下一行，继续。
                print(f"    ⚠ 刷新 summary 失败：{error}", flush=True)
    finally:
        client.close()
        try:
            write_summary(output)
        except (ValueError, OSError) as error:
            print(f"  ⚠ 最终 summary 失败：{error}", flush=True)
        # 续跑时 `run_judge` 只返回本轮处理过的题 ⇒ 从磁盘读回全部，分母才是固定的。
        results = _results_from_disk(output, compiled)
        skipped = _count_lines(skipped_path)
        errors = _count_lines(errors_path)
        notes = (
            "固定采集 seq；原文有序重放；Add/Search 失败按策略跳过或阻断，逐事件检查点续跑。"
            + (f" 跳过 add 批 {skipped} 条（见 skipped.jsonl）。" if skipped else "")
            + (
                f" 用户级错误 {errors} 条（见 errors.jsonl，重跑同一 run-id 可续）。"
                if errors
                else ""
            )
        )
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
            notes=notes,
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
    execute.add_argument(
        "--on-add-error",
        choices=("skip", "abort"),
        default="skip",
        help=(
            "Add 的服务端/网络层失败怎么处理：skip = 记入 skipped.jsonl 并继续"
            "（线上那一批同样会丢）；abort = 阻断该用户。4xx 永远不跳过"
        ),
    )
    execute.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一用户失败就退出（调试用）；缺省记录到 errors.jsonl 并继续下一个用户",
    )
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
