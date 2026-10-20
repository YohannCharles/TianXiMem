"""官方采集按数据集分类、固定清单重放评测与两版成绩对比。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
from eval.harness.official_judging import StreamingOfficialJudge
from eval.harness.plan_driver import compile_plan, execute_plan, freeze_run
from eval.harness.run_record import build_record, config_fingerprint, write_record
from eval.harness.scheduling import rerank_user_group, run_grouped_user_tasks, run_user_tasks
from eval.jsonl_io import write_line
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


def serial_prefix(args: argparse.Namespace) -> list[dict] | dict:
    """两臂共用已执行前缀，兼容单个串行前缀及多阶段并发记录。"""
    path = getattr(args, "serial_prefix", None)
    if not path:
        return []
    prefix = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prefix, (list, dict)) or (
        isinstance(prefix, dict)
        and (set(prefix) != {"stages"} or not isinstance(prefix["stages"], list))
    ):
        raise ValueError("serial-prefix 必须是用户边界列表或带 stages 的阶段列表")
    prior = {}
    for stage in prefix_stages(prefix):
        if (
            not isinstance(stage, dict)
            or not {"workers", "users"} <= set(stage) <= {"workers", "users", "rerank_routing"}
            or stage.get("rerank_routing", "historical") not in {"historical", "query-bucket"}
            or type(stage["workers"]) is not int or stage["workers"] < 1
            or not isinstance(stage["users"], list)
        ):
            raise ValueError("serial-prefix 含无效阶段")
        seen = set()
        for row in stage["users"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"source_user_id", "next_event"}
                or not isinstance(row["source_user_id"], str)
                or row["source_user_id"] in seen
                or type(row["next_event"]) is not int
                or row["next_event"] < prior.get(row["source_user_id"], 0)
            ):
                raise ValueError("serial-prefix 含重复用户或倒退的事件边界")
            seen.add(row["source_user_id"])
            prior[row["source_user_id"]] = row["next_event"]
    return prefix


def prefix_stages(prefix: list[dict] | dict) -> list[dict]:
    return prefix["stages"] if isinstance(prefix, dict) else [{"workers": 1, "users": prefix}]


def evaluation_fingerprint(args: argparse.Namespace) -> dict:
    """冻结答案/裁判口径；不保存 API 密钥。产品版本与配置允许在两臂间不同。"""
    root = Path(__file__).resolve().parents[2]
    code = sorted((root / "eval" / "harness").glob("*.py"))
    code += [Path(__file__), root / "eval" / "datasets" / "official_capture.py"]
    archive = benchmark_dir()
    upstream = sorted((archive / ".upstream").rglob("*.py"))
    scheduling = (
        f"serial Add/Search; saved-search judging overlap; workers={args.judge_workers}"
        if getattr(args, "judge_workers", 0)
        else "serial users; ordered checkpoints"
    )
    if getattr(args, "ingest_workers", 1) > 1:
        scheduling = (
            f"users={args.ingest_workers}; ordered per-user Add/Search; "
            f"saved-search judge workers={args.judge_workers}"
        )
    if getattr(args, "rerank_local_workers", 0):
        scheduling += (f"; fixed user pools local={args.rerank_local_workers}"
                       f",memory3={args.rerank_memory3_workers}; source-user sha256 modulo 3")
    if getattr(args, "judge_scheduling", "overlap") == "after-replay":
        scheduling += "; answer/judge begins after all Add/Search workers finish"
    if getattr(args, "serial_prefix", None):
        scheduling += "; non-prefix users overlap staged prefix replay"
    return {
        "answer_model": api_config.ANSWER_MODEL,
        "judge_model": api_config.JUDGE_MODEL,
        "answer_base": api_config.ANSWER_API_BASE,
        "judge_base": api_config.JUDGE_API_BASE,
        "max_tokens": args.max_tokens,
        "top_k": 100,
        "memory_date": args.memory_date,
        "annotate_mark": args.annotate_mark,
        "answer_context_policy": (
            json.loads(Path(os.environ["OFFICIAL_ANSWER_CONTEXT_POLICY"]).read_text(encoding="utf-8"))
            if os.environ.get("OFFICIAL_ANSWER_CONTEXT_POLICY") else None
        ),
        "llm_inflight_per_slot": int(os.environ.get("OFFICIAL_LLM_INFLIGHT_PER_SLOT", "0")),
        "scheduling": scheduling,
        "serial_prefix": serial_prefix(args),
        "code_sha256": {str(p.relative_to(root)): sha256(p) for p in code},
        "upstream_sha256": {str(p.relative_to(archive)): sha256(p) for p in upstream},
    }


def _append_jsonl(path: Path, row: dict) -> None:
    """逐条追加并 flush——长跑被打断时，跳过记录必须留在磁盘上。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        write_line(handle, row)


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())


def _failure_status(output: Path) -> str:
    """Audit events are cumulative; report current unresolved work separately."""
    def rows(name):
        path = output / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if path.is_file() else []
    skipped = rows("skipped.jsonl")
    failed_adds = {r["request_id"] for r in skipped}
    recovered = {r["request_id"] for r in rows("recovered-adds.jsonl")
                 if r.get("status") == "recovered"}
    errors = rows("errors.jsonl")
    unresolved = set()
    for error in errors:
        user = error.get("wire_user_id")
        qid = error.get("question_id")
        labels = rows(f"{user}/labels.jsonl") if user else []
        done = {str(r["id"]) for r in labels if not str(r.get("label", "")).endswith("_ERROR")}
        resolved = str(qid) in done if qid is not None else len(done) >= error.get("n_questions", 1)
        if not resolved:
            unresolved.add((user, qid))
    return (f"Add 失败历史 {len(skipped)} 次/{len(failed_adds)} 批"
            f"，已恢复 {len(failed_adds & recovered)} 批，待处理 {len(failed_adds - recovered)} 批"
            f"；用户/题目错误历史 {len(errors)} 次，待处理 {len(unresolved)} 项")


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
    replay_only = getattr(args, "replay_only", False)
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
    prefix = serial_prefix(args)
    by_source = {s.user_id: (i, s, w) for i, (s, w) in enumerate(zip(plans, compiled, strict=True))}
    for stage in prefix_stages(prefix):
        for row in stage["users"]:
            entry = by_source.get(row["source_user_id"])
            if entry is None or row["next_event"] > len(entry[2].events):
                raise ValueError("serial-prefix 的用户/事件边界不在固定题单中")
    if not replay_only:
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
    if not replay_only and (missing := [key for key, value in required.items() if not value]):
        raise ValueError(f"未配置答案/裁判端点：{missing}；使用 uv run --env-file .env")
    pipeline = pipeline_for(benchmark_dir(), "official-capture")
    if not replay_only and not pipeline.is_file():
        raise ValueError(f"评分 pipeline 缺失：{pipeline}")
    output = Path(args.reports_dir) / "runs" / args.run_id
    runtime = {
        "run_id": args.run_id,
        "service_revision": args.service_revision,
        "service_revision_note": "caller-declared; verify against the running service checkout",
        "service_base": args.base_url,
        "top_k": 100,
        "replay_only": replay_only,
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
    file_lock = threading.RLock()
    thread_state = threading.local()
    clients = []
    stop = threading.Event()
    active_users = {}
    replay_completed = 0
    ingest_workers = getattr(args, "ingest_workers", 1)
    group_workers = {}
    if getattr(args, "rerank_local_workers", 0):
        group_workers = {"local": args.rerank_local_workers, "memory3": args.rerank_memory3_workers}
        write_json(output / "rerank-user-groups.json", {
            "policy": "source-user sha256 modulo 3; local=0, memory3=1/2",
            "workers": group_workers,
            "users": {wire.sample.user_id: rerank_user_group(source.user_id)
                      for source, wire in zip(plans, compiled, strict=True)},
        })
    replay_phase = {"kind": "initializing"}
    prefix_user_ids = {
        row["source_user_id"]
        for stage in prefix_stages(prefix)
        for row in stage["users"]
    }

    def get_client():
        if not hasattr(thread_state, "client"):
            thread_state.client = ServiceClient(args.base_url)
            with file_lock:
                clients.append(thread_state.client)
        return thread_state.client

    def replay_progress():
        write_json(output / "ingest-progress.json", {
            "workers": ingest_workers,
            "phase": dict(replay_phase),
            "active_users": dict(active_users),
            "users_replayed_this_pass": replay_completed,
            "users_total": len(plans),
        })
    skipped_path = output / "skipped.jsonl"
    errors_path = output / "errors.jsonl"
    streaming = None
    sources_by_wire = {w.sample.user_id: s for s, w in zip(plans, compiled, strict=True)}

    def record_error(wire_user: str, error: Exception, qid: str | None = None) -> None:
        source = sources_by_wire[wire_user]
        with file_lock:
            _append_jsonl(
                errors_path,
                {
                    "user_id": source.user_id,
                    "wire_user_id": wire_user,
                    "n_questions": 1 if qid is not None else len(source.questions),
                    "question_id": qid,
                    "error": type(error).__name__,
                    "detail": (str(error) or "")[:12000],
                },
            )
        print(
            f"    ✗ {'题目 ' + qid if qid else '用户'}失败"
            f"（{type(error).__name__}: {str(error)[:180]}）"
            "⇒ 记入 errors.jsonl，保留检查点供续跑",
            flush=True,
        )

    if not replay_only and getattr(args, "judge_workers", 0):
        streaming = StreamingOfficialJudge(
            output,
            pipeline,
            workers=args.judge_workers,
            max_tokens=args.max_tokens,
            on_error=lambda user, qid, error: record_error(user, error, qid),
            deferred=getattr(args, "judge_scheduling", "overlap") == "after-replay",
        )

    def make_on_add_error(user_id: str, seq_by_request: dict[str, int]):
        """服务端/网络层失败 ⇒ **像线上那样跳过这一批并记账**；4xx 立刻抛。

        线上那一批 add 若注定 500（例如单块超过 embedding 上限），AML 重试 32 次
        重试后仍会丢失——本地继续重投只会让整轮永远停在同一个用户上。4xx（尤其 409
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
            with file_lock:
                wire_user = runtime["users"][user_id]
                payload_name = hashlib.sha256(event.request_id.encode()).hexdigest() + ".json"
                payload_file = Path("failed-adds") / wire_user / payload_name
                write_json(output / payload_file, {
                    "request_id": event.request_id,
                    "user_id": wire_user,
                    "session_id": event.session_id,
                    "messages": list(event.messages),
                })
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
                        "payload_file": str(payload_file),
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

    def replay_user(
        index, source, wire, *, boundary=None, prefix_index=None, prefix_routing=None
    ):
        nonlocal replay_completed
        if stop.is_set():
            raise InterruptedError("User replay cancelled")
        with file_lock:
            active_users[wire.sample.user_id] = {
                "source_index": index,
                "questions": len(source.questions),
                "stage": "serial-prefix" if boundary is not None else "replay",
                **({"prefix_index": prefix_index} if boundary is not None else {}),
                **({"prefix_routing": prefix_routing} if boundary is not None else {}),
                **({"rerank_backend": rerank_user_group(source.user_id)}
                   if group_workers and boundary is None else {}),
            }
            replay_progress()
        try:
            print(f"[{index}/{len(plans)}] {len(source.questions)} 题", flush=True)
            user_dir = output / wire.sample.user_id
            add_seq = {
                event.request_id: event.seq for event in source.events if event.kind == "add"
            }
            try:
                on_search = None
                if streaming is not None:
                    by_qid = {str(row["seq"]): row for row in source.questions}

                    def on_search(qid, rows):
                        item = build_official_items(
                            [by_qid[qid]],
                            {qid: rows},
                            date_mode=args.memory_date,
                            annotate_mark=args.annotate_mark,
                        )[0]
                        streaming.submit(wire.sample.user_id, item)
                        if args.stop_on_error:
                            streaming.raise_if_failed()

                hits = execute_plan(
                    wire,
                    client=get_client(),
                    out_dir=user_dir,
                    on_add_error=(
                        make_on_add_error(source.user_id, add_seq)
                        if args.on_add_error == "skip"
                        else None
                    ),
                    **({"on_search": on_search} if on_search is not None else {}),
                    stop_requested=stop.is_set,
                    stop_at_event=boundary,
                )
                if streaming is None and boundary is None and not replay_only:
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
                if stop.is_set() or args.stop_on_error or isinstance(error, FatalAddError):
                    stop.set()
                    raise
                record_error(wire.sample.user_id, error)
        finally:
            with file_lock:
                active_users.pop(wire.sample.user_id, None)
                if boundary is None:
                    replay_completed += 1
                replay_progress()

    print(
        f"{args.run_id}：固定 {len(questions)} 题 / {len(plans)} 个用户；"
        f"跨用户重放并发 {ingest_workers} → {output}", flush=True,
    )
    main_executor = None
    main_future = None

    def replay_items(items):
        """Run independent users and print progress as each user finishes."""
        replay_args = dict(
            user_id=lambda pair: pair[1].sample.user_id,
            run=lambda _, pair: replay_user(by_source[pair[0].user_id][0] + 1, *pair),
            cancel=stop.set,
        )
        if group_workers:
            iterator = run_grouped_user_tasks(
                items,
                **replay_args,
                group_of=lambda pair: rerank_user_group(pair[0].user_id),
                group_workers=group_workers,
            )
        else:
            iterator = run_user_tasks(items, **replay_args, workers=ingest_workers)
        for _index in iterator:
            if replay_only:
                print(f"Add/Search 用户进度 {replay_completed}/{len(plans)}", flush=True)
                continue
            try:
                summary = streaming.refresh_summary() if streaming else write_summary(output)
                print(
                    f"已判 {summary['completed']}/{summary['expected']}"
                    f"；{_failure_status(output)}",
                    flush=True,
                )
            except (ValueError, OSError) as error:
                print(f"    ⚠ 刷新 summary 失败：{error}", flush=True)

    try:
        all_pairs = list(zip(plans, compiled, strict=True))
        prefix_pairs = [
            pair for pair in all_pairs if pair[0].user_id in prefix_user_ids
        ]
        main_pairs = [
            pair for pair in all_pairs if pair[0].user_id not in prefix_user_ids
        ]

        # 前缀只约束前缀清单中的用户；其余用户立即进入主并发池。
        if main_pairs:
            replay_phase = {"kind": "main", "workers": ingest_workers}
            if group_workers:
                replay_phase.update(rerank_routing="user-groups", group_workers=group_workers)
            with file_lock:
                replay_progress()
            main_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="official-main")
            main_future = main_executor.submit(replay_items, main_pairs)
            print(
                f"前缀阶段并行主重放：{len(main_pairs)} 个非前缀用户已进入 "
                f"{ingest_workers} 路并发；前缀用户 {len(prefix_pairs)} 个按边界回放",
                flush=True,
            )

        # 每个前缀用户按冻结边界推进；无关用户已在独立的主池运行。
        for stage_index, stage in enumerate(prefix_stages(prefix)):
            replay_phase = {"kind": "prefix", "index": stage_index, "workers": stage["workers"]}
            if "rerank_routing" in stage:
                replay_phase["rerank_routing"] = stage["rerank_routing"]
            with file_lock:
                replay_progress()
            ready = []
            for row in stage["users"]:
                index, source, wire = by_source[row["source_user_id"]]
                cp = output / wire.sample.user_id / "checkpoint.json"
                cursor = json.loads(cp.read_text(encoding="utf-8"))["next_event"]
                if cursor < row["next_event"]:
                    ready.append((index, source, wire, row["next_event"]))
            if ready:
                print(f"复现前缀阶段：并发 {stage['workers']} / {len(ready)} 用户", flush=True)
            for _ in run_user_tasks(
                ready,
                user_id=lambda item: item[2].sample.user_id,
                run=lambda _, item, stage_index=stage_index, stage=stage: replay_user(
                    item[0] + 1, item[1], item[2],
                    boundary=item[3], prefix_index=stage_index,
                    prefix_routing=stage.get("rerank_routing"),
                ),
                workers=stage["workers"], cancel=stop.set,
            ):
                pass

        if main_future is not None:
            main_future.result()
            main_executor.shutdown(wait=True)
            main_executor = None
            main_future = None

        # 前缀用户完成冻结边界后，继续回放它们尚未处理的事件。
        replay_phase = {"kind": "main", "workers": ingest_workers}
        if group_workers:
            replay_phase.update(rerank_routing="user-groups", group_workers=group_workers)
        with file_lock:
            replay_progress()
        replay_items(prefix_pairs)
        if streaming is not None and streaming.deferred:
            replay_phase = {"kind": "answer-judge", "workers": 0}
            replay_progress()
            print("Add/Search 本轮已结束；开始处理已冻结的答案/判分队列……", flush=True)
            streaming.start()
    finally:
        stop.set()
        if main_executor is not None:
            main_executor.shutdown(wait=True, cancel_futures=True)
        if streaming is not None:
            print("等待后台逐题作答/判分队列结束……", flush=True)
            streaming.close(drain=sys.exc_info()[0] is None)
        for client in clients:
            client.close()
        if replay_only:
            search_count = 0
            completed_users = 0
            for wire in compiled:
                state = json.loads(
                    (output / wire.sample.user_id / "checkpoint.json").read_text(encoding="utf-8")
                )
                search_count += len(state["hits"])
                completed_users += state["next_event"] == len(wire.events)
            write_json(output / "retrieval-status.json", {
                "stage": "replay-only",
                "expected_searches": len(questions),
                "saved_searches": search_count,
                "completed_users": completed_users,
                "expected_users": len(compiled),
                "complete": search_count == len(questions) and completed_users == len(compiled),
                "answer_judge_started": False,
            })
            print(
                f"Search 已落盘 {search_count}/{len(questions)}；本轮不执行 answer/judge",
                flush=True,
            )
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
    if streaming and args.stop_on_error:
        streaming.raise_if_failed()
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
    execute.add_argument(
        "--replay-only", action="store_true",
        help="只按原时间线执行 Add/Search 并保存 checkpoint；不启动答案或裁判",
    )
    execute.add_argument("--ingest-workers", type=int, default=1, help="跨用户 Add/Search 并发数")
    execute.add_argument("--rerank-local-workers", type=int, default=0)
    execute.add_argument("--rerank-memory3-workers", type=int, default=0)
    execute.add_argument("--serial-prefix", default=None, help="两臂共用的串行前缀边界 JSON")
    execute.add_argument(
        "--judge-workers", type=int, default=0,
        help="正数启用 Search 落盘后逐题后台作答/判分；0 保留用户整批串行模式",
    )
    execute.add_argument("--judge-scheduling", choices=("overlap", "after-replay"), default="overlap")
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
            if ((args.rerank_local_workers or args.rerank_memory3_workers)
                and (args.rerank_local_workers < 1 or args.rerank_memory3_workers < 1
                     or args.rerank_local_workers + args.rerank_memory3_workers != args.ingest_workers)):
                raise ValueError("两组并发必须均为正且相加等于 ingest-workers")
            if (
                args.max_tokens < 1
                or args.ingest_workers < 1
                or args.judge_workers < 0
                or (args.judge_scheduling == "after-replay" and args.judge_workers == 0)
                or args.run_id in ("", ".", "..")
                or Path(args.run_id).name != args.run_id
            ):
                raise ValueError(
                    "max-tokens/ingest-workers 必须为正，judge-workers 非负，"
                    "run-id 必须是单层目录名"
                )
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
