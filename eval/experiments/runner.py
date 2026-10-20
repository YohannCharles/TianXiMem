"""一轮评测的**机制层**：加载 → 投喂 → 检索 → 裁判（`run_round`），外加两个"跑之前就该
失败"的检查（`judge_preconditions` / `missing_pipeline`）。

本模块负责机制层：`run.py` 保留命令行与薄 façade（`main` / `build_parser` /
退出码与不可达提示），run record 的组装与打印在 [`record.py`](./record.py)。
`eval.experiments.run` 的 import 与调用保持兼容（`run.py` 把这边的名字**重新导出**）。

⚠ 测试/消费方若 `monkeypatch` 本模块的名字，**要打在实现所在的这里**
（`eval.experiments.runner`）——打在 `run.py` 的 façade 上不会影响这里的实现。

⚠ **"打 HTTP、不 import `src/`"这条边界在 [`../CLAUDE.md`](../CLAUDE.md)，本文件不重复。**
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

from eval.datasets import (
    Sample,
    load_beam,
    load_clbench,
    load_corporatebench,
    load_feverous,
    load_halumem,
    load_hybridqa,
    load_locomo,
    load_longmemeval,
    load_medmemorybench,
    load_memtrapbench,
    load_mquake,
    load_musique,
    load_personamem,
    load_tempreason,
)
from eval.datasets.aml import INPUT_CONTRACT, load_plans
from eval.harness import (
    DEFAULT_ADD_SHAPE,
    ServiceClient,
    build_input_items,
    pipeline_for,
    run_judge,
)
from eval.harness.api_config import ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL, ENDPOINTS
from eval.harness.plan_driver import compile_plan, execute_plan, freeze_run
from eval.harness.scheduling import run_pipeline, worker_counts


class EmptyDatasetError(RuntimeError):
    """`--dataset` 加载出**零个样本**——几乎总是"归档没取回"或 `bench_dir` 指错了。

    ⚠ **它得是自己的类型，不能借用裸 `ValueError`**：`run_round` 里另有一个
    `ValueError`（"未知数据集"），那是**用法错误**、提示完全不同；两者混在一起
    会让"数据集名打错了"收到一句"请先 `make fetch-data`"。
    """


#: AML 固定传 `top_k=100`。**这里也用它**——预检要验的就是线上那个值。
DEFAULT_TOP_K: Final[int] = 100

#: 独立本地评测入口；各自的输入与评分范围见加载器和 pipeline。
DATASETS: Final[tuple[str, ...]] = (
    "beam",
    "clbench",
    "corporatebench",
    "feverous",
    "halumem",
    "hybridqa",
    "locomo-refined",
    "longmemeval-s",
    "medmemorybench",
    "memtrapbench",
    "mquake-remastered",
    "musique",
    "personamem-v2",
    "tempreason",
)


def _load(
    dataset: str, bench_dir: Path, limit: int | None, *, spread: bool = False
) -> list[Sample]:
    """加载 + `limit`。

    ⚠ **各数据集的 `limit` 不是同一个机制**，别当成对称的：
    LongMemEval 的 `limit` 在**加载器里**做（那份 277 MB，全量解析要几十秒、几 GB 内存），
    而 LoCoMo 的加载器没有这个参数（它整份才 10 段对话，切片在加载后做）。

    ⚠ **`spread` 对 LongMemEval 与 CL-Bench 有意义**：两者都**按类型分块**
    （前者 `question_type`、后者 `context_category`），所以"前 N 题"会落在单一类型上
    ⇒ 部分跑要用**分层抽样**才代表整个数据集（实现在
    `eval/datasets/sampling.py` 的 `stratified_sample`）。LoCoMo 是"10 段对话 ×
    若干题"的结构，切片天然跨段，不需要它——**传了也只当没看见**（不静默改语义）。
    """
    if dataset == "locomo-refined":
        samples = load_locomo(bench_dir)
        return samples[:limit] if limit is not None else samples
    if dataset == "longmemeval-s":
        return load_longmemeval(bench_dir, limit=limit, spread=spread)
    if dataset == "clbench":
        return load_clbench(bench_dir, limit=limit, spread=spread)
    if dataset == "beam":
        return load_beam(bench_dir, limit=limit, spread=spread)
    if dataset == "halumem":
        return load_halumem(bench_dir, limit=limit, spread=spread)
    if dataset == "musique":
        return load_musique(bench_dir, limit=limit, spread=spread)
    if dataset == "hybridqa":
        return load_hybridqa(bench_dir, limit=limit, spread=spread)
    if dataset == "feverous":
        return load_feverous(bench_dir, limit=limit, spread=spread)
    # ↓ 这三份**没有官方 pipeline**（`extra_pipeline.py` 是我们自写的）⇒ 分数只在仓内比。
    if dataset == "tempreason":
        return load_tempreason(bench_dir, limit=limit, spread=spread)
    if dataset == "personamem-v2":
        return load_personamem(bench_dir, limit=limit, spread=spread)
    if dataset == "mquake-remastered":
        return load_mquake(bench_dir, limit=limit, spread=spread)
    if dataset == "medmemorybench":
        return load_medmemorybench(bench_dir, limit=limit, spread=spread)
    if dataset == "memtrapbench":
        return load_memtrapbench(bench_dir, limit=limit, spread=spread)
    if dataset == "corporatebench":
        return load_corporatebench(bench_dir, limit=limit, spread=spread)
    raise ValueError(f"未知数据集 {dataset!r}——只有 {' / '.join(DATASETS)} 有加载器")


def judge_preconditions() -> list[str]:
    """裁判链路的**前置条件**：归档 pipeline 要的那几个名字现在必须能读到值。

    ⚠ **这是"跑之前就该失败"的那一类。** `run_judge` 起的是 **subprocess**，而 subprocess
    只继承 `os.environ`——仓库里的 `.env` 是 `common/config/` **自己读**的，
    **不会**进环境变量。⇒ 少了这一步，`Add` 与 `Search` 会**全部正常跑完**
    （服务自己读 `.env`），然后在**裁判那一步**才炸：那时整轮的 embedding 与检索时间
    已经付出去了，而你拿到的是一个没有分数的 run。

    ⇒ 入口是 `make eval`（它带 `uv run --env-file`）；手敲 `uv run python -m ...` 时要自己带。
    """
    missing = [
        name
        for name, value in (
            ("AML_BASE_URL", ANSWER_API_BASE),
            ("AML_API_KEY", ANSWER_API_KEY),
            ("AML_MODEL", ANSWER_MODEL),
        )
        if not value
    ]
    return missing


def missing_pipeline(dataset: str, bench_dir: Path) -> Path | None:
    """裁判脚本在不在？**在跑之前查**。

    ⚠ 与 `judge_preconditions` 是**同一类**（都是"跑之前就该失败"），但它更贵：
    `run_judge` 起 subprocess 时**不会**因为脚本不存在而立刻失败——整轮的 `Add` 与
    `Search` 会**全部正常跑完**（embedding 的钱已经付了），然后**第一个样本**的裁判
    那一步才以 `RuntimeError` 炸（子进程重试 3 次后抛，stderr 里只有一句 python 的
    `can't open file`）。⇒ 检查放在加载之前，一行都不用跑。

    只可能缺**归档里那几份**（locomo / lme / clbench / beam）——另两类的路径在本仓内。
    """
    try:
        path = pipeline_for(bench_dir, dataset)
    except ValueError:
        return None  # 未知数据集：`--dataset` 的更早一处已经拦了，别在这里报重复的错
    return None if path.exists() else path


def run_round(
    *,
    dataset: str,
    base_url: str,
    bench_dir: Path,
    out_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None,
    skip_ingest: bool = False,
    date_mode: str = "none",
    annotate_mark: str = "paren",
    spread: bool = False,
    max_questions: int | None = None,
    fallback_base_url: str | None = None,
    add_shape: str = DEFAULT_ADD_SHAPE,
    client: ServiceClient | None = None,
    input_contract: str = "native",
    aml_time_style: str = "synthetic",
    aml_pool: Path | None = None,
    execution_fingerprint: dict | None = None,
    judge_workers: int = 0,
    ingest_workers: int = 0,
) -> tuple[list[Sample], list]:
    """跑一轮的**机制部分**：加载 → 投喂 → 检索 → 裁判。返回 `(samples, results)`。

    与 `main` 分开，是为了让测试能塞一个 `httpx.MockTransport` 的客户端进来
    （[`../../tests/test_experiments.py`](../../tests/test_experiments.py) 就是这么做的），
    而 CLI 参数解析不参与那条路径。

    Independent users run through bounded preparation and judgment pools. Each
    user's Add batches / AML events stay ordered; Search sees that user's complete
    native history or the original AML checkpoint. Zero means automatic workers.
    """
    if ingest_workers < 0 or judge_workers < 0:
        raise ValueError("ingest_workers and judge_workers must be nonnegative")
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}——只有 {' / '.join(DATASETS)} 有加载器")
    if input_contract == INPUT_CONTRACT:
        if skip_ingest:
            raise ValueError(
                "AML plans resume from event checkpoints; "
                "--skip-ingest cannot preserve historical Search state"
            )
        if top_k != DEFAULT_TOP_K:
            raise ValueError("AML input contract requires top_k=100")
        plans = [
            compile_plan(p, namespace=out_dir.name)
            for p in load_plans(
                bench_dir,
                dataset,
                limit=limit,
                spread=spread,
                max_questions=max_questions,
                add_shape=add_shape,
                time_style=aml_time_style,
                pool=aml_pool,
            )
        ]
        if not plans:
            raise EmptyDatasetError(f"{dataset}: no AML input plans")
        from eval.harness.api_config import JUDGE_API_BASE, JUDGE_MODEL

        policy_files = sorted(Path(__file__).parents[1].joinpath("harness").glob("*.py"))
        pipeline = pipeline_for(bench_dir, dataset)
        freeze_run(
            out_dir,
            plans,
            execution={
                "base_url": base_url,
                "top_k": top_k,
                "date_mode": date_mode,
                "annotate_mark": annotate_mark,
                "fallback_base_url": fallback_base_url,
                "answer_model": ANSWER_MODEL,
                "answer_base": ANSWER_API_BASE,
                "judge_model": JUDGE_MODEL,
                "judge_base": JUDGE_API_BASE,
                "harness_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in policy_files
                },
                "pipeline_sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                "scheduling": dict(
                    zip(
                        ("ingest_workers", "judge_workers"),
                        worker_counts(
                            ingest_workers,
                            judge_workers,
                            users=len(plans),
                            endpoints=len(ENDPOINTS),
                        ),
                        strict=True,
                    )
                ),
                "declarations": execution_fingerprint or {},
            },
        )
        owns_client = client is None
        client = client or ServiceClient(
            base_url, fallback=ServiceClient(fallback_base_url) if fallback_base_url else None
        )
        try:
            ingest_workers, judge_workers = worker_counts(
                ingest_workers, judge_workers, users=len(plans), endpoints=len(ENDPOINTS)
            )
            print(f"  AML 用户并行 {ingest_workers} 路；判分并行 {judge_workers} 路", flush=True)

            def prepare(index, plan):
                sample = plan.sample
                print(
                    f"  [{index + 1}/{len(plans)}] {sample.user_id}: AML 有序 Add/Search",
                    flush=True,
                )
                started = time.monotonic()
                hits = execute_plan(
                    plan, client=client, out_dir=out_dir / sample.user_id, top_k=top_k
                )
                elapsed = time.monotonic() - started
                items = build_input_items(
                    sample, hits, date_mode=date_mode, annotate_mark=annotate_mark
                )
                for item in items:
                    item["input_contract"] = input_contract
                    if "answer_contract" in plan.metadata:
                        item["answer_contract"] = plan.metadata["answer_contract"]
                return items, {
                    "event_seconds": elapsed,
                    "ingest_workers": ingest_workers,
                    "judge_workers": judge_workers,
                    "endpoint_index": index % len(ENDPOINTS),
                }

            def judge(index, plan, prepared):
                items, timings = prepared
                started = time.monotonic()
                rows = run_judge(
                    pipeline,
                    items,
                    out_dir / plan.sample.user_id,
                    dataset=dataset,
                    endpoint_index=index % len(ENDPOINTS),
                )
                timings["judge_seconds"] = time.monotonic() - started
                _write_timings(out_dir / plan.sample.user_id, timings)
                print(f"  ✓ {plan.sample.user_id}: {len(items)} 题已判", flush=True)
                return rows

            by_index = dict(
                run_pipeline(
                    plans,
                    user_id=lambda p: p.sample.user_id,
                    prepare=prepare,
                    judge=judge,
                    ingest_workers=ingest_workers,
                    judge_workers=judge_workers,
                )
            )
            return [p.sample for p in plans], [r for i in range(len(plans)) for r in by_index[i]]
        finally:
            if owns_client:
                client.close()
    if input_contract != "native" or aml_time_style != "synthetic" or aml_pool is not None:
        raise ValueError("Unknown input contract or AML-only options used with native input")
    samples = _load(dataset, bench_dir, limit, spread=spread)
    if not samples:
        raise EmptyDatasetError(f"{dataset}：一个 sample 都没加载到——bench_dir={bench_dir} 对吗？")

    owns_client = client is None
    client = client or ServiceClient(
        base_url,
        # 超限兜底（只为 B1）：见 `driver.ServiceClient.__init__` 的注释。
        fallback=ServiceClient(fallback_base_url) if fallback_base_url else None,
        add_shape=add_shape,
    )
    try:
        pipeline = pipeline_for(bench_dir, dataset)
        ingest_workers, judge_workers = worker_counts(
            ingest_workers, judge_workers, users=len(samples), endpoints=len(ENDPOINTS)
        )
        print(f"  投喂/检索并行 {ingest_workers} 路；判分并行 {judge_workers} 路", flush=True)

        def prepare(index, sample):
            started = time.monotonic()
            batches_count = 0 if skip_ingest else client.ingest(sample)
            ingested = time.monotonic()
            questions = (
                sample.questions if max_questions is None else sample.questions[:max_questions]
            )
            print(
                f"  [{index + 1}/{len(samples)}] {sample.user_id}：语料已投喂，"
                f"开始检索 {len(questions)} 题",
                flush=True,
            )
            hits = {
                q.qid: client.search(user_id=sample.user_id, query=q.question, top_k=top_k)
                for q in questions
            }
            searched = time.monotonic()
            scoped = (
                sample if questions is sample.questions else replace(sample, questions=questions)
            )
            items = build_input_items(
                scoped, hits, date_mode=date_mode, annotate_mark=annotate_mark
            )
            return items, {
                "ingest_seconds": ingested - started,
                "search_seconds": searched - ingested,
                "batches": batches_count,
                "ingest_workers": ingest_workers,
                "judge_workers": judge_workers,
                "endpoint_index": index % len(ENDPOINTS),
            }

        def judge(index, sample, prepared):
            items, timings = prepared
            started = time.monotonic()
            rows = run_judge(
                pipeline,
                items,
                out_dir / sample.user_id,
                dataset=dataset,
                endpoint_index=index % len(ENDPOINTS),
            )
            timings["judge_seconds"] = time.monotonic() - started
            _write_timings(out_dir / sample.user_id, timings)
            print(f"  ✓ {sample.user_id}：{len(items)} 题已判", flush=True)
            return rows

        by_index = dict(
            run_pipeline(
                samples,
                user_id=lambda s: s.user_id,
                prepare=prepare,
                judge=judge,
                ingest_workers=ingest_workers,
                judge_workers=judge_workers,
            )
        )
        return samples, [row for index in range(len(samples)) for row in by_index[index]]
    finally:
        if owns_client:
            client.close()


def _write_timings(user_dir: Path, timings: dict) -> None:
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "timings.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(timings, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
