"""通用 runner —— **给定数据集与配置，跑一轮，落一份 run record**。

> **本文件是"把零件串起来"的那一层**，它不实现其中任何一环：
>
> ```text
> eval.datasets          加载 + 归一化（schema 落差预处理）
>   → driver.ServiceClient   切批喂 Add（走 HTTP）、逐题 Search
>   → judge.build_input_items / run_judge   注入记忆 + 起归档 pipeline 裁判
>   → run_record.build_record / write_record    配置指纹 + 数据指纹 + 分数
> ```
>
> **一条边界**：打 HTTP、**不 import `src/tianximem`**（[`../CLAUDE.md`](../CLAUDE.md)）。
> 所以本文件只依赖 `eval/` 与标准库 + `httpx`。

## 为什么需要一个通用 runner

每一条命名对照都要"跑一轮并落一份可比记录"，
而**它们的差别只在配置与开关上**——把这一轮跑通的地方只该有一处。
各 arm 的脚本（[`t1_timestamp.py`](./t1_timestamp.py)、
[`t2_cross_session.py`](./t2_cross_session.py)）
做的是"声明自己改了哪几个开关 + 指向哪份配置快照"，**然后调本模块**。

## 用法

```bash
# 打已在跑的服务（`make serve`），跑全量 LoCoMo-Refined
uv run python eval/experiments/run.py --dataset locomo-refined --embedder Qwen/Qwen3-Embedding-4B

# 冒烟：只跑 3 个 sample，且不重投已有语料
uv run python eval/experiments/run.py --dataset longmemeval-s --limit 3 --skip-ingest
```

⚠ **`--limit` 会写进数据指纹的 `note`**——"跑了一小撮"与"跑完了"的数字
**看起来一样**，不写下来就会有人拿它们比。
⚠ **重跑是安全的**：`request_id` 由 `(user_id, session_id, 批序号)` 确定性派生，
`applied_batches` 的批次级幂等守卫会命中。但**换过渲染模板/配对规则后不行**
——`applied_batches` 会**直接放行而不改写**（同一 `request_id` 已应用过），
于是你测的是**已写入的语料 + 新代码**，**静默**。⇒ 改了模板/配对规则就必须换干净的库与集合。

## 本文件的形状

本模块负责**命令行**那一层：`main` / `build_parser` / 退出码与不可达提示；
"把零件串起来"的代码（`run_round` 与它的加载、两个前置检查）在
[`runner.py`](./runner.py)，run record 的组装与打印在 [`record.py`](./record.py)。
整条评测链从本文件的 `main` 进入；下面导入块把实现模块的公开名字重新导出。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

import httpx
from eval.datasets import benchmark_dir, shape_note
from eval.datasets.aml import INPUT_CONTRACT, required_datasets, validate_options
from eval.datasets.prepare import PreparationError, ensure_dataset

# ⚠ **绝对 import**：本文件是**当脚本跑的**（`python eval/experiments/run.py`），
# 相对 import 在那种调用方式下会抛 `ImportError: no attempt relative import...`。
from eval.experiments.recipes import recipe_for

# ── façade：搬去 runner.py / record.py 的名字在这层**重新导出** ─────────────
# 保持 `from eval.experiments.run import X` 兼容（`arms.py`、`tests/`、`tools/` 都在用）。
# ⚠ 重新导出的是**名字的快照**：`monkeypatch.setattr(run, "X", ...)` 打不到移出后的
#   真实现——要补丁就去补 `runner.X` / `record.X`（同 `eval/harness/judge.py` 的写法）。
from eval.experiments.record import (  # noqa: F401
    _assemble_record,
    _count_questions,
    _default_llm,
    _parse_switches,
    _print_result,
    _read_service_metrics,
    _warn_about_rerank,
    derive_run_id,
    models_fingerprint,
    truncation_note,
)
from eval.experiments.runner import (  # noqa: F401
    DATASETS,
    DEFAULT_TOP_K,
    EmptyDatasetError,
    _load,
    _write_timings,
    judge_preconditions,
    missing_pipeline,
    run_round,
)
from eval.harness import ADD_SHAPES, DEFAULT_ADD_SHAPE, config_fingerprint, write_record
from eval.harness.judge import DATE_MODES, MARKS

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_PRECONDITION_FAILED: Final[int] = 2


#: 服务地址：与 `eval/harness/__init__.py` 的示例、`make serve` 的缺省端口一致。
DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000"


def _unreachable_hint(base_url: str, error: Exception) -> str:
    """服务不可用时的提示。

    ⚠ **"连不上"在有的机器上不表现为拒绝连接**：WSL + Docker Desktop 的端口转发会把
    任意 localhost 端口接成 **502 Bad Gateway**（转发在、容器不在——`Makefile` 的
    `qdrant-up` 那条注释里记着同一个现象）。所以这里把 5xx 与传输层错误**一起**当成
    "服务没就绪"，而不是只认 `ConnectError`。
    """
    return f"服务 {base_url} 不可用（{error}）——先 `make serve`，或用 --base-url 指向已在跑的服务"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="跑一轮代理评测并落一份 run record（§13）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--input-contract",
        choices=("native", INPUT_CONTRACT),
        default="native",
        help="native 保留当前加载/包装；aml-v1 使用版本化的 AML 近似事件计划",
    )
    parser.add_argument(
        "--aml-time-style",
        choices=("synthetic", "inline"),
        default="synthetic",
        help="AML 对话时间形态；inline 仅适用于 HaluMem",
    )
    parser.add_argument(
        "--aml-pool",
        type=Path,
        default=None,
        help="FEVEROUS 的声明页面池 JSON（titles 列表）；缺省从本地捕获 Add 恢复标题",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"服务地址（缺省 {DEFAULT_BASE_URL}）"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="§2.2 固定的 100")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 个 sample（**会写进数据指纹的 note**——截断过的数字不可与全量比）",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "每个 sample **最多判几道题**（冒烟用）。⚠ 语料照常整份 Add——只裁题目不裁语料，"
            "否则检索没有素材，答案与分数全是假的。会写进数据指纹的 note"
        ),
    )
    parser.add_argument("--skip-ingest", action="store_true", help="跳过 Add，复用已有语料")
    parser.add_argument("--offline", action="store_true", help="禁止下载数据，缺失或校验不符就退出")
    parser.add_argument(
        "--frozen",
        action="store_true",
        help=(
            "用**冻结口径**（`eval/experiments/recipes.py`）取 `--limit` / `--spread`——"
            '"这个数据集跑多少题"的唯一声明处。**与显式的 `--limit` / `--spread` 互斥**。'
            "要判断改动值不值，两次 run 必须同口径 ⇒ 用这个，别手抄数字"
        ),
    )
    parser.add_argument(
        "--spread",
        action="store_true",
        help=(
            "**分层抽样**（LongMemEval 与 CL-Bench 用得上）：两者都按类型分块，"
            "所以 `--limit N` 不加本标志时取到的是**单一类型**——部分跑要用它跨类取题"
        ),
    )
    parser.add_argument(
        "--memory-date",
        choices=DATE_MODES,
        default="none",
        help=(
            "注入时怎么带日期。**逐档的口径与实测结论写在 `harness/judge.py` 的 `DATE_MODES` 上**"
            "（本处不复制一份，免得漂移）；默认 none，`annotate` 是目前最好的一档"
        ),
    )
    parser.add_argument(
        "--annotate-mark",
        choices=MARKS,
        default="paren",
        help="`--memory-date annotate` 的记号：paren（`last Tues (…2023)`）/ tag（`[= …]`）",
    )
    parser.add_argument(
        "--add-shape",
        choices=ADD_SHAPES,
        default=DEFAULT_ADD_SHAPE,
        help=(
            "`Add` 正文按哪种形态渲染。**缺省 official = 线上那个**（逐数据集加 `<标签>: ` "
            "前缀、`system` 折成 `user`）；`native` = 数据集的原始形状（改之前的，只作对照）；"
            "`alluser` = 再把 role 全折成 user（**判分池的形态**，只在 LoCoMo/ScriptMem 那一簇"
            "实测到）。⚠ 不同形态的分数**不可互比**，它会写进数据指纹"
        ),
    )
    parser.add_argument("--run-id", default=None, help="缺省 <dataset>-<UTC 时间戳>")
    parser.add_argument("--step", default="step-0", help="这次 run 属于哪个 Step（§16）")
    parser.add_argument("--profile", default="local", help="configs/<profile>.yaml")
    parser.add_argument(
        "--configs-dir",
        default="configs",
        help="配置快照所在目录——**跑 arm 时指向 `configs/runs/<arm>/`**",
    )
    parser.add_argument("--reports-dir", default="eval/reports", help="run record 的落点")
    parser.add_argument(
        "--fallback-base-url",
        default=None,
        help=(
            "**超限兜底**（只为 B1）：主服务返回 5xx 时，改问这个实例（同一个 vendor、"
            "另一档 RETRIEVAL_MODE）。⚠ **用过的题数只记在 `ServiceClient` 上，"
            "runner 不把它写进 run record**（那条路没有实现）"
        ),
    )
    parser.add_argument(
        "--switches", default=None, help="消融臂声明，如 '{\"rerank.enabled\": false}'"
    )
    parser.add_argument("--notes", default="", help="写进 run record 的自由文本")
    parser.add_argument("--embedder", default=os.environ.get("TIANXIMEM_EMBED_MODEL", ""))
    parser.add_argument("--llm", default="")
    parser.add_argument("--reranker", default=os.environ.get("TIANXIMEM_RERANKER_MODEL", ""))
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "服务写的 §14 指标快照路径（缺省取 `TIANXIMEM_METRICS_PATH`）。"
            "`rerank.degraded > 0` 会在这里被念出来——**端点是死是活只有这一条会说**（V12）"
        ),
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="裁判 subprocess 的超时（秒）。**缺省不设**：Full run 要跑 0.5–2 天",
    )
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=0,
        help="不同用户投喂/检索并行度；缺省 0 = 自动 4 路。"
        "每个用户内部保持原批次/事件顺序；1 = 串行",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=0,
        help="作答/判分并行度；缺省 0 = 每个对话端点一路。端点按原 sample 序号轮转；1 = 串行",
    )
    return parser


def _exit_for_error(error: Exception, *, base_url: str) -> int:
    """把"跑一轮"里的三类失败映射成**退出码 + 一句可照做的提示**。

    ⚠ **三类必须分开，混为一谈会把人引向错误的动作**：

    | 类 | 含义 | 提示什么 |
    | --- | --- | --- |
    | `502/503/504` | 有的机器上意味着**服务根本没起**（WSL + Docker 把死端口接成 502） | 起服务 |
    | 其余 5xx | **服务在跑，是这一步真的失败了**（比如网关抖了一下） | 看服务端日志 |
    | 4xx | 服务在跑、但拒绝了我们的请求 ⇒ **runner 自己的 bug**，不是前置条件 | 改 runner |
    """
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in (502, 503, 504):
            print(_unreachable_hint(base_url, error), file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        if status < 500:
            print(f"服务拒绝了请求（{status}）：{error}", file=sys.stderr)
            return EXIT_FAILED
        body = (error.response.text or "").strip()[:400]
        print(
            f"服务内部错误（{status}）——**服务在跑，是这一步失败了**（看服务端日志）：\n"
            f"  {error}\n  {body}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    if isinstance(error, httpx.TransportError):
        print(_unreachable_hint(base_url, error), file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    if isinstance(error, EmptyDatasetError):
        print(
            f"{error}\n"
            "  ⇒ 按顺序查两件事：\n"
            "     1) `bench_dir` 指对了没有（`TIANXIMEM_BENCHMARK_DIR`，"
            "默认 `dataset/`）；\n"
            "     2) 归档取回来了没有——**新机器上 clone 完一个数据文件都没有**：\n"
            "        `make fetch-data`（公开源，不需要凭据），见 docs/benchmark-data.md",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED
    # 剩下的只可能是 FileNotFoundError（调用点的 except 元组钉着）
    print(
        f"归档里缺文件：{error}\n先 `make fetch-data`（见 docs/benchmark-data.md）",
        file=sys.stderr,
    )
    return EXIT_PRECONDITION_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        print("⚠ `--limit` 不能为负——负值会变成 Python 切片，取到的题数看起来正常但语义反了", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    if args.max_questions is not None and args.max_questions < 0:
        print("⚠ `--max-questions` 不能为负", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    try:
        if args.input_contract == INPUT_CONTRACT:
            validate_options(
                args.dataset,
                add_shape=args.add_shape,
                time_style=args.aml_time_style,
                pool=args.aml_pool,
            )
            if args.skip_ingest or args.top_k != DEFAULT_TOP_K:
                raise ValueError(
                    "AML plans use their event checkpoints and require top_k=100; "
                    "do not use --skip-ingest"
                )
        elif args.aml_pool is not None or args.aml_time_style != "synthetic":
            raise ValueError("--aml-pool/--aml-time-style require --input-contract aml-v1")
    except (ValueError, FileNotFoundError) as error:
        print(f"输入契约前置检查失败：{error}", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    bench_dir = benchmark_dir()
    reports_dir = Path(args.reports_dir)
    run_id = args.run_id or derive_run_id(
        args.dataset + ("-" + args.input_contract if args.input_contract != "native" else "")
    )
    out_dir = reports_dir / "runs" / run_id

    if args.frozen:
        # **冻结口径**（[`recipes.py`](./recipes.py)）——"这个数据集跑多少题"的唯一声明处。
        # ⚠ 与显式的 `--limit` / `--spread` **不许混用**：两处同时给就会出现
        # "记录的口径"与"实际跑的口径"分家，而那是**不可比**里最难查的一种。
        clashing = [
            name for name, given in (("--limit", args.limit), ("--spread", args.spread)) if given
        ]
        if clashing:
            print(
                f"⚠ `--frozen` 与 {' / '.join(clashing)} 不能同时给——"
                "冻结口径就是为了让'跑的是什么'只有一个来源。",
                file=sys.stderr,
            )
            return EXIT_PRECONDITION_FAILED
        chosen = recipe_for(args.dataset, input_contract=args.input_contract)
        args.limit, args.spread = chosen.limit, chosen.spread
        expected = f"预期 {chosen.n_questions} 题"
        if args.input_contract == INPUT_CONTRACT and args.dataset == "feverous" and args.aml_pool:
            expected = "自定义页面池；公开 dev 抽样题数以 input-manifest.json 为准"
        print(f"  冻结口径 {args.dataset}: {chosen.flags()}  ⇒ {expected}")

    # ⚠ `note` 由两半拼成，**两半都是"不可比"的警示**：
    #   · 截断（`--limit` / `--max-questions`）——跑了一小撮
    #   · 数据集的**形状是我们造的**那些约定（`shape_note`）——改它会改分数，
    #     而两次 run 的 record 否则长得一模一样
    note = "\n".join(
        part
        for part in (
            truncation_note(args.limit, max_questions=args.max_questions, spread=args.spread),
            "**冻结口径**（`eval/experiments/recipes.py`）——与同口径的历史 run 可比。"
            if args.frozen
            else "",
            "AML 输入近似；语料范围、时间形态和批界见 input-manifest.json。"
            "评分复用本地管线，不等同 AML 平台。"
            if args.input_contract == INPUT_CONTRACT
            else shape_note(args.dataset),
        )
        if part
    )

    missing = judge_preconditions()
    if missing:
        print(
            "裁判那一步跑不起来：归档 pipeline 读不到 " + " / ".join(missing) + "。\n"
            "  ⇒ 用 `make eval`（它带 `uv run --env-file .env`），"
            "或自己加 `uv run --env-file .env`。\n"
            "  ⚠ `.env` 是 `common/config/` 自己读的，**不会**进环境变量——"
            "而裁判是 subprocess，只继承环境变量。",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED

    try:
        for required in (
            required_datasets(args.dataset)
            if args.input_contract == INPUT_CONTRACT
            else (args.dataset,)
        ):
            ensure_dataset(required, bench_dir, offline=args.offline)
    except PreparationError as error:
        print(f"数据准备失败：{error}", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED

    absent = missing_pipeline(args.dataset, bench_dir)
    if absent is not None:
        print(
            f"裁判脚本不在：{absent}\n"
            "  ⇒ 归档里那几份 pipeline 是**跟着数据集一起取回**的：`make fetch-data`"
            "（见 docs/benchmark-data.md）。\n"
            "  ⚠ 查在这里是因为**再不查就来不及了**：脚本缺了不会被「起进程」拦下，"
            "整轮的\n"
            "     `Add` 与 `Search` 会全部跑完（embedding 的钱已经付了），"
            "到第一个样本的裁判那步才炸。",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED

    try:
        samples, results = run_round(
            dataset=args.dataset,
            base_url=args.base_url,
            bench_dir=bench_dir,
            out_dir=out_dir,
            top_k=args.top_k,
            limit=args.limit,
            max_questions=args.max_questions,
            skip_ingest=args.skip_ingest,
            date_mode=args.memory_date,
            annotate_mark=args.annotate_mark,
            spread=args.spread,
            fallback_base_url=args.fallback_base_url,
            add_shape=args.add_shape,
            input_contract=args.input_contract,
            aml_time_style=args.aml_time_style,
            aml_pool=args.aml_pool,
            judge_workers=args.judge_workers,
            ingest_workers=args.ingest_workers,
            execution_fingerprint={
                "config": config_fingerprint(
                    args.profile, _parse_switches(args.switches), configs_dir=Path(args.configs_dir)
                ),
                "models": models_fingerprint(
                    embedder=args.embedder, llm=args.llm or _default_llm(), reranker=args.reranker
                ),
                "frozen": args.frozen,
                "limit": args.limit,
                "spread": args.spread,
                "max_questions": args.max_questions,
            },
        )
    except (
        httpx.HTTPStatusError,
        httpx.TransportError,
        FileNotFoundError,
        EmptyDatasetError,
        ValueError,
    ) as error:
        if isinstance(error, ValueError) and not isinstance(error, EmptyDatasetError):
            print(f"输入计划失败：{error}", file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        return _exit_for_error(error, base_url=args.base_url)

    record = _assemble_record(
        args, run_id=run_id, bench_dir=bench_dir, samples=samples, results=results, note=note
    )
    _print_result(record, write_record(record, reports_dir))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
