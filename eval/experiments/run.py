"""通用 runner —— **给定数据集与配置，跑一轮，落一份 run record**（§12.1 / §13）。

> **本文件是"把零件串起来"的那一层**，它不实现其中任何一环：
>
> ```text
> eval.datasets          加载 + 归一化（schema 落差预处理）
>   → driver.ServiceClient   切批喂 Add（走 HTTP）、逐题 Search
>   → judge.build_input_items / run_judge   注入记忆 + 起归档 pipeline 裁判
>   → run_record.build_record / write_record    配置指纹 + 数据指纹 + 分数
> ```
>
> **一条边界**：打 HTTP、**不 import `src/tianxi_am`**（[`../CLAUDE.md`](../CLAUDE.md)）。
> 所以本文件只依赖 `eval/` 与标准库 + `httpx`。

## 为什么需要一个通用 runner

§13 的每一条对照（A0 / A3 / A4 / T1 / T2）都要"跑一轮并落一份可比记录"，
而**它们的差别只在配置与开关上**——把这一轮跑通的地方只该有一处。
各 arm 的脚本（[`t1_timestamp.py`](./t1_timestamp.py)、
[`t2_cross_session.py`](./t2_cross_session.py)）
做的是"声明自己改了哪几个开关 + 指向哪份配置快照"，**然后调本模块**。

## 用法

```bash
# 打已在跑的服务（`make serve`），跑全量 LoCoMo-Refined
uv run python eval/experiments/run.py --dataset locomo-refined --embedder Qwen/Qwen3-Embedding-8B

# 冒烟：只跑 3 个 sample，且不重投已有语料
uv run python eval/experiments/run.py --dataset longmemeval-s --limit 3 --skip-ingest
```

⚠ **`--limit` 会写进数据指纹的 `note`**——"跑了一小撮"与"跑完了"的数字
**看起来一样**，不写下来就会有人拿它们比（§13）。
⚠ **重跑是安全的**：`request_id` 由 `(user_id, session_id, 批序号)` 确定性派生，
`applied_batches` 的批次级幂等守卫会命中（§6.5）。但**换过渲染模板/配对规则后不行**
——那会改变 `pair_idx` 的分配，旧行不会被覆盖（`pair_idx` 是读-改-写）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
from eval.datasets import (
    Sample,
    benchmark_dir,
    data_fingerprint,
    load_locomo,
    load_longmemeval,
)
from eval.harness import (
    ServiceClient,
    build_input_items,
    build_record,
    pipeline_for,
    run_judge,
    write_record,
)
from eval.harness.api_config import ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL
from eval.harness.judge import DATE_MODES, MARKS

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_PRECONDITION_FAILED: Final[int] = 2

#: §2.2：AML 固定传 `top_k=100`。**这里也用它**——预检要验的就是线上那个值。
DEFAULT_TOP_K: Final[int] = 100

#: 服务地址：与 `eval/harness/__init__.py` 的示例、`make serve` 的缺省端口一致。
DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000"

#: 有加载器的数据集（**只服务两个计分数据集**，§12.4）。
DATASETS: Final[tuple[str, ...]] = ("locomo-refined", "longmemeval-s")


def _load(dataset: str, bench_dir: Path, limit: int | None) -> list[Sample]:
    """加载 + `limit`。

    ⚠ **两份数据集的 `limit` 不是同一个机制**，别当成对称的：
    LongMemEval 的 `limit` 在**加载器里**做（那份 277 MB，全量解析要几十秒、几 GB 内存），
    而 LoCoMo 的加载器没有这个参数（它整份才 10 段对话，切片在加载后做）。
    """
    if dataset == "locomo-refined":
        samples = load_locomo(bench_dir)
        return samples[:limit] if limit is not None else samples
    if dataset == "longmemeval-s":
        return load_longmemeval(bench_dir, limit=limit)
    raise ValueError(f"未知数据集 {dataset!r}——只有 {' / '.join(DATASETS)} 有加载器")


def derive_run_id(dataset: str, *, now: datetime | None = None) -> str:
    """`<dataset>-<UTC 时间戳>`——**一次 run 一个 id，且 id 里带得动数据集**。

    不用随机串：`eval/reports/runs/<run_id>.json` 是要被人翻的，
    而"这个文件是跑哪份数据的"应当从文件名就能读出来。
    """
    stamp = (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{dataset}-{stamp}"


def models_fingerprint(*, embedder: str, llm: str, reranker: str) -> dict[str, str]:
    """§12.1 R1 的模型指纹：**换模型会让所有阈值与权重失效**。

    ⚠ **这三行是"声明"，不是"探测"**（与 `.env.example` 里 reranker 那条同一口径）：
    端点本身不回可信的模型名（reranker 回的永远是服务端路径）。
    `src/` 那边的真值在 `configs/*.yaml` 的 `models.embedder`，而 **harness 不解析 yaml**
    ——解析会与 `common/config.py` 抢同一份知识
    （[`../harness/run_record.py`](../harness/run_record.py)
    的 `config_fingerprint` 只逐字节哈希它，理由在那个 docstring 里）。
    ⇒ 所以这里由调用方声明，缺了就**显式写成"没说"**，而不是编一个看起来合理的值。
    """
    return {
        "embedder": embedder or "<未声明：传 --embedder 或设 TIANXI_EMBED_MODEL>",
        "llm": llm or "<未声明：传 --llm>",
        "reranker": reranker or "disabled",
    }


def _default_llm() -> str:
    """归档 pipeline 的模型名从 `api_config` 来——**那是它们唯一的配置源**。

    `--model` 之类的 CLI 参数是死的（`answer()` / `evaluate()` 在协程开头用
    `api_config` 的常量重写 `args.*`），所以 harness 侧能读的也只有同一个名字。
    """
    return ANSWER_MODEL or ""


def judge_preconditions() -> list[str]:
    """裁判链路的**前置条件**：归档 pipeline 要的那几个名字现在必须能读到值。

    ⚠ **这是"跑之前就该失败"的那一类。** `run_judge` 起的是 **subprocess**，而 subprocess
    只继承 `os.environ`——仓库里的 `.env` 是 `common/config.py` **自己读**的，
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


def _count_questions(samples: list[Sample]) -> int:
    return sum(len(sample.questions) for sample in samples)


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
    client: ServiceClient | None = None,
) -> tuple[list[Sample], list]:
    """跑一轮的**机制部分**：加载 → 投喂 → 检索 → 裁判。返回 `(samples, results)`。

    与 `main` 分开，是为了让测试能塞一个 `httpx.MockTransport` 的客户端进来
    （[`../../tests/test_experiments.py`](../../tests/test_experiments.py) 就是这么做的），
    而 CLI 参数解析不参与那条路径。
    """
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}——只有 {' / '.join(DATASETS)} 有加载器")
    samples = _load(dataset, bench_dir, limit)
    if not samples:
        raise ValueError(f"{dataset}：一个 sample 都没加载到——bench_dir={bench_dir} 对吗？")

    owns_client = client is None
    client = client or ServiceClient(base_url)
    try:
        results: list = []
        for index, sample in enumerate(samples, start=1):
            if not skip_ingest:
                client.ingest(sample)
            hits_by_qid = {
                question.qid: client.search(
                    user_id=sample.user_id, query=question.question, top_k=top_k
                )
                for question in sample.questions
            }
            items = build_input_items(
                sample, hits_by_qid, date_mode=date_mode, annotate_mark=annotate_mark
            )
            results += run_judge(pipeline_for(bench_dir, dataset), items, out_dir / sample.user_id)
            print(f"  [{index}/{len(samples)}] {sample.user_id}：{len(items)} 题已判", flush=True)
        return samples, results
    finally:
        if owns_client:
            client.close()


def _parse_switches(raw: str | None) -> dict[str, Any]:
    """`--switches '{"neighbor.radius": 0}'`——**消融臂显式声明自己改了哪几项**。

    它与配置快照 hash 是两回事：快照回答"配置文件长什么样"，这里回答"这次跑的是哪一组消融"
    （[`../harness/run_record.py`](../harness/run_record.py) 的 `config_fingerprint`）。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--switches 不是合法 JSON：{error}") from None
    if not isinstance(parsed, dict):
        raise SystemExit("--switches 必须是一个 JSON 对象（键=开关名，值=该臂的取值）")
    return parsed


def truncation_note(limit: int | None) -> str:
    """`--limit` 的警示语——写进数据指纹的 `note`（§13）。

    **这是本文件唯一一处"截断是否发生过"的判断**：跑了一小撮与跑完了的数字
    **在 run record 里长得一模一样**，不写下来就会有人拿它们比。
    """
    if limit is None:
        return ""
    return f"⚠ **截断跑**：只加载了前 {limit} 个 sample——不可与全量比"


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
        "--base-url", default=DEFAULT_BASE_URL, help=f"服务地址（缺省 {DEFAULT_BASE_URL}）"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="§2.2 固定的 100")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 个 sample（**会写进数据指纹的 note**——截断过的数字不可与全量比）",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="跳过 Add，复用已有语料")
    parser.add_argument(
        "--memory-date",
        choices=DATE_MODES,
        default="none",
        help=(
            "注入时怎么带日期：none（默认）/ per-item（每条前缀）/ header（再在顶部写明）"
            " / annotate（把句子里的相对表达就地注解成绝对日期——见 eval/harness/annotate.py）"
        ),
    )
    parser.add_argument(
        "--annotate-mark",
        choices=MARKS,
        default="paren",
        help="`--memory-date annotate` 的记号：paren（`last Tues (…2023)`）/ tag（`[= …]`）",
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
        "--switches", default=None, help="消融臂声明，如 '{\"rerank.enabled\": false}'"
    )
    parser.add_argument("--notes", default="", help="写进 run record 的自由文本")
    parser.add_argument("--embedder", default=os.environ.get("TIANXI_EMBED_MODEL", ""))
    parser.add_argument("--llm", default="")
    parser.add_argument("--reranker", default=os.environ.get("TIANXI_RERANKER_MODEL", ""))
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="裁判 subprocess 的超时（秒）。**缺省不设**：Full run 要跑 0.5–2 天",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bench_dir = benchmark_dir()
    reports_dir = Path(args.reports_dir)
    run_id = args.run_id or derive_run_id(args.dataset)
    out_dir = reports_dir / "runs" / run_id

    note = truncation_note(args.limit)

    missing = judge_preconditions()
    if missing:
        print(
            "裁判那一步跑不起来：归档 pipeline 读不到 " + " / ".join(missing) + "。\n"
            "  ⇒ 用 `make eval`（它带 `uv run --env-file .env`），"
            "或自己加 `uv run --env-file .env`。\n"
            "  ⚠ `.env` 是 `common/config.py` 自己读的，**不会**进环境变量——"
            "而裁判是 subprocess，只继承环境变量。",
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
            skip_ingest=args.skip_ingest,
            date_mode=args.memory_date,
            annotate_mark=args.annotate_mark,
        )
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        # ⚠ **502/503/504 与其余 5xx 是两件事**：前者在有的机器上意味着"服务根本没起"
        #   （WSL + Docker Desktop 的转发把死端口接成 502），后者是**服务在跑、这一步真的失败了**
        #   ——比如网关抖了一下。混为一谈会让人跑去重启一个好端端的服务。
        if status in (502, 503, 504):
            print(_unreachable_hint(args.base_url, error), file=sys.stderr)
            return EXIT_PRECONDITION_FAILED
        if status < 500:
            # 4xx = 服务在跑、但拒绝了我们的请求 ⇒ **runner 自己的 bug**，不是前置条件。
            print(f"服务拒绝了请求（{status}）：{error}", file=sys.stderr)
            return EXIT_FAILED
        body = (error.response.text or "").strip()[:400]
        print(
            f"服务内部错误（{status}）——**服务在跑，是这一步失败了**（看服务端日志）：\n"
            f"  {error}\n  {body}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    except httpx.TransportError as error:
        print(_unreachable_hint(args.base_url, error), file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    except FileNotFoundError as error:
        print(
            f"归档里缺文件：{error}\n先 `make fetch-data`（见 docs/benchmark-data.md）",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED

    record = build_record(
        run_id=run_id,
        step=args.step,
        profile=args.profile,
        bench_dir=bench_dir,
        samples=samples,
        results=results,
        data_fingerprint=data_fingerprint(
            bench_dir,
            args.dataset,
            n_samples=len(samples),
            n_questions=_count_questions(samples),
            note=note,
        ),
        models=models_fingerprint(
            embedder=args.embedder,
            llm=args.llm or _default_llm(),
            reranker=args.reranker,
        ),
        switches=_parse_switches(args.switches),
        configs_dir=Path(args.configs_dir),
        notes=args.notes,
    )
    path = write_record(record, reports_dir)

    overall = record.scores["overall"]
    print(
        f"\n{run_id}：overall={overall}"
        f"（n={record.breakdown.get('abstention', {}).get('n', 0)} 道拒答另计）"
    )
    for category, entry in record.breakdown.items():
        print(f"  {category}: {entry}")
    print(f"\nrun record → {path}")
    print("⚠ 数字只有 eval/reports/ 一个家：结论写进 ledger.md，不要复制别处。")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
