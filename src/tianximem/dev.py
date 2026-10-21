"""TianXiMem 跨平台开发命令（**命令的唯一实现在这里**）。

`Makefile` 只提供兼容别名，实际命令由本模块实现。Windows 上没有 `make` 也能运行同一组命令（见 [`docs/active-spec/development.md`](../../docs/active-spec/development.md)）。

三条约定：

* `python -m tianximem.dev help`（同 `--help`）是**说明的唯一入口**——每个子命令的 `--help`
  带着 Makefile 注释中的边界、判据与常量出处；
* 多余参数（`ARGS='…'`）**原样下传**给被调用的脚本，清单见 `_PASSTHROUGH_COMMANDS`；
* 子进程一律 `subprocess.run([...], shell=False)`——不依赖 make / grep / awk / sed / date。
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
_DEPLOY_ENV_FILE = _REPO_ROOT / "deploy" / ".env"


def _run(command: Sequence[str], *, cwd: Path = _REPO_ROOT) -> int:
    """Run a child process without relying on a shell or shell utilities.

    相对路径一律以仓库根为基准（`deploy/compose.yaml`、`tools/…`、`eval/…` 都这么写）。
    ⚠ 可执行文件不在 PATH 里时给**清晰的退出码 127**（WSL 里的 `docker` 就是这种情况），
    而不是一条 traceback。
    """

    try:
        completed = subprocess.run(list(command), cwd=cwd, check=False)
    except FileNotFoundError:
        print(f"找不到命令：{command[0]}（不在 PATH 里）", file=sys.stderr)
        return 127
    return completed.returncode


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


def _uv(*args: str) -> list[str]:
    executable = shutil.which("uv")
    if executable is None:
        raise SystemExit("找不到 uv；请先安装 uv 并加入 PATH")
    return [executable, *args]


def _uv_run(*args: str) -> list[str]:
    """`uv run python …`——**不显式读 `.env`**（脚本自己会读，或只读 `os.environ`）。"""

    return [*_uv("run"), "python", *args]


def _uv_run_env(*args: str) -> list[str]:
    """`uv run [--env-file .env] python …`——需要 `.env` 的评测与自检走这条。

    ⚠ **`.env` 由 [`common/config/`](common/config/) 读**
      ⇒ `serve` **不需要**额外的 `--env-file`，别再加一条会与它打架的注入路径。
      但 `tools/check_env.py` 是**独立进程**、只读 `os.environ`，所以 `check` 那一项
      显式带 `--env-file`——它也刻意**不依赖配置层**（配置层坏了它还得能跑）。
    ⚠ `--env-file` **只在 `.env` 存在时才加**：`uv run` 碰到不存在的 env 文件会**直接报错**，
      而取数据**不需要任何密钥**，不该卡在"还没建 `.env`"上（新机器上 `fetch-data` 甚至
      可以排在 `cp .env.example .env` 之前）。存在时就必须带上——见 `eval` 的说明。
    """

    command = _uv("run")
    if _ENV_FILE.is_file():
        command.extend(("--env-file", str(_ENV_FILE)))
    return [*command, "python", *args]


def _uvicorn_args() -> list[str]:
    return _python(
        "-m",
        "uvicorn",
        "tianximem.service.app:create_app_from_env",
        "--factory",
        "--workers",
        "1",
    )


def _docker(*args: str) -> list[str]:
    return ["docker", *args]


def _compose(*args: str) -> list[str]:
    """`docker compose -f deploy/compose.yaml …`（cwd 由 `_run` 钉在仓库根）。

    ⚠ daemon 在 **Windows 侧**：WSL 里 docker CLI 可能因 socket
      权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
      端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
    """

    return _docker("compose", "-f", "deploy/compose.yaml", *args)


def _dated_run_dir(prefix: str) -> str:
    """返回 `eval/reports/runs/<prefix>-YYYYMMDD`。"""

    return f"eval/reports/runs/{prefix}-{datetime.now().strftime('%Y%m%d')}"


def _config_check() -> int:
    """Load the repository default profile with deterministic placeholder env."""

    from tianximem.common.config import (
        ENV_EMBED_API_KEY,
        ENV_EMBED_BASE_URL,
        ENV_LLM_API_KEY,
        ENV_LLM_BASE_URL,
        ENV_LLM_MODEL,
        load_config,
    )

    env = {
        ENV_EMBED_BASE_URL: "http://config-check.invalid/v1",
        ENV_EMBED_API_KEY: "config-check",
        ENV_LLM_BASE_URL: "http://config-check.invalid/v1",
        ENV_LLM_API_KEY: "config-check",
        ENV_LLM_MODEL: "config-check",
    }
    config = load_config(env, config_dir=_REPO_ROOT / "configs")
    print(f"profile={config.profile}")
    print(f"router.enabled={config.router.enabled}")
    print(f"agentic.enabled={config.agentic.enabled}")
    print(f"agentic.mode={config.agentic.mode}")
    print(f"retrieval.grounded_evidence={config.retrieval.grounded_evidence}")
    print(f"packaging.annotate_relatives={config.packaging.annotate_relatives}")
    return 0


def _clean() -> int:
    targets = (
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "var/*.db",
        "var/*.db-journal",
        "var/*.db-wal",
        "var/*.db-shm",
    )
    for pattern in targets:
        for raw in glob.glob(str(_REPO_ROOT / pattern)):
            path = Path(raw)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    return 0


def _data_command(args: argparse.Namespace) -> list[str]:
    command = {
        "data-check": "--check",
        "data-fetch": "--fetch",
        "data-patch": "--patch",
    }[args.command]
    result = _python("-m", "eval.datasets.prepare", command)
    if args.dataset:
        for dataset in args.dataset:
            result.extend(("--dataset", dataset))
    elif args.tier:
        result.extend(("--tier", args.tier))
    if args.offline:
        result.append("--offline")
    if args.dir is not None:
        result.extend(("--dir", str(args.dir)))
    return result


# ── 命令实现 ─────────────────────────────────────────────────────────────────
# 每个函数只干一件事：把参数拼成子进程。所有坑、判据与常量出处写在 `build_parser`
# 里的 `help=` / `notes=` 上（`<cmd> --help` 能看到）。
def _cmd_help(args: argparse.Namespace, extra: list[str]) -> int:
    build_parser().print_help()
    return 0


def _cmd_sync(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv("sync", "--all-extras"))


def _cmd_lock(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv("lock"))


def _cmd_format(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_python("-m", "ruff", "format", "src", "eval", "tools", "tests"))


def _cmd_format_check(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_python("-m", "ruff", "format", "--check", "src", "eval", "tools", "tests"))


def _cmd_lint(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_python("-m", "ruff", "check", *args.paths))


def _cmd_typecheck(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_python("-m", "mypy", "src"))


def _cmd_test(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_python("-m", "pytest", *extra))


def _cmd_serve(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uvicorn_args())


def _cmd_config_check(args: argparse.Namespace, extra: list[str]) -> int:
    return _config_check()


def _cmd_check(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("tools/check_env.py", *extra))


def _cmd_data(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_data_command(args))


def _cmd_contract_check(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("eval/smoke/preflight.py", *extra))


def _cmd_diagnose(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("tools/diagnose_run.py", "--run-id", args.run_id, *extra))


def _cmd_probe_reranker(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("tools/probe_reranker.py", *extra))


def _cmd_router_eval(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("tools/router_eval.py", "--out", args.out, *extra))


def _cmd_coq_eval(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("tools/coq_eval.py", "--out", args.out, *extra))


def _cmd_coq_decompose(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(
        _uv_run_env(
            "tools/coq_decompose_probe.py",
            "--manifest",
            "eval/coq-musique.jsonl",
            "--out",
            "eval/reports/runs/coq-decompose",
            *extra,
        )
    )


def _cmd_order_probe(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("tools/order_probe.py", "--processes", str(args.processes), *extra))


def _cmd_export_multihop(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("tools/export_multihop_kit.py", *extra))


def _cmd_eval(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("-m", "eval.experiments.run", "--dataset", args.dataset, *extra))


def _cmd_baseline(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(
        _uv_run_env("-m", "eval.experiments.run", "--dataset", args.dataset, "--frozen", *extra)
    )


def _cmd_replay_official(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("-m", "eval.experiments.replay_official", *extra))


def _cmd_t1(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run_env("-m", "eval.experiments.t1_timestamp", *extra))


def _cmd_t2(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("-m", "eval.experiments.t2_cross_session", *extra))


def _cmd_t2_dump(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_uv_run("tools/t2_retrieval_dump.py", *extra))


def _cmd_qdrant_up(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_compose("up", "-d", "qdrant", *extra))


def _cmd_qdrant_down(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_compose("stop", "qdrant", *extra))


def _cmd_image_build(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_docker("build", "-f", "deploy/Dockerfile", "-t", "tianximem:local", ".", *extra))


def _cmd_up(args: argparse.Namespace, extra: list[str]) -> int:
    if not _DEPLOY_ENV_FILE.is_file():
        print("缺 deploy/.env —— 先 cp deploy/.env.example deploy/.env 并填值", file=sys.stderr)
        return 1
    return _run(_compose("up", "-d", "--build", *extra))


def _cmd_down(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(_compose("down", *extra))


def _cmd_deploy_check(args: argparse.Namespace, extra: list[str]) -> int:
    return _run(
        _compose(
            "exec",
            "-T",
            "app",
            "python",
            "eval/smoke/preflight.py",
            "--base-url",
            "http://127.0.0.1:8000",
            *extra,
        )
    )


def _cmd_clean(args: argparse.Namespace, extra: list[str]) -> int:
    return _clean()


def _cmd_smoke(args: argparse.Namespace, extra: list[str]) -> int:
    # `@echo …; @false` 的替代：占位也要**明确失败**，别让它静默成功。
    print("NOT_IMPLEMENTED: smoke discriminator is not available")
    return 1


# ── 子进程工厂：命令名 → 实现 ────────────────────────────────────────────────
_HANDLERS: dict[str, Callable[[argparse.Namespace, list[str]], int]] = {
    "help": _cmd_help,
    "sync": _cmd_sync,
    "lock": _cmd_lock,
    "format": _cmd_format,
    "format-check": _cmd_format_check,
    "lint": _cmd_lint,
    "typecheck": _cmd_typecheck,
    "test": _cmd_test,
    "serve": _cmd_serve,
    "config-check": _cmd_config_check,
    "check": _cmd_check,
    "contract-check": _cmd_contract_check,
    "diagnose": _cmd_diagnose,
    "probe-reranker": _cmd_probe_reranker,
    "data-check": _cmd_data,
    "data-patch": _cmd_data,
    "data-fetch": _cmd_data,
    "router-eval": _cmd_router_eval,
    "coq-eval": _cmd_coq_eval,
    "coq-decompose": _cmd_coq_decompose,
    "order-probe": _cmd_order_probe,
    "export-multihop": _cmd_export_multihop,
    "eval": _cmd_eval,
    "baseline": _cmd_baseline,
    "replay-official": _cmd_replay_official,
    "t1": _cmd_t1,
    "t2": _cmd_t2,
    "t2-dump": _cmd_t2_dump,
    "qdrant-up": _cmd_qdrant_up,
    "qdrant-down": _cmd_qdrant_down,
    "image-build": _cmd_image_build,
    "up": _cmd_up,
    "down": _cmd_down,
    "deploy-check": _cmd_deploy_check,
    "smoke": _cmd_smoke,
    "clean": _cmd_clean,
}

# `ARGS='…'` / 多余参数会**原样下传**给被调用的脚本的命令。
# 不在这里的命令见到多余参数一律**报错**（`serve` 尤其重要：别让 `--workers N` 之类的透传
# 绕开 `--workers 1` 那条约束；其余命令的额外参数没有任何去处，静默丢掉正是本仓最不想要的
# 那种"看起来正常"）。
_PASSTHROUGH_COMMANDS = frozenset(
    {
        "test",
        "check",
        "contract-check",
        "diagnose",
        "probe-reranker",
        "router-eval",
        "coq-eval",
        "coq-decompose",
        "order-probe",
        "export-multihop",
        "eval",
        "baseline",
        "replay-official",
        "t1",
        "t2",
        "t2-dump",
        "qdrant-up",
        "qdrant-down",
        "image-build",
        "up",
        "down",
        "deploy-check",
    }
)


def _add_command(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
    notes: str | None = None,
) -> argparse.ArgumentParser:
    """注册一个子命令：`help_text` 进命令清单，`notes` 进 `<cmd> --help`。

    ⚠ `help_text` 会被 argparse 做一次 `%` 格式化（裸 `%` 会让**整个** `--help` 直接炸），
    所以百分比只许写在 `notes` 里。
    """

    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=help_text if notes is None else f"{help_text}\n\n{notes}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


def _add_environment_commands(subparsers: argparse._SubParsersAction) -> None:
    """安装 / 格式化 / 检查 / 起服务——开发环路里最先用到的一批。"""

    _add_command(
        subparsers, "sync", "安装依赖（`--all-extras`：会装上 [local] 的本地模型栈，数 GB）"
    )
    _add_command(subparsers, "lock", "生成/更新 uv.lock")
    _add_command(subparsers, "format", "格式化（ruff format src / eval / tools / tests）")
    _add_command(subparsers, "format-check", "检查格式化（不改文件）")
    lint = _add_command(subparsers, "lint", "静态检查（ruff/mypy 只查 src/）")
    lint.add_argument("paths", nargs="*", default=["src"])
    _add_command(
        subparsers,
        "typecheck",
        "类型检查（mypy）",
        notes="""\
类型检查当前只覆盖进入提交链路的 `src/`。扩展范围前，先确保新增范围在本机可稳定通过，
避免让门禁长期处于失败状态。
""",
    )
    _add_command(
        subparsers,
        "config-check",
        "加载默认 profile 并打印关键开关（CI 的配置契约检查）",
        notes="""\
用**确定性的占位环境变量**加载 `configs/default.yaml`——不读 `.env`、不连任何端点，
只看配置层能不能装配起来（CI 用这一条）。
""",
    )
    _add_command(subparsers, "test", "跑单元测试")
    _add_command(
        subparsers,
        "serve",
        "起 Add/Search 服务（FastAPI + uvicorn，单进程）",
        notes="""\
⚠ 必须使用 `--workers 1`。多进程写入、SQLite 锁争用和每进程 Qdrant 客户端尚未验证；
   配置校验与进程检查会同时拒绝多 worker 启动。
⚠ 使用 `--factory`，避免在模块导入阶段读取环境变量。
⚠ 启动需要 `.env` 中的 embedding 端点和 `configs/default.yaml`；缺失时直接失败。
   开发期建议设置 `TIANXIMEM_PROFILE=local`，使用独立的 `memories_dev` 集合。
""",
    )
    _add_command(
        subparsers,
        "check",
        "环境自检：Qdrant 可达 / 模型端点可用 / 密钥已填",
        notes="""\
⚠ 不检查本机 GPU；部署依赖远程模型端点。
⚠ 对话端点检查同时验证回答正文不混入推理过程。
与 `contract-check` 的分工：本项检查环境和端点可用性，不验证 HTTP 契约。
""",
    )


def _add_data_commands(subparsers: argparse._SubParsersAction) -> None:
    """`dataset/` 的准备与校验。"""

    for name, help_text, notes in (
        (
            "data-check",
            "离线校验指定数据集与评分依赖；`--tier all` 校验全档",
            """\
⚠ 清单里记两套哈希：`upstream_sha256`（上游原始字节）与 `sha256`（本地归档，打完补丁）。
   7 个 pipeline 有一处**已声明的本地修订**（上游那份跑不起来），见 docs/benchmark-data.md。
""",
        ),
        (
            "data-patch",
            "仅应用指定数据集评分源码的已声明补丁（不下载）",
            """\
⚠ 它只做清单里声明的那一处替换（`contextlib.nullcontext` 包住 `Path.open`），
   且**幂等**——已经打过的再跑是零处改动。tests/test_benchmark_archive.py 钉住了这条。
""",
        ),
        (
            "data-fetch",
            "按数据集准备 dataset/；`--tier all` 取全档，默认 locomo-refined",
            """\
⚠ 该命令读取 `.env`，因为 `TIANXIMEM_BENCHMARK_DIR` 是数据路径的唯一入口。
⚠ 下载公开数据不需要凭据，但需要能访问 GitHub 与 Hugging Face。
⚠ 目标目录不存在时无需预先创建；校验成功后才原子发布数据文件。
""",
        ),
    ):
        parser = _add_command(subparsers, name, help_text, notes)
        parser.add_argument("--dataset", action="append")
        parser.add_argument("--tier", choices=("required", "optional", "official-extra", "all"))
        parser.add_argument("--offline", action="store_true")
        parser.add_argument("--dir", type=Path)


def _add_evaluation_commands(subparsers: argparse._SubParsersAction) -> None:
    """评测、探针与导出——**真正打网关**的那一批（代理评测，不占 Smoke 配额）。"""

    _add_command(
        subparsers,
        "contract-check",
        "§2 契约合规自查（打真 HTTP：本地自启服务，**不消耗 Smoke 配额**）",
        notes="""\
打真 HTTP、看真响应——契约层只有走 HTTP 才碰得到（eval/CLAUDE.md 的边界）。
缺省自己起一个 uvicorn（临时库 + 临时缓存 + 独立的 memories_preflight 集合），跑完关掉并 drop。
退出码：0 全过 · 1 有检查没过 · 2 前置条件不满足（Qdrant 或 embedding 网关不可达）。
⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY（由 common/config/ 读）与可达的 Qdrant。
打已经在跑的服务：`python -m tianximem.dev contract-check --base-url http://127.0.0.1:8000`
""",
    )
    diagnose = _add_command(
        subparsers,
        "diagnose",
        "跑批诊断：把「低分」与「模型不行」分开（`--run-id RUN`；只读产物，不打网关）",
        notes="""\
接入新数据集或调查低分时先运行。输出包括拒答率、拒答与检索命中的交叉、
标签分布（含 `JUDGE_ERROR`）以及逐类表现。详细判读说明在脚本 docstring。
用法：make diagnose RUN=base-mqk
""",
    )
    diagnose.add_argument(
        "--run-id", metavar="RUN", required=True, help="eval/reports/runs/<run-id>"
    )
    _add_command(
        subparsers,
        "probe-reranker",
        "精排探针：打真网关，验连通性 + 它在链上真的起作用（不消耗 Smoke 配额）",
        notes="""\
与 contract-check 的分工：那一个验**契约形状**（不关心精排好不好用），
本项验**精排这一环**（端点通不通、线格式对不对、名次有没有真的换掉）。
退出码：0 端点与线格式都正常 · 1 端点不可用或线格式不符 · 2 没配 reranker。
⚠ **顺序有没有变不影响退出码**——那是信息，不是判据（一个诚实但保守的 reranker
   完全可能给出与 RRF 相同的顺序）。
""",
    )
    router_eval = _add_command(
        subparsers,
        "router-eval",
        "查询路由的二分类小样本评测（真端点，**不消耗 Smoke 配额**）",
        notes="""\
落点默认 `eval/reports/runs/router-<YYYYMMDD>`：`summary.json` 保存混淆矩阵、
逐类 precision/recall/F1 和成本，`rows.jsonl` 保存逐题预测。
⚠ 它只调用路由器，不发 Add/Search、不建集合、不写真源。
⚠ 需要 `.env` 中完整的对话端点配置。
⚠ `router.enabled` 控制服务默认路径；本工具独立执行路由评测。
判读时同时看类别分布和混淆矩阵，避免退化策略产生虚高准确率。
""",
    )
    router_eval.add_argument(
        "--out",
        default=_dated_run_dir("router"),
        help="落盘目录（默认 eval/reports/runs/router-YYYYMMDD）",
    )

    coq_eval = _add_command(
        subparsers,
        "coq-eval",
        "多跳检索（Chain-of-Query）的四臂真实评测：E0/E1/E2 ＋ 每轮精排消融",
        notes="""\
落点默认 eval/reports/runs/coq-<YYYYMMDD>：`report.json`（各臂准确率 + §14 用量 + 证据覆盖）
与 `<臂>-hits.json`（逐题返回的 id）与 `<臂>/`（逐题答案与判分）。
⚠ **它会真的调网关**（embedding / 精排 / 对话模型）并**新建一个隔离 Qdrant collection**
   + 一份隔离 SQLite 真源（`<out>/coq-eval.db`）——**不碰开发库**。
⚠ 需要 `.env` 的 `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`（多跳与路由器共用）
   与 `TIANXIMEM_RERANKER_*`（每轮精排要用它）。
⚠ 跑一次要**几十分钟**（4 臂 × 25 题，多跳臂每题最多 3 次 LLM + 4 次精排）。
   续跑：`ARGS='--skip-add'`（复用已经建好的隔离真源）。
常用变体：`ARGS='--arms e0,e1,e2,a --limit-questions 5'`（先冒烟）·
   `ARGS='--skip-judge'`（只测检索覆盖，不调用裁判）。
""",
    )
    coq_eval.add_argument(
        "--out",
        default=_dated_run_dir("coq"),
        help="产物目录（默认 eval/reports/runs/coq-YYYYMMDD）",
    )

    _add_command(
        subparsers,
        "coq-decompose",
        "多跳分解能力探针：只提供问题，检查模型能否拆出依赖链",
        notes="""\
⚠ 它不执行检索、不创建集合，只对每题调用一次对话模型。
判定由人工完成：逐条比较模型分解与金标，并填写 `eval/coq-decompose-labels.jsonl` 的 `defect`。
变体：`ARGS='--limit 5'`（冒烟）· `ARGS='--reuse'`（不调模型，只重算已有分解）。
重判应使用 `--reuse`，确保标签对应同一份候选分解。
产物中的 `decompositions.md` 将原问题、模型分解与金标分解并排展示；它包含数据集原文，只写入 ignored 的 `runs/`。
""",
    )
    order_probe = _add_command(
        subparsers,
        "order-probe",
        "位置性质探针：乱序并发投喂与顺序投喂得到逐字一致的真源",
        notes="""\
⚠ 它会执行真实 embedding，因此不进入 `make test`。
⚠ 需要空闲端口和独立集合；协议会自行启动并关闭测试服务。
⚠ `--processes 4` 用四个独立进程共享同一套存储，用于验证多进程写入是否符合约束。
⚠ 测试结束后保留 `var/order-a/`、`var/order-b/` 与对应 Qdrant 集合供比对；需要时手动清理。
""",
    )
    order_probe.add_argument(
        "--processes",
        type=int,
        default=1,
        help="B 臂用几个独立进程（默认 1；4 = 测多 worker 安不安全）",
    )

    _add_command(
        subparsers,
        "export-multihop",
        "打包链式多跳题集（含共享语料池）给外部项目用 → var/exports/",
        notes="""\
产物：`var/exports/musique-multihop-<日期>/{questions.jsonl, haystack-corpus.jsonl, README.md}`
      ＋同名 `.zip`。⚠ **含数据集原文** ⇒ 只进 `var/`（gitignored），⛔ 绝不进 git。
⭐ 它**逐条校验后才打包**：可答 / 严格链 / **每一跳的答案（含末答）都在支持段落正文里**
   / 题面不泄中间跳。任一条不过就点名报出并**非零退出**——这种包交出去比不交更糟。
⚠ `haystack-corpus.jsonl` 是评测前提：如果只提供目标题自己的候选段落，
   检索没有选择空间，链式依赖会消失。
""",
    )
    eval_command = _add_command(
        subparsers,
        "eval",
        "跑一轮代理评测（§13）：`--dataset` 与透传参数可覆盖",
        notes="""\
⚠ **需要服务在跑**（`make serve`）——runner 打 HTTP、不 import `src/`（eval/CLAUDE.md 的边界）。
  连不上服务时退出码 2，并提示去起服务（不是"跑失败"）。
⚠ `--env-file` **不是多余的**：裁判那一步起的是 subprocess，它只继承环境变量，而
  `.env` 是 `common/config/` 自己读的。少了它 → Add/Search 全跑完、**裁判才炸**。
  缺哪个名字会在跑之前直说（退出码 2）。
⚠ 缺 `--embedder` 时 run record 的模型指纹会写“未声明”；需要可比记录时显式提供模型标识。
⚠ 冒烟用 `ARGS='--limit 3'`；截断运行会写入数据指纹警示，不能与全量比较。
""",
    )
    eval_command.add_argument(
        "--dataset", default="locomo-refined", help="数据集名称"
    )

    baseline = _add_command(
        subparsers,
        "baseline",
        "跑一轮**冻结口径**的基线/对照（§13）：`--dataset` 与透传参数可覆盖",
        notes="""\
**判断"改动值不值"就用这个**：口径（跑多少题、要不要分层）由
`eval/experiments/recipes.py` 给，不手抄数字——抄错一个数的后果是分数看起来完全正常、却不可比。
⚠ 它**拒绝**与手写的 `--limit` / `--spread` 同时给（那两个只在做消融或冒烟时才用）。
⚠ 各数据集的运行墙钟差异很大；按改动影响面选择数据集。
✅ 默认按独立用户并行投喂/检索，判分单独并行；可用
  `ARGS='--ingest-workers 4 --judge-workers 4'` 控制两段（单用户批次顺序不变）。
  两项均设 1 恢复串行。⚠ 别同时跑两条不同的评测链。
⚠ 本仓噪声底是 **346 题上 ~1pt** ⇒ 几十题的小跑法读不出改动。
用法：make baseline DATASET=locomo-refined
      make baseline DATASET=clbench ARGS='--run-id clb-after-rerank --switches {…}'
""",
    )
    baseline.add_argument(
        "--dataset", default="locomo-refined", help="数据集名称"
    )

    _add_command(
        subparsers,
        "replay-official",
        "重放官方采集流量并判分（README §6.2 线①）：透传参数可覆盖",
        notes="""\
**按采集的 timeline 交错重放**（不是"灌完再问"）——41% 的题早于该 user 的最后一条 add，
先灌完再问会读到未来。payload 用采集原文（不加前缀、不重切批）。
⚠ 前置：套件得先重建过——`uv run python tools/build_official_kit.py`（读采集目录里那三份）。
⚠ 冒烟用 `ARGS='--users 1 --max-questions 3'`；正式的用 `ARGS='--users 261 --run-id …'`。
⚠ 与 `make eval` 同一条纪律：**别同时跑两条不同的链**（抢同一份网关与重试预算）。
""",
    )
    _add_command(
        subparsers,
        "t1",
        "§13 的 T1 实验：时间戳前缀 带/不带（**改渲染 = 重建索引**）",
        notes="""\
两个臂各需**独立集合**（渲染不同 ⇒ 向量不同），见该脚本顶部。
""",
    )
    _add_command(
        subparsers,
        "t2",
        "§13 的 T2 实验：跨 session 失败归因（汇总人工标注，机器半边见 t2-dump）",
        notes="""\
产出的是**待填的标注表**的汇总——三类由人填，脚本拒绝代填（半张表的分布更危险）。
""",
    )
    _add_command(
        subparsers,
        "t2-dump",
        "T2 的机器半边：133 道 multi-session 题的纯 BM25 名次 → 待填表",
        notes="""\
⚠ 它直接查询 Qdrant 的 BM25 单路，不走服务；服务主路径仍是混合检索。
⚠ 前提：同一批语料已经写入目标集合。
""",
    )


def _add_docker_commands(subparsers: argparse._SubParsersAction) -> None:
    """容器形态：整栈（检索服务 + Qdrant）。

    与 `qdrant-up` / `qdrant-down` 的分工：那两条只起库、`serve` 在本机跑服务（**开发环路**）；
    `image-build` / `up` / `down` 把**服务本身**也放进容器（**部署环路**，见 deploy/CLAUDE.md）。
    """

    _add_command(
        subparsers,
        "qdrant-up",
        "起 Qdrant server（必须 server 模式；单分片）",
        notes="""\
⚠ **带服务名 `qdrant`**：compose 里现在还有 `app`（容器形态的那一套，见 `up`），
   不带服务名会把检索服务也一并起起来，和 `make serve` 抢 8000。
⚠ 如果 WSL 中的 Docker CLI 无法访问宿主机 daemon，请在 Windows 侧运行同一命令。
   端口转发后，WSL 仍可通过 localhost 访问 Qdrant。
⚠ 报 `502 Bad Gateway` 而不是「拒绝连接」= 转发在、**容器没起**——正是该跑这条命令的时候。
""",
    )
    _add_command(
        subparsers,
        "qdrant-down",
        "停 Qdrant",
        notes="""\
⚠ `stop` 而不是 `down`：`down` 是**按项目**拆的，会把 `app` 一起拆掉
   （哪怕你只是想停一下 Qdrant）。要整栈下线用 `make down`。
""",
    )
    _add_command(
        subparsers,
        "image-build",
        "构建检索服务镜像（上下文是仓库根，不是 deploy/）",
        notes="""\
⚠ `-f deploy/Dockerfile .` 的最后那个 `.` **不能省**：Dockerfile 要 pyproject.toml /
   uv.lock / src/ / configs/，所以上下文必须是仓库根（deploy/.dockerignore 在根上）。
⚠ 镜像里【不装 [local] extra】（torch 那几个 G）——提交链路的三个模型全在远端网关。
""",
    )
    _add_command(
        subparsers,
        "up",
        "起整栈（检索服务 + Qdrant，**容器形态**）",
        notes="""\
⚠ 前置检查是**故意**的：compose 自己的报错是 `env file ... not found`，
   它不会告诉你"该从哪拷"。而缺 .env 时 compose 仍可能起容器 ⇒ 服务在容器里
   因缺 embedding 密钥响亮失败、restart 策略再把它反复拉起来——看着像崩溃循环。
""",
    )
    _add_command(
        subparsers,
        "down",
        "停整栈（**保留卷**；加 `-v` 才会删真源）",
        notes="""\
⚠ **默认不删卷**：`tianxi_data` 里是 SQLite 真源，删了不可恢复（Qdrant 那份才可重建）。
""",
    )
    _add_command(
        subparsers,
        "deploy-check",
        "对**已部署的容器**跑契约预检（真 HTTP，不消耗 Smoke 配额）",
        notes="""\
⚠ 预检跑在**容器里**而不是本机：它要打 `http://127.0.0.1:8000`（容器自己的网卡）。
   从本机打是 `make contract-check` 那条路（`--base-url http://127.0.0.1:<宿主端口>`）。
""",
    )


def _add_housekeeping_commands(subparsers: argparse._SubParsersAction) -> None:
    """占位与清理。"""

    _add_command(
        subparsers,
        "smoke",
        "AML 端到端判别实验（尚未实现）",
        notes="""\
该命令目前是占位实现并明确返回失败，避免被误认为可用。
""",
    )
    _add_command(
        subparsers,
        "clean",
        "清运行时产物（保留 dataset/）",
        notes="""\
⚠ 真源是 `var/tianxi.db`，不是 `data/`。
⚠ `tianxi.db` 是 `var/` 中唯一不可重建的状态；Qdrant 卷和 embedding 缓存均可重建。
⚠ 本命令可能删除真源，执行前确认。
""",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tianximem.dev",
        description="TianXiMem 开发命令（`help` 与 `--help` 等价；每条命令的 `--help` 带参数与那些坑）。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_command(subparsers, "help", "列出所有命令与说明（与 `--help` 同一份）")
    _add_environment_commands(subparsers)
    _add_data_commands(subparsers)
    _add_evaluation_commands(subparsers)
    _add_docker_commands(subparsers)
    _add_housekeeping_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in _PASSTHROUGH_COMMANDS:
        parser.error(
            f"`{args.command}` 不接受透传参数：{' '.join(extra)}"
            f"（`python -m tianximem.dev {args.command} --help` 看它接受什么）"
        )
    return _HANDLERS[args.command](args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
