"""TianXiMem 跨平台开发命令（**命令的唯一实现在这里**）。

历史入口是 Unix 向的 `Makefile`；它现在只剩一行转发（`uv run python -m tianximem.dev <cmd>`），
Windows 上没有 `make` 也能跑同一组命令（见 [`docs/active-spec/development.md`](../../docs/active-spec/development.md)）。

三条约定：

* `python -m tianximem.dev help`（同 `--help`）是**说明的唯一入口**——每个子命令的 `--help`
  带着原来住在 Makefile 注释里的那些坑、判据与常量出处；
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

    ⚠ **`.env` 由 [`common/config/`](common/config/) 读**（2026-09-24 起）
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

    ⚠ daemon 在 **Windows 侧**（Docker Desktop，D12）：WSL 里 docker CLI 可能因 socket
      权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
      端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
    """

    return _docker("compose", "-f", "deploy/compose.yaml", *args)


def _dated_run_dir(prefix: str) -> str:
    """`eval/reports/runs/<prefix>-YYYYMMDD`（原来由 Makefile 的 `$(shell date)` 算）。"""

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
    print("TODO(Step 6): 加一条只有它能回答的记忆，看分数是否变化")
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
⚠ **只查 `src/`，不查 `eval/` `tests/` `tools/`**（2026-09-25 定）：后三者现在有
  19 处存量报错（`conftest.py` 的 sys.path 形状、测试里的 float timeout 等），
  一起纳入等于新门禁一上来就是红的，而"永远是红的门禁"与"没有门禁"是同一个东西。
  ⇒ 先钉住**唯一进提交链路**的 `src/`；要扩到其余目录时，先清完那 19 处再改这里的路径
  （`_cmd_lint` 的默认 `paths` 与 `_cmd_typecheck` 里的 `src`）。
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
⚠ 必须 --workers 1。⚠ **D25 之后原因变了**：旧理由是"按 session 串行化用的是进程内锁，
   多 worker 会静默失效"——`SessionLocks` 已删，那条不再成立。现在拒的是
   **"放开多 worker 需要的验证一件都没做"**（并发写压力 / busy_timeout 多进程争用 /
   各自的 Qdrant 客户端）。⇒ **不要用"反正现在安全了"当理由把它去掉。**
   两条防线都会拦：`--workers N`（N>1）被 assert_single_process() 拦下，
   TIANXIMEM_WORKERS != 1 被配置校验拦下（见 src/tianximem/common/config/validate.py）。
⚠ 用 --factory 而不是模块级 app：模块级 app 会让"导入本模块"就要求环境变量齐备，
   测试没法只导入工厂函数（见 src/tianximem/service/app.py 的 docstring）。
⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY 与 configs/default.yaml；
   缺了会在启动时响亮失败（不会带着空值跑起来）。
   开发期建议 TIANXIMEM_PROFILE=local（换用独立的 memories_dev 集合）。
""",
    )
    _add_command(
        subparsers,
        "check",
        "环境自检：Qdrant 可达 / 三段模型端点可用 / 密钥已填（含 V7 思考探针）",
        notes="""\
⚠ 不检查 nvidia-smi —— 模型不在本机（D14）。
⚠ LLM 那一项顺带就是 §17.3 V7 的探针：网关默认开思考时，归档 pipeline 的
   `generated_answer` 里会混进推理过程，而**裁判读到的就是那个**。
与 `contract-check` 的分工：本项只看"环境通不通"，不打契约。
""",
    )


def _add_data_commands(subparsers: argparse._SubParsersAction) -> None:
    """`dataset/` 的准备与校验（D35 的 `--dataset` / `--tier all` / `--offline` / `--dir`）。"""

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
⚠ 它**必须读 `.env`**（2026-10-04）：`TIANXIMEM_BENCHMARK_DIR` 是数据路径的唯一入口，
   而 `.env.example` 明写"本地开发把它指到开发数据集即可"。以前这条用裸 `python3`、
   **不读 `.env`**，于是把它改过的人会看到"fetch 下到 benchmark_data/、eval 去别处找"
   ——报错仍然是"缺文件"（响亮），但两边对不上。
⚠ **不需要任何凭据**（公开源，请求只带 User-Agent），只要外网直连
   raw.githubusercontent.com 与 huggingface.co。
⚠ 目录不存在不用先 mkdir：prepare 会建立暂存路径，校验后才发布数据文件。
   测试依据所需文件是否存在决定 skip，不再依据顶层目录。
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
**接完一个新数据集、或看到任何低分时的第一件事**。它报四件事：
  ① 拒答率（逐字同一句 = 被 prompt 逼的指纹）  ② 拒答 × gold 在不在检索结果里
  ③ label 分布（含 JUDGE_ERROR）              ④ 逐类分化
⚠ 2026-10-03 四个数据集的接错缺陷**全是它这套手法找出来的**；判读树在脚本 docstring 里。
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
        "查询路由的三分类小样本评测（46 题，真端点，**不消耗 Smoke 配额**）",
        notes="""\
落点默认 eval/reports/runs/router-<YYYYMMDD>：`summary.json`（混淆矩阵 + 逐类 P/R/F1 + 成本）
与 `rows.jsonl`（逐题预测）。
⚠ 它**只调路由器**——不发 Add / Search、不建集合、不写真源 ⇒ 开发机上可以反复跑。
⚠ 需要 `.env` 的 `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`（缺了在跑之前直说）。
⚠ **`router.enabled` 的开与关不影响本目标**：工具是"显式要评测路由器"，
   那个键管的是**服务的默认行为**（**2026-10-14 起 `default.yaml` 里默认开**，D40）。
常用变体：`ARGS='--boundary-only --repeat 3'`（稳定性）· `ARGS='--smoke'`（端到端冒烟）。
判据与结论见 eval/reports/router-20261010.md——⚠ **别只读准确率**：那一条在有类别塌缩时
会给出一个好看的假数（退化解"永远答 DEFAULT"就有 0.47）。
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
   `ARGS='--skip-judge'`（只测检索覆盖，不叫裁判）。
口径与结论见 eval/reports/coq-multihop-20261010.md。
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
        "多跳分解能力探针：**只看问题**，看模型能不能拆出依赖链（2026-10-12）",
        notes="""\
⚠ 它**不打检索、不建集合**，只叫对话模型**拆一次/题** ⇒ 很便宜、可反复跑。
⛔ **判定不由模型做**：`judge` 那一版已作废（它至少错判 4/25，且是事实性错误——连金标几跳都
   数错）。现在是**人**逐条读候选 vs 金标，填 `eval/coq-decompose-labels.jsonl` 的 `defect`。
⚠ 结论见 eval/reports/coq-musique-20261011.md §4.6：**模型拆不清**（步数 68%、结构同构 **60%**），
   失败形态与 CoQ 循环里的"第二轮重述原问题"**同型** ⇒ 病在提示词侧不在检索侧。
变体：`ARGS='--limit 5'`（冒烟）· `ARGS='--reuse'`（**不调模型**，只拿新标签/新算法重算）。
⚠ 重判**必须**用 `--reuse`：重新摇一次模型就是换一份候选分解，而标签是逐条读着那一份填的。
⭐ 产物里的 `decompositions.md` 是**人读的那一份**（原问题 / 模型拆解 / 金标拆解并排），
   逐条复核时直接复制它——⚠ 它含数据集原文，落在 runs/（gitignored）。
""",
    )
    order_probe = _add_command(
        subparsers,
        "order-probe",
        "D25 性质探针：乱序+并发投喂 vs 顺序投喂，断言真源**逐字一致**",
        notes="""\
⚠ **它会花掉真实的 embedding 调用**（conv-26 全量 ≈ 216 块）⇒ 不进 `make test`。
⚠ 需要 8131 与 8140..8140+N-1 这些端口空着；它自己起干净服务、各自一个集合，跑完关掉。
⚠ `--processes 4`（默认 1）让 B 臂用 **4 个独立进程**共享同一套存储
   ⇒ 那一条回答的是「`uvicorn --workers N` 安不安全」——与 `--workers 1` 那条约束直接相关。
   默认 1 时它测的是 D25 本身（乱序/并发 vs 顺序）。
⚠ 跑完**不会自动清库**：`var/order-a/` `var/order-b/` 留着供比对（都在 gitignore 里）。
   Qdrant 里那两个 `memories_ab_order_*` 集合也是——要清就手动 drop。
判据与 D25 的关系见 eval/reports/ledger.md 的「D25」一节。
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
⚠ 包里那个 `haystack-corpus.jsonl` 是**前提不是附件**：只喂目标题自己那 20 段，
   证据就全在里面、检索无事可做、链式依赖自动消失（2026-10-10 栽过，见 README）。
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
⚠ 缺 `--embedder` 时 run record 的模型指纹会写"未声明"——**R1 要求记下换没换模型**，
  所以想留下可比记录就设 `TIANXIMEM_EMBED_MODEL`（或 `ARGS='--embedder ...'`）。
⚠ 冒烟用 `ARGS='--limit 3'`；**截断跑会在数据指纹的 note 里留警示**，别拿它跟全量比。
""",
    )
    eval_command.add_argument(
        "--dataset", default="locomo-refined", help="数据集（只有两个计分数据集有加载器，§12.4）"
    )

    baseline = _add_command(
        subparsers,
        "baseline",
        "跑一轮**冻结口径**的基线/对照（§13）：`--dataset` 与透传参数可覆盖",
        notes="""\
**判断"改动值不值"就用这个**：口径（跑多少题、要不要分层）由
`eval/experiments/recipes.py` 给，不手抄数字——抄错一个数的后果是分数看起来完全正常、却不可比。
⚠ 它**拒绝**与手写的 `--limit` / `--spread` 同时给（那两个只在做消融或冒烟时才用）。
⚠ 墙钟差着量级：locomo ≈44 分（**建议当唯一哨兵**）、mquake ≈4 分、
  clbench ≈3.5 小时、beam ≈1 小时、lme ≈12–20 小时。见 `eval/reports/ledger.md`。
✅ 默认按独立用户并行投喂/检索，判分单独并行；可用
  `ARGS='--ingest-workers 4 --judge-workers 4'` 控制两段（单用户批次顺序不变）。
  两项均设 1 恢复串行。⚠ 别同时跑两条不同的评测链。
⚠ 本仓噪声底是 **346 题上 ~1pt** ⇒ 几十题的小跑法读不出改动。
用法：make baseline DATASET=locomo-refined
      make baseline DATASET=clbench ARGS='--run-id clb-after-rerank --switches {…}'
""",
    )
    baseline.add_argument(
        "--dataset", default="locomo-refined", help="数据集（只有两个计分数据集有加载器，§12.4）"
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
⚠ 它**直查 Qdrant**（纯 BM25 单路），不走服务：D15 之后服务只有混合检索一种形态，
   给服务加单路查询是 roadmap 里登记过但未实现的切片（本实验不需要它）。
⚠ 前提：语料已经喂进去了（`make eval` 跑过同一批题）。
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
⚠ daemon 在 **Windows 侧**（Docker Desktop，D12）：WSL 里 docker CLI 可能因 socket
   权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
   端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
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
        "S1 判别实验（§17.1）——Smoke 跑通后第一件事",
        notes="""\
⛔ 现在**还不能跑**：这里只有占位实现，跑起来**明确失败**（Step 6 才补真判据）——
   别让它静默成功，否则会误以为某一步已经实现。
""",
    )
    _add_command(
        subparsers,
        "clean",
        "清运行时产物（保留 dataset/）",
        notes="""\
⚠ 真源是 var/tianxi.db，**不是 data/**（2026-09-24 修正：这条目标原先删的是 data/*.db，
   而没有那个目录——于是它一直静默地什么也没删）。为什么不叫 data/ 见 var/CLAUDE.md。
⚠ **tianxi.db 是 var/ 里唯一不可重建的东西**（embed_cache/ 能重建，只是要重付一遍 embedding
   的钱；Qdrant 卷是 deploy/compose.yaml 的 Docker 命名卷、不在 var/ 下，也能从真源全量重建）
   ——本命令清得了它，**跑之前想清楚**。
   刻意不清 embed_cache/：它是"暂时不用"，不是"要扔掉"。
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
