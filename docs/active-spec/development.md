# 跨平台开发入口

最后核对：2026-10-15。本文只规定开发命令入口，不替代根目录 PRD。

项目的规范入口是 Python 模块，不依赖 Unix shell（命令的定义与说明都在 `src/tianximem/dev.py`）：

```text
python -m tianximem.dev help                 # 全部命令与说明（= --help；`make help` 转发到它）
python -m tianximem.dev sync | lock | format | format-check | lint | typecheck | test
python -m tianximem.dev serve                # Add/Search 服务（单进程）
python -m tianximem.dev config-check | check
python -m tianximem.dev data-check --dataset locomo-refined     # data-patch / data-fetch 同形
python -m tianximem.dev contract-check | diagnose | probe-reranker
python -m tianximem.dev router-eval | coq-eval | coq-decompose | order-probe | export-multihop
python -m tianximem.dev eval | baseline | replay-official | t1 | t2 | t2-dump
python -m tianximem.dev qdrant-up | qdrant-down | image-build | up | down | deploy-check
python -m tianximem.dev smoke | clean
```

用 `uv run` 起同一组命令：`uv run python -m tianximem.dev <cmd>`。每个子命令的 `--help`
带着它的参数、退出码与那些坑（`help` 与 `--help` 是同一份）。

多余参数**原样下传**给被调用的脚本：`python -m tianximem.dev eval --limit 3`；
`Makefile` 侧是 `make eval ARGS='--limit 3'`。`DATASET=…` / `RUN=…` / `PROCESSES=…` /
`ROUTER_OUT=…` / `COQ_OUT=…` 这些变量照旧可用（没设时用 dev.py 里的默认值——日期落点也在那里算）。

Linux、macOS、Windows 都使用同一组参数。Windows PowerShell 不需要安装 `make`；
`Makefile` 只是 Linux/macOS 上的兼容别名（每个目标一行转发），命令行为由 `tianximem.dev` 实现。

数据准备支持 `--dataset NAME`（可重复）、`--tier all`、`--offline` 和 `--dir PATH`。
路径在 Python 中通过 `pathlib` 处理，不拼接平台专用分隔符。服务与测试子进程使用
参数列表启动，不经过 shell，因此空格路径和 Windows 路径不会被重新解析。

Docker 那几条子命令（`qdrant-up` / `up` / `deploy-check` …）仍然需要 Docker Desktop 或
Docker Engine；WSL 里 docker CLI 可能因 socket 权限用不了，那就去 Windows 侧跑同一条命令
（见 `deploy/CLAUDE.md`）。
