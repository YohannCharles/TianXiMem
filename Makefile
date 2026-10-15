# TianXiMem — 常用命令入口（**兼容别名**）
#
# 命令的定义、参数与那些坑**只有一个家**：`src/tianximem/dev.py`。
#   · 每个目标只有一行转发：`uv run python -m tianximem.dev <cmd> …`；
#   · `make help` = `python -m tianximem.dev help`（不再靠 grep|awk 拼表）；
#   · `<cmd> --help` 带该命令的参数、退出码与判据。
#   · 占位只剩 `smoke` 一个——它**明确失败**，别误以为某一步已经实现。
#
# ⚠ **`make` 不是每台机器都有**（2026-09-24 核实过两台）：
#   · Windows 侧那台 `which make` / `where make` 都找不到 —— 直接用模块入口
#     `uv run python -m tianximem.dev <cmd>`，见 docs/active-spec/development.md；
#   · WSL 侧那台**已装**（`sudo apt-get install make`）。
#
# 变量透传：`ARGS='…'` 原样下传；下面几个设了才转成参数（没设 = 用 dev.py 里的默认值）。
ARGS ?=
RUN_ARG = $(if $(RUN),--run-id $(RUN))
PROCESSES_ARG = $(if $(PROCESSES),--processes $(PROCESSES))
ROUTER_OUT_ARG = $(if $(ROUTER_OUT),--out $(ROUTER_OUT))
COQ_OUT_ARG = $(if $(COQ_OUT),--out $(COQ_OUT))

# 跑评测/数据准备时的数据集。**只有两个计分数据集有加载器**（§12.4）。
DATASET ?= locomo-refined
DATASET_FLAG = $(if $(filter all,$(DATASET)),--tier all,--dataset $(DATASET))

.PHONY: help sync check config-check fmt lint typecheck test qdrant-up qdrant-down serve contract-check probe-reranker \
        router-eval coq-eval coq-decompose order-probe export-multihop image-build up down deploy-check \
        eval replay-official t1 t2 t2-dump smoke clean lock \
        fetch-data data-check data-patch

help:
	@uv run python -m tianximem.dev help

# ── 开发环路 ──────────────────────────────────────────────────────────
sync:
	uv run python -m tianximem.dev sync

lock:
	uv run python -m tianximem.dev lock

fmt:
	uv run python -m tianximem.dev format

lint: typecheck
	uv run python -m tianximem.dev lint

typecheck:
	uv run python -m tianximem.dev typecheck

test:
	uv run python -m tianximem.dev test $(ARGS)

serve:
	uv run python -m tianximem.dev serve

config-check:
	uv run python -m tianximem.dev config-check

check:
	uv run python -m tianximem.dev check $(ARGS)

# ── 数据准备（D35）────────────────────────────────────────────────────
data-check:
	uv run python -m tianximem.dev data-check $(DATASET_FLAG)

data-patch:
	uv run python -m tianximem.dev data-patch $(DATASET_FLAG)

fetch-data:
	uv run python -m tianximem.dev data-fetch $(DATASET_FLAG)

# ── 契约 / 探针 / 评测 ────────────────────────────────────────────────
contract-check:
	uv run python -m tianximem.dev contract-check $(ARGS)

diagnose:
	uv run python -m tianximem.dev diagnose $(RUN_ARG) $(ARGS)

probe-reranker:
	uv run python -m tianximem.dev probe-reranker $(ARGS)

router-eval:
	uv run python -m tianximem.dev router-eval $(ROUTER_OUT_ARG) $(ARGS)

coq-eval:
	uv run python -m tianximem.dev coq-eval $(COQ_OUT_ARG) $(ARGS)

coq-decompose:
	uv run python -m tianximem.dev coq-decompose $(ARGS)

order-probe:
	uv run python -m tianximem.dev order-probe $(PROCESSES_ARG) $(ARGS)

export-multihop:
	uv run python -m tianximem.dev export-multihop $(ARGS)

eval:
	uv run python -m tianximem.dev eval --dataset $(DATASET) $(ARGS)

baseline:
	uv run python -m tianximem.dev baseline --dataset $(DATASET) $(ARGS)

replay-official:
	uv run python -m tianximem.dev replay-official $(ARGS)

t1:
	uv run python -m tianximem.dev t1 $(ARGS)

t2:
	uv run python -m tianximem.dev t2 $(ARGS)

t2-dump:
	uv run python -m tianximem.dev t2-dump $(ARGS)

# ── 容器形态（部署环路，见 deploy/CLAUDE.md）───────────────────────────
qdrant-up:
	uv run python -m tianximem.dev qdrant-up $(ARGS)

qdrant-down:
	uv run python -m tianximem.dev qdrant-down $(ARGS)

image-build:
	uv run python -m tianximem.dev image-build $(ARGS)

up:
	uv run python -m tianximem.dev up $(ARGS)

down:
	uv run python -m tianximem.dev down $(ARGS)

deploy-check:
	uv run python -m tianximem.dev deploy-check $(ARGS)

# ── 占位与清理 ────────────────────────────────────────────────────────
smoke:
	uv run python -m tianximem.dev smoke

clean:
	uv run python -m tianximem.dev clean
