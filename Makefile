# TianXi_AM — 常用命令入口
#
# 已经能真跑：`sync` / `lock` / `fmt` / `lint` / `test` / `check` / `serve` /
#            `contract-check` / `qdrant-up` / `qdrant-down` / `data-check` / `fetch-data`。
# 其余目标仍指向尚未存在的模块，先以 echo 占位——**别让它们静默成功**，
# 否则会误以为某一步已经实现。Step 0 / Step 1 起逐个替换为真命令。
#
# ⚠ **`make` 不是每台机器都有**（2026-09-24 核实过两台）：
#   · Windows 侧那台 `which make` / `where make` 都找不到 —— 装一个
#     （`choco install make` 或 Git Bash 的 make 包）；装之前要手敲每个目标下面那行命令。
#   · WSL 侧那台**已装**（`sudo apt-get install make`）。
#   Makefile 始终是命令的**唯一声明处**——目标是什么，与"这台机器装没装 make"无关。
#
# ⚠ **`.env` 由 [`common/config.py`](src/tianxi_am/common/config.py) 读**（2026-09-24 起）
#   ⇒ `serve` **不需要**额外的 `--env-file`，别再加一条会与它打架的注入路径。
#   但 `tools/check_env.py` 是**独立进程**、只读 `os.environ`，所以 `check` 那一项
#   显式带 `--env-file`——它也刻意**不依赖配置层**（配置层坏了它还得能跑）。
ENV_FILE ?= .env

.PHONY: help sync check fmt lint test qdrant-up qdrant-down serve contract-check probe-reranker \
        smoke t2 clean \
        fetch-data data-check

help:  ## 列出所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── 现在就能跑 ─────────────────────────────────────────────────────────
sync:  ## 安装依赖（`--all-extras`：会装上 [local] 的本地模型栈，数 GB）
	uv sync --all-extras

lock:  ## 生成/更新 uv.lock
	uv lock

fmt:  ## 格式化
	uv run ruff format .

lint:  ## 静态检查
	uv run ruff check .

data-check:  ## 校验 benchmark_data/ 与出处清单是否逐字节一致
	python3 tools/fetch_benchmark_data.py --check

fetch-data:  ## 取回 benchmark_data/ 归档（公开源，按 commit / 哈希钉死）
	python3 tools/fetch_benchmark_data.py --fetch

check:  ## 环境自检：Qdrant 可达 / 三段模型端点可用 / 密钥已填（含 V7 思考探针）
	uv run --env-file $(ENV_FILE) python tools/check_env.py
# ⚠ 不检查 nvidia-smi —— 模型不在本机（D14）。
# ⚠ LLM 那一项顺带就是 §17.3 V7 的探针：网关默认开思考时，归档 pipeline 的
#    `generated_answer` 里会混进推理过程，而**裁判读到的就是那个**。
# 与 `contract-check` 的分工：本项只看"环境通不通"，不打契约。

qdrant-up:  ## 起 Qdrant server（必须 server 模式；单分片）
	docker compose -f deploy/compose.yaml up -d
# ⚠ daemon 在 **Windows 侧**（Docker Desktop，D12）：WSL 里 docker CLI 可能因 socket
#    权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
#    端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
# ⚠ 报 `502 Bad Gateway` 而不是「拒绝连接」= 转发在、**容器没起**——正是该跑这条命令的时候。

qdrant-down:  ## 停 Qdrant
	docker compose -f deploy/compose.yaml down

serve:  ## 起 Add/Search 服务（FastAPI + uvicorn，单进程）
	uv run uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
# ⚠ 必须 --workers 1：§15 的按 session 串行化用的是进程内锁，多 worker 会**静默失效**。
#    两条防线都会拦：`--workers N`（N>1）被 assert_single_process() 拦下，
#    TIANXI_WORKERS != 1 被配置校验拦下（见 src/tianxi_am/common/config.py）。
# ⚠ 用 --factory 而不是模块级 app：模块级 app 会让"导入本模块"就要求环境变量齐备，
#    测试没法只导入工厂函数（见 src/tianxi_am/service/app.py 的 docstring）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY 与 configs/default.yaml；
#    缺了会在启动时响亮失败（不会带着空值跑起来）。
#    开发期建议 TIANXI_PROFILE=local（换用独立的 memories_dev 集合）。

test:  ## 跑单元测试
	uv run pytest

contract-check:  ## §2 契约合规自查（打真 HTTP：本地自启服务，**不消耗 Smoke 配额**）
	uv run python eval/smoke/preflight.py
# 打真 HTTP、看真响应——契约层只有走 HTTP 才碰得到（eval/CLAUDE.md 的边界）。
# 缺省自己起一个 uvicorn（临时库 + 临时缓存 + 独立的 memories_preflight 集合），跑完关掉并 drop。
# 退出码：0 全过 · 1 有检查没过 · 2 前置条件不满足（Qdrant 或 embedding 网关不可达）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY（由 common/config.py 读）与可达的 Qdrant。
# 打已经在跑的服务：uv run python eval/smoke/preflight.py --base-url http://127.0.0.1:8000

probe-reranker:  ## 精排探针：打真网关，验连通性 + 它在链上真的起作用（不消耗 Smoke 配额）
	uv run python tools/probe_reranker.py
# 与 contract-check 的分工：那一个验**契约形状**（不关心精排好不好用），
# 本项验**精排这一环**（端点通不通、线格式对不对、名次有没有真的换掉）。
# 退出码：0 端点与线格式都正常 · 1 端点不可用或线格式不符 · 2 没配 reranker。
# ⚠ **顺序有没有变不影响退出码**——那是信息，不是判据（一个诚实但保守的 reranker
#    完全可能给出与 RRF 相同的顺序）。

t2:  ## §13 的 T2 实验：跨 session 失败归因（半天工作量）
	@echo "TODO(Step 1): 133 道 multi-session 题（含 12 道拒答题单列）"
	@echo "  → 人工分三类：没召回 / 召回但被截断 / 在里面但排序靠后"
	@false

smoke:  ## S1 判别实验（§17.1）——Smoke 跑通后第一件事
	@echo "TODO(Step 6): 加一条只有它能回答的记忆，看分数是否变化"
	@false

clean:  ## 清运行时产物（不动 benchmark_data/）
	rm -rf data/*.db data/*.db-journal data/*.db-wal data/*.db-shm
	rm -rf .pytest_cache .mypy_cache .ruff_cache
