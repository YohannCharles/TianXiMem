# TianXi_AM — 常用命令入口
#
# 已经能真跑：`sync` / `lock` / `fmt` / `lint` / `test` / `data-check` / `fetch-data`
#            / `check` / `serve` / `contract-check`。
# 其余目标仍指向尚未存在的模块，先以 echo 占位——**别让它们静默成功**，
# 否则会误以为某一步已经实现。

# `check` / `serve` 需要 .env 里的密钥与路径。**uv 的 `--env-file` 负责注入**——
# 服务自己**不读** `.env`（全仓没有 dotenv/BaseSettings 的调用；`pydantic-settings`
# 虽是依赖但尚未使用）。别把 `.env` 想成"启动时自动生效"。
ENV_FILE ?= .env
SERVICE_URL ?= http://127.0.0.1:8000

.PHONY: help sync check fmt lint test qdrant-up qdrant-down serve contract-check smoke t2 clean \
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

# ── 以下仍指向上层尚未落地的模块，跑起来会明确失败 ──────────────────────
check:  ## 环境自检：Qdrant 可达 / 三段模型端点可用 / 密钥已填（含 V7 思考探针）
	uv run --env-file $(ENV_FILE) python tools/check_env.py
# ⚠ 不检查 nvidia-smi —— 模型不在本机（D14）。
# ⚠ LLM 那一项顺带就是 §17.3 V7 的探针：网关默认开思考时，归档 pipeline 的
#    `generated_answer` 里会混进推理过程，而**裁判读到的就是那个**。

qdrant-up:  ## 起 Qdrant server（必须 server 模式；单分片）
	docker compose -f deploy/compose.yaml up -d
# ⚠ daemon 在 **Windows 侧**（Docker Desktop，D12）：WSL 里 docker CLI 可能因 socket
#    权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
#    端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
# ⚠ 报 `502 Bad Gateway` 而不是「拒绝连接」= 转发在、**容器没起**——正是该跑这条命令的时候。

qdrant-down:  ## 停 Qdrant
	docker compose -f deploy/compose.yaml down

serve:  ## 起 Add/Search 服务（FastAPI + uvicorn，单进程）
	uv run --env-file $(ENV_FILE) uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
# ⚠ 必须 --workers 1：§15 的按 session 串行化用的是进程内锁，多 worker 会**静默失效**。
#    用 --factory 而不是模块级 app：模块级 app 会让"导入本模块"就要求环境变量齐备，
#    测试没法只导入工厂函数（见 src/tianxi_am/service/app.py 的 docstring）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY；缺了会在启动时响亮失败。

test:  ## 跑单元测试
	uv run pytest

contract-check:  ## §2 契约预检——打在$(SERVICE_URL)上（跑 Smoke 之前先本地过一遍）
	uv run python -m eval.smoke.preflight --base-url $(SERVICE_URL)
# 先 `make serve` 起来再跑。清单是 docs/contract.md §4，实现是 eval/smoke/preflight.py。
# ⚠ 它会往那个服务写一小份数据（user_id 带随机后缀，不与其他 run 冲突）。

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
