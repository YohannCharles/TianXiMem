# TianXi_AM — 常用命令入口
#
# 脚手架阶段：只有依赖管理这一步真正可跑。
# 其余目标全部指向尚未存在的模块，先以 echo 占位——**别让它们静默成功**，
# 否则会误以为某一步已经实现。Step 1 起逐个替换为真命令。

.PHONY: help sync check fmt lint test qdrant-up qdrant-down serve contract-check smoke t2 clean

help:  ## 列出所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── 现在就能跑 ─────────────────────────────────────────────────────────
sync:  ## 安装依赖（[tool.uv] package=false，不构建本仓库）
	uv sync --all-extras

lock:  ## 生成/更新 uv.lock
	uv lock

fmt:  ## 格式化
	uv run ruff format .

lint:  ## 静态检查
	uv run ruff check .

# ── 以下均未实现，跑起来会明确失败 ──────────────────────────────────────
check:  ## 环境自检：Qdrant 可达 / 密钥可用 / GPU 可见
	@echo "TODO(Step 0): 未实现。需检查：Qdrant server 模式可达（§6.3）、"
	@echo "  docker 存在、nvidia-smi 可见 L20、两个 API key 可用"
	@false

qdrant-up:  ## 起 Qdrant server（必须 server 模式；单分片）
	@echo "TODO(Step 1): 未实现。§6.3 要求 server 模式，版本按 §7.3 钉死 ≥ v1.17.0。"
	@echo "  本机当前无 docker，需先解决运行环境。"
	@false

qdrant-down:  ## 停 Qdrant
	@echo "TODO(Step 1)"
	@false

serve:  ## 起 Add/Search 服务（FastAPI + uvicorn，单进程）
	@echo "TODO(Step 1): uvicorn tianxi_am.service.app:app --workers 1"
	@echo "  必须是 --workers 1：§15 的按 session 串行化用的是进程内锁，多 worker 会失效"
	@false

test:  ## 跑单元测试
	@echo "TODO(Step 1)"; @false

contract-check:  ## §2 契约合规自查（top_k 精确计数 / 200 形状 / 字段齐全）
	@echo "TODO(Step 1)"
	@false

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
