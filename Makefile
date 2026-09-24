# TianXi_AM — 常用命令入口
#
# 已经能真跑：`sync` / `lock` / `fmt` / `lint` / `test` / `data-check` / `fetch-data`。
# 其余目标仍指向尚未存在的模块，先以 echo 占位——**别让它们静默成功**，
# 否则会误以为某一步已经实现。Step 0 / Step 1 起逐个替换为真命令。

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

# ── 以下均未实现，跑起来会明确失败 ──────────────────────────────────────
check:  ## 环境自检：Qdrant 可达 / 模型端点可用 / 密钥已填
	@echo "TODO(Step 0): 未实现。需检查："
	@echo "  Qdrant server 模式可达（§6.3；Docker Desktop 已就位，见 D12）"
	@echo "  自建网关的三段模型端点可用（LLM / embed / reranker）"
	@echo "  .env 里的 key 已填（含 ANSWER_* / JUDGE_* 那一组，见 .env.example 末尾）"
	@echo "  ⚠ 不检查 nvidia-smi —— 模型不在本机（D14）"
	@false

qdrant-up:  ## 起 Qdrant server（必须 server 模式；单分片）
	@echo "TODO(Step 1): 未实现。§6.3 要求 server 模式，版本按 §7.3 钉死 ≥ v1.17.0。"
	@echo "  docker 已就位（Docker Desktop，2026-09-23；daemon 走 Windows 命名管道）"
	@echo "  ⚠ 这个 shell 里若没有 docker，去 Windows 侧 Git Bash 跑（见 deploy/CLAUDE.md）"
	@false

qdrant-down:  ## 停 Qdrant
	@echo "TODO(Step 1)"
	@false

serve:  ## 起 Add/Search 服务（FastAPI + uvicorn，单进程）
	uv run uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
# ⚠ 必须 --workers 1：§15 的按 session 串行化用的是进程内锁，多 worker 会**静默失效**。
#    用 --factory 而不是模块级 app：模块级 app 会让"导入本模块"就要求环境变量齐备，
#    测试没法只导入工厂函数（见 src/tianxi_am/service/app.py 的 docstring）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY；缺了会在启动时响亮失败。

test:  ## 跑单元测试
	uv run pytest

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
