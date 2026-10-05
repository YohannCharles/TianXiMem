# TianXiMem — 常用命令入口
#
# 已经能真跑：`sync` / `lock` / `fmt` / `lint` / `typecheck` / `test` / `check` / `serve` /
#            `contract-check` / `probe-reranker` / `eval` / `t1` / `t2` / `t2-dump` /
#            `qdrant-up` / `qdrant-down` / `data-check` / `fetch-data`。
# 其余目标仍指向尚未存在的模块，先以 echo 占位——**别让它们静默成功**，
# 否则会误以为某一步已经实现。Step 0 / Step 1 起逐个替换为真命令。
#
# ⚠ **`make` 不是每台机器都有**（2026-09-24 核实过两台）：
#   · Windows 侧那台 `which make` / `where make` 都找不到 —— 装一个
#     （`choco install make` 或 Git Bash 的 make 包）；装之前要手敲每个目标下面那行命令。
#   · WSL 侧那台**已装**（`sudo apt-get install make`）。
#   Makefile 始终是命令的**唯一声明处**——目标是什么，与"这台机器装没装 make"无关。
#
# ⚠ **`.env` 由 [`common/config.py`](src/tianximem/common/config.py) 读**（2026-09-24 起）
#   ⇒ `serve` **不需要**额外的 `--env-file`，别再加一条会与它打架的注入路径。
#   但 `tools/check_env.py` 是**独立进程**、只读 `os.environ`，所以 `check` 那一项
#   显式带 `--env-file`——它也刻意**不依赖配置层**（配置层坏了它还得能跑）。
ENV_FILE ?= .env

# `--env-file` **只在 `.env` 存在时才加**：`uv run` 碰到不存在的 env 文件会**直接报错**，
# 而取数据**不需要任何密钥**，不该卡在"还没建 `.env`"上（新机器上 `fetch-data` 甚至
# 可以排在 `cp .env.example .env` 之前）。存在时就必须带上——见 `fetch-data` 的说明。
ENV_FILE_FLAG = $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE))

# 跑评测时的数据集（`make eval`）。**只有两个计分数据集有加载器**（§12.4）。
DATASET ?= locomo-refined

# `make order-probe` 的 B 臂用几个**独立进程**（见那个目标的说明）。
# 默认 1 = 测 D25 本身；`PROCESSES=4` = 测「多 worker 到底安不安全」。
PROCESSES ?= 1

.PHONY: help sync check fmt lint typecheck test qdrant-up qdrant-down serve contract-check probe-reranker \
        order-probe image-build up down deploy-check \
        eval t1 t2 t2-dump smoke clean \
        fetch-data data-check data-patch

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

lint: typecheck  ## 静态检查（ruff 全仓 + mypy 只查 src/）
	uv run ruff check .

# ⚠ **只查 `src/`，不查 `eval/` `tests/` `tools/`**（2026-09-25 定）：后三者现在有
#   19 处存量报错（`conftest.py` 的 sys.path 形状、测试里的 float timeout 等），
#   一起纳入等于新门禁一上来就是红的，而"永远是红的门禁"与"没有门禁"是同一个东西。
#   ⇒ 先钉住**唯一进提交链路**的 `src/`；要扩到其余目录时，先清完那 19 处再改这一行。
typecheck:  ## 类型检查（mypy）
	uv run mypy src

data-check:  ## 校验 benchmark_data/ 与出处清单是否逐字节一致（**含本地修订后的哈希**）
	python3 tools/fetch_benchmark_data.py --check
# ⚠ 清单里记两套哈希：`upstream_sha256`（上游原始字节）与 `sha256`（本地归档，打完补丁）。
#    7 个 pipeline 有一处**已声明的本地修订**（上游那份跑不起来），见 docs/benchmark-data.md。

data-patch:  ## 把已声明的本地修订就地打到归档上（**不下载**，网络不通时用这个）
	python3 tools/fetch_benchmark_data.py --patch
# ⚠ 它只做清单里声明的那一处替换（`contextlib.nullcontext` 包住 `Path.open`），
#    且**幂等**——已经打过的再跑是零处改动。tests/test_benchmark_archive.py 钉住了这条。

fetch-data:  ## 取回 benchmark_data/ 归档（公开源，按 commit / 哈希钉死）
	uv run $(ENV_FILE_FLAG) python tools/fetch_benchmark_data.py --fetch
# ⚠ 它**必须读 `.env`**（2026-10-04）：`TIANXIMEM_BENCHMARK_DIR` 是数据路径的唯一入口，
#   而 `.env.example` 明写"本地开发把它指到开发数据集即可"。以前这条用裸 `python3`、
#   **不读 `.env`**，于是把它改过的人会看到"fetch 下到 benchmark_data/、eval 去别处找"
#   ——报错仍然是"缺文件"（响亮），但两边对不上。
# ⚠ **不需要任何凭据**（公开源，请求只带 User-Agent），只要外网直连
#   raw.githubusercontent.com 与 huggingface.co。
# ⚠ 目录不存在**不用先 mkdir**：`--fetch` 会在连接成功后自己建（见工具里 `_download`
#   的注释——网络不通时不留空目录，那会让 `needs_archive` 守卫从 skip 变 fail）。

check:  ## 环境自检：Qdrant 可达 / 三段模型端点可用 / 密钥已填（含 V7 思考探针）
	uv run --env-file $(ENV_FILE) python tools/check_env.py
# ⚠ 不检查 nvidia-smi —— 模型不在本机（D14）。
# ⚠ LLM 那一项顺带就是 §17.3 V7 的探针：网关默认开思考时，归档 pipeline 的
#    `generated_answer` 里会混进推理过程，而**裁判读到的就是那个**。
# 与 `contract-check` 的分工：本项只看"环境通不通"，不打契约。

qdrant-up:  ## 起 Qdrant server（必须 server 模式；单分片）
	docker compose -f deploy/compose.yaml up -d qdrant
# ⚠ **带服务名 `qdrant`**：compose 里现在还有 `app`（容器形态的那一套，见 `up`），
#    不带服务名会把检索服务也一并起起来，和 `make serve` 抢 8000。
# ⚠ daemon 在 **Windows 侧**（Docker Desktop，D12）：WSL 里 docker CLI 可能因 socket
#    权限用不了，那就去 **Windows 侧 Git Bash** 跑同一条命令（见 deploy/CLAUDE.md）。
#    端口是转发过来的，所以 WSL 里访问 localhost:6333 照样成立。
# ⚠ 报 `502 Bad Gateway` 而不是「拒绝连接」= 转发在、**容器没起**——正是该跑这条命令的时候。

qdrant-down:  ## 停 Qdrant
	docker compose -f deploy/compose.yaml stop qdrant
# ⚠ `stop` 而不是 `down`：`down` 是**按项目**拆的，会把 `app` 一起拆掉
#    （哪怕你只是想停一下 Qdrant）。要整栈下线用 `make down`。

# ── 容器形态：整栈（检索服务 + Qdrant）───────────────────────────────────
# 与上面两条的分工：`qdrant-up` 只起库、`serve` 在本机跑服务（**开发环路**）；
# 下面三条把**服务本身**也放进容器（**部署环路**，见 deploy/CLAUDE.md）。

image-build:  ## 构建检索服务镜像（上下文是仓库根，不是 deploy/）
	docker build -f deploy/Dockerfile -t tianximem:local .
# ⚠ `-f deploy/Dockerfile .` 的最后那个 `.` **不能省**：Dockerfile 要 pyproject.toml /
#    uv.lock / src/ / configs/，所以上下文必须是仓库根（deploy/.dockerignore 在根上）。
# ⚠ 镜像里【不装 [local] extra】（torch 那几个 G）——提交链路的三个模型全在远端网关。

up:  ## 起整栈（检索服务 + Qdrant，**容器形态**）
	@test -f deploy/.env || { echo "缺 deploy/.env —— 先 cp deploy/.env.example deploy/.env 并填值"; exit 1; }
	docker compose -f deploy/compose.yaml up -d --build
# ⚠ 前置检查是**故意**的：compose 自己的报错是 `env file ... not found`，
#    它不会告诉你"该从哪拷"。而缺 .env 时 compose 仍可能起容器 ⇒ 服务在容器里
#    因缺 embedding 密钥响亮失败、restart 策略再把它反复拉起来——看着像崩溃循环。

down:  ## 停整栈（**保留卷**；加 `-v` 才会删真源）
	docker compose -f deploy/compose.yaml down
# ⚠ **默认不删卷**：`tianxi_data` 里是 SQLite 真源，删了不可恢复（Qdrant 那份才可重建）。

deploy-check:  ## 对**已部署的容器**跑契约预检（真 HTTP，不消耗 Smoke 配额）
	docker compose -f deploy/compose.yaml exec -T app python eval/smoke/preflight.py --base-url http://127.0.0.1:8000
# ⚠ 预检跑在**容器里**而不是本机：它要打 `http://127.0.0.1:8000`（容器自己的网卡）。
#    从本机打是 `make contract-check` 那条路（`--base-url http://127.0.0.1:<宿主端口>`）。

serve:  ## 起 Add/Search 服务（FastAPI + uvicorn，单进程）
	uv run uvicorn tianximem.service.app:create_app_from_env --factory --workers 1
# ⚠ 必须 --workers 1。⚠ **D25 之后原因变了**：旧理由是"按 session 串行化用的是进程内锁，
#    多 worker 会静默失效"——`SessionLocks` 已删，那条不再成立。现在拒的是
#    **"放开多 worker 需要的验证一件都没做"**（并发写压力 / busy_timeout 多进程争用 /
#    各自的 Qdrant 客户端）。⇒ **不要用"反正现在安全了"当理由把它去掉。**
#    两条防线都会拦：`--workers N`（N>1）被 assert_single_process() 拦下，
#    TIANXIMEM_WORKERS != 1 被配置校验拦下（见 src/tianximem/common/config.py）。
# ⚠ 用 --factory 而不是模块级 app：模块级 app 会让"导入本模块"就要求环境变量齐备，
#    测试没法只导入工厂函数（见 src/tianximem/service/app.py 的 docstring）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY 与 configs/default.yaml；
#    缺了会在启动时响亮失败（不会带着空值跑起来）。
#    开发期建议 TIANXIMEM_PROFILE=local（换用独立的 memories_dev 集合）。

test:  ## 跑单元测试
	uv run pytest

contract-check:  ## §2 契约合规自查（打真 HTTP：本地自启服务，**不消耗 Smoke 配额**）
	uv run python eval/smoke/preflight.py
# 打真 HTTP、看真响应——契约层只有走 HTTP 才碰得到（eval/CLAUDE.md 的边界）。
# 缺省自己起一个 uvicorn（临时库 + 临时缓存 + 独立的 memories_preflight 集合），跑完关掉并 drop。
# 退出码：0 全过 · 1 有检查没过 · 2 前置条件不满足（Qdrant 或 embedding 网关不可达）。
# ⚠ 需要 .env 里的 AML_EMB_BASE_URL / AML_EMB_API_KEY（由 common/config.py 读）与可达的 Qdrant。
# 打已经在跑的服务：uv run python eval/smoke/preflight.py --base-url http://127.0.0.1:8000

diagnose:  ## 跑批诊断：把「低分」与「模型不行」分开（RUN=run-id；只读产物，不打网关）
	uv run python tools/diagnose_run.py --run-id $(RUN)
# **接完一个新数据集、或看到任何低分时的第一件事**。它报四件事：
#   ① 拒答率（逐字同一句 = 被 prompt 逼的指纹）  ② 拒答 × gold 在不在检索结果里
#   ③ label 分布（含 JUDGE_ERROR）              ④ 逐类分化
# ⚠ 2026-10-03 四个数据集的接错缺陷**全是它这套手法找出来的**；判读树在脚本 docstring 里。
# 用法：make diagnose RUN=base-mqk

probe-reranker:  ## 精排探针：打真网关，验连通性 + 它在链上真的起作用（不消耗 Smoke 配额）
	uv run python tools/probe_reranker.py
# 与 contract-check 的分工：那一个验**契约形状**（不关心精排好不好用），
# 本项验**精排这一环**（端点通不通、线格式对不对、名次有没有真的换掉）。
# 退出码：0 端点与线格式都正常 · 1 端点不可用或线格式不符 · 2 没配 reranker。
# ⚠ **顺序有没有变不影响退出码**——那是信息，不是判据（一个诚实但保守的 reranker
#    完全可能给出与 RRF 相同的顺序）。

order-probe:  ## D25 性质探针：乱序+并发投喂 vs 顺序投喂，断言真源**逐字一致**
	uv run python tools/order_probe.py --processes $(PROCESSES)
# ⚠ **它会花掉真实的 embedding 调用**（conv-26 全量 ≈ 216 块）⇒ 不进 `make test`。
# ⚠ 需要 8131 与 8140..8140+N-1 这些端口空着；它自己起干净服务、各自一个集合，跑完关掉。
# ⚠ `PROCESSES=4`（默认 1）让 B 臂用 **4 个独立进程**共享同一套存储
#    ⇒ 那一条回答的是「`uvicorn --workers N` 安不安全」——与 `--workers 1` 那条约束直接相关。
#    默认 1 时它测的是 D25 本身（乱序/并发 vs 顺序）。
# ⚠ 跑完**不会自动清库**：`var/order-a/` `var/order-b/` 留着供比对（都在 gitignore 里）。
#    Qdrant 里那两个 `memories_ab_order_*` 集合也是——要清就手动 drop。
# 判据与 D25 的关系见 eval/reports/ledger.md 的「D25」一节。

eval:  ## 跑一轮代理评测（§13）：DATASET / ARGS 可覆盖
	uv run --env-file $(ENV_FILE) python -m eval.experiments.run --dataset $(DATASET) $(ARGS)
# ⚠ **需要服务在跑**（`make serve`）——runner 打 HTTP、不 import `src/`（eval/CLAUDE.md 的边界）。
#   连不上服务时退出码 2，并提示去起服务（不是"跑失败"）。
# ⚠ `--env-file` **不是多余的**：裁判那一步起的是 subprocess，它只继承环境变量，而
#   `.env` 是 `common/config.py` 自己读的。少了它 → Add/Search 全跑完、**裁判才炸**。
#   缺哪个名字会在跑之前直说（退出码 2）。
# ⚠ 缺 `--embedder` 时 run record 的模型指纹会写"未声明"——**R1 要求记下换没换模型**，
#   所以想留下可比记录就设 `TIANXIMEM_EMBED_MODEL`（或 `ARGS='--embedder ...'`）。
# ⚠ 冒烟用 `ARGS='--limit 3'`；**截断跑会在数据指纹的 note 里留警示**，别拿它跟全量比。

baseline:  ## 跑一轮**冻结口径**的基线/对照（§13）：DATASET / ARGS 可覆盖
	uv run --env-file $(ENV_FILE) python -m eval.experiments.run --dataset $(DATASET) --frozen $(ARGS)
# **判断"改动值不值"就用这个**：口径（跑多少题、要不要分层）由
# `eval/experiments/recipes.py` 给，不手抄数字——抄错一个数的后果是分数看起来完全正常、却不可比。
# ⚠ 它**拒绝**与手写的 `--limit` / `--spread` 同时给（那两个只在做消融或冒烟时才用）。
# ⚠ 墙钟差着量级：locomo ≈44 分（**建议当唯一哨兵**）、mquake ≈4 分、
#   clbench ≈7 小时、lme ≈12–20 小时。见 `eval/reports/ledger.md`。
# ⛔ **一次只跑一条**：网关在 Cloudflare 后面（源站时限 ~125 秒 ⇒ HTTP 524），
#   并发会把 524 从"不会发生"变成"随机发生"，**而客户端超时调多大都没用**。
#   别被"本机 CPU 只用了 2%"误导——瓶颈在远端网关。
# ⚠ 本仓噪声底是 **346 题上 ~1pt** ⇒ 几十题的小跑法读不出改动。
# 用法：make baseline DATASET=locomo-refined
#       make baseline DATASET=clbench ARGS='--run-id clb-after-rerank --switches {…}'

t1:  ## §13 的 T1 实验：时间戳前缀 带/不带（**改渲染 = 重建索引**）
	uv run --env-file $(ENV_FILE) python -m eval.experiments.t1_timestamp
# 两个臂各需**独立集合**（渲染不同 ⇒ 向量不同），见该脚本顶部。

t2:  ## §13 的 T2 实验：跨 session 失败归因（汇总人工标注，机器半边见 t2-dump）
	uv run python -m eval.experiments.t2_cross_session
# 产出的是**待填的标注表**的汇总——三类由人填，脚本拒绝代填（半张表的分布更危险）。

t2-dump:  ## T2 的机器半边：133 道 multi-session 题的纯 BM25 名次 → 待填表
	uv run python tools/t2_retrieval_dump.py
# ⚠ 它**直查 Qdrant**（纯 BM25 单路），不走服务：D15 之后服务只有混合检索一种形态，
#    给服务加单路查询是 roadmap 里登记过但未实现的切片（本实验不需要它）。
# ⚠ 前提：语料已经喂进去了（`make eval` 跑过同一批题）。

smoke:  ## S1 判别实验（§17.1）——Smoke 跑通后第一件事
	@echo "TODO(Step 6): 加一条只有它能回答的记忆，看分数是否变化"
	@false

clean:  ## 清运行时产物（不动 benchmark_data/）
	rm -rf var/*.db var/*.db-journal var/*.db-wal var/*.db-shm
	rm -rf .pytest_cache .mypy_cache .ruff_cache
# ⚠ 真源是 var/tianxi.db，**不是 data/**（2026-09-24 修正：这条目标原先删的是 data/*.db，
#    而没有那个目录——于是它一直静默地什么也没删）。为什么不叫 data/ 见 var/CLAUDE.md。
# ⚠ **tianxi.db 是 var/ 里唯一不可重建的东西**（qdrant_storage/ 与 embed_cache/ 都能重建，
#    后者只是要重付一遍 embedding 的钱）——本目标清得了它，**跑之前想清楚**。
#    刻意不清 embed_cache/ 与 qdrant_storage/：它们是"暂时不用"，不是"要扔掉"。
