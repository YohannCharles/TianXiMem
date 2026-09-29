# TianXi_AM 检索服务镜像 —— Add / Search 两个 HTTP 端点（§2.1）
#
# 构建（**上下文必须是仓库根**，不是本目录）：
#     docker build -f deploy/Dockerfile -t tianxi-am:local .
# 或者直接用 compose（推荐）：
#     docker compose -f deploy/compose.yaml up -d --build
#
# ── 三条不可协商的性质，改这个文件时不要碰 ────────────────────────────────
#
# 1. **单进程**（§15）。`--workers 1` 写死在 ENTRYPOINT 里，不给运行时改的机会。
#    即使 D25 之后位置已进程无关，"多 worker 安全"这条**验证从没做过**
#    （并发写压力 / busy_timeout 多进程争用 / 各自的 Qdrant 客户端）。
#    ⇒ 想放开是**另一处偏离**，先验再改，别在这里顺手调。
#
# 2. **不装 `[local]` extra**。torch / transformers / sentence-transformers 是**开发期**
#    本地模型的依赖（数 GB）。提交链路的三个模型全在远端网关（§2.3 / D12），
#    镜像里装它们只会在部署时白占几个 G。
#    ⚠ **因此镜像里没有 `qwen3.5-9b`、也没有本地 cross-encoder reranker**——
#      这是设计，不是缺件。**同时这也意味着"微调本地模型"这条路在本镜像里不存在**
#      （§12.5 / §17.3 P2：数据不得用于训练）。
#
# 3. **tiktoken 的 BPE 文件烘进镜像**。`tiktoken.get_encoding("o200k_base")` 第一次调用
#    会**联网**去 openaipublic.blob.core.windows.net 取 5MB 的 BPE 文件。
#    不预取 ⇒ 服务器一出网就被墙 / 无外网时，**第一次 `/search` 会在数 token 这一步挂住**，
#    而表现是"请求超时"，不是"缺文件"（§6.4 要求必须用这个分词器，没有降级方案）。

# ══ 构建阶段 ═══════════════════════════════════════════════════════════════
# ⚠ **用 `python:3.12-slim-bookworm` + 只把 `uv` 二进制的镜像拷进来**，
#   而不是 `ghcr.io/astral-sh/uv:python3.12-…`：
#   ① 两个阶段用**同一个基础镜像** ⇒ venv 里的解释器路径逐字相同，
#      跨阶段复制 venv 不会踩"软链指到不存在的 python"（uv 的 python 镜像与官方镜像
#      布局一致，但那是个"碰巧一致"，不值得赌）；
#   ② uv 的镜像只有滚动 tag 或很旧的钉版，`/uv` 这个拷贝法能**同时钉死 uv 版本**。
#   ⚠ 版本与本机（`uv --version`）保持一致：`--frozen` 不重新解 lock，但两边差太远时
#      lock 格式的兼容性没有保证。
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

# 字节码预编译：省掉容器启动时第一次 import 的编译开销。
# `copy` 而不是 hardlink：跨阶段复制需要真实文件。
# `UV_PYTHON_DOWNLOADS=never`：基础镜像里已经有 3.12 了，**不许 uv 自己去下解释器**
#   ——那会让构建多一次外网依赖，而且下下来的路径与运行阶段对不上。
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ① 先只装依赖（**不含本仓库**）：这一层的缓存键只有 pyproject.toml + uv.lock
#    ⇒ 改业务代码不会让几层依赖重装。
COPY pyproject.toml uv.lock README.md ./
# ⚠ **不加 `--all-extras`**：那会把 `[local]` 的 torch 栈装进来（见文件头第 2 条）。
# ⚠ `--no-dev` 排除 `[dependency-groups] dev`（pytest / ruff / mypy）——运行时不跑测试。
RUN uv sync --frozen --no-dev --no-install-project

# ② 再装本仓库。
#    ⚠ **`--no-editable` 不能省**：uv 对 workspace 根项目**默认装成 editable**，
#       那会在 site-packages 里留一个指向 `/app/src` 的 `.pth`——而运行阶段**不拷 src/**
#       ⇒ 容器起来就是 `ModuleNotFoundError: No module named 'tianxi_am'`，
#       且这个错看起来像"镜像没装对包"，其实是"装成了指向源码目录的软链"。
#    ⚠ 也不能反过来"那就把 src 拷进运行阶段"——那等于镜像里有两份代码（venv 里一份、
#       /app/src 一份），改一份不改另一份时**不报错**。
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# ③ 预取 tiktoken 的 BPE 文件（见文件头第 3 条）。
#    `TIKTOKEN_CACHE_DIR` 是 tiktoken 认的环境变量：设了它，缓存就是一个**确定的**目录
#    （不设则落在 /tmp 下一个按 URL 哈希命名的目录——重启容器就没了，等于每次重下）。
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN .venv/bin/python -c "\
import tiktoken, os;\
enc = tiktoken.get_encoding('o200k_base');\
assert enc.encode('ok'), 'BPE 文件没取到';\
print('tiktoken 缓存:', os.listdir('/opt/tiktoken-cache'))"

# ══ 运行阶段 ═══════════════════════════════════════════════════════════════
# 与构建阶段同族（bookworm）同版本（3.12）⇒ venv 里的解释器软链指向的路径两边一致。
FROM python:3.12-slim-bookworm AS runtime

# `PYTHONUNBUFFERED`：日志立刻进 docker logs（**不加则 Full run 跑两天看不到进度**）。
# `PYTHONDONTWRITEBYTECODE` 不加——构建阶段已经预编译好了，运行时不该再写 .pyc。
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH"

# 非 root 运行。uid 固定 10001 ⇒ 宿主上若要做绑定挂载，权限是可预期的。
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=root:root /app/.venv /app/.venv
COPY --from=builder --chown=root:root /opt/tiktoken-cache /opt/tiktoken-cache

# 配置（阈值 / 权重 / 模型名 / 集合名）——**烘进镜像**。
# 理由：它是被评审、被 diff 的产物，不是运行时输入。换 profile 用 `TIANXI_PROFILE` 选，
# 换一整套用 `TIANXI_CONFIG_DIR` 指（两条都是环境变量，见 .env.example 的 ②层分工）。
COPY --chown=root:root configs/ /app/configs/

# 契约预检：**在容器里就能对已部署的实例跑**（打真 HTTP，不 import src/，只依赖 httpx）。
# 平台怎么验我们无从左右，但部署完自己先过一遍是免费的——Smoke 额度只有 30 次。
COPY --chown=root:root eval/__init__.py /app/eval/__init__.py
COPY --chown=root:root eval/smoke/ /app/eval/smoke/

# 真源与缓存的落点。**必须是一个卷**（见 compose.yaml）：
#   · SQLite 是**唯一不可重建物**（Qdrant 可从它全量重建，反过来不行）
#   · embedding 缓存不落盘 ⇒ 每次重启重付一遍 embedding 的钱
# ⚠ 先建目录并 chown ⇒ 命名卷首次挂载时 Docker 会**继承镜像里该目录的属主**，
#    于是 app 用户能写。目录不存在的话卷归 root，服务启动即失败（而且是写库时才炸）。
RUN mkdir -p /data/embed_cache /data/capture && chown -R app:app /data

# `TIANXI_WORKERS=1`：§15 的第二道防线（第一道是 `common/config.py` 的 validate()）。
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    TIANXI_SQLITE_PATH=/data/tianxi.db \
    TIANXI_EMBED_CACHE_DIR=/data/embed_cache \
    TIANXI_CAPTURE_PATH=/data/capture/requests.jsonl \
    TIANXI_WORKERS=1

# ⚠ 刻意**不**提供 `TIANXI_HOST` / `TIANXI_PORT` 这种变量：没有任何代码读它们
#   ⇒ 设了也不生效，而"我明明改过端口"会变成一次排查。端口与监听地址写在下面的
#   ENTRYPOINT 里，改端口要改这个文件（**改不动就说明你改错地方了**）。

USER app

EXPOSE 8000

# 探活用 `/health`（S4）：**不碰下游**（不查 Qdrant、不调模型）——它是"进程还活着"，
# 不是"依赖都通"。这样 Qdrant 短暂抖动不会让平台判我们不健康而重启容器。
# ⚠ 用 venv 里的 python + httpx，不额外装 curl。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys,httpx; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code == 200 else 1)"]

# ⚠ `--workers 1` 与 `--factory` 都写死：
#   · `--workers 1`：见文件头第 1 条。**不要改成 ${TIANXI_WORKERS}**——那就给了运行时改的口子，
#     而 `assert_single_process()` 只能拦下"真的派了子进程"的情形，构造不出来时它拦不住。
#   · `--factory`：模块级 `app = ...` 会让"导入本模块"就要求环境变量齐备（见 app.py 的 docstring）。
# ⚠ **不要加 `--reload`**：它用 multiprocessing 起子进程，会被 `assert_single_process()` 拦下
#   （两者在子进程里形状相同，故意一起拦）。
ENTRYPOINT ["uvicorn", "tianxi_am.service.app:create_app_from_env", \
            "--factory", "--workers", "1", \
            "--host", "0.0.0.0", "--port", "8000"]
