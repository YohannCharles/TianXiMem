"""服务入口：装配对象图 + FastAPI 实例。

## 配置从哪来

**`common.config.load_config()`**——全包**唯一**读环境变量的地方。
`service/` 自己不碰 `os.environ`，只接收它构造出来的 `AppConfig`。

## 装配顺序里有三条**不可违反**的约束

1. **索引侧与查询侧共用同一个 `Embedder` 实例**——否则两边的 `dim` 与缓存坐标系
   可能不一致，而**不一致只会表现为"检索结果很差"，不会报错**。
2. **缓存坐标系取自 `Embedder` 解析后的模型名**（不是配置里的原始字符串）——
   否则 Step 5 切模型时**旧缓存会静默命中**。
3. **查询侧的前缀层必须在缓存之上**：
   `QueryInstructionEmbedder(CachingEmbedder(...))`。反过来会出现"同一个键对应两个
   不同的值" ⇒ 检索结果依赖缓存状态（见 `embed/query_instruction.py`）。

## 启动方式（`uvicorn --factory`，**不要**在模块级建 app）

```bash
uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
```

**必须 `--workers 1`**（§15）：**放开多 worker 需要的验证没做过**
（并发写压力、`busy_timeout` 争用、多进程各自的 Qdrant 客户端）
——**别用"位置已经是请求的纯函数了"当理由去掉它**。
`assert_single_process()` 拦下配置与命令行两条路上的违规（`common/config.py`）。

刻意**没有**模块级 `app = ...`：那会让"导入本模块"就要求环境变量齐备，测试没法只导入
工厂函数——`--factory` 正好用来表达"入口是一个函数"。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from tianxi_am.common.config import (
    ENV_RERANKER_API_KEY,
    ENV_RERANKER_BASE_URL,
    AppConfig,
    assert_single_process,
    load_config,
)
from tianxi_am.common.render import template_version
from tianxi_am.common.tokens import load_counter
from tianxi_am.embed.base import (
    CachingEmbedder,
    DiskVectorCache,
    EmbeddingCoordinate,
    OpenAICompatEmbedder,
)
from tianxi_am.embed.query_instruction import QueryInstructionEmbedder
from tianxi_am.embed.qwen3_embedding import Qwen3EmbeddingEmbedder
from tianxi_am.embed.text_embedding_v4 import (
    DEFAULT_MODEL as TEXT_EMBEDDING_V4_MODEL,
)
from tianxi_am.embed.text_embedding_v4 import (
    TextEmbeddingV4Embedder,
)
from tianxi_am.observability import MetricsSink, NullMetricsSink, SnapshotMetricsSink
from tianxi_am.rank import RemoteReranker
from tianxi_am.retrieve import DenseArm, EvidenceChecker, HybridRetriever, make_hybrid_params
from tianxi_am.service.capture import CaptureMiddleware, RequestCapture
from tianxi_am.service.errors import register_error_handlers
from tianxi_am.service.pipeline import AddPipeline, SearchPipeline
from tianxi_am.service.routes import build_router
from tianxi_am.store.qdrant_store import QdrantStore
from tianxi_am.store.sqlite_store import SqliteStore

__all__ = [
    "Services",
    "build_capture",
    "build_embedder",
    "build_metrics_sink",
    "build_reranker",
    "build_services",
    "create_app",
    "create_app_from_env",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    """装配好的对象图。测试可以直接构造它，绕开 HTTP 与网络。"""

    config: AppConfig
    store: SqliteStore
    qdrant: QdrantStore
    embedder: QueryInstructionEmbedder
    add: AddPipeline
    search: SearchPipeline
    #: 请求原文采集（`capture.enabled`）。**`None` = 没开或没打开成功**——
    #: 那时 HTTP 层连中间件都不装（零开销、零行为差异）。
    capture: RequestCapture | None = None

    def close(self) -> None:
        """释放长生命周期资源。

        ⚠ **`SqliteStore` 没有 `close()`**——它**不持有连接**（短生命周期模型：
        连接随每次读/写操作开关，见 `store/sqlite_store.py` 的连接模型一节）。
        所以本方法只剩两个 HTTP 客户端：Qdrant 与 reranker。

        ⚠ reranker 从**流水线**上取（而不是在这里再存一份），免得出现"装配时接上了、
        关闭时漏掉"的两份状态——那种漏掉的表现是**连接池在退出时没释放**，
        平时完全看不出来。
        """
        self.qdrant.close()
        reranker = self.search.reranker
        if isinstance(reranker, RemoteReranker):
            reranker.close()
        if self.capture is not None:
            self.capture.close()


def build_services(config: AppConfig) -> Services:
    """从配置装配全部依赖。**唯一的装配点。**

    ⚠ 这里**每一个**取值都来自 `config`——本函数里不该再出现任何字面量阈值。
    """
    store = SqliteStore.open(
        config.storage.sqlite.path,
        busy_timeout_ms=config.storage.sqlite.busy_timeout_ms,
    )
    # ⚠ 检索参数在**装配处注入 store**：`retrieve/` 负责校验与初值（`make_hybrid_params`），
    #   `store/` 负责把它们翻成 Qdrant 语法并**执行**。漏掉这层注入（编排者自持 params、
    #   这里也不传 `hybrid=`）⇒ yaml 里的 `retrieval.prefetch_limit` / `rrf.weights`
    #   **静默无效**，两边默认值恰好相等时尤其看不出来。回归用例：
    #   `tests/test_retrieve.py::test_build_services_forwards_retrieval_params_to_the_store`。
    qdrant = QdrantStore(
        url=config.storage.qdrant.url,
        collection=config.storage.qdrant.collection,
        hybrid=make_hybrid_params(
            prefetch_limit=config.retrieval.prefetch_limit,
            weights=config.retrieval.rrf.weights,
            rrf_k=config.retrieval.rrf.k,
        ),
    )

    inner = build_embedder(config)
    # ⚠ 坐标系取【解析后】的模型名（见模块 docstring 第 2 条），**外加模板变体**：
    #   T1 的"带日期"臂改了正文 ⇒ 它的坐标必须与"不带"臂不同，否则两臂的向量会
    #   互相静默复用（`common/render.py` 的 `template_version`）。
    cache = DiskVectorCache(
        config.cache.embed.dir,
        EmbeddingCoordinate(
            model=inner.model,
            render_template=template_version(inject_abs_time=config.packaging.inject_abs_time),
        ),
    )
    # ⚠ 顺序不能反：前缀层在缓存**之上**（模块 docstring 第 3 条）
    embedder = QueryInstructionEmbedder(
        CachingEmbedder(inner, cache),
        instruction=config.retrieval.query_instruction,
    )
    dense = DenseArm(embedder)

    return Services(
        config=config,
        store=store,
        qdrant=qdrant,
        embedder=embedder,
        add=AddPipeline(
            store=store,
            qdrant=qdrant,
            embedder=embedder,
            inject_abs_time=config.packaging.inject_abs_time,
        ),
        search=SearchPipeline(
            store=store,
            qdrant=qdrant,
            retriever=HybridRetriever(store=qdrant, dense=dense),
            # ⚠ `EvidenceChecker()` **不带 instrument** ⇒ 落到 `NullCheckerInstrument`，
            #   即**每轮判定被丢弃**。**这是刻意的，不是漏接**：真正的聚合口
            #   `observability/` 尚未实现，而换成 `InMemory*` 会**无界增长**
            #   （Full run 连跑 0.5–2 天，§2.2）且那串记录没有任何读取方。
            #   ⇒ D13 的"**每轮判定必须记录**"落在 `observability/` 落地的时候，
            #   **在那之前别把它当成已满足。**
            checker=EvidenceChecker(),
            # ⚠ 分词器**在这里就加载**（不是第一次请求时才加载）：它要联网取 BPE 文件
            #   （`common/tokens.py`），把失败暴露在**启动时**而不是某个用户的请求里。
            counter=load_counter(config.budget.tokenizer),
            budget_tokens=config.budget.max_tokens,
            seed_limit=config.neighbor.expansion_seed_limit,
            radius=config.neighbor.radius,
            reranker=build_reranker(config),
            inject_abs_time=config.packaging.inject_abs_time,
            seed_placement=config.neighbor.seed_placement,
            annotate_relatives=config.packaging.annotate_relatives,
            metrics=build_metrics_sink(config),
        ),
        # ⚠ 采集**在装配时就打开**（见 `build_capture`）：写不了要在启动日志里说，
        #   而不是等跑完一整场才发现文件是空的。
        capture=build_capture(config),
    )


def build_capture(config: AppConfig) -> RequestCapture | None:
    """按配置构造**请求原文采集**。**该没有的时候就是 `None`。**

    | 条件 | 结果 | 记什么 |
    | --- | --- | --- |
    | `capture.enabled = false`（默认） | `None` | 什么都不记——**不装中间件**，零开销 |
    | 打开文件失败（路径不可写…） | `None` | **ERROR**——否则"开了却没记到"是一次静默失败 |

    ⚠ **失败不抛异常**：采集是诊断旁路，不该有能力让服务起不来（启动失败 = 没进场，
    是最坏的一类失败）。但它**必须响亮**——所以这里把 ERROR 打出来，而不是让它
    安静地记个空文件。真失败时读者能从启动日志立刻看出"采集已经关了"。

    ⚠ **打开就在启动时做**：写文件的能力是**启动期**的事实，不是运行期才出现的
    （`configs/capture` 的路径归 env，容器里是卷里的一个目录）。等到第一次 `/add`
    才发现路径写不了，那已经是"跑了一整场却什么都没记到"。
    """
    if not config.capture.enabled:
        return None
    capture = RequestCapture(
        path=config.capture.path,
        max_bytes=config.capture.max_bytes,
    )
    if not capture.open():
        return None
    logger.warning(
        "请求采集已开启：`/add` 与 `/search` 的**原文**会写进 %s"
        "（每份 %d 字节，写满自动换下一份：`.part2.jsonl`、`.part3.jsonl` …）。"
        "⚠ 它记的是官方发来的原始请求体，**没有总量上限**——跑完记得把 "
        "`capture.enabled` 关回去。",
        capture.path,
        config.capture.max_bytes,
    )
    return capture


def build_metrics_sink(config: AppConfig) -> MetricsSink:
    """按配置构造 §14 的指标出口。**该没有的时候就是 `NullMetricsSink`。**

    与 reranker **同一套口径**：没配 `TIANXI_METRICS_PATH` 就是不记，**服务照常起**
    ——缺省不是错误。这里没有"想用却没配全"那一档（一个路径不存在就是不存在），
    所以也不需要 `build_reranker` 那样的 WARNING。

    ⚠ **两臂对照时每个服务要各给一个路径**：两个进程写同一个文件会互相盖掉，
    而症状只是"计数比预期少"，不报错。
    """
    if not config.metrics_path:
        return NullMetricsSink()
    return SnapshotMetricsSink(path=Path(config.metrics_path))


def build_reranker(config: AppConfig) -> RemoteReranker | None:
    """按配置构造 reranker；**该没有的时候就是 `None`**（D12 的降级形态）。

    | 条件 | 结果 | 记什么 |
    | --- | --- | --- |
    | `rerank.enabled = false` | `None` | `rerank_disabled`——**刻意的消融**，不告警 |
    | 端点或密钥为空 | `None` | `rerank_disabled` + **WARNING**——多半是漏配了 |
    | 齐备 | `RemoteReranker` | 正常调用；失败走 `rerank_degraded` |

    ⚠ **"没配"不报错是刻意的**：reranker 是唯一不被规则保证可用的组件（D12），缺了它
    服务必须照常起——这与 `embed_base_url` 那种"缺了就拒绝启动"正相反。但**"想用却没配全"
    与"明确关掉"是两件事**，所以前者要留下一条 WARNING：否则它会表现为"每次检索都静默
    不精排"（键与初值见 config-reference §7）。

    ⚠ 构造**不打网络**：探活会把启动变成一次远程依赖，而 reranker 不可用本来就有一条
    **已实现且已测**的降级路径。
    """
    if not config.rerank.enabled:
        logger.info("rerank.enabled = false ⇒ 不构造 reranker，Search 直接用融合名次")
        return None

    missing = [
        name
        for name, value in (
            (ENV_RERANKER_BASE_URL, config.reranker_base_url),
            (ENV_RERANKER_API_KEY, config.reranker_api_key),
        )
        if not value
    ]
    if missing:
        logger.warning(
            "rerank.enabled = true，但 %s 为空 ⇒ 不构造 reranker，"
            "每次检索都会走【未精排】的 RRF 顺序（且计入 rerank_disabled）。"
            "填好 `.env` 或把 rerank.enabled 显式设为 false。",
            " 与 ".join(missing),
        )
        return None

    return RemoteReranker(
        base_url=config.reranker_base_url,
        api_key=config.reranker_api_key,
        model=config.reranker_model,
        timeout=config.rerank.timeout_seconds,
        # ⚠ **与 base_url 是同一件事的两半**：它说"对面那个网关收哪种线格式"。
        #   两个网关互斥（vLLM 要 `queries` 数组、自研封装要 `query` 字符串），
        #   而填错的表现是**每次检索都静默降级**——见 common/config.py 的同名常量。
        envelope=config.rerank.envelope,
    )


def build_embedder(config: AppConfig) -> OpenAICompatEmbedder:
    """按 `models.embedder` 挑实现。**两个实现只差模型名与批大小。**

    | `models.embedder` | 类 | 批大小 |
    | --- | --- | --- |
    | `text-embedding-v4`（提交期，§2.3） | `TextEmbeddingV4Embedder` | **10**（端点硬限） |
    | 其他（开发期的 `Qwen3-Embedding-8B`） | `Qwen3EmbeddingEmbedder` | 64 |

    ⚠ **判据是模型名本身，不是另开一个配置键**：批大小是这个模型的 **API 事实**，
    让它能与模型名各说各话（`provider: dashscope` + `model: Qwen/…`）只会造出一种
    **配得出、跑不通**的状态。漏掉这一步的表现是：yaml 写了 `text-embedding-v4`，
    服务仍用开发期的类 ⇒ **提交期第一批 `Add` 就 400**，而本地代理评测看不出来。

    ⚠ 端点与密钥**不参与**这个判断——它们在 `.env` 的 `AML_EMB_*`（那是"embedding 端点"
    这个位置的名字，**值随部署而变**：开发期 = 自建主网关，提交期 = DashScope）。
    """

    if config.models.embedder == TEXT_EMBEDDING_V4_MODEL:
        return TextEmbeddingV4Embedder(
            base_url=config.embed_base_url, api_key=config.embed_api_key
        )
    return Qwen3EmbeddingEmbedder(
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        model=config.models.embedder,
    )


def create_app(services: Services) -> FastAPI:
    """用装配好的依赖建 app——**测试用这个**，不碰环境变量。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        services.close()

    app = FastAPI(title="TianXi_AM", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(build_router(add_pipeline=services.add, search_pipeline=services.search))
    # ⚠ **不装**是常态（默认关）：那时请求路径上一个字节都没多。
    if services.capture is not None:
        app.add_middleware(CaptureMiddleware, capture=services.capture)
    app.state.services = services
    return app


def create_app_from_env() -> FastAPI:
    """`uvicorn --factory` 的入口：**校验进程形态** → 读配置 → 装配 → 建 app。

    ⚠ `assert_single_process()` 守的是：`--workers N` 是 **§15 的偏离**，而多进程会各自持一个
    SQLite 连接池、把 `busy_timeout` 的争用放大。⇒ 放开 worker 是**单列的后续**，
    **在那之前这道守卫不许绕**。
    """
    assert_single_process()
    return create_app(build_services(load_config()))
