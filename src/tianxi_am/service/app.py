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
   不同的值" ⇒ 检索结果依赖缓存状态（`embed/query_instruction.py` 的 docstring）。

## 启动方式（`uvicorn --factory`，**不要**在模块级建 app）

```bash
uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
```

**必须 `--workers 1`**（§15）：Add 的串行化用的是**进程内**锁，多 worker 会**静默失效**。
`assert_single_process()` 会拦下配置与命令行两条路上的违规——见 `common/config.py`。

刻意**没有**模块级 `app = ...`：那会让"导入本模块"就要求环境变量齐备，
测试没法只导入工厂函数。`--factory` 正好用来表达"入口是一个函数"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from tianxi_am.common.config import AppConfig, assert_single_process, load_config
from tianxi_am.embed.base import CachingEmbedder, DiskVectorCache, EmbeddingCoordinate
from tianxi_am.embed.query_instruction import QueryInstructionEmbedder
from tianxi_am.embed.qwen3_embedding import Qwen3EmbeddingEmbedder
from tianxi_am.pairing.pairing import BatchLimits
from tianxi_am.retrieve import DenseArm, EvidenceChecker, HybridRetriever, make_hybrid_params
from tianxi_am.service.errors import register_error_handlers
from tianxi_am.service.locks import SessionLocks
from tianxi_am.service.pipeline import AddPipeline, SearchPipeline
from tianxi_am.service.routes import build_router
from tianxi_am.store.qdrant_store import QdrantStore
from tianxi_am.store.sqlite_store import SqliteStore

__all__ = ["Services", "build_services", "create_app", "create_app_from_env"]


@dataclass(slots=True)
class Services:
    """装配好的对象图。测试可以直接构造它，绕开 HTTP 与网络。"""

    config: AppConfig
    store: SqliteStore
    qdrant: QdrantStore
    embedder: QueryInstructionEmbedder
    locks: SessionLocks
    add: AddPipeline
    search: SearchPipeline

    def close(self) -> None:
        """释放长生命周期资源。

        ⚠ **`SqliteStore` 没有 `close()`**——它**不持有连接**（短生命周期模型：
        连接随每次读/写操作开关，见 `store/sqlite_store.py` 的连接模型一节）。
        所以本方法只剩 Qdrant 的客户端。
        """
        self.qdrant.close()


def build_services(config: AppConfig) -> Services:
    """从配置装配全部依赖。**唯一的装配点。**

    ⚠ 这里**每一个**取值都来自 `config`——本函数里不该再出现任何字面量阈值。
    """
    store = SqliteStore.open(
        config.storage.sqlite.path,
        busy_timeout_ms=config.storage.sqlite.busy_timeout_ms,
    )
    qdrant = QdrantStore(
        url=config.storage.qdrant.url,
        collection=config.storage.qdrant.collection,
    )

    inner = Qwen3EmbeddingEmbedder(
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        model=config.models.embedder,
    )
    # ⚠ 坐标系取【解析后】的模型名（见模块 docstring 第 2 条）
    cache = DiskVectorCache(config.cache.embed.dir, EmbeddingCoordinate(model=inner.model))
    # ⚠ 顺序不能反：前缀层在缓存**之上**（模块 docstring 第 3 条）
    embedder = QueryInstructionEmbedder(
        CachingEmbedder(inner, cache),
        instruction=config.retrieval.query_instruction,
    )
    dense = DenseArm(embedder)

    locks = SessionLocks()
    return Services(
        config=config,
        store=store,
        qdrant=qdrant,
        embedder=embedder,
        locks=locks,
        add=AddPipeline(
            store=store,
            qdrant=qdrant,
            embedder=embedder,
            locks=locks,
            limits=BatchLimits(
                max_messages=config.pairing.batch_max_messages,
                max_words=config.pairing.batch_max_words,
            ),
        ),
        search=SearchPipeline(
            store=store,
            qdrant=qdrant,
            retriever=HybridRetriever(
                store=qdrant,
                dense=dense,
                params=make_hybrid_params(
                    prefetch_limit=config.retrieval.prefetch_limit,
                    weights=config.retrieval.rrf.weights,
                    rrf_k=config.retrieval.rrf.k,
                ),
            ),
            checker=EvidenceChecker(),
        ),
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
    app.state.services = services
    return app


def create_app_from_env() -> FastAPI:
    """`uvicorn --factory` 的入口：**校验进程形态** → 读配置 → 装配 → 建 app。

    ⚠ `assert_single_process()` 必须在**建对象图之前**：多 worker 下每个 worker 都会
    自己开一个 `SessionLocks`，那时再报错也已经晚了——**锁已经形同虚设**。
    """
    assert_single_process()
    return create_app(build_services(load_config()))
