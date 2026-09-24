"""服务入口：装配对象图 + FastAPI 实例。

## 装配顺序里有两条**不可违反**的约束

1. **索引侧与查询侧共用同一个 `Embedder` 实例**——否则两边的 `dim` 与缓存坐标系
   可能不一致，而**不一致只会表现为"检索结果很差"，不会报错**。
2. **缓存坐标系取自 `Embedder` 解析后的模型名**（不是 settings 里的原始字符串）——
   否则模型名写空时的默认值会与坐标系对不上，Step 5 切模型时**旧缓存会静默命中**。

## 启动方式（`uvicorn --factory`，**不要**在模块级建 app）

```bash
uvicorn tianxi_am.service.app:create_app_from_env --factory --workers 1
```

**必须 `--workers 1`**（§15）：Add 的串行化用的是**进程内**锁，多 worker 会**静默失效**。

刻意**没有**模块级 `app = ...`：那会让"导入本模块"就要求环境变量齐备，
测试没法只导入工厂函数。`--factory` 正好用来表达"入口是一个函数"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from tianxi_am.embed.base import CachingEmbedder, DiskVectorCache, EmbeddingCoordinate
from tianxi_am.embed.qwen3_embedding import DEFAULT_MODEL as DEFAULT_EMB_MODEL
from tianxi_am.embed.qwen3_embedding import Qwen3EmbeddingEmbedder
from tianxi_am.retrieve import DenseArm, EvidenceChecker, HybridRetriever, make_hybrid_params
from tianxi_am.service.errors import register_error_handlers
from tianxi_am.service.locks import SessionLocks
from tianxi_am.service.pipeline import AddPipeline, SearchPipeline
from tianxi_am.service.routes import build_router
from tianxi_am.service.settings import ServiceSettings
from tianxi_am.store.qdrant_store import COLLECTION_DEFAULT, QdrantStore
from tianxi_am.store.sqlite_store import SqliteStore

__all__ = ["Services", "build_services", "create_app", "create_app_from_env"]


@dataclass(slots=True)
class Services:
    """装配好的对象图。测试可以直接构造它，绕开 HTTP 与网络。"""

    settings: ServiceSettings
    store: SqliteStore
    qdrant: QdrantStore
    embedder: CachingEmbedder
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


def build_services(settings: ServiceSettings) -> Services:
    """从配置装配全部依赖。**唯一的装配点。**"""
    store = SqliteStore.open(settings.sqlite_path)
    qdrant = QdrantStore(url=settings.qdrant_url, collection=COLLECTION_DEFAULT)

    inner = Qwen3EmbeddingEmbedder(
        base_url=settings.emb_base_url,
        api_key=settings.emb_api_key,
        model=settings.emb_model or DEFAULT_EMB_MODEL,
    )
    # ⚠ 坐标系取【解析后】的模型名（见模块 docstring 第 2 条）
    cache = DiskVectorCache(settings.embed_cache_dir, EmbeddingCoordinate(model=inner.model))
    embedder = CachingEmbedder(inner, cache)
    dense = DenseArm(embedder)

    locks = SessionLocks()
    return Services(
        settings=settings,
        store=store,
        qdrant=qdrant,
        embedder=embedder,
        locks=locks,
        add=AddPipeline(store=store, qdrant=qdrant, embedder=embedder, locks=locks),
        search=SearchPipeline(
            store=store,
            qdrant=qdrant,
            retriever=HybridRetriever(store=qdrant, dense=dense, params=make_hybrid_params()),
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
    """`uvicorn --factory` 的入口：读环境变量 → 装配 → 建 app。"""
    return create_app(build_services(ServiceSettings.from_env()))
