"""从 SQLite 真源**全量重建** Qdrant 派生索引。

**什么时候需要它**：正文一改，embedding 输入就改 ⇒ 索引作废。常见触发项有两类：

* 渲染模板或日期前缀变了；
* 更换 embedding 模型，维度变化时必须重建集合。

## 为什么不能靠"把 Add 重放一遍"

`Add` 是**批次级幂等**的（`applied_batches` 守卫）——同一个 `request_id` 再来一次
会被**静默跳过**，于是索引里仍是对应既有文本的向量。所以重建必须走**这条独立的路**：
读 SQLite 真源 → 用**当前的**渲染函数重新渲染 → 重新 embedding → upsert 覆盖。

## 用法

```bash
uv run python -m tools.reindex --drop          # 丢掉集合重建（最干净）
uv run python -m tools.reindex                 # 不丢，按 id upsert 覆盖
uv run python -m tools.reindex --user conv-26  # 只重建一个 user（调试用）
```

⚠ 它按**当前配置**渲染（`TIANXIMEM_PROFILE` / `TIANXIMEM_CONFIG_DIR` 决定读哪份 yaml）——
所以"重建"的含义是"把索引对齐到当前配置"。
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Final

from tianximem.common.config import load_config
from tianximem.common.render import render_pair
from tianximem.service.app import build_services

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reindex.py", description="从 SQLite 全量重建 Qdrant 索引"
    )
    parser.add_argument("--drop", action="store_true", help="先丢掉集合（最干净的重建）")
    parser.add_argument("--user", default=None, help="只重建这个 user_id")
    args = parser.parse_args(argv)

    config = load_config()
    print(
        f"配置：profile={config.profile} 集合={config.storage.qdrant.collection} "
        f"embedder={config.models.embedder} 日期前缀={config.packaging.inject_abs_time}"
    )
    services = build_services(config)
    try:
        qdrant = services.qdrant
        if args.drop and qdrant.exists():
            qdrant.drop_collection()
            print("已丢掉集合（下面按需重建）")

        # ⚠ 渲染函数与 `Add` 路径**同一个**（`render_pair` + 同一个开关）——
        #   两处若不同，"检索命中的是什么"与"模型读到的是什么"就会静默漂移。
        renderer = partial(render_pair, inject_abs_time=config.packaging.inject_abs_time)

        # `iter_pairs(user_id=None)` 取全库；按 user 分批只是让日志可读
        with services.store.read() as conn:
            pairs = services.store.iter_pairs(conn, user_id=args.user)
        if not pairs:
            print("真源里没有 QA 对——没什么可建的（Add 过语料吗？）")
            return EXIT_FAILED

        by_user: dict[str, list] = {}
        for pair in pairs:
            by_user.setdefault(pair.user_id, []).append(pair)
        total = 0
        for user_id, group in sorted(by_user.items()):
            indexed = qdrant.index_pairs(group, services.embedder, renderer=renderer)
            total += indexed
            print(f"  {user_id}: {indexed} 条")
        print(f"\n重建完成：{total} 条 / {len(by_user)} 个 user")
        print("⚠ 缓存键是**渲染后文本的哈希**：正文变了 ⇒ 旧缓存不会被命中（§7.2），这是对的。")
        return EXIT_OK
    finally:
        services.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
