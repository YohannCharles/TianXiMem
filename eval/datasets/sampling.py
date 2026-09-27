"""分层抽样——**"跑一部分"时按类别按比例取题**（两个数据集共用）。

## 为什么必须有它（2026-09-26，LongMemEval 上的真实教训）

`lme_s_cleaned.json` **按 `question_type` 分块排**（实测 7 个连续块：70 个
`single-session-user` → 62 个 `multi-session` → …）。所以 `entries[:limit]`
**取到的是单一类型**——"跑一部分"会变成"**跑一类**"，而**分数看起来完全正常**。
CL-Bench 同理（按 `context_category` 排）。

这正是本项目反复要避免的那类失败：**没有报错，只是结论错**。

## 确定性

组内**等间隔**取（不随机、不用 seed）⇒ 同一份文件永远给同一批题，
所以它是**可复现的部分跑**，不是"随机抽一点"。
组内至少取 1 题 ⇒ 实际条数**可能略多于 `limit`**（组数很小时最多多几）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["stratified_sample"]


def stratified_sample(
    entries: Sequence[dict],
    limit: int,
    *,
    key: Callable[[dict], str],
) -> list[dict]:
    """按 `key(entry)` **按比例**取约 `limit` 条；`limit` 不小于总数时**原样返回**。

    `key` 由调用方给（LongMemEval 用 `question_type`，CL-Bench 用 `context_category`）
    ——**类别字段的知识留在各自的数据集模块里**，这里只做抽样。
    """
    if limit >= len(entries):
        return list(entries)
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        groups.setdefault(key(entry), []).append(index)
    picked: list[int] = []
    for indices in groups.values():
        share = max(1, round(limit * len(indices) / len(entries)))
        step = len(indices) / share
        picked.extend(indices[min(len(indices) - 1, int(i * step))] for i in range(share))
    picked.sort()  # 回到文件原顺序，让"截断"这件事本身不改变题目顺序
    return [entries[i] for i in picked]
