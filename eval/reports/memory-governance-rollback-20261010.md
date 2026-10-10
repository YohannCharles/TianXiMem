# 三项记忆治理候选回退记录（2026-10-10）

> **旧实验归档（2026-10-10）**：本页旧 `runs/` / `configs/runs/` 路径及命令仅作历史出处。
> 当前脚本、快照和原始产物已归档；恢复方式见 [归档说明](runs/README.md)。结论与成绩保留。

按用户“先回退变更”的要求，撤销本轮当前状态引句、长历史综合引句和显式遗忘抑制的产品实现。实验报告、逐题结果、冻结源码、数据库及向量缓存保留。

本轮产品候选未提交。回退以 `a3872a8` 为源码参照：仅包含本轮差异的文件使用 `git restore --source=HEAD --worktree`；同时含其他改动的文件逐段撤销。未执行 `reset --hard`、`clean`、提交、推送或历史改写，也没有主动改变暂存内容。

| 范围 | 处置 |
| --- | --- |
| `service/pipeline.py`、`common/render.py`、`store/sqlite_store.py` | 恢复到 HEAD，撤销引句投影、控制策略读取及各检索路径的抑制接线 |
| `facts/{controls,excerpts}.py`、`retrieve/{controls,excerpts}.py` | 从当前产品源码移出，完整原文件已备份 |
| `tests/test_memory_controls.py`、`tests/test_excerpts.py` | 随候选移出，完整原文件已备份 |
| `common/config.py`、`service/app.py`、`configs/default.yaml` | 仅移除 `memory_controls`、`excerpt_evidence`、`history_excerpts`、`excerpt_min_terms` 及接线 |
| 架构、配置说明及模块指南 | 撤销候选许可与当前实现声明；D36–D38 保留编号，标记为已撤销历史 |
| 方法、实现、PersonaMem 报告 | 保留原方法和结果，增加回退状态说明；实验登记表更新状态 |

保留的独立改动包括：1024 维与缓存坐标修正、现役 embedding/reranker 网关配置、memory2/memory3 对话端点及判分线程池、事实抽取的疑问句过滤与 `memory-facts-v7`。已核对回退前记录的其他修改文件，内容均保持不变；原有暂存报告也保持原暂存内容。

回退前备份：[`var/memory-governance-rollback-20261010-094242/`](../../var/memory-governance-rollback-20261010-094242/)。其中保存完整工作区差异、64 个相关/已修改文件的副本、SHA256 清单及回退范围。备份目录属于忽略的本地运行产物，没有写入 Git。

停止了本轮八个隔离实验服务，端口为 8025、8026、8027、8028、8031、8032、8041、8042。保留其数据库和产物。用户的 8000 服务未重启，仍可能载有启动时的旧源码；下次重启才会加载回退后的源码。

冻结配置 `configs/runs/memory-governance-20261009/`、`configs/runs/personamem-forgetting-20261009/` 为历史复现材料。其候选开关已从当前配置 schema 删除，不能直接用于启动当前版本；复现须配合该轮冻结源码。

验证：Ruff 检查通过；mypy 检查通过（43 个源码文件）；`git diff --check` 通过。当前产品源码、测试和活动配置中没有遗留的候选开关或模块引用。

回退相关的 17 个测试文件共 **510 passed**，覆盖配置、渲染、Add/Search HTTP 契约、邻域、打包、存储、事实取证、缓存、精排、指标、采集、隔离与幂等。全量 pytest 在 **688 passed** 后，由于 FEVEROUS 真实 Wikipedia 加载检查耗时过长而主动中断；没有完成全量测试，不能把前述通过数当作全量通过。这两次覆盖有重叠，不累加通过数。测试没有发起新的答案/裁判基准跑批。

原始实验结论见 [PersonaMem 完整结果分析](personamem-20261009.md) 和 [三项能力结果分析](memory-governance-20261009.md)。本次回退没有重跑基准，不把撤销操作解释为分数提升。
