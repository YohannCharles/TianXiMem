# MemMachine 参考源码与学习入口

> 最后核对：2026-10-08。当前范围由项目负责人限定为取得源码，用于学习与改进 TianXiMem。

MemMachine 是采用 [Apache-2.0 许可证](https://github.com/MemMachine/MemMachine/blob/ad8ff24b0b5c73f189eab9bb7342d4655ab85ca6/LICENSE)
的开源记忆系统。参考源码位于 [`eval/baselines/memmachine/`](../eval/baselines/memmachine/)，
固定版本、完整 commit 与上游出处统一登记在 [参考实现清单](reference-implementations.md#0-代码在哪可复现)。
本地副本保持上游原样，单独 gitignored；来源与学习笔记纳入本仓版本管理。
当前没有安装环境、服务适配或评测接入，也没有本仓测得的 MemMachine 成绩。

## 获取同一份源码

本工作区已有源码。以下命令仅供新工作区复现，不安装依赖：

```bash
git clone --depth 1 --branch v0.4.0 https://github.com/MemMachine/MemMachine.git eval/baselines/memmachine
git -C eval/baselines/memmachine rev-parse HEAD
git -C eval/baselines/memmachine status --short
```

核对 HEAD 与参考实现清单中的完整 commit 一致，且工作区无改动。
后续升级参考版本时另记出处；源码查看不需要启动服务或调用模型。

## 研读顺序与改进方向

先读 [`main/memmachine.py`](../eval/baselines/memmachine/packages/server/src/memmachine_server/main/memmachine.py)
的写入与查询编排，再沿下表进入具体机制。下列方向是待验证的改进假设，尚无收益结论。

| 顺序 | 学习内容与源码入口 | 对 TianXiMem 的参考价值 |
| --- | --- | --- |
| 1 | [`episodic_memory/event_memory/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/episodic_memory/event_memory/)：事件、分段、派生向量与来源回取 | 对照长上下文及跨会话证据的组织方式，用 LoCoMo / LongMemEval 失败题区分漏召回与预算截断 |
| 2 | [`common/reranker/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/common/reranker/)：BM25 与 RRF | 研究候选重排能否提升关键证据位置；适用于多跳及事实核验中的证据选择 |
| 3 | [`episodic_memory/declarative_memory/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/episodic_memory/declarative_memory/) 与 [`retrieval_agent/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/retrieval_agent/) | 学习图关系与多轮查询，结合 MuSiQue / MedMemoryBench 判断错误是否来自缺失关系或证据；agent 按 v2 边界评估 |
| 4 | [`semantic_memory/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/semantic_memory/)：带引用的画像与更新 | 参考偏好、更正及冲突的来源组织，用 PersonaMem / HaluMem / MemTrapBench 案例检查；LLM 抽取另作 v2 设计 |
| 5 | [`episodic_memory/short_term_memory/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/episodic_memory/short_term_memory/)：工作记忆与摘要 | 检查摘要如何保留日期、否定、更正与来源 ID；生成式摘要仍按 v2 边界评估 |
| 6 | [`common/episode_store/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/common/episode_store/) 与 [`episodic_memory_manager.py`](../eval/baselines/memmachine/packages/server/src/memmachine_server/episodic_memory/episodic_memory_manager.py) | 对照持久化、隔离、生命周期与跨存储失败恢复，改进真源与派生索引的一致性 |

## 读源码时需要核对的差异

- event 路径中的 BM25 对向量召回的候选重排；不能直接等同于 TianXiMem 的全库 BM25 / dense 两路召回。
- [`server/api_v2/`](../eval/baselines/memmachine/packages/server/src/memmachine_server/server/api_v2/) 的 org / project 作用域需要与本仓 user / session 隔离契约分别对照。
- 原生 Add 的 `AddMemoriesSpec` 没有请求幂等键；`main/memmachine.py` 的 `add_episodes` 存在跨存储写入及部分失败修复待办，研读时关注重复请求与恢复边界。
- event、declarative、short-term、semantic 和 retrieval agent 是不同模块与配置路径，比较设计时记录实际读取的机制。

学习后自行实现概念并保留来源，遵守 PRD §1 的从零搭建边界。
具体改进先用固定失败案例验证；需要正式对照时，再按 [实验登记](experiments.md)
冻结配置并将成绩与逐题证据落到 [`eval/reports/`](../eval/reports/)。
