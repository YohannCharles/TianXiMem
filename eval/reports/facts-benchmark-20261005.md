事实抽取后的七数据集固定题集重跑（完成）

结果更新于 `2026-10-05T17:32:52.108426+00:00`。产品固定为 `67fb106e77d1dec2e02bdf726623bcb587309193`，
用户确认按历史相同题集重跑；不是原始数据全量，不消耗 AML Smoke/Full 配额。
回答/裁判模型 Qwen/Qwen3.5-9B，答案 temperature 0、上限 1024，
HTTP Add/Search、top_k 100、official Add、邻接半径 0、rerank 关闭。

| 数据集 | 题数 | 历史完全正确精度 | 当前完全正确精度 | 差值 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| corporatebench | 250 | 26.40% | 49.20% | +22.80pt | 完成 |
| mquake-remastered | 768 | 54.04% | 71.35% | +17.32pt | 完成 |
| tempreason | 332 | 79.52% | 94.58% | +15.06pt | 完成 |
| memtrapbench | 250 | 42.40% | 44.00% | +1.60pt | 完成 |
| locomo-refined | 346 | 76.59% | 77.46% | +0.87pt | 完成 |
| longmemeval-s | 101 | 53.47% | 52.48% | -0.99pt | 完成 |
| medmemorybench | 388 | — | 44.33% | — | 完成 |

七份共 2,435 题，候选题在本轮重新生成答案。后台迁移保留本轮已完成的答案，
只复用本轮真实 HTTP 检索响应；续跑前核对完整评分输入一致，不复用历史基准答案。
六份有完整历史参照的数据，题号集合与共同原始文件字节指纹一致。

CorporateBench 历史答案提示词不同，历史差值不能全部归因于事实抽取。
本轮额外的 250 题开关对照使用当前相同提示词、同一原文库与向量集合：

| CorporateBench 当前开关对照 | 完全正确精度 | 标量 EM / 列表 set-F1 均分 |
| --- | ---: | ---: |
| 关闭共同取证 | 44.00% | 48.39% |
| 开启共同取证 | 49.20% | 51.40% |

各数据集裁判不同，不计算跨数据集平均值，也不把本地精度解释为 AML 榜分。
历史比较还包含独立新建索引和模型/裁判波动；小幅差异不足以单次定论。

| 本轮范围核对 | 答案 / 应有题数 | 判分条数 | 裁判错误 | 返回单条事实的题数 |
| --- | ---: | ---: | ---: | ---: |
| off corporatebench | 250/250 | 250 | 0 | 0 |
| on corporatebench | 250/250 | 250 | 0 | 10 |
| on mquake-remastered | 768/768 | 768 | 0 | 0 |
| on tempreason | 332/332 | 332 | 0 | 0 |
| on memtrapbench | 250/250 | 250 | 3 | 0 |
| on locomo-refined | 346/346 | 346 | 0 | 5 |
| on longmemeval-s | 101/101 | 101 | 0 | 0 |
| on medmemorybench | 388/388 | 388 | 0 | 0 |

核对了 1822 条派生事实的逐字原文出处与用户归属，
并核对实际返回事实的正文、用户隔离、top_k 和完成题号集合。
物理扫描覆盖不代表语义完整，返回原文的事实路径未计入单条事实题数。

自动核对或运行存在错误，详见 `var/facts-benchmark-20261005/background.log`。

产品、加载器、答案/裁判代码、pipeline 和配置指纹在
`runs/facts-benchmark-20261005/manifest.json`；运行中变更会停止队列。
独立新建原文库和向量集合，embedding 缓存作一致快照复制，既有服务与产物保留。
逐题检索、输入、答案、判分以及失败产物全部保留。

实时状态：`runs/facts-benchmark-20261005/status.json`。
来源核对：`runs/facts-benchmark-20261005/audit-summary.json`。
原文库/最终 SQLite 快照、日志：`var/facts-benchmark-20261005/`。
执行脚本：`runs/facts-benchmark-20261005/run_suite.py` 与 `supervise.py`。
