事实抽取后的七数据集固定题集重跑（2026-10-05，运行中）

用户确认按历史相同题集重跑。这是固定样本范围的完整评测，不是原始数据全量，
也不消耗 AML Smoke/Full 配额。产品固定为 `67fb106`，回答模型与裁判模型均为
`Qwen/Qwen3.5-9B`，temperature 0、答案上限 1024，真实 HTTP Add/Search，
top_k 100、official Add 形态、邻接半径 0、开发期关闭 rerank。

按用户要求于 2026-10-05 17:05（北京时间）转入独立后台进程，串行继续运行。
保留本轮已完成的 62 条答案，续跑前核对完整检索输入一致。
后台结束后自动核对来源、更新本报告和台账，并保留未完成阶段与错误日志。
进程状态在 `var/facts-benchmark-20261005/background-control.json`，
后台日志在 `var/facts-benchmark-20261005/background.log`。

| 数据集 | 本轮完整题数 | 历史参照 | 历史完全正确精度 | 当前结果 |
| --- | ---: | --- | ---: | --- |
| CorporateBench | 250 | `base-cb-rescore-20261004` | 26.40% | 完成：49.20%，+22.80 个百分点 |
| MQuAKE-Remastered | 768 | `mqk-fixtest` | 54.04% | 完成：71.35%，+17.32 个百分点 |
| TempReason | 332 | `tr-promptfix` | 79.52% | 完成：94.58%，+15.06 个百分点 |
| MemTrapBench | 250 | `mtb-fixtest` | 42.40% | 完成：44.00%，+1.60 个百分点 |
| LoCoMo-Refined | 346 | `base-locomo0` | 76.59% | 完成：77.46%，+0.87 个百分点 |
| LongMemEval-S | 101 | `base-lme2` | 53.47% | 完成：52.48%，−0.99 个百分点 |
| MedMemoryBench zh | 388 | 旧 `base-mmb` 未完成 | 无完整基准 | 运行中 |

2026-10-05 22:44（北京时间）进度快照：开启事实取证的七份已完成六份，
共 2,047/2,435 题完整生成并判分；另完成 CorporateBench 关闭事实取证对照 250 题。
最后一份 MedMemoryBench 进行中：已检索 85/388 题、生成 81 条答案、
完成 77 条判分，暂不报完整精度；后台进程仍运行。
已完成运行的题数均齐全，MemTrapBench 有 3 条裁判 ERROR，按既定口径计入错误，
其它已完成运行未发现裁判 ERROR 标签。
六份中五份较历史提高、一份下降；LoCoMo 多答对 3 题，LongMemEval 少答对 1 题，
这些小幅差异还不能确定是事实取证造成。

后台结束后会自动更新完整报告和台账；运行中快照不是最后一份的完整结果。

CorporateBench 当前相同提示词下：关闭共同取证 44.00%（110/250），
开启共同取证 49.20%（123/250），相差 +5.20 个百分点。
标量 EM / 列表 set-F1 均分为 48.39% → 51.40%（+3.01 个百分点）。
历史 26.40% 使用旧提示词，不能把历史总涨幅全部归因于事实抽取；
其它历史比较也包含新建索引和模型波动，尚无本轮关闭事实取证对照。

七份共 2,435 题，串行运行，各候选题重新生成答案。CorporateBench 的历史
参照仍用旧答案 prompt，故另加 250 题“当前提示词、关闭事实取证”对照；该对照
与开启事实取证臂共用同一份本轮原文库和向量集合，只改变共同取证开关。
因此历史 CorporateBench 升降分不能全部归因于事实抽取，以新开关对照判断。

原文库与向量集合均独立新建，embedding 缓存从现存派生缓存作一致快照复制。
六份有历史完整记录的数据集已核对题号集合和共同原始文件字节指纹一致。
所有配置、产品/加载/回答/评分代码及 pipeline 文件指纹写入
`runs/facts-benchmark-20261005/manifest.json`，运行中变更这些代码会停止队列。
既有历史服务和运行产物保留。

执行脚本：`runs/facts-benchmark-20261005/run_suite.py`。
配置：`configs/runs/facts-benchmark-20261005/{on,off}/`。
逐题原始检索响应：`runs/facts-benchmark-20261005/searches-<on|off>-<dataset>.jsonl`。
答案/判分：`runs/facts-benchmark-20261005-<on|off>-<dataset>/<user_id>/`。
实时状态：`runs/facts-benchmark-20261005/status.json`。
服务与评测日志、原文库及最终 SQLite 快照：`var/facts-benchmark-20261005/`。

MedMemoryBench 曾因网关超时中断，本轮保持同样上下文预算，完整记录实际
完成量、错误与裁判解析失败；中断不填作 0% 精度，不把失败当作完成。
实验期间不修改产品、答案提示词和评分规则。
