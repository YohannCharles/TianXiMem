# 临床多跳作答与证据诊断（2026-10-07）

## 范围与冻结

从 [低分子任务清单](subtask-priorities-20261007.md) 的临床多跳开始。复用
`optimization-12h-20261007-a13-full-medmemorybench` 的逐题 Search 文本，
不重新 Add/Search、不改变排序、不更新评分阈值。该快照的临床多跳为 0/39；
当前 main 已发布 v1.2，仍需区分历史快照成绩与当前代码的新跑分。

复现目录为 `runs/clinical-multihop-20261007/`：

- `probe.py`：原始就诊文字审计与串行可续跑对照。
- `manifest.json`：全部输入指纹、源 parquet 哈希、两份提示词和固定题号。
- `source-audit.json`：源就诊文字在 Search 中的精确匹配检查。
- `results.jsonl`：每题每臂的回答、原裁判结果、模型与提示词指纹。
- `reparsed-results.jsonl` / `quote-recovery.json`：保存原响应后的字符串格式修复审计。
- `prompt-isolation.json`：与修改前提交的共享作答入口逐字对照。

快速门槛按题号哈希选六位不同患者，先选题后调用模型。答案两臂均使用
Qwen/Qwen3.5-9B、temperature=0、max_tokens=2000；裁判沿用本地归档的
上游 MedMemoryBench 多跳模板，输出预算保持原有 2000。落地后的临床子集
使用 harness 原有答案预算 1024，确认提示词与冻结候选逐字一致，再完成全部
39 题。原始批次保持原解析器；之后仅对已保存的格式错误响应重新解析，保留原记录。
本轮只验证临床多跳；Memory governance、Epistemic safety and privacy 均未测，
不能将这轮当成完整代理回归或 AML 维度成绩。

## 已确认的两个问题

**短答要求不符合该任务的格式。** 本地通用提示词要求一句短答，上游的
[固定版本 QA 要求](https://github.com/AQ-MedAI/MedMemoryBench/blob/7227bc105b84a1a9f7a75861eb9e1be3ea502882/utils/prompts_qa.py)
则要求呈现病史依据、推导步骤和综合结论。实验解释提示词只读取问题与 Search
文本，要求引用患者的相关日期、用药与测量值；不将金标答案、推理链或来源会话
标注加入提示词。

**长上下文仍有证据缺口。** 39 题标注涉及的源就诊共 120 个题—就诊组合，
19 个组合没有任何长度至少 80 字符的源轮次完整匹配到 Search，也都没有相应的
日期前缀；16/39 题至少涉及一个这样的就诊。其余 23 题的每个来源就诊都有部分文字命中。
这些只是去空白后的原文精确匹配计数，不能等同于临床节点的语义覆盖，
也不能据此认定剩下 23 题已经检索齐全。

还有标注与原文的适配疑点：首个固定题 `mmb-14-session_60_mcd_2` 的 gold
要求 ATP/紧密连接蛋白机制，而标注指定的原始就诊未出现这些词；其 session 60
原文讲的是高压项目周与连续麻辣外卖后的排便变化，不能直接当成 gold 描述的
低蛋白午餐后假急迫已被历史记录证实。这类病例需要保留不确定性，不能向 Search
补写金标机制或不存在的经历来换取判分。

## 实验结果

| 对照 | 输入与答案预算 | 正确 / 题数 | 正确率 | 口径 |
| --- | --- | ---: | ---: | --- |
| A13 历史短答 | 冻结 Search，1024 | 0/39 | 0.00% | 既有完整临床子集 |
| 本轮旧短答 | 相同 Search，2000 | 0/6 | 0.00% | 与下一行同题、同模型和预算 |
| 本轮解释提示词 | 相同 Search，2000 | 1/6 | 16.67% | 固定快速门槛 |
| 原始就诊 + 解释提示词 | 标注选来源就诊原文，2000 | 2/6 | 33.33% | 仅定位问题，不是服务收益 |
| 落地后解释提示词 | 冻结 Search，1024 | 3/39 | 7.69% | 39 题全部完成，无缺题或重复 |

快速门槛以外的 33 题中，新增正确 2 题（2/33），并非只在原先通过的病例上复用
结论。完整 39 题的基线是历史结果；只对固定六题重新执行了同预算旧提示词。
没有重答其余 MedMemoryBench 题类或其他十份数据集，不能据此重写其完整成绩。

三道新增正确题分别是：

- `mmb-3-session_60_mcd_1`：睡眠与颈动脉变化，引用夜间血压下降约 7%、IMT 1.23 mm。
- `mmb-11-session_60_mcd_1`：肝酶变化，引用既往肝损伤、MTX 20 mg/周与 ALT 52 U/L。
- `mmb-17-session_60_mcd_1`：晚睡与膝部酸胀，引用睡眠 5.5 小时、炎症指标与滑膜炎信息。

**本轮收益来自本地 QA 适配。Add/Search、向量、事实索引与排序均未改。** 原始就诊
对照仍有四题失败，说明直接补入这些来源片段不能让本模型和裁判全部通过。
后续还需区分患者事实的提取、跨就诊连接、临床机制推理及标注适配问题。

## 落地与校验

`extra_pipeline.py` 只对 native 临床多跳分类使用新提示词。答案附带输入、提示词、
答案端点/模型与实际输出预算的指纹；旧短答、变更后的输入或不同模型/预算不得
静默复用。金标与原始未检索历史不进入作答请求；采集请求不从 gold 推断题类。

原始批次出现 3 条 `JUDGE_ERROR`：一条 note 用中文闭引号，两条 note 用单引号。
修复只规范化字符串语法，再从完整 JSON 读取显式布尔；未知布尔、截断对象和 note
里的引述不能代替外层裁决。用已保存响应重解析后，三条全部恢复为 WRONG，正确数
仍为 3/39；原始错误未覆盖，也没有重新调用裁判获取另一份结论。

共享作答入口对 3,001 条冻结输入逐字比较：39 条临床提示词改变，其余 2,962 条不变。
这是输入渲染隔离核对，不能代替其他题类的实时重答回归。MedMemoryBench 的字符串
解析修复由其四类 LLM 判分共用。

验证：`test_medmemorybench_answer.py`、`test_datasets_extra.py`、`test_official_capture.py`
与 `test_harness.py` 合计 **195 passed**；Ruff、格式与 diff 空白检查通过。
覆盖实际 CLI 请求、金标隔离、旧答案续跑拒绝、其他医学题类的原提示词，以及
混用引号、伪造内层裁决、非布尔与截断响应。

同预算快速六题的答案 + 裁判平均耗时由 44.66 秒升至 57.03 秒；完整新臂平均
56.47 秒/题。这是本地答判成本，不是 Search 或 AML 请求延迟。

## 后续优先项

继续拆分临床证据与推理问题前，先按原次序诊断 MemTrapBench 数字游戏：原冻结
47 题中 41 条参考解包含阶乘，而现有答案没有使用阶乘。这个静态现象还不能区分
历史无解结论的干扰与模型解题限制；下一轮应固定官方提示词与题号，仅比较原
历史和空记忆。该对照尚未执行，不向作答提示词加入从参考解得出的运算提示。

## 复现

```bash
uv run --env-file .env python eval/reports/runs/clinical-multihop-20261007/probe.py prepare
uv run --env-file .env python eval/reports/runs/clinical-multihop-20261007/probe.py run --phase quick --arms original explain
uv run --env-file .env python eval/reports/runs/clinical-multihop-20261007/probe.py run --phase quick --arms oracle_explain
uv run --env-file .env python eval/reports/runs/clinical-multihop-20261007/probe.py run --phase clinical --arms explain_1024
uv run python eval/reports/runs/clinical-multihop-20261007/probe.py isolation
uv run python eval/reports/runs/clinical-multihop-20261007/probe.py reparse
uv run --env-file .env python eval/reports/runs/clinical-multihop-20261007/probe.py report
```

原始就诊 oracle 仅用于诊断：gold 标注选择来源就诊，提示词仍只收到这些就诊的
原始对话，不收到标准答案或推理节点。这种输入不是服务可用的策略，不计为
Search 改进成绩。
