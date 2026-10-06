# AML 输入适配

本包将公开数据构造成版本化的 Add/Search 事件计划（PRD §12.3、§13）。
入口和支持范围在 `registry.py`，运行方法见
[`docs/benchmark-data.md`](../../../docs/benchmark-data.md) 的「公开数据的 AML 输入适配」。
它模拟已采集的输入规则；逐请求原文回放仍由 `official_capture.py` 负责。

- 原始数据读取复用上级加载器；不能从已经有损渲染的文本反推结构。
- 事件中的 HTTP 输入只包含消息、用户/会话标识和 query；金标保存在独立的 `Sample.questions`。
- 更新与检查点必须按事件顺序执行，不能把未来 Add 移到早期 Search 前。
- `common.py` 中的合成时间、语料分组和各专用适配器的范围都是明确的本地近似。
  缺少对齐依据的家族显式失败，不回落为声称 AML 对齐的原生模式。
- FEVEROUS 页面池来自声明的标题清单或采集 Add，禁止从 gold evidence 补页。
- 本包不联网、不调用模型、不 import 服务或 harness；字符切分、切批和 HTTP 由 harness 执行。
- 修改规则必须更新输入版本/说明，并验证输入指纹、原始问题与金标的隔离。
