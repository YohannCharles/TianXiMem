# PersonaMem-v2 检索记忆输入修复（2026-10-06）

## 问题与最终行为

原始数据集的 MCQ 路径在 `_build_personamem_items` 中注入完整原始对话，
Search 结果只写入未被归档答案函数读取的字段。因而此前成绩无法衡量 Add/Search 改进。

修复后：原始对话用于 Add；每题 Search 结果按返回顺序、已有 token 前缀预算渲染为
`retrieved_context`，适配器将其转换为归档函数读取的 `chat_history`。
模型仍收到官方问题与选项模板，选项构造、洗牌、正确选项映射和 `evaluate_mcq` 保持不变。
尾部 system 消息继续使用既有网关兼容折叠。

- 空检索只显示 `(no memories)`，不会回填原始对话或另一字段的旧记忆。
- 缺少检索字段或字段类型错误直接失败，不能静默使用完整历史。
- 答案记录标识 `personamem-search-memory-v1` 并保存逐题输入指纹。
  旧答案或变更后的输入不能按 id 静默跳过，要求新 run-id。
- 官方采集重放的 PersonaMem 开放题入口已有检索注入；该入口和其裁判未修改。
- 归档 `benchmark_data/pipeline_v2_personamem.py` 保持原样；修改的是仓库内适配器。

这是本地检索评测的答案输入口径变更，不证明 AML 线上采用相同注入方式。
此前完整历史精度不能直接与新口径比较；原七数据集结果也没有加入 PersonaMem。

## 验证

实现前手工构造检索历史，确认官方消息保留选项、排除原始历史，空检索不回填。
实现后的相关回归：`test_personamem_pipeline.py`、`test_harness.py`、
`test_datasets_extra.py`、`test_official_capture.py` 共 **182 passed**。
覆盖逐题命中隔离、检索顺序、空结果、缺失/错误类型、实际模型请求正文、
旧答案拒绝复用和同输入续跑。Ruff 检查及格式检查通过。
另跑实验 runner 回归 `test_experiments.py`：**54 passed**，合计 **236 个测试通过**。

真实答案与官方 MCQ 判分冒烟：**Qwen/Qwen3.5-9B，2/2 正确**。
两题将原始历史与检索记忆设成相反的晨间活动偏好（瑜伽 / 游泳），
模型均按检索记忆选择；发出前也检查了隐藏原始历史没有进入消息。
这是合成案例的输入链路验证，不是数据集精度，也未运行完整 Add/Search 基准。

复现（使用新环境中的答案配置，串行调用）：

```bash
uv run --env-file .env python -m eval.reports.runs.personamem-pipeline-fix-20261006.smoke
uv run pytest -q tests/test_personamem_pipeline.py tests/test_harness.py tests/test_datasets_extra.py tests/test_official_capture.py
```

冒烟脚本和原始输入、答案、官方判分及模型记录：
`runs/personamem-pipeline-fix-20261006/{smoke.py,input.jsonl,answers.jsonl,labels.jsonl,result.json}`。
脚本可复用同输入答案；变更记忆或问题需使用新的产物目录。

下一轮数据集评测使用新的 run-id，通过 `eval/experiments/run.py --dataset personamem-v2 --frozen`
驱动 HTTP Add/Search，按既有冻结题集建立新口径基线；本次未启动整轮。
