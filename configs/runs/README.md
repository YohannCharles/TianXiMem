# 实验配置快照

用户指定的 Git 标签 `v1.0` / `v1.1.1` native 对比与补题已完成，最新结果见
[`结果台账`](../../eval/reports/ledger.md)。
旧 T1/A3、共同取证、记忆治理、PersonaMem 和外部基线等实验快照已从当前工作树清理。
活动 profile 仍在父目录的 `default.yaml`、`local.yaml`、`submit.yaml`。

## 后续版本对照

- 从对应标签的独立工作树取得服务源码及 `configs/`，不要用当前配置替代标签配置。
  `v1.0` 使用旧包名 `tianxi_am`，`v1.1.1` 使用 `tianximem`；分别按各标签的启动入口运行。
- 用同一套通用评测入口、数据版本、冻结选题、输入契约、答案/裁判和模型口径驱动两版服务。
- 两版分别使用独立干净的 SQLite、Qdrant 集合；模型坐标匹配的 embedding 文本缓存可复用。
- 同时固定投喂与判分调度参数；通用入口默认有界用户并发，用户内部批次/事件顺序不变。
  两项都设为 `1` 可恢复串行；具体参数见评测入口说明。
- 跑前新建本轮配置快照，记录标签实际 SHA、配置与模型指纹；需要兼容覆盖时显式记录。
  旧报告的阶段名称与成绩不能冒充本轮按标签重跑的结果。

本目录保留为新快照的落点，`TIANXIMEM_CONFIG_DIR` 仍可指向其中某份配置。
临时快照默认被 Git 忽略；正式对照需要共享的配置经核对后用 `git add -f` 显式提交。
评测入口与产物约定见 [`eval/reports/runs/README.md`](../../eval/reports/runs/README.md)。

## 历史恢复

旧快照保存在清理前提交 `05c0c35a1b94bfba36d396806f09725dd8b86f64`。
只读查看某份历史配置，例如：

```bash
git show 05c0c35:configs/runs/cov-wide/default.yaml
git show v1.0:configs/default.yaml
git show v1.1.1:configs/default.yaml
```

主线压缩后，旧 SHA 可能需要先从完整历史 bundle 导入对象；命令与本机归档的查找方法
见上述产物目录说明。历史恢复只用于取证，不重新启用旧实验计划。
