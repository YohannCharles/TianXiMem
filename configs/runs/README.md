# 实验配置快照

2026-10-10 起，后续计划只重跑 Git 标签 `v1.0`、`v1.2` 的版本对照。
旧 T1/A3、共同取证、记忆治理、PersonaMem 和外部基线等实验快照已从当前工作树清理。
活动 profile 仍在父目录的 `default.yaml`、`local.yaml`、`submit.yaml`。

## 后续版本对照

- 从对应标签的独立工作树取得服务源码及 `configs/`，不要用当前配置替代标签配置。
  `v1.0` 使用旧包名 `tianxi_am`，`v1.2` 使用 `tianximem`；分别按各标签的启动入口运行。
- 用同一套通用评测入口、数据版本、冻结选题、输入契约、答案/裁判和模型口径驱动两版服务。
- 两版分别使用干净的 SQLite、Qdrant 集合与匹配模型坐标的 embedding 缓存。
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
git show v1.2:configs/default.yaml
```

主线压缩后，旧 SHA 可能需要先从完整历史 bundle 导入对象；命令与本机归档的查找方法
见上述产物目录说明。历史恢复只用于取证，不重新启用旧实验计划。
