# 评测产物与旧实验归档

2026-10-10 按用户后续只重跑 `v1.0`、`v1.2` 的计划清理旧实验目录。
当前目录仅保留本说明，作为新跑批产物的落点；历史报告与成绩台账仍保留在父目录。

## 本次清理

| 范围 | 从当前工作树移出的已跟踪文件 | 原文件行数 |
| --- | ---: | ---: |
| `configs/runs/` 的旧实验配置 | 78 | 5,561 |
| 本目录的旧探针、复现脚本、补丁和代码副本 | 121 | 21,425 |
| 合计 | 199 | 26,986 |

两个旧目录共 4,785 个文件、2,014,076,434 字节，包含 4,586 个未入 Git 的历史产物。
内容已整体移动到本机 `.git/retired-runs-20261010-104344/`，并核对文件大小与已跟踪文件 SHA256。
这次腾空了工作目录；本地归档仍占磁盘，不宣称释放了这部分磁盘空间。

历史脚本与配置的版本库出处为清理前提交
`05c0c35a1b94bfba36d396806f09725dd8b86f64`。
原始 JSON/JSONL 和日志没有进入 Git，只能从本机归档或另存的备份取回。
归档是本机恢复材料，不会随 clone 分发。

## 后续重跑入口

- `eval/experiments/run.py` / `make eval`：公开数据集通用评测；选题用 `--frozen`，
  明确记录 `native` 或 `aml-v1` 输入契约。
- `eval/experiments/replay_official.py` / `make replay-official`：官方采集重放。
- `eval/experiments/recipes.py`：冻结选题口径；加载器、HTTP harness、评分器与通用诊断工具均保留。
- 服务源码和配置分别取标签 `v1.0`、`v1.2`；配置与隔离要求见
  [`configs/runs/README.md`](../../../configs/runs/README.md)。

本次清理没有启动评测，也没有更新任何历史成绩。
[`scores-20261007.md`](../scores-20261007.md) 中的旧版本列是历史功能阶段结果，
不等同于按两个发布标签同轮重跑；新的比较必须使用新 run-id、指纹和成绩记录。

新产物和一次性探针默认被 Git 忽略。需要长期复用的程序放在 `eval/experiments/` 或 `tools/`；
正式比较需要共享的配置和清单经核对后显式提交，结果结论写入父目录的报告与台账。

从仓库根目录检查通用入口，不会发起评测：

```bash
.venv/bin/python -m eval.experiments.run --help
.venv/bin/python -m eval.experiments.replay_official --help
```

## 查看历史材料

只读查看已入库文件，例如：

主线压缩后，旧提交不再由开发分支保活。若 `git show` 找不到旧 SHA，先从本机完整历史
备份导入对象（不创建分支），再执行下面的只读命令；备份不会随 clone 分发：

```bash
git bundle unbundle .git/post-v12-consolidation-20261010-105931/before-consolidation.bundle
```

```bash
git show 05c0c35:eval/reports/runs/query-router-20261008/probe.py
```

本机归档的 `metadata.json` 记录原提交、版本标签与目录映射，`files.json` 记录逐文件清单：

```text
.git/retired-runs-20261010-104344/
├── metadata.json
├── files.json
├── configs-runs/    # 移动前的 configs/runs/
└── report-runs/     # 移动前的 eval/reports/runs/，含未入 Git 的原始结果
```

历史报告中的旧路径与命令只作出处记录，不再表示当前工作树中的可运行入口。
