# dataset/ — 本地评测数据

本目录只放评测材料；项目代码在 [`../eval/datasets/`](../eval/datasets/)，
评测执行与评分适配在 [`../eval/harness/`](../eval/harness/)（PRD §12）。

> **目录形状、准备与校验流程（命令 / 下载→校验→发布机制 / 迁移 / `--offline` · `--check` 语义）、
> 出处链与许可一处声明在 [`../docs/benchmark-data.md`](../docs/benchmark-data.md)**——本文件不重复。
> 清单的唯一来源为 [`../eval/datasets/manifest.py`](../eval/datasets/manifest.py)，
> 目录映射为 `layout.py`，准备和迁移入口为 `prepare.py`。

三条只在本目录承重的纪律：

- **一个数据集一个目录，保留上游数据字节**——schema 归一化复用已有加载器，
  不把裁判答案混入 Add，不生成一份与原数据脱节的"清洗版"。
- **只有本文件进 Git**；下载的数据、上游源码、历史存档与临时文件均被忽略。
  `.legacy/` 里的东西**不能当题库使用**，`.tmp/` 可以清理。
- **服务内部不得引入数据集文件名、下载器或网络下载**；数据根路径由 `.env` 的
  `TIANXIMEM_BENCHMARK_DIR` 指定（默认 `dataset`）。
