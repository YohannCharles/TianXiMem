# 数据集目录整理与按需准备验证（2026-10-06）

对应 PRD §12.2 / §12.3 / §12.5 与 [D35](../../docs/decisions.md)。
本次验证数据准备、目录迁移和加载一致性。

## 结果

旧 `benchmark_data/` 已迁入 `dataset/`，按数据集分目录。下载清单、目录映射和
准备逻辑分别在 `eval/datasets/{manifest,layout,prepare}.py`。
上游评分代码和参考材料在 `.upstream/`，无法重取的 `rh*.md` 在 `.legacy/`。
未登记文件也保留在 `.legacy/unregistered/`。`.env` 的数据根已改为 `dataset`。

迁移前后，清单内 **360 项**均通过 sha256 校验；来源标识、固定版本和本地修订哈希
保持不变。另有 **7 个**旧缓存文件随迁移保留，共移动 **367 个**文件。
数据内容、题目、答案与评分口径不变；run record 的数据根地址随目录迁移改变。
已有报告和已生成的官方套件保持原样。

## 准备行为

- `make fetch-data DATASET=<名称>` 只准备指定数据集与评分依赖，缺省为 LoCoMo-Refined。
- `make eval` 在发 Add/Search 前准备所选数据集；`--offline` 禁止下载。
- 官方采集重放只准备所选家族的裁判材料，记忆继续取采集原文。
- 缺失文件先落独立 `.tmp/` 子目录，校验上游哈希、应用已声明补丁并校验本地哈希后，
  原子移入目标。失败临时字节会清理，重跑复用已校验文件。
- 本地文件哈希不符时保留原件并明确失败；loader 和 import 不联网。
- schema 处理复用现有加载器，不重复保存一份通用清洗数据。
- `dataset/CLAUDE.md` 可入库，数据与下载的上游材料被 Git 和 Docker 排除。
- `tools/fetch_benchmark_data.py` 保留为兼容命令入口，清单仅在新模块中维护。

## 验证

| 检查 | 结果 |
| --- | --- |
| 迁移前后全档 sha256 | **360 项一致** |
| `uv run pytest -q` | **1,079 passed** |
| 补充 `--patch` 缺语料回归后的准备与归档测试 | **21 passed** |
| `make lint`（mypy + 全仓 Ruff） | 通过 |
| `git diff --check` | 通过 |
| 标准库独立运行新入口及旧兼容入口（`python -S`） | 通过 |
| Git 忽略规则 | 仅 `dataset/CLAUDE.md` 出现在未跟踪项目文件中，数据与存档被忽略 |

实网下载在独立空临时目录中完成：

- AML：**8 个**材料文件，含 **7 个**官方 pipeline；下载、补丁、双哈希校验和离线复用通过。
- LoCoMo-Refined：**5 个**材料文件；加载结果与本机迁移数据逐对象相同，
  共 **10 段对话 / 1,382 道题**；再次离线准备通过。

临时验证目录已自动清理。其他数据集使用迁移字节做全档校验，加载与冻结题数由
已有回归用例验证；实网下载覆盖范围为上面列出的材料。

## 复核命令

```bash
make data-check DATASET=all
make lint
uv run pytest -q
uv run pytest -q tests/test_dataset_prepare.py tests/test_benchmark_archive.py
python3 -S -m eval.datasets.prepare --list
python3 -S tools/fetch_benchmark_data.py --list
```
