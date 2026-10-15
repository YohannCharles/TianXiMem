# 当前配置事实源

运行时配置分为两类：

- `configs/default.yaml` 是产品 profile 的有效默认值。
- `src/tianximem/common/config/model.py` 中的 dataclass 默认值只服务于直接构造配置对象、旧实验快照和单元测试，不代表产品 profile。

因此，缺失键的旧快照可以保留兼容行为，但新部署必须从 `default.yaml` 读取完整 profile。所有影响检索路径的开关都必须在 `default.yaml`、配置解析器和测试中同时出现。

当前产品 profile 的关键值：

| 键 | 当前值 |
| --- | --- |
| `retrieval.grounded_evidence` | `true` |
| `packaging.annotate_relatives` | `true` |
| `router.enabled` | `true` |
| `agentic.enabled` | `true` |
| `agentic.mode` | `plan` |

兼容默认值不得在 Makefile、服务装配或评测脚本中重复声明。需要变更 profile 时，先修改 YAML，再更新本文件和对应回归测试。
