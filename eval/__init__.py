"""代理评测。

**这一层与 `src/tianximem` 之间只有一条通道：HTTP。** harness 不得 `import`
服务内部模块——理由有两条（[`CLAUDE.md`](./CLAUDE.md)）：

1. 外部基线只以独立 Add/Search 服务的形式存在。走进程内调用会让基线变成特例，
   不同基线之间也不可比。
2. Smoke 契约校验必须经过 HTTP——**只有打 HTTP 才碰得到契约层**。

`eval/` 是包（有 `__init__.py`）只是为了 import 稳定，不代表它可被 `src/` 依赖：
**依赖方向永远是 `eval/` → HTTP → `src/`**，反向 import 会让上面两条同时失效。
"""
