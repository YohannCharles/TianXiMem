"""代理评测（§12 / §13）。

**这一层与 `src/tianximem` 之间只有一条通道：HTTP。** harness 不得 `import`
服务内部模块——理由有两条，都不是洁癖（[`CLAUDE.md`](./CLAUDE.md)）：

1. §13 要求 B1（ReFind）在我们自己的 harness 里重跑，而 B1 只以"另一个 Add/Search
   服务"的形式存在。走进程内调用会让 B1 变成特例，两条基线不可比。
2. §13 要求主路径能通过 Smoke 契约校验——**只有打 HTTP 才碰得到契约层**。

`eval/` 是包（有 `__init__.py`）只是为了 import 稳定，不代表它可被 `src/` 依赖：
**依赖方向永远是 `eval/` → HTTP → `src/`**，反向 import 会让上面两条同时失效。
"""
