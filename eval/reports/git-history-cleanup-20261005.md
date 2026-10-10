# Git 历史整理与主线收口记录（2026-10-05 / 10-07 / 10-10）

> **合并说明**：本文由原 `git-history-cleanup-20261005.md` / `git-history-cleanup-20261007.md` 2 份专题报告合并而成（2026-10-10），内容逐字保留、仅标题降一级，未改写。

## 目录

- [Git 历史整理记录（2026-10-05）](#git-历史整理记录2026-10-05)
- [Git 历史整理与 v1.2 发布记录（2026-10-07）](#git-历史整理与-v12-发布记录2026-10-07)
- [v1.2 后的改动合并为一个主线提交（2026-10-10）](#v12-后的改动合并为一个主线提交2026-10-10)

---

## Git 历史整理记录（2026-10-05）

按用户授权，只重组 `a7ed941b0000f927d7c1887c034a68f96ec55708` 之后的提交。
该基点及全部祖先对象保持原样。原 128 个提交（含 8 个 merge）整理为 12 个线性主题提交。

正式主线：`main`，跟踪 `upstream/main`；`upstream/HEAD` 是指向 `upstream/main` 的符号引用。
重组前 HEAD：`67fb106e77d1dec2e02bdf726623bcb587309193`。12 个主题提交的末提交：`3f28d7d472e9beeb70b00c04de113981c7fb3333`。
随后按用户授权追加文档和评测归档提交，详见下节。

### 整理后的提交

| 序号 | 提交 | 内容 |
| --- | --- | --- |
| 1 | `c4be04e3` | chore(repo)：统一 TianXiMem 包名与部署标识 |
| 2 | `53db5712` | refactor(repo)：清理死代码、依赖和格式债 |
| 3 | `9f0b0306` | fix(data)：修复新机器数据获取与环境配置读取 |
| 4 | `cd5a48cc` | fix(pairing)：连续 user 段只取末条参与配对 |
| 5 | `def612c5` | fix(eval)：保证数据加载、切批和运行输入完整 |
| 6 | `092bd029` | fix(eval)：修正数据集判分适配与连续分记录 |
| 7 | `0efaf76e` | feat(eval)：冻结题集口径与串行基准运行配置 |
| 8 | `50911dcb` | feat(eval)：启用经过对照验证的数据集专用回答提示词 |
| 9 | `7df44d39` | feat(tools)：增加答案诊断、Oracle 与小样本 HTTP 验证 |
| 10 | `51fd2c98` | feat(memory)：统一来源事实索引与 Add/Search 取证流程 |
| 11 | `af2f1dc1` | feat(memory)：支持中文显式事实和共同查询表达 |
| 12 | `3f28d7d4` | docs(eval)：归档基准、改进实验和失败回退记录 |

### 验证与保留范围

- 原 HEAD 与重组末提交 `3f28d7d` 的完整 Git 文件树相等；重组本身未修改既有文件内容。
- 基点之前的历史原封不动，整理范围内无 merge。
- 重组时保留原有 23 个已修改文件、未跟踪文件与真实 Git 索引；后续文档提交另行记录。
- 最终版本：1,043 个测试通过；中间提交：配对 26、评测 201、共同取证 137、中文 58 个测试通过。
- 评分单独提交通过集合精度、空集合与拒答、日期数字误解析检查；223 次 Python 语法检查通过。
- 备份 bundle 验证及 Git 对象连接检查通过。
- 后台七数据集评测保持运行；其 manifest 中的原产品 SHA 仍是有效对象，代码指纹不变。
- 实验配置、成功/失败记录、回退证据与复现脚本全部保留。
- 本轮进行中的七数据集评测文件和用户未提交文档不纳入历史重组。
- 按用户最新要求，整理后的历史替换远端旧主线 `4f74271`，当前本地分支为 `main`。
- 推送使用远端旧 SHA 的 `--force-with-lease` 保护；`upstream/main` 与 `upstream/HEAD` 指向新主线。
- 删除过渡分支 `feat/source-backed-memory`（本地及远端）；此前旧整理名称及临时分支也已清理。
- 保留旧主线的本地历史分支引用归档到独立 bundle，再移除引用，旧提交不再出现在普通分支图中。
- `a7ed941b` 及更早历史不变；不执行对象垃圾回收，原 SHA 可通过 bundle 恢复复核。

### 备份与复核

原历史最初保存为 `backup/history-before-20261005-222758`；为清理旧分支图，随后连同其他旧引用归档为 bundle 并移除分支引用。
完整迁移前备份（含所有原分支引用）：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/before-main-promotion.bundle`。
移除引用清单与迁移检查：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/main-promotion.json`。
独立 bundle：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/history-before.bundle`。
完整元数据与按文件主题对应表：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/rewrite.json`。
原工作区补丁与索引：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/working-tree.patch`、`/home/buptc/project/TianXiMem/.git/history-cleanup-20261005-222758/index-before`。

查看整理后的主线：

```bash
git log --reverse --oneline a7ed941b..HEAD
git diff --exit-code 67fb106e77d1dec2e02bdf726623bcb587309193 3f28d7d472e9beeb70b00c04de113981c7fb3333
```

### 后续文档与评测归档

用户随后授权提交此前保留在工作区中的变更，按主题独立记录：

| 提交 | 内容 |
| --- | --- |
| `09fcfa8` | 25 份维护文档：根目录及模块 CLAUDE.md、Codex 入口、facts 模块说明 |
| `c16c9d9` | 七数据集重跑的冻结配置、后台脚本、审计脚本及运行中报告 |

本历史对应表独立归档。上述提交不改产品实现或评分口径，后台评测仍按原冻结指纹运行。
文档通过差异检查，评测脚本通过 Ruff 与格式检查；原始逐题产物及运行时状态继续忽略。

### 旧提交对应表

按文件主题对应；一条旧提交混合多个主题时一对多。专用路径或失败后回退的改动，
对应共同结构和实验归档，不表示失败实现被保留在最终产品中。
原 SHA 可通过备份 bundle 恢复复核；此表未改写实验报告中的历史 SHA。

| 原提交 | 原说明 | 新提交（主题对应） |
| --- | --- | --- |
| `3fef10af` | 对齐仓库名：tianxi_am → tianximem（包 / 发行名 / 环境变量前缀 / 镜像 tag） | `c4be04e3` |
| `d3e9e3f4` | 合并 redo-rename：标识符对齐仓库名 TianXiMem（在 upstream/main 上重放） | `c4be04e3` |
| `32ef6933` | 整洁化 P1-a：格式重排 15 个文件 + 删掉 3 个已判决的探针 | `53db5712` |
| `4343269c` | 整洁化 P1-b：删掉 4 个确证死符号 + 1 个死依赖（净 −30 行，0 增） | `53db5712` |
| `b740c720` | 整洁化 P2：给「平台截断窗口 + 分词器」补一条两侧相等的门禁 | `53db5712` |
| `65eeef3e` | 整洁化 P3：18 处注释/文档漂移修正（不改任何行为） | `53db5712` |
| `e1af009e` | 整洁化 P3-b：清掉"之前是 X、现在是 Y"的沿革叙述（净 −35 行） | `53db5712` |
| `bd324b1b` | 整洁化 P3-b 补：对齐 tools/ 与 docs 的措辞（「已删除」→「不入档」） | `53db5712` |
| `22f12cba` | 整洁化 P3-c：扫掉 D28 没扫干净的下游引用（31 文件） | `53db5712` |
| `d6c30884` | 整洁化 P1-c：清掉最后 6 个文件的格式债（`ruff format --check` 现已全过） | `53db5712` |
| `90bf2523` | 合并 chore/cleanup-pass：整洁化（格式 / 死代码 / 门禁 / 文档漂移 / D28 下游引用） | `53db5712` |
| `8d79d509` | 冻结抽样口径 + 扩窗默认关闭（D31）+ 两个打死长跑批的 harness 缺陷 | `9f0b0306`, `cd5a48cc`, `def612c5`, `092bd029`, `0efaf76e`, `50911dcb`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `5c808163` | 记两条实测：网关的 Cloudflare 524 上限 + mquake 冻结口径下的第一个基线 | `0efaf76e`, `3f28d7d4` |
| `ffc6c82b` | 按配置哈希核了一遍所有基线：D31 之后有两行的口径已经不对 | `3f28d7d4` |
| `efa922d8` | 把"建基线要串行"写进「怎么跑」与 make 目标 | `9f0b0306`, `3f28d7d4` |
| `6253ed21` | LME 冻结口径 300 → 100；队列移出 personamem 与 clbench（两处降档） | `0efaf76e`, `3f28d7d4` |
| `d1b42f2d` | 修 harness 的 tiktoken 漏参：正文里字面的 `<\|endoftext\|>` 会打死整轮跑批 | `092bd029`, `50911dcb` |
| `47a7e641` | 冻结口径基线批跑完：6 个数据集落地（第 7 个被 CF 524 打死，重跑中） | `3f28d7d4` |
| `4230c99b` | 把 CL-Bench 的部分分接成 `partial_credit`；clbench 加回基线队列 | `092bd029`, `0efaf76e`, `50911dcb`, `3f28d7d4` |
| `a6828840` | 格式化：recipes.py 的 clbench 注记（ruff format） | `0efaf76e` |
| `6bbd0701` | 门禁的采样不够大：CL-Bench 真的会产出超嵌入窗口的块（立 V18，clbench 基线暂停） | `def612c5`, `3f28d7d4` |
| `192879a7` | D32：配对只取 user 段的**最后一条** —— 修掉"块超嵌入窗口"（V18） | `cd5a48cc`, `def612c5`, `51fd2c98`, `3f28d7d4` |
| `59ec622d` | 记 D32 对各数据集语料的影响面：7 个不变、3 个要重跑（且必须换干净库） | `3f28d7d4` |
| `52d81c6f` | 合并 chore/oct02-03-baselines-d31-d32：D31 扩窗默认关 + 冻结抽样口径 + 基线批 + 部分分 + D32 配对修正 | `9f0b0306`, `cd5a48cc`, `def612c5`, `092bd029`, `0efaf76e`, `50911dcb`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `42e06b0b` | 决定：不为绕开 524 压缩返回的记忆量；clbench / beam 本地放弃 | `0efaf76e`, `3f28d7d4` |
| `9cdf6a93` | 加「验证一个改动的最短路径」到 eval/experiments/CLAUDE.md | `3f28d7d4` |
| `551338de` | 修三处「接错了数据集/截断」的 harness 缺陷：memtrapbench 与 tempreason 的低分不是能力结论 | `def612c5`, `092bd029`, `50911dcb` |
| `50577105` | 台账：两处 harness 缺陷的完整归因 + LME 的 D32 干净基线 | `3f28d7d4` |
| `9c90a3a4` | 台账：tempreason 与 memtrapbench 的修复验证结果 | `3f28d7d4` |
| `54342110` | mquake 换专用答案 prompt（A/B 定版，0.3997 → 预计 0.5208） | `092bd029`, `50911dcb` |
| `422a0842` | 台账：mquake 修复验证（0.3997 → 0.5404）+ corporatebench 结构性结论 | `3f28d7d4` |
| `1a90b52f` | 合并 chore/rerun-results-clb-mmb-lme：三处 harness 修复（tempreason/memtrapbench/mquake）+ D32 干净基线 + 决策记录 | `def612c5`, `092bd029`, `0efaf76e`, `50911dcb`, `3f28d7d4` |
| `02308b57` | 新增 tools/diagnose_run.py：把「低分」与「模型不行」分开 | `9f0b0306`, `0efaf76e`, `7df44d39` |
| `db9b4a7b` | medmemorybench：修两处「判分链路丢题」+ 题号跨 persona 撞车 | `def612c5`, `092bd029`, `50911dcb` |
| `79da6a4e` | 诊断工具：自校准「证据在不在」这条判据，并修它自己的三处误读 | `0efaf76e`, `7df44d39` |
| `a375a221` | tempreason：切句不许切缩写——`F.C.` 后面那一半是**区间** | `def612c5`, `50911dcb`, `51fd2c98` |
| `288e0aa6` | 新增 tools/ab_answer_prompt.py：答案 prompt 的 A/B（只重答拒答题） | `7df44d39` |
| `6c118392` | tempreason 换专用答案 prompt（0.6386 → 0.7952）+ 两处工具假绿 | `092bd029`, `0efaf76e`, `50911dcb`, `7df44d39`, `3f28d7d4` |
| `f0a9cae3` | 收尾：把两份新工具登记进状态表，清掉因这轮而过期的说法 | `092bd029`, `50911dcb`, `7df44d39`, `3f28d7d4` |
| `ecf64e4c` | 合并 chore/audit-round2：跑批诊断工具 + medmemorybench 判分链修复 + tempreason 切句/prompt（0.5512 → 0.7952） | `9f0b0306`, `def612c5`, `092bd029`, `0efaf76e`, `50911dcb`, `7df44d39`, `51fd2c98`, `3f28d7d4` |
| `8fc2f133` | 台账更正：corporatebench 的 token 上限重量（121,824/96.6% → 118,607/99.3%） | `3f28d7d4` |
| `cd5f4f96` | D33：全局计数 / 聚合类问题 park 到 v2——理由是**本地测不了**，不是"它不重要" | `3f28d7d4` |
| `7e231746` | 修 `pairing` 之后的文档漂移：组合规则的说明停在 D24，且**说反了 S5 那条取舍** | `cd5a48cc`, `51fd2c98`, `3f28d7d4` |
| `4f742713` | 合并 chore/d33-and-pairing-docs：D33（聚合类题 park 到 v2）+ pairing 文档漂移修正 | `cd5a48cc`, `51fd2c98`, `3f28d7d4` |
| `a0a87061` | `make fetch-data` 在全新 clone 上会失败：一道"目录得先存在"的门挡住了唯一能建目录的事 | `9f0b0306`, `def612c5` |
| `96b5b869` | `make fetch-data` 不读 `.env`：把数据路径改过的人，fetch 与 eval 会指向两个地方 | `9f0b0306` |
| `4d945453` | README：「快速开始」补成一条能走通的**新机器上手**路径 | `9f0b0306`, `51fd2c98` |
| `3d71ee8a` | 合并 fix/onboarding-path：新机器上手路径（fetch 的目录守卫 / .env / README） | `9f0b0306`, `def612c5`, `51fd2c98` |
| `9065f758` | memtrapbench：少一个场景目录会**静默少题**——`glob` 对不存在的目录不报错 | `def612c5`, `50911dcb` |
| `5b00c539` | 缺数据 / 缺脚本：两条「跑之前就该失败」的出口，原先都不在 | `0efaf76e` |
| `7287028b` | 合并 fix/missing-data-paths：缺数据/缺脚本时三条出口（memtrapbench 场景守卫 / 空样本 / 裁判脚本） | `0efaf76e` |
| `efa3b56a` | CorporateBench：修复评测适配并记录记忆可回答性诊断 | `092bd029`, `50911dcb`, `7df44d39`, `3f28d7d4` |
| `693d0f47` | 设计 Add/Search 的事实索引与条件证据选择方案 | `3f28d7d4` |
| `6b6ae1ec` | 记录六小时目标并添加固定案例 HTTP 评测工具 | `7df44d39`, `3f28d7d4` |
| `e3ded348` | 冻结六小时实验基线配置与隔离索引 | `3f28d7d4` |
| `d5693b47` | 补充目标流程：手工片段预验证通过后才实现代码 | `3f28d7d4` |
| `6a005514` | 记录基线与两组手工片段预验证：拒绝丢失正文关系的截短 | `3f28d7d4` |
| `0347f5b7` | 实验：按显式会议主题与日期条件选择原文证据 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `8c82c705` | 记录条件筛选真实收益与任职事实预验证结果 | `3f28d7d4` |
| `83842d82` | 实验：Add 持久化任职事实并按个人检索独立证据 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `d4f118fb` | 记录员工事实索引的小样本收益与控制结果 | `3f28d7d4` |
| `7f9e70ac` | 记录排序和原文截取失败以及关系链预验证收益 | `3f28d7d4` |
| `2c7a9416` | Add 索引明确关系原文并按实体连接取少量记忆 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `2dd0f0b4` | Revert "Add 索引明确关系原文并按实体连接取少量记忆" | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `651820c7` | 保留关系检索退化原因和 R03 回退记录 | `3f28d7d4` |
| `35903b4a` | 保留未解析替换原文以修复精简关系证据的退化 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `96f108f3` | 记录替换原文覆盖收益与时间区间预验证 | `3f28d7d4` |
| `c67fa40b` | Add 索引明确月份区间并由 Search 选择原始时序证据 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `2e8237d4` | 记录区间检索首轮和库存片段快测结果 | `3f28d7d4` |
| `f37df37d` | 保留时间区间检索并记录额外回归与月份歧义 | `3f28d7d4` |
| `14f4ae2f` | 从 Add 已有库存陈述提取独立数量事实并供 Search 取证 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `77eabcf8` | Revert "从 Add 已有库存陈述提取独立数量事实并供 Search 取证" | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `8730b775` | 保留库存无收益和会议计数快测记录及 Git 回退证据 | `3f28d7d4` |
| `82b89dde` | 库存事实按原消息日期和物品原文顺序返回 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `0887f094` | 记录原文物品顺序的预验证和库存真实收益 | `3f28d7d4` |
| `354ed67c` | 记录工作状态原文观察的名单预验证 | `3f28d7d4` |
| `f12ade1f` | 保存有来源的工作状态观察以检索月份和季度名单证据 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `da0fcfea` | 记录工作观察增益与 LoCoMo 原片段预验证 | `3f28d7d4` |
| `08f5ea9a` | 保留失败主题与交易快测及本人动作预验证记录 | `3f28d7d4` |
| `801f036f` | Add 保存本人已参加活动的原话并在 Search 按说话人取证 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `a9b3ae4a` | 记录 LoCoMo 活动索引的真实 HTTP 增益和时间题基线 | `3f28d7d4` |
| `5c786a0e` | 保留时间口径及指代快测失败与状态报告预验证 | `3f28d7d4` |
| `5e753df2` | Add 保存本人当前状态报告并按 Search 的事件对象返回原文 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `4aec20b1` | 记录状态报告 HTTP 增益及个人指代与创业端点快测 | `3f28d7d4` |
| `3129b6f7` | 记录索引覆盖快测和 MemTrapBench 判分与算术不一致 | `3f28d7d4` |
| `30667641` | Add 绑定个人指代原话并在 Search 保留独立声明来源 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `168acc7a` | 按完整输出纠正矩阵列表截断的人工复核记录 | `3f28d7d4` |
| `98c17193` | 记录 R11 中性指代片段的小样本收益 | `3f28d7d4` |
| `e1b86700` | 按来源版本补齐旧记忆索引并阻止部分事实短路 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `2fa46588` | 记录旧库按需补索引的跨数据集 HTTP 冷启动验证 | `3f28d7d4` |
| `2fc3de33` | 默认启用已验证的有界来源取证与旧库兼容 | `0efaf76e`, `51fd2c98`, `af2f1dc1` |
| `d821e11e` | 记录金额快测失败与默认配置交付 | `3f28d7d4` |
| `374df633` | 保留合并消息中每位具名说话人的事实归属 | `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `85371128` | 对完整字面输入的 CSV 和字段提取避免旧模板干扰 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `f3b5be1f` | 记录具名归属验证与当前字面输入快测 | `3f28d7d4` |
| `7ea0e197` | 检索：启用已通过独立验证的当前字面输入分支 | `0efaf76e`, `51fd2c98`, `af2f1dc1` |
| `2cd42cb3` | 实验：记录当前字面输入验证与交友来源快测 | `3f28d7d4` |
| `60de7c2b` | 事实检索：保留独立朋友归属与志愿地点原话 | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `8808683e` | 检索：启用通过目标与控制验证的朋友事实 | `0efaf76e`, `51fd2c98`, `af2f1dc1` |
| `f0226cc3` | 实验：记录朋友事实四片段实际验证 | `3f28d7d4` |
| `fb4a5d8b` | 实验：校正当前语料下的基线比较 | `3f28d7d4` |
| `098423e4` | 实验：记录完整任职来源下的日期计数失败 | `3f28d7d4` |
| `d7d298e2` | 实验：保留市场来源顺序与收入事实快测结果 | `3f28d7d4` |
| `3bd6ecb0` | 实验：记录个人指代引用边界预验证 | `3f28d7d4` |
| `2578441b` | 事实抽取：个人指代不采信虚构或引用续行 | `51fd2c98`, `3f28d7d4` |
| `d4f7396f` | 实验：记录个人引用修复和旧题输入一致性 | `3f28d7d4` |
| `e8b9ed2f` | 实验：记录会议缺席名单的精确原文快测失败 | `3f28d7d4` |
| `9bc1daaa` | 实验：修正标题预验证并记录尾部问法成本 | `3f28d7d4` |
| `e61c0380` | 检索：识别显式会议标题尾部和参与问法 | `51fd2c98`, `3f28d7d4` |
| `5c2aa082` | 实验：记录显式标题尾部的实际上下文成本 | `3f28d7d4` |
| `311b72f8` | 实验：记录完整当前字段和名称列表的快测 | `3f28d7d4` |
| `39c8931b` | 检索：完整依赖版本与当前指标名称不取旧模板 | `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `acda384a` | docs(eval): record literal-field results and variable quick checks | `3f28d7d4` |
| `96f92693` | fix(search): retain memory for unresolved bare variables | `51fd2c98`, `3f28d7d4` |
| `1b093694` | docs(eval): record successful bare-variable HTTP verification | `3f28d7d4` |
| `60094a5c` | docs(config): describe validated current-payload retrieval guard | `0efaf76e`, `51fd2c98` |
| `1658e896` | docs(eval): summarize retained diagnostic gains and final integrity checks | `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `60f5a737` | docs(eval): verify current default configuration on legacy memories | `3f28d7d4` |
| `59b65955` | docs(eval): register six-hour targeted results in the ledger | `3f28d7d4` |
| `ff33fad4` | docs(eval): document retained Add and Search evidence flow | `3f28d7d4` |
| `7b064a8f` | docs(eval): record source ownership integrity audit | `3f28d7d4` |
| `ffeeee29` | docs(eval): close six-hour goal and preserve experiment artifacts | `3f28d7d4` |
| `3eee19ad` | docs(eval): correct CorporateBench meeting-count gain description | `3f28d7d4` |
| `e63fecd6` | eval: verify cross-domain list and count evidence generalization | `3f28d7d4` |
| `f9548a37` | feat(facts): share source-backed relation and inventory evidence | `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `cedce704` | docs(eval): record shared evidence diagnostics and regression checks | `3f28d7d4` |
| `cc496be3` | refactor: unify source-backed evidence across Add/Search | `9f0b0306`, `0efaf76e`, `51fd2c98`, `af2f1dc1`, `3f28d7d4` |
| `67fb106e` | feat: support explicit Chinese source-backed evidence | `51fd2c98`, `af2f1dc1`, `3f28d7d4` |

---

## Git 历史整理与 v1.2 发布记录（2026-10-07）

### 授权与范围

用户授权重组 `v1.1` 之后的已推送及本地提交，已开启远端 Allow force pushes，
并明确选择将最终版本提升为 `main`，在最后提交创建、推送 `v1.2`。

基点：`139e58268ef08ccf7299b89a467ad05bff5c6a5a`（`v1.1^{commit}`）。
整理前远端 `main`：`116193051f00d605055c545f409a5adc02b2f492`。
整理前本地 `main`：`a8b1544d4da01b10d7ac3bef6bbbb2f529adc1c0`。
整理前候选 HEAD：`90a0cde2183d05fac15a3f5db52e46d35046df67`。

共 29 个原提交（远端 6 个、本地 23 个）重组为 14 个线性主题提交；
随后单独追加本记录，`v1.2` 指向本记录所在的最后提交。
`candidate/v1.2-grounded-evidence` 保留在第 14 个主题提交，最终版本由 `main` 发布。

### 主题提交

| 序号 | 新提交 | 内容 |
| --- | --- | --- |
| 1 | `14d6fdc4e565` | feat(datasets): 统一数据准备并接入四个新增基准 |
| 2 | `8a2bc287d547` | feat(eval): 补齐官方金标恢复与有序重放 |
| 3 | `9a7ccdb5f4b8` | fix(eval): 使用 PersonaMem 逐题检索记忆作答 |
| 4 | `e5a44419eaea` | feat(eval): 增加版本化 AML 输入计划与逐事件续跑 |
| 5 | `855ea1f98818` | chore(eval): 归档基线配置、冻结回归脚本与评测报告 |
| 6 | `95796f041be5` | docs(repo): 对齐数据布局、评测边界与模块维护规范 |
| 7 | `9e36be2dfe3f` | fix(memory): 按来源时间处理明确更正并隔离搜索诊断 |
| 8 | `1a00a10defdd` | fix(facts): 修正通用角色标签和用户自述的来源归属 |
| 9 | `1e2135f4094e` | feat(retrieve): 完善就业句法与完整字面路径取证 |
| 10 | `d0cb189cb9b8` | fix(retrieve): 保留无法与更正比序的未知时间声明 |
| 11 | `a7474660bb4c` | fix(eval): 统一 JSONL 读写并修复转义与分行 |
| 12 | `780b59ca1a9f` | refactor(repo): 收敛公共工具并清理无用实现 |
| 13 | `ad0f0bce8e55` | docs(repo): 同步当前规范并收敛重复说明 |
| 14 | `b1511e079092` | docs(eval): 归档三版本成绩与候选验证边界 |

评测配置、复现脚本、阶段过程和最终报告集中归档；
独立验证及回退需要的四个共同取证提交保持原顺序。
混合清扫提交按内容拆分：JSONL 修复进入 `fix(eval)`，
工具收口与死代码清理进入 `refactor(repo)`。
提交标题统一使用 `type(scope): 中文说明`，原作者、作者日期与已有共同署名保留。

### 文件树与验证

- 第 6 个主题提交的树与整理前本地 `main` 完全相等：
  `c941988cdcb703ddad086b9e22be5d1084a87a40`。
- 第 14 个主题提交的完整树与整理前候选 HEAD 完全相等：
  `027db3e95a233e947ef617798f232f597070f164`。
- 最后提交只新增本记录；既有代码、测试、配置与报告内容保持原字节。
- 四个共同取证主题提交逐个复用原提交的完整树，保留独立回退状态。
- 文档归并涉及的 Python 文件去除文档字符串后 AST 一致。
- 整理前等价源码的全仓检查：1248 项通过，无失败、错误或跳过。
- Ruff 全仓检查通过；mypy 检查 42 个源文件通过。
- 213 个已跟踪 Python 文件通过 AST 语法检查；
  12 个实际 CLI 的 `--help` 检查通过。
- 14 个主题提交的差异检查、Git 对象连接检查与备份 bundle 验证通过。

中间提交在独立 worktree 验证，测试使用现有本地数据。
首次缺数据导致的隔离验证日志另存为 `*.without-data`，接入现有材料后重新验证。

| 主题序号 | 新提交 | 相关测试 |
| --- | --- | --- |
| 1 | `14d6fdc4e565` | 71 passed in 3.26s |
| 2 | `8a2bc287d547` | 13 passed, 1 skipped in 0.23s |
| 3 | `9a7ccdb5f4b8` | 9 passed in 0.06s |
| 4 | `e5a44419eaea` | 108 passed, 1 skipped in 279.90s (0:04:39) |
| 7 | `9e36be2dfe3f` | 76 passed in 0.58s |
| 8 | `1a00a10defdd` | 168 passed in 0.86s |
| 9 | `1e2135f4094e` | 125 passed in 0.56s |
| 10 | `d0cb189cb9b8` | 122 passed in 0.64s |
| 11 | `a7474660bb4c` | 117 passed, 1 skipped in 17.63s |
| 12 | `780b59ca1a9f` | 187 passed in 20.02s |

第 5、6 个主题是归档和维护说明，结合文件树相等核验；
第 13、14 个主题结合 AST 与最终完整文件树核验。
全部验证日志与机器可读结果保存在下述备份目录。

### 发布与追溯

既有标签的对象和目标提交保持原样：

| 标签 | 标签对象 | 目标提交 |
| --- | --- | --- |
| v1.0 | `ac8485bfeed69902104ffc3566d8df0c21b97c1d` | `a7ed941b0000f927d7c1887c034a68f96ec55708` |
| v1.1 | `a62d94a62ff0ec9e0cd5291ee0b723c736412ab0` | `139e58268ef08ccf7299b89a467ad05bff5c6a5a` |

`v1.1` 及其全部祖先保持原对象。远端更新使用绑定旧 `main` SHA
`116193051f00d605055c545f409a5adc02b2f492` 的 `--force-with-lease`；
`main` 与新建附注标签 `v1.2` 使用原子推送同时发布。
推送后核对远端 `main`、`v1.2` 标签对象及其 `^{}` 解引用目标，
结果另存为 `publish-verification.json`。

现有实验结果、缺失样本、失败尝试与验收结论继续以原报告为准，
见[优化记录](optimization-20261007.md)与[版本对照](scores-20261007.md)。
`v1.2` 标记用户指定的源码版本；历史报告中的原产品 SHA 保留，
通过下表和 bundle 恢复追溯。

### 备份

本机备份目录：`/home/buptc/project/TianXiMem/.git/history-cleanup-20261007-183335`。

- `before-rewrite.bundle`：完整整理前分支、标签及可达对象。
- `snapshot.json`、`refs-before.txt`、`remote-before.txt`：整理前指针与文件树。
- `old-commits.json`、`rewrite.json`：完整提交元数据、主题分组及新旧 SHA 对应。
- `baseline-pytest.xml`、`stage-*.xml`、`stage-*.log`、`ruff.log`、`mypy.log`：
  测试和静态检查证据。
- `index-before`：整理前主工作区索引。
- `publish-verification.json`：发布后远端核验结果。

现有评测 baseline worktree 的目录与原 HEAD 保留。
已合入的旧本地主题分支按既有方案清理；独立 rename 分支与远端其他分支保留。
不执行对象垃圾回收，原 SHA 同时由独立 bundle 保障恢复。

### 旧提交对应表

按提交主题对应；混合清扫与注释提交允许一对多。
对应关系用于追溯，不能将合并后的主题 SHA 当成原实验实际使用的 SHA。

| 原提交 | 原说明 | 新主题提交 |
| --- | --- | --- |
| `fa38111f7043` | feat(datasets): 统一数据准备并接入四个新增基准 | `14d6fdc4e565` |
| `63fad3ef2d8b` | feat(eval): 补齐官方采集金标恢复与有序重放 | `8a2bc287d547` |
| `dfce93b2d420` | fix(eval): 让 PersonaMem 作答使用逐题检索记忆 | `9a7ccdb5f4b8` |
| `ca784a255776` | feat(eval): 增加版本化 AML 输入计划与逐事件续跑 | `e5a44419eaea` |
| `7e5c3993aefb` | chore(eval): 归档基准配置、输入审计和复现报告 | `855ea1f98818` |
| `116193051f00` | docs(repo): 对齐数据布局、评测边界与维护说明 | `95796f041be5` |
| `6121d6f9f555` | docs(eval): 归档十一数据集基线与持续优化验证方案 | `855ea1f98818` |
| `43a184c06582` | docs(eval): 记录片段快验、失败回退与冻结对照脚本 | `855ea1f98818` |
| `649db5a32e48` | docs(eval): 校验实际运行指纹并归档个人归属反例 | `855ea1f98818` |
| `04c01bd92fbc` | docs(eval): 记录来源反例、失败回退与先验片段验证 | `855ea1f98818` |
| `8d3917035be4` | docs(eval): 记录来源审查小样本验收并启动关联对照 | `855ea1f98818` |
| `337623c1dd89` | docs(eval): 核验冻结候选全仓测试并补充分数归因审计 | `855ea1f98818` |
| `5af90f589f0e` | docs(eval): 归档关联否决、就业句法与未知时间片段验证 | `855ea1f98818` |
| `82e1680896df` | docs(eval): 记录最终候选检查并恢复中断的验证链 | `855ea1f98818` |
| `901e9818d4c6` | feat(eval): 使用独立双模型服务调度冻结样本并保留已完成结果 | `855ea1f98818` |
| `3fff87bb9eb0` | fix(eval): 恢复超长上下文失败后的冻结样本续跑并保留错误 | `855ea1f98818` |
| `a8b1544d4da0` | docs(eval): 归档冻结回归结束状态与两题上下文受限缺失 | `855ea1f98818` |
| `7ed1502599de` | fix(memory): 按来源时间处理明确更正并隔离搜索诊断 | `9e36be2dfe3f` |
| `b28d2948bc03` | fix(facts): 修正通用角色标签和用户自述的来源归属 | `1a00a10defdd` |
| `d8671701cb5f` | feat(retrieve): 归档就业句法与完整字面路径候选 | `1e2135f4094e` |
| `014e6e6578a5` | fix(retrieve): 保留无法与更正比序的未知时间声明 | `d0cb189cb9b8` |
| `974b363c1c2a` | docs(eval): 整理三版本成绩与候选提交的验证边界 | `b1511e079092` |
| `05c952d2cf4e` | fix(eval)：JSONL 读写收口到唯一实现，修 3 个不转义的写方与 4 个会劈行的读方 | `a7474660bb4c` |
| `d911b1ca583d` | docs：修正一批与代码/Dn 相反的过期陈述 | `ad0f0bce8e55` |
| `240751cb72cf` | docs：消掉四处文档内/文档间的自我重复 | `ad0f0bce8e55` |
| `0c8246c5f470` | docs：把四处跨文档重述收敛成「一处声明 + 其余指针」 | `ad0f0bce8e55` |
| `4157c3f0d8e7` | refactor：把第二批判出的重复实现收口（哈希 5→1、CLI 形状 4→1、run 产物读取 2→1） | `780b59ca1a9f` |
| `e9935c57d963` | chore：第三轮清扫——删 6 处已验证的死代码，收掉 3 处漏改的 JSONL 写方 | `a7474660bb4c`、`780b59ca1a9f` |
| `90a0cde2183d` | docs：删掉不必要的"过去 X → 现在 Y"叙述，只留现状与理由 | `780b59ca1a9f`、`ad0f0bce8e55` |


---

## v1.2 后的改动合并为一个主线提交（2026-10-10）

按用户授权，将 `v1.2` 后所有开发分支的有效改动与当前未提交的实验目录清理
合并为 `main` 上的一个提交，并关闭其余本地及远端开发分支。
本节所在的整理提交直接以 `6dbba175937a4ae0014808b38e8d25add6fbc0cc` 为唯一父提交；
发布标签 `v1.0`、`v1.1`、`v1.2` 及其祖先保持原样，本地既有 baseline 归档标签保留。

### 纳入范围与原提交

审计本地 `main`、`wip/improvements-20261010` 和远端 `upstream/main`、`upstream/Andy`，
共有以下 9 个可达的发布后提交（包含原合并提交）：

| 原 SHA | 原内容 |
| --- | --- |
| `05c0c35a1b94bfba36d396806f09725dd8b86f64` | chore(repo): 合并保留改动、实验归档与目录清理 |
| `af956924e62050ab1f591527d0bfc43865eaac49` | docs(open-questions): 归档主办方对九维能力画像的书面答复 |
| `de0e1ec42f48e00a801e549b1f47ec20be73144f` | docs(eval): 补充四个 official-extra 数据集的统计报告 |
| `aa33923d83bdf466697ae8932914b88fe20853b3` | chore(git): 忽略 .codegraph/ 与 .tmp/ |
| `fd37a53a617301304581896a2ed50ae406ad212e` | Merge pull request #1 from YohannCharles/Andy |
| `6323e62c8d1c41d16955ddde3009b9d7f57bed1a` | fix(eval): 修正三处数据清单 URL（sha 不变，均已逐字节验证） |
| `b8bc41bb11b4b7f14c67cca51605343c868eda5f` | fix(eval): 数据准备修复 Windows 上的两处崩坏 |
| `a3872a83651a754768126d032fd59275fb36ff4b` | docs(baselines): 添加 MemMachine 源码参考与学习入口 |
| `f6a4e0def2c7a520adf96d1c30f12010f8f3cd38` | fix(eval): 修正临床多跳作答和裁判解析 |

`05c0c35` 本身已收拢原 `693ea08`、`7ace550`、`892cc60` 与当时工作区改动；
对应关系和更早完整历史仍在 `.git/squash-backup-20261010-102954/`。
这次额外纳入远端 Windows 数据准备与 URL 修复、主办方书面答复，以及 `Andy`
尚未合入的忽略规则与四份数据集统计。统计正文完整并入
[数据集主题卷](datasets-20261006.md)，原 `docs/` 路径保留导航；仅降低标题层级、调整相对链接。

当前未提交的清理一并纳入：旧 `configs/runs/`、`eval/reports/runs/` 文件从工作树移出，
原始字节与忽略产物保存在本机归档；两个目录各只保留 README。恢复方式见
[评测产物与旧实验归档](runs/README.md)。已回退的候选实现不重新启用，历史成绩不重判。

### 分支处置

- 本地仅保留 `main`，删除 `candidate/v1.2-grounded-evidence`、`rename-to-tianximem`、
  `wip/improvements-20261010`。候选分支已包含在发布标签中；改名分支是发布前旧历史，
  包名、环境变量前缀和部署标识的改名已由历史整理进入主线，不把旧快照重新覆盖到当前源码。
- 远端仅保留 `main`，删除 `Andy`、`cese`、`fix/user-run-merge`、
  `qwen3-models-and-doc-fixes`；后三者已包含在主线祖先中。
  已在远端删除的 `jxh` 跟踪引用通过 fetch prune 清除。
- 推送将主线更新与开发分支删除放在同一 `--atomic` 操作中，并对每个分支绑定审计时的
  `--force-with-lease=<ref>:<旧 SHA>`，避免覆盖审计后的他人更新。标签不参与此次推送。

### 备份与验证

本机完整恢复材料：`.git/post-v12-consolidation-20261010-105931/`。
`before-consolidation.bundle` 保存修改前全部分支与标签；另存 refs、远端 SHA、索引、
已提交历史、工作树/暂存区补丁和未跟踪文件。`metadata.json`、`commit-map.json`
记录原边界、树对象、分支处置和新 SHA；推送结果记录在 `publish-verification.json`。
旧报告中的实验执行 SHA 保留原义，不替换为整理提交 SHA。

| 检查 | 结果 |
| --- | --- |
| 完整历史 bundle 校验 | 通过 |
| 三方内容合并后的工作树与预期树等价 | 通过；文档规范化前逐树核对 |
| 代码与活动配置对象 | 与审计后的分支合并树逐 blob 一致 |
| 数据清单 SHA256 | 与发布基点一致；仅修复来源 URL |
| 四份原统计正文 | 全文保留；仅标题级别与相对链接改变 |
| pytest（排除真实语料冻结选题用例） | **1,264 passed，14 deselected** |
| Ruff / mypy | 通过；mypy 检查 43 个源文件 |
| URL 空格转义的离线请求检查 | 通过；已有百分号编码保持不变 |
| Git diff 空白、归档入口与新增导航链接 | 通过 |

测试输出保存在备份目录的 `pytest.log`。本次没有启动新的 benchmark、Smoke 或 Full，
不把回归测试作为新版本成绩。
