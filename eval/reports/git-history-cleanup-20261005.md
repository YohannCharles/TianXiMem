# Git 历史整理记录（2026-10-05）

按用户授权，只重组 `a7ed941b0000f927d7c1887c034a68f96ec55708` 之后的提交。
该基点及全部祖先对象保持原样。原 128 个提交（含 8 个 merge）整理为 12 个线性主题提交。

正式主线：`main`，跟踪 `upstream/main`；`upstream/HEAD` 是指向 `upstream/main` 的符号引用。
重组前 HEAD：`67fb106e77d1dec2e02bdf726623bcb587309193`。12 个主题提交的末提交：`3f28d7d472e9beeb70b00c04de113981c7fb3333`。
随后按用户授权追加文档和评测归档提交，详见下节。

## 整理后的提交

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

## 验证与保留范围

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

## 备份与复核

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

## 后续文档与评测归档

用户随后授权提交此前保留在工作区中的变更，按主题独立记录：

| 提交 | 内容 |
| --- | --- |
| `09fcfa8` | 25 份维护文档：根目录及模块 CLAUDE.md、Codex 入口、facts 模块说明 |
| `c16c9d9` | 七数据集重跑的冻结配置、后台脚本、审计脚本及运行中报告 |

本历史对应表独立归档。上述提交不改产品实现或评分口径，后台评测仍按原冻结指纹运行。
文档通过差异检查，评测脚本通过 Ruff 与格式检查；原始逐题产物及运行时状态继续忽略。

## 旧提交对应表

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
