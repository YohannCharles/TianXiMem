#!/usr/bin/env python3
"""benchmark_data/ 归档的**出处与哈希清单**——`make fetch-data` / `make data-check` 的实现。

## 为什么需要它

`benchmark_data/` **整目录被 `.gitignore` 排除**（理由见 [`.gitignore`](../.gitignore) 与
[`docs/benchmark-data.md`](../docs/benchmark-data.md)）：AML 仓**没有 LICENSE**、README 明写
"benchmark corpora … Do not submit any of these materials"，第三方数据又带 CC BY-NC 等约束。
所以**字节不进版本库**。

但"不入库"不等于"各自去搜"。归档的每一份都有确定的出处，且都能**按 commit / revision 钉死**：

    AML 评测 pipeline  ← github.com/AML-memory/agent-memory-leaderboard（按 commit SHA）
    LoCoMo-Refined     ← github.com/mem-eval-suite/LoCoMo_refined（按 commit SHA）
    ScriptMem          ← github.com/memorax-ai/ScriptMem（按 commit SHA）
    LongMemEval        ← huggingface.co/datasets/xiaowu0162/longmemeval-cleaned（按 revision）
    CL-Bench           ← huggingface.co/datasets/tencent/CL-bench（按 revision）
    PersonaMem         ← huggingface.co/datasets/bowen-upenn/PersonaMem-v1（`questions_*.csv`）
                         与 -v2（README）——两者都是各自的 README/问题表
    原始 LoCoMo         ← github.com/snap-research/locomo（按 commit SHA）
    LongMemEval README ← github.com/xiaowu0162/LongMemEval（按 commit SHA）

**唯一取不回来的是 `rh*.md` 三份**（抓取时原始 URL 已 404，是搜索引擎索引副本）——
它们不参与评测，只作"我们当时看到了什么"的存档。

于是"所有人都需要的那份"**不需要住在任何一台机器上**，也不需要任何共享盘：
**本文件是唯一真源，哈希是"到手了"的唯一判据。**

## 用法

    make data-check                 # 校验本地已有的是否与清单逐字节一致（默认）
    make fetch-data                 # 取回缺失/不符的（只取 required + optional 里有出处的）

环境变量 `TIANXIMEM_BENCHMARK_DIR` 决定落点（D16：数据路径的唯一入口，代码不得硬编码
`benchmark_data/`）；未设时用默认值 `benchmark_data`。

## 两条读法

* **`required`**：跑代理评测至少需要这些。少一份，Step 0 就交付不出来。
  （`required` 里 99% 的体积是 `lme_s_cleaned.json` 那一个文件。）
* **`archive-only`**：**取不回来**——原始 URL 已 404，或本来就是搜索引擎索引里抓的副本。
  它们只作为"我们当时看到了什么"的存档，**引用其结论前必须回原始页面复核**。
  校验脚本遇到它们只报告、不失败。

## ⚠ 本地修订（`local_patch`）—— 归档**不再逐字节等于上游**

AML 公开的 7 个 pipeline **在 `answer` / `evaluate` 第一步就崩**：

```python
async with httpx.AsyncClient(timeout=120) as client, output.open("a", encoding="utf-8") as handle:
```

`pathlib.Path.open()` 返回的 `TextIOWrapper` **没有异步上下文协议** ⇒ `TypeError`。
已核：上游钉住的那个 revision 上**逐字节就是这样**（`upstream_sha256` 记着那份的哈希），
所以不是我们取错了文件——**发布的参考实现本身跑不起来**。

**处置（团队决定）**：**就地修订 + 记成已声明的偏离**，修法只有两处机械替换：

| 改什么 | 改成 |
| --- | --- |
| `..., <path>.open("<mode>", encoding="utf-8") as handle:`
| → `..., contextlib.nullcontext(<path>.open("<mode>", encoding="utf-8")) as handle:` |
| （文件顶部） | 补一行 `import contextlib`（缺才补） |

**它不碰任何 prompt、不碰任何判分逻辑**——只把"同步文件句柄当异步上下文用"这处语法问题绕过去。
⇒ `eval/harness/CLAUDE.md` 那条「契约以 pipeline 代码为准」仍然成立，但**要带着这个星号读**。

两条哈希因此**故意不同名**：

* `upstream_sha256` = **上游原始字节**（`--fetch` 下载后先校这个）
* `sha256` = **本地归档字节**（打完补丁，`data-check` 校的是这个）

⇒ `--fetch` 落下来就是**打过补丁**的版本；`--patch` 给已到手的文件就地打（网络不通时用）。

## 与文档的分工

* **本文件**：出处、revision、sha256（上游 + 本地两套）—— 机器可查的部分，**哈希只写这一处**。
* [`docs/benchmark-data.md`](../docs/benchmark-data.md)：人读的部分（哪些文件是真数据、
  许可表、残留鉴定、schema 落差的指路）。**两边不重复同一条事实。**
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── 三个上游的钉死点 ─────────────────────────────────────────────────────
# ⚠ 改这里等于改"我们评测的是哪个版本"。AML 自己的 README 写明了这件事的分量：
#   "Public leaderboard comparisons are valid only when the complete evaluation
#    contract matches, including the benchmark release and pipeline version."
#   ⇒ 升级上游不是"顺手拉个新版本"，是一次需要留痕的更换。
AML_REPO = "https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard"
# "Publish public evaluation pipelines"，2026-08-27
AML_REV = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
LOCOMO_REPO = "https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined"
LOCOMO_REV = "887091190789e8d6760e70b9edd696539923dc4f"
SCRIPTMEM_REPO = "https://raw.githubusercontent.com/memorax-ai/ScriptMem"
SCRIPTMEM_REV = "22ac7e7e70124280d8af6100262ee7f88fff3436"
LME_DS = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve"
LME_REV = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

# ── 出处：全部逐字节对上 ─────────────────────────────────────────────
CLBENCH_DS = "https://huggingface.co/datasets/tencent/CL-bench/resolve"
CLBENCH_REV = "b28a5832a09b0d96c0cf4c22e90d7c60ede25b80"
PM_V1_DS = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1/resolve"
PM_V1_REV = "a8076d5608c93ba2a28983cd78aa99b01a163ae7"
PM_V2_DS = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve"
PM_V2_REV = "ed956dea41521fc4499acbc63f966e0fd3c053ba"
LOCOMO_ORIG_REPO = "https://raw.githubusercontent.com/snap-research/locomo"
LOCOMO_ORIG_REV = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LME_README_REPO = "https://raw.githubusercontent.com/xiaowu0162/LongMemEval"
LME_README_REV = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

# `required` 的判据：**跑代理评测（LoCoMo-Refined + LongMemEval）需要的最小集**
# + 契约权威（pipeline 代码）+ 许可出处（readme）。其余四份数据集的数据属 `optional`——
# §12.4 明确代理评测覆盖不到它们，不拉不卡任何 Step，但留着省一次重下。
MANIFEST: list[dict[str, str]] = [
    # ── AML 评测 pipeline（契约与裁判 prompt 的唯一来源）──────────────────
    {
        "name": "aml_readme.md",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/README.md",
        "sha256": "f72335e7df2c697e10d1495db217682eacf5736e40703b11108286a13c088945",
        "note": "AML 仓根 README。'Deliberately not included' 与'上游许可'两处措辞的出处",
    },
    {
        "name": "pipeline_locomo-refined.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/locomo-refined/pipeline.py",
        "sha256": "958f051cdfa715ca5be21d01ab3ac7a9078ff45b70629536713e53fe0fdf7aaf",
        "upstream_sha256": "31a49fc09e381bc117b59063c7a4cea543a672b23e1264aedb03290239671c1a",
        "local_patch": "async-open",
        "note": "LoCoMo/LME 共用契约的代表；答案与裁判 prompt 都在这里",
    },
    {
        "name": "pipeline_longmemeval-s.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/longmemeval-s/pipeline.py",
        "sha256": "79acd09e4dfdef9db50887c2d4baae297ba272e1f513ee2b4654e4185b311e42",
        "upstream_sha256": "c7f17c363bd0803cfeedd2b793a0cb0a74ab119a70eeced2c7a835b2b2aecb46",
        "local_patch": "async-open",
        "note": "补入（原归档缺）。它证实了 `{{speaker_1_memories}}` 注入形状",
    },
    {
        "name": "clb_pipeline.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/clbench/pipeline.py",
        "sha256": "9a76336b26fcb372515fd58d085dbf93c60e2a2abf21e18beecea4eee3160893",
        "upstream_sha256": "65b9719dc22e055aad808f68d757c145b9f9115cb2275b1ee373610c6a8116c7",
        "local_patch": "async-open",
        "note": "CL-Bench；写 `model_output` 而非 `generated_answer`",
    },
    {
        "name": "pipeline_beam.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/beam/pipeline.py",
        "sha256": "5ec1c4b675030ba1aeac61e1c262e552b9c99bb42a839b9fd20962c12fd4422a",
        "upstream_sha256": "3889e26ae696abc67711daab0ebf24e3082ed50d5f85df9d9f95540a0a9b8763",
        "local_patch": "async-open",
        "note": "**唯一传 `enable_thinking: False` 的一条**（V7 的证据在 pipeline_beam.py:263）",
    },
    {
        "name": "pipeline_scriptmem.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/scriptmem/pipeline.py",
        "sha256": "3df8e63542aad5efccd9d80d87f4987e8b6cc387c36ff1eafba598c4ce0fdc67",
        "upstream_sha256": "0f9931999b701f18d34b3bc1a5bb347330fafdaaa1e9133e1411973d9dc0acb1",
        "local_patch": "async-open",
        "note": "",
    },
    {
        "name": "pipeline_v1_personamem.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/personamem/pipeline_v1.py",
        "sha256": "f2054e3ed0bcba4a3aa92df7222ab0ca5cd87277ada09449e753b58569beebb7",
        "upstream_sha256": "a697a9c80214731037bc9cde6e57833423ff65a90393ead6e19b0b7d3ba90ccd",
        "local_patch": "async-open",
        "note": "补入（原归档只有 v2）",
    },
    {
        "name": "pipeline_v2_personamem.py",
        "tier": "required",
        "url": f"{AML_REPO}/{AML_REV}/data/personamem/pipeline_v2.py",
        "sha256": "a890c078a0015fe7788164b5274b40ca16e1feccba7a32f9659849830606dc08",
        "upstream_sha256": "c044b1ad10a87a94cfe0b007ec006a4146f7c191ca66b12cb4089ed70363a296",
        "local_patch": "async-open",
        "note": "PersonaMem **不读检索字段**——§12.4 用它解释代理评测为何不能外推",
    },
    # ── LoCoMo-Refined（代理评测的两份之一）────────────────────────────────
    {
        "name": "locomo_refined.json",
        "tier": "required",
        "url": f"{LOCOMO_REPO}/{LOCOMO_REV}/data/raw/locomo_refined.json",
        "sha256": "1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17",
        "note": "pretty-printed JSON 数组——**不是** Add 的输入源（见下一条）",
    },
    {
        "name": "questions.jsonl",
        "tier": "required",
        "url": f"{LOCOMO_REPO}/{LOCOMO_REV}/data/public/questions.jsonl",
        "sha256": "4dd84cf65a28ece0ed6b2a1b7b2700ea639f4e7b2c5e1afcc511e47e81732790",
        "note": "1,382 题 + evidence_messages + gold。**Add 的对话源是同一仓的 "
        "data/public/conversations.jsonl，本清单不收录**——它属 eval/datasets/ 的 clone（D16）",
    },
    {
        "name": "locomo_refined_readme.md",
        "tier": "required",
        "url": f"{LOCOMO_REPO}/{LOCOMO_REV}/README.md",
        "sha256": "411a5d7cf48b6a9f05c74e5084f036ad97eaa745759fb7ed8f667db7abb2fb12",
        "note": "CC BY-NC 4.0 的原文依据（readme:9 徽章 + :296 正文）",
    },
    # ── LongMemEval（代理评测的另一份；也是唯一的大文件）──────────────────
    {
        "name": "lme_s_cleaned.json",
        "tier": "required",
        "url": f"{LME_DS}/{LME_REV}/longmemeval_s_cleaned.json",
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "note": "**265MB，占了必需集的 99%**。这个 sha256 就是 HF 的 LFS oid（实测相等）。"
        "上游许可：MIT（数据集卡 `license: mit`）",
    },
    # ── optional：另外四份数据集，代理评测覆盖不到 ─────────────────────────
    {
        "name": "scriptmem_q.jsonl",
        "tier": "optional",
        "url": f"{SCRIPTMEM_REPO}/{SCRIPTMEM_REV}/data/public/questions.jsonl",
        "sha256": "5d9625c59f60473f5a70183a83531bc90a9273ad869fea794cdde79924b3d0f3",
        "note": "只有题目。**对话原文确实未发布**——上游 data/public/conversations.jsonl "
        "存在但只有 1,683 字节，内容是 `Synthetic example utterance showing the dialogue schema.` "
        "（已核对）",
    },
    {
        "name": "scriptmem_readme.md",
        "tier": "optional",
        "url": f"{SCRIPTMEM_REPO}/{SCRIPTMEM_REV}/README.md",
        "sha256": "bb9b064c2407ee38f72ebb27d32830c5f269362ac7b02d28c3fbace09a41c69d",
        "note": "CC BY-NC 4.0（readme:9 + :183）",
    },
    {
        "name": "clbench.jsonl",
        "tier": "optional",
        "url": f"{CLBENCH_DS}/{CLBENCH_REV}/CL-bench.jsonl",
        "sha256": "d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be",
        "note": "**出处已核**：HF `tencent/CL-bench`（sha256 = LFS oid，实测相等）。"
        "⚠ 它与 AML 的 `clb_pipeline.py` **不是一回事**——后者 docstring 明写自己跟随的是"
        " repository task 给的实现，**不是** `Tencent-Hunyuan/CL-bench` 的 `infer.py`",
    },
    {
        "name": "locomo10.json",
        "tier": "optional",
        "url": f"{LOCOMO_ORIG_REPO}/{LOCOMO_ORIG_REV}/data/locomo10.json",
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "note": "**出处已核**：github.com/snap-research/locomo 的"
        " `data/locomo10.json`（blob 逐字节一致）。**原始 LoCoMo**——代理评测用 refined 那份",
    },
    {
        "name": "pm_32k.csv",
        "tier": "optional",
        "url": f"{PM_V1_DS}/{PM_V1_REV}/questions_32k.csv",
        "sha256": "cccd34cf53e0bc4d9536c04cff5ca045156d9a4e227e83327112482840bbc93c",
        "note": "**出处已核**：HF `bowen-upenn/**PersonaMem-v1**` 的"
        " `questions_32k.csv`（blob 逐字节一致）。⚠ **不是 v2 仓**（v2 仓只有"
        " `benchmark/*.csv`）——本地文件名是被改过的",
    },
    {
        "name": "pm_questions_128k.csv",
        "tier": "optional",
        "url": f"{PM_V1_DS}/{PM_V1_REV}/questions_128k.csv",
        "sha256": "f0e137c3167fadbffbce5be2786105283c3299972c4ac1f158939155fd1578a7",
        "note": "同上——**v1** 仓的 `questions_128k.csv`（blob 逐字节一致）",
    },
    {
        "name": "pm_questions_1M.csv",
        "tier": "optional",
        "url": f"{PM_V1_DS}/{PM_V1_REV}/questions_1M.csv",
        "sha256": "f24db37c1ef49e8f3cc6585da3d7e1638bb9bfb32c95df4de065e4666e448e1f",
        "note": "同上——**v1** 仓的 `questions_1M.csv`（blob 逐字节一致）",
    },
    {
        "name": "pmv2.md",
        "tier": "optional",
        "url": f"{PM_V2_DS}/{PM_V2_REV}/README.md",
        "sha256": "18b068b91e3bfc5615806ebf5c74b143b79cbce22d41e7e26df50bbf5cc7b98a",
        "note": "**出处已核**：HF `bowen-upenn/PersonaMem-**v2**` 的"
        " `README.md`（blob 逐字节一致）。`license: cc-by-4.0` 的出处（pmv2.md:2）",
    },
    {
        "name": "lme_readme.md",
        "tier": "optional",
        "url": f"{LME_README_REPO}/{LME_README_REV}/README.md",
        "sha256": "c4ff45676683d9e2f7cf7d9099d26426f14635ec110dbb1da818d1019a142573",
        "note": "**出处已核**：github.com/xiaowu0162/LongMemEval 的 `README.md`（blob 逐字节一致）",
    },
    {
        "name": "rh.md",
        "tier": "archive-only",
        "url": "",
        "sha256": "d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed",
        "note": "❌ 不是数据——14 字节，全文 `404: Not Found`",
    },
    {
        "name": "rh2.md",
        "tier": "archive-only",
        "url": "",
        "sha256": "5f8069015710754c37650b00bf6216458c3593d5378ad04da6f0f74c6c52291c",
        "note": "❌ 不是 AML 材料——第三方系统 MemoryHub 的说明；"
        "原始 URL 已 404，是搜索引擎索引副本",
    },
    {
        "name": "rh3.md",
        "tier": "archive-only",
        "url": "",
        "sha256": "deac24af0c0e4f4ad2464dcbe2c4a79a07562997111f2bc5a52edfc8584d76e2",
        "note": "❌ 同上——MemoryHub 的跑分结果。**不要当成数据集的读取格式依据**",
    },
]

# ── 已删除 ────────────────────────────────────────────────────
# 留档是为了让"这个文件去哪了"有答案，而不是让下一个人重新从 /tmp 里翻出来。
# 哈希记下来，万一将来要复核某个已写进文档的数字（例如 lme_test.json 的 1,230 个
# 空 session），至少知道该找哪一份、以及它长什么样。
DELETED: list[dict[str, str]] = [
    {
        "name": "lme_test.json",
        "sha256": "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894",
        "note": "266MB。明令不用（多 1,230 个空 session + 15 个干扰 session，"
        "会污染按 20 条切批的埋点）。**删它是为了删掉一个等着被误用的坑**",
    },
    {
        "name": "beam.json",
        "sha256": "f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59",
        "note": "15 字节，`Entry not found`——失败下载的残留",
    },
    {
        "name": "beam_rows.json",
        "sha256": "b858dcadeb37aa87e47175192abda1b1de3a63841e349179593a98403fe62cf4",
        "note": '29 字节，`{"error":"Unexpected error."}`——同上',
    },
    {
        "name": "beam_100k.json",
        "sha256": "37e05479c60db09c18ab7b32eb987d82949fc1b93365ae80ce8addc5f8d7f0c3",
        "note": "HF datasets-server 的分页响应（`num_rows_total=20`，实际只取到 1 行）"
        "——**不是数据集**。⇒ BEAM 的数据实际上从未在归档里",
    },
]


#: 本地修订：`async with A as x, B as y:` 里的 `B` 是**同步**上下文管理器（`Path.open`）。
#:
#: 用 `contextlib.nullcontext` 包一层即可——它**同时支持同步与异步**两种协议
#: （Python 3.10+）。为什么不改成两层嵌套 `with`：那要给整个循环体重新缩进，
#: 而**逐行改动**才可能在评审时一眼看出"只动了这一行"。
_ASYNC_OPEN = re.compile(
    r"async with httpx\.AsyncClient\((?P<client>timeout=[^)]*)\) as client, "
    r"(?P<open>\w+\.open\(\"(?P<mode>[aw])\", encoding=\"utf-8\"\)) as (?P<var>\w+):"
)


def _patch_async_open(text: str) -> tuple[str, int]:
    """本地修订 `async-open`：`Path.open(...)` 包一层 `contextlib.nullcontext(...)`。

    返回 `(修订后的文本, 改动处数)`；**幂等**——已经改过的文本再跑得到 `(原文, 0)`。
    """
    patched, count = _ASYNC_OPEN.subn(
        lambda m: (
            f"async with httpx.AsyncClient({m['client']}) as client, "
            f"contextlib.nullcontext({m['open']}) as {m['var']}:"
        ),
        text,
    )
    if count and "import contextlib" not in patched:
        if "import argparse" in patched:
            patched = patched.replace("import argparse", "import argparse\nimport contextlib", 1)
        else:  # pragma: no cover —— 7 个 pipeline 都 import argparse
            raise RuntimeError("补 contextlib 失败：这些文件都没有 `import argparse` 可挂靠")
    return patched, count


#: 修订 id → 实现。清单里写 `local_patch` 就是这里的键。
LOCAL_PATCHES = {"async-open": _patch_async_open}


def apply_local_patch(entry: dict, path: Path) -> tuple[bool, str]:
    """按清单里的 `local_patch` 就地修订一个文件。返回 `(是否改过, 说明)`。"""
    patch_id = entry.get("local_patch")
    if not patch_id:
        return False, "无需修订"
    text = path.read_text(encoding="utf-8")
    patched, count = LOCAL_PATCHES[patch_id](text)
    if not count:
        return False, "已经是修订后的版本"
    path.write_text(patched, encoding="utf-8")
    return True, f"修订 {count} 处（{patch_id}）"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    """下到临时文件再原子改名——**别让半截文件看起来像"已经到手"**。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tianximem/fetch-benchmark-data"})
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(dest)


def _process_entry(entry: dict, *, root: Path, fetch: bool, patch: bool) -> bool:
    """处理清单里的一项，**返回它是否有问题**（= `main` 里那个计数）。

    分支多，但判据只有一条：**这一项最终是否与清单相符**。
    """
    path = root / entry["name"]
    label = f"{entry['name']:<28}"
    if path.exists() and patch and entry.get("local_patch"):
        changed, what = apply_local_patch(entry, path)
        print(f"  {'✎' if changed else '·'} {label} {what}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            print(f"    ✗ 修订后仍与清单不符（{actual[:16]}…）")
            return True
        return False
    if path.exists():
        actual = _sha256(path)
        if actual == entry["sha256"]:
            marker = "（含本地修订）" if entry.get("local_patch") else ""
            print(f"  ✓ {label} 一致{marker}")
            return False
        hint = ""
        if entry.get("local_patch"):
            hint = (
                f"\n      它是 `local_patch={entry['local_patch']}` 的文件："
                f"这份哈希是**打完补丁之后**的——对不上就用 `--patch` 重打（或 `--fetch` 重取）"
            )
        print(
            f"  ✗ {label} **哈希不符**（本地 {actual[:16]}… ≠ 清单 {entry['sha256'][:16]}…）{hint}"
        )
        return True
    if entry["url"] and fetch:
        print(f"  ↓ {label} 取回 {entry['url'].rsplit('/', 1)[0].split('/')[-1]}…")
        try:
            _download(entry["url"], path)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"    ✗ 下载失败：{exc}")
            return True
        actual = _sha256(path)
        expected = entry.get("upstream_sha256", entry["sha256"])
        if actual != expected:
            print(f"    ✗ 取回的字节与清单不符（{actual[:16]}…）——上游可能被改过，**别用这份**")
            return True
        if entry.get("local_patch"):
            changed, what = apply_local_patch(entry, path)
            actual = _sha256(path)
            if actual != entry["sha256"]:
                print(f"    ✗ 修订后哈希不符（{actual[:16]}…）")
                return True
            print(f"    ✓ 取回 + {what}，与清单一致")
            return False
        print("    ✓ 取回并校验通过")
        return False
    if entry["url"]:
        print(f"  · {label} 缺失（`--fetch` 可取回）")
        return True
    what = (
        "取不回来（原始 URL 已失效）" if entry["tier"] == "archive-only" else "出处未核，无法取回"
    )
    print(f"  ? {label} {what}——需人工拷贝")
    return entry["tier"] != "archive-only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="取回缺失或不符的（默认只校验）")
    parser.add_argument("--check", action="store_true", help="只校验，不下载（默认行为）")
    parser.add_argument(
        "--patch",
        action="store_true",
        help="把 `local_patch` 就地打到已到手的文件上（**不下载**；网络不通时用这个）",
    )
    parser.add_argument(
        "--tier",
        choices=["required", "all"],
        default="all",
        help="required 只处理必需集（约 270MB）；all 连 optional 一起",
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get("TIANXIMEM_BENCHMARK_DIR", "benchmark_data"),
        help="归档落点，默认取 TIANXIMEM_BENCHMARK_DIR（D16）",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"✗ 归档目录不存在：{root}（用 --dir 或 TIANXIMEM_BENCHMARK_DIR 指定）")
        return 1

    wanted = [e for e in MANIFEST if args.tier == "all" or e["tier"] == "required"]
    bad = 0
    for entry in wanted:
        if _process_entry(entry, root=root, fetch=args.fetch, patch=args.patch):
            bad += 1

    print()
    if bad:
        print(f"✗ {bad} 项有问题（见上）")
    else:
        print(f"✓ {len(wanted)} 项全部与清单一致")
    if DELETED:
        print(f"\n（另有 {len(DELETED)} 份已删除，理由与哈希见本文件 DELETED）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
