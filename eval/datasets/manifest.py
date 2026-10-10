"""公开评测材料的下载清单：URL、固定版本、上游与本地 sha256（PRD §12）。

这是唯一清单；本地目录映射在 layout.py，下载与校验在 prepare.py。
name 保留旧归档的标识，避免改变数据指纹与历史出处；不代表当前本地路径。
"""

from __future__ import annotations

import re

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
HYBRID_CORPUS_REPO = "https://codeload.github.com/wenhuchen/WikiTables-WithLinks"
HYBRID_CORPUS_REV = "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"

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

# ── official-extra：官方 2026-09-29 那轮真实流量里出现过、公开 6 管线之外的数据集 ──
#
# 判据与出处：那轮流量的语料普查（`official-dataset-2026-09-29/README.md` §4.2.2）。
# ⚠ 这份清单**不改**「代理评测只覆盖 LoCoMo-Refined + LongMemEval」那条边界（§12.4）：
# 它记录的是"官方实际在跑什么"，用于复盘与补标签，**不是新的评测基线**。
#
# `_OFFICIAL_EXTRA_FILES` 是紧凑表：**一条两行**（路径一行、缩进 4 格的 sha256 一行）。
# 为什么不写成 `"路径": "哈希"` 的 dict —— 最长的那几个路径加 64 位哈希超过 100 列，
# 会被 E501 拦下；而 E501 是**物理**行宽，塞进三引号里也躲不掉。
_OFFICIAL_EXTRA_META: dict[str, dict] = {
    "mquake-remastered": {
        "host": "hf",
        "repo": "henryzhongsc/MQuAKE-Remastered",
        "rev": "b54712d4b464d7e2d4edccd4022f95ddbcb719e7",
        "note": "CC-BY-4.0。多跳知识更新（CF3k / CF6334 / CF9k / T）。"
        "流量里 `Corpus: {…}` 与 `Corpus: UPDATE:` 两族属它",
    },
    "beam": {
        "host": "hf",
        "repo": "Mohammadta/BEAM",
        "rev": "3205395e897e7318c7b094ef4e6047b9b82dbb03",
        "note": "CC BY-SA 4.0。只取 100K 档（流量里只出现过 `beam_100k`）；500K / 1M 不取。"
        "⚠ 档位名是 100K 不是 128K",
    },
    "personamem-v2": {
        "host": "hf",
        "repo": "bowen-upenn/PersonaMem-v2",
        "rev": "ed956dea41521fc4499acbc63f966e0fd3c053ba",
        "note": "CC BY 4.0。题面取 `benchmark/text/benchmark.csv`（5,000 题 / 200 persona）；"
        "记忆取 `data/chat_history_32k/`——⚠ **只取 benchmark.csv 引用到的那 200 份**"
        "（上游该目录有 1,000 份，另外 800 份没有任何题引用；缺一份该 persona 就没法跑）。"
        "`chat_history_128k`（约 519 MB）与 1M 档未取",
    },
    "tempreason": {
        "host": "hf",
        "repo": "tonytan48/TempReason",
        "rev": "1646dea364ac667dd6098646da7d6638de00cc71",
        "note": "CC BY-SA 3.0。只取 test_*（train / val 不取）",
    },
    "corporatebench": {
        "host": "hf",
        "repo": "epiq-ai-labs/corporatebench",
        "rev": "c6a0ead2670fb1a2fc3cd1c74d43aeae33e19207",
        "note": "Apache-2.0。只取 zenith 一家——流量里是虚构公司 Zenith Labs 的邮件线程",
    },
    "medmemorybench": {
        "host": "hf",
        "repo": "Cyan27/MedMemoryBench",
        "rev": "1f3140476b20245945a6c52a82038847fe7ef25f",
        "note": "⚠ 许可两处冲突（HF 数据集卡 CC BY-NC-SA 4.0 / GitHub 写 CC BY 4.0）。"
        "只取 data/zh 干净档",
    },
    "memtrapbench": {
        "host": "gh",
        "repo": "zjunlp/MemTrapBench",
        "rev": "02fc32a2b5ffecce893e353b29fcd49242ea71cf",
        "note": "许可 unknown（整仓无 LICENSE）。记忆陷阱题；整仓取回（含它自己的评测代码）",
    },
    "medmemorybench-code": {
        "host": "gh",
        "repo": "AQ-MedAI/MedMemoryBench",
        "rev": "7227bc105b84a1a9f7a75861eb9e1be3ea502882",
        "note": "⚠ **只取判分那一路的代码**（`metrics/` + `utils/prompts_judge.py` + "
        "数据集/评测器骨架），不取 methods/（2545 个文件的方法实现）与 data/"
        "（HF 那份已经取过）。用途：判分口径（`extra_pipeline.py` 照它实现、prompt 直接从它读）",
    },
    "doc-pp": {
        "host": "gh",
        "repo": "hwanchang00/doc-pp",
        "rev": "a70798e0d1f3f38cda68bed86876c474f9f453ed",
        "note": "CC BY 4.0。文档披露政策（Doc-PP）。整仓取回：`data.zip` + `data.z01`–`z06` "
        "是**分卷 zip**，解包办法见 docs/benchmark-data.md"
        "（解出的 `data/` 是派生物，不进本清单）",
    },
    "halumem": {
        "host": "hf",
        "repo": "IAAR-Shanghai/HaluMem",
        "rev": "cb04336aa1b732d4b24f5186c552456b4099806e",
        "note": "**CC-BY-NC-ND-4.0**（非商用、禁演绎）——仅本仓内部评测用，不得再分发。"
        "只取 Medium 档（流量里那个 persona 访谈族是它）；Long（约 106 MB）不取。"
        "`eval_tools.py` 是上游的**裁判 prompt**（我们只读、不复制，见 "
        "`recover_halumem.py` 的文件头）",
    },
    "hybridqa": {
        "host": "gh",
        "repo": "wenhuchen/HybridQA",
        "rev": "db22fda8c5951438fade3c69d75b350335ba93b3",
        "note": "MIT。表格+段落混合的多跳问答。只取 `released_data/dev*.json`"
        "（test 的答案封存到官方评测，dev 够用）",
    },
    "musique": {
        "host": "hf",
        "repo": "voidful/MuSiQue",
        "rev": "a7d9f9adf6191604fc67cde318ee1a86fcf7babc",
        "note": "CC BY 4.0（HF 镜像；官方发布走 Google Drive）。"
        "⚠ **两份都要**：`ans` 给金标、`full` 给 `paragraphs`（join 靠语料的多重集，"
        "不比题面——见 `recover_corpusqa.py`）",
    },
    "feverous": {
        "host": "fever",
        "repo": "feverous",
        "rev": "",
        "note": "标注按 Wikipedia 条款（CC BY-SA 3.0）。事实核查（claim → 三分类）。"
        "只取 dev challenges 那一份",
    },
}

#: 数据集目录名 → 紧凑文件表。**一行一条命令式**：路径行 + 缩进 4 格的哈希行。
_OFFICIAL_EXTRA_FILES: dict[str, str] = {
    "mquake-remastered": """
README.md
    d95310f367c0c7f3230b5e091275c09f16a097b12b05b3d53b61ac34ef9d9320
data/CF3k-00000-of-00001.parquet
    12c8bc1fe1a5e6b9edc15e2348898a92b4c6b50f6a0693793239af9c0cf9eee0
data/CF6334-00000-of-00001.parquet
    720352ea9926d805391870f7feb70394b23f2bc7478024ecd7cca600f13096ea
data/CF9k-00000-of-00001.parquet
    74f5cd123b2b6d4117398d543400e9a6d4c730814b0ea42cbfadbbfc8d309534
data/T-00000-of-00001.parquet
    7ac75092f2241749d520a2f17ee65d00737354ed9931d372cf4c702ba5802e7f
""",
    "beam": """
README.md
    30b9ff56f4ca52e4078754e91f5e8dd8cd73b1b1246a42a000f11ce6b940400d
data/100K-00000-of-00001.parquet
    c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace
""",
    "personamem-v2": """
README.md
    18b068b91e3bfc5615806ebf5c74b143b79cbce22d41e7e26df50bbf5cc7b98a
benchmark/text/benchmark.csv
    95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52
data/chat_history_32k/chat_history_250913_163134_persona10.json
     037435a0633e16d2b1452b81b193b266173df8dc039c62d8c655ed161a1331eb
data/chat_history_32k/chat_history_250913_163134_persona101.json
     92bcf8719ccd00e8decd064c298dc5423e2deaf9c642a5bb1dfdee886436e49d
data/chat_history_32k/chat_history_250913_163134_persona107.json
     6c346a2ffea6e11cddf7eed2a75328fdf35ddc9225573a7c856620e306ba0755
data/chat_history_32k/chat_history_250913_163134_persona110.json
     20ae5ef442530409c19e3b4642ce0f6be9478bf8ddcca6da09d6627ac04b84f8
data/chat_history_32k/chat_history_250913_163134_persona120.json
     daee7e3be50ae64271efabe6fffb9d238ee355338aace57f849830aff07bcbcc
data/chat_history_32k/chat_history_250913_163134_persona136.json
     5f7607228ecbb8da10ba2be01761854394d6b11e83522656bfd91ce6e09d150b
data/chat_history_32k/chat_history_250913_163134_persona137.json
     4624ab39cb649b578a1e9ffcfd2413c173feba97c552e1552fcff04e8e31fce2
data/chat_history_32k/chat_history_250913_163134_persona139.json
     a4eca1cfcdbf247f71a9b68ad0fbeec99a172e716071f4e967ba7cabb93dfbb1
data/chat_history_32k/chat_history_250913_163134_persona158.json
     dff9f3a82830670fadfa6caa2b3d7741d72ced881ca1824e8e589cc2711913ad
data/chat_history_32k/chat_history_250913_163134_persona168.json
     925725558cb9a46ef1887c567471841dc3dd8a0265ae3ae983f575f3a74bd7b7
data/chat_history_32k/chat_history_250913_163134_persona174.json
     de7f2e1b7d62515c8112019a96a99ed066085fdc06e68c9d763d3f6ceb905de3
data/chat_history_32k/chat_history_250913_163134_persona184.json
     043e4bf181644131f67b4f2cb5e0c5b52eaa2b09b74e63599b23632cd1c1964f
data/chat_history_32k/chat_history_250913_163134_persona198.json
     c865da8c0bd12b37072ad2bd04c81f94ec6538917c2a261b0b018a0a24ebc1f1
data/chat_history_32k/chat_history_250913_163134_persona199.json
     d856eb005b7070f2038b5ed4194e2d9345d7bee1976b4bc3c19476664af241cf
data/chat_history_32k/chat_history_250913_163134_persona208.json
     61b9d1ebb0909fe6eba6ff9206d373d2369b70f3214a37e11597a09942ea4b72
data/chat_history_32k/chat_history_250913_163134_persona209.json
     f08838bf143b9657e3eb723fd50104bc8f39e4ab44fcb0a2ccc84cbac21230a1
data/chat_history_32k/chat_history_250913_163134_persona210.json
     663db27c168c54c9693012e49c8d5eeb9d8aa6dafd343ec269c65928ecf1f4ba
data/chat_history_32k/chat_history_250913_163134_persona213.json
     bff60eee91eafc78e14eb089599e934d7d37e175989b3a9932d92601601d5fcb
data/chat_history_32k/chat_history_250913_163134_persona215.json
     c3e457bb3a786a3f84dd93a1627d92d204524003c7f107b439c7d2b7a70dcb73
data/chat_history_32k/chat_history_250913_163134_persona218.json
     649a9a77782c5899577f6f33a299c9ed5b12f17398b838f9536b18d432618471
data/chat_history_32k/chat_history_250913_163134_persona221.json
     103f5d722249afc31fd4b2f78acd9f3bd4aca258f09a90b6280b54027d99d051
data/chat_history_32k/chat_history_250913_163134_persona23.json
     e2fcc999ca483a2269d47264b1f4fce9884d8f93b7efc50e9b386ea66952ef0d
data/chat_history_32k/chat_history_250913_163134_persona235.json
     54276f245a580ed9e21a57e9acd4e5b5e13d4ebbc2b16c5a3dcb2dc32270abf8
data/chat_history_32k/chat_history_250913_163134_persona237.json
     0cbbaa93c0d5bdd29b6fc9b66a448e8224a07ee77fb4b9397acb711b4c5aae06
data/chat_history_32k/chat_history_250913_163134_persona244.json
     741a2484c1c93c483a4a32d81d5ec052820bb8101b94ae1eaef4d41499770255
data/chat_history_32k/chat_history_250913_163134_persona247.json
     22e3efa2f00d807899e18655859291c67bf80e00bd337ed620b79926dde81cf4
data/chat_history_32k/chat_history_250913_163134_persona25.json
     6f7775ce8d0f12b9babb317f7a8652aa6514fb5a79e988c1e4cdb12b56bcb270
data/chat_history_32k/chat_history_250913_163134_persona254.json
     9db47c89503b42b382c4cfb37beef8f192c3c17d086120f03aeb7c78cf8954e8
data/chat_history_32k/chat_history_250913_163134_persona256.json
     ccb9cfb5511de007a4fb2e85e74500557403289412ef391aa16425e71c14625f
data/chat_history_32k/chat_history_250913_163134_persona259.json
     332fe0f3f38a4906a70e46445eccd2a8d00a68fdf631451eedc0ac9e8c7c3a92
data/chat_history_32k/chat_history_250913_163134_persona260.json
     68607da0ffacbdc9f407dea1d5ef290e029648cf2b587f37df85f13ee7af42f5
data/chat_history_32k/chat_history_250913_163134_persona261.json
     d0d2cd1454f8d302f37847dd4001f94637ad5d618e99e08f40504fd6f6c0d856
data/chat_history_32k/chat_history_250913_163134_persona265.json
     428f41867f3b407c6ec4a6d2cfe2490fbe0fc2dfba77cd7eacc2d2b999453cde
data/chat_history_32k/chat_history_250913_163134_persona275.json
     4cc36aa8c8429b6c354904d772d453f02ac21b6ba9c72283db761346f03f43ca
data/chat_history_32k/chat_history_250913_163134_persona277.json
     c3ba6796a1e1b8aaf98840f7b2724d1f9ba704ed4a93b93ded796a1d95220f43
data/chat_history_32k/chat_history_250913_163134_persona280.json
     62239ba6b4c4f92dad660f66a8c28f23321e85322170017b0c279793da4ab072
data/chat_history_32k/chat_history_250913_163134_persona281.json
     e81c6092a428580f3a1978633dfadde1f4ac73c703d964b3fa902dd2438d006e
data/chat_history_32k/chat_history_250913_163134_persona289.json
     3b2cae150ff1a8b3db632a11dcda6cffba5e6c8a88a0642d5c9e4b44d373807f
data/chat_history_32k/chat_history_250913_163134_persona292.json
     855beea1d1f3b2befbcb19b35843c96b446b37453aa54cb45b383ee44d5197d7
data/chat_history_32k/chat_history_250913_163134_persona294.json
     2d444c2d989db5e05072616dfe97bb9bc08e4168d7e4b3ee938bd608e7f88d30
data/chat_history_32k/chat_history_250913_163134_persona296.json
     70c80ab61b7dfe068f5ca761a7b0fa406704305acf02ddd25177f5727d562d36
data/chat_history_32k/chat_history_250913_163134_persona298.json
     ae893ba2333431f731779232a0d23b14b3adbc9b7eff2bb458a312bdf83e9cb9
data/chat_history_32k/chat_history_250913_163134_persona299.json
     92dfe675d3367b57a6d2269224223197bc32d65abefc042fb633436be5a5eb8c
data/chat_history_32k/chat_history_250913_163134_persona30.json
     98f987eaf627db76cef2d584daebb121c1c7f7d5e88d85482dabd4aae0533062
data/chat_history_32k/chat_history_250913_163134_persona307.json
     f0b0c82f72e075ad57ce528a50015f41d2ed19009b47020c38e2009f1111a03a
data/chat_history_32k/chat_history_250913_163134_persona309.json
     9f98c2009b9aca61193448c4e53b44aaa2f80b3b4a1535f16a12a4006e429d28
data/chat_history_32k/chat_history_250913_163134_persona310.json
     710020a24942fd5f46b0ebd54f23c0ac0d3576f8e1972e74e64b47f8b5694d44
data/chat_history_32k/chat_history_250913_163134_persona312.json
     8f4af9ce7b197addcab05a41e287817ef55efd2d0b921de72ee086e9e5dd2e94
data/chat_history_32k/chat_history_250913_163134_persona314.json
     03ff163d7dab4178b0099fc06ebc6eae33812fa260f25daacf1111327af46c66
data/chat_history_32k/chat_history_250913_163134_persona318.json
     39930dff7229d52aa006bf046f7725b1cd4d8eed1375b62b8e023694dabdc581
data/chat_history_32k/chat_history_250913_163134_persona319.json
     b10bb6544a0908a9f36072bd60a1e0991ef8af6cf3d3f4c339ab9a9725781232
data/chat_history_32k/chat_history_250913_163134_persona321.json
     0372438f5254124577aa7402ce4096a8266c6176d686f3d61a25762ee491d136
data/chat_history_32k/chat_history_250913_163134_persona327.json
     d59702d8a476cc4bdf552272282578e34cb221c666db43e7e1c22befb35d80d9
data/chat_history_32k/chat_history_250913_163134_persona328.json
     26d9ce6f6203c38f80a1bf844d2ca8f16777c7329ccd3691e8448573892ca330
data/chat_history_32k/chat_history_250913_163134_persona332.json
     430d9f0769e559276ec2ed0e1fe91eb56833a25ad0cbf4aac3ed5b77ee235c05
data/chat_history_32k/chat_history_250913_163134_persona346.json
     79ca7c47e95afbf44134aefd96f26f27b781c9e005f8c2f79fd65cea0804210d
data/chat_history_32k/chat_history_250913_163134_persona351.json
     dc81a52a7d9c0e68f7283dee9507c95c5cbbc0ae208e8d2dc0a4336cea169f43
data/chat_history_32k/chat_history_250913_163134_persona355.json
     570852049126f9ec3d1e5c04c57911a492cd693dc96c26a7d3a563629428822a
data/chat_history_32k/chat_history_250913_163134_persona361.json
     94f5c46711fbb1b97896e988f5b6d005e307838c716f608ea5c185f49e0e71a5
data/chat_history_32k/chat_history_250913_163134_persona363.json
     3fa4373cc542dd783c04720ae12e3b2d9fd0b900208a3633838df88d6e103b9e
data/chat_history_32k/chat_history_250913_163134_persona365.json
     48a8c8606012b924e0f5a14e0b5b39f20c4b257622f21e52128d89c7e38eadc7
data/chat_history_32k/chat_history_250913_163134_persona370.json
     c9bc46646607bdb9edc5f3af08dd95065b7411c719c137de81bfe2b44fce3639
data/chat_history_32k/chat_history_250913_163134_persona371.json
     18395f4e3436edb772726ee6c7a402b38704da4def5b0a3c14a0745fffc2edf0
data/chat_history_32k/chat_history_250913_163134_persona377.json
     e0ec7524c4f2489469571bd1e8e41150a2eec35625f2f1ea5fc4ff7334df18ec
data/chat_history_32k/chat_history_250913_163134_persona382.json
     cbe290edd9ad1e7dd140590ac5156042e3bea46d085cc9ef0fc16fe1b504557e
data/chat_history_32k/chat_history_250913_163134_persona39.json
     79fb9facd43ae394ce0d4324cf7594966bf7d40b786059e37ec7d8779c6f61b8
data/chat_history_32k/chat_history_250913_163134_persona408.json
     670473959bb25d8f2d89722a4dbcf140fbc1c88076ccf719c0c6833aecfe5325
data/chat_history_32k/chat_history_250913_163134_persona411.json
     21d92442f8343d9c80df64eac96e532ab57f286517961892b082f587e2067fa4
data/chat_history_32k/chat_history_250913_163134_persona425.json
     9ecbad6b9f76e97240e92d2d1f2427d45d93ed5a48e0dc54e3d165e8a590c829
data/chat_history_32k/chat_history_250913_163134_persona429.json
     4cf61466b06e52fbb831ccd86c0674151ff7e3ee96aa7701fa39e0ddb20c409d
data/chat_history_32k/chat_history_250913_163134_persona432.json
     0d7cb270f936cb37bc31d6948d08b36da82cc41a4bd5d018d41444b09694e4f1
data/chat_history_32k/chat_history_250913_163134_persona436.json
     6e8a8b81ac087060516c4489fa76eb820b2cb5146f73855c9bec17b0ce1d6dd6
data/chat_history_32k/chat_history_250913_163134_persona439.json
     cf6e818a97e8998a5613443654d7d531e6ab8495c520919779edb1c750b3a55d
data/chat_history_32k/chat_history_250913_163134_persona44.json
     b81e0ea155cfad1fdcd952d942a4f3cb837ab2ef24ef2b1e717962c38cbbe5da
data/chat_history_32k/chat_history_250913_163134_persona445.json
     e4c3b349234d6a152ea34b1d7e7547dcb1c37d3b1101d6baf5e6ecedcdd0c5bf
data/chat_history_32k/chat_history_250913_163134_persona451.json
     c6cc597365af36e74c91d03e86a20031f2cbc4d7bcb69080939fc0f0bb965496
data/chat_history_32k/chat_history_250913_163134_persona462.json
     99f5c4a32b8145035ec9c21283f70a37df3cb4528b6286260dee6cfd23484aa3
data/chat_history_32k/chat_history_250913_163134_persona479.json
     b4a85c41163229b6a33596f36c3ce3f261e4df26cf6eaff55dd0516ee85b6b8a
data/chat_history_32k/chat_history_250913_163134_persona486.json
     5ab6a5a81170354b51491ab8ace8da4ecd256166eaa9f8d4521ba4bdb06a8b44
data/chat_history_32k/chat_history_250913_163134_persona493.json
     95a57318e160b6d09eee72ea9327a9333cfe826751949deb14c0efd0e14c38c9
data/chat_history_32k/chat_history_250913_163134_persona494.json
     3f02392f7ff8d5e4862e96911e3f825dd123dc92103d8248ad5b16518d145cdd
data/chat_history_32k/chat_history_250913_163134_persona499.json
     1a7accc6533625a83ae05a4e9143f8e9130ed14fb82291d719e501f9add30463
data/chat_history_32k/chat_history_250913_163134_persona513.json
     4d56fdd4f2df3ec9d8f3ac01205f2adcc3b95f91deabe916155408bc7137794b
data/chat_history_32k/chat_history_250913_163134_persona521.json
     c0527d65a7f4c5688294f41482f3c9a64784077ad48ee09befa41dfedc0205e9
data/chat_history_32k/chat_history_250913_163134_persona522.json
     891366fba042c160ed94474116e83b3dceb25910eb61fee89864d1aaf4e0866a
data/chat_history_32k/chat_history_250913_163134_persona526.json
     1124236e9c0d8d200e8c50422b52f148389ea01201bad4deab4f1bbec536d48c
data/chat_history_32k/chat_history_250913_163134_persona527.json
     fad4d4aedd07e0d9cb9497fe7aa6eed2665d218e5f164614d5fa5ec9106b1036
data/chat_history_32k/chat_history_250913_163134_persona528.json
     7dd31bbc761419c11e9dfe4f3f256a0aeaf7f8f34eee536df032f63432275ec4
data/chat_history_32k/chat_history_250913_163134_persona529.json
     29d1747a41b084717e9be4bb31c1940186806f47374bd1342ab54a4230b5043f
data/chat_history_32k/chat_history_250913_163134_persona534.json
     4072012e5872918990c612d698b50c16b0aeff95b52a23504db33588ea0a7979
data/chat_history_32k/chat_history_250913_163134_persona54.json
     dcbd7816f6a17e1f0022ccabec810c3958cff66cd2cb6029e927f0383f62e470
data/chat_history_32k/chat_history_250913_163134_persona542.json
     45909cd963e5585645660ef04c7317e605b7a568b3a64df17c59c3b8a376509f
data/chat_history_32k/chat_history_250913_163134_persona543.json
     74263cde83264be3f3459b177ed3b5833778ca56aee83e343b74dc0606da3d84
data/chat_history_32k/chat_history_250913_163134_persona548.json
     85c256893c227d684b0f8dfa2b2688cc4610e3b41be9a736e8cdc1e07d443c4f
data/chat_history_32k/chat_history_250913_163134_persona549.json
     8c3574fbc4697c6eeed390b5b002add31b3a6680e4a922f665ed201a77cee906
data/chat_history_32k/chat_history_250913_163134_persona55.json
     9475d7ba49185d5405c035727543af15d2dcda36024eb369f1c012a1390e28a6
data/chat_history_32k/chat_history_250913_163134_persona554.json
     2cd5a37e1646d6f449d8e577f0b04b587d22a567b895e4730a863c16858ea755
data/chat_history_32k/chat_history_250913_163134_persona559.json
     f8eb74cd64c0098be8b8becdf625eee26ca0fc41c4745c801c4ad2c5e2b3bd99
data/chat_history_32k/chat_history_250913_163134_persona570.json
     0ddce2154c8119e9da69f3fe8651c4abb13ff7582267949c08c1ec5f4e6b6c8f
data/chat_history_32k/chat_history_250913_163134_persona578.json
     53070ebb673523e56076fa8da1f0f928bc22ceaa8dcd7b986ea07f42ca3bc529
data/chat_history_32k/chat_history_250913_163134_persona583.json
     9e454690bc59eb5d8acb2cbdc1f5787a52a5361690591d26584b69184e8045f9
data/chat_history_32k/chat_history_250913_163134_persona584.json
     64f46173de0449361f8682ea8f7ffa0e5c64aca6f0e470d16910f557c5f0ea7f
data/chat_history_32k/chat_history_250913_163134_persona589.json
     98e45d0e6225924a7970bc8b58ec17669b1be919edfca167611dcebea0923f10
data/chat_history_32k/chat_history_250913_163134_persona59.json
     e5bcfeecad924e371b7ad49f56ba71c5ce019f1efbb82109b0ceaba8f4b08fe6
data/chat_history_32k/chat_history_250913_163134_persona590.json
     1dab7a718cc7e2932b090cc6d901c67f7e731abbcbf916cd33dba8c8a85ceef7
data/chat_history_32k/chat_history_250913_163134_persona595.json
     1e4de470f187b7375140e26abac3a7ac7ead213441e698f3efc791855254bc84
data/chat_history_32k/chat_history_250913_163134_persona599.json
     081c7c5a742835e7958a20979c92a2147397d27da771f7510b41f6802f9fbba6
data/chat_history_32k/chat_history_250913_163134_persona60.json
     83aeabc73b56c55c4dc2202bddc1fc012f3a8b6d8537dbe39dc0ec0e667bb296
data/chat_history_32k/chat_history_250913_163134_persona601.json
     155dbc78520f3851006470c6ed8cf84d80e2b1b323813c348a53712c6d30ccc2
data/chat_history_32k/chat_history_250913_163134_persona604.json
     4382e520b558258bc3296133cd37160b43ee2f60ba1ab0aa0e82106fe77498c9
data/chat_history_32k/chat_history_250913_163134_persona613.json
     1c7b5fd2e6d3744cb22ff119d1e495c5522d10bbf62d7d67c3b509b48c694cd2
data/chat_history_32k/chat_history_250913_163134_persona617.json
     3a91384c0ed3436228c053f715b905850be725e57a5bdecabe7b59b190bada25
data/chat_history_32k/chat_history_250913_163134_persona621.json
     22ebf63a1d1902cd4f3c0e23376f8f84618a8bfb0606ca126d2ba90b035c6300
data/chat_history_32k/chat_history_250913_163134_persona626.json
     45b270520bed0a87d66ec1a1f62a9466e9ef8ae552ae45b7c276e664a59116d0
data/chat_history_32k/chat_history_250913_163134_persona63.json
     b3152814812ba356e2c02eae0856051a5ef125e6e779e865a2b8f7836d82e756
data/chat_history_32k/chat_history_250913_163134_persona635.json
     ce8a47ef868395088160b572830836f99a1aa3eb89291347cd3e80b738c1037c
data/chat_history_32k/chat_history_250913_163134_persona636.json
     487c40bfd05f5af906a31f7391548931981ded30dfb7b7407582e39bb14558a3
data/chat_history_32k/chat_history_250913_163134_persona643.json
     617fc7f85bd18c41f935b406eb412ea590318171d733a06a4c7f87be06409baa
data/chat_history_32k/chat_history_250913_163134_persona644.json
     dbac0f5694f01ae15ad9ca024da4c92f416a96b56a6fc3f656e455a3b92174b4
data/chat_history_32k/chat_history_250913_163134_persona649.json
     5a8cd973c13dafa43bac81846ef51000b36a6f0f187d5c7876faffc28de2392e
data/chat_history_32k/chat_history_250913_163134_persona650.json
     18dc3ba6f5157f80dd88e144e989394ab0f26781ea3f5fa8b67ec27304e357de
data/chat_history_32k/chat_history_250913_163134_persona652.json
     2b35cf1980242045cbe77b6fb1bb9bbd72fcb37fbd661e05ec3a729992318a75
data/chat_history_32k/chat_history_250913_163134_persona656.json
     abdb687b42473b6195a935b753b1747a0ff69bd195d2915239bd5b0ae084d9d9
data/chat_history_32k/chat_history_250913_163134_persona66.json
     73744bd77ced34dae4cd10ff4a258f52d89c7e62bb43c32bed6fac321bdd38ec
data/chat_history_32k/chat_history_250913_163134_persona660.json
     b93dfcd1704847b904f6b649bf455bbc58d749fc503529be602b67b86ec16784
data/chat_history_32k/chat_history_250913_163134_persona662.json
     a10ffb2d8fc1d058b33bc2b8e96c1b39615c6d37584d84a28a7cd634c379cd69
data/chat_history_32k/chat_history_250913_163134_persona668.json
     ab6efc9dde27c3b226f43c9181ee5923805b716d605a5a7a4ed02a7b05850cdb
data/chat_history_32k/chat_history_250913_163134_persona67.json
     7d43f07200a7e7974b113465ece51266ffa14b82e8eb8664b859e184c57acb33
data/chat_history_32k/chat_history_250913_163134_persona670.json
     692993b1bb7fb731799a5fa4f691d6e7eeb217fb9ffa94c867fc780eba7886a4
data/chat_history_32k/chat_history_250913_163134_persona677.json
     e7f9429091cd1b81dbe4c6a317fc0ac4a816a2ebe2c97a2cf85baf9ab2b45fa1
data/chat_history_32k/chat_history_250913_163134_persona678.json
     3cf8bf44d67415948933b91ad55ccc82d241ccd6d8565460ae002f8f564a0ce8
data/chat_history_32k/chat_history_250913_163134_persona679.json
     6ff3a2ecf6fcb4ae103f3abc218d7b264a2b29a548d9301e75e8ea651f47f61d
data/chat_history_32k/chat_history_250913_163134_persona687.json
     1a79a0a93302736692d0f0a76bb2c1887ed83a6d75d29c4971be4ab48fc335d0
data/chat_history_32k/chat_history_250913_163134_persona689.json
     0fa8629d226fd98bec096ff7f1ba387d783f1b4fc6c495b39b3ec6d6c70ed082
data/chat_history_32k/chat_history_250913_163134_persona692.json
     13d526bc5417c53fe89825e1dfe634dab73d9dcffce6971e2816a33869ea51ea
data/chat_history_32k/chat_history_250913_163134_persona697.json
     f8ca8d88c55baaf784db6f933e0e93e3bca9460c51ea9b7004b40ac07104ce65
data/chat_history_32k/chat_history_250913_163134_persona70.json
     c207f70523f84346277549c66998afe68fab9d09ffcd37a901c2019a505837d8
data/chat_history_32k/chat_history_250913_163134_persona713.json
     c5621b7b3d323170f980a7a3b5c0d586f606eff3908c109fa31c193e5747d0a2
data/chat_history_32k/chat_history_250913_163134_persona714.json
     f863ffd8e07da32bf8e07fa5f40d6fc07efe1c481a9d0dad6cd4b913561ed983
data/chat_history_32k/chat_history_250913_163134_persona716.json
     ac5248f8321a9e50fadbcbe09ca92223a77e5b34e85bd4c2f17eb3da5b23bcb0
data/chat_history_32k/chat_history_250913_163134_persona72.json
     e34f58106cf00da5b68d68c10b55ee01e67422d841a4d48fdc49fb71e2b74bd7
data/chat_history_32k/chat_history_250913_163134_persona721.json
     b3dc0a239b0d575b507eebe5eac24518c3cc27c434715a024732a6adf8be7d60
data/chat_history_32k/chat_history_250913_163134_persona737.json
     007f5584269bf496a7313ec7dd4ad7d5f27e0887dec29d8bbb11d19b3ed6dba7
data/chat_history_32k/chat_history_250913_163134_persona740.json
     edd68f856da7e784dbb164589777283f4efca1a865e0e4135d399f2da7fd7dd9
data/chat_history_32k/chat_history_250913_163134_persona741.json
     28d0bf63e56fa40d0f283618ae6783dc2115c119c8af4e7393e3e6785f16def6
data/chat_history_32k/chat_history_250913_163134_persona749.json
     934c8645b8285ce4c1c08c11791c6610ee3fb1533e3ff74b07b2649fa331d00a
data/chat_history_32k/chat_history_250913_163134_persona753.json
     98b0dfb6de4a19ffb375b55d96269f63f7547e8cae5cd87ca0db68fc0f2b9013
data/chat_history_32k/chat_history_250913_163134_persona76.json
     2c45cad263251aa99b69ce4cda64d39961ef1959c1893e730413604f2d791a2c
data/chat_history_32k/chat_history_250913_163134_persona761.json
     3d27975a85138a9d37a5dd85331147a3930436d28cd2790e3c7c921f67f27ad9
data/chat_history_32k/chat_history_250913_163134_persona764.json
     5efd44982fbf4b05dc74a68bf91baabb450a210666049b607ffca9cc81649750
data/chat_history_32k/chat_history_250913_163134_persona767.json
     07b9c9448a3d4e7ed0f643fabf1d0d6b388e9588b7c1b3a87f7ed06f66ce5b33
data/chat_history_32k/chat_history_250913_163134_persona78.json
     5cbde0878d0aae3172bf15982b51b8398b9adba6475d352b3c616a4888728e90
data/chat_history_32k/chat_history_250913_163134_persona787.json
     7f9b168eb82b5897c9fe4646f7c33691613e839228e25d6dd28b6d0ada1c8700
data/chat_history_32k/chat_history_250913_163134_persona800.json
     0414f852565ea5cfd9d321f0926b79787b63a2f1be7acbebc49a00a9b7ce52d3
data/chat_history_32k/chat_history_250913_163134_persona802.json
     54c39b6aadc1b8b2e7ea9871a0569769a3db8ae9bad0e6fe3456e80d7663861a
data/chat_history_32k/chat_history_250913_163134_persona810.json
     23dcc2c431d3f70d2bd70187d3babbda2b8b4f1bc039f3be644d81d42e6dc156
data/chat_history_32k/chat_history_250913_163134_persona811.json
     30c7d8c32740721b14c662b2c7dd59103ed5a4d6f8d07727ffdbcecdda33aeee
data/chat_history_32k/chat_history_250913_163134_persona816.json
     b10e22b82957afc629fa4ea10105912d1b7800bc11e245e0d1d0f31c001b548c
data/chat_history_32k/chat_history_250913_163134_persona820.json
     f5c95d8406d466f9150f6bc8f4a3848252119589d45aafe4501bc1dad8bfb3de
data/chat_history_32k/chat_history_250913_163134_persona822.json
     7a07c2eb17335c8e8e67921d08facc0a9ecb17ce63005043a777c1341d1b938d
data/chat_history_32k/chat_history_250913_163134_persona823.json
     c9eb2ec2c37da6d149fbe6d87c68e9d2251fa0a7bd22cabbf836da4401f30721
data/chat_history_32k/chat_history_250913_163134_persona826.json
     b972f440904cbaa9ba2b4fd0dd61ccbb74b33b764ff4bab6fe179af2dc0c3a42
data/chat_history_32k/chat_history_250913_163134_persona829.json
     53ab6bb336f28323aa1255c4b2419488894c3713fcb584a820aab37ccb337017
data/chat_history_32k/chat_history_250913_163134_persona837.json
     f0365d16d50a60da6a7cc2fd47db925d62b0687b4f5d48c47109df5f4e24d554
data/chat_history_32k/chat_history_250913_163134_persona845.json
     4987608fada9c2d1320376c50bd58363d2a47f6337c615fe09037232992660f8
data/chat_history_32k/chat_history_250913_163134_persona849.json
     2ba2752a6e915ff8fd054531566d150073ee70ffd6900f8a9213d9019674aba4
data/chat_history_32k/chat_history_250913_163134_persona859.json
     2c7bbe4b57176f0fde3ca39dd47a8c09245374f1492c497cc9ed6ae637b7a2c1
data/chat_history_32k/chat_history_250913_163134_persona86.json
     f993c9f9ed0f775e30c7fe9c05271d2bec7825de1d115e284b3f00986e26442e
data/chat_history_32k/chat_history_250913_163134_persona865.json
     61d7cf9c36b931037a2754951445b999396da053e0f4effe3f39659c51817f89
data/chat_history_32k/chat_history_250913_163134_persona866.json
     27448a5cc7e2d35aa6089ba098a0573c98e57a4a9ac0a96f658604381814167a
data/chat_history_32k/chat_history_250913_163134_persona867.json
     855bf6eb5d78514d9d5bf5e3d64f2f6e9f3169a1a8fcc2c5cb60d2ae8d63563d
data/chat_history_32k/chat_history_250913_163134_persona879.json
     547b16b2ff1ac7746d3751964267ed52ef77a261878c4dce954b06eda52879e4
data/chat_history_32k/chat_history_250913_163134_persona88.json
     8b45f9541d373fa726927e0b352592150820c4feabc7522feb0f181ca00fc03b
data/chat_history_32k/chat_history_250913_163134_persona883.json
     9d24959a8e160f5431af9e0bd84a275acf7db5f109cd6c84145311b47b3c04a7
data/chat_history_32k/chat_history_250913_163134_persona884.json
     f7373a8c3ff0ff90a7fb4bd6f2e6d1fd4b588e95077c461c8b969045dbfec640
data/chat_history_32k/chat_history_250913_163134_persona892.json
     7453cd812d152acdc9ddf44c871a943d0469b63662eb5a86e4156abc6f4c36b5
data/chat_history_32k/chat_history_250913_163134_persona893.json
     d43efd4bb49c380c3b83734f637d9f80a563e2a4109a27e2fa355f345f3fe169
data/chat_history_32k/chat_history_250913_163134_persona899.json
     992245f2830d449424e014c6f5eda999b6b9a106cb4b52e9765dfec2bb345c59
data/chat_history_32k/chat_history_250913_163134_persona901.json
     79faa4457fd159730e26db1e75871f24c0948ba5d803d2839fd7d50afd09973e
data/chat_history_32k/chat_history_250913_163134_persona902.json
     ea959b904c2e8eafa636139dfc120d2712a5b8f7e282bcf7bbf9ac492f75370a
data/chat_history_32k/chat_history_250913_163134_persona914.json
     07eb8af2938d34c8576988bb9d57ac5c3115a862807f4621f479a212aa82843f
data/chat_history_32k/chat_history_250913_163134_persona917.json
     73fd69c46053ea650c338d902a9c20f8aff9e7d622c0c818e9ee6a539e24c72a
data/chat_history_32k/chat_history_250913_163134_persona924.json
     483f64d0766448d4229db9ca307c2134d1bfdc3fdec4c980e8bb017a0525ef37
data/chat_history_32k/chat_history_250913_163134_persona936.json
     19fe2ceec2768b7577c2bdefef17c60b6a20d012920c2b363d4dfa13884f3c6f
data/chat_history_32k/chat_history_250913_163134_persona938.json
     85f2df94a5c3a4f75c9306c20d3de44ff8fd9f06462b99e1a7f9170c1a6239be
data/chat_history_32k/chat_history_250913_163134_persona941.json
     fffa7383967b6ebb1cd84beb0f49ad1c12bf0e7e2f81633bf565d0675cbab3ab
data/chat_history_32k/chat_history_250913_163134_persona942.json
     fdd1d6e7ed6c287b98b612ebf4c9295cb9960a7cd6a84149d78671df1b21f421
data/chat_history_32k/chat_history_250913_163134_persona947.json
     5a59c71bbb98ff9097f895906edb065296ec7f3af776d5d7e9b09561e03f3ec7
data/chat_history_32k/chat_history_250913_163134_persona948.json
     fc1780760640b05e4ff0fa773416e90adcb7e39ed716772a3896df58520eafe8
data/chat_history_32k/chat_history_250913_163134_persona959.json
     0ce22e03475d7eedc041294d6983e8b7702e9c5a72e8ed3fa09c31beeb0a533e
data/chat_history_32k/chat_history_250913_163134_persona96.json
     e1ed3a0390d58f16fb64a1346d38d473a0e0cf075cf0ab2e22447a36ce88213d
data/chat_history_32k/chat_history_250913_163134_persona973.json
     c8943bd3d9691e885d8de2bc1663154c1f2a8c6e5035c4079719fb939770c394
data/chat_history_32k/chat_history_250913_163134_persona974.json
     aa21a261aad743de8f7505056d6a36716e2516332852c6ca460c74edc0b8cb33
data/chat_history_32k/chat_history_250913_163134_persona977.json
     631c9c50b1073cae1429f440dd2878f63bd371ef04c6092dabd4617cd705e2c5
data/chat_history_32k/chat_history_250913_163134_persona978.json
     f3df523ba9a67e71ab9c72bc4762f6550535f44f6765fdcba607aa6bb6aa8ce0
data/chat_history_32k/chat_history_250913_163134_persona985.json
     f3182201d5c7d7a4e996ef2f793d8b627af42c907ccb4e632d8add0d2f5beb53
data/chat_history_32k/chat_history_250913_163134_persona986.json
     7f5cfc7c6ac2be3612071bccde4d3a474524f860690a821303a11bd5a63612b3
data/chat_history_32k/chat_history_250913_163134_persona989.json
     7732ab4d1fd38c0acca0856092382a4642fa7c148c85e54914b5d1110c193081
data/chat_history_32k/chat_history_250913_163134_persona995.json
     7732d4a78fd9744cbb5f4d6cdf145df4868a6abdf04d344080561f322351a0ab
data/chat_history_32k/chat_history_250913_163134_persona998.json
     4e17c2aac93237670416911c4876ea41b834f8d34a9e388a7d78929733528374
""",
    "tempreason": """
README.md
    fdfc9d374bd8b3b5fdc4b39c68d543af598631c2869a45778b9951e2dd9e520a
test_l1.json
    4c85d596df6d03a68a08faf23a9079c4bf72ffa2238672445699a2e994011bff
test_l1_future.json
    455cb9955be6061a867fb2b29a4f971c8393e3d72383bfdd3e991914b8fe1077
test_l2.json
    9292d14519e34441c43efda9a0739a27a498665039f0e2c354c3dcde709c9021
test_l3.json
    b6fb4468d5accab156b2d2481c27d0860ea54f8604045a172f42c5e36978d7dd
""",
    "corporatebench": """
LICENSE
    c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
README.md
    6c15299ed5271909084d1bbc4b6d2e7bc018aa5b9bc465b1b83ecfcbfffc5359
data/integrated_qa/zenith_questions.json
    b0fb16b3ef1fb018bb15ccb02331a656e2143b188b836afb2a7954b67053c50d
data/kb/zenith.kb
    420763ea2c34c4b5b1125e424126aa1e7ac2c82704774ada8122aeef59228b99
data/kb_qa/zenith_questions.json
    896d415484ebcc5ffcbb3687c6cffcea79ea91c7ed39810076027c55704955c8
data/native_eml/zenith.zip
    be90c13c858a7834ff7a24b85425e7bf1d40bdd7efc8d7dde0e6457028f384a0
data/topic_qa/zenith_questions.json
    26452df4fe2366957faef8d47caab2576e606eb6579383ec5c0cccc4cb65558f
""",
    "medmemorybench": """
README.md
    8e3fca4e95db689fe643562b944b9e417413d0a214476ad896e68a9c11339464
data/zh/clinical_reports.parquet
    ce6d42886e3bd613aa0c1bdb8629c33ed7f8d9ad8af1aaf2e4afc245b92fc02f
data/zh/dialogues.parquet
    705d19c47c45ed0a0774e6fd3f77fdd0fd2ffe74d1d4857a5f02b0e7748df9c7
data/zh/events.parquet
    c41e01c7b3a7e226a69d5a73ef5dbcfbc269dff970bedd2eaaadb471f64c5175
data/zh/personas.parquet
    51e4f4ef468589caa55d0f9a20df08a5d90bb45f4c329ded38ab72ce4c405714
data/zh/queries.parquet
    46ece36c4fc024dac106ce6cf44b57663b48133bad589b13916b202cf8fbb4a0
data/zh/trap_events.parquet
    368a6c552983f07dc973d035bbd98cb776d4d1fecf48786ea9c94e23903c7f6b
""",
    "memtrapbench": """
AdaptiveMem/skill.txt
    cc279ded25d30649c55ee2a62866bf41537d2aa58d2475748ddd6da27b60b4ad
README.md
    c7c234973d162903722d704dd3d5c9afae0fecb5fa803cc0aa6254f2f36452f2
case/case_aggregate_memory_final_responses.md
    0917413a5714a5e96972a49ad1b03d9771510380ac31ad2beada05167f0b910c
case/case_study_hurt_id0004.txt
    f40bbac042eb54a969a085c686f3d7bfb3ca45ad57536aec1e4c072505f8c0f5
case/case_study_inertia_id0008.txt
    7fc1195ca97627795f601dd5119e6b5f21feb767591706072f17dbae3edef254
case/case_study_number_game_id0123.txt
    009f918397f51c556a90110061fd76c1d99a4109a20eb26a7de391859ec63b84
case/case_study_poison_id0004.txt
    b3928de8f2589911884ac3b2fa92413f83c1d56058873b68b2a31205f337aa3c
case/case_study_unclear_id0010.txt
    025b3e21f9def1afc3b628a119cf1ca6445509707a30166c3d64ab2c6d3f748e
case/hallucination.json
    d6921621fcc81eb638329f45ccd79ed02df79ae4b9a19d1761f77be2ba2afb08
case/hurt.json
    e60ec486a98c64d1dc91499f7bd1d7af47d090848b4b972beb55a9b189dcbd96
case/inerti.json
    eee99fd71fe04bd83bef5711a7e0cea2160afef43a86978a682c406b3193a693
case/number_game.json
    aea7306a1dc578877a5da6144afeb5fdd88e40bc61892fc633bc5997e2eff9fd
case/poison.json
    e60ec486a98c64d1dc91499f7bd1d7af47d090848b4b972beb55a9b189dcbd96
case/unclear.json
    98a8cbba80b080ea2358e3cda5a14f6980566dbf56013785ec1dea325b3b471b
construct_data/benchmark_construction.tex
    8002985b72a5425e9941000913df4dc81399cf152941ae3814f104972594d2e6
construct_data/cognitive_bias/inertia_build_benchmark.txt
    68426874aed9337d24d2d8cef95bf15405ea920c223cdb682ed6c10dfe625076
construct_data/safety/poison_build_benchmark_green.txt
    c512351d37f96b560849d5ce634e6babd51a7558e52951bc0f6755d17e2c44b5
construct_data/safety/poison_build_benchmark_red.txt
    7b4411db77a89a2ea74fa24be99a558205bbe432dbcb35a45e4a45c3db5a1991
construct_data/task_boundary/hallucination_build_benchmark.txt
    c12d6a25a0ba7894183f61006bdd51de6b1c14034eb47c4f4d3a92d5a36d6fa7
construct_data/task_boundary/unclear_build_benchmark.txt
    df890211444d03ba64de44a55b27fe8cef30aa3d0b68e4b7ce6f11f5411cf21e
construct_data/trauma/hurt_build_benchmark.txt
    8369f62acd6db243323b2d26e17963ac8f084b70ce9b6a55d41be7650ea03a49
docs/case_study.png
    0160e6e8e264f538ac1f89fbec0ca1700e2ad2625599ef0e2c0a73ab2424dd04
docs/data_construction.pdf
    3a19bd30113725c4e60058b107e200eeb2a0a7c893784650547d6069546e153e
docs/data_construction.png
    86ec8f540fe36abdd79af6c2593ff6c5d4ac70a44206ea0078da1b99cdb153fb
docs/data_construction.svg
    90ed433156703d78c1f3542f8bfb7662f7ce6d4bf727649252a1da6d6f6f52e7
docs/memtrapbench_logo.png
    06c9b923a74dc186f83b0ab839a2b72c84e8eb57aee042b6dc4df60eabdd7f1f
eval/cognitive_bias/judge_prompt_per_dimension_number_game.txt
    85615b649b007536277d96c0ca767a2cfb60d6db024dd8e1609bdcb7c340f040
eval/safety/judge_prompt_per_dimension_poison.txt
    eddba38a651e6a1c3f77b7138412f988d7d4bc5b2011e4d761e678856d625795
eval/shared/judge_prompt_per_dimension.txt
    1c06c9384c93f9cba2ef29a1c1469786e303697561f3b1f7ee6ff46e28e443f7
eval/shared/prompt_user_mem.txt
    c3d120751d0a3d7fda595a5dd56b67cf6b4672dc9f34d30dcc6a7ef4206b357f
eval/shared/prompt_user_with.txt
    2dbb2222a7ea4111708de6fbf1cc0f28ded607c9681ffc868de0d132d5e82d77
eval/shared/prompt_user_without.txt
    45acd647fdd66924a864f90c43a8a8601bf9cac018477be3024eec64eb6cb23f
memtrapbench/cognitive_bias/Inertia_150.json
    ffef4f2c5c1fddbfe50c725f2b209a7ecde09213dc6522b492b19a3f12f64905
memtrapbench/cognitive_bias/inertia_seed.json
    1f9728c49891b825851b9563d95aea9722c364658c4016e523664d7974d27fa3
memtrapbench/cognitive_bias/number_game_200.json
    66ab441d4ae2f5d1c6bf3cbd95b0d58a02ec703374afaf749638c32c9e936f3e
memtrapbench/safety/poison_200.json
    548fcd4d3b8d1f250f1811d868095e3556d17cbf66a56fef073277680f98db2e
memtrapbench/task_boundary/hallucination_150.json
    5b1e846bfb7cc04424272473323f40da79750fbdce5d48a4edb412e5e83f6da7
memtrapbench/task_boundary/hallucination_seed.json
    9a26a1e2d499c91080dbeab791d156259104db6fdec54774ccd23e305c85ce8b
memtrapbench/task_boundary/unclear_200.json
    5e83916af4c391bea2a69f0548c8db9da85cde1abd4e8c9b4045afe8e0d7f7a4
memtrapbench/task_boundary/unclear_seed.json
    960e5cbd2d4e7a9b9be437b0ba82e62a5a532d0d4a237364c7e9d481e227ac73
memtrapbench/trauma/hurt.json
    53eb51379e1435f2ad031091b3ca2f89197d02d5bea0961f1eb84d3163bd79c7
memtrapbench/trauma/hurt_150.json
    cfd827bcce603d8a528e06dbd782f578b9c852ebf2607e7cee206b760b20bf0a
requirements.txt
    7358297a02c639146ee2a6f468dda26d443f35b988aca520d3c79ee96fef4819
runners/eval/eval_benchmark.py
    7c9c2449c9f06a0d8f5f609cc09d04a1ef2e6d32032e96cf2ce3e4fd4929cd55
runners/eval/eval_common.py
    bbe5299f2de9c7f28a7b8254cbcb4f7dee520d9877eaa8e632f0080d5e7d2ddc
runners/eval/eval_no_memory.py
    243fb1b1926b0cff0316bb16b67ced1a8e01a0635b56760363a01a56ac4d53b7
runners/eval/eval_retrieved_memory.py
    b41ff711453b7879504c828fdc28755c81195a79ea85dc0b1bcf150db725fc59
runners/eval/eval_with_memory.py
    39951efc3b10aa28a03f490331a4d7aaf310e6c7392bef01018c73b05280541d
runners/extract/summarize_judge_scores.py
    4c50609e0851d4a86f037488947f667a2c8b2696005a9c002ce2ab7a1616ab58
""",
    "medmemorybench-code": """
README.md
    eb26d205013fc3ebd2f22f5935c00e0723e574b7303e9c28247a745318eabf3b
benchmarks/base.py
    dd75d2e9d4fa0330bdab420f460a5400f13368a827e3efef4c25a6800803aff9
benchmarks/medmemorybench/dataset.py
    e2b98b283c88d99fe4327d3734b184d1421d6058c8cf122909aa03ed021858ab
benchmarks/medmemorybench/evaluator.py
    1c50d9e6c64ea0cd4ce4ba76d329daeee7790f2f484738038cb1a546aca5b2f0
configs/dataset_config/medmemorybench.yaml
    053d4419d9d792b37480b9d8bd8ef7a9098b142ba04ea1f488e64a73e6c1dfae
metrics/__init__.py
    c26aba132416bf732c0f7ef4bbfdd1e6ecaea623490b318e082b61a023c010a7
metrics/base.py
    ba6d073cb8b981a7ab5d10049de128911bf889cc529cae32f22df7be6b19a1cf
metrics/llm_judge.py
    3b8457c892961fdcb7b6b5e9905f56c05059d5d9fe17f947710def394e436b34
metrics/string_match.py
    0ed9fa71551c4d25f680867e331a62891d8d79411052acc337ef3a071717aeeb
utils/__init__.py
    5eda29eca1564f8ae4f6c4f062ec507417cdbd3fa157a8ace5dac285485f974c
utils/prompts_judge.py
    d8834ff85d19f635335bc36d55d015ddfbfc15df0b1386310dc53aac98c4b88a
utils/templates.py
    f62d83c8dbb9dae58bac500d5ba1017e84d445eb67f7f322cb4d2b90c56f8004
""",
    "doc-pp": """
README.md
    402136b1bf55d371e9d9a62e01d4a7f386b2a901f056664d252d1fdc90ddc6ec
assets/ACL 2025_Find-1297-slide.pdf
    dffc880ac94e25df40180df17c2cca67222dde93eccb2537d1e62dfbf22841bf
assets/ACL 2025_Find-poster.png
    113e45aae1570a8730e5c70868abf660093805486152a347ffec8c0a9f355069
assets/fig_doc-pp_intro.png
    1e2709dc10f67c147ae317aca745641d357c4ee10e83dce724531b5e29087b67
data.z01
    9549c68468e1aca8cf5b76dbd2acffc3b608c634d91b08316660cf7d02bf1a8f
data.z02
    d465562fed8733f75dcaf1a5a78468886cd9b62d7cc2b869f5df48aa042b5b7a
data.z03
    9e306b1abb8d450a982afc6955f2e878f0030b3764326c2d85c2e520f50a561d
data.z04
    1115b4ffc4c5d76a02ef4f5e1c67cd58e9209b3076f3256acb6537944bfd216b
data.z05
    5fa2b6e4b49f45b07ac28583330d9ea252277f9cb18903d9899cd71f76ec453e
data.z06
    b888919b209d942e88e392d8a08a06b7678f4f194a324b4ac95fa7656ef3f756
data.zip
    57558c34c3cf54c32afd4611b3df66a4bcd1e7452ee937563c7919972f575319
prompts/check_initial_faithfulness_checklists.py
    e884f04d3d3912980273d9fe5defdeeb81700da6a4a848f60208f9e5c31a3032
prompts/evaluate_model.py
    667c9e281fcc31633beacda7d9770ab59821771e68c52620aff75053b7b19d4d
prompts/generate_initial_faithfulness_checklists.py
    ac49b97afece6ae4c24b6675bddd06be8ec538466eb5edcf30310e6ae5dccce7
prompts/judge_evaluation.py
    68c7d077aff17ceaf587567ebe1c096a2e7087afb3d49c88b9cbbd2d80d924b5
requirements.txt
    652fcc2fe5922c8cce63880eb740d6c85c55fa7b51acde38644e4b0f3671cc49
scripts/02_evaluate_model.sh
    6a866f86da1fab776b0248bd52337615fd881f60e10061681f4f745d5e45d8bc
scripts/03_judge_evaluation.sh
    fc6ba4d7e4c71d4a9e90e9aaeb3e3678b2f6ad188eef16ca6493a62ae5bb1ee3
scripts/04_mitigation_cot.sh
    debe3c7547da3701fedb59f03d67e8ccc2aaa8524a4cb7a78079ff4e41259832
scripts/04_mitigation_dva.sh
    61f735b949101cf0a869300ec8bc9483d32954d83976ddbf69d3adddd9101be4
scripts/04_mitigation_revision.sh
    7ed56da9d89e178d2d489b254dd9f72a9363219985bfdfb192afbc8b2b6766f4
src/00_base_openrouter_call.py
    975e9cf33b55235f73526c60d65a1a2abf012fb668e0edd6ff630771d2528f20
src/00_generate_initial_faithfulness_checklists.py
    b6120390102accb441cdea7ef86aa362071f02a209b8462b810761a1a4311336
src/01_check_initial_faithfulness_checklists.py
    0f0bb4596b9ca17c3d6acf981a53ba3cd87c23572ead777945cc8da196f8139c
src/02_evaluate_model.py
    3ba193c33d460add8fcc8215442031200d7a19d94649d9f63f5de30f81a7124f
src/02_evaluate_model_provider_fixed.py
    429e3a8112beb57464f8b09e55c0b5293ccba610d8206408a7b597614cb8c9f0
src/02_length_ablation.py
    242a8e2401ff9033a48d366c678012b2918bf72403ee8f2e4f5d332d1adc15ca
src/02_test_evaluate_model.py
    b3c9bbd2d182b8d24a8fcfae09915b050bace309e82a36e45d4aea389a8d78d3
src/03_judge_evaluation.py
    ea7ac4fb6366f4fefa8245d2551e1895a4d682226a204f5436a0daffbdea6755
src/04_mitigation_cot_judge.py
    1705c959893437090686d930fd1ba9c725d687958cf0697e5b8e0eebb52324b2
src/04_mitigation_cot_provider_fixed.py
    8243e5b12cdca712ed875afb83c61219d59a6bd7c60fc464d5046effc935336e
src/04_mitigation_dva_judge.py
    c4dee3fae52ba915b0080f05d32cf31ab08fddb3d54c57ec4df2c484550b8573
src/04_mitigation_dva_provider_fixed.py
    7b3c0bdea87da1120f97b44f76d33e47b76a8394701cd016f74285f43a04d240
src/04_mitigation_revision_judge.py
    fee487fdf01182248ad7f3095e20ef72e5efc82d47347783baa4802629d56680
src/04_mitigation_revision_provider_fixed.py
    85c3240d7cace48d07c17b5ee91cd5f8de455135f3f77bbd7969d859a55b2bd0
src/decode_test_output.py
    374bddc4a996e23529f65e58c8769a295d598a90c4678b4caa22c02b95ee5c4e
""",
    "halumem": """
README.md
    f079d020d55c744ec43992d15e48b4e03a75c7f7c9a3544a1f621d39e5fc3b84
HaluMem-Medium.jsonl
    486fbc130a5c8781a2af27ffa508a1d7855245137aa449c193ac4d29c45634e7
https://raw.githubusercontent.com/MemTensor/HaluMem/718f16ff0c83413b1c86fa83fc13cc1a639871f9/eval/eval_tools.py
    0c08e5ecb8c93945bafc4bd0336bd6c9756b40d175f442ce44aca4a43169ee3b
""",
    "hybridqa": """
README.md
    f9a7981378ad0edad312220240871af8f6782d6ce94a7b32e0a23b1d1ce7fd41
https://raw.githubusercontent.com/wenhuchen/HybridQA/db22fda8c5951438fade3c69d75b350335ba93b3/released_data/dev.json
    424272b233735a70ed8ef5af4a615373d114f472168c686c4370d54c92d58ac1
https://raw.githubusercontent.com/wenhuchen/HybridQA/db22fda8c5951438fade3c69d75b350335ba93b3/released_data/dev_reference.json
    617cd141a09550e85a0634b68b2e16727a87f9a6005002d31f40d839dfa389ed
""",
    "musique": """
https://raw.githubusercontent.com/StonyBrookNLP/musique/922ac98f19a201998dbdae6d7f2887a5258dbdeb/README.md
    7a3fb6a960f3fa122d2a1a81b73b9cf5a39ac983b11735d158490947960b9359
musique_ans_v1.0_dev.jsonl
    15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b
musique_full_v1.0_dev.jsonl
    8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a
""",
    "feverous": """
https://raw.githubusercontent.com/Raldir/FEVEROUS/32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6/README.md
    87a67df3fbf3b3dd696dfce724a2b0f8570d06dd21eefab3b15a0f3938b317dd
feverous_dev_challenges.jsonl
    1ac8cfd964d4dcedc5de3375850fe3f93d39b89a73a475734bde864da1701f8f
""",
}

_OFFICIAL_EXTRA_TPL = {
    "hf": "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}",
    "gh": "https://raw.githubusercontent.com/{repo}/{rev}/{path}",
    #: FEVEROUS 没有仓内分发点（官方站点直接放文件），rev 为空 ⇒ `{rev}/` 不出现。
    "fever": "https://fever.ai/download/feverous/{path}",
}


def _parse_file_table(block: str) -> dict[str, str]:
    """紧凑表 → `{路径: sha256}`：**一条两行**（路径 + 缩进的哈希），`strip` 后成对读。"""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    table = {}
    for i in range(0, len(lines), 2):
        rel, digest = lines[i], lines[i + 1]
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"表里这行不是 sha256：{digest!r}"
        table[rel] = digest
    return table


def _expand_official_extra() -> list[dict[str, str]]:
    """紧凑表 → MANIFEST 的逐文件条目（`tier = official-extra`）。

    ⚠ 表里的路径可以写成**整条 URL**（以 `http` 开头）：有两个数据集的数据在 HF 镜像上、
    而 README（许可口径）只在**上游仓**里——一个 slug 一个 host 的表达力不够，
    又不想为一个 README 新开一个 slug。落点仍按 `{slug}/{文件名}`。
    """
    entries = []
    for slug, meta in _OFFICIAL_EXTRA_META.items():
        template = _OFFICIAL_EXTRA_TPL[meta["host"]]
        for rel, digest in _parse_file_table(_OFFICIAL_EXTRA_FILES[slug]).items():
            if rel.startswith("http"):
                name, url = f"{slug}/{rel.rsplit('/', 1)[-1]}", rel
            else:
                name = f"{slug}/{rel}"
                url = template.format(repo=meta["repo"], rev=meta["rev"], path=rel)
            entries.append(
                {
                    "name": name,
                    "tier": "official-extra",
                    "url": url,
                    "sha256": digest,
                    "note": meta["note"],
                }
            )
    return entries


MANIFEST.extend(_expand_official_extra())

# 配套语料及评分函数：固定 revision；下载包保持原字节，解包规则显式声明。
MANIFEST.extend(
    [
        {
            "name": "hybridqa/wiki_tables_with_links.tar.gz",
            "tier": "official-extra",
            "url": f"{HYBRID_CORPUS_REPO}/tar.gz/{HYBRID_CORPUS_REV}",
            "sha256": "df9a79f5080ec5372af6f4133f4cb95c9381c620fd9634b4d11808a069bc7036",
            "archive_root": f"WikiTables-WithLinks-{HYBRID_CORPUS_REV}",
            "note": (
                "官方 HybridQA README 指向的表格/关联段落语料；"
                "只提取 dev 引用的完整 table/request 对"
            ),
        },
        {
            "name": "hybridqa/evaluate_script.py",
            "tier": "official-extra",
            "url": "https://raw.githubusercontent.com/wenhuchen/HybridQA/db22fda8c5951438fade3c69d75b350335ba93b3/evaluate_script.py",
            "sha256": "80621c80d343bed345b914c2124185e013171ab4e7a6fbe54c6dfc8895b9ddb4",
            "note": "上游 answer EM/token-F1 函数；只加载函数与标准库 import，不执行脚本 CLI",
        },
        {
            "name": "hybridqa/WikiTables-WithLinks.README.md",
            "tier": "official-extra",
            "url": "https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/dc066e1a6d5281511d8b73a6107d5ad2824cc2b2/README.md",
            "sha256": "a29dbbea518ddc9e6840464e6fa6e12a93237d65ebc39f956a517db31bf1c9bf",
            "note": "语料 schema 与版本说明；不使用已失效的 HybridQA S3 预处理包",
        },
        {
            "name": "feverous/feverous_scorer.py",
            "tier": "official-extra",
            "url": "https://raw.githubusercontent.com/Raldir/FEVEROUS/32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6/src/feverous/evaluation/feverous_scorer.py",
            "sha256": "e09104e3a28fe0950f4e4f61adeec70319432dfb57ca261f5e556bd48f9e99a2",
            "note": "上游标签/完整证据组评分函数；NEI 同样要求证据组，按函数行为执行",
        },
        {
            "name": "feverous/download_data.sh",
            "tier": "official-extra",
            "url": "https://raw.githubusercontent.com/Raldir/FEVEROUS/32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6/download_data.sh",
            "sha256": "d42c2ab8100d5e2ba454445346e7e579c1111acef71e3d8b84e03ca02317d03e",
            "note": "官方语料下载来源说明；本仓不直接执行该 shell 脚本",
        },
        {
            "name": "feverous/feverous_wikiv1.db",
            "tier": "official-extra",
            "url": "https://fever.ai/download/feverous/feverous-wiki-pages-db.zip",
            "sha256": "a980581f55d46a252090b29269954503735b6f00274d05225476a650ab940276",
            "archive_member": "feverous_wikiv1.db",
            "bytes": "53486538752",
            "download_bytes": "10353775701",
            "download_sha256": "e25e034d9848c75ab3311a7a7ad8e80e769240b5a36b055da475f20071314881",
            "note": (
                "官方 SQLite 全语料；压缩包 MD5 与 Zenodo 4911508 的 "
                "b01271df477efdfca99441c7dc6fe859 一致"
            ),
        },
    ]
)

# ── 不入档（归档里刻意不收的）──────────────────────────────────
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
