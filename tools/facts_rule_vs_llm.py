"""Rule vs LLM fact layers, grounded-evidence branch only (evaluation tool).

Four stages, each resumable:

```text
sample   从已归档的语料库 + v1.1.x 归档题单里挑一份跨数据集样本
index    给样本里的每个用户建两套事实库：规则抽取 / LLM 抽取（同一 qa_pairs）
run      逐题只跑共同事实分支：确定性编译 → （失败才）LLM planner → 执行器
report   命中率（非空选择）与精确率（命中题里证据含金标答案）
```

⛔ 这不是产品代码：模型只写**离线实验缓存**，事实落进 `var/` 下的临时库，不进真源。
事实的引句仍按产品口径校验（逐字出现在来源里），否则写入会被 store 拒绝。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fact_llm_gate import _trigger_supported  # noqa: E402

from tianximem.facts import evidence as ev  # noqa: E402
from tianximem.facts.evidence import EVIDENCE_VERSION, key, make_fact, relation_key  # noqa: E402
from tianximem.facts.grammar import _INTERVAL, _REPLACEMENT, month_index  # noqa: E402
from tianximem.facts.paths import RELATION_NAMES  # noqa: E402
from tianximem.facts.query import compile_query  # noqa: E402
from tianximem.llm.openai_chat import OpenAiCompatChat  # noqa: E402
from tianximem.pairing.apply import index_pair_facts  # noqa: E402
from tianximem.service.grounded_planner import GroundedPlanner  # noqa: E402
from tianximem.service.search_stages import GroundedEvidenceStage  # noqa: E402
from tianximem.store.sqlite_store import QaPair, SqliteStore  # noqa: E402

ARTIFACTS = ROOT / "eval/reports/runs/version-native-24h-20261010/artifacts/runs"
#: CorporateBench 不在本轮：它的 LLM 抽取已在 answer-targeted-50 那一节单独做过，
#: 这里只跑其余十二个数据集。
DATASETS = (
    ("locomo-refined", None),
    ("longmemeval-s", None),
    ("halumem", None),
    ("musique", None),
    ("mquake-remastered", None),
    ("tempreason", None),
    ("hybridqa", None),
    ("medmemorybench", None),
    ("memtrapbench", None),
    ("clbench", None),
    ("personamem-v2", None),
    ("beam", None),
)

#: 允许 LLM 断言的规范化关系名（查询编译器能查到的那些）——别名表的值 + 名词表。
ALLOWED_RELATIONS = frozenset(
    set(RELATION_NAMES) | {v for v in ev._ALIASES.values() if isinstance(v, str)}
)

#: 这些数据集在题单里不带可比对的金标（裁判口径不是"答案串"），只进命中率分母。
#: ⚠ **HaluMem 已移出**（2026-10-17）：它的 `gold_answer` 是可比对的答案串
#: （`{'answer': '1983-05-06'}`），此前把它当不可测会让 14 个正确命中全被记成 "-"。
_UNMEASURABLE = {"beam", "clbench", "personamem-v2", "memtrapbench"}


#: ⛔ **不要给关系词表**：2026-10-16 实测这个模型一看到"allowed labels"就一律返回空
#: （0 词表 3s 抽对，15 词表 13s 返回 `[]`）。关系名由模型自由写，规范化交给本地映射。
#: ⚠ **2026-10-18 改写**：旧提示词（`Extract every …` + `Return [] if nothing is stated.`）
#: 在**对话/邮件**类记录上一条都不抽，只在 `Corpus: A — r — B` 上有效——实测 24 条记录
#: **0 条事实**。逐行二分把触发点定位到标题句与规则 9 的"返回空"指令同时在场；改写后
#: 同一批记录 **22/24 非空、93 条事实**。判定口径见报告
#: [`../eval/reports/common-facts-rule-vs-llm-20261016.md`](../eval/reports/common-facts-rule-vs-llm-20261016.md) §十五。
EXTRACT_PROMPT = """Extract subject-relation-object triples from the record as a JSON array of {{"subject","relation","object"}}.

Rules:
1. One relation per sentence or labeled field; subject and object must both appear in it.
2. Plans, questions, negations and quoted third parties are not present facts.
3. Inside a named speaker's turn, "I"/"my" refers to that speaker.
4. Use the record's wording for the relation.
5. A statement ending in "(this replaces the earlier value)" states the NEW value: the
   subject and object are the two sides of the sentence, e.g. "The name of the current
   head of state in United States of America is Joe Biden (this replaces the earlier
   value)" -> {"subject":"United States of America","relation":"head of state","object":"Joe Biden"}.
6. A record shaped "A - r - B" states subject A, relation r, object B.
7. If the record states a date of birth, emit relation "born on" with the date as the object.

RECORD:
{{record}}
"""

#: 模型自由写的关系名 → 规范化关系。映射靠"许可触发词"（同 `fact_llm_gate` 的那张表）。
_RELATION_BY_TRIGGER = {}


def canonical_relation(scope: str, subject: str, object_: str) -> str | None:
    """free-form 关系名落不到词表时，按触发词把 (scope, subject, object) 映射到规范关系。"""
    if not _RELATION_BY_TRIGGER:
        from tools.fact_llm_gate import _TRIGGERS

        for relation, triggers in _TRIGGERS.items():
            if relation not in VOCABULARY:
                continue
            for trigger in triggers:
                _RELATION_BY_TRIGGER.setdefault(trigger, relation)
    folded = scope.casefold()
    for trigger, relation in sorted(_RELATION_BY_TRIGGER.items(), key=lambda kv: -len(kv[0])):
        index = folded.find(trigger)
        while index >= 0:
            tail = folded[index + len(trigger) : index + len(trigger) + 180]
            if object_.casefold() in tail:
                return relation
            index = folded.find(trigger, index + 1)
    return None


_SPEAKER_LINE = re.compile(r"(?m)^\s*(?:\[[^\]]{1,20}\]\s*)?[A-Za-z][A-Za-z0-9 _.'’-]{0,40}:\s*\S")


def _turn_scopes(record: str) -> list[str]:
    """按**说话人轮次**切：一轮里跨句子的本人陈述仍属于同一个说话人。

    `fact_llm_gate._scopes` 只在"≥2 个说话人行"时按轮次切，单说话人的记录会退化成
    句子级——那样 "Jon: ... Lost my job as a banker" 里的 "Jon" 与 "banker" 就不同范围了。
    """
    starts = [match.start() for match in _SPEAKER_LINE.finditer(record)]
    if not starts:
        from tools.fact_llm_gate import _scopes

        return list(_scopes(record))
    turns = [record[start:] for start in starts]
    return [record[: starts[0]]] + turns if starts[0] else turns


def local_scope(record: str, subject: str, object_: str) -> str | None:
    """含 subject 与 object 的最小局部范围（说话人轮次 → 段落 → 句子）。"""
    key_subject, key_object = key(subject), key(object_)
    for scope in _turn_scopes(record):
        folded = key(scope)
        if key_subject in folded and key_object in folded:
            return scope.strip()
    from tools.fact_llm_gate import _scopes

    for scope in _scopes(record):
        folded = key(scope)
        if key_subject in folded and key_object in folded:
            return scope.strip()
    return None


#: 自由关系名 → 规范关系：**只收规范词表里已有的关系**，按观察到的动词归并
#: （"went to a fair" 与规则层的 `participated` 是同一件事）。落不到这里的
#: （`showed off` / `use for` / `made` …）就是词表里没有的关系，**不应该**硬塞。
_LABEL_SYNONYMS = {
    "went to": "participated",
    "went on": "participated",
    "competed in": "participated",
    "took part in": "participated",
    "took home first in": "participated",
    "took first in": "participated",
    "participating in": "participated",
    "volunteering at": "volunteer",
    "volunteering as": "volunteer",
    "volunteers at": "volunteer",
    "accepted for": "accepted",
    "accepted to": "accepted",
    "got accepted for": "accepted",
    "moved from": "moved from",
    "moved to": "live",
    "lives in": "live",
    "works for": "employee",
    "works at": "employee",
    "worked at": "employee",
    "lost my job at": "employee",
    "lost job at": "employee",
    "lost job as": "occupation",
    "job as": "occupation",
    "studied at": "educated at",
    "attended school at": "educated at",
    "friends with": "friend",
    "made friends with": "friend",
    # 2026-10-17 扩：按"被词表挡掉的关系名"清单补齐（relation_gap 报告里的 top 项）。
    "born on": "date of birth",
    "date of birth": "date of birth",
    "birth date": "date of birth",
    "born in": "place of birth",
    "was born in": "place of birth",
    "birthplace": "place of birth",
    "place of birth": "place of birth",
    "written by": "author",
    "wrote": "author",
    "authored by": "author",
    "directed by": "director / manager",
    "director of": "director / manager",
    "served as": "position",
    "held the position of": "position",
    "holds the position of": "position",
    "married to": "spouse",
    "spouse of": "spouse",
    "son of": "child",
    "daughter of": "child",
    "child of": "child",
    "created by": "creator",
    "creator of": "creator",
    "located": "located in",
    "located in": "located in",
    "is located in": "located in",
    "is in": "located in",
    "located at": "located in",
    "headquartered in": "headquarters location",
    "based in": "headquarters location",
    "formed in": "location of formation",
    "notable work": "notable work",
    "known for": "notable work",
    "performed by": "performer",
    "stars": "performer",
    "starring": "performer",
    "capital of": "capital",
    "founded in": "location of formation",
    "founded by": "founded by",
    "member of": "member",
    "played for": "team",
    "plays for": "team",
    "played in": "team",
    "worked for": "employee",
    "employed by": "employee",
    "employer of": "employee",
    # 2026-10-17 第二批：模型照抄**原文动词短语**（"is a citizen of"），而词表里是名词
    # （`country of citizenship`）。这一批全部来自 MQuAKE 更新句的实测输出。
    "is a citizen of": "country of citizenship",
    "citizen of": "country of citizenship",
    "is a national of": "country of citizenship",
    "nationality": "country of citizenship",
    "is associated with the sport of": "sport",
    "associated with the sport of": "sport",
    "plays the sport of": "sport",
    "is located in the continent of": "continent",
    "located in the continent of": "continent",
    "was created in the country of": "country of origin",
    "created in the country of": "country of origin",
    "is the head of the government of": "head of government",
    "is the head of the state of": "head of state",
    "is the head of": "head of state",
    "speaks the language of": "languages spoken, written or signed",
    "speaks": "languages spoken, written or signed",
    "died in the city of": "place of death",
    "died in": "place of death",
    "is located in the city of": "headquarters location",
    "headquarters is located in": "headquarters location",
    "was founded in the city of": "location of formation",
    "was founded in": "location of formation",
    "was developed by": "developer",
    "was performed by": "performer",
    "was educated at": "educated at",
    "was married to": "spouse",
    "is married to": "spouse",
    "was created by": "creator",
    "was founded by": "founded by",
    "is employed by": "employee",
    "is associated with": "sport",
    "played the position of": "position played on team / speciality",
    "plays the position of": "position played on team / speciality",
    "position played on team / speciality": "position played on team / speciality",
    "official language is": "official language",
    "is the official language of": "official language",
    "head of government": "head of government",
    "head of state": "head of state",
    "headquarters location": "headquarters location",
    "chairperson": "chairperson",
    "chairperson of": "chairperson",
    "chief executive officer of": "chief executive officer",
    "original broadcaster of": "original broadcaster",
    "manufacturer of": "manufacturer",
    "genre": "genre",
    "occupation": "occupation",
    "officeholder": "officeholder",
    "location of formation": "location of formation",
    "place of death": "place of death",
    "developer of": "developer",
    "performer of": "performer",
    "country of origin": "country of origin",
    "country of citizenship": "country of citizenship",
    "capital": "capital",
    "author": "author",
    "creator": "creator",
    "sport": "sport",
    "continent": "continent",
    "spouse": "spouse",
    "child": "child",
    "educated at": "educated at",
    "employer": "employee",
    "employee": "employee",
    "languages spoken, written or signed": "languages spoken, written or signed",
    "is the author of": "author",
    "is affiliated with the religion of": "religion or worldview",
    "works in the field of": "occupation",
    "produced by": "manufacturer",
    "head of the government": "head of government",
}


def _synonym_relation(label: str) -> str | None:
    normalized = key(label)
    if normalized in _LABEL_SYNONYMS:
        return _LABEL_SYNONYMS[normalized]
    for phrase, relation in sorted(_LABEL_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if normalized.startswith(phrase):
            return relation
    return None


#: **前缀式更新句**的本地确定性拆解（2026-10-17）。模型在这类句子上要么返回空、
#: 要么把整个名词短语当主语（`"The capital of Malawi" is Oradea`）——两种都不是事实。
#: 这里按**来源句法**拆（同 `grammar._UPDATE_FORMS` 的思路），拆不出就放弃：
#: 这是模型输出的**本地兜底**，不是第二套抽取器（只在更新句上、只在模型没给对时跑）。
_PREFIX_UPDATE = re.compile(
    r"^The (author|capital|official language|chairperson|chief executive officer|"
    r"original broadcaster|manufacturer|developer|performer|director) of (.+?) is (.+)$",
    re.I,
)
_RELATION_BY_PREFIX = {
    "author": "author",
    "capital": "capital",
    "official language": "official language",
    "chairperson": "chairperson",
    "chief executive officer": "chief executive officer",
    "original broadcaster": "original broadcaster",
    "manufacturer": "manufacturer",
    "developer": "developer",
    "performer": "performer",
    "director": "director / manager",
}


def _local_update_facts(pair_question: str | None) -> list[dict[str, str]]:
    """更新句里模型没拆对时的本地兜底：只处理**前缀式**与「X speaks the language of Y」。"""
    text = pair_question or ""
    if "(this replaces the earlier value)" not in text:
        return []
    body = _REPLACEMENT.sub("", text).strip()
    body = re.sub(r"^Corpus:\s*", "", body, flags=re.I).strip()
    out: list[dict[str, str]] = []
    match = _PREFIX_UPDATE.fullmatch(body)
    if match:
        out.append(
            {
                "subject": match[2].strip(),
                "relation": _RELATION_BY_PREFIX[match[1].casefold()],
                "object": match[3].strip(),
            }
        )
        return out
    match = re.fullmatch(r"(.+?) speaks the language of (.+)", body, re.I)
    if match:
        out.append(
            {
                "subject": match[1].strip(),
                "relation": "languages spoken, written or signed",
                "object": match[2].strip(),
            }
        )
    return out


#: 结构化字段块（签名 / 署名 / Corpus 行）不长，且不需要动词就能断言关系——
#: 规则抽取在这几类上就是这么做的，闸门不能只认动词触发词。
_STRUCTURED_SCOPE_LIMIT = 220
#: 记录超过这个长度就切成 scope 逐段喂：整段长文本会让这个模型返回空。
_CHUNK_LIMIT = 1500
_MAX_CHUNKS = 12


#: LLM 臂允许的关系 = 查询编译器认识的 + **规则层实际产出过的** + **gold 集里验证过的关系标签**
#: （公平：两边同一个词表，由 `stage_index` 从规则库与 gold 集补齐）。
VOCABULARY: set[str] = set(ALLOWED_RELATIONS) | {"located in", "date of birth"}

#: 探活过的端点下标；为空表示还没探过（`_chat` 就会轮换全部端点）。
LIVE_ENDPOINTS: list[int] = []


def llm_gate(record: str, subject: str, relation: str, object_: str) -> tuple[bool, str]:
    """本地闸门：同一条记录里的局部范围 + 明确许可（触发词或结构化字段）。"""
    name = relation_key(relation)
    if name not in VOCABULARY:
        return False, "relation-not-allowed"
    scope = local_scope(record, subject, object_)
    if scope is None:
        return False, "cross-record-endpoints"
    if _trigger_supported(scope, relation, object_):
        return True, "trigger"
    if scope.lstrip().startswith("Corpus:"):
        return True, "corpus-row"
    if len(scope) <= _STRUCTURED_SCOPE_LIMIT:
        return True, "structured-field"
    return False, "missing-local-trigger"


def _gold_groups(item: dict[str, Any]) -> list[list[str]] | None:
    gold = item.get("gold_answer")
    if gold is None:
        return None
    if isinstance(gold, str):
        return [[gold]]
    if isinstance(gold, list):
        return [[str(x)] for x in gold if str(x).strip()]
    if isinstance(gold, dict):
        if "answer" in gold:
            answer = gold["answer"]
            if isinstance(answer, list):
                if str(gold.get("answer_type", "")).startswith("list"):
                    return [[str(x) for x in answer]]
                return [[str(x)] for x in answer]
            return [[str(answer)]]
        if "answers" in gold:
            contents = [
                str(a.get("content"))
                for a in gold["answers"]
                if isinstance(a, dict) and a.get("is_correct", True)
            ]
            return [[c] for c in contents if c.strip()]
        if "gold_standard" in gold:
            return [[str(gold["gold_standard"])]]
    return None


def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values.setdefault(name.strip(), value.strip().strip("\"'"))
    values.update({k: v for k, v in os.environ.items() if v})
    return values


def probe_endpoints(values: dict[str, str]) -> list[int]:
    """探活：只把真的能应答的端点放进 `LIVE_ENDPOINTS`（gw1 会整段时间 502，
    不探活的话一半请求白跑一趟再重试）。返回存活下标。"""
    global LIVE_ENDPOINTS
    live: list[int] = []
    endpoints = _endpoints(values)
    for index in range(len(endpoints)):
        chat = OpenAiCompatChat(
            base_url=endpoints[index][0],
            api_key=endpoints[index][1],
            model=values.get("AML_MODEL", "Qwen/Qwen3.5-9B"),
            timeout=30.0,
        )
        try:
            chat.complete("ping", max_tokens=4, temperature=0.0)
            live.append(index)
        except Exception:  # noqa: BLE001 - 探活失败就跳过
            pass
        finally:
            chat.close()
    LIVE_ENDPOINTS = live
    print(f"live endpoints: {live} of {len(endpoints)}", flush=True)
    return live


def _live_endpoint(values: dict[str, str]) -> int:
    live = probe_endpoints(values)
    return live[0] if live else 0


def _endpoints(values: dict[str, str]) -> list[tuple[str, str]]:
    """两个对话端点都要用：单个网关挂掉是常态（2026-10-16 实测 gw1 502 / gw2 正常）。"""
    pairs = [
        (values.get("AML_BASE_URL", ""), values.get("AML_API_KEY", "")),
        (values.get("AML_BASE_URL_2", ""), values.get("AML_API_KEY_2", "")),
    ]
    return [(b, k) for b, k in pairs if b and k]


def _chat(values: dict[str, str], index: int = 0, timeout: float = 120.0) -> OpenAiCompatChat:
    endpoints = _endpoints(values)
    pool = LIVE_ENDPOINTS or list(range(len(endpoints)))
    base, api_key = endpoints[pool[index % len(pool)]]
    return OpenAiCompatChat(
        base_url=base,
        api_key=api_key,
        model=values.get("AML_MODEL", "Qwen/Qwen3.5-9B"),
        timeout=timeout,
    )


# ── sample ──────────────────────────────────────────────────────────────


def stage_sample(args: argparse.Namespace) -> int:
    """按"每数据集配对预算 + 目标题量"贪心选用户（先小的），保证每集都有足够题。"""
    budgets = {
        "locomo-refined": (130, 25),
        "longmemeval-s": (130, 8),
        "halumem": (60, 12),
        "musique": (120, 25),
        "mquake-remastered": (120, 25),
        "tempreason": (120, 25),
        "hybridqa": (120, 25),
        "medmemorybench": (100, 25),
        "memtrapbench": (120, 25),
        "clbench": (60, 25),
        "personamem-v2": (100, 25),
        "beam": (100, 25),
    }
    conn = sqlite3.connect(args.truth)
    sizes = dict(conn.execute("select user_id, count(*) from qa_pairs group by user_id"))
    rng = random.Random(args.seed)
    spec: list[dict[str, Any]] = []
    for dataset, fixed_user in DATASETS:
        pair_budget, question_target = budgets.get(dataset, (250, 25))
        folder = ARTIFACTS / f"native24h-20261010-v1.1.1-{dataset}"
        users = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if fixed_user and fixed_user in users:
            order = [fixed_user]
        else:
            order = sorted(users, key=lambda u: sizes.get(u, 0))
            rng.shuffle(order)
            order.sort(key=lambda u: sizes.get(u, 0))
        chosen: list[str] = []
        spent = 0
        questions: list[dict[str, Any]] = []
        for user in order:
            if chosen and (spent + sizes.get(user, 0) > pair_budget or len(questions) >= question_target):
                break
            path = folder / user / "input.jsonl"
            if not path.exists():
                continue
            items = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            usable = [
                item
                for item in items
                if _gold_groups(item) is not None or dataset in _UNMEASURABLE
            ]
            if not usable:
                continue
            kept = usable if len(usable) <= question_target - len(questions) else rng.sample(
                usable, question_target - len(questions)
            )
            chosen.append(user)
            spent += sizes.get(user, 0)
            questions.extend(
                {
                    "qid": f"{user}:{item.get('id', item.get('idx'))}",
                    "question": item["question"],
                    "gold": _gold_groups(item),
                    "measurable": dataset not in _UNMEASURABLE,
                }
                for item in sorted(kept, key=lambda i: str(i.get("id", i.get("idx"))))
            )
        spec.append(
            {
                "dataset": dataset,
                "user_id": chosen[0],
                "users": chosen,
                "pairs": spent,
                "questions": questions,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    total_pairs = sum(s["pairs"] for s in spec)
    total_q = sum(len(s["questions"]) for s in spec)
    print(f"sample: {len(spec)} datasets, {total_pairs} pairs, {total_q} questions")
    for s in spec:
        print(
            f"   {s['dataset']:18s} users={len(s['users'])} pairs={s['pairs']:4d} "
            f"q={len(s['questions'])}"
        )
    return 0


# ── index ───────────────────────────────────────────────────────────────


def _copy_schema(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    con = sqlite3.connect(target)
    con.executescript((ROOT / "src/tianximem/store/schema.sql").read_text(encoding="utf-8"))
    con.commit()
    con.close()
    return None


def _insert_pairs(target: Path, source: Path, users: list[str]) -> None:
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    cols = [d[1] for d in src.execute("pragma table_info(qa_pairs)")]
    placeholders = ",".join("?" for _ in cols)
    for user in users:
        rows = src.execute(
            f"select {','.join(cols)} from qa_pairs where user_id=?", (user,)
        ).fetchall()
        dst.executemany(
            f"insert into qa_pairs ({','.join(cols)}) values ({placeholders})", rows
        )
    dst.commit()
    dst.close()
    src.close()
    return None


def stage_index(args: argparse.Namespace) -> int:
    spec = json.loads(args.sample.read_text(encoding="utf-8"))
    users = [u for s in spec for u in s.get("users", [s["user_id"]])]
    rule_db, llm_db = args.dir / "rule.sqlite", args.dir / "llm.sqlite"
    args.dir.mkdir(parents=True, exist_ok=True)
    # ⚠ **`--rule-only` 不许碰 `llm.sqlite`**：它曾把另一个臂整库重建成规则层，
    # 于是两臂"逐题逐字相同"——那不是结论，是这条覆盖（2026-10-18 实测踩到）。
    targets = (rule_db,) if args.rule_only else (rule_db, llm_db)
    for target in targets:
        _copy_schema(target, args.truth)
        _insert_pairs(target, args.truth, users)

    values_probe = _env()
    probe_endpoints(values_probe)
    store = SqliteStore.open(rule_db)
    with store.transaction() as conn:
        for pair in store.iter_pairs(conn):
            index_pair_facts(store, conn, pair)
    print(f"rule facts: {_fact_count(rule_db)}")
    VOCABULARY.update(_gold_relation_labels())
    with sqlite3.connect(rule_db) as con:
        VOCABULARY.update(
            row[0]
            for row in con.execute(
                "select distinct relation_key from memory_facts where version=?", (EVIDENCE_VERSION,)
            )
            if row[0]
        )
    print(f"vocabulary: {len(VOCABULARY)} relations")
    if args.rule_only:
        return 0
    values = _env()
    cache_path = args.dir / "llm-facts.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                # 带 error 的行不进缓存：下次调用会重试它（网关会整段时间 5xx）。
                if not row.get("error"):
                    cache[row["id"]] = row
    llm_store = SqliteStore.open(llm_db)
    with llm_store.read() as conn:
        pairs = llm_store.iter_pairs(conn)
    wanted = tuple(x for x in (args.users or "").split(",") if x)
    pending = (
        []
        if args.write_only
        else [
            p
            for p in pairs
            if p.id not in cache and (not wanted or p.user_id.startswith(wanted))
        ]
    )
    print(f"llm extraction: cached={len(pairs) - len(pending)} pending={len(pending)}")
    if pending:
        vocabulary = tuple(sorted(VOCABULARY))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_extract_one, pair, i, vocabulary, values, args.max_tokens): pair
                for i, pair in enumerate(pending)
            }
            done = 0
            with cache_path.open("a", encoding="utf-8") as handle:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    cache[row["id"]] = row
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/{len(pending)}", flush=True)
    _write_llm_facts(llm_db, pairs, cache)
    print(f"llm facts: {_fact_count(llm_db)}")
    return 0


def _gold_relation_labels() -> set[str]:
    """`eval/facts/` 的人工金标里出现过的关系标签——那是这套管线验证过的说法。"""
    labels: set[str] = set()
    for path in sorted((ROOT / "eval/facts").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except ValueError:
                continue
            for item in (case.get("relations") or ()):
                if isinstance(item, dict) and str(item.get("relation", "")).strip():
                    candidate = relation_key(str(item["relation"]))
                    if candidate:
                        labels.add(candidate)
    return labels


def _fact_count(path: Path) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "select count(*) from memory_facts where version=?", (EVIDENCE_VERSION,)
        ).fetchone()[0]
    finally:
        con.close()


def _extract_one(
    pair: QaPair,
    index: int,
    vocabulary: tuple[str, ...],
    values: dict[str, str],
    max_tokens: int,
) -> dict[str, Any]:
    record = (pair.question or "") + ("\n" + pair.answer if pair.answer else "")
    from tools.fact_llm_gate import _scopes

    last = ""
    facts: list[dict[str, str]] = []
    chunks = [record] if len(record) <= _CHUNK_LIMIT else list(_scopes(record))[:_MAX_CHUNKS]
    for chunk in chunks:
        for attempt in range(3):
            chat = _chat(values, index + attempt)
            try:
                raw = chat.complete(
                    EXTRACT_PROMPT.replace("{record}", chunk),
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                facts.extend(_parse(raw))
                break
            except Exception as exc:  # noqa: BLE001 - cache the failure and retry
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0 * (attempt + 1))
            finally:
                chat.close()
    return {"id": pair.id, "facts": facts, "error": last or None}


def _parse(raw: str) -> list[dict[str, str]]:
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fence:
        text = fence[1]
    payload: Any = None
    try:
        payload = json.loads(text)
    except ValueError:
        # 前后有解释文字时取第一个 JSON 数组／对象（⛔ 数组要先试：`\{.*\}` 会命中
        # 数组里那个对象，把它当 dict 读下来就一条都不剩）。
        match = re.search(r"\[.*\]|\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except ValueError:
                return []
    # 三种形状都收：裸数组（当前提示词）、{"relations": [...]}、以及单条关系对象。
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("relations"), list):
        items = payload["relations"]
    elif isinstance(payload, dict) and {"subject", "relation", "object"} <= set(payload):
        items = [payload]
    else:
        items = None
    out: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        values = {k: str(item.get(k, "")).strip() for k in ("subject", "relation", "object")}
        if all(values.values()):
            out.append(values)
    return out


def _write_llm_facts(db: Path, pairs: list[QaPair], cache: dict[str, dict[str, Any]]) -> None:
    store = SqliteStore.open(db)
    with store.transaction() as conn:
        conn.execute("DELETE FROM memory_facts WHERE version=?", (EVIDENCE_VERSION,))
        conn.execute("DELETE FROM evidence_coverage")
        for pair in pairs:
            row = cache.get(pair.id)
            if row is None:
                continue
            kept = 0
            seen: set[tuple[str, str, str]] = set()
            model_facts = list(row.get("facts", []))
            # 更新句上模型空返回 / 把整个名词短语当主语时，用本地句法兜底。
            # 兜底**只替换掉同宾语的那条坏主语**（"The capital of Malawi" is Oradea），
            # 不删模型给出的其他事实——真实体也可能以 "The" 开头（The Beatles）。
            if not model_facts or any(
                fact["subject"].casefold().startswith(("the ", "a ", "an "))
                for fact in model_facts
            ):
                known = {(key(f["subject"]), key(f["object"])) for f in model_facts}
                for extra in _local_update_facts(pair.question):
                    if (key(extra["subject"]), key(extra["object"])) in known:
                        continue
                    model_facts = [
                        fact
                        for fact in model_facts
                        if not (
                            key(fact["object"]) == key(extra["object"])
                            and fact["subject"].casefold().startswith(("the ", "a ", "an "))
                        )
                    ]
                    model_facts.append(extra)
            for fact in model_facts:
                subject = fact["subject"]
                relation = relation_key(fact["relation"])
                object_ = fact["object"]
                # 引句必须**逐字落在 question 或 answer 里**（store 会验）——两侧分别找范围。
                quote, side = "", "question"
                for candidate_side, text in (("question", pair.question or ""), ("answer", pair.answer or "")):
                    if not text:
                        continue
                    found = local_scope(text, subject, object_)
                    if found:
                        quote, side = found, candidate_side
                        break
                if not quote:
                    continue
                full = (pair.question or "") + ("\n" + pair.answer if pair.answer else "")
                # ⚠ **先归一化**：模型标签若恰好撞进词表（"located"、"participated"），
                # 先过闸门就会被原样保留，同义表永远没机会把它对齐到计划问的关系名。
                # 同义表本身**就是许可**（它只收词表里已有的关系），不再要求触发词；
                # 没命中同义表时才走"闸门 → 触发词映射"。
                mapped = _synonym_relation(fact["relation"])
                if mapped is not None:
                    relation = mapped
                else:
                    ok, _reason = llm_gate(full, subject, relation, object_)
                    if not ok:
                        mapped = canonical_relation(quote, subject, object_)
                        if mapped is None:
                            continue
                        relation = mapped
                signature = (key(subject), relation, key(object_))
                if signature in seen:
                    continue
                seen.add(signature)
                # 显式月份区间与规则层同一套解析：interval 问法靠 `start_month/end_month` 定位。
                qualifiers: dict[str, Any] = {"projection": "fact"}
                span = _INTERVAL.search(quote)
                if span:
                    start, end = month_index(span[2]), month_index(span[3])
                    if start is not None and end is not None and start <= end:
                        qualifiers.update(start_month=start, end_month=end)
                # 显式替换的标记与规则层同一口径（`grammar._REPLACEMENT`）：没有它，
                # 执行器的 `_current` 认不出这条是更新，新旧两条会被一起当成普通陈述。
                if _REPLACEMENT.search(quote):
                    qualifiers.update(replaces_earlier=True, observation=True)
                made = make_fact(
                    parent_memory_id=pair.id,
                    user_id=pair.user_id,
                    subject=subject,
                    relation=relation,
                    object=object_,
                    source_quote=quote,
                    source_side=side,
                    statement=f"{subject} {relation} {object_}",
                    event_time=pair.event_time,
                    position=f"llm:{kept}:{relation}",
                    qualifiers=qualifiers,
                )
                store.insert_evidence(conn, (made,))
                kept += 1
            store.mark_fact_index_coverage(conn, pair, version=EVIDENCE_VERSION)
    return None


# ── run ─────────────────────────────────────────────────────────────────


def stage_run(args: argparse.Namespace) -> int:
    spec = json.loads(args.sample.read_text(encoding="utf-8"))
    values = _env()
    planner = GroundedPlanner(
        _chat(values, _live_endpoint(values), timeout=20.0),
        max_output_tokens=256,
        temperature=0.0,
    )
    plan_cache: dict[str, Any] = {}
    if args.plans.exists():
        plan_cache = json.loads(args.plans.read_text(encoding="utf-8"))
    layers = {"rule": SqliteStore.open(args.dir / "rule.sqlite"),
              "llm": SqliteStore.open(args.dir / "llm.sqlite")}
    stages = {
        name: GroundedEvidenceStage(
            store=store, evidence_limit=12, evidence_hop_limit=4, fact_backfill_limit=1024
        )
        for name, store in layers.items()
    }
    rows: list[dict[str, Any]] = []
    for group in spec:
        for item in group["questions"]:
            user = item["qid"].split(":", 1)[0]
            question = item["question"]
            if question not in plan_cache:
                plan = compile_query(question)
                source = "rule"
                status = "compiled"
                if plan is None:
                    decision = planner.decide(question)
                    plan, source, status = decision.plan, "llm", decision.status
                plan_cache[question] = {
                    "source": source if plan is not None else None,
                    "status": status if plan is not None else (
                        "no_plan" if source == "rule" else status
                    ),
                }
                if plan is not None:
                    plan_cache[question]["plan"] = _plan_json(plan)
                    plan_cache[question]["_plan"] = plan
                    plan = plan
            cached = plan_cache[question]
            plan = cached.get("_plan") or _plan_from_json(cached.get("plan"))
            if plan is None:
                rows.extend(
                    {
                        "dataset": group["dataset"],
                        "user_id": user,
                        "qid": item["qid"],
                        "layer": layer,
                        "status": cached["status"],
                        "plan_source": None,
                        "gold": item["gold"],
                        "measurable": item.get("measurable", True),
                        "hit": False,
                        "evidence": "",
                    }
                    for layer in layers
                )
                continue
            for layer, stage in stages.items():
                decision = stage.decide(user, plan)
                text = ""
                if decision.selection is not None:
                    text = _evidence_text(layers[layer], user, decision.selection)
                rows.append(
                    {
                        "dataset": group["dataset"],
                        "user_id": user,
                        "qid": item["qid"],
                        "layer": layer,
                        "status": decision.reason,
                        "plan_source": cached["source"],
                        "gold": item["gold"],
                        "measurable": item.get("measurable", True),
                        "hit": decision.selection is not None,
                        "evidence": text[:20000],
                    }
                )
            print(f"  {group['dataset']:16s} {item['qid']:12s} done", flush=True)
    clean = {k: {n: v for n, v in cached.items() if n != "_plan"} for k, cached in plan_cache.items()}
    try:
        json.dumps(clean, ensure_ascii=False)
    except TypeError:
        for key, cached in clean.items():
            for name, value in cached.items():
                try:
                    json.dumps(value)
                except TypeError as exc:
                    print(f"DEBUG unserializable {key!r}.{name}: {type(value)} {str(value)[:120]} {exc}")
        raise
    args.plans.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            try:
                encoded = json.dumps(row, ensure_ascii=False)
            except TypeError:
                for name, value in row.items():
                    try:
                        json.dumps(value)
                    except TypeError as exc:
                        print(f"DEBUG row {row.get('qid')!r}.{name}: {type(value)} {value!r:.120} {exc}")
                raise
            handle.write(encoded + "\n")
    print(f"rows: {len(rows)} -> {args.out}")
    return 0


def _plan_json(plan: Any) -> dict[str, Any]:
    return {
        "operator": plan.operator,
        "patterns": [
            {
                "relations": list(p.relations),
                "subject": p.subject,
                "object": p.object,
                "object_terms": p.object_terms,
                "container_terms": p.container_terms,
                "source_ids": list(p.source_ids),
            }
            for p in plan.patterns
        ],
        "projection": plan.projection,
        "date_bounds": list(plan.date_bounds) if plan.date_bounds else None,
        "per_occurrence": plan.per_occurrence,
        "guard_literal": plan.guard_literal,
        "require_complete_walk": plan.require_complete_walk,
        "walk_root": plan.walk_root,
        "walk_relations": list(plan.walk_relations),
        "resolve_references": plan.resolve_references,
        "month": plan.month,
        "direction": plan.direction,
        "reference": plan.reference,
        "bind_side": plan.bind_side,
        "query": plan.query,
    }


def _plan_from_json(data: dict[str, Any] | None) -> Any:
    if not data:
        return None
    from tianximem.facts.query import FactPattern, QueryPlan

    return QueryPlan(
        data["operator"],
        tuple(
            FactPattern(
                tuple(p["relations"]),
                subject=p["subject"],
                object=p["object"],
                object_terms=p["object_terms"],
                container_terms=p["container_terms"],
                source_ids=tuple(p["source_ids"]),
            )
            for p in data["patterns"]
        ),
        projection=data["projection"],
        resolve_references=data["resolve_references"],
        query=data["query"],
        guard_literal=data["guard_literal"],
        date_bounds=tuple(data["date_bounds"]) if data["date_bounds"] else None,
        month=data["month"],
        direction=data["direction"],
        reference=data["reference"],
        bind_side=data["bind_side"],
        require_complete_walk=data["require_complete_walk"],
        per_occurrence=data["per_occurrence"],
        walk_root=data["walk_root"],
        walk_relations=tuple(data["walk_relations"]),
    )


def _evidence_text(store: SqliteStore, user: str, selection: Any) -> str:
    facts = list(selection.facts)
    parts = [f.content for f in facts]
    if selection.projection == "source":
        parents = sorted({f.parent_memory_id for f in facts})
        with store.read() as conn:
            for pair in store.fetch_pairs_by_ids(conn, parents):
                parts.append((pair.question or "") + "\n" + (pair.answer or ""))
    return "\n\n".join(parts)


# ── report ──────────────────────────────────────────────────────────────


_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_PROSE_DATE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})[,.]?\s+(\d{4})\b|"
    r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?[,.]?\s+(\d{4})\b",
    re.I,
)
_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[,.]?\s+(\d{4})\b",
    re.I,
)
_MONTHS = {name.lower(): index for index, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1
)}
_MONTHS.update({name[:3].lower(): value for name, value in _MONTHS.items()})


def _dates(text: str) -> set[tuple[int, int, int]]:
    """证据里出现过的日期（ISO 与 "16 March, 2023" 两种写法都认）。"""
    found: set[tuple[int, int, int]] = set()
    for year, month, day in _ISO_DATE.findall(text):
        found.add((int(year), int(month), int(day)))
    for day, month_name, year, month_name2, day2, year2 in _PROSE_DATE.findall(text):
        if day:
            month = _MONTHS.get(month_name.lower())
            target = (int(year), month, int(day)) if month else None
        else:
            month = _MONTHS.get(month_name2.lower())
            target = (int(year2), month, int(day2)) if month else None
        if target:
            found.add(target)
    return found


def _month_hit(text: str, gold: str) -> bool:
    """月粒度金标（"January, 2023"）：证据里有**落在那个月**的日期就算命中。"""
    match = _MONTH_YEAR.search(gold)
    if not match:
        return False
    month = _MONTHS.get(match[1].lower())
    if not month:
        return False
    year = int(match[2])
    if f"{year:04d}-{month:02d}" in text:
        return True
    return any((y, m) == (year, month) for y, m, _ in _dates(text))


def _gold_hit(groups: list[list[str]] | None, text: str) -> bool | None:
    """金标是否出现在所返回的证据里；`None` = 该题没给可比对的金标。

    日期按**日历**比（证据写 `2023-03-16`、金标写 `16 March, 2023` 算命中），
    其余按归一化子串。
    """
    if not groups:
        return None
    haystack = key(text)
    evidence_dates = _dates(text)
    for group in groups:
        ok = True
        for item in group:
            needle = key(str(item))
            if not needle:
                continue
            if needle in haystack:
                continue
            target = next(iter(_dates(str(item))), None)
            if target and target in evidence_dates:
                continue
            if _month_hit(text, str(item)):
                continue
            # LoCoMo 的金标把多个答案塞进一个字符串（"fair, networking events"）——
            # 整体不在就拆开，要求每一段都在（**只在这一项上**加这条宽容）。
            parts = [part.strip() for part in str(item).split(",") if part.strip()]
            if len(parts) > 1 and all(key(part) in haystack for part in parts):
                continue
            ok = False
            break
        if ok:
            return True
    return False


def _covered_users(db: Path, cache_path: Path) -> set[str]:
    """LLM 臂**已经抽完**的用户：该用户的每一篇 qa_pair 都有成功的缓存行。"""
    if not cache_path.exists():
        return set()
    done = {
        row["id"]
        for row in (
            json.loads(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not row.get("error")
    }
    by_user: dict[str, list[str]] = {}
    with sqlite3.connect(db) as con:
        for user_id, pair_id in con.execute("select user_id, id from qa_pairs"):
            by_user.setdefault(user_id, []).append(pair_id)
    return {user for user, ids in by_user.items() if all(pid in done for pid in ids)}


def stage_report(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[str] = [
        "| 数据集 | 层 | 题数 | 命中 | 命中率 | 可比命中 | 证据含金标 | 精确率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total = {"rule": [0, 0, 0, 0], "llm": [0, 0, 0, 0]}
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        for layer in ("rule", "llm"):
            picked = [r for r in rows if r["dataset"] == dataset and r["layer"] == layer]
            hits = [r for r in picked if r["hit"]]
            judged = [
                (r, _gold_hit(r["gold"], r["evidence"]))
                for r in hits
                if r.get("measurable", True)
            ]
            good = [r for r, ok in judged if ok]
            total[layer][0] += len(picked)
            total[layer][1] += len(hits)
            total[layer][2] += len(judged)
            total[layer][3] += len(good)
            rate = f"{len(hits) / len(picked):.2f}" if picked else "-"
            prec = f"{len(good) / len(judged):.2f}" if judged else "-"
            out.append(
                f"| {dataset} | {layer} | {len(picked)} | {len(hits)} | {rate} | "
                f"{len(judged)} | {len(good)} | {prec} |"
            )
    for layer in ("rule", "llm"):
        n, hits, judged, good = total[layer]
        out.append(
            f"| **合计** | {layer} | {n} | {hits} | {hits / n:.2f} | {judged} | {good} | "
            f"{good / judged if judged else 0:.2f} |"
        )
    status: dict[str, dict[str, int]] = {}
    for r in rows:
        status.setdefault(r["layer"], {}).setdefault(r["status"], 0)
        status[r["layer"]][r["status"]] += 1
    out.append("")
    for layer, counts in status.items():
        out.append(f"* `{layer}` 分支结局：" + "、".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    covered = _covered_users(args.dir / "llm.sqlite", args.coverage) if args.coverage else set()
    if covered:
        scoped = [
            r for r in rows if r["qid"].split(":", 1)[0] in covered
        ]
        out += ["", "## 只看 LLM 臂已抽完的用户（同一批题、两臂同分母）", "",
                "| 数据集 | 层 | 题数 | 命中 | 命中率 | 可比命中 | 证据含金标 | 精确率 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        totals = {"rule": [0, 0, 0, 0], "llm": [0, 0, 0, 0]}
        for dataset in dict.fromkeys(r["dataset"] for r in scoped):
            for layer in ("rule", "llm"):
                picked = [r for r in scoped if r["dataset"] == dataset and r["layer"] == layer]
                hits = [r for r in picked if r["hit"]]
                judged = [(r, _gold_hit(r["gold"], r["evidence"])) for r in hits if r.get("measurable", True)]
                good = [r for r, ok in judged if ok]
                totals[layer][0] += len(picked)
                totals[layer][1] += len(hits)
                totals[layer][2] += len(judged)
                totals[layer][3] += len(good)
                rate = f"{len(hits) / len(picked):.2f}" if picked else "-"
                prec = f"{len(good) / len(judged):.2f}" if judged else "-"
                out.append(
                    f"| {dataset} | {layer} | {len(picked)} | {len(hits)} | {rate} | "
                    f"{len(judged)} | {len(good)} | {prec} |"
                )
        for layer in ("rule", "llm"):
            n, hits, judged, good = totals[layer]
            out.append(
                f"| **合计** | {layer} | {n} | {hits} | {hits / n:.2f} | {judged} | {good} | "
                f"{good / judged if judged else 0:.2f} |"
            )
    # 逐题差异：只看"一臂命中、另一臂没命中"的题（用户要找的就是这些）。
    questions = {q["qid"]: q for group in json.loads(args.sample.read_text(encoding="utf-8")) for q in group["questions"]}
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_qid.setdefault(row["qid"], {})[row["layer"]] = row
    diffs = [
        (qid, layers)
        for qid, layers in by_qid.items()
        if ("llm" in layers) != ("rule" in layers)
        or layers.get("rule", {}).get("hit") != layers.get("llm", {}).get("hit")
    ]
    out += ["", "## 两臂逐题差异（只有一臂命中）", ""]
    for qid, layers in sorted(diffs):
        item = questions.get(qid, {})
        cells = []
        for layer in ("rule", "llm"):
            row = layers.get(layer)
            if row is None:
                cells.append(f"{layer}: -")
                continue
            verdict = _gold_hit(row["gold"], row["evidence"]) if row["hit"] else None
            mark = "命中" if row["hit"] else row["status"]
            cells.append(f"{layer}: {mark}" + ("（证据含金标）" if verdict else "（证据不含金标）" if verdict is False else ""))
        out.append(f"* `{qid}` {str(item.get('question',''))[:70]} | " + " · ".join(cells))
    if not diffs:
        out.append("* （无：两臂在每一题上的命中与否都相同）")
    print("\n".join(out))
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("sample")
    p.add_argument("--truth", type=Path, default=ROOT / "var/facts-v11-native-20261013/truth.db")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--questions", type=int, default=25)
    p.add_argument("--min-pairs", type=int, default=120)
    p.add_argument("--max-pairs", type=int, default=320)
    p.add_argument("--seed", type=int, default=20261016)
    p.set_defaults(func=stage_sample)

    p = sub.add_parser("index")
    p.add_argument("--truth", type=Path, default=ROOT / "var/facts-v11-native-20261013/truth.db")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--rule-only", action="store_true")
    p.add_argument("--write-only", action="store_true", help="只把现有缓存写进 llm.sqlite")
    p.add_argument("--users", default="", help="只抽这些用户前缀（逗号分隔）")
    p.set_defaults(func=stage_index)

    p = sub.add_parser("run")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True)
    p.set_defaults(func=stage_run)

    p = sub.add_parser("report")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dir", type=Path, help="含 llm.sqlite 的目录（配合 --coverage 出对照表）")
    p.add_argument("--coverage", type=Path, help="LLM 抽取缓存；只统计已抽完的用户")
    p.set_defaults(func=stage_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
