"""按公开 case 测 JSON 事实的更新前/后状态；小范围语料是显式近似。"""

from __future__ import annotations

import json
from pathlib import Path

from ..mquake import case_chunks
from ..preprocess import Question, Sample
from .plan import AddEvent, InputPlan, SearchEvent

UPDATE = "UPDATE: replace the prior value of this subject and relation with the following fact.\n"


def _fact(labels, ids) -> dict:
    if len(labels) != 3 or len(ids) != 3:
        raise ValueError("AML MQuAKE: labeled/ID triples must each have three entries")
    return dict(
        zip(
            ("subject", "relation", "object", "subject_id", "relation_id", "object_id"),
            [*labels, *ids],
            strict=True,
        )
    )


def load(root: Path, *, limit: int | None, spread: bool) -> list[InputPlan]:
    plans = []
    for stem, chunk in case_chunks(root, limit=limit, spread=spread, cases_per_user=1):
        row = chunk[0]
        if not row.get("questions"):
            continue
        uid = f"mqk-{stem}-{row['case_id']}"
        originals = [
            _fact(labels, ids)
            for labels, ids in zip(row["orig_triples_labeled"], row["orig_triples"], strict=True)
        ]
        new_facts = [
            _fact(labels, ids)
            for labels, ids in zip(row["new_triples_labeled"], row["new_triples"], strict=True)
        ]
        edited = {tuple(triple[:2]) for triple in row["edit_triples"]}
        # 新答案链会走到原答案链之外。非编辑的链边是源数据中的普通事实，
        # 编辑边先放 target_true，后通过 UPDATE 放 target_new；答案字段不选事实。
        originals.extend(
            fact for fact in new_facts if (fact["subject_id"], fact["relation_id"]) not in edited
        )
        updates = []
        for triple, rewrite in zip(row["edit_triples"], row["requested_rewrite"], strict=True):
            candidates = [
                fact
                for fact in new_facts + originals
                if (fact["subject_id"], fact["relation_id"]) == tuple(triple[:2])
            ]
            if not candidates or not rewrite.get("target_new_str"):
                raise ValueError(f"AML MQuAKE {uid}: cannot identify update subject/relation")
            fact = candidates[0]
            if not rewrite.get("target_true_str") or not rewrite.get("target_true_id"):
                raise ValueError(f"AML MQuAKE {uid}: missing original update value")
            originals.append(
                _fact(
                    [fact["subject"], fact["relation"], rewrite["target_true_str"]],
                    [triple[0], triple[1], rewrite["target_true_id"]],
                )
            )
            updates.append(
                _fact(
                    [fact["subject"], fact["relation"], rewrite["target_new_str"]],
                    triple,
                )
            )
        originals = list({json.dumps(fact, sort_keys=True): fact for fact in originals}.values())
        events = [
            AddEvent(
                "original",
                tuple(
                    {"role": "user", "content": "Corpus: " + json.dumps(fact, ensure_ascii=False)}
                    for fact in originals
                ),
            )
        ]
        questions = []
        for phase, answer_key, alias_key in (
            ("original", "answer", "answer_alias"),
            ("updated", "new_answer", "new_answer_alias"),
        ):
            if phase == "updated":
                if not updates:
                    raise ValueError(f"AML MQuAKE {uid}: missing updates")
                events.append(
                    AddEvent(
                        "updated",
                        tuple(
                            {
                                "role": "user",
                                "content": "Corpus: "
                                + UPDATE
                                + json.dumps(fact, ensure_ascii=False),
                            }
                            for fact in updates
                        ),
                    )
                )
            if not row.get(answer_key):
                raise ValueError(f"AML MQuAKE {uid}: missing {phase} evaluation answer")
            gold = sorted({str(row[answer_key]), *map(str, row.get(alias_key) or [])})
            for index, question in enumerate(row["questions"]):
                qid = f"{uid}-{phase}-{index}"
                questions.append(Question(qid, str(question), gold, stem))
                events.append(SearchEvent(qid, str(question)))
        plans.append(
            InputPlan(
                Sample(uid, "mquake-remastered", (), tuple(questions)),
                tuple(events),
                {
                    "scope": "one public case per user; original then updated QA",
                    "time_style": "none",
                },
            )
        )
    return plans
