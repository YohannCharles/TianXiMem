"""Recover BEAM rubric failures using the upstream single-criterion protocol.

All criteria must receive explicit valid model scores. No missing index is
filled or inferred. Saved answers and successful rubric judgments are reused.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import httpx

from eval.harness import api_config
from eval.harness.extra_pipeline import _repair_memtrap_string_escapes


def parse_score(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = json.loads(_repair_memtrap_string_escapes(raw))
    if not isinstance(value, dict):
        raise ValueError("BEAM judge must return an object")
    score, reason = value.get("score"), value.get("reason")
    if isinstance(score, bool) or score not in (0, 0.5, 1):
        raise ValueError("BEAM score must be 0, 0.5, or 1")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("BEAM judge must provide a nonempty reason")
    return {"score": float(score), "reason": reason}


def single_rubric_template(pipeline: Path) -> str:
    for node in ast.parse(pipeline.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "UNIFIED_LLM_JUDGE_BASE_PROMPT" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing pinned upstream rubric prompt")


def judge_independently(
    pipeline: Path, item: dict, answer: str, out_dir: Path, endpoint_index: int
) -> dict:
    template = single_rubric_template(pipeline)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for index, rubric in enumerate(item["rubric"]):
        prompt = (
            template.replace("<question>", item["question"])
            .replace("<rubric_item>", rubric)
            .replace("<llm_response>", answer)
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"prompt": prompt, "model": api_config.JUDGE_MODEL, "max_tokens": 1024},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        completed = out_dir / f"{index}-completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text(encoding="utf-8"))
            if saved["input_sha256"] != fingerprint:
                raise ValueError("Saved rubric judgment has different input")
            scores.append(parse_score(saved["response"]))
            continue
        errors = []
        for attempt in range(2):
            selected = (endpoint_index + attempt) % len(api_config.ENDPOINTS)
            base, key = api_config.ENDPOINTS[selected]
            payload = {
                "model": api_config.JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            }
            try:
                with httpx.Client(timeout=120, trust_env=False) as client:
                    response = client.post(
                        base.rstrip("/") + "/chat/completions",
                        headers={"Authorization": "Bearer " + key},
                        json=payload,
                    )
                data = response.json()
                (out_dir / f"{index}-attempt-{attempt}.json").write_text(
                    json.dumps(
                        {
                            "input_sha256": fingerprint,
                            "endpoint_index": selected,
                            "status": response.status_code,
                            "response": data,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                response.raise_for_status()
                raw = data["choices"][0]["message"]["content"]
                score = parse_score(raw)
                temp = completed.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(
                        {"input_sha256": fingerprint, "response": raw, "endpoint_index": selected},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                temp.replace(completed)
                scores.append(score)
                break
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError(f"Rubric {index} still failed: {errors[-1]}")
    return {
        "id": item["id"],
        "question_type": item["question_type"],
        "judge_model": api_config.JUDGE_MODEL,
        "rubric_scores": [
            {"rubric": criterion, **score}
            for criterion, score in zip(item["rubric"], scores, strict=True)
        ],
        "llm_judge_score": sum(score["score"] for score in scores) / len(scores),
        "judge_response": json.dumps(
            {"scores": [{"index": index, **score} for index, score in enumerate(scores)]},
            ensure_ascii=False,
        ),
        "recovery_protocol": (
            "upstream single-criterion prompt; all explicit scores required; rubric metric only"
        ),
    }
