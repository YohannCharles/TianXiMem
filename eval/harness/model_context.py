"""Fit BEAM's answer context prefix to the local model's token envelope.

This is an explicit local input adaptation, separate from Search and from the
platform's o200k prefix. The original item is never modified.
"""

from __future__ import annotations

import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

_TOKENIZER_LOCK = Lock()


@lru_cache(maxsize=2)
def _load_tokenizer(model: str, revision: str, cache_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model, revision=revision, cache_dir=cache_dir, use_fast=True, local_files_only=True
    )


def fit_beam_context(
    pipeline: Path, item: dict, config: dict, *, tokenizer=None
) -> tuple[dict, dict]:
    template = None
    for node in ast.parse(pipeline.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "ANSWER_GENERATION_FOR_RAG" for target in node.targets
        ):
            template = ast.literal_eval(node.value)
            break
    if template is None:
        raise ValueError("Missing pinned BEAM answer template")
    if tokenizer is None:
        with _TOKENIZER_LOCK:
            tokenizer = _load_tokenizer(
                config["model"], config["tokenizer_revision"], config["cache_dir"]
            )
    context = item["context"]
    if not isinstance(context, str):
        raise ValueError("BEAM prefix adaptation requires text context")
    limit = (
        config["max_context_tokens"]
        - config["answer_max_tokens"]
        - config["overhead_margin_tokens"]
    )

    def count(value):
        values = {"context": value, "question": item["question"]}
        prompt = re.sub(r"<(context|question)>", lambda match: values[match[1]], template)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    original_tokens = count(context)
    adjusted = dict(item)
    if original_tokens > limit:
        if count("") > limit:
            raise ValueError("Question and prompt exceed the model budget even with no context")
        low, high = 0, len(context)
        while low < high:
            middle = (low + high + 1) // 2
            if count(context[:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        adjusted["context"] = context[:low]
    kept_tokens = count(adjusted["context"])
    if kept_tokens > limit or not context.startswith(adjusted["context"]):
        raise ValueError("Model prefix adaptation failed its budget/prefix invariant")
    audit = {
        "contract": "beam-local-qwen-context-prefix-v1",
        "id": item["id"],
        "model": config["model"],
        "tokenizer_revision": config["tokenizer_revision"],
        "max_context_tokens": config["max_context_tokens"],
        "answer_max_tokens": config["answer_max_tokens"],
        "overhead_margin_tokens": config["overhead_margin_tokens"],
        "prompt_budget_tokens": limit,
        "original_prompt_tokens": original_tokens,
        "actual_prompt_tokens": kept_tokens,
        "removed_prompt_tokens": original_tokens - kept_tokens,
        "original_context_characters": len(context),
        "actual_context_characters": len(adjusted["context"]),
        "removed_context_characters": len(context) - len(adjusted["context"]),
        "truncation_character_offset": len(adjusted["context"]),
        "original_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "actual_context_sha256": hashlib.sha256(adjusted["context"].encode()).hexdigest(),
        "truncated": adjusted["context"] != context,
    }
    return adjusted, audit
