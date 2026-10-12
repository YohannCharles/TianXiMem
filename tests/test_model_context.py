from eval.harness.model_context import fit_beam_context


class ByteTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


def test_prefix_keeps_unicode_question_and_original_item(tmp_path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text('ANSWER_GENERATION_FOR_RAG = "<context>|<question>"')
    original = {"id": "q", "context": "甲乙丙丁", "question": "Q", "rubric": ["R"]}
    config = {
        "model": "stub",
        "tokenizer_revision": "fixed",
        "max_context_tokens": 12,
        "answer_max_tokens": 3,
        "overhead_margin_tokens": 1,
    }
    adjusted, audit = fit_beam_context(pipeline, original, config, tokenizer=ByteTokenizer())
    assert original["context"] == "甲乙丙丁"
    assert (
        adjusted["context"] == "甲乙"
        and adjusted["question"] == "Q"
        and adjusted["rubric"] == ["R"]
    )
    assert audit["actual_prompt_tokens"] + 3 + 1 <= 12
    assert audit["removed_context_characters"] == 2 and audit["removed_prompt_tokens"] == 6


def test_context_inside_model_limit_is_not_changed(tmp_path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text('ANSWER_GENERATION_FOR_RAG = "<context>|<question>"')
    original = {"id": "q", "context": "short", "question": "Q"}
    config = {
        "model": "stub",
        "tokenizer_revision": "fixed",
        "max_context_tokens": 100,
        "answer_max_tokens": 3,
        "overhead_margin_tokens": 1,
    }
    adjusted, audit = fit_beam_context(pipeline, original, config, tokenizer=ByteTokenizer())
    assert adjusted == original and audit["truncated"] is False
