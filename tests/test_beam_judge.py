import json as jsonlib

import pytest
from eval.harness.beam_judge import parse_score


def test_recovers_quotes_in_reason_without_changing_score():
    assert parse_score('{"score": 0.5, "reason": "Mentions "book club" but omits the date."}') == {
        "score": 0.5,
        "reason": 'Mentions "book club" but omits the date.',
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"reason":"missing score"}',
        '{"score": 1}',
        '{"score": true, "reason":"x"}',
        '{"score": 0.25, "reason":"x"}',
        '{"score": 1, "reason":""}',
        '{"score": 1, "reason":"truncated',
        '{"scores":[{"index":0,"score":1,"reason":"x"}]}',
    ],
)
def test_rejects_incomplete_or_invalid_single_rubric(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_score(raw)


def test_all_rubrics_required_and_saved_zero_is_not_rejudged(tmp_path, monkeypatch):
    import httpx
    from eval.harness import beam_judge

    pipeline = tmp_path / "upstream.py"
    pipeline.write_text('UNIFIED_LLM_JUDGE_BASE_PROMPT = "<question> <rubric_item> <llm_response>"')
    item = {
        "id": "q",
        "question": "question",
        "question_type": "temporal_reasoning",
        "rubric": ["first", "second"],
    }
    calls = []
    fail_second = True

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, headers, json):
            prompt = json["messages"][0]["content"]
            calls.append(prompt)
            body = {"score": 0, "reason": "wrong"}
            if "second" in prompt:
                body = (
                    {"reason": "missing score"} if fail_second else {"score": 1, "reason": "right"}
                )
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": jsonlib.dumps(body)}}]},
            )

    monkeypatch.setattr(beam_judge.httpx, "Client", Client)
    monkeypatch.setattr(beam_judge.api_config, "ENDPOINTS", [("https://stub", "key")])
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        beam_judge.judge_independently(pipeline, item, "unchanged answer", out, 0)
    assert (out / "0-completed.json").exists()
    calls.clear()
    fail_second = False
    result = beam_judge.judge_independently(pipeline, item, "unchanged answer", out, 0)
    assert result["llm_judge_score"] == 0.5
    assert len(calls) == 1 and "second" in calls[0]
