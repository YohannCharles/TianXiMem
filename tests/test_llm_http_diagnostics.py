import httpx
import pytest
from eval.harness import extra_base


@pytest.mark.parametrize("status", [400, 502])
def test_chat_error_keeps_endpoint_status_and_response(monkeypatch, status):
    attempts = []

    def post(url, **kwargs):
        attempts.append(url)
        return httpx.Response(
            status,
            json={"error": "input exceeds context" if status == 400 else "bad gateway"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(extra_base.time, "sleep", lambda _: None)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        extra_base._chat("http://llm/v1", "secret", "qwen", "prompt", max_tokens=8, timeout=3)
    assert f"HTTP {status}" in str(caught.value)
    assert "http://llm/v1/chat/completions" in str(caught.value)
    assert "input exceeds context" in str(caught.value) if status == 400 else "bad gateway" in str(caught.value)
    assert "secret" not in str(caught.value)
    assert len(attempts) == (1 if status == 400 else 3)
