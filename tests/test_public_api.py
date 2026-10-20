"""公网验收入口必须拒绝不可达目标及 HTTP 200 的空检索假成功。"""

import httpx
import pytest
from eval.smoke.public_api import run_checks
from fastapi.testclient import TestClient

from tianximem.service import create_app


@pytest.mark.parametrize("failure", ["http_503", "connection_error"])
def test_unhealthy_target_stops_before_writing(failure):
    seen = []

    def handle(request):
        seen.append(request.url.path)
        if failure == "connection_error":
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(503, json={"detail": "unavailable"})

    with httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handle)) as client:
        report, code = run_checks(client, nonce="test-public-offline")
    assert code == 2
    assert seen == ["/health"]
    assert len(report.failed) == 1


@pytest.mark.parametrize("empty_search", [False, True])
def test_public_checks_require_retrieved_content(wired_dated, monkeypatch, empty_search):
    # 通用夹具要求用例指定召回项；这里让假索引的写入立即出现在假检索中。
    qdrant = wired_dated.qdrant
    original_index = qdrant.index_pairs

    def index_pairs(pairs, *args, **kwargs):
        pairs = list(pairs)
        count = original_index(pairs, *args, **kwargs)
        for pair in pairs:
            ids = qdrant.by_user.setdefault(pair.user_id, [])
            if pair.id not in ids:
                ids.append(pair.id)
        return count

    monkeypatch.setattr(qdrant, "index_pairs", index_pairs)
    with TestClient(create_app(wired_dated.services)) as client:
        if empty_search:
            original_post = client.post

            def post(path, **kwargs):
                if path == "/search":
                    return httpx.Response(200, json={"data": []})
                return original_post(path, **kwargs)

            monkeypatch.setattr(client, "post", post)
        report, code = run_checks(client, nonce="test-public-contract")
    assert len(report.results) == 15
    if empty_search:
        assert code == 1
        assert any("立即可检索" in result.name for result in report.failed)
    else:
        assert code == 0, report.render()
        assert report.failed == []
