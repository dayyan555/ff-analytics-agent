"""The real Cube REST client against a mocked transport: auth header, Continue wait, errors."""

from __future__ import annotations

import json

import httpx
import jwt
import pytest

import app.tools.cube as cube_module
from app.tools.cube import Cube, CubeClient, CubeError


def make_client(handler) -> CubeClient:
    client = CubeClient("http://cube.test/", "top-secret-of-at-least-thirty-two-bytes", max_wait_s=5)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_requests_are_signed_with_an_hs256_jwt_without_bearer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"cubes": []})

    make_client(handler).meta()
    assert not seen["auth"].startswith("Bearer ")
    assert jwt.decode(seen["auth"], "top-secret-of-at-least-thirty-two-bytes", algorithms=["HS256"])["exp"] > 0


def test_load_polls_while_cube_says_continue_wait(monkeypatch):
    monkeypatch.setattr(cube_module.time, "sleep", lambda s: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) < 3:
            return httpx.Response(200, json={"error": "Continue wait"})
        return httpx.Response(200, json={"queryType": "regularQuery", "results": [{"data": [{"x": "1"}]}]})

    body = make_client(handler).load({"measures": ["m.x"]})
    assert body["results"][0]["data"] == [{"x": "1"}]
    assert len(calls) == 3 and all(c["queryType"] == "multi" for c in calls)  # identical re-sends


def test_continue_wait_gives_up_after_the_deadline(monkeypatch):
    monkeypatch.setattr(cube_module.time, "sleep", lambda s: None)
    ticks = iter(range(0, 100, 3))
    monkeypatch.setattr(cube_module.time, "monotonic", lambda: next(ticks))
    client = make_client(lambda r: httpx.Response(200, json={"error": "Continue wait"}))
    with pytest.raises(CubeError) as info:
        client.load({"measures": ["m.x"]})
    assert "still computing" in info.value.message


@pytest.mark.parametrize("status, body, expect", [
    (403, {"error": "Invalid token"}, "403: Invalid token"),
    (400, {"error": "'nope' not found for path 'mp.nope'"}, "400: 'nope' not found"),
    (502, "<html>bad gateway</html>", "502: <html>bad gateway"),
])
def test_http_errors_become_cube_errors(status, body, expect):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)

    with pytest.raises(CubeError) as info:
        make_client(handler).dry_run({"measures": ["m.x"]})
    assert str(info.value).startswith(expect)


def test_transport_failures_become_cube_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(CubeError) as info:
        make_client(handler).meta()
    assert info.value.status is None and "ConnectError" in info.value.message


def test_missing_secret_is_a_clear_error():
    with pytest.raises(CubeError, match="CUBEJS_API_SECRET"):
        CubeClient("http://cube.test", "")


def test_tools_wrap_the_client_and_are_named_for_the_trace():
    client = make_client(lambda r: httpx.Response(200, json={"results": [{"data": []}]}))
    cube = Cube(client)
    assert [t.name for t in (cube.dry_run_tool, cube.load_tool)] == ["cube_dry_run", "cube_load"]
    assert cube.load({"measures": ["m.x"]}) == {"results": [{"data": []}]}
