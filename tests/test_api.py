"""The FastAPI layer with injected fakes: one response shape on every path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.tools.cube import CubeError
from app.tools.llm import FakePlanLLM
from app.web.api import PAYLOAD_KEYS, create_app, http_status
from tests.conftest import VALUES, StubCube, plan_json, result_set, row

Q1_ROWS = [row(channel="meta", spend="20345.12"), row(channel="google", spend="13108.40")]


def client_for(catalog, llm=None, cube=None, loader=None) -> TestClient:
    app = create_app(cube=cube or StubCube([result_set(Q1_ROWS)]), llm=llm or FakePlanLLM([plan_json()]),
                     loader=loader or (lambda c: catalog))
    return TestClient(app)


def test_health_reports_cube_catalog_and_tracing(catalog):
    with client_for(catalog) as client:
        body = client.get("/health").json()
    assert body["cube"] == "ok" and body["cube_url_kind"] == "local"
    assert body["catalog"]["loaded"] is True
    assert body["catalog"]["measures"] == list(catalog.measures)
    assert body["catalog"]["dimensions"] == ["channel", "campaign_name", "country", "objective", "device"]
    assert body["catalog"]["values"] == VALUES
    assert body["catalog"]["coverage"] == {"first": "2026-03-01", "last": "2026-08-31"}
    assert body["catalog_error"] is None
    assert body["langfuse"] == "disabled" and body["model"] == "openrouter/free"
    assert body["as_of"] == "2026-09-14" and body["busy"] is False


def test_examples_lists_the_nine_questions(catalog):
    with client_for(catalog) as client:
        examples = client.get("/examples").json()["examples"]
    assert len(examples) == 9
    assert set(examples[0]) == {"question", "kind", "expect"}
    assert [e["expect"] for e in examples] == ["answer"] * 6 + ["unsupported", "clarify", "no_data"]
    assert [e["kind"] for e in examples] == ["normal"] * 5 + ["compare", "unsupported", "ambiguous", "empty"]


def test_index_serves_the_ui(catalog):
    with client_for(catalog) as client:
        resp = client.get("/")
    assert resp.status_code == 200 and resp.headers["cache-control"] == "no-store"
    assert 'id="ask-btn"' in resp.text


def test_ask_clarify_is_200_without_cube_calls(catalog):
    cube = StubCube()
    llm = FakePlanLLM([plan_json(intent="clarify", measures=[], period=None, message="which period?")])
    with client_for(catalog, llm=llm, cube=cube) as client:
        resp = client.post("/ask", json={"question": "How did we do recently?"})
    body = resp.json()
    assert resp.status_code == 200 and body["outcome"] == "clarify"
    assert body["cube_calls"] == 0 and cube.calls == [] and body["llm_calls"] == 1
    assert body["error_kind"] is None and body["error_stage"] is None


def test_ask_q1_answer_payload(catalog):
    with client_for(catalog) as client:
        resp = client.post("/ask", json={"question": "How much did we spend by channel in August 2026?", "as_of": "2026-09-14"})
    body = resp.json()
    assert resp.status_code == 200
    assert tuple(body) == PAYLOAD_KEYS
    assert body["outcome"] == "answer" and body["as_of"] == "2026-09-14"
    assert "$20,345.12" in body["answer_body"] and body["answer"] == f"{body['answer_body']}\n\n{body['footer']}"
    assert body["plan"]["measures"] == ["spend"] and body["filters"] == []
    assert body["cube_query"]["timeDimensions"][0]["dateRange"] == ["2026-08-01", "2026-08-31"]
    assert len(body["normalized"]) == 1 and body["rows"] == [Q1_ROWS]
    assert body["result"]["totals"] == {"spend": "33453.52"}  # Decimal -> str
    assert body["llm_model"] == "fake/model:free" and body["llm_cost"] == "0"
    assert body["llm_calls"] == 1 and body["cube_calls"] == 2
    assert body["trace_id"] is None and body["trace_url"] is None  # tracing disabled
    assert isinstance(body["timing_ms"], int)


def test_ask_cube_error_is_503(catalog):
    llm = FakePlanLLM([plan_json()])
    with client_for(catalog, llm=llm, cube=StubCube(error=CubeError(503, "upstream unavailable"))) as client:
        resp = client.post("/ask", json={"question": "spend by channel"})
    body = resp.json()
    assert resp.status_code == 503
    assert (body["outcome"], body["error_kind"], body["error_stage"]) == ("error", "cube", "graph")
    assert body["cube_calls"] == 1 and body["llm_calls"] == 1


def test_catalog_failure_is_503_and_retried_lazily(catalog):
    def broken(cube):
        raise CubeError(403, "Forbidden")

    llm = FakePlanLLM([plan_json()])
    with client_for(catalog, llm=llm, loader=broken) as client:
        assert client.get("/health").json()["catalog"] == {"loaded": False}
        resp = client.post("/ask", json={"question": "spend by channel"})
        body = resp.json()
        assert resp.status_code == 503
        assert (body["outcome"], body["error_kind"], body["error_stage"]) == ("error", "cube", "catalog")
        assert "403: Forbidden" in body["error"] and body["answer"] == body["error"]
        assert tuple(body) == PAYLOAD_KEYS and body["llm_calls"] == 0 and body["cube_calls"] == 0
        assert llm.calls == 0  # the graph was never invoked

        client.app.state.loader = lambda c: catalog  # the semantic layer comes back
        resp = client.post("/ask", json={"question": "spend by channel"})
        assert resp.status_code == 200 and resp.json()["outcome"] == "answer"
        assert client.get("/health").json()["catalog_error"] is None


@pytest.mark.parametrize("payload", [{"question": ""}, {}, {"question": "x" * 501}, {"question": "q", "as_of": "not-a-date"}])
def test_bad_request_keeps_the_payload_shape(catalog, payload):
    with client_for(catalog) as client:
        resp = client.post("/ask", json=payload)
    body = resp.json()
    assert resp.status_code == 400
    assert tuple(body) == PAYLOAD_KEYS
    assert (body["outcome"], body["error_kind"]) == ("error", "bad_request")
    assert body["answer"].startswith("Invalid request")


@pytest.mark.parametrize("outcome, kind, status", [
    ("answer", None, 200), ("clarify", None, 200), ("unsupported", None, 200), ("no_data", None, 200),
    ("error", "cube", 503), ("error", "llm", 502), ("error", "free_guard", 502), ("error", "validation", 502),
    ("error", "internal", 500), ("error", "busy", 409), ("error", "bad_request", 400),
    ("error", None, 500), ("error", "something-else", 500),
])
def test_http_status_table(outcome, kind, status):
    assert http_status(outcome, kind) == status
