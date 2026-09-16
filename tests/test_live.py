"""Opt-in live checks against the real Cube deployment and OpenRouter: ``RUN_LIVE=1 uv run pytest tests/test_live.py``.

Configuration comes from ``.env`` / the shell (see ``LIVE_ENV`` in conftest);
the process env itself is pinned to stub values for the offline suite. Budget:
2 Cube calls for the catalog, a few Cube calls for the tools, then one question
through the free router (usually 3 model turns, plus any explicit retry).
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.models.state import Deps
from app.runtime import run_question
from app.tools.cube import Cube, CubeClient, load_catalog
from app.tools.llm import OpenRouterLLM
from app.tools.toolkit import Toolkit
from evals.agent_eval import CASES, check
from tests.conftest import AS_OF, LIVE_ENV, MP

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to hit Cube and OpenRouter")

CUBE_URL = LIVE_ENV.get("CUBE_URL", "").rstrip("/")


@pytest.fixture(scope="module")
def cube() -> Cube:
    assert CUBE_URL and LIVE_ENV.get("CUBEJS_API_SECRET"), "CUBE_URL and CUBEJS_API_SECRET are required"
    return Cube(CubeClient(CUBE_URL, LIVE_ENV["CUBEJS_API_SECRET"]))


@pytest.fixture(scope="module")
def catalog(cube):
    return load_catalog(cube)


def test_catalog_shape(catalog):
    view = catalog.views["marketing_performance"]
    assert set(view.measures) >= {"spend", "purchases", "revenue", "roas", "cost_per_purchase"}
    assert set(view.dimensions) == {"channel", "campaign_name", "country", "objective", "device"}
    assert all(view.values[d] for d in view.dimensions)
    assert view.values["country"] == ["DE", "UK", "US"] and view.values["device"] == ["desktop", "mobile"]
    first, last = view.coverage
    assert first <= last


def test_tools_against_the_real_semantic_layer(cube, catalog):
    tk = Toolkit(cube, catalog)
    assert tk.run("describe_view", {"view": "marketing_performance"})["measures"]
    bad = tk.run("run_query", {"query": {"measures": [MP + "spent"]}})
    assert bad["ok"] is False and bad["did_you_mean"] == [MP + "spend"]
    good = tk.run("run_query", {"query": {"measures": [MP + "spend"], "dimensions": [MP + "channel"],
                                          "timeDimensions": [{"dimension": MP + "date", "dateRange": ["2026-08-01", "2026-08-31"]}]}})
    assert good["ok"] and good["row_count"] >= 1 and good["periods"] == [{"from": "2026-08-01", "to": "2026-08-31"}]
    # Exercise the reviewed zero-versus-empty behavior against Cube itself.
    for channel, period, present in [("email", ["2026-08-01", "2026-08-31"], True),
                                     ("email", ["2025-01-01", "2025-12-31"], False)]:
        result = tk.run("run_query", {"query": {
            "measures": [MP + "spend"],
            "filters": [{"member": MP + "channel", "operator": "equals", "values": [channel]}],
            "timeDimensions": [{"dimension": MP + "date", "dateRange": period}],
        }})
        assert result["ok"] and result["has_data"] is present


def test_cube_cloud_rejects_unsigned_requests():
    if "cubecloudapp.dev" not in CUBE_URL:
        pytest.skip("only Cube Cloud always enforces JWT auth")
    assert httpx.get(f"{CUBE_URL}/cubejs-api/v1/meta", timeout=30).status_code == 403


def test_one_question_through_the_free_router(cube, catalog):
    assert LIVE_ENV.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is required"
    llm = OpenRouterLLM(LIVE_ENV["OPENROUTER_API_KEY"], LIVE_ENV.get("OPENROUTER_APP_TITLE", "ff-analytics-agent"),
                        LIVE_ENV.get("OPENROUTER_APP_URL", "https://github.com"))
    deps = Deps(llm=llm, cube=cube, catalog=catalog, as_of=AS_OF)
    import json
    case = json.loads(CASES.read_text())[0]  # July total, independently calculated from the seed data
    out = run_question(case["question"], deps)
    assert check(out, case["expect"]) == [], check(out, case["expect"])
    assert all(m.endswith(":free") for m in out["llm_models"])
    assert out["llm_cost"] == "0"
    assert any(q["ok"] for q in out["queries"])
