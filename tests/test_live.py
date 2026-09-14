"""Opt-in live checks against the real Cube deployment and OpenRouter: ``RUN_LIVE=1 uv run pytest tests/test_live.py``.

Configuration comes from ``.env`` / the shell (see ``LIVE_ENV`` in conftest);
the process env itself is pinned to stub values for the offline suite. Budget:
2 Cube calls for the catalog, then 1 LLM request + 2 Cube calls for one question.
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.models.state import Deps
from app.runtime import run_question
from app.tools.cube import Cube, CubeClient, load_catalog
from app.tools.llm import OpenRouterPlanLLM
from tests.conftest import AS_OF, LIVE_ENV

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
    assert set(catalog.measures) >= {"spend", "purchases", "revenue", "roas", "cost_per_purchase"}
    assert set(catalog.dimensions) == {"channel", "campaign_name", "country", "objective", "device"}
    assert set(catalog.values) == set(catalog.dimensions)
    assert all(catalog.values[d] for d in catalog.dimensions)
    assert catalog.values["country"] == ["DE", "UK", "US"] and catalog.values["device"] == ["desktop", "mobile"]
    first, last = catalog.coverage
    assert first <= last


def test_cube_cloud_rejects_unsigned_requests():
    if "cubecloudapp.dev" not in CUBE_URL:
        pytest.skip("only Cube Cloud always enforces JWT auth")
    assert httpx.get(f"{CUBE_URL}/cubejs-api/v1/meta", timeout=30).status_code == 403


def test_one_question_through_the_free_router(cube, catalog):
    assert LIVE_ENV.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is required"
    llm = OpenRouterPlanLLM(LIVE_ENV["OPENROUTER_API_KEY"], LIVE_ENV.get("OPENROUTER_APP_TITLE", "ff-analytics-agent"),
                            LIVE_ENV.get("OPENROUTER_APP_URL", "https://github.com"))
    deps = Deps(llm=llm, cube=cube, catalog=catalog, as_of=AS_OF)
    out = run_question("How much did we spend by channel in August 2026?", deps)
    assert out["outcome"] != "error", out.get("error")
    assert out["llm_model"].endswith(":free")
    assert out["llm_cost"] == "0"
    assert out["cube_calls"] == 2
