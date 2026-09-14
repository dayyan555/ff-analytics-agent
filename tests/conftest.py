"""Shared fixtures: a hand-built catalog, a stub Cube, canned plans.

The environment is set *before* ``app`` is imported: Langfuse reads its keys
once at first use, and with dummy keys plus ``LANGFUSE_TRACING_ENABLED=false``
it installs a no-op tracer (no network, no per-call re-initialisation).

The catalog mirrors the demo dataset (``warehouse/``) and the Cube model
(``cube/model``): 12 LLM-visible measures, 5 string dimensions with their
known values, and data from 2026-03-01 to 2026-08-31.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# The real configuration, for the opt-in live tests only (RUN_LIVE=1); captured before the stub values below.
LIVE_ENV: dict[str, str] = {**{k: v for k, v in dotenv_values(Path(__file__).resolve().parents[1] / ".env").items() if v},
                            **os.environ}

os.environ.update(
    LANGFUSE_PUBLIC_KEY="pk-lf-test",
    LANGFUSE_SECRET_KEY="sk-lf-test",
    LANGFUSE_TRACING_ENABLED="false",
    CUBE_URL="http://stub",
    CUBEJS_API_SECRET="x",
    OPENROUTER_API_KEY="sk-or-test",
)

import json  # noqa: E402
from datetime import date  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from app.models.catalog import Catalog, Member  # noqa: E402
from app.models.state import Deps  # noqa: E402
from app.tools.llm import FakePlanLLM  # noqa: E402

MP = "marketing_performance."
AS_OF = date(2026, 9, 14)
COVERAGE = (date(2026, 3, 1), date(2026, 8, 31))

CAMPAIGNS = sorted([
    "Brand Search US", "Generic Search US", "Shopping DE", "Prospecting Video", "Retargeting Carousel", "Summer Sale",
    "TikTok Spark UK", "TikTok Creator DE", "LinkedIn Leads UK", "LinkedIn Thought Leadership", "Winback Series",
    "Newsletter Promo",
])
VALUES: dict[str, list[str]] = {
    "channel": ["email", "google", "linkedin", "meta", "tiktok"],
    "campaign_name": CAMPAIGNS,
    "country": ["DE", "UK", "US"],
    "objective": ["awareness", "conversion", "retention"],
    "device": ["desktop", "mobile"],
}

# short, agg_type, format, currency, short_title, description  (mirrors cube/model/cubes/campaign_daily.yml)
MEASURES = [
    ("spend", "sum", "currency_2", "USD", "Spend", "Total advertising spend in USD in the period."),
    ("impressions", "sum", "number_0", None, "Impressions", "Ad impressions in the period."),
    ("clicks", "sum", "number_0", None, "Clicks", "Ad clicks in the period."),
    ("purchases", "sum", "number_0", None, "Purchases",
     "Purchase events attributed to the campaign, counted on the purchase date (UTC)."),
    ("revenue", "sum", "currency_2", "USD", "Revenue", "Purchase revenue in USD (no refunds)."),
    ("cost_per_purchase", "number", "currency_2", "USD", "Cost per purchase",
     "spend / purchases over the same period and grouping; null when spend = 0 or purchases = 0."),
    ("roas", "number", "number_2", None, "ROAS", "revenue / spend; null when spend = 0."),
    ("cpc", "number", "currency_2", "USD", "CPC", "spend / clicks; null when clicks = 0."),
    ("cpm", "number", "currency_2", "USD", "CPM", "spend per 1,000 impressions; null when impressions = 0."),
    ("ctr", "number", "percent_2", None, "CTR", "clicks / impressions; null when impressions = 0."),
    ("conversion_rate", "number", "percent_1", None, "Conversion rate", "purchases / clicks; null when clicks = 0."),
    ("aov", "number", "currency_2", "USD", "Average order value", "revenue / purchases; null when purchases = 0."),
]
# short, title, description  (mirrors cube/model/cubes/campaigns.yml and campaign_daily.yml)
DIMENSIONS = [
    ("channel", "Channel", "Marketing channel: google, meta, tiktok, linkedin or email"),
    ("campaign_name", "Campaign name", "Campaign name"),
    ("country", "Country", "Country the campaign runs in: US, UK or DE"),
    ("objective", "Objective", "Campaign objective: awareness, conversion or retention"),
    ("device", "Device", "Device the ad was served on / the purchase was made on: mobile or desktop"),
]


def build_catalog() -> Catalog:
    cat = Catalog()
    for short, agg, fmt, cur, short_title, desc in MEASURES:
        cat.measures[short] = Member(MP + short, short, short_title, short_title, "number", agg, fmt, cur, desc)
    for short, title, desc in DIMENSIONS:
        cat.dimensions[short] = Member(MP + short, short, title, title, "string", None, None, None, desc)
    cat.values = {d: list(v) for d, v in VALUES.items()}
    cat.coverage = COVERAGE
    return cat


def build_annotation() -> dict[str, Any]:
    """What Cube's ``/load`` returns alongside the rows (titles and formats per member)."""
    return {
        "measures": {
            MP + short: {"title": short_title, "shortTitle": short_title, "description": desc, "type": "number",
                         "format": fmt, **({"currency": cur} if cur else {})}
            for short, _agg, fmt, cur, short_title, desc in MEASURES
        },
        "dimensions": {MP + short: {"title": title, "shortTitle": title, "type": "string"} for short, title, _ in DIMENSIONS},
        "timeDimensions": {},
    }


def result_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One entry of Cube's ``results`` list."""
    return {"data": rows, "annotation": build_annotation()}


def row(**values: Any) -> dict[str, Any]:
    """A Cube row: short member names -> fully-qualified keys; numbers become strings like Cube sends them."""
    return {MP + k: (None if v is None else (v if isinstance(v, str) else str(v))) for k, v in values.items()}


BASE_PLAN = {
    "intent": "query", "measures": ["spend"], "dimension": "channel", "period": "2026-08", "compare_period": None,
    "filters": [], "order_by": None, "direction": None, "limit": None, "granularity": None, "message": None,
}


def plan_json(**overrides: Any) -> str:
    """A complete Plan as the model would return it, with overrides."""
    return json.dumps({**BASE_PLAN, **overrides})


class StubCube:
    """Offline stand-in for ``app.tools.cube.Cube``: canned results or a raised error; records calls."""

    def __init__(self, results: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def base_url(self) -> str:
        return "http://stub"

    def ready(self) -> bool:
        return True

    def dry_run(self, query: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.calls.append(("dry_run", query))
        if self.error:
            raise self.error
        return {"normalizedQueries": [query]}

    def load(self, query: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.calls.append(("load", query))
        if self.error:
            raise self.error
        return {"results": self.results}


@pytest.fixture
def catalog() -> Catalog:
    return build_catalog()


@pytest.fixture
def annotation() -> dict[str, Any]:
    return build_annotation()


@pytest.fixture
def deps(catalog: Catalog):
    """Factory: ``deps(replies_or_llm, cube=StubCube())`` -> Deps with the fixture catalog."""

    def make(llm: Any = None, cube: Any = None, as_of: date = AS_OF) -> Deps:
        if llm is None or isinstance(llm, (str, list)):
            llm = FakePlanLLM([llm] if isinstance(llm, str) else llm)
        return Deps(llm=llm, cube=cube or StubCube(), catalog=catalog, as_of=as_of)

    return make
