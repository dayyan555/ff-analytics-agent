"""Shared fixtures: a hand-built catalog, a stub Cube, a scripted model.

The environment is set *before* ``app`` is imported: Langfuse reads its keys
once at first use, and with dummy keys plus ``LANGFUSE_TRACING_ENABLED=false``
it installs a no-op tracer (no network, no per-call re-initialisation).

The catalog mirrors the demo dataset (``warehouse/``) and the Cube model
(``cube/model``): one view with 12 measures, 5 string dimensions with their
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
from langchain_core.messages import AIMessage  # noqa: E402

from app.models.catalog import Catalog, Member, View  # noqa: E402
from app.models.state import Deps  # noqa: E402
from app.tools.cube import CubeError  # noqa: E402
from app.tools.llm import FakeLLM  # noqa: E402

VIEW = "marketing_performance"
MP = VIEW + "."
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
    view = View(name=VIEW, title="Marketing performance", description="Daily marketing performance by campaign and channel.",
                time_dimension=MP + "date", coverage=COVERAGE)
    for short, agg, fmt, cur, short_title, desc in MEASURES:
        view.measures[short] = Member(MP + short, short, short_title, short_title, "measure", "number", agg, fmt, cur, desc)
    for short, title, desc in DIMENSIONS:
        view.dimensions[short] = Member(MP + short, short, title, title, "dimension", "string", None, None, None, desc)
    view.values = {d: list(v) for d, v in VALUES.items()}
    view.value_counts = {d: len(v) for d, v in VALUES.items()}
    return Catalog(views={VIEW: view})


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


# --------------------------------------------------------------------------- Cube query the model would write

def cube_query(measures: list[str], dimensions: list[str] | None = None, date_range: tuple[str, str] = ("2026-08-01", "2026-08-31"),
               **extra: Any) -> dict[str, Any]:
    q: dict[str, Any] = {"measures": [MP + m for m in measures],
                         "timeDimensions": [{"dimension": MP + "date", "dateRange": list(date_range)}]}
    if dimensions:
        q["dimensions"] = [MP + d for d in dimensions]
    q.update(extra)
    return q


# --------------------------------------------------------------------------- scripted model replies

def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "call-1") -> AIMessage:
    """An AIMessage carrying one native tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


def final(text: str, kind: str = "answer", call_id: str = "call-final") -> AIMessage:
    return tool_call("final_answer", {"kind": kind, "text": text}, call_id)


def json_call(name: str, args: dict[str, Any] | None = None) -> str:
    """The same call written with the JSON tool protocol (plain text)."""
    return json.dumps({"tool": name, "args": args or {}})


class StubCube:
    """Offline stand-in for ``app.tools.cube.Cube``: canned results per load, or a raised error; records calls."""

    def __init__(self, results: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
                 error: Exception | None = None, dry_run_error: Exception | None = None):
        # one list of result sets for every load, or a queue of them (one entry per load, in order)
        self._queue = list(results) if results and isinstance(results[0], list) else None
        self.results = results if self._queue is None else None
        self.error = error
        self.dry_run_error = dry_run_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def base_url(self) -> str:
        return "http://stub"

    def ready(self) -> bool:
        return True

    def meta(self) -> dict[str, Any]:
        return {"cubes": []}

    def dry_run(self, query: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.calls.append(("dry_run", query))
        if self.dry_run_error:
            raise self.dry_run_error
        if self.error:
            raise self.error
        return {"normalizedQueries": [query]}

    def load(self, query: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.calls.append(("load", query))
        if self.error:
            raise self.error
        if self._queue is not None:
            return {"results": self._queue.pop(0) if self._queue else []}
        return {"results": self.results or []}


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
        if llm is None or isinstance(llm, (str, list, AIMessage)):
            llm = FakeLLM([llm] if isinstance(llm, (str, AIMessage)) else llm)
        return Deps(llm=llm, cube=cube or StubCube(), catalog=catalog, as_of=as_of)

    return make


__all__ = ["CubeError"]
