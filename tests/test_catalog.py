"""``load_catalog`` reads every view from ``/meta`` and, per view, the known values plus coverage from ONE grouped ``/load``.

A tiny fake Cube pins the startup query shape: the string dimensions all
travel in ``dimensions``, the measures are only ``first_date``/``last_date``,
and no time dimension is set (the whole warehouse, not a window). The catalog's
own search and did-you-mean helpers are covered at the end.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.models.catalog import Catalog
from app.tools.cube import CubeError, load_catalog
from tests.conftest import DIMENSIONS, MEASURES, MP, VALUES, build_catalog, row

STRING_DIMENSIONS = [d for d, _, _ in DIMENSIONS]


def meta_payload() -> dict[str, Any]:
    """The subset of Cube's ``/meta`` the loader reads, for the view plus one unrelated cube."""
    measures = [
        {"name": MP + short, "title": title, "shortTitle": title, "type": "number", "aggType": agg,
         "format": fmt, "description": desc, **({"currency": cur} if cur else {})}
        for short, agg, fmt, cur, title, desc in MEASURES
    ] + [
        {"name": MP + "first_date", "title": "First date", "type": "time", "aggType": "time", "description": "Earliest day with data"},
        {"name": MP + "last_date", "title": "Last date", "type": "time", "aggType": "time", "description": "Latest day with data"},
    ]
    dimensions = [
        {"name": MP + short, "title": title, "shortTitle": title, "type": "string", "description": desc}
        for short, title, desc in DIMENSIONS
    ] + [{"name": MP + "date", "title": "Date", "type": "time", "description": "Calendar day (UTC)."}]
    return {"cubes": [
        {"name": "campaign_daily", "type": "cube", "measures": [{"name": "campaign_daily.spend", "type": "number"}], "dimensions": []},
        {"name": "marketing_performance", "type": "view", "title": "Marketing performance",
         "description": "Daily marketing performance by campaign and channel.", "measures": measures, "dimensions": dimensions},
    ]}


# One grouped row per (campaign, device) for a handful of campaigns; values deliberately unsorted and repeated.
GROUPED_ROWS = [
    row(channel="meta", campaign_name="Summer Sale", country="UK", objective="conversion", device="mobile",
        first_date="2026-07-10T00:00:00.000", last_date="2026-07-31T00:00:00.000"),
    row(channel="google", campaign_name="Brand Search US", country="US", objective="conversion", device="desktop",
        first_date="2026-03-01T00:00:00.000", last_date="2026-08-31T00:00:00.000"),
    row(channel="google", campaign_name="Brand Search US", country="US", objective="conversion", device="mobile",
        first_date="2026-03-01T00:00:00.000", last_date="2026-08-31T00:00:00.000"),
    row(channel="tiktok", campaign_name="TikTok Creator DE", country="DE", objective="conversion", device="mobile",
        first_date="2026-06-01T00:00:00.000", last_date="2026-08-31T00:00:00.000"),
    row(channel="linkedin", campaign_name="LinkedIn Thought Leadership", country="US", objective="awareness", device="desktop",
        first_date="2026-03-01T00:00:00.000", last_date="2026-05-31T00:00:00.000"),
    row(channel="email", campaign_name="Newsletter Promo", country="DE", objective="retention", device="mobile",
        first_date="2026-03-02T00:00:00.000", last_date="2026-08-30T00:00:00.000"),
    row(channel="email", campaign_name="Winback Series", country="US", objective="retention", device=None,  # a NULL group
        first_date=None, last_date=None),
]


class FakeCube:
    """Just enough of ``app.tools.cube.Cube`` for the loader: ``meta()`` and ``load()``, recording the load query."""

    def __init__(self, meta: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None):
        self._meta = meta_payload() if meta is None else meta
        self._rows = GROUPED_ROWS if rows is None else rows
        self.loads: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return "http://fake"

    def meta(self) -> dict[str, Any]:
        return self._meta

    def load(self, query: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.loads.append(query)
        return {"results": [{"data": self._rows, "annotation": {}}]}


def test_load_catalog_builds_values_for_every_dimension_and_the_coverage():
    cube = FakeCube()
    cat = load_catalog(cube)
    assert isinstance(cat, Catalog) and list(cat.views) == ["marketing_performance"]  # cubes are not exposed
    view = cat.views["marketing_performance"]
    assert view.description == "Daily marketing performance by campaign and channel."
    assert list(view.measures) == [m[0] for m in MEASURES]  # INTERNAL first_date/last_date excluded
    assert list(view.dimensions) == STRING_DIMENSIONS and view.time_dimension == MP + "date"
    assert view.values == {
        "channel": ["email", "google", "linkedin", "meta", "tiktok"],
        "campaign_name": ["Brand Search US", "LinkedIn Thought Leadership", "Newsletter Promo", "Summer Sale",
                          "TikTok Creator DE", "Winback Series"],
        "country": ["DE", "UK", "US"],
        "objective": ["awareness", "conversion", "retention"],
        "device": ["desktop", "mobile"],  # the NULL group contributes no value
    }
    assert view.value_counts["campaign_name"] == 6
    assert view.coverage == (date(2026, 3, 1), date(2026, 8, 31))
    assert set(view.values) == set(VALUES)


def test_load_catalog_issues_one_grouped_load_over_all_string_dimensions():
    cube = FakeCube()
    load_catalog(cube)
    assert len(cube.loads) == 1
    query = cube.loads[0]
    assert query["measures"] == [MP + "first_date", MP + "last_date"]
    assert sorted(query["dimensions"]) == sorted(MP + d for d in STRING_DIMENSIONS)
    assert "timeDimensions" not in query and "filters" not in query
    assert query["timezone"] == "UTC" and query["limit"] >= 100  # more than enough for every group


def test_load_catalog_keeps_member_metadata_from_meta():
    cat = load_catalog(FakeCube())
    cpm = cat.member(MP + "cpm")
    assert (cpm.format, cpm.currency, cpm.agg_type, cpm.kind) == ("currency_2", "USD", "number", "measure")
    assert cpm.description == "spend per 1,000 impressions; null when impressions = 0."
    assert cat.member(MP + "country").description == "Country the campaign runs in: US, UK or DE"
    assert cat.summary() == {"views": [{"name": "marketing_performance", "description": "Daily marketing performance by campaign and channel.",
                                        "measures": 12, "dimensions": 5, "data_from": "2026-03-01", "data_to": "2026-08-31"}]}


def test_load_catalog_fails_without_views():
    with pytest.raises(CubeError, match="no views"):
        load_catalog(FakeCube(meta={"cubes": [{"name": "other", "type": "cube", "measures": [], "dimensions": []}]}))


def test_load_catalog_fails_on_an_empty_warehouse():
    with pytest.raises(CubeError, match="empty"):
        load_catalog(FakeCube(rows=[]))


# --------------------------------------------------------------------------- search and suggestions

def test_search_scores_exact_synonyms_highest_and_partial_matches_lower():
    cat = build_catalog()
    hits = cat.search("cost per acquisition")
    assert hits[0][0].short == "cost_per_purchase" and hits[0][1] == 1.0
    assert all(score <= 1.0 for _, score in hits)
    assert cat.search("orders")[0][0].short == "purchases"
    assert cat.search("spend")[0][0].short == "spend"  # the name beats descriptions that mention spend
    partial = cat.search("purchase revenue")
    assert partial[0][0].short == "revenue" and partial[0][1] < 1.0  # both words, but not an exact name
    assert cat.search("") == [] and cat.search("zzz") == []


def test_search_can_be_scoped_to_a_view():
    cat = build_catalog()
    assert cat.search("spend", view="marketing_performance")[0][0].short == "spend"
    assert cat.search("spend", view="nope") == cat.search("spend")  # unknown view: search everything


def test_suggest_prefers_close_short_names():
    cat = build_catalog()
    assert cat.suggest(MP + "spent") == [MP + "spend"]
    assert cat.suggest("purchase")[0] == MP + "purchases"
    assert cat.suggest("marketing_performance.zzz") == []
