"""The six tools: names, result contract, caps, error shapes."""

from __future__ import annotations

import json

import pytest

from app.tools.cube import CubeError
from app.tools.toolkit import MAX_ROWS, Toolkit
from tests.conftest import MP, VALUES, StubCube, cube_query, result_set, row


@pytest.fixture
def toolkit(catalog):
    return Toolkit(StubCube(results=[result_set([row(channel="meta", spend=13689.34), row(channel="email", spend=0)])]), catalog)


def test_tool_names_and_descriptions(toolkit):
    names = [t.name for t in toolkit.tools()]
    assert names == ["list_views", "describe_view", "search_fields", "find_dimension_values", "run_query", "final_answer"]
    for t in toolkit.tools():
        assert t.description and len(t.description) > 40, t.name
    assert set(toolkit.tools()[4].args) == {"query"}  # config is injected, not exposed to the model
    assert set(toolkit.tools()[5].args) == {"kind", "text"}


def test_list_views(toolkit):
    out = toolkit.run("list_views", {})
    assert out["ok"] and out["views"][0] == {
        "name": "marketing_performance", "description": "Daily marketing performance by campaign and channel.",
        "measures": 12, "dimensions": 5, "data_from": "2026-03-01", "data_to": "2026-08-31",
    }


def test_describe_view_lists_full_names_formats_and_example_values(toolkit):
    out = toolkit.run("describe_view", {"view": "marketing_performance"})
    assert out["ok"] and out["time_dimension"] == MP + "date" and out["data_from"] == "2026-03-01"
    spend = next(m for m in out["measures"] if m["name"] == MP + "spend")
    assert spend == {"name": MP + "spend", "title": "Spend", "aggregation": "sum", "format": "currency USD",
                     "description": "Total advertising spend in USD in the period."}
    roas = next(m for m in out["measures"] if m["name"] == MP + "roas")
    assert roas["aggregation"] == "ratio" and roas["format"] == "number"
    ctr = next(m for m in out["measures"] if m["name"] == MP + "ctr")
    assert ctr["format"] == "percent"
    campaign = next(d for d in out["dimensions"] if d["name"] == MP + "campaign_name")
    assert campaign["value_count"] == 12 and campaign["example_values"] == VALUES["campaign_name"][:3]
    assert "query_format" in out and "compareDateRange" in out["query_format"]
    assert out["showing"] == out["total"] == {"measures": 12, "dimensions": 5} and "hint" not in out


def test_describe_view_unknown(toolkit):
    out = toolkit.run("describe_view", {"view": "finance"})
    assert out == {"ok": False, "error": "unknown view 'finance'", "hint": "valid views: marketing_performance"}


@pytest.mark.parametrize("text, expected_first", [
    ("cost per acquisition", MP + "cost_per_purchase"),
    ("CPA", MP + "cost_per_purchase"),
    ("return on ad spend", MP + "roas"),
    ("average order value", MP + "aov"),
    ("campaigns", MP + "campaign_name"),
    ("spent", MP + "spend"),
])
def test_search_fields_finds_synonyms(toolkit, text, expected_first):
    out = toolkit.run("search_fields", {"text": text})
    assert out["ok"] and out["matches"][0]["name"] == expected_first, out
    assert 0 < out["matches"][0]["score"] <= 1 and out["matches"][0]["kind"] in ("measure", "dimension")


def test_search_fields_no_match_hints_unsupported(toolkit):
    out = toolkit.run("search_fields", {"text": "profit margin ebitda"})
    assert out["ok"] and out["matches"] == [] and "unsupported" in out["hint"]


def test_find_dimension_values_contains_and_aliases(toolkit):
    out = toolkit.run("find_dimension_values", {"dimension": MP + "campaign_name", "text": "summer"})
    assert out["matches"] == ["Summer Sale"] and out["total_values"] == 12 and "equals" in out["note"]
    out = toolkit.run("find_dimension_values", {"dimension": MP + "country", "text": "Germany"})
    assert out["matches"] == ["DE"]
    out = toolkit.run("find_dimension_values", {"dimension": MP + "channel", "text": "facebook"})
    assert out["matches"] == ["meta"]


def test_find_dimension_values_lists_when_no_text_and_hints_when_nothing_matches(toolkit):
    out = toolkit.run("find_dimension_values", {"dimension": MP + "channel"})
    assert out["matches"] == VALUES["channel"] and out["text"] is None
    out = toolkit.run("find_dimension_values", {"dimension": MP + "channel", "text": "snapchat"})
    assert out["ok"] and out["matches"] == [] and "no value contains 'snapchat'" in out["hint"]
    out = toolkit.run("find_dimension_values", {"dimension": MP + "spend", "text": "x"})
    assert not out["ok"] and "unknown dimension" in out["error"]


def test_run_query_returns_rows_with_short_keys_columns_and_periods(toolkit, catalog):
    out = toolkit.run("run_query", {"query": cube_query(["spend"], ["channel"])})
    assert out["ok"] and out["row_count"] == 2 and out["showing"] == 2
    assert out["rows"] == [{"channel": "meta", "spend": "13689.34"}, {"channel": "email", "spend": "0"}]
    assert out["columns"] == [
        {"key": "channel", "name": MP + "channel", "title": "Channel", "type": "string"},
        {"key": "spend", "name": MP + "spend", "title": "Spend", "type": "number", "format": "currency USD", "aggregation": "sum"},
    ]
    assert out["periods"] == [{"from": "2026-08-01", "to": "2026-08-31"}]
    assert out["query"]["timezone"] == "UTC" and out["query"]["limit"] == MAX_ROWS
    assert toolkit.cube_calls == 2 and [c[0] for c in toolkit.cube.calls] == ["dry_run", "load"]


def test_run_query_accepts_a_json_string_and_caps_the_limit(toolkit):
    out = toolkit.run("run_query", {"query": json.dumps({**cube_query(["spend"]), "limit": 500})})
    assert out["ok"] and out["query"]["limit"] == MAX_ROWS


@pytest.mark.parametrize("query, fragment", [
    ("not json", "must be a JSON object"),
    ({}, "must be a JSON object"),
    ({"sql": "select 1"}, "unknown query keys"),
])
def test_run_query_rejects_bad_shapes_before_cube(toolkit, query, fragment):
    out = toolkit.run("run_query", {"query": query})
    assert not out["ok"] and fragment in out["error"] and "hint" in out
    assert toolkit.cube.calls == []


def test_run_query_gives_cube_errors_back_with_did_you_mean(catalog):
    cube = StubCube(dry_run_error=CubeError(500, "Error: 'spent' not found for path 'marketing_performance.spent'"))
    out = Toolkit(cube, catalog).run("run_query", {"query": cube_query(["spent"])})
    assert not out["ok"] and out["did_you_mean"] == [MP + "spend"] and "describe_view" in out["hint"]
    assert out["error"].startswith("Error: 'spent' not found")


def test_run_query_format_errors_go_back_to_the_model(catalog):
    cube = StubCube(dry_run_error=CubeError(400, 'Invalid query format: "filters[0]" does not match any of the allowed types'))
    out = Toolkit(cube, catalog).run("run_query", {"query": cube_query(["spend"])})
    assert not out["ok"] and "did_you_mean" not in out and "compareDateRange" in out["hint"]


@pytest.mark.parametrize("error", [CubeError(403, "Invalid token"), CubeError(429, "quota"), CubeError(None, "ConnectError")])
def test_run_query_raises_on_auth_quota_and_transport_failures(catalog, error):
    with pytest.raises(CubeError):
        Toolkit(StubCube(error=error), catalog).run("run_query", {"query": cube_query(["spend"])})


def test_run_query_empty_result_is_ok_with_a_note(catalog):
    out = Toolkit(StubCube(results=[result_set([])]), catalog).run("run_query", {"query": cube_query(["spend"])})
    assert out["ok"] and out["row_count"] == 0 and out["rows"] == [] and "data_from" in out["note"]


def test_run_query_flattens_compare_and_granularity_rows(catalog):
    july = {**row(spend=100), "compareDateRange": "2026-07-01T00:00:00.000 - 2026-07-31T23:59:59.999"}
    august = {**row(spend=150), "compareDateRange": "2026-08-01T00:00:00.000 - 2026-08-31T23:59:59.999"}
    series = {MP + "date.month": "2026-07-01T00:00:00.000", MP + "date": "2026-07-01T00:00:00.000", MP + "spend": "100"}
    cube = StubCube(results=[[result_set([july]), result_set([august])], [result_set([series])]])
    tk = Toolkit(cube, catalog)
    out = tk.run("run_query", {"query": cube_query(["spend"])})
    assert [r["date_range"] for r in out["rows"]] == ["2026-07-01 to 2026-07-31", "2026-08-01 to 2026-08-31"]
    assert {c["key"] for c in out["columns"]} == {"spend", "date_range"}
    out = tk.run("run_query", {"query": cube_query(["spend"])})
    assert out["rows"] == [{"date_month": "2026-07-01", "spend": "100"}]  # the plain date key is dropped
    assert out["columns"][0] == {"key": "date_month", "name": MP + "date.month", "title": "date (month)", "type": "time"}


def test_unknown_tool_and_bad_arguments_are_ok_false(toolkit):
    out = toolkit.run("run_sql", {"sql": "select 1"})
    assert not out["ok"] and "unknown tool" in out["error"] and "run_query" in out["hint"]
    out = toolkit.run("describe_view", {})
    assert not out["ok"] and "invalid arguments" in out["error"]


def test_final_answer_is_a_plain_echo(toolkit):
    assert toolkit.run("final_answer", {"kind": "clarify", "text": "Which period?"}) == {"ok": True, "kind": "clarify", "text": "Which period?"}


def test_describe_view_caps_measures_and_dimensions_separately(catalog):
    from app.models.catalog import Member
    view = catalog.views["marketing_performance"]
    for i in range(45):
        view.measures[f"m{i}"] = Member(MP + f"m{i}", f"m{i}", f"M{i}", f"M{i}", "measure", "number", "sum", None, None, "")
    out = Toolkit(StubCube(), catalog).run("describe_view", {"view": "marketing_performance"})
    assert len(out["measures"]) == 40 and len(out["dimensions"]) == 5  # the dimensions are never hidden
    assert out["total"] == {"measures": 57, "dimensions": 5} and "search_fields" in out["hint"]


def test_find_dimension_values_searches_cube_when_the_cache_is_incomplete(catalog):
    view = catalog.views["marketing_performance"]
    view.values_complete["campaign_name"] = False
    cube = StubCube(results=[result_set([row(campaign_name="Summer Sale"), row(campaign_name="Summer Clearance")])])
    out = Toolkit(cube, catalog).run("find_dimension_values", {"dimension": MP + "campaign_name", "text": "summer"})
    assert out["matches"] == ["Summer Clearance", "Summer Sale"] and out["total_values"] == "12+"
    assert cube.calls[-1][1]["filters"] == [{"member": MP + "campaign_name", "operator": "contains", "values": ["summer"]}]


def test_find_dimension_values_refuses_the_time_dimension(catalog):
    from app.models.catalog import Member
    catalog.views["marketing_performance"].times["date"] = Member(MP + "date", "date", "Date", "Date", "dimension", "time", None, None, None, "Calendar day")
    out = Toolkit(StubCube(), catalog).run("find_dimension_values", {"dimension": MP + "date", "text": "2026"})
    assert not out["ok"] and "timeDimensions" in out["hint"]


def test_compare_results_share_the_row_budget(catalog):
    july = [{**row(campaign_name=f"c{i}", spend=i), "compareDateRange": "2026-07-01T00:00:00.000 - 2026-07-31T23:59:59.999"} for i in range(40)]
    august = [{**row(campaign_name=f"c{i}", spend=i), "compareDateRange": "2026-08-01T00:00:00.000 - 2026-08-31T23:59:59.999"} for i in range(40)]
    out = Toolkit(StubCube(results=[result_set(july), result_set(august)]), catalog).run("run_query", {"query": cube_query(["spend"], ["campaign_name"])})
    ranges = [r["date_range"] for r in out["rows"]]
    assert ranges.count("2026-07-01 to 2026-07-31") == 25 and ranges.count("2026-08-01 to 2026-08-31") == 25
    assert out["showing"] == 50 and not out["complete"] and "may be incomplete" in out["note"]


def test_query_errors_carry_the_status(catalog):
    cube = StubCube(dry_run_error=CubeError(500, "Error: Connection refused"))
    out = Toolkit(cube, catalog).run("run_query", {"query": cube_query(["spend"])})
    assert out["status"] == 500 and "did_you_mean" not in out
