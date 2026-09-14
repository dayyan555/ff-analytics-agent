"""``build_query`` is pure: plan + catalog + as_of -> Cube JSON (or an early exit)."""

from __future__ import annotations

import json
from datetime import date

import pytest
from langgraph.runtime import Runtime

from app.agent.answer import render
from app.agent.dates import Period, PeriodError, overlap, resolve
from app.agent.nodes import build_query
from tests.conftest import AS_OF, COVERAGE, MP, plan_json

AUG = ["2026-08-01", "2026-08-31"]
JUL = ["2026-07-01", "2026-07-31"]
ALL = ["2026-03-01", "2026-08-31"]


def state_for(**plan):
    return {"question": "q", "as_of": AS_OF.isoformat(), "plan": json.loads(plan_json(**plan)), "cube_calls": 0, "notes": []}


def build(deps, **plan):
    """Run the node directly on a plan (as the model would return it)."""
    return build_query(state_for(**plan), Runtime(context=deps()))


# --------------------------------------------------------------------------- golden Cube JSON

def test_q1_spend_by_channel_in_august(deps):
    out = build(deps)  # the base plan is Q1
    assert out["cube_query"] == {
        "measures": [MP + "spend"],
        "dimensions": [MP + "channel"],
        "timeDimensions": [{"dimension": MP + "date", "dateRange": AUG}],
        "order": {MP + "spend": "desc"},
        "limit": 50,
        "timezone": "UTC",
    }
    assert (out["period"]["start"], out["period"]["end"]) == tuple(AUG)
    assert out["compare_period"] is None and "outcome" not in out


def test_q2_all_time_uses_the_coverage_window(deps):
    out = build(deps, measures=["purchases"], dimension="campaign_name", period="all_time", order_by="purchases")
    q = out["cube_query"]
    assert q["timeDimensions"] == [{"dimension": MP + "date", "dateRange": ALL}]
    assert q["order"] == {MP + "purchases": "desc"}
    assert out["period"]["label"] == "all available data"


def test_q3_ratios_travel_with_their_parts_and_rank_by_roas(deps):
    out = build(deps, measures=["roas", "cost_per_purchase"], dimension="campaign_name", period="last_month", order_by="roas")
    q = out["cube_query"]
    assert q["measures"] == [MP + m for m in ("roas", "cost_per_purchase", "revenue", "spend", "purchases")]
    assert q["order"] == {MP + "roas": "desc"}
    assert q["timeDimensions"][0]["dateRange"] == AUG  # last month relative to 2026-09-14


def test_q4_compare_uses_one_compare_date_range_ordered_by_the_dimension(deps):
    out = build(deps, intent="compare", measures=["spend", "purchases", "cost_per_purchase"], period="2026-07", compare_period="2026-08")
    assert out["cube_query"] == {
        "measures": [MP + m for m in ("spend", "purchases", "cost_per_purchase")],
        "dimensions": [MP + "channel"],
        "timeDimensions": [{"dimension": MP + "date", "compareDateRange": [JUL, AUG]}],
        "order": {MP + "channel": "asc"},
        "limit": 50,
        "timezone": "UTC",
    }
    assert out["period"]["start"] == JUL[0] and out["compare_period"]["end"] == AUG[1]


def test_e1_filter_without_dimension_groups_by_the_filtered_dimension(deps):
    out = build(deps, measures=["purchases"], dimension=None, filters=[{"dimension": "campaign_name", "value": "summer sale"}])
    q = out["cube_query"]
    assert q["dimensions"] == [MP + "campaign_name"]  # rule 13a: an empty set yields data: [], not one zero row
    assert q["filters"] == [{"member": MP + "campaign_name", "operator": "equals", "values": ["Summer Sale"]}]  # canonical casing
    assert q["order"] == {MP + "purchases": "desc"}


def test_cost_like_measures_rank_ascending_by_default(deps):
    out = build(deps, measures=["cost_per_purchase"], dimension="campaign_name", order_by="cost_per_purchase")
    assert out["cube_query"]["order"] == {MP + "cost_per_purchase": "asc"}


def test_previous_period_compares_with_the_month_before(deps):
    out = build(deps, intent="compare", measures=["spend"], period="2026-08", compare_period="previous_period")
    # periods are always reported earlier -> later, whichever one the model called "period"
    assert out["cube_query"]["timeDimensions"][0]["compareDateRange"] == [JUL, AUG]
    assert (out["period"]["start"], out["compare_period"]["start"]) == ("2026-07-01", "2026-08-01")


# --------------------------------------------------------------------------- the generalised dimensions

def test_country_filter_value_is_canonicalised(deps):
    out = build(deps, measures=["roas"], dimension="device", filters=[{"dimension": "country", "value": "de"}])
    q = out["cube_query"]
    assert q["filters"] == [{"member": MP + "country", "operator": "equals", "values": ["DE"]}]
    assert q["dimensions"] == [MP + "device"]  # the explicit dimension wins over the filtered one


def test_unknown_country_value_is_unsupported_with_the_known_values_as_a_hint(deps):
    state = state_for(measures=["roas"], dimension=None, filters=[{"dimension": "country", "value": "france"}])
    out = build_query(state, Runtime(context=deps()))
    assert out["outcome"] == "unsupported" and "cube_query" not in out
    assert out["rejected"]["filter_values"] == [{"dimension": "country", "value": "france"}]
    assert out["rejected"]["dimension"] is None and out["rejected"]["measures"] == []
    body, _ = render({**state, **out}, deps().catalog)
    assert "No country named 'france'. Known: DE, UK, US." in body


@pytest.mark.parametrize("dimension, spoken, canonical", [
    ("country", "Germany", "DE"), ("country", "united states", "US"), ("channel", "Facebook", "meta"),
    ("device", "phone", "mobile"), ("campaign_name", "brand search us", "Brand Search US"),
])
def test_common_synonyms_map_to_catalog_values(deps, dimension, spoken, canonical):
    state = state_for(measures=["spend"], dimension=None, filters=[{"dimension": dimension, "value": spoken}])
    out = build_query(state, Runtime(context=deps()))
    assert out["cube_query"]["filters"] == [{"member": MP + dimension, "operator": "equals", "values": [canonical]}]


def test_two_values_of_one_dimension_become_one_or_filter(deps):
    state = state_for(measures=["purchases"], dimension=None,
                      filters=[{"dimension": "country", "value": "US"}, {"dimension": "country", "value": "UK"}])
    out = build_query(state, Runtime(context=deps()))
    assert out["cube_query"]["filters"] == [{"member": MP + "country", "operator": "equals", "values": ["US", "UK"]}]


def test_device_filter_with_a_country_dimension_keeps_both(deps):
    out = build(deps, measures=["purchases"], dimension="country", filters=[{"dimension": "device", "value": "Mobile"}])
    q = out["cube_query"]
    assert q["dimensions"] == [MP + "country"]
    assert q["filters"] == [{"member": MP + "device", "operator": "equals", "values": ["mobile"]}]
    assert q["order"] == {MP + "purchases": "desc"}


def test_unknown_filter_dimension_is_unsupported(deps):
    out = build(deps, filters=[{"dimension": "region", "value": "EMEA"}])
    assert out["outcome"] == "unsupported" and "cube_query" not in out
    assert out["rejected"]["dimension"] == "region" and out["rejected"]["filter_values"] == []


@pytest.mark.parametrize("measure, parts", [
    ("cpm", ["spend", "impressions"]),
    ("ctr", ["clicks", "impressions"]),
    ("aov", ["revenue", "purchases"]),
])
def test_new_ratios_travel_with_their_numerator_and_denominator(deps, measure, parts):
    out = build(deps, measures=[measure], dimension="channel")
    assert out["cube_query"]["measures"] == [MP + m for m in (measure, *parts)]


def test_cpm_ranks_ascending_by_default(deps):
    out = build(deps, measures=["cpm"], dimension="channel", order_by="cpm")
    assert out["cube_query"]["order"] == {MP + "cpm": "asc"}


def test_ctr_and_aov_rank_descending_by_default(deps):
    assert build(deps, measures=["ctr"], dimension="channel", order_by="ctr")["cube_query"]["order"] == {MP + "ctr": "desc"}
    assert build(deps, measures=["aov"], dimension="objective", order_by="aov")["cube_query"]["order"] == {MP + "aov": "desc"}


def test_last_3_months_resolves_to_the_three_complete_months_before_today(deps):
    out = build(deps, measures=["roas"], dimension="device", period="last_3_months")
    assert out["cube_query"]["timeDimensions"] == [{"dimension": MP + "date", "dateRange": ["2026-06-01", "2026-08-31"]}]
    assert out["period"]["label"] == "last 3 months (June to August 2026)"
    assert out["notes"] == []


def test_quarter_token_resolves_to_the_calendar_quarter(deps):
    out = build(deps, measures=["aov"], dimension="objective", period="2026-Q2")
    assert out["cube_query"]["timeDimensions"] == [{"dimension": MP + "date", "dateRange": ["2026-04-01", "2026-06-30"]}]
    assert out["period"]["label"] == "Q2 2026" and out["notes"] == []


def test_partially_covered_quarter_queries_with_a_coverage_note(deps):
    out = build(deps, period="2026-Q3")
    assert out["cube_query"]["timeDimensions"][0]["dateRange"] == ["2026-07-01", "2026-09-30"]
    assert out["notes"] == ["data covers only 2026-07-01 to 2026-08-31 within this range"]


def test_last_13_months_asks_for_clarification(deps):
    out = build(deps, period="last_13_months")
    assert out["outcome"] == "clarify" and "cube_query" not in out
    assert out["notes"] == ["last_N_months supports 1 to 12 months"]


# --------------------------------------------------------------------------- period grammar

@pytest.mark.parametrize("token, start, end", [
    ("last_month", date(2026, 8, 1), date(2026, 8, 31)),
    ("last_1_months", date(2026, 8, 1), date(2026, 8, 31)),
    ("last_3_months", date(2026, 6, 1), date(2026, 8, 31)),
    ("last_6_months", date(2026, 3, 1), date(2026, 8, 31)),
    ("all_time", date(2026, 3, 1), date(2026, 8, 31)),
    ("2026-07", date(2026, 7, 1), date(2026, 7, 31)),
    ("2026-Q2", date(2026, 4, 1), date(2026, 6, 30)),
    ("2026-q1", date(2026, 1, 1), date(2026, 3, 31)),
    ("2026-07-10..2026-07-20", date(2026, 7, 10), date(2026, 7, 20)),
    (" 2026-08 ", date(2026, 8, 1), date(2026, 8, 31)),
])
def test_resolve_tokens(token, start, end):
    period = resolve(token, AS_OF, COVERAGE)
    assert (period.start, period.end) == (start, end)


@pytest.mark.parametrize("token", ["last_month", "last_3_months", "all_time", "2026-07", "2026-Q2", "2026-07-10..2026-07-20"])
def test_fully_covered_tokens_carry_no_coverage_note(token):
    assert resolve(token, AS_OF, COVERAGE).coverage_note is None


def test_resolve_previous_period_of_a_month_is_the_prior_month():
    base = resolve("2026-08", AS_OF, COVERAGE)
    prev = resolve("previous_period", AS_OF, COVERAGE, base=base)
    assert (prev.start, prev.end) == (date(2026, 7, 1), date(2026, 7, 31))


def test_resolve_previous_period_of_a_range_has_the_same_length():
    base = resolve("2026-08-10..2026-08-19", AS_OF, COVERAGE)
    prev = resolve("previous_period", AS_OF, COVERAGE, base=base)
    assert (prev.start, prev.end) == (date(2026, 7, 31), date(2026, 8, 9))


@pytest.mark.parametrize("token", [
    "", None, "recently", "2026-13", "2026-Q5", "last_0_months", "last_13_months", "2026-08-31..2026-08-01",
    "2025-01-01..2026-06-30", "2026-10", "2026-Q4", "previous_period",
])
def test_resolve_rejects_bad_tokens(token):
    with pytest.raises(PeriodError):
        resolve(token, AS_OF, COVERAGE)


def test_resolve_notes_partial_coverage():
    period = resolve("2026-02-15..2026-03-15", AS_OF, COVERAGE)
    assert overlap(period, COVERAGE) == "partial"
    assert period.coverage_note == "data covers only 2026-03-01 to 2026-03-15 within this range"
    q3 = resolve("2026-Q3", AS_OF, COVERAGE)
    assert q3.coverage_note == "data covers only 2026-07-01 to 2026-08-31 within this range"
    assert overlap(Period(date(2024, 1, 1), date(2024, 12, 31), "2024"), COVERAGE) == "none"


# --------------------------------------------------------------------------- early exits (no Cube query built)

def test_null_period_asks_for_clarification(deps):
    out = build(deps, period=None)
    assert out["outcome"] == "clarify" and "cube_query" not in out


def test_clarify_intent_passes_through(deps):
    assert build(deps, intent="clarify", measures=[], period=None)["outcome"] == "clarify"


@pytest.mark.parametrize("measures", [["profit_margin"], ["first_date"], ["profit_margin", "last_date"]])
def test_only_unknown_or_internal_measures_is_unsupported(deps, measures):
    out = build(deps, measures=measures)
    assert out["outcome"] == "unsupported"
    assert out["rejected"]["measures"] == measures


def test_unknown_measures_next_to_known_ones_are_dropped_with_a_note(deps):
    out = build(deps, measures=["spend", "last_date", "profit_margin"], order_by="profit_margin")
    assert "outcome" not in out
    assert out["cube_query"]["measures"] == [MP + "spend"]
    assert out["plan"]["measures"] == ["spend"] and out["plan"]["order_by"] is None
    assert out["notes"] == ["ignored unknown metric 'last_date' (not in the semantic layer)",
                            "ignored unknown metric 'profit_margin' (not in the semantic layer)"]


def test_unsupported_intent_echoes_the_unknown_name(deps):
    out = build(deps, intent="unsupported", measures=["profit_margin"], dimension="campaign_name", period="all_time")
    assert out["outcome"] == "unsupported" and out["rejected"]["measures"] == ["profit_margin"]


def test_unknown_dimension_is_unsupported(deps):
    out = build(deps, dimension="region")
    assert out["outcome"] == "unsupported" and out["rejected"]["dimension"] == "region"


@pytest.mark.parametrize("dimension", ["channel", "campaign_name", "country", "objective", "device"])
def test_every_catalog_dimension_groups(deps, dimension):
    out = build(deps, dimension=dimension)
    assert out["cube_query"]["dimensions"] == [MP + dimension]


def test_unknown_filter_value_is_unsupported(deps):
    out = build(deps, dimension=None, filters=[{"dimension": "channel", "value": "snapchat"}])
    assert out["outcome"] == "unsupported"
    assert out["rejected"]["filter_values"] == [{"dimension": "channel", "value": "snapchat"}]


def test_out_of_coverage_period_is_no_data_without_a_query(deps):
    out = build(deps, period="2024-01-01..2024-12-31")
    assert out["outcome"] == "no_data" and "cube_query" not in out
    assert out["period"]["start"] == "2024-01-01"


def test_granularity_is_unsupported(deps):
    out = build(deps, granularity="week")
    assert out["outcome"] == "unsupported" and "granularity" in out["rejected"]["feature"]


def test_overlapping_compare_periods_ask_for_clarification(deps):
    out = build(deps, intent="compare", period="2026-07", compare_period="2026-07-15..2026-08-15")
    assert out["outcome"] == "clarify" and "overlap" in out["notes"][0]


def test_compare_without_a_second_period_asks_for_clarification(deps):
    assert build(deps, intent="compare", period="2026-07", compare_period=None)["outcome"] == "clarify"


def test_compare_with_one_side_outside_coverage_still_queries(deps):
    out = build(deps, intent="compare", period="2026-02", compare_period="2026-03")
    assert "cube_query" in out and "outcome" not in out


def test_compare_with_both_sides_outside_coverage_is_no_data(deps):
    out = build(deps, intent="compare", period="2026-01", compare_period="2026-02")
    assert out["outcome"] == "no_data" and "cube_query" not in out


@pytest.mark.parametrize("text", [
    '{"intent": "clarify", "measures": null, "dimension": null, "period": null, "compare_period": null, "filters": [], '
    '"order_by": null, "direction": null, "limit": null, "granularity": null, "message": "Please specify"}}',  # null list + extra brace
    '```json\n{"intent":"clarify","measures":[],"period":null}\n```',
    'Sure, here it is: {"intent":"clarify","measures":[]} — let me know',
    '<think>{ not json</think>{"intent":"clarify","measures":[]}',
])
def test_parse_plan_tolerates_real_free_model_output(text):
    from app.models.plan import parse_plan
    assert parse_plan(text).intent == "clarify"


def test_filters_across_dimensions_are_anded_and_aliases_deduplicated(deps):
    state = state_for(measures=["purchases"], dimension=None,
                      filters=[{"dimension": "country", "value": "US"}, {"dimension": "country", "value": "usa"},
                               {"dimension": "device", "value": "phone"}])
    out = build_query(state, Runtime(context=deps()))
    assert out["cube_query"]["filters"] == [
        {"member": MP + "country", "operator": "equals", "values": ["US"]},
        {"member": MP + "device", "operator": "equals", "values": ["mobile"]},
    ]
    assert out["cube_query"]["dimensions"] == [MP + "country"]  # the first filtered dimension groups


@pytest.mark.parametrize("spoken, canonical", [
    ("Spend", "spend"), ("ROAS", "roas"), ("cost per purchase", "cost_per_purchase"), ("CPA", "cost_per_purchase"),
    ("Return on ad spend", "roas"), ("click-through rate", "ctr"), ("average order value", "aov"),
])
def test_metric_names_are_normalised_not_rejected(deps, spoken, canonical):
    out = build_query(state_for(measures=[spoken], dimension="Channel"), Runtime(context=deps()))
    assert "outcome" not in out
    assert out["cube_query"]["measures"][0] == MP + canonical and out["cube_query"]["dimensions"] == [MP + "channel"]
    assert out["plan"]["measures"] == [canonical] and out["plan"]["dimension"] == "channel"


def test_unknown_metric_stays_unknown_without_fuzzy_matching(deps):
    out = build_query(state_for(measures=["conversions"]), Runtime(context=deps()))
    assert out["outcome"] == "unsupported" and out["rejected"]["measures"] == ["conversions"]


def test_superlative_question_without_order_by_is_still_a_ranking(deps):
    state = state_for(measures=["purchases"], dimension="campaign_name", order_by=None)
    state["question"] = "Which campaign generated the most purchases?"
    out = build_query(state, Runtime(context=deps()))
    assert out["plan"]["order_by"] == "purchases" and "ranked by purchases (inferred from the question)" in out["notes"]
