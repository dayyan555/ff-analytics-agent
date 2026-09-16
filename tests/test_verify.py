"""The numbers check: what counts as backed by the data and what does not."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.agent.verify import derivable, numbers_in, unverified_numbers

QUERY = {
    "ok": True, "row_count": 3,
    "columns": [{"key": "channel", "type": "string"}, {"key": "spend", "type": "number", "aggregation": "sum"},
                {"key": "purchases", "type": "number", "aggregation": "sum"},
                {"key": "ctr", "type": "number", "format": "percent", "aggregation": "ratio"}],
    "rows": [{"channel": "meta", "spend": "13689.34", "purchases": "412", "ctr": "0.0345"},
             {"channel": "google", "spend": "11020.10", "purchases": "610", "ctr": "0.041"},
             {"channel": "email", "spend": "0", "purchases": "88", "ctr": None}],
}


def test_numbers_in_skips_dates_years_quarters_and_small_bare_integers():
    text = "In Q3 2026 (2026-08-01 to 2026-08-31) the top 3 channels spent $13,689.34, 11,020.1 and 0; CTR was 3.45% on 412 orders."
    found = [(raw, value) for raw, value, _ in numbers_in(text)]
    assert found == [("$13,689.34", Decimal("13689.34")), ("11,020.1", Decimal("11020.1")), ("3.45%", Decimal("3.45")),
                     ("412", Decimal("412"))]


def test_numbers_in_handles_k_and_m_suffixes():
    found = {raw: value for raw, value, _ in numbers_in("about 13.7k spend and 1.2M impressions")}
    assert found == {"13.7k": Decimal("13700.0"), "1.2M": Decimal("1200000.0")}


def test_derivable_includes_cells_percentages_additive_totals_deltas_and_shares():
    values = derivable([QUERY])
    assert Decimal("13689.34") in values and Decimal("412") in values
    assert Decimal("3.45") in values  # ctr as a percentage
    assert Decimal("24709.44") in values  # spend total
    assert Decimal("1110") in values  # purchases total
    assert any(abs(v - Decimal("-19.5")) < Decimal("0.05") for v in values)  # google vs meta spend, -19.5 %
    assert not any(abs(v - Decimal("33.23")) < Decimal("0.005") for v in values)  # CPA must be queried from Cube
    assert any(abs(v - Decimal("55.4")) < Decimal("0.05") for v in values)  # meta share of spend
    assert Decimal(3) in values  # row count


@pytest.mark.parametrize("text", [
    "Meta spent $13,689.34 on 412 purchases; Google $11,020.10 on 610.",
    "Total spend was $24,709.44 across 1,110 purchases.",
    "Meta's CTR was 3.45% versus 4.1% for Google.",
    "Google spent 19.5% less than Meta ($2,669.24 less).",
    "Spend was about $13.7k on Meta.",
    "Rounded: Meta spent $13,689 and Google $11,020.",
])
def test_backed_numbers_pass(text):
    assert unverified_numbers(text, [QUERY]) == []


@pytest.mark.parametrize("text, bad", [
    ("Meta spent $15,000.00 in August.", ["$15,000.00"]),
    ("CTR was 5.2% on Meta.", ["5.2%"]),
    ("Meta had 4,120 purchases.", ["4,120"]),
    ("Revenue was $99,999 and spend $13,689.34.", ["$99,999"]),
])
def test_invented_numbers_are_caught(text, bad):
    assert unverified_numbers(text, [QUERY]) == bad


def test_failed_queries_do_not_back_anything():
    assert unverified_numbers("spend was 13689.34", [{**QUERY, "ok": False}]) == ["13689.34"]


def test_k_and_m_suffixes_are_matched_at_their_own_precision():
    assert unverified_numbers("Meta spent about $14k; Google $11k.", [QUERY]) == []
    assert unverified_numbers("Meta spent about $13.7k.", [QUERY]) == []
    assert unverified_numbers("Meta spent about $16k.", [QUERY]) == ["$16k"]
    assert numbers_in("1.2 million impressions")[0][1] == Decimal("1200000.0")


def test_amounts_that_look_like_years_are_still_checked():
    assert unverified_numbers("Spend was $2,026 on Meta.", [QUERY]) == ["$2,026"]
    assert unverified_numbers("In 2026 Meta spent $13,689.34.", [QUERY]) == []  # a bare year is exempt


def test_underscore_emphasis_does_not_hide_a_number():
    assert unverified_numbers("Meta spent __$15,000__.", [QUERY]) == ["$15,000"]


def test_percentage_points_and_group_totals_are_derivable():
    assert unverified_numbers("CTR rose 0.65 points from 3.45% to 4.1%.", [QUERY]) == []
    compare = {
        "ok": True, "row_count": 4,
        "columns": [{"key": "channel", "type": "string"}, {"key": "spend", "type": "number", "aggregation": "sum"}, {"key": "date_range", "type": "string"}],
        "rows": [{"channel": "meta", "spend": "100", "date_range": "July"}, {"channel": "google", "spend": "50", "date_range": "July"},
                 {"channel": "meta", "spend": "80", "date_range": "August"}, {"channel": "google", "spend": "70", "date_range": "August"}],
    }
    assert unverified_numbers("Total spend was 150 in July and 150 in August; Meta fell 20% while Google rose 40%.", [compare]) == []
    assert unverified_numbers("Total spend fell 17.5% overall.", [compare]) == ["17.5%"]  # no pair, total or share gives this


def test_long_columns_only_compare_neighbours_and_same_group_rows():
    series = {"ok": True, "row_count": 40,
              "columns": [{"key": "date_day", "type": "time"}, {"key": "spend", "type": "number"}],
              "rows": [{"date_day": f"2026-08-{i:02d}", "spend": str(1000 + 7 * i)} for i in range(1, 41)]}
    assert unverified_numbers("Spend rose from 1,007 to 1,014 day over day.", [series]) == []
    # 1,280 - 1,007 = 273 is a difference between two far-apart days: not derivable in a long column
    assert unverified_numbers("The spread was 273 between the first and the fortieth day.", [series]) == ["273"]


def test_non_finite_cells_are_ignored():
    q = {**QUERY, "rows": [{"channel": "x", "spend": "NaN", "purchases": "Infinity", "ctr": None}]}
    assert unverified_numbers("Spend was 100.", [q]) == ["100"]
