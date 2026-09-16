"""A numerical consistency check, not proof that the narrative is correct.

Accept cells, metadata-approved percentage formatting, sums/shares of complete
additive results, and differences within a metric. Ratios must come from Cube.
This does not bind a figure to the entity, metric or direction named in prose.
"""

from __future__ import annotations

import re
import json
from decimal import Decimal, InvalidOperation
from typing import Any

MAX_CELLS = 400  # beyond this the pairwise derivations are skipped (rows are capped at 50 anyway)
SMALL_INT_EXEMPT = 31  # bare integers up to this are ranks, counts of rows, day numbers: not checked
ALL_PAIRS_UP_TO = 12  # in a column with more values, differences are only derived between neighbours and within a group

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?\b")
_QUARTER = re.compile(r"\bQ[1-4]\b", re.I)
_YEAR = re.compile(
    r"\b(?:Q[1-4]|January|February|March|April|May|June|July|August|September|October|November|December|in|during)"
    r"\s+(?:19|20)\d{2}\b(?!\s+(?:purchases|orders|clicks|impressions))", re.I,
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])([-+]?\$?\d[\d,]*(?:\.\d+)?)\s*(%|thousand|million|billion|k|m|K|M|x|×)?(?![A-Za-z0-9])")
_SCALE = {"k": 3, "thousand": 3, "m": 6, "million": 6, "billion": 9}


def numbers_in(text: str) -> list[tuple[str, Decimal, int]]:
    """(as written, value, decimals) for every number in the text that is worth checking.

    ``decimals`` is the precision the number was written with; it goes negative for
    "$14k" (-3: to the nearest thousand) or "1.2M" (-5).
    """
    cleaned = _QUARTER.sub(" ", _YEAR.sub(" ", _DATE.sub(" ", text)))
    found: list[tuple[str, Decimal, int]] = []
    for m in _NUMBER.finditer(cleaned):
        raw, suffix = m.group(1), (m.group(2) or "")
        digits = raw.replace("$", "").replace(",", "").lstrip("+")
        try:
            value = Decimal(digits)
        except InvalidOperation:
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = _SCALE.get(suffix.lower())
        if scale:
            value, decimals = value.scaleb(scale), decimals - scale
        bare = "$" not in raw and suffix == "" and decimals == 0
        if bare and (abs(value) <= SMALL_INT_EXEMPT or ("," not in raw and 1900 <= value <= 2099)):
            continue
        found.append((m.group(0).strip(), value, decimals))
    return found


def _cells(query: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """One result's finite numeric cells and their semantic metadata."""
    rows: list[dict[str, Any]] = []
    columns = {c["key"]: c for c in query.get("columns", []) if c.get("type") == "number"}
    for r in query.get("rows", []):
        cells = {k: v for k, v in r.items() if k not in columns}
        for key in columns:
            try:
                value = Decimal(str(r.get(key)))
            except InvalidOperation:
                continue
            if value.is_finite():
                cells[key] = value
        rows.append(cells)
    return rows, columns


def _pairs(values: list[Decimal], groups: list[Any]) -> set[tuple[Decimal, Decimal]]:
    """Which pairs of a column's values may be compared: every pair in a small column; otherwise
    neighbours in result order and values that share a group label (the same channel across periods)."""
    n = len(values)
    if n <= ALL_PAIRS_UP_TO:
        return {(values[i], values[j]) for i in range(n) for j in range(i + 1, n)}
    pairs = {(values[i], values[i + 1]) for i in range(n - 1)}
    by_group: dict[Any, list[Decimal]] = {}
    for g, v in zip(groups, values):
        by_group.setdefault(g, []).append(v)
    for members in by_group.values():
        pairs |= {(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))}
    return pairs


def _deltas(a: Decimal, b: Decimal, as_points: bool) -> set[Decimal]:
    out = {b - a, a - b}
    if as_points:
        out |= {(b - a) * 100, (a - b) * 100}  # percentage-point changes of a ratio
    if a:
        change = (b - a) / a * 100
        out |= {change, abs(change)}  # "fell 19%" states an unsigned magnitude
    if b:
        change = (a - b) / b * 100
        out |= {change, abs(change)}
    return out


def derivable(queries: list[dict[str, Any]]) -> set[Decimal]:
    values: set[Decimal] = set()
    # Full member names keep metrics from different views separate. Only deltas
    # cross queries; totals never add duplicate/retried or overlapping queries.
    comparisons: dict[tuple[str, bool, str], list[tuple[Decimal, Any]]] = {}
    for query in queries:
        if not query.get("ok") or query.get("has_data") is False:
            continue
        rows, columns = _cells(query)
        values.add(Decimal(len(rows)))
        for key, column in columns.items():
            percent = str(column.get("format", "")).startswith("percent")
            metric = (column.get("name", key), percent, "cell")
            col_rows = [r for r in rows if key in r]
            by_period: dict[Any, list[Decimal]] = {}
            for r in col_rows:
                value = r[key]
                values.add(value)
                if percent:
                    values.add(value * 100)
                group = tuple(sorted((k, str(v)) for k, v in r.items() if k not in columns and k != "date_range"))
                comparisons.setdefault(metric, []).append((value, group))
                by_period.setdefault(r.get("date_range"), []).append(value)
            if column.get("aggregation") == "sum" and query.get("complete", True):
                for cells in by_period.values():
                    total = sum(cells, Decimal(0))
                    values.add(total)
                    if total:
                        values.update(v / total * 100 for v in cells)
                    if len(cells) > 1:
                        comparisons.setdefault((*metric[:2], "total"), []).append((total, ("total",)))
    if sum(len(c) for c in comparisons.values()) <= MAX_CELLS:
        for (_, percent, _), cells in comparisons.items():
            for a, b in _pairs([c[0] for c in cells], [c[1] for c in cells]):
                values |= _deltas(a, b, percent)
    return values


def relevant_queries(queries: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """The final one or two distinct successful queries that can support the answer."""
    unique: dict[str, dict[str, Any]] = {}
    for query in queries:
        if query.get("ok"):
            signature = json.dumps([query.get("query"), query.get("periods"), query.get("columns")], sort_keys=True)
            unique.pop(signature, None)
            unique[signature] = query
    return list(unique.values())[-limit:]


def _matches(written: Decimal, decimals: int, candidates: set[Decimal]) -> bool:
    tolerance = Decimal(1).scaleb(-decimals) / 2  # half a unit of the last written digit
    return any(abs(c - written) <= tolerance for c in candidates)


def unverified_numbers(text: str, queries: list[dict[str, Any]]) -> list[str]:
    """The numbers in ``text`` that cannot be derived from the query results."""
    candidates = derivable(queries)
    return [raw for raw, value, decimals in numbers_in(text) if not _matches(value, decimals, candidates)]
