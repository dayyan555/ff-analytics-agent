"""Answer templates. Deterministic: every number comes from ``state["result"]``.

Number formatting follows the semantic layer's own ``format``/``currency``
metadata (from the ``/load`` annotation, falling back to ``/meta``), so the
UI, the footer and the README all agree on what a metric looks like.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.models.catalog import RATIOS, Catalog, Member
from app.models.state import AgentState


# --------------------------------------------------------------------------- formatting

def format_value(value: Decimal | None, member: Member | None, fmt: str | None = None, currency: str | None = None) -> str:
    if value is None:
        return "undefined"
    fmt = fmt or (member.format if member else None) or "number"
    currency = currency or (member.currency if member else None)
    base, _, precision = fmt.partition("_")
    if base == "currency":
        p = int(precision) if precision else 2
        sign, magnitude = ("-", -value) if value < 0 else ("", value)
        text = f"{magnitude.quantize(Decimal(1).scaleb(-p), rounding=ROUND_HALF_UP):,.{p}f}"
        return f"{sign}${text}" if (currency or "USD").upper() == "USD" else f"{sign}{text} {currency}"
    if base == "percent":
        p = int(precision) if precision else 1
        return f"{_trim(value * 100, p)}%"
    p = int(precision) if precision else (0 if value == value.to_integral_value() else 2)
    return _trim(value, p)


def _trim(value: Decimal, precision: int) -> str:
    text = f"{value.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP):,.{precision}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _pct(value: Decimal | None, from_zero: bool) -> str:
    if from_zero:
        return "n/a (from 0)"
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{_trim(value * 100, 1)}%"


class _Fmt:
    """Formatter bound to one query's annotation (with catalog fallback)."""

    def __init__(self, catalog: Catalog, annotation: dict[str, Any] | None):
        self.catalog = catalog
        self.ann = (annotation or {}).get("measures", {})

    def value(self, short: str, value: Decimal | None) -> str:
        entry = self.ann.get(self.catalog.member(short), {})
        return format_value(value, self.catalog.measures.get(short), entry.get("format"), entry.get("currency"))

    def title(self, short: str) -> str:
        entry = self.ann.get(self.catalog.member(short), {})
        member = self.catalog.measures.get(short) or self.catalog.dimensions.get(short)
        return entry.get("shortTitle") or (member.short_title if member else short)

    def description(self, short: str) -> str:
        entry = self.ann.get(self.catalog.member(short), {})
        member = self.catalog.measures.get(short)
        return entry.get("description") or (member.description if member else "")


# --------------------------------------------------------------------------- entry point

def render(state: AgentState, catalog: Catalog) -> tuple[str, str]:
    outcome = state.get("outcome", "error")
    if outcome == "answer":
        body = _render_result(state, catalog)
    elif outcome == "clarify":
        body = _clarify(state, catalog)
    elif outcome == "unsupported":
        body = _unsupported(state, catalog)
    elif outcome == "no_data":
        body = _no_data(state, catalog)
    else:
        body = _error(state)
    return body, _footer(state, catalog)


# --------------------------------------------------------------------------- outcome messages

_SAFE_NAME = re.compile(r"[A-Za-z0-9_ .\-]{1,40}")


def _safe_name(text: Any) -> str:
    """Model-supplied names are echoed only if they look like names (bounded, plain characters)."""
    text = str(text)
    return text if _SAFE_NAME.fullmatch(text) else "an unrecognised name"


def _coverage(catalog: Catalog) -> str:
    first, last = catalog.coverage or (None, None)
    return f"{first} to {last}"


def _clarify(state: AgentState, catalog: Catalog) -> str:
    reasons = "; ".join(state.get("notes", []))
    metrics = ", ".join(catalog.measures)
    text = (
        "I need one more detail to answer this. Which period do you mean (for example 'last month', "
        f"'2026-07', or '2026-07-01..2026-07-31'; data covers {_coverage(catalog)}), and which metric ({metrics})?"
    )
    return f"{text}\nReason: {reasons}." if reasons else text


def _unsupported(state: AgentState, catalog: Catalog) -> str:
    rejected = state.get("rejected") or {}
    lines = ["I can't answer that from the semantic layer."]
    lines += [f"Unknown metric '{_safe_name(m)}'." for m in rejected.get("measures", [])[:5]]
    if rejected.get("dimension"):
        lines.append(f"Unknown dimension '{_safe_name(rejected['dimension'])}'.")
    for fv in rejected.get("filter_values", [])[:5]:
        pool = catalog.values.get(fv["dimension"], [])
        lines.append(f"No {fv['dimension'].replace('_', ' ')} named '{_safe_name(fv['value'])}'. Known: {', '.join(pool)}.")
    if rejected.get("feature"):
        lines.append(f"Not supported yet: {rejected['feature']}.")
    lines.append(f"Available metrics: {', '.join(catalog.measures)}. Dimensions: {', '.join(catalog.dimensions)}.")
    return "\n".join(lines)


def _no_data(state: AgentState, catalog: Catalog) -> str:
    period = state.get("period") or {}
    query = state.get("cube_query")
    span = f"{period.get('start')} to {period.get('end')}" if period else "that period"
    if query is None:  # the requested period lies entirely outside the data
        return f"Data covers {_coverage(catalog)}; there is nothing for {span}."
    values = [v for f in query.get("filters", []) for v in f["values"]]
    subject = f"for {', '.join(values)} " if values else ""
    return (f"No rows {subject}between {period.get('start')} and {period.get('end')} "
            f"(the warehouse covers {_coverage(catalog)}).")


def _error(state: AgentState) -> str:
    kind, detail = state.get("error_kind"), state.get("error") or ""
    if kind == "cube":
        return f"The semantic layer returned an error ({detail}). No answer was produced."
    if kind == "free_guard":
        return f"Refused: the router did not serve a demonstrably free model ({detail}). No answer was produced."
    if kind == "validation":
        return f"Unexpected response shape from the semantic layer ({detail}). No answer was produced."
    model = state.get("llm_model") or "unknown"
    return f"The model did not return a usable plan (routed model: {model}; {detail}). Please retry."


# --------------------------------------------------------------------------- results

def _render_result(state: AgentState, catalog: Catalog) -> str:
    result = state["result"]
    fmt = _Fmt(catalog, state.get("annotation"))
    shape = result["shape"]
    text = {"breakdown": _breakdown, "ranking": _ranking, "compare": _compare}[shape](state, result, fmt)
    caveats = [*(state.get("notes") or []), *(result.get("caveats") or [])]
    return text + ("\n\nCaveats: " + "; ".join(caveats) if caveats else "")


def _label(period: dict[str, Any] | None) -> str:
    return period.get("label", "") if period else ""


def _breakdown(state: AgentState, result: dict[str, Any], fmt: _Fmt) -> str:
    dim, measures = result["dimension"], result["requested"]
    period = _label(state.get("period"))
    titles = ", ".join(fmt.title(m) for m in measures)
    lines = [f"{titles} by {fmt.title(dim).lower()}, {period}:" if dim else f"{titles}, {period} (total):"]
    for row in result["rows"]:
        cells = [f"{fmt.title(m)} {fmt.value(m, row['values'].get(m))}" for m in measures]
        if "spend" in measures and (row["values"].get("spend") or 0) == 0:
            cells[measures.index("spend")] += " (no paid media)"
        lines.append(f"- {row['key']}: " + ", ".join(cells) if dim else "- " + ", ".join(cells))
    totals = result.get("totals") or {}
    if dim and totals and len(result["rows"]) > 1:
        cells = [f"{fmt.title(m)} {fmt.value(m, totals[m])}" for m in measures if m in totals]
        if cells:
            lines.append("- Total: " + ", ".join(cells))
    return "\n".join(lines)


def _ranking(state: AgentState, result: dict[str, Any], fmt: _Fmt) -> str:
    rank = result["ranking"]
    by, winner = rank["by"], rank["winner"]
    period = _label(state.get("period"))
    superlative = "lowest" if rank["direction"] == "asc" else "highest"
    if winner is None:
        return f"No row has a defined {fmt.title(by)} for {period}."
    detail = ""
    if by in RATIOS:
        num, den, factor = RATIOS[by]
        detail = (f" ({fmt.title(num)} {fmt.value(num, winner['values'].get(num))} / {fmt.title(den)} "
                  f"{fmt.value(den, winner['values'].get(den))}" + (f" × {factor:,}" if factor != 1 else "") + ")")
    lines = [f"{winner['key']} has the {superlative} {fmt.title(by)} for {period}: {fmt.value(by, winner['values'][by])}{detail}."]
    if rank["tied"]:
        lines.append("Tied at the top: " + ", ".join(str(k) for k in rank["tied"]) + ".")
    lines.append(f"Ranking by {fmt.title(by)}:")
    for i, row in enumerate(rank["top"], 1):
        others = [f"{fmt.title(m)} {fmt.value(m, row['values'].get(m))}" for m in result["requested"] if m != by]
        lines.append(f"{i}. {row['key']}: {fmt.value(by, row['values'][by])}" + (f" ({', '.join(others)})" if others else ""))
    if result["excluded"]:
        lines.append("Excluded: " + "; ".join(f"{e['key']} ({e['reason']})" for e in result["excluded"]) + ".")
    return "\n".join(lines)


def _compare(state: AgentState, result: dict[str, Any], fmt: _Fmt) -> str:
    deltas = result["deltas"]
    a_label, b_label = _label(state.get("period")), _label(state.get("compare_period"))
    dim, measures = result["dimension"], result["requested"]
    lines: list[str] = []
    if deltas["empty_side"]:
        lines.append(f"No data in {a_label if deltas['empty_side'] == 'a' else b_label}.")
    tot = deltas["totals"]
    parts = []
    for m in measures if not deltas["empty_side"] else []:  # totals are meaningless with an empty side
        if m in tot["a"] and m in tot["b"]:
            d = tot["delta"][m]
            parts.append(f"{fmt.title(m)} {fmt.value(m, tot['a'][m])} → {fmt.value(m, tot['b'][m])} "
                         f"({'+' if (d['delta'] or 0) >= 0 else ''}{fmt.value(m, d['delta'])}, {_pct(d['pct'], d['from_zero'])})")
    if parts:
        lines.append(f"Between {a_label} and {b_label}: " + "; ".join(parts) + ".")
    lines.append(f"By {fmt.title(dim).lower() if dim else 'total'} ({a_label} → {b_label}):")
    for row in deltas["rows"]:
        cells = []
        for m in measures:
            a = fmt.value(m, row["a"].get(m)) if row["a"] else "no data"
            b = fmt.value(m, row["b"].get(m)) if row["b"] else "no data"
            d = row["delta"][m]
            cells.append(f"{fmt.title(m)} {a} → {b} ({_pct(d['pct'], d['from_zero'])})")
        flag = "" if row["status"] == "both" else f" [{row['status']}]"
        lines.append(f"- {row['key'] if dim else 'total'}{flag}: " + "; ".join(cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- footer

def _footer(state: AgentState, catalog: Catalog) -> str:
    parts: list[str] = []
    period, compare = state.get("period"), state.get("compare_period")
    if period:
        span = f"{period['start']} to {period['end']}"
        if compare:
            span += f" vs {compare['start']} to {compare['end']}"
        parts.append(f"Period: {span} (UTC, inclusive)")
    query = state.get("cube_query") or {}
    filters = ", ".join(f"{f['member'].split('.', 1)[1]} = {v}" for f in query.get("filters", []) for v in f["values"])
    parts.append(f"Filters: {filters or 'none'}")
    fmt = _Fmt(catalog, state.get("annotation"))
    plan = state.get("plan") or {}
    if state.get("outcome") in ("answer", "no_data"):
        for m in plan.get("measures", []):
            if m in catalog.measures:
                parts.append(f"{m} = {fmt.description(m)}")
    if state.get("llm_model"):
        parts.append(f"model: {state['llm_model']}")
        parts.append(f"cost: ${state.get('llm_cost') or '0'}")
    parts.append(f"cube calls: {state.get('cube_calls', 0)}")
    return " · ".join(parts)
