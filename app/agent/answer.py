"""Render the checked narrative alongside tables drawn directly from Cube rows."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from app.models.catalog import Catalog, Member
from app.models.state import AgentState
from app.agent.verify import relevant_queries

MAX_TABLE_ROWS = 20


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
    if precision:
        return f"{value.quantize(Decimal(1).scaleb(-int(precision)), rounding=ROUND_HALF_UP):,.{int(precision)}f}"
    return _trim(value, 0 if value == value.to_integral_value() else 2)


def _trim(value: Decimal, precision: int) -> str:
    text = f"{value.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP):,.{precision}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _cell(value: Any, column: dict[str, Any], catalog: Catalog) -> str:
    if value is None:
        return "—"
    if column.get("type") != "number":
        return str(value)[:10] if column.get("type") == "time" else str(value)
    try:
        return format_value(Decimal(str(value)), catalog.member(column["name"]))
    except InvalidOperation:
        return str(value)


def table(query: dict[str, Any], catalog: Catalog) -> str:
    """A plain-text table of one query's rows (capped), columns titled from the annotation."""
    columns = query.get("columns") or []
    rows = query.get("rows") or []
    if not columns or not rows:
        return ""
    cells = [[_cell(r.get(c["key"]), c, catalog) for c in columns] for r in rows[:MAX_TABLE_ROWS]]
    widths = [max(len(c["title"]), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    align = [c.get("type") == "number" for c in columns]

    def line(values: list[str]) -> str:
        return "  ".join(v.rjust(w) if a else v.ljust(w) for v, w, a in zip(values, widths, align)).rstrip()

    out = [line([c["title"] for c in columns]), line(["-" * w for w in widths])]
    out += [line(row) for row in cells]
    if len(rows) > MAX_TABLE_ROWS:
        out.append(f"… {len(rows) - MAX_TABLE_ROWS} more row(s) — see “Under the hood”")
    if query.get("complete") is False:
        out.append(query.get("note") or "These results are limited and may be incomplete.")
    return "\n".join(out)


# --------------------------------------------------------------------------- entry point

def render(state: AgentState, catalog: Catalog) -> tuple[str, str]:
    outcome = state.get("outcome", "error")
    final = state.get("final") or {}
    text = _plain(final.get("text") or "")
    shown = _tables_to_show(state.get("queries", []))
    if outcome == "answer":
        body = text or "Here is what the semantic layer returned."
        for q in shown:
            heading = f"{q.get('query_id', 'Result')} — {_query_scope(q)}\n" if len(shown) > 1 else ""
            body += f"\n\n{heading}{table(q, catalog) or 'No matching rows.'}"
    elif outcome == "no_data":
        body = text or "No rows matched this question in the semantic layer."
        coverage = _coverage(catalog)
        if coverage and coverage not in body:
            body += f"\n(Data covers {coverage}.)"
    elif outcome in ("clarify", "unsupported"):
        body = text or ("Could you give me one more detail?" if outcome == "clarify"
                        else "I can't answer that from the semantic layer.")
    else:
        body = _error(state)
        if shown:  # the run failed after data came back: show the data, not the narrative
            body += "\nThe rows the semantic layer returned before that:\n\n" + table(shown[-1], catalog)
    return body, _footer(state, catalog)


def _tables_to_show(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each distinct successful query, including different periods/filters.

    Only identical requests are deduplicated; the latest result wins.
    """
    return relevant_queries(queries)


def _query_scope(query: dict[str, Any]) -> str:
    periods = query.get("periods") or []
    span = " vs ".join(f"{p['from']} to {p['to']}" for p in periods)
    filters = query.get("query", {}).get("filters", [])
    labels = [
        f"{f['member'].split('.', 1)[-1]} {f.get('operator', '')} {', '.join(map(str, f.get('values', [])))}"
        if "member" in f else str(f)
        for f in filters
    ]
    return f"Period: {span or 'all available dates'} (UTC, inclusive) · Filters: {', '.join(labels) or 'none'}"


_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_HEADING = re.compile(r"^#{1,6}\s*", re.M)


def _plain(text: str) -> str:
    """The narrative is shown as plain text: drop the markdown emphasis and heading markers models like to add."""
    return _HEADING.sub("", _EMPHASIS.sub(r"\2", text)).strip()


def _coverage(catalog: Catalog) -> str:
    spans = [v.coverage for v in catalog.views.values() if v.coverage]
    if not spans:
        return ""
    return f"{min(s[0] for s in spans)} to {max(s[1] for s in spans)}"


def _error(state: AgentState) -> str:
    kind, detail = state.get("error_kind"), state.get("error") or ""
    if kind == "free_guard":
        return f"Refused: the router did not return a demonstrably free model ({detail})."
    if kind == "llm":
        return f"The model call failed ({detail}). No answer was produced; please retry."
    if kind == "cube":
        return f"The semantic layer rejected the request ({detail}). No answer was produced."
    if kind == "budget":
        return f"I could not finish this question ({detail}). Try a simpler question."
    if kind == "validation":
        return f"I rejected the model's answer because it was not backed by the data ({detail})."
    return f"Something went wrong ({detail or 'unknown error'})."


def _footer(state: AgentState, catalog: Catalog) -> str:
    parts: list[str] = []
    shown = _tables_to_show(state.get("queries", []))
    definitions: dict[str, str] = {}
    for query in shown:
        prefix = f"{query.get('query_id', 'Result')}: " if len(shown) > 1 else ""
        parts.append(prefix + _query_scope(query))
        if state.get("outcome") in ("answer", "no_data"):
            for col in query.get("columns", []):
                member = catalog.member(col.get("name", ""))
                if member and member.kind == "measure" and member.description:
                    definitions[member.name] = f"{member.short} = {member.description}"
    parts.extend(definitions.values())
    models = list(dict.fromkeys(state.get("llm_models", [])))
    if models:
        parts.append("model: " + ", ".join(models))
        parts.append(f"cost: ${state.get('llm_cost') or '0'}")
    parts.append(f"model calls: {state.get('llm_calls', 0)}")
    parts.append(f"cube calls: {state.get('cube_calls', 0)}")
    parts.append(f"tool calls: {len(state.get('steps', []))}")
    return " · ".join(parts)
