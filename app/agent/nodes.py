"""The five graph nodes and their routers.

Responsibilities are deliberately separate:
- ``interpret``        the only LLM call: question -> Plan (JSON form)
- ``build_query``      deterministic: validates the Plan against the catalog, resolves dates,
                       builds the Cube query, and owns every plan-level exit
- ``query_cube``       the only I/O: dry-run + load through the Cube tools
- ``validate_results`` deterministic checks and derived numbers
- ``answer``           deterministic templates; never computes, never invents
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from app.agent import answer as templates
from app.agent.dates import Period, PeriodError, overlap, periods_overlap, resolve
from app.agent.prompt import build_messages, repair_message
from app.models.catalog import COST_LIKE, RATIOS, safe_name
from app.models.plan import Plan, parse_plan
from app.models.state import AgentState, Deps
from app.tools.cube import CubeError
from app.tools.llm import FreeInferenceViolation, LLMError

CUBE_LIMIT = 50
_SUPERLATIVE = re.compile(r"\b(most|highest|top|best|least|lowest|bottom|worst|strongest|weakest)\b", re.I)
DEFAULT_TOP_N = 10
RECOMPUTE_TOLERANCE = Decimal("0.001")  # relative


# --------------------------------------------------------------------------- interpret

MAX_PLAN_ATTEMPTS = 3  # planning attempts per question (each may retry once on a 429)


def interpret(state: AgentState, runtime: Runtime[Deps], config: RunnableConfig) -> dict[str, Any]:
    """The only LLM call: question -> Plan.

    An empty or off-task reply (no JSON object at all — e.g. a content-safety model
    answering "User Safety: safe") is re-rolled with a fresh request; a malformed JSON
    reply gets one repair turn with the error fed back. At most MAX_PLAN_ATTEMPTS attempts.
    """
    deps = runtime.context
    before = deps.llm.calls
    base_messages = build_messages(state["question"], deps.catalog, deps.as_of)
    messages = base_messages
    out: dict[str, Any] = {"plan_raw": None, "plan": None, "llm_model": None, "llm_cost": None}
    repaired = False
    try:
        for _attempt in range(MAX_PLAN_ATTEMPTS):
            reply = deps.llm.plan(messages, config)
            out["plan_raw"], out["llm_model"] = reply.text, reply.model_name
            out["llm_cost"] = str(reply.cost) if reply.cost is not None else None
            try:
                out["plan"] = parse_plan(reply.text).model_dump()
                return _with_calls(out, deps, before)
            except OutputParserException as exc:
                if "{" in reply.text and not repaired:  # looks like JSON: one repair turn
                    messages = [*base_messages, AIMessage(content=reply.text), repair_message(str(exc))]
                    repaired = True
                else:  # empty / off-task: re-roll the router with a clean request
                    messages = base_messages
        # never echo the model's text: it may contain anything
        out.update(outcome="error", error_kind="llm",
                   error=f"the reply was not a valid plan ({MAX_PLAN_ATTEMPTS} attempts)")
    except LLMError as exc:
        out.update(outcome="error", error_kind="llm", error=f"{exc.kind}: {exc.detail}")
    except FreeInferenceViolation as exc:
        out.update(outcome="error", error_kind="free_guard", error=str(exc))
    return _with_calls(out, deps, before)


def _with_calls(out: dict[str, Any], deps: Deps, before: int) -> dict[str, Any]:
    out["llm_calls"] = deps.llm.calls - before  # per-question delta: the LLM instance outlives requests
    return out


def route_after_interpret(state: AgentState) -> str:
    return "answer" if state.get("outcome") == "error" else "build_query"


# --------------------------------------------------------------------------- build_query

def build_query(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    cat = runtime.context.catalog
    plan = Plan.model_validate(state["plan"])

    if plan.intent == "clarify":
        return {"outcome": "clarify"}

    # 1. every name the model used must exist in the semantic layer (case, spaces and common
    #    synonyms are normalised first: "ROAS", "cost per purchase" are not user errors)
    resolved = {m: cat.match_measure(m) for m in plan.measures}
    unknown_measures = [m for m, r in resolved.items() if r is None]
    known_measures = list(dict.fromkeys(r for r in resolved.values() if r))
    ignored: list[str] = []
    if unknown_measures and known_measures and plan.intent != "unsupported":
        # answer what the semantic layer has, and say what it does not have
        ignored, unknown_measures = unknown_measures, []
    order_by = cat.match_measure(plan.order_by) if plan.order_by else None
    plan = plan.model_copy(update={"measures": known_measures, "order_by": order_by if order_by in known_measures else None})
    dimension_name = cat.match_dimension(plan.dimension) if plan.dimension else None
    bad_dimension = plan.dimension if plan.dimension and dimension_name is None else None
    plan = plan.model_copy(update={"dimension": dimension_name})
    filters: list[tuple[str, str]] = []
    bad_values: list[dict[str, str]] = []
    for f in plan.filters:
        dim = cat.match_dimension(f.dimension)
        if dim is None:
            bad_dimension = bad_dimension or f.dimension
            continue
        canonical = cat.match_value(dim, f.value)
        if canonical:
            filters.append((dim, canonical))
        else:
            bad_values.append({"dimension": dim, "value": f.value})
    if plan.intent == "unsupported" or unknown_measures or bad_dimension or bad_values:
        nothing_named = not (unknown_measures or bad_dimension or bad_values)
        return {"outcome": "unsupported", "rejected": {
            "measures": unknown_measures, "dimension": bad_dimension, "filter_values": bad_values,
            "feature": "the question asks for something outside the metrics and dimensions listed" if nothing_named else None}}
    if plan.granularity:
        return {"outcome": "unsupported", "rejected": {
            "measures": [], "dimension": None, "filter_values": [], "feature": "time-series breakdown by granularity"}}

    # 2. periods are resolved by code, never by the model
    if plan.period is None:
        return {"outcome": "clarify", "notes": ["no period was given"], "plan": plan.model_dump()}
    if plan.intent == "compare" and plan.compare_period is None:
        return {"outcome": "clarify", "notes": ["compare with which period?"], "plan": plan.model_dump()}
    assert cat.coverage is not None
    try:
        period = resolve(plan.period, runtime.context.as_of, cat.coverage)
    except PeriodError as exc:
        return {"outcome": "clarify", "notes": [str(exc)]}
    compare: Period | None = None
    if plan.intent == "compare":
        try:
            compare = resolve(plan.compare_period, runtime.context.as_of, cat.coverage, base=period)
        except PeriodError as exc:
            return {"outcome": "clarify", "notes": [str(exc)]}
        if periods_overlap(period, compare):
            return {"outcome": "clarify", "notes": ["the two periods overlap"], "period": period.as_dict(),
                    "compare_period": compare.as_dict()}
        if compare.end < period.start:  # always report earlier -> later
            period, compare = compare, period
    out: dict[str, Any] = {"period": period.as_dict(), "compare_period": compare.as_dict() if compare else None}
    coverages = [overlap(period, cat.coverage)] + ([overlap(compare, cat.coverage)] if compare else [])
    if all(c == "none" for c in coverages):
        return {**out, "outcome": "no_data"}
    notes = [p.coverage_note for p in (period, compare) if p and p.coverage_note]
    notes += [f"ignored unknown metric '{safe_name(m)}' (not in the semantic layer)" for m in ignored[:5]]

    # 3. the query itself
    measures = list(plan.measures)
    for m in plan.measures:  # ratios always travel with their numerator and denominator
        for part in RATIOS.get(m, ())[:2]:
            if part not in measures:
                measures.append(part)
    # group by the filtered dimension when no grouping was asked for: an empty match then
    # returns zero rows instead of one row of zeros
    dimension = plan.dimension or (filters[0][0] if filters else None)
    order_by = plan.order_by or measures[0]
    direction = plan.direction or ("asc" if order_by in COST_LIKE else "desc")
    if plan.order_by is None and dimension and _SUPERLATIVE.search(state["question"]):
        # the question asks for a ranking but the model left order_by empty: rank by the first metric
        plan = plan.model_copy(update={"order_by": order_by})
        notes.append(f"ranked by {order_by} (inferred from the question)")

    mp = cat.member
    time_dimension: dict[str, Any] = {"dimension": cat.time_dimension}
    if compare:
        time_dimension["compareDateRange"] = [[period.start.isoformat(), period.end.isoformat()],
                                             [compare.start.isoformat(), compare.end.isoformat()]]
    else:
        time_dimension["dateRange"] = [period.start.isoformat(), period.end.isoformat()]
    query: dict[str, Any] = {"measures": [mp(m) for m in measures]}
    if dimension:
        query["dimensions"] = [mp(dimension)]
    query["timeDimensions"] = [time_dimension]
    if filters:  # several values of one dimension are one `equals` filter (OR), not several (AND)
        by_dim: dict[str, list[str]] = {}
        for d, v in filters:
            if v not in by_dim.setdefault(d, []):
                by_dim[d].append(v)
        query["filters"] = [{"member": mp(d), "operator": "equals", "values": vals} for d, vals in by_dim.items()]
    query["order"] = {mp(dimension): "asc"} if (compare and dimension) else {mp(order_by): direction}
    query["limit"] = CUBE_LIMIT
    query["timezone"] = "UTC"
    out["plan"] = plan.model_dump()  # the plan actually executed (names normalised, unknown metrics removed)
    return {**out, "cube_query": query, "notes": notes}


def route_after_build(state: AgentState) -> str:
    return "answer" if state.get("outcome") else "query_cube"


# --------------------------------------------------------------------------- query_cube

def query_cube(state: AgentState, runtime: Runtime[Deps], config: RunnableConfig) -> dict[str, Any]:
    cube = runtime.context.cube
    query = state["cube_query"]
    calls = state.get("cube_calls", 0)
    try:
        calls += 1
        normalized = cube.dry_run(query, config).get("normalizedQueries", [])
        calls += 1
        results = cube.load(query, config).get("results", [])
    except CubeError as exc:
        return {"cube_calls": calls, "outcome": "error", "error_kind": "cube", "error": str(exc)}
    return {
        "cube_calls": calls,
        "normalized": normalized,
        "rows": [r.get("data", []) for r in results],
        "annotation": results[0].get("annotation") if results else None,
    }


def route_after_query(state: AgentState) -> str:
    return "answer" if state.get("outcome") else "validate_results"


# --------------------------------------------------------------------------- validate_results

def validate_results(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    cat = runtime.context.catalog
    plan = Plan.model_validate(state["plan"])
    query = state["cube_query"]
    mp = cat.member
    measures = [m.split(".", 1)[1] for m in query["measures"]]
    dimension = query["dimensions"][0].split(".", 1)[1] if query.get("dimensions") else None
    row_sets = state.get("rows", [])

    expected_sets = 2 if plan.intent == "compare" else 1
    if len(row_sets) != expected_sets:
        return {"outcome": "error", "error_kind": "validation",
                "error": f"expected {expected_sets} result set(s), got {len(row_sets)}"}
    if all(len(rows) == 0 for rows in row_sets):
        return {"outcome": "no_data"}

    parsed_sets: list[list[dict[str, Any]]] = []
    for rows in row_sets:
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if dimension and mp(dimension) not in row:
                return {"outcome": "error", "error_kind": "validation", "error": f"missing '{mp(dimension)}' in a result row"}
            values: dict[str, Decimal | None] = {}
            for m in measures:
                key = mp(m)
                if key not in row:
                    return {"outcome": "error", "error_kind": "validation", "error": f"missing '{key}' in a result row"}
                try:
                    values[m] = None if row[key] is None else Decimal(str(row[key]))
                except InvalidOperation:
                    return {"outcome": "error", "error_kind": "validation", "error": f"non-numeric value for '{key}'"}
            parsed.append({"key": row[mp(dimension)] if dimension else None, "values": values})
        parsed_sets.append(parsed)

    additive = [m for m in measures if m not in RATIOS]
    # an ungrouped SUM over an empty set is one row of zeros — treat it as no data
    if dimension is None and all(_single_all_zero_row(rows, additive) for rows in parsed_sets if rows):
        return {"outcome": "no_data"}
    if any(len(rows) >= CUBE_LIMIT for rows in parsed_sets):  # totals would be over a truncated set
        return {"outcome": "error", "error_kind": "validation",
                "error": f"more than {CUBE_LIMIT} groups in the result; totals would be incomplete"}

    caveats: list[str] = []
    for rows in parsed_sets:  # runtime proof that Cube's ratios are what we say they are
        for row in rows:
            for ratio, (num, den, factor) in RATIOS.items():
                v, n, d = row["values"].get(ratio), row["values"].get(num), row["values"].get(den)
                if v is not None and n is not None and d not in (None, 0):
                    expected = n / d * factor
                    if abs(v - expected) > max(abs(expected) * RECOMPUTE_TOLERANCE, Decimal("0.0001")):
                        caveats.append(f"{ratio} for '{row['key']}' differs from {num}/{den} recomputed in code")

    result: dict[str, Any] = {
        "shape": "compare" if plan.intent == "compare" else ("ranking" if (plan.order_by and dimension) else "breakdown"),
        "dimension": dimension, "measures": measures, "requested": list(plan.measures),
        "rows": parsed_sets[0], "totals": _totals(parsed_sets[0], additive),
        "ranking": None, "excluded": [], "deltas": None, "caveats": caveats,
    }

    if result["shape"] == "ranking":
        by = plan.order_by or measures[0]
        direction = query["order"].get(mp(by), "desc")
        ranked = [r for r in parsed_sets[0] if r["values"].get(by) is not None]
        result["excluded"] = [
            {"key": r["key"], "reason": _null_reason(by, r["values"])}
            for r in parsed_sets[0] if r["values"].get(by) is None
        ]
        ranked.sort(key=lambda r: r["values"][by], reverse=(direction == "desc"))
        winner = ranked[0] if ranked else None
        tied = [r["key"] for r in ranked if winner and r["values"][by] == winner["values"][by]]
        result["ranking"] = {"by": by, "direction": direction, "winner": winner,
                             "tied": tied if len(tied) > 1 else [], "top": ranked[: plan.limit or DEFAULT_TOP_N]}
        if not ranked:
            caveats.append(f"{by} is undefined for every row")

    if result["shape"] == "compare":
        a_rows, b_rows = parsed_sets[0], (parsed_sets[1] if len(parsed_sets) > 1 else [])
        a_by, b_by = {r["key"]: r["values"] for r in a_rows}, {r["key"]: r["values"] for r in b_rows}
        keys = list(dict.fromkeys([*a_by, *b_by]))
        rows = []
        for key in keys:
            a, b = a_by.get(key), b_by.get(key)
            status = "both" if a and b else ("new" if b else "ended")
            rows.append({"key": key, "status": status, "a": a, "b": b, "delta": _deltas(a, b, measures)})
        ta, tb = _totals(a_rows, additive), _totals(b_rows, additive)
        for side in (ta, tb):  # ratios over the totals, so the headline can state e.g. the overall CPA change
            for ratio, (num, den, factor) in RATIOS.items():
                if ratio in measures and side.get(den) not in (None, 0) and num in side:
                    side[ratio] = side[num] / side[den] * factor
        result["deltas"] = {"rows": rows, "totals": {"a": ta, "b": tb, "delta": _deltas(ta, tb, measures)},
                            "empty_side": "a" if not a_rows else ("b" if not b_rows else None)}

    return {"result": result, "outcome": "answer"}


def _single_all_zero_row(rows: list[dict[str, Any]], additive: list[str]) -> bool:
    return len(rows) == 1 and all((rows[0]["values"].get(m) or 0) == 0 for m in additive)


def _totals(rows: list[dict[str, Any]], additive: list[str]) -> dict[str, Decimal]:
    return {m: sum((r["values"][m] for r in rows if r["values"].get(m) is not None), Decimal(0)) for m in additive}


def _deltas(a: dict[str, Any] | None, b: dict[str, Any] | None, measures: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in measures:
        av, bv = (a or {}).get(m), (b or {}).get(m)
        delta = (bv - av) if av is not None and bv is not None else None
        pct = (delta / av) if delta is not None and av not in (None, 0) else None
        out[m] = {"delta": delta, "pct": pct, "from_zero": av == 0 and bv not in (None, 0)}
    return out


def _null_reason(ratio: str, values: dict[str, Any]) -> str:
    num, den = RATIOS.get(ratio, (None, None, 1))[:2]
    for part in (den, num):
        if part and values.get(part) in (None, 0):
            return f"{part} is 0, so {ratio} is undefined"
    return f"{ratio} is undefined"


# --------------------------------------------------------------------------- answer

def answer(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    body, footer = templates.render(state, runtime.context.catalog)
    return {"answer": f"{body}\n\n{footer}" if footer else body, "answer_body": body, "footer": footer}
