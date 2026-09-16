"""The tools the model can call. Every one of them is backed by Cube.

Result contract (what the model reads back):
  * always a JSON object; ``{"ok": true, ...}`` or ``{"ok": false, "error": ..., "hint": ...}``
  * discovery fields and values are capped and say so (``showing`` / ``total``)
  * every identifier the model must reuse (field names, dimension values) is returned
    exactly as it has to be written in a query
"""

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from app.models.catalog import ALIASES, VALUE_CACHE, VALUE_SAMPLE, Catalog, Member, View, format_label, format_name
from app.models.state import CubeAPI, FinalKind
from app.tools.cube import CubeError

MAX_ROWS = 50  # rows returned to the model per query (the query's limit is capped to this)
MAX_FIELDS = 40  # fields listed by describe_view before it points at search_fields
MAX_VALUES = 20  # values returned by find_dimension_values
MAX_MATCHES = 5  # fields returned by search_fields
QUERY_KEYS = frozenset({"measures", "dimensions", "timeDimensions", "filters", "order", "limit", "segments"})
TERMINAL_STATUSES = frozenset({401, 403, 429})  # not the model's fault: auth or quota -> the run stops
FINAL_ANSWER = "final_answer"
RUN_QUERY = "run_query"

QUERY_FORMAT = (
    "Cube query JSON: {\"measures\": [field, ...], \"dimensions\": [field, ...], "
    "\"timeDimensions\": [{\"dimension\": time_field, \"dateRange\": [\"YYYY-MM-DD\", \"YYYY-MM-DD\"], "
    "\"granularity\": \"day\"|\"week\"|\"month\" (only for a time series; each row is labelled by the period's START date, weeks start on Monday)}], "
    "\"filters\": [{\"member\": field, \"operator\": \"equals\"|\"notEquals\"|\"contains\"|\"gt\"|\"gte\"|\"lt\"|\"lte\", "
    "\"values\": [...]}], \"order\": {field: \"asc\"|\"desc\"}, \"limit\": n}. "
    "To compare two periods put \"compareDateRange\": [[from, to], [from, to]] in the timeDimension instead of dateRange."
)


JSON_PROTOCOL = """
Tool protocol: reply with exactly one JSON object and nothing else, either
{"tool": "<tool name>", "args": {...}} to call a tool, or
{"tool": "final_answer", "args": {"kind": "answer|clarify|unsupported", "text": "..."}} to finish.
Tools: """


def json_protocol_text(tools: list[BaseTool]) -> str:
    """The plain-text tool protocol any model can follow, with the tool list."""
    return JSON_PROTOCOL + "; ".join(f"{t.name}({', '.join(t.args)}): {t.description}" for t in tools)


def _short(view: str, key: str) -> str:
    return key[len(view) + 1:] if key.startswith(view + ".") else key


class Toolkit:
    """The six tools, bound to one Cube connection and one catalog."""

    def __init__(self, cube: CubeAPI, catalog: Catalog):
        self.cube = cube
        self.catalog = catalog
        self.cube_calls = 0
        self._tools = [
            StructuredTool.from_function(func=self.list_views, name="list_views"),
            StructuredTool.from_function(func=self.describe_view, name="describe_view"),
            StructuredTool.from_function(func=self.search_fields, name="search_fields"),
            StructuredTool.from_function(func=self.find_dimension_values, name="find_dimension_values"),
            StructuredTool.from_function(func=self.run_query, name=RUN_QUERY),
            StructuredTool.from_function(func=self.final_answer, name=FINAL_ANSWER),
        ]

    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def run(self, name: str, args: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        """Execute one tool call by name; unknown names and bad arguments come back as ``ok: false``."""
        tool = next((t for t in self._tools if t.name == name), None)
        if tool is None:
            return {"ok": False, "error": f"unknown tool '{name}'",
                    "hint": "available tools: " + ", ".join(t.name for t in self._tools)}
        try:
            return tool.invoke(args, config=config)
        except CubeError:
            raise
        except Exception as exc:  # pydantic validation of the arguments, mostly
            return {"ok": False, "error": f"invalid arguments for {name}: {str(exc)[:300]}",
                    "hint": tool.description}

    # ------------------------------------------------------------------ tools

    def list_views(self) -> dict[str, Any]:
        """List the views (subject areas) of the data model with their date coverage. Call describe_view next."""
        return {"ok": True, "views": [v.summary() for v in self.catalog.views.values()]}

    def describe_view(self, view: str) -> dict[str, Any]:
        """Describe one view: its measures (metrics) and dimensions with descriptions and formats, the time
        dimension, the data coverage and the query format. Call it before your first run_query on a view."""
        v = self.catalog.view(view)
        if v is None:
            return {"ok": False, "error": f"unknown view '{view}'",
                    "hint": "valid views: " + ", ".join(self.catalog.views) or "none"}
        measures, dimensions = list(v.measures.values()), list(v.dimensions.values())
        out: dict[str, Any] = {
            "ok": True, "view": v.name, "description": v.description,
            "time_dimension": v.time_dimension,
            "data_from": v.coverage[0].isoformat() if v.coverage else None,
            "data_to": v.coverage[1].isoformat() if v.coverage else None,
            "measures": [m.describe() for m in measures[:MAX_FIELDS]],
            "dimensions": [self._describe_dimension(v, m) for m in dimensions[:MAX_FIELDS]],
            "showing": {"measures": min(len(measures), MAX_FIELDS), "dimensions": min(len(dimensions), MAX_FIELDS)},
            "total": {"measures": len(measures), "dimensions": len(dimensions)},
            "query_format": QUERY_FORMAT,
        }
        if len(measures) > MAX_FIELDS or len(dimensions) > MAX_FIELDS:
            out["hint"] = f"only the first {MAX_FIELDS} measures and {MAX_FIELDS} dimensions are listed; use search_fields to find the others"
        return out

    def _describe_dimension(self, view: View, m: Member) -> dict[str, Any]:
        d = m.describe()
        values = view.values.get(m.short, [])
        d["value_count"] = view.value_counts.get(m.short, len(values))
        d["example_values"] = values[:VALUE_SAMPLE]
        return d

    def search_fields(self, text: str, view: str | None = None) -> dict[str, Any]:
        """Search the data model for measures and dimensions matching some words (a metric name, a synonym,
        part of a description). Use it when describe_view is too long or a field is not found by name."""
        matches = self.catalog.search(text, view=view, limit=MAX_MATCHES)
        pool = self.catalog.views[view].members() if view in self.catalog.views else self.catalog.members()
        out: dict[str, Any] = {
            "ok": True, "text": text, "searched": len(pool), "showing": len(matches),
            "matches": [{**m.describe(), "kind": m.kind, "score": score} for m, score in matches],
        }
        if not matches:
            out["hint"] = "no field matches; the data model may not have this metric — consider final_answer with kind unsupported"
        return out

    def find_dimension_values(self, dimension: str, text: str | None = None) -> dict[str, Any]:
        """Find the exact values of a dimension (e.g. campaign names, countries) that contain some text, or list
        the first values when text is empty. Use the returned spelling in filters."""
        member = self.catalog.member(dimension)
        if member is None or member.kind != "dimension":
            return {"ok": False, "error": f"unknown dimension '{dimension}'",
                    "hint": "use the full name from describe_view, e.g. marketing_performance.channel"}
        if member.type == "time":
            return {"ok": False, "error": f"'{dimension}' is the time dimension; it has no list of values",
                    "hint": "filter on time with timeDimensions: [{dimension, dateRange: [from, to]}]"}
        view = self.catalog.views[member.view]
        cached = view.values.get(member.short, [])
        complete = view.values_complete.get(member.short, True)
        total = view.value_counts.get(member.short, len(cached))
        wanted = (text or "").strip()
        if not wanted:
            matches = cached[:MAX_VALUES]
        else:
            alias = ALIASES.get(member.short, {}).get(wanted.lower())
            needles = {wanted.lower()} | ({alias.lower()} if alias else set())
            if complete:
                matches = [v for v in cached if any(n in v.lower() for n in needles)][:MAX_VALUES]
            else:  # more values than we cache: let the warehouse search
                matches = self._search_values(member.name, sorted(needles))
        out: dict[str, Any] = {"ok": True, "dimension": member.name, "text": wanted or None,
                               "matches": matches, "showing": len(matches),
                               "total_values": total if complete else f"{total}+"}
        if matches:
            out["note"] = "use the exact spelling above in filters: {member, operator: equals, values: [...]}"
        elif wanted:
            out["hint"] = f"no value contains '{wanted}'; call without text to list the first {MAX_VALUES} of {out['total_values']}"
        return out

    def _search_values(self, dimension: str, needles: list[str]) -> list[str]:
        values: list[str] = []
        for needle in needles:
            self.cube_calls += 1
            results = self.cube.load({"dimensions": [dimension], "limit": MAX_VALUES,
                                      "filters": [{"member": dimension, "operator": "contains", "values": [needle]}]})
            rows = (results.get("results") or [{}])[0].get("data", [])
            values += [str(r[dimension]) for r in rows if r.get(dimension) is not None and str(r[dimension]) not in values]
        return sorted(values)[:MAX_VALUES]

    def run_query(self, query: dict[str, Any] | str) -> dict[str, Any]:
        """Run a Cube query (see query_format from describe_view) against the semantic layer and return the rows.
        Field names must be the full names from describe_view; dates are absolute YYYY-MM-DD. If ok is false,
        fix the query using error and hint and call run_query again."""
        if isinstance(query, str):
            try:
                query = json.loads(query)
            except ValueError:
                return {"ok": False, "error": "query must be a JSON object", "hint": QUERY_FORMAT}
        if not isinstance(query, dict) or not query:
            return {"ok": False, "error": "query must be a JSON object", "hint": QUERY_FORMAT}
        unknown = set(query) - QUERY_KEYS
        if unknown:
            return {"ok": False, "error": f"unknown query keys: {sorted(unknown)}", "hint": QUERY_FORMAT}
        query = {**query, "timezone": "UTC"}
        limit = query.get("limit", MAX_ROWS)
        if limit is None:
            limit = MAX_ROWS
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return {"ok": False, "error": "limit must be a positive integer", "hint": QUERY_FORMAT}
        query["limit"] = min(limit, MAX_ROWS)

        try:  # no explicit config: LangChain's context config nests these Cube tool spans under run_query
            self.cube_calls += 1
            normalized = self.cube.dry_run(query).get("normalizedQueries") or []
            self.cube_calls += 1
            results = self.cube.load(query).get("results") or []
            has_data = self._has_data(query, results)
        except CubeError as exc:
            if exc.status in TERMINAL_STATUSES or exc.status is None:
                raise  # auth, quota or transport: the run stops with error_kind = cube
            return self._query_error(query, exc)

        rows, columns, truncated = self._flatten(results)
        grouped = bool(query.get("dimensions")) or any(td.get("granularity") for td in query.get("timeDimensions", []))
        hit_limit = grouped and query["limit"] == MAX_ROWS and any(len(r.get("data", [])) >= MAX_ROWS for r in results)
        complete = not truncated and not hit_limit
        out: dict[str, Any] = {
            "ok": True, "query": query, "row_count": len(rows), "showing": min(len(rows), MAX_ROWS),
            "periods": _periods(normalized), "columns": columns, "rows": rows[:MAX_ROWS],
            "has_data": has_data, "complete": complete,
        }
        if not complete:
            out["note"] = (
                f"Results are limited to {len(out['rows'])} rows and may be incomplete. "
                "Do not infer overall totals or shares from these rows; request an ungrouped aggregate through Cube."
            )
        elif has_data is False:
            out["note"] = "no rows for this query; check the period against data_from/data_to and the filter values"
        return out

    def _has_data(self, query: dict[str, Any], results: list[dict[str, Any]]) -> bool | None:
        """Distinguish an empty ungrouped aggregate from actual zero-valued facts.

        For a zero/NULL aggregate, group the same measures and filters by the
        view's date and request one row. This checks existence through Cube
        without adding a metric to the deployed semantic model.
        """
        data = [row for result in results for row in result.get("data", [])]
        if not data:
            return False
        if query.get("dimensions") or any(td.get("granularity") for td in query.get("timeDimensions", [])):
            return True
        measures = query.get("measures", [])
        try:
            if any(Decimal(str(row[m])) != 0 for row in data for m in measures if row.get(m) is not None):
                return True
        except InvalidOperation:
            return None
        members = [self.catalog.member(m) for m in measures]
        views = {m.view for m in members if m}
        view = self.catalog.view(next(iter(views))) if len(views) == 1 else None
        if view is None or not view.time_dimension:
            return None  # absence has not been established; do not label a zero as missing data
        probe = {**query, "dimensions": [view.time_dimension], "limit": 1, "order": {}}
        self.cube_calls += 1
        return any(r.get("data") for r in self.cube.load(probe).get("results", []))

    def _query_error(self, query: dict[str, Any], exc: CubeError) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "query": query, "status": exc.status, "error": exc.message[:300]}
        missing = _missing_field(exc.message)
        if missing:
            out["did_you_mean"] = self.catalog.suggest(missing)
            out["hint"] = "fix the field name and call run_query again; describe_view lists every valid name"
        else:
            out["hint"] = "fix the query and call run_query again. " + QUERY_FORMAT
        return out

    def _flatten(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        """Cube's result sets -> rows keyed by short field names, a column list with titles and formats,
        and whether any result set had to be cut (several sets share the row budget)."""
        rows: list[dict[str, Any]] = []
        columns: dict[str, dict[str, Any]] = {}
        per_set = max(1, MAX_ROWS // max(1, len(results)))
        truncated = False
        for result in results:
            ann = {**result.get("annotation", {}).get("dimensions", {}),
                   **result.get("annotation", {}).get("measures", {}),
                   **result.get("annotation", {}).get("timeDimensions", {})}
            data = result.get("data", [])
            if len(results) > 1 and len(data) > per_set:
                data, truncated = data[:per_set], True
            for raw in data:
                row: dict[str, Any] = {}
                # with a granularity Cube returns both "view.date.month" and "view.date"; keep the granular one
                granular = {k.rsplit(".", 1)[0] for k in raw if k.count(".") == 2}
                for key, value in raw.items():
                    if key == "compareDateRange":
                        row["date_range"] = _range_label(value)
                        columns.setdefault("date_range", {"key": "date_range", "name": "compareDateRange",
                                                          "title": "Date range", "type": "string"})
                        continue
                    if key in granular:
                        continue
                    grain = key.rsplit(".", 1)[1] if key.count(".") == 2 else None
                    member = self.catalog.member(key.rsplit(".", 1)[0] if grain else key)
                    view = member.view if member else key.split(".", 1)[0]
                    short = _short(view, key).replace(".", "_")
                    row[short] = str(value)[:10] if grain and value is not None else value
                    if short not in columns:
                        entry = ann.get(key, {})
                        title = entry.get("shortTitle") or entry.get("title") or (short.rsplit("_", 1)[0] if grain else short)
                        col: dict[str, Any] = {"key": short, "name": key,
                                               "title": f"{title} ({grain})" if grain else title,
                                               "type": entry.get("type") or ("time" if grain else member.type if member else "string")}
                        label = format_label(format_name(entry.get("format")) or (member.format if member else None),
                                             entry.get("currency") or (member.currency if member else None))
                        if label:
                            col["format"] = label
                        if member and member.kind == "measure":
                            col["aggregation"] = "sum" if member.agg_type == "sum" else "ratio"
                        columns[short] = col
                rows.append(row)
        return rows, list(columns.values()), truncated or len(rows) > MAX_ROWS

    def final_answer(self, kind: FinalKind, text: str) -> dict[str, Any]:
        """Finish. kind "answer": a short answer using only numbers that appear in run_query results;
        kind "clarify": ask the user one specific question when the request is ambiguous;
        kind "unsupported": the data model cannot answer this (say what is missing). Never guess numbers."""
        return {"ok": True, "kind": kind, "text": text}  # intercepted by the graph; never sent to the model


# ---------------------------------------------------------------------- helpers

def _missing_field(message: str) -> str | None:
    """Cube: "Error: 'spent' not found for path 'marketing_performance.spent'" -> the full path."""
    marker = "not found for path '"
    if marker in message:
        rest = message.split(marker, 1)[1]
        return rest.split("'", 1)[0]
    return None


def _range_label(value: Any) -> str:
    """Cube's compareDateRange row key: "2026-07-01T00:00:00.000 - 2026-07-31T23:59:59.999" -> "2026-07-01 to 2026-07-31"."""
    text = str(value)
    parts = [p.strip()[:10] for p in text.split(" - ")]
    return " to ".join(parts) if len(parts) == 2 else text


def _periods(normalized: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The absolute date ranges Cube resolved (one per result set)."""
    periods = []
    for q in normalized:
        for td in q.get("timeDimensions", []):
            if td.get("dateRange"):
                start, end = td["dateRange"]
                periods.append({"from": str(start)[:10], "to": str(end)[:10]})
    return periods
