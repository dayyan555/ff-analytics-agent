"""Cube REST client and the LangChain tools the graph calls.

This module is the agent's only route to data. It never touches ClickHouse:
it speaks Cube's REST API (``/meta``, ``/dry-run``, ``/load``) with an
HS256 JWT signed from ``CUBEJS_API_SECRET``.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import httpx
import jwt
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langfuse import get_client, observe

from app.models.catalog import INTERNAL, VALUE_CACHE, Catalog, Kind, Member, View, format_name

CONTINUE_WAIT = "Continue wait"


class CubeError(Exception):
    """A Cube API or transport failure (status is None for transport errors)."""

    def __init__(self, status: int | None, message: str):
        super().__init__(f"{status or 'connection'}: {message}")
        self.status = status
        self.message = message


class CubeClient:
    def __init__(self, base_url: str, api_secret: str, *, timeout: float = 30.0, max_wait_s: float = 120.0):
        if not base_url or not api_secret:
            raise CubeError(None, "CUBE_URL and CUBEJS_API_SECRET must be set")
        self.base_url = base_url.rstrip("/")
        self._secret = api_secret
        self._max_wait_s = max_wait_s
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    # -- auth -------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        token = jwt.encode({"exp": int(time.time()) + 3600}, self._secret, algorithm="HS256")
        return {"Authorization": token, "Content-Type": "application/json"}

    # -- endpoints --------------------------------------------------------
    def ready(self) -> bool:
        try:
            return self._http.get(f"{self.base_url}/readyz", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def meta(self) -> dict[str, Any]:
        return self._request("GET", "/cubejs-api/v1/meta")

    def dry_run(self, query: dict[str, Any]) -> dict[str, Any]:
        """Validate a query without touching the warehouse; returns Cube's normalized form."""
        return self._request("POST", "/cubejs-api/v1/dry-run", {"query": query})

    def load(self, query: dict[str, Any]) -> dict[str, Any]:
        """Run a query. Always ``queryType: multi`` so the response is ``{"results": [...]}``
        for both regular and ``compareDateRange`` queries (one entry per date range)."""
        body = {"query": query, "queryType": "multi"}
        deadline = time.monotonic() + self._max_wait_s
        while True:
            data = self._request("POST", "/cubejs-api/v1/load", body)
            if data.get("error") != CONTINUE_WAIT:
                return data
            if time.monotonic() > deadline:  # Cube is still computing (cold start or slow warehouse)
                raise CubeError(None, "the semantic layer is still computing; please retry")
            time.sleep(0.5)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            resp = self._http.request(method, f"{self.base_url}{path}", json=body, headers=self._headers())
        except (httpx.HTTPError, jwt.PyJWTError) as exc:
            raise CubeError(None, f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text[:200]}
        if resp.status_code != 200:
            raise CubeError(resp.status_code, str(data.get("error", resp.text[:200])))
        return data


class Cube:
    """Cube exposed as two LangChain tools (``cube_dry_run``, ``cube_load``).

    The graph invokes the tools with its ``RunnableConfig`` so each call is a
    tool span in the Langfuse trace; the model never selects them.
    """

    def __init__(self, client: CubeClient):
        self.client = client
        self.dry_run_tool = StructuredTool.from_function(
            func=client.dry_run, name="cube_dry_run",
            description="Validate a Cube query and resolve its date ranges without running it.",
        )
        self.load_tool = StructuredTool.from_function(
            func=client.load, name="cube_load",
            description="Run a Cube query against the semantic layer and return rows.",
        )

    @property
    def base_url(self) -> str:
        return self.client.base_url

    def ready(self) -> bool:
        return self.client.ready()

    def meta(self) -> dict[str, Any]:
        return self.client.meta()

    def dry_run(self, query: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return self.dry_run_tool.invoke({"query": query}, config=config)

    def load(self, query: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return self.load_tool.invoke({"query": query}, config=config)


# -- catalog ---------------------------------------------------------------

def _member(entry: dict[str, Any], kind: Kind) -> Member:
    name = entry["name"]
    return Member(
        name=name,
        short=name.split(".", 1)[1],
        title=entry.get("title") or name,
        short_title=entry.get("shortTitle") or entry.get("title") or name,
        kind=kind,
        type=entry.get("type", ""),
        agg_type=entry.get("aggType"),
        format=format_name(entry.get("format")),
        currency=entry.get("currency"),
        description=entry.get("description") or "",
    )


def _as_date(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def _load_view(cube: Cube, entry: dict[str, Any]) -> View:
    view = View(name=entry["name"], title=entry.get("title") or entry["name"], description=entry.get("description") or "")
    for m in entry.get("measures", []):
        member = _member(m, "measure")
        if member.short not in INTERNAL:
            view.measures[member.short] = member
    for d in entry.get("dimensions", []):
        member = _member(d, "dimension")
        if member.type == "time":
            view.times[member.short] = member
            view.time_dimension = view.time_dimension or member.name
        else:
            view.dimensions[member.short] = member

    # One grouped query per view: the known values of every string dimension, and — when the
    # view exposes the internal coverage measures — the first and last day with data.
    internal = [m["name"] for m in entry.get("measures", []) if m["name"].split(".", 1)[1] in INTERNAL]
    dims = [m.name for m in view.dimensions.values()]
    if not dims and not internal:
        return view
    query: dict[str, Any] = {"dimensions": dims, "limit": VALUE_CACHE, "timezone": "UTC"}
    if internal:
        query["measures"] = internal
    results = cube.load(query).get("results") or []
    rows = results[0].get("data", []) if results else []
    complete = len(rows) < VALUE_CACHE  # a full page means the grouped query was cut: counts are lower bounds
    for member in view.dimensions.values():
        values = sorted({str(r[member.name]) for r in rows if r.get(member.name) is not None})
        view.values[member.short] = values
        view.value_counts[member.short] = len(values)
        view.values_complete[member.short] = complete
    if internal:
        first, last = (f"{view.name}.first_date", f"{view.name}.last_date")
        firsts = [_as_date(r[first]) for r in rows if r.get(first)]
        lasts = [_as_date(r[last]) for r in rows if r.get(last)]
        if firsts and lasts:
            view.coverage = (min(firsts), max(lasts))
    return view


@observe(name="cube.catalog", as_type="tool", capture_input=False)  # the Cube object holds the API secret
def load_catalog(cube: Cube) -> Catalog:
    """Read every view from ``/meta`` plus, per view, one ``/load`` for known values and coverage."""
    get_client().update_current_span(input={"cube_url": cube.base_url})
    meta = cube.meta()
    entries = [c for c in meta.get("cubes", []) if c.get("type") == "view"]
    if not entries:
        raise CubeError(None, "no views found in /meta (the data model must expose at least one view)")
    catalog = Catalog()
    for entry in entries:
        catalog.views[entry["name"]] = _load_view(cube, entry)
    if all(v.coverage is None for v in catalog.views.values()):
        raise CubeError(None, "the warehouse appears to be empty (no coverage dates)")
    get_client().update_current_span(output=catalog.summary())
    return catalog
