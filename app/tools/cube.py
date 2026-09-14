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

from app.models.catalog import INTERNAL, VIEW, Catalog, Member, format_name

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

def _member(entry: dict[str, Any]) -> Member:
    name = entry["name"]
    return Member(
        name=name,
        short=name.split(".", 1)[1],
        title=entry.get("title") or name,
        short_title=entry.get("shortTitle") or entry.get("title") or name,
        type=entry.get("type", ""),
        agg_type=entry.get("aggType"),
        format=format_name(entry.get("format")),
        currency=entry.get("currency"),
        description=entry.get("description") or "",
    )


def _as_date(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


@observe(name="cube.catalog", as_type="tool", capture_input=False)  # the Cube object holds the API secret
def load_catalog(cube: Cube) -> Catalog:
    """Read the view's vocabulary from ``/meta`` and the data coverage from one ``/load``."""
    get_client().update_current_span(input={"view": VIEW})
    meta = cube.meta()
    view = next((c for c in meta.get("cubes", []) if c.get("name") == VIEW), None)
    if view is None:
        raise CubeError(None, f"view '{VIEW}' not found in /meta")

    catalog = Catalog()
    for entry in view.get("measures", []):
        m = _member(entry)
        if m.short not in INTERNAL:
            catalog.measures[m.short] = m
    for entry in view.get("dimensions", []):
        d = _member(entry)
        if d.type == "time":
            catalog.time_dimension = d.name
        else:
            catalog.dimensions[d.short] = d

    mp = catalog.member
    dims = list(catalog.dimensions)
    results = cube.load({  # one grouped query: known values of every string dimension + coverage
        "measures": [mp("first_date"), mp("last_date")],
        "dimensions": [mp(d) for d in dims],
        "limit": 1000, "timezone": "UTC",
    }).get("results") or []
    rows = results[0].get("data", []) if results else []
    for d in dims:
        catalog.values[d] = sorted({r[mp(d)] for r in rows if r.get(mp(d)) is not None})
    firsts = [_as_date(r[mp("first_date")]) for r in rows if r.get(mp("first_date"))]
    lasts = [_as_date(r[mp("last_date")]) for r in rows if r.get(mp("last_date"))]
    if not firsts or not lasts:
        raise CubeError(None, "the warehouse appears to be empty (no coverage dates)")
    catalog.coverage = (min(firsts), max(lasts))
    get_client().update_current_span(output=catalog.summary())
    return catalog
