"""FastAPI app: the UI, a health check, the example questions and ``POST /ask``.

Every ``/ask`` response has the same key set, whatever happened, so the UI
renders one shape. One question runs at a time (``run_lock``); the catalog
is loaded once at startup and retried lazily if the semantic layer was down.
"""

from __future__ import annotations

from app import config  # noqa: F401  (must be first: loads .env before Langfuse is initialised)

import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.models.state import Deps
from app.runtime import build_cube, build_llm, lf, run_question, trace_url_for
from app.tools.cube import load_catalog
from app.web.questions import EXAMPLE_QUESTIONS

log = logging.getLogger(__name__)
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"
MODEL_ID = "openrouter/free"

PAYLOAD_KEYS = (
    "question", "as_of", "outcome", "error_kind", "error", "error_stage", "answer", "answer_body", "footer",
    "plan", "plan_raw", "rejected", "period", "compare_period", "filters", "cube_query", "normalized", "rows",
    "annotation", "result", "notes", "llm_model", "llm_cost", "llm_calls", "cube_calls", "trace_id", "trace_url",
    "timing_ms",
)
ERROR_STATUS = {"cube": 503, "llm": 502, "free_guard": 502, "validation": 502, "internal": 500, "busy": 409,
                "bad_request": 400}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    as_of: date | None = None


# --------------------------------------------------------------------------- pure helpers

def http_status(outcome: str | None, error_kind: str | None) -> int:
    if outcome in ("answer", "clarify", "unsupported", "no_data"):
        return 200
    return ERROR_STATUS.get(error_kind or "", 500)


def to_payload(question: str, as_of: date, out: dict[str, Any], timing_ms: int) -> dict[str, Any]:
    outcome = out.get("outcome")
    trace_id = out.get("trace_id")
    payload = {
        "question": question, "as_of": as_of.isoformat(), "outcome": outcome,
        "error_kind": out.get("error_kind"), "error": out.get("error"),
        "error_stage": "graph" if outcome == "error" else None,
        "answer": out.get("answer"), "answer_body": out.get("answer_body"), "footer": out.get("footer"),
        "plan": out.get("plan"), "plan_raw": out.get("plan_raw"), "rejected": out.get("rejected"),
        "period": out.get("period"), "compare_period": out.get("compare_period"),
        "filters": (out.get("cube_query") or {}).get("filters") or [],
        "cube_query": out.get("cube_query"), "normalized": out.get("normalized") or [],
        "rows": out.get("rows") or [], "annotation": out.get("annotation"), "result": out.get("result"),
        "notes": out.get("notes") or [],
        "llm_model": out.get("llm_model"), "llm_cost": out.get("llm_cost"),
        "llm_calls": out.get("llm_calls", 0), "cube_calls": out.get("cube_calls", 0),
        "trace_id": trace_id, "trace_url": trace_url_for(trace_id), "timing_ms": timing_ms,
    }
    return {k: payload[k] for k in PAYLOAD_KEYS}


def error_payload(question: str, as_of: date, kind: str, message: str, stage: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = dict.fromkeys(PAYLOAD_KEYS)
    payload.update(
        question=question, as_of=as_of.isoformat(), outcome="error", error_kind=kind, error=message,
        error_stage=stage, answer=message, answer_body=message, footer="",
        filters=[], normalized=[], rows=[], notes=[], llm_calls=0, cube_calls=0, timing_ms=0,
    )
    return payload


def respond(payload: dict[str, Any], status: int) -> JSONResponse:
    content = jsonable_encoder(payload, custom_encoder={Decimal: str, date: date.isoformat})
    return JSONResponse(status_code=status, content=content)


# --------------------------------------------------------------------------- app state

_shutdown_done = threading.Event()


CATALOG_RETRY_S = 60.0


def _flush() -> None:
    try:
        lf.flush()
    except Exception as exc:  # tracing must never break a request, but do say so
        log.warning("langfuse flush failed: %s", exc)


def _shutdown_once() -> None:
    """Shut the Langfuse client down at most once per process: a second
    ``shutdown()`` makes every later ``flush()`` block forever (langfuse 4.15),
    which matters when tests start several apps in one process."""
    if _shutdown_done.is_set():
        return
    _shutdown_done.set()
    try:
        lf.shutdown()
    except Exception as exc:
        log.warning("langfuse shutdown failed: %s", exc)


def ensure_catalog(app: FastAPI, *, force: bool = True) -> Any:
    """Load the catalog if it is missing. Never raises; returns the catalog or None.

    ``force=False`` (the /health poll) retries at most once per ``CATALOG_RETRY_S``
    so a misconfigured secret does not produce an error trace every 10 seconds.
    """
    state = app.state
    with state.catalog_lock:
        if state.catalog is None:
            if not force and time.monotonic() < state.catalog_retry_at:
                return None
            try:
                state.catalog = state.loader(state.cube)
                state.catalog_error = None
            except Exception as exc:  # CubeError or a transport failure
                state.catalog_error = (
                    f"The semantic layer is unreachable ({exc}, {state.cube.base_url}). No answer was produced."
                )
                state.catalog_retry_at = time.monotonic() + CATALOG_RETRY_S
                log.error("catalog unavailable: %s", state.catalog_error)
            finally:
                _flush()
        return state.catalog


def create_app(*, cube: Any = None, llm: Any = None, loader: Any = load_catalog) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cube = cube or build_cube()
        app.state.llm = llm or build_llm()
        app.state.loader = loader
        app.state.as_of = config.settings.as_of
        app.state.catalog = None
        app.state.catalog_error = None
        app.state.run_lock = threading.Lock()
        app.state.catalog_lock = threading.Lock()
        app.state.catalog_retry_at = 0.0
        ensure_catalog(app)  # degraded start if Cube is down; retried lazily on /ask
        yield
        _shutdown_once()

    app = FastAPI(title="ff-analytics-agent", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = exc.body if isinstance(exc.body, dict) else {}
        question = str(body.get("question") or "")
        as_of = getattr(request.app.state, "as_of", config.settings.as_of)
        return respond(error_payload(question, as_of, "bad_request", f"Invalid request: {exc.errors()[0]['msg']}"), 400)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_INDEX, headers={"Cache-Control": "no-store"})

    @app.get("/examples")
    async def examples() -> dict[str, Any]:
        return {"examples": [e._asdict() for e in EXAMPLE_QUESTIONS]}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        cube_ok = bool(state.cube.ready())
        if state.catalog is None and cube_ok:
            ensure_catalog(request.app, force=False)
        return {
            "cube": "ok" if cube_ok else "down",
            "cube_url_kind": "cloud" if "cubecloudapp.dev" in state.cube.base_url else "local",
            "catalog": {"loaded": True, **state.catalog.summary()} if state.catalog else {"loaded": False},
            "catalog_error": state.catalog_error,
            "langfuse": "enabled" if config.settings.langfuse_enabled else "disabled",
            "model": MODEL_ID,
            "as_of": state.as_of.isoformat(),
            "busy": state.run_lock.locked(),
        }

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> JSONResponse:
        state = request.app.state
        as_of = req.as_of or state.as_of
        if not state.run_lock.acquire(blocking=False):
            return respond(error_payload(req.question, as_of, "busy", "Another question is being answered; try again in a moment."), 409)
        try:
            catalog = ensure_catalog(request.app)
            if catalog is None:
                return respond(error_payload(req.question, as_of, "cube", state.catalog_error, stage="catalog"), 503)
            deps = Deps(llm=state.llm, cube=state.cube, catalog=catalog, as_of=as_of)
            started = time.perf_counter()
            out = run_question(req.question, deps)
            payload = to_payload(req.question, as_of, out, int((time.perf_counter() - started) * 1000))
            return respond(payload, http_status(payload["outcome"], payload["error_kind"]))
        except Exception as exc:
            log.exception("unexpected failure while answering %r", req.question)
            return respond(error_payload(req.question, as_of, "internal", f"{type(exc).__name__}: {exc}"), 500)
        finally:
            _flush()
            state.run_lock.release()

    return app


app = create_app()
