"""Wiring: build the dependencies, run one question inside a Langfuse trace."""

from __future__ import annotations

from typing import Any

from app import config  # noqa: F401  (loads .env before the Langfuse client is created)
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.agent.graph import graph
from app.config import settings
from app.models.state import Deps
from app.tools.cube import Cube, CubeClient
from app.tools.llm import OpenRouterPlanLLM

lf = get_client()  # one client per process: get_trace_url() caches the project id on the instance


def build_cube() -> Cube:
    return Cube(CubeClient(settings.cube_url, settings.cube_api_secret))


def build_llm() -> OpenRouterPlanLLM:
    return OpenRouterPlanLLM(settings.openrouter_api_key, settings.openrouter_app_title, settings.openrouter_app_url)


def run_question(question: str, deps: Deps) -> dict[str, Any]:
    """Run the graph for one question; the whole run is one Langfuse trace."""
    as_of = deps.as_of
    with propagate_attributes(trace_name="marketing-agent", tags=["assessment"], metadata={"as_of": as_of.isoformat()}):
        with lf.start_as_current_observation(as_type="agent", name="marketing-agent", input={"question": question}) as root:
            out = graph.invoke(
                {"question": question, "as_of": as_of.isoformat(), "cube_calls": 0, "notes": []},
                context=deps,
                config={"callbacks": [CallbackHandler()], "run_name": "analytics-graph"},
            )
            root.update(
                output={"answer": out.get("answer"), "outcome": out.get("outcome")},
                metadata={
                    "llm_model": out.get("llm_model"), "llm_cost": out.get("llm_cost"),
                    "llm_calls": out.get("llm_calls", 0), "cube_calls": out.get("cube_calls", 0),
                    "error_kind": out.get("error_kind"),
                },
                level="ERROR" if out.get("outcome") == "error" else "DEFAULT",
            )
            root.score_trace(name="outcome", value=out.get("outcome", "error"), data_type="CATEGORICAL")
            root.score_trace(
                name="free_inference_ok",
                value=0.0 if out.get("error_kind") == "free_guard" else 1.0,
                data_type="BOOLEAN",
            )
            trace_id = root.trace_id
    if not trace_id or set(trace_id) == {"0"}:  # NoOp tracer when tracing is disabled
        trace_id = None
    return {**out, "trace_id": trace_id}


def trace_url_for(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    try:
        return lf.get_trace_url(trace_id=trace_id)
    except Exception:  # no keys, network, or auth — the UI falls back to showing the id
        return None
