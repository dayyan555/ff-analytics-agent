"""LangGraph state and the runtime dependencies injected into nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from operator import add
from typing import Annotated, Any, Literal, NamedTuple, Protocol, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from app.models.catalog import Catalog

Outcome = Literal["answer", "clarify", "unsupported", "no_data", "error"]
ErrorKind = Literal["free_guard", "llm", "cube", "validation"]


class PlanReply(NamedTuple):
    text: str
    model_name: str | None
    cost: Decimal | None


class PlanLLM(Protocol):
    """One planning call; ``calls`` counts HTTP requests actually made."""

    calls: int

    def plan(self, messages: list[BaseMessage], config: RunnableConfig | None = None) -> PlanReply: ...


class CubeTools(Protocol):
    """The two LangChain tools wrapping the Cube REST API (see app/tools/cube.py)."""

    def dry_run(self, query: dict, config: RunnableConfig | None = None) -> dict: ...

    def load(self, query: dict, config: RunnableConfig | None = None) -> dict: ...


@dataclass
class Deps:
    llm: PlanLLM
    cube: CubeTools
    catalog: Catalog
    as_of: date


class AgentState(TypedDict, total=False):
    # input
    question: str
    as_of: str
    # interpret
    plan_raw: str | None
    plan: dict[str, Any] | None
    llm_model: str | None
    llm_cost: str | None
    llm_calls: int
    # build_query
    rejected: dict[str, Any] | None
    period: dict[str, Any] | None
    compare_period: dict[str, Any] | None
    cube_query: dict[str, Any] | None
    # query_cube
    normalized: list[dict[str, Any]]
    rows: list[list[dict[str, Any]]]
    annotation: dict[str, Any] | None
    cube_calls: int
    # validate_results
    result: dict[str, Any] | None
    # outcome
    outcome: Outcome
    error_kind: ErrorKind | None
    error: str | None
    # answer
    answer: str
    answer_body: str
    footer: str
    notes: Annotated[list[str], add]
