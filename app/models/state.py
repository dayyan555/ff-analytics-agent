"""LangGraph state and the runtime dependencies injected into nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from operator import add
from typing import Annotated, Any, Literal, NamedTuple, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages

from app.models.catalog import Catalog

Outcome = Literal["answer", "clarify", "unsupported", "no_data", "error"]
ErrorKind = Literal["free_guard", "llm", "cube", "validation", "budget"]
FinalKind = Literal["answer", "clarify", "unsupported"]


class LLMReply(NamedTuple):
    message: AIMessage
    model_name: str | None
    cost: Decimal | None


class AgentLLM(Protocol):
    """One model call with the tool schemas attached; ``calls`` counts HTTP requests actually made."""

    calls: int
    native_tools: bool  # False once the router proved it cannot route tool requests (JSON protocol only)

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool], config: RunnableConfig | None = None) -> LLMReply: ...


class CubeAPI(Protocol):
    """The Cube REST calls the toolkit needs (see app/tools/cube.py)."""

    base_url: str

    def ready(self) -> bool: ...

    def meta(self) -> dict: ...

    def dry_run(self, query: dict, config: RunnableConfig | None = None) -> dict: ...

    def load(self, query: dict, config: RunnableConfig | None = None) -> dict: ...


@dataclass
class Deps:
    llm: AgentLLM
    cube: CubeAPI
    catalog: Catalog
    as_of: date


class ToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class AgentState(TypedDict, total=False):
    # input
    question: str
    as_of: str
    # the conversation the model sees (system + question + tool calls + tool results)
    messages: Annotated[list[BaseMessage], add_messages]
    # agent
    llm_calls: int
    llm_models: Annotated[list[str], add]
    llm_cost: str | None
    pending: list[ToolCall]  # tool calls the model asked for, to be run by the tools node
    final: dict[str, Any] | None  # {"kind": ..., "text": ...} once the model called final_answer
    # tools
    steps: Annotated[list[dict[str, Any]], add]  # one entry per tool call: name, args, summary
    queries: Annotated[list[dict[str, Any]], add]  # every run_query: query, ok, rows, columns, error ...
    cube_calls: int
    # verify
    verify_failures: int
    verification: dict[str, Any] | None
    # outcome
    outcome: Outcome
    error_kind: ErrorKind | None
    error: str | None
    # answer
    answer: str
    answer_body: str
    footer: str
    notes: Annotated[list[str], add]
