"""The explicit agent flow: interpret -> build_query -> query_cube -> validate_results -> answer.

Every early exit (clarify, unsupported, no data, error) goes through ``answer``
so the user always gets a rendered, honest reply.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    answer,
    build_query,
    interpret,
    query_cube,
    route_after_build,
    route_after_interpret,
    route_after_query,
    validate_results,
)
from app.models.state import AgentState, Deps


def build_graph() -> CompiledStateGraph:
    g = StateGraph(AgentState, context_schema=Deps)
    g.add_node("interpret", interpret)
    g.add_node("build_query", build_query)
    g.add_node("query_cube", query_cube)
    g.add_node("validate_results", validate_results)
    g.add_node("answer", answer)

    g.add_edge(START, "interpret")
    g.add_conditional_edges("interpret", route_after_interpret, ["build_query", "answer"])
    g.add_conditional_edges("build_query", route_after_build, ["query_cube", "answer"])
    g.add_conditional_edges("query_cube", route_after_query, ["validate_results", "answer"])
    g.add_edge("validate_results", "answer")
    g.add_edge("answer", END)
    return g.compile(name="analytics-graph")


graph = build_graph()


def mermaid() -> str:
    return graph.get_graph().draw_mermaid()
