"""The explicit agent flow: prepare -> agent <-> tools -> verify -> answer.

The model decides which tools to call; the graph decides what happens after
each reply (loop, verify, or stop) and enforces the hard limits. Every exit
goes through ``answer`` so the user always gets a rendered, honest reply.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    agent,
    answer,
    prepare,
    route_after_agent,
    route_after_tools,
    route_after_verify,
    tools,
    verify,
)
from app.models.state import AgentState, Deps


def build_graph() -> CompiledStateGraph:
    g = StateGraph(AgentState, context_schema=Deps)
    g.add_node("prepare", prepare)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("verify", verify)
    g.add_node("answer", answer)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "verify", "answer", "agent"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "answer"])
    g.add_conditional_edges("verify", route_after_verify, ["agent", "answer"])
    g.add_edge("answer", END)
    return g.compile(name="analytics-graph")


graph = build_graph()


def mermaid() -> str:
    return graph.get_graph().draw_mermaid()
