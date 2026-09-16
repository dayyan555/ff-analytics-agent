"""The graph's nodes and routers.

    prepare -> agent <-> tools ... -> verify -> answer

``agent`` is the only node that calls a model; ``tools`` is the only node that
touches data (through the Cube-backed toolkit). ``verify`` checks numerical
consistency, not the meaning of the narrative. Every exit — clarify,
unsupported, no data, error, budget exhausted — still goes through ``answer``,
so the user always gets a rendered, honest reply.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from app.agent.answer import render
from app.agent.prompt import initial_messages
from app.agent.verify import numbers_in, relevant_queries, unverified_numbers
from app.models.state import AgentState, Deps, ToolCall
from app.tools.cube import CubeError
from app.tools.llm import FreeInferenceViolation, LLMError, content_text, with_json_protocol
from app.tools.toolkit import FINAL_ANSWER, RUN_QUERY, Toolkit

MAX_MODEL_CALLS = 7  # one recovery turn beyond the normal discovery/query/answer path
MAX_VERIFY_RETRIES = 1  # how often a rejected answer goes back to the model
FINAL_KINDS = ("answer", "clarify", "unsupported")
JSON_CALL_ID = "json-call"  # tool calls parsed from plain text carry no provider id

_JSON_START = re.compile(r"\{")


# --------------------------------------------------------------------------- prepare

def prepare(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    toolkit = Toolkit(deps.cube, deps.catalog)
    views = toolkit.list_views()["views"]  # seeded: no model call spent on it
    messages = initial_messages(
        state["question"], views, deps.as_of, MAX_MODEL_CALLS,
        json_protocol=not deps.llm.native_tools, tools=toolkit.tools(),
    )
    return {"messages": messages, "llm_calls": 0, "cube_calls": 0, "verify_failures": 0, "pending": [], "final": None}


# --------------------------------------------------------------------------- agent

def agent(state: AgentState, runtime: Runtime[Deps], config: RunnableConfig) -> dict[str, Any]:
    """One model turn: the reply is tool calls (native or JSON protocol), a final answer, or prose."""
    deps = runtime.context
    toolkit = Toolkit(deps.cube, deps.catalog)
    calls = state.get("llm_calls", 0)
    if calls >= MAX_MODEL_CALLS:
        return {"outcome": "error", "error_kind": "budget", "pending": [],
                "error": f"the agent did not finish within {MAX_MODEL_CALLS} model turns"}

    messages = state["messages"]
    if not deps.llm.native_tools:  # the router switched to the JSON protocol: the system message must describe it
        messages = with_json_protocol(messages, toolkit.tools())
    before = deps.llm.calls
    try:
        reply = deps.llm.invoke(messages, toolkit.tools(), config)
    except FreeInferenceViolation as exc:
        return {"outcome": "error", "error_kind": "free_guard", "error": str(exc), "pending": [],
                "llm_calls": calls + (deps.llm.calls - before)}
    except LLMError as exc:
        return {"outcome": "error", "error_kind": "llm", "error": f"{exc.kind}: {exc.detail}", "pending": [],
                "llm_calls": calls + (deps.llm.calls - before)}

    out: dict[str, Any] = {
        "llm_calls": calls + (deps.llm.calls - before),
        "llm_models": [reply.model_name] if reply.model_name else [],
        "llm_cost": str(reply.cost) if reply.cost is not None else None,
        "messages": ([messages[0]] if messages[0] is not state["messages"][0] else []) + [reply.message],  # same id: replaces
        "pending": [],
    }
    tool_calls = parse_tool_calls(reply.message, {t.name for t in toolkit.tools()})
    if len(tool_calls) == 1 and tool_calls[0]["name"] == FINAL_ANSWER and not tool_calls[0].get("error"):
        out["final"] = _final(tool_calls[0])
        return out
    if tool_calls:  # tools to run (extra or mixed-in final_answer calls are refused by the tools node, one reply per id)
        out["pending"] = tool_calls
        return out

    text = content_text(reply.message).strip()
    if text and any(q.get("ok") for q in state.get("queries", [])):  # prose after data: the model is narrating
        out["final"] = {"kind": "answer", "text": text, "tool_call_id": None}
        return out
    out["messages"].append(HumanMessage("Use the tools to answer, then call final_answer. Do not answer in prose."))
    return out


def _final(call: ToolCall) -> dict[str, Any]:
    args = call.get("args") or {}
    kind = str(args.get("kind") or "answer").strip().lower()
    text = str(args.get("text") or "").strip()
    call_id = call.get("id")
    return {"kind": kind if kind in FINAL_KINDS else "answer", "text": text,
            "tool_call_id": call_id if call_id != JSON_CALL_ID else None}


_CALL_KEYS = ("tool", "name", "args", "arguments", "input", "parameters")


def parse_tool_calls(msg: AIMessage, known: set[str] | None = None) -> list[ToolCall]:
    """Native tool calls first (including ones whose arguments were not valid JSON, so every id gets a
    reply); otherwise the JSON protocol in the text — one object or several, only for known tool names."""
    calls: list[ToolCall] = []
    for tc in msg.tool_calls or []:
        args = tc.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"query": args}
        calls.append({"id": tc.get("id") or JSON_CALL_ID, "name": tc["name"], "args": args if isinstance(args, dict) else {}})
    for tc in getattr(msg, "invalid_tool_calls", None) or []:
        calls.append({"id": tc.get("id") or JSON_CALL_ID, "name": tc.get("name") or "", "args": {},
                      "error": f"the arguments were not valid JSON ({str(tc.get('error') or '')[:120]})"})
    if calls:
        return calls
    text = re.sub(r"<think>.*?(?:</think>|\Z)", "", content_text(msg), flags=re.S)  # an unclosed block is reasoning too
    decoder = json.JSONDecoder()
    pos = 0
    while (m := _JSON_START.search(text, pos)):
        try:
            obj, end = decoder.raw_decode(text, m.start())
        except ValueError:
            pos = m.start() + 1
            continue
        pos = end
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        if not isinstance(name, str) or (known is not None and name not in known):
            continue  # not a call: models quote {"name": "marketing_performance.spend", ...} objects in prose
        args = obj.get("args", obj.get("arguments", obj.get("input", obj.get("parameters"))))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        if not isinstance(args, dict):  # flattened form: {"tool": "final_answer", "kind": ..., "text": ...}
            args = {k: v for k, v in obj.items() if k not in _CALL_KEYS}
        calls.append({"id": JSON_CALL_ID, "name": name, "args": args})
    return calls


# --------------------------------------------------------------------------- tools

def tools(state: AgentState, runtime: Runtime[Deps], config: RunnableConfig) -> dict[str, Any]:
    """Run the tool calls the model asked for; every result goes back to the model verbatim."""
    deps = runtime.context
    toolkit = Toolkit(deps.cube, deps.catalog)
    messages: list[BaseMessage] = []
    steps: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    out: dict[str, Any] = {"pending": []}
    step_no = len(state.get("steps", []))
    query_no = len(state.get("queries", []))

    for call in state.get("pending", []):
        name, args = call["name"], call.get("args") or {}
        if call.get("error"):
            result = {"ok": False, "error": f"{name or 'tool call'}: {call['error']}",
                      "hint": "send the arguments as one JSON object with the tool's parameters"}
        elif name == FINAL_ANSWER:
            result = {"ok": False, "error": "final_answer ignored: call it on its own, after you have seen the results"}
        else:
            try:
                result = toolkit.run(name, args, config)
            except CubeError as exc:
                out.update(outcome="error", error_kind="cube", error=f"{exc.status or 'connection'}: {exc.message}")
                break
            if name == RUN_QUERY and _repeated_server_error(result, state.get("queries", []) + queries):
                out.update(outcome="error", error_kind="cube", error=f"{result.get('status')}: {result.get('error')}")
                break
        step_no += 1
        if name == RUN_QUERY:
            query_no += 1
            result = {"query_id": f"q{query_no}", **result}
            queries.append(result)
        steps.append({"step": step_no, "tool": name, "args": args, "ok": bool(result.get("ok")), "summary": summarise(name, result)})
        messages.append(_tool_result_message(call, result))

    out.update(messages=messages, steps=steps, queries=queries, cube_calls=state.get("cube_calls", 0) + toolkit.cube_calls)
    return out


def _repeated_server_error(result: dict[str, Any], previous: list[dict[str, Any]]) -> bool:
    """A 5xx the model cannot fix (the warehouse is down) comes back identical on the retry; stop then."""
    if result.get("ok") or int(result.get("status") or 0) < 500:
        return False
    return any(not q.get("ok") and q.get("status") == result.get("status") and q.get("error") == result.get("error")
               for q in previous)


def _tool_result_message(call: ToolCall, result: dict[str, Any]) -> BaseMessage:
    payload = json.dumps(result, default=str)
    if call["id"] == JSON_CALL_ID:  # JSON protocol: no provider tool-call id to answer to
        return HumanMessage(f"Result of {call['name']}:\n{payload}")
    return ToolMessage(content=payload, tool_call_id=call["id"], name=call["name"])


def summarise(name: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"error: {str(result.get('error', ''))[:120]}"
    if name == "describe_view":
        return f"{len(result.get('measures', []))} measures, {len(result.get('dimensions', []))} dimensions"
    if name == "search_fields":
        return f"{result.get('showing', 0)} match(es): " + ", ".join(m["name"] for m in result.get("matches", []))
    if name == "find_dimension_values":
        return f"{result.get('showing', 0)} of {result.get('total_values', 0)} values: " + ", ".join(result.get("matches", [])[:5])
    if name == RUN_QUERY:
        return f"{result.get('row_count', 0)} row(s), {len(result.get('columns', []))} column(s)"
    if name == "list_views":
        return f"{len(result.get('views', []))} view(s)"
    return "ok"


# --------------------------------------------------------------------------- verify

def verify(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Check numerical consistency before accepting a final answer; one retry."""
    final = state.get("final") or {}
    kind, text = final.get("kind", "answer"), final.get("text", "")
    ok_queries = relevant_queries(state.get("queries", []))
    if kind in ("clarify", "unsupported"):
        bad = unverified_numbers(text, ok_queries)  # a question back to the user must not smuggle figures in
        if bad:
            return _reject(state, final, f"a {kind} reply must not contain figures that are not in query results: "
                                          + ", ".join(bad[:8]) + ". Rephrase without them", bad, fallback_kind=kind)
        return {"outcome": kind, "verification": {"checked": len(numbers_in(text)), "unverified": []}}

    if not ok_queries:
        return _reject(state, final, "a final answer of kind 'answer' needs a successful run_query first; "
                                      "run the query, or use kind 'clarify' / 'unsupported'", [])
    bad = unverified_numbers(text, ok_queries)
    if bad:
        return _reject(state, final, "these numbers are not in the query results: " + ", ".join(bad[:8]) +
                       ". Use only numbers from the results (or run another query), then call final_answer again", bad)
    outcome = "no_data" if _empty(ok_queries[-1]) else "answer"  # the answer is about the last query
    return {"outcome": outcome, "verification": {"checked": len(numbers_in(text)), "unverified": []}}


def _empty(query: dict[str, Any]) -> bool:
    """Only an empty result or an explicit Cube existence check establishes absence."""
    return query.get("has_data") is False or not query.get("rows")


def _reject(state: AgentState, final: dict[str, Any], reason: str, bad: list[str],
            fallback_kind: str | None = None) -> dict[str, Any]:
    failures = state.get("verify_failures", 0)
    verification = {"checked": len(numbers_in(final.get("text", ""))), "unverified": bad}
    if failures >= MAX_VERIFY_RETRIES:
        if fallback_kind:  # keep the outcome, drop the text: the renderer's template for that kind is shown instead
            return {"outcome": fallback_kind, "final": {**final, "text": ""}, "verification": verification,
                    "notes": [f"{fallback_kind} text dropped: {reason[:120]}"]}
        return {"outcome": "error", "error_kind": "validation", "error": reason, "verification": verification}
    result = {"ok": False, "error": reason, "hint": "answer again using only numbers from the results"}
    payload = json.dumps(result)
    call_id = final.get("tool_call_id")
    message: BaseMessage = (ToolMessage(content=payload, tool_call_id=call_id, name=FINAL_ANSWER) if call_id
                            else HumanMessage(f"Result of {FINAL_ANSWER}:\n{payload}"))
    return {"messages": [message], "verify_failures": failures + 1, "final": None,
            "notes": [f"answer rejected once: {reason[:120]}"]}


# --------------------------------------------------------------------------- answer

def answer(state: AgentState, runtime: Runtime[Deps]) -> dict[str, Any]:
    body, footer = render(state, runtime.context.catalog)
    return {"answer": f"{body}\n\n{footer}" if footer else body, "answer_body": body, "footer": footer, "pending": []}


# --------------------------------------------------------------------------- routers

def route_after_agent(state: AgentState) -> str:
    if state.get("outcome"):
        return "answer"
    if state.get("final"):
        return "verify"
    if state.get("pending"):
        return "tools"
    return "agent"  # the model answered in prose too early; it was nudged (the budget stops the loop)


def route_after_tools(state: AgentState) -> str:
    return "answer" if state.get("outcome") else "agent"


def route_after_verify(state: AgentState) -> str:
    return "answer" if state.get("outcome") else "agent"
