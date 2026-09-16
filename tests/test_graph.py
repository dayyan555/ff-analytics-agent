"""The agent loop end to end, offline: a scripted model and a stub Cube."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.graph import graph
from app.agent.nodes import MAX_MODEL_CALLS, parse_tool_calls
from app.tools.cube import CubeError
from app.tools.llm import FakeLLM, FreeInferenceViolation, LLMError
from tests.conftest import MP, StubCube, cube_query, final, json_call, result_set, row, tool_call

Q1 = "How much did we spend by channel in August 2026?"
ROWS = [row(channel="meta", spend=13689.34), row(channel="google", spend=11020.10), row(channel="email", spend=0)]


def run(deps_factory, replies, cube=None, question=Q1):
    deps = deps_factory(FakeLLM(list(replies)), cube=cube)
    return graph.invoke({"question": question, "as_of": "2026-09-14", "notes": []}, context=deps), deps


# --------------------------------------------------------------------------- happy path

def test_describe_query_answer(deps):
    out, d = run(deps, [
        tool_call("describe_view", {"view": "marketing_performance"}),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta spent the most in August 2026: $13,689.34, ahead of Google at $11,020.10; email had no paid spend."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer" and not out.get("error_kind")
    assert out["llm_calls"] == 3 and out["cube_calls"] == 2 and d.llm.calls == 3
    assert [s["tool"] for s in out["steps"]] == ["describe_view", "run_query"]
    assert out["queries"][0]["query_id"] == "q1" and out["queries"][0]["row_count"] == 3
    assert out["final"]["kind"] == "answer"
    assert out["verification"] == {"checked": 2, "unverified": []}
    body = out["answer_body"]
    assert body.startswith("Meta spent the most") and "Channel" in body and "$13,689.34" in body and "$0.00" in body
    assert "Period: 2026-08-01 to 2026-08-31 (UTC, inclusive)" in out["footer"]
    assert "spend = Total advertising spend in USD in the period." in out["footer"]
    assert "model: fake/model:free" in out["footer"] and "cost: $0" in out["footer"]
    assert "model calls: 3 · cube calls: 2 · tool calls: 2" in out["footer"]
    assert out["llm_models"] == ["fake/model:free"] * 3


def test_the_model_sees_seeded_views_then_tool_results(deps):
    out, d = run(deps, [
        tool_call("describe_view", {"view": "marketing_performance"}),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta: $13,689.34."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    first = d.llm.seen[0]
    assert isinstance(first[0], SystemMessage) and isinstance(first[1], HumanMessage)
    assert '"name": "marketing_performance"' in first[0].content and "2026-03-01" in first[0].content
    assert "Today is 2026-09-14" in first[0].content and "Tool protocol" not in first[0].content
    assert "spend" not in first[0].content.split("How to work")[0].split("Data model")[1]  # no fields in the prompt
    second = d.llm.seen[1]
    assert isinstance(second[-1], ToolMessage) and second[-1].tool_call_id == "call-1"
    described = json.loads(second[-1].content)
    assert described["ok"] and any(m["name"] == MP + "spend" for m in described["measures"])
    third = d.llm.seen[2]
    rows = json.loads(third[-1].content)["rows"]
    assert rows[0] == {"channel": "meta", "spend": "13689.34"}


def test_json_protocol_replies_are_parsed_and_answered_as_human_messages(deps):
    llm = FakeLLM([
        json_call("describe_view", {"view": "marketing_performance"}),
        "```json\n" + json_call("run_query", {"query": cube_query(["spend"], ["channel"])}) + "\n```",
        json_call("final_answer", {"kind": "answer", "text": "Meta spent $13,689.34."}),
    ], native_tools=False)
    deps_ = deps(llm, cube=StubCube(results=[result_set(ROWS)]))
    out = graph.invoke({"question": Q1, "as_of": "2026-09-14", "notes": []}, context=deps_)
    assert out["outcome"] == "answer" and out["llm_calls"] == 3
    assert "Tool protocol" in llm.seen[0][0].content  # the JSON protocol is in the prompt when native tools are off
    assert isinstance(llm.seen[1][-1], HumanMessage) and llm.seen[1][-1].content.startswith("Result of describe_view")


def test_prose_after_data_counts_as_the_answer(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        AIMessage(content="Meta spent $13,689.34 in August 2026, the most of any channel."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer" and out["final"]["text"].startswith("Meta spent")


def test_prose_before_any_data_is_nudged_once(deps):
    out, d = run(deps, [
        AIMessage(content="Sure! Let me think about marketing spend."),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        final("Meta spent $13,689.34."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer" and out["llm_calls"] == 3
    nudge = d.llm.seen[1][-1]
    assert isinstance(nudge, HumanMessage) and "final_answer" in nudge.content


# --------------------------------------------------------------------------- the model fixing its own query

def test_a_bad_field_name_comes_back_with_did_you_mean_and_the_model_retries(deps):
    cube = StubCube(results=[result_set(ROWS)])
    calls = {"n": 0}
    original_dry_run = cube.dry_run

    def dry_run(query, config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CubeError(500, "Error: 'spent' not found for path 'marketing_performance.spent'")
        return original_dry_run(query, config)

    cube.dry_run = dry_run
    out, d = run(deps, [
        tool_call("run_query", {"query": cube_query(["spent"], ["channel"])}),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta spent $13,689.34."),
    ], cube=cube)
    assert out["outcome"] == "answer"
    assert [q["ok"] for q in out["queries"]] == [False, True] and out["queries"][0]["did_you_mean"] == [MP + "spend"]
    fed_back = json.loads(d.llm.seen[1][-1].content)
    assert fed_back["ok"] is False and "describe_view" in fed_back["hint"]
    assert out["cube_calls"] == 3  # failed dry-run + dry-run + load


# --------------------------------------------------------------------------- verify

def test_an_invented_number_is_rejected_once_then_accepted(deps):
    out, d = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        final("Meta spent $15,000.00 in August 2026."),
        final("Meta spent $13,689.34 in August 2026.", call_id="call-final-2"),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer" and out["llm_calls"] == 3 and out["verify_failures"] == 1
    rejection = d.llm.seen[2][-1]
    assert isinstance(rejection, ToolMessage) and rejection.tool_call_id == "call-final"
    assert "$15,000.00" in rejection.content
    assert out["notes"] and out["notes"][0].startswith("answer rejected once")


def test_two_invented_answers_end_as_a_validation_error_with_the_table(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        final("Meta spent $15,000.00."),
        final("Meta spent $16,000.00.", call_id="call-final-2"),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "error" and out["error_kind"] == "validation"
    assert out["verification"]["unverified"] == ["$16,000.00"]
    assert "not backed by the data" in out["answer_body"] and "$13,689.34" in out["answer_body"]  # the rows are still shown


def test_an_answer_without_any_query_is_rejected(deps):
    out, d = run(deps, [
        final("Meta spent $13,689.34."),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        final("Meta spent $13,689.34.", call_id="call-final-2"),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer" and out["llm_calls"] == 3
    assert "needs a successful run_query" in d.llm.seen[1][-1].content


@pytest.mark.parametrize("kind", ["clarify", "unsupported"])
def test_clarify_and_unsupported_pass_through_without_data(deps, kind):
    out, _ = run(deps, [final("Which period do you mean? Data covers 2026-03-01 to 2026-08-31.", kind=kind)],
                 question="How did we do recently?")
    assert out["outcome"] == kind and out["cube_calls"] == 0 and out["llm_calls"] == 1
    assert out["answer_body"].startswith("Which period") and "Period:" not in out["footer"]


def test_an_ungrouped_zero_aggregate_with_no_matching_records_is_no_data(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["purchases", "cost_per_purchase"])}),
        final("Summer Sale had 0 purchases in August 2026."),
    ], cube=StubCube(results=[[result_set([row(purchases=0, cost_per_purchase=None)])], [result_set([])]]))
    assert out["outcome"] == "no_data"
    assert out["queries"][0]["has_data"] is False and out["cube_calls"] == 3


def test_markdown_markers_are_stripped_from_the_narrative(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        final("## Spend\n**Meta** spent $13,689.34."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["answer_body"].startswith("Spend\nMeta spent $13,689.34.")


def test_different_periods_or_columns_keep_both_tables(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"], ("2026-07-01", "2026-08-31"))}),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta spent $13,689.34."),
    ], cube=StubCube(results=[result_set(ROWS)]))
    assert out["answer_body"].count("Channel") == 2
    assert "2026-07-01 to 2026-08-31" in out["footer"] and "2026-08-01 to 2026-08-31" in out["footer"]
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["purchases"], ["device"])}),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta spent $13,689.34 and desktop had 5 purchases."),
    ], cube=StubCube(results=[[result_set([row(device="desktop", purchases=5)])], [result_set(ROWS)]]))
    assert "Device" in out["answer_body"] and "Channel" in out["answer_body"]  # different shapes: both


def test_empty_rows_become_no_data(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["purchases"], filters=[{"member": MP + "campaign_name", "operator": "equals", "values": ["Summer Sale"]}])}),
        final("No purchases were recorded for Summer Sale in August 2026."),
    ], cube=StubCube(results=[result_set([])]))
    assert out["outcome"] == "no_data" and "Data covers 2026-03-01 to 2026-08-31" in out["answer_body"]
    assert "Filters: campaign_name equals Summer Sale" in out["footer"]


# --------------------------------------------------------------------------- failures

def test_cube_auth_failure_is_terminal(deps):
    out, _ = run(deps, [tool_call("run_query", {"query": cube_query(["spend"])})],
                 cube=StubCube(error=CubeError(403, "Invalid token")))
    assert out["outcome"] == "error" and out["error_kind"] == "cube" and "403: Invalid token" in out["error"]
    assert out["llm_calls"] == 1 and out["cube_calls"] == 1


def test_llm_errors_and_the_free_guard(deps):
    for exc, kind in [(LLMError("rate_limited", 429, "daily cap"), "llm"), (FreeInferenceViolation("non-free model served"), "free_guard")]:
        d = deps(FakeLLM(raise_=exc))
        out = graph.invoke({"question": Q1, "as_of": "2026-09-14", "notes": []}, context=d)
        assert out["outcome"] == "error" and out["error_kind"] == kind
        assert out["llm_calls"] == 1 and out["cube_calls"] == 0


def test_the_step_budget_stops_a_runaway_loop(deps):
    replies = [tool_call("describe_view", {"view": "marketing_performance"}, f"call-{i}") for i in range(MAX_MODEL_CALLS + 3)]
    out, d = run(deps, replies)
    assert out["outcome"] == "error" and out["error_kind"] == "budget"
    assert out["llm_calls"] == MAX_MODEL_CALLS and d.llm.calls == MAX_MODEL_CALLS
    assert f"within {MAX_MODEL_CALLS} model turns" in out["answer_body"]


def test_final_answer_mixed_with_other_calls_is_refused(deps):
    mixed = AIMessage(content="", tool_calls=[
        {"name": "run_query", "args": {"query": cube_query(["spend"], ["channel"])}, "id": "c1"},
        {"name": "final_answer", "args": {"kind": "answer", "text": "Meta: $1."}, "id": "c2"},
    ])
    out, d = run(deps, [mixed, final("Meta spent $13,689.34.")], cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer"
    refused = json.loads(d.llm.seen[1][-1].content)
    assert refused["ok"] is False and "final_answer ignored" in refused["error"]


# --------------------------------------------------------------------------- parsing

def test_parse_tool_calls_native_and_json_variants():
    native = AIMessage(content="", tool_calls=[{"name": "run_query", "args": {"query": {"measures": []}}, "id": "x"}])
    assert parse_tool_calls(native) == [{"id": "x", "name": "run_query", "args": {"query": {"measures": []}}}]
    text = AIMessage(content='<think>hmm</think> I will call a tool: {"name": "describe_view", "arguments": {"view": "v"}} done')
    assert parse_tool_calls(text) == [{"id": "json-call", "name": "describe_view", "args": {"view": "v"}}]
    assert parse_tool_calls(AIMessage(content="just prose {not json}")) == []
    two = AIMessage(content=json_call("describe_view", {"view": "a"}) + "\n" + json_call("describe_view", {"view": "b"}))
    assert [c["args"]["view"] for c in parse_tool_calls(two)] == ["a", "b"]


def test_cost_is_reported_as_a_plain_zero(deps):
    d = deps(FakeLLM([final("Which period?", kind="clarify")], cost=Decimal("0.0")))
    out = graph.invoke({"question": Q1, "as_of": "2026-09-14", "notes": []}, context=d)
    assert out["llm_cost"] == "0.0" or out["llm_cost"] == "0"


# --------------------------------------------------------------------------- end-to-end regression cases

def test_json_protocol_rejection_is_a_human_message_not_a_tool_message(deps):
    llm = FakeLLM([
        json_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        json_call("final_answer", {"kind": "answer", "text": "Meta spent $15,000.00."}),
        json_call("final_answer", {"kind": "answer", "text": "Meta spent $13,689.34."}),
    ], native_tools=False)
    d = deps(llm, cube=StubCube(results=[result_set(ROWS)]))
    out = graph.invoke({"question": Q1, "as_of": "2026-09-14", "notes": []}, context=d)
    assert out["outcome"] == "answer" and out["verify_failures"] == 1
    rejection = llm.seen[2][-1]
    assert isinstance(rejection, HumanMessage) and "$15,000.00" in rejection.content
    assert not any(isinstance(m, ToolMessage) for m in llm.seen[2])


def test_invalid_native_tool_arguments_get_a_reply_per_id(deps):
    broken = AIMessage(content="", tool_calls=[], invalid_tool_calls=[
        {"name": "run_query", "args": "{measures: [", "id": "bad-1", "error": "Expecting property name", "type": "invalid_tool_call"}])
    out, d = run(deps, [broken, tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"), final("Meta spent $13,689.34.")],
                 cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer"
    reply = d.llm.seen[1][-1]
    assert isinstance(reply, ToolMessage) and reply.tool_call_id == "bad-1"
    assert json.loads(reply.content)["ok"] is False and "not valid JSON" in reply.content
    assert out["steps"][0]["ok"] is False and out["steps"][0]["tool"] == "run_query"


def test_quoted_field_objects_in_prose_are_not_tool_calls():
    prose = AIMessage(content='The field is {"name": "marketing_performance.spend", "title": "Spend"} and I will use it.')
    assert parse_tool_calls(prose, {"run_query", "describe_view", "final_answer"}) == []


def test_flattened_final_answer_json_and_unclosed_think_blocks():
    flat = AIMessage(content='{"tool": "final_answer", "kind": "clarify", "text": "Which period?"}')
    assert parse_tool_calls(flat, {"final_answer"}) == [{"id": "json-call", "name": "final_answer", "args": {"kind": "clarify", "text": "Which period?"}}]
    cut = AIMessage(content='<think>I could call {"tool": "run_query", "args": {}} but')
    assert parse_tool_calls(cut, {"run_query"}) == []


def test_several_final_answer_calls_are_each_answered(deps):
    two = AIMessage(content="", tool_calls=[
        {"name": "final_answer", "args": {"kind": "answer", "text": "A"}, "id": "f1"},
        {"name": "final_answer", "args": {"kind": "answer", "text": "B"}, "id": "f2"},
    ])
    out, d = run(deps, [tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}), two, final("Meta spent $13,689.34.")],
                 cube=StubCube(results=[result_set(ROWS)]))
    assert out["outcome"] == "answer"
    replies = [m for m in d.llm.seen[2] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in replies][-2:] == ["f1", "f2"]


def test_figures_in_a_clarification_are_rejected_then_dropped(deps):
    out, d = run(deps, [
        final("Do you mean the $48,000 spent in July or the 2,500 purchases?", kind="clarify"),
        final("Do you mean July or August, and which metric?", kind="clarify", call_id="c2"),
    ], question="How did we do?")
    assert out["outcome"] == "clarify" and out["answer_body"].startswith("Do you mean July or August")
    assert "$48,000" in d.llm.seen[1][-1].content
    out, _ = run(deps, [
        final("The $48,000 figure is not in the model.", kind="unsupported"),
        final("Roughly 2,500 purchases are not available.", kind="unsupported", call_id="c2"),
    ], question="x")
    assert out["outcome"] == "unsupported" and "$48,000" not in out["answer_body"] and "2,500" not in out["answer_body"]
    assert out["answer_body"] == "I can't answer that from the semantic layer."


def test_switching_to_the_json_protocol_mid_run_adds_it_to_the_system_prompt(deps):
    llm = FakeLLM([tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}), final("Meta spent $13,689.34.")])
    d = deps(llm, cube=StubCube(results=[result_set(ROWS)]))
    original_invoke = llm.invoke

    def flip_then_invoke(messages, tools, config=None):
        llm.native_tools = False  # the router said "no endpoints support tool use" on this call
        return original_invoke(messages, tools, config)

    llm.invoke = flip_then_invoke
    out = graph.invoke({"question": Q1, "as_of": "2026-09-14", "notes": []}, context=d)
    assert out["outcome"] == "answer"
    assert "Tool protocol:" not in llm.seen[0][0].content and "Tool protocol:" in llm.seen[1][0].content
    assert isinstance(out["messages"][0], SystemMessage) and "Tool protocol:" in out["messages"][0].content
    assert sum(isinstance(m, SystemMessage) for m in out["messages"]) == 1  # replaced by id, not appended


def test_prose_after_a_failed_query_is_not_the_answer(deps):
    cube = StubCube(results=[result_set(ROWS)])
    calls = {"n": 0}
    original = cube.dry_run

    def dry_run(query, config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CubeError(500, "Error: 'spent' not found for path 'marketing_performance.spent'")
        return original(query, config)

    cube.dry_run = dry_run
    out, d = run(deps, [
        tool_call("run_query", {"query": cube_query(["spent"], ["channel"])}),
        AIMessage(content="Oops, 'spent' is wrong; I will use 'spend'."),
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}, "call-2"),
        final("Meta spent $13,689.34."),
    ], cube=cube)
    assert out["outcome"] == "answer" and out["verify_failures"] == 0 and out["llm_calls"] == 4


def test_the_same_server_error_twice_stops_the_run(deps):
    cube = StubCube(dry_run_error=CubeError(500, "Error: Connection refused"))
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"])}),
        tool_call("run_query", {"query": cube_query(["spend"])}, "call-2"),
        final("x"),
    ], cube=cube)
    assert out["outcome"] == "error" and out["error_kind"] == "cube" and "Connection refused" in out["error"]
    assert out["llm_calls"] == 2


def test_no_data_and_the_table_follow_the_last_query(deps):
    out, _ = run(deps, [
        tool_call("run_query", {"query": cube_query(["spend"], ["channel"])}),
        tool_call("run_query", {"query": cube_query(["purchases"], filters=[{"member": MP + "campaign_name", "operator": "equals", "values": ["Summer Sale"]}])}, "call-2"),
        final("Summer Sale had no purchases in August 2026."),
    ], cube=StubCube(results=[[result_set(ROWS)], [result_set([])]]))
    assert out["outcome"] == "no_data" and "Channel" not in out["answer_body"]
