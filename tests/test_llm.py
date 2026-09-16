"""The OpenRouter binding: no silent retries, native tools with a JSON-protocol fallback, the free guard."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from openrouter.errors import NotFoundResponseError, NotFoundResponseErrorData, TooManyRequestsResponseError, TooManyRequestsResponseErrorData

import app.tools.llm as llm_module
from app.tools.llm import FreeInferenceViolation, LLMError, OpenRouterLLM


def completion(cost, model="x/y:free", content="hello", tool_call=None) -> dict:
    """A chat.completion body as the OpenRouter SDK would parse it."""
    message = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": tool_call}]
    return {
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": model, "provider": "P",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_call else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost},
    }


def rate_limited(retry_after="1", message="slow down") -> TooManyRequestsResponseError:
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=httpx.Request("POST", "https://x"))
    data = TooManyRequestsResponseErrorData.model_validate({"error": {"code": 429, "message": message}})
    return TooManyRequestsResponseError(data, response, "{}")


def not_found(message: str) -> NotFoundResponseError:
    response = httpx.Response(404, request=httpx.Request("POST", "https://x"))
    data = NotFoundResponseErrorData.model_validate({"error": {"code": 404, "message": message}})
    return NotFoundResponseError(data, response, "{}")


def describe_view(view: str) -> dict:
    """Describe a view."""
    return {}


TOOLS = [StructuredTool.from_function(describe_view)]


@pytest.fixture
def llm() -> OpenRouterLLM:
    return OpenRouterLLM("sk-or-test", "t", "https://x")


def test_bound_chat_model_kwargs(llm):
    chat = llm._llm
    assert chat.model_name == "openrouter/free"
    assert chat.request_timeout == 90000  # milliseconds
    assert chat.max_retries == 0
    assert chat.model_kwargs == {"retries": None}  # also disables the SDK's own retry config; no response_format
    assert llm.native_tools is True


def test_invoke_sends_the_tools_and_returns_the_routed_model_and_cost(llm):
    sent = []

    def fake_send(**kwargs):
        sent.append(kwargs)
        return completion(cost=0, tool_call={"name": "describe_view", "arguments": '{"view": "v"}'})

    llm._llm.client.chat.send = fake_send
    reply = llm.invoke([HumanMessage("q")], TOOLS)
    assert reply.model_name.endswith(":free") and reply.cost == 0 and llm.calls == 1
    assert reply.message.tool_calls[0]["name"] == "describe_view" and reply.message.tool_calls[0]["args"] == {"view": "v"}
    assert sent[0]["retries"] is None and sent[0]["model"] == "openrouter/free"
    assert [t["function"]["name"] for t in sent[0]["tools"]] == ["describe_view"]
    assert "response_format" not in sent[0]


def test_router_without_tool_support_switches_to_the_json_protocol_once(llm):
    sent = []

    def fake_send(**kwargs):
        sent.append("tools" in kwargs and kwargs["tools"] is not None)
        if sent[-1]:
            raise not_found("No endpoints found that support tool use")
        return completion(cost=0, content='{"tool": "describe_view", "args": {"view": "v"}}')

    llm._llm.client.chat.send = fake_send
    reply = llm.invoke([HumanMessage("q")], TOOLS)
    assert reply.message.content.startswith('{"tool"') and llm.calls == 2
    assert sent == [True, False] and llm.native_tools is False
    llm.invoke([HumanMessage("q")], TOOLS)  # stays on the JSON protocol
    assert sent == [True, False, False]


def test_other_404s_are_the_data_policy_error(llm):
    def fake_send(**kwargs):
        raise not_found("No endpoints found matching your data policy")

    llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        llm.invoke([HumanMessage("q")], TOOLS)
    assert info.value.kind == "data_policy" and llm.calls == 1 and llm.native_tools is True


def test_non_zero_cost_is_a_free_inference_violation(llm):
    llm._llm.client.chat.send = lambda **kwargs: completion(cost=0.001)
    with pytest.raises(FreeInferenceViolation):
        llm.invoke([HumanMessage("q")], TOOLS)
    assert llm.calls == 1


def test_non_free_model_is_a_free_inference_violation(llm):
    llm._llm.client.chat.send = lambda **kwargs: completion(cost=0, model="openai/gpt-4o")
    with pytest.raises(FreeInferenceViolation):
        llm.invoke([HumanMessage("q")], TOOLS)


def test_429_is_retried_once_then_reported(llm, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)

    def fake_send(**kwargs):
        raise rate_limited()

    llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        llm.invoke([HumanMessage("q")], TOOLS)
    assert info.value.kind == "rate_limited" and info.value.status == 429
    assert llm.calls == 2 and slept == [1.0]  # honoured Retry-After


def test_daily_cap_429_is_not_slept_on(llm, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)

    def fake_send(**kwargs):
        raise rate_limited(retry_after="86400", message="free-models-per-day")

    llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        llm.invoke([HumanMessage("q")], TOOLS)
    assert info.value.kind == "rate_limited" and slept == [] and llm.calls == 1


def test_transport_failures_become_llm_errors(llm):
    def fake_send(**kwargs):
        raise httpx.ReadTimeout("free router stalled", request=httpx.Request("POST", "https://x"))

    llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        llm.invoke([HumanMessage("q")], TOOLS)
    assert info.value.kind == "unavailable" and "ReadTimeout" in info.value.detail and llm.calls == 1


def test_http_200_error_body_is_unavailable(llm):
    llm._llm.client.chat.send = lambda **kwargs: {"error": {"code": 502, "message": "provider down"}}
    with pytest.raises(LLMError) as info:
        llm.invoke([HumanMessage("q")], TOOLS)
    assert info.value.kind == "unavailable" and info.value.status == 502


def test_float_zero_cost_is_reported_as_zero(llm):
    llm._llm.client.chat.send = lambda **kwargs: completion(cost=0.0)
    assert str(llm.invoke([HumanMessage("q")], TOOLS).cost) == "0"


def test_max_price_reaches_the_sdk_as_strings(llm):
    from openrouter import components, utils
    from openrouter.types import OptionalNullable
    prefs = llm._llm.openrouter_provider
    model = utils.get_pydantic_model(prefs, OptionalNullable[components.ProviderPreferences])
    assert model.max_price.prompt == "0" and model.max_price.completion == "0"  # ints would come back Unset


def test_429_hint_in_the_body_is_honoured(llm, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    data = TooManyRequestsResponseErrorData.model_validate({"error": {"code": 429, "message": "slow down",
                                                                      "metadata": {"headers": {"X-RateLimit-Reset": "0"}}}})

    def fake_send(**kwargs):
        raise TooManyRequestsResponseError(data, response, "{}")

    llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError):
        llm.invoke([HumanMessage("q")], TOOLS)
    assert slept == [1.0]  # reset in the past -> the minimum wait, from the body, not the 20 s default
