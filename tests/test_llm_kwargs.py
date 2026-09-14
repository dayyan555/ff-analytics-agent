"""The OpenRouter binding: no silent retries, JSON mode, and the free guard on the reply."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import HumanMessage
from openrouter.errors import TooManyRequestsResponseError, TooManyRequestsResponseErrorData

import app.tools.llm as llm_module
from app.tools.llm import FreeInferenceViolation, LLMError, OpenRouterPlanLLM


def completion(cost, model="x/y:free", content='{"intent": "clarify"}') -> dict:
    """A chat.completion body as the OpenRouter SDK would parse it."""
    return {
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": model, "provider": "P",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost},
    }


def rate_limited() -> TooManyRequestsResponseError:
    response = httpx.Response(429, headers={"retry-after": "1"}, request=httpx.Request("POST", "https://x"))
    data = TooManyRequestsResponseErrorData.model_validate({"error": {"code": 429, "message": "slow down"}})
    return TooManyRequestsResponseError(data, response, "{}")


@pytest.fixture
def plan_llm() -> OpenRouterPlanLLM:
    return OpenRouterPlanLLM("sk-or-test", "t", "https://x")


def test_bound_chat_model_kwargs(plan_llm):
    llm = plan_llm._llm
    assert llm.model_name == "openrouter/free"
    assert llm.request_timeout == 60000  # milliseconds
    assert llm.max_retries == 0
    assert llm.model_kwargs["retries"] is None  # also disables the SDK's own retry config
    assert llm.model_kwargs["response_format"] == {"type": "json_object"}


def test_plan_returns_the_routed_model_and_cost(plan_llm):
    sent = []

    def fake_send(**kwargs):
        sent.append(kwargs)
        return completion(cost=0)

    plan_llm._llm.client.chat.send = fake_send
    reply = plan_llm.plan([HumanMessage("q")])
    assert reply.text == '{"intent": "clarify"}'
    assert reply.model_name.endswith(":free") and reply.cost == 0
    assert plan_llm.calls == 1 and len(sent) == 1
    assert sent[0]["retries"] is None and sent[0]["model"] == "openrouter/free"
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["messages"][0]["content"] == "q"


def test_non_zero_cost_is_a_free_inference_violation(plan_llm):
    plan_llm._llm.client.chat.send = lambda **kwargs: completion(cost=0.001)
    with pytest.raises(FreeInferenceViolation):
        plan_llm.plan([HumanMessage("q")])
    assert plan_llm.calls == 1


def test_429_is_retried_once_then_reported(plan_llm, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)

    def fake_send(**kwargs):
        raise rate_limited()

    plan_llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        plan_llm.plan([HumanMessage("q")])
    assert info.value.kind == "rate_limited" and info.value.status == 429
    assert plan_llm.calls == 2
    assert slept == [1.0]  # honoured Retry-After


def test_transport_failures_become_llm_errors(plan_llm):
    def fake_send(**kwargs):
        raise httpx.ReadTimeout("free router stalled", request=httpx.Request("POST", "https://x"))

    plan_llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        plan_llm.plan([HumanMessage("q")])
    assert info.value.kind == "unavailable" and "ReadTimeout" in info.value.detail
    assert plan_llm.calls == 1  # no silent retry


def test_daily_cap_429_is_not_slept_on(plan_llm, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    response = httpx.Response(429, headers={"retry-after": "86400"}, request=httpx.Request("POST", "https://x"))
    data = TooManyRequestsResponseErrorData.model_validate({"error": {"code": 429, "message": "daily cap"}})

    def fake_send(**kwargs):
        raise TooManyRequestsResponseError(data, response, "{}")

    plan_llm._llm.client.chat.send = fake_send
    with pytest.raises(LLMError) as info:
        plan_llm.plan([HumanMessage("q")])
    assert info.value.kind == "rate_limited"
    assert slept == [] and plan_llm.calls == 1  # hint above the cap: give up immediately


def test_float_zero_cost_is_reported_as_zero(plan_llm):
    plan_llm._llm.client.chat.send = lambda **kwargs: completion(cost=0.0)
    reply = plan_llm.plan([HumanMessage("q")])
    assert str(reply.cost) == "0"


def test_http_200_error_body_is_unavailable_not_a_repair_turn(plan_llm):
    plan_llm._llm.client.chat.send = lambda **kwargs: {"error": {"code": 502, "message": "provider down"}}
    with pytest.raises(LLMError) as info:
        plan_llm.plan([HumanMessage("q")])
    assert info.value.kind == "unavailable" and info.value.status == 502
