"""OpenRouter free-router access, with the zero-paid-inference guard.

Two things are enforced in code rather than trusted from configuration:
the request never retries silently (the SDK's default is up to an hour of
5xx backoff), and every reply must come from a ``:free`` model at cost 0.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openrouter import ChatOpenRouter
from openrouter.errors import OpenRouterError

from app.models.plan import PLAN_RESPONSE_FORMAT
from app.models.state import PlanReply

MODEL_ID = "openrouter/free"
LLMErrorKind = Literal["rate_limited", "data_policy", "unavailable"]


class FreeInferenceViolation(Exception):
    """The router served something that is not demonstrably free."""


class LLMError(Exception):
    def __init__(self, kind: LLMErrorKind, status: int | None, detail: str):
        super().__init__(f"{kind} ({status}): {detail}")
        self.kind = kind
        self.status = status
        self.detail = detail


def assert_free(model_name: str | None, cost: Decimal | None) -> None:
    """Raise unless the routed model id ends in ``:free`` and the reported cost is exactly 0.

    A missing cost is a violation too: OpenRouter always reports ``usage.cost``,
    so its absence means we cannot prove the call was free.
    """
    if not isinstance(model_name, str) or not model_name.endswith(":free"):
        raise FreeInferenceViolation(f"non-free model served: {model_name!r}")
    if cost is None:
        raise FreeInferenceViolation(f"cost missing from the response for {model_name}")
    if cost != 0:
        raise FreeInferenceViolation(f"non-zero cost {cost} for {model_name}")


MAX_BACKOFF_S = 60.0


def _retry_after_seconds(headers) -> float | None:
    """Wait hint from a 429: Retry-After, else X-RateLimit-Reset (epoch ms), else 20 s.

    Returns None when the hint exceeds ``MAX_BACKOFF_S`` (a daily-cap 429): the
    caller should give up immediately instead of blocking the server for hours.
    """
    wait = 20.0
    try:
        if headers.get("retry-after"):
            wait = float(headers["retry-after"])
        elif headers.get("x-ratelimit-reset"):
            wait = float(headers["x-ratelimit-reset"]) / 1000.0 - datetime.now(timezone.utc).timestamp()
    except (TypeError, ValueError):
        pass
    if wait > MAX_BACKOFF_S:
        return None
    return max(1.0, wait)


def _classify(exc: Exception) -> LLMError:
    """Map any failure of the planning call to a typed LLMError (never a paid fallback)."""
    if isinstance(exc, OpenRouterError):
        if exc.status_code == 429:
            return LLMError("rate_limited", 429, exc.message)
        if exc.status_code == 404:
            return LLMError(
                "data_policy", 404,
                "no free endpoint matches your account's data policy - enable the free-endpoint "
                "privacy toggles in OpenRouter Settings > Privacy",
            )
        return LLMError("unavailable", exc.status_code, exc.message)
    if isinstance(exc, ValueError):  # ChatOpenRouter: HTTP-200 body carrying an error, or no choices
        match = re.search(r"\(code: (\d{3})\)", str(exc))
        status = int(match.group(1)) if match else None
        return LLMError("rate_limited" if status == 429 else "unavailable", status, str(exc)[:300])
    return LLMError("unavailable", None, f"{type(exc).__name__}: {str(exc)[:300]}")  # httpx timeouts etc.


RELAXED_RESPONSE_FORMAT = {"type": "json_object"}
_SCHEMA_UNSUPPORTED = re.compile(r"structured|response_format|json_schema", re.I)


class OpenRouterPlanLLM:
    """Production ``PlanLLM``: one planning request, one explicit 429 retry, nothing else.

    The request carries the Plan as a strict JSON schema so that models honouring
    ``response_format`` return plain JSON. In practice the free router still routes to
    models that ignore it, which is why ``interpret`` re-rolls and repairs; if the router
    itself rejects the schema (HTTP 400), the process falls back to plain JSON mode.
    """

    def __init__(self, api_key: str, app_title: str, app_url: str):
        self.calls = 0
        self.response_format = PLAN_RESPONSE_FORMAT
        self._llm = ChatOpenRouter(
            model=MODEL_ID,
            api_key=api_key,
            temperature=0,
            max_tokens=4000,  # reasoning models spend tokens before the JSON; free, so leave room
            timeout=90_000,  # milliseconds; some free models are slow
            max_retries=0,  # no LangChain-side retry config ...
            model_kwargs={
                "retries": None,  # ... and none from the OpenRouter SDK either
                "response_format": PLAN_RESPONSE_FORMAT,
            },
            reasoning={"effort": "low"},
            openrouter_provider={"max_price": {"prompt": 0, "completion": 0}},
            app_title=app_title,
            app_url=app_url,
        )

    def plan(self, messages: list[BaseMessage], config: RunnableConfig | None = None) -> PlanReply:
        try:
            msg = self._invoke(messages, config)
        except OpenRouterError as exc:
            if exc.status_code == 400 and self._relax_if_schema_unsupported(exc):
                return self.plan(messages, config)  # one retry in plain JSON mode
            if exc.status_code != 429:
                raise _classify(exc) from exc
            wait = _retry_after_seconds(exc.headers)
            if wait is None:
                raise LLMError("rate_limited", 429, "rate limit hint exceeds 60 s; not retrying") from exc
            time.sleep(wait)
            try:
                msg = self._invoke(messages, config)
            except Exception as again:
                raise _classify(again) from again
        except Exception as exc:
            raise _classify(exc) from exc

        meta = msg.response_metadata or {}
        raw_cost = meta.get("cost")
        try:
            cost = Decimal(str(raw_cost)) if raw_cost is not None else None
        except InvalidOperation as exc:
            raise FreeInferenceViolation(f"cost is not numeric: {raw_cost!r}") from exc
        if cost is not None and cost == 0:
            cost = Decimal(0)  # the SDK parses cost as float; keep "0", not "0.0"
        reply = PlanReply(text=_content(msg), model_name=meta.get("model_name"), cost=cost)
        assert_free(reply.model_name, reply.cost)
        return reply

    def _invoke(self, messages: list[BaseMessage], config: RunnableConfig | None) -> AIMessage:
        self.calls += 1
        return self._llm.invoke(messages, config=config)

    def _relax_if_schema_unsupported(self, exc: OpenRouterError) -> bool:
        """Switch this process to plain JSON mode if the router rejected the strict schema."""
        if self.response_format is RELAXED_RESPONSE_FORMAT or not _SCHEMA_UNSUPPORTED.search(exc.message or exc.body or ""):
            return False
        self.response_format = RELAXED_RESPONSE_FORMAT
        self._llm.model_kwargs = {**self._llm.model_kwargs, "response_format": RELAXED_RESPONSE_FORMAT}
        return True


class FakePlanLLM:
    """Offline ``PlanLLM`` for the tests: canned replies, or a raised exception."""

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        model_name: str = "fake/model:free",
        cost: Decimal | None = Decimal(0),
        raise_: Exception | None = None,
    ):
        self._replies = list(replies or [])
        self._model_name = model_name
        self._cost = cost
        self._raise = raise_
        self.calls = 0
        self.seen: list[list[BaseMessage]] = []

    def plan(self, messages: list[BaseMessage], config: RunnableConfig | None = None) -> PlanReply:
        self.calls += 1
        self.seen.append(list(messages))
        if self._raise is not None:
            raise self._raise
        text = self._replies.pop(0) if self._replies else ""
        reply = PlanReply(text=text, model_name=self._model_name, cost=self._cost)
        assert_free(reply.model_name, reply.cost)
        return reply


def _content(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, list):  # content blocks
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return content or ""
