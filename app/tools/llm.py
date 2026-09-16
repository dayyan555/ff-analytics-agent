"""OpenRouter free-router access, with the zero-paid-inference guard.

Two things are enforced in code rather than trusted from configuration:
the request never retries silently (the SDK's default is up to an hour of
5xx backoff), and every reply must come from a ``:free`` model at cost 0.

Tool calling: the tool schemas are attached natively (``bind_tools``). The
free router picks a random model per request and not every free model can
route tool requests; when the router says so (a 404 mentioning tool use),
this process switches to the JSON tool protocol described in the prompt
(``{"tool": ..., "args": ...}`` in plain text), which any model can follow.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from openrouter.errors import OpenRouterError

from app.models.state import LLMReply
from app.tools.toolkit import json_protocol_text

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


def _rate_limit_headers(exc: OpenRouterError) -> dict[str, str]:
    """The HTTP headers, plus the X-RateLimit-* ones OpenRouter puts in the 429 body (error.metadata.headers)."""
    headers = {k.lower(): str(v) for k, v in dict(getattr(exc, "headers", None) or {}).items()}
    metadata = getattr(getattr(getattr(exc, "data", None), "error", None), "metadata", None) or {}
    body_headers = metadata.get("headers") if isinstance(metadata, dict) else None
    if isinstance(body_headers, dict):
        headers.update({k.lower(): str(v) for k, v in body_headers.items()})
    return headers


def _retry_after_seconds(headers: dict[str, str]) -> float | None:
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
    """Map any failure of the model call to a typed LLMError (never a paid fallback)."""
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


_TOOLS_UNSUPPORTED = re.compile(r"tool", re.I)


class OpenRouterLLM:
    """Production ``AgentLLM``: one request per call, one explicit 429 retry, nothing else."""

    def __init__(self, api_key: str, app_title: str, app_url: str):
        self.calls = 0
        self.native_tools = True
        self._llm = ChatOpenRouter(
            model=MODEL_ID,
            api_key=api_key,
            temperature=0,
            max_tokens=4000,  # reasoning models spend tokens before the answer; free, so leave room
            timeout=90_000,  # milliseconds; some free models are slow
            max_retries=0,  # no LangChain-side retry config ...
            model_kwargs={"retries": None},  # ... and none from the OpenRouter SDK either
            reasoning={"effort": "low"},
            openrouter_provider={"max_price": {"prompt": "0", "completion": "0"}},  # strings: the SDK drops ints
            app_title=app_title,
            app_url=app_url,
        )

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool], config: RunnableConfig | None = None) -> LLMReply:
        try:
            msg = self._invoke(messages, tools, config)
        except OpenRouterError as exc:
            if exc.status_code == 404 and self.native_tools and tools and _TOOLS_UNSUPPORTED.search(exc.message or exc.body or ""):
                self.native_tools = False  # this router cannot route tool requests: JSON protocol from now on
                return self.invoke(with_json_protocol(messages, tools), tools, config)
            if exc.status_code != 429:
                raise _classify(exc) from exc
            wait = _retry_after_seconds(_rate_limit_headers(exc))
            if wait is None:
                raise LLMError("rate_limited", 429, "rate limit hint exceeds 60 s; not retrying") from exc
            time.sleep(wait)
            try:
                msg = self._invoke(messages, tools, config)
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
        reply = LLMReply(message=msg, model_name=meta.get("model_name"), cost=cost)
        assert_free(reply.model_name, reply.cost)
        return reply

    def _invoke(self, messages: list[BaseMessage], tools: list[BaseTool], config: RunnableConfig | None) -> AIMessage:
        self.calls += 1
        runnable = self._llm.bind_tools(tools) if (tools and self.native_tools) else self._llm
        return runnable.invoke(messages, config=config)


def with_json_protocol(messages: list[BaseMessage], tools: list[BaseTool]) -> list[BaseMessage]:
    """The same conversation with the JSON tool protocol appended to the system message (once)."""
    if not messages or not isinstance(messages[0], SystemMessage) or "Tool protocol:" in str(messages[0].content):
        return messages
    system = SystemMessage(content=str(messages[0].content) + json_protocol_text(tools), id=messages[0].id)
    return [system, *messages[1:]]


class FakeLLM:
    """Offline ``AgentLLM`` for the tests: scripted replies (AIMessages or plain text), or a raised exception."""

    def __init__(
        self,
        replies: list[AIMessage | str] | None = None,
        *,
        model_name: str = "fake/model:free",
        cost: Decimal | None = Decimal(0),
        raise_: Exception | None = None,
        native_tools: bool = True,
    ):
        self._replies = list(replies or [])
        self._model_name = model_name
        self._cost = cost
        self._raise = raise_
        self.native_tools = native_tools
        self.calls = 0
        self.seen: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool], config: RunnableConfig | None = None) -> LLMReply:
        self.calls += 1
        self.seen.append(list(messages))
        if self._raise is not None:
            raise self._raise
        reply = self._replies.pop(0) if self._replies else ""
        msg = reply if isinstance(reply, AIMessage) else AIMessage(content=reply)
        out = LLMReply(message=msg, model_name=self._model_name, cost=self._cost)
        assert_free(out.model_name, out.cost)
        return out


def content_text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, list):  # content blocks
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return content or ""
