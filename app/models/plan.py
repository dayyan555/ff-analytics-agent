"""The LLM output contract.

The model's only job is to fill in this form. Everything after that
(name validation, date math, the Cube query, the numbers, the prose)
is deterministic code.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Intent = Literal["query", "compare", "clarify", "unsupported"]


class Filter(BaseModel, extra="forbid"):
    dimension: str  # checked against the catalog's dimensions in build_query
    value: str  # matched case-insensitively against the catalog's known values


class Plan(BaseModel, extra="forbid"):
    intent: Intent
    measures: list[str] = Field(default=[], max_length=12)  # plain str: an unknown name is a catalog rejection
    dimension: str | None = None
    period: str | None = None  # grammar string, see app/agent/dates.py
    compare_period: str | None = None  # grammar string or "previous_period"
    filters: list[Filter] = Field(default=[], max_length=10)
    order_by: str | None = None
    direction: Literal["asc", "desc"] | None = None
    limit: int | None = Field(default=None, ge=1, le=50)  # rows the ranking template renders
    granularity: Literal["day", "week", "month"] | None = None  # set -> unsupported
    message: str | None = None  # never rendered; visible in the UI "Plan" tab and the trace

    @field_validator("measures", "filters", mode="before")
    @classmethod
    def _null_is_empty(cls, value: Any) -> Any:  # weak models write null where the schema says []
        return [] if value is None else value

    @model_validator(mode="after")
    def _consistent(self) -> "Plan":
        if self.intent in ("query", "compare") and not self.measures:
            raise ValueError("measures must contain at least one metric for query/compare intents")
        if self.order_by is not None and self.order_by not in self.measures:
            raise ValueError("order_by must be one of the requested measures")
        return self


def strict_schema() -> dict:
    """The Plan as an OpenAI-style strict JSON schema: every key required (nullable where
    optional) and no extra keys. Sent as ``response_format`` so models that honour it return
    plain JSON; the free router does not reliably filter on it, so the parser and the
    repair/re-roll loop in ``interpret`` remain the real safeguard."""
    schema = Plan.model_json_schema()

    def tighten(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)

    tighten(schema)
    return schema


PLAN_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "plan", "strict": True, "schema": strict_schema()}}


_THINK = re.compile(r"<think>.*?</think>", re.S)
_DECODER = json.JSONDecoder()


def parse_plan(text: str) -> Plan:
    """Parse the model's reply into a Plan.

    Tolerates ``<think>`` blocks, code fences, prose before or after the JSON
    and stray trailing braces: the first complete JSON object in the text is
    decoded and validated. Raises ``OutputParserException`` otherwise.
    """
    cleaned = _THINK.sub("", text or "")
    start = cleaned.find("{")
    if start == -1:
        raise OutputParserException("no JSON object in the reply", llm_output=text)
    try:
        data, _ = _DECODER.raw_decode(cleaned, start)
    except json.JSONDecodeError as exc:
        raise OutputParserException(f"invalid JSON: {exc.msg} at position {exc.pos}", llm_output=text) from exc
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'plan'}: {e['msg']}" for e in exc.errors()[:4])
        raise OutputParserException(f"plan does not match the schema: {problems}", llm_output=text) from exc
