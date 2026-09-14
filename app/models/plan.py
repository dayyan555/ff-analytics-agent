"""The LLM output contract.

The model's only job is to fill in this form. Everything after that
(name validation, date math, the Cube query, the numbers, the prose)
is deterministic code.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

Intent = Literal["query", "compare", "clarify", "unsupported"]


class Filter(BaseModel, extra="forbid"):
    dimension: str  # checked against the catalog's dimensions in build_query
    value: str  # matched case-insensitively against the catalog's known values


class Plan(BaseModel, extra="forbid"):
    intent: Intent
    measures: list[str] = Field(default=[], max_length=10)  # plain str: an unknown name is a catalog rejection
    dimension: str | None = None
    period: str | None = None  # grammar string, see app/agent/dates.py
    compare_period: str | None = None  # grammar string or "previous_period"
    filters: list[Filter] = Field(default=[], max_length=10)
    order_by: str | None = None
    direction: Literal["asc", "desc"] | None = None
    limit: int | None = Field(default=None, ge=1, le=50)  # rows the ranking template renders
    granularity: Literal["day", "week", "month"] | None = None  # set -> unsupported
    message: str | None = None  # never rendered; visible in the UI "Plan" tab and the trace

    @model_validator(mode="after")
    def _consistent(self) -> "Plan":
        if self.intent in ("query", "compare") and not self.measures:
            raise ValueError("measures must contain at least one metric for query/compare intents")
        if self.order_by is not None and self.order_by not in self.measures:
            raise ValueError("order_by must be one of the requested measures")
        return self


PARSER = PydanticOutputParser(pydantic_object=Plan)


def parse_plan(text: str) -> Plan:
    """Parse the model's reply into a Plan.

    Tolerates code fences, prose or ``<think>`` prefixes around the JSON
    object; raises ``OutputParserException`` (the only exception type
    ``PydanticOutputParser`` emits) on anything that is not a valid Plan.
    """
    start, end = text.find("{"), text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end > start else text
    if not candidate.strip():
        raise OutputParserException("empty reply", llm_output=text)
    return PARSER.parse(candidate)
