"""The example questions shared by ``GET /examples`` and ``examples.py``."""

from __future__ import annotations

from typing import NamedTuple


class Example(NamedTuple):
    question: str
    kind: str  # normal | compare | unsupported | ambiguous | empty
    expect: str  # the expected outcome: answer | unsupported | clarify | no_data


EXAMPLE_QUESTIONS: list[Example] = [
    Example("How much did we spend by channel in August 2026?", "normal", "answer"),
    Example("Which campaign generated the most purchases?", "normal", "answer"),
    Example("Which campaign had the strongest result relative to spend last month?", "normal", "answer"),
    Example("Which device had the better ROAS in Germany over the last 3 months?", "normal", "answer"),
    Example("What was the average order value by objective in Q2 2026?", "normal", "answer"),
    Example("What changed between July and August by channel?", "compare", "answer"),
    Example("Which campaign had the best profit margin?", "unsupported", "unsupported"),
    Example("How did we do recently?", "ambiguous", "clarify"),
    Example("How many purchases did Summer Sale get in August 2026?", "empty", "no_data"),
]
