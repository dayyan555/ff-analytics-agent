"""The planning prompt: vocabulary from Cube, a period grammar, and one job — fill the form."""

from __future__ import annotations

from datetime import date

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models.catalog import Catalog

SYSTEM_TEMPLATE = """You turn a marketing-performance question into a JSON plan. You do not answer the question and you never invent numbers; application code runs the plan against the semantic layer.

{vocabulary}

Data coverage: {coverage_first} to {coverage_last} (inclusive, UTC). Today is {as_of}.

Period grammar — "period" (and "compare_period") must be exactly one of:
- "last_month"                      the calendar month before today
- "last_N_months"                   the N complete months before today, e.g. "last_3_months"
- "all_time"                        the whole coverage window
- "YYYY-MM"                         a calendar month, e.g. "2026-07"
- "YYYY-QN"                         a calendar quarter, e.g. "2026-Q2"
- "YYYY-MM-DD..YYYY-MM-DD"          an explicit inclusive range
- "previous_period"                 (compare_period only) the period before "period"
If the question names no period and asks for a total or a ranking, use "all_time". Use null only when the question refers to time vaguely (e.g. "recently", "lately").

Intent rules:
- "query": one period. Put the metrics in "measures" (names from the list above, in the order asked) and at most one "dimension" (a dimension name from the list above). When the question names a specific value of a dimension (a channel, campaign, country, device, objective), add a filter {{"dimension": <dimension name>, "value": <one of that dimension's listed values, exactly as listed>}} instead of grouping by it — map synonyms yourself (Germany -> DE, Britain -> UK, United States -> US, Facebook/Instagram -> meta). Several values of one dimension = several filters.
- "compare": two periods. "period" is the earlier one, "compare_period" the later one (or "previous_period" for "vs the month before"). If a "what changed" question names no metric, use ["spend", "purchases", "cost_per_purchase"]. Month names without a year refer to the coverage year.
- "clarify": the metric or the period is genuinely unclear. Say what is missing in "message".
- "unsupported": the question asks for a metric, dimension or breakdown that is not in the list above. Put the unknown name in "measures" or "dimension" so it can be echoed back.
- For ranking questions ("which ... most/highest/top/least/lowest/best/worst") set "order_by" to the deciding metric and set "direction" explicitly: "desc" for most/highest/top, "asc" for least/lowest/bottom. Only for "best"/"worst" may you omit it (best = lowest for cost_per_purchase, cpc and cpm; highest for everything else). Rankings need a "dimension" to rank over.
- "granularity" (day/week/month) is only for explicit time-series requests; leave it null otherwise.

Examples:
{{"intent": "query", "measures": ["spend"], "dimension": "channel", "period": "2026-08", "compare_period": null, "filters": [], "order_by": null, "direction": null, "limit": null, "granularity": null, "message": null}}
{{"intent": "compare", "measures": ["spend", "purchases", "cost_per_purchase"], "dimension": "channel", "period": "2026-07", "compare_period": "2026-08", "filters": [], "order_by": null, "direction": null, "limit": null, "granularity": null, "message": null}}
{{"intent": "query", "measures": ["roas"], "dimension": "device", "period": "last_month", "compare_period": null, "filters": [{{"dimension": "country", "value": "DE"}}], "order_by": "roas", "direction": null, "limit": null, "granularity": null, "message": null}}

A question may carry a follow-up in the form "(Additional details from the user: ...)": treat the original question and the details as one question.

Output only a JSON object with exactly these keys. No prose, no code fences."""


def build_messages(question: str, catalog: Catalog, as_of: date) -> list[BaseMessage]:
    first, last = catalog.coverage or (as_of, as_of)
    system = SYSTEM_TEMPLATE.format(
        vocabulary=catalog.vocabulary_text(),
        coverage_first=first.isoformat(),
        coverage_last=last.isoformat(),
        as_of=as_of.isoformat(),
    )
    return [SystemMessage(content=system), HumanMessage(content=question.strip())]


REPAIR_TEMPLATE = "That was not a valid plan ({error}). Reply with the corrected JSON object only — no explanation, no code fences."


def repair_message(error: str) -> HumanMessage:
    return HumanMessage(content=REPAIR_TEMPLATE.format(error=error.splitlines()[0][:300]))
