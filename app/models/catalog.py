"""The semantic vocabulary, read from Cube's ``/meta`` at startup.

The agent may only ever ask for what is in here. Metric definitions are
Cube ``description`` strings, so the LLM prompt, the answer footer and
the README all quote the same text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

VIEW = "marketing_performance"
INTERNAL = frozenset({"first_date", "last_date"})  # coverage bounds; hidden from the LLM

# ratio -> (numerator, denominator, factor): ratio == numerator / denominator * factor
RATIOS: dict[str, tuple[str, str, int]] = {
    "cost_per_purchase": ("spend", "purchases", 1),
    "roas": ("revenue", "spend", 1),
    "cpc": ("spend", "clicks", 1),
    "cpm": ("spend", "impressions", 1000),
    "ctr": ("clicks", "impressions", 1),
    "conversion_rate": ("purchases", "clicks", 1),
    "aov": ("revenue", "purchases", 1),
}
COST_LIKE = frozenset({"cost_per_purchase", "cpc", "cpm"})  # "best" means lowest

# How people (and models) commonly write the metrics and dimensions -> the catalog's names.
# Matching is otherwise case-insensitive with spaces/hyphens read as underscores; no fuzzy matching.
NAME_SYNONYMS: dict[str, str] = {
    "cost per purchase": "cost_per_purchase", "cpa": "cost_per_purchase", "cost per acquisition": "cost_per_purchase",
    "return on ad spend": "roas", "cost per click": "cpc", "cost per mille": "cpm", "cost per thousand": "cpm",
    "click through rate": "ctr", "click-through rate": "ctr", "conversion rate": "conversion_rate", "cvr": "conversion_rate",
    "average order value": "aov", "order value": "aov", "sales": "revenue", "orders": "purchases",
    "campaign": "campaign_name", "campaigns": "campaign_name", "channels": "channel", "countries": "country",
    "devices": "device", "objectives": "objective",
}

# Names and values the warehouse uses never contain digits; anything else the model
# invents is not echoed back (it could carry made-up numbers or instructions).
_SAFE_NAME = re.compile(r"(?!.*\d)[A-Za-z_][A-Za-z_ .\-]{0,39}")
_PROMPT_VALUE = re.compile(r"[A-Za-z0-9 _.&'/\-]{1,60}")  # dimension values allowed into the system prompt


def canonical_name(name: str) -> str:
    """Normalise a model-supplied metric/dimension name: case, spaces, hyphens, known synonyms."""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    key = NAME_SYNONYMS.get(key, key)
    return re.sub(r"[\s\-]+", "_", key)


def safe_name(text: Any) -> str:
    """A model-supplied name, echoed only if it looks like one; otherwise a placeholder."""
    text = str(text)
    return text if _SAFE_NAME.fullmatch(text) else "(unreadable name)"


def format_name(value: Any) -> str | None:
    """Cube reports a named format as a string ("currency_2") or, for named numeric formats,
    as {"type": "custom-numeric", "value": "$,.2~f", "alias": "currency_2"}."""
    if isinstance(value, dict):
        return value.get("alias") or value.get("type")
    return value

# common ways people name a dimension value -> the value the warehouse uses
ALIASES: dict[str, dict[str, str]] = {
    "country": {"germany": "DE", "deutschland": "DE", "uk": "UK", "united kingdom": "UK", "britain": "UK",
                "great britain": "UK", "england": "UK", "us": "US", "usa": "US", "united states": "US", "america": "US"},
    "channel": {"facebook": "meta", "instagram": "meta", "google ads": "google", "adwords": "google",
                "tik tok": "tiktok", "linked in": "linkedin", "e-mail": "email", "newsletter": "email"},
    "device": {"phone": "mobile", "smartphone": "mobile", "computer": "desktop", "pc": "desktop", "laptop": "desktop"},
}


@dataclass(frozen=True)
class Member:
    name: str  # "marketing_performance.spend"
    short: str  # "spend"
    title: str
    short_title: str
    type: str  # number | string | time
    agg_type: str | None  # sum | number | time | ...
    format: str | None  # currency_2 | number_0 | percent_1 | ...
    currency: str | None
    description: str


@dataclass
class Catalog:
    measures: dict[str, Member] = field(default_factory=dict)  # short -> member, INTERNAL excluded
    dimensions: dict[str, Member] = field(default_factory=dict)  # string dimensions: channel, campaign_name, ...
    values: dict[str, list[str]] = field(default_factory=dict)  # dimension -> known values (for filters)
    time_dimension: str = f"{VIEW}.date"
    coverage: tuple[date, date] | None = None

    def member(self, short: str) -> str:
        return f"{VIEW}.{short}"

    def match_measure(self, name: str) -> str | None:
        key = canonical_name(name)
        return key if key in self.measures else None

    def match_dimension(self, name: str) -> str | None:
        key = canonical_name(name)
        return key if key in self.dimensions else None

    def match_value(self, dimension: str, value: str) -> str | None:
        """Return the canonical value of ``dimension`` for a case-insensitive match (or a known alias), or None."""
        wanted = value.strip().lower()
        wanted = ALIASES.get(dimension, {}).get(wanted, wanted).lower()
        return next((v for v in self.values.get(dimension, []) if v.lower() == wanted), None)

    def vocabulary_text(self) -> str:
        lines = ["Metrics (measures):"]
        lines += [f"- {m.short}: {m.description}" for m in self.measures.values()]
        lines.append("Dimensions (group by, or filter to one value):")
        for d in self.dimensions.values():
            # warehouse values are data, not instructions: only plain, short values reach the prompt
            known = ", ".join(v for v in self.values.get(d.short, []) if _PROMPT_VALUE.fullmatch(v))
            lines.append(f"- {d.short}: {d.description}" + (f" Values: {known}." if known else ""))
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "measures": list(self.measures),
            "dimensions": list(self.dimensions),
            "values": self.values,
            "coverage": {"first": self.coverage[0].isoformat(), "last": self.coverage[1].isoformat()}
            if self.coverage
            else None,
        }
