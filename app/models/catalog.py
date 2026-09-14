"""The semantic vocabulary, read from Cube's ``/meta`` at startup.

The agent may only ever ask for what is in here. Metric definitions are
Cube ``description`` strings, so the LLM prompt, the answer footer and
the README all quote the same text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

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
            known = ", ".join(self.values.get(d.short, []))
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
