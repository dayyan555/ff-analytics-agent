"""The data model as the agent knows it, read from Cube's ``/meta`` at startup.

Only views are exposed (Cube's public entry points); the tools in
``app/tools/toolkit.py`` are thin wrappers over this catalog and the Cube
client. Field descriptions are Cube ``description`` strings, so the tool
results, the answer footer and the README all quote the same text.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

INTERNAL = frozenset({"first_date", "last_date"})  # coverage bounds; not shown to the model
VALUE_SAMPLE = 3  # example values per dimension in describe_view
VALUE_CACHE = 1000  # known values cached per dimension at startup (find_dimension_values goes to Cube beyond that)
MIN_SEARCH_SCORE = 0.5  # one generic word out of three ("customer lifetime value") is not a metric match

Kind = Literal["measure", "dimension"]

# How people (and models) commonly write the fields -> extra words the search matches on.
SYNONYMS: dict[str, list[str]] = {
    "cost_per_purchase": ["cpa", "cost per acquisition", "cost per order", "cost per conversion"],
    "roas": ["return on ad spend", "return on advertising spend"],
    "cpc": ["cost per click"],
    "cpm": ["cost per mille", "cost per thousand impressions"],
    "ctr": ["click through rate", "click-through rate"],
    "conversion_rate": ["cvr", "conversion"],
    "aov": ["average order value", "basket size", "order value"],
    "revenue": ["sales", "income"],
    "purchases": ["orders", "conversions", "sales count"],
    "spend": ["cost", "budget", "ad spend", "spent"],
    "campaign_name": ["campaign", "campaigns"],
    "channel": ["platform", "network", "source"],
    "device": ["mobile", "desktop", "phone"],
    "country": ["market", "region", "geo"],
}

# common ways people name a dimension value -> the value the warehouse uses
ALIASES: dict[str, dict[str, str]] = {
    "country": {"germany": "DE", "deutschland": "DE", "uk": "UK", "united kingdom": "UK", "britain": "UK",
                "great britain": "UK", "england": "UK", "us": "US", "usa": "US", "united states": "US", "america": "US"},
    "channel": {"facebook": "meta", "instagram": "meta", "google ads": "google", "adwords": "google",
                "tik tok": "tiktok", "linked in": "linkedin", "e-mail": "email", "newsletter": "email"},
    "device": {"phone": "mobile", "smartphone": "mobile", "computer": "desktop", "pc": "desktop", "laptop": "desktop"},
}

_WORD = re.compile(r"[a-z0-9]+")


def format_name(value: Any) -> str | None:
    """Cube reports a named format as a string ("currency_2") or, for named numeric formats,
    as {"type": "custom-numeric", "value": "$,.2~f", "alias": "currency_2"}."""
    if isinstance(value, dict):
        return value.get("alias") or value.get("type")
    return value


def format_label(fmt: str | None, currency: str | None) -> str | None:
    """"currency USD", "percent", "number" — what the tools tell the model about a field."""
    if not fmt:
        return None
    base = fmt.partition("_")[0]
    return f"{base} {currency}".strip() if base == "currency" else base


@dataclass(frozen=True)
class Member:
    name: str  # "marketing_performance.spend" — what a query must use
    short: str  # "spend"
    title: str
    short_title: str
    kind: Kind
    type: str  # number | string | time
    agg_type: str | None  # sum | number | time | ...
    format: str | None  # currency_2 | number_0 | percent_1 | ...
    currency: str | None
    description: str

    @property
    def view(self) -> str:
        return self.name.split(".", 1)[0]

    def names_text(self) -> str:
        """What the field is called: name, titles, synonyms."""
        return " ".join([self.short.replace("_", " "), self.title, self.short_title, *SYNONYMS.get(self.short, [])]).lower()

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "title": self.short_title}
        if self.kind == "measure":
            out["aggregation"] = "ratio" if self.agg_type == "number" else (self.agg_type or "number")
            if (label := format_label(self.format, self.currency)):
                out["format"] = label
        else:
            out["type"] = self.type
        out["description"] = self.description
        return out


@dataclass
class View:
    name: str
    title: str
    description: str
    measures: dict[str, Member] = field(default_factory=dict)  # short -> member, INTERNAL excluded
    dimensions: dict[str, Member] = field(default_factory=dict)  # string dimensions only
    times: dict[str, Member] = field(default_factory=dict)  # time dimensions (no values; queried via timeDimensions)
    time_dimension: str | None = None  # "marketing_performance.date" — the first time dimension
    values: dict[str, list[str]] = field(default_factory=dict)  # dimension short -> cached known values
    value_counts: dict[str, int] = field(default_factory=dict)  # dimension short -> known count
    values_complete: dict[str, bool] = field(default_factory=dict)  # False when the cache was cut at VALUE_CACHE
    coverage: tuple[date, date] | None = None

    def member(self, name: str) -> Member | None:
        short = name.split(".", 1)[1] if name.startswith(self.name + ".") else name
        return self.measures.get(short) or self.dimensions.get(short) or self.times.get(short)

    def members(self) -> list[Member]:
        return [*self.measures.values(), *self.dimensions.values(), *self.times.values()]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "measures": len(self.measures), "dimensions": len(self.dimensions),
            "data_from": self.coverage[0].isoformat() if self.coverage else None,
            "data_to": self.coverage[1].isoformat() if self.coverage else None,
        }


@dataclass
class Catalog:
    views: dict[str, View] = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------
    def view(self, name: str) -> View | None:
        return self.views.get(name)

    def member(self, name: str) -> Member | None:
        """Fully-qualified lookup: "marketing_performance.spend" -> Member."""
        view_name, _, _short = name.partition(".")
        view = self.views.get(view_name)
        return view.member(name) if view else None

    def members(self) -> list[Member]:
        return [m for v in self.views.values() for m in v.members()]

    # -- search ------------------------------------------------------------
    def search(self, text: str, view: str | None = None, limit: int = 5) -> list[tuple[Member, float]]:
        """Lexical search over name, title, synonyms and description; scores in (0, 1].

        No embeddings and no extra inference: a query word found in the field's name, title
        or synonyms counts 1, one found only in its description counts 0.5; the score is the
        average over the query words, and an exact name/title/synonym match scores 1.
        """
        words = _WORD.findall(text.lower())
        if not words:
            return []
        pool = self.views[view].members() if view and view in self.views else self.members()
        phrase = " ".join(words)
        scored: list[tuple[Member, float]] = []
        for m in pool:
            names, description = m.names_text(), m.description.lower()
            points = sum(1.0 if w in names else 0.5 if w in description else 0.0 for w in words)
            if not points:
                continue
            exact = phrase in (m.short.replace("_", " "), m.short, m.title.lower(), m.short_title.lower(),
                               *SYNONYMS.get(m.short, []))
            score = 1.0 if exact else min(points / len(words), 0.95)
            if score >= MIN_SEARCH_SCORE:
                scored.append((m, round(score, 2)))
        scored.sort(key=lambda t: (-t[1], t[0].name))
        return scored[:limit]

    def suggest(self, name: str, limit: int = 3) -> list[str]:
        """"Did you mean": the closest valid field names to a name Cube rejected."""
        view_name, _, short = name.rpartition(".") if "." in name else ("", "", name)
        pool = self.views[view_name].members() if view_name in self.views else self.members()
        by_short = {m.short: m.name for m in pool}
        close = [by_short[s] for s in difflib.get_close_matches(short, list(by_short), n=limit, cutoff=0.6)]
        close += [n for s, n in by_short.items() if short in s and n not in close]
        return close[:limit]

    def summary(self) -> dict[str, Any]:
        return {"views": [v.summary() for v in self.views.values()]}
