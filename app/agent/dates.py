"""Period grammar: the few tokens the LLM may emit, resolved to absolute inclusive dates.

Resolution is anchored to a pinned ``as_of`` date (``AGENT_AS_OF_DATE``), not the
wall clock, so the demo data stays answerable and answers are reproducible.
Cube's own relative strings ("last month") are deliberately not used for the
same reason.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Literal

MAX_SPAN_DAYS = 366
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_QUARTER = re.compile(r"^(\d{4})-q([1-4])$")
_LAST_MONTHS = re.compile(r"^last_(\d{1,2})_months$")
_RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")

Overlap = Literal["none", "partial", "full"]


class PeriodError(ValueError):
    """The token is not part of the grammar or describes an invalid range."""


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str
    coverage_note: str | None = None

    @property
    def is_calendar_month(self) -> bool:
        return self.start.day == 1 and self.end == _month_end(self.start)

    @property
    def is_calendar_quarter(self) -> bool:
        return (self.start.day == 1 and self.start.month in (1, 4, 7, 10)
                and self.end == _month_end(date(self.start.year, self.start.month + 2, 1)))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = self.start.isoformat(), self.end.isoformat()
        return d


def _month_end(d: date) -> date:
    nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    return start, _month_end(start)


def resolve(token: str | None, as_of: date, coverage: tuple[date, date], *, base: Period | None = None) -> Period:
    """Turn a grammar token into a Period. Raises PeriodError for anything else."""
    if not token:
        raise PeriodError("no period given")
    tok = token.strip().lower()
    try:
        period = _resolve_token(tok, as_of, coverage, base)
    except PeriodError:
        raise
    except (OverflowError, ValueError) as exc:  # date arithmetic outside the calendar
        raise PeriodError("invalid period") from exc
    if period.start > period.end:
        raise PeriodError("period start is after its end")
    if (period.end - period.start).days + 1 > MAX_SPAN_DAYS:
        raise PeriodError("period longer than a year")
    if period.start > as_of:
        raise PeriodError("period starts in the future")
    return _with_coverage_note(period, coverage)


def _resolve_token(tok: str, as_of: date, coverage: tuple[date, date], base: Period | None) -> Period:
    if tok == "last_month":
        prev = as_of.replace(day=1) - timedelta(days=1)
        start, end = _month_bounds(prev.year, prev.month)
        period = Period(start, end, f"last month ({start:%B %Y})")
    elif m := _LAST_MONTHS.match(tok):  # the N complete calendar months before today
        n = int(m.group(1))
        if not 1 <= n <= 12:
            raise PeriodError("last_N_months supports 1 to 12 months")
        end = as_of.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        for _ in range(n - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        if n == 1:
            label = f"last month ({end:%B %Y})"
        elif start.year != end.year:
            label = f"last {n} months ({start:%B %Y} to {end:%B %Y})"
        else:
            label = f"last {n} months ({start:%B} to {end:%B %Y})"
        period = Period(start, end, label)
    elif m := _QUARTER.match(tok):
        year, q = int(m.group(1)), int(m.group(2))
        start = date(year, 3 * q - 2, 1)
        end = _month_end(date(year, 3 * q, 1))
        period = Period(start, end, f"Q{q} {year}")
    elif tok == "all_time":
        period = Period(coverage[0], coverage[1], "all available data")
    elif tok == "previous_period":
        if base is None:
            raise PeriodError("previous_period needs a base period")
        if base.is_calendar_month:
            prev = base.start - timedelta(days=1)
            start, end = _month_bounds(prev.year, prev.month)
        elif base.is_calendar_quarter:
            prev = base.start - timedelta(days=1)  # last day of the previous quarter
            start = date(prev.year, 3 * ((prev.month - 1) // 3) + 1, 1)
            end = prev
        else:
            length = (base.end - base.start).days + 1
            end = base.start - timedelta(days=1)
            start = end - timedelta(days=length - 1)
        period = Period(start, end, f"previous period ({start} to {end})")
    elif m := _MONTH.match(tok):
        start, end = _month_bounds(int(m.group(1)), int(m.group(2)))
        period = Period(start, end, f"{start:%B %Y}")
    elif m := _RANGE.match(tok):
        start, end = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        period = Period(start, end, f"{start} to {end}")
    else:
        raise PeriodError("unknown period token")  # never echo the model's text
    return period


def overlap(period: Period, coverage: tuple[date, date]) -> Overlap:
    if period.end < coverage[0] or period.start > coverage[1]:
        return "none"
    if period.start >= coverage[0] and period.end <= coverage[1]:
        return "full"
    return "partial"


def _with_coverage_note(period: Period, coverage: tuple[date, date]) -> Period:
    if overlap(period, coverage) != "partial":
        return period
    lo, hi = max(period.start, coverage[0]), min(period.end, coverage[1])
    return Period(period.start, period.end, period.label, f"data covers only {lo} to {hi} within this range")


def periods_overlap(a: Period, b: Period) -> bool:
    return a.start <= b.end and b.start <= a.end
