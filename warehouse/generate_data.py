"""Deterministic fake marketing data for the ClickHouse warehouse.

`build_rows(seed)` returns rows for the three tables in `schema.sql`, in the
exact column order of the DDL. Values are Python `date`, `datetime` and
`Decimal`. `purchased_at` is a timezone-aware UTC `datetime`: clickhouse-connect
interprets naive datetimes in the *local* zone on insert, which would shift rows
across day boundaries.

The window is 2026-03-01..2026-08-31 (184 days). Every campaign-day produces one
`ad_spend` row per device (mobile, desktop) for paid campaigns, and purchases are
scaled by a per-month seasonality multiplier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

CENTS = Decimal("0.01")
DEVICES: tuple[str, ...] = ("mobile", "desktop")  # fixed order: keeps the RNG stream stable
SEASONALITY: dict[int, float] = {3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 7: 1.2, 8: 1.1}  # month -> purchase multiplier

MAR_1, AUG_31 = date(2026, 3, 1), date(2026, 8, 31)


def _monthly(value: int, months: range = range(3, 9)) -> dict[int, int]:
    """Same daily mean for every month in the window (Mar..Aug)."""
    return {m: value for m in months}


@dataclass(frozen=True)
class Campaign:
    """One planted campaign: identity, active window and its daily means."""

    id: int
    name: str
    channel: str
    country: str
    objective: str
    start: date
    end: date
    spend_per_day: dict[int, int] | None  # month number -> mean USD; None = no paid media
    cpm: float | None  # USD per 1,000 impressions
    ctr: float | None  # clicks / impressions
    purchases_per_day: dict[int, int]  # month number -> mean purchases (before seasonality)
    aov: int  # mean order value, USD
    mobile_share: float  # mobile share of spend; also the default mobile weight for purchases
    mobile_purchase_share: float | None = None  # override the purchase device weight (Shopping DE)

    def spend_mean(self, month: int, device: str) -> float:
        share = self.mobile_share if device == "mobile" else 1 - self.mobile_share
        return self.spend_per_day[month] * share

    def purchase_mean(self, month: int) -> float:
        return self.purchases_per_day[month] * SEASONALITY[month]

    @property
    def mobile_purchase_weight(self) -> float:
        return self.mobile_share if self.mobile_purchase_share is None else self.mobile_purchase_share


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign(1, "Brand Search US", "google", "US", "conversion", MAR_1, AUG_31, _monthly(150), 12, 0.04, _monthly(9), 60, 0.6),
    Campaign(2, "Generic Search US", "google", "US", "conversion", MAR_1, AUG_31, {3: 110, 4: 110, 5: 110, 6: 110, 7: 155, 8: 155}, 10, 0.025, _monthly(4), 55, 0.6),
    Campaign(3, "Shopping DE", "google", "DE", "conversion", MAR_1, AUG_31, _monthly(120), 10, 0.025, _monthly(6), 75, 0.55, mobile_purchase_share=0.35),
    Campaign(4, "Prospecting Video", "meta", "US", "awareness", MAR_1, AUG_31, _monthly(400), 8, 0.01, _monthly(2), 110, 0.8),
    Campaign(5, "Retargeting Carousel", "meta", "US", "conversion", MAR_1, date(2026, 8, 15), _monthly(90), 15, 0.05, _monthly(6), 85, 0.75),
    Campaign(6, "Summer Sale", "meta", "UK", "conversion", date(2026, 7, 10), date(2026, 7, 31), _monthly(200), 9, 0.03, _monthly(5), 70, 0.75),
    Campaign(7, "TikTok Spark UK", "tiktok", "UK", "awareness", MAR_1, AUG_31, _monthly(180), 6, 0.015, {3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 1}, 50, 0.95),
    Campaign(8, "TikTok Creator DE", "tiktok", "DE", "conversion", date(2026, 6, 1), AUG_31, _monthly(130), 6, 0.015, _monthly(4), 65, 0.95),
    Campaign(9, "LinkedIn Leads UK", "linkedin", "UK", "conversion", MAR_1, AUG_31, _monthly(160), 40, 0.008, _monthly(2), 320, 0.5),
    Campaign(10, "LinkedIn Thought Leadership", "linkedin", "US", "awareness", MAR_1, date(2026, 5, 31), _monthly(140), 40, 0.008, _monthly(1), 250, 0.5),
    # Email campaigns have no ad_spend rows; mobile_share only weights the purchase device.
    Campaign(11, "Winback Series", "email", "US", "retention", MAR_1, AUG_31, None, None, None, _monthly(4), 45, 0.6),
    Campaign(12, "Newsletter Promo", "email", "DE", "retention", MAR_1, AUG_31, None, None, None, _monthly(5), 40, 0.6),
)

CAMPAIGN_COLUMNS = ["campaign_id", "campaign_name", "channel", "country", "objective", "start_date", "end_date"]
AD_SPEND_COLUMNS = ["date", "campaign_id", "device", "spend", "impressions", "clicks"]
PURCHASE_COLUMNS = ["purchase_id", "purchased_at", "campaign_id", "device", "revenue"]


def _days(start: date, end: date):
    """Yield every date from start to end, inclusive."""
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def _money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(CENTS)


def build_rows(seed: int = 42) -> dict[str, list[tuple]]:
    """Build campaigns / ad_spend / purchases rows from one seeded RNG."""
    rng = random.Random(seed)
    campaigns: list[tuple] = []
    ad_spend: list[tuple] = []
    purchases: list[tuple] = []
    purchase_id = 0

    for c in sorted(CAMPAIGNS, key=lambda c: c.id):
        campaigns.append((c.id, c.name, c.channel, c.country, c.objective, c.start, c.end))
        for day in _days(c.start, c.end):
            if c.spend_per_day is not None:
                for device in DEVICES:
                    spend = _money(c.spend_mean(day.month, device) * rng.uniform(0.85, 1.15))
                    impressions = round(float(spend) / c.cpm * 1000)
                    clicks = round(impressions * c.ctr)
                    ad_spend.append((day, c.id, device, spend, impressions, clicks))
            for _ in range(round(c.purchase_mean(day.month) * rng.uniform(0.7, 1.3))):
                purchase_id += 1
                device = "mobile" if rng.random() < c.mobile_purchase_weight else "desktop"
                purchased_at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(seconds=rng.randrange(86_400))
                revenue = _money(c.aov * rng.uniform(0.7, 1.4))
                purchases.append((purchase_id, purchased_at, c.id, device, revenue))

    return {"campaigns": campaigns, "ad_spend": ad_spend, "purchases": purchases}


def _summary(rows: dict[str, list[tuple]]) -> str:
    """Row counts plus a few aggregates, for eyeballing the data story."""
    by_id = {c.id: c for c in CAMPAIGNS}
    spend_by_channel: dict[str, Decimal] = {c.channel: Decimal(0) for c in CAMPAIGNS}
    for day, cid, device, spend, *_ in rows["ad_spend"]:
        channel = by_id[cid].channel
        spend_by_channel[channel] = spend_by_channel.get(channel, Decimal(0)) + spend
    purchases_by_campaign: dict[str, int] = {}
    purchases_by_device: dict[str, int] = {}
    for _, _, cid, device, _ in rows["purchases"]:
        name = by_id[cid].name
        purchases_by_campaign[name] = purchases_by_campaign.get(name, 0) + 1
        purchases_by_device[device] = purchases_by_device.get(device, 0) + 1

    lines = [f"{table}: {len(data)} rows" for table, data in rows.items()]
    lines.append("spend by channel: " + ", ".join(f"{k}={v}" for k, v in sorted(spend_by_channel.items())))
    lines.append("purchases by campaign: " + ", ".join(f"{k}={v}" for k, v in purchases_by_campaign.items()))
    lines.append("purchases by device: " + ", ".join(f"{k}={v}" for k, v in sorted(purchases_by_device.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary(build_rows()))
