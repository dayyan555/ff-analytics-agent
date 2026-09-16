"""The test catalog must mirror the Cube model, or the offline suite proves nothing about production."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.models.catalog import INTERNAL
from tests.conftest import DIMENSIONS, MEASURES

MODEL = Path(__file__).resolve().parents[1] / "cube" / "model"


def _yaml(path: str) -> dict:
    return yaml.safe_load((MODEL / path).read_text())


def test_conftest_measures_mirror_the_cube_model():
    measures = {m["name"]: m for m in _yaml("cubes/campaign_daily.yml")["cubes"][0]["measures"]}
    assert set(measures) - INTERNAL == {m[0] for m in MEASURES}
    for short, _agg, fmt, currency, _title, description in MEASURES:
        assert measures[short]["format"] == fmt, short
        assert measures[short].get("currency") == currency, short
        assert measures[short]["description"] == description, short


def test_conftest_dimensions_and_ratios_mirror_the_cube_model():
    campaigns = {d["name"] for d in _yaml("cubes/campaigns.yml")["cubes"][0]["dimensions"] if d.get("public", True)}
    daily = {d["name"] for d in _yaml("cubes/campaign_daily.yml")["cubes"][0]["dimensions"] if d.get("public", True)}
    assert (campaigns | daily) - {"date"} == {d[0] for d in DIMENSIONS}
    view = _yaml("views/marketing_performance.yml")["views"][0]
    included = {name for c in view["cubes"] for name in c["includes"]}
    ratios = {m[0] for m in MEASURES if m[1] == "number"}
    assert ratios <= included and INTERNAL <= included
    ratio_types = {m["name"]: m["type"] for m in _yaml("cubes/campaign_daily.yml")["cubes"][0]["measures"]}
    assert all(ratio_types[r] == "number" for r in ratios)
