"""Agent behaviour with the real model: for every case in ``evals/cases.json`` run the whole
graph live and check scope, comparison periods, ranking direction, golden values
and the numerical consistency result. Passing does not prove the narrative is correct.

    make eval          # prints a table and a summary; writes evals/last_run.json

Cost: usually 3–4 free-router requests per data question, with explicit retries
counted separately. The full suite may exceed the daily quota; --smoke runs four cases. The cases are deliberately different from the UI's
example questions so the score is not inflated by memorisation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import sys
from pathlib import Path

from app import config  # noqa: F401  (loads .env)
from app.config import settings
from app.models.state import Deps
from app.runtime import build_cube, build_llm, lf, run_question
from app.tools.cube import load_catalog
from app.agent.verify import relevant_queries

CASES = Path(__file__).with_name("cases.json")
RESULTS = Path(__file__).with_name("last_run.json")


def _short(name: str) -> str:
    return name.split(".", 1)[-1]


def check(out: dict, expect: dict) -> list[str]:
    """Check scope, comparison coverage, ordering and golden values independently of prose."""
    problems: list[str] = []
    wanted = expect["outcome"] if isinstance(expect["outcome"], list) else [expect["outcome"]]
    if out.get("outcome") not in wanted:
        return [f"outcome {out.get('outcome')} (wanted {'/'.join(wanted)}: {out.get('error') or ''})"]
    answer = str(out.get("answer_body") or out.get("answer") or "").lower()
    for phrase in expect.get("answer_contains", []):
        if phrase.lower() not in answer:
            problems.append(f"answer does not disclose {phrase!r}")
    ok = relevant_queries(out.get("queries", []))
    if not expect.get("measures"):
        return problems
    if not ok:
        return ["no successful query"]

    expected_periods = {tuple(p) for p in expect.get("periods", [expect["period"]] if expect.get("period") else [])}
    # A comparison may be one compareDateRange call or one call per period.
    selected = ok if len(expected_periods) > 1 else ok[-1:]
    periods = {(p["from"], p["to"]) for q in selected for p in q.get("periods", [])}
    if periods != expected_periods:
        problems.append(f"periods {sorted(periods)} (wanted {sorted(expected_periods)})")
    for result in selected:
        query = result["query"]
        measures = {_short(m) for m in query.get("measures", [])}
        time_members = {_short(td.get("dimension", "")) for td in query.get("timeDimensions", [])}
        dimensions = {_short(d) for d in query.get("dimensions", [])} - time_members
        if not set(expect["measures"]) <= measures:
            problems.append(f"measures {sorted(measures)} do not include {expect['measures']}")
        if not expect.get("no_data") and dimensions != set(expect.get("dimensions", [])):
            problems.append(f"dimensions {sorted(dimensions)} (wanted {expect.get('dimensions', [])})")
        filters = {(_short(f.get("member", "")), f.get("operator"), tuple(str(v).lower() for v in f.get("values", [])))
                   for f in query.get("filters", [])}
        expected_filters = {(k, "equals", (v.lower(),)) for k, v in expect.get("filters", {}).items()}
        allowed_filters = expected_filters | {(k, "equals", (v.lower(),)) for k, v in expect.get("allowed_filters", {}).items()}
        if not expected_filters <= filters or not filters <= allowed_filters:
            problems.append(f"filters {filters} (required {expected_filters}; allowed {allowed_filters})")
        grains = {td["granularity"] for td in query.get("timeDimensions", []) if td.get("granularity")}
        if grains != ({expect["granularity"]} if expect.get("granularity") else set()):
            problems.append(f"granularity {grains} (wanted {expect.get('granularity')})")
        if expect.get("order"):
            order = query.get("order", {})
            ordered = list(order.items()) if isinstance(order, dict) else order
            first = (_short(ordered[0][0]), str(ordered[0][1]).lower()) if ordered else None
            if first != tuple(expect["order"]):
                problems.append(f"primary order {first} (wanted {expect['order']})")
    if expect.get("no_data") and any(q.get("has_data") is not False for q in selected):
        problems.append("expected an explicit empty result")
    rows = [r for q in selected for r in q.get("rows", [])]
    for expected_row in expect.get("rows", []):
        if not any(_row_matches(r, expected_row) for r in rows):
            problems.append(f"golden row missing or incorrect: {expected_row}")
    verification = out.get("verification")
    if not verification or verification.get("unverified"):
        problems.append("numerical consistency check missing or failed")
    return problems


def _row_matches(actual: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        try:
            left, right = Decimal(str(actual[key])), Decimal(str(value))
        except InvalidOperation:
            if actual[key] != value:
                return False
        else:
            if not left.is_finite() or abs(left - right) > Decimal("0.000001"):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="three useful questions and one ambiguous question")
    args = parser.parse_args()
    cube, llm = build_cube(), build_llm()
    try:
        catalog = load_catalog(cube)
    except Exception as exc:
        print(f"Cannot load the Cube catalog from {cube.base_url}: {exc}")
        return 2
    deps = Deps(llm=llm, cube=cube, catalog=catalog, as_of=settings.as_of)
    cases = json.loads(CASES.read_text())
    if args.smoke:
        cases = [cases[i] for i in (0, 1, 2, 10)]
    results, passed = [], 0
    for i, case in enumerate(cases, start=1):
        out = run_question(case["question"], deps)
        problems = check(out, case["expect"])
        passed += not problems
        models = ", ".join(dict.fromkeys(out.get("llm_models", []))) or "n/a"
        print(f"[{i}/{len(cases)}] {'PASS' if not problems else 'FAIL'} · {case['question']}")
        print(f"      outcome {out.get('outcome')} · model calls {out.get('llm_calls', 0)} · tool calls {len(out.get('steps', []))}"
              f" · cube calls {out.get('cube_calls', 0)} · {models}")
        for p in problems:
            print(f"      - {p}")
        results.append({"question": case["question"], "expect": case["expect"], "outcome": out.get("outcome"),
                        "problems": problems, "steps": out.get("steps"), "final": out.get("final"),
                        "queries": out.get("queries"), "verification": out.get("verification"),
                        "llm_models": out.get("llm_models"), "llm_calls": out.get("llm_calls"),
                        "cube_calls": out.get("cube_calls"), "trace_id": out.get("trace_id")})
    summary = {"passed": passed, "total": len(cases), "llm_requests": llm.calls,
               "as_of": settings.as_of.isoformat(), "run_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    print(f"\n{passed}/{len(cases)} cases passed · {llm.calls} free-router requests · written to {RESULTS.name}")
    lf.flush()
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
