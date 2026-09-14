"""Plan accuracy: how often does the free router turn a question into the right Plan?

The one non-deterministic step of the graph is ``interpret``. This script runs it with the
real model for every case in ``evals/plan_cases.json``, lets ``build_query`` normalise the
result exactly as in production, and compares the executed plan field by field against
the expected values. No Cube query is run (the plan is decided before any data call), so
the only cost is one model request per case (plus repairs/re-rolls, all counted).

    make eval          # prints a table and a summary; writes evals/last_run.json

The golden cases are deliberately different from the prompt's few-shot examples and from
the UI's example questions, so the score is not inflated by memorisation.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from app import config  # noqa: F401  (loads .env)
from langgraph.runtime import Runtime

from app.agent.dates import PeriodError, resolve
from app.agent.nodes import build_query, interpret
from app.config import settings
from app.models.state import Deps
from app.runtime import build_cube, build_llm
from app.tools.cube import load_catalog

CASES = Path(__file__).with_name("plan_cases.json")
RESULTS = Path(__file__).with_name("last_run.json")
FIELDS = ("intent", "measures", "dimension", "period", "compare_period", "filters", "order_by")


def executed_plan(question: str, deps: Deps) -> tuple[str, dict | None, dict]:
    """Run interpret + build_query only; return (category, normalised plan, raw interpret output)."""
    runtime = Runtime(context=deps)
    state = {"question": question, "as_of": deps.as_of.isoformat(), "cube_calls": 0, "notes": []}
    out = interpret(state, runtime, {})
    if out.get("outcome") == "error":
        return "error", None, out
    built = build_query({**state, **out}, runtime)
    plan = built.get("plan") or out["plan"]
    category = built.get("outcome") or "query"  # a built query means the plan was executable
    return category, plan, out


def same_period(got: str | None, want: str, as_of: date, coverage: tuple[date, date], base=None) -> bool:
    """Two tokens are the same period if they resolve to the same dates ("last_month" == "2026-08")."""
    if not got:
        return False
    try:
        a, b = resolve(got, as_of, coverage, base=base), resolve(want, as_of, coverage, base=base)
    except PeriodError:
        return got.lower() == want.lower()
    return (a.start, a.end) == (b.start, b.end)


def compare(expect: dict, category: str, plan: dict | None, as_of: date, coverage: tuple[date, date]) -> dict[str, bool]:
    checks = {"category": category == expect["category"]}
    if expect["category"] != "query" or plan is None:
        return checks
    base = None
    for field in FIELDS:
        if field not in expect:
            continue
        got = plan.get(field)
        if field == "filters":
            got = sorted((f["dimension"], f["value"].lower()) for f in got or [])
            want = sorted((d, v.lower()) for d, v in expect[field])
            checks[field] = got == want
        elif field == "period":
            checks[field] = same_period(got, expect[field], as_of, coverage)
            base = resolve(got, as_of, coverage) if checks[field] else None
        elif field == "compare_period":
            checks[field] = same_period(got, expect[field], as_of, coverage, base=base)
        else:
            checks[field] = got == expect[field]
    return checks


def main() -> int:
    cases = json.loads(CASES.read_text())
    rescore = "--rescore" in sys.argv  # re-grade the plans saved by the last run; no model requests
    saved = {r["question"]: r for r in json.loads(RESULTS.read_text())["cases"]} if rescore else {}
    if rescore:
        coverage = tuple(date.fromisoformat(d) for d in json.loads(RESULTS.read_text())["summary"]["coverage"])
        as_of = settings.as_of
    else:
        cube, llm = build_cube(), build_llm()
        catalog = load_catalog(cube)
        deps = Deps(llm=llm, cube=cube, catalog=catalog, as_of=settings.as_of)
        coverage, as_of = catalog.coverage, settings.as_of
    rows, field_hits, field_total = [], 0, 0
    for i, case in enumerate(cases, 1):
        if rescore:
            r = saved[case["question"]]
            category, plan, out, requests = r["category"], r["plan"], {"llm_model": r["model"]}, r["requests"]
        else:
            before = llm.calls
            category, plan, out = executed_plan(case["question"], deps)
            requests = llm.calls - before
        checks = compare(case["expect"], category, plan, as_of, coverage)
        ok = all(checks.values())
        field_hits += sum(checks.values()); field_total += len(checks)
        wrong = [k for k, v in checks.items() if not v]
        rows.append({"question": case["question"], "ok": ok, "category": category, "wrong_fields": wrong,
                     "model": out.get("llm_model"), "requests": requests, "plan": plan})
        print(f"[{i:2d}/{len(cases)}] {'PASS' if ok else 'FAIL'}  {case['question']}")
        if not ok:
            print(f"        got category={category} plan={json.dumps({k: (plan or {}).get(k) for k in FIELDS}, default=str)}")
            print(f"        wrong: {', '.join(wrong)}  (model: {out.get('llm_model')})")
    passed = sum(r["ok"] for r in rows)
    total_requests = sum(r["requests"] for r in rows)
    summary = {"run_date": date.today().isoformat(), "as_of": as_of.isoformat(),
               "coverage": [coverage[0].isoformat(), coverage[1].isoformat()],
               "cases": len(rows), "passed": passed,
               "field_accuracy": round(field_hits / field_total, 3) if field_total else None,
               "requests": total_requests, "models": sorted({r["model"] for r in rows if r["model"]})}
    RESULTS.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, default=str))
    print(f"\nPLAN ACCURACY: {passed}/{len(rows)} cases exact ({summary['field_accuracy']:.0%} of checked fields) "
          f"· {total_requests} model requests · models seen: {len(summary['models'])} · written to {RESULTS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
