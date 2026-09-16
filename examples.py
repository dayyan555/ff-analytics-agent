"""Run the example questions end-to-end (``make examples``).

Exercises the live stack (Cube Cloud + OpenRouter free router + Langfuse) with
the same questions the UI offers, printing the tool calls and the answer per
question and a total. Budget: about 3 model requests per question (7 model turns at most).
Exit code 2 if the semantic layer is unreachable or any question ends in ``error``.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal

from app import config  # noqa: F401  (loads .env before the Langfuse client is created)
from app.models.state import Deps
from app.runtime import build_cube, build_llm, lf, run_question
from app.tools.cube import load_catalog
from app.web.questions import EXAMPLE_QUESTIONS


def main() -> int:
    cube, llm = build_cube(), build_llm()
    try:
        catalog = load_catalog(cube)
    except Exception as exc:  # CubeError or transport failure: nothing to run
        print(f"Cannot load the Cube catalog from {cube.base_url}: {exc}")
        lf.flush()
        return 2
    deps = Deps(llm=llm, cube=cube, catalog=catalog, as_of=config.settings.as_of)

    total = len(EXAMPLE_QUESTIONS)
    cube_calls, cost, refusals = 0, Decimal(0), 0
    failed = False
    for i, example in enumerate(EXAMPLE_QUESTIONS, start=1):
        out = run_question(example.question, deps)
        outcome = out.get("outcome", "error")
        failed = failed or outcome == "error"
        cube_calls += out.get("cube_calls", 0)
        cost += Decimal(out.get("llm_cost") or 0)
        refusals += out.get("error_kind") == "free_guard"
        verdict = "as expected" if outcome == example.expect else f"expected {example.expect}"
        print(f"[{i}/{total}] ({example.kind}) {example.question}")
        models = ", ".join(dict.fromkeys(out.get("llm_models", []))) or "n/a"
        print(
            f"outcome: {outcome} ({verdict}) · models: {models}"
            f" · model calls: {out.get('llm_calls', 0)} · tool calls: {len(out.get('steps', []))}"
            f" · cube calls: {out.get('cube_calls', 0)} · trace: {out.get('trace_id') or 'n/a'}"
        )
        for step in out.get("steps", []):
            print(f"    {step['step']}. {step['tool']}({json.dumps(step['args'])[:100]}) -> {step['summary']}")
        for line in str(out.get("answer", "")).splitlines():
            print(f"    {line}".rstrip())
        print()

    print(f"TOTAL — LLM requests: {llm.calls} · Cube calls: {cube_calls} · reported cost: ${cost}"
          f" · free-guard refusals: {refusals}")
    lf.flush()
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
