# Marketing analytics agent

A focused, production-minded agent that answers marketing questions through **Cube over ClickHouse**. The model selects semantic queries; Cube defines and calculates the metrics; deterministic application code controls execution, checks the returned numbers, and renders the evidence.

```text
Browser → FastAPI → LangGraph → Cube REST → ClickHouse
                       ├── OpenRouter free router
                       └── Langfuse tracing
```

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The application uses managed Cube and ClickHouse deployments, an OpenRouter key, and Langfuse keys for tracing.

### Run the offline checks first

```bash
uv sync --extra seed
make test
```

These tests use a scripted model and stub Cube. They make no external requests and do not require service credentials. They test the application logic, not the quality of the live model.

### Use an existing deployment

```bash
cp .env.example .env
# Set CUBE_URL, CUBEJS_API_SECRET, OPENROUTER_API_KEY and LANGFUSE_*.
make cube-check
make serve                  # http://127.0.0.1:8000
make smoke                  # three data questions and one ambiguous question
```

The deployment URL and Cube API secret must be shared privately; credentials are not included in this repository. OpenRouter and Langfuse also require valid keys. Do not run `make seed` against an existing deployment unless rebuilding its synthetic data is intended.

### Create your own deployment

1. Create a ClickHouse Cloud service and fill in `CLICKHOUSE_*` in `.env`.
2. Run `make seed`. It **drops and recreates** the three `marketing` tables, then loads deterministic fake data.
3. Create a Cube Cloud deployment importing this repository with project directory `cube`. Configure its ClickHouse connection and `CUBEJS_API_SECRET`.
4. Put the deployment's base URL and matching secret in `.env` and follow the existing-deployment commands above.

Cloud cold starts and free-model quotas can delay or prevent a live run. Confirm that the managed deployments are available before a demonstration. No local Docker deployment is included.

## Supported scope

One marketing view, USD, UTC, and synthetic data covering **1 March–31 August 2026**. The following scenarios show the intended scope; exact narrative wording can vary with the routed free model.

| Question | Expected behavior |
|---|---|
| How much did we spend by channel in August 2026? | Period-filtered spend breakdown |
| Which campaign had the most purchases in July 2026? | Purchases ranked descending |
| Which campaign had the strongest ROAS in August 2026? | Cube's revenue/spend metric ranked descending |
| Compare spend in July and August 2026 by channel | Both periods, with each query's scope visible |
| Daily clicks for Shopping DE in the first week of August 2026 | Daily series with an exact campaign filter |
| How much did email spend in August 2026? | A legitimate zero, with matching records confirmed |
| What happened lately? | One clarification question |
| What is customer lifetime value by channel? | Explain that the metric is unavailable |
| Purchases for Winback Series in 2025 | Explain that there are no matching records |

Relative dates use `AGENT_AS_OF_DATE=2026-09-14`. “Last N months” means N completed calendar months. Explicit periods are preserved even outside coverage; the prompt asks the model to explain any missing coverage. Date interpretation and narrative wording remain model decisions.

The agent is single-turn. The UI joins a clarification reply to the original question; it does not provide general conversation memory. Results are limited to 50 rows across comparison sets. A result that reaches its query limit is conservatively marked potentially incomplete. Overall totals must be requested as Cube aggregates, rather than inferred from a limited breakdown.

## Design decisions

### 1. Warehouse schema and generated data

| Table | Grain | Seeded rows |
|---|---|---:|
| `campaigns` | One campaign | 12 |
| `ad_spend` | Campaign × day × device | 2,956 |
| `purchases` | One attributed purchase event | 8,672 |

`warehouse/generate_data.py` uses `random.Random(42)` and campaign parameters for March–August 2026. Campaigns have different spend, purchase rates, active windows and device mixes. Examples include a July-only campaign, a paused campaign, a falling-return campaign, and email purchases without paid media spend. This creates useful aggregation and imperfect-data cases without hardcoding answers in the agent.

Each purchase has one campaign attribution resolved upstream. There are no refunds, multiple currencies, or attribution-lag/cohort calculations. Spend and purchases are compared over the same calendar period.

### 2. Cube measures and dimensions

`campaign_daily` stacks spend facts and purchase events with `UNION ALL`, then joins the campaign table many-to-one. This avoids multiplying spend by the number of purchase events. The exposed view is `marketing_performance`.

| Measures | Definition |
|---|---|
| Spend, revenue, impressions, clicks, purchases | Additive sums |
| Cost per purchase | Spend / purchases; NULL when spend or purchases is zero |
| ROAS | Revenue / spend; NULL when spend is zero |
| CPC, CPM | Spend / clicks; spend × 1,000 / impressions |
| CTR, conversion rate | Clicks / impressions; purchases / clicks |
| Average order value | Revenue / purchases |

Ratios are **ratios of sums** at the requested grouping and period. They are never sums or averages of row-level ratios. `NULLIF` handles undefined denominators; `toFloat64` avoids integer division. Treating zero-spend CPA as undefined is a deliberate paid-media reporting convention, so email is not ranked as a zero-cost acquisition campaign.

Dimensions are campaign, channel, country, objective, device and date. Internal first/last-date measures provide coverage metadata. Cube descriptions and formatting metadata drive tool responses and displayed tables.

### 3. LangGraph state and nodes

The [generated graph](docs/graph.md) has five nodes:

| Node | Responsibility |
|---|---|
| `prepare` | Seed view summaries, the question, date convention and tool instructions |
| `agent` | Call the free model and interpret its native or JSON tool calls |
| `tools` | Execute Cube-backed discovery and queries; return errors the model can repair |
| `verify` | Require a successful query and run the numerical consistency check; allow one correction |
| `answer` | Render the narrative, distinct query tables, periods, filters and metric definitions |

State holds messages, query results, tool steps, model/Cube call counts, verification details and outcome. Runtime dependencies are injected, so tests can exercise the same graph offline. The loop allows seven model turns, which gives one recovery turn beyond the normal discovery/query/answer path. Explicit network retries inside a turn are also counted.

### 4. Model, tool and code boundary

The model has six tools: `list_views`, `describe_view`, `search_fields`, `find_dimension_values`, `run_query`, and `final_answer`. Discovery reads Cube metadata; queries go through Cube `/dry-run` and `/load`. The application never connects directly to ClickHouse. Only the seed utility does so.

The model chooses fields, values, periods and the narrative. Cube owns metric calculation. Code owns request limits, error handling, numerical checks and table rendering. Native tool calls are preferred; a plain JSON protocol handles a router response indicating that tool use is unavailable.

The explicit graph keeps tool, validation, and retry decisions visible and testable. A fixed interpret→query→answer pipeline would be simpler. Dynamic discovery adds model calls and failure modes, but lets the agent inspect new public Cube fields without a code change. Its scalability beyond this small semantic model has not been established.

### 5. Validation and its limits

- Cube validates the semantic query before execution. Code fixes the timezone to UTC and caps positive integer limits.
- An ungrouped zero/NULL aggregate triggers a one-row query grouped by date, preserving the original measures, filters and period. This distinguishes a real zero from an aggregate over no matching records without changing the deployed Cube model.
- The numerical check accepts returned values at the written precision. Only percent-formatted metrics may be multiplied by 100. Only complete, additive results may support totals and shares; totals are separate per comparison period and never summed across repeated queries. Ratios must be queried from Cube. The final one or two distinct successful queries are the evidence set for verification and display.
- Differences and percentage changes within a metric can support comparison text. Candidate differences are bounded for longer series. Direct scalar matching preserves signs and uses rounding tolerance, without an extra percentage allowance.
- The first rejection returns a reason to the model; a second rejects the narrative and shows the data already retrieved.

**This is a numerical consistency check, not semantic verification.** It does not establish that a figure belongs to the entity or metric named in a sentence, that a ranking is described correctly, or that words such as “rose” and “fell” have the right direction. Small bare integers up to 31 are exempt to avoid confusing ranks and calendar references with metrics. ISO dates and recognized calendar-year phrases are skipped. Tests pin these limitations so they remain visible.

### 6. Imperfect questions and errors

Ambiguous questions should produce one clarification. Missing metrics should produce an unsupported response. Cube query errors return repair hints; authentication, quota and transport failures stop the run. Repeated server errors, model errors, exhausted budgets and failed numerical checks produce explicit errors. Empty results use `no_data`; zero-valued results with matching facts remain answers.

Each API response includes the answer, tool steps, queries, rows, numerical check and trace information. The final one or two distinct queries retain separate labeled tables; identical queries are deduplicated. The UI exposes all raw query history under “Under the hood.”

### 7. What would change before production

Use evaluated models with reliable tool behavior; add checks tying structured claims to specific result cells; introduce authenticated user access and Cube row-level controls; add real conversation state if required. Catalog search, caching and retrieval would need measurement against a larger model before choosing embeddings or other infrastructure. Real customer data would require a separate review of model-provider and tracing-service data handling.

## Inference and tracing

The only inference model configured is `openrouter/free`, with zero maximum prompt/completion prices and no paid fallback. Every response must report a `:free` model and zero cost. SDK retries are disabled; one rate-limit retry and a tool-protocol fallback are explicit and counted.

Langfuse records the question, graph nodes, model generations, Cube calls and final response. The trace includes routed models, reported cost, call counts and outcome. Tracing is optional for local startup. Offline tests disable it, including the opt-in live test module; use `make smoke` or the UI to generate and inspect a real trace.

## Validation commands

```bash
make test                   # offline regression suite
RUN_LIVE=1 make test         # adds four live integration checks
make smoke                  # three useful live questions + one ambiguous question, with tracing
make eval                   # all 15 cases; may exceed the available free quota
make examples               # ten UI examples, with expected outcomes printed
```

The evaluation checks expected outcome, exact periods and dimensions, filters, requested metrics, granularity, ranking direction and selected golden numeric results. Comparisons require both periods. Golden values come from the deterministic seed data independently of model output. Reports are written to ignored `evals/last_run.json`, including run time, scope, queries, outcomes and trace IDs. Passing these checks still does not prove that every sentence is correct.

Current verification: `make test` reports **162 passed and 4 live tests skipped**; `RUN_LIVE=1 uv run pytest tests/test_live.py -q` reports **4 passed** against the configured services; `make check-secrets` reports **clean**.

## AI assistance

OpenAI Codex supported targeted parts of the work: discussing the implementation plan and verification flow, reproducing correctness problems, making focused fixes, and drafting or expanding evaluation cases and automated tests. The architecture, metric definitions, scope, and tradeoffs were selected and reviewed by the author.

AI-assisted changes were reviewed through the Git diff, the offline regression suite, live integration checks, the 15-case behavior evaluation, manual UI runs, and Langfuse trace inspection. The checks cover inflated numbers, invalid ratio aggregation, zero versus missing data, result limits, comparison rendering, evaluation false positives, free-model enforcement, and accidental credential exposure.
