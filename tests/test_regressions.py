"""Regression coverage for numerical checks, query limits, and evaluation behavior."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.agent.graph import graph
from app.agent.verify import relevant_queries, unverified_numbers
from app.agent.prompt import system_prompt
from app.tools.toolkit import MAX_ROWS, Toolkit
from evals.agent_eval import check
from tests.conftest import MP, StubCube, cube_query, final, result_set, row, tool_call
from tests.test_verify import QUERY


def test_inflated_spend_is_rejected_by_the_whole_graph(deps):
    d = deps([
        tool_call('run_query', {'query': cube_query(['spend'], ['channel'])}),
        final('Meta spent $1,368,934 in August 2026.'),
        final('Meta spent $13,689.34 in August 2026.', call_id='correction'),
    ], cube=StubCube([result_set([row(channel='meta', spend='13689.34')])]))
    out = graph.invoke({'question': 'Meta spend in August 2026'}, context=d)
    assert out['verify_failures'] == 1 and out['outcome'] == 'answer'
    assert out['final']['text'] == 'Meta spent $13,689.34 in August 2026.'


def test_ratios_are_not_summed_or_recomputed_from_other_columns():
    q = {'ok': True, 'columns': [{'key': 'roas', 'type': 'number', 'aggregation': 'ratio'}],
         'rows': [{'roas': '2.5'}, {'roas': '3.5'}]}
    assert unverified_numbers('Overall ROAS is 6.0x.', [q]) == ['6.0x']
    assert unverified_numbers('Meta CPA was $33.23.', [QUERY]) == ['$33.23']
    assert unverified_numbers('CTR was 7.55%.', [QUERY]) == ['7.55%']


def test_signed_cells_and_written_precision_are_preserved():
    q = {'ok': True, 'columns': [{'key': 'spend', 'type': 'number', 'aggregation': 'sum'}],
         'rows': [{'spend': '13689.34'}]}
    assert unverified_numbers('Spend was -$13,689.34.', [q]) == ['-$13,689.34']
    assert unverified_numbers('Spend was $13,690.', [q]) == ['$13,690']
    assert unverified_numbers('Spend was $13,689.', [q]) == []
    assert unverified_numbers('Meta had 2,026 purchases.', [QUERY]) == ['2,026']
    assert unverified_numbers('No purchases were recorded in 2025.', []) == []


def test_partial_results_and_duplicate_queries_cannot_back_totals():
    assert unverified_numbers('Total spend was $24,709.44.', [{**QUERY, 'complete': False}]) == ['$24,709.44']
    assert unverified_numbers('Total spend was $49,418.88.', [QUERY, QUERY]) == ['$49,418.88']


def test_narrative_entity_assignment_remains_a_documented_limitation():
    # This check does not parse sentence meaning. Keep the limitation explicit.
    assert unverified_numbers('Google spent $13,689.34 and Meta spent $11,020.10.', [QUERY]) == []


def test_zero_spend_is_an_answer_when_the_presence_query_finds_records(deps):
    query = cube_query(['spend'], filters=[{'member': MP + 'channel', 'operator': 'equals', 'values': ['email']}])
    cube = StubCube([[result_set([row(spend='0.00')])], [result_set([row(date='2026-08-01', spend='0.00')])]])
    d = deps([tool_call('run_query', {'query': query}), final('Email spent $0 in August 2026.')], cube=cube)
    out = graph.invoke({'question': 'Email spend in August 2026'}, context=d)
    assert out['outcome'] == 'answer' and '$0.00' in out['answer_body']
    assert out['queries'][0]['has_data'] is True
    probe = cube.calls[-1][1]
    assert probe['dimensions'] == [MP + 'date'] and probe['limit'] == 1
    assert probe['filters'] == query['filters'] and probe['timeDimensions'] == query['timeDimensions']


def test_single_result_at_limit_warns_model_and_user(deps):
    rows = [row(channel=f'c{i}', spend='100') for i in range(MAX_ROWS)]
    d = deps([tool_call('run_query', {'query': cube_query(['spend'], ['channel'])}),
              final('The returned rows show $100 for each channel.')], cube=StubCube([result_set(rows)]))
    out = graph.invoke({'question': 'Spend by channel in August 2026'}, context=d)
    assert out['outcome'] == 'answer'
    result = out['queries'][0]
    assert result['complete'] is False and 'may be incomplete' in result['note']
    assert 'may be incomplete' in out['answer_body']


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '50'])
def test_invalid_limits_do_not_reach_cube(catalog, limit):
    cube = StubCube()
    result = Toolkit(cube, catalog).run_query(cube_query(['spend'], limit=limit))
    assert not result['ok'] and not cube.calls


def test_ungrouped_total_with_limit_one_is_complete(catalog):
    result = Toolkit(StubCube([result_set([row(spend='500')])]), catalog).run_query(cube_query(['spend'], limit=1))
    assert result['complete'] is True and 'note' not in result


def test_ranked_limit_one_is_complete_for_the_requested_query(catalog):
    query = cube_query(['cost_per_purchase'], ['channel'], limit=1, order={MP + 'cost_per_purchase': 'asc'})
    result = Toolkit(StubCube([result_set([row(channel='google', cost_per_purchase='19.62')])]), catalog).run_query(query)
    assert result['complete'] is True and 'note' not in result


def test_more_comparison_sets_than_the_row_budget_are_marked_incomplete(catalog):
    sets = [result_set([{**row(spend='100'), 'compareDateRange': str(i)}]) for i in range(MAX_ROWS + 1)]
    result = Toolkit(StubCube(sets), catalog).run_query(cube_query(['spend']))
    assert result['complete'] is False and len(result['rows']) == MAX_ROWS


def test_identical_queries_deduplicate_but_comparison_keeps_both_periods(deps):
    queries = [cube_query(['spend'], date_range=('2026-07-01', '2026-07-31')), cube_query(['spend'])]
    cube = StubCube([[result_set([row(spend=100)])], [result_set([row(spend=80)])]])
    d = deps([tool_call('run_query', {'query': queries[0]}), tool_call('run_query', {'query': queries[1]}, 'second'),
              final('Spend fell from $100 in July 2026 to $80 in August 2026.')], cube=cube)
    out = graph.invoke({'question': 'Compare July and August spend'}, context=d)
    for text in (out['answer_body'], out['footer']):
        assert '2026-07-01 to 2026-07-31' in text and '2026-08-01 to 2026-08-31' in text
    from app.agent.answer import _tables_to_show
    q = out['queries'][0]
    assert len(_tables_to_show([q, {**q, 'query_id': 'retry'}])) == 1


def test_only_the_last_two_distinct_queries_support_and_render_the_answer():
    queries = [
        {'ok': True, 'query': {'measures': [f'm{i}']}, 'columns': [], 'rows': []}
        for i in range(3)
    ]
    assert relevant_queries(queries) == queries[1:]


def test_prompt_resolves_relative_months_and_requires_vague_questions_to_clarify(catalog):
    text = system_prompt([catalog.views['marketing_performance'].summary()], __import__('datetime').date(2026, 9, 14), 6,
                         json_protocol=False)
    assert 'last month = 2026-08-01 to 2026-08-31' in text
    assert 'last 3 months = 2026-06-01 to 2026-08-31' in text
    assert 'last 6 months = 2026-03-01 to 2026-08-31' in text
    assert 'What happened lately?' in text and 'clarify without querying' in text
    assert 'Never approximate, proxy or substitute a metric' in text and 'CLV/LTV' in text


def test_low_confidence_metric_overlap_does_not_substitute_clv_for_aov(catalog):
    assert catalog.search('customer lifetime value') == []
    assert catalog.search('average order value')[0][0].short == 'aov'


CASES = json.loads((Path(__file__).parents[1] / 'evals/cases.json').read_text())


def evaluation_result(query, periods, rows=None):
    return {'ok': True, 'query': query, 'periods': [{'from': a, 'to': b} for a, b in periods], 'rows': rows or []}


def evaluated(queries):
    return {'outcome': 'answer', 'queries': queries, 'verification': {'checked': 1, 'unverified': []}}


def test_comparison_eval_rejects_a_single_month_and_accepts_both_protocols():
    expect = CASES[5]['expect']
    wrong = evaluation_result(cube_query(['spend', 'purchases'], ['channel']), [('2026-08-01', '2026-08-31')])
    assert any('periods' in p for p in check(evaluated([wrong]), expect))
    separate = [evaluation_result(cube_query(['spend', 'purchases'], ['channel'], tuple(period)), [period]) for period in expect['periods']]
    assert check({**evaluated(separate), 'answer_body': 'Q3 is a partial quarter.'}, expect) == []
    compare = cube_query(['spend', 'purchases'], ['channel'])
    compare['timeDimensions'][0] = {'dimension': MP + 'date', 'compareDateRange': expect['periods']}
    assert check({**evaluated([evaluation_result(compare, expect['periods'])]), 'answer_body': 'Q3 is partial.'}, expect) == []


def test_eval_rejects_wrong_order_filters_values_and_extra_grouping():
    expect = CASES[2]['expect']
    query = cube_query(['cost_per_purchase'], ['channel'], tuple(expect['period']), order={MP + 'cost_per_purchase': 'desc'})
    result = evaluation_result(query, [expect['period']])
    assert any('order' in p for p in check(evaluated([result]), expect))
    expect = CASES[0]['expect']
    query = cube_query(['spend'], date_range=tuple(expect['period']))
    result = evaluation_result(query, [expect['period']], [{'spend': '1.00'}])
    assert any('golden row' in p for p in check(evaluated([result]), expect))
    result['rows'] = expect['rows']
    assert check(evaluated([result]), expect) == []
    query['dimensions'] = [MP + 'channel']
    assert any('dimensions' in p for p in check(evaluated([result]), expect))
    query['dimensions'] = []
    query['filters'] = [{'member': MP + 'country', 'operator': 'notEquals', 'values': ['US']}]
    assert any('filters' in p for p in check(evaluated([result]), expect))


def test_golden_values_are_calculated_from_the_seed_data():
    # Independent of agent planning, Cube responses and the narrative verifier.
    from warehouse.generate_data import build_rows
    data = build_rows()
    total = sum((r[3] for r in data['ad_spend'] if r[0].month == 7), Decimal(0))
    assert str(total) == CASES[0]['expect']['rows'][0]['spend']
    for expected in CASES[1]['expect']['rows']:
        assert sum(1 for r in data['purchases'] if r[1].month == 8 and r[3] == expected['device']) == expected['purchases']
