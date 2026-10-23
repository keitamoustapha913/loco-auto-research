from streamlit.testing.v1 import AppTest
from dashboard.presentation import real_comparison_rows, localization_change


def feedback(rid, score, split='approved', version='compatible-v2'):
    return dict(run_id=rid, candidate_id='candidate_' + rid, split_id=split, evaluator_version=version,
                metrics={'dev_real/logical_au_spro_0_05_compatible': score})


def test_comparison_uses_campaign_order_and_real_values_only():
    data = [feedback('c', .273), feedback('a', .188), feedback('historical', .9), feedback('b', .241)]
    rows = real_comparison_rows(data, ['a', 'b', 'c'])
    assert [r['Logical AU-sPRO'] for r in rows] == [.188, .241, .273]
    assert rows[0]['Logical AUROC'] is None
    assert round(localization_change(rows), 1) == 45.2
    assert real_comparison_rows(data, []) == []


def test_missing_and_mixed_protocols_do_not_produce_relative_claim():
    assert localization_change(real_comparison_rows([feedback('a', 0), feedback('b', .2)], ['a','b'])) is None
    assert localization_change(real_comparison_rows([feedback('a', None), feedback('b', .2)], ['a','b'])) is None
    assert localization_change(real_comparison_rows([feedback('a', .1), feedback('b', .2, version='other')], ['a','b'])) is None


def test_outcome_and_results_render_with_honest_state():
    script = '''
from dashboard.components import campaign_outcome_card, scientific_results_comparison
campaign_outcome_card(dict(name='Fixture', status='LIMIT_REACHED', reason='experiment count reached',
 created_at=0, ended_at=1444, codex_calls=10, failure_events=0,
 config={'limits':{'max_experiments':3,'max_codex_calls':12,'max_gpu_hours':240/3600}},
 usage={'experiments':3,'successful':3,'gpu_seconds_live':76.6}), pending=0, tracking='Connected')
scientific_results_comparison([dict(run_id='a',candidate_id='fixture',split_id='approved',evaluator_version='compatible',metrics={})], ['a'])
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert [(m.label,m.value) for m in app.metric][:3] == [('Experiments','3 / 3'),('GPU-slot time','76.6 / 240 s'),('Codex calls','10 / 12')]
    assert any('Completed' in m.value for m in app.markdown)
    assert app.dataframe[0].value.iloc[0]['Logical AU-sPRO'] == 'Unavailable'
    failed = AppTest.from_string(script.replace("'successful':3", "'successful':2").replace('failure_events=0', 'failure_events=1')).run()
    assert not failed.exception
    assert not any(m.value == '**Completed**' for m in failed.markdown)


def test_candidate_order_does_not_follow_feedback_delivery(tmp_path, monkeypatch):
    import json
    from orchestrator.state_store import StateStore
    from dashboard.data_access import real_feedback_summaries
    store = StateStore(tmp_path / 'ledger.sqlite')
    monkeypatch.setenv('LOCO_LEDGER', str(store.path))
    first = store.create_run('h1', 'first', 'juice_bottle')
    second = store.create_run('h2', 'second', 'juice_bottle')
    with store.connect() as con:
        for rid, version in [(second, 'old'), (first, 'current'), (second, 'current')]:
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,0,'REAL_DEVELOPMENT_FEEDBACK',?)",
                        (rid, json.dumps({'evaluator_version': version, 'metrics': {}})))
    records = real_feedback_summaries()
    assert [r['run_id'] for r in records] == [first, second]
    assert [r['evaluator_version'] for r in records] == ['current', 'current']
    assert [r['Candidate'] for r in real_comparison_rows(records)] == ['First', 'Second']
