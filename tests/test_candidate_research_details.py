import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from orchestrator.state_store import StateStore
from dashboard.data_access import candidate_research_details


def test_details_are_linked_to_selected_hypothesis_and_run(tmp_path, monkeypatch):
    store = StateStore(tmp_path/'ledger.sqlite')
    monkeypatch.setenv('LOCO_LEDGER', str(store.path))
    store.add_proposal({'hypothesis_id':'selected','claim':'Selected claim', 'references':['https://example.org/paper']})
    store.add_proposal({'hypothesis_id':'other','claim':'Wrong claim'})
    rid = store.create_run('selected','model_one','juice_bottle', config={'seed':7})
    other = store.create_run('other','model_two','juice_bottle')
    with store.connect() as con:
        for hid in ('selected','other'):
            con.execute("INSERT INTO events(time,kind,payload_json) VALUES(0,'IMPLEMENTER_RESPONSE',?)",
                        (json.dumps({'hypothesis_id':hid,'response':{'caveats':[hid]}}),))
        for run, text in ((rid,'Selected assessment'),(other,'Wrong assessment')):
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,0,'CRITIC_REVIEW',?)",
                        (run,json.dumps({'objective_measured_results':text})))
        # A later implementation must not be attributed to this older run.
        con.execute("INSERT INTO events(time,kind,payload_json) VALUES(9999999999,'IMPLEMENTER_RESPONSE',?)",
                    (json.dumps({'hypothesis_id':'selected','response':{'caveats':['future']}}),))
    detail = candidate_research_details(rid)
    assert detail['proposal']['claim'] == 'Selected claim'
    assert detail['implementation']['caveats'] == ['selected']
    assert detail['review']['objective_measured_results'] == 'Selected assessment'
    assert candidate_research_details('missing') is None
    app = AppTest.from_string(f'from dashboard.components import candidate_research_panel\ncandidate_research_panel({rid!r})').run()
    assert not app.exception
    rendered = '\n'.join(m.value for m in app.markdown)
    assert 'Selected claim' in rendered and 'Selected assessment' in rendered
    assert 'Wrong claim' not in rendered and 'Wrong assessment' not in rendered
    assert any(b.url == 'https://example.org/paper' for b in app.get('link_button'))


def test_missing_research_records_have_explicit_empty_states(tmp_path,monkeypatch):
    store = StateStore(tmp_path/'ledger.sqlite')
    monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    rid = store.create_run('unknown','fixture','juice_bottle')
    app = AppTest.from_string(f'from dashboard.components import candidate_research_panel\ncandidate_research_panel({rid!r})').run()
    assert not app.exception
    assert any('No linked research proposal' in e.value for e in app.info)
    assert any('No Critic assessment' in e.value for e in app.info)
