"""A result is readable before opening any filters, previews or technical views."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_execution_dashboard import ledger
from dashboard.research_data import SPRO,AP


def test_results_default_reference_selection_and_lazy_preview(ledger,monkeypatch):
    store,campaigns,cid,other=ledger
    ids=[]
    for seed,value in [(7,.8),(19,.65)]:
        rid=store.create_run('scientific-question','candidate_'+str(seed),'juice_bottle',config={'seed':seed,'plan_arm':'baseline'})
        ids.append(rid);campaigns.attach_run(other,rid)
        with store.connect() as con:
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,1,'REAL_DEVELOPMENT_FEEDBACK',?)",(rid,json.dumps({
                'metrics':{SPRO:value,AP:.6,'dev_real/normal_false_positives':1,'dev_real/normal_images':47},
                'evaluator_version':'compatible','split_id':'fixture'})))
    monkeypatch.setattr('dashboard.research_data.active_incumbent',lambda campaign:{'run_id':ids[0]})
    monkeypatch.setattr('dashboard.detail_views.artifact_status',lambda record:('Available','fixture'))
    def forbidden(*args,**kwargs):raise AssertionError('opening Results must not load image artifacts')
    monkeypatch.setattr('dashboard.detail_views.real_heatmap_sample',forbidden)
    page=Path(__file__).parents[1]/'dashboard/pages/03_Results.py'
    app=AppTest.from_file(page).run()
    assert not app.exception
    assert app.selectbox[0].label=='Model' and app.selectbox[0].value==ids[0]
    assert app.metric[0].label=='Logical AU-sPRO' and app.metric[0].value=='0.8000'
    assert any('Current research reference' in c.value for c in app.caption)
    for label in ('Localization preview','Model details','Browse and compare models'):
        assert any(e.label==label and not e.proto.expanded for e in app.expander)
    app.selectbox[0].select(ids[1]).run()
    assert not app.exception and app.metric[0].value=='0.6500'
    assert not any('Current research reference' in c.value for c in app.caption)
