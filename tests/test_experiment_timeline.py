"""Chronology comes from execution evidence, not rank or delayed evaluation."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from dashboard.experiment_timeline import execution_contexts,timeline_rows,timestamp
from dashboard.data_access import campaign_summary
from dashboard.research_data import model_results
from test_execution_dashboard import ledger


def make_run(store,campaigns,cid,name,start,end,*,frozen=False):
    rid=store.create_run('h',name,'juice_bottle',config={'seed':7,'plan_arm':'a','execution_kind':'frozen' if frozen else 'train'})
    campaigns.attach_run(cid,rid)
    with store.connect() as con:
        con.execute("UPDATE runs SET started_at=?,finished_at=?,status=?,metrics_json=? WHERE run_id=?",(start,end,'REVIEWED' if end else 'QUEUED',json.dumps({'training/loop_elapsed_seconds':2}),rid))
    return rid


def test_actual_start_order_is_stable_not_creation_or_feedback_order(ledger):
    store,campaigns,cid,other=ledger
    later=make_run(store,campaigns,other,'created_first',20,25)
    first=make_run(store,campaigns,other,'started_first',10,13)
    queued=make_run(store,campaigns,other,'not_started',None,None)
    context=execution_contexts(other)
    assert list(context)==[first,later,queued]
    assert context[first]['order']==1 and context[later]['order']==2 and context[queued]['order'] is None
    assert context[first]['job_seconds']==3
    rows=timeline_rows(campaign_summary(other))
    assert [r['#'] for r in rows]==[1,2,None]
    assert rows[0]['Training']=='2 s' and rows[0]['Job time']=='3 s'
    assert rows[-1]['Started']=='Not recorded' and rows[-1]['Job time']=='Not recorded'


def test_job_time_does_not_include_later_promotion_or_critic_wait(ledger):
    store,campaigns,cid,other=ledger
    rid=make_run(store,campaigns,other,'promoted',10,999)
    with store.connect() as con:
        con.execute("UPDATE runs SET status='PROMOTED' WHERE run_id=?",(rid,))
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,20,'TRANSITION',?)",(rid,json.dumps({'from':'RUNNING','to':'COMPLETED'})))
    assert execution_contexts(other)[rid]['job_seconds']==10
    assert execution_contexts(other)[rid]['finished_at']==20


def test_frozen_models_failed_jobs_and_campaign_scoping(ledger):
    store,campaigns,cid,other=ledger
    original=make_run(store,campaigns,cid,'original',5,8)
    frozen=make_run(store,campaigns,other,'frozen',10,13,frozen=True)
    failed=make_run(store,campaigns,other,'failed',14,17)
    with store.connect() as con:con.execute("UPDATE runs SET status='FAILED' WHERE run_id=?",(failed,))
    assert execution_contexts()[original]['order']==1
    rows=timeline_rows(campaign_summary(other))
    assert [r['#'] for r in rows]==[1,2]
    assert rows[0]['Training']=='Not retrained' and rows[1]['Status']=='Failed'
    assert original not in [r['Run'] for r in rows]


def test_dates_are_explicit_paris_times_including_dst():
    assert timestamp(None)=='Not recorded'
    assert timestamp(1792931638).startswith('2026-10-25 13:33:58 CET')
    assert timestamp(1782864000).endswith('CEST')


def test_evaluation_cache_does_not_invent_current_duration_and_model_origin_is_linked(ledger):
    store,campaigns,cid,other=ledger
    rid=make_run(store,campaigns,cid,'old_model',10,20)
    with store.connect() as con:con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,30,'REAL_DEVELOPMENT_FEEDBACK',?)",(rid,json.dumps({'metrics':{'dev_real/evaluation_wall_seconds':999},'reused_from_evaluation_id':'original','evaluator_version':'compatible'})))
    context=execution_contexts(cid)[rid]
    assert context['evaluation_reused'] and context['evaluation_seconds'] is None
    assert timeline_rows(campaign_summary(cid))[0]['Evaluation']=='Reused'
    assert model_results()[0]['execution']['campaign_id']==cid


def test_research_timeline_is_visible_and_numbering_does_not_reset_after_paging(ledger):
    store,campaigns,cid,other=ledger
    for i in range(17):make_run(store,campaigns,other,'model_'+str(i),100+i*10,105+i*10)
    app=AppTest.from_file(Path(__file__).parents[1]/'dashboard/pages/02_Research.py').run()
    assert not app.exception
    assert list(app.dataframe[0].value['#'])==list(range(1,16))
    app.selectbox[0].select(2).run()
    assert not app.exception
    assert list(app.dataframe[0].value['#'])==[16,17]
