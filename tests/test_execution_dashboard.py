"""Offline regression of the observed boundary, geometry and dashboard failures."""
import copy
import json
import time
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from orchestrator.experiment_runner import validate_training_timing
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from benchmark.synthetic_geometry import disjoint_copy_domain,validate_candidate_geometry
from benchmark.train_adapter import CandidateContractError
from scripts.training_phase import TrainingClock,run_training_phases
from test_training_loop import timed_module,loop_config
from test_confirmation_controls import future_plan
from orchestrator.confirmation import verify_pairs
from dashboard import research_data as data
from dashboard.data_access import real_feedback_summaries,campaign_summary
from test_training_throughput import config
from test_experiment_plan import laboratory
from plan_fixtures import experiment_plan


def test_observed_completed_boundary_and_invalid_timings():
    cfg=loop_config(900)
    audit={'protocol':'prepared-training-step-v1','verified':True,'status':'COMPLETED',
        'duration_seconds':900.0088658127934,'completion_policy':'elapsed-training-loop-v1',
        'last_iteration_seconds':.008824584074318409,'completed_steps':101408}
    metrics={'training/duration_seconds':audit['duration_seconds']}
    validate_training_timing(cfg,metrics,audit)
    for changes in ({'duration_seconds':899.999999},{'duration_seconds':900.02},
                    {'verified':False},{'completed_steps':0},{'last_iteration_seconds':float('nan')},
                    {'last_iteration_seconds':0},{'duration_seconds':float('nan')}):
        wrong=dict(audit,**changes)
        with pytest.raises(ValueError):validate_training_timing(cfg,{'training/duration_seconds':wrong['duration_seconds']},wrong)
    with pytest.raises(ValueError):validate_training_timing(cfg,{'training/duration_seconds':900},audit)


def test_update_boundary_includes_inter_iteration_publication(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    original=TrainingClock.publish
    def publication(clock):
        original(clock)
        if clock.status in {'PREPARING','TRAINING'}:now[0]+=.00005
    monkeypatch.setattr(TrainingClock,'publish',publication)
    _,duration=run_training_phases(timed_module(now),loop_config(),tmp_path,tmp_path/'checkpoint',budget=.9)
    audit=json.loads((tmp_path/'training_timing.json').read_text())
    assert duration>=.9 and audit['last_iteration_seconds']>=.5
    validate_training_timing(loop_config(),{'training/duration_seconds':duration},audit)


def test_progress_io_is_coalesced_but_terminal_counters_exact(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    path=tmp_path/'audit.json';clock=TrainingClock(path,900,target=True,loop=True)
    before=path.stat().st_mtime_ns
    for i in range(500):
        clock.steps+=1;now[0]+=.001;clock.status='TRAINING';clock.publish()
    assert path.stat().st_mtime_ns==before
    clock.status='COMPLETED';clock.verified=True;clock.publish()
    assert json.loads(path.read_text())['completed_steps']==500


def test_progress_publication_does_not_change_prepared_learning(tmp_path,monkeypatch):
    now=[0.];monkeypatch.setattr(time,'perf_counter',lambda:now[0])
    results=[];original=TrainingClock.publish
    for enabled in (True,False):
        folder=tmp_path/str(enabled);folder.mkdir()
        def publish(clock):
            if not enabled:clock.last_publish=None
            original(clock)
        monkeypatch.setattr(TrainingClock,'publish',publish)
        results.append(run_training_phases(timed_module(now),loop_config(),folder,folder/'checkpoint',preflight_steps=2)[0])
    assert results[0]==results[1]


@pytest.mark.parametrize('side,valid',[(1,True),(32,True),(42,True),(43,True),(44,False),(48,False),(128,False)])
def test_complete_disjoint_geometry_boundaries(side,valid):
    if valid:assert disjoint_copy_domain(128,128,[side])
    else:
        with pytest.raises(CandidateContractError,match='no nonoverlapping donor'):disjoint_copy_domain(128,128,[side])


def test_restricted_geometry_and_saved_primitive_preflight(tmp_path):
    assert disjoint_copy_domain(128,128,[48],destination_y=[0,32],destination_x=[0,80])
    with pytest.raises(CandidateContractError):disjoint_copy_domain(128,128,[48],destination_y=[40,40],destination_x=[38,38])
    source=tmp_path/'model.py';source.write_text('''
def disjoint_origin(side,y,x,u):
    if not 16 <= side <= 48:raise ValueError()
    yy,xx=np.mgrid[:129-side,:129-side]
    return yy,xx
''')
    before=source.read_bytes()
    with pytest.raises(CandidateContractError,match='side=44'):validate_candidate_geometry(source)
    assert source.read_bytes()==before


def test_missing_controls_and_frozen_not_applicable():
    plan=future_plan();plan['confirmation_requirements']=None
    assert verify_pairs(plan,[])['status']=='NOT_MEASURED'
    for arm in plan['arms']:arm['execution']='frozen'
    assert verify_pairs(plan,[])['status']=='NOT_APPLICABLE'


@pytest.fixture
def ledger(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite');campaigns=CampaignState(store)
    cid=campaigns.create(config());campaigns.set_status(cid,'STOPPED','fixture complete');other=campaigns.create(config())
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path));monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    return store,campaigns,cid,other


def event(store,cid,kind,payload):
    with store.connect() as con:con.execute('INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,1,?,?)',(cid,kind,json.dumps(payload)))


def test_more_than_100_evaluations_and_campaign_filter(ledger):
    store,campaigns,cid,other=ledger
    ids=[]
    for i in range(105):
        rid=store.create_run('h','candidate','juice_bottle',config={'seed':i});ids.append(rid)
        campaigns.attach_run(cid if i==104 else other,rid)
        with store.connect() as con:con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,1,'REAL_DEVELOPMENT_FEEDBACK',?)",(rid,json.dumps({'metrics':{},'evaluator_version':'compatible'})))
    assert len(real_feedback_summaries())==105
    assert [r['run_id'] for r in real_feedback_summaries(cid)]==[ids[-1]]
    assert data.model_results(cid)[0]['config']['seed']==104


def test_comparison_scope_resumed_critique_and_missing_cells(ledger):
    store,campaigns,cid,other=ledger
    p={'plan_id':'p','hypothesis_id':'h','kind':'controlled_comparison','baseline_arm_id':'a',
       'arms':[{'arm_id':'a','execution':'train','difference':'baseline'},{'arm_id':'b','execution':'train','difference':'intervention'}],'seeds':[7,19,31]}
    event(store,cid,'EXPERIMENT_PLAN_REGISTERED',{'plan':p})
    rid=store.create_run('h','candidate','juice_bottle',config={'plan_id':'p','plan_arm':'a','seed':7})
    campaigns.attach_run(cid,rid)
    event(store,cid,'EXPERIMENT_PLAN_RESULT',{'result':{'plan_id':'p','status':'INCOMPLETE'}})
    event(store,other,'CRITIQUE',{'evidence_campaign_id':cid,'plan_id':'p','review':{'hypothesis_status':'inconclusive'}})
    record=data.comparison_records(cid)[0]
    assert record['status']=='Incomplete' and record['review']['hypothesis_status']=='inconclusive'
    matrix=data.comparison_matrix(record)
    assert len(matrix)==6 and sum(r['Status']=='MISSING' for r in matrix)==5
    assert data.comparison_records(other)==[]


def test_artifact_states_and_authorization_are_distinct(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    summary={'run_id':'r','evaluation_id':'e','artifact_availability':{'status':'ARCHIVED_METRICS_ONLY'}}
    assert data.artifact_status(summary)[0]=='Archived'
    summary.pop('artifact_availability')
    assert data.artifact_status(summary)[0]=='Missing unexpectedly'
    summary.pop('evaluation_id');assert data.artifact_status(summary)[0]=='Not generated'
    for state in ('LIMIT_REACHED','STOPPED','COMPLETED'):
        assert data.authorization_state({'status':state,'config':{}},False)[0]=='Inactive'
    assert data.authorization_state({'status':'RUNNING','config':{'security':{'accepted':True}}})[0]=='Active authorization'


def test_populated_campaign_pages_render_and_filter(ledger,monkeypatch):
    store,campaigns,cid,other=ledger
    p={'plan_id':'p','hypothesis_id':'h','kind':'controlled_comparison','baseline_arm_id':'a',
       'arms':[{'arm_id':'a','execution':'train','difference':'baseline'}],'seeds':[7]}
    event(store,other,'EXPERIMENT_PLAN_REGISTERED',{'plan':p})
    event(store,other,'EXPERIMENT_PLAN_RESULT',{'result':{'plan_id':'p','status':'COMPLETE_EXPLORATORY'}})
    event(store,other,'CRITIQUE',{'plan_id':'p','review':{'summary':'Actual recorded scientific critique'}})
    from dashboard.data_access import service_status
    monkeypatch.setattr('dashboard.data_access.service_status',lambda:{'mlflow':True,'dashboard':True,'tracking':True})
    for name in ('01_Home.py','02_Research.py','03_Results.py','04_System.py'):
        entry=Path(__file__).parents[1]/('dashboard/app.py' if name=='01_Home.py' else 'dashboard/pages/'+name)
        app=AppTest.from_file(entry,default_timeout=15).run()
        assert not app.exception,name
        if name=='02_Research.py':
            assert any('Actual recorded' in m.value for m in app.markdown)
            assert len(app.expander)<10
            app.sidebar.selectbox[0].select(cid).run()
            assert not app.exception and any('No comparisons' in m.value for m in app.info)


def test_geometry_fails_actual_plan_before_dispatch(laboratory):
    from orchestrator.experiment_plan import execute_plan
    store,campaigns,cid,runtime,repo,calls,_=laboratory
    source=repo/'candidates/fixture/model.py'
    source.write_text(source.read_text()+"\nSYNTHETIC_EDIT_GEOMETRY=[{'kind':'disjoint_square_copy','height':128,'width':128,'sides':[48]}]\n")
    result=execute_plan(store,campaigns,cid,runtime,experiment_plan())
    assert result['status']=='INCOMPLETE' and not calls
    assert 'impossible disjoint-copy domain' in str(result['problems'])


def test_complete_geometry_proof_matches_exhaustive_integer_fixture():
    for size in range(2,10):
        for side in range(1,size+1):
            expected=all(any(sy+side<=y or sy>=y+side or sx+side<=x or sx>=x+side
                for sy in range(size-side+1) for sx in range(size-side+1))
                for y in range(size-side+1) for x in range(size-side+1))
            try:actual=disjoint_copy_domain(size,size,[side])
            except CandidateContractError:actual=False
            assert bool(actual)==expected


@pytest.mark.parametrize('declaration',[None,{}, {'kind':'disjoint_square_copy','height':128,'width':128,'sides':2},
    {'kind':'disjoint_square_copy','height':128,'width':128,'sides':[32],'destination_y':[0]}])
def test_invalid_geometry_declarations_are_repairable_contract_errors(tmp_path,declaration):
    source=tmp_path/'model.py';source.write_text('SYNTHETIC_EDIT_GEOMETRY='+repr([declaration]))
    with pytest.raises(CandidateContractError):validate_candidate_geometry(source)


def test_infrastructure_records_do_not_become_scientific_models(ledger):
    store,campaigns,cid,other=ledger
    rid=store.create_run('infrastructure-smoke','toy_normal_ae','juice_bottle')
    campaigns.attach_run(other,rid)
    with store.connect() as con:
        con.execute("UPDATE runs SET status='REVIEWED',started_at=1 WHERE run_id=?",(rid,))
        con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,1,'REAL_DEVELOPMENT_FEEDBACK','{\"metrics\":{}}')",(rid,))
    runs=data.campaign_runs(other)
    stats=data.campaign_stats(campaign_summary(other),runs,[])
    assert stats['jobs']==1 and stats['trained']==stats['reviewed']==0
    assert data.model_results(other)==[]
