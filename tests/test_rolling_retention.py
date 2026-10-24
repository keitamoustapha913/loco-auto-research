"""No scientific inference: deterministic ledger and byte fixtures."""
import hashlib
import json
from pathlib import Path

import pytest
from orchestrator import rolling_retention as rr, model_registry as mr, storage_cleanup as sc
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState


@pytest.fixture
def history(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'controller/experiments.sqlite');CampaignState(store)
    models={};ids=[]
    monkeypatch.setattr(mr,'retained',lambda s:models)
    monkeypatch.setattr(mr,'retained_category',lambda s,m:'juice_bottle')
    monkeypatch.setattr(mr,'verify_archive',lambda m:None)
    monkeypatch.setattr(mr,'research_states',lambda s:{'cohort':{'retention_id':ids[0]}})
    monkeypatch.setattr(mr,'states',lambda s:{})
    for i in range(12):
        rid=store.create_run('hyp'+str(i),'candidate'+str(i),'juice_bottle');ids.append(rid)
        root=tmp_path/'jobs'/rid;root.mkdir(parents=True)
        (root/'checkpoint').mkdir();(root/'checkpoint/model.pt').write_bytes(b'checkpoint'+bytes([i]))
        (root/'run_manifest.json').write_text('{"dataset":"normal"}')
        (root/'source.py').write_text('original scientific source')
        (root/'case_scores.json').write_text('[{"ap":0.5}]')
        (root/'heatmap.png').write_bytes(b'visual')
        archive=tmp_path/'controller/retained_models'/rid;archive.mkdir(parents=True)
        (archive/'model.pt').write_bytes(b'checkpoint'+bytes([i]));(archive/'model.py').write_text('source')
        with store.connect() as con:
            con.execute("UPDATE runs SET status='REVIEWED',created_at=?,artifact_dir=?,config_json=?,metrics_json=? WHERE run_id=?",
                        (i,str(root),'{}','{"scientific_metric":0.5}',rid))
        models[rid]={'retention_id':rid,'run_id':rid,'archive':str(archive),'created_at':i,
                     'metrics':{mr.SPRO:1-i*.01,mr.AP:.6,mr.FPR:.1,mr.LATENCY:5,mr.PARAMETERS:100,mr.MEMORY:100}}
    return store,tmp_path,ids,models


def test_six_best_three_recent_and_incumbent(history):
    store,root,ids,_=history
    p=rr.preview(store,root)
    assert p['selected']==ids[:6]+list(reversed(ids[9:]))
    assert {r['run_id'] for r in p['candidates']}==set(ids[6:9])
    assert len(p['selected'])==9


def test_retirement_preserves_numerical_history_source_and_hashes(history):
    store,root,ids,models=history
    before=[dict(r) for r in store.list_runs()]
    r=rr.retire(store,root)
    assert r['actions']==9 and not r['errors']
    for rid in ids[6:9]:
        job=root/'jobs'/rid
        assert not (job/'checkpoint/model.pt').exists()
        assert not (Path(models[rid]['archive'])/'model.pt').exists()
        assert (job/'source.py').read_text()=='original scientific source'
        assert json.loads((job/'case_scores.json').read_text())==[{'ap':.5}]
        assert (job/'scientific_record.json').is_file()
        assert rr.availability(store,rid)['checkpoint_available'] is False
    assert [dict(r) for r in store.list_runs()]==before
    with store.connect() as con:
        audit=[json.loads(r[0]) for r in con.execute("SELECT payload_json FROM events WHERE kind='ARTIFACT_RETIRED'")]
    assert len(audit)==9 and all(len(r['sha256'])==64 for r in audit)
    assert rr.retire(store,root)['actions']==0
    assert len(rr.selection(store)['selected'])==9


def test_active_comparison_protection_and_quota_not_disabled(history):
    store,root,ids,_=history
    with store.connect() as con:
        con.execute("UPDATE runs SET status='RUNNING',config_json=? WHERE run_id=?",(json.dumps({'frozen_baseline':{'run_id':ids[6]}}),ids[7]))
    p=rr.preview(store,root)
    assert ids[6] in p['protected'] and ids[7] in p['protected'] and p['conflicts']
    rr.retire(store,root)
    assert (root/'jobs'/ids[6]/'checkpoint/model.pt').exists()
    assert sc.inventory(root)['limit_bytes']==100*2**30


def test_symlink_is_preserved_and_reported(history):
    store,root,ids,_=history
    external=root/'never_delete.pt';external.write_bytes(b'essential')
    (root/'jobs'/ids[6]/'unsafe.pt').symlink_to(external)
    p=rr.preview(store,root)
    assert any(b['run_id']==ids[6] for b in p['blocked'])
    rr.retire(store,root)
    assert external.read_bytes()==b'essential'
    assert (root/'jobs'/ids[6]/'checkpoint/model.pt').exists()


def test_crash_after_prepared_retirement_resumes(history,monkeypatch):
    store,root,ids,_=history
    original=Path.unlink
    def crash(path,*args,**kwargs):
        if path.name=='model.pt':raise OSError('interrupted unlink')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',crash)
    assert rr.retire(store,root)['errors']
    assert rr.availability(store,ids[6])['status']=='ARCHIVED_METRICS_ONLY'
    monkeypatch.setattr(Path,'unlink',original)
    assert rr.retire(store,root)['actions']>0
    assert rr.retire(store,root)['actions']==0


def test_new_best_changes_selection_without_changing_champion_rules(history,monkeypatch):
    store,root,ids,models=history
    models[ids[8]]['metrics'][mr.SPRO]=1
    models[ids[8]]['metrics'][mr.AP]=1
    monkeypatch.setattr(mr,'research_states',lambda s:{'cohort':{'retention_id':ids[8]}})
    p=rr.selection(store)
    assert ids[8] in p['selected'] and len(p['selected'])==9


def test_archived_frozen_replication_rejected_explicitly():
    with pytest.raises(ValueError,match='intentionally archived'):
        mr.verify_archive({'artifact_availability':{'checkpoint_available':False}})


def test_selected_visuals_keep_canonical_maps_in_retired_run(history):
    store,root,ids,_=history
    original=root/'jobs'/ids[6]/'feedback/eval/case_0000/map.npy'
    original.parent.mkdir(parents=True);original.write_bytes(b'verified map bytes')
    selected=root/'jobs'/ids[0]/'feedback/eval';(selected/'case_0000').mkdir(parents=True)
    ref=selected/'case_0000/map.npy.ref.json'
    ref.write_text(json.dumps({'target':str(original.relative_to(root/'jobs')),'version':1,'sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'bytes':len(original.read_bytes())}))
    with store.connect() as con:
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,0,?,?)',
                    (ids[0],'REAL_DEVELOPMENT_FEEDBACK',json.dumps({'artifact_dir':str(selected)})))
    rr.retire(store,root)
    assert original.exists() and ref.exists()
    assert not (root/'jobs'/ids[6]/'checkpoint/model.pt').exists()


def test_local_mlflow_copies_retire_but_metrics_stay(history):
    store,root,ids,_=history
    mid='abcdef0123456789';art=root/'mlflow/artifacts/1'/mid/'artifacts';art.mkdir(parents=True)
    (art/'heatmap.png').write_bytes(b'duplicate visual');(art/'metrics.json').write_text('{"ap":0.5}')
    with store.connect() as con:con.execute('UPDATE runs SET mlflow_run_id=? WHERE run_id=?',(mid,ids[6]))
    rr.retire(store,root)
    assert not (art/'heatmap.png').exists() and (art/'metrics.json').is_file()


def test_completed_study_not_protected_for_entire_long_campaign(history):
    store,root,ids,_=history
    with store.connect() as con:
        con.execute("INSERT INTO campaigns(id,name,status,config_json,created_at,updated_at,baseline_gpu_seconds,baseline_codex_calls) VALUES('long','long','RUNNING','{}',0,0,0,0)")
        for rid in (ids[6],ids[7]):
            con.execute("UPDATE runs SET config_json=? WHERE run_id=?",(json.dumps({'campaign_id':'long','plan_id':'p'}),rid))
        con.execute("INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES('long',0,'EXPERIMENT_PLAN_REGISTERED',?)",(json.dumps({'plan':{'plan_id':'p','baseline':{'run_id':ids[6]}}}),))
    assert ids[6] in rr.selection(store)['protected'] and ids[7] in rr.selection(store)['protected']
    with store.connect() as con:
        con.execute("INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES('long',1,'EXPERIMENT_PLAN_RESULT',?)",(json.dumps({'result':{'plan_id':'p','status':'COMPLETE_EXPLORATORY'}}),))
    assert ids[6] not in rr.selection(store)['protected']
    assert ids[7] not in rr.selection(store)['protected']


def test_bulky_numerical_evidence_losslessly_compressed(history):
    import gzip
    store,root,ids,_=history
    path=root/'jobs'/ids[6]/'diagnostic_long_trace.json'
    original=json.dumps({'per_update':[{'loss':0.123,'source_digest':'a'*64}]*30000}).encode()
    path.write_bytes(original)
    report=rr.retire(store,root)
    assert not report['errors'] and not path.exists()
    assert gzip.decompress(Path(str(path)+'.gz').read_bytes())==original
    assert rr.read_text(path,max_bytes=len(original))==original.decode()
    assert rr.read_text(path,max_bytes=100) is None
    assert rr.retire(store,root)['actions']==0


def test_retirement_honors_existing_writer_locks(history):
    import fcntl
    store,root,ids,_=history
    with (root/'controller/tracking.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sc.cleanup(store,root)['busy']
    assert (root/'jobs'/ids[6]/'checkpoint/model.pt').exists()


def test_dashboard_history_stays_visible_with_archived_status(history,monkeypatch):
    from dashboard.data_access import list_runs
    store,root,ids,_=history
    monkeypatch.setenv('LOCO_RUNTIME',str(root));monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    rr.retire(store,root)
    rows=list_runs()
    assert len(rows)==12
    archived=next(r for r in rows if r['run_id']==ids[6])
    assert archived['artifact_availability']['status']=='ARCHIVED_METRICS_ONLY'
    assert archived['metrics']=={'scientific_metric':.5}


def test_prior_confirmed_checkpoint_remains_independently_protected(history):
    store,root,ids,_=history
    with store.connect() as con:
        con.execute("INSERT INTO events(time,kind,payload_json) VALUES(0,'CHAMPION_DECISION',?)",(json.dumps({'status':'PROMOTED','challenger':ids[7]}),))
    plan=rr.preview(store,root)
    assert ids[7] in plan['protected'] and plan['conflicts']
    rr.retire(store,root)
    assert (root/'jobs'/ids[7]/'checkpoint/model.pt').exists()
