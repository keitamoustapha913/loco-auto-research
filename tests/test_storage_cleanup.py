"""Artificial byte-for-byte scientific evidence; no research or real data."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from benchmark.feedback_artifacts import load_map, map_bytes, map_sha256
from orchestrator import storage_cleanup as sc
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.development_feedback import verify_feedback_cache


@pytest.fixture
def lab(tmp_path):
    store=StateStore(tmp_path/'controller/experiments.sqlite');CampaignState(store)
    def add(name, values=None):
        rid=store.create_run(name,name,'juice_bottle')
        output=tmp_path/'jobs'/rid/'feedback/eval';case=output/'case_0000';case.mkdir(parents=True)
        array=np.zeros((128,128),np.float32) if values is None else values
        np.save(case/'map.npy',array)
        sha=hashlib.sha256((case/'map.npy').read_bytes()).hexdigest()
        scores=json.dumps([{'case_index':0,'map_sha256':sha}]).encode()
        (output/'case_scores.json').write_bytes(scores);(output/'localization.json').write_text('{}')
        summary=dict(evaluation_id='eval',cache_identity=name,checkpoint_unchanged=True,
            artifact_dir=str(output),counts={'images':1},cache_artifact_hashes={'case_scores.json':hashlib.sha256(scores).hexdigest()})
        with store.connect() as con:
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)",(rid,time.time(),'REAL_DEVELOPMENT_FEEDBACK',json.dumps(summary)))
        checkpoint=output.parents[1]/'checkpoint';checkpoint.mkdir()
        (checkpoint/'model.pt').write_bytes(b'unique trained model '+name.encode())
        return rid,output,summary,case/'map.npy'
    return store,tmp_path,add


def test_preview_is_read_only_and_duplicate_cleanup_preserves_every_byte(lab):
    store,root,add=lab
    _,out,a,path=add('first');_,other,b,duplicate=add('second')
    raw=path.read_bytes();checkpoint=(out.parents[1]/'checkpoint/model.pt').read_bytes()
    with store.connect() as con:before=[tuple(r) for r in con.execute('select * from runs')]
    view=sc.cleanup(store,root,dry_run=True)
    assert len(view['duplicates'])==1 and path.exists() and duplicate.exists()
    report=sc.cleanup(store,root)
    assert report['actions']==2 and report['reclaimed_bytes']>len(raw)
    assert not path.exists() and not duplicate.exists()
    assert Path(str(path)+'.bs4.gz').is_file() and Path(str(duplicate)+'.ref.json').is_file()
    assert map_bytes(path,root/'jobs')==map_bytes(duplicate,root/'jobs')==raw
    assert np.array_equal(load_map(duplicate,root/'jobs'),np.zeros((128,128),np.float32))
    assert verify_feedback_cache(a,'first',root/'jobs')==out
    assert verify_feedback_cache(b,'second',root/'jobs')==other
    assert (out.parents[1]/'checkpoint/model.pt').read_bytes()==checkpoint
    with store.connect() as con:assert [tuple(r) for r in con.execute('select * from runs')]==before
    assert sc.cleanup(store,root)['actions']==0


def test_protects_active_jobs_and_frozen_baseline_inputs(lab):
    store,root,add=lab
    rid,out,_,path=add('active');baseline,_,_,frozen=add('historical')
    with store.connect() as con:
        con.execute("UPDATE runs SET status='RUNNING',config_json=? WHERE run_id=?",(json.dumps({'frozen_baseline':{'run_id':baseline}}),rid))
    assert sc.cleanup(store,root)['actions']==0
    assert path.exists() and frozen.exists()


def test_live_locks_block_cleanup_and_never_block_training(lab):
    store,root,add=lab;_,_,_,path=add('model')
    for name in ('gpu.lock','feedback.lock','storage_cleanup.lock'):
        with (root/'controller'/name).open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert sc.cleanup(store,root)['busy']
            assert path.exists()


def test_active_readout_protects_its_transitive_prediction_directory(lab):
    store,root,add=lab
    active,_,_,current=add('active');_,origin,_,historical=add('historical')
    with store.connect() as con:
        con.execute("UPDATE runs SET status='RUNNING' WHERE run_id=?",(active,))
        row=con.execute("SELECT id,payload_json FROM events WHERE run_id=? AND kind='REAL_DEVELOPMENT_FEEDBACK'",(active,)).fetchone()
        summary=json.loads(row['payload_json']);summary['prediction_artifact_dir']=str(origin)
        con.execute('UPDATE events SET payload_json=? WHERE id=?',(json.dumps(summary),row['id']))
    assert sc.cleanup(store,root)['actions']==0 and current.exists() and historical.exists()


def test_hash_mismatch_and_symlink_preserve_ambiguous_evidence(lab):
    store,root,add=lab;_,out,summary,path=add('model')
    path.write_bytes(b'changed unique evidence')
    assert sc.cleanup(store,root)['errors'] and path.read_bytes()==b'changed unique evidence'
    outside=root/'outside';outside.write_bytes(b'never touch')
    path.unlink();path.symlink_to(outside)
    assert not sc.preview(store,root)['lossless_pack_candidates']
    assert sc.cleanup(store,root)['actions']==0 and outside.read_bytes()==b'never touch'


def test_reference_escape_cycle_and_corrupt_compression_rejected(lab):
    store,root,add=lab;_,_,_,path=add('model');path.unlink()
    ref=Path(str(path)+'.ref.json')
    ref.write_text(json.dumps(dict(version=1,target='../outside',sha256='a'*64,bytes=1)))
    with pytest.raises(ValueError,match='unsafe'):map_bytes(path,root/'jobs')
    ref.write_text(json.dumps(dict(version=1,target=str(path.relative_to(root/'jobs')),sha256='a'*64,bytes=1)))
    with pytest.raises(ValueError,match='cyclic'):map_bytes(path,root/'jobs')
    ref.unlink();Path(str(path)+'.gz').write_bytes(b'not gzip')
    with pytest.raises(OSError):map_bytes(path,root/'jobs')


def test_interruption_after_replacement_and_after_unlink_is_recoverable(lab,monkeypatch):
    store,root,add=lab;_,_,_,path=add('model');raw=path.read_bytes()
    original=sc._remove
    def interrupt(*args,**kwargs):raise OSError('simulated interruption before unlink')
    monkeypatch.setattr(sc,'_remove',interrupt)
    assert sc.cleanup(store,root)['errors'] and path.read_bytes()==raw
    assert Path(str(path)+'.bs4.gz').exists()
    monkeypatch.setattr(sc,'_remove',original)
    event=sc._event
    def crash(store,kind,**payload):
        if kind=='STORAGE_CLEANUP_DELETED':raise OSError('simulated crash after unlink')
        return event(store,kind,**payload)
    monkeypatch.setattr(sc,'_event',crash)
    assert sc.cleanup(store,root)['errors'] and not path.exists()
    assert map_bytes(path,root/'jobs')==raw
    monkeypatch.setattr(sc,'_event',event)
    sc.cleanup(store,root)
    with store.connect() as con:
        records=[json.loads(r[0]) for r in con.execute("select payload_json from events where kind='STORAGE_CLEANUP_DELETED'")]
    assert any(r.get('recovered_after_restart') for r in records)


def test_expired_owned_staging_removed_but_recent_unknown_and_referenced_preserved(lab):
    store,root,add=lab;_,out,_,_=add('model')
    stale=out.parent/'.feedback-reuse-expired/result';stale.mkdir(parents=True)
    old=stale/'summary.json';old.write_text('{}');os.utime(old,(0,0))
    recent=out.parent/'.feedback-reuse-recent';recent.mkdir();(recent/'summary.json').write_text('{}')
    unknown=out.parent/'.feedback-reuse-ambiguous';unknown.mkdir();(unknown/'checkpoint.pt').write_bytes(b'unknown evidence');os.utime(unknown/'checkpoint.pt',(0,0))
    kept=out.parent/'.feedback-reuse-referenced';kept.mkdir();(kept/'summary.json').write_text('{}');os.utime(kept/'summary.json',(0,0))
    with store.connect() as con:con.execute('INSERT INTO settings VALUES(?,?)',('important_reference',str(kept)))
    sc.cleanup(store,root)
    assert not stale.parent.exists() and recent.exists() and unknown.exists() and kept.exists()


def test_periodic_checks_are_throttled_and_restart_uses_persistent_schedule(lab,monkeypatch):
    store,root,add=lab;add('model');clock=[time.time()]
    monkeypatch.setattr(sc.time,'time',lambda:clock[0])
    monkeypatch.setattr(sc,'inventory',lambda _:{'limit_bytes':100,'observed_bytes':99})
    calls=[]
    def clean(*a,**kw):
        calls.append(1)
        with store.connect() as con:con.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',('storage_cleanup_last',str(clock[0])))
        return dict(busy=False)
    monkeypatch.setattr(sc,'cleanup',clean);sc._next_check.clear()
    sc.maybe_cleanup(root,store);sc.maybe_cleanup(root,store)
    assert len(calls)==1
    sc._next_check.clear();clock[0]+=1700;sc.maybe_cleanup(root,store)
    assert len(calls)==1
    clock[0]+=101;sc.maybe_cleanup(root,store)
    assert len(calls)==2


def test_quota_admission_reclaims_and_rechecks_without_expanding_limit(lab,monkeypatch):
    from orchestrator.budgets import enforce_disk,DiskBudgetExceeded
    store,root,add=lab;add('one',np.zeros((512,512),np.float32));add('two',np.zeros((512,512),np.float32))
    before=sc.inventory(root)['observed_bytes'];limit=before-500000
    monkeypatch.setenv('LOCO_DISK_BUDGET_GIB',str(limit/2**30))
    with pytest.raises(DiskBudgetExceeded):enforce_disk(root,cleanup=False)
    after=enforce_disk(root)
    assert after['observed_bytes']<after['limit_bytes']==limit
    assert sc.cleanup(store,root)['actions']==0


def test_lossless_packing_preserves_localization_scores(tmp_path):
    from benchmark.feedback_metrics import saturated_region_curve
    import gzip
    path=tmp_path/'map.npy';values=np.arange(64,dtype=np.float32).reshape(8,8)/64
    np.save(path,values);raw=path.read_bytes()
    Path(str(path)+'.gz').write_bytes(gzip.compress(raw));path.unlink()
    actual=load_map(path,tmp_path)
    assert np.array_equal(actual,values)
    def metric(array):return saturated_region_curve(array[:4].ravel(),[(array[4:].ravel(),8)])
    assert metric(actual)==metric(values)


def test_legacy_map_hashes_allow_lossless_storage_without_rewriting_history(lab):
    store,root,add=lab;rid,out,_,path=add('legacy')
    with store.connect() as con:
        row=con.execute("select id,payload_json from events where run_id=? AND kind='REAL_DEVELOPMENT_FEEDBACK'",(rid,)).fetchone()
        old=json.loads(row['payload_json']);old.pop('cache_identity');old.pop('cache_artifact_hashes')
        saved=json.dumps(old);con.execute('update events set payload_json=? where id=?',(saved,row['id']))
    raw=path.read_bytes();scores=(out/'case_scores.json').read_bytes()
    assert sc.cleanup(store,root)['actions']==1
    assert map_bytes(path,root/'jobs')==raw and (out/'case_scores.json').read_bytes()==scores
    with store.connect() as con:assert con.execute('select payload_json from events where id=?',(row['id'],)).fetchone()[0]==saved


def test_controller_lifecycle_triggers_start_comparison_and_finish(lab,monkeypatch):
    from orchestrator import autonomy
    from test_time_campaigns import simple
    store,root,add=lab;campaigns=CampaignState(store);cid=campaigns.create(simple('new'))
    triggers=[]
    monkeypatch.setattr(sc,'maybe_cleanup',lambda *a,**kw:triggers.append(kw.get('reason')))
    monkeypatch.setattr(autonomy,'preflight',lambda *a:None)
    monkeypatch.setattr(autonomy,'reconcile_unfinished_runs',lambda *a:None)
    monkeypatch.setattr(autonomy,'recover_research_attempt',lambda *a:False)
    monkeypatch.setattr(autonomy,'stop_reason',lambda *a:(None,''))
    monkeypatch.setattr(autonomy,'run_cycle',lambda *a:campaigns.set_status(cid,'COMPLETED','fixture completed'))
    autonomy.run_forever(cid,root,store.path,idle_seconds=0)
    assert triggers==['controller startup','comparison completed or abandoned','campaign finished']


def test_linked_jobs_root_or_lock_never_followed(lab):
    store,root,add=lab
    outside=root/'outside';outside.mkdir();(root/'jobs').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='unsafe linked'):sc.cleanup(store,root)
    assert not list(outside.iterdir())


def test_duplicate_reference_preserves_origin_bytes_and_mlflow_metadata(lab):
    store,root,add=lab;_,origin,a,path=add('original');_,out,b,copy_path=add('copy')
    metadata=(out/'case_scores.json').read_bytes()
    mlflow=root/'mlflow/artifacts';mlflow.mkdir(parents=True);artifact=mlflow/'case_scores.json';artifact.write_bytes(metadata)
    sc.cleanup(store,root)
    assert (out/'case_scores.json').read_bytes()==artifact.read_bytes()==metadata
    assert verify_feedback_cache(b,'copy',root/'jobs')==out
    assert map_sha256(copy_path,root/'jobs')==map_sha256(path,root/'jobs')


def test_incompressible_maps_are_not_repeatedly_read_or_compressed(lab,monkeypatch):
    store,root,add=lab
    _,_,_,path=add('random',np.random.default_rng(1).integers(0,256,(512,512),dtype=np.uint8))
    assert sc.cleanup(store,root)['actions']==0 and path.exists()
    def fail(*a,**k):raise AssertionError('unchanged incompressible map must not be read again')
    monkeypatch.setattr(sc,'map_bytes',fail)
    assert sc.cleanup(store,root)['actions']==0


def test_failed_unlink_is_not_double_counted_after_retry(lab,monkeypatch):
    store,root,add=lab;_,_,_,path=add('model')
    unlink=Path.unlink
    def reject(self,*a,**k):
        if self==path:raise PermissionError('fixture unlink denied')
        return unlink(self,*a,**k)
    monkeypatch.setattr(Path,'unlink',reject)
    assert sc.cleanup(store,root)['errors'] and path.exists()
    monkeypatch.setattr(Path,'unlink',unlink)
    assert sc.cleanup(store,root)['actions']==1
    sc.cleanup(store,root)
    with store.connect() as con:
        assert con.execute("select count(*) from events where kind='STORAGE_CLEANUP_DELETED'").fetchone()[0]==1
        assert con.execute("select count(*) from events where kind='STORAGE_CLEANUP_ABORTED'").fetchone()[0]==1
