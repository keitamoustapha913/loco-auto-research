"""Artificial ledger integration: no inference, CUDA, Codex or real-data access."""
import copy
import hashlib
import json
import subprocess

import pytest

from orchestrator import model_registry as registry
from orchestrator.state_store import StateStore
from orchestrator.campaign_state import CampaignState
from orchestrator.autonomy import load_campaign_config
from orchestrator.experiment_plan import register_plan, digest, comparison_result
from orchestrator.development_feedback import record_summary, latest_summary
from plan_fixtures import experiment_plan


@pytest.fixture
def lab(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    for name in ('baseline','challenger'):
        folder=repo/'candidates'/name;folder.mkdir(parents=True)
        (folder/'model.py').write_text('# artificial '+name+'\n')
    for args in (['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@localhost','commit','-qm','fixture']):
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    store=StateStore(tmp_path/'controller/experiments.sqlite');campaigns=CampaignState(store)
    cfg=load_campaign_config(registry.REPO/'configs/autonomy.yaml');cfg['limits']['max_experiments']=30
    cid=campaigns.create(cfg)
    store.add_proposal({'hypothesis_id':'fixture'})

    def add(seed=3,name='baseline',spro=.2,ap=.2,plan=None,arm='control',**overrides):
        source=(repo/'candidates'/name/'model.py').read_bytes()
        p=plan or experiment_plan()
        config=dict(p['training'],seed=seed,model_sha256=hashlib.sha256(source).hexdigest(),
                    inference={'mode':arm},diagnostics=p['diagnostics'],execution_kind='train')
        if plan:config.update(plan_sha256=digest(plan),plan_arm=arm,plan_id=plan['plan_id'],campaign_id=cid)
        rid=store.create_run('fixture',name,'juice_bottle',config=config,git_sha=commit,manifest_sha256='normal-only-fixture')
        folder=tmp_path/'jobs'/rid;checkpoint=folder/'checkpoint';checkpoint.mkdir(parents=True)
        # Same method/seed is reproducible, different seeds have different learned bytes.
        weights=f'{name}-{seed}'.encode();cp_hash=hashlib.sha256(weights).hexdigest()
        (checkpoint/'model.pt').write_bytes(weights)
        (checkpoint/'checkpoint_manifest.json').write_text('{}')
        (folder/'run_manifest.json').write_text(json.dumps({'checkpoint_sha256':cp_hash,'good_manifest_sha256':'normal-only-fixture'}))
        for state in ('IMPLEMENTED','VALIDATED','QUEUED','RUNNING'):store.transition(rid,state)
        store.set_run_result(rid,{'normal/image_false_positives':0,'protocol/seed':seed,
            'protocol/source_exposures':4,'protocol/optimizer_updates':2,'diagnostic/fixture_check/mean_score':1},
            gpu_seconds=.1,artifact_dir=str(folder))
        store.transition(rid,'COMPLETED')
        output=folder/'feedback'/'fixture';output.mkdir(parents=True)
        cases=[{'category':category,'source_sha256':str(i)} for i,category in enumerate(
            ['good','good','logical_anomalies','structural_anomalies'])]
        (output/'case_scores.json').write_text(json.dumps(cases))
        metrics={registry.SPRO:spro,registry.AP:ap,registry.FPR:0.,registry.LATENCY:10.,registry.PARAMETERS:100.,registry.MEMORY:50.,
                 'dev_real/normal_false_positives':0,'dev_real/normal_images':2,'dev_real/normal_only_threshold':.3,
                 'dev_real/logical_image_auroc':.7,'dev_real/structural_image_auroc':.7,
                 'dev_real/structural_au_spro_0_05_compatible':.3}
        metrics.update(overrides.pop('metrics',{}))
        summary={'evaluation_id':'evaluation-'+rid,'status':'COMPLETE_COMPATIBLE_DEVELOPMENT','checkpoint_unchanged':True,
                 'checkpoint_sha256':cp_hash,'source_sha256':hashlib.sha256(source).hexdigest(),
                 'training_manifest_sha256':'normal-only-fixture','split_id':'artificial-development-only',
                 'evaluator_version':'fixture-v1','inference_device':'cpu','measurement_protocol':'fixture batch one',
                 'threshold_source':'normal validation only','metrics':metrics,'artifact_dir':str(output),
                 'counts':{'images':4,'categories':{'good':2,'logical_anomalies':1,'structural_anomalies':1}}}
        summary.update(overrides);record_summary(store,rid,summary)
        return rid
    return store,campaigns,cid,repo,add


def prepared(lab,gain=.1,**change):
    store,c,cid,repo,add=lab
    old=add();registry.initialize(store,repo)
    p=experiment_plan()
    p['required_measurements'] += [{'source':'real','key':k} for k in
        (registry.SPRO,registry.AP,registry.FPR,registry.LATENCY,registry.PARAMETERS,registry.MEMORY)]
    p['evaluation']['real_development_feedback']=True
    register_plan(store,c,cid,p)
    ids=[]
    for arm in ('control','intervention'):
        for seed in p['seeds']:
            ids.append(add(seed=seed,name='baseline' if arm=='control' else 'challenger',
                spro=.2+(gain if arm=='intervention' else 0),ap=.2+(gain if arm=='intervention' else 0),
                plan=p,arm=arm,**(change if arm=='intervention' else {})))
    rows=c.runs(cid);feedback={r['run_id']:latest_summary(store,r['run_id']) for r in rows}
    result=comparison_result(p,rows,feedback)
    assert result['status']=='COMPLETE_CONTROLLED'
    return p,result,old,ids


def test_historical_incumbent_survives_worse_newer_and_restart(lab):
    store,c,cid,repo,add=lab
    best=add(spro=.4,ap=.3);registry.initialize(store,repo)
    add(name='challenger',spro=.1,ap=.1);registry.initialize(store,repo)
    progress=registry.progress(StateStore(store.path,read_only=True))
    assert progress['cohorts'][0]['champion']['run_id']==best
    assert progress['cohorts'][0]['state']['confirmation']=='HISTORICAL_EXPLORATORY'
    assert progress['last_verified_improvement_at'] is None


def test_complete_matched_multiseed_improvement_promotes_and_rollback_preserves(lab):
    store,c,cid,repo,add=lab;p,result,old,ids=prepared(lab)
    old_head=copy.deepcopy(registry.states(store))
    decisions=registry.consider_comparison(store,c,cid,p,result,repo)
    assert decisions[0]['status']=='PROMOTED',decisions
    model=registry.progress(store)['cohorts'][0]
    assert model['state']['confirmation']=='REPEATED_SEED_SUPPORTED'
    assert model['champion']['run_id']==ids[3] # fixed first seed, not cherry-picked best
    assert decisions[0]['intervals'][registry.SPRO]['lower']>0
    assert len(model['state']['confirmation_run_ids'])==3
    # Replay cannot re-promote; every retained checkpoint remains available.
    assert registry.consider_comparison(store,c,cid,p,result,repo)==decisions
    restored=registry.rollback(store,model['cohort_id'],'operator fixture recovery')
    assert restored['retention_id']==old_head[model['cohort_id']]['retention_id']
    assert registry.retained(store)[restored['retention_id']]['run_id']==old


@pytest.mark.parametrize('problem',['worse','incomplete','single_seed','fpr','latency','pixel_ap','cohort','baseline','changed_policy','detection','structural'])
def test_rejected_challenger_never_replaces_incumbent(lab,problem,monkeypatch):
    store,c,cid,repo,add=lab
    kwargs={}
    if problem=='fpr':kwargs['metrics']={registry.FPR:.5,'dev_real/normal_false_positives':1}
    if problem=='latency':kwargs['metrics']={registry.LATENCY:20.}
    if problem=='pixel_ap':kwargs['metrics']={registry.AP:.1}
    if problem=='detection':kwargs['metrics']={'dev_real/logical_image_auroc':.4}
    if problem=='structural':kwargs['metrics']={'dev_real/structural_au_spro_0_05_compatible':.1}
    if problem=='cohort':kwargs['evaluator_version']='incompatible-v2'
    p,result,old,ids=prepared(lab,gain=-.1 if problem=='worse' else .1,**kwargs)
    head=copy.deepcopy(registry.states(store))
    if problem=='incomplete':
        store.transition(ids[-1],'REVIEWED');store.transition(ids[-1],'REJECTED')
        # A missing required measurement is not a successful scientific result.
        with store.connect() as con:
            row=store.get_run(ids[-1]);m=json.loads(row['metrics_json']);m.pop('protocol/optimizer_updates')
            con.execute('UPDATE runs SET metrics_json=? WHERE run_id=?',(json.dumps(m),ids[-1]))
    elif problem=='single_seed':
        rules=registry.policy();rules['minimum_seeds']=4
        monkeypatch.setattr(registry,'policy',lambda:rules)
    elif problem=='baseline':
        with store.connect() as con:
            row=store.get_run(ids[0]);cfg=json.loads(row['config_json']);cfg['optimizer']='unmatched'
            con.execute('UPDATE runs SET config_json=? WHERE run_id=?',(json.dumps(cfg),ids[0]))
    elif problem=='changed_policy':
        rules=registry.policy();rules['minimum_spro_gain']=.5
        monkeypatch.setattr(registry,'policy',lambda:rules)
    d=registry.consider_comparison(store,c,cid,p,result,repo)
    assert d[0]['status']=='RETAIN_CHAMPION',d
    for cid_,prior in head.items():assert registry.states(store)[cid_]==prior


def test_archive_tamper_and_false_positive_denominator_rejected(lab):
    store,c,cid,repo,add=lab;rid=add();registry.initialize(store,repo)
    model=next(iter(registry.retained(store).values()))
    from pathlib import Path
    checkpoint=Path(model['archive'])/'model.pt';checkpoint.chmod(0o600);checkpoint.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='integrity'):registry.verify_archive(model)
    bad=add(name='challenger',metrics={registry.FPR:.3})
    with pytest.raises(ValueError,match='denominator'):registry.retain(store,bad,latest_summary(store,bad),repo)


def test_incompatible_evaluators_and_samples_have_separate_champions(lab):
    store,c,cid,repo,add=lab
    add(spro=.3);add(name='challenger',spro=.9,evaluator_version='fixture-v2')
    registry.initialize(store,repo)
    assert len(registry.states(store))==2
    for group in registry.progress(store)['cohorts']:
        assert len({m['conditions']['evaluator_version'] for m in group['models']})==1


def test_whole_history_memory_exceeds_five_and_is_cross_campaign(lab):
    store,c,cid,repo,add=lab
    for i in range(8):store.add_proposal({'hypothesis_id':f'failed_{i}','research_question':f'question {i}'})
    add();registry.initialize(store,repo)
    c.set_status(cid,'COMPLETED');c.create(load_campaign_config(registry.REPO/'configs/autonomy.yaml'))
    memory=registry.scientific_memory(store)
    assert len(memory['hypothesis_index'])==9
    assert memory['cohorts'][0]['champion']['configuration']['seed']==3
    assert memory['global_gpu_seconds']==pytest.approx(.1)


def test_pareto_preserves_localization_false_positive_tradeoff(lab):
    store,c,cid,repo,add=lab
    add(spro=.4,ap=.3,metrics={registry.FPR:.5,'dev_real/normal_false_positives':1})
    good=add(name='challenger',spro=.3,ap=.2)
    registry.initialize(store,repo);group=registry.progress(store)['cohorts'][0]
    assert len(group['pareto_retention_ids'])==2
    assert group['champion']['run_id']==good
    assert group['best_observed_spro']==.4


def test_concurrent_promotion_and_restart_are_idempotent(lab):
    from concurrent.futures import ThreadPoolExecutor
    store,c,cid,repo,add=lab;p,result,old,ids=prepared(lab)
    registry.initialize(store,repo)
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers=list(pool.map(lambda _:registry.consider_comparison(store,c,cid,p,result,repo),range(2)))
    assert answers[0]==answers[1]
    assert len(registry.records(store,'CHAMPION_DECISION'))==1
    fresh=StateStore(store.path)
    assert registry.progress(fresh)['cohorts'][0]['state']['confirmation']=='REPEATED_SEED_SUPPORTED'


def test_promotion_requires_preregistered_metrics_and_cannot_use_changed_head(lab):
    store,c,cid,repo,add=lab;p,result,old,ids=prepared(lab)
    state=registry.states(store);cid_,head=next(iter(state.items()))
    # Simulate an intervening operator recovery/accepted comparison.
    head['revision']+=1
    with store.connect() as con:con.execute('UPDATE settings SET value=? WHERE name=?',(json.dumps(head),registry.PREFIX+cid_))
    d=registry.consider_comparison(store,c,cid,p,result,repo)
    assert d[0]['status']=='RETAIN_CHAMPION'
    assert any('not pinned' in reason for reason in d[0]['reasons'])


def test_model_retention_does_not_rewrite_runs_or_feedback(lab):
    store,c,cid,repo,add=lab;rid=add()
    before=store.get_run(rid);feedback=latest_summary(store,rid)
    registry.initialize(store,repo);registry.initialize(store,repo)
    assert store.get_run(rid)==before and latest_summary(store,rid)==feedback
    assert len(registry.records(store,'MODEL_RETAINED'))==1
    assert len(registry.records(store,'CHAMPION_INITIALIZED'))==1


def test_dashboard_progress_is_read_only_and_labels_exploratory(lab,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from dashboard.data_access import cumulative_progress
    store,c,cid,repo,add=lab;add();registry.initialize(store,repo)
    monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    before=len(store.events(1000))
    summary=cumulative_progress()
    assert summary['cohorts'][0]['state']['confirmation']=='HISTORICAL_EXPLORATORY'
    app=AppTest.from_string('from dashboard.cumulative import progress_card\nfrom dashboard.data_access import cumulative_progress\nprogress_card(cumulative_progress())').run()
    assert not app.exception
    assert any('superiority unconfirmed' in item.value for item in app.markdown)
    assert len(store.events(1000))==before


def test_codex_receives_whole_history_memory_without_raw_dataset(lab,monkeypatch):
    from orchestrator import controller
    store,c,cid,repo,add=lab;add();registry.initialize(store,repo)
    monkeypatch.setenv('LOCO_LEDGER',str(store.path));captured={}
    def invoke(role,root,schema,prompt,**kw):captured.update(prompt=prompt,**kw);return {'response':{}}
    monkeypatch.setattr(controller,'invoke',invoke)
    controller._budgeted_invoke_once(store,'researcher',repo,None,'fixture',personal_mode=True)
    assert 'cumulative/evidence.json' in captured['reference_files']
    memory=json.loads(captured['reference_files']['cumulative/evidence.json'].read_text())
    assert memory['cohorts'][0]['champion']['candidate_id']=='baseline'
    assert 'mask' not in memory and 'images' not in memory


@pytest.mark.parametrize('fixed',[False,True])
def test_exact_frozen_multimethod_matrix_executes_and_promotes(lab,monkeypatch,tmp_path,fixed):
    """Full trusted plan/queue/retention path with artificial worker measurements.

    Independently implemented methods need no shared model.py or rewritten baseline:
    compare their exact previously trained checkpoints in a frozen arm matrix.
    """
    from orchestrator import controller, development_feedback
    from orchestrator.experiment_plan import execute_plan,baseline_reference
    from benchmark.data_guard import build_manifest,manifest_fingerprint
    from scripts.demo_run import make_fixture
    import shutil
    store,c,cid,repo,add=lab
    seeds=(3,17,29)
    base=[add(seed=s) for s in seeds]
    normal=tmp_path/'good';make_fixture(normal)
    manifest=build_manifest(normal);fingerprint=manifest_fingerprint(manifest)
    (tmp_path/'good_manifest.json').write_text(json.dumps(manifest))
    # Stamp only artificial records with the tiny fixture's actual provenance.
    def stamp(rid):
        with store.connect() as con:con.execute('UPDATE runs SET manifest_sha256=? WHERE run_id=?',(fingerprint,rid))
        path=__import__('pathlib').Path(store.get_run(rid)['artifact_dir'])/'run_manifest.json'
        provenance=json.loads(path.read_text());provenance['good_manifest_sha256']=fingerprint
        path.write_text(json.dumps(provenance))
        summary=latest_summary(store,rid);summary['training_manifest_sha256']=fingerprint
        summary['evaluation_id']+='-pinned';record_summary(store,rid,summary)
    for rid in base:stamp(rid)
    registry.initialize(store,repo)
    challengers=[add(seed=s,name='challenger',spro=.4,ap=.4) for s in seeds]
    for rid in challengers:stamp(rid)
    p=experiment_plan();p['diagnostics']=[]
    p['arms'][1]['inference']={'mode':'control'}
    p['required_measurements']=[{'source':'real','key':k} for k in
        (registry.SPRO,registry.AP,registry.FPR,registry.LATENCY,registry.PARAMETERS,registry.MEMORY)]
    p['evaluation']['real_development_feedback']=True
    for arm,ids in zip(p['arms'],(base,challengers)):
        arm['execution']='frozen';arm['baseline_by_seed']={}
        if fixed and arm['arm_id']=='control':
            ids=[base[0]]*len(seeds);arm['fixed_checkpoint_seed']=seeds[0]
        for seed,rid in zip(seeds,ids):
            ref,_,_=baseline_reference(store,rid,repo)
            if ref not in p['baselines']:p['baselines'].append(ref)
            arm['baseline_by_seed'][str(seed)]=ref['baseline_id']
    with store.connect() as con:
        cfg=json.loads(c.get(cid)['config_json']);cfg['research']['real_development_feedback']=True
        con.execute('UPDATE campaigns SET config_json=? WHERE id=?',(json.dumps(cfg),cid))
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path));monkeypatch.setenv('LOCO_LEDGER',str(store.path))
    monkeypatch.setattr(controller,'REPO',repo)
    def worker(*args,**kw):
        rid=kw['expected_run_id'];row=store.get_run(rid);cfg=json.loads(row['config_json'])
        original=store.get_run(cfg['frozen_baseline']['run_id']);folder=tmp_path/'jobs'/rid
        shutil.copytree(original['artifact_dir'],folder)
        store.transition(rid,'RUNNING')
        metrics=json.loads(original['metrics_json']);metrics.update({'protocol/optimizer_updates':0,'protocol/source_exposures':0})
        store.set_run_result(rid,metrics,gpu_seconds=.01,artifact_dir=str(folder));store.transition(rid,'COMPLETED')
        summary=latest_summary(store,original['run_id']);summary['artifact_dir']=str(folder/'feedback/fixture')
        summary['evaluation_id']='frozen-'+rid;record_summary(store,rid,summary)
        return {'run_id':rid,'status':'COMPLETED'}
    monkeypatch.setattr(controller,'run_worker_once',worker)
    monkeypatch.setattr(development_feedback,'evaluate_run',lambda runtime,s,rid,**kw:latest_summary(s,rid))
    result=execute_plan(store,c,cid,tmp_path,p)
    assert result['status']=='COMPLETE_CONTROLLED',result['problems']
    assert len(c.runs(cid))==6
    d=registry.consider_comparison(store,c,cid,p,result,repo)
    assert d[0]['status']=='PROMOTED',d
    assert registry.progress(store)['cohorts'][0]['champion']['candidate_id']=='challenger'
    if fixed:
        assert [cell['training_seed'] for cell in result['cells'][:3]]==[3,3,3]
        assert [cell['seed'] for cell in result['cells'][:3]]==list(seeds)
        assert d[0]['intervals'][registry.SPRO]['unit']=='challenger training seed versus one frozen incumbent'


def test_recovery_after_critique_finishes_promotion_without_new_jobs(lab):
    store,c,cid,repo,add=lab;p,result,old,ids=prepared(lab)
    c.event(cid,'EXPERIMENT_PLAN_RESULT',result=result)
    c.event(cid,'CRITIQUE',plan_id=p['plan_id'],review={'hypothesis_outcome':'supported'})
    before=[r['run_id'] for r in c.runs(cid)]
    registry.recover_comparisons(store,repo)
    assert registry.progress(store)['cohorts'][0]['state']['confirmation']=='REPEATED_SEED_SUPPORTED'
    assert [r['run_id'] for r in c.runs(cid)]==before
    count=len(registry.records(store,'CHAMPION_DECISION'))
    registry.recover_comparisons(store,repo)
    assert len(registry.records(store,'CHAMPION_DECISION'))==count


def test_zero_job_incomplete_comparison_is_counted(lab):
    store,c,cid,repo,add=lab;p=experiment_plan()
    result=comparison_result(p,[])
    c.event(cid,'EXPERIMENT_PLAN_RESULT',result=result)
    assert registry.progress(store)['incomplete_comparisons']==1
