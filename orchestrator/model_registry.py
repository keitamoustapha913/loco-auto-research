"""Cumulative, development-only model retention in the authoritative ledger.

No fitting, inference, agent invocation or model selection from final-holdout data.
An exploratory incumbent is retained separately from evidence of a confirmed gain.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from benchmark.data_guard import sha256_file

REPO = Path(__file__).resolve().parents[1]
SPRO = 'dev_real/logical_au_spro_0_05_compatible'
AP = 'dev_real/logical_pixel_ap_pooled'
FPR = 'dev_real/normal_image_fpr'
LATENCY = 'dev_real/inference_e2e_ms_p95'
PARAMETERS = 'dev_real/parameters'
MEMORY = 'dev_real/peak_cpu_rss_mib'
PREFIX = 'champion:'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def policy():
    return json.loads((REPO/'configs/champion_policy.json').read_text())


def records(store, kind):
    with store.connect() as con:
        return [dict(r, payload=json.loads(r['payload_json'])) for r in con.execute(
            'SELECT * FROM events WHERE kind=? ORDER BY id', (kind,))]


def event(con, kind, payload, run_id=None):
    con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                (run_id, time.time(), kind, json.dumps(payload, sort_keys=True, allow_nan=False)))


def states(store):
    with store.connect() as con:
        return {r['name'][len(PREFIX):]:json.loads(r['value']) for r in con.execute(
            'SELECT name,value FROM settings WHERE name LIKE ?', (PREFIX+'%',))}


def cohort(feedback, sample_hash):
    fields = {k:feedback[k] for k in ('split_id', 'evaluator_version', 'training_manifest_sha256',
              'inference_device', 'measurement_protocol', 'threshold_source')}
    fields['evaluation_samples_sha256'] = sample_hash
    return canonical(fields), fields


def retain(store, run_id, feedback, repo_root=REPO):
    """Copy verified bytes; the immutable archive survives candidate workspace edits."""
    from orchestrator.experiment_plan import baseline_reference,verify_baseline
    reference, source, row = baseline_reference(store, run_id, repo_root)
    configuration=json.loads(row['config_json'])
    learning_row=row;learning_configuration=configuration;seen={run_id}
    while learning_configuration.get('frozen_baseline'):
        _,learning_row=verify_baseline(store,learning_configuration['frozen_baseline'],repo_root)
        if learning_row['run_id'] in seen:raise ValueError('cyclic frozen model provenance')
        seen.add(learning_row['run_id']);learning_configuration=json.loads(learning_row['config_json'])
    if (row['category']=='fixture' or row['candidate_id'] in {'toy_normal_ae','toy_normal_ae_cuda'} or
        row['hypothesis_id'] in {'infrastructure-smoke','fixture-demo'} or
        configuration.get('experiment_type')=='infrastructure_test'):
        raise ValueError('infrastructure tests are not research models')
    if feedback.get('checkpoint_unchanged') is not True or feedback.get('status') != 'COMPLETE_COMPATIBLE_DEVELOPMENT':
        raise ValueError('incomplete or non-frozen real-development evaluation')
    if feedback['checkpoint_sha256'] != reference['checkpoint_sha256'] or feedback['source_sha256'] != reference['source_sha256']:
        raise ValueError('feedback does not describe the frozen model bytes')
    if feedback['training_manifest_sha256'] != row['manifest_sha256']:
        raise ValueError('training provenance mismatch')
    job = Path(row['artifact_dir']).resolve()
    if json.loads((job/'run_manifest.json').read_text()).get('good_manifest_sha256')!=row['manifest_sha256']:
        raise ValueError('normal-only learning manifest provenance mismatch')
    output = Path(feedback['artifact_dir'])
    if output.is_symlink() or not output.resolve().is_relative_to(job/'feedback'):
        raise ValueError('feedback artifacts escape the originating run')
    case_path = output/'case_scores.json'
    if case_path.is_symlink(): raise ValueError('unsafe per-image record')
    cases = json.loads(case_path.read_text())
    identities = sorted((r['category'], r['source_sha256']) for r in cases)
    if len(set(identities)) != len(identities): raise ValueError('duplicate evaluation identities')
    counts = {name:sum(r[0] == name for r in identities) for name in
              ('good','logical_anomalies','structural_anomalies')}
    if counts != feedback['counts']['categories'] or sum(counts.values()) != feedback['counts']['images']:
        raise ValueError('evaluation image coverage mismatch')
    metrics = feedback['metrics']
    required = (SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY,'dev_real/normal_false_positives',
                'dev_real/normal_images','dev_real/normal_only_threshold','dev_real/logical_image_auroc',
                'dev_real/structural_image_auroc','dev_real/structural_au_spro_0_05_compatible')
    if any(isinstance(metrics.get(k),bool) or not isinstance(metrics.get(k),(int,float)) or
           not math.isfinite(metrics[k]) for k in required):
        raise ValueError('missing/nonfinite required real-development measurement')
    n, fp = metrics['dev_real/normal_images'], metrics['dev_real/normal_false_positives']
    if n != counts['good'] or not 0 <= fp <= n or fp != int(fp) or abs(metrics[FPR]-fp/n)>1e-12:
        raise ValueError('false-positive denominator mismatch')
    if any(not 0<=metrics[k]<=1 for k in (SPRO,AP,FPR)):
        raise ValueError('localization/FPR outside unit interval')
    cid, conditions = cohort(feedback, canonical(identities))
    key = canonical({'run':run_id,'evaluation':feedback['evaluation_id'],'reference':reference,'conditions':conditions})
    root = store.path.parent/'retained_models'
    folder = root/key
    if root.is_symlink() or folder.is_symlink(): raise ValueError('unsafe retention path')
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    files = {'model.py':source,'configuration.json':row['config_json'].encode(),
             'evaluation.json':json.dumps(feedback,sort_keys=True).encode(),
             'case_scores.json':case_path.read_bytes(),
             'run_manifest.json':(job/'run_manifest.json').read_bytes()}
    audit_path=job/'training_audit.json'
    if audit_path.is_file():
        if audit_path.is_symlink():raise ValueError('unsafe training audit')
        files['training_audit.json']=audit_path.read_bytes()
    hashes = {}
    for name in ('model.pt','checkpoint_manifest.json'):
        src=job/'checkpoint'/name; dst=folder/name
        if dst.is_symlink(): raise ValueError('unsafe retained file')
        if dst.exists():
            if sha256_file(dst)!=sha256_file(src): raise ValueError('retained checkpoint changed')
        else: shutil.copyfile(src,dst); dst.chmod(0o444)
        hashes[name]=sha256_file(dst)
    for name,data in files.items():
        dst=folder/name
        if dst.exists():
            if dst.is_symlink() or dst.read_bytes()!=data: raise ValueError('retained evidence changed')
        else: dst.write_bytes(data); dst.chmod(0o444)
        hashes[name]=hashlib.sha256(data).hexdigest()
    if (hashes['model.pt']!=reference['checkpoint_sha256'] or
        hashes['checkpoint_manifest.json']!=reference['checkpoint_manifest_sha256']):
        raise ValueError('checkpoint bytes changed while being retained')
    result={'retention_id':key,'run_id':run_id,'candidate_id':row['candidate_id'],'hypothesis_id':row['hypothesis_id'],
            'git_commit':row['git_sha'],'reference':reference,'configuration':json.loads(row['config_json']),
            'category':row['category'],
            'cohort_id':cid,'conditions':conditions,'metrics':metrics,'archive':str(folder),'hashes':hashes,
            'created_at':row['created_at'],'evaluation_id':feedback['evaluation_id'],
            'training_metrics':json.loads(row['metrics_json']),
            'learning_configuration':learning_configuration,'learning_run_id':learning_row['run_id'],
            'learning_metrics':json.loads(learning_row['metrics_json']),
            'evidence_status':'exploratory; not a confirmed improvement','policy_sha256':canonical(policy())}
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        prior=con.execute("SELECT payload_json FROM events WHERE kind='MODEL_RETAINED' AND run_id=?",(run_id,)).fetchall()
        if not any(json.loads(r[0])['retention_id']==key for r in prior): event(con,'MODEL_RETAINED',result,run_id)
    return result


def verify_archive(model):
    root=Path(model['archive'])
    if root.is_symlink(): raise ValueError('retained archive redirected')
    for name,expected in model['hashes'].items():
        path=root/name
        if path.is_symlink() or sha256_file(path)!=expected: raise ValueError('retained model integrity failure: '+name)


def retained_category(store, model):
    """Resolve legacy records from the authoritative run; never patch history."""
    from orchestrator.experiment_plan import digest
    verify_archive(model)
    row=store.get_run(model['run_id'])
    if row is None:raise ValueError('retained model has no authoritative historical run')
    reference=model['reference'];root=Path(model['archive'])
    if (reference['run_id']!=row['run_id'] or model['candidate_id']!=row['candidate_id'] or
        model['hypothesis_id']!=row['hypothesis_id'] or model['git_commit']!=row['git_sha']):
        raise ValueError('retained model/run identity or source commit mismatch')
    configuration=json.loads(row['config_json'])
    archived=json.loads((root/'configuration.json').read_text())
    if (archived!=configuration or model['configuration']!=configuration or
        digest(configuration)!=reference['configuration_sha256']):
        raise ValueError('retained configuration differs from authoritative provenance')
    if (sha256_file(root/'model.py')!=reference['source_sha256'] or
        configuration.get('model_sha256')!=reference['source_sha256'] or
        sha256_file(root/'model.pt')!=reference['checkpoint_sha256'] or
        sha256_file(root/'checkpoint_manifest.json')!=reference['checkpoint_manifest_sha256']):
        raise ValueError('retained source/checkpoint provenance mismatch')
    provenance=json.loads((root/'run_manifest.json').read_text())
    evaluation=json.loads((root/'evaluation.json').read_text())
    fingerprint=row['manifest_sha256']
    if (not fingerprint or provenance.get('good_manifest_sha256')!=fingerprint or
        evaluation.get('training_manifest_sha256')!=fingerprint or
        model['conditions'].get('training_manifest_sha256')!=fingerprint):
        raise ValueError('retained normal-learning dataset provenance mismatch')
    category=row['category']
    if not isinstance(category,str) or not category:raise ValueError('historical run has no dataset/category')
    for label,record in [('retained record',model),('configuration',configuration),('run manifest',provenance)]:
        for key in ('category','dataset'):
            if key in record and record[key]!=category:
                raise ValueError(label+' dataset/category mismatches authoritative run')
    return category


def admissible(model, rules):
    m=model['metrics']
    return m[FPR]<=rules['max_normal_fpr'] and m[PARAMETERS]<=rules['max_parameters']


def frontier(models):
    def vector(m):
        x=m['metrics'];return (-x[SPRO],-x[AP],x[FPR],x[LATENCY],x[PARAMETERS],x[MEMORY])
    return [m['retention_id'] for m in models if not any(
        all(a<=b for a,b in zip(vector(other),vector(m))) and
        any(a<b for a,b in zip(vector(other),vector(m))) for other in models)]


def retained(store):
    return {r['payload']['retention_id']:r['payload'] for r in records(store,'MODEL_RETAINED')}


def initialize(store, repo_root=REPO):
    """Import historical evidence without rewriting any run or historical metric.

    Bootstrap once per comparable cohort, then scores alone never replace a head.
    """
    if store.read_only: raise PermissionError('retention requires trusted ledger writer')
    with store.connect() as con:
        rows=con.execute("SELECT run_id,payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' AND id IN (SELECT MAX(id) FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' GROUP BY run_id)").fetchall()
    known={(m['run_id'],m['evaluation_id']) for m in retained(store).values()}
    excluded=[]
    for r in rows:
        feedback=json.loads(r['payload_json'])
        if (r['run_id'],feedback['evaluation_id']) in known: continue
        try:
            model=retain(store,r['run_id'],feedback,repo_root)
            from orchestrator.mlflow_tracker import enqueue_run_tracking
            enqueue_run_tracking(store,r['run_id'],event_key_suffix='retained:'+model['retention_id'])
        except (ValueError,KeyError,OSError,__import__('subprocess').CalledProcessError) as exc:
            excluded.append({'run_id':r['run_id'],'reason':str(exc)})
    models=retained(store);rules=policy()
    for cid in {m['cohort_id'] for m in models.values()}:
        candidates=[m for m in models.values() if m['cohort_id']==cid and admissible(m,rules)]
        if not candidates: continue
        best=max(candidates,key=lambda m:(m['metrics'][SPRO],m['metrics'][AP],-m['metrics'][FPR]))
        verify_archive(best)
        state={'retention_id':best['retention_id'],'confirmation':'HISTORICAL_EXPLORATORY',
               'policy_sha256':canonical(rules),'revision':1,'last_improvement_event':None}
        with store.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute('SELECT 1 FROM settings WHERE name=?',(PREFIX+cid,)).fetchone(): continue
            event(con,'CHAMPION_INITIALIZED',dict(state,cohort_id=cid,reason='best admissible historical localization; superiority unconfirmed'),best['run_id'])
            con.execute('INSERT INTO settings(name,value) VALUES(?,?)',(PREFIX+cid,json.dumps(state,sort_keys=True)))
    return {'retained':len(models),'excluded':excluded,'cohorts':len(states(store))}


def plan_context(store):
    return {'policy':policy(),'policy_sha256':canonical(policy()),'champions':states(store)}


def seed_interval(values, rules):
    """Resample paired training-seed differences, never individual pixels."""
    import numpy as np
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(rules['bootstrap_seed'])
    means=x[rng.integers(0,len(x),size=(rules['bootstrap_repeats'],len(x)))].mean(axis=1)
    alpha=(1-rules['confidence'])/2
    return {'mean_difference':float(x.mean()),'lower':float(np.quantile(means,alpha)),
            'upper':float(np.quantile(means,1-alpha)),'unit':'paired training seed',
            'seeds':len(x),'confidence':rules['confidence']}


def consider_comparison(store, campaigns, campaign_id, plan, result, repo_root=REPO):
    """Promotion is deterministic, atomic and independent of the critic's preferences."""
    from orchestrator.experiment_plan import comparison_result,digest,plan_history
    from orchestrator.development_feedback import latest_summary
    arms=[a for a in plan['arms'] if a['arm_id']!=plan['baseline_arm_id'] or len(plan['arms'])==1]
    keys=[canonical([campaign_id,digest(plan),a['arm_id']]) for a in arms]
    previous={r['payload']['decision_id']:r['payload'] for r in records(store,'CHAMPION_DECISION')}
    if keys and all(k in previous for k in keys):return [previous[k] for k in keys]
    initialize(store,repo_root)
    registration=next((r for r in plan_history(campaigns,campaign_id) if r['plan_sha256']==digest(plan)),{})
    context=registration.get('promotion_context',{})
    rows=[r for r in campaigns.runs(campaign_id) if json.loads(r['config_json']).get('plan_sha256')==digest(plan)]
    actual=comparison_result(plan,rows,{r['run_id']:latest_summary(store,r['run_id']) or {} for r in rows})
    rules=policy();models=retained(store)
    decisions=[]
    by_run={m['run_id']:m for m in models.values()}
    for arm in arms:
        key=canonical([campaign_id,digest(plan),arm['arm_id']])
        existing=next((r['payload'] for r in records(store,'CHAMPION_DECISION') if r['payload'].get('decision_id')==key),None)
        if existing:
            decisions.append(existing);continue
        reason=[];intervals={};chosen=None;incumbent=None;head=None;cid=None;fixed_incumbent=False
        from orchestrator.planning_contract import evidence_classification
        eligibility=evidence_classification(plan,rows)
        if not eligibility['promotion_eligible']:reason.extend(eligibility['reasons'])
        if result['status']!='COMPLETE_CONTROLLED' or actual['status']!='COMPLETE_CONTROLLED': reason.append('comparison incomplete or exploratory')
        if context.get('policy_sha256')!=canonical(rules):reason.append('promotion policy absent or changed since registration')
        if len(plan['seeds'])<rules['minimum_seeds']:reason.append('insufficient independent training seeds')
        requested={r['key'] for r in plan['required_measurements'] if r['source']=='real'}
        if not {SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY}<=requested:
            reason.append('promotion endpoints were not preregistered')
        cells={(json.loads(r['config_json'])['plan_arm'],json.loads(r['config_json']).get('plan_pair_seed',json.loads(r['config_json'])['seed'])):by_run.get(r['run_id']) for r in rows}
        pairs=[(cells.get((arm['arm_id'],s)),cells.get((plan['baseline_arm_id'],s))) for s in plan['seeds']]
        if not pairs or any(a is None or b is None for a,b in pairs):reason.append('missing retained real-development pair')
        else:
            cohort_ids={m['cohort_id'] for pair in pairs for m in pair}
            if len(cohort_ids)!=1:reason.append('incompatible evaluation cohorts')
            else:
                cid=next(iter(cohort_ids));head=states(store).get(cid)
                pinned=context.get('champions',{}).get(cid)
                if not head or head!=pinned:reason.append('incumbent changed or was not pinned before execution')
                else:
                    incumbent=models[head['retention_id']]
                    if not any(b['reference']['checkpoint_sha256']==incumbent['reference']['checkpoint_sha256'] for _,b in pairs):
                        reason.append('exact incumbent checkpoint was not evaluated as a baseline')
                    # A matched retraining of the exact incumbent source/config is permitted;
                    # changes to training settings other than seed and plan metadata are not.
                    ignored={'seed','campaign_id','campaign_name','timeout_seconds','plan_id','plan_sha256','plan_arm',
                             'execution_kind','execution_mode','frozen_baseline','diagnostics','diagnostic_provider',
                             'cuda_smoke_required','trusted_source_sha256'}
                    def scientific(cfg):return {k:v for k,v in cfg.items() if k not in ignored and not k.startswith('plan_')}
                    incumbent_cfg=incumbent.get('learning_configuration',incumbent['configuration'])
                    for _,b in pairs:
                        origin_cfg=b.get('learning_configuration',b['configuration'])
                        if (b['reference']['source_sha256']!=incumbent['reference']['source_sha256'] or
                            scientific(origin_cfg)!=scientific(incumbent_cfg) or
                            b['configuration'].get('inference',{})!=incumbent['configuration'].get('inference',{})):
                            reason.append('baseline is not the exact incumbent method/configuration')
                    fixed_seed=next(a for a in plan['arms'] if a['arm_id']==plan['baseline_arm_id']).get('fixed_checkpoint_seed')
                    fixed_incumbent=(fixed_seed is not None and
                        all(b['reference']['checkpoint_sha256']==incumbent['reference']['checkpoint_sha256'] for _,b in pairs))
                    # Explicit repeated inference of one fixed incumbent is a
                    # conditional model comparison, never baseline seed replication.
                    for side in (0,1):
                        if len({p[side]['reference']['checkpoint_sha256'] for p in pairs})<len(plan['seeds']):
                            reason.append('repeated checkpoint is not independent seed replication')
                    for a,b in pairs:
                        if not admissible(a,rules):reason.append('deployment FPR/parameter ceiling exceeded')
                        if a['metrics'][FPR]>b['metrics'][FPR]:reason.append('normal false positives regressed')
                        if a['metrics'][AP]<b['metrics'][AP]:reason.append('pixel precision regressed in a seed')
                        for metric,drop in (('dev_real/logical_image_auroc',rules['max_logical_detection_drop']),
                                            ('dev_real/structural_au_spro_0_05_compatible',rules['max_structural_localization_drop'])):
                            if b['metrics'][metric]-a['metrics'][metric]>drop:reason.append('secondary performance regression: '+metric)
                        for metric,ratio in ((LATENCY,rules['max_latency_ratio']),(MEMORY,rules['max_memory_ratio'])):
                            if a['metrics'][metric]>b['metrics'][metric]*ratio:reason.append('efficiency regression: '+metric)
                    for metric,gain in ((SPRO,rules['minimum_spro_gain']),(AP,rules['minimum_pixel_ap_gain'])):
                        delta=[a['metrics'][metric]-b['metrics'][metric] for a,b in pairs]
                        intervals[metric]=seed_interval(delta,rules)
                        if fixed_incumbent:
                            intervals[metric].update(unit='challenger training seed versus one frozen incumbent',
                                limitation='baseline training-seed and image-population uncertainty unmeasured; no method-level causal claim')
                        if min(delta)<=gain or intervals[metric]['lower']<=gain:reason.append('repeatable benefit not established: '+metric)
                    # Retain a representative seed; never select the best seed as the result.
                    chosen=pairs[0][0]
                    if chosen['metrics'][SPRO]<incumbent['metrics'][SPRO] or chosen['metrics'][AP]<incumbent['metrics'][AP]:
                        reason.append('representative checkpoint would regress retained localization')
        decision={'decision_id':key,'campaign_id':campaign_id,'plan_id':plan['plan_id'],'arm':arm['arm_id'],
                  'cohort_id':cid,'status':'RETAIN_CHAMPION' if reason else 'PROMOTED',
                  'reasons':sorted(set(reason)) or ['paired repeated-seed localization benefit under frozen policy'],
                  'intervals':intervals,'policy_sha256':canonical(rules),
                  'previous':head,'challenger':chosen['retention_id'] if chosen else None,
                  'comparison_scope':'challenger seeds versus fixed incumbent checkpoint' if fixed_incumbent else 'paired method training seeds',
                  'confirmation_run_ids':[a['run_id'] for a,b in pairs if a and b]}
        if chosen:verify_archive(chosen)
        if incumbent:verify_archive(incumbent)
        with store.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            existing=next((json.loads(r[0]) for r in con.execute(
                "SELECT payload_json FROM events WHERE kind='CHAMPION_DECISION'") if json.loads(r[0]).get('decision_id')==key),None)
            if existing:
                decisions.append(existing);continue
            current=con.execute('SELECT value FROM settings WHERE name=?',(PREFIX+cid,)).fetchone() if cid else None
            if not reason and (not current or json.loads(current[0])!=head):
                decision['status']='RETAIN_CHAMPION';decision['reasons']=['concurrent incumbent change; new comparison required']
            if decision['status']=='PROMOTED':
                state={'retention_id':chosen['retention_id'],'confirmation':'REPEATED_SEED_SUPPORTED',
                       'revision':head['revision']+1,'policy_sha256':canonical(rules),'last_improvement_event':key,
                       'confirmation_run_ids':decision['confirmation_run_ids'],'intervals':intervals}
                state['comparison_scope']=decision['comparison_scope']
                con.execute('UPDATE settings SET value=? WHERE name=?',(json.dumps(state,sort_keys=True),PREFIX+cid))
            event(con,'CHAMPION_DECISION',decision,chosen['run_id'] if chosen else None)
        decisions.append(decision)
    return decisions


def recover_comparisons(store, repo_root=REPO):
    """Finish deterministic promotion bookkeeping after a crash, never replay jobs."""
    from orchestrator.campaign_state import CampaignState
    campaigns=CampaignState(store)
    with store.connect() as con:
        rows=con.execute("SELECT campaign_id,kind,payload_json FROM campaign_events WHERE kind IN ('CRITIQUE','EXPERIMENT_PLAN_REGISTERED','EXPERIMENT_PLAN_RESULT') ORDER BY id").fetchall()
    registered={};results={};reviewed=set()
    for r in rows:
        value=json.loads(r['payload_json']);cid=r['campaign_id']
        if r['kind']=='EXPERIMENT_PLAN_REGISTERED':registered[(cid,value['plan']['plan_id'])]=value['plan']
        elif r['kind']=='EXPERIMENT_PLAN_RESULT':results[(cid,value['result']['plan_id'])]=value['result']
        elif value.get('plan_id'):reviewed.add((cid,value['plan_id']))
    for key in reviewed & registered.keys() & results.keys():
        decisions=consider_comparison(store,campaigns,key[0],registered[key],results[key],repo_root)
        # Persist durable tracking delivery even if the previous process exited
        # between the promotion transaction and its MLflow outbox insertion.
        from orchestrator.mlflow_tracker import enqueue_run_tracking
        for run in campaigns.runs(key[0]):
            if json.loads(run['config_json']).get('plan_id')!=key[1]:continue
            with store.connect() as con:
                latest=con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='CUMULATIVE_REVIEW' ORDER BY id DESC LIMIT 1",(run['run_id'],)).fetchone()
                if not latest or json.loads(latest[0])!=decisions:event(con,'CUMULATIVE_REVIEW',decisions,run['run_id'])
            enqueue_run_tracking(store,run['run_id'],event_key_suffix='cumulative-recovery:'+canonical(decisions))


def rollback(store, cohort_id, reason):
    """Explicit trusted operator recovery. Original promotion and bytes are retained."""
    if not reason.strip():raise ValueError('rollback reason required')
    models=retained(store)
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        head=json.loads(con.execute('SELECT value FROM settings WHERE name=?',(PREFIX+cohort_id,)).fetchone()[0])
        decisions=[json.loads(r[0]) for r in con.execute("SELECT payload_json FROM events WHERE kind='CHAMPION_DECISION' ORDER BY id DESC")]
        prior=next((d['previous'] for d in decisions if d['status']=='PROMOTED' and d['challenger']==head['retention_id']),None)
        if not prior:raise ValueError('no prior champion to restore')
        verify_archive(models[prior['retention_id']])
        restored=dict(prior,revision=head['revision']+1)
        con.execute('UPDATE settings SET value=? WHERE name=?',(json.dumps(restored,sort_keys=True),PREFIX+cohort_id))
        event(con,'CHAMPION_ROLLBACK',{'cohort_id':cohort_id,'previous':head,'restored':restored,'reason':reason})
    return restored


def progress(store):
    """Cheap read-only aggregate; integrity is rechecked on promotion and rollback."""
    models=retained(store);heads=states(store);comparisons={}
    with store.connect() as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='campaign_events'").fetchone():
            for r in con.execute("SELECT campaign_id,payload_json FROM campaign_events WHERE kind='EXPERIMENT_PLAN_RESULT' ORDER BY id"):
                value=json.loads(r['payload_json'])['result']
                comparisons[(r['campaign_id'],value['plan_sha256'])]=value
        for r in con.execute("SELECT payload_json FROM events WHERE kind='SCIENTIFIC_COMPARISON' ORDER BY id"):
            value=json.loads(r[0])
            if not any(key[1]==value['plan_sha256'] for key in comparisons):comparisons[('legacy',value['plan_sha256'])]=value
        usage=dict(con.execute('SELECT COALESCE(SUM(gpu_seconds),0) AS gpu_seconds,COUNT(*) AS attempts FROM runs').fetchone())
    decisions=[r['payload'] for r in records(store,'CHAMPION_DECISION')]
    cohorts=[]
    for cid,head in heads.items():
        group=[m for m in models.values() if m['cohort_id']==cid]
        cohorts.append({'cohort_id':cid,'state':head,'champion':models[head['retention_id']],
            'pareto_retention_ids':frontier(group),'models':group,
            'best_observed_spro':max(m['metrics'][SPRO] for m in group),
            'best_observed_pixel_ap':max(m['metrics'][AP] for m in group),
            'best_supported_spro':models[head['retention_id']]['metrics'][SPRO] if head['confirmation']=='REPEATED_SEED_SUPPORTED' else None,
            'best_supported_pixel_ap':models[head['retention_id']]['metrics'][AP] if head['confirmation']=='REPEATED_SEED_SUPPORTED' else None})
    active_improvements={head.get('last_improvement_event') for head in heads.values()}
    last=max((r['time'] for r in records(store,'CHAMPION_DECISION') if r['payload']['status']=='PROMOTED' and
              r['payload']['decision_id'] in active_improvements),default=None)
    scientific_filter="category!='fixture' AND candidate_id NOT IN ('toy_normal_ae','toy_normal_ae_cuda') AND hypothesis_id NOT IN ('infrastructure-smoke','fixture-demo') AND COALESCE(json_extract(config_json,'$.experiment_type'),'')!='infrastructure_test'"
    with store.connect() as con:
        since=con.execute('SELECT COUNT(*) FROM runs WHERE '+scientific_filter+' AND created_at>?',(last or 0,)).fetchone()[0]
        research_gpu=con.execute('SELECT COALESCE(SUM(gpu_seconds),0) FROM runs WHERE '+scientific_filter).fetchone()[0]
        research_calls=con.execute("SELECT COUNT(*) FROM events WHERE kind='CODEX_CALL' AND json_extract(payload_json,'$.campaign_id') IS NOT NULL").fetchone()[0]
    return {'cohorts':cohorts,'decisions':decisions,'complete_comparisons':sum(r['status']=='COMPLETE_CONTROLLED' for r in comparisons.values()),
            'incomplete_comparisons':sum(r['status']=='INCOMPLETE' for r in comparisons.values()),
            'attempts_since_verified_improvement':since,'last_verified_improvement_at':last,
            'global_gpu_seconds':usage['gpu_seconds'],'global_codex_calls':store.codex_call_count(),
            'research_gpu_seconds':research_gpu,'campaign_codex_calls':research_calls,
            'caveat':'Custom development split; compatible sPRO, original-scorer equivalence unverified. No final-holdout generalization claim.'}


def localization_history(store, models=None):
    """Observed incumbent history, starting at retention initialization, not backfilled wins.

    Attempts create flat steps. Only recorded promotion/rollback events change the
    retained model. Same attempt count may have multiple events; keep event order.
    """
    models=retained(store) if models is None else models
    with store.connect() as con:
        attempts=[dict(r) for r in con.execute("SELECT created_at FROM runs WHERE category!='fixture' AND candidate_id NOT IN ('toy_normal_ae','toy_normal_ae_cuda') AND hypothesis_id NOT IN ('infrastructure-smoke','fixture-demo') AND COALESCE(json_extract(config_json,'$.experiment_type'),'')!='infrastructure_test' ORDER BY created_at,rowid")]
        events=[dict(r) for r in con.execute("SELECT id,time,kind,payload_json FROM events WHERE kind IN ('CHAMPION_INITIALIZED','CHAMPION_DECISION','CHAMPION_ROLLBACK') ORDER BY time,id")]
    changes=[]
    for record in events:
        value=json.loads(record['payload_json']);kind=record['kind']
        if kind=='CHAMPION_INITIALIZED':
            state=value;label='Exploratory incumbent retained'
        elif kind=='CHAMPION_ROLLBACK':
            state=value['restored'];label='Operator rollback: '+value['reason']
        elif value.get('status')=='PROMOTED':
            state={'retention_id':value['challenger'],'confirmation':'REPEATED_SEED_SUPPORTED'}
            label='Supported development promotion'
        else:continue
        model=models.get(state['retention_id'])
        if not model:continue
        changes.append((record['time'],record['id'],model,state,label))
    timeline={}
    # Merge admissions and actual retention events chronologically. Never imply
    # a model was the incumbent before its recorded retention date.
    stream=[(r['created_at'],0,None) for r in attempts]+[(t,i,(m,s,label)) for t,i,m,s,label in changes]
    heads={};count=0
    for timestamp,order,change in sorted(stream,key=lambda row:(row[0],row[1])):
        if change is None:
            count+=1
            updated=list(heads.items())
            label='Champion retained'
        else:
            model,state,label=change;cid=model['cohort_id'];heads[cid]=(model,state)
            updated=[(cid,(model,state))]
        for cid,(model,state) in updated:
            timeline.setdefault(cid,[]).append({'attempt':count,'time':timestamp,
                'logical_spro':model['metrics'][SPRO],'logical_pixel_ap':model['metrics'][AP],
                'candidate':model['candidate_id'],'run_id':model['run_id'],
                'confirmation':state['confirmation'],'event':label,'event_order':order})
    return timeline


def scientific_memory(store):
    """Whole-history compact index plus retained provenance, never anomaly data."""
    result=progress(store)
    with store.connect() as con:
        proposals=[dict(r) for r in con.execute('SELECT hypothesis_id,status,proposal_json FROM hypotheses ORDER BY created_at')]
        reviews=[dict(r) for r in con.execute("SELECT run_id,payload_json FROM events WHERE kind IN ('CRITIC_REVIEW','SCIENTIFIC_COMPARISON') ORDER BY id")]
        failures=[dict(r) for r in con.execute("SELECT run_id,hypothesis_id,error,status FROM runs WHERE status IN ('FAILED','OOM','TIMED_OUT','INTERRUPTED','INVALID')")]
    result['hypothesis_index']=[{'hypothesis_id':r['hypothesis_id'],'status':r['status'],
        'question':json.loads(r['proposal_json']).get('research_question',json.loads(r['proposal_json']).get('title',''))} for r in proposals]
    result['failures']=failures
    # Collapse identical per-run comparison/critique records without dropping
    # earlier ideas. Detailed immutable evidence remains in the ledger snapshots.
    history={}
    for row in reviews:
        payload=json.loads(row['payload_json'])
        identity=canonical(payload)
        if identity not in history:
            history[identity]={'run_ids':[], 'summary':{k:payload[k] for k in
                ('plan_id','hypothesis_id','status','problems','hypothesis_outcome','objective_measured_results',
                 'critic_concerns','suggested_falsification','recommended_next_action','paired_seed_differences',
                 'paired_image_intervals') if k in payload}}
        history[identity]['run_ids'].append(row['run_id'])
    result['evidence_history']=list(history.values())
    for group in result['cohorts']:
        group['model_index']=[{k:m[k] for k in ('retention_id','run_id','candidate_id','hypothesis_id','reference','metrics')}
                              for m in group.pop('models')]
    return result


def status_summary(store):
    """Concise operator view; full provenance remains in research report/memory."""
    summary=progress(store)
    for group in summary['cohorts']:
        model=group.pop('champion');group.pop('models')
        group['champion']={k:model[k] for k in ('run_id','candidate_id','reference','metrics','conditions','archive')}
        confirmed=group['state']['confirmation']=='REPEATED_SEED_SUPPORTED'
        group['best_supported_spro']=model['metrics'][SPRO] if confirmed else None
        group['best_supported_pixel_ap']=model['metrics'][AP] if confirmed else None
    summary['decisions']=summary['decisions'][-10:]
    return summary
