"""Read-only campaign -> comparison -> arm -> run projections of durable evidence.

No StateStore initialization, model loading, promotion, cleanup or agent calls.
"""
import json
import time
from pathlib import Path
from dashboard.data_access import rows, campaign_summary, real_feedback_summaries, runtime_root

SPRO='dev_real/logical_au_spro_0_05_compatible'
AP='dev_real/logical_pixel_ap_pooled'
LABEL='LOCO custom-split development feedback — not official full-test results.'


def decode(value, default=None):
    try:return json.loads(value) if isinstance(value,str) else value
    except (TypeError,ValueError):return {} if default is None else default


def campaigns():
    return [dict(r,config=decode(r['config_json'])) for r in rows("SELECT * FROM campaigns ORDER BY (status='RUNNING') DESC,created_at DESC")]


def campaign_runs(campaign_id):
    found=rows('SELECT r.* FROM runs r JOIN campaign_runs cr ON cr.run_id=r.run_id WHERE cr.campaign_id=? ORDER BY r.created_at,r.rowid',(campaign_id,))
    states={r['name'][len('artifact_retention:'):]:decode(r['value']) for r in rows("SELECT name,value FROM settings WHERE name LIKE 'artifact_retention:%'")}
    return [dict(r,config=decode(r['config_json']),metrics=decode(r['metrics_json']),artifact_availability=states.get(r['run_id'],{})) for r in found]


def comparison_records(campaign_id):
    events=rows('SELECT * FROM campaign_events WHERE campaign_id=? ORDER BY id',(campaign_id,))
    records={}
    for event in events:
        p=decode(event['payload_json'])
        if event['kind']=='EXPERIMENT_PLAN_REGISTERED':
            plan=p['plan'];records[plan['plan_id']]={'plan':plan,'registered_at':event['time'],'result':None,'review':None}
    # Reviews can be resumed in a later campaign; evidence_campaign_id is the
    # authoritative origin, rather than the review-call's host campaign.
    for event in events:
        p=decode(event['payload_json'])
        if event['kind']=='EXPERIMENT_PLAN_RESULT':
            result=p.get('result',{});pid=result.get('plan_id')
            if pid not in records:
                pid=next((k for k,v in records.items() if p.get('result',{}).get('plan_sha256')==__import__('orchestrator.experiment_plan',fromlist=['digest']).digest(v['plan'])),None)
            if pid in records:records[pid]['result']=result
    for event in rows("SELECT campaign_id,payload_json FROM campaign_events WHERE kind='CRITIQUE' ORDER BY id"):
        p=decode(event['payload_json']);origin=p.get('evidence_campaign_id') or event['campaign_id']
        if origin==campaign_id and p.get('plan_id') in records:records[p['plan_id']]['review']=p.get('review')
    proposals={r['hypothesis_id']:decode(r['proposal_json']) for r in rows('SELECT hypothesis_id,proposal_json FROM hypotheses')}
    runs=campaign_runs(campaign_id)
    for pid,record in records.items():
        record['proposal']=proposals.get(record['plan']['hypothesis_id'],{})
        record['runs']=[r for r in runs if r['config'].get('plan_id')==pid]
        result=record['result'];statuses={r['status'] for r in record['runs']}
        if result and result.get('status')=='INCOMPLETE':status='Incomplete'
        elif record['review']:status='Reviewed'
        elif result and result.get('status','').startswith('COMPLETE'):status='Awaiting critique'
        elif 'RUNNING' in statuses:status='Running'
        elif statuses & {'FAILED','OOM','TIMED_OUT','CANCELLED','INTERRUPTED'}:status='Failed'
        else:status='Planned'
        record['status']=status
        record['expected']=len(record['plan']['arms'])*len(record['plan']['seeds'])
        record['completed']=sum(r['status'] in {'COMPLETED','REVIEWED','PROMOTED'} for r in record['runs'])
    return list(records.values())


def campaign_stats(campaign, runs, comparisons):
    from dashboard.presentation import classify_run, INFRASTRUCTURE
    scientific=[r for r in runs if classify_run(r)!=INFRASTRUCTURE and r.get('candidate_id')!='toy_normal_ae_cuda']
    ended=campaign.get('ended_at') or time.time();elapsed=max(0,ended-campaign['created_at'])
    limits=campaign['config'].get('limits',{});hours=limits.get('research_hours',limits.get('max_wall_hours',0))
    return {'jobs':sum(r.get('started_at') is not None for r in runs),
            'trained':sum(r['config'].get('execution_kind')!='frozen' and r['status'] in {'COMPLETED','REVIEWED','PROMOTED'} for r in scientific),
            'reviewed':sum(r['status']=='REVIEWED' for r in scientific),
            'failed':sum(r['status'] in {'FAILED','OOM','TIMED_OUT'} for r in runs),
            'complete':sum(bool(c['result']) and c['result'].get('status','').startswith('COMPLETE') for c in comparisons),
            'incomplete':sum(bool(c['result']) and c['result'].get('status')=='INCOMPLETE' for c in comparisons),
            'elapsed':elapsed,'remaining':max(0,hours*3600-elapsed)}


def model_results(campaign_id=None):
    # No oldest-N cutoff: archived numerical measurements remain searchable.
    feedback=real_feedback_summaries(campaign_id)
    run_records={r['run_id']:r for r in rows('SELECT * FROM runs ORDER BY created_at,rowid')}
    memberships={r['run_id']:r['campaign_id'] for r in rows('SELECT * FROM campaign_runs')}
    from dashboard.presentation import classify_run, INFRASTRUCTURE
    result=[]
    for f in feedback:
        r=run_records[f['run_id']];f['run']=dict(r,config=decode(r['config_json']),metrics=decode(r['metrics_json']))
        if classify_run(f['run'])==INFRASTRUCTURE or r['candidate_id']=='toy_normal_ae_cuda':continue
        f['campaign_id']=memberships.get(f['run_id']);f['config']=f['run']['config']
        result.append(f)
    return result


def research_incumbents():
    heads=[decode(r['value']) for r in rows("SELECT value FROM settings WHERE name LIKE 'research_incumbent:%'")]
    retained={p['retention_id']:p for r in rows("SELECT payload_json FROM events WHERE kind='MODEL_RETAINED' ORDER BY id") if (p:=decode(r['payload_json'])).get('retention_id')}
    result=[]
    for head in heads:
        model=retained.get(head.get('retention_id'))
        if model:result.append(dict(model,research_state=head))
    return result


def active_incumbent(campaign=None):
    models=research_incumbents()
    if campaign:
        dataset=campaign['config'].get('campaign',{}).get('dataset')
        # Resolve dataset through the authoritative learning run for legacy heads.
        for model in models:
            run=rows('SELECT category FROM runs WHERE run_id=?',(model['run_id'],))
            model['_category']=run[0]['category'] if run else None
        models=[m for m in models if m['_category']==dataset]
        latest=real_feedback_summaries(campaign['id'])
        if latest:
            keys=('split_id','evaluator_version','training_manifest_sha256','inference_device','measurement_protocol','threshold_source')
            matching=[m for m in models if all(m.get('conditions',{}).get(k)==latest[-1].get(k) for k in keys)]
            if matching:models=matching
    # Different cohorts are never ranked against each other. Most recently
    # retained cohort is the fallback; alternatives remain visible on Results.
    return max(models,key=lambda m:m.get('created_at',0),default=None)


def authorization_state(campaign, paused=False):
    if not campaign:return ('No campaign','No campaign execution is active. Eligibility is checked by loco-lab research start.')
    if campaign['status'] not in {'RUNNING','PAUSED','STOPPING'}:
        return ('Inactive','Historical campaign authorization is preserved; starting new research requires the normal authorization checks.')
    if paused or campaign['status']=='PAUSED':return ('Paused','Research is paused.')
    security=campaign['config'].get('security',{})
    return ('Active authorization','Current campaign uses its recorded authorization; shared-UID risk acceptance is scoped to that authorization.') if security else ('Not verified','No recorded campaign security settings available.')


def artifact_status(summary):
    state=summary.get('artifact_availability',{}).get('status')
    if state=='ARCHIVED_METRICS_ONLY':return ('Archived','Full artifacts intentionally retired; numerical results and provenance remain available.')
    if not summary.get('evaluation_id'):return ('Not generated','No heatmap evaluation artifact was recorded.')
    root=runtime_root()/'jobs';folder=root/summary['run_id']/'feedback'/summary['evaluation_id']
    try:
        from benchmark.feedback_artifacts import safe_path
        safe_path(folder,root)
        if not folder.is_dir() or not (folder/'case_scores.json').is_file():
            return ('Missing unexpectedly','Recorded evaluation metadata is absent; this is not an archival marker.')
    except FileNotFoundError:return ('Missing unexpectedly','Recorded evaluation artifacts are absent; no archival marker exists.')
    except PermissionError:return ('Temporarily unavailable','Artifact access is currently unavailable.')
    except (ValueError,OSError):return ('Integrity failure','Artifact path failed trusted safety checks.')
    return ('Available','Retained artifact metadata found; image and heatmap hashes are verified when opened.')


def comparison_matrix(record):
    lookup={(r['config'].get('plan_arm'),r['config'].get('plan_pair_seed',r['config'].get('seed'))):r for r in record['runs']}
    return [{'Arm':arm['arm_id'],'Execution':arm['execution'],'Seed':seed,
        'Status':lookup.get((arm['arm_id'],seed),{}).get('status','MISSING'),
        'Run':lookup.get((arm['arm_id'],seed),{}).get('run_id','')}
        for arm in record['plan']['arms'] for seed in record['plan']['seeds']]


def leaderboard_row(f):
    m=f['metrics'];cfg=f.get('config',{});run=f.get('run',{})
    return {'Run':f['run_id'],'Candidate':f['candidate_id'].replace('_',' '),'Arm':cfg.get('plan_arm','—'),
        'Seed':cfg.get('seed'),'Status':run.get('status','Unavailable'),
        'Logical AU-sPRO':m.get(SPRO),'Logical pixel AP':m.get(AP),
        'Logical AUROC':m.get('dev_real/logical_image_auroc'),
        'Normal FP':f"{m.get('dev_real/normal_false_positives','?')} / {m.get('dev_real/normal_images','?')}",
        'Structural sPRO':m.get('dev_real/structural_au_spro_0_05_compatible'),
        'p95 (ms)':m.get('dev_real/inference_e2e_ms_p95'),'Device':f.get('inference_device','Unavailable'),
        'Parameters':m.get('dev_real/parameters'),'Artifacts':artifact_status(f)[0]}
