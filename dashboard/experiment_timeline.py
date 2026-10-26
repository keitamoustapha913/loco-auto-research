"""Recorded execution chronology, independent of proposal or feedback delivery order."""
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from dashboard.data_access import rows
from dashboard.campaign_timing import training_seconds
from dashboard.presentation import format_duration

ZONE=ZoneInfo('Europe/Paris')


def timestamp(value):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):return 'Not recorded'
    return datetime.fromtimestamp(value,ZONE).strftime('%Y-%m-%d %H:%M:%S %Z')


def seconds(value):
    return format_duration(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 else 'Not recorded'


def execution_contexts(campaign_id=None):
    query='''SELECT r.*,c.name AS campaign_name,c.created_at AS campaign_started,
        cr.campaign_id,
        (SELECT MIN(e.time) FROM events e WHERE e.run_id=r.run_id AND e.kind='TRANSITION'
            AND json_extract(e.payload_json,'$.from')='RUNNING'
            AND json_extract(e.payload_json,'$.to') IN ('COMPLETED','FAILED','OOM','TIMED_OUT','CANCELLED','INTERRUPTED','INVALID')
            AND e.time>=r.started_at) AS job_ended_at,
        (SELECT payload_json FROM events e WHERE e.run_id=r.run_id AND e.kind='REAL_DEVELOPMENT_FEEDBACK' ORDER BY e.id DESC LIMIT 1) AS feedback_json
        FROM runs r JOIN campaign_runs cr ON cr.run_id=r.run_id JOIN campaigns c ON c.id=cr.campaign_id'''
    if campaign_id is not None:query+=' WHERE cr.campaign_id=?'
    query+=' ORDER BY (r.started_at IS NULL),r.started_at,r.created_at,r.rowid'
    records=rows(query,(campaign_id,) if campaign_id is not None else ())
    counters=defaultdict(int);result={}
    for row in records:
        run=dict(row,config=json.loads(row['config_json']),metrics=json.loads(row['metrics_json']))
        started=row['started_at'];ended=row['job_ended_at'];cid=row['campaign_id']
        if ended is None and row['status'] not in {'REJECTED','PROMOTED'}:ended=row['finished_at']
        order=None
        if started is not None:counters[cid]+=1;order=counters[cid]
        duration=ended-started if started is not None and ended is not None and ended>=started else None
        feedback=json.loads(row['feedback_json']) if row['feedback_json'] else {}
        evaluation=feedback.get('metrics',{}).get('dev_real/evaluation_wall_seconds')
        frozen=run['config'].get('execution_kind')=='frozen' or bool(run['config'].get('frozen_baseline'))
        result[row['run_id']]={'order':order,'campaign_id':cid,'campaign_name':row['campaign_name'],
            'campaign_started':row['campaign_started'],'started_at':started,'finished_at':ended,
            'job_seconds':duration,'training_seconds':None if frozen else training_seconds(run),
            'evaluation_seconds':None if feedback.get('reused_from_evaluation_id') else evaluation,
            'evaluation_reused':bool(feedback.get('reused_from_evaluation_id')),
            'execution_kind':'Frozen checkpoint' if frozen else 'New model','run':run}
    return result


def timeline_rows(campaign, *, now=None):
    contexts=execution_contexts(campaign['id']);now=time.time() if now is None else now
    plans=rows("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_REGISTERED' ORDER BY id",(campaign['id'],))
    studies={}
    for record in plans:
        pid=json.loads(record['payload_json'])['plan']['plan_id']
        if pid not in studies:studies[pid]=len(studies)+1
    output=[]
    for rid,context in contexts.items():
        run=context['run'];config=run['config'];elapsed=context['job_seconds']
        active=run['status']=='RUNNING' and campaign.get('status')=='RUNNING' and campaign.get('worker_running')
        if elapsed is None and active and context['started_at'] is not None:elapsed=max(0,now-context['started_at'])
        output.append({'#':context['order'],'Started':timestamp(context['started_at']),
            'Training':seconds(context['training_seconds']) if context['execution_kind']=='New model' else 'Not retrained',
            'Job time':seconds(elapsed)+(' · running' if active else ''),'Status':run['status'].replace('_',' ').title(),
            'Study':studies.get(config.get('plan_id')),'Model':run['candidate_id'].replace('_',' '),
            'Arm':config.get('plan_arm','Legacy'),'Seed':config.get('seed'),
            'Finished':timestamp(context['finished_at']),
            'Evaluation':'Reused' if context['evaluation_reused'] else seconds(context['evaluation_seconds']),
            'Type':context['execution_kind'],'Run':rid})
    return output
