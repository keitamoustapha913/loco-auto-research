"""Plain research/training durations; technical accounting stays in System."""
import json
import math
import os
from pathlib import Path
import time
import streamlit as st
from dashboard.presentation import format_duration


def training_seconds(run):
    metrics=run.get('metrics') or json.loads(run.get('metrics_json','{}'))
    # Historical arithmetic-only values must never be relabelled as loop time.
    value=metrics.get('training/loop_elapsed_seconds')
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0:return value
    config=run.get('config') or json.loads(run.get('config_json','{}'))
    if run.get('status')=='RUNNING' and config.get('training_completion_policy')=='elapsed-training-loop-v1':
        rid=run['run_id']
        if not rid or any(not (ch.isascii() and (ch.isalnum() or ch in '_-')) for ch in rid):return None
        root=Path(os.getenv('LOCO_RUNTIME','.runtime')).expanduser().resolve()/'jobs'
        path=root/rid/'training_timing.json'
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size>16384:return None
            audit=json.loads(path.read_text());value=audit.get('loop_elapsed_seconds')
            if audit.get('completion_policy')=='elapsed-training-loop-v1' and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0:return value
        except (OSError,ValueError):pass
    return None


def campaign_training_seconds(campaign, runs):
    ids=set(campaign.get('run_ids',[]))
    def belongs(run):
        config=run.get('config') or json.loads(run.get('config_json','{}'))
        return run['run_id'] in ids or config.get('campaign_id')==campaign.get('id')
    return sum(training_seconds(run) or 0 for run in runs if belongs(run))


def campaign_timing_card(campaign, runs):
    if campaign is None:
        from dashboard.data_access import campaign_summary
        campaign=campaign_summary()
        if not campaign:return
    training=campaign_training_seconds(campaign,runs)
    with st.container(border=True):
        st.write('**'+campaign['name']+' · '+campaign['status'].replace('_',' ').title()+'**')
        left,right=st.columns(2)
        left.metric('Research duration',format_duration((campaign.get('ended_at') or time.time())-campaign['created_at']))
        right.metric('Training-loop duration',format_duration(training))
        if any('training/duration_seconds' in (r.get('metrics') or json.loads(r.get('metrics_json','{}'))) and training_seconds(r) is None for r in runs):
            st.caption('Historical arithmetic-only training measurements are excluded from loop totals; their original values remain in run details.')
        diagnostic=[]
        for r in runs:
            m=r.get('metrics') or json.loads(r.get('metrics_json','{}'))
            if 'training/learning_seconds' in m:
                diagnostic.append({'Candidate':r.get('candidate_id',r['run_id']),
                    'Loop seconds':m.get('training/loop_elapsed_seconds'),
                    'Learning computation seconds':m['training/learning_seconds'],
                    'Loop overhead seconds':m.get('training/loop_overhead_seconds')})
        if diagnostic:
            with st.expander('Technical training-efficiency measurements'):
                st.dataframe(diagnostic,hide_index=True)
        if campaign.get('reason'):st.caption(campaign['reason'])
