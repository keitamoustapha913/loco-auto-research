"""Plain research/training durations; technical accounting stays in System."""
import json
import time
import streamlit as st
from dashboard.presentation import format_duration


def training_seconds(run):
    metrics=run.get('metrics') or json.loads(run.get('metrics_json','{}'))
    return metrics.get('training/duration_seconds')


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
        right.metric('Measured training duration',format_duration(training))
        if campaign.get('reason'):st.caption(campaign['reason'])
