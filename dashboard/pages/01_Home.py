"""A compact answer to: what is happening, and what is the best retained model?"""
import json
import os
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import campaign_picker,metric_columns
from dashboard.research_data import campaign_runs,comparison_records,campaign_stats,active_incumbent,SPRO,AP
from dashboard.data_access import campaign_summary,queue_paused
from dashboard.campaign_timing import campaign_training_seconds
from dashboard.presentation import format_duration,autonomy_state
from dashboard.detail_views import readable_name

page_intro('Research overview','Current activity and the best retained development model.')
campaign=campaign_picker()


@st.fragment(run_every='15s')
def overview():
    current=campaign_summary(campaign['id']) if campaign else None
    if not current:
        state,reason=autonomy_state(os.environ,queue_paused())
        st.metric('Autonomous research',state);st.caption(reason)
        st.info('No campaign recorded yet.')
        if state=='Authorized':st.page_link('pages/02_Research.py',label='Review current research →')
        return
    runs=campaign_runs(current['id']);comparisons=comparison_records(current['id']);stats=campaign_stats(current,runs,comparisons)
    running=current['status'] in {'RUNNING','PAUSED','STOPPING'}
    with st.container(border=True):
        st.subheader('Research in progress' if running else 'Campaign finished')
        active=next((r for r in runs if r['status']=='RUNNING'),None)
        agent=next((e for e in current['events'] if e['kind'] in {'AGENT_STARTED','AGENT_COMPLETED','AGENT_FAILED'}),None)
        if running:
            if current['status']=='PAUSED':st.write('Research is paused.')
            elif not current['worker_running']:st.warning('Worker unavailable; inspect System.')
            elif active:st.write(readable_name(active['candidate_id'])+f" · seed {active['config'].get('seed','?')}")
            elif agent and agent['kind']=='AGENT_STARTED':st.write('Codex '+json.loads(agent['payload_json']).get('role','research agent').replace('_',' ')+' is working.')
            else:st.write('Preparing the next research operation.')
            st.caption(format_duration(stats['remaining'])+' remaining')
        else:st.caption(current.get('reason') or current['status'].replace('_',' ').capitalize())
        metric_columns([('Research elapsed',format_duration(stats['elapsed'])),('Training-loop duration',format_duration(campaign_training_seconds(current,runs)))])
        st.caption(f"{stats['trained']} models trained · {stats['complete']} comparisons complete · {stats['incomplete']} incomplete · {stats['failed']} failed jobs")
        if running:
            total=stats['elapsed']+stats['remaining']
            st.progress(min(1.,stats['elapsed']/total) if total else 0.)
        st.page_link('pages/02_Research.py',label='Open campaign →')
    incumbent=active_incumbent(current)
    if incumbent:
        with st.container(border=True):
            st.subheader('Best retained model')
            st.write('**'+readable_name(incumbent['candidate_id'])+'** · seed '+str(incumbent['configuration'].get('seed','?')))
            metrics=incumbent['metrics']
            metric_columns([('Logical AU-sPRO',f'{metrics[SPRO]:.4f}'),('Logical pixel AP',f'{metrics[AP]:.4f}')])
            st.caption(f"Normal false positives {metrics['dev_real/normal_false_positives']} / {metrics['dev_real/normal_images']} · Exploratory research reference")
            delta=incumbent['research_state'].get('localization_delta',{})
            if SPRO in delta and AP in delta:st.caption(f"Since previous reference: sPRO {delta[SPRO]:+.4f} · AP {delta[AP]:+.4f}")
            st.page_link('pages/03_Results.py',label='Open model results →')
    st.caption('Selected campaign only · Real custom-split development feedback · Compatible scorer; not official full-test results or confirmed generalization.')

overview()
