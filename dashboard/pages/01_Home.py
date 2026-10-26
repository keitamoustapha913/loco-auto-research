"""Current research at a glance. Campaign details and model tables live elsewhere."""
import json
import streamlit as st
from dashboard.components import page_intro
from dashboard.campaign_timing import campaign_training_seconds
from dashboard.view import campaign_picker,metric_columns
from dashboard.research_data import campaign_runs,comparison_records,campaign_stats,active_incumbent,SPRO,AP,LABEL
from dashboard.data_access import campaign_summary,queue_paused
from dashboard.presentation import format_duration,autonomy_state
import os

page_intro('Research overview','Campaign activity, scientific progress and the retained research reference.')
campaign=campaign_picker()


@st.fragment(run_every='15s')
def overview():
    current=campaign_summary(campaign['id']) if campaign else None
    if not current:
        state,reason=autonomy_state(os.environ,queue_paused())
        st.metric('Autonomous research',state);st.caption(reason)
        st.info('No campaign has been recorded yet.')
        if state=='Authorized':st.page_link('pages/02_Research.py',label='Review current research →')
        return
    runs=campaign_runs(current['id']);comparisons=comparison_records(current['id']);s=campaign_stats(current,runs,comparisons)
    metric_columns([('Campaign',current['status'].replace('_',' ').title()),('Research elapsed',format_duration(s['elapsed'])),('Time remaining',format_duration(s['remaining'])),('Training-loop duration',format_duration(campaign_training_seconds(current,runs)))])
    allowance=current['config']['limits'].get('research_hours',current['config']['limits'].get('max_wall_hours',0))*3600
    st.progress(min(1.,s['elapsed']/allowance) if allowance else 0.,text=current['name'])
    active=next((r for r in runs if r['status']=='RUNNING'),None)
    agent=next((e for e in current['events'] if e['kind'] in {'AGENT_STARTED','AGENT_COMPLETED','AGENT_FAILED'}),None)
    with st.container(border=True):
        st.subheader('Current activity')
        if current['status'] not in {'RUNNING','PAUSED','STOPPING'}:
            st.write('Campaign finished. '+(current.get('reason') or ''))
        elif not current['worker_running']:st.warning('Campaign worker is absent; recorded work may be interrupted.')
        elif active:st.write(f"**{active['candidate_id'].replace('_',' ')}** · {active['config'].get('plan_arm','')} · seed {active['config'].get('seed','—')}")
        elif agent and agent['kind']=='AGENT_STARTED':st.write('Codex '+json.loads(agent['payload_json']).get('role','research agent').replace('_',' ')+' is working.')
        else:st.write('Controller is selecting or preparing the next research operation.')
        if current.get('current_hypothesis'):st.caption('Current / last investigation: '+current['current_hypothesis'].replace('_',' '))
    metric_columns([('Comparisons complete',s['complete']),('Incomplete',s['incomplete']),('New models trained',s['trained']),('Runs reviewed',s['reviewed'])])
    st.caption(f"Selected campaign only · {s['jobs']} executed jobs · {s['failed']} failed jobs. Completion describes execution; confirmation requires separate scientific evidence.")
    incumbent=active_incumbent(current)
    if incumbent:
        st.subheader('Active research incumbent')
        st.write('**'+incumbent['candidate_id'].replace('_',' ')+'** · seed '+str(incumbent['configuration'].get('seed','Unavailable')))
        metrics=incumbent['metrics'];state=incumbent['research_state']
        metric_columns([('Logical AU-sPRO@5%',f'{metrics[SPRO]:.4f}'),('Logical pixel AP',f'{metrics[AP]:.4f}'),('Normal false positives',f"{metrics['dev_real/normal_false_positives']} / {metrics['dev_real/normal_images']}")])
        st.caption('Exploratory research reference · '+LABEL+' Compatible scorer; original-code equivalence unverified.')
        score=state.get('selection',{}).get('score')
        st.caption(('Selection score '+f'{score:.4f}' if score is not None else 'Selection score unavailable')+' · 60% sPRO + 20% pixel AP + 10% normal performance + 10% efficiency')
        delta=state.get('localization_delta',{})
        if SPRO in delta and AP in delta:
            st.caption(f"Change from previous incumbent: AU-sPRO {delta[SPRO]:+.4f} · pixel AP {delta[AP]:+.4f}. Selection gives localization 80% weight; this is not confirmed superiority.")
        else:st.caption('Change from previous incumbent unavailable; no measured improvement is inferred.')
    failed=[r for r in runs if r.get('error')]
    if failed:
        with st.expander(f"{len(failed)} recorded job failures"):
            for r in failed:st.write(r['run_id'][:8]);st.code(r['error'][-4500:],language='text')
    a,b=st.columns(2)
    a.page_link('pages/02_Research.py',label='Explore comparisons →')
    b.page_link('pages/03_Results.py',label='Compare model results →')

overview()
