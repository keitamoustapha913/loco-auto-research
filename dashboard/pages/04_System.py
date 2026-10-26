"""Operational health, authorization scope and retained storage; no model ranking."""
import time
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import metric_columns,table
from dashboard.research_data import authorization_state,decode
from dashboard.data_access import campaign_summary,gpu_status,service_status,tracking_health,pending_outbox,queue_paused,tracking_errors,log_tail,runtime_root,rows,list_runs
from dashboard.presentation import format_date,classify_run,INFRASTRUCTURE

page_intro('System','Laboratory health and available storage.')


@st.cache_data(ttl=120)
def storage_inventory(path):
    from orchestrator.storage_cleanup import inventory
    return inventory(path)


@st.fragment(run_every='30s')
def health():
    campaign=campaign_summary();services=service_status();gpu=gpu_status()
    ready=all(services.values()) and tracking_health()=='Connected'
    st.subheader('Services healthy' if ready else 'Attention needed')
    if campaign and campaign['status']=='RUNNING' and not campaign['worker_running']:
        st.warning('The campaign is active but its research worker is unavailable.')
    st.caption(('Research worker running' if campaign and campaign['worker_running'] else 'No research worker running')+
        f" · {pending_outbox()} tracking deliveries pending")
    if gpu.get('available'):st.caption(f"{gpu.get('name','GPU')} · {gpu['utilization_percent']}% utilization")
    with st.expander('Services and authorization'):
        table([{'Service':name.title(),'Health':'Running' if okay else 'Unavailable'} for name,okay in services.items()])
        state,reason=authorization_state(campaign,queue_paused());st.write('**'+state+'** · '+reason)
        st.caption('New execution eligibility is checked by loco-lab research start. Historical campaign authorization is distinct from present activity.')
        if campaign:st.caption(campaign['id']+' · '+(campaign.get('reason') or campaign['status']))
    st.subheader('Storage')
    try:
        storage=storage_inventory(str(runtime_root()));used=storage['observed_bytes']/2**30;quota=storage['limit_bytes']/2**30
        metric_columns([('Available',f'{max(0,quota-used):.1f} GiB'),('Used',f'{used:.1f} / {quota:.0f} GiB')])
        st.progress(min(1.,used/quota) if quota else 0.)
    except (OSError,ValueError) as exc:st.warning('Storage inventory unavailable: '+str(exc))
    retained=[dict(run_id=r['name'].split(':',1)[1],**decode(r['value'])) for r in rows("SELECT name,value FROM settings WHERE name LIKE 'artifact_retention:%'")]
    archived=[r for r in retained if r.get('status')=='ARCHIVED_METRICS_ONLY']
    archived_ids={r['run_id'] for r in archived}
    full=[]
    for r in rows("SELECT run_id,candidate_id,category,status,config_json,artifact_dir FROM runs WHERE status IN ('COMPLETED','REVIEWED','PROMOTED','REJECTED')"):
        cfg=decode(r['config_json'])
        if r['category']=='fixture' or r['candidate_id'] in {'toy_normal_ae','toy_normal_ae_cuda'} or cfg.get('execution_kind')=='frozen' or cfg.get('frozen_baseline') or r['run_id'] in archived_ids:continue
        path=runtime_root()/'jobs'/r['run_id']/'checkpoint/model.pt'
        if not path.is_symlink() and path.resolve().is_relative_to((runtime_root()/'jobs').resolve()) and path.is_file():full.append(dict(r,status='FULL'))
    st.caption(f"{len(full)} retained models · {len(archived)} archived records")
    last=rows("SELECT time,payload_json FROM events WHERE kind='STORAGE_CLEANUP_PASS' ORDER BY id DESC LIMIT 1")
    if last:st.caption('Last automatic cleanup: '+format_date(last[0]['time']))
    else:st.caption('No completed cleanup audit recorded.')

    with st.expander('Retained records and cleanup audit'):
        if full:table([{'Run':r['run_id'],'Status':r['status']} for r in full])
        if last:st.json(decode(last[0]['payload_json']))
    errors=tracking_errors()
    if errors:
        st.subheader('Tracking errors')
        for error in errors:st.error(str(error.get('last_error')))
    with st.expander('Service logs and technical diagnostics'):
        service=st.selectbox('Service',['research','tracking','mlflow','dashboard'])
        st.code(log_tail(service,lines=35) or 'No recorded log entries.',language='text')
        tests=[r for r in list_runs(500) if classify_run(r)==INFRASTRUCTURE]
        if tests:table([{'Infrastructure check':r['candidate_id'],'Run':r['run_id'],'Status':r['status']} for r in tests[:20]])
        st.link_button('Open MLflow','http://127.0.0.1:56000')
        st.caption('MLflow loopback URL requires the existing SSH tunnel when viewed remotely.')

health()
