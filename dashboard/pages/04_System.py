"""Operational health, authorization scope and retained storage; no model ranking."""
import time
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import metric_columns,table
from dashboard.research_data import authorization_state,decode
from dashboard.data_access import campaign_summary,gpu_status,service_status,tracking_health,pending_outbox,queue_paused,tracking_errors,log_tail,runtime_root,rows,list_runs
from dashboard.presentation import format_date,classify_run,INFRASTRUCTURE

page_intro('Laboratory health','Services, authorization scope, storage retention and actionable diagnostics.')


@st.cache_data(ttl=120)
def storage_inventory(path):
    from orchestrator.storage_cleanup import inventory
    return inventory(path)


@st.fragment(run_every='30s')
def health():
    campaign=campaign_summary();services=service_status();gpu=gpu_status()
    metric_columns([('Research worker','Running' if campaign and campaign['worker_running'] else 'Inactive'),('MLflow',tracking_health()),('Tracking deliveries pending',pending_outbox())])
    st.caption('Research activity: '+(campaign['status'].replace('_',' ').title() if campaign else 'No campaign'))
    table([{'Service':name.title(),'Health':'Running' if okay else 'Unavailable'} for name,okay in services.items()])
    if gpu.get('available'):st.caption(f"{gpu.get('name','GPU')} · utilization {gpu['utilization_percent']}% · {gpu['memory_used_mib']:,} / {gpu['memory_total_mib']:,} MiB")
    st.subheader('Authorization and execution')
    state,reason=authorization_state(campaign,queue_paused());st.write('**'+state+'** · '+reason)
    st.caption('New execution eligibility is checked against the authorization file by loco-lab research start. The dashboard does not grant authorization or infer eligibility from legacy environment flags.')
    if campaign:st.caption(f"Last / active campaign: {campaign['id']} · {campaign.get('reason') or 'No stop reason'}")
    st.subheader('Storage and retention')
    try:
        storage=storage_inventory(str(runtime_root()));used=storage['observed_bytes']/2**30;quota=storage['limit_bytes']/2**30
        metric_columns([('Runtime used',f'{used:.2f} GiB'),('Runtime quota',f'{quota:.0f} GiB'),('Available under quota',f'{max(0,quota-used):.2f} GiB')])
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
    st.caption(f"{len(full)} full-retained model-producing records · {len(archived)} metrics-only archives. Rolling policy selects nine, with explicit active/protected-reference exceptions.")
    last=rows("SELECT time,payload_json FROM events WHERE kind='STORAGE_CLEANUP_PASS' ORDER BY id DESC LIMIT 1")
    if last:st.caption('Last automatic cleanup: '+format_date(last[0]['time']))
    else:st.caption('No completed cleanup audit recorded.')
    st.caption('Cleanup runs at startup, completion, admission pressure and about every 30 minutes. This read-only page never runs cleanup.')
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
