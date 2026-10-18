import os
import time
import streamlit as st
from dashboard.data_access import list_runs, get_events, pending_outbox, tracking_health, gpu_status, ledger_summary, service_status, runtime_root

st.title('Research overview')
st.caption('Good-only development. Synthetic localization measurements are proxies, never official LOCO scores.')

@st.fragment(run_every='5s')
def live_overview():
    runs = list_runs(1000)
    summary = ledger_summary()
    counts = summary['counts']
    services = service_status()
    gpu = gpu_status()
    try:
        from benchmark.data_guard import load_manifest
        data, _ = load_manifest(runtime_root() / 'good_manifest.json', verify_hashes=False)
        dataset = ', '.join(sorted({e['category'] for e in data['images']}))
        images = len(data['images'])
    except Exception:
        dataset, images = 'Unavailable', 0
    budget = float(os.getenv('LOCO_GPU_BUDGET_HOURS', '4'))
    consumed = summary['gpu_seconds'] / 3600
    st.write(f"**Campaign:** {os.getenv('LOCO_CAMPAIGN', 'Unspecified')} · **Dataset:** {dataset} ({images} normal images)")
    st.write('**Laboratory:**', 'Running' if all(services.values()) else 'Partial / stopped', '· **GPU:**', gpu.get('name', 'Unavailable'), f"{gpu.get('memory_used_mib', 'N/A')} MiB in use")
    a, b, c, d = st.columns(4)
    a.metric('Completed', sum(counts.get(s, 0) for s in ('COMPLETED', 'REVIEWED', 'PROMOTED')))
    b.metric('Failed / timeout / OOM', sum(counts.get(s, 0) for s in ('FAILED', 'TIMED_OUT', 'OOM')))
    c.metric('GPU hours consumed', f'{consumed:.2f}')
    d.metric('GPU hours remaining', f'{max(0, budget-consumed):.2f}')
    st.write('**Run states:**', counts or 'No experiments yet')
    st.write('**MLflow:**', tracking_health(), '· **Pending tracking events:**', pending_outbox())
    active = next((r for r in runs if r['status'] == 'RUNNING'), None)
    queued = next((r for r in reversed(runs) if r['status'] == 'QUEUED'), None)
    st.write('**Current experiment:**', f"{active['run_id']} · {int(time.time() - active['started_at'])} s elapsed" if active and active.get('started_at') else 'Idle')
    st.write('**Next queued:**', queued['run_id'] if queued else 'None')
    decisions = [e for e in get_events(100) if e['kind'] in ('CRITIC_REVIEW', 'TRANSITION')]
    st.subheader('Recent research decisions')
    if decisions:
        st.dataframe(decisions[:15], hide_index=True, use_container_width=True)
    else:
        st.info('No research decisions recorded yet.')

live_overview()
