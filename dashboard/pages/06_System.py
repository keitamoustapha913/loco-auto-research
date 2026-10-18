import os
import shutil
import streamlit as st
from dashboard.data_access import get_connection, pending_outbox, tracking_health, tracking_errors, list_runs, gpu_status, service_status, log_tail

st.title('System health')

@st.fragment(run_every='5s')
def live_system():
    gpu = gpu_status()
    services = service_status()
    st.write('**GPU:**', gpu)
    st.write('**CUDA:**', 'Available' if gpu['available'] else 'Unavailable')
    st.write('**Services:**', services)
    st.write('**MLflow:**', tracking_health(), '· **Pending events:**', pending_outbox())
    st.write('**Codex CLI:**', shutil.which('codex') or 'Unavailable')
    st.write('**Research autonomy:**', 'Enabled by operator' if os.getenv('LOCO_SANDBOX_VERIFIED') == '1' and os.getenv('LOCO_ALLOW_UNATTENDED') == '1' else 'Disabled: OS isolation and authorization not both verified')
    st.write('**Approved GPU budget:**', os.getenv('LOCO_GPU_BUDGET_HOURS', '4'), 'hours')
    st.write('**Job timeout:**', os.getenv('LOCO_JOB_TIMEOUT_SECONDS', '1800'), 'seconds')
    runs = list_runs()
    st.write('**Active runs:**', [r['run_id'] for r in runs if r['status'] == 'RUNNING'])
    connection = get_connection()
    st.write('**Ledger:**', 'Connected' if connection else 'Unavailable')
    if connection:
        connection.close()
    errors = tracking_errors()
    if errors:
        st.subheader('Tracking retry errors')
        st.dataframe(errors, hide_index=True, use_container_width=True)
    name = st.selectbox('Service log', ['tracking', 'mlflow', 'dashboard'])
    st.code(log_tail(name) or 'No log entries yet.')
    st.info('Queue changes and smoke tests require the operator CLI. Dashboard views are read only.')

live_system()
