"""Operational detail lives here, away from the scientific overview."""
import os
import shutil

import streamlit as st

from dashboard.components import page_intro
from dashboard.data_access import (
    campaign_summary, get_connection, gpu_status, list_runs, log_tail, pending_outbox,
    queue_paused, service_status, tracking_errors, tracking_health,
)
from dashboard.presentation import INFRASTRUCTURE, autonomy_state, classify_run, format_date, run_label

page_intro("System", "GPU, services, tracking, and operator diagnostics.")


@st.cache_data(ttl=60)
def cuda_ready():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, OSError):
        return False


@st.fragment(run_every="15s")
def live_system():
    gpu = gpu_status()
    services = service_status()
    mlflow = tracking_health()
    conn = get_connection()
    ledger = conn is not None
    if conn:
        conn.close()
    state, reason = autonomy_state(os.environ, queue_paused())
    campaign = campaign_summary()

    a, b, c, d = st.columns(4)
    a.metric("GPU", "RTX 4090" if gpu.get("available") else "Unavailable")
    b.metric("CUDA", "Available" if cuda_ready() else "Unavailable")
    c.metric("Ledger", "Connected" if ledger else "Unavailable")
    d.metric("MLflow", mlflow)
    if gpu.get("available"):
        st.caption(f"{gpu['memory_used_mib']:,} / {gpu['memory_total_mib']:,} MiB GPU memory used · {gpu['utilization_percent']}% GPU utilization")
    st.subheader("Services")
    for name, label in (("mlflow", "MLflow tracking"), ("dashboard", "Dashboard"), ("tracking", "Tracking synchronization")):
        st.write(f"**{label}:** {'Running' if services.get(name) else 'Stopped or unhealthy'}")
    st.link_button("Open MLflow", "http://127.0.0.1:56000")
    st.caption(f"{pending_outbox()} tracking events pending · Dashboard is read-only; queue control and smoke tests use `loco-lab`.")

    st.subheader("Research authorization")
    st.write(f"**{state}** · {reason}")
    if campaign:
        limits = campaign.get('config', {}).get('limits', {})
        st.write(f"**Campaign:** {campaign['status'].replace('_', ' ').title()} · "
                 f"{campaign['usage']['gpu_seconds_live']/3600:.2f} / {limits.get('max_gpu_hours', 0):.2f} GPU h · "
                 f"{campaign['codex_calls']} / {limits.get('max_codex_calls', '—')} calls")
    if queue_paused():
        st.caption("The research queue is paused by the trusted controller.")

    with st.expander("Infrastructure tests and active process"):
        runs = list_runs(500)
        active = [r for r in runs if r["status"] == "RUNNING"]
        st.write("Active experiment:", ", ".join(run_label(r) for r in active) if active else "None")
        tests = [r for r in runs if classify_run(r) == INFRASTRUCTURE]
        if tests:
            st.dataframe([{"Candidate": run_label(r), "Status": r["status"].title(), "Date": format_date(r.get("created_at"))} for r in tests], hide_index=True, width="stretch")
        else:
            st.caption("No infrastructure test recorded.")

    with st.expander("Limits, tooling, and tracking errors"):
        st.write(f"Approved GPU budget: {os.getenv('LOCO_GPU_BUDGET_HOURS', '4')} h")
        st.write(f"Job timeout: {os.getenv('LOCO_JOB_TIMEOUT_SECONDS', '1800')} s")
        st.write(f"Codex CLI: {'Available' if shutil.which('codex') else 'Unavailable'}")
        for error in tracking_errors():
            st.write(f"Tracking retry ({error['attempts']} attempts): {error['last_error']}")

    with st.expander("Service logs"):
        name = st.selectbox("Service", ["tracking", "mlflow", "dashboard", "research"])
        st.code(log_tail(name, lines=40) or "No log entries yet.", language="text")


live_system()
