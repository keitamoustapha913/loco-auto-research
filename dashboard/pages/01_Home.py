"""At-a-glance state: service health and research authorization are different facts."""
import os
import time

import streamlit as st

from dashboard.components import page_intro
from dashboard.data_access import (
    dataset_summary, get_events, get_proposals, gpu_status, ledger_summary,
    list_runs, pending_outbox, queue_paused, service_status, tracking_health,
)
from dashboard.presentation import (
    autonomy_state, classify_run, format_date, format_duration, parse_event,
    research_summary, run_label,
)

page_intro("Home", "Laboratory health, research activity, and approved compute.")


@st.cache_data(ttl=60)
def cached_dataset_summary():
    return dataset_summary()


@st.fragment(run_every="10s")
def live_home():
    runs = list_runs(500)
    proposals = get_proposals()
    events = get_events(200)
    services = service_status()
    gpu = gpu_status()
    dataset = cached_dataset_summary()
    summary = research_summary(runs)
    state, reason = autonomy_state(os.environ, queue_paused())
    lab_ok = all(services.values()) and tracking_health() == "Connected"
    active = next((run for run in runs if run.get("status") == "RUNNING"), None)
    queued = next((run for run in reversed(runs) if run.get("status") == "QUEUED"), None)
    next_proposal = next((p for p in sorted(proposals, key=lambda item: item["proposal"].get("priority_rank", 999)) if p["status"] == "PROPOSED"), None)

    left, right = st.columns(2, gap="large")
    with left:
        st.metric("Laboratory", "Running" if lab_ok else "Needs attention")
        st.caption("MLflow, dashboard, and tracking worker" if lab_ok else "Open System to inspect service health.")
    with right:
        st.metric("Autonomous research", state)
        st.caption(reason)

    if state == "Authorized":
        st.page_link("pages/02_Research.py", label="Review current research →", icon="🔬")

    campaign = os.getenv("LOCO_CAMPAIGN", "Unspecified campaign")
    categories = ", ".join(dataset["categories"]) or "Dataset unavailable"
    st.caption(f"{campaign}  ·  {categories}  ·  {dataset['images']} permitted good images")
    a, b, c, d = st.columns(4)
    a.metric("Research models evaluated", summary["evaluated"], help="Completed research pilots and full evaluations; infrastructure tests excluded.")
    b.metric("Infrastructure tests", summary["infrastructure_completed"], help="Completed toy or fixture runs used to verify the laboratory.")
    c.metric("GPU", gpu.get("name", "Unavailable").replace("NVIDIA GeForce ", "") if gpu.get("available") else "Unavailable",
             help=f"{gpu.get('memory_used_mib', 0):,} / {gpu.get('memory_total_mib', 0):,} MiB in use" if gpu.get("available") else None)
    consumed = ledger_summary()["gpu_seconds"] / 3600
    budget = float(os.getenv("LOCO_GPU_BUDGET_HOURS", "4"))
    d.metric("GPU budget remaining", f"{max(0, budget - consumed):.2f} h", help=f"{consumed:.2f} of {budget:.2f} approved GPU hours used across all runs.")

    current, direction = st.columns(2, gap="large")
    with current:
        st.subheader("Current activity")
        if active:
            elapsed = format_duration(time.time() - active["started_at"]) if active.get("started_at") else "—"
            st.write(f"**{run_label(active)}** · {classify_run(active)}")
            st.caption(f"Experiment running for {elapsed}. Codex activity is not separately tracked live.")
        else:
            st.write("**No experiment running**")
            st.caption("No active Codex task is reported by the controller.")
        if queued:
            st.caption(f"Next queued: {run_label(queued)} · {classify_run(queued)}")
        elif next_proposal:
            st.caption(f"Next proposed: {next_proposal['proposal'].get('claim', next_proposal['hypothesis_id'])}")
        else:
            st.caption("No research experiment is scheduled.")
    with direction:
        st.subheader("Latest outcome")
        if runs:
            latest = runs[0]
            st.write(f"**{run_label(latest)}** · {str(latest['status']).title()}")
            st.caption(f"{classify_run(latest)} · {format_date(latest.get('finished_at') or latest.get('created_at'))}")
            if latest.get("error"):
                st.caption(f"Failure: {latest['error']}")
        else:
            st.write("**No experiments recorded**")
            st.caption("Research results will appear here after an approved run completes.")
        decisions = [e for e in events if e["kind"] in {"CRITIC_REVIEW", "HYPOTHESIS_RANKING"}]
        if decisions:
            payload = parse_event(decisions[0])
            decision = payload.get("priority_rationale") or payload.get("recommended_next_action") or "A research decision was recorded."
            st.caption(f"Latest research decision: {decision}")
        else:
            st.caption("No scientific decision has been recorded yet.")

    with st.expander("Tracking and run details"):
        st.write(f"MLflow: {tracking_health()} · Pending tracking events: {pending_outbox()}")
        st.write(f"Research failures: {summary['research_failed']}")


live_home()
