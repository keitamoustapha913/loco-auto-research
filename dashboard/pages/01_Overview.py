import streamlit as st
from dashboard.data_access import list_runs, get_events, pending_outbox, tracking_health
st.title("Research overview")
st.caption("Scientific development observability; no access to sealed LOCO test data")
runs = list_runs()
a,b,c,d = st.columns(4)
a.metric("Runs", len(runs))
b.metric("Completed", sum(r["status"] in ("COMPLETED","REVIEWED","PROMOTED") for r in runs))
c.metric("Running", sum(r["status"] == "RUNNING" for r in runs))
d.metric("GPU hours recorded", f"{sum(r['gpu_seconds'] for r in runs) / 3600:.2f}")
st.write("**MLflow:**", tracking_health(), "· **Pending tracking events:**", pending_outbox())
active = [r for r in runs if r["status"] in ("QUEUED", "RUNNING")]
if active:
    st.subheader("Active queue")
    st.dataframe([{"run_id":r["run_id"], "candidate":r["candidate_id"], "stage":r["stage"], "status":r["status"]} for r in active], use_container_width=True)
else:
    st.info("No active jobs. Run the fixture demo or queue an approved experiment using the operator CLI.")
st.subheader("Recent state changes")
events = get_events(20)
if events:
    st.dataframe(events, hide_index=True, use_container_width=True)
else:
    st.caption("No events yet. This is not simulated data.")
