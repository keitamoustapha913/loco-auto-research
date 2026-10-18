import streamlit as st
from dashboard.data_access import list_runs, get_mlflow_url
from dashboard.charts import run_table
st.title("Experiment comparisons")
st.warning("Synthetic proxy localization is not official LOCO logical sPRO.")
runs = list_runs()
if not runs:
    st.info("No runs in the ledger yet.")
    st.stop()
stages = sorted({r["stage"] for r in runs})
statuses = sorted({r["status"] for r in runs})
selected_stages = st.multiselect("Stage", stages, default=stages)
selected_status = st.multiselect("Status", statuses, default=statuses)
filtered = [r for r in runs if r["stage"] in selected_stages and r["status"] in selected_status]
st.dataframe(run_table(filtered), use_container_width=True, hide_index=True)
if filtered:
    run_by_id = {r["run_id"]:r for r in filtered}
    selection = st.selectbox("Inspect experiment provenance", list(run_by_id))
    selected = run_by_id[selection]
    st.json({k: selected[k] for k in ("run_id", "hypothesis_id", "candidate_id", "category", "stage", "status", "git_sha", "manifest_sha256", "config", "metrics")})
    url = get_mlflow_url(selected)
    if url:
        st.link_button("Open associated MLflow run", url)
    else:
        st.caption("MLflow record not linked yet; queued for synchronization if needed.")
