import streamlit as st
from dashboard.data_access import list_runs
from orchestrator.decision_engine import pareto_frontier
st.title("Pareto view")
st.warning("Quality axis is held-out **synthetic pixel AUROC proxy**, never official LOCO sPRO.")
runs = list_runs()
points = []
front = set(pareto_frontier(runs))
for r in runs:
    m = r["metrics"]
    if "dev_synthetic/pixel_auroc" in m and "inference/e2e_ms_p95" in m:
        points.append({"Run":r["run_id"], "Candidate":r["candidate_id"], "Status":r["status"], "Latency p95 (ms)":m["inference/e2e_ms_p95"], "Synthetic pixel AUROC proxy":m["dev_synthetic/pixel_auroc"], "Pareto": "Nondominated" if r["run_id"] in front else "Dominated"})
if not points:
    st.info("Insufficient measured experiments for a Pareto comparison.")
else:
    import pandas as pd
    import plotly.express as px
    table = pd.DataFrame(points)
    st.plotly_chart(px.scatter(table, x="Latency p95 (ms)", y="Synthetic pixel AUROC proxy", color="Pareto", hover_name="Run", hover_data=["Candidate", "Status"]), use_container_width=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Pareto membership is a screening aid; not independent evidence of real logical-anomaly generalization.")
