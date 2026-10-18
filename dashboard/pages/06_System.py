import os
from pathlib import Path
import streamlit as st
from dashboard.data_access import runtime_root, get_connection, pending_outbox, tracking_health, list_runs
st.title("System health · read only")
runtime = runtime_root()
st.metric("MLflow", tracking_health())
st.metric("Tracking outbox pending", pending_outbox())
st.write("Runtime directory:", str(runtime))
st.write("GPU-hours configured:", os.getenv("LOCO_GPU_BUDGET_HOURS", "4"))
runs = list_runs()
st.write("Running:", [r["run_id"] for r in runs if r["status"] == "RUNNING"])
st.write("Interrupted:", [r["run_id"] for r in runs if r["status"] == "INTERRUPTED"])
conn = get_connection()
st.write("Controller DB reachable:", conn is not None)
if conn is not None:
    conn.close()
st.info("Start/pause/reconcile must be done via protected operator CLI. The dashboard cannot execute jobs or change policies.")
