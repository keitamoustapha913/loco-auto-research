"""Research dashboard: read-only Streamlit display of SQLite + MLflow."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
st.set_page_config(page_title="LOCO Autoresearch", page_icon="🔬", layout="wide")
st.sidebar.title("LOCO Research Lab")
st.sidebar.caption("RTX 4090 · guarded single-GPU research")
st.sidebar.warning("All development anomaly metrics are **synthetic proxies**, NOT official LOCO scores.")
pg = st.navigation([
    st.Page("pages/01_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/02_Experiments.py", title="Experiments", icon="🧪"),
    st.Page("pages/03_Heatmaps.py", title="Heatmaps", icon="🌡️"),
    st.Page("pages/04_Pareto.py", title="Pareto", icon="📈"),
    st.Page("pages/05_Research.py", title="Research", icon="🧠"),
    st.Page("pages/06_System.py", title="System", icon="🖥️"),
])
pg.run()
