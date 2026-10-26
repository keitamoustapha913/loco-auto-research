"""Read-only laboratory interface backed by the trusted SQLite ledger."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

from orchestrator.lab_cli import load_settings

load_settings()
st.set_page_config(page_title="LOCO Research Lab", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
section.main > div {padding-top: 2rem;}
[data-testid="stMainBlockContainer"] {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2vw, 2rem);}
[data-testid="stMetric"] {min-width: 0;}
[data-testid="stBaseButton-tertiary"] {justify-content: flex-start; text-align: left;}
[data-testid="stBaseButton-tertiary"] > div {width: 100%; text-align: left; justify-content: flex-start;}
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-wrap: wrap;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex: 1 1 160px; min-width: 160px;}
}
</style>''', unsafe_allow_html=True)
st.sidebar.caption("LOCO Research Lab · Normal-only training")
st.sidebar.divider()

page = st.navigation([
    st.Page("pages/01_Home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/02_Research.py", title="Research", icon=":material/science:"),
    st.Page("pages/03_Results.py", title="Results", icon=":material/insights:"),
    st.Page("pages/04_System.py", title="System", icon=":material/monitor_heart:"),
])
page.run()
