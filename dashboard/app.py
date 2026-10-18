"""Read-only laboratory interface backed by the trusted SQLite ledger."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

from orchestrator.lab_cli import load_settings

load_settings()
st.set_page_config(page_title="LOCO Research Lab", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("LOCO Research Lab")
st.sidebar.caption("Good-only development")
st.sidebar.divider()

page = st.navigation([
    st.Page("pages/01_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/02_Research.py", title="Research", icon="🔬"),
    st.Page("pages/03_Results.py", title="Results", icon="📊"),
    st.Page("pages/04_System.py", title="System", icon="⚙️"),
])
page.run()
