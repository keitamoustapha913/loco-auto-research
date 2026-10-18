import json
from pathlib import Path
import streamlit as st
from dashboard.data_access import get_proposals, get_events
st.title("Research hypotheses and failures")
proposals = get_proposals()
if not proposals:
    st.info("No structured Codex proposals recorded. The system begins with open research freedom, not a mandatory model architecture.")
for item in proposals:
    p = item["proposal"]
    with st.expander(f"{item['hypothesis_id']} · {p.get('candidate_family', 'unspecified')}"):
        st.markdown(p.get("claim", "No claim supplied"))
        st.json(p)
st.subheader("Evidence library")
repo = Path(__file__).resolve().parents[2]
index = json.loads((repo / "literature" / "evidence_index.json").read_text())
st.dataframe(index["sources"], use_container_width=True, hide_index=True)
st.subheader("Negative-result record")
st.markdown((repo / "research_memory" / "failed_approaches.md").read_text())
st.caption("Research notes are not authoritative metrics; inspect linked experiment runs.")
