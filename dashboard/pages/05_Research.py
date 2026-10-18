import json
from pathlib import Path
import streamlit as st
from dashboard.data_access import get_proposals, get_events, list_runs, get_mlflow_url
st.title("Research hypotheses and failures")
proposals = get_proposals()
if not proposals:
    st.info("No structured Codex proposals recorded. The system begins with open research freedom, not a mandatory model architecture.")
for item in proposals:
    p = item["proposal"]
    with st.expander(f"{item['hypothesis_id']} · {p.get('candidate_family', 'unspecified')}"):
        st.markdown(p.get("claim", "No claim supplied"))
        st.json(p)
if proposals:
    st.subheader('Hypothesis queue')
    st.dataframe([{'hypothesis': p['hypothesis_id'], 'status': p['status'],
                   'rank': p['proposal'].get('rank', 'Unranked'),
                   'family': p['proposal'].get('candidate_family', '')} for p in proposals],
                 hide_index=True, use_container_width=True)
st.subheader('Measured candidates and failures')
runs = list_runs()
if runs:
    st.dataframe([{'run': r['run_id'], 'candidate': r['candidate_id'], 'hypothesis': r['hypothesis_id'],
                   'status': r['status'], 'reason': r['error'], 'git_commit': r['git_sha'],
                   'mlflow': get_mlflow_url(r)} for r in runs], hide_index=True, use_container_width=True)
else:
    st.info('No candidates have been measured yet.')
st.subheader('Critic assessments and next directions')
reviews = [e for e in get_events(200) if e['kind'] in ('CRITIC_REVIEW', 'PROPOSAL')]
if reviews:
    for event in reviews[:20]:
        with st.expander(f"{event['kind']} · {event.get('run_id') or 'research'}"):
            try:
                st.json(json.loads(event['payload_json']))
            except ValueError:
                st.write(event['payload_json'])
else:
    st.info('No critic assessments or follow-up proposals recorded yet.')
st.subheader("Evidence library")
repo = Path(__file__).resolve().parents[2]
index = json.loads((repo / "literature" / "evidence_index.json").read_text())
st.dataframe(index["sources"], use_container_width=True, hide_index=True)
scout = repo / 'literature' / 'scout_notes.json'
if scout.is_file():
    with st.expander('Literature scout investigation (unverified leads)'):
        st.json(json.loads(scout.read_text()))
for title, filename in (('Hypothesis notes', 'hypotheses.md'), ('Discoveries', 'discoveries.md')):
    st.subheader(title)
    content = (repo / 'research_memory' / filename).read_text()
    st.markdown(content if content.strip() else 'No entries yet.')
st.subheader("Negative-result record")
st.markdown((repo / "research_memory" / "failed_approaches.md").read_text())
st.caption("Research notes are not authoritative metrics; inspect linked experiment runs.")
