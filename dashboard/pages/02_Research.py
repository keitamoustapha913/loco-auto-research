"""Research intent, evidence, decisions and the next controlled step."""
import json
from pathlib import Path

import streamlit as st

from dashboard.components import page_intro
from dashboard.data_access import get_events, get_proposals, list_runs
from dashboard.presentation import format_date, parse_event, research_runs, run_label

REPO = Path(__file__).resolve().parents[2]
page_intro("Research", "What Codex investigated, why it matters, and what evidence supports the next step.")
st.subheader("Objective")
st.write("Find a compact, fast, good-only method that produces meaningful logical anomaly maps. Development uses normal images and clearly labeled synthetic proxies.")

proposals = get_proposals()
runs = research_runs(list_runs(500))
events = get_events(300)
reviews = [e for e in events if e["kind"] == "CRITIC_REVIEW" and any(r["run_id"] == e.get("run_id") for r in runs)]
scout_file = REPO / "literature" / "scout_notes.json"

st.subheader("Research cycle")
evidence = [
    ("Literature", scout_file.is_file()),
    ("Hypotheses", bool(proposals)),
    ("Implementation", any(p["status"] in {"IMPLEMENTED", "QUEUED", "COMPLETED", "FAILED"} for p in proposals)),
    ("Experiment", bool(runs)),
    ("Critique", bool(reviews)),
    ("Next decision", any(e["kind"] == "HYPOTHESIS_RANKING" for e in events)),
]
st.write("  →  ".join(f"{'✓' if done else '○'} {label}" for label, done in evidence))

if not proposals and not runs:
    st.info("No live research cycle has been recorded. The completed toy runs are infrastructure checks and do not count as evaluated research models.")
else:
    active = next((r for r in runs if r["status"] == "RUNNING"), None)
    st.write("**Current task:**", f"Training {run_label(active)}" if active else "No research experiment running")
    proposed = [p for p in proposals if p["status"] == "PROPOSED"]
    proposed.sort(key=lambda item: item["proposal"].get("priority_rank", 999))
    if proposed:
        st.caption(f"Next planned hypothesis: {proposed[0]['proposal'].get('claim', proposed[0]['hypothesis_id'])}")

if proposals:
    st.subheader("Hypotheses")
    for item in sorted(proposals, key=lambda p: (p["proposal"].get("priority_rank", 999), -p["created_at"])):
        proposal = item["proposal"]
        rank = proposal.get("priority_rank")
        title = f"#{rank} · {proposal.get('candidate_family', item['hypothesis_id'])} · {item['status'].title()}" if rank else f"{item['hypothesis_id']} · {item['status'].title()}"
        with st.expander(title):
            st.write(proposal.get("claim", "No claim recorded."))
            st.caption(f"Why this priority: {proposal.get('priority_rationale', 'No rationale recorded.')}")
            st.write("**Baseline:**", proposal.get("baseline", "—"))
            st.write("**Validation:**", proposal.get("evaluation_plan", "—"))
            st.write("**Shortcut risk:**", proposal.get("risk_and_shortcuts", "—"))
            if proposal.get("references"):
                st.write("**References:**", "; ".join(proposal["references"]))
if reviews:
    st.subheader("What the experiments taught us")
    for event in reviews[:8]:
        review = parse_event(event)
        matching = next((r for r in runs if r["run_id"] == event.get("run_id")), None)
        with st.expander(f"{run_label(matching) if matching else 'Research run'} · {format_date(event['time'])}"):
            st.write(review.get("objective_measured_results", "No measured-result summary recorded."))
            concerns = review.get("critic_concerns") or []
            if concerns:
                st.write("**Concerns:**", "; ".join(concerns))
            st.write("**Recommended next step:**", str(review.get("recommended_next_action", "Not recorded")).replace("_", " "))
            st.caption(f"Confidence: {review.get('confidence', 'Not recorded')}")
elif runs:
    st.subheader("What the experiments taught us")
    st.caption("Research runs exist, but no critic assessment has been recorded yet.")

index = json.loads((REPO / "literature" / "evidence_index.json").read_text())
with st.expander("Scientific references and research notes"):
    for source in index.get("sources", []):
        st.markdown(f"[{source['title']}]({source['url']}) · {source.get('status', 'reference')}")
    if scout_file.is_file():
        st.caption("Literature Scout notes are unverified leads until sources are checked.")
    for filename, title in (("discoveries.md", "Discoveries"), ("failed_approaches.md", "Failed approaches")):
        content = (REPO / "research_memory" / filename).read_text().strip()
        if content:
            st.markdown(f"**{title}**")
            st.markdown(content)
