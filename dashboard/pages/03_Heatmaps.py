import streamlit as st
from dashboard.data_access import list_runs, allowed_preview
st.title("Heatmap inspection")
st.caption("Only explicitly allowlisted development thumbnails; original official test images and masks are never displayed.")
runs = [r for r in list_runs() if r.get("artifact_dir")]
if not runs:
    st.info("No preview artifacts yet. Generate the artificial good-data demo or run an approved development experiment.")
    st.stop()
run_by_id = {r["run_id"]: r for r in runs}
run_id = st.selectbox("Development run", list(run_by_id))
selected = run_by_id[run_id]
st.write('**Candidate:**', selected['candidate_id'], '· **Run:**', run_id)
st.write('**Preview anomaly score (max map value):**', selected.get('metrics', {}).get('dev_synthetic/preview_anomaly_score', 'Not recorded for this run'))
columns = st.columns(5)
for column, filename, title in zip(columns,
    ('normal_preview.png', 'synthetic_input.png', 'synthetic_heatmap.png', 'synthetic_overlay.png', 'synthetic_mask.png'),
    ('Good source image', 'Synthetic input', 'Anomaly heatmap', 'Overlay', 'Synthetic development mask')):
    with column:
        st.subheader(title)
        preview = allowed_preview(selected, filename)
        if preview:
            st.image(str(preview), use_container_width=True)
        else:
            st.info("Not available")
st.caption("The example is a synthetic development artifact. Heatmap color scaling is for visual inspection, not calibrated global thresholding.")
