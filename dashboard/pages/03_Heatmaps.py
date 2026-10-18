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
c1, c2, c3 = st.columns(3)
for column, filename, title in ((c1,"normal_preview.png","Good image"),(c2,"synthetic_heatmap.png","Synthetic anomaly heatmap"),(c3,"synthetic_overlay.png","Synthetic overlay")):
    with column:
        st.subheader(title)
        preview = allowed_preview(selected, filename)
        if preview:
            st.image(str(preview), use_container_width=True)
        else:
            st.info("Not available")
st.caption("The example is a synthetic development artifact. Heatmap color scaling is for visual inspection, not calibrated global thresholding.")
