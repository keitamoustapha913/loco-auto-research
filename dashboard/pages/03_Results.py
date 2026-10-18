"""Experiment details, development preview artifacts and scientific trade-offs."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import page_intro, run_details
from dashboard.data_access import allowed_preview, heatmap_sample, list_runs
from dashboard.presentation import (
    INFRASTRUCTURE, METRICS, classify_run, comparable_research_runs,
    experiment_rows, format_metric, metric_value, research_runs, research_summary, run_label,
)
from orchestrator.decision_engine import pareto_frontier

page_intro("Results", "Measured runs and development previews. Synthetic proxies are never official LOCO scores.")
runs = list_runs(500)
summary = research_summary(runs)
models_tab, heatmaps_tab, compare_tab = st.tabs(["Models", "Heatmaps", "Compare"])

with models_tab:
    st.caption(f"{summary['evaluated']} research models evaluated. {summary['infrastructure_completed']} infrastructure tests completed.")
    search, filter_col = st.columns([3, 1])
    query = search.text_input("Search models or hypotheses", placeholder="Candidate or hypothesis", key="models_search").strip().lower()
    show_tests = filter_col.toggle("Show infrastructure tests", value=False)
    default_ids = {r["run_id"] for r in research_runs(runs)}
    filtered = [r for r in runs if (show_tests or r["run_id"] in default_ids) and
                (not query or query in str(r.get("candidate_id", "")).lower() or query in str(r.get("hypothesis_id", "")).lower())]
    if not filtered:
        if not runs:
            st.info("No experiments have been recorded yet.")
        elif not show_tests and summary["evaluated"] == 0:
            st.info("No research model has been evaluated yet. Turn on ‘Show infrastructure tests’ to inspect the verified toy runs.")
        else:
            st.info("No runs match this search.")
    else:
        table = experiment_rows(filtered)
        visible = [{k: row[k] for k in ("Model", "Type", "Status", "Synthetic localization AUROC", "Latency (p95)", "GPU slot time", "Date")} for row in table]
        st.dataframe(visible, hide_index=True, width="stretch", height=min(360, 38 * (len(visible) + 1)),
                     column_config={"Synthetic localization AUROC": st.column_config.TextColumn(help=METRICS["dev_synthetic/pixel_auroc"][1]),
                                    "Latency (p95)": st.column_config.TextColumn(help=METRICS["inference/e2e_ms_p95"][1])})
        run_by_id = {r["run_id"]: r for r in filtered}
        selected = st.selectbox("Experiment details", list(run_by_id), format_func=lambda rid: run_label(run_by_id[rid], include_date=True))
        run_details(run_by_id[selected])

with heatmaps_tab:
    previews = [r for r in runs if allowed_preview(r, "normal_preview.png") and allowed_preview(r, "synthetic_heatmap.png")]
    if not previews:
        st.info("No development heatmap preview has been recorded yet.")
    else:
        by_id = {r["run_id"]: r for r in previews}
        picker, opacity_control = st.columns([3, 2], gap="large")
        selected = picker.selectbox("Experiment", list(by_id), format_func=lambda rid: run_label(by_id[rid], include_date=True), key="heatmap_run")
        opacity = opacity_control.slider("Heatmap opacity", 0, 100, 38, 5, format="%d%%") / 100
        run = by_id[selected]
        st.caption(f"{classify_run(run)} · Synthetic development sample 1 · {run.get('category', 'unknown')} good-image source")
        sample = heatmap_sample(run, opacity)
        if sample is None:
            st.error("The preview is incomplete, unsafe, or its image dimensions do not align.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.write("**Original good image**")
                st.image(sample["original"], width="stretch")
            with right:
                st.write("**Synthetic anomaly overlay**")
                st.image(sample["overlay"], width="stretch")
            score = metric_value(run, "dev_synthetic/preview_anomaly_score")
            st.caption(f"Preview anomaly score: {format_metric(score, 'score')} · Original coordinates: {sample['size'][0]} × {sample['size'][1]} px. The color map is scaled for visual inspection, not a calibrated threshold.")
            with st.expander("Inspect synthetic input, raw heatmap and generated mask"):
                c1, c2, c3 = st.columns(3)
                c1.image(sample["synthetic"], caption="Synthetic input", width="stretch")
                c2.image(sample["heatmap"], caption="Raw anomaly heatmap", width="stretch")
                if sample["mask"]:
                    c3.image(sample["mask"], caption="Generated synthetic mask", width="stretch")
                else:
                    c3.caption("No synthetic mask saved for this run.")

with compare_tab:
    comparable = comparable_research_runs(runs)
    if not comparable:
        st.info("Pareto comparison will appear after at least two different research candidates have complete, comparable development measurements on the same good-image manifest. Infrastructure tests are excluded.")
    else:
        st.caption("Research candidates only · same dataset fingerprint · synthetic localization proxy, not official LOCO performance")
        measurement = st.selectbox("Trade-off", [
            "Localization proxy vs latency", "Localization proxy vs model size",
            "Latency vs model size", "Peak GPU memory vs localization proxy",
        ])
        axes = {
            "Localization proxy vs latency": ("inference/e2e_ms_p95", "dev_synthetic/pixel_auroc"),
            "Localization proxy vs model size": ("inference/checkpoint_mib", "dev_synthetic/pixel_auroc"),
            "Latency vs model size": ("inference/checkpoint_mib", "inference/e2e_ms_p95"),
            "Peak GPU memory vs localization proxy": ("inference/peak_vram_mib", "dev_synthetic/pixel_auroc"),
        }
        x_key, y_key = axes[measurement]
        frontier = set(pareto_frontier(comparable, include_resources=True))
        points = [{"Candidate": run_label(r), "Frontier": "Nondominated" if r["run_id"] in frontier else "Other",
                   "Run": r["run_id"][:8], x_key: metric_value(r, x_key), y_key: metric_value(r, y_key)} for r in comparable]
        labels = {key: METRICS[key][0] + (" (ms)" if METRICS[key][2] == "ms" else " (MiB)" if METRICS[key][2] == "MiB" else "") for key in (x_key, y_key)}
        chart = px.scatter(pd.DataFrame(points), x=x_key, y=y_key, color="Frontier", hover_name="Candidate",
                           hover_data=["Run"], labels=labels, color_discrete_map={"Nondominated": "#1D6B66", "Other": "#8A94A3"})
        chart.update_traces(marker={"size": 13})
        chart.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(chart, width="stretch")
        with st.expander("How this comparison is selected"):
            st.write("Only completed research runs with localization proxy, latency, checkpoint size and peak GPU memory measurements are eligible. The frontier uses those measured axes and model parameter count. Synthetic proxies cannot establish official LOCO performance.")
