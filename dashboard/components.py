"""Reusable, restrained Streamlit components for research facts."""
from __future__ import annotations

import streamlit as st

from dashboard.data_access import get_mlflow_url
from dashboard.presentation import METRICS, classify_run, format_date, format_duration, format_metric, metric_value


def page_intro(title: str, description: str):
    st.title(title)
    st.caption(description)


def run_details(run: dict, *, compact=False):
    st.subheader(str(run.get("candidate_id") or "Unnamed model").replace("_", " "))
    st.caption(f"{classify_run(run)} · {run.get('category', 'Unknown dataset')} · {format_date(run.get('created_at'))}")
    if run.get("error"):
        st.error(f"Run failed: {run['error']}")
    columns = st.columns(3)
    columns[0].metric("Status", str(run.get("status") or "Unknown").title())
    columns[1].metric("GPU time", format_duration(run.get("gpu_seconds")))
    columns[2].metric("Inference latency", format_metric(metric_value(run, "inference/e2e_ms_p95"), "ms"))
    if not compact:
        st.write("**Research hypothesis:**", str(run.get("hypothesis_id") or "—").replace("_", " "))
        if (run.get('config') or {}).get('context_arm'):
            st.caption(f"Context arm: {run['config']['context_arm'].title()}")
        for key in ("dev_synthetic/pixel_auroc", "normal/image_fpr_at_threshold", "inference/checkpoint_mib", "inference/peak_vram_mib"):
            label, help_text, kind = METRICS[key]
            st.metric(label, format_metric(metric_value(run, key), kind), help=help_text)
        if metric_value(run, 'dev_synthetic/pixel_ap') is not None:
            with st.expander('Synthetic ablation diagnostics'):
                for key in ('dev_synthetic/pixel_ap', 'dev_synthetic/alternate_pixel_auroc',
                            'dev_synthetic/alternate_pixel_ap', 'dev_synthetic/relocated_pixel_ap',
                            'robustness/artifact_matched_good_fpr',
                            'robustness/photometric_good_fpr'):
                    label, help_text, kind = METRICS[key]
                    st.metric(label, format_metric(metric_value(run, key), kind), help=help_text)
        st.caption("Synthetic measurements describe development examples only; no official LOCO score is shown here.")
    with st.expander("Technical provenance"):
        st.write("Run ID:", run.get("run_id") or "—")
        st.write("Git commit:", run.get("git_sha") or "—")
        st.write("Dataset fingerprint:", run.get("manifest_sha256") or "—")
        st.write("MLflow run ID:", run.get("mlflow_run_id") or "—")
        st.write("Started:", format_date(run.get("started_at")))
        st.write("Finished:", format_date(run.get("finished_at")))
        st.write("Seed:", (run.get("config") or {}).get("seed", "—"))
        url = get_mlflow_url(run)
        if url:
            st.link_button("Open this run in MLflow", url)
