import streamlit as st
from dashboard.data_access import list_runs
from orchestrator.decision_engine import pareto_frontier
import pandas as pd
import plotly.express as px

st.title('Pareto trade-offs')
st.warning('Localization quality is a synthetic development proxy, never official LOCO sPRO.')
runs = list_runs()
front = set(pareto_frontier(runs))
measurements = {
    'Localization proxy versus latency': ('inference/e2e_ms_p95', 'dev_synthetic/pixel_auroc'),
    'Localization proxy versus model size': ('inference/checkpoint_mib', 'dev_synthetic/pixel_auroc'),
    'Latency versus model size': ('inference/checkpoint_mib', 'inference/e2e_ms_p95'),
    'Peak VRAM versus localization proxy': ('inference/peak_vram_mib', 'dev_synthetic/pixel_auroc'),
}
for title, (x, y) in measurements.items():
    st.subheader(title)
    rows = [{'run': r['run_id'], 'candidate': r['candidate_id'], x: r['metrics'][x], y: r['metrics'][y],
             'frontier': 'Nondominated' if r['run_id'] in front else 'Other'}
            for r in runs if x in r['metrics'] and y in r['metrics']]
    if rows:
        st.plotly_chart(px.scatter(pd.DataFrame(rows), x=x, y=y, color='frontier', hover_name='run', hover_data=['candidate']), use_container_width=True)
    else:
        st.info('Measurements for this comparison are not available yet.')
