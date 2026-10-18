import streamlit as st
from dashboard.data_access import list_runs, get_mlflow_url
from dashboard.charts import run_table

st.title('Experiments')
st.warning('Synthetic development metrics are proxies, not official LOCO results.')
runs = list_runs()
if not runs:
    st.info('No experiments recorded yet.')
    st.stop()
stages = sorted({r['stage'] for r in runs})
statuses = sorted({r['status'] for r in runs})
selected_stages = st.multiselect('Stage', stages, default=stages)
selected_status = st.multiselect('Status', statuses, default=statuses)
filtered = [r for r in runs if r['stage'] in selected_stages and r['status'] in selected_status]
table = run_table(filtered)
for entry, run in zip(table, filtered):
    entry['mlflow_url'] = get_mlflow_url(run)
st.dataframe(table, use_container_width=True, hide_index=True)
if filtered:
    by_id = {r['run_id']: r for r in filtered}
    selected_ids = st.multiselect('Compare up to two runs', list(by_id), max_selections=2)
    for column, rid in zip(st.columns(max(1, len(selected_ids))), selected_ids):
        run = by_id[rid]
        with column:
            st.subheader(f"{run['candidate_id']} · {rid[:8]}")
            st.write('Status:', run['status'])
            if run.get('error'):
                st.error(run['error'])
            st.json({'run_id': rid, 'hypothesis': run['hypothesis_id'], 'git_sha': run['git_sha'],
                     'dataset_fingerprint': run['manifest_sha256'], 'configuration': run['config'],
                     'metrics': run['metrics']})
            url = get_mlflow_url(run)
            if url:
                st.link_button('Open MLflow run', url)
