"""Reusable, restrained Streamlit components for research facts."""
from __future__ import annotations

import streamlit as st

from dashboard.campaign_timing import training_seconds
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
    columns[1].metric("Training-loop duration", format_duration(training_seconds(run)))
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
        st.write("Training timing definition:", (run.get("config") or {}).get("training_completion_policy", "historical original definition"))
        st.write("Original training measurement:", format_duration((run.get("metrics") or {}).get("training/duration_seconds")))
        st.write("Learning computation (diagnostic):", format_duration((run.get("metrics") or {}).get("training/learning_seconds")))
        st.write("GPU-slot usage (diagnostic):", format_duration(run.get("gpu_seconds")))
        st.write("Run ID:", run.get("run_id") or "—")
        st.write("Git commit:", run.get("git_sha") or "—")
        st.write("Dataset fingerprint:", run.get("manifest_sha256") or "—")
        st.write("MLflow run ID:", run.get("mlflow_run_id") or "—")
        st.write("Started:", format_date(run.get("started_at")))
        st.write("Finished:", format_date(run.get("finished_at")))
        st.write("Seed:", (run.get("config") or {}).get("seed", "—"))
        st.write("Metric evaluator:", (run.get("config") or {}).get("metric_evaluator_version", "legacy development v1"))
        url = get_mlflow_url(run)
        if url:
            st.link_button("Open this run in MLflow", url)


def development_result_card(feedback):
    from dashboard.presentation import development_card_metrics
    st.subheader(feedback['candidate_id'].replace('_',' ').title())
    st.caption('Development · Real anomalies · Custom LOCO split, not official full-test results')
    comparison = feedback.get('scientific_comparison')
    if comparison:
        st.caption(f"Scientific comparison: {comparison['status'].replace('_',' ').title()} · "
                   f"{comparison['recorded_jobs']} / {comparison['expected_jobs']} planned jobs. Completion does not establish hypothesis support.")
        if comparison['status'] == 'INCOMPLETE':
            with st.expander('Missing comparison evidence'):
                st.write(comparison['problems'])
    values = development_card_metrics(feedback)
    for offset in range(0,len(values),3):
        for column,(label,value) in zip(st.columns(3),values[offset:offset+3]):
            column.metric(label,value)
    st.caption('Localization: compatible AU-sPRO at 5% FPR; original-scorer equivalence unverified. Pixel AP is supplementary.')
    with st.expander('Measurement details'):
        st.write('Evaluator:',feedback['evaluator_version'])
        st.write('Split:',feedback['split_id'])
        st.write('Run:',feedback['run_id'])
        st.write('Threshold:',feedback['metrics'].get('dev_real/normal_only_threshold','Unavailable'))
        st.write('Inference protocol:',feedback.get('measurement_protocol','See recorded evaluation provenance'))


def campaign_outcome_card(campaign, *, pending, tracking):
    from dashboard.presentation import format_duration
    import time
    used, limits = campaign['usage'], campaign.get('config', {}).get('limits', {})
    with st.container(border=True):
        title, state = st.columns([3, 2])
        title.markdown('**Campaign outcome**')
        completed = (campaign['status'] == 'COMPLETED' or
                     (campaign['status'] == 'LIMIT_REACHED' and campaign.get('reason') == 'experiment count reached'
                      and used.get('successful', 0) == used['experiments'] and used['experiments'] > 0))
        state.markdown('**Completed**' if completed else '**' + campaign['status'].replace('_', ' ').title() + '**')
        left, right = st.columns(2)
        left.metric('Experiments', f"{used['experiments']} / {limits.get('max_experiments', '—')}")
        right.metric('Codex calls', f"{campaign['codex_calls']} / {limits.get('max_codex_calls', '—')}")
        left.metric('GPU-slot time', f"{used['gpu_seconds_live']:.1f} / {limits.get('max_gpu_hours', 0)*3600:g} s")
        right.metric('Campaign duration', format_duration((campaign.get('ended_at') or time.time()) - campaign['created_at']))
        failure_count = campaign.get('failure_events', 0)
        failure_label = f'{failure_count} recorded failure events' if failure_count else 'No failures recorded'
        tracking_label = 'MLflow synchronized' if tracking == 'Connected' and pending == 0 else f'MLflow {tracking.lower()} · {pending} pending events'
        st.caption(f'{failure_label} · {tracking_label}')
        st.caption(f"{campaign['name']} · {campaign.get('reason') or 'Campaign in progress'}")


def scientific_results_comparison(feedback, run_ids=None):
    import pandas as pd
    import plotly.express as px
    from dashboard.presentation import real_comparison_rows, localization_change
    rows = real_comparison_rows(feedback, run_ids)
    st.subheader('Scientific results')
    st.caption('Real-anomaly development measurements, not synthetic scores. Custom LOCO split, not official full-test results.')
    if not rows:
        st.info('No real-development evaluations are recorded for this campaign yet.')
        return
    visible = [{k: ('Unavailable' if r[k] is None else f'{r[k]:.3f}') if k != 'Candidate' else r[k]
                for k in ('Candidate', 'Logical AUROC', 'Logical AU-sPRO', 'Pixel AP')} for r in rows]
    st.caption('Click a candidate row to inspect its research, implementation, experiment, and critique.')
    selection = st.dataframe(visible, hide_index=True, width='stretch',
                             on_select='rerun', selection_mode='single-row', key='scientific_candidate_selection')
    selected_rows = selection.selection.rows
    ordered_ids = [rid for rid in run_ids if any(f['run_id'] == rid for f in feedback)] if run_ids is not None else [f['run_id'] for f in feedback]
    if selected_rows and 0 <= selected_rows[0] < len(ordered_ids):
        candidate_research_panel(ordered_ids[selected_rows[0]])
    st.caption('AU-sPRO uses the compatible development evaluator at 5% FPR; equivalence to the original MVTec scorer is unverified. Pixel AP is supplementary.')
    protocols = {(r['split'], r['evaluator']) for r in rows}
    if len(protocols) != 1 or not rows[0]['split'] or not rows[0]['evaluator']:
        st.info('Select results with the same split and evaluator version to compare localization scores.')
        return
    measured = [r for r in rows if r['Logical AU-sPRO'] is not None]
    if measured:
        st.subheader('Logical anomaly localization')
        st.caption('Compatible AU-sPRO @5% · Higher is better')
        chart = px.bar(pd.DataFrame(measured), x='Logical AU-sPRO', y='Candidate', orientation='h',
                       text='Logical AU-sPRO', color_discrete_sequence=['#609ED4'])
        chart.update_traces(texttemplate='%{x:.3f}', textposition='outside')
        chart.update_layout(yaxis={'autorange': 'reversed', 'title': None,
                                  'categoryorder': 'array', 'categoryarray': [r['Candidate'] for r in measured]},
                            xaxis={'rangemode': 'tozero'}, showlegend=False,
                            height=max(260, len(measured)*65), margin=dict(l=0,r=45,t=5,b=20))
        st.plotly_chart(chart, width='stretch')
    change = localization_change(rows)
    if change is not None:
        st.caption(f'Last versus first evaluated candidate: {change:+.1f}% relative localization score change. '
                   'Descriptive comparison only; differing models and single seeds do not establish statistical significance or a causal improvement.')


def candidate_research_panel(run_id):
    from dashboard.data_access import candidate_research_details
    from urllib.parse import urlparse
    detail = candidate_research_details(run_id)
    if not detail:
        st.info('Research details are not recorded for this run.')
        return
    run, proposal, implementation, review = (detail[k] for k in ('run', 'proposal', 'implementation', 'review'))
    st.subheader('Candidate research and work performed')
    st.caption(run['candidate_id'].replace('_', ' ').capitalize())
    with st.expander('Research question and proposed method', expanded=True):
        for key, label in [('research_question','Research question'), ('claim','Hypothesis'),
                           ('candidate_family','Method family'), ('controlled_change','Proposed change'),
                           ('baseline','Planned baseline'), ('expected_observation','Predicted result'),
                           ('priority_rationale','Why Codex selected it'), ('evaluation_plan','Planned evaluation'),
                           ('risk_and_shortcuts','Known risks and shortcuts')]:
            if proposal.get(key):
                st.markdown(f'**{label}**')
                st.write(proposal[key])
        if not proposal:
            st.info('No linked research proposal was recorded.')
        st.caption('The proposal describes intended work. The execution and critique below show what was actually recorded.')
    with st.expander('Implementation and experiment execution', expanded=True):
        st.write('**Recorded status:**', run['status'].replace('_',' ').title())
        if run.get('error'):
            st.error(run['error'])
        for key, label in [('implemented_files','Files reported by the Implementer'),
                           ('external_weights','Declared external weights'), ('new_dependencies','Declared dependencies'),
                           ('caveats','Implementation caveats')]:
            if key in implementation:
                st.markdown(f'**{label}**')
                for value in implementation[key]:
                    st.write('• ' + str(value))
                if not implementation[key]:
                    st.caption('None declared.')
        if not implementation:
            st.caption('No Implementer response is linked to this run.')
        for event in detail['history']:
            if event['kind'] == 'CANDIDATE_TESTS':
                st.write('**Validation:**', event['payload'].get('tests', 'Not recorded'))
                st.code(event['payload'].get('output',''), language=None)
        config = run['config']
        settings = [{'Setting':key.replace('_',' ').capitalize(), 'Recorded value':str(value)}
                    for key,value in config.items() if isinstance(value,(str,int,float,bool))]
        st.dataframe(settings, hide_index=True, width='stretch')
        st.write('**Training duration:**', format_duration(run['metrics'].get('training/duration_seconds')))
        st.write('**GPU-slot usage:**', format_duration(run.get('gpu_seconds')))
        st.caption('Training settings and timings are ledger records; declared implementation details are agent reports.')
    with st.expander('Research sources and references'):
        references = proposal.get('references') or []
        if not references:
            st.info('No references were recorded for this hypothesis.')
        for reference in references:
            ref = str(reference)
            if urlparse(ref).scheme in ('https','http') and ' ' not in ref:
                st.link_button(ref, ref)
            else:
                st.write(ref)
        st.caption('These are references cited in the selected hypothesis. Listing a reference does not establish independent verification or reproduction.')
    with st.expander('What was learned and what happens next', expanded=True):
        if review:
            st.write(review.get('objective_measured_results', 'No measured-result assessment recorded.'))
            st.markdown('**Critic concerns**')
            for concern in review.get('critic_concerns', []):
                st.write('• ' + str(concern))
            st.write('**Recommended action:**', review.get('recommended_next_action','Not recorded').replace('_',' '))
            st.write('**Proposed falsification / next experiment:**', review.get('suggested_falsification','Not recorded'))
            st.caption('Recorded Codex critique; claims should be checked against the measured results above.')
        else:
            st.info('No Critic assessment has been recorded for this run yet.')
        for event in detail['history']:
            if event['kind'] in ('NEXT_DECISION','NEXT_DECISION_CORRECTION'):
                st.write('**Controller-recorded next decision:**', event['payload'].get('reason') or event['payload'].get('action'))
    with st.expander('Source and tracking provenance'):
        for key in ('run_id','hypothesis_id','git_sha','manifest_sha256','mlflow_run_id'):
            st.write(key.replace('_',' ').title()+':', run.get(key) or 'Not recorded')
        url = get_mlflow_url(run)
        if url:
            st.link_button('Open candidate run in MLflow', url)
