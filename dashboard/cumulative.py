"""Small read-only cumulative progress component shared by existing pages."""
import streamlit as st

from orchestrator.model_registry import SPRO, AP


def progress_card(summary):
    if not summary or not summary['cohorts']:
        st.caption('No frozen real-development incumbent retained yet. No confirmed localization improvement.')
        return
    st.subheader('Best retained development models')
    for group in summary['cohorts']:
        model=group['champion'];metrics=model['metrics'];state=group['state']
        confirmed=state['confirmation']=='REPEATED_SEED_SUPPORTED'
        st.write(f"**{model['candidate_id']}** · " + ('Repeated-seed development support' if confirmed else 'Exploratory incumbent — superiority unconfirmed'))
        a,b,c=st.columns(3)
        a.metric('Retained logical AU-sPRO@5%',f'{metrics[SPRO]:.4f}',help='Compatible development scorer; original-code equivalence unverified.')
        b.metric('Retained logical pixel AP',f'{metrics[AP]:.4f}')
        c.metric('Normal false positives',f"{int(metrics['dev_real/normal_false_positives'])} / {int(metrics['dev_real/normal_images'])}")
        st.caption(f"{model['conditions']['split_id']} · {model['conditions']['evaluator_version']} · "
                   f"{len(group['pareto_retention_ids'])} Pareto alternatives · "
                   + ('Confirmed relative development gain; no final-holdout claim.' if confirmed else 'No confirmed champion improvement in this cohort.'))
        if confirmed:st.caption(state.get('comparison_scope','Paired development comparison'))
    st.caption(f"{summary['complete_comparisons']} complete controlled comparisons · "
               f"{summary['incomplete_comparisons']} incomplete · "
               f"{summary['attempts_since_verified_improvement']} attempts since last confirmed improvement")
