"""Small read-only cumulative progress component shared by existing pages."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from orchestrator.model_registry import SPRO, AP


def progress_card(summary):
    if not summary or not summary['cohorts']:
        st.caption('No frozen real-development incumbent retained yet. No confirmed localization improvement.')
        return
    st.subheader('Best retained development models')
    for group in summary['cohorts']:
        research=group.get('research_incumbent')
        model=research or group['champion'];metrics=model['metrics'];state=group['state']
        selection=group.get('research_state') or {}
        confirmed=state['confirmation']=='REPEATED_SEED_SUPPORTED' and model['run_id']==group['champion']['run_id']
        st.write(f"**{model['candidate_id']}** · " + ('Repeated-seed development support' if confirmed else 'Active exploratory research incumbent — superiority unconfirmed'))
        if selection:
            score=selection['selection']['score']
            st.metric('Research selection score · 80% localization',f'{score:.4f}',help='60% logical sPRO, 20% pixel AP, 10% normal false-positive performance, 10% efficiency. Fixed normalization; comparable cohorts only.')
            delta=selection.get('localization_delta',{})
            st.caption(f"Change from previous research reference: AU-sPRO {delta.get(SPRO,0):+.4f} · pixel AP {delta.get(AP,0):+.4f}")
        a,b,c=st.columns(3)
        a.metric('Retained logical AU-sPRO@5%',f'{metrics[SPRO]:.4f}',help='Compatible development scorer; original-code equivalence unverified.')
        b.metric('Retained logical pixel AP',f'{metrics[AP]:.4f}')
        c.metric('Normal false positives',f"{int(metrics['dev_real/normal_false_positives'])} / {int(metrics['dev_real/normal_images'])}")
        st.caption(f"{model['conditions']['split_id']} · {model['conditions']['evaluator_version']} · "
                   f"{len(group['pareto_retention_ids'])} Pareto alternatives · "
                   + ('Confirmed relative development gain; no final-holdout claim.' if confirmed else 'No confirmed champion improvement in this cohort.'))
        if confirmed:st.caption(state.get('comparison_scope','Paired development comparison'))
        with st.expander('Selection details and scientific confirmation'):
            if selection:
                st.write(selection['selection'])
                st.write(selection['deployment_readiness'])
            st.caption('Scientific registry reference: '+group['champion']['candidate_id']+' · '+state['confirmation'])
            localization_chart(summary.get('localization_history',{}).get(group['cohort_id'],[]),key='localization-history:'+group['cohort_id'])
    st.caption(f"{summary['complete_comparisons']} complete controlled comparisons · "
               f"{summary.get('complete_exploratory_comparisons',0)} complete exploratory · "
               f"{summary['incomplete_comparisons']} incomplete · "
               f"{summary['attempts_since_verified_improvement']} attempts since last confirmed improvement")


def localization_chart(history,key=None):
    st.markdown('**Best retained logical localization score**')
    if not history:
        st.caption('No recorded retention history yet.')
        return
    rows=pd.DataFrame(history)
    dates=pd.to_datetime(rows['time'],unit='s',utc=True).dt.strftime('%Y-%m-%d %H:%M UTC')
    custom=list(zip(rows['candidate'],rows['confirmation'],rows['event'],dates))
    fig=go.Figure(go.Scatter(x=rows['attempt'],y=rows['logical_spro'],mode='lines+markers',
        line={'shape':'hv','color':'#487aa5','width':2},
        marker={'size':7,'color':['#487aa5' if c=='REPEATED_SEED_SUPPORTED' else '#8b9298' for c in rows['confirmation']]},
        customdata=custom,hovertemplate='Research attempt %{x}<br>Retained AU-sPRO: %{y:.4f}<br>%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>'))
    fig.update_layout(height=270,margin={'l':10,'r':10,'t':10,'b':10},showlegend=False,
        xaxis_title='Research attempts',yaxis_title='Logical AU-sPRO@5% · development',
        yaxis={'range':[0,1]},xaxis={'tickformat':'d','dtick':max(1,int((rows['attempt'].max()-rows['attempt'].min())/6))})
    st.plotly_chart(fig,width='stretch',key=key)
    st.caption('Recorded retention history only. Failed or unsuccessful challengers do not raise the line. '
               'Gray points are exploratory; supported promotions are blue. Explicit operator rollbacks are shown in tooltips. '
               'Compatible development scorer; not independent-holdout performance.')
    if not any(row['event']=='Supported development promotion' for row in history):
        st.caption('No supported promotions recorded yet. Tracking starts at retention initialization; earlier experiments were not retrospectively certified.')
