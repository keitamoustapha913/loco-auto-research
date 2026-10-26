"""Compact shared navigation and presentation; scientific records stay read-only."""
import streamlit as st
from dashboard.research_data import campaigns
from dashboard.data_access import campaign_summary
from dashboard.presentation import format_date


def campaign_picker():
    records=campaigns()
    if not records:return None
    ids=[r['id'] for r in records];by_id={r['id']:r for r in records}
    old=st.session_state.get('selected_campaign',ids[0]);index=ids.index(old) if old in ids else 0
    chosen=st.sidebar.selectbox('Campaign',ids,index=index,
        format_func=lambda cid:f"{format_date(by_id[cid]['created_at'])} · {by_id[cid]['status'].replace('_',' ').title()}",key='campaign_picker')
    st.session_state['selected_campaign']=chosen
    st.sidebar.caption(by_id[chosen]['name']+' · '+chosen[:8])
    return campaign_summary(chosen)


def metric_columns(values):
    for col,(name,value) in zip(st.columns(len(values)),values):col.metric(name,value)


def table(data, **kwargs):
    return st.dataframe(data,hide_index=True,width='stretch',**kwargs)
