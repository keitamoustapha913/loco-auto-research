"""A short campaign inbox; open an investigation to examine its science."""
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import campaign_picker
from dashboard.research_data import comparison_records,campaign_runs,campaign_stats
from dashboard.detail_views import readable_name,comparison_details
from dashboard.presentation import format_duration

page_intro('Research','Choose an investigation to see its findings and model runs.')
campaign=campaign_picker()


@st.dialog('Investigation',width='large')
def inspect_comparison(record):
    comparison_details(record)


if not campaign:
    st.info('No research campaigns recorded.')
else:
    records=comparison_records(campaign['id']);stats=campaign_stats(campaign,campaign_runs(campaign['id']),records)
    st.caption(f"{stats['complete']} comparisons complete · {stats['incomplete']} incomplete · {stats['trained']} models trained · {format_duration(stats['elapsed'])}")
    search_box,filters=st.columns([3,1])
    search=search_box.text_input('Search investigations',placeholder='Search this campaign',label_visibility='collapsed').lower().strip()
    with filters.popover('Filter'):
        statuses=st.multiselect('Status',sorted({r['status'] for r in records}))
    filtered=[r for r in records if (not search or search in str(r['plan']).lower()) and (not statuses or r['status'] in statuses)]
    if not filtered:st.info('No comparisons match this campaign and search.')
    else:
        pages=max(1,(len(filtered)+5)//6)
        if pages>1:
            _,control=st.columns([3,2]);page=control.selectbox('Page',range(1,pages+1),format_func=lambda p:f'Page {p} of {pages}',label_visibility='collapsed')
        else:page=1
        st.caption(f'{len(filtered)} investigations · Click a title to open')
        for record in filtered[(page-1)*6:page*6]:
            with st.container(border=True):
                title,status=st.columns([4,1])
                if title.button(readable_name(record['plan']['hypothesis_id']),key='study:'+record['plan']['plan_id'],type='tertiary',width='stretch'):
                    inspect_comparison(record)
                status.caption(record['status']+f" · {record['completed']}/{record['expected']}")
