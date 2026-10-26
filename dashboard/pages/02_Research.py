"""A short campaign inbox; open an investigation to examine its science."""
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import campaign_picker,table
from dashboard.research_data import comparison_records,campaign_runs,campaign_stats
from dashboard.detail_views import readable_name,comparison_details
from dashboard.presentation import format_duration
from dashboard.experiment_timeline import timeline_rows,timestamp

page_intro('Research','Experiments in execution order, with their times and scientific findings.')
campaign=campaign_picker()


@st.dialog('Investigation',width='large')
def inspect_comparison(record):
    comparison_details(record)


if not campaign:
    st.info('No research campaigns recorded.')
else:
    records=comparison_records(campaign['id']);stats=campaign_stats(campaign,campaign_runs(campaign['id']),records)
    st.caption(f"{stats['complete']} comparisons complete · {stats['incomplete']} incomplete · {stats['trained']} models trained · {format_duration(stats['elapsed'])}")
    timeline,investigations=st.tabs(['Timeline','Investigations'])
    with timeline:
        chronology=timeline_rows(campaign)
        st.caption('Experiment numbers follow actual job starts, oldest first. Times: Europe/Paris. Studies are scientific comparisons; each arm/seed is a separate experiment.')
        if not chronology:st.info('No model jobs recorded for this campaign yet.')
        else:
            pages=max(1,(len(chronology)+14)//15)
            page=st.selectbox('Timeline page',range(1,pages+1),format_func=lambda p:f'Page {p} of {pages}') if pages>1 else 1
            table(chronology[(page-1)*15:page*15],height=460,column_order=['#','Started','Training','Job time','Status','Model','Arm','Seed','Study','Finished','Evaluation','Type','Run'],column_config={
                '#':st.column_config.NumberColumn('Experiment',width='small',format='%d'),
                'Started':st.column_config.TextColumn('Started · Paris',width='medium'),
                'Training':st.column_config.TextColumn(width='small'),
                'Job time':st.column_config.TextColumn(width='small'),
                'Status':st.column_config.TextColumn(width='small')})
            st.caption('Training is measured training-loop time. Job time includes setup and saving; the separate development evaluation is listed in its own column. Critique time is shared by the study, not assigned to individual models. Missing timestamps are not estimated.')
    with investigations:
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
                    if title.button(str(record['order'])+'. '+readable_name(record['plan']['hypothesis_id']),key='study:'+record['plan']['plan_id'],type='tertiary',width='stretch'):
                        inspect_comparison(record)
                    status.caption(record['status']+f" · {record['completed']}/{record['expected']}")
                    start=min((r['started_at'] for r in record['runs'] if r.get('started_at') is not None),default=None)
                    st.caption(('Started '+timestamp(start)) if start is not None else 'Planned '+timestamp(record['registered_at'])+' · No job started')
