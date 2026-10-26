"""Campaign explorer: comparisons, arm/seed coverage, scientific reviews."""
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import campaign_picker,table,metric_columns
from dashboard.research_data import comparison_records,campaign_runs,campaign_stats,comparison_matrix
from dashboard.presentation import format_duration

page_intro('Campaign explorer','Follow each scientific question through its arms, seeds, results and critique.')
campaign=campaign_picker()
if not campaign:
    st.info('No research campaigns recorded.')
else:
    records=comparison_records(campaign['id']);runs=campaign_runs(campaign['id']);stats=campaign_stats(campaign,runs,records)
    metric_columns([('Research duration',format_duration(stats['elapsed'])),('Execution-complete',stats['complete']),('Incomplete',stats['incomplete']),('Reviewed runs',stats['reviewed'])])
    st.caption(f"{campaign['name']} · {campaign['status'].replace('_',' ').title()} · {campaign.get('reason') or ''}")
    search=st.text_input('Search investigations',placeholder='Scientific question or hypothesis').lower().strip()
    statuses=st.multiselect('Comparison status',sorted({r['status'] for r in records}))
    filtered=[r for r in records if (not search or search in str(r['plan']).lower()) and (not statuses or r['status'] in statuses)]
    if not filtered:st.info('No comparisons match this campaign and search.')
    else:
        page=st.number_input('Page',1,max(1,(len(filtered)+7)//8),1)
        visible=filtered[(page-1)*8:page*8]
        table([{'Comparison':r['plan']['hypothesis_id'].replace('_',' '),'Status':r['status'],'Cells complete':f"{r['completed']} / {r['expected']}",'Arms':len(r['plan']['arms']),'Seeds':', '.join(map(str,r['plan']['seeds']))} for r in visible])
        chosen=st.selectbox('Investigation details',range(len(visible)),format_func=lambda i:visible[i]['plan']['hypothesis_id'].replace('_',' '))
        record=visible[chosen];plan=record['plan'];result=record['result'] or {}
        st.subheader(plan['hypothesis_id'].replace('_',' ').capitalize())
        st.write(f"**{record['status']}** · {plan['kind'].replace('_',' ')} · Baseline arm: {plan['baseline_arm_id']}")
        st.write(record['proposal'].get('claim') or plan.get('scientific_question') or plan.get('hypothesis') or plan.get('rationale') or plan.get('expected_observations',''))
        for arm in plan['arms']:st.write(f"**{arm['arm_id']}** — {arm['difference']}")
        table(comparison_matrix(record))
        st.caption('MISSING means no recorded job for that declared cell. Frozen-checkpoint jobs are inference investigations, not newly trained models.')
        left,right=st.columns(2)
        with left:
            st.subheader('Scientific review')
            if record['review']:
                review=record['review'];st.write('**'+str(review.get('hypothesis_outcome',review.get('hypothesis_status',review.get('hypothesis_decision',review.get('recommended_next_action',review.get('decision',review.get('verdict','Recorded'))))))).replace('_',' ').title()+'**')
                for field in ('summary','scientific_conclusion','evidence_summary','rationale','recommended_next_action','what_worked','what_failed','objective_measured_results','interpretation'):
                    if review.get(field):st.write(str(review[field])[:900]+('… See complete critique.' if len(str(review[field]))>900 else ''))
                with st.expander('Complete critique and limitations'):st.json(review)
            elif result.get('status','').startswith('COMPLETE'):st.info('Review pending or interrupted. Completed measurements remain valid; no critique is fabricated.')
            else:st.caption('Scientific review unavailable or not yet due.')
        with right:
            st.subheader('Evidence status')
            eligibility=result.get('scientific_eligibility',{})
            st.write(eligibility.get('evidence_level','Exploratory / not independently confirmed'))
            controls=result.get('controls',result.get('comparison_controls',{}))
            st.caption('Equal seeds do not prove matching source, mask or initialization streams.')
            if controls:
                st.write('Trusted paired controls: **'+controls.get('paired_control_verification',{}).get('status','NOT_MEASURED')+'**')
                for finding in controls.get('confounds',[]):st.caption(finding)
                with st.expander('Training-control receipts and confounds'):st.json(controls)
            if result.get('problems'):st.write(result['problems'])
        differences=result.get('paired_seed_differences',[])
        primary=[d for d in differences if d.get('measurement') in {'real:dev_real/logical_au_spro_0_05_compatible','real:dev_real/logical_pixel_ap_pooled'}]
        if primary:
            st.subheader('Paired localization differences')
            table(primary)
            st.caption('Paired seed labels; causal interpretation additionally requires the trusted controls above. Source-image intervals and training-seed replication are distinct.')
        with st.expander('Measurements, paired differences and execution evidence'):
            st.json(result or {'status':'No published result'})
        with st.expander('Frozen plan and controlled settings'):st.json(plan)
        failures=[r for r in record['runs'] if r.get('error')]
        if failures:
            with st.expander('Failure evidence',expanded=True):
                for r in failures:st.write(r['run_id']);st.code(r['error'][-8000:],language='text')
