"""One model, one result card. Browsing and comparisons are optional."""
import streamlit as st
from dashboard.components import page_intro
from dashboard.view import campaign_picker,table
from dashboard.research_data import model_results,active_incumbent,leaderboard_row,SPRO,AP
from dashboard.detail_views import model_label,model_card
from dashboard.data_access import latest_multiseed_comparison

page_intro('Results','Logical anomaly localization on real development images.')
@st.dialog('Model result',width='large')
def inspect_model(record,reference):
    model_card(record,reference,key='inspect_'+record['run_id'])

campaign=campaign_picker()
feedback=model_results();incumbent=active_incumbent(campaign)
incumbent_id=incumbent['run_id'] if incumbent else None
by_id={f['run_id']:f for f in feedback}
if not feedback:
    st.info('No evaluated models yet.')
else:
    default=incumbent_id if incumbent_id in by_id else feedback[-1]['run_id']
    ids=[default]+[f['run_id'] for f in reversed(feedback) if f['run_id']!=default]
    selected=st.selectbox('Model',ids,format_func=lambda rid:model_label(by_id[rid]))
    model_card(by_id[selected],incumbent_id)

    with st.expander('Browse and compare models'):
        browse,compare=st.tabs(['Browse','Compare'])
        with browse:
            search=st.text_input('Search models',placeholder='Name, arm, seed or run ID').lower().strip()
            scope=st.radio('Campaign',['All campaigns','Selected campaign'],horizontal=True)
            filtered=[f for f in feedback if (scope=='All campaigns' or campaign and f.get('campaign_id')==campaign['id']) and
                (not search or search in (model_label(f)+' '+f['run_id']).lower())]
            st.caption(f'{len(filtered)} evaluated model runs · Execution order; click column headings to sort')
            if filtered:
                pages=max(1,(len(filtered)+19)//20)
                page=st.selectbox('Page',range(1,pages+1)) if pages>1 else 1
                visible=filtered[(page-1)*20:page*20]
                event=table([{k:v for k,v in leaderboard_row(f).items() if k in {'#','Started','Candidate','Arm','Seed','Logical AU-sPRO','Logical pixel AP','Normal FP','Artifacts'}}
                    for f in visible],on_select='rerun',selection_mode='single-row',key='browse:'+','.join(f['run_id'] for f in visible),column_config={
                        '#':st.column_config.NumberColumn('Experiment',format='%d',width='small'),
                        'Started':st.column_config.TextColumn('Started · Paris',width='medium'),
                        'Logical AU-sPRO':st.column_config.NumberColumn(format='%.4f'),
                        'Logical pixel AP':st.column_config.NumberColumn(format='%.4f')})
                opened=visible[event.selection.rows[0]]['run_id'] if event.selection.rows else None
                identity=(tuple(f['run_id'] for f in visible),opened)
                if opened and st.session_state.get('opened_model_row')!=identity:
                    st.session_state['opened_model_row']=identity
                    inspect_model(by_id[opened],incumbent_id)
                elif not opened:st.session_state['opened_model_row']=None
            else:st.info('No models match this search.')

        with compare:
            chosen=st.multiselect('Choose models',ids,format_func=lambda rid:model_label(by_id[rid]),max_selections=12)
            if len(chosen)>=2:
                records=[by_id[rid] for rid in chosen]
                keys=('split_id','evaluator_version','training_manifest_sha256','inference_device','measurement_protocol','threshold_source')
                compatible=all(all(f.get(k) is not None and f.get(k)==records[0].get(k) for k in keys) for f in records)
                table([leaderboard_row(f) for f in records])
                if not compatible:st.warning('Protocols differ or are unverified. These values do not establish a paired improvement.')
                else:
                    reference=records[0];differences=[]
                    for f in records[1:]:
                        differences.append({'Model':model_label(f),
                            'Δ logical sPRO':f['metrics'][SPRO]-reference['metrics'][SPRO] if all(x['metrics'].get(SPRO) is not None for x in (f,reference)) else None,
                            'Δ pixel AP':f['metrics'][AP]-reference['metrics'][AP] if all(x['metrics'].get(AP) is not None for x in (f,reference)) else None})
                    table(differences)
                    st.caption('Descriptive differences from the first model. Research contains the paired-control evidence, critiques and uncertainty.')

study=latest_multiseed_comparison()
if study:
    with st.expander('Historical synthetic comparison'):
        st.subheader('Three-seed comparison')
        st.caption('Synthetic proxy measurements; not real-anomaly results.')
        table([{'Arm':arm,'Primary AP':study['across_seed'][arm]['primary_pixel_ap']['mean'],
            'Seed SD':study['across_seed'][arm]['primary_pixel_ap']['sample_sd']} for arm in study['arms']])
        st.json(study['paired_differences'])
