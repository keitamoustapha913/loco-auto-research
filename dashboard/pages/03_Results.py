"""Scientific model results and retained visual evidence; no campaign overview dump."""
import streamlit as st
from dashboard.components import page_intro,candidate_research_panel
from dashboard.view import campaign_picker,table,metric_columns
from dashboard.research_data import model_results,active_incumbent,leaderboard_row,artifact_status,LABEL,SPRO,AP
from dashboard.data_access import real_heatmap_sample
from dashboard.campaign_timing import training_seconds
from dashboard.presentation import format_duration

page_intro('Model results','Compare logical localization, false positives and inference efficiency.')
campaign=campaign_picker()
st.caption(LABEL+' Compatible AU-sPRO@5%; original-scorer equivalence remains unverified. Missing scores are unavailable, never zero.')
scope=st.radio('Result scope',['All recorded models','Selected campaign'],horizontal=True)
feedback=model_results(campaign['id'] if campaign and scope=='Selected campaign' else None)
incumbent=active_incumbent(campaign);incumbent_id=incumbent['run_id'] if incumbent else None
with st.expander('Filter and sort models'):
    query=st.text_input('Search models',placeholder='Candidate, arm, seed or run ID').strip().lower()
    a,b,c=st.columns(3)
    family=a.multiselect('Model family',sorted({f['candidate_id'] for f in feedback}))
    comparisons=b.multiselect('Comparison',sorted({f['config'].get('plan_id','Legacy / unassigned') for f in feedback}))
    seeds=c.multiselect('Training seed',sorted({str(f['config'].get('seed','Unavailable')) for f in feedback}))
    a,b=st.columns(2)
    statuses=a.multiselect('Run status',sorted({f['run']['status'] for f in feedback}))
    order=b.selectbox('Sort models',['Execution order','Latest first','Logical AU-sPRO','Logical pixel AP'])
filtered=[f for f in feedback if (not query or query in (f['candidate_id']+' '+str(f['config'].get('seed'))+' '+f['run_id']+' '+str(f['config'].get('plan_arm',''))).lower()) and (not family or f['candidate_id'] in family) and (not comparisons or f['config'].get('plan_id','Legacy / unassigned') in comparisons) and (not seeds or str(f['config'].get('seed','Unavailable')) in seeds) and (not statuses or f['run']['status'] in statuses)]
if order=='Latest first':filtered=list(reversed(filtered))
elif order in ['Logical AU-sPRO','Logical pixel AP']:filtered=sorted(filtered,key=lambda f:f['metrics'].get(SPRO if order=='Logical AU-sPRO' else AP,-1),reverse=True)
if not filtered:
    st.info('No evaluated models match this scope. Numerical records remain available after artifact retirement.')
else:
    st.subheader('Real-anomaly development results')
    tabs=st.tabs(['Model details','Leaderboard','Compare selected models'])
    with tabs[1]:
        st.caption(f'{len(filtered)} evaluated model runs. Click a row to inspect its measurements. Frozen readouts retain their original training seed.')
        page=st.number_input('Model page',1,max(1,(len(filtered)+24)//25),1)
        visible=filtered[(page-1)*25:page*25]
        grid=[]
        for f in visible:
            row=leaderboard_row(f);row['Research reference']='Active incumbent' if f['run_id']==incumbent_id else 'Alternative';grid.append(row)
        event=table(grid,on_select='rerun',selection_mode='single-row',key='model_leaderboard:'+','.join(f['run_id'] for f in visible),
            column_config={'Logical AU-sPRO':st.column_config.NumberColumn(format='%.4f'),'Logical pixel AP':st.column_config.NumberColumn(format='%.4f'),'Run':st.column_config.TextColumn(width='small')})
    with tabs[0]:
        ids=[f['run_id'] for f in filtered];by_id={f['run_id']:f for f in filtered}
        default=incumbent_id if incumbent_id in ids else max(filtered,key=lambda f:f['run']['created_at'])['run_id']
        if event.selection.rows:
            chosen=visible[event.selection.rows[0]]['run_id']
            st.session_state['selected_model']=chosen
        remembered=st.session_state.get('selected_model',default)
        selected=st.selectbox('Model details',ids,index=ids.index(remembered) if remembered in ids else ids.index(default),format_func=lambda rid:f"{by_id[rid]['candidate_id'].replace('_',' ')} · {by_id[rid]['config'].get('plan_arm','legacy')} · seed {by_id[rid]['config'].get('seed','?')} · {rid[:8]}")
        f=by_id[selected];m=f['metrics'];cfg=f['config'];run=f['run']
        with st.container(border=True):
            st.subheader(f['candidate_id'].replace('_',' ').capitalize())
            st.caption(f"{cfg.get('plan_arm','Legacy arm')} · seed {cfg.get('seed','Unavailable')} · {selected[:8]} · {run['status'].title()}")
            if selected==incumbent_id:
                st.success('Active exploratory research incumbent · 80% localization-weighted selection. Scientific confirmation is separate.')
            comparison=f.get('scientific_comparison',{})
            st.caption('Scientific evidence: '+str(comparison.get('scientific_eligibility',{}).get('evidence_level','Exploratory; confirmed superiority not established')))
            score=lambda key: f"{m[key]:.4f}" if m.get(key) is not None else 'Unavailable'
            metric_columns([('Logical AU-sPRO@5%',score(SPRO)),('Logical pixel AP',score(AP)),('Logical image AUROC',score('dev_real/logical_image_auroc'))])
            metric_columns([('Normal false positives',f"{m.get('dev_real/normal_false_positives','?')} / {m.get('dev_real/normal_images','?')}"),('Structural localization',score('dev_real/structural_au_spro_0_05_compatible')),('Inference p95 · '+f.get('inference_device','unknown').upper(),f"{m['dev_real/inference_e2e_ms_p95']:.1f} ms" if m.get('dev_real/inference_e2e_ms_p95') is not None else 'Unavailable')])
            parameters=m.get('dev_real/parameters');size=m.get('dev_real/serialized_model_bytes');ram=m.get('dev_real/peak_cpu_rss_mib')
            metric_columns([('Parameters',f'{int(parameters):,}' if parameters is not None else 'Unavailable'),('Serialized model',f'{size/2**20:.3f} MiB' if size is not None else 'Unavailable'),('Peak process RAM',f'{ram:.1f} MiB' if ram is not None else 'Unavailable')])
            duration=training_seconds(run)
            st.caption('Training-loop duration: '+(format_duration(duration) if duration is not None else 'Unavailable / historical timing definition'))
            availability,message=artifact_status(f);st.write('**Artifacts: '+availability+'** · '+message)
            if availability=='Available' and st.checkbox('Open localization viewer',key='open_localization'):
                index=st.number_input('Development image index',0,164,0,1)
                try:
                    sample=real_heatmap_sample(f,int(index))
                    for col,key,label in zip(st.columns(4),('original','heatmap','overlay','ground_truth'),('Original','Heatmap','Overlay','Ground truth')):col.image(sample[key],caption=label,width='stretch')
                    st.caption('Original-resolution alignment and hashes verified. Display-only mask union; compatible sPRO retains individual regions and saturation rules.')
                except FileNotFoundError:st.error('Artifact missing unexpectedly. No intentional archival marker was found.')
                except PermissionError:st.warning('Artifact temporarily unavailable: permission or access failure.')
                except (ValueError,KeyError,TypeError) as exc:st.error('Artifact integrity verification failed: '+str(exc))
                except OSError as exc:st.warning('Artifact temporarily unavailable: '+str(exc))
            with st.expander('Measurement protocol, efficiency and provenance'):
                st.json({k:f.get(k) for k in ('split_id','evaluator_version','checkpoint_sha256','source_sha256','measurement_protocol','inference_device','threshold_source')})
                st.json(m)
            with st.expander('Candidate implementation and research references'):candidate_research_panel(selected)
    with tabs[2]:
        by_id={f['run_id']:f for f in filtered}
        chosen=st.multiselect('Models to compare',list(by_id),format_func=lambda rid:by_id[rid]['candidate_id'].replace('_',' ')+f" · seed {by_id[rid]['config'].get('seed','?')} · {rid[:8]}",max_selections=12)
        selected=[by_id[rid] for rid in chosen]
        if len(selected)<2:st.info('Select at least two models. Comparisons require matching dataset and measurement protocols.')
        else:
            keys=('split_id','evaluator_version','training_manifest_sha256','inference_device','measurement_protocol','threshold_source')
            compatible=all(all(f.get(k) is not None and f.get(k)==selected[0].get(k) for k in keys) for f in selected)
            if not compatible:st.warning('Incompatible or unverified measurement protocols. Values are shown separately; no paired improvement is claimed.')
            table([leaderboard_row(f) for f in selected])
            if compatible and all(f['metrics'].get(SPRO) is not None and f['metrics'].get(AP) is not None for f in selected):
                st.scatter_chart([{'Candidate':f['candidate_id'],'Seed':str(f['config'].get('seed','?')),'Logical AU-sPRO':f['metrics'][SPRO],'Logical pixel AP':f['metrics'][AP]} for f in selected],x='Logical AU-sPRO',y='Logical pixel AP',color='Candidate')
            if compatible:
                baseline=selected[0];deltas=[]
                for f in selected[1:]:
                    paired=(f['config'].get('plan_id') and f['config'].get('plan_id')==baseline['config'].get('plan_id') and f['config'].get('seed')==baseline['config'].get('seed'))
                    deltas.append({'Model':f['candidate_id'],'Seed':f['config'].get('seed'),'Same plan / seed':bool(paired),
                        'Δ logical sPRO':f['metrics'][SPRO]-baseline['metrics'][SPRO] if f['metrics'].get(SPRO) is not None and baseline['metrics'].get(SPRO) is not None else None,'Δ pixel AP':f['metrics'][AP]-baseline['metrics'][AP] if f['metrics'].get(AP) is not None and baseline['metrics'].get(AP) is not None else None})
                table(deltas)
                st.caption('Descriptive differences relative to the first selection. Matching seed labels alone does not establish causal control or seed uncertainty. Review trusted joins and critiques on Research.')

# Historical synthetic studies remain inspectable, clearly separated from real
# development results; they never become the primary model ranking.
from dashboard.data_access import latest_multiseed_comparison
study=latest_multiseed_comparison()
if study:
    with st.expander('Historical synthetic comparison'):
        st.subheader('Three-seed comparison')
        st.caption('Synthetic development proxy only; independent of real-anomaly results above.')
        table([{'Arm':arm,'Primary AP mean':study['across_seed'][arm]['primary_pixel_ap']['mean'],
            'Between-seed SD':study['across_seed'][arm]['primary_pixel_ap']['sample_sd'],
            'Alternate AP mean':study['across_seed'][arm]['alternate_pixel_ap']['mean']} for arm in study['arms']])
        st.json(study['paired_differences'])
