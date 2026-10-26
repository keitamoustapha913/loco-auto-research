"""Progressive disclosure: concise answers first, scientific records on demand."""
import re
import streamlit as st
from dashboard.view import table,metric_columns
from dashboard.research_data import SPRO,AP,LABEL,artifact_status,comparison_matrix
from dashboard.presentation import format_duration
from dashboard.campaign_timing import training_seconds
from dashboard.data_access import real_heatmap_sample
from dashboard.components import candidate_research_panel
from dashboard.experiment_timeline import timestamp,seconds


def readable_name(identifier):
    name=identifier.replace('_',' ')
    if name.startswith('dense '):name=name[6:]
    return re.sub(r'\brgb\b','RGB',re.sub(r' v\d+$','',name).capitalize(),flags=re.IGNORECASE)


def model_label(record):
    cfg=record['config']
    return f"{readable_name(record['candidate_id'])} · {cfg.get('plan_arm','legacy')} · seed {cfg.get('seed','?')}"


def model_card(record, incumbent_id=None, *, key='model'):
    metrics=record['metrics'];cfg=record['config'];run=record['run']
    st.subheader(readable_name(record['candidate_id']))
    reference='Current research reference · ' if record['run_id']==incumbent_id else ''
    evidence=record.get('scientific_comparison',{}).get('scientific_eligibility',{}).get('evidence_level','Exploratory')
    st.caption(f"{reference}{str(evidence).replace('_',' ').capitalize()} · seed {cfg.get('seed','?')} · {cfg.get('plan_arm','legacy')}")
    execution=record.get('execution')
    if execution:
        number=str(execution['order']) if execution['order'] is not None else 'not recorded'
        st.caption('Experiment #'+number+' · Campaign started '+timestamp(execution['campaign_started']))
        st.caption('Started '+timestamp(execution['started_at'])+' · Finished '+timestamp(execution['finished_at']))
        st.caption('Job duration '+seconds(execution['job_seconds'])+' · Development evaluation '+('reused' if execution['evaluation_reused'] else seconds(execution['evaluation_seconds'])))
    else:
        st.caption('Execution order not linked to a campaign · Started '+timestamp(run.get('started_at'))+' · Finished '+timestamp(run.get('finished_at')))
    score=lambda name:f'{metrics[name]:.4f}' if metrics.get(name) is not None else 'Unavailable'
    metric_columns([('Logical AU-sPRO',score(SPRO)),('Logical pixel AP',score(AP)),
        ('Normal false positives',f"{metrics.get('dev_real/normal_false_positives','?')} / {metrics.get('dev_real/normal_images','?')}")])
    latency=metrics.get('dev_real/inference_e2e_ms_p95');parameters=metrics.get('dev_real/parameters')
    metric_columns([('Logical AUROC',score('dev_real/logical_image_auroc')),
        ('Structural localization',score('dev_real/structural_au_spro_0_05_compatible')),
        ('Inference p95 · '+record.get('inference_device','unknown').upper(),f'{latency:.1f} ms' if latency is not None else 'Unavailable')])
    duration=training_seconds(run);memory=metrics.get('dev_real/peak_cpu_rss_mib')
    st.caption(('Parameters '+f'{int(parameters):,}' if parameters is not None else 'Parameters unavailable')+
        (' · Peak process RAM '+f'{memory:.1f} MiB' if memory is not None else '')+
        (' · Training loop '+format_duration(duration) if duration is not None else ' · Historical training timing'))
    st.caption('Custom development split · Compatible AU-sPRO@5% · Not official full-test results')
    availability,message=artifact_status(record)
    if availability=='Available':
        with st.expander('Localization preview'):
            index=st.number_input('Image',0,164,0,1,key=key+'_image')
            blend=st.checkbox('Blend prediction over the original',key=key+'_blend')
            if st.button('Show image, heatmap and mask',key=key+'_preview'):
                try:
                    sample=real_heatmap_sample(record,int(index))
                    for col,field,label in zip(st.columns(3),('original','overlay' if blend else 'heatmap','ground_truth'),('Original','Prediction','Ground truth')):
                        col.image(sample[field],caption=label,width='stretch')
                    st.caption('Original coordinates. Mask union is display-only; scoring retains individual regions.')
                except FileNotFoundError:st.error('Artifact missing unexpectedly; no intentional archival marker was found.')
                except PermissionError:st.warning('Artifact temporarily unavailable: access denied.')
                except (ValueError,KeyError,TypeError) as exc:st.error('Artifact integrity verification failed: '+str(exc))
                except OSError as exc:st.warning('Artifact temporarily unavailable: '+str(exc))
    else:st.caption(availability+' · '+message)
    with st.expander('Model details'):
        research,measurements=st.tabs(['Research','Measurements'])
        with research:candidate_research_panel(record['run_id'])
        with measurements:
            st.caption(LABEL+' Original-scorer equivalence remains unverified.')
            st.json({k:record.get(k) for k in ('run_id','split_id','evaluator_version','checkpoint_sha256','source_sha256','measurement_protocol','inference_device','threshold_source')})
            st.json(metrics)


def comparison_details(record):
    plan=record['plan'];result=record['result'] or {};review=record.get('review')
    st.subheader(readable_name(plan['hypothesis_id']))
    st.caption('Study #'+str(record.get('order','?'))+' · '+record['status']+f" · {record['completed']} / {record['expected']} model runs complete")
    st.caption('Plan recorded '+timestamp(record['registered_at']))
    findings,design,technical=st.tabs(['Finding','Arms and seeds','Evidence'])
    with findings:
        st.write(record.get('proposal',{}).get('claim') or plan.get('scientific_question') or plan.get('hypothesis') or 'See the frozen plan for the scientific question.')
        if review:
            st.write('**'+str(review.get('hypothesis_outcome',review.get('hypothesis_status','Reviewed'))).capitalize()+'**')
            conclusion=review.get('summary') or review.get('scientific_conclusion') or review.get('objective_measured_results')
            if conclusion:st.write(conclusion)
            if review.get('recommended_next_action'):st.caption('Next decision: '+review['recommended_next_action'].replace('_',' '))
        elif result.get('status','').startswith('COMPLETE'):st.info('Awaiting scientific review. Completed measurements are preserved.')
        else:st.info('No completed scientific review yet.')
        failures=[r for r in record['runs'] if r.get('error')]
        for run in failures:
            with st.expander('Failure · '+run['run_id'][:8]):st.code(run['error'],language='text')
        if result.get('problems'):
            with st.expander('Missing evidence'):st.write(result['problems'])
    with design:
        for arm in plan['arms']:st.write('**'+arm['arm_id']+'** — '+arm['difference'])
        table(comparison_matrix(record))
        differences=[d for d in result.get('paired_seed_differences',[]) if d.get('measurement') in {'real:'+SPRO,'real:'+AP}]
        if differences:table(differences)
        st.caption('Matching seed labels alone does not establish matched training streams or causal evidence.')
    with technical:
        controls=result.get('comparison_controls',{})
        st.write('Trusted paired controls: '+controls.get('paired_control_verification',{}).get('status','NOT_MEASURED'))
        with st.expander('Complete critique'):st.json(review or {'status':'Review pending or unavailable'})
        with st.expander('Frozen plan'):st.json(plan)
        with st.expander('Measurements and control receipts'):st.json(result)
