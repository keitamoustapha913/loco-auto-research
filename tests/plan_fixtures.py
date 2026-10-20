"""Deterministic infrastructure-only plans, never scientific model proposals."""
import copy
import json


def agent_plan(plan):
    plan=copy.deepcopy(plan)
    def pairs(value):return [{'name':key,'value_json':json.dumps(val)} for key,val in value.items()]
    plan['training']=pairs(plan['training'])
    for arm in plan['arms']:
        for key in ('training_overrides','inference'):arm[key]=pairs(arm[key])
        arm['baseline_by_seed']=[{'seed':int(k),'baseline_id':v} for k,v in arm['baseline_by_seed'].items()]
    for d in plan['diagnostics']:d['config']=pairs(d['config'])
    return plan


def experiment_plan(hypothesis_id='fixture', seeds=(3,17,29), arms=('control','intervention')):
    return {'schema_version':'1.0','plan_id':hypothesis_id+'_comparison','hypothesis_id':hypothesis_id,
        'kind':'controlled_comparison' if len(arms)>1 else 'exploratory','seeds':list(seeds),'baseline_arm_id':arms[0],
        'training':{'steps':2,'batch_size':2,'size':16,'epochs':4,
                    'expected_optimizer_updates':2,'expected_source_exposures':4},
        'controlled_variables':['steps','batch_size','size','expected_optimizer_updates','expected_source_exposures'],
        'baselines':[], 'paired_comparisons':[], 'arms':[{'arm_id':name,'difference':'Fixture output mode '+name,
            'execution':'train','training_overrides':{},'inference':{'mode':name},
            'source_arm_id':'','baseline_by_seed':{}} for name in arms],
        'diagnostics':[{'diagnostic_id':'fixture_check','config':{'protocol':'deterministic_fixture'},
                         'required_metrics':['mean_score']}],
        'required_measurements':[{'source':'run','key':'normal/image_false_positives'},
                                 {'source':'diagnostic','key':'fixture_check/mean_score'}],
        'evaluation':{'synthetic_diagnostics':False,'real_development_feedback':False,
                      'metric_evaluator_version':'development-v2.0'},
        'resource_budget':{'max_jobs':len(seeds)*len(arms),'max_job_seconds':5,
                           'max_gpu_seconds':len(seeds)*len(arms)*5,'max_wall_seconds':100},
        'completion_criteria':['Every declared arm, seed, setting and measured endpoint must be present.'],
        'failure_criteria':{'on_job_failure':'continue','max_failed_jobs':2}}
