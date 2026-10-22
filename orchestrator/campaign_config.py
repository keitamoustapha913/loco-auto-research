"""Normalize operator time budgets without rewriting historical configurations."""
from __future__ import annotations
import copy
import math

TIME_VERSION = 'campaign-time-v1'
MAX_CONSECUTIVE_RETRIES = 10
LEGACY_LIMITS = ('max_gpu_hours', 'max_wall_hours', 'max_experiments', 'max_codex_calls',
                 'max_job_seconds', 'max_consecutive_failures', 'max_parallel_gpu_jobs')


def time_based(config):
    return config.get('configuration_version') == TIME_VERSION


def normalize(config):
    config = copy.deepcopy(config)
    limits = config['limits']
    simplified = any(k in limits for k in ('research_hours', 'training_minutes_per_experiment'))
    if simplified:
        if not time_based(config) and set(limits) - {'research_hours', 'training_minutes_per_experiment'}:
            raise ValueError('do not mix time settings with legacy campaign limits')
        for key in ('research_hours', 'training_minutes_per_experiment'):
            value = limits.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + ' must be a finite positive number')
        hours, minutes = limits['research_hours'], limits['training_minutes_per_experiment']
        if not math.isfinite(hours*3600) or not math.isfinite(minutes*60):
            raise ValueError('time values must produce finite positive durations')
        config['configuration_version'] = TIME_VERSION
        # Adapter fields for existing scientific contracts. Counts are unlimited;
        # serialization, deadlines, disk safety and failure recovery still apply.
        config['limits'] = dict(research_hours=hours, training_minutes_per_experiment=minutes,
            max_wall_hours=hours, max_gpu_hours=hours, max_experiments=None, max_codex_calls=None,
            max_job_seconds=math.ceil(hours*3600),
            max_consecutive_failures=MAX_CONSECUTIVE_RETRIES, max_parallel_gpu_jobs=1)
    else:
        for key in LEGACY_LIMITS:
            value = limits.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('legacy campaign limits must be finite positive numbers')
    mode = config['campaign'].setdefault('start_from', 'previous_best')
    if mode not in {'previous_best', 'new'}:
        raise ValueError('start_from must be previous_best or new')
    return config


def remaining_wall(row, now):
    import json
    return max(0., row['created_at'] + json.loads(row['config_json'])['limits']['max_wall_hours']*3600 - now)


def operation_seconds(row, legacy_limit):
    """All simplified research operations share the original absolute deadline."""
    import json,time
    remaining=remaining_wall(row,time.time())
    return remaining if time_based(json.loads(row['config_json'])) else min(legacy_limit,remaining)


def operation_cancelled(campaigns, campaign_id):
    import time
    row=campaigns.get(campaign_id)
    return row['status'] not in {'RUNNING','PAUSED'} or remaining_wall(row,time.time())<=0


def plan_seconds(plan, config):
    """Conservative admission allowance, not a change to a frozen plan."""
    if plan.get('schema_version') in {'2.0','3.0','4.0'}:return 0  # No independent plan reservation.
    if time_based(config):return plan['resource_budget']['max_wall_seconds']
    jobs = len(plan['arms']) * len(plan['seeds'])
    feedback = config['research'].get('feedback_cpu_timeout_seconds', 600) if plan['evaluation']['real_development_feedback'] else 0
    critique = 180  # The scientific-plan controller always requires a final review.
    return jobs * (plan['resource_budget']['max_job_seconds'] + feedback) + critique


def starting_context(store, config, strict=False):
    """References only: never load weights, compare incompatible cohorts or promote."""
    mode = config['campaign'].get('start_from', 'previous_best')
    result = {'start_from': mode, 'dataset': config['campaign']['dataset'], 'models': [],
              'weight_reuse': 'Only an explicit validated scientific plan may request weight reuse.'}
    if mode == 'previous_best':
        from orchestrator.model_registry import states, retained, verify_archive, admissible, policy, retained_category
        models = retained(store)
        for cohort, state in states(store).items():
            model = models[state['retention_id']]
            if retained_category(store,model) != result['dataset'] or not admissible(model, policy()): continue
            verify_archive(model)
            result['models'].append({'cohort_id': cohort, 'confirmation': state['confirmation'],
                'run_id': model['run_id'], 'candidate_id': model['candidate_id'],
                'reference': model['reference'], 'archive': model['archive'], 'metrics': model['metrics']})
        if strict and not result['models']:
            raise ValueError('No eligible retained model for this dataset; use start_from: new.')
    return result


def throughput_ratio(profile):
    """Conservative steady end-to-end/learning ratio; never a new timer."""
    if not isinstance(profile,dict) or profile.get('version')!='training-throughput-v1':
        raise ValueError('missing trusted training-throughput-v1 preflight profile')
    n=profile.get('sampled_updates');learning=profile.get('sampled_learning_seconds');wall=profile.get('sampled_step_wall_seconds')
    if isinstance(n,bool) or not isinstance(n,int) or n<16:
        raise ValueError('at least 16 post-warmup updates required for throughput admission')
    for x in (learning,wall):
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0:
            raise ValueError('invalid observed throughput duration')
    if wall<learning:raise ValueError('observed step wall time cannot exclude learning')
    return max(1.,wall/learning)*1.25  # Admission uncertainty margin, not an execution limit.


def remaining_plan_work(store, config):
    """All remaining declared cells, including the currently disposable-smoked cell."""
    import json
    with store.connect() as con:
        events=con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_REGISTERED' ORDER BY id DESC",(config['campaign_id'],)).fetchall()
        plan=next((json.loads(e[0])['plan'] for e in events if json.loads(e[0])['plan']['plan_id']==config.get('plan_id')),None)
        if plan is None:
            if config.get('plan_sha256'):raise ValueError('throughput admission requires the registered complete experiment matrix')
            return dict(jobs=1,newly_trained=1)
        done={}
        for r in con.execute('SELECT r.config_json,r.status FROM runs r JOIN campaign_runs cr ON cr.run_id=r.run_id WHERE cr.campaign_id=?',(config['campaign_id'],)):
            effective=json.loads(r['config_json'])
            if effective.get('plan_sha256')==config.get('plan_sha256'):
                done[(effective['plan_arm'],effective.get('plan_pair_seed',effective['seed']))]=r['status']
    remaining=[a for a in plan['arms'] for seed in plan['seeds'] if done.get((a['arm_id'],seed)) not in {'COMPLETED','REVIEWED','PROMOTED','REJECTED'}]
    return dict(jobs=len(remaining),newly_trained=sum(a['execution']=='train' for a in remaining))


def training_execution_estimate(store, config, *, jobs=1, newly_trained=1, real_feedback=True,
                                critique=False, source_sha256=None, throughput=None,training_settings=None,loop_timed=False):
    """End-to-end target projection, including cancelled partial-training evidence.

    Unknown implementations require measured queued preflight before full training.
    CPU observations never predict CUDA throughput. No extra execution deadline.
    """
    import json
    evaluation=[];ratios=[];exact=[];loop_extras=[]
    device=config['campaign'].get('gpu_device','cuda')
    with store.connect() as con:
        for row in con.execute("SELECT * FROM runs WHERE category=? ORDER BY created_at DESC LIMIT 100",(config['campaign']['dataset'],)):
            effective=json.loads(row['config_json']);metrics=json.loads(row['metrics_json'])
            if not loop_timed and effective.get('training_completion_policy')=='elapsed-training-loop-v1':continue
            learning=metrics.get('training/duration_seconds')
            if effective.get('execution_kind')=='frozen' or effective.get('device','cpu')!=device:continue
            if (isinstance(learning,bool) or not isinstance(learning,(float,int)) or not math.isfinite(learning)
                or learning<1 or not row['finished_at'] or not row['started_at']):continue
            wall=row['finished_at']-row['started_at']
            if wall<learning:continue
            if loop_timed and effective.get('training_completion_policy')=='elapsed-training-loop-v1':
                loop_extras.append(max(0,wall-learning))
            value=wall/learning*1.25
            ratios.append(value)
            if source_sha256 and effective.get('model_sha256')==source_sha256:exact.append(value)
        for row in con.execute("SELECT payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' ORDER BY id DESC LIMIT 30"):
            value=json.loads(row[0]).get('metrics',{}).get('dev_real/evaluation_wall_seconds')
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0:evaluation.append(value)
    if throughput is not None:
        if throughput.get('source_sha256')!=source_sha256 or throughput.get('device')!=device:
            raise ValueError('throughput source/device differs from effective candidate')
        if training_settings and any(throughput.get(k)!=training_settings[k] for k in ('batch_size','size')):
            raise ValueError('throughput probe batch size/resolution differs from planned training')
        ratio=throughput_ratio(throughput);basis='queued source/device-matched measured preflight; 25% uncertainty margin'
        observed=True
    elif exact:
        ratio=max(exact);basis='source/device-matched historical end-to-end observations, including cancelled runs';observed=True
    elif source_sha256:
        ratio=1.;basis='unmeasured new source: preliminary learning floor only; queued throughput admission required';observed=False
    else:
        ratio=max([1.]+ratios);basis='preliminary dataset/device historical end-to-end observations, including cancelled runs';observed=bool(ratios)
    # v4's target already includes repeated preparation, transfers, synchronization
    # and integrity. Never multiply it by a historical arithmetic/wall ratio.
    if loop_timed:
        ratio=1.;basis='elapsed training-loop target; computation/preparation ratio is diagnostic only'
    setup=max([60.]+loop_extras) if loop_timed else 60.
    feedback=max([180.]+evaluation) if real_feedback else 0.
    overshoot=0.
    if throughput:
        initial=throughput.get('initial_preparation_seconds',0)
        if isinstance(initial,(int,float)) and not isinstance(initial,bool) and math.isfinite(initial):setup=max(setup,initial)
        extra=throughput.get('non_loop_preflight_seconds',0)
        if loop_timed and isinstance(extra,(int,float)) and not isinstance(extra,bool) and math.isfinite(extra):setup=max(setup,extra)
        if loop_timed:
            step=throughput.get('sampled_max_step_wall_seconds',0)
            if isinstance(step,(int,float)) and not isinstance(step,bool) and math.isfinite(step):overshoot=max(0,step)
    target=config['limits']['training_minutes_per_experiment']*60
    return {'training_seconds_per_model':target,'execution_to_learning_ratio':ratio,
            'projected_training_execution_seconds_per_model':target*ratio+overshoot,
            'preparation_and_diagnostics_estimate_seconds':setup,'throughput_observed':observed,
            'feedback_estimate_seconds':feedback,'critique_estimate_seconds':90. if critique else 0.,
            'required_seconds':newly_trained*(target*ratio+overshoot)+jobs*(setup+feedback)+(90. if critique else 0.),
            'basis':basis+'; scheduling estimates only, original campaign deadline unchanged'}
