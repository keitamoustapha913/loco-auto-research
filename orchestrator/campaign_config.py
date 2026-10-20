"""Normalize operator time budgets without rewriting historical configurations."""
from __future__ import annotations
import copy
import math

TIME_VERSION = 'campaign-time-v1'
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
            max_consecutive_failures=3, max_parallel_gpu_jobs=1)
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


def plan_seconds(plan, config):
    """Conservative admission allowance, not a change to a frozen plan."""
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
