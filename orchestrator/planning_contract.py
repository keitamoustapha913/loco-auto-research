"""Agent contract and trusted static admission share controller-owned rules.

No candidate import/execution, inferred historical defaults or scientific rewrites.
"""
from __future__ import annotations

import ast
import copy
import json
import subprocess
from pathlib import Path


MATCHED_SAMPLING_CONTROLS = ('sampling_protocol', 'sampling_seed')

CONFIRMATION_REQUIREMENTS = {'type':['object','null'],'additionalProperties':False,
    'properties':{'normal_images':{'type':'integer','minimum':1},
        'max_additional_false_positives':{'type':'integer','minimum':0},
        'paired_controls':{'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'string'}},
        'uncertainty':{'type':'string','minLength':20}},
    'required':['normal_images','max_additional_false_positives','paired_controls','uncertainty']}


class PlanningValidationError(ValueError):
    def __init__(self, issues):
        self.issues=issues
        super().__init__('Scientific plan rejected: '+json.dumps(issues,sort_keys=True))


def local_issues(plan):
    """Semantic key/value checks independent of any candidate implementation."""
    from orchestrator.experiment_plan import RESERVED,training_settings
    issues=[]
    for arm in plan['arms']:
        settings=training_settings(plan,arm)
        invalid=sorted(k for k in settings if not k or any(c.isspace() for c in k))
        if invalid:issues.append({'code':'INVALID_TRAINING_FIELD_NAMES','arm':arm['arm_id'],'fields':invalid,
            'message':'Setting names must be nonempty keys without whitespace; scientific explanations belong outside settings.'})
        bad=sorted(k for k in settings if k in RESERVED or k.startswith('plan_'))
        if bad:issues.append({'code':'RESERVED_TRAINING_FIELDS','arm':arm['arm_id'],'fields':bad,
            'message':'Remove controller-owned keys from scientific settings; the controller supplies them.'})
        missing=[k for k in plan['controlled_variables'] if k not in settings]
        if missing:issues.append({'code':'CONTROLLED_VARIABLE_KEYS','arm':arm['arm_id'],'keys':missing,
            'available_keys':sorted(k for k in settings if k not in RESERVED and not k.startswith('plan_')),
            'message':'Use actual declared setting keys, with matching values in every arm, not prose.'})
    for key in plan['controlled_variables']:
        values=[training_settings(plan,a).get(key) for a in plan['arms']]
        if all(key in training_settings(plan,a) for a in plan['arms']) and any(v!=values[0] for v in values[1:]):
            issues.append({'code':'CONTROLLED_VARIABLE_VALUES','key':key,'values':values,'message':'A declared controlled value differs between arms.'})
    return issues


def baseline_capability(store, reference, repo_root):
    from orchestrator.experiment_plan import verify_baseline,REQUIRED_TRAINING_POSITIVE
    source,row=verify_baseline(store,reference,repo_root)
    configuration=json.loads(row['config_json'])
    tree=ast.parse(source)
    load=next((n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='load'),None)
    # Match train_adapter.load_predictor: **kwargs alone is not acknowledgement.
    named=bool(load and 'inference_config' in [a.arg for a in load.args.args+load.args.kwonlyargs])
    provenance_path=Path(row['artifact_dir'])/('host_verified_training.json' if row.get('status') in {'FAILED','OOM'} else 'run_manifest.json')
    provenance=json.loads(provenance_path.read_text())
    audit=provenance.get('training_audit',{})
    explicit_schedule=all(k in configuration for k in REQUIRED_TRAINING_POSITIVE+('expected_optimizer_updates',))
    timed_schedule=(configuration.get('training_completion_policy')=='elapsed-training-loop-v1' and
                    all(k in configuration for k in ('batch_size','size','training_budget_seconds')))
    replication=(explicit_schedule or timed_schedule) and all(k in audit for k in ('seed','optimizer_updates','source_exposures'))
    return {'reference':reference,'configuration':configuration,'recorded_inference':configuration.get('inference',{}),
            'named_inference_config_parameter':named,'requires_exact_inference':True,'frozen_training_overrides':{},
            'required_image_size':configuration.get('size'),'actual_training_seed':configuration.get('seed'),
            'replication_audit_evidence_available':replication,
            'replication_note':'Exact source/configuration may be proposed for additional seeds; implementation validation still applies.' if replication else 'Missing explicit training settings/audit; do not invent defaults or claim matched seed replication.'}


def evidence_classification(plan, rows=None):
    """Trusted eligibility ceiling; completion, measured matching and policy follow."""
    from orchestrator.experiment_plan import MATCHED_TRAINING_CONTROLS,digest,learning_controls
    from orchestrator.model_registry import SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY,policy
    reasons=[]
    if plan['kind']!='controlled_comparison':reasons.append('Agent requested exploratory evidence; no automatic upgrade.')
    if len(plan['arms'])<2:reasons.append('At least two arms are required.')
    if len(plan['seeds'])<policy()['minimum_seeds']:reasons.append('Insufficient independent training seeds.')
    if any(a.get('fixed_checkpoint_seed') is not None for a in plan['arms']):
        reasons.append('Single frozen training seed is exploratory, not independent baseline seed replication.')
    if not set(learning_controls(plan))<=set(plan['controlled_variables']):
        reasons.append('Training settings/exposures/optimizer updates are not all declared matched controls.')
    requested={r['key'] for r in plan['required_measurements'] if r['source']=='real'}
    if not {SPRO,AP,FPR,LATENCY,PARAMETERS,MEMORY}<=requested:reasons.append('Required real-localization and deployment endpoints are not preregistered.')
    if not plan['completion_criteria']:reasons.append('Predeclared acceptance/completion criteria are missing.')
    if plan.get('schema_version')=='4.0':
        reasons.extend(comparison_controls(plan)['confounds'])
        if plan.get('confirmation') and not plan.get('confirmation_requirements'):
            reasons.append('Future confirmation requires sample-aware normal counts, observed paired controls and scoped uncertainty.')
    if rows is not None:
        if plan.get("schema_version")=="4.0" and plan.get("confirmation"):
            reasons.extend(comparison_controls(plan,rows)["confounds"])
        for arm in plan['arms']:
            selected=[r for r in rows if json.loads(r['config_json']).get('plan_sha256')==digest(plan) and json.loads(r['config_json']).get('plan_arm')==arm['arm_id']]
            configs=[json.loads(r['config_json']) for r in selected]
            if len({c.get('seed') for c in configs})!=len(plan['seeds']):reasons.append('Recorded independent training seeds do not cover an arm.')
            for c in configs:
                original=c.get('baseline_effective_configuration',c)
                if plan.get('schema_version')=='4.0' and original.get('training_completion_policy')!='elapsed-training-loop-v1':
                    reasons.append('Historical arithmetic-target baseline is not a matched elapsed-loop training control.')
                if any(k not in original or original[k]!=dict(plan['training'],**arm['training_overrides']).get(k) for k in learning_controls(plan)):
                    reasons.append('Recorded baseline learning configuration does not match controlled training.')
    return {'promotion_eligible':not reasons,'evidence_level':'CONTROLLED' if not reasons else 'EXPLORATORY',
            'reasons':list(dict.fromkeys(reasons)),
            'message':'Eligibility ceiling only; complete measured evidence, independent checkpoints, integrity and policy still required.'}


def comparison_controls(plan, rows=()):
    """Report confounds separately from matrix completeness or score ranking.

    Matching a pair label does not change a checkpoint's authentic training seed.
    Actual sample/mask streams are unknown unless measured; equal wall time is
    explicitly not equal exposure. This never rewrites a submitted plan.
    """
    from orchestrator.experiment_plan import training_seed, training_settings, digest
    confounds=[];counts=[]
    arms=plan['arms']
    if len(arms)<2:confounds.append('No comparison arm.')
    for seed in plan['seeds']:
        actual=[training_seed(a,seed) for a in arms]
        if len(set(actual))>1:confounds.append(f'Pair {seed} has unmatched actual training seeds {actual}.')
    settings=[training_settings(plan,a) for a in arms]
    all_keys=set().union(*(set(c) for c in settings))
    declared=set(plan['controlled_variables'])
    unrecorded=sorted(k for k in all_keys if all(k in c and c[k]==settings[0].get(k) for c in settings) and k not in declared)
    if unrecorded:confounds.append('Equal settings not declared controlled: '+', '.join(unrecorded))
    if any(a.get('fixed_checkpoint_seed') is not None for a in arms):
        confounds.append('A single frozen training seed is not a repeated matched-method baseline.')
    for row in rows:
        cfg=json.loads(row['config_json'])
        if cfg.get('plan_sha256')!=digest(plan):continue
        metrics=json.loads(row['metrics_json'])
        counts.append({'arm':cfg.get('plan_arm'),'seed':cfg.get('seed'),
            'optimizer_updates':metrics.get('protocol/optimizer_updates'),
            'source_exposures':metrics.get('protocol/source_exposures')})
    # A sampling protocol and seed are executable candidate settings, not prose.
    stream_declared=set(MATCHED_SAMPLING_CONTROLS)<=declared
    if not stream_declared:confounds.append('Sampling protocol/independent sampling seed are not declared matched controls.')
    verification={'status':'NOT_MEASURED','pairs':[]}
    if rows and plan.get('schema_version')=='4.0':
        from orchestrator.confirmation import verify_pairs
        verification=verify_pairs(plan,rows)
        if verification['status']!='VERIFIED':confounds.append('Actual paired training controls are not verified: '+verification['status'])
    return {'intended_differences':{a['arm_id']:a['difference'] for a in arms},
        'changed_training_settings':sorted(k for k in all_keys if any(c.get(k)!=settings[0].get(k) for c in settings[1:])),
        'actual_seeds_by_arm':{a['arm_id']:[training_seed(a,s) for s in plan['seeds']] for a in arms},
        'confounds':list(dict.fromkeys(confounds)), 'observed_training_counts':counts,
        'sampling_stream_status':verification['status'] if rows else 'DECLARED_NOT_INDEPENDENTLY_VERIFIED' if stream_declared else 'UNVERIFIED',
        'paired_control_verification':verification,
        'causal_claim_eligible':plan['kind']=='controlled_comparison' and not confounds,
        'reproducibility_status':'UNCONFIRMED',
        'interpretation':'Equal elapsed training time does not imply equal updates/exposures or identical sampled prefixes. '
                         'Source-bootstrap uncertainty is conditional on checkpoints, not training-seed uncertainty.'}


def inspect_plan(plan, store=None, repo_root=None):
    """Collect all previously observed failure classes, without modifying input."""
    from orchestrator.experiment_plan import REPO,validate_plan,plan_schema
    from orchestrator.controller_errors import CandidateValidationError
    from benchmark.train_adapter import CandidateContractError
    from jsonschema import validate, ValidationError
    try:
        validate(plan,plan_schema(plan))
    except ValidationError as exc:
        return {'valid':False,'errors':[{'code':'PLAN_SCHEMA','message':exc.message,'path':list(exc.path)}],
                'notices':[],'scientific_eligibility':{'promotion_eligible':False},'baseline_capabilities':{}}
    original_measurements=copy.deepcopy(plan['required_measurements'])
    issues=local_issues(plan)
    try:plan=validate_plan(plan)
    except PlanningValidationError:pass
    except (CandidateValidationError,CandidateContractError) as exc:issues.append({'code':'PLAN_CONTRACT','message':str(exc)})
    capabilities={}
    if store is not None:
        if plan.get('confirmation') and plan.get('confirmation_requirements'):
            pinned=json.loads((REPO/'configs/development_feedback.json').read_text())['split_id']
            with store.connect() as con:
                observations=con.execute("SELECT payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' ORDER BY id DESC LIMIT 500").fetchall()
            normal_counts={v.get('metrics',{}).get('dev_real/normal_images') for row in observations
                for v in [json.loads(row[0])] if v.get('split_id')==pinned}
            if plan['confirmation_requirements']['normal_images'] not in normal_counts:
                issues.append({'code':'NORMAL_SAMPLE_COUNT','message':'Confirmation normal_images must match trusted feedback counts on the pinned development split.',
                    'observed_counts':sorted(n for n in normal_counts if n is not None)})
        for ref in plan['baselines']:
            try:capabilities[ref['baseline_id']]=baseline_capability(store,ref,repo_root or REPO)
            except (ValueError,OSError,subprocess.SubprocessError) as exc:issues.append({'code':'BASELINE_INTEGRITY','baseline_id':ref['baseline_id'],'message':str(exc)})
        for arm in plan['arms']:
            if not arm['baseline_by_seed']:continue
            for bid in set(arm['baseline_by_seed'].values()):
                cap=capabilities.get(bid)
                if not cap:continue
                if arm['inference']!=cap['recorded_inference']:
                    issues.append({'code':'BASELINE_INFERENCE_NOT_EXACT','arm':arm['arm_id'],'baseline_id':bid,
                        'expected':cap['recorded_inference'],'requested':arm['inference'],
                        'message':'Historical frozen arms must use the exact recorded inference configuration.'})
                if arm['inference'] and not cap['named_inference_config_parameter']:
                    issues.append({'code':'UNSUPPORTED_BASELINE_INFERENCE_CONFIG','arm':arm['arm_id'],'baseline_id':bid,
                        'message':'Recorded load() has no named inference_config parameter; no new settings can be passed.'})
                for key in set(plan['controlled_variables'])|{'size'}:
                    expected=cap['configuration'].get(key)
                    requested=dict(plan['training'],**arm['training_overrides']).get(key)
                    if key not in cap['configuration'] or expected!=requested:
                        issues.append({'code':'UNVERIFIABLE_BASELINE_CONTROL','arm':arm['arm_id'],'baseline_id':bid,'key':key,
                            'expected':expected,'requested':requested,'message':'Cannot assert matching unrecorded or different historical values.'})
    classification=evidence_classification(plan)
    if plan.get('schema_version')=='4.0' and any(
        capabilities[bid]['configuration'].get('training_completion_policy')!='elapsed-training-loop-v1'
        for arm in plan['arms'] for bid in arm['baseline_by_seed'].values() if bid in capabilities):
        classification.update(promotion_eligible=False,evidence_level='EXPLORATORY')
        classification['reasons'].append('Frozen historical baseline does not establish matched elapsed-loop training.')
    if issues:
        classification.update(promotion_eligible=False,evidence_level='EXPLORATORY')
        classification['reasons'].append('Semantic admission failed; no execution or promotion permitted.')
    if plan.get('confirmation') and not classification['promotion_eligible']:
        issues.append({'code':'CONFIRMATION_PREREQUISITES','message':'Confirmation must preregister all policy endpoints, independent seeds and matched controls.', 'reasons':classification['reasons']})
    notices=[] if classification['promotion_eligible'] else [{'code':'NONPROMOTABLE_EVIDENCE_LEVEL',**classification}]
    normalizations=[{'source':old['source'],'submitted':old['key'],'canonical':new['key']}
                    for old,new in zip(original_measurements,plan['required_measurements']) if old!=new]
    return {'valid':not issues,'errors':issues,'notices':notices,'scientific_eligibility':classification,
            'baseline_capabilities':capabilities,'diagnostic_name_normalizations':normalizations,
            'canonical_plan':plan if not issues else None,'comparison_controls':comparison_controls(plan)}


def admit_plan(plan, store, repo_root=None):
    report=inspect_plan(plan,store,repo_root)
    if report['errors']:raise PlanningValidationError(report['errors'])
    return report


def planning_instructions(capabilities=(), *, time_only=True):
    from orchestrator.experiment_plan import RESERVED,REQUIRED_TRAINING_POSITIVE,MATCHED_TRAINING_CONTROLS
    return json.dumps({'contract':'controller-planning-v4' if time_only else 'controller-planning-v2','controller_owned_fields':sorted(RESERVED),
        'controller_owned_prefix':'plan_',
        'timing':('New plans use schema_version 4.0, no resource_budget or independent job/GPU/plan deadlines. Jobs are exactly arms times seeds. Only the remaining campaign research deadline and required elapsed training-loop target apply. Preserve complete comparisons; report INCOMPLETE if interrupted.' if time_only else 'Legacy schema_version 1.0 plans retain their explicitly authorized resource_budget restrictions.'),'required_positive_training_keys':['batch_size','size'] if time_only else list(REQUIRED_TRAINING_POSITIVE),
        'required_nonnegative_training_key':None if time_only else 'expected_optimizer_updates',
        'required_training':('Each new model must run its genuine optimization loop for the complete elapsed operator target. Initialization and final saving are excluded; in-loop batch preparation, transfers, learning, integrity checks and bookkeeping count. Declare TRAINING_PROTOCOL=elapsed-training-loop-v1 and yield TrainingLoopStart(objective) after initialization/cache fill and before the first batch iteration. No fixed steps/epochs or expected final update/exposure counts. Declare a scientifically appropriate continuing schedule; progress-aware training_steps receives elapsed_seconds, target_seconds, target_reached, optimizer_updates and source_exposures. Finalize only on target_reached; never restart/reinitialize or pad with sleep/dummy work.' if time_only else 'Preserve the exact authorized legacy training steps, epochs, exposures and original timing restrictions.'),
        'confirmation':('Set confirmation=true only for a proposed confirmed promotion study. Preregister the exact real endpoint keys: dev_real/logical_au_spro_0_05_compatible, dev_real/logical_pixel_ap_pooled, dev_real/normal_image_fpr, dev_real/inference_e2e_ms_p95, dev_real/parameters, dev_real/peak_cpu_rss_mib, with at least three independent seeds and matched controls. Ordinary exploratory plans may set false.' if time_only else 'Legacy plans retain their original evidence and promotion classification; do not add new fields.'),
        'confirmation_requirements':('For a future confirmation set normal_images to the actual good-image count, max_additional_false_positives to a predeclared integer count (one of 47 is 2.13%, not 2%), paired_controls to relevant actual observations: initial_state, optimizer, input:0 etc (indices of prepared TrainingStep.args). Select inputs encoding source/augmentation/mask streams that should match; exclude the intervened target/input and explain why. Matching seeds alone does not verify streams. State uncertainty separately for independent paired training seeds, grouped/paired source images, normal-binomial uncertainty and repeated development-set selection. Use null for exploratory work. This does not change historical criteria or assert population/generalization guarantees.'),
        'controlled_variables':'Names of real training keys, not sentences. Values must be declared, equal across arms and recorded in every frozen baseline.',
        'matched_training_controls_for_confirmation':['batch_size','size',*MATCHED_SAMPLING_CONTROLS] if time_only else list(MATCHED_TRAINING_CONTROLS),
        'diagnostic_measurements':'diagnostics[].required_metrics is authoritative. For source=diagnostic use diagnostic_id/metric; diagnostic/diagnostic_id/metric is a storage alias accepted only for the same declared metric. Never duplicate requests via aliases or invent metric names. Paired metric names must be declared.',
        'frozen_inference':'Copy recorded_inference exactly; {} encodes [] in wire settings. Never invent loader options.',
        'normal_diagnostic_inputs':'diagnose receives separate calibration_good_images, calibration_source_ids and assessment_source_ids from validation/good, plus frozen_normal_threshold. Declare min_calibration_images/min_assessment_images in config; no assessment calibration, real anomalies or masks.',
        'fixed_checkpoint':'Declare the actual fixed seed; pairing indices are not independent baseline training. Such comparisons are exploratory and cannot confirm a champion.',
        'evidence_level':'Agent kind is intent only. Trusted admission/completion derives eligibility; text cannot authorize promotion.',
        'scientific_changes':'Invalid scientific settings must be revised by the agent, never automatically stripped, defaulted or reinterpreted.',
        'frozen_baselines':list(capabilities)},sort_keys=True)


def constrain_output_schema(schema, capabilities=(), *, time_only=True, target_training=True, loop_training=True,
                            research_request=False):
    """Generate wire constraints from the same rules used by admission."""
    result=copy.deepcopy(schema)
    if research_request and 'proposals' in result.get('properties',{}):
        # A request is research intent, never a shell command or an executable
        # experiment. Trusted routing requires an empty batch and a real query.
        result['properties']['literature_request']={'type':'string',
            'description':'Empty when proposing; otherwise a specific external knowledge gap, with proposals=[].'}
        if 'literature_request' not in result['required']:result['required'].append('literature_request')
        result['properties']['proposals']['minItems']=0
    def visit(node):
        if not isinstance(node,dict):return
        props=node.get('properties',{})
        if time_only and 'estimated_gpu_minutes' in props:
            props['estimated_gpu_minutes'].pop('maximum',None)
        if {'training','controlled_variables','arms','seeds'}<=props.keys():
            if time_only:
                props['schema_version'] = {'type':'string','const':('4.0' if loop_training else '3.0') if target_training else '2.0'}
                if target_training:
                    props['confirmation']={'type':'boolean'}
                    if 'confirmation' not in node['required']:node['required'].append('confirmation')
                    if loop_training:
                        props['confirmation_requirements']=copy.deepcopy(CONFIRMATION_REQUIREMENTS)
                        if 'confirmation_requirements' not in node['required']:node['required'].append('confirmation_requirements')
                props.pop('resource_budget',None)
                node['required'] = [key for key in node.get('required',[]) if key!='resource_budget']
            pair=props['training'].get('items',{})
            if pair.get('properties',{}).get('name') is not None:
                # Provider names stay plain strings. Reserved names/whitespace
                # and control semantics are enforced by local_issues, not regex.
                pair['properties']['name'].pop('pattern',None)
                props['controlled_variables']['items'].pop('pattern',None)
                props['required_measurements']['description']='Diagnostic requests reference diagnostics[].required_metrics using source=diagnostic and key=diagnostic_id/metric. Trusted code resolves only unambiguous declared storage aliases.'
                props['controlled_variables']['description']='Actual common training setting key names, with equal recorded values; never prose.'
                arm=props['arms']['items']
                arm['properties']['training_overrides']['items']['properties']['name'].pop('pattern',None)
                if capabilities:
                    train=copy.deepcopy(arm);frozen=copy.deepcopy(arm)
                    train['properties']['execution']={'const':'train','type':'string'}
                    frozen['properties']['execution']={'const':'frozen','type':'string'}
                    choices=[]
                    for cap in capabilities:
                        wire=[{'name':k,'value_json':json.dumps(v,sort_keys=True)} for k,v in cap['recorded_inference'].items()]
                        if wire not in choices:choices.append(wire)
                    frozen['properties']['inference']['enum']=choices
                    # source_arm_id interventions use newly implemented predictor
                    # contracts, so historical constraints apply when that ID is empty.
                    frozen['properties']['source_arm_id']={'type':'string','const':''}
                    dependent=copy.deepcopy(arm);dependent['properties']['execution']={'const':'frozen','type':'string'}
                    dependent['properties']['source_arm_id']['minLength']=1
                    props['arms']['items']={'anyOf':[train,frozen,dependent]}
            return
        for name in ('properties','$defs','definitions'):
            for child in node.get(name,{}).values():visit(child)
    visit(result)
    return result
