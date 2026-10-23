"""Operator CLI: deterministic state gates around agent-driven research.

This CLI intentionally separates propose / implement / queue / worker from final
benchmarking; no hidden auto-approvals or sealed evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
import sys
from orchestrator.budgets import BudgetExceeded, max_job_seconds, enforce
from orchestrator.codex_runner import invoke, create_candidate_worktree, InvalidAgentOutput
from orchestrator.gpu_queue import claim_next, gpu_lock
from orchestrator.experiment_runner import execute_run
from orchestrator.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]


def run_worker_once(store, manifest_path, runtime, personal_mode_authorized=False, cancel=None,
                    expected_run_id=None):
    with gpu_lock(runtime):
        if expected_run_id:
            expected = store.get_run(expected_run_id)
            if not expected or expected['status'] != 'QUEUED':
                raise ValueError('expected campaign job is not queued')
            expected_config = json.loads(expected['config_json'])
            duration = (0 if expected_config.get('plan_contract_version') in {'experiment-plan-v2.0','experiment-plan-v3.0','experiment-plan-v4.0'} else
                        min(max_job_seconds(expected_config), int(expected_config.get('timeout_seconds', max_job_seconds(expected_config)))))
        else:
            duration = max_job_seconds()
        job = claim_next(store, max_duration=duration, expected_run_id=expected_run_id)
        if not job:
            return {"status": "IDLE"}
        try:
            if personal_mode_authorized or cancel:
                outcome = execute_run(job, store, REPO, Path(manifest_path), Path(runtime),
                                      personal_mode_authorized=personal_mode_authorized, cancel=cancel)
            else:
                outcome = execute_run(job, store, REPO, Path(manifest_path), Path(runtime))
        except Exception as exc:
            current = store.get_run(job["run_id"])
            if current["status"] == "RUNNING":
                store.transition(job["run_id"], "FAILED", reason=str(exc))
            raise
        return {"run_id": job["run_id"], "status": outcome}


def _budgeted_invoke_once(store, role, repo_root, schema, prompt, campaign=None, **kwargs):
    requested = kwargs.pop('evidence_run_ids', ())
    capabilities=[]
    if kwargs.get('personal_mode') and role in {'researcher','implementer','critic'}:
        from orchestrator.model_registry import initialize, recover_comparisons, scientific_memory, canonical
        initialize(store,repo_root)
        recover_comparisons(store,repo_root)
        memory=scientific_memory(store)
        folder=store.path.parent/'research_evidence'/'cumulative'/canonical(memory)
        if folder.parent.is_symlink() or folder.is_symlink():raise PermissionError('unsafe cumulative context')
        folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        path=folder/'evidence.json'
        data=json.dumps(memory,sort_keys=True).encode()
        if path.exists():
            if path.is_symlink() or path.read_bytes()!=data:raise PermissionError('cumulative context changed')
        else:path.write_bytes(data);path.chmod(0o444)
        from orchestrator.experiment_plan import research_evidence
        evidence, files = research_evidence(store, repo_root, requested)
        files['cumulative/evidence.json']=path
        prompt += ('\nPersistent whole-history scientific memory is at /workspace/evidence/cumulative/evidence.json. '
                   'Active research selection weights logical AU-sPRO 60%, pixel AP 20%, normal performance 10%, efficiency 10%. '
                   'Use the active research incumbent as the primary reference; weight reuse requires an explicit plan. '
                   'Confirmed scientific champions are separate and require controlled verification. Review unfinished comparisons, failures, '
                   'prior ablations and confirmation needs before choosing research. You choose methods freely. '
                   'An exploratory record score is not a confirmed improvement. Current retained heads: '+
                   json.dumps([{k:c.get(k) for k in ('cohort_id','research_state','research_incumbent','state','champion')} for c in memory['cohorts']]))
        kwargs['reference_files']=files
        if role=='researcher':
            from orchestrator.planning_contract import baseline_capability
            for item in evidence:
                ref=item.get('frozen_reference')
                if ref:capabilities.append(baseline_capability(store,ref,repo_root))
        if evidence:
            # Full evidence is mounted read-only. Repeating every configuration,
            # critique and feedback block here needlessly exhausts agent context.
            index=[{**{k:item[k] for k in ('run_id','hypothesis_id','candidate_id','status','git_sha')},
                    'failure':item['error'][:500],
                    'comparison_status':item['comparison'].get('status'),
                    'evidence_file':f"/workspace/evidence/{item['run_id']}/evidence.json"}
                   for item in evidence]
            prompt += ('\nRelevant committed prior implementations, failures, exact frozen baseline references and '
                       'aggregate results follow. Read their /workspace/evidence/<run_id>/model.py, '
                       'configuration.json and evidence.json. Historical one-arm results are exploratory, '
                       'not controlled proof; a high single metric is not a winner. Read detailed evidence files '
                       'and cumulative memory as needed; this is an index, not a five-result history window.\n' + json.dumps(index))
            kwargs['reference_files'] = files
    if campaign:
        campaigns, campaign_id = campaign
        row = campaigns.get(campaign_id)
        limits = json.loads(row['config_json'])['limits']
        config = json.loads(row['config_json'])
        from orchestrator.campaign_config import time_based
        if time_based(config):
            context = next((json.loads(e['payload_json']) for e in campaigns.events(campaign_id, 10000)
                            if e['kind']=='STARTING_CONTEXT'), {})
            prompt += '\nOperator starting context (references, never automatic weight reuse): '+json.dumps(context)
            prompt += (f"\nFor new v4 plans, required elapsed training-loop time per model: {limits['training_minutes_per_experiment']*60:g} seconds. "
                       "Choose a continuing optimization procedure that reaches the complete target; finite schedules must not stop early; "
                       "never pad time with idle or dummy work. Explain the continuing scientific "
                       "schedule in evaluation_plan. Loop time includes batch loading/preparation, augmentation, transfers, learning, integrity checks and bookkeeping. Initialization, preflight, checkpoint saving and evaluation are excluded. Yield TrainingLoopStart(objective) before the first batch fetch. "
                       "One-time setup, initialization, smoke and evaluation use only the remaining overall research deadline. In-loop preparation counts toward the elapsed loop target. Older frozen plans keep their original timing contracts. "
                       "Preserve all declared arms, seeds and controls; never stretch or truncate a frozen plan.")
        remaining = row['created_at'] + limits['max_wall_hours'] * 3600 - __import__('time').time()
        if time_based(config):
            if remaining <= 0:
                from orchestrator.autonomy import CampaignHalted
                campaigns.set_status(campaign_id,'LIMIT_REACHED','insufficient research duration for '+role+' and required validation')
                raise CampaignHalted('insufficient research duration for '+role+' and required validation')
        elif remaining < 10:
            raise RuntimeError('campaign wall-time budget exhausted before Codex call')
        if time_based(config):
            from orchestrator.campaign_config import operation_seconds,operation_cancelled
            kwargs['timeout']=operation_seconds(row,180)
            kwargs['deadline']=row['created_at']+limits['max_wall_hours']*3600
            kwargs['cancel']=lambda:operation_cancelled(campaigns,campaign_id)
        else:
            kwargs['timeout'] = min(int(kwargs.get('timeout', 180)), remaining)
        recent_failures = [e for e in campaigns.events(campaign_id,100)
                           if e['kind'] in {'ATTEMPT_FAILED','VALIDATION_FAILED','PLAN_REVIEW_FAILED','CONTROLLER_ERROR','RECOVERY_RESTORED'}][:10]
        if recent_failures:
            prompt += '\nRecent exact campaign failures (do not erase or rename them): '+json.dumps(recent_failures)
    if role in {'researcher', 'critic'} and not kwargs.get('personal_mode'):
        from orchestrator.development_feedback import latest_summary
        feedback = []
        for r in store.list_runs(30):
            summary = latest_summary(store, r['run_id'])
            if summary:
                feedback.append(dict(summary, run_id=r['run_id'], candidate_id=r['candidate_id'], hypothesis_id=r['hypothesis_id']))
        if feedback:
            prompt += ('\nAuthorized real-development feedback follows. Use aggregate evidence to select methods, '
                       'but never use anomaly images/masks to fit model weights, normalization or thresholds. '
                       'Custom split, not official full-test performance. Missing AU-sPRO is unavailable, not zero. '
                       + json.dumps(feedback))
    if role=='researcher':
        from orchestrator.planning_contract import planning_instructions
        prompt += '\nTRUSTED EXECUTABLE PLANNING CONTRACT:\n'+planning_instructions(capabilities, time_only=not campaign or time_based(config))
    # Compile and inspect the exact generated provider schema before reserving
    # any campaign/global call. Schema defects are host errors, never agent repairs.
    schema_value=None
    if schema is not None:
        from orchestrator.codex_runner import provider_output_schema
        schema_value=json.loads(Path(schema).read_text())
        if role=='researcher':
            from orchestrator.planning_contract import constrain_output_schema
            schema_value=constrain_output_schema(schema_value,capabilities, time_only=not campaign or time_based(config),
                                                 research_request=bool(kwargs.get('personal_mode')))
        provider_output_schema(schema_value)
    if campaign:
        campaigns.reserve_codex_call(campaign_id, role)
        campaigns.event(campaign_id, 'AGENT_STARTED', role=role)
    else:
        store.reserve_codex_call(role)
    try:
        if role=='researcher' and schema_value is not None:
            import tempfile
            with tempfile.TemporaryDirectory(prefix='loco-planning-contract-') as folder:
                generated=Path(folder)/'schema.json'
                generated.write_text(json.dumps(schema_value))
                result=invoke(role,repo_root,generated,prompt,**kwargs)
        else:
            result = invoke(role, repo_root, schema, prompt, **kwargs)
    except Exception as exc:
        if campaign:
            campaigns.event(campaign_id, 'AGENT_FAILED', role=role, error=str(exc))
        raise
    if campaign:
        campaigns.event(campaign_id, 'AGENT_COMPLETED', role=role)
    return result


def budgeted_invoke(store, role, repo_root, schema, prompt, campaign=None, **kwargs):
    # Researcher/Implementer combine response validation with their downstream
    # checks below, sharing one repair allowance. Scout/Critic validate at the
    # structured-output boundary and receive the same correction mechanism.
    if role in {'scout', 'critic'}:
        def request(feedback, attempt):
            return _budgeted_invoke_once(store, role, repo_root, schema, prompt + feedback,
                                         campaign=campaign, **kwargs)
        result, _, _ = _with_agent_correction(store, campaign, role, request, lambda result: None)
        return result
    return _budgeted_invoke_once(store, role, repo_root, schema, prompt, campaign=campaign, **kwargs)


def _validate_research_proposals(store, proposals, count, require_gpu_pilot):
    """Validate the entire batch before persisting any accepted proposal."""
    if len(proposals) < count:
        raise ValueError(f"researcher returned {len(proposals)}; need at least {count}")
    # Distinct scientific questions may concern the same architecture: losses,
    # learning dynamics, ablations and independent replication are valid research.
    # Mechanism/identity deduplication below remains mandatory.
    ranks = [p.get('priority_rank') for p in proposals]
    if len(set(ranks)) != len(proposals) or any(not isinstance(r, int) or r < 1 for r in ranks):
        raise ValueError('hypothesis priority ranks must be distinct positive integers')
    prior = [json.loads(row['proposal_json']) for row in list_proposals(store)]
    def signature(item):
        return ' '.join((item.get('claim', '') + ' ' + item.get('controlled_change', '')).lower().split())
    for proposal in proposals:
        followup = proposal.get('followup_of', '')
        followup_kind = proposal.get('followup_kind', 'new')
        if followup_kind != 'new' and not any(p['hypothesis_id']==followup for p in prior):
            raise ValueError('scientific follow-up must name an existing parent hypothesis')
        if followup_kind == 'new' and followup:
            raise ValueError('new hypothesis cannot silently reuse a parent identity')
        minutes = proposal.get('estimated_gpu_minutes')
        needs_training=any(a.get('execution')=='train' for a in proposal.get('experiment_plan',{}).get('arms',[]))
        if require_gpu_pilot and needs_training and (isinstance(minutes, bool) or not isinstance(minutes, (int, float))
                                  or not __import__('math').isfinite(minutes) or minutes <= 0
                                  or 'cpu-only' in proposal.get('evaluation_plan', '').lower()):
            raise ValueError('CUDA pilot requires GPU-trained hypotheses within authorized limits; CPU-only proposals remain research leads')
        if any(not isinstance(dep, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:[<>=!~]+[0-9][A-Za-z0-9_.-]*)?', dep)
               for dep in proposal.get('new_dependencies', [])):
            raise ValueError('new_dependencies must contain package names only; use [] for none')
        duplicate = next((old for old in prior if proposal['hypothesis_id'] == old['hypothesis_id'] or
                          (not (old['hypothesis_id']==followup and followup_kind!='new') and
                           SequenceMatcher(None, signature(proposal), signature(old)).ratio() > 0.92)), None)
        if duplicate:
            raise ValueError(f"duplicate research proposal: {proposal['hypothesis_id']}; "
                             f"matches {duplicate['hypothesis_id']}: {duplicate.get('claim', '')} "
                             f"Controlled change: {duplicate.get('controlled_change', '')}")
        if 'experiment_plan' not in proposal:
            raise ValueError('structured experiment_plan required for newly generated hypotheses')
        if proposal['experiment_plan'].get('hypothesis_id')!=proposal['hypothesis_id']:
            raise ValueError('plan hypothesis identity differs from its proposal')
        from orchestrator.planning_contract import admit_plan
        admit_plan(proposal['experiment_plan'],store,REPO)
        prior.append(proposal)


def _validation_event(store, campaign, kind, role, **payload):
    """Retain rejection/repair evidence without marking a research failure."""
    data = dict(payload, role=role)
    with store.connect() as con:
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)',
                    (__import__('time').time(), kind, json.dumps(data)))
    if campaign:
        campaign[0].event(campaign[1], kind, **data)


def _with_agent_correction(store, campaign, role, request, validate, *,
                           hypothesis_id=None, initial_feedback=None):
    """One shared correction for agent-owned output/validation failures.

    Classification uses the failure's origin, not message matching. Only the
    validation callback handles generated proposals/files/tests; host operations
    (workspace creation, scheduling, database writes) stay outside that boundary.
    """
    import sqlite3
    import traceback
    from orchestrator.autonomy import CampaignHalted, await_authorized_stage
    feedback = ('\nTrusted candidate execution failure to correct:\n' + initial_feedback) if initial_feedback else ''
    attempts = 1 if initial_feedback is not None else 2
    timed=False
    if campaign:
        from orchestrator.campaign_config import time_based
        timed=time_based(json.loads(campaign[0].get(campaign[1])['config_json']))
    for attempt in range(attempts):
        if campaign:
            await_authorized_stage(campaign[0], campaign[1], finishing=True)
        phase = 'agent_response'
        response = None
        try:
            response = request(feedback, attempt)
            phase = 'generated_work_validation'
            validation = validate(response)
            if timed:campaign[0].validated_stage(campaign[1],role,hypothesis_id=hypothesis_id)
            return response, validation, initial_feedback is not None or attempt > 0
        except Exception as exc:
            # These controls must never become instructions to change the host,
            # request extra privileges, edit budgets, or weaken trusted checks.
            if isinstance(exc, (BudgetExceeded, CampaignHalted, PermissionError, sqlite3.Error, subprocess.SubprocessError)):
                raise
            if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
                raise
            if phase == 'agent_response' and not isinstance(exc, InvalidAgentOutput):
                raise  # CLI/transport/host failure, rather than generated output.
            detail = {'hypothesis_id': hypothesis_id, 'attempt': attempt+1, 'stage': phase,
                      'error_type': type(exc).__name__, 'error': str(exc),
                      'traceback': traceback.format_exc(limit=8)[-12000:]}
            _validation_event(store, campaign, 'VALIDATION_FAILED', role, **detail)
            if attempt+1 == attempts:
                raise
            _validation_event(store, campaign, 'REPAIR_REQUESTED', role, **detail)
            feedback = ('\nTrusted failure report for correction:\n' + json.dumps(detail) +
                        '\nDiagnose and correct the cause in your own generated output or assigned candidate files. '
                        'Return a complete response satisfying the same contract. Preserve scientific intent and evidence. '
                        'Do not weaken tests, trusted evaluators, restrictions or budgets. If correction is impossible '
                        'within your authority, report that limitation accurately. Rejected response: ' +
                        json.dumps(response, default=str)[-24000:])


def add_research_proposals(store, prompt, count=3, campaign=None, personal_mode=False,
                           require_gpu_pilot=False):
    schema = REPO / "schemas" / "proposals.schema.json"
    previous = [json.loads(row['proposal_json']) for row in list_proposals(store)[:50]]
    instruction = prompt + '\nPreviously recorded hypotheses (do not repeat IDs or mechanisms): ' + json.dumps([
        {key: p.get(key, '') for key in ('hypothesis_id', 'claim', 'controlled_change')} for p in previous])
    instruction += (' Return distinct mechanisms; renaming the same idea is insufficient. '
                    'Use unique hypothesis IDs and valid priority ranks. Different architecture families are not required; '
                    'distinct training changes, analysis-driven improvements and honest replications are valid. '
                    'Existing models, implementations, measured failures and scientific memory are primary research resources. '
                    'Choose whether to improve, investigate, replicate, ablate or explore a fundamentally different method. '
                    'Cite internal run/plan/commit evidence when it motivates a proposal; external papers are optional. ')
    if personal_mode:
        instruction += (' The generated batch schema has literature_request. Normally return literature_request="" '
                        'and complete executable proposals grounded in internal evidence. If an external knowledge gap '
                        'requires the Literature Scout, return that precise question in literature_request and proposals=[]. '
                        'The trusted controller routes the question within the same budgets. Do not fabricate plans or '
                        'citations to request research. After supplied Scout findings, finish a complete proposal batch. ')
    instruction += (' Every proposal requires experiment_plan matching the controller-generated structured-output schema (the on-disk v1 schema is retained only for legacy compatibility). '
                    'Declare every arm, seed, complete training settings, size, batch_size, a continuing target-aware schedule, '
                    'prediction settings, required '
                    'measurements, executable diagnose hook requests, controlled variables and resource estimates (not independent deadlines). '
                    'Frozen baselines must use the exact supplied run ID, source/checkpoint/configuration hashes '
                    'and seed, never a historical hypothesis name. Do not claim future measurements are available. '
                    'COMPLETE only means requested work executed; missing arms or diagnostics mean INCOMPLETE. '
                    'You may explicitly complete, replicate or ablate a previous hypothesis using followup_of '
                    'and followup_kind, with a new unique hypothesis/plan identity. Do not evade an incomplete '
                    'comparison by superficially renaming a different single-arm experiment. Use followup_kind=new '
                    'and followup_of="" only for new mechanisms. ')
    if campaign:
        campaigns, cid = campaign
        limits = json.loads(campaigns.get(cid)['config_json'])['limits']
        usage = campaigns.usage(cid)
        wall = max(0, limits['max_wall_hours']*3600-usage['wall_seconds'])
        remaining = {'max_jobs': (max(0, int(wall)) if limits['max_experiments'] is None else limits['max_experiments']-usage['experiments']),
                     'max_gpu_seconds':min(wall, limits['max_gpu_hours']*3600-usage['gpu_seconds']),
                     'max_wall_seconds':wall, 'max_job_seconds':limits['max_job_seconds']}
        from orchestrator.campaign_config import time_based
        if time_based(json.loads(campaigns.get(cid)['config_json'])):
            instruction += '\nRemaining overall research seconds: '+str(wall)+'. Use schema_version 4.0 without resource_budget. Job count is the complete arms × seeds matrix.'
        else:
            instruction += '\nRemaining ceilings for ONE comparison (not spending targets): '+json.dumps(remaining)
    def request(feedback, attempt):
        notes=''
        for stage in range(2):
            output = budgeted_invoke(store, "researcher", REPO, schema, instruction + notes + feedback, timeout=240,
                                     campaign=campaign, personal_mode=personal_mode)["response"]
            _validation_event(store, campaign, 'RESEARCHER_RESPONSE', 'researcher',
                              attempt=attempt+1, response=output)
            query=output.get('literature_request','')
            if not query:return output
            if not personal_mode or not isinstance(query,str) or not query.strip() or output.get('proposals') or stage:
                raise InvalidAgentOutput('literature request must be a specific question with an empty proposal batch; use the supplied findings to finish proposals')
            if campaign and not json.loads(campaign[0].get(campaign[1])['config_json'])['research'].get('literature_scout',True):
                raise InvalidAgentOutput('Literature Scout is disabled; use available internal evidence or report the knowledge gap')
            import hashlib
            request_id=hashlib.sha256(query.strip().encode()).hexdigest()
            cached=next((json.loads(e['payload_json'])['leads'] for e in campaign[0].events(campaign[1],100000)
                         if e['kind']=='REQUESTED_LITERATURE_COMPLETED' and json.loads(e['payload_json']).get('request_id')==request_id),None) if campaign else None
            if cached is None:
                cached=scout_literature(store,campaign=campaign,personal_mode=personal_mode,question=query.strip())
                if campaign:campaign[0].event(campaign[1],'REQUESTED_LITERATURE_COMPLETED',request_id=request_id,question=query.strip(),leads=cached)
            notes='\nScout findings for your requested knowledge gap (agent summaries, not independent verification): '+json.dumps(cached)
    def validate(output):
        from orchestrator.experiment_plan import decode_agent_plan
        for proposal in output['proposals']:
            if proposal.get('experiment_plan'):
                proposal['experiment_plan'] = decode_agent_plan(proposal['experiment_plan'])
        _validate_research_proposals(store, output['proposals'], count, require_gpu_pilot)
        from orchestrator.experiment_plan import verify_baseline
        for proposal in output['proposals']:
            plan = proposal['experiment_plan']
            for reference in plan['baselines']: verify_baseline(store, reference, REPO)
            if campaign and plan.get('schema_version') in {'3.0','4.0'}:
                from orchestrator.campaign_config import training_execution_estimate
                cfg=json.loads(campaigns.get(cid)['config_json'])
                estimate=training_execution_estimate(store,cfg,jobs=len(plan['arms'])*len(plan['seeds']),
                    newly_trained=sum(a['execution']=='train' for a in plan['arms'])*len(plan['seeds']),
                    real_feedback=plan['evaluation']['real_development_feedback'],critique=True,
                    source_sha256='not-implemented-yet',loop_timed=plan.get('schema_version')=='4.0')
                if estimate['required_seconds']>remaining['max_wall_seconds']:
                    raise ValueError('required-duration matrix cannot fit remaining research time including preparation/evaluation; revise the plan, not the target')
            if campaign and plan.get('resource_budget') and any(plan['resource_budget'][key] > allowance for key,allowance in remaining.items()):
                raise ValueError('proposed comparison exceeds remaining authorized campaign allowance')
            if campaign:
                from orchestrator.campaign_config import time_based, plan_seconds
                configuration=json.loads(campaigns.get(cid)['config_json'])
                if time_based(configuration) and plan_seconds(plan,configuration)>remaining['max_wall_seconds']:
                    raise ValueError('complete comparison including frozen evaluation and critique cannot fit remaining research duration')
        return output['proposals']
    _, proposals, _ = _with_agent_correction(store, campaign, 'researcher', request, validate)
    for proposal in proposals:
        store.add_proposal(proposal)
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)',
                        (__import__('time').time(), 'HYPOTHESIS_RANKING', json.dumps({
                            'hypothesis_id': proposal['hypothesis_id'], 'priority_rank': proposal['priority_rank'],
                            'priority_rationale': proposal['priority_rationale']})))
    return proposals


def scout_literature(store, campaign=None, personal_mode=False, question=None):
    """Untrusted citation leads, not independently verified paper findings."""
    response = budgeted_invoke(
        store, "scout", REPO, REPO / "schemas" / "literature.schema.json",
        (question or "Investigate logical anomaly localization research for MVTec LOCO AD without presupposing a method family.") +
        " Describe likely failure modes "
        "and inference memory requirements. Cite sources; if you cannot verify a "
        "reference, explicitly mark it unverified. Do not access any local test data. "
        "Do not call MCP tools; use a source URL only when you can verify it within the time limit.", timeout=120,
        campaign=campaign, personal_mode=personal_mode
    )["response"]
    filename = REPO / "literature" / "scout_notes.json"
    filename.write_text(json.dumps({"generated_by": "Codex literature scout; UNVERIFIED", "result": response}, indent=2))
    return response


def list_proposals(store):
    with store.connect() as con:
        rows = con.execute("SELECT * FROM hypotheses ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def _validate_candidate_files(tree, hypothesis_id):
    """Allow repair of extra regular files only inside the assigned directory.

    Outside-workspace changes and symlinks remain hard permission failures.
    The controller never removes files on the agent's behalf.
    """
    candidate = tree / 'candidates' / hypothesis_id
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(tree.resolve()):
        raise PermissionError('candidate directory must not be a symlink')
    approved_names = {'model.py', 'README.md', 'test_model.py', 'candidate_manifest.json'}
    approved_files = {f'candidates/{hypothesis_id}/{name}' for name in approved_names}
    changed = subprocess.run(['git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=all'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    extras = set()
    for line in changed:
        filename = line[3:].replace('"', '')
        path = tree / filename
        if ('..' in Path(filename).parts or not path.is_relative_to(candidate)
                or path.is_symlink() or not path.resolve().is_relative_to(candidate.resolve())):
            raise PermissionError(f'Agent attempted to modify out-of-scope path: {filename}')
        if filename not in approved_files:
            extras.add(filename)
    for item in candidate.rglob('*'):
        if item.is_symlink() or not item.resolve().is_relative_to(candidate.resolve()):
            raise PermissionError(f'Agent wrote an unsafe candidate path: {item.name}')
        if item.parent != candidate or item.name not in approved_names:
            extras.add(str(item.relative_to(tree)))
    if extras:
        raise ValueError('Unapproved candidate files inside assigned workspace: ' + ', '.join(sorted(extras)) +
                         '. Remove these extra generated files. Allowed files: model.py, README.md, '
                         'test_model.py, candidate_manifest.json. Put licensing/provenance notes in README.md '
                         'or candidate_manifest.json. Do not modify files outside your assigned candidate directory.')


def implement_proposal(store, hypothesis_id, campaign=None, personal_mode=False,
                       require_torch_cuda=False, validation_runtime=None,
                       repair_feedback=None):
    with store.connect() as con:
        row = con.execute("SELECT proposal_json FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)).fetchone()
    if not row:
        raise KeyError(hypothesis_id)
    proposal = json.loads(row["proposal_json"])
    if campaign:
        from orchestrator.campaign_config import time_based, plan_seconds, remaining_wall
        campaign_row=campaign[0].get(campaign[1]);configuration=json.loads(campaign_row['config_json'])
        if time_based(configuration) and remaining_wall(campaign_row,__import__('time').time()) < plan_seconds(proposal['experiment_plan'],configuration):
            from orchestrator.autonomy import CampaignHalted
            campaign[0].set_status(campaign[1],'LIMIT_REACHED','insufficient research duration to implement and execute complete plan')
            raise CampaignHalted('insufficient research duration to implement and execute complete plan')
    reuse=False
    if campaign:
        from orchestrator.campaign_config import time_based
        reuse=time_based(json.loads(campaign[0].get(campaign[1])['config_json'])) and (REPO/'candidates/worktrees'/hypothesis_id).is_dir()
    if repair_feedback is None and not reuse:
        tree = create_candidate_worktree(REPO, hypothesis_id)
    else:
        tree = (REPO / 'candidates' / 'worktrees' / hypothesis_id).resolve(strict=True)
        if not tree.is_relative_to((REPO / 'candidates' / 'worktrees').resolve()):
            raise PermissionError('repair workspace escapes assigned candidates')
    allowed_file = tree / "candidates" / hypothesis_id / "model.py"
    schema = REPO / "schemas" / "implementation.schema.json"
    instruction = (
        f"Implement the exact declared scientific experiment for hypothesis {hypothesis_id} "
        f"NOW as the module candidates/{hypothesis_id}/model.py "
        "inside your assigned worktree. The fixed trusted runner calls "
        "train(config: dict, allowed_train_manifest: Path, output_dir: Path), then "
        "load(output_dir / 'checkpoint_manifest.json', device=config['device']), then "
        "predict(images) where images are RGB float32 [B,3,H,W]. Return finite "
        "anomaly_maps [B,1,H,W] and image_scores [B]. Save "
        "output_dir/checkpoint_manifest.json and output_dir/model.pt; the trusted "
        "runner measures the latter artifact. Read-only scripts/run_candidate.py "
        "shows the precise protocol. Do not alter its calling convention. "
        "Actually create model.py, a short README, mandatory test_model.py and optionally candidate_manifest.json; "
        "a proposal or plan alone fails verification. "
        "Tests run first inside a CPU-only sandbox with no NVIDIA devices. At least one real CPU test must pass. "
        "Test rejection of non-training splits before device setup or image reads, invalid inputs, and applicable output-shape logic on CPU. "
        "Never skip the whole test module or class based on CUDA availability. Apply CUDA skip rules only to individual GPU tests. "
        "The trusted queued job then runs a separate bounded CUDA train/load/predict smoke check using two train/good images, "
        "steps=2, epochs=1 and batch_size=1, before pilot training. Honor those bounded smoke settings. "
        "Use system temporary directories for test fixtures because candidate source is read-only. "
        "Only model.py, README.md, test_model.py and candidate_manifest.json are permitted in your candidate directory. "
        "Put licensing notes in README.md or candidate_manifest.json; do not create LICENSE or other extra files. "
        "Do not change other directories. " +
        ("This pilot must implement a real trainable PyTorch model. If config['device'] is cuda, "
         "move model and training tensors to CUDA, train there, and run inference there. "
         "Do not claim GPU execution if no CUDA tensors are used. " if require_torch_cuda else "") +
        f"Proposal: {json.dumps(proposal, sort_keys=True)}"
    )
    if campaign:
        from orchestrator.campaign_config import time_based
        if time_based(json.loads(campaign[0].get(campaign[1])['config_json'])):
            instruction += ('\nTime-budgeted learning requires training_steps(config, allowed_train_manifest, output_dir), '
                'a generator yielding scripts.training_phase.TrainingStep(objective=Module returning scalar differentiable loss, '
                'optimizer=initialized Optimizer, args=tuple of prepared device tensors). Initialize models and optimizers, '
                'prepare batches/transfers, log progress and save checkpoint metadata BETWEEN yields, never inside forward. '
                'Return checkpoint metadata from the generator. The trusted loop executes forward/loss, backward and optimizer '
                'updates and measures only those phases. Explicit use_closure=True supports closure-based optimizers. '
                'Read scripts/training_phase.py. Preserve train/load/predict for legacy callers. No learning outside yielded steps; '
                'the trusted driver rejects unmeasured parameter changes. Missing/unsupported boundaries are UNVERIFIABLE, '
                'never a reason to time the whole train function or alter a frozen scientific plan.')
    if proposal.get('experiment_plan'):
        instruction += ('\nImplement this exact experiment_plan, not a default demonstration. Honor all declared '
            'training settings, seeds, arms and inference modes. steps is a total schedule, not an implicit one-epoch cap. '
            'Save checkpoint metadata training_audit={seed, optimizer_updates, source_exposures}; optimizer_updates '
            'counts actual optimizer.step calls across all optimizers. The runner independently counts calls and '
            'rejects mismatches with expected_optimizer_updates/expected_source_exposures. Exposure counts must be '
            'instrumented in your minibatch loop, including sampling with replacement. Smoke checks use their own '
            'two-update/two-exposure configuration and disposable weights. For nonempty prediction settings implement '
            'load(checkpoint_manifest, device, inference_config) and set predictor.inference_config to the exact '
            'consumed dictionary; never ignore settings. For each requested diagnostic implement '
            'diagnose(predictor, assessment_good_images, diagnostic_config), returning {metrics:{...}, ...} '
            'with all named finite measurements. It runs on frozen weights in inference mode on disjoint normal '
            'assessment inputs and permitted derived synthetic data, never real-development images/masks. '
            'Persist per-image diagnostic evidence where meaningful in the returned JSON. A missing arm, seed, '
            'control or endpoint means INCOMPLETE; do not substitute a historical recipe from its name.')
    import subprocess
    from orchestrator.experiment_runner import verify_candidate_source
    def request(feedback, attempt):
        timeout = 600
        if campaign:
            from orchestrator.campaign_config import time_based, plan_seconds, remaining_wall
            row=campaign[0].get(campaign[1]);configuration=json.loads(row['config_json'])
            if time_based(configuration):
                timeout=remaining_wall(row,__import__('time').time())
                if timeout <= 0:
                    from orchestrator.autonomy import CampaignHalted
                    campaign[0].set_status(campaign[1],'LIMIT_REACHED','insufficient research duration to implement and execute complete plan')
                    raise CampaignHalted('insufficient research duration to implement and execute complete plan')
        response = budgeted_invoke(store, "implementer", REPO, schema, instruction + feedback,
                                   timeout=timeout, worktree=tree, campaign=campaign, personal_mode=personal_mode,
                                   evidence_run_ids=[b['run_id'] for b in proposal.get('experiment_plan',{}).get('baselines',[])])
        _validation_event(store, campaign, 'IMPLEMENTER_RESPONSE', 'implementer',
                          hypothesis_id=hypothesis_id, attempt=attempt+1, response=response['response'])
        return response
    def validate(response):
        if response["response"].get("new_dependencies") or response['response'].get('external_weights'):
            raise PermissionError("Agent requested dependencies or external weights; provenance and operator approval required")
        _validate_candidate_files(tree, hypothesis_id)
        if response['response'].get('candidate_id') != hypothesis_id or not all(
                response['response'].get(k) is True for k in ('train_contract', 'load_contract', 'predict_contract')):
            raise ValueError('implementer did not attest the assigned candidate contract')
        if not allowed_file.is_file():
            raise ValueError('implementation missing model.py')
        verify_candidate_source(allowed_file)
        if proposal.get('experiment_plan'):
            import ast
            functions = {n.name:n for n in ast.parse(allowed_file.read_text()).body if isinstance(n,ast.FunctionDef)}
            plan = proposal['experiment_plan']
            if plan['diagnostics'] and 'diagnose' not in functions:
                raise ValueError('experiment plan requires diagnose(predictor, assessment_good_images, diagnostic_config)')
            if any(a['execution']=='train' and a['inference'] for a in plan['arms']):
                if 'inference_config' not in {a.arg for a in functions['load'].args.args}:
                    raise ValueError('planned inference intervention requires explicit load inference_config argument')
        if campaign:
            from orchestrator.campaign_config import time_based
            if time_based(json.loads(campaign[0].get(campaign[1])['config_json'])):
                from orchestrator.experiment_runner import require_training_phases
                require_training_phases(allowed_file)
        if validation_runtime is not None:
            from orchestrator.experiment_runner import probe_candidate_interface, run_candidate_unit_tests
            good_root = Path(os.environ['LOCO_GOOD_ROOT'])
            timing={}
            if campaign and time_based(json.loads(campaign[0].get(campaign[1])['config_json'])):
                from orchestrator.campaign_config import operation_seconds,operation_cancelled
                timing={'timeout':operation_seconds(campaign[0].get(campaign[1]),30),
                        'cancel':lambda:operation_cancelled(campaign[0],campaign[1])}
            probe_candidate_interface(REPO, allowed_file, good_root, validation_runtime, hypothesis_id,**timing)
            if timing:timing['timeout']=operation_seconds(campaign[0].get(campaign[1]),30)
            cpu_result = run_candidate_unit_tests(REPO, allowed_file, good_root, validation_runtime, hypothesis_id,**timing)
            if campaign:
                campaign[0].event(campaign[1], 'CANDIDATE_TESTS', hypothesis_id=hypothesis_id, **cpu_result)
            return cpu_result
        return None
    response, cpu_result, repair_used = _with_agent_correction(store, campaign, 'implementer',
        request, validate, hypothesis_id=hypothesis_id, initial_feedback=repair_feedback)
    store.set_hypothesis_status(hypothesis_id, "IMPLEMENTED")
    return {"worktree": str(tree), "candidate_module": str(allowed_file), "agent": response["response"],
            "cpu_validation": cpu_result, "repair_used": repair_used}


def queue_candidate(store, hypothesis_id, candidate_id, category, stage, device, manifest_path, seconds=300,
                    personal_mode_authorized=False, campaign_id=None, campaign_name=None,
                    context_arm=None, seed=None, synthetic_evaluation=True, cuda_smoke_required=False,
                    effective_config=None, run_id=None):
    from benchmark.data_guard import load_manifest, manifest_fingerprint
    from orchestrator.experiment_runner import candidate_file, sha256
    with store.connect() as con:
        declared = con.execute('SELECT proposal_json FROM hypotheses WHERE hypothesis_id=?',(hypothesis_id,)).fetchone()
    if declared and json.loads(declared[0]).get('experiment_plan') and effective_config is None:
        raise ValueError('structured experiment plan must use loco-lab research; refusing default queue settings')
    if candidate_id != "toy_normal_ae" and not personal_mode_authorized and os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Operator must verify OS isolation before queueing untrusted candidates")
    if effective_config is not None and effective_config.get('plan_sha256'):
        from orchestrator.planning_contract import admit_plan
        with store.connect() as con:
            registered=con.execute("SELECT payload_json FROM campaign_events WHERE campaign_id=? AND kind='EXPERIMENT_PLAN_REGISTERED'",
                                   (campaign_id,)).fetchall()
        record=next((json.loads(r[0]) for r in registered if json.loads(r[0])['plan_sha256']==effective_config['plan_sha256']),None)
        if record is None:raise ValueError('planned queue admission requires a registered validated protocol')
        admit_plan(record['plan'],store,REPO)
        if 'experiment-plan-v'+record['plan']['schema_version'] != effective_config.get('plan_contract_version'):
            raise ValueError('effective timing contract differs from registered scientific plan')
    frozen = (effective_config or {}).get('frozen_baseline')
    if frozen:
        from orchestrator.experiment_plan import verify_baseline
        _, origin = verify_baseline(store, frozen, REPO)
        module = None
    else:
        module = candidate_file(REPO, candidate_id)
    manifest, paths = load_manifest(manifest_path)
    if not any(item["category"] == category for item in manifest["images"]):
        raise ValueError("category absent from authorized good-only manifest")
    from orchestrator.campaign_config import time_based
    campaign_config = None
    if campaign_id:
        from orchestrator.campaign_state import CampaignState
        campaign_row = CampaignState(store).get(campaign_id)
        if not campaign_row: raise ValueError('unknown campaign')
        campaign_config = json.loads(campaign_row['config_json'])
    if effective_config and effective_config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'} and not frozen:
        from orchestrator.campaign_config import training_execution_estimate,remaining_wall
        estimate=training_execution_estimate(store,campaign_config,source_sha256=sha256(module),loop_timed=effective_config.get('plan_contract_version')=='experiment-plan-v4.0')
        if remaining_wall(campaign_row,__import__('time').time())<estimate['required_seconds']:
            raise ValueError('insufficient research time for required training plus preparation and evaluation')
    if campaign_config and time_based(campaign_config) and not frozen:
        from orchestrator.experiment_runner import require_training_phases
        require_training_phases(module,target=True,loop=effective_config.get('plan_contract_version')=='experiment-plan-v4.0') if effective_config and effective_config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'} else require_training_phases(module)
    campaign_timed = bool(effective_config and effective_config.get('plan_contract_version') in {'experiment-plan-v2.0','experiment-plan-v3.0','experiment-plan-v4.0'})
    enforce(store, next_gpu_seconds=seconds if device == "cuda" and not campaign_timed else 0, campaign_id=campaign_id)
    timeout = (None if campaign_timed else min(int(seconds),
               campaign_config['limits']['max_job_seconds'] if campaign_config and time_based(campaign_config) else max_job_seconds()))
    if effective_config is None and seed is not None and (candidate_id != 'masked_context_ablation' or seed not in {7, 17, 29}):
        raise ValueError('seed is restricted to the predeclared masked-context study seeds')
    cfg = {"device": device, "size": 128, "epochs": 1, "seed": 7 if seed is None else seed,
           "timeout_seconds": timeout, "category": category, "model_sha256": frozen['source_sha256'] if frozen else sha256(module),
           "metric_evaluator_version": "development-v2.0", "synthetic_evaluation": bool(synthetic_evaluation)}
    if cuda_smoke_required:
        cfg['cuda_smoke_required'] = True
    if context_arm is not None:
        if candidate_id != 'masked_context_ablation' or context_arm not in {'local', 'global', 'combined', 'neither', 'spatial'}:
            raise ValueError('context_arm is only available for the reviewed masked-context ablation')
        cfg.update(context_arm=context_arm, steps=100, batch_size=4)
    if effective_config is not None:
        # Compiled by trusted plan code, not defaults or agent shell commands.
        cfg = dict(effective_config)
        cfg.update(device=device, category=category, timeout_seconds=timeout,
                   synthetic_evaluation=bool(synthetic_evaluation), cuda_smoke_required=bool(cuda_smoke_required))
        actual_hash = frozen['source_sha256'] if frozen else sha256(module)
        if cfg.get('model_sha256') != actual_hash:
            raise ValueError('planned source hash changed before queue submission')
    if campaign_id:
        cfg["campaign_id"] = campaign_id
        if campaign_name:
            cfg["campaign_name"] = campaign_name
        cfg["execution_mode"] = "personal_bwrap" if personal_mode_authorized else "isolated"
    if campaign_config and time_based(campaign_config):
        cfg['campaign_configuration_version'] = campaign_config['configuration_version']
        cfg['training_budget_seconds'] = campaign_config['limits']['training_minutes_per_experiment']*60
    if campaign_timed:
        cfg.pop('timeout_seconds',None)
        cfg['research_deadline'] = campaign_row['created_at']+campaign_config['limits']['max_wall_hours']*3600
        if effective_config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'} and not frozen:
            cfg['training_completion_policy']='elapsed-training-loop-v1' if effective_config.get('plan_contract_version')=='experiment-plan-v4.0' else 'required-training-target-v1'
            cfg['diagnostic_preflight_required']=True
    candidate_tree = REPO / "candidates" / "worktrees" / candidate_id
    if frozen:
        git_sha = origin['git_sha']
    elif candidate_tree.is_dir() and module.resolve().is_relative_to(candidate_tree.resolve()):
        git_sha = subprocess.run(["git", "-C", str(candidate_tree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    else:
        git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if run_id and store.get_run(run_id):
        prior = store.get_run(run_id)
        if json.loads(prior['config_json']) != cfg:
            raise ValueError('stable plan job identity has conflicting configuration')
        return run_id
    run_id = store.create_run(hypothesis_id, candidate_id, category, stage, cfg, git_sha=git_sha,
                              manifest_sha256=manifest_fingerprint(manifest), run_id=run_id)
    for state in ("IMPLEMENTED", "VALIDATED", "QUEUED"):
        store.transition(run_id, state)
    return run_id


def run_research_campaign(store, manifest_path, device="cuda", max_experiments=3):
    """Bounded, operator-enabled Codex -> implementation -> worker -> critic cycle.

    This path is refused until OS isolation is independently verified. It does
    not and cannot open the sealed LOCO official test directory.
    """
    if os.getenv("LOCO_ALLOW_UNATTENDED") != "1" or os.getenv("LOCO_SANDBOX_VERIFIED") != "1":
        raise PermissionError("Set LOCO_ALLOW_UNATTENDED=1 only AFTER restricted OS identities/mounts are tested")
    if not manifest_path:
        raise ValueError("campaign needs --manifest")
    from benchmark.data_guard import load_manifest, CATEGORIES
    manifest, _ = load_manifest(manifest_path)
    if not list_proposals(store):
        leads = scout_literature(store)
        add_research_proposals(store, "Propose 3 genuinely distinct, falsifiable MVTec LOCO logical heatmap methods; normal-only learning, no inference memory bank, no sealed benchmark access. Treat these scout leads as UNVERIFIED until independently checked: " + json.dumps(leads))
    completed = []
    proposals = [p for p in list_proposals(store) if p["status"] == "PROPOSED"]
    proposals.sort(key=lambda row: json.loads(row['proposal_json']).get('priority_rank', 999))
    if any(json.loads(row['proposal_json']).get('experiment_plan') for row in proposals):
        raise ValueError('legacy campaign cannot execute structured plans; use loco-lab research start --config APPROVED_CONFIG')
    for row in proposals[:max_experiments]:
        enforce(store, next_gpu_seconds=300 if device.startswith("cuda") else 0)
        hypothesis_id = row["hypothesis_id"]
        proposal = json.loads(row["proposal_json"])
        if proposal["new_dependencies"]:
            store.set_hypothesis_status(hypothesis_id, "INVALID")
            continue
        try:
            implement_proposal(store, hypothesis_id)
            eligible_categories = [cat for cat in CATEGORIES if any(item["category"] == cat and item["split"] == "validation/good" for item in manifest["images"])]
            if not eligible_categories:
                raise ValueError("No permitted normal validation images")
            category = eligible_categories[0]
            rid = queue_candidate(store, hypothesis_id, hypothesis_id, category, "smoke", device, manifest_path, 300)
            outcome = run_worker_once(store, manifest_path, Path(os.getenv("LOCO_RUNTIME", ".runtime")))
            data = store.get_run(rid)
            if data["status"] == "COMPLETED":
                measured = data["metrics_json"]
                critique = budgeted_invoke(store, "critic", REPO, REPO / "schemas" / "review.schema.json", f"Review immutable synthetic-only measured run {rid}: {measured}. Hypothesis: {proposal}. Never equate proxy results with official LOCO.", timeout=180)
                with store.connect() as con:
                    con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)", (rid, __import__('time').time(), "CRITIC_REVIEW", json.dumps(critique["response"])))
                from orchestrator.decision_engine import eligible
                ok, reason = eligible(data)
                store.transition(rid, "REVIEWED")
                store.transition(rid, "PROMOTED" if ok and critique["response"]["recommended_next_action"] != "reject" else "REJECTED", reason=reason)
                from orchestrator.mlflow_tracker import enqueue_run_tracking
                enqueue_run_tracking(store, rid)
            store.set_hypothesis_status(hypothesis_id, "COMPLETED")
            completed.append({"hypothesis": hypothesis_id, "run_id": rid, "outcome": outcome})
        except Exception as exc:
            store.set_hypothesis_status(hypothesis_id, "FAILED")
            completed.append({"hypothesis": hypothesis_id, "error": str(exc)})
    # Persist a factual audit report; agent interpretation cannot overwrite metrics.
    from orchestrator.reporting import write_campaign_report
    report_file = write_campaign_report(store, Path(os.getenv("LOCO_RUNTIME", ".runtime")) / "exports")
    # If the scheduled batch is exhausted, use the observed failures/scores to
    # seed the next iteration rather than reverting to a fixed search list.
    remaining = [p for p in list_proposals(store) if p["status"] == "PROPOSED"]
    followup = []
    if not remaining and completed:
        history = [{"hypothesis":x.get("hypothesis"), "run":x.get("run_id"), "status":store.get_run(x["run_id"])["status"] if x.get("run_id") else "FAILED", "metrics":json.loads(store.get_run(x["run_id"])["metrics_json"]) if x.get("run_id") else {}, "error":x.get("error", "")} for x in completed]
        try:
            followup = add_research_proposals(store, "Generate THREE NEW, distinct and falsifiable research hypotheses based on these measured outcomes. Critically consider synthetic shortcut learning; do not retest rejected identical ideas: " + json.dumps(history))
        except Exception as exc:
            followup = [{"error":str(exc)}]
    return {"completed": completed, "report_file": str(report_file), "next_proposals":followup, "disclaimer": "Development synthetic proxies ONLY, no official LOCO test scores"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guarded LOCO research controller")
    ap.add_argument("command", choices=["init", "status", "pause", "resume", "reconcile", "propose", "implement", "queue-toy", "worker-once", "review", "cycle", "queue-candidate", "campaign", "scout"])
    ap.add_argument("--manifest", help="validated good-only manifest JSON")
    ap.add_argument("--hypothesis", help="hypothesis ID for implementation or review")
    ap.add_argument("--run", help="run ID for review")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--candidate", help="candidate ID for validated queue submission")
    ap.add_argument("--category", default="breakfast_box")
    ap.add_argument("--stage", default="smoke", choices=["smoke", "pilot", "full"])
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-experiments", type=int, default=3)
    ap.add_argument("--prompt", default="Propose three distinct falsifiable methods for LOCO logical anomaly localization without memory banks; focus on good-only learning and safe development validation.")
    args = ap.parse_args(argv)
    store = StateStore()
    runtime = Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()
    if args.command == "init":
        outcome = {"db": str(store.path), "runtime": str(runtime), "status": "initialized"}
    elif args.command == "status":
        outcome = {"paused": store.is_paused(), "gpu_hours_recorded": store.total_gpu_seconds() / 3600, "runs": store.list_runs(25), "tracking_pending": len(store.pending_tracking(10000))}
    elif args.command in ("pause", "resume"):
        store.set_paused(args.command == "pause")
        outcome = {"paused": store.is_paused(), "note": "Active jobs are not killed by pause; no new jobs will start."}
    elif args.command == "reconcile":
        from orchestrator.gpu_queue import GpuBusy
        try:
            with gpu_lock(runtime):
                outcome = {"interrupted": store.reconcile_after_restart()}
        except GpuBusy:
            outcome = {"interrupted": [], "note": "GPU worker lock held; no reconciliation performed"}
    elif args.command == "scout":
        enforce(store)
        outcome = {"literature_leads": scout_literature(store), "warning": "Unverified literature summaries; human/source cross-check required"}
    elif args.command == "propose":
        enforce(store)
        outcome = {"proposals": add_research_proposals(store, args.prompt)}
    elif args.command == "implement":
        enforce(store)
        if not args.hypothesis:
            ap.error("--hypothesis is required")
        outcome = implement_proposal(store, args.hypothesis)
    elif args.command == "queue-candidate":
        if not (args.manifest and args.hypothesis and args.candidate):
            ap.error("queue-candidate needs --manifest, --hypothesis, --candidate")
        if args.stage == "full" and os.getenv("LOCO_APPROVE_FULL") != "1":
            raise PermissionError("Full experimental training requires explicit operator approval")
        outcome = {"run_id": queue_candidate(store, args.hypothesis, args.candidate, args.category, args.stage, args.device, args.manifest, args.timeout), "status": "QUEUED"}
    elif args.command == "campaign":
        outcome = run_research_campaign(store, args.manifest, args.device, max(1, min(args.max_experiments, 20)))
    elif args.command == "queue-toy":
        if not args.manifest:
            ap.error("--manifest is required")
        from benchmark.data_guard import load_manifest, manifest_fingerprint
        manifest, _ = load_manifest(args.manifest)
        run_id = store.create_run("fixture-demo", "toy_normal_ae", "fixture", "smoke", {"device": args.device, "manifest": str(Path(args.manifest).resolve()), "epochs": 1, "size": 64}, manifest_sha256=manifest_fingerprint(manifest))
        for s in ("IMPLEMENTED", "VALIDATED", "QUEUED"):
            store.transition(run_id, s)
        outcome = {"run_id": run_id, "status": "QUEUED"}
    elif args.command == "worker-once":
        if not args.manifest:
            ap.error("--manifest is required")
        outcome = run_worker_once(store, args.manifest, runtime)
    elif args.command == "review":
        from orchestrator.decision_engine import eligible
        if not args.run:
            ap.error("--run is required")
        run = store.get_run(args.run)
        if run is None:
            raise KeyError(args.run)
        ok, rationale = eligible(run)
        if run["status"] == "COMPLETED":
            store.transition(args.run, "REVIEWED")
            store.transition(args.run, "PROMOTED" if ok else "REJECTED", reason=rationale)
            from orchestrator.mlflow_tracker import enqueue_run_tracking
            enqueue_run_tracking(store, args.run)
        outcome = {"run_id": args.run, "eligible": ok, "reason": rationale, "status": store.get_run(args.run)["status"]}
    elif args.command == "cycle":
        # Operator-triggered, bounded autonomous *research* step. Never accesses sealed data.
        enforce(store)
        existing = list_proposals(store)
        if not existing:
            scout = scout_literature(store)
            outcome = {"stage": "proposals", "scout": scout, "proposals": add_research_proposals(store, args.prompt + " Unverified literature scout leads: " + json.dumps(scout))}
        else:
            outcome = {"stage": "pending_operator_validation", "proposals_count": len(existing), "next": "controller implement --hypothesis ID; validate candidate then enqueue using approved runner"}
    print(json.dumps(outcome, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
