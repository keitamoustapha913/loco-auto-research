"""Repair depends on failure origin, not an enumeration of known messages."""
from pathlib import Path
import sqlite3
import pytest
from orchestrator import controller
from orchestrator.controller_errors import CandidateValidationError
from orchestrator.codex_runner import AgentFailure, InvalidAgentOutput
from orchestrator.budgets import CodexBudgetExhausted
from orchestrator.state_store import StateStore


class PreviouslyUnseenValidationError(Exception):
    pass


@pytest.mark.parametrize('error', [SyntaxError('unseen syntax issue'), TypeError('unexpected tensor contract'),
    KeyError('new required field'), AssertionError('unfamiliar invariant'), RuntimeError('new candidate diagnostic'),
    PreviouslyUnseenValidationError('novel diagnostic'), TimeoutError('candidate validation deadline')])
def test_unfamiliar_generated_work_failures_receive_context_and_one_correction(tmp_path,error):
    store=StateStore(tmp_path/'ledger.sqlite')
    prompts=[]
    def request(feedback,attempt):
        prompts.append(feedback)
        return {'generated':attempt}
    def validate(response):
        if response['generated']==0:
            # Isolated generated-code execution reports its ownership explicitly.
            try:raise error
            except Exception as exc:raise CandidateValidationError(str(exc)) from exc
        return 'accepted'
    response,valid,repaired=controller._with_agent_correction(store,None,'implementer',request,validate)
    assert valid=='accepted' and repaired and len(prompts)==2
    assert type(error).__name__ in prompts[1] and 'traceback' in prompts[1]
    events=store.events()
    assert len([e for e in events if e['kind']=='REPAIR_REQUESTED'])==1


@pytest.mark.parametrize('error', [PermissionError('workspace escape'), CodexBudgetExhausted('global limit'),
    sqlite3.OperationalError('ledger unavailable'), OSError('disk failure')])
def test_security_budget_and_host_failures_never_become_agent_repair(tmp_path,error):
    store=StateStore(tmp_path/'ledger.sqlite');calls=[]
    def request(feedback,attempt):calls.append(1);return {}
    def validate(response):raise error
    with pytest.raises(type(error)):
        controller._with_agent_correction(store,None,'implementer',request,validate)
    assert calls==[1] and store.events()==[]


def test_unfamiliar_validation_failure_stops_after_single_correction(tmp_path):
    store=StateStore(tmp_path/'ledger.sqlite');calls=[]
    def request(feedback,attempt):calls.append(1);return {}
    def validate(response):raise CandidateValidationError('still broken')
    with pytest.raises(CandidateValidationError):
        controller._with_agent_correction(store,None,'researcher',request,validate)
    assert len(calls)==2


@pytest.mark.parametrize('role',['scout','critic'])
def test_malformed_agent_response_is_corrected_for_all_roles(tmp_path,monkeypatch,role):
    store=StateStore(tmp_path/'ledger.sqlite');prompts=[]
    def invoke(*a,**kw):
        prompts.append(a[3])
        if len(prompts)==1:raise InvalidAgentOutput('required field missing from structured response')
        return {'response':{'schema_version':'1.0'}}
    monkeypatch.setattr(controller,'invoke',invoke)
    assert controller.budgeted_invoke(store,role,Path('.'),None,'fixture')['response']['schema_version']=='1.0'
    assert len(prompts)==2 and 'required field missing' in prompts[1]
    assert store.codex_call_count()==2


def test_cli_infrastructure_failure_is_not_an_output_correction(tmp_path,monkeypatch):
    store=StateStore(tmp_path/'ledger.sqlite');calls=[]
    def invoke(*a,**kw):calls.append(1);raise AgentFailure('CLI unavailable')
    monkeypatch.setattr(controller,'invoke',invoke)
    with pytest.raises(AgentFailure):controller.budgeted_invoke(store,'critic',Path('.'),None,'fixture')
    assert len(calls)==1 and store.codex_call_count()==1
