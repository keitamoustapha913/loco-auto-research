"""No live agent calls: large whole-history prompts travel over stdin."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.codex_runner import invoke,provider_output_schema,InvalidAgentOutput


@pytest.mark.parametrize('personal_mode',[False,True])
def test_large_prompt_uses_stdin_without_command_argument_expansion(tmp_path,monkeypatch,personal_mode):
    import orchestrator.codex_runner as runner
    import orchestrator.codex_sandbox as sandbox
    repo=Path(__file__).resolve().parents[1]
    schema=tmp_path/'schema.json';schema.write_text(json.dumps({'type':'object'}))
    captured={};output={}
    def sandbox_command(root,role,schema,response,task,candidate,**kw):
        output['path']=response
        return ['bwrap','codex','exec','-']
    monkeypatch.setattr(sandbox,'command',sandbox_command)
    def run(command,**kwargs):
        captured.update(command=command,**kwargs)
        response=output['path'] if personal_mode else Path(command[command.index('--output-last-message')+1])
        response.write_text('{}')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(runner.subprocess,'run',run)
    result=invoke('researcher',repo,schema,'evidence '*40000,personal_mode=personal_mode)
    assert captured['command'][-1]=='-'
    assert len(captured['input'])>131072
    assert max(map(len,captured['command']))<131072
    assert result['response']=={}


def test_detailed_personal_history_is_available_as_files_not_duplicated_in_prompt(tmp_path,monkeypatch):
    from orchestrator import controller,experiment_plan,model_registry
    from orchestrator.state_store import StateStore
    store=StateStore(tmp_path/'ledger.sqlite');captured={}
    detail='DETAILED_HISTORY_SENTINEL'*10000
    monkeypatch.setattr(model_registry,'initialize',lambda *a:None)
    monkeypatch.setattr(model_registry,'recover_comparisons',lambda *a:None)
    monkeypatch.setattr(model_registry,'scientific_memory',lambda *a:{'cohorts':[],'complete_history':detail})
    item={'run_id':'old_run','hypothesis_id':'old_idea','candidate_id':'old_model','status':'REVIEWED',
          'git_sha':'a'*40,'error':'','comparison':{'status':'COMPLETE_EXPLORATORY'},'large_detail':detail}
    monkeypatch.setattr(experiment_plan,'research_evidence',lambda *a:([item],{}))
    def invoke(role,repo,schema,prompt,**kw):captured.update(prompt=prompt,**kw);return {'response':{}}
    monkeypatch.setattr(controller,'invoke',invoke)
    controller._budgeted_invoke_once(store,'researcher',Path('.'),None,'fixture',personal_mode=True)
    assert len(captured['prompt'])<10000
    memory=captured['reference_files']['cumulative/evidence.json'].read_text()
    assert detail in memory and 'DETAILED_HISTORY_SENTINEL' not in captured['prompt']
    assert '/workspace/evidence/old_run/evidence.json' in captured['prompt']


def test_provider_schema_subset_preserves_original_and_literal_constraints():
    schema={'type':'object','properties':{'seeds':{'type':'array','uniqueItems':True,'items':{'type':'integer'}},
        'literal':{'const':{'uniqueItems':True}}},'required':['seeds']}
    compiled=provider_output_schema(schema)
    assert 'uniqueItems' not in compiled['properties']['seeds']
    assert schema['properties']['seeds']['uniqueItems'] is True
    assert compiled['properties']['literal']['const']=={'uniqueItems':True}


def test_provider_uniqueitems_omission_never_accepts_duplicate_seeds(tmp_path,monkeypatch):
    import orchestrator.codex_runner as runner
    repo=Path(__file__).resolve().parents[1]
    schema=tmp_path/'schema.json';schema.write_text(json.dumps({'type':'object','properties':{
        'seeds':{'type':'array','uniqueItems':True,'items':{'type':'integer'}}},'required':['seeds']}))
    def run(command,**kwargs):
        provider=Path(command[command.index('--output-schema')+1])
        assert 'uniqueItems' not in json.loads(provider.read_text())['properties']['seeds']
        Path(command[command.index('--output-last-message')+1]).write_text('{"seeds":[7,7]}')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(runner.subprocess,'run',run)
    with pytest.raises(InvalidAgentOutput,match='non-unique'):
        invoke('researcher',repo,schema,'fixture')
