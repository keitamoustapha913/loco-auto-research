import json
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError
from orchestrator.decision_engine import pareto_frontier, eligible
from orchestrator.experiment_runner import safe_candidate_id, candidate_file
from orchestrator.codex_runner import invoke, AgentFailure


def mk_run(rid,q,ms):
    return {'run_id':rid, 'status':'COMPLETED', 'metrics_json':json.dumps({'normal/image_fpr_at_threshold':0.0,'dev_synthetic/pixel_auroc':q,'inference/e2e_ms_p95':ms})}

def test_pareto_quality_and_latency():
    a=mk_run('a',0.8,10)
    b=mk_run('b',0.9,8)
    c=mk_run('c',0.95,12)
    assert set(pareto_frontier([a,b,c])) == {'b','c'}
    assert eligible(a)[0]

def test_candidate_path_escape_denied():
    with pytest.raises(ValueError):
        safe_candidate_id('../bad')
    root=Path(__file__).resolve().parents[1]
    assert candidate_file(root,'toy_normal_ae').name == 'model.py'

def test_invalid_agent_role():
    with pytest.raises(ValueError):
        invoke('untrusted',Path.cwd(),Path('schemas/proposals.schema.json'),'hi')

def test_schema_requires_complete_research_evidence():
    schema = json.loads((Path(__file__).parents[1]/'schemas/proposals.schema.json').read_text())
    with pytest.raises(ValidationError):
        validate({'schema_version':'1.0','proposals':[{'hypothesis_id':'foo'}]}, schema)
