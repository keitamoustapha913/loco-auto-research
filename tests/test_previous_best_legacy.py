"""Legacy retention compatibility, with immutable artificial archived evidence."""
import copy
import json
import pytest
from orchestrator import model_registry as registry
from orchestrator.campaign_config import starting_context
from test_model_registry import lab as registry_lab
from test_time_campaigns import simple


def historical(registry_lab):
    store,c,cid,repo,add=registry_lab
    rid=add(seed=7);registry.initialize(store,repo)
    head=next(iter(registry.states(store).values()))
    model=copy.deepcopy(registry.retained(store)[head['retention_id']]);model.pop('category')
    with store.connect() as con:
        con.execute("UPDATE events SET payload_json=? WHERE kind='MODEL_RETAINED' AND run_id=?",(json.dumps(model),rid))
    return store,model


def test_legacy_category_resolved_without_rewriting_record(registry_lab):
    store,model=historical(registry_lab)
    before=registry.states(store);events=registry.records(store,'MODEL_RETAINED')
    context=starting_context(store,simple('previous_best'),strict=True)
    assert context['models'][0]['run_id']==model['run_id']
    assert context['models'][0]['reference']==model['reference']
    assert registry.states(store)==before and registry.records(store,'MODEL_RETAINED')==events
    assert 'category' not in registry.retained(store)[model['retention_id']]
    registry.verify_archive(model)


def test_new_does_not_reuse_legacy_incumbent(registry_lab):
    store,model=historical(registry_lab);before=registry.states(store)
    assert starting_context(store,simple('new'),strict=True)['models']==[]
    assert registry.states(store)==before
    registry.verify_archive(model)


@pytest.mark.parametrize('problem',['category','source_hash','checkpoint_hash','configuration_hash','missing_run','run_identity','manifest'])
def test_missing_or_mismatched_provenance_is_never_inferred(registry_lab,problem):
    store,model=historical(registry_lab)
    if problem=='category':model['category']='pushpins'
    elif problem=='source_hash':model['reference']['source_sha256']='0'*64
    elif problem=='checkpoint_hash':model['reference']['checkpoint_sha256']='0'*64
    elif problem=='configuration_hash':model['reference']['configuration_sha256']='0'*64
    elif problem=='run_identity':model['candidate_id']='unrelated'
    elif problem=='manifest':model['conditions']['training_manifest_sha256']='0'*64
    else:
        with store.connect() as con:con.execute('DELETE FROM runs WHERE run_id=?',(model['run_id'],))
    with pytest.raises(ValueError):registry.retained_category(store,model)


def test_actual_configuration_category_mismatch_is_rejected(registry_lab):
    store,model=historical(registry_lab)
    # An inconsistent authoritative record must not silently change cohort/dataset.
    with store.connect() as con:con.execute("UPDATE runs SET category='pushpins' WHERE run_id=?",(model['run_id'],))
    with pytest.raises(ValueError,match='No eligible retained'):
        starting_context(store,simple('previous_best'),strict=True)
