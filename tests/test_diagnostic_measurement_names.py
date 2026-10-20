"""Deterministic naming/dispatch fixtures, never live agents or GPU experiments."""
import copy
import json

import pytest

from benchmark.train_adapter import (diagnostic_measurement_catalog,canonical_diagnostic_measurement,
                                     CandidateContractError)
from orchestrator.experiment_plan import (canonicalize_diagnostic_measurements,decode_agent_plan,
    validate_plan,execute_plan,digest,plan_history)
from orchestrator.planning_contract import inspect_plan
from plan_fixtures import experiment_plan,agent_plan
from test_experiment_plan import laboratory


def aliased():
    plan=experiment_plan()
    plan['required_measurements'][1]['key']='diagnostic/fixture_check/mean_score'
    return plan


def test_declarations_define_names_and_storage_prefix_exactly_once():
    p=experiment_plan();catalog=diagnostic_measurement_catalog(p['diagnostics'])
    assert catalog=={'fixture_check/mean_score':'diagnostic/fixture_check/mean_score'}
    for name in ('fixture_check/mean_score','diagnostic/fixture_check/mean_score'):
        assert canonical_diagnostic_measurement(catalog,name)=='fixture_check/mean_score'


def test_alias_canonicalization_is_immutable_idempotent_and_scientifically_identical():
    p=aliased();original=copy.deepcopy(p)
    canonical=canonicalize_diagnostic_measurements(p)
    assert canonical==experiment_plan()
    assert canonicalize_diagnostic_measurements(canonical)==canonical
    assert decode_agent_plan(agent_plan(p))==canonical
    assert validate_plan(p)==canonical
    report=inspect_plan(p)
    assert report['valid'] and report['canonical_plan']==canonical
    assert report['diagnostic_name_normalizations']==[{'source':'diagnostic',
        'submitted':'diagnostic/fixture_check/mean_score','canonical':'fixture_check/mean_score'}]
    assert p==original


@pytest.mark.parametrize('name',[
    'unknown/mean_score','fixture_check/unknown','diagnostic/unknown/mean_score',
    'diagnostic/fixture_check/unknown','diagnostic/diagnostic/fixture_check/mean_score',
    'mean_score','fixture_check/mean_score/unknown'])
def test_unknown_or_undeclared_aliases_never_get_guessed(name):
    p=aliased();p['required_measurements'][1]['key']=name
    with pytest.raises(CandidateContractError,match='unknown diagnostic ID, undeclared metric'):
        decode_agent_plan(p)
    assert not inspect_plan(p)['valid']


def test_ambiguous_prefix_aliases_are_rejected_from_declarations():
    p=experiment_plan();p['diagnostics'].append({'diagnostic_id':'diagnostic',
        'config':{},'required_metrics':['fixture_check/mean_score']})
    with pytest.raises(CandidateContractError,match='ambiguous diagnostic measurement alias'):
        canonicalize_diagnostic_measurements(p)


@pytest.mark.parametrize('duplicate', ['id','metric'])
def test_duplicate_declarations_are_rejected(duplicate):
    p=experiment_plan()
    if duplicate=='id':p['diagnostics'].append(copy.deepcopy(p['diagnostics'][0]))
    else:p['diagnostics'][0]['required_metrics'].append('mean_score')
    with pytest.raises(CandidateContractError,match='duplicate'):
        canonicalize_diagnostic_measurements(p)


@pytest.mark.parametrize('source,key',[
    ('diagnostic','fixture_check/mean_score'),
    ('diagnostic','diagnostic/fixture_check/mean_score'),
    ('run','diagnostic/fixture_check/mean_score')])
def test_duplicate_measurement_aliases_and_conflicting_sources_are_rejected(source,key):
    p=aliased();p['required_measurements'].append({'source':source,'key':key})
    with pytest.raises(ValueError,match='duplicate/conflicting measurement'):
        canonicalize_diagnostic_measurements(p)


def test_explicit_run_ledger_diagnostic_references_must_also_be_declared():
    p=experiment_plan();p['required_measurements'][1]={'source':'run','key':'diagnostic/fixture_check/mean_score'}
    assert canonicalize_diagnostic_measurements(p)==p # Existing explicit references retain identity.
    p['required_measurements'][1]['key']='diagnostic/fixture_check/invented'
    with pytest.raises(CandidateContractError,match='undeclared metric'):
        validate_plan(p)


def test_paired_comparisons_require_declared_metrics_not_just_diagnostic_id():
    p=experiment_plan();p['paired_comparisons']=[{'comparison_id':'missing','diagnostic_id':'fixture_check',
        'left_arm':'control','right_arm':'intervention','metric':'invented','bootstrap_seed':7,'repeats':100}]
    with pytest.raises(ValueError,match='paired comparison requests an undeclared diagnostic metric'):
        validate_plan(p)


def test_aliases_dispatch_same_complete_six_job_matrix_and_recover_without_duplicates(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory
    p=aliased();original=copy.deepcopy(p);canonical=canonicalize_diagnostic_measurements(p)
    result=execute_plan(store,c,cid,runtime,p)
    assert result['status']=='COMPLETE_CONTROLLED' and not result['problems']
    assert [(cfg['plan_arm'],cfg['seed']) for cfg in calls]==[(a,s) for a in ('control','intervention') for s in (3,17,29)]
    assert all(cfg['plan_sha256']==digest(canonical) for cfg in calls)
    assert plan_history(c,cid)[0]['plan']==canonical
    assert all(not cell['missing'] for cell in result['cells'])
    assert all('diagnostic:fixture_check/mean_score' in cell['measurements'] for cell in result['cells'])
    assert execute_plan(store,c,cid,runtime,canonical)==result
    assert len(calls)==6 and len(c.runs(cid))==6 and p==original


def test_bad_names_fail_before_any_job_or_plan_is_admitted(laboratory):
    store,c,cid,runtime,repo,calls,_=laboratory
    p=aliased();p['required_measurements'][1]['key']='diagnostic/fixture_check/invented'
    with pytest.raises(CandidateContractError):execute_plan(store,c,cid,runtime,p)
    assert calls==[] and not c.runs(cid) and not plan_history(c,cid)


def test_diagnostic_storage_namespace_cannot_be_mislabeled_as_real_feedback():
    p=aliased();p['required_measurements'][1]['source']='real'
    with pytest.raises(ValueError,match='conflicts with measurement source'):
        canonicalize_diagnostic_measurements(p)
