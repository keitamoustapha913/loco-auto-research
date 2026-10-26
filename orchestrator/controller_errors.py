"""Error ownership at trusted controller / generated-work boundaries."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import traceback


class CandidateValidationError(ValueError):
    """A known generated-output/candidate contract failure the agent can repair."""


class TrustedControllerError(RuntimeError):
    """Unknown trusted failure: never candidate repair input."""
    def __init__(self, detail):
        self.detail=detail
        self.operation=detail['operation']
        self.fatal=detail['fatal']
        super().__init__(detail['error'])


def source_version():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
        ('controller.py','controller_errors.py','experiment_runner.py','campaign_state.py','autonomy.py',
         'planning_contract.py','experiment_plan.py','codex_runner.py','../benchmark/train_adapter.py','../benchmark/synthetic_geometry.py'))).hexdigest()


def trusted_failure(exc, operation, *, fatal=True):
    frames=traceback.extract_tb(exc.__traceback__)
    # Exclude hypothesis IDs/messages so one deterministic bug is recognized
    # across candidates. Code version separates a reviewed software correction.
    location=[(Path(f.filename).name,f.name,f.lineno) for f in frames]
    identity={'type':type(exc).__name__,'location':location,'source_version':source_version()}
    return TrustedControllerError(dict(identity,operation=operation,fatal=fatal,
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
        error=f'{type(exc).__name__}: {exc}',traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))))


def candidate_check(function, *args, **kwargs):
    """Convert only explicitly owned adapter contract errors, never raw host errors."""
    from benchmark.train_adapter import CandidateContractError
    try:return function(*args,**kwargs)
    except CandidateContractError as exc:
        raise CandidateValidationError(str(exc)) from exc


def blocked_operation(campaign, operation):
    if not campaign:return None
    version=source_version()
    for event in campaign[0].events(campaign[1],100000):
        if event['kind']=='TRUSTED_INFRASTRUCTURE_ERROR':
            detail=json.loads(event['payload_json'])
            if detail.get('operation')==operation and detail.get('source_version')==version:
                return TrustedControllerError(detail)
    return None


def record_trusted_failure(store, campaign, failure, *, role=None, hypothesis_id=None):
    detail=dict(failure.detail,role=role,hypothesis_id=hypothesis_id,ownership='trusted_controller')
    # This record, including the full traceback, survives worker restart. It is
    # never returned to a candidate agent as a request to repair trusted files.
    if campaign:
        previous=blocked_operation(campaign,failure.operation)
        if previous:return
        campaign[0].event(campaign[1],'TRUSTED_INFRASTRUCTURE_ERROR',**detail)
    else:
        import time
        with store.connect() as con:
            con.execute('INSERT INTO events(time,kind,payload_json) VALUES(?,?,?)',
                        (time.time(),'TRUSTED_INFRASTRUCTURE_ERROR',json.dumps(detail)))
