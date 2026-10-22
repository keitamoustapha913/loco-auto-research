"""Method-agnostic candidate contract; candidate modules are not trusted code."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any, Protocol
import numpy as np

class Predictor(Protocol):
    def predict(self, images: Any) -> dict: ...

class CandidateContractError(ValueError):
    pass


DIAGNOSTIC_PREFIX = 'diagnostic/'


def diagnostic_measurement_catalog(diagnostics):
    """Declared canonical identifiers -> ledger keys; no inferred metric names."""
    catalog={};ids=set()
    for declaration in diagnostics:
        identifier=declaration['diagnostic_id']
        if identifier in ids:raise CandidateContractError('duplicate diagnostic identity: '+identifier)
        ids.add(identifier)
        for metric in declaration['required_metrics']:
            canonical=f'{identifier}/{metric}'
            if canonical in catalog:raise CandidateContractError('duplicate declared diagnostic metric: '+canonical)
            catalog[canonical]=DIAGNOSTIC_PREFIX+canonical
    for canonical,storage in catalog.items():
        if storage in catalog:
            raise CandidateContractError('ambiguous diagnostic measurement alias: '+storage)
    return catalog


def canonical_diagnostic_measurement(catalog, name):
    """Accept canonical or exactly one declared storage alias, never guess/strip."""
    matches=[key for key,storage in catalog.items() if name in (key,storage)]
    if len(matches)!=1:
        raise CandidateContractError('unknown diagnostic ID, undeclared metric or ambiguous alias: '+name)
    return matches[0]


class DecodedTrainingCache:
    """Per-training decoded pixels only; never attach this cache to a predictor.

    The caller's reader and source ordering are unchanged. The trusted runner has
    already verified the good-only manifest and mounted its data read-only. Every
    entry is immutable and bounded; stacking a batch creates separate writable data.
    """
    def __init__(self, max_bytes=128 * 2**20):
        from collections import OrderedDict
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError('positive decoded-cache byte capacity required')
        self.capacity=max_bytes;self.bytes=0;self.entries=OrderedDict()
        self.hits=0;self.misses=0

    def get(self, key, reader):
        import numpy as np
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        value=np.asarray(reader())
        if value.dtype.hasobject:raise ValueError('decoded cache requires numeric pixels')
        # Immutable bytes, not just a resettable NumPy writeable flag.
        frozen=np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)
        self.misses+=1
        if frozen.nbytes<=self.capacity:
            while self.bytes+frozen.nbytes>self.capacity:
                _,old=self.entries.popitem(last=False);self.bytes-=old.nbytes
            self.entries[key]=frozen;self.bytes+=frozen.nbytes
        return frozen


def validate_diagnostic_output(declaration, result, source_ids, paired_metrics=()):
    """One payload contract for cheap preflight, production and paired analysis."""
    import math
    if not isinstance(result,dict) or not isinstance(result.get('metrics'),dict):
        raise CandidateContractError('diagnostic must return a metrics dictionary')
    values=result['metrics'];declared=set(declaration['required_metrics'])
    if set(values)!=declared:
        raise CandidateContractError(f"diagnostic {declaration['diagnostic_id']}: missing {sorted(declared-set(values))}; undeclared {sorted(set(values)-declared)}")
    def numeric(value):
        return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
    if not all(numeric(v) for v in values.values()):raise CandidateContractError('diagnostic metrics must be finite measured numbers')
    records=result.get('per_image')
    if paired_metrics and records is None:raise CandidateContractError('paired diagnostic requires per_image records')
    if records is not None:
        if not isinstance(records,list) or not all(isinstance(r,dict) for r in records):raise CandidateContractError('invalid diagnostic per_image records')
        ids=[r.get('source_id') for r in records]
        if not all(isinstance(i,str) for i in ids) or len(ids)!=len(source_ids) or len(set(ids))!=len(ids) or set(ids)!=set(source_ids):raise CandidateContractError('diagnostic per-image source identities differ')
        for record in records:
            if not isinstance(record.get('metrics'),dict):raise CandidateContractError('diagnostic per_image requires nested metrics dictionary')
            if set(paired_metrics)-record['metrics'].keys():raise CandidateContractError('paired diagnostic omitted a requested per-image metric')
            if not all(numeric(v) for v in record['metrics'].values()):raise CandidateContractError('invalid per-image diagnostic metric')
    return result


def load_predictor(module, checkpoint, device, inference_config=None):
    """Never silently discard an explicitly requested prediction intervention."""
    if inference_config:
        import inspect
        if 'inference_config' not in inspect.signature(module.load).parameters:
            raise CandidateContractError('requested inference settings require load(..., inference_config=...)')
        predictor = module.load(checkpoint, device=device, inference_config=dict(inference_config))
        if getattr(predictor, 'inference_config', None) != inference_config:
            raise CandidateContractError('predictor did not acknowledge the exact inference configuration')
        return predictor
    return module.load(checkpoint, device=device)


def load_candidate(module_file: str | Path):
    """Load candidate only from trusted, isolated subprocesses, NEVER the controller."""
    module_file = Path(module_file).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("loco_candidate", module_file)
    if spec is None or spec.loader is None:
        raise CandidateContractError("cannot load candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(callable(getattr(module, name, None)) for name in ("train", "load")):
        raise CandidateContractError("candidate must implement train and load")
    return module


def validate_prediction(result: dict, batch_shape: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, dict) or "anomaly_maps" not in result or "image_scores" not in result:
        raise CandidateContractError("missing anomaly_maps or image_scores")
    maps = result["anomaly_maps"]
    scores = result["image_scores"]
    if hasattr(maps, "detach"):
        maps = maps.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    maps = np.asarray(maps)
    scores = np.asarray(scores)
    b, c, h, w = batch_shape
    if b == 0 or c != 3:
        raise CandidateContractError("RGB input batch must be nonempty")
    if maps.shape != (b, 1, h, w) or scores.shape != (b,):
        raise CandidateContractError(f"bad output shape: maps={maps.shape} scores={scores.shape} expected={(b, 1, h, w)} {(b,)}")
    if not np.issubdtype(maps.dtype, np.floating) or not np.issubdtype(scores.dtype, np.floating):
        raise CandidateContractError("outputs must be floats")
    if not np.isfinite(maps).all() or not np.isfinite(scores).all():
        raise CandidateContractError("nonfinite anomaly output")
    return maps, scores
