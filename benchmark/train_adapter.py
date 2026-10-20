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
