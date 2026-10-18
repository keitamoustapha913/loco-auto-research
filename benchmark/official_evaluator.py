"""Human-only sealed benchmark activation guard.

Not an implementation of official MVTec LOCO sPRO. Run the pinned official
scorer in a separate identity/container. No adaptive Codex feedback permitted.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

class SealedEvaluationDenied(PermissionError):
    pass


def validate_release_manifest(checkpoint: str | Path, expected_sha256: str):
    if os.getenv("LOCO_SEALED_EVALUATOR") != "1":
        raise SealedEvaluationDenied("Only the separately permissioned final-evaluator identity may run the official benchmark")
    path = Path(checkpoint).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("finalist checkpoint checksum mismatch")
    return {"checkpoint": str(path), "sha256": actual, "status": "FROZEN_READY_FOR_EXTERNAL_OFFICIAL_EVALUATOR"}
