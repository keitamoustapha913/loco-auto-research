import pytest
from benchmark.official_evaluator import validate_release_manifest, SealedEvaluationDenied

def test_human_sealed_evaluator_denied_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCO_SEALED_EVALUATOR', raising=False)
    checkpoint = tmp_path/'frozen.bin'
    checkpoint.write_bytes(b'test')
    with pytest.raises(SealedEvaluationDenied):
        validate_release_manifest(checkpoint, '0'*64)
