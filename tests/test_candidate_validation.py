"""Candidate validation must execute tests, not merely collect skipped cases."""
import sys

import pytest

from orchestrator.experiment_runner import run_candidate_unit_tests


@pytest.mark.parametrize('source,accepted', [
    ('def test_ok(): assert True', True),
    ('import pytest\n@pytest.mark.skip(reason="CUDA absent")\ndef test_gpu(): pass', False),
    ('import pytest\npytest.skip("missing dependency", allow_module_level=True)', False),
    ('import pytest\n@pytest.mark.xfail\ndef test_expected(): assert False', False),
    ('# no tests', False),
    ('def test_bad(): assert False', False),
    ('import pytest\ndef test_ok(): pass\n@pytest.mark.skip(reason="optional")\ndef test_optional(): pass', True),
])
def test_cpu_gate_requires_successful_test_body(tmp_path, monkeypatch, source, accepted):
    model = tmp_path / 'model.py'
    model.write_text('')
    tests = tmp_path / 'test_model.py'
    tests.write_text(source)

    def local_fixture_command(repo, module, data, output, inner, use_gpu):
        assert use_gpu is False  # Validation cannot silently consume GPU budget.
        assert inner[0] == '-c'
        return [sys.executable, '-c', inner[1].replace('/candidate/test_model.py', str(tests))]

    monkeypatch.setattr('orchestrator.sandbox.command', local_fixture_command)
    args = (tmp_path, model, tmp_path, tmp_path / 'runtime', 'fixture')
    if accepted:
        assert run_candidate_unit_tests(*args)['tests'] == 'passed'
    else:
        with pytest.raises(ValueError, match='CPU unit tests failed'):
            run_candidate_unit_tests(*args)


def test_missing_required_suite_is_rejected(tmp_path):
    model = tmp_path / 'model.py'
    model.write_text('')
    with pytest.raises(ValueError, match='mandatory test_model.py'):
        run_candidate_unit_tests(tmp_path, model, tmp_path, tmp_path, 'fixture')
