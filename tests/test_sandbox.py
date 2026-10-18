import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from orchestrator.sandbox import command

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / 'candidates' / 'toy_normal_ae' / 'model.py'


def test_sandbox_mount_allowlist(tmp_path):
    data = tmp_path / 'data'; output = tmp_path / 'output'
    data.mkdir(); output.mkdir()
    cmd = command(REPO, MODEL, data, output, ['-c', 'pass'])
    joined = ' '.join(cmd)
    assert '--unshare-all' in cmd and '--clearenv' in cmd
    assert str(REPO / 'benchmark' / 'official_evaluator.py') not in joined
    assert str(REPO / 'orchestrator') not in joined
    assert str(Path.home() / '.ssh') not in joined
    assert '--bind' in cmd and '/output' in cmd
    assert '--ro-bind' in cmd and '/data' in cmd


def test_candidate_outside_repository_rejected(tmp_path):
    model = tmp_path / 'model.py'; model.write_text('pass')
    data = tmp_path / 'data'; output = tmp_path / 'output'
    data.mkdir(); output.mkdir()
    with pytest.raises(ValueError):
        command(REPO, model, data, output, ['-c', 'pass'])


@pytest.mark.skipif(os.getenv('LOCO_TEST_BWRAP') != '1', reason='requires operator-authorized user namespaces')
def test_adversarial_mount_and_secret_denial(tmp_path):
    data = tmp_path / 'data'; output = tmp_path / 'output'
    data.mkdir(); output.mkdir()
    outside = tmp_path / 'outside.txt'; outside.write_text('dummy sentinel')
    script = '''import json, os, socket
from pathlib import Path
result = {}
result["outside_readable"] = Path("%s").exists()
result["controller_visible"] = Path("/lab/orchestrator/controller.py").exists()
result["evaluator_visible"] = Path("/lab/benchmark/official_evaluator.py").exists()
result["secret_passed"] = "OPENAI_API_KEY" in os.environ
try:
 Path("/lab/benchmark/data_guard.py").write_text("tamper")
 result["trusted_write"] = True
except OSError:
 result["trusted_write"] = False
sock = socket.socket(); sock.settimeout(1)
result["host_loopback_reachable"] = sock.connect_ex(("127.0.0.1", 56000)) == 0
Path("/output/allowed.txt").write_text("ok")
result["output_write"] = Path("/output/allowed.txt").read_text() == "ok"
print(json.dumps(result))''' % outside
    cmd = command(REPO, MODEL, data, output, ['-c', script])
    process = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             env={'OPENAI_API_KEY': 'fake-sentinel'})
    assert process.returncode == 0, process.stderr
    results = json.loads(process.stdout)
    assert results == {'outside_readable': False, 'controller_visible': False,
                       'evaluator_visible': False, 'secret_passed': False,
                       'trusted_write': False, 'host_loopback_reachable': False,
                       'output_write': True}
