"""Tests may exercise local processes/Bubblewrap, never a live research agent."""
from pathlib import Path
import subprocess
import pytest


@pytest.fixture(autouse=True)
def no_live_codex(monkeypatch):
    original=subprocess.Popen
    def guarded(command,*args,**kwargs):
        tokens=command if isinstance(command,(list,tuple)) else [command]
        if any(Path(str(t)).name in {'codex','codex.js'} for t in tokens):
            raise AssertionError('Offline regression attempted a live Codex command; mock the invocation boundary')
        return original(command,*args,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',guarded)
