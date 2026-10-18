"""Bubblewrap mount boundary for untrusted candidate processes.

This launcher does not assert that the host's official evaluator permissions are
correct. Unattended research remains gated on a separate operator verification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

from benchmark.data_guard import load_manifest

BENCHMARK_FILES = ('__init__.py', 'data_guard.py', 'train_adapter.py', 'stage7_controls.py',
                   'metrics.py', 'metrics_v2.py', 'validation.py', 'inference_benchmark.py')


def available():
    return shutil.which('bwrap') is not None and sys.platform.startswith('linux')


def write_sandbox_manifest(source, output_dir):
    """Translate the verified good-only root to a read-only sandbox mount."""
    manifest, _ = load_manifest(source)
    sandbox_manifest = dict(manifest, root='/data')
    path = Path(output_dir) / 'sandbox_good_manifest.json'
    path.write_text(json.dumps(sandbox_manifest, sort_keys=True))
    return path, Path(manifest['root']).resolve(strict=True)


def command(repo_root, candidate_file, data_root, output_dir, inner_command, use_gpu=False, input_files=None):
    if not available():
        raise RuntimeError('Bubblewrap is required for untrusted candidate execution')
    repo = Path(repo_root).resolve(strict=True)
    model = Path(candidate_file).resolve(strict=True)
    data = Path(data_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    if not model.is_relative_to(repo / 'candidates'):
        raise ValueError('candidate must reside in candidates/')
    if not data.is_dir() or not output.is_dir():
        raise ValueError('sandbox data and output directories must exist')
    bwrap = shutil.which('bwrap')
    args = [bwrap, '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
            '--setenv', 'HOME', '/tmp', '--setenv', 'PYTHONPATH', '/lab',
            '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'XDG_CACHE_HOME', '/tmp',
            '--setenv', 'CUDA_CACHE_PATH', '/tmp', '--setenv', 'OMP_NUM_THREADS', '2',
            '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
            '--ro-bind', '/lib64', '/lib64', '--ro-bind', '/bin', '/bin',
            '--ro-bind', '/sys', '/sys', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp', '--dir', '/lab', '--dir', '/lab/benchmark',
            '--dir', '/lab/scripts', '--dir', '/candidate', '--dir', '/inputs',
            '--ro-bind', sys.prefix, '/opt/venv',
            '--ro-bind', sys.base_prefix, sys.base_prefix,
            '--ro-bind', data, '/data', '--bind', output, '/output',
            '--ro-bind', model.parent, '/candidate']
    for name in BENCHMARK_FILES:
        args += ['--ro-bind', repo / 'benchmark' / name, '/lab/benchmark/' + name]
    for name in ('__init__.py', 'run_candidate.py', 'stage7_eval.py', 'stage7_texture_eval.py'):
        args += ['--ro-bind', repo / 'scripts' / name, '/lab/scripts/' + name]
    args += ['--ro-bind', repo / 'pyproject.toml', '/lab/pyproject.toml']
    for name, path in (input_files or {}).items():
        if name not in {'config.json', 'good_manifest.json', 'checkpoint_manifest.json', 'model.pt'}:
            raise ValueError('unapproved sandbox input name')
        args += ['--ro-bind', Path(path).resolve(strict=True), '/inputs/' + name]
    if use_gpu:
        devices = sorted(Path('/dev').glob('nvidia*'))
        if not any(p.name == 'nvidia0' for p in devices):
            raise RuntimeError('NVIDIA GPU device unavailable to Bubblewrap')
        for device in devices:
            if device.is_file() or device.is_char_device():
                args += ['--dev-bind', device, str(device)]
    args += ['--chdir', '/output', '--', '/opt/venv/bin/python', *inner_command]
    return [str(arg) for arg in args]
