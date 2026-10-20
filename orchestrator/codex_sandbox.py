"""Narrow personal-server Bubblewrap workspace for Codex agent calls.

This is mount isolation with a shared host UID, not a separate Linux identity.
Codex needs network for its API; candidate training uses the separate no-network
boundary in orchestrator.sandbox and never receives this authentication mount.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil


def command(repo_root: Path, role: str, schema: Path, output: Path, task: str,
            candidate_dir: Path | None = None, reference_files=None) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("Bubblewrap is required for personal-server Codex calls")
    repo = Path(repo_root).resolve(strict=True)
    schema = Path(schema).resolve(strict=True)
    output = Path(output).resolve()
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.is_file():
        raise RuntimeError("Codex authentication file is unavailable")
    node = shutil.which("node")
    codex = shutil.which("codex")
    if not node or not codex:
        raise RuntimeError("Codex CLI and Node.js are required")
    node_root = Path(node).resolve().parents[1]
    codex_js = Path(codex).resolve()
    if not codex_js.is_relative_to(node_root):
        raise RuntimeError("Codex CLI must be installed under the selected Node.js prefix")
    args = [bwrap, "--unshare-all", "--share-net", "--die-with-parent",
            "--new-session", "--clearenv", "--setenv", "HOME", "/home/agent",
            "--setenv", "CODEX_HOME", "/home/agent/.codex", "--setenv", "PATH", "/opt/node/bin:/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin", "--dir", "/etc",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/home", "--dir", "/home/agent", "--dir", "/home/agent/.codex",
            "--dir", "/opt", "--dir", "/workspace", "--dir", "/workspace/agents",
            "--dir", "/workspace/benchmark", "--dir", "/workspace/scripts", "--dir", "/workspace/candidates",
            "--dir", "/workspace/literature", "--dir", "/output",
            "--ro-bind", node_root, "/opt/node",
            "--ro-bind", auth, "/home/agent/.codex/auth.json",
            "--bind", output.parent, "/output",
            "--ro-bind", schema, "/workspace/schema.json"]
    for filename in ('resolv.conf', 'hosts', 'nsswitch.conf'):
        source = Path('/etc') / filename
        if source.is_file():
            args += ['--ro-bind', source.resolve(strict=True), '/etc/' + filename]
    if Path('/etc/ssl').is_dir():
        args += ['--ro-bind', '/etc/ssl', '/etc/ssl']
    for filename in ("AGENTS.md", "RESEARCH_CHARTER.md", "research_program.md"):
        args += ["--ro-bind", repo / filename, "/workspace/" + filename]
    for filename in ("train_adapter.py", "validation.py"):
        args += ["--ro-bind", repo / "benchmark" / filename, "/workspace/benchmark/" + filename]
    args += ['--ro-bind', repo / 'scripts' / 'run_candidate.py', '/workspace/scripts/run_candidate.py']
    args += ['--ro-bind', repo / 'schemas' / 'experiment_plan_agent.schema.json', '/workspace/experiment_plan.schema.json']
    if reference_files:
        import re
        allowed = Path(os.environ.get('LOCO_LEDGER', str(Path(os.environ.get('LOCO_RUNTIME', '.runtime'))/'controller/experiments.sqlite'))).resolve().parent/'research_evidence'
        args += ['--dir','/workspace/evidence']
        directories = set()
        for alias, source in reference_files.items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+/(model.py|configuration.json|evidence.json)',alias):
                raise PermissionError('unapproved research evidence alias')
            source = Path(source).resolve(strict=True)
            if not source.is_relative_to(allowed) or not source.is_file():
                raise PermissionError('research evidence must be a controller-owned snapshot')
            group = alias.split('/')[0]
            if group not in directories:
                args += ['--dir','/workspace/evidence/'+group]; directories.add(group)
            args += ['--ro-bind',source,'/workspace/evidence/'+alias]
    args += ["--ro-bind", repo / "literature" / "evidence_index.json", "/workspace/literature/evidence_index.json"]
    role_file = repo / "agents" / f"{role}.md"
    args += ["--ro-bind", role_file, "/workspace/agents/" + role_file.name]
    if candidate_dir is not None:
        candidate = Path(candidate_dir).resolve(strict=True)
        if role != "implementer" or not candidate.is_relative_to(repo / "candidates" / "worktrees"):
            raise PermissionError("Codex writable workspace must be an assigned candidate directory")
        if candidate.name != candidate.parent.parent.name or candidate.parent.name != 'candidates':
            raise ValueError("candidate workspace ID mismatch")
        args += ["--dir", "/workspace/candidates/" + candidate.name,
                 "--bind", candidate, "/workspace/candidates/" + candidate.name]
    elif role == "implementer":
        raise ValueError("implementer requires assigned candidate directory")
    args += ["--chdir", "/workspace", "--", "/opt/node/bin/node",
             "/opt/node/lib/node_modules/@openai/codex/bin/codex.js", "exec",
             "--skip-git-repo-check", "--cd", "/workspace", "--sandbox",
             "workspace-write" if role == "implementer" else "read-only",
             "--output-schema", "/workspace/schema.json",
             "--output-last-message", "/output/" + output.name, '-']
    return [str(x) for x in args]
