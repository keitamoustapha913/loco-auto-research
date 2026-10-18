"""Bounded, structured Codex invocation. No researcher-authored shell commands run."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
from jsonschema import validate

class AgentFailure(RuntimeError):
    pass


def invoke(role: str, repo_root: Path, schema_path: Path, prompt: str, timeout=180, worktree=None):
    if role not in {"scout", "researcher", "implementer", "critic"}:
        raise ValueError("unknown agent role")
    root = Path(repo_root).resolve(strict=True)
    cwd = Path(worktree).resolve(strict=True) if worktree else root
    if role == "implementer":
        operator_uid = root.stat().st_uid
        if os.getenv('LOCO_CODEX_ISOLATION_VERIFIED') != '1' or os.geteuid() == operator_uid:
            raise AgentFailure('Implementer Codex execution requires a separate verified OS identity and workspace mounts')
        if worktree is None or not cwd.is_relative_to((root / "candidates").resolve()):
            raise AgentFailure("implementer needs assigned candidates/ worktree")
    elif worktree is not None:
        raise AgentFailure("only implementer receives a writeable candidate worktree")
    role_text = (root / "agents" / f"{role}.md").read_text()
    charter = (root / "RESEARCH_CHARTER.md").read_text()
    task = f"{role_text}\n\nHUMAN RESEARCH CHARTER:\n{charter}\n\nTASK:\n{prompt}"
    schema = json.loads(Path(schema_path).read_text())
    with tempfile.TemporaryDirectory(prefix="codex-response-") as temp:
        response_file = Path(temp) / "response.json"
        command = ["codex", "exec", "--cd", str(cwd), "--sandbox", "workspace-write" if role == "implementer" else "read-only", "--output-schema", str(Path(schema_path).resolve()), "--output-last-message", str(response_file), task]
        env = {k: v for k, v in os.environ.items() if k not in ("LOCO_SEALED_ROOT", "MLFLOW_TRACKING_TOKEN", "GITHUB_TOKEN")}
        try:
            done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=min(int(timeout), 600), check=False)
        except FileNotFoundError as exc:
            raise AgentFailure("Codex CLI not installed on server") from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentFailure("Codex invocation timed out") from exc
        if done.returncode or not response_file.is_file():
            raise AgentFailure(f"Codex exited {done.returncode}: {done.stderr[-800:]}")
        try:
            output = json.loads(response_file.read_text())
            validate(instance=output, schema=schema)
        except (ValueError, Exception) as exc:
            raise AgentFailure("Codex produced invalid schema-constrained JSON") from exc
    return {"response": output, "prompt_sha256": hashlib.sha256(task.encode()).hexdigest(), "schema_sha256": hashlib.sha256(Path(schema_path).read_bytes()).hexdigest()}


def create_candidate_worktree(repo_root, hypothesis_id, base_ref="HEAD"):
    """Operator must first approve ID and use dedicated OS user for implementer."""
    root = Path(repo_root).resolve(strict=True)
    if not hypothesis_id or not all(c.isascii() and (c.isalnum() or c in "_-") for c in hypothesis_id):
        raise ValueError("invalid hypothesis ID")
    destination = root / "candidates" / "worktrees" / hypothesis_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(destination), base_ref], check=True)
    return destination
