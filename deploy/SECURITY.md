# Deployment and actual security boundary

**Mandatory warning:** Candidate code is untrusted. `AGENTS.md`, `git worktree`, `realpath` checks, a Python subprocess or `codex --sandbox` flag by themselves **do not protect** a readable LOCO test set against malicious/inadvertent Python code. Run full research only after applying and testing OS-enforced isolation.

## Current candidate boundary

The trusted runner can now launch candidates through Bubblewrap (`orchestrator/sandbox.py`). It uses Linux user, mount, PID and network namespaces. Only individual trusted development runner files, the selected candidate directory, an authorized good-image root, read-only configuration/manifest inputs, the Python environment, and a writable run output directory are mounted. The official evaluator module is not mounted. Controller secrets are removed from the subprocess environment; Bubblewrap clears all inherited environment variables. CUDA device nodes are selectively passed through for GPU jobs. The trusted controller still owns budget checks, the GPU lock and subprocess timeout. Set `LOCO_FORCE_SANDBOX_TOY=1` to exercise this boundary with the included toy reference candidate.

This does not authorize unattended research. A malicious Python candidate could still falsify its own predictions or measurement file inside the sandbox, and the Codex implementer identity has not been provisioned. The final evaluator's separate account and permissions must be independently verified by the operator without exposing official data to this research process.

## Recommended identities
- `loco-operator`: owns policies and sealed final-evaluation approval. Not used for autonomous execution.
- `loco-codex`: can read authorized good data and write only an isolated candidate worktree; cannot read `/srv/loco-sealed`, controller ledger, evaluator sources or MLflow credential files.
- `loco-worker`: reads authorized good/synthetic data, candidate checkpoint outputs and write-only run artifacts; cannot read `/srv/loco-sealed` or controller secrets.
- `loco-sealed-evaluator`: has separate OS identity and sole read access to `/srv/loco-sealed/official_test` and pinned final evaluator. Do not join it to Codex worker groups.

Use restrictive Linux ACLs and independent Unix users in addition to the candidate mount namespace. Do not use broad group/world permissions. Secrets must be injected privately into controller processes, never candidate env. Training candidate dependencies must be approved/pinned. Per-job cgroup and disk quotas remain a deployment task.

The following **system-level changes require operator approval and are not performed by the repository**. Adapt names and paths to the host before execution:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin loco-codex
sudo install -d -m 0700 -o loco-codex -g loco-codex /var/lib/loco-codex
# Provision a private Codex credential for that identity, a candidate-only
# workspace mount, and ACLs denying controller, evaluator, SSH and runtime data.
# Configure a bounded service/cgroup for implementer and candidate processes.
```

The implementer entry point now rejects execution under the operator repository owner's UID, even if `LOCO_CODEX_ISOLATION_VERIFIED=1` is set. A separate identity launcher and host ACL verification are still required before that flag can be meaningful.

## Fail-closed tests before unattended mode
1. From the *actual Codex identity*, try opening an official test image; confirm permission denied.
2. From the *actual candidate identity*, try opening ground-truth masks, controller DB, and role-policy files for writing; confirm permission denied.
3. Confirm candidate can open only its authorized good manifest and approved outputs.
4. Confirm symlink and `..` traversal cannot escape authorized mounts.
5. Confirm MLflow and Streamlit bind 127.0.0.1 and use SSH tunnels; do not publish ports 56000/8501 publicly.
6. Confirm candidate can't change evaluator code or commit policy edits to trusted branch.
7. Confirm controller/worker run in cgroups with memory/time/disk ceilings and only one GPU job.
8. Record `LOCO_SANDBOX_VERIFIED=1` only **after** the above tests pass; it is an operator assertion, not a substitute for confinement.

## Dataset policy
Do not commit, upload, or redistribute LOCO test imagery, annotations or derived pixel masks in the shared repository, MLflow, or dashboard. Only a separate offline/manual official evaluation process can read these files; its scores do not enter ongoing agent research memory. Follow dataset and third-party weight/license conditions.

## Known limits
Bubblewrap isolates candidate filesystem and network access on this Linux host; its availability and GPU pass-through were tested with the toy reference candidate. A separate Codex implementer identity, evaluator account review, and cgroup/disk ceilings are not provisioned. Do not set `LOCO_SANDBOX_VERIFIED=1` or `LOCO_ALLOW_UNATTENDED=1` yet. `controller reconcile` marks orphaned RUNNING jobs INTERRUPTED and does not blindly retry them; OS process identity must be checked before manual release.
