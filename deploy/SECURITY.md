# Deployment and actual security boundary

**Mandatory warning:** Candidate code is untrusted. `AGENTS.md`, `git worktree`, `realpath` checks, a Python subprocess or `codex --sandbox` flag by themselves **do not protect** a readable LOCO test set against malicious/inadvertent Python code. Run full research only after applying and testing OS-enforced isolation.

## Recommended identities
- `loco-operator`: owns policies and sealed final-evaluation approval. Not used for autonomous execution.
- `loco-codex`: can read authorized good data and write only an isolated candidate worktree; cannot read `/srv/loco-sealed`, controller ledger, evaluator sources or MLflow credential files.
- `loco-worker`: reads authorized good/synthetic data, candidate checkpoint outputs and write-only run artifacts; cannot read `/srv/loco-sealed` or controller secrets.
- `loco-sealed-evaluator`: has separate OS identity and sole read access to `/srv/loco-sealed/official_test` and pinned final evaluator. Do not join it to Codex worker groups.

Use dedicated containers or mount namespaces plus restrictive Linux ACLs, independent Unix users and service confinement. Do not use broad group/world permissions. Secrets must be injected privately into controller processes, never candidate env. Training candidate dependencies must be approved/pinned; CPU/GPU resource usage requires cgroups or systemd property limits beyond the Python timeout.

## Fail-closed tests before unattended mode
1. From the *actual Codex identity*, try opening an official test image; confirm permission denied.
2. From the *actual candidate identity*, try opening ground-truth masks, controller DB, and role-policy files for writing; confirm permission denied.
3. Confirm candidate can open only its authorized good manifest and approved outputs.
4. Confirm symlink and `..` traversal cannot escape authorized mounts.
5. Confirm MLflow and Streamlit bind 127.0.0.1 and use SSH tunnels; do not publish ports 5000/8501 publicly.
6. Confirm candidate can't change evaluator code or commit policy edits to trusted branch.
7. Confirm controller/worker run in cgroups with memory/time/disk ceilings and only one GPU job.
8. Record `LOCO_SANDBOX_VERIFIED=1` only **after** the above tests pass; it is an operator assertion, not a substitute for confinement.

## Dataset policy
Do not commit, upload, or redistribute LOCO test imagery, annotations or derived pixel masks in the shared repository, MLflow, or dashboard. Only a separate offline/manual official evaluation process can read these files; its scores do not enter ongoing agent research memory. Follow dataset and third-party weight/license conditions.

## Known V1 limits
Filesystem manifest checks are defense in depth but do not contain arbitrary Python. A full containerized/rootless sandbox is environment specific and must be provisioned on your remote server. The provided service templates are **operator-reviewed examples**, not a claim that remote isolation has been deployed. `controller reconcile` marks orphaned RUNNING jobs INTERRUPTED and does not blindly retry them; OS process identity must be checked before manual release.
