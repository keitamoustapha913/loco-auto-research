# LOCO Autoresearch — guarded autonomous research lab

**A functioning, testable research-lab codebase** for normal-image-only MVTec LOCO AD research on one RTX 4090. The goal is to discover compact, low-latency, memory-bank-free anomaly localization methods, **not to prescribe an architecture**. Four Codex role prompts, a durable controller, single-GPU worker, MLflow experiment tracking and six-page Streamlit dashboard are included.

**Current scope / truthful status:** A tested CPU toy autoencoder and synthetic proxy demonstration is included. It is *not* an official LOCO baseline or published benchmark result. Real LOCO reproduction, final sealed evaluation, trained models and remote deployment require your dataset/compute and operator setup. The service templates and Codex integrations are included but are not activated automatically. This repository cannot magically establish an OS-level sandbox: you must provision separate Linux identities or containers as described in `docs/security.md` (see `deploy/SECURITY.md`).

## Quick start on Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
python -m pytest -q
python -m scripts.doctor
export LOCO_RUNTIME="$PWD/.runtime"   # local-only demonstration
export LOCO_LEDGER="$LOCO_RUNTIME/controller/experiments.sqlite"
python -m scripts.demo_run
```

To view the real MLflow UI and dashboard (in separate shells):

```bash
bash scripts/start_mlflow.sh
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

The demo uses **only manufactured geometric normal images** and creates artificial held-out corruption masks. Its metrics are explicitly synthetic development proxies. If MLflow is offline, events accumulate durably in the SQLite outbox; after bringing MLflow up, run `python -m scripts.sync_mlflow`.

From the laptop:

```bash
ssh -N -L 5000:127.0.0.1:5000 -L 8501:127.0.0.1:8501 user@your-server
```

Open `http://127.0.0.1:8501` for the six-page research dashboard, and `http://127.0.0.1:5000` for MLflow. Keep both services bound to loopback.

## Repository contract

The original requested directories and named files have been preserved. Helper files live alongside them, not instead of them. `research_memory/experiments.sqlite` is intentionally **created at runtime**, not checked into Git. Set `LOCO_LEDGER` or default to `${LOCO_RUNTIME}/controller/experiments.sqlite`.

- `AGENTS.md`, `RESEARCH_CHARTER.md`, `research_program.md`: policies/research freedom/budgets.
- `literature/`: source notes and evidence index; not an automatically vetted literature corpus.
- `agents/`: scout, researcher, implementer and critic role instructions.
- `candidates/`: method code; `toy_normal_ae` is an **illustration**, not a research claim.
- `benchmark/`: good-only manifest guard, stable candidate I/O, development-only metrics, true timing.
- `orchestrator/`: SQLite state, single-GPU locks, sandbox-conscious subprocess launching, Codex JSON interface, budgets, MLflow outbox.
- `dashboard/`: Streamlit six-page, read-only data access. No direct job submission from browser.
- `scripts/`: demo, manifest generation, worktree creation, MLflow sync, service launch.
- `deploy/`: service and security setup.

## Important commands

```bash
python -m scripts.make_manifest --root /srv/loco-runtime/good_data --output /srv/loco-runtime/good_manifest.json
python -m orchestrator.controller init
python -m orchestrator.controller status
python -m orchestrator.controller pause
python -m orchestrator.controller resume
python -m orchestrator.controller worker-once
python -m scripts.sync_mlflow
```

`controller scout` can request unverified literature leads, and `controller propose` can call the installed Codex CLI for a validated research proposal; **it never automatically authorizes execution**. `controller cycle` is an operator-triggered, budget-bounded research step. `controller campaign --manifest /srv/loco-runtime/good_manifest.json --device cuda --max-experiments 3` executes a guarded propose–implement–train–critic loop only after the operator has verified OS-level containment and enabled `LOCO_ALLOW_UNATTENDED=1` and `LOCO_SANDBOX_VERIFIED=1`. These flags do not themselves create a sandbox. See `deploy/OPERATIONS.md` for commands and requirements. Codex implementation is isolated to a candidate worktree by configuration and host access controls; its output is treated as untrusted.

## Scientific boundaries

- MVTec LOCO categories: `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`.
- Model fitting uses good images only. Synthetic anomalies derived from permitted good images are allowed, but **synthetic scores are not real LOCO logical sPRO**.
- Official LOCO test images/masks are kept on an independent, sealed account/storage domain and never enter the development MLflow or dashboard.
- No inference-time training-image feature bank or nearest-neighbor database. Check candidate architectures for violations before promotion.
- End-to-end latency includes preprocessing, inference and heatmap conversion. GPU timings require synchronization.

See `deploy/SECURITY.md` for the operator checklist, `deploy/OPERATIONS.md` for commands, and `docs/` intentionally not required. This is infrastructure for researching methods; no fabricated MVTec benchmark scores are supplied.

## Initialize Git after unpacking

Run `bash scripts/init_repository.sh`, inspect staged files, and commit before using worktrees (worktrees require an existing commit). The archive deliberately does not include `.git`, datasets, checkpoints, or runtime databases. **If you already have a working Git repository, merge these files carefully instead of running initialization or replacing your current code.**

## What's verified / not verified

A test suite includes integration of the toy candidate through the real trusted worker, SQL ledger, metrics, artifact previews and MLflow event outbox, plus unit/security tests. Streamlit and MLflow servers require optional dependencies and are not proven running on your remote host by these local tests. End-to-end *real* MVTec LOCO research, Codex-driven unbounded research, real RTX 4090 benchmarking and final official evaluation must be executed and reviewed by the operator; this source tree does not contain any real LOCO benchmark measurements.

**Do not run arbitrary third-party candidate code as your regular server user.** This source contains safety preflight checks but requires OS-level confinement on the remote GPU host.
