# LOCO Autoresearch laboratory

A guarded, good-image-only research laboratory for compact logical anomaly localization. SQLite is the authoritative experiment ledger; MLflow stores experiment runs and artifacts. The four-page Streamlit dashboard shows live development state. Synthetic development scores are **proxies, never official LOCO results**.

## Daily operation

On the remote server, from this repository:

```bash
uv sync --extra all
uv run --extra all loco-lab start
uv run --extra all loco-lab status
uv run --extra all loco-lab doctor
uv run --extra all loco-lab logs
uv run --extra all loco-lab smoke --dataset juice_bottle --device cuda
uv run --extra all loco-lab stop
```

`start` launches MLflow, the dashboard, and the durable tracking worker on loopback. It is safe to repeat. `restart` restarts owned services. `stop` only terminates processes started by this command. The application does **not** launch experiments or unattended Codex work at startup. The smoke command is an explicit, bounded, operator-initiated test of the toy candidate, the guarded GPU runner, SQLite, and MLflow. Its measurements are infrastructure evidence only. Existing controller commands remain available for approved research workflows.

The CLI reads `../loco-research.env` (or `LOCO_ENV_FILE`) as literal `export NAME=value` assignments with simple variable references. It does not execute shell statements. Explicit environment variables take precedence. The existing runtime, ledger, good-data path, budget, and MLflow URI are preserved. The good-only manifest must already exist at `$LOCO_RUNTIME/good_manifest.json`. Keep the official anomalous LOCO test data and masks outside this account and runtime.

On your laptop, open one SSH tunnel:

```bash
ssh -N \
  -L 8501:127.0.0.1:8501 \
  -L 56000:127.0.0.1:56000 \
  theyeq-admin-pc@100.91.233.99
```

The services run on the remote server; your browser displays them locally at [Dashboard](http://127.0.0.1:8501) and [MLflow](http://127.0.0.1:56000). Keep the tunnel open while browsing.

The dashboard has **Home** for laboratory and research status, **Research** for hypotheses and decisions, **Results** for models, heatmaps and comparisons, and **System** for service health and logs. Toy smoke runs are labeled infrastructure tests and excluded from research counts and Pareto plots. The runner currently saves one aligned development heatmap preview per run. Dashboard controls remain read-only; operator actions use `loco-lab`.

## Research and security boundaries

The research roles remain Literature Scout, Hypothesis Researcher, Implementation Agent, Research Critic, Deterministic Experiment Runner, and Decision Engine. The operator must verify actual OS-level isolation before allowing unattended candidate execution. `LOCO_SANDBOX_VERIFIED=1` and `LOCO_ALLOW_UNATTENDED=1` are operator assertions, not a sandbox. The browser is read-only; queue changes and smoke tests require the trusted CLI. Normal-only images and synthetic corruptions are used for development. No inference-time training-image memory bank is permitted. The sealed official test set remains inaccessible.

Untrusted candidate execution now uses a Bubblewrap mount and network namespace with read-only good data and trusted inputs, a candidate-only code mount, and a writable run output mount. The included toy candidate can be tested through this path with `LOCO_FORCE_SANDBOX_TOY=1 loco-lab smoke --dataset juice_bottle --device cuda`. Unattended Codex implementation remains blocked pending a separately verified Linux identity, ACLs, and resource ceilings; see `deploy/SECURITY.md`.

MLflow synchronization is automatic and retryable. A terminal run state and its tracking event are committed together in SQLite. If MLflow is down, events remain pending with retry errors; after recovery the worker retries. Run UUIDs are mapped to MLflow run IDs, including crash recovery by run tag. See `deploy/OPERATIONS.md` and `deploy/SECURITY.md` for operational details.
