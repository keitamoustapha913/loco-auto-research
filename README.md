# LOCO Autoresearch laboratory

A guarded, normal-only learning research laboratory for compact logical anomaly localization. SQLite is the authoritative experiment ledger; MLflow stores experiment runs and artifacts. The four-page Streamlit dashboard shows live development state. Synthetic development scores are **proxies, never official LOCO results**.

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

The CLI reads `../loco-research.env` (or `LOCO_ENV_FILE`) as literal `export NAME=value` assignments with simple variable references. It does not execute shell statements. Explicit environment variables take precedence. The existing runtime, ledger, good-data path, budget, and MLflow URI are preserved. The good-only manifest must already exist at `$LOCO_RUNTIME/good_manifest.json`. Keep the final holdout outside this account and runtime. The separately approved custom development partition is inference-only; see the real-development feedback section below.

## Autonomous campaign

The persistent research controller uses the same `loco-lab` entry point:

```bash
uv run --extra all loco-lab research start
uv run --extra all loco-lab research status
uv run --extra all loco-lab research pause
uv run --extra all loco-lab research resume
uv run --extra all loco-lab research report
uv run --extra all loco-lab research stop
```

The human-owned `configs/autonomy.yaml` sets the campaign and per-campaign ceilings. It defaults to **not authorized** for unattended personal-server execution. Starting it requires an operator to review the shared-UID risk in `deploy/SECURITY.md` and explicitly set `security.operator_accepts_shared_uid_risk: true` in that file. This authorizes only the limits in that configuration; increasing global limits still requires operator action. The application startup command never starts a campaign. A stopped campaign cannot resume; a crashed worker resumes the same campaign ledger state, marks any interrupted GPU job as interrupted, and never replays that job. Pause prevents new work; stop cancels the active job and queued campaign work. The dashboard remains read-only and shows actual campaign state from SQLite.

Agent calls run in a Bubblewrap mount namespace with only the charter, agent instructions, selected benchmark contract files, and the assigned candidate directory visible. Codex API access requires a shared network namespace; the candidate GPU process runs in the existing separate, no-network Bubblewrap sandbox and never receives Codex authentication. This is a **personal-server shared-host-UID mode**, not verified multi-user OS isolation. The approved four-hour campaign is not started by installation or service startup. Final-holdout images, masks and evaluation remain outside the campaign. Explicitly configured real-development feedback uses only the pinned approved partition through a separate inference stage.

The trusted controller charges Codex calls before invocation and GPU time on success, failure, timeout, interruption, and cancellation. It stops at wall, GPU, experiment, call, or consecutive-failure limits. Agent output is schema-checked and used only as research proposals, implementation files in the assigned workspace, and critique; it cannot supply shell commands to the scheduler. Candidate dependencies and external weights require separate review. Synthetic development scores remain proxies; real-development scores are labeled separately as custom-split feedback.

The ledger's `gpu_seconds` is **exclusive GPU slot wall time**, charged even if a candidate uses CPU internally. Peak VRAM and CUDA allocation measurements are separate evidence of actual device use. A zero peak VRAM value must not be described as GPU computation.

For a CUDA research pilot, `research.fresh_hypotheses` makes Codex rank new proposals against measured prior runs and critic findings; `research.require_torch_cuda` rejects CPU-only pilot proposals before candidate implementation. The controller still limits each generated job to five minutes and verifies the candidate interface inside Bubblewrap. The first measured CUDA pilot is recorded in the ledger and MLflow as `cuda_masked_context_prediction` (research run `b0817b1f-d185-4827-8779-a69252e7b7e9`). Its synthetic scores are development proxies, and its critic marked the hypothesis inconclusive pending a matched local-context ablation and artifact-matched controls. The default campaign authorization remains disabled.

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

Untrusted candidate execution uses a Bubblewrap mount and network namespace with read-only good data and trusted inputs, a candidate-only code mount, and a writable run output mount. The included toy candidate can be tested through this path with `LOCO_FORCE_SANDBOX_TOY=1 loco-lab smoke --dataset juice_bottle --device cuda`. The separate-identity design and residual personal-server risks are documented in `deploy/SECURITY.md`.

MLflow synchronization is automatic and retryable. A terminal run state and its tracking event are committed together in SQLite. If MLflow is down, events remain pending with retry errors; after recovery the worker retries. Run UUIDs are mapped to MLflow run IDs, including crash recovery by run tag. See `deploy/OPERATIONS.md` and `deploy/SECURITY.md` for operational details.

## Real-anomaly development feedback

The approved partition is imported separately from normal learning data:

```bash
uv run --extra all loco-lab feedback import
uv run --extra all loco-lab feedback status
```

Both transferred files must already exist in `$LOCO_RUNTIME/incoming/juice_bottle_dev_v1/`.
Archive and canonical manifest hashes are pinned in `configs/development_feedback.json`.
The importer rejects other splits, extra members, unsafe paths and invalid mask associations.
No dataset download or final-holdout access is implemented.

For an explicitly authorized, bounded CPU evaluation of an existing frozen run:

```bash
uv run --extra all loco-lab feedback evaluate \
  --run-id <frozen-run-id> --timeout-seconds 600 --accept-shared-uid-risk
```

A fresh no-network Bubblewrap worker sees one image, read-only checkpoint/source, and its output directory.
Annotation masks and other evaluation images are absent. Learning inputs remain `train/good`;
normal-only calibration is reused unchanged. Checkpoint and inference-state integrity are checked.
The shared-host-UID risk remains; inference mode and optimizer guards are defense in depth,
not proof against deliberately malicious Python code. See [the integration report](results/reports/real_development_feedback.md).

**Current scope:** logical and structural image AUROC, normal false-positive counts/rate,
CPU batch-one preprocessing/prediction/resize latency, parameter count and process RSS.
AU-sPRO and structural localization scoring are **unavailable**, because the official reference
archive returned HTTP 404. Pixel AP is not substituted. Full-resolution raw maps are retained
for later versioned scoring once a verified reference is available.

`configs/autonomy_real_feedback.yaml` proposes a two-experiment campaign (120 seconds per GPU job,
four GPU-slot minutes, one wall hour, eight Codex calls), with authorization **disabled**.
After separate approval, the operator prepares a private mode-0600 copy with the approved risk setting.
The exact launch command for that approved copy is:

```bash
uv run --extra all loco-lab research start \
  --config "$HOME/workspace-research/loco-runtime/authorizations/real_feedback.yaml"
```

With `research.real_development_feedback: true`, the existing controller evaluates each completed
candidate before critique, queues MLflow delivery automatically, and supplies aggregate evidence
to researcher/critic calls. No per-experiment approval or manual synchronization is needed inside
an authorized campaign. Feedback failure counts against the existing consecutive-failure limit;
the original training run is retained. Interrupted inference is retained for operator review and
never silently replayed. MLflow and Results distinguish custom-split feedback from synthetic metrics.

Every score is labeled **LOCO custom-split development feedback — not official full-test results.**
Repeated use adapts research to this development partition. The off-server final holdout must be
used only after final model selection, reported as a custom held-out split, and never fed back
into the same campaign. No campaign is authorized or launched by this integration.
