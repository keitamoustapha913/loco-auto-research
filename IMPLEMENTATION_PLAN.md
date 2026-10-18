# Autonomous Codex Research Laboratory for MVTec LOCO AD
## Implementation plan v2 — preserving the repository, Streamlit dashboard, and MLflow

**Status:** implementation specification (not a claim that components are deployed)  
**Target hardware:** laptop (control/inspection) + remote Linux AI server with 1 NVIDIA RTX 4090, 24 GB VRAM  
**Research mission:** autonomously discover and experimentally assess **compact, memory-bank-free methods for precise logical anomaly heatmap localization** on the five MVTec LOCO AD categories, without prescribing a network architecture in advance.  
**Build principle:** Codex proposes/implements/reviews; *trusted, deterministic software* controls data, execution, evaluation, budgets, and experiment promotion.

> **Critical product requirements:** Retain the requested repository tree; retain a **real, persistent MLflow tracking server** and a **working Streamlit dashboard**. Implement both before agent-driven experiments, not as an optional final polish phase.

---

## 0. Decisions and constraints

### Scientific rules

1. **Dataset:** MVTec LOCO AD: `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`.
2. **Permitted model-fitting data:** good images only; synthetic logical anomalies generated from permitted good images are permitted. Synthetic examples, pseudo-labels and external pretrained weights must have recorded provenance.
3. **Development:** good-image validation (normal false-positive/calibration metrics), synthetic development anomalies, held-out synthesis generators, normal-image perturbations and edited-but-still-good controls. Real LOCO anomalous test images/masks **must not be in the agent's accessible environment**.
4. **Sealed benchmark:** a separately permissioned final evaluator alone reads official anomalous LOCO test images/masks and computes official logical/structural metrics. Its metrics do **not** enter Codex's adaptive selection history for that benchmark campaign.
5. **Inference:** no dataset-derived per-image embedding bank, no nearest-neighbor retrieval against stored normal images, no training-image lookup. Learned network weights and compact parametric statistics are allowed with disclosed dependence on training data.
6. **Objectives:** prioritize accurate, meaningful **pixel-level logical anomaly localization**; also measure normal false-positive behavior, synthetic generalization, inference latency, peak VRAM, checkpoint/model size and stability. Do not optimize image AUROC alone.
7. **Scientific interpretation:** synthetic metrics are **proxy results**, never official LOCO scores. Do not assert improvement on real LOCO without a locked final evaluation.
8. **No forced model family:** Codex may investigate teacher–student, predictive, component-centric, generative, discriminative, and other methods, subject to the same interface and safety checks.
9. **License:** verify MVTec LOCO's CC BY-NC-SA 4.0 data conditions and license terms for imported papers, code and model weights; no automatic redistribution of dataset files.

### Engineering rules

- One RTX 4090 -> **one active training/benchmark GPU job** at a time, through an exclusive trusted worker.
- Run Codex as bounded, separate invocations with validated structured outputs; do not rely on one unbounded conversational session.
- A candidate can edit **only** an isolated worktree. Trusted evaluator, controls, experiment DB and policy files are outside its write permissions.
- All processes must have resource limits, pause/stop controls, execution timeouts, traceable run IDs and restart recovery.
- Keep all generated datasets, checkpoints, artifacts, logs, SQLite databases and credentials out of Git.
- Prefer simple single-host services (`systemd`, SQLite, MLflow, Streamlit); defer Kubernetes, Ray and distributed infrastructure.

### Initial engineering targets (hypotheses, not achieved measurements)

- Investigate candidates below **10 M inference-time parameters** and **20 ms batch-1 end-to-end inference** where feasible; calibrate targets after reproducing baselines and measuring actual input resolution.
- Begin with **5-minute smoke tests** and **15–30-minute pilots** where scientifically informative; longer experiments require budget authorization.
- Record p50/p95 (and optionally p99) latency after CUDA warm-up, **preprocess + GPU forward + anomaly map + postprocess**, input resolution, batch size, GPU model and total timing methodology.

---

## 1. Repository structure — preserved

Do **not** remove or rename any of the specified root files and directories. Add implementation helpers **inside** the existing folders; minimal `schemas/`, `tests/`, `configs/`, and `scripts/` are additive utilities, not replacements. If these already exist, reuse them.

```text
loco-autoresearch/
│
├── AGENTS.md                         # Durable rules for Codex and roles
├── RESEARCH_CHARTER.md               # Human-owned scientific constraints
├── research_program.md              # Research priorities, resource budget, gates
│
├── literature/
│   ├── papers/                       # Indexed paper notes and permitted PDFs
│   └── evidence_index.json           # References, code sources, claims, caveats
│
├── research_memory/
│   ├── hypotheses.md                 # Research ideas and their disposition
│   ├── discoveries.md                # Verified findings, not marketing claims
│   ├── failed_approaches.md          # Negative results and failure analyses
│   └── experiments.sqlite            # Controller's authoritative event ledger *
│
├── agents/
│   ├── scout.md                      # Literature search/review operating prompt
│   ├── researcher.md                 # Hypothesis and experiment design prompt
│   ├── implementer.md                # Worktree-limited coding prompt
│   └── critic.md                     # Scientific skepticism and review prompt
│
├── candidates/
│   └── [isolated candidate code]     # Registered isolated Git worktrees
│
├── benchmark/
│   ├── train_adapter.py              # Stable train/load/predict protocol
│   ├── validation.py                 # Approved development/robustness metrics
│   ├── inference_benchmark.py        # Deterministic latency, VRAM and size
│   ├── data_guard.py                 # Add: manifest/split/permission controls
│   ├── metrics.py                    # Add: validated metric primitives
│   └── fixtures/                     # Add: tiny synthetic unit-test fixtures
│
├── orchestrator/
│   ├── controller.py                 # Start/pause/resume; state machine
│   ├── gpu_queue.py                  # Exclusive single-GPU job ownership
│   ├── experiment_runner.py          # Bounded subprocess and artifact capture
│   ├── decision_engine.py            # Deterministic ranking/promote/reject
│   ├── state_store.py                # Add: durable DB transactions/events
│   ├── codex_runner.py               # Add: Codex JSON invocation and validation
│   ├── mlflow_tracker.py             # Add: MLflow instrumentation and recovery
│   ├── budgets.py                    # Add: compute/usage/disk budget enforcement
│   └── reporting.py                  # Add: daily summary compilation
│
├── results/
│   ├── metrics/                      # Human-readable exports, CSV/Parquet/JSON
│   ├── heatmaps/                     # Representative normal/synthetic examples
│   └── reports/                      # Daily progress and stage-gate reports
│
├── dashboard/
│   ├── app.py                        # Add: Streamlit entrypoint/navigation
│   ├── data_access.py                # Add: READ-ONLY ledger + MLflow API wrapper
│   ├── charts.py                     # Add: shared plots/metrics formatting
│   └── pages/                        # Add: actual multi-page dashboard views
│       ├── 01_Overview.py
│       ├── 02_Experiments.py
│       ├── 03_Heatmaps.py
│       ├── 04_Pareto.py
│       ├── 05_Research.py
│       └── 06_System.py
│
├── schemas/                          # Add: strict versioned JSON contracts
├── configs/                          # Add: non-secret service/expt templates
├── tests/                            # Add: unit/integration/security tests
└── scripts/                          # Preserve existing CLI/demo scripts
```

`*` The **real** `experiments.sqlite` must be Git-ignored. Keep a placeholder or schema/migration scripts tracked. Likewise, MLflow metadata and artifacts live in runtime storage, not Git.

**Preserve existing project assets:** audit/reuse any established `data/`, `runs/`, `reports/generated/`, normal-only dataset preparation, GPU demo, checksums and test suite. Do not duplicate an already working run ledger just to satisfy a diagram. Any existing run IDs must remain resolvable.

### Runtime storage, explicitly outside tracked code

```text
/srv/loco-runtime/                     # Configurable location, not hardcoded
├── good_data/                         # Normal train/validation images
├── synthetic_dev/                     # Generated development/robustness data
├── jobs/                              # Per-run stdout/stderr and manifests
├── checkpoints/                       # Frozen trained candidate outputs
├── controller/experiments.sqlite      # Operational ledger (durable)
├── mlflow/
│   ├── mlflow.db                      # MLflow tracking backend DB
│   └── artifacts/                    # MLflow run artifacts
└── exports/                           # CSV/reports/dashboard thumbnails

/srv/loco-sealed/                       # Different OS identity/permissions
├── official_test/                     # Official good + anomaly test data/masks
└── official_evaluator/                # Pinned source + executable
```

The actual paths may differ. **The sealed directory must not be readable by Codex, candidate processes or the autonomous controller**. A `.gitignore`, a role prompt or dashboard filter alone is not a security boundary.

---

## 2. Component ownership: avoid two competing sources of truth

| Component | Owns | Does not own |
|---|---|---|
| `experiments.sqlite` operational ledger | hypotheses, queue, state transitions, leases, permissions, budgets, decisions, event history, cross-system IDs | Numerical metric truth or large files |
| **MLflow** | run metrics, tags, parameters, model references, heatmaps, plots, performance profiles, benchmark artifacts | job scheduling, research permissions, budget enforcement |
| **Streamlit dashboard** | read-only presentation from ledger and MLflow, comparisons, anomalies, audit drill-down | mutation of truth, running arbitrary Python, accessing sealed LOCO test files |
| Candidate worktrees | method implementation, training/inference adapter code and model-specific configs | the trusted evaluator, policy or tracking DB |
| Research-memory Markdown | concise human/agent-readable validated evidence and negative results | raw authoritative metrics, unrestricted access to official test results |
| Sealed evaluator | frozen final-test metrics in separate access domain | feeding real LOCO test scores into ongoing adaptive search |

**Linking key:** Every run has a stable `run_id` stored in the SQLite ledger and tagged as `research.run_id` in MLflow. The ledger stores the `mlflow_run_id`. MLflow may be temporarily unavailable; runs then queue tracking events in an outbox and replay them idempotently. **Do not write the same metrics independently from multiple locations.** The validated runner generates measurement records; MLflow ingests them; the dashboard reads them.

---

## 3. MLflow — required from the first real experiment

### 3.1 Deployment

- On the server, run an MLflow Tracking Server using **SQLite for tracking metadata** and **local persistent artifact storage**. A single RTX 4090 host does not require PostgreSQL initially.
- Pin the MLflow Python version in the dependency lockfile. Keep its DB distinct from the controller's operational SQLite DB.
- Bind to `127.0.0.1` and access the UI from the laptop via SSH tunnel or a private, authenticated network. Do not expose a raw MLflow port publicly.
- Configure and test DB backups, artifacts backups and migrations before unattended operation; verify restore using an isolated throwaway server.
- Track workspaces/experiment naming by campaign and category as required, **but do not fragment one comparable run across unrelated experiments**.

Illustrative service command (paths need to exist and permissions must be configured):

```bash
mlflow server \
  --backend-store-uri sqlite:////srv/loco-runtime/mlflow/mlflow.db \
  --artifacts-destination /srv/loco-runtime/mlflow/artifacts \
  --host 127.0.0.1 \
  --port 5000

# In the training worker service environment:
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The exact invocation should be smoke-tested against the pinned MLflow version. If changing artifact serving modes, create new experiments as needed rather than silently breaking existing artifact URIs.

### 3.2 Required parameters, tags, metrics and artifacts

**Tags/identity (every run):** `research.run_id`, `research.hypothesis_id`, `research.candidate_id`, `research.campaign_id`, `research.parent_run_id`, `git.commit`, `protocol.version`, `dataset.manifest_sha256`, `candidate.code_sha256`, `dependency.lock_sha256`, `data.allowed_splits`, `run.stage`, `category`, `seed`, `gpu.name`, `synthetic.generator_version`, `train.external_pretraining`.

**Parameters (selected):** input shape, image resize policy, training step/image budget, optimizer settings, batch size, feature extraction mode, synthetic generator configuration (when used), declared model-specific configuration, confidence calibration method.

**Metrics (use strict prefixes):**

| Namespace | Examples | Visibility |
|---|---|---|
| `train/*` | `loss`, `steps`, `images_seen`, `gpu_seconds` | Agents + dashboard |
| `normal/*` | `image_fpr_at_threshold`, `pixel_fpr`, `score_quantile` | Agents + dashboard |
| `dev_synthetic/*` | `localization_auroc`, `spro_proxy`, `image_auroc` | Agents + dashboard, visibly marked *synthetic* |
| `robustness/*` | `cross_generator_gap`, `edited_good_fpr`, `nuisance_fpr` | Agents + dashboard |
| `inference/*` | `e2e_ms_p50`, `e2e_ms_p95`, `model_ms_p50`, `peak_vram_mib`, `checkpoint_mib`, `parameters` | Agents + dashboard |
| `budget/*` | `gpu_hours_cumulative`, `codex_calls`, `disk_gib` | Ops + dashboard |
| `official_test/*` | `logical_spro`, `structural_spro`, `logical_image_auroc` | **Sealed evaluator / separate report only**; never development experiment namespace |

**Artifacts:** approved thumbnails/overlay/heatmap triplets; `run_manifest.json`; `candidate_config.yaml`; stdout/stderr excerpts; test results; latency profile; selected checkpoint manifest and hashes; per-category synthetic analyses; critic report; dependency snapshot; lineage links. Never upload raw official test images/masks to shared MLflow.

**Run lifecycle:** create run -> mark stage -> log checked metrics and artifacts -> finish with success/failed/killed status -> reconcile to ledger. OOMs and invalid candidates must still have traceable run records where possible.

### 3.3 MLflow acceptance tests

- A toy CPU run and bounded GPU run appear in the MLflow UI with an identical stable `run_id` tag to the operational ledger.
- Charts show training metrics; artifacts show heatmap examples and JSON manifests.
- Changing a candidate's Git commit or dataset manifest changes the logged hashes.
- Killing an experiment marks the ledger and MLflow appropriately; no completed run is mislabeled as successful.
- Stop the MLflow server during a run: tracking events buffer durably and are reconciled exactly once when it returns.
- No official LOCO anomaly label or pixel mask is present in agent-accessible tags, filenames or artifacts.

---

## 4. Streamlit research dashboard — required MVP before Codex autonomy

**Technology:** Streamlit multipage app, `dashboard/app.py` + `dashboard/pages/`, with `st.navigation`/`st.Page` for page routing. The app **must not own** a job scheduler or evaluation logic.

**Source of data:** read-only SQLite connection to the controller ledger (use immutable snapshots or a read-only DB URI), MLflow Tracking API, and explicitly allowlisted pre-generated thumbnail/report paths. Read-only dashboard service account; no GPU required.

### 4.1 Pages and minimum features

| Page | Required information | V1 behavior |
|---|---|---|
| **Overview** | current experiment, controller state, latest finding, candidates queued/running/failed, GPU-hour and token/credit budgets, 24-hour timeline | Refreshes from committed state; distinguishes simulated vs measured |
| **Experiments** | filter by category, hypothesis, stage, outcome, seed, candidate family, run ID; metric table; links to MLflow run | Compare 2+ runs with metrics and provenance; sortable table |
| **Heatmaps** | side-by-side input, prediction map and overlay; separate normal and permitted synthetic cases; category selector; pixel alignment note | Click from a run to its approved example artifacts; never expose sealed examples |
| **Pareto** | quality proxies vs p95 latency, size, peak VRAM; dominated/promoted/rejected labels | Interactive scatter with run ID, stage, and caveats; no test data |
| **Research** | literature sources, hypothesis graph/list, failures, critic assessments, next planned experiment and rationale | Trace a proposal through run IDs to evidence and decision |
| **System** | GPU queue, last heartbeat, worker error and restart count, disk use, tracking availability, budgets, last backup | Diagnostics only; mutation through protected CLI, not agent-facing dashboard |

### 4.2 Mandatory dashboard rules

- **Never label synthetic proxy sPRO as `LOCO logical sPRO`**. Use explicit labels and help text (e.g. `Held-out synthetic localization proxy`).
- For missing-object cases, show annotation/attribution assumptions in the report rather than implying unique ground-truth placement.
- Every plot/filter selection must provide the contributing run IDs and provenance link.
- Handle empty/new installations cleanly; show useful state before any models are trained.
- If MLflow is down, show cached/ledger status and a tracking-degraded warning, never fabricate results.
- Do not make Streamlit actions write to `experiments.sqlite`, launch candidate jobs, edit policy or reveal official test files.
- Operate on loopback/private network, with SSH tunnel; no public unauthenticated dashboard.

### 4.3 Dashboard acceptance tests

- With fixture runs, all six pages render without accessing real datasets.
- One selected run links to the correct MLflow run and downloads only allowlisted artifacts.
- Runs missing metrics display `N/A`, not zero.
- The dashboard visibly differentiates queued, running, completed, failed and promoted states.
- Rendering a heatmap does not ever attempt to open the sealed test directory.
- Dashboard stays readable on the laptop at ordinary browser sizes and loads normally with 100+ logged runs (pagination/cache where needed).

---

## 5. Stable method-agnostic experiment contracts

No hypothesis-specific logic should enter the trusted evaluation code. Candidate model creators must implement the same interface:

```python
# Conceptual protocol: implement and validate via typed adapters.
train(config, allowed_train_manifest, output_dir) -> checkpoint_manifest
load(checkpoint_manifest, device) -> predictor
predict(images: BxCxHxW) -> {
    "anomaly_maps": FloatTensor[B, 1, H, W], # finite, aligned, higher=more abnormal
    "image_scores": FloatTensor[B],            # declared aggregation or direct
    "metadata": { ... }
}
```

- Fixed I/O conventions: RGB order, transforms, input resize policy, spatial alignment, anomaly-score orientation and output pixel coordinate system.
- Seed control, dataset manifests, external pretraining declaration, exact checkpoint inference dependencies, no training-image feature lookup.
- Candidate cannot replace trusted `benchmark/validation.py`, `benchmark/metrics.py`, or `benchmark/inference_benchmark.py` during its experiment.
- Unit fixtures test NaN/Inf maps, shape mismatches, constant scores, empty scores, double normalization, CUDA device naming (`cuda` vs `cuda:0`), zero-batch errors, mask misalignment and invalid FPR calibration.
- Compare matched training compute/image exposures or disclose an explicitly different compute budget. Reject attempts to change the evaluation protocol as part of a method improvement.

### Minimum machine-readable records

**Proposal JSON:** `hypothesis_id`, `research_question`, `claim`, `references`, `expected_observation`, `controlled_change`, `evaluation_plan`, `estimated_gpu_minutes`, `risk_and_shortcuts`, `new_dependencies`, `candidate_family`, `schema_version`.

**Run JSON:** `run_id`, `mlflow_run_id`, `hypothesis_id`, `candidate_id`, `parent_run_id`, `git_sha`, `dataset_manifest_sha256`, `config_sha256`, `dependency_sha256`, `stage`, `status`, `category`, `seed`, `times`, `gpu_seconds`, `exit_code`, `artifact_hashes`.

**Review JSON:** `run_id`, `objective_measured_results`, `critic_concerns`, `suggested_falsification`, `confidence`, `recommended_next_action`; controller validates and treats free-text as advice, not authority.

Preserve these contracts in `schemas/` and version them. Never execute a free-form shell command extracted directly from a proposal. Use an allowlisted runner entrypoint and vetted argument schema.

---

## 6. Phased implementation with deliverables and exit gates

### P0 — Audit and preserve working infrastructure

**Deliver:** `results/reports/p0_inventory.md`, current-tests report, mapping of pre-existing data processing, GPU demo, runs ledger and hash-verification code to the new modules. Verify `nvidia-smi`, PyTorch CUDA and bounded GPU demo. Review `.gitignore` and existing commits before editing.

**Do not:** replace the normal-only pipeline, delete local run artifacts, rename folders for aesthetics, or treat prior dummy model latency as anomaly-detection performance.

**Gate:** existing tests pass, no tracked images/checkpoints/secrets, original run IDs/checksums verifiable and baseline commit identified.

### P1 — Research governance and actual filesystem isolation

**Deliver:** `AGENTS.md`, `RESEARCH_CHARTER.md`, `research_program.md`, `benchmark/data_guard.py`, split/manifest validator, negative-access tests. Establish OS identities or equivalent mount/container separation for `good_data` vs `loco-sealed`. Document permitted external pretrained weights and data sources.

**Gate:** malicious test candidate *cannot read* an official test image or label; paths/symlinks/manifest confusion rejected; human owns charter/policy updates.

### P2 — Stable model, validation and inference contracts

**Deliver:** `benchmark/train_adapter.py`, `validation.py`, `metrics.py`, `inference_benchmark.py`, fixtures and schema tests. Start with tiny deterministic predictors, not expensive real networks.

**Gate:** output-map spatial correctness and calibrated-good FPR tests pass; synthetic-proxy results clearly distinguishable from real LOCO metrics; batch-one latency methodology verified with CUDA synchronization.

### P3 — Persistent ledger, single GPU queue **and MLflow**

**Deliver:** `orchestrator/state_store.py`, `gpu_queue.py`, `experiment_runner.py`, `mlflow_tracker.py`, event/metric schemas, `systemd` service templates. Start the local MLflow server. Maintain stable run-ID mapping and a durable tracking outbox.

**State transitions:** `PROPOSED -> IMPLEMENTED -> VALIDATED -> QUEUED -> RUNNING -> COMPLETED -> REVIEWED -> PROMOTED/REJECTED` with `INVALID`, `FAILED`, `OOM`, `TIMED_OUT`, `CANCELLED` terminal error states.

**Gate:** two submitted GPU jobs never overlap; killing/restarting worker reconciles run state without duplicate execution; required params/metrics/artifacts appear in MLflow; outbox recovery verified; budget exhaustion prevents new launches.

### P4 — **Streamlit dashboard MVP** + one full baseline integration

**Deliver:** all six dashboard pages connected to fixture and real non-sealed experiments; links to MLflow run details. Register and reproduce one licensed, normal-only, memory-bank-free baseline candidate (choose on evidence; EfficientAD, CSAD and MuDeNet are *research references*, not mandatory architectures).

**Gate:** from laptop browser, view queue status, MLflow runs, heatmap overlays, synthetic proxy metrics, end-to-end latency and run provenance for one complete five-category baseline. Dashboard is usable **before** autonomous Codex research starts.

### P5 — Codex role framework and autonomous proposal/implementation

**Deliver:** `agents/scout.md`, `researcher.md`, `implementer.md`, `critic.md`; `orchestrator/codex_runner.py`, `controller.py`, `decision_engine.py`; strict proposal/review JSON schema and Git worktree isolation. Run bounded Codex CLI calls; store prompt/schema/response hashes and call usage when obtainable.

**Scout:** retrieve paper/source metadata and evidence cards; record uncertainty, dates, licensing, external code. **Researcher:** propose competing falsifiable hypotheses, approximate cost, expected failure signals. **Implementer:** touch candidate worktree only, add tests, expose required adapter. **Critic:** challenge methodology with measured outcomes, not editable metrics.

**Gate:** from a research question, Codex proposes **3 materially different hypotheses**, implements 3 valid candidate smoke tests and, budget permitting, pilot evidence for each; select/reject with traceable measured evidence and generate a new evidence-driven next hypothesis.

### P6 — Generalization, synthetic-shortcut controls and Pareto ranking

**Deliver:** independently implemented *development* vs *held-out robustness* synthetic generators; edited-but-logically-good controls; per-category localizations; revised Pareto rules and negative-result recording. Require ≥3 declared seeds for finalists where compute permits; acknowledge uncertainty for incomplete replication.

**Compare:** held-out synthetic localization, normal false positive rates, cross-generator gap, good-image edit sensitivity, p95 end-to-end latency, model size, VRAM, and reproducibility. Do **not** use locked official LOCO scores as an adaptive fitness signal.

**Gate:** agent rejects at least one impressive synthetic-only candidate because of edit-artifact sensitivity or held-out-generator failure; dashboard shows why it was rejected and retains its run.

### P7 — Bounded unattended operation + mature dashboard

**Deliver:** restricted `systemd` services, health monitor, budget/pause/stop CLI, automatic retry boundaries, nightly/periodic report, backup/restore scripts, dashboard operations page, MLflow artifact retention policy, alert on repeated failures. Keep the dashboard read-only; operator uses a separately protected CLI.

**Budgets:** per-job GPU minutes, maximum GPU hours per campaign, Codex call/usage budget, concurrent codex subprocess cap, disk quota, maximum consecutive errors, dependency/software-install approval. Idle or degraded components fail closed; no autonomous policy relaxation.

**Gate:** bounded unattended campaign survives service restarts and MLflow outage; no duplicate GPU job; stops at configured limits; publishes a daily summary and all run/MLflow references.

### P8 — Frozen independent LOCO final benchmark

**Deliver:** one or more pre-registered/frozen finalists, locked checkpoint/config/commit/environment checksums, separate evaluator account using pinned official LOCO evaluation scripts; category-level logical sPRO, structural sPRO, image-level AUROC where applicable, defect-type analysis, latency and memory details, uncertainty and limitations.

**Gate:** externally auditable benchmark report; no test-data exposure to research worker; scientific claims match the exact protocol. If results are later used to revise models, explicitly reclassify it as **test-informed development** and obtain a fresh untouched independent evaluation source for new final claims.

### Dependency order

```text
P0 existing repository audit
   -> P1 actual data isolation
   -> P2 invariant candidate/evaluator contracts
   -> P3 MLflow + ledger + one-GPU scheduler
   -> P4 usable Streamlit dashboard + first full baseline
   -> P5 autonomous Codex research loop
   -> P6 scientific robustness and Pareto promotion
   -> P7 bounded unattended operation
   -> P8 sealed official evaluation
```

**Strict development rule:** Do not defer P3 MLflow or P4 dashboard until after autonomy is working. They are foundational acceptance gates.

---

## 7. Trusted scheduling and recovery behavior

1. Controller selects eligible proposals by schema-validity, literature evidence, cost and information gain.
2. Creates one isolated Git worktree, pins commit/lockfile and creates a `PROPOSED` ledger event.
3. Worktree implementer produces a candidate + tests. Trusted runner validates protocol/split permissions.
4. GPU worker acquires OS `flock` **and** a durable lease; only then starts run under resource-controlled subprocess group.
5. Runner writes `run_manifest.json`, stdout/stderr and metric payloads to a temporary run directory, then atomically finalizes artifacts/hashes.
6. MLflow tracker logs payload and artifact refs with the same `run_id`; event outbox buffers tracking outages.
7. Trusted controller computes promotion/rejection from validated metrics and Pareto rules. The critic can propose follow-ups but cannot alter metrics or policy.
8. Controller writes a human-readable scientific lesson to Markdown research memory with MLflow/ledger links.
9. Upon reboot, reconcile subprocess PID/process-group and lease state; classify orphaned jobs conservatively and **never blindly retrain** a previously running run.

**Security:** A file lock alone does not prevent untrusted Python from evading process restrictions. Use OS identity / container / resource isolation to enforce the GPU-job boundary and prevent candidate subprocesses from touching controller secrets or benchmark directories. Separate literature network access from offline training execution.

---

## 8. Dashboard and MLflow demonstration: exact end-to-end acceptance test

A minimal but realistic, authorized experiment must demonstrate:

1. A candidate is registered in `candidates/` with hypothesis ID and immutable code/config hashes.
2. The worker queues a run, starts it on the RTX 4090 and creates an MLflow run automatically.
3. MLflow shows loss history, normal-FPR/calibration, held-out synthetic proxy metrics, GPU timing, model size and artifact preview.
4. The dashboard **Overview** displays running/completed state, GPU minutes and latest experiment decision.
5. **Experiments** offers a link to MLflow, run ID and provenance; **Heatmaps** renders normal and synthetic predictions; **Pareto** compares quality against speed/size.
6. Failure is injected (killed worker or temporary MLflow outage). Run state reconciles correctly and no run is silently duplicated.
7. `test` data access from candidate process fails at the filesystem boundary; the dashboard cannot display official test imagery.
8. Restart restores queue, MLflow references and dashboard views.

If the result isn't observable in both MLflow and Streamlit, **P4 is not complete**.

---

## 9. Suggested milestone artifacts and test matrix

| Milestone | Artifacts | Automated tests |
|---|---|---|
| P0 | Inventory, working baseline record | Existing regression suite and GPU smoke |
| P1 | Research policies + manifests | dataset split, symlink/path escape, forbidden-test-access tests |
| P2 | Adapter, metric fixtures, inference profiler | output shape, NaN, maps, CUDA sync, normal calibration |
| P3 | One-GPU queue + MLflow server | queue contention, kill, retry, outbox, MLflow linking |
| P4 | Six dashboard pages + first baseline | Streamlit smoke, read-only data access, correct MLflow links, artifact rules |
| P5 | Four agent roles + Codex controller | JSON validation, worktree confinement, mock calls, bounded 3-hypothesis demo |
| P6 | Robustness panel + promotion | synthesis leakage controls, deterministic Pareto selection, reproducibility |
| P7 | Continuous services | restart recovery, time-budget stop, disk quota, report generation, backup restore |
| P8 | Locked final report | evaluator checksum, data firewall, metric fixtures, frozen release manifest |

### Performance and quality policy

The dashboard should include **scientific caveats** beside every development metric. Scoreboards must not sort a synthetic proxy as if it is official benchmark ground truth. Official benchmark scores stay outside the adaptive agent's accessible history. Only authorized final reports may show those results, and those reports should not be ingested by an ongoing autonomous campaign against the same fixed test set.

---

## 10. First three Codex work packages — ready to paste

### Work package 1 — P0/P1 only

> Audit the existing `loco-autoresearch` repository. Do not rebuild or overwrite the established normal-only dataset pipeline, GPU demo, experiment ledger, hashes, tests or history. Keep all specified repository folders. Read this implementation plan, then complete P0 and P1 only: inventory, `AGENTS.md`, `RESEARCH_CHARTER.md`, `research_program.md`, authorized dataset manifests, and tests proving the official test set is inaccessible to candidate processes. Run the existing and new tests. Do not train new architectures. Stop at the gate and report changed files, tests and residual risks.

### Work package 2 — P2/P3, including **real MLflow**

> Implement and test the method-agnostic train/load/predict adapter, good-only validation, synthetic-only development metrics, honest batch-one GPU timing, schema-validated run payloads, durable SQLite ledger, single RTX 4090 job queue and MLflow Tracking Server integration. Every run must have linked SQLite and MLflow IDs; failures and GPU metrics must be logged; tracking outages must reconcile. Do not implement autonomous paper-search or new model architectures. Include demonstrations and tests for P2/P3 gates.

### Work package 3 — P4, including **working Streamlit dashboard**

> Implement all six Streamlit dashboard pages inside `dashboard/` and connect them read-only to the experiment ledger and MLflow. Include experiment comparisons, synthetic-only heatmaps, Pareto plots, research history, live job state and GPU budgets. Reuse a valid normal-only baseline to produce one end-to-end five-category demonstration and show its linked MLflow run and dashboard pages. Do not expose official test data or make dashboard controls modify research state. Stop once P4 criteria pass.

**Only after these three work packages pass** should Codex implement P5 autonomous literature search, hypothesis generation and self-directed experiments.

---

## 11. Operator access from the laptop

Illustrative SSH tunnels to private server services:

```bash
ssh -N \
  -L 5000:127.0.0.1:5000 \
  -L 8501:127.0.0.1:8501 \
  youruser@your-server
```

- MLflow UI: `http://127.0.0.1:5000`
- Research dashboard: `http://127.0.0.1:8501`
- Codex operates on server/worktrees through its own restricted identity; the operator runs protected start/pause/resume/stop commands via SSH. Neither web UI is the security boundary itself.

Recommended independent service names: `loco-mlflow.service`, `loco-dashboard.service`, `loco-worker.service`, `loco-controller.service`. Use `After=`/`Requires=` relationships where appropriate, read-only/limited filesystem mounts, restart backoff, restrictive umask and centralized journals.

---

## 12. First autonomous-research milestone: definition of success

A bounded demonstration succeeds if **all** are true:

- Three distinct scientific hypotheses were independently generated and linked to actual references and anticipated failure modes.
- Three candidate worktrees passed validation/smoke tests; pilots ran serially under a fixed total GPU-hour budget.
- Every run has a durable ledger ID, MLflow run, timing/VRAM metrics, artifact hashes and accessible dashboard page.
- Codex retains negative results and derives a new hypothesis from observed evidence rather than repeating the starting list.
- At least one convincing synthetic shortcut case is identified or explicitly tested, with a documented negative result if none is found.
- The data firewall remains effective; official LOCO test masks never reach the agent, dashboard, MLflow or candidate worktrees.
- The full campaign can stop, restart and recover without duplicate GPU training or lost decisions.
- A researcher can inspect everything from the laptop, including heatmaps, Pareto tradeoffs, experimental lineage and the next proposed research direction.

**Final principle:** MLflow is the experimental evidence store, Streamlit is the scientific control-room display, SQLite is the operational controller, and Codex is the researcher. None may silently replace the others.

---

## References / implementation sources (verify pinned versions during installation)

- MVTec LOCO AD and official evaluation: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- Karpathy's autoresearch project: https://github.com/karpathy/autoresearch
- OpenAI Codex iterative programmatic execution example: https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex
- MLflow tracking-server deployment: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- MLflow artifact-store configuration: https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/
- Streamlit multipage navigation: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation

*This is a staged build plan, not an assertion that the services, agents or baseline methods are currently deployed on the user's server.*
