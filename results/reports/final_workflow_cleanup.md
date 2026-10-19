# Final research workflow cleanup

2026-10-19. No research training, host privilege changes, model architecture changes, or additional campaign authorization.

## Kept and consolidated

The existing Codex agents, adaptive campaign controller, single-GPU queue, guarded runner, PyTorch/CUDA, authoritative SQLite ledger, durable MLflow outbox and four-page Streamlit dashboard remain. All historical runs, metrics, checkpoints and artifacts remain in place.

Normal operation is exactly:

```bash
uv run --extra all loco-lab start
uv run --extra all loco-lab research start --config APPROVED_CONFIG
```

Status, pause, resume, report and stop remain under `loco-lab`. Approval is once per bounded campaign, not per experiment. Default authorization remains false. The default and real-feedback templates enable the trusted real evaluator, and disable synthetic diagnostics by default. `research.synthetic_diagnostics: true` opts into existing synthetic controls. Synthetic scores no longer gate eligibility for real-development research. The critic selects directions; no new scientific hypothesis or architecture was prescribed. Fixed literature leads were removed from the campaign prompt in favor of the charter, literature evidence and measured history.

One trusted `benchmark/feedback_metrics.py` now provides real-development detection/localization. `orchestrator/development_feedback.py` manages frozen inference and durable feedback. Existing normal calibration utilities and model-independent adapter are reused. Historical metrics v1/v2 and Stage 5–8 scripts remain because regression tests and frozen scientific protocols depend on them. They do not define the primary new-campaign ranking.

## Removed after dependency inspection

- `scripts/start_mlflow.sh`: obsolete conflicting port-5000 launcher.
- `scripts/init_repository.sh`: unnecessary repository initialization wrapper.
- `scripts/create_worktree.py`: standalone wrapper; controller still uses the existing worktree function.
- `scripts/sync_mlflow.py`: standalone wrapper; automatic tracking worker remains.
- `scripts/daily_report.py`: standalone scheduler wrapper; reporting functionality remains.
- `configs/research.yaml`: unconsumed duplicate policy/configuration with obsolete per-trial approval wording.
- Seven separate controller/dashboard/MLflow/report/worker service and timer templates. Neither system nor user systemd had installed LOCO units. No host units were modified. The optional unified `loco-lab.service` remains.
- Makefile manual service/worker targets; retained targets forward to the unified CLI.

Git retains all removed source. Unrelated pre-existing `.python-version` and `deploy/STAGE3_SECURITY_PROCEDURE.md` remain untracked and untouched.

## Repository responsibilities

```text
agents/, literature/, research_memory/  Scientific reasoning, evidence, history
candidates/                            Candidate source; historical checkpoints stay in runtime
benchmark/                             Data contract, normal calibration, versioned trusted scoring
orchestrator/                          Existing CLI, campaigns, queue, sandbox, ledger, tracking
scripts/                               Internal runners and retained reproducibility tools
configs/                               Campaign limits and pinned dataset/protocol definitions
dashboard/                             Home, Research, Results, System
results/                               Scientific reports and artifacts
deploy/systemd/loco-lab.service         Optional unified service template
tests/                                 Regression, evaluation and isolation tests
```

## Data and learning boundary

Verified development split `loco-feedback-v1.0-f91c7f27aad59de58c5a`: 165 images (47 good, 71 logical, 47 structural), 124 associated masks. Both transfer/archive integrity and split manifest remain pinned. Training receives only 335 normal training entries. The 54 normal validation images split deterministically into 18 calibration and 36 assessment images. No real anomaly or evaluation score fits weights, loss parameters, normalization, thresholds, a memory bank or test-time adaptation. Aggregate measurements can inform research selection.

Candidate inference sees one opaque image per fresh no-network Bubblewrap worker, no annotation masks, and read-only source/checkpoints. Prediction uses inference mode; standard optimizer steps are prohibited; checkpoint, parameter, buffer and wrapper-state integrity are checked. Trusted scoring alone reads masks. Final holdout stays on the laptop; no listing or manifest is requested. These protections do not prove safety against malicious Python that undoes in-process guards or exploit the documented shared host UID; no stronger isolation is claimed.

## Evaluator definition and provenance

Version: `real-development-v2.0-compatible`.

The current [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) still links an evaluation archive returning HTTP 404. Its exact URL remains in `configs/development_feedback.json`. No original archive was obtained and no original-code hash/equivalence is claimed.

The **compatible development implementation** follows [Bergmann et al., §§3.3–3.4](https://doi.org/10.1007/s11263-022-01578-9): average `min(detected region pixels / saturation pixels, 1)` over annotated regions; integrate against background-pixel FPR through 0.05, normalized by 0.05. Logical and structural subsets each include all development good images. Region identity is mask-file plus defect value; disconnected parts are retained together. Relative saturation scales by that region's size; absolute saturation uses public defect metadata, bounded by region area. Only background exclusion and supplementary pixel diagnostics use mask union. Scores tied at a threshold enter together. Integration interpolates the 5% boundary. Analytical fixtures and an independent direct-overlap oracle test these semantics; original-code numerical equivalence remains unverified.

| Required measurement | Implementation |
|---|---|
| Logical/structural image AUROC | Separate groups, same good-image controls, model-map maximum score |
| Normal image FPR | Strict score > frozen normal threshold; count, denominator, threshold |
| Logical/structural AU-sPRO@5% | Exact tied-score compatible development sweep, individual saturation regions |
| Logical pixel AP/AUROC | Exact pooled diagnostics over logical + good images, union labels; not sPRO substitutes |
| Full-resolution heatmaps | Finite aligned 800×1600 maps; read-only validated viewer |
| p50/p95 CPU inference latency | Batch one, first inference in fresh worker; decode/resize/predict/map restore; excludes model load/process startup |
| Parameters/serialized size/RAM | Total, trainable-before-freeze, zero inference-trainable, checkpoint bytes, worker peak RSS |
| CUDA inference timing/memory | Existing normal-image GPU runner synchronizes timings and resets peak allocation for inference; new normal-only path includes original-image preprocessing/restoration |
| Training time/GPU-slot time | Original run measurements retained and linked; feedback CPU time is separate |

Real-development CPU inference does not measure or imply CUDA latency/VRAM. GPU feedback would require a separately budgeted GPU execution path; it was not run. The existing guarded CUDA normal-image benchmark remains available within authorized training jobs.

## Actual frozen-checkpoint integration

**LOCO custom-split development feedback — not official full-test results.**

- Originating run: `26f7493b-c22a-403c-9de2-cf9f6494abc7` (existing combined seed 7; unchanged).
- Checkpoint SHA256: `61fee93cbe37ce8e4a420cf97bd692db3db461986f53d44ca0eca276bad24b98`.
- New evaluation: `8cf168df6c2d7206f9b13857`; prior detection-v1 evaluation remains stored.
- MLflow originating run: `c0b1d7f3b3844a5e9691e49db2d05c87`.

| Measurement | Value |
|---|---:|
| Logical / structural image AUROC | 0.506443 / 0.415120 |
| Logical / structural compatible AU-sPRO@5% | 0.190492 / 0.222422 |
| Logical pooled pixel AP / AUROC | 0.141292 / 0.773457 |
| Logical / structural annotation regions | 77 / 47 |
| Normal false positives | 1 / 47 (2.13%) |
| Frozen threshold | 58.9317626953125 |
| CPU batch-one p50 / p95 | 86.40 / 90.96 ms |
| Total / trainable before freeze / during inference | 3420 / 3420 / 0 |
| Serialized checkpoint | 16,823 bytes |
| Peak worker RAM | 371.91 MiB |
| Total evaluation wall time | 187.58 seconds, below 600-second limit |
| New training / GPU execution | None / zero |

A bounded trusted-scoring subprocess independently re-read the saved maps and exactly reproduced all eight localization measurements; inference and training were not repeated. No scientific improvement is asserted from this pipeline check. Supplementary pixel AUROC and image AUROC answer different questions. Saturated region scoring retains the logical annotation semantics that ordinary pixel AP does not.

## Tracking, monitoring and tests

MLflow automatically received localization and efficiency metrics; pending outbox count was zero. Historical SQLite metrics were not rewritten; new versioned events/artifacts preserve evaluation provenance. MLflow metric history is appended; future deliveries also include evaluator-version metric namespaces. Researcher/critic prompts receive attributed run/candidate/hypothesis summaries. Inference and trusted scoring share a supervised process-group timeout/cancellation helper; no scoring operation can silently outlive the campaign deadline. The training→feedback→critique→next-decision integration is tested; no live Codex campaign was launched.

The actual Results page rendered successfully against the real ledger; the real logical heatmap preview was 800×1600. Synthetic comparisons/previews are optional expandable sections. MLflow and Streamlit stayed healthy with unchanged service PIDs; only the tracking worker was reloaded after source changes.

Full regression with real Bubblewrap fixtures: **170 passed**, zero skipped (8.67 seconds). Tests include normal-only training and calibration, disabled synthetic generators, frozen checkpoint/state mutation rejection, mask associations, multi-region saturation, ties, analytical/reference fixtures, shape alignment, detection separation, unavailable-CUDA rejection, tracking retries/idempotence, feedback to Codex and historical preservation.

## Remaining limits

Original scorer equivalence remains unverified until its source is recovered and compared. Shared-UID and malicious in-process-code risks remain explicit. Development reuse influences research selection and cannot establish independent final-holdout performance. No campaign approval, budget increase, CUDA installation change or new service was performed.
