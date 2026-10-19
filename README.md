# LOCO Autoresearch

A normal-only weight-training laboratory with anomaly-informed research selection.
SQLite owns experiment state; MLflow tracks measurements/artifacts; Streamlit displays them.

## Daily workflow

Install once with `uv sync --extra all`. Then:

```bash
uv run --extra all loco-lab start
uv run --extra all loco-lab research start --config APPROVED_CONFIG
```

`APPROVED_CONFIG` is a private operator-owned campaign YAML with reviewed budgets and explicit
shared-UID Bubblewrap risk acceptance. Templates remain **unauthorized**. Startup does not grant
campaign authorization. `configs/autonomy_real_feedback.yaml` proposes two serialized jobs,
120 seconds each, four GPU-slot minutes, one wall hour and eight Codex calls.

Within that authorization, Codex investigates, implements, trains on good images, receives
frozen-model real-development measurements, critiques and selects its next experiment.
There are no separate worker, scorer, dashboard or MLflow-sync startup commands.

```bash
uv run --extra all loco-lab status
uv run --extra all loco-lab research status
uv run --extra all loco-lab research pause
uv run --extra all loco-lab research resume
uv run --extra all loco-lab research report
uv run --extra all loco-lab research stop
uv run --extra all loco-lab logs
uv run --extra all loco-lab stop
```

Use `doctor` for diagnostics and `smoke --dataset juice_bottle --device cuda` only for an explicitly
requested infrastructure test. Infrastructure tests are excluded from scientific rankings.

## Data and evaluation

- Learning: 335 `train/good` images, with permitted normal-derived augmentation.
- Normal validation: 54 images; reproducible disjoint 18-image threshold calibration and 36-image assessment.
- Frozen real-development feedback: 47 good, 71 logical and 47 structural images; 124 masks.
- Split: `loco-feedback-v1.0-f91c7f27aad59de58c5a`.
- Final holdout stays off-server. No final-holdout import or research feedback is supported.

Real anomalies never fit weights, losses, normalization, thresholds, memory banks or test-time
adaptation. Aggregate scores may guide architecture/hypothesis selection. Masks are read only
by trusted scoring; each no-network candidate inference process sees one opaque image.
Checkpoints and predictor state are integrity-checked. Shared-UID isolation and in-process
Python guards are not a proof against malicious code; retain source review and sandbox checks.

`benchmark/feedback_metrics.py` is the versioned real-development scoring entry point:
detection AUROC/FPR, separate logical/structural AU-sPRO@5%, and supplementary logical pixel AP/AUROC.
The sPRO scorer is a **compatible development implementation** of the published definition.
The original official archive still returns 404; original-code equivalence is unverified.
It retains individual mask/defect saturation semantics. Pixel AP does not replace sPRO.

Real results are labeled **LOCO custom-split development feedback — not official full-test results**.
The four pages remain Home, Research, Results and System. Results separates real measurements
and full-resolution heatmaps from optional historical synthetic diagnostics. Missing scores
are unavailable, never zero. New campaigns disable synthetic diagnostics by default;
`research.synthetic_diagnostics: true` enables them for a justified research use.

CPU feedback reports batch-one p50/p95 decode→preprocess→prediction→map-restoration latency,
parameters, serialized bytes and process RAM. It does not claim CUDA inference. The existing
CUDA training runner measures synchronized normal-image inference and resets inference peak
memory independently of training. Measurements state their device/protocol.

## Configuration and access

The CLI safely loads `../loco-research.env` (or `LOCO_ENV_FILE`); explicitly set environment variables
win. Existing runtime paths, ports, resource ceilings and CUDA packages are preserved.
The development archive has already been imported; `loco-lab feedback status` verifies it.
The optional user service is `deploy/systemd/loco-lab.service`; no host service is enabled automatically.

From the laptop:

```bash
ssh -N -L 8501:127.0.0.1:8501 -L 56000:127.0.0.1:56000 theyeq-admin-pc@100.91.233.99
```

Services run on the server; open http://127.0.0.1:8501 in the laptop browser.
MLflow remains at http://127.0.0.1:56000.

Run regression tests with `uv run --extra all pytest`.
See [cleanup/evaluation report](results/reports/final_workflow_cleanup.md) for exact metric definitions,
measured integration results, removals, historical compatibility and remaining limitations.
