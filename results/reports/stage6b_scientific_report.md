# Stage 6B confirmatory synthetic-development study

Campaign `5aae9e6c-5162-46aa-b455-6c503fe16fb1`; 2026-10-18; authorized normal `juice_bottle` images only. **No official LOCO scores.** All 15 prescribed CUDA jobs completed; the campaign stopped automatically at its experiment-count ceiling.

## Fixed protocol and provenance

- Arms: local, global, combined, neither, spatial; seeds: 7, 17, 29. All used 100 optimizer updates, 400 normal-image exposures, 128-pixel preprocessing, 3,420 parameters and the same authorized training, calibration and assessment splits.
- Evaluator: `development-v2.0`; 18 disjoint good calibration images and 36 good assessment images. The same source, donor pairing and synthetic-generator code evaluated every arm.
- Dataset fingerprint: `30a59fec4030867900a3396f5ca4191c0ef7995bf75fee7781d0f2ea2a629b3d`. Generator source SHA-256: `a894ff71a4192957a86041cd0cb815055657eb0541686831eeda6ae8ce60a0d2`. Calibration/assessment split hashes are in `stage6_comparison.json` and each run manifest.
- Synthetic masks identify edited target footprints. The copy source remains present; these masks do not evaluate missing-object localization. Same-location donor controls are not certified logically valid.

## Individual seed results

AP columns below are **pooled-pixel** synthetic AP, not mean per-image AP. Normal and matched-control alarms are counts out of 36. Every run has a checkpoint, per-image results and an aligned 800×1600 preview in MLflow and the dashboard.

| Seed | Arm | Primary AP | Alternate AP | Copy-insertion AP | Normal alarms | Matched-control alarms | p95 latency (ms) | GPU-slot (s) | MLflow |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | local | 0.3073 | 0.1011 | 0.0665 | 1/36 | 1/36 | 1.127 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/5c248c3af412447eac34b05fa582b0b3) |
| 7 | global | 0.3557 | 0.0995 | 0.0598 | 1/36 | 1/36 | 1.148 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/1f0a4bb82d76401298b95de0a2dbdf42) |
| 7 | combined | 0.3156 | 0.1032 | 0.0661 | 1/36 | 1/36 | 1.244 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/c0b1d7f3b3844a5e9691e49db2d05c87) |
| 7 | neither | 0.5450 | 0.0662 | 0.0401 | 1/36 | 1/36 | 1.127 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/8ab7e2459fde4b728833e1ecee2a9158) |
| 7 | spatial | 0.5450 | 0.0662 | 0.0401 | 1/36 | 1/36 | 1.048 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/6086f057ced14fe4b1c7dcee68c68fa4) |
| 17 | local | 0.2521 | 0.1010 | 0.0705 | 1/36 | 1/36 | 1.009 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/4a354a5c6ef94d1d9431a127d87a5b45) |
| 17 | global | 0.3102 | 0.0990 | 0.0635 | 0/36 | 0/36 | 1.147 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/a7a1543c844b496f8038386510673c37) |
| 17 | combined | 0.2619 | 0.1027 | 0.0698 | 1/36 | 1/36 | 1.158 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/ba6a7917c347497d83fdad1c34400e5b) |
| 17 | neither | 0.4862 | 0.0672 | 0.0433 | 0/36 | 0/36 | 1.042 | 14.02 | [run](http://127.0.0.1:56000/#/experiments/1/runs/4e54abf07ceb4e66a0f5c41c8e97c38d) |
| 17 | spatial | 0.4862 | 0.0672 | 0.0433 | 0/36 | 0/36 | 1.054 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/b28006bd95c24bcc828731b8a374e08c) |
| 29 | local | 0.2514 | 0.1008 | 0.0678 | 0/36 | 0/36 | 1.006 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/17dd8fce40574bb48adc60d0ea7840c0) |
| 29 | global | 0.3113 | 0.0987 | 0.0610 | 0/36 | 0/36 | 1.134 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/d1cc1c4e88674737a6f5af96dd71d408) |
| 29 | combined | 0.2605 | 0.1025 | 0.0671 | 0/36 | 0/36 | 1.217 | 14.52 | [run](http://127.0.0.1:56000/#/experiments/1/runs/6628a1cc6e734708abf6bc0a22b37445) |
| 29 | neither | 0.4980 | 0.0665 | 0.0407 | 0/36 | 0/36 | 1.034 | 14.02 | [run](http://127.0.0.1:56000/#/experiments/1/runs/0466ce5fbe22446588fae37ca55c92e0) |
| 29 | spatial | 0.4980 | 0.0665 | 0.0407 | 0/36 | 0/36 | 1.051 | 14.53 | [run](http://127.0.0.1:56000/#/experiments/1/runs/322e6903e7994c52844787d54c8e75df) |

Peak measured VRAM was 72.93 MiB per job. All runs used the same 3,420-parameter network. The p95 timing is the existing batch-one end-to-end runner measurement. Per-run training durations, photometric alarms, original overlays, mask files, source hashes and exact run UUIDs remain in the ledger and MLflow.

## Across-seed paired comparison

The following are **equal-weight mean per-positive-image AP ± sample SD across three training seeds**. They differ from pooled AP because the estimands differ.

| Arm | Primary macro AP | Alternate macro AP | Copy-insertion macro AP | Mean p95 latency (ms) |
|---|---:|---:|---:|---:|
| local | 0.2831 ± 0.0296 | 0.1199 ± 0.0017 | 0.0895 ± 0.0021 | 1.048 |
| global | 0.3380 ± 0.0248 | 0.1213 ± 0.0022 | 0.0829 ± 0.0018 | 1.143 |
| combined | 0.2915 ± 0.0292 | 0.1237 ± 0.0019 | 0.0906 ± 0.0019 | 1.206 |
| neither | 0.5262 ± 0.0295 | 0.0781 ± 0.0002 | 0.0495 ± 0.0028 | 1.068 |
| spatial | 0.5262 ± 0.0295 | 0.0781 ± 0.0002 | 0.0495 ± 0.0028 | 1.051 |

Paired arm-minus-local macro AP differences are formed on the **same 36 assessment images within each seed**. Each per-seed 95% interval bootstraps 18 declared donor clusters (2,000 fixed-seed resamples); the SD across three training seeds is reported separately. These are conditional synthetic-development intervals, not independent-pixel or official-benchmark confidence intervals.

| Arm minus local | Primary difference | Alternate difference | Copy-insertion difference |
|---|---:|---:|---:|
| global | +0.0549 (seed SD 0.0048) | +0.0015 (seed SD 0.0005) | -0.0066 (seed SD 0.0003) |
| combined | +0.0083 (seed SD 0.0007) | +0.0038 (seed SD 0.0003) | +0.0010 (seed SD 0.0003) |
| neither | +0.2430 (seed SD 0.0077) | -0.0418 (seed SD 0.0018) | -0.0400 (seed SD 0.0008) |
| spatial | +0.2430 (seed SD 0.0077) | -0.0418 (seed SD 0.0018) | -0.0400 (seed SD 0.0008) |

**Predeclared combined-versus-local rule failed.** The +0.03 minimum mean per-image AP gain on both primary and alternate generators was not reached: observed gains were +0.0083 and +0.0038. The paired primary intervals cross zero in two of three seeds; copy-insertion intervals cross zero in all three.

Global-minus-local primary AP was +0.0549; all three per-seed image-cluster intervals were positive (seed 7: [0.0190, 0.0787], seed 17: [0.0324, 0.0816], seed 29: [0.0313, 0.0836]). Copy-insertion AP was -0.0066; all three intervals were negative (respectively [-0.0104, -0.0022], [-0.0110, -0.0026], [-0.0105, -0.0025]). Alternate AP differed by only +0.0015 and its per-seed intervals all crossed zero. Thus the primary gain is reproducible on this fixed generator and does **not** demonstrate transferable logical localization.

Neither and spatial gave bit-identical heatmaps and per-image results for every seed; their trained tensor states and scales were also identical. The serialized checkpoint files differ only in arm metadata. The visible-mask input differs between these implementations, but the 1×1 network is scored and trained only at hidden pixels where that input is zero. They are **one effective baseline**, not independent corroboration. This baseline exceeded all context arms on the primary proxy while losing markedly on alternate and copy-insertion proxies, strongly motivating a spatial-shortcut audit.

## Normal and edit controls

Per-seed normal alarms (out of the same 36 images) were local [1,1,0], global [1,0,0], combined [1,1,0], neither [1,0,0], spatial [1,0,0]. Same-location donor and photometric controls had the same alarm counts as unmodified normal images within each run; gentle appearance-only edits had 0/36 alarms. Version 2 selected each run's threshold using only 18 separate good calibration images. A 0/36 or 1/36 result remains imprecise (Wilson upper 95% limits about 9.6% and 14.2%); repeated seeds do not create 108 independent assessment images. The donor control remains semantically uncertified and cannot resolve edit-artifact sensitivity by itself.

## Critic assessment and next direction

The single budgeted Research Critic call recommended **investigate**, with medium confidence. It found no basis to promote the global-context hypothesis: its primary synthetic advantage failed to transfer to the copy-insertion generator, while the strong spatial baseline result exposes location or edit-footprint shortcuts. The critic also noted limited source-cluster independence, one category, mask ambiguity and incomplete evidence for real logical defects.

Its next falsifiable proposal is a separately authorized **counterfactual evaluation of frozen checkpoints**: use newly held-out authorized good source clusters; vary expected position and rendering independently; pair edited-still-good controls with violation edits using matched operations; define missing-object/source/destination mask semantics before evaluation; report paired source-cluster intervals, seed variation, normal/control alarms and full latency. The present 389-image authorization may not provide fresh clusters disjoint from all training and assessment images, so data availability must be checked before that study. **No next campaign was launched.**

## Resources and delivery

- Aggregate GPU-slot time: 216.85 s of 1,800 s authorized. Wall time: 262.6 s of four hours. Jobs: 15/15. Codex calls: 1/1. No campaign failures; all 15 runs are `REVIEWED`.
- All 15 MLflow runs are `FINISHED` and now include the v2 evaluator tag and per-image artifact. The final run includes `stage6_comparison.json` and the Critic assessment. The dashboard Results page renders the three-seed comparison and loads all 15 aligned heatmap examples. Tracking outbox: zero pending.
- The campaign stopped at its experiment-count limit; MLflow, Streamlit and tracking remain running, while the GPU worker is idle and autonomy is disabled. No official anomalous test data or masks were accessed.
