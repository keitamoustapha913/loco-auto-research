# Stage 6 metric and synthetic-proxy audit

Audit date: 2026-10-18. Scope: authorized `juice_bottle` good-only development
data and the three preserved Stage 5 runs. This report does not contain or use
official LOCO anomalies or masks. Stage 6B GPU execution is **not authorized**.

## Historical record and evaluator versions

The original Stage 5 measurements remain untouched in SQLite, MLflow and their
job directories. The three run IDs are `70b0bd9b-95f0-4c9e-a0e3-c1757f95a59b`
(combined), `3565b411-8ad4-4eb3-a1dc-3bc0e462accb` (local) and
`e5f6f6a2-2959-46ef-909e-a725c45f1c63` (global). Their original protocol
manifest is `1.1`; it did not name a metric evaluator version. We refer to
these measurements as **legacy development v1**. The source remains in
`benchmark/metrics.py` and `benchmark/validation.py` for reproduction.

New runs use **development-v2.0** from `benchmark/metrics_v2.py` and
`benchmark.validation.evaluate_dev_v2`. The evaluator version, mask semantics,
generator source hash, calibration and assessment support are written to the
new run manifest and tagged in MLflow. Version 2 does not rewrite v1 results.
Its upper-order-statistic normal threshold differs from v1 interpolation, so
normal false-positive rates and any threshold-dependent controls are **not
directly comparable across versions**. AP values from valid binary masks use
the same tie-aware pooled-pixel formula, but future comparisons must still
check evaluator and generator versions, splits and masks. No Stage 5 model was
retrained or rescored during this audit. Full-resolution raw score maps for all
36 images were not retained, so their pooled AP cannot be recomputed from
the display heatmap PNG without changing the estimand.

## AP correctness

For valid binary masks, v1 pooled AP ranks all `[N,H,W]` pixels together,
groups equal scores, and integrates precision over positive recall increments.
Deterministic fixtures match scikit-learn average precision, including ties.
Higher scores mean more anomalous. Pixel AUROC is localization ranking;
image AUROC uses each image's maximum map score and answers a different
classification question. Neither is official LOCO logical localization.

**Confirmed bug:** v1 AP and pixel AUROC cast masks to `uint8`/`int8` before
validating binary labels. For example, a mask value of 2 can become an accepted
label or a value such as 256 can become zero. Version 2 validates raw mask
values and exact aligned `[N,H,W]` shapes before casting; it rejects NaN or
infinite scores. No-positive AP is explicitly undefined (`NaN`), all-positive
AP is 1, and macro AP omits and counts no-positive images. Production synthetic
generators must contain positives or the run fails instead of logging a
nonfinite measurement. Pooled AP weights every pixel equally; macro AP gives
each positive image equal weight. They need not match. Version 2 records both
and their image/pixel denominators. Per-image AP remains available for paired
comparisons; millions of pixels are not treated as independent bootstrap
samples.

## Why all three normal FPRs were 13.89%

This is **image-level**, not pixel- or region-level, FPR. The authorized 54
normal validation images were deterministically split into 18 good-only
threshold-calibration images and 36 separate good assessment images. The
v1 rule took the 95th-percentile interpolated calibration image maximum.
The alarm rule is strictly `image_max > threshold`.

| Arm | v1 threshold | False alarms | Assessed normal images | Alarm image indices |
|---|---:|---:|---:|---|
| Combined | 49.3090496063 | 5 | 36 | 0, 7, 10, 13, 23 |
| Local | 49.5191724777 | 5 | 36 | 0, 7, 10, 13, 23 |
| Global | 48.6093765259 | 5 | 36 | 0, 7, 10, 13, 23 |

Thus each measured `5/36 = 13.888…%`; the equality is **not rounding** and
not threshold reuse. The same five image indices also alarm on the
same-location donor and mild photometric controls in the saved per-image
records. The model score ordering and thresholds led to matching decisions;
individual score maps were not retained to attribute the pattern more deeply.
Version 2 records counts and denominators. With only 18 calibration images,
its conservative 5%-budget upper order statistic is the maximum observed
calibration score, yielding zero empirical calibration alarms. This does **not**
guarantee a 5% population FPR; the 36-image assessment remains independent
and should be reported with uncertainty. No threshold uses anomalous test data.

## Mask alignment and shortcut assessment

The development runner evaluates 128×128 masks on 128×128 model maps.
For the one saved preview per run, it bilinearly restores the score map and
synthetic RGB input to the source 800×1600 grid, and uses nearest-neighbor
interpolation for the binary mask. Shape and label tests pass. This is an
aligned transformation of the **model-space edit footprint**, not a certified
pixel annotation of a real logical defect. Square preprocessing changes aspect
ratio and discards detail before restoration; the display cannot recover it.

- Primary `white_patch_v1`: a high-contrast white square with an exact edit
  footprint. It is an infrastructure sensitivity probe, weak as a logical
  anomaly proxy. Its appearance can be learned as a shortcut.
- Alternate `relocated_copy_v1`: a target patch is overwritten by content
  copied from another position in the **same** good image. The source is left
  intact. The target-only mask marks an insertion footprint, never a
  missing-object region.
- Feathered copy-insertion: the same target-only operation with alpha blending.
  The copy and alternate generators can share identical target geometry for
  the same seed/size; different seeds in Stage 5 change locations but do not
  establish independent logical anomaly families. The feathered mask includes
  partially blended edge pixels. It is a proxy annotation for the edit support.
- Same-location donor control: uses a patch from another good assessment image
  with the same blend support. It is **not certified logically valid** and is
  excluded from any claim about logical validity. The mild photometric control
  preserves layout but still changes appearance. Both are scored as controls,
  not anomaly labels. Unmodified good images supply the main FPR denominator.

The Stage 5 primary/alternate AP gap is large: combined 0.3156/0.1032,
local 0.3073/0.1011, global 0.3557/0.0995. The copy-insertion APs are
0.0661, 0.0665 and 0.0598 respectively. Global's primary advantage does not
transfer to these related edits. This is evidence of generator sensitivity,
not proof of a specific shortcut mechanism or real-world failure.

No current synthetic generator establishes a defensible missing-object
localization target. A future missing-object protocol needs explicit source,
absence and context-region semantics plus human semantic review. No sealed
ground truth is available to settle this during development.

## Paired statistics and interpretation

Stage 5 reported both pooled AP and equal-weight per-image paired AP.
Differences between their deltas are expected because the estimands differ;
they are not a numerical bug. The earlier 18-donor-cluster bootstrap used one
training seed and cannot quantify seed-to-seed variability or population
significance. Version 2 retains per-image metrics and provides a fixed-seed
paired bootstrap over images or explicitly declared shared-donor clusters.
The prepared study reports sample SD across three training seeds **separately**
from within-seed image-pair intervals. It uses identical assessment image
indices and generator source fingerprints across arms. With only three seeds
and one category, all intervals remain screening evidence.

The predeclared combined-versus-local success rule required at least +0.03
mean per-image AP on both primary and alternate generators, positive paired
lower bounds and no material control-FPR increase. Stage 5 gains were only
+0.0079 and +0.0040. The rule failed. The broader contextual mechanism remains
inconclusive because global alone improved the primary proxy while worsening
the related copy-insertion proxies.

## Stage 6B readiness and proposed resources

The **metric implementation** is ready for a bounded synthetic-only
confirmatory study after separate authorization. Scientific validity remains
limited by one category, target-only copy masks, uncertified donor controls,
small normal calibration/assessment sets and shared synthetic geometry. The
study must not promote an architecture on these proxies alone.

`configs/stage6_multiseed.yaml` is disabled by default. It fixes five
same-capacity context arms (`local`, `global`, `combined`, `neither`, `spatial`),
three seeds (7, 17, 29), 100 optimizer updates and 400 image exposures per arm,
128-pixel preprocessing, the same good-image splits and version-2 evaluator.
The neither arm removes local and global image context but retains coordinates
and mask visibility; the spatial baseline also removes mask visibility. Both
still train the same 3,420-parameter network on good images. They therefore
match steps, capacity and input data, though effective information and compute
paths differ by design. No external weights or reference-image memory bank are
used. The deterministic controller executes the fixed 15-job grid and calls
the Research Critic once after the full paired comparison. It does not adapt
the confirmatory grid to intermediate results.

The **maximum** is 15 serialized jobs × 120 s = 1,800 GPU-slot seconds
(30 minutes), four wall hours and **one Codex call**. Stage 5 used about 14 s
GPU-slot time per arm, implying roughly 210 s (3.5 minutes) for 15 similar
jobs, subject to actual runtime and possible failures. The existing global
resource budget also applies. Failed, timed-out and interrupted jobs consume
their measured GPU-slot time. No Stage 6B GPU job ran during this audit.

After explicit authorization, the operator must prepare a private copy of
`configs/stage6_multiseed.yaml` with only
`security.operator_accepts_shared_uid_risk: true`, then run:

```bash
uv run --extra all loco-lab research start --config /tmp/loco-stage6-approved.yaml
```

The repository copy stays disabled. Shared-host-UID Bubblewrap remains a
documented residual risk; approval is for this bounded personal-server study
only. MLflow and Streamlit continue as existing services. A one-category
synthetic study cannot substitute for independent, human-authorized official
evaluation.
