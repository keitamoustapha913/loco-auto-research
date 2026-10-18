# Stage 7 — frozen-model generalization diagnostics

**Status:** completed CPU diagnostic, 2026-10-18. No model was retrained, no GPU job was launched, and no official LOCO anomaly image, label, or mask was used. This is synthetic development evidence, not an official benchmark result or independent scientific confirmation.

## Frozen protocol and provenance

The fixed protocol is [`configs/stage7_frozen_eval.json`](../../configs/stage7_frozen_eval.json): `stage7-frozen-diagnostics-v1` with the unchanged `development-v2.0` evaluator, 12 validation-good assessment images selected at stride three from Stage 6B's 36-image assessment partition, edit seed 13007, three Stage 6B training seeds (7, 17, 29), and CPU-only prediction. The 12 images were **already assessed in Stage 6B**; this is a diagnostic follow-up, not a new-image holdout. No new category was authorized. The good-only dataset fingerprint is `30a59fec4030867900a3396f5ca4191c0ef7995bf75fee7781d0f2ea2a629b3d`. Each frozen source hash, checkpoint hash, threshold, original run UUID and subset fingerprint is retained in `results/metrics/stage7/`.

The 15 Stage 6B runs remain unchanged in SQLite and MLflow. Stage 7 loaded 12 unique checkpoints: local, global, combined and neither for each seed. **Neither and spatial are one effective baseline:** across all three seeds their trained tensors, residual scales and prior per-image predictions are identical. The spatial arm is excluded from new comparisons. Original training budgets and preprocessing remain 100 updates, 400 good-image exposures and 128×128 input. Each model has 3,420 parameters. Original CUDA batch-one p95 latency was approximately 1.0–1.24 ms and peak VRAM 72.93 MiB per run; the frozen CPU diagnostic measured median prediction time around 48–53 ms, which is not a replacement for the original CUDA latency. The new study used **zero GPU-slot seconds**.

The saved candidate implementation trained these weights from scratch on authorized normal images and declared no external pretrained weights. Its original candidate Git commit and every checkpoint hash remain in the Stage 6B run records; Stage 7 verified the current candidate source against the recorded source hash before loading any checkpoint.

Candidate prediction ran in the existing no-network Bubblewrap boundary with only good-image data and the selected frozen checkpoint mounted read-only. The authorized output directory alone was writable. The evaluator retained each run's threshold calibrated on 18 separate good validation images; it did not recalibrate on the 12 assessment images. Raw per-image scores, AP, threshold, alarms and latency, plus 800×1600 original/overlay/mask previews for seed 7, are in the ignored local artifact directory `results/metrics/stage7/`. This report is the committed scientific record; the generated images and checkpoint files are deliberately not committed.

## Fixed controls and mask meanings

- **Absolute-position diagnostic:** translate the entire 128×128 frame eight pixels using reflected padding. Relative component arrangement is unchanged, but reflected borders are an artifact. Image-level score and frozen-threshold alarms are measured; aligned-map translation equivariance was **not** measured.
- **Rendering diagnostic:** multiply the entire image by 1.08 and clip to the valid range. Geometry is unchanged. This isolates sensitivity to this one illumination operation.
- **Insertion and matched edits:** copy a good-image patch to a distant target with fixed feathering. A donor patch pasted at the same target and a mild appearance-only paste use the same target geometry. Donor edits are **not certified logically valid**.
- **Removal and relocation proxies:** fill an arbitrary source patch using its local median, then optionally paste its content at the target. Source and target have separate masks. The source is **not a segmented component**, so its footprint is not a defensible missing-object localization target. A target-only copy mask never stands for source removal.
- All AP masks are exact **edit-footprint proxy annotations**, not semantic logical anomaly masks. The pasted appearance and edit boundary can be detected without reasoning about the product's logic. The 128×128 binary masks were nearest-neighbor mapped to the source 800×1600 coordinates only for visualization; scoring used aligned native 128×128 maps and masks.

## Measured results

The table gives equal-weight mean **per-image footprint AP across 12 images**, then mean over three frozen training seeds. It is not pooled-pixel AP. The per-run JSON also retains pooled-pixel AP and all image values; these estimands differ. Target footprint prevalence is about 2.7% of pixels, so a low AP can still exceed chance without detecting a logical anomaly.

| Frozen arm | Insertion target AP | Relocation target AP | Removal source AP | Relocation source AP | Normal alarms by seed (of 12 each) | Illumination alarms by seed (of 12 each) |
|---|---:|---:|---:|---:|---|---|
| Local | 0.1100 | 0.1148 | 0.0395 | 0.0359 | 0, 0, 0 | 1, 1, 1 |
| Global | 0.1085 | 0.1124 | 0.0426 | 0.0389 | 0, 0, 0 | 0, 1, 1 |
| Combined | 0.1126 | 0.1173 | 0.0399 | 0.0364 | 0, 0, 0 | 0, 1, 1 |
| Neither (spatial deduplicated) | 0.0618 | 0.0639 | 0.0561 | 0.0524 | 0, 0, 0 | 0, 1, 1 |

The global-minus-local paired mean insertion-target AP difference is **−0.0015**, and relocation-target is **−0.0025**. The combined-minus-local differences are **+0.0026** and **+0.0024**. Every corresponding per-seed 95% image-bootstrap interval crosses zero. For example, global-minus-local insertion-target intervals are [−0.0131, 0.0103], [−0.0138, 0.0094] and [−0.0130, 0.0089]. The neither-minus-local insertion-target difference is −0.0482, with all three per-seed intervals below zero; yet the same baseline had the highest Stage 6B primary proxy AP. This reversal is evidence of **generator dependence**, consistent with shortcut learning; it does not identify the precise shortcut mechanism. The source-footprint AP differences are small and uncertain, and source masks are not semantic removal labels.

The 1.08 illumination gain increased mean image scores by about 4.5 across arms and generated alarms in nine of the twelve model-seed evaluations (each evaluation reuses the same 12 images). Unmodified normal images, whole-frame translations, same-location donor edits and appearance-only patch edits produced zero alarms in those evaluations. The zero control alarms do **not** certify donor validity or population specificity. The calibration set has only 18 images, and each arm has its own threshold. Score changes below threshold remain relevant. All three seeds share the same image identities, so 0/36 across seeds is **not** 36 independent normal observations.

The mean image-score response to patch insertion was slightly negative for every arm on these 12 images, while the gain response was strongly positive. This weakens an image-level anomaly interpretation of the patch AP and suggests that local rank/contrast and global alarm behavior should be separated. Semantic component arrangement, logical validity and missing-object localization remain untested.

### Separately versioned texture and shift diagnostic

The original fixed AP protocol was **not changed after observing results**. A supplemental, exploratory CPU-only protocol, `stage7-texture-equivariance-v2`, applies deterministic fine RGB noise (standard deviation 0.02) and measures heatmap correlation after a whole-frame eight-pixel shift. It uses the same 12 good images and 12 frozen checkpoints. Its first implementation, retained locally as `results/metrics/stage7_texture_invalid_alignment/`, incorrectly aligned only the horizontal axis of a two-axis translation; **all v1 shift-correlation values are invalid and excluded**. The corrected v2 aligns both axes, has a deterministic regression test, and retains its own config hash and case data in `results/metrics/stage7_texture/texture_summary.json`.

| Arm | Mean fine-noise score rise | Fine-noise map correlation | Aligned shift-map correlation | Fine-noise / shift alarms |
|---|---:|---:|---:|---|
| Local | +0.774 | 0.9867 | 0.9573 | 0 / 0 in each seed |
| Global | +0.780 | 0.9868 | 0.9632 | 0 / 0 in each seed |
| Combined | +0.771 | 0.9865 | 0.9661 | 0 / 0 in each seed |
| Neither | +1.005 | 0.9880 | 1.0000 | 0 / 0 in each seed |

These correlations describe map shape on overlapping pixels; they do not test logical validity. High shifted-map correlation is compatible with a spatial or appearance shortcut—the neither baseline is almost perfectly correlated while losing insertion-target AP. Fine noise raises scores below threshold, so zero alarms do not imply texture insensitivity. This particular noise is a rendering proxy, not a faithful model of every lighting or texture change. The illumination-gain effect is larger and sometimes crosses threshold.

## Interpretation and Research Critic

**Reliable within this fixed development set:** the Stage 6B primary gain for global context fails to transfer consistently to independent generators; the new insertion-footprint global/local contrast is also unresolved. The neither/spatial duplication is exact and should never be counted as two confirmations. The gain perturbation demonstrably changes scores and sometimes crosses the normal threshold. All models' source-footprint AP is weak under an explicitly nonsemantic mask.

**Likely shortcut-sensitive:** Stage 6B's spatial baseline led its primary generator yet performed poorly on alternate and copy-insertion generators; Stage 7 again reverses its ranking on insertion targets. This pattern is compatible with spatial or edit-footprint shortcuts, but the fixed study cannot attribute causality between absolute position, renderer artifacts and mask geometry. Zero donor-control alarms are inconclusive because these edits are not certified valid. Pooled and macro AP answer different questions; neither supplies semantic ground truth. Threshold comparisons also inherit run-specific small-sample calibration noise.

The **real Codex Research Critic** reviewed the completed AP study and was called again after the corrected v2 diagnostic. The final schema-validated review is `investigate`, high confidence in the recommendation to investigate rather than promote. It found that aligned-map correlation supports only limited translation consistency, while illumination sensitivity and uncertain semantic controls remain unresolved. It recommended independently specified semantic regions, matched verified valid/invalid edits, separate renderer and position factors, fresh source images and source-cluster paired intervals before any promotion. The initial review and final review, with prompt/schema hashes, are retained as `results/metrics/stage7/stage7_critic.json` and `stage7_critic_final.json`. The Critic saw numerical summaries rather than direct checkpoint and manifest artifacts, so its provenance reservations were appropriately cautious; the trusted evaluator independently verified frozen source/checkpoint hashes and the good-only manifest before inference.

## Evidence-ranked next hypotheses

The **real Codex Hypothesis Researcher** was called again after the final Critic review and returned three schema-validated, falsifiable directions. Its first proposals remain preserved, but the following are the **final evidence-adaptive ranking**; all are proposals, not approved architectures or training jobs. Full responses and prompt/schema hashes are retained as `results/metrics/stage7/stage7_proposals.json` and `stage7_proposals_final.json`.

1. **Component cardinality:** with independently validated component extraction held fixed, compare a compact good-only count likelihood against the same component evidence without the count term. Test excess components against matched valid additions; localize a missing component only where an independent expected region can be justified. Highest potential for explicit count violations and precise excess-object support, but component segmentation and valid/invalid edit certification are prerequisites. Related component-aware work includes [CSAD](https://arxiv.org/abs/2408.15628) and the original [LOCO/GCAD paper](https://link.springer.com/article/10.1007/s11263-022-01578-9); this is not a reproduction. The Researcher's five-GPU-minute figure is a **future unapproved smoke ceiling**, not a campaign allocation.
2. **Component relations:** with the same fixed component extraction and a fixed unary baseline, add only compact pairwise type/geometry compatibility. Test count-preserving invalid arrangements against independently certified valid arrangements and score implicated participant interiors, not arbitrary seam footprints. It has high logical potential but attribution ambiguity and pair enumeration can hurt localization and latency. [ComAD](https://arxiv.org/abs/2305.08509) motivates explicit component relationships; the proposed controlled test is independent. Future unapproved smoke ceiling: five GPU minutes.
3. **Photometric nuisance normalization:** keep the frozen predictor fixed and compare a good-calibrated luminance normalization against current preprocessing. It directly tests the observed gain-induced score drift, adds no learned parameters, and should be cheap at inference. It cannot establish logical reasoning on its own and may erase meaningful color or fill cues. [EfficientAD](https://arxiv.org/abs/2303.14535) is background on efficient dense detection, not evidence for this untested normalization. Future unapproved smoke ceiling: five GPU minutes; a CPU exploratory check is possible only with a disjoint authorized good calibration set.

Ranking considers potential logical generalization, localization precision, normal-only feasibility, inference size/speed, shortcut risk and GPU cost. The first decision is **not** to train any of these now: obtain a predeclared edit-validity protocol, reliable component evidence and fresh permitted good-image source clusters first. The current 389 images provide no new images disjoint from all prior learning and assessment. Without that data and independently defensible semantic edit labels, an additional synthetic AP optimization would not resolve the scientific question. No official anomaly data is requested.

## Proposed bounded GPU evaluation; separate authorization required

[`configs/stage7_gpu_eval.yaml`](../../configs/stage7_gpu_eval.yaml) is **disabled**. If CUDA replication of this *frozen* diagnostic is useful, the proposed ceiling is 12 unique checkpoints × 60 seconds, 720 GPU-slot seconds (12 minutes) total, one job at a time, 45 minutes wall time, zero Codex calls and zero training. The same good-only samples, fixed masks and `development-v2.0` evaluator would be used. This would check CUDA inference behavior and speed, **not** solve the lack of independent images or semantic logical labels. It must be separately authorized; no flag was enabled and no GPU run was started.

Before training any new research candidate, a distinct data/validity and campaign authorization will be needed. The approved Stage 6B budget was not reused or extended. Existing MLflow, Streamlit, ledger and CUDA installation were not changed by this study.

## Reproduction and checks

On the authorized server, from the repository root, the fixed CPU evaluations can be reproduced with:

```bash
.venv/bin/python -m scripts.stage7_study \
  --runtime "$HOME/workspace-research/loco-runtime" \
  --ledger "$HOME/workspace-research/loco-runtime/controller/experiments.sqlite" \
  --manifest "$HOME/workspace-research/loco-runtime/good_manifest.json" \
  --output results/metrics/stage7
.venv/bin/python -m scripts.stage7_texture_study \
  --runtime "$HOME/workspace-research/loco-runtime" \
  --ledger "$HOME/workspace-research/loco-runtime/controller/experiments.sqlite" \
  --manifest "$HOME/workspace-research/loco-runtime/good_manifest.json" \
  --output results/metrics/stage7_texture
```

Both commands refuse altered fixed protocol files, modified candidate source or checkpoint hashes, non-good assessment manifests and an incomplete Stage 6B run grid. They use CPU-only Bubblewrap inference and do not change the ledger or MLflow. The full automated regression suite passed **124 tests; 3 optional namespace tests were skipped**. A real Bubblewrap CPU evaluation completed for all 12 checkpoints. MLflow `/health` and Streamlit `/_stcore/health` both returned HTTP 200 after the study.
