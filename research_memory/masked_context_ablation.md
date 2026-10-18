# Predeclared masked-context ablation (juice_bottle pilot)

## Evidence prompting the study

The reviewed CUDA run `b0817b1f-d185-4827-8779-a69252e7b7e9` trained the
combined local-plus-global masked predictor for 100 updates on authorized normal
images. Primary synthetic pixel AUROC was 0.9531, alternate-generator AUROC was
0.8027, normal-image false-positive rate was 13.89%, and the synthetic region
proxy was 0.2781. The Research Critic marked the original hypothesis
**inconclusive** because no matched local arm or localization AP was measured,
and the generator gap raised shortcut concerns. These values are development
proxies, not official MVTec LOCO results.

## Controlled question and arms

Does broad context improve localization on independently generated development
edits without increasing false alarms on normal or layout-preserving controls?

The reviewed candidate `masked_context_ablation` derives from parent Git commit
`3367a666bb77fc1122666a741cf3d6554fb7eb97`. Its `local`, `global`, and
`combined` arms keep the same 3,420-parameter network, targets, 16-pixel
complementary masks, guard band, optimizer, score assembly, four inference
passes, and original-resolution map restoration. Only the two context input
tensors change. Each arm uses seed 7, 128-pixel input, batch size four and 100
updates on the same 335 authorized normal training images. Actual update and
exposure records must match 100 and 400 before comparison proceeds.

The first run is the combined anchor. A real Codex Researcher call receives its
measured metrics and Critic review, then chooses **local or global** as the
second arm in a schema-constrained decision citing the source run. The final
remaining arm completes the paired comparison if budget remains. The choice of
the second run is therefore an evidence-dependent recorded decision.

## Frozen development evaluation

The trusted runner keeps 18 good validation images for normal-score threshold
calibration and 36 disjoint good images for assessment. It applies the same
primary white-patch and alternate relocated-copy generators to every arm. A
third feathered relocation generator uses an identical patch/blend mask for
two controls: a donor patch from another good image at the **same location**
and a mild photometric edit of the existing patch. The donor control is only
*plausibly* layout-preserving; no annotation certifies it as logically valid.
Donor images are paired within the assessment set, yielding 18 donor clusters.

Record synthetic pixel AUROC and AP for each generator, normal and control
false-positive rates at the separately calibrated threshold, region proxy,
full batch-one latency, parameter count, checkpoint size, peak VRAM, training
duration, exclusive GPU-slot time and aligned heatmaps. Per-image development
metrics permit a fixed-seed paired bootstrap over donor clusters. This interval
describes this one category, synthetic edits and one training seed only.

For this pilot, a useful combined-versus-local result requires a mean per-image
AP gain of at least 0.03 on both primary and alternate generators, positive
paired 95% lower bounds, and no material increase in normal, same-location
donor, or photometric-control false alarms. A generator-specific gain or
control degradation argues for shortcut learning. The original five-category
macro claim remains untested regardless of this outcome.

## Authorization and resource ceiling

The reviewable configuration is `configs/masked_context_ablation.yaml`. It
currently **refuses to start**: `operator_accepts_shared_uid_risk: false`.
Approval of a separate personal-server campaign is required before changing
that single value in an operator-owned copy. The proposed ceiling is three
serialized jobs, at most 120 seconds each and six GPU-slot minutes total,
four Codex calls, two wall hours and one consecutive failure. The trusted GPU
lock, Bubblewrap candidate boundary, good-only manifest, and existing MLflow
outbox remain in force. No full four-hour campaign or official evaluation is
authorized by this plan.
