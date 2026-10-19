# Research charter — human-owned

Mission: discover precise, compact, efficient, memory-bank-free logical anomaly heatmap localization methods on **MVTec LOCO AD**, using good-image learning with optional synthetic anomalies derived from permitted good images.

## Invariants

- All five categories eventually evaluated: breakfast_box, juice_bottle, pushpins, screw_bag, splicing_connectors.
- Final-holdout images, masks, listings and private manifests remain off-server and inaccessible to development. Operator amendment (2026-10-19): the pinned `loco-feedback-v1.0-f91c7f27aad59de58c5a` partition is authorized for trusted inference-only development evaluation and aggregate method-selection feedback. It is not an untouched official test benchmark.
- No model selection based on sealed test metrics. Using sealed results to revise a model changes the protocol and must be disclosed as test-informed development.
- The standard predictor contract is float anomaly map `[B,1,H,W]` with higher values meaning more anomalous, aligned to original input pixel coordinates.
- Promoted inference must not consult a dataset-specific nearest-neighbor bank, raw reference images, or normal-image feature storage; trained weights/compact parametric summaries may be permitted with explicit documentation.
- Trust measurements from deterministic tooling, not unverified agent prose. Link each run to code, dataset, configuration and hashes.
- Minimize end-to-end batch-1 latency and maximize *meaningful* logical localization, not image classification alone.

## Data policy

Good-only training. Good-only calibrations. Synthetic development anomalies and edited-still-good controls are allowed if sourced from permitted good splits and transparently labeled. Official test is strictly sealed. External pretrained models and source code require licenses and provenance.

## Promotion and fairness

Pilot experiments are screening, not definitive. Repeat promising experiments across multiple seeds, disclose training compute/image exposures, compare against baselines and analyze synthetic shortcut sensitivity. Report Pareto trade-offs and incomplete evidence. Human approval is required for official test evaluation, policy modifications, unapproved new dependencies, and budget increases.

## Authorized real-development feedback — operator amendment, 2026-10-19

Only the transferred, hash-pinned development partition may be imported. Training remains 335 `train/good` images with normal-derived augmentation; development images/masks must never enter gradient updates, weight learning, feature banks, normalization fitting or threshold calibration. Candidate inference gets one opaque image at a time, no masks or peer cases. Codex may use aggregate scores to select methods and architectures within an authorized bounded campaign. Report all such measurements as “LOCO custom-split development feedback — not official full-test results.” Final laptop holdout scores must not feed back into this campaign; later final evaluation is a custom held-out LOCO split. Missing reference localization metrics remain unavailable, never replaced by a proxy carrying an official metric name.
