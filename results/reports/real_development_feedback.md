# Real-anomaly development feedback integration

Date: 2026-10-19. Scope: normal-only learning, inference-only custom development feedback.

**LOCO custom-split development feedback — not official full-test results.**

## Imported partition

- Split: `loco-feedback-v1.0-f91c7f27aad59de58c5a`.
- 165 images: 47 good, 71 logical anomalies, 47 structural anomalies.
- 124 masks; six images have multiple masks. Masks remain separate instances with original defect-value encodings and public saturation metadata; they are not flattened into a pixel-AP substitute.
- Archive SHA256: `cdd465e98b2a8ea6b3645c7a6b7a4244106ccc253f6cd551eec9f3296fd7b8cd`.
- Both archive and canonical manifest are pinned in `configs/development_feedback.json`.
- Import: `$LOCO_RUNTIME/feedback/juice_bottle/v1/`, separate from `good_data`.
- All 292 archive members matched declared hashes/sizes, image dimensions/modes, image-to-mask associations and defect encoding. No links, special files, undeclared members or alternative splits are accepted.
- No laptop access, full-dataset download, final-holdout listing or private holdout manifest was requested.

## Learning/evaluation boundary

The training API now receives an explicitly filtered `train/good` manifest. The existing guarded training process mounts only authorized good data, not the feedback tree. Training provenance records policy, count and training-manifest fingerprint. Existing frozen checkpoints/configurations and historical synthetic metrics were not changed.

Each development image is evaluated in a fresh no-network **CPU** Bubblewrap process. The worker receives an opaque image filename, read-only candidate source/checkpoint, an empty `/data`, and one writable output directory. Masks, peer images, controller SQLite and credentials are absent. The trusted component alone reads manifest/category metadata and computes scores. Candidate outputs cannot redirect trusted reads using symlinks.

The worker sets modules to evaluation mode, disables parameter gradients, runs prediction with `torch.inference_mode()`, prohibits standard optimizer steps, and checks parameters, buffers and wrapper tensor/scalar state before/after prediction. Source/checkpoint/provenance hashes must match the original run and remain unchanged. Full-resolution maps are bilinearly restored to 800×1600 and checked for finite values and exact shape. Masks are never resized or supplied to the candidate.

These are meaningful filesystem/network and accidental-fitting protections, **not verified separate-user isolation or a proof against deliberately malicious Python**. Candidate code executes in the inference process and could deliberately undo Python guards, temporarily adapt hidden state, or falsify predictions. Read-only checkpoint mounts prevent persisting weight changes there; a fresh worker for each image prevents ordinary cross-image adaptation. Source review and the documented shared-host-UID risk remain necessary. The normal-training mount still contains normal validation data for the trusted runner; the train-only API is a contract, not a separate OS boundary between those normal subsets.

No new users, privileges, services, CUDA packages or campaign authorizations were introduced. Only the tracking worker was reloaded to pick up new event handling; MLflow and Streamlit continued running.

## Metric definitions and explicit blocker

Evaluator: `real-development-detection-v1.0`.

- Logical image AUROC compares all 71 logical images against the 47 development normals.
- Structural image AUROC compares all 47 structural images against the same 47 normals.
- Score is the maximum model-resolution anomaly-map value, higher = more anomalous, matching the originating normal-calibration score. AUROC uses scikit-learn with tied-score handling. This is a documented development detection score, not a claim of complete reference-evaluator equivalence.
- Normal false positives use strict `score > threshold`; report count and denominator. The originating threshold was calibrated on 18 good validation images; its earlier assessment used 36 disjoint normal validation images. No threshold, normalization or weights are fitted on the new feedback images.
- Batch-one CPU latency includes image loading, frozen resize, prediction and restoration to original resolution. It excludes fresh-process startup and model loading. RSS is whole worker-process peak memory, not model-only memory or VRAM.
- **Logical AU-sPRO and structural localization metrics are unavailable.** The evaluation-code archive linked from the [official MVTec LOCO page](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) returned HTTP 404. The exact attempted URL is stored in the approval configuration. No unverified mirror or pixel-AP substitute was labeled official.
- Original-resolution raw `.npy` maps for all images, per-case scores/hashes, three display heatmaps and the immutable summary are retained. Integrating a verified official scorer with its multi-mask saturation semantics is the remaining localization blocker. Any later scoring must create a new versioned result, never overwrite this detection evaluation.

## Authorized live integration result

The operator separately authorized one CPU evaluation with a 600-second ceiling and no training/GPU use.

- Frozen originating run: `26f7493b-c22a-403c-9de2-cf9f6494abc7`.
- Candidate: Stage 6B combined-context ablation, seed 7, original size 128.
- Checkpoint SHA256: `61fee93cbe37ce8e4a420cf97bd692db3db461986f53d44ca0eca276bad24b98`.
- Evaluation ID: `8ec6dde8348b089ef7647773`.
- MLflow run: `c0b1d7f3b3844a5e9691e49db2d05c87` (same originating training run).

| Measurement | Actual value |
|---|---:|
| Logical image AUROC | 0.506443 |
| Structural image AUROC | 0.415120 |
| Normal false positives | 1 / 47 (2.13%) |
| Frozen normal-only threshold | 58.9317626953125 |
| Parameters | 3,420 |
| CPU batch-one latency, p95 | 88.46 ms |
| Worker peak RSS | 371.11 MiB |
| Total evaluation wall time | 133.91 s |
| GPU time / training jobs | 0 / 0 |
| Original-resolution maps | 165 |

Inference-state and checkpoint integrity checks passed. These results do not demonstrate useful real logical detection or localization. This single frozen candidate was chosen for infrastructure verification, not as the winner of a real-anomaly comparison. No significance or cross-seed robustness claim is made.

## Automatic research feedback and monitoring

The existing campaign loop, when explicitly configured for real feedback, performs this bounded stage after training and before critique. It uses the remaining campaign wall-time budget, honors stop requests and the existing consecutive-failure limit. Completed results become an atomic SQLite event plus durable MLflow outbox item. Failed/interrupted inference is retained for review and cannot silently replay. Successful repeated requests are idempotent.

Researcher and critic prompts receive aggregate real-development summaries with explicit prohibitions against anomaly fitting and warnings about unavailable localization. The automatic training → feedback → critique → next-decision ordering is covered by an integration test. **No new live Codex call or autonomous campaign was run in this assignment.** This verifies the prompt/data interface, not a new scientific agent decision.

MLflow automatically delivered all new metrics/tags and summary/per-case/preview artifacts with zero pending tracking events. Original SQLite metrics remain unchanged. Results renders real development scores separately from synthetic scores; rendering against the actual ledger succeeded. Health checks confirmed MLflow, Streamlit and tracking healthy, CUDA/RTX 4090 available, GPU worker idle and research worker stopped.

## Verification

Tests cover pinned archive/schema integrity, missing transfers, split/holdout rejection, mask association and multiple masks, import idempotence and symlink rejection, analytical AUROC ties/FPR boundaries, train-only manifest inputs, inference-mode/state/checkpoint mutation checks, spatial output shape, artifact path safety, atomic/idempotent outbox handling and outage recovery, historical metric preservation, critic feedback, automatic campaign ordering, disabled authorization and Results rendering.

**Full regression: 159 passed, zero skipped**, with `LOCO_TEST_BWRAP=1 .venv/bin/pytest -ra` (7.96 seconds). Live Bubblewrap adversarial fixtures verify controller/secret invisibility, trusted-source write rejection and host-loopback denial. The ordinary sandboxed run also passed (154 passed, three namespace tests skipped before the final two tests were added).

## Next operation and limits

`configs/autonomy_real_feedback.yaml` is a **disabled proposal**, not an authorization: two serialized GPU jobs, 120 seconds each, four total GPU-slot minutes, one wall hour, eight Codex calls, and at most 600 CPU seconds of feedback per completed candidate. Existing global ceilings still apply. A verified localization scorer should be integrated before claiming a localization-driven campaign; any earlier campaign would have only detection feedback and synthetic localization proxies.

After separate operator approval, use a private mode-0600 operator-owned copy at `$LOCO_RUNTIME/authorizations/real_feedback.yaml`, retaining the approved bounds and explicitly accepting the shared-UID risk. Then:

```bash
uv run --extra all loco-lab research start \
  --config "$HOME/workspace-research/loco-runtime/authorizations/real_feedback.yaml"
```

No private authorization file was created and no future campaign was enabled. Repeated feedback will adapt methods to this custom development split. Final validation must use the laptop's independent holdout after model selection, reported as a **custom held-out LOCO split**, with no feedback into the same method-selection campaign.
