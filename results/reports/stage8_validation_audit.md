# Stage 8 — semantic validation audit

**2026-10-18. Status: validation framework implemented; semantic benchmark not yet certified.** No Stage 6B model was retrained. No CUDA job, privileged host change, official anomalous LOCO image, label or mask was used. Historical SQLite and MLflow records remain unchanged.

## Permitted evidence and candidate rules

The verified good-only `juice_bottle` manifest has **335 training and 54 validation images**, all **800×1600**, fingerprint `30a59fec4030867900a3396f5ca4191c0ef7995bf75fee7781d0f2ea2a629b3d`. Inspection of permitted normal images found a bottle, fruit pictogram, lower “100% Juice” label and liquid whose color varies among the observed fruit variants. Across inspected normals, one bottle and one front pictogram, an upper pictogram/lower text order and a plausible fruit/liquid appearance relationship are *empirical observations*. None is certified as a universal product rule by these images alone. Component count, expected missing-part location, identity equivalence and valid placement need independent product knowledge or review.

The [official MVTec LOCO description](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) and [dataset paper](https://link.springer.com/article/10.1007/s11263-022-01578-9) establish the general distinction between structural defects and violated logical constraints. They do not certify the draft rule used here. The full [semantic evaluation specification](../../schemas/semantic_development.md) defines the review gate, five localization target policies, source-coordinate masks, threshold use and metrics without consulting the sealed official test set.

## Independence audit

Stages 5–6 used the **335 good training images** for fitting and all **54 good validation images**, split as 18 calibration and 36 assessment images in the common runner. Stage 6B used that split across 15 jobs. Stage 7 reused 12 of the 36 assessment images. Stage 8's three sources (`validation/good/020`, `/030`, `/040`) are also in that assessment partition. The 9 Stage 8 images are three originals and six derived edits of those **same three source scenes**. **Fresh independent source images: 0.** Multiple edit strengths and three training seeds do not increase source independence.

The next confirmatory study needs newly permitted good images with capture/physical-object provenance, independent good-status review, a versioned fingerprinted manifest and source/donor-cluster separation from training, calibration and all prior assessment. Different filenames or pixel hashes alone are insufficient. Source allocation, rule labels, generator versions, target policy, calibration and evaluation code must be frozen before comparative model results are opened.

## Versioned cases and quality control

`semantic-development-v1.3` produces three sets from authorized good images:

| Set | Cases | Independent semantic verification | Current use |
|---|---:|---:|---|
| Manifest-good originals | 3 | Good split and SHA-256 validated | Normal-image scores and false alarms |
| Putative valid liquid edits | 3 | **0 reviewed** | Provisional control scores/alarms only |
| Putative pictogram/liquid mismatches | 3 | **0 reviewed; 0 semantic targets** | Provisional score contrast and edit-footprint proxy AP only |

The generator uses an image-derived red-liquid mask within a draft bottle polygon and a smooth color operation at two strengths. It does not move a rectangular patch. Both edits use the same spatial footprint; **status remains confounded with edit magnitude**, and neither edit's logical validity has been certified. Draft pictogram and liquid regions are displayed for review but are not semantic ground-truth masks. The footprint is a separate binary artifact of the renderer.

Visual pre-score inspection rejected three earlier local drafts: v1.0 recolored printed labels; v1.1 excluded whole sticker rectangles and introduced obvious colored stripes; v1.2 still recolored the cherry pictogram. Their local generated artifacts were retained as invalid. Version 1.3 excludes the pictogram and bright sticker pixels and was frozen **before** frozen-model inference. It still may contain color-selection boundaries and unnatural liquid appearance; no claim of artifact-free realism follows from this check.

Each case manifest records source path/hash, derived image/hash, transform, draft rule, intended label, proposed violation, draft implicated regions and review state. The evaluator derives review status from separate, case-hash-bound reviewer files. Its blinded packet generator uses opaque IDs and neutral filenames; two distinct reviewers must inspect source and edit without model outputs, agree on status and region, and supply reasons. Logic-only agreement cannot unlock semantic AP. A reviewed semantic mask must be independently drawn at 800×1600 with its own hash and target policy. The code cannot prove reviewer independence or adjudicate product specifications; the operator must enforce those parts of the workflow. Human-reviewed synthetic examples would remain synthetic development examples, not naturally occurring anomalies.

## Frozen-checkpoint CPU diagnostic

The existing no-network Bubblewrap boundary evaluated **12 unique Stage 6B checkpoints** (local, global, combined and one deduplicated neither/spatial baseline; seeds 7, 17, 29) against the same nine case files. Checkpoints, source hashes and original good-only thresholds were verified. No threshold was fitted to the edits. The model input was 128×128; binary masks were resized from 800×1600 with nearest neighbor. **Semantic AP is unavailable for every run** because no case has an independently verified logical label and accepted target.

| Frozen arm | Normal alarms by seed, each /3 | Provisional valid alarms by seed, each /3 | Provisional mismatch alarms by seed, each /3 | Mean mismatch − valid image score by seed | Mean edit-footprint AP proxy |
|---|---|---|---|---|---:|
| Local | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | −0.741, −1.219, −0.103 | 0.3435 |
| Global | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | −0.811, −1.282, −0.255 | 0.3104 |
| Combined | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | −0.762, −1.232, −0.149 | 0.3354 |
| Neither | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | −1.000, −1.747, 0.000 | 0.2279 |

These are **three shared sources per seed**, not nine independent scenes per arm. Global minus local's paired difference in the proposed mismatch-versus-valid score contrast averaged **−0.0953** across seeds. Fixed-seed image-bootstrap 95% intervals on the three source identities were [−0.167, 0.068], [−0.165, 0.061] and [−0.193, −0.117]; bootstrap seeds are 8107, 8117 and 8129 for checkpoint seeds 7, 17 and 29. The conditional intervals are descriptive at this sample size. The stronger recolor scored lower than the weaker edit for all local/global/combined seeds even though edited-region AP was measurable. This directly demonstrates how an edit-footprint proxy can diverge from the proposed logical decision. It **does not** show that these models fail on verified logical defects: no such defect was available.

The original Stage 6B runs already record 3,420 parameters per checkpoint, approximately 1.0–1.24 ms CUDA batch-one p95 latency and 72.93 MiB peak VRAM. Stage 8 added approximately 18 seconds of wall-clock CPU evaluation and **0 GPU-slot seconds**. Original resource and MLflow records were neither changed nor overwritten. Local raw case scores, generator manifests and images remain in ignored `results/metrics/stage8_semantic/` and `results/metrics/stage8_eval/`; generated data are not committed.

## Critic and evidence-adaptive direction

The **real Codex Research Critic** used the measured summary and returned `reject` for promotion, with high confidence. Its rejection concerns the current validation construct and promotion claim: the fruit/liquid rule, weak edit and strong edit have zero independent verification; edit strength is confounded with intended status; an edit-footprint mask is not a semantic target. It proposed status-by-strength counterbalancing, equally strong verified-valid edits, sham operations, a second generator and source-cluster uncertainty before model selection. The schema-constrained response and prompt/schema hashes are retained in `results/metrics/stage8_eval/critic.json`.

The **real Codex Hypothesis Researcher** then ranked three falsifiable directions, all conditional on the validation gate. The full structured response and hashes are retained in `results/metrics/stage8_eval/proposals.json`:

1. **Conditional component relationships.** With component extraction, fitting data and calibration fixed, compare a compact conditional compatibility score with factorized component evidence. Test verified count-preserving mismatches against equally edited valid combinations. This most directly addresses logical generalization and could localize implicated components, but segmentation and target attribution are hard; end-to-end latency is unmeasured. The Researcher's rough 35-GPU-minute research ceiling is **not authorized**. Component-level motivation appears in [CSAD](https://arxiv.org/abs/2408.15628); this proposal is not a reproduction.
2. **Discrete component cardinality.** Add a compact count/occupancy term with fixed extraction and compare to an area baseline on independently reviewed deletion/duplication and count-preserving size controls. Extra-component localization is clearer than missing-component localization; the latter needs an independently justified expected region. Extraction errors and fixed-position shortcuts are key risks. Rough unapproved research ceiling: 20 GPU minutes.
3. **Conservative photometric normalization.** Change preprocessing only and test whether verified-valid illumination controls become quieter while an independently verified logical contrast is preserved. This is fast and small but may erase the color relationship under investigation; it cannot establish logical reasoning on its own. Rough unapproved ceiling: 10 GPU minutes.

No model architecture has been selected. The **next action is validation**, not another synthetic AP optimization: obtain reviewer agreement on the rule and cases, then fresh good sources and status-balanced editing conditions. If those cannot be established, reject this fruit/liquid proxy and choose a rule with clearer product semantics (for example, a separately certified component-count constraint). None of the existing frozen candidates merits promotion for logical localization on current evidence.

## Bounded next GPU experiment — proposed, disabled

[`configs/stage8_next_pilot.yaml`](../../configs/stage8_next_pilot.yaml) is disabled and has no automatic launcher. **After** fresh-source and semantic-review gates pass, it proposes two serialized jobs: fixed factorized component evidence versus a conditional relationship treatment, at most **300 seconds each**, **600 GPU-slot seconds total**, **90 minutes wall time**, and **two Codex calls**. It would use the existing guarded runner, good-only fitting and source-disjoint calibration. This is a small controlled pilot; any larger multi-seed study would need its own protocol and authorization. No Stage 7 CUDA plan or new autonomous campaign was launched.

## Verification

Deterministic tests cover good-only source validation, case provenance, duplicate/independent review gates, target-mask semantics, 800×1600-to-128×128 alignment, sticker/pictogram exclusion, abstention from semantic AP, matched-case comparisons and sandbox mounts. The real Stage 8 CPU comparison completed under Bubblewrap. The full regression suite on 2026-10-18 reported **132 passed, 3 skipped** (the skips require an operator-authorized user namespace). A read-only host-side `loco-lab status` check reported the RTX 4090, connected ledger, running MLflow/dashboard/tracking synchronization, zero pending tracking events and idle GPU worker. Research autonomy was disabled; the prior Stage 6 campaign was stopped at its experiment limit.
