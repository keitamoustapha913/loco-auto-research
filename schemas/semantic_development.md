# Semantic development benchmark — `semantic-development-v1.3`

This specification governs **good-image-derived development examples**. It does not supply official MVTec LOCO anomaly ground truth. The official anomalous data and masks remain sealed. The [MVTec dataset description](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) and [LOCO paper](https://link.springer.com/article/10.1007/s11263-022-01578-9) define logical anomalies in general as violations of constraints on otherwise permissible content; neither source certifies our proposed `juice_bottle` edits.

## What the permitted normal images support

The authorized `juice_bottle` manifest contains 335 `train/good` and 54 `validation/good` RGB images, all 800×1600. Direct inspection of permitted normal images shows a bottle, a front fruit pictogram, a separate lower juice label and liquid appearance that varies with the pictured fruit. These observations motivate **candidate** rules about component presence, count, identity and relative arrangement. They do not establish that every bottle must have one pictogram, that a particular liquid color uniquely identifies a fruit, or that a changed liquid color is necessarily a logical defect. Product specifications or blinded independent reviewers must establish each rule before a semantic metric is reported.

The implemented draft tests only a proposed pictogram/liquid identity relationship on three cherry-pictogram normal sources. It recolors image-derived reddish liquid pixels while retaining the pictogram region. A weak edit is a *putative* valid control and a strong edit is a *putative* mismatch. They use the same renderer and footprint but differ in strength. This confounds status with edit magnitude; both are **provisional**. Earlier pre-score drafts that recolored sticker pixels, formed hard exclusion stripes or recolored the pictogram are retained locally as rejected generator revisions. Version 1.3 was frozen before model inference.

## Sample classes and gate

| Class | Current count | Ground truth status | Allowed metric use |
|---|---:|---|---|
| Unmodified manifest-good | 3 | Good split and hashes verified | Normal score and frozen-threshold false alarms |
| Putative valid edit | 3 | No independent validity review | Provisional edit response and alarms only |
| Putative identity mismatch | 3 | No independent violation or localization review | Provisional score contrast and **edit-footprint proxy** AP only |
| Independently reviewed valid edit | 0 | Requires two distinct blinded reviewers | Valid-edit alarm and paired semantic control analyses |
| Independently reviewed logical violation with accepted target | 0 | Requires two reviews and separate semantic mask | Semantic localization AP under frozen target policy |

Every case records source relative path and SHA-256, derived image and footprint SHA-256, versioned transformation and strength, proposed logical rule, intended label, explicit expected violation, draft implicated regions and review status. The manifest is written under ignored `results/metrics/stage8_semantic/`, with the source dataset fingerprint, generator hash and configuration hash. Only `validation/good` entries in the authorized good-only manifest are accepted. No image may be silently relabeled by editing the generated manifest; `reviews.json` is separate and checked against each case digest.

## Independent review workflow

1. Freeze `configs/stage8_semantic_dev.json`, generator source, source-image manifest and case images **before** showing model scores. For a genuinely independent study, first acquire new permitted good source clusters under a new manifest; new filenames or different edits of the same prior scenes are not independent data.
2. Give each reviewer a separate opaque-ID packet containing source, edited image and draft regions. The packet and form contain no intended status, model heatmap, score, source filename or other reviewer's answer. The operator can describe the proposed rule without revealing which edit strength is intended to be valid. Ask whether the rule is valid for this product, whether the edit remains valid, violates it or is indeterminate, whether artifacts invalidate the comparison, and what region(s) should be credited for localization. A reviewer may reject the proposed rule.
3. Generate separate packets and forms with `python -m scripts.stage8_review --cases results/metrics/stage8_semantic --reviewer REVIEWER_ID --packet-dir PRIVATE_PACKET_DIR --output REVIEWER_ID.json`. Reviewers fill `source_inspected`, `verdict`, `target_region_accepted` and a substantive `reason`. Two distinct identities and matching judgments are required. The operator merges files with `python -m scripts.stage8_review --cases results/metrics/stage8_semantic --merge alice.json bob.json --output results/metrics/stage8_semantic/reviews.json`; this maps opaque IDs back to cases and prints statuses. Identity authenticity and actual blindness are operator responsibilities; the code cannot prove them. Do not expose `case_manifest.json` to reviewers because it contains intended labels.
4. Preserve disagreement, uncertainty and rejected cases. The current gate reports `reviewed_violation_logic_only` if the logic is agreed but no independently drawn target exists. This **does not** unlock semantic AP. A semantic target needs a separate human-drawn binary mask at original 800×1600 coordinates, provenance, rationale, hash and agreement before scoring. Review labels and target definitions must be frozen before comparing models.
5. Human-reviewed synthetic examples remain synthetic development data. They are not equivalent to naturally occurring logical anomalies or official benchmark labels.

## Localization policy by logical failure type

| Failure type | Semantic target before scoring | What is separately reported |
|---|---|---|
| Missing component | Independently justified **expected component region**, derived from good-layout evidence or product specification | Removal footprint, seam ring, region uncertainty; if expected region is unidentifiable, localization is undefined |
| Extra component | Visible extra-instance pixels from an independent component annotation | Paste footprint and compositing seam |
| Incorrect identity | Independently identified wrong component, or a predeclared pair of implicated regions when attribution is ambiguous | Edited pixels and unchanged counterpart region separately |
| Incorrect position | Displaced component pixels and expected slot as **two separate targets**, each justified independently | Translation/edit footprint and border artifacts |
| Invalid relationship | Independently specified participating component masks or ambiguity set | Per-component attribution, seam and renderer regions |

`benchmark.semantic_validation.TARGET_POLICIES` and `validate_localization_target` enforce the type, independent provenance, nonempty binary original-coordinate mask and content hash. A missing component requires explicit expected-region evidence. The present liquid/pictogram draft regions are **not** semantic ground truth, and the recolor footprint is never substituted for a relation target.

## Frozen scoring rules

- Evaluator: `development-v2.0`; higher map values mean more anomalous. Reuse each frozen run's threshold calibrated on its separate 18 good validation images. Do not calibrate on edits or official data.
- Report exact normal and **reviewed-valid** alarm numerators and denominators. Report provisional alarms in a separate column. Three source images repeated across seeds remain three source clusters.
- For reviewed violations with independent masks, report per-image and pooled pixel AP, equal-weight positive-image macro AP, target policy, counts and image-cluster uncertainty. Pooled and macro AP are different estimands. No reviewed target means **no semantic AP**.
- Report edit-footprint AP as a proxy with its prevalence, never as logical localization. Keep mismatched/valid paired image-score differences, sham and equally strong valid edits, generator-specific results and nuisance alarms. Resample whole source/donor clusters, not pixels; report training-seed variation separately.
- Use identical source identities, edited case hashes, preprocessing and thresholds across frozen candidates. Map image edits and masks in original 800×1600 coordinates; resize masks with nearest neighbor to 128×128 for model-space scoring and anomaly maps with bilinear interpolation back to original coordinates. Deterministic tests check spatial alignment.

## Independence and readiness

Stages 5–6 trained on the 335 available good training images and used all 54 good validation images (18 threshold calibration, 36 assessment). Stage 7 reused 12 of those assessment images. Stage 8's three sources (validation `020`, `030`, `040`) are in the same assessment partition. **Fresh independent source count: zero.** Different generator seeds or image edits do not change that count.

Before a confirmatory comparison, acquire separately permitted good images with capture session/physical source provenance, independent good-status review and a new fingerprinted manifest. Prevent source-object and donor overlap with training, threshold calibration and prior assessment; identical file hashes alone do not establish independence. Freeze rules, review outcomes, source allocation, generators, target policies, thresholds, candidate checkpoints and metrics before viewing comparative results. Require at least two artifact generators with status counterbalanced against edit strength and direction. If no sufficiently verified examples exist, report the study as a validation feasibility audit and stop model ranking. The current benchmark is **not yet ready** for a claim of transferable logical localization.
