# First logical review package — 2026-10-19

The [frozen protocol](../../schemas/logical_review_v1.md) and local [visual review page](../metrics/logical_review_v1_final/review.html) form a small **human-review candidate set**, not a certified benchmark. The package contains two previously assessed good `juice_bottle` target images (`020`, `030`), two paired same-fruit label edits and two paired cross-fruit label edits. Source and donor hashes, transformations and images are recorded in its investigator manifest. All sources are authorized good validation images outside prior threshold calibration; none is a new independent holdout.

| Proposed case type | Images | Human-verified | Semantic-mask ready |
|---|---:|---:|---:|
| Unmodified good target | 2 | Good-only manifest verified | Not applicable |
| Same-fruit edited control | 2 | 0 | Not applicable |
| Cross-fruit relational inconsistency | 2 | 0 | 0 |

The observed normal-image association between fruit pictogram and liquid appearance motivates the rule. It does not prove that a swapped label is invalid under a real product specification. The [MVTec LOCO dataset description](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) supports the general concept of logical constraints; [MVTec's explanation](https://www.mvtec.com/doc/merlic/5.8/manual/en-us/Content/Tool_reference/Processing/Deep_learning_ai/detect_anomalies_in_the_global_context.html) uses a generic mislabeled juice example, but it does not certify these generated cases. The single candidate rule is therefore **ready for independent human validation, not ready for model scoring**.

Visual inspection found that full-sticker swaps produced bright edge stripes, so a rejected local draft was retained and the final generator edits only the label interior. The final edits have fewer border artifacts but can still expose donor print texture and feathering. A same-fruit edit uses the identical procedure for each target; this is a necessary control, not proof that artifact difficulty is matched. Stage 8's liquid-recolor examples remain provisional and were not promoted: color-to-flavor validity and localization are even less certain.

The blue box highlights a **discussion region around the fruit illustration**. It is not an anomaly ground-truth mask. For a violated relationship, assigning all blame to the fruit, to the liquid, or to both is scientifically ambiguous. Reviewers must resolve the attribution policy and independently annotate a target before localization AP is defined. No missing-component or quantity case was generated because deleting or duplicating a sticker cleanly, and identifying a defensible missing region, are unresolved.

**Immediate next action:** two reviewers independently inspect `review.html`, copy `review_form_template.json`, provide valid/violation/uncertain decisions and explain rule support, artifact concerns and region semantics. Rejected or uncertain cases stay out of certified metrics. If the rule itself is not defensible, stop this validation line. No GPU, training, autonomous campaign or official test access occurred.
