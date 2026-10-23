# Scientific execution preflight and evidence scope

This maintenance changes execution contracts, not models, scoring definitions,
checkpoint history, authorization or the 60/20/10/10 research-incumbent policy.
Only campaign elapsed time and each new model's elapsed training-loop target remain
operator timing controls. No new service, research stage or mandatory literature step.

## Normal-only diagnostic inputs

`diagnose(predictor, assessment_good_images, diagnostic_config)` receives:

- `calibration_good_images`: separate RGB float32 `[C,3,H,W]` normal validation pixels.
- `calibration_source_ids` and `assessment_source_ids`: SHA256 source identities;
  duplicates and intersection are rejected.
- `frozen_normal_threshold`: the trusted normal-only calibration result, unchanged.
- `diagnostic_id` and `validation_smoke`.

The established deterministic validation split remains first third calibration,
remainder assessment: 18 and 36 for the current 54 normal validation images.
This is reused development evidence, not a fresh independent confirmation set.
The helper supplies copies and overwrites agent-supplied identity/threshold aliases.
Existing predictor/checkpoint mutation checks remain active after diagnostics.
Diagnostics may estimate their declared normal-only operating points using the
calibration partition, never the assessment images or real-development feedback.

Declare `min_calibration_images` and `min_assessment_images` in diagnostic config.
These are input requirements, not operator settings. Validate counts against the
verified good manifest before implementation where available, at queue admission,
and before candidate import/training in the worker. Missing resources produce a
precise count/error, never an assessment-image fallback. The exact same input
builder and output validator serve disposable training preflight and production.

A new trainable model still needs the full train/load/predict contract and trusted
training_steps protocol. An all-frozen investigation implements only the declared
`diagnose` provider plus executable CPU tests. It does not need fictitious training
functions or true train/load/predict attestations; its response must still identify
the assigned candidate correctly. Exact historical models load through their own
unchanged loaders. Provider code executes only within the existing Bubblewrap
boundary. Model and trusted scoring mounts are unchanged; no anomaly masks or
final-holdout data are added to candidate or agent inputs.

## Scientific controls

Admission and completion expose actual training seeds, intended differences,
changed training settings, undeclared common controls, observed updates/exposures,
and sampling-stream evidence status. Pair labels do not replace real training
seeds. `COMPLETE` remains execution coverage, not proof of the hypothesis.

For new v4 confirmation proposals, missing matched controls prevent confirmation
admission. Declare sampling_protocol and sampling_seed independently of model
initialization randomness, and all unchanged scientific settings as controls.
New v4 Critic causal claims additionally require eligible matched design. Exploratory
comparisons remain admissible and can inform research without becoming confirmed
champions. Existing frozen plans are not rewritten or retrospectively reclassified.

A matched design is an eligibility ceiling, not verification that arbitrary candidate
Python consumed the same stochastic sequence. Actual sample/mask-prefix identity
still needs explicit instrumentation; the report labels it unverified. Equal elapsed
loop time can produce unequal exposures/updates: those observed counts remain visible.
Three independent seed pairs and image-level bootstrap uncertainty answer different
questions. No new seed count or exposure defaults are inserted into an agent plan.

## Informative follow-ups and failures

Researcher prompts and contracts require six answers using existing proposal fields:
question, motivating evidence, competing explanations, support/rejection outcomes,
actionable connection to real localization, and complete-matrix feasibility. Recent
comparison outcomes and missing work are supplied alongside cumulative evidence.
A synthetic diagnostic must explain how its possible outcomes change the next
scientific action. No hardcoded architecture, forced literature search or diagnostic
quota is introduced; usefulness remains a scientific judgment requiring live review.

CPU import/test failures retain complete stdout/stderr in the existing runtime
probe directory. Repair messages include the decisive tail, actual exception and
traceback. Production job logs and checkpoint integrity checks remain unchanged.
Predictable shape/metric/per-image mistakes can be caught by CPU tests and the
existing disposable queued smoke before a full training loop. Such checks cannot
prove learned behavior or eliminate every possible runtime/scientific failure.

Existing verified real-feedback caching, CPU concurrency and immutable input/version
cache keys are unchanged. A diagnostic-only correction reuses a verified frozen
checkpoint; it does not retrain a model or manufacture new independent evidence.
No CUDA performance or scientific improvement is claimed by these offline tests.
