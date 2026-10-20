# LOCO Autoresearch

A normal-only weight-training laboratory with anomaly-informed research selection.
SQLite owns experiment state; MLflow tracks measurements/artifacts; Streamlit displays them.

## Daily workflow

Install once with `uv sync --extra all`. Then:

```bash
uv run --extra all loco-lab start
uv run --extra all loco-lab research start --config APPROVED_CONFIG
```

`APPROVED_CONFIG` is a private operator-owned YAML with the existing dataset,
research, tracking and security settings. The normal template is `configs/autonomy.yaml`;
it remains **unauthorized**. Service startup never authorizes research.

The only research budget choices are elapsed campaign time and desired training time per model:

```yaml
campaign:
  name: juice-bottle-research
  dataset: juice_bottle
  mode: autonomous
  gpu_device: cuda
  start_from: previous_best  # or new
limits:
  research_hours: 4
  training_minutes_per_experiment: 15
```

Keep the existing `research`, `tracking` and `security` sections in your private
configuration. Omitted sections inherit the safe template defaults; shared-UID
Bubblewrap risk acceptance remains false unless explicitly authorized by the operator.

- **Research hours** count all elapsed time: agents, implementation, training, CPU
  evaluation, critique, waiting and pauses. Restart/resume does not reset the clock.
  Each independently authorized new campaign receives its own clock.
- **Training minutes** count only prepared model forward/loss, backward and optimizer
  updates. Model/optimizer initialization, batch loading/preparation/transfers, checkpoint
  saving, coding, calibration, evaluation and critique are excluded. CUDA setup work is
  synchronized before timing; queued learning kernels finish inside the measured phase.
  Codex chooses steps, epochs and exposures and must justify its estimated training
  duration or a shorter schedule. This is a ceiling/target, not fabricated training time:
  convergence or a justified experiment can finish earlier. Actual duration is recorded.
- **Previous best** verifies eligible retained references for the same dataset and
  supplies accumulated evidence. Incompatible evaluation cohorts stay separate.
  Missing eligible references cause a clear error; choose `new` for an independent
  start. Weights are reused only when explicitly requested by a validated plan.
- **New** creates independent campaign state without replacing retained models,
  deleting history or automatically starting from their weights. Past findings remain
  available as scientific evidence.

The existing scheduler serializes jobs. Trusted code enforces separate training and
whole-job deadlines, reserves time for complete comparisons, normal-only CPU feedback
and critique, and stops rather than reducing arms, seeds or training settings. Frozen
scientific plans retain their original schedules and resource ceilings. Incomplete work
stays INCOMPLETE and cannot support promotion. Three consecutive operational failures
still stop a campaign; disk limits, sandbox gates and bounded repairs remain in force.

**Migration:** replace the entire old `limits` block with the two fields above; do
not mix formats. Existing seven-limit authorizations are still accepted **as legacy
bounded studies**, with their explicit limits and global ceilings unchanged. They are
never silently expanded or rewritten. New time-based campaigns neither require nor
consume the lifetime allowances in `LOCO_GPU_BUDGET_HOURS`,
`LOCO_JOB_TIMEOUT_SECONDS` or `LOCO_CODEX_MAX_CALLS`; historical counters are preserved
for auditing. Those environment variables still govern legacy/standalone commands.
No environment-file edit is required for a new-format campaign.

Home, Results and campaign reports show elapsed research and measured training time.
Technical GPU-slot/call counters remain available in System diagnostics and ledger
records. Training timing protocol is recorded in `training_timing.json`, including
failed/time-limited learning attempts, and synchronized through the existing outbox.

### Training-only candidate contract

Time-based learning requires `training_steps(config, allowed_train_manifest, output_dir)`:
a generator that yields `scripts.training_phase.TrainingStep` objects. Initialize the
objective (`torch.nn.Module` returning a scalar differentiable loss) and optimizer before
yielding, prepare device-resident inputs between yields, and save checkpoint/metadata
after the final yield. Return checkpoint metadata from the generator. The trusted loop
runs forward/loss, backward and the optimizer update; it never times generator setup or
finalization. Closure-based optimizers are supported through explicit `use_closure=True`.
The existing train/load/predict contract is retained for legacy callers and inference.

The worker rejects I/O, model/optimizer creation within the learning step and optimizer
updates or gradient-enabled training forwards outside it. Inputs must be prepared tensors
and scalars, not file paths/loaders/callbacks. These are checked Python contract boundaries
inside the existing sandbox, **not** a new OS security boundary against malicious Python.
Candidates with unsupported/custom training procedures must expose a verifiable contract
or report incompatibility; the controller does not rewrite scientific schedules.

Missing `training_steps` is **UNVERIFIABLE** and rejected before admitting a time-budgeted
GPU comparison. Historical candidates/checkpoints remain untouched. Frozen inference
needs no training interface. Legacy unbounded-by-training-time calls are explicitly marked
`verified: false` (`legacy-candidate-train-call-v1`); their wall duration is recorded as `training/function_wall_seconds`, not
`training/duration_seconds`, and is not a verified training-only measurement. New verified measurements use `prepared-training-step-v1`.

Within that authorization, Codex investigates, implements, trains on good images, receives
frozen-model real-development measurements, critiques and selects its next experiment.
There are no separate worker, scorer, dashboard or MLflow-sync startup commands.

```bash
uv run --extra all loco-lab status
uv run --extra all loco-lab research status
uv run --extra all loco-lab research pause
uv run --extra all loco-lab research resume
uv run --extra all loco-lab research report
uv run --extra all loco-lab research stop
uv run --extra all loco-lab logs
uv run --extra all loco-lab stop
```

Use `doctor` for diagnostics and `smoke --dataset juice_bottle --device cuda` only for an explicitly
requested infrastructure test. Infrastructure tests are excluded from scientific rankings.

## Data and evaluation

- Learning: 335 `train/good` images, with permitted normal-derived augmentation.
- Normal validation: 54 images; reproducible disjoint 18-image threshold calibration and 36-image assessment.
- Frozen real-development feedback: 47 good, 71 logical and 47 structural images; 124 masks.
- Split: `loco-feedback-v1.0-f91c7f27aad59de58c5a`.
- Final holdout stays off-server. No final-holdout import or research feedback is supported.

Real anomalies never fit weights, losses, normalization, thresholds, memory banks or test-time
adaptation. Aggregate scores may guide architecture/hypothesis selection. Masks are read only
by trusted scoring; each no-network candidate inference process sees one opaque image.
Checkpoints and predictor state are integrity-checked. Shared-UID isolation and in-process
Python guards are not a proof against malicious code; retain source review and sandbox checks.

`benchmark/feedback_metrics.py` is the versioned real-development scoring entry point:
detection AUROC/FPR, separate logical/structural AU-sPRO@5%, and supplementary logical pixel AP/AUROC.
The sPRO scorer is a **compatible development implementation** of the published definition.
The original official archive still returns 404; original-code equivalence is unverified.
It retains individual mask/defect saturation semantics. Pixel AP does not replace sPRO.

Real results are labeled **LOCO custom-split development feedback — not official full-test results**.
The four pages remain Home, Research, Results and System. Results separates real measurements
and full-resolution heatmaps from optional historical synthetic diagnostics. Missing scores
are unavailable, never zero. New campaigns disable synthetic diagnostics by default;
`research.synthetic_diagnostics: true` enables them for a justified research use.

CPU feedback reports batch-one p50/p95 decode→preprocess→prediction→map-restoration latency,
parameters, serialized bytes and process RAM. It does not claim CUDA inference. The existing
CUDA training runner measures synchronized normal-image inference and resets inference peak
memory independently of training. Measurements state their device/protocol.

## Configuration and access

The CLI safely loads `../loco-research.env` (or `LOCO_ENV_FILE`); explicitly set environment variables
win. Existing runtime paths, ports, resource ceilings and CUDA packages are preserved.
The development archive has already been imported; `loco-lab feedback status` verifies it.
The optional user service is `deploy/systemd/loco-lab.service`; no host service is enabled automatically.

From the laptop:

```bash
ssh -N -L 8501:127.0.0.1:8501 -L 56000:127.0.0.1:56000 theyeq-admin-pc@100.91.233.99
```

Services run on the server; open http://127.0.0.1:8501 in the laptop browser.
MLflow remains at http://127.0.0.1:56000.

Run regression tests with `uv run --extra all pytest`.
See [cleanup/evaluation report](results/reports/final_workflow_cleanup.md) for exact metric definitions,
measured integration results, removals, historical compatibility and remaining limitations.

### Candidate validation and repair

Every new candidate supplies `test_model.py`. CPU validation runs inside the existing
no-network Bubblewrap boundary without GPU devices. Tests must exercise applicable
split rejection, input validation and output/shape logic on CPU. Missing tests and
suites with no successful test bodies (all skipped or xfailed) are rejected. CUDA
skip decorators belong on individual GPU tests rather than the entire test class.

For autonomous CUDA pilots, the queued job first launches a disposable CUDA
train/load/predict check on two authorized `train/good` images, requesting two
optimizer steps, one epoch and batch size one at the pilot preprocessing size.
The smoke stage has a maximum 30-second deadline within the original job timeout.
It verifies finite aligned predictions, checkpoint/state integrity and observed
CUDA allocation/parameter placement. These checks are protocol diagnostics, not
a proof of arbitrary candidate behavior. Smoke weights are never used for pilot
training or scientific feedback. The pilot starts fresh only after smoke passes.
Both stages hold the same scheduler lock; successful, failed, OOM and timed-out
smoke work is charged as GPU-slot usage and retained in SQLite/MLflow.

The Implementer receives exact static/CPU failure output for one repair in its
assigned workspace. A registered scientific comparison freezes its implementation:
a failed CUDA smoke, pilot, or diagnostic remains a charged failed cell and makes
the comparison INCOMPLETE. It is never silently repaired into a different model
inside a partly executed comparison. The Critic receives the exact failure; Codex
can propose an explicitly linked repair/completion or replication plan within the
remaining campaign allowance. Interrupted cells are never silently replayed.
Permission/dependency failures remain approval gates. Historical campaigns and
candidate snapshots are not rewritten.

Proposal generation also has one bounded correction: the Researcher sees prior
hypotheses, and a rejected batch is returned with the precise duplicate/ranking
error. The whole corrected batch is revalidated before any hypothesis is saved;
renaming an existing mechanism does not evade deduplication. Rejected responses
and repair requests remain in the ledger. Extra generated files within the
assigned candidate directory can consume the Implementer's existing single
repair attempt: the agent must remove them and put licensing notes in an allowed
README/manifest. The file allowlist is unchanged. Changes outside the candidate
directory, symlinks and unapproved dependencies remain hard failures. These
corrections use normal Codex reservations and never retry a denied budget request.

Self-correction is applied at agent-response and generated-work boundaries, not
by matching known error messages. All roles can correct malformed structured
responses. Researcher and Implementer corrections share their existing single
retry with downstream validation, covering unfamiliar validation exceptions,
syntax/interface errors and CPU test failures before comparison execution.
Failures carry their phase, exception type, traceback and rejected output back
to the responsible agent. Missing or malformed candidate result artifacts are
recorded as charged failed runs and reported to the scientific comparison review. Permission,
authorization, budget, ledger, host filesystem and CLI transport failures are
not instructions for agents to change infrastructure; they stop safely. This
mechanism provides bounded correction opportunities, not guaranteed recovery
from every possible failure.


### Scientific experiment plans

Codex supplies the scientific design; the trusted controller executes it. New
proposals require an immutable `experiment_plan`. Historical prose-only proposals
are preserved but cannot be silently selected or converted into new experiments.

The canonical contract is [experiment_plan.schema.json](schemas/experiment_plan.schema.json).
The CLI's closed agent-output representation is
[experiment_plan_agent.schema.json](schemas/experiment_plan_agent.schema.json):
scientific settings use `{name, value_json}` pairs, decoded without changing values.
These representations are checked together by regression tests.

A plan specifies hypothesis/plan IDs, an explicit baseline arm, arms and intended
differences, seeds, training settings/overrides, prediction settings, controlled
variables, required metrics, diagnostics, paired comparisons, resource ceilings,
and completion/failure criteria. It may request built-in synthetic evaluation,
trusted pinned real-development evaluation, or both; requested measurements are
never silently disabled. Unsupported measurements remain missing evidence.

Every requested arm × seed cell receives a stable run UUID. The queue preserves
its complete settings rather than inserting seed 7, one epoch, or a default
prediction mode. Missing/failed/interrupted cells are retained and not replayed.
Job creation and campaign ownership/count admission are atomic. Comparison
completion, job completion, and scientific hypothesis support are distinct:

- `INCOMPLETE`: an arm, seed, setting, counter, measurement or diagnostic is missing,
  failed or incompatible; budget stops also preserve this status.
- `COMPLETE_EXPLORATORY`: the declared screening work was measured.
- `COMPLETE_CONTROLLED`: the declared comparison was measured; this alone does not
  prove causality, significance, generalization or hypothesis support.

Frozen baselines reference exact run IDs and hashes of committed source,
checkpoint, checkpoint metadata and original configuration, including the seed.
The controller loads the committed source even when its old workspace changed;
it never reconstructs a model from its historical name. Frozen jobs do not call
`train`. Explicit prediction interventions require a supported `inference_config`
and acknowledgement; unsupported modes fail rather than reverting to defaults.
A shared pinned diagnostic implementation can inspect frozen baseline predictions
without changing baseline model source or checkpoint.

The runner independently counts PyTorch optimizer-step calls. Candidate loops must
record training-audit seed, update and source-exposure counts; counts must match
the declared plan. **Source exposures remain candidate-instrumented**, not an
OS-enforced counter. Model-independent software cannot prove that an arbitrary
Python implementation scientifically realizes a hypothesis; source review and
meaningful candidate tests remain necessary.

Diagnostics run under inference mode with frozen-state/checkpoint checks, on
separate normal assessment images and permitted good-derived synthetic controls.
They receive no real anomaly images or annotation masks. Optional paired
comparisons require identical per-image source IDs and use the existing fixed-seed
image bootstrap. Intervals across images are distinct from training-seed variation;
image independence must be justified, including donor-sharing risks. Full-resolution
real development scoring retains the existing compatible scorer and holdout boundary.

The ledger stores registration, exact job settings, source/evaluator/control-code
hashes, ordered data identities, training counters, failures, coverage and critique.
Changing code or source order during a comparison makes it incomplete. MLflow
receives training audits, diagnostic evidence and the complete comparison artifact
through its existing durable outbox. Result cards show comparison completeness.
The Critic cannot promote an incomplete comparison or claim controlled causality
from exploratory work. Codex receives committed relevant prior code/configuration,
failed attempts, recent evidence and strong historical results through narrow
read-only mounts. Explicit follow-up lineage permits completing or replicating a
hypothesis without disguising it as a new method.

Daily operation is unchanged:

```bash
uv run --extra all loco-lab start
uv run --extra all loco-lab research start --config APPROVED_CONFIG
```

The same status, pause, resume, report and stop commands apply. An authorized
campaign may execute complete comparisons automatically within its original
limits. Updating this code does not authorize or start a campaign.

### Cumulative localization improvement

The trusted controller retains frozen real-development models across campaigns.
An **exploratory incumbent** is a recoverable model, not a claim of confirmed
superiority. A **supported promotion** requires complete controlled evidence.
Normal-only weight learning, normal-only calibration, inference-only feedback,
the final-holdout boundary, sandbox and authorization budgets are unchanged.

`configs/champion_policy.json` is operator-owned and unavailable to candidate
writers. Its hash and incumbent revision are frozen when a plan is registered.
Changing it requires a new comparison; it cannot retroactively qualify evidence.
The initial conservative policy requires:

- At least three fixed challenger training seeds and the exact retained checkpoint
  among the baseline evaluations. A baseline can have matched independent training
  seeds for confirmed promotion. An explicitly declared `fixed_checkpoint_seed`
  remains an exploratory comparison: pairing indices never fabricate independent
  baseline training. Pin original source, checkpoint and configuration references.
- Identical development samples, split, evaluator, normal-training fingerprint,
  inference device, timing protocol and threshold-calibration protocol.
- Positive logical AU-sPRO@5% **and** pixel-AP differences in every paired seed,
  with positive lower bounds from a fixed 95% paired-seed bootstrap. The sPRO
  implementation remains **compatible development**, not verified official code.
- No increase in normal false positives versus the matched baseline, and at most
  10% normal FPR (at most 4/47 on this split). At most 10M parameters. No more than
  10% relative latency or measured process-memory regression; CPU timing/memory
  are never represented as GPU measurements. Logical detection and structural
  localization may each decline by at most 0.02 absolute per paired seed.
- A representative checkpoint chosen by the first fixed seed, never the best
  observed seed. Its localization must not regress the retained checkpoint.

These defaults are conservative deployment rules, not scientific laws. A promising
trade-off remains a Pareto alternative even when it cannot replace the incumbent.
The initial historical incumbent is the highest AU-sPRO candidate meeting the FPR
and parameter ceilings, with pixel AP breaking ties. No historical single-seed
result is upgraded to reproducible proof. Later record scores alone never replace
it. Promotion decisions and explicit trusted-operator rollback (`model_registry.rollback`)
are append-only; older checkpoints and promotion evidence remain available.

The fixed-incumbent option makes legacy single-seed checkpoints usable without
modifying their code or inventing audit hooks. These comparisons cannot confirm
champion promotion under policy v2. Baseline seed variation remains unknown; the
critic cannot make a causal method/architecture claim. Original training settings
are retained, and unequal training budgets must not be listed as controlled.

Different methods need not share a model implementation. Train them separately
under their declared plans, then use a multi-arm **frozen** comparison referencing
each exact checkpoint by seed. Replicate the incumbent using its byte-identical
source and original learning configuration, with additional seeds explicitly
declared; do not reimplement a historical method from its name. Existing frozen
arm support executes the final matrix without retraining either method.

The existing SQLite ledger records retention, immutable champion revisions,
promotion/rejection reasons, paired-seed uncertainty and rollback. Verified bytes
are copied under `$LOCO_RUNTIME/controller/retained_models/` with hashes and
owner-only parent directories. This is integrity checking and recovery storage,
not protection against the administrator's shared host UID or a disk backup.
MLflow receives retention provenance and cumulative decisions through its durable
outbox. Home and Results show the retained model and its confirmation status;
`loco-lab research status` and `report` include cumulative progress. Read-only
monitoring never changes the champion. Initialization preserves historical runs,
metrics and campaign counters. Crash recovery finishes deterministic promotion
bookkeeping without repeating training, inference or agent calls.

Before proposing methods, Codex receives a whole-history scientific-memory index,
retained incumbents, Pareto source references, failures, completed ablations and
incomplete comparisons. Detailed source/evidence snapshots are read-only. No
five-result window discards older findings. Codex retains scientific freedom;
these guards constrain execution and promotion, not architectures or hypotheses.

Uncertainty here measures **between-training-seed** variation on the same fixed
development set. Three seeds give weak distributional evidence; this is not an
image-population interval, multiplicity correction, or proof of generalization.
Repeated anomaly-informed method selection can overfit this development split.
Image-cluster uncertainty, additional normal acquisition or stronger replication
must be declared and measured when needed. No development promotion guarantees
improvement on the independent final holdout. Missing confirmation remains
unavailable, never a fabricated zero or a rising progress curve.

Home and Results include a retained-localization step chart built from ledger
retention events. Research attempts leave it flat until a supported promotion;
operator rollbacks are explicitly identified. The chart starts when retention
tracking began, with exploratory points in gray. Historical record scores are
not backfilled as verified promotions; one recorded incumbent appears as one point.

### Trusted scientific planning interface

`orchestrator/planning_contract.py` derives the researcher prompt and constrained
response schema from the same reserved settings and training requirements used
by trusted admission. Controller-owned fields (including device/category/seed)
never belong in agent training settings. `controlled_variables` contains actual
setting keys, with equal declared values and verifiable historical values.

The researcher receives exact frozen configurations, recorded inference options
and a static loader-signature assessment. A legacy loader without an explicit
`inference_config` argument requires its recorded empty inference configuration;
settings cannot be invented from a historical model name. Authentic multi-seed
references can support a controlled comparison; unrecorded historical training
settings are never filled from current defaults. Candidate implementation must
still pass the existing interface and audit checks before execution.

Full semantic checks run before proposal acceptance, registration and GPU queue
admission. Errors name their code, arm, baseline and offending keys/values. The
existing bounded correction mechanism returns these errors to the researcher;
trusted code never changes the submitted science. A valid exploratory plan may
execute within a separately authorized campaign but cannot confirm promotion.
`COMPLETE_CONTROLLED` describes exact execution coverage; the separate trusted
`scientific_eligibility` gate additionally requires matched training controls,
independent seeds and preregistered real-localization/deployment endpoints. Missing
work remains `INCOMPLETE`. Measured matching, immutable baseline integrity, paired
uncertainty and the frozen promotion policy remain necessary after admission.

Provider schema compilation runs a deterministic local preflight **before** a
campaign/global Codex call reservation. Planning setting names are plain strings;
trusted Python checks retain reserved-field rejection (including device/category
and plan_*), valid key names, controlled values, baseline integrity, distinct seeds
and promotion eligibility. The provider-only removal of uniqueItems never removes
the original output/semantic validator's uniqueness check.

The preflight scans nested schema branches/references, rejects lookarounds,
backreferences, unsupported/unreviewed keywords and invalid strict object shapes.
Schema defects are infrastructure errors and do not trigger an agent repair call.
Checks follow the [documented Structured Outputs subset](https://developers.openai.com/api/docs/guides/structured-outputs)
and saved provider errors. Passing these local checks does **not** establish actual
provider acceptance; that requires a separately authorized successful live call.

Diagnostic measurement names derive from `diagnostics[].required_metrics`. The
canonical request is `{source: "diagnostic", key: "diagnostic_id/metric_name"}`;
trusted decoding also accepts the exact declared ledger alias
`diagnostic/diagnostic_id/metric_name`. This is name canonicalization only: no
scientific settings, arm, seed, metric definition or budget is changed. The shared
contract emits/reads the storage prefix once. Unknown names, ambiguous aliases,
duplicate/conflicting requests and undeclared paired/output metrics are rejected.
Canonicalization is idempotent and occurs before plan hashing/registration;
original agent responses remain in the audit history, and offline admission reports
include the exact name changes and canonical plan.
