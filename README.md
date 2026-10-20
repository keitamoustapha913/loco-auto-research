# LOCO Autoresearch

A normal-only weight-training laboratory with anomaly-informed research selection.
SQLite owns experiment state; MLflow tracks measurements/artifacts; Streamlit displays them.

## Daily workflow

Install once with `uv sync --extra all`. Then:

```bash
uv run --extra all loco-lab start
uv run --extra all loco-lab research start --config APPROVED_CONFIG
```

`APPROVED_CONFIG` is a private operator-owned campaign YAML with reviewed budgets and explicit
shared-UID Bubblewrap risk acceptance. Templates remain **unauthorized**. Startup does not grant
campaign authorization. `configs/autonomy_real_feedback.yaml` proposes two serialized jobs,
120 seconds each, four GPU-slot minutes, one wall hour and eight Codex calls.

Codex calls have two separate ceilings: the campaign YAML's `max_codex_calls`
and the lifetime ledger allowance `LOCO_CODEX_MAX_CALLS` (default 30). A new
campaign does not reset the global allowance. `loco-lab research status` shows
both counts, limits and remaining allowances, plus completed and failed agent
invocations. Both reservations are admitted atomically; a rejected request
consumes neither allowance and stops the worker as `BUDGET_EXHAUSTED`, without
retrying or counting a research failure. Admitted calls remain charged on CLI
failure or worker crash; a crash between admission and invocation is retained
conservatively rather than refunded or replayed. Historical reservation counts
are preserved and explicitly labeled legacy where they may include blocked
requests. Changing an operator ceiling requires separate authorization.

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
