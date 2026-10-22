# Authoritative model training-loop duration — v4

New autonomous plans use schema `4.0` and `elapsed-training-loop-v1`.
The operator still configures only `research_hours` and
`training_minutes_per_experiment`; no live authorization is edited by migration.

## Why the eight-hour attempt failed

The historical v3 controller waited for 900 accumulated forward/backward/optimizer
seconds. Repeated image decoding and other preparation were excluded, so its first
job ran about 7h46m and accumulated only 237.8 learning seconds. It was cancelled;
the six-cell comparison remains INCOMPLETE. Its metrics and checkpoints are not
reclassified or overwritten. The prior throughput audit remains historical evidence.

## Explicit boundary, no inferred initialization

```python
from scripts.training_phase import TrainingLoopStart, TrainingStep
TRAINING_PROTOCOL = 'elapsed-training-loop-v1'

def training_steps(config, allowed_train_manifest, output_dir):
    # Initialize the chosen model/optimizer and finite good-only pixel cache here.
    objective, optimizer = initialize(config, allowed_train_manifest)
    yield TrainingLoopStart(objective)
    while True:
        batch = fetch_and_prepare_next_batch()  # Counts, including the first batch.
        progress = yield TrainingStep(objective, optimizer, (batch,), source_exposures=len(batch))
        if progress['target_reached']:
            break
    return save_checkpoint_and_metadata(output_dir, progress)  # Does not count.
```

The functions in this example are candidate-defined placeholders, not a prescribed
architecture, loss or sampling method. The trusted runtime drains initialization
CUDA work before starting its monotonic loop clock. It checks model state around
batch preparation and executes real differentiable forward/loss, backward and
optimizer updates. Their completion synchronization is inside the loop duration.

Batch I/O, synthetic masks, augmentation, transfers, integrity checks and loop
bookkeeping count. Initialization, cache fill, disposable preflight, final saving,
evaluation and critique do not. They all consume the original overall campaign
deadline. Sleep padding is rejected inside the loop. No automatic restart of finite
schedules or silent extension of frozen experiments is allowed.

After each synchronized update, trusted progress contains elapsed loop seconds,
target seconds, target_reached, update count and source presentations. At the first
update boundary reaching the target, the clock is frozen before the generator saves.
The final iteration can cause a small overshoot, which is recorded and checked against
that iteration's duration. Early return, missing boundary, invalid output or campaign
cancellation is not successful full training. Frozen readout/diagnostic jobs train
for zero seconds and cannot accidentally enter this loop.

## Metrics and compatibility

| Measurement | Meaning for v4 |
|---|---|
| `training/loop_elapsed_seconds` | Authoritative elapsed training loop |
| `training/duration_seconds` | Compatibility alias for loop elapsed, v4 only |
| `training/learning_seconds` | Synchronized mathematical-update time, diagnostic |
| `training/loop_overhead_seconds` | Loop elapsed minus mathematical-update time |
| `training/initialization_seconds` | Setup before explicit loop boundary |
| `training/checkpoint_seconds` | Generator finalization after the last update |
| `protocol/optimizer_updates`, `protocol/source_exposures` | Actual instrumented counters |

The existing SQLite/outbox records these measurements. MLflow adds a timing-semantics
tag for new v4 training runs. Streamlit prominently shows loop elapsed and research
elapsed; efficiency diagnostics are expandable. Historical arithmetic-only durations
remain under their original definitions and do not enter loop totals. Confirmation
cannot claim matched training controls against a historical arithmetic-target frozen
baseline; exploratory frozen comparisons remain possible with explicit limitations.

## Admission and throughput

For six new models, allocate 6×900 = 5,400 loop seconds (90 minutes), then add
initialization, saving, diagnostics, inference-only feedback, critique and iteration
overshoot estimates. Do not multiply 900 seconds by the old arithmetic/wall ratio:
preparation is already counted. Existing measured post-warmup throughput informs
efficiency; measured non-loop preflight overhead and past v4 stage costs inform
admission. CPU observations are not presented as CUDA rates.

An unfit complete matrix returns a recoverable planning error and remains INCOMPLETE;
the controller does not omit seeds/arms, shorten training, reset the campaign clock
or introduce a job/GPU/agent deadline. Ten-retry recovery and cancellation remain.

The bounded, immutable decoded-training-pixel cache from `bef74ea` / candidate
revision `22dfd1d` remains available to maximize useful updates. It does not store
features for inference or change stochastic edit generation, masks, seeds or order.
Existing scientific model sources are not silently converted to v4: a new candidate
implementation must explicitly implement the boundary and receive a new source pin.

## Verification limits

Scaled CPU tests cover counted batch work and updates, excluded initialization/
finalization/evaluation, early return, padding rejection, missing marker, overall
deadline cancellation, complete paired multi-seed dispatch, timing-labelled tracking
and rendering, and historical compatibility. A full CPU candidate worker runs inside
the existing Bubblewrap boundary. No live research campaign, real-development
evaluation or CUDA operation is authorized by these tests.

Six models can fit a four-hour campaign if measured non-loop research, implementation,
setup, evaluation and critique fit in the other approximately 150 minutes. This is a
conditional admission estimate, not a guarantee. A separately authorized short RTX
4090 verification of the new boundary and clock remains necessary; no 15-minute
training session is needed to validate it.
