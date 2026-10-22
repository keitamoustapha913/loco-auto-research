# Training throughput audit — 22 October 2026

## Scope and historical evidence

Campaign `37a0577a-be18-481a-98e7-8b2bd021a6d6` ran for eight hours.
Its first job `b95531d3-c33e-5132-8f43-a61bc93616d7` occupied approximately
27,982 seconds, recording 237.784799 learning seconds and 146,939 updates.
The six-cell comparison remained INCOMPLETE. No completed checkpoint or
real-development score resulted. Neither historical records nor the incumbent
were changed by this maintenance.

The exact historical source is commit
`ed65e8ae60b815a8fee6ab038c442b81c4701739`, file
`candidates/seam_censoring_continuing_controlled_redesign_v3/model.py`, SHA256
`218a45f056c1728d56e9b25a0f7fd399b316c7d238ec4bc6eb6e1657b299211a`.
Its `_read` decoded and resized every selected original image at every update.
It also regenerated synthetic views and all three supports, hashed all input/
target/support tensors, and hashed model state around preparation. These latter
operations remain intact because they enforce scientific controls/integrity.

## CPU measurements on the actual candidate and permitted data

Two CPU threads; original 800×1600 normal images, batch 8, 128×128 preprocessing,
32 indexed updates, seed 367, matched initial weights. Manifest SHA validation was performed before access.
No development anomalies, annotation masks, CUDA jobs or agents were used.

| Stage | Original mean milliseconds/update |
|---|---:|
| Image opening | 0.779 |
| Decode and RGB conversion | 158.471 |
| Bilinear resize | 18.343 |
| Float conversion/normalization | 0.511 |
| Batch stacking | 0.190 |
| Synthetic preparation and masks | 5.238 |
| Input/target/support SHA256 | 1.608 |
| Model fingerprint, twice | 0.144 |
| CPU forward/loss/backward/Adam | 88.618 |

Repeated decode/resize was the dominant measured preparation cost. Its roughly
178 ms matches the historical roughly 190 ms elapsed per update, whereas the
historical synchronized CUDA learning averaged approximately 1.62 ms/update.
This agreement supports the diagnosis; it is not a retrospective CUDA profile.
Physical GPU transfers, device synchronization and GPU kernel time were **not
measured during this maintenance**, because GPU execution was not authorized.

Matched repeated CPU comparison (same source-index/edit streams):

| Stage | Original ms/update | Cached ms/update |
|---|---:|---:|
| Image preparation | 177.009 | 0.227 |
| Synthetic preparation/masks | 5.412 | 5.123 |
| Stream SHA256 | 1.610 | 1.643 |
| Model fingerprint, twice | 0.142 | 0.143 |
| CPU learning | 91.813 | 90.663 |
| Sum of measured stages | 275.986 | 97.799 |

The steady measured CPU stages improved approximately 2.82×. Initial cache fill
cost **6.700 seconds**, storing **52.6875 MiB** for 281 fitting images. This
one-time cost is additional and excluded from learning. The cache has a fixed
128 MiB capacity; oversize entries are not retained. No cached pixels are
serialized or retained by inference. Both timed runs used seed 367 and identical
initial weights; their final weights also matched bit for bit after 32 updates.
No performance claim is made for CUDA yet.

## Changes and scientific equivalence

- `DecodedTrainingCache` reuses immutable decoded pixels keyed by source path,
  verified source hash and preprocessing size. It is scoped to a training
  generator. Original reading/resize/normalization code remains unchanged.
- The existing candidate workspace preloads its finite permitted fitting set
  before learning and stacks independent writable batches in the original indexed
  order. Stochastic edits are regenerated every update with unchanged seeds.
- The performance-only candidate revision is local commit `22dfd1d` on the existing
  `loco-candidate/seam_censoring_continuing_controlled_redesign_v3` branch.
  The historical source commit and frozen plans are untouched. The optimized
  workspace has a new source hash and requires a new explicit plan/source pin;
  it cannot silently replace the source of a frozen comparison.
- Six CPU equivalence checks: full/random/censored × seeds 367/373, eight genuine
  updates each, against exact historical source. All final tensor values,
  initial-state hashes, indexed source/edit digests, entire-stream hashes and
  counters matched **bit for bit**. Each saved technical-fixture checkpoint hash
  was checked. These fixtures are not scientific results.
- Optimizers, losses, architecture, masks, sampling, learning schedule and the
  900-second required target are unchanged. Existing mutation/early-stop rejection
  tests continue to pass. No weaker hash or sampled integrity check was introduced.

## Timing and admission

The production clock now records preparation, integrity checks, validation,
audit-file I/O, setup synchronization and learning synchronization separately.
Setup synchronization is outside learning; completion synchronization remains
inside. The last 64 steady update samples are bounded; the first four updates
are excluded from steady throughput. Initialization/finalization are separate.
No profiler bookkeeping or preparation is added to learning duration.

Historical admission now includes failed/cancelled partial training instead of
considering only successful short runs and adding their overhead once. It scales
observed elapsed/learning ratios to the full required target. Same-source/device
observations are distinguished from preliminary estimates for new implementations.
The old source's measured rate rejects six 900-second models within eight hours.

New v3 queued disposable preflight measures **32 updates at the actual pilot
batch, fitting split and preprocessing**; a batch-one/two-image fixture cannot
certify a batch-eight pilot. After diagnostics, the host checks source/device/
batch/resolution and projects the entire remaining arm×seed matrix, setup,
feedback and critique with a 25% throughput uncertainty margin. This is scheduling
estimation, not another execution timeout or a guarantee of future throughput.
An unfit matrix returns recoverable admission evidence, keeps the comparison
INCOMPLETE and cannot run its full training targets. Smoke weights are discarded.
The original campaign deadline and full actual-learning target remain enforced.

## Verification and readiness

Regression: **518 passing tests**, including CPU Bubblewrap fixtures; no skipped
tests in the full run. The production candidate CPU validator also checks the
candidate's existing 15 tests against the current trusted modules. Offline tests
cover cache immutability/eviction, source streams, profiling exclusions, malformed
or mismatched throughput, cancelled history, full matrix projection and recoverable
admission without starting jobs.

**A six-job CUDA comparison is not yet demonstrated feasible.** Its irreducible
learning alone is 90 minutes. The cache removes a measured repeated cost, but CPU
learning throughput cannot establish GPU end-to-end throughput. A separately
authorized real-candidate RTX 4090 benchmark must use batch 8, size 128, the actual
fitting sources, all arm semantics, warm cache, original RNG/hashes and production
integrity/timer paths. It should measure transfers, synchronization, preparation,
learning and end-to-end rate over enough updates to assess stability, then include
initialization, checkpoint/diagnostic and feedback costs for all six cells. No
such benchmark or new campaign was launched during this task.
