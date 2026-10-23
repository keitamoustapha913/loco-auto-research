# Frozen diagnostic storage and state integrity

Maintenance following campaign `ca4f9dbf-7a61-4883-9bd5-9ef2c0fd0721`.

## Disk rejection

The previous guard sums logical file sizes across the **entire runtime**, not
just one job or sandbox. `LOCO_DISK_BUDGET_GIB` defaults to 100 GiB; no override
was present in the live environment file. The failed event recorded only the
generic rejection, not the exact historical process environment or byte count.
The post-campaign read-only inventory measured approximately 100.39 GiB:

- jobs: 94.184 GiB, including 86.548 GiB of full-resolution feedback maps;
- controller: 3.576 GiB;
- MLflow: 2.006 GiB;
- the remainder: authorized datasets, incoming archive, logs and configuration.

Four historical exact-feedback reuse records duplicated 3,379,284,480 bytes
(3.147 GiB) of maps. The old `copytree` reused computation but duplicated storage.
The blocked comparison recorded zero jobs: this rejection was based on existing
runtime usage, not an inflated diagnostic reservation. Filesystem free space
does not override this quota. No evidence established stale temporary files or
an accounting error as the cause.

The protection remains in force, with no quota increase or historical deletion.
`enforce_disk` reports the runtime root, scope, observed logical bytes, required
additional bytes, limit, filesystem free bytes, policy and actionable cause.
Before dispatching any cell, plan execution checks the complete pending frozen
matrix's concrete checkpoint-copy bytes. This is a lower bound, not a guarantee
that unknown future outputs or concurrent writes will fit; each job retains its
existing guard.

New exact-feedback reuse copies only per-run metadata and references the already
verified prediction directory. Every reuse hashes its maps against the recorded
case identities. There are no hardlinks or symlinks, and no original prediction
directories are mounted into candidate processes. The viewer resolves these
references with the same path/hash/alignment checks. MLflow retains each run's
own summary and score metadata. Reuse references point directly to the original
map directory, rather than forming dependency chains.

New metadata is published atomically; failed copies remove only their new staging
directory. Existing interrupted outputs and historical evidence are untouched.
The live runtime therefore **remains above the default cap**: this maintenance
does not authorize storage removal, archival or an increased quota.

## BatchNorm digest

The saved failing assertion called the **candidate's** `_state_digest`, not the
trusted hasher. The campaign's second Implementer response explicitly reported
object-ID reuse and repaired it by retaining strong references during traversal.
Its final immutable snapshot `eabe974` contains that repair; the ledger records
six passing candidate CPU tests afterward. The original pre-repair source was
not separately snapshotted, so its exact byte-level implementation cannot be
replayed. Historical candidate source and tests have not been edited here.

Independent pre-fix CPU fixtures also exposed trusted-hasher gaps: non-persistent
registered buffers and module mode flags were omitted. The trusted hasher now:

- retains strong references, preventing recycled temporary-wrapper IDs;
- includes state_dict parameters, persistent buffers and extra state;
- includes registered non-persistent buffers and public module state/modes;
- continues hashing wrapper tensors, arrays and inspectable metadata, rejecting
  uninspectable objects.

An inexpensive CPU integrity self-check runs before candidate import in the
production interface probe. Unchanged BatchNorm inference must pass; deliberate
running-statistic, counter and non-persistent-buffer changes must be detected.
Failure here has no candidate-result marker and follows trusted-infrastructure
error handling, rather than candidate repair. Candidate test failures retain
their existing actionable traceback and candidate-owned repair path. Implementer
instructions point to the trusted digest and require buffer-mutation tests.

## Offline verification

- Before correction, deterministic fixtures failed for non-persistent buffers and
  mode changes. After correction all eight focused state tests pass.
- A bounded-quota fixture rejects the old full-map copy requirement but admits
  reference-based reuse, with unchanged map hashes and mutation detection.
- Genuine quota/free-space exhaustion still rejects before writes. A three-arm
  frozen fixture dispatches all cells without learning when within quota and
  records INCOMPLETE with zero new jobs when the full copy lower bound exceeds it.
- Failed metadata publication removes its staging directory without modifying
  original artifacts. Reference escapes are rejected. Full-resolution viewer
  alignment remains tested at 800 by 1600.
- `LOCO_TEST_BWRAP=1 .venv/bin/pytest`: **618 passed, no skips**, on the host.
  The outer tool sandbox blocks Bubblewrap's NETLINK_ROUTE setup; it is not a
  production validation success and was followed by the passing host run.

No Codex calls, GPU work, campaigns, real-development evaluations, live
configuration changes or scientific-model modifications were performed.
The seed-607 exploratory incumbent and its checkpoint/metadata hashes remain
unchanged. Training-loop timing, campaign deadlines, scientific scoring and
localization-first selection are unchanged. The compatible scorer's numerical
version is unchanged; the existing worker-source hash continues invalidating
cache identities when trusted inference implementation changes.
