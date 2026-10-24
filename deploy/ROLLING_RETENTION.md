# Rolling scientific artifact retention

`rolling-nine-v1` is trusted housekeeping, not a research-selection change or an
operator budget. The existing 60/20/10/10 research score selects six strongest
eligible, intact model-producing runs. Current research incumbents are mandatory
members of that group; three additional latest model-producing runs fill nine
unique slots. When fewer than six eligible models exist, latest trained runs fill
the remaining slots. Toy infrastructure runs and frozen diagnostic evaluations do
not consume slots. Scores across different evaluation cohorts are storage
priorities, not evidence of scientific superiority between protocols.

The ledger's confirmed champion and active/unfinished comparison dependencies
are separately protected. These can temporarily exceed nine; the cleanup report
explicitly lists conflicts. Completed comparisons in an active multi-day campaign
do not protect every historical job for the entire campaign. The current
incumbent and scientific confirmation are never modified by storage maintenance.

Outside the selected/protected set, terminal ledger-owned runs retain source,
configuration, original hashes, scientific records, numerical results, critiques,
negative evidence and per-image measurements. Known checkpoint/image/array files
in job roots, retained archives and locally associated MLflow artifact roots are
retired. Textual evidence larger than 2 MiB is losslessly gzip-compressed, with
byte-for-byte restoration checked before removing its uncompressed copy. The
trusted bounded observation reader supports compressed text. Dataset roots,
authorizations, manifests, SQLite and MLflow databases are never deletion roots.

Each retired run has `artifact_availability.json`, a durable ledger setting and
an extra `scientific_record.json` snapshot. Original records and scores remain
unchanged. Checkpoint hashes describe historical bytes, not current availability.
The dashboard labels metrics-only records, MLflow receives the availability tag
and manifest through its normal outbox, and frozen-baseline admission rejects
retired checkpoints explicitly. Full-resolution viewers cannot display retired
maps. Canonical map chains needed by selected runs remain protected even if the
canonical owner's checkpoint is retired.

Retirement uses the existing storage/GPU/feedback locks and additionally takes
the tracking lock. Plan validation/registration shares the storage lock, so new
baseline dependencies cannot be published during retirement. Symlinks and unsafe
paths are refused. A durable prepare record and availability status precede
unlink; each file has a hash and deletion audit. Interrupted passes are safe to
repeat. Missing provenance causes preservation and a reported exception, never
speculative deletion. Losslessly compressed evidence can be restored using gzip;
retired checkpoint/image bytes cannot be recovered by this mechanism.

Existing startup, completion/abandonment, approximately 30-minute tracking-worker,
pre-admission and campaign-end cleanup hooks run retention even without quota
pressure. Checks are bounded and serialized; there is no new service or setting.
Existing running Python processes load the updated implementation on their next
normal restart. The one-time migration invokes the same production cleanup.
The 100 GiB quota and existing admission rejection remain unchanged. Lightweight
scientific history necessarily continues to grow; nine full runs is not a promise
of unlimited storage for arbitrary model sizes or separately protected records.

## Initial verified migration (24 October 2026)

Before: 79.565 GiB. Removed bulky bytes/losslessly compressed evidence reclaimed
64.774 GiB: 54.282 GiB images/arrays, 40.69 MiB checkpoint copies and 10.452 GiB
text compression. No temporary-file reclamation was needed. After tracking sync:
14.848 GiB used, 85.152 GiB available under the unchanged 100 GiB quota.
156 runs were marked metrics-only. Ten ambiguous pre-manifest directories
(3.32 MiB total) were preserved. No active/confirmed retention conflicts existed.

| Fully retained run | Arm | Seed |
| --- | --- | --- |
| ffce29eb-b48d-5ee4-8ddd-f39ab6e2a4c0 | GroupNorm | 461 |
| 0891b9a1-5f10-5611-80a2-cb9d1398e54b | Constant LR, active incumbent | 607 |
| 9d351041-b8d6-51b8-bb25-171e8bddeb66 | BCE replication | 653 |
| c0061881-7f9b-50fa-89e5-e4e32a0ef47a | Annealed LR | 607 |
| 0a2d2925-1506-5a70-aff5-5581bca927a1 | Overlap objective | 503 |
| fa1809f5-ab5c-5246-83fe-48b83e130d68 | Dice replication | 653 |
| 4b75735b-dc89-5747-8fe3-32b33195e55b | Rectangle edits, recent | 727 |
| 7cc5a119-29f1-5ba6-897e-f3322bd8d470 | Square edits, recent | 727 |
| c937b849-deb1-5655-8240-f5541d009639 | Learned affine, recent | 701 |

The migration evidence is under
`$LOCO_RUNTIME/controller/rolling-retention-migration-20261024-123147/`:
`preview.json`, `before.json`, `passes.json`, `result.json`, `integration.json`.
All nine immutable model archives and 1,485 retained maps were hash-verified
before and after cleanup. Historical run/configuration/metric fingerprints and
incumbent references were unchanged; SQLite integrity passed. All 156 archived
MLflow availability tags and dashboard history labels were checked, with zero
pending tracking events. Only the existing tracking worker was reloaded to
activate the new periodic implementation; no research was started.
