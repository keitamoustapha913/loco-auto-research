# Automatic storage housekeeping

The existing 100 GiB runtime quota remains enforced. No operator settings or
services were added. Normal startup and research commands are unchanged.

Housekeeping checks at controller startup, approximately every 30 minutes via
the existing tracking worker's idle loop, after each
finalized experiment/comparison, before storage admission rejection, and at
campaign exit. It aims for 20 GiB quota headroom when safe. Each pass is bounded
to about 30 seconds/1 GiB of source map processing; GPU/feedback activity makes
the pass defer rather than block. This is a maintenance work bound, not a new
scientific experiment deadline. A persisted due time and process-local fast
check avoid repeated inventories. Unchanged incompressible maps are remembered
by inode, size and nanosecond modification/change times rather than recompressed.

## What is reclaimed

- **Exact duplicate evaluation maps:** recorded per-map hashes index candidates;
  actual duplicate and canonical bytes are verified before replacing the copy
  with a small `map.npy.ref.json` reference rooted inside runtime/jobs.
- **Unique derived maps:** an uncompressed NPY may become `map.npy.bs4.gz` (a
  reversible four-byte shuffle followed by gzip) or ordinary `map.npy.gz` only when
  decompression has been compared byte-for-byte with the original and saves at
  least 5%. Original NPY hashes remain authoritative. There is no model execution
  or approximate regeneration.
- **Abandoned owned staging:** only known `.feedback-reuse-*`, `.storage-pack-*`
  and `.storage-json-*` staging older than 24 hours, unreferenced by scientific
  records, with no unknown payloads or symlinks. Recent/active staging is retained.

Known consumers (trusted scorer, cache verification and Streamlit) use
`benchmark.feedback_artifacts` to read original NPY bytes through these storage
representations. Existing summary, case-score and metric records are unchanged.
MLflow's local summaries, scores and previews remain in their original locations;
its artifacts are never deleted. Technical consumers should use `load_map` or
`map_bytes` rather than assuming that a logical NPY path remains uncompressed.

## What is protected

All checkpoints, checkpoint manifests, archived model sources/configuration,
champions/incumbents/Pareto references, unique scientific evidence, hypotheses,
per-image metrics, negative findings, logs, dataset/authorization files, SQLite
and MLflow records are preserved. Unknown caches and failed outputs are not
deleted just because they are old or large. The decoded training cache is
in-memory and does not need disk pruning.

GPU, feedback and cleanup locks serialize producers and housekeeping. Active
campaign jobs, queued/running jobs, and transitive frozen baseline inputs are
protected. The reader rejects path escapes, symlinks, cycles, excessive reference
chains and oversized decoded maps. Cleanup checks original-file identity again
before unlinking. The existing shared-host-UID limitation still applies; these
are trusted-process safeguards, not a new OS identity boundary.

## Audit and recovery

SQLite events `STORAGE_CLEANUP_PREPARED/DELETED/ABORTED` record path, original hash,
reason, replacement and byte counts. Staging has corresponding `STORAGE_TEMP_*`
events. Replacements and reference files are fsynced and atomically published
before unlinking originals. An interruption leaves either the original or a
verified readable replacement. The next pass reconciles prepared operations;
failed unlinks are aborted before retry so reclaimed bytes are not double counted.
Cleanup failure is recorded and does not turn into a candidate repair request.

The initial inventory, deletion preview, protected-file fingerprints and outcome
are under `$LOCO_RUNTIME/controller/storage_cleanup/`. These are maintenance
records, separate from scientific measurements. If safe reclamation is exhausted,
the original quota still rejects new work with observed/required/limit/free bytes.
No scientific evidence is sacrificed merely to reach the headroom target.

## Initial verified deployment (2026-10-24)

Runtime usage fell from **100.387 GiB to 79.649 GiB**: **20.738 GiB reclaimed**
and **20.351 GiB available** under the unchanged 100 GiB quota. The read-only
inventory identified 2,310 duplicate maps (11.015 GiB), including the previously
identified four copied evaluations. Lossless packing supplied the remaining
reclamation. All 7,339 replacements were restored and hashed against original
NPY bytes after cleanup.

Protected before/after fingerprints matched for 494 checkpoint files, 2,274
MLflow artifacts, retained model archives/references, live configuration and all
scientific tables (including 163 runs, 301 hypotheses and 23 campaigns). SQLite
maintenance events and technical cleanup state were the only ledger additions.
MLflow/Streamlit health checks returned 200; the Results page rendered without
exceptions after cleanup. No model inference or training was used to reclaim
space. Two controlled helper interruptions exercised resumability while the
compression implementation was tuned; the final verification includes replacements
from all passes, not only the final helper session.

On four historical full-resolution maps, ordinary gzip ratios were 0.905, 0.011,
0.676 and 0.911; reversible byte-shuffled gzip ratios were 0.683, 0.013, 0.400 and
0.711. Three cases compressed faster with shuffling. Every sampled byte stream
restored exactly. These are storage measurements, not scientific model results.
