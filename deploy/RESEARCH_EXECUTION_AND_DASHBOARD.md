# Execution corrections and dashboard verification

The four pages have separate responsibilities: Home reports the selected
campaign's current state; Research explores comparisons, arms, seeds and critiques;
Results compares numerical model evidence and opens retained visual artifacts;
System reports services, authorization scope and storage. Campaign selection is
shared. Comparisons paginate eight at a time and model tables 25 at a time.

## Training boundaries

Training still completes using elapsed training-loop time, including batch
preparation. The clock now measures the final iteration from the preceding
completed-update boundary, including inter-update publication/bookkeeping. New
audits identify this boundary definition. Legacy audits lacking that definition
have a bounded 100-microsecond bookkeeping allowance; new audits retain the
one-microsecond numerical allowance. Neither allows finishing below target,
missing trusted audits, mismatched counters or excessive final-update overshoot.
Historical failed jobs remain failed; no historical result is reclassified.

Live timing publication is coalesced to once per second. Terminal audits always
publish exact measured elapsed duration, learning diagnostics and update/exposure
counts. Learning, sampling, integrity hashes and the 15-minute target are unchanged.
A local metadata-only benchmark of 2,000 progress publications measured 0.0590 s
uncoalesced versus 0.00036 s coalesced. This is not a model/CUDA throughput claim.

The recorded campaign's 35 accepted audits contain 21,547.85 s of candidate
preparation, 1,355.42 s of integrity checks and 1,013.49 s of audit I/O. CUDA
synchronization submeasurements overlap learning and must not be added twice.
On generated CPU fixture pixels, 100 batches through the actual negative-tail
candidate's preparation functions took 0.1785 s for synthetic preparation and
0.2772 s for its record tensor hashing. Frozen candidate sources are unchanged;
these CPU timings cannot certify CUDA throughput or attribute every preparation
second. Existing immutable decoded-image caching is preserved. A separately
authorized real-candidate CUDA benchmark is needed to measure training gains.

## Complete-domain geometry preflight

`benchmark/synthetic_geometry.py` proves donor feasibility over declared integer
square sizes and rectangular destination domains. `SYNTHETIC_EDIT_GEOMETRY` is a
literal candidate declaration, not a new operator setting or a plan rewrite.
The historical guarded `disjoint_origin`/two-dimensional `mgrid` primitive is also
recognized without executing candidate code. For 128-square images with all
integer destinations, side 43 remains feasible, while side 44 has an impossible
central destination; side 48 at (40,40) is also impossible. Touching patches are
nonoverlapping. The CPU probe and actual plan dispatch check before GPU work.
Failures are candidate contract errors, not controller repair requests.

This is a proof of the supported declared geometry, not a proof of arbitrary
Python sampling code. Other primitives require representative/exhaustive CPU
validation and explicit support; frozen protocols are never silently repaired.

## Actual paired controls

The historical 12-hour plans set `confirmation_requirements` to null even when
completion prose requested joins. The runner therefore collected no trusted
paired-control receipts. Those historical comparisons remain exploratory and
NOT_MEASURED. Future exploratory paired studies can request actual observations
without asserting confirmation. Confirmation requires declared measured controls.

Verification distinguishes VERIFIED, MISMATCHED, NOT_MEASURED and NOT_APPLICABLE.
Actual initial parameter/buffer bytes, initial optimizer configuration and each
selected prepared-input stream are checked against provenance hashes. The common
update prefix is compared; additional updates/exposures remain explicit confounds.
Only declared inputs are verified. Equal seeds, candidate-produced prose or
source-image intervals cannot substitute for measured streams or seed replication.

## Data and visual acceptance

Read-only queries against campaign dd063885-8e70-4afa-8008-e59926ba00c3 resolve
37 executed jobs, 35 reviewed/evaluated trained models, 10 execution-complete
comparisons, three incomplete comparisons and two failed jobs. Twelve of its
13 comparisons have recorded critiques. Cross-campaign resumed reviews join by
`evidence_campaign_id` and plan ID. The oldest-100 evaluation cutoff is removed;
35 real evaluations are now visible under the selected campaign.

Home and Results resolve the existing visible-change seed-863 research incumbent
with logical compatible AU-sPRO 0.805236 and pixel AP 0.673069. No registry writes
or promotion occurs. Cohorts with incompatible measurement conditions are not
ranked as interchangeable. Archived artifacts retain all numerical results and
are labeled intentionally archived, distinct from missing, temporarily
unavailable, not generated or failed-integrity artifacts.

Each live page was inspected at 1440x1000 and 900x1000 with no Streamlit exceptions
or document-level horizontal overflow. The selected-campaign Results filter was
also exercised. Screenshots are technical verification evidence outside scientific
history. Existing scientific heatmap alignment/integrity tests remain authoritative;
no new development evaluation, GPU run or research-agent call was performed.

The System view is read-only: it never invokes cleanup, grants authorization,
changes quotas or controls jobs. It distinguishes inactive historical campaigns
from present execution eligibility (checked by the existing CLI). It reports the
original 100 GiB quota, nine full model-producing records and metrics-only archives.

A future authorized live campaign remains necessary to observe new candidate
control declarations and real GPU training efficiency. Original-scorer equivalence
and independent final-holdout generalization remain unverified.

Final validation: 712 tests passed with `LOCO_TEST_BWRAP=1`, including real CPU
Bubblewrap integration. SQLite-to-MLflow comparison found zero mismatches among
910 development metric values for the regression campaign. The historical
run-metrics fingerprint and all retained incumbent archive hashes were unchanged;
tracking had zero pending deliveries. No live research configuration was edited.
