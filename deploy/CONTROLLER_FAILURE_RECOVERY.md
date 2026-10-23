# Controller failures and repair ownership

Campaign `9b7a82b8-7c69-4a89-87ee-c83d5baa7c72` recorded 75 unbound-local
validation failures. An import inside the new-training branch of the Implementer
validation closure made `time_based` local to that closure. The all-frozen branch
skipped the import and then used it for campaign-aware CPU validation. The import
now precedes both branches. The regression exercises `implement_proposal` with a
real active CampaignState, a frozen diagnostic provider, CPU validation, and no
training functions. It failed at the original binding before the correction.

## Ownership, not exception wording

Known generated-work checks raise `CandidateValidationError`, adapter contract
errors, `PlanningValidationError` or `InvalidAgentOutput`. These can use the
existing bounded correction and ten-failure recovery policy. Isolated candidate
imports/test workers report an explicit result marker; failure before that result
is infrastructure failure, not an assertion about candidate correctness. Unknown
Python exceptions, including a raw ValueError in trusted helpers, do not become
candidate repair requests. Successful-stage ledger updates are a separate trusted
phase, even when the preceding candidate validation passed.

`TrustedControllerError` records the full exception chain/traceback, operation,
source-code version, trusted locations and a stable fingerprint excluding candidate
IDs and dynamic messages. The durable `TRUSTED_INFRASTRUCTURE_ERROR` event is not
fed to the candidate Implementer as a repair request. Unknown planning/scoring
infrastructure exceptions are not silently converted into invalid agent output or
scientific negative results.

A pre-dispatch fault isolated to frozen-diagnostic validation quarantines that
operation for the campaign/code version. The same branch is blocked before another
Implementer invocation, including different hypotheses and controller restarts.
The Researcher sees the unavailable operation and must choose a safe executable
investigation or report the limitation. Pure frozen readouts without a diagnostic
provider do not require implementation. There is no architecture prescription.

Safe isolated recovery immediately uses the existing verified-state restoration;
it does not first burn ten candidate retries. Pending comparisons remain
INCOMPLETE, their plans and checkpoints are preserved, and abandoned attempts are
not replayed. Verified completed stages stay reusable. Fatal/unknown execution or
state-commit defects, unavailable verified recovery and integrity/safety failures
stop the campaign safely. The original campaign start/deadline is never reset.
A reviewed trusted-code correction changes the operation-version identity; this
never automatically resumes a stopped campaign or changes its authorization.

## Missing implementation files

The five missing-model failures had empty/no-work implementation responses. Saved
caveats explicitly cited the role instruction “Run only pre-approved bounded
commands via trusted runner.” That named no available agent tool. The actual
Codex Bubblewrap command supplies a writable `/workspace/candidates/<id>` mount,
mounted charter/program/evidence and the existing Codex shell/editing interface.
The controller's trusted candidate runner runs after the response, not as an agent
command tool. The role and invocation now explain that distinction and give exact
sandbox paths, without adding mounts, privileges, tools or security flags.

Validation requires actual model.py plus an implemented_files entry identifying
it, then source/interface and CPU-test checks. A blocked/empty response cannot
silently accept leftover files. Missing files return the exact required sandbox
path and remain candidate-owned errors. CPU/static work is permitted only within
the existing assigned mounts; GPU jobs, installs, protected edits and credential
access remain prohibited. Real historical CLI tool transcripts were not retained,
so exact per-call tool availability or internal agent reasoning cannot be proven
from caveats alone. Offline tests demonstrate that the production mounts support
policy reads and candidate file creation without invoking Codex.

## Verification boundaries

Offline tests cover the actual active frozen-only controller branch, new-training
validation, readout-only reuse, explicit candidate repairs, raw trusted failures,
durable operation blocking, safe independent continuation, fatal recovery,
immutable retained archives, ten recoverable failures and original deadlines.
CPU Bubblewrap tests use artificial fixtures/dummy credentials. No live agents,
GPU work, real-development evaluation or campaigns are run by this maintenance.
No model, metric definition, selection weight, live configuration or scientific
record is changed. Future adapter validators must explicitly identify genuine
candidate errors; unknown infrastructure errors intentionally fail closed.

Final offline verification: `LOCO_TEST_BWRAP=1 .venv/bin/python -m pytest -o addopts= -q`
passed **602 tests**, with no skips. This includes the actual active-campaign
all-frozen validation closure using real CPU Bubblewrap import and pytest, while
agent calls are mocked. The seed-607 incumbent archive for run
`0891b9a1-5f10-5611-80a2-cb9d1398e54b` was checked read-only: every recorded archive
hash verified and its exploratory selection score remained 0.7296111124829123.
Live Codex compliance with the clarified instructions remains unverified; no new
agent call or scientific evaluation was performed.
