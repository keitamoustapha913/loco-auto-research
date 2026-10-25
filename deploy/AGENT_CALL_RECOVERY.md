# Agent deadlines and interrupted scientific reviews

`limits.time_minutes_per_research_agents_calls` defaults to 30 minutes and accepts
positive values up to 30. Each Scout, Researcher, Implementer, Critic, and repair
invocation receives the earlier of this elapsed limit and the original campaign
deadline. Progress never resets it. This is independent of evaluation and the
15-minute **elapsed training-loop** target. Existing saved campaign configurations
remain unchanged; missing settings use the 30-minute safety default.

The runner drains stdin/stdout/stderr concurrently, retains bounded tails, observes
JSON progress events (`codex exec --json`) and process activity, and reaps its owned
process group on every exit. Silence alone is not a stall. A terminal `turn.failed`
event or an expired deadline can end an invocation; explicit authentication error
codes stop safely rather than requesting candidate edits. Existing ten-failure
recovery and trusted-error quarantine remain in force.

Diagnostics are owner-readable JSON in `controller/agent_invocations/<id>.json`.
They contain invocation/role identity, timestamps, prompt hash (never prompt), exit
code, termination reason, latest process sample/output activity, event types,
response-file existence, bounded response/stdout/stderr tails and exception trace.
Known credential formats are redacted before serialization. Do not treat redaction
or shared-UID Bubblewrap as a protection against malicious same-user code.
Each text tail is at most 64 KiB, with at most 32 event summaries and an 8 KiB
traceback. Only the latest 128 finished invocation diagnostic files remain; ledger invocation
and failure records survive transcript retirement. Existing periodic cleanup also
enforces this bound. An older diagnostic file may have been retired; absence is
not proof of success. Orphaned atomic-publication temporaries expire after an hour.

## Review recovery

`pending_reviews` finds complete comparisons without a recorded critique.
`reconstruct_review` checks the original plan/configuration identities, completed
run coverage and existing checkpoint/Git source provenance. Intentionally archived
checkpoints are explicitly unavailable for exact replay; numerical evidence remains.
`review_plan(..., evidence_campaign_id=...)` uses the original rows and result, with
the currently authorized campaign paying for the Critic call. It never trains or
evaluates. A `previous_best` campaign can resume old pending reviews; `new` remains
independent. A timed-out review is deferred for the current campaign immediately,
avoiding another blind 30-minute wait; other recoverable errors retain the existing
ten-retry allowance before deferral. Independent research can continue;
completed measurements are never rewritten as incomplete because of a Critic error.
Review state is `REVIEW_PENDING`, `REVIEW_INTERRUPTED`, `REVIEW_DEFERRED`, or `CRITIQUE`.

Historical read-only replay of campaign `f78ecd54-df1f-42fc-890f-25966bb78e7a`
reconstructed `dense_visible_rgb_change_targets_v1_plan`, hash
`6d8535291dbf5209add935a1a6ffcbab76a14ef918d1efd1334a9d509935068e`,
with four completed exploratory cells and four intact checkpoints. No critique
was generated. The original nine-hour wait's provider/CLI/network cause remains
unknown: the previous runner discarded partial stdout and temporary session files.
The verified orchestration cause was synchronous waiting until the campaign deadline.

## Future confirmation, not retroactive reinterpretation

New generated v4 plans include nullable `confirmation_requirements`. Exploratory
plans may use null. Confirmation specifies normal-image sample size, an integer
maximum additional false-positive count, relevant paired controls, and uncertainty
scope. For 47 normals, one additional alarm is 1/47 (2.13%); a 2% allowance cannot
permit it. Trusted admission checks available pinned-split counts; confirmation
applies the predeclared count allowance and reports marginal Wilson intervals.
Population rates and paired differences are not inferred from those intervals.
Historical plans keep their original acceptance criteria. Deployment readiness is
separate; exploratory ranking still uses 60/20/10/10 localization/AP/normal/efficiency.

Optional confirmation observations hash actual initial parameters/buffers, optimizer
configuration, and each declared prepared `TrainingStep.args` input (`input:0`, etc.)
before the trusted update. Hash sequences contain no images and are included in
artifact provenance. Cross-arm joins verify the entire common update prefix;
unequal final optimizer updates/exposures remain explicit. Choose inputs encoding
the source/edit/mask streams intended to match, excluding the scientific intervention.
Missing or mismatched receipts cannot support a causal claim. An undeclared stream
is unmeasured, not verified by equal seeds. These checks retain the existing shared
Python trust boundary; they are not OS-enforced proof against hostile candidates.

Critiques distinguish independent training-seed uncertainty, grouped/paired source
uncertainty, small normal-sample uncertainty, and repeated development selection.
Source bootstraps do not replicate training. Final-holdout generalization remains
untested. Live provider recovery and CUDA confirmation-observation overhead require
later authorized observation; offline processes and scaled CPU tests do not prove them.
