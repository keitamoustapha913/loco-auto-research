# Decision-driven research follow-ups

The planner chooses scientific methods. Trusted code supplies measurements,
execution integrity and a compact history; it does not score scientific value or
set training/diagnostic quotas. Existing schemas, promotion rules and budgets are
unchanged.

## Why the previous branch kept narrowing

Campaign `3b031678-d856-4e63-991b-cadbb3f7491b` recorded five reviewed frozen
investigations, 15 runs, zero new training and 3,615.505 seconds of job execution.
The initial rationale emphasized information per **training** minute despite
substantial inference/diagnostic costs. Follow-up generation explicitly asked for
hypotheses based specifically on the last run and Critic, while repeated Critic
recommendations were `investigate`. Whole-history memory existed but did not
synthesize these repeated checkpoint uses, costs and tested explanations. This
encouraged continuation of the last branch; it was not a hardcoded Dice model.

## Corrections

- `model_registry.research_decisions` derives a read-only, whole-history index of
  questions, claims, recorded outcomes, findings, remaining concerns, advisory
  follow-ups and measured job costs. It groups frozen studies by exact checkpoint
  hashes and dataset rather than names. Unfinished work remains unfinished.
- Cumulative evidence includes that index. The prompt highlights reviewed
  questions and repeated-checkpoint costs from prior campaigns, including when a
  new campaign has no runs yet. Full findings remain in immutable mounted memory.
  A measured live read of 49 studies/six checkpoint groups took about 0.018 seconds;
  no model execution is required.
- Repetition with reviewed findings explicitly prompts reconsideration of marginal
  value. It does not reject a plan, impose a count limit, or assert that a different
  diagnostic on the same model is scientifically equivalent.
- Planner and Critic compare existing-evidence investigation, model development,
  confirmation and redirection. The last Critic suggestion is advisory. Existing
  `expected_observation`, `evaluation_plan` and `priority_rationale` fields should
  describe support/rejection, outcome-to-action branches and total elapsed cost.
  If every result leads to the same action, reconsider the test or explain its
  independent scientific benefit. After a useful diagnosis, consider a feasible
  model/training/scoring improvement rather than automatically a narrower proxy.
- Known saved diagnostic records now have a bounded availability inventory in
  `investigation_observations`: counts, field names and candidate-declared analysis
  status. No arbitrary strings, record values, images, masks or private real-case
  identities are newly exported. Available records do not establish completed
  cross-arm analysis. Unsupported analysis should be identified as a gap rather
  than repeatedly proposing inference that cannot supply it.
- Future `NEXT_DECISION` events retain the finding and additionally record the
  proposed investigation as advisory. Historical events are untouched.

Real logical compatible AU-sPRO and pixel AP retain their 60%/20% selection
weights. Normal false positives and efficiency retain 10% each. A proxy gain is
not real-localization improvement. Matched seeds/controls and independent training
replication remain necessary for corresponding causal/reproducibility claims.
The current incumbent is a research reference, never an architecture restriction.

## Offline evidence and limitations

`tests/fixtures/research_decisions_campaign.json` is a read-only projection of the
actual saved sequence, retaining plan/checkpoint identities, recorded timings,
questions and Critic findings. It contains no images, weights or fabricated agent
decisions and is not an executable scientific plan. Replay verifies that repeated
studies cannot hide behind new hypothesis names and that the rejected rounding
explanation, negative coverage evidence and unresolved work remain available.

Deterministic tests exercise prompt transport, legacy-memory compatibility,
fresh-campaign access to old findings, admitted frozen studies, model-development
plans and continued controller operation after a comparison. Agent transports in
these tests are mocked; this is not proof that live Codex will choose a better
experiment. No live calls or experiments were performed in this maintenance task.

Expected scientific value is a judgment for Codex, not a reliable English-keyword
validator. These changes remove last-run anchoring and make repetition/costs
visible, but cannot guarantee improvement or forbid every low-value proposal.
Observe the next authorized campaign to verify whether decisions become more
actionable. Missing joint analyses and authentic learning traces are not invented
or retroactively completed here. No new data-analysis execution interface was
introduced.

Validation: `LOCO_TEST_BWRAP=1 .venv/bin/pytest` completed with **646 passed,
no skips**. Planning/critique guidance is scoped to those roles; the Implementer
continues to receive an explicit instruction to execute the exact approved arms,
seeds, controls and schedule. No campaign or agent was launched for validation.
