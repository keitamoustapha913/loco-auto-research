# Audited categorical versus deterministic completion

Implementation of frozen `categorical_regression_audited_redesign_v3`, plan hash
`50656331af3de0705eafab4cf90aa01e237db0fc8705f865ad702c288060c17b`.
No execution is authorized by this source or its offline tests.

## Exact source and learning semantics

Derived directly from candidate source SHA256
`9456cfd663ac7f930bdc8b4a00b625533139f1822b9b469f84288f8027bb7a03`,
commit `0005982da2ee124dd2a3b8e350f0840e0c88efe7`, run
`bfe8fc62-8b6d-4283-a844-8697e40d1125`. Existing workspace source reuse is
operator-authorized; no external implementation, weights or dependencies are added.
The original `Completion`, `Predictor`, `load`, device handling and hash helper are
unchanged. RGB vocabulary, coordinate channels, quantized-target MSE, categorical
cross-entropy, four mask phases and Adam(lr=0.002) remain unchanged. Both independently
initialized-then-matched trunks train jointly in **every** job; only the saved
objective selects which original trunk/readout is deployed.

Both arms use seeds 7, 19 and 31; each job performs 120 actual optimizer steps,
8 source presentations per step (960 total), size 128, grid 24, width 32, bins 4.
Source order follows the supplied manifest; torch's device generator uses seed+1.
Training audit records actual source and mask order digests, source ordering,
optimizer calls and presentations. Paired jobs must match these digests. The only
short schedule is the runner's disposable `validation_smoke` contract: 2 updates,
batch 1, 2 presentations, no retained smoke weights. It is never substituted for
an experimental job. Checkpoints retain both original state dictionaries.

Only validated train/good manifests are accepted. Candidate code does not calibrate
thresholds, inspect development anomalies, adapt at inference or retain training
image features. The trusted runner owns normal calibration and frozen real feedback.

## Declared within-source diagnostic

Exact trusted-runner generators are reused: white-patch seed 5000+i and same-image
target-only copy insertion seed 7000+i. No donor sources are used. Two-pixel square
binary erosion separates each editing footprint into interior and boundary.
Interior AP excludes boundary pixels but keeps outside-background negatives;
boundary AP excludes interior pixels but keeps outside-background negatives.
Each metric averages source-level AP; per-image support counts and identities are
retained for declared paired bootstraps. A footprint is a synthetic proxy, never a
verified missing-component or real logical localization target.

Photometric controls are clip(0.93*x+0.03,0,1). Alarm uses the map maximum strictly
greater than the runner-provided frozen normal-only threshold, with numerator,
denominator and per-source alarms. No threshold or normalization fitting occurs.
Diagnostics are inference-only and emit exactly the seven declared metrics.

## Offline verification and future execution

`test_model.py` requires CPU checks with no CUDA access. It uses artificial normal
fixtures, not new research results. The repository integration tests check source
identity, six exact queue cells and required output contracts. GPU performance and
new real-development results remain unmeasured until separately authorized.

Use the existing trusted entry point after authorization:
`orchestrator.experiment_plan.execute_plan(store, campaigns, campaign_id, runtime, plan)`.
It requires a RUNNING, authorized bounded campaign and normal dataset manifest;
no direct candidate training command bypasses it. The generic `research start`
path may invoke implementation/planning agents and is not a frozen-plan-only
launcher. For this prepared comparison, use execute_plan with the exact plan,
then the existing review_plan for a separately budgeted critic call. No controller,
scheduler or security redesign is required. Model selection remains exploratory;
no confirmed champion promotion is permitted.
