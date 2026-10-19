# Codex research agent contract

Read `RESEARCH_CHARTER.md` and `research_program.md` before any research action. These are human-owned and cannot be altered by implementer agents. Your task is to discover which approach works, **not** to implement a predetermined architecture.

Rules:
1. Never request or access final-holdout images, masks, listings or private manifests. Only the trusted evaluator may read the pinned, operator-authorized development partition; agents receive aggregate evaluation summaries.
2. Model learning uses train/good only. Real development anomalies are evaluation-only: no weight, feature-bank, normalization or threshold fitting. Aggregate feedback may guide method selection. State the provenance of external pretrained weights and code.
3. Only edit your candidate's worktree. Do not alter benchmark code, policies, tracking DB, dashboard, controller, or any shared evaluation protocol.
4. Do not use per-image training feature banks or reference-image nearest-neighbor retrieval at inference.
5. Every proposal must be falsifiable and provide costs, a baseline, plausible shortcut risks and a validation plan.
6. Treat synthetic proxy performance as **proxy**, never report it as real LOCO performance.
7. Record negative results. Do not discard errors or silently adjust evaluation to improve apparent outcomes.
8. Your role may propose changes to trusted infrastructure as a review item, but cannot apply them.
9. No arbitrary shell commands from research JSON are executed by the trusted scheduler.
10. Stop for budget, filesystem, permission and dependency approval gates.
