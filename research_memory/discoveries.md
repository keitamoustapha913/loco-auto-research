# discoveries

## Stage 7 frozen-checkpoint diagnostics (2026-10-18)

The 15 Stage 6B runs are preserved. Neither-component and spatial are one effective baseline: trained tensors, residual scales and saved per-image predictions are identical for each seed. On a separate versioned *development diagnostic* using 12 reused good validation images and frozen checkpoints, global-minus-local insertion-target footprint AP was −0.0015 on average across seeds, with every per-seed paired interval spanning zero. A 1.08 gain raised image scores by about 4.5 and caused some normal-image alarms; fine noise raised scores below threshold. Correctly aligned whole-frame shift maps had high correlation, which does not establish semantic reasoning. All masks are edit footprints, not official or semantic anomaly labels. See [Stage 7 report](../results/reports/stage7_generalization_study.md) for protocols, uncertainty, limitations and full results.

No existing candidate is promoted for logical localization. Fresh permitted good-image clusters and independently certified valid/invalid edits are needed before a confirmatory claim. No official anomalies or masks were accessed.
