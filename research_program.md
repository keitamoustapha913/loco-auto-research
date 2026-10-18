# Research program (operator editable)

## Main question
Which compact, train-on-good-only, memory-bank-free method best produces meaningful logical anomaly heatmaps on MVTec LOCO AD, at low end-to-end latency?

## Candidate freedom
Investigation is open to any justified family, including predictive, student–teacher, component-aware, generative, statistical, discriminative, self-supervised and hybrid approaches. Existing references (EfficientAD, CSAD, MuDeNet, GCAD) are starting literature, not imposed architecture.

## Default acceptance constraints (targets, not results)
- <= 10 million inference parameters as an initial screening target
- <= 20 ms batch-1 end-to-end latency on RTX 4090 as an initial target
- no dataset-derived image-level feature bank / kNN lookup
- full-map aligned anomaly score, finite and reproducible

## Budget and progression
Start with CPU sanity -> 5-minute GPU smoke -> 15–30-minute pilot -> pre-approved full trials -> independent final evaluation. The system enforces remaining GPU hours, disk usage and Codex-call ceilings. All limits configurable through operator-approved environment variables; no auto-expansion.

## Evidence and objectives
Normal-good false-positive controls, synthetic held-out localization proxy, edit-artifact controls, full latency, peak VRAM, model size, reproducibility, per-category performance. Failed methods remain in the scientific record. Normal-only proxies cannot establish superiority on real LOCO logical anomaly localization.

## Immediate milestone
Three scientifically distinct *independently proposed* candidate hypotheses; smoke-test and, if allowed by budget, pilot them sequentially; record MLflow and dashboard evidence; produce a justified next hypothesis. No claims of already completing this milestone.
