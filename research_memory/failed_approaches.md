# failed_approaches

## 2026-10-18 — toy reference generator sensitivity (development only)

Run `02243e69-9a90-45e8-ab0e-d945485fc771` used the authorized `juice_bottle` good-only manifest (335 training, 54 validation images) and one bounded CUDA epoch. The toy normal autoencoder achieved pixel AUROC **0.9887** on the occlusion generator and **0.6712** on a disjoint patch-copy generator; the gap was **0.3175**. Normal-image FPR was **0.0278** on the held-out good assessment subset. These are synthetic development proxies, not official LOCO measurements. The large generator gap is a shortcut/generalization concern, not evidence that the model succeeds or fails on real logical anomalies. Do not promote this infrastructure reference as a scientific method. MLflow run: `19b8f859a2784f999f982d7d685d46c8`.
