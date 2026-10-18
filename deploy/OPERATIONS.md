# Operator guide — RTX 4090 single-host lab

## 1. Provision
Install Python 3.11 or newer, NVIDIA driver, CUDA-enabled PyTorch, Codex CLI, Git and SSH. From repo root: `pip install -e '.[all]'`, `pytest -q`, and `python -c 'import torch; print(torch.cuda.is_available())'`. Verify true on server. The toy demo works on CPU and requires no LOCO data.

Choose `LOCO_RUNTIME=/srv/loco-runtime` with writable `jobs/`, `controller/`, `mlflow/` and separate authorized `good_data/`. Choose `/srv/loco-sealed` mounted only into final evaluator identity. Follow `SECURITY.md` before enabling non-toy candidate execution.

## 2. Start MLflow + dashboard

```bash
export LOCO_RUNTIME=/srv/loco-runtime
export LOCO_LEDGER=$LOCO_RUNTIME/controller/experiments.sqlite
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
bash scripts/start_mlflow.sh
# separate process:
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Start/restart only with controlled user permissions and a trusted package environment. Use ssh port-forwarding from laptop. MLflow tracking backend and artifacts require routine snapshots/backups.

## 3. Prepare authorized good-only data

Directory shape: `good_data/<category>/train/good/*.png` and `good_data/<category>/validation/good/*.png`. This importer deliberately does not accept any path containing `test`. The supplied MVTec release may use `train/good`, `validation/good` or require an operator-built good-only validation split; preserve provenance and never silently use official test good images to create a development validation set.

```bash
python -m scripts.make_manifest --root "$LOCO_RUNTIME/good_data" --output "$LOCO_RUNTIME/good_manifest.json"
python -m orchestrator.controller init
python -m orchestrator.controller queue-toy --manifest "$LOCO_RUNTIME/good_manifest.json" --device cpu
python -m orchestrator.controller worker-once --manifest "$LOCO_RUNTIME/good_manifest.json"
python -m scripts.sync_mlflow
```

The toy model is not a benchmark baseline. The synthetic corruption example is intentionally rudimentary and **not an adequate proxy for real logical errors**; the agent must improve the *research evaluation methodology* through operator-reviewed changes, not edit the trusted metric implementation ad hoc.

## 4. Codex research

```bash
python -m orchestrator.controller propose
python -m orchestrator.controller status
python -m orchestrator.controller implement --hypothesis <validated_hypothesis_id>
# Operator must review and test the worktree/permissions before queueing real training.
```

The proposal agent returns strict JSON. The implementer writes only inside its own detached worktree. Unknown dependencies require explicit approval. The demo's toy run is the only pre-wired submitter; for another candidate, use the controller's validated registered-candidate interface (see `queue-candidate` command and `configs/experiment.example.json`). `controller cycle` is a bounded research step that records proposals or reviews pending ones, not an unlimited daemon.

## 5. Pause/resume and recovery

```bash
python -m orchestrator.controller pause
python -m orchestrator.controller resume
python -m orchestrator.controller reconcile
python -m scripts.sync_mlflow
```

Pause prevents new starts, it does not kill active jobs. After a crash, `reconcile` marks stale RUNNING as INTERRUPTED; inspect actual processes before requeue. Do not automatically re-run an experiment with a potentially already running GPU child.

## 6. Backups
Stop writers or use SQLite backup API. Back up the operational SQLite DB, MLflow SQLite DB, and MLflow artifacts together. Confirm restoration on a separate directory. Do not back up sealed official data into the agent-accessible runtime. Never commit DB files or image data to Git.

## 7. Real benchmark
A separately permissioned human-operated evaluator should import the pinned official MVTec scoring implementation and frozen finalist artifacts with checksums. It must never report test anomaly maps, masks or scores into Codex-visible MLflow/ledger. This repository does **not** masquerade its synthetic PRO proxy as official sPRO.
