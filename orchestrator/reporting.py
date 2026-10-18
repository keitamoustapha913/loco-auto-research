"""Generate truthful, reproducible human-readable research status reports."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from orchestrator.state_store import StateStore
from orchestrator.decision_engine import pareto_frontier


def write_campaign_report(store: StateStore, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = store.list_runs(5000)
    frontier = set(pareto_frontier(runs))
    now = datetime.now(timezone.utc)
    lines = [f"# Research campaign report — {now.isoformat()}", "", "**Development results only. No official LOCO anomaly test metrics are included.**", "", f"- Total runs: {len(runs)}", f"- GPU hours charged: {store.total_gpu_seconds()/3600:.3f}", f"- Codex calls charged: {store.codex_call_count()}", f"- Tracking events pending: {len(store.pending_tracking(10000))}", "", "## Experiments", ""]
    for r in runs:
        metrics = json.loads(r["metrics_json"])
        lines += [f"### {r['run_id']}", f"- Candidate: {r['candidate_id']} · Category: {r['category']} · Stage: {r['stage']}", f"- Status: {r['status']} · MLflow run: {r['mlflow_run_id'] or 'pending'}", f"- Git commit: {r['git_sha'] or 'unrecorded'} · Dataset fingerprint: {r['manifest_sha256']}", f"- GPU seconds: {r['gpu_seconds']:.1f}", f"- Nondominated synthetic-proxy Pareto point: {'yes' if r['run_id'] in frontier else 'no'}", f"- Synthetic pixel AUROC proxy: {metrics.get('dev_synthetic/pixel_auroc', 'not measured')}", f"- E2E p95 latency ms: {metrics.get('inference/e2e_ms_p95', 'not measured')}", f"- Error: {r['error'] or 'none'}", ""]
    filename = output_dir / f"research_report_{now.strftime('%Y%m%d_%H%M%S')}.md"
    filename.write_text("\n".join(lines))
    return filename
