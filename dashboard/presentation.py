"""Small, testable view model. Ledger facts stay separate from scientific claims."""
from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Iterable


COMPLETED = {"COMPLETED", "REVIEWED", "PROMOTED"}
INFRASTRUCTURE = "Infrastructure test"
PILOT = "Research pilot"
FULL = "Full research evaluation"
OFFICIAL = "Official benchmark evaluation"
UNCLASSIFIED = "Unclassified run"


def direction_label(value: str) -> str:
    if value.startswith('evidence-followup:'):
        parts = value.split(':', 2)
        if len(parts) == 3:
            return f"{parts[2].replace('_', ' ').title()} {parts[1].replace('_', ' ')} using the latest experiment evidence"
    return value.replace('_', ' ') if value else 'No next direction recorded'
METRICS = {
    "dev_synthetic/pixel_auroc": ("Synthetic localization AUROC", "Area under the pixel ROC curve on synthetic development changes; not official LOCO performance.", "score"),
    "dev_synthetic/pixel_ap": ("Synthetic localization AP", "Micro pixel average precision on the primary synthetic development edit; affected by mask prevalence and not official LOCO performance.", "score"),
    "dev_synthetic/alternate_pixel_auroc": ("Alternate-generator AUROC", "Pixel AUROC on an independently defined synthetic development generator; not official LOCO performance.", "score"),
    "dev_synthetic/alternate_pixel_ap": ("Alternate-generator AP", "Micro pixel average precision on the alternate synthetic generator; not official LOCO performance.", "score"),
    "dev_synthetic/relocated_pixel_ap": ("Copy-insertion AP", "Pooled pixel AP on the edited target footprint of feathered patch copying. The source remains present; this is not a missing-object or official LOCO localization score.", "score"),
    "robustness/artifact_matched_good_fpr": ("Matched-control false alarm rate", "Share of same-location good-image donor edits above the separately calibrated normal threshold. These controls are plausibly layout-preserving, not certified valid.", "percent"),
    "robustness/photometric_good_fpr": ("Photometric-control false alarm rate", "Share of mild appearance edits above the separately calibrated normal threshold; object layout is preserved.", "percent"),
    "dev_synthetic/region_pro_proxy_NOT_OFFICIAL": ("Synthetic region overlap proxy", "Region overlap on synthetic development masks; not official LOCO sPRO.", "score"),
    "normal/image_fpr_at_threshold": ("Normal image false positive rate", "Share of permitted good validation images above the selected threshold.", "percent"),
    "inference/e2e_ms_p95": ("Inference latency (p95)", "95th percentile end-to-end batch-one inference time in milliseconds.", "ms"),
    "inference/checkpoint_mib": ("Model size", "Serialized checkpoint size in mebibytes.", "MiB"),
    "inference/peak_vram_mib": ("Peak GPU memory", "Peak allocated CUDA memory during measured inference, in mebibytes.", "MiB"),
    "inference/parameters": ("Parameters", "Number of trainable and fixed model parameters in the predictor.", "count"),
}


def classify_run(run: dict) -> str:
    """Use explicit provenance. Completion alone never confers research status."""
    config = run.get("config") or {}
    if run.get("candidate_id") == "toy_normal_ae" or run.get("hypothesis_id") in {"infrastructure-smoke", "fixture-demo"} or run.get("category") == "fixture" or config.get("experiment_type") == "infrastructure_test":
        return INFRASTRUCTURE
    stage = str(run.get("stage") or "").lower()
    if not run.get("candidate_id") or not run.get("hypothesis_id"):
        return UNCLASSIFIED
    if stage == "official" and config.get("evaluation_protocol") == "sealed_official" and config.get("independent_evaluator_attested") is True:
        return OFFICIAL
    if stage == "full":
        return FULL
    if stage in {"smoke", "pilot"}:
        return PILOT
    return UNCLASSIFIED


def research_runs(runs: Iterable[dict]) -> list[dict]:
    return [run for run in runs if classify_run(run) in {PILOT, FULL, OFFICIAL}]


def research_summary(runs: Iterable[dict]) -> dict:
    all_runs = list(runs)
    scientific = research_runs(all_runs)
    return {
        "evaluated": sum(run.get("status") in COMPLETED for run in scientific),
        "infrastructure_completed": sum(classify_run(run) == INFRASTRUCTURE and run.get("status") in COMPLETED for run in all_runs),
        "research_failed": sum(run.get("status") in {"FAILED", "TIMED_OUT", "OOM", "INTERRUPTED"} for run in scientific),
    }


def comparable_research_runs(runs: Iterable[dict]) -> list[dict]:
    """Require two distinct completed models measured on the same permitted set."""
    candidates = [r for r in research_runs(runs) if r.get("status") in COMPLETED and
                  all(metric_value(r, key) is not None for key in (
                      "dev_synthetic/pixel_auroc", "inference/e2e_ms_p95",
                      "inference/checkpoint_mib", "inference/peak_vram_mib"))]
    groups = {}
    for run in candidates:
        key = (run.get("category"), run.get("manifest_sha256"))
        groups.setdefault(key, []).append(run)
    eligible = [group for group in groups.values() if len({r.get("candidate_id") for r in group}) >= 2]
    return max(eligible, key=len) if eligible else []


def metric_value(run: dict, key: str):
    value = (run.get("metrics") or {}).get(key)
    return value if isinstance(value, (float, int)) and math.isfinite(value) else None


def format_metric(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "count":
        return f"{int(value):,}"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "MiB":
        return f"{value:.1f} MiB"
    return str(value)


def format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d} s"
    hours, minute = divmod(minutes, 60)
    return f"{hours} h {minute:02d} min"


def format_date(timestamp) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%d %b %Y, %H:%M") if timestamp else "—"


def run_label(run: dict, include_date: bool = False) -> str:
    name = str(run.get("candidate_id") or "Unnamed candidate").replace("_", " ")
    if include_date:
        return f"{name} · {format_date(run.get('created_at'))} · {str(run.get('run_id') or '')[:8]}"
    return name


def experiment_rows(runs: Iterable[dict]) -> list[dict]:
    result = []
    for run in runs:
        result.append({
            "Model": run_label(run),
            "Hypothesis": str(run.get("hypothesis_id") or "—").replace("_", " "),
            "Type": classify_run(run),
            "Status": str(run.get("status") or "—").replace("_", " ").title(),
            "Synthetic localization AUROC": format_metric(metric_value(run, "dev_synthetic/pixel_auroc"), "score"),
            "Normal image false positive rate": format_metric(metric_value(run, "normal/image_fpr_at_threshold"), "percent"),
            "Latency (p95)": format_metric(metric_value(run, "inference/e2e_ms_p95"), "ms"),
            "Model size": format_metric(metric_value(run, "inference/checkpoint_mib"), "MiB"),
            "GPU slot time": format_duration(run.get("gpu_seconds")),
            "Date": format_date(run.get("created_at")),
        })
    return result


def parse_event(event: dict) -> dict:
    try:
        payload = json.loads(event.get("payload_json") or "{}")
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def autonomy_state(flags: dict, paused: bool) -> tuple[str, str]:
    if flags.get("LOCO_SANDBOX_VERIFIED") != "1" or flags.get("LOCO_CODEX_ISOLATION_VERIFIED") != "1":
        return "Blocked", "Research isolation has not been verified for Codex and candidate execution."
    if flags.get("LOCO_ALLOW_UNATTENDED") != "1":
        return "Awaiting approval", "The operator has not enabled unattended research."
    if paused:
        return "Paused", "The approved research queue is paused."
    return "Authorized", "Approved research may run within the configured limits."


def development_card_metrics(feedback):
    """Only measured real-development values; never borrow synthetic/GPU figures."""
    import math
    m = feedback.get('metrics', {})
    def number(key, suffix='', digits=2):
        value = m.get('dev_real/' + key)
        if not isinstance(value, (int,float)) or not math.isfinite(value): return 'Unavailable'
        return f'{value:,.{digits}f}{suffix}'
    fp, total = m.get('dev_real/normal_false_positives'), m.get('dev_real/normal_images')
    device = feedback.get('inference_device', 'cpu').upper()
    return [
        ('Logical AUROC', number('logical_image_auroc')),
        ('Logical AU-sPRO', number('logical_au_spro_0_05_compatible')),
        ('Logical pixel AP', number('logical_pixel_ap_pooled')),
        ('Structural AUROC', number('structural_image_auroc')),
        ('Structural localization', number('structural_au_spro_0_05_compatible')),
        ('Normal false positives', f'{int(fp)} / {int(total)}' if fp is not None and total is not None else 'Unavailable'),
        (f'Inference p95 · {device}', number('inference_e2e_ms_p95',' ms',1)),
        ('Parameters', number('parameters','',0)),
        ('Peak inference VRAM', number('peak_cuda_inference_mib',' MiB',1) if device.startswith('CUDA') else 'Unavailable · CPU'),
    ]


def real_comparison_rows(feedback, run_ids=None):
    """Measured real-development endpoints only, in campaign execution order."""
    import math
    by_id = {f['run_id']: f for f in feedback}
    selected = [by_id[rid] for rid in run_ids if rid in by_id] if run_ids is not None else list(feedback)
    result = []
    for f in selected:
        def metric(key):
            value = f.get('metrics', {}).get('dev_real/' + key)
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
        result.append({'Candidate': f['candidate_id'].removeprefix('next_').replace('_', ' ').capitalize(),
                       'Logical AUROC': metric('logical_image_auroc'),
                       'Logical AU-sPRO': metric('logical_au_spro_0_05_compatible'),
                       'Pixel AP': metric('logical_pixel_ap_pooled'),
                       'split': f.get('split_id'), 'evaluator': f.get('evaluator_version')})
    return result


def localization_change(rows):
    """Descriptive change only for a shared, explicit evaluation protocol."""
    if len(rows) < 2 or not rows[0]['split'] or not rows[0]['evaluator']:
        return None
    if len({(r['split'], r['evaluator']) for r in rows}) != 1:
        return None
    first, last = rows[0]['Logical AU-sPRO'], rows[-1]['Logical AU-sPRO']
    if first is None or last is None or first <= 0:
        return None
    return (last - first) / first * 100
