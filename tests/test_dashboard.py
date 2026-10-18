from pathlib import Path
from PIL import Image
from streamlit.testing.v1 import AppTest
from orchestrator.state_store import StateStore

from dashboard.data_access import allowed_preview, heatmap_sample, list_runs, get_mlflow_url
from dashboard.charts import run_table
from dashboard.presentation import (
    INFRASTRUCTURE, FULL, OFFICIAL, PILOT, METRICS, autonomy_state,
    classify_run, comparable_research_runs, experiment_rows, format_duration,
    format_metric, research_summary,
)

def test_read_only_empty_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path))
    monkeypatch.setenv('LOCO_LEDGER',str(tmp_path/'missing.db'))
    assert list_runs() == []
    assert run_table([]) == []

def test_no_sealed_preview(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCO_RUNTIME',str(tmp_path/'runtime'))
    forbidden=tmp_path/'sealed'/'official_test'
    forbidden.mkdir(parents=True)
    (forbidden/'synthetic_heatmap.png').write_bytes(b'bad')
    assert allowed_preview({'run_id':'x','artifact_dir':str(forbidden)},'synthetic_heatmap.png') is None
    assert allowed_preview({'run_id':'x','artifact_dir':str(forbidden)},'test.png') is None

def test_mlflow_link_missing():
    assert get_mlflow_url({'mlflow_run_id':''}) is None


def make_run(candidate="toy_normal_ae", hypothesis="infrastructure-smoke", stage="smoke", status="COMPLETED", **extra):
    return {"run_id": extra.pop("run_id", candidate), "candidate_id": candidate,
            "hypothesis_id": hypothesis, "category": "juice_bottle", "stage": stage,
            "status": status, "config": {}, "metrics": {
                "dev_synthetic/pixel_auroc": 0.82,
                "inference/e2e_ms_p95": 8.4,
                "inference/checkpoint_mib": 2.5,
                "inference/peak_vram_mib": 520,
            }, "manifest_sha256": "good-only-fingerprint", "gpu_seconds": 35, **extra}


def test_infrastructure_is_excluded_from_scientific_counts_and_frontier():
    toys = [make_run(run_id=str(i)) for i in range(3)]
    assert all(classify_run(r) == INFRASTRUCTURE for r in toys)
    assert research_summary(toys) == {"evaluated": 0, "infrastructure_completed": 3, "research_failed": 0}
    assert comparable_research_runs(toys) == []
    pilot = make_run("candidate_a", "hypothesis_a", "pilot")
    assert classify_run(pilot) == PILOT
    assert research_summary(toys + [pilot])["evaluated"] == 1


def test_stage_and_provenance_labels_do_not_turn_proxies_into_official_scores():
    full = make_run("candidate_a", "hypothesis_a", "full")
    official = make_run("candidate_b", "hypothesis_b", "official", config={"evaluation_protocol": "sealed_official", "independent_evaluator_attested": True})
    unverified = make_run("candidate_c", "hypothesis_c", "official")
    assert classify_run(full) == FULL
    assert classify_run(official) == OFFICIAL
    assert classify_run(unverified) == "Unclassified run"
    assert "Synthetic" in METRICS["dev_synthetic/pixel_auroc"][0]
    assert "not official" in METRICS["dev_synthetic/pixel_auroc"][1]
    assert format_metric(0.125, "percent") == "12.5%"
    assert format_metric(8.42, "ms") == "8.4 ms"
    assert format_duration(125) == "2 min 05 s"
    row = experiment_rows([full])[0]
    assert row["Synthetic localization AUROC"] == "0.820"
    assert row["Latency (p95)"] == "8.4 ms"
    assert "dev_synthetic/" not in str(row)


def test_comparison_requires_two_models_on_same_authorized_manifest():
    a = make_run("candidate_a", "hypothesis_a", "pilot")
    b = make_run("candidate_b", "hypothesis_b", "pilot")
    assert len(comparable_research_runs([a, b])) == 2
    b["manifest_sha256"] = "different"
    assert comparable_research_runs([a, b]) == []


def test_autonomy_states_and_empty_research():
    assert research_summary([])["evaluated"] == 0
    flags = {}
    assert autonomy_state(flags, False)[0] == "Blocked"
    flags.update(LOCO_SANDBOX_VERIFIED="1", LOCO_CODEX_ISOLATION_VERIFIED="1")
    assert autonomy_state(flags, False)[0] == "Awaiting approval"
    flags["LOCO_ALLOW_UNATTENDED"] = "1"
    assert autonomy_state(flags, True)[0] == "Paused"
    assert autonomy_state(flags, False)[0] == "Authorized"


def test_heatmap_alignment_and_artifact_allowlist(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    job = runtime / "jobs" / "run-1"
    job.mkdir(parents=True)
    monkeypatch.setenv("LOCO_RUNTIME", str(runtime))
    run = {"run_id": "run-1", "artifact_dir": str(job)}
    for name in ("normal_preview.png", "synthetic_input.png", "synthetic_heatmap.png", "synthetic_mask.png"):
        Image.new("RGB", (80, 40), "red").save(job / name)
    sample = heatmap_sample(run, 0.5)
    assert sample["size"] == (80, 40)
    assert sample["overlay"].size == sample["original"].size
    Image.new("RGB", (40, 80), "red").save(job / "synthetic_heatmap.png")
    assert heatmap_sample(run) is None
    assert allowed_preview(run, "../../controller/experiments.sqlite") is None
    outside = tmp_path / "private.png"
    Image.new("RGB", (80, 40)).save(outside)
    (job / "synthetic_heatmap.png").unlink()
    (job / "synthetic_heatmap.png").symlink_to(outside)
    assert allowed_preview(run, "synthetic_heatmap.png") is None


def test_four_page_navigation_and_empty_page_rendering(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCO_RUNTIME", str(tmp_path))
    monkeypatch.setenv("LOCO_LEDGER", str(tmp_path / "missing.db"))
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    source = app.read_text()
    assert source.count("st.Page(") == 4
    for label in ("Home", "Research", "Results", "System"):
        assert f'title="{label}"' in source
    for page in (app, *(app.parent / "pages" / name for name in ("01_Home.py", "02_Research.py", "03_Results.py", "04_System.py"))):
        rendered = AppTest.from_file(str(page), default_timeout=10).run()
        assert not rendered.exception, (page, [item.message for item in rendered.exception])


def test_home_authorized_action_disappears_when_queue_is_paused(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.sqlite"
    store = StateStore(ledger)
    monkeypatch.setenv("LOCO_RUNTIME", str(tmp_path))
    monkeypatch.setenv("LOCO_LEDGER", str(ledger))
    for name in ("LOCO_SANDBOX_VERIFIED", "LOCO_CODEX_ISOLATION_VERIFIED", "LOCO_ALLOW_UNATTENDED"):
        monkeypatch.setenv(name, "1")
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    rendered = AppTest.from_file(str(app), default_timeout=10).run()
    assert not rendered.exception
    assert next(item.value for item in rendered.metric if item.label == "Autonomous research") == "Authorized"
    assert len(rendered.get("page_link")) == 1
    store.set_paused(True)
    rendered.run()
    assert next(item.value for item in rendered.metric if item.label == "Autonomous research") == "Paused"
    assert not rendered.get("page_link")
