from pathlib import Path
from dashboard.data_access import allowed_preview, list_runs, get_mlflow_url
from dashboard.charts import run_table

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
