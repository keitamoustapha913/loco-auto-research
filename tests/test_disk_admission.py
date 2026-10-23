"""Storage protection is independent of free filesystem space and candidate repair."""
import pytest
from orchestrator.budgets import enforce_disk, DiskBudgetExceeded, BudgetExceeded


def test_runtime_quota_includes_history_and_reports_precise_consumption(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCO_DISK_BUDGET_GIB',str(100/2**30))
    (tmp_path/'history').write_bytes(b'h'*101)
    with pytest.raises(DiskBudgetExceeded) as caught:enforce_disk(tmp_path)
    detail=caught.value.detail
    assert isinstance(caught.value,BudgetExceeded)
    assert detail['observed_bytes']==101 and detail['limit_bytes']==100
    assert detail['filesystem_free_bytes']>100
    assert 'scope=entire runtime' in str(caught.value)
    assert 'observed_bytes=101' in str(caught.value)
    assert (tmp_path/'history').read_bytes()==b'h'*101


def test_complete_matrix_copy_lower_bound_is_checked_before_any_write(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCO_DISK_BUDGET_GIB',str(100/2**30))
    (tmp_path/'history').write_bytes(b'h'*70)
    assert enforce_disk(tmp_path,required_bytes=30)['observed_bytes']==70
    with pytest.raises(DiskBudgetExceeded,match='required_additional_bytes=31'):enforce_disk(tmp_path,required_bytes=31)
    assert list(tmp_path.iterdir())==[tmp_path/'history']


def test_real_filesystem_exhaustion_is_rejected_even_with_runtime_allowance(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('orchestrator.budgets.shutil.disk_usage',lambda _:SimpleNamespace(free=5))
    with pytest.raises(DiskBudgetExceeded,match='insufficient filesystem free space'):enforce_disk(tmp_path,required_bytes=6)


def test_no_external_symlink_targets_charged_or_followed(tmp_path,monkeypatch):
    root=tmp_path/'runtime';root.mkdir()
    external=tmp_path/'outside';external.write_bytes(b'h'*100)
    (root/'link').symlink_to(external)
    monkeypatch.setenv('LOCO_DISK_BUDGET_GIB',str(5/2**30))
    assert enforce_disk(root)['observed_bytes']==0
