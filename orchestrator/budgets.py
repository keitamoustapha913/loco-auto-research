"""Operator-managed compute budgets; controls are external to agent JSON."""
from __future__ import annotations
import os
from pathlib import Path

class BudgetExceeded(RuntimeError):
    pass


class CodexBudgetExhausted(BudgetExceeded):
    """A denied call is a stop condition, not a failed research attempt."""


def enforce(store, next_gpu_seconds=0, runtime=None, campaign_id=None):
    if store.is_paused():
        raise BudgetExceeded("controller paused by operator")
    elapsed_campaign = False
    if campaign_id:
        import json, time
        from orchestrator.campaign_config import time_based, remaining_wall
        with store.connect() as con:
            row = con.execute('SELECT * FROM campaigns WHERE id=?', (campaign_id,)).fetchone()
        if row is None or row['status'] != 'RUNNING':
            raise BudgetExceeded('campaign is not running')
        elapsed_campaign = time_based(json.loads(row['config_json']))
        if elapsed_campaign:
            remaining = remaining_wall(row, time.time())
            if remaining <= 0 or remaining < float(next_gpu_seconds):
                raise BudgetExceeded('insufficient remaining research duration')
    if not elapsed_campaign and store.total_gpu_seconds() + float(next_gpu_seconds) > float(os.getenv("LOCO_GPU_BUDGET_HOURS", "4")) * 3600:
        raise BudgetExceeded("campaign GPU-hour budget exceeded")
    root = Path(runtime or os.getenv("LOCO_RUNTIME", ".runtime"))
    max_bytes = float(os.getenv("LOCO_DISK_BUDGET_GIB", "100")) * 2**30
    if root.exists():
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())
        if size > max_bytes:
            raise BudgetExceeded("runtime disk budget exceeded")


def max_job_seconds(config=None):
    if config and config.get('plan_contract_version')=='experiment-plan-v2.0':
        import time
        return max(0, config['research_deadline']-time.time())
    if config and config.get('campaign_id'):
        # Trusted queue compiler supplies this value only after plan admission.
        from orchestrator.campaign_config import TIME_VERSION
        if config.get('campaign_configuration_version') == TIME_VERSION:
            return max(1, int(config['timeout_seconds']))
    return max(1, int(os.getenv("LOCO_JOB_TIMEOUT_SECONDS", "1800")))
