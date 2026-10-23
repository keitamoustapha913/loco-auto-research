"""Operator-managed compute budgets; controls are external to agent JSON."""
from __future__ import annotations
import os
import shutil
from pathlib import Path

class BudgetExceeded(RuntimeError):
    pass


class CodexBudgetExhausted(BudgetExceeded):
    """A denied call is a stop condition, not a failed research attempt."""


class DiskBudgetExceeded(BudgetExceeded):
    """An operator storage guard, not an error in generated candidate code."""
    def __init__(self, detail):
        self.detail = detail
        super().__init__(
            'runtime disk budget exceeded: scope=entire runtime; '
            f"root={detail['root']}; observed_bytes={detail['observed_bytes']}; "
            f"required_additional_bytes={detail['required_additional_bytes']}; "
            f"limit_bytes={detail['limit_bytes']}; filesystem_free_bytes={detail['filesystem_free_bytes']}; "
            f"policy={detail['policy']}; reason={detail['reason']}. "
            'Preserve research evidence; operator storage review is required. '
            'Free filesystem space does not override the runtime quota.')


def enforce_disk(runtime=None, *, required_bytes=0):
    """Existing logical-byte quota, including historical evidence, with context.

    required_bytes is a lower bound from concrete pending copies, not an agent
    estimate. No historical exclusions, auto-cleanup or quota expansion.
    """
    if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
        raise ValueError('disk admission requires nonnegative integer bytes')
    root = Path(runtime or os.getenv('LOCO_RUNTIME', '.runtime'))
    limit = float(os.getenv('LOCO_DISK_BUDGET_GIB', '100')) * 2**30
    size = 0
    if root.exists():
        for path in root.rglob('*'):
            try:
                if path.is_file() and not path.is_symlink():
                    size += path.stat().st_size
            except FileNotFoundError:
                continue  # a concurrent temporary worker file was removed
    ancestor = root.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    detail = dict(root=str(root.resolve()), observed_bytes=size, required_additional_bytes=required_bytes,
                  limit_bytes=limit, filesystem_free_bytes=free,
                  policy='LOCO_DISK_BUDGET_GIB' if 'LOCO_DISK_BUDGET_GIB' in os.environ else 'default 100 GiB runtime quota')
    if size + required_bytes > limit or required_bytes > free:
        detail['reason'] = 'runtime logical-byte quota' if size + required_bytes > limit else 'insufficient filesystem free space'
        raise DiskBudgetExceeded(detail)
    return detail


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
    enforce_disk(runtime)


def max_job_seconds(config=None):
    if config and config.get('plan_contract_version') in {'experiment-plan-v2.0','experiment-plan-v3.0','experiment-plan-v4.0'}:
        import time
        return max(0, config['research_deadline']-time.time())
    if config and config.get('campaign_id'):
        # Trusted queue compiler supplies this value only after plan admission.
        from orchestrator.campaign_config import TIME_VERSION
        if config.get('campaign_configuration_version') == TIME_VERSION:
            return max(1, int(config['timeout_seconds']))
    return max(1, int(os.getenv("LOCO_JOB_TIMEOUT_SECONDS", "1800")))
