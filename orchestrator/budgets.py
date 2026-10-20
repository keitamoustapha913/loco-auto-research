"""Operator-managed compute budgets; controls are external to agent JSON."""
from __future__ import annotations
import os
from pathlib import Path

class BudgetExceeded(RuntimeError):
    pass


class CodexBudgetExhausted(BudgetExceeded):
    """A denied call is a stop condition, not a failed research attempt."""


def enforce(store, next_gpu_seconds=0, runtime=None):
    if store.is_paused():
        raise BudgetExceeded("controller paused by operator")
    max_hours = float(os.getenv("LOCO_GPU_BUDGET_HOURS", "4"))
    if store.total_gpu_seconds() + float(next_gpu_seconds) > max_hours * 3600:
        raise BudgetExceeded("campaign GPU-hour budget exceeded")
    root = Path(runtime or os.getenv("LOCO_RUNTIME", ".runtime"))
    max_bytes = float(os.getenv("LOCO_DISK_BUDGET_GIB", "100")) * 2**30
    if root.exists():
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())
        if size > max_bytes:
            raise BudgetExceeded("runtime disk budget exceeded")


def max_job_seconds():
    return max(1, int(os.getenv("LOCO_JOB_TIMEOUT_SECONDS", "1800")))
