"""Single-host exclusive GPU worker using flock and transactional claim."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
from contextlib import contextmanager
from orchestrator.state_store import StateStore
from orchestrator.budgets import enforce

class GpuBusy(RuntimeError):
    pass


@contextmanager
def gpu_lock(runtime=None):
    root = Path(runtime or os.getenv("LOCO_RUNTIME", ".runtime"))
    lock = root / "controller" / "gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GpuBusy("RTX 4090 worker already active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def claim_next(store: StateStore, max_duration=1800, expected_run_id=None):
    enforce(store, next_gpu_seconds=max_duration)
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        active = con.execute("SELECT COUNT(*) AS n FROM runs WHERE status='RUNNING'").fetchone()["n"]
        if active:
            return None
        row = con.execute("SELECT run_id FROM runs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return None
        run_id = row["run_id"]
        if expected_run_id is not None and run_id != expected_run_id:
            raise GpuBusy("another queued experiment precedes this campaign job")
    # Called while holding an OS-level flock. Transition uses a new transaction.
    store.transition(run_id, "RUNNING", expect="QUEUED")
    return store.get_run(run_id)
