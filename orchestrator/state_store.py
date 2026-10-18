"""Transactional operational ledger; numerical measurements originate in trusted runners.

Do not treat SQLite permissions as a sandbox: the writer process must be trusted.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

STATES = {
    "PROPOSED", "IMPLEMENTED", "VALIDATED", "QUEUED", "RUNNING", "COMPLETED",
    "REVIEWED", "PROMOTED", "REJECTED", "INVALID", "FAILED", "OOM", "TIMED_OUT", "CANCELLED", "INTERRUPTED"
}
TRANSITIONS = {
    "PROPOSED": {"IMPLEMENTED", "INVALID", "CANCELLED"},
    "IMPLEMENTED": {"VALIDATED", "INVALID", "CANCELLED"},
    "VALIDATED": {"QUEUED", "REJECTED", "CANCELLED"},
    "QUEUED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "OOM", "TIMED_OUT", "CANCELLED", "INTERRUPTED"},
    "COMPLETED": {"REVIEWED"},
    "REVIEWED": {"PROMOTED", "REJECTED"},
}


def default_db_path():
    return Path(os.getenv("LOCO_LEDGER", str(Path(os.getenv("LOCO_RUNTIME", ".runtime")) / "controller" / "experiments.sqlite"))).expanduser()


class StateStore:
    def __init__(self, path=None, read_only=False):
        self.path = Path(path or default_db_path()).expanduser().resolve()
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.initialize()

    def connect(self):
        if self.read_only:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10)
            con.execute("PRAGMA query_only=ON")
        else:
            con = sqlite3.connect(str(self.path), timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=15000")
        return con

    def initialize(self):
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, hypothesis_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL, category TEXT NOT NULL,
                    stage TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    config_json TEXT NOT NULL, git_sha TEXT DEFAULT '',
                    manifest_sha256 TEXT DEFAULT '', mlflow_run_id TEXT DEFAULT '',
                    metrics_json TEXT DEFAULT '{}', error TEXT DEFAULT '', gpu_seconds REAL DEFAULT 0,
                    artifact_dir TEXT DEFAULT '', lease_owner TEXT DEFAULT '',
                    started_at REAL, finished_at REAL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                    time REAL NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY, created_at REAL NOT NULL,
                    proposal_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PROPOSED'
                );
                CREATE TABLE IF NOT EXISTS tracking_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL, delivered_at REAL, attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO settings(name,value) VALUES('paused','0');
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, time);
            """)

    def add_proposal(self, proposal):
        now = time.time()
        with self.connect() as con:
            con.execute("INSERT INTO hypotheses(hypothesis_id,created_at,proposal_json,status) VALUES(?,?,?,?)", (proposal["hypothesis_id"], now, json.dumps(proposal, sort_keys=True), "PROPOSED"))
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)", (now, "PROPOSAL", json.dumps({"hypothesis_id": proposal["hypothesis_id"]})))

    def set_hypothesis_status(self, hypothesis_id, status):
        if status not in {"PROPOSED", "IMPLEMENTED", "INVALID", "QUEUED", "COMPLETED", "FAILED"}:
            raise ValueError("invalid hypothesis status")
        with self.connect() as con:
            cursor = con.execute("UPDATE hypotheses SET status=? WHERE hypothesis_id=?", (status, hypothesis_id))
            if not cursor.rowcount:
                raise KeyError(hypothesis_id)

    def create_run(self, hypothesis_id, candidate_id, category, stage="smoke", config=None, git_sha="", manifest_sha256="", run_id=None):
        if category not in ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors", "fixture"):
            raise ValueError("unknown category")
        run_id = run_id or str(uuid.uuid4())
        now = time.time()
        with self.connect() as con:
            con.execute("INSERT INTO runs(run_id,hypothesis_id,candidate_id,category,stage,status,created_at,updated_at,config_json,git_sha,manifest_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, hypothesis_id, candidate_id, category, stage, "PROPOSED", now, now, json.dumps(config or {}, sort_keys=True), git_sha, manifest_sha256))
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)", (run_id, now, "CREATED", "{}"))
        return run_id

    def get_run(self, run_id):
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit=250):
        with self.connect() as con:
            rows = con.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (min(int(limit), 5000),)).fetchall()
        return [dict(r) for r in rows]

    def events(self, limit=100):
        with self.connect() as con:
            rows = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (min(int(limit), 1000),)).fetchall()
        return [dict(r) for r in rows]

    def transition(self, run_id, next_state, reason="", expect=None):
        if next_state not in STATES:
            raise ValueError("unknown state")
        now = time.time()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            old = row["status"]
            if expect and old != expect:
                raise ValueError(f"race: expected {expect}, got {old}")
            if next_state not in TRANSITIONS.get(old, set()):
                raise ValueError(f"invalid transition {old}->{next_state}")
            updates = {"status": next_state, "updated_at": now}
            if next_state == "RUNNING":
                updates["started_at"] = now
            if next_state in {"COMPLETED", "FAILED", "OOM", "TIMED_OUT", "CANCELLED", "INTERRUPTED", "INVALID", "REJECTED", "PROMOTED"}:
                updates["finished_at"] = now
            if reason:
                updates["error"] = reason[:2000]
            query = ",".join(f"{k}=?" for k in updates)
            con.execute(f"UPDATE runs SET {query} WHERE run_id=?", (*updates.values(), run_id))
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)", (run_id, now, "TRANSITION", json.dumps({"from": old, "to": next_state, "reason": reason})))

    def set_run_result(self, run_id, metrics, gpu_seconds=0, artifact_dir=""):
        if not isinstance(metrics, dict) or not all(isinstance(v, (float, int)) for v in metrics.values()):
            raise TypeError("metrics must be validated numeric values")
        import math
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError("nonfinite metrics")
        with self.connect() as con:
            row = con.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row or row["status"] != "RUNNING":
                raise ValueError("results may be recorded only for RUNNING job")
            con.execute("UPDATE runs SET metrics_json=?, gpu_seconds=?, artifact_dir=?, updated_at=? WHERE run_id=?", (json.dumps(metrics, sort_keys=True), float(gpu_seconds), str(artifact_dir), time.time(), run_id))

    def account_gpu_usage(self, run_id, gpu_seconds):
        """Charge failed/timeout GPU jobs too; a failed run still consumed compute."""
        with self.connect() as con:
            con.execute("UPDATE runs SET gpu_seconds=?, updated_at=? WHERE run_id=?", (max(0.0, float(gpu_seconds)), time.time(), run_id))

    def codex_call_count(self):
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) AS n FROM events WHERE kind='CODEX_CALL'").fetchone()["n"])

    def reserve_codex_call(self, role):
        max_calls = int(os.getenv("LOCO_CODEX_MAX_CALLS", "30"))
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            count = con.execute("SELECT COUNT(*) AS n FROM events WHERE kind='CODEX_CALL'").fetchone()["n"]
            if count >= max_calls:
                raise RuntimeError("Codex call budget exhausted; obtain operator approval")
            con.execute("INSERT INTO events(run_id,time,kind,payload_json) VALUES(NULL,?,?,?)", (time.time(), "CODEX_CALL", json.dumps({"role": role, "count": count+1})))

    def set_mlflow_id(self, run_id, mlflow_run_id):
        with self.connect() as con:
            con.execute("UPDATE runs SET mlflow_run_id=? WHERE run_id=?", (mlflow_run_id, run_id))

    def set_paused(self, paused):
        with self.connect() as con:
            con.execute("UPDATE settings SET value=? WHERE name='paused'", ("1" if paused else "0",))

    def is_paused(self):
        with self.connect() as con:
            row = con.execute("SELECT value FROM settings WHERE name='paused'").fetchone()
        return bool(row and row["value"] == "1")

    def total_gpu_seconds(self):
        with self.connect() as con:
            row = con.execute("SELECT COALESCE(SUM(gpu_seconds),0) AS total FROM runs").fetchone()
        return float(row["total"])

    def enqueue_tracking(self, key, run_id, payload):
        with self.connect() as con:
            con.execute("INSERT OR IGNORE INTO tracking_outbox(event_key,run_id,payload_json,created_at) VALUES(?,?,?,?)", (key, run_id, json.dumps(payload, sort_keys=True), time.time()))

    def pending_tracking(self, limit=50):
        with self.connect() as con:
            rows = con.execute("SELECT * FROM tracking_outbox WHERE delivered_at IS NULL ORDER BY id LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def tracking_delivered(self, row_id):
        with self.connect() as con:
            con.execute("UPDATE tracking_outbox SET delivered_at=?, attempts=attempts+1, last_error='' WHERE id=?", (time.time(), row_id))

    def tracking_failed(self, row_id, error):
        with self.connect() as con:
            con.execute("UPDATE tracking_outbox SET attempts=attempts+1, last_error=? WHERE id=?", (str(error)[:200], row_id))

    def reconcile_after_restart(self):
        """Never rerun a previously RUNNING process automatically."""
        with self.connect() as con:
            ids = [r["run_id"] for r in con.execute("SELECT run_id FROM runs WHERE status='RUNNING'").fetchall()]
        for run_id in ids:
            self.transition(run_id, "INTERRUPTED", reason="Worker restarted; manual investigation required", expect="RUNNING")
        return ids
