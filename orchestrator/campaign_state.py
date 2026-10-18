"""Durable campaign bookkeeping in the existing authoritative SQLite ledger."""
from __future__ import annotations

import json
import time
import uuid


ACTIVE = {"RUNNING", "PAUSED", "STOPPING"}
TERMINAL = {"STOPPED", "LIMIT_REACHED", "FAILED", "COMPLETED"}


class CampaignState:
    def __init__(self, store):
        self.store = store
        if store.read_only:
            return
        with store.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    config_json TEXT NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, ended_at REAL,
                    baseline_gpu_seconds REAL NOT NULL,
                    baseline_codex_calls INTEGER NOT NULL,
                    codex_calls INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    active_run_id TEXT NOT NULL DEFAULT '',
                    current_hypothesis TEXT NOT NULL DEFAULT '',
                    next_hypothesis TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS campaign_runs (
                    run_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                    FOREIGN KEY(campaign_id) REFERENCES campaigns(id)
                );
                CREATE TABLE IF NOT EXISTS campaign_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT NOT NULL,
                    time REAL NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_campaign_events ON campaign_events(campaign_id,id);
            """)

    def latest(self):
        import sqlite3
        try:
            with self.store.connect() as con:
                row = con.execute("SELECT * FROM campaigns ORDER BY created_at DESC LIMIT 1").fetchone()
        except sqlite3.OperationalError as exc:
            if self.store.read_only and 'no such table' in str(exc):
                return None
            raise
        return dict(row) if row else None

    def get(self, campaign_id):
        with self.store.connect() as con:
            row = con.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        return dict(row) if row else None

    def create(self, config):
        now = time.time()
        campaign_id = str(uuid.uuid4())
        baseline_gpu = self.store.total_gpu_seconds()
        baseline_calls = self.store.codex_call_count()
        with self.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            active = con.execute("SELECT id FROM campaigns WHERE status IN ('RUNNING','PAUSED','STOPPING') LIMIT 1").fetchone()
            if active:
                raise RuntimeError(f"campaign {active['id']} is already active")
            con.execute("INSERT INTO campaigns(id,name,status,config_json,created_at,updated_at,baseline_gpu_seconds,baseline_codex_calls) VALUES(?,?,?,?,?,?,?,?)",
                        (campaign_id, config["campaign"]["name"], "RUNNING", json.dumps(config, sort_keys=True), now, now,
                         baseline_gpu, baseline_calls))
            con.execute("INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)",
                        (campaign_id, now, "STARTED", "{}"))
        return campaign_id

    def set_status(self, campaign_id, status, reason=""):
        if status not in ACTIVE | TERMINAL:
            raise ValueError("invalid campaign status")
        now = time.time()
        with self.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if row is None:
                raise KeyError(campaign_id)
            if row["status"] in TERMINAL and status in ACTIVE:
                raise ValueError("stopped campaigns cannot be resumed")
            con.execute("UPDATE campaigns SET status=?,reason=?,updated_at=?,ended_at=? WHERE id=?",
                        (status, reason[:1000], now, now if status in TERMINAL else None, campaign_id))
            con.execute("INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)",
                        (campaign_id, now, status, json.dumps({"reason": reason[:1000]})))

    def update(self, campaign_id, **fields):
        allowed = {"active_run_id", "current_hypothesis", "next_hypothesis", "consecutive_failures", "reason"}
        if not fields or set(fields) - allowed:
            raise ValueError("invalid campaign update")
        with self.store.connect() as con:
            con.execute(f"UPDATE campaigns SET {','.join(k+'=?' for k in fields)},updated_at=? WHERE id=?",
                        (*fields.values(), time.time(), campaign_id))

    def event(self, campaign_id, kind, **payload):
        with self.store.connect() as con:
            con.execute("INSERT INTO campaign_events(campaign_id,time,kind,payload_json) VALUES(?,?,?,?)",
                        (campaign_id, time.time(), kind, json.dumps(payload, sort_keys=True, default=str)))

    def events(self, campaign_id, limit=100):
        with self.store.connect() as con:
            records = con.execute("SELECT * FROM campaign_events WHERE campaign_id=? ORDER BY id DESC LIMIT ?", (campaign_id, limit)).fetchall()
        return [dict(r) for r in records]

    def reserve_codex_call(self, campaign_id):
        with self.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT config_json,codex_calls,status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not row or row["status"] != "RUNNING":
                raise RuntimeError("campaign is not running")
            maximum = int(json.loads(row["config_json"])["limits"]["max_codex_calls"])
            if row["codex_calls"] >= maximum:
                raise RuntimeError("campaign Codex-call budget exhausted")
            con.execute("UPDATE campaigns SET codex_calls=codex_calls+1,updated_at=? WHERE id=?", (time.time(), campaign_id))

    def attach_run(self, campaign_id, run_id):
        with self.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("INSERT INTO campaign_runs(run_id,campaign_id) VALUES(?,?)", (run_id, campaign_id))
            con.execute("UPDATE campaigns SET active_run_id=?,updated_at=? WHERE id=?", (run_id, time.time(), campaign_id))

    def runs(self, campaign_id):
        with self.store.connect() as con:
            records = con.execute("SELECT r.* FROM runs r JOIN campaign_runs cr ON cr.run_id=r.run_id WHERE cr.campaign_id=? ORDER BY r.created_at", (campaign_id,)).fetchall()
        return [dict(r) for r in records]

    def usage(self, campaign_id):
        row = self.get(campaign_id)
        if not row:
            raise KeyError(campaign_id)
        runs = self.runs(campaign_id)
        return {"gpu_seconds": sum(float(r["gpu_seconds"] or 0) for r in runs),
                "experiments": len(runs), "codex_calls": row["codex_calls"],
                "wall_seconds": (row['ended_at'] or time.time()) - row["created_at"]}
