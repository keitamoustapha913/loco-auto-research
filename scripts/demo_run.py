"""Generate artificial normal images and run the full ledger/worker/MLflow pipeline."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from benchmark.data_guard import build_manifest, manifest_fingerprint
from orchestrator.state_store import StateStore
from orchestrator.controller import run_worker_once


def make_fixture(root: Path):
    from benchmark.data_guard import CATEGORIES
    for cat in CATEGORIES:
        for split, count in (("train/good", 4), ("validation/good", 3)):
            folder = root / cat / split
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                pic = Image.new("RGB", (64,64), (185, 188, 190))
                draw = ImageDraw.Draw(pic)
                draw.rectangle((8, 12, 25, 29), fill=(70 + i*3, 80, 115))
                draw.ellipse((34, 28, 50, 44), fill=(102, 138 + i*2, 65))
                pic.save(folder / f"{i:04}.png")


def main():
    root = Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()
    os.environ["LOCO_RUNTIME"] = str(root)
    db = root / "controller" / "experiments.sqlite"
    os.environ.setdefault("LOCO_LEDGER", str(db))
    toy_data = root / "good_data"
    if not toy_data.exists() or not list(toy_data.rglob("*.png")):
        make_fixture(toy_data)
    manifest = build_manifest(toy_data)
    manifest_path = root / "good_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    store = StateStore(db)
    run_id = store.create_run("toy-fixture", "toy_normal_ae", "fixture", "smoke", {"device": "cpu", "epochs": 1, "size": 64, "seed": 7, "timeout_seconds": 300}, manifest_sha256=manifest_fingerprint(manifest))
    for status in ("IMPLEMENTED", "VALIDATED", "QUEUED"):
        store.transition(run_id, status)
    outcome = run_worker_once(store, manifest_path, root)
    from orchestrator.mlflow_tracker import sync_outbox
    try:
        tracking = sync_outbox(store)
    except Exception as exc:
        tracking = {"offline": str(exc), "pending": len(store.pending_tracking())}
    print(json.dumps({"job": outcome, "tracking": tracking, "dashboard_ledger": str(db)}, indent=2))

if __name__ == "__main__":
    main()
