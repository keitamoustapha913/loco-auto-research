"""Consistent SQLite backup for trusted operator; never touches sealed test data."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sqlite3
from datetime import datetime, timezone
from orchestrator.state_store import default_db_path


def backup_db(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as conn_in, sqlite3.connect(str(destination)) as conn_out:
        conn_in.backup(conn_out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    out=Path(args.output).resolve()
    runtime=Path(os.getenv('LOCO_RUNTIME','.runtime')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    files={'controller.sqlite':default_db_path(), 'mlflow.sqlite':runtime/'mlflow'/'mlflow.db'}
    for name,src in files.items():
        if src.is_file():
            backup_db(src,out/name)
    artifacts=runtime/'mlflow'/'artifacts'
    if artifacts.is_dir():
        shutil.copytree(artifacts,out/'mlflow_artifacts',dirs_exist_ok=True)
    print(f'Backed up metadata and accessible MLflow artifacts to {out}; verify restore separately.')
if __name__=='__main__': main()
