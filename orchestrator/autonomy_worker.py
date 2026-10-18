"""Detached trusted campaign worker. Started only through loco-lab research start."""
from __future__ import annotations

import argparse
import sys

from orchestrator.autonomy import run_forever
from orchestrator.lab_cli import load_settings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args(argv)
    runtime, ledger = load_settings()
    try:
        run_forever(args.campaign_id, runtime, ledger)
        return 0
    except Exception as exc:
        print(f"campaign worker: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
