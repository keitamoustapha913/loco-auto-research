"""Durable MLflow outbox consumer. SQLite remains the authoritative ledger."""
from __future__ import annotations

import json
import logging
import os
import signal
import time

from orchestrator.mlflow_tracker import sync_outbox
from orchestrator.state_store import StateStore


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('tracking')
    stopping = False

    def stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    store = StateStore()
    while not stopping:
        try:
            result = sync_outbox(store)
            from orchestrator.storage_cleanup import maybe_cleanup
            maybe_cleanup(store.path.parent.parent, store, reason='tracking idle housekeeping')
            if result['delivered'] or result['failed']:
                logger.info(json.dumps({'event': 'tracking_sync', **result, 'time': time.time()}))
        except Exception as exc:
            logger.error(json.dumps({'event': 'tracking_error', 'error': str(exc), 'time': time.time()}))
        for _ in range(20):
            if stopping:
                break
            time.sleep(0.5)


if __name__ == '__main__':
    main()
