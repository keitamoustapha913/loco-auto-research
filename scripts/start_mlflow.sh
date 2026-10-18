#!/usr/bin/env bash
set -euo pipefail
RUNTIME="${LOCO_RUNTIME:-$PWD/.runtime}"
mkdir -p "$RUNTIME/mlflow/artifacts"
exec mlflow server \
    --backend-store-uri "sqlite:///$RUNTIME/mlflow/mlflow.db" \
    --artifacts-destination "$RUNTIME/mlflow/artifacts" \
    --host 127.0.0.1 --port 5000
