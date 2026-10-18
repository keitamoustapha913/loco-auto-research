"""Batch-one end-to-end and model-only timing (CUDA synchronized)."""
from __future__ import annotations
import time
import numpy as np
from benchmark.train_adapter import validate_prediction


def _sync(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except ImportError:
        pass


def benchmark_predict(predictor, images, device="cpu", warmup=3, repeats=15, preprocess=None, postprocess=None):
    """preprocess/postprocess are measured inside e2e timer; cannot claim device-copy timing absent a copy."""
    if len(images) != 1 or repeats <= 0 or warmup < 0:
        raise ValueError("expect batch-one, positive repeats and nonnegative warmup")
    model_ms, total_ms = [], []
    for iteration in range(warmup + repeats):
        _sync(device)
        t0 = time.perf_counter()
        x = preprocess(images) if preprocess else images
        _sync(device)
        t1 = time.perf_counter()
        out = predictor.predict(x)
        _sync(device)
        t2 = time.perf_counter()
        out = postprocess(out) if postprocess else out
        validate_prediction(out, tuple(images.shape))
        _sync(device)
        t3 = time.perf_counter()
        if iteration >= warmup:
            model_ms.append((t2 - t1) * 1000)
            total_ms.append((t3 - t0) * 1000)
    result = {"inference/model_ms_p50": float(np.median(model_ms)), "inference/e2e_ms_p50": float(np.median(total_ms)), "inference/e2e_ms_p95": float(np.percentile(total_ms, 95)), "inference/batch_size": 1}
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            result["inference/peak_vram_mib"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
    except ImportError:
        pass
    return result
