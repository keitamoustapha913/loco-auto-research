"""Entry point inside restricted candidate sandbox; validates approved good manifest.

IMPORTANT: this executes untrusted model code. Deployment MUST use OS isolation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from benchmark.data_guard import load_manifest, manifest_fingerprint
from benchmark.train_adapter import load_candidate, validate_prediction
from benchmark.validation import evaluate_dev
from benchmark.inference_benchmark import benchmark_predict


def load_image(path, size):
    im = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy() / 255


def synthetic_corruption(x, seed=0):
    """Development-only: visibly imperfect occlusion, used to test pipeline, NOT claims."""
    rng = np.random.default_rng(seed)
    changed = x.copy()
    _, h, w = x.shape
    r = min(h, w) // 5
    y = int(rng.integers(h // 4, max(h // 4+1, h-r-h//4)))
    xx = int(rng.integers(w // 4, max(w // 4+1, w-r-w//4)))
    changed[:, y:y+r, xx:xx+r] = 0.95
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y+r, xx:xx+r] = 1
    return changed, mask


def save_heatmaps(job_dir, good_image, synthetic_image, anomaly_map):
    from PIL import Image
    arr = np.asarray(anomaly_map)
    lo, hi = np.percentile(arr, [2, 98])
    score = np.clip((arr - lo) / max(1e-8, hi - lo), 0, 1)
    image_rgb = (synthetic_image.transpose(1,2,0) * 255).astype(np.uint8)
    good_rgb = (good_image.transpose(1,2,0) * 255).astype(np.uint8)
    heatmap = np.stack((score * 255, np.zeros_like(score), (1 - score) * 70), axis=-1).astype(np.uint8)
    overlay = (image_rgb.astype(float) * 0.62 + heatmap.astype(float) * 0.38).clip(0,255).astype(np.uint8)
    Image.fromarray(good_rgb).save(job_dir / "normal_preview.png")
    Image.fromarray(heatmap).save(job_dir / "synthetic_heatmap.png")
    Image.fromarray(overlay).save(job_dir / "synthetic_overlay.png")


def run(module_file, manifest_file, config, output):
    from benchmark.data_guard import sha256_file
    from benchmark.metrics import synthetic_region_pro_proxy
    output.mkdir(parents=True, exist_ok=True)
    manifest, paths = load_manifest(manifest_file)
    category = config.get("category", "fixture")
    if category != "fixture":
        # Freeze a category-only authorized subset for fair single-category runs.
        entries = [e for e in manifest["images"] if e["category"] == category]
        if not entries:
            raise ValueError(f"No authorized images for requested category: {category}")
        manifest = dict(manifest, images=entries)
        manifest_file = output / "category_good_manifest.json"
        manifest_file.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        from benchmark.data_guard import validate_manifest
        paths = validate_manifest(manifest)
    size = int(config.get("size", 64))
    device = config.get("device", "cpu")
    mod = load_candidate(module_file)
    checkpoint_meta = mod.train(config, manifest_file, output / "checkpoint")
    predictor = mod.load(output / "checkpoint" / "checkpoint_manifest.json", device=device)
    valid_paths = [p for p, entry in zip(paths, manifest["images"]) if entry["split"] == "validation/good"]
    if len(valid_paths) < 2:
        raise ValueError("Need >=2 normal validation images for disjoint calibration and assessment")
    # A disjoint good subset calibrates scores; remaining good images assess FPR.
    calibrate_paths = valid_paths[:max(1, len(valid_paths) // 3)]
    score_paths = valid_paths[len(calibrate_paths):]
    calibration = np.stack([load_image(p, size) for p in calibrate_paths])
    normal = np.stack([load_image(p, size) for p in score_paths])
    corrupt, masks = zip(*(synthetic_corruption(im, i + 5000) for i, im in enumerate(normal)))
    synthetic = np.stack(corrupt)
    all_pred_normal, all_pred_synth, all_pred_cal = [], [], []
    for array, store in ((calibration, all_pred_cal), (normal, all_pred_normal), (synthetic, all_pred_synth)):
        for im in array:
            response = predictor.predict(im[None])
            maps, _ = validate_prediction(response, (1, 3, size, size))
            store.append(maps[0, 0])
    all_pred_cal = np.stack(all_pred_cal)
    all_pred_normal = np.stack(all_pred_normal)
    all_pred_synth = np.stack(all_pred_synth)
    metrics = evaluate_dev(all_pred_normal, all_pred_synth, np.stack(masks), calibration_maps=all_pred_cal)
    normal_threshold = metrics["normal/threshold"]
    # Appearance edits that maintain the fixture's object layout test sensitivity
    # to superficial edits; these are not a substitute for real logical controls.
    edited_good = np.clip(normal * 0.93 + 0.03, 0, 1)
    edited_scores = []
    for im in edited_good:
        result = predictor.predict(im[None])
        _, scores = validate_prediction(result, (1, 3, size, size))
        edited_scores.append(float(scores[0]))
    metrics["robustness/edited_good_fpr"] = float(np.mean(np.asarray(edited_scores) > normal_threshold))
    metrics.update(benchmark_predict(predictor, normal[:1], device=device, warmup=2, repeats=8))
    try:
        region_proxy = synthetic_region_pro_proxy(all_pred_synth, np.stack(masks), steps=20)
        if np.isfinite(region_proxy):
            metrics["dev_synthetic/region_pro_proxy_NOT_OFFICIAL"] = region_proxy
    except RuntimeError:
        pass
    checkpoint = output / "checkpoint" / "model.pt"
    metrics["inference/checkpoint_mib"] = checkpoint.stat().st_size / 2**20
    try:
        import torch
        model = getattr(predictor, "net", None)
        if model is not None:
            metrics["inference/parameters"] = sum(p.numel() for p in model.parameters())
    except ImportError:
        pass
    save_heatmaps(output, normal[0], synthetic[0], all_pred_synth[0])
    import sys
    provenance = {"protocol_version": "1.0", "python_version": sys.version.split()[0], "development_calibration_count":len(calibrate_paths), "normal_assessment_count":len(score_paths), "dependencies_pyproject_sha256":sha256_file(Path(__file__).resolve().parents[1] / "pyproject.toml"), "config_sha256": __import__("hashlib").sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "metric_status": "SYNTHETIC_DEVELOPMENT_PROXY_ONLY", "model_module_sha256": sha256_file(module_file), "good_manifest_sha256": manifest_fingerprint(manifest), "checkpoint_sha256": sha256_file(checkpoint), "image_count_normal_validation": len(valid_paths), "device": device, "size": size}
    provenance["artifact_hashes"] = {filename:sha256_file(output / filename) for filename in ("normal_preview.png", "synthetic_heatmap.png", "synthetic_overlay.png")}
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2))
    return {"metrics": {k: float(v) for k,v in metrics.items()}, "provenance": provenance}


def main():
    parser = argparse.ArgumentParser()
    for key in ("module", "manifest", "config", "output", "result"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    output = run(Path(args.module), Path(args.manifest), json.loads(Path(args.config).read_text()), Path(args.output))
    Path(args.result).write_text(json.dumps(output, indent=2))
    print(json.dumps({"metric_kind": "synthetic_proxy", "metrics": output["metrics"]}))

if __name__ == "__main__":
    main()
