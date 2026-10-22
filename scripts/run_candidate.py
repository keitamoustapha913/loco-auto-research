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
from benchmark.train_adapter import load_candidate, load_predictor, validate_prediction, diagnostic_measurement_catalog, validate_diagnostic_output
from benchmark.validation import evaluate_dev_v2
from benchmark.metrics_v2 import EVALUATOR_VERSION, pixel_ap_report, pixel_auroc
from benchmark.inference_benchmark import benchmark_predict


def train_measured(module, config, manifest, output):
    from scripts.training_phase import run_training_phases
    if callable(getattr(module,'training_steps',None)) or config.get('training_budget_seconds') is not None:
        return run_training_phases(module,config,manifest,output,budget=config.get('training_budget_seconds'))
    # Legacy authorizations/infra fixtures retain their original calling contract.
    # This wall duration is explicitly unverified, never training-only compliance.
    device=str(config.get('device','cpu'))
    if device.startswith('cuda'):
        import torch
        torch.cuda.synchronize()
    started=time.perf_counter()
    try:
        metadata=module.train(config,manifest,output)
        if device.startswith('cuda'):torch.cuda.synchronize()
        return metadata,time.perf_counter()-started
    finally:
        (Path(output).parent/'training_timing.json').write_text(json.dumps({
            'protocol':'legacy-candidate-train-call-v1','verified':False,
            'status':'UNVERIFIABLE','function_wall_seconds':time.perf_counter()-started,
            'reason':'legacy train call includes setup; no training-only measurement'}))


def load_image(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy() / 255


def restore_anomaly_map(anomaly_map, original_size):
    """Map a model-space heatmap to the source image's (width, height) grid."""
    array = np.asarray(anomaly_map, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('expected finite 2D anomaly map')
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError('invalid original image dimensions')
    return np.asarray(Image.fromarray(array, mode='F').resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


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


def alternate_corruption(x, seed=0):
    """A second good-image-only generator for shortcut sensitivity checks."""
    rng = np.random.default_rng(seed)
    changed = x.copy()
    _, height, width = x.shape
    patch = max(2, min(height, width) // 6)
    source_y = int(rng.integers(0, height - patch + 1))
    source_x = int(rng.integers(0, width - patch + 1))
    target_y = (source_y + height // 2) % (height - patch + 1)
    target_x = (source_x + width // 2) % (width - patch + 1)
    changed[:, target_y:target_y+patch, target_x:target_x+patch] = x[:, source_y:source_y+patch, source_x:source_x+patch]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[target_y:target_y+patch, target_x:target_x+patch] = 1
    return changed, mask


def paired_patch_edits(image, same_location_good_donor, seed=0):
    """Relocated patch and two layout-preserving controls with identical blend mask.

    Both inputs must be permitted good validation images. The donor control is
    plausibly layout-preserving, not certified logically valid by annotation.
    """
    source = np.asarray(image, dtype=np.float32)
    donor = np.asarray(same_location_good_donor, dtype=np.float32)
    if source.shape != donor.shape or source.ndim != 3 or source.shape[0] != 3:
        raise ValueError('paired edits need aligned RGB good images')
    _, height, width = source.shape
    patch = max(4, min(height, width) // 6)
    rng = np.random.default_rng(seed)
    source_y = int(rng.integers(0, height - patch + 1))
    source_x = int(rng.integers(0, width - patch + 1))
    target_y = (source_y + height // 2) % (height - patch + 1)
    target_x = (source_x + width // 2) % (width - patch + 1)
    edge = min(3, patch // 4)
    yy, xx = np.mgrid[:patch, :patch]
    distance = np.minimum.reduce((yy + 1, xx + 1, patch - yy, patch - xx))
    alpha = (0.8 * np.minimum(distance / max(edge, 1), 1.0))[None]
    base = source[:, target_y:target_y + patch, target_x:target_x + patch]
    relocated = source.copy()
    control = source.copy()
    photometric = source.copy()
    relocated[:, target_y:target_y + patch, target_x:target_x + patch] = (
        base * (1 - alpha) + source[:, source_y:source_y + patch, source_x:source_x + patch] * alpha)
    control[:, target_y:target_y + patch, target_x:target_x + patch] = (
        base * (1 - alpha) + donor[:, target_y:target_y + patch, target_x:target_x + patch] * alpha)
    photometric[:, target_y:target_y + patch, target_x:target_x + patch] = (
        base * (1 - alpha) + np.clip(base * 0.95 + 0.025, 0, 1) * alpha)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[target_y:target_y + patch, target_x:target_x + patch] = 1
    return relocated, control, photometric, mask


def split_validation_paths(valid_paths):
    """Stable, disjoint good-only calibration and assessment partitions."""
    paths = list(valid_paths)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError('Need at least two distinct normal validation images')
    cut = max(1, len(paths) // 3)
    return paths[:cut], paths[cut:]


def save_heatmaps(job_dir, good_image, synthetic_image, anomaly_map, mask, original_image=None):
    from PIL import Image
    if original_image is not None:
        width, height = original_image.size
        anomaly_map = restore_anomaly_map(anomaly_map, (width, height))
        synthetic_image = np.asarray(Image.fromarray((synthetic_image.transpose(1,2,0) * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2,0,1) / 255
        mask = np.asarray(Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        good_image = np.asarray(original_image, dtype=np.float32).transpose(2,0,1) / 255
    arr = np.asarray(anomaly_map)
    lo, hi = np.percentile(arr, [2, 98])
    score = np.clip((arr - lo) / max(1e-8, hi - lo), 0, 1)
    image_rgb = (synthetic_image.transpose(1,2,0) * 255).astype(np.uint8)
    good_rgb = (good_image.transpose(1,2,0) * 255).astype(np.uint8)
    heatmap = np.stack((score * 255, np.zeros_like(score), (1 - score) * 70), axis=-1).astype(np.uint8)
    overlay = (image_rgb.astype(float) * 0.62 + heatmap.astype(float) * 0.38).clip(0,255).astype(np.uint8)
    Image.fromarray(good_rgb).save(job_dir / "normal_preview.png")
    Image.fromarray(image_rgb).save(job_dir / "synthetic_input.png")
    Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(job_dir / "synthetic_mask.png")
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
    # Candidate learning receives only train/good; validation is used later by the runner.
    training_manifest = dict(manifest, images=[e for e in manifest['images'] if e['split'] == 'train/good'])
    if not training_manifest['images']:
        raise ValueError('normal training images required')
    training_manifest_file = output / 'training_good_manifest.json'
    training_manifest_file.write_text(json.dumps(training_manifest, sort_keys=True))
    training_audit = {}
    if config.get('execution_kind') == 'frozen':
        # The trusted host copied the exact checkpoint; candidate train is never called.
        training_duration_seconds = 0.0
        training_audit = {'optimizer_updates': 0, 'source_exposures': 0, 'seed': config['seed'],
                          'measurement_source': 'trusted frozen checkpoint reuse; no learning'}
    elif config.get('plan_sha256'):
        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook
        observed = []
        def step_hook(optimizer, args, kwargs):
            observed.append(type(optimizer).__name__)
        hook = register_optimizer_step_post_hook(step_hook)
        torch.manual_seed(config['seed'])
        if str(device).startswith('cuda'): torch.cuda.manual_seed_all(config['seed'])
        try:
            checkpoint_meta, training_duration_seconds = train_measured(mod, config, training_manifest_file, output / 'checkpoint')
        finally:
            hook.remove()
        metadata = json.loads((output/'checkpoint/checkpoint_manifest.json').read_text())
        training_audit = metadata.get('training_audit', {})
        for name in ('optimizer_updates', 'source_exposures', 'seed'):
            value = training_audit.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'planned candidate must record integer training_audit.{name}')
        if training_audit['seed'] != config['seed'] or training_audit['optimizer_updates'] != len(observed):
            raise ValueError('observed optimizer updates or seed differ from training audit')
        if config.get('training_completion_policy') in {'required-training-target-v1','elapsed-training-loop-v1'}:
            timing=json.loads((output/'training_timing.json').read_text())
            if timing['completed_steps']!=len(observed) or timing['source_exposures']!=training_audit['source_exposures']:
                raise ValueError('target optimizer/exposure audit differs from trusted learning steps')
            training_audit=dict(training_audit,observed_optimizer_step_calls=len(observed),exposure_measurement_source='prepared TrainingStep source_exposures descriptors')
        else:
            for name, actual in [('expected_optimizer_updates', len(observed)),
                                 ('expected_source_exposures', training_audit['source_exposures'])]:
                if config[name] != actual:
                    raise ValueError(f'planned {name}={config[name]} but observed/reported {actual}')
            training_audit = dict(training_audit, observed_optimizer_step_calls=len(observed),
                                  optimizer_classes=sorted(set(observed)),
                                  exposure_measurement_source='candidate instrumented; not an OS-enforced counter')
    else:
        checkpoint_meta, training_duration_seconds = train_measured(mod, config, training_manifest_file, output / 'checkpoint')
    timing_audit=json.loads((output/'training_timing.json').read_text()) if (output/'training_timing.json').is_file() else {}
    learning_frozen_files = {p.name: sha256_file(p) for p in (output/'checkpoint').iterdir()
                             if p.name in {'model.pt','checkpoint_manifest.json'}}
    predictor = load_predictor(mod, output / 'checkpoint/checkpoint_manifest.json', device,
                               config.get('inference'))
    if config.get('plan_sha256'):
        import torch
        from scripts.feedback_infer import state_digest, modules_of
        for net in modules_of(predictor):
            net.eval(); net.requires_grad_(False)
            if any(p.device.type != torch.device(device).type for p in net.parameters()):
                raise ValueError('planned inference device differs from model parameter device')
        initial_frozen_state = state_digest(predictor)
    valid_paths = [p for p, entry in zip(paths, manifest["images"]) if entry["split"] == "validation/good"]
    # A disjoint good subset calibrates scores; remaining good images assess FPR.
    calibrate_paths, score_paths = split_validation_paths(valid_paths)
    calibration = np.stack([load_image(p, size) for p in calibrate_paths])
    normal = np.stack([load_image(p, size) for p in score_paths])
    synthetic_enabled = config.get('synthetic_evaluation', True)
    if synthetic_enabled:
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
        metrics, evaluation = evaluate_dev_v2(all_pred_normal, all_pred_synth, np.stack(masks),
                                               calibration_maps=all_pred_cal)
        if not np.isfinite(metrics['dev_synthetic/pixel_ap']):
            raise ValueError('primary generator yielded no positive synthetic pixels')
        metrics['training/duration_seconds'] = float(training_duration_seconds)
        metrics["dev_synthetic/preview_anomaly_score"] = float(np.max(all_pred_synth[0]))
        alternate, alternate_masks = zip(*(alternate_corruption(im, i + 7000) for i, im in enumerate(normal)))
        alternate_maps = []
        for im in alternate:
            response = predictor.predict(im[None])
            maps, _ = validate_prediction(response, (1, 3, size, size))
            alternate_maps.append(maps[0, 0])
        alternate_auroc = pixel_auroc(np.stack(alternate_maps), np.stack(alternate_masks))
        metrics['dev_synthetic/alternate_pixel_auroc'] = float(alternate_auroc)
        alternate_ap = pixel_ap_report(np.stack(alternate_maps), np.stack(alternate_masks))
        if not np.isfinite(alternate_ap['pooled']):
            raise ValueError('alternate generator yielded no positive synthetic pixels')
        metrics['dev_synthetic/alternate_pixel_ap'] = alternate_ap['pooled']
        metrics['dev_synthetic/alternate_pixel_ap_macro_positive_images'] = alternate_ap['macro_positive_images']
        metrics['dev_synthetic/generator_gap'] = float(metrics['dev_synthetic/pixel_auroc'] - alternate_auroc)
        paired = [paired_patch_edits(normal[i], normal[(i ^ 1) if (i ^ 1) < len(normal) else i - 1], seed=i + 9000)
                  for i in range(len(normal))]
        relocated, matched_controls, photometric_controls, matched_masks = zip(*paired)
        paired_maps, control_scores, photometric_scores = [], [], []
        for altered, control, photometric in zip(relocated, matched_controls, photometric_controls):
            prediction = predictor.predict(altered[None])
            maps, _ = validate_prediction(prediction, (1, 3, size, size))
            paired_maps.append(maps[0, 0])
            control_prediction = predictor.predict(control[None])
            _, scores = validate_prediction(control_prediction, (1, 3, size, size))
            control_scores.append(float(scores[0]))
            photometric_prediction = predictor.predict(photometric[None])
            _, scores = validate_prediction(photometric_prediction, (1, 3, size, size))
            photometric_scores.append(float(scores[0]))
        paired_maps = np.stack(paired_maps)
        matched_masks = np.stack(matched_masks)
        metrics['dev_synthetic/relocated_pixel_auroc'] = pixel_auroc(paired_maps, matched_masks)
        relocated_ap = pixel_ap_report(paired_maps, matched_masks)
        if not np.isfinite(relocated_ap['pooled']):
            raise ValueError('copy-insertion generator yielded no positive synthetic pixels')
        metrics['dev_synthetic/relocated_pixel_ap'] = relocated_ap['pooled']
        metrics['dev_synthetic/relocated_pixel_ap_macro_positive_images'] = relocated_ap['macro_positive_images']
        normal_threshold = metrics["normal/threshold"]
        metrics['robustness/artifact_matched_good_fpr'] = float(np.mean(np.asarray(control_scores) > normal_threshold))
        metrics['robustness/photometric_good_fpr'] = float(np.mean(np.asarray(photometric_scores) > normal_threshold))
        metrics['robustness/artifact_matched_good_false_positives'] = int(np.count_nonzero(np.asarray(control_scores) > normal_threshold))
        metrics['robustness/photometric_good_false_positives'] = int(np.count_nonzero(np.asarray(photometric_scores) > normal_threshold))
        case_metrics = []
        for index in range(len(normal)):
            case_metrics.append({
                'image_index': index, 'donor_pair_cluster': index // 2,
                'primary_pixel_ap': pixel_ap_report(
                    all_pred_synth[index:index + 1], np.asarray(masks)[index:index + 1])['pooled'],
                'alternate_pixel_ap': pixel_ap_report(
                    np.asarray(alternate_maps)[index:index + 1], np.asarray(alternate_masks)[index:index + 1])['pooled'],
                'relocated_pixel_ap': pixel_ap_report(
                    paired_maps[index:index + 1], matched_masks[index:index + 1])['pooled'],
                'primary_positive_pixels': int(np.asarray(masks)[index].sum()),
                'alternate_positive_pixels': int(np.asarray(alternate_masks)[index].sum()),
                'copy_insertion_positive_pixels': int(matched_masks[index].sum()),
                'normal_alarm': bool(np.max(all_pred_normal[index]) > normal_threshold),
                'matched_control_alarm': bool(control_scores[index] > normal_threshold),
                'photometric_control_alarm': bool(photometric_scores[index] > normal_threshold),
            })
        (output / 'development_case_metrics.json').write_text(json.dumps({
            'schema_version': '2.0', 'metric_evaluator_version': EVALUATOR_VERSION,
            'metric_status': 'SYNTHETIC_DEVELOPMENT_PROXY_ONLY',
            'cases': case_metrics}, sort_keys=True, indent=2))
        # Appearance edits that maintain the fixture's object layout test sensitivity
        # to superficial edits; these are not a substitute for real logical controls.
        edited_good = np.clip(normal * 0.93 + 0.03, 0, 1)
        edited_scores = []
        for im in edited_good:
            result = predictor.predict(im[None])
            _, scores = validate_prediction(result, (1, 3, size, size))
            edited_scores.append(float(scores[0]))
        metrics["robustness/edited_good_fpr"] = float(np.mean(np.asarray(edited_scores) > normal_threshold))
        metrics["robustness/edited_good_false_positives"] = int(np.count_nonzero(np.asarray(edited_scores) > normal_threshold))
        metrics.update(benchmark_predict(predictor, normal[:1], device=device, warmup=2, repeats=8))
        try:
            region_proxy = synthetic_region_pro_proxy(all_pred_synth, np.stack(masks), steps=20)
            if np.isfinite(region_proxy):
                metrics["dev_synthetic/region_pro_proxy_NOT_OFFICIAL"] = region_proxy
        except RuntimeError:
            pass
    else:
        import torch
        from benchmark.metrics_v2 import calibrate_image_threshold, image_false_positive_report
        from scripts.feedback_infer import state_digest
        frozen_state = state_digest(predictor)
        with torch.inference_mode():
            def scores_for(images):
                return [float(validate_prediction(predictor.predict(im[None]), (1,3,size,size))[0].max()) for im in images]
            calibrated = calibrate_image_threshold(scores_for(calibration))
            assessed = image_false_positive_report(scores_for(normal), calibrated['threshold'])
            metrics = {'normal/threshold':calibrated['threshold'],
                       'normal/image_fpr_at_threshold':assessed['image_fpr'],
                       'normal/image_false_positives':assessed['false_positives'],
                       'normal/image_assessed':assessed['images_assessed'],
                       'normal/calibration_images':calibrated['calibration_images'],
                       'normal/calibration_false_positives':calibrated['calibration_false_positives'],
                       'training/duration_seconds':training_duration_seconds}
            with Image.open(score_paths[0]) as source:
                geometry = source.size
            metrics.update(benchmark_predict(predictor, normal[:1], device=device, warmup=2, repeats=8,
                preprocess=lambda _:load_image(score_paths[0],size)[None],
                postprocess=lambda result:restore_anomaly_map(validate_prediction(result,(1,3,size,size))[0][0,0],geometry)))
        if frozen_state != state_digest(predictor):
            raise ValueError('predictor state changed during normal calibration/evaluation')
        evaluation = {'threshold_calibration':calibrated,'normal_assessment':assessed,
                      'inference_protocol':'batch one; decode/resize/prediction/full-resolution restore; 2 warmups, 8 repeats',
                      'device':device,'synthetic_evaluation':False}
    if config.get('plan_sha256'):
        import torch
        from scripts.feedback_infer import state_digest
        frozen_state = state_digest(predictor)
        if frozen_state != initial_frozen_state:
            raise ValueError('normal assessment changed frozen predictor state')
        diagnostic_module = load_candidate('/diagnostics/model.py') if config.get('diagnostic_provider') else mod
        catalog=diagnostic_measurement_catalog(config.get('diagnostics', []))
        if config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'} and config.get('execution_kind')=='train':
            receipt={'version':'completed-learning-v1','phase':'diagnostics_ready',
                'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                'model_module_sha256':sha256_file(module_file),'good_manifest_sha256':manifest_fingerprint(manifest),
                'checkpoint_sha256':learning_frozen_files['model.pt'],
                'checkpoint_manifest_sha256':learning_frozen_files['checkpoint_manifest.json'],
                'training_audit':training_audit,'training_timing':timing_audit,
                'normal_threshold':metrics['normal/threshold']}
            (output/'training_receipt.json').write_text(json.dumps(receipt,sort_keys=True))
        for request in config.get('diagnostics', []):
            if not callable(getattr(diagnostic_module, 'diagnose', None)):
                raise ValueError(f"missing diagnose hook for {request['diagnostic_id']}")
            assessment_ids = [sha256_file(path) for path in score_paths]
            with torch.inference_mode():
                diagnostic = diagnostic_module.diagnose(predictor, normal.copy(), dict(request['config'],
                    diagnostic_id=request['diagnostic_id'], frozen_normal_threshold=metrics['normal/threshold'],
                    assessment_source_ids=assessment_ids))
            paired=config.get('paired_diagnostic_metrics',{}).get(request['diagnostic_id'],[])
            if config.get('plan_contract_version') in {'experiment-plan-v3.0','experiment-plan-v4.0'}:
                validate_diagnostic_output(request,diagnostic,assessment_ids,paired)
            values = diagnostic.get('metrics', {}) if isinstance(diagnostic, dict) else {}
            if not set(request['required_metrics']).issubset(values):raise ValueError('diagnostic omitted required measurements')
            for key,value in values.items():
                canonical=f"{request['diagnostic_id']}/{key}"
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('diagnostic metrics must be finite measured numbers')
                if canonical not in catalog:raise ValueError('diagnostic returned undeclared metric: '+canonical)
                metrics[catalog[canonical]]=float(value)
            if 'per_image' in diagnostic:
                ids=[r.get('source_id') for r in diagnostic['per_image']]
                if len(ids)!=len(assessment_ids) or set(ids)!=set(assessment_ids):raise ValueError('diagnostic per-image identities differ from normal assessment sources')
            if frozen_state != state_digest(predictor):
                raise ValueError('diagnostic changed frozen predictor state')
            (output/f"diagnostic_{request['diagnostic_id']}.json").write_text(json.dumps(diagnostic, allow_nan=False))
        if any(sha256_file(output/'checkpoint'/name) != sha for name,sha in learning_frozen_files.items()):
            raise ValueError('diagnostic changed frozen checkpoint')
        metrics.update({'protocol/optimizer_updates': training_audit['optimizer_updates'],
                        'protocol/source_exposures': training_audit['source_exposures'], 'protocol/seed': config['seed']})
        (output/'training_audit.json').write_text(json.dumps(training_audit, sort_keys=True))
    if timing_audit.get('protocol')=='legacy-candidate-train-call-v1':
        metrics.pop('training/duration_seconds',None)
        metrics['training/function_wall_seconds']=timing_audit['function_wall_seconds']
    metrics['training/timing_verified']=float(timing_audit.get('verified', config.get('execution_kind')=='frozen'))
    if timing_audit.get('protocol')=='prepared-training-step-v1':
        metrics['training/phase_steps']=timing_audit['completed_steps']
        if config.get('training_completion_policy') in {'required-training-target-v1','elapsed-training-loop-v1'}:
            metrics['training/target_seconds']=config['training_budget_seconds']
            metrics['training/target_reached']=float(timing_audit['duration_seconds']>=config['training_budget_seconds'])
            metrics['training/overshoot_seconds']=timing_audit['overshoot_seconds']
        if config.get('training_completion_policy')=='elapsed-training-loop-v1':
            metrics['training/loop_elapsed_seconds']=timing_audit['loop_elapsed_seconds']
            metrics['training/learning_seconds']=timing_audit['learning_seconds']
            metrics['training/loop_overhead_seconds']=max(0,timing_audit['loop_elapsed_seconds']-timing_audit['learning_seconds'])
            metrics['training/initialization_seconds']=timing_audit['execution_profile']['initial_preparation_seconds']
            metrics['training/checkpoint_seconds']=timing_audit['execution_profile']['finalization_seconds']
    checkpoint = output / "checkpoint" / "model.pt"
    if config.get('plan_contract_version')=='experiment-plan-v4.0' and config.get('execution_kind')=='frozen':
        metrics.update({'training/loop_elapsed_seconds':0.,'training/learning_seconds':0.,'training/loop_overhead_seconds':0.})
    metrics["inference/checkpoint_mib"] = checkpoint.stat().st_size / 2**20
    try:
        import torch
        model = getattr(predictor, "net", None)
        if model is not None:
            metrics["inference/parameters"] = sum(p.numel() for p in model.parameters())
            metrics["inference/trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError:
        pass
    with Image.open(score_paths[0]) as source:
        original = source.convert('RGB')
        original_size = original.size
        if synthetic_enabled:
            save_heatmaps(output, normal[0], synthetic[0], all_pred_synth[0], masks[0], original_image=original)
    import sys
    generator_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {"training_input_policy": "train/good only; real development images and masks never mounted",
                  "training_images": len(training_manifest['images']),
                  "training_manifest_fingerprint": manifest_fingerprint(training_manifest),
                  "training_timing_definition":timing_audit.get("timing_definition","historical-unverified"), "training_timing_protocol":timing_audit.get("protocol", "frozen-no-training"), "protocol_version": "2.0", "metric_evaluator_version": EVALUATOR_VERSION,
                  "evaluation": evaluation, "python_version": sys.version.split()[0],
                  "development_calibration_count": len(calibrate_paths), "normal_assessment_count": len(score_paths),
                  "dependencies_pyproject_sha256": sha256_file(Path(__file__).resolve().parents[1] / "pyproject.toml"),
                  "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "metric_status": "SYNTHETIC_DEVELOPMENT_PROXY_ONLY", "model_module_sha256": sha256_file(module_file),
                  "good_manifest_sha256": manifest_fingerprint(manifest), "checkpoint_sha256": sha256_file(checkpoint),
                  "image_count_normal_validation": len(valid_paths), "device": device, "size": size,
                  "preview_original_width": original_size[0], "preview_original_height": original_size[1],
                  "artifact_hashes": {filename: sha256_file(output / filename) for filename in
                      ("normal_preview.png", "synthetic_input.png", "synthetic_mask.png", "synthetic_heatmap.png", "synthetic_overlay.png", "development_case_metrics.json") if (output / filename).is_file()},
                  "artifact_control_status": "same-location donor edit is not certified logically valid; mild photometric edit preserves layout",
                  "synthetic_generator_protocol": "white_patch_v1, target_only_copy_insertion_v1, feathered_target_only_copy_insertion_v1",
                  "synthetic_mask_semantics": "edited target footprint only; source remains present; no missing-object localization label",
                  "synthetic_generator_source_sha256": generator_fingerprint,
                  "calibration_path_sha256": hashlib.sha256('\n'.join(str(p) for p in calibrate_paths).encode()).hexdigest(),
                  "assessment_path_sha256": hashlib.sha256('\n'.join(str(p) for p in score_paths).encode()).hexdigest()}
    if not synthetic_enabled:
        provenance['metric_status'] = 'NORMAL_ONLY_CALIBRATION_AND_EFFICIENCY'
        for key in list(provenance):
            if key.startswith('synthetic_') or key == 'artifact_control_status': provenance.pop(key)
    if config.get('plan_sha256'):
        import os
        provenance.update(effective_configuration=config, training_audit=training_audit,
                          plan_sha256=config['plan_sha256'], plan_id=config['plan_id'],
                          plan_arm=config['plan_arm'], execution_kind=config['execution_kind'],
                          trusted_source_sha256=config['trusted_source_sha256'],
                          frozen_baseline=config.get('frozen_baseline'),
                          execution_environment={'torch':torch.__version__,'numpy':np.__version__,
                            'cuda_build':torch.version.cuda,'device':device,
                            'gpu_name':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()})
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2))
    return {"metrics": {k: float(v) for k,v in metrics.items()}, "provenance": provenance}


def main():
    parser = argparse.ArgumentParser()
    for key in ("module", "manifest", "config", "output", "result"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    output = run(Path(args.module), Path(args.manifest), json.loads(Path(args.config).read_text()), Path(args.output))
    Path(args.result).write_text(json.dumps(output, indent=2))
    print(json.dumps({"metric_kind": output["provenance"]["metric_status"], "metrics": output["metrics"]}))

if __name__ == "__main__":
    main()
