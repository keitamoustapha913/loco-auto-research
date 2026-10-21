"""Disposable CUDA contract check, launched only inside an already queued job.

The host runner owns the GPU lock, timeout, cancellation and time accounting.
This worker never calibrates or evaluates on development anomalies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import torch

from benchmark.data_guard import load_manifest, sha256_file
from benchmark.train_adapter import load_candidate, validate_prediction
from scripts.feedback_infer import state_digest
from scripts.run_candidate import load_image, split_validation_paths


def run(module_file, manifest_file, config, output):
    if not str(config.get('device', '')).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('CUDA smoke requires an available authorized CUDA device')
    manifest, paths = load_manifest(manifest_file)
    selected = [(e, p) for e, p in zip(manifest['images'], paths)
                if e['split'] == 'train/good' and e['category'] == config['category']][:2]
    if not selected:
        raise ValueError('CUDA smoke requires authorized normal training images')
    output.mkdir(parents=True, exist_ok=False)
    smoke_config = dict(config, steps=2, epochs=1, batch_size=1, synthetic_evaluation=False,
                        validation_smoke=True, expected_optimizer_updates=2, expected_source_exposures=2)
    smoke_manifest = output / 'training_good_manifest.json'
    smoke_manifest.write_text(json.dumps(dict(manifest, images=[e for e, _ in selected])))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    module = load_candidate(module_file)
    checkpoint_dir = output / 'checkpoint'
    if config.get('training_budget_seconds') is not None:
        from scripts.training_phase import run_training_phases
        # Separate disposable model: never reuse its weights or learning clock.
        # New plans enforce the same per-model training-only ceiling here too.
        smoke_budget = (config['training_budget_seconds']
                        if config.get('plan_contract_version')=='experiment-plan-v2.0' else None)
        smoke_config.pop('training_budget_seconds',None)
        run_training_phases(module,smoke_config,smoke_manifest,checkpoint_dir,budget=smoke_budget,
                            preflight_steps=2 if config.get('plan_contract_version')=='experiment-plan-v3.0' else None)
    else:
        module.train(smoke_config, smoke_manifest, checkpoint_dir)
    training_cuda_bytes = torch.cuda.max_memory_allocated()
    if training_cuda_bytes <= 0:
        raise ValueError('CUDA smoke observed no CUDA tensor allocation during training')
    checkpoint = checkpoint_dir / 'model.pt'
    metadata = checkpoint_dir / 'checkpoint_manifest.json'
    before = (sha256_file(checkpoint), sha256_file(metadata))
    predictor = module.load(metadata, device=config['device'])
    models = [predictor] if isinstance(predictor, torch.nn.Module) else [
        value for value in vars(predictor).values() if isinstance(value, torch.nn.Module)]
    parameters = [p for model in models for p in model.parameters()]
    if parameters and not all(p.device.type == 'cuda' for p in parameters):
        raise ValueError('CUDA smoke predictor parameters are not on CUDA')
    for value in vars(predictor).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
    state_before = state_digest(predictor)
    size = int(config.get('size', 128))
    with torch.inference_mode():
        for _, path in selected:
            batch = load_image(path, size)[None]
            validate_prediction(predictor.predict(batch), batch.shape)
    if config.get('diagnostic_preflight_required'):
        import numpy as np
        from benchmark.train_adapter import validate_diagnostic_output
        valid=[p for e,p in zip(manifest['images'],paths) if e['split']=='validation/good' and e['category']==config['category']]
        calibrate,assess=split_validation_paths(valid)
        if not calibrate or not assess:raise ValueError('diagnostic preflight requires disjoint authorized normal calibration/assessment')
        normal=np.stack([load_image(p,size) for p in assess]);ids=[sha256_file(p) for p in assess]
        with torch.inference_mode():
            threshold=max(float(np.max(predictor.predict(load_image(p,size)[None])['anomaly_maps'])) for p in calibrate)
            for declaration in config.get('diagnostics',[]):
                result=module.diagnose(predictor,normal.copy(),dict(declaration['config'],
                    diagnostic_id=declaration['diagnostic_id'],frozen_normal_threshold=threshold,assessment_source_ids=ids,validation_smoke=True))
                validate_diagnostic_output(declaration,result,ids,config.get('paired_diagnostic_metrics',{}).get(declaration['diagnostic_id'],[]))
        if before!=(sha256_file(checkpoint),sha256_file(metadata)) or state_before!=state_digest(predictor):
            raise ValueError('diagnostic preflight mutated frozen model state')
    torch.cuda.synchronize()
    if before != (sha256_file(checkpoint), sha256_file(metadata)) or state_before != state_digest(predictor):
        raise ValueError('CUDA smoke inference mutated checkpoint or predictor state')
    summary = {'status': 'PASSED', 'kind': 'infrastructure_validation_only',
               'training_images': len(selected), 'optimizer_step_request': 2,
               'device': str(config['device']), 'wall_seconds': time.monotonic()-started,
               'training_peak_cuda_allocated_bytes': training_cuda_bytes,
               'checkpoint_unchanged': True, 'pilot_reuses_smoke_weights': False}
    (output / 'summary.json').write_text(json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('module', 'manifest', 'config', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.module, args.manifest, json.loads(args.config.read_text()), args.output)))


if __name__ == '__main__':
    main()
