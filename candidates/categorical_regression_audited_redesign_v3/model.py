"""Good-only masked completion with paired categorical/regression objectives."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ORIGIN_SOURCE_SHA256 = "9456cfd663ac7f930bdc8b4a00b625533139f1822b9b469f84288f8027bb7a03"
ORIGIN_COMMIT = "0005982da2ee124dd2a3b8e350f0840e0c88efe7"
CANDIDATE_ID = "categorical_regression_audited_redesign_v3"


class Completion(nn.Module):
    def __init__(self, width=32, bins=4):
        super().__init__()
        self.bins = bins
        axis = torch.linspace(0, 1, bins)
        self.register_buffer('vocabulary', torch.cartesian_prod(axis, axis, axis))
        self.context = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.categorical = nn.Conv2d(width, bins ** 3, 1)
        self.deterministic = nn.Conv2d(width, 3, 1)

    def labels(self, x):
        q = (x.clamp(0, 1) * (self.bins - 1)).round().long()
        return q[:, 0] * self.bins ** 2 + q[:, 1] * self.bins + q[:, 2]

    def forward(self, x, mask):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        coordinates = torch.stack((yy, xx))[None].expand(b, -1, -1, -1)
        z = self.context(torch.cat((x * (1 - mask), mask, coordinates), 1))
        return self.categorical(z), self.deterministic(z)


def _device(value):
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    return device


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(config: dict, allowed_train_manifest: Path, output_dir: Path):
    device = _device(config.get('device', 'cpu'))
    seed = int(config.get('seed', 7))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    manifest_path = Path(allowed_train_manifest)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['images']
    if not entries or any(e['split'] != 'train/good' for e in entries):
        raise ValueError('only a nonempty train/good manifest is permitted')
    from benchmark.data_guard import load_manifest, manifest_fingerprint
    manifest, authorized_paths = load_manifest(manifest_path)
    if any(e['category'] != 'juice_bottle' for e in entries):
        raise ValueError('only authorized juice_bottle normal training inputs are permitted')
    grid = int(config.get('completion_grid', 24))
    width = int(config.get('completion_width', 32))
    bins = int(config.get('vocabulary_bins', 4))
    steps = int(config.get('steps', config.get('train_steps', 120)))
    batch = int(config.get('batch_size', 8))
    arm = config.get('objective', 'categorical')
    if not (8 <= grid <= 64 and 4 <= width <= 128 and 2 <= bins <= 8
            and 1 <= steps <= 1000 and 1 <= batch <= 32 and arm in ('categorical', 'deterministic')):
        raise ValueError('pilot configuration outside bounded limits')
    expected = {'size': 128, 'completion_grid': 24, 'completion_width': 32, 'vocabulary_bins': 4}
    if any(config.get(k, v) != v for k, v in expected.items()):
        raise ValueError('configuration differs from the frozen scientific plan')
    counts = (steps, batch, config.get('expected_optimizer_updates'), config.get('expected_source_exposures'))
    required = (2, 1, 2, 2) if config.get('validation_smoke') else (120, 8, 120, 960)
    if counts != required:
        raise ValueError('explicit schedule/audit counts differ from frozen plan or disposable smoke contract')
    arrays = []
    for path in authorized_paths:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert('RGB').resize((grid, grid), Image.Resampling.BILINEAR),
                                     dtype=np.float32).transpose(2, 0, 1).copy() / 255)
    # Training images exist only during fitting and are never serialized.
    data = torch.from_numpy(np.stack(arrays)).to(device)
    net = Completion(width, bins).to(device)
    # Separate trunks start identically: gradients of one objective cannot affect the other.
    baseline = Completion(width, bins).to(device)
    baseline.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(list(net.parameters()) + list(baseline.parameters()), lr=0.002)
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    losses = []
    source_order = hashlib.sha256()
    mask_order = hashlib.sha256()
    optimizer_updates = 0
    source_exposures = 0
    net.train()
    baseline.train()
    for step in range(steps):
        indices = torch.randint(len(data), (batch,), generator=rng, device=device)
        x = data[indices]
        sampled = indices.detach().cpu().tolist()
        source_exposures += len(sampled)
        for index in sampled:
            source_order.update(entries[index]['sha256'].encode('ascii') + b'\n')
        # Four deterministic interleaved masks, identical in both arms and inference.
        yy, xx = torch.meshgrid(torch.arange(grid, device=device), torch.arange(grid, device=device), indexing='ij')
        phase = step % 4
        mask = (((yy % 2) * 2 + xx % 2) == phase).float()[None, None].expand(batch, -1, -1, -1)
        mask_order.update(mask.detach().cpu().to(torch.uint8).numpy().tobytes())
        labels = net.labels(x)
        targets = F.embedding(labels, net.vocabulary).permute(0, 3, 1, 2)
        logits, _ = net(x, mask)
        _, regression = baseline(x, mask)
        ce = (F.cross_entropy(logits, labels, reduction='none') * mask[:, 0]).sum() / mask.sum()
        mse = ((regression - targets).square() * mask).sum() / (3 * mask.sum())
        loss = ce + mse
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_updates += 1
        losses.append([float(ce.detach()), float(mse.detach())])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'categorical': net.state_dict(), 'deterministic': baseline.state_dict()}, output_dir / 'model.pt')
    meta = {'schema_version': '1.0', 'candidate_id': CANDIDATE_ID,
            'grid': grid, 'width': width, 'bins': bins, 'objective': arm, 'seed': seed,
            'updates_per_arm': steps, 'image_exposures_per_arm': steps * batch,
            'training_audit': {'seed': seed, 'optimizer_updates': optimizer_updates,
                'source_exposures': source_exposures,
                'sampled_source_order_sha256': source_order.hexdigest(),
                'mask_order_sha256': mask_order.hexdigest(),
                'ordered_training_sources_sha256': hashlib.sha256(json.dumps(
                    [(e['path'], e['sha256']) for e in entries], separators=(',', ':')).encode()).hexdigest()},
            'training_manifest_fingerprint': manifest_fingerprint(manifest),
            'origin_source_sha256': ORIGIN_SOURCE_SHA256, 'origin_commit': ORIGIN_COMMIT,
            'training_images': len(entries), 'device': str(device), 'losses_ce_mse': losses,
            'config': config, 'manifest_sha256': _hash(manifest_path), 'source_sha256': _hash(__file__),
            'model_sha256': _hash(output_dir / 'model.pt'), 'torch_version': torch.__version__,
            'representation': 'fixed RGB bilinear patch means; no pretrained weights',
            'assessment_status': 'not executed; paired synthetic falsification remains incomplete'}
    (output_dir / 'checkpoint_manifest.json').write_text(json.dumps(meta, indent=2))
    return meta


class Predictor:
    def __init__(self, net, grid, objective, device):
        self.net = net.eval()
        self.grid, self.objective, self.device = grid, objective, device

    @torch.inference_mode()
    def predict(self, images):
        x = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 3 or not torch.isfinite(x).all():
            raise ValueError('expected nonempty finite RGB BCHW input')
        h, w = x.shape[-2:]
        x = F.interpolate(x, (self.grid, self.grid), mode='bilinear', align_corners=False).clamp(0, 1)
        labels = self.net.labels(x)
        target = F.embedding(labels, self.net.vocabulary).permute(0, 3, 1, 2)
        yy, xx = torch.meshgrid(torch.arange(self.grid, device=self.device),
                                torch.arange(self.grid, device=self.device), indexing='ij')
        score = torch.zeros_like(x[:, :1])
        for phase in range(4):
            mask = (((yy % 2) * 2 + xx % 2) == phase).float()[None, None].expand(len(x), -1, -1, -1)
            logits, regression = self.net(x, mask)
            residual = (F.cross_entropy(logits, labels, reduction='none')[:, None]
                        if self.objective == 'categorical' else (regression - target).square().mean(1, keepdim=True))
            score += residual * mask
        maps = F.interpolate(score, (h, w), mode='bilinear', align_corners=False)
        if not torch.isfinite(maps).all():
            raise RuntimeError('nonfinite anomaly maps')
        return {'anomaly_maps': maps.cpu().numpy(), 'image_scores': maps.flatten(1).amax(1).cpu().numpy()}


def load(checkpoint_manifest: Path, device='cpu'):
    path = Path(checkpoint_manifest)
    meta = json.loads(path.read_text())
    artifact = path.parent / 'model.pt'
    if _hash(artifact) != meta['model_sha256']:
        raise ValueError('checkpoint hash mismatch')
    device = _device(device)
    net = Completion(meta['width'], meta['bins']).to(device)
    states = torch.load(artifact, map_location=device, weights_only=True)
    net.load_state_dict(states[meta['objective']])
    return Predictor(net, meta['grid'], meta['objective'], device)


DIAGNOSTIC_SETTINGS = {
    'alternate_generator': 'runner_alternate_corruption_seed_7000_plus_source_index',
    'emit_per_image': True,
    'interior_erosion_pixels': 2,
    'photometric_offset': 0.03,
    'photometric_scale': 0.93,
    'primary_generator': 'runner_synthetic_corruption_seed_5000_plus_source_index',
}


def _interior_and_boundary(mask, radius):
    """Square binary erosion: exclude the two-pixel interior footprint edge.

    AP uses background negatives; boundary pixels are excluded from interior AP,
    and interior pixels from boundary AP. These remain editing-footprint proxies.
    """
    positive = np.asarray(mask, dtype=bool)
    interior = positive.copy()
    for _ in range(radius):
        padded = np.pad(interior, 1, constant_values=False)
        interior = np.logical_and.reduce([
            padded[y:y+positive.shape[0], x:x+positive.shape[1]]
            for y in range(3) for x in range(3)])
    boundary = positive & ~interior
    if not interior.any() or not boundary.any() or positive.all():
        raise ValueError('interior/boundary AP requires positive regions and background support')
    return interior, boundary


def _localization_parts(prediction, mask, radius):
    from benchmark.metrics_v2 import binary_average_precision
    interior, boundary = _interior_and_boundary(mask, radius)
    scores = np.asarray(prediction)
    if scores.shape != interior.shape or not np.isfinite(scores).all():
        raise ValueError('unaligned or nonfinite diagnostic heatmap')
    return (binary_average_precision(interior[~boundary], scores[~boundary]),
            binary_average_precision(boundary[~interior], scores[~interior]),
            {'footprint_pixels': int(np.count_nonzero(mask)),
             'interior_pixels': int(interior.sum()), 'boundary_pixels': int(boundary.sum())})


def diagnose(predictor, assessment_good_images, diagnostic_config):
    """Frozen within-source controls; no training, calibration or donor images."""
    from scripts.run_candidate import synthetic_corruption, alternate_corruption
    from benchmark.train_adapter import validate_prediction
    if not torch.is_inference_mode_enabled():
        raise ValueError('diagnostics require inference-only execution')
    if any(diagnostic_config.get(k) != v for k, v in DIAGNOSTIC_SETTINGS.items()):
        raise ValueError('diagnostic settings differ from the frozen scientific plan')
    if diagnostic_config.get('diagnostic_id') != 'within_source_proxy':
        raise ValueError('unsupported diagnostic identity')
    images = np.asarray(assessment_good_images)
    ids = diagnostic_config.get('assessment_source_ids', [])
    threshold = diagnostic_config.get('frozen_normal_threshold')
    if (images.ndim != 4 or not len(images) or images.shape[1] != 3
            or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1
            or len(ids) != len(images) or len(set(ids)) != len(ids)
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)):
        raise ValueError('finite normal assessment inputs, unique identities and frozen threshold required')
    records = []
    for index, (source, source_id) in enumerate(zip(images, ids)):
        primary, primary_mask = synthetic_corruption(source, seed=5000+index)
        alternate, alternate_mask = alternate_corruption(source, seed=7000+index)
        maps = []
        for edited in (primary, alternate):
            prediction = validate_prediction(predictor.predict(edited[None]), (1, *source.shape))[0][0, 0]
            maps.append(prediction)
        primary_ap, primary_boundary, primary_support = _localization_parts(maps[0], primary_mask, 2)
        alternate_ap, alternate_boundary, alternate_support = _localization_parts(maps[1], alternate_mask, 2)
        photometric = np.clip(source * diagnostic_config['photometric_scale']
                              + diagnostic_config['photometric_offset'], 0, 1)
        photo_map = validate_prediction(predictor.predict(photometric[None]), (1, *source.shape))[0]
        alarm = int(float(photo_map.max()) > threshold)
        records.append({'source_id': source_id,
            'metrics': {'primary_interior_ap': primary_ap, 'alternate_interior_ap': alternate_ap,
                        'primary_boundary_ap': primary_boundary, 'alternate_boundary_ap': alternate_boundary,
                        'photometric_alarm': alarm, 'photometric_false_positives': alarm, 'assessment_sources': 1},
            'generator_seeds': {'primary': 5000+index, 'alternate': 7000+index},
            'proxy_mask_support': {'primary': primary_support, 'alternate': alternate_support}})
    keys = ('primary_interior_ap', 'alternate_interior_ap', 'primary_boundary_ap',
            'alternate_boundary_ap', 'photometric_alarm')
    metrics = {key: float(np.mean([r['metrics'][key] for r in records])) for key in keys}
    metrics.update(photometric_false_positives=sum(r['metrics']['photometric_alarm'] for r in records),
                   assessment_sources=len(records))
    return {'metrics': metrics, 'per_image': records,
            'label': 'Within-source synthetic editing-footprint proxies; not certified logical targets',
            'aggregation': 'equal-weight assessment sources; alarm numerator/denominator recorded',
            'localization_definition': 'interior AP excludes footprint boundary; boundary AP excludes interior; both retain background negatives',
            'frozen_normal_threshold': float(threshold), 'generator_configuration': dict(DIAGNOSTIC_SETTINGS)}
