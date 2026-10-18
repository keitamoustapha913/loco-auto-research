"""Scratch-trained, good-only masked conditional prediction; no image bank."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


def masks(size, phase, device):
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
    hidden = (((y // 16) % 2) * 2 + (x // 16) % 2 == phase).float()[None, None]
    visible = 1 - F.max_pool2d(hidden, 7, stride=1, padding=3)
    return hidden, visible


def targets(x):
    smooth = F.avg_pool2d(x, 5, stride=1, padding=2)
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 0, 1))
    return torch.cat((x, smooth, dx, dy), 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(9, 48, 1), nn.SiLU(), nn.Conv2d(48, 48, 1), nn.SiLU(), nn.Conv2d(48, 12, 1))

    def forward(self, x, visible, arm):
        # All image aggregation happens AFTER exclusion; no target leakage.
        local = F.avg_pool2d(x * visible, 31, 1, 15) / F.avg_pool2d(visible, 31, 1, 15).clamp_min(1e-6)
        if arm in ('global', 'combined'):
            context = F.adaptive_avg_pool2d(x * visible, (4, 4)) / F.adaptive_avg_pool2d(visible, (4, 4)).clamp_min(1e-6)
            context = F.interpolate(context, x.shape[-2:], mode='bilinear', align_corners=False)
            if arm == 'global':
                local = context
        elif arm == 'local':
            context = local
        elif arm in ('neither', 'spatial'):
            local = torch.zeros_like(local)
            context = torch.zeros_like(local)
        else:
            raise ValueError('unknown context arm')
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        coords = torch.stack((yy, xx))[None].expand(x.shape[0], -1, -1, -1)
        visibility = visible.expand(x.shape[0], -1, -1, -1)
        if arm == 'spatial':
            visibility = torch.zeros_like(visibility)
        return self.layers(torch.cat((local, context, coords, visibility), 1))


def _device(value):
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    return device


class Predictor:
    def __init__(self, net, size, arm, scale, device):
        self.net, self.size, self.arm, self.scale, self.device = net.eval(), size, arm, scale, device

    @torch.no_grad()
    def predict(self, images):
        x = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        if x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != 3 or not torch.isfinite(x).all():
            raise ValueError('expected finite nonempty RGB [B,3,H,W]')
        original = x.shape[-2:]
        x = F.interpolate(x, (self.size, self.size), mode='bilinear', align_corners=False)
        target = targets(x)
        score = torch.zeros_like(x[:, :1])
        for phase in range(4):
            hidden, visible = masks(self.size, phase, self.device)
            residual = ((self.net(x, visible, self.arm) - target).square() / self.scale).mean(1, keepdim=True)
            score += hidden * residual
        score = F.interpolate(score, original, mode='bilinear', align_corners=False)
        return {'anomaly_maps': score.cpu().numpy(), 'image_scores': score.flatten(1).amax(1).cpu().numpy()}


def train(config: dict, allowed_train_manifest: Path, output_dir: Path):
    device = _device(config.get('device', 'cpu'))
    seed = int(config.get('seed', 0))
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    size = int(config.get('size', 128))
    if size < 32 or size > 256:
        raise ValueError('pilot size must be 32..256')
    steps = int(config.get('steps', config.get('updates', 100)))
    batch = int(config.get('batch_size', 4))
    if not 1 <= steps <= 100 or not 1 <= batch <= 16:
        raise ValueError('pilot permits 1..100 updates and batch 1..16')
    arm = config.get('context_arm', 'combined')
    if arm not in ('global', 'local', 'combined', 'neither', 'spatial'):
        raise ValueError('context_arm must be local, global, combined, neither or spatial')
    manifest_path = Path(allowed_train_manifest)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    entries = [e for e in manifest['images'] if e['split'] == 'train/good' and (config.get('category', 'fixture') == 'fixture' or e['category'] == config['category'])]
    if not entries:
        raise ValueError('no authorized train/good entries')
    # Only explicit fitting entries are opened. Validation is reserved to runner.
    def read(entry):
        path = Path(entry['path'])
        if not path.is_absolute():
            path = Path(manifest.get('dataset_root', manifest.get('root', manifest_path.parent))) / path
        with Image.open(path) as im:
            return torch.from_numpy(np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).copy()).permute(2, 0, 1) / 255
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config.get('lr', 0.001)))
    rng = np.random.default_rng(seed)
    sums = torch.zeros(1, 12, 1, 1, device=device)
    count = 0
    losses = []
    start = time.monotonic()
    initial = next(net.parameters()).detach().clone()
    for step in range(steps):
        if time.monotonic() - start > min(float(config.get('train_seconds', 20)), 20):
            break
        x = torch.stack([read(entries[int(i)]) for i in rng.integers(len(entries), size=batch)]).to(device)
        hidden, visible = masks(size, step % 4, device)
        target = targets(x)
        error = (net(x, visible, arm) - target).square()
        loss = (error * hidden).sum() / (hidden.sum() * batch * 12)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sums += (error.detach() * hidden).sum((0, 2, 3), keepdim=True)
        count += int(hidden.sum().item()) * batch
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError('budget expired before any update')
    changed = bool((initial != next(net.parameters()).detach()).any().item())
    if not changed:
        raise RuntimeError('no parameter update')
    scale = (sums / count).clamp_min(1e-4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state': net.state_dict(), 'scale': scale.cpu(), 'size': size, 'arm': arm}, output_dir / 'model.pt')
    metadata = {'schema_version': '1.0', 'model_file': 'model.pt', 'device': str(device), 'training_tensor_device': str(x.device), 'parameter_updated': changed, 'updates': len(losses), 'image_exposures': len(losses) * batch, 'losses': losses, 'config': config, 'seed': seed, 'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checkpoint_sha256': hashlib.sha256((output_dir / 'model.pt').read_bytes()).hexdigest(), 'torch_version': torch.__version__, 'numpy_version': np.__version__, 'metric_status': 'UNVALIDATED_DEVELOPMENT_PROXY_ONLY', 'external_weights': []}
    (output_dir / 'checkpoint_manifest.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def load(checkpoint_manifest, device='cpu'):
    device = _device(device)
    path = Path(checkpoint_manifest)
    meta = json.loads(path.read_text())
    artifact = path.parent / 'model.pt'
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    state = torch.load(artifact, map_location=device, weights_only=True)
    net = Net().to(device)
    net.load_state_dict(state['state'])
    return Predictor(net, state['size'], state['arm'], state['scale'].to(device), device)
