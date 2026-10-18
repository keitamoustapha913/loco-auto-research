"""Small illustrative normal-only autoencoder, NOT a validated LOCO method."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def _torch():
    import torch
    from torch import nn
    return torch, nn


def create_model():
    torch, nn = _torch()
    return nn.Sequential(
        nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(),
        nn.Conv2d(12, 12, 3, padding=1), nn.ReLU(),
        nn.Conv2d(12, 3, 1), nn.Sigmoid()
    )


class ModelWrapper:
    def __init__(self, net, device):
        self.net = net.eval()
        self.device = device

    def predict(self, images):
        torch, _ = _torch()
        x = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            reconstruction = self.net(x)
            maps = (x - reconstruction).square().mean(dim=1, keepdim=True)
            scores = maps.flatten(1).amax(dim=1)
        return {"anomaly_maps": maps, "image_scores": scores, "metadata": {"method": "toy_normal_ae"}}


def train(config, allowed_train_manifest, output_dir):
    """Trains only over already validated normal manifest paths."""
    from PIL import Image
    from benchmark.data_guard import load_manifest
    torch, nn = _torch()
    manifest, paths = load_manifest(allowed_train_manifest)
    paths = [p for p, entry in zip(paths, manifest["images"]) if entry["split"] == "train/good"]
    if not paths:
        raise ValueError("no normal training images")
    device = config.get("device", "cpu")
    torch.manual_seed(int(config.get("seed", 7)))
    net = create_model().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=float(config.get("lr", 0.002)))
    size = int(config.get("size", 64))
    losses = []
    for epoch in range(int(config.get("epochs", 1))):
        for path in paths:
            im = Image.open(path).convert("RGB").resize((size, size))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy() / 255).unsqueeze(0).to(device)
            prediction = net(x)
            loss = nn.functional.mse_loss(prediction, x)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(net.cpu().state_dict(), checkpoint)
    meta = {"schema_version": "1.0", "method": "toy_normal_ae", "checkpoint": "model.pt", "size": size, "steps": len(losses), "final_loss": losses[-1]}
    (output / "checkpoint_manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(checkpoint_manifest, device="cpu"):
    torch, _ = _torch()
    manifest_path = Path(checkpoint_manifest)
    meta = json.loads(manifest_path.read_text())
    checkpoint = (manifest_path.parent / meta["checkpoint"]).resolve(strict=True)
    if not checkpoint.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("checkpoint escapes model output")
    net = create_model()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return ModelWrapper(net.to(device), device)
