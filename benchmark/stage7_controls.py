"""Predeclared, good-image-only diagnostic edits; masks mark edit footprints.

These transformations are not certified logical anomalies or valid product states.
No semantic component masks are available. All randomness is local to the seed.
"""
from __future__ import annotations

import numpy as np

PROTOCOL_VERSION = "stage7-frozen-diagnostics-v1"


def whole_image_controls(image: np.ndarray, shift: int = 8, gain: float = 1.08):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 3 or min(x.shape[1:]) <= 2 * shift:
        raise ValueError("expected RGB image larger than the fixed shift")
    # A whole-frame translation preserves relative layout; reflected edges are
    # visible rendering artifacts and must be assessed separately.
    padded = np.pad(x, ((0, 0), (shift, shift), (shift, shift)), mode="reflect")
    translated = padded[:, :x.shape[1], 2 * shift:2 * shift + x.shape[2]]
    illuminated = np.clip(x * gain, 0, 1)
    return {"position_shift": translated.copy(), "illumination": illuminated.astype(np.float32)}


def texture_noise_control(image: np.ndarray, seed: int, sigma: float = .02):
    """Fixed independent fine-grain rendering perturbation; not semantic validity."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 3 or not 0 < sigma <= .05:
        raise ValueError('bounded RGB texture control required')
    rng = np.random.default_rng(seed)
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)


def patch_controls(image: np.ndarray, donor: np.ndarray, seed: int):
    """Return matched insertion, same-site edit, removal and relocation footprints.

    The donor edit is *uncertified* as logically valid. Removal erases an
    arbitrary image patch, not a segmented object. Relocation has separate
    source and destination masks; only the source footprint is a removal proxy.
    """
    x, d = np.asarray(image, np.float32), np.asarray(donor, np.float32)
    if x.shape != d.shape or x.ndim != 3 or x.shape[0] != 3:
        raise ValueError("paired good images must be aligned RGB")
    h, w = x.shape[1:]
    side = max(4, min(h, w) // 6)
    if min(h, w) < 3 * side:
        raise ValueError("image too small for disjoint edit regions")
    rng = np.random.default_rng(seed)
    sy = int(rng.integers(0, h - side + 1))
    sx = int(rng.integers(0, w - side + 1))
    ty = (sy + h // 2) % (h - side + 1)
    tx = (sx + w // 2) % (w - side + 1)
    if abs(sy - ty) < side and abs(sx - tx) < side:
        ty, tx = (sy + side) % (h - side + 1), (sx + side) % (w - side + 1)
    source_mask = np.zeros((h, w), np.uint8)
    target_mask = np.zeros((h, w), np.uint8)
    source_mask[sy:sy + side, sx:sx + side] = 1
    target_mask[ty:ty + side, tx:tx + side] = 1
    yy, xx = np.mgrid[:side, :side]
    distance = np.minimum.reduce((yy + 1, xx + 1, side - yy, side - xx))
    alpha = (0.8 * np.minimum(distance / max(1, min(3, side // 4)), 1.0))[None]

    def paste(base, y, xx, content):
        changed = base.copy()
        original = changed[:, y:y + side, xx:xx + side]
        changed[:, y:y + side, xx:xx + side] = original * (1 - alpha) + content * alpha
        return changed

    patch = x[:, sy:sy + side, sx:sx + side]
    insertion = paste(x, ty, tx, patch)
    # Same edit geometry, different appearance. The donor's logical validity
    # is unknown; this measures editing-process response only.
    donor_control = paste(x, ty, tx, d[:, ty:ty + side, tx:tx + side])
    appearance_control = paste(x, ty, tx, np.clip(x[:, ty:ty + side, tx:tx + side] * .95 + .025, 0, 1))
    # Constant local surround is a deliberately artifact-rich removal proxy.
    surround = x[:, max(0, sy - 1):sy + side + 1, max(0, sx - 1):sx + side + 1]
    fill = np.median(surround.reshape(3, -1), axis=1)[:, None, None]
    removal = paste(x, sy, sx, np.broadcast_to(fill, patch.shape))
    relocation = paste(removal, ty, tx, patch)
    return {"insertion": insertion, "donor_control": donor_control,
            "appearance_control": appearance_control, "removal_proxy": removal,
            "relocation_proxy": relocation, "source_mask": source_mask,
            "target_mask": target_mask}
