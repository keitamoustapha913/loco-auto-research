from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from scripts.run_candidate import load_image, restore_anomaly_map, alternate_corruption, split_validation_paths


def test_original_geometry_and_deterministic_preprocessing(tmp_path):
    path = tmp_path / 'normal.png'
    image = np.zeros((37, 53, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(path)
    first = load_image(path, 16)
    assert first.shape == (3, 16, 16)
    assert np.array_equal(first, load_image(path, 16))
    grid = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    restored = restore_anomaly_map(grid, (53, 37))
    assert restored.shape == (37, 53) and np.isfinite(restored).all()
    with pytest.raises(ValueError):
        restore_anomaly_map(np.full((16, 16), np.nan), (53, 37))


def test_alternate_good_only_generator_reproducible():
    image = np.random.default_rng(1).random((3, 32, 32), dtype=np.float32)
    output, mask = alternate_corruption(image, 7)
    second, second_mask = alternate_corruption(image, 7)
    assert np.array_equal(output, second)
    assert np.array_equal(mask, second_mask)
    assert mask.shape == (32, 32) and mask.sum() > 0


def test_validation_partitions_are_stable_and_disjoint():
    paths = [Path(f'normal_{i}.png') for i in range(54)]
    calibration, assessment = split_validation_paths(paths)
    assert len(calibration) == 18 and len(assessment) == 36
    assert not set(calibration) & set(assessment)
    assert (calibration, assessment) == split_validation_paths(paths)
    with pytest.raises(ValueError):
        split_validation_paths([paths[0], paths[0]])
