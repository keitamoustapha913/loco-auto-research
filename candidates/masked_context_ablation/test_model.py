"""Run only through an operator-approved bounded test runner."""
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
import model


class ContractTests(unittest.TestCase):
    def test_roundtrip_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(5)
            image = root / 'good.png'
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(image)
            manifest = root / 'good.json'
            manifest.write_text(json.dumps({'images': [{'path': str(image), 'split': 'train/good', 'category': 'fixture'}]}))
            meta = model.train({'device': 'cpu', 'size': 32, 'steps': 2, 'batch_size': 1}, manifest, root / 'out')
            self.assertTrue(meta['parameter_updated'])
            predictor = model.load(root / 'out/checkpoint_manifest.json')
            x = rng.random((2, 3, 35, 47), dtype=np.float32)
            result = predictor.predict(x)
            self.assertEqual(result['anomaly_maps'].shape, (2, 1, 35, 47))
            self.assertEqual(result['image_scores'].shape, (2,))
            self.assertTrue(np.isfinite(result['anomaly_maps']).all())
            self.assertGreater(float(result['anomaly_maps'].std()), 0)
            np.testing.assert_array_equal(result['anomaly_maps'], predictor.predict(x)['anomaly_maps'])

    def test_exclusion(self):
        net = model.Net().eval()
        x = torch.rand(1, 3, 64, 64)
        hidden, visible = model.masks(64, 0, 'cpu')
        edited = x + (1 - visible) * torch.rand_like(x)
        for arm in ('local', 'global', 'combined', 'neither', 'spatial'):
            torch.testing.assert_close(net(x, visible, arm), net(edited, visible, arm))

    def test_arms_share_parameterization_and_differ_only_by_context(self):
        net = model.Net().eval()
        x = torch.rand(1, 3, 64, 64)
        _, visible = model.masks(64, 0, 'cpu')
        outputs = {arm: net(x, visible, arm) for arm in ('local', 'global', 'combined')}
        self.assertEqual({value.shape for value in outputs.values()}, {(1, 12, 64, 64)})
        self.assertGreater(float((outputs['local'] - outputs['global']).abs().max()), 0)
        self.assertGreater(float((outputs['combined'] - outputs['global']).abs().max()), 0)

    def test_neither_and_spatial_baseline_cannot_read_image_pixels(self):
        net = model.Net().eval()
        x = torch.rand(1, 3, 64, 64)
        _, visible = model.masks(64, 0, 'cpu')
        for arm in ('neither', 'spatial'):
            torch.testing.assert_close(net(x, visible, arm), net(torch.zeros_like(x), visible, arm))
        _, other_visible = model.masks(64, 1, 'cpu')
        torch.testing.assert_close(net(x, visible, 'spatial'), net(x, other_visible, 'spatial'))

    def test_cuda_gate(self):
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                model._device('cuda')


if __name__ == '__main__':
    unittest.main()
