import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from preprocessing import (
    build_normalization_config, build_preprocessing_config,
    normalize_x_np, normalize_y_np, denormalize_y_np,
    preprocessing_from_checkpoint, RAW_X_SCALE, RAW_Y_SCALE,
)
from friction_emulator import friction_emulator as inference
from train import predict_raw

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingOptionsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = np.column_stack((rng.uniform(1e5, 1e7, 160), rng.uniform(1e-7, 2e-5, 160)))
        self.y = (self.x[:, :1] * self.x[:, 1:] ** (-2 / 3))
        self.x_floor = np.array([1e-30, 1e-12])
        self.y_floor = np.array([1e-30])

    def config(self, transform, scaling):
        return build_preprocessing_config(transform, scaling, self.x, self.y,
                                          self.x_floor, self.y_floor)

    def test_transform_scaling_combinations_and_legacy_equivalence(self):
        for transform, scaling in itertools.product(('none', 'sqrt', 'log'), ('fixed', 'standard')):
            with self.subTest(transform=transform, scaling=scaling):
                norm = self.config(transform, scaling)
                np.testing.assert_allclose(denormalize_y_np(normalize_y_np(self.y, norm), norm), self.y,
                                           rtol=1e-12)
                if scaling == 'standard':
                    np.testing.assert_allclose(normalize_x_np(self.x, norm).mean(0), 0, atol=1e-12)
                    np.testing.assert_allclose(normalize_x_np(self.x, norm).std(0), 1, atol=1e-12)
                else:
                    scale = {'none': lambda x: x, 'sqrt': np.sqrt, 'log': lambda x: np.abs(np.log(x))}[transform]
                    np.testing.assert_array_equal(norm.x_std, scale(RAW_X_SCALE))
                    np.testing.assert_array_equal(norm.y_std, scale(RAW_Y_SCALE))
                if scaling == 'standard' or transform == 'none':
                    legacy = build_normalization_config(norm.mode, self.x, self.y, self.x_floor, self.y_floor)
                    np.testing.assert_array_equal(normalize_x_np(self.x, norm), normalize_x_np(self.x, legacy))
                    np.testing.assert_array_equal(normalize_y_np(self.y, norm), normalize_y_np(self.y, legacy))

    def test_checkpoint_inference_legacy_and_explicit_metadata(self):
        model = inference.FrictionMLP(in_dim=2, h1=4, h2=4, out_dim=1)
        # A bounded constant prediction also exercises log + fixed without overflow.
        for p in model.parameters():
            p.data.zero_()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            for transform, scaling in itertools.product(('none', 'sqrt', 'log'), ('fixed', 'standard')):
                norm = self.config(transform, scaling)
                checkpoint = dict(in_dim=2, h1=4, h2=4, out_dim=1, state_dict=model.state_dict(),
                                  **{k: getattr(norm, k) for k in ('x_mean','x_std','y_mean','y_std','x_floor','y_floor')})
                expected = predict_raw(model, self.x, norm, torch.device('cpu'), 64)
                for metadata in ({'normalization': norm.mode}, {'transform': transform, 'scaling': scaling}):
                    with self.subTest(transform=transform, scaling=scaling, metadata=metadata):
                        c = {**checkpoint, **metadata}
                        torch.save(c, path)
                        inference.init_model(str(path), device='cpu')
                        np.testing.assert_allclose(inference.predict_alpha2_np(self.x), expected, rtol=1e-6)
                        loaded = preprocessing_from_checkpoint(c)
                        np.testing.assert_array_equal(normalize_x_np(self.x, loaded), normalize_x_np(self.x, norm))

    def test_training_cli_artifacts_and_matching_holdouts(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            data = tmp / 'data'
            data.mkdir()
            np.savetxt(data / 'weertman_rank_0.csv', np.column_stack((self.x, self.y)), delimiter=',')
            reports = []
            # Exercise all user-visible combinations through training and export.
            for transform, sampling, scaling in itertools.product(('none', 'sqrt', 'log'), ('uniform', 'weighted'), ('standard', 'fixed')):
                name = f'{transform}_{sampling}_{scaling}'
                cmd = [sys.executable, str(ROOT / 'train.py'), '--folder', str(data), '--n-ranks', '1',
                       '--device', 'cpu', '--epochs', '1', '--n-seeds', '1', '--h1', '4', '--h2', '4',
                       '--batch-size', '32', '--train-samples', '50', '--transform', transform,
                       '--sampling', sampling, '--scaling', scaling]
                result = subprocess.run(cmd, cwd=tmp, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, name + '\n' + result.stdout + result.stderr)
                out = tmp / 'friction_emulator'
                c = torch.load(out / f'model_{name}.pt', map_location='cpu', weights_only=False)
                self.assertEqual((c['transform'], c['sampling'], c['scaling']), (transform, sampling, scaling))
                self.assertEqual(c['train_samples'], 50)
                self.assertEqual(c['joint_sampling_enabled'], sampling == 'weighted')
                self.assertTrue(c['vmag_filter_enabled'])
                self.assertTrue((out / f'model_{name}.txt').exists())
                with np.load(out / f'training_report_{name}.npz') as report:
                    reports.append(report['test_x_raw'].copy())
                    self.assertTrue(np.isfinite(report['test_pred']).all())
            for inputs in reports[1:]:
                np.testing.assert_array_equal(inputs, reports[0])


if __name__ == '__main__':
    unittest.main()
