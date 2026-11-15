"""Known answers, held-out groups, provenance and evidence-level regressions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from dosimetry import GafchromicEngine
from validation import (ValidationConfig, write_synthetic_film, error_metrics, wilson_interval,
                        evaluate_reference_manifest, leave_one_dose_out, mcmc_diagnostics,
                        run_validation, _table, _sha256)


class NumericalValidationTests(unittest.TestCase):
    def test_metrics_have_known_values_and_validate_shapes(self):
        metrics = error_metrics(np.array([2., 4.]), np.array([1., 3.]))
        self.assertEqual(metrics['bias_gy'], 1)
        self.assertEqual(metrics['rmse_gy'], 1)
        self.assertEqual(metrics['peak_error_gy'], 1)
        with self.assertRaises(ValueError):
            error_metrics(np.zeros(2), np.zeros(3))
        with self.assertRaises(ValueError):
            error_metrics(np.array([np.nan]), np.zeros(1))

    def test_wilson_interval_is_not_a_pixel_count_confidence_claim(self):
        low, high = wilson_interval(95, 100)
        self.assertLess(low, .95); self.assertGreater(high, .95)
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        self.assertGreater(wilson_interval(1, 1)[1] - wilson_interval(1, 1)[0], .5)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_config_rejects_invalid_or_misleading_study_sizes(self):
        for kwargs in ({'seed': -1}, {'repetitions': 0}, {'image_size': 2},
                       {'posterior_samples': 1}, {'scan_repeats': 1}, {'sigma_od': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ValidationConfig(**kwargs).validate()

    def test_forward_fixture_has_known_intensities_and_repeatable_noise(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = ValidationConfig(image_size=12, repetitions=3, posterior_samples=2, sigma_od=0)
            truth = np.full((12, 12), 6.)
            data = write_synthetic_film(root/'known.tif', truth, np.random.default_rng(42), config, matched_lra=False)
            expected = np.rint(np.array([60000*10**-.6, 56000*10**-.3375, 52000*10**-.15])).astype(np.uint16)
            np.testing.assert_array_equal(tifffile.imread(root/'known.tif'), np.tile(expected, (12, 12, 1)))
            self.assertEqual(data['clipped_channel_values'], 0)
            self.assertEqual(set(data['sidecar_hashes']), {'std', 'sem', 'meta'})
            noisy = ValidationConfig(image_size=12, repetitions=3, posterior_samples=2)
            write_synthetic_film(root/'a.tif', truth, np.random.default_rng(7), noisy)
            write_synthetic_film(root/'b.tif', truth, np.random.default_rng(7), noisy)
            np.testing.assert_array_equal(tifffile.imread(root/'a.tif'), tifffile.imread(root/'b.tif'))
            np.testing.assert_array_equal(tifffile.imread(root/'a_scan_sem.tif'), tifffile.imread(root/'b_scan_sem.tif'))

    def test_holdout_excludes_all_replicates_at_a_dose_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); training = root/'training'; training.mkdir()
            config = ValidationConfig(image_size=12, repetitions=3, posterior_samples=2, sigma_od=0)
            doses = [0, 1, 2, 4, 4, 8, 12, 20]
            for i, dose in enumerate(doses):
                write_synthetic_film(training/f'{i:02d}.tif', np.full((12, 12), dose), np.random.default_rng(i), config)
            hashes = {path.name: _sha256(path) for path in training.iterdir()}
            calls = []
            original = GafchromicEngine.run_calibration
            def record(engine, folder, training_doses, **kwargs):
                calls.append(list(training_doses))
                return original(engine, folder, training_doses, **kwargs)
            with patch.object(GafchromicEngine, 'run_calibration', record):
                rows = leave_one_dose_out(training, doses, root/'folds', roi_size=3, posterior_samples=2)
            held_four = [row for row in rows if row['held_dose_gy'] == 4]
            self.assertEqual(len(held_four), 6)
            self.assertTrue(any(4 not in values and len(values) == 6 for values in calls))
            self.assertTrue(all(row['status'] == 'computed' for row in rows))
            self.assertTrue(all(row['extrapolation'] for row in rows if row['held_dose_gy'] == 20))
            self.assertTrue(all(row['held_dose_gy'] != 0 for row in rows))
            self.assertLess(max(abs(row['error_gy']) for row in rows), .1)
            self.assertEqual(hashes, {path.name: _sha256(path) for path in training.iterdir()})
            self.assertEqual(list((root/'folds').iterdir()), [])

    def test_report_never_overwrites_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); (path/'measurement.txt').write_text('original')
            with self.assertRaises(FileExistsError):
                run_validation(path)
            self.assertEqual((path/'measurement.txt').read_text(), 'original')
            self.assertFalse((path/'report.json').exists())


class ReferenceManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        np.save(self.root/'dose.npy', np.array([[0., 2.], [4., 6.]]))
        np.save(self.root/'sigma.npy', np.full((2, 2), .2))
        self.manifest = self.root/'reference.json'

    def write(self, case, **kwargs):
        self.manifest.write_text(json.dumps({'schema_version': 1, 'unit': 'Gy', 'cases': [case], **kwargs}))

    def test_scalar_reference_compares_roi_mean_not_invented_uniform_map(self):
        self.write({'id': 'reference_1', 'dose_map': 'dose.npy', 'sigma_map': 'sigma.npy',
                    'reference_mean_gy': 4, 'reference_sigma_gy': .1,
                    'independent_reference': True, 'uncertainties_independent': True})
        report = evaluate_reference_manifest(self.manifest)
        row = report['cases'][0]
        self.assertEqual(row['estimated_mean_gy'], 3)
        self.assertEqual(row['mean_difference_gy'], -1)
        self.assertNotIn('rmse_gy', row)
        self.assertAlmostEqual(row['estimated_mean_sigma_upper_bound_gy'], .2)
        self.assertAlmostEqual(row['combined_mean_sigma_upper_bound_gy'], np.hypot(.2, .1))
        self.assertEqual(row['reference_independence'], 'declared_not_verified')
        self.assertEqual(report['acceptance'], 'not_assessed')

    def test_map_comparison_requires_alignment_and_has_known_bias(self):
        np.save(self.root/'truth.npy', np.array([[1., 3.], [5., 7.]]))
        case = {'id': 'map_1', 'dose_map': 'dose.npy', 'reference_map': 'truth.npy'}
        self.write(case)
        with self.assertRaisesRegex(ValueError, 'alignment_confirmed'):
            evaluate_reference_manifest(self.manifest)
        self.write({**case, 'alignment_confirmed': True})
        row = evaluate_reference_manifest(self.manifest)['cases'][0]
        self.assertEqual(row['bias_gy'], -1)
        self.assertEqual(row['rmse_gy'], 1)
        self.assertNotIn('spatial_fraction_inside_nominal_95_band', row)

    def test_manifest_units_masks_and_unique_identifiers_are_checked(self):
        case = {'id': 'case1', 'dose_map': 'dose.npy', 'reference_mean_gy': 3}
        self.write(case, unit='cGy')
        with self.assertRaises(ValueError):
            evaluate_reference_manifest(self.manifest)
        np.save(self.root/'mask.npy', np.full((2, 2), 2))
        self.write({**case, 'mask': 'mask.npy'})
        with self.assertRaisesRegex(ValueError, 'mask'):
            evaluate_reference_manifest(self.manifest)
        self.write(case, cases=[case, case])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluate_reference_manifest(self.manifest)

    def test_invalid_reference_creates_an_error_report_not_a_pass(self):
        self.write({'id': 'missing', 'dose_map': 'absent.npy', 'reference_mean_gy': 3})
        report = run_validation(self.root/'report', ValidationConfig(repetitions=3),
                                reference_manifest=self.manifest, synthetic=False)
        self.assertEqual(report['status'], 'completed_with_errors')
        self.assertEqual(report['physical_validation']['status'], 'not_established')
        self.assertTrue((self.root/'report'/'report.html').exists())
        self.assertFalse((self.root/'report'/'.incomplete').exists())

    def test_html_escapes_input_text(self):
        markup = _table([{'label': '<script>alert(1)</script>'}])
        self.assertIn('&lt;script&gt;', markup)
        self.assertNotIn('<script>', markup)


class ValidationIntegrationTests(unittest.TestCase):
    def test_small_full_tiff_pipeline_report_and_independent_trial_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'report'
            config = ValidationConfig(seed=42, repetitions=3, image_size=12, posterior_samples=2)
            report = run_validation(root, config)
            self.assertEqual(report['status'], 'completed', report['errors'])
            self.assertEqual(report['physical_validation']['status'], 'not_established')
            self.assertEqual(len(report['solver_comparison']), 36)
            self.assertEqual(len(report['coverage']['observations']), 9)
            self.assertTrue(all(row['experiments'] == 1 for row in report['coverage']['summary']))
            self.assertTrue(all(row['empirical_sd_gy'] is None for row in report['coverage']['summary']))
            hotspots = [row for row in report['solver_comparison'] if row['case'] == 'single_pixel_hotspot']
            self.assertTrue(all(row['peak_error_gy'] < -10 for row in hotspots))
            self.assertTrue(all(row['outside_calibration_range'] for row in report['solver_comparison'] if row['case'] == 'high_dose_80'))
            loaded = json.loads((root/'report.json').read_text())
            self.assertEqual(loaded['configuration']['seed'], 42)
            self.assertIn('dosimetry.py', loaded['source_hashes'])
            self.assertEqual(len(loaded['calibration_inputs']), 7)
            self.assertIn('data:image/png;base64,', (root/'report.html').read_text())
            self.assertTrue((root/'coverage_observations.csv').exists())
            self.assertFalse((root/'.incomplete').exists())


@unittest.skipUnless(importlib.util.find_spec('arviz'), 'Optional Bayesian diagnostics dependency is absent.')
class MCMCDiagnosticTests(unittest.TestCase):
    def test_single_chain_is_not_declared_converged(self):
        import arviz as az
        with tempfile.TemporaryDirectory() as folder:
            trace = az.from_dict({'posterior': {'b': np.random.default_rng(4).normal(size=(1, 20))}})
            path = Path(folder)/'trace.nc'; trace.to_netcdf(path); trace.close()
            result = mcmc_diagnostics(path)
            self.assertEqual(result['status'], 'insufficient_chains_or_draws')
            self.assertEqual(result['convergence'], 'not_established_by_software')
            self.assertIsNone(result['divergences'])

    def test_rank_ess_and_divergences_are_reported_without_approval(self):
        import arviz as az
        with tempfile.TemporaryDirectory() as folder:
            divergence = np.zeros((4, 100), dtype=bool); divergence[0, 0] = True
            trace = az.from_dict({'posterior': {'b': np.random.default_rng(4).normal(size=(4, 100))},
                                  'sample_stats': {'diverging': divergence}})
            path = Path(folder)/'trace.nc'; trace.to_netcdf(path); trace.close()
            result = mcmc_diagnostics(path)
            self.assertEqual(result['status'], 'computed')
            self.assertEqual(result['divergences'], 1)
            self.assertGreater(result['parameters']['b']['ess_bulk_min'], 0)
            self.assertGreater(result['parameters']['b']['rhat_max'], 0)
            self.assertEqual(result['convergence'], 'not_established_by_software')


if __name__ == '__main__':
    unittest.main()
