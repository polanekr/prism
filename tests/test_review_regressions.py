"""Distribution meaning, exact parallel propagation and legacy noise provenance."""
from qt_helpers import dispose_widget
import contextlib
import io
import json
import tempfile
import warnings
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from dose_solvers import calibration_worker_count, ordered_realizations
from dosimetry import GafchromicEngine
from scan_data import load_scan_noise, scan_paths, LegacyScanNoiseWarning
from uncertainty import beam_dose_distribution


class BeamDistributionTests(unittest.TestCase):
    def test_peak_and_area_scaling_are_distinct_without_changing_dah(self):
        doses = np.r_[np.zeros(10), np.ones(30), np.full(60, 2.)]
        peak = beam_dose_distribution(doses, bins=3, normalization='peak')
        area = beam_dose_distribution(doses, bins=3, normalization='area')
        self.assertEqual(peak['histogram'].max(), 100.)
        self.assertEqual(area['histogram'].sum(), 100.)
        self.assertEqual(area['histogram'].max(), 60.)
        np.testing.assert_array_equal(peak['dah_area'], area['dah_area'])
        self.assertEqual(peak['dah_dose'][0], 0.)
        self.assertEqual(peak['dah_area'][0], 100.)
        self.assertEqual(peak['dah_area'][-1], 0.)
        self.assertTrue(np.all(np.diff(peak['dah_area']) <= 0))
        threshold = np.nextafter(0., np.inf)
        self.assertEqual(peak['dah_area'][np.flatnonzero(peak['dah_dose'] == threshold)[0]], 90.)

    def test_high_offset_doses_use_all_bins_without_changing_dah(self):
        doses = np.linspace(10., 12., 1001)
        fine = beam_dose_distribution(doses, bins=50, normalization='area')
        coarse = beam_dose_distribution(doses, bins=10, normalization='area')
        self.assertEqual(len(fine['histogram']), 50)
        self.assertEqual(np.count_nonzero(fine['histogram']), 50)
        np.testing.assert_allclose(np.diff(fine['edges']), .04)
        self.assertAlmostEqual(fine['histogram'].sum(), 100.)
        np.testing.assert_array_equal(fine['dah_dose'], coarse['dah_dose'])
        np.testing.assert_array_equal(fine['dah_area'], coarse['dah_area'])
        translated = beam_dose_distribution(doses-10., bins=50, normalization='area')
        np.testing.assert_allclose(fine['edges']-10., translated['edges'], atol=1e-14)
        nearly_constant = beam_dose_distribution([11., np.nextafter(11., np.inf)], normalization='area')
        self.assertAlmostEqual(nearly_constant['histogram'].sum(), 100.)
        self.assertTrue(np.all(np.diff(nearly_constant['edges']) > 0))

    def test_constant_dose_and_zero_dose_have_correct_endpoint_jumps(self):
        for dose in (0., 7.):
            data = beam_dose_distribution(np.full(100, dose))
            self.assertEqual(data['histogram'].max(), 100.)
            self.assertEqual(data['dah_area'][0], 100.)
            self.assertEqual(data['dah_area'][-1], 0.)
            self.assertEqual(data['dah_dose'][-1], np.nextafter(dose, np.inf))
            self.assertTrue(np.all(data['edges'] >= 0))

    def test_large_roi_keeps_bounded_plot_size_and_all_area(self):
        data = beam_dose_distribution(np.linspace(0, 100, 100000))
        self.assertLessEqual(data['dah_dose'].size, 2049)
        self.assertEqual(data['dah_area'][0], 100.)
        self.assertEqual(data['dah_area'][-1], 0.)


class ParallelCalibrationTests(unittest.TestCase):
    def test_worker_count_respects_samples_and_scratch_budget(self):
        self.assertEqual(calibration_worker_count((100, 100), 4, 100), 4)
        self.assertEqual(calibration_worker_count((100, 100), 4, 2), 2)
        self.assertEqual(calibration_worker_count((4000, 4000), 8, 100), 1)
        for value in (0, 9, True):
            with self.assertRaises(ValueError): calibration_worker_count((10, 10), value, 100)

    def test_parallel_samples_preserve_bitwise_std_and_engine_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = GafchromicEngine(directory)
            params = {'R': [0., 1.2, 6., 1.], 'G': [0., .9, 10., 1.], 'B': [0., .6, 18., 1.]}
            engine.calib_params = {ch: np.array(values) for ch, values in params.items()}
            engine.calib_doses = np.array([0., 1., 2., 4., 8., 20.])
            rng = np.random.default_rng(42)
            for ch in 'RGB':
                engine.calib_samples[ch] = np.tile(engine.calib_params[ch], (6, 1))
                engine.calib_samples[ch][:, 1] *= 1+rng.normal(0, .005, 6)
            truth = np.full((16, 16), 6.)
            od = np.stack([engine.rational_func_od(truth, *params[ch]) for ch in 'RGB'], axis=-1)
            noise = np.full_like(od, .003)
            for method in ('fast', 'mayer', 'admm'):
                dose, _ = engine._solve_dose(od, method, noise)
                serial = engine._compute_calibration_std_map(od, dose, method, noise, workers=1)
                parallel = engine._compute_calibration_std_map(od, dose, method, noise, workers=4)
                np.testing.assert_array_equal(serial, parallel)
            for ch in 'RGB': np.testing.assert_array_equal(engine.calib_params[ch], params[ch])

    def test_worker_failure_does_not_turn_into_partial_std(self):
        def calculate(index):
            if index == 2: raise ValueError('invalid realization')
            return index
        with self.assertRaisesRegex(ValueError, 'invalid realization'):
            list(ordered_realizations(calculate, range(10), 3))


class LegacyNoiseReviewTests(unittest.TestCase):
    def test_legacy_warning_is_deduplicated_but_noise_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            mean = Path(directory)/'film.tif'; paths = scan_paths(mean)
            tifffile.imwrite(mean, np.full((3, 4, 3), 1000, dtype=np.uint16), photometric='rgb')
            tifffile.imwrite(paths['std'], np.full((3, 4, 3), 6., dtype=np.float32), photometric='rgb')
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                first, _ = load_scan_noise(mean, (3, 4, 3))
                second, _ = load_scan_noise(mean, (3, 4, 3))
            self.assertEqual(sum(issubclass(item.category, LegacyScanNoiseWarning) for item in captured), 1)
            np.testing.assert_array_equal(first, second)
            paths['meta'].write_text(json.dumps({'scan_count': 3}))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                corrected, source = load_scan_noise(mean, (3, 4, 3))
            self.assertFalse(captured)
            self.assertEqual(source, 'repeat_std_to_sem')
            np.testing.assert_allclose(corrected, first/np.sqrt(3))

    def test_mcmc_unknown_adaptation_is_rejected_without_changing_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = GafchromicEngine(directory)
            original = engine.calib_params
            with self.assertRaisesRegex(ValueError, 'adaptation'):
                engine.run_mcmc_calibration('unused', [], adaptation='invalid')
            self.assertIs(engine.calib_params, original)


class BeamChartReviewTests(unittest.TestCase):
    def test_gui_defaults_and_optional_band_are_explicit(self):
        import sys
        from PyQt5.QtWidgets import QApplication
        from prism_gui import PrismDosimetryGUI
        app = QApplication.instance() or QApplication([])
        streams = sys.stdout, sys.stderr
        window = PrismDosimetryGUI()
        sys.stdout, sys.stderr = streams
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(window, '_fit_2d_gaussian', return_value=None):
                path = Path(directory)/'beam_dose.npy'
                np.save(path, np.tile(np.arange(40, dtype=float), (40, 1)))
                np.save(path.with_name('beam_std.npy'), np.full((40, 40), .5))
                window.load_and_display_dose(path)
                window.morphology_tab.invalidate_analysis()
                self.assertEqual(window.combo_hist_normalization.currentIndex(), 0)
                self.assertEqual(window.curve_roi_hist.yData.max(), 100.)
                self.assertEqual(window.curve_roi_dah.yData[0], 100.)
                self.assertEqual(window.curve_roi_dah.yData[-1], 0.)
                self.assertEqual(window.curve_dah_low.yData.size, 0)
                original = window.curve_roi_dah.yData.copy()
                window.combo_hist_normalization.setCurrentIndex(1)
                self.assertAlmostEqual(window.curve_roi_hist.yData.sum(), 100.)
                np.testing.assert_array_equal(window.curve_roi_dah.yData, original)
                window.combo_dah_band.setCurrentIndex(1)
                self.assertTrue(np.all(window.curve_dah_low.yData <= window.curve_roi_dah.yData))
                self.assertTrue(np.all(window.curve_dah_high.yData >= window.curve_roi_dah.yData))
                self.assertIn('not a confidence interval', window.plot_roi_hist.toolTip())
        finally:
            window.morphology_tab.invalidate_analysis()
            dispose_widget(window); app.processEvents()
            sys.stdout, sys.stderr = streams
