"""Analytical, ensemble and display checks for uncertainty propagation."""
from qt_helpers import dispose_widget
from qt_helpers import wait_for_tasks
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
import pyqtgraph as pg
import tifffile
from scipy.ndimage import median_filter
from PyQt5.QtWidgets import QApplication

from dosimetry import GafchromicEngine
from prism_gui import PrismDosimetryGUI
from bio_dosimetry import BioDosimetryTab
from uncertainty import coupled_dose_variance, mean_sigma_bound, cumulative_area, dose_distribution_bounds


PARAMS = {'R': np.array([0., 1., 4., 1.]), 'G': np.array([0., .8, 8., 1.]),
          'B': np.array([0., .5, 12., 1.])}


class SolverUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = GafchromicEngine(self.temp.name)
        self.engine.calib_params = {ch: p.copy() for ch, p in PARAMS.items()}
        self.engine.calib_doses = np.array([0, 1, 5, 10, 40])

    def od(self, doses):
        doses = np.asarray(doses, dtype=float)
        return np.stack([self.engine.rational_func_od(doses, *PARAMS[ch]) for ch in 'RGB'], axis=-1)

    def test_all_solvers_propagate_their_own_filtered_parameter_realizations(self):
        doses = np.linspace(2, 12, 35).reshape(5, 7)
        doses[2, 3] = 20
        od = self.od(doses)
        noise = np.full(od.shape, .003)
        for ch, direction in zip('RGB', [1, -1, 2]):
            samples = np.tile(PARAMS[ch], (3, 1))
            samples[:, 1] *= 1 + direction * np.array([-.01, 0, .01])
            self.engine.calib_samples[ch] = samples
        original = {ch: p.copy() for ch, p in self.engine.calib_params.items()}
        for method in ('fast', 'mayer', 'admm'):
            with self.subTest(method=method):
                expected = []
                solver = getattr(self.engine, '_solver_' + method)
                for i in range(3):
                    params = {ch: self.engine.calib_samples[ch][i] for ch in 'RGB'}
                    realization, _ = solver(od, od_noise_map=noise, params_dict=params)
                    expected.append(median_filter(realization, size=3))
                actual = self.engine._compute_calibration_std_map(od, doses, method, noise)
                np.testing.assert_allclose(actual, np.std(expected, axis=0, ddof=1), atol=1e-12)
        for ch in 'RGB':
            np.testing.assert_array_equal(self.engine.calib_params[ch], original[ch])

    def test_fast_calibration_uncertainty_does_not_mix_in_unused_blue_channel(self):
        doses = np.full((5, 5), 25.)
        for ch in 'RGB':
            self.engine.calib_samples[ch] = np.tile(PARAMS[ch], (3, 1))
        self.engine.calib_samples['B'][:, 1] *= [.8, 1, 1.2]
        actual = self.engine._compute_calibration_std_map(self.od(doses), doses, 'fast')
        np.testing.assert_allclose(actual, 0, atol=1e-12)

    def test_covariance_fallback_includes_parameter_cross_terms(self):
        for ch in 'RGB':
            self.engine.calib_cov[ch] = np.zeros((4, 4))
        cov = self.engine.calib_cov['R']
        cov[0, 0], cov[2, 2], cov[0, 2], cov[2, 0] = .0001, .001, .0002, .0002
        doses = np.full((5, 5), 2.)
        actual = self.engine._compute_calibration_std_map(self.od(doses), doses, 'fast')
        # D = c*(OD-a)/(a+b-OD); at D=2, dD/da=-9 and dD/dc=.5.
        expected = np.sqrt(81 * .0001 + .25 * .001 + 2 * (-9) * .5 * .0002)
        np.testing.assert_allclose(actual, expected, rtol=2e-7)
        self.assertEqual(self.engine._calibration_uncertainty_source(), 'parameter_covariance')

    def test_missing_parameter_uncertainty_is_reported_and_not_replaced_by_od_noise(self):
        doses = np.full((3, 3), 2.)
        with self.assertWarnsRegex(UserWarning, 'Calibration uncertainty unavailable'):
            actual = self.engine._compute_calibration_std_map(self.od(doses), doses)
        np.testing.assert_array_equal(actual, 0)
        self.assertEqual(self.engine._calibration_uncertainty_source(), 'unavailable')

    def test_single_parameter_sample_uses_covariance_when_available(self):
        for ch in 'RGB':
            self.engine.calib_samples[ch] = PARAMS[ch][None, :]
            self.engine.calib_cov[ch] = np.zeros((4, 4))
        self.engine.calib_cov['R'][0, 0] = .0001
        doses = np.full((3, 3), 2.)
        actual = self.engine._compute_calibration_std_map(self.od(doses), doses)
        np.testing.assert_allclose(actual, .09, rtol=1e-7)

    def test_fast_noise_includes_derivative_of_blend_weights(self):
        od = self.od(np.full((3, 3), 8.))
        # Deliberately inconsistent green dose, while the red dose lies inside the transition.
        od[:, :, 1] = self.engine.rational_func_od(7., *PARAMS['G'])
        noise = np.full(od.shape, .001)
        nominal, sigma = self.engine._solver_fast(od, noise)
        variance = np.zeros(nominal.shape)
        for ch in (0, 1):
            plus, minus = od.copy(), od.copy()
            plus[:, :, ch] += 1e-6
            minus[:, :, ch] -= 1e-6
            gradient = (self.engine._solver_fast(plus)[0] - self.engine._solver_fast(minus)[0]) / 2e-6
            variance += (gradient * noise[:, :, ch])**2
        discrepancy = .5 * .5 * 1.
        np.testing.assert_allclose(sigma**2, variance + discrepancy**2, rtol=1e-7)

    def test_zero_dose_does_not_force_fast_or_mayer_uncertainty_to_zero(self):
        od = self.od(np.zeros((3, 3)))
        for method in ('fast', 'mayer', 'admm'):
            with self.subTest(method=method):
                dose, sigma = self.engine._solve_dose(od, method)
                np.testing.assert_allclose(dose, 0)
                self.assertTrue(np.all(sigma > 0))
                self.assertTrue(np.all(np.isfinite(sigma)))

    def test_admm_local_variance_matches_full_information_matrix(self):
        doses = np.full((5, 5), 6.)
        od = self.od(doses)
        noise = np.full(od.shape, .003)
        dose, sigma = self.engine._solver_admm(od, noise)
        jac = np.array([[self.engine.rational_func_dod_dd(6., *PARAMS[ch]),
                         self.engine.rational_func_od(6., *PARAMS[ch])] for ch in 'RGB'])
        information = jac.T @ np.diag(np.array([1., 1., .8]) / .003**2) @ jac
        expected = np.sqrt(np.linalg.inv(information)[0, 0])
        np.testing.assert_allclose(dose, 6., atol=1e-10)
        np.testing.assert_allclose(sigma, expected, rtol=1e-8)
        self.assertGreater(expected, 1.1 / np.sqrt(information[0, 0]))

    def test_admm_enforces_positive_dose_after_each_tv_update(self):
        from skimage.restoration import denoise_tv_chambolle
        y, x = np.indices((24, 24))
        truth = 12*np.exp(-((x-11.5)**2+(y-11.5)**2)/(2*4.8**2))
        od = np.maximum(self.od(truth) + np.random.default_rng(42).normal(0, .003, (24, 24, 3)), 0)
        previous_dual_inputs = []
        def tv(data, **kwargs):
            previous_dual_inputs.append(data.copy())
            return denoise_tv_chambolle(data, **kwargs)
        with patch('dose_solvers.denoise_tv_chambolle', side_effect=tv):
            dose, sigma = self.engine._solver_admm(od)
        self.assertEqual(len(previous_dual_inputs), 6)
        self.assertTrue(np.all(np.isfinite(dose)))
        self.assertTrue(np.all(dose >= 0))
        self.assertTrue(np.all(np.isfinite(sigma)))
        dose_only, omitted_sigma = self.engine._solver_admm(od, dose_only=True)
        np.testing.assert_array_equal(dose, dose_only)
        self.assertIsNone(omitted_sigma)

    def test_admm_projects_tv_state_before_updating_dual_variable(self):
        od = self.od(np.full((3, 3), 6.))
        with patch('dose_solvers.denoise_tv_chambolle', return_value=np.full((3, 3), -.01)):
            dose, sigma = self.engine._solver_admm(od)
        np.testing.assert_array_equal(dose, np.zeros((3, 3)))
        self.assertTrue(np.all(np.isfinite(sigma)))

    def test_invalid_output_names_film_solver_and_actual_component(self):
        root = Path(self.temp.name); path = root/'bad_film.tif'
        tifffile.imwrite(path, np.full((5, 5, 3), 30000, dtype=np.uint16), photometric='rgb')
        with patch.object(self.engine, '_solve_dose', return_value=(np.full((5, 5), -1.), np.ones((5, 5)))), \
                patch.object(self.engine, '_compute_calibration_std_map', return_value=np.zeros((5, 5))), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, r'bad_film.tif.*solver=admm.*dose: 0 nonfinite, 25 negative'):
            self.engine.process_films([str(path)], '', method='admm', blank_rgb=[60000, 56000, 52000])
        self.assertFalse((root/'bad_film_dose.npy').exists())

    def test_unidentifiable_thickness_is_not_given_finite_uncertainty(self):
        self.engine.calib_params = {ch: PARAMS['R'].copy() for ch in 'RGB'}
        value = self.engine.rational_func_od(6., *PARAMS['R'])
        _, sigma = self.engine._solver_admm(np.full((3, 3, 3), value))
        self.assertTrue(np.all(np.isinf(sigma)))

    def test_processing_exports_separate_components_and_assumptions(self):
        folder = Path(self.temp.name)
        blank_rgb = np.array([60000., 56000., 52000.])
        od = self.od(np.full((5, 5), 2.))
        film = np.rint(blank_rgb * 10**-od).astype(np.uint16)
        path = folder / 'film.tif'
        tifffile.imwrite(path, film, photometric='rgb')
        for ch in 'RGB':
            self.engine.calib_samples[ch] = np.tile(PARAMS[ch], (3, 1))
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine.process_films([str(path)], '', method='fast', blank_rgb=blank_rgb)
        meta = json.loads((folder / 'film_meta.json').read_text())
        fit = np.load(folder / meta['std_fit_filename'])
        calibration = np.load(folder / meta['std_calib_filename'])
        total = np.load(folder / meta['std_filename'])
        dose = np.load(folder / 'film_dose.npy')
        floor = np.where(dose > .01, dose * .005, 1e-4)
        np.testing.assert_allclose(total, np.maximum(np.hypot(fit, calibration), floor))
        np.testing.assert_allclose(calibration, 0)
        self.assertTrue(meta['calibration_uncertainty_available'])
        self.assertEqual(meta['calibration_uncertainty_source'], 'parameter_samples')
        self.assertEqual(meta['uncertainty_model_version'], 3)
        self.assertIn('solver_diagnostics', meta)

    def test_invalid_noise_is_rejected_before_writing_outputs(self):
        path = Path(self.temp.name) / 'film.tif'
        with patch.object(self.engine, '_load_tiff_with_meta', return_value=(np.full((3, 3, 3), 100.), {})), \
                patch('film_processing.load_scan_noise', return_value=(np.full((3, 3, 3), np.nan), 'mean_sem')), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, 'Invalid OD uncertainty'):
            self.engine.process_films([str(path)], '', method='fast', blank_rgb=[200, 200, 200])
        self.assertFalse(path.with_name('film_dose.npy').exists())


class UncertaintyBoundsTests(unittest.TestCase):
    def test_local_information_inverse_and_singular_cases(self):
        self.assertAlmostEqual(float(coupled_dose_variance(2, 1, 3)), .6)
        self.assertAlmostEqual(float(coupled_dose_variance(2, 0, 0)), .5)
        self.assertTrue(np.isinf(coupled_dose_variance(1, 1, 1)))
        self.assertTrue(np.isinf(coupled_dose_variance(0, 0, 0)))

    def test_perfectly_correlated_errors_do_not_shrink_with_pixel_count(self):
        for count in (1, 4, 100):
            self.assertEqual(mean_sigma_bound(np.full((count, 5), 2.), axis=0).tolist(), [2.] * 5)

    def test_cumulative_area_keeps_cells_outside_plotted_threshold_range(self):
        np.testing.assert_allclose(cumulative_area([0, 1, 100], [0, 1, 2]), [100, 200/3, 100/3])

    def test_distribution_bounds_contain_nominal_without_confidence_claim(self):
        doses, sigmas = np.array([0., 1., 2.]), np.array([0., 1., 3.])
        thresholds = np.array([0., 1., 3., 100.])
        bounds = dose_distribution_bounds(doses, sigmas, thresholds)
        central = cumulative_area(doses, thresholds)
        self.assertTrue(np.all(bounds['dah_low'] <= central))
        self.assertTrue(np.all(bounds['dah_high'] >= central))
        self.assertEqual(bounds['dah_low'][0], 100)
        self.assertAlmostEqual(bounds['dah_high'][2], 100/3)
        self.assertLessEqual(bounds['d90_low'], np.percentile(doses, 10))
        self.assertGreaterEqual(bounds['d90_high'], np.percentile(doses, 10))

    def test_zero_sigma_envelopes_collapse_to_nominal(self):
        doses, thresholds = np.array([0, 1, 2]), np.array([0, 1, 2, 3])
        bounds = dose_distribution_bounds(doses, np.zeros(3), thresholds)
        np.testing.assert_allclose(bounds['dah_low'], cumulative_area(doses, thresholds))
        np.testing.assert_allclose(bounds['dah_high'], bounds['dah_low'])
        self.assertEqual(bounds['d90_low'], bounds['d90_high'])


class ProfileAndSampleUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        pg.setConfigOptions(imageAxisOrder='row-major')

    def test_profile_uses_conservative_band_and_clears_stale_band(self):
        fake = SimpleNamespace(current_dose_map=np.full((10, 10), 5.), current_std_map=np.full((10, 10), .2),
                               v_line=SimpleNamespace(value=lambda: 5), h_line=SimpleNamespace(value=lambda: 5),
                               spin_band=SimpleNamespace(value=lambda: 5), combo_prof_mode=SimpleNamespace(currentIndex=lambda: 0),
                               plot_prof_x=pg.PlotWidget(), plot_prof_y=pg.PlotWidget())
        self.addCleanup(dispose_widget, fake.plot_prof_x)
        self.addCleanup(dispose_widget, fake.plot_prof_y)
        for name in ('curve_prof_x', 'curve_prof_y', 'curve_prof_x_up', 'curve_prof_x_dn',
                     'curve_prof_y_up', 'curve_prof_y_dn'):
            setattr(fake, name, pg.PlotCurveItem())
        PrismDosimetryGUI.update_profiles(fake, 0, 10, 0, 10)
        np.testing.assert_allclose(fake.curve_prof_x_up.yData, 5.2)
        np.testing.assert_allclose(fake.curve_prof_y_up.xData, 5.2)
        fake.current_std_map = None
        PrismDosimetryGUI.update_profiles(fake, 0, 10, 0, 10)
        self.assertEqual(fake.curve_prof_x_up.yData.size, 0)
        self.assertEqual(fake.curve_prof_y_up.xData.size, 0)

    def test_shared_front_back_uncertainty_is_not_assumed_independent(self):
        tab = BioDosimetryTab()
        self.addCleanup(dispose_widget, tab)
        tab.front_dose = tab.back_dose = np.full((20, 20), 10.)
        tab.front_std = tab.back_std = np.full((20, 20), .2)
        tab.img_front.setImage(tab.front_dose)
        tab.img_back.setImage(tab.back_dose)
        tab.geometry.d_air_front = tab.geometry.d_air_back = 0
        roi = pg.EllipseROI([3, 3], [6, 6])
        tab.plot.addItem(roi)
        tab.rois.append(roi)
        with patch('bio_dosimetry.QMessageBox.information'), patch('bio_dosimetry.QMessageBox.warning') as warning:
            tab.calculate_doses()
            wait_for_tasks(tab.tasks)
        warning.assert_not_called()
        self.assertAlmostEqual(tab.results_data[1]['uncert_dose'], .2)


if __name__ == '__main__':
    unittest.main()
