"""Known-dose controls isolate spatial anchoring from response-model accuracy."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dosimetry import GafchromicEngine
from processing_settings import ProcessingSettings
from dose_solvers import thickness_anchor_weights
from validation import FORWARD_PARAMETERS


class ThicknessAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = GafchromicEngine(self.temp.name)
        self.engine.calib_params = {ch: np.array(p) for ch, p in FORWARD_PARAMETERS.items()}
        self.engine.calib_doses = np.array([0., 1., 10., 20.])

    def od(self, dose):
        return np.stack([self.engine.rational_func_od(dose, *self.engine.calib_params[ch]) for ch in 'RGB'], axis=-1)

    def test_out_of_range_edge_cannot_shift_known_eleven_gy_field(self):
        truth = np.full((80, 80), 11.)
        ideal = self.od(truth)
        corrupted = ideal.copy()
        corrupted[:5] = [2., 1.8, 1.6]
        corrupted[-5:] = [2., 1.8, 1.6]
        corrupted[:, :5] = [2., 1.8, 1.6]
        for method in ('mayer', 'admm'):
            with self.subTest(method=method):
                clean, _ = self.engine._solve_dose(ideal, method, dose_only=True)
                diagnostics = {}
                actual, _ = self.engine._solve_dose(corrupted, method, dose_only=True, diagnostics=diagnostics)
                np.testing.assert_allclose(clean, 11., atol=1e-10)
                np.testing.assert_allclose(actual[20:60, 20:60], 11., atol=.005)
                self.assertGreater(diagnostics['anchor_pixels'], 0)
                self.assertGreater(diagnostics['excluded_pixels'], 0)
                self.assertEqual(diagnostics['thickness_at_limit_fraction'], 0.)

    def test_explicit_film_mask_excludes_plausible_but_wrong_edge_channels(self):
        ideal = self.od(np.full((80, 80), 11.))
        corrupted = ideal.copy()
        corrupted[:12] = self.od(np.full((12, 80), 18.))*np.array([1., .95, 1.])
        valid = self.engine._anchor_valid_mask(corrupted, (.2, .2, .8, .8))
        for method in ('mayer', 'admm'):
            with self.subTest(method=method):
                actual, _ = self.engine._solve_dose(corrupted, method, anchor_mask=valid, dose_only=True)
                np.testing.assert_allclose(actual[25:55, 25:55], 11., atol=.005)
                # Masking eligibility does not replace excluded doses by zero.
                self.assertGreater(actual[3, 40], 11.)

    def test_no_calibration_supported_anchor_is_explicit_and_does_not_clip_dose(self):
        od = self.od(np.full((12, 12), 80.))
        for method in ('mayer', 'admm'):
            diagnostics = {}
            actual, _ = self.engine._solve_dose(od, method, dose_only=True, diagnostics=diagnostics)
            np.testing.assert_allclose(actual, 80., atol=1e-9)
            self.assertEqual(diagnostics['anchor_status'], 'unavailable')
            self.assertEqual(diagnostics['valid_pixels'], 0)

    def test_single_in_range_outlier_cannot_select_the_anchor_core(self):
        dose = np.full((80, 80), 6.)
        dose[0, 0] = 20.
        valid = np.ones(dose.shape, bool)
        weights = thickness_anchor_weights(dose, valid)
        self.assertGreater(np.count_nonzero(weights), 6000)
        self.assertLessEqual(weights[0, 0], weights[40, 40])

    def test_same_fixed_mask_is_used_for_serial_and_parallel_parameter_draws(self):
        od = self.od(np.full((24, 24), 11.))
        od[:3] = [2., 1.8, 1.6]
        mask = self.engine._anchor_valid_mask(od, (.15, .15, .85, .85))
        noise = np.full_like(od, .002)
        for ch in 'RGB':
            samples = np.tile(self.engine.calib_params[ch], (3, 1))
            samples[:, 1] *= [.99, 1., 1.01]
            self.engine.calib_samples[ch] = samples
        for method in ('mayer', 'admm'):
            dose, _ = self.engine._solve_dose(od, method, noise, dose_only=True, anchor_mask=mask)
            serial = self.engine._compute_calibration_std_map(od, dose, method, noise,
                median_filtering=False, workers=1, anchor_mask=mask)
            parallel = self.engine._compute_calibration_std_map(od, dose, method, noise,
                median_filtering=False, workers=3, anchor_mask=mask)
            candidates = [self.engine._solve_dose(od, method, noise,
                {ch: self.engine.calib_samples[ch][i] for ch in 'RGB'}, dose_only=True, anchor_mask=mask)[0]
                for i in range(3)]
            np.testing.assert_array_equal(serial, parallel)
            np.testing.assert_allclose(serial, np.std(candidates, axis=0, ddof=1), atol=1e-10)

    def test_processing_and_replay_keep_explicit_bounds_and_diagnostics(self):
        import contextlib
        import io
        import json
        import tifffile
        from project_state import replay_run
        root = Path(self.temp.name)
        od = self.od(np.full((24, 24), 11.))
        raw = root/'film.tif'
        tifffile.imwrite(raw, np.rint(60000*10**(-od)).astype(np.uint16), photometric='rgb')
        for ch in 'RGB':
            self.engine.calib_samples[ch] = np.tile(self.engine.calib_params[ch], (2, 1))
        settings = ProcessingSettings(method='admm', blank_rgb=(60000,)*3,
            anchor_bounds=(.25, .25, .75, .75))
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine.process_films([str(raw)], '', settings=settings)
        meta = json.loads((root/'film_meta.json').read_text())
        self.assertEqual(meta['anchor_bounds_normalized'], [.25, .25, .75, .75])
        self.assertEqual(meta['solver_diagnostics']['valid_pixels'], 144)
        self.assertEqual(meta['solver_diagnostics']['anchor_status'], 'available')
        manifest = next((root/'.runs').glob('*/run.json'))
        self.assertEqual(json.loads(manifest.read_text())['settings']['anchor_bounds'], [.25, .25, .75, .75])
        with contextlib.redirect_stdout(io.StringIO()): replay_run(manifest, root/'replayed')
        np.testing.assert_array_equal(np.load(root/'film_dose.npy'), np.load(root/'replayed/film_dose.npy'))
        np.testing.assert_array_equal(np.load(root/'film_std.npy'), np.load(root/'replayed/film_std.npy'))

    def test_bounds_round_trip_and_bad_masks_are_rejected(self):
        settings = ProcessingSettings(anchor_bounds=[.1, .2, .8, .9])
        self.assertEqual(ProcessingSettings.from_dict(settings.to_dict()), settings)
        for bad in ((.1, .2, .1, .9), (-.1, 0, 1, 1), (0, 0, np.nan, 1), (0, 1)):
            with self.assertRaises(ValueError): ProcessingSettings(anchor_bounds=bad)
        od = self.od(np.full((5, 5), 11.))
        for bad in (np.ones((5, 5)), np.ones((4, 5), bool)):
            with self.assertRaisesRegex(ValueError, 'Anchor mask'):
                self.engine._solve_dose(od, 'admm', anchor_mask=bad)
