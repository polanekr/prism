"""Numerical and headless Qt regressions; run with unittest discovery."""
from qt_helpers import dispose_widget
from qt_helpers import wait_for_tasks
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
import tifffile
import pyqtgraph as pg
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication

from dosimetry import GafchromicEngine
from film_alignment import resample_to_reference
from bio_dosimetry import BioDosimetryTab
from prism_gui import EngineWorker


PARAMS = {'R': [0, 1, 4, 1], 'G': [0, 0.8, 8, 1], 'B': [0, 0.5, 12, 1]}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = GafchromicEngine(str(self.root / 'results'))
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def load(self, payload):
        path = self.root / 'calibration.json'
        path.write_text(json.dumps(payload))
        self.engine.load_calibration(str(path))

    def films(self):
        folder = self.root / 'films'
        folder.mkdir()
        doses = [0, 1, 2, 4, 8, 12]
        blank = np.array([60000., 56000., 52000.])
        for i, dose in enumerate(doses):
            od = np.array([self.engine.rational_func_od(dose, *PARAMS[ch]) for ch in 'RGB'])
            rgb = np.rint(blank * 10 ** -od).astype(np.uint16)
            tifffile.imwrite(folder / f'{i:02}.tif', np.tile(rgb, (12, 12, 1)), photometric='rgb')
        return folder, doses

    def test_load_replaces_all_optional_state(self):
        self.load({'fitted_params': PARAMS, 'doses_gy': [0, 1],
                   'measured_ods': {ch: [0, .1] for ch in 'RGB'},
                   'calib_cov': {ch: np.eye(4).tolist() for ch in 'RGB'},
                   'calib_samples': {ch: [p, p] for ch, p in PARAMS.items()}})
        self.engine.calib_traces['R'] = object()
        self.load({'fitted_params': PARAMS})
        self.assertEqual(self.engine.calib_samples, {})
        self.assertEqual(self.engine.calib_cov, {})
        self.assertEqual(self.engine.calib_traces, {})
        self.assertIsNone(self.engine.calib_doses)
        self.assertIsNone(self.engine.calib_ods)

    def test_invalid_load_preserves_active_state(self):
        self.load({'fitted_params': PARAMS})
        old = self.engine.calib_params
        old_lut = self.engine.ratio_lut_func
        bad_cases = [
            {'fitted_params': {'R': PARAMS['R']}},
            {'fitted_params': {**PARAMS, 'B': [0, -1, 2, 1]}},
            {'fitted_params': PARAMS, 'calib_samples': {'B': []}},
            {'fitted_params': PARAMS, 'calib_cov': {'B': [[1]]}},
            {'fitted_params': PARAMS, 'calib_cov': {'R': (-np.eye(4)).tolist()}},
            {'fitted_params': PARAMS, 'calib_sigma_od': {'G': float('nan')}},
            {'fitted_params': PARAMS, 'doses_gy': [0, float('inf')]},
            {'fitted_params': PARAMS, 'doses_gy': [0, 1],
             'measured_ods': {ch: [0] for ch in 'RGB'}},
        ]
        for payload in bad_cases:
            with self.subTest(payload=payload), self.assertRaises((ValueError, KeyError)):
                self.load(payload)
            self.assertIs(self.engine.calib_params, old)
            self.assertIs(self.engine.ratio_lut_func, old_lut)

    def test_invalid_inputs_never_write_or_replace_calibration(self):
        folder, doses = self.films()
        self.load({'fitted_params': PARAMS})
        old = self.engine.calib_params
        invalid = [doses[:-1], [0, 1, 2, 3, 4, float('nan')],
                   [0, 1, 2, 3, 4, -1], [1, 2, 3, 4, 5, 6],
                   [0, 1, 4, 2, 8, 12], [0, 0, 1, 1, 2, 2]]
        for values in invalid:
            with self.subTest(doses=values), self.assertRaises(ValueError):
                self.engine.run_calibration(str(folder), values)
            self.assertIs(self.engine.calib_params, old)
            self.assertFalse((Path(self.engine.output_folder) / 'calibration_master.json').exists())

    def test_later_channel_failure_preserves_state_and_master(self):
        folder, doses = self.films()
        self.load({'fitted_params': PARAMS})
        old = self.engine.calib_params
        master = Path(self.engine.output_folder) / 'calibration_master.json'
        master.write_text('previous master')
        with patch('calibration.curve_fit', side_effect=[
            (np.array([0, 1, 4]), np.eye(3) * 1e-6), RuntimeError('fit failed')
        ]), self.assertRaisesRegex(ValueError, 'channel G'):
            self.engine.run_calibration(str(folder), doses)
        self.assertIs(self.engine.calib_params, old)
        self.assertEqual(master.read_text(), 'previous master')

    def test_nonfinite_covariance_is_rejected(self):
        folder, doses = self.films()
        with patch('calibration.curve_fit', return_value=(np.array([0, 1, 4]), np.full((3, 3), np.inf))), \
                self.assertRaisesRegex(ValueError, 'not finite'):
            self.engine.run_calibration(str(folder), doses)
        self.assertEqual(self.engine.calib_params, {})

    def test_real_calibration_recovers_each_rgb_curve(self):
        folder, doses = self.films()
        self.engine.run_calibration(str(folder), doses, roi_size=4, n_posterior_samples=10)
        for ch in 'RGB':
            recovered = self.engine.inv_rational_func_dose(
                np.array(self.engine.calib_ods[ch]), self.engine.calib_params[ch])
            np.testing.assert_allclose(recovered, doses, atol=0.02)
        master = Path(self.engine.output_folder) / 'calibration_master.json'
        other = GafchromicEngine(str(self.root / 'other'))
        other.load_calibration(str(master))
        np.testing.assert_allclose(other.calib_params['G'], self.engine.calib_params['G'])

    def test_atomic_metadata_write_preserves_previous_file(self):
        master = Path(self.engine.output_folder) / 'calibration_master.json'
        master.write_text('previous master')
        with self.assertRaises(ValueError):
            self.engine._save_calibration_metadata({'invalid': float('nan')})
        self.assertEqual(master.read_text(), 'previous master')
        self.assertEqual(list(master.parent.glob('*.tmp')), [])

    def test_calibration_files_exclude_noise_and_support_tiff_suffixes(self):
        for name in ('00.TIF', '01.tiff', '02.TIFF', '03.tif', '00_scan_std.TIF', 'notes.txt'):
            (self.root / name).touch()
        files, _ = self.engine._calibration_inputs(self.root, [0, 1, 2, 3])
        self.assertEqual([f.name for f in files], ['00.TIF', '01.tiff', '02.TIFF', '03.tif'])
        self.assertEqual(self.engine._scan_std_path(self.root / '00.TIF'), str(self.root / '00_scan_std.TIF'))

    def test_processing_preserves_channel_intensities_and_noise(self):
        raw = np.tile([100., 1000., 10.], (9, 9, 1))
        noise = np.tile([1., 100., .01], (9, 9, 1))
        blank = raw * 2
        self.engine.calib_params = PARAMS
        captured = {}
        def record_od(image, reference, *args):
            np.testing.assert_array_equal(image, raw)
            np.testing.assert_array_equal(reference, blank)
            return np.log10(reference / image)
        def solver(od, od_noise_map=None):
            captured['noise'] = od_noise_map
            return np.ones((9, 9)), np.full((9, 9), .1)
        with patch.object(self.engine, '_load_tiff_with_meta', side_effect=[(blank, {}), (raw, {})]), \
                patch.object(self.engine, '_calculate_od_map', side_effect=record_od), \
                patch.object(self.engine, '_solver_fast', side_effect=solver), \
                patch.object(self.engine, '_compute_calibration_std_map', return_value=np.zeros((9, 9))), \
                patch('film_processing.load_scan_noise', side_effect=[(noise, 'mean_sem'), (noise, 'mean_sem')]):
            self.engine.process_films(['film.tif'], 'blank.tif', method='fast')
        expected = np.sqrt((noise / (np.log(10) * raw))**2 + (noise / (np.log(10) * blank))**2)
        np.testing.assert_allclose(captured['noise'], expected)

    def test_mcmc_failure_preserves_active_state(self):
        self.load({'fitted_params': PARAMS})
        old = self.engine.calib_params
        def fail(staged, *args):
            staged.calib_params['R'] = np.array([0, 2, 3, 1])
            raise RuntimeError('sampling failed')
        with patch.object(GafchromicEngine, '_run_mcmc_calibration', fail), self.assertRaises(RuntimeError):
            self.engine.run_mcmc_calibration('unused', [0, 1, 2, 3])
        self.assertIs(self.engine.calib_params, old)


class AlignmentTests(unittest.TestCase):
    def test_identity_with_larger_reference_has_invalid_border(self):
        data = np.full((4, 6), 5.)
        aligned, valid = resample_to_reference(data, np.eye(3), (8, 10))
        self.assertEqual(aligned.shape, (8, 10))
        np.testing.assert_array_equal(aligned[:4, :6], data)
        self.assertEqual(valid.sum(), data.size)
        self.assertEqual(aligned[~valid].sum(), 0)

    def test_translation_moves_marker_and_does_not_extend_edges(self):
        data = np.zeros((7, 9)); data[2, 3] = 10
        aligned, valid = resample_to_reference(data, [[1, 0, 2], [0, 1, 1], [0, 0, 1]], (10, 14))
        self.assertEqual(np.unravel_index(aligned.argmax(), aligned.shape), (3, 5))
        self.assertFalse(valid[0].any())

    def test_fractional_translation_interpolates(self):
        data = np.tile(np.arange(8.), (7, 1))
        aligned, valid = resample_to_reference(data, [[1, 0, .5], [0, 1, 0], [0, 0, 1]], data.shape)
        self.assertAlmostEqual(aligned[3, 4], 3.5)
        self.assertTrue(valid[3, 4])


class QtIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = BioDosimetryTab()
        self.addCleanup(dispose_widget, self.tab)

    def test_numerical_transform_matches_display_with_rotation_and_shift(self):
        data = np.zeros((5, 7)); data[1, 4] = 10
        self.tab.front_dose = data
        self.tab.img_front.setImage(data)
        self.tab.img_back.setImage(np.zeros((9, 11)))
        self.tab.img_front.setTransformOriginPoint(3, 2)
        self.tab.sl_rot.setValue(300)
        self.tab.sl_shift_x.setValue(2)
        self.tab.sl_shift_y.setValue(1)
        matrix = self.tab._front_to_back_transform()
        for x, y in [(0, 0), (4, 1), (6, 4)]:
            displayed = self.tab.img_front.mapToItem(self.tab.img_back, QPointF(x, y))
            mapped = matrix @ [x, y, 1]
            np.testing.assert_allclose(mapped[:2], [displayed.x(), displayed.y()], atol=1e-12)
        # An exact 90-degree case also checks the direction of numerical resampling.
        self.tab.img_front.setRotation(90)
        aligned, _ = resample_to_reference(data, self.tab._front_to_back_transform(), (9, 11))
        self.assertEqual(np.unravel_index(aligned.argmax(), aligned.shape), (4, 6))

    def test_changing_alignment_invalidates_previous_results(self):
        self.tab.front_dose = np.ones((5, 5))
        self.tab.results_data[1] = {'mean_dose': 5}
        self.tab.table.setRowCount(1)
        self.tab.sl_shift_x.setValue(1)
        self.assertFalse(self.tab.results_data)
        self.assertEqual(self.tab.table.rowCount(), 0)

    def test_worker_restores_output_folder_when_calibration_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = GafchromicEngine(folder)
            worker = EngineWorker('calibrate', engine, folder='other', doses=[0, 1])
            errors = []
            worker.error.connect(errors.append)
            with patch.object(engine, 'run_calibration', side_effect=ValueError('invalid input')), \
                    contextlib.redirect_stdout(io.StringIO()):
                worker.run()
            self.assertEqual(engine.output_folder, folder)
            self.assertEqual(len(errors), 1)

    def test_invalid_film_load_preserves_existing_map(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'film_dose.npy'
            original = np.ones((5, 5))
            self.tab.front_dose = original
            self.tab.back_spacing = .0846
            bad_cases = [
                (np.zeros((5, 5, 3)), None, None),
                (np.full((5, 5), np.nan), None, None),
                (np.ones((5, 5)), {'pixel_spacing_mm': .2}, None),
                (np.ones((5, 5)), {'pixel_spacing_mm': .0846}, np.ones((2, 2))),
            ]
            for dose, metadata, std in bad_cases:
                np.save(path, dose)
                if metadata is not None:
                    path.with_name('film_meta.json').write_text(json.dumps(metadata))
                if std is not None:
                    np.save(path.with_name('film_std.npy'), std)
                with patch('bio_dosimetry.QFileDialog.getOpenFileName', return_value=(str(path), '')), \
                        patch('bio_dosimetry.QMessageBox.critical') as error:
                    self.tab.load_film('front')
                self.assertIs(self.tab.front_dose, original)
                error.assert_called_once()

    def test_reloading_front_film_keeps_slider_transform(self):
        self.tab.sl_shift_x.setValue(7)
        self.tab.sl_rot.setValue(120)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'film_dose.npy'
            np.save(path, np.ones((8, 12)))
            with patch('bio_dosimetry.QFileDialog.getOpenFileName', return_value=(str(path), '')), \
                    patch('bio_dosimetry.QMessageBox.critical') as error:
                self.tab.load_film('front')
            error.assert_not_called()
        self.assertEqual(self.tab.img_front.pos().x(), 7)
        self.assertEqual(self.tab.img_front.rotation(), 12)
        self.assertEqual(self.tab.img_front.transformOriginPoint(), QPointF(6, 4))

    def prepare_sample(self, roi_pos):
        self.tab.front_dose = np.full((10, 10), 10.)
        self.tab.back_dose = np.full((20, 20), 20.)
        self.tab.img_front.setImage(self.tab.front_dose)
        self.tab.img_back.setImage(self.tab.back_dose)
        self.tab.geometry.d_air_front = self.tab.geometry.d_air_back = 0
        roi = pg.EllipseROI(roi_pos, [4, 4])
        self.tab.plot.addItem(roi)
        self.tab.rois.append(roi)

    def test_sample_within_overlap_calculates_on_reference_grid(self):
        self.prepare_sample([2, 2])
        with patch('bio_dosimetry.QMessageBox.information'), \
                patch('bio_dosimetry.QMessageBox.warning') as warning:
            self.tab.calculate_doses()
            wait_for_tasks(self.tab.tasks)
        warning.assert_not_called()
        self.assertAlmostEqual(self.tab.results_data[1]['mean_dose'], np.sqrt(200))

    def test_sample_outside_overlap_is_rejected(self):
        self.prepare_sample([8, 8])
        with patch('bio_dosimetry.QMessageBox.information') as success, \
                patch('bio_dosimetry.QMessageBox.warning') as warning:
            self.tab.calculate_doses()
            wait_for_tasks(self.tab.tasks)
        warning.assert_called_once()
        success.assert_not_called()
        self.assertFalse(self.tab.results_data)


if __name__ == '__main__':
    unittest.main()
