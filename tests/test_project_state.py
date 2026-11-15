"""Peak preservation, replay provenance and portable GUI session round trips."""
from qt_helpers import dispose_widget
import contextlib
import io
import json
from unittest.mock import patch
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PyQt5.QtWidgets import QApplication

from dosimetry import GafchromicEngine
from project_state import (atomic_json, begin_run, calibration_snapshot, json_value,
                           replay_run, restore_session, save_session, file_record)
from validation import FORWARD_PARAMETERS, ValidationConfig, write_synthetic_film


class ProcessingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'; self.raw.mkdir()
        self.config = ValidationConfig(image_size=12, repetitions=3, posterior_samples=2, sigma_od=0)
        self.blank = self.raw/'blank.tif'
        write_synthetic_film(self.blank, np.zeros((12, 12)), np.random.default_rng(4), self.config)
        self.film = self.raw/'film.tif'
        self.truth = np.full((12, 12), 2.); self.truth[6, 6] = 20.
        write_synthetic_film(self.film, self.truth, np.random.default_rng(5), self.config)
        self.engine = GafchromicEngine(str(self.root/'processed'))
        self.engine.calib_params = {ch: np.array(values) for ch, values in FORWARD_PARAMETERS.items()}
        self.engine.calib_samples = {ch: np.tile(values, (3, 1)) for ch, values in self.engine.calib_params.items()}
        self.engine.calib_sigma_od = {'R': .0011, 'G': .0012, 'B': .0013}
        self.engine._prepare_ratio_lut()

    def tearDown(self):
        self.temp.cleanup()

    def process(self, median, method='fast'):
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine.process_films([str(self.film)], str(self.blank), method=method, median_filtering=median)
        return np.load(Path(self.engine.output_folder)/'film_dose.npy')

    def test_unfiltered_one_pixel_peak_is_preserved_and_default_filters_remove_it(self):
        for method in ('fast', 'mayer'):
            with self.subTest(method=method):
                filtered = self.process(True, method)
                unfiltered = self.process(False, method)
                self.assertAlmostEqual(filtered.max(), 2., delta=.1)
                self.assertAlmostEqual(unfiltered[6, 6], 20., delta=.1)
                self.assertAlmostEqual(unfiltered[0, 0], 2., delta=.1)
        meta = json.loads((Path(self.engine.output_folder)/'film_meta.json').read_text())
        self.assertFalse(meta['median_filtering'])
        self.assertEqual(meta['fit_uncertainty_after_median'], 'not_applied')

    def test_wider_peak_and_parameter_uncertainty_follow_filter_choice(self):
        self.truth[4:9, 4:9] = 20.
        write_synthetic_film(self.film, self.truth, np.random.default_rng(5), self.config)
        self.assertAlmostEqual(self.process(True).max(), 20., delta=.1)
        od = np.stack([v[1]*self.truth/(self.truth+v[2]) for v in FORWARD_PARAMETERS.values()], axis=-1)
        for ch in 'RGB':
            self.engine.calib_samples[ch][1, 1] *= 1.01
        sigma = self.engine._compute_calibration_std_map(od, self.truth, 'fast', median_filtering=False)
        candidates = [self.engine._solve_dose(od, 'fast', params_dict={ch: self.engine.calib_samples[ch][i] for ch in 'RGB'})[0] for i in range(3)]
        np.testing.assert_allclose(sigma, np.std(candidates, axis=0, ddof=1), atol=1e-10)

    def test_run_snapshot_replays_from_a_moved_project_and_detects_changed_input(self):
        original = self.process(False)
        manifests = list((Path(self.engine.output_folder)/'.runs').glob('*/run.json'))
        report = json.loads(manifests[0].read_text())
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['settings']['median_filtering'], False)
        self.assertEqual(len(report['outputs']), 5)
        self.assertEqual(report['calibration']['calib_sigma_od'], self.engine.calib_sigma_od)
        self.assertTrue(report['environment']['sources'])
        moved = self.root/'moved'; shutil.copytree(self.root/'processed', moved/'processed')
        shutil.copytree(self.raw, moved/'raw')
        copied = moved/'processed'/manifests[0].relative_to(self.root/'processed')
        with contextlib.redirect_stdout(io.StringIO()): replay_run(copied, moved/'replayed')
        np.testing.assert_array_equal(original, np.load(moved/'replayed'/'film_dose.npy'))
        (moved/'raw'/'film_scan_sem.tif').write_bytes(b'changed noise')
        with self.assertRaisesRegex(ValueError, 'changed project input'):
            replay_run(copied, moved/'must-not-exist')
        self.assertFalse((moved/'must-not-exist').exists())

    def test_cleaning_sampling_uses_local_seed_and_worker_forwards_filter_choice(self):
        from unittest.mock import Mock
        from prism_gui import EngineWorker
        worker = EngineWorker('process', Mock(random_seed=42), files=[], blank='', method='fast',
                              cleaning=False, blank_rgb=None, median_filtering=False)
        worker.run()
        self.assertFalse(worker.engine.process_films.call_args.kwargs['median_filtering'])
        samples = []
        fake = Mock()
        fake.means_ = np.array([[0.], [1.]])
        fake.fit.side_effect = lambda values: samples.append(values.copy())
        fake.predict_proba.side_effect = lambda values: np.column_stack([np.ones(len(values)), np.zeros(len(values))])
        od = np.random.default_rng(7).uniform(0, .5, (12, 12, 3))
        with patch('dosimetry.GaussianMixture', return_value=fake):
            self.engine._clean_od_map_biharmonic(od)
            np.random.default_rng(9).normal(size=1000)
            self.engine._clean_od_map_biharmonic(od)
        self.assertEqual(len(samples), 2)
        np.testing.assert_array_equal(samples[0], samples[1])

    def test_cross_drive_paths_fall_back_to_absolute_without_losing_hashes(self):
        with patch('project_state.os.path.relpath', side_effect=ValueError('different drives')):
            record = file_record(self.film, self.root)
        self.assertEqual(record['path'], self.film.resolve().as_posix())
        self.assertEqual(record['status'], 'present')
        self.assertEqual(len(record['sha256']), 64)

    def test_failed_run_is_recorded_and_atomic_json_preserves_previous_state(self):
        self.film.unlink()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError): self.process(False)
        report = json.loads(next((Path(self.engine.output_folder)/'.runs').glob('*/run.json')).read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['outputs'], [])
        path = self.root/'project.json'; atomic_json(path, {'valid': 1})
        with self.assertRaises(ValueError): atomic_json(path, {'invalid': float('nan')})
        self.assertEqual(json.loads(path.read_text()), {'valid': 1})
        self.assertEqual(list(self.root.glob('*.tmp')), [])


class ProjectSessionTests(unittest.TestCase):
    process = ProcessingStateTests.process
    def setUp(self):
        ProcessingStateTests.setUp(self)
        self.critical_patch = patch('PyQt5.QtWidgets.QMessageBox.critical', side_effect=AssertionError('Unexpected GUI error dialog'))
        self.critical_patch.start()
        self.addCleanup(self.critical_patch.stop)
        self.app = QApplication.instance() or QApplication([])
        from prism_gui import PrismDosimetryGUI
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.window = PrismDosimetryGUI()
        sys.stdout, sys.stderr = self.stdout, self.stderr
        window = self.window
        window.project_dir = self.root; window.raw_dir = self.raw
        window.processed_dir = Path(self.engine.output_folder)
        window.analysis_dir = self.root/'analysis'; window.analysis_dir.mkdir()
        window.engine = self.engine
        window.refresh_file_lists()
        window.combo_blank_film.setCurrentText('blank.tif')
        window.global_blank_rgb = [60000., 56000., 52000.]
        window.global_blank_std = [1., 2., 3.]
        self.process(False)
        window.load_and_display_dose(window.processed_dir/'film_dose.npy')
        window.load_and_display_raw(self.film)

    def tearDown(self):
        dispose_widget(self.window)
        self.app.processEvents()
        sys.stdout, sys.stderr = self.stdout, self.stderr
        ProcessingStateTests.tearDown(self)

    def test_project_round_trip_restores_calibration_blank_rois_and_bio_alignment(self):
        window = self.window
        window.chk_median_filter.setChecked(False)
        window.combo_solver.setCurrentIndex(1)
        window.roi.setPos([2, 3]); window.roi.setSize([4, 5]); window.roi.setAngle(12)
        window.bio_tab.load_film('front', window.processed_dir/'film_dose.npy')
        window.bio_tab.load_film('back', window.processed_dir/'film_dose.npy')
        for index in range(window.bio_tab.combo_roi_type.count()):
            window.bio_tab.combo_roi_type.setCurrentIndex(index); window.bio_tab.add_selected_roi()
            window.bio_tab.rois[-1].setAngle(index*7)
        window.bio_tab.combo_dvh_range.setCurrentIndex(1)
        window.bio_tab.sl_shift_x.setValue(3); window.bio_tab.sl_rot.setValue(25)
        window.bio_tab.spins['d_air_front'].setValue(4.)
        states = [json_value(roi.saveState()) for roi in window.bio_tab.rois]
        window.refresh_file_lists()
        self.assertEqual(window.global_blank_rgb, [60000., 56000., 52000.])
        path = save_session(window)
        saved = json.loads(path.read_text())
        window.chk_median_filter.setChecked(True); window.global_blank_rgb = None
        window.roi.setPos([90, 80]); window.bio_tab.clear_all_rois()
        with contextlib.redirect_stdout(io.StringIO()): restore_session(window)
        self.assertFalse(window.chk_median_filter.isChecked())
        self.assertEqual(window.combo_solver.currentIndex(), 1)
        self.assertEqual(window.global_blank_rgb, [60000., 56000., 52000.])
        self.assertEqual(json_value(window.roi.saveState()), saved['roi'])
        self.assertEqual([json_value(roi.saveState()) for roi in window.bio_tab.rois], states)
        self.assertEqual(window.bio_tab.combo_dvh_range.currentIndex(), 1)
        self.assertEqual(window.bio_tab.sl_shift_x.value(), 3)
        self.assertEqual(window.bio_tab.sl_rot.value(), 25)
        self.assertEqual(window.bio_tab.spins['d_air_front'].value(), 4.)
        self.assertEqual(window.bio_tab.results_data, {})
        self.assertEqual(window.engine.calib_sigma_od, self.engine.calib_sigma_od)

    def test_saving_std_view_restores_dose_levels_and_histogram_zoom_choice(self):
        w = self.window
        w.colorbar.setLevels(1., 8.)
        w.combo_cmap.setCurrentText('plasma')
        w.combo_hist_range.setCurrentIndex(0)
        w.btn_toggle_map.click()
        w.colorbar.setLevels(.1, .5)
        path = save_session(w)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['dose_levels'], [1., 8.])
        with contextlib.redirect_stdout(io.StringIO()): restore_session(w)
        self.assertFalse(w.btn_toggle_map.isChecked())
        np.testing.assert_allclose(w.img_item_dose.getLevels(), [1., 8.])
        self.assertEqual(w.combo_cmap.currentText(), 'plasma')
        self.assertEqual(w.combo_hist_range.currentIndex(), 0)

    def test_explicit_anchor_boundary_reaches_worker_and_round_trips_session(self):
        w = self.window
        w.chk_anchor_boundary.setChecked(True)
        w.film_boundary.setPos([2, 3]); w.film_boundary.setSize([7, 6])
        w.combo_solver.setCurrentIndex(0)
        w.list_raw_files.clearSelection()
        for index in range(w.list_raw_files.count()):
            if w.list_raw_files.item(index).text() == 'film.tif':
                w.list_raw_files.item(index).setSelected(True)
        with patch('prism_gui.EngineWorker') as worker:
            w.convert_films()
            bounds = worker.call_args.kwargs['settings'].anchor_bounds
            np.testing.assert_allclose(bounds, [2/12, 3/12, 9/12, 9/12])
            worker.return_value.start.assert_called_once()
        w._set_ui_busy(False)
        save_session(w)
        w.chk_anchor_boundary.setChecked(False)
        w.film_boundary.setPos([0, 0])
        with contextlib.redirect_stdout(io.StringIO()): restore_session(w)
        self.assertTrue(w.chk_anchor_boundary.isChecked())
        np.testing.assert_allclose(w.film_boundary.pos(), [2, 3])
        np.testing.assert_allclose(w.film_boundary.size(), [7, 6])

    def test_explicit_anchor_boundary_cannot_silently_apply_to_another_film(self):
        w = self.window
        w.chk_anchor_boundary.setChecked(True)
        w.list_raw_files.clearSelection()
        for index in range(w.list_raw_files.count()):
            if w.list_raw_files.item(index).text() == 'blank.tif':
                w.list_raw_files.item(index).setSelected(True)
        with patch('prism_gui.EngineWorker') as worker, patch('prism_gui.QMessageBox.warning') as warning:
            w.convert_films()
            worker.assert_not_called()
            warning.assert_called_once()

    def test_viewer_layout_round_trip_and_preflight_rejection(self):
        w = self.window
        w.combo_view_layout.setCurrentIndex(2)
        w.chk_view_raw.setChecked(True)
        w.chk_view_stats.setChecked(False)
        path = save_session(w)
        state = json.loads(path.read_text())
        w.combo_view_layout.setCurrentIndex(1)
        with contextlib.redirect_stdout(io.StringIO()): restore_session(w)
        self.assertEqual(w.combo_view_layout.currentIndex(), 2)
        self.assertTrue(w.chk_view_raw.isChecked())
        self.assertFalse(w.chk_view_stats.isChecked())
        engine = w.engine
        state['viewer_layout']['preferences']['compact']['ratios']['images'] = [1., 1.]
        atomic_json(path, state)
        with self.assertRaisesRegex(ValueError, 'panel proportions'): restore_session(w)
        self.assertIs(w.engine, engine)
        self.assertEqual(w.combo_view_layout.currentIndex(), 2)

    def test_exposure_form_metrics_export_film_binding_and_project_round_trip(self):
        from qt_helpers import wait_for_tasks
        w=self.window;p=w.morphology_tab.measured
        p.inputs['shots'].setText('1000');p.inputs['duration_s'].setText('2')
        p.inputs['particle_pulse_fs'].setText('100');p.inputs['laser_pulse_fs'].setText('30')
        for key,value in {'pulse_plane':'film','pulse_shape':'gaussian_fwhm','pulse_structure':'one_per_event'}.items():
            p.choices[key].setCurrentIndex(p.choices[key].findData(value))
        p.run();wait_for_tasks(p.tasks)
        self.assertIsNotNone(p.result,p.status.text())
        metrics=p.result['metrics']
        self.assertAlmostEqual(metrics['temporal']['dose_per_shot_gy'],metrics['mean_dose_gy']/1000)
        self.assertAlmostEqual(metrics['temporal']['average_rate_gy_s'],metrics['mean_dose_gy']/2)
        self.assertEqual(metrics['roi_pixels'],144)
        exported=self.root/'beam.json'
        with patch('beam_analysis_panel.QFileDialog.getSaveFileName',return_value=(str(exported),'')):
            p.export()
        self.assertEqual(json.loads(exported.read_text())['experiment']['shots'],1000)
        with np.load(exported.with_suffix('.npz')) as arrays:
            np.testing.assert_array_equal(arrays['mask'],p.result['mask'])
        other=w.processed_dir/'other_dose.npy';np.save(other,np.full((12,12),8.))
        w.load_and_display_dose(other)
        self.assertEqual(p.inputs['shots'].text(),'')
        w.load_and_display_dose(w.processed_dir/'film_dose.npy')
        self.assertEqual(p.inputs['shots'].text(),'1000')
        save_session(w)
        p.inputs['shots'].setText('2')
        with contextlib.redirect_stdout(io.StringIO()):restore_session(w)
        p=w.morphology_tab.measured
        self.assertEqual(p.read_experiment().shots,1000)
        self.assertEqual(p.read_experiment().laser_pulse_fs,30.)
        self.assertIsNone(p.result)
        state=json.loads((self.root/'project.json').read_text());engine=w.engine
        state['beam_experiments']['records']['film']['shots']=-1
        atomic_json(self.root/'project.json',state)
        with self.assertRaises(ValueError):restore_session(w)
        self.assertIs(w.engine,engine)

    def test_exposure_edits_invalidate_results_and_unavailable_rate_maps(self):
        from qt_helpers import wait_for_tasks
        p=self.window.morphology_tab.measured
        p.run();wait_for_tasks(p.tasks)
        self.assertIsNotNone(p.result)
        self.assertFalse(p.map_choice.model().item(1).isEnabled())
        self.assertFalse(p.map_choice.model().item(3).isEnabled())
        p.inputs['shots'].setText('10')
        self.assertIsNone(p.result);self.assertFalse(p.export_button.isEnabled())
        p.run();wait_for_tasks(p.tasks)
        self.assertTrue(p.map_choice.model().item(1).isEnabled())
        p.inputs['shots'].setText('0');p.run()
        self.assertIsNone(p.result)
        self.assertIn('positive integer',p.status.text())

    def test_moved_project_restores_and_switching_clears_previous_session(self):
        save_session(self.window)
        with tempfile.TemporaryDirectory() as temporary:
            moved = Path(temporary)/'project'
            shutil.copytree(self.root, moved)
            self.window.project_dir = moved
            self.window.raw_dir = moved/'raw'; self.window.processed_dir = moved/'processed'
            self.window.analysis_dir = moved/'analysis'
            with contextlib.redirect_stdout(io.StringIO()): restore_session(self.window)
            self.assertEqual(self.window.current_dose_path, moved/'processed'/'film_dose.npy')
            np.testing.assert_array_equal(self.window.current_dose_map, np.load(moved/'processed'/'film_dose.npy').astype(np.float32))
            self.window._reset_project_session()
            self.assertIsNone(self.window.engine)
            self.assertIsNone(self.window.global_blank_rgb)
            self.assertIsNone(self.window.current_dose_map)
            self.assertIsNone(self.window.bio_tab.back_dose)

    def test_older_project_controls_restore_with_compatible_defaults(self):
        path = save_session(self.window)
        state = json.loads(path.read_text())
        state.pop('beam_experiments')
        state.pop('viewer_layout')
        state['bio'].pop('dvh_range')
        for name in ('combo_mcmc_adaptation', 'spin_uncertainty_workers', 'combo_hist_normalization', 'combo_dah_band', 'combo_hist_range', 'chk_anchor_boundary'):
            state['controls'].pop(name)
        atomic_json(path, state)
        with contextlib.redirect_stdout(io.StringIO()): restore_session(self.window)
        self.assertEqual(self.window.spin_uncertainty_workers.value(), 1)
        self.assertEqual(self.window.combo_hist_normalization.currentIndex(), 1)
        self.assertEqual(self.window.combo_dah_band.currentIndex(), 1)

    def test_changed_file_and_unknown_schema_leave_active_session_intact(self):
        path = save_session(self.window)
        engine = self.window.engine
        state = json.loads(path.read_text()); state['schema_version'] = 999
        atomic_json(path, state)
        with self.assertRaisesRegex(ValueError, 'Unsupported'): restore_session(self.window)
        self.assertIs(self.window.engine, engine)
        state['schema_version'] = 1; atomic_json(path, state)
        self.film.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed project input'): restore_session(self.window)
        self.assertIs(self.window.engine, engine)


if __name__ == '__main__':
    unittest.main()
