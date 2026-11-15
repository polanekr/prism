"""Background lifecycle, Qt/core equivalence and numerical regression contracts."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget
from scipy.ndimage import map_coordinates

from background_tasks import TaskController, running_jobs
from bio_analysis import SampleRegion, sample_coordinates
from morphology import BeamMorphology, analyze_morphology
from processing_settings import ProcessingSettings, MorphologySettings
from qt_helpers import wait_for_tasks


class TaskLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def test_work_is_off_main_thread_and_event_loop_remains_responsive(self):
        controller = TaskController(); started = threading.Event(); release = threading.Event()
        result = []; ticks = []
        timer = QTimer(); timer.setInterval(5); timer.timeout.connect(lambda: ticks.append(1)); timer.start()
        def calculate(check, progress):
            started.set()
            while not release.wait(.005): check()
            return QThread.currentThread() is self.app.thread()
        controller.ready.connect(result.append)
        controller.submit(calculate)
        self.assertTrue(started.wait(2))
        QTest.qWait(50)
        self.assertTrue(ticks)
        release.set(); wait_for_tasks(controller); timer.stop()
        self.assertEqual(result, [False])

    def test_replacement_coalesces_pending_jobs_and_discards_stale_result(self):
        controller = TaskController(); started = threading.Event(); release = threading.Event()
        calls = []; results = []
        def slow(check, progress):
            started.set(); release.wait(2)
            return 'old'  # Even a non-cooperative function may not publish after cancellation.
        controller.ready.connect(results.append)
        controller.submit(slow); self.assertTrue(started.wait(2))
        controller.submit(lambda check, progress: calls.append('middle'))
        controller.submit(lambda check, progress: calls.append('new') or 'new')
        release.set(); wait_for_tasks(controller)
        self.assertEqual(calls, ['new']); self.assertEqual(results, ['new'])

    def test_cancel_does_not_block_or_publish_and_owner_can_be_deleted(self):
        owner = QWidget(); controller = TaskController(owner); started = threading.Event()
        results = []; controller.ready.connect(results.append)
        def calculation(check, progress):
            started.set()
            while True:
                check(); time.sleep(.005)
        controller.submit(calculation); self.assertTrue(started.wait(2))
        start = time.monotonic(); controller.cancel()
        self.assertLess(time.monotonic()-start, .2)
        owner.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        deadline = time.monotonic()+3
        while running_jobs() and time.monotonic() < deadline: QTest.qWait(5)
        QCoreApplication.processEvents()
        self.assertFalse(running_jobs()); self.assertEqual(results, [])

    def test_failure_is_reported_without_publishing_partial_results(self):
        controller = TaskController(); errors = []; results = []
        controller.failed.connect(errors.append); controller.ready.connect(results.append)
        def fail(check, progress): raise ValueError('expected failure')
        controller.submit(fail); wait_for_tasks(controller)
        self.assertEqual(len(errors), 1); self.assertIn('expected failure', errors[0]); self.assertEqual(results, [])


class AnalysisContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def test_core_modules_import_without_qt(self):
        result = subprocess.run([sys.executable, '-c',
            'import dose_solvers, bio_analysis, morphology, processing_settings, sys; '
            'assert not any(name.startswith(("PyQt", "pyqtgraph")) for name in sys.modules)'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_engine_import_remains_available_without_optional_bayesian_backend(self):
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('pymc', 'arviz'): raise ImportError('optional dependency absent')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dosimetry import GafchromicEngine, BAYES_AVAILABLE
assert BAYES_AVAILABLE is False
assert callable(GafchromicEngine.process_films)
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_version_cache_does_not_cache_sources_or_share_mutable_dicts(self):
        from project_state import runtime_record, _package_versions
        _package_versions.cache_clear()
        try:
            with patch('project_state.metadata.version', return_value='test-version') as version:
                first = runtime_record(); calls = version.call_count
                first['packages']['numpy'] = 'modified'
                second = runtime_record()
                self.assertGreater(calls, 0); self.assertEqual(version.call_count, calls)
                self.assertEqual(second['packages']['numpy'], 'test-version')
                self.assertIsNot(first['sources'], second['sources'])
        finally:
            _package_versions.cache_clear()

    def test_settings_validate_and_round_trip_saved_processing_contract(self):
        original = ProcessingSettings(method='hybrid', blank_rgb=[60000, 56000, 52000], median_filtering=False)
        restored = ProcessingSettings.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(original, restored); self.assertEqual(restored.method, 'fast')
        for kwargs in ({'method': 'unknown'}, {'blank_rgb': [0, 1, 2]}, {'median_filtering': 1}, {'random_seed': -1}):
            with self.assertRaises(ValueError): ProcessingSettings(**kwargs)

    def test_affine_snapshot_matches_qt_extraction_for_rotated_polygon_and_ellipse(self):
        data = np.arange(25*30, dtype=float).reshape(25, 30)/100
        image = pg.ImageItem(data)
        for roi in (pg.EllipseROI([3.2, 4.3], [9.1, 6.4], angle=17),
                    pg.PolyLineROI([[3.3, 4.2], [13.2, 5.7], [8.5, 12.1]], closed=True)):
            polygon = isinstance(roi, pg.PolyLineROI)
            shape, vectors, origin = roi.getAffineSliceParams(data, image, fromBoundingRect=polygon)
            region = SampleRegion(1, shape, vectors, origin, None, None)
            expected = pg.ROI.getArrayRegion(roi, data, image, fromBoundingRect=polygon)
            actual = map_coordinates(data, sample_coordinates(region), order=1, mode='constant', cval=0., prefilter=False)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)

    def test_background_morphology_roi_matches_original_elliptical_extraction(self):
        from morphology import analyze_morphology_region
        data = np.arange(25*30, dtype=np.float32).reshape(25, 30)/100
        image = pg.ImageItem(data)
        roi = pg.EllipseROI([3.2, 4.3], [9.1, 6.4], angle=17)
        shape, vectors, origin = roi.getAffineSliceParams(data, image)
        region = SampleRegion(1, shape, vectors, origin, None, None)
        with patch('morphology.analyze_morphology') as calculate:
            analyze_morphology_region(data, None, region, .1, MorphologySettings())
            np.testing.assert_allclose(calculate.call_args.args[0], roi.getArrayRegion(data, image), atol=1e-12)

    def test_segmentation_reduction_matches_per_label_reference(self):
        y, x = np.indices((80, 90))
        data = 2 + np.sin(x*.9)**2 + np.cos(y*.8)**2
        metrics = BeamMorphology.compute_advanced_segmentation_and_entropy(data, .1)
        labels = metrics['segmented_map']; count = metrics['num_islands']
        self.assertGreater(count, 2)
        expected = np.mean([np.sum(labels == i)*.1**2 for i in range(1, count+1)])
        self.assertAlmostEqual(metrics['mean_island_area_mm2'], expected, places=13)

    def test_constant_map_has_finite_autocorrelation_and_explicit_undefined_moments(self):
        result = analyze_morphology(np.ones((16, 16)), None, .1, MorphologySettings())
        self.assertTrue(np.all(np.isfinite(result['autocorr'])))
        self.assertIsNone(result['metrics']['correlation_length_mm'])
        self.assertIsNone(result['metrics']['skewness']); self.assertIsNone(result['metrics']['kurtosis'])
        json.dumps(result['metrics'], allow_nan=False)

    def test_pipeline_cancellation_stops_before_expensive_radon(self):
        checks = []
        def check():
            checks.append(1)
            if len(checks) >= 3: raise RuntimeError('cancel checkpoint')
        with patch.object(BeamMorphology, 'compute_radon_sinogram') as radon:
            with self.assertRaisesRegex(RuntimeError, 'cancel checkpoint'):
                analyze_morphology(np.ones((16, 16)), None, .1, MorphologySettings(), check)
            radon.assert_not_called()

    def test_morphology_slider_coalesces_requests_and_uses_latest_value(self):
        from morphology_tab import MorphologyTab
        data = np.full((20, 20), 2.)
        image = pg.ImageItem(data); roi = pg.EllipseROI([2, 2], [12, 12])
        window = SimpleNamespace(roi=roi, img_item_dose=image, current_dose_map=data,
                                 current_std_map=None, current_meta={})
        tab = MorphologyTab(window)
        tab.pages.setCurrentIndex(1)
        try:
            with patch.object(tab.tasks, 'submit') as submit:
                for value in (1, 2, 3, 4, 5): tab.slider_sigma.setValue(value)
                QTest.qWait(350)
                self.assertEqual(submit.call_count, 1)
                with patch('morphology_tab.analyze_morphology_region') as calculate:
                    submit.call_args.args[0](lambda: None, lambda message: None)
                    self.assertEqual(calculate.call_args.args[4].scatter_sigma_mm, .5)
            tab.invalidate_analysis()
            self.assertFalse(tab.current_metrics)
        finally:
            tab.invalidate_analysis(); tab.deleteLater()


if __name__ == '__main__': unittest.main()
