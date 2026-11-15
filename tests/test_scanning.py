"""Hardware-free acquisition, cancellation and non-overwriting file tests."""
import contextlib
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
import tifffile
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QListWidget, QComboBox

from scan import ScanCancelled, ScanWorker, FilmScannerDialog
from scan_data import (ScanAccumulator, scan_paths, is_raw_film, next_scan_path,
                       reserve_scan, publish_scan_files, load_scan_noise)
from dosimetry import GafchromicEngine
from prism_gui import PrismDosimetryGUI


class ScanStatisticsTests(unittest.TestCase):
    def test_online_statistics_match_sample_sd_and_sem(self):
        images = np.array([np.full((3, 4, 3), n, dtype=np.uint16) for n in (100, 104, 108)])
        accumulator = ScanAccumulator()
        for image in images:
            accumulator.add(image)
        mean, std, sem = accumulator.finish()
        np.testing.assert_array_equal(mean, np.mean(images, axis=0))
        np.testing.assert_allclose(std, np.std(images, axis=0, ddof=1))
        np.testing.assert_allclose(sem, np.std(images, axis=0, ddof=1) / np.sqrt(3), rtol=1e-7)

    def test_single_scan_has_no_invented_noise(self):
        accumulator = ScanAccumulator()
        image = np.full((3, 4, 3), 120, dtype=np.uint16)
        accumulator.add(image)
        mean, std, sem = accumulator.finish()
        np.testing.assert_array_equal(mean, image)
        self.assertIsNone(std)
        self.assertIsNone(sem)

    def test_invalid_or_mismatched_scans_fail(self):
        accumulator = ScanAccumulator()
        with self.assertRaises(ValueError):
            accumulator.finish()
        with self.assertRaises(ValueError):
            accumulator.add(np.ones((3, 4, 3), dtype=np.uint8))
        accumulator.add(np.ones((3, 4, 3), dtype=np.uint16))
        with self.assertRaises(ValueError):
            accumulator.add(np.ones((3, 5, 3), dtype=np.uint16))


class ScanFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'film_001.tif'
        self.paths = scan_paths(self.path)

    def test_index_skips_existing_mean_sidecars_metadata_and_locks(self):
        for index, key in enumerate(('mean', 'std', 'sem', 'meta', 'lock'), start=1):
            scan_paths(self.root / f'film_{index:03d}.tif')[key].touch()
        path, index = next_scan_path(self.root, 'film', 1)
        self.assertEqual(index, 6)
        self.assertEqual(path.name, 'film_006.tif')
        with self.assertRaises(ValueError):
            next_scan_path(self.root, 'film', 1, max_index=5)
        for prefix in ('../film', 'x/y', 'x\\y', 'x:y'):
            with self.assertRaises(ValueError):
                next_scan_path(self.root, prefix, 1)

    def test_reservation_never_removes_another_reservation(self):
        with reserve_scan(self.path):
            with self.assertRaises(FileExistsError):
                with reserve_scan(self.path):
                    self.fail('A second reservation was accepted.')
            self.assertTrue(self.paths['lock'].exists())
        self.assertFalse(self.paths['lock'].exists())

    def test_preexisting_output_is_preserved(self):
        self.path.write_bytes(b'original measurement')
        with self.assertRaises(FileExistsError):
            with reserve_scan(self.path):
                self.fail('An existing measurement was accepted.')
        self.assertEqual(self.path.read_bytes(), b'original measurement')
        self.assertFalse(self.paths['lock'].exists())

    def test_racing_writer_is_preserved_and_our_partial_files_rolled_back(self):
        source = self.root / 'staged'
        source.write_bytes(b'new measurement')
        with self.assertRaises(FileExistsError):
            with reserve_scan(self.path):
                self.path.write_bytes(b'racing writer')
                publish_scan_files([(source, self.paths['std']), (source, self.path)])
        self.assertEqual(self.path.read_bytes(), b'racing writer')
        self.assertFalse(self.paths['std'].exists())
        self.assertFalse(self.paths['lock'].exists())

    def test_disk_error_rolls_back_new_files(self):
        source = self.root / 'staged'
        source.write_bytes(b'new measurement')
        with self.assertRaises(OSError):
            with reserve_scan(self.path), patch('scan_data.os.fsync', side_effect=OSError('disk full')):
                publish_scan_files([(source, self.path)])
        self.assertFalse(self.path.exists())
        self.assertFalse(self.paths['lock'].exists())

    def test_publication_cancel_cleans_only_its_own_outputs(self):
        source = self.root / 'staged'
        source.write_bytes(b'new measurement')
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ScanCancelled('cancelled during copy')
        with self.assertRaises(ScanCancelled):
            with reserve_scan(self.path):
                publish_scan_files([(source, self.path)], check)
        self.assertTrue(source.exists())
        self.assertFalse(self.path.exists())
        self.assertFalse(self.paths['lock'].exists())

    def test_auxiliary_tiffs_are_not_raw_films(self):
        self.assertTrue(is_raw_film('film.TIFF'))
        for name in ('film_scan_std.tif', 'film_scan_sem.TIFF', 'film_scan_meta.json'):
            self.assertFalse(is_raw_film(name))

    def test_sem_preferred_and_legacy_std_handling_is_explicit(self):
        noise = np.full((3, 4, 3), 4., dtype=np.float32)
        tifffile.imwrite(self.paths['std'], noise, photometric='rgb')
        with self.assertWarnsRegex(UserWarning, 'Repetition count unavailable'):
            legacy, source = load_scan_noise(self.path, noise.shape)
        np.testing.assert_allclose(legacy, 4)
        self.assertEqual(source, 'legacy_repeat_std_count_unknown')
        self.paths['meta'].write_text(json.dumps({'scan_count': 4}))
        derived, source = load_scan_noise(self.path, noise.shape)
        np.testing.assert_allclose(derived, 2)
        self.assertEqual(source, 'repeat_std_to_sem')
        tifffile.imwrite(self.paths['sem'], np.ones(noise.shape, dtype=np.float32), photometric='rgb')
        preferred, source = load_scan_noise(self.path, noise.shape)
        np.testing.assert_allclose(preferred, 1)
        self.assertEqual(source, 'mean_sem')

    def test_noise_loader_rejects_incomplete_or_inconsistent_acquisitions(self):
        with reserve_scan(self.path), self.assertRaisesRegex(ValueError, 'progress'):
            load_scan_noise(self.path, (3, 4, 3))
        self.paths['meta'].write_text(json.dumps({'scan_count': 2}))
        with self.assertRaisesRegex(ValueError, 'Missing noise'):
            load_scan_noise(self.path, (3, 4, 3))
        tifffile.imwrite(self.paths['sem'], np.full((3, 4, 3), np.nan), photometric='rgb')
        with self.assertRaisesRegex(ValueError, 'Invalid scan noise'):
            load_scan_noise(self.path, (3, 4, 3))
        self.paths['meta'].write_text(json.dumps({'scan_count': 1}))
        with self.assertRaisesRegex(ValueError, 'single-scan'):
            load_scan_noise(self.path, (3, 4, 3))


class ScanWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'film_001.tif'

    def worker(self, count=3):
        return ScanWorker('scan', filename=str(self.path), device_name='test-device',
                          num_scans=count, res=300, profile='Full Bed Area', timeout_s=2)

    def acquisition(self, values):
        values = iter(values)
        def acquire(worker, path, options):
            tifffile.imwrite(path, np.full((3, 4, 3), next(values), dtype=np.uint16), photometric='rgb')
        return acquire

    def test_acquisition_saves_sd_sem_count_and_dpi(self):
        worker = self.worker()
        results = []
        worker.finished_signal.connect(lambda success, value: results.append((success, value)))
        with patch.object(ScanWorker, '_acquire_pass', self.acquisition([100, 104, 108])):
            worker.run()
        self.assertEqual(results, [(True, str(self.path))])
        paths = scan_paths(self.path)
        np.testing.assert_allclose(tifffile.imread(paths['mean']), 104)
        np.testing.assert_allclose(tifffile.imread(paths['std']), 4)
        np.testing.assert_allclose(tifffile.imread(paths['sem']), 4 / np.sqrt(3), rtol=1e-7)
        meta = json.loads(paths['meta'].read_text())
        self.assertEqual(meta['scan_count'], 3)
        self.assertEqual(meta['left_mm'], 0)
        self.assertAlmostEqual(meta['pixel_spacing_mm'], 25.4/300)
        engine = GafchromicEngine(str(self.root / 'processed'))
        _, tiff_meta = engine._load_tiff_with_meta(self.path)
        self.assertAlmostEqual(tiff_meta['pixel_spacing_mm'], 25.4/300)
        self.assertEqual(list(self.root.glob('.prism-scan-*')), [])
        self.assertFalse(paths['lock'].exists())

    def test_single_scan_saves_no_nan_or_bogus_noise_tiff(self):
        worker = self.worker(1)
        with patch.object(ScanWorker, '_acquire_pass', self.acquisition([100])):
            worker._run_scan()
        paths = scan_paths(self.path)
        self.assertTrue(paths['mean'].exists())
        self.assertFalse(paths['std'].exists())
        self.assertFalse(paths['sem'].exists())
        noise, source = load_scan_noise(self.path, (3, 4, 3))
        self.assertIsNone(noise)
        self.assertEqual(source, 'unavailable_single_scan')

    def test_cancel_between_passes_does_not_publish_partial_acquisition(self):
        worker = self.worker()
        def acquire(worker, path, options):
            tifffile.imwrite(path, np.ones((3, 4, 3), dtype=np.uint16), photometric='rgb')
            worker.request_cancel()
        with patch.object(ScanWorker, '_acquire_pass', acquire), self.assertRaises(ScanCancelled):
            worker._run_scan()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_raw_is_rejected_before_hardware_is_called(self):
        self.path.write_bytes(b'previous scan')
        with patch.object(ScanWorker, '_acquire_pass') as acquire, self.assertRaises(FileExistsError):
            self.worker()._run_scan()
        acquire.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'previous scan')

    def test_windows_does_not_request_unsupported_naps2_bit_depth(self):
        worker = self.worker()
        worker.is_windows = True
        with patch.object(worker, '_run_command') as command, \
                self.assertRaisesRegex(RuntimeError, '16 bits per RGB channel'):
            worker._acquire_pass(self.root / 'pass.tif', [])
        command.assert_not_called()

    def test_timeout_terminates_subprocess(self):
        worker = self.worker()
        processes = []
        real_popen = subprocess.Popen
        def start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch('scan.subprocess.Popen', side_effect=start), self.assertRaises(TimeoutError):
            worker._run_command([sys.executable, '-c', 'import time; time.sleep(60)'], timeout_s=.15)
        self.assertIsNotNone(processes[0].poll())

    def test_cancel_terminates_running_subprocess(self):
        worker = self.worker()
        processes = []
        real_popen = subprocess.Popen
        def start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        timer = threading.Timer(.1, worker.request_cancel)
        timer.start()
        self.addCleanup(timer.join)
        with patch('scan.subprocess.Popen', side_effect=start), self.assertRaises(ScanCancelled):
            worker._run_command([sys.executable, '-c', 'import time; time.sleep(60)'])
        self.assertIsNotNone(processes[0].poll())

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Uses Linux process status to check descendant shutdown.')
    def test_timeout_also_stops_descendant_ignoring_sigterm(self):
        worker = self.worker()
        pid_file = self.root / 'child.pid'
        child_code = ('import os, signal, sys, time; '
                      'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                      'open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(60)')
        parent_code = ('import subprocess, sys, time; '
                       'subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]]); time.sleep(60)')
        processes = []
        real_popen = subprocess.Popen
        def start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        def cleanup():
            if processes:
                try:
                    os.killpg(processes[0].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.addCleanup(cleanup)
        with patch('scan.subprocess.Popen', side_effect=start), self.assertRaises(TimeoutError):
            worker._run_command([sys.executable, '-c', parent_code, child_code, str(pid_file)], timeout_s=.5)
        self.assertTrue(pid_file.exists())
        status = Path('/proc') / pid_file.read_text().strip() / 'stat'
        deadline = time.monotonic() + 1
        while status.exists() and status.read_text().split(') ', 1)[1][0] != 'Z' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(not status.exists() or status.read_text().split(') ', 1)[1][0] == 'Z')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Uses Linux process status to check descendant shutdown.')
    def test_failed_parent_does_not_leave_a_scanner_descendant(self):
        worker = self.worker()
        pid_file = self.root / 'failed-parent-child.pid'
        child_code = ('import os, signal, sys, time; '
                      'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                      'open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(60)')
        parent_code = ('import os, subprocess, sys, time\n'
                       'subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])\n'
                       'while not os.path.exists(sys.argv[2]): time.sleep(.01)\n'
                       'sys.exit(2)')
        with self.assertRaises(RuntimeError):
            worker._run_command([sys.executable, '-c', parent_code, child_code, str(pid_file)])
        status = Path('/proc') / pid_file.read_text().strip() / 'stat'
        deadline = time.monotonic() + 1
        while status.exists() and status.read_text().split(') ', 1)[1][0] != 'Z' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(not status.exists() or status.read_text().split(') ', 1)[1][0] == 'Z')

    def test_scanner_failure_and_discovery_timeout_are_reported(self):
        worker = self.worker()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            worker._run_command([sys.executable, '-c', 'import sys; sys.stderr.write("test failure"); sys.exit(2)'])
        finder = ScanWorker('find_scanners')
        results = []
        finder.finished_signal.connect(lambda *args: results.append(args))
        with patch.object(finder, '_run_command', side_effect=TimeoutError('device timeout')):
            finder.run()
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0][0])
        self.assertIn('timeout', results[0][1])

    def test_full_bed_does_not_inherit_stale_crop_offset(self):
        worker = self.worker(1)
        worker.kwargs['left'] = 136
        with patch.object(ScanWorker, '_acquire_pass', self.acquisition([100])):
            worker._run_scan()
        metadata = json.loads(scan_paths(self.path)['meta'].read_text())
        self.assertEqual(metadata['left_mm'], 0)


class ScanDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.object(FilmScannerDialog, 'find_scanners'):
            self.dialog = FilmScannerDialog(self.temp.name)
        self.addCleanup(self.shutdown)
        self.dialog.list_scanners.addItem('test-device')
        self.dialog.list_scanners.setCurrentRow(0)
        self.dialog.chk_date.setChecked(False)

    def shutdown(self):
        if self.dialog.worker is not None:
            worker = self.dialog.worker
            worker.request_cancel()
            worker.wait(3000)
            self.app.processEvents()
        self.dialog.close()

    def wait_idle(self):
        deadline = time.monotonic() + 3
        while self.dialog._busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(self.dialog._busy)

    def test_next_index_is_used_and_existing_index_skipped(self):
        root = Path(self.temp.name)
        (root / 'film_007.tif').write_bytes(b'existing raw')
        self.dialog.spin_next_idx.setValue(7)
        with patch.object(self.dialog, '_start_worker') as start:
            self.dialog.start_scan()
        self.assertEqual(Path(start.call_args.kwargs['filename']).name, 'film_008.tif')
        self.assertEqual((root / 'film_007.tif').read_bytes(), b'existing raw')
        self.dialog.worker = SimpleNamespace(task='scan')
        self.dialog._on_task_result(True, 'saved')
        self.assertEqual(self.dialog.spin_next_idx.value(), 9)
        self.dialog.worker = None
        self.dialog.reset_index()
        self.assertEqual(self.dialog.spin_next_idx.value(), 1)

    def test_failed_scan_can_retry_the_same_index(self):
        self.dialog.spin_next_idx.setValue(8)
        self.dialog._scan_index = 8
        self.dialog.worker = SimpleNamespace(task='scan')
        self.dialog._on_task_result(False, 'failed')
        self.assertEqual(self.dialog.spin_next_idx.value(), 8)
        self.dialog.worker = None

    def test_close_cancels_and_waits_for_thread_exit_without_blocking_ui(self):
        def scanning(worker):
            while not worker._cancelled.wait(.01):
                pass
            worker._check_cancel()
        with patch.object(ScanWorker, '_run_scan', scanning):
            self.dialog.start_scan()
            self.assertTrue(self.dialog._busy)
            self.assertFalse(self.dialog.grp_dir.isEnabled())
            self.assertTrue(self.dialog.btn_cancel.isEnabled())
            event = QCloseEvent()
            self.dialog.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertTrue(self.dialog._close_pending)
            self.wait_idle()
        self.assertIsNone(self.dialog.worker)

    def test_escape_uses_safe_cancellation_path(self):
        def scanning(worker):
            worker._cancelled.wait(2)
            worker._check_cancel()
        with patch.object(ScanWorker, '_run_scan', scanning):
            self.dialog.start_scan()
            self.dialog.reject()
            self.assertTrue(self.dialog._close_pending)
            self.wait_idle()

    def test_gui_lists_and_engine_reject_auxiliary_and_in_progress_files(self):
        root = Path(self.temp.name)
        for name in ('film.tif', 'film_scan_std.tif', 'film_scan_sem.TIFF', 'active.tif'):
            (root / name).touch()
        scan_paths(root / 'active.tif')['lock'].touch()
        processed = root / 'processed'; processed.mkdir()
        fake = SimpleNamespace(project_dir=root, raw_dir=root, processed_dir=processed,
                               list_raw_files=QListWidget(), combo_blank_film=QComboBox(), list_proc_files=QListWidget(),
                               _invalidate_blank_rgb=lambda: None)
        PrismDosimetryGUI.refresh_file_lists(fake)
        self.assertEqual(fake.list_raw_files.count(), 1)
        self.assertEqual(fake.list_raw_files.item(0).text(), 'film.tif')
        self.assertEqual(fake.combo_blank_film.count(), 1)
        engine = GafchromicEngine(str(processed))
        engine.calib_params = {'R': [0, 1, 4, 1]}
        for name in ('film_scan_std.tif', 'film_scan_sem.TIFF', 'active.tif'):
            with self.assertRaises(ValueError):
                engine.process_films([str(root / name)], '', blank_rgb=[100, 100, 100])


if __name__ == '__main__':
    unittest.main()
