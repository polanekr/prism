"""Launcher environment and optional-feature installation checks."""
from qt_helpers import dispose_widget
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidgetItem

from bio_dosimetry import BioDosimetryTab
from prism_app import qt_environment, prepare_runtime, main
from sample_statistics import calculate_sample_statistics


class LauncherTests(unittest.TestCase):
    def test_linux_uses_own_qt_and_preserves_other_libraries(self):
        root = Path('/tmp/PRISM with spaces/Qt5')
        environment, _ = qt_environment(
            {'LD_LIBRARY_PATH': '/other/lib', 'QT_QPA_PLATFORM': 'offscreen'}, system='Linux', qt_root=root)
        self.assertEqual(environment['LD_LIBRARY_PATH'], f'{root / "lib"}{os.pathsep}/other/lib')
        self.assertEqual(environment['QT_QPA_PLATFORM'], 'offscreen')
        self.assertEqual(environment['PYQTGRAPH_QT_LIB'], 'PyQt5')
        self.assertEqual(environment['QT_PLUGIN_PATH'], str(root / 'plugins'))
        again, _ = qt_environment(environment, system='Linux', qt_root=root)
        self.assertEqual(environment, again)

    def test_windows_does_not_force_linux_platform_or_library_paths(self):
        environment, _ = qt_environment({}, system='Windows', qt_root=Path('Qt5'))
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertNotIn('QT_QPA_PLATFORM', environment)
        self.assertEqual(environment['PYQTGRAPH_QT_LIB'], 'PyQt5')
        self.assertEqual(environment['QT_QPA_PLATFORM_PLUGIN_PATH'], str(Path('Qt5/plugins/platforms')))

    def test_linux_reexec_keeps_arguments_and_offscreen_diagnostic(self):
        class Reexecuted(Exception):
            pass
        env = {'LD_LIBRARY_PATH': '/bundled/Qt5/lib', 'PYQTGRAPH_QT_LIB': 'PyQt5'}
        with patch('prism_app.qt_environment', return_value=(env, Path('/bundled/Qt5'))), \
                patch.dict(os.environ, {}, clear=True), patch('prism_app.sys.platform', 'linux'), \
                patch('prism_app.os.execvpe', side_effect=Reexecuted) as execute, \
                self.assertRaises(Reexecuted):
            prepare_runtime(['--check', '--require-excel'], check=True)
        self.assertEqual(execute.call_args.args[1], [sys.executable, '-m', 'prism_app', '--check', '--require-excel'])
        self.assertEqual(execute.call_args.args[2]['QT_QPA_PLATFORM'], 'offscreen')

    def test_diagnostic_starts_in_a_new_interpreter(self):
        process = subprocess.run([sys.executable, '-m', 'prism_app', '--check'],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        report = json.loads(process.stdout)
        self.assertTrue(report['ok'])
        self.assertEqual(report['qt']['platform_plugin'], 'offscreen')
        self.assertIn('PyQt5', report['core'])
        self.assertNotIn('PyQt6', report['core'])

    def test_windows_scanner_reports_bit_depth_limit_without_opening_ui(self):
        with patch('prism_app.prepare_runtime'), patch('prism_app.sys.platform', 'win32'), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main(['--scanner'])
        self.assertEqual(result, 1)
        self.assertIn('16 bits per RGB channel', errors.getvalue())


class OptionalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = BioDosimetryTab()
        self.addCleanup(dispose_widget, self.tab)
        data = calculate_sample_statistics([0, 1, 2], [.1, .2, .3], [1, 2, 1])
        data['z_weight_back'] = .5
        self.tab.results_data[3] = data
        self.tab.table.setRowCount(1)
        item = QTableWidgetItem('Sample #3')
        item.setData(Qt.UserRole, 3)
        self.tab.table.setItem(0, 0, item)

    def test_missing_excel_explains_installation_and_creates_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.xlsx'
            with patch.dict(sys.modules, {'openpyxl': None}), \
                    patch('bio_dosimetry.QFileDialog.getSaveFileName', return_value=(str(path), 'Excel Files (*.xlsx)')), \
                    patch('bio_dosimetry.QMessageBox.warning') as warning:
                self.tab.export_sample_table()
            warning.assert_called_once()
            self.assertIn('--extra excel', warning.call_args.args[2])
            self.assertFalse(path.exists())

    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'Optional Excel extra is not installed.')
    def test_excel_export_roundtrips_numeric_values(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.xlsx'
            with patch('bio_dosimetry.QFileDialog.getSaveFileName', return_value=(str(path), 'Excel Files (*.xlsx)')), \
                    patch('bio_dosimetry.QMessageBox.information'), \
                    patch('bio_dosimetry.QMessageBox.critical') as error:
                self.tab.export_sample_table()
            error.assert_not_called()
            workbook = load_workbook(path, read_only=True)
            try:
                rows = list(workbook.active.values)
            finally:
                workbook.close()
            self.assertEqual(rows[1][0], 'Sample #3')
            self.assertEqual(rows[1][1], self.tab.results_data[3]['mean_dose'])
            self.assertEqual(rows[1][4], 0)

    def test_mcmc_menu_is_disabled_when_optional_support_is_unavailable(self):
        from prism_gui import PrismDosimetryGUI
        streams = sys.stdout, sys.stderr
        window = None
        try:
            with patch('prism_gui.BAYES_AVAILABLE', False):
                window = PrismDosimetryGUI()
            self.assertFalse(window.combo_calib_method.model().item(1).isEnabled())
            self.assertIn('--extra bayes', window.combo_calib_method.toolTip())
        finally:
            sys.stdout, sys.stderr = streams
            if window is not None:
                window.close()


class BayesianInstallationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pymc') and importlib.util.find_spec('h5netcdf'),
                         'Optional Bayesian extra is not installed.')
    def test_sampling_and_trace_export_smoke(self):
        import pymc as pm
        import arviz as az
        with tempfile.TemporaryDirectory() as folder, pm.Model():
            pm.Normal('test_parameter', mu=0, sigma=1)
            trace = pm.sample(draws=8, tune=8, chains=1, cores=1, random_seed=42,
                              progressbar=False, compute_convergence_checks=False)
            path = Path(folder) / 'trace.nc'
            trace.to_netcdf(path)
            loaded = az.from_netcdf(path)
            try:
                self.assertIn('test_parameter', loaded.posterior)
                self.assertEqual(loaded.posterior.sizes['draw'], 8)
            finally:
                loaded.close()


if __name__ == '__main__':
    unittest.main()
