"""Known distributions and UI/export regressions for sample statistics."""
from qt_helpers import dispose_widget
from qt_helpers import wait_for_tasks
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from bio_dosimetry import BioDosimetryTab, EppendorfROI
from sample_statistics import axisymmetric_volume_weights, calculate_sample_statistics


class SampleStatisticsTests(unittest.TestCase):
    def test_known_weighted_distribution_includes_zero_dose(self):
        result = calculate_sample_statistics([0, 1, 2], [.1, .2, .4], [1, 2, 1], bins=3)
        self.assertAlmostEqual(result['mean_dose'], 1)
        self.assertAlmostEqual(result['spread_dose'], np.sqrt(.5))
        self.assertAlmostEqual(result['uncert_dose'], .225)
        self.assertAlmostEqual(result['uncert_pct'], 22.5)
        self.assertEqual(result['min_dose'], 0)
        np.testing.assert_allclose(result['hist_y'], [25, 50, 25])
        np.testing.assert_allclose(result['dose_stats'][1], [.1, .2, .4])
        np.testing.assert_array_equal(result['raw_voxels'], [0, 1, 2])

    def test_high_offset_sample_uses_all_bins_and_preserves_statistics(self):
        doses = np.linspace(10., 12., 1001)
        sigmas = np.full(doses.shape, .2)
        weights = np.linspace(1., 2., doses.size)
        fine = calculate_sample_statistics(doses, sigmas, weights)
        coarse = calculate_sample_statistics(doses, sigmas, weights, bins=10)
        np.testing.assert_allclose(np.diff(fine['hist_edges']), .04)
        self.assertEqual(np.count_nonzero(fine['hist_y']), 50)
        self.assertAlmostEqual(fine['hist_y'].sum(), 100.)
        np.testing.assert_allclose(fine['dose_stats'][1], .2)
        for key in ('mean_dose', 'spread_dose', 'uncert_dose', 'min_dose', 'max_dose'):
            self.assertEqual(fine[key], coarse[key])
        np.testing.assert_array_equal(fine['raw_voxels'], doses)
        nearly_constant = calculate_sample_statistics([11., np.nextafter(11., np.inf)], [.2, .2], [1., 1.])
        self.assertAlmostEqual(nearly_constant['hist_y'].sum(), 100.)
        self.assertTrue(np.all(np.diff(nearly_constant['hist_edges']) > 0))

    def test_all_zero_sample_has_real_volume_and_undefined_percentages(self):
        result = calculate_sample_statistics([0, 0], [0, .2], [1, 3])
        self.assertEqual(result['mean_dose'], 0)
        self.assertEqual(result['spread_dose'], 0)
        self.assertAlmostEqual(result['uncert_dose'], .15)
        self.assertIsNone(result['uncert_pct'])
        self.assertIsNone(result['spread_pct'])
        self.assertAlmostEqual(result['hist_y'].sum(), 100)
        self.assertEqual(result['hist_y'][0], 100)
        self.assertGreaterEqual(result['hist_edges'].min(), 0)
        self.assertTrue(np.isnan(result['dose_stats'][1, 1:]).all())

    def test_small_doses_are_not_thresholded_out(self):
        result = calculate_sample_statistics([0, .0005], [0, .0001], [1, 1])
        self.assertAlmostEqual(result['mean_dose'], .00025)
        np.testing.assert_array_equal(result['raw_voxels'], [0, .0005])
        self.assertAlmostEqual(result['hist_y'].sum(), 100)

    def test_zero_volume_cells_do_not_change_extrema_or_mean(self):
        result = calculate_sample_statistics([1, 2, 100], [.1, .2, 5], [1, 1, 0])
        self.assertEqual(result['max_dose'], 2)
        self.assertEqual(result['mean_dose'], 1.5)
        self.assertEqual(len(result['raw_voxels']), 2)
        self.assertAlmostEqual(result['hist_y'].sum(), 100)

    def test_invalid_data_is_rejected_instead_of_silently_dropped(self):
        cases = [([], [], []), ([1, 2], [1], [1, 1]),
                 ([np.nan], [1], [1]), ([1], [np.inf], [1]),
                 ([-1], [1], [1]), ([1], [-1], [1]),
                 ([1], [1], [-1]), ([1], [1], [0])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                calculate_sample_statistics(*args)

    def test_volume_scale_does_not_change_relative_statistics(self):
        first = calculate_sample_statistics([0, 2, 3], [0, .2, .3], [1, 2, 3])
        scaled = calculate_sample_statistics([0, 2, 3], [0, .2, .3], [10, 20, 30])
        for key in ('mean_dose', 'spread_dose', 'uncert_dose', 'hist_y'):
            np.testing.assert_allclose(first[key], scaled[key])

    def test_tube_weights_follow_contour_and_are_symmetric(self):
        mask = np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool)
        weights = axisymmetric_volume_weights(mask)
        self.assertTrue(np.all(weights[mask] > 0))
        self.assertTrue(np.all(weights[~mask] == 0))
        np.testing.assert_allclose(weights, weights[:, ::-1])
        self.assertGreater(weights[0].sum(), weights[1].sum())
        self.assertGreater(weights[1].sum(), weights[2].sum())


class SampleUIAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        pg.setConfigOptions(imageAxisOrder='row-major')

    def setUp(self):
        self.tab = BioDosimetryTab()
        self.addCleanup(dispose_widget, self.tab)
        self.tab.geometry.d_air_front = self.tab.geometry.d_air_back = 0

    def films(self, dose):
        self.tab.front_dose = np.array(dose, dtype=float)
        self.tab.back_dose = np.array(dose, dtype=float)
        self.tab.img_front.setImage(self.tab.front_dose)
        self.tab.img_back.setImage(self.tab.back_dose)

    def calculate(self):
        with patch('bio_dosimetry.QMessageBox.information') as success, \
                patch('bio_dosimetry.QMessageBox.warning') as warning:
            self.tab.calculate_doses()
            wait_for_tasks(self.tab.tasks)
        warning.assert_not_called()
        success.assert_called_once()

    def ellipse(self, position, size):
        roi = pg.EllipseROI(position, size)
        self.tab.plot.addItem(roi)
        self.tab.rois.append(roi)
        return roi

    def test_high_dose_dvh_zoom_preserves_bins_and_volume_and_full_axis_is_selectable(self):
        self.films(np.tile(np.linspace(10., 12., 24), (24, 1)))
        self.ellipse([5, 5], [10, 10])
        self.calculate()
        self.tab.table.selectRow(0)
        result = self.tab.results_data[1]
        edges = self.tab.curve_dvh.xData.copy()
        volumes = self.tab.curve_dvh.yData.copy()
        self.assertGreater(self.tab.plot_dvh.viewRange()[0][0], 9.)
        self.assertAlmostEqual(volumes.sum(), 100.)
        self.tab.combo_dvh_range.setCurrentIndex(1)
        self.assertLessEqual(self.tab.plot_dvh.viewRange()[0][0], 0.)
        np.testing.assert_array_equal(self.tab.curve_dvh.xData, edges)
        np.testing.assert_array_equal(self.tab.curve_dvh.yData, volumes)
        np.testing.assert_array_equal(result['hist_y'], volumes)
        self.tab.combo_dvh_range.setCurrentIndex(0)
        self.assertGreater(self.tab.plot_dvh.viewRange()[0][0], 9.)
        self.assertIs(self.tab.results_data[1], result)

    def test_constant_and_zero_dvh_show_the_entire_occupied_bin(self):
        for value in (0., 11.):
            with self.subTest(dose=value):
                self.tab.clear_all_rois()
                self.films(np.full((24, 24), value))
                self.ellipse([5, 5], [10, 10])
                self.calculate()
                self.tab.table.selectRow(0)
                result = self.tab.results_data[1]
                occupied = np.flatnonzero(result['hist_y'] > 0)
                lo, hi = self.tab.plot_dvh.viewRange()[0]
                self.assertLessEqual(lo, result['hist_edges'][occupied[0]])
                self.assertGreaterEqual(hi, result['hist_edges'][occupied[-1]+1])
                self.assertGreater(hi, lo)
                self.assertAlmostEqual(self.tab.curve_dvh.yData.sum(), 100.)

    def test_unirradiated_roi_is_not_skipped_or_given_a_positive_floor(self):
        self.films(np.zeros((20, 20)))
        self.ellipse([3, 3], [6, 6])
        self.calculate()
        result = self.tab.results_data[1]
        self.assertEqual(result['mean_dose'], 0)
        self.assertEqual(result['min_dose'], 0)
        self.assertTrue(np.all(result['raw_voxels'] == 0))
        self.assertAlmostEqual(result['hist_y'].sum(), 100)
        self.assertEqual(self.tab.table.item(0, 2).text(), 'N/A')
        self.assertEqual(self.tab.table.item(0, 3).text(), 'N/A')

    def test_shape_padding_is_not_counted_as_zero_dose(self):
        self.films(np.full((20, 20), 7.))
        self.ellipse([3, 3], [8, 6])
        self.calculate()
        result = self.tab.results_data[1]
        self.assertAlmostEqual(result['mean_dose'], 7)
        self.assertAlmostEqual(result['min_dose'], 7)
        self.assertLess(result['raw_voxels'].size, 8 * 6)
        self.assertAlmostEqual(result['spread_dose'], 0)

    def test_zero_dose_does_not_erase_supplied_absolute_uncertainty(self):
        self.films(np.zeros((20, 20)))
        self.tab.front_std = np.full((20, 20), .2)
        self.tab.back_std = np.full((20, 20), .2)
        self.ellipse([3, 3], [6, 6])
        self.calculate()
        result = self.tab.results_data[1]
        self.assertEqual(result['mean_dose'], 0)
        self.assertGreater(result['uncert_dose'], 0)
        self.assertIsNone(result['uncert_pct'])

    def test_partially_unirradiated_sample_retains_low_dose_volume(self):
        dose = np.zeros((20, 20)); dose[:, 6:] = 8
        self.films(dose)
        roi = self.ellipse([2, 2], [8, 6])
        self.calculate()
        mask = roi.renderShapeMask(8, 6).T > .5
        expected = dose[2:8, 2:10][mask]
        result = self.tab.results_data[1]
        np.testing.assert_allclose(result['raw_voxels'], expected, atol=1e-12)
        self.assertAlmostEqual(result['mean_dose'], np.mean(expected))
        self.assertEqual(result['min_dose'], 0)
        self.assertAlmostEqual(result['hist_y'][0], 100 * np.count_nonzero(expected == 0) / expected.size)
        self.assertAlmostEqual(result['hist_y'].sum(), 100)

    def test_rotated_polygon_uses_its_actual_local_bounds(self):
        self.films(np.full((30, 30), 4.))
        roi = pg.PolyLineROI([[4, 3], [10, 3], [7, 9]], closed=True, pos=[7, 7])
        roi.setAngle(25)
        self.tab.plot.addItem(roi)
        self.tab.rois.append(roi)
        self.calculate()
        result = self.tab.results_data[1]
        self.assertAlmostEqual(result['mean_dose'], 4)
        self.assertAlmostEqual(result['min_dose'], 4)
        self.assertGreater(result['raw_voxels'].size, 1)
        self.assertAlmostEqual(result['hist_y'].sum(), 100)

    def test_skipped_roi_does_not_change_selected_sample_identity(self):
        self.films(np.full((20, 20), 3.))
        self.ellipse([2, 2], [0, 0])
        self.ellipse([3, 3], [6, 6])
        self.calculate()
        self.assertEqual(set(self.tab.results_data), {2})
        self.assertEqual(self.tab.table.item(0, 0).data(Qt.UserRole), 2)
        self.assertIn('Sample #2', self.tab.plot_dvh.getPlotItem().titleLabel.text)
        np.testing.assert_allclose(self.tab.curve_dvh.yData, self.tab.results_data[2]['hist_y'])

    def test_moving_roi_or_changing_geometry_invalidates_exports(self):
        self.films(np.full((300, 300), 3.))
        self.tab.add_selected_roi()
        self.calculate()
        self.tab.rois[0].setPos([10, 10])
        self.assertFalse(self.tab.results_data)
        self.assertEqual(self.tab.table.rowCount(), 0)
        self.calculate()
        self.tab.spins['d_sample'].setValue(3)
        self.assertFalse(self.tab.results_data)
        self.assertEqual(self.tab.table.rowCount(), 0)

    def test_later_roi_failure_does_not_publish_partial_results(self):
        self.films(np.full((20, 20), 3.))
        self.ellipse([2, 2], [4, 4])
        self.ellipse([18, 18], [4, 4])
        with patch('bio_dosimetry.QMessageBox.warning') as warning:
            self.tab.calculate_doses()
            wait_for_tasks(self.tab.tasks)
        warning.assert_called_once()
        self.assertFalse(self.tab.results_data)
        self.assertEqual(self.tab.table.rowCount(), 0)

    def test_tube_statistics_use_contour_volume_weights(self):
        dose = np.tile(np.arange(40.), (40, 1))
        self.films(dose)
        roi = EppendorfROI([5, 5], spacing=1)
        self.tab.plot.addItem(roi)
        self.tab.rois.append(roi)
        self.calculate()
        result = self.tab.results_data[1]
        mask = roi.get_mask((8, 9))
        expected_weights = axisymmetric_volume_weights(mask)[mask]
        np.testing.assert_allclose(result['weights'], expected_weights)
        self.assertAlmostEqual(result['mean_dose'], np.average(result['raw_voxels'], weights=expected_weights))
        self.assertAlmostEqual(result['hist_y'].sum(), 100)

    def test_csv_and_npz_preserve_volume_normalization_and_precision(self):
        dose = np.tile(np.linspace(0, 1, 20), (20, 1))
        self.films(dose)
        self.ellipse([2, 2], [8, 6])
        self.calculate()
        result = self.tab.results_data[1]
        with tempfile.TemporaryDirectory() as folder:
            ddvh_path = Path(folder) / 'hist.csv'
            npz_path = Path(folder) / 'samples.npz'
            table_path = Path(folder) / 'table.csv'
            with patch('bio_dosimetry.QMessageBox.information'), \
                    patch('bio_dosimetry.QMessageBox.critical') as error:
                for export, path in ((self.tab.export_csv, ddvh_path),
                                     (self.tab.export_numpy, npz_path),
                                     (self.tab.export_sample_table, table_path)):
                    with patch('bio_dosimetry.QFileDialog.getSaveFileName', return_value=(str(path), '')):
                        export()
                error.assert_not_called()
            with ddvh_path.open(newline='') as f:
                rows = list(csv.reader(f))
            volumes = [float(row[1]) for row in rows[2:] if row]
            np.testing.assert_allclose(volumes, result['hist_y'])
            self.assertAlmostEqual(sum(volumes), 100)
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive['format_version'], 2)
                np.testing.assert_allclose(archive['sample_1_hist_volume_pct'], result['hist_y'])
                np.testing.assert_allclose(archive['sample_1_hist_edges'], result['hist_edges'])
                self.assertIn('mean_pixel_sigma_gy', archive['stat_names'])
            with table_path.open(newline='', encoding='utf-8-sig') as f:
                table_rows = list(csv.reader(f))
            self.assertEqual(float(table_rows[1][1]), result['mean_dose'])
            self.assertNotEqual(float(table_rows[1][1]), float(self.tab.table.item(0, 1).text()))


if __name__ == '__main__':
    unittest.main()
