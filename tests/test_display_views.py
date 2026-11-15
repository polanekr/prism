"""Display ranges must not leak between dose/STD or alter numerical data."""
from qt_helpers import dispose_widget
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtWidgets import QApplication
from prism_gui import PrismDosimetryGUI


class DisplayViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.streams = sys.stdout, sys.stderr
        self.window = PrismDosimetryGUI()
        sys.stdout, sys.stderr = self.streams
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'beam_dose.npy'
        yy, xx = np.mgrid[:80, :120]
        self.dose = (10 + xx/120 + yy/80).astype(np.float32)
        self.std = (.2 + xx/1200 + yy/800).astype(np.float32)
        # Film-edge artifacts must not flatten the interior display.
        self.dose[:4] = 10000
        self.std[:4] = 1000000
        np.save(self.path, self.dose)
        np.save(self.path.with_name('beam_std.npy'), self.std)
        self.fit = patch.object(self.window, '_fit_2d_gaussian', return_value=None)
        self.fit.start()
        self.window.load_and_display_dose(self.path)
        self.app.processEvents()

    def tearDown(self):
        self.fit.stop()
        self.window.morphology_tab.invalidate_analysis()
        dispose_widget(self.window)
        self.app.processEvents()
        self.temp.cleanup()
        sys.stdout, sys.stderr = self.streams

    def toggle(self):
        self.window.btn_toggle_map.click()
        self.app.processEvents()

    def test_dose_std_round_trip_preserves_independent_windows_and_colors(self):
        w = self.window
        self.assertLess(w.colorbar.getLevels()[1], 13)
        w.colorbar.setLevels(10.3, 11.7)
        w.combo_cmap.setCurrentText('plasma')
        dose_lut = w.colorbar.getLookupTable(n=256).copy()
        self.toggle()
        self.assertLess(w.colorbar.getLevels()[1], .5)
        self.assertLess(w.colorbar.getHistogramRange()[1], 1.)
        np.testing.assert_array_equal(w.img_item_dose.image, self.std)
        w.colorbar.setLevels(.22, .38)
        w.combo_cmap.setCurrentText('inferno')
        self.toggle()
        np.testing.assert_array_equal(w.img_item_dose.image, self.dose)
        np.testing.assert_allclose(w.img_item_dose.getLevels(), [10.3, 11.7])
        np.testing.assert_array_equal(w.colorbar.getLookupTable(n=256), dose_lut)
        w.img_item_dose.render()
        self.assertIsNotNone(w.img_item_dose.qimage)
        # Both low and high film doses have different colors after switching back.
        image = w.img_item_dose.qimage
        self.assertNotEqual(image.pixelColor(10, 10), image.pixelColor(100, 60))
        self.toggle()
        np.testing.assert_allclose(w.img_item_dose.getLevels(), [.22, .38])
        w.film_boundary.setPos([12, 8]); w.film_boundary.setSize([90, 60])
        w.update_boundary_and_profiles()
        self.assertLess(w.colorbar.getLevels()[1], .5)
        self.toggle()
        self.assertEqual(w.combo_cmap.currentText(), 'plasma')
        self.assertLess(w.colorbar.getLevels()[1], 13)
        np.testing.assert_array_equal(w.current_dose_map, self.dose)
        np.testing.assert_array_equal(w.current_std_map, self.std)

    def test_roi_zoom_changes_only_viewport_and_profiles_follow_film(self):
        w = self.window
        x, y = w.curve_roi_dah.xData.copy(), w.curve_roi_dah.yData.copy()
        self.assertGreater(w.plot_roi_hist.viewRange()[0][0], 9)
        w.combo_hist_range.setCurrentIndex(1)
        self.assertLessEqual(w.plot_roi_hist.viewRange()[0][0], 0)
        np.testing.assert_array_equal(w.curve_roi_dah.xData, x)
        np.testing.assert_array_equal(w.curve_roi_dah.yData, y)
        w.film_boundary.setPos([12, 8]); w.film_boundary.setSize([90, 60])
        w.update_boundary_and_profiles()
        np.testing.assert_allclose(w.plot_prof_x.viewRange()[0], [12, 102])
        np.testing.assert_allclose(w.plot_prof_y.viewRange()[1], [8, 68])
        before = w.curve_prof_x.yData.copy()
        w.plot_dose.setRange(xRange=(-100, 200), yRange=(-100, 200))
        np.testing.assert_allclose(w.plot_prof_x.viewRange()[0], [12, 102])
        self.toggle()
        np.testing.assert_array_equal(w.curve_prof_x.yData, before)
        self.assertLessEqual(w.glw_prof_x.minimumHeight(), 140)
        self.assertLessEqual(w.glw_prof_y.minimumWidth(), 100)

    def test_loading_another_film_from_std_resets_dose_window(self):
        self.window.combo_cmap.setCurrentText('plasma')
        self.toggle()
        self.window.load_and_display_dose(self.path)
        self.assertFalse(self.window.btn_toggle_map.isChecked())
        self.assertEqual(self.window.combo_cmap.currentText(), 'plasma')
        np.testing.assert_array_equal(self.window.img_item_dose.image, self.dose)
        self.assertGreater(self.window.colorbar.getLevels()[0], 9)
        self.assertLess(self.window.colorbar.getLevels()[1], 13)
