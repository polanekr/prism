"""Responsive layout, splitter interaction and camera/ROI preservation."""
from qt_helpers import dispose_widget
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtWidgets import QApplication
from prism_gui import PrismDosimetryGUI
from viewer_layout import ViewerLayout


class ViewerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.streams = sys.stdout, sys.stderr
        self.window = PrismDosimetryGUI()
        sys.stdout, sys.stderr = self.streams
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name)/'beam_dose.npy'
        yy, xx = np.mgrid[:128, :128]
        dose = 2+11*np.exp(-((xx-64)**2+(yy-64)**2)/1600)
        np.save(path, dose)
        np.save(path.with_name('beam_std.npy'), np.full(dose.shape, .2))
        with patch.object(self.window, '_fit_2d_gaussian', return_value=None):
            self.window.load_and_display_dose(path)
        self.window.resize(1366, 768)
        self.window.show()
        self.pump()

    def tearDown(self):
        self.window.morphology_tab.invalidate_analysis()
        dispose_widget(self.window)
        self.pump()
        self.temp.cleanup()
        sys.stdout, sys.stderr = self.streams

    def pump(self):
        for _ in range(12): self.app.processEvents()

    def test_laptop_layout_gives_map_priority_and_small_windows_wrap_controls(self):
        w = self.window
        self.assertEqual(w.width(), 1366)
        self.assertEqual(w.height(), 768)
        self.assertEqual(w.viewer_layout.mode, 'compact')
        self.assertTrue(w.raw_view_panel.isHidden())
        self.assertTrue(w.stats_scroll.isHidden())
        self.assertGreater(w.glw_dose.width(), 600)
        self.assertGreater(w.glw_dose.height(), 300)
        self.assertLess(w.glw_prof_x.height(), .3*w.profile_vertical_splitter.height())
        self.assertLess(w.glw_prof_y.width(), .25*w.profile_horizontal_splitter.width())
        w.resize(1024, 640); self.pump()
        self.assertEqual(w.width(), 1024)
        self.assertTrue(w.viewer_layout._toolbar_stacked)
        self.assertGreater(w.glw_dose.height(), 150)
        self.assertGreater(w.glw_dose.width(), w.glw_prof_y.width())
        for group in w.viewer_tool_groups:
            self.assertLessEqual(group.geometry().right(), w.viewer_tab.width())

    def test_horizontal_profile_plot_tracks_displayed_film_width(self):
        from PyQt5.QtCore import QPointF
        w = self.window
        data = w.curve_prof_x.yData.copy()
        def global_x(canvas, scene_point):
            return canvas.viewport().mapToGlobal(canvas.mapFromScene(scene_point)).x()
        def assert_aligned():
            self.pump()
            x1, x2, _, _ = w._film_pixel_bounds()
            view = w.plot_dose.getViewBox()
            film_left = global_x(w.glw_dose, view.mapViewToScene(QPointF(x1, 0)))
            film_right = global_x(w.glw_dose, view.mapViewToScene(QPointF(x2, 0)))
            rect = w.plot_prof_x.getViewBox().sceneBoundingRect()
            profile_left = global_x(w.glw_prof_x, rect.topLeft())
            profile_right = global_x(w.glw_prof_x, rect.bottomRight())
            self.assertAlmostEqual(profile_left, film_left, delta=2)
            self.assertAlmostEqual(profile_right, film_right, delta=2)
            self.assertLess(w.glw_prof_x.width(), .8*w.profile_x_panel.width())
        assert_aligned()
        w.profile_horizontal_splitter.moveSplitter(200, 1)
        assert_aligned()
        w.resize(1920, 1080)
        assert_aligned()
        # Zoom and pan with the complete film still visible.
        w.plot_dose.setRange(xRange=(-20, 150), yRange=(-10, 140), padding=0)
        assert_aligned()
        np.testing.assert_array_equal(w.curve_prof_x.yData, data)
        # Changing the film boundary changes the profile's data span and width.
        w.film_boundary.setPos([20, 15]); w.film_boundary.setSize([75, 90])
        w.update_boundary_and_profiles()
        assert_aligned()
        np.testing.assert_allclose(w.plot_prof_x.viewRange()[0], [20, 95])

    def test_large_window_uses_detailed_and_explicit_mode_wins(self):
        w = self.window
        w.resize(1920, 1080); self.pump()
        self.assertEqual(w.viewer_layout.mode, 'detailed')
        self.assertFalse(w.raw_view_panel.isHidden())
        self.assertFalse(w.stats_scroll.isHidden())
        w.combo_view_layout.setCurrentIndex(2); self.pump()
        self.assertEqual(w.viewer_layout.mode, 'compact')
        self.assertTrue(w.raw_view_panel.isHidden())
        w.combo_view_layout.setCurrentIndex(1)
        w.resize(1366, 768); self.pump()
        self.assertEqual(w.viewer_layout.mode, 'detailed')
        self.assertEqual(w.width(), 1366)
        self.assertFalse(w.stats_scroll.isHidden())

    def test_focus_keeps_live_roi_data_camera_and_std_window(self):
        w = self.window
        w.roi.setPos([35, 40]); w.roi.setSize([25, 30])
        w.plot_dose.setRange(xRange=(20, 100), yRange=(25, 105), padding=0)
        w.btn_toggle_map.click()
        w.colorbar.setLevels(.1, .3)
        self.pump()
        before_camera = deepcopy(w.plot_dose.getViewBox().state['targetRange'])
        w.resize(1440, 800); self.pump()
        np.testing.assert_allclose(w.plot_dose.getViewBox().state['targetRange'], before_camera)
        roi_state = w.roi.saveState()
        dose = w.current_dose_map.copy()
        profile = w.curve_prof_x.yData.copy()
        map_item, roi_item = w.img_item_dose, w.roi
        previous_size = w.glw_dose.size()
        w.btn_map_focus.click(); self.pump()
        self.assertTrue(w.glw_prof_x.isHidden())
        self.assertTrue(w.glw_prof_y.isHidden())
        self.assertTrue(w.control_scroll.isHidden())
        self.assertGreater(w.glw_dose.width(), previous_size.width())
        self.assertGreater(w.glw_dose.height(), previous_size.height())
        np.testing.assert_allclose(w.plot_dose.getViewBox().state['targetRange'], before_camera)
        # Camera edits made while focused must also survive the return.
        w.plot_dose.setRange(xRange=(40, 90), yRange=(45, 95), padding=0)
        focus_camera = deepcopy(w.plot_dose.getViewBox().state['targetRange'])
        w.btn_map_focus.click(); self.pump()
        self.assertIs(w.img_item_dose, map_item)
        self.assertIs(w.roi, roi_item)
        self.assertEqual(w.roi.saveState(), roi_state)
        np.testing.assert_allclose(w.plot_dose.getViewBox().state['targetRange'], focus_camera)
        np.testing.assert_array_equal(w.img_item_dose.getLevels(), [.1, .3])
        np.testing.assert_array_equal(w.current_dose_map, dose)
        np.testing.assert_array_equal(w.curve_prof_x.yData, profile)
        self.assertEqual(w._display_map_mode, 'std')
        self.assertFalse(w.control_scroll.isHidden())

    def test_dragged_splitters_and_visibility_survive_focus_and_mode_round_trip(self):
        w = self.window
        w.profile_horizontal_splitter.moveSplitter(140, 1)
        w.profile_vertical_splitter.moveSplitter(340, 1)
        w.chk_view_raw.setChecked(True)
        w.chk_view_stats.setChecked(True)
        self.pump()
        ratios = deepcopy(w.viewer_layout.snapshot()['preferences']['compact']['ratios'])
        w.btn_map_focus.click(); self.pump()
        w.btn_map_focus.click(); self.pump()
        actual = w.viewer_layout.snapshot()['preferences']['compact']['ratios']
        for key in ('profile_horizontal', 'profile_vertical'):
            np.testing.assert_allclose(actual[key], ratios[key], atol=.015)
        self.assertFalse(w.raw_view_panel.isHidden())
        self.assertFalse(w.stats_scroll.isHidden())
        w.combo_view_layout.setCurrentIndex(1); self.pump()
        w.combo_view_layout.setCurrentIndex(2); self.pump()
        self.assertTrue(w.chk_view_raw.isChecked())
        self.assertTrue(w.chk_view_stats.isChecked())

    def test_visibility_switch_reopens_a_manually_collapsed_panel(self):
        w = self.window
        w.chk_view_raw.setChecked(True); self.pump()
        w.images_splitter.moveSplitter(0, 1); self.pump()
        self.assertEqual(w.images_splitter.sizes()[0], 0)
        w.chk_view_raw.setChecked(False)
        w.chk_view_raw.setChecked(True); self.pump()
        self.assertGreater(w.images_splitter.sizes()[0], 0)

    def test_oblique_profile_stays_hidden_after_focus_and_resize(self):
        w = self.window
        w.combo_prof_mode.setCurrentIndex(1); self.pump()
        self.assertTrue(w.glw_prof_y.isHidden())
        self.assertFalse(w.glw_prof_x.isHidden())
        w.btn_map_focus.click(); w.resize(1920, 1080); self.pump()
        self.assertTrue(w.glw_prof_x.isHidden())
        w.btn_map_focus.click(); self.pump()
        self.assertTrue(w.glw_prof_y.isHidden())
        self.assertFalse(w.glw_prof_x.isHidden())
        w.combo_prof_mode.setCurrentIndex(0); self.pump()
        self.assertFalse(w.glw_prof_y.isHidden())

    def test_layout_snapshot_restores_preferences_but_not_transient_focus(self):
        w = self.window
        w.combo_view_layout.setCurrentIndex(2)
        w.chk_view_raw.setChecked(True)
        w.profile_horizontal_splitter.moveSplitter(155, 1)
        self.pump()
        state = w.viewer_layout.snapshot()
        w.btn_map_focus.click()
        focused_state = w.viewer_layout.snapshot()
        self.assertEqual(focused_state, state)
        w.viewer_layout.restore(state); self.pump()
        self.assertFalse(w.btn_map_focus.isChecked())
        self.assertEqual(w.combo_view_layout.currentIndex(), 2)
        self.assertFalse(w.raw_view_panel.isHidden())
        self.assertTrue(w.stats_scroll.isHidden())
        bad = deepcopy(state)
        bad['preferences']['compact']['ratios']['profile_vertical'] = [float('nan'), 0]
        with self.assertRaises(ValueError): w.viewer_layout.restore(bad)
        self.assertFalse(w.btn_map_focus.isChecked())
