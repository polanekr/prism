import sys
import os
import subprocess
import datetime
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLabel, QFileDialog, QListWidget, 
    QComboBox, QCheckBox, QLineEdit, QGroupBox, QTabWidget, 
    QSplitter, QMessageBox, QProgressBar, QAbstractItemView,
    QDialog, QDialogButtonBox, QSlider, QSpinBox, QGraphicsPathItem 
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPageSize, QPdfWriter, QPainterPath
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, map_coordinates

# Force pyqtgraph to use PyQt5
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg

from dosimetry import GafchromicEngine, BAYES_AVAILABLE
pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)


# =============================================================================
# 2D GAUSSIAN FITTING FUNCTION (ROTATED ELLIPTICAL GAUSSIAN)
# =============================================================================
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


# =============================================================================
# BLANK FILM ROI SELECTION DIALOG
# =============================================================================
class BlankRoiDialog(QDialog):
    def __init__(self, blank_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blank Film Reference Selection")
        self.resize(800, 600)
        self.blank_path = blank_path
        self.mean_rgb = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel(
            "<b>Position the red rectangle over a clean, unmarked area on the unexposed film!</b><br>"
            "The mean RGB of this region will be used as a uniform background reference."
        )
        layout.addWidget(lbl)

        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        layout.addWidget(self.glw)

        try:
            raw_img = tifffile.imread(str(self.blank_path))
            if raw_img.ndim == 3 and raw_img.shape[2] > 3:
                raw_img = raw_img[:, :, :3]
            raw_img = np.flip(raw_img, axis=0) 
            self.raw_img_data = raw_img.astype(np.float64)
            if raw_img.dtype == np.uint16:
                display_img = (raw_img / 256).astype(np.uint8)
            else:
                display_img = raw_img.copy()
            self.img_item.setImage(display_img)
        except Exception as e:
            self.raw_img_data = np.zeros((500, 500, 3))
            QMessageBox.critical(self, "Load Error", f"Could not load blank film:\n{e}")

        h, w = self.raw_img_data.shape[:2]
        self.roi = pg.RectROI([w // 2 - 50, h // 2 - 50], [100, 100], pen=pg.mkPen('r', width=2))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.plot.addItem(self.roi)

        self.stats_lbl = QLabel("<b>Mean RGB:</b> -")
        self.stats_lbl.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.stats_lbl)

        self.roi.sigRegionChanged.connect(self.update_stats)
        
        self.bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        layout.addWidget(self.bbox)

        self.update_stats()

    def update_stats(self):
        try:
            roi_data = self.roi.getArrayRegion(self.raw_img_data, self.img_item)
            if roi_data is not None and roi_data.size > 0:
                if self.raw_img_data.ndim == 3:
                    self.mean_rgb = tuple(np.mean(roi_data, axis=(0, 1)))
                    self.stats_lbl.setText(f"<b>Mean RGB:</b> R={self.mean_rgb[0]:.2f}, G={self.mean_rgb[1]:.2f}, B={self.mean_rgb[2]:.2f}")
                else:
                    m_val = float(np.mean(roi_data))
                    self.mean_rgb = (m_val, m_val, m_val)
                    self.stats_lbl.setText(f"<b>Mean Value:</b> {m_val:.2f} (Grayscale)")
        except Exception as e:
            self.stats_lbl.setText(f"<b>Error calculating stats:</b> {e}")


# =============================================================================
# BACKGROUND WORKER THREAD
# =============================================================================
class EngineWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, task_type, engine, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.engine = engine
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == 'calibrate':
                method = self.kwargs.get('method', 'standard')
                folder = self.kwargs.get('folder')
                doses = self.kwargs.get('doses')
                cleaning = self.kwargs.get('cleaning', False)
                orig_out = self.engine.output_folder
                self.engine.output_folder = folder
                if method == 'mcmc':
                    if not BAYES_AVAILABLE:
                        raise RuntimeError("PyMC/ArviZ packages are not available for MCMC calibration!")
                    self.engine.run_mcmc_calibration(folder, doses, use_cleaning=cleaning)
                else:
                    self.engine.run_calibration(folder, doses)
                self.engine.output_folder = orig_out
                self.finished.emit("Calibration completed successfully!")

            elif self.task_type == 'process':
                files = self.kwargs.get('files')
                blank = self.kwargs.get('blank')
                method = self.kwargs.get('method')
                cleaning = self.kwargs.get('cleaning')
                blank_rgb = self.kwargs.get('blank_rgb')
                self.engine.process_films(files, blank, method=method, use_cleaning=cleaning, blank_rgb=blank_rgb)
                self.finished.emit(f"Successfully converted {len(files)} film(s) to dose maps!")
                
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================
class PrismDosimetryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PRISM - Particle Beam & LWFA Dosimetry Suite")
        self.resize(1720, 980)

        self.project_dir = None
        self.raw_dir = None
        self.processed_dir = None
        self.analysis_dir = None
        self.calib_dir = None
        
        self.engine = None
        self.current_dose_map = None
        self.current_raw_img = None
        self.current_meta = {}
        self.current_film_name = "dose_map"
        self.global_blank_rgb = None
        self.isocurves = []
        self.is_white_mode = False

        self.pen_prof_x = pg.mkPen('y', width=3.0)
        self.pen_prof_y = pg.mkPen('c', width=3.0)
        self.pen_hist = pg.mkPen('#00bfff', width=2.5)
        self.pen_dah = pg.mkPen('#ff3333', width=4.0)

        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        
        self.viewer_tab = self._create_viewer_tab()
        self.tabs.addTab(self.viewer_tab, "🔬 LWFA Beam Analysis & Profiles")
        self.tabs.addTab(self._create_placeholder_tab("Magnetic Spectrometer & Dispersion", "WIP"), "⚡ Spectrometer (WIP)")
        self.tabs.addTab(self._create_placeholder_tab("RCF Stack & Depth-Dose Bragg Curve", "WIP"), "🧱 RCF Stack Analyzer (WIP)")
        
        splitter.setSizes([380, 1340])

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. Project Session
        proj_group = QGroupBox("1. Project Session")
        proj_layout = QVBoxLayout(proj_group)
        self.btn_new_proj_scan = QPushButton("🆕 New Project & Open Scanner")
        self.btn_new_proj_scan.setStyleSheet("font-weight: bold; background-color: #2b8a3e; color: white;")
        self.btn_new_proj_scan.clicked.connect(self.create_project_and_launch_scanner)
        proj_layout.addWidget(self.btn_new_proj_scan)

        self.btn_open_proj = QPushButton("📁 Open Existing Project")
        self.btn_open_proj.clicked.connect(self.select_project_folder)
        proj_layout.addWidget(self.btn_open_proj)

        self.btn_refresh = QPushButton("🔄 Refresh File Lists")
        self.btn_refresh.clicked.connect(self.refresh_file_lists)
        proj_layout.addWidget(self.btn_refresh)

        self.lbl_proj_path = QLabel("No project selected")
        self.lbl_proj_path.setStyleSheet("color: gray; font-size: 11px;")
        proj_layout.addWidget(self.lbl_proj_path)
        layout.addWidget(proj_group)

        # 2. Calibration
        calib_group = QGroupBox("2. Calibration")
        calib_layout = QVBoxLayout(calib_group)
        self.btn_sel_calib_dir = QPushButton("📂 Select Calibration Folder")
        self.btn_sel_calib_dir.clicked.connect(self.select_calibration_folder)
        self.lbl_calib_status = QLabel("Status: No calibration loaded")
        self.lbl_calib_status.setStyleSheet("color: #d9534f; font-weight: bold;")
        calib_layout.addWidget(self.btn_sel_calib_dir)
        calib_layout.addWidget(self.lbl_calib_status)
        calib_layout.addWidget(QLabel("Doses [Gy]:"))
        self.input_doses = QLineEdit("0, 1, 2, 4, 6, 8, 10")
        calib_layout.addWidget(self.input_doses)
        hbox_cal_opt = QHBoxLayout()
        self.combo_calib_method = QComboBox()
        self.combo_calib_method.addItems(["Standard (Curve Fit)", "Bayesian (MCMC)"])
        self.chk_calib_clean = QCheckBox("bFDR cleaning")
        hbox_cal_opt.addWidget(self.combo_calib_method)
        hbox_cal_opt.addWidget(self.chk_calib_clean)
        calib_layout.addLayout(hbox_cal_opt)
        hbox_cal_btns = QHBoxLayout()
        self.btn_run_calib = QPushButton("⚙️ Run Calibration")
        self.btn_run_calib.clicked.connect(self.run_new_calibration)
        self.btn_load_json = QPushButton("📥 Load JSON")
        self.btn_load_json.clicked.connect(self.load_existing_calibration)
        hbox_cal_btns.addWidget(self.btn_run_calib)
        hbox_cal_btns.addWidget(self.btn_load_json)
        calib_layout.addLayout(hbox_cal_btns)
        layout.addWidget(calib_group)

        # 3. Conversion
        conv_group = QGroupBox("3. Conversion")
        conv_layout = QVBoxLayout(conv_group)
        conv_layout.addWidget(QLabel("Raw scanned films:"))
        self.list_raw_files = QListWidget()
        self.list_raw_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_raw_files.itemClicked.connect(self.on_raw_file_clicked)
        conv_layout.addWidget(self.list_raw_files)
        
        conv_layout.addWidget(QLabel("Unexposed (Blank) reference:"))
        blank_hbox = QHBoxLayout()
        self.combo_blank_film = QComboBox()
        self.combo_blank_film.currentIndexChanged.connect(self._invalidate_blank_rgb)
        self.btn_set_blank_roi = QPushButton("🎯 Set ROI")
        self.btn_set_blank_roi.clicked.connect(self.open_blank_roi_dialog)
        blank_hbox.addWidget(self.combo_blank_film)
        blank_hbox.addWidget(self.btn_set_blank_roi)
        conv_layout.addLayout(blank_hbox)

        conv_layout.addWidget(QLabel("Solver method:"))
        self.combo_solver = QComboBox()
        self.combo_solver.addItems(["fast (Vectorized, fast)", "hybrid (Pixel-wise optimization)"])
        self.chk_conv_clean = QCheckBox("Apply bFDR artifact removal")
        conv_layout.addWidget(self.combo_solver)
        conv_layout.addWidget(self.chk_conv_clean)
        
        hbox_conv_btns = QHBoxLayout()
        self.btn_conv_selected = QPushButton("▶️ Convert Selected")
        self.btn_conv_selected.clicked.connect(lambda: self.convert_films(batch_all=False))
        self.btn_conv_all = QPushButton("⏩ Convert All (Batch)")
        self.btn_conv_all.clicked.connect(lambda: self.convert_films(batch_all=True))
        hbox_conv_btns.addWidget(self.btn_conv_selected)
        hbox_conv_btns.addWidget(self.btn_conv_all)
        conv_layout.addLayout(hbox_conv_btns)
        layout.addWidget(conv_group)

        # 4. Processed
        proc_group = QGroupBox("4. Processed Dose Maps")
        proc_layout = QVBoxLayout(proc_group)
        self.list_proc_files = QListWidget()
        self.list_proc_files.itemClicked.connect(self.on_proc_file_clicked)
        proc_layout.addWidget(self.list_proc_files)
        layout.addWidget(proc_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        return panel

    def _create_viewer_tab(self):
        tab = QWidget()
        main_v_layout = QVBoxLayout(tab)

        # TOOLBAR: Visualization & Profile modes
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(QLabel("<b>Colormap:</b>"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["viridis", "plasma", "inferno", "turbo", "bipolar"])
        self.combo_cmap.currentTextChanged.connect(self._change_colormap)
        tool_layout.addWidget(self.combo_cmap)

        self.btn_theme_toggle = QPushButton("☀️ Light Mode")
        self.btn_theme_toggle.setStyleSheet("font-weight: bold; padding: 3px 8px;")
        self.btn_theme_toggle.clicked.connect(self._toggle_ui_theme)
        tool_layout.addWidget(self.btn_theme_toggle)

        self.chk_isodose = QCheckBox("Isodose Contours (20, 50, 80, 90%)")
        self.chk_isodose.setChecked(False)
        self.chk_isodose.toggled.connect(self._toggle_isodose)
        tool_layout.addWidget(self.chk_isodose)

        self.chk_show_roi = QCheckBox("Show Analysis ROI")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.toggled.connect(self._toggle_roi_visibility)
        tool_layout.addWidget(self.chk_show_roi)

        tool_layout.addWidget(QLabel("<b>Profile Mode:</b>"))
        self.combo_prof_mode = QComboBox()
        self.combo_prof_mode.addItems(["Orthogonal (X/Y)", "Oblique (Angle/Line)"])
        self.combo_prof_mode.currentIndexChanged.connect(self._on_profile_mode_changed)
        tool_layout.addWidget(self.combo_prof_mode)

        tool_layout.addWidget(QLabel("Band Width (px):"))
        self.spin_band = QSpinBox()
        self.spin_band.setRange(1, 35)
        self.spin_band.setValue(3)
        self.spin_band.valueChanged.connect(lambda: self.update_profiles())
        tool_layout.addWidget(self.spin_band)

        tool_layout.addWidget(QLabel("<b>Split View / Overlay:</b>"))
        self.slider_blend = QSlider(Qt.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(100) # 100: Pure Dose, 0: Pure Raw
        self.slider_blend.valueChanged.connect(self._update_blended_view)
        tool_layout.addWidget(self.slider_blend)

        main_v_layout.addLayout(tool_layout)

        # --- KÖZÖS KOMPOZIT EXPORT KONTRÉNER (Splitter helyett egyetlen widget a hibátlan mentéshez) ---
        self.composite_export_container = QWidget()
        self.composite_export_container.setStyleSheet("QWidget { background-color: #1a1a1a; color: #EEEEEE; }")
        comp_layout = QHBoxLayout(self.composite_export_container)
        comp_layout.setContentsMargins(4, 4, 4, 4)
        comp_layout.setSpacing(8)

        # 1. BAL OLDAL: Nyers szkennelt film panelje
        self.raw_widget = QWidget()
        raw_layout = QVBoxLayout(self.raw_widget)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_raw_header = QLabel("<b>Raw Scanned Film (RGB) with Synced Reticle</b>")
        raw_layout.addWidget(self.lbl_raw_header)
        
        self.glw_raw = pg.GraphicsLayoutWidget()
        self.plot_raw = self.glw_raw.addPlot()
        self.plot_raw.setAspectLocked(True)
        self.plot_raw.setLabel('bottom', 'X [mm]')
        self.plot_raw.setLabel('left', 'Y [mm]')
        self.img_item_raw = pg.ImageItem()
        self.plot_raw.addItem(self.img_item_raw)
        
        # Szinkronizált célkereszt a nyers filmen
        self.v_line_raw = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
        self.h_line_raw = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
        self.plot_raw.addItem(self.v_line_raw, ignoreBounds=True)
        self.plot_raw.addItem(self.h_line_raw, ignoreBounds=True)
        raw_layout.addWidget(self.glw_raw)

        # 2. JOBB OLDAL: Teljes dózistérkép és LWFA Analytics Dashboard
        self.export_widget = QWidget()
        export_layout = QVBoxLayout(self.export_widget)
        export_layout.setContentsMargins(0, 0, 0, 0)
        
        header_hbox = QHBoxLayout()
        header_hbox.addWidget(QLabel("<b>Beam Dose Map [Gy] & LWFA Analytics</b>"))
        self.btn_save_snapshot = QPushButton("📸 Save Publication Figure (300 DPI)")
        self.btn_save_snapshot.setStyleSheet("font-weight: bold; background-color: #0078D7; color: white;")
        self.btn_save_snapshot.clicked.connect(self.save_dose_snapshot)
        header_hbox.addStretch()
        header_hbox.addWidget(self.btn_save_snapshot)
        export_layout.addLayout(header_hbox)

        # Graphics Layout a dózistérképnek és a profiloknak
        self.glw_dose = pg.GraphicsLayoutWidget()
        
        self.plot_prof_y = self.glw_dose.addPlot(row=0, col=0)
        self.plot_prof_y.setLabel('bottom', 'Dose [Gy]')
        self.plot_prof_y.setLabel('left', 'Y (CAX) [mm]')
        self.plot_prof_y.setFixedWidth(200)
        self.curve_prof_y = self.plot_prof_y.plot(pen=self.pen_prof_y)

        self.plot_dose = self.glw_dose.addPlot(row=0, col=1)
        self.plot_dose.setLabel('bottom', 'X [mm]')
        self.plot_dose.setLabel('left', 'Y [mm]')
        self.plot_dose.setAspectLocked(True)

        # Kétrétegű megjelenítés (Split View): Z-értékek negatívak, hogy a tengelyek előtte legyenek
        self.img_item_raw_bg = pg.ImageItem()
        self.img_item_raw_bg.setZValue(-10)
        self.plot_dose.addItem(self.img_item_raw_bg)
        
        self.img_item_dose = pg.ImageItem(autoDownsample=False)
        self.img_item_dose.setZValue(-5)
        self.plot_dose.addItem(self.img_item_dose)

        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.img_item_dose)
        self.colorbar.gradient.loadPreset('viridis')
        self.colorbar.setMinimumWidth(100)
        self.colorbar.setMaximumWidth(130)
        self.glw_dose.addItem(self.colorbar, row=0, col=2)
        self.colorbar.sigLevelsChanged.connect(self._on_lut_levels_changed)
        self.colorbar.sigLookupTableChanged.connect(self._on_lut_gradient_changed)

        self.plot_prof_x = self.glw_dose.addPlot(row=1, col=1)
        self.plot_prof_x.setLabel('left', 'Dose [Gy]')
        self.plot_prof_x.setLabel('bottom', 'Axis / Path [mm]')
        self.plot_prof_x.setFixedHeight(190)
        self.curve_prof_x = self.plot_prof_x.plot(pen=self.pen_prof_x)
        self.curve_prof_fit = self.plot_prof_x.plot(pen=pg.mkPen('r', width=1.5, style=Qt.DashLine))

        self.glw_dose.ci.layout.setColumnStretchFactor(0, 0)
        self.glw_dose.ci.layout.setColumnStretchFactor(1, 10)
        self.glw_dose.ci.layout.setColumnStretchFactor(2, 0)
        self.glw_dose.ci.layout.setRowStretchFactor(0, 8)
        self.glw_dose.ci.layout.setRowStretchFactor(1, 0)

        # Célkeresztek
        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('y', width=1.5, style=Qt.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=Qt.DashLine))
        self.plot_dose.addItem(self.v_line, ignoreBounds=True)
        self.plot_dose.addItem(self.h_line, ignoreBounds=True)
        self.v_line.sigPositionChanged.connect(self._on_crosshair_moved)
        self.h_line.sigPositionChanged.connect(self._on_crosshair_moved)

        # Ferde vonalzó (Oblique Line ROI)
        self.line_roi = pg.LineSegmentROI([[20, 20], [80, 80]], pen=pg.mkPen('y', width=2.5))
        self.line_roi.sigRegionChanged.connect(self._on_oblique_moved)
        self.plot_dose.addItem(self.line_roi)
        self.line_roi.hide()

        # Film boundary (Piros keret)
        self.film_boundary = pg.RectROI([10, 10], [100, 100], pen=pg.mkPen('r', width=1.5, style=Qt.DashLine))
        self.film_boundary.addScaleHandle([1, 1], [0, 0])
        self.film_boundary.addScaleHandle([0, 0], [1, 1])
        self.plot_dose.addItem(self.film_boundary)
        self.film_boundary.sigRegionChanged.connect(self.update_boundary_and_profiles)

        # Nyaláb ROI (Fehér ellipszis)
        self.roi = pg.EllipseROI([50, 50], [100, 100], pen=pg.mkPen('w', width=2))
        self.plot_dose.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_roi_stats)

        export_layout.addWidget(self.glw_dose, stretch=4)

        # ALSÓ SÁV: 4 oszlopos metrikák és hisztogram / DAH
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(2, 2, 2, 2)
        bottom_layout.setSpacing(8)

        self.stats_group = QGroupBox("Beam Parameters & Medical Physics Metrics")
        self.stats_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; } QLabel { font-size: 11px; }")
        
        stats_grid = QGridLayout(self.stats_group)
        stats_grid.setContentsMargins(6, 8, 6, 4)
        stats_grid.setHorizontalSpacing(6)
        stats_grid.setVerticalSpacing(2)
        
        # Col 0: Spot Geometry & Pointing (Fantombeli profil)
        self.lbl_beam_peak = QLabel("<b>Peak:</b> - Gy @ (-, -) mm")
        self.lbl_beam_com = QLabel("<b>CoM:</b> (-, -) mm")
        self.lbl_beam_fwhm = QLabel("<b>FWHM (X/Y):</b> - / - mm")
        self.lbl_roi_area = QLabel("<b>ROI Area:</b> - mm²")
        stats_grid.addWidget(QLabel("<span style='color: #4CAF50;'><b>--- Geometry ---</b></span>"), 0, 0)
        stats_grid.addWidget(self.lbl_beam_peak, 1, 0)
        stats_grid.addWidget(self.lbl_beam_com, 2, 0)
        stats_grid.addWidget(self.lbl_beam_fwhm, 3, 0)
        stats_grid.addWidget(self.lbl_roi_area, 4, 0)

        # Col 1: 2D Model (Fantombeli szóródási modell)
        self.lbl_fit_amp = QLabel("<b>Fit Amp:</b> - Gy")
        self.lbl_fit_sigma = QLabel("<b>Spread (σx, σy):</b> - , - mm")
        self.lbl_fit_theta = QLabel("<b>Tilt Angle (θ):</b> - deg")
        self.lbl_total_dose = QLabel("<b>Integral Dose:</b> - Gy·mm²")
        stats_grid.addWidget(QLabel("<span style='color: #FF9800;'><b>--- 2D Fit & Energy ---</b></span>"), 0, 1)
        stats_grid.addWidget(self.lbl_fit_amp, 1, 1)
        stats_grid.addWidget(self.lbl_fit_sigma, 2, 1)
        stats_grid.addWidget(self.lbl_fit_theta, 3, 1)
        stats_grid.addWidget(self.lbl_total_dose, 4, 1)

        # Col 2: Fantom Dózisstatisztika és Térbeli Eltérés
        self.lbl_dose_stats = QLabel("<b>Mean Dose:</b> - Gy")
        self.lbl_dose_std = QLabel("<b>Spatial STD:</b> ±- Gy (- %)")
        self.lbl_dose_minmax = QLabel("<b>Min / Max:</b> - / - Gy")
        self.lbl_d90_val = QLabel("<b>D90%:</b> - Gy (V95%: -%)")
        stats_grid.addWidget(QLabel("<span style='color: #2196F3;'><b>--- Phantom Dose ---</b></span>"), 0, 2)
        stats_grid.addWidget(self.lbl_dose_stats, 1, 2)
        stats_grid.addWidget(self.lbl_dose_std, 2, 2)
        stats_grid.addWidget(self.lbl_dose_minmax, 3, 2)
        stats_grid.addWidget(self.lbl_d90_val, 4, 2)

        # Col 3: QA & Method
        self.lbl_penumbra = QLabel("<b>Penumbra (80-20):</b> - mm")
        self.lbl_flatness = QLabel("<b>Flatness:</b> - %")
        self.lbl_symmetry = QLabel("<b>Symmetry:</b> - %")
        self.lbl_conv_method = QLabel("<b>Method:</b> -")
        stats_grid.addWidget(QLabel("<span style='color: #E91E63;'><b>--- Beam QA ---</b></span>"), 0, 3)
        stats_grid.addWidget(self.lbl_penumbra, 1, 3)
        stats_grid.addWidget(self.lbl_flatness, 2, 3)
        stats_grid.addWidget(self.lbl_symmetry, 3, 3)
        stats_grid.addWidget(self.lbl_conv_method, 4, 3)

        bottom_layout.addWidget(self.stats_group, stretch=3)

        # DAH & Profile Chart
        self.plot_roi_hist = pg.PlotWidget(title="Beam Dose Distribution & Cumulative DAH")
        self.plot_roi_hist.setLabel('bottom', 'Dose [Gy]')
        self.plot_roi_hist.setLabel('left', 'Relative Area [%]')
        self.plot_roi_hist.setYRange(0, 105)
        self.plot_roi_hist.setMinimumWidth(260)
        self.curve_roi_hist = self.plot_roi_hist.plot(
            stepMode="center", fillLevel=0, fillOutline=True, 
            brush=(0, 120, 220, 90), pen=self.pen_hist
        )
        self.curve_roi_dah = self.plot_roi_hist.plot(pen=self.pen_dah)
        
        bottom_layout.addWidget(self.plot_roi_hist, stretch=2)
        export_layout.addWidget(bottom_panel, stretch=1)

        # Mindkét panelt (nyers kép és komplett elemző tábla) beletesszük a közös konténerbe
        comp_layout.addWidget(self.raw_widget, stretch=1)
        comp_layout.addWidget(self.export_widget, stretch=3)

        main_v_layout.addWidget(self.composite_export_container)
        return tab

    def _create_placeholder_tab(self, title, description):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        lbl = QLabel(f"<h2>{title}</h2><p>{description}</p>")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        return tab

    def _toggle_ui_theme(self):
        """Kézi váltás sötét és világos felület között a monitoron."""
        self.is_white_mode = not self.is_white_mode
        self._apply_export_theme(white_mode=self.is_white_mode)
        
        if self.is_white_mode:
            self.btn_theme_toggle.setText("🌙 Dark Mode")
        else:
            self.btn_theme_toggle.setText("☀️ Light Mode")

    def _apply_export_theme(self, white_mode: bool, is_export: bool = False):
        """Átkapcsolja a grafikonokat, tengelyeket és háttereket fehérre vagy sötétre."""
        bg_color = '#ffffff' if white_mode else '#1a1a1a'
        text_color = '#000000' if white_mode else '#EEEEEE'
        border_color = '#cccccc' if white_mode else '#444444'

        w_factor = 2.8 if is_export else 1.0

        # 1. Teljes kompozit konténer, dobozok és feliratok stíluslapja
        self.composite_export_container.setStyleSheet(f"""
            QWidget {{ background-color: {bg_color}; color: {text_color}; }}
            QGroupBox {{ border: 1px solid {border_color}; margin-top: 6px; padding-top: 8px; font-weight: bold; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 7px; padding: 0 3px; color: {text_color}; }}
            QLabel {{ color: {text_color}; }}
        """)

        # 2. Háttérszín beállítása a vásznakon (beleértve a nyers nézetet is!)
        self.glw_raw.setBackground(bg_color)
        self.glw_dose.setBackground(bg_color)
        self.plot_roi_hist.setBackground(bg_color)

        # 3. Tengelyek színezése és vastagítása
        plot_items = [
            self.plot_raw,
            self.plot_prof_y, 
            self.plot_dose, 
            self.plot_prof_x, 
            self.plot_roi_hist.getPlotItem()
        ]
        axis_pen_width = 2.0 * w_factor if white_mode else 1.0 * w_factor
        for p in plot_items:
            for axis_name in ['left', 'bottom', 'top', 'right']:
                ax = p.getAxis(axis_name)
                ax.setTextPen(pg.mkPen(text_color))
                ax.setPen(pg.mkPen(text_color, width=axis_pen_width))

        # 4. Diagram felirat és ROI körvonal kontrasztja
        self.plot_roi_hist.setTitle("Beam Dose Distribution & Cumulative DAH", color=text_color)
        roi_pen = pg.mkPen('k' if white_mode else 'w', width=2.0 * w_factor)
        self.roi.setPen(roi_pen)

        # 5. Görbék és DAH hisztogram vonalvastagságai a szorzóval
        if white_mode:
            self.pen_prof_x = pg.mkPen('#d87a00', width=3.5 * w_factor)
            self.pen_prof_y = pg.mkPen('#0055aa', width=3.5 * w_factor)
            self.pen_hist = pg.mkPen('#003388', width=2.5 * w_factor)
            self.pen_dah = pg.mkPen('#b71c1c', width=4.5 * w_factor)
            brush_hist = pg.mkBrush(0, 80, 180, 100)
        else:
            self.pen_prof_x = pg.mkPen('y', width=2.5 * w_factor)
            self.pen_prof_y = pg.mkPen('c', width=2.5 * w_factor)
            self.pen_hist = pg.mkPen('#00bfff', width=2.5 * w_factor)
            self.pen_dah = pg.mkPen('#ff3333', width=3.5 * w_factor)
            brush_hist = pg.mkBrush(0, 150, 255, 60)

        self.curve_prof_x.setPen(self.pen_prof_x)
        self.curve_prof_y.setPen(self.pen_prof_y)
        self.curve_roi_hist.setPen(self.pen_hist)
        self.curve_roi_hist.setBrush(brush_hist)
        self.curve_roi_dah.setPen(self.pen_dah)

        if hasattr(self, 'isocurves'):
            for c in self.isocurves:
                orig_pen = c.opts['pen']
                c.setPen(pg.mkPen(orig_pen.color(), width=2.5 * w_factor))

    # =========================================================================
    # TOOLBAR & INTERACTIVE PROFILE CONTROLS
    # =========================================================================
    def _change_colormap(self, cmap_name):
        self.colorbar.gradient.loadPreset(cmap_name)
        if self.current_dose_map is not None:
            self._on_lut_gradient_changed()

    def _on_profile_mode_changed(self, idx):
        if idx == 0:  # Orthogonal
            self.v_line.show()
            self.h_line.show()
            self.line_roi.hide()
            self.plot_prof_y.show()
            self.plot_prof_x.setLabel('bottom', 'X (CAX) [mm]')
        else:  # Oblique
            self.v_line.hide()
            self.h_line.hide()
            self.line_roi.show()
            self.plot_prof_y.hide()
            self.plot_prof_x.setLabel('bottom', 'Oblique Path [mm]')
        self.update_profiles()

    def _toggle_isodose(self, checked):
        # 1. Korábbi kontúrvonalak törlése
        for item in self.isocurves:
            if item.scene() is not None:
                self.plot_dose.removeItem(item)
        self.isocurves.clear()

        # 2. Új izodózis szintek generálása a piros keret (film_boundary) alapján
        if checked and self.current_dose_map is not None:
            h, w = self.current_dose_map.shape
            pos = self.film_boundary.pos()
            size = self.film_boundary.size()

            # Kivágási határok a piros négyszög alapján
            x1 = int(np.clip(pos.x(), 0, w - 1))
            y1 = int(np.clip(pos.y(), 0, h - 1))
            x2 = int(np.clip(pos.x() + size.x(), x1 + 2, w))
            y2 = int(np.clip(pos.y() + size.y(), y1 + 2, h))

            sub_dose = self.current_dose_map[y1:y2, x1:x2]
            if sub_dose.size < 9:
                return

            # Szintek meghatározása KIZÁRÓLAG az aktív mezőből
            max_d = float(np.nanmax(sub_dose))
            min_d = float(np.nanmin(sub_dose))

            if max_d - min_d > 0.05:
                # Enyhe simítás, hogy a film fizikai szemcsézete ne okozzon mikrohurkokat
                smooth_d = gaussian_filter(np.nan_to_num(sub_dose, nan=min_d), sigma=1.8)

                fractions = [0.20, 0.50, 0.80, 0.90]
                levels = [min_d + f * (max_d - min_d) for f in fractions]
                pens = [
                    pg.mkPen((0, 150, 255), width=2.0),     # 20%: Kék
                    pg.mkPen((0, 255, 120), width=2.0),     # 50%: Zöld
                    pg.mkPen((255, 220, 0), width=2.2),     # 80%: Sárga
                    pg.mkPen((255, 40, 40), width=2.5)      # 90%: Piros
                ]

                # Explicit X és Y koordinátaháló megadása a piros keret valós pozíciójával
                sh_y, sh_x = smooth_d.shape
                grid_x = np.arange(x1, x1 + sh_x)
                grid_y = np.arange(y1, y1 + sh_y)

                fig = Figure()
                ax = fig.subplots()
                # grid_x és grid_y átadásával a Matplotlib eleve a helyes globális koordinátákat számolja
                cs = ax.contour(grid_x, grid_y, smooth_d, levels=levels)

                for level_idx, pen in enumerate(pens):
                    paths_list = []
                    if hasattr(cs, 'allsegs') and len(cs.allsegs) > level_idx:
                        paths_list = cs.allsegs[level_idx]
                    elif hasattr(cs, 'collections') and len(cs.collections) > level_idx:
                        col = cs.collections[level_idx]
                        paths_list = [p.vertices for p in col.get_paths()]

                    for seg in paths_list:
                        if len(seg) > 2:
                            x_coords = seg[:, 0]
                            y_coords = seg[:, 1]

                            curve = pg.PlotCurveItem(x=x_coords, y=y_coords, pen=pen)
                            curve.setZValue(5)  # A képrétegek felett
                            self.plot_dose.addItem(curve)
                            self.isocurves.append(curve)

    def _toggle_roi_visibility(self, visible: bool):
        """Elrejti vagy megjeleníti az analízis ellipszist (ROI), miközben a számítások megmaradnak."""
        if hasattr(self, 'roi') and self.roi is not None:
            self.roi.setVisible(visible)

    def _update_blended_view(self):
        """Hardveres, azonnali rétegkeverés opacitással (nem fehéredik ki, nem kavarja meg a színskálát)."""
        if self.current_dose_map is None:
            return

        alpha = self.slider_blend.value() / 100.0

        # Frissítjük az alsó nyers réteget, ha van betöltve
        if self.current_raw_img is not None:
            self.img_item_raw_bg.setImage(self.current_raw_img, autoLevels=False)
            self.img_item_raw_bg.setLevels([0, 255])
        
        # A felső dózisréteg átlátszóságának állítása
        self.img_item_dose.setOpacity(alpha)

    # =========================================================================
    # ACTIONS & SLOTS
    # =========================================================================
    def create_project_and_launch_scanner(self):
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder for the New Project", 
            str(self.project_dir.parent) if self.project_dir else str(Path.home())
        )
        if not parent_dir: return

        from PyQt5.QtWidgets import QInputDialog
        default_name = f"LWFA_Shot_{datetime.datetime.now().strftime('%Y%m%d')}"
        project_name, ok = QInputDialog.getText(
            self, "New Project Name", "Enter folder name for the new experiment/shot:",
            text=default_name
        )
        if not ok or not project_name.strip(): return

        new_project_path = Path(parent_dir) / project_name.strip()
        self.project_dir = new_project_path
        self.raw_dir = self.project_dir / "raw"
        self.processed_dir = self.project_dir / "processed"
        self.analysis_dir = self.project_dir / "analysis"

        try:
            for d in [self.raw_dir, self.processed_dir, self.analysis_dir]:
                d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Directory Error", f"Failed to create directories:\n{e}")
            return

        self.lbl_proj_path.setText(f"Active Project:\n{self.project_dir}")
        self.lbl_proj_path.setStyleSheet("color: #2b8a3e; font-weight: bold; font-size: 11px;")

        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=str(self.processed_dir))
        else:
            self.engine.output_folder = str(self.processed_dir)

        self.refresh_file_lists()

        scanner_script = Path(__file__).parent / "scan.py"
        if not scanner_script.exists():
            QMessageBox.warning(self, "Scanner Missing", f"Could not find '{scanner_script.name}'.")
            return

        try:
            subprocess.Popen([sys.executable, str(scanner_script), str(self.raw_dir)])
            QMessageBox.information(
                self, "Project Created", 
                f"Project directory ready:\n{self.project_dir}\n\nScanner launched pointing to 'raw' folder."
            )
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to open scanner:\n{e}")

    def _invalidate_blank_rgb(self):
        self.global_blank_rgb = None

    def open_blank_roi_dialog(self):
        blank_name = self.combo_blank_film.currentText()
        if not blank_name:
            QMessageBox.warning(self, "Error", "No unexposed film selected!")
            return
        blank_path = str(self.raw_dir / blank_name)
        dialog = BlankRoiDialog(blank_path, self)
        
        if dialog.exec_() == QDialog.Accepted and dialog.mean_rgb is not None:
            self.global_blank_rgb = dialog.mean_rgb
            QMessageBox.information(
                self, "Success", 
                f"Blank reference ROI stored globally!\n"
                f"Mean RGB: R={self.global_blank_rgb[0]:.1f}, G={self.global_blank_rgb[1]:.1f}, B={self.global_blank_rgb[2]:.1f}"
            )

    def _on_lut_levels_changed(self):
        if self.current_dose_map is not None and self.slider_blend.value() == 100:
            levels = self.colorbar.getLevels()
            self.img_item_dose.setLevels(levels, update=True)

    def _on_lut_gradient_changed(self):
        if self.current_dose_map is not None and self.slider_blend.value() == 100:
            lut = self.colorbar.getLookupTable(img=self.current_dose_map, n=256)
            self.img_item_dose.setLookupTable(lut, update=True)

    def save_dose_snapshot(self):
        if self.current_dose_map is None:
            QMessageBox.warning(self, "No Dose Map", "Please load a dose map first!")
            return

        from PyQt5.QtWidgets import QInputDialog
        theme_choice, ok = QInputDialog.getItem(
            self, "Export Figure Theme", 
            "Select theme for publication figure:",
            ["White Background (Publication Standard)", "Dark Background (Screen/Slide Standard)"],
            0, False
        )
        if not ok:
            return

        use_white_mode = "White" in theme_choice

        default_dir = str(self.analysis_dir) if self.analysis_dir else ""
        mode_suffix = "white" if use_white_mode else "dark"
        default_name = f"{self.current_film_name}_composite_{mode_suffix}.png"
        default_path = os.path.join(default_dir, default_name) if default_dir else default_name

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save High-Resolution Publication Figure", 
            default_path, "High-Res PNG (*.png);;High-Res TIFF (*.tif *.tiff);;PDF Document (*.pdf)"
        )
        if not file_path:
            return

        try:
            self.btn_save_snapshot.setVisible(False)
            
            # 1. Export téma és vonalvastagságok beállítása
            self._apply_export_theme(white_mode=use_white_mode, is_export=True)
            self.update_profiles()
            self.update_roi_stats()
            QApplication.processEvents()

            SCALE_FACTOR = 3.2  # 300 DPI szorzó

            if file_path.lower().endswith('.pdf'):
                pdf_writer = QPdfWriter(file_path)
                pdf_writer.setResolution(300)
                pdf_writer.setPageSize(QPageSize(QPageSize.A4))
                painter = QPainter(pdf_writer)
                # Egyetlen hívással lerendereljük a teljes kompozit konténert
                self.composite_export_container.render(painter)
                painter.end()
            else:
                # 300 DPI kép előállítása
                src_size = self.composite_export_container.size()
                target_size = src_size * SCALE_FACTOR
                
                image = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
                dpm = int(300 / 0.0254)
                image.setDotsPerMeterX(dpm)
                image.setDotsPerMeterY(dpm)
                image.fill(Qt.white if use_white_mode else Qt.black)

                painter = QPainter(image)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                painter.scale(SCALE_FACTOR, SCALE_FACTOR)
                
                # Egyetlen, atomi hívással lerendereljük a teljes közös konténert (nyers film + műszerfal)
                self.composite_export_container.render(painter)
                painter.end()
                
                image.save(file_path)

            # 2. Visszaállítás képernyő-módba
            self._apply_export_theme(white_mode=self.is_white_mode, is_export=False)
            self.update_profiles()
            self.update_roi_stats()
            self.btn_save_snapshot.setVisible(True)
            QApplication.processEvents()

            QMessageBox.information(
                self, "Saved", 
                f"Composite publication figure ({'White' if use_white_mode else 'Dark'}) saved (300 DPI):\n{file_path}"
            )
        except Exception as e:
            self._apply_export_theme(white_mode=self.is_white_mode, is_export=False)
            self.btn_save_snapshot.setVisible(True)
            QMessageBox.critical(self, "Error", f"Snapshot failed:\n{e}")

    # =========================================================================
    # PROJECT & CALIBRATION MANAGEMENT
    # =========================================================================
    def select_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not folder: return
        selected_path = Path(folder)
        self.project_dir = selected_path.parent if selected_path.name.lower() == "raw" else selected_path

        self.raw_dir = self.project_dir / "raw"
        self.processed_dir = self.project_dir / "processed"
        self.analysis_dir = self.project_dir / "analysis"

        for d in [self.raw_dir, self.processed_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.lbl_proj_path.setText(f"Active Project:\n{self.project_dir}")
        self.lbl_proj_path.setStyleSheet("color: #2b8a3e; font-weight: bold; font-size: 11px;")
        
        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=str(self.processed_dir))
        else:
            self.engine.output_folder = str(self.processed_dir)
        self.refresh_file_lists()

    def refresh_file_lists(self):
        if not self.project_dir: return
        self.list_raw_files.clear()
        self.combo_blank_film.clear()
        raw_files = sorted([f for f in self.raw_dir.iterdir() if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]])
        for f in raw_files:
            self.list_raw_files.addItem(f.name)
            self.combo_blank_film.addItem(f.name)
        self.list_proc_files.clear()
        proc_files = sorted(list(self.processed_dir.glob("*_dose.npy")))
        for f in proc_files:
            self.list_proc_files.addItem(f.name)

    def select_calibration_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if not folder: return
        self.calib_dir = Path(folder)
        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=str(self.calib_dir))
        master_json = self.calib_dir / "calibration_master.json"
        if master_json.exists():
            try:
                self.engine.load_calibration(str(master_json))
                self.lbl_calib_status.setText(f"✅ Loaded:\n{master_json.name}")
                self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load calibration:\n{e}")
        else:
            self.lbl_calib_status.setText(f"📂 Selected ({self.calib_dir.name}).")
            self.lbl_calib_status.setStyleSheet("color: #f0ad4e; font-weight: bold;")

    def load_existing_calibration(self):
        start_dir = str(self.calib_dir) if self.calib_dir else ""
        json_path, _ = QFileDialog.getOpenFileName(self, "Load JSON", start_dir, "JSON Files (*.json)")
        if not json_path: return
        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=os.path.dirname(json_path))
        try:
            self.engine.load_calibration(json_path)
            self.lbl_calib_status.setText(f"✅ Loaded: {os.path.basename(json_path)}")
            self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration error:\n{e}")

    def run_new_calibration(self):
        if not self.calib_dir:
            QMessageBox.warning(self, "Missing Folder", "Select the calibration folder first!")
            return
        try:
            doses = [float(x.strip()) for x in self.input_doses.text().split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Invalid Format", "Enter comma-separated numbers!")
            return
        method = 'mcmc' if "MCMC" in self.combo_calib_method.currentText() else 'standard'
        cleaning = self.chk_calib_clean.isChecked()

        self._set_ui_busy(True)
        self.worker = EngineWorker('calibrate', self.engine, folder=str(self.calib_dir), doses=doses, method=method, cleaning=cleaning)
        self.worker.finished.connect(self._on_calib_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_calib_finished(self, msg):
        self._set_ui_busy(False)
        self.lbl_calib_status.setText(f"✅ Active ({self.calib_dir.name})")
        self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
        QMessageBox.information(self, "Done", msg)

    def convert_films(self, batch_all=False):
        if not self.project_dir or not self.engine:
            QMessageBox.warning(self, "Error", "Open a project folder first!")
            return
        if not self.engine.calib_params:
            QMessageBox.warning(self, "Error", "No calibration loaded!")
            return
        if self.global_blank_rgb is None:
            QMessageBox.warning(self, "Error", "Set the Blank Film ROI first via '🎯 Set ROI'!")
            return
        
        blank_name = self.combo_blank_film.currentText()
        blank_path = str(self.raw_dir / blank_name)

        if batch_all:
            items = [self.list_raw_files.item(i).text() for i in range(self.list_raw_files.count())]
        else:
            items = [item.text() for item in self.list_raw_files.selectedItems()]
        
        if not items:
            QMessageBox.warning(self, "Error", "Select at least one film!")
            return

        file_paths = [str(self.raw_dir / name) for name in items]
        method = 'hybrid' if "hybrid" in self.combo_solver.currentText() else 'fast'
        cleaning = self.chk_conv_clean.isChecked()

        self.engine.output_folder = str(self.processed_dir)
        self._set_ui_busy(True)
        self.worker = EngineWorker('process', self.engine, files=file_paths, blank=blank_path, method=method, cleaning=cleaning, blank_rgb=self.global_blank_rgb)
        self.worker.finished.connect(self._on_process_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_process_finished(self, msg):
        self._set_ui_busy(False)
        self.refresh_file_lists()
        QMessageBox.information(self, "Conversion Complete", msg)

    def _on_worker_error(self, err_msg):
        self._set_ui_busy(False)
        QMessageBox.critical(self, "Error", f"Execution error:\n{err_msg}")

    def _set_ui_busy(self, busy: bool):
        self.progress_bar.setVisible(busy)
        self.btn_run_calib.setEnabled(not busy)
        self.btn_conv_selected.setEnabled(not busy)
        self.btn_conv_all.setEnabled(not busy)
        self.btn_set_blank_roi.setEnabled(not busy)

    # =========================================================================
    # DISPLAY & INTERACTIVE ANALYSIS
    # =========================================================================
    def on_raw_file_clicked(self, item):
        raw_path = self.raw_dir / item.text()
        self.load_and_display_raw(raw_path)
        base_name = raw_path.stem
        dose_path = self.processed_dir / f"{base_name}_dose.npy"
        if dose_path.exists():
            self.load_and_display_dose(dose_path)

    def on_proc_file_clicked(self, item):
        dose_path = self.processed_dir / item.text()
        self.load_and_display_dose(dose_path)
        base_name = item.text().replace("_dose.npy", "")
        for ext in [".tif", ".tiff", ".TIF", ".TIFF"]:
            raw_candidate = self.raw_dir / f"{base_name}{ext}"
            if raw_candidate.exists():
                self.load_and_display_raw(raw_candidate)
                break

    def load_and_display_raw(self, path: Path):
        try:
            img = tifffile.imread(str(path))
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            img = np.flip(img, axis=0)
            if img.dtype == np.uint16:
                img_disp = (img / 256).astype(np.uint8)
            else:
                img_disp = img
            self.current_raw_img = img_disp
            self.img_item_raw.setImage(img_disp)
            self.plot_raw.autoRange()

            if self.current_meta:
                spacing = float(self.current_meta.get("pixel_spacing_mm", 0.0846))
                self.plot_raw.getAxis('bottom').setScale(spacing)
                self.plot_raw.getAxis('left').setScale(spacing)
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to display raw image:\n{e}")

    def load_and_display_dose(self, dose_path: Path):
        try:
            dose_map = np.load(str(dose_path)).astype(np.float32)
            dose_map = np.flip(dose_map, axis=0)
            self.current_dose_map = dose_map
            self.current_film_name = dose_path.stem.replace("_dose", "")

            meta_path = dose_path.with_name(dose_path.name.replace("_dose.npy", "_meta.json"))
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.current_meta = json.load(f)

            min_d, max_d = float(np.min(dose_map)), float(np.max(dose_map))
            if min_d == max_d: max_d = min_d + 1.0

            self.img_item_dose.setImage(dose_map, autoLevels=False)
            self.colorbar.setLevels(min_d, max_d)
            self.img_item_dose.setLevels((min_d, max_d), update=True)
            
            h, w = dose_map.shape
            self.v_line.setPos(w // 2)
            self.h_line.setPos(h // 2)
            self.v_line_raw.setPos(w // 2)
            self.h_line_raw.setPos(h // 2)

            # Oblique line setup
            self.line_roi.movePoint(self.line_roi.getHandles()[0], [int(w * 0.2), int(h * 0.2)])
            self.line_roi.movePoint(self.line_roi.getHandles()[1], [int(w * 0.8), int(h * 0.8)])

            roi_w, roi_h = min(120, w // 4), min(120, h // 4)
            self.roi.setPos([w // 2 - roi_w // 2, h // 2 - roi_h // 2])
            self.roi.setSize([roi_w, roi_h])

            margin_x, margin_y = int(w * 0.05), int(h * 0.05)
            self.film_boundary.setPos([margin_x, margin_y])
            self.film_boundary.setSize([w - 2 * margin_x, h - 2 * margin_y])

            spacing = float(self.current_meta.get("pixel_spacing_mm", 0.0846))
            self.plot_dose.getAxis('bottom').setScale(spacing)
            self.plot_dose.getAxis('left').setScale(spacing)
            self.plot_raw.getAxis('bottom').setScale(spacing)
            self.plot_raw.getAxis('left').setScale(spacing)

            self.plot_dose.autoRange()
            self._toggle_isodose(self.chk_isodose.isChecked())
            self.update_boundary_and_profiles()
            self.update_roi_stats()
        except Exception as e:
            QMessageBox.critical(self, "Dose Map Error", f"Failed to load dose map:\n{e}")

    def update_boundary_and_profiles(self):
        if self.current_dose_map is None: return
        pos = self.film_boundary.pos()
        size = self.film_boundary.size()
        h, w = self.current_dose_map.shape

        x1 = int(np.clip(pos.x(), 0, w - 1))
        y1 = int(np.clip(pos.y(), 0, h - 1))
        x2 = int(np.clip(pos.x() + size.x(), x1 + 1, w))
        y2 = int(np.clip(pos.y() + size.y(), y1 + 1, h))

        active_data = self.current_dose_map[y1:y2, x1:x2]
        if active_data.size > 0:
            min_d, max_d = float(np.min(active_data)), float(np.max(active_data))
            if min_d == max_d: max_d = min_d + 1.0
            self.colorbar.setLevels(min_d, max_d)
            self.img_item_dose.setLevels((min_d, max_d), update=True)
            self.colorbar.setHistogramRange(min_d, max_d, padding=0.05)
        self.update_profiles(x1, x2, y1, y2)

    def _on_crosshair_moved(self, *args):
        # Synced Reticle updating on Raw View
        cx, cy = self.v_line.value(), self.h_line.value()
        self.v_line_raw.setPos(cx)
        self.h_line_raw.setPos(cy)
        self.update_profiles()

    def _on_oblique_moved(self, *args):
        self.update_profiles()

    def update_profiles(self, x1=None, x2=None, y1=None, y2=None):
        if self.current_dose_map is None: return
        h, w = self.current_dose_map.shape
        spacing = float(self.current_meta.get("pixel_spacing_mm", 0.0846))
        band = self.spin_band.value()

        if not isinstance(x1, (int, float, np.integer, np.floating)):
            pos = self.film_boundary.pos()
            size = self.film_boundary.size()
            x1 = int(np.clip(pos.x(), 0, w - 1))
            y1 = int(np.clip(pos.y(), 0, h - 1))
            x2 = int(np.clip(pos.x() + size.x(), x1 + 1, w))
            y2 = int(np.clip(pos.y() + size.y(), y1 + 1, h))

        # 1. Oblique Profile Mode
        if self.combo_prof_mode.currentIndex() == 1:
            handles = self.line_roi.getHandles()
            if len(handles) < 2:
                return

            # A vonalzó két végpontjának lekérdezése (képpont koordinátákban)
            p1 = self.line_roi.mapToParent(handles[0].pos())
            p2 = self.line_roi.mapToParent(handles[1].pos())
            x0, y0 = float(p1.x()), float(p1.y())
            x1_pt, y1_pt = float(p2.x()), float(p2.y())

            # A vonal hossza képpontban és fizikai mm-ben
            dx = x1_pt - x0
            dy = y1_pt - y0
            length_px = np.hypot(dx, dy)
            if length_px < 2:
                self.curve_prof_x.setData([], [])
                return

            num_samples = int(np.ceil(length_px))
            t = np.linspace(0, 1, num_samples)
            cx_line = x0 + t * dx
            cy_line = y0 + t * dy

            # Sávos átlagolás a merőleges normálvektor mentén (Band Width)
            nx = -dy / length_px
            ny = dx / length_px
            half_band = band // 2

            if band > 1 and half_band >= 1:
                offsets = np.arange(-half_band, half_band + 1)
                all_profiles = []
                for off in offsets:
                    sample_x = cx_line + off * nx
                    sample_y = cy_line + off * ny
                    coords = np.vstack([sample_y, sample_x])  # row-major: sor (Y), oszlop (X)
                    sampled = map_coordinates(self.current_dose_map, coords, order=1, mode='nearest')
                    all_profiles.append(sampled)
                prof = np.mean(all_profiles, axis=0)
            else:
                coords = np.vstack([cy_line, cx_line])  # row-major: [Y, X]
                prof = map_coordinates(self.current_dose_map, coords, order=1, mode='nearest')

            # Fizikai távolságtengely kiszámítása mm-ben a vonalzó mentén
            s_axis_mm = np.linspace(0, length_px * spacing, num_samples)

            self.curve_prof_x.setData(s_axis_mm, prof)
            self.curve_prof_fit.setData([], [])
            return

        # 2. Orthogonal Profiles (with Band Averaging)
        c_x = int(np.clip(self.v_line.value(), 0, w - 1))
        c_y = int(np.clip(self.h_line.value(), 0, h - 1))

        # X Profile (Averaged across Band)
        y_low = max(y1, c_y - band // 2)
        y_high = min(y2, c_y + band // 2 + 1)
        if y_high > y_low:
            prof_x = np.mean(self.current_dose_map[y_low:y_high, x1:x2], axis=0)
            if len(prof_x) > 0:
                x_mm = np.arange(x1, x2) * spacing
                self.curve_prof_x.setData(x_mm - x_mm[np.argmax(prof_x)], prof_x, pen=self.pen_prof_x)
            else: self.curve_prof_x.setData([], [])
        else: self.curve_prof_x.setData([], [])

        # Y Profile (Averaged across Band)
        x_low = max(x1, c_x - band // 2)
        x_high = min(x2, c_x + band // 2 + 1)
        if x_high > x_low:
            prof_y = np.mean(self.current_dose_map[y1:y2, x_low:x_high], axis=1)
            if len(prof_y) > 0:
                y_mm = np.arange(y1, y2) * spacing
                self.curve_prof_y.setData(prof_y, y_mm - y_mm[np.argmax(prof_y)], pen=self.pen_prof_y)
            else: self.curve_prof_y.setData([], [])
        else: self.curve_prof_y.setData([], [])

    # =========================================================================
    # ADVANCED LWFA BEAM METRICS & 2D GAUSSIAN FIT
    # =========================================================================
    def update_roi_stats(self):
        if self.current_dose_map is None: return
        roi_data = self.roi.getArrayRegion(self.current_dose_map, self.img_item_dose)
        if roi_data is None or roi_data.size == 0: return

        rh, rw = roi_data.shape
        if rh < 3 or rw < 3: return
        
        yy, xx = np.ogrid[:rh, :rw]
        cy, cx = (rh - 1) / 2.0, (rw - 1) / 2.0
        ry, rx = rh / 2.0, rw / 2.0
        ellipse_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0

        valid_pixels = roi_data[ellipse_mask]
        if valid_pixels.size == 0: return

        spacing = float(self.current_meta.get("pixel_spacing_mm", 0.0846))
        pos = self.roi.pos()

        # 1. Alapvető dózisstatisztikák
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        num_px = int(valid_pixels.size)
        area_mm2 = num_px * (spacing ** 2)
        integral_dose = float(np.sum(valid_pixels) * (spacing ** 2))

        # Térbeli dózisszórás (százalékban a fantomon belüli homogenitásra)
        spatial_cov_pct = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0

        # 2. Orvosfizikai lefedettség: D90% és V95%
        # D90%: A dózis, amit a pixelek 90%-a meghalad (10. percentilis)
        d90_val = float(np.percentile(valid_pixels, 10))
        # V95%: A maximális dózis 95%-át elérő pixelek aránya [%]
        v95_count = np.sum(valid_pixels >= 0.95 * max_val)
        v95_val = float((v95_count / num_px) * 100.0) if num_px > 0 else 0.0

        # 3. Peak és Súlypont (CoM) koordináták
        max_idx = np.unravel_index(np.argmax(roi_data * ellipse_mask), roi_data.shape)
        peak_x_mm = (pos.x() + max_idx[1]) * spacing
        peak_y_mm = (pos.y() + max_idx[0]) * spacing

        masked_d = np.where(ellipse_mask, roi_data, 0.0)
        tot_mass = np.sum(masked_d)
        if tot_mass > 0:
            y_indices, x_indices = np.indices(roi_data.shape)
            com_x = np.sum(x_indices * masked_d) / tot_mass
            com_y = np.sum(y_indices * masked_d) / tot_mass
            com_x_mm = (pos.x() + com_x) * spacing
            com_y_mm = (pos.y() + com_y) * spacing
            
            # Spatial STD (Térbeli kiterjedés / szórás mm-ben)
            var_x = np.sum(((x_indices - com_x) ** 2) * masked_d) / tot_mass
            var_y = np.sum(((y_indices - com_y) ** 2) * masked_d) / tot_mass
            spatial_std_mm = np.sqrt(var_x + var_y) * spacing
        else:
            com_x_mm, com_y_mm = peak_x_mm, peak_y_mm
            spatial_std_mm = 0.0

        # 4. FWHM, Penumbra, Flatness és Symmetry
        prof_x = np.mean(masked_d, axis=0)
        prof_y = np.mean(masked_d, axis=1)
        fwhm_x = self._calc_fwhm(prof_x, spacing)
        fwhm_y = self._calc_fwhm(prof_y, spacing)
        penumbra_mm = self._calc_penumbra(prof_x, prof_y, spacing)
        flatness_pct, symmetry_pct = self._calc_flatness_symmetry(prof_x)

        # 5. 2D Aszimmetrikus Gauss-illesztés
        fit_amp, sig_x, sig_y, theta_deg = self._fit_2d_gaussian(roi_data, ellipse_mask, spacing)

        # 6. Konverziós módszer kiolvasása a metaadatokból
        method_str = self.current_meta.get("method", "Standard / Fast")
        if "hybrid" in str(method_str).lower():
            method_disp = "Hybrid (Pixel-wise Opt)"
        elif "fast" in str(method_str).lower():
            method_disp = "Fast Vectorized"
        else:
            method_disp = str(method_str).capitalize()

        # Címkék frissítése
        self.lbl_beam_peak.setText(f"<b>Peak:</b> {max_val:.2f} Gy @ ({peak_x_mm:.1f}, {peak_y_mm:.1f}) mm")
        self.lbl_beam_com.setText(f"<b>CoM:</b> ({com_x_mm:.1f}, {com_y_mm:.1f}) mm")
        self.lbl_beam_fwhm.setText(f"<b>FWHM (X/Y):</b> {fwhm_x} / {fwhm_y} mm")
        self.lbl_roi_area.setText(f"<b>ROI Area:</b> {area_mm2:.1f} mm²")

        self.lbl_fit_amp.setText(f"<b>Fit Amp:</b> {fit_amp:.2f} Gy")
        self.lbl_fit_sigma.setText(f"<b>Spread:</b> {sig_x:.2f}, {sig_y:.2f} mm")
        self.lbl_fit_theta.setText(f"<b>Tilt (θ):</b> {theta_deg:.1f}°")
        self.lbl_total_dose.setText(f"<b>Integral Dose:</b> {integral_dose:.1f} Gy·mm²")

        self.lbl_dose_stats.setText(f"<b>Mean Dose:</b> {mean_val:.2f} Gy")
        self.lbl_dose_std.setText(f"<b>Spatial STD:</b> ±{std_val:.2f} Gy ({spatial_cov_pct:.1f}%)")
        self.lbl_dose_minmax.setText(f"<b>Min / Max:</b> {min_val:.2f} / {max_val:.2f} Gy")
        self.lbl_d90_val.setText(f"<b>D90%:</b> {d90_val:.2f} Gy | <b>V95%:</b> {v95_val:.1f}%")

        self.lbl_penumbra.setText(f"<b>Penumbra (80-20):</b> {penumbra_mm}")
        self.lbl_flatness.setText(f"<b>Flatness:</b> {flatness_pct}")
        self.lbl_symmetry.setText(f"<b>Symmetry:</b> {symmetry_pct}")
        self.lbl_conv_method.setText(f"<b>Method:</b> {method_disp}")

        # 7. Hisztogram és DAH diagram frissítése
        y, x = np.histogram(valid_pixels, bins=50)
        bin_centers = (x[:-1] + x[1:]) / 2
        y_norm = y / np.max(y) * 100.0 if np.max(y) > 0 else y
        self.curve_roi_hist.setData(x, y_norm, pen=self.pen_hist)

        cum_y = np.cumsum(y[::-1])[::-1]
        cum_y_norm = cum_y / np.max(cum_y) * 100.0 if np.max(cum_y) > 0 else cum_y
        self.curve_roi_dah.setData(bin_centers, cum_y_norm, pen=self.pen_dah)

    def _calc_penumbra(self, prof_x, prof_y, spacing):
        """Kiszámolja a 80%-20% orvosfizikai penumbrát az átlagolt profilokból."""
        try:
            penumbras = []
            for prof in [prof_x, prof_y]:
                p_max = np.max(prof)
                if p_max <= 0: continue
                val_80 = 0.80 * p_max
                val_20 = 0.20 * p_max
                idx_80 = np.where(prof >= val_80)[0]
                idx_20 = np.where(prof >= val_20)[0]
                if len(idx_80) > 0 and len(idx_20) > 0:
                    # Bal oldali penumbra
                    left_p = abs(idx_80[0] - idx_20[0]) * spacing
                    # Jobb oldali penumbra
                    right_p = abs(idx_20[-1] - idx_80[-1]) * spacing
                    penumbras.append((left_p + right_p) / 2.0)
            if penumbras:
                return f"{np.mean(penumbras):.2f} mm"
            return "-"
        except Exception:
            return "-"

    def _calc_flatness_symmetry(self, prof):
        """Kiszámolja a klinikai/nyalábteszt Flatness és Symmetry értékeket (AAPM/IEC szabvány szerint)."""
        try:
            p_max = np.max(prof)
            if p_max <= 0: return "-", "-"
            # 50% FWHM határok meghatározása
            idx_50 = np.where(prof >= 0.50 * p_max)[0]
            if len(idx_50) < 5: return "-", "-"
            
            # A 80%-os belső magterület (flattened area)
            left_edge, right_edge = idx_50[0], idx_50[-1]
            f_width = right_edge - left_edge
            inner_left = int(left_edge + 0.10 * f_width)
            inner_right = int(right_edge - 0.10 * f_width)
            
            central_region = prof[inner_left:inner_right]
            if len(central_region) < 3: return "-", "-"
            
            d_max = np.max(central_region)
            d_min = np.min(central_region)
            flatness = ((d_max - d_min) / (d_max + d_min)) * 100.0

            # Szimmetria a csúcs körüli tükrözéssel
            c_idx = np.argmax(prof)
            half_span = min(c_idx - left_edge, right_edge - c_idx)
            if half_span > 2:
                left_wing = prof[c_idx - half_span:c_idx]
                right_wing = prof[c_idx + 1:c_idx + 1 + half_span][::-1]
                sym_diff = np.abs(left_wing - right_wing)
                symmetry = (np.max(sym_diff) / prof[c_idx]) * 100.0
                return f"{flatness:.1f} %", f"{symmetry:.1f} %"
            return f"{flatness:.1f} %", "-"
        except Exception:
            return "-", "-"

    def _calc_fwhm(self, prof, spacing):
        try:
            half_max = np.max(prof) / 2.0
            idx = np.where(prof >= half_max)[0]
            if len(idx) > 1:
                return f"{(idx[-1] - idx[0]) * spacing:.2f}"
            return "-"
        except: return "-"

    def _fit_2d_gaussian(self, data, mask, spacing):
        try:
            h, w = data.shape
            x = np.arange(w) * spacing
            y = np.arange(h) * spacing
            X, Y = np.meshgrid(x, y)

            data_flat = data.ravel()
            x_flat, y_flat = X.ravel(), Y.ravel()
            mask_flat = mask.ravel()

            fit_x = (x_flat[mask_flat], y_flat[mask_flat])
            fit_y = data_flat[mask_flat]

            # Initial parameters: amp, xo, yo, sigma_x, sigma_y, theta, offset
            amp0 = float(np.max(fit_y))
            xo0 = float(w * spacing / 2.0)
            yo0 = float(h * spacing / 2.0)
            p0 = [amp0, xo0, yo0, xo0 / 3.0, yo0 / 3.0, 0.0, float(np.min(fit_y))]

            bounds = (
                [0, 0, 0, 0.05, 0.05, -np.pi, 0],
                [amp0 * 2.0, w * spacing, h * spacing, w * spacing, h * spacing, np.pi, amp0]
            )

            popt, _ = curve_fit(twoD_Gaussian, fit_x, fit_y, p0=p0, bounds=bounds, maxfev=800)
            amp, xo, yo, sx, sy, th, off = popt
            return amp, sx, sy, np.degrees(th)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PrismDosimetryGUI()
    window.show()
    sys.exit(app.exec_())