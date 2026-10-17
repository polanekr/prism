import sys
import os
import json
from pathlib import Path
import numpy as np
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLabel, QFileDialog, QListWidget, 
    QComboBox, QCheckBox, QLineEdit, QGroupBox, QTabWidget, 
    QSplitter, QMessageBox, QProgressBar, QAbstractItemView,
    QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Force pyqtgraph to use PyQt5
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg

# Import custom dosimetry engine
from dosimetry import GafchromicEngine, BAYES_AVAILABLE

# PyQtGraph global configuration
pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)


# =============================================================================
# BLANK FILM ROI SELECTION DIALOG (Golyóálló, 16-bites számítással)
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
            "The mean RGB of this region will be used as a uniform background reference<br>"
            "to prevent artifacts (like scratches or text) from burning into the dose maps."
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
            raw_img = np.flip(raw_img, axis=0) # Függőleges tükrözés
            
            # Ezt használjuk a PONTOS matematikai átlagszámításhoz (16-bit / eredeti)
            self.raw_img_data = raw_img.astype(np.float64)

            # Ezt csak a képernyőn való MEGJELENÍTÉSHEZ használjuk (8-bit)
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
            # FIGYELEM: Az eredeti, RAW (pl. 16-bites) képből olvassuk ki a pixeleket!
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
                self.finished.emit("Calibration completed successfully and saved to the calibration folder!")

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
        self.setWindowTitle("PRISM - Gafchromic EBT3 Film Dosimetry Suite")
        self.resize(1650, 960)

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
        
        # Globális változó a Blank ROI átlagának tárolására
        self.global_blank_rgb = None

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
        self.tabs.addTab(self.viewer_tab, "🔍 2D Dose Map & Profiles")

        self.gamma_tab = self._create_placeholder_tab("Gamma Analysis & Profiles", "WIP")
        self.tabs.addTab(self.gamma_tab, "📊 Gamma Analysis (WIP)")

        self.tab_3d = self._create_placeholder_tab("3D Dose Volume Reconstruction", "WIP")
        self.tabs.addTab(self.tab_3d, "🧊 3D Reconstruction (WIP)")

        splitter.setSizes([390, 1260])

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        proj_group = QGroupBox("1. Project Session")
        proj_layout = QVBoxLayout(proj_group)
        self.btn_open_proj = QPushButton("📁 Open / Create Project Folder")
        self.btn_open_proj.clicked.connect(self.select_project_folder)
        self.btn_refresh = QPushButton("🔄 Refresh File Lists")
        self.btn_refresh.clicked.connect(self.refresh_file_lists)
        self.lbl_proj_path = QLabel("No project selected")
        self.lbl_proj_path.setStyleSheet("color: gray; font-size: 11px;")
        proj_layout.addWidget(self.btn_open_proj)
        proj_layout.addWidget(self.btn_refresh)
        proj_layout.addWidget(self.lbl_proj_path)
        layout.addWidget(proj_group)

        calib_group = QGroupBox("2. Calibration (Shared Folder)")
        calib_layout = QVBoxLayout(calib_group)
        self.btn_sel_calib_dir = QPushButton("📂 Select Calibration Folder")
        self.btn_sel_calib_dir.clicked.connect(self.select_calibration_folder)
        self.lbl_calib_status = QLabel("Status: No calibration loaded")
        self.lbl_calib_status.setStyleSheet("color: #d9534f; font-weight: bold;")
        calib_layout.addWidget(self.btn_sel_calib_dir)
        calib_layout.addWidget(self.lbl_calib_status)
        calib_layout.addWidget(QLabel("Doses [Gy] (comma-separated, 0 = blank):"))
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

        conv_group = QGroupBox("3. Conversion (raw -> processed)")
        conv_layout = QVBoxLayout(conv_group)
        conv_layout.addWidget(QLabel("Raw scanned films (Ctrl/Shift for multi-select):"))
        self.list_raw_files = QListWidget()
        self.list_raw_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_raw_files.itemClicked.connect(self.on_raw_file_clicked)
        conv_layout.addWidget(self.list_raw_files)
        
        conv_layout.addWidget(QLabel("Unexposed (Blank / 0 Gy) reference film:"))
        
        # --- ÚJ: Blank ROI Kijelölő Gomb ---
        blank_hbox = QHBoxLayout()
        self.combo_blank_film = QComboBox()
        self.combo_blank_film.currentIndexChanged.connect(self._invalidate_blank_rgb)
        
        self.btn_set_blank_roi = QPushButton("🎯 Set ROI")
        self.btn_set_blank_roi.setToolTip("Select a clean area on the blank film to use as global background reference")
        self.btn_set_blank_roi.clicked.connect(self.open_blank_roi_dialog)
        
        blank_hbox.addWidget(self.combo_blank_film)
        blank_hbox.addWidget(self.btn_set_blank_roi)
        conv_layout.addLayout(blank_hbox)
        # -----------------------------------

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

        proc_group = QGroupBox("4. Processed Dose Maps (processed)")
        proc_layout = QVBoxLayout(proc_group)
        self.list_proc_files = QListWidget()
        self.list_proc_files.itemClicked.connect(self.on_proc_file_clicked)
        proc_layout.addWidget(self.list_proc_files)
        layout.addWidget(proc_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        return panel

    def _create_viewer_tab(self):
        tab = QWidget()
        main_v_layout = QVBoxLayout(tab)
        images_splitter = QSplitter(Qt.Horizontal)

        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(QLabel("<b>Raw Scanned Film (RGB)</b>"))
        self.glw_raw = pg.GraphicsLayoutWidget()
        self.plot_raw = self.glw_raw.addPlot()
        self.plot_raw.setAspectLocked(True)
        self.plot_raw.setLabel('bottom', 'X [mm]')
        self.plot_raw.setLabel('left', 'Y [mm]')
        self.img_item_raw = pg.ImageItem()
        self.plot_raw.addItem(self.img_item_raw)
        raw_layout.addWidget(self.glw_raw)
        images_splitter.addWidget(raw_widget)

        dose_widget = QWidget()
        dose_layout = QVBoxLayout(dose_widget)
        dose_layout.setContentsMargins(0, 0, 0, 0)
        
        header_hbox = QHBoxLayout()
        header_hbox.addWidget(QLabel("<b>Converted Dose Map [Gy] (Drag crosshair & elliptical ROI!)</b>"))
        self.btn_save_snapshot = QPushButton("📸 Save Dose Map Snapshot (with Overlay)")
        self.btn_save_snapshot.clicked.connect(self.save_dose_snapshot)
        header_hbox.addStretch()
        header_hbox.addWidget(self.btn_save_snapshot)
        dose_layout.addLayout(header_hbox)

        self.glw_dose = pg.GraphicsLayoutWidget()
        
        self.plot_prof_y = self.glw_dose.addPlot(row=0, col=0)
        self.plot_prof_y.setLabel('bottom', 'Dose [Gy]')
        self.plot_prof_y.setLabel('left', 'Y (CAX) [mm]')
        self.plot_prof_y.setMaximumWidth(185)
        self.curve_prof_y = self.plot_prof_y.plot(pen=pg.mkPen('c', width=2))

        self.plot_dose = self.glw_dose.addPlot(row=0, col=1)
        self.plot_dose.setLabel('bottom', 'X [mm]')
        self.plot_dose.setLabel('left', 'Y [mm]')
        self.plot_dose.setAspectLocked(True)
        self.img_item_dose = pg.ImageItem(autoDownsample=False)
        self.plot_dose.addItem(self.img_item_dose)

        self.colorbar = pg.HistogramLUTItem()
        self.colorbar.setImageItem(self.img_item_dose)
        self.colorbar.gradient.loadPreset('viridis')
        self.glw_dose.addItem(self.colorbar, row=0, col=2)
        self.colorbar.sigLevelsChanged.connect(self._on_lut_levels_changed)
        self.colorbar.sigLookupTableChanged.connect(self._on_lut_gradient_changed)

        self.canvas_stats_label = pg.LabelItem(justify='left')
        self.canvas_stats_label.setText("<b>ROI Stats:</b><br>No data", color='#EEEEEE')
        self.glw_dose.addItem(self.canvas_stats_label, row=1, col=0)

        self.plot_prof_x = self.glw_dose.addPlot(row=1, col=1)
        self.plot_prof_x.setLabel('left', 'Dose [Gy]')
        self.plot_prof_x.setLabel('bottom', 'X (CAX) [mm]')
        self.plot_prof_x.setMaximumHeight(185)
        self.curve_prof_x = self.plot_prof_x.plot(pen=pg.mkPen('y', width=2))

        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('y', width=1.5, style=Qt.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=Qt.DashLine))
        self.plot_dose.addItem(self.v_line, ignoreBounds=True)
        self.plot_dose.addItem(self.h_line, ignoreBounds=True)
        self.v_line.sigPositionChanged.connect(self._on_crosshair_moved)
        self.h_line.sigPositionChanged.connect(self._on_crosshair_moved)

        self.film_boundary = pg.RectROI([10, 10], [100, 100], pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.film_boundary.addScaleHandle([1, 1], [0, 0])
        self.film_boundary.addScaleHandle([0, 0], [1, 1])
        self.plot_dose.addItem(self.film_boundary)
        self.film_boundary.sigRegionChanged.connect(self.update_boundary_and_profiles)

        self.roi = pg.EllipseROI([50, 50], [100, 100], pen=pg.mkPen('w', width=2))
        self.plot_dose.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_roi_stats)

        dose_layout.addWidget(self.glw_dose)
        images_splitter.addWidget(dose_widget)
        images_splitter.setSizes([480, 780])
        main_v_layout.addWidget(images_splitter, stretch=3)

        roi_group = QGroupBox("Elliptical ROI Spatial Statistics & Uncertainty")
        roi_h_layout = QHBoxLayout(roi_group)

        stats_widget = QWidget()
        stats_grid = QGridLayout(stats_widget)
        self.lbl_roi_mean = QLabel("<b>Mean Dose:</b> - Gy")
        self.lbl_roi_std = QLabel("<b>Spatial Std Dev:</b> - Gy")
        self.lbl_roi_median = QLabel("<b>Median:</b> - Gy")
        self.lbl_roi_min = QLabel("<b>Minimum:</b> - Gy")
        self.lbl_roi_max = QLabel("<b>Maximum:</b> - Gy")
        self.lbl_roi_pixels = QLabel("<b>ROI Area:</b> - mm² (- px)")
        stats_grid.addWidget(self.lbl_roi_mean, 0, 0)
        stats_grid.addWidget(self.lbl_roi_std, 1, 0)
        stats_grid.addWidget(self.lbl_roi_median, 2, 0)
        stats_grid.addWidget(self.lbl_roi_min, 0, 1)
        stats_grid.addWidget(self.lbl_roi_max, 1, 1)
        stats_grid.addWidget(self.lbl_roi_pixels, 2, 1)
        roi_h_layout.addWidget(stats_widget, stretch=1)

        self.plot_roi_hist = pg.PlotWidget(title="ROI Dose Distribution (Histogram)")
        self.plot_roi_hist.setLabel('bottom', 'Dose [Gy]')
        self.plot_roi_hist.setLabel('left', 'Count (Pixels)')
        self.curve_roi_hist = self.plot_roi_hist.plot(stepMode="center", fillLevel=0, fillOutline=True, brush=(0, 150, 255, 100))
        roi_h_layout.addWidget(self.plot_roi_hist, stretch=2)

        main_v_layout.addWidget(roi_group, stretch=1)
        return tab

    def _create_placeholder_tab(self, title, description):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        lbl = QLabel(f"<h2>{title}</h2><p>{description}</p>")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        return tab

    # =========================================================================
    def _invalidate_blank_rgb(self):
        """Törli a memóriából a Blank ROI-t, ha a felhasználó másik filmet választ a legördülőből."""
        self.global_blank_rgb = None

    def open_blank_roi_dialog(self):
        """Megnyitja az ablakot, ahol a felhasználó kijelölheti az üres film referencia területét."""
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
                f"Blank reference ROI saved globally for this session!\n"
                f"Mean RGB: R={self.global_blank_rgb[0]:.1f}, G={self.global_blank_rgb[1]:.1f}, B={self.global_blank_rgb[2]:.1f}"
            )

    # =========================================================================
    def _on_lut_levels_changed(self):
        if self.current_dose_map is not None:
            levels = self.colorbar.getLevels()
            self.img_item_dose.setLevels(levels, update=True)

    def _on_lut_gradient_changed(self):
        if self.current_dose_map is not None:
            lut = self.colorbar.getLookupTable(img=self.current_dose_map, n=256)
            self.img_item_dose.setLookupTable(lut, update=True)

    def save_dose_snapshot(self):
        if self.current_dose_map is None:
            QMessageBox.warning(self, "No Dose Map", "Please load a converted dose map first!")
            return
        default_dir = str(self.analysis_dir) if self.analysis_dir else ""
        default_name = f"{self.current_film_name}_snapshot.png"
        default_path = os.path.join(default_dir, default_name) if default_dir else default_name

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", default_path, "PNG Image (*.png);;TIFF (*.tif *.tiff)")
        if file_path:
            try:
                pixmap = self.glw_dose.grab()
                pixmap.save(file_path)
                QMessageBox.information(self, "Saved", f"Snapshot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save snapshot:\n{e}")

    # =========================================================================
    def select_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select or Create Project Folder")
        if not folder: return
        selected_path = Path(folder)
        if selected_path.name.lower() == "raw":
            self.project_dir = selected_path.parent
        else:
            self.project_dir = selected_path

        self.raw_dir = self.project_dir / "raw"
        self.processed_dir = self.project_dir / "processed"
        self.analysis_dir = self.project_dir / "analysis"

        for d in [self.raw_dir, self.processed_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.lbl_proj_path.setText(f"Active Project:\n{self.project_dir}")
        self.lbl_proj_path.setStyleSheet("color: #2b8a3e; font-weight: bold; font-size: 11px;")
        
        # --- JAVÍTOTT RÉSZ ---
        # Ha már létezik a feldolgozó motor (benne a kalibrációval), 
        # NE hozzunk létre újat, csak frissítsük a kimeneti mappát!
        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=str(self.processed_dir))
        else:
            self.engine.output_folder = str(self.processed_dir)
        # ---------------------
        
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
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Films Folder")
        if not folder: return
        self.calib_dir = Path(folder)
        if self.engine is None:
            self.engine = GafchromicEngine(output_folder=str(self.calib_dir))
        master_json = self.calib_dir / "calibration_master.json"
        if master_json.exists():
            try:
                self.engine.load_calibration(str(master_json))
                self.lbl_calib_status.setText(f"✅ Auto-loaded:\n{master_json.name}")
                self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load calibration:\n{e}")
        else:
            self.lbl_calib_status.setText(f"📂 Selected ({self.calib_dir.name}).\nRun new or load JSON!")
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
            QMessageBox.warning(self, "Invalid Format", "Enter numbers separated by commas!")
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
        self.lbl_calib_status.setText(f"✅ Active Calibration ({self.calib_dir.name})")
        self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold;")
        QMessageBox.information(self, "Done", msg)

    def convert_films(self, batch_all=False):
        if not self.project_dir or not self.engine:
            QMessageBox.warning(self, "Error", "Please open a project folder first!")
            return
        if not self.engine.calib_params:
            QMessageBox.warning(self, "Error", "No calibration loaded!")
            return
        
        # --- ELLENŐRZÉS: Be van állítva a Blank ROI? ---
        if self.global_blank_rgb is None:
            QMessageBox.warning(self, "Error", "Please set the Blank Film ROI first by clicking the '🎯 Set ROI' button!")
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
        
        # A konverternek most már garantáltan átadjuk a 16-bites RGB átlagot!
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
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")

    def _set_ui_busy(self, busy: bool):
        self.progress_bar.setVisible(busy)
        self.btn_run_calib.setEnabled(not busy)
        self.btn_conv_selected.setEnabled(not busy)
        self.btn_conv_all.setEnabled(not busy)
        self.btn_set_blank_roi.setEnabled(not busy)

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
        self.update_profiles()

    def update_profiles(self, x1=None, x2=None, y1=None, y2=None):
        if self.current_dose_map is None: return
        h, w = self.current_dose_map.shape
        spacing = float(self.current_meta.get("pixel_spacing_mm", 0.0846))

        if not isinstance(x1, (int, float, np.integer, np.floating)):
            pos = self.film_boundary.pos()
            size = self.film_boundary.size()
            x1 = int(np.clip(pos.x(), 0, w - 1))
            y1 = int(np.clip(pos.y(), 0, h - 1))
            x2 = int(np.clip(pos.x() + size.x(), x1 + 1, w))
            y2 = int(np.clip(pos.y() + size.y(), y1 + 1, h))

        c_x = int(np.clip(self.v_line.value(), 0, w - 1))
        c_y = int(np.clip(self.h_line.value(), 0, h - 1))

        if y1 <= c_y < y2:
            prof_x = self.current_dose_map[c_y, x1:x2]
            if len(prof_x) > 0:
                x_mm = np.arange(x1, x2) * spacing
                x_mm_shifted = x_mm - x_mm[np.argmax(prof_x)]
                self.curve_prof_x.setData(x_mm_shifted, prof_x)
            else: self.curve_prof_x.setData([], [])
        else: self.curve_prof_x.setData([], [])

        if x1 <= c_x < x2:
            prof_y = self.current_dose_map[y1:y2, c_x]
            if len(prof_y) > 0:
                y_mm = np.arange(y1, y2) * spacing
                y_mm_shifted = y_mm - y_mm[np.argmax(prof_y)]
                self.curve_prof_y.setData(prof_y, y_mm_shifted)
            else: self.curve_prof_y.setData([], [])
        else: self.curve_prof_y.setData([], [])

    def update_roi_stats(self):
        if self.current_dose_map is None: return
        roi_data = self.roi.getArrayRegion(self.current_dose_map, self.img_item_dose)
        if roi_data is None or roi_data.size == 0: return

        rh, rw = roi_data.shape
        if rh < 2 or rw < 2: return
        
        yy, xx = np.ogrid[:rh, :rw]
        cy, cx = (rh - 1) / 2.0, (rw - 1) / 2.0
        ry, rx = rh / 2.0, rw / 2.0
        ellipse_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0

        valid_pixels = roi_data[ellipse_mask]
        if valid_pixels.size == 0: return

        mean_val = float(np.mean(valid_pixels))
        spatial_std = float(np.std(valid_pixels))
        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        median_val = float(np.median(valid_pixels))
        num_px = int(valid_pixels.size)

        pixel_spacing_mm = float(self.current_meta.get("pixel_spacing_mm", 0.0846))
        area_mm2 = num_px * (pixel_spacing_mm ** 2)
        diam_x_mm = rw * pixel_spacing_mm
        diam_y_mm = rh * pixel_spacing_mm

        sys_err_ratio = 0.04
        mean_unc = np.sqrt((mean_val * sys_err_ratio)**2 + (spatial_std / np.sqrt(num_px))**2)
        median_unc = median_val * sys_err_ratio
        min_unc = min_val * sys_err_ratio
        max_unc = max_val * sys_err_ratio

        self.lbl_roi_mean.setText(f"<b>Mean Dose:</b> {mean_val:.3f} ± {mean_unc:.3f} Gy")
        self.lbl_roi_std.setText(f"<b>Spatial Std Dev:</b> {spatial_std:.3f} Gy")
        self.lbl_roi_median.setText(f"<b>Median:</b> {median_val:.3f} ± {median_unc:.3f} Gy")
        self.lbl_roi_min.setText(f"<b>Minimum:</b> {min_val:.3f} ± {min_unc:.3f} Gy")
        self.lbl_roi_max.setText(f"<b>Maximum:</b> {max_val:.3f} ± {max_unc:.3f} Gy")
        self.lbl_roi_pixels.setText(f"<b>ROI Area:</b> {area_mm2:.2f} mm² (Ø {diam_x_mm:.1f} × {diam_y_mm:.1f} mm)")

        stats_html = (
            f"<div style='font-size: 10pt; color: #FFFFFF;'>"
            f"<b>File:</b> {self.current_film_name}<br>"
            f"<b>Mean:</b> {mean_val:.3f} &plusmn; {mean_unc:.3f} Gy<br>"
            f"<b>Spatial Std:</b> {spatial_std:.3f} Gy<br>"
            f"<b>Median:</b> {median_val:.3f} &plusmn; {median_unc:.3f} Gy<br>"
            f"<b>Min / Max:</b> {min_val:.2f} &plusmn; {min_unc:.2f} / {max_val:.2f} &plusmn; {max_unc:.2f} Gy<br>"
            f"<b>Area:</b> {area_mm2:.2f} mm&sup2;<br>"
            f"<b>Diameters:</b> {diam_x_mm:.1f} &times; {diam_y_mm:.1f} mm"
            f"</div>"
        )
        self.canvas_stats_label.setText(stats_html)

        y, x = np.histogram(valid_pixels, bins=40)
        self.curve_roi_hist.setData(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PrismDosimetryGUI()
    window.show()
    sys.exit(app.exec_())