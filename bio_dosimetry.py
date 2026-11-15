import os
import numpy as np
import pyqtgraph as pg
import csv
from bio_analysis import SampleRegion, analyze_samples
from processing_settings import BioSettings
from background_tasks import TaskController
from sample_statistics import axisymmetric_volume_weights

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, 
    QFileDialog, QListWidget, QComboBox, QSlider, QGroupBox, QSplitter, 
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QPainterPath
from PyQt5.QtCore import QPointF

class EppendorfROI(pg.ROI):
    """
    Merev testként mozgatható, forgatható Eppendorf-cső ROI.
    Alapból lefelé álló kúppal és egyetlen felső magasság-állító füllel (folyadékszint).
    """
    def __init__(self, pos, tube_type="1.5 ml", spacing=0.0846, **kwargs):
        self.spacing = spacing
        self.tube_type = tube_type

        # Standard Eppendorf belső méretek mm-ben: (szélesség, kezdeti folyadékmagasság, kúp magassága)
        dims = {
            "0.5 ml": (5.5, 6.0, 3.5),
            "1.5 ml": (8.8, 8.0, 5.0),
            "2.0 ml": (9.2, 9.0, 4.5)
        }
        w_mm, h_mm, self.cone_h_mm = dims.get(tube_type, (8.8, 8.0, 5.0))
        
        w_px = w_mm / self.spacing
        h_px = h_mm / self.spacing
        self.cone_h_px = self.cone_h_mm / self.spacing

        # Alap ROI téglalap: pos a bal-felső sarok
        super().__init__(pos, size=[w_px, h_px], **kwargs)

        # 1. Forgató fül a jobb-felső saroknál
        self.addRotateHandle([1.0, 0.0], [0.5, 1.0])
        # 2. Folyadékmagasság-állító fül (csak fel-le mozog a felső peremen)
        self.addScaleHandle([0.5, 0.0], [0.5, 1.0])

    def get_contour_path(self):
        """Megadja az Eppendorf-cső alakjának zárt QPainterPath kontúrját a lokális ROI koordinátákban."""
        w, h = self.state['size']
        cone_h = min(self.cone_h_px, h * 0.8) # kúp magassága nem haladhatja meg a teljes magasságot
        tip_r = w * 0.18 # az alsó csúcs legömbölyítése

        path = QPainterPath()
        # Bal felső sarok (meniszkusz szintje)
        path.moveTo(0, 0)
        # Felső él (folyadék felszín)
        path.lineTo(w, 0)
        # Jobb függőleges hengeres fal a kúp kezdetéig
        path.lineTo(w, h - cone_h)
        # Jobb lejtős fal a hegyhez
        path.lineTo(w / 2.0 + tip_r, h - tip_r * 0.3)
        # Lekerekített orr (lefelé álló alsó hegy)
        path.quadTo(w / 2.0, h, w / 2.0 - tip_r, h - tip_r * 0.3)
        # Bal lejtős fal felfelé
        path.lineTo(0, h - cone_h)
        # Zárás vissza a meniszkuszhoz
        path.closeSubpath()
        return path

    def paint(self, p, opt, widget):
        """Kirajzolja a meniszkuszt és a cső kontúrját."""
        p.save()
        p.setPen(self.currentPen)
        p.drawPath(self.get_contour_path())
        
        # Opcionális: meniszkusz jelzése szaggatott vonallal
        p.setPen(pg.mkPen(color=self.currentPen.color(), style=Qt.DashLine))
        p.drawLine(QPointF(0, 0), QPointF(self.state['size'][0], 0))
        p.restore()

    def get_mask(self, shape):
        """Létrehozza a maszkot a kivágott téglalap méretéhez igazítva."""
        from PyQt5.QtGui import QImage, QPainter
        from PyQt5.QtCore import Qt

        h, w = shape
        img = QImage(w, h, QImage.Format_Mono)
        img.fill(0)

        p = QPainter(img)
        p.setBrush(Qt.white)
        p.setPen(Qt.NoPen)
        
        # A kontúr skálázása a kivágott numpy mátrix méretéhez
        path = self.get_contour_path()
        scale_x = w / max(self.state['size'][0], 1e-5)
        scale_y = h / max(self.state['size'][1], 1e-5)
        
        from PyQt5.QtGui import QTransform
        trans = QTransform().scale(scale_x, scale_y)
        scaled_path = trans.map(path)
        
        p.drawPath(scaled_path)
        p.end()

        # Visszaalakítás boolean numpy maszkká
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, img.bytesPerLine()))
        mask = np.unpackbits(arr, axis=1)[:, :w] > 0
        return mask

class BioSampleGeometry:
    """Fizikai paraméterek és Z_eff számítás a 3D interpolációhoz."""
    def __init__(self):
        # Alapértelmezett geometriai sűrűségek (g/cm^3)
        self.rho_pmma = 1.18
        self.rho_air = 0.0012
        self.rho_water = 1.0
        
        # Vastagságok (mm)
        self.d_pmma_front = 5.0
        self.d_air_front = 1.0
        self.d_sample = 2.0
        self.d_air_back = 1.0
        self.d_pmma_back = 5.0

    def get_weights_and_scattering(self):
        """Kiszámolja az interpolációs súlyt (W) és a légrés miatti szóródási szigmát."""
        z_front = self.d_pmma_front * self.rho_pmma + self.d_air_front * self.rho_air
        z_sample_center = z_front + (self.d_sample / 2.0) * self.rho_water
        z_back = self.d_sample * self.rho_water + self.d_air_back * self.rho_air + self.d_pmma_back * self.rho_pmma
        z_total = z_front + z_back
        
        # Interpolációs súly (0.0 = teljesen a Front, 1.0 = teljesen a Back film)
        weight = z_sample_center / z_total if z_total > 0 else 0.5
        
        # Levegő miatti laterális elektron-szóródás (közelítés: 0.15 mm blur / 1 mm levegő)
        total_air = self.d_air_front + self.d_air_back
        scatter_sigma_mm = total_air * 0.15 
        
        return weight, scatter_sigma_mm

class BioDosimetryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.film_paths = {}
        self.front_dose = None
        self.back_dose = None
        self.front_std = None
        self.back_std = None
        self.spacing = 0.0846  # Default 300 DPI mm/pixel
        self.front_spacing = None
        self.back_spacing = None
        self.rois = []
        self.results_data = {}
        self.geometry = BioSampleGeometry()

        
        self.tasks = TaskController(self)
        self.tasks.ready.connect(self._display_results)
        self.tasks.failed.connect(self._calculation_error)
        self.tasks.progress.connect(self._show_progress)
        self.tasks.busy_changed.connect(self._set_calculation_busy)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # ==========================================
        # BAL PANEL: Vezérlés és Geometria
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Film betöltés
        io_group = QGroupBox("1. Load Dose Maps (*_dose.npy)")
        io_layout = QVBoxLayout(io_group)
        self.btn_load_front = QPushButton("Load FRONT Film")
        self.btn_load_front.clicked.connect(lambda: self.load_film('front'))
        self.btn_load_back = QPushButton("Load BACK Film")
        self.btn_load_back.clicked.connect(lambda: self.load_film('back'))
        
        self.lbl_front = QLabel("Front: Not loaded")
        self.lbl_back = QLabel("Back: Not loaded")
        self.lbl_front.setStyleSheet("color: red;")
        self.lbl_back.setStyleSheet("color: red;")
        
        io_layout.addWidget(self.btn_load_front)
        io_layout.addWidget(self.lbl_front)
        io_layout.addWidget(self.btn_load_back)
        io_layout.addWidget(self.lbl_back)
        left_layout.addWidget(io_group)

        # Regisztráció (Alignment)
        align_group = QGroupBox("2. Manual Alignment (Front to Back)")
        align_layout = QGridLayout(align_group)
        
        align_layout.addWidget(QLabel("Shift X:"), 0, 0)
        self.sl_shift_x = QSlider(Qt.Horizontal)
        self.sl_shift_x.setRange(-200, 200)
        self.sl_shift_x.valueChanged.connect(self.update_alignment)
        align_layout.addWidget(self.sl_shift_x, 0, 1)

        align_layout.addWidget(QLabel("Shift Y:"), 1, 0)
        self.sl_shift_y = QSlider(Qt.Horizontal)
        self.sl_shift_y.setRange(-200, 200)
        self.sl_shift_y.valueChanged.connect(self.update_alignment)
        align_layout.addWidget(self.sl_shift_y, 1, 1)

        align_layout.addWidget(QLabel("Rotate:"), 2, 0)
        self.sl_rot = QSlider(Qt.Horizontal)
        self.sl_rot.setRange(-450, 450) # -45.0 to 45.0 fok
        self.sl_rot.valueChanged.connect(self.update_alignment)
        align_layout.addWidget(self.sl_rot, 2, 1)
        
        self.btn_reset_align = QPushButton("Reset Alignment")
        self.btn_reset_align.clicked.connect(self.reset_alignment)
        align_layout.addWidget(self.btn_reset_align, 3, 0, 1, 2)
        left_layout.addWidget(align_group)
        
        # Geometria
        geo_group = QGroupBox("3. Phantom Geometry & Air Cavity (mm)")
        geo_layout = QGridLayout(geo_group)
        self.spins = {}
        labels = [
            ('d_pmma_front', 'Front PMMA:'), ('d_air_front', 'Air Gap 1:'),
            ('d_sample', 'Sample Liquid:'), ('d_air_back', 'Air Gap 2:'),
            ('d_pmma_back', 'Back PMMA:')
        ]
        from PyQt5.QtWidgets import QDoubleSpinBox
        for i, (key, label) in enumerate(labels):
            geo_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0, 50.0)
            spin.setSingleStep(0.5)
            spin.setValue(getattr(self.geometry, key))
            spin.valueChanged.connect(lambda val, k=key: setattr(self.geometry, k, val))
            spin.valueChanged.connect(self._invalidate_results)
            self.spins[key] = spin
            geo_layout.addWidget(spin, i, 1)
        left_layout.addWidget(geo_group)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ==========================================
        # KÖZÉPSŐ PANEL: Vizualizáció és ROI
        # ==========================================
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        
        toolbar = QHBoxLayout()
        
        # ROI típus választó menü előre definiált méretekkel
        toolbar.addWidget(QLabel("<b>ROI Tool:</b>"))
        self.combo_roi_type = QComboBox()
        self.combo_roi_type.addItems([
            "🧪 Eppendorf 1.5 ml",
            "🧪 Eppendorf 0.5 ml",
            "🧪 Eppendorf 2.0 ml",
            "⭕ Circular / Well",
            "✏️ Freehand Polygon"
        ])
        toolbar.addWidget(self.combo_roi_type)
        
        self.btn_add_roi = QPushButton("➕ Add ROI")
        self.btn_add_roi.clicked.connect(self.add_selected_roi)
        toolbar.addWidget(self.btn_add_roi)
        
        self.btn_clear_rois = QPushButton("🗑️ Clear ROIs")
        self.btn_clear_rois.clicked.connect(self.clear_all_rois)
        toolbar.addWidget(self.btn_clear_rois)
        
        self.btn_calc = QPushButton("⚡ Calculate Sample Doses")
        self.btn_calc.setStyleSheet("background-color: #2b8a3e; color: white; font-weight: bold;")
        self.btn_calc.clicked.connect(self.calculate_doses)
        toolbar.addWidget(self.btn_calc)
        self.btn_cancel = QPushButton("Cancel Calculation")
        self.btn_cancel.clicked.connect(self._invalidate_results)
        self.btn_cancel.setEnabled(False)
        self.lbl_progress = QLabel("")
        
        toolbar.addWidget(QLabel("<b>Overlay Blend:</b>"))
        self.sl_blend = QSlider(Qt.Horizontal)
        self.sl_blend.setRange(0, 100)
        self.sl_blend.setValue(50)
        self.sl_blend.valueChanged.connect(self.update_blend)
        toolbar.addWidget(self.sl_blend)
        
        mid_layout.addLayout(toolbar)
        status_row = QHBoxLayout()
        status_row.addWidget(self.btn_cancel)
        self.lbl_progress.setWordWrap(True)
        status_row.addWidget(self.lbl_progress, 1)
        mid_layout.addLayout(status_row)

        self.glw = pg.GraphicsLayoutWidget()
        
        # 1. Hátsó film színsávja (Balra)
        self.hist_back = pg.HistogramLUTItem()
        self.hist_back.setMinimumWidth(80)
        self.hist_back.setMaximumWidth(110)
        self.glw.addItem(self.hist_back, row=0, col=0)
        
        # Középső rajzterület
        self.plot = self.glw.addPlot(row=0, col=1)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        
        # 2. Mellső film színsávja (Jobbra)
        self.hist_front = pg.HistogramLUTItem()
        self.hist_front.setMinimumWidth(80)
        self.hist_front.setMaximumWidth(110)
        self.glw.addItem(self.hist_front, row=0, col=2)
        
        self.glw.ci.layout.setColumnStretchFactor(0, 0)
        self.glw.ci.layout.setColumnStretchFactor(1, 10)
        self.glw.ci.layout.setColumnStretchFactor(2, 0)
        
        # Rétegek
        self.img_back = pg.ImageItem()
        self.plot.addItem(self.img_back)
        self.hist_back.setImageItem(self.img_back)
        self.hist_back.gradient.loadPreset('viridis')
        
        self.img_front = pg.ImageItem()
        self.img_front.setZValue(2)
        self.img_front.setOpacity(0.5)
        from PyQt5.QtGui import QPainter
        self.img_front.setCompositionMode(QPainter.CompositionMode_Plus)
        self.plot.addItem(self.img_front)
        self.hist_front.setImageItem(self.img_front)
        self.hist_front.gradient.loadPreset('inferno')
        
        mid_layout.addWidget(self.glw)
        splitter.addWidget(mid_panel)

        # ==========================================
        # JOBB PANEL: Eredmények és DVH
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ROI ID", "Mean [Gy]", "Spread [%]", "Mean pixel uncertainty [%]", "Min [Gy]", "Max [Gy]"])
        self.table.horizontalHeaderItem(3).setToolTip(
            'Volume-weighted mean pixel uncertainty bound / mean dose. '
            'Unknown front/back correlation is treated conservatively; this is not uncertainty of the sample mean.')
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.plot_selected_dvh)
        right_layout.addWidget(QLabel("<b>Sample Results:</b>"))
        right_layout.addWidget(self.table, stretch=1)
        
        # --- 1. TÁBLÁZAT ALATTI GOMB: Táblázat adatainak exportálása (CSV/Excel) ---
        self.btn_export_table = QPushButton("📊 Export Sample Table (CSV / Excel)")
        self.btn_export_table.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_table.clicked.connect(self.export_sample_table)
        right_layout.addWidget(self.btn_export_table)

        self.btn_export = QPushButton("💾 Export Sample Data (.npz)")
        self.btn_export.clicked.connect(self.export_numpy)
        right_layout.addWidget(self.btn_export)

        self.combo_dvh_range = QComboBox()
        self.combo_dvh_range.addItems(["Dose axis: sample range", "Dose axis: full, from 0 Gy"])
        self.combo_dvh_range.setToolTip("Zoom to occupied dose bins. Bin volumes, statistics and exported data are unchanged.")
        self.combo_dvh_range.currentIndexChanged.connect(self.plot_selected_dvh)
        right_layout.addWidget(self.combo_dvh_range)

        # DVH Grafikon
        self.plot_dvh = pg.PlotWidget(title="Differential Dose-Volume Histogram (dDVH)")
        self.plot_dvh.setLabel('bottom', 'Dose [Gy]')
        self.plot_dvh.setLabel('left', 'Relative Volume [%]')
        self.curve_dvh = self.plot_dvh.plot(stepMode="center", fillLevel=0, fillOutline=True, brush=(0, 150, 255, 90), pen=pg.mkPen('c', width=2))
        right_layout.addWidget(self.plot_dvh, stretch=1)
        
        # --- 2. HISZTOGRAM ALATTI GOMB: dDVH CSV exportálása ---
        self.btn_export_ddvh = QPushButton("📈 Export dDVH Data (CSV)")
        self.btn_export_ddvh.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_ddvh.clicked.connect(self.export_csv)
        right_layout.addWidget(self.btn_export_ddvh)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700, 400])

    def update_blend(self):
        """Vezérli az elülső és hátsó film láthatóságát (0: csak Back, 100: csak Front)."""
        val = self.sl_blend.value() / 100.0
        if hasattr(self, 'img_front') and hasattr(self, 'img_back'):
            self.img_front.setOpacity(val)
            self.img_back.setOpacity(1.0 - (val * 0.5))  # Back nem tűnik el teljesen

    def load_film(self, target, path=None, report_errors=True):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, f"Load {target.upper()} Dose Map", "", "Numpy Arrays (*_dose.npy *.npy)")
        if not path:
            return
        
        try:
            from pathlib import Path
            p = Path(path)
            dose_map = np.load(str(p)).astype(np.float32)
            if (dose_map.ndim != 2 or 0 in dose_map.shape
                    or not np.all(np.isfinite(dose_map)) or np.any(dose_map < 0)):
                raise ValueError('Dose maps must be finite, nonnegative, nonempty 2D arrays.')

            import json
            meta_path = p.with_name(p.stem.removesuffix('_dose') + '_meta.json')
            meta = {}
            spacing = 0.0846
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                spacing = float(meta.get('pixel_spacing_mm', spacing))
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError('Pixel spacing must be finite and positive.')
            other_spacing = self.back_spacing if target == 'front' else self.front_spacing
            if other_spacing is not None and not np.isclose(spacing, other_spacing, rtol=1e-4, atol=0):
                raise ValueError('Front and back pixel spacing must match. Resample the films before loading.')
            
            # Robusztus std fájlkeresés több elnevezési sémára
            std_candidates = [
                p.with_name(p.name.replace("_dose.npy", "_std.npy")),
                p.with_name(p.stem.replace("_dose", "") + "_std.npy"),
                p.with_name(p.stem + "_std.npy")
            ]
            
            std_map = None
            for cand in std_candidates:
                if cand.exists() and cand != p:
                    std_map = np.load(str(cand)).astype(np.float32)
                    if (std_map.shape != dose_map.shape or not np.all(np.isfinite(std_map))
                            or np.any(std_map < 0)):
                        raise ValueError('The STD map must match the dose map and contain finite, nonnegative values.')
                    break

            # Peremhibák, nullák és a fals kiugró artefaktumok szűrése az ablakoláshoz
            valid_data = dose_map[(dose_map > 0.1) & (dose_map < 100.0)]
            if valid_data.size > 0:
                min_d = float(np.percentile(valid_data, 2.0))
                max_d = float(np.percentile(valid_data, 98.0))
            else:
                active_data = dose_map[dose_map > 0.05]
                if active_data.size > 0:
                    min_d = float(np.percentile(active_data, 2.0))
                    max_d = float(np.percentile(active_data, 98.0))
                else:
                    min_d, max_d = 0.0, 10.0
                
            if min_d >= max_d:
                max_d = min_d + 1.0

            max_d = min(max_d, 60.0)

            std_status = "✓ std OK" if std_map is not None else "⚠ std missing (2% fallback)"
            if std_map is not None and meta.get('calibration_uncertainty_available') is False:
                std_status = '⚠ partial std (calibration uncertainty unavailable)'

            if target == 'front':
                self.front_dose = dose_map
                self.front_std = std_map
                self.front_spacing = spacing
                self.img_front.setImage(dose_map, autoLevels=False)
                self.hist_front.setLevels(min_d, max_d)
                self.hist_front.setHistogramRange(min_d, max_d, padding=0.05)
                self.img_front.setLevels((min_d, max_d))
                
                self.lbl_front.setText(f"Front: {p.name} [{std_status}]")
                self.lbl_front.setStyleSheet("color: green; font-weight: bold;")
                
                h, w = dose_map.shape
                self.img_front.setTransformOriginPoint(w / 2.0, h / 2.0)
            else:
                self.back_dose = dose_map
                self.back_std = std_map
                self.back_spacing = spacing
                self.img_back.setImage(dose_map, autoLevels=False)
                self.hist_back.setLevels(min_d, max_d)
                self.hist_back.setHistogramRange(min_d, max_d, padding=0.05)
                self.img_back.setLevels((min_d, max_d))
                
                self.lbl_back.setText(f"Back: {p.name} [{std_status}]")
                self.lbl_back.setStyleSheet("color: green; font-weight: bold;")
                
            if self.spacing != spacing:
                self.clear_all_rois()
            self.spacing = spacing
            self.update_alignment()
            self.plot.autoRange()
            self.film_paths[target] = p.resolve()
            return True
            
        except Exception as e:
            if not report_errors: raise
            QMessageBox.critical(self, "Error", f"Failed to load map:\n{e}")

    def update_alignment(self):
        """Hardver-gyorsított PyQtGraph vizuális igazítás (számolás nélkül)."""
        self._invalidate_results()
        if self.front_dose is None: return
        dx = self.sl_shift_x.value()
        dy = self.sl_shift_y.value()
        rot_deg = self.sl_rot.value() / 10.0
        
        self.img_front.setPos(dx, dy)
        self.img_front.setRotation(rot_deg)

    def _invalidate_results(self):
        self.tasks.cancel()
        self.lbl_progress.setText("Cancellation requested" if self.tasks.busy else "")
        self.results_data.clear()
        self.table.setRowCount(0)
        self.curve_dvh.clear()
        self.plot_dvh.setTitle('Differential Dose-Volume Histogram (dDVH)')

    def _front_to_back_transform(self):
        """Derive the numerical transform from the actual displayed image items."""
        origin = self.img_front.mapToItem(self.img_back, QPointF(0, 0))
        x_axis = self.img_front.mapToItem(self.img_back, QPointF(1, 0))
        y_axis = self.img_front.mapToItem(self.img_back, QPointF(0, 1))
        return np.array([
            [x_axis.x() - origin.x(), y_axis.x() - origin.x(), origin.x()],
            [x_axis.y() - origin.y(), y_axis.y() - origin.y(), origin.y()],
            [0.0, 0.0, 1.0],
        ])

    def reset_alignment(self):
        self.sl_shift_x.setValue(0)
        self.sl_shift_y.setValue(0)
        self.sl_rot.setValue(0)

    def add_selected_roi(self):
        if self.back_dose is None:
            QMessageBox.warning(self, "Warning", "Load the BACK film first to place ROIs!")
            return
            
        h, w = self.back_dose.shape
        cx, cy = w // 2, h // 2
        selected_text = self.combo_roi_type.currentText()
        
        if "Eppendorf" in selected_text:
            tube_type = "1.5 ml"
            if "0.5" in selected_text: tube_type = "0.5 ml"
            elif "2.0" in selected_text: tube_type = "2.0 ml"
            
            roi = EppendorfROI([cx - 20, cy - 30], tube_type=tube_type, spacing=self.spacing, pen=pg.mkPen('#00ffcc', width=2))
            lbl_color = '#00ffcc'
            
        elif "Circular" in selected_text:
            r_px = int(4.0 / self.spacing)
            roi = pg.EllipseROI([cx - r_px//2, cy - r_px//2], [r_px, r_px], pen=pg.mkPen('y', width=2))
            lbl_color = 'y'
            
        else: # Szabadkézi poligon
            r = int(3.5 / self.spacing)
            poly_pts = [
                [cx - r, cy - r],
                [cx + r, cy - r],
                [cx + 1.2*r, cy],
                [cx, cy + 1.3*r],
                [cx - 1.2*r, cy]
            ]
            roi = pg.PolyLineROI(poly_pts, closed=True, pen=pg.mkPen('#ff00ff', width=2))
            lbl_color = '#ff00ff'

        roi.project_kind = self.combo_roi_type.currentIndex()
        self.plot.addItem(roi)
        self.rois.append(roi)
        roi.sigRegionChanged.connect(self._invalidate_results)
        self._invalidate_results()
        
        # Címke elhelyezése
        text = pg.TextItem(f"#{len(self.rois)}", color=lbl_color, anchor=(0, 1))
        text.setParentItem(roi)

    def clear_all_rois(self):
        for roi in self.rois:
            self.plot.removeItem(roi)
        self.rois.clear()
        self._invalidate_results()

    def calculate_doses(self):
        if self.front_dose is None or self.back_dose is None:
            QMessageBox.warning(self, "Error", "Both films must be loaded!")
            return
        if not self.rois:
            QMessageBox.warning(self, "Error", "Add at least one ROI!")
            return

        self._invalidate_results()
        regions = []
        for identifier, roi in enumerate(self.rois, start=1):
            shape, vectors, origin = roi.getAffineSliceParams(self.back_dose, self.img_back,
                fromBoundingRect=isinstance(roi, pg.PolyLineROI))
            shape_int = tuple(int(np.ceil(value)) for value in shape)
            if min(shape_int) <= 0:
                mask = np.zeros(shape_int, dtype=bool); weights = np.zeros(shape_int)
            elif isinstance(roi, EppendorfROI):
                mask = roi.get_mask(shape_int); weights = axisymmetric_volume_weights(mask)
            else:
                mask = roi.renderShapeMask(shape_int[1], shape_int[0]).T > .5
                weights = np.ones(shape_int)
            mask.setflags(write=False); weights.setflags(write=False)
            regions.append(SampleRegion(identifier, tuple(shape), tuple(tuple(v) for v in vectors),
                                        tuple(origin), mask, weights))
        def snapshot(value):
            if value is None: return None
            result = np.array(value, copy=True); result.setflags(write=False); return result
        front, back, front_std, back_std = map(snapshot, (self.front_dose, self.back_dose, self.front_std, self.back_std))
        transform = self._front_to_back_transform()
        weight, scattering = self.geometry.get_weights_and_scattering()
        settings = BioSettings(weight, scattering, self.spacing)
        regions = tuple(regions)
        self.tasks.submit(lambda check, progress: analyze_samples(front, back, front_std, back_std,
                            transform, settings, regions, check, progress))

    def _show_progress(self, message): self.lbl_progress.setText(message)

    def _set_calculation_busy(self, busy):
        self.btn_calc.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        if not busy: self.lbl_progress.setText("")

    def _calculation_error(self, error):
        QMessageBox.warning(self, 'Sample calculation failed', error)

    def _display_results(self, payload):
        results = payload['results']; skipped = payload['skipped']
        # Publish only after every nonempty sample has passed validation.
        self.results_data = results
        for roi_id, data in results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [f'Sample #{roi_id}', f"{data['mean_dose']:.2f}",
                      self._format_percentage(data['spread_pct']),
                      self._format_percentage(data['uncert_pct']),
                      f"{data['min_dose']:.2f}", f"{data['max_dose']:.2f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 0:
                    item.setData(Qt.UserRole, roi_id)
                self.table.setItem(row, col, item)

        self.table.selectRow(0)
        self.plot_selected_dvh()
        message = 'Calculations and dDVH extraction complete!'
        if skipped:
            message += '\nEmpty ROIs skipped: ' + ', '.join(f'#{roi_id}' for roi_id in skipped)
        QMessageBox.information(self, 'Success', message)

    @staticmethod
    def _format_percentage(value):
        return f'{value:.1f}' if value is not None else 'N/A'

    def plot_selected_dvh(self):
        selected = self.table.selectedItems()
        if not selected:
            self.curve_dvh.clear()
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        roi_id = item.data(Qt.UserRole) if item is not None else None
        
        data = self.results_data.get(roi_id)
        if data:
            x_edges = data.get('hist_edges')
            y = data.get('hist_y')
            
            if x_edges is not None and y is not None:
                # A stepMode="center" esetén len(x) == len(y) + 1 feltétel szükséges
                self.curve_dvh.setData(x=x_edges, y=y, stepMode="center")
                self.plot_dvh.setTitle(f"Sample #{roi_id} dDVH (Mean: {data['mean_dose']:.2f} Gy)")
                if self.combo_dvh_range.currentIndex() == 0:
                    occupied = np.flatnonzero(np.asarray(y) > 0)
                    first, last = (occupied[0], occupied[-1]+1) if occupied.size else (0, len(x_edges)-1)
                    lower, upper = float(x_edges[first]), float(x_edges[last])
                else:
                    lower, upper = 0., float(x_edges[-1])
                self.plot_dvh.setXRange(lower, upper, padding=.04)
                self.plot_dvh.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def export_numpy(self):
        if not self.results_data:
            QMessageBox.warning(self, "Error", "No calculated data to export!")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Sample Data to NPZ", "biological_samples_data.npz", "Numpy Archives (*.npz)")
        if not path: return
        
        try:
            export_dict = {
                'format_version': np.array(2),
                'stat_names': np.array(['mean_dose_gy', 'spatial_std_gy', 'mean_pixel_sigma_gy',
                                        'min_dose_gy', 'max_dose_gy']),
                'dose_stats_rows': np.array(['dose_bin_center_gy', 'mean_pixel_sigma_gy']),
                'uncertainty_definition': np.array(
                    'Volume-weighted mean pixel uncertainty bound; unknown front/back correlation; '
                    'not uncertainty of the sample mean.'),
            }
            for roi_id, data in self.results_data.items():
                prefix = f"sample_{roi_id}_"
                export_dict[prefix + 'raw_voxels'] = data['raw_voxels']
                export_dict[prefix + 'raw_sigmas'] = data['raw_sigmas']
                export_dict[prefix + 'dose_stats'] = data['dose_stats']
                export_dict[prefix + 'weights'] = data['weights']
                export_dict[prefix + 'hist_edges'] = data['hist_edges']
                export_dict[prefix + 'hist_volume_pct'] = data['hist_y']
                export_dict[prefix + 'z_weight_back'] = np.array(data['z_weight_back'])
                export_dict[prefix + 'stats'] = np.array([
                    data['mean_dose'], data['spread_dose'], data['uncert_dose'], data['min_dose'], data['max_dose']
                ])
                
            np.savez_compressed(path, **export_dict)
            QMessageBox.information(self, "Exported", f"Data saved successfully to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_csv(self):
        if not self.results_data:
            QMessageBox.warning(self, "Error", "No calculated data to export!")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save dDVH to CSV", "biological_samples_ddvh.csv", "CSV Files (*.csv)")
        if not path: return
        
        try:
            with open(path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for roi_id, data in self.results_data.items():
                    writer.writerow([f"--- SAMPLE #{roi_id} ---"])
                    writer.writerow(["Dose Bin [Gy]", "Relative Volume [%]",
                                     "Bin Lower [Gy]", "Bin Upper [Gy]"])
                    x_centers = data['hist_x']
                    y = data['hist_y']
                    edges = data['hist_edges']
                    for d, vol, lower, upper in zip(x_centers, y, edges[:-1], edges[1:]):
                        writer.writerow([float(d), float(vol), float(lower), float(upper)])
                    writer.writerow([])
            QMessageBox.information(self, "Exported", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_sample_table(self):
        """A mintastatisztikai táblázat mentése CSV vagy Excel fájlba."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()
        if row_count == 0:
            QMessageBox.warning(self, "Export Error", "The table is empty. Calculate sample doses first!")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Sample Metrics",
            "sample_metrics.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not file_path:
            return

        try:
            # Fejlécek
            headers = []
            for col in range(col_count):
                item = self.table.horizontalHeaderItem(col)
                headers.append(item.text() if item else f"Column_{col+1}")

            # Sorok adatai
            rows = []
            for r in range(row_count):
                roi_id = self.table.item(r, 0).data(Qt.UserRole)
                data = self.results_data[roi_id]
                rows.append([f'Sample #{roi_id}', data['mean_dose'],
                             data['spread_pct'] if data['spread_pct'] is not None else 'N/A',
                             data['uncert_pct'] if data['uncert_pct'] is not None else 'N/A',
                             data['min_dose'], data['max_dose']])

            if file_path.endswith('.xlsx') or "Excel" in selected_filter:
                try:
                    from openpyxl import Workbook
                except ImportError:
                    QMessageBox.warning(self, 'Excel support unavailable',
                                        'Install optional Excel support with: uv sync --locked --extra excel. '
                                        'CSV export is available without it.')
                    return
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                workbook = Workbook(write_only=True)
                sheet = workbook.create_sheet('Sample metrics')
                sheet.append(headers)
                for row in rows:
                    sheet.append(row)
                workbook.save(file_path)
            else:
                import csv
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)

            QMessageBox.information(self, "Export Successful", f"Sample metrics saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export table data:\n{str(e)}")

    def _create_eppendorf_points(self, center_x, center_y, width_mm=4.0, height_mm=6.0):
        """
        Generál egy Eppendorf-cső alsó hegyét formázó zárt poligon pontlistát pixel koordinátákban.
        width_mm: folyadékoszlop szélessége (belső átmérő)
        height_mm: folyadékoszlop magassága a hegytől mérve
        """
        sp = self.spacing
        w_px = (width_mm / sp) / 2.0
        h_px = height_mm / sp
        
        # A hegy lekerekítéséhez (alsó ív) pontok
        tip_r = w_px * 0.35  # cső aljának gömbi lekerekítése
        tip_points = []
        for angle in np.linspace(np.pi, 0, 7):
            tip_points.append([tip_r * np.cos(angle), -tip_r * np.sin(angle)])
            
        # Zárt kontúr: bal felső -> jobb felső -> jobb lejtő -> lekerekített alsó csúcs -> bal lejtő
        pts = [
            [-w_px, h_px],          # Bal felső sarok (meniszkusz szintje)
            [w_px, h_px],           # Jobb felső sarok
            [w_px, h_px * 0.45],    # Párhuzamos fal vége jobb oldalon
            [tip_r, 0],             # Kúpos szűkülés kezdete a hegyhez
        ]
        
        # Alsó lekerekített csúcs
        for tp in tip_points:
            pts.append([tp[0], tp[1]])
            
        pts.append([-tip_r, 0])
        pts.append([-w_px, h_px * 0.45])  # Bal kúpos fal
        
        # Eltolás a megadott középpontba
        final_points = [[center_x + p[0], center_y - h_px/2.0 + p[1]] for p in pts]
        return final_points
