"""Morphology widgets; all numerical work is delegated to cancellable jobs."""
import os
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QComboBox, QSlider, QFrame, QMessageBox, QTabWidget, QSizePolicy
from background_tasks import TaskController
from morphology import analyze_morphology_region
from bio_analysis import SampleRegion
from processing_settings import MorphologySettings

class MorphologyTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_metrics = {}
        self.tasks = TaskController(self)
        self.tasks.ready.connect(self._display_analysis)
        self.tasks.failed.connect(self._analysis_error)
        self.tasks.progress.connect(self._show_progress)
        self.tasks.busy_changed.connect(self._set_busy)
        self.debounce = QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(200)
        self.debounce.timeout.connect(self.run_analysis)
        self._init_ui()
        self.btn_export.setEnabled(False)
        self.pages.currentChanged.connect(lambda index: self.schedule_analysis() if index == 1 else None)
        self.chk_snr_mask.toggled.connect(self.schedule_analysis)
        self.combo_shot_mode.currentIndexChanged.connect(self.schedule_analysis)
        main_window.roi.sigRegionChangeFinished.connect(self.schedule_analysis)
        main_window.roi.sigRegionChangeFinished.connect(self.measured.invalidate)
        if hasattr(main_window, 'film_boundary'):
            main_window.film_boundary.sigRegionChangeFinished.connect(self.measured.invalidate)

    def _init_ui(self):
        outer = QVBoxLayout(self)
        self.pages = QTabWidget(); self.pages.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored); outer.addWidget(self.pages)
        from beam_analysis_panel import BeamAnalysisPanel
        self.measured = BeamAnalysisPanel(self.main_window)
        self.pages.addTab(self.measured, 'Measured beam & exposure')
        legacy = QWidget(); self.pages.addTab(legacy, 'Exploratory morphology')
        layout = QVBoxLayout(legacy)
        note = QLabel('Exploratory transforms. Gaussian sharpening is an assumed PSF scenario, not source-beam recovery. '
                      'Intensity moments/entropy are not spatial beam moments; maxima are not proven filaments.')
        note.setWordWrap(True); layout.addWidget(note)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        # --- VEZÉRLŐ ÉS STATISZTIKAI PANEL ---
        panel_frame = QFrame()
        panel_frame.setStyleSheet("""
            QFrame#ControlPanel {
                background-color: #2e3440;
                border: 1px solid #4c566a;
                border-radius: 6px;
            }
        """)
        panel_frame.setObjectName("ControlPanel")
        
        panel_layout = QVBoxLayout(panel_frame)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(8)

        # 1. sor: Gombok, beállítások
        row1 = QHBoxLayout()
        self.btn_analyze = QPushButton("🔬 Run Analysis")
        self.btn_analyze.setStyleSheet("background-color: #5e81ac; color: #eceff4; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_analyze.clicked.connect(self.run_analysis)
        row1.addWidget(self.btn_analyze)

        lbl_mode = QLabel("<b>Mode:</b>")
        lbl_mode.setStyleSheet("color: #eceff4;")
        row1.addWidget(lbl_mode)

        self.combo_shot_mode = QComboBox()
        self.combo_shot_mode.addItems(["Accumulated footprint", "Single-exposure footprint"])
        self.combo_shot_mode.setStyleSheet("color: #eceff4; background-color: #3b4252; padding: 3px;")
        row1.addWidget(self.combo_shot_mode)
        
        self.chk_snr_mask = QCheckBox("Apply SNR > 3 Mask")
        self.chk_snr_mask.setStyleSheet("color: #eceff4;")
        row1.addWidget(self.chk_snr_mask)
        
        self.lbl_sigma = QLabel("<b>Scatter σ:</b> 0.0 mm")
        self.lbl_sigma.setStyleSheet("color: #eceff4;")
        row1.addWidget(self.lbl_sigma)
        
        self.slider_sigma = QSlider(Qt.Horizontal)
        self.slider_sigma.setRange(0, 30)
        self.slider_sigma.setValue(0)
        self.slider_sigma.setFixedWidth(110)
        self.slider_sigma.valueChanged.connect(self.on_sigma_changed)
        row1.addWidget(self.slider_sigma)
        
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_cancel.clicked.connect(self.invalidate_analysis)
        self.btn_cancel.setEnabled(False)
        row1.addWidget(self.btn_cancel)
        self.lbl_progress = QLabel("")
        self.btn_export = QPushButton("💾 Save Data & Figures")
        self.btn_export.setStyleSheet("background-color: #88c0d0; color: #2e3440; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_export.clicked.connect(self.export_figures)
        row1.addWidget(self.btn_export)
        
        row1.addStretch()
        panel_layout.addLayout(row1)
        self.lbl_progress.setWordWrap(True)
        panel_layout.addWidget(self.lbl_progress)

        # 2. sor: Statisztikák (kifejezetten látható, explicit színekkel)
        row2 = QHBoxLayout()
        self.lbl_moments = QLabel("<b>Skewness:</b> - | <b>Kurtosis:</b> - | <b>Entropy:</b> -")
        self.lbl_moments.setStyleSheet("color: #eceff4; font-size: 12px;")
        row2.addWidget(self.lbl_moments)

        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #4c566a; font-weight: bold;")
        row2.addWidget(sep1)

        self.lbl_radiobio = QLabel("<b>PVDR:</b> - | <b>FHV:</b> - % | <b>LDG:</b> - Gy/mm | <b>Lacunarity:</b> -")
        self.lbl_radiobio.setStyleSheet("color: #b48ead; font-size: 12px; font-weight: bold;")
        row2.addWidget(self.lbl_radiobio)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #4c566a; font-weight: bold;")
        row2.addWidget(sep2)

        self.lbl_watershed = QLabel("<b>Islands:</b> - | <b>Mean Area:</b> - mm² | <b>Dose in Islands:</b> - %")
        self.lbl_watershed.setStyleSheet("color: #a3be8c; font-size: 12px; font-weight: bold;")
        row2.addWidget(self.lbl_watershed)
        
        row2.addStretch()
        panel_layout.addLayout(row2)

        layout.addWidget(panel_frame)

        # --- VIZUALIZÁCIÓS RÁCS (2 sor × 4 oszlop) ---
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, stretch=1)

        # 1. sor: PSD, 2D Autocorr, Radial Autocorr, Radon Sinogram
        self.plot_psd = self.glw.addPlot(title="2D Power Spectrum (PSD)", row=0, col=0)
        self.img_psd = pg.ImageItem()
        self.plot_psd.addItem(self.img_psd)
        self.plot_psd.setAspectLocked(True)
        
        self.plot_autocorr = self.glw.addPlot(title="2D Autocorrelation", row=0, col=1)
        self.img_autocorr = pg.ImageItem()
        self.plot_autocorr.addItem(self.img_autocorr)
        self.plot_autocorr.setAspectLocked(True)

        self.plot_rad_ac = self.glw.addPlot(title="Radial Autocorr (Size Scale)", row=0, col=2)
        self.plot_rad_ac.setLabel('bottom', 'Radius [mm]')
        self.plot_rad_ac.setLabel('left', 'Correlation')
        self.curve_rad_ac = self.plot_rad_ac.plot(pen=pg.mkPen('#88c0d0', width=2))

        self.plot_radon = self.glw.addPlot(title="Radon Sinogram (Banding)", row=0, col=3)
        self.img_radon = pg.ImageItem()
        self.plot_radon.addItem(self.img_radon)
        self.plot_radon.setLabel('bottom', 'Angle [deg]')
        
        # 2. sor: Zernike Reconstruction, Zernike Coefficients (Bar), Watershed Islands
        self.plot_zernike_map = self.glw.addPlot(title="Zernike Reconstruction", row=1, col=0)
        self.img_zernike = pg.ImageItem()
        self.plot_zernike_map.addItem(self.img_zernike)
        self.plot_zernike_map.setAspectLocked(True)

        self.plot_zernike_bar = self.glw.addPlot(title="Zernike intensity shape coefficients", row=1, col=1)
        self.bar_zernike = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#ebcb8b')
        self.plot_zernike_bar.addItem(self.bar_zernike)
        self.plot_zernike_bar.setLabel('bottom', 'Mode')

        self.plot_watershed = self.glw.addPlot(title="Watershed candidate maxima", row=1, col=2, colspan=2)
        self.img_watershed = pg.ImageItem()
        self.plot_watershed.addItem(self.img_watershed)
        self.plot_watershed.setAspectLocked(True)

        # Színskálák
        lut_psd = pg.colormap.get('inferno').getLookupTable()
        pos = np.array([0.0, 0.5, 1.0])
        colors = np.array([[0, 50, 255, 255], [255, 255, 255, 255], [255, 50, 0, 255]], dtype=np.ubyte)
        lut_bipolar = pg.ColorMap(pos, colors).getLookupTable()
        lut_turbo = pg.colormap.get('turbo').getLookupTable()

        self.img_psd.setLookupTable(lut_psd)
        self.img_autocorr.setLookupTable(lut_bipolar)
        self.img_radon.setLookupTable(lut_psd)
        self.img_zernike.setLookupTable(lut_psd)
        self.img_watershed.setLookupTable(lut_turbo)

    def on_sigma_changed(self, value):
        self.lbl_sigma.setText(f"<b>Scatter σ:</b> {value / 10.0:.1f} mm")
        self.schedule_analysis()

    def _show_progress(self, message): self.lbl_progress.setText(message)

    def _set_busy(self, busy):
        self.btn_cancel.setEnabled(busy)
        self.btn_export.setEnabled(not busy and bool(self.current_metrics))
        if not busy: self.lbl_progress.setText("")

    def invalidate_analysis(self, include_measured=True):
        if include_measured and hasattr(self, 'measured'): self.measured.invalidate()
        self.debounce.stop(); self.tasks.cancel()
        self.lbl_progress.setText("Cancellation requested" if self.tasks.busy else "")
        self.current_metrics = {}
        self.btn_export.setEnabled(False)
        for image in (self.img_psd, self.img_autocorr, self.img_radon, self.img_zernike, self.img_watershed): image.clear()
        self.curve_rad_ac.setData([], []); self.bar_zernike.setOpts(height=np.zeros(6))
        for label in (self.lbl_moments, self.lbl_radiobio, self.lbl_watershed): label.setText('—')

    def schedule_analysis(self, *args):
        self.invalidate_analysis(include_measured=False)
        if self.pages.currentIndex() == 1: self.debounce.start()

    def run_analysis(self):
        self.invalidate_analysis(include_measured=False)
        window = self.main_window
        if window.current_dose_map is None: return
        shape, vectors, origin = window.roi.getAffineSliceParams(window.current_dose_map, window.img_item_dose)
        if min(shape) < 2: return
        region = SampleRegion(0, tuple(shape), tuple(tuple(v) for v in vectors), tuple(origin), None, None)
        data = np.array(window.current_dose_map, copy=True); data.setflags(write=False)
        sigma = None
        if window.current_std_map is not None:
            sigma = np.array(window.current_std_map, copy=True); sigma.setflags(write=False)
        settings = MorphologySettings(scatter_sigma_mm=self.slider_sigma.value()/10.,
            snr_mask=self.chk_snr_mask.isChecked(), shot_mode=self.combo_shot_mode.currentText())
        spacing = float(window.current_meta.get('pixel_spacing_mm', .0846))
        self.tasks.submit(lambda check, progress: analyze_morphology_region(data, sigma, region, spacing, settings, check, progress))

    def _analysis_error(self, error):
        self.current_metrics = {}
        QMessageBox.warning(self, 'Morphology failed', error)

    def _display_analysis(self, result):
        self.img_psd.setImage(result['psd'], autoLevels=True)
        self.img_autocorr.setImage(result['autocorr'], autoLevels=False); self.img_autocorr.setLevels([-1, 1])
        self.curve_rad_ac.setData(result['radii'], result['radial'])
        self.img_radon.setImage(result['sinogram'], autoLevels=True)
        self.img_zernike.setImage(result['reconstruction'], autoLevels=True)
        self.bar_zernike.setOpts(x=np.arange(len(result['coefficients'])), height=result['coefficients'])
        self.plot_zernike_bar.getAxis('bottom').setTicks([list(enumerate(result['labels']))])
        self.img_watershed.setImage(result['segmented_map'], autoLevels=True)
        metrics = result['metrics']
        def number(value): return f'{value:.2f}' if value is not None else 'N/A'
        self.plot_rad_ac.setTitle(f"Radial Autocorr (Length: {number(metrics['correlation_length_mm'])} mm)")
        self.lbl_moments.setText(f"Skewness: {number(metrics['skewness'])} | Kurtosis: {number(metrics['kurtosis'])} | Entropy: {number(metrics['shannon_entropy'])}")
        self.lbl_radiobio.setText(f"PVDR: {metrics['pvdr']:.2f} | FHV: {metrics['fhv_pct']:.1f}% | Mean LDG: {metrics['mean_ldg_gy_mm']:.1f} Gy/mm | Lacunarity: {metrics['lacunarity']:.3f}")
        self.lbl_watershed.setText(f"Islands: {metrics['num_islands']} | Mean Area: {metrics['mean_island_area_mm2']:.2f} mm² | Dose in Islands: {metrics['island_dose_fraction_pct']:.1f}%")
        self.current_metrics = metrics

    def export_figures(self):
        if not hasattr(self.main_window, 'analysis_dir') or not self.main_window.analysis_dir:
            QMessageBox.warning(self, "Export Error", "No active project! Please open a project first.")
            return
            
        out_dir = str(self.main_window.analysis_dir)
        base_name = getattr(self.main_window, 'current_film_name', 'morphology_scan')
        
        plots_to_export = {
            "psd": self.plot_psd,
            "autocorr": self.plot_autocorr,
            "radial_ac": self.plot_rad_ac,
            "sinogram": self.plot_radon,
            "zernike_map": self.plot_zernike_map,
            "zernike_bar": self.plot_zernike_bar,
            "watershed_islands": self.plot_watershed,
            "composite_grid": self.glw.scene()
        }
        
        try:
            # 1. JSON mentés
            if hasattr(self, 'current_metrics') and self.current_metrics:
                json_path = os.path.join(out_dir, f"{base_name}_morphology_metrics.json")
                with open(json_path, 'w') as f:
                    json.dump(self.current_metrics, f, indent=4)
                    
            # 2. Nagy felbontású képek mentése
            for suffix, item in plots_to_export.items():
                exporter = pg.exporters.ImageExporter(item)
                exporter.parameters()['width'] = 2000
                filename = os.path.join(out_dir, f"{base_name}_{suffix}.png")
                exporter.export(filename)
                
            QMessageBox.information(self, "Export Successful", f"Data and all 7 figures saved to:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save data:\n{str(e)}")

