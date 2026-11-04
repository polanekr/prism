import os
import sys
import subprocess
import re
import datetime
import tempfile
import platform
import numpy as np
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox,
    QFileDialog, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ScanWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs
        self.is_windows = platform.system() == "Windows"

    def run(self):
        if self.task == "find_scanners":
            self._find_scanners()
        elif self.task == "warmup":
            self._run_warmup()
        elif self.task == "scan":
            self._run_scan()

    def _find_scanners(self):
        devices = []
        try:
            if self.is_windows:
                result = subprocess.run(['naps2.console', '/getdevices'], capture_output=True, text=True, check=True)
                for line in result.stdout.splitlines():
                    if line.strip() and not line.startswith("NAPS2"):
                        devices.append(line.strip())
            else:
                result = subprocess.run(['scanimage', '-L'], capture_output=True, text=True, check=True)
                devices = re.findall(r"device ['`](.*?)['`]", result.stdout)
            self.finished_signal.emit(True, "\n".join(devices))
        except Exception as e:
            self.log_signal.emit(f"Scanner discovery failed: {e}. Is NAPS2/SANE installed?")
            self.finished_signal.emit(False, "")

    def _build_cmd(self):
        res = self.kwargs.get('res', 300)
        l, t = self.kwargs.get('left', 0), self.kwargs.get('top', 0)
        w, h = self.kwargs.get('width', 0), self.kwargs.get('height', 0)
        profile = self.kwargs.get('profile', "Full Bed Area")
        raw_mode = self.kwargs.get('raw_mode', True)

        if self.is_windows:
            cmd = ['/r', str(res), '/bitdepth', '16']
            if profile != "Full Bed Area" and w > 0 and h > 0:
                cmd.extend(['/crop', f'{l}mm,{t}mm,{w}mm,{h}mm'])
        else:
            cmd = ['--format=tiff', '--mode=Color', '--depth=16', f'--resolution={res}', '--source', 'Transparency Unit']
            if raw_mode:
                cmd.append('--color-correction=None')
            if profile != "Full Bed Area" and w > 0 and h > 0:
                cmd.extend(['-l', str(l), '-t', str(t), '-x', str(w), '-y', str(h)])
        return cmd

    def _run_warmup(self):
        device_name = self.kwargs.get('device_name')
        cmd_opts = self._build_cmd()
        self.log_signal.emit("\n--- STARTING LAMP WARM-UP ---")
        try:
            for i in range(1, 4):
                self.log_signal.emit(f"Blank exposure ({i}/3) in progress...")
                if self.is_windows:
                    tmp_file = tempfile.mktemp(suffix=".tif")
                    subprocess.run(['naps2.console', '/o', tmp_file, '/d', device_name] + cmd_opts, check=True)
                    if os.path.exists(tmp_file): os.remove(tmp_file)
                else:
                    subprocess.run(['scanimage', '-d', device_name] + cmd_opts, stdout=subprocess.DEVNULL, check=True)
            self.log_signal.emit("✔ Warm-up complete. Lamp temperature stabilized.")
            self.finished_signal.emit(True, "Warmup Done")
        except Exception as e:
            self.log_signal.emit(f"❌ Warm-up failed: {e}")
            self.finished_signal.emit(False, str(e))

    def _run_scan(self):
        device_name = self.kwargs.get('device_name')
        final_filename = self.kwargs.get('filename')
        num_scans = self.kwargs.get('num_scans', 3)
        dpi = int(self.kwargs.get('res', 300))
        cmd_opts = self._build_cmd()
        
        temp_files = []
        try:
            for i in range(num_scans):
                self.log_signal.emit(f"Scanning pass {i+1} of {num_scans}...")
                tmp_file = tempfile.mktemp(suffix=".tif")
                if self.is_windows:
                    subprocess.run(['naps2.console', '/o', tmp_file, '/d', device_name] + cmd_opts, check=True)
                else:
                    process = subprocess.run(['scanimage', '-d', device_name] + cmd_opts, capture_output=True, check=True)
                    with open(tmp_file, 'wb') as f:
                        f.write(process.stdout)
                temp_files.append(tmp_file)

            self.log_signal.emit("Processing and averaging scans...")
            # arrays: list of (H, W, 3) float32 tömbök
            arrays = [tifffile.imread(f).astype(np.float32) for f in temp_files]
            stack = np.stack(arrays, axis=0) # Alak: (N, H, W, 3)
            avg_img = np.mean(stack, axis=0)
            avg_img = np.clip(avg_img, 0, 65535)
            std_img = np.std(stack, axis=0, ddof=1)

            # Pozíció lekérése a GUI beállításokból
            left_offset_mm = float(self.kwargs.get('left', 0.0))
            
            # Metaadat string generálása
            metadata_str = f"left_mm={left_offset_mm};scanner_center_mm=108.0"

            # 1. Átlagolt TIFF mentése
            tifffile.imwrite(final_filename, 
                             avg_img.astype(np.uint16),
                             resolution=(dpi, dpi),
                             description=metadata_str)

            # 2. Szóráskép mentése (lehet float32 TIFF vagy NumPy tömb)
            std_filename = final_filename.replace(".tif", "_scan_std.tif")
            tifffile.imwrite(std_filename,
                             std_img.astype(np.float32),
                             resolution=(dpi, dpi),
                             description="Pixel-wise standard deviation across scan repetitions")
            
            self.log_signal.emit(f"✔ Final TIFF & STD map saved: {os.path.basename(final_filename)}")
            self.finished_signal.emit(True, final_filename)
        except Exception as e:
            self.log_signal.emit(f"❌ Acquisition error: {e}")
            self.finished_signal.emit(False, str(e))
        finally:
            for f in temp_files:
                if os.path.exists(f): os.remove(f)

class FilmScannerDialog(QDialog):
    def __init__(self, target_directory=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PRISM - Radiochromic Film Scanner (EBT3)")
        self.resize(650, 750)
        self.is_windows = platform.system() == "Windows"
        self.scan_counter = 0

        if target_directory and os.path.isdir(target_directory):
            self.target_dir = os.path.abspath(target_directory)
        else:
            self.target_dir = os.path.expanduser("~")

        self._init_ui()
        self.find_scanners()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Scanners
        layout.addWidget(QLabel(f"<b>Available Scanners ({'NAPS2' if self.is_windows else 'SANE'}):</b>"))
        self.list_scanners = QListWidget()
        self.list_scanners.setFixedHeight(60)
        layout.addWidget(self.list_scanners)

        hbox_scan_btns = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Refresh Scanners")
        self.btn_refresh.clicked.connect(self.find_scanners)
        self.btn_warmup = QPushButton("🔥 Fast Warm-up (3x Blank)")
        self.btn_warmup.clicked.connect(self.start_warmup)
        hbox_scan_btns.addWidget(self.btn_refresh)
        hbox_scan_btns.addWidget(self.btn_warmup)
        layout.addLayout(hbox_scan_btns)

        # 2. Directory & Naming
        grp_dir = QGroupBox("Save Destination & Naming")
        lay_dir = QGridLayout(grp_dir)
        self.txt_dir = QLineEdit(self.target_dir)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_dir)
        lay_dir.addWidget(QLabel("Target Dir:"), 0, 0)
        lay_dir.addWidget(self.txt_dir, 0, 1)
        lay_dir.addWidget(self.btn_browse, 0, 2)

        lay_dir.addWidget(QLabel("Prefix:"), 1, 0)
        self.txt_prefix = QLineEdit("film")
        self.chk_date = QCheckBox("Append Date")
        self.chk_date.setChecked(True)
        
        # --- Kezdő index beállító SpinBox ---
        self.lbl_idx = QLabel("Next Idx:")
        self.spin_next_idx = QSpinBox()
        self.spin_next_idx.setRange(1, 9999)
        self.spin_next_idx.setValue(1)
        
        self.btn_reset_idx = QPushButton("Reset Index")
        self.btn_reset_idx.clicked.connect(self.reset_index)
        
        hbox_name = QHBoxLayout()
        hbox_name.addWidget(self.txt_prefix)
        hbox_name.addWidget(self.chk_date)
        hbox_name.addWidget(self.lbl_idx)
        hbox_name.addWidget(self.spin_next_idx)
        hbox_name.addWidget(self.btn_reset_idx)
        lay_dir.addLayout(hbox_name, 1, 1, 1, 2)
        layout.addWidget(grp_dir)

        # 3. Geometry
        grp_geo = QGroupBox("Scan Area & Geometry")
        lay_geo = QGridLayout(grp_geo)
        self.combo_profile = QComboBox()
        self.profiles = {
            "Standard Film (50.4 x 48.4 mm)": (136.0, 122.0, 50.4, 48.4),
            "Calibration Strip (30 x 150 mm)": (120.0, 50.0, 30.0, 150.0),
            "Full Bed Area": None,
            "Custom Region": None
        }
        self.combo_profile.addItems(list(self.profiles.keys()))
        self.combo_profile.currentIndexChanged.connect(self.on_profile_change)
        lay_geo.addWidget(QLabel("Preset Profile:"), 0, 0)
        lay_geo.addWidget(self.combo_profile, 0, 1, 1, 3)

        self.spins = {}
        for i, (lbl, key) in enumerate([("Left (mm):", "left"), ("Top (mm):", "top"), ("Width (mm):", "width"), ("Height (mm):", "height")]):
            row = 1 + (i // 2)
            col = (i % 2) * 2
            lay_geo.addWidget(QLabel(lbl), row, col)
            spin = QDoubleSpinBox()
            spin.setRange(0, 500)
            spin.setDecimals(1)
            self.spins[key] = spin
            lay_geo.addWidget(spin, row, col + 1)
        layout.addWidget(grp_geo)
        self.on_profile_change()

        # 4. Physics
        grp_phys = QGroupBox("Advanced Scanning (Physics & Averaging)")
        lay_phys = QGridLayout(grp_phys)
        lay_phys.addWidget(QLabel("Scans to Average:"), 0, 0)
        self.spin_scans = QSpinBox()
        self.spin_scans.setRange(1, 5)
        self.spin_scans.setValue(3)
        lay_phys.addWidget(self.spin_scans, 0, 1)

        lay_phys.addWidget(QLabel("DPI:"), 0, 2)
        self.combo_dpi = QComboBox()
        self.combo_dpi.addItems(["75", "150", "300", "600"])
        self.combo_dpi.setCurrentText("300")
        lay_phys.addWidget(self.combo_dpi, 0, 3)

        self.chk_raw = QCheckBox("Force RAW Mode (No Color Correction)")
        self.chk_raw.setChecked(True)
        lay_phys.addWidget(self.chk_raw, 1, 0, 1, 4)
        layout.addWidget(grp_phys)

        # 5. Run & Log
        self.btn_scan = QPushButton("▶ Start Acquisition (Scan & Average)")
        self.btn_scan.setStyleSheet("background-color: #2b8a3e; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_scan.clicked.connect(self.start_scan)
        layout.addWidget(self.btn_scan)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        layout.addWidget(self.log_text)

    def write_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def reset_index(self):
        self.scan_counter = 0
        self.write_log("Index reset. Next sequence index will be: 001")

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Target Directory", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def on_profile_change(self):
        sel = self.combo_profile.currentText()
        coords = self.profiles.get(sel)
        enabled = sel == "Custom Region"
        for key, spin in self.spins.items():
            if coords is not None:
                idx = ["left", "top", "width", "height"].index(key)
                spin.setValue(coords[idx])
            spin.setEnabled(enabled)

    def get_selected_scanner(self):
        items = self.list_scanners.selectedItems()
        if not items:
            if self.list_scanners.count() > 0:
                return self.list_scanners.item(0).text()
            self.write_log("Error: No scanner found!")
            return None
        return items[0].text()

    def set_ui_busy(self, busy):
        self.btn_scan.setEnabled(not busy)
        self.btn_warmup.setEnabled(not busy)
        self.btn_refresh.setEnabled(not busy)

    def find_scanners(self):
        self.set_ui_busy(True)
        self.list_scanners.clear()
        self.write_log("Searching for scanners...")
        self.worker = ScanWorker("find_scanners")
        self.worker.log_signal.connect(self.write_log)
        self.worker.finished_signal.connect(self._on_scanners_found)
        self.worker.start()

    def _on_scanners_found(self, success, data):
        self.set_ui_busy(False)
        if success and data:
            devices = data.split("\n")
            self.list_scanners.addItems(devices)
            self.list_scanners.setCurrentRow(0)
            self.write_log(f"Devices detected: {len(devices)}")
        else:
            self.write_log("Warning: No compatible scanner found.")

    def get_kwargs(self):
        return {
            'res': self.combo_dpi.currentText(),
            'left': self.spins['left'].value(),
            'top': self.spins['top'].value(),
            'width': self.spins['width'].value(),
            'height': self.spins['height'].value(),
            'profile': self.combo_profile.currentText(),
            'raw_mode': self.chk_raw.isChecked(),
            'num_scans': self.spin_scans.value()
        }

    def start_warmup(self):
        dev = self.get_selected_scanner()
        if not dev: return
        self.set_ui_busy(True)
        kwargs = self.get_kwargs()
        kwargs['device_name'] = dev
        self.worker = ScanWorker("warmup", **kwargs)
        self.worker.log_signal.connect(self.write_log)
        self.worker.finished_signal.connect(lambda s, msg: self.set_ui_busy(False))
        self.worker.start()

    def start_scan(self):
        dev = self.get_selected_scanner()
        if not dev: return
        target = self.txt_dir.text()
        if not os.path.isdir(target):
            QMessageBox.warning(self, "Error", "Target directory does not exist!")
            return

        self.scan_counter += 1
        base = self.txt_prefix.text().strip() or "film"
        if self.chk_date.isChecked():
            fname = f"{datetime.datetime.now().strftime('%Y%m%d')}_{base}_{self.scan_counter:03d}.tif"
        else:
            fname = f"{base}_{self.scan_counter:03d}.tif"
        full_path = os.path.join(target, fname)

        self.write_log(f"\n--- ACQUISITION STARTED: {fname} ---")
        self.set_ui_busy(True)
        
        kwargs = self.get_kwargs()
        kwargs['device_name'] = dev
        kwargs['filename'] = full_path
        
        self.worker = ScanWorker("scan", **kwargs)
        self.worker.log_signal.connect(self.write_log)
        self.worker.finished_signal.connect(lambda s, msg: self.set_ui_busy(False))
        self.worker.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    window = FilmScannerDialog(target_directory=target_dir)
    window.show()
    sys.exit(app.exec_())