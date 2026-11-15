import os
import json
import sys
import subprocess
import re
import datetime
import tempfile
import platform
import signal
import threading
import time
from pathlib import Path
import numpy as np
import tifffile
from scan_data import ScanAccumulator, next_scan_path, reserve_scan, publish_scan_files

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox, QGroupBox,
    QFileDialog, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

WINDOWS_SCAN_UNAVAILABLE = (
    'Integrated Windows acquisition is unavailable. PRISM requires 16 bits per RGB channel; '
    'the documented NAPS2 CLI color mode provides 8 bits per channel. '
    'Acquire a 16-bit RGB TIFF in your scanner software, copy it into the project raw folder, '
    'and refresh the file list.'
)

class ScanCancelled(RuntimeError):
    pass


class ScanWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs
        self.is_windows = platform.system() == 'Windows'
        self._cancelled = threading.Event()

    def request_cancel(self):
        self._cancelled.set()

    def _check_cancel(self):
        if self._cancelled.is_set():
            raise ScanCancelled('Acquisition cancelled.')

    def run(self):
        try:
            task = {'find_scanners': self._find_scanners, 'warmup': self._run_warmup,
                    'scan': self._run_scan}.get(self.task)
            if task is None:
                raise ValueError(f'Unknown scanner task: {self.task}')
            result = task()
        except ScanCancelled as error:
            self.log_signal.emit(str(error))
            self.finished_signal.emit(False, str(error))
        except Exception as error:
            self.log_signal.emit(f'Scanner task failed: {error}')
            self.finished_signal.emit(False, str(error))
        else:
            self.finished_signal.emit(True, result)

    def _stop_process(self, process):
        if self.is_windows and process.poll() is not None:
            return
        if self.is_windows:
            # Terminate only this worker's process tree, never other scanner tasks.
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if self.is_windows:
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=2)
        finally:
            if not self.is_windows:
                # The parent can exit before a descendant that ignored SIGTERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _run_command(self, command, stdout=None, capture_text=False, timeout_s=None):
        """Poll bounded scanner processes without buffering TIFF bytes in RAM."""
        self._check_cancel()
        timeout = float(timeout_s if timeout_s is not None else self.kwargs.get('timeout_s', 300))
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('The scanner timeout must be finite and positive.')
        with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as captured:
            destination = captured if capture_text else (stdout if stdout is not None else subprocess.DEVNULL)
            options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if self.is_windows
                       else {'start_new_session': True})
            with subprocess.Popen(command, stdout=destination, stderr=errors, **options) as process:
                deadline = time.monotonic() + timeout
                try:
                    while True:
                        self._check_cancel()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f'{command[0]} timed out after {timeout:g} seconds.')
                        try:
                            code = process.wait(timeout=min(0.1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    self._check_cancel()
                    if code:
                        errors.seek(0, os.SEEK_END)
                        errors.seek(max(0, errors.tell() - 65536))
                        detail = errors.read().decode('utf-8', errors='replace').strip()
                        raise RuntimeError(f'{command[0]} failed (exit {code}): {detail}')
                except BaseException:
                    self._stop_process(process)
                    raise
            if capture_text:
                captured.seek(0)
                return captured.read().decode('utf-8', errors='replace')
        return ''

    def _find_scanners(self):
        timeout = min(float(self.kwargs.get('timeout_s', 300)), 15)
        if self.is_windows:
            output = self._run_command(['naps2.console', '--listdevices', '--driver', 'twain'],
                                       capture_text=True, timeout_s=timeout)
            devices = [line.strip() for line in output.splitlines()
                       if line.strip() and not line.startswith('NAPS2')]
        else:
            output = self._run_command(['scanimage', '-L'], capture_text=True, timeout_s=timeout)
            devices = re.findall(r"device ['`](.*?)['`]", output)
        return '\n'.join(devices)

    def _build_cmd(self):
        res = int(self.kwargs.get('res', 300))
        l, t = float(self.kwargs.get('left', 0)), float(self.kwargs.get('top', 0))
        w, h = float(self.kwargs.get('width', 0)), float(self.kwargs.get('height', 0))
        profile = self.kwargs.get('profile', 'Full Bed Area')
        raw_mode = self.kwargs.get('raw_mode', True)
        if res <= 0 or not np.all(np.isfinite([l, t, w, h])) or min(l, t, w, h) < 0:
            raise ValueError('Resolution and scan geometry must be finite and valid.')
        cropped = profile != 'Full Bed Area'
        if cropped and (w <= 0 or h <= 0):
            raise ValueError('A cropped scan requires positive width and height.')
        if self.is_windows:
            # A 48-bit acquisition backend is required; no NAPS2 16-bit flag exists.
            cmd = []
        else:
            cmd = ['--format=tiff', '--mode=Color', '--depth=16', f'--resolution={res}',
                   '--source', 'Transparency Unit']
            if raw_mode:
                cmd.append('--color-correction=None')
            if cropped:
                cmd.extend(['-l', str(l), '-t', str(t), '-x', str(w), '-y', str(h)])
        return cmd

    def _acquire_pass(self, path, command_options):
        self._check_cancel()
        device = self.kwargs['device_name']
        if self.is_windows:
            raise RuntimeError(WINDOWS_SCAN_UNAVAILABLE)
        else:
            with Path(path).open('wb') as output:
                self._run_command(['scanimage', '-d', device] + command_options, stdout=output)

    def _run_warmup(self):
        options = self._build_cmd()
        self.log_signal.emit('Starting three lamp warm-up exposures...')
        with tempfile.TemporaryDirectory(prefix='prism-warmup-') as temporary:
            for i in range(3):
                self.log_signal.emit(f'Warm-up exposure {i + 1}/3...')
                self._acquire_pass(Path(temporary) / f'warmup_{i}.tif', options)
        self.log_signal.emit('Three warm-up exposures completed.')
        return 'Warmup Done'

    def _run_scan(self):
        final_path = Path(self.kwargs['filename'])
        count = int(self.kwargs.get('num_scans', 3))
        if count < 1:
            raise ValueError('At least one scan is required.')
        options = self._build_cmd()
        dpi = int(self.kwargs.get('res', 300))
        accumulator = ScanAccumulator()
        with reserve_scan(final_path) as paths, tempfile.TemporaryDirectory(
                prefix='.prism-scan-', dir=final_path.parent) as temporary:
            temporary = Path(temporary)
            for i in range(count):
                self._check_cancel()
                self.log_signal.emit(f'Scanning pass {i + 1} of {count}...')
                path = temporary / f'pass_{i}.tif'
                self._acquire_pass(path, options)
                self._check_cancel()
                accumulator.add(tifffile.imread(path))
                path.unlink()
            self._check_cancel()
            self.log_signal.emit('Computing average, repeat SD and SEM of the average...')
            mean, std, sem = accumulator.finish()
            cropped = self.kwargs.get('profile', 'Full Bed Area') != 'Full Bed Area'
            left = float(self.kwargs.get('left', 0)) if cropped else 0.0
            description = f'left_mm={left};scanner_center_mm=108.0;scan_count={count}'
            staged = []
            for key, image in (('std', std), ('sem', sem), ('mean', mean)):
                if image is None:
                    continue
                self._check_cancel()
                staging_path = temporary / f'{key}.tif'
                tifffile.imwrite(staging_path, image, photometric='rgb', resolution=(dpi, dpi),
                                 resolutionunit='INCH', description=description)
                staged.append((staging_path, paths[key]))
            metadata = {
                'format_version': 1, 'scan_count': count,
                'acquired_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'dpi': dpi, 'pixel_spacing_mm': 25.4 / dpi, 'left_mm': left,
                'scanner_center_mm': 108.0, 'device': self.kwargs['device_name'],
                'profile': self.kwargs.get('profile', 'Full Bed Area'),
                'mean_filename': paths['mean'].name,
                'repeat_std_filename': paths['std'].name if std is not None else None,
                'mean_sem_filename': paths['sem'].name if sem is not None else None,
                'noise_model': 'independent_repeat_sem' if sem is not None else 'unavailable_single_scan',
                'sem_assumption': 'independent repetitions; drift and correlation are not included',
            }
            meta_path = temporary / 'metadata.json'
            meta_path.write_text(json.dumps(metadata, indent=4, allow_nan=False), encoding='utf-8')
            staged.append((meta_path, paths['meta']))
            publish_scan_files(staged, self._check_cancel)
        self.log_signal.emit(f'Saved {final_path.name} ({count} pass(es)).')
        if std is None:
            self.log_signal.emit('Repeat SD and mean SEM are unavailable from a single scan; no noise TIFF was created.')
        return str(final_path)


class FilmScannerDialog(QDialog):
    def __init__(self, target_directory=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PRISM - Radiochromic Film Scanner (EBT3)")
        self.resize(650, 750)
        self.is_windows = platform.system() == "Windows"
        self.worker = None
        self._busy = False
        self._close_pending = False
        self._scan_index = None

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
        self.grp_dir = grp_dir
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
        self.grp_geo = grp_geo
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
        self.grp_phys = grp_phys
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
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(10, 1800)
        self.spin_timeout.setValue(300)
        self.spin_timeout.setSuffix(' s')
        lay_phys.addWidget(QLabel('Timeout per exposure:'), 2, 0)
        lay_phys.addWidget(self.spin_timeout, 2, 1, 1, 3)
        layout.addWidget(grp_phys)

        # 5. Run & Log
        self.btn_scan = QPushButton("▶ Start Acquisition (Scan & Average)")
        self.btn_scan.setStyleSheet("background-color: #2b8a3e; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_scan.clicked.connect(self.start_scan)
        layout.addWidget(self.btn_scan)
        self.btn_cancel = QPushButton('Cancel current task')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_task)
        layout.addWidget(self.btn_cancel)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        layout.addWidget(self.log_text)
        if self.is_windows:
            note = QLabel(WINDOWS_SCAN_UNAVAILABLE)
            note.setWordWrap(True)
            layout.addWidget(note)
            self.set_ui_busy(False)

    def write_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def reset_index(self):
        self.spin_next_idx.setValue(1)
        self.write_log('Index reset to 001. Existing or reserved filenames will be skipped.')

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
            elif sel == 'Full Bed Area':
                spin.setValue(0)
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
        self._busy = busy
        self.btn_scan.setEnabled(not busy and not self.is_windows)
        self.btn_warmup.setEnabled(not busy and not self.is_windows)
        self.btn_refresh.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.list_scanners.setEnabled(not busy)
        for group in (self.grp_dir, self.grp_geo, self.grp_phys):
            group.setEnabled(not busy)

    def _start_worker(self, task, **kwargs):
        if self._busy or self._close_pending:
            return
        self.set_ui_busy(True)
        self.worker = ScanWorker(task, **kwargs)
        self.worker.log_signal.connect(self.write_log)
        self.worker.finished_signal.connect(self._on_task_result)
        # Unlock only when the QThread has actually exited, not at its last log signal.
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_task_result(self, success, data):
        if self.worker.task == 'find_scanners':
            self._on_scanners_found(success, data)
        elif self.worker.task == 'scan' and success:
            self.spin_next_idx.setValue(min(self._scan_index + 1, self.spin_next_idx.maximum()))

    def _on_worker_finished(self):
        self.worker = None
        self.set_ui_busy(False)
        if self._close_pending:
            self.close()

    def cancel_task(self):
        if self.worker is not None:
            self.write_log('Cancellation requested; waiting for the scanner process to stop...')
            self.btn_cancel.setEnabled(False)
            self.worker.request_cancel()

    def closeEvent(self, event):
        if self._busy:
            self._close_pending = True
            self.cancel_task()
            event.ignore()
        else:
            event.accept()

    def reject(self):
        # Escape must take the same safe shutdown path as closing the window.
        if self._busy:
            self.close()
        else:
            super().reject()

    def find_scanners(self):
        if self._busy: return
        self.list_scanners.clear()
        self.write_log("Searching for scanners...")
        self._start_worker('find_scanners', timeout_s=self.spin_timeout.value())

    def _on_scanners_found(self, success, data):
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
            'num_scans': self.spin_scans.value(),
            'timeout_s': self.spin_timeout.value(),
        }

    def start_warmup(self):
        if self._busy: return
        dev = self.get_selected_scanner()
        if not dev: return
        kwargs = self.get_kwargs()
        kwargs['device_name'] = dev
        self._start_worker('warmup', **kwargs)

    def start_scan(self):
        if self._busy: return
        dev = self.get_selected_scanner()
        if not dev: return
        target = self.txt_dir.text()
        if not os.path.isdir(target):
            QMessageBox.warning(self, "Error", "Target directory does not exist!")
            return

        base = self.txt_prefix.text().strip() or "film"
        if self.chk_date.isChecked():
            base = f"{datetime.datetime.now().strftime('%Y%m%d')}_{base}"
        try:
            full_path, index = next_scan_path(target, base, self.spin_next_idx.value(), self.spin_next_idx.maximum())
        except ValueError as error:
            QMessageBox.warning(self, 'Cannot acquire scan', str(error))
            return
        if index != self.spin_next_idx.value():
            self.write_log(f'Occupied or reserved indexes skipped; using {index:03d}.')
        self.spin_next_idx.setValue(index)
        self._scan_index = index

        self.write_log(f"\n--- ACQUISITION STARTED: {full_path.name} ---")
        
        kwargs = self.get_kwargs()
        kwargs['device_name'] = dev
        kwargs['filename'] = str(full_path)
        
        self._start_worker('scan', **kwargs)


if __name__ == '__main__':
    from prism_app import main
    arguments = ['--scanner']
    if len(sys.argv) > 1:
        arguments.extend(['--target-directory', sys.argv[1]])
    sys.exit(main(arguments))
