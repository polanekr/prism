import os
import sys
import subprocess
import re
import threading
import datetime
import tempfile
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import tifffile

class FilmScannerApp(tk.Tk):
    def __init__(self, target_directory=None):
        super().__init__()
        self.title("PRISM - Radiochromic Film Scanner (EBT3)")
        self.geometry("720x820")
        
        # Operációs rendszer detektálása (Windows = NAPS2, Linux = SANE)
        self.is_windows = platform.system() == "Windows"

        self.scanned_files = []
        self.scan_counter = 0

        if target_directory and os.path.isdir(target_directory):
            initial_dir = os.path.abspath(target_directory)
        else:
            initial_dir = os.path.expanduser("~")

        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. SZKENNEREK LISTÁJA ---
        ttk.Label(main_frame, text=f"Available Scanners ({'NAPS2 / WIA' if self.is_windows else 'SANE'}):").pack(anchor=tk.W)
        self.scanner_list = tk.Listbox(main_frame, height=3, exportselection=False)
        self.scanner_list.pack(fill=tk.X, pady=(0, 5))
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        self.refresh_btn = ttk.Button(btn_frame, text="🔄 Refresh Scanners", command=self.find_scanners)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.warmup_btn = ttk.Button(btn_frame, text="🔥 Fast Warm-up (3x Blank)", command=self.start_warmup)
        self.warmup_btn.pack(side=tk.LEFT)

        # --- 2. MENTÉS HELYE ---
        dir_frame = ttk.LabelFrame(main_frame, text="Save Destination (Target Directory)", padding="8")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        self.dir_var = tk.StringVar(value=initial_dir)
        ttk.Entry(dir_frame, textvariable=self.dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_directory).pack(side=tk.LEFT)

        # --- 3. FÁJLNÉV BEÁLLÍTÁSOK ---
        name_frame = ttk.LabelFrame(main_frame, text="Filename & Sequence Settings", padding="8")
        name_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(name_frame, text="Prefix:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.base_name_var = tk.StringVar(value="film")
        ttk.Entry(name_frame, textvariable=self.base_name_var, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)
        self.add_date_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(name_frame, text="Append Date", variable=self.add_date_var).grid(row=0, column=2, padx=15, sticky=tk.W)
        ttk.Button(name_frame, text="Reset Index", command=self.reset_counter).grid(row=0, column=3, padx=5, sticky=tk.E)

        # --- 4. SZKENNELÉSI TERÜLET ---
        area_frame = ttk.LabelFrame(main_frame, text="Scan Area & Geometry", padding="8")
        area_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(area_frame, text="Preset Profile:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.profile_var = tk.StringVar()
        self.profiles = {
            "Standard Film (50.4 x 48.4 mm)": (136.0, 122.0, 50.4, 48.4),
            "Calibration Strip (30 x 150 mm)": (120.0, 50.0, 30.0, 150.0),
            "Full Bed Area": None,
            "Custom Region": None
        }
        self.profile_combo = ttk.Combobox(area_frame, textvariable=self.profile_var, values=list(self.profiles.keys()), state="readonly")
        self.profile_combo.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W)
        self.profile_combo.set("Standard Film (50.4 x 48.4 mm)")
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_change)

        self.left_var, self.top_var = tk.DoubleVar(), tk.DoubleVar()
        self.width_var, self.height_var = tk.DoubleVar(), tk.DoubleVar()

        ttk.Label(area_frame, text="Left (mm):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.E)
        self.left_spin = ttk.Spinbox(area_frame, from_=0.0, to=500.0, increment=0.1, textvariable=self.left_var, width=8)
        self.left_spin.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(area_frame, text="Top (mm):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.E)
        self.top_spin = ttk.Spinbox(area_frame, from_=0.0, to=500.0, increment=0.1, textvariable=self.top_var, width=8)
        self.top_spin.grid(row=1, column=3, padx=5, pady=2)
        ttk.Label(area_frame, text="Width (mm):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.E)
        self.width_spin = ttk.Spinbox(area_frame, from_=0.0, to=500.0, increment=0.1, textvariable=self.width_var, width=8)
        self.width_spin.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(area_frame, text="Height (mm):").grid(row=2, column=2, padx=5, pady=2, sticky=tk.E)
        self.height_spin = ttk.Spinbox(area_frame, from_=0.0, to=500.0, increment=0.1, textvariable=self.height_var, width=8)
        self.height_spin.grid(row=2, column=3, padx=5, pady=2)

        # --- 5. ORVOSFIZIKAI BEÁLLÍTÁSOK (LRA & ÁTLAGOLÁS) ---
        physics_frame = ttk.LabelFrame(main_frame, text="Advanced Scanning (Noise Reduction & LRA)", padding="8")
        physics_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(physics_frame, text="Scans to Average:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.num_scans_var = tk.IntVar(value=3)
        ttk.Spinbox(physics_frame, from_=1, to=5, textvariable=self.num_scans_var, width=5).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(physics_frame, text="DPI:").grid(row=0, column=2, padx=(15,5), sticky=tk.W)
        self.res_var = tk.StringVar(value="300")
        ttk.Combobox(physics_frame, textvariable=self.res_var, values=["75", "150", "300", "600"], width=6, state="readonly").grid(row=0, column=3, padx=5, sticky=tk.W)

        self.lra_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(physics_frame, text="Apply LRA (Lateral Response Artifact) Correction", variable=self.lra_var).grid(row=1, column=0, columnspan=4, pady=(5,0), padx=5, sticky=tk.W)

        self.raw_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(physics_frame, text="Force RAW Mode (No Color Correction)", variable=self.raw_mode_var).grid(row=2, column=0, columnspan=4, pady=(2,0), padx=5, sticky=tk.W)

        # --- 6. FUTTATÁS ÉS LOG ---
        self.scan_btn = ttk.Button(main_frame, text="▶ Start Acquisition (Scan & Average)", command=self.start_scan)
        self.scan_btn.pack(fill=tk.X, pady=10, ipady=8)

        self.log_text = tk.Text(main_frame, height=9, state=tk.DISABLED, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.on_profile_change()
        self.after(500, self.find_scanners)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def safe_log(self, message):
        self.after(0, self.log, message)

    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.dir_var.get(), title="Select Target Directory")
        if directory:
            self.dir_var.set(directory)

    def reset_counter(self):
        self.scan_counter = 0
        self.log("Index reset. Next sequence index will be: 001")

    def on_profile_change(self, event=None):
        selection = self.profile_var.get()
        coords = self.profiles.get(selection)
        if coords is not None:
            self.left_var.set(coords[0])
            self.top_var.set(coords[1])
            self.width_var.set(coords[2])
            self.height_var.set(coords[3])
            state = "disabled"
        else:
            state = "normal" if selection == "Custom Region" else "disabled"
        for widget in [self.left_spin, self.top_spin, self.width_spin, self.height_spin]:
            widget.config(state=state)

    def get_selected_scanner(self):
        selection = self.scanner_list.curselection()
        if not selection:
            if self.scanner_list.size() > 0:
                return self.scanner_list.get(0)
            else:
                self.log("Error: No scanner selected or available!")
                return None
        return self.scanner_list.get(selection[0])

    def find_scanners(self):
        self.refresh_btn.config(state=tk.DISABLED)
        self.log("Searching for scanners...")
        self.scanner_list.delete(0, tk.END)
        threading.Thread(target=self._find_scanners_thread, daemon=True).start()

    def _find_scanners_thread(self):
        try:
            devices = []
            if self.is_windows:
                result = subprocess.run(['naps2.console', '/getdevices'], capture_output=True, text=True, check=True)
                for line in result.stdout.splitlines():
                    if line.strip() and not line.startswith("NAPS2"):
                        devices.append(line.strip())
            else:
                result = subprocess.run(['scanimage', '-L'], capture_output=True, text=True, check=True)
                devices = re.findall(r"device ['`](.*?)['`]", result.stdout)
            self.after(0, self._update_scanner_list, devices)
        except Exception as e:
            self.safe_log(f"Scanner discovery failed: {e}. Is NAPS2/SANE installed in PATH?")
            self.after(0, lambda: self.refresh_btn.config(state=tk.NORMAL))

    def _update_scanner_list(self, devices):
        if devices:
            for dev in devices:
                self.scanner_list.insert(tk.END, dev)
            self.scanner_list.selection_set(0)
            self.log(f"Devices detected: {len(devices)}")
        else:
            self.log("Warning: No compatible scanner found.")
        self.refresh_btn.config(state=tk.NORMAL)

    def build_base_command(self):
        res = self.res_var.get()
        profile = self.profile_var.get()
        l, t, x, y = self.left_var.get(), self.top_var.get(), self.width_var.get(), self.height_var.get()
        
        if self.is_windows:
            cmd = ['/r', str(res), '/bitdepth', '16']
            if profile != "Full Bed Area" and x > 0 and y > 0:
                cmd.extend(['/crop', f'{l}mm,{t}mm,{x}mm,{y}mm'])
        else:
            cmd = ['--format=tiff', '--mode=Color', '--depth=16', f'--resolution={res}', '--source', 'Transparency Unit']
            if self.raw_mode_var.get():
                cmd.append('--color-correction=None')
            if profile != "Full Bed Area" and x > 0 and y > 0:
                cmd.extend(['-l', str(l), '-t', str(t), '-x', str(x), '-y', str(y)])
        return cmd

    def start_warmup(self):
        device_name = self.get_selected_scanner()
        if not device_name: return
        self.scan_btn.config(state=tk.DISABLED)
        self.warmup_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_warmup_thread, args=(device_name,), daemon=True).start()

    def run_warmup_thread(self, device_name):
        self.safe_log("\n--- STARTING LAMP WARM-UP ---")
        cmd_opts = self.build_base_command()
        
        try:
            for i in range(1, 4):
                self.safe_log(f"Blank exposure ({i}/3) in progress...")
                if self.is_windows:
                    # Windows: Mentés átmeneti fájlba, majd azonnali törlés (RAM és disk kímélése)
                    tmp_file = tempfile.mktemp(suffix=".tif")
                    subprocess.run(['naps2.console', '/o', tmp_file, '/d', device_name] + cmd_opts, check=True)
                    if os.path.exists(tmp_file): os.remove(tmp_file)
                else:
                    # Linux: stdout irányítása a semmibe (DEVNULL) -> Nem eszi a RAM-ot!
                    subprocess.run(['scanimage', '-d', device_name] + cmd_opts, stdout=subprocess.DEVNULL, check=True)
                    
            self.safe_log("✔ Warm-up complete. Lamp temperature stabilized.")
        except Exception as e:
            self.safe_log(f"❌ Warm-up failed: {e}")
        finally:
            self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.warmup_btn.config(state=tk.NORMAL))

    def start_scan(self):
        device_name = self.get_selected_scanner()
        if not device_name: return
        directory = self.dir_var.get()
        if not os.path.isdir(directory):
            self.log("Error: Target directory does not exist!")
            return

        self.scan_counter += 1
        base = self.base_name_var.get().strip() or "film"
        if self.add_date_var.get():
            filename_str = f"{datetime.datetime.now().strftime('%Y%m%d')}_{base}_{self.scan_counter:03d}.tif"
        else:
            filename_str = f"{base}_{self.scan_counter:03d}.tif"
            
        current_filename = os.path.join(directory, filename_str)
        self.log(f"\n--- ACQUISITION STARTED: {filename_str} ---")

        self.scan_btn.config(state=tk.DISABLED)
        self.warmup_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.run_scan_thread, args=(device_name, current_filename), daemon=True).start()

    def run_scan_thread(self, device_name, final_filename):
        cmd_opts = self.build_base_command()
        num_scans = self.num_scans_var.get()
        apply_lra = self.lra_var.get()
        dpi = int(self.res_var.get())
        left_offset_mm = self.left_var.get()
        
        temp_files = []
        try:
            # 1. TÖBBSZÖRÖS SZKENNELÉS
            for i in range(num_scans):
                self.safe_log(f"Scanning pass {i+1} of {num_scans}...")
                tmp_file = tempfile.mktemp(suffix=".tif")
                
                if self.is_windows:
                    subprocess.run(['naps2.console', '/o', tmp_file, '/d', device_name] + cmd_opts, check=True)
                else:
                    process = subprocess.run(['scanimage', '-d', device_name] + cmd_opts, capture_output=True, check=True)
                    with open(tmp_file, 'wb') as f:
                        f.write(process.stdout)
                temp_files.append(tmp_file)

            # 2. ÁTLAGOLÁS (Szenzorzaj csökkentése)
            self.safe_log("Processing and averaging scans...")
            arrays = [tifffile.imread(f).astype(np.float32) for f in temp_files]
            avg_img = np.mean(arrays, axis=0)
            
            # 3. LRA (LATERAL RESPONSE ARTIFACT) KORREKCIÓ
            if apply_lra and avg_img.ndim >= 2:
                self.safe_log("Applying geometric LRA flat-field correction...")
                h, w = avg_img.shape[:2]
                
                # A szkenner optikai tengelye (Epson gépeknél az üveglap közepe) ~ 108 mm
                # Kiszámítjuk minden pixel fizikai X koordinátáját (mm)
                x_mm = left_offset_mm + np.arange(w) * (25.4 / dpi)
                dist_from_center = x_mm - 108.0
                
                # EBT3 LRA intenzitás korrekció (Tipikus Epson paraméter: 1.5e-5)
                # A fényszórás miatt a széleken alacsonyabb a mért intenzitás (hamis sötétedés/magas OD)
                # Ezzel a szorzóval kompenzáljuk az intenzitást a szélek felé haladva.
                lra_factor = 1.0 + 1.5e-5 * (dist_from_center ** 2)
                
                # A szorzót alkalmazzuk minden pixeloszlopra (színcsatornától függetlenül)
                if avg_img.ndim == 3:
                    avg_img = avg_img * lra_factor[None, :, None]
                else:
                    avg_img = avg_img * lra_factor[None, :]
                    
            # 4. MENTÉS
            avg_img = np.clip(avg_img, 0, 65535)
            tifffile.imwrite(final_filename, avg_img.astype(np.uint16))
            self.safe_log(f"✔ Final TIFF saved: {os.path.basename(final_filename)}")
            self.scanned_files.append(final_filename)

        except Exception as e:
            self.safe_log(f"❌ Acquisition error: {str(e)}")
        finally:
            # Takarítás
            for f in temp_files:
                if os.path.exists(f): os.remove(f)
            self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.warmup_btn.config(state=tk.NORMAL))


if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    app = FilmScannerApp(target_directory=target_dir)
    app.mainloop()