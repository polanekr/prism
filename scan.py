import os
import sys
import subprocess
import re
import threading
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class FilmScannerApp(tk.Tk):
    def __init__(self, target_directory=None):
        super().__init__()
        self.title("PRISM - Radiochromic Film Scanner (EBT3)")
        self.geometry("680x740")

        self.scanned_files = []
        self.scan_counter = 0

        # Ha érkezett mappa argumentumként, azt használjuk, különben a felhasználói kezdőmappát
        if target_directory and os.path.isdir(target_directory):
            initial_dir = os.path.abspath(target_directory)
        else:
            initial_dir = os.path.expanduser("~")

        # Fő konténer
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Szkennerek listája
        ttk.Label(main_frame, text="Available Scanners:").pack(anchor=tk.W)
        self.scanner_list = tk.Listbox(main_frame, height=3, exportselection=False)
        self.scanner_list.pack(fill=tk.X, pady=(0, 5))
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        self.refresh_btn = ttk.Button(btn_frame, text="🔄 Refresh Scanners", command=self.find_scanners)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.warmup_btn = ttk.Button(btn_frame, text="🔥 Warm-up Lamp (3 Blank Scans)", command=self.start_warmup)
        self.warmup_btn.pack(side=tk.LEFT)

        # 2. Mentés helye
        dir_frame = ttk.LabelFrame(main_frame, text="Save Destination (Target Directory)", padding="8")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        self.dir_var = tk.StringVar(value=initial_dir)
        ttk.Entry(dir_frame, textvariable=self.dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_directory).pack(side=tk.LEFT)

        # 3. Fájlnév beállítások
        name_frame = ttk.LabelFrame(main_frame, text="Filename & Sequence Settings", padding="8")
        name_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(name_frame, text="Prefix:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.base_name_var = tk.StringVar(value="film")
        ttk.Entry(name_frame, textvariable=self.base_name_var, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        self.add_date_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(name_frame, text="Append Date (YYYYMMDD)", variable=self.add_date_var).grid(row=0, column=2, padx=15, sticky=tk.W)
        ttk.Button(name_frame, text="Reset Index", command=self.reset_counter).grid(row=0, column=3, padx=5, sticky=tk.E)

        # 4. Szkennelési Terület Profilok
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

        # 5. Felbontás & Hardver beállítások
        hw_frame = ttk.Frame(main_frame)
        hw_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(hw_frame, text="Resolution (DPI):").pack(side=tk.LEFT, padx=(0, 5))
        self.res_var = tk.StringVar(value="300")
        self.res_combo = ttk.Combobox(hw_frame, textvariable=self.res_var, values=["75", "150", "300", "600"], width=6, state="readonly")
        self.res_combo.pack(side=tk.LEFT, padx=(0, 20))

        self.raw_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hw_frame, text="Force RAW Mode (Disable Color Correction)", variable=self.raw_mode_var).pack(side=tk.LEFT)

        # 6. Futtatás gomb
        self.scan_btn = ttk.Button(main_frame, text="▶ Scan Next Film (16-bit TIFF)", command=self.start_scan)
        self.scan_btn.pack(fill=tk.X, pady=10, ipady=8)

        # 7. Log ablak
        self.log_text = tk.Text(main_frame, height=10, state=tk.DISABLED, bg="#f8f9fa", font=("Consolas", 9))
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
                self.log("Error: No available scanner detected!")
                return None
        return self.scanner_list.get(selection[0])

    def find_scanners(self):
        self.refresh_btn.config(state=tk.DISABLED)
        self.log("Searching for SANE scanners in the background...")
        self.scanner_list.delete(0, tk.END)
        threading.Thread(target=self._find_scanners_thread, daemon=True).start()

    def _find_scanners_thread(self):
        try:
            result = subprocess.run(['scanimage', '-L'], capture_output=True, text=True, check=True)
            devices = re.findall(r"device ['`](.*?)['`]", result.stdout)
            self.after(0, self._update_scanner_list, devices)
        except Exception as e:
            self.safe_log(f"Scanner discovery failed: {e}")
            self.after(0, lambda: self.refresh_btn.config(state=tk.NORMAL))

    def _update_scanner_list(self, devices):
        if devices:
            for dev in devices:
                self.scanner_list.insert(tk.END, dev)
            self.scanner_list.selection_set(0)
            self.log(f"Devices detected: {', '.join(devices)}")
        else:
            self.log("Warning: No SANE-compatible scanner found.")
        self.refresh_btn.config(state=tk.NORMAL)

    def build_base_command(self):
        res = self.res_var.get()
        cmd = ['--format=tiff', '--mode=Color', '--depth=16', f'--resolution={res}', '--source', 'Transparency Unit']
        
        if self.raw_mode_var.get():
            cmd.append('--color-correction=None')
            
        profile = self.profile_var.get()
        if profile != "Full Bed Area":
            l, t, x, y = self.left_var.get(), self.top_var.get(), self.width_var.get(), self.height_var.get()
            if x > 0 and y > 0:
                cmd.extend(['-l', str(l), '-t', str(t), '-x', str(x), '-y', str(y)])
        return cmd

    def start_warmup(self):
        device_name = self.get_selected_scanner()
        if not device_name:
            return
        
        command_options = self.build_base_command()
        self.scan_btn.config(state=tk.DISABLED)
        self.warmup_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_warmup_thread, args=(device_name, command_options), daemon=True).start()

    def run_warmup_thread(self, device_name, command_options):
        self.safe_log("\n--- STARTING LAMP WARM-UP ---")
        base_cmd = ['scanimage', '-d', device_name] + command_options
        try:
            for i in range(1, 4):
                self.safe_log(f"Blank exposure ({i}/3) in progress...")
                subprocess.run(base_cmd, capture_output=True, check=True, timeout=300)
            self.safe_log("Warm-up complete. Scanner lamp temperature stabilized.")
        except Exception as e:
            self.safe_log(f"Warm-up failed: {e}")
        finally:
            self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.warmup_btn.config(state=tk.NORMAL))

    def start_scan(self):
        device_name = self.get_selected_scanner()
        if not device_name:
            return

        directory = self.dir_var.get()
        if not os.path.isdir(directory):
            self.log(f"Error: Target directory does not exist: {directory}")
            return

        self.scan_counter += 1
        base_name = self.base_name_var.get().strip() or "film"
        if self.add_date_var.get():
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            filename_str = f"{date_str}_{base_name}_{self.scan_counter:03d}.tif"
        else:
            filename_str = f"{base_name}_{self.scan_counter:03d}.tif"
            
        current_filename = os.path.join(directory, filename_str)
        self.log("\n--- INITIATING FILM SCAN ---")
        self.log(f"Destination: {current_filename}")

        command_options = self.build_base_command()
        self.scan_btn.config(state=tk.DISABLED)
        self.warmup_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.run_scan_thread, args=(device_name, command_options, current_filename), daemon=True).start()

    def run_scan_thread(self, device_name, command_options, current_filename):
        command = ['scanimage', '-d', device_name] + command_options
        try:
            process = subprocess.run(command, capture_output=True, check=True, timeout=300)
            with open(current_filename, 'wb') as f:
                f.write(process.stdout)
            self.safe_log(f"✔ Scan completed successfully! Saved to: {os.path.basename(current_filename)}")
            self.scanned_files.append(current_filename)
        except subprocess.CalledProcessError as e:
            self.safe_log(f"❌ Scanner command failed:\n{e.stderr.decode('utf-8', 'ignore')}")
        except Exception as e:
            self.safe_log(f"❌ Scan error: {str(e)}")
        finally:
            self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.warmup_btn.config(state=tk.NORMAL))

if __name__ == '__main__':
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    app = FilmScannerApp(target_directory=target_dir)
    app.mainloop()