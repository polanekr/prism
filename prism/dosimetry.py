"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: dosimetry.py

This module handles the processing of Gafchromic EBT3 radiochromic films.
It preserves the original, validated logic for:
1. Optical Density (OD) calculation.
2. Standard and Bayesian MCMC calibration.
3. Precise dose mapping using a hybrid multi-channel solver (scipy.minimize).
4. Artifact cleaning (bFDR).

Author: [Your Name]
License: MIT
"""

import os
import json
import warnings
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Scientific computing
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import median_filter, sobel, label, zoom
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

# Bayesian inference (optional)
try:
    import pymc as pm
    import arviz as az
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    warnings.warn("PyMC/ArviZ not found. Bayesian calibration will be unavailable.")

class NumpyEncoder(json.JSONEncoder):
    """Helper to serialise numpy arrays to JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class GafchromicEngine:
    """
    Core engine for Gafchromic EBT3 film dosimetry.
    """

    def __init__(self, output_folder: str = "results", batch_metadata: dict = None):
        """
        Initialize the dosimetry engine.

        Args:
            output_folder (str): Directory to save processed dose maps and metadata.
            batch_metadata (dict, optional): Dictionary of metadata to append to all processed files.
        """
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        self.calib_params = {} 
        self.calib_doses = None
        self.calib_ods = None
        self.ratio_lut_func = None 
        self.batch_metadata = batch_metadata

    # =========================================================================
    # 0. CORE MATHEMATICS (OD CALCULATION)
    # =========================================================================

    def _calculate_od_map(self, img: np.ndarray, blank_img: np.ndarray) -> np.ndarray:
        """
        Calculates the Optical Density (OD) map.
        OD = log10(Blank / Film)
        """
        # Safety against division by zero
        img_safe = np.where(img <= 0, 1e-6, img)
        blank_safe = np.where(blank_img <= 0, 1e-6, blank_img)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            od_map = np.log10(blank_safe / img_safe)
            
        # Cleanup
        od_map = np.nan_to_num(od_map, nan=0.0, posinf=0.0, neginf=0.0)
        od_map[od_map < 0] = 0.0
        
        return od_map

    @staticmethod
    def rational_func_od(d: float, a: float, b: float, c: float, e: float = 1.0) -> float:
        """
        Rational function model (Mendez-Lewis type).
        OD = a + (b * D^e) / (D^e + c)
        """
        return a + (b * d**e) / (d**e + c)

    # =========================================================================
    # 1. CALIBRATION
    # =========================================================================

    def load_calibration(self, master_json_path: str):
        """Loads calibration parameters from a JSON file."""
        if not os.path.exists(master_json_path):
            raise FileNotFoundError(f"Calibration file not found: {master_json_path}")
            
        with open(master_json_path, 'r') as f:
            data = json.load(f)
            self.calib_params = {k: np.array(v) for k, v in data['fitted_params'].items()}
            
            # Restore calibration data points
            if 'measured_ods' in data:
                self.calib_ods = data['measured_ods']
            if 'doses_gy' in data:
                self.calib_doses = np.array(data['doses_gy'])
                
            self._prepare_ratio_lut()
        print(f"  -> Calibration loaded from: {master_json_path}")

    def run_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50):
        """
        Performs standard calibration using Scipy's curve_fit.
        """
        print(f"\n--- Running Standard Calibration: {calibration_folder} ---")
        folder = Path(calibration_folder)
        files = sorted(list(folder.glob("*.tif")))
        
        if len(files) == 0: raise ValueError("No .tif files found!")
        
        # Load Blank
        blank_img_raw, _ = self._load_tiff_with_meta(files[0])
        blank_img = median_filter(blank_img_raw, size=5)
        blank_roi_stats, _ = self._get_roi_stats(blank_img, size=roi_size)
        
        measured_ods = {'R': [], 'G': [], 'B': []}
        measured_sigs = {'R': [], 'G': [], 'B': []}

        print("  -> Reading film data...")
        for fpath in files:
            img_raw, _ = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=3) 
            
            od_map = self._calculate_od_map(img, blank_img)
            means, stds = self._get_roi_stats(od_map, size=roi_size)
            
            for i, ch in enumerate(['R', 'G', 'B']):
                measured_ods[ch].append(means[i])
                measured_sigs[ch].append(stds[i])

        print("  -> Fitting curves...")
        d_arr = np.array(doses_gy)
        
        self.calib_doses = d_arr
        self.calib_ods = measured_ods 
        
        for i, ch in enumerate(['R', 'G', 'B']):
            ods = measured_ods[ch]
            sigs = measured_sigs[ch]
            sigs = [s if s > 1e-6 else 1.0 for s in sigs]
            
            p0 = [0, max(ods)*2 if len(ods)>0 else 1, 10.0, 1.0]
            try:
                popt, _ = curve_fit(self.rational_func_od, d_arr, ods, p0=p0, sigma=sigs, maxfev=5000)
                self.calib_params[ch] = popt
            except Exception as e:
                print(f"  ERROR fitting channel {ch}: {e}")
        
        self._prepare_ratio_lut()
        
        # Save Calibration Master File
        meta = {
            "fitted_params": {k: v.tolist() for k,v in self.calib_params.items()},
            "measured_ods": measured_ods,
            "blank_rgb": blank_roi_stats.tolist(),
            "doses_gy": doses_gy,
            "method": "Standard_CurveFit",
            "date": str(datetime.now())
        }
        out_path = os.path.join(self.output_folder, "calibration_master.json")
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"  -> Calibration saved to {out_path}")

    def run_mcmc_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50, 
                             draws: int = 5000, chains: int = 4, use_cleaning: bool = True):
        """
        Performs Bayesian MCMC calibration using PyMC.
        """
        if not BAYES_AVAILABLE:
            print("  ERROR: PyMC is not available.")
            return

        print(f"\n--- Running MCMC Calibration: {calibration_folder} (Cleaning={use_cleaning}) ---")
        folder = Path(calibration_folder)
        files = sorted(list(folder.glob("*.tif")))
        if len(files) == 0: raise ValueError("No .tif files found!")

        print("  -> Loading images...")
        blank_img_raw, _ = self._load_tiff_with_meta(files[0])
        blank_img = median_filter(blank_img_raw, size=5)
        blank_roi_stats, _ = self._get_roi_stats(blank_img, size=roi_size)
        
        measured_ods = {'R': [], 'G': [], 'B': []}
        
        for fpath in files:
            img_raw, _ = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=3)
            od_map = self._calculate_od_map(img, blank_img)
            
            if use_cleaning:
                od_map = self._clean_od_map_bfdr(od_map)
            
            means_od, _ = self._get_roi_stats(od_map, size=roi_size)
            for i, ch in enumerate(['R', 'G', 'B']):
                measured_ods[ch].append(means_od[i])

        d_arr = np.array(doses_gy)
        self.calib_doses = d_arr
        self.calib_ods = measured_ods

        print("  -> Sampling posterior (PyMC)...")
        self.calib_params = {}
        plot_data = [] 
        
        def rational_fit_simple(d, a, b, c): return a + (b * d) / (d + c)

        for ch in ['R', 'G', 'B']:
            od_data = np.array(measured_ods[ch])
            
            try:
                p0_guess = [0.0, 5.0, 10.0]
                popt, _ = curve_fit(rational_fit_simple, d_arr, od_data, p0=p0_guess, 
                                    bounds=([0, 0, 0.01], [np.inf, np.inf, np.inf]), maxfev=5000)
                mu_a, mu_b, mu_c = popt
            except:
                mu_a, mu_b, mu_c = 0.0, 5.0, 10.0

            with pm.Model() as model:
                a = pm.Normal('a', mu=mu_a, sigma=0.05) 
                b = pm.TruncatedNormal('b', mu=mu_b, sigma=mu_b * 0.5 + 1.0, lower=0.01)
                c = pm.TruncatedNormal('c', mu=mu_c, sigma=mu_c * 0.5 + 1.0, lower=0.01)
                sigma = pm.HalfNormal('sigma', sigma=0.1)
                
                mu_od = a + (b * d_arr) / (d_arr + c)
                pm.Normal('obs', mu=mu_od, sigma=sigma, observed=od_data)
                
                trace = pm.sample(draws=draws, chains=chains, target_accept=0.97, progressbar=False)
                
                p_mean = az.summary(trace)['mean']
                self.calib_params[ch] = np.array([p_mean['a'], p_mean['b'], p_mean['c'], 1.0])
                
                trace_path = os.path.join(self.output_folder, f"calib_trace_{ch}.nc")
                if os.path.exists(trace_path): os.remove(trace_path)
                trace.to_netcdf(trace_path)
                
                plot_data.append({'ch': ch, 'trace': trace, 'doses': d_arr, 'ods': od_data})

        self._prepare_ratio_lut()
        
        meta = {
            "fitted_params": {k: v.tolist() for k,v in self.calib_params.items()},
            "measured_ods": measured_ods,
            "blank_rgb": blank_roi_stats.tolist(),
            "doses_gy": doses_gy,
            "method": "MCMC_Hybrid_Init",
            "date": str(datetime.now())
        }
        with open(os.path.join(self.output_folder, "calibration_master.json"), 'w') as f:
            json.dump(meta, f, indent=4)
            
        print(f"  -> MCMC Calibration saved.")
        self._plot_mcmc_results(plot_data)

    def validate_calibration_curve(self, channel: str = 'red', plot_result: bool = True):
        """
        Back-calculates doses from calibration ODs to verify the fit quality.
        """
        if not self.calib_params:
            print("Error: No calibration loaded!")
            return

        ch_key = channel[0].upper()
        if ch_key not in ['R', 'G', 'B']: ch_key = 'R'

        print(f"\n--- Validating Calibration ({ch_key} channel) ---")
        
        if self.calib_doses is None or self.calib_ods is None:
            print("Error: Calibration data missing.")
            return

        nominal_doses = self.calib_doses
        measured_ods = np.array(self.calib_ods[ch_key])
        params = self.calib_params[ch_key]
        
        a, b, c = params[:3]
        e = params[3] if len(params) > 3 else 1.0
        
        num = c * (measured_ods - a)
        den = (a + b) - measured_ods
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = num / den
            ratio[ratio < 0] = 0
            calculated_doses = np.power(ratio, 1/e)
            
        calculated_doses = np.nan_to_num(calculated_doses, nan=0.0)

        print(f"{'Nominal [Gy]':<15} | {'Measured OD':<12} | {'Calc [Gy]':<15} | {'Error [Gy]':<15}")
        print("-" * 75)
        
        abs_errors = []
        for nom, meas_od, calc in zip(nominal_doses, measured_ods, calculated_doses):
            abs_err = calc - nom
            abs_errors.append(abs_err)
            print(f"{nom:<15.2f} | {meas_od:<12.4f} | {calc:<15.2f} | {abs_err:<15.3f}")
            
        rmse = np.sqrt(np.mean(np.array(abs_errors)**2))
        print("-" * 75)
        print(f"RMSE: {rmse:.3f} Gy")
        
        if plot_result:
            self._plot_calibration_check(nominal_doses, calculated_doses, ch_key)
            
        return calculated_doses, abs_errors

    def _prepare_ratio_lut(self):
        if not self.calib_params: return
        d = np.linspace(0, 60, 6000)
        r = self.rational_func_od(d, *self.calib_params['R'])
        b = self.rational_func_od(d, *self.calib_params['B'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = r / b
        ratio = np.nan_to_num(ratio, nan=0.0)
        self.ratio_lut_func = interp1d(ratio, d, bounds_error=False, fill_value="extrapolate")

    # =========================================================================
    # 2. FILE I/O HELPERS
    # =========================================================================
    
    def _load_tiff_with_meta(self, path: str):
        """Loads TIFF and extracts DPI/Resolution metadata."""
        img = tifffile.imread(path).astype(np.float64)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        meta = {'pixel_spacing_mm': 0.0846} # Default 300 DPI
        try:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                if 'XResolution' in tags:
                    x_res = tags['XResolution'].value
                    dpi = x_res[0] / x_res[1] if isinstance(x_res, tuple) else float(x_res)
                    if dpi < 1: dpi = 72.0
                    unit = tags['ResolutionUnit'].value if 'ResolutionUnit' in tags else 2
                    if unit == 2: meta['pixel_spacing_mm'] = 25.4 / dpi
                    elif unit == 3: meta['pixel_spacing_mm'] = 10.0 / dpi
        except Exception: pass
        return img, meta

    def _get_roi_stats(self, img, center=None, size=50):
        h, w, c = img.shape
        if center is None: cx, cy = w // 2, h // 2
        else: cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - size); x2 = min(w, cx + size)
        y1 = max(0, cy - size); y2 = min(h, cy + size)
        roi = img[y1:y2, x1:x2, :]
        return np.mean(roi, axis=(0, 1)), np.std(roi, axis=(0, 1))

    # =========================================================================
    # 3. PROCESSING (DOSE MAPPING)
    # =========================================================================

    def process_films(self, file_list: list, blank_path: str, method: str = 'hybrid', use_cleaning: bool = False):
        """
        Batch processes a list of film scans into dose maps.
        
        Args:
            file_list: List of paths to scanned films.
            blank_path: Path to the blank (unexposed) film scan.
            method: 'hybrid' (slow, precise) or 'fast' (vectorized).
            use_cleaning: Apply artifact removal.
        """
        if not self.calib_params: raise ValueError("No calibration loaded!")
        
        print(f"\n--- Processing {len(file_list)} films (Method={method}, Cleaning={use_cleaning}) ---")
        
        blank_img_raw, _ = self._load_tiff_with_meta(blank_path)
        blank_img = median_filter(blank_img_raw, size=5)
        
        for idx, fpath in tqdm(enumerate(file_list), total=len(file_list)):
            fname = os.path.basename(fpath)
            base_name = os.path.splitext(fname)[0]
            
            img_raw, meta = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=3)
            
            # Auto-Resize Blank
            if img.shape != blank_img.shape:
                scale_h = img.shape[0] / blank_img.shape[0]
                scale_w = img.shape[1] / blank_img.shape[1]
                current_blank = zoom(blank_img, (scale_h, scale_w, 1.0), order=1)
            else:
                current_blank = blank_img
            
            od_map = self._calculate_od_map(img, current_blank)
            
            if use_cleaning:
                od_map = self._clean_od_map_bfdr(od_map)
            
            # --- SOLVER SELECTION ---
            if method == 'hybrid':
                dose_map = self._solver_hybrid(od_map, fname_log=base_name)
            else:
                dose_map = self._solver_fast(od_map)
            
            dose_map = median_filter(dose_map, size=3)
            
            np.save(os.path.join(self.output_folder, f"{base_name}_dose.npy"), dose_map)
            
            meta_out = {
                "original_filename": fname,
                "pixel_spacing_mm": meta.get("pixel_spacing_mm", 0.0846),
                "max_dose": float(np.max(dose_map)),
                "unit": "Gy",
                "method": method,
                "cleaning_applied": use_cleaning
            }
            
            if self.batch_metadata:
                for key, value in self.batch_metadata.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        if idx < len(value):
                            val_to_write = value[idx]
                            if hasattr(val_to_write, 'item'): val_to_write = val_to_write.item()
                            meta_out[key] = val_to_write
                    else:
                        meta_out[key] = value

            with open(os.path.join(self.output_folder, f"{base_name}_meta.json"), 'w') as f:
                json.dump(meta_out, f, cls=NumpyEncoder, indent=4)

        print(f"Processing complete. Results saved to: {self.output_folder}")

    def _solver_fast(self, od_map: np.ndarray) -> np.ndarray:
        """Fast vectorized solver."""
        def inv(od, params):
            a, b, c = params[:3]
            e = params[3] if len(params)>3 else 1.0
            num = c * (od - a)
            den = (a + b) - od
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = num / den
                ratio[ratio < 0] = 0
                val = np.power(ratio, 1/e)
            return np.nan_to_num(val, nan=0.0)
            
        dr = inv(np.clip(od_map[:,:,0], self.calib_params['R'][0], None), self.calib_params['R'])
        dg = inv(np.clip(od_map[:,:,1], self.calib_params['G'][0], None), self.calib_params['G'])
        return np.nan_to_num((dr + dg)/2, nan=0.0)

    def _solver_hybrid(self, od_map, fname_log=""):
        """
        The FULL Hybrid Implementation (Original Logic).
        1. Fast Estimate (Warm Start)
        2. Masking (Only active pixels)
        3. Scipy Minimize loop per pixel
        """
        h, w, c = od_map.shape
        
        # 1. Warm Start
        dose_map = self._solver_fast(od_map) 
        
        # 2. Masking (Threshold > 0.005 OD)
        mask = od_map[:,:,0] > 0.005 
        pixels_y, pixels_x = np.where(mask)
        num_pixels = len(pixels_y)
        
        params_R = self.calib_params['R']
        params_G = self.calib_params['G']
        params_B = self.calib_params['B']
        weights = np.array([1.0, 0.5, 0.1]) 
        
        # Closure for optimization
        def objective(x, meas_rgb):
            D, t = x 
            od_r = self.rational_func_od(D, *params_R) * t
            od_g = self.rational_func_od(D, *params_G) * t
            od_b = self.rational_func_od(D, *params_B) * t
            diff = np.array([od_r, od_g, od_b]) - meas_rgb
            return np.sum(weights * diff**2)

        bnds = ((0, 60), (0.9, 1.1))
        opts = {'ftol': 1e-4, 'maxiter': 10, 'disp': False}

        # 3. The Optimization Loop
        for i in tqdm(range(num_pixels), desc=f"  Optimizing ({fname_log})", leave=False):
            y, x = pixels_y[i], pixels_x[i]
            meas_rgb = od_map[y, x, :]
            d_start = dose_map[y, x]
            if np.isnan(d_start) or d_start < 0: d_start = 0.0
            x0 = [d_start, 1.0]

            res = minimize(objective, x0, args=(meas_rgb,), 
                           method='L-BFGS-B', bounds=bnds, options=opts)
            dose_map[y, x] = res.x[0]

        return dose_map

    def _clean_od_map_bfdr(self, od_map, fdr_threshold=0.99, safety_limit=0.10, max_defect_size=500):
        """
        Artifact removal using bFDR and morphological protection.
        """
        ref_channel = od_map[:,:,0] 
        gx = sobel(ref_channel, axis=0)
        gy = sobel(ref_channel, axis=1)
        grad_mag = np.hypot(gx, gy)
        
        data = grad_mag.flatten().reshape(-1, 1)
        if len(data) > 100000:
            sample_data = np.random.choice(data.flatten(), 100000).reshape(-1, 1)
        else:
            sample_data = data
            
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(sample_data)
            
            means = gmm.means_.flatten()
            noise_idx = np.argmax(means)
            
            probs = gmm.predict_proba(data)
            prob_noise = probs[:, noise_idx].reshape(ref_channel.shape)
            
            mask = prob_noise > fdr_threshold
            
            # Morphological Protection
            labeled_mask, num_features = label(mask)
            component_sizes = np.bincount(labeled_mask.ravel())
            too_big_labels = np.where(component_sizes > max_defect_size)[0]
            too_big_labels = too_big_labels[too_big_labels > 0]
            
            if len(too_big_labels) > 0:
                penumbra_pixels = np.isin(labeled_mask, too_big_labels)
                mask[penumbra_pixels] = False
                
            ratio = np.mean(mask)
            if ratio > safety_limit:
                print(f"  [Cleaning] Warning: Too many artifacts ({ratio*100:.1f}%). Skipping.")
                return od_map

            if np.sum(mask) == 0: return od_map 
                
            # Inpainting
            cleaned_od = od_map.copy()
            coords_valid = np.array(np.nonzero(~mask)).T
            
            for i in range(3):
                channel = cleaned_od[:,:,i]
                values_valid = channel[~mask]
                grid_x, grid_y = np.nonzero(mask)
                if len(grid_x) > 0:
                    interp_vals = griddata(coords_valid, values_valid, (grid_x, grid_y), method='nearest')
                    channel[mask] = interp_vals
                cleaned_od[:,:,i] = channel
            return cleaned_od

        except Exception as e:
            print(f"  [Cleaning] Error: {e}. Returning original.")
            return od_map

    # =========================================================================
    # 4. PLOTTING HELPERS
    # =========================================================================

    def _plot_mcmc_results(self, plot_data):
        """Visualizes MCMC traces."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        d_plot = np.linspace(0, max(plot_data[0]['doses']) * 1.1, 100)
        
        for i, data in enumerate(plot_data):
            ax = axes[i]
            ch = data['ch']
            trace = data['trace']
            d_meas = data['doses']
            od_meas = data['ods']
            
            post = trace.posterior
            a_s = post['a'].values.flatten()
            b_s = post['b'].values.flatten()
            c_s = post['c'].values.flatten()
            
            idxs = np.random.choice(len(a_s), size=min(500, len(a_s)), replace=False)
            A = a_s[idxs][:, None]; B = b_s[idxs][:, None]; C = c_s[idxs][:, None]
            D = d_plot[None, :]
            curves = A + (B * D) / (D + C)
            
            mean_curve = np.mean(curves, axis=0)
            hdi_low = np.percentile(curves, 2.5, axis=0)
            hdi_high = np.percentile(curves, 97.5, axis=0)
            
            ax.fill_between(d_plot, hdi_low, hdi_high, color='blue', alpha=0.2, label='95% HDI')
            ax.plot(d_plot, mean_curve, 'b-', label='MCMC Mean')
            ax.plot(d_meas, od_meas, 'ro', label='Measured')
            ax.set_title(f"Channel {ch}")
            if i == 0: ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_calibration_check(self, nominal, calculated, channel):
        """Plots measured vs calculated doses."""
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
        plot_color = color_map.get(channel, 'red')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Identity Plot
        max_val = max(np.max(nominal), np.max(calculated)) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', label="Ideal (y=x)", alpha=0.5)
        ax1.scatter(nominal, calculated, color=plot_color, s=50, edgecolors='black', label='Measured')
        ax1.set_xlabel("Nominal Dose [Gy]")
        ax1.set_ylabel("Back-Calculated Dose [Gy]")
        ax1.set_title(f"Calibration Fit ({channel})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residual Plot
        errors = calculated - nominal
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.scatter(nominal, errors, color=plot_color, s=50, edgecolors='black')
        ax2.set_xlabel("Nominal Dose [Gy]")
        ax2.set_ylabel("Residual Error [Gy]")
        ax2.set_title("Residuals")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# OPTIONAL UTILITY: Compare Methods
# =============================================================================

def compare_calibration_methods(engine, calib_folder, doses_gy, channel='red'):
    """
    Runs both Standard and MCMC calibration and compares them.
    """
    ch = channel[0].upper()
    
    print("=== 1. RUNNING STANDARD CALIBRATION ===")
    engine.run_calibration(calib_folder, doses_gy)
    params_std = engine.calib_params[ch].copy()
    _, errs_std = engine.validate_calibration_curve(channel=channel, plot_result=False)
    rmse_std = np.sqrt(np.mean(np.array(errs_std)**2))
    
    print("\n=== 2. RUNNING MCMC CALIBRATION ===")
    engine.run_mcmc_calibration(calib_folder, doses_gy, use_cleaning=False)
    params_mcmc = engine.calib_params[ch].copy()
    _, errs_mcmc = engine.validate_calibration_curve(channel=channel, plot_result=False)
    rmse_mcmc = np.sqrt(np.mean(np.array(errs_mcmc)**2))
    
    print(f"\n=== COMPARISON ({ch} channel) ===")
    print(f"Standard RMSE: {rmse_std:.4f} Gy")
    print(f"MCMC RMSE:     {rmse_mcmc:.4f} Gy")
    
    # Plot Comparison
    doses = np.array(engine.calib_doses)
    meas_ods = np.array(engine.calib_ods[ch])
    d_plot = np.linspace(0, max(doses)*1.1, 200)
    
    od_std = engine.rational_func_od(d_plot, *params_std)
    od_mcmc = engine.rational_func_od(d_plot, *params_mcmc)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(doses, meas_ods, color='black', s=60, label='Measured', zorder=5)
    plt.plot(d_plot, od_std, 'r--', lw=2, label=f'Standard Fit (RMSE={rmse_std:.3f})')
    plt.plot(d_plot, od_mcmc, 'b-', lw=2, alpha=0.7, label=f'MCMC Fit (RMSE={rmse_mcmc:.3f})')
    plt.xlabel("Dose [Gy]")
    plt.ylabel("OD")
    plt.title(f"Calibration Method Comparison ({ch})")
    plt.legend()
    plt.show()