"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: dosimetry.py (Extended with Pixel-wise Bayesian Uncertainty Propagation)

This module handles the processing of Gafchromic EBT3 radiochromic films:
1. Optical Density (OD) calculation & local OD noise estimation.
2. Standard and Bayesian MCMC calibration with parameter posterior/covariance export.
3. Precise dose AND uncertainty (std) mapping using a hybrid multi-channel solver + Jacobian covariance.
4. Artifact cleaning (bFDR).

Author: [Your Name / PRISM Team]
License: MIT
"""

import os
import json
import warnings
import numpy as np
import tifffile
import matplotlib
# A nem-interaktív 'Agg' motor beállítása a szálbiztos háttér-rajzoláshoz
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Scientific computing
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import median_filter, uniform_filter, sobel, label, zoom
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
    Core engine for Gafchromic EBT3 film dosimetry with full pixel-wise uncertainty propagation.
    """

    def __init__(self, output_folder: str = "results", batch_metadata: dict = None):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        self.calib_params = {} 
        self.calib_cov = {}          # ÚJ: 3x3 vagy 4x4 paraméter kovarianciamátrixok csatornánként
        self.calib_samples = {}      # ÚJ: Posterior paraméter-minták (M x 4) a gyors MC hibaterjedéshez
        self.calib_sigma_od = {}     # ÚJ: Kalibrációból becsült alap OD zaj csatornánként
        self.calib_traces = {}
        self.calib_doses = None
        self.calib_ods = None
        self.ratio_lut_func = None 
        self.batch_metadata = batch_metadata

    # =========================================================================
    # 0. CORE MATHEMATICS (OD CALCULATION & DERIVATIVES)
    # =========================================================================

    def _calculate_od_map(self, img: np.ndarray, blank_img: np.ndarray) -> np.ndarray:
        """
        Calculates the Optical Density (OD) map: OD = log10(Blank / Film)
        """
        img_safe = np.where(img <= 0, 1e-6, img)
        blank_safe = np.where(blank_img <= 0, 1e-6, blank_img)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            od_map = np.log10(blank_safe / img_safe)
            
        od_map = np.nan_to_num(od_map, nan=0.0, posinf=0.0, neginf=0.0)
        od_map[od_map < 0] = 0.0
        return od_map

    def _estimate_od_noise_map(self, raw_od_map: np.ndarray, filtered_od_map: np.ndarray) -> np.ndarray:
        """
        Becsli a lokális pixelenkénti OD mérési zajt a szűretlen és a mediánszűrt OD különbségéből,
        kiegészítve a kalibrációs alapzajjal (baseline scanner noise floor).
        """
        diff = np.abs(raw_od_map - filtered_od_map)
        # Levágjuk a meredek élekből/műtermékekből adódó kiugró különbségeket (max 0.015 OD zaj)
        diff_clipped = np.minimum(diff, 0.015)
        local_var = uniform_filter(diff_clipped ** 2, size=(5, 5, 1))
        
        noise_floor = np.array([
            min(self.calib_sigma_od.get('R', 0.003), 0.005),
            min(self.calib_sigma_od.get('G', 0.003), 0.005),
            min(self.calib_sigma_od.get('B', 0.004), 0.006)
        ])[None, None, :]
        
        return np.sqrt(np.maximum(local_var, noise_floor ** 2))

    @staticmethod
    def rational_func_od(d, a: float, b: float, c: float, e: float = 1.0):
        """Kompatibilis 3 és 4 paraméteres hívással is (*params kicsomagoláshoz)."""
        d_safe = np.maximum(d, 0.0)
        if e == 1.0:
            return a + (b * d_safe) / (d_safe + c)
        de = np.power(d_safe, e)
        return a + (b * de) / (de + c)

    @staticmethod
    def rational_func_od_3p(d, a: float, b: float, c: float):
        """Kifejezetten a scipy.optimize.curve_fit számára (csak 3 szabad paraméter)."""
        d_safe = np.maximum(d, 0.0)
        return a + (b * d_safe) / (d_safe + c)

    @staticmethod
    def rational_func_dod_dd(d, a: float, b: float, c: float, e: float = 1.0):
        """
        Analitikus parciális derivált a dózis szerint: d(OD)/dD = (b * c * e * D^(e-1)) / (D^e + c)^2
        """
        d_safe = np.maximum(d, 1e-6)
        de = np.power(d_safe, e)
        num = b * c * e * np.power(d_safe, e - 1.0)
        den = (de + c) ** 2
        return num / np.maximum(den, 1e-12)

    @staticmethod
    def inv_rational_func_dose(od, params):
        """Inverz racionális függvény: OD -> Dózis [Gy] aszimptota-védelemmel."""
        a, b, c = params[:3]
        e = params[3] if len(params) > 3 else 1.0
        
        max_valid_od = a + b - 1e-4
        od_safe = np.clip(od, a, max_valid_od)
        
        num = c * (od_safe - a)
        den = (a + b) - od_safe
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = num / np.maximum(den, 1e-12)
            ratio = np.maximum(ratio, 0.0)
            val = np.power(ratio, 1.0 / e)
        return np.nan_to_num(val, nan=0.0)

    # =========================================================================
    # 1. CALIBRATION (WITH COVARIANCE & POSTERIOR SAMPLES EXPORT)
    # =========================================================================

    def load_calibration(self, master_json_path: str):
        """Loads calibration parameters AND uncertainty structures from JSON."""
        if not os.path.exists(master_json_path):
            raise FileNotFoundError(f"Calibration file not found: {master_json_path}")
            
        with open(master_json_path, 'r') as f:
            data = json.load(f)
            self.calib_params = {k: np.array(v, dtype=np.float64) for k, v in data['fitted_params'].items()}
            
            if 'measured_ods' in data:
                self.calib_ods = data['measured_ods']
            if 'doses_gy' in data:
                self.calib_doses = np.array(data['doses_gy'], dtype=np.float64)
            if 'calib_cov' in data:
                self.calib_cov = {k: np.array(v, dtype=np.float64) for k, v in data['calib_cov'].items()}
            if 'calib_samples' in data:
                self.calib_samples = {k: np.array(v, dtype=np.float64) for k, v in data['calib_samples'].items()}
            if 'calib_sigma_od' in data:
                self.calib_sigma_od = {k: float(v) for k, v in data['calib_sigma_od'].items()}
            else:
                self.calib_sigma_od = {'R': 0.003, 'G': 0.003, 'B': 0.004}
                
            self._prepare_ratio_lut()
        print(f"  -> Calibration loaded from: {master_json_path} "
              f"(Posterior samples available: {len(self.calib_samples) > 0})")

    def run_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50, n_posterior_samples: int = 100):
        """
        Performs standard calibration using Scipy's curve_fit AND saves parameter covariance.
        """
        print(f"\n--- Running Standard Calibration: {calibration_folder} ---")
        folder = Path(calibration_folder)
        files = sorted(list(folder.glob("*.tif")))
        if len(files) == 0: raise ValueError("No .tif files found!")
        
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

        print("  -> Fitting curves & extracting covariance...")
        d_arr = np.array(doses_gy, dtype=np.float64)
        self.calib_doses = d_arr
        self.calib_ods = measured_ods 
        
        rng = np.random.default_rng(42)
        
        for i, ch in enumerate(['R', 'G', 'B']):
            ods = np.array(measured_ods[ch], dtype=np.float64)
            sigs = np.maximum(np.array(measured_sigs[ch]), 0.002)
            self.calib_sigma_od[ch] = float(np.median(sigs))
            
            max_od = float(np.max(ods)) if len(ods) > 0 else 0.2
            max_d = float(np.max(d_arr)) if len(d_arr) > 0 else 10.0
            
            # Kezdőértékek a csatorna meredekségéhez igazítva (kék csatornán nagyobb c kell)
            b0 = max(max_od * 2.5, 0.3)
            s0 = max(max_od / max(max_d, 1.0) * 1.2, 0.005)
            c0 = np.clip(b0 / s0, 2.0, 150.0)
            
            p0 = [float(ods[0]) if len(ods) > 0 else 0.0, b0, c0]
            bounds = ([-0.03, max_od * 1.02, 0.5], [0.03, 25.0, 500.0])
            
            try:
                popt3, pcov3 = curve_fit(
                    self.rational_func_od_3p, d_arr, ods, p0=p0, sigma=sigs, 
                    bounds=bounds, absolute_sigma=False, maxfev=10000
                )
                popt = np.array([popt3[0], popt3[1], popt3[2], 1.0])
                pcov = np.zeros((4, 4))
                pcov[:3, :3] = pcov3
                
                self.calib_params[ch] = popt
                self.calib_cov[ch] = pcov
                
                samples3 = rng.multivariate_normal(popt3, pcov3, size=n_posterior_samples)
                samples3[:, 1] = np.maximum(samples3[:, 1], max_od * 1.01)
                samples3[:, 2] = np.maximum(samples3[:, 2], 0.5)
                
                self.calib_samples[ch] = np.column_stack([samples3, np.ones(n_posterior_samples)])
            except Exception as e:
                print(f"  ERROR fitting channel {ch}: {e}")
        
        self._prepare_ratio_lut()
        
        meta = {
            "fitted_params": {k: v.tolist() for k, v in self.calib_params.items()},
            "calib_cov": {k: v.tolist() for k, v in self.calib_cov.items()},
            "calib_samples": {k: v.tolist() for k, v in self.calib_samples.items()},
            "calib_sigma_od": self.calib_sigma_od,
            "measured_ods": measured_ods,
            "blank_rgb": blank_roi_stats.tolist(),
            "doses_gy": doses_gy,
            "method": "Standard_CurveFit",
            "date": str(datetime.now())
        }
        out_path = os.path.join(self.output_folder, "calibration_master.json")
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=4)
        print(f"  -> Calibration & Uncertainty structures saved to {out_path}")

    def run_mcmc_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50, 
                             draws: int = 5000, chains: int = 4, use_cleaning: bool = True,
                             n_saved_samples: int = 100):
        """
        Performs Bayesian MCMC calibration using PyMC and saves posterior parameter realizations.
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

        d_arr = np.array(doses_gy, dtype=np.float64)
        self.calib_doses = d_arr
        self.calib_ods = measured_ods

        print("  -> Sampling posterior (PyMC)...")
        self.calib_params = {}
        self.calib_cov = {}
        self.calib_samples = {}
        self.calib_sigma_od = {}
        plot_data = [] 
        
        def rational_reparam(d, a, b, s0):
            return a + (b * s0 * d) / (s0 * d + b)

        rng = np.random.default_rng(42)

        for ch in ['R', 'G', 'B']:
            od_data = np.array(measured_ods[ch], dtype=np.float64)
            max_od = float(np.max(od_data)) if len(od_data) > 0 else 0.2
            max_d = float(np.max(d_arr)) if len(d_arr) > 0 else 10.0
            
            s0_guess = max(max_od / max(max_d, 1.0) * 1.5, 0.005)
            b_guess = max(max_od * 2.5, 0.3)
            
            try:
                popt_re, _ = curve_fit(
                    rational_reparam, d_arr, od_data, 
                    p0=[float(od_data[0]), b_guess, s0_guess],
                    bounds=([-0.03, max_od * 1.05, 1e-4], [0.03, 15.0, 2.0]), 
                    maxfev=10000
                )
                mu_a, mu_b, mu_s0 = [float(x) for x in popt_re]
            except Exception:
                mu_a, mu_b, mu_s0 = 0.0, b_guess, s0_guess

            b_low = max(max_od * 1.02, 0.05)
            mu_a_safe = float(np.clip(mu_a, -0.02, 0.02))
            mu_b_safe = float(np.clip(mu_b, b_low + 0.1, 12.0))
            mu_s0_safe = float(np.clip(mu_s0, 5e-4, 1.5))

            with pm.Model() as model:
                # Felső határ (upper) elhagyva -> stabil log-transzformáció, nincs overflow!
                a = pm.Normal('a', mu=mu_a_safe, sigma=0.01)
                b = pm.TruncatedNormal('b', mu=mu_b_safe, sigma=max(mu_b_safe * 0.4, 0.3), lower=b_low)
                s0 = pm.TruncatedNormal('s0', mu=mu_s0_safe, sigma=max(mu_s0_safe * 0.35, 0.015), lower=1e-4)
                
                c = pm.Deterministic('c', b / s0)
                sigma = pm.HalfNormal('sigma', sigma=0.005)
                
                mu_od = a + (b * d_arr) / (d_arr + c)
                pm.Normal('obs', mu=mu_od, sigma=sigma, observed=od_data)
                
                init_vals = {'a': mu_a_safe, 'b': mu_b_safe, 's0': mu_s0_safe, 'sigma': 0.003}
                trace = pm.sample(
                    draws=draws, tune=1500, chains=chains, cores=1, 
                    init='jitter+adapt_diag', target_accept=0.95, 
                    initvals=init_vals, progressbar=False
                )
                self.calib_traces[ch] = trace
                
                # Posterior minták kinyerése és mentése (a for cikluson BELÜL!)
                post = trace.posterior
                a_s = post['a'].values.flatten()
                b_s = post['b'].values.flatten()
                c_s = post['c'].values.flatten()
                sig_s = post['sigma'].values.flatten()
                e_s = np.ones_like(a_s)
                
                self.calib_params[ch] = np.array([
                    np.median(a_s), np.median(b_s), np.median(c_s), 1.0
                ])
                self.calib_sigma_od[ch] = float(np.median(sig_s))
                
                params_stack = np.vstack([a_s, b_s, c_s, e_s])
                self.calib_cov[ch] = np.cov(params_stack)
                
                idx_sub = rng.choice(len(a_s), size=min(n_saved_samples, len(a_s)), replace=False)
                self.calib_samples[ch] = params_stack[:, idx_sub].T
                
                trace_path = os.path.join(self.output_folder, f"calib_trace_{ch}.nc")
                if os.path.exists(trace_path): os.remove(trace_path)
                trace.to_netcdf(trace_path)
                
                plot_data.append({'ch': ch, 'trace': trace, 'doses': d_arr, 'ods': od_data})

        self._prepare_ratio_lut()
        
        meta = {
            "fitted_params": {k: v.tolist() for k, v in self.calib_params.items()},
            "calib_cov": {k: v.tolist() for k, v in self.calib_cov.items()},
            "calib_samples": {k: v.tolist() for k, v in self.calib_samples.items()},
            "calib_sigma_od": self.calib_sigma_od,
            "measured_ods": measured_ods,
            "blank_rgb": blank_roi_stats.tolist(),
            "doses_gy": doses_gy,
            "method": "MCMC_Hybrid_Init",
            "date": str(datetime.now())
        }
        with open(os.path.join(self.output_folder, "calibration_master.json"), 'w') as f:
            json.dump(meta, f, indent=4)
            
        print(f"  -> MCMC Calibration & {n_saved_samples} posterior realizations saved.")
        if len(plot_data) > 0:
            self._plot_mcmc_results(plot_data)

    def validate_calibration_curve(self, channel: str = 'red', plot_result: bool = True):
        """Back-calculates doses from calibration ODs to verify fit quality."""
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
        calculated_doses = self.inv_rational_func_dose(measured_ods, self.calib_params[ch_key])

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
    # 3. PROCESSING (DOSE & PIXEL-WISE STD MAPPING)
    # =========================================================================

    def _compute_calibration_std_map(self, od_map: np.ndarray, dose_map: np.ndarray) -> np.ndarray:
        """
        Kiszámolja a kalibrációs görbe bizonytalanságából eredő pixelenkénti szórást (sigma_calib).
        Robusztus percentilis módszert (16th - 84th) használ az elszálló MCMC aszimptoták szűrésére.
        """
        w_r = np.clip((10.0 - dose_map) / 4.0, 0.0, 1.0)
        w_g = 1.0 - w_r

        if 'R' in self.calib_samples and 'G' in self.calib_samples:
            samples_R = self.calib_samples['R']
            samples_G = self.calib_samples['G']
            n_samples = min(len(samples_R), len(samples_G))
            
            od_r = od_map[:, :, 0]
            od_g = od_map[:, :, 1]
            
            # Tömb az összes MCMC minta eredményének ideiglenes tárolására
            all_d_samples = np.zeros((n_samples, dose_map.shape[0], dose_map.shape[1]), dtype=np.float32)
            
            for i in range(n_samples):
                dr_i = self.inv_rational_func_dose(od_r, samples_R[i])
                dg_i = self.inv_rational_func_dose(od_g, samples_G[i])
                d_i = w_r * dr_i + w_g * dg_i
                
                # Fizikai biztonsági vágás: egyetlen MCMC minta sem prediktálhat 150 Gy-nél többet
                all_d_samples[i] = np.clip(d_i, 0.0, 150.0)
                
            # Robusztus szórás számítás a 16. és 84. percentilisből (+/- 1 szórás sáv)
            # Ez azonnal eldobja a felrobbanó (1000 Gy+) aszimptota mintákat!
            p16 = np.percentile(all_d_samples, 16, axis=0)
            p84 = np.percentile(all_d_samples, 84, axis=0)
            std_calib = (p84 - p16) / 2.0
            
            return np.nan_to_num(std_calib, nan=0.0)
        else:
            # Fallback régebbi calibration_master.json fájlokhoz (Analitikus Jacobi)
            dod_dd_r = self.rational_func_dod_dd(dose_map, *self.calib_params['R'])
            dod_dd_g = self.rational_func_dod_dd(dose_map, *self.calib_params['G'])
            
            sig_r = self.calib_sigma_od.get('R', 0.003) / np.maximum(dod_dd_r, 1e-4)
            sig_g = self.calib_sigma_od.get('G', 0.003) / np.maximum(dod_dd_g, 1e-4)
            
            std_calib = np.sqrt((w_r * sig_r)**2 + (w_g * sig_g)**2)
            std_calib[dose_map <= 1e-4] = 0.0
            return np.nan_to_num(std_calib, nan=0.0)

    def process_films(self, file_list: list, blank_path: str, method: str = 'hybrid', 
                      use_cleaning: bool = False, blank_rgb: tuple = None):
        """
        Batch processes a list of film scans into dose maps (*_dose.npy) AND 
        pixel-wise uncertainty maps (*_std.npy).
        """
        if not self.calib_params: raise ValueError("No calibration loaded!")
        
        print(f"\n--- Processing {len(file_list)} films (Method={method}, Cleaning={use_cleaning}) ---")
        
        if blank_rgb is None:
            blank_img_raw, _ = self._load_tiff_with_meta(blank_path)
            blank_img = median_filter(blank_img_raw, size=5)
        
        for idx, fpath in tqdm(enumerate(file_list), total=len(file_list)):
            fname = os.path.basename(fpath)
            base_name = os.path.splitext(fname)[0]
            
            img_raw, meta = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=3)
            
            if blank_rgb is not None:
                current_blank = np.ones_like(img) * np.array(blank_rgb)
            else:
                if img.shape != blank_img.shape:
                    scale_h = img.shape[0] / blank_img.shape[0]
                    scale_w = img.shape[1] / blank_img.shape[1]
                    current_blank = zoom(blank_img, (scale_h, scale_w, 1.0), order=1)
                else:
                    current_blank = blank_img
            
            # Nyers és szűrt OD térkép a lokális scanner-zaj becsléséhez
            raw_od_map = self._calculate_od_map(img_raw, current_blank)
            od_map = self._calculate_od_map(img, current_blank)
            
            if use_cleaning:
                od_map = self._clean_od_map_bfdr(od_map)
                
            od_noise_map = self._estimate_od_noise_map(raw_od_map, od_map)
            
            # 1. Dózis és illesztési/mérési szórás (std_fit) kiszámítása
            if method == 'hybrid':
                dose_map, std_fit_map = self._solver_hybrid(od_map, od_noise_map=od_noise_map, fname_log=base_name)
            else:
                dose_map, std_fit_map = self._solver_fast(od_map, od_noise_map=od_noise_map)
            
            # 2. Kalibrációs görbe bizonytalanságának (std_calib) kiszámítása
            std_calib_map = self._compute_calibration_std_map(od_map, dose_map)
            
            # 3. Teljes pixelenkénti szórás kombinálása: sigma_tot = sqrt(sigma_fit^2 + sigma_calib^2)
            std_total_map = np.sqrt(std_fit_map**2 + std_calib_map**2)
            
            # Medián szűrés a kiugró egyedi pixelhibák simítására
            dose_map = median_filter(dose_map, size=3)
            std_total_map = median_filter(std_total_map, size=3)
            
            # Fizikai korlátok biztosítása (ne legyen 0 vagy NaN szórás ott, ahol van dózis)
            min_std = np.where(dose_map > 0.01, dose_map * 0.005, 1e-4)
            std_total_map = np.clip(np.nan_to_num(std_total_map, nan=0.01), min_std, np.inf)
            
            # MENTÉS: *_dose.npy ÉS *_std.npy
            dose_out_path = os.path.join(self.output_folder, f"{base_name}_dose.npy")
            std_out_path = os.path.join(self.output_folder, f"{base_name}_std.npy")
            np.save(dose_out_path, dose_map)
            np.save(std_out_path, std_total_map)
            
            active_mask = dose_map > (0.1 * np.max(dose_map))
            mean_rel_std = float(np.mean(std_total_map[active_mask] / dose_map[active_mask]) * 100.0) if np.any(active_mask) else 0.0
            
            meta_out = {
                "original_filename": fname,
                "std_filename": f"{base_name}_std.npy",
                "pixel_spacing_mm": meta.get("pixel_spacing_mm", 0.0846),
                "max_dose": float(np.max(dose_map)),
                "mean_relative_uncertainty_pct": round(mean_rel_std, 2),
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

        print(f"Processing complete. Dose (*_dose.npy) and Uncertainty (*_std.npy) saved to: {self.output_folder}")

    def _solver_fast(self, od_map: np.ndarray, od_noise_map: np.ndarray = None):
        """
        Fast vectorized solver returning BOTH (dose_map, std_fit_map).
        """
        dr = self.inv_rational_func_dose(od_map[:, :, 0], self.calib_params['R'])
        dg = self.inv_rational_func_dose(od_map[:, :, 1], self.calib_params['G'])
        
        weight_r = np.clip((10.0 - dr) / 4.0, 0.0, 1.0)
        weight_g = 1.0 - weight_r
        dose_map = np.nan_to_num(weight_r * dr + weight_g * dg, nan=0.0)
        
        if od_noise_map is None:
            od_noise_map = np.full_like(od_map, 0.003)
            
        # Analitikus hibaterjedés: sigma_D = |dD/dOD| * sigma_OD + csatorna-eltérés
        dod_dd_r = self.rational_func_dod_dd(dr, *self.calib_params['R'])
        dod_dd_g = self.rational_func_dod_dd(dg, *self.calib_params['G'])
        
        std_r = od_noise_map[:, :, 0] / np.maximum(dod_dd_r, 1e-4)
        std_g = od_noise_map[:, :, 1] / np.maximum(dod_dd_g, 1e-4)
        
        # Ha a két csatorna (R és G) eltérő dózist mond az átmeneti zónában, az növeli a bizonytalanságot
        channel_discrepancy = weight_r * weight_g * np.abs(dr - dg)
        std_fit_map = np.sqrt((weight_r * std_r)**2 + (weight_g * std_g)**2 + channel_discrepancy**2)
        std_fit_map[dose_map <= 1e-4] = 0.0
        
        return dose_map, np.nan_to_num(std_fit_map, nan=0.0)

    def _solver_hybrid(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, fname_log: str = ""):
        """
        Full Hybrid 3-channel Micke solver returning (dose_map, std_fit_map).
        Computes exact 2x2 Jacobian covariance matrix (J^T W J)^(-1) at the optimum (D*, t*) for every pixel.
        """
        # 1. Warm Start
        dose_map, std_fit_map = self._solver_fast(od_map, od_noise_map)
        
        if od_noise_map is None:
            od_noise_map = np.full_like(od_map, 0.003)
            
        # 2. Masking (Threshold > 0.005 OD)
        mask = od_map[:, :, 0] > 0.005 
        pixels_y, pixels_x = np.where(mask)
        num_pixels = len(pixels_y)
        
        params_R = self.calib_params['R']
        params_G = self.calib_params['G']
        params_B = self.calib_params['B']
        
        def objective(x, meas_rgb):
            D, t = x 
            od_r = self.rational_func_od(D, *params_R) * t
            od_g = self.rational_func_od(D, *params_G) * t
            od_b = self.rational_func_od(D, *params_B) * t
            
            w_r = 1.0 if D < 8.0 else np.exp(-(D - 8.0) / 2.0)
            w_g = 0.5 if D < 8.0 else 1.0
            weights_dyn = np.array([w_r, w_g, 0.1])
            
            diff = np.array([od_r, od_g, od_b]) - meas_rgb
            return np.sum(weights_dyn * diff**2)

        bnds = ((0.0, 60.0), (0.9, 1.1))
        opts = {'ftol': 1e-4, 'maxiter': 10, 'disp': False}

        # 3. The Optimization Loop + Analytical Jacobian Covariance
        t_opt_array = np.ones(num_pixels, dtype=np.float64)
        d_opt_array = np.zeros(num_pixels, dtype=np.float64)
        
        for i in tqdm(range(num_pixels), desc=f"  Optimizing ({fname_log})", leave=False):
            y, x = pixels_y[i], pixels_x[i]
            meas_rgb = od_map[y, x, :]
            d_start = dose_map[y, x]
            if np.isnan(d_start) or d_start < 0: d_start = 0.0
            
            res = minimize(objective, [d_start, 1.0], args=(meas_rgb,), 
                           method='L-BFGS-B', bounds=bnds, options=opts)
            d_opt_array[i] = res.x[0]
            t_opt_array[i] = res.x[1]

        # Visszaírjuk az optimalizált dózist
        dose_map[pixels_y, pixels_x] = d_opt_array

        # 4. STABIL VEKTORIZÁLT BIZONYTALANSÁG SZÁMÍTÁS AZ AKTÍV PIXELEKRE
        if num_pixels > 0:
            D_v = d_opt_array
            t_v = t_opt_array
            meas_v = od_map[pixels_y, pixels_x, :]         # (N, 3)
            sig_od_v = od_noise_map[pixels_y, pixels_x, :] # (N, 3)

            # Parciális deriváltak: d(OD*t)/dD = t * dOD/dD
            dod_r = self.rational_func_dod_dd(D_v, *params_R) * t_v
            dod_g = self.rational_func_dod_dd(D_v, *params_G) * t_v
            dod_b = self.rational_func_dod_dd(D_v, *params_B) * t_v
            J_D = np.stack([dod_r, dod_g, dod_b], axis=1)  # (N, 3)

            # Dinamikus súlyok az optimumban
            w_r = np.where(D_v < 8.0, 1.0, np.exp(-(D_v - 8.0) / 2.0))
            w_g = np.where(D_v < 8.0, 0.5, 1.0)
            w_b = np.full_like(D_v, 0.1)
            W = np.stack([w_r, w_g, w_b], axis=1)          # (N, 3)

            # 1. Szkenner zajból eredő dózisbizonytalanság: Var(D_noise) = sum(w_i^2 * sig_D_i^2) / (sum w_i)^2
            # Ahol sig_D_i = sig_OD_i / (dOD_i/dD)
            sig_D_channels = sig_od_v / np.maximum(J_D, 1e-3)
            w_norm = W / np.sum(W, axis=1, keepdims=True)
            var_D_noise = np.sum((w_norm ** 2) * (sig_D_channels ** 2), axis=1)

            # 2. Csatornák közötti inkonzisztencia a vastagság-korrekció (t_v) után
            # Kiszámoljuk, milyen dózist adna az R és G csatorna a korrigált OD-ból (meas_v / t_v)
            dr_corr = self.inv_rational_func_dose(meas_v[:, 0] / t_v, params_R)
            dg_corr = self.inv_rational_func_dose(meas_v[:, 1] / t_v, params_G)
            
            # Csak az átmeneti/közös zónában súlyozzuk a két fő csatorna eltérését
            var_D_discrepancy = (w_norm[:, 0] * (dr_corr - D_v))**2 + (w_norm[:, 1] * (dg_corr - D_v))**2

            std_fit_map[pixels_y, pixels_x] = np.sqrt(var_D_noise + var_D_discrepancy)

        return dose_map, std_fit_map

    def _clean_od_map_bfdr(self, od_map, fdr_threshold=0.99, safety_limit=0.10, max_defect_size=500):
        """Artifact removal using bFDR and morphological protection."""
        ref_channel = od_map[:, :, 0] 
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
                
            cleaned_od = od_map.copy()
            coords_valid = np.array(np.nonzero(~mask)).T
            
            for i in range(3):
                channel = cleaned_od[:, :, i]
                values_valid = channel[~mask]
                grid_x, grid_y = np.nonzero(mask)
                if len(grid_x) > 0:
                    interp_vals = griddata(coords_valid, values_valid, (grid_x, grid_y), method='nearest')
                    channel[mask] = interp_vals
                cleaned_od[:, :, i] = channel
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
            
            mean_curve = np.median(curves, axis=0)
            hdi_low = np.percentile(curves, 2.5, axis=0)
            hdi_high = np.percentile(curves, 97.5, axis=0)
            
            ax.fill_between(d_plot, hdi_low, hdi_high, color='blue', alpha=0.2, label='95% HDI')
            ax.plot(d_plot, mean_curve, 'b-', label='MCMC Mean')
            ax.plot(d_meas, od_meas, 'ro', label='Measured')
            ax.set_title(f"Channel {ch}")
            if i == 0: ax.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.output_folder, "mcmc_calibration_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  -> Kalibrációs ábra elmentve: {plot_path}")

    def _plot_calibration_check(self, nominal, calculated, channel):
        """Plots measured vs calculated doses."""
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
        plot_color = color_map.get(channel, 'red')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        max_val = max(np.max(nominal), np.max(calculated)) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', label="Ideal (y=x)", alpha=0.5)
        ax1.scatter(nominal, calculated, color=plot_color, s=50, edgecolors='black', label='Measured')
        ax1.set_xlabel("Nominal Dose [Gy]")
        ax1.set_ylabel("Back-Calculated Dose [Gy]")
        ax1.set_title(f"Calibration Fit ({channel})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        errors = calculated - nominal
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.scatter(nominal, errors, color=plot_color, s=50, edgecolors='black')
        ax2.set_xlabel("Nominal Dose [Gy]")
        ax2.set_ylabel("Residual Error [Gy]")
        ax2.set_title("Residuals")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()