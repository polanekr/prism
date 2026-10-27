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

from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic
from scipy.ndimage import distance_transform_edt

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
        """
        Általánosított racionális (Hill/Mendez-Lewis) modell:
        OD(D) = a + (b * D^e) / (D^e + c^e)  (vagy c_eff = c^e kompatibilitással)
        """
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
        Analitikus parciális derivált a dózis szerint:
        d(OD)/dD = (b * c * e * D^(e-1)) / (D^e + c)^2
        """
        d_safe = np.maximum(d, 1e-4)
        if e == 1.0:
            return (b * c) / np.maximum((d_safe + c) ** 2, 1e-12)
        de = np.power(d_safe, e)
        num = b * c * e * np.power(d_safe, e - 1.0)
        den = (de + c) ** 2
        return num / np.maximum(den, 1e-12)

    @staticmethod
    def inv_rational_func_dose(od, params):
        """Inverz racionális függvény: OD -> Dózis [Gy] aszimptota-védelemmel."""
        a, b, c = params[:3]
        e = params[3] if len(params) > 3 else 1.0
        
        max_valid_od = a + b * 0.995
        od_safe = np.clip(od, a, max_valid_od)
        
        num = c * (od_safe - a)
        den = (a + b) - od_safe
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.maximum(num / np.maximum(den, 1e-12), 0.0)
            val = ratio if e == 1.0 else np.power(ratio, 1.0 / e)
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
            # Nullapont korrekció (D=0-nál netOD=0)
            if len(ods) > 0 and d_arr[0] == 0:
                ods = np.maximum(0.0, ods - ods[0])
                measured_ods[ch] = ods.tolist()

            sigs = np.maximum(np.array(measured_sigs[ch]), 0.002)
            self.calib_sigma_od[ch] = float(np.median(sigs))
            
            max_od = max(float(np.max(ods)), 0.05) if len(ods) > 0 else 0.2
            max_d = max(float(np.max(d_arr)), 10.0) if len(d_arr) > 0 else 10.0
            
            b_min = max_od * 1.02
            b_max = max(25.0, max_od * 10.0)
            b0 = np.clip(max_od * 2.5, b_min + 0.05, b_max - 0.1)
            
            s0 = max(max_od / max_d * 1.2, 0.005)
            c0 = float(np.clip(b0 / s0, 1.0, 450.0))
            a0 = float(np.clip(ods[0] if len(ods) > 0 else 0.0, -0.025, 0.025))
            
            p0 = [a0, b0, c0]
            bounds = ([-0.03, b_min, 0.5], [0.03, b_max, 500.0])
            
            try:
                popt3, pcov3 = curve_fit(
                    self.rational_func_od_3p, d_arr, ods, p0=p0, sigma=sigs, 
                    bounds=bounds, absolute_sigma=False, maxfev=15000
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
                print(f"  WARNING fitting channel {ch} ({e}), using analytic fallback.")
                b_fb = max_od * 2.5
                c_fb = b_fb * max_d / max_od
                popt = np.array([0.0, b_fb, c_fb, 1.0])
                self.calib_params[ch] = popt
                self.calib_cov[ch] = np.eye(4) * 0.01
                self.calib_samples[ch] = np.tile(popt, (n_posterior_samples, 1))
        
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
                             draws: int = 3000, chains: int = 4, use_cleaning: bool = True,
                             n_saved_samples: int = 100):
        """
        Kiterjesztett dózistartományú (0-50+ Gy) Bayes-i MCMC kalibráció:
        - Fixálja a netOD(0) = 0 fizikai peremfeltételt (a = 0).
        - Illeszti az 'e' görbületi kitevőt (a kék csatorna kis dózisú hűségéért).
        - LogNormal priorokat használ, így SOHA nem lép fel HMC quadpotential overflow!
        """
        if not BAYES_AVAILABLE:
            print("  ERROR: PyMC is not available.")
            return

        print(f"\n--- Running Extended-Range MCMC Calibration: {calibration_folder} (Cleaning={use_cleaning}) ---")
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
        # Biztosítjuk, hogy a netOD a D=0 pontban pontosan 0-hoz legyen igazítva
        for ch in ['R', 'G', 'B']:
            od0 = measured_ods[ch][0] if d_arr[0] == 0 else 0.0
            measured_ods[ch] = [max(0.0, float(val - od0)) for val in measured_ods[ch]]

        self.calib_doses = d_arr
        self.calib_ods = measured_ods

        print("  -> Sampling posterior (PyMC LogNormal Hill-model)...")
        self.calib_params = {}
        self.calib_cov = {}
        self.calib_samples = {}
        self.calib_sigma_od = {}
        plot_data = [] 
        
        def hill_fit(d, b, c, e):
            de = np.power(np.maximum(d, 0.0), e)
            return (b * de) / (de + c)

        rng = np.random.default_rng(42)

        for ch in ['R', 'G', 'B']:
            od_data = np.array(measured_ods[ch], dtype=np.float64)
            max_od = max(float(np.max(od_data)), 0.1)
            
            # Kezdeti becslés a (b, c, e) paraméterekre (a = 0 fix!)
            try:
                p0_guess = [max_od * 1.8, 12.0 if ch != 'B' else 35.0, 1.0 if ch != 'B' else 1.2]
                popt, _ = curve_fit(
                    hill_fit, d_arr, od_data, p0=p0_guess,
                    bounds=([max_od * 1.02, 0.5, 0.65], [15.0, 500.0, 1.6]),
                    maxfev=10000
                )
                mu_b, mu_c, mu_e = [float(x) for x in popt]
            except Exception:
                mu_b, mu_c, mu_e = max_od * 2.0, 15.0, 1.0

            with pm.Model() as model:
                # LogNormal priorok: természetes módon > 0, végtelenül differenciálható, NINCS OVERFLOW!
                b = pm.LogNormal('b', mu=np.log(mu_b), sigma=0.35)
                c = pm.LogNormal('c', mu=np.log(mu_c), sigma=0.45)
                e = pm.LogNormal('e', mu=np.log(mu_e), sigma=0.15)
                sigma = pm.HalfNormal('sigma', sigma=0.005)
                
                d_pos = np.maximum(d_arr, 1e-8)
                de = d_pos ** e
                # D=0-nál pontosan 0
                mu_od = pm.math.switch(d_arr <= 0, 0.0, (b * de) / (de + c))
                
                pm.Normal('obs', mu=mu_od, sigma=sigma, observed=od_data)
                
                init_vals = {'b': mu_b, 'c': mu_c, 'e': mu_e, 'sigma': 0.003}
                trace = pm.sample(
                    draws=draws, tune=1000, chains=chains, cores=1, 
                    init='adapt_diag', target_accept=0.95, 
                    initvals=init_vals, progressbar=False
                )
                self.calib_traces[ch] = trace
                
                post = trace.posterior
                b_s = post['b'].values.flatten()
                c_s = post['c'].values.flatten()
                e_s = post['e'].values.flatten()
                sig_s = post['sigma'].values.flatten()
                a_s = np.zeros_like(b_s) # a = 0 fixen!
                
                self.calib_params[ch] = np.array([
                    0.0, np.median(b_s), np.median(c_s), np.median(e_s)
                ])
                self.calib_sigma_od[ch] = float(np.median(sig_s))
                
                params_stack = np.vstack([a_s, b_s, c_s, e_s])
                self.calib_cov[ch] = np.cov(params_stack)
                
                idx_sub = rng.choice(len(b_s), size=min(n_saved_samples, len(b_s)), replace=False)
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
            "method": "MCMC_LogNormal_Hill",
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
        Kiszámolja a kalibrációs görbe bizonytalanságából eredő pixelenkénti szórást (sigma_calib)
        0-tól 50+ Gy-ig mindhárom csatorna (R, G, B) érzékenység-súlyozásával.
        """
        if all(ch in self.calib_samples for ch in ['R', 'G', 'B']):
            sR, sG, sB = self.calib_samples['R'], self.calib_samples['G'], self.calib_samples['B']
            n_samples = min(len(sR), len(sG), len(sB))
            
            # Érzékenység-alapú súlyok a teljes 0-50+ Gy tartományra
            w_r = np.clip((10.0 - dose_map) / 4.0, 0.0, 1.0)
            w_b = np.clip((dose_map - 18.0) / 10.0, 0.0, 0.7)
            w_g = np.maximum(0.0, 1.0 - w_r - w_b)
            
            od_r, od_g, od_b = od_map[:, :, 0], od_map[:, :, 1], od_map[:, :, 2]
            all_d_samples = np.zeros((n_samples, dose_map.shape[0], dose_map.shape[1]), dtype=np.float32)
            
            for i in range(n_samples):
                dr_i = self.inv_rational_func_dose(od_r, sR[i])
                dg_i = self.inv_rational_func_dose(od_g, sG[i])
                db_i = self.inv_rational_func_dose(od_b, sB[i])
                d_i = w_r * dr_i + w_g * dg_i + w_b * db_i
                all_d_samples[i] = np.clip(d_i, 0.0, 150.0)
                
            p16 = np.percentile(all_d_samples, 16, axis=0)
            p84 = np.percentile(all_d_samples, 84, axis=0)
            return np.nan_to_num((p84 - p16) / 2.0, nan=0.0)
        else:
            dod_r = self.rational_func_dod_dd(dose_map, *self.calib_params['R'])
            sig_r = self.calib_sigma_od.get('R', 0.003) / np.maximum(dod_r, 1e-4)
            return np.nan_to_num(sig_r, nan=0.0)

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
                od_map = self._clean_od_map_biharmonic(od_map)
                
            od_noise_map = self._estimate_od_noise_map(raw_od_map, od_map)
            
            # 1. Dózis és illesztési/mérési szórás (std_fit) kiszámítása
            if method == 'mayer':
                dose_map, std_fit_map = self._solver_mayer(od_map, od_noise_map=od_noise_map)
            elif method == 'admm':
                dose_map, std_fit_map = self._solver_admm(od_map, od_noise_map=od_noise_map)
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

    def _solver_admm(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, params_dict: dict = None, max_iter: int = 6):
        """
        Plug-and-Play ADMM (Alternating Direction Method of Multipliers) Solver.
        Globálisan, iteratívan választja szét a Dózist és a Vastagságot:
        - Dózis (D): L1 Total Variation (TV) regularizáció (megőrzi az éles LWFA profilokat és csúcsokat).
        - Vastagság (t): L2 Makroszkopikus simítás (fizikai folytonosság).
        Futásidő: ~0.5 másodperc (a korábbi Hybrid percekig tartó for-ciklusa helyett).
        """
        if params_dict is None: params_dict = self.calib_params
        if od_noise_map is None: od_noise_map = np.full_like(od_map, 0.003)

        pR, pG, pB = params_dict['R'], params_dict['G'], params_dict['B']

        # Warm start a fast solver alapjaival
        dR = self.inv_rational_func_dose(od_map[:, :, 0], pR)
        dG = self.inv_rational_func_dose(od_map[:, :, 1], pG)
        dB = self.inv_rational_func_dose(od_map[:, :, 2], pB)

        wR = np.clip((10.0 - dR) / 4.0, 0.0, 1.0)
        wG = 1.0 - wR
        max_calib_d = float(np.max(self.calib_doses)) if self.calib_doses is not None else 15.0
        wB = np.clip((dG - 30.0) / 20.0, 0.0, 0.25) if max_calib_d >= 30.0 else np.zeros_like(dG)
        wG = np.maximum(0.0, wG - wB)

        D = np.nan_to_num(wR * dR + wG * dG + wB * dB, nan=0.0)
        t = np.ones_like(D)

        # ADMM Dual & Segédváltozók
        Z_D = np.copy(D)
        U_D = np.zeros_like(D)
        rho = 20.0  # ADMM büntetőparaméter a TV csatoláshoz
        tv_weight = 0.05  # Erős, de éleket megtartó simítás

        signal_mask = od_map[:, :, 0] > 0.015

        for _ in range(max_iter):
            # 1. Gradiens Lépés (Adathűség - Data Fidelity)
            od_pred = np.stack([
                self.rational_func_od(D, *pR) * t,
                self.rational_func_od(D, *pG) * t,
                self.rational_func_od(D, *pB) * t
            ], axis=-1)

            dod_dD = np.stack([
                self.rational_func_dod_dd(D, *pR) * t,
                self.rational_func_dod_dd(D, *pG) * t,
                self.rational_func_dod_dd(D, *pB) * t
            ], axis=-1)
            
            dod_dt = np.stack([
                self.rational_func_od(D, *pR),
                self.rational_func_od(D, *pG),
                self.rational_func_od(D, *pB)
            ], axis=-1)

            ch_reliability = np.stack([
                np.where(D < 8.0, 1.0, np.exp(-(D - 8.0) / 3.0)),
                np.where(D > 1.5, 1.0, 0.5),
                np.where(D > 2.0, 0.8, 0.3)
            ], axis=-1)
            W = ch_reliability / np.maximum(od_noise_map ** 2, 1e-6)

            res = od_map - od_pred
            H_DD = np.sum(W * (dod_dD ** 2), axis=-1)
            H_tt = np.sum(W * (dod_dt ** 2), axis=-1)
            
            g_D = np.sum(W * res * dod_dD, axis=-1)
            g_t = np.sum(W * res * dod_dt, axis=-1)

            # 2. Dózis (D) frissítése ADMM momentummal
            D_unreg = D + g_D / np.maximum(H_DD, 1e-6)
            D = (H_DD * D_unreg + rho * (Z_D - U_D)) / (H_DD + rho)
            D = np.where(signal_mask, np.maximum(D, 0.0), 0.0)

            # 3. Vastagság (t) frissítése Proximal L2 operátorral
            t_unreg = t + g_t / np.maximum(H_tt, 1e-6)
            t_smooth = uniform_filter(t_unreg, size=15)
            beam_core = np.where(D > (0.5 * np.max(D)), D ** 2, 0.0)
            w_sum = np.sum(beam_core)
            mean_t_core = np.sum(t_smooth * beam_core) / w_sum if w_sum > 0 else 1.0
            t = np.where(signal_mask, np.clip(t_smooth - mean_t_core + 1.0, 0.95, 1.05), 1.0)

            # 4. Z-update (Proximal Total Variation operátor a nyaláb profiljára)
            Z_D = denoise_tv_chambolle(D + U_D, weight=tv_weight)

            # 5. U-update (Dual Multiplier)
            U_D = U_D + D - Z_D

        # Hibaszámítás a konvergált állapotban
        det = np.maximum(H_DD * H_tt, 1e-6)
        std_noise = np.sqrt(np.maximum(H_tt / det, 0.0))
        
        od_corr = od_map / t[:, :, None]
        dR_c = self.inv_rational_func_dose(od_corr[:, :, 0], pR)
        dG_c = self.inv_rational_func_dose(od_corr[:, :, 1], pG)
        dB_c = self.inv_rational_func_dose(od_corr[:, :, 2], pB)
        ch_disc = wR * wG * np.abs(dR_c - dG_c) + wG * wB * np.abs(dG_c - dB_c)
        
        std_fit_map = np.sqrt(std_noise ** 2 + ch_disc ** 2)
        std_fit_map[Z_D <= 1e-4] = 0.0

        return Z_D, np.nan_to_num(std_fit_map, nan=0.0)

    def _solver_mayer(self, od_map: np.ndarray, od_noise_map: np.ndarray = None, params_dict: dict = None):
        """
        Javított, torzításmentes (unbiased) Bayes-i Mayer-Micke 3-csatornás solver.
        - Megszünteti a ~10%-os szisztematikus alulbecslést a fast/hybrid solverekhez képest.
        - A B csatornát elsődlegesen a rétegvastagság (delta_t) szeparálására használja,
          és csak a G csatorna valódi telítődése (>30 Gy) felett engedi közvetlenül a dózisba.
        - Dózis-súlyozott nullapont-horgonyzást alkalmaz a vastagságtérképre a nyalábmagban.
        """
        if params_dict is None:
            params_dict = self.calib_params
        if od_noise_map is None:
            od_noise_map = np.full_like(od_map, 0.003)

        pR, pG, pB = params_dict['R'], params_dict['G'], params_dict['B']

        # 1. Csatornánkénti direkt inverzió
        dR = self.inv_rational_func_dose(od_map[:, :, 0], pR)
        dG = self.inv_rational_func_dose(od_map[:, :, 1], pG)
        dB = self.inv_rational_func_dose(od_map[:, :, 2], pB)

        # Dinamikus csatornasúlyok az alapdózishoz (konzisztens a fast és hybrid solverekkel!):
        # - 0..8 Gy: R dominál, 6..10 Gy között sima átmenet G-be
        wR = np.clip((10.0 - dR) / 4.0, 0.0, 1.0)
        wG = 1.0 - wR
        
        # A B csatornát CSAK extrém magas dózisnál (>30 Gy, ahol már a G is telítődik)
        # és csak kis súllyal keverjük a bázisba, hogy ne húzza le a 10-30 Gy-es tartományt!
        max_calib_d = float(np.max(self.calib_doses)) if self.calib_doses is not None else 15.0
        if max_calib_d >= 30.0:
            wB = np.clip((dG - 30.0) / 20.0, 0.0, 0.25)
            wG = np.maximum(0.0, wG - wB)
        else:
            wB = np.zeros_like(dG)

        # Torzításmentes bázis dózistérkép (pontosan egyezik a fast solver skálájával <30 Gy-ig)
        D_base = np.nan_to_num(wR * dR + wG * dG + wB * dB, nan=0.0)

        # 2. Fizikai modell predikció és Jacobi-deriváltak a D_base munkapontban
        od_pred = np.stack([
            self.rational_func_od(D_base, *pR),
            self.rational_func_od(D_base, *pG),
            self.rational_func_od(D_base, *pB)
        ], axis=-1)
        
        dod_dD = np.stack([
            self.rational_func_dod_dd(D_base, *pR),
            self.rational_func_dod_dd(D_base, *pG),
            self.rational_func_dod_dd(D_base, *pB)
        ], axis=-1)

        # 3. Egzakt Bayes-i súlyozás az OD-térben (1 / sigma_OD^2)
        # A B csatorna itt kap kulcsszerepet: mivel alacsony az érzékenysége a dózisra (kicsi dod_dD),
        # de érzékeny a rétegvastagságra (od_pred), kiválóan megköti a delta_t-t!
        ch_reliability = np.stack([
            np.where(D_base < 8.0, 1.0, np.exp(-(D_base - 8.0) / 3.0)), # R telítődik >10 Gy felett
            np.where(D_base > 1.5, 1.0, 0.5),                          # G a fő munkacsatorna
            np.where(D_base > 2.0, 0.8, 0.3)                           # B stabilizálja a vastagságot
        ], axis=-1)
        
        W_ch = ch_reliability / np.maximum(od_noise_map ** 2, 1e-6)
        res = od_map - od_pred

        # 2x2-es Gauss-Newton Hesse-mátrix (H) és gradiens (g) a (delta_D, delta_t) rendszerre:
        # Modell: OD_meas ≈ OD_pred(D_base) + dod_dD * delta_D + OD_pred(D_base) * delta_t
        H_DD = np.sum(W_ch * (dod_dD ** 2), axis=-1)
        H_Dt = np.sum(W_ch * dod_dD * od_pred, axis=-1)
        
        # JAVÍTVA: A W_ch már tartalmazza az 1/sigma_OD^2 szorzót, így a vastagság priorja
        # egyszerűen 1 / sigma_t^2 (ahol sigma_t = 0.015, azaz 1.5% gyári EBT3 vastagságtűrés).
        sigma_t_prior = 0.015
        H_tt = np.sum(W_ch * (od_pred ** 2), axis=-1) + (1.0 / (sigma_t_prior ** 2))

        g_D = np.sum(W_ch * res * dod_dD, axis=-1)
        g_t = np.sum(W_ch * res * od_pred, axis=-1)

        # Cramer-szabály a csatolt (delta_D, delta_t) rendszerre
        det = np.maximum(H_DD * H_tt - H_Dt ** 2, 1e-6)
        delta_t_raw = (H_DD * g_t - H_Dt * g_D) / det
        
        # Csak ott becsülünk vastagságot, ahol van érdemi jel (OD > 0.015)
        signal_mask = od_map[:, :, 0] > 0.015
        delta_t_raw = np.where(signal_mask, np.clip(delta_t_raw, -0.035, 0.035), 0.0)

        # 4. MULTI-SCALE VASTAGSÁG-KORREKCIÓ (Finom homogenitás + Csúcsdózis-védelem)
        # a) Enyhe (3x3 px) szűrés a nyers delta_t-n: megtartja a valódi (0.2-1.0 mm-es) 
        #    filmvastagság-ingadozásokat, így a térbeli STD ~1.1%-ról ~0.5-0.7%-ra csökken!
        delta_t_local = uniform_filter(delta_t_raw, size=3)
        
        # b) Makroszkopikus (35x35 px ~ 3 mm) trend: ezzel biztosítjuk, hogy a korrekció
        #    ne tudja "ellapítani" egy szűk LWFA nyaláb valódi dóziscsúcsát!
        delta_t_macro = uniform_filter(delta_t_raw, size=35)
        
        # A nyalábmag súlyozott átlagát levonjuk a makroszkopikus trendből (nullapont-horgonyzás)
        beam_core_weights = np.where(D_base > (0.5 * np.max(D_base)), D_base ** 2, 0.0)
        weight_sum = np.sum(beam_core_weights)
        if weight_sum > 0:
            mean_dt_core = np.sum(delta_t_macro * beam_core_weights) / weight_sum
        else:
            mean_dt_core = 0.0
            
        # Kombináció: a nagyfrekvenciás film-inhomogenitást (local - macro) 100%-ban korrigáljuk,
        # a makroszkopikus vastagságtrendet pedig óvatosan (50%-os súllyal) a horgonyzott nullapont körül:
        delta_t_smooth = (delta_t_local - delta_t_macro) + 0.5 * (delta_t_macro - mean_dt_core)
        delta_t_smooth = np.where(signal_mask, np.clip(delta_t_smooth, -0.03, 0.03), 0.0)

        # 5. Korrigált dózis kiszámítása:
        # Miután a simított delta_t_smooth ismert, az optimális dózis-frissítés analitikusan:
        # delta_D = (g_D - H_Dt * delta_t_smooth) / H_DD, vagy ekvivalensen a korrigált OD-ból:
        od_corr = od_map / (1.0 + delta_t_smooth[:, :, None])
        dR_c = self.inv_rational_func_dose(od_corr[:, :, 0], pR)
        dG_c = self.inv_rational_func_dose(od_corr[:, :, 1], pG)
        dB_c = self.inv_rational_func_dose(od_corr[:, :, 2], pB)
        
        dose_map = np.nan_to_num(wR * dR_c + wG * dG_c + wB * dB_c, nan=0.0)

        # 6. Egzakt utólagos (posterior) pixelenkénti bizonytalanság (Cramér-Rao határ + csatorna-inkonzisztencia)
        # A 2x2-es inverz Hesse-mátrix (kovariancia) [0,0] eleme adja a vastagság-bizonytalansággal
        # növelt dózis-varianciát: Var(D) = H_tt / det
        var_D_cramer = H_tt / det
        std_noise = np.sqrt(np.maximum(var_D_cramer, 0.0))
        
        # Csatornák közötti maradvány-eltérés a vastagság-korrekció után
        ch_disc = wR * wG * np.abs(dR_c - dG_c) + wG * wB * np.abs(dG_c - dB_c)
        std_fit_map = np.sqrt(std_noise ** 2 + ch_disc ** 2)
        std_fit_map[dose_map <= 1e-4] = 0.0

        return dose_map, np.nan_to_num(std_fit_map, nan=0.0)

    def _clean_od_map_biharmonic(self, od_map, fdr_threshold=0.99, safety_limit=0.10, max_defect_size=500):
        """Artifact removal using GMM anomaly detection and fast Biharmonic Inpainting."""
        ref_channel = od_map[:, :, 0] 
        gx = sobel(ref_channel, axis=0)
        gy = sobel(ref_channel, axis=1)
        grad_mag = np.hypot(gx, gy)
        
        data = grad_mag.flatten().reshape(-1, 1)
        sample_data = np.random.choice(data.flatten(), min(len(data), 100000)).reshape(-1, 1)
            
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(sample_data)
            noise_idx = np.argmax(gmm.means_.flatten())
            prob_noise = gmm.predict_proba(data)[:, noise_idx].reshape(ref_channel.shape)
            
            mask = prob_noise > fdr_threshold
            labeled_mask, num_features = label(mask)
            component_sizes = np.bincount(labeled_mask.ravel())
            too_big_labels = np.where(component_sizes > max_defect_size)[0]
            too_big_labels = too_big_labels[too_big_labels > 0]
            
            if len(too_big_labels) > 0:
                mask[np.isin(labeled_mask, too_big_labels)] = False
                
            ratio = np.mean(mask)
            if ratio > safety_limit:
                print(f"  [Cleaning] Warning: Too many artifacts ({ratio*100:.1f}%). Skipping.")
                return od_map

            if np.sum(mask) == 0: return od_map 
                
            cleaned_od = np.zeros_like(od_map)
            # Gyors, áramlástani folytonosságot biztosító biharmonikus kitöltés csatornánként
            for i in range(3):
                cleaned_od[:, :, i] = inpaint_biharmonic(od_map[:, :, i], mask, channel_axis=None)
            return cleaned_od

        except Exception as e:
            print(f"  [Cleaning] Error: {e}. Returning original.")
            return od_map

    # =========================================================================
    # 4. PLOTTING HELPERS
    # =========================================================================

    def _plot_mcmc_results(self, plot_data):
        """Visualizes MCMC traces (supports both 3p and Hill e-exponent models)."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        d_plot = np.linspace(0, max(plot_data[0]['doses']) * 1.1, 100)
        
        for i, data in enumerate(plot_data):
            ax = axes[i]
            ch = data['ch']
            trace = data['trace']
            d_meas = data['doses']
            od_meas = data['ods']
            
            post = trace.posterior
            b_s = post['b'].values.flatten()
            c_s = post['c'].values.flatten()
            a_s = post['a'].values.flatten() if 'a' in post else np.zeros_like(b_s)
            e_s = post['e'].values.flatten() if 'e' in post else np.ones_like(b_s)
            
            idxs = np.random.choice(len(b_s), size=min(500, len(b_s)), replace=False)
            A = a_s[idxs][:, None]; B = b_s[idxs][:, None]
            C = c_s[idxs][:, None]; E = e_s[idxs][:, None]
            D = np.maximum(d_plot[None, :], 0.0)
            
            DE = np.power(D, E)
            curves = A + (B * DE) / (DE + C)
            
            median_curve = np.median(curves, axis=0)
            hdi_low = np.percentile(curves, 2.5, axis=0)
            hdi_high = np.percentile(curves, 97.5, axis=0)
            
            ax.fill_between(d_plot, hdi_low, hdi_high, color='blue', alpha=0.2, label='95% HDI')
            ax.plot(d_plot, median_curve, 'b-', lw=2, label='MCMC Median')
            ax.plot(d_meas, od_meas, 'ro', label='Measured')
            ax.set_title(f"Channel {ch}")
            ax.set_xlabel("Dose [Gy]")
            ax.set_ylabel("netOD")
            ax.grid(True, alpha=0.3)
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