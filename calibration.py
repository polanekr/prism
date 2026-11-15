"""Calibration persistence, fitting and plots; independent of Qt widgets."""
import json
import os
import tempfile
import warnings
import time
from copy import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scan_data import is_raw_film, load_scan_noise
from bayes_support import pm, az, BAYES_AVAILABLE, BAYES_IMPORT_ERROR

class CalibrationMixin:
    _CALIBRATION_FIELDS = (
        'calib_params', 'calib_cov', 'calib_samples', 'calib_sigma_od',
        'calib_traces', 'calib_doses', 'calib_ods', 'ratio_lut_func',
    )


    def _new_calibration_state(self):
        """Stage a calibration without changing the currently active one."""
        staged = copy(self)
        for name in self._CALIBRATION_FIELDS:
            setattr(staged, name, None if name in (
                'calib_doses', 'calib_ods', 'ratio_lut_func') else {})
        return staged


    def _adopt_calibration(self, staged):
        self.random_seed = staged.random_seed
        for name in self._CALIBRATION_FIELDS:
            setattr(self, name, getattr(staged, name))


    @staticmethod
    def _scan_std_path(path):
        path = Path(path)
        return str(path.with_name(path.stem + '_scan_std' + path.suffix))


    def _save_calibration_metadata(self, metadata):
        """Replace the master only after a complete JSON file has been written."""
        metadata = {**metadata, 'random_seed': self.random_seed}
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.output_folder, suffix='.tmp',
                                             delete=False) as f:
                temporary_path = f.name
                json.dump(metadata, f, indent=4, allow_nan=False)
            os.replace(temporary_path, Path(self.output_folder) / 'calibration_master.json')
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)


    @staticmethod
    def _calibration_inputs(folder, doses_gy):
        files = sorted(f for f in Path(folder).iterdir()
                       if f.is_file() and is_raw_film(f))
        doses = np.asarray(doses_gy, dtype=np.float64)
        if doses.ndim != 1 or len(files) != doses.size:
            raise ValueError('Provide exactly one dose per calibration film (in filename order).')
        if doses.size == 0 or not np.all(np.isfinite(doses)) or np.any(doses < 0):
            raise ValueError('Calibration doses must be finite and nonnegative.')
        if doses[0] != 0 or np.any(np.diff(doses) < 0):
            raise ValueError('The first film must be the zero-dose blank; doses must be nondecreasing.')
        if np.unique(doses).size < 4:
            raise ValueError('Calibration requires at least four distinct doses, including zero.')
        return files, doses


    def load_calibration(self, master_json_path: str):
        """Validate a complete replacement before changing any active calibration."""
        with open(master_json_path, 'r') as f:
            data = json.load(f)
        staged = self._new_calibration_state()
        seed = data.get('random_seed', 42)
        if type(seed) is not int or not 0 <= seed < 2**32 - 3:
            raise ValueError('Invalid calibration random seed.')
        staged.random_seed = seed
        for ch in ('R', 'G', 'B'):
            params = np.asarray(data['fitted_params'][ch], dtype=np.float64)
            if (params.shape not in ((3,), (4,)) or not np.all(np.isfinite(params))
                    or np.any(params[1:] <= 0)):
                raise ValueError(f'Invalid calibration parameters for channel {ch}.')
            staged.calib_params[ch] = params
            n_params = params.size
            for field, target in (('calib_cov', staged.calib_cov),
                                  ('calib_samples', staged.calib_samples)):
                if ch not in data.get(field, {}):
                    continue
                arr = np.asarray(data[field][ch], dtype=np.float64)
                if field == 'calib_cov':
                    valid = (arr.shape == (n_params, n_params)
                             and np.all(np.isfinite(arr)) and np.allclose(arr, arr.T))
                    if valid:
                        valid = np.min(np.linalg.eigvalsh(arr)) >= -1e-10 * max(1.0, np.max(np.abs(arr)))
                else:
                    valid = (arr.ndim == 2 and arr.shape[0] > 0
                             and arr.shape[1] == n_params and np.all(np.isfinite(arr))
                             and np.all(arr[:, 1:] > 0))
                if not valid:
                    raise ValueError(f'Invalid {field} for channel {ch}.')
                target[ch] = arr
            sigma = float(data.get('calib_sigma_od', {}).get(ch, 0.004 if ch == 'B' else 0.003))
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError(f'Invalid OD noise for channel {ch}.')
            staged.calib_sigma_od[ch] = sigma
        if 'doses_gy' in data:
            doses = np.asarray(data['doses_gy'], dtype=np.float64)
            if doses.ndim != 1 or doses.size == 0 or not np.all(np.isfinite(doses)) or np.any(doses < 0):
                raise ValueError('Invalid calibration doses.')
            staged.calib_doses = doses
        if 'measured_ods' in data:
            staged.calib_ods = {}
            for ch in ('R', 'G', 'B'):
                ods = np.asarray(data['measured_ods'][ch], dtype=np.float64)
                if (ods.ndim != 1 or ods.size == 0 or not np.all(np.isfinite(ods))
                        or (staged.calib_doses is not None and ods.size != staged.calib_doses.size)):
                    raise ValueError(f'Invalid measured ODs for channel {ch}.')
                staged.calib_ods[ch] = ods.tolist()
        staged._prepare_ratio_lut()
        self._adopt_calibration(staged)
        print(f"  -> Calibration loaded from: {master_json_path} "
              f"(Posterior samples available: {len(self.calib_samples) > 0})")


    def run_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50, n_posterior_samples: int = 100):
        """Fit and save a new calibration, preserving the active state on failure."""
        if roi_size < 1:
            raise ValueError('ROI size must be positive.')
        staged = self._new_calibration_state()
        staged._run_calibration(calibration_folder, doses_gy, roi_size, n_posterior_samples)
        self._adopt_calibration(staged)


    def _run_calibration(self, calibration_folder, doses_gy, roi_size, n_posterior_samples):
        """
        Performs standard calibration using Scipy's curve_fit AND saves parameter covariance.
        Uses _scan_std.tif for empirical noise weighting if available.
        """
        print(f"\n--- Running Standard Calibration: {calibration_folder} ---")
        files, d_arr = self._calibration_inputs(calibration_folder, doses_gy)
        if n_posterior_samples < 1:
            raise ValueError('At least one posterior sample is required.')
        
        # --- BLANK FILM ÉS ZAJ BETÖLTÉSE ---
        blank_img_raw, _ = self._load_tiff_with_meta(files[0])
        blank_img = median_filter(blank_img_raw, size=(5, 5, 1))
        blank_roi_stats, _ = self._get_roi_stats(blank_img, size=roi_size)
        
        blank_std_img, blank_noise_source = load_scan_noise(files[0], blank_img_raw.shape)
        scan_noise_sources = {}
        
        measured_ods = {'R': [], 'G': [], 'B': []}
        measured_sigs = {'R': [], 'G': [], 'B': []}

        print("  -> Reading film data and extracting ROI statistics...")
        for fpath in files:
            img_raw, _ = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=(3, 3, 1))
            od_map = self._calculate_od_map(img, blank_img)
            
            # Alapértelmezett térbeli szórás (ha nincs szkennerzaj fájl)
            means_od, spatial_stds_od = self._get_roi_stats(od_map, size=roi_size)
            
            img_std, noise_source = load_scan_noise(fpath, img_raw.shape)
            scan_noise_sources[fpath.name] = noise_source
            if blank_std_img is not None and img_std is not None:
                # Egzakt hibaterjedés a nyers I és blank I alapján
                term_img = (img_std / (np.log(10) * np.maximum(img_raw, 1.0))) ** 2
                term_blank = (blank_std_img / (np.log(10) * np.maximum(blank_img_raw, 1.0))) ** 2
                od_noise_map = np.sqrt(term_img + term_blank)
                
                # A zajtérkép átlaga a ROI-ban adja a mérési bizonytalanságot
                means_noise, _ = self._get_roi_stats(od_noise_map, size=roi_size)
                stds_to_use = means_noise
            else:
                stds_to_use = spatial_stds_od
            
            for i, ch in enumerate(['R', 'G', 'B']):
                measured_ods[ch].append(means_od[i])
                measured_sigs[ch].append(stds_to_use[i])

        print("  -> Fitting curves & extracting covariance...")
        self.calib_doses = d_arr
        self.calib_ods = measured_ods 
        
        rng = np.random.default_rng(self.random_seed)
        
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
                # curve_fit a valós mérési szórásokkal súlyoz (sigma=sigs)
                popt3, pcov3 = curve_fit(
                    self.rational_func_od_3p, d_arr, ods, p0=p0, sigma=sigs, 
                    bounds=bounds, absolute_sigma=False, maxfev=15000
                )
                if not np.all(np.isfinite(popt3)) or not np.all(np.isfinite(pcov3)):
                    raise ValueError('Fit parameters or covariance are not finite.')
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
                raise ValueError(f'Calibration fit failed for channel {ch}: {e}') from e
        
        self._prepare_ratio_lut()
        
        meta = {
            "fitted_params": {k: v.tolist() for k, v in self.calib_params.items()},
            "calib_cov": {k: v.tolist() for k, v in self.calib_cov.items()},
            "calib_samples": {k: v.tolist() for k, v in self.calib_samples.items()},
            "calib_sigma_od": self.calib_sigma_od,
            "measured_ods": measured_ods,
            "blank_rgb": blank_roi_stats.tolist(),
            "doses_gy": d_arr.tolist(),
            "method": "Standard_CurveFit",
            "noise_model": "scanner_mean_noise_or_spatial_fallback",
            "scan_noise_sources": scan_noise_sources,
            "blank_noise_source": blank_noise_source,
            "date": str(datetime.now())
        }
        out_path = os.path.join(self.output_folder, "calibration_master.json")
        self._save_calibration_metadata(meta)
        print(f"  -> Calibration & Uncertainty structures saved to {out_path}")


    def run_mcmc_calibration(self, calibration_folder: str, doses_gy: list, roi_size: int = 50, 
                             draws: int = 3000, chains: int = 4, use_cleaning: bool = True,
                             n_saved_samples: int = 100, adaptation: str = 'adapt_diag'):
        """Run MCMC in an isolated state and activate only a completed calibration."""
        if roi_size < 1:
            raise ValueError('ROI size must be positive.')
        if adaptation not in ('adapt_diag', 'adapt_full'):
            raise ValueError('Unknown MCMC adaptation.')
        staged = self._new_calibration_state()
        staged._run_mcmc_calibration(calibration_folder, doses_gy, roi_size,
                                     draws, chains, use_cleaning, n_saved_samples, adaptation)
        self._adopt_calibration(staged)


    def _run_mcmc_calibration(self, calibration_folder, doses_gy, roi_size,
                              draws, chains, use_cleaning, n_saved_samples, adaptation):
        """
        Kiterjesztett dózistartományú (0-50+ Gy) Bayes-i MCMC kalibráció.
        Heteroszkedasztikus hibamodellt használ: sigma_total^2 = sigma_measured^2 + sigma_model^2
        """
        if not BAYES_AVAILABLE:
            raise RuntimeError('MCMC support is unavailable. Install it with: uv sync --locked --extra bayes. '
                               f'Details: {BAYES_IMPORT_ERROR}')

        print(f"\n--- Running Extended-Range MCMC Calibration: {calibration_folder} (Cleaning={use_cleaning}) ---")
        files, d_arr = self._calibration_inputs(calibration_folder, doses_gy)
        if min(draws, chains, n_saved_samples) < 1:
            raise ValueError('MCMC draws, chains and saved samples must be positive.')

        print("  -> Loading images & empirical uncertainties...")
        blank_img_raw, _ = self._load_tiff_with_meta(files[0])
        blank_img = median_filter(blank_img_raw, size=(5, 5, 1))
        blank_roi_stats, _ = self._get_roi_stats(blank_img, size=roi_size)
        
        blank_std_img, blank_noise_source = load_scan_noise(files[0], blank_img_raw.shape)
        scan_noise_sources = {}
        
        measured_ods = {'R': [], 'G': [], 'B': []}
        measured_sigs = {'R': [], 'G': [], 'B': []}
        
        for fpath in files:
            img_raw, _ = self._load_tiff_with_meta(fpath)
            img = median_filter(img_raw, size=(3, 3, 1))
            od_map = self._calculate_od_map(img, blank_img)
            
            if use_cleaning:
                od_map = self._clean_od_map_biharmonic(od_map)
            
            means_od, spatial_stds_od = self._get_roi_stats(od_map, size=roi_size)
            
            img_std, noise_source = load_scan_noise(fpath, img_raw.shape)
            scan_noise_sources[fpath.name] = noise_source
            if blank_std_img is not None and img_std is not None:
                term_img = (img_std / (np.log(10) * np.maximum(img_raw, 1.0))) ** 2
                term_blank = (blank_std_img / (np.log(10) * np.maximum(blank_img_raw, 1.0))) ** 2
                od_noise_map = np.sqrt(term_img + term_blank)
                
                means_noise, _ = self._get_roi_stats(od_noise_map, size=roi_size)
                stds_to_use = means_noise
            else:
                stds_to_use = spatial_stds_od
                
            for i, ch in enumerate(['R', 'G', 'B']):
                measured_ods[ch].append(means_od[i])
                measured_sigs[ch].append(stds_to_use[i])

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
        sampling_times = {}
        
        def hill_fit(d, b, c, e):
            de = np.power(np.maximum(d, 0.0), e)
            return (b * de) / (de + c)

        rng = np.random.default_rng(self.random_seed)

        for ch in ['R', 'G', 'B']:
            od_data = np.array(measured_ods[ch], dtype=np.float64)
            sig_meas = np.maximum(np.array(measured_sigs[ch], dtype=np.float64), 0.001)
            max_od = max(float(np.max(od_data)), 0.1)
            
            try:
                p0_guess = [max_od * 1.8, 12.0 if ch != 'B' else 35.0, 1.0 if ch != 'B' else 1.2]
                popt, _ = curve_fit(
                    hill_fit, d_arr, od_data, p0=p0_guess, sigma=sig_meas,
                    bounds=([max_od * 1.02, 0.5, 0.65], [15.0, 500.0, 1.6]),
                    maxfev=10000
                )
                mu_b, mu_c, mu_e = [float(x) for x in popt]
            except Exception:
                mu_b, mu_c, mu_e = max_od * 2.0, 15.0, 1.0

            with pm.Model() as model:
                # LogNormal priorok a fizikai paraméterekre
                b = pm.LogNormal('b', mu=np.log(mu_b), sigma=0.35)
                c = pm.LogNormal('c', mu=np.log(mu_c), sigma=0.45)
                e = pm.LogNormal('e', mu=np.log(mu_e), sigma=0.15)
                
                # A film inhomogenitásból eredő strukturális hiba
                sigma_model = pm.HalfNormal('sigma_model', sigma=0.005)
                
                d_pos = np.maximum(d_arr, 1e-8)
                de = d_pos ** e
                mu_od = pm.math.switch(d_arr <= 0, 0.0, (b * de) / (de + c))
                
                # Teljes variancia = Mérési variancia (ismert) + Modell variancia (keresett)
                sigma_tot = pm.math.sqrt(sig_meas**2 + sigma_model**2)
                
                pm.Normal('obs', mu=mu_od, sigma=sigma_tot, observed=od_data)
                
                init_vals = {'b': mu_b, 'c': mu_c, 'e': mu_e, 'sigma_model': 0.003}
                print(f'  -> MCMC {ch}: {chains} chains × {draws} draws + 1000 tuning, {adaptation}.')
                sampling_start = time.perf_counter()
                trace = pm.sample(
                    draws=draws, tune=1000, chains=chains, cores=1, 
                    init=adaptation, target_accept=0.95, 
                    initvals=init_vals, progressbar=False,
                    random_seed=self.random_seed + 'RGB'.index(ch)
                )
                sampling_times[ch] = time.perf_counter()-sampling_start
                print(f'  -> MCMC {ch} completed in {sampling_times[ch]:.1f} s.')
                self.calib_traces[ch] = trace
                
                post = trace.posterior
                b_s = post['b'].values.flatten()
                c_s = post['c'].values.flatten()
                e_s = post['e'].values.flatten()
                sig_model_s = post['sigma_model'].values.flatten()
                a_s = np.zeros_like(b_s) # a = 0 fixen!
                
                self.calib_params[ch] = np.array([
                    0.0, np.median(b_s), np.median(c_s), np.median(e_s)
                ])
                # Eltároljuk a strukturális (modell) bizonytalanságot
                self.calib_sigma_od[ch] = float(np.median(sig_model_s))
                
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
            "doses_gy": d_arr.tolist(),
            "method": "MCMC_LogNormal_Hill",
            "mcmc_settings": {"draws": draws, "tune": 1000, "chains": chains, "cores": 1,
                              "adaptation": adaptation, "target_accept": 0.95,
                              "sampling_seconds_by_channel": sampling_times},
            "noise_model": "scanner_mean_noise_or_spatial_fallback_plus_model_sigma",
            "scan_noise_sources": scan_noise_sources,
            "blank_noise_source": blank_noise_source,
            "date": str(datetime.now())
        }
        if len(plot_data) > 0:
            self._plot_mcmc_results(plot_data)
        self._save_calibration_metadata(meta)
        print(f"  -> MCMC Calibration & {n_saved_samples} posterior realizations saved.")


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


