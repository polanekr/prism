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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import sobel, label
from sklearn.mixture import GaussianMixture
from skimage.restoration import inpaint_biharmonic
from bayes_support import BAYES_AVAILABLE, BAYES_IMPORT_ERROR, pm, az
from calibration import CalibrationMixin
from dose_solvers import DoseSolverMixin
from film_processing import ProcessingMixin, NumpyEncoder
from morphology import BeamMorphology


class GafchromicEngine(CalibrationMixin, DoseSolverMixin, ProcessingMixin):
    """
    Core engine for Gafchromic EBT3 film dosimetry with full pixel-wise uncertainty propagation.
    """

    def __init__(self, output_folder: str = "results", batch_metadata: dict = None, random_seed: int = 42):
        if type(random_seed) is not int or not 0 <= random_seed < 2**32 - 3:
            raise ValueError("random_seed must be an integer in [0, 2**32 - 3).")
        self.random_seed = random_seed
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
    # 1. CALIBRATION (WITH COVARIANCE & POSTERIOR SAMPLES EXPORT)
    # =========================================================================


    def _prepare_ratio_lut(self):
        if not self.calib_params: return
        d = np.linspace(0, 60, 6000)
        r = self.rational_func_od(d, *self.calib_params['R'])
        b = self.rational_func_od(d, *self.calib_params['B'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = r / b
        ratio = np.nan_to_num(ratio, nan=0.0)
        self.ratio_lut_func = interp1d(ratio, d, bounds_error=False, fill_value="extrapolate")


    def _get_roi_stats(self, img, center=None, size=50):
        h, w, c = img.shape
        if center is None: cx, cy = w // 2, h // 2
        else: cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - size); x2 = min(w, cx + size)
        y1 = max(0, cy - size); y2 = min(h, cy + size)
        roi = img[y1:y2, x1:x2, :]
        return np.mean(roi, axis=(0, 1)), np.std(roi, axis=(0, 1))


    def _clean_od_map_biharmonic(self, od_map, fdr_threshold=0.99, safety_limit=0.10, max_defect_size=500):
        """Artifact removal using GMM anomaly detection and fast Biharmonic Inpainting."""
        ref_channel = od_map[:, :, 0] 
        gx = sobel(ref_channel, axis=0)
        gy = sobel(ref_channel, axis=1)
        grad_mag = np.hypot(gx, gy)
        
        data = grad_mag.flatten().reshape(-1, 1)
        sample_data = np.random.default_rng(self.random_seed).choice(data.flatten(), min(len(data), 100000)).reshape(-1, 1)
            
        try:
            gmm = GaussianMixture(n_components=2, random_state=self.random_seed)
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
            
            idxs = np.random.default_rng(self.random_seed).choice(len(b_s), size=min(500, len(b_s)), replace=False)
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
