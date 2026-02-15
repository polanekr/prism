"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: analytics.py

This module provides tools for physical beam analysis and validation:
1. PDD Metrics (R50, Rp, E0).
2. Profile Analysis (FWHM, Penumbra, Flatness).
3. Gamma Index Analysis (1D/2D).

Author: [Your Name / PRISM Team]
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter

class DoseAnalyst:
    """
    Comprehensive class for dosimetric analysis and validation metrics.
    """

    @staticmethod
    def analyze_pdd(z_mm: np.ndarray, dose_val: np.ndarray, normalize: bool = True):
        """
        Extracts PDD metrics (R50, Rp, E0) from depth-dose data.
        
        Args:
            z_mm: Depth coordinates (mm).
            dose_val: Dose values.
            normalize: If True, normalizes max dose to 100%.
            
        Returns:
            dict: Metrics (R50_mm, Rp_mm, E0_MeV, d_max_mm).
            tuple: (z_dense, d_dense) smoothed curve data.
        """
        z = np.array(z_mm)
        d = np.array(dose_val)
        
        # Smoothing (Savitzky-Golay)
        try:
            window = min(len(d) // 5 * 2 + 1, 15)
            if window < 5: window = 3
            d_smooth = savgol_filter(d, window_length=window, polyorder=3)
        except: 
            d_smooth = d

        # Normalization
        max_val = np.max(d_smooth)
        if normalize:
            d_smooth = (d_smooth / max_val) * 100.0
            d_norm = (d / np.max(d)) * 100.0
        else:
            d_norm = d

        # Interpolation for high-res metrics
        f_interp = interp1d(z, d_smooth, kind='cubic', bounds_error=False, fill_value="extrapolate")
        z_dense = np.linspace(z[0], z[-1], 2000)
        d_dense = f_interp(z_dense)

        # Find d_max
        idx_max = np.argmax(d_dense)
        dmax_z = z_dense[idx_max]

        # Analyze Falloff region
        z_falloff = z_dense[idx_max:]
        d_falloff = d_dense[idx_max:]
        
        def get_depth_at_dose(target):
            if len(z_falloff) < 5: return np.nan
            # Find root of (D - target)
            spline = UnivariateSpline(z_falloff, d_falloff - target, s=0)
            roots = spline.roots()
            if len(roots) > 0: return roots[0]
            return np.nan

        r90 = get_depth_at_dose(90.0)
        r80 = get_depth_at_dose(80.0)
        r50 = get_depth_at_dose(50.0)

        # Rp construction (Tangent at inflection point)
        grad = np.gradient(d_falloff, z_falloff)
        idx_infl = np.argmin(grad)
        slope = grad[idx_infl]
        z_infl = z_falloff[idx_infl]
        d_infl = d_falloff[idx_infl]
        
        # Background level (estimated from tail)
        bg_level = np.mean(d_norm[-int(len(d_norm)*0.1):]) if len(d_norm) > 10 else 0.0
        
        if slope != 0:
            rp = z_infl + (bg_level - d_infl) / slope
        else:
            rp = np.nan

        # Energy estimation (Electron approx formula: E = 2.33 * R50)
        # Note: Valid for broad beams, approximate for VHEE/FLASH
        e0 = 2.33 * (r50 / 10.0) if not np.isnan(r50) else np.nan

        results = {
            'd_max_mm': dmax_z,
            'R90_mm': r90, 
            'R50_mm': r50, 
            'Rp_mm': rp,
            'E0_MeV': e0,
            'Surface_Dose_%': d_dense[0],
            'Background_%': bg_level
        }
        
        return results, (z_dense, d_dense)

    @staticmethod
    def analyze_profile(x_mm: np.ndarray, dose_val: np.ndarray, normalize: bool = True):
        """
        Calculates beam profile metrics (FWHM, Penumbra, Flatness).
        """
        x = np.array(x_mm)
        d = np.array(dose_val)
        
        try: 
            d_smooth = savgol_filter(d, 11, 3)
        except: 
            d_smooth = d
            
        if normalize: 
            d_smooth = (d_smooth / np.max(d_smooth)) * 100.0
            
        # Helper for finding widths
        def get_width_at_level(level):
            spline = UnivariateSpline(x, d_smooth - level, s=0)
            roots = np.sort(spline.roots())
            return roots

        r50 = get_width_at_level(50)
        r80 = get_width_at_level(80)
        r20 = get_width_at_level(20)
        
        res = {
            'FWHM_mm': np.nan, 
            'Center_mm': np.nan, 
            'Penumbra_L_mm': np.nan, 
            'Penumbra_R_mm': np.nan, 
            'Flatness_%': np.nan
        }
        
        if len(r50) >= 2:
            # FWHM (Full Width at Half Maximum)
            res['FWHM_mm'] = r50[-1] - r50[0]
            res['Center_mm'] = (r50[-1] + r50[0]) / 2.0
            
            # Penumbra (80-20%)
            if len(r20) > 0 and len(r80) > 0:
                # Left side (Rising)
                l20 = [r for r in r20 if r < res['Center_mm']]
                l80 = [r for r in r80 if r < res['Center_mm']]
                if l20 and l80: 
                    res['Penumbra_L_mm'] = abs(l80[-1] - l20[0])
                
                # Right side (Falling)
                r80_r = [r for r in r80 if r > res['Center_mm']]
                r20_r = [r for r in r20 if r > res['Center_mm']]
                if r80_r and r20_r: 
                    res['Penumbra_R_mm'] = abs(r20_r[0] - r80_r[-1])

            # Flatness (within 80% of FWHM)
            w = res['FWHM_mm'] * 0.8
            c = res['Center_mm']
            mask = (x >= c - w/2) & (x <= c + w/2)
            if np.any(mask):
                d_roi = d_smooth[mask]
                res['Flatness_%'] = 100 * (np.max(d_roi) - np.min(d_roi)) / (np.max(d_roi) + np.min(d_roi))
                
        return res, (x, d_smooth)

    @staticmethod
    def calculate_gamma_index(ref_z, ref_d, eval_z, eval_d, dose_tol=3.0, dist_tol=3.0, local_norm=False):
        """
        1D Gamma Analysis comparing two distributions (e.g. meas vs sim).
        
        Args:
            dose_tol: Dose tolerance in % (e.g., 3%).
            dist_tol: Distance tolerance in mm (e.g., 3mm).
            local_norm: If True, % is relative to local dose. If False, relative to global max.
        """
        # Interpolate eval to a denser grid for searching
        z_dense = np.linspace(min(ref_z), max(ref_z), 500)
        
        # Normalize both to 100% of Ref max
        max_ref = np.max(ref_d)
        d_r_norm = (ref_d / max_ref) * 100.0
        d_e_norm = (eval_d / max_ref) * 100.0
        
        # Interpolators
        f_eval = interp1d(eval_z, d_e_norm, kind='linear', bounds_error=False, fill_value=0.0)
        d_e_dense = f_eval(z_dense)
        
        gamma_values = []
        
        for i, z_r in enumerate(ref_z):
            d_r = d_r_norm[i]
            
            # Threshold: Ignore low dose regions (<10%)
            if d_r < 10.0:
                gamma_values.append(0.0)
                continue
                
            # Search window in Eval
            window_mask = np.abs(z_dense - z_r) < (dist_tol * 3)
            z_win = z_dense[window_mask]
            d_win = d_e_dense[window_mask]
            
            if len(z_win) == 0:
                gamma_values.append(10.0) # Fail
                continue
            
            # Dose difference denominator
            if local_norm:
                denom_dd = (d_r * dose_tol / 100.0) ** 2
            else:
                denom_dd = (100.0 * dose_tol / 100.0) ** 2 # Relative to global max (100)
                
            denom_dta = dist_tol ** 2
            
            # Gamma equation min search
            terms = (d_win - d_r)**2 / denom_dd + (z_win - z_r)**2 / denom_dta
            gamma = np.sqrt(np.min(terms))
            gamma_values.append(gamma)
            
        gamma_values = np.array(gamma_values)
        pass_rate = np.sum(gamma_values <= 1.0) / len(gamma_values) * 100.0
        
        return gamma_values, pass_rate