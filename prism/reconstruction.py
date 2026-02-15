"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: reconstruction.py

This module contains the core class `Bayesian3DVolumeReconstructor` responsible for:
1. Loading sparse 2D PDD (Percentage Depth Dose) measurements.
2. Reconstructing a full 3D volumetric dose distribution using Gaussian Process regression.
3. Aligning the 3D volume to physical films (Front/Back).
4. Correcting for spectral changes (beam hardening) using effective attenuation.

Author: [Your Name / PRISM Team]
License: MIT
"""

import os
import glob
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift, zoom, uniform_filter
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# Try importing the Bayesian Fitter (optional dependency)
try:
    from prism.validation import FullPDDBayesianFitter
except ImportError:
    print("WARNING: 'prism.validation' module not found! Bayesian fitting might be unavailable.")

warnings.filterwarnings("ignore")


class Bayesian3DVolumeReconstructor:
    """
    Reconstructs a 3D dose volume from sparse 2D film measurements using
    Bayesian inference (Gaussian Processes) and physical constraints.
    """

    def __init__(self, pdd_folder: str, ssd_mm: float = 300.0, energy_mev: float = 0.35, energy_std_mev: float = None):
        """
        Initialize the reconstructor by loading PDD data and metadata.

        Args:
            pdd_folder (str): Path to the folder containing _dose.npy and _meta.json files.
            ssd_mm (float): Source-to-Surface Distance in mm.
            energy_mev (float): Nominal beam energy in MeV (for initial guess).
            energy_std_mev (float, optional): Uncertainty of the beam energy.
        """
        self.ssd = ssd_mm
        self.energy = energy_mev
        self.energy_std = energy_std_mev
        
        # State variables
        self.volume = None      
        self.uncertainty = None 
        self.coords = None 
        self.roi_masks = {} 
        self.debug_tips = []
        
        # Status flags
        self.is_built = False
        self.is_aligned = False
        self.is_scaled = False
        self.is_spectrally_corrected = False
        
        # Physical parameters
        self.pix_mm = None
        
        # 1. Load Data
        self.pdd_data = self._load_pdd_files(pdd_folder)
        
        if self.pix_mm is None:
            raise ValueError("Error: Could not determine pixel spacing from JSON files!")
            
        print(f"  -> Detected pixel spacing: {self.pix_mm:.4f} mm")
        
        # 2. Initialize Fitter (will be run later)
        self.fitter = None
        
        # Internal backups for reset
        self._volume_backup = None
        self._uncertainty_backup = None

    def _load_pdd_files(self, folder: str) -> dict:
        """Loads .npy dose maps and extracting depth/pixel info from .json."""
        data = {}
        files = glob.glob(os.path.join(folder, "*_dose.npy"))
        print(f"--- Loading PDD Data ({len(files)} files) ---")
        
        for fpath in files:
            fname = os.path.basename(fpath)
            if fpath.endswith("_dose.npy"):
                json_path = fpath.replace("_dose.npy", "_meta.json")
            else:
                json_path = os.path.splitext(fpath)[0] + ".json"
            
            if not os.path.exists(json_path):
                continue
            
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            current_pix = meta.get('pixel_spacing') or meta.get('pixel_spacing_mm')
            if self.pix_mm is None and current_pix is not None:
                self.pix_mm = float(current_pix)

            depth = meta.get('depth_mm')
            if depth is not None:
                img = np.load(fpath)
                data[float(depth)] = img
                
        if not data:
            raise ValueError("No valid PDD files found in the specified folder!")
        return data

    def _create_and_run_fitter(self, beam_type: str = 'photon'):
        """
        Creates and runs the Bayesian PDD fitter to estimate the depth-dose trend.
        """
        print("\n--- Fitting Physical Model (Bayesian Fitter) ---")
        z_meas = []
        dose_meas = []
        
        # Extract central ROI average for fitting
        roi_px = int(2.0 / self.pix_mm)
        
        for z in sorted(self.pdd_data.keys()):
            img = self.pdd_data[z]
            h, w = img.shape
            cy, cx = h//2, w//2
            y1, y2 = max(0, cy-roi_px), min(h, cy+roi_px)
            x1, x2 = max(0, cx-roi_px), min(w, cx+roi_px)
            
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                dose_val = np.mean(roi)
                z_meas.append(z)
                dose_meas.append(dose_val)
        
        # Instantiate Fitter
        try:
            self.fitter = FullPDDBayesianFitter.create_from_physical_params(
                z_meas, 
                dose_meas, 
                E0_MeV=self.energy, 
                E0_std_MeV=self.energy_std,
                SSD_mm=self.ssd,
                mode_override=beam_type
            )
            self.fitter.build_model()
            self.fitter.run_fit()
        except NameError:
             print("  -> Fitter module not available, skipping advanced physics fit.")
             
        return self.fitter

    def build_volume(self, roi_size_mm: float = 40.0, z_res_mm: float = 0.2, 
                     beam_type: str = 'photon', divergence_correction: bool = False, 
                     shape_threshold_ratio: float = 0.15,
                     perform_alignment: bool = True, align_crop_mm: float = 15.0):
        """
        STEP 1: Reconstruct the full 3D Dose Volume from sparse measurements.
        
        Args:
            roi_size_mm (float): Size of the Region of Interest to reconstruct.
            z_res_mm (float): Z-axis resolution (slice thickness).
            beam_type (str): 'photon' or 'electron'.
            divergence_correction (bool): Correct for beam divergence (1/r^2).
            shape_threshold_ratio (float): Noise threshold relative to max dose.
            perform_alignment (bool): Align layers based on Center of Mass.
            align_crop_mm (float): Crop edges before alignment to avoid markers/text.
        """
        corr_status = "ON" if divergence_correction else "OFF"
        align_status = f"ON (Crop={align_crop_mm}mm)" if perform_alignment else "OFF"
        
        print(f"\n--- 1. Building 3D Volume (DivCorr: {corr_status}, Align: {align_status}) ---")
        
        self.volume = None
        if self.fitter is None and 'FullPDDBayesianFitter' in globals():
             self._create_and_run_fitter(beam_type=beam_type)

        # 1. PDD Trend Estimation
        max_depth = max(self.pdd_data.keys()) + 5.0
        z_axis = np.arange(0, max_depth, z_res_mm)
        
        if self.fitter:
            pdd_trend_mean, _ = self.fitter.predict(z_axis)
        else:
             # Fallback if fitter is missing: simple interpolation
             keys = sorted(self.pdd_data.keys())
             vals = [np.mean(self.pdd_data[k]) for k in keys]
             pdd_trend_mean = np.interp(z_axis, keys, vals)

        # Anchoring to measured data (at approx 16mm depth or similar)
        anchor_depth = 16.0
        keys = sorted(self.pdd_data.keys())
        meas_vals = [np.percentile(self.pdd_data[k], 99) for k in keys]
        dose_meas_anchor = np.interp(anchor_depth, keys, meas_vals)
        dose_fit_anchor = np.interp(anchor_depth, z_axis, pdd_trend_mean)
        
        # Avoid division by zero
        if dose_fit_anchor > 0:
            pdd_trend = pdd_trend_mean * (dose_meas_anchor / dose_fit_anchor)
        else:
            pdd_trend = pdd_trend_mean

        # 2. Learning Profile Shape (Gaussian Process)
        stack_list = []
        train_z = []
        
        roi_px = int(roi_size_mm / self.pix_mm)
        target_shape = (2*roi_px, 2*roi_px)
        
        global_max = np.max([np.max(self.pdd_data[k]) for k in keys])
        noise_threshold = global_max * shape_threshold_ratio 

        print(f"  -> Analyzing Profiles (Threshold: {noise_threshold:.3f} Gy)...")
        last_valid_z = 0.0
        
        for z in keys:
            img = self.pdd_data[z]
            local_max_robust = np.percentile(img, 99)
            
            if local_max_robust < noise_threshold: continue
            last_valid_z = z 

            # Divergence Correction (scaling)
            if divergence_correction:
                mag_factor = (self.ssd + z) / self.ssd
                try: img_normalized = zoom(img, 1.0/mag_factor, order=1)
                except RuntimeError: img_normalized = img 
            else:
                img_normalized = img 

            h, w = img_normalized.shape
            
            # --- ALIGNMENT LOGIC ---
            shift_y, shift_x = 0, 0
            
            if perform_alignment:
                img_for_calc = img_normalized.copy()
                
                # CROP EDGES (Remove markers)
                if align_crop_mm > 0:
                    cut_px = int(align_crop_mm / self.pix_mm)
                    if 2*cut_px < h and 2*cut_px < w:
                        img_for_calc[:cut_px, :] = 0  # Top
                        img_for_calc[-cut_px:, :] = 0 # Bottom
                        img_for_calc[:, :cut_px] = 0  # Left
                        img_for_calc[:, -cut_px:] = 0 # Right
                
                # Thresholding
                thresh_val = local_max_robust * 0.5 
                thresh_img = np.where(img_for_calc > thresh_val, img_for_calc, 0)
                
                if np.sum(thresh_img) == 0: 
                    cy, cx = h // 2, w // 2
                else:
                    cy, cx = center_of_mass(thresh_img)
                
                # Calculate shift
                target_y, target_x = h // 2, w // 2
                shift_y, shift_x = target_y - cy, target_x - cx
            
            # Apply Shift
            if shift_y != 0 or shift_x != 0:
                img_aligned = shift(img_normalized, shift=[shift_y, shift_x], order=1)
            else:
                img_aligned = img_normalized
            
            # Extract ROI
            target_y, target_x = h // 2, w // 2
            y_start = max(0, target_y-roi_px); y_end = min(h, target_y+roi_px)
            x_start = max(0, target_x-roi_px); x_end = min(w, target_x+roi_px)
            crop = img_aligned[y_start:y_end, x_start:x_end]
            
            # Padding if crop is smaller than target
            if crop.shape != target_shape:
                pad_crop = np.zeros(target_shape)
                dy, dx = crop.shape
                oy, ox = (target_shape[0] - dy) // 2, (target_shape[1] - dx) // 2
                pad_crop[oy:oy+dy, ox:ox+dx] = crop
                crop = pad_crop

            if np.max(crop) > 0:
                stack_list.append(crop / local_max_robust)
                train_z.append(z)

        # GP Fit
        if not stack_list: raise ValueError("Error: No films found with sufficient signal!")
        
        print(f"  -> Learning shape from {len(stack_list)} layers.")
        stack_array = np.array(stack_list)
        n_samples, ny, nx = stack_array.shape
        
        kernel = 1.0 * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-2)
        gp.fit(np.array(train_z).reshape(-1, 1), stack_array.reshape(n_samples, -1))
        
        # Prediction (Shape Freezing after valid range)
        z_valid_mask = z_axis <= (last_valid_z + 2.0)
        z_valid = z_axis[z_valid_mask]
        
        y_mean_valid, _ = gp.predict(z_valid.reshape(-1, 1), return_std=True)
        shape_valid = y_mean_valid.reshape(len(z_valid), ny, nx).clip(0, 1.1)
        
        last_good_shape = shape_valid[-1]
        full_shape = []
        for i, z in enumerate(z_axis):
            if i < len(shape_valid): full_shape.append(shape_valid[i])
            else: full_shape.append(last_good_shape)
        full_shape = np.array(full_shape)
        
        # 3. RECONSTRUCTION
        final_volume = []
        target_h, target_w = ny, nx
        
        for i, z in enumerate(z_axis):
            current_shape = full_shape[i]
            if divergence_correction:
                mag_factor = (self.ssd + z) / self.ssd
                slice_zoomed = zoom(current_shape, mag_factor, order=1)
            else:
                slice_zoomed = current_shape
            
            # Final cropping/padding to fixed size
            zh, zw = slice_zoomed.shape
            slice_fixed = np.zeros((target_h, target_w))
            
            cty, ctx = zh // 2, zw // 2
            y1 = max(0, cty - target_h//2); y2 = min(zh, cty + target_h//2)
            x1 = max(0, ctx - target_w//2); x2 = min(zw, ctx + target_w//2)
            h_fill, w_fill = y2 - y1, x2 - x1
            ty1, tx1 = target_h//2 - h_fill//2, target_w//2 - w_fill//2
            
            slice_fixed[ty1:ty1+h_fill, tx1:tx1+w_fill] = slice_zoomed[y1:y2, x1:x2]
            final_volume.append(slice_fixed)

        self.volume = np.array(final_volume) * pdd_trend.reshape(-1, 1, 1)
        
        # Cleanup noise
        cleanup_thresh = np.max(self.volume) * 0.005
        self.volume[self.volume < cleanup_thresh] = 0.0
        self.uncertainty = self.volume * 0.05 

        self.coords = {
            'z': z_axis,
            'y': np.arange(ny) * self.pix_mm - (ny * self.pix_mm / 2),
            'x': np.arange(nx) * self.pix_mm - (nx * self.pix_mm / 2)
        }
        
        self._volume_backup = self.volume.copy()
        self._uncertainty_backup = self.uncertainty.copy()
        self.is_built = True
        print("  -> 3D Volume Construction Complete (Backup created).")
        
    def create_backup_from_current_state(self):
        """Creates a manual backup point from the current volume state."""
        if self.volume is None:
            raise RuntimeError("Error: No volume to backup!")

        print("--- Creating Backup from Current State ---")
        self._volume_backup = self.volume.copy()
        
        if self.uncertainty is not None:
            self._uncertainty_backup = self.uncertainty.copy()
        else:
            self._uncertainty_backup = np.zeros_like(self.volume)

        self.is_built = True
        self.is_aligned = False
        self.is_scaled = False
        self.is_spectrally_corrected = False
        print("  -> Backup successful. reset_volume() is now available.")
        
    def reset_volume(self):
        """Resets the volume to the original state (after build_volume)."""
        if not self.is_built or not hasattr(self, '_volume_backup') or self._volume_backup is None:
            print("  ERROR: No backup found! (Run build_volume first)")
            return

        print("\n--- Resetting Volume ---")
        self.volume = self._volume_backup.copy()
        self.uncertainty = self._uncertainty_backup.copy()
        
        self.is_aligned = False
        self.is_scaled = False
        self.is_spectrally_corrected = False
        print("  -> Volume reset to original unscaled state.")

    def register_and_scale(self, front_film_map: np.ndarray, front_film_depth_mm: float, 
                           roi_size_mm: float = 10.0, crop_edges_mm: float = 8.0):
        """
        STEP 2: Align and Scale the Volume to the Front Film.
        
        Args:
            front_film_map: 2D dose map of the front film.
            front_film_depth_mm: Physical depth of the film.
            roi_size_mm: Size of the central ROI for scaling factor calculation.
            crop_edges_mm: Margin to crop from film edges to avoid marker artifacts during alignment.
        """
        if not self.is_built: raise RuntimeError("Run build_volume first!")
        print(f"\n--- 2. Registration & Scaling (Front Film: {front_film_depth_mm}mm) ---")
        
        z_idx = np.argmin(np.abs(self.coords['z'] - front_film_depth_mm))
        vol_slice = self.volume[z_idx, :, :]
        
        # --- A) SMART SHIFT (Center of Mass Alignment) ---
        print(f"  -> Finding Center of Mass (Edge Crop: {crop_edges_mm} mm)...")
        
        margin_px = int(crop_edges_mm / self.pix_mm)
        h, w = front_film_map.shape
        
        # Crop film to remove markers
        if margin_px < h//2 and margin_px < w//2:
            film_crop = front_film_map[margin_px:-margin_px, margin_px:-margin_px]
            com_crop = center_of_mass(film_crop)
            com_film = (com_crop[0] + margin_px, com_crop[1] + margin_px)
        else:
            print("  ERROR: Crop margin too large! Using full image.")
            com_film = center_of_mass(front_film_map)

        if np.sum(vol_slice) == 0:
            vol_slice = self.volume[np.argmax(np.sum(self.volume, axis=(1,2))), :, :]
        
        com_vol = center_of_mass(vol_slice)
        
        shift_y = com_film[0] - com_vol[0]
        shift_x = com_film[1] - com_vol[1]
        
        print(f"  -> Detected Shift: Y={shift_y*self.pix_mm:.1f}mm, X={shift_x*self.pix_mm:.1f}mm")
        
        if abs(shift_x) > 0.01 or abs(shift_y) > 0.01:
            for z in range(self.volume.shape[0]):
                self.volume[z] = shift(self.volume[z], shift=[shift_y, shift_x], order=1, cval=0.0)
            print("  -> Volume shifted to match film center.")
            self.is_aligned = True
        else:
            print("  -> No shift needed.")

        # --- B) SCALING (Central ROI) ---
        vol_slice_shifted = self.volume[z_idx, :, :]
        cy, cx = h // 2, w // 2 
        
        r_px = int((roi_size_mm / 2.0) / self.pix_mm)
        if r_px < 1: r_px = 1
        
        roi_film = front_film_map[cy-r_px : cy+r_px, cx-r_px : cx+r_px]
        roi_vol  = vol_slice_shifted[cy-r_px : cy+r_px, cx-r_px : cx+r_px]
        
        val_film = np.mean(roi_film)
        val_vol = np.mean(roi_vol)
        
        if val_vol == 0:
            val_vol = 1.0
            print("  ERROR: Model ROI is empty! (Alignment failed?)")

        scale_factor = val_film / val_vol
        
        self.volume *= scale_factor
        self.uncertainty *= scale_factor
        self.is_scaled = True
        
        print(f"  -> Scaling Method: Central ROI ({roi_size_mm}x{roi_size_mm} mm)")
        print(f"  -> Scale Factor: {scale_factor:.4f}")
        print(f"  -> Ref Dose (Front ROI): {val_film:.2f} Gy")

    def correct_spectrum_with_cavity(self, rear_film_map: np.ndarray, rear_film_depth_mm: float, 
                                     mask_name: str, gap_thickness_mm: float, gap_material: str = 'air'):
        """
        STEP 3: Spectral Correction using Rear Film (Effective Attenuation).
        Calculates the slope correction factor by comparing the model prediction
        with the measured dose behind a cavity (gap).
        """
        if not self.is_scaled: 
            print("WARNING: Model not scaled yet! Correction might be inaccurate.")
            
        if mask_name not in self.roi_masks: 
            raise ValueError(f"Mask not found: {mask_name}")
        
        print(f"\n--- 3. Spectral Correction (Rear Film: {rear_film_depth_mm}mm) ---")
        
        z_idx = np.argmin(np.abs(self.coords['z'] - rear_film_depth_mm))
        vol_slice = self.volume[z_idx, :, :]
        
        mask_3d = self.roi_masks[mask_name]
        mask_proj = np.max(mask_3d, axis=0) > 0
        
        # --- SIZE ALIGNMENT (Auto-Crop / Pad) ---
        h_model, w_model = vol_slice.shape
        h_film, w_film = rear_film_map.shape
        
        if (h_model != h_film) or (w_model != w_film):
            print(f"  WARNING: Size mismatch! Model: {h_model}x{w_model}, Film: {h_film}x{w_film}")
            print("  -> Auto-aligning (Center Crop/Pad)...")
            
            film_aligned = np.zeros((h_model, w_model))
            cy_m, cx_m = h_model // 2, w_model // 2
            cy_f, cx_f = h_film // 2, w_film // 2
            
            dy = min(h_model, h_film) // 2
            dx = min(w_model, w_film) // 2
            
            y1_src = cy_f - dy; y2_src = cy_f + dy
            x1_src = cx_f - dx; x2_src = cx_f + dx
            
            y1_dst = cy_m - dy; y2_dst = cy_m + dy
            x1_dst = cx_m - dx; x2_dst = cx_m + dx
            
            h_copy = min(y2_src-y1_src, y2_dst-y1_dst)
            w_copy = min(x2_src-x1_src, x2_dst-x1_dst)
            
            film_aligned[y1_dst:y1_dst+h_copy, x1_dst:x1_dst+w_copy] = \
                rear_film_map[y1_src:y1_src+h_copy, x1_src:x1_src+w_copy]
                
            rear_film_map = film_aligned 
        
        if np.sum(mask_proj) == 0:
            print("  ERROR: Mask projection is empty on rear film!")
            return

        # 1. Measured Values
        vals_measured = rear_film_map[mask_proj]
        avg_measured = np.mean(vals_measured)
        
        # 2. Gap Correction (Measurement -> Solid Equivalent)
        mu_pmma = 0.0287 # 1/mm @ 350 keV approx
        
        if gap_material == 'air': mu_gap = 0.0001
        elif gap_material == 'water': mu_gap = 0.010
        else: mu_gap = mu_pmma 
            
        mu_diff = mu_pmma - mu_gap
        gap_factor = np.exp(-mu_diff * gap_thickness_mm)
        
        avg_solid_equivalent = avg_measured * gap_factor
        
        print(f"  -> Measured Dose (Behind Gap): {avg_measured:.2f} Gy")
        print(f"  -> Gap Correction ({gap_thickness_mm}mm {gap_material}): factor={gap_factor:.3f}")
        print(f"  -> Solid-Equivalent Dose:      {avg_solid_equivalent:.2f} Gy")
        
        # 3. Model Comparison
        vals_model = vol_slice[mask_proj]
        avg_model = np.mean(vals_model)
        
        print(f"  -> Model Dose (Solid):         {avg_model:.2f} Gy")
        
        if avg_model == 0:
            print("  ERROR: Model dose is 0 under mask!")
            return

        ratio = avg_solid_equivalent / avg_model
        print(f"  -> Correction Ratio: {ratio:.4f}")
        
        if abs(ratio - 1.0) < 0.05:
            print("  -> VALIDATION OK: Model matches measurement (<5% error). No correction applied.")
            self.is_spectrally_corrected = True
            return

        # 4. Apply Slope Correction
        print("  -> DEVIATION DETECTED: Applying spectral slope correction...")
        slope = (ratio - 1.0) / rear_film_depth_mm
        
        z_values = self.coords['z']
        correction_vector = 1.0 + slope * z_values
        
        self.volume = self.volume * correction_vector[:, np.newaxis, np.newaxis]
        self.is_spectrally_corrected = True
        print(f"  -> Slope applied: {slope:.2e} /mm")

    def debug_spectral_correction(self, rear_film_package, rear_depth_mm: float, roi_name: str, gap_thickness_mm: float = 0.0):
        """
        Visual check of alignment between ROI Mask and Rear Film Shadow.
        Saves 'DEBUG_correction_alignment.png'.
        """
        print(f"\n--- DEBUG: Position & Shadow Check (Gap: {gap_thickness_mm} mm) ---")
        
        rear_film_map, rear_meta, rear_extent = rear_film_package
        
        if roi_name not in self.roi_masks: return
        target_mask_3d = self.roi_masks[roi_name]

        mask_proj = np.max(target_mask_3d, axis=0) 
        
        z_values = self.coords['z']
        rear_z_idx = np.argmin(np.abs(z_values - rear_depth_mm))
        vol_slice = self.volume[rear_z_idx, :, :]
        
        xs = self.coords['x']
        ys = self.coords['y']
        vol_extent = [xs[0], xs[-1], ys[0], ys[-1]]
        contour_data = [(xs, ys, mask_proj)]

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        self.plot_dose_map(mask_proj, vol_extent, ax=axes[0], 
                           title=f"Mask Projection ({roi_name})", cmap='gray_r')
        
        self.plot_dose_map(rear_film_map, rear_extent, ax=axes[1], 
                           title=f"Film + Mask Shadow",
                           mask_contours=contour_data)
        
        self.plot_dose_map(vol_slice, vol_extent, ax=axes[2], 
                           title="Model Dose (Solid)",
                           mask_contours=contour_data)
        
        plt.tight_layout()
        plt.savefig(f"DEBUG_{roi_name}_alignment.png")
        print(f"  -> Debug image saved: DEBUG_{roi_name}_alignment.png")
        plt.close(fig)

    def get_voxel_data(self, mask_name: str, calibration_k_factor: float = 1.0):
        """Returns raw voxel doses and uncertainties within a mask."""
        if mask_name not in self.roi_masks: return None, None
        
        mask = self.roi_masks[mask_name]
        
        raw_doses = self.volume[mask].flatten()
        raw_uncerts = self.uncertainty[mask].flatten()
        
        return raw_doses * calibration_k_factor, raw_uncerts * calibration_k_factor
    
    def get_dvh_statistics(self, mask_name: str, calibration_k_factor: float = 1.0, 
                           n_bins: int = 50, embryo_diameter_mm: float = 1.0):
        """
        Calculates differential DVH (Dose Volume Histogram) statistics.
        
        Args:
            embryo_diameter_mm: Smoothing kernel size to simulate the physical size 
                                of the biological entity (e.g., embryo).
        """
        if mask_name not in self.roi_masks:
            raise ValueError(f"Mask not found: {mask_name}")
            
        # 1. Smoothing (Embryo Simulation)
        if embryo_diameter_mm > 0:
            pixel_size = self.pix_mm
            kernel_size = int(round(embryo_diameter_mm / pixel_size))
            
            if kernel_size > 1:
                volume_to_use = uniform_filter(self.volume, size=kernel_size, mode='reflect')
                uncert_to_use = uniform_filter(self.uncertainty, size=kernel_size, mode='reflect')
            else:
                volume_to_use = self.volume
                uncert_to_use = self.uncertainty
        else:
            volume_to_use = self.volume
            uncert_to_use = self.uncertainty

        # 2. Masking
        mask = self.roi_masks[mask_name]
        doses_raw = volume_to_use[mask].flatten() * calibration_k_factor
        uncerts_raw = uncert_to_use[mask].flatten() * calibration_k_factor
        
        if len(doses_raw) == 0: raise ValueError(f"Mask ({mask_name}) is empty!")

        # 3. Histogram
        hist_counts, bin_edges = np.histogram(doses_raw, bins=n_bins)
        total_voxels = len(doses_raw)
        weights = hist_counts / total_voxels
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 4. Bin Uncertainties
        bin_stds = np.zeros_like(bin_centers)
        digitized_indices = np.digitize(doses_raw, bin_edges)
        
        for i in range(len(bin_centers)):
            bin_idx = i + 1 
            in_bin_mask = (digitized_indices == bin_idx)
            if np.any(in_bin_mask):
                bin_stds[i] = np.mean(uncerts_raw[in_bin_mask])
            else:
                bin_stds[i] = 1e-3

        return {
            'dose_stats': np.vstack([bin_centers, bin_stds]),
            'weights': weights,
            'meta': {
                'voxel_count': total_voxels,
                'min_dose': np.min(doses_raw),
                'max_dose': np.max(doses_raw),
                'mean_dose': np.mean(doses_raw),
                'embryo_smoothing_mm': embryo_diameter_mm
            }
        }
        
    def get_equivalent_uniform_dose(self, mask_name: str, a_parameter: float = 1.0):
        """
        Calculates EUD (Equivalent Uniform Dose) using the Niemierko formula.
        
        a = 1.0   -> Mean Dose
        a = -10.0 -> Minimum Dose (Tumor)
        a = +10.0 -> Maximum Dose (OAR)
        """
        stats = self.get_dvh_statistics(mask_name, n_bins=100)
        dose_bins = stats['dose_stats'][0, :]
        weights = stats['weights']
        
        if abs(a_parameter) < 1e-3:
            # Geometric mean (limit as a->0)
            valid_idx = dose_bins > 0
            d_valid = dose_bins[valid_idx]
            w_valid = weights[valid_idx] / np.sum(weights[valid_idx])
            eud = np.exp(np.sum(w_valid * np.log(d_valid)))
        else:
            sum_val = np.sum(weights * (dose_bins ** a_parameter))
            eud = sum_val ** (1.0 / a_parameter)
            
        print(f"[{mask_name}] EUD (a={a_parameter}): {eud:.2f} Gy")
        return eud

    # --- ROI MASK GENERATORS ---

    def _create_rotated_cylindrical_mask(self, name, tip_position, angle_deg, radius, 
                                         tip_length, cylinder_length, tip_shape='round',
                                         rounding_radius_mm=None):
        """Internal helper for creating rotated cylindrical ROIs (tubes, chambers)."""
        if self.volume is None: return
        print(f"[{name}] Creating Mask: R={radius:.2f}mm, Tip={tip_length:.1f}mm ({tip_shape}), Angle={angle_deg}°")

        tx, ty, tz = tip_position
        Z, Y, X = self.coords['z'], self.coords['y'], self.coords['x']
        
        dx = X[None, None, :] - tx
        dy = Y[None, :, None] - ty
        dz = Z[:, None, None] - tz 
        
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        
        w = dx * c + dy * s      # Longitudinal axis
        u = -dx * s + dy * c     
        v = dz                   
        r_sq = u**2 + v**2
        
        # 1. Body
        mask_cyl = (w >= tip_length) & (w <= (tip_length + cylinder_length)) & (r_sq <= radius**2)
        
        # 2. Tip
        mask_tip_region = (w >= 0) & (w < tip_length)
        
        if tip_shape == 'round':
            dist_from_sphere_center = (w - tip_length)**2 + r_sq
            mask_tip_shape = dist_from_sphere_center <= radius**2
            
        elif tip_shape == 'rounded_conical':
            # Eppendorf style
            r_big = radius
            r_small = rounding_radius_mm if rounding_radius_mm else radius * 0.25
            cone_length = tip_length - r_small
            w_cone = w - r_small
            
            mask_small_sphere = (w < r_small) & ( ((w - r_small)**2 + r_sq) <= r_small**2 )
            
            current_radius = r_small + (w_cone / cone_length) * (r_big - r_small)
            mask_cone_part = (w >= r_small) & (r_sq <= current_radius**2)
            
            mask_tip_shape = mask_small_sphere | mask_cone_part
        else:
            mask_tip_shape = False

        mask_tip = mask_tip_region & mask_tip_shape
        self.roi_masks[name] = mask_cyl | mask_tip
        
        vox_count = np.sum(self.roi_masks[name])
        vol_mm3 = vox_count * (self.pix_mm**2) * (Z[1]-Z[0])
        print(f"  -> Created: {vol_mm3:.1f} mm3")
        
    def add_eppendorf_mask(self, name, volume_ml=1.5, tip_position=(0,0,15), angle_from_x_deg=0.0, filled_height_mm=None):
        """Creates an ROI mask for an Eppendorf tube (1.5ml or 2.0ml)."""
        radius = 5.4  # ~10.8 mm diameter
        
        if abs(volume_ml - 1.5) < 0.1:
            tip_length = 17.5 
            tip_shape = 'rounded_conical'
            total_nominal_length = 39.0 
            rounding_r = 1.8 
        elif abs(volume_ml - 2.0) < 0.1:
            tip_length = radius 
            tip_shape = 'round'
            total_nominal_length = 40.0
            rounding_r = None 
        else:
            print(f"ERROR: Unknown Eppendorf volume: {volume_ml}ml")
            return

        total_length = filled_height_mm if filled_height_mm else total_nominal_length
        cyl_length = max(0, total_length - tip_length)

        self._create_rotated_cylindrical_mask(name, tip_position, angle_from_x_deg, radius,
                                              tip_length, cyl_length, tip_shape, rounding_r)

    def add_monolayer_well_mask(self, name, center_coords, radius_mm=3.2, z_thickness_mm=0.5):
        """Creates a flat cylindrical mask for cell monolayers (96-well plate)."""
        if self.volume is None: return
        print(f"[{name}] Monolayer Well (R={radius_mm}mm, Z={center_coords[2]}mm)...")
        
        cx, cy, cz = center_coords
        Z, Y, X = self.coords['z'], self.coords['y'], self.coords['x']
        
        mask_z = (np.abs(Z[:, None, None] - cz) <= (z_thickness_mm / 2.0))
        dist_sq_xy = (X[None, None, :] - cx)**2 + (Y[None, :, None] - cy)**2
        mask_xy = dist_sq_xy <= radius_mm**2
        
        self.roi_masks[name] = mask_z & mask_xy

    # --- PLOTTING & EXPORT ---

    def plot_ortho_views(self, slice_coords=(0,0,15)):
        """Plots Axial, Coronal, and Sagittal views of the volume."""
        if self.volume is None: return
        cx, cy, cz = slice_coords
        
        ix = np.argmin(np.abs(self.coords['x'] - cx))
        iy = np.argmin(np.abs(self.coords['y'] - cy))
        iz = np.argmin(np.abs(self.coords['z'] - cz))
        
        x_min, x_max = self.coords['x'][0], self.coords['x'][-1]
        y_min, y_max = self.coords['y'][0], self.coords['y'][-1]
        z_min, z_max = self.coords['z'][0], self.coords['z'][-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # 1. Axial (XY)
        im1 = axes[0].imshow(self.volume[iz, :, :], cmap='turbo', origin='lower',
                       extent=[x_min, x_max, y_min, y_max], aspect='equal')
        axes[0].set_title(f"Axial (XY) @ Z={self.coords['z'][iz]:.1f} mm")
        axes[0].set_xlabel("X [mm]"); axes[0].set_ylabel("Y [mm]")
        
        # 2. Coronal (XZ)
        im2 = axes[1].imshow(self.volume[:, iy, :], cmap='turbo', origin='upper',
                       extent=[x_min, x_max, z_max, z_min], aspect='auto')
        axes[1].set_title(f"Coronal (XZ) @ Y={self.coords['y'][iy]:.1f} mm")
        axes[1].set_xlabel("X [mm]"); axes[1].set_ylabel("Depth Z [mm]")

        # 3. Sagittal (YZ)
        im3 = axes[2].imshow(self.volume[:, :, ix], cmap='turbo', origin='upper',
                       extent=[y_min, y_max, z_max, z_min], aspect='auto')
        axes[2].set_title(f"Sagittal (YZ) @ X={self.coords['x'][ix]:.1f} mm")
        axes[2].set_xlabel("Y [mm]"); axes[2].set_ylabel("Depth Z [mm]")

        plt.tight_layout()
        cbar = fig.colorbar(im3, ax=axes.ravel().tolist(), orientation='horizontal', 
                            fraction=0.05, pad=0.1, label="Dose [Gy]")
        plt.show()
    
    def save_volume(self, filepath: str):
        """Saves the volume and metadata to a compressed .npz file."""
        if not self.is_built:
            print("Warning: Volume is not built yet, nothing to save.")
            return

        print(f"--- Saving 3D Volume to {filepath} ---")
        np.savez_compressed(
            filepath,
            volume=self.volume,
            uncertainty=self.uncertainty,
            z_coords=self.coords['z'],
            y_coords=self.coords['y'],
            x_coords=self.coords['x'],
            pix_mm=self.pix_mm,
            is_aligned=self.is_aligned,
            is_scaled=self.is_scaled,
            is_spectrally_corrected=self.is_spectrally_corrected
        )
        print("  -> Save successful.")

    def load_volume_from_file(self, filepath: str):
        """Loads volume from .npz and reconstructs coordinates if missing."""
        print(f"--- Loading Volume from: {filepath} ---")
        if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        self.volume = data['volume']
        
        if 'pixel_spacing' in data:
            self.pix_mm = float(data['pixel_spacing'])
        elif self.pix_mm is None:
            self.pix_mm = 0.08466 # Default 300 DPI
            
        if 'coords' in data:
            self.coords = data['coords'].item()
        else:
            print("  WARNING: Coords missing, reconstructing...")
            nz, ny, nx = self.volume.shape
            z_res = 0.2
            self.coords = {
                'z': np.arange(nz) * z_res,
                'y': np.arange(ny) * self.pix_mm - (ny * self.pix_mm / 2.0),
                'x': np.arange(nx) * self.pix_mm - (nx * self.pix_mm / 2.0)
            }

        self.uncertainty = data['uncertainty'] if 'uncertainty' in data else np.zeros_like(self.volume)
        self.create_backup_from_current_state()
        
    @staticmethod
    def load_dose_map(file_path: str):
        """Loads a .npy dose map and associated JSON metadata."""
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        
        dose_map = np.load(file_path)
        
        if file_path.endswith("_dose.npy"):
            json_path = file_path.replace("_dose.npy", "_meta.json")
        else:
            json_path = os.path.splitext(file_path)[0] + ".json"
            if not os.path.exists(json_path): json_path = os.path.splitext(file_path)[0] + "_meta.json"
        
        meta = {}
        pix_mm = 0.08466
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f: meta = json.load(f)
            pix_mm = float(meta.get('pixel_spacing_mm') or meta.get('pixel_spacing') or pix_mm)
            
        h, w = dose_map.shape
        width_mm = w * pix_mm
        height_mm = h * pix_mm
        extent = [-width_mm/2, width_mm/2, -height_mm/2, height_mm/2]
        
        meta['physical_size_mm'] = (height_mm, width_mm)
        return dose_map, meta, extent
    
    @staticmethod
    def plot_dose_map(dose_map, extent, ax=None, title="Dose Map", mask_contours=None, cmap='jet'):
        """Helper to plot a 2D dose map with physical units."""
        if ax is None: fig, ax = plt.subplots(figsize=(6, 6))
        
        im = ax.imshow(dose_map, cmap=cmap, extent=extent, origin='lower')
        
        if mask_contours:
            for x_grid, y_grid, mask_proj in mask_contours:
                ax.contour(x_grid, y_grid, mask_proj, colors='white', linewidths=1.5, linestyles='--')
        
        ax.set_title(title)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
        try: plt.colorbar(im, ax=ax, label="Dose [Gy]", fraction=0.046, pad=0.04)
        except: pass
        return im

###############################################################################
# Utility Functions (Recommend moving to prism/viz.py and prism/dosimetry.py)
###############################################################################

def plot_dvh_comparison(filenames, labels=None, cumulative=False):
    """Compare multiple DVH files."""
    plt.figure(figsize=(10, 6))
    if labels is None: labels = [f"DVH {i+1}" for i in range(len(filenames))]
        
    for i, fname in enumerate(filenames):
        data = np.load(fname, allow_pickle=True)
        d, w = data['dose_bins'], data['weights']
        
        if cumulative:
            idx = np.argsort(d)
            d_sorted, w_sorted = d[idx], w[idx]
            cum_w = np.cumsum(w_sorted[::-1])[::-1]
            cum_w = cum_w / cum_w[0] * 100
            plt.plot(d_sorted, cum_w, lw=2, label=labels[i])
            plt.ylabel("Volume [%]")
            plt.ylim(0, 105)
        else:
            plt.plot(d, w, lw=2, alpha=0.8, label=labels[i])
            plt.fill_between(d, 0, w, alpha=0.2)
            plt.ylabel("Relative Frequency")

    plt.xlabel("Dose [Gy]")
    title_type = "Cumulative (cDVH)" if cumulative else "Differential (dDVH)"
    plt.title(f"Dose-Volume Histograms ({title_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

class DoseMetrics:
    """Helper for calculating EUD and processing batch files."""
    @staticmethod
    def calculate_eud_from_file(file_path, a=-10.0):
        if not os.path.exists(file_path): return None
        try:
            with np.load(file_path, allow_pickle=True) as data:
                bins = np.nan_to_num(data['dose_bins'])
                weights = np.nan_to_num(data['weights'])
                
            total_weight = np.sum(weights)
            if total_weight <= 0: return 0.0
            weights /= total_weight
            
            if abs(a - 1.0) < 1e-6: return np.sum(weights * bins)
            
            valid_mask = weights > 1e-9
            bins_active = bins[valid_mask]
            weights_active = weights[valid_mask]
            if len(bins_active) == 0: return 0.0

            if a < 0:
                if np.min(bins_active) < 1e-6: return 0.0
                sum_val = np.sum(weights_active * (bins_active ** a))
                return sum_val ** (1.0 / a)
            else:
                sum_val = np.sum(weights_active * (bins_active ** a))
                return sum_val ** (1.0 / a)
        except Exception as e:
            print(f"Error calculating EUD for {file_path}: {e}")
            return None