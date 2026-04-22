#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:01:35 2026

@author: polanekr
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import center_of_mass, median_filter, label
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

class FilmAnalyzer:
    def __init__(self, npy_path: str, meta_path: str = None, crop_threshold_pct=5, remove_borders_px=15):
        """
        Incializálás, fizikai perem-vágás, okos vágás és profil elemzés.
        """
        self.npy_path = npy_path
        raw_map = np.load(npy_path)
        self.dose_map = np.nan_to_num(raw_map, nan=0.0)
        
        # 1. Fizikai vágás: Eldobjuk a szkenner és a fólia széleit (nincs több homogén keret!)
        self._remove_film_borders(border_pixels=remove_borders_px)
        
        # 2. Metaadatok betöltése
        self.pixel_spacing_mm = self._load_meta(meta_path)
        
        # 3. Spot közepének meghatározása a nyers képen
        self.cx_orig, self.cy_orig = self._find_beam_center(self.dose_map)
        
        # 4. AUTOMATIKUS VÁGÁS (PROMINENCIA ALAPJÁN)
        self._apply_smart_crop(crop_threshold_pct)
        
        # 5. KOORDINÁTA-RENDSZER ELTOLÁSA (0-központú)
        self._update_coordinates()

    def _remove_film_borders(self, border_pixels):
        """Fizikailag levágja a kép széleit (artifaktok eltávolítása)."""
        h, w = self.dose_map.shape
        if h > 2*border_pixels and w > 2*border_pixels:
            self.dose_map = self.dose_map[border_pixels:-border_pixels, border_pixels:-border_pixels]

    def _load_meta(self, meta_path):
        if meta_path is None:
            meta_path = self.npy_path.replace('_dose.npy', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                return meta.get('pixel_spacing_mm', 0.0846)
        return 0.0846

    def _find_beam_center(self, d_map):
        smoothed = median_filter(d_map, size=5)
        threshold = np.max(smoothed) * 0.8
        mask = smoothed > threshold
        if np.sum(mask) > 0:
            cy_pix, cx_pix = center_of_mass(smoothed * mask)
            return int(cx_pix), int(cy_pix)
        return d_map.shape[1] // 2, d_map.shape[0] // 2

    def _apply_smart_crop(self, threshold_pct):
        """A vágás most a Háttér és a Maximum különbsége (Prominencia) alapján történik."""
        baseline = np.percentile(self.dose_map, 5) # A film legkevésbé sugárzott 5%-a
        peak = np.max(self.dose_map)
        prominence = peak - baseline
        
        # Vágási küszöb: A háttér felett a csúcs x%-a (pl. 5%)
        threshold = baseline + prominence * (threshold_pct / 100.0)
        
        base_mask = self.dose_map > threshold
        labeled_mask, num_features = label(base_mask)
        
        if num_features > 0:
            sizes = np.bincount(labeled_mask.ravel())
            sizes[0] = 0 
            main_label = sizes.argmax()
            main_mask = (labeled_mask == main_label)
        else:
            main_mask = base_mask 
            
        rows = np.any(main_mask, axis=1)
        cols = np.any(main_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols): return 
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Szűk, 3 pixeles margó, hogy a grafikon tényleg csak a lényeget mutassa
        margin = 3
        ymin = max(0, ymin - margin)
        ymax = min(self.dose_map.shape[0], ymax + margin)
        xmin = max(0, xmin - margin)
        xmax = min(self.dose_map.shape[1], xmax + margin)
        
        self.dose_map = self.dose_map[ymin:ymax, xmin:xmax]
        self.cx = self.cx_orig - xmin
        self.cy = self.cy_orig - ymin

    def _update_coordinates(self):
        h, w = self.dose_map.shape
        self.x_mm = (np.arange(w) - self.cx) * self.pixel_spacing_mm
        self.y_mm = (np.arange(h) - self.cy) * self.pixel_spacing_mm

    def _compute_profile_metrics(self, x, profile):
        """Bolondbiztos, magas háttérdózisra felkészített profil-matek."""
        # Profil simítása
        try:
            window = min(11, len(profile) // 2 * 2 + 1)
            prof_smooth = savgol_filter(profile, window, 3)
        except:
            prof_smooth = profile

        baseline = np.percentile(prof_smooth, 5)
        peak = np.max(prof_smooth)
        prominence = peak - baseline

        # Ha túl lapos, nincs mit mérni
        if prominence < 0.1:
            return prof_smooth, {'FWHM': np.nan, 'Penumbra_L': np.nan, 'Penumbra_R': np.nan, 'Flatness': np.nan, 'half_level': np.nan}

        def get_roots(level):
            spline = UnivariateSpline(x, prof_smooth - level, s=0)
            return np.sort(spline.roots())

        # FWHM (A Prominencia felénél!)
        half_level = baseline + prominence * 0.5
        roots_50 = get_roots(half_level)
        
        fwhm, center = np.nan, 0.0
        if len(roots_50) >= 2:
            fwhm = roots_50[-1] - roots_50[0]
            center = (roots_50[-1] + roots_50[0]) / 2.0

        # Penumbra (20-80 a Prominenciára)
        p_L, p_R = np.nan, np.nan
        roots_20 = get_roots(baseline + prominence * 0.2)
        roots_80 = get_roots(baseline + prominence * 0.8)

        if len(roots_20) >= 2 and len(roots_80) >= 2:
            l20 = [r for r in roots_20 if r < center]; l80 = [r for r in roots_80 if r < center]
            if l20 and l80: p_L = abs(l80[-1] - l20[0])

            r20 = [r for r in roots_20 if r > center]; r80 = [r for r in roots_80 if r > center]
            if r20 and r80: p_R = abs(r20[0] - r80[-1])

        # Flatness (A FWHM 80%-án belül)
        flatness = np.nan
        if not np.isnan(fwhm):
            mask = (x >= center - fwhm * 0.4) & (x <= center + fwhm * 0.4)
            if np.any(mask):
                roi = prof_smooth[mask]
                flatness = 100 * (np.max(roi) - np.min(roi)) / (np.max(roi) + np.min(roi))

        return prof_smooth, {
            'FWHM': fwhm, 'Penumbra_L': p_L, 'Penumbra_R': p_R, 
            'Flatness': flatness, 'half_level': half_level, 'center': center
        }

    def calculate_central_dose(self, roi_radius_mm=2.0):
        roi_radius_pix = int(roi_radius_mm / self.pixel_spacing_mm)
        y, x = np.ogrid[:self.dose_map.shape[0], :self.dose_map.shape[1]]
        mask = ((x - self.cx)**2 + (y - self.cy)**2) <= roi_radius_pix**2
        
        if np.sum(mask) == 0: return 0.0, 0.0
        return np.mean(self.dose_map[mask]), np.std(self.dose_map[mask])

    def extract_profiles(self):
        return self.dose_map[self.cy, :], self.dose_map[:, self.cx]

    def generate_report(self, save_path=None):
        profile_x_raw, profile_y_raw = self.extract_profiles()
        
        # --- 1. Adatok elemzése a saját beépített metódussal ---
        d_smooth_x, res_x = self._compute_profile_metrics(self.x_mm, profile_x_raw)
        d_smooth_y, res_y = self._compute_profile_metrics(self.y_mm, profile_y_raw)
        
        central_dose, central_std = self.calculate_central_dose(roi_radius_mm=2.0)
        
        def safe_fmt(val): return f"{val:.2f}" if not np.isnan(val) else "nan"

        # --- 2. Riport nyomtatása ---
        print("\n" + "="*60)
        print(f" DOSIMETRY REPORT: {os.path.basename(self.npy_path)}")
        print("="*60)
        print(f"Central Dose (2mm ROI) : {central_dose:.2f} \u00B1 {central_std:.2f} Gy")
        print(f"Max Point Dose         : {np.max(self.dose_map):.2f} Gy")
        print("-" * 60)
        print("HORIZONTAL (X) PROFILE:")
        print(f"  FWHM                 : {safe_fmt(res_x['FWHM'])} mm")
        print(f"  Penumbra (L / R)     : {safe_fmt(res_x['Penumbra_L'])} / {safe_fmt(res_x['Penumbra_R'])} mm")
        print(f"  Flatness (80% FWHM)  : {safe_fmt(res_x['Flatness'])} %")
        print("-" * 60)
        print("VERTICAL (Y) PROFILE:")
        print(f"  FWHM                 : {safe_fmt(res_y['FWHM'])} mm")
        print(f"  Penumbra (Top / Bot) : {safe_fmt(res_y['Penumbra_L'])} / {safe_fmt(res_y['Penumbra_R'])} mm")
        print(f"  Flatness (80% FWHM)  : {safe_fmt(res_y['Flatness'])} %")
        print("="*60)

        # --- 3. Publikációkész Ábra ---
        plt.style.use('default')
        fig = plt.figure(figsize=(12, 8), dpi=300)
        gs = GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

        # A) 2D Térkép
        ax_map = fig.add_subplot(gs[:, 0])
        extent = [self.x_mm[0], self.x_mm[-1], self.y_mm[-1], self.y_mm[0]]
        im = ax_map.imshow(self.dose_map, cmap='inferno', extent=extent, origin='upper', interpolation='gaussian')
        
        ax_map.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
        ax_map.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
        ax_map.text(0.05, 0.95, f"Central Dose:\n{central_dose:.2f} Gy", transform=ax_map.transAxes, 
                    fontsize=11, fontweight='bold', va='top', bbox=bbox_props)
        
        ax_map.set_title("Relative 2D Dose Map", fontsize=14, fontweight='bold')
        ax_map.set_xlabel("Relative X [mm]", fontsize=12)
        ax_map.set_ylabel("Relative Y [mm]", fontsize=12)
        plt.colorbar(im, ax=ax_map, label='Absolute Dose [Gy]')

        # B) Horizontális Profil
        ax_x = fig.add_subplot(gs[0, 1])
        ax_x.plot(self.x_mm, profile_x_raw, 'o', color='lightgray', markersize=3)
        ax_x.plot(self.x_mm, d_smooth_x, 'r-', linewidth=2)
        ax_x.axvline(0, color='black', linestyle=':', alpha=0.5)
        
        if not np.isnan(res_x['FWHM']):
            ax_x.hlines(res_x['half_level'], res_x['center'] - res_x['FWHM']/2, res_x['center'] + res_x['FWHM']/2, 
                        colors='blue', linestyles='-', lw=2, label=f'FWHM: {res_x["FWHM"]:.1f} mm')
            ax_x.legend(fontsize=9, loc='upper right')

        ax_x.set_title("Horizontal Profile (X)", fontweight='bold')
        ax_x.set_xlabel("Distance from Center [mm]")
        ax_x.set_ylabel("Dose [Gy]")
        ax_x.grid(True, alpha=0.3)
        # SZIGORÚ VÁGÁS A TENGELYEN:
        ax_x.set_xlim([self.x_mm[0], self.x_mm[-1]])

        # C) Vertikális Profil
        ax_y = fig.add_subplot(gs[1, 1])
        ax_y.plot(self.y_mm, profile_y_raw, 'o', color='lightgray', markersize=3)
        ax_y.plot(self.y_mm, d_smooth_y, 'b-', linewidth=2)
        ax_y.axvline(0, color='black', linestyle=':', alpha=0.5)
        
        if not np.isnan(res_y['FWHM']):
            ax_y.hlines(res_y['half_level'], res_y['center'] - res_y['FWHM']/2, res_y['center'] + res_y['FWHM']/2, 
                        colors='red', linestyles='-', lw=2, label=f'FWHM: {res_y["FWHM"]:.1f} mm')
            ax_y.legend(fontsize=9, loc='upper right')

        ax_y.set_title("Vertical Profile (Y)", fontweight='bold')
        ax_y.set_xlabel("Distance from Center [mm]")
        ax_y.set_ylabel("Dose [Gy]")
        ax_y.grid(True, alpha=0.3)
        # SZIGORÚ VÁGÁS A TENGELYEN:
        ax_y.set_xlim([self.y_mm[0], self.y_mm[-1]])

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Publication figure saved to: {save_path}")
            
        plt.show()
        
        return {"Central_Dose_Gy": central_dose, "Metrics_X": res_x, "Metrics_Y": res_y}

# ==========================================
# HASZNÁLATI PÉLDA (Tedd a szkript végére)
# ==========================================
if __name__ == "__main__":
    # Példa: tegyük fel, hogy az elemzett film fájlja "film_001_dose.npy"
    # cseréld ki egy valós fájlnevedre!
    npy_file = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-11-27-eSYLOS-eli60052/PDD2/F_025_dose.npy" 
    
    if os.path.exists(npy_file):
        analyzer = FilmAnalyzer(npy_path=npy_file)
        # Elkészíti az elemzést és elmenti a képet
        report_data = analyzer.generate_report(save_path="/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-11-27-eSYLOS-eli60052/PDD2/F_025_dose.png")
    else:
        print("Kérlek futtasd le ezt úgy, hogy létezik a megadott npy fájl!")