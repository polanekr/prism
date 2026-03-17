#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:59:04 2026

@author: polanekr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates

# Importáljuk a reconstructor osztályt a saját fájlodból
from prism.reconstruction import Bayesian3DVolumeReconstructor

def generate_local_planes_survival(pdd_folder: str, front_film_path: str, rear_film_path: str):
    print("--- 9. Ábra: Lokális Cső-koordinátás Túlélési Térkép (Szorosra vágva) ---")

    # =========================================================================
    # 1. REKONSTRUKCIÓS PIPELINE
    # =========================================================================
    try:
        recon = Bayesian3DVolumeReconstructor(pdd_folder=pdd_folder, ssd_mm=300.0)
    except Exception as e:
        print(f"  [!] Hiba a PDD mappa beolvasásakor: {e}")
        return

    recon.build_volume(roi_size_mm=90.0, z_res_mm=0.5, perform_alignment=True)

    if os.path.exists(front_film_path):
        front_map, _, _ = recon.load_dose_map(front_film_path)
        recon.register_and_scale(front_map, front_film_depth_mm=14.5)

    # 2.0 ml-es cső maszkjának paraméterei
    mask_name = "Eppendorf_2_0ml"
    tube_angle = -75.0
    tx, ty, tz = 2.6, 7.6, 22.0
    recon.add_eppendorf_mask(mask_name, volume_ml=2.0, tip_position=(tx, ty, tz), 
                             angle_from_x_deg=tube_angle, filled_height_mm=20.0)

    if os.path.exists(rear_film_path):
        rear_map, _, _ = recon.load_dose_map(rear_film_path)
        recon.correct_spectrum_with_cavity(
            rear_film_map=rear_map, rear_film_depth_mm=31.2, 
            mask_name=mask_name, gap_thickness_mm=10.8, gap_material='water'
        )

    spatial_data = recon.get_spatial_dose_map(mask_name)
    dose_3d = spatial_data['dose_3d']
    mask_3d = spatial_data['mask_3d']
    
    Z_orig = spatial_data['coords']['z']
    Y_orig = spatial_data['coords']['y']
    X_orig = spatial_data['coords']['x']

    # =========================================================================
    # 2. BIOLÓGIA SZÁMÍTÁSA
    # =========================================================================
    alpha, beta = 0.30, 0.030
    survival_3d = np.exp(-alpha * dose_3d - beta * dose_3d**2)
    log_surv_3d = np.log10(survival_3d + 1e-10)

    # =========================================================================
    # 3. LOKÁLIS KOORDINÁTA RENDSZEREK ÉS KIVÁGÁS
    # =========================================================================
    print("  -> Lokális síkok kivágása interpolációval...")
    theta = np.radians(tube_angle)
    c, s = np.cos(theta), np.sin(theta)

    # JAVÍTÁS: A W-tengelyt (mélység) drasztikusan szűkítettük -11-től 11-ig (fesztáv: 22 mm).
    # A cső sugara ~5.4 mm, így mindkét oldalon pont egy elegáns ~5.6 mm-es margó marad!
    u_axis = np.linspace(-3, 24, 135)   # Fesztáv: 27 mm
    v_axis = np.linspace(-8, 8, 80)     # Fesztáv: 16 mm
    w_axis = np.linspace(-11, 11, 110)  # Fesztáv: 22 mm

    def extract_plane(U_grid, V_grid, W_grid, volume_data, order, cval):
        X_eval = tx + U_grid * c - V_grid * s
        Y_eval = ty + U_grid * s + V_grid * c
        Z_eval = tz + W_grid
        idx_z = (Z_eval - Z_orig[0]) / (Z_orig[1] - Z_orig[0])
        idx_y = (Y_eval - Y_orig[0]) / (Y_orig[1] - Y_orig[0])
        idx_x = (X_eval - X_orig[0]) / (X_orig[1] - X_orig[0])
        return map_coordinates(volume_data, np.stack([idx_z, idx_y, idx_x]), order=order, cval=cval)

    # --- A. AXIAL (U-V sík) ---
    U_ax, V_ax = np.meshgrid(u_axis, v_axis, indexing='ij')
    W_ax = np.zeros_like(U_ax)
    dose_ax = extract_plane(U_ax, V_ax, W_ax, dose_3d, order=1, cval=0.0)
    surv_ax = extract_plane(U_ax, V_ax, W_ax, log_surv_3d, order=1, cval=0.0)
    mask_ax = extract_plane(U_ax, V_ax, W_ax, mask_3d.astype(float), order=0, cval=0.0) > 0.5
    ext_ax = [v_axis[0], v_axis[-1], u_axis[-1], u_axis[0]]

    # --- B. CORONAL (U-W sík) ---
    U_cor, W_cor = np.meshgrid(u_axis, w_axis, indexing='ij')
    V_cor = np.zeros_like(U_cor)
    dose_cor = extract_plane(U_cor, V_cor, W_cor, dose_3d, order=1, cval=0.0)
    surv_cor = extract_plane(U_cor, V_cor, W_cor, log_surv_3d, order=1, cval=0.0)
    mask_cor = extract_plane(U_cor, V_cor, W_cor, mask_3d.astype(float), order=0, cval=0.0) > 0.5
    ext_cor = [w_axis[0], w_axis[-1], u_axis[-1], u_axis[0]] 

    # --- C. SAGITTAL (V-W sík) FIX 1 CM-NÉL ---
    V_sag, W_sag = np.meshgrid(v_axis, w_axis, indexing='ij')
    U_sag = np.full_like(V_sag, 10.0)
    dose_sag = extract_plane(U_sag, V_sag, W_sag, dose_3d, order=1, cval=0.0)
    surv_sag = extract_plane(U_sag, V_sag, W_sag, log_surv_3d, order=1, cval=0.0)
    mask_sag = extract_plane(U_sag, V_sag, W_sag, mask_3d.astype(float), order=0, cval=0.0) > 0.5
    ext_sag = [w_axis[0], w_axis[-1], v_axis[-1], v_axis[0]]

    # =========================================================================
    # 4. ÁBRA RAJZOLÁSA
    # =========================================================================
    # JAVÍTOTT ARÁNYOK: 16 mm vs 22 mm vs 22 mm fizikai szélességekhez igazítva!
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={'width_ratios': [16, 22, 22]}, constrained_layout=True)
    
    valid_dose = dose_3d[mask_3d]
    valid_surv = log_surv_3d[mask_3d]
    vmin_d, vmax_d = np.min(valid_dose), np.max(valid_dose)
    vmin_s, vmax_s = np.min(valid_surv), np.max(valid_surv)

    def plot_panel(ax, bg_data, fg_data, mask_data, extent, title, xlabel, ylabel, cmap, vmin, vmax):
        ax.set_facecolor('#E0E0E0')
        ax.imshow(bg_data, extent=extent, origin='upper', cmap='gray', alpha=0.15, vmin=vmin, vmax=vmax, aspect='equal')
        im = ax.imshow(np.where(mask_data, fg_data, np.nan), extent=extent, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(mask_data, levels=[0.5], extent=extent, origin='upper', colors='black', linewidths=1.5)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        return im

    # --- FELSŐ SOR: DÓZIS ---
    plot_panel(axes[0, 0], dose_ax, dose_ax, mask_ax, ext_ax, 
               "A. Axial (Top View)", "Tube Width [mm]", "Tube Length (Tip to Cap) [mm]", 'magma', vmin_d, vmax_d)
    
    axes[0, 0].text(0.05, 0.95, '\u2297', transform=axes[0, 0].transAxes,
                    fontsize=18, fontweight='bold', color='navy', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='navy'))
    
    im_d = plot_panel(axes[0, 1], dose_cor, dose_cor, mask_cor, ext_cor, 
                      "B. Coronal (Side View)", "Beam Depth Axis [mm]", "Tube Length (Tip to Cap) [mm]", 'magma', vmin_d, vmax_d)
    
    # Rövid nyíl: Most, hogy keskenyebb a doboz, 22%-os hosszig nyújtjuk, hogy szépen beérjen a cső elé
    axes[0, 1].annotate('', xy=(0.22, 0.92), xytext=(0.04, 0.92), xycoords='axes fraction',
                        arrowprops=dict(facecolor='navy', edgecolor='navy', shrink=0.0, width=2.0, headwidth=8))

    plot_panel(axes[0, 2], dose_sag, dose_sag, mask_sag, ext_sag, 
               "C. Sagittal (Cross-Section @ 1cm)", "Beam Depth Axis [mm]", "Tube Width [mm]", 'magma', vmin_d, vmax_d)
    
    axes[0, 2].annotate('', xy=(0.22, 0.92), xytext=(0.04, 0.92), xycoords='axes fraction',
                        arrowprops=dict(facecolor='navy', edgecolor='navy', shrink=0.0, width=2.0, headwidth=8))

    fig.colorbar(im_d, ax=axes[0, :], location='right', pad=0.02, fraction=0.03).set_label("Absolute Dose [Gy]", fontweight='bold')

    # --- ALSÓ SOR: TÚLÉLÉS ---
    plot_panel(axes[1, 0], dose_ax, surv_ax, mask_ax, ext_ax, 
               "D. Axial Cell Survival", "Tube Width [mm]", "Tube Length (Tip to Cap) [mm]", 'turbo', vmin_s, vmax_s)

    im_s = plot_panel(axes[1, 1], dose_cor, surv_cor, mask_cor, ext_cor, 
                      "E. Coronal Cell Survival", "Beam Depth Axis [mm]", "Tube Length (Tip to Cap) [mm]", 'turbo', vmin_s, vmax_s)
    
    plot_panel(axes[1, 2], dose_sag, surv_sag, mask_sag, ext_sag, 
               "F. Sagittal Cell Survival", "Beam Depth Axis [mm]", "Tube Width [mm]", 'turbo', vmin_s, vmax_s)

    cbar = fig.colorbar(im_s, ax=axes[1, :], location='right', pad=0.02, fraction=0.03)
    cbar.set_label("Cell Survival Probability (Log10)", fontweight='bold')
    
    ticks = np.linspace(vmin_s, vmax_s, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'$10^{{{t:.1f}}}$' for t in ticks])

    filename = "fig9_local_planes_survival_final.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    # SAJÁT ELÉRÉSI UTAK
    MY_PDD_FOLDER = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/PDD1/"
    FRONT_FILM = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/F_007_dose.npy"
    REAR_FILM = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/F_008_dose.npy"
    
    generate_local_planes_survival(MY_PDD_FOLDER, FRONT_FILM, REAR_FILM)