#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:38:40 2026

@author: polanekr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from prism.reconstruction import Bayesian3DVolumeReconstructor

def generate_alignment_figure(pdd_folder, film_path):
    print("--- 7. Ábra: Alignment és Maszk Vetület ---")
    
    # 1. Térfogat és Film betöltése
    recon = Bayesian3DVolumeReconstructor(pdd_folder=pdd_folder, ssd_mm=300.0)
    recon.build_volume(roi_size_mm=50.0, z_res_mm=0.5, perform_alignment=False)
    
    film_map, meta, extent = recon.load_dose_map(film_path)
    
    # 2. Célpont (Eppendorf cső) létrehozása
    mask_name = "Eppendorf_Alignment_Test"
    recon.add_eppendorf_mask(
        name=mask_name, 
        volume_ml=2.0, 
        tip_position=(2.6, 7.6, 22.0), # Egy tipikus off-axis pozíció
        angle_from_x_deg=-75.0, 
        filled_height_mm=38.0
    )
    
    mask_3d = recon.roi_masks[mask_name]
    
    # A 3D maszk levetítése 2D-be (Z tengely mentén)
    mask_proj = np.max(mask_3d, axis=0)
    
    # A modell fizikai koordinátái (hogy rá tudjuk rajzolni a filmre)
    xs, ys = recon.coords['x'], recon.coords['y']
    
    # 3. Rajzolás
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    # B PANEL: Nyers Dózistérkép (Gafchromic)
    ax1 = axes[0]
    im1 = ax1.imshow(film_map, cmap='turbo', extent=extent, origin='lower')
    ax1.set_title("A. Raw Gafchromic Dose Map", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X Position [mm]", fontsize=12)
    ax1.set_ylabel("Y Position [mm]", fontsize=12)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Dose [Gy]")
    
    # C PANEL: Maszk Vetület (Alignment)
    ax2 = axes[1]
    im2 = ax2.imshow(film_map, cmap='turbo', extent=extent, origin='lower', alpha=0.8)
    
    # Kontúr rárajzolása
    ax2.contour(xs, ys, mask_proj, levels=[0.5], colors='white', linewidths=2.5, linestyles='--')
    
    # Egy kis annotáció, ami mutatja a csövet
    ax2.annotate('Virtual Tube \nProjection', xy=(2.6, 7.6), xytext=(-15, 26),
                 textcoords='offset points', color='white', fontweight='bold', fontsize=11,
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=6))
    
    ax2.set_title("B. Spatial Registration & Virtual Mask", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X Position [mm]", fontsize=12)
    ax2.set_ylabel("Y Position [mm]", fontsize=12)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Dose [Gy]")
    
    filename = "fig6_masking.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    # Írd be a PDD mappát és EGY KIVÁLASZTOTT FBX FRONT FILMET!
    MY_PDD = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/PDD1/"
    FRONT_FILM = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/F_008_dose.npy"
    
    generate_alignment_figure(MY_PDD, FRONT_FILM)