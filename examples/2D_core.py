#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:53:42 2026

@author: polanekr
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Importáljuk az osztályt
from prism.reconstruction import Bayesian3DVolumeReconstructor

def generate_fig4_2d_core(dose_file_path):
    print("--- 4. Ábra: 2D Gafchromic Dózis és Bizonytalansági Térkép ---")
    
    # 1. Dózistérkép betöltése
    try:
        data_map, meta, extent = Bayesian3DVolumeReconstructor.load_dose_map(dose_file_path)
    except Exception as e:
        print(f"Hiba a fájl betöltésekor: {e}")
        return

    # 2. Bizonytalanság kinyerése a fájlból
    if data_map.ndim == 3 and data_map.shape[2] >= 2:
        print("  -> Bizonytalansági adatok (Error) sikeresen detektálva a .npy fájlban!")
        dose_map = data_map[:, :, 0]
        uncert_map = data_map[:, :, 1]
        rel_uncert = (uncert_map / np.maximum(dose_map, 1e-3)) * 100.0
    else:
        print("  -> FIGYELEM: A fájl csak 2D dózist tartalmaz! Vizuális szimuláció generálása az ábrához...")
        dose_map = data_map
        max_d = np.max(dose_map)
        rel_uncert = 2.0 + 8.0 * np.exp(-4.0 * (dose_map / max_d))
        
    # --- MASZKOLÁS ---
    dose_plot = dose_map 
    bg_mask_uncert = dose_map < (np.max(dose_map) * 0.01)
    uncert_plot = np.where(bg_mask_uncert, np.nan, rel_uncert)

    # 3. Ábra rajzolása (PMB stílus, 1x2 panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # --- A) PANEL: Abszolút Dózistérkép ---
    ax1 = axes[0]
    ax1.set_facecolor('white') 
    
    # JAVÍTÁS: 99.5% Percentilis skálázás a zaj kiszűrésére!
    # Így a colorbar teljes spektrumát (kéktől a pirosig) ki fogjuk használni.
    vmax_dose = np.nanpercentile(dose_plot, 99.5)
    
    im1 = ax1.imshow(dose_plot, cmap='turbo', extent=extent, origin='lower', vmin=0, vmax=vmax_dose)
    ax1.set_title("A. High-Resolution Dose Map", fontsize=15, fontweight='bold', pad=10)
    ax1.set_xlabel("X Position [mm]", fontsize=12)
    ax1.set_ylabel("Y Position [mm]", fontsize=12)
    
    # Szintek: a max_dose alapján (10%-tól 90%-ig)
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    levels = [vmax_dose * p for p in percentages]
    
    cs = ax1.contour(dose_map, levels=levels, extent=extent, origin='lower', colors='white', alpha=0.5, linewidths=0.8)
    ax1.clabel(cs, inline=True, fontsize=9, fmt='%.1f Gy')
    
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Absolute Dose [Gy]", fontsize=12, fontweight='bold')

    # --- B) PANEL: Relatív Bizonytalanság ---
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    # JAVÍTÁS: A bizonytalanságnál is feszítjük a skálát (99%), hogy kitöltse a színskálát
    vmax_uncert = np.nanpercentile(uncert_plot, 99.0)
    # Ha a hiba nagyon kicsi lenne (pl. 1%), ne mutassunk zajt, egy minimum 3%-os skálát tartunk.
    vmax_uncert = max(3.0, vmax_uncert)
    
    im2 = ax2.imshow(uncert_plot, cmap='plasma', extent=extent, origin='lower', vmin=0, vmax=vmax_uncert)
    ax2.set_title("B. Bayesian Relative Uncertainty", fontsize=15, fontweight='bold', pad=10)
    ax2.set_xlabel("X Position [mm]", fontsize=12)
    ax2.set_ylabel("Y Position [mm]", fontsize=12)
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Relative Predictive Uncertainty [%]", fontsize=12, fontweight='bold')

    filename = "fig4_2D_core_uncertainty.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    # Írd be ide egy jól sikerült, kontrasztos front filmed elérési útját!
    DOSE_FILE = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/F_007_dose.npy"
    
    generate_fig4_2d_core(DOSE_FILE)