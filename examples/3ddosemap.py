#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:35:41 2026

@author: polanekr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Importáljuk a saját osztályodat
from prism.reconstruction import Bayesian3DVolumeReconstructor

def generate_global_beam_figure(pdd_folder):
    print("--- 6. Ábra: Globális 3D Nyaláb Ortogonális Metszetei ---")
    
    # 1. Térfogat felépítése
    recon = Bayesian3DVolumeReconstructor(pdd_folder=pdd_folder, ssd_mm=300.0)
    recon.build_volume(roi_size_mm=60.0, z_res_mm=0.5, perform_alignment=True, divergence_correction=True)
    
    vol = recon.volume
    coords = recon.coords
    
    # 2. Metszetek kiválasztása (A nyaláb legforróbb pontja és a geometria közepe)
    # Keresünk egy mélységet a Dmax környékén (pl. 15 mm)
    z_target = 15.0
    iz = np.argmin(np.abs(coords['z'] - z_target))
    real_z = coords['z'][iz]
    
    # A nyaláb közepe (X=0, Y=0 környéke)
    ix = np.argmin(np.abs(coords['x'] - 0.0))
    iy = np.argmin(np.abs(coords['y'] - 0.0))
    
    # Határok az ábrázoláshoz
    x_ext = [coords['x'][0], coords['x'][-1]]
    y_ext = [coords['y'][0], coords['y'][-1]]
    z_ext = [coords['z'][-1], coords['z'][0]] # Fentről lefelé nő
    
    vmin, vmax = 0, np.max(vol)
    
    # 3. Ábra rajzolása (PMB stílus, 1x3 panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    
    # A) Axial (XY)
    im0 = axes[0].imshow(vol[iz, :, :], cmap='magma', origin='lower',
                         extent=[x_ext[0], x_ext[1], y_ext[0], y_ext[1]], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"A. Axial Plane (Z = {real_z:.1f} mm)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X Position [mm]", fontsize=12)
    axes[0].set_ylabel("Y Position [mm]", fontsize=12)
    axes[0].axvline(0, color='white', linestyle='--', alpha=0.4, lw=1)
    axes[0].axhline(0, color='white', linestyle='--', alpha=0.4, lw=1)
    
    # B) Coronal (XZ)
    axes[1].imshow(vol[:, iy, :], cmap='magma', origin='upper',
                   extent=[x_ext[0], x_ext[1], z_ext[0], z_ext[1]], aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"B. Coronal Plane (Y = 0.0 mm)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X Position [mm]", fontsize=12)
    axes[1].set_ylabel("Depth Z [mm]", fontsize=12)
    axes[1].axvline(0, color='white', linestyle='--', alpha=0.4, lw=1)
    axes[1].axhline(real_z, color='white', linestyle='--', alpha=0.4, lw=1)
    
    # C) Sagittal (YZ)
    axes[2].imshow(vol[:, :, ix], cmap='magma', origin='upper',
                   extent=[y_ext[0], y_ext[1], z_ext[0], z_ext[1]], aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"C. Sagittal Plane (X = 0.0 mm)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Y Position [mm]", fontsize=12)
    axes[2].set_ylabel("Depth Z [mm]", fontsize=12)
    axes[2].axvline(0, color='white', linestyle='--', alpha=0.4, lw=1)
    axes[2].axhline(real_z, color='white', linestyle='--', alpha=0.4, lw=1)
    
    # Közös Colorbar
    cbar = fig.colorbar(im0, ax=axes, location='right', aspect=30, pad=0.02)
    cbar.set_label("Absolute Reconstructed Dose [Gy]", fontsize=12, fontweight='bold')
    
    for ax in axes:
        ax.tick_params(axis='both', labelsize=10)
        
    filename = "fig6_global_3D_beam.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    # Írd be a saját PDD mappádat!
    MY_PDD = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/PDD1/"
    generate_global_beam_figure(MY_PDD)