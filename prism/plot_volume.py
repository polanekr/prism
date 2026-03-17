#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 19:39:40 2026

@author: polanekr
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_publication_ready_ortho(volume, z_coords, y_coords, x_coords, output_filename="dose_distribution.png"):
    """
    Publikációkész 3-paneles (Ortho) nézet generálása.
    JAVÍTVA: A contour origin illesztése az imshow-hoz.
    """
    
    # 1. Beállítások
    cmap = 'jet'
    contour_levels = [10, 50, 80, 90, 95]
    font_size = 12
    max_dose = np.max(volume)
    
    # 2. Szeletelés (Maximum helyén)
    max_idx = np.unravel_index(np.argmax(volume), volume.shape)
    iz, iy, ix = max_idx
    
    # Ha a széleken van a max, korrigáljuk középre
    if iz == 0 or iz == volume.shape[0]-1: iz = volume.shape[0] // 2
    if iy == 0 or iy == volume.shape[1]-1: iy = volume.shape[1] // 2
    if ix == 0 or ix == volume.shape[2]-1: ix = volume.shape[2] // 2

    print(f"Szeletelés: Z={z_coords[iz]:.1f}mm, Y={y_coords[iy]:.1f}mm, X={x_coords[ix]:.1f}mm")

    slice_axial = volume[iz, :, :]      # XY
    slice_coronal = volume[:, iy, :]    # XZ
    slice_sagittal = volume[:, :, ix]   # YZ
    
    # 3. Ábrázolás
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    
    # --- JAVÍTOTT Megjelenítő Segédfüggvény ---
    def show_slice(ax, data, x_axis, y_axis, title, xlabel, ylabel, invert_yaxis=False):
        # Extent: [left, right, bottom, top]
        # Figyelem: Ha invert_yaxis=True, akkor a top a kisebb érték (0), bottom a nagyobb (mélység)
        extent = [x_axis[0], x_axis[-1], y_axis[-1] if invert_yaxis else y_axis[0], y_axis[0] if invert_yaxis else y_axis[-1]]
        
        # Origin meghatározása
        origin = 'upper' if invert_yaxis else 'lower'
        
        # 1. Kép (Heatmap)
        im = ax.imshow(data, cmap=cmap, origin=origin, 
                       extent=extent, aspect='auto', vmin=0, vmax=max_dose)
        
        # 2. Kontúrok (Izodózis)
        data_rel = (data / max_dose) * 100.0
        
        # *** JAVÍTÁS ITT: origin paraméter átadása a contour-nak is! ***
        cs = ax.contour(data_rel, levels=contour_levels, extent=extent, origin=origin,
                        colors='white', linewidths=0.8, alpha=0.7)
        
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d%%')
        
        ax.set_title(title, fontsize=font_size, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Ha invertáltuk a tengelyt (Z), akkor az extent már beállította a határokat, 
        # de a matplotlib néha automatikusan növekvő sorrendbe rendezi a tengelyt.
        # Biztosítjuk, hogy a 0 legyen felül.
        if invert_yaxis:
            ax.set_ylim(y_axis[-1], y_axis[0]) # Bottom (Deep), Top (0)

        return im

    # --- Panel 1: Axial (XY) ---
    # Y tengely felfelé nő -> invert_yaxis=False
    show_slice(axes[0], slice_axial, x_coords, y_coords, 
               f"Axial (Z = {z_coords[iz]:.1f} mm)", "X [mm]", "Y [mm]", invert_yaxis=False)

    # --- Panel 2: Coronal (XZ) ---
    # Z tengely lefelé nő -> invert_yaxis=True
    show_slice(axes[1], slice_coronal, x_coords, z_coords, 
               f"Coronal (Y = {y_coords[iy]:.1f} mm)", "X [mm]", "Depth Z [mm]", invert_yaxis=True)

    # --- Panel 3: Sagittal (YZ) ---
    # Z tengely lefelé nő -> invert_yaxis=True
    im3 = show_slice(axes[2], slice_sagittal, y_coords, z_coords, 
                     f"Sagittal (X = {x_coords[ix]:.1f} mm)", "Y [mm]", "Depth Z [mm]", invert_yaxis=True)

    # Colorbar
    cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Absorbed Dose [Gy]", fontsize=font_size)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Javított ábra elmentve: {output_filename}")
    plt.show()

# --- HASZNÁLAT (Példa) ---
if __name__ == "__main__":
    # 1. Betöltés (Ha van fájlod)
    try:
        data = np.load('/home/polanekr/Kutatás/PRISM-Framework/examples/results/films/PDD1.npz') # Vagy a te fájlneved
        vol = data['volume']
        
        # Koordináták kezelése (ha nincsenek elmentve, generáljuk)
        if 'z_coords' in data:
            z = data['z_coords']
            y = data['y_coords']
            x = data['x_coords']
        else:
            # Ha csak a volume van meg, és tudjuk a felbontást (pl. 0.5 mm)
            pix_mm = 0.5
            nz, ny, nx = vol.shape
            z = np.arange(nz) * pix_mm
            y = np.arange(ny) * pix_mm - (ny * pix_mm / 2)
            x = np.arange(nx) * pix_mm - (nx * pix_mm / 2)
            
        # 2. Rajzolás
        plot_publication_ready_ortho(vol, z, y, x, "publication_figure.png")
        
    except FileNotFoundError:
        print("Nem találtam .npz fájlt, generálok egy teszt adatot...")
        # (Itt jönne a dummy generátor, de a te esetedben van fájl)
        