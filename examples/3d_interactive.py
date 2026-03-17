#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 20:04:58 2026

@author: polanekr
"""

import plotly.graph_objects as go
import numpy as np
import os

def plot_3d_interactive(volume, z_coords, y_coords, x_coords, level=25, opacity=0.4, save_html=True, max_voxels_per_axis=100):
    """
    Optimalizált 3D megjelenítő.
    - Automatikus downsampling (hogy ne fagyjon le a böngésző).
    - Helyes Aspect Ratio.
    """
    
    # 1. Adatellenőrzés
    if volume is None or np.max(volume) == 0:
        print("HIBA: Üres dózistérfogat.")
        return

    # 2. AUTOMATIKUS RITKÍTÁS (Downsampling)
    # Ha a tömb túl nagy (pl. > 100 voxel egy tengelyen), ritkítjuk.
    # Ez drasztikusan gyorsítja a megjelenítést, de az alak megmarad.
    nz, ny, nx = volume.shape
    max_dim = max(nz, ny, nx)
    
    step = 1
    if max_dim > max_voxels_per_axis:
        step = int(np.ceil(max_dim / max_voxels_per_axis))
        print(f"FIGYELEM: A 3D adat túl nagy ({nz}x{ny}x{nx}). Ritkítás: minden {step}. voxel.")
    
    # Szeletelés (Slicing) a step-pel
    vol_small = volume[::step, ::step, ::step]
    z_small = z_coords[::step]
    y_small = y_coords[::step]
    x_small = x_coords[::step]

    # 3. Adatok előkészítése a Plotly-nak
    max_d = np.max(volume) # Az eredeti max-ot használjuk referenciának
    isovalue = max_d * (level / 100.0)
    
    # Meshgrid 'ij' indexeléssel (Z, Y, X sorrendhez)
    Z, Y, X = np.meshgrid(z_small, y_small, x_small, indexing='ij')
    
    # 4. Figure Létrehozása
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_small.flatten(),
        isomin=isovalue,
        isomax=max_d,
        surface_count=5, # 5 réteg a level és a max között
        opacity=opacity,
        caps=dict(x_show=False, y_show=False), # Nyitott végek, hogy belelássunk
        colorscale='Jet',
        colorbar=dict(title='Dose [Gy]')
    ))
    
    # 5. Formázás (Aspect Ratio javítva!)
    fig.update_layout(
        title=f"3D Interactive Dose (>{level}%, Step={step})",
        width=1000, height=800, # Fix ablakméret segít
        scene=dict(
            xaxis=dict(title='X [mm]'),
            yaxis=dict(title='Y [mm]'),
            zaxis=dict(title='Depth Z [mm]', autorange='reversed'), # Z lefelé nő
            aspectmode='data' # <--- EZ A KULCS a torzulás ellen
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # 6. Megjelenítés / Mentés
    if save_html:
        filename = "3d_dose_view.html"
        try:
            fig.write_html(filename)
            print(f"3D Ábra mentve: {os.path.abspath(filename)}")
            
            # Próbáljuk megnyitni
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(filename))
            
        except Exception as e:
            print(f"Mentési hiba: {e}")
            # Ha a mentés nem megy, próbáljuk inline (Notebook esetén)
            fig.show()
    else:
        fig.show()

# --- TESZT ---
if __name__ == "__main__":
    data = np.load('/home/polanekr/Kutatás/PRISM-Framework/examples/results/films/PDD1.npz') # Vagy a te fájlneved
    vol = data['volume']
    
    # Koordináták kezelése (ha nincsenek elmentve, generáljuk)
    if 'z_coords' in data:
        z = data['z_coords']
        y = data['y_coords']
        x = data['x_coords']
    plot_3d_interactive(vol, z, y, x)