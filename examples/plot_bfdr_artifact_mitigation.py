#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:21:39 2026

@author: polanekr
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import sobel, label, uniform_filter
from scipy.interpolate import griddata
from sklearn.mixture import GaussianMixture
import tifffile
import os

def find_best_roi(grad_map, roi_size=400):
    """
    Automatikusan megkeresi a képen a 'legizgalmasabb' (legnagyobb gradiensű) részt.
    """
    print(f"  -> Automatikus fókuszpont keresése ({roi_size}x{roi_size} pixel)...")
    density = uniform_filter(grad_map, size=roi_size)
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    cy, cx = max_idx[0], max_idx[1]
    
    h, w = grad_map.shape
    half = roi_size // 2
    
    y1 = max(0, cy - half); y2 = min(h, cy + half)
    x1 = max(0, cx - half); x2 = min(w, cx + half)
    
    return x1, x2, y1, y2

# JAVÍTVA: Visszakerült a zoom_coords paraméter!
def generate_bfdr_figure(image_path, fdr_threshold=0.99, max_defect_size=500, auto_zoom=False, zoom_coords=None):
    """
    Legenerálja a publikációkész 3-paneles bFDR ábrát nyers TIFF-ből.
    """
    print("1. Kép betöltése...")
    img = tifffile.imread(image_path).astype(np.float64)
    
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
        
    print("2. Pseudo-OD (Optikai Sűrűség) konverzió...")
    img_safe = np.clip(img, 1.0, None)
    bg_ref = np.percentile(img_safe, 99.9) 
    od_img = np.log10(bg_ref / img_safe)
    od_img[od_img < 0] = 0.0 

    if od_img.ndim == 3:
        ref_channel = od_img[:, :, 0]
    else:
        ref_channel = od_img

    print("3. Sobel gradiensek számítása...")
    gx = sobel(ref_channel, axis=0)
    gy = sobel(ref_channel, axis=1)
    grad_mag = np.hypot(gx, gy)

    # --- ZOOM LOGIKA (Auto vagy Manuális) ---
    x1, x2, y1, y2 = None, None, None, None
    
    if auto_zoom:
        x1, x2, y1, y2 = find_best_roi(grad_mag, roi_size=400)
        print(f"  -> Auto-Zoom beállítva: X({x1}-{x2}), Y({y1}-{y2})")
    elif zoom_coords is not None:
        x1, x2, y1, y2 = zoom_coords
        print(f"  -> Manuális Zoom beállítva: X({x1}-{x2}), Y({y1}-{y2})")

    # Ha van valamilyen zoom, kivágjuk a képeket
    if x1 is not None:
        if od_img.ndim == 3:
            od_img = od_img[y1:y2, x1:x2, :]
        else:
            od_img = od_img[y1:y2, x1:x2]
        grad_mag = grad_mag[y1:y2, x1:x2]
        ref_channel = ref_channel[y1:y2, x1:x2]

    print("4. GMM (Gaussian Mixture Model) illesztése...")
    data = grad_mag.flatten().reshape(-1, 1)
    if len(data) > 100000:
        sample_data = np.random.choice(data.flatten(), 100000).reshape(-1, 1)
    else:
        sample_data = data
        
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(sample_data)
    
    means = gmm.means_.flatten()
    noise_idx = np.argmax(means) 
    
    probs = gmm.predict_proba(data)
    prob_noise = probs[:, noise_idx].reshape(ref_channel.shape)
    mask = prob_noise > fdr_threshold
    
    labeled_mask, num_features = label(mask)
    component_sizes = np.bincount(labeled_mask.ravel())
    too_big_labels = np.where(component_sizes > max_defect_size)[0]
    too_big_labels = too_big_labels[too_big_labels > 0]
    
    if len(too_big_labels) > 0:
        penumbra_pixels = np.isin(labeled_mask, too_big_labels)
        mask[penumbra_pixels] = False

    print(f"  -> Talált hibás pixelek a kivágaton: {np.sum(mask)} db")

    print("5. Inpainting (Tisztítás)...")
    cleaned_img = od_img.copy()
    if np.sum(mask) > 0:
        coords_valid = np.array(np.nonzero(~mask)).T
        if cleaned_img.ndim == 3:
            for i in range(3):
                channel = cleaned_img[:, :, i]
                values_valid = channel[~mask]
                grid_x, grid_y = np.nonzero(mask)
                interp_vals = griddata(coords_valid, values_valid, (grid_x, grid_y), method='nearest')
                channel[mask] = interp_vals
                cleaned_img[:, :, i] = channel
        else:
            values_valid = cleaned_img[~mask]
            grid_x, grid_y = np.nonzero(mask)
            interp_vals = griddata(coords_valid, values_valid, (grid_x, grid_y), method='nearest')
            cleaned_img[mask] = interp_vals

    print("6. Publikációkész Ábra Generálása...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def normalize_for_plot(im_array):
        if im_array.ndim == 3:
            im_norm = im_array - np.min(im_array)
            return im_norm / max(np.max(im_norm), 1e-6)
        return im_array

    img_plot = normalize_for_plot(od_img)
    cleaned_plot = normalize_for_plot(cleaned_img)
    cmap_2d = 'gray' if od_img.ndim == 2 else None

    # Panel A
    axes[0].imshow(img_plot, cmap=cmap_2d, origin='lower')
    axes[0].set_title("A. Original Optical Density", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel B
    axes[1].imshow(grad_mag, cmap='viridis', origin='lower')
    red_cmap = ListedColormap(['none', 'red'])
    axes[1].imshow(mask, cmap=red_cmap, alpha=0.8, origin='lower')
    axes[1].set_title("B. Gradient Map & bFDR Mask", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel C
    axes[2].imshow(cleaned_plot, cmap=cmap_2d, origin='lower')
    axes[2].set_title("C. Cleaned Image (Inpainting)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Bilateral False Discovery Rate (bFDR) Artifact Mitigation", fontsize=16, y=1.02)
    plt.tight_layout()
    
    filename = "fig1_bfdr_artifact_mitigation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Ábra elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    my_tiff_path = "/home/polanekr/Kutatás/PRISM-Framework/examples/F_024.tif" # <--- Cseréld ki a valósra!
    
    # A leolvasott koordinátáid (x_min, x_max, y_min, y_max)x
    my_zoom = (75, 150, 75, 125) 
    
    # Most már hibátlanul átveszi a zoom_coords paramétert!
    generate_bfdr_figure(my_tiff_path, fdr_threshold=0.95,      # 99% helyett elég, ha 85%-ig biztos benne
                         max_defect_size=2000,
                         auto_zoom=False)
    
# %%
import tifffile
import matplotlib.pyplot as plt

# Írd be a fájlodat
img = tifffile.imread("/home/polanekr/Kutatás/PRISM-Framework/examples/F_024.tif")

plt.figure(figsize=(10, 8))
# Ha RGB, mutassuk csak a piros csatornát, hogy jobban látszódjon a dózis
plt.imshow(img[:,:,0] if img.ndim==3 else img, cmap='gray')
plt.title("Keresd meg a porszemet! (Olvasd le az X és Y tengelyt)")
plt.colorbar()
plt.show()
