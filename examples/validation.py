#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:28:17 2026

@author: polanekr
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Importáljuk a saját PRISM osztályaidat
from prism.reconstruction import Bayesian3DVolumeReconstructor
from prism.analytics import DoseAnalyst

def extract_prism_dose(pdd_folder, front_film, mask_type, mask_params):
    """
    Lefuttatja a rekonstrukciót a helyes fizikai paraméterekkel és kinyeri a voxel eloszlást.
    """
    print(f"\n>>> Validációs Rekonstrukció Indítása: {pdd_folder} <<<")
    try:
        recon = Bayesian3DVolumeReconstructor(pdd_folder=pdd_folder, ssd_mm=300.0)
        
        # 1. 3D Térfogat felépítése - ALIGNMENT KIKAPCSOLVA (fizikai pozíció megtartása)
        recon.build_volume(roi_size_mm=50.0, z_res_mm=0.5, perform_alignment=False, divergence_correction=True)

        # 2. Tiszta skálázás a front fólia alapján (ELTOLÁS NÉLKÜL)
        if os.path.exists(front_film):
            f_map, _, _ = recon.load_dose_map(front_film)
            z_idx = np.argmin(np.abs(recon.coords['z'] - mask_params.get('front_z', 0.0)))
            vol_slice = recon.volume[z_idx]
            
            val_film = np.percentile(f_map, 99)
            val_vol = np.percentile(vol_slice, 99)
            scale_factor = val_film / val_vol if val_vol > 0 else 1.0
            
            recon.volume *= scale_factor
            recon.uncertainty *= scale_factor
            
        mask_name = "Validation_Mask"
        if mask_type == 'semiflex':
            recon._create_rotated_cylindrical_mask(
                name=mask_name, tip_position=mask_params['position'], 
                angle_deg=mask_params.get('angle', 0.0), radius=2.75, 
                tip_length=0.0, cylinder_length=6.5, tip_shape='none'
            )
        elif mask_type == 'fbx':
            recon.add_eppendorf_mask(
                name=mask_name, volume_ml=mask_params.get('volume_ml', 2.0), 
                tip_position=mask_params['position'], angle_from_x_deg=mask_params.get('angle', -75.0), 
                filled_height_mm=mask_params.get('fill_height', 38.0)
            )

        # 3. TELJES voxel eloszlás kinyerése a hegedű ábrához
        doses, uncerts = recon.get_voxel_data(mask_name)
        if doses is None or len(doses) == 0:
            print("  [!] HIBA: A maszk üres!")
            return None
            
        mean_dose = np.mean(doses)
        std_dose = np.std(doses) 

        # 4. PDD háttérgörbe kinyerése
        roi_px = int(2.0 / recon.pix_mm)
        ny, nx = recon.volume.shape[1:]
        max_slice = recon.volume[np.argmax(np.max(recon.volume, axis=(1,2)))]
        from scipy.ndimage import center_of_mass
        cy_f, cx_f = center_of_mass(max_slice > (np.max(max_slice)*0.5))
        cy, cx = int(cy_f), int(cx_f)
        
        y1, y2 = max(0, cy - roi_px), min(ny, cy + roi_px)
        x1, x2 = max(0, cx - roi_px), min(nx, cx + roi_px)
        
        pdd_raw = np.mean(recon.volume[:, y1:y2, x1:x2], axis=(1,2))
        pdd_unc = np.mean(recon.uncertainty[:, y1:y2, x1:x2], axis=(1,2))
        z_raw = recon.coords['z']
        
        metrics, (z_dense, pdd_smooth_pct) = DoseAnalyst.analyze_pdd(z_raw, pdd_raw, normalize=True)
        
        dmax_val = np.max(pdd_raw)
        pdd_unc_pct = (pdd_unc / dmax_val) * 100.0 if dmax_val > 0 else pdd_unc
        f_unc = interp1d(z_raw, pdd_unc_pct, kind='linear', fill_value="extrapolate", bounds_error=False)
        pdd_unc_dense_pct = f_unc(z_dense)
        
        print(f"  -> SIKER: PRISM Átlag = {mean_dose:.2f} Gy")
        
        return {
            'mean': mean_dose,
            'std': std_dose,
            'raw_doses': doses, 
            'pdd_z': z_dense,
            'pdd_mean_pct': pdd_smooth_pct,
            'pdd_std_pct': pdd_unc_dense_pct
        }
        
    except Exception as e:
        print(f"  [!] Hiba történt: {e}")
        return None

def plot_validation_figure(user_data):
    print("\n=== Ábra Generálása ===")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # --- PANEL A: PDD ---
    ax = axes[0]
    pdd_data = user_data['pdd']
    meas_z = np.array(pdd_data['meas_z'])
    meas_d = np.array(pdd_data['meas_dose'])
    meas_d_pct = (meas_d / np.max(meas_d)) * 100.0
    
    if pdd_data.get('prism_z') is not None:
        pz = pdd_data['prism_z']
        p_mean = pdd_data['prism_mean_pct']
        p_std = pdd_data['prism_std_pct']
        ax.fill_between(pz, p_mean - p_std, p_mean + p_std, color='dodgerblue', alpha=0.3, label='PRISM ±1 SD')
        ax.plot(pz, p_mean, 'b-', lw=2, label='PRISM 3D Reconstruction')
        
    ax.plot(meas_z, meas_d_pct, 'ro', markersize=7, markeredgecolor='k', zorder=5, label='Ion Chamber Data')
    ax.set_title("A. Depth-Dose (PDD) Validation", fontsize=13, fontweight='bold')
    ax.set_xlabel("Depth in Phantom [mm]", fontsize=11)
    ax.set_ylabel("Relative Dose [%]", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')

    # --- VIOLIN PLOT SEGÉDFÜGGVÉNY (Függetlenített tengelyekkel) ---
    def plot_violin_validation(ax, data_list, title, color_face, color_edge):
        meas_vals, prism_means, all_doses = [], [], []
        can_draw_violin = True
        
        for item in data_list:
            if 'prism_mean' in item and item['prism_mean'] is not None:
                meas_vals.append(item['measured_dose'])
                prism_means.append(item['prism_mean'])
                
                if 'prism_raw_doses' in item and item['prism_raw_doses'] is not None:
                    d = item['prism_raw_doses']
                    if len(d) > 10000: d = np.random.choice(d, 10000, replace=False)
                    all_doses.append(d)
                else:
                    can_draw_violin = False
            
        meas_vals = np.array(meas_vals)
        prism_means = np.array(prism_means)
        
        # 1. TENGELYEK FÜGGETLEN SKÁLÁZÁSA
        x_max = np.max(meas_vals) * 1.4 if len(meas_vals) > 0 else 10.0
        
        if can_draw_violin and len(all_doses) > 0:
            global_max_dose = max([np.max(d) for d in all_doses])
            y_max = max(x_max, global_max_dose * 1.1)
        else:
            y_max = x_max
            
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        # Ideális egyenes rajzolása
        ax.plot([0, x_max], [0, x_max], 'k--', alpha=0.5, label='Ideal (Mean: y=x)')
        
        if can_draw_violin and len(all_doses) > 0:
            parts = ax.violinplot(all_doses, positions=meas_vals, widths=x_max*0.12, 
                                  showmeans=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color_face)
                pc.set_edgecolor(color_edge)
                pc.set_alpha(0.6)
            parts['cmeans'].set_color('red') 
            parts['cmins'].set_color(color_edge)
            parts['cmaxes'].set_color(color_edge)
            parts['cbars'].set_color(color_edge)
            
            ax.plot(meas_vals, prism_means, 'o', color='white', markeredgecolor='red', markersize=6, label='Volume Mean')
        elif len(meas_vals) > 0:
            ax.plot(meas_vals, prism_means, 'o', color=color_edge, markeredgecolor='black', markersize=8, label='Volume Mean')

        for i in range(len(meas_vals)):
            diff_pct = ((prism_means[i] - meas_vals[i]) / meas_vals[i]) * 100
            ax.annotate(f"{diff_pct:+.1f}%", (meas_vals[i] + x_max*0.08, prism_means[i]), 
                        ha='left', va='center', fontsize=10, fontweight='bold', color='red')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Measured Dose (Mean) [Gy]", fontsize=11)
        ax.set_ylabel("PRISM Voxel Dose Distribution [Gy]", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left')

    # --- PANEL B: SEMIFLEX ---
    plot_violin_validation(axes[1], user_data['semiflex'], "B. Point Dose Validation (Semiflex)", 'lightskyblue', 'dodgerblue')

    # --- PANEL C: FBX ---
    plot_violin_validation(axes[2], user_data['fbx'], "C. Volumetric Validation (FBX)", 'moccasin', 'darkorange')

    filename = "fig_physical_validation_final.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    USER_DATA = {
        'pdd': {
            'meas_z': [0.0, 4.5, 9.0, 15.5, 22.0, 35.0, 46.5, 58.0], 
            'meas_dose': [28.34, 28.54, 24.39, 21.08, 17.64, 13.29, 10.25, 7.8], 
            'prism_z': None,     
            'prism_mean_pct': None,
            'prism_std_pct': None
        },
        
        'semiflex': [
            {
                'measured_dose': 6.75, 
                'prism_mean': 0.0,    
                'prism_std': 0.0,
                'folder': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/PDD1',
                'front': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/F_003_dose.npy',
                'params': {
                    'position': (3.0,4.4,22), 
                    'angle': -75.0,                
                    'front_z': 14.5
                }
            },
            {
                'measured_dose': 12.59, 
                'prism_mean': 0.0, 
                'prism_std': 0.0,
                'folder': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/PDD1',
                'front': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/F_005_dose.npy',
                'params': {
                    'position': (3.0,1.4,22), 
                    'angle': -77.0,
                    'front_z': 14.5
                }
            }
        ],
        
        'fbx': [
            {
                'measured_dose': 2.17, 
                'prism_mean': 0.0, 
                'prism_std': 0.0,
                'folder': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/PDD1',
                'front': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/F_007_dose.npy',
                'params': {
                    'volume_ml': 2.0,
                    'position': (2.5,8.4,22), 
                    'angle': -75.0,
                    'fill_height': 28.0,
                    'front_z': 14.5
                }
            },
            {
                'measured_dose': 4.59, 
                'prism_mean': 0.0, 
                'prism_std': 0.0,
                'folder': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/PDD1',
                'front': '/home/polanekr/Kutatás/esylos-dosimetry/data/interim/2025-12-15-eSYLOS-eli60052/F_009_dose.npy',
                'params': {
                    'volume_ml': 2.0,
                    'position': (5.5,8.4,22),
                    'angle': -75.0,
                    'fill_height': 27.0,
                    'front_z': 14.5
                }
            }
        ]
    }

    RUN_RECONSTRUCTION = True 
    
    if RUN_RECONSTRUCTION:
        pdd_captured = False 
        
        for item in USER_DATA['semiflex']:
            if os.path.exists(item['folder']):
                res = extract_prism_dose(item['folder'], item['front'], 'semiflex', item['params'])
                if res:
                    item['prism_mean'] = res['mean']
                    item['prism_std'] = res['std']
                    item['prism_raw_doses'] = res['raw_doses'] 
                    if not pdd_captured:
                        USER_DATA['pdd']['prism_z'] = res['pdd_z']
                        USER_DATA['pdd']['prism_mean_pct'] = res['pdd_mean_pct']
                        USER_DATA['pdd']['prism_std_pct'] = res['pdd_std_pct']
                        pdd_captured = True
                        
        for item in USER_DATA['fbx']:
            if os.path.exists(item['folder']):
                res = extract_prism_dose(item['folder'], item['front'], 'fbx', item['params'])
                if res:
                    item['prism_mean'] = res['mean']
                    item['prism_std'] = res['std']
                    item['prism_raw_doses'] = res['raw_doses']
                    if not pdd_captured:
                        USER_DATA['pdd']['prism_z'] = res['pdd_z']
                        USER_DATA['pdd']['prism_mean_pct'] = res['pdd_mean_pct']
                        USER_DATA['pdd']['prism_std_pct'] = res['pdd_std_pct']
                        pdd_captured = True

    plot_validation_figure(USER_DATA)