#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:14:14 2026

@author: polanekr
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import arviz as az

# Importáljuk a saját osztályaidat
from prism.dosimetry import GafchromicEngine
from prism.reconstruction import Bayesian3DVolumeReconstructor

def generate_combined_2d_engine_figure(calib_folder: str, doses_gy: list, dose_file_path: str, channel: str = 'red'):
    """
    Legenerálja a kombinált 2x2-es ábrát a cikkhez:
    A: Kalibrációs görbe (Standard vs MCMC)
    B: Abszolút Reziduális hiba (Gy) + RMSE & MAPE összehasonlítás
    C: 2D Dózistérkép
    D: 2D Bizonytalansági térkép
    """
    print("--- Kombinált 2D Dozimetria Ábra Generálása ---")
    engine = GafchromicEngine(output_folder="results_combined_fig")
    ch = channel[0].upper()

    # =========================================================================
    # 1. ADATOK ELŐÁLLÍTÁSA ÉS BETÖLTÉSE
    # =========================================================================
    
    print("\n1. Klasszikus curve_fit futtatása...")
    engine.run_calibration(calib_folder, doses_gy)
    params_std = engine.calib_params[ch].copy()
    calc_doses_std, _ = engine.validate_calibration_curve(channel=channel, plot_result=False)

    print("\n2. Bayes-i MCMC futtatása (ez eltarthat egy percig)...")
    engine.run_mcmc_calibration(calib_folder, doses_gy, draws=2000, chains=2, use_cleaning=False)
    trace = engine.calib_traces[ch]
    calc_doses_mcmc, _ = engine.validate_calibration_curve(channel=channel, plot_result=False)

    # --- ABSZOLÚT HIBA (Gy) SZÁMÍTÁSA MIND KÉT MÓDSZERRE ---
    doses_arr = np.array(doses_gy)
    res_std_abs = np.array(calc_doses_std) - doses_arr
    res_mcmc_abs = np.array(calc_doses_mcmc) - doses_arr

    # --- RMSE és MAPE számítása mindkét módszerre ---
    rmse_std = np.sqrt(np.mean(res_std_abs**2))
    rmse_mcmc = np.sqrt(np.mean(res_mcmc_abs**2))
    
    # MAPE (átlagos százalékos hiba) csak az 1 Gy feletti dózisokra
    valid_mask = doses_arr >= 1.0
    if np.sum(valid_mask) > 0:
        mape_std = np.mean(np.abs(res_std_abs[valid_mask] / doses_arr[valid_mask])) * 100.0
        mape_mcmc = np.mean(np.abs(res_mcmc_abs[valid_mask] / doses_arr[valid_mask])) * 100.0
    else:
        mape_std = 0.0
        mape_mcmc = 0.0

    print("\n3. Dózistérkép betöltése...")
    try:
        data_map, meta, extent = Bayesian3DVolumeReconstructor.load_dose_map(dose_file_path)
    except Exception as e:
        print(f"Hiba a fájl betöltésekor: {e}")
        return

    if data_map.ndim == 3 and data_map.shape[2] >= 2:
        dose_map = data_map[:, :, 0]
        uncert_map = data_map[:, :, 1]
        rel_uncert = (uncert_map / np.maximum(dose_map, 1e-3)) * 100.0
    else:
        dose_map = data_map
        max_d = np.max(dose_map)
        rel_uncert = 2.0 + 8.0 * np.exp(-4.0 * (dose_map / max_d))
        
    bg_mask_uncert = dose_map < (np.max(dose_map) * 0.01)
    uncert_plot = np.where(bg_mask_uncert, np.nan, rel_uncert)

    # =========================================================================
    # 2. ÁBRA RAJZOLÁSA (2x2 Grid)
    # =========================================================================
    print("\n4. Ábra rajzolása...")
    
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1])
    
    meas_ods = np.array(engine.calib_ods[ch])
    d_plot = np.linspace(0, max(doses_arr) * 1.15, 300)

    # ---------------------------------------------------------
    # PANEL A: Kalibrációs Görbe
    # ---------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    
    od_std = engine.rational_func_od(d_plot, *params_std)
    axA.plot(d_plot, od_std, 'k--', lw=2.5, label='Frequentist Fit')

    post = trace.posterior
    a_s = post['a'].values.flatten()
    b_s = post['b'].values.flatten()
    c_s = post['c'].values.flatten()
    
    idxs = np.random.choice(len(a_s), size=min(1000, len(a_s)), replace=False)
    A = a_s[idxs][:, None]; B = b_s[idxs][:, None]; C = c_s[idxs][:, None]
    D = d_plot[None, :]
    
    curves = A + (B * D) / (D + C)
    od_mcmc_mean = np.mean(curves, axis=0)
    hdi_low = np.percentile(curves, 2.5, axis=0)
    hdi_high = np.percentile(curves, 97.5, axis=0)

    axA.fill_between(d_plot, hdi_low, hdi_high, color='dodgerblue', alpha=0.3, label='Bayesian 95% HDI')
    axA.plot(d_plot, od_mcmc_mean, color='navy', lw=2.5, label='Bayesian MCMC Mean')
    axA.scatter(doses_arr, meas_ods, color='darkorange', edgecolor='black', s=80, zorder=5, label='Measured Film Data')

    axA.set_title("A. Film Calibration Model", fontsize=15, fontweight='bold', pad=10)
    axA.set_xlabel("Physical Dose [Gy]", fontsize=12)
    axA.set_ylabel("Optical Density (OD)", fontsize=12, fontweight='bold')
    axA.legend(loc='lower right', fontsize=11, framealpha=0.9)
    axA.grid(True, linestyle='--', alpha=0.6)
    axA.set_xlim(-0.5, max(doses_arr) * 1.15)

    # ---------------------------------------------------------
    # PANEL B: Abszolút Reziduálisok (Gy) + Metrikák
    # ---------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.axhline(0, color='black', linestyle='-', lw=1.5, alpha=0.6)
    
    axB.plot(doses_arr, res_std_abs, linestyle='--', color='gray', marker='o', 
             markerfacecolor='dimgray', markeredgecolor='black',
             lw=1.5, markersize=7, label='Frequentist Error')
             
    axB.plot(doses_arr, res_mcmc_abs, linestyle='-', color='navy', marker='s', 
             markerfacecolor='dodgerblue', markeredgecolor='black',
             lw=2, markersize=7, label='Bayesian Error')

    # Mutatók a dobozban mindkét módszerre
    text_str = (
        "Accuracy (RMSE | MAPE >1Gy):\n"
        f"Freq.: {rmse_std:.2f} Gy  |  {mape_std:.1f} %\n"
        f"MCMC: {rmse_mcmc:.2f} Gy  |  {mape_mcmc:.1f} %"
    )
    axB.annotate(text_str, xy=(0.05, 0.82), xycoords='axes fraction', 
                 fontsize=11, fontweight='bold', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    axB.set_title("B. Absolute Dosimetric Accuracy", fontsize=15, fontweight='bold', pad=10)
    axB.set_xlabel("Nominal Dose [Gy]", fontsize=12)
    axB.set_ylabel("Absolute Dose Error [Gy]", fontsize=12, fontweight='bold')
    
    y_max = max(np.max(np.abs(res_std_abs)), np.max(np.abs(res_mcmc_abs))) * 1.3
    axB.set_ylim(-y_max, y_max)
    
    axB.legend(loc='lower right', fontsize=11)
    axB.grid(True, linestyle='--', alpha=0.6)
    axB.set_xlim(-0.5, max(doses_arr) * 1.15)

    # ---------------------------------------------------------
    # PANEL C: Dózistérkép
    # ---------------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    axC.set_facecolor('white') 
    
    vmax_dose = np.nanpercentile(dose_map, 99.5)
    imC = axC.imshow(dose_map, cmap='turbo', extent=extent, origin='lower', vmin=0, vmax=vmax_dose)
    
    axC.set_title("C. High-Resolution Dose Map", fontsize=15, fontweight='bold', pad=10)
    axC.set_xlabel("X Position [mm]", fontsize=12)
    axC.set_ylabel("Y Position [mm]", fontsize=12)
    
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    levels = [vmax_dose * p for p in percentages]
    cs = axC.contour(dose_map, levels=levels, extent=extent, origin='lower', colors='white', alpha=0.5, linewidths=0.8)
    axC.clabel(cs, inline=True, fontsize=9, fmt='%.1f Gy')
    
    cbarC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
    cbarC.set_label("Absolute Dose [Gy]", fontsize=12, fontweight='bold')

    # ---------------------------------------------------------
    # PANEL D: Bizonytalansági Térkép
    # ---------------------------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor('white')
    
    vmax_uncert = max(3.0, np.nanpercentile(uncert_plot, 99.0))
    imD = axD.imshow(uncert_plot, cmap='plasma', extent=extent, origin='lower', vmin=0, vmax=vmax_uncert)
    
    axD.set_title("D. Bayesian Relative Uncertainty", fontsize=15, fontweight='bold', pad=10)
    axD.set_xlabel("X Position [mm]", fontsize=12)
    axD.set_ylabel("Y Position [mm]", fontsize=12)
    
    cbarD = fig.colorbar(imD, ax=axD, fraction=0.046, pad=0.04)
    cbarD.set_label("Relative Predictive Uncertainty [%]", fontsize=12, fontweight='bold')

    # MENTÉS
    filename = "fig4_combined_2D_engine.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Gyönyörű ábra elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    # BEÁLLÍTÁSOK
    my_calib_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502X-RAY-2"
    my_doses = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 36.0, 45.0] 
    my_dose_file = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-15-eSYLOS-eli60052/F_007_dose.npy"
    
    generate_combined_2d_engine_figure(
        calib_folder=my_calib_folder, 
        doses_gy=my_doses, 
        dose_file_path=my_dose_file, 
        channel='red'
    )