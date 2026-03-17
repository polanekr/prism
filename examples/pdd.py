#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 07:49:31 2026

@author: polanekr
"""

# %% Imports

from prism.dosimetry import GafchromicEngine
from prism.reconstruction import Bayesian3DVolumeReconstructor
from prism.analytics import DoseAnalyst
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

raw_films = '/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/2025-11-27-eSYLOS-eli60052/PDD2'
processed_films = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-11-27-eSYLOS-eli60052"
depths = [0.2, 2.2, 7.0, 11.83, 16.66, 21.46, 26.26, 33.26, 40.26, 52.26, 64.26, 76.26, 88.26, 104.76, 121.26, 137.76, 154.26, 170.76, 187.26, 203.72, 220.24 , 236.94]
calib_ebt_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502/long_axis/calibration_master.json"
calib_ebtxd_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/06282401XD/long-axis/calibration_master.json"
blanckxd = '/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/06282401XD/long-axis/1-000Gy-1.tif'
blanck = '/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502/long_axis/000Gy.tif'

# %% Convert raw films to dosemaps

engine = GafchromicEngine(processed_films+'/PDD2', batch_metadata={"depth_mm":depths})

engine.load_calibration(calib_ebt_folder)
# %% Process films

files = sorted(glob.glob(os.path.join(raw_films, "*.tif")))
files_to_process = [f for f in files if f != blanckxd]
print(f"Talált PDD fájlok: {len(files)} db")
print(f"Ebből feldolgozandó: {len(files_to_process)} db")

engine.process_films(file_list=files_to_process, blank_path=blanckxd)

# %%
film = np.load(processed_films+'/PDD2/F_029_dose.npy')
plt.imshow(film)
print(np.mean(film[95-5:95+5, 96-5:96+5]))

# %% reconstruction

recon = Bayesian3DVolumeReconstructor(pdd_folder=processed_films+'/PDD2', ssd_mm=1200, energy_mev=40, energy_std_mev=10)
recon.build_volume(beam_type='electron', divergence_correction=True)

# %% Analysis
vol = recon.volume
z_coords = recon.coords['z']
x_coords = recon.coords['x']
Z_dim, Y_dim, X_dim = vol.shape
cy, cx = Y_dim // 2, X_dim // 2
roi_size_px=5
# Kivágunk egy kis ROI-t a középpont körül és átlagoljuk (sima 1D PDD görbe)
half_roi = roi_size_px // 2
central_roi = vol[:, cy-half_roi:cy+half_roi+1, cx-half_roi:cx+half_roi+1]
pdd_raw = np.mean(central_roi, axis=(1, 2))

analysis = DoseAnalyst()

pdd_metrics, (z_dense, d_dense) = analysis.analyze_pdd(z_coords, pdd_raw, normalize=True)

print(f"  d_max mélység: {pdd_metrics['d_max_mm']:.2f} mm")
print(f"  R50 (50%-os mélység): {pdd_metrics['R50_mm']:.2f} mm")
print(f"  Rp (Gyakorlati hatótáv): {pdd_metrics['Rp_mm']:.2f} mm")
if not np.isnan(pdd_metrics['E0_MeV']):
    print(f"  Becsült energia (E0): {pdd_metrics['E0_MeV']:.2f} MeV")

# PDD Ábrázolása
plt.figure(figsize=(10, 6))
plt.plot(z_coords, (pdd_raw / np.max(pdd_raw)) *100, 'ko', label='Raw Data', alpha=0.5)
plt.plot(z_dense, d_dense, 'b-', lw=2, label='Smoothed PDD')

# Metrikák berajzolása
plt.axvline(pdd_metrics['d_max_mm'], color='r', linestyle='--', label='d_max')
if not np.isnan(pdd_metrics['R50_mm']):
    plt.axvline(pdd_metrics['R50_mm'], color='g',
                    linestyle='-.', label='R50')

plt.title("Central Axis PDD with Metrics", fontsize=14, fontweight='bold')
plt.xlabel("Depth in Phantom (Z) [mm]", fontsize=12)
plt.ylabel("Relative Dose [%]", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.4)
plt.show()

# A nyers (átlagolt) 1D profil maximuma:
max_dose_pdd = np.max(pdd_raw)
print(f"A centrális tengelyen mért maximális dózis: {max_dose_pdd:.2e} Gy")

# A simított (szépített) görbe maximuma:
max_dose_smoothed = np.max(d_dense)
print(f"A simított görbe maximuma: {max_dose_smoothed:.2f} Gy")

# És hogy ez milyen mélyen volt (ezt a DoseAnalyst kiszámolta):
print(f"A maximum helye (d_max): {pdd_metrics['d_max_mm']:.2f} mm")

# %%
m = 0.0254 * 0.0254 * 0.02 * 1.19 #cm x cm x cm x g/cm3
joule = vol * m / 1000
print(np.sum(joule))