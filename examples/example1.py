#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:40:41 2026

@author: polanekr
"""

# %% Imports
from prism.dosimetry import GafchromicEngine, compare_calibration_methods
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
# %% Calibration
cal_engine = GafchromicEngine(output_folder="/home/polanekr/Kutatás/PRISM-Framework/examples/results")
cal_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502X-RAY-2"
cal_doses = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 36.0, 45.0]
#cal_engine.run_mcmc_calibration(cal_folder, cal_doses)
compare_calibration_methods(cal_engine, cal_folder, cal_doses)
# %% Film processing
blanck = cal_folder + '/c_001.tif'
films_path = "/home/polanekr/Kutatás/PRISM-Framework/examples/results/films"
pdd1_path = "/home/polanekr/Kutatás/PRISM-Framework/examples/results/films/PDD1"
pdd2_path = "/home/polanekr/Kutatás/PRISM-Framework/examples/results/films/PDD2"
FBX_path = "/home/polanekr/Kutatás/PRISM-Framework/examples/results/films/FBX"

#%%
film_engine = GafchromicEngine(output_folder=films_path, batch_metadata={'depth_mm':2.2})
film_engine.load_calibration('/home/polanekr/Kutatás/PRISM-Framework/examples/results/calibration_master.json')

files = sorted(glob.glob(os.path.join('/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/2025-12-15-eSYLOS-eli60052', "*.tif")))
files_to_process = [f for f in files if f != blanck]
print(f"Talált PDD fájlok: {len(files)} db")
print(f"Ebből feldolgozandó: {len(files_to_process)} db")

film_engine.process_films(files_to_process, blanck, use_cleaning=True)

# %% Display
film = np.load(films_path+'/F_004_dose.npy')
plt.imshow(film)

# %%
depths = [0.0, 4.5, 9.0, 15.5, 22.0, 28.5, 35.0, 46.5, 58.0, 69.5, 81.0, 92.5]
pdd1_engine = GafchromicEngine(output_folder=pdd1_path, batch_metadata={'depth_mm':depths})
pdd1_engine.load_calibration('/home/polanekr/Kutatás/PRISM-Framework/examples/results/calibration_master.json')

files = sorted(glob.glob(os.path.join('/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/2025-12-15-eSYLOS-eli60052/PDD1', "*.tif")))
files_to_process = [f for f in files if f != blanck]
print(f"Talált PDD fájlok: {len(files)} db")
print(f"Ebből feldolgozandó: {len(files_to_process)} db")

pdd1_engine.process_films(files_to_process, blanck, use_cleaning=True)

# %%
depths = [0.0, 4.5, 9.0, 15.5, 22.0, 28.5, 35.0, 46.5, 58.0, 69.5, 81.0, 92.5]
pdd2_engine = GafchromicEngine(output_folder=pdd2_path, batch_metadata={'depth_mm':depths})
pdd2_engine.load_calibration('/home/polanekr/Kutatás/PRISM-Framework/examples/results/calibration_master.json')

files = sorted(glob.glob(os.path.join('/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/2025-12-15-eSYLOS-eli60052/PDD2', "*.tif")))
files_to_process = [f for f in files if f != blanck]
print(f"Talált PDD fájlok: {len(files)} db")
print(f"Ebből feldolgozandó: {len(files_to_process)} db")

pdd2_engine.process_films(files_to_process, blanck, use_cleaning=True)

# %%
fbx_engine = GafchromicEngine(output_folder=FBX_path, batch_metadata={'depth_mm':2.2})
fbx_engine.load_calibration('/home/polanekr/Kutatás/PRISM-Framework/examples/results/calibration_master.json')

files = sorted(glob.glob(os.path.join('/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/2025-12-15-eSYLOS-eli60052/FBX-cal', "*.tif")))
files_to_process = [f for f in files if f != blanck]
print(f"Talált PDD fájlok: {len(files)} db")
print(f"Ebből feldolgozandó: {len(files_to_process)} db")

fbx_engine.process_films(files_to_process, blanck, use_cleaning=True)

# %% Reconstruction
from prism.reconstruction import Bayesian3DVolumeReconstructor

engine = Bayesian3DVolumeReconstructor(pdd1_path, ssd_mm=300, energy_mev=0.25, energy_std_mev=0.01)

engine.build_volume(roi_size_mm=25.0, z_res_mm=0.25, beam_type='photon', divergence_correction=True, shape_threshold_ratio=0.15)

engine.plot_ortho_views(slice_coords=(0, 0, 15))

engine.save_volume(pdd1_path)

# %% Eppendorf tubes
ref_film = engine.load_dose_map(films_path + '/F_003_dose.npy')
rear_film = engine.load_dose_map(films_path + '/F_004_dose.npy')
engine.add_eppendorf_mask('Semiflex', volume_ml=2.0, tip_position=(2.7,5.4,20.6),
                          angle_from_x_deg=286, filled_height_mm=9, radius=3.5)
engine.debug_spectral_correction(rear_film, 31.2, 'Semiflex')

# %% Histograme
engine.reset_volume()
engine.register_and_scale(ref_film[0], front_film_depth_mm=14.5)
engine.correct_spectrum_with_cavity(
        rear_film_map=rear_film[0], 
        rear_film_depth_mm=31.2, 
        mask_name="Semiflex",
        gap_thickness_mm=12.0,   # Becsült effektív levegő vastagság
        gap_material='water'
    )
hist1 = engine.get_equivalent_uniform_dose('Semiflex')