#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:26:13 2026

@author: polanekr
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import os

def load_or_mock_real_dvhs(nominal_doses, n_bins=50):
    """
    Betölti a valós DVH fájlokat minden egyes dózisszintre.
    Ha a fájlok nem léteznek, egyedi (nem felskálázott!) mock DVH-t generál.
    
    Visszatér:
    - dose_matrix: (N_doses, N_bins) méretű mátrix a bin közepekkel.
    - weight_matrix: (N_doses, N_bins) méretű mátrix a voxelek súlyával (0-1).
    """
    N_doses = len(nominal_doses)
    dose_matrix = np.zeros((N_doses, n_bins))
    weight_matrix = np.zeros((N_doses, n_bins))
    
    for i, nom_d in enumerate(nominal_doses):
        # ---------------------------------------------------------
        # IDE JÖHET A VALÓS FÁJL BEOLVASÁSOD!
        # Példa:
        # file_path = f"results/dvh_measurements/{nom_d}Gy.npz"
        # if os.path.exists(file_path):
        #     data = np.load(file_path)
        #     dose_matrix[i, :] = data['dose_bins']
        #     weight_matrix[i, :] = data['weights']
        #     continue
        # ---------------------------------------------------------
        
        # Ha nincs fájl, szimuláljuk az adott mérést:
        if nom_d == 0:
            # Kontroll csoport (0 Gy)
            doses = np.zeros(1000)
        else:
            # Valósághűbb inhomogén eloszlás (pl. 5% szórás + 10% "cold spot" a cső szélén)
            doses = np.random.normal(nom_d, nom_d * 0.05, 8000)
            doses = np.append(doses, np.random.normal(nom_d * 0.85, nom_d * 0.08, 2000))
        
        hist, bin_edges = np.histogram(doses, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weights = hist / np.sum(hist)
        
        dose_matrix[i, :] = bin_centers
        weight_matrix[i, :] = weights
        
    return dose_matrix, weight_matrix

def run_real_dvh_recovery():
    print("1. Valós DVH mérések beolvasása és kísérlet szimulálása...")
    # --- GROUND TRUTH ---
    true_alpha = 0.30  # Gy^-1
    true_beta = 0.03   # Gy^-2
    
    nominal_doses = np.array([0, 1, 2, 4, 6, 8, 10])
    
    # 1. DVH Mátrixok létrehozása (Minden sor egy külön mérés/besugárzás)
    dose_matrix, weight_matrix = load_or_mock_real_dvhs(nominal_doses, n_bins=50)
    
    # 2. Igazi túlélés (S) kiszámítása a valós DVH alapján
    true_S_total = np.zeros(len(nominal_doses))
    for i in range(len(nominal_doses)):
        D_i = dose_matrix[i, :] # Az adott mérés dózis-binjei
        W_i = weight_matrix[i, :] # Az adott mérés súlyai
        
        # Súlyozott összeg: S = sum( w_i * exp(-a*D_i - b*D_i^2) )
        S_bins = np.exp(-true_alpha * D_i - true_beta * (D_i**2))
        true_S_total[i] = np.sum(W_i * S_bins)
        
    # 3. Kísérleti (biológiai) zaj hozzáadása
    np.random.seed(42)
    noise_std = 0.15 # 15% relatív hiba
    noisy_S = true_S_total * np.random.normal(1.0, noise_std, size=len(nominal_doses))
    noisy_S = np.clip(noisy_S, 0.0001, 1.0) # Vágás fizikai határok közé
    noisy_log_S = np.log(noisy_S)
    
    print("2. Bayes-i MCMC futtatása (Mátrix műveletekkel)...")
    # --- BAYES-I MODELL ---
    with pm.Model() as lq_model:
        # Priorok
        alpha = pm.TruncatedNormal('alpha', mu=0.2, sigma=0.5, lower=0.0)
        beta = pm.TruncatedNormal('beta', mu=0.05, sigma=0.1, lower=0.0)
        sigma = pm.HalfNormal('sigma', sigma=0.5)
        
        # Mátrixok átadása a PyTensor-nak
        # Shape: (N_doses, N_bins)
        D_mat = pt.as_tensor_variable(dose_matrix)
        W_mat = pt.as_tensor_variable(weight_matrix)
        
        # Kiszámoljuk a túlélést minden binre (Tensor művelet az egész mátrixon!)
        log_S_mat = -alpha * D_mat - beta * (D_mat**2)
        S_mat = pt.exp(log_S_mat)
        
        # Integrálás a térfogatra (Összeadás a sorok, azaz a bineken: axis=1)
        S_tot = pt.sum(S_mat * W_mat, axis=1)
        
        # Likelihood illesztése a mért adatokra (logS)
        mu_log_S = pt.log(S_tot)
        obs = pm.Normal('obs', mu=mu_log_S, sigma=sigma, observed=noisy_log_S)
        
        # MCMC Futtatása
        trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.98, progressbar=True)
    
    print("3. Ábra Generálása...")
    # --- ÁBRÁZOLÁS ---
    az.style.use("arviz-darkgrid")
    
    post = trace.posterior
    alpha_samples = post['alpha'].values.flatten()
    beta_samples = post['beta'].values.flatten()
    
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.1, wspace=0.1)
    
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)
    
    ax_marg_x.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_marg_y.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    for spine in ['top', 'right', 'left']: ax_marg_x.spines[spine].set_visible(False)
    for spine in ['top', 'right', 'bottom']: ax_marg_y.spines[spine].set_visible(False)
    ax_marg_x.grid(False); ax_marg_y.grid(False)

    az.plot_kde(alpha_samples, beta_samples, ax=ax_joint,
                contourf_kwargs={"cmap": "Blues", "alpha": 0.5},
                contour_kwargs={"colors": "blue", "linewidths": 1.5, "alpha": 0.8})
    
    ax_joint.plot(true_alpha, true_beta, 'ro', markersize=8, zorder=5)
    ax_joint.axvline(true_alpha, color='red', linestyle='--', alpha=0.7)
    ax_joint.axhline(true_beta, color='red', linestyle='--', alpha=0.7)
    
    kde_a = gaussian_kde(alpha_samples)
    x_a = np.linspace(alpha_samples.min(), alpha_samples.max(), 200)
    ax_marg_x.plot(x_a, kde_a(x_a), color='blue', lw=2)
    ax_marg_x.fill_between(x_a, 0, kde_a(x_a), alpha=0.3, color='blue')
    ax_marg_x.axvline(true_alpha, color='red', linestyle='--', alpha=0.7, lw=2)
    
    kde_b = gaussian_kde(beta_samples)
    y_b = np.linspace(beta_samples.min(), beta_samples.max(), 200)
    ax_marg_y.plot(kde_b(y_b), y_b, color='blue', lw=2)
    ax_marg_y.fill_betweenx(y_b, 0, kde_b(y_b), alpha=0.3, color='blue')
    ax_marg_y.axhline(true_beta, color='red', linestyle='--', alpha=0.7, lw=2)
    
    ax_joint.set_xlabel(r"Radiosensitivity $\alpha$ [Gy$^{-1}$]", fontsize=14, fontweight='bold')
    ax_joint.set_ylabel(r"Radiosensitivity $\beta$ [Gy$^{-2}$]", fontsize=14, fontweight='bold')
    ax_joint.tick_params(axis='both', which='major', labelsize=11)
    
    custom_lines = [Line2D([0], [0], color='red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='blue', lw=2, alpha=0.5)]
    ax_joint.legend(custom_lines, ['True Parameters (Ground Truth)', '95% Posterior Density'], 
                    loc='upper right', fontsize=11)

    fig.suptitle("DVH-Integrated Parameter Recovery (Real Data)", fontsize=16, y=0.95)
    
    filename = "real_dvh_parameter_recovery_LQ.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSiker! Az ábra elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    run_real_dvh_recovery()