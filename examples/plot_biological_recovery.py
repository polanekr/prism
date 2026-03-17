#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:14:41 2026

@author: polanekr
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

def run_parameter_recovery():
    """
    Bayes-i Paraméter-visszanyerés az LQ modellhez.
    Generál egy profi Joint Plot-ot elforgatott marginálisokkal.
    """
    print("1. Szintetikus Adatok Generálása...")
    # --- GROUND TRUTH (Az "Igazi" paraméterek) ---
    true_alpha = 0.30  # Gy^-1
    true_beta = 0.03   # Gy^-2
    
    doses = np.array([0, 1, 2, 4, 6, 8, 10])
    
    true_log_S = -true_alpha * doses - true_beta * (doses**2)
    true_S = np.exp(true_log_S)
    
    np.random.seed(42)
    noise_std = 0.15 
    noisy_S = true_S * np.random.normal(1.0, noise_std, size=len(doses))
    noisy_S = np.clip(noisy_S, 0.001, 1.0)
    noisy_log_S = np.log(noisy_S)
    
    print("2. Bayes-i MCMC Mintavételezés (PyMC)...")
    # --- BAYES-I MODELL (PyMC) ---
    with pm.Model() as lq_model:
        alpha = pm.TruncatedNormal('alpha', mu=0.2, sigma=0.5, lower=0.0)
        beta = pm.TruncatedNormal('beta', mu=0.05, sigma=0.1, lower=0.0)
        sigma = pm.HalfNormal('sigma', sigma=0.5)
        
        mu_log_S = -alpha * doses - beta * (doses**2)
        obs = pm.Normal('obs', mu=mu_log_S, sigma=sigma, observed=noisy_log_S)
        
        trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.95, progressbar=False)
    
    print("3. Publikációkész Ábra Generálása (Joint Plot)...")
    # --- ÁBRÁZOLÁS (GridSpec Joint Plot) ---
    az.style.use("arviz-darkgrid")
    
    # Poszterior minták kinyerése
    post = trace.posterior
    alpha_samples = post['alpha'].values.flatten()
    beta_samples = post['beta'].values.flatten()
    
    # GridSpec beállítása (4x4-es rács, amiből a fő ábra 3x3-at kap)
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.1, wspace=0.1)
    
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)
    
    # Marginális tengelyek elrejtése
    ax_marg_x.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    ax_marg_y.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    for spine in ['top', 'right', 'left']: ax_marg_x.spines[spine].set_visible(False)
    for spine in ['top', 'right', 'bottom']: ax_marg_y.spines[spine].set_visible(False)
    ax_marg_x.grid(False); ax_marg_y.grid(False)

    # 1. KÖZÉPSŐ (2D) ÁBRA
    az.plot_kde(alpha_samples, beta_samples, ax=ax_joint,
                contourf_kwargs={"cmap": "Blues", "alpha": 0.5},
                contour_kwargs={"colors": "blue", "linewidths": 1.5, "alpha": 0.8})
    
    # Ground truth pont és vonalak
    ax_joint.plot(true_alpha, true_beta, 'ro', markersize=8, zorder=5)
    ax_joint.axvline(true_alpha, color='red', linestyle='--', alpha=0.7)
    ax_joint.axhline(true_beta, color='red', linestyle='--', alpha=0.7)
    
    # 2. FELSŐ (1D Alpha) ÁBRA
    kde_a = gaussian_kde(alpha_samples)
    x_a = np.linspace(alpha_samples.min(), alpha_samples.max(), 200)
    ax_marg_x.plot(x_a, kde_a(x_a), color='blue', lw=2)
    ax_marg_x.fill_between(x_a, 0, kde_a(x_a), alpha=0.3, color='blue')
    ax_marg_x.axvline(true_alpha, color='red', linestyle='--', alpha=0.7, lw=2)
    
    # 3. JOBB OLDALI (1D Beta) ÁBRA - 90 FOKKAL ELFORGATVA
    kde_b = gaussian_kde(beta_samples)
    y_b = np.linspace(beta_samples.min(), beta_samples.max(), 200)
    # Figyelem: Itt megcseréljük az X és Y tengelyt a rajzolásnál!
    ax_marg_y.plot(kde_b(y_b), y_b, color='blue', lw=2)
    ax_marg_y.fill_betweenx(y_b, 0, kde_b(y_b), alpha=0.3, color='blue')
    ax_marg_y.axhline(true_beta, color='red', linestyle='--', alpha=0.7, lw=2)
    
    # --- FORMATÁLÁS ---
    ax_joint.set_xlabel(r"Radiosensitivity $\alpha$ [Gy$^{-1}$]", fontsize=14, fontweight='bold')
    ax_joint.set_ylabel(r"Radiosensitivity $\beta$ [Gy$^{-2}$]", fontsize=14, fontweight='bold')
    ax_joint.tick_params(axis='both', which='major', labelsize=11)
    
    custom_lines = [Line2D([0], [0], color='red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='blue', lw=2, alpha=0.5)]
    ax_joint.legend(custom_lines, ['True Parameters (Ground Truth)', '95% Posterior Density'], 
                    loc='upper right', fontsize=11)

    fig.suptitle("Bayesian Parameter Recovery: Linear-Quadratic Model", fontsize=16, y=0.95)
    
    filename = "bayesian_parameter_recovery_LQ.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSiker! Az ábra elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    run_parameter_recovery()