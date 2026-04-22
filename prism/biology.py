"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: biology.py

This module contains the Bayesian biological modeling classes for:
1. Cell Survival (Linear-Quadratic model with Error-in-Variables).
2. Fish/Embryo Survival (Weibull model with Voxel-Level Integration).
3. Population Analysis (Group-based survival with EUD).
4. Model Comparison (RBE calculation, parameter difference analysis).

Author: [Your Name / PRISM Team]
License: MIT
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.optimize import brentq

class BaseBioModel:
    """
    Base class for all biological models.
    Handles data storage and PyMC model context.
    """
    def __init__(self, name="BioModel"):
        self.name = name
        self.experiments = [] # List for experimental data
        self.idata = None     # Inference Data (after sampling)
        self.model = None     # PyMC model object

    def add_experiment(self, dose_distribution, outcome, **covariates):
        """
        Add data for a single experiment.
        
        Args:
            dose_distribution: 1D numpy array (Gy), doses of voxels in the volume.
            outcome: Measured biological response (e.g., colony count, survivors).
            covariates: Additional data (e.g., n_seeded, days, etc.)
        """
        # Downsampling if too many voxels (for PyMC speed)
        # 10,000 voxels are sufficient for statistics.
        if len(dose_distribution) > 10000:
            dose_distribution = np.random.choice(dose_distribution, 10000, replace=False)
            
        data = {
            'doses': dose_distribution,
            'outcome': outcome
        }
        data.update(covariates)
        self.experiments.append(data)
        print(f"[{self.name}] Experiment added. (Voxels: {len(dose_distribution)}, Outcome: {outcome})")

    def build_model(self):
        """Must be implemented by child classes."""
        raise NotImplementedError("The build_model method must be implemented!")

    def sample(self, draws=1000, tune=1000, chains=2):
        if self.model is None:
            self.build_model()
            
        print(f"[{self.name}] Starting sampling ({draws} draws, {tune} tune)...")
        with self.model:
            # 1. Mintavétel
            self.idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.99)
            
            # 2. Log-Likelihood számítása (EZ AZ ÚJ LÉPÉS!)
            # Ez teszi lehetővé a LOO/WAIC összehasonlítást
            pm.compute_log_likelihood(self.idata)
            
        return self.idata

    def plot_trace(self):
        if self.idata:
            az.plot_trace(self.idata)
            plt.show()

    def summary(self):
        if self.idata:
            return az.summary(self.idata)
        return None
    
    def diagnose(self):
        if self.idata is None: return
        print("--- MCMC Diagnostics ---")
        # 1. Divergences
        div = self.idata.sample_stats.diverging.sum().item()
        print(f"Divergences: {div} (Should be 0)")
        
        # 2. R-hat summary
        s = az.summary(self.idata, var_names=["alpha", "beta", "pe"])
        max_rhat = s['r_hat'].max()
        print(f"Max R-hat: {max_rhat:.3f} (Should be < 1.01)")
        
        # 3. Rank Plot
        az.plot_rank(self.idata, var_names=["alpha", "beta"])
        plt.show()

# ==============================================================================
# 1. CELL MODEL: LINEAR-QUADRATIC (Colony Survival)
# ==============================================================================
class CellSurvivalLQModel(BaseBioModel):
    """
    Updated LQ Model (Cell Survival / Colony Formation).
    
    Features:
    - Input: Histogram (dDVH) for accurate dose inhomogeneity handling.
    - Model: Linear-Quadratic (LQ) -> S = exp(-alpha*D - beta*D^2)
    - Likelihood: NegativeBinomial (handles biological overdispersion).
    - Error-in-Variables: Accounts for dose uncertainty (bin_stds).
    """
    def __init__(self, cell_line="Generic", name="CellSurvival_LQ"):
        """
        Args:
            cell_line: 'U251', 'HaCaT' or 'Generic'. Sets biological priors.
        """
        super().__init__(name=f"{name}_{cell_line}")
        self.cell_line = cell_line
        
    def _get_priors(self):
        """
        Returns cell-line specific prior parameters.
        Format: {param: (mu, sigma)}
        """
        if self.cell_line == "U251":
            # Glioblastoma: Radioresistant, flatter curve
            return {
                'alpha': (0.15, 0.10), # Low alpha (resistance)
                'beta':  (0.04, 0.02)
            }
        elif self.cell_line == "HaCaT":
            # Keratinocyte: "Normal" response, medium sensitivity
            return {
                'alpha': (0.25, 0.10), # Steeper initial slope
                'beta':  (0.05, 0.02)  # Definite curvature (repair mechanism)
            }
        else:
            # Generic
            return {
                'alpha': (0.30, 0.20),
                'beta':  (0.05, 0.02)
            }

    def add_experiment_from_histogram(self, hist_data, colony_count, n_seeded, **kwargs):
        """
        Add experimental data in histogram format.
        
        Args:
            hist_data: Output of 'get_dvh_statistics' (dict).
                       Contains: 'dose_stats' (2xN), 'weights' (N).
            colony_count: Counted colonies (outcome).
            n_seeded: Number of seeded cells.
        """
        # Unpack data
        dose_stats = hist_data['dose_stats'] # Row 0: Dose, Row 1: Std
        weights = hist_data['weights']
        
        if len(weights) != dose_stats.shape[1]:
            raise ValueError("Number of weights and dose bins do not match!")

        data = {
            'dose_bins': dose_stats[0, :],  # (N_bins,)
            'bin_stds':  dose_stats[1, :],  # (N_bins,)
            'weights':   weights,           # (N_bins,)
            'outcome':   int(colony_count), # Scalar
            'n_seeded':  int(n_seeded)      # Scalar
        }
        data.update(kwargs) 
        
        # Calculate mean dose for console log
        mean_dose = np.sum(data['dose_bins'] * weights)
        self.experiments.append(data)
        print(f"[{self.name}] Sample added: {colony_count} colonies (Mean Dose: {mean_dose:.2f} Gy)")

    def build_model(self):
        n_experiments = len(self.experiments)
        if n_experiments == 0: return

        # Organize data into matrices (N_exp, N_bins)
        dose_meas = np.vstack([e['dose_bins'] for e in self.experiments])
        dose_std  = np.vstack([e['bin_stds'] for e in self.experiments])
        weights   = np.vstack([e['weights'] for e in self.experiments])
        
        outcomes  = np.array([e['outcome'] for e in self.experiments])
        n_seededs = np.array([e['n_seeded'] for e in self.experiments]) 

        priors = self._get_priors()
        
        print(f"[{self.name}] Building model with '{self.cell_line}' priors: {priors}")

        with pm.Model() as self.model:
            # --- 1. PARAMETERS (Priors) ---
            # Alpha: Linear component
            alpha = pm.TruncatedNormal('alpha', 
                                     mu=priors['alpha'][0], 
                                     sigma=priors['alpha'][1], 
                                     lower=0.0)
            # Beta: Quadratic component
            beta = pm.TruncatedNormal('beta', 
                                    mu=priors['beta'][0], 
                                    sigma=priors['beta'][1], 
                                    lower=0.0)
            
            # PE (Plating Efficiency)
            pe = pm.Beta('pe', alpha=2, beta=2)
            
            # Overdispersion for Negative Binomial
            dispersion = pm.HalfNormal('dispersion', sigma=10.0)

            # --- 2. ERROR-IN-VARIABLES (The Secret Weapon) ---
            # Handling uncertainty in measured dose
            # sigma + 1e-3 for numerical stability
            true_dose = pm.Normal('true_dose', mu=dose_meas, sigma=dose_std + 1e-3, shape=dose_meas.shape)
            
            # Dose cannot be negative (physical constraint)
            true_dose_clipped = pm.math.maximum(true_dose, 0.0)

            # --- 3. LQ MODEL & INTEGRATION ---
            # S_bin = exp(-alpha*D - beta*D^2)
            exponent = -alpha * true_dose_clipped - beta * (true_dose_clipped**2)
            
            # Numerical protection
            exponent = pm.math.clip(exponent, -20, 0)
            S_bins = pm.math.exp(exponent)
            
            # Weighted average (Handles inhomogeneity!)
            # S_total = sum(S_bin * weight)
            S_weighted = pm.math.sum(S_bins * weights, axis=1)
            
            # --- 4. LIKELIHOOD ---
            # Expected colony count: mu = N_seeded * PE * S_total
            mu = n_seededs * pe * S_weighted
            
            # Observation
            pm.NegativeBinomial('obs', mu=mu, alpha=dispersion, observed=outcomes)

    def calculate_ld50(self):
        """
        Calculates LD50 (where Survival Fraction = 0.5).
        LQ model: alpha*D + beta*D^2 = ln(2)
        Solved using quadratic formula.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return None

        print("\n--- LD50 Calculation (Cell Survival) ---")
        post = self.idata.posterior
        
        # Flatten parameters
        a = post['beta'].values.flatten()  # quadratic term (a)
        b = post['alpha'].values.flatten() # linear term (b)
        
        # Target: S = 0.5 -> -ln(0.5) = ln(2)
        # Equation: a*D^2 + b*D - ln(2) = 0
        c = -np.log(2) 
        
        discriminant = b**2 - 4 * a * c
        
        # Only valid, real solutions
        valid_idx = (discriminant >= 0) & (a > 1e-6) 
        
        if np.sum(valid_idx) < len(a) * 0.9:
            print("WARNING: Some samples did not yield real solutions (curve too flat).")
        
        d_vals = (-b[valid_idx] + np.sqrt(discriminant[valid_idx])) / (2 * a[valid_idx])
        
        # Statistics
        mean_ld50 = np.mean(d_vals)
        hpd_low = np.percentile(d_vals, 2.5)
        hpd_high = np.percentile(d_vals, 97.5)
        
        print(f"Estimated LD50: {mean_ld50:.2f} Gy")
        print(f"95% Credible Interval: [{hpd_low:.2f}, {hpd_high:.2f}] Gy")
        
        return mean_ld50, (hpd_low, hpd_high)

    def plot_survival_curves(self, max_dose=10.0):
        """
        Plots the Survival Curve (Survival Fraction vs Dose).
        """
        if self.idata is None: return
        
        post = self.idata.posterior
        alpha_s = post['alpha'].values.flatten()
        beta_s = post['beta'].values.flatten()
        pe_s = post['pe'].values.flatten() 
        
        d_axis = np.linspace(0, max_dose, 100)
        
        # Generate curves (S = exp(-alpha*D - beta*D^2))
        exponent = -alpha_s[:, None] * d_axis[None, :] - beta_s[:, None] * (d_axis[None, :]**2)
        survivals = np.exp(exponent)
        
        # Statistics
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        plt.figure(figsize=(8, 6))
        
        # 1. Model Band
        plt.fill_between(d_axis, hpd_low, hpd_high, color='blue', alpha=0.2, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='blue', lw=2, label='LQ Model')
        
        # 2. Measured Data (Points)
        mean_pe = np.mean(pe_s)
        
        for i, exp in enumerate(self.experiments):
            # X: Mean Dose
            d_mean = np.sum(exp['dose_bins'] * exp['weights'])
            
            # Y: Measured SF = Colony / (Seeded * PE)
            sf_measured = exp['outcome'] / (exp['n_seeded'] * mean_pe)
            
            # Poisson error bar
            y_err = (np.sqrt(exp['outcome']) / exp['outcome']) * sf_measured if exp['outcome'] > 0 else 0
            
            label_text = 'Measured Data' if i == 0 else ""
            
            plt.errorbar(d_mean, sf_measured, yerr=y_err, fmt='o', color='red', capsize=3, label=label_text)

        plt.yscale('log')
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Fraction (SF)")
        plt.title(f"LQ Cell Survival Curve ({self.cell_line})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.ylim(0.001, 1.2)
        plt.show()

    def check_posterior_predictive(self, samples=500):
        """
        Checks if the model can predict the measured colony counts.
        """
        if self.idata is None: return
        
        print(f"--- Posterior Predictive Check ({samples} samples) ---")
        if 'posterior_predictive' not in self.idata:
            with self.model:
                pm.sample_posterior_predictive(self.idata, extend_inferencedata=True, random_seed=42)
        
        ppc = self.idata.posterior_predictive
        
        # Extract PPC Data
        pred_counts = ppc['obs'].stack(sample=("chain", "draw")).transpose("sample", "obs_dim_0").values
        
        n_exps = len(self.experiments)
        
        plt.figure(figsize=(10, 6))
        
        # Boxplot for each experiment
        plt.boxplot([pred_counts[:, i] for i in range(n_exps)], positions=range(n_exps), widths=0.6, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        
        # Measured values (Red dots)
        real_counts = [e['outcome'] for e in self.experiments]
        plt.plot(range(n_exps), real_counts, 'ro', markersize=8, label='Measured Count')
        
        # Labels (Doses)
        dose_labels = [f"{np.sum(e['dose_bins']*e['weights']):.1f} Gy" for e in self.experiments]
        plt.xticks(range(n_exps), dose_labels, rotation=45)
        
        plt.ylabel("Colony Count")
        plt.xlabel("Experimental Group (Mean Dose)")
        plt.title("Model Validation: Measured vs Predicted Counts")
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        
    def plot_dose_response(self, max_dose=10.0, scale='log', show_ld50=True):
        """
        Plots the Dose-Response curve (same as survival curve but different visualization style).
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return
        
        # 1. Posterior Data
        post = self.idata.posterior
        alpha_s = post['alpha'].values.flatten()
        beta_s = post['beta'].values.flatten()
        pe_s = post['pe'].values.flatten()
        mean_pe = np.mean(pe_s)
        
        # 2. X axis and Model Curve
        d_axis = np.linspace(0, max_dose, 200)
        
        # LQ formula
        exponent = -alpha_s[:, None] * d_axis[None, :] - beta_s[:, None] * (d_axis[None, :]**2)
        survivals = np.exp(exponent)
        
        # Statistics
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        # 3. Plotting
        plt.figure(figsize=(10, 7))
        
        plt.fill_between(d_axis, hpd_low, hpd_high, color='navy', alpha=0.2, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='navy', lw=2, label='LQ Model Fit')
        
        # 4. Measured Data
        for i, exp in enumerate(self.experiments):
            d_mean = np.sum(exp['dose_bins'] * exp['weights'])
            sf_measured = exp['outcome'] / (exp['n_seeded'] * mean_pe)
            
            if exp['outcome'] > 0:
                yerr = sf_measured * (1.0 / np.sqrt(exp['outcome']))
            else:
                yerr = 0.0
            
            lbl = 'Measured Data (Mean Dose)' if i == 0 else ""
            
            plt.errorbar(d_mean, sf_measured, yerr=yerr, fmt='o', 
                         color='red', ecolor='darkred', capsize=4, markersize=6, 
                         label=lbl, zorder=5)

        # 5. LD50 Line
        if show_ld50:
            a_mean = np.mean(beta_s)
            b_mean = np.mean(alpha_s)
            c = -np.log(2)
            disc = b_mean**2 - 4 * a_mean * c
            if disc >= 0 and a_mean > 1e-6:
                ld50_val = (-b_mean + np.sqrt(disc)) / (2 * a_mean)
                
                plt.axvline(ld50_val, color='green', linestyle='--', lw=2, label=f'LD50: {ld50_val:.2f} Gy')
                plt.axhline(0.5, color='gray', linestyle=':', alpha=0.6)

        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Fraction (SF)")
        plt.title(f"Dose-Response Curve (LQ Model)")
        
        if scale == 'log':
            plt.yscale('log')
            plt.ylim(bottom=0.005, top=1.2) 
        else:
            plt.ylim(0, 1.1)
            
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()
        
        
class CellSurvivalLQLModel(BaseBioModel):
    """
    Linear-Quadratic-Linear (LQL) Model for highly inhomogeneous fields.
    
    Prevents artificial over-penalization of high-dose "hot spots" by assuming 
    the survival curve transitions to a purely linear exponential decay above 
    a threshold dose (D_t).
    
    Equations:
    D < D_t  : S = exp(-alpha*D - beta*D^2)
    D >= D_t : S = exp(-alpha*D_t - beta*D_t^2 - gamma*(D - D_t))
               where gamma = alpha + 2*beta*D_t (to ensure smooth derivative)
    """
    def __init__(self, cell_line="Generic", name="CellSurvival_LQL"):
        super().__init__(name=f"{name}_{cell_line}")
        self.cell_line = cell_line

    def add_experiment_from_histogram(self, hist_data, colony_count, n_seeded, **kwargs):
        """Same data ingestion as the LQ model."""
        dose_stats = hist_data['dose_stats']
        weights = hist_data['weights']
        
        data = {
            'dose_bins': dose_stats[0, :],
            'bin_stds':  dose_stats[1, :],
            'weights':   weights,
            'outcome':   int(colony_count),
            'n_seeded':  int(n_seeded)
        }
        data.update(kwargs)
        self.experiments.append(data)
        
        mean_dose = np.sum(data['dose_bins'] * weights)
        print(f"[{self.name}] Sample added: {colony_count} colonies (Mean Dose: {mean_dose:.2f} Gy)")

    def build_model(self):
        if not self.experiments: return

        dose_meas = np.vstack([e['dose_bins'] for e in self.experiments])
        dose_std  = np.vstack([e['bin_stds'] for e in self.experiments])
        weights   = np.vstack([e['weights'] for e in self.experiments])
        outcomes  = np.array([e['outcome'] for e in self.experiments])
        n_seededs = np.array([e['n_seeded'] for e in self.experiments]) 

        print(f"[{self.name}] Building LQL model...")

        with pm.Model() as self.model:
            # --- 1. PARAMETERS ---
            alpha = pm.TruncatedNormal('alpha', mu=0.2, sigma=0.15, lower=0.0)
            beta = pm.TruncatedNormal('beta', mu=0.03, sigma=0.02, lower=0.0)
            
            # D_t: Threshold dose where the curve becomes linear (typically 8-12 Gy for most cells)
            d_t = pm.TruncatedNormal('d_t', mu=10.0, sigma=2.0, lower=5.0, upper=20.0)
            
            pe = pm.Beta('pe', alpha=2, beta=2)
            dispersion = pm.HalfNormal('dispersion', sigma=10.0)

            # --- 2. ERROR-IN-VARIABLES ---
            true_dose = pm.Normal('true_dose', mu=dose_meas, sigma=dose_std + 1e-3, shape=dose_meas.shape)
            true_dose_clipped = pm.math.maximum(true_dose, 0.0)

            # --- 3. LQL PIECEWISE INTEGRATION ---
            # Part 1: Standard LQ up to D_t
            log_s_lq = -alpha * true_dose_clipped - beta * (true_dose_clipped**2)
            
            # Part 2: Linear extension beyond D_t
            # Calculate survival exactly at D_t
            log_s_dt = -alpha * d_t - beta * (d_t**2)
            # Calculate the slope (derivative) at D_t for a smooth transition
            slope_dt = -alpha - 2 * beta * d_t
            # Apply linear slope for the remaining dose (D - D_t)
            log_s_lin = log_s_dt + slope_dt * (true_dose_clipped - d_t)
            
            # Switch between the two based on voxel dose
            exponent = pm.math.switch(true_dose_clipped < d_t, log_s_lq, log_s_lin)
            
            exponent = pm.math.clip(exponent, -20, 0)
            S_bins = pm.math.exp(exponent)
            
            S_weighted = pm.math.sum(S_bins * weights, axis=1)
            
            # --- 4. LIKELIHOOD ---
            mu = n_seededs * pe * S_weighted
            pm.NegativeBinomial('obs', mu=mu, alpha=dispersion, observed=outcomes)
            
            # Save log-likelihood for LOO comparison
            pm.Deterministic('log_lik', pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=dispersion), outcomes))
            
    def plot_survival_curves(self, max_dose=12.0):
        """Plots the piecewise LQL Survival Curve."""
        if self.idata is None: return
        
        post = self.idata.posterior
        alpha_s = post['alpha'].values.flatten()
        beta_s = post['beta'].values.flatten()
        dt_s = post['d_t'].values.flatten()
        pe_s = post['pe'].values.flatten()
        
        d_axis = np.linspace(0, max_dose, 200)
        
        # Mátrixos számítás az LQL egyenletre
        D_matrix = d_axis[None, :]
        a_mat = alpha_s[:, None]
        b_mat = beta_s[:, None]
        dt_mat = dt_s[:, None]
        
        log_s_lq = -a_mat * D_matrix - b_mat * (D_matrix**2)
        
        log_s_dt = -a_mat * dt_mat - b_mat * (dt_mat**2)
        slope_dt = -a_mat - 2 * b_mat * dt_mat
        log_s_lin = log_s_dt + slope_dt * (D_matrix - dt_mat)
        
        exponent = np.where(D_matrix < dt_mat, log_s_lq, log_s_lin)
        survivals = np.exp(exponent)
        
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='darkred', alpha=0.2, label='95% Credible Interval (LQL)')
        plt.plot(d_axis, mean_curve, color='darkred', lw=2, label='LQL Model Fit')
        
        mean_pe = np.mean(pe_s)
        for i, exp in enumerate(self.experiments):
            d_mean = np.sum(exp['dose_bins'] * exp['weights'])
            sf_measured = exp['outcome'] / (exp['n_seeded'] * mean_pe)
            y_err = (np.sqrt(exp['outcome']) / exp['outcome']) * sf_measured if exp['outcome'] > 0 else 0
            
            label_text = 'Measured Data' if i == 0 else ""
            plt.errorbar(d_mean, sf_measured, yerr=y_err, fmt='o', color='black', capsize=3, label=label_text)

        plt.yscale('log')
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Fraction (SF)")
        plt.title(f"LQL Cell Survival Curve ({self.cell_line})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.ylim(0.001, 1.2)
        plt.show()

    def calculate_ld50(self):
        """Calculates LD50 considering the piecewise LQL nature."""
        if self.idata is None: return None
        post = self.idata.posterior
        a = post['alpha'].values.flatten()
        b = post['beta'].values.flatten()
        dt = post['d_t'].values.flatten()
        
        target_log_s = -np.log(2)
        ld50_samples = np.zeros_like(a)
        
        log_s_at_dt = -a * dt - b * (dt**2)
        mask_lq = target_log_s >= log_s_at_dt 
        
        # LQ szakaszon lévők kiszámítása
        disc = a**2 - 4 * b * (-np.log(2))
        ld50_samples[mask_lq] = (-a[mask_lq] + np.sqrt(disc[mask_lq])) / (2 * b[mask_lq])
        
        # Lineáris szakaszon lévők kiszámítása
        slope = -a - 2 * b * dt
        ld50_samples[~mask_lq] = dt[~mask_lq] + (target_log_s - log_s_at_dt[~mask_lq]) / slope[~mask_lq]
        
        mean_ld50 = np.mean(ld50_samples)
        hpd_low = np.percentile(ld50_samples, 2.5)
        hpd_high = np.percentile(ld50_samples, 97.5)
        return mean_ld50, (hpd_low, hpd_high)
        
    def calculate_rbe(self, ref_ld50, ref_ld50_std=0.0):
        """Calculates RBE relative to a reference LD50."""
        if self.idata is None: return
        print(f"\n--- RBE Calculation (Ref LD50: {ref_ld50:.2f} Gy) ---")
        
        post = self.idata.posterior
        a = post['alpha'].values.flatten()
        b = post['beta'].values.flatten()
        dt = post['d_t'].values.flatten()
        
        target_log_s = -np.log(2)
        ld50_test_samples = np.zeros_like(a)
        
        log_s_at_dt = -a * dt - b * (dt**2)
        mask_lq = target_log_s >= log_s_at_dt 
        
        disc = a**2 - 4 * b * (-np.log(2))
        ld50_test_samples[mask_lq] = (-a[mask_lq] + np.sqrt(disc[mask_lq])) / (2 * b[mask_lq])
        slope = -a - 2 * b * dt
        ld50_test_samples[~mask_lq] = dt[~mask_lq] + (target_log_s - log_s_at_dt[~mask_lq]) / slope[~mask_lq]
        
        n_samples = len(ld50_test_samples)
        if ref_ld50_std > 0:
            ld50_ref_samples = np.random.normal(ref_ld50, ref_ld50_std, n_samples)
        else:
            ld50_ref_samples = np.full(n_samples, ref_ld50)
            
        rbe_samples = ld50_ref_samples / ld50_test_samples
        mean_rbe = np.mean(rbe_samples)
        
        import arviz as az
        hdi = az.hdi(rbe_samples, hdi_prob=0.95)
        
        print(f"RBE (via LD50): {mean_rbe:.2f}")
        print(f"95% Credible Interval: [{hdi[0]:.2f}, {hdi[1]:.2f}]")
        
        plt.figure(figsize=(6, 4))
        plt.hist(rbe_samples, bins=50, density=True, alpha=0.7, color='darkred')
        plt.axvline(mean_rbe, color='k', linestyle='--', label=f'Mean: {mean_rbe:.2f}')
        plt.xlabel("RBE Value")
        plt.title(f"RBE Probability Distribution ({self.name})")
        plt.legend()
        plt.show()
        
        return mean_rbe, hdi

# ==============================================================================
# 2. FISH MODEL: SURVIVAL ANALYSIS (Weibull)
# ==============================================================================
class FishSurvivalModel(BaseBioModel):
    def __init__(self):
        super().__init__(name="FishSurvival_VoxelIntegrated")

    def add_experiment(self, dose_distribution, daily_deaths, n_start, observation_days):
        """
        Add data for a fish group/tank.
        
        Args:
            dose_distribution: 1D array (Gy), doses of voxels.
            daily_deaths: List/Array, counts of deaths per day.
            n_start: Initial number of embryos (e.g., 20).
            observation_days: List of days when counting occurred.
        """
        # Downsampling for speed (monte carlo integration of the volume)
        if len(dose_distribution) > 5000:
            dose_distribution = np.random.choice(dose_distribution, 5000, replace=False)
            
        total_dead = sum(daily_deaths)
        survived = n_start - total_dead
        
        if survived < 0:
            raise ValueError(f"Error: More dead ({total_dead}) than started ({n_start})!")

        # Multinomial vector: [d1, d2, ..., dn, survived]
        observed_counts = np.array(daily_deaths + [survived], dtype=int)
        
        data = {
            'doses': dose_distribution, # Distribution, NOT average!
            'observed_counts': observed_counts,
            'n_start': n_start,
            'days': np.array(observation_days)
        }
        self.experiments.append(data)
        print(f"[{self.name}] Group added: {n_start} embryos, {total_dead} deaths.")

    def build_model(self):
        """
        Weibull Survival with Voxel-Level Integration.
        Hazard(t, D) = lambda(D) * k * t^(k-1)
        S(t, D) = exp( -lambda(D) * t^k )
        """
        with pm.Model() as self.model:
            # --- 1. GLOBAL PARAMETERS (PRIORS) ---
            
            # Weibull shape parameter (k)
            shape_k = pm.Gamma('shape_k', alpha=2.0, beta=1.0)
            
            # Dose-Effect parameters (Log-linear Hazard)
            # lambda(D) = lambda_base * exp(beta * D)
            lambda_base = pm.HalfNormal('lambda_base', sigma=0.5)
            beta_dose = pm.HalfNormal('beta_dose', sigma=0.5)

            # --- 2. LIKELIHOOD PER EXPERIMENT ---
            
            for i, exp in enumerate(self.experiments):
                doses_voxels = exp['doses']     # Vector (N_voxel)
                obs_counts = exp['observed_counts'] # Vector (Days + 1)
                days = exp['days']              # Vector (Days)
                n_start = exp['n_start']        # Scalar
                
                # --- A. VOXEL LEVEL SURVIVAL CALCULATION ---
                # lambda = base * exp(beta * D)
                lambda_voxels = lambda_base * pm.math.exp(beta_dose * doses_voxels)
                
                probs = []
                S_prev_group = 1.0 # Everyone alive at t=0
                
                for t in days:
                    # 1. Survival for each voxel at day t
                    S_voxels_t = pm.math.exp(-lambda_voxels * (t ** shape_k))
                    
                    # 2. INTEGRATION (Volume Averaging)
                    # --- SOFTMIN INTEGRATION ---
                    # Makes the model sensitive to hot-spots.
                    p_exp = -4.0 
                    
                    S_pow = (S_voxels_t + 1e-9) ** p_exp
                    S_pow_mean = pm.math.mean(S_pow)
                    S_curr_group = S_pow_mean ** (1.0 / p_exp)
                    
                    # 3. Probability of death in interval (t-1, t]
                    p_interval = S_prev_group - S_curr_group
                    
                    # Numerical stability
                    p_interval = pm.math.switch(p_interval < 1e-6, 1e-6, p_interval)
                    
                    probs.append(p_interval)
                    S_prev_group = S_curr_group 
                
                # 4. Probability of survival at the end
                S_last = pm.math.switch(S_prev_group < 1e-6, 1e-6, S_prev_group)
                probs.append(S_last)
                
                probs_vector = pm.math.stack(probs)
                probs_vector = probs_vector / pm.math.sum(probs_vector)
                
                # --- B. MULTINOMIAL LIKELIHOOD ---
                pm.Multinomial(f'obs_{i}', n=n_start, p=probs_vector, observed=obs_counts)

    def plot_survival_curves(self, max_dose=15.0, num_curves=5):
        """
        Plots S(t) curves for different theoretical doses.
        """
        if self.idata is None: return
        
        post = self.idata.posterior
        k_mean = post['shape_k'].mean().item()
        lam_base_mean = post['lambda_base'].mean().item()
        beta_mean = post['beta_dose'].mean().item()
        
        max_day = max([np.max(e['days']) for e in self.experiments])
        t_axis = np.linspace(0, max_day + 1, 50)
        
        doses_to_plot = np.linspace(0, max_dose, num_curves)
        
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('coolwarm')
        
        for i, D in enumerate(doses_to_plot):
            # Lambda(D) = lambda_base * exp(beta * D)
            lam_D = lam_base_mean * np.exp(beta_mean * D)
            
            # S(t) = exp(-lam * t^k)
            S_t = np.exp(-lam_D * (t_axis ** k_mean))
            
            color = cmap(i / (num_curves - 1))
            plt.plot(t_axis, S_t, lw=2, color=color, label=f'{D:.1f} Gy')
            
        plt.xlabel("Days")
        plt.ylabel("Survival Probability S(t)")
        plt.title(f"Weibull Survival Curves (0 - {max_dose} Gy)")
        plt.legend(title="Dose")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
        
    def calculate_ld50(self, day=None):
        """
        Calculates LD50 from posterior distribution for a specific day.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return None

        if day is None:
            day = max([np.max(e['days']) for e in self.experiments])

        post = self.idata.posterior
        k_samples = post['shape_k'].values.flatten()
        lam_samples = post['lambda_base'].values.flatten()
        beta_samples = post['beta_dose'].values.flatten()

        print(f"\n--- LD50 Calculation (Day: {day}) ---")
        
        # Formula: LD50 = (1/beta) * ln( ln(2) / (lambda * t^k) )
        denominator = lam_samples * (day ** k_samples)
        denominator = np.where(denominator < 1e-9, 1e-9, denominator)
        
        term = np.log(2) / denominator
        valid_idx = term > 0
        
        if np.sum(valid_idx) < len(term) * 0.9:
            print("WARNING: LD50 undefined for parameters (mortality too low).")
            return None

        ld50_samples = (1.0 / beta_samples[valid_idx]) * np.log(term[valid_idx])
        
        # Statistics
        mean_ld50 = np.mean(ld50_samples)
        hpd_low = np.percentile(ld50_samples, 2.5)
        hpd_high = np.percentile(ld50_samples, 97.5)
        
        print(f"Estimated LD50: {mean_ld50:.2f} Gy")
        print(f"95% Credible Interval: [{hpd_low:.2f}, {hpd_high:.2f}] Gy")
        
        return mean_ld50, (hpd_low, hpd_high)

    def plot_dose_response(self, day=None, max_dose=20.0):
        """
        Plots Dose-Response curve for a specific day.
        """
        if self.idata is None: return

        if day is None:
            day = max([np.max(e['days']) for e in self.experiments])

        post = self.idata.posterior
        k_samples = post['shape_k'].values.flatten()
        lam_samples = post['lambda_base'].values.flatten()
        beta_samples = post['beta_dose'].values.flatten()
        
        d_axis = np.linspace(0, max_dose, 100)
        
        # Vectorized calculation: (N_samples, N_dose_points)
        lambda_D_matrix = lam_samples[:, None] * np.exp(beta_samples[:, None] * d_axis[None, :])
        
        time_factor = day ** k_samples[:, None]
        survivals = np.exp(-lambda_D_matrix * time_factor)
        
        # Statistics
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='green', alpha=0.3, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='darkgreen', lw=2, label=f'Dose-Response (Day {day})')
        
        # Draw LD50 line
        try:
            ld50_res = self.calculate_ld50(day)
            if ld50_res:
                ld50_val, (l50_min, l50_max) = ld50_res
                if 0 < ld50_val < max_dose:
                    plt.axvline(ld50_val, color='red', linestyle='--', label=f'LD50: {ld50_val:.1f} Gy')
                    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
                    plt.axvspan(l50_min, l50_max, color='red', alpha=0.1)
        except Exception as e:
            print(f"Failed to plot LD50: {e}")

        # Measured points (Approximate: Avg Dose vs Survival Ratio)
        for exp in self.experiments:
            avg_dose = np.mean(exp['doses'])
            if np.max(exp['days']) == day:
                final_survived = exp['observed_counts'][-1]
                ratio = final_survived / exp['n_start']
                plt.scatter(avg_dose, ratio, color='black', zorder=5, s=30, marker='o')

        plt.xlabel("Dose [Gy]")
        plt.ylabel(f"Survival Probability (Day {day})")
        plt.title(f"Fish Model Dose-Response (T={day} days)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
        

class FishSurvivalModel_LQ(BaseBioModel):
    def __init__(self):
        super().__init__(name="FishSurvival_LQ_Weibull")

    def add_experiment(self, dose_distribution, daily_deaths, n_start, observation_days):
        """Same as FishSurvivalModel."""
        if len(dose_distribution) > 1000:
            dose_distribution = np.random.choice(dose_distribution, 5000, replace=False)
            
        total_dead = sum(daily_deaths)
        survived = n_start - total_dead
        if survived < 0: raise ValueError("More dead than alive!")

        observed_counts = np.array(daily_deaths + [survived], dtype=int)
        
        data = {
            'doses': dose_distribution,
            'observed_counts': observed_counts,
            'n_start': n_start,
            'days': np.array(observation_days)
        }
        self.experiments.append(data)
        print(f"[{self.name}] Group added: {n_start} embryos (LQ mode).")

    def build_model(self):
        """
        ACCELERATED LQ-Weibull Model.
        """
        with pm.Model() as self.model:
            # 1. Priors
            shape_k = pm.Gamma('shape_k', alpha=2.0, beta=1.0) 
            lambda_base = pm.HalfNormal('lambda_base', sigma=0.2)
            alpha = pm.HalfNormal('alpha', sigma=0.5)
            beta = pm.HalfNormal('beta', sigma=0.02) 

            # 2. Vectorized Likelihood
            for i, exp in enumerate(self.experiments):
                doses_voxels = exp['doses']
                obs_counts = exp['observed_counts']
                days = exp['days']
                n_start = exp['n_start']
                
                # LQ Hazard
                # lambda = base * exp(alpha*D + beta*D^2)
                exponent = alpha * doses_voxels + beta * (doses_voxels**2)
                
                # Numerical protection (overflow prevention)
                exponent = pm.math.clip(exponent, 0, 20) 
                
                lambda_voxels = lambda_base * pm.math.exp(exponent)
                
                # Time matrix
                lam_reshaped = lambda_voxels[:, None] # (N_vox, 1)
                time_factor = days[None, :] ** shape_k # (1, N_days)
                
                # S(t) Matrix (N_vox, N_days)
                S_matrix = pm.math.exp(-lam_reshaped * time_factor)
                
                # Integration (SoftMin)
                p_exp = -4.0  
                
                S_pow = (S_matrix + 1e-9)**p_exp
                S_pow_mean = pm.math.mean(S_pow, axis=0)
                S_group_t = (S_pow_mean)**(1.0 / p_exp)
                
                # Probabilities
                S_extended = pm.math.concatenate([[1.0], S_group_t])
                p_intervals = S_extended[:-1] - S_extended[1:]
                
                # Use maximum instead of switch for better gradients
                p_intervals = pm.math.maximum(p_intervals, 1e-9)
                p_survived = pm.math.maximum(S_extended[-1], 1e-9)
                
                probs_vector = pm.math.concatenate([p_intervals, [p_survived]])
                probs_vector = probs_vector / pm.math.sum(probs_vector)
                
                pm.Multinomial(f'obs_{i}', n=n_start, p=probs_vector, observed=obs_counts)

    def plot_dose_response(self, day=None, max_dose=20.0):
        """
        Plot adapted specifically for LQ model.
        """
        if self.idata is None: return
        if day is None: day = max([np.max(e['days']) for e in self.experiments])

        post = self.idata.posterior
        k_s = post['shape_k'].values.flatten()
        lam_s = post['lambda_base'].values.flatten()
        alp_s = post['alpha'].values.flatten()
        bet_s = post['beta'].values.flatten()
        
        d_axis = np.linspace(0, max_dose, 100)
        
        # Matrix operations for speed
        exponent = (alp_s[:, None] * d_axis[None, :]) + (bet_s[:, None] * (d_axis[None, :]**2))
        lambda_D = lam_s[:, None] * np.exp(exponent)
        
        time_factor = day ** k_s[:, None]
        survivals = np.exp(-lambda_D * time_factor)
        
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='purple', alpha=0.3, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='purple', lw=2, label=f'LQ-Weibull Model (Day {day})')
        
        for exp in self.experiments:
            if np.max(exp['days']) == day:
                avg_dose = np.mean(exp['doses'])
                final_survived = exp['observed_counts'][-1]
                ratio = final_survived / exp['n_start']
                plt.scatter(avg_dose, ratio, color='black', zorder=5)

        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Probability")
        plt.title(f"Linear-Quadratic Fish Model (T={day})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
            
    def calculate_ld50(self, day=None):
        """
        Calculate LD50 using the quadratic formula from LQ parameters.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return None

        if day is None:
            day = max([np.max(e['days']) for e in self.experiments])

        post = self.idata.posterior
        k_s = post['shape_k'].values.flatten()
        lam_s = post['lambda_base'].values.flatten()
        alp_s = post['alpha'].values.flatten()
        bet_s = post['beta'].values.flatten()

        print(f"\n--- LQ-Weibull LD50 Calculation (Day: {day}) ---")
        
        # Target hazard rate for 50% survival
        target_lambda = np.log(2) / (day ** k_s)
        
        # Calculate log-hazard increment (Constant C)
        # lambda(D) = lambda_base * exp(LQ) -> LQ = ln(target / base)
        ratio = target_lambda / lam_s
        valid_idx = ratio > 1.0 
        
        if np.sum(valid_idx) < len(ratio) * 0.1:
            print("WARNING: Control survival too low for LD50 calc.")
            return None

        C = np.log(ratio[valid_idx])
        a = bet_s[valid_idx]
        b = alp_s[valid_idx]
        
        # Quadratic equation: a*D^2 + b*D - C = 0
        discriminant = b**2 + 4 * a * C
        ld50_samples = (-b + np.sqrt(discriminant)) / (2 * a)
        
        mean_ld50 = np.mean(ld50_samples)
        hpd_low = np.percentile(ld50_samples, 2.5)
        hpd_high = np.percentile(ld50_samples, 97.5)
        
        print(f"Estimated LD50 (LQ): {mean_ld50:.2f} Gy")
        print(f"95% Credible Interval: [{hpd_low:.2f}, {hpd_high:.2f}] Gy")
        
        return mean_ld50, (hpd_low, hpd_high)

    def plot_survival_curves(self, max_dose=15.0, num_curves=5):
        """
        Plots time-dependent survival curves based on LQ parameters.
        """
        if self.idata is None: return
        
        post = self.idata.posterior
        k_mean = post['shape_k'].mean().item()
        lam_base_mean = post['lambda_base'].mean().item()
        alpha_mean = post['alpha'].mean().item()
        beta_mean = post['beta'].mean().item()
        
        max_day = max([np.max(e['days']) for e in self.experiments])
        t_axis = np.linspace(0, max_day + 1, 50)
        doses_to_plot = np.linspace(0, max_dose, num_curves)
        
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('magma_r') 
        
        for i, D in enumerate(doses_to_plot):
            exponent = alpha_mean * D + beta_mean * (D**2)
            lam_D = lam_base_mean * np.exp(exponent)
            S_t = np.exp(-lam_D * (t_axis ** k_mean))
            
            color = cmap(i / len(doses_to_plot))
            plt.plot(t_axis, S_t, lw=2, color=color, label=f'{D:.1f} Gy')
            
        plt.xlabel("Days")
        plt.ylabel("Survival Probability S(t)")
        plt.title(f"LQ-Weibull Survival Curves (0 - {max_dose} Gy)")
        plt.legend(title="Dose")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()

# ==============================================================================
# 4. POPULATION MODEL: EPPENDORF / GROUP (Histogram based)
# ==============================================================================
class FishSurvivalModel_Population(BaseBioModel):
    """
    Optimized model for Eppendorf tube experiments (e.g., 96 embryos/group).
    
    Differences:
    1. Bin-based (Histogram) integration: Faster than raw voxels.
    2. Weighted Average (Population Integral): Assumes group survival is the
       average of individual survival probabilities.
    """
    def __init__(self, name="Fish_Population_LQ"):
        super().__init__(name=name)
        self.n_bins = 50 

    def add_experiment(self, dose_distribution, daily_deaths, n_start, observation_days):
        """
        Adds data. Automatically converts dose distribution to histogram.
        """
        # 1. Create Histogram (Compression)
        hist_counts, bin_edges = np.histogram(dose_distribution, bins=self.n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weights = hist_counts / np.sum(hist_counts)
        
        # 2. Prepare Observations
        total_dead = sum(daily_deaths)
        survived = n_start - total_dead
        
        if survived < 0:
            raise ValueError(f"Error: More dead ({total_dead}) than started ({n_start})!")
            
        obs_vector = np.array(daily_deaths + [survived], dtype=int)
        
        data = {
            'dose_bins': bin_centers,      # (N_bins,)
            'weights': weights,            # (N_bins,)
            'obs_vector': obs_vector,      # (N_days + 1,)
            'n_start': n_start,
            'days': np.array(observation_days)
        }
        self.experiments.append(data)
        print(f"[{self.name}] Group added: {n_start} embryos (Mean Dose: {np.mean(dose_distribution):.2f} Gy)")
        
    def add_experiment_from_histogram(self, hist_data, daily_deaths, n_start, observation_days):
        """
        Add data directly from loaded dictionary.
        """
        dose_stats = hist_data['dose_stats']
        weights = hist_data['weights']
        
        total_dead = sum(daily_deaths)
        survived = n_start - total_dead
        if survived < 0: raise ValueError("More dead than started!")
        
        obs_vector = np.array(daily_deaths + [survived], dtype=int)
        
        data = {
            'dose_bins': dose_stats[0, :], 
            'bin_stds':  dose_stats[1, :], 
            'weights':   weights,
            'obs_vector': obs_vector,
            'n_start':   n_start,
            'days':      np.array(observation_days)
        }
        self.experiments.append(data)
        
        mean_dose = np.sum(data['dose_bins'] * weights)
        print(f"[{self.name}] Group added (Mean Dose: {mean_dose:.2f} Gy)")

    def build_model(self):
        n_groups = len(self.experiments)
        if n_groups == 0: return

        # Matrix organization
        dose_meas = np.vstack([e['dose_bins'] for e in self.experiments])
        dose_std  = np.vstack([e['bin_stds'] for e in self.experiments])
        weights   = np.vstack([e['weights'] for e in self.experiments])
        obs_matrix = np.vstack([e['obs_vector'] for e in self.experiments])
        time_points = self.experiments[0]['days']
        n_totals = np.sum(obs_matrix, axis=1)

        with pm.Model() as self.model:
            # --- 1. PARAMETERS ---
            k = pm.Gamma('k', alpha=2.0, beta=1.0)
            intercept = pm.Normal('intercept', mu=-3, sigma=1.0)
            alpha = pm.HalfNormal('alpha', sigma=0.5)
            beta = pm.HalfNormal('beta', sigma=0.05)
            
            # --- 2. ERROR-IN-VARIABLES ---
            # True Dose is a latent variable normally distributed around measurement.
            true_dose = pm.Normal('true_dose', mu=dose_meas, sigma=dose_std, shape=dose_meas.shape)
            true_dose_clipped = pm.math.maximum(true_dose, 0.0)

            # --- 3. PHYSICS MODEL (LQ-Weibull) ---
            log_lambdas = intercept + alpha * true_dose_clipped + beta * (true_dose_clipped**2)
            lambda_bins = pm.math.exp(pm.math.clip(log_lambdas, -10, 20))
            
            # Time survival S(t)
            lam_broad = lambda_bins[:, :, None]
            import pytensor.tensor as pt
            t_broad = pt.as_tensor_variable(time_points)[None, None, :]
            
            S_bins_t = pm.math.exp(-lam_broad * (t_broad ** k))
            
            # --- 4. POPULATION INTEGRATION ---
            # Weighted average
            w_broad = weights[:, :, None]
            S_pop = pm.math.sum(S_bins_t * w_broad, axis=1)
            
            # --- 5. LIKELIHOOD (Dirichlet-Multinomial) ---
            ones = pt.ones((n_groups, 1))
            S_extended = pt.concatenate([ones, S_pop], axis=1)
            p_death = pm.math.maximum(S_extended[:, :-1] - S_extended[:, 1:], 1e-9)
            p_survive = pm.math.maximum(S_extended[:, -1:], 1e-9)
            probs_vector = pt.concatenate([p_death, p_survive], axis=1)
            probs_vector = probs_vector / pt.sum(probs_vector, axis=1, keepdims=True)
            
            # Robustness (Overdispersion)
            phi = pm.HalfNormal('phi', sigma=50.0)
            pm.DirichletMultinomial('obs', n=n_totals, a=probs_vector * phi, observed=obs_matrix)

    def calculate_ld50(self, day=None):
        """
        CORRECTED LD50 CALCULATION (Adjusted for Inhomogeneity).
        
        Problem: Intrinsic LD50 assumes homogeneous dose. Experiments have inhomogeneity.
        Solution: Calculate "Apparent LD50" by scaling the reference dose distribution
        until 50% survival is reached.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return None
        
        if day is None: 
            day = self.experiments[0]['days'][-1]
            
        print(f"\n--- Experimental (Apparent) LD50 Estimation (Day {day}) ---")

        # 1. Reference Distribution (Shape from first group)
        ref_doses = self.experiments[0]['dose_bins']
        ref_weights = self.experiments[0]['weights']
        
        current_mean = np.sum(ref_doses * ref_weights)
        norm_doses = ref_doses / current_mean
        
        # 2. Posterior Parameters
        post = self.idata.posterior
        k_s = post['k'].values.flatten()
        int_s = post['intercept'].values.flatten()
        alp_s = post['alpha'].values.flatten()
        bet_s = post['beta'].values.flatten()
        
        # 3. Search: Which mean dose scale gives 50% survival?
        dose_scales = np.linspace(0, 50, 200) 
        ld50_samples = []
        
        # Sample every 10th for speed
        indices = np.arange(0, len(k_s), 10) 
        
        for i in indices:
            kk = k_s[i]
            ii = int_s[i]
            aa = alp_s[i]
            bb = bet_s[i]
            
            t_pow = day ** kk
            D_matrix = np.outer(dose_scales, norm_doses)
            
            log_lam = ii + aa * D_matrix + bb * (D_matrix**2)
            lam = np.exp(log_lam)
            
            S_bins = np.exp(-lam * t_pow)
            S_pop = np.sum(S_bins * ref_weights, axis=1)
            
            # Where do we cross 0.5?
            idx = np.argmin(np.abs(S_pop - 0.5))
            ld50_samples.append(dose_scales[idx])
            
        ld50_samples = np.array(ld50_samples)
        
        mean_ld50 = np.mean(ld50_samples)
        hpd_low = np.percentile(ld50_samples, 2.5)
        hpd_high = np.percentile(ld50_samples, 97.5)
        
        print(f"Estimated Apparent LD50: {mean_ld50:.2f} Gy (Mean Dose)")
        print(f"95% Credible Interval: [{hpd_low:.2f}, {hpd_high:.2f}] Gy")
        
        return mean_ld50, (hpd_low, hpd_high)

    def plot_dose_response(self, day=None, max_dose=25.0):
        """Plots population level dose-response curve."""
        if self.idata is None: return
        if day is None: day = self.experiments[0]['days'][-1]

        post = self.idata.posterior
        k_s = post['k'].values.flatten()
        int_s = post['intercept'].values.flatten()
        alp_s = post['alpha'].values.flatten()
        bet_s = post['beta'].values.flatten()
        
        d_axis = np.linspace(0, max_dose, 100)
        
        log_lam = (int_s[:, None] + 
                   alp_s[:, None] * d_axis[None, :] + 
                   bet_s[:, None] * (d_axis[None, :]**2))
        lam = np.exp(log_lam)
        
        t_factor = day ** k_s[:, None]
        survivals = np.exp(-lam * t_factor)
        
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='orange', alpha=0.3, label='95% CI')
        plt.plot(d_axis, mean_curve, color='darkorange', lw=2, label=f'Population Model (Day {day})')
        
        for exp in self.experiments:
             avg_dose = np.sum(exp['dose_bins'] * exp['weights'])
             final_surv = exp['obs_vector'][-1] / exp['n_start']
             plt.scatter(avg_dose, final_surv, color='black', s=20)

        plt.xlabel("Dose [Gy]")
        plt.ylabel(f"Survival Probability (Day {day})")
        plt.title("Population Level Dose-Response")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
        
    def plot_survival_curves(self, max_dose=20.0, num_curves=5):
        """Plots time-dependent survival curves."""
        if self.idata is None:
            print("Error: Run .sample() first!")
            return
        
        post = self.idata.posterior
        k_mean = post['k'].mean().item()
        int_mean = post['intercept'].mean().item()
        alpha_mean = post['alpha'].mean().item()
        beta_mean = post['beta'].mean().item()
        
        max_day = max([np.max(e['days']) for e in self.experiments])
        t_axis = np.linspace(0, max_day + 1, 100)
        doses_to_plot = np.linspace(0, max_dose, num_curves)
        
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('coolwarm')
        
        for i, D in enumerate(doses_to_plot):
            log_lam = int_mean + alpha_mean * D + beta_mean * (D**2)
            lam_D = np.exp(log_lam)
            S_t = np.exp(-lam_D * (t_axis ** k_mean))
            
            color = cmap(i / (len(doses_to_plot) - 1)) if len(doses_to_plot) > 1 else 'blue'
            plt.plot(t_axis, S_t, lw=2, color=color, label=f'{D:.1f} Gy')
            
        plt.xlabel("Days")
        plt.ylabel("Survival Probability S(t)")
        plt.title(f"Population Model Survival Curves (0 - {max_dose} Gy)")
        plt.legend(title="Dose")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.show()
        
    def check_posterior_predictive(self, samples=500):
        """
        Validates model by comparing observed cumulative deaths vs predicted.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return

        print(f"--- Posterior Predictive Check ({samples} samples) ---")
        
        if 'posterior_predictive' not in self.idata:
            with self.model:
                pm.sample_posterior_predictive(
                    self.idata, 
                    extend_inferencedata=True,
                    random_seed=42,
                    progressbar=True
                )
        
        ppc = self.idata.posterior_predictive
        
        n_groups = len(self.experiments)
        cols = 3
        rows = (n_groups + cols - 1) // cols
        
        if n_groups == 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            axes = [ax]
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).flatten()
        
        for i, exp in enumerate(self.experiments):
            ax = axes[i]
            n_start = exp['n_start']
            days = exp['days']
            
            real_dead_per_interval = exp['obs_vector'][:-1] 
            real_cum_dead = np.cumsum(real_dead_per_interval)
            real_cum_perc = real_cum_dead / n_start
            
            if 'obs' not in ppc:
                 print("Error: 'obs' variable missing from PPC.")
                 return
                 
            ppc_samples = ppc['obs'].stack(sample=("chain", "draw")).transpose("sample", ...).values
            group_samples = ppc_samples[:, i, :] 
            dead_samples = group_samples[:, :-1]
            cum_dead_samples = np.cumsum(dead_samples, axis=1)
            cum_perc_samples = cum_dead_samples / n_start
            
            mean_pred = np.mean(cum_perc_samples, axis=0)
            lower = np.percentile(cum_perc_samples, 2.5, axis=0)
            upper = np.percentile(cum_perc_samples, 97.5, axis=0)
            
            avg_dose = exp['dose_mean'] if 'dose_mean' in exp else np.mean(exp['dose_bins'])
            
            ax.fill_between(days, lower, upper, color='green', alpha=0.3, label='Model 95%')
            ax.plot(days, mean_pred, color='green', linestyle='--', alpha=0.8)
            ax.plot(days, real_cum_perc, 'o-', color='black', lw=2, label='Measured')
            
            ax.set_title(f"Group {i+1} (D ~ {avg_dose:.1f} Gy)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            if i == 0: ax.legend(loc='upper left', fontsize='small')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.text(0.5, 0.02, 'Days', ha='center')
        fig.text(0.02, 0.5, 'Cumulative Mortality (%)', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.03, 0.03, 1, 1])
        plt.show()
        
        
class EUD_SurvivalModel(BaseBioModel):
    """
    Error-in-Variables LQ Model.
    
    This model DOES NOT use full volumetric histograms (sensitive to cold spots).
    Instead, it expects scalar inputs:
    - Mean Dose in the relevant volume (e.g., pellet).
    - Dose Std to handle uncertainty.
    """
    
    def __init__(self, cell_line="U251", name="EUD_Model"):
        super().__init__(name=f"{name}_{cell_line}")
        self.cell_line = cell_line
        self.experiments = []
        self.idata = None
        self.model = None

    @staticmethod
    def get_sediment_dose_stats(dvh_stats, top_percentile=0.20):
        """
        HELPER FUNCTION: Calculates statistics for the 'warm' part of the histogram.
        """
        if 'weights' in dvh_stats:
            weights = np.array(dvh_stats['weights'])
        elif 'hist' in dvh_stats:
            weights = np.array(dvh_stats['hist'])
        else:
            raise KeyError(f"ERROR: Weights not found! Keys: {list(dvh_stats.keys())}")

        bins = None
        if 'dose_bins' in dvh_stats:
            bins = np.array(dvh_stats['dose_bins'])
        elif 'bins' in dvh_stats:
            bins = np.array(dvh_stats['bins'])
        elif 'bin_centers' in dvh_stats:
            bins = np.array(dvh_stats['bin_centers'])
        elif 'dose_stats' in dvh_stats:
            stats = dvh_stats['dose_stats']
            if isinstance(stats, np.ndarray) and stats.ndim == 2 and stats.shape[0] >= 1:
                bins = stats[0]
            elif isinstance(stats, dict):
                 max_d = stats.get('max') or stats.get('max_dose')
                 min_d = stats.get('min') or stats.get('min_dose') or 0.0
                 if max_d is not None:
                     bins = np.linspace(min_d, max_d, len(weights))
        
        if bins is None:
             raise KeyError(f"Could not find dose axis! Available keys: {list(dvh_stats.keys())}")

        if np.mean(bins) < 0.1:
            avg = np.average(bins, weights=weights)
            if len(bins) > 1:
                var = np.average((bins - avg)**2, weights=weights)
                return avg, np.sqrt(var)
            return avg, 0.1

        sorted_indices = np.argsort(bins)
        sorted_weights = weights[sorted_indices]
        sorted_bins = bins[sorted_indices]
        
        cumulative_weight_reverse = np.cumsum(sorted_weights[::-1])
        if len(cumulative_weight_reverse) == 0:
             return 0.0, 0.0

        cutoff_idx_reverse = np.searchsorted(cumulative_weight_reverse, top_percentile)
        real_start_idx = len(weights) - 1 - cutoff_idx_reverse
        real_start_idx = max(0, real_start_idx)
        
        rel_bins = sorted_bins[real_start_idx:]
        rel_weights = sorted_weights[real_start_idx:]
        
        if np.sum(rel_weights) > 0:
            rel_weights = rel_weights / np.sum(rel_weights)
        else:
            return np.max(bins), 0.1 
            
        mean_d = np.average(rel_bins, weights=rel_weights)
        variance = np.average((rel_bins - mean_d)**2, weights=rel_weights)
        std_d = np.sqrt(variance)
        
        return mean_d, std_d

    def add_experiment_from_dvh(self, dvh_data, outcome, n_seeded, batch_id=0, top_percentile=0.20):
        """
        Add experiment with batch ID.
        Same batch_id -> Shared PE. Different batch_id -> Independent PE.
        """
        mean_d, std_d = self.get_sediment_dose_stats(dvh_data, top_percentile)
        
        self.experiments.append({
            'dose_mean': mean_d,
            'dose_std': std_d,
            'outcome': int(outcome),
            'n_seeded': int(n_seeded),
            'batch_id': int(batch_id), 
            'raw_dvh_name': dvh_data.get('name', 'Unknown')
        })
        print(f"[{self.name}] Data (Batch {batch_id}): D={mean_d:.2f} Gy -> SF ~ {outcome/n_seeded:.2f}")

    def build_model(self):
        """
        Builds the PyMC model with Error-in-Variables.
        """
        if not self.experiments:
            print("Error: No data!")
            return

        d_means = np.array([e['dose_mean'] for e in self.experiments])
        d_stds  = np.array([e['dose_std'] for e in self.experiments])
        outcomes = np.array([e['outcome'] for e in self.experiments])
        n_seededs = np.array([e['n_seeded'] for e in self.experiments])
        
        d_stds = np.maximum(d_stds, 0.05) 

        print(f"--- Building Model ({len(d_means)} points) ---")
        
        with pm.Model() as self.model:
            if self.cell_line == "U251":
                alpha = pm.TruncatedNormal('alpha', mu=0.15, sigma=0.1, lower=0.0)
                beta = pm.TruncatedNormal('beta', mu=0.04, sigma=0.02, lower=0.0)
            else:
                alpha = pm.TruncatedNormal('alpha', mu=0.3, sigma=0.2, lower=0.0)
                beta = pm.TruncatedNormal('beta', mu=0.05, sigma=0.03, lower=0.0)
            
            pe = pm.Beta('pe', alpha=2, beta=2)
            dispersion = pm.HalfNormal('dispersion', sigma=10.0)

            # Error-in-Variables
            true_dose = pm.Normal('true_dose', mu=d_means, sigma=d_stds, shape=len(d_means))
            true_dose_clipped = pm.math.maximum(true_dose, 0.0)

            # LQ Model
            exponent = -alpha * true_dose_clipped - beta * (true_dose_clipped**2)
            exponent = pm.math.clip(exponent, -20, 0)
            S = pm.math.exp(exponent)
            
            # Likelihood
            mu = n_seededs * pe * S
            pm.NegativeBinomial('obs', mu=mu, alpha=dispersion, observed=outcomes)

    def sample(self, draws=3000, tune=1000, chains=4, target_accept=0.99):
        if self.model is None: self.build_model()
        with self.model:
            # 1. Mintavétel
            self.idata = pm.sample(draws=draws, tune=tune, target_accept=target_accept, chains=chains, return_inferencedata=True)
            
            # 2. Log-Likelihood számítása (EZ AZ ÚJ LÉPÉS!)
            pm.compute_log_likelihood(self.idata)    


    def calculate_ld50(self):
        """
        Calculates LD50 from posterior samples.
        alpha*D + beta*D^2 - ln(2) = 0
        """
        if self.idata is None: return
        
        post = self.idata.posterior
        alpha = post['alpha'].values.flatten()
        beta = post['beta'].values.flatten()
        
        c = -np.log(2)
        discriminant = alpha**2 - 4 * beta * c
        valid = discriminant >= 0
        
        sqrt_disc = np.sqrt(discriminant[valid])
        ld50_samples = (-alpha[valid] + sqrt_disc) / (2 * beta[valid])
        
        mean_ld50 = np.mean(ld50_samples)
        hdi = az.hdi(ld50_samples, hdi_prob=0.95)
        
        print(f"\n--- LD50 Result ({self.name}) ---")
        print(f"LD50: {mean_ld50:.2f} Gy")
        print(f"95% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}] Gy")
        return mean_ld50, hdi

    def plot_dose_response(self):
        """
        Plots Dose-Response Curve.
        Includes horizontal error bars (xerr) for dose uncertainty!
        """
        if self.idata is None: return
        
        post = self.idata.posterior
        alpha_s = post['alpha'].values.flatten()
        beta_s = post['beta'].values.flatten()
        pe_s = post['pe'].values.flatten()
        mean_pe = np.mean(pe_s)
        
        max_dose = max([e['dose_mean'] for e in self.experiments]) * 1.2
        if max_dose < 5: max_dose = 10
        
        d_axis = np.linspace(0, max_dose, 200)
        exponent = -alpha_s[:, None] * d_axis[None, :] - beta_s[:, None] * (d_axis[None, :]**2)
        survivals = np.exp(exponent)
        
        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)
        
        plt.figure(figsize=(10, 7))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='navy', alpha=0.2, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='navy', lw=2, label='LQ Model')
        
        for i, exp in enumerate(self.experiments):
            x = exp['dose_mean']
            x_err = exp['dose_std']
            y = exp['outcome'] / (exp['n_seeded'] * mean_pe)
            
            if exp['outcome'] > 0:
                y_err = y * (1.0 / np.sqrt(exp['outcome']))
            else:
                y_err = 0
            
            label = "Measured (+/- 1 SD)" if i == 0 else ""
            plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', 
                         color='red', ecolor='darkred', capsize=3, elinewidth=1, 
                         label=label, alpha=0.8)

        mean_ld50, _ = self.calculate_ld50()
        plt.axvline(mean_ld50, color='green', linestyle='--', label=f'LD50: {mean_ld50:.2f} Gy')
        plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        plt.yscale('log')
        plt.ylim(0.005, 1.2)
        plt.xlim(left=0)
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Fraction (SF)")
        plt.title(f"Corrected Dose-Response ({self.cell_line})")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()
        
    def plot_survival_curves(self, max_dose=None):
        """
        Classic SF vs Dose plot with horizontal error bars.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return

        post = self.idata.posterior
        alpha_s = post['alpha'].values.flatten()
        beta_s = post['beta'].values.flatten()
        pe_s = post['pe'].values.flatten()
        mean_pe = np.mean(pe_s)

        if max_dose is None:
            max_measured = max([e['dose_mean'] for e in self.experiments])
            max_dose = max_measured * 1.2
            if max_dose < 5: max_dose = 8.0

        d_axis = np.linspace(0, max_dose, 200)
        exponent = -alpha_s[:, None] * d_axis[None, :] - beta_s[:, None] * (d_axis[None, :]**2)
        survivals = np.exp(exponent)

        mean_curve = np.mean(survivals, axis=0)
        hpd_low = np.percentile(survivals, 2.5, axis=0)
        hpd_high = np.percentile(survivals, 97.5, axis=0)

        plt.figure(figsize=(9, 7))
        plt.fill_between(d_axis, hpd_low, hpd_high, color='blue', alpha=0.2, label='95% Credible Interval')
        plt.plot(d_axis, mean_curve, color='blue', lw=2, label='LQ Model Fit')

        for i, exp in enumerate(self.experiments):
            dose_val = exp['dose_mean']
            dose_std = exp['dose_std']
            sf_measured = exp['outcome'] / (exp['n_seeded'] * mean_pe)
            
            if exp['outcome'] > 0:
                y_err = sf_measured * (1.0 / np.sqrt(exp['outcome']))
            else:
                y_err = 0.0

            label = 'Measured Data (Mean Dose +/- Std)' if i == 0 else ""
            plt.errorbar(dose_val, sf_measured, xerr=dose_std, yerr=y_err, 
                         fmt='o', color='red', ecolor='darkred', 
                         capsize=3, elinewidth=1.5, alpha=0.8, label=label)

        plt.yscale('log')
        plt.ylim(0.005, 1.5)
        plt.xlim(left=0)
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Survival Fraction (SF)")
        plt.title(f"Survival Curve: {self.cell_line} (Error-in-Variables Fit)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()
        
    def check_posterior_predictive(self, samples=500):
        """
        Validates model prediction accuracy via boxplots.
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return
        
        print(f"--- Posterior Predictive Check ({samples} samples) ---")
        if 'posterior_predictive' not in self.idata:
            with self.model:
                pm.sample_posterior_predictive(
                    self.idata, 
                    extend_inferencedata=True, 
                    random_seed=42,
                    progressbar=True
                )
        
        ppc = self.idata.posterior_predictive
        
        if 'obs' in ppc:
            pred_counts = ppc['obs'].stack(sample=("chain", "draw")).transpose("sample", "obs_dim_0").values
        else:
            print("Error: 'obs' variable missing from PPC.")
            return
        
        n_exps = len(self.experiments)
        plt.figure(figsize=(12, 6))
        
        plt.boxplot(
            [pred_counts[:, i] for i in range(n_exps)], 
            positions=range(n_exps), 
            widths=0.6, 
            patch_artist=True, 
            boxprops=dict(facecolor="lightblue", alpha=0.6),
            medianprops=dict(color="navy"),
            showfliers=False 
        )
        
        real_counts = [e['outcome'] for e in self.experiments]
        plt.plot(range(n_exps), real_counts, 'ro', markersize=8, markeredgecolor='black', label='Measured Count')
        
        dose_labels = [f"{e['dose_mean']:.1f} Gy" for e in self.experiments]
        plt.xticks(range(n_exps), dose_labels, rotation=45, ha='right')
        
        plt.ylabel("Colony Count")
        plt.xlabel("Experimental Group (Mean Dose)")
        plt.title(f"Model Validation ({self.name}): Measured vs Predicted")
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def calculate_rbe(self, ref_ld50, ref_ld50_std=0.0):
        """
        Calculates RBE relative to a reference radiation.
        RBE = LD50_ref / LD50_test
        """
        if self.idata is None:
            print("Error: Run .sample() first!")
            return

        print(f"\n--- RBE Calculation (Ref LD50: {ref_ld50} +/- {ref_ld50_std} Gy) ---")
        
        post = self.idata.posterior
        alpha = post['alpha'].values.flatten()
        beta = post['beta'].values.flatten()
        c = -np.log(2)
        
        discriminant = alpha**2 - 4 * beta * c
        valid = discriminant >= 0
        
        ld50_test_samples = (-alpha[valid] + np.sqrt(discriminant[valid])) / (2 * beta[valid])
        
        n_samples = len(ld50_test_samples)
        if ref_ld50_std > 0:
            ld50_ref_samples = np.random.normal(ref_ld50, ref_ld50_std, n_samples)
        else:
            ld50_ref_samples = np.full(n_samples, ref_ld50)
            
        rbe_samples = ld50_ref_samples / ld50_test_samples
        
        mean_rbe = np.mean(rbe_samples)
        hdi = az.hdi(rbe_samples, hdi_prob=0.95)
        
        print(f"RBE (via LD50): {mean_rbe:.2f}")
        print(f"95% Credible Interval: [{hdi[0]:.2f}, {hdi[1]:.2f}]")
        
        plt.figure(figsize=(6, 4))
        plt.hist(rbe_samples, bins=50, density=True, alpha=0.7, color='purple')
        plt.axvline(mean_rbe, color='k', linestyle='--', label=f'Mean: {mean_rbe:.2f}')
        plt.xlabel("RBE Value")
        plt.title("RBE Probability Distribution")
        plt.legend()
        plt.show()
        
        return mean_rbe, hdi

        
    def plot_goodness_of_fit(self):
        """
        Single Model Goodness of Fit Check (Observed vs Predicted).
        """
        if self.idata is None:
            print("ERROR: Model not sampled yet!")
            return
            
        if not hasattr(self.idata, 'posterior_predictive'):
            print("ERROR: Run PPC first (check_posterior_predictive)!")
            return

        print("--- Goodness of Fit Check ---")

        try:
            obs_keys = list(self.idata.observed_data.data_vars.keys())
            
            if 'colony_counts' in obs_keys:
                obs_var = 'colony_counts'
            elif 'obs' in obs_keys:
                obs_var = 'obs'
            else:
                obs_var = obs_keys[0]
                
            obs_counts = self.idata.observed_data[obs_var].values
            
            ppc = self.idata.posterior_predictive
            ppc_keys = list(ppc.data_vars.keys())
            
            if f"{obs_var}_pred" in ppc_keys:
                ppc_var = f"{obs_var}_pred"
            elif obs_var in ppc_keys:
                ppc_var = obs_var
            else:
                ppc_var = ppc_keys[0]
            
            # Reconstruct n_seeded
            n_seeds = None
            if hasattr(self.idata, 'constant_data') and 'n_seeded' in self.idata.constant_data:
                n_seeds = self.idata.constant_data['n_seeded'].values
            elif hasattr(self, 'experiments') and len(self.experiments) > 0:
                try:
                    n_seeds_raw = np.array([e['n_seeded'] for e in self.experiments])
                    if len(n_seeds_raw) == len(obs_counts):
                        n_seeds = n_seeds_raw
                except:
                    pass

            if n_seeds is None:
                print("  WARNING: 'n_seeded' not found. Assuming 2000.")
                n_seeds = np.ones_like(obs_counts) * 2000 
            
            # Calculations
            pe_mean = self.idata.posterior['pe'].mean().item()
            sf_obs = (obs_counts / n_seeds) / pe_mean
            
            # Predicted
            pred_counts_mean = ppc[ppc_var].mean(dim=["chain", "draw"]).values
            sf_pred_mean = (pred_counts_mean / n_seeds) / pe_mean
            
            pred_low = ppc[ppc_var].quantile(0.025, dim=["chain", "draw"]).values
            pred_high = ppc[ppc_var].quantile(0.975, dim=["chain", "draw"]).values
            
            sf_pred_low = (pred_low / n_seeds) / pe_mean
            sf_pred_high = (pred_high / n_seeds) / pe_mean
            
            sf_err_low = np.maximum(sf_pred_mean - sf_pred_low, 0)
            sf_err_high = np.maximum(sf_pred_high - sf_pred_mean, 0)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 7))
            
            all_vals = np.concatenate([sf_obs.flatten(), sf_pred_mean.flatten()])
            valid_vals = all_vals[all_vals > 0]
            if len(valid_vals) == 0:
                min_val, max_val = 0.001, 1.0
            else:
                min_val = valid_vals.min() * 0.5
                max_val = valid_vals.max() * 1.5
                
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal (y=x)')
            
            ax.errorbar(
                sf_obs.flatten(), 
                sf_pred_mean.flatten(), 
                yerr=[sf_err_low.flatten(), sf_err_high.flatten()], 
                fmt='o', color='purple', ecolor='gray', alpha=0.6, 
                label='Observed vs Predicted'
            )
            
            ax.set_title("Goodness of Fit: Observed vs Predicted", fontsize=14, fontweight='bold')
            ax.set_xlabel("Observed Survival (SF)", fontsize=12)
            ax.set_ylabel("Model Predicted (SF)", fontsize=12)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def run_ppc(self):
        """Runs PPC and saves it to idata."""
        if self.idata is None:
            print("ERROR: Run .sample() first!")
            return

        print(f"Generating PPC predictions for '{self.cell_line}'...")
        with self.model:
            pm.sample_posterior_predictive(self.idata, extend_inferencedata=True)
        print("Done.")

# ==============================================================================
# 5. UTILITY & COMPARISON CLASSES
# ==============================================================================

def create_control_dvh_data(n_bins=100, max_dose_ref=8.0):
    """
    Creates dummy DVH data for 0 Gy Control group.
    Forces bin 0 center to be exactly 0.0 Gy.
    """
    bin_edges = np.linspace(0, max_dose_ref, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers[0] = 0.0
    
    weights = np.zeros(n_bins)
    weights[0] = 1.0 
    bin_stds = np.zeros(n_bins)
    
    return {
        'dose_stats': np.vstack([bin_centers, bin_stds]),
        'weights': weights,
        'meta': {
            'voxel_count': 10000,
            'min_dose': 0.0,
            'max_dose': 0.0,
            'mean_dose': 0.0,
            'embryo_smoothing_mm': 0.0
        }
    }

def create_xray_data_point(nominal_dose, relative_uncertainty=0.05, n_bins=100):
    """
    Creates dummy DVH data for homogeneous (X-ray) points.
    """
    doses = np.zeros(n_bins)
    stds = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    
    doses[0] = nominal_dose
    stds[0] = nominal_dose * relative_uncertainty
    weights[0] = 1.0
    
    return {
        'dose_stats': np.vstack([doses, stds]),
        'weights': weights,
        'meta': {
            'voxel_count': 1,
            'min_dose': nominal_dose,
            'max_dose': nominal_dose,
            'mean_dose': nominal_dose,
            'embryo_smoothing_mm': 0.0
        }
    }

class BioModelComparator:
    """
    Compares two biological models (Reference vs Test).
    """
    def __init__(self, model_ref, model_test, names=("Reference", "Test")):
        if model_ref.idata is None or model_test.idata is None:
            raise ValueError("Error: Both models must be sampled first!")
            
        self.m_ref = model_ref
        self.m_test = model_test
        self.names = names
        self.post_ref = model_ref.idata.posterior
        self.post_test = model_test.idata.posterior
        
    def compare_controls(self):
        """Analyzes stability of control group parameters (PE)."""
        pe_ref = self.post_ref['pe'].values.flatten()
        pe_test = self.post_test['pe'].values.flatten()
        
        mean_ref, mean_test = np.mean(pe_ref), np.mean(pe_test)
        
        print(f"\n--- 1. Control Stability (PE) ---")
        print(f"  {self.names[0]} PE: {mean_ref:.3f} (std: {np.std(pe_ref):.3f})")
        print(f"  {self.names[1]} PE: {mean_test:.3f} (std: {np.std(pe_test):.3f})")
        
        rel_diff = abs(mean_ref - mean_test) / mean_ref
        if rel_diff > 0.2: 
            print("  WARNING: Significant difference in PE (>20%)!")
        else:
            print("  OK: Controls are stable (<20% difference).")
            
    def compare_lq_parameters(self):
        """Analyzes consistency of LQ parameters (Alpha, Alpha/Beta)."""
        a_ref = self.post_ref['alpha'].values.flatten()
        b_ref = self.post_ref['beta'].values.flatten()
        
        a_test = self.post_test['alpha'].values.flatten()
        b_test = self.post_test['beta'].values.flatten()
        
        ab_ref = a_ref / (b_ref + 1e-6)
        ab_test = a_test / (b_test + 1e-6)
        
        print(f"\n--- 2. LQ Parameter Consistency ---")
        prob_alpha_higher = np.mean(a_test > a_ref) * 100
        print(f"  Alpha ({self.names[0]}): {np.mean(a_ref):.3f}")
        print(f"  Alpha ({self.names[1]}): {np.mean(a_test):.3f}")
        print(f"  -> Probability Alpha_{self.names[1]} > Alpha_{self.names[0]}: {prob_alpha_higher:.1f}%")
        
        if prob_alpha_higher > 95:
            print("  -> CONSISTENT: Test arm has significantly higher linear component.")
        elif prob_alpha_higher < 5:
            print("  -> CONTRADICTION: Reference arm has significantly higher linear component.")
        else:
            print("  -> OVERLAP: No significant difference in Alpha.")

        print(f"  A/B ({self.names[0]}): {np.mean(ab_ref):.1f} Gy")
        print(f"  A/B ({self.names[1]}): {np.mean(ab_test):.1f} Gy")
        
    def analyze_rbe(self, dose_range=(0.5, 10.0), steps=50):
        """Universal RBE Analysis (works for LQ and LQL)."""
        doses = np.linspace(dose_range[0], dose_range[1], steps)
        
        def get_log_survival(dose, alpha, beta, d_t=None):
            lq = -alpha * dose - beta * (dose**2)
            if d_t is None: return lq
            else:
                log_s_dt = -alpha * d_t - beta * (d_t**2)
                slope = -alpha - 2 * beta * d_t
                lin = log_s_dt + slope * (dose - d_t)
                return np.where(dose < d_t, lq, lin)

        ar = self.post_ref['alpha'].values.flatten()
        br = self.post_ref['beta'].values.flatten()
        at = self.post_test['alpha'].values.flatten()
        bt = self.post_test['beta'].values.flatten()
        
        has_dt = 'd_t' in self.post_ref
        dtr = self.post_ref['d_t'].values.flatten() if has_dt else None
        dtt = self.post_test['d_t'].values.flatten() if has_dt else None

        n_samples = 200 
        indices = np.random.choice(len(ar), n_samples, replace=False)
        rbe_matrix = np.zeros((n_samples, len(doses)))
        
        print(f"\n--- Universal RBE Analysis ({'LQL' if has_dt else 'LQ'}) ---")
        
        for i, idx in enumerate(indices):
            a_r, b_r = ar[idx], br[idx]
            dt_r = dtr[idx] if has_dt else None
            a_t, b_t = at[idx], bt[idx]
            dt_t = dtt[idx] if has_dt else None
            
            for j, d_test in enumerate(doses):
                log_s_target = get_log_survival(d_test, a_t, b_t, dt_t)
                
                def target_func(d):
                    return get_log_survival(d, a_r, b_r, dt_r) - log_s_target
                try:
                    d_ref_sol = brentq(target_func, 0, 100)
                    rbe_matrix[i, j] = d_ref_sol / d_test
                except:
                    rbe_matrix[i, j] = np.nan 

        rbe_mean = np.nanmean(rbe_matrix, axis=0)
        rbe_low = np.nanpercentile(rbe_matrix, 2.5, axis=0)
        rbe_high = np.nanpercentile(rbe_matrix, 97.5, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(doses, rbe_low, rbe_high, color='darkgreen', alpha=0.3, label='95% CI')
        plt.plot(doses, rbe_mean, color='darkgreen', lw=2, label='Mean RBE')
        plt.axhline(1.0, color='gray', linestyle='--')
        
        plt.xlabel(f"Dose ({self.names[1]}) [Gy]")
        plt.ylabel("RBE")
        plt.title(f"RBE Curve: {self.names[1]} vs {self.names[0]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        idx_2Gy = np.argmin(np.abs(doses - 2.0))
        rbe_2Gy = rbe_matrix[:, idx_2Gy]
        prob_gt_1 = np.mean(rbe_2Gy > 1.0) * 100
        print(f"  RBE @ 2.0 Gy: {np.nanmean(rbe_2Gy):.2f} (Prob RBE>1: {prob_gt_1:.1f}%)")
        
    def plot_comparison_dashboard(self, max_dose=10.0):
        """Summary Dashboard."""
        pe_ref = self.post_ref['pe'].values.flatten()
        pe_test = self.post_test['pe'].values.flatten()
        a_ref = self.post_ref['alpha'].values.flatten()
        a_test = self.post_test['alpha'].values.flatten()
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. PE
        az.plot_posterior(pe_ref, ax=axes[0], hdi_prob=0.95, round_to=3, textsize=10, kind='kde', color='navy', label=self.names[0])
        az.plot_posterior(pe_test, ax=axes[0], hdi_prob=0.95, round_to=3, textsize=10, kind='kde', color='darkred', label=self.names[1])
        axes[0].set_title("1. Control Stability (PE)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Plating Efficiency (PE)")
        axes[0].legend(loc='upper right')
        
        # 2. Alpha
        az.plot_posterior(a_ref, ax=axes[1], hdi_prob=0.95, round_to=3, textsize=10, kind='kde', color='navy', label=self.names[0])
        az.plot_posterior(a_test, ax=axes[1], hdi_prob=0.95, round_to=3, textsize=10, kind='kde', color='darkred', label=self.names[1])
        axes[1].set_title("2. Radiosensitivity (Alpha)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Linear Component [1/Gy]")
        axes[1].legend(loc='upper right')
        
        # 3. Survival Curves
        doses = np.linspace(0, max_dose, 100)
        b_ref = self.post_ref['beta'].values.flatten()
        b_test = self.post_test['beta'].values.flatten()
        
        surv_ref = np.exp(-np.mean(a_ref)*doses - np.mean(b_ref)*(doses**2))
        surv_test = np.exp(-np.mean(a_test)*doses - np.mean(b_test)*(doses**2))
        
        n_samples = 500
        idx = np.random.choice(len(a_ref), n_samples)
        
        s_ref_samples = np.exp(-a_ref[idx, None]*doses[None, :] - b_ref[idx, None]*(doses[None, :]**2))
        s_test_samples = np.exp(-a_test[idx, None]*doses[None, :] - b_test[idx, None]*(doses[None, :]**2))
        
        axes[2].plot(doses, surv_ref, color='navy', lw=2.5, label=self.names[0])
        axes[2].fill_between(doses, np.percentile(s_ref_samples, 2.5, axis=0), np.percentile(s_ref_samples, 97.5, axis=0), color='navy', alpha=0.15)
        
        axes[2].plot(doses, surv_test, color='darkred', lw=2.5, label=self.names[1])
        axes[2].fill_between(doses, np.percentile(s_test_samples, 2.5, axis=0), np.percentile(s_test_samples, 97.5, axis=0), color='darkred', alpha=0.15)
        
        min_val = min(np.min(surv_ref), np.min(surv_test))
        lower_limit = max(min_val * 0.5, 1e-5) 
        axes[2].set_yscale('log')
        axes[2].set_ylim(bottom=lower_limit, top=1.2)
        
        axes[2].set_title("3. Survival Curves", fontsize=14, fontweight='bold')
        axes[2].set_xlabel("Dose [Gy]")
        axes[2].set_ylabel("Survival Fraction (SF)")
        axes[2].grid(True, which="both", alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_bivariate_analysis(self):
        """Alpha vs Beta Bivariate Plot."""
        import seaborn as sns 
        
        a_ref = self.post_ref['alpha'].values.flatten()
        b_ref = self.post_ref['beta'].values.flatten()
        a_test = self.post_test['alpha'].values.flatten()
        b_test = self.post_test['beta'].values.flatten()
        
        plt.figure(figsize=(10, 8))
        
        sns.kdeplot(x=a_ref, y=b_ref, cmap="Blues", fill=True, alpha=0.5, thresh=0.05, levels=10, label=f"{self.names[0]} (Ref)")
        sns.kdeplot(x=a_ref, y=b_ref, color="navy", linewidths=1.5, levels=[0.05, 0.32])
        plt.scatter(np.mean(a_ref), np.mean(b_ref), color='navy', s=100, marker='X', zorder=10, edgecolors='white')

        sns.kdeplot(x=a_test, y=b_test, cmap="Reds", fill=True, alpha=0.5, thresh=0.05, levels=10, label=f"{self.names[1]} (Test)")
        sns.kdeplot(x=a_test, y=b_test, color="darkred", linewidths=1.5, levels=[0.05, 0.32])
        plt.scatter(np.mean(a_test), np.mean(b_test), color='darkred', s=100, marker='X', zorder=10, edgecolors='white')

        plt.title("Biological Fingerprint: Alpha vs Beta", fontsize=16, fontweight='bold')
        plt.xlabel(r"Linear Component ($\alpha$) [Gy$^{-1}$]", fontsize=14)
        plt.ylabel(r"Quadratic Component ($\beta$) [Gy$^{-2}$]", fontsize=14)
        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='navy', lw=4, alpha=0.6), Line2D([0], [0], color='darkred', lw=4, alpha=0.6)]
        plt.legend(custom_lines, [self.names[0], self.names[1]], loc='upper right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_difference_distribution(self, parameter='alpha', rope=None, color_diff=True):
        """Difference Plot."""
        val_ref = self.post_ref[parameter].values.flatten()
        val_test = self.post_test[parameter].values.flatten()
        
        diff = val_test - val_ref
        prob_gt_0 = np.mean(diff > 0)
        prob_lt_0 = np.mean(diff < 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        az.plot_posterior(diff, hdi_prob=0.95, kind='kde', color='gray', point_estimate='mean', textsize=12, ax=ax, show=False)
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='No Difference')
        
        if color_diff:
            line = ax.get_lines()[0]
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
            ax.fill_between(x_data, y_data, where=(x_data >= 0), color='darkred', alpha=0.3, label=f'{self.names[1]} > {self.names[0]}')
            ax.fill_between(x_data, y_data, where=(x_data < 0), color='navy', alpha=0.3, label=f'{self.names[1]} < {self.names[0]}')

        if rope is not None:
            ax.axvline(rope[0], color='green', linestyle=':', alpha=0.6)
            ax.axvline(rope[1], color='green', linestyle=':', alpha=0.6)
            ax.axvspan(rope[0], rope[1], color='green', alpha=0.1, label='Region of Practical Equivalence')
            
            prob_in_rope = np.mean((diff > rope[0]) & (diff < rope[1]))
            ax.text(0.5, 0.05, f"In ROPE: {prob_in_rope*100:.1f}%", transform=ax.transAxes, ha='center', color='green', fontweight='bold')

        ax.set_title(f"Difference Analysis: $\Delta${parameter}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(f"Difference ({self.names[1]} - {self.names[0]})", fontsize=13)
        ax.set_ylabel("Probability Density", fontsize=13)
        
        stats_msg = (
            f"Probabilities:\n----------------\n"
            f"P({self.names[1]} > {self.names[0]}): {prob_gt_0*100:.1f}%\n"
            f"P({self.names[1]} < {self.names[0]}): {prob_lt_0*100:.1f}%"
        )
        ax.text(0.02, 0.95, stats_msg, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9))

        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        low_lim = np.percentile(diff, 0.5)
        high_lim = np.percentile(diff, 99.5)
        pad = (high_lim - low_lim) * 0.1
        ax.set_xlim(low_lim - pad, high_lim + pad)

        plt.tight_layout()
        plt.show()
        
    def plot_goodness_of_fit(self):
        """PPC Plot for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        models_to_plot = [(self.m_ref, self.names[0], 'navy', axes[0]), (self.m_test, self.names[1], 'darkred', axes[1])]
        
        for model, name, color, ax in models_to_plot:
            if model.idata is None or not hasattr(model.idata, 'posterior_predictive'):
                print(f"ERROR ({name}): No PPC data. Run: pm.sample_posterior_predictive(...)")
                continue

            try:
                # Dinamikus változónév-keresés (így az 'obs' és a 'colony_counts' is működik)
                obs_keys = list(model.idata.observed_data.data_vars.keys())
                obs_var = 'colony_counts' if 'colony_counts' in obs_keys else ('obs' if 'obs' in obs_keys else obs_keys[0])
                obs_counts = model.idata.observed_data[obs_var].values
                
                # Sejtszám visszakeresése
                if hasattr(model.idata, 'constant_data') and 'n_seeded' in model.idata.constant_data:
                    n_seeds = model.idata.constant_data['n_seeded'].values
                elif hasattr(model, 'experiments') and len(model.experiments) > 0:
                    n_seeds = np.array([e['n_seeded'] for e in model.experiments])
                else:
                    n_seeds = np.ones_like(obs_counts) * 2000
                
                pe_mean = model.idata.posterior['pe'].mean().item()
                sf_obs = (obs_counts / n_seeds) / pe_mean
                
                ppc = model.idata.posterior_predictive
                ppc_keys = list(ppc.data_vars.keys())
                ppc_var = f"{obs_var}_pred" if f"{obs_var}_pred" in ppc_keys else (obs_var if obs_var in ppc_keys else ppc_keys[0])
                
                pred_counts_mean = ppc[ppc_var].mean(dim=["chain", "draw"]).values
                sf_pred = (pred_counts_mean / n_seeds) / pe_mean
                
                pred_low = ppc[ppc_var].quantile(0.025, dim=["chain", "draw"]).values
                pred_high = ppc[ppc_var].quantile(0.975, dim=["chain", "draw"]).values
                
                sf_err_low = np.maximum(sf_obs - ((pred_low / n_seeds) / pe_mean), 0)
                sf_err_high = np.maximum(((pred_high / n_seeds) / pe_mean) - sf_obs, 0)
                
                all_vals = np.concatenate([sf_obs.flatten(), sf_pred.flatten()])
                valid_vals = all_vals[all_vals > 0]
                if len(valid_vals) == 0: min_val, max_val = 0.001, 1.0
                else: min_val, max_val = valid_vals.min() * 0.5, valid_vals.max() * 1.5
                
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal (y=x)')
                
                ax.errorbar(sf_obs.flatten(), sf_pred.flatten(), yerr=[sf_err_low.flatten(), sf_err_high.flatten()], fmt='o', color=color, ecolor='gray', alpha=0.6, label='Observed vs Predicted')
                
                ax.set_title(f"Goodness of Fit: {name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Observed Survival (SF)", fontsize=12)
                ax.set_ylabel("Model Predicted (SF)", fontsize=12)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val)
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
                
            except Exception as e:
                print(f"ERROR ({name}): Data error in idata: {e}")

        plt.tight_layout()
        plt.show()
        
    def compare_model_fit(self):
        """LOO comparison between Reference and Test models."""
        comp = az.compare({
            self.names[0]: self.m_ref.idata, 
            self.names[1]: self.m_test.idata
        }, ic="loo", scale="deviance")
        print("\n--- Model Comparison (LOO) ---")
        print(comp)
        az.plot_compare(comp)
        plt.show()

class FishModelComparator:
    """Compare two Fish (Weibull) models."""
    def __init__(self, model_ref, model_test, names=("Reference", "Test")):
        if model_ref.idata is None or model_test.idata is None:
            raise ValueError("Error: Run .sample() first!")
        self.m_ref = model_ref
        self.m_test = model_test
        self.names = names
        self.post_ref = model_ref.idata.posterior
        self.post_test = model_test.idata.posterior

    def compare_sensitivity(self):
        """Compare beta_dose (sensitivity)."""
        b_ref = self.post_ref['beta_dose'].values.flatten()
        b_test = self.post_test['beta_dose'].values.flatten()
        
        mean_ref, mean_test = np.mean(b_ref), np.mean(b_test)
        prob_higher = np.mean(b_test > b_ref) * 100
        
        print(f"\n--- Dose Sensitivity (Beta_dose) Comparison ---")
        print(f"  {self.names[0]} Beta: {mean_ref:.3f}")
        print(f"  {self.names[1]} Beta: {mean_test:.3f}")
        print(f"  -> Probability {self.names[1]} is more sensitive: {prob_higher:.1f}%")
        
        if prob_higher > 95:
            print("  -> SIGNIFICANT: Test radiation kills more effectively.")
        else:
            print("  -> NO significant difference.")

    def compare_baseline_hazard(self):
        """Check control stability (lambda_base)."""
        l_ref = self.post_ref['lambda_base'].values.flatten()
        l_test = self.post_test['lambda_base'].values.flatten()
        print(f"\n--- Control Stability (Baseline Hazard) ---")
        print(f"  {self.names[0]} Lambda: {np.mean(l_ref):.4f}")
        print(f"  {self.names[1]} Lambda: {np.mean(l_test):.4f}")
        
        diff = abs(np.mean(l_ref) - np.mean(l_test)) / np.mean(l_ref)
        if diff > 0.3: print("  WARNING: Control hazard rates differ!")
        else: print("  OK: Controls are stable.")

    def analyze_rbe_at_day(self, day=4.0, dose_range=(0.0, 25.0)):
        """Calculate RBE for a specific day based on LD50."""
        br = self.post_ref['beta_dose'].values.flatten()
        lr = self.post_ref['lambda_base'].values.flatten()
        kr = self.post_ref['shape_k'].values.flatten()
        
        bt = self.post_test['beta_dose'].values.flatten()
        lt = self.post_test['lambda_base'].values.flatten()
        kt = self.post_test['shape_k'].values.flatten()
        
        def calculate_ld50(beta, lam, k, t):
            denom = lam * (t**k)
            denom = np.maximum(denom, 1e-9)
            val = np.log(np.log(2) / denom) / beta
            return np.maximum(val, 0)
            
        ld50_ref = calculate_ld50(br, lr, kr, day)
        ld50_test = calculate_ld50(bt, lt, kt, day)
        
        valid_idx = (ld50_ref < 100) & (ld50_test < 100) & (ld50_test > 0.1)
        rbe_dist = ld50_ref[valid_idx] / ld50_test[valid_idx]
        
        print(f"\n--- RBE Analysis (Day {day}) ---")
        print(f"  LD50 ({self.names[0]}): {np.mean(ld50_ref):.2f} Gy")
        print(f"  LD50 ({self.names[1]}): {np.mean(ld50_test):.2f} Gy")
        
        mean_rbe = np.mean(rbe_dist)
        hpd_low = np.percentile(rbe_dist, 2.5)
        hpd_high = np.percentile(rbe_dist, 97.5)
        
        print(f"  Estimated RBE: {mean_rbe:.2f} (95% CI: {hpd_low:.2f} - {hpd_high:.2f})")
        
        doses = np.linspace(dose_range[0], dose_range[1], 100)
        
        exponent_r = -np.mean(lr) * np.exp(np.mean(br) * doses) * (day**np.mean(kr))
        surv_r = np.exp(exponent_r)
        
        exponent_t = -np.mean(lt) * np.exp(np.mean(bt) * doses) * (day**np.mean(kt))
        surv_t = np.exp(exponent_t)
        
        plt.figure(figsize=(8, 6))
        plt.plot(doses, surv_r, 'b-', lw=2, label=f'{self.names[0]} (Ref)')
        plt.plot(doses, surv_t, 'r-', lw=2, label=f'{self.names[1]} (Test)')
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='LD50 level')
        plt.xlabel("Dose [Gy]")
        plt.ylabel(f"Survival Fraction (Day {day})")
        plt.title(f"Fish Model Comparison (Day: {day})")
        plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1.05)
        plt.show()