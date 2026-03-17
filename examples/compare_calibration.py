import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from prism.dosimetry import GafchromicEngine

# --- SEGÉDFÜGGVÉNY: Inverz Mendez-Lewis (Támogatja az 'n' paramétert is) ---
def calc_dose_from_od(od_array, a, b, c, n=1.0):
    """
    Kiszámítja a dózist az optikai sűrűségből (OD).
    Képlet: D = [ c * (OD - a) / (a + b - OD) ]^(1/n)
    """
    od_array = np.array(od_array)
    # Numerikus védelem: a nevező ne legyen nulla vagy negatív
    denominator = np.maximum(a + b - od_array, 1e-6)
    
    # Alap kiszámítása
    base = c * (od_array - a) / denominator
    base = np.maximum(base, 0.0)  # Ne vonjunk negatív számból gyököt
    
    return base ** (1.0 / n)

def plot_calibration_comparison(calib_folder, doses_gy, channel='R'):
    """
    Összehasonlítja a klasszikus és a Bayes-i filmkalibrációt.
    """
    engine = GafchromicEngine()
    
    # --- 1. Klasszikus Illesztés ---
    print(f"Futtatás: Klasszikus Curve Fit ({channel} csatorna)...")
    engine.run_calibration(calib_folder, doses_gy)
    params_std = engine.calib_params[channel].copy()
    
    # Hibaszámítás (Visszaszámolt dózisok)
    ods_measured = np.array(engine.calib_ods[channel])
    doses_gy_arr = np.array(doses_gy)
    
    # Itt a *params_std kibontja mind a 3 vagy 4 paramétert (a, b, c, [n])
    doses_calc_std = calc_dose_from_od(ods_measured, *params_std)
    res_std = doses_calc_std - doses_gy_arr
    rmse_std = np.sqrt(np.mean(res_std**2))
    
    # --- 2. Bayes-i Illesztés ---
    print("\nFuttatás: Bayes-i MCMC...")
    engine.run_mcmc_calibration(calib_folder, doses_gy, use_cleaning=False)
    trace = engine.calib_traces[channel]
    
    # Poszterior paraméterek kinyerése
    post = trace.posterior
    a_samples = post['a'].values.flatten()
    b_samples = post['b'].values.flatten()
    c_samples = post['c'].values.flatten()
    
    # Ellenőrizzük, hogy illesztett-e 'n' paramétert is a Bayes-i modell
    if 'n' in post:
        n_samples = post['n'].values.flatten()
    else:
        n_samples = np.ones_like(a_samples) # Ha nem, akkor n=1.0
    
    # Hibaszámítás (Átlagos poszterior alapján)
    a_mean, b_mean, c_mean, n_mean = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples), np.mean(n_samples)
    doses_calc_mcmc = calc_dose_from_od(ods_measured, a_mean, b_mean, c_mean, n_mean)
    res_mcmc = doses_calc_mcmc - doses_gy_arr
    rmse_mcmc = np.sqrt(np.mean(res_mcmc**2))
    
    print(f"\n--- EREDMÉNYEK ({channel} CSATORNA) ---")
    print(f"Klasszikus RMSE: {rmse_std:.4f} Gy")
    print(f"Bayes-i RMSE:    {rmse_mcmc:.4f} Gy")

    # --- 3. Ábrázolás ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    d_axis = np.linspace(0, max(doses_gy) * 1.1, 200)
    
    # A) Klasszikus görbe
    od_std_curve = engine.rational_func_od(d_axis, *params_std)
    ax1.plot(d_axis, od_std_curve, 'k--', lw=2, label=f'Classic Fit (RMSE: {rmse_std:.2f} Gy)')
    
    # B) Bayes-i görbe és HDI (Credible Interval)
    # Generáljuk az OD értékeket minden egyes MCMC mintára (ritkítva a sebességért)
    od_mcmc_samples = np.array([engine.rational_func_od(d_axis, a, b, c, n) 
                                for a, b, c, n in zip(a_samples[::10], b_samples[::10], c_samples[::10], n_samples[::10])])
    
    od_mcmc_mean = np.mean(od_mcmc_samples, axis=0)
    hdi_95 = az.hdi(od_mcmc_samples, hdi_prob=0.95)
    
    ax1.fill_between(d_axis, hdi_95[:, 0], hdi_95[:, 1], color='red', alpha=0.3, label='Bayesian 95% HDI')
    ax1.plot(d_axis, od_mcmc_mean, 'r-', lw=2, label=f'Bayesian Mean (RMSE: {rmse_mcmc:.2f} Gy)')
    
    # C) Mérési pontok
    ax1.plot(doses_gy, ods_measured, 'bo', markersize=8, label='Measured Data')
    
    ax1.set_ylabel('Net Optical Density (OD)', fontsize=12)
    ax1.set_title('Calibration Curve Comparison: Frequentist vs. Bayesian', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=11)
    
    # --- 4. Reziduálisok Ábrázolása (Alsó panel) ---
    ax2.axhline(0, color='gray', linestyle='-', lw=1)
    
    # A pontokat kissé eltoljuk az X tengelyen, hogy ne fedjék egymást a ploton
    offset = max(doses_gy) * 0.015
    
    ax2.errorbar(doses_gy_arr - offset, res_std, fmt='ks', markersize=6, label='Classic Residuals')
    ax2.errorbar(doses_gy_arr + offset, res_mcmc, fmt='ro', markersize=6, label='Bayesian Residuals')
    
    ax2.set_xlabel('Nominal Dose [Gy]', fontsize=12)
    ax2.set_ylabel('$\Delta$ Dose [Gy]', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=300)
    print("\nÁbra elmentve: calibration_comparison.png")
    plt.show()

# --- FUTTATÁS ---
if __name__ == "__main__":
    cal_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502X-RAY-2"
    cal_doses = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 36.0, 45.0]

    plot_calibration_comparison(calib_folder=cal_folder, doses_gy=cal_doses, channel='R')