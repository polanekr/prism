import numpy as np
import matplotlib.pyplot as plt
import os
import arviz as az

# Importáljuk a meglévő motorodat a dosimetry.py-ból
from prism.dosimetry import GafchromicEngine

def generate_figure_2(calib_folder: str, doses_gy: list, channel: str = 'red'):
    """
    Legenerálja a 2. Ábrát (Frequentist vs Bayesian Calibration) a cikkhez.
    """
    print("--- 2. Ábra Generálása: Kalibrációs Összehasonlítás ---")
    engine = GafchromicEngine(output_folder="results_fig2")
    ch = channel[0].upper()

    # ==========================================
    # 1. Klasszikus (Frequentist) Kalibráció
    # ==========================================
    print("\n1. Klasszikus curve_fit futtatása...")
    engine.run_calibration(calib_folder, doses_gy)
    params_std = engine.calib_params[ch].copy()
    
    # Reziduálisok kinyerése (Számított dózisok visszaszámolása)
    calc_doses_std, _ = engine.validate_calibration_curve(channel=channel, plot_result=False)
    # Reziduális = Mért (Nominális) - Számított
    res_std = np.array(doses_gy) - np.array(calc_doses_std)

    # ==========================================
    # 2. Bayes-i (MCMC) Kalibráció
    # ==========================================
    print("\n2. Bayes-i MCMC futtatása (ez eltarthat egy percig)...")
    # A tisztítást (use_cleaning) kikapcsoljuk, hogy gyorsabb legyen, 
    # de bekapcsolhatod, ha zajosak a kalibrációs filmek.
    engine.run_mcmc_calibration(calib_folder, doses_gy, draws=2000, chains=2, use_cleaning=False)
    params_mcmc = engine.calib_params[ch].copy()
    trace = engine.calib_traces[ch]
    
    # Reziduálisok kinyerése
    calc_doses_mcmc, _ = engine.validate_calibration_curve(channel=channel, plot_result=False)
    res_mcmc = np.array(doses_gy) - np.array(calc_doses_mcmc)

    # ==========================================
    # 3. Publikációkész Ábra Rajzolása
    # ==========================================
    print("\n3. Ábra rajzolása...")
    
    # Tengelyek és elrendezés beállítása (Felső 3 rész, alsó 1 rész)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)
    
    doses = np.array(doses_gy)
    meas_ods = np.array(engine.calib_ods[ch])
    d_plot = np.linspace(0, max(doses) * 1.15, 300) # Sima görbéhez sűrű X tengely

    # --- FELSŐ PANEL: OD vs Dózis ---
    
    # 1. Klasszikus görbe
    od_std = engine.rational_func_od(d_plot, *params_std)
    ax1.plot(d_plot, od_std, 'k--', lw=2.5, label='Frequentist Fit (Non-Linear Least Squares)')

    # 2. MCMC HDI Sáv és Átlag számítása
    post = trace.posterior
    a_s = post['a'].values.flatten()
    b_s = post['b'].values.flatten()
    c_s = post['c'].values.flatten()
    
    # Minta kiválasztása a poszteriorból a görbékhez
    idxs = np.random.choice(len(a_s), size=min(1000, len(a_s)), replace=False)
    A = a_s[idxs][:, None]; B = b_s[idxs][:, None]; C = c_s[idxs][:, None]
    D = d_plot[None, :]
    
    # Racionális függvény: OD = a + (b*D)/(D+c)
    curves = A + (B * D) / (D + C)
    od_mcmc_mean = np.mean(curves, axis=0)
    hdi_low = np.percentile(curves, 2.5, axis=0)
    hdi_high = np.percentile(curves, 97.5, axis=0)

    # HDI Sáv rajzolása (kék áttetsző)
    ax1.fill_between(d_plot, hdi_low, hdi_high, color='dodgerblue', alpha=0.3, 
                     label='Bayesian 95% Highest Density Interval (HDI)')
    
    # MCMC Átlag görbe
    ax1.plot(d_plot, od_mcmc_mean, color='navy', lw=2.5, label='Bayesian MCMC Mean')

    # 3. Mérési Pontok
    ax1.scatter(doses, meas_ods, color='darkorange', edgecolor='black', s=80, zorder=5, 
                label='Measured Film Data')

    # Felső panel esztétika
    ax1.set_ylabel("Optical Density (OD)", fontsize=13, fontweight='bold')
    #ax1.set_title("Calibration Curves: Frequentist vs. Hierarchical Bayesian Approach", 
    #              fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- ALSÓ PANEL: Reziduálisok ---
    
    # Nullvonal
    ax2.axhline(0, color='black', linestyle='-', lw=1.5, alpha=0.6)
    
    # Klasszikus reziduálisok (Szürke/Fekete)
    ax2.plot(doses, res_std, linestyle='--', color='gray', marker='o', 
             markerfacecolor='dimgray', markeredgecolor='black',
             lw=1.5, markersize=7, label='Frequentist Residuals')
    
    # Bayes-i reziduálisok (Kék)
    ax2.plot(doses, res_mcmc, linestyle='-', color='navy', marker='s', 
             markerfacecolor='dodgerblue', markeredgecolor='black',
             lw=2, markersize=7, label='Bayesian Residuals')

    # Alsó panel esztétika
    ax2.set_xlabel("Nominal Dose [Gy]", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Dose Error [Gy]\n(Measured - Calculated)", fontsize=11)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # X tengely finomhangolása
    ax2.set_xlim(-0.5, max(doses) * 1.15)

    # Ábra mentése és megjelenítése
    plt.tight_layout()
    filename = "fig2_calibration_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    # =========================================================================
    # BEÁLLÍTÁSOK
    # Add meg a kalibrációs filmjeidet tartalmazó mappát és a hozzájuk tartozó dózisokat!
    # (Fontos: A mappában a filmek névsor szerint (ABC) legyenek sorrendben a dózisokkal!)
    # =========================================================================
    
    my_calib_folder = "/home/polanekr/Kutatás/esylos-dosimetry/data/raw/measurements/film_calibrations/02032502X-RAY-2"
    
    # Példa dózisok (írd át a sajátjaidra! Az első általában 0 a Blank filmhez)
    my_doses = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 36.0, 45.0] 
    
    # Futtatás
    generate_figure_2(calib_folder=my_calib_folder, doses_gy=my_doses, channel='red')