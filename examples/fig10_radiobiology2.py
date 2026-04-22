"""
NOTE: This script generates Figure 7, which serves as a purely conceptual, 
pedagogical demonstration of "Cold-Spot Sheltering" and "False Radioresistance".
To keep the visual demonstration mathematically clean and isolated from sampling 
noise, this script uses synthetic bimodal dose distributions and deterministic 
curve fitting (scipy.optimize.curve_fit) rather than the full Bayesian PRISM pipeline.

For the actual experimental validation using the PRISM PyMC MCMC framework, 
please refer to `radiobiology3.py` (Figure 10).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit



def generate_fig7_synthetic_composite():
    print("--- 7. Ábra: Szintetikus dDVH és Sejttúlélési Modell ---")
    np.random.seed(42)

    # =========================================================================
    # 1. SZINTETIKUS "CSÚNYA" dDVH GENERÁLÁSA (Hidegfolt dominancia)
    # =========================================================================
    # Egy aszimmetrikus eloszlás: 70% megkapja az előírt dózist, de 30% hidegfoltban van
    x_base = np.linspace(0.4, 1.4, 200)
    w_base = 0.7 * np.exp(-0.5 * ((x_base - 1.05) / 0.08)**2) + \
             0.3 * np.exp(-0.5 * ((x_base - 0.75) / 0.15)**2)
    w_base = w_base / np.sum(w_base)
    
    # Skálázzuk úgy, hogy az eloszlás átlaga PONTOSAN 1.0 Gy legyen!
    mean_base = np.sum(w_base * x_base)
    x_norm = x_base / mean_base 

    prescribed_doses = np.array([0, 2, 4, 6, 8, 10, 12])

    # =========================================================================
    # 2. SUGÁRBIOLÓGIAI PARAMÉTEREK ÉS SZIMULÁCIÓ
    # =========================================================================
    true_alpha = 0.30  
    true_beta  = 0.030 
    
    def lq_model(D, alpha, beta):
        return np.exp(-alpha * D - beta * D**2)

    mean_doses = []
    prism_eud_doses = []
    measured_survivals = []
    
    # Adatok kiszámolása minden előírt dózisra
    for D_target in prescribed_doses:
        if D_target == 0:
            mean_doses.append(0.0)
            prism_eud_doses.append(0.0)
            measured_survivals.append(1.0)
            continue
            
        # Dózis bin-ek felskálázása
        current_bins = x_norm * D_target
        current_mean = np.sum(w_base * current_bins) # Ez így pont D_target lesz
        mean_doses.append(current_mean)
        
        # Valós túlélés a hidegfoltok miatt (Integrálás a teljes térfogatra)
        S_true_total = np.sum(w_base * lq_model(current_bins, true_alpha, true_beta))
        
        # 3% kísérleti zaj (hogy a fittelés ne legyen "túl tökéletes")
        S_noisy = S_true_total * np.random.normal(1.0, 0.03)
        measured_survivals.append(S_noisy)
        
        # PRISM EUD kiszámolása (az inverz LQ függvénnyel)
        eud = (-true_alpha + np.sqrt(true_alpha**2 - 4 * true_beta * np.log(S_true_total))) / (2 * true_beta)
        prism_eud_doses.append(eud)
        
    mean_doses = np.array(mean_doses)
    prism_eud_doses = np.array(prism_eud_doses)
    survivals = np.array(measured_survivals)

    # =========================================================================
    # 3. GÖRBEILLESZTÉSEK (FITTING)
    # =========================================================================
    # Hagyományos fittelés a Mean Dose alapján (Hibás lesz!)
    popt_trad, _ = curve_fit(lq_model, mean_doses, survivals, p0=[0.2, 0.02])
    alpha_trad, beta_trad = popt_trad

    # PRISM fittelés az EUD alapján (Visszaadja a valósat)
    popt_prism, _ = curve_fit(lq_model, prism_eud_doses, survivals, p0=[0.2, 0.02])
    alpha_prism, beta_prism = popt_prism
    
    d_plot = np.linspace(0, 13, 200)
    surv_trad_plot = lq_model(d_plot, alpha_trad, beta_trad)
    surv_prism_plot = lq_model(d_plot, alpha_prism, beta_prism)

    # =========================================================================
    # 4. ÁBRA RAJZOLÁSA (1x2 KOMPOZIT)
    # =========================================================================
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1, 1.3]) # A túlélési ábra egy kicsit szélesebb

    color_trad = 'crimson'
    color_prism = 'navy'

    # --- PANEL A: HISZTOGRAMOK (dDVH) ---
    ax1 = fig.add_subplot(gs[0])
    
    # Rajzoljuk ki a 2, 6, 10 Gy-es hisztogramokat (hogy ne legyen zsúfolt)
    colors = ['#FFb347', '#FF7F50', '#c23b22']
    plot_doses = [2, 6, 10]
    
    for D_target, c in zip(plot_doses, colors):
        bins = x_norm * D_target
        # Skálázzuk a magasságot a vizualizációhoz
        ax1.fill_between(bins, 0, w_base, color=c, alpha=0.4, label=f'Prescribed: {D_target} Gy')
        ax1.plot(bins, w_base, color=c, lw=2)
        
        # Rajzoljuk be a Mean Dose helyét egy szaggatott vonallal
        ax1.axvline(D_target, color='black', linestyle='--', alpha=0.5)

    ax1.set_title("A. Synthetic Dose-Volume Histograms (dDVH)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Absorbed Dose [Gy]", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Relative Volume (Probability)", fontsize=12, fontweight='bold')
    ax1.set_yticks([]) # Y-tengely számait elrejtjük, mert csak az alak a fontos
    
    # Annotáció a hidegfoltokhoz
    ax1.annotate('Cold Spot\nTail', xy=(1.5, np.max(w_base)*0.1), xytext=(0.5, np.max(w_base)*0.3),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=11, fontweight='bold')
                 
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- PANEL B: SEJTTÚLÉLÉS ÉS PARAMÉTEREK ---
    ax2 = fig.add_subplot(gs[1])

    # Hagyományos (Mean Dose)
    ax2.plot(d_plot, surv_trad_plot, linestyle='--', color=color_trad, lw=2.5, label='Fit on Mean Dose')
    ax2.plot(mean_doses, survivals, 's', color=color_trad, markeredgecolor='black', markersize=9, zorder=5)

    # PRISM (EUD)
    ax2.fill_between(d_plot, surv_prism_plot * np.exp(-0.06 * d_plot), surv_prism_plot * np.exp(0.06 * d_plot), 
                    color='dodgerblue', alpha=0.2)
    ax2.plot(d_plot, surv_prism_plot, linestyle='-', color=color_prism, lw=2.5, label='Fit on PRISM EUD')
    ax2.plot(prism_eud_doses, survivals, 'o', color='dodgerblue', markeredgecolor='black', markersize=9, zorder=6)

    # Nyilak
    for i in range(1, len(mean_doses)):
        ax2.annotate("", xy=(prism_eud_doses[i]+0.15, survivals[i]), 
                     xytext=(mean_doses[i]-0.15, survivals[i]),
                     arrowprops=dict(arrowstyle="->", color='gray', lw=2, ls='solid'))

    ax2.text(12, survivals[-1]*3.5, 'EUD Shift', color='gray', fontsize=11, fontweight='bold', ha='center')

    # PARAMÉTER ÖSSZEHASONLÍTÓ SZÖVEGDOBOZ
    param_text = (
        "Parameter Recovery Comparison\n"
        "-------------------------------------\n"
        f"Ground Truth : $\\alpha={true_alpha:.3f}$, $\\beta={true_beta:.3f}$\n"
        f"EUD Fit: $\\alpha={alpha_prism:.3f}$, $\\beta={beta_prism:.3f}$\n"
        f"Mean Dose Fit: $\\alpha={alpha_trad:.3f}$, $\\beta={beta_trad:.3f}$"
    )
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.05, 0.25, param_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='bottom', bbox=props, fontfamily='monospace')

    # Esztétika
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1.2)
    ax2.set_xlim(-0.5, 13)
    
    ax2.set_title("B. Radiobiological Bias: False Radioresistance", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Dose [Gy]", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Surviving Fraction", fontsize=12, fontweight='bold')
    
    ax2.grid(True, which="major", linestyle='-', alpha=0.5)
    ax2.grid(True, which="minor", linestyle=':', alpha=0.3)
    ax2.legend(loc='lower left', fontsize=11)

    filename = "fig10_synthetic_composite.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    generate_fig7_synthetic_composite()