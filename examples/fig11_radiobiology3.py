import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# =========================================================================
# PRISM MODULOK IMPORTÁLÁSA
# =========================================================================
from prism.biology import CellSurvivalLQModel

def generate_experimental_2x2_hybrid(npy_files: dict):
    print("--- Kísérleti Validációs Ábra (2x2) Hibrid Illesztéssel ---")
    np.random.seed(42)

    # A sejtek IGAZI paraméterei (ezt kell a modelleknek visszatalálni)
    true_alpha = 0.30  
    true_beta  = 0.030 
    true_pe    = 0.50  

    def lq_model_simple(D, alpha, beta):
        return np.exp(-alpha * D - beta * D**2)

    nominal_doses = [0.0] + sorted(list(npy_files.keys()))
    valid_doses = [d for d in nominal_doses if d > 0]
    
    max_dose_val = max(valid_doses) if valid_doses else 15.0
    
    # JAVÍTÁS 1: A vonalakat csak 5%-kal engedjük túl a max dózison, 
    # de a dobozt (axis) 15%-kal nagyobbra vesszük, így a vonal nem fog nekimenni a falnak!
    max_dose_curve = max_dose_val * 1.05 
    axis_x_limit = max_dose_val * 1.15

    # Kiderítjük a bin-ek számát a valós fájlokból
    n_bins = 50 
    for filepath in npy_files.values():
        if os.path.exists(filepath):
            try:
                data = np.load(filepath, allow_pickle=True).item()
                n_bins = len(data['weights'])
                break
            except Exception:
                pass

    # =========================================================================
    # 1. ADATOK ELŐKÉSZÍTÉSE
    # =========================================================================
    real_data = {}
    narrow_data = {}
    
    for dose in nominal_doses:
        if dose == 0.0:
            x_arr = np.linspace(0, 0.1, n_bins)
            w_arr = np.zeros(n_bins); w_arr[0] = 1.0
        else:
            filepath = npy_files.get(dose, "")
            if os.path.exists(filepath):
                data = np.load(filepath, allow_pickle=True).item()
                x_arr = data['dose_stats'][0, :]
                w_arr = data['weights']
                w_arr /= np.sum(w_arr) 
            else:
                x_arr = np.linspace(dose-0.1, dose+0.1, n_bins)
                w_arr = np.zeros(n_bins); w_arr[n_bins//2] = 1.0

        real_data[dose] = {'x': x_arr, 'w': w_arr}
        
        # Enyhébb szűkítés (sigma = 0.60 Gy) a szélesebb harang alakért
        peak_x = x_arr[np.argmax(w_arr)]
        w_narrow = w_arr * np.exp(-0.5 * ((x_arr - peak_x) / 0.60)**2)
        w_narrow /= np.sum(w_narrow)
        narrow_data[dose] = {'x': x_arr, 'w': w_narrow}

    # =========================================================================
    # 2. ÁBRA RAJZOLÁSA ÉS ILLESZTÉSEK (2x2 GRID)
    # =========================================================================
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = GridSpec(2, 2, width_ratios=[1, 1.3])

    color_trad = 'crimson'
    color_prism = 'navy'
    hist_colors_real = ['#FFb347', '#FF7F50', '#c23b22', '#8b0000']
    hist_colors_narrow = ['#a1d99b', '#74c476', '#31a354', '#006d2c']

    # Vonalrajzolás vektor
    d_plot = np.linspace(0, max_dose_curve, 200)

    for row_idx in range(2):
        print(f"\n================ ROW {row_idx+1} PROCESSING ================")
        
        prism_model = CellSurvivalLQModel(cell_line="Generic", name=f"PRISM_Row{row_idx}")

        colonies_list = []
        n_seeded_list = []
        mean_doses_physical = []

        for D_target in nominal_doses:
            bins = real_data[D_target]['x'] if row_idx == 0 else narrow_data[D_target]['x']
            weights = real_data[D_target]['w'] if row_idx == 0 else narrow_data[D_target]['w']
            stds = np.ones_like(bins) * 0.05 

            # JAVÍTÁS 2: Dinamikus sejtszám a laborgyakorlat szerint!
            # Megakadályozza, hogy a magas dózisoknál a pont 0-ba (mínusz végtelenbe) hulljon.
            if D_target < 3: n_seeded = 2000
            elif D_target < 6: n_seeded = 10000
            elif D_target < 10: n_seeded = 50000
            elif D_target < 13: n_seeded = 200000
            else: n_seeded = 1000000

            S_true = np.sum(weights * np.exp(-true_alpha * bins - true_beta * (bins**2)))
            expected_colonies = n_seeded * true_pe * S_true
            
            colonies = np.random.poisson(expected_colonies)
            # Matematikai biztonsági háló (hogy sohase lehessen 0)
            if colonies <= 0: colonies = 1 
                
            colonies_list.append(colonies)
            n_seeded_list.append(n_seeded)
            mean_doses_physical.append(np.sum(bins * weights))

            hist_dict = {
                'dose_stats': np.vstack([bins, stds]),
                'weights': weights
            }
            prism_model.add_experiment_from_histogram(hist_dict, colonies, n_seeded)

        print("  -> Futtatás: Hagyományos (Mean Dose) Curve Fit...")
        pe_est = colonies_list[0] / n_seeded_list[0]
        sf_observations = np.array(colonies_list) / (np.array(n_seeded_list) * pe_est)
        
        try:
            popt_trad, _ = curve_fit(lq_model_simple, mean_doses_physical, sf_observations, p0=[0.2, 0.02])
            mean_a_t, mean_b_t = popt_trad[0], popt_trad[1]
        except:
            mean_a_t, mean_b_t = 0.2, 0.02

        surv_trad_plot = lq_model_simple(d_plot, mean_a_t, mean_b_t)

        print("  -> Futtatás: PRISM (Volumetrikus) PyMC Modell...")
        prism_model.sample(draws=1000, tune=500, chains=2)

        post_prism = prism_model.idata.posterior
        a_p = post_prism['alpha'].values.flatten()
        b_p = post_prism['beta'].values.flatten()
        pe_p = post_prism['pe'].values.flatten()
        mean_a_p, mean_b_p, mean_pe_p = np.mean(a_p), np.mean(b_p), np.mean(pe_p)

        surv_samples_prism = np.exp(-a_p[:, None] * d_plot[None, :] - b_p[:, None] * (d_plot[None, :]**2))
        hdi_lower = np.percentile(surv_samples_prism, 2.5, axis=0)
        hdi_upper = np.percentile(surv_samples_prism, 97.5, axis=0)
        surv_prism_plot = np.mean(surv_samples_prism, axis=0)

        prism_eud_doses = []
        for i, D_target in enumerate(nominal_doses):
            bins = real_data[D_target]['x'] if row_idx == 0 else narrow_data[D_target]['x']
            weights = real_data[D_target]['w'] if row_idx == 0 else narrow_data[D_target]['w']
            
            S_expected = np.sum(weights * np.exp(-mean_a_p * bins - mean_b_p * (bins**2)))
            val = (-mean_a_p + np.sqrt(mean_a_p**2 - 4 * mean_b_p * np.log(S_expected))) / (2 * mean_b_p)
            prism_eud_doses.append(val)

        # ---------------------------------------------------------------------
        # BAL OSZLOP: HISZTOGRAMOK
        # ---------------------------------------------------------------------
        ax_hist = fig.add_subplot(gs[row_idx, 0])
        color_palette = hist_colors_real if row_idx == 0 else hist_colors_narrow
        
        max_y_val = 0
        for i, D_target in enumerate(valid_doses):
            c_color = color_palette[i % len(color_palette)]
            bins = real_data[D_target]['x'] if row_idx == 0 else narrow_data[D_target]['x']
            weights = real_data[D_target]['w'] if row_idx == 0 else narrow_data[D_target]['w']
            
            plot_mask = bins <= max_dose_curve
            ax_hist.fill_between(bins[plot_mask], 0, weights[plot_mask], color=c_color, alpha=0.5, label=f'{D_target} Gy')
            ax_hist.plot(bins[plot_mask], weights[plot_mask], color=c_color, lw=2)
            
            if np.max(weights[plot_mask]) > max_y_val: max_y_val = np.max(weights[plot_mask])

        ax_hist.set_yticks([])
        ax_hist.set_title("A. Real FBX Measurements (dDVH)" if row_idx == 0 else "C. Homogenized (Mild Truncation)", fontsize=13, fontweight='bold')
        if row_idx == 1: ax_hist.set_xlabel("Absorbed Dose [Gy]", fontsize=12, fontweight='bold')
        ax_hist.set_ylabel("Relative Volume", fontsize=11, fontweight='bold')
        ax_hist.grid(True, linestyle=':', alpha=0.6)
        ax_hist.legend(loc='upper left', title="Nominal Dose")
        
        ax_hist.set_xlim(0, axis_x_limit)
        ax_hist.set_ylim(0, max_y_val * 1.15)

        # ---------------------------------------------------------------------
        # JOBB OSZLOP: TÚLÉLÉSI GÖRBÉK
        # ---------------------------------------------------------------------
        ax_surv = fig.add_subplot(gs[row_idx, 1])

        ax_surv.plot(d_plot, surv_trad_plot, linestyle='--', color=color_trad, lw=2.5, label='Simple Fit on Mean Dose')
        ax_surv.plot(mean_doses_physical, sf_observations, 's', color=color_trad, markeredgecolor='black', markersize=8, zorder=5)

        ax_surv.fill_between(d_plot, hdi_lower, hdi_upper, color='dodgerblue', alpha=0.2, label='95% HDI (Volumetric Uncertainty)')

        ax_surv.plot(d_plot, surv_prism_plot, linestyle='-', color=color_prism, lw=2.5, label='Bayesian Fit on PRISM EUD')
        ax_surv.plot(prism_eud_doses, sf_observations, 'o', color='dodgerblue', markeredgecolor='black', markersize=8, zorder=6)

        arrow_drawn = False
        for i in range(1, len(mean_doses_physical)):
            shift = mean_doses_physical[i] - prism_eud_doses[i]
            if shift > 0.05:
                ax_surv.annotate("", xy=(prism_eud_doses[i]+0.2, sf_observations[i]), 
                                 xytext=(mean_doses_physical[i]-0.2, sf_observations[i]),
                                 arrowprops=dict(arrowstyle="->", color='gray', lw=2, ls='solid'))
                arrow_drawn = True

        if arrow_drawn and row_idx == 0:
            ax_surv.text(mean_doses_physical[-1] - 0.5, sf_observations[-1]*3.0, 'EUD Shift', color='gray', fontsize=11, fontweight='bold', ha='right')

        param_text = (
            f"True : $\\alpha={true_alpha:.3f}$, $\\beta={true_beta:.3f}$\n"
            f"PRISM: $\\alpha={mean_a_p:.3f}$, $\\beta={mean_b_p:.3f}$\n"
            f"Mean : $\\alpha={mean_a_t:.3f}$, $\\beta={mean_b_t:.3f}$"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        # A doboz továbbra is a Jobb Felső sarokban!
        ax_surv.text(0.41, 0.35, param_text, transform=ax_surv.transAxes, fontsize=11, 
                     verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

        # JAVÍTÁS 3: Teljesen biztonságos Y-határ számítás, hogy ne bukhasson a tengely alá a kék sáv!
        min_curve_val = np.min(hdi_lower)
        min_obs_val = np.min([s for s in sf_observations if s > 0])
        absolute_min = min(min_curve_val, min_obs_val)
        y_bottom = 10**(np.floor(np.log10(absolute_min)) - 0.5)
        
        ax_surv.set_title("B. Radiobiological Response" if row_idx == 0 else "D. Radiobiological Response", fontsize=13, fontweight='bold')
        if row_idx == 1: ax_surv.set_xlabel("Dose [Gy]", fontsize=12, fontweight='bold')
        if row_idx == 0: ax_surv.set_ylabel("Surviving Fraction (Log Scale)", fontsize=12, fontweight='bold')
        
        ax_surv.grid(True, which="major", linestyle='-', alpha=0.5)
        ax_surv.grid(True, which="minor", linestyle=':', alpha=0.3)
        ax_surv.legend(loc='lower left', fontsize=10)
        
        ax_surv.set_yscale('log')
        
        # Az X tengely tágabb (axis_x_limit), mint a rajzolt görbe (max_dose_curve)
        ax_surv.set_xlim(-0.5, axis_x_limit)
        ax_surv.set_ylim(y_bottom, 1.5) 

    filename = "fig11_FBX_simulation_radiobiology.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Ábra elmentve mint: {filename}")
    plt.show()

    
if __name__ == "__main__":
    # -------------------------------------------------------------------
    # IDE ADD MEG A VALÓS .npy FÁJLOK ELÉRÉSI ÚTJÁT ÉS A NÉVLEGES DÓZIST!
    # (A 0 Gy-t a szkript automatikusan kezeli)
    # -------------------------------------------------------------------
    my_real_files = {
        4: "results/F07_F08_Right.npy",
        8: "results/F09_F10_Right.npy",
        15: "results/F11_F12_Right.npy"
    }
    
    # Teszt futtatás:
    generate_experimental_2x2_hybrid(my_real_files)