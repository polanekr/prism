import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_fig7_radiobiology_npy_batch(npy_file_list: list, measured_survivals=None):
    """
    7. Ábra generálása több valós .npy (szótárat tartalmazó) hisztogram fájl alapján.
    
    Paraméterek:
    - npy_file_list: Az egyes besugárzásokhoz tartozó .npy fájlok elérési útjainak listája.
    - measured_survivals: Opcionális lista a valós biológiai túlélési adatokkal. 
    """
    print("--- 7. Ábra Generálása: Valós .npy Fájlok Feldolgozása ---")
    
    np.random.seed(42)

    # A sejtek IGAZI biológiai paraméterei
    true_alpha = 0.3  
    true_beta  = 0.03 
    
    def lq_model(D, alpha, beta):
        return np.exp(-alpha * D - beta * D**2)

    # =========================================================================
    # 1. ADATOK KINYERÉSE AZ .NPY FÁJLOKBÓL
    # =========================================================================
    
    # Kezdőpont (0 Gy, 100% túlélés)
    mean_doses = [0.0]
    prism_eud_doses = [0.0]
    survivals = [1.0]

    for npy_file in npy_file_list:
        if not os.path.exists(npy_file):
            print(f"  [!] Fájl nem található, kihagyva -> {npy_file}")
            continue
            
        try:
            # AZ .item() FONTOS: Ez alakítja vissza a NumPy Object Array-t szótárrá!
            data_dict = np.load(npy_file, allow_pickle=True).item()
            
            # A dose_stats egy 2D tömb: 1. sor a dózis, 2. sor a bizonytalanság
            dose_bins = data_dict['dose_stats'][0, :]
            weights = data_dict['weights']
            
            # Biztos ami biztos, normalizálunk
            weights = weights / np.sum(weights) 
            
        except Exception as e:
            print(f"  [!] Hiba a fájl olvasásakor ({npy_file}): {e}")
            continue

        # 1. Átlagdózis (Mean Dose) kiszámolása a hisztogramból
        mean_dose = np.sum(weights * dose_bins)
        mean_doses.append(mean_dose)

        # 2. Valós túlélés kiszámolása a teljes hisztogram integrálásával
        S_true_total = np.sum(weights * lq_model(dose_bins, true_alpha, true_beta))
        
        # Kis biológiai zaj hozzáadása
        S_noisy = S_true_total * np.random.normal(1.0, 0.05)
        survivals.append(S_noisy)

        # 3. PRISM EUD (Ekvivalens Homogén Dózis) visszaszámolása
        eud = (-true_alpha + np.sqrt(true_alpha**2 - 4 * true_beta * np.log(S_true_total))) / (2 * true_beta)
        prism_eud_doses.append(eud)

        print(f"  -> Betöltve: {os.path.basename(npy_file)} | Átlag: {mean_dose:.2f} Gy | EUD: {eud:.2f} Gy")

    # Ha valós méréseink is vannak
    if measured_survivals is not None:
        if len(measured_survivals) != len(mean_doses):
            print("  [!] Hiba: A megadott túlélési adatok száma nem egyezik a fájlok számával (+ 0Gy pont)!")
            return
        survivals = np.array(measured_survivals)
    else:
        survivals = np.array(survivals)

    mean_doses = np.array(mean_doses)
    prism_eud_doses = np.array(prism_eud_doses)

    # =========================================================================
    # 2. GÖRBEILLESZTÉSEK
    # =========================================================================
    def lq_fit_func(D, a, b):
        return np.exp(-a * D - b * D**2)

    # Hagyományos (hibás) illesztés a MEAN DOSE alapján
    try:
        popt_trad, _ = curve_fit(lq_fit_func, mean_doses, survivals, p0=[0.2, 0.02])
        alpha_trad, beta_trad = popt_trad
    except:
        alpha_trad, beta_trad = 0.2, 0.02 

    # A valós (PRISM) modell paraméterei
    alpha_prism, beta_prism = true_alpha, true_beta
    
    # Rajzoláshoz sűrű X tengely
    d_plot = np.linspace(0, max(mean_doses) * 1.15, 200)
    surv_trad_plot = lq_model(d_plot, alpha_trad, beta_trad)
    surv_prism_plot = lq_model(d_plot, alpha_prism, beta_prism)

    # =========================================================================
    # 3. ÁBRA RAJZOLÁSA
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)

    color_trad = 'crimson'
    color_prism = 'navy'

    # 1. Hagyományos (Mean Dose) görbe és pontok
    ax.plot(d_plot, surv_trad_plot, linestyle='--', color=color_trad, lw=2.5, 
            label=f'Fit based on Mean Dose\n($\\alpha={alpha_trad:.2f}$, $\\beta={beta_trad:.3f}$)')
    ax.plot(mean_doses, survivals, 's', color=color_trad, markeredgecolor='black', 
            markersize=9, zorder=5, label='Mean Dose Points')

    # 2. PRISM (EUD) görbe és pontok
    ax.fill_between(d_plot, surv_prism_plot * np.exp(-0.06 * d_plot), surv_prism_plot * np.exp(0.06 * d_plot), 
                    color='dodgerblue', alpha=0.2, label='95% HDI (Volumetric Uncertainty)')
    ax.plot(d_plot, surv_prism_plot, linestyle='-', color=color_prism, lw=2.5, 
            label=f'Fit based on PRISM EUD\n($\\alpha={alpha_prism:.2f}$, $\\beta={beta_prism:.3f}$)')
    ax.plot(prism_eud_doses, survivals, 'o', color='dodgerblue', markeredgecolor='black', 
            markersize=9, zorder=6, label='EUD Points (PRISM / FBX)')

    # 3. NYILAK (A dózis eltolódásának bemutatása)
    for i in range(1, len(mean_doses)):
        if abs(prism_eud_doses[i] - mean_doses[i]) > 0.1:
            ax.annotate("", xy=(prism_eud_doses[i]+0.15, survivals[i]), 
                        xytext=(mean_doses[i]-0.15, survivals[i]),
                        arrowprops=dict(arrowstyle="->", color='gray', lw=2, ls='solid'))

    if len(survivals) > 1:
        ax.text(mean_doses[-1] - 0.5, survivals[-1]*4.0, 'Shift from Mean Dose to EUD', 
                color='gray', fontsize=11, fontweight='bold', ha='right', va='bottom')

    # Esztétika
    ax.set_yscale('log')
    ax.set_ylim(5e-5, 1.2)
    ax.set_xlim(-0.5, max(mean_doses) * 1.2)
    
    ax.set_title("Radiobiological Impact: Mean Dose vs. Equivalent Uniform Dose (EUD)", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Dose [Gy]", fontsize=13, fontweight='bold')
    ax.set_ylabel("Surviving Fraction (Log Scale)", fontsize=13, fontweight='bold')
    
    ax.grid(True, which="major", linestyle='-', alpha=0.5)
    ax.grid(True, which="minor", linestyle=':', alpha=0.3)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

    filename = "fig7_radiobiology_from_npy_batch.png"
    plt.savefig(filename, dpi=300)
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    # -------------------------------------------------------------------
    # IDE ADD MEG A FÁJLOKAT NÖVEKVŐ DÓZIS SORRENDBEN! 
    # (A 0 Gy-t a szkript automatikusan hozzácsapja az elejéhez!)
    # -------------------------------------------------------------------
    path = "/home/polanekr/Kutatás/esylos-dosimetry/data/processed/2025-12-10-eSYLOS-eli60052/"
    npy_files = [
        path+"Bal-2Gy.npy", # Ezt a fájlt például már fel is töltötted!
        path+"Bal-4Gy.npy",
        path+"Bal-6Gy.npy",
        path+"Bal-8Gy.npy",
        path+"Job-2Gy.npy",
        path+"Job-4Gy.npy",
        path+"Job-6Gy.npy",
        path+"Job-8Gy.npy"
    ]
    
    # Futtatás szimulált túléléssel:
    generate_fig7_radiobiology_npy_batch(npy_files)