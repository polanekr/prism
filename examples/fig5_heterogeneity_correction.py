import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def generate_fig4_heterogeneity():
    print("--- 4. Ábra Generálása: Spektrális és Üreg-korrekció ---")
    
    # Vászon beállítása (Constrained layout a tökéletes margókért)
    fig = plt.figure(figsize=(24, 5.5), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1.1, 1, 1.3])
    
    # Színpaletta (Nyomtatás-biztos)
    color_uncorrected = 'gray'
    color_corrected = 'navy'
    color_film = 'darkorange'

    # ==========================================
    # PANEL A: Sematikus Elrendezés (2D Keresztmetszet)
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("A. Experimental Setup\n(Cross-section)", fontsize=14, fontweight='bold', pad=17)
    
    # 1. PMMA Fantom (Világosszürke téglalapok)
    phantom_front = patches.Rectangle((0, -20), 15, 40, facecolor='whitesmoke', edgecolor='black', hatch='//')
    phantom_rear = patches.Rectangle((25, -20), 15, 40, facecolor='whitesmoke', edgecolor='black', hatch='//')
    ax1.add_patch(phantom_front)
    ax1.add_patch(phantom_rear)
    
    # 2. Üreg / Eppendorf cső (Levegő, fehér)
    cavity = patches.Rectangle((15, -10), 10, 20, facecolor='white', edgecolor='black', lw=1.5)
    ax1.add_patch(cavity)
    ax1.text(20, 0, 'Air\nCavity', ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    ax1.text(7.5, -15, 'PMMA', ha='center', va='center', fontsize=12, color='gray')
    ax1.text(32.5, -15, 'PMMA', ha='center', va='center', fontsize=12, color='gray')
    
    # 3. Filmek (Narancssárga vonalak)
    ax1.vlines(0, -20, 20, colors=color_film, linestyles='solid', lw=4, label='Front Film')
    ax1.vlines(40, -20, 20, colors=color_film, linestyles='solid', lw=4, label='Rear Film')
    
    # 4. Sugárzás iránya (Nyilak balról)
    for y_arrow in [-15, -5, 5, 15]:
        ax1.arrow(-8, y_arrow, 6, 0, head_width=2, head_length=1.5, fc='navy', ec='navy', alpha=0.6)
    ax1.text(-5, 22, 'Radiation Beam', ha='center', color='navy', fontweight='bold')

    ax1.set_xlim(-10, 45)
    ax1.set_ylim(-25, 25)
    ax1.set_xlabel("Depth Z [mm]", fontsize=12)
    ax1.set_ylabel("Off-axis Distance [mm]", fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(False)

    # ==========================================
    # PANEL B: Rear Film Hőtérkép (Az üreg "árnyéka")
    # ==========================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("B. Rear Film Readout\n(Cavity Shadow Effect)", fontsize=14, fontweight='bold', pad=15)
    
    # Hőtérkép generálása
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    
    # Alap nyaláb (Gauss)
    beam_base = 50 * np.exp(-(X**2 + Y**2) / (2 * 12**2))
    
    # Üreg árnyéka (mivel levegő, KEVESEBBET nyel el, így a cső mögött TÖBB dózis jut a filmre!)
    # Téglalap alakú árnyék (a cső vetülete)
    cavity_mask = (np.abs(X) < 5) & (np.abs(Y) < 10)
    # Lágyítjuk a széleket egy kis Gauss szűréssel (simítással egyenértékű matematikai trükk)
    shadow_effect = 15 * np.exp(-(X**2 / (2*4**2)) - (Y**2 / (2*9**2)))
    
    rear_dose = beam_base + shadow_effect
    
    im = ax2.imshow(rear_dose, cmap='magma', extent=[-20, 20, -20, 20], origin='lower')
    
    # Jelöljük a cső körvonalát
    rect = patches.Rectangle((-5, -10), 10, 20, linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--')
    ax2.add_patch(rect)
    ax2.text(0, 12, "Higher Dose\nBehind Cavity", color='white', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel("X [mm]", fontsize=12)
    ax2.set_ylabel("Y [mm]", fontsize=12)
    plt.colorbar(im, ax=ax2, label="Dose [Gy]", fraction=0.046, pad=0.04)

    # ==========================================
    # PANEL C: Mélydózis Görbe (Korrekció Előtte/Utána)
    # ==========================================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("C. Depth-Dose Profile\n(Heterogeneity Correction)", fontsize=14, fontweight='bold', pad=15)
    
    z_axis = np.linspace(0, 40, 200)
    mu_pmma = 0.05 # Tömör fantom elnyelési tényezője
    mu_air = 0.001 # Levegő elnyelési tényezője (szinte nulla)
    
    # 1. Korrekció Nélküli modell (Sima exponenciális csökkenés, mintha minden PMMA lenne)
    dose_uncorrected = 100 * np.exp(-mu_pmma * z_axis)
    
    # 2. PRISM Korrigált modell
    dose_corrected = np.zeros_like(z_axis)
    for i, z in enumerate(z_axis):
        if z <= 15:
            # Cső előtt: Ugyanaz, mint a PMMA
            dose_corrected[i] = 100 * np.exp(-mu_pmma * z)
        elif z > 15 and z <= 25:
            # Csövön belül: Szinte nincs elnyelés (levegő), a dózis alig csökken
            dose_15 = 100 * np.exp(-mu_pmma * 15)
            dose_corrected[i] = dose_15 * np.exp(-mu_air * (z - 15))
        else:
            # Cső után: Újra PMMA elnyelés, de egy MAGASABB szintről indulva!
            dose_15 = 100 * np.exp(-mu_pmma * 15)
            dose_25 = dose_15 * np.exp(-mu_air * 10)
            dose_corrected[i] = dose_25 * np.exp(-mu_pmma * (z - 25))

    # Ábrázolás
    ax3.plot(z_axis, dose_uncorrected, linestyle='--', color=color_uncorrected, lw=2.5, 
             label='Uncorrected (Solid Phantom Assumption)')
    ax3.plot(z_axis, dose_corrected, linestyle='-', color=color_corrected, lw=2.5, 
             label='PRISM Corrected (Actual Heterogeneity)')
    
    # Jelöljük az üreg határait
    ax3.axvspan(15, 25, color='lightgray', alpha=0.3, label='Air Cavity')
    ax3.axvline(15, color='black', linestyle=':', lw=1)
    ax3.axvline(25, color='black', linestyle=':', lw=1)
    
    # Jelöljük a Front és Rear filmek helyét
    ax3.plot(0, 100, 'o', color=color_film, markersize=8, markeredgecolor='black', zorder=5, label='Front Film')
    ax3.plot(40, dose_corrected[-1], 'o', color=color_film, markersize=8, markeredgecolor='black', zorder=5, label='Rear Film Readout')
    
    # Mutatjuk a különbséget a Rear filmen
    ax3.annotate('Correction\nGap', xy=(39, dose_uncorrected[-1]), xytext=(30, dose_uncorrected[-1] + 10),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10, fontweight='bold')

    ax3.set_xlabel("Depth Z [mm]", fontsize=12)
    ax3.set_ylabel("Central Axis Dose [%]", fontsize=12)
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlim(-2, 42)

    # Mentés és megjelenítés
    filename = "fig5_heterogeneity_correction.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    generate_fig4_heterogeneity()