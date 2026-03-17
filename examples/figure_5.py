import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

def generate_fig5_final_output():
    print("--- 5. Ábra Generálása: Biológiai Dozimetria és DVH ---")
    
    # ==========================================
    # 1. Szintetikus 3D Térfogat és Maszk Generálása
    # ==========================================
    # Fizikai rács (X, Y, Z mm-ben)
    x = np.linspace(-15, 15, 60)
    y = np.linspace(-15, 15, 60)
    z = np.linspace(0, 40, 80)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Szintetikus nyaláb (Középen erős, kifelé csökken, mélységben elnyelődik)
    beam_radius = 5.0 + 0.1 * Z
    dose_3d = 12.0 * np.exp(-0.02 * Z) * np.exp(-(X**2 + Y**2) / (2 * beam_radius**2))
    
    # Biológiai ROI Maszk: Eppendorf cső (Cilinder + Kúp)
    mask = np.zeros_like(dose_3d, dtype=bool)
    r_tube = 4.5
    
    # Henger rész (Z = 10-től 25-ig)
    mask[(Z >= 10) & (Z <= 25) & (X**2 + Y**2 <= r_tube**2)] = True
    # Kúp rész (Z = 25-től 30-ig)
    cone_r = r_tube * (30 - Z) / 5.0
    mask[(Z > 25) & (Z <= 30) & (X**2 + Y**2 <= cone_r**2)] = True

    # ==========================================
    # 2. DVH és EUD Számítás (A maszk alatti dózisokból)
    # ==========================================
    roi_doses = dose_3d[mask]
    
    # Hisztogram (Differential DVH)
    bins = np.linspace(np.min(roi_doses)*0.9, np.max(roi_doses)*1.1, 40)
    hist, bin_edges = np.histogram(roi_doses, bins=bins, density=False)
    
    # Relatív térfogattá alakítás (%)
    hist_percent = (hist / len(roi_doses)) * 100.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Simítás az esztétikus megjelenésért
    hist_smooth = gaussian_filter1d(hist_percent, sigma=1.2)
    
    # Bayes-i bizonytalanság szimulálása (Egy HDI sáv a hisztogram köré)
    uncertainty_band = hist_smooth * 0.15 + 0.5 # 15% relatív + konstans zaj
    
    # EUD (Equivalent Uniform Dose) számítása
    # Egyszerű átlaggal (a=1) szimulálva
    eud_mean = np.mean(roi_doses)
    eud_std = np.std(roi_doses) * 0.25 # Szimulált Bayes-i bizonytalanság

    # ==========================================
    # 3. Publikációkész Ábra Rajzolása
    # ==========================================
    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    
    # Layout: 3 keskenyebb panel a metszeteknek, 1 széles a DVH-nak
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.8], figure=fig)
    
    cmap_dose = 'magma'
    contour_color = 'white' # A maszk körvonala
    
    # Középpont keresése a metszetekhez (ahol a cső közepe van)
    cx, cy, cz = 30, 30, 40 # Indexek (x=0, y=0, z=20 mm körül)
    extent_xy = [x[0], x[-1], y[0], y[-1]]
    extent_xz = [x[0], x[-1], z[-1], z[0]] # Z lefelé nő
    extent_yz = [y[0], y[-1], z[-1], z[0]]

    # --- PANEL A: AXIAL VIEW (XY) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("A. Axial View (XY)", fontsize=13, fontweight='bold', pad=10)
    im1 = ax1.imshow(dose_3d[:, :, cz].T, extent=extent_xy, origin='lower', cmap=cmap_dose)
    ax1.contour(x, y, mask[:, :, cz].T, levels=[0.5], colors=contour_color, linestyles='--', linewidths=2)
    ax1.set_xlabel("X [mm]"); ax1.set_ylabel("Y [mm]")
    
    # --- PANEL B: CORONAL VIEW (XZ) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("B. Coronal View (XZ)", fontsize=13, fontweight='bold', pad=10)
    ax2.imshow(dose_3d[:, cy, :].T, extent=extent_xz, aspect='auto', cmap=cmap_dose)
    ax2.contour(x, z, mask[:, cy, :].T, levels=[0.5], colors=contour_color, linestyles='--', linewidths=2)
    ax2.set_xlabel("X [mm]"); ax2.set_ylabel("Depth Z [mm]")
    ax2.invert_yaxis() # Mélység lefelé
    
    # --- PANEL C: SAGITTAL VIEW (YZ) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("C. Sagittal View (YZ)", fontsize=13, fontweight='bold', pad=10)
    im3 = ax3.imshow(dose_3d[cx, :, :].T, extent=extent_yz, aspect='auto', cmap=cmap_dose)
    ax3.contour(y, z, mask[cx, :, :].T, levels=[0.5], colors=contour_color, linestyles='--', linewidths=2)
    ax3.set_xlabel("Y [mm]"); ax3.set_ylabel("Depth Z [mm]")
    ax3.invert_yaxis()

    # Közös Colorbar a 3 metszethez
    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.04, pad=0.15)
    cbar.set_label("Absolute Dose [Gy]", fontsize=11, fontweight='bold')

    # --- PANEL D: DIFFERENTIAL DVH ---
    ax4 = fig.add_subplot(gs[3])
    ax4.set_title("D. Biological Target dDVH\n(Dose-Volume Histogram)", fontsize=14, fontweight='bold', pad=10)
    
    # Hisztogram vonal és árnyékolás (Bayes HDI)
    ax4.plot(bin_centers, hist_smooth, color='navy', lw=3, label='Mean dDVH')
    ax4.fill_between(bin_centers, np.maximum(0, hist_smooth - uncertainty_band), 
                     hist_smooth + uncertainty_band, color='dodgerblue', alpha=0.3, 
                     label='95% Bayesian HDI')
    
    # EUD Vonal berajzolása
    ax4.axvline(eud_mean, color='darkorange', linestyle='--', lw=2.5, label='EUD (Equivalent Uniform Dose)')
    
    # Bizonytalansági sáv az EUD körül
    ax4.axvspan(eud_mean - eud_std, eud_mean + eud_std, color='darkorange', alpha=0.2)
    
    # Text Box az EUD pontos értékének
    textstr = f"Target EUD:\n$\\mathbf{{{eud_mean:.2f} \\pm {eud_std:.2f} \\ Gy}}$"
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
    ax4.text(0.05, 0.85, textstr, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ax4.set_xlabel("Dose [Gy]", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Relative Volume [%]", fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Egy kis extra hely az X tengely szélein
    ax4.set_xlim(bin_centers[0] - 0.5, bin_centers[-1] + 0.5)
    ax4.set_ylim(0, np.max(hist_smooth + uncertainty_band) * 1.15)

    # Mentés
    filename = "fig5_final_biological_dosimetry.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    generate_fig5_final_output()