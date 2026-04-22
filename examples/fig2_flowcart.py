import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_prism_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Színek a fázisokhoz
    c_exp = '#E8F5E9'  # Zöldes (Experiment)
    c_2d  = '#E3F2FD'  # Kékes (2D Dosimetry)
    c_3d  = '#FFF3E0'  # Narancsos (3D Recon)
    c_bio = '#FCE4EC'  # Rózsaszínes (Biology)
    edge = '#424242'
    
    def draw_box(x, y, w, h, text, color, title=None):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     ec=edge, fc=color, lw=2)
        ax.add_patch(box)
        if title:
            ax.text(x+w/2, y+h-0.3, title, ha='center', va='top', fontsize=14, fontweight='bold')
            ax.text(x+w/2, y+h/2-0.2, text, ha='center', va='center', fontsize=13, linespacing=1.4)
        else:
            ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=13, fontweight='bold', linespacing=1.4)
        return (x+w, y+h/2), (x, y+h/2), (x+w/2, y), (x+w/2, y+h)

    def draw_arrow(start, end, label="", x_offset=0.0, y_offset=0.4, ha='center', va='bottom'):
        # A nyíl megrajzolása
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2.5, color=edge))
        
        # A felirat elhelyezése (ha van)
        if label:
            mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2
            ax.text(mx + x_offset, my + y_offset, label, 
                    ha=ha, va=va, fontsize=14, fontstyle='italic', color='#616161')

    # Fázisok háttere
    ax.add_patch(patches.Rectangle((-1, -1), 16, 9, fc='#FAFAFA', ec='none', zorder=-1))
    
    # Y szintek
    y_top = 6
    y_mid = 3.5
    y_bot = 1
    
    # --- PHASE 1: Experimental ---
    ax.text(1, 8, "1. Experimental Setup & Irradiation", fontsize=18, fontweight='bold')
    out_beam, _, _, _ = draw_box(0, y_top, 3, 1.2, "Irradiation Setup\n(LWFA / Clinical Beam)", c_exp)
    out_det, in_det, out_det_bot, _ = draw_box(4, y_top, 3, 1.2, "Reference Detectors\n(Semiflex, FBX)", c_exp)
    out_film, in_film, out_film_bot, _ = draw_box(8, y_top, 3, 1.2, "Radiochromic Films\n(Sparse 2D Stacks)", c_exp)
    
    draw_arrow(out_beam, in_det)
    draw_arrow((7, y_top+0.6), in_film)
    
    # --- PHASE 2: 2D Dosimetry ---
    ax.text(1, 5.2, "2. Film Dosimetry & Calibration", fontsize=18, fontweight='bold')
    out_calib, in_calib, _, _ = draw_box(4, y_mid, 3, 1.2, "Bayesian Calibration\n(MCMC Parameter Fitting)", c_2d)
    out_2dmap, in_2dmap, _, in_2dmap_top = draw_box(8, y_mid, 3, 1.2, "2D Dose Maps\n(Hybrid Solver & bFDR)", c_2d)
    
    draw_arrow(out_film_bot, in_2dmap_top, "TIFF files", x_offset=0.2, y_offset=0.0, ha='left', va='center')
    draw_arrow(out_calib, in_2dmap, "Calibration curve", x_offset=-0.8, y_offset=1.0, ha='left', va='center')
    
    # --- PHASE 3: 3D PRISM Reconstruction ---
    ax.text(1, 2.7, "3. PRISM 3D Reconstruction", fontsize=18, fontweight='bold')
    out_gp, in_gp, _, in_gp_top = draw_box(8, y_bot, 3.5, 1.2, "Gaussian Process Regression\n(3D Volume Generation)", c_3d)
    out_het, in_het, _, _ = draw_box(4, y_bot, 3, 1.2, "Heterogeneity Correction\n(Phantom/Cavity effects)", c_3d)
    
    draw_arrow((11.5, y_mid+0.6), (12.5, y_mid+0.6))
    draw_arrow((12.5, y_mid+0.6), (12.5, y_bot+0.6))
    draw_arrow((12.5, y_bot+0.6), (11.5, y_bot+0.6), "Sparse 2D PDD", x_offset=0.0, y_offset=-0.2, ha='left', va='center')
    
    draw_arrow(in_gp, out_het)
    
    # --- PHASE 4: Biological Modeling ---
    ax.text(1, 0, "4. Radiobiological Output", fontsize=18, fontweight='bold')
    _, _, _, in_bio_top = draw_box(0, y_bot, 3, 1.2, "Biological Dosimetry\n(dDVH & EUD with \nBayesian Uncertainty)", c_bio)
    
    draw_arrow(in_het, (3, y_bot+0.6), "Corrected 3D Volume", x_offset=-0.8, y_offset=-0.9, ha='left', va='center')
    
    # Validációs nyíl (Detektorokból a Végeredményre)
    ax.annotate("", xy=(1.5, y_bot+1.2), xytext=(5.5, y_top), 
                arrowprops=dict(arrowstyle="<->", lw=2, color='#E53935', linestyle='dashed'))
    ax.text(2.5, 4.5, "Physical Validation\n(Gamma Index)", color='#E53935', fontweight='bold', ha='center')

    plt.xlim(-1, 15)
    plt.ylim(-0.5, 8.5)
    plt.tight_layout()
    plt.savefig("fig_2_framework_flowchart.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_prism_flowchart()