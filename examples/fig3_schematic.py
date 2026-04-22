#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISM Framework - 3. Ábra (Kétpaneles Változat - Finomított Geometria V5)
A: Independent PDD Learning Phase (GPR Baseline)
B: Shot-Specific Biological Irradiation (Registration & Correction)
Középen összehúzott panelekkel!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_prism_setup_schematic_2panels_v5():
    print("--- 3. Ábra: Kétpaneles PRISM Kísérleti Elrendezés (Összehúzott V5) ---")
    
    # Kisebb szélesség (18 -> 15), hogy eleve közelebb legyenek
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    
    # --- KÖZÖS GEOMETRIAI PARAMÉTEREK ---
    source_y = 70       
    surface_y = 0
    phantom_width = 150
    film_width = 50 
    roi_width = 60
    phantom_bottom = -100
    
    beam_half_angle = 14 
    tan_theta = np.tan(np.radians(beam_half_angle))
    
    # PMMA lapok: felszín közelében sűrűbb, majd egyenletes 1.5 cm (15 egység)
    slabs = [
        (0, -5), (-5, -10), (-10, -15), (-15, -25),
        (-25, -40), (-40, -55), (-55, -70), (-70, -85), (-85, -100)
    ]
    
    # ÚJ: left_margin és right_margin paraméterek a középső holttér eltüntetéséhez
    def draw_base_setup(ax, title, show_axes=False, left_margin=70, right_margin=70):
        """Közös elemek (forrás, nyaláb, fantom lapok, tengelyek) megrajzolása."""
        ax.axis('off')
        
        # Dinamikus margók beállítása!
        ax.set_xlim(-phantom_width/2 - left_margin, phantom_width/2 + right_margin) 
        ax.set_ylim(phantom_bottom - 10, source_y + 15)
        
        # Cím
        ax.text(0, source_y + 10, title, ha='center', va='bottom', 
                fontsize=18, fontweight='bold', color='#1A237E')
        
        # 1. Sugárforrás
        ax.plot(0, source_y, 'ro', markersize=10, markeredgecolor='black', zorder=10)
        ax.annotate('Divergent Point Source\n(e.g., Laser-driven)', 
                    xy=(0, source_y), xytext=(-15, source_y - 5),
                    ha='right', va='top', fontsize=14, fontweight='bold', color='darkred')
        
        # 2. Nyaláb kúp (Arany/Sárga)
        cone_x = surface_y - phantom_bottom
        beam_x_bottom = cone_x * tan_theta + (source_y * tan_theta)
        beam_triangle = np.array([
            [0, source_y],
            [-beam_x_bottom, phantom_bottom],
            [beam_x_bottom, phantom_bottom]
        ])
        ax.add_patch(patches.Polygon(beam_triangle, closed=True, color='gold', alpha=0.12, zorder=1))
        ax.plot([0, -beam_x_bottom], [source_y, phantom_bottom], color='orange', linestyle='--', lw=1.5, zorder=2)
        ax.plot([0, beam_x_bottom], [source_y, phantom_bottom], color='orange', linestyle='--', lw=1.5, zorder=2)

        # 3. PMMA Fantom (Halványkék átlátszó lapok, szürke kerettel)
        for (top, bottom) in slabs:
            h = abs(top - bottom)
            ax.add_patch(patches.Rectangle((-phantom_width/2, bottom), phantom_width, h, 
                                           linewidth=1.0, edgecolor='#90A4AE', facecolor='#E3F2FD', 
                                           alpha=0.4, zorder=3))
            
        ax.text(0, phantom_bottom - 5, 'Tissue-Equivalent PMMA Phantom Slabs', 
                ha='center', va='top', fontsize=14, fontweight='bold', color='dimgray')

        # 4. Tengelyek (SSD és Z-axis)
        if show_axes:
            axis_x = -phantom_width/2 - 40 
            
            # SSD Nyíl
            ax.annotate('', xy=(axis_x, surface_y), xytext=(axis_x, source_y),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax.text(axis_x - 5, (source_y + surface_y)/2, 'SSD', 
                    ha='right', va='center', fontsize=14, fontweight='bold', rotation=90)
            
            # Z-tengely Nyíl
            ax.annotate('', xy=(axis_x, phantom_bottom), xytext=(axis_x, surface_y),
                        arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
            ax.text(axis_x - 5, phantom_bottom/2, 'Depth in phantom (Z axis)', 
                    ha='right', va='center', fontsize=14, fontweight='bold', rotation=90)
            
            # Szaggatott vonal a felszín jelölésére
            ax.plot([axis_x - 10, phantom_width/2], [surface_y, surface_y], color='gray', linestyle=':', lw=1)

    # =========================================================================
    # PANEL A: PDD LEARNING PHASE (GPR BASELINE)
    # Bal oldalon nagy margó a tengelyeknek, jobb oldalon minimális holttér
    # =========================================================================
    draw_base_setup(ax1, "A. Step 1: Independent PDD Learning Phase", show_axes=True, left_margin=60, right_margin=10)
    
    # Sűrűbb filmek a felszínnél, majd egyenletes 1.5 cm-es közökkel
    pdd_films_y = [0, -5, -10, -15, -25, -40, -55, -70, -85]
    for y in pdd_films_y:
        ax1.plot([-film_width/2, film_width/2], [y, y], color='crimson', lw=3.0, zorder=6)
        
    ax1.annotate('Sparse 2D PDD\nFilm Stack', xy=(film_width/2, -10), xytext=(film_width/2 + 20, -10),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2), 
                ha='left', va='center', fontsize=14, fontweight='bold', color='crimson')

    # GPR 3D Térfogat "illúziója"
    ax1.add_patch(patches.Rectangle((-roi_width/2, pdd_films_y[-1]), roi_width, abs(pdd_films_y[-1]), 
                                 linewidth=0, facecolor='dodgerblue', alpha=0.25, zorder=4))
    
    ax1.annotate('Continuous 3D\nBaseline Volume\n(GPR Interpolation)', xy=(-roi_width/2, -35), xytext=(-roi_width/2 -3 , -20),
                arrowprops=dict(arrowstyle='->', color='mediumblue', lw=2), 
                ha='right', va='center', fontsize=14, fontweight='bold', color='mediumblue')

    # =========================================================================
    # PANEL B: SHOT-SPECIFIC BIOLOGICAL IRRADIATION
    # Bal oldalon minimális holttér, jobb oldalon nagy margó a szövegeknek
    # =========================================================================
    draw_base_setup(ax2, "B. Step 2: Shot-Specific Biological Irradiation", show_axes=False, left_margin=10, right_margin=100)
    
    front_y = -25
    rear_y = -55
    
    # ROI / FOV terület (Halvány zöld)
    ax2.add_patch(patches.Rectangle((-roi_width/2, rear_y), roi_width, abs(rear_y - front_y), 
                                 linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.25, linestyle='--', zorder=5))
    
    ax2.annotate('Computational ROI/FOV\n(Registration & Correction)', xy=(-roi_width/2, (front_y+rear_y)/2), xytext=(-roi_width/2 - 10, (front_y+rear_y)/2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2), 
                ha='right', va='center', fontsize=14, fontweight='bold', color='green')

    # Biológiai minta: Kisebb és a felső kétharmadban (Top: ~-27, Bottom: ~-45, Közép: -36)
    sample_height = 18 
    sample_width = 16
    sample_y_center = -36
    ax2.add_patch(patches.FancyBboxPatch((-sample_width/2, sample_y_center - sample_height/2), sample_width, sample_height,
                                       boxstyle="round,pad=1.5", facecolor='skyblue', edgecolor='darkblue', lw=1.5, zorder=7))
    
    ax2.annotate('Biological Sample\n(Heterogeneous Cavity)', xy=(sample_width/2 + 2, sample_y_center), xytext=(film_width/2 + 15, sample_y_center),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2), 
                ha='left', va='center', fontsize=14, fontweight='bold', color='darkblue')

    # Front Film
    ax2.plot([-film_width/2, film_width/2], [front_y, front_y], color='red', lw=3.5, zorder=8)
    ax2.annotate('Front Film\n(Spatial Alignment &\nAbsolute Scaling)', xy=(film_width/2, front_y), xytext=(film_width/2 + 15, front_y + 8),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5, edgecolor='none'), 
                ha='left', va='center', fontsize=14, fontweight='bold', color='red')

    # Rear Film
    ax2.plot([-film_width/2, film_width/2], [rear_y, rear_y], color='darkorange', lw=3.5, zorder=8)
    ax2.annotate('Rear Film\n(Cavity Shadow &\nSpectral Correction)', xy=(film_width/2, rear_y), xytext=(film_width/2 + 15, rear_y - 8),
                arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1, headwidth=5, edgecolor='none'), 
                ha='left', va='center', fontsize=14, fontweight='bold', color='darkorange')

    # Margók összehúzása és mentés
    plt.tight_layout(w_pad=0.5) # w_pad=0.5 drasztikusan lecsökkenti a két ábra közötti távolságot!
    filename = "fig3_prism_two_step_schematic.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    draw_prism_setup_schematic_2panels_v5()