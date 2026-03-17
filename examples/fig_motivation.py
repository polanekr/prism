#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:19:46 2026

@author: polanekr
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_fig1_motivation():
    print("--- 1. Ábra Generálása: A Pre-klinikai Dozimetria Kihívásai ---")
    
    # Vászon beállítása
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off') # Kikapcsoljuk a tengelyeket, mert ez egy infografika
    
    # Háttérszín
    ax.add_patch(patches.Rectangle((0, 0), 15, 9, fc='#FAFAFA', ec='none', zorder=-1))
    
    # Főcím
    ax.text(7.5, 8.5, "The Challenges of Advanced Pre-Clinical Dosimetry", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#212121')
    ax.text(7.5, 8.1, "Why standard mean-dose assumptions fail in radiobiology", 
            ha='center', va='center', fontsize=14, fontstyle='italic', color='#616161')

    # ==========================================
    # SEÉGDFÜGGVÉNY A DOBOZOKHOZ
    # ==========================================
    def draw_box(x, y, w, h, title, text, fc, ec):
        # Enyhe árnyék
        ax.add_patch(patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1", 
                                            fc='black', alpha=0.1, ec='none'))
        # Fő doboz
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     fc=fc, ec=ec, lw=2.5)
        ax.add_patch(box)
        
        # Cím (vastaggal)
        ax.text(x+w/2, y+h-0.25, title, ha='center', va='top', 
                fontsize=12, fontweight='bold', color=ec)
        # Szöveg
        ax.text(x+w/2, y+h/2-0.1, text, ha='center', va='center', 
                fontsize=11, linespacing=1.5, color='#212121')
        
        # Visszaadjuk a doboz alsó közepének koordinátáit a nyilakhoz
        return (x+w/2, y)

    def draw_arrow(start, end, color='#424242', lw=2.5):
        ax.annotate("", xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.6", 
                                    lw=lw, color=color, shrinkA=0, shrinkB=0))

    # ==========================================
    # 1. SZINT: A 4 KIHÍVÁS (Pirosas tónus)
    # ==========================================
    c_prob_bg = '#FFEBEE' # Világos piros
    c_prob_fg = '#D32F2F' # Sötét piros
    
    # 4 doboz elosztása a 15 egységnyi szélességen
    w = 3.2
    h = 1.6
    y_prob = 6.0
    gap = (15 - (4 * w)) / 5
    
    x1 = gap
    x2 = x1 + w + gap
    x3 = x2 + w + gap
    x4 = x3 + w + gap
    
    bottom1 = draw_box(x1, y_prob, w, h, "Beam Fluctuations", 
                       "Shot-to-shot variations in\ndose rate and beam profile\n(e.g., LWFA, FLASH)", 
                       c_prob_bg, c_prob_fg)
    
    bottom2 = draw_box(x2, y_prob, w, h, "Steep Gradients", 
                       "Millimeter-scale dose drop-offs\nrequiring high spatial\nresolution dosimetry", 
                       c_prob_bg, c_prob_fg)
    
    bottom3 = draw_box(x3, y_prob, w, h, "Heterogeneous Media", 
                       "Beam attenuation and scattering\nchanges at air/plastic/water\ninterfaces", 
                       c_prob_bg, c_prob_fg)
    
    bottom4 = draw_box(x4, y_prob, w, h, "Volumetric Uncertainty", 
                       "Single point measurements\nfail to represent the full\n3D biological target", 
                       c_prob_bg, c_prob_fg)

    # ==========================================
    # 2. SZINT: A MEGOLDÁS (PRISM) (Kékes tónus)
    # ==========================================
    c_sol_bg = '#E3F2FD'
    c_sol_fg = '#1976D2'
    
    w_prism = 10.0
    h_prism = 1.6
    x_prism = (15 - w_prism) / 2
    y_prism = 3.2
    
    # PRISM doboz megrajzolása
    draw_box(x_prism, y_prism, w_prism, h_prism, 
             "The PRISM Framework", 
             "Integrates multi-detector physics measurements, 2D film dosimetry,\nand Gaussian Process Regression to reconstruct continuous 3D dose volumes.", 
             c_sol_bg, c_sol_fg)

    # Nyilak a problémáktól a PRISM-ig
    # Célpontok elosztása a PRISM doboz tetején
    target_y = y_prism + h_prism
    draw_arrow(bottom1, (x_prism + w_prism*0.15, target_y), color=c_prob_fg)
    draw_arrow(bottom2, (x_prism + w_prism*0.38, target_y), color=c_prob_fg)
    draw_arrow(bottom3, (x_prism + w_prism*0.62, target_y), color=c_prob_fg)
    draw_arrow(bottom4, (x_prism + w_prism*0.85, target_y), color=c_prob_fg)

    # ==========================================
    # 3. SZINT: A VÉGEREDMÉNY (Zöldes tónus)
    # ==========================================
    c_out_bg = '#E8F5E9'
    c_out_fg = '#388E3C'
    
    w_out = 8.0
    h_out = 1.4
    x_out = (15 - w_out) / 2
    y_out = 0.5
    
    draw_box(x_out, y_out, w_out, h_out, 
             "Precision Radiobiology", 
             "Provides voxel-level uncertainty and accurate Equivalent Uniform Dose (EUD)\nfor reliable cell survival modeling (Linear-Quadratic Model).", 
             c_out_bg, c_out_fg)

    # Nyíl a PRISM-ből a Végeredménybe
    draw_arrow((7.5, y_prism), (7.5, y_out + h_out), color=c_sol_fg, lw=3.5)

    # Margók beállítása és mentés
    plt.xlim(0, 15)
    plt.ylim(0, 9)
    plt.tight_layout()
    filename = "fig1_motivation_infographic.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    generate_fig1_motivation()