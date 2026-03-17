#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:16:40 2026

@author: polanekr
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_prism_setup_schematic():
    print("--- 3. Ábra: Kísérleti Elrendezés Vázlata (ROI/FOV kiegészítéssel) ---")
    
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.axis('off') 
    
    # --- GEOMETRIAI PARAMÉTEREK ---
    source_y = 120
    surface_y = 0
    phantom_width = 150
    film_width = 50 
    roi_width = 60 # A szoftverben beállított rekonstrukciós ROI mérete
    
    beam_half_angle = 12 
    tan_theta = np.tan(np.radians(beam_half_angle))
    
    slabs = [
        (0, -5), (-5, -10), (-10, -15), (-15, -25),
        (-25, -55), (-55, -70), (-70, -90)
    ]
    
    films = [0, -5, -10, -15, -25, -55, -70, -90]
    front_film_y = -25
    rear_film_y = -55
    
    target_y = -40
    target_width = 20
    target_height = 12 
    
    # --- 1. NYALÁB ---
    bottom_y = slabs[-1][1]
    beam_x_bottom = (source_y - bottom_y) * tan_theta
    beam_polygon = patches.Polygon(
        [[0, source_y], [-beam_x_bottom, bottom_y], [beam_x_bottom, bottom_y]],
        closed=True, facecolor='gold', alpha=0.15, edgecolor='orange', linewidth=2, linestyle='--'
    )
    ax.add_patch(beam_polygon)
    
    # --- 2. PMMA FANTOM LAPOK ---
    for top, bottom in slabs:
        height = top - bottom
        rect = patches.Rectangle((-phantom_width/2, bottom), phantom_width, height, 
                                 linewidth=1, edgecolor='dimgray', facecolor='aliceblue', alpha=0.6)
        ax.add_patch(rect)
        
    # --- ÚJ: 3. REKONSTRUKCIÓS ROI / FOV ---
    # Egy lila szaggatott téglalap, ami mutatja a 3D rekonstrukciós térfogatot
    roi_rect = patches.Rectangle((-roi_width/2, bottom_y), roi_width, surface_y - bottom_y, 
                                 linewidth=2.5, edgecolor='mediumpurple', facecolor='mediumpurple', 
                                 alpha=0.15, linestyle='-.', zorder=3)
    ax.add_patch(roi_rect)

    # --- 4. BEÁGYAZOTT MINTA ---
    box = patches.FancyBboxPatch((-target_width/2, target_y - target_height/2), 
                                 target_width, target_height, boxstyle="round,pad=3", 
                                 facecolor='white', edgecolor='black', hatch='////', linewidth=1.5, zorder=6)
    ax.add_patch(box)
    
    ax.annotate('Embedded Sample\n(e.g., Tube/Chamber)', xy=(target_width/2 + 2, target_y), 
                xytext=(target_width/2 + 35, target_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, fontweight='bold', va='center', ha='left')

    # --- 5. GAFCHROMIC FILMEK ---
    for fy in films:
        if fy == front_film_y:
            color, lw, zorder = 'red', 3, 5
        elif fy == rear_film_y:
            color, lw, zorder = 'darkorange', 3, 5
        else:
            color, lw, zorder = 'crimson', 2, 4
            
        ax.plot([-film_width/2, film_width/2], [fy, fy], color=color, linewidth=lw, zorder=zorder)
        
    # --- 6. SUGÁRFORRÁS ---
    ax.plot(0, source_y, 'o', markerfacecolor='gold', markeredgecolor='red', 
            markersize=12, markeredgewidth=2, zorder=10)
    ax.text(0, source_y + 8, "Point Source", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # ==========================================================
    # --- 7. ANNOTÁCIÓK - BAL OLDAL ---
    # ==========================================================
    left_x_z = -phantom_width/2 - 20
    left_x_ssd = -phantom_width/2 - 50

    ax.annotate('', xy=(left_x_z, bottom_y), xytext=(left_x_z, surface_y),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2))
    ax.text(left_x_z - 6, (surface_y + bottom_y)/2, "Depth in Phantom\n(Z axis)", 
            ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

    ax.annotate('', xy=(left_x_ssd, surface_y), xytext=(left_x_ssd, source_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(left_x_ssd - 6, (source_y + surface_y)/2, "Source-to-Surface\nDistance (SSD)", 
            ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
            
    # ==========================================================
    # --- 8. ANNOTÁCIÓK - JOBB OLDAL ---
    # ==========================================================
    right_x = phantom_width/2 + 15

    # ÚJ: ROI felirat
    ax.annotate('3D Reconstruction\nROI / FOV', xy=(roi_width/2, -5), xytext=(right_x, 25),
                arrowprops=dict(arrowstyle='->', color='mediumpurple', lw=2), 
                ha='left', va='center', fontsize=10, fontweight='bold', color='indigo')

    ax.annotate('Gafchromic Films\nfor 3D Reconstruction', xy=(film_width/2, -10), xytext=(right_x, 0),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5), 
                ha='left', va='center', fontsize=10, fontweight='bold', color='crimson')

    ax.annotate('Front Film\n(Dose Scaling)', xy=(film_width/2, front_film_y), xytext=(right_x, front_film_y + 10),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6, edgecolor='none'), 
                fontsize=10, fontweight='bold', color='red', va='center')

    ax.annotate('Rear Film\n(Heterogeneity Correction)', xy=(film_width/2, rear_film_y), xytext=(right_x, rear_film_y - 10),
                arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1, headwidth=6, edgecolor='none'), 
                fontsize=10, fontweight='bold', color='darkorange', va='center')
                
    ax.annotate('PMMA Phantom Slabs', xy=(phantom_width/2, -85), xytext=(right_x-10, -85),
                ha='left', va='center', fontsize=11, color='dimgray', fontweight='bold')

    # --- BEÁLLÍTÁSOK ---
    ax.set_xlim(-phantom_width/2 - 100, phantom_width/2 + 130)
    ax.set_ylim(bottom_y - 15, source_y + 35)
    ax.set_aspect('equal') 
    
    filename = "fig3_prism_experimental_setup_final.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nKész! Gyönyörű vázlat elmentve: {filename}")
    plt.show()

if __name__ == "__main__":
    draw_prism_setup_schematic()