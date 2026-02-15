"""
PRISM: Probabilistic Reconstruction of Inhomogeneous Systems Methodology
Module: viz.py

This module handles advanced visualization using Matplotlib and Plotly.
1. Interactive 3D Isodose Surfaces.
2. Gamma Map Visualization.
3. DVH Plotting.

Author: [Your Name / PRISM Team]
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_3d_interactive(volume, coords, level=50, opacity=0.6, step_limit=100):
    """
    Creates an interactive 3D isodose plot using Plotly.
    
    Args:
        volume: 3D numpy array (Z, Y, X).
        coords: Dictionary with 'x', 'y', 'z' arrays (mm).
        level: Isodose level in % (relative to max).
        step_limit: Downsample factor if volume is too large (max pixels per axis).
    """
    if volume is None:
        print("Error: No volume data to plot.")
        return

    # Downsampling for performance if needed
    s = 1
    nz, ny, nx = volume.shape
    if max(nz, ny, nx) > step_limit:
        s = int(max(nz, ny, nx) / step_limit)
    
    vol_small = volume[::s, ::s, ::s]
    x_small = coords['x'][::s]
    y_small = coords['y'][::s]
    z_small = coords['z'][::s]

    # Calculate threshold
    max_d = np.max(volume)
    val = max_d * (level / 100.0)
    
    # Meshgrid for Plotly (X, Y, Z flattened)
    X, Y, Z = np.meshgrid(x_small, y_small, z_small)
    
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_small.flatten(),
        isomin=val,
        isomax=max_d, # Draw everything above 'level'
        surface_count=2, # Draw surface at min and max
        opacity=opacity,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale='Jet'
    ))
    
    fig.update_layout(
        title=f"3D Isodose Surface (>{level}%)",
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Depth Z [mm]',
            zaxis=dict(autorange='reversed') # Z down
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_gamma_map(ref_img, eval_img, title="Gamma Map", gamma_crit=(3,3)):
    """
    Plots a 2D difference map and a pseudo-gamma map.
    """
    diff = np.abs(ref_img - eval_img)
    max_ref = np.max(ref_img)
    
    # Simple relative difference map for visualization
    rel_diff = (diff / max_ref) * 100.0
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Difference
    im0 = ax[0].imshow(rel_diff, cmap='bwr', vmin=-10, vmax=10)
    ax[0].set_title("Relative Difference (%)")
    plt.colorbar(im0, ax=ax[0])
    
    # Pass/Fail Map (Binary Gamma proxy)
    # Green = Pass (< 3%), Red = Fail (> 3%)
    tol_perc = gamma_crit[0]
    pass_map = np.where(rel_diff <= tol_perc, 1, 0)
    
    cmap_binary = plt.cm.get_cmap('RdYlGn')
    im1 = ax[1].imshow(pass_map, cmap=cmap_binary, vmin=0, vmax=1)
    ax[1].set_title(f"Pass/Fail Map ({tol_perc}% criteria)")
    
    # Fake legend for Pass/Fail
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Pass'),
                       Patch(facecolor='red', label='Fail')]
    ax[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_dvh(volume, bin_size=0.1):
    """
    Plots Cumulative Dose-Volume Histogram (cDVH).
    """
    doses = volume.flatten()
    doses = doses[doses > 0] # Filter background
    
    if len(doses) == 0: return
    
    doses_rel = (doses / np.max(doses)) * 100.0
    
    bins = np.arange(0, 101, bin_size)
    counts, edges = np.histogram(doses_rel, bins=bins)
    
    # Cumulative sum from right to left
    cum_vol = np.cumsum(counts[::-1])[::-1]
    cum_vol_perc = cum_vol / cum_vol[0] * 100.0
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Metrics
    d90 = np.percentile(doses_rel, 10) # 90% of volume receives at least... (wait, percentiles inverted for DVH definition)
    # Standard def: D90 = Dose that covers 90% of volume
    # This corresponds to the 10th percentile of the dose distribution data
    d95 = np.percentile(doses_rel, 5)
    d5 = np.percentile(doses_rel, 95)
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, cum_vol_perc, 'b-', linewidth=2)
    plt.axvline(d95, color='orange', linestyle='--', label=f'D95: {d95:.1f}%')
    plt.axvline(d5, color='red', linestyle='--', label=f'D5 (Hot): {d5:.1f}%')
    
    plt.xlabel("Dose [%]")
    plt.ylabel("Volume [%]")
    plt.title("Cumulative Dose-Volume Histogram (DVH)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()