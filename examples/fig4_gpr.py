import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

def generate_fig3_gpr_principle_v5():
    print("--- 3. Ábra Generálása: 3D GPR Rekonstrukció Elve (Végső Layout) ---")
    
    # ==========================================
    # 1. Szintetikus "Nyaláb" Adatok Generálása
    # ==========================================
    z_dense = np.linspace(0, 40, 200)
    
    def pdd_curve(z):
        return 100 * (1.1 * np.exp(-0.04 * z) - 0.5 * np.exp(-0.3 * z))
    
    d_dense_true = pdd_curve(z_dense)
    
    z_sparse = np.array([1.0, 4.0, 9.0, 18.0, 32.0])
    d_sparse = pdd_curve(z_sparse) + np.random.normal(0, 1.5, len(z_sparse))
    
    x = np.linspace(-15, 15, 50)
    y = np.linspace(-15, 15, 50)
    X, Y = np.meshgrid(x, y)
    
    def beam_2d(z, amplitude):
        width = 4.0 + z * 0.1 
        return amplitude * np.exp(-(X**2 + Y**2) / (2 * width**2))

    # ==========================================
    # 2. Gauss-folyamat (GPR) Illesztése
    # ==========================================
    kernel = 1.0 * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    
    gp.fit(z_sparse.reshape(-1, 1), d_sparse)
    d_pred, d_std = gp.predict(z_dense.reshape(-1, 1), return_std=True)

    # ==========================================
    # 3. Publikációkész Ábra Rajzolása
    # ==========================================
    
    # Constrained layout a legjobb a 3D-hez
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1.2, 0.9, 1.2], figure=fig)
    
    cmap_beam = 'magma' 
    global_levels = np.linspace(0, 90, 20)

    # --- PANEL A: Ritka 2D Filmek ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_title("A. Sparse 2D Measurements\n(Gafchromic Films)", fontsize=14, fontweight='bold', pad=15)
    
    for z, amp in zip(z_sparse, d_sparse):
        Z_plane = np.full_like(X, z)
        C_plane = beam_2d(z, amp)
        ax1.contourf(X, Y, Z_plane, C_plane, zdir='z', offset=z, cmap=cmap_beam, alpha=0.9, levels=global_levels)
        ax1.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], [z, z, z, z, z], color='gray', alpha=0.4)

    ax1.set_xlim(-15, 15); ax1.set_ylim(-15, 15); ax1.set_zlim(0, 40)
    
    # JAVÍTÁS: A labelpad értékek csökkentése, hogy a felirat közelebb húzódjon a tengelyhez
    ax1.set_xlabel("X [mm]", labelpad=5)
    ax1.set_ylabel("Y [mm]", labelpad=5)
    ax1.set_zlabel("Depth Z [mm]", labelpad=0) # Szigorúan behúzzuk a Z tengelyt!
    
    ax1.view_init(elev=20, azim=-45)
    ax1.invert_zaxis() 
    ax1.dist = 12 # JAVÍTÁS: Még jobban kizoomolunk, hogy a doboz biztosan elférjen!

    # --- PANEL B: 1D Gauss-folyamat ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("B. Gaussian Process Regression\n(Matérn Kernel Interpolation)", fontsize=14, fontweight='bold', pad=15)
    
    ax2.plot(z_sparse, d_sparse, 'o', color='darkorange', markeredgecolor='black', markersize=8, zorder=5, label='Measured Films')
    ax2.plot(z_dense, d_pred, '-', color='navy', lw=2.5, label='GP Mean Prediction')
    ax2.fill_between(z_dense, d_pred - 1.96*d_std, d_pred + 1.96*d_std, color='dodgerblue', alpha=0.3, label='95% Confidence Interval')
    
    ax2.set_xlabel("Depth Z [mm]", fontsize=12)
    ax2.set_ylabel("Central Axis Dose [%]", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- PANEL C: Folytonos 3D Térfogat ---
    ax3 = fig.add_subplot(gs[2], projection='3d')
    ax3.set_title("C. Continuous 3D Volume\n(Volumetric Reconstruction)", fontsize=14, fontweight='bold', pad=15)
    
    z_render = np.linspace(0, 40, 50)
    d_render = gp.predict(z_render.reshape(-1, 1))
    
    for z, amp in zip(z_render, d_render):
        Z_plane = np.full_like(X, z)
        C_plane = beam_2d(z, amp)
        C_plane[C_plane < 15] = np.nan 
        ax3.contourf(X, Y, Z_plane, C_plane, zdir='z', offset=z, cmap=cmap_beam, alpha=0.25, levels=global_levels)

    ax3.set_xlim(-15, 15); ax3.set_ylim(-15, 15); ax3.set_zlim(0, 40)
    
    # JAVÍTÁS a legszélső panelen is!
    ax3.set_xlabel("X [mm]", labelpad=5)
    ax3.set_ylabel("Y [mm]", labelpad=5)
    ax3.set_zlabel("Depth Z [mm]", labelpad=0) # Itt lógott le eddig!
    
    ax3.view_init(elev=20, azim=-45)
    ax3.invert_zaxis()
    ax3.dist = 12 # Kamera hátrébb húzása

    filename = "fig4_gpr_principle.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Kész! Ábra elmentve mint: {filename}")
    plt.show()

if __name__ == "__main__":
    generate_fig3_gpr_principle_v5()