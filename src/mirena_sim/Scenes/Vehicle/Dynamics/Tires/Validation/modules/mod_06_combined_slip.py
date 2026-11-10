#!/usr/bin/env python3
"""
Module 06: Combined Slip & Friction Ellipse Analysis
Generates:
  F1: Friction ellipse diagrams Fx-Fy across slip angles and slip ratios
  F2: Lateral force reduction factor G_y_kappa
  F3: Longitudinal force reduction factor G_x_alpha
PROMINENTLY LABELED AS MODEL CHARACTERIZATION ONLY (Reference parameters).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, add_characterization_badge
from calibrate_mf62 import Pacejka62Fitter


def generate_combined_slip_suite(evaluator, compound_name, out_dir):
    """
    F1, F2, F3: Friction ellipse and combined slip reduction factors (Gy_kappa and Gx_alpha).
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=150)

    p2 = evaluator.params_m2
    fz_nom = 887.0

    # Panel 1: Friction Ellipse Fx vs Fy at multiple slip angles
    ax1 = axes[0]
    alpha_sweeps_deg = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    kappa_range = np.linspace(-0.25, 0.25, 150)
    cols = ["#7f8c8d", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    for a_deg, col in zip(alpha_sweeps_deg, cols):
        a_rad = np.radians(a_deg)
        fx_list = []
        fy_list = []
        for k in kappa_range:
            comb = Pacejka62Fitter.compute_combined_forces(p2, k, a_rad, fz_nom, gamma=0.0, p_kpa=83.0)
            fx_list.append(comb["Fx"])
            fy_list.append(comb["Fy"])

        ax1.plot(fy_list, fx_list, "-", color=col, lw=2.0, label=f"$\\alpha = {a_deg:.0f}^\\circ$")

    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"F1: Friction Ellipse at $F_z = {fz_nom:.0f}$ N, $P = 83$ kPa ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Lateral Force $F_y$ [N]", fontsize=10)
    ax1.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(True, alpha=0.3)
    add_characterization_badge(ax1, "MODEL CHARACTERIZATION (DEFAULT PARAMS)")

    # Panel 2: Lateral Force Reduction Factor G_y_kappa = Fy(alpha, kappa) / Fy(alpha, 0)
    ax2 = axes[1]
    k_pos_range = np.linspace(0.0, 0.20, 100)
    for a_deg, col in zip([2.0, 4.0, 8.0], ["#2ca02c", "#ff7f0e", "#d62728"]):
        a_rad = np.radians(a_deg)
        comb_pure = Pacejka62Fitter.compute_combined_forces(p2, 0.0, a_rad, fz_nom, gamma=0.0, p_kpa=83.0)
        fy_pure = comb_pure["Fy"]
        g_y_list = []
        for k in k_pos_range:
            comb_comb = Pacejka62Fitter.compute_combined_forces(p2, -k, a_rad, fz_nom, gamma=0.0, p_kpa=83.0) # braking
            fy_comb = comb_comb["Fy"]
            g_y = float(fy_comb / max(fy_pure, 1e-4))
            g_y_list.append(g_y)
        ax2.plot(k_pos_range * 100.0, g_y_list, "-", color=col, lw=2.2, label=f"Braking $\\alpha = {a_deg:.0f}^\\circ$")

    ax2.axhline(1.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title("F2: Lateral Grip Reduction $G_{y\\kappa} = F_y(\\alpha, \\kappa) / F_y(\\alpha, 0)$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Longitudinal Slip Ratio $|\\kappa|$ [%]", fontsize=10)
    ax2.set_ylabel("Normalized Lateral Grip $G_{y\\kappa}$", fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    add_characterization_badge(ax2, "COMBINED LATERAL FACTOR (RBY, RCY)")

    # Panel 3: Longitudinal Force Reduction Factor G_x_alpha = Fx(kappa, alpha) / Fx(kappa, 0)
    ax3 = axes[2]
    a_pos_deg = np.linspace(0.0, 12.0, 100)
    for k_val, col in zip([0.05, 0.10, 0.15], ["#1f77b4", "#e67e22", "#d62728"]):
        comb_pure = Pacejka62Fitter.compute_combined_forces(p2, -k_val, 0.0, fz_nom, gamma=0.0, p_kpa=83.0)
        fx_pure = comb_pure["Fx"]
        g_x_list = []
        for a_deg in a_pos_deg:
            comb_comb = Pacejka62Fitter.compute_combined_forces(p2, -k_val, np.radians(a_deg), fz_nom, gamma=0.0, p_kpa=83.0)
            fx_comb = comb_comb["Fx"]
            g_x = float(fx_comb / max(abs(fx_pure), 1e-4))
            g_x_list.append(abs(g_x))
        ax3.plot(a_pos_deg, g_x_list, "-", color=col, lw=2.2, label=f"Longitudinal Slip $|\\kappa| = {k_val*100:.0f}\\%$")

    ax3.axhline(1.0, color="gray", lw=0.6, linestyle=":")
    ax3.set_title("F3: Longitudinal Grip Reduction $G_{x\\alpha} = F_x(\\kappa, \\alpha) / F_x(\\kappa, 0)$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax3.set_ylabel("Normalized Longitudinal Grip $G_{x\\alpha}$", fontsize=10)
    ax3.set_ylim(0.0, 1.05)
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.grid(True, alpha=0.3)
    add_characterization_badge(ax3, "COMBINED LONGITUDINAL FACTOR (RBX, RCX)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig06_combined_slip_friction_ellipse_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
