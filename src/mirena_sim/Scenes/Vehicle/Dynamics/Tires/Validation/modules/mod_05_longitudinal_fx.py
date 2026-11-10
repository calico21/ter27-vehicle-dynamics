#!/usr/bin/env python3
"""
Module 05: Longitudinal Tire Behavior (Fx)
Generates:
  E1: Longitudinal force curves Fx(kappa) across 5 normal loads
  E2: Longitudinal stiffness C_kappa vs Fz
  E3: Peak Fx_max and longitudinal friction coefficient mu_x vs Fz
PROMINENTLY LABELED AS MODEL CHARACTERIZATION ONLY (no TTC drive/brake sweeps).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, add_characterization_badge, LOAD_LEVELS_INFO
from calibrate_mf62 import Pacejka62Fitter


def generate_longitudinal_characterization(evaluator, compound_name, out_dir):
    """
    E1, E2, E3: Fx vs slip ratio kappa, longitudinal stiffness C_kappa, and friction mu_x vs Fz.
    Avoids confusing dual axes by utilizing 3 dedicated, clean subplots.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)

    k_dense = np.linspace(-0.25, 0.25, 301)
    p2 = evaluator.params_m2

    # Panel 1: Fx vs kappa across 5 loads
    ax1 = axes[0]
    for l_info in LOAD_LEVELS_INFO:
        col = l_info["color"]
        fz = l_info["target"]
        fx_c = Pacejka62Fitter.compute_fx0(p2, k_dense, np.full_like(k_dense, fz), gamma=0.0, p_kpa=83.0)
        ax1.plot(k_dense * 100.0, fx_c, "-", color=col, lw=2.2, label=f"Fz = {fz:.0f} N (Peak={np.max(fx_c):.0f} N)")

    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"E1: Longitudinal Force $F_x(\\kappa)$ across 5 Loads ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=10)
    ax1.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(True, alpha=0.3)
    add_characterization_badge(ax1, "MODEL CHARACTERIZATION ONLY (NO TTC DRIVE/BRAKE DATA)")

    # Compute sensitivities across continuous Fz
    fz_range = np.linspace(150.0, 1250.0, 100)
    c_kappa_list = []
    fxmax_list = []
    mu_x_list = []

    for fz in fz_range:
        fx_vals = Pacejka62Fitter.compute_fx0(p2, k_dense, np.full_like(k_dense, fz), gamma=0.0, p_kpa=83.0)
        idx_0 = np.argmin(np.abs(k_dense))
        c_k = float((fx_vals[idx_0 + 1] - fx_vals[idx_0 - 1]) / (k_dense[idx_0 + 1] - k_dense[idx_0 - 1]))
        c_kappa_list.append(c_k / 1000.0) # kN/slip
        fxmax_list.append(np.max(fx_vals))
        mu_x_list.append(np.max(fx_vals) / fz)

    # Panel 2: Longitudinal Slip Stiffness C_kappa vs Fz
    ax2 = axes[1]
    ax2.plot(fz_range, c_kappa_list, "b-", lw=2.2, label="Slip Stiffness $C_\\kappa$")
    ax2.set_title(f"E2: Slip Stiffness $C_\\kappa(F_z)$ ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.set_ylabel("Slip Stiffness $C_\\kappa$ [kN/slip]", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    add_characterization_badge(ax2, "LITERATURE PARAMETERS (PKX1, PKX2)")

    # Panel 3: Peak Force Fx_max and Friction mu_x vs Fz
    ax3 = axes[2]
    ax3.plot(fz_range, mu_x_list, "r-", lw=2.2, label="Longitudinal Friction $\\mu_x = F_{x,\\max}/F_z$")
    ax3.set_title(f"E3: Peak Longitudinal Friction $\\mu_x(F_z)$ ({c_upper})", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax3.set_ylabel("Friction Coefficient $\\mu_x$", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(True, alpha=0.3)
    add_characterization_badge(ax3, "LITERATURE PARAMETERS (PCX1, PDX1)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig05_longitudinal_characterization_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
