#!/usr/bin/env python3
"""
Module 07: Other Forces & Moments (Mx, My, and Vertical Stiffness Kz)
Generates:
  G1: Overturning moment Mx vs Camber and Normal Load with Binned Means & Metrics Box
  G3: Vertical load-deflection characterization from TTC Spring runs with residual analysis
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import apply_theme, add_metrics_box, LOAD_LEVELS_INFO, COLOR_EXP
from core.metrics import compute_metrics
from core.features import extract_vertical_features
from calibrate_mf62 import Pacejka62Fitter


def generate_mx_and_vertical_stiffness(evaluator, df_train, df_spring, compound_name, out_dir):
    """
    G1 & G3: Overturning moment Mx vs camber with binned error bars and experimental vertical stiffness.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150)

    p2 = evaluator.params_m2
    r0 = evaluator.r0_m2

    # Panel 1: Overturning Moment Mx vs Camber Angle gamma across Loads
    gamma_sweep_rad = np.linspace(-np.radians(5.0), np.radians(5.0), 101)
    gamma_sweep_deg = np.degrees(gamma_sweep_rad)

    # Subset where alpha is near zero
    sub_mx = df_train[(np.abs(df_train["alpha"]) < 0.01) & (np.abs(df_train["p_kpa"] - 83.0) < 5.0)]
    fy_zero_sub, _ = Pacejka62Fitter.compute_fy0(p2, sub_mx["alpha"].values, sub_mx["f_z"].values, sub_mx["gamma"].values, sub_mx["p_kpa"].values)
    mx_pred_sub = Pacejka62Fitter.compute_mx(p2, sub_mx["f_z"].values, sub_mx["gamma"].values, fy_zero_sub, sub_mx["p_kpa"].values, r0=r0)

    metrics_mx = {
        "MF 6.2": compute_metrics(sub_mx["m_x"].values, mx_pred_sub)
    }

    for l_info in LOAD_LEVELS_INFO:
        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        sub = sub_mx[(sub_mx["f_z"] >= b_lo) & (sub_mx["f_z"] < b_hi)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else l_info["target"]

        if len(sub) > 0:
            # Low-alpha background points
            ax1.scatter(np.degrees(sub["gamma"]), sub["m_x"], s=5, alpha=0.15, color=col)

            # Statistically binned means +- 1-sigma across discrete cambers
            sub_c = sub.copy()
            sub_c["g_deg"] = np.degrees(sub_c["gamma"]).round(0)
            binned_mx = sub_c.groupby("g_deg", observed=True)["m_x"].agg(["mean", "std", "count"]).dropna()
            ax1.errorbar(binned_mx.index, binned_mx["mean"], yerr=binned_mx["std"], fmt="o", markersize=5,
                         color=col, ecolor=col, elinewidth=1.2, capsize=2, alpha=0.85,
                         label=f"Exp Binned Mean (~{fz_mean:.0f} N)")

        fy_zero, _ = Pacejka62Fitter.compute_fy0(p2, 0.0, fz_mean, gamma_sweep_rad, 83.0)
        mx_pred = Pacejka62Fitter.compute_mx(p2, fz_mean, gamma_sweep_rad, fy_zero, 83.0, r0=r0)
        ax1.plot(gamma_sweep_deg, mx_pred, "-", color=col, lw=2.0, label=f"MF 6.2 ({fz_mean:.0f} N)")

    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"G1: Overturning Moment $M_x$ vs Camber $\\gamma$ ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax1.set_ylabel("Overturning Moment $M_x$ [Nm]", fontsize=10)
    ax1.legend(ncol=2, fontsize=7.5, loc="lower right")
    ax1.grid(True, alpha=0.3)
    add_metrics_box(ax1, metrics_mx, loc="upper left", title="Overturning Moment Metrics", unit_force="Nm")

    # Panel 2: Vertical Stiffness Characterization from TTC Spring Data
    rho_m = r0 - df_spring["r_l"].values
    rho_mm = rho_m * 1000.0
    fz_spring = df_spring["f_z"].values

    ax2.scatter(rho_mm, fz_spring, s=2, alpha=0.15, color="#16a085", label=f"TTC Spring Test ({len(df_spring):,} pts)")
    rho_line = np.linspace(0.0, max(np.max(rho_m), 0.015), 100)
    kz_fit = p2.get("VERTICAL_STIFFNESS", 115000.0)
    fz_fit_line = kz_fit * rho_line

    feat_vert = extract_vertical_features(rho_m, fz_spring)
    ax2.plot(rho_line * 1000.0, fz_fit_line, "r-", lw=2.4,
             label=f"Calibrated $k_z = {kz_fit:.0f}$ N/m ({kz_fit/1000:.1f} N/mm)\n$R^2 = {feat_vert['r2']:.3f}$ | RMSE = {feat_vert['rmse_mm']:.2f} mm")

    # Binned means across deflection
    bins_rho = np.linspace(0.0, max(np.max(rho_mm), 12.0), 15)
    df_sp_copy = df_spring.copy()
    df_sp_copy["rho_mm"] = rho_mm
    df_sp_copy["bin"] = pd.cut(df_sp_copy["rho_mm"], bins_rho)
    binned_sp = df_sp_copy.groupby("bin", observed=True)[["rho_mm", "f_z"]].mean().dropna()
    ax2.plot(binned_sp["rho_mm"], binned_sp["f_z"], "ko", markersize=5, label="Binned Deflection Means")

    ax2.set_title(f"G3: Vertical Load-Deflection Characterization ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Radial Tire Deflection $\\rho = R_0 - R_l$ [mm]", fontsize=10)
    ax2.set_ylabel("Normal Force $F_z$ [N]", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # Inset / text explanation of deflection
    ax2.text(0.55, 0.15, "Contact patch radial deflection\n$\\rho = R_0 - R_l$ derived directly\nfrom loaded drum spindle radius.\nLinear spring model valid up to 1.2 kN.",
             transform=ax2.transAxes, fontsize=8.0, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"))

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig07_other_forces_moments_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
