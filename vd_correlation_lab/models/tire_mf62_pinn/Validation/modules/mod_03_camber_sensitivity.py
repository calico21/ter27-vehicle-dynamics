#!/usr/bin/env python3
"""
Module 03: Camber Sensitivity Analysis
Generates:
  C1: Fy vs alpha across 5 camber angles (2x3 Multi-Panel Layout with in-plot metrics boxes)
  C2: Zero-slip camber thrust with binned means and 1-sigma error bars
  C3: Peak lateral force and cornering stiffness sensitivity to camber
  C5: 2D Camber residual error heatmap
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import (
    apply_theme, add_metrics_box, COLOR_EXP,
    COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB,
    STYLE_M1, STYLE_M2, STYLE_M3, LOAD_LEVELS_INFO
)
from core.metrics import compute_metrics
from core.features import extract_lateral_features


def generate_camber_sensitivity_suite(evaluator, df_val, compound_name, out_dir):
    """
    C1 & C2: 2x3 Multi-Panel Layout for 5 discrete camber angles,
    plus Panel 6 showing zero-slip camber thrust with statistical error bars.
    Loads are computed dynamically from each data subset.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5), dpi=150)

    cambers_deg = [-4.0, -2.0, 0.0, 2.0, 4.0]
    c_cols = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    a_dense_deg = np.degrees(a_dense)

    # Filter to nominal load and pressure
    sub_camber_val = df_val[(df_val["f_z"] >= 770.0) & (df_val["f_z"] < 990.0) & (np.abs(df_val["p_kpa"] - 83.0) < 5.0)]
    fz_nom_actual = float(np.mean(sub_camber_val["f_z"])) if len(sub_camber_val) > 0 else 887.0

    # Panels 1 to 5: Individual Camber Angles
    for idx, (g_deg, col) in enumerate(zip(cambers_deg, c_cols)):
        row = idx // 3
        col_idx = idx % 3
        ax = axes[row, col_idx]

        g_rad = np.radians(g_deg)
        sub = sub_camber_val[np.abs(np.degrees(sub_camber_val["gamma"]) - g_deg) < 0.4]
        fz_actual = float(np.mean(sub["f_z"])) if len(sub) > 0 else fz_nom_actual

        if len(sub) > 0:
            # Low-alpha scatter
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.15, color=COLOR_EXP, label=f"TTC Points (N={len(sub):,})")

            # Binned means +- 1-sigma
            bins_a = np.linspace(-13.0, 13.0, 27)
            sub_copy = sub.copy()
            sub_copy["a_deg"] = np.degrees(sub_copy["alpha"])
            sub_copy["bin"] = pd.cut(sub_copy["a_deg"], bins_a)
            binned = sub_copy.groupby("bin", observed=True)["f_y"].agg(["mean", "std", "count"]).dropna()
            bin_centers = [b.mid for b in binned.index]

            ax.errorbar(bin_centers, binned["mean"], yerr=binned["std"], fmt="o", markersize=4,
                         color=col, ecolor=col, elinewidth=1.2, capsize=2, alpha=0.85, label="Binned Mean $\\pm 1\\sigma$")

            # In-plot metrics box strictly for this camber angle
            eval_c = evaluator.eval_dataset(sub)
            metrics_camber = {
                "Baseline": compute_metrics(eval_c["fy_exp"], eval_c["fy_m1"]),
                "Optimized": compute_metrics(eval_c["fy_exp"], eval_c["fy_m2"]),
                "Hybrid": compute_metrics(eval_c["fy_exp"], eval_c["fy_m3"])
            }
            add_metrics_box(ax, metrics_camber, loc="upper left", title=f"γ = {g_deg:+.0f}° ({fz_actual:.0f} N)")

        # Model curves dynamically evaluated at this panel's actual normal load
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_actual), gamma=g_rad, p=83.0)
        ax.plot(a_dense_deg, preds["fy_m1"], **STYLE_M1, label="M1: Baseline" + (" (Collapsed)" if g_deg == -4.0 else ""))
        ax.plot(a_dense_deg, preds["fy_m2"], **STYLE_M2, label="M2: Optimized")
        ax.plot(a_dense_deg, preds["fy_m3"], **STYLE_M3, label="M3: Hybrid")

        ax.axhline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.axvline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.set_title(f"C1: Camber $\\gamma = {g_deg:+.0f}^\\circ$ ($F_z \\approx {fz_actual:.0f}$ N) — {c_upper}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
        ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
        ax.set_xlim(-14.0, 14.0)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.grid(True, alpha=0.3)

    # Panel 6: Zero-Slip Camber Thrust Resolution (alpha <= 0.10 deg)
    ax6 = axes[1, 2]
    g_dense = np.linspace(-np.radians(4.5), np.radians(4.5), 101)
    g_dense_deg = np.degrees(g_dense)

    for l_info in [LOAD_LEVELS_INFO[1], LOAD_LEVELS_INFO[3]]:
        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        sub_c0 = df_val[(df_val["f_z"] >= b_lo) & (df_val["f_z"] < b_hi) & (np.abs(df_val["alpha"]) <= np.radians(0.10))]
        fz_actual_c0 = float(np.mean(sub_c0["f_z"])) if len(sub_c0) > 0 else l_info["target"]

        if len(sub_c0) > 0:
            ax6.scatter(np.degrees(sub_c0["gamma"]), sub_c0["f_y"], s=8, alpha=0.20, color=col)
            sub_c0_copy = sub_c0.copy()
            sub_c0_copy["g_deg"] = np.degrees(sub_c0_copy["gamma"]).round(0)
            binned_c0 = sub_c0_copy.groupby("g_deg", observed=True)["f_y"].agg(["mean", "std", "count"]).dropna()
            ax6.errorbar(binned_c0.index, binned_c0["mean"], yerr=binned_c0["std"], fmt="s", markersize=6,
                         color=col, ecolor=col, elinewidth=1.5, capsize=3, label=f"Exp Binned Mean (~{fz_actual_c0:.0f} N)")

        p_c0 = evaluator.eval_pure_lateral(np.zeros_like(g_dense), np.full_like(g_dense, fz_actual_c0), gamma=g_dense, p=83.0)
        ax6.plot(g_dense_deg, p_c0["fy_m1"], ":", color=col, lw=1.6, label=f"M1: Base ({fz_actual_c0:.0f} N)")
        ax6.plot(g_dense_deg, p_c0["fy_m2"], "--", color=col, lw=2.0, label=f"M2: Opt ({fz_actual_c0:.0f} N)")
        ax6.plot(g_dense_deg, p_c0["fy_m3"], "-", color=col, lw=2.4, label=f"M3: Hyb ({fz_actual_c0:.0f} N)")

    ax6.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.set_title("C2: Zero-Slip Camber Thrust Resolution ($|\\alpha| \\leq 0.10^\\circ$)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax6.set_ylabel("Zero-Slip Lateral Force $F_y(0)$ [N]", fontsize=10)
    ax6.legend(ncol=2, fontsize=7.5, loc="upper left")
    ax6.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig03_camber_sensitivity_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_camber_derivatives_and_heatmaps(evaluator, df_val, compound_name, out_dir):
    """
    C3, C4, C5: Peak force camber sensitivity dFy_max/dgamma and 2D camber residual error scatter map.
    Loads are computed dynamically from data subsets.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=150)

    # Panel 1: Peak Lateral Force vs Camber Angle across loads
    gamma_range = np.linspace(-np.radians(4.5), np.radians(4.5), 50)
    gamma_range_deg = np.degrees(gamma_range)
    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)

    for l_info in [LOAD_LEVELS_INFO[1], LOAD_LEVELS_INFO[3]]:
        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        sub_l = df_val[(df_val["f_z"] >= b_lo) & (df_val["f_z"] < b_hi)]
        fz_l_actual = float(np.mean(sub_l["f_z"])) if len(sub_l) > 0 else l_info["target"]
        fymax_m1, fymax_m2, fymax_m3 = [], [], []

        for g in gamma_range:
            p = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_l_actual), gamma=g, p=83.0)
            fymax_m1.append(np.max(p["fy_m1"]))
            fymax_m2.append(np.max(p["fy_m2"]))
            fymax_m3.append(np.max(p["fy_m3"]))

        ax1.plot(gamma_range_deg, fymax_m1, ":", color=col, lw=1.8, label=f"M1: Baseline ({fz_l_actual:.0f} N)")
        ax1.plot(gamma_range_deg, fymax_m2, "--", color=col, lw=2.0, label=f"M2: Opt ({fz_l_actual:.0f} N)")
        ax1.plot(gamma_range_deg, fymax_m3, "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_l_actual:.0f} N)")

    ax1.set_title(f"C3: Peak Lateral Force $F_{{y,\\max}}(\\gamma)$ Sensitivity ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax1.set_ylabel("Peak Lateral Force $F_{y,\\max}$ [N]", fontsize=10)
    ax1.legend(loc="lower center", fontsize=8.0)
    ax1.grid(True, alpha=0.3)

    # Panel 2: 2D Residual Error Scatter Map (alpha vs gamma)
    sub_val = df_val[(df_val["f_z"] >= 770.0) & (df_val["f_z"] < 990.0) & (np.abs(df_val["p_kpa"] - 83.0) < 5.0)]
    fz_res_actual = float(np.mean(sub_val["f_z"])) if len(sub_val) > 0 else 887.0
    eval_sub = evaluator.eval_dataset(sub_val)
    err_m2 = np.abs(eval_sub["fy_m2"] - eval_sub["fy_exp"])

    # Clean scatter plot with colormap replacing hexbin
    sc = ax2.scatter(
        np.degrees(eval_sub["alpha"]), np.degrees(eval_sub["gamma"]),
        c=err_m2, cmap="YlOrRd", s=10, alpha=0.55, vmin=0, vmax=200
    )
    cb = fig.colorbar(sc, ax=ax2)
    cb.set_label("Absolute Residual $|F_{y,\\text{opt}} - F_{y,\\text{exp}}|$ [N]")

    ax2.set_title(f"C5: 2D Camber Absolute Residual Map ($F_z \\approx {fz_res_actual:.0f}$ N) — {c_upper}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax2.set_xlim(-14.0, 14.0)
    ax2.set_ylim(-5.0, 5.0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig03_camber_derivatives_and_heatmaps_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
