#!/usr/bin/env python3
"""
Module 02: Pure Lateral Behavior (Fy)
Generates:
  B1: Fy vs alpha across 5 physical loads (2x3 Multi-Panel Layout with Binned Means & In-Plot Metrics Boxes)
  B2: Cornering stiffness C_alpha vs Fz (in N/rad and N/deg)
  B3: Peak lateral force Fy_max and friction coefficient mu_y vs Fz (Load Sensitivity)
  B5: Normalized post-peak force retention |Fy| / |Fy_max| across loads
  B6: Statistically binned zero-slip camber thrust Fy(alpha=0, gamma) with 1-sigma error bars
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import (
    apply_theme, add_metrics_box, LOAD_LEVELS_INFO,
    COLOR_EXP, COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB,
    STYLE_M1, STYLE_M2, STYLE_M3
)
from core.metrics import compute_metrics
from core.features import extract_lateral_features


def generate_fy_vs_alpha_5loads(evaluator, df_val, compound_name, out_dir):
    """
    B1: 2x3 Multi-Panel Layout providing an individual panel for each of the 5 physical normal loads,
    plus a multi-load overview panel. Each load panel contains low-alpha scatter, binned means +- 1-sigma,
    and a strictly subset-evaluated in-plot metrics box.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5), dpi=150)

    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    a_dense_deg = np.degrees(a_dense)

    # Filter to nominal operating envelope (P ~ 83 kPa, gamma ~ 0 deg)
    nom_mask = (np.abs(df_val["gamma"]) < 0.005) & (np.abs(df_val["p_kpa"] - 83.0) < 5.0)
    df_nom = df_val[nom_mask]

    # Subplots 1 to 5: Load Levels L1 to L5
    for idx, l_info in enumerate(LOAD_LEVELS_INFO):
        row = idx // 3
        col_idx = idx % 3
        ax = axes[row, col_idx]

        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        sub = df_nom[(df_nom["f_z"] >= b_lo) & (df_nom["f_z"] < b_hi)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else l_info["target"]

        if len(sub) > 0:
            # Low-alpha background scatter
            ax.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.15, color=COLOR_EXP, label=f"TTC Scatter (N={len(sub):,})")

            # Statistically binned experimental means +- 1-sigma
            bins_a = np.linspace(-13.0, 13.0, 27)
            sub_copy = sub.copy()
            sub_copy["a_deg"] = np.degrees(sub_copy["alpha"])
            sub_copy["bin"] = pd.cut(sub_copy["a_deg"], bins_a)
            binned = sub_copy.groupby("bin", observed=True)["f_y"].agg(["mean", "std", "count"]).dropna()
            bin_centers = [b.mid for b in binned.index]

            ax.errorbar(bin_centers, binned["mean"], yerr=binned["std"], fmt="o", markersize=4,
                         color=col, ecolor=col, elinewidth=1.2, capsize=2, alpha=0.85, label="Binned Mean $\\pm 1\\sigma$")

            # Compute metrics strictly on this load level
            eval_l = evaluator.eval_dataset(sub)
            metrics_load = {
                "Baseline": compute_metrics(eval_l["fy_exp"], eval_l["fy_m1"]),
                "Optimized": compute_metrics(eval_l["fy_exp"], eval_l["fy_m2"]),
                "Hybrid": compute_metrics(eval_l["fy_exp"], eval_l["fy_m3"])
            }
            add_metrics_box(ax, metrics_load, loc="upper left", title=f"L{idx+1} Metrics ({fz_mean:.0f} N)")

        # Model curves
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_mean), gamma=0.0, p=83.0)
        ax.plot(a_dense_deg, preds["fy_m1"], **STYLE_M1, label="M1: Baseline")
        ax.plot(a_dense_deg, preds["fy_m2"], **STYLE_M2, label="M2: Optimized")
        ax.plot(a_dense_deg, preds["fy_m3"], **STYLE_M3, label="M3: Hybrid")

        ax.axhline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.axvline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.set_title(f"L{idx+1}: $F_z \\approx {fz_mean:.0f}$ N ({l_info['name']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
        ax.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
        ax.set_xlim(-14.0, 14.0)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.grid(True, alpha=0.3)

    # Panel 6: Multi-Load Comparison Overlay (All 5 Loads — Full Experimental Points + Models)
    ax6 = axes[1, 2]
    for l_info in LOAD_LEVELS_INFO:
        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        sub = df_nom[(df_nom["f_z"] >= b_lo) & (df_nom["f_z"] < b_hi)]
        fz_mean = float(np.mean(sub["f_z"])) if len(sub) > 0 else l_info["target"]

        # Plot all individual experimental test points for this load level
        if len(sub) > 0:
            ax6.scatter(
                np.degrees(sub["alpha"]), sub["f_y"],
                s=2, alpha=0.15, color=col,
                label=f"Exp ~{fz_mean:.0f} N"
            )

        # Overlay model curve
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_mean), gamma=0.0, p=83.0)
        ax6.plot(a_dense_deg, preds["fy_m2"], "-", color=col, lw=2.2, label=f"M2 ~{fz_mean:.0f} N")

    ax6.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.set_title(f"B1: Multi-Load Progression Summary ({c_upper})", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax6.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
    ax6.set_xlim(-14.0, 14.0)
    ax6.legend(ncol=2, loc="lower right", fontsize=7.0)
    ax6.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig02_fy_vs_alpha_5loads_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_cornering_stiffness_and_peak(evaluator, df_val, compound_name, out_dir):
    """
    B2, B3, B4: Cornering stiffness C_alpha, Peak Force Fy_max, and Friction Coefficient mu_y vs Fz.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)

    fz_range = np.linspace(150.0, 1250.0, 100)
    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)

    c_alpha_m1, c_alpha_m2, c_alpha_m3 = [], [], []
    fy_max_m1, fy_max_m2, fy_max_m3 = [], [], []
    mu_y_m1, mu_y_m2, mu_y_m3 = [], [], []

    for fz in fz_range:
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz), gamma=0.0, p=83.0)
        f1 = extract_lateral_features(a_dense, preds["fy_m1"])
        f2 = extract_lateral_features(a_dense, preds["fy_m2"])
        f3 = extract_lateral_features(a_dense, preds["fy_m3"])

        c_alpha_m1.append(f1["stiffness_n_rad"])
        c_alpha_m2.append(f2["stiffness_n_rad"])
        c_alpha_m3.append(f3["stiffness_n_rad"])

        fy_max_m1.append(f1["fy_max_n"])
        fy_max_m2.append(f2["fy_max_n"])
        fy_max_m3.append(f3["fy_max_n"])

        mu_y_m1.append(f1["fy_max_n"] / fz)
        mu_y_m2.append(f2["fy_max_n"] / fz)
        mu_y_m3.append(f3["fy_max_n"] / fz)

    # Experimental feature extraction per load level
    exp_fz, exp_ca, exp_fymax, exp_mu = [], [], [], []
    nom_mask = (np.abs(df_val["gamma"]) < 0.005) & (np.abs(df_val["p_kpa"] - 83.0) < 5.0)
    df_nom = df_val[nom_mask]

    for l_info in LOAD_LEVELS_INFO:
        b_lo, b_hi = l_info["bin"]
        sub = df_nom[(df_nom["f_z"] >= b_lo) & (df_nom["f_z"] < b_hi)]
        if len(sub) > 100:
            fz_m = float(np.mean(sub["f_z"]))
            bins_a = np.linspace(-np.radians(12.5), np.radians(12.5), 35)
            df_b = sub.groupby(pd.cut(sub["alpha"], bins_a), observed=True)[["alpha", "f_y"]].mean().dropna()
            if len(df_b) > 10:
                feat = extract_lateral_features(df_b["alpha"].values, df_b["f_y"].values)
                exp_fz.append(fz_m)
                exp_ca.append(feat["stiffness_n_rad"])
                exp_fymax.append(feat["fy_max_n"])
                exp_mu.append(feat["fy_max_n"] / fz_m)

    # Panel 1: Cornering Stiffness C_alpha vs Fz (N/rad and N/deg)
    ax1 = axes[0]
    if len(exp_fz) > 0:
        ax1.scatter(exp_fz, exp_ca, color=COLOR_EXP, s=55, zorder=5, label="Exp TTC Points")
    ax1.plot(fz_range, c_alpha_m1, **STYLE_M1, label="M1: Baseline")
    ax1.plot(fz_range, c_alpha_m2, **STYLE_M2, label="M2: Optimized")
    ax1.plot(fz_range, c_alpha_m3, **STYLE_M3, label="M3: Hybrid")
    ax1.set_title(f"B2: Cornering Stiffness $C_\\alpha(F_z)$ — {c_upper}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax1.set_ylabel("Cornering Stiffness $C_\\alpha$ [N/rad]", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Peak Lateral Force Fy_max vs Fz
    ax2 = axes[1]
    if len(exp_fz) > 0:
        ax2.scatter(exp_fz, exp_fymax, color=COLOR_EXP, s=55, zorder=5, label="Exp TTC Points")
    ax2.plot(fz_range, fy_max_m1, **STYLE_M1, label="M1: Baseline")
    ax2.plot(fz_range, fy_max_m2, **STYLE_M2, label="M2: Optimized")
    ax2.plot(fz_range, fy_max_m3, **STYLE_M3, label="M3: Hybrid")
    ax2.set_title(f"B3: Peak Lateral Force $F_{{y,\\max}}(F_z)$ — {c_upper}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.set_ylabel("Peak Lateral Force $F_{y,\\max}$ [N]", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Friction Coefficient mu_y vs Fz (Load Sensitivity)
    ax3 = axes[2]
    if len(exp_fz) > 0:
        ax3.scatter(exp_fz, exp_mu, color=COLOR_EXP, s=55, zorder=5, label="Exp TTC Points")
    ax3.plot(fz_range, mu_y_m1, **STYLE_M1, label="M1: Baseline")
    ax3.plot(fz_range, mu_y_m2, **STYLE_M2, label="M2: Optimized")
    ax3.plot(fz_range, mu_y_m3, **STYLE_M3, label="M3: Hybrid")
    ax3.set_title(f"B4: Lateral Friction $\\mu_y(F_z)$ [Load Sensitivity]", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax3.set_ylabel("Friction Coeff $\\mu_y = F_{y,\\max}/F_z$", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig02_stiffness_and_load_sensitivity_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_post_peak_and_zero_slip(evaluator, df_val, compound_name, out_dir):
    """
    B5 & B6: Normalized post-peak force retention |Fy| / |Fy_max| and
    statistically binned zero-slip camber thrust with 1-sigma error bars.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=150)

    # Panel 1: Normalized Post-Peak Retention Ratio |Fy| / |Fy_max| (alpha in 6 to 14 deg)
    a_dense = np.linspace(np.radians(6.0), np.radians(14.0), 150)
    a_dense_deg = np.degrees(a_dense)
    target_loads = [LOAD_LEVELS_INFO[1], LOAD_LEVELS_INFO[3]] # L2 (~442N) & L4 (~887N)

    for l_info in target_loads:
        col = l_info["color"]
        fz_l = l_info["target"]

        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_l), gamma=0.0, p=83.0)
        # Full sweep for peak normalization
        a_full = np.linspace(0.0, np.radians(14.0), 200)
        p_full = evaluator.eval_pure_lateral(a_full, np.full_like(a_full, fz_l), gamma=0.0, p=83.0)
        peak_m1 = np.max(p_full["fy_m1"])
        peak_m2 = np.max(p_full["fy_m2"])
        peak_m3 = np.max(p_full["fy_m3"])

        ax1.plot(a_dense_deg, preds["fy_m1"] / peak_m1, ":", color=col, lw=1.8, label=f"M1: Baseline ({fz_l:.0f} N)")
        ax1.plot(a_dense_deg, preds["fy_m2"] / peak_m2, "--", color=col, lw=2.0, label=f"M2: Opt ({fz_l:.0f} N)")
        ax1.plot(a_dense_deg, preds["fy_m3"] / peak_m3, "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_l:.0f} N)")

    ax1.set_title(f"B5: Normalized Post-Peak Force Retention $|F_y| / |F_{{y,\\max}}|$ ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Normalized Grip $|F_y| / |F_{y,\\max}|$", fontsize=10)
    ax1.set_ylim(0.70, 1.02)
    ax1.legend(fontsize=8.0, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Statistically Binned Zero-Slip Force (|alpha| <= 0.10 deg) vs Camber
    g_dense = np.linspace(-np.radians(4.5), np.radians(4.5), 101)
    g_dense_deg = np.degrees(g_dense)

    for l_info in target_loads:
        b_lo, b_hi = l_info["bin"]
        col = l_info["color"]
        fz_l = l_info["target"]

        # Strict zero slip filter: |alpha| <= 0.10 deg
        sub_z = df_val[(df_val["f_z"] >= b_lo) & (df_val["f_z"] < b_hi) & (np.abs(df_val["alpha"]) <= np.radians(0.10))]
        if len(sub_z) > 0:
            # Low-alpha background points
            ax2.scatter(np.degrees(sub_z["gamma"]), sub_z["f_y"], s=10, alpha=0.20, color=col)

            # Binned means +- 1-sigma across discrete cambers
            sub_z_c = sub_z.copy()
            sub_z_c["g_deg"] = np.degrees(sub_z_c["gamma"]).round(0)
            binned_g = sub_z_c.groupby("g_deg", observed=True)["f_y"].agg(["mean", "std", "count"]).dropna()

            ax2.errorbar(binned_g.index, binned_g["mean"], yerr=binned_g["std"], fmt="s", markersize=6,
                         color=col, ecolor=col, elinewidth=1.5, capsize=3,
                         label=f"Exp Binned Mean $\\pm 1\\sigma$ (~{fz_l:.0f} N)")

        p_z = evaluator.eval_pure_lateral(np.zeros_like(g_dense), np.full_like(g_dense, fz_l), gamma=g_dense, p=83.0)
        ax2.plot(g_dense_deg, p_z["fy_m1"], ":", color=col, lw=1.8, label=f"M1: Baseline ({fz_l:.0f} N)")
        ax2.plot(g_dense_deg, p_z["fy_m2"], "--", color=col, lw=2.0, label=f"M2: Opt ({fz_l:.0f} N)")
        ax2.plot(g_dense_deg, p_z["fy_m3"], "-", color=col, lw=2.4, label=f"M3: Hybrid ({fz_l:.0f} N)")

    ax2.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title("B6: Zero-Slip Camber Thrust $F_y(\\alpha=0, \\gamma)$ [Binned $|\\alpha| \\leq 0.10^\\circ$]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax2.set_ylabel("Lateral Force $F_y(0)$ [N]", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig02_post_peak_and_zero_slip_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
