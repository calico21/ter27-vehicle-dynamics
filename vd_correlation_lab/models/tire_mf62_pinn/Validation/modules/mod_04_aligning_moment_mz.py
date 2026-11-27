#!/usr/bin/env python3
"""
Module 04: Self-Aligning Moment (Mz) & Pneumatic Trail Analysis
Generates:
  D1: Mz vs alpha across 5 normal loads (2x3 Multi-Panel Layout with in-plot metrics boxes)
  D2: Pneumatic trail t(alpha) and residual aligning torque Mzr(alpha) separation
  D4: Illustrative steering torque feedback (tau_steer = -Mz + c_m * Fy, c_m = 15 mm)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import (
    apply_theme, add_metrics_box, add_characterization_badge,
    LOAD_LEVELS_INFO, COLOR_EXP, COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB,
    STYLE_M1, STYLE_M2, STYLE_M3
)
from core.metrics import compute_metrics
from calibrate_mf62 import Pacejka62Fitter


def generate_mz_vs_alpha_5loads(evaluator, df_val, compound_name, out_dir):
    """
    D1: 2x3 Multi-Panel Layout for Self-Aligning Torque Mz vs alpha across 5 loads.
    Each load panel features low-alpha scatter, binned means +- 1-sigma, and an in-plot metrics box.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5), dpi=150)

    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    a_dense_deg = np.degrees(a_dense)

    # Nominal mask (P ~ 83 kPa, gamma ~ 0 deg)
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
            ax.scatter(np.degrees(sub["alpha"]), sub["m_z"], s=3, alpha=0.15, color=COLOR_EXP, label=f"TTC Points (N={len(sub):,})")

            # Statistically binned experimental means +- 1-sigma
            bins_a = np.linspace(-13.0, 13.0, 27)
            sub_copy = sub.copy()
            sub_copy["a_deg"] = np.degrees(sub_copy["alpha"])
            sub_copy["bin"] = pd.cut(sub_copy["a_deg"], bins_a)
            binned = sub_copy.groupby("bin", observed=True)["m_z"].agg(["mean", "std", "count"]).dropna()
            bin_centers = [b.mid for b in binned.index]

            ax.errorbar(bin_centers, binned["mean"], yerr=binned["std"], fmt="o", markersize=4,
                         color=col, ecolor=col, elinewidth=1.2, capsize=2, alpha=0.85, label="Binned Mean $\\pm 1\\sigma$")

            eval_l = evaluator.eval_dataset(sub)
            metrics_load_mz = {
                "Baseline": compute_metrics(eval_l["mz_exp"], eval_l["mz_m1"]),
                "Optimized": compute_metrics(eval_l["mz_exp"], eval_l["mz_m2"]),
                "Hybrid": compute_metrics(eval_l["mz_exp"], eval_l["mz_m3"])
            }
            add_metrics_box(ax, metrics_load_mz, loc="upper right", title=f"L{idx+1} Mz Metrics ({fz_mean:.0f} N)", unit_force="Nm")

        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_mean), gamma=0.0, p=83.0)
        ax.plot(a_dense_deg, preds["mz_m1"], **STYLE_M1, label="M1: Baseline")
        ax.plot(a_dense_deg, preds["mz_m2"], **STYLE_M2, label="M2: Optimized")
        ax.plot(a_dense_deg, preds["mz_m3"], **STYLE_M3, label="M3: Hybrid")

        ax.axhline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.axvline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.set_title(f"L{idx+1}: $F_z \\approx {fz_mean:.0f}$ N ({l_info['name']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
        ax.set_ylabel("Aligning Torque $M_z$ [Nm]", fontsize=10)
        ax.set_xlim(-14.0, 14.0)
        ax.legend(loc="lower left", fontsize=7.5)
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
                np.degrees(sub["alpha"]), sub["m_z"],
                s=2, alpha=0.15, color=col,
                label=f"Exp ~{fz_mean:.0f} N"
            )

        # Overlay model curve
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_mean), gamma=0.0, p=83.0)
        ax6.plot(a_dense_deg, preds["mz_m2"], "-", color=col, lw=2.2, label=f"M2 ~{fz_mean:.0f} N")

    ax6.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax6.set_title(f"D1: Multi-Load Aligning Torque Progression ({c_upper})", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax6.set_ylabel("Aligning Torque $M_z$ [Nm]", fontsize=10)
    ax6.set_xlim(-14.0, 14.0)
    ax6.legend(ncol=2, loc="lower left", fontsize=7.0)
    ax6.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig04_mz_vs_alpha_5loads_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_pneumatic_trail_and_steering(evaluator, df_val, compound_name, out_dir):
    """
    D2 & D4: Pneumatic trail t(alpha), residual moment Mzr(alpha), and illustrative steering torque feedback.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150)

    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    a_dense_deg = np.degrees(a_dense)
    p2 = evaluator.params_m2
    r0 = evaluator.r0_m2
    fnom = evaluator.fnomin

    # Panel 1: Pneumatic Trail t(alpha) across 5 loads
    for l_info in LOAD_LEVELS_INFO:
        col = l_info["color"]
        fz = l_info["target"]
        dfz = (fz - fnom) / fnom

        Dt = (fz / fnom) * r0 * (p2.get("QDZ1", 0.15) + p2.get("QDZ2", -0.04) * dfz)
        Ct = p2.get("QCZ1", 1.65)
        Bt = (p2.get("QBZ1", 6.0) + p2.get("QBZ2", 1.0) * dfz)
        Et = np.clip((p2.get("QEZ1", 0.05) + p2.get("QEZ2", 0.90) * dfz), -10.0, 1.0)

        t_arg = Bt * a_dense
        trail = Dt * np.cos(Ct * np.arctan(t_arg - Et * (t_arg - np.arctan(t_arg)))) * np.cos(a_dense)
        ax1.plot(a_dense_deg, trail * 1000.0, "-", color=col, lw=2.2, label=f"Fz = {fz:.0f} N (Peak={np.max(trail)*1000:.1f} mm)")

    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"D2: Pneumatic Trail $t(\\alpha)$ across Loads ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Pneumatic Trail $t$ [mm]", fontsize=10)
    ax1.set_xlim(-14.0, 14.0)
    ax1.legend(loc="upper right", fontsize=8.0)
    ax1.grid(True, alpha=0.3)
    add_characterization_badge(ax1, "MODEL CHARACTERIZATION ONLY (BENCH CANNOT MEASURE TRAIL SEPARATELY)")

    # Panel 2: Illustrative Total Steering Torque (tau_steer = -Mz + c_m * Fy, c_m = 15 mm)
    c_m = 0.015 # 15 mm mechanical trail
    for l_info in [LOAD_LEVELS_INFO[1], LOAD_LEVELS_INFO[3]]: # 442 N and 887 N
        col = l_info["color"]
        fz = l_info["target"]
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz), gamma=0.0, p=83.0)

        fy_m2 = preds["fy_m2"]
        mz_m2 = preds["mz_m2"]
        tau_steer = -mz_m2 + c_m * fy_m2

        ax2.plot(a_dense_deg, tau_steer, "-", color=col, lw=2.4, label=f"Total Feedback $\\tau_{{\\text{{steer}}}}$ ({fz:.0f} N)")
        ax2.plot(a_dense_deg, -mz_m2, "--", color=col, lw=1.6, alpha=0.7, label=f"Pneumatic Term $-M_z$ ({fz:.0f} N)")
        ax2.plot(a_dense_deg, c_m * fy_m2, ":", color=col, lw=1.6, alpha=0.7, label=f"Mechanical Term $c_m F_y$ ({fz:.0f} N)")

    ax2.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title(f"D4: Steering Kingpin Torque Feedback ($c_m = 15$ mm) — {c_upper}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Steering Torque [Nm]", fontsize=10)
    ax2.set_xlim(-14.0, 14.0)
    ax2.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig04_pneumatic_trail_and_steering_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
