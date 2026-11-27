#!/usr/bin/env python3
"""
Module 08: Inflation Pressure Sensitivity Analysis
Generates:
  H1: Fy vs alpha across experimental inflation pressures (70, 83, 95 kPa) with In-Plot Metrics Box
  H2: Mz vs alpha across inflation pressures
  H3: Pressure difference Delta Fy(P) = Fy(P) - Fy(83 kPa)
  H4: Cornering stiffness C_alpha(P) and peak force Fy_max(P) vs continuous pressure
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, add_metrics_box, COLOR_EXP
from core.metrics import compute_metrics
from core.features import extract_lateral_features


def generate_pressure_sensitivity_suite(evaluator, df_val, compound_name, out_dir):
    """
    H1 through H4: 2x2 comprehensive layout for inflation pressure sensitivity.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), dpi=150)

    a_dense = np.linspace(-np.radians(14.0), np.radians(14.0), 301)
    a_dense_deg = np.degrees(a_dense)
    pressures_val = [70.0, 83.0, 95.0]
    p_cols = ["#3498db", "#2ecc71", "#e74c3c"]

    # Filter to nominal load and zero camber
    sub_p_val = df_val[(df_val["f_z"] >= 770.0) & (df_val["f_z"] < 990.0) & (np.abs(df_val["gamma"]) < 0.005)]
    fz_nom = float(np.mean(sub_p_val["f_z"])) if len(sub_p_val) > 0 else 887.0
    eval_p = evaluator.eval_dataset(sub_p_val)

    metrics_p_fy = {
        "Baseline": compute_metrics(eval_p["fy_exp"], eval_p["fy_m1"]),
        "Optimized": compute_metrics(eval_p["fy_exp"], eval_p["fy_m2"]),
        "Hybrid": compute_metrics(eval_p["fy_exp"], eval_p["fy_m3"])
    }
    metrics_p_mz = {
        "Baseline": compute_metrics(eval_p["mz_exp"], eval_p["mz_m1"]),
        "Optimized": compute_metrics(eval_p["mz_exp"], eval_p["mz_m2"]),
        "Hybrid": compute_metrics(eval_p["mz_exp"], eval_p["mz_m3"])
    }

    # Panel 1 (Top Left): Fy vs alpha across pressures
    ax1 = axes[0, 0]
    for p_val, col in zip(pressures_val, p_cols):
        sub = sub_p_val[np.abs(sub_p_val["p_kpa"] - p_val) < 5.0]
        if len(sub) > 0:
            ax1.scatter(np.degrees(sub["alpha"]), sub["f_y"], s=3, alpha=0.15, color=col,
                        label=f"Exp P~{p_val:.0f} kPa")
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_nom), gamma=0.0, p=p_val)
        ax1.plot(a_dense_deg, preds["fy_m2"], "--", color=col, lw=2.0, label=f"M2: Opt ({p_val:.0f} kPa)")
        ax1.plot(a_dense_deg, preds["fy_m3"], "-", color=col, lw=2.4, label=f"M3: Hyb ({p_val:.0f} kPa)")

    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"H1: Lateral Force Inflation Sensitivity $F_y(\\alpha, P)$ at $F_z \\approx {fz_nom:.0f}$ N ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
    ax1.legend(ncol=2, fontsize=7.5, loc="lower right")
    ax1.grid(True, alpha=0.3)
    add_metrics_box(ax1, metrics_p_fy, loc="upper left", title="Fy Pressure Subset")

    # Panel 2 (Top Right): Mz vs alpha across pressures
    ax2 = axes[0, 1]
    for p_val, col in zip(pressures_val, p_cols):
        sub = sub_p_val[np.abs(sub_p_val["p_kpa"] - p_val) < 5.0]
        if len(sub) > 0:
            ax2.scatter(np.degrees(sub["alpha"]), sub["m_z"], s=3, alpha=0.15, color=col,
                        label=f"Exp P~{p_val:.0f} kPa")
        preds = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_nom), gamma=0.0, p=p_val)
        ax2.plot(a_dense_deg, preds["mz_m2"], "--", color=col, lw=2.0, label=f"M2: Opt ({p_val:.0f} kPa)")
        ax2.plot(a_dense_deg, preds["mz_m3"], "-", color=col, lw=2.4, label=f"M3: Hyb ({p_val:.0f} kPa)")

    ax2.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title(f"H2: Aligning Torque Inflation Sensitivity $M_z(\\alpha, P)$ ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Aligning Torque $M_z$ [Nm]", fontsize=10)
    ax2.legend(ncol=2, fontsize=7.5, loc="lower left")
    ax2.grid(True, alpha=0.3)
    add_metrics_box(ax2, metrics_p_mz, loc="upper right", title="Mz Pressure Subset", unit_force="Nm")

    # Panel 3 (Bottom Left): Differential Force Delta Fy(P) relative to 83 kPa
    ax3 = axes[1, 0]
    p_nom_pred = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_nom), gamma=0.0, p=83.0)["fy_m2"]
    for p_val, col in zip([70.0, 95.0], ["#3498db", "#e74c3c"]):
        p_val_pred = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_nom), gamma=0.0, p=p_val)["fy_m2"]
        delta_p = p_val_pred - p_nom_pred
        ax3.plot(a_dense_deg, delta_p, "-", color=col, lw=2.2, label=f"$\\Delta F_y({p_val:.0f} - 83\\text{{ kPa}})$")

    ax3.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.set_title("H3: Lateral Force Difference $\\Delta F_y(P) = F_y(P) - F_y(83\\text{ kPa})$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax3.set_ylabel("$\\Delta F_y$ [N]", fontsize=10)
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    # Panel 4 (Bottom Right): Cornering Stiffness and Peak Force vs Inflation Pressure
    ax4 = axes[1, 1]
    p_cont_range = np.linspace(55.0, 110.0, 50)
    ca_p_list = []
    fymax_p_list = []

    for p_c in p_cont_range:
        pred_p = evaluator.eval_pure_lateral(a_dense, np.full_like(a_dense, fz_nom), gamma=0.0, p=p_c)["fy_m2"]
        feat_p = extract_lateral_features(a_dense, pred_p)
        ca_p_list.append(feat_p["stiffness_n_rad"])
        fymax_p_list.append(feat_p["fy_max_n"])

    ax4_twin = ax4.twinx()
    l1 = ax4.plot(p_cont_range, ca_p_list, "b-", lw=2.2, label="Cornering Stiffness $C_\\alpha(P)$")
    l2 = ax4_twin.plot(p_cont_range, fymax_p_list, "r--", lw=2.0, label="Peak Lateral Force $F_{y,\\max}(P)$")

    ax4.axvline(83.0, color="gray", linestyle=":", label="Nominal Pressure (83 kPa)")
    ax4.set_title(f"H4: Cornering Properties vs Inflation Pressure ({c_upper})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Inflation Pressure $P$ [kPa]", fontsize=10)
    ax4.set_ylabel("Cornering Stiffness $C_\\alpha$ [N/rad]", fontsize=10, color="b")
    ax4_twin.set_ylabel("Peak Lateral Force $F_{y,\\max}$ [N]", fontsize=10, color="r")
    ax4.grid(True, alpha=0.3)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="lower left", fontsize=8.0)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig08_pressure_sensitivity_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
