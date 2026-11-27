#!/usr/bin/env python3
"""
Module 11: Hybrid Residual PINN Analysis
Generates:
  K1 & K2: Residual correction magnitude & 2D correction heatmaps (Delta Fy, Delta Mz)
  K3: Relative correction ratio r_Fy = |Delta Fy| / max(|Fy_MF|, 50 N)
  K4: Model improvement map Delta RMSE = RMSE_hyb - RMSE_opt across domain slices
  K6: Physical consistency verification (Zero-Slip Passivity, Airborne Suppression, Domain Falloff)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, COLOR_M2_OPT, COLOR_M3_HYB


def generate_pinn_correction_maps(evaluator, compound_name, out_dir):
    """
    K1, K2, K3: Residual correction magnitude Delta Fy(alpha, Fz), Delta Mz, and relative ratio r_Fy.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)

    # 2D Grid
    alpha_grid = np.linspace(-np.radians(14.0), np.radians(14.0), 100)
    fz_grid = np.linspace(150.0, 1250.0, 80)
    A, FZ = np.meshgrid(alpha_grid, fz_grid)

    dfy, dmz = evaluator.eval_pinn(A, FZ, np.zeros_like(A), np.full_like(A, 83.0))

    # Evaluate physical MF lateral force
    fy_mf = evaluator.eval_pure_lateral(A, FZ, gamma=0.0, p=83.0)["fy_m2"]
    r_fy = np.abs(dfy) / np.maximum(np.abs(fy_mf), 50.0) * 100.0

    # Panel 1: Delta Fy(alpha, Fz)
    ax1 = axes[0]
    c1 = ax1.contourf(np.degrees(A), FZ, dfy, levels=25, cmap="coolwarm", vmin=-150, vmax=150)
    fig.colorbar(c1, ax=ax1, label="PINN Correction $\\Delta F_y$ [N]")
    ax1.set_title(f"K1/K2: Neural Correction $\\Delta F_y(\\alpha, F_z)$ ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Normal Load $F_z$ [N]", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Delta Mz(alpha, Fz)
    ax2 = axes[1]
    c2 = ax2.contourf(np.degrees(A), FZ, dmz, levels=25, cmap="coolwarm", vmin=-6.0, vmax=6.0)
    fig.colorbar(c2, ax=ax2, label="PINN Correction $\\Delta M_z$ [Nm]")
    ax2.set_title(f"K2: Neural Correction $\\Delta M_z(\\alpha, F_z)$ ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Relative Correction Ratio r_Fy [%]
    ax3 = axes[2]
    c3 = ax3.contourf(np.degrees(A), FZ, r_fy, levels=np.linspace(0, 25, 26), cmap="viridis")
    fig.colorbar(c3, ax=ax3, label="Relative Correction $r_{Fy}$ [%]")
    ax3.set_title("K3: Relative Correction Ratio $r_{Fy} = |\\Delta F_y| / |F_y|$ [%]", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax3.set_ylabel("Normal Load $F_z$ [N]", fontsize=10)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig11_pinn_correction_maps_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_pinn_improvement_and_gating(evaluator, sliced_metrics, compound_name, out_dir):
    """
    K4, K5, K6: Delta RMSE improvement bar chart and physical consistency verification (gating).
    Fixes twinx() bug by utilizing 4 dedicated subplots with correct independent physical units.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.5), dpi=150)

    # Panel 1 (Top Left): Delta RMSE = RMSE_hyb - RMSE_opt across domain slices
    ax1 = axes[0, 0]
    slice_names = []
    delta_rmse_vals = []
    slice_labels = []

    for s_name, s_data in sliced_metrics.items():
        rmse_opt = s_data.get("model2_optimized_mf62", {}).get("fy", {}).get("rmse")
        rmse_hyb = s_data.get("model3_hybrid_pinn", {}).get("fy", {}).get("rmse")
        if rmse_opt is not None and rmse_hyb is not None:
            delta = rmse_hyb - rmse_opt
            slice_names.append(s_name)
            delta_rmse_vals.append(delta)
            lbl = s_name.replace("L1_Load_", "L1 ").replace("L2_Load_", "L2 ").replace("L3_Load_", "L3 ").replace("L4_Load_", "L4 ").replace("L5_Load_", "L5 ")
            lbl = lbl.replace("Slip_", "").replace("Pressure_", "").replace("Camber_", "γ=")
            slice_labels.append(lbl)

    x_pos = np.arange(len(delta_rmse_vals))
    cols = ["#27ae60" if v < 0 else "#e74c3c" for v in delta_rmse_vals]
    bars = ax1.bar(x_pos, delta_rmse_vals, color=cols, alpha=0.85, edgecolor="#7f8c8d")
    ax1.axhline(0.0, color="gray", lw=0.8, linestyle="-")

    for b, v in zip(bars, delta_rmse_vals):
        y_txt = v - 0.5 if v < 0 else v + 0.2
        ax1.text(b.get_x() + b.get_width() / 2.0, y_txt, f"{v:+.1f}", ha="center",
                 va="top" if v < 0 else "bottom", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(slice_labels, rotation=45, ha="right", fontsize=8.0)
    ax1.set_title(f"K4: Model Improvement $\\Delta \\text{{RMSE}} = \\text{{RMSE}}_{{\\text{{hyb}}}} - \\text{{RMSE}}_{{\\text{{opt}}}}$ ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("$\\Delta$ RMSE [N] (Negative = Hybrid Better)", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2 (Top Right): Domain Falloff Envelope (alpha up to 20 deg)
    ax2 = axes[0, 1]
    alpha_wide_deg = np.linspace(0.0, 20.0, 201)
    alpha_wide_rad = np.radians(alpha_wide_deg)
    dfy_wide, dmz_wide = evaluator.eval_pinn(
        alpha_wide_rad,
        np.full_like(alpha_wide_rad, 887.0),
        np.zeros_like(alpha_wide_rad),
        np.full_like(alpha_wide_rad, 83.0)
    )
    ax2.plot(alpha_wide_deg, dfy_wide, "g-", lw=2.2, label="Hybrid Correction $\\Delta F_y(\\alpha)$")
    ax2.axvline(12.6, color="#e67e22", linestyle="--", lw=1.5, label="Calibrated Sweep Limit (12.6°)")
    ax2.axvspan(12.6, 20.0, color="#fef9e7", alpha=0.5, label="Falloff Envelope Active")
    ax2.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title("K6: Extreme Slip Domain Falloff Envelope (Smooth Extrapolation Gate)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax2.set_ylabel("Correction $\\Delta F_y$ [N]", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3 (Bottom Left): Zero-Slip Passivity Verification
    ax3 = axes[1, 0]
    alpha_small_deg = np.linspace(-3.0, 3.0, 201)
    alpha_small_rad = np.radians(alpha_small_deg)
    dfy_zero_slip, _ = evaluator.eval_pinn(
        alpha_small_rad,
        np.full_like(alpha_small_rad, 887.0),
        np.zeros_like(alpha_small_rad),
        np.full_like(alpha_small_rad, 83.0)
    )
    ax3.plot(alpha_small_deg, dfy_zero_slip, "b-", lw=2.2, label="Passivity Gate $\\Delta F_y(\\alpha)$")
    ax3.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.scatter([0.0], [0.0], color="red", s=50, zorder=5, label="Exact Zero-Slip Origin (0 N)")
    ax3.set_title("K6: Zero-Slip Passivity Gate $(\\Delta F_y \\to 0$ as $\\alpha \\to 0, \\gamma = 0)$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax3.set_ylabel("Correction $\\Delta F_y$ [N]", fontsize=10)
    ax3.set_xlim(-3.0, 3.0)
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    # Panel 4 (Bottom Right): Airborne Load Suppression Gate
    ax4 = axes[1, 1]
    fz_airborne = np.linspace(0.0, 400.0, 201)
    dfy_airborne, _ = evaluator.eval_pinn(
        np.full_like(fz_airborne, np.radians(6.0)),
        fz_airborne,
        np.zeros_like(fz_airborne),
        np.full_like(fz_airborne, 83.0)
    )
    ax4.plot(fz_airborne, dfy_airborne, color="#9b59b6", lw=2.2, label="Airborne Gate $\\Delta F_y(F_z)$ at $\\alpha=6^\\circ$")
    ax4.axvline(100.0, color="#e74c3c", linestyle="--", lw=1.5, label="Full Suppression Threshold ($F_z \\leq 100$ N)")
    ax4.axvspan(0.0, 100.0, color="#fdedec", alpha=0.5, label="Suppression Zone")
    ax4.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax4.scatter([0.0], [0.0], color="red", s=50, zorder=5, label="Airborne Zero (0 N at $F_z=0$)")
    ax4.set_title("K6: Airborne Load Suppression Gate $(\\Delta F_y \\to 0$ as $F_z \\to 0)$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax4.set_ylabel("Correction $\\Delta F_y$ [N]", fontsize=10)
    ax4.set_xlim(0.0, 400.0)
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig11_pinn_improvement_and_gating_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
