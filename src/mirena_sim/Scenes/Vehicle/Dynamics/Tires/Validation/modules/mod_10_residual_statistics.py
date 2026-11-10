#!/usr/bin/env python3
"""
Module 10: Statistical Residual Analysis
Generates:
  J1: 2D Hexbin Density Parity plots for Lateral Force Fy and Aligning Torque Mz
  J2: Residual distributions (Histogram, KDE, 95th and 99th percentile markers)
  J3: Residual trends vs operating condition (alpha, Fz, gamma) with IQR bands
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core.style import apply_theme, COLOR_EXP, COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB
from core.metrics import compute_metrics, decompose_error


def generate_parity_and_qq_suite(eval_val, compound_name, out_dir):
    """
    J1 & J2: 4-panel figure with 2D Hexbin Parity for Fy and Mz, and Residual Histograms with Percentiles.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.0), dpi=150)

    fy_exp = eval_val["fy_exp"]
    fy_m2 = eval_val["fy_m2"]
    fy_m3 = eval_val["fy_m3"]

    mz_exp = eval_val["mz_exp"]
    mz_m2 = eval_val["mz_m2"]
    mz_m3 = eval_val["mz_m3"]

    m2_m = compute_metrics(fy_exp, fy_m2)
    m3_m = compute_metrics(fy_exp, fy_m3)
    mz2_m = compute_metrics(mz_exp, mz_m2)
    mz3_m = compute_metrics(mz_exp, mz_m3)

    # Panel 1: Lateral Force Parity Scatter (Model 2 Optimized)
    ax1 = axes[0, 0]
    lims_fy = [-2800, 2800]
    ax1.scatter(
        fy_exp, fy_m2, s=2.5, alpha=0.18, color=COLOR_M2_OPT,
        label=f"M2: Opt (RMSE={m2_m['rmse']:.1f}N, R²={m2_m['r2']:.3f})"
    )

    ax1.plot(lims_fy, lims_fy, "r--", lw=1.8, label="Perfect Parity (y = x)")
    # Regression line
    slope, intercept, r_val, _, _ = stats.linregress(fy_exp, fy_m2)
    x_line = np.array(lims_fy)
    ax1.plot(x_line, slope * x_line + intercept, "k-", lw=1.5,
             label=f"Fit: y = {slope:.3f}x {intercept:+.1f} (R²={r_val**2:.3f})")

    ax1.set_title(f"J1: Lateral Force Parity $F_{{y,\\text{{pred}}}}$ vs $F_{{y,\\text{{exp}}}}$ ({c_upper} N={len(fy_exp):,})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Experimental Lateral Force $F_{y,\\text{exp}}$ [N]", fontsize=10)
    ax1.set_ylabel("Predicted Lateral Force $F_{y,\\text{opt}}$ [N]", fontsize=10)
    ax1.set_xlim(lims_fy)
    ax1.set_ylim(lims_fy)
    ax1.legend(loc="upper left", fontsize=8.0)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Self-Aligning Moment Parity Scatter (Model 2)
    ax2 = axes[0, 1]
    lims_mz = [-95, 95]
    ax2.scatter(
        mz_exp, mz_m2, s=2.5, alpha=0.18, color="#8e44ad",
        label=f"M2: Opt (RMSE={mz2_m['rmse']:.2f}Nm, R²={mz2_m['r2']:.3f})"
    )

    ax2.plot(lims_mz, lims_mz, "r--", lw=1.8, label="Perfect Parity (y = x)")
    slope_m, intercept_m, r_val_m, _, _ = stats.linregress(mz_exp, mz_m2)
    x_line_m = np.array(lims_mz)
    ax2.plot(x_line_m, slope_m * x_line_m + intercept_m, "k-", lw=1.5,
             label=f"Fit: y = {slope_m:.3f}x {intercept_m:+.2f} (R²={r_val_m**2:.3f})")

    ax2.set_title(f"J1: Aligning Torque Parity $M_{{z,\\text{{pred}}}}$ vs $M_{{z,\\text{{exp}}}}$ ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Experimental Aligning Torque $M_{z,\\text{exp}}$ [Nm]", fontsize=10)
    ax2.set_ylabel("Predicted Aligning Torque $M_{z,\\text{opt}}$ [Nm]", fontsize=10)
    ax2.set_xlim(lims_mz)
    ax2.set_ylim(lims_mz)
    ax2.legend(loc="upper left", fontsize=8.0)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Lateral Force Residual Error Distribution with Percentile Markers
    ax3 = axes[1, 0]
    err_fy_m2 = fy_m2 - fy_exp
    err_fy_m3 = fy_m3 - fy_exp

    p95_fy = np.percentile(np.abs(err_fy_m2), 95)
    p99_fy = np.percentile(np.abs(err_fy_m2), 99)

    ax3.hist(err_fy_m2, bins=70, density=True, alpha=0.55, color=COLOR_M2_OPT,
             label=f"M2: Opt (RMSE={m2_m['rmse']:.1f}N, Bias={m2_m['bias']:+.1f}N)")
    ax3.hist(err_fy_m3, bins=70, density=True, alpha=0.50, color=COLOR_M3_HYB,
             label=f"M3: Hybrid (RMSE={m3_m['rmse']:.1f}N, Bias={m3_m['bias']:+.1f}N)")

    ax3.axvline(-p95_fy, color="#e67e22", linestyle="--", lw=1.4, label=f"95% Bound (±{p95_fy:.1f} N)")
    ax3.axvline(p95_fy, color="#e67e22", linestyle="--", lw=1.4)
    ax3.axvline(-p99_fy, color="#e74c3c", linestyle=":", lw=1.4, label=f"99% Bound (±{p99_fy:.1f} N)")
    ax3.axvline(p99_fy, color="#e74c3c", linestyle=":", lw=1.4)

    ax3.set_title(f"J2: Lateral Force Residual Distribution $\\Delta F_y$ ({c_upper})", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Residual $\\Delta F_y = F_{y,\\text{pred}} - F_{y,\\text{exp}}$ [N]", fontsize=10)
    ax3.set_ylabel("Probability Density", fontsize=10)
    ax3.set_xlim(-450, 450)
    ax3.legend(loc="upper left", fontsize=8.0)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Aligning Torque Residual Error Distribution
    ax4 = axes[1, 1]
    err_mz_m2 = mz_m2 - mz_exp
    err_mz_m3 = mz_m3 - mz_exp

    p95_mz = np.percentile(np.abs(err_mz_m2), 95)
    p99_mz = np.percentile(np.abs(err_mz_m2), 99)

    ax4.hist(err_mz_m2, bins=70, density=True, alpha=0.55, color=COLOR_M2_OPT,
             label=f"M2: Opt (RMSE={mz2_m['rmse']:.2f}Nm, Bias={mz2_m['bias']:+.2f}Nm)")
    ax4.hist(err_mz_m3, bins=70, density=True, alpha=0.50, color=COLOR_M3_HYB,
             label=f"M3: Hybrid (RMSE={mz3_m['rmse']:.2f}Nm, Bias={mz3_m['bias']:+.2f}Nm)")

    ax4.axvline(-p95_mz, color="#e67e22", linestyle="--", lw=1.4, label=f"95% Bound (±{p95_mz:.2f} Nm)")
    ax4.axvline(p95_mz, color="#e67e22", linestyle="--", lw=1.4)
    ax4.axvline(-p99_mz, color="#e74c3c", linestyle=":", lw=1.4, label=f"99% Bound (±{p99_mz:.2f} Nm)")
    ax4.axvline(p99_mz, color="#e74c3c", linestyle=":", lw=1.4)

    ax4.set_title(f"J2: Aligning Torque Residual Distribution $\\Delta M_z$ ({c_upper})", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Residual $\\Delta M_z = M_{z,\\text{pred}} - M_{z,\\text{exp}}$ [Nm]", fontsize=10)
    ax4.set_ylabel("Probability Density", fontsize=10)
    ax4.set_xlim(-18, 18)
    ax4.legend(loc="upper left", fontsize=8.0)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig10_parity_and_distributions_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_spatial_residuals_and_conditions(eval_val, compound_name, out_dir):
    """
    J3 & J5: Residual error trends vs operating conditions (alpha, Fz, gamma) with IQR bands.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)

    alpha_deg = np.degrees(eval_val["alpha"])
    fz = eval_val["f_z"]
    gamma_deg = np.degrees(eval_val["gamma"])
    err_fy_m2 = eval_val["fy_m2"] - eval_val["fy_exp"]
    err_fy_m3 = eval_val["fy_m3"] - eval_val["fy_exp"]

    df_err = pd.DataFrame({
        "alpha_deg": alpha_deg,
        "fz": fz,
        "gamma_deg": gamma_deg,
        "err_m2": err_fy_m2,
        "err_m3": err_fy_m3
    })

    # Panel 1: Residual vs Slip Angle alpha with IQR
    ax1 = axes[0]
    bins_a = np.linspace(-13.0, 13.0, 27)
    df_err["a_bin"] = pd.cut(df_err["alpha_deg"], bins_a)
    grp_a = df_err.groupby("a_bin", observed=True)["err_m2"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).dropna()
    a_mids = [b.mid for b in grp_a.index]

    ax1.plot(a_mids, grp_a["median"], "b-o", lw=2.0, markersize=5, label="M2 Median Residual")
    ax1.fill_between(a_mids, grp_a["q25"], grp_a["q75"], color="b", alpha=0.20, label="M2 IQR [25%-75%]")
    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"J3: Residual $\\Delta F_y$ vs Slip Angle $\\alpha$ ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Lateral Residual $\\Delta F_y$ [N]", fontsize=10)
    ax1.set_ylim(-200, 200)
    ax1.legend(loc="lower center", fontsize=8.0)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residual vs Normal Load Fz
    ax2 = axes[1]
    bins_fz = np.linspace(150.0, 1250.0, 23)
    df_err["fz_bin"] = pd.cut(df_err["fz"], bins_fz)
    grp_fz = df_err.groupby("fz_bin", observed=True)["err_m2"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).dropna()
    fz_mids = [b.mid for b in grp_fz.index]

    ax2.plot(fz_mids, grp_fz["median"], "g-s", lw=2.0, markersize=5, label="M2 Median Residual")
    ax2.fill_between(fz_mids, grp_fz["q25"], grp_fz["q75"], color="g", alpha=0.20, label="M2 IQR")
    ax2.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title(f"J3: Residual $\\Delta F_y$ vs Normal Load $F_z$ ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax2.set_ylabel("Lateral Residual $\\Delta F_y$ [N]", fontsize=10)
    ax2.set_ylim(-200, 200)
    ax2.legend(loc="lower center", fontsize=8.0)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Error Decomposition (Bias^2 vs Variance) across Loads
    ax3 = axes[2]
    load_names = ["L1", "L2", "L3", "L4", "L5"]
    fz_bounds = [330.0, 550.0, 770.0, 990.0]
    dig_fz = np.digitize(fz, fz_bounds)

    bias_sq_list = []
    var_list = []
    for i in range(5):
        m = dig_fz == i
        if np.sum(m) > 10:
            decomp = decompose_error(eval_val["fy_exp"][m], eval_val["fy_m2"][m])
            bias_sq_list.append(decomp["bias_sq"])
            var_list.append(decomp["variance"])
        else:
            bias_sq_list.append(0.0)
            var_list.append(0.0)

    x_idx = np.arange(len(load_names))
    ax3.bar(x_idx, bias_sq_list, 0.45, label="Systematic $\\text{Bias}^2$", color="#e74c3c", alpha=0.85)
    ax3.bar(x_idx, var_list, 0.45, bottom=bias_sq_list, label="Stochastic Variance $\\sigma^2$", color="#2980b9", alpha=0.85)

    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(load_names, fontsize=9.5)
    ax3.set_title(f"J5: Error Decomposition: $\\text{{MSE}} = \\text{{Bias}}^2 + \\sigma^2$ ({c_upper})", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Load Level", fontsize=10)
    ax3.set_ylabel("Mean Squared Error [N²]", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig10_spatial_residuals_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
