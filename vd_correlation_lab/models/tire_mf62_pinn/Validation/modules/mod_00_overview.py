#!/usr/bin/env python3
"""
Module 00: Overview & Executive Comparison Dashboard
Generates global executive comparison figures and domain-sliced performance summaries.
Organized into 4 clean panels:
  1. Global Fy & Mz Validation Metrics with Percentage Improvement
  2. 5 Physical Load Slices (L1 to L5)
  3. 3 Camber Angle Slices (0°, 2°, 4°)
  4. 3 Slip Regimes (Linear, Transitional, Saturation) + Executive Metrics Box
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB


def generate_overview_dashboard(eval_val, metrics_3way, sliced_metrics, compound_name, out_dir, long_metrics=None):
    """
    Creates an executive dashboard summarizing Lateral (Fy), Longitudinal (Fx),
    and Combined-Slip (Gyk * Fy0_sw) validation performance across M1, M2, and M3.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    if long_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.0), dpi=150)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 11.0), dpi=150)

    # Panel 1: Global Validation RMSE & Percentage Improvement
    ax1 = axes[0, 0]
    models = ["M1: Baseline", "M2: Optimized", "M3: Hybrid"]
    colors = [COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB]

    rmse_fy = [
        metrics_3way["validation"]["model1_baseline_mf62"]["fy"]["rmse"],
        metrics_3way["validation"]["model2_optimized_mf62"]["fy"]["rmse"],
        metrics_3way["validation"]["model3_hybrid_pinn"]["fy"]["rmse"]
    ]
    bars1 = ax1.bar(models, rmse_fy, color=colors, width=0.55, edgecolor="#7f8c8d", alpha=0.9)
    pct_gain_m2 = (rmse_fy[0] - rmse_fy[1]) / rmse_fy[0] * 100.0
    pct_gain_m3 = (rmse_fy[0] - rmse_fy[2]) / rmse_fy[0] * 100.0

    for i, (b, v) in enumerate(zip(bars1, rmse_fy)):
        if i == 0:
            lbl = f"{v:.1f} N\n(Baseline)"
        elif i == 1:
            lbl = f"{v:.1f} N\n(-{pct_gain_m2:.1f}%)"
        else:
            lbl = f"{v:.1f} N\n(-{pct_gain_m3:.1f}%)"
        ax1.text(b.get_x() + b.get_width() / 2.0, v + 2.0, lbl,
                 ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    ax1.set_title(f"Global Lateral Force $F_y$ Validation RMSE ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE [N]", fontsize=10)
    ax1.set_ylim(0, max(rmse_fy) * 1.25)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: Normal Load Slices (L1 to L5)
    ax2 = axes[0, 1]
    load_keys = ["L1_Load_220N", "L2_Load_440N", "L3_Load_665N", "L4_Load_885N", "L5_Load_1110N"]
    load_labels = ["L1 (~220N)", "L2 (~440N)", "L3 (~665N)", "L4 (~885N)", "L5 (~1110N)"]
    x = np.arange(len(load_keys))
    w = 0.25

    m1_loads = [sliced_metrics.get(k, {}).get("model1_baseline_mf62", {}).get("fy", {}).get("rmse", 0) for k in load_keys]
    m2_loads = [sliced_metrics.get(k, {}).get("model2_optimized_mf62", {}).get("fy", {}).get("rmse", 0) for k in load_keys]
    m3_loads = [sliced_metrics.get(k, {}).get("model3_hybrid_pinn", {}).get("fy", {}).get("rmse", 0) for k in load_keys]

    ax2.bar(x - w, m1_loads, w, label="M1: Baseline", color=COLOR_M1_BASE, alpha=0.85)
    ax2.bar(x, m2_loads, w, label="M2: Optimized", color=COLOR_M2_OPT, alpha=0.85)
    ax2.bar(x + w, m3_loads, w, label="M3: Hybrid", color=COLOR_M3_HYB, alpha=0.90)

    ax2.set_xticks(x)
    ax2.set_xticklabels(load_labels, fontsize=9.0)
    ax2.set_title("Lateral Force RMSE across 5 Physical Normal Loads", fontsize=11, fontweight="bold")
    ax2.set_ylabel("RMSE [N]", fontsize=10)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Camber Angle Slices (0°, 2°, 4°)
    ax3 = axes[1, 0]
    camber_keys = ["Camber_0deg", "Camber_2deg", "Camber_4deg"]
    camber_labels = ["0° Camber", "2° Camber", "4° Camber"]
    x_c = np.arange(len(camber_keys))

    m1_c = [sliced_metrics.get(k, {}).get("model1_baseline_mf62", {}).get("fy", {}).get("rmse", 0) for k in camber_keys]
    m2_c = [sliced_metrics.get(k, {}).get("model2_optimized_mf62", {}).get("fy", {}).get("rmse", 0) for k in camber_keys]
    m3_c = [sliced_metrics.get(k, {}).get("model3_hybrid_pinn", {}).get("fy", {}).get("rmse", 0) for k in camber_keys]

    b3_1 = ax3.bar(x_c - w, m1_c, w, label="M1: Baseline", color=COLOR_M1_BASE, alpha=0.85)
    b3_2 = ax3.bar(x_c, m2_c, w, label="M2: Optimized", color=COLOR_M2_OPT, alpha=0.85)
    b3_3 = ax3.bar(x_c + w, m3_c, w, label="M3: Hybrid", color=COLOR_M3_HYB, alpha=0.90)

    for i in range(len(camber_keys)):
        c_gain = (m1_c[i] - m2_c[i]) / m1_c[i] * 100.0 if m1_c[i] > 0 else 0
        ax3.text(x_c[i], m2_c[i] + 2.0, f"{-c_gain:+.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=COLOR_M2_OPT)

    ax3.set_xticks(x_c)
    ax3.set_xticklabels(camber_labels, fontsize=9.5)
    ax3.set_title("Camber-Dependent Lateral Force RMSE (0°, 2°, 4°)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("RMSE [N]", fontsize=10)
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.grid(True, axis="y", alpha=0.3)

    # Panel 4: Slip Regime Slices (Linear, Transitional, Saturation)
    ax4 = axes[1, 1]
    slip_keys = ["Slip_Linear_le_3deg", "Slip_Transitional_3_8deg", "Slip_Saturation_gt_8deg"]
    slip_labels = ["Linear (|α|≤3°)", "Transitional (3°<|α|≤8°)", "Saturation (|α|>8°)"]
    x_s = np.arange(len(slip_keys))

    m1_s = [sliced_metrics.get(k, {}).get("model1_baseline_mf62", {}).get("fy", {}).get("rmse", 0) for k in slip_keys]
    m2_s = [sliced_metrics.get(k, {}).get("model2_optimized_mf62", {}).get("fy", {}).get("rmse", 0) for k in slip_keys]
    m3_s = [sliced_metrics.get(k, {}).get("model3_hybrid_pinn", {}).get("fy", {}).get("rmse", 0) for k in slip_keys]

    ax4.bar(x_s - w, m1_s, w, label="M1: Baseline", color=COLOR_M1_BASE, alpha=0.85)
    ax4.bar(x_s, m2_s, w, label="M2: Optimized", color=COLOR_M2_OPT, alpha=0.85)
    ax4.bar(x_s + w, m3_s, w, label="M3: Hybrid", color=COLOR_M3_HYB, alpha=0.90)

    ax4.set_xticks(x_s)
    ax4.set_xticklabels(slip_labels, fontsize=8.5)
    ax4.set_title("Slip-Regime Lateral Force RMSE", fontsize=11, fontweight="bold")
    ax4.set_ylabel("RMSE [N]", fontsize=10)
    ax4.legend(fontsize=8.5, loc="upper left")
    ax4.grid(True, axis="y", alpha=0.3)

    if long_metrics:
        src_note = "TTC R9 R20 D/B" if c_upper == "R20" else "TTC R9 R20 Surrogate"
        # Panel 5 (axes[0, 2]): Longitudinal & Combined Global RMSE (M1 Baseline vs M2 Unified)
        ax5 = axes[0, 2]
        cats = ["Pure Val Fx", "Comb Val Fx", "Comb Val Fy\n(Gyk*Fy0_sw)"]
        m1_v = [
            long_metrics["pure_val_m1"]["rmse"],
            long_metrics["comb_val_fx_m1"]["rmse"],
            long_metrics["comb_val_fy_gyk_m1"]["rmse"],
        ]
        m2_v = [
            long_metrics["pure_val_m2"]["rmse"],
            long_metrics["comb_val_fx_m2"]["rmse"],
            long_metrics["comb_val_fy_gyk_m2"]["rmse"],
        ]
        xc = np.arange(len(cats))
        wc = 0.34
        ax5.bar(xc - wc / 2, m1_v, wc, label="M1: Baseline (Lit)", color=COLOR_M1_BASE, alpha=0.85)
        b_m2 = ax5.bar(xc + wc / 2, m2_v, wc, label="M2: Unified Calibrated", color=COLOR_M2_OPT, alpha=0.90)
        for bi, (v1, v2) in enumerate(zip(m1_v, m2_v)):
            g = (v1 - v2) / max(v1, 1e-4) * 100.0
            ax5.text(xc[bi] + wc / 2, v2 + 3.0, f"{v2:.1f} N\n({-g:+.1f}%)", ha="center", va="bottom",
                     fontsize=8.0, fontweight="bold", color=COLOR_M2_OPT)
        ax5.set_xticks(xc)
        ax5.set_xticklabels(cats, fontsize=8.5)
        ax5.set_title(f"Longitudinal & Combined Val RMSE ({src_note})", fontsize=10.5, fontweight="bold")
        ax5.set_ylabel("RMSE [N]", fontsize=10)
        ax5.set_ylim(0, max(m1_v + m2_v) * 1.28)
        ax5.legend(fontsize=8.0, loc="upper right")
        ax5.grid(True, axis="y", alpha=0.3)

        # Panel 6 (axes[1, 2]): Longitudinal Val Fx RMSE by Load & Unseen Speeds (M1 vs M2)
        ax6 = axes[1, 2]
        slice_keys = ["~220 N", "~660 N", "~880 N", "~1100 N", "24 kph", "72 kph"]
        m1_s_l = [long_metrics["sliced_pure_val"].get(k, {}).get("m1", {}).get("rmse", 0.0) for k in slice_keys]
        m2_s_l = [long_metrics["sliced_pure_val"].get(k, {}).get("m2", {}).get("rmse", 0.0) for k in slice_keys]
        xs = np.arange(len(slice_keys))
        ax6.bar(xs - wc / 2, m1_s_l, wc, label="M1: Baseline (Lit)", color=COLOR_M1_BASE, alpha=0.85)
        ax6.bar(xs + wc / 2, m2_s_l, wc, label="M2: Unified Calibrated", color=COLOR_M2_OPT, alpha=0.90)
        ax6.set_xticks(xs)
        ax6.set_xticklabels(slice_keys, fontsize=8.5)
        ax6.set_title(f"Pure Fx Val RMSE by Load & Out-of-Sample Speed ({c_upper})", fontsize=10.5, fontweight="bold")
        ax6.set_ylabel("RMSE [N]", fontsize=10)
        ax6.legend(fontsize=8.0, loc="upper left")
        ax6.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig00_overview_dashboard_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
