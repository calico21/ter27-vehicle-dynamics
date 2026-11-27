#!/usr/bin/env python3
"""
Module 06: Combined Slip & Experimental Friction Envelope Validation
Validates Model 1 (Literature Baseline), Model 2 (Unified Calibrated MF 6.2), and Model 3 (Hybrid PINN)
against real Calspan TTC Round 9 Drive/Brake combined-slip data (SA in {-3, -6} deg):
  - F1: Combined-slip Fx(kappa, alpha) vs experimental points at ~885 N, 83 kPa, 0 deg IA
  - F2: Lateral force reduction G_yk = Fy(alpha, kappa) / Fy0_sweep(alpha) vs experiment
  - F3: Longitudinal force reduction G_xa = Fx(kappa, alpha) / Fx0(kappa) vs experiment
  - F4: Unified Friction Envelope (Fx vs Fy Polar across alpha in [0, -12] deg + 18x6 Ref Overlay)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import (
    apply_theme, add_metrics_box,
    COLOR_M1_BASE, COLOR_M2_OPT, COLOR_M3_HYB,
    STYLE_M1, STYLE_M2, STYLE_M3,
)
from core.metrics import compute_metrics
from calibrate_mf62 import Pacejka62Fitter, DRIVE_BRAKE_PROCESSED_DIR


def _load_comb_and_pure():
    df_c = pd.read_csv(os.path.join(DRIVE_BRAKE_PROCESSED_DIR, "R20_18x6_combined_slip.csv"), low_memory=False)
    df_p = pd.read_csv(os.path.join(DRIVE_BRAKE_PROCESSED_DIR, "R20_18x6_pure_longitudinal.csv"), low_memory=False)
    return df_c[df_c["split"] == "train"].copy(), df_c[df_c["split"] == "val"].copy(), df_p[df_p["split"] == "val"].copy()


def _binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    xc, ym, ys = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if np.sum(m) >= 5:
            xc.append(float(np.mean(x[m])))
            ym.append(float(np.mean(y[m])))
            ys.append(float(np.std(y[m])))
    return np.array(xc), np.array(ym), np.array(ys)


def generate_combined_slip_suite(evaluator, compound_name, out_dir, df_comb_train=None, df_comb_val=None, df_pure_val=None):
    """
    F1-F4 (2x2): Real TTC Round 9 combined-slip validation & unified friction envelope.
    """
    apply_theme()
    if df_comb_val is None or df_pure_val is None:
        df_comb_train, df_comb_val, df_pure_val = _load_comb_and_pure()

    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    src_tag = "R20 18x6 D/B" if c_upper == "R20" else "R20 18x6 Surrogate"
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), dpi=150)

    k_dense = np.linspace(-0.22, 0.20, 220)
    k_edges = np.linspace(-0.22, 0.20, 21)
    fz_nom = 885.0

    # Evaluate combined validation dataframe once
    res_cv = evaluator.eval_combined(
        df_comb_val["kappa"].values, df_comb_val["alpha"].values,
        df_comb_val["f_z"].values, df_comb_val["gamma"].values, df_comb_val["p_kpa"].values,
    )
    refs_v = {
        sw: float((s.loc[s["kappa"].abs() < 0.015, "f_y"] / s.loc[s["kappa"].abs() < 0.015, "f_z"]).mean())
        for sw, s in df_comb_val.groupby("global_sweep_id") if (s["kappa"].abs() < 0.015).sum() >= 3
    }
    fy0_sw_val = df_comb_val["global_sweep_id"].map(refs_v).values * df_comb_val["f_z"].values

    # Panel 1: Combined-Slip Fx(kappa, alpha) vs Experiment (~885 N, 83 kPa, 0 deg IA)
    ax1 = axes[0, 0]
    for sa_t, col, mk, src in [(0.0, "#1f77b4", "o", df_pure_val), (-3.0, "#e67e22", "s", df_comb_val), (-6.0, "#d62728", "^", df_comb_val)]:
        s = src[(src["sa_deg"].sub(sa_t).abs() < 0.6) & (src["f_z"].sub(fz_nom).abs() <= 80.0) &
                (src["p_kpa"].sub(83.0).abs() < 6.0) & (src["ia_deg"].abs() < 0.3)]
        if len(s) >= 10:
            xc, ym, ys = _binned(s["kappa"].values, s["f_x"].values, k_edges)
            ax1.errorbar(xc * 100.0, ym, yerr=ys, fmt=mk, ms=4, capsize=2, color=col, alpha=0.85,
                         label=f"Exp $\\alpha={sa_t:.0f}^\\circ$")
        ev_c = evaluator.eval_combined(k_dense, np.full_like(k_dense, np.radians(sa_t)), fz_nom, 0.0, 83.0)
        ax1.plot(k_dense * 100.0, ev_c["fx_m1"], color=col, lw=1.2, ls=":", alpha=0.8)
        ax1.plot(k_dense * 100.0, ev_c["fx_m2"], color=col, lw=2.1, label=f"M2 $\\alpha={sa_t:.0f}^\\circ$")
    add_metrics_box(ax1, {
        "Baseline": compute_metrics(df_comb_val["f_x"].values, res_cv["fx_m1"]),
        "Unified": compute_metrics(df_comb_val["f_x"].values, res_cv["fx_m2"]),
    }, loc="upper left", title=f"Comb Fx Val ({src_tag})")
    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"F1: Combined-Slip $F_x(\\kappa, \\alpha)$ (~885 N, 83 kPa) — {c_upper}", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
    ax1.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=9.5)
    ax1.legend(ncol=2, loc="lower right", fontsize=7.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Lateral Force Reduction Factor Gyk = Fy(alpha, kappa) / Fy0_sweep(alpha)
    ax2 = axes[0, 1]
    gyk_emp = df_comb_val["f_y"].values / np.where(np.abs(fy0_sw_val) > 150.0, fy0_sw_val, np.nan)
    m_valid_gyk = np.isfinite(gyk_emp) & (gyk_emp >= -0.2) & (gyk_emp <= 1.4)
    for sa_t, col, mk in [(-3.0, "#e67e22", "s"), (-6.0, "#d62728", "^")]:
        ms = m_valid_gyk & (df_comb_val["sa_deg"].sub(sa_t).abs().values < 0.6)
        xc, ym, ys = _binned(df_comb_val["kappa"].values[ms], gyk_emp[ms], k_edges)
        ax2.errorbar(xc * 100.0, ym, yerr=ys, fmt=mk, ms=4, capsize=2, color=col, alpha=0.85,
                     label=f"Exp $F_y/F_{{y0,\\mathrm{{sw}}}}$ ($\\alpha={sa_t:.0f}^\\circ$)")
        ev_c = evaluator.eval_combined(k_dense, np.full_like(k_dense, np.radians(sa_t)), fz_nom, 0.0, 83.0)
        ax2.plot(k_dense * 100.0, ev_c["gyk_m1"], color=col, lw=1.3, ls=":", alpha=0.85, label=f"M1 Lit ($\\alpha={sa_t:.0f}^\\circ$)")
        ax2.plot(k_dense * 100.0, ev_c["gyk_m2"], color=col, lw=2.2, label=f"M2 Cal ($\\alpha={sa_t:.0f}^\\circ$)")
    add_metrics_box(ax2, {
        "Baseline": compute_metrics(df_comb_val["f_y"].values, res_cv["gyk_m1"] * fy0_sw_val),
        "Unified": compute_metrics(df_comb_val["f_y"].values, res_cv["gyk_m2"] * fy0_sw_val),
    }, loc="upper left", title="Comb Fy (Gyk * Fy0_sw)")
    ax2.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax2.set_title(f"F2: Lateral Attenuation $G_{{y\\kappa}}$ (Cal RBY/RHY1 vs Lit) — {c_upper}", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
    ax2.set_ylabel("Normalized Lateral Grip $G_{y\\kappa}$ [-]", fontsize=9.5)
    ax2.set_ylim(-0.08, 1.18)
    ax2.legend(ncol=2, loc="lower center", fontsize=7.2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Longitudinal Force Reduction Factor Gxa = Fx(kappa, alpha) / Fx0(kappa)
    ax3 = axes[1, 0]
    fx0_val = evaluator.eval_pure_longitudinal(
        df_comb_val["kappa"].values, df_comb_val["f_z"].values, df_comb_val["gamma"].values, df_comb_val["p_kpa"].values
    )["fx_m2"]
    gxa_emp = df_comb_val["f_x"].values / np.where(np.abs(fx0_val) > 250.0, fx0_val, np.nan)
    m_valid_gxa = np.isfinite(gxa_emp) & (gxa_emp >= 0.0) & (gxa_emp <= 1.45)
    for sa_t, col, mk in [(-3.0, "#e67e22", "s"), (-6.0, "#d62728", "^")]:
        ms = m_valid_gxa & (df_comb_val["sa_deg"].sub(sa_t).abs().values < 0.6)
        xc, ym, ys = _binned(df_comb_val["kappa"].values[ms], gxa_emp[ms], k_edges)
        ax3.errorbar(xc * 100.0, ym, yerr=ys, fmt=mk, ms=4, capsize=2, color=col, alpha=0.85,
                     label=f"Exp $F_x/F_{{x0}}$ ($\\alpha={sa_t:.0f}^\\circ$)")
        ev_c = evaluator.eval_combined(k_dense, np.full_like(k_dense, np.radians(sa_t)), fz_nom, 0.0, 83.0)
        ax3.plot(k_dense * 100.0, ev_c["gxa_m1"], color=col, lw=1.3, ls=":", alpha=0.85, label=f"M1 Lit ($\\alpha={sa_t:.0f}^\\circ$)")
        ax3.plot(k_dense * 100.0, ev_c["gxa_m2"], color=col, lw=2.2, label=f"M2 Cal ($\\alpha={sa_t:.0f}^\\circ$)")
    ax3.axhline(1.0, color="gray", lw=0.7, linestyle=":")
    ax3.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.set_title(f"F3: Longitudinal Attenuation $G_{{x\\alpha}}$ (SA $\\in \\{{-3^\\circ, -6^\\circ\\}}$) — {c_upper}", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
    ax3.set_ylabel("Normalized Longitudinal Grip $G_{x\\alpha}$ [-]", fontsize=9.5)
    ax3.set_ylim(0.05, 1.22)
    ax3.legend(ncol=2, loc="lower center", fontsize=7.2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Unified Friction Envelope (Fx vs Fy Polar across alpha in [0, -12] deg + 18x6 Ref Data)
    ax4 = axes[1, 1]
    for sa_t, col in [(0.0, "#1f77b4"), (-3.0, "#e67e22"), (-6.0, "#d62728"), (-9.0, "#9467bd"), (-12.0, "#27ae60")]:
        ev_c = evaluator.eval_combined(k_dense, np.full_like(k_dense, np.radians(sa_t)), fz_nom, 0.0, 83.0)
        ax4.plot(ev_c["fy_m2"], ev_c["fx_m2"], "--", color=col, lw=1.6, alpha=0.85)
        ax4.plot(ev_c["fy_m3"], ev_c["fx_m3"], "-", color=col, lw=2.2, label=f"M3 Unified $\\alpha={sa_t:.0f}^\\circ$")
    for sa_t, col in [(0.0, "#1f77b4"), (-3.0, "#e67e22"), (-6.0, "#d62728")]:
        src = df_pure_val if sa_t == 0.0 else df_comb_val
        s = src[(src["sa_deg"].sub(sa_t).abs() < 0.6) & (src["f_z"].sub(fz_nom).abs() <= 80.0) &
                (src["p_kpa"].sub(83.0).abs() < 6.0) & (src["ia_deg"].abs() < 0.3)]
        if len(s) > 0:
            ax4.scatter(s["f_y"], s["f_x"], s=5, alpha=0.22, color=col, rasterized=True,
                        label=f"TTC 18x6 Exp ({sa_t:.0f}°)" if sa_t != 0 else None)
    ax4.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax4.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax4.set_title(f"F4: Unified M2/M3 Friction Envelope vs 18x6 Ref ($F_z={fz_nom:.0f}$ N) — {c_upper}", fontsize=10.5, fontweight="bold")
    ax4.set_xlabel("Lateral Force $F_y$ [N]", fontsize=9.5)
    ax4.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=9.5)
    ax4.text(0.03, 0.03,
             "Unified 16x7.5-10 M3 (solid) & M2 (dashed)\n"
             "combines Cornering Fy0+PINN with Round 9 Gxa/Gyk.\n"
             "Scatter = raw 18x6.0-10 R20 (~30% narrower Fy0).",
             transform=ax4.transAxes, fontsize=7.8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7", alpha=0.9))
    ax4.legend(ncol=2, loc="upper left", fontsize=7.2)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Module 06: Combined-Slip Validation & Friction Envelope ({c_upper} — {src_tag})",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(out_dir, f"fig06_combined_slip_friction_ellipse_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
