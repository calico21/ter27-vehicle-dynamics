#!/usr/bin/env python3
"""
Module 05: Pure Longitudinal Behavior & Experimental Validation (Fx)
Validates Model 1 (Literature Baseline) vs Model 2 (Unified Calibrated MF 6.2)
against real Calspan TTC Round 9 Drive/Brake experimental data:
  - fig05_longitudinal_characterization_{compound}.png (2x3):
      * 4 Drive/Brake vertical load levels (~220, ~660, ~880, ~1100 N @ 83 kPa, 0 deg IA)
      * Initial longitudinal stiffness C_kappa(Fz) across pressures (55-97 kPa)
      * Peak longitudinal friction mu_x(Fz) & global validation metrics box
  - fig05_longitudinal_pressure_and_residuals_{compound}.png (1x3):
      * Pressure sensitivity Fx(kappa) at ~885 N (55, 69, 83, 97 kPa)
      * Experimental vs predicted Fx parity (Train @ 40 kph & Out-of-Sample Val @ 24/40/72 kph)
      * Residuals vs slip ratio kappa with binned mean +- 1 sigma error band

Note: Experimental Drive/Brake data are Calspan TTC Round 9 Hoosier 18.0x6.0-10 R20
(6-in & 7-in rims). For R25B, curves are compared against the R20 Drive/Brake surrogate
and prominently annotated as such (no R25B Drive/Brake TTC dataset exists).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.style import (
    apply_theme, add_metrics_box,
    COLOR_EXP, COLOR_M1_BASE, COLOR_M2_OPT,
    STYLE_M1, STYLE_M2,
)
from core.metrics import compute_metrics
from calibrate_mf62 import Pacejka62Fitter, DRIVE_BRAKE_PROCESSED_DIR


def _load_pure_splits():
    df_pure = pd.read_csv(os.path.join(DRIVE_BRAKE_PROCESSED_DIR, "R20_18x6_pure_longitudinal.csv"), low_memory=False)
    return df_pure[df_pure["split"] == "train"].copy(), df_pure[df_pure["split"] == "val"].copy()


def _binned_curve(x, y, edges):
    idx = np.digitize(x, edges) - 1
    xc, ym, ys = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if np.sum(m) >= 5:
            xc.append(float(np.mean(x[m])))
            ym.append(float(np.mean(y[m])))
            ys.append(float(np.std(y[m])))
    return np.array(xc), np.array(ym), np.array(ys)


def _sweep_features(df_all):
    rows = []
    for sw_id, sub in df_all.groupby("global_sweep_id"):
        k = sub["kappa"].values
        fx = sub["f_x"].values
        fz_m = float(sub["f_z"].median())
        p_m = float(sub["p_kpa"].median())
        ia_m = float(sub["ia_deg"].median())
        m_lin = np.abs(k) <= 0.025
        ck = float(np.polyfit(k[m_lin], fx[m_lin], 1)[0]) if np.sum(m_lin) >= 12 and np.std(k[m_lin]) > 0.005 else np.nan
        # Binned peaks to avoid single-sample load ripple bias
        pk_d, pk_b = np.nan, np.nan
        for mask, is_d in [(k > 0.03, True), (k < -0.03, False)]:
            if np.sum(mask) >= 20:
                ks, fxs = k[mask], fx[mask]
                ed = np.linspace(ks.min(), ks.max(), 9)
                bi = np.digitize(ks, ed[1:-1])
                b_means = [float(np.mean(fxs[bi == i])) for i in range(8) if np.sum(bi == i) >= 3]
                if len(b_means) >= 3:
                    if is_d:
                        pk_d = max(b_means)
                    else:
                        pk_b = min(b_means)
        rows.append({
            "global_sweep_id": sw_id, "split": str(sub["split"].iloc[0]),
            "fz_med": fz_m, "p_kpa_med": p_m, "ia_deg_med": ia_m,
            "c_kappa_kn": ck / 1000.0 if np.isfinite(ck) else np.nan,
            "mu_drv": pk_d / fz_m if np.isfinite(pk_d) else np.nan,
            "mu_brk": -pk_b / fz_m if np.isfinite(pk_b) else np.nan,
        })
    return pd.DataFrame(rows)


def generate_longitudinal_characterization(evaluator, compound_name, out_dir, df_pure_train=None, df_pure_val=None):
    """
    E1-E6 (2x3): Real TTC Round 9 Drive/Brake Fx(kappa) across 4 vertical loads,
    initial stiffness C_kappa(Fz), and peak friction mu_x(Fz).
    """
    apply_theme()
    if df_pure_train is None or df_pure_val is None:
        df_pure_train, df_pure_val = _load_pure_splits()

    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    src_tag = "R20 Drive/Brake (18x6)" if c_upper == "R20" else "R20 Drive/Brake Surrogate (No R25B D/B)"

    df_all = pd.concat([df_pure_train, df_pure_val], ignore_index=True)
    df_nom = df_all[(df_all["ia_deg"].abs() < 0.3) & ((df_all["p_kpa"] - 83.0).abs() < 6.0)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.2), dpi=150)
    k_dense = np.linspace(-0.24, 0.22, 301)
    k_edges = np.linspace(-0.24, 0.22, 24)
    load_specs = [
        (220.0, "L1 (~220 N)", "#1f77b4", axes[0, 0]),
        (660.0, "L2 (~660 N)", "#ff7f0e", axes[0, 1]),
        (880.0, "L3 (~880 N)", "#d62728", axes[0, 2]),
        (1100.0, "L4 (~1100 N)", "#9467bd", axes[1, 0]),
    ]

    for target_fz, lbl, col, ax in load_specs:
        sub = df_nom[(df_nom["f_z"] - target_fz).abs() <= 80.0]
        fz_mean = float(sub["f_z"].median()) if len(sub) > 0 else target_fz
        sub_tr = sub[sub["split"] == "train"]
        sub_va = sub[sub["split"] == "val"]

        if len(sub_tr) > 0:
            ax.scatter(sub_tr["kappa"] * 100.0, sub_tr["f_x"], s=3, alpha=0.14, color=COLOR_EXP,
                       label=f"TTC Train (N={len(sub_tr):,})", rasterized=True)
        if len(sub_va) > 0:
            ax.scatter(sub_va["kappa"] * 100.0, sub_va["f_x"], s=5, alpha=0.20, marker="^", color=col,
                       label=f"TTC Val (N={len(sub_va):,})", rasterized=True)
        if len(sub) > 0:
            xc, ym, ys = _binned_curve(sub["kappa"].values, sub["f_x"].values, k_edges)
            ax.errorbar(xc * 100.0, ym, yerr=ys, fmt="o", ms=3.5, color=col, ecolor=col,
                        elinewidth=1.1, capsize=2, alpha=0.85, label="Exp Mean ± 1σ")
            ev_s = evaluator.eval_longitudinal_dataset(sub)
            add_metrics_box(ax, {
                "Baseline": compute_metrics(ev_s["fx_exp"], ev_s["fx_m1"]),
                "Unified": compute_metrics(ev_s["fx_exp"], ev_s["fx_m2"]),
            }, loc="upper left", title=f"{lbl} (83 kPa, 0° IA)")

        pr = evaluator.eval_pure_longitudinal(k_dense, np.full_like(k_dense, fz_mean), gamma=0.0, p=83.0)
        ax.plot(k_dense * 100.0, pr["fx_m1"], **STYLE_M1, label="M1: Baseline (Lit)")
        ax.plot(k_dense * 100.0, pr["fx_m2"], **STYLE_M2, label="M2: Unified Calibrated")
        ax.axhline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.axvline(0.0, color="gray", lw=0.6, linestyle=":")
        ax.set_title(f"{lbl}: $F_z \\approx {fz_mean:.0f}$ N ({c_upper})", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
        ax.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=9.5)
        ax.set_xlim(-25.0, 23.0)
        ax.legend(loc="lower right", fontsize=7.2)
        ax.grid(True, alpha=0.3)

    # Compute sweep-level experimental stiffness & peaks across all valid pure sweeps
    sw_feats = _sweep_features(df_all)
    p_cols = {55: ("#d73027", "55 kPa"), 69: ("#fc8d59", "69 kPa"),
              83: ("#2980b9", "83 kPa"), 97: ("#27ae60", "97 kPa")}
    sw_feats["p_bin"] = sw_feats["p_kpa_med"].map(lambda x: min(p_cols.keys(), key=lambda pb: abs(pb - x)))
    fz_grid = np.linspace(180.0, 1250.0, 120)

    # Panel 5: Initial Longitudinal Slip Stiffness C_kappa vs Fz
    ax5 = axes[1, 1]
    pr_b = evaluator.eval_pure_longitudinal(0.0, fz_grid, 0.0, 83.0)
    ax5.plot(fz_grid, pr_b["kxk_m1"] / 1000.0, **STYLE_M1, label="M1: Baseline (83 kPa)")
    for pb, (col, p_lbl) in p_cols.items():
        s = sw_feats[sw_feats["p_bin"] == pb]
        s_tr, s_va = s[s["split"] == "train"], s[s["split"] == "val"]
        ax5.scatter(s_tr["fz_med"], s_tr["c_kappa_kn"], color=col, s=22, alpha=0.80, marker="o", label=f"Exp {p_lbl}")
        ax5.scatter(s_va["fz_med"], s_va["c_kappa_kn"], color=col, s=30, alpha=0.95, marker="^", edgecolors="k", linewidths=0.4)
        pr_p = evaluator.eval_pure_longitudinal(0.0, fz_grid, 0.0, float(pb))
        ax5.plot(fz_grid, pr_p["kxk_m2"] / 1000.0, color=col, lw=1.8)
    ax5.set_title(f"E5: Longitudinal Slip Stiffness $C_\\kappa(F_z)$ ({c_upper})", fontsize=10.5, fontweight="bold")
    ax5.set_xlabel("Normal Load $F_z$ [N]", fontsize=9.5)
    ax5.set_ylabel("Initial Slip Stiffness $C_\\kappa$ [kN/unit slip]", fontsize=9.5)
    ax5.legend(ncol=2, loc="upper left", fontsize=7.2)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Peak Longitudinal Friction mu_x vs Fz + Global Validation Metrics Box
    ax6 = axes[1, 2]
    for pb, (col, p_lbl) in p_cols.items():
        s = sw_feats[sw_feats["p_bin"] == pb]
        ax6.scatter(s["fz_med"], s["mu_drv"], color=col, s=22, alpha=0.8, marker="o", label=f"Exp {p_lbl} (Drive)")
        ax6.scatter(s["fz_med"], s["mu_brk"], color=col, s=24, alpha=0.75, marker="x")
        mu_m2 = [
            float(np.max(np.abs(evaluator.eval_pure_longitudinal(k_dense, fz_i, 0.0, float(pb))["fx_m2"]))) / fz_i
            for fz_i in fz_grid
        ]
        ax6.plot(fz_grid, mu_m2, color=col, lw=1.8)
    mu_m1 = [
        float(np.max(np.abs(evaluator.eval_pure_longitudinal(k_dense, fz_i, 0.0, 83.0)["fx_m1"]))) / fz_i
        for fz_i in fz_grid
    ]
    ax6.plot(fz_grid, mu_m1, **STYLE_M1, label="M1: Baseline (83 kPa)")

    ev_val = evaluator.eval_longitudinal_dataset(df_pure_val)
    add_metrics_box(ax6, {
        "Baseline": compute_metrics(ev_val["fx_exp"], ev_val["fx_m1"]),
        "Unified": compute_metrics(ev_val["fx_exp"], ev_val["fx_m2"]),
    }, loc="lower right", title=f"Pure Fx Val ({src_tag})")

    ax6.set_title(f"E6: Peak Friction $\\mu_x(F_z)$ & Global Validation ({c_upper})", fontsize=10.5, fontweight="bold")
    ax6.set_xlabel("Normal Load $F_z$ [N]", fontsize=9.5)
    ax6.set_ylabel("Normalized Peak $|F_{x,\\max}|/F_z$ [-]", fontsize=9.5)
    ax6.set_ylim(1.4, 3.05)
    ax6.legend(ncol=2, loc="upper right", fontsize=7.0)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f"Module 05: Pure Longitudinal Experimental Validation ({c_upper} — {src_tag})",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(out_dir, f"fig05_longitudinal_characterization_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_longitudinal_pressure_and_residuals(evaluator, df_pure_train, df_pure_val, compound_name, out_dir):
    """
    E7-E9 (1x3): Pressure & rim sensitivity (~885 N), Train/Val speed parity, and residuals vs kappa.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    src_tag = "R20 18x6 D/B" if c_upper == "R20" else "R20 18x6 Surrogate"

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)
    df_all = pd.concat([df_pure_train, df_pure_val], ignore_index=True)

    # Panel 1: Pressure Sensitivity at ~885 N, 0 deg IA
    ax1 = axes[0]
    k_dense = np.linspace(-0.24, 0.22, 260)
    k_edges = np.linspace(-0.23, 0.21, 22)
    p_defs = [
        ("8psi_40kph", 55.2, "#d73027", "55 kPa (8 psi)"),
        ("10psi_40kph", 69.0, "#fc8d59", "69 kPa (10 psi)"),
        ("final_12psi_40kph", 83.0, "#2980b9", "83 kPa (12 psi)"),
        ("14psi_40kph", 96.6, "#27ae60", "97 kPa (14 psi)"),
    ]
    sub_885 = df_all[(df_all["ia_deg"].abs() < 0.3) & ((df_all["f_z"] - 885.0).abs() <= 80.0)]
    for blk, p_nom, col, lbl in p_defs:
        s = sub_885[sub_885["block_id"] == blk]
        if len(s) >= 10:
            xc, ym, ys = _binned_curve(s["kappa"].values, s["f_x"].values, k_edges)
            ax1.errorbar(xc * 100.0, ym, yerr=ys, fmt="o", ms=3.5, elinewidth=1.0, capsize=2, color=col, alpha=0.85)
            pr = evaluator.eval_pure_longitudinal(k_dense, np.full_like(k_dense, float(s["f_z"].median())),
                                                  0.0, float(s["p_kpa"].median()))
            ax1.plot(k_dense * 100.0, pr["fx_m2"], "-", color=col, lw=2.0, label=lbl)
    ax1.axhline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax1.set_title(f"E7: Pressure Sensitivity (~885 N, 0° IA) — {c_upper}", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
    ax1.set_ylabel("Longitudinal Force $F_x$ [N]", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=8.0)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Validation Parity (24 / 40 / 72 kph)
    ax2 = axes[1]
    ev_val = evaluator.eval_longitudinal_dataset(df_pure_val)
    spd = pd.cut(df_pure_val["v_x"] * 3.6, [0, 32, 55, 100], labels=["24 kph", "40 kph", "72 kph"])
    for lbl_s, col_s in [("24 kph", "#1b9e77"), ("40 kph", COLOR_M2_OPT), ("72 kph", "#e41a1c")]:
        m = (spd == lbl_s).values
        if np.any(m):
            ax2.scatter(ev_val["fx_exp"][m], ev_val["fx_m2"][m], s=4, alpha=0.18, color=col_s,
                        label=f"{lbl_s} (N={int(np.sum(m)):,})", rasterized=True)
    lim = 3100.0
    m2_met = compute_metrics(ev_val["fx_exp"], ev_val["fx_m2"])
    ax2.plot([-lim, lim], [-lim, lim], "k-", lw=1.4, label="1:1 Parity")
    ax2.plot([-lim, lim], [-lim + m2_met["rmse"], lim + m2_met["rmse"]], "k--", lw=0.9, alpha=0.7, label="±1 RMSE")
    ax2.plot([-lim, lim], [-lim - m2_met["rmse"], lim - m2_met["rmse"]], "k--", lw=0.9, alpha=0.7)
    add_metrics_box(ax2, {
        "Baseline": compute_metrics(ev_val["fx_exp"], ev_val["fx_m1"]),
        "Unified": m2_met,
    }, loc="upper left", title=f"Val Parity ({src_tag})")
    ax2.set_title(f"E8: Exp vs. Predicted $F_x$ Parity ({c_upper})", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Experimental $F_x$ [N]", fontsize=9.5)
    ax2.set_ylabel("Predicted $F_x$ [N]", fontsize=9.5)
    ax2.legend(loc="lower right", fontsize=7.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Validation Residuals vs kappa
    ax3 = axes[2]
    res_m2 = ev_val["fx_m2"] - ev_val["fx_exp"]
    res_m1 = ev_val["fx_m1"] - ev_val["fx_exp"]
    ax3.scatter(ev_val["kappa"] * 100.0, res_m2, s=3, alpha=0.12, color=COLOR_M2_OPT, label="M2 Residuals", rasterized=True)
    xc, ym2, ys2 = _binned_curve(ev_val["kappa"], res_m2, np.linspace(-0.25, 0.23, 28))
    xc1, ym1, _ = _binned_curve(ev_val["kappa"], res_m1, np.linspace(-0.25, 0.23, 28))
    ax3.plot(xc1 * 100.0, ym1, **STYLE_M1, label="M1 Binned Mean")
    ax3.plot(xc * 100.0, ym2, "r-", lw=2.0, label="M2 Binned Mean")
    ax3.fill_between(xc * 100.0, ym2 - ys2, ym2 + ys2, color="r", alpha=0.2, label="M2 ±1σ Band")
    ax3.axhline(0.0, color="k", lw=1.0, linestyle="--")
    ax3.axvline(0.0, color="gray", lw=0.6, linestyle=":")
    ax3.set_title(f"E9: Validation Residuals vs $\\kappa$ ({c_upper})", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Longitudinal Slip Ratio $\\kappa$ [%]", fontsize=9.5)
    ax3.set_ylabel("Residual ($F_{x,\\text{pred}} - F_{x,\\text{exp}}$) [N]", fontsize=9.5)
    ax3.set_xlim(-25.0, 23.0)
    ax3.set_ylim(-600.0, 600.0)
    ax3.legend(loc="upper right", fontsize=7.8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig05_longitudinal_pressure_and_residuals_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
