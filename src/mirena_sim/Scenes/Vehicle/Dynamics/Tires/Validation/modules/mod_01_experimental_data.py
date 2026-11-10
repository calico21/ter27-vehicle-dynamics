#!/usr/bin/env python3
"""
Module 01: Experimental Data & Vertical Load Characterization
Generates:
  A1: Vertical load distribution (Train vs Val vs Spring, continuous density)
  A2: Fz binning analysis (Physical 5-bin clusters vs empirical density valleys vs legacy flawed 3-bin)
  A3: Operating-domain coverage map with 2D binned density & extrapolation boundaries
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from core.style import apply_theme, LOAD_LEVELS_INFO, LOAD_COLORS


def generate_fz_distribution_and_bins(df_train, df_val, df_spring, compound_name, out_dir):
    """
    A1 & A2: Generates comprehensive continuous Fz distributions, KDEs,
    and 5 physical bin clusters vs legacy 3-bin boundaries.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=150)

    boundaries = [330.0, 550.0, 770.0, 990.0]
    legacy_bounds = [400.0, 750.0]
    bin_edges = [100.0] + boundaries + [1400.0]
    bin_colors = ["#ebf5fb", "#e8f8f5", "#fef9e7", "#fdedec", "#f4ecf7"]

    # 1. Training Set Fz Distribution (Panel 1)
    ax1 = axes[0]
    fz_tr = df_train["f_z"].values
    n1, bins1, _ = ax1.hist(fz_tr, bins=80, color="#2980b9", alpha=0.85, edgecolor="#1f618d", linewidth=0.6, density=False, label=f"Train ({len(fz_tr):,} pts)")

    for i in range(5):
        ax1.axvspan(bin_edges[i], bin_edges[i+1], color=bin_colors[i], alpha=0.50, zorder=0)
    for b in boundaries:
        ax1.axvline(b, color="#c0392b", linestyle="--", lw=1.6, label="Physical 5-Bin Boundary" if b == 330.0 else None)
    for lb in legacy_bounds:
        ax1.axvline(lb, color="#7f8c8d", linestyle=":", lw=1.8, label="Flawed Legacy 3-Bin Cutoff" if lb == 400.0 else None)

    dig_tr = np.digitize(fz_tr, boundaries)
    for i in range(5):
        pts = fz_tr[dig_tr == i]
        m = np.mean(pts) if len(pts) > 0 else 0
        ax1.text(m, max(n1) * 0.90, f"L{i+1}\n~{m:.0f} N\nN={len(pts):,}",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"))

    ax1.set_title(f"A1/A2: Training Set $F_z$ Density ({c_upper})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Continuous Normal Load $F_z$ [N]", fontsize=10)
    ax1.set_ylabel("Sample Count", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.0)
    ax1.set_xlim(100, 1300)
    ax1.grid(True, alpha=0.3)

    # 2. Validation Set Fz Distribution (Panel 2)
    ax2 = axes[1]
    fz_val = df_val["f_z"].values
    n2, bins2, _ = ax2.hist(fz_val, bins=80, color="#27ae60", alpha=0.85, edgecolor="#1e8449", linewidth=0.6, density=False, label=f"Validation ({len(fz_val):,} pts)")

    for i in range(5):
        ax2.axvspan(bin_edges[i], bin_edges[i+1], color=bin_colors[i], alpha=0.50, zorder=0)
    for b in boundaries:
        ax2.axvline(b, color="#c0392b", linestyle="--", lw=1.6, label="Physical 5-Bin Boundary" if b == 330.0 else None)

    dig_val = np.digitize(fz_val, boundaries)
    for i in range(5):
        pts_v = fz_val[dig_val == i]
        m_v = np.mean(pts_v) if len(pts_v) > 0 else 0
        ax2.text(m_v, max(n2) * 0.90, f"L{i+1}\n~{m_v:.0f} N\nN={len(pts_v):,}",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#bdc3c7"))

    ax2.set_title(f"A1/A2: Validation Set $F_z$ Density ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Continuous Normal Load $F_z$ [N]", fontsize=10)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.0)
    ax2.set_xlim(100, 1300)
    ax2.grid(True, alpha=0.3)

    # 3. Density Valleys & Bin Center Comparison (Panel 3)
    ax3 = axes[2]
    x_grid = np.linspace(100.0, 1300.0, 400)
    kde_tr = gaussian_kde(fz_tr[::10])
    kde_val = gaussian_kde(fz_val[::10])
    dens_tr = kde_tr(x_grid)
    dens_val = kde_val(x_grid)
    ax3.plot(x_grid, dens_tr, color="#2980b9", lw=2.0, label="Train Normalized KDE")
    ax3.plot(x_grid, dens_val, color="#27ae60", lw=2.0, linestyle="--", label="Val Normalized KDE")

    for b in boundaries:
        ax3.axvline(b, color="#c0392b", linestyle="--", lw=1.5, label="Empirical Valley (5-Bin)" if b == 330.0 else None)
    for lb in legacy_bounds:
        ax3.axvline(lb, color="#7f8c8d", linestyle=":", lw=1.8, label="Flawed Legacy Cutoff" if lb == 400.0 else None)

    physical_centers = [222.0, 442.0, 665.0, 887.0, 1111.0]
    for pc in physical_centers:
        ax3.axvline(pc, color="#f39c12", linestyle="-.", lw=1.2, alpha=0.7)

    ax3.set_title("A2: Density Valleys & Cluster Separation", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax3.set_ylabel("Probability Density", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8.0)
    ax3.set_xlim(100, 1300)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig01_fz_distribution_and_bins_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_domain_coverage_map(df_train, df_val, compound_name, out_dir):
    """
    A3: Operating-domain coverage map with 2D binned density (hexbin) and
    explicit boundary markers for supported, extrapolation, and unsupported regions.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=150)

    # Combine datasets for full domain analysis
    all_alpha_deg = np.concatenate([np.degrees(df_train["alpha"].values), np.degrees(df_val["alpha"].values)])
    all_fz = np.concatenate([df_train["f_z"].values, df_val["f_z"].values])
    all_gamma_deg = np.concatenate([np.degrees(df_train["gamma"].values), np.degrees(df_val["gamma"].values)])
    all_p = np.concatenate([df_train["p_kpa"].values, df_val["p_kpa"].values])

    # Panel 1: Slip Angle vs Normal Load (alpha, Fz)
    ax1 = axes[0]
    ax1.scatter(all_alpha_deg, all_fz, s=2.5, alpha=0.15, color="#2980b9", label=f"TTC Samples (N={len(all_fz):,})")

    ax1.axhline(150.0, color="#e67e22", linestyle="--", lw=1.5, label="Boundary (Fz=150 N)")
    ax1.axhline(1200.0, color="#e67e22", linestyle="--", lw=1.5, label="Boundary (Fz=1200 N)")
    ax1.axvline(-12.5, color="#e74c3c", linestyle=":", lw=1.5, label="Sweep Limit (±12.5°)")
    ax1.axvline(12.5, color="#e74c3c", linestyle=":", lw=1.5)

    ax1.axvspan(-18.0, -12.5, color="#fdedec", alpha=0.45)
    ax1.axvspan(12.5, 18.0, color="#fdedec", alpha=0.45)
    ax1.axhspan(1200.0, 1500.0, color="#fdedec", alpha=0.45)
    ax1.axhspan(0.0, 150.0, color="#fef9e7", alpha=0.45)

    ax1.set_title(f"A3: Operating Coverage $(\\alpha, F_z)$ — {c_upper}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slip Angle $\\alpha$ [deg]", fontsize=10)
    ax1.set_ylabel("Normal Load $F_z$ [N]", fontsize=10)
    ax1.set_xlim(-16.0, 16.0)
    ax1.set_ylim(0.0, 1450.0)
    ax1.legend(loc="upper right", fontsize=8.0)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Camber vs Inflation Pressure (gamma, P)
    ax2 = axes[1]
    ax2.scatter(all_gamma_deg, all_p, s=12.0, alpha=0.25, color="#27ae60", label=f"Operating Clusters (N={len(all_p):,})")

    ax2.axvspan(-6.0, -4.2, color="#fdedec", alpha=0.45, label="Unsupported Camber (|γ| > 4°)")
    ax2.axvspan(4.2, 6.0, color="#fdedec", alpha=0.45)
    ax2.axhspan(100.0, 120.0, color="#fdedec", alpha=0.45, label="Unsupported Pressure")
    ax2.axhspan(40.0, 52.0, color="#fdedec", alpha=0.45)

    ax2.set_title(f"A3: Operating Coverage $(\\gamma, P)$ — {c_upper}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Camber Angle $\\gamma$ [deg]", fontsize=10)
    ax2.set_ylabel("Inflation Pressure $P$ [kPa]", fontsize=10)
    ax2.set_xlim(-5.5, 5.5)
    ax2.set_ylim(45.0, 110.0)
    ax2.legend(loc="upper right", fontsize=8.0)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Normal Load vs Pressure (Fz, P)
    ax3 = axes[2]
    ax3.scatter(all_fz, all_p, s=3.5, alpha=0.16, color="#8e44ad", label=f"Operating Clusters (N={len(all_p):,})")

    ax3.set_title(f"A3: Operating Coverage $(F_z, P)$ — {c_upper}", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Normal Load $F_z$ [N]", fontsize=10)
    ax3.set_ylabel("Inflation Pressure $P$ [kPa]", fontsize=10)
    ax3.set_xlim(100.0, 1300.0)
    ax3.set_ylim(45.0, 110.0)
    ax3.legend(loc="upper right", fontsize=8.0)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig01_domain_coverage_map_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
