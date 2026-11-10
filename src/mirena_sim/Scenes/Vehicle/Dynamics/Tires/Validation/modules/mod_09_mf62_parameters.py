#!/usr/bin/env python3
"""
Module 09: MF 6.2 Parameter Characterization & Identifiability
Generates:
  I1: Scaled Jacobian singular value spectra across calibration stages
  I2: Horizontal parameter uncertainty bars sorted by identifiability tiers (< 5%, 5-20%, > 20%)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme


def generate_parameter_identifiability_suite(evaluator, compound_name, out_dir):
    """
    I1 & I2: Singular value spectra and sorted horizontal parameter uncertainties.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    calib_rep_path = os.path.join(base_dir, f"calibration_report_{c_lower}.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.5), dpi=150)

    # Panel 1: SVD Spectra
    if os.path.exists(calib_rep_path):
        with open(calib_rep_path) as fp:
            calib_rep = json.load(fp)
        stages = calib_rep.get("identifiability_stages", {})
        marker_list = ["o", "s", "^", "D", "v", "P"]

        all_params = []
        all_uncertainties = []

        for (st_key, st_info), m in zip(stages.items(), marker_list):
            s_vals = st_info.get("singular_values", [])
            cond = st_info.get("scaled_condition_number", 1.0)
            st_name = st_info.get("stage", st_key)
            if len(s_vals) > 0:
                ax1.semilogy(range(1, len(s_vals) + 1), s_vals, f"-{m}", lw=1.8, markersize=6,
                             label=f"{st_name} (cond={cond:.1e})")

            params_dict = st_info.get("parameters", {})
            for p_name, p_stats in params_dict.items():
                if p_stats.get("optimal_value") is not None:
                    all_params.append(p_name)
                    all_uncertainties.append(min(p_stats.get("relative_uncertainty_pct", 0.0), 100.0))

        ax1.set_title(f"I1: Scaled Jacobian SVD Spectra across Stages ({c_upper})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Singular Value Index", fontsize=10)
        ax1.set_ylabel("Singular Value $\\sigma_i$ (Log Scale)", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8.0)
        ax1.grid(True, which="both", alpha=0.3)

        # Panel 2: Sorted Horizontal Uncertainty Bars
        if len(all_params) > 0:
            # Sort by uncertainty ascending
            sort_idx = np.argsort(all_uncertainties)
            sorted_params = [all_params[i] for i in sort_idx]
            sorted_u = [all_uncertainties[i] for i in sort_idx]

            y_pos = np.arange(len(sorted_params))
            cols = ["#27ae60" if u < 5.0 else "#f39c12" if u < 20.0 else "#e74c3c" for u in sorted_u]
            bars = ax2.barh(y_pos, sorted_u, color=cols, alpha=0.85, edgecolor="#7f8c8d", height=0.65)

            ax2.axvline(5.0, color="#27ae60", linestyle="--", lw=1.2, label="High Confidence (< 5%)")
            ax2.axvline(20.0, color="#f39c12", linestyle="--", lw=1.2, label="Moderate Confidence (< 20%)")

            # Value labels on bars
            for b, u in zip(bars, sorted_u):
                ax2.text(u + 1.2, b.get_y() + b.get_height() / 2.0, f"{u:.1f}%",
                         va="center", fontsize=7.5, fontweight="bold")

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_params, fontsize=8.5)
            ax2.set_title(f"I2: Parameter Relative Uncertainties $\\sigma_\\theta / |\\theta|$ [%] ({c_upper})", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Relative Asymptotic Uncertainty [%]", fontsize=10)
            ax2.set_xlim(0, 115)
            ax2.legend(loc="lower right", fontsize=8.0)
            ax2.grid(True, axis="x", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Calibration report not found", ha="center", va="center")
        ax2.text(0.5, 0.5, "Calibration report not found", ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig09_mf62_parameters_identifiability_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
