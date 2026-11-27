#!/usr/bin/env python3
"""
Module 12: Transient Tire Dynamics & Carcass Relaxation
Generates:
  L1: Steer step response in effective slip angle alpha_eff(t) across speeds (5, 10, 20 m/s)
  L2: Transient lateral force buildup Fy(t) and carcass relaxation delay with 63.2% and 95% markers
PROMINENTLY LABELED AS NUMERICAL CHARACTERIZATION (MF 6.2 IS STEADY-STATE).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from core.style import apply_theme, add_characterization_badge


def generate_transient_relaxation_suite(evaluator, compound_name, out_dir):
    """
    L1 & L2: First-order carcass relaxation response across vehicle speeds.
    Incorporates explicit time constant (tau = sigma / V) and response indicators.
    """
    apply_theme()
    c_upper = compound_name.upper()
    c_lower = compound_name.lower()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=150)

    t_sim = np.linspace(0.0, 0.150, 600) # 150 ms
    dt = t_sim[1] - t_sim[0]
    speeds = [5.0, 10.0, 20.0]
    s_cols = ["#3498db", "#9b59b6", "#e67e22"]
    sigma = 0.180 # 180 mm relaxation length

    # Panel 1: Effective Slip Angle alpha_eff(t)
    for v_val, col in zip(speeds, s_cols):
        alpha_t = 0.0
        alpha_traj = []
        tau = sigma / max(v_val, 0.20)
        decay = np.exp(-dt / tau)

        for t in t_sim:
            alpha_target = 0.10 if t >= 0.010 else 0.0 # 0.1 rad step steer at t = 10 ms
            alpha_t = alpha_target + (alpha_t - alpha_target) * decay
            alpha_traj.append(alpha_t)

        ax1.plot(t_sim * 1000.0, np.degrees(alpha_traj), "-", color=col, lw=2.2,
                 label=f"V = {v_val*3.6:.0f} km/h ($\\tau$ = {tau*1000:.1f} ms)")

        # 63.2% response time = 10 ms + tau
        t_63 = (0.010 + tau) * 1000.0
        val_63 = np.degrees(0.10 * (1.0 - np.exp(-1.0)))
        ax1.plot(t_63, val_63, "o", color=col, markersize=5)

    ax1.axvline(10.0, color="gray", linestyle=":", label="Steer Step Input (10 ms)")
    ax1.axhline(np.degrees(0.10), color="k", linestyle="--", lw=1.2, label="Commanded Slip (5.73°)")
    ax1.set_title("L1: Carcass Relaxation: Contact Patch Slip Angle $\\alpha_{\\text{eff}}(t)$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time [ms]", fontsize=10)
    ax1.set_ylabel("Effective Slip Angle $\\alpha_{\\text{eff}}$ [deg]", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(True, alpha=0.3)
    add_characterization_badge(ax1, "NUMERICAL CHARACTERIZATION (MF 6.2 IS STEADY-STATE)")

    # Panel 2: Transient Lateral Force Buildup Fy(t)
    fz_nom = 887.0
    for v_val, col in zip(speeds, s_cols):
        alpha_t = 0.0
        fy_traj = []
        tau = sigma / max(v_val, 0.20)
        decay = np.exp(-dt / tau)

        for t in t_sim:
            alpha_target = 0.10 if t >= 0.010 else 0.0
            alpha_t = alpha_target + (alpha_t - alpha_target) * decay
            preds = evaluator.eval_pure_lateral(np.array([alpha_t]), np.array([fz_nom]), gamma=0.0, p=83.0)
            fy_traj.append(preds["fy_m3"][0])

        ax2.plot(t_sim * 1000.0, fy_traj, "-", color=col, lw=2.2, label=f"V = {v_val*3.6:.0f} km/h ($\\tau$ = {tau*1000:.1f} ms)")

    # Steady-state asymptote
    fy_ss = evaluator.eval_pure_lateral(np.array([0.10]), np.array([fz_nom]), gamma=0.0, p=83.0)["fy_m3"][0]
    ax2.axhline(fy_ss, color="red", linestyle="--", lw=1.5, label=f"Steady-State MF 6.2 ($F_y={fy_ss:.0f}$ N)")
    ax2.axvline(10.0, color="gray", linestyle=":")

    ax2.set_title(f"L2: Transient Cornering Force Buildup $F_y(t)$ at $F_z=887$ N ({c_upper})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time [ms]", fontsize=10)
    ax2.set_ylabel("Lateral Force $F_y$ [N]", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8.0)
    ax2.grid(True, alpha=0.3)
    add_characterization_badge(ax2, r"CARCASS FIRST-ORDER LAG MODEL ($\sigma = 0.180$ m)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"fig12_transient_relaxation_{c_lower}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
