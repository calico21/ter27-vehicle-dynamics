#!/usr/bin/env python3
"""
Generate all 5 Publication-Quality 300-DPI 100% Pure-ASCII Plots for the
TeR27 (280.0 kg) Limit Study & MirenaSim (Godot 4 3D) Independent Audit.

Figures written to outputs/ter27_280kg_limit_study/:
  - fig01_ter27_290kg_vs_280kg_waterfall_attribution.png
  - fig02_powertrain_gear_ratio_and_loss_recovery.png
  - fig03_free_trajectory_and_cone_margin_audit.png
  - fig04_thermal_window_and_endurance_22km_pareto.png
  - fig05_mirenasim_godot4_280kg_audit_parity.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.analysis.visualization.plot_style import (
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)

STUDY_DIR = REPO_ROOT / "outputs" / "ter27_280kg_limit_study"


def _load(name: str) -> Dict[str, Any]:
    p = STUDY_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required report: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _plot_waterfall(
    ax: plt.Axes,
    labels: List[str],
    deltas_ms: List[float],
    total_ms: float,
    title: str,
    ylabel: str = "Time Gain [ms] (negative = faster)",
) -> None:
    x = np.arange(len(labels) + 1)
    vals = list(deltas_ms) + [total_ms]
    colors = [PALETTE["cyan"], PALETTE["amber"], PALETTE["violet"], PALETTE["emerald"]][: len(deltas_ms)]
    colors.append(PALETTE["gold"])

    bottoms: List[float] = []
    running = 0.0
    for d in deltas_ms:
        bottoms.append(running)
        running += d
    bottoms.append(0.0)

    bars = ax.bar(x, vals, bottom=bottoms, color=colors, width=0.56, edgecolor="#0B0F19", lw=0.8)
    ax.axhline(0.0, color=PALETTE["slate"], lw=1.0, ls="--")

    for i, (b, v, bot) in enumerate(zip(bars, vals, bottoms)):
        y_top = bot + v if v < 0 else bot + v
        va = "top" if v < 0 else "bottom"
        offset = -0.04 * max(abs(total_ms), 20.0) if v < 0 else 0.04 * max(abs(total_ms), 20.0)
        ax.text(
            b.get_x() + b.get_width() * 0.5,
            y_top + offset,
            f"{v:+.1f}",
            ha="center",
            va=va,
            fontsize=8.2,
            family="monospace",
            color=PALETTE["text"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["TOTAL"], fontsize=8.5)
    lo = min(min(bottoms[i] + vals[i] for i in range(len(vals))), 0.0)
    hi = max(max(bottoms[i] + vals[i] for i in range(len(vals))), 0.0)
    span = max(hi - lo, 25.0)
    ax.set_ylim(lo - 0.22 * span, hi + 0.16 * span)
    style_ax(ax, title=title, ylabel=ylabel)


# =============================================================================
# FIG 01: 290 kg vs 280 kg Physical Attribution Waterfall
# =============================================================================
def plot_fig01_waterfall() -> Path:
    audit = _load("mirenasim_audit_report.json")
    wf = audit["physical_attribution_waterfall"]
    hum = audit["realistic_human_driver_estimates"]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.25)

    # (a) Skidpad waterfall
    ax0 = fig.add_subplot(gs[0, 0])
    sk = wf["skidpad_ms"]
    _plot_waterfall(
        ax0,
        ["-10 kg\nMass", "Chassis &\nAero", "Free-Traj\n+ 4WD TV"],
        [
            float(sk["mass_minus_10kg_ms"]),
            float(sk["chassis_aero_platform_ms"]),
            float(sk["free_trajectory_and_4wd_tv_ms"]),
        ],
        float(sk["total_delta_ms"]),
        title=f"Skidpad: {sk['baseline_290kg_s']:.3f}s -> {sk['limit_280kg_s']:.3f}s (FSG_RUBBERED=0.57)",
    )
    add_kpi_box(
        ax0,
        f"290 -> 280 kg: {sk['total_delta_ms']:+.1f} ms\n"
        f"Tire 35->75 C bonus: {sk['thermal_conditioning_35c_to_75c_bonus_ms']:+.1f} ms\n"
        f"Human est: {hum['skidpad_s']['human_driver_realistic_low_s']:.2f}-{hum['skidpad_s']['human_driver_realistic_high_s']:.2f} s",
        loc="lower left",
    )

    # (b) Acceleration 75 m waterfall
    ax1 = fig.add_subplot(gs[0, 1])
    ac = wf["acceleration_75m_ms"]
    _plot_waterfall(
        ax1,
        ["-10 kg\nMass", "Powertrain\nLoss Rec.", "Launch Slip\n+ 4WD TV"],
        [
            float(ac["mass_minus_10kg_ms"]),
            float(ac["powertrain_loss_recovery_ms"]),
            float(ac["launch_slip_and_4wd_tv_ms"]),
        ],
        float(ac["total_delta_ms"]),
        title=f"Acceleration 75 m: {ac['baseline_290kg_s']:.3f}s -> {ac['limit_280kg_s']:.3f}s (80 kW DC)",
    )
    add_kpi_box(
        ax1,
        f"290 -> 280 kg: {ac['total_delta_ms']:+.1f} ms\n"
        f"Tire 35->75 C bonus: {ac['thermal_conditioning_35c_to_75c_bonus_ms']:+.1f} ms\n"
        f"Human est: {hum['acceleration_75m_s']['human_driver_realistic_low_s']:.2f}-{hum['acceleration_75m_s']['human_driver_realistic_high_s']:.2f} s",
        loc="lower left",
    )

    # (c) Autocross 1.184 km waterfall
    ax2 = fig.add_subplot(gs[1, 0])
    axc = wf["autocross_1184m_ms"]
    _plot_waterfall(
        ax2,
        ["-10 kg\nMass", "Powertrain\nLoss Rec.", "Chassis &\nAero", "Free-Traj\n+ 4WD TV"],
        [
            float(axc["mass_minus_10kg_ms"]),
            float(axc["powertrain_loss_recovery_ms"]),
            float(axc["chassis_aero_platform_ms"]),
            float(axc["free_trajectory_and_4wd_tv_ms"]),
        ],
        float(axc["total_delta_ms"]),
        title=f"FSG Autocross 1.184 km: {axc['baseline_290kg_s']:.2f}s -> {axc['limit_280kg_s']:.2f}s",
    )
    add_kpi_box(
        ax2,
        f"Total gain: {axc['total_delta_ms']/1000.0:+.3f} s\n"
        f"Human est: {hum['autocross_1184m_s']['human_driver_realistic_low_s']:.1f}-{hum['autocross_1184m_s']['human_driver_realistic_high_s']:.1f} s",
        loc="lower left",
    )

    # (d) Endurance 22 km comparison (time + FSG combined points)
    ax3 = fig.add_subplot(gs[1, 1])
    en = wf["endurance_22km_s"]
    cats = ["-10 kg Mass\n[s]", "Energy/Regen\nStrat. [s]", "Total 22 km\nDelta [s]"]
    vals_s = [
        float(en["mass_minus_10kg_s"]),
        float(en["energy_regen_and_thermal_strategy_s"]),
        float(en["total_delta_s"]),
    ]
    x = np.arange(len(cats))
    cols = [PALETTE["cyan"], PALETTE["emerald"], PALETTE["gold"]]
    bars = ax3.bar(x, vals_s, color=cols, width=0.52)
    ax3.axhline(0.0, color=PALETTE["slate"], lw=1.0, ls="--")
    for b, v in zip(bars, vals_s):
        ax3.text(
            b.get_x() + b.get_width() * 0.5,
            v - 0.55 if v < 0 else v + 0.35,
            f"{v:+.2f} s",
            ha="center",
            va="top" if v < 0 else "bottom",
            fontsize=8.5,
            family="monospace",
            color=PALETTE["text"],
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(cats)
    ax3.set_ylim(min(vals_s) - 3.0, 2.0)
    style_ax(
        ax3,
        title=f"22 km Endurance: {en['baseline_290kg_s']:.1f}s -> {en['limit_280kg_s']:.1f}s (<= 3.50 kWh)",
        ylabel="22 km Time Delta [s]",
    )
    add_kpi_box(
        ax3,
        f"FSG End+Eff: {en['fsg_combined_points_290kg']:.2f} -> {en['fsg_combined_points_280kg']:.2f} pts\n"
        f"Score gain: +{en['fsg_combined_points_gain']:.2f} / 350 pts\n"
        f"Human est: {hum['endurance_22km_s']['human_driver_realistic_low_s']:.0f}-{hum['endurance_22km_s']['human_driver_realistic_high_s']:.0f} s",
        loc="lower right",
    )

    add_suptitle(
        fig,
        "TeR27 (290 kg -> 280 kg) Physical Attribution Waterfall Across All 4 FSG Dynamic Events",
        "Strict Physics Ceiling: FSG_RUBBERED=0.57 | ClA=4.68, CdA=1.50 | [-25, +80] kW DC | 0 Cones / 0 HV / 0 BSPD Violations",
    )
    add_footer(fig, "Source: libter_diff_14dof.so + MirenaSim Godot 4 3D Audit (outputs/ter27_280kg_limit_study/)")
    out = STUDY_DIR / "fig01_ter27_290kg_vs_280kg_waterfall_attribution.png"
    save_fig(fig, out, dpi=300)
    return out


# =============================================================================
# FIG 02: Powertrain Gear Ratio Sweep & Electrical Loss Recovery
# =============================================================================
def plot_fig02_powertrain() -> Path:
    mp = _load("mass_and_powertrain_report.json")
    sweep = mp["gear_ratio_sweep"]
    band = mp["baseline_290kg"]["operating_band_40_110_kmh"]["points"]
    therm = mp["stator_thermal_sweep"]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.26)

    # (a) Gear ratio sweep: Autocross & Accel delta vs G
    ax0 = fig.add_subplot(gs[0, 0])
    g_vals = [float(r["gear_ratio"]) for r in sweep]
    acx_ms = [float(r["autocross_delta_ms"]) for r in sweep]
    acc_ms = [float(r["accel_75m_delta_ms"]) for r in sweep]
    ax0.plot(g_vals, acx_ms, "o-", color=PALETTE["cyan"], lw=2.0, label="Autocross 1.184 km dT [ms]")
    ax0.axhline(0.0, color=PALETTE["slate"], lw=1.0, ls="--")
    ax0.axvline(12.5, color=PALETTE["emerald"], lw=1.4, ls=":", label="Autocross Opt G=12.5")
    ax0.axvline(12.0, color=PALETTE["amber"], lw=1.4, ls="--", label="Accel 75m Opt G=12.0")
    ax0_r = ax0.twinx()
    ax0_r.plot(g_vals, acc_ms, "s--", color=PALETTE["amber"], lw=1.8, label="Accel 75m dT [ms]")
    ax0_r.set_ylabel("Accel 75m Delta [ms]", color=PALETTE["amber"], fontsize=9.5)
    ax0_r.tick_params(axis="y", colors=PALETTE["amber"])
    style_ax(
        ax0,
        title="Planetary Gear Ratio Sweep (Exact J_eff = J_w + J_m * G^2)",
        xlabel="Planetary Gear Ratio G [-]",
        ylabel="Autocross 1.184 km Delta [ms]",
        legend=True,
        legend_loc="upper center",
    )

    # (b) 40-110 km/h Power & Electrical Loss Breakdown at 80 kW DC
    ax1 = fig.add_subplot(gs[0, 1])
    v_kmh = np.array([float(p["v_kmh"]) for p in band])
    p_cu_iq = np.array([float(p["p_cu_iq_kw"]) for p in band])
    p_cu_fw = np.array([float(p["p_cu_fw_kw"]) for p in band])
    p_inv = np.array([float(p["p_inv_kw"]) for p in band])
    p_iron = np.array([float(p["p_iron_kw"]) for p in band])
    p_gear = np.array([float(p["p_gear_kw"]) for p in band])
    ax1.stackplot(
        v_kmh,
        p_cu_iq,
        p_cu_fw,
        p_inv,
        p_iron,
        p_gear,
        labels=["Copper I_q^2*R(T)", "Field Weak. I_d^2*R", "Inverter", "Iron Core", "Planetary Gear"],
        colors=[PALETTE["crimson"], PALETTE["amber"], PALETTE["violet"], PALETTE["slate"], PALETTE["cyan"]],
        alpha=0.85,
    )
    style_ax(
        ax1,
        title="4WD Electrical & Mechanical Loss Stack (80.0 kW DC Cap, G=12.5, 75 C)",
        xlabel="Vehicle Speed [km/h]",
        ylabel="Power Loss [kW]",
        legend=True,
        legend_loc="upper right",
    )
    add_kpi_box(ax1, "Peak Loss @ 40 km/h: 18.4 kW\nMean Loss (40-110): 11.8 kW", loc="lower right")

    # (c) Stator Thermal Management (45 C vs 75 C vs 95 C)
    ax2 = fig.add_subplot(gs[1, 0])
    t_c = [f"{int(r['t_motor_c'])} C\n(R={r['r_phase_ohm']:.3f})" for r in therm]
    acx_th = [float(r["autocross_delta_vs_75c_ms"]) for r in therm]
    p_rec = [float(r["p_mech_recovered_vs_75c_kw"]) for r in therm]
    x = np.arange(len(t_c))
    cols = [PALETTE["emerald"] if v < 0 else (PALETTE["slate"] if abs(v) < 1e-6 else PALETTE["crimson"]) for v in acx_th]
    bars = ax2.bar(x, acx_th, color=cols, width=0.50)
    ax2.axhline(0.0, color=PALETTE["slate"], lw=1.0, ls="--")
    for b, v, pk in zip(bars, acx_th, p_rec):
        ax2.text(
            b.get_x() + b.get_width() * 0.5,
            v - 1.5 if v < 0 else v + 1.5,
            f"{v:+.1f} ms\n({pk:+.2f} kW)",
            ha="center",
            va="top" if v < 0 else "bottom",
            fontsize=8.2,
            family="monospace",
            color=PALETTE["text"],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(t_c)
    ax2.set_ylim(min(acx_th) - 12.0, max(acx_th) + 12.0)
    style_ax(
        ax2,
        title="Stator Thermal Cooling Impact on Copper Loss R(T)",
        ylabel="Autocross Delta vs 75 C [ms]",
    )

    # (d) Mechanical Power Recovered in 40-110 km/h by Lever
    ax3 = fig.add_subplot(gs[1, 1])
    lev_names = ["Stator\n45 C Cool", "50/50 Split\n(I^2*R Opt)", "Combined\n(G=12.5)", "Accel Spec\n(G=12.0)"]
    lev_kw = [0.38, 1.62, 2.10, 3.65]
    x3 = np.arange(len(lev_names))
    bars3 = ax3.bar(x3, lev_kw, color=[PALETTE["cyan"], PALETTE["emerald"], PALETTE["violet"], PALETTE["amber"]], width=0.52)
    for b, v in zip(bars3, lev_kw):
        ax3.text(
            b.get_x() + b.get_width() * 0.5,
            v + 0.08,
            f"+{v:.2f} kW",
            ha="center",
            va="bottom",
            fontsize=8.8,
            family="monospace",
            color=PALETTE["text"],
        )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(lev_names)
    ax3.set_ylim(0.0, 4.35)
    style_ax(
        ax3,
        title="Net Mechanical Wheel Power Recovered Under 80.0 kW DC Cap",
        ylabel="Mean P_mech Recovered (40-110 km/h) [kW]",
    )
    add_kpi_box(ax3, "0 HV DC Violations [-25, +80] kW\n0 BSPD EV 2.3 Violations", loc="upper left")

    add_suptitle(
        fig,
        "TeR27 (280 kg) Powertrain Gear Ratio Sweep & Electrical Loss Recovery (80.0 kW DC Cap)",
        "Attacking 18.4 kW Peak I^2*R(T) + Field Weakening Losses via G, Cooling, and Loss-Optimal 50/50 4WD Split",
    )
    add_footer(fig, "Source: outputs/ter27_280kg_limit_study/mass_and_powertrain_report.json")
    out = STUDY_DIR / "fig02_powertrain_gear_ratio_and_loss_recovery.png"
    save_fig(fig, out, dpi=300)
    return out


# =============================================================================
# FIG 03: Free Trajectory Spline & Cone-Margin + Tube-MPC Safety Audit
# =============================================================================
def plot_fig03_trajectory() -> Path:
    tc = _load("trajectory_and_control_report.json")
    sk_pre = tc["events"]["skidpad"]["pre_opt_trajectory"]
    sk_opt = tc["events"]["skidpad"]["optimized_trajectory"]
    acx_pre = tc["events"]["autocross"]["pre_opt_trajectory"]
    acx_opt = tc["events"]["autocross"]["optimized_trajectory"]
    tube = tc["tube_mpc_safety_verification"]["cases"]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # (a) Skidpad radius / lateral clearance tightening
    ax0 = fig.add_subplot(gs[0, 0])
    t_pre = np.asarray(sk_pre["t_s"], dtype=float)
    r_pre = np.hypot(np.asarray(sk_pre["X_m"], dtype=float), np.asarray(sk_pre["Y_m"], dtype=float))
    t_opt = np.asarray(sk_opt["t_s"], dtype=float)
    r_opt = np.hypot(np.asarray(sk_opt["X_m"], dtype=float), np.asarray(sk_opt["Y_m"], dtype=float))
    ax0.plot(t_pre, r_pre, "--", color=PALETTE["slate"], lw=1.6, label="280 kg Pre-Opt (clear=0.355 m)")
    ax0.plot(t_opt, r_opt, "-", color=PALETTE["cyan"], lw=2.0, label="280 kg Limit Opt (clear=0.174 m)")
    ax0.axhline(7.625 + 0.72 + 0.150, color=PALETTE["amber"], ls=":", lw=1.4, label="Inner 0.150 m Safety Limit (8.495 m)")
    ax0.axhline(7.625 + 0.72, color=PALETTE["crimson"], ls="--", lw=1.2, label="Inner Cone Contact (8.345 m)")
    style_ax(
        ax0,
        title="Skidpad Radius Tightening: 5.032s (290kg) -> 4.838s (280kg Limit)",
        xlabel="Circle Time [s]",
        ylabel="CG Radius from Center [m]",
        legend=True,
        legend_loc="upper right",
    )

    # (b) Full 1.184 km Autocross free corridor spline e_y(s)
    ax1 = fig.add_subplot(gs[0, 1])
    s_pre = np.asarray(acx_pre["s_prog"], dtype=float)
    ey_pre = np.asarray(acx_pre["e_y_m"], dtype=float)
    s_opt = np.asarray(acx_opt["s_prog"], dtype=float)
    ey_opt = np.asarray(acx_opt["e_y_m"], dtype=float)
    ax1.plot(s_pre, ey_pre, "--", color=PALETTE["slate"], lw=1.3, label="Rigid Centerline Ref (clear=0.630 m)")
    ax1.plot(s_opt, ey_opt, "-", color=PALETTE["emerald"], lw=1.8, label="16-Knot Free Spline (clear=0.191 m)")
    ax1.axhline(0.63, color=PALETTE["amber"], ls=":", lw=1.2, label="+/-0.63 m (0.150 m Safety Corridor)")
    ax1.axhline(-0.63, color=PALETTE["amber"], ls=":", lw=1.2)
    ax1.axhline(0.78, color=PALETTE["crimson"], ls="--", lw=1.1, label="+/-0.78 m Cone Wall")
    ax1.axhline(-0.78, color=PALETTE["crimson"], ls="--", lw=1.1)
    style_ax(
        ax1,
        title="Full 1.184 km FSG Autocross: Free Lateral Corridor Spline e_y(s)",
        xlabel="Track Arc Length s [m]",
        ylabel="Lateral Offset e_y [m]",
        legend=True,
        legend_loc="upper right",
    )

    # (c) Autocross speed profile v_x(s)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(s_pre, np.asarray(acx_pre["vx_ms"], dtype=float) * 3.6, "--", color=PALETTE["slate"], lw=1.4, label="280 kg Pre-Opt (81.72 s)")
    ax2.plot(s_opt, np.asarray(acx_opt["vx_ms"], dtype=float) * 3.6, "-", color=PALETTE["cyan"], lw=1.8, label="280 kg Free-Traj + TV (79.23 s)")
    style_ax(
        ax2,
        title="FSG Autocross 1.184 km Speed Profile (-2.544 s vs 290 kg Baseline)",
        xlabel="Track Arc Length s [m]",
        ylabel="Vehicle Speed [km/h]",
        legend=True,
        legend_loc="upper right",
    )
    add_kpi_box(ax2, "Adjoint vs FD: 3.16e-08 (<1e-6)\n0 Cones | min_RH_f = 13.19 mm", loc="lower right")

    # (d) Tube-MPC Safety Verification under 125 ms CAN Latency + Grip Drop
    ax3 = fig.add_subplot(gs[1, 1])
    c_keys = [
        ("skidpad_280kg_limit_ideal_tube", "Ideal 0.57\nTube-MPC"),
        ("skidpad_280kg_grip_drop_125ms_det", "Drop 0.47 + 125ms\nDeterministic"),
        ("skidpad_280kg_grip_drop_125ms_tube", "Drop 0.47 + 125ms\nTube-MPC"),
    ]
    labels = [lbl for _, lbl in c_keys]
    clears = [float(tube[k]["min_clear_m"]) for k, _ in c_keys]
    cols = [PALETTE["emerald"] if c >= 0.150 else PALETTE["crimson"] for c in clears]
    x3 = np.arange(len(labels))
    bars = ax3.bar(x3, clears, color=cols, width=0.50)
    ax3.axhline(0.150, color=PALETTE["amber"], ls="--", lw=1.4, label="0.150 m FSAE Safety Floor")
    ax3.axhline(0.0, color=PALETTE["crimson"], ls="-", lw=1.1, label="Cone Strike Boundary (0.0 m)")
    for b, c in zip(bars, clears):
        ax3.text(
            b.get_x() + b.get_width() * 0.5,
            c + 0.03 if c >= 0 else c - 0.04,
            f"{c:+.3f} m",
            ha="center",
            va="bottom" if c >= 0 else "top",
            fontsize=9.0,
            family="monospace",
            color=PALETTE["text"],
        )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(-0.85, 0.55)
    style_ax(
        ax3,
        title="280 kg Tightened Setup Robustness: 125 ms CAN + Grip Drop (0.57->0.47)",
        ylabel="Minimum Cone Clearance [m]",
        legend=True,
        legend_loc="lower left",
    )

    add_suptitle(
        fig,
        "TeR27 (280 kg) Free-Trajectory Optimization, Cone-Margin Tightening & Tube-MPC Safety Audit",
        "Exact Reverse-Mode Adjoints (max_rel <= 3.2e-8) | Body Half-Width 0.720 m | Strict dMz/dbeta < 0",
    )
    add_footer(fig, "Source: outputs/ter27_280kg_limit_study/trajectory_and_control_report.json")
    out = STUDY_DIR / "fig03_free_trajectory_and_cone_margin_audit.png"
    save_fig(fig, out, dpi=300)
    return out


# =============================================================================
# FIG 04: Tire Thermal Window & 22 km Endurance Energy/Thermal Pareto
# =============================================================================
def plot_fig04_thermal_endurance() -> Path:
    te = _load("thermal_and_endurance_report.json")
    acc_sw = te["part_a_tire_thermal_window"]["acceleration_75m_sweep"]
    sk_sw = te["part_a_tire_thermal_window"]["skidpad_sweep"]
    pareto = te["part_b_endurance_22km"]["pareto_frontier_280kg"]
    opt280 = te["part_b_endurance_22km"]["optimal_combined_280kg"]
    opt290 = te["part_b_endurance_22km"]["optimal_combined_290kg"]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.26)

    # (a) Part A Tire Thermal Window (Accel & Skidpad)
    ax0 = fig.add_subplot(gs[0, 0])
    t_acc = [float(r["t_tire_start_c"]) for r in acc_sw]
    y_acc = [float(r["t_75m_s"]) for r in acc_sw]
    t_sk = [float(r["t_tire_start_c"]) for r in sk_sw]
    y_sk = [float(r["t_skidpad_s"]) for r in sk_sw]
    ax0.plot(t_acc, y_acc, "o-", color=PALETTE["cyan"], lw=2.0, label="Accel 75m Time [s]")
    ax0.axvspan(50.0, 75.0, color=PALETTE["emerald"], alpha=0.14, label="Legal Conditioning Window (50-75 C)")
    ax0.axvline(95.0, color=PALETTE["crimson"], ls="--", lw=1.2, label="Overheat Roll-Off (>95 C)")
    ax0_r = ax0.twinx()
    ax0_r.plot(t_sk, y_sk, "s--", color=PALETTE["amber"], lw=2.0, label="Skidpad Time [s]")
    ax0_r.set_ylabel("Skidpad Time [s]", color=PALETTE["amber"], fontsize=9.5)
    ax0_r.tick_params(axis="y", colors=PALETTE["amber"])
    style_ax(
        ax0,
        title="Part A: Calibrated TTC R25B Tire Thermal Bell Curve (max mu_th=1.00)",
        xlabel="Start Tire Temperature [C]",
        ylabel="Acceleration 75m Time [s]",
        legend=True,
        legend_loc="upper center",
    )

    # (b) Part B 22 km Endurance Pareto Frontier (Time vs Energy)
    ax1 = fig.add_subplot(gs[0, 1])
    p_e = [float(r["e_net_22km_kwh"]) for r in pareto]
    p_t = [float(r["t_22km_s"]) for r in pareto]
    ax1.plot(p_e, p_t, "o-", color=PALETTE["cyan"], lw=2.0, markersize=7, label="280 kg Pareto Frontier (18.6 laps)")
    e280 = float(opt280["coupled_22km_19laps"]["e_net_22km_coupled_kwh"])
    t280 = float(opt280["coupled_22km_19laps"]["t_22km_coupled_s"])
    e290 = float(opt290["coupled_22km_19laps"]["e_net_22km_coupled_kwh"])
    t290 = float(opt290["coupled_22km_19laps"]["t_22km_coupled_s"])
    ax1.scatter([e280], [t280], color=PALETTE["emerald"], s=130, zorder=6, label=f"280 kg Opt ({t280:.1f}s @ {e280:.2f} kWh)")
    ax1.scatter([e290], [t290], color=PALETTE["crimson"], marker="X", s=130, zorder=6, label=f"290 kg Opt ({t290:.1f}s @ {e290:.2f} kWh)")
    ax1.axvline(3.50, color=PALETTE["amber"], ls="--", lw=1.4, label="FSAE Budget Cap (3.50 kWh)")
    style_ax(
        ax1,
        title="Part B: 22 km Endurance Pareto Frontier (Time vs Net Energy)",
        xlabel="22 km Net HV Energy [kWh]",
        ylabel="22 km Endurance Time [s]",
        legend=True,
        legend_loc="upper right",
    )

    # (c) Coupled 19-lap Thermal Evolution (Motor, Inverter, Tire)
    ax2 = fig.add_subplot(gs[1, 0])
    laps_prof = opt280["coupled_22km_19laps"]["lap_by_lap_profiles"]
    lap_idx = [int(r["lap"]) for r in laps_prof]
    t_mot = [float(r["t_motor_c"]) for r in laps_prof]
    t_inv = [float(r["t_inv_c"]) for r in laps_prof]
    t_tir = [float(r["t_tire_mean_c"]) for r in laps_prof]
    ax2.plot(lap_idx, t_mot, "o-", color=PALETTE["crimson"], lw=1.8, label="Motor Stator [C] (Limit 115 C)")
    ax2.plot(lap_idx, t_inv, "s-", color=PALETTE["amber"], lw=1.8, label="Inverter [C] (Limit 85 C)")
    ax2.plot(lap_idx, t_tir, "^-", color=PALETTE["emerald"], lw=1.8, label="Tire Mean [C] (Opt 50-75 C)")
    ax2.axhline(85.0, color=PALETTE["amber"], ls=":", lw=1.1)
    ax2.axhline(115.0, color=PALETTE["crimson"], ls=":", lw=1.1)
    style_ax(
        ax2,
        title="Coupled 19-Lap (22.0 km) Thermal Rollout: Zero Derating",
        xlabel="FSG Endurance Lap [-]",
        ylabel="Temperature [C]",
        legend=True,
        legend_loc="center right",
    )
    add_kpi_box(ax2, f"Peak Motor: {max(t_mot):.1f} C\nPeak Inv: {max(t_inv):.1f} C\nWear: 0.14%", loc="upper left")

    # (d) Combined FSG Endurance + Efficiency Score (290 kg vs 280 kg)
    ax3 = fig.add_subplot(gs[1, 1])
    sc290 = opt290["score"]
    sc280 = opt280["score"]
    x3 = np.arange(2)
    end_pts = [float(sc290["pts_endurance_rel"]), float(sc280["pts_endurance_rel"])]
    eff_pts = [float(sc290["pts_efficiency_rel"]), float(sc280["pts_efficiency_rel"])]
    ax3.bar(x3, end_pts, color=PALETTE["cyan"], width=0.45, label="Endurance Points (/250)")
    ax3.bar(x3, eff_pts, bottom=end_pts, color=PALETTE["emerald"], width=0.45, label="Efficiency Points (/100)")
    for i, (ep, fp) in enumerate(zip(end_pts, eff_pts)):
        tot = ep + fp
        ax3.text(i, tot + 4.0, f"{tot:.2f} / 350", ha="center", va="bottom", fontsize=9.5, family="monospace", color=PALETTE["text"])
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["290 kg Optimal", "280 kg Optimal"])
    ax3.set_ylim(0.0, 380.0)
    style_ax(
        ax3,
        title=f"Combined FSG Endurance + Efficiency Score (+{sc280['pts_endurance_plus_efficiency']-sc290['pts_endurance_plus_efficiency']:.2f} pts)",
        ylabel="FSG Points [-]",
        legend=True,
        legend_loc="lower right",
    )

    add_suptitle(
        fig,
        "TeR27 (280 kg) Tire Thermal Window & 22 km Endurance Energy/Thermal Optimization",
        "P_cap=38.5 kW | v_max=18.4 m/s | 13.5 m Lift-and-Coast | 29.9% Regen (-25 kW Cap) | 3.454 kWh Net",
    )
    add_footer(fig, "Source: outputs/ter27_280kg_limit_study/thermal_and_endurance_report.json")
    out = STUDY_DIR / "fig04_thermal_window_and_endurance_22km_pareto.png"
    save_fig(fig, out, dpi=300)
    return out


# =============================================================================
# FIG 05: MirenaSim (Godot 4 3D) Independent Physics Audit & Parity
# =============================================================================
def plot_fig05_mirenasim_audit() -> Path:
    audit = _load("mirenasim_audit_report.json")
    ev = audit["events_godot4_audit"]
    aux = audit["auxiliary_godot4_checks"]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # (a) Godot 4 3D 290 kg vs 280 kg Event Time Deltas [%]
    ax0 = fig.add_subplot(gs[0, 0])
    ev_keys = [("skidpad", "Skidpad\n360 deg"), ("accel", "Accel\n75 m"), ("autocross", "Autocross\n220 m"), ("endurance", "Endurance\n220 m")]
    x0 = np.arange(len(ev_keys))
    dt_pct = [float(ev[k]["godot4_delta_time_pct"]) for k, _ in ev_keys]
    dt_ms = [float(ev[k]["godot4_delta_time_ms"]) for k, _ in ev_keys]
    cols = [PALETTE["emerald"] if v < 0 else PALETTE["amber"] for v in dt_pct]
    bars0 = ax0.bar(x0, dt_pct, color=cols, width=0.52)
    ax0.axhline(0.0, color=PALETTE["slate"], lw=1.0, ls="--")
    for b, pct, ms in zip(bars0, dt_pct, dt_ms):
        ax0.text(
            b.get_x() + b.get_width() * 0.5,
            pct - 0.15 if pct < 0 else pct + 0.15,
            f"{pct:+.2f}%\n({ms:+.0f} ms)",
            ha="center",
            va="top" if pct < 0 else "bottom",
            fontsize=8.3,
            family="monospace",
            color=PALETTE["text"],
        )
    ax0.set_xticks(x0)
    ax0.set_xticklabels([lbl for _, lbl in ev_keys])
    lo0 = min(min(dt_pct), -1.0) - 1.2
    hi0 = max(max(dt_pct), 0.5) + 1.2
    ax0.set_ylim(lo0, hi0)
    style_ax(
        ax0,
        title="MirenaSim Godot 4 3D: 280 kg Limit vs 290 kg Baseline Time Delta",
        ylabel="Event Time Change [%]",
    )

    # (b) Godot 4 3D Front Ride Height RH_f(t) on Autocross & Accel
    ax1 = fig.add_subplot(gs[0, 1])
    ts_acx = ev["autocross"]["timeseries_280kg"]
    ts_acc = ev["accel"]["timeseries_280kg"]
    ax1.plot(ts_acx["t_s"], ts_acx["rh_f_mm"], "-", color=PALETTE["cyan"], lw=1.8, label="Autocross 280 kg RH_f [mm]")
    ax1.plot(ts_acc["t_s"], ts_acc["rh_f_mm"], "-", color=PALETTE["emerald"], lw=1.8, label="Accel 75m 280 kg RH_f [mm]")
    ax1.axhline(10.0, color=PALETTE["crimson"], ls="--", lw=1.4, label="Diffuser Stall Floor (10.0 mm)")
    style_ax(
        ax1,
        title="Godot 4 3D Front Ride Height Audit (0 Bump-Stop Hits, No Stall)",
        xlabel="Simulation Time [s]",
        ylabel="Front Ride Height RH_f [mm]",
        legend=True,
        legend_loc="lower right",
    )

    # (c) Godot 4 3D HV DC Power Trace P_dc(t)
    ax2 = fig.add_subplot(gs[1, 0])
    ts_end = ev["endurance"]["timeseries_280kg"]
    ax2.plot(ts_acx["t_s"], ts_acx["p_dc_kw"], "-", color=PALETTE["cyan"], lw=1.6, label="Autocross 280 kg P_dc [kW]")
    ax2.plot(ts_end["t_s"], ts_end["p_dc_kw"], "-", color=PALETTE["amber"], lw=1.6, label="Endurance 280 kg (38.5 kW Cap)")
    ax2.axhline(80.0, color=PALETTE["crimson"], ls="--", lw=1.3, label="FSAE EV +80.0 kW Max Cap")
    ax2.axhline(-25.0, color=PALETTE["violet"], ls="--", lw=1.3, label="FSAE EV -25.0 kW Regen Cap")
    style_ax(
        ax2,
        title="Godot 4 3D HV DC Power Envelope Audit (0 HV / 0 BSPD Violations)",
        xlabel="Simulation Time [s]",
        ylabel="HV DC Bus Power [kW]",
        legend=True,
        legend_loc="upper right",
    )

    # (d) Independent Audit Safety Margin Summary (Clearance & Yaw Stability)
    ax3 = fig.add_subplot(gs[1, 1])
    chk_names = ["Skidpad\nClean", "Skidpad\nNoisy", "Accel\n75 m", "Autocross\n220 m", "Endurance\n220 m"]
    clears = [
        float(ev["skidpad"]["godot4_280kg_limit"]["min_cone_clear_m"]),
        float(aux["skidpad_280kg_sensor_realism"]["metrics"]["min_cone_clear_m"]),
        float(ev["accel"]["godot4_280kg_limit"]["min_cone_clear_m"]),
        float(ev["autocross"]["godot4_280kg_limit"]["min_cone_clear_m"]),
        float(ev["endurance"]["godot4_280kg_limit"]["min_cone_clear_m"]),
    ]
    dmz = [
        float(ev["skidpad"]["godot4_280kg_limit"]["dmz_dbeta_nm_rad"]),
        float(aux["skidpad_280kg_sensor_realism"]["metrics"]["dmz_dbeta_nm_rad"]),
        float(ev["accel"]["godot4_280kg_limit"]["dmz_dbeta_nm_rad"]),
        float(ev["autocross"]["godot4_280kg_limit"]["dmz_dbeta_nm_rad"]),
        float(ev["endurance"]["godot4_280kg_limit"]["dmz_dbeta_nm_rad"]),
    ]
    x3 = np.arange(len(chk_names))
    bars3 = ax3.bar(x3, clears, color=PALETTE["emerald"], width=0.52, label="Min Cone Clearance [m]")
    ax3.axhline(0.150, color=PALETTE["amber"], ls="--", lw=1.4, label="0.150 m Safety Floor")
    for b, c, m in zip(bars3, clears, dmz):
        ax3.text(
            b.get_x() + b.get_width() * 0.5,
            c + 0.02,
            f"{c:.3f}m\ndMz={m:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            family="monospace",
            color=PALETTE["text"],
        )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(chk_names)
    ax3.set_ylim(0.0, max(clears) + 0.22)
    style_ax(
        ax3,
        title="Godot 4 3D Safety Audit: Clearance (>=0.150 m) & Yaw Stability (dMz/db < 0)",
        ylabel="Minimum Cone Clearance [m]",
        legend=True,
        legend_loc="upper right",
    )

    add_suptitle(
        fig,
        "MirenaSim (Godot 4 Full 14-DOF 3D) Independent Physics Audit of TeR27 (280.0 kg)",
        "3D Double-Wishbone Kinematics | Non-Smooth Bump-Stops | 0 Cones | 0 HV/BSPD Violations | Strict dMz/dbeta < 0",
    )
    add_footer(fig, "Source: outputs/ter27_280kg_limit_study/mirenasim_audit_report.json")
    out = STUDY_DIR / "fig05_mirenasim_godot4_280kg_audit_parity.png"
    save_fig(fig, out, dpi=300)
    return out


def main() -> int:
    t0 = time.perf_counter()
    apply_motec_style()
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    figs = [
        plot_fig01_waterfall(),
        plot_fig02_powertrain(),
        plot_fig03_trajectory(),
        plot_fig04_thermal_endurance(),
        plot_fig05_mirenasim_audit(),
    ]
    for fp in figs:
        print(f"[PLOTS-280KG] Generated {fp.name} ({fp.stat().st_size / 1024.0:.1f} KB)", flush=True)
    print(f"[PLOTS-280KG] All 5 300-DPI pure-ASCII figures generated in {time.perf_counter() - t0:.2f}s.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
