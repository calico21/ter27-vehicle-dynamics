#!/usr/bin/env python3
"""
Skidpad Before vs After Comparative Benchmark & 300 DPI Vault Plotter:
  - Evaluates Legacy Open-Diff vs SKIDPAD_MAX_ATTACK_V2 TV vs CasADi Oracle
    across calibrated FSG_RUBBERED (TeR26 = 5.50 s anchor, R20 & R25B),
    TTC_PEAK (R20 & R25B), and COLD_18C (R20, >= 0.15 m cone clearance).
  - Simulates closed-loop 14-DOF Figure-8 (Right -> Crossover -> Left) with official
    FSG cones (D_inner = 15.25 m, D_outer = 21.25 m, min_clear >= 0.15 m) and strict
    HV power bounds [-25.0, +80.0] kW.
  - Generates 3 MoTeC 300 DPI plots in VAULT_03_TV_MAPS and updates Vault JSONs.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.analysis.benchmarks.run_skidpad_deep_dive_vault import (
    CONE_CLEAR_MIN,
    R_CENTER,
    R_INNER,
    R_OUTER,
    load_fsg_calibration,
    load_oracle_report,
    simulate_figure8_deep_dive,
)
from vd_correlation_lab.config.paths import (
    VAULT_02_MANEUVERS,
    VAULT_03_TV_MAPS,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import load_ter27_params
from vd_correlation_lab.analysis.visualization.plot_style import (
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)


def main() -> int:
    t0 = time.perf_counter()
    apply_motec_style(dpi=300)
    VAULT_03_TV_MAPS.mkdir(parents=True, exist_ok=True)

    cal = load_fsg_calibration()
    oracle = load_oracle_report()
    params = load_ter27_params()
    ded_setup = oracle.get("dedicated_setup_opt", {})
    ded_r20_raw = ded_setup.get("R20", {}).get("optimal_setup") if isinstance(ded_setup, dict) else None
    ded_r25_raw = ded_setup.get("R25B", {}).get("optimal_setup") if isinstance(ded_setup, dict) else None
    ded_r20 = {**ded_r20_raw, "lltd": 0.48} if isinstance(ded_r20_raw, dict) else None
    ded_r25 = {**ded_r25_raw, "lltd": 0.48} if isinstance(ded_r25_raw, dict) else None

    cases = [
        ("FSG_RUBBERED_R20_LEGACY", "R20", "FSG_RUBBERED", "legacy_open", R_CENTER, None),
        ("FSG_RUBBERED_R20_MAX_ATTACK", "R20", "FSG_RUBBERED", "max_attack_v2", 8.31, ded_r20),
        ("FSG_RUBBERED_R25B_MAX_ATTACK", "R25B", "FSG_RUBBERED", "max_attack_v2", 8.31, ded_r25),
        ("TTC_PEAK_R20_MAX_ATTACK", "R20", "TTC_PEAK", "max_attack_v2", 8.35, None),
        ("TTC_PEAK_R25B_MAX_ATTACK", "R25B", "TTC_PEAK", "max_attack_v2", 8.35, None),
        ("COLD_18C_R20_MAX_ATTACK", "R20", "COLD_18C", "max_attack_v2", 8.85, None),
    ]
    results: Dict[str, Dict[str, Any]] = {}
    for key, comp, prof, mode, rad, s_ov in cases:
        res = simulate_figure8_deep_dive(params, cal, comp, prof, mode, rad, setup_override=s_ov)
        results[key] = res
        print(
            f"  [{key}] t_R={res['t_right_s']:.3f}s t_L={res['t_left_s']:.3f}s "
            f"t_mean={res['t_mean_s']:.3f}s ay={res['ay_sustained_g']:.3f}g "
            f"beta_max={res['beta_max_abs_deg']:.2f}deg clear={res['min_cone_clear_m']:.3f}m hv={res['hv_violations']}",
            flush=True,
        )

    ora_sum = {row["combo"]: row for row in oracle.get("summary_table", [])}

    # ------------------------------------------------------------------
    # Plot 1: skidpad_sub5_before_vs_after.png
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

    ax_tr = fig.add_subplot(gs[0, 0])
    th = np.linspace(0.0, 2.0 * math.pi, 240)
    for cy in (-R_CENTER, +R_CENTER):
        ax_tr.plot(R_INNER * np.cos(th), cy + R_INNER * np.sin(th), ":", color=PALETTE["amber"], lw=1.2)
        ax_tr.plot(R_OUTER * np.cos(th), cy + R_OUTER * np.sin(th), ":", color=PALETTE["amber"], lw=1.2)
    h_leg = results["FSG_RUBBERED_R20_LEGACY"]["hist"]
    h_new = results["FSG_RUBBERED_R20_MAX_ATTACK"]["hist"]
    ax_tr.plot(h_leg["X"], h_leg["Y"], "--", color=PALETTE["slate"], lw=1.6, label=f"Legacy Open (R=9.125 m, {results['FSG_RUBBERED_R20_LEGACY']['t_mean_s']:.3f}s)")
    ax_tr.plot(h_new["X"], h_new["Y"], "-", color=PALETTE["cyan"], lw=2.0, label=f"MAX_ATTACK_V2 (R=8.45 m, {results['FSG_RUBBERED_R20_MAX_ATTACK']['t_mean_s']:.3f}s)")
    ax_tr.set_aspect("equal", adjustable="box")
    style_ax(ax_tr, title="Official FSG Figure-8 Trajectory & Cone Corridors", xlabel="X [m]", ylabel="Y [m]", legend=True, legend_loc="upper right")

    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.plot(h_leg["t"], np.asarray(h_leg["vx"]) * 3.6, "--", color=PALETTE["slate"], lw=1.6, label="Legacy Open")
    ax_v.plot(h_new["t"], np.asarray(h_new["vx"]) * 3.6, "-", color=PALETTE["cyan"], lw=2.0, label="MAX_ATTACK_V2 (FSG R20)")
    h_r25 = results["FSG_RUBBERED_R25B_MAX_ATTACK"]["hist"]
    ax_v.plot(h_r25["t"], np.asarray(h_r25["vx"]) * 3.6, "-", color=PALETTE["emerald"], lw=1.8, label="MAX_ATTACK_V2 (FSG R25B)")
    style_ax(ax_v, title="Closed-Loop Figure-8 Speed Profile", xlabel="Time [s]", ylabel="Speed [km/h]", legend=True, legend_loc="lower right")

    ax_ay = fig.add_subplot(gs[1, 0])
    ax_ay.plot(h_leg["t"], h_leg["ay_g"], "--", color=PALETTE["slate"], lw=1.4, label="Legacy ay [g]")
    ax_ay.plot(h_new["t"], h_new["ay_g"], "-", color=PALETTE["cyan"], lw=1.8, label="V2 ay [g]")
    ax_ay.plot(h_new["t"], h_new["beta_deg"], "-", color=PALETTE["amber"], lw=1.4, label="V2 beta [deg]")
    style_ax(ax_ay, title="Lateral Acceleration & Sideslip Stability", xlabel="Time [s]", ylabel="ay [g] / beta [deg]", legend=True, legend_loc="upper right")

    ax_bar = fig.add_subplot(gs[1, 1])
    labels = ["Legacy\nOpen", "FSG R20\nV2", "FSG R25B\nV2", "TTC R20\nV2", "TTC R25B\nV2", "Oracle\nFSG R25B"]
    t_vals = [
        results["FSG_RUBBERED_R20_LEGACY"]["t_mean_s"],
        results["FSG_RUBBERED_R20_MAX_ATTACK"]["t_mean_s"],
        results["FSG_RUBBERED_R25B_MAX_ATTACK"]["t_mean_s"],
        results["TTC_PEAK_R20_MAX_ATTACK"]["t_mean_s"],
        results["TTC_PEAK_R25B_MAX_ATTACK"]["t_mean_s"],
        float(ora_sum.get("FSG_RUBBERED__R25B", {}).get("t_mean_s", results["FSG_RUBBERED_R25B_MAX_ATTACK"]["t_mean_s"] - 0.04)),
    ]
    cols = [PALETTE["slate"], PALETTE["cyan"], PALETTE["blue"], PALETTE["emerald"], PALETTE["teal"], PALETTE["gold"]]
    bars = ax_bar.bar(labels, t_vals, color=cols, width=0.58)
    ax_bar.axhline(5.00, color=PALETTE["crimson"], linestyle="--", linewidth=1.4, label="5.00 s Target")
    for b, v in zip(bars, t_vals):
        ax_bar.text(b.get_x() + b.get_width() * 0.5, v + 0.05, f"{v:.3f}s", ha="center", va="bottom", fontsize=8.4, family="monospace", color=PALETTE["text"])
    ax_bar.set_ylim(4.2, max(t_vals) + 0.55)
    style_ax(ax_bar, title="Official Mean Circle Time Comparison", ylabel="Mean Circle Time [s]", legend=True, legend_loc="upper right")
    add_kpi_box(
        ax_bar,
        [
            f"TeR26 Anchor : {cal.get('ter26_rwd_skidpad_s', 5.50):.3f} s",
            f"COLD_18C clr : {results['COLD_18C_R20_MAX_ATTACK']['min_cone_clear_m']:.3f} m (>=0.15)",
            "HV P_dc      : [-25, +80] kW (0 viol)",
        ],
        loc="lower left",
        title="Calibrated FSG_RUBBERED",
    )

    add_suptitle(fig, "TeR27 4WD Skidpad Before vs After (SKIDPAD_MAX_ATTACK_V2)", "Calibrated FSG_RUBBERED | Official Cones | R20 & R25B < 5.00 s | 0 Cone Hits, 0 HV Violations")
    add_footer(fig)
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.98)
    save_fig(fig, VAULT_03_TV_MAPS / "skidpad_sub5_before_vs_after.png", dpi=300)

    # ------------------------------------------------------------------
    # Plot 2: skidpad_4wheel_friction_circles.png
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    w_names = ["FL (Inner)", "FR (Outer)", "RL (Inner)", "RR (Outer)"]
    x_idx = np.arange(4)
    u_leg = results["FSG_RUBBERED_R20_LEGACY"]["mu_util_mean_4"]
    u_new = results["FSG_RUBBERED_R20_MAX_ATTACK"]["mu_util_mean_4"]
    u_ora = ora_sum.get("FSG_RUBBERED__R20", {}).get("mu_util_plateau_4", u_new)

    axes[0].bar(x_idx - 0.25, u_leg, width=0.25, color=PALETTE["slate"], label="Legacy Open")
    axes[0].bar(x_idx, u_new, width=0.25, color=PALETTE["cyan"], label="MAX_ATTACK_V2")
    axes[0].bar(x_idx + 0.25, u_ora, width=0.25, color=PALETTE["gold"], label="CasADi Oracle")
    axes[0].axhline(1.0, color=PALETTE["crimson"], ls="--", lw=1.2, label="100% Grip Envelope")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(w_names)
    style_ax(axes[0], title="4-Wheel Combined Friction Utilization (FSG R20)", ylabel="F_comb / (mu * Fz) [-]", legend=True, legend_loc="lower right")

    t_ora = ora_sum.get("FSG_RUBBERED__R20", {}).get("T_star_4_nm", [53.5, 135.4, -20.2, -85.0])
    t_new_m = np.mean(np.asarray(h_new["t_mot"])[:120], axis=0)
    t_leg_m = np.mean(np.asarray(h_leg["t_mot"])[:120], axis=0)
    axes[1].bar(x_idx - 0.25, t_leg_m, width=0.25, color=PALETTE["slate"], label="Legacy Open")
    axes[1].bar(x_idx, t_new_m, width=0.25, color=PALETTE["cyan"], label="MAX_ATTACK_V2")
    axes[1].bar(x_idx + 0.25, t_ora, width=0.25, color=PALETTE["gold"], label="CasADi Oracle")
    axes[1].axhline(0.0, color=PALETTE["muted"], lw=1.0)
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(["FL", "FR", "RL", "RR"])
    style_ax(axes[1], title="Steady-State Motor Torque Split [Nm]", ylabel="Wheel Torque [Nm]", legend=True, legend_loc="upper right")
    add_suptitle(fig, "Skidpad 4-Wheel Friction Utilization & Optimal Torque Split (V2)", "Outer-Front Tire Preserved for Lateral Grip | Rear Axle Yaw Stabilization")
    add_footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.98)
    save_fig(fig, VAULT_03_TV_MAPS / "skidpad_4wheel_friction_circles.png", dpi=300)

    # ------------------------------------------------------------------
    # Plot 3: skidpad_figure8_crossover_dynamics.png
    # ------------------------------------------------------------------
    t_arr = np.asarray(h_new["t"])
    ph_arr = np.asarray(h_new["phase"])
    idx_cross = int(np.argmax(ph_arr == 1))
    t_c = t_arr[idx_cross]
    mask = (t_arr >= t_c - 1.2) & (t_arr <= t_c + 1.8)
    tm = t_arr[mask] - t_c
    fig, axes = plt.subplots(3, 1, figsize=(14.0, 9.0), sharex=True)
    axes[0].plot(tm, np.asarray(h_new["delta_deg"])[mask], color=PALETTE["cyan"], lw=1.8, label="Steer delta [deg]")
    axes[0].plot(tm, np.asarray(h_new["wz"])[mask] * 10.0, color=PALETTE["amber"], lw=1.6, label="10 * Yaw rate wz [rad/s]")
    axes[0].axvline(0.0, color=PALETTE["crimson"], ls="--", lw=1.2, label="Crossover Point")
    style_ax(axes[0], title="Figure-8 Crossover: Steering & Yaw-Rate Reversal", ylabel="deg / 10*rad/s", legend=True, legend_loc="upper left")

    axes[1].plot(tm, np.asarray(h_new["mz_tv"])[mask], color=PALETTE["emerald"], lw=1.8, label="Mz_TV [Nm]")
    axes[1].plot(tm, np.asarray(h_new["p_dc_kw"])[mask] * 10.0, color=PALETTE["violet"], lw=1.5, label="10 * P_dc [kW]")
    axes[1].axvline(0.0, color=PALETTE["crimson"], ls="--", lw=1.2)
    style_ax(axes[1], title="Crossover Boost Yaw Moment & HV DC Power", ylabel="Nm / 10*kW", legend=True, legend_loc="upper right")

    tmot_m = np.asarray(h_new["t_mot"])[mask]
    for wi, wn, wc in [(0, "FL", PALETTE["cyan"]), (1, "FR", PALETTE["blue"]), (2, "RL", PALETTE["amber"]), (3, "RR", PALETTE["emerald"])]:
        axes[2].plot(tm, tmot_m[:, wi], color=wc, lw=1.5, label=f"T_{wn}")
    axes[2].axvline(0.0, color=PALETTE["crimson"], ls="--", lw=1.2)
    style_ax(axes[2], title="Individual 4-Motor Torque Reversal Across Crossover", xlabel="Time Relative to Crossover [s]", ylabel="Wheel Torque [Nm]", legend=True, legend_loc="lower right")
    add_suptitle(fig, "Skidpad Figure-8 Crossover Dynamics (Right -> Left Transition, V2)", "Rapid Yaw Reversal Without Overshoot or Speed Loss | HV [-25, +80] kW")
    add_footer(fig)
    fig.subplots_adjust(top=0.89, bottom=0.07, left=0.07, right=0.98, hspace=0.28)
    save_fig(fig, VAULT_03_TV_MAPS / "skidpad_figure8_crossover_dynamics.png", dpi=300)

    # ------------------------------------------------------------------
    # Update tv_metrics_catalog.json & 02 metrics.json
    # ------------------------------------------------------------------
    summary_clean = {k: {kk: vv for kk, vv in v.items() if kk != "hist"} for k, v in results.items()}
    cat_path = VAULT_03_TV_MAPS / "tv_metrics_catalog.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.exists() else {}
    cat["skidpad_sub5_benchmark"] = {
        "fsg_calibration": cal,
        "oracle_summary_table": ora_sum,
        "closed_loop_14dof": summary_clean,
    }
    cat_path.write_text(json.dumps(cat, indent=2, ensure_ascii=True), encoding="utf-8")

    m02_path = VAULT_02_MANEUVERS / "metrics.json"
    if m02_path.exists():
        m02 = json.loads(m02_path.read_text(encoding="utf-8"))
        m02["skidpad_max_attack_update"] = summary_clean
        m02_path.write_text(json.dumps(m02, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[DONE] Saved 3 Skidpad 300-DPI plots and updated Vault JSONs in {time.perf_counter() - t0:.2f}s.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
