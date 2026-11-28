#!/usr/bin/env python3
"""Generate all 8 clean publication-grade 300-DPI pure-white (#FFFFFF) figures.

All titles, labels, legends, annotations, and code are 100% pure ASCII.
No dark backgrounds or MoTeC-style themes are used anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEEP_VAL_DIR = REPO_ROOT / "outputs" / "differentiable_deep_validation"
LIMIT_280_DIR = REPO_ROOT / "outputs" / "ter27_280kg_limit_study"
FINAL_VAL_DIR = REPO_ROOT / "outputs" / "final_release_validation"
FIGURES_DIR = FINAL_VAL_DIR / "figures"

COLORS: Dict[str, str] = {
    "blue": "#1D4ED8",
    "emerald": "#059669",
    "amber": "#D97706",
    "red": "#DC2626",
    "slate": "#4B5563",
    "dark": "#111827",
    "grid": "#E5E7EB",
    "white": "#FFFFFF",
    "light_blue": "#DBEAFE",
    "light_emerald": "#D1FAE5",
    "light_amber": "#FEF3C7",
    "light_red": "#FEE2E2",
    "light_slate": "#F3F4F6",
}

ALL_FIGURES: List[str] = [
    "fig01_ter26_empirical_validation.png",
    "fig02_ter27_280kg_lap_times_and_attribution.png",
    "fig03_suspension_4x3_matrix_heatmap.png",
    "fig04_grip_and_weather_envelope.png",
    "fig05_tire_thermal_window_sensitivity.png",
    "fig06_manufacturing_mass_and_aero_sensitivity.png",
    "fig07_tube_mpc_robustness_all_faults.png",
    "fig08_endurance_22km_energy_and_thermal_budget.png",
]


def apply_clean_scientific_style() -> None:
    """Apply strict pure-white (#FFFFFF) scientific publication rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "savefig.transparent": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelcolor": "#111827",
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#D1D5DB",
            "legend.fontsize": 8.3,
            "text.color": "#111827",
        }
    )


def _style_ax(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#111827", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.2, color="#111827")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.2, color="#111827")


def _save_clean_fig(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(
        out_path,
        dpi=300,
        facecolor="#FFFFFF",
        edgecolor="#FFFFFF",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)
    return out_path


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_clean_figure(path: Path) -> Dict[str, Any]:
    """Programmatically verify existence, 300 DPI metadata, and #FFFFFF corners."""
    if not path.exists():
        return {"path": str(path), "exists": False, "passed": False, "error": "missing"}
    with Image.open(path) as img:
        dpi = img.info.get("dpi", (0, 0))
        rgb = img.convert("RGB")
        w, h = rgb.size
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((w - 1, 0)),
            rgb.getpixel((0, h - 1)),
            rgb.getpixel((w - 1, h - 1)),
        ]
    dpi_ok = abs(float(dpi[0]) - 300.0) <= 1.0 and abs(float(dpi[1]) - 300.0) <= 1.0
    white_ok = all(px == (255, 255, 255) for px in corners)
    return {
        "path": str(path),
        "exists": True,
        "size_px": [w, h],
        "dpi": [round(float(dpi[0]), 2), round(float(dpi[1]), 2)],
        "corners_rgb": [list(c) for c in corners],
        "dpi_ok": dpi_ok,
        "white_bg_ok": white_ok,
        "passed": bool(dpi_ok and white_ok),
    }


# =============================================================================
# FIG 01: TeR26 Empirical Validation (Before vs After Adjoint Calibration)
# =============================================================================
def plot_fig01_ter26_empirical_validation() -> Path:
    apply_clean_scientific_style()
    rep = _load_json(DEEP_VAL_DIR / "ter26_adjoint_calibration_report.json")
    win = rep["plot_timeseries_windows"]["held_out_val"]
    summary = rep["before_vs_after_summary"]["all_5_runs"]

    t = np.asarray(win["t_s"], dtype=float)
    t = t - t[0]

    fig = plt.figure(figsize=(14.5, 9.2), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

    # Panel (a): Yaw rate omega_z on held-out validation run
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, win["wz_real_rads"], color=COLORS["dark"], lw=1.6, label="Real CAN Log")
    ax0.plot(t, win["wz_before_rads"], color=COLORS["slate"], lw=1.1, ls="--", alpha=0.85, label="Before Calibration")
    ax0.plot(t, win["wz_after_rads"], color=COLORS["blue"], lw=1.4, label="After Adjoint SysID")
    _style_ax(
        ax0,
        title=f"(a) Held-Out Yaw Rate omega_z ({win['run_id']})",
        xlabel="Window Time [s]",
        ylabel="omega_z [rad/s]",
    )
    ax0.legend(loc="upper right")

    # Panel (b): Lateral acceleration ay on held-out validation run
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, win["ay_real_ms2"], color=COLORS["dark"], lw=1.6, label="Real CAN Log")
    ax1.plot(t, win["ay_before_ms2"], color=COLORS["slate"], lw=1.1, ls="--", alpha=0.85, label="Before Calibration")
    ax1.plot(t, win["ay_after_ms2"], color=COLORS["emerald"], lw=1.4, label="After Adjoint SysID")
    _style_ax(
        ax1,
        title=f"(b) Held-Out Lateral Accel ay ({win['run_id']})",
        xlabel="Window Time [s]",
        ylabel="ay [m/s^2]",
    )
    ax1.legend(loc="upper right")

    # Panel (c): Longitudinal velocity vx & acceleration ax RMSE reduction
    ax2 = fig.add_subplot(gs[1, 0])
    channels = ["wz_rads", "ay_ms2", "ax_ms2", "vx_ms"]
    ch_labels = ["omega_z\n[rad/s]", "ay\n[m/s^2]", "ax\n[m/s^2]", "vx\n[m/s]"]
    r2_before = [summary[c]["before_r2"] for c in channels]
    r2_after = [summary[c]["after_r2"] for c in channels]
    x = np.arange(len(channels))
    w = 0.34
    b1 = ax2.bar(x - w / 2, r2_before, width=w, color=COLORS["slate"], alpha=0.75, label="Before SysID")
    b2 = ax2.bar(x + w / 2, r2_after, width=w, color=COLORS["blue"], label="After Adjoint SysID")
    for bar, val in zip(b1, r2_before):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.8, color=COLORS["slate"])
    for bar, val in zip(b2, r2_after):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["blue"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(ch_labels)
    ax2.set_ylim(0.75, 1.04)
    _style_ax(
        ax2,
        title="(c) R^2 Correlation Across All 5 TeR26 Track Runs",
        ylabel="Coefficient of Determination R^2 [-]",
    )
    ax2.legend(loc="lower right")

    # Panel (d): RMSE reduction percentage across channels
    ax3 = fig.add_subplot(gs[1, 1])
    rmse_red = [summary[c]["rmse_reduction_pct"] for c in channels]
    bar_colors = [COLORS["blue"], COLORS["emerald"], COLORS["amber"], COLORS["blue"]]
    bars = ax3.bar(x, rmse_red, width=0.5, color=bar_colors)
    for bar, val in zip(bars, rmse_red):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"-{val:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111827")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ch_labels)
    ax3.set_ylim(0, max(rmse_red) * 1.22)
    _style_ax(
        ax3,
        title="(d) RMSE Reduction After Adjoint Calibration (All 5 Runs)",
        ylabel="RMSE Reduction [%]",
    )

    fig.suptitle(
        "TeR26 Empirical CAN Telemetry Validation: Before vs After Adjoint SysID (R20, COLD_TEST_OCT mu=0.47)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig01_ter26_empirical_validation.png")


# =============================================================================
# FIG 02: TeR27 280 kg Lap Times & Physical Attribution Waterfall
# =============================================================================
def plot_fig02_ter27_280kg_lap_times_and_attribution() -> Path:
    apply_clean_scientific_style()
    summary = _load_json(LIMIT_280_DIR / "SUMMARY_REPORT.json")
    wf = summary["physical_attribution_waterfall"]
    exec_res = summary["executive_results"]

    fig = plt.figure(figsize=(15.0, 9.2), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.25)

    def _draw_waterfall(ax: plt.Axes, labels: List[str], deltas: List[float], total: float, title: str) -> None:
        x = np.arange(len(labels) + 1)
        vals = list(deltas) + [total]
        cols = [COLORS["blue"], COLORS["emerald"], COLORS["amber"], COLORS["red"]][: len(deltas)] + [COLORS["slate"]]
        bots: List[float] = []
        cur = 0.0
        for d in deltas:
            bots.append(cur)
            cur += d
        bots.append(0.0)
        bars = ax.bar(x, vals, bottom=bots, width=0.54, color=cols, edgecolor="#FFFFFF", lw=0.6)
        ax.axhline(0.0, color="#374151", lw=0.9, ls="--")
        lo = min(min(bots[i] + vals[i] for i in range(len(vals))), 0.0)
        hi = max(max(bots[i] + vals[i] for i in range(len(vals))), 0.0)
        span = max(hi - lo, 10.0)
        for b, v, bot in zip(bars, vals, bots):
            y_pos = bot + v
            va = "top" if v < 0 else "bottom"
            off = -0.03 * span if v < 0 else 0.03 * span
            ax.text(
                b.get_x() + b.get_width() / 2,
                y_pos + off,
                f"{v:+.1f}",
                ha="center",
                va=va,
                fontsize=8.2,
                fontweight="bold",
                color="#111827",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels + ["TOTAL"])
        ax.set_ylim(lo - 0.18 * span, hi + 0.14 * span)
        _style_ax(ax, title=title, ylabel="Delta Time [ms] (negative = faster)")

    sk = wf["skidpad_ms"]
    _draw_waterfall(
        fig.add_subplot(gs[0, 0]),
        ["-10 kg\nMass", "Chassis &\nAero", "Free-Traj\n+ 4WD TV"],
        [float(sk["mass_minus_10kg_ms"]), float(sk["chassis_aero_platform_ms"]), float(sk["free_trajectory_and_4wd_tv_ms"])],
        float(sk["total_delta_ms"]),
        f"(a) Skidpad 360 deg: {sk['baseline_290kg_s']:.3f}s -> {sk['limit_280kg_s']:.3f}s",
    )

    ac = wf["acceleration_75m_ms"]
    _draw_waterfall(
        fig.add_subplot(gs[0, 1]),
        ["-10 kg\nMass", "Powertrain\nLoss Rec.", "Launch Slip\n+ 4WD TV"],
        [float(ac["mass_minus_10kg_ms"]), float(ac["powertrain_loss_recovery_ms"]), float(ac["launch_slip_and_4wd_tv_ms"])],
        float(ac["total_delta_ms"]),
        f"(b) Acceleration 75 m: {ac['baseline_290kg_s']:.3f}s -> {ac['limit_280kg_s']:.3f}s",
    )

    ax_wf = wf["autocross_1184m_ms"]
    _draw_waterfall(
        fig.add_subplot(gs[1, 0]),
        ["-10 kg\nMass", "Powertrain", "Chassis &\nAero", "Free-Spline\n+ 4WD TV"],
        [
            float(ax_wf["mass_minus_10kg_ms"]),
            float(ax_wf["powertrain_loss_recovery_ms"]),
            float(ax_wf["chassis_aero_platform_ms"]),
            float(ax_wf["free_trajectory_and_4wd_tv_ms"]),
        ],
        float(ax_wf["total_delta_ms"]),
        f"(c) Autocross 1.184 km: {ax_wf['baseline_290kg_s']:.2f}s -> {ax_wf['limit_280kg_s']:.2f}s",
    )

    # Panel (d): CasADi vs Godot 4 Sim2Sim Parity Summary (normalized % delta 280 vs 290)
    ax3 = fig.add_subplot(gs[1, 1])
    events = ["Skidpad", "Accel 75m", "Autocross\n(Sector/Lap)"]
    casadi_pct = [
        100.0 * (exec_res["skidpad"]["casadi_280kg_limit_s"] - exec_res["skidpad"]["casadi_290kg_s"]) / exec_res["skidpad"]["casadi_290kg_s"],
        100.0 * (exec_res["acceleration_75m"]["casadi_280kg_limit_s"] - exec_res["acceleration_75m"]["casadi_290kg_s"]) / exec_res["acceleration_75m"]["casadi_290kg_s"],
        100.0 * (exec_res["autocross_1184m"]["casadi_280kg_limit_s"] - exec_res["autocross_1184m"]["casadi_290kg_s"]) / exec_res["autocross_1184m"]["casadi_290kg_s"],
    ]
    godot_pct = [
        100.0 * (exec_res["skidpad"]["godot4_280kg_limit_s"] - exec_res["skidpad"]["godot4_290kg_s"]) / exec_res["skidpad"]["godot4_290kg_s"],
        100.0 * (exec_res["acceleration_75m"]["godot4_280kg_limit_s"] - exec_res["acceleration_75m"]["godot4_290kg_s"]) / exec_res["acceleration_75m"]["godot4_290kg_s"],
        100.0 * (exec_res["autocross_1184m"]["godot4_sector_280kg_limit_s"] - exec_res["autocross_1184m"]["godot4_sector_290kg_s"]) / exec_res["autocross_1184m"]["godot4_sector_290kg_s"],
    ]
    x = np.arange(len(events))
    w = 0.34
    b1 = ax3.bar(x - w / 2, casadi_pct, width=w, color=COLORS["blue"], label="CasADi 14-DOF")
    b2 = ax3.bar(x + w / 2, godot_pct, width=w, color=COLORS["emerald"], label="Godot 4 MirenaSim 14-DOF")
    ax3.axhline(0.0, color="#374151", lw=0.9, ls="--")
    for b, v in zip(b1, casadi_pct):
        ax3.text(b.get_x() + b.get_width() / 2, v - 0.12, f"{v:.2f}%", ha="center", va="top", fontsize=8.0, fontweight="bold", color=COLORS["blue"])
    for b, v in zip(b2, godot_pct):
        ax3.text(b.get_x() + b.get_width() / 2, v - 0.12, f"{v:.2f}%", ha="center", va="top", fontsize=8.0, fontweight="bold", color=COLORS["emerald"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(events)
    ax3.set_ylim(min(min(casadi_pct), min(godot_pct)) * 1.28, 0.4)
    _style_ax(ax3, title="(d) 280 kg vs 290 kg Improvement: CasADi vs Godot 4", ylabel="Relative Time Change [%]")
    ax3.legend(loc="lower left")

    fig.suptitle(
        "TeR27 (280.0 kg) Limit Study: Lap Time Gains & Physical Attribution (FSG_RUBBERED mu=0.57)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig02_ter27_280kg_lap_times_and_attribution.png")


# =============================================================================
# FIG 08: Endurance 22 km Energy & Thermal Budget
# =============================================================================
def plot_fig08_endurance_22km_energy_and_thermal_budget() -> Path:
    apply_clean_scientific_style()
    rep = _load_json(LIMIT_280_DIR / "thermal_and_endurance_report.json")
    part_b = rep["part_b_endurance_22km"]
    pareto = [p for p in part_b["pareto_frontier_280kg"] if p.get("feasible", False)]
    opt280 = part_b["optimal_combined_280kg"]
    opt290 = part_b["optimal_combined_290kg"]

    fig = plt.figure(figsize=(14.5, 8.8), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel (a): 22 km Time vs Net Energy Pareto Frontier
    ax0 = fig.add_subplot(gs[0, 0])
    e_kwh = [p["e_net_22km_kwh"] for p in pareto]
    t_22 = [p["t_22km_s"] for p in pareto]
    ax0.scatter(e_kwh, t_22, c=COLORS["blue"], s=36, alpha=0.85, label="Feasible 280 kg Strategies")
    ax0.scatter(
        [opt280["score"]["e_net_22km_kwh"]],
        [opt280["score"]["t_22km_s"]],
        c=COLORS["emerald"],
        s=110,
        marker="*",
        zorder=5,
        label=f"Optimal 280 kg ({opt280['score']['e_net_22km_kwh']:.3f} kWh, {opt280['score']['t_22km_s']:.1f} s)",
    )
    ax0.scatter(
        [opt290["score"]["e_net_22km_kwh"]],
        [opt290["score"]["t_22km_s"]],
        c=COLORS["amber"],
        s=85,
        marker="D",
        zorder=5,
        label=f"Optimal 290 kg ({opt290['score']['e_net_22km_kwh']:.3f} kWh, {opt290['score']['t_22km_s']:.1f} s)",
    )
    ax0.axvline(3.50, color=COLORS["red"], lw=1.3, ls="--", label="3.50 kWh Budget Limit")
    _style_ax(ax0, title="(a) Endurance 22 km Time vs Net Energy Pareto", xlabel="Net 22 km Energy [kWh]", ylabel="22 km Total Time [s]")
    ax0.legend(loc="upper right", fontsize=7.8)

    # Panel (b): Energy Breakdown (Positive, Regen, Net, Loss)
    ax1 = fig.add_subplot(gs[0, 1])
    s280 = opt280["single_lap_metrics"]
    cats = ["Positive\nTraction", "Regen\nRecovered", "Net\nConsumed", "Powertrain\nLosses"]
    n_laps = 22000.0 / float(rep["metadata"]["track_length_m"])
    vals_280 = [
        s280["e_pos_lap_kj"] * n_laps / 3600.0,
        s280["e_regen_lap_kj"] * n_laps / 3600.0,
        s280["e_net_22km_kwh"],
        s280["e_loss_22km_kwh"],
    ]
    cols = [COLORS["blue"], COLORS["emerald"], COLORS["amber"], COLORS["red"]]
    x = np.arange(len(cats))
    bars = ax1.bar(x, vals_280, width=0.52, color=cols)
    for b, v in zip(bars, vals_280):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.3f} kWh", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax1.axhline(3.50, color=COLORS["red"], lw=1.1, ls="--", label="3.50 kWh Net Cap")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats)
    ax1.set_ylim(0, max(vals_280) * 1.2)
    _style_ax(ax1, title="(b) 22 km Energy Budget Breakdown (280 kg Optimal)", ylabel="Energy [kWh]")
    ax1.legend(loc="upper right")

    # Panel (c): Steady-State Thermal Margin (Motor, Inverter, Tire vs Derate/Limit)
    ax2 = fig.add_subplot(gs[1, 0])
    subsystems = ["Motor\nStator", "Inverter\nMOSFET", "Tire\nCarcass"]
    ss_temps = [s280["t_motor_ss_c"], s280["t_inv_ss_c"], s280["t_tire_ss_mean_c"]]
    limits = [
        part_b["thermal_models"]["motor"]["t_derate_start_c"],
        part_b["thermal_models"]["inverter"]["t_derate_start_c"],
        part_b["thermal_models"]["tire"]["t_opt_high_c"],
    ]
    x_t = np.arange(len(subsystems))
    w_t = 0.34
    b_ss = ax2.bar(x_t - w_t / 2, ss_temps, width=w_t, color=COLORS["emerald"], label="280 kg Steady-State [deg C]")
    b_lim = ax2.bar(x_t + w_t / 2, limits, width=w_t, color=COLORS["slate"], alpha=0.65, label="Derate / Upper Limit [deg C]")
    for b, v in zip(b_ss, ss_temps):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f} C", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=COLORS["emerald"])
    for b, v in zip(b_lim, limits):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f} C", ha="center", va="bottom", fontsize=8.2, color=COLORS["slate"])
    ax2.set_xticks(x_t)
    ax2.set_xticklabels(subsystems)
    ax2.set_ylim(0, max(limits) * 1.22)
    _style_ax(ax2, title="(c) 22 km Steady-State Thermal Headroom", ylabel="Temperature [deg C]")
    ax2.legend(loc="upper left")

    # Panel (d): FSG Scoring Comparison (290 kg vs 280 kg)
    ax3 = fig.add_subplot(gs[1, 1])
    sc_290 = opt290["score"]
    sc_280 = opt280["score"]
    score_cats = ["Endurance\nPoints (/250)", "Efficiency\nPoints (/100)", "Combined\nPoints (/350)"]
    v290 = [sc_290["pts_endurance_rel"], sc_290["pts_efficiency_rel"], sc_290["pts_endurance_plus_efficiency"]]
    v280 = [sc_280["pts_endurance_rel"], sc_280["pts_efficiency_rel"], sc_280["pts_endurance_plus_efficiency"]]
    x_s = np.arange(len(score_cats))
    b_90 = ax3.bar(x_s - w_t / 2, v290, width=w_t, color=COLORS["amber"], label="290 kg Optimal")
    b_80 = ax3.bar(x_s + w_t / 2, v280, width=w_t, color=COLORS["blue"], label="280 kg Optimal")
    for b, v in zip(b_90, v290):
        ax3.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, color=COLORS["amber"], fontweight="bold")
    for b, v in zip(b_80, v280):
        ax3.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, color=COLORS["blue"], fontweight="bold")
    ax3.set_xticks(x_s)
    ax3.set_xticklabels(score_cats)
    ax3.set_ylim(0, 390)
    _style_ax(ax3, title=f"(d) FSG Endurance + Efficiency Score (+{sc_280['pts_endurance_plus_efficiency'] - sc_290['pts_endurance_plus_efficiency']:.2f} pts)", ylabel="FSG Points [-]")
    ax3.legend(loc="upper left")

    fig.suptitle(
        "TeR27 (280.0 kg) 22 km Endurance Energy Management & Thermal Budget (3.50 kWh Cap, [-25, +80] kW)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig08_endurance_22km_energy_and_thermal_budget.png")


def _load_matrix_rows(axis: str) -> List[Dict[str, Any]]:
    mat = _load_json(FINAL_VAL_DIR / "all_scenarios_metrics.json")
    return [r for r in mat["scenarios"] if r.get("axis") == axis]


# =============================================================================
# FIG 03: Suspension 4x3 Matrix Heatmap (4 Topologies x 3 Vehicle Models)
# =============================================================================
def plot_fig03_suspension_4x3_matrix_heatmap() -> Path:
    apply_clean_scientific_style()
    rows = _load_matrix_rows("axis1_architecture")

    susp_order = [
        "analytic_1d_lut",
        "3d_double_wishbone_lut",
        "decoupled_heave_roll",
        "standard_pushrod",
    ]
    susp_labels = [
        "Analytic 1D LUT",
        "3D Double Wishbone",
        "Decoupled Heave-Roll",
        "Standard Pushrod",
    ]
    model_order = ["fast_3dof", "full_14dof", "differentiable_casadi_14dof"]
    model_labels = ["Fast 3-DOF", "Full 14-DOF", "CasADi 14-DOF"]

    events = [
        ("skidpad", "(a) Skidpad 360 deg Time [s]"),
        ("acceleration_75m", "(b) Acceleration 75 m Time [s]"),
        ("autocross_1184m", "(c) Autocross 1.184 km Time [s]"),
        ("endurance_22km", "(d) Endurance 22 km Time [s]"),
    ]

    fig = plt.figure(figsize=(15.0, 9.4), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.28)

    for idx, (ev, title) in enumerate(events):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        mat = np.zeros((len(susp_order), len(model_order)), dtype=float)
        safe_mat = np.zeros((len(susp_order), len(model_order)), dtype=bool)
        clear_mat = np.zeros((len(susp_order), len(model_order)), dtype=float)

        for i, sm in enumerate(susp_order):
            for j, vm in enumerate(model_order):
                match = [
                    r for r in rows
                    if r["event"] == ev and r["suspension_mode"] == sm and r["vehicle_model"] == vm
                ]
                if match:
                    mat[i, j] = float(match[0]["event_time_s"])
                    safe_mat[i, j] = bool(match[0]["safe_pass"])
                    clear_mat[i, j] = float(match[0]["min_clear_m"])

        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8.0)

        vmin, vmax = float(np.min(mat)), float(np.max(mat))
        mid = vmin + 0.55 * (vmax - vmin)
        for i in range(len(susp_order)):
            for j in range(len(model_order)):
                val = mat[i, j]
                txt_color = "#FFFFFF" if val > mid else "#111827"
                tag = "PASS" if safe_mat[i, j] else "WARN"
                fmt = f"{val:.3f}s" if ev in ("skidpad", "acceleration_75m") else f"{val:.2f}s"
                ax.text(
                    j,
                    i,
                    f"{fmt}\n({tag}, {clear_mat[i, j]:.2f}m)",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    fontweight="bold",
                    color=txt_color,
                )

        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(model_labels, fontsize=8.5)
        ax.set_yticks(np.arange(len(susp_order)))
        ax.set_yticklabels(susp_labels, fontsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#111827", pad=8)
        ax.grid(False)

    fig.suptitle(
        "TeR27 (280.0 kg) 4x3 Architecture Matrix: Suspension Topology x Vehicle Model (FSG_RUBBERED mu=0.57)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig03_suspension_4x3_matrix_heatmap.png")


# =============================================================================
# FIG 04: Grip & Weather Envelope
# =============================================================================
def plot_fig04_grip_and_weather_envelope() -> Path:
    apply_clean_scientific_style()
    rows = _load_matrix_rows("axis2_track_grip")

    grips = ["COLD_TEST_OCT", "DRYING_TRACK", "FSG_RUBBERED", "SPLIT_MU_057_040"]
    grip_labels = ["COLD_TEST_OCT\n(mu=0.47)", "DRYING_TRACK\n(mu=0.52)", "FSG_RUBBERED\n(mu=0.57)", "SPLIT_MU\n(0.57 / 0.40)"]
    grip_colors = [COLORS["blue"], COLORS["amber"], COLORS["emerald"], COLORS["red"]]

    def _get(ev: str, gp: str, field: str) -> float:
        m = [r for r in rows if r["event"] == ev and r["grip_profile"] == gp]
        return float(m[0][field]) if m else 0.0

    fig = plt.figure(figsize=(14.8, 9.0), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    x = np.arange(len(grips))

    # Panel (a): Skidpad & Accel 75m times
    ax0 = fig.add_subplot(gs[0, 0])
    sk_t = [_get("skidpad", g, "event_time_s") for g in grips]
    ac_t = [_get("acceleration_75m", g, "event_time_s") for g in grips]
    w = 0.34
    b1 = ax0.bar(x - w / 2, sk_t, width=w, color=COLORS["blue"], label="Skidpad 360 deg [s]")
    b2 = ax0.bar(x + w / 2, ac_t, width=w, color=COLORS["emerald"], label="Accel 75 m [s]")
    for b, v in zip(b1, sk_t):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["blue"])
    for b, v in zip(b2, ac_t):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["emerald"])
    ax0.set_xticks(x)
    ax0.set_xticklabels(grip_labels)
    ax0.set_ylim(0, max(sk_t) * 1.18)
    _style_ax(ax0, title="(a) Skidpad & Acceleration 75 m vs Grip Profile", ylabel="Event Time [s]")
    ax0.legend(loc="upper right")

    # Panel (b): Autocross 1.184 km time
    ax1 = fig.add_subplot(gs[0, 1])
    ax_t = [_get("autocross_1184m", g, "event_time_s") for g in grips]
    bars = ax1.bar(x, ax_t, width=0.5, color=grip_colors)
    for b, v in zip(bars, ax_t):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f} s", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grip_labels)
    ax1.set_ylim(min(ax_t) * 0.85, max(ax_t) * 1.1)
    _style_ax(ax1, title="(b) Autocross 1.184 km Lap Time vs Grip Profile", ylabel="Autocross Time [s]")

    # Panel (c): Peak lateral ay & longitudinal ax acceleration envelope
    ax2 = fig.add_subplot(gs[1, 0])
    ay_pk = [_get("skidpad", g, "max_ay_g") for g in grips]
    ax_pk = [_get("acceleration_75m", g, "max_ax_g") for g in grips]
    b1 = ax2.bar(x - w / 2, ay_pk, width=w, color=COLORS["blue"], label="Peak Lateral ay (Skidpad) [g]")
    b2 = ax2.bar(x + w / 2, ax_pk, width=w, color=COLORS["amber"], label="Peak Long. ax (Accel 75m) [g]")
    for b, v in zip(b1, ay_pk):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}g", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["blue"])
    for b, v in zip(b2, ax_pk):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}g", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=COLORS["amber"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(grip_labels)
    ax2.set_ylim(0, max(max(ay_pk), max(ax_pk)) * 1.22)
    _style_ax(ax2, title="(c) Peak Tire-Road Acceleration Envelope", ylabel="Acceleration [g]")
    ax2.legend(loc="upper left")

    # Panel (d): Minimum cone clearance across Autocross & Skidpad
    ax3 = fig.add_subplot(gs[1, 1])
    clr_sk = [_get("skidpad", g, "min_clear_m") for g in grips]
    clr_ax = [_get("autocross_1184m", g, "min_clear_m") for g in grips]
    b1 = ax3.bar(x - w / 2, clr_sk, width=w, color=COLORS["emerald"], label="Skidpad min_clear [m]")
    b2 = ax3.bar(x + w / 2, clr_ax, width=w, color=COLORS["slate"], label="Autocross min_clear [m]")
    ax3.axhline(0.150, color=COLORS["red"], lw=1.4, ls="--", label="Safety Floor (0.150 m)")
    for b, v in zip(b1, clr_sk):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COLORS["emerald"])
    for b, v in zip(b2, clr_ax):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COLORS["slate"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(grip_labels)
    ax3.set_ylim(0, max(max(clr_sk), max(clr_ax)) * 1.22)
    _style_ax(ax3, title="(d) Cone Clearance Margin (Split-mu Uses Tube-MPC)", ylabel="Min Cone Clearance [m]")
    ax3.legend(loc="upper right", fontsize=7.8)

    fig.suptitle(
        "TeR27 (280.0 kg) Track Grip & Weather Envelope (COLD_TEST_OCT=0.47 to FSG_RUBBERED=0.57 + Split-mu)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig04_grip_and_weather_envelope.png")


# =============================================================================
# FIG 05: Tire Thermal Window Sensitivity (25 C to 95 C)
# =============================================================================
def plot_fig05_tire_thermal_window_sensitivity() -> Path:
    apply_clean_scientific_style()
    rows = _load_matrix_rows("axis3_tire_thermal")
    therm_rep = _load_json(LIMIT_280_DIR / "thermal_and_endurance_report.json")["part_a_tire_thermal_window"]

    temps = sorted(set(float(r["t_tire_c"]) for r in rows))

    def _series(ev: str, field: str) -> List[float]:
        out = []
        for tc in temps:
            m = [r for r in rows if r["event"] == ev and abs(float(r["t_tire_c"]) - tc) < 1e-3]
            out.append(float(m[0][field]) if m else 0.0)
        return out

    fig = plt.figure(figsize=(14.5, 8.8), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel (a): Fine TTC thermal grip multiplier curve from thermal_and_endurance_report
    ax0 = fig.add_subplot(gs[0, 0])
    fine_t = [float(p["t_tire_start_c"]) for p in therm_rep["acceleration_75m_sweep"]]
    fine_mu = [float(p["mu_thermal"]) for p in therm_rep["acceleration_75m_sweep"]]
    ax0.axvspan(65.0, 90.0, color=COLORS["light_emerald"], alpha=0.65, label="Optimal Window (65-90 deg C)")
    ax0.plot(fine_t, fine_mu, color=COLORS["blue"], lw=2.2, marker="o", ms=4.5, label="Calibrated TTC mu_thermal")
    ax0.axhline(1.0, color=COLORS["slate"], lw=1.0, ls="--")
    _style_ax(ax0, title="(a) Calibrated R25B Thermal Grip Multiplier mu_thermal", xlabel="Tire Temperature [deg C]", ylabel="mu_thermal [-]")
    ax0.legend(loc="lower center")

    # Panel (b): Skidpad & Accel 75 m vs Tire Temp
    ax1 = fig.add_subplot(gs[0, 1])
    sk_t = _series("skidpad", "event_time_s")
    ac_t = _series("acceleration_75m", "event_time_s")
    ax1.axvspan(65.0, 90.0, color=COLORS["light_emerald"], alpha=0.55)
    ax1.plot(temps, sk_t, color=COLORS["blue"], lw=2.0, marker="o", ms=6, label="Skidpad 360 deg [s]")
    ax1.plot(temps, ac_t, color=COLORS["emerald"], lw=2.0, marker="s", ms=6, label="Accel 75 m [s]")
    for tx, sy, ay in zip(temps, sk_t, ac_t):
        ax1.annotate(f"{sy:.2f}", (tx, sy), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.6, color=COLORS["blue"])
        ax1.annotate(f"{ay:.2f}", (tx, ay), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.6, color=COLORS["emerald"])
    _style_ax(ax1, title="(b) Skidpad & Accel 75 m vs Tire Temp", xlabel="Tire Temperature [deg C]", ylabel="Event Time [s]")
    ax1.legend(loc="upper right")

    # Panel (c): Autocross 1.184 km vs Tire Temp
    ax2 = fig.add_subplot(gs[1, 0])
    ax_t = _series("autocross_1184m", "event_time_s")
    ax2.axvspan(65.0, 90.0, color=COLORS["light_emerald"], alpha=0.55, label="Optimal Window (65-90 deg C)")
    ax2.plot(temps, ax_t, color=COLORS["red"], lw=2.2, marker="D", ms=6, label="Autocross 1.184 km [s]")
    for tx, vy in zip(temps, ax_t):
        ax2.annotate(f"{vy:.2f}s", (tx, vy), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.0, fontweight="bold", color=COLORS["red"])
    ax2.set_ylim(min(ax_t) - 1.0, max(ax_t) + 1.5)
    _style_ax(ax2, title="(c) Autocross 1.184 km Lap Time vs Tire Temp", xlabel="Tire Temperature [deg C]", ylabel="Autocross Time [s]")
    ax2.legend(loc="upper right")

    # Panel (d): Peak lateral ay & cone clearance vs Tire Temp
    ax3 = fig.add_subplot(gs[1, 1])
    ay_g = _series("skidpad", "max_ay_g")
    clr = _series("autocross_1184m", "min_clear_m")
    ax3.plot(temps, ay_g, color=COLORS["blue"], lw=2.0, marker="o", ms=6, label="Skidpad Peak ay [g]")
    ax3_r = ax3.twinx()
    ax3_r.plot(temps, clr, color=COLORS["amber"], lw=2.0, marker="^", ms=6, ls="--", label="Autocross min_clear [m]")
    ax3_r.axhline(0.150, color=COLORS["red"], lw=1.1, ls=":", label="Safety Floor (0.150 m)")
    ax3_r.spines["top"].set_visible(False)
    ax3_r.set_ylabel("Min Cone Clearance [m]", fontsize=9.2, color=COLORS["amber"])
    _style_ax(ax3, title="(d) Peak Lateral Grip & Cone Margin vs Tire Temp", xlabel="Tire Temperature [deg C]", ylabel="Peak Lateral ay [g]")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=7.8)

    fig.suptitle(
        "TeR27 (280.0 kg) R25B Tire Thermal Window Sensitivity (25 deg C Cold to 95 deg C Overheat)",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig05_tire_thermal_window_sensitivity.png")


# =============================================================================
# FIG 06: Manufacturing Mass & Aero Degradation Sensitivity
# =============================================================================
def plot_fig06_manufacturing_mass_and_aero_sensitivity() -> Path:
    apply_clean_scientific_style()
    rows = _load_matrix_rows("axis4_mass_aero")

    masses = sorted(set(float(r["mass_kg"]) for r in rows))
    cla_nom = 4.68
    cla_deg = 4.212

    def _series(ev: str, cla: float, field: str) -> List[float]:
        out = []
        for m_kg in masses:
            m = [
                r for r in rows
                if r["event"] == ev and abs(float(r["mass_kg"]) - m_kg) < 1e-3 and abs(float(r["cl_a_m2"]) - cla) < 1e-2
            ]
            out.append(float(m[0][field]) if m else 0.0)
        return out

    fig = plt.figure(figsize=(14.5, 8.8), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    events = [
        ("skidpad", "(a) Skidpad 360 deg Time vs Mass & Aero", "Time [s]", 3),
        ("acceleration_75m", "(b) Acceleration 75 m Time vs Mass & Aero", "Time [s]", 3),
        ("autocross_1184m", "(c) Autocross 1.184 km Time vs Mass & Aero", "Time [s]", 2),
        ("endurance_22km", "(d) Endurance 22 km Time vs Mass & Aero", "Time [s]", 1),
    ]

    for idx, (ev, title, ylab, dec) in enumerate(events):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        y_nom = _series(ev, cla_nom, "event_time_s")
        y_deg = _series(ev, cla_deg, "event_time_s")
        ax.plot(masses, y_nom, color=COLORS["blue"], lw=2.2, marker="o", ms=6, label="Nominal Aero (ClA=4.68 m^2)")
        ax.plot(masses, y_deg, color=COLORS["red"], lw=2.2, marker="s", ms=6, ls="--", label="Degraded Aero (ClA=4.212 m^2, -10%)")
        ax.axvline(280.0, color=COLORS["emerald"], lw=1.2, ls=":", label="280 kg Target")
        for mx, vn, vd in zip(masses, y_nom, y_deg):
            ax.annotate(f"{vn:.{dec}f}", (mx, vn), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7.6, color=COLORS["blue"])
            ax.annotate(f"{vd:.{dec}f}", (mx, vd), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.6, color=COLORS["red"])
        span = max(max(y_deg) - min(y_nom), 0.05)
        ax.set_ylim(min(y_nom) - 0.22 * span, max(y_deg) + 0.22 * span)
        ax.set_xticks(masses)
        _style_ax(ax, title=title, xlabel="Vehicle Mass [kg]", ylabel=ylab)
        ax.legend(loc="upper left", fontsize=7.8)

    fig.suptitle(
        "TeR27 Manufacturing Mass (275-290 kg) & Aero Degradation (-10% ClA) Sensitivity",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig06_manufacturing_mass_and_aero_sensitivity.png")


# =============================================================================
# FIG 07: Tube-MPC Robustness Across All Faults
# =============================================================================
def plot_fig07_tube_mpc_robustness_all_faults() -> Path:
    apply_clean_scientific_style()
    rows = _load_matrix_rows("axis5_fault_control")

    faults = [
        "nominal",
        "125ms_can_2x_imu_noise",
        "grip_drop_0p57_to_0p47",
        "3wd_limp_home_braking",
        "3wd_limp_home_slalom",
    ]
    fault_labels = [
        "Nominal",
        "125ms CAN\n+2x IMU",
        "Grip Drop\n0.57->0.47",
        "3WD Limp\nBraking",
        "3WD Limp\nSlalom",
    ]

    def _get(ev: str, ctrl: str, flt: str, field: str) -> float:
        m = [r for r in rows if r["event"] == ev and r["controller_mode"] == ctrl and r["fault_mode"] == flt]
        return float(m[0][field]) if m else 0.0

    fig = plt.figure(figsize=(15.0, 9.2), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.25)
    x = np.arange(len(faults))
    w = 0.35

    # Panel (a): Autocross minimum cone clearance
    ax0 = fig.add_subplot(gs[0, 0])
    det_clr = [_get("autocross_1184m", "deterministic_4wd_tv", f, "min_clear_m") for f in faults]
    mpc_clr = [_get("autocross_1184m", "refined_asymmetric_tube_mpc", f, "min_clear_m") for f in faults]
    b1 = ax0.bar(x - w / 2, det_clr, width=w, color=COLORS["amber"], label="Deterministic 4WD TV")
    b2 = ax0.bar(x + w / 2, mpc_clr, width=w, color=COLORS["emerald"], label="Refined Asymmetric Tube-MPC")
    ax0.axhline(0.150, color=COLORS["red"], lw=1.4, ls="--", label="Safety Floor (0.150 m)")
    for b, v in zip(b1, det_clr):
        ax0.text(b.get_x() + b.get_width() / 2, max(v, 0.01) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=COLORS["amber"], fontweight="bold")
    for b, v in zip(b2, mpc_clr):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=COLORS["emerald"], fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(fault_labels, fontsize=8.2)
    ax0.set_ylim(0, max(max(det_clr), max(mpc_clr)) * 1.24)
    _style_ax(ax0, title="(a) Autocross Min Cone Clearance Under Faults", ylabel="Min Clearance [m]")
    ax0.legend(loc="upper right", fontsize=7.8)

    # Panel (b): Total cones hit across all 4 events per fault mode
    ax1 = fig.add_subplot(gs[0, 1])
    events_all = ["skidpad", "acceleration_75m", "autocross_1184m", "endurance_22km"]
    det_cones = [sum(_get(ev, "deterministic_4wd_tv", f, "cones_hit") for ev in events_all) for f in faults]
    mpc_cones = [sum(_get(ev, "refined_asymmetric_tube_mpc", f, "cones_hit") for ev in events_all) for f in faults]
    b1 = ax1.bar(x - w / 2, det_cones, width=w, color=COLORS["red"], label="Deterministic 4WD TV")
    b2 = ax1.bar(x + w / 2, mpc_cones, width=w, color=COLORS["blue"], label="Refined Asymmetric Tube-MPC (0 cones)")
    for b, v in zip(b1, det_cones):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{int(v)}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=COLORS["red"])
    for b, v in zip(b2, mpc_cones):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{int(v)}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=COLORS["blue"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(fault_labels, fontsize=8.2)
    ax1.set_ylim(0, max(max(det_cones), 1.0) * 1.25)
    _style_ax(ax1, title="(b) Total Cones Hit Across All 4 Events", ylabel="Cones Hit [count]")
    ax1.legend(loc="upper left", fontsize=7.8)

    # Panel (c): Autocross lap time comparison
    ax2 = fig.add_subplot(gs[1, 0])
    det_t = [_get("autocross_1184m", "deterministic_4wd_tv", f, "event_time_s") for f in faults]
    mpc_t = [_get("autocross_1184m", "refined_asymmetric_tube_mpc", f, "event_time_s") for f in faults]
    b1 = ax2.bar(x - w / 2, det_t, width=w, color=COLORS["slate"], alpha=0.8, label="Deterministic 4WD TV")
    b2 = ax2.bar(x + w / 2, mpc_t, width=w, color=COLORS["blue"], label="Refined Asymmetric Tube-MPC")
    for b, v in zip(b1, det_t):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7.4, color=COLORS["slate"])
    for b, v in zip(b2, mpc_t):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=COLORS["blue"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(fault_labels, fontsize=8.2)
    ax2.set_ylim(min(min(det_t), min(mpc_t)) * 0.9, max(max(det_t), max(mpc_t)) * 1.08)
    _style_ax(ax2, title="(c) Autocross 1.184 km Time Under Faults", ylabel="Autocross Time [s]")
    ax2.legend(loc="upper left", fontsize=7.8)

    # Panel (d): HV DC Power Envelope (P_dc_min & P_dc_max vs [-25, +80] kW)
    ax3 = fig.add_subplot(gs[1, 1])
    mpc_pmin = [_get("autocross_1184m", "refined_asymmetric_tube_mpc", f, "p_dc_min_kw") for f in faults]
    mpc_pmax = [_get("autocross_1184m", "refined_asymmetric_tube_mpc", f, "p_dc_max_kw") for f in faults]
    ax3.bar(x, mpc_pmax, width=0.45, color=COLORS["blue"], label="Tube-MPC P_dc_max [kW]")
    ax3.bar(x, mpc_pmin, width=0.45, color=COLORS["emerald"], label="Tube-MPC P_dc_min (Regen) [kW]")
    ax3.axhline(80.0, color=COLORS["red"], lw=1.3, ls="--", label="HV Limit (+80 / -25 kW)")
    ax3.axhline(-25.0, color=COLORS["red"], lw=1.3, ls="--")
    ax3.axhline(0.0, color="#374151", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(fault_labels, fontsize=8.2)
    ax3.set_ylim(-35, 95)
    _style_ax(ax3, title="(d) Tube-MPC HV DC Power Envelope (0 Violations)", ylabel="HV DC Power [kW]")
    ax3.legend(loc="upper right", fontsize=7.8)

    fig.suptitle(
        "TeR27 (280.0 kg) Fault & Control Robustness: Deterministic 4WD TV vs Refined Asymmetric Tube-MPC",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    return _save_clean_fig(fig, "fig07_tube_mpc_robustness_all_faults.png")


def generate_all_clean_plots() -> Dict[str, Any]:
    """Generate all 8 clean publication plots and verify each PNG."""
    plot_fns = [
        ("fig01_ter26_empirical_validation.png", plot_fig01_ter26_empirical_validation),
        ("fig02_ter27_280kg_lap_times_and_attribution.png", plot_fig02_ter27_280kg_lap_times_and_attribution),
        ("fig03_suspension_4x3_matrix_heatmap.png", plot_fig03_suspension_4x3_matrix_heatmap),
        ("fig04_grip_and_weather_envelope.png", plot_fig04_grip_and_weather_envelope),
        ("fig05_tire_thermal_window_sensitivity.png", plot_fig05_tire_thermal_window_sensitivity),
        ("fig06_manufacturing_mass_and_aero_sensitivity.png", plot_fig06_manufacturing_mass_and_aero_sensitivity),
        ("fig07_tube_mpc_robustness_all_faults.png", plot_fig07_tube_mpc_robustness_all_faults),
        ("fig08_endurance_22km_energy_and_thermal_budget.png", plot_fig08_endurance_22km_energy_and_thermal_budget),
    ]
    results: Dict[str, Any] = {}
    for name, fn in plot_fns:
        p = fn()
        results[name] = verify_clean_figure(p)
    return results


if __name__ == "__main__":
    res = generate_all_clean_plots()
    print(json.dumps(res, indent=2))
