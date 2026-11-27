#!/usr/bin/env python3
"""
Trackside Mechanical Setup Co-Optimization Comparison Plotter (Vault 07).

Reads:
  outputs/validation_vault/07_design_tradeoffs_and_fs_points/trackside_setup_optimization_report.json

Generates:
  outputs/validation_vault/07_design_tradeoffs_and_fs_points/trackside_setup_event_comparison.png

Compares Setup Base vs Setup Optimal LTS across all 4 Formula Student dynamic events:
  - Skidpad (s)
  - Acceleration 75 m (s)
  - Autocross (s)
  - Endurance 22 km (s, with energy budget kWh)

Pure ASCII, 300 DPI, MoTeC i2 dark engineering theme via `plot_style.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.config.paths import VAULT_07_DESIGN_POINTS  # noqa: E402
from vd_correlation_lab.analysis.visualization.plot_style import (  # noqa: E402
    PALETTE,
    add_footer,
    add_kpi_box,
    add_suptitle,
    apply_motec_style,
    save_fig,
    style_ax,
)

REPORT_PATH = VAULT_07_DESIGN_POINTS / "trackside_setup_optimization_report.json"
OUT_PNG_PATH = VAULT_07_DESIGN_POINTS / "trackside_setup_event_comparison.png"

EVENT_SPECS: List[Tuple[str, str, List[str]]] = [
    ("skidpad", "Skidpad", ["skidpad", "Skidpad", "SKIDPAD"]),
    ("accel", "Acceleration 75 m", ["accel", "acceleration", "acceleration_75m", "accel_75m", "Acceleration"]),
    ("autocross", "Autocross", ["autocross", "Autocross", "AUTOCROSS"]),
    ("endurance", "Endurance 22 km", ["endurance", "endurance_22km", "Endurance", "ENDURANCE"]),
]

SETUP_PARAM_SPECS: List[Tuple[str, str, Tuple[float, float], List[str]]] = [
    ("lltd", "LLTD [-]", (0.35, 0.62), ["lltd", "LLTD"]),
    ("camber_f_deg", "Camber F [deg]", (-2.5, -0.2), ["camber_f_deg", "camber_front_deg"]),
    ("camber_r_deg", "Camber R [deg]", (-2.2, -0.2), ["camber_r_deg", "camber_rear_deg"]),
    ("toe_f_deg", "Toe F [deg]", (-0.5, 0.5), ["toe_f_deg", "toe_front_deg"]),
    ("toe_r_deg", "Toe R [deg]", (0.0, 0.6), ["toe_r_deg", "toe_rear_deg"]),
    ("brake_bias", "Brake Bias [-]", (0.55, 0.78), ["brake_bias", "brake_bias_front"]),
    ("rake_mm", "Rake [mm]", (-8.0, 8.0), ["rake_mm", "rake"]),
]


def _pick_dict(container: Dict[str, Any], aliases: List[str]) -> Optional[Dict[str, Any]]:
    for key in aliases:
        if key in container and isinstance(container[key], dict):
            return container[key]
    low_map = {str(k).lower(): v for k, v in container.items()}
    for key in aliases:
        lk = key.lower()
        if lk in low_map and isinstance(low_map[lk], dict):
            return low_map[lk]
    return None


def _pick_float(d: Dict[str, Any], keys: List[str], default: float = float("nan")) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return default


def _extract_event_metrics(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize event comparison records from trackside_setup_optimization_report.json."""
    events_root: Dict[str, Any] = report
    for candidate in ("events", "event_results", "comparison", "results"):
        if candidate in report and isinstance(report[candidate], dict):
            events_root = report[candidate]
            break

    base_setup_global = (
        report.get("base_setup")
        or report.get("setup_base")
        or {}
    )

    out: Dict[str, Dict[str, Any]] = {}
    for canonical, label, aliases in EVENT_SPECS:
        ev = _pick_dict(events_root, aliases)
        if ev is None:
            raise KeyError(f"Missing event '{canonical}' in {REPORT_PATH}")

        t_base = _pick_float(
            ev,
            [
                "base_time_s",
                "t_base_s",
                "time_base_s",
                "baseline_time_s",
                "setup_base_time_s",
            ],
        )
        t_opt = _pick_float(
            ev,
            [
                "optimal_time_s",
                "opt_time_s",
                "t_opt_s",
                "time_opt_s",
                "setup_optimal_time_s",
            ],
        )
        if np.isnan(t_base) and isinstance(ev.get("base"), dict):
            t_base = _pick_float(ev["base"], ["time_s", "t_s", "lap_time_s", "total_time_s"])
        if np.isnan(t_opt) and isinstance(ev.get("optimal"), dict):
            t_opt = _pick_float(ev["optimal"], ["time_s", "t_s", "lap_time_s", "total_time_s"])

        delta_s = _pick_float(ev, ["delta_time_s", "delta_s", "dt_s"], default=t_opt - t_base)
        delta_pct = _pick_float(
            ev,
            ["delta_pct", "delta_time_pct"],
            default=(100.0 * delta_s / max(abs(t_base), 1e-6)),
        )

        opt_setup = (
            ev.get("optimal_setup")
            or ev.get("setup_optimal")
            or (ev.get("optimal", {}).get("setup") if isinstance(ev.get("optimal"), dict) else None)
            or {}
        )
        base_setup = (
            ev.get("base_setup")
            or ev.get("setup_base")
            or (ev.get("base", {}).get("setup") if isinstance(ev.get("base"), dict) else None)
            or base_setup_global
            or {}
        )

        e_base_kwh = _pick_float(ev, ["base_energy_kwh", "e_base_kwh"])
        e_opt_kwh = _pick_float(ev, ["optimal_energy_kwh", "opt_energy_kwh", "e_opt_kwh", "e_22km_kwh"])

        out[canonical] = {
            "label": label,
            "t_base_s": t_base,
            "t_opt_s": t_opt,
            "delta_s": delta_s,
            "delta_pct": delta_pct,
            "base_setup": base_setup,
            "optimal_setup": opt_setup,
            "e_base_kwh": e_base_kwh,
            "e_opt_kwh": e_opt_kwh,
        }
    return out


def _get_setup_val(setup_dict: Dict[str, Any], aliases: List[str], fallback: float = 0.0) -> float:
    for k in aliases:
        if k in setup_dict and setup_dict[k] is not None:
            return float(setup_dict[k])
    return fallback


def generate_comparison_plot(report_path: Path = REPORT_PATH, out_path: Path = OUT_PNG_PATH) -> Path:
    if not report_path.exists():
        raise FileNotFoundError(f"Missing optimization report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = _extract_event_metrics(report)

    apply_motec_style(dpi=300)
    fig = plt.figure(figsize=(16.5, 10.2))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.05, 1.15], hspace=0.34, wspace=0.28)

    # Row 1: 4 per-event Base vs Optimal LTS bar charts with exact delta callouts
    event_order = ["skidpad", "accel", "autocross", "endurance"]
    panel_tags = ["A", "B", "C", "D"]
    for idx, (ekey, ptag) in enumerate(zip(event_order, panel_tags)):
        ax = fig.add_subplot(gs[0, idx])
        ev = events[ekey]
        t_b = ev["t_base_s"]
        t_o = ev["t_opt_s"]
        dt = ev["delta_s"]
        dpct = ev["delta_pct"]

        bars = ax.bar(
            [0, 1],
            [t_b, t_o],
            width=0.52,
            color=[PALETTE["slate"], PALETTE["emerald"]],
            edgecolor=[PALETTE["white"], PALETTE["white"]],
            linewidth=0.85,
            alpha=0.92,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base Setup", "Optimal LTS"], fontsize=8.4)

        y_min = min(t_b, t_o)
        y_max = max(t_b, t_o)
        span = max(y_max - y_min, y_max * 0.04)
        ax.set_ylim(max(0.0, y_min - 1.8 * span), y_max + 1.35 * span)

        for b, val in zip(bars, (t_b, t_o)):
            fmt = f"{val:.3f} s" if ekey in ("skidpad", "accel") else f"{val:.2f} s"
            ax.text(
                b.get_x() + b.get_width() * 0.5,
                val + 0.18 * span,
                fmt,
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontfamily="DejaVu Sans Mono",
                fontweight="bold",
                color=PALETTE["white"],
            )

        delta_str = f"{dt:+.3f} s ({dpct:+.2f}%)" if ekey in ("skidpad", "accel") else f"{dt:+.2f} s ({dpct:+.2f}%)"
        kpi_lines = [f"Delta: {delta_str}"]
        if ekey == "endurance" and np.isfinite(ev["e_opt_kwh"]):
            kpi_lines.append(f"E_opt: {ev['e_opt_kwh']:.2f} kWh")
        add_kpi_box(
            ax,
            kpi_lines,
            loc="upper right",
            fontsize=7.6,
            border_color=PALETTE["emerald"] if dt <= 0.0 else PALETTE["amber"],
        )
        style_ax(
            ax,
            title=f"{ptag}. {ev['label']}",
            ylabel="Event Time [s]" if idx == 0 else "",
        )

    # Row 2 Left (cols 0..1): Percentage improvement summary across all 4 events
    ax_pct = fig.add_subplot(gs[1, 0:2])
    y_pos = np.arange(len(event_order))
    gains_pct = [-events[k]["delta_pct"] for k in event_order]
    labels_ev = [events[k]["label"] for k in event_order]
    bar_cols = [PALETTE["cyan"], PALETTE["violet"], PALETTE["amber"], PALETTE["emerald"]]
    ax_pct.barh(y_pos, gains_pct, height=0.55, color=bar_cols, edgecolor=PALETTE["white"], linewidth=0.8, alpha=0.92)
    ax_pct.axvline(0.0, color=PALETTE["white"], linewidth=1.0)
    max_g = max(max(gains_pct), 0.5)
    ax_pct.set_xlim(min(0.0, min(gains_pct) - 0.5), max_g * 1.38)
    for i, k in enumerate(event_order):
        ev = events[k]
        g = gains_pct[i]
        txt = f"{ev['delta_s']:+.3f} s ({ev['delta_pct']:+.2f}%)" if k in ("skidpad", "accel") else f"{ev['delta_s']:+.2f} s ({ev['delta_pct']:+.2f}%)"
        ax_pct.text(
            max(g, 0.0) + max_g * 0.03,
            i,
            txt,
            ha="left",
            va="center",
            fontsize=8.3,
            fontfamily="DejaVu Sans Mono",
            color=PALETTE["white"],
        )
    ax_pct.set_yticks(y_pos)
    ax_pct.set_yticklabels(labels_ev, fontsize=8.8)
    ax_pct.invert_yaxis()
    style_ax(
        ax_pct,
        title="E. Relative Lap / Event Time Gain (Setup Optimal LTS vs Base)",
        xlabel="Time Reduction [% faster]",
    )
    fs_pts = report.get("fs_points")
    if isinstance(fs_pts, dict) and "base" in fs_pts and "optimal" in fs_pts:
        b_pts = float(fs_pts["base"].get("total_dynamic_pts", 0.0))
        o_pts = float(fs_pts["optimal"].get("total_dynamic_pts", 0.0))
        d_pts = float(fs_pts.get("delta_total_pts", o_pts - b_pts))
        add_kpi_box(
            ax_pct,
            [
                f"FS Dyn Base : {b_pts:.2f} pts",
                f"FS Dyn Opt  : {o_pts:.2f} pts",
                f"Delta Pts   : {d_pts:+.2f} pts",
            ],
            loc="lower right",
            title="FS POINTS GAIN",
            fontsize=7.8,
            border_color=PALETTE["emerald"],
        )

    # Row 2 Right (cols 2..3): Normalized 7-parameter Trackside Setup Matrix Heatmap
    ax_hm = fig.add_subplot(gs[1, 2:4])
    n_params = len(SETUP_PARAM_SPECS)
    n_ev = len(event_order)
    norm_mat = np.zeros((n_params, n_ev), dtype=float)
    raw_mat = np.zeros((n_params, n_ev), dtype=float)
    for pi, (_, _, (lo, hi), aliases) in enumerate(SETUP_PARAM_SPECS):
        for ei, ekey in enumerate(event_order):
            val = _get_setup_val(events[ekey]["optimal_setup"], aliases, fallback=0.5 * (lo + hi))
            raw_mat[pi, ei] = val
            norm_mat[pi, ei] = np.clip((val - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    im = ax_hm.imshow(norm_mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_hm.set_xticks(np.arange(n_ev))
    ax_hm.set_xticklabels(["Skidpad", "Accel 75m", "Autocross", "Endurance"], fontsize=8.4)
    ax_hm.set_yticks(np.arange(n_params))
    ax_hm.set_yticklabels([sp[1] for sp in SETUP_PARAM_SPECS], fontsize=8.3)
    for pi in range(n_params):
        for ei in range(n_ev):
            v = raw_mat[pi, ei]
            nv = norm_mat[pi, ei]
            txt_col = "#0D1117" if nv > 0.62 else PALETTE["white"]
            ax_hm.text(
                ei,
                pi,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontfamily="DejaVu Sans Mono",
                fontweight="bold",
                color=txt_col,
            )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.03)
    cbar.set_label("Normalized Bound [0=Min, 1=Max]", color=PALETTE["text"], fontsize=8.2)
    cbar.ax.tick_params(colors=PALETTE["muted"], labelsize=7.8)
    style_ax(
        ax_hm,
        title="F. Optimal Trackside Mechanical Setup Sheet per Event",
    )
    ax_hm.grid(False)

    grip_note = str(report.get("grip_profile", "FSG_RUBBERED (lambda_muy=0.57, R25B @ 75 C)"))
    add_suptitle(
        fig,
        "TeR27 4WD TRACKSIDE SETUP CO-OPTIMIZATION: BASE vs OPTIMAL LTS",
        f"CasADi Joint Trajectory + 7-DOF Mechanical Setup Co-Optimization | {grip_note}",
    )
    add_footer(fig)
    fig.subplots_adjust(left=0.055, right=0.96, top=0.89, bottom=0.07)
    return save_fig(fig, out_path, dpi=300)


def main() -> int:
    out = generate_comparison_plot()
    print(f"[DONE] Saved 300 DPI trackside setup comparison plot -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
