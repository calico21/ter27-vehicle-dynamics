#!/usr/bin/env python3
"""
Milliken Yaw Moment Diagram (YMD) analyzer & plotter for MirenaSim (TeR27 4WD).
Reads vd_correlation_lab/results/ymd/ymd_raw.json (or runs godot_ymd_probe.gd if missing),
computes Stability Index, Control Index, Trim Max Ay, delta_lim(v), axle saturation,
and exports:
  - vd_correlation_lab/results/ymd/ymd_summary.json
  - vd_correlation_lab/results/ymd/01_ymd_open_vs_tv4_40_65_90kmh.png
  - vd_correlation_lab/results/ymd/02_stability_control_and_axle_saturation.png
  - vd_correlation_lab/results/ymd/REPORTE_YMD_TeR27.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = LAB_DIR.parent
sys.path.insert(0, str(LAB_DIR))
from vd_correlation_lab.analysis.visualization.style import (  # noqa: E402
    C_HORIZON,
    C_MUTED,
    C_OPEN,
    C_REAL,
    C_TER27,
    apply_style,
    save_fig,
)

RESULTS_DIR = LAB_DIR / "results" / "ymd"
RAW_JSON = RESULTS_DIR / "ymd_raw.json"
PROBE_GD = Path(__file__).resolve().parent / "godot_ymd_probe.gd"


def ensure_raw_data(force_probe: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if force_probe or not RAW_JSON.exists():
        cmd = [
            "godot",
            "--headless",
            "--path",
            str(REPO_ROOT / "src" / "mirena_sim"),
            "-s",
            str(PROBE_GD),
            f"--out={RAW_JSON}",
        ]
        print("Running Godot YMD probe:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return json.loads(RAW_JSON.read_text(encoding="utf-8"))


def _build_grid(sweep: dict, betas: np.ndarray, deltas: np.ndarray) -> dict[str, np.ndarray]:
    nb, nd = len(betas), len(deltas)
    b_map = {round(float(b), 4): i for i, b in enumerate(betas)}
    d_map = {round(float(d), 4): j for j, d in enumerate(deltas)}
    fields = ["ay_g", "cn", "mz_nm", "sat_front", "sat_rear", "roll_deg", "mz_tv_nm"]
    arrs = {k: np.zeros((nb, nd), dtype=float) for k in fields}
    for pt in sweep["points"]:
        i = b_map[round(float(pt["beta_deg"]), 4)]
        j = d_map[round(float(pt["delta_deg"]), 4)]
        for k in fields:
            arrs[k][i, j] = float(pt[k])
    return arrs


def _interp_trim_at_beta(ay_row: np.ndarray, cn_row: np.ndarray, deltas: np.ndarray, extra: dict[str, np.ndarray], b_idx: int):
    """Find trim crossings (CN = 0) along delta for a fixed beta row."""
    trims = []
    for j in range(len(deltas) - 1):
        c0, c1 = cn_row[j], cn_row[j + 1]
        if c0 == 0.0 or c0 * c1 < 0.0:
            denom = c1 - c0
            u = 0.0 if abs(denom) < 1e-12 else -c0 / denom
            u = float(np.clip(u, 0.0, 1.0))
            ay_t = float(ay_row[j] + u * (ay_row[j + 1] - ay_row[j]))
            d_t = float(deltas[j] + u * (deltas[j + 1] - deltas[j]))
            sf_t = float(extra["sat_front"][b_idx, j] + u * (extra["sat_front"][b_idx, j + 1] - extra["sat_front"][b_idx, j]))
            sr_t = float(extra["sat_rear"][b_idx, j] + u * (extra["sat_rear"][b_idx, j + 1] - extra["sat_rear"][b_idx, j]))
            trims.append({"ay_g": ay_t, "delta_deg": d_t, "sat_front": sf_t, "sat_rear": sr_t})
    return trims


def _interp_cn_at_ay(ay_col: np.ndarray, cn_col: np.ndarray, target_ay: float) -> float | None:
    """Interpolate CN(beta) at target_ay along a constant-beta curve (monotone near trim)."""
    for j in range(len(ay_col) - 1):
        a0, a1 = ay_col[j], ay_col[j + 1]
        if (a0 - target_ay) * (a1 - target_ay) <= 0.0 and abs(a1 - a0) > 1e-6:
            u = (target_ay - a0) / (a1 - a0)
            return float(cn_col[j] + u * (cn_col[j + 1] - cn_col[j]))
    return None


def analyze_sweep(sweep: dict, betas: np.ndarray, deltas: np.ndarray) -> dict:
    g = _build_grid(sweep, betas, deltas)
    ay = g["ay_g"]
    cn = g["cn"]

    # Trim points across all beta rows (take positive Ay branch for right/left symmetry)
    trim_pts = []
    for i, b in enumerate(betas):
        for tr in _interp_trim_at_beta(ay[i], cn[i], deltas, g, i):
            tr["beta_deg"] = float(b)
            trim_pts.append(tr)

    pos_trims = [t for t in trim_pts if t["ay_g"] >= 0.0]
    if pos_trims:
        max_trim = max(pos_trims, key=lambda t: t["ay_g"])
    else:
        i_max, j_max = np.unravel_index(np.argmax(ay), ay.shape)
        max_trim = {
            "ay_g": float(ay[i_max, j_max]),
            "delta_deg": float(deltas[j_max]),
            "beta_deg": float(betas[i_max]),
            "sat_front": float(g["sat_front"][i_max, j_max]),
            "sat_rear": float(g["sat_rear"][i_max, j_max]),
        }

    # Critical steer angle delta_lim at maximum Ay (at beta = 0 or overall trim)
    i_b0 = int(np.argmin(np.abs(betas)))
    j_ay_peak_b0 = int(np.argmax(ay[i_b0]))
    delta_peak_b0 = float(deltas[j_ay_peak_b0])
    ay_peak_b0 = float(ay[i_b0, j_ay_peak_b0])

    # Control Index dCN/ddelta at beta = 0, delta = 0
    j_d0 = int(np.argmin(np.abs(deltas)))
    if 0 < j_d0 < len(deltas) - 1:
        d_cn_d_delta = float((cn[i_b0, j_d0 + 1] - cn[i_b0, j_d0 - 1]) / (deltas[j_d0 + 1] - deltas[j_d0 - 1]))
    else:
        d_cn_d_delta = 0.0

    # Stability Index dCN/dbeta at beta=0 (Ay=0g), Ay=1.0g, and near limit (0.85 * max_trim_ay)
    def stab_at_ay(ay_target: float) -> float:
        vals = []
        bs = []
        for i in range(len(betas)):
            c_val = _interp_cn_at_ay(ay[i], cn[i], ay_target)
            if c_val is not None:
                vals.append(c_val)
                bs.append(float(betas[i]))
        if len(bs) >= 2:
            # Linear slope around beta closest to 0
            idx_c = int(np.argmin(np.abs(np.array(bs))))
            i0 = max(0, idx_c - 1)
            i1 = min(len(bs) - 1, idx_c + 1)
            if abs(bs[i1] - bs[i0]) > 1e-6:
                return float((vals[i1] - vals[i0]) / (bs[i1] - bs[i0]))
        return float("nan")

    # At Ay=0, evaluate directly along delta=0 (closest to straight-ahead)
    if 0 < i_b0 < len(betas) - 1:
        stab_0g = float((cn[i_b0 + 1, j_d0] - cn[i_b0 - 1, j_d0]) / (betas[i_b0 + 1] - betas[i_b0 - 1]))
    else:
        stab_0g = float("nan")
    stab_1g = stab_at_ay(1.0)
    stab_lim = stab_at_ay(0.85 * float(max_trim["ay_g"]))

    # Axle saturation at limit trim
    sf_lim = float(max_trim["sat_front"])
    sr_lim = float(max_trim["sat_rear"])
    if abs(sf_lim - sr_lim) < 0.03:
        limit_balance = "NEUTRAL"
    elif sf_lim > sr_lim:
        limit_balance = "UNDERSTEER (front saturates first)"
    else:
        limit_balance = "OVERSTEER (rear saturates first)"

    return {
        "controller": sweep["controller"],
        "v_kmh": float(sweep["v_kmh"]),
        "stability_index_0g_per_deg": stab_0g,
        "stability_index_1g_per_deg": stab_1g,
        "stability_index_lim_per_deg": stab_lim,
        "control_index_per_deg": d_cn_d_delta,
        "trim_max_ay_g": float(max_trim["ay_g"]),
        "trim_delta_deg": float(max_trim["delta_deg"]),
        "trim_beta_deg": float(max_trim["beta_deg"]),
        "delta_lim_b0_deg": delta_peak_b0,
        "ay_peak_b0_g": ay_peak_b0,
        "sat_front_at_limit": sf_lim,
        "sat_rear_at_limit": sr_lim,
        "limit_balance": limit_balance,
        "grid": g,
        "trim_pts": trim_pts,
    }


def plot_ymd_panels(analyzed: list[dict], betas: np.ndarray, deltas: np.ndarray) -> Path:
    apply_style()
    speeds = sorted({a["v_kmh"] for a in analyzed})
    fig, axes = plt.subplots(1, len(speeds), figsize=(15.5, 5.2), sharey=True)
    if len(speeds) == 1:
        axes = [axes]

    for ax, v_kmh in zip(axes, speeds):
        for ctrl, col, lw, ls, lbl in [
            ("OPEN_4WD", C_OPEN, 1.15, "--", "OPEN 4WD"),
            ("TV4_AUTOCROSS", C_HORIZON, 1.35, "-", "TV4 AUTOCROSS"),
        ]:
            rec = next(a for a in analyzed if a["v_kmh"] == v_kmh and a["controller"] == ctrl)
            ay = rec["grid"]["ay_g"]
            cn = rec["grid"]["cn"]
            # Plot constant-beta lines
            for i in range(len(betas)):
                ax.plot(ay[i], cn[i], color=col, alpha=0.38 if ctrl == "OPEN_4WD" else 0.70, linewidth=lw, linestyle=ls)
            # Plot constant-delta lines (subset for readability)
            for j, d in enumerate(deltas):
                if abs(d) in (0.0, 2.0, 4.0, 6.0, 10.0, 18.0):
                    ax.plot(ay[:, j], cn[:, j], color=col, alpha=0.22 if ctrl == "OPEN_4WD" else 0.45, linewidth=0.85, linestyle=":")
            # Highlight beta=0 spine
            i0 = int(np.argmin(np.abs(betas)))
            ax.plot(ay[i0], cn[i0], color=col, linewidth=2.2, linestyle=ls, label=f"{lbl} (Ay_trim={rec['trim_max_ay_g']:.2f}g)")

        ax.axhline(0.0, color=C_REAL, linewidth=1.2, linestyle="-")
        ax.axvline(0.0, color=C_MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(f"YMD @ {v_kmh:.0f} km/h")
        ax.set_xlabel("Lateral Acceleration Ay [g]")
        ax.legend(loc="upper left", fontsize=7.5)

    axes[0].set_ylabel("Yaw Moment Coefficient CN = Mz / (m * g * L)")
    out_path = RESULTS_DIR / "01_ymd_open_vs_tv4_40_65_90kmh.png"
    save_fig(fig, out_path)
    return out_path


def plot_indices_and_saturation(analyzed: list[dict], max_steer_deg: float) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    speeds = sorted({a["v_kmh"] for a in analyzed})

    for ctrl, col, mk, lbl in [
        ("OPEN_4WD", C_OPEN, "o", "OPEN 4WD"),
        ("TV4_AUTOCROSS", C_HORIZON, "s", "TV4 AUTOCROSS"),
    ]:
        recs = [next(a for a in analyzed if a["v_kmh"] == v and a["controller"] == ctrl) for v in speeds]
        axes[0, 0].plot(speeds, [r["stability_index_1g_per_deg"] for r in recs], color=col, marker=mk, linewidth=1.8, label=f"{lbl} (@1.0g)")
        axes[0, 0].plot(speeds, [r["stability_index_lim_per_deg"] for r in recs], color=col, marker=mk, linestyle="--", alpha=0.7, label=f"{lbl} (@lim)")
        axes[0, 1].plot(speeds, [r["control_index_per_deg"] for r in recs], color=col, marker=mk, linewidth=1.8, label=lbl)
        axes[1, 0].plot(speeds, [abs(r["trim_delta_deg"]) for r in recs], color=col, marker=mk, linewidth=1.8, label=f"{lbl} |delta_trim_max|")
        axes[1, 1].plot(speeds, [r["sat_front_at_limit"] * 100.0 for r in recs], color=col, marker=mk, linewidth=1.8, label=f"{lbl} Front")
        axes[1, 1].plot(speeds, [r["sat_rear_at_limit"] * 100.0 for r in recs], color=col, marker="^", linestyle="--", linewidth=1.5, label=f"{lbl} Rear")

    axes[0, 0].axhline(0.0, color=C_REAL, linewidth=1.0)
    axes[0, 0].set_title("Stability Index dCN/dbeta [1/deg] (>0 = Stable in ISO 8855)")
    axes[0, 0].set_xlabel("Speed [km/h]")
    axes[0, 0].set_ylabel("dCN/dbeta [1/deg]")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_title("Control Index dCN/ddelta [1/deg] (beta=0)")
    axes[0, 1].set_xlabel("Speed [km/h]")
    axes[0, 1].set_ylabel("dCN/ddelta [1/deg]")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].axhline(max_steer_deg, color=C_TER27, linestyle="--", linewidth=1.5, label=f"Unscaled Stick Max ({max_steer_deg:.1f} deg)")
    axes[1, 0].set_title("Trim Limit Steer Angle vs Unscaled Stick Cap")
    axes[1, 0].set_xlabel("Speed [km/h]")
    axes[1, 0].set_ylabel("Steer Angle [deg]")
    axes[1, 0].legend(fontsize=7.5)

    axes[1, 1].axhline(100.0, color=C_MUTED, linestyle=":", linewidth=1.0)
    axes[1, 1].set_title("Axle Lateral Saturation at Trim Limit [%]")
    axes[1, 1].set_xlabel("Speed [km/h]")
    axes[1, 1].set_ylabel("|Fy| / (mu * Fz) [%]")
    axes[1, 1].legend(fontsize=7)

    out_path = RESULTS_DIR / "02_stability_control_and_axle_saturation.png"
    save_fig(fig, out_path)
    return out_path


def write_report(raw: dict, analyzed: list[dict]) -> Path:
    lines = [
        "# Milliken Yaw Moment Diagram (YMD) — TeR27 4WD",
        "",
        f"- **Vehicle:** `{raw['vehicle_id']}` (`m = {raw['mass_kg']:.1f} kg`, `L = {raw['wheelbase_m']:.3f} m`, `Cl*A = {raw['cl_a']:.2f} m^2`, `max_steer = {raw['max_steer_deg']:.2f} deg`, `use_3d_kinematics = {raw['use_3d_kinematics']}`)",
        "",
        "## 1. YMD Summary Table (40 / 65 / 90 km/h)",
        "",
        "| Speed [km/h] | Controller | dCN/dbeta @0g | dCN/dbeta @1g | dCN/dbeta @lim | dCN/ddelta | Trim Max Ay [g] | delta_trim [deg] | beta_trim [deg] | Sat Front / Rear [%] | Limit Balance |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in analyzed:
        lines.append(
            f"| {r['v_kmh']:.0f} | {r['controller']} | {r['stability_index_0g_per_deg']:+.4f} | {r['stability_index_1g_per_deg']:+.4f} | {r['stability_index_lim_per_deg']:+.4f} | {r['control_index_per_deg']:+.4f} | {r['trim_max_ay_g']:.2f} | {r['trim_delta_deg']:.1f} | {r['trim_beta_deg']:.1f} | {r['sat_front_at_limit']*100:.0f}% / {r['sat_rear_at_limit']*100:.0f}% | {r['limit_balance']} |"
        )

    lines.extend(["", "## 2. Low-Speed Tight Corner Check (25 km/h)", ""])
    for ls in raw.get("low_speed_ackermann", []):
        lines.append(
            f"- **R = {ls['radius_m']:.1f} m** (`Ay = {ls['ay_tgt_g']:.2f} g`): `delta_trim = {ls['delta_trim_deg']:.1f} deg` vs kinematic Ackermann `{ls['ackermann_kin_deg']:.1f} deg` (understeer delta `{ls['understeer_delta_deg']:+.1f} deg`, Sat F/R `{ls['sat_front']*100:.0f}% / {ls['sat_rear']*100:.0f}%`)."
        )

    tv_65 = next(r for r in analyzed if r["v_kmh"] == 65.0 and r["controller"] == "TV4_AUTOCROSS")
    op_65 = next(r for r in analyzed if r["v_kmh"] == 65.0 and r["controller"] == "OPEN_4WD")
    stick_ratio_65 = raw["max_steer_deg"] / max(abs(tv_65["trim_delta_deg"]), 1.0)
    lines.extend([
        "",
        "## 3. Technical Verdict",
        "",
        f"- **Physical stability (`dCN/dbeta`):** At 65 km/h, `OPEN_4WD` has `dCN/dbeta = {op_65['stability_index_1g_per_deg']:+.4f}/deg` (@1g) and `{op_65['stability_index_lim_per_deg']:+.4f}/deg` (@lim), while `TV4_AUTOCROSS` has `{tv_65['stability_index_1g_per_deg']:+.4f}/deg` (@1g) and `{tv_65['stability_index_lim_per_deg']:+.4f}/deg` (@lim) with `{tv_65['limit_balance']}`.",
        f"- **Steer authority mismatch:** At 65 km/h the car reaches its lateral limit at `delta_trim = {abs(tv_65['trim_delta_deg']):.1f} deg`, whereas an unconditioned gamepad stick commands up to `{raw['max_steer_deg']:.1f} deg` (`{stick_ratio_65:.1f}x` the limit steer angle) in ~`0.125 s` (`manual_steer_rate = 8.0 /s`), injecting massive front slip angle transients.",
    ])
    rep_path = RESULTS_DIR / "REPORTE_YMD_TeR27.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rep_path


def main() -> None:
    raw = ensure_raw_data(force_probe="--probe" in sys.argv)
    betas = np.array(raw["beta_grid_deg"], dtype=float)
    deltas = np.array(raw["delta_grid_deg"], dtype=float)

    analyzed = [analyze_sweep(sw, betas, deltas) for sw in raw["sweeps"]]
    summary = {
        "vehicle_id": raw["vehicle_id"],
        "max_steer_deg": raw["max_steer_deg"],
        "low_speed_ackermann": raw.get("low_speed_ackermann", []),
        "sweeps": [{k: v for k, v in r.items() if k not in ("grid", "trim_pts")} for r in analyzed],
    }
    (RESULTS_DIR / "ymd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    p1 = plot_ymd_panels(analyzed, betas, deltas)
    p2 = plot_indices_and_saturation(analyzed, float(raw["max_steer_deg"]))
    rep = write_report(raw, analyzed)

    print("YMD Summary written ->", RESULTS_DIR / "ymd_summary.json")
    print("Plots written ->", p1.name, ",", p2.name)
    print("Report written ->", rep)
    print(rep.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
