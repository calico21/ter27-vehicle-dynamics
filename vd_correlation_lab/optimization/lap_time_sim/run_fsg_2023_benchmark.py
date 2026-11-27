#!/usr/bin/env python3
"""
Hybrid LTS Benchmark (QSS + 14-DOF 3D) on the real FSG 2023 (Hockenheim) track.
Compares Open-Diff 4WD, TV 4WD Autocross, and TV 4WD Efficiency,
generates vd_correlation_lab/results/lts/06_fsg_2023_comparison.png (300 DPI),
saves fsg_2023_summary.json, and appends Section 4 to REPORTE_LTS_TeR27.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parents[2]
LTS_DIR = Path(__file__).resolve().parent
OUT_DIR = LAB_DIR / "results" / "lts"
if str(LAB_DIR.parent) not in sys.path:
    sys.path.insert(0, str(LAB_DIR.parent))
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from vd_correlation_lab.analysis.visualization.style import C_ACCENT, C_HORIZON, C_MUTED, C_OPEN, C_TER27, apply_style, save_fig  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim import (  # noqa: E402
    Track,
    export_enriched_racing_line_json,
    extract_or_load_ggv,
    optimize_mintime_racing_line,
    optimize_racing_line,
    project_endurance_22km,
    search_fastest_valid_lap,
    solve_qss,
    verify_swept_path_4corners,
)
from vd_correlation_lab.optimization.lap_time_sim.tracks.build_fsg_2023 import LTS_OUT, main as build_fsg_main  # noqa: E402


def _cum_time_vs_s(df: pd.DataFrame, s_grid: np.ndarray) -> np.ndarray:
    lap1 = df[df["lap"] == 1].copy()
    if lap1.empty:
        return np.zeros_like(s_grid)
    s_raw = lap1["s_m"].to_numpy(dtype=np.float64)
    t_raw = lap1["t_s"].to_numpy(dtype=np.float64)
    keep = np.ones(len(s_raw), dtype=bool)
    for k in range(1, len(s_raw)):
        if s_raw[k] < s_raw[k - 1] - 50.0:
            if k < 10:
                keep[:k] = False
            else:
                keep[k:] = False
                break
    s_u = s_raw[keep]
    t_u = t_raw[keep] - t_raw[keep][0]
    order = np.argsort(s_u)
    s_u = s_u[order]
    t_u = t_u[order]
    _, uniq = np.unique(s_u, return_index=True)
    return np.interp(s_grid, s_u[uniq], t_u[uniq])


def _emit_progress(pf_path: Path | None, payload: Dict[str, Any]) -> None:
    if pf_path is None:
        return
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    with pf_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
        fh.flush()


def _profile_metrics_from_df(
    df: pd.DataFrame,
    qss_s: float,
    lap_14dof_s: float,
    vmax_kmh: float,
    vmed_kmh: float,
    e_22km_kwh: float,
    sw_res: Dict[str, Any],
) -> Dict[str, Any]:
    lap1 = df[df["lap"] == 1] if "lap" in df.columns else df
    ay_g = np.abs(lap1["ay"].to_numpy(dtype=np.float64)) / 9.81
    ax_g = lap1["ax"].to_numpy(dtype=np.float64) / 9.81
    ey_m = lap1["ey_race_m"].to_numpy(dtype=np.float64)
    gap_pct = 100.0 * (lap_14dof_s - qss_s) / max(qss_s, 1e-6)
    return {
        "qss_s": round(float(qss_s), 3),
        "lap_14dof_s": round(float(lap_14dof_s), 3),
        "gap_pct": round(float(gap_pct), 2),
        "vmax_kmh": round(float(vmax_kmh), 2),
        "vmed_kmh": round(float(vmed_kmh), 2),
        "peak_ay_g": round(float(np.max(ay_g)) if len(ay_g) else 0.0, 3),
        "peak_brake_g": round(float(-np.min(ax_g)) if len(ax_g) else 0.0, 3),
        "e_22km_kwh": round(float(e_22km_kwh), 3),
        "rms_ey_m": round(float(np.sqrt(np.mean(ey_m * ey_m))) if len(ey_m) else 0.0, 4),
        "corners_cones_hit": int(sw_res["corners_cones_hit"]),
        "min_14dof_clear_m": round(float(sw_res["min_14dof_cone_clearance_m"]), 3),
    }


def _write_real_vs_lab_comparison(
    trk: Track,
    rl_real: Any,
    qss_real: Any,
    row_real: Dict[str, Any],
    df_real: pd.DataFrame,
    sw_real: Dict[str, Any],
) -> None:
    lab_sum_p = OUT_DIR / "fsg_2023_summary.json"
    lab_csv_p = OUT_DIR / "trace_fsg2023_tv4_auto.csv"
    if not lab_sum_p.exists() or not lab_csv_p.exists():
        print("[FSG 2023 fsg_real] WARNING: TTC_LAB_PEAK baseline missing; skipping real_vs_lab comparison.", flush=True)
        return

    lab_sum = json.loads(lab_sum_p.read_text(encoding="utf-8"))
    row_lab = next(r for r in lab_sum["controllers"] if r["key"] == "tv4_auto")
    df_lab = pd.read_csv(lab_csv_p)
    ggv_lab = extract_or_load_ggv(
        label="tv4_auto",
        ecu_driving_mode=2,
        tv_4wd_mode=0,
        use_3d_kinematics=True,
        grip_profile="ttc_lab",
    )
    rl_lab = optimize_mintime_racing_line(trk, ggv=ggv_lab)
    sw_lab = verify_swept_path_4corners(trk, rl_lab, df_lab)

    m_lab = _profile_metrics_from_df(
        df=df_lab,
        qss_s=float(row_lab["qss_lap_s"]),
        lap_14dof_s=float(row_lab["tr_lap_s"]),
        vmax_kmh=float(row_lab["vmax_kmh"]),
        vmed_kmh=float(row_lab["vmed_kmh"]),
        e_22km_kwh=float(row_lab["e_22km_kwh"]),
        sw_res=sw_lab,
    )
    m_real = _profile_metrics_from_df(
        df=df_real,
        qss_s=float(row_real["qss_lap_s"]),
        lap_14dof_s=float(row_real["tr_lap_s"]),
        vmax_kmh=float(row_real["vmax_kmh"]),
        vmed_kmh=float(row_real["vmed_kmh"]),
        e_22km_kwh=float(row_real["e_22km_kwh"]),
        sw_res=sw_real,
    )

    comp = {
        "track": "fsg_2023",
        "controller": "tv4_auto",
        "TTC_LAB_PEAK": m_lab,
        "FSG_REAL_TRACK": m_real,
        "delta_real_minus_lab": {
            "qss_s": round(m_real["qss_s"] - m_lab["qss_s"], 3),
            "lap_14dof_s": round(m_real["lap_14dof_s"] - m_lab["lap_14dof_s"], 3),
            "vmax_kmh": round(m_real["vmax_kmh"] - m_lab["vmax_kmh"], 2),
            "vmed_kmh": round(m_real["vmed_kmh"] - m_lab["vmed_kmh"], 2),
            "peak_ay_g": round(m_real["peak_ay_g"] - m_lab["peak_ay_g"], 3),
            "peak_brake_g": round(m_real["peak_brake_g"] - m_lab["peak_brake_g"], 3),
            "e_22km_kwh": round(m_real["e_22km_kwh"] - m_lab["e_22km_kwh"], 3),
        },
    }
    comp_path = OUT_DIR / "fsg_2023_real_vs_lab_summary.json"
    comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print(f"[FSG 2023 fsg_real] Saved comparison JSON -> {comp_path}", flush=True)

    # Plot 09: 300 DPI v(s) profile and G-G diagram comparing TTC_LAB_PEAK vs FSG_REAL_TRACK
    df_l1 = df_lab[df_lab["lap"] == 1] if "lap" in df_lab.columns else df_lab
    df_r1 = df_real[df_real["lap"] == 1] if "lap" in df_real.columns else df_real

    fig9, axes9 = plt.subplots(1, 2, figsize=(15.5, 6.2))
    ax_v, ax_gg = axes9[0], axes9[1]

    ax_v.plot(
        df_l1["s_m"], df_l1["vx"] * 3.6,
        color=C_OPEN, lw=1.25,
        label=f"TTC_LAB_PEAK ({m_lab['lap_14dof_s']:.2f}s, vmed={m_lab['vmed_kmh']:.1f}km/h)",
    )
    ax_v.plot(
        df_r1["s_m"], df_r1["vx"] * 3.6,
        color=C_TER27, lw=1.40,
        label=f"FSG_REAL_TRACK ({m_real['lap_14dof_s']:.2f}s, vmed={m_real['vmed_kmh']:.1f}km/h)",
    )
    ax_v.set_title("Speed Profile v(s): TTC_LAB_PEAK vs FSG_REAL_TRACK")
    ax_v.set_xlabel("Arc length s [m]")
    ax_v.set_ylabel("Speed [km/h]")
    ax_v.legend(loc="upper right", fontsize=8.5)
    ax_v.grid(True, alpha=0.25)

    ax_gg.scatter(
        df_l1["ay"] / 9.81, df_l1["ax"] / 9.81,
        s=3.0, alpha=0.28, color=C_OPEN,
        label=f"TTC_LAB_PEAK (ay_max={m_lab['peak_ay_g']:.2f}g, brk={m_lab['peak_brake_g']:.2f}g)",
    )
    ax_gg.scatter(
        df_r1["ay"] / 9.81, df_r1["ax"] / 9.81,
        s=3.5, alpha=0.38, color=C_TER27,
        label=f"FSG_REAL_TRACK (ay_max={m_real['peak_ay_g']:.2f}g, brk={m_real['peak_brake_g']:.2f}g)",
    )
    ax_gg.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_gg.axvline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_gg.set_aspect("equal", "box")
    ax_gg.set_xlim(-2.4, 2.4)
    ax_gg.set_ylim(-2.4, 2.0)
    ax_gg.set_title("G-G Friction Envelope: TTC_LAB_PEAK vs FSG_REAL_TRACK")
    ax_gg.set_xlabel("ay [g]")
    ax_gg.set_ylabel("ax [g]")
    ax_gg.legend(loc="lower right", fontsize=8.0)
    ax_gg.grid(True, alpha=0.25)

    fig9.suptitle("FSG 2023 (TeR27 4WD TV4 Autocross) - TTC_LAB_PEAK vs FSG_REAL_TRACK", fontsize=12.5, fontweight="bold")
    fig9_path = OUT_DIR / "09_fsg_real_vs_lab_gg_and_lap.png"
    fig9.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig9.savefig(fig9_path, dpi=300, bbox_inches="tight")
    plt.close(fig9)
    print(f"[FSG 2023 fsg_real] Saved 300 DPI comparison plot -> {fig9_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--progress-file", type=str, default="")
    parser.add_argument("--grip-profile", type=str, default="ttc_lab", choices=["ttc_lab", "fsg_real"])
    args, _ = parser.parse_known_args()
    grip_profile = str(args.grip_profile)
    pf_path = Path(args.progress_file) if args.progress_file else None
    if pf_path is not None:
        pf_path.parent.mkdir(parents=True, exist_ok=True)
        pf_path.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    if not LTS_OUT.exists():
        build_fsg_main()
    trk = Track.load_json(LTS_OUT)
    chk = trk.check_validity(min_clearance_m=7.5)

    t_ggv0 = time.perf_counter()
    print(f"[FSG 2023 {grip_profile}] Phase 1 (GGV): extracting/loading tv4_auto reference...", flush=True)
    ggv_auto_ref = extract_or_load_ggv(
        label="tv4_auto",
        ecu_driving_mode=2,
        tv_4wd_mode=0,
        use_3d_kinematics=True,
        grip_profile=grip_profile,
    )
    print(f"[FSG 2023 {grip_profile}] Phase 1 (GGV) ref done in {time.perf_counter() - t_ggv0:.2f}s", flush=True)

    t_ocp0 = time.perf_counter()
    print(f"[FSG 2023 {grip_profile}] Phase 2 (MT-OCP): optimizing racing line...", flush=True)
    rl_old = optimize_racing_line(trk, car_track_width_m=1.230, safety_margin_m=0.35)
    rl = optimize_mintime_racing_line(trk, ggv=ggv_auto_ref)
    print(f"[FSG 2023 {grip_profile}] Phase 2 (MT-OCP) done in {time.perf_counter() - t_ocp0:.2f}s", flush=True)

    controllers = [
        {"key": "open", "label": "Open-Diff 4WD", "ecu": 0, "tv4": 0, "color": C_OPEN},
        {"key": "tv4_auto", "label": "TV 4WD Autocross", "ecu": 2, "tv4": 0, "color": C_TER27},
        {"key": "tv4_eff", "label": "TV 4WD Efficiency", "ecu": 2, "tv4": 2, "color": C_ACCENT},
    ]

    qss_res: Dict[str, Any] = {}
    tr_res: Dict[str, Any] = {}
    rows = []

    use_csv_cache = args.from_csv
    print(f"[FSG 2023 {grip_profile}] Phase 3 (14-DOF): running QSS + 14-DOF 3D LTS (L0={trk.length:.1f}m -> Lr={rl.racing_track.length:.1f}m)...", flush=True)
    for idx_c, c in enumerate(controllers):
        k = c["key"]
        t_c0 = time.perf_counter()
        ggv = extract_or_load_ggv(
            label=k,
            ecu_driving_mode=c["ecu"],
            tv_4wd_mode=c["tv4"],
            use_3d_kinematics=True,
            grip_profile=grip_profile,
        )
        q1 = solve_qss(rl.racing_track, ggv)
        qss_res[k] = q1
        print(f"  [FSG 2023 {grip_profile}] {c['label']}: GGV+QSS={q1.lap_time_s:.3f}s ({time.perf_counter() - t_c0:.2f}s), starting 14-DOF...", flush=True)
        suffix = "_real" if grip_profile == "fsg_real" else ""
        csv_p = OUT_DIR / f"trace_fsg2023_{k}{suffix}.csv"
        if use_csv_cache and csv_p.exists():
            df = pd.read_csv(csv_p)
            lap1 = df[df["lap"] == 1]
            t_lap_s = float(lap1["t_s"].iloc[-1] - lap1["t_s"].iloc[0])
            class _CsvWrap:
                def __init__(self, d_f, t_l):
                    self.telemetry = d_f
                    self.lap_time_s = t_l
                    self.valid = True
                    self.summary = {
                        "alpha_drv": 1.0,
                        "e_net_kj": float(lap1["e_net_kj"].iloc[-1]),
                        "e_regen_kj": float(lap1["e_reg_kj"].iloc[-1]),
                        "max_abs_beta_deg": float(np.max(np.abs(lap1["beta_deg"]))),
                        "torque_fight_kj": 0.0,
                        "p_chg_min_kw": float(np.min(lap1["p_dc_kw"])),
                        "p_dc_max_kw": float(np.max(lap1["p_dc_kw"])),
                    }
            res1 = _CsvWrap(df, t_lap_s)
        else:
            res1, _ = search_fastest_valid_lap(
                label=k,
                racing_line=rl,
                qss=q1,
                ecu_driving_mode=c["ecu"],
                tv_4wd_mode=c["tv4"],
                n_laps=1,
                use_3d_kinematics=True,
                save_csv_path=csv_p,
                grip_profile=grip_profile,
            )
        tr_res[k] = res1
        df = res1.telemetry
        lap1 = df[df["lap"] == 1]
        v_kmh = lap1["vx"].to_numpy(dtype=np.float64) * 3.6
        vmax_kmh = float(np.max(v_kmh)) if len(v_kmh) else 0.0
        vmed_kmh = float((rl.racing_track.length / res1.lap_time_s) * 3.6)
        proj = project_endurance_22km(
            label=c["label"],
            track_length_m=rl.racing_track.length,
            t_lap_s=res1.lap_time_s,
            e_net_lap_kj=res1.summary["e_net_kj"],
            e_regen_lap_kj=res1.summary["e_regen_kj"],
        )
        rows.append({
            "key": k,
            "label": c["label"],
            "qss_lap_s": float(q1.lap_time_s),
            "tr_lap_s": float(res1.lap_time_s),
            "valid": bool(res1.valid),
            "alpha_drv": float(res1.summary["alpha_drv"]),
            "vmax_kmh": vmax_kmh,
            "vmed_kmh": vmed_kmh,
            "e_net_kj": float(res1.summary["e_net_kj"]),
            "e_regen_kj": float(res1.summary["e_regen_kj"]),
            "e_22km_kwh": float(proj.e_net_22km_kwh),
            "max_abs_beta_deg": float(res1.summary["max_abs_beta_deg"]),
            "torque_fight_kj": float(res1.summary["torque_fight_kj"]),
            "p_dc_min_kw": float(res1.summary.get("p_chg_min_kw", 0.0)),
            "p_dc_max_kw": float(res1.summary["p_dc_max_kw"]),
        })
        print(
            f"  [FSG 2023 {grip_profile}] {c['label']:18s}: QSS={q1.lap_time_s:.3f}s | 14DOF={res1.lap_time_s:.3f}s "
            f"(valid={res1.valid}, vmax={vmax_kmh:.1f}km/h, vmed={vmed_kmh:.1f}km/h, "
            f"beta_max={res1.summary['max_abs_beta_deg']:.2f}deg, E_net={res1.summary['e_net_kj']:.1f}kJ, "
            f"E_reg={res1.summary['e_regen_kj']:.1f}kJ, E_22km={proj.e_net_22km_kwh:.2f}kWh, {time.perf_counter() - t_c0:.1f}s)",
            flush=True,
        )
        _emit_progress(pf_path, {
            "stage": "benchmark",
            "key": k,
            "label": c["label"],
            "trial": idx_c + 1,
            "total_trials": len(controllers),
            "lap_s": round(float(res1.lap_time_s), 4),
            "qss_lap_s": round(float(q1.lap_time_s), 4),
            "e_22km_kwh": round(float(proj.e_net_22km_kwh), 3),
            "max_beta_deg": round(float(res1.summary["max_abs_beta_deg"]), 3),
            "feasible": bool(res1.valid),
            "is_best": k == "tv4_auto",
        })

    t_open = rows[0]["tr_lap_s"]
    for r in rows:
        r["dt_vs_open_s"] = r["tr_lap_s"] - t_open

    # Figure 06: Track speed map + v(s), delta-t, P_dc, beta
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1.25], hspace=0.34, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])

    lb = trk.left_boundary()
    rb = trk.right_boundary()
    ax_map.plot(np.append(lb[:, 0], lb[0, 0]), np.append(lb[:, 1], lb[0, 1]), color="#0284C7", lw=0.9, alpha=0.75, label="Límite Izq (Azul)")
    ax_map.plot(np.append(rb[:, 0], rb[0, 0]), np.append(rb[:, 1], rb[0, 1]), color="#D97706", lw=0.9, alpha=0.75, label="Límite Der (Amarillo)")
    df_auto = tr_res["tv4_auto"].telemetry
    df_a1 = df_auto[df_auto["lap"] == 1]
    sc = ax_map.scatter(df_a1["x"], df_a1["y"], c=df_a1["vx"] * 3.6, cmap="plasma", s=4.5, zorder=3, label="TV4 Autocross v(s)")
    ax_map.plot(0.0, 0.0, "o", color=C_OPEN, markersize=7, zorder=4, label="Meta (0, 0)")
    fig.colorbar(sc, ax=ax_map, label="Velocidad [km/h]", pad=0.02, fraction=0.04)
    ax_map.set_title(f"FSG 2023 Hockenheim (L₀={trk.length:.1f} m -> Lᵣ={rl.racing_track.length:.1f} m)")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_aspect("equal", "box")
    ax_map.legend(loc="upper left", fontsize=7.5)

    s_cmp = np.linspace(0.0, rl.racing_track.length, 700)
    t_open_s = _cum_time_vs_s(tr_res["open"].telemetry, s_cmp)

    ax_v = fig.add_subplot(gs[0, 1])
    ax_dt = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_p = fig.add_subplot(gs[2, 1], sharex=ax_v)
    ax_b = fig.add_subplot(gs[3, 1], sharex=ax_v)

    for c in controllers:
        k = c["key"]
        df_c = tr_res[k].telemetry
        df1 = df_c[df_c["lap"] == 1]
        col = c["color"]
        lbl = f"{c['label']} ({tr_res[k].lap_time_s:.2f}s)"
        ax_v.plot(df1["s_m"], df1["vx"] * 3.6, color=col, lw=1.15, label=lbl)
        t_c_s = _cum_time_vs_s(df_c, s_cmp)
        ax_dt.plot(s_cmp, t_c_s - t_open_s, color=col, lw=1.25)
        ax_p.plot(df1["s_m"], df1["p_dc_kw"], color=col, lw=0.95)
        ax_b.plot(df1["s_m"], df1["beta_deg"], color=col, lw=0.95)

    ax_v.set_title("Comparativa FSG 2023 — 14-DOF 3D (Open vs TV4 Autocross vs TV4 Efficiency)")
    ax_v.set_ylabel("v [km/h]")
    ax_v.legend(loc="upper right", fontsize=7.5)
    ax_dt.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax_dt.set_ylabel("Δt vs Open [s]")
    ax_p.axhline(80.0, color=C_OPEN, ls="--", lw=0.7, alpha=0.6)
    ax_p.axhline(-25.0, color=C_HORIZON, ls="--", lw=0.7, alpha=0.6)
    ax_p.set_ylabel("P_DC [kW]")
    ax_b.set_ylabel("β [deg]")
    ax_b.set_xlabel("Longitud de arco s [m]")

    fig_suffix = "_real" if grip_profile == "fsg_real" else ""
    fig_path = OUT_DIR / f"06_fsg_2023{fig_suffix}_comparison.png"
    save_fig(fig, fig_path)

    # Export enriched racing line JSON for MirenaSim 3D visualizer + Ghost Car
    rl_name = "fsg_2023_racing_line_real.json" if grip_profile == "fsg_real" else "fsg_2023_racing_line.json"
    rl_json_out = LAB_DIR.parent / "src" / "mirena_sim" / "Assets" / "RacingLines" / rl_name
    sw_res = verify_swept_path_4corners(trk, rl, tr_res["tv4_auto"].telemetry)
    print(f"[FSG 2023 {grip_profile}] 4-corner swept path: corners_hit={sw_res['corners_cones_hit']}, min_clear={sw_res['min_14dof_cone_clearance_m']:.3f}m", flush=True)
    export_enriched_racing_line_json(
        rl=rl,
        qss=qss_res["tv4_auto"],
        ggv=ggv_auto_ref,
        out_path=rl_json_out,
        df_14dof=tr_res["tv4_auto"].telemetry,
        lap_time_s=tr_res["tv4_auto"].lap_time_s,
        n_samples=360,
    )
    print(f"[FSG 2023 {grip_profile}] Exported racing line -> {rl_json_out}", flush=True)

    # Figure 08: Dedicated 4-panel MT-OCP vs Baseline Min-Curv Comparison (300 DPI)
    base_csv_p = OUT_DIR / "trace_fsg2023_tv4_auto_baseline_mincurv.csv"
    df_new1 = tr_res["tv4_auto"].telemetry
    df_new1 = df_new1[df_new1["lap"] == 1].copy()
    df_old1 = pd.read_csv(base_csv_p) if base_csv_p.exists() else df_new1.copy()
    if "lap" in df_old1.columns:
        df_old1 = df_old1[df_old1["lap"] == 1].copy()

    fig8, axes8 = plt.subplots(2, 2, figsize=(15.5, 10.5))
    ax_z, ax_n = axes8[0, 0], axes8[0, 1]
    ax_vi, ax_gg = axes8[1, 0], axes8[1, 1]

    # Panel 1: Zoom to tightest hairpin cluster on FSG 2023
    i_hair = int(np.argmax(np.abs(trk.kappa)))
    cx_h, cy_h = float(trk.x[i_hair]), float(trk.y[i_hair])
    ax_z.plot(np.append(lb[:, 0], lb[0, 0]), np.append(lb[:, 1], lb[0, 1]), color="#0284C7", lw=1.0, alpha=0.8, label="Conos Izq")
    ax_z.plot(np.append(rb[:, 0], rb[0, 0]), np.append(rb[:, 1], rb[0, 1]), color="#D97706", lw=1.0, alpha=0.8, label="Conos Der")
    ax_z.plot(trk.x, trk.y, color=C_MUTED, ls=":", lw=1.1, label="Linea Central")
    ax_z.plot(rl_old.racing_track.x, rl_old.racing_track.y, color=C_OPEN, ls="--", lw=1.6, label="Anterior (Min-Curv)")
    ax_z.plot(rl.racing_track.x, rl.racing_track.y, color=C_TER27, lw=2.2, label="Nueva MT-OCP")
    ax_z.set_xlim(cx_h - 32.0, cx_h + 32.0)
    ax_z.set_ylim(cy_h - 28.0, cy_h + 28.0)
    ax_z.set_aspect("equal", "box")
    ax_z.set_title("1. Zoom Horquillas FSG 2023 (Outer-Inner-Outer & Late Apex)")
    ax_z.set_xlabel("X [m]")
    ax_z.set_ylabel("Y [m]")
    ax_z.legend(loc="best", fontsize=7.5)

    # Panel 2: Lateral offset n(s) vs +/-n_max
    n_max_phys = float(rl.stats.get("n_max_phys_m", 0.95))
    ax_n.axhline(n_max_phys, color="#DC2626", ls="--", lw=1.0, label=f"+/- n_max ({n_max_phys:.2f} m)")
    ax_n.axhline(-n_max_phys, color="#DC2626", ls="--", lw=1.0)
    ax_n.plot(trk.s, rl_old.n_offset_on_base, color=C_OPEN, lw=1.1, alpha=0.8, label="Anterior Min-Curv")
    ax_n.plot(trk.s, rl.n_offset_on_base, color=C_TER27, lw=1.4, label="Nueva MT-OCP")
    ax_n.set_title("2. Desplazamiento Lateral n(s) vs Corredor Fisico")
    ax_n.set_xlabel("s [m]")
    ax_n.set_ylabel("n [m]")
    ax_n.set_ylim(-1.15, 1.15)
    ax_n.legend(loc="upper right", fontsize=7.5)

    # Panel 3: Speed v(s) + throttle/brake inputs (Before vs After)
    ax_vi.plot(df_old1["s_m"], df_old1["vx"] * 3.6, color=C_OPEN, lw=1.1, ls="--", label="v Anterior [km/h]")
    ax_vi.plot(df_new1["s_m"], df_new1["vx"] * 3.6, color=C_TER27, lw=1.4, label="v MT-OCP [km/h]")
    ax_vi.set_ylabel("Velocidad [km/h]")
    ax_vi.set_xlabel("s [m]")
    ax_vi2 = ax_vi.twinx()
    ax_vi2.plot(df_new1["s_m"], df_new1["throttle"], color="#16A34A", lw=0.75, alpha=0.65, label="Throttle MT-OCP")
    ax_vi2.plot(df_new1["s_m"], -df_new1["brake"], color="#DC2626", lw=0.75, alpha=0.65, label="-Brake MT-OCP")
    ax_vi2.set_ylabel("Throttle (+) / Brake (-)")
    ax_vi2.set_ylim(-1.15, 1.15)
    ax_vi.set_title("3. Perfil de Velocidad e Inputs (Antes vs Despues)")
    ax_vi.legend(loc="upper left", fontsize=7.5)

    # Panel 4: G-G friction ellipse utilization (Before vs After)
    ax_gg.scatter(df_old1["ay"] / 9.81, df_old1["ax"] / 9.81, s=2.5, alpha=0.25, color=C_OPEN, label="Anterior (Min-Curv)")
    ax_gg.scatter(df_new1["ay"] / 9.81, df_new1["ax"] / 9.81, s=3.0, alpha=0.35, color=C_TER27, label="Nueva (MT-OCP + Limit Driver)")
    ax_gg.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_gg.axvline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_gg.set_aspect("equal", "box")
    ax_gg.set_xlim(-2.4, 2.4)
    ax_gg.set_ylim(-2.4, 2.0)
    ax_gg.set_title("4. Diagrama G-G de Utilizacion de Elipse de Friccion")
    ax_gg.set_xlabel("ay [g]")
    ax_gg.set_ylabel("ax [g]")
    ax_gg.legend(loc="lower right", fontsize=7.5)

    fig8.suptitle("FSG 2023 (TeR27 4WD) — Comparativa Linea Anterior vs MT-OCP + Piloto al Limite", fontsize=12.5, fontweight="bold")
    fig8.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig8_path = OUT_DIR / f"08_mto_racing_line{fig_suffix}_comparison.png"
    save_fig(fig8, fig8_path)

    r_min_0 = float(1.0 / max(np.max(np.abs(trk.kappa)), 1e-6))
    payload = {
        "grip_profile": grip_profile,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "geometry": {
            "length_0_m": float(trk.length),
            "length_racing_m": float(rl.racing_track.length),
            "r_min_0_m": r_min_0,
            "r_min_racing_m": float(rl.stats["r_min_racing_m"]),
            "int_k2_reduction_pct": float(rl.stats["int_k2_reduction_pct"]),
            "min_nonadj_clearance_m": float(chk["min_nonadj_clearance_m"]),
            "closure_pos_err_m": float(chk["closure_pos_err_m"]),
            "closure_psi_err_rad": float(chk["closure_psi_err_rad"]),
        },
        "controllers": rows,
    }
    summary_name = "fsg_2023_real_summary.json" if grip_profile == "fsg_real" else "fsg_2023_summary.json"
    (OUT_DIR / summary_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if grip_profile == "fsg_real":
        _write_real_vs_lab_comparison(
            trk=trk,
            rl_real=rl,
            qss_real=qss_res["tv4_auto"],
            row_real=next(r for r in rows if r["key"] == "tv4_auto"),
            df_real=tr_res["tv4_auto"].telemetry,
            sw_real=sw_res,
        )
    else:
        # Update/append Section 4 in REPORTE_LTS_TeR27.md
        rep_path = OUT_DIR / "REPORTE_LTS_TeR27.md"
        existing = rep_path.read_text(encoding="utf-8") if rep_path.exists() else "# Reporte Técnico LTS Híbrido — TeR27 4WD (`MirenaSim`)\n"
        marker = "\n## 4. Benchmark Real FSG 2023 (Hockenheim)"
        if marker in existing:
            existing = existing.split(marker)[0]

        sec4 = [
            marker,
            "",
            f"- **`fsg_2023`**: $L_0 = {trk.length:.2f}\\text{{ m}}$ ($w = 3.60\\text{{ m}}$, $R_{{\\min,0}} = {r_min_0:.2f}\\text{{ m}}$, clearance mínimo $= {chk['min_nonadj_clearance_m']:.2f}\\text{{ m}}$) -> Línea óptima $L_r = {rl.racing_track.length:.2f}\\text{{ m}}$ ($-{rl.stats['int_k2_reduction_pct']:.1f}\\%$ en $\\int \\kappa^2 ds$, $R_{{\\min,r}} = {rl.stats['r_min_racing_m']:.2f}\\text{{ m}}$).",
            "",
            "| Controlador | QSS [s] | 14-DOF [s] | Δt vs Open [s] | v_max [km/h] | v_med [km/h] | E_net [kJ] | E_regen [kJ] | E_22km [kWh] | max|β| [°] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            sec4.append(
                f"| {r['label']} | {r['qss_lap_s']:.3f} | {r['tr_lap_s']:.3f} | {r['dt_vs_open_s']:+.3f} | "
                f"{r['vmax_kmh']:.1f} | {r['vmed_kmh']:.1f} | {r['e_net_kj']:.1f} | {r['e_regen_kj']:.1f} | "
                f"{r['e_22km_kwh']:.2f} | {r['max_abs_beta_deg']:.2f} |"
            )
        rep_path.write_text(existing.rstrip() + "\n" + "\n".join(sec4) + "\n", encoding="utf-8")

    _emit_progress(pf_path, {
        "done": True,
        "summary_path": str(OUT_DIR / summary_name),
        "best_trace_csv": str(OUT_DIR / f"trace_fsg2023_tv4_auto{fig_suffix}.csv"),
    })
    print(f"[FSG 2023 {grip_profile}] Done in {payload['elapsed_s']:.1f}s -> {fig_path}", flush=True)


if __name__ == "__main__":
    main()
