#!/usr/bin/env python3
"""
Full MirenaSim Hybrid LTS Benchmark & Sensitivity Suite for TeR27 4WD.
Generates all results, FEA load spectra, 300-DPI MoTeC i2 figures, and REPORTE_LTS_TeR27.md
under vd_correlation_lab/results/lts/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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

from vd_correlation_lab.analysis.visualization.style import (
    C_ACCENT,
    C_FL,
    C_FR,
    C_HORIZON,
    C_MUTED,
    C_OPEN,
    C_REAL,
    C_RL,
    C_RR,
    C_TER26,
    C_TER27,
    apply_style,
    save_fig,
)
from vd_correlation_lab.optimization.lap_time_sim import (
    Track,
    compute_fs_scores,
    extract_or_load_ggv,
    optimize_racing_line,
    project_endurance_22km,
    run_transient_lap,
    search_fastest_valid_lap,
    solve_qss,
)
from vd_correlation_lab.optimization.lap_time_sim.tracks.generate_tracks import ensure_all_reference_tracks


def _cum_time_vs_s(df: pd.DataFrame, s_grid: np.ndarray) -> np.ndarray:
    """Interpolate elapsed lap time vs monotonic arc length s on Lap 1."""
    lap1 = df[df["lap"] == 1].copy()
    if lap1.empty:
        return np.zeros_like(s_grid)
    # Unwrap periodic s slightly to ensure strict monotonicity across lap 1
    s_raw = lap1["s_m"].to_numpy(dtype=np.float64)
    t_raw = lap1["t_s"].to_numpy(dtype=np.float64)
    # Drop early wrap tail if > length - 5 at start or < 5 at end
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
    # Remove duplicate s
    _, uniq = np.unique(s_u, return_index=True)
    return np.interp(s_grid, s_u[uniq], t_u[uniq])


def main() -> None:
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # ------------------------------------------------------------------
    # 1. Reference tracks & minimum-curvature racing lines
    # ------------------------------------------------------------------
    tracks = ensure_all_reference_tracks()
    trk_fsg = tracks["fsg_autocross_endurance"]
    trk_mt10 = tracks["mirena_track10"]
    rl_fsg = optimize_racing_line(trk_fsg, car_track_width_m=1.230, safety_margin_m=0.35)
    rl_mt10 = optimize_racing_line(trk_mt10, car_track_width_m=1.230, safety_margin_m=0.35)

    # ------------------------------------------------------------------
    # 2. A/B/C/D Controller Comparison (GGV + QSS + Stage-2 Transient)
    # ------------------------------------------------------------------
    controllers = [
        {"key": "open", "label": "Open-Diff 4WD", "ecu": 0, "tv4": 0, "color": C_OPEN},
        {"key": "v3", "label": "TV V3", "ecu": 1, "tv4": 0, "color": C_HORIZON},
        {"key": "tv4_auto", "label": "TV 4WD Autocross", "ecu": 2, "tv4": 0, "color": C_TER27},
        {"key": "tv4_eff", "label": "TV 4WD Efficiency", "ecu": 2, "tv4": 2, "color": C_ACCENT},
    ]

    ggv_map: Dict[str, Any] = {}
    qss_fsg: Dict[str, Any] = {}
    tr_fsg: Dict[str, Any] = {}
    qss_mt10: Dict[str, Any] = {}
    tr_mt10: Dict[str, Any] = {}
    projections_fsg = []

    print("[LTS] 1/4 Extracting GGV + Running A/B/C/D on fsg_autocross_endurance & mirena_track10...")
    for c in controllers:
        k = c["key"]
        ggv = extract_or_load_ggv(label=k, ecu_driving_mode=c["ecu"], tv_4wd_mode=c["tv4"], use_3d_kinematics=True)
        ggv_map[k] = ggv

        q1 = solve_qss(rl_fsg.racing_track, ggv)
        qss_fsg[k] = q1
        csv1 = OUT_DIR / f"trace_fsg_{k}.csv"
        res1, _ = search_fastest_valid_lap(
            label=k,
            racing_line=rl_fsg,
            qss=q1,
            ecu_driving_mode=c["ecu"],
            tv_4wd_mode=c["tv4"],
            n_laps=1,
            use_3d_kinematics=True,
            save_csv_path=csv1,
        )
        tr_fsg[k] = res1
        projections_fsg.append(
            project_endurance_22km(
                label=c["label"],
                track_length_m=rl_fsg.racing_track.length,
                t_lap_s=res1.lap_time_s,
                e_net_lap_kj=float(res1.summary["e_net_kj"]),
                e_regen_lap_kj=float(res1.summary["e_regen_kj"]),
            )
        )
        print(
            f"  [FSG] {c['label']:18s}: QSS={q1.lap_time_s:.3f}s | 14DOF={res1.lap_time_s:.3f}s "
            f"(valid={res1.valid}, alpha={res1.summary['alpha_drv']:.4f}, beta_max={res1.summary['max_abs_beta_deg']:.2f}deg, "
            f"E_net={res1.summary['e_net_kj']:.1f}kJ, E_reg={res1.summary['e_regen_kj']:.1f}kJ)"
        )

        # Cross-track validation on real Mirena Track10 (3.0 m minimum FS width)
        q2 = solve_qss(rl_mt10.racing_track, ggv)
        qss_mt10[k] = q2
        csv2 = OUT_DIR / f"trace_track10_{k}.csv"
        res2, _ = search_fastest_valid_lap(
            label=f"{k}_t10",
            racing_line=rl_mt10,
            qss=q2,
            ecu_driving_mode=c["ecu"],
            tv_4wd_mode=c["tv4"],
            n_laps=1,
            use_3d_kinematics=True,
            save_csv_path=csv2,
        )
        tr_mt10[k] = res2
        print(
            f"  [T10] {c['label']:18s}: QSS={q2.lap_time_s:.3f}s | 14DOF={res2.lap_time_s:.3f}s "
            f"(valid={res2.valid}, E_net={res2.summary['e_net_kj']:.1f}kJ)"
        )

    fs_scores = compute_fs_scores(projections_fsg)

    # ------------------------------------------------------------------
    # 3. Multi-Lap Continuous Endurance Stress Test (3 Laps: Auto & Eff)
    # ------------------------------------------------------------------
    print("[LTS] 2/4 Running 3-lap continuous Endurance stress tests...")
    multilap_results: Dict[str, Any] = {}
    for k, ecu_m, tv4_m in [("tv4_auto", 2, 0), ("tv4_eff", 2, 2)]:
        alpha_opt = float(tr_fsg[k].summary["alpha_drv"])
        ml_csv = OUT_DIR / f"trace_fsg_{k}_3laps.csv"
        ml_res = run_transient_lap(
            label=f"{k}_3laps",
            racing_line=rl_fsg,
            qss=qss_fsg[k],
            ecu_driving_mode=ecu_m,
            tv_4wd_mode=tv4_m,
            alpha_drv=alpha_opt,
            n_laps=3,
            use_3d_kinematics=True,
            save_csv_path=ml_csv,
        )
        multilap_results[k] = ml_res
        print(
            f"  [3-Laps {k}] valid={ml_res.valid} nan={ml_res.summary['nan_detected']} "
            f"laps={[round(lp['lap_time_s'], 3) for lp in ml_res.summary['per_lap']]}"
        )

    # ------------------------------------------------------------------
    # 4. Design Sensitivity Studies + Efficiency Pareto Frontier
    # ------------------------------------------------------------------
    print("[LTS] 3/4 Running design sensitivities & Efficiency Pareto frontier...")
    alpha_auto_nom = float(tr_fsg["tv4_auto"].summary["alpha_drv"])
    t_nom_tr = tr_fsg["tv4_auto"].lap_time_s
    e_nom_tr = float(tr_fsg["tv4_auto"].summary["e_net_kj"])
    t_nom_qss = qss_fsg["tv4_auto"].lap_time_s

    sens_specs = [
        {"key": "mass_m15", "label": "Mass -15 kg (275 kg)", "mass_kg": 275.0, "cl_a_scale": None, "use_3d": True},
        {"key": "mass_p15", "label": "Mass +15 kg (305 kg)", "mass_kg": 305.0, "cl_a_scale": None, "use_3d": True},
        {"key": "cla_m10", "label": "Aero Cl*A -10%", "mass_kg": None, "cl_a_scale": 0.90, "use_3d": True},
        {"key": "cla_p10", "label": "Aero Cl*A +10%", "mass_kg": None, "cl_a_scale": 1.10, "use_3d": True},
        {"key": "kin_param", "label": "Parametric Susp (no 3D)", "mass_kg": None, "cl_a_scale": None, "use_3d": False},
    ]
    sens_rows: List[Dict[str, Any]] = []
    for sp in sens_specs:
        g_s = extract_or_load_ggv(
            label=f"sens_{sp['key']}",
            ecu_driving_mode=2,
            tv_4wd_mode=0,
            use_3d_kinematics=sp["use_3d"],
            mass_kg=sp["mass_kg"],
            cl_a_scale=sp["cl_a_scale"],
        )
        q_s = solve_qss(rl_fsg.racing_track, g_s)
        # Verify in full 14-DOF transient loop using matching speed margin or fast calibration
        r_s, _ = search_fastest_valid_lap(
            label=f"sens_{sp['key']}",
            racing_line=rl_fsg,
            qss=q_s,
            ecu_driving_mode=2,
            tv_4wd_mode=0,
            n_laps=1,
            use_3d_kinematics=sp["use_3d"],
            mass_kg=sp["mass_kg"],
            cl_a_scale=sp["cl_a_scale"],
        )
        sens_rows.append({
            "key": sp["key"],
            "label": sp["label"],
            "qss_lap_s": q_s.lap_time_s,
            "qss_dt_ms": (q_s.lap_time_s - t_nom_qss) * 1000.0,
            "tr_lap_s": r_s.lap_time_s,
            "tr_dt_ms": (r_s.lap_time_s - t_nom_tr) * 1000.0,
            "tr_valid": r_s.valid,
            "tr_alpha": float(r_s.summary["alpha_drv"]),
            "e_net_kj": float(r_s.summary["e_net_kj"]),
            "e_delta_pct": 100.0 * (float(r_s.summary["e_net_kj"]) - e_nom_tr) / max(e_nom_tr, 1.0),
            "ay_max_22ms": float(g_s.max_ay(22.0)),
        })
        print(
            f"  [Sens {sp['label']:24s}] QSS dT={sens_rows[-1]['qss_dt_ms']:+.0f}ms | "
            f"14DOF dT={sens_rows[-1]['tr_dt_ms']:+.0f}ms | dE={sens_rows[-1]['e_delta_pct']:+.2f}%"
        )

    # Efficiency Pareto grid: p_eff_max_w x mech_brake_start (full 14-DOF verification)
    pareto_rows: List[Dict[str, Any]] = []
    alpha_eff_nom = float(tr_fsg["tv4_eff"].summary["alpha_drv"])
    qss_eff_by_p: Dict[float, Any] = {}
    for p_lim in [40000.0, 47500.0, 55000.0]:
        g_p = extract_or_load_ggv(
            label=f"eff_p{int(p_lim//100)}",
            ecu_driving_mode=2,
            tv_4wd_mode=2,
            use_3d_kinematics=True,
            p_eff_max_w=p_lim,
        )
        qss_eff_by_p[p_lim] = solve_qss(rl_fsg.racing_track, g_p)

    for p_lim in [40000.0, 47500.0, 55000.0]:
        for mbs in [0.65, 0.75, 0.80]:
            r_p = run_transient_lap(
                label=f"pareto_p{int(p_lim//100)}_mbs{int(round(mbs*100))}",
                racing_line=rl_fsg,
                qss=qss_eff_by_p[p_lim],
                ecu_driving_mode=2,
                tv_4wd_mode=2,
                alpha_drv=alpha_eff_nom,
                n_laps=1,
                use_3d_kinematics=True,
                p_eff_max_w=p_lim,
                mech_brake_start=mbs,
            )
            proj = project_endurance_22km(
                label=f"{p_lim/1000:.1f}kW / mbs={mbs:.2f}",
                track_length_m=rl_fsg.racing_track.length,
                t_lap_s=r_p.lap_time_s,
                e_net_lap_kj=float(r_p.summary["e_net_kj"]),
                e_regen_lap_kj=float(r_p.summary["e_regen_kj"]),
            )
            pareto_rows.append({
                "p_eff_max_kw": p_lim / 1000.0,
                "mech_brake_start": mbs,
                "valid": r_p.valid,
                "t_lap_s": r_p.lap_time_s,
                "t_22km_s": proj.t_22km_s,
                "e_net_lap_kj": proj.e_net_lap_kj,
                "e_regen_lap_kj": proj.e_regen_lap_kj,
                "e_net_22km_kwh": proj.e_net_22km_kwh,
                "e_t2_index": proj.e_t2_index,
                "regen_pct": proj.regen_pct,
            })
            print(
                f"  [Pareto P={p_lim/1000:.1f}kW, mbs={mbs:.2f}] t={r_p.lap_time_s:.3f}s "
                f"E_22km={proj.e_net_22km_kwh:.2f}kWh regen={proj.regen_pct:.1f}% E*t^2={proj.e_t2_index/1e6:.3f}e6"
            )

    # ------------------------------------------------------------------
    # 5. Export 3D Suspension Load Spectrum (FEA/Fatigue Ready)
    # ------------------------------------------------------------------
    loads_3d = tr_fsg["tv4_auto"].summary["suspension_loads"]
    corners_names = ["FL", "FR", "RL", "RR"]
    spec_rows = []
    for i, cname in enumerate(corners_names):
        spec_rows.append({
            "corner": cname,
            "fz_peak_n": float(loads_3d["fz_peak_n"][i]),
            "fz_rms_n": float(loads_3d["fz_rms_n"][i]),
            "fx_peak_n": float(loads_3d["fx_peak_n"][i]),
            "fx_rms_n": float(loads_3d["fx_rms_n"][i]),
            "fy_peak_n": float(loads_3d["fy_peak_n"][i]),
            "fy_rms_n": float(loads_3d["fy_rms_n"][i]),
            "dz_peak_mm": float(loads_3d["dz_peak_m"][i]) * 1000.0,
            "dz_rms_mm": float(loads_3d["dz_rms_m"][i]) * 1000.0,
            "arb_force_peak_n": float(loads_3d["farb_peak_n"][i]),
            "arb_force_rms_n": float(loads_3d["farb_rms_n"][i]),
        })
    pd.DataFrame(spec_rows).to_csv(OUT_DIR / "suspension_loads_3d_spectrum.csv", index=False)
    (OUT_DIR / "suspension_loads_3d_spectrum.json").write_text(
        json.dumps(spec_rows, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 6. Generate 5 MoTeC i2 300-DPI Figures
    # ------------------------------------------------------------------
    print("[LTS] 4/4 Rendering 5 MoTeC i2 figures (300 DPI) & REPORTE_LTS_TeR27.md...")
    df_auto = tr_fsg["tv4_auto"].telemetry
    df_open = tr_fsg["open"].telemetry
    df_v3 = tr_fsg["v3"].telemetry
    df_eff = tr_fsg["tv4_eff"].telemetry

    # Figure 01: Track & Racing Line + Speed/Ay Coloring
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax0, ax1 = axes
    lb = trk_fsg.left_boundary()
    rb = trk_fsg.right_boundary()
    ax0.plot(lb[:, 0], lb[:, 1], color=C_HORIZON, lw=0.9, alpha=0.65, label="Límite Izq (Conos Azules)")
    ax0.plot(rb[:, 0], rb[:, 1], color=C_TER27, lw=0.9, alpha=0.65, label="Límite Der (Conos Amarillos)")
    ax0.plot(trk_fsg.x, trk_fsg.y, color=C_MUTED, ls="--", lw=0.9, label="Línea Central")
    sc = ax0.scatter(df_auto["x"], df_auto["y"], c=df_auto["vx"] * 3.6, cmap="turbo", s=3.5, label="Trayectoria 14-DOF (TV4)")
    fig.colorbar(sc, ax=ax0, label="Velocidad [km/h]", pad=0.02, fraction=0.04)
    ax0.set_title(f"FSG Autocross/Endurance ({trk_fsg.length:.0f} m) — Línea Óptima & Velocidad")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")
    ax0.set_aspect("equal", "box")
    ax0.legend(loc="lower right", fontsize=7)

    # Right subplot: lateral acceleration & curvature reduction on FSG
    ax1_top = ax1
    ax1_top.plot(trk_fsg.s, trk_fsg.kappa, color=C_MUTED, lw=1.0, alpha=0.7, label=f"Curvatura Centro (max |κ|={rl_fsg.centerline_max_abs_kappa:.3f} m⁻¹)")
    ax1_top.plot(rl_fsg.racing_track.s, rl_fsg.racing_track.kappa, color=C_HORIZON, lw=1.2, label=f"Línea Óptima (-{rl_fsg.stats['int_k2_reduction_pct']:.1f}% ∫κ²ds)")
    ax1_top.set_xlabel("Longitud de arco s [m]")
    ax1_top.set_ylabel("Curvatura κ [1/m]")
    ax1_top.set_title(f"Perfil de Curvatura & |a_y| Real (14-DOF TV4)")
    ax1_r = ax1_top.twinx()
    ax1_r.plot(df_auto["s_m"], np.abs(df_auto["ay"]) / 9.81, color=C_TER27, lw=0.9, alpha=0.6, label="|ay| 14-DOF [g]")
    ax1_r.set_ylabel("|ay| [g]", color=C_TER27)
    ax1_top.legend(loc="upper right", fontsize=7.5)
    save_fig(fig, OUT_DIR / "01_track_and_racing_line.png")

    # Figure 02: G-G-V Envelope & QSS vs 14-DOF
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    g_auto = ggv_map["tv4_auto"]
    # Left: G-G cloud 14-DOF vs GGV contour at v=14 & 26 m/s
    axes[0].scatter(df_auto["ay"] / 9.81, df_auto["ax"] / 9.81, c=df_auto["vx"] * 3.6, cmap="viridis", s=3.5, alpha=0.55, label="Nube Transitoria 14-DOF")
    for vv, col, ls in [(14.0, C_TER27, "--"), (26.0, C_OPEN, "-")]:
        ay_m = g_auto.max_ay(vv) / 9.81
        ay_half = np.linspace(-ay_m, ay_m, 121)
        ax_top = np.array([g_auto.max_ax(vv, abs(a * 9.81)) / 9.81 for a in ay_half])
        ax_bot = np.array([g_auto.min_ax(vv, abs(a * 9.81)) / 9.81 for a in ay_half[::-1]])
        ay_c = np.concatenate([ay_half, ay_half[::-1], ay_half[:1]])
        ax_c = np.concatenate([ax_top, ax_bot, ax_top[:1]])
        axes[0].plot(ay_c, ax_c, color=col, ls=ls, lw=1.8, label=f"Envolvente GGV @ {vv:.0f} m/s ({ay_m:.2f}g)")
    axes[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    axes[0].axvline(0, color="k", lw=0.6, alpha=0.4)
    axes[0].set_title("Envolvente G-G-V (14-DOF 3D) vs Nube Real de Vuelta")
    axes[0].set_xlabel("Aceleración Lateral ay [g]")
    axes[0].set_ylabel("Aceleración Longitudinal ax [g]")
    axes[0].legend(fontsize=7.5)

    # Right: Velocity trace QSS vs 14-DOF
    q_a = qss_fsg["tv4_auto"]
    axes[1].plot(q_a.s, q_a.v_qss * 3.6, color=C_HORIZON, lw=1.4, ls="--", label=f"Etapa 1 QSS ({q_a.lap_time_s:.2f} s)")
    axes[1].plot(df_auto["s_m"], df_auto["vx"] * 3.6, color=C_TER27, lw=1.2, label=f"Etapa 2 14-DOF ({tr_fsg['tv4_auto'].lap_time_s:.2f} s)")
    axes[1].set_title("Correlación de Velocidad: QSS vs 14-DOF Transitorio (TV4 Auto)")
    axes[1].set_xlabel("Longitud de arco s [m]")
    axes[1].set_ylabel("Velocidad [km/h]")
    axes[1].legend(fontsize=8)
    save_fig(fig, OUT_DIR / "02_ggv_envelope_and_qss_vs_14dof.png")

    # Figure 03: Controller Comparison & Delta-t
    fig, axes = plt.subplots(4, 1, figsize=(13.0, 10.5), sharex=True)
    s_cmp = np.linspace(0.0, rl_fsg.racing_track.length, 600)
    t_open_s = _cum_time_vs_s(df_open, s_cmp)
    for c in controllers:
        df_c = tr_fsg[c["key"]].telemetry
        t_c_s = _cum_time_vs_s(df_c, s_cmp)
        axes[0].plot(df_c["s_m"], df_c["vx"] * 3.6, color=c["color"], lw=1.1, label=f"{c['label']} ({tr_fsg[c['key']].lap_time_s:.2f} s)")
        axes[1].plot(s_cmp, t_c_s - t_open_s, color=c["color"], lw=1.3, label=f"{c['label']} ({tr_fsg[c['key']].lap_time_s - tr_fsg['open'].lap_time_s:+.2f} s)")
        axes[2].plot(df_c["s_m"], df_c["p_dc_kw"], color=c["color"], lw=0.9, alpha=0.85)
        axes[3].plot(df_c["s_m"], df_c["beta_deg"], color=c["color"], lw=0.9, alpha=0.85)
    axes[0].set_ylabel("Vel [km/h]"); axes[0].set_title("Comparativa de Controladores en Vuelta Completa — TeR27 4WD (FSG 1.08 km)")
    axes[0].legend(loc="lower right", ncol=2, fontsize=7.5)
    axes[1].axhline(0, color="k", lw=0.7); axes[1].set_ylabel("Δt vs Open [s]"); axes[1].legend(loc="lower left", ncol=2, fontsize=7.5)
    axes[2].axhline(80.0, color=C_OPEN, ls=":", lw=0.9); axes[2].axhline(-25.0, color=C_ACCENT, ls=":", lw=0.9)
    axes[2].set_ylabel("P_DC [kW]")
    axes[3].set_ylabel("Deriva β [deg]"); axes[3].set_xlabel("Longitud de arco s [m]")
    save_fig(fig, OUT_DIR / "03_controller_comparison_delta_t.png")

    # Figure 04: Multi-Lap Endurance & Efficiency Pareto
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    df_ml_eff = multilap_results["tv4_eff"].telemetry
    df_ml_auto = multilap_results["tv4_auto"].telemetry
    axes[0, 0].plot(df_ml_auto["t_s"], df_ml_auto["soc"] * 100.0, color=C_TER27, lw=1.5, label="Autocross 3 Laps SoC [%]")
    axes[0, 0].plot(df_ml_eff["t_s"], df_ml_eff["soc"] * 100.0, color=C_ACCENT, lw=1.5, label="Efficiency 3 Laps SoC [%]")
    axes[0, 0].set_title("Endurance 3 Vueltas: Evolución SoC")
    axes[0, 0].set_xlabel("Tiempo [s]"); axes[0, 0].set_ylabel("SoC [%]"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(df_ml_auto["t_s"], df_ml_auto["t_surf_max_c"], color=C_TER27, lw=1.2, label="Neumático T_surf max (Auto)")
    axes[0, 1].plot(df_ml_eff["t_s"], df_ml_eff["t_surf_max_c"], color=C_ACCENT, lw=1.2, label="Neumático T_surf max (Eff)")
    axes[0, 1].plot(df_ml_auto["t_s"], df_ml_auto["disc_max_c"], color=C_OPEN, ls="--", lw=1.1, label="Disco Freno T_max (Auto)")
    axes[0, 1].set_title("Evolución Térmica Neumáticos & Frenos (3 Vueltas)")
    axes[0, 1].set_xlabel("Tiempo [s]"); axes[0, 1].set_ylabel("Temperatura [°C]"); axes[0, 1].legend(fontsize=7.5)

    axes[1, 0].plot(df_ml_auto["t_s"], df_ml_auto["ekf_vx"] - df_ml_auto["vx"], color=C_HORIZON, lw=1.0, label="Error EKF vx (Auto)")
    axes[1, 0].plot(df_ml_auto["t_s"], df_ml_auto["mu_hat_min"], color=C_TER27, lw=1.2, label="mu_hat min (Auto)")
    axes[1, 0].set_title("Estabilidad de Observadores ECU TV4 (3 Vueltas)")
    axes[1, 0].set_xlabel("Tiempo [s]"); axes[1, 0].set_ylabel("Error EKF [m/s] / μ_hat"); axes[1, 0].legend(fontsize=8)

    # Bottom-Right: Pareto Frontier Efficiency (zoomed to Pareto band)
    df_par = pd.DataFrame(pareto_rows)
    offsets = {0.65: (-18, 6), 0.75: (5, 5), 0.80: (6, -10)}
    for mbs_v, col, mk in [(0.65, C_MUTED, "s"), (0.75, C_HORIZON, "^"), (0.80, C_ACCENT, "o")]:
        sub = df_par[df_par["mech_brake_start"] == mbs_v]
        axes[1, 1].plot(sub["t_22km_s"], sub["e_net_22km_kwh"], color=col, marker=mk, lw=1.5, ms=7, alpha=0.85, label=f"Pedalera mbs={mbs_v:.2f}")
        if mbs_v == 0.80:
            for _, row in sub.iterrows():
                axes[1, 1].annotate(f"{row['p_eff_max_kw']:.1f} kW", (row["t_22km_s"], row["e_net_22km_kwh"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
    if not df_par.empty:
        e_min = float(df_par["e_net_22km_kwh"].min())
        e_max = float(df_par["e_net_22km_kwh"].max())
        e_span = max(e_max - e_min, 0.15)
        axes[1, 1].set_ylim(e_min - 0.08 * e_span, e_max + 0.14 * e_span)
        t_min = float(df_par["t_22km_s"].min())
        t_max = float(df_par["t_22km_s"].max())
        t_span = max(t_max - t_min, 2.0)
        axes[1, 1].set_xlim(t_min - 0.08 * t_span, t_max + 0.08 * t_span)
    axes[1, 1].set_title("Frontera de Pareto Efficiency (22 km) — Pack HV = 7.2 kWh")
    axes[1, 1].set_xlabel("Tiempo Proyectado 22 km [s]"); axes[1, 1].set_ylabel("Energía Neta 22 km [kWh]")
    axes[1, 1].legend(fontsize=7.5, loc="upper right")
    save_fig(fig, OUT_DIR / "04_endurance_multilap_and_pareto.png")

    # Figure 05: Sensitivity Tornado + 3D Suspension Load Spectra
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    y_pos = np.arange(len(sens_rows))
    dt_vals = [r["tr_dt_ms"] for r in sens_rows]
    bar_cols = [C_ACCENT if v <= 0 else C_OPEN for v in dt_vals]
    axes[0].barh(y_pos, dt_vals, color=bar_cols, alpha=0.85, height=0.58)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([r["label"] for r in sens_rows], fontsize=8.5)
    axes[0].set_xlim(-1400, 1850)
    for yp, r in zip(y_pos, sens_rows):
        v_dt = r["tr_dt_ms"]
        if v_dt >= 0:
            axes[0].text(v_dt + 30, yp, f"{v_dt:+.0f} ms ({r['e_delta_pct']:+.1f}% E)", va="center", ha="left", fontsize=7.5)
        else:
            axes[0].text(v_dt - 30, yp, f"{v_dt:+.0f} ms ({r['e_delta_pct']:+.1f}% E)", va="center", ha="right", fontsize=7.5)
    axes[0].set_title("Sensibilidad 14-DOF de Diseño (Δt_lap vs Nominal)")
    axes[0].set_xlabel("Variación de Tiempo de Vuelta Δt [ms]")

    # Right: Histogram & Peak/RMS of 3D Vertical & Lateral Loads
    x_c = np.arange(4)
    w_b = 0.26
    axes[1].bar(x_c - w_b, [r["fz_peak_n"] for r in spec_rows], width=w_b, color=C_HORIZON, label="Fz Pico [N]")
    axes[1].bar(x_c, [r["fy_peak_n"] for r in spec_rows], width=w_b, color=C_TER27, label="|Fy| Pico [N]")
    axes[1].bar(x_c + w_b, [r["fx_peak_n"] for r in spec_rows], width=w_b, color=C_ACCENT, label="|Fx| Pico [N]")
    axes[1].set_xticks(x_c)
    axes[1].set_xticklabels(corners_names)
    axes[1].set_title("Espectro de Cargas 3D por Esquina (Vuelta Rápida TV4 Auto)")
    axes[1].set_ylabel("Fuerza Pico [N]")
    axes[1].legend(fontsize=8)
    save_fig(fig, OUT_DIR / "05_sensitivity_tornado_and_loads.png")

    # ------------------------------------------------------------------
    # 7. Persist Structured Summary & Markdown Report
    # ------------------------------------------------------------------
    summary_payload = {
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "tracks": {
            "fsg_autocross_endurance": {"base": trk_fsg.check_validity(), "racing_line": rl_fsg.stats},
            "mirena_track10": {"base": trk_mt10.check_validity(), "racing_line": rl_mt10.stats},
        },
        "ggv_tv4_auto": {
            "speeds_ms": ggv_map["tv4_auto"].speeds.tolist(),
            "ay_max_ms2": ggv_map["tv4_auto"].ay_max.tolist(),
            "ax_max_0_ms2": ggv_map["tv4_auto"].ax_max[:, 0].tolist(),
            "ax_min_0_ms2": ggv_map["tv4_auto"].ax_min[:, 0].tolist(),
        },
        "controllers_fsg": [
            {
                "key": c["key"],
                "label": c["label"],
                "qss_lap_s": qss_fsg[c["key"]].lap_time_s,
                "qss_solve_ms": qss_fsg[c["key"]].solve_time_ms,
                "tr_lap_s": tr_fsg[c["key"]].lap_time_s,
                "tr_qss_ratio": tr_fsg[c["key"]].lap_time_s / qss_fsg[c["key"]].lap_time_s,
                "valid": tr_fsg[c["key"]].valid,
                "alpha_drv": tr_fsg[c["key"]].summary["alpha_drv"],
                "max_abs_ey_base_m": tr_fsg[c["key"]].summary["max_abs_ey_base_m"],
                "max_abs_beta_deg": tr_fsg[c["key"]].summary["max_abs_beta_deg"],
                "e_net_kj": tr_fsg[c["key"]].summary["e_net_kj"],
                "e_regen_kj": tr_fsg[c["key"]].summary["e_regen_kj"],
                "torque_fight_kj": tr_fsg[c["key"]].summary["torque_fight_kj"],
                "p_dc_max_kw": tr_fsg[c["key"]].summary["p_dc_max_kw"],
                "p_chg_min_kw": tr_fsg[c["key"]].summary["p_chg_min_kw"],
            }
            for c in controllers
        ],
        "controllers_track10": [
            {
                "key": c["key"],
                "label": c["label"],
                "qss_lap_s": qss_mt10[c["key"]].lap_time_s,
                "tr_lap_s": tr_mt10[c["key"]].lap_time_s,
                "valid": tr_mt10[c["key"]].valid,
                "e_net_kj": tr_mt10[c["key"]].summary["e_net_kj"],
                "e_regen_kj": tr_mt10[c["key"]].summary["e_regen_kj"],
            }
            for c in controllers
        ],
        "fs_scores_22km": fs_scores,
        "multilap_endurance": {
            k: {
                "valid": multilap_results[k].valid,
                "nan_detected": multilap_results[k].summary["nan_detected"],
                "per_lap": multilap_results[k].summary["per_lap"],
            }
            for k in ["tv4_auto", "tv4_eff"]
        },
        "sensitivities": sens_rows,
        "pareto_efficiency": pareto_rows,
        "suspension_loads_spectrum": spec_rows,
    }
    (OUT_DIR / "lts_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    # Render concise engineering markdown report
    md_lines = [
        "# Reporte Técnico LTS Híbrido — TeR27 4WD (`MirenaSim`)\n",
        f"- **Duración total del benchmark:** {summary_payload['elapsed_s']:.1f} s\n",
        "## 1. Trazados y Línea de Carrera de Mínima Curvatura\n",
        f"- **`fsg_autocross_endurance`**: $L_0 = {trk_fsg.length:.1f}\\text{{ m}}$ ($w = 3.60\\text{{ m}}$, $R_{{\\min,0}} = {1/max(abs(trk_fsg.kappa)):.2f}\\text{{ m}}$) -> Línea óptima $L_r = {rl_fsg.racing_track.length:.1f}\\text{{ m}}$ ($-{rl_fsg.stats['int_k2_reduction_pct']:.1f}\\%$ en $\\int \\kappa^2 ds$, $R_{{\\min,r}} = {rl_fsg.stats['r_min_racing_m']:.2f}\\text{{ m}}$).\n",
        f"- **`mirena_track10`** (`track10.json` del repo): $L_0 = {trk_mt10.length:.1f}\\text{{ m}}$ ($w = 3.00\\text{{ m}}$, $R_{{\\min,0}} = {1/max(abs(trk_mt10.kappa)):.2f}\\text{{ m}}$) -> Línea óptima $L_r = {rl_mt10.racing_track.length:.1f}\\text{{ m}}$ ($-{rl_mt10.stats['int_k2_reduction_pct']:.1f}\\%$ en $\\int \\kappa^2 ds$, $R_{{\\min,r}} = {rl_mt10.stats['r_min_racing_m']:.2f}\\text{{ m}}$).\n",
        "\n## 2. Comparativa A/B/C/D en `fsg_autocross_endurance` (14-DOF 3D + Proyección 22 km)\n",
        "| Controlador | QSS [s] | 14-DOF [s] | Δt vs Open | Ratio 14D/QSS | max|β| [°] | E_net [kJ] | E_reg [kJ] | T_22km [s] | E_22km [kWh] | E·t² rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    t_open_lap = tr_fsg["open"].lap_time_s
    for c, fs in zip(controllers, fs_scores):
        k = c["key"]
        md_lines.append(
            f"| {c['label']} | {qss_fsg[k].lap_time_s:.3f} | {tr_fsg[k].lap_time_s:.3f} | "
            f"{tr_fsg[k].lap_time_s - t_open_lap:+.3f} | {tr_fsg[k].lap_time_s / qss_fsg[k].lap_time_s:.4f} | "
            f"{tr_fsg[k].summary['max_abs_beta_deg']:.2f} | {tr_fsg[k].summary['e_net_kj']:.1f} | "
            f"{tr_fsg[k].summary['e_regen_kj']:.1f} | {fs['t_22km_s']:.1f} | {fs['e_net_22km_kwh']:.2f} | {fs['e_t2_rel_best']:.3f} |"
        )
    md_lines.extend([
        "\n## 3. Sensibilidades de Diseño (14-DOF 3D)\n",
        "| Caso | QSS Δt [ms] | 14-DOF Δt [ms] | ΔE [%] | ay_max @ 22 m/s [m/s²] |",
        "|---|---:|---:|---:|---:|",
    ])
    for sr in sens_rows:
        md_lines.append(
            f"| {sr['label']} | {sr['qss_dt_ms']:+.0f} | {sr['tr_dt_ms']:+.0f} | {sr['e_delta_pct']:+.2f} | {sr['ay_max_22ms']:.2f} |"
        )
    (OUT_DIR / "REPORTE_LTS_TeR27.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[LTS] Complete in {summary_payload['elapsed_s']:.1f} s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
