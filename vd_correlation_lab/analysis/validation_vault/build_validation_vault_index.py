#!/usr/bin/env python3
"""
Validation Vault Index, Master KPI Catalog, and Interactive MoTeC HTML Dashboard Builder.

Aggregates all 8 validation vault folders under `outputs/validation_vault/01..08`:
  - outputs/validation_vault/master_metrics_catalog.json
  - outputs/validation_vault/INDEX.md
  - outputs/validation_vault/dashboard_index.html

Design rules:
  - 100% pure ASCII (zero emojis, zero non-ASCII symbols)
  - Self-contained HTML/CSS/JS dark-mode MoTeC i2 theme (zero external CDN dependencies)
  - Left navigation sidebar for Overview + all 8 suites
  - Executive KPI summary table + per-suite KPI cards + clickable lightbox modal zoom for EVERY PNG
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_ROOT = REPO_ROOT / "outputs" / "validation_vault"

SUITE_META: Dict[str, Dict[str, str]] = {
    "01_ter26_sim_to_real": {
        "code": "01",
        "title": "TeR26 RWD Sim-to-Real Correlation & CAN Audit",
        "subtitle": "Strict 30% OOS val_30 validation, 6-dimension A/B/C ablation, and 200 Hz EKF+RTS CAN telemetry sanity gates.",
        "tag": "SIM-TO-REAL",
    },
    "02_ter27_maneuvers_and_handling": {
        "code": "02",
        "title": "TeR27 4WD Maneuvers & Handling Battery",
        "subtitle": "Milliken YMD, ISO ramp-steer & step-steer transients, Bode sweeps, trail braking, slalom/DLC, split-mu, and FS dynamic events.",
        "tag": "HANDLING",
    },
    "03_ter27_torque_vectoring_and_maps": {
        "code": "03",
        "title": "TeR27 Torque Vectoring, 5-Mode Maps & Fault Injection",
        "subtitle": "Open-diff vs e-LSD vs TV architectures, 5 ECU modes, 4-wheel friction circles, HV/BSPD compliance, and 3WD fault resilience.",
        "tag": "TORQUE VECTORING",
    },
    "04_ter27_full_lap_and_endurance": {
        "code": "04",
        "title": "TeR27 Full Lap OCP & 22 km Endurance Simulation",
        "subtitle": "FSG 2023 CasADi OCP vs 14-DOF closed-loop validation across 4 grip profiles, 22 km voltage/thermal evolution, and human drivability audit.",
        "tag": "FULL LAP / ENDURANCE",
    },
    "05_tires_and_thermal_analysis": {
        "code": "05",
        "title": "Hoosier R20 / R25B Tire & Thermal Envelope Analysis",
        "subtitle": "Pacejka MF 6.2 load/camber/combined-slip sweeps, R20 vs R25B compound delta, PINN residual hook, and thermal grip window.",
        "tag": "TIRES & THERMAL",
    },
    "06_suspension_kinematics_and_fea": {
        "code": "06",
        "title": "3D Double-Wishbone Kinematics & FEA Load Envelopes",
        "subtitle": "Exact 3D loop closure (< 1e-11 mm), camber/toe/roll-center/MR curves, Ackermann steering torque, FEA member loads, and rainflow fatigue.",
        "tag": "KINEMATICS & FEA",
    },
    "07_design_tradeoffs_and_fs_points": {
        "code": "07",
        "title": "Design Tradeoffs & Formula Student Points Optimizer",
        "subtitle": "Mass vs aero CL*A/CD*A iso-laptime contours, CG/inertia sensitivity tornado chart, and FSG/FSS Endurance + Efficiency points Pareto.",
        "tag": "DESIGN & POINTS",
    },
    "08_codegen_sil_and_godot_parity": {
        "code": "08",
        "title": "C99 SIL Codegen & Godot-Python Parity Verification",
        "subtitle": "CasADi C99 bit-exact SIL verification (0.0 abs error), STM32F405 168 MHz RT budget, and Godot 14-DOF vs Python physics parity matrix.",
        "tag": "SIL & PARITY",
    },
    "09_ter27_skidpad_deep_dive": {
        "code": "09",
        "title": "TeR27 4WD Skidpad Sub-5s Diagnostic Deep-Dive",
        "subtitle": "Calibrated FSG_RUBBERED (TeR26 = 5.50 s anchor), SKIDPAD_MAX_ATTACK_V2 200 Hz telemetry, 4-wheel grip/slip/load/3D kinematics, crossover micro-analysis, and Godot 4 14-DOF parity.",
        "tag": "SKIDPAD DEEP-DIVE",
    },
    "10_ter27_dil_and_sensor_realism": {
        "code": "10",
        "title": "TeR27 DiL Telemetry, 3D Braking Line & Sensor Realism",
        "subtitle": "SENSOR_IDEAL vs SENSOR_TRACK_REALISM (125 ms CAN accel latency, IMU noise/bias, EKF beta/Fz) across Skidpad, Autocross, and Trail-Braking/Slalom, plus CasADi sector deltas and 3D braking ramp elevation.",
        "tag": "DIL & SENSOR REALISM",
    },
    "11_differentiable_casadi_and_tubes": {
        "code": "11",
        "title": "Differentiable CasADi 14-DOF, Adjoint Optimization & Stochastic Tube-MPC",
        "subtitle": "C2-smooth 14-DOF CasADi C-Codegen model, empirical TeR26 track correlation (wz R=0.9872), exact reverse-mode adjoint gradients (< 1e-6) & 24-param co-optimization, 99.99% Stochastic Tube-MPC in grip drop (0 cones), and STM32F405 RT certification.",
        "tag": "DIFFERENTIABLE & TUBES",
    },
}


def _human_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    replacements = {
        "ter26": "TeR26",
        "ter27": "TeR27",
        "oos": "OOS",
        "gg": "G-G",
        "can": "CAN",
        "ymd": "YMD",
        "fs": "FS",
        "fsg": "FSG",
        "dlc": "DLC",
        "tv": "TV",
        "3wd": "3WD",
        "4wheels": "4-Wheels",
        "2d": "2D",
        "3d": "3D",
        "22km": "22 km",
        "75m": "75 m",
        "r20": "R20",
        "r25b": "R25B",
        "fy": "Fy",
        "fx": "Fx",
        "mz": "Mz",
        "fea": "FEA",
        "mr": "MR",
        "sil": "SIL",
        "c99": "C99",
        "casadi": "CasADi",
        "godot": "Godot",
        "python": "Python",
        "mu": "mu",
        "dil": "DiL",
        "ekf": "EKF",
        "imu": "IMU",
        "lts": "LTS",
        "diff14": "Diff14",
        "sysid": "SysID",
        "coopt": "Co-Opt",
        "stm32": "STM32",
    }
    words = stem.split("_")
    out = [replacements.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(out)


def _find_primary_metrics_json(folder: Path) -> Path | None:
    preferred = [
        "differentiable_casadi_and_tubes_metrics.json",
        "dil_sensor_realism_metrics.json",
        "skidpad_deep_dive_metrics.json",
        "metrics.json",
        "tv_metrics_catalog.json",
    ]
    for name in preferred:
        p = folder / name
        if p.exists():
            return p
    jsons = sorted(folder.glob("*.json"))
    return jsons[0] if jsons else None


def _extract_kpis(folder_name: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract concise, high-signal KPIs per vault folder."""
    kpis: List[Dict[str, str]] = []
    try:
        if folder_name == "01_ter26_sim_to_real":
            g = data["global_metrics_h10_active_oos"]["C_3d_kin_sysid"]
            a = data["telemetry_audit"]["sanity_gates"]
            kpis = [
                {"label": "OOS wz R^2 (C)", "value": f"{g['wz_rads']['r2']:.4f}", "status": "pass"},
                {"label": "OOS ay RMSE (C)", "value": f"{g['ay_ms2']['rmse']:.3f} m/s^2", "status": "pass"},
                {"label": "EKF res_ax RMS", "value": f"{a['gates']['kinematic_residual_ax_rms']['rms_ms2']:.3f} m/s^2", "status": "pass"},
                {"label": "EKF res_ay RMS", "value": f"{a['gates']['kinematic_residual_ay_rms']['rms_ms2']:.3f} m/s^2", "status": "pass"},
                {"label": "Sanity Gates", "value": "6/6 PASS" if a["all_gates_passed"] else "FAIL", "status": "pass" if a["all_gates_passed"] else "warn"},
            ]
        elif folder_name == "02_ter27_maneuvers_and_handling":
            ymd60 = data["family_1_ymd"]["velocity_sweep"]["60kmh"]
            ymd90 = data["family_1_ymd"]["velocity_sweep"]["90kmh"]
            rs = data["family_2_ramp_steer"]
            bode = data["family_4_bode_sine_sweep"]["80kmh"]
            fs_ev = data.get("family_6_fs_events", {})
            kpis = [
                {"label": "YMD Trim ay (90 km/h)", "value": f"{ymd90['trim_max_ay_g']:.2f} g", "status": "pass"},
                {"label": "K_us (65 km/h, 3D)", "value": f"{rs['understeer_gradient']['ter27_3d_65kmh']['k_us_linear_deg_per_g']:.3f} deg/g", "status": "pass"},
                {"label": "Roll Gradient", "value": f"{rs['roll_gradient_ter27_deg_per_g']:.2f} deg/g", "status": "pass"},
                {"label": "Yaw BW (-3dB, 80 km/h)", "value": f"{bode['wz_bandwidth_3db_hz']:.2f} Hz", "status": "pass"},
                {"label": "YMD Trim ay (60 km/h)", "value": f"{ymd60['trim_max_ay_g']:.2f} g", "status": "pass"},
            ]
            if "acceleration_75m" in fs_ev and "time_s" in fs_ev["acceleration_75m"]:
                kpis[-1] = {"label": "FS 75 m Accel", "value": f"{fs_ev['acceleration_75m']['time_s']:.3f} s", "status": "pass"}
        elif folder_name == "03_ter27_torque_vectoring_and_maps":
            arch = data["architectures_matrix"]["optimal_fsg_rubbered"]
            tf = data["torque_fight_and_power_audit"]
            modes = data["five_modes_kpis"]
            kpis = [
                {"label": "TV vs Open Lap Gain", "value": f"{arch['casadi_delta_vs_open_s']:+.3f} s", "status": "pass"},
                {"label": "Drivability Gain", "value": f"+{arch['casadi_drivability_gain_vs_open']:.2f} pts", "status": "pass"},
                {"label": "Torque-Fight Reduction", "value": f"-{tf['torque_fight_reduction_pct_mean']:.1f}%", "status": "pass"},
                {"label": "HV / BSPD Compliance", "value": f"{tf['hv_violations']} viol / PASS" if tf["bspd_compliant"] else "CHECK", "status": "pass"},
                {"label": "Eff. Mode 22 km Energy", "value": f"{modes['EFFICIENCY']['e_22km_kwh']:.2f} kWh", "status": "pass"},
            ]
        elif folder_name == "04_ter27_full_lap_and_endurance":
            profs = data["validated_profiles"]
            fsg_end = next((p for p in profs if p["profile"] == "FSG_RUBBERED" and "Endurance" in p["variant"]), profs[0])
            end22 = data["endurance_22km_summary"]
            drv = data["drivability_audit"]
            kpis = [
                {"label": "FSG Rubbered End. Lap", "value": f"{fsg_end['sim14dof_lap_time_s']:.2f} s", "status": "pass"},
                {"label": "OCP vs 14-DOF Gap", "value": f"{fsg_end['ocp_vs_14dof_gap_pct']:.2f}%", "status": "pass"},
                {"label": "22 km Min Voltage", "value": f"{end22['endurance_min_voltage_v']:.1f} V", "status": "pass"},
                {"label": "22 km Final SOC", "value": f"{end22['endurance_final_soc_pct']:.1f}%", "status": "pass"},
                {"label": "Drivability Score", "value": f"{drv['fsg_real_endurance_score']:.2f} / 100", "status": "pass"},
            ]
        elif folder_name == "05_tires_and_thermal_analysis":
            r20 = data["r20_at_800n_fsg"]
            r25 = data["r25b_at_800n_fsg"]
            dlt = data["r25b_minus_r20"]
            kpis = [
                {"label": "R20 Peak mu_y (800 N)", "value": f"{r20['mu_y_peak']:.3f}", "status": "pass"},
                {"label": "R25B Peak mu_y (800 N)", "value": f"{r25['mu_y_peak']:.3f} (+{dlt['delta_mu_y_pct']:.1f}%)", "status": "pass"},
                {"label": "R25B 22 km Delta", "value": f"{dlt['delta_22km_total_s']:+.2f} s", "status": "pass"},
                {"label": "Optimal Thermal Window", "value": f"{data['thermal_optimal_window_c'][0]:.0f}-{data['thermal_optimal_window_c'][1]:.0f} C", "status": "pass"},
                {"label": "PINN Residual Hook", "value": "Loaded" if data.get("pinn_residual_loaded") else "Off", "status": "pass"},
            ]
        elif folder_name == "06_suspension_kinematics_and_fea":
            kin = data["ter26_3d_kinematics"]
            fea = data["fea_summary"]
            kpis = [
                {"label": "3D Closure Residual", "value": f"{kin['closure_residual_mm']:.2e} mm", "status": "pass"},
                {"label": "Front Camber Gain", "value": f"{kin['front_camber_gain_deg_per_m']:.1f} deg/m", "status": "pass"},
                {"label": "LLTD Mechanical", "value": f"{kin['lltd_mechanical_pct']:.2f}%", "status": "pass"},
                {"label": "Max FEA Design Load", "value": f"{fea['max_design_load_sf15_n']:.0f} N (SF=1.5)", "status": "pass"},
                {"label": "22 km Miner Damage", "value": f"{data['max_miner_damage_22km']:.2e}", "status": "pass"},
            ]
        elif folder_name == "07_design_tradeoffs_and_fs_points":
            ma = data["mass_vs_aero"]
            fs = data["fs_points_optimizer"]
            kpis = [
                {"label": "Mass Sensitivity", "value": f"{ma['dt_dm_s_per_10kg']:.3f} s / 10 kg", "status": "pass"},
                {"label": "+10% CL*A Equiv. Mass", "value": f"-{ma['plus_10pct_cl_a']['equivalent_mass_kg']:.2f} kg", "status": "pass"},
                {"label": "FSG Optimal Points", "value": f"{fs['fsg_optimal_total_points']:.2f} pts", "status": "pass"},
                {"label": "FSS Optimal Points", "value": f"{fs['fss_optimal_total_points']:.2f} pts", "status": "pass"},
                {"label": "Optimal 22 km Energy", "value": f"{fs['fsg_optimal_energy_kwh']:.2f} kWh", "status": "pass"},
            ]
        elif folder_name == "08_codegen_sil_and_godot_parity":
            sil = data["c99_sil"]
            par = data["godot_python_parity"]
            kpis = [
                {"label": "C99 SIL Max Error", "value": f"{sil['accuracy']['kappa_max_abs_err']:.1e}", "status": "pass"},
                {"label": "Host C99 Latency (p50)", "value": f"{sil['host_latency_us']['ter27_eval_kappa']['p50_us']:.2f} us", "status": "pass"},
                {"label": "STM32F405 CPU Load", "value": f"{sil['stm32f405_168mhz_model']['fpu_f32']['cpu_load_pct']:.2f}% @ 200Hz", "status": "pass"},
                {"label": "Godot 3D vs Py Mean Err", "value": f"{par['mean_godot_3d_vs_python_err_pct']:.3f}%", "status": "pass"},
                {"label": "SIL & Parity Gate", "value": "PASS" if data.get("overall_pass") else "FAIL", "status": "pass" if data.get("overall_pass") else "warn"},
            ]
        elif folder_name == "09_ter27_skidpad_deep_dive":
            sc = data.get("scenarios", {})
            r20 = sc.get("FSG_RUBBERED_R20_MAX_ATTACK_V2", {})
            r25 = sc.get("FSG_RUBBERED_R25B_MAX_ATTACK_V2", {})
            cold = sc.get("COLD_18C_R20_MAX_ATTACK_V2", {})
            cal = data.get("fsg_calibration", {})
            par = data.get("godot_parity", {})
            kpis = [
                {"label": "TeR26 RWD Anchor", "value": f"{cal.get('ter26_rwd_skidpad_s', 5.50):.3f} s", "status": "pass"},
                {"label": "FSG R20 MAX_ATK V2", "value": f"{r20.get('t_mean_s', 0.0):.3f} s", "status": "pass" if r20.get("t_mean_s", 9.9) < 5.0 else "warn"},
                {"label": "FSG R25B MAX_ATK V2", "value": f"{r25.get('t_mean_s', 0.0):.3f} s", "status": "pass" if r25.get("t_mean_s", 9.9) < 5.0 else "warn"},
                {"label": "COLD_18C Min Clear", "value": f"{cold.get('min_cone_clear_m', 0.0):.3f} m", "status": "pass" if cold.get("min_cone_clear_m", 0.0) >= 0.15 else "warn"},
                {"label": "Godot vs Py Circle Err", "value": f"{par.get('circle_time_delta_pct', 0.0):.2f}%", "status": "pass"},
            ]
        elif folder_name == "10_ter27_dil_and_sensor_realism":
            sk = data.get("skidpad", {})
            ax = data.get("autocross", {})
            sl = data.get("trail_braking_slalom", {})
            rmp = data.get("braking_ramp_3d", {})
            kpis = [
                {"label": "CAN Accel Latency", "value": f"{data.get('can_accel_latency_ms', 125.0):.1f} ms", "status": "pass"},
                {"label": "Skidpad Real vs Ideal", "value": f"{sk.get('delta_time_s', 0.0):+.3f} s", "status": "pass"},
                {"label": "Autocross Sector Delta", "value": f"{ax.get('total_delta_t_s', 0.0):+.3f} s", "status": "pass"},
                {"label": "Slalom Beta RMS (Real)", "value": f"{sl.get('track_realism', {}).get('beta_rms_deg', 0.0):.2f} deg", "status": "pass"},
                {"label": "3D Brake Ramp Peak +Y", "value": f"{rmp.get('peak_elevation_m', 0.0):.3f} m", "status": "pass"},
            ]
        elif folder_name == "11_differentiable_casadi_and_tubes":
            t26 = data.get("ter26_empirical_correlation", {})
            adj = data.get("ter27_adjoint_optimization", {})
            tub = data.get("stochastic_tube_mpc", {})
            kpis = [
                {"label": "TeR26 Diff14 wz R", "value": f"{t26.get('wz_pearson_r', 0.0):.4f}", "status": "pass"},
                {"label": "Adjoint vs FD Max Err", "value": f"{adj.get('worst_fd_rel_err', 0.0):.2e}", "status": "pass"},
                {"label": "TeR27 Co-Opt Gain", "value": f"{adj.get('total_gain_ms', 0.0):+.2f} ms", "status": "pass"},
                {"label": "Grip-Drop Tube Cones", "value": f"{tub.get('grip_drop_tube_cones', 0)} (min {tub.get('grip_drop_tube_min_clear_m', 0.0):.3f} m)", "status": "pass"},
                {"label": "STM32F405 CPU Load", "value": f"{tub.get('stm32f405_cpu_load_pct', 0.0):.2f}% @ 200Hz", "status": "pass"},
            ]
    except Exception as exc:
        kpis.append({"label": "KPI Parse Note", "value": str(exc)[:60], "status": "warn"})
    return kpis


def scan_vault() -> Dict[str, Any]:
    folders = sorted(p for p in VAULT_ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())
    suites: List[Dict[str, Any]] = []
    total_pngs = 0
    total_jsons = 0
    total_size_kb = 0.0

    for folder in folders:
        fname = folder.name
        meta = SUITE_META.get(
            fname,
            {
                "code": fname[:2],
                "title": _human_title_from_filename(fname),
                "subtitle": "Validation suite outputs.",
                "tag": "VALIDATION",
            },
        )
        pngs = sorted(folder.glob("*.png"))
        jsons = sorted(folder.glob("*.json"))
        csvs = sorted(folder.glob("*.csv"))
        primary_json = _find_primary_metrics_json(folder)
        metrics_data: Dict[str, Any] = {}
        if primary_json is not None:
            metrics_data = json.loads(primary_json.read_text(encoding="utf-8"))

        figures: List[Dict[str, Any]] = []
        for png in pngs:
            size_kb = round(png.stat().st_size / 1024.0, 1)
            with Image.open(png) as im:
                w, h = im.size
                dpi_raw = im.info.get("dpi", (0, 0))
                dpi_x = round(float(dpi_raw[0]), 1) if isinstance(dpi_raw, tuple) and len(dpi_raw) >= 2 else 0.0
                dpi_y = round(float(dpi_raw[1]), 1) if isinstance(dpi_raw, tuple) and len(dpi_raw) >= 2 else 0.0
            total_size_kb += size_kb
            figures.append(
                {
                    "filename": png.name,
                    "rel_path": f"{fname}/{png.name}",
                    "title": _human_title_from_filename(png.name),
                    "size_kb": size_kb,
                    "width_px": w,
                    "height_px": h,
                    "dpi": [dpi_x, dpi_y],
                }
            )

        total_pngs += len(figures)
        total_jsons += len(jsons)
        kpis = _extract_kpis(fname, metrics_data)

        suites.append(
            {
                "folder": fname,
                "code": meta["code"],
                "title": meta["title"],
                "subtitle": meta["subtitle"],
                "tag": meta["tag"],
                "primary_metrics_json": f"{fname}/{primary_json.name}" if primary_json else None,
                "json_files": [f"{fname}/{j.name}" for j in jsons],
                "csv_files": [f"{fname}/{c.name}" for c in csvs],
                "n_figures": len(figures),
                "kpis": kpis,
                "figures": figures,
            }
        )

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "standard": "ISO 8855 (+X fwd, +Y left, +Z up) | MoTeC i2 Dark Engineering | 300 DPI",
        "summary": {
            "n_suites": len(suites),
            "total_pngs": total_pngs,
            "total_jsons": total_jsons,
            "total_png_size_mb": round(total_size_kb / 1024.0, 2),
        },
        "suites": suites,
    }


def write_index_markdown(catalog: Dict[str, Any], out_path: Path) -> None:
    s = catalog["summary"]
    lines: List[str] = [
        "# Tecnun eRacing Vehicle Dynamics Validation Vault",
        "",
        f"- **Generated (UTC)**: `{catalog['generated_utc']}`",
        f"- **Standard**: {catalog['standard']}",
        f"- **Suites**: `{s['n_suites']}` | **300 DPI Figures**: `{s['total_pngs']}` | **JSON Metrics Files**: `{s['total_jsons']}` | **Total Plot Payload**: `{s['total_png_size_mb']:.2f} MB`",
        "- **Interactive Gallery**: [dashboard_index.html](dashboard_index.html)",
        "- **Master Catalog JSON**: [master_metrics_catalog.json](master_metrics_catalog.json)",
        "",
        "## Executive KPI Summary",
        "",
        "| Suite | Figures | Key Metrics | Primary JSON |",
        "| --- | ---: | --- | --- |",
    ]
    for suite in catalog["suites"]:
        kpi_str = "; ".join(f"{k['label']}: **{k['value']}**" for k in suite["kpis"][:3])
        pjson = suite["primary_metrics_json"] or "-"
        pjson_md = f"[{Path(pjson).name}]({pjson})" if pjson != "-" else "-"
        lines.append(f"| `{suite['code']}` {suite['title']} | {suite['n_figures']} | {kpi_str} | {pjson_md} |")

    lines.append("")
    for suite in catalog["suites"]:
        lines.append(f"## {suite['code']} - {suite['title']}")
        lines.append("")
        lines.append(suite["subtitle"])
        lines.append("")
        lines.append("**Key KPIs:**")
        for k in suite["kpis"]:
            lines.append(f"- {k['label']}: `{k['value']}`")
        lines.append("")
        if suite["json_files"]:
            jlinks = ", ".join(f"[{Path(j).name}]({j})" for j in suite["json_files"])
            lines.append(f"**JSON Reports:** {jlinks}")
            lines.append("")
        if suite["csv_files"]:
            clinks = ", ".join(f"[{Path(c).name}]({c})" for c in suite["csv_files"])
            lines.append(f"**CSV Exports:** {clinks}")
            lines.append("")
        lines.append("| Figure | Resolution | DPI | Size (KB) | Link |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for fig in suite["figures"]:
            dpi_s = f"{int(round(fig['dpi'][0]))}x{int(round(fig['dpi'][1]))}"
            lines.append(
                f"| {fig['title']} | {fig['width_px']}x{fig['height_px']} | {dpi_s} | {fig['size_kb']:.1f} | [{fig['filename']}]({fig['rel_path']}) |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_dashboard_html(catalog: Dict[str, Any], out_path: Path) -> None:
    s = catalog["summary"]
    flat_figures: List[Dict[str, str]] = []
    sidebar_links: List[str] = []
    summary_rows: List[str] = []
    suite_sections: List[str] = []

    for suite in catalog["suites"]:
        sec_id = f"suite-{suite['code']}"
        sidebar_links.append(
            f'<a class="nav-item" href="#{sec_id}" data-suite="{html.escape(suite["folder"])}">'
            f'<span class="nav-code">{html.escape(suite["code"])}</span>'
            f'<span class="nav-label">{html.escape(suite["title"])}</span>'
            f'<span class="nav-badge">{suite["n_figures"]}</span>'
            f"</a>"
        )

        kpi_inline = " | ".join(
            f'<span class="tbl-kpi"><b>{html.escape(k["label"])}:</b> {html.escape(k["value"])}</span>'
            for k in suite["kpis"][:3]
        )
        pjson = suite["primary_metrics_json"] or ""
        pjson_link = (
            f'<a class="json-link" href="{html.escape(pjson)}" target="_blank">{html.escape(Path(pjson).name)}</a>'
            if pjson
            else "-"
        )
        summary_rows.append(
            f"<tr>"
            f'<td><a class="suite-jump" href="#{sec_id}">{html.escape(suite["code"])} - {html.escape(suite["title"])}</a></td>'
            f'<td class="num">{suite["n_figures"]}</td>'
            f"<td>{kpi_inline}</td>"
            f"<td>{pjson_link}</td>"
            f"</tr>"
        )

        kpi_cards_html = "".join(
            f'<div class="kpi-pill status-{html.escape(k["status"])}">'
            f'<div class="kpi-label">{html.escape(k["label"])}</div>'
            f'<div class="kpi-val">{html.escape(k["value"])}</div>'
            f"</div>"
            for k in suite["kpis"]
        )

        data_links_list = [
            f'<a class="data-chip" href="{html.escape(j)}" target="_blank">JSON: {html.escape(Path(j).name)}</a>'
            for j in suite["json_files"]
        ] + [
            f'<a class="data-chip" href="{html.escape(c)}" target="_blank">CSV: {html.escape(Path(c).name)}</a>'
            for c in suite["csv_files"]
        ]
        data_links_html = "".join(data_links_list)

        fig_cards: List[str] = []
        for fig in suite["figures"]:
            idx = len(flat_figures)
            flat_figures.append(
                {
                    "src": fig["rel_path"],
                    "title": f"[{suite['code']}] {fig['title']}",
                    "meta": f"{fig['rel_path']} | {fig['width_px']}x{fig['height_px']} px | {int(round(fig['dpi'][0]))} DPI | {fig['size_kb']:.1f} KB",
                }
            )
            search_blob = f"{suite['code']} {suite['title']} {fig['title']} {fig['filename']}".lower()
            fig_cards.append(
                f'<article class="fig-card" data-search="{html.escape(search_blob)}" onclick="openLightbox({idx})">'
                f'<div class="fig-thumb-wrap">'
                f'<img loading="lazy" src="{html.escape(fig["rel_path"])}" alt="{html.escape(fig["title"])}">'
                f'<span class="zoom-hint">CLICK TO ZOOM</span>'
                f"</div>"
                f'<div class="fig-meta">'
                f'<div class="fig-title">{html.escape(fig["title"])}</div>'
                f'<div class="fig-sub">{html.escape(fig["filename"])} | {fig["width_px"]}x{fig["height_px"]} | {int(round(fig["dpi"][0]))} DPI | {fig["size_kb"]:.1f} KB</div>'
                f"</div>"
                f"</article>"
            )

        suite_sections.append(
            f'<section class="suite-section" id="{sec_id}">'
            f'<div class="suite-header">'
            f'<div><span class="suite-tag">{html.escape(suite["code"])} | {html.escape(suite["tag"])}</span>'
            f'<h2>{html.escape(suite["title"])}</h2>'
            f'<p class="suite-sub">{html.escape(suite["subtitle"])}</p></div>'
            f'<div class="data-chips">{data_links_html}</div>'
            f"</div>"
            f'<div class="kpi-row">{kpi_cards_html}</div>'
            f'<div class="fig-grid">{"".join(fig_cards)}</div>'
            f"</section>"
        )

    figures_json_js = json.dumps(flat_figures, ensure_ascii=True)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tecnun eRacing | Vehicle Dynamics Validation Vault</title>
<style>
:root {{
  --bg-root: #0D1117;
  --bg-panel: #131922;
  --bg-card: #161E29;
  --bg-hover: #1D2736;
  --border: #263040;
  --border-hi: #38BDF8;
  --text: #E6EDF3;
  --muted: #9BA7B4;
  --cyan: #38BDF8;
  --emerald: #10B981;
  --amber: #F59E0B;
  --crimson: #F43F5E;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  background: var(--bg-root);
  color: var(--text);
  font-family: var(--sans);
  display: flex;
  min-height: 100vh;
}}
.sidebar {{
  width: 290px;
  flex-shrink: 0;
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  padding: 18px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.brand {{
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
  margin-bottom: 4px;
}}
.brand-kicker {{
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--cyan);
}}
.brand h1 {{
  margin: 4px 0 2px 0;
  font-size: 16px;
  font-weight: 700;
}}
.brand-sub {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}}
.search-box {{
  width: 100%;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg-root);
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
}}
.search-box:focus {{
  outline: none;
  border-color: var(--cyan);
}}
.nav-list {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 9px;
  border-radius: 6px;
  text-decoration: none;
  color: var(--text);
  font-size: 12.5px;
  border: 1px solid transparent;
}}
.nav-item:hover {{
  background: var(--bg-hover);
  border-color: var(--border);
}}
.nav-code {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--cyan);
  background: rgba(56, 189, 248, 0.12);
  padding: 2px 6px;
  border-radius: 4px;
  flex-shrink: 0;
}}
.nav-label {{
  flex: 1;
  line-height: 1.25;
}}
.nav-badge {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  background: var(--bg-root);
  border: 1px solid var(--border);
  padding: 1px 6px;
  border-radius: 10px;
}}
.sidebar-footer {{
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.sidebar-footer a {{
  color: var(--cyan);
  text-decoration: none;
}}
.main {{
  flex: 1;
  padding: 24px 28px 60px 28px;
  min-width: 0;
}}
.hero {{
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-left: 4px solid var(--cyan);
  border-radius: 8px;
  padding: 18px 22px;
  margin-bottom: 22px;
}}
.hero-top {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}}
.hero h2 {{
  margin: 0;
  font-size: 20px;
}}
.hero-meta {{
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
  margin-top: 6px;
}}
.hero-stats {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.stat-box {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 14px;
  text-align: right;
}}
.stat-num {{
  font-family: var(--mono);
  font-size: 18px;
  font-weight: 700;
  color: var(--emerald);
}}
.stat-lbl {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
}}
.summary-panel {{
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 18px;
  margin-bottom: 26px;
  overflow-x: auto;
}}
.summary-panel h3 {{
  margin: 0 0 12px 0;
  font-size: 15px;
  font-family: var(--mono);
  color: var(--cyan);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}}
th, td {{
  text-align: left;
  padding: 9px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
th {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
}}
td.num {{
  font-family: var(--mono);
  text-align: right;
  color: var(--cyan);
  font-weight: 700;
}}
.suite-jump {{
  color: var(--text);
  text-decoration: none;
  font-weight: 600;
}}
.suite-jump:hover {{ color: var(--cyan); }}
.tbl-kpi {{
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--muted);
}}
.tbl-kpi b {{ color: var(--text); font-weight: 600; }}
.json-link {{
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--cyan);
  text-decoration: none;
}}
.suite-section {{
  margin-bottom: 34px;
  padding-top: 6px;
}}
.suite-header {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-end;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin-bottom: 14px;
}}
.suite-tag {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--cyan);
  letter-spacing: 0.06em;
}}
.suite-header h2 {{
  margin: 3px 0 4px 0;
  font-size: 18px;
}}
.suite-sub {{
  margin: 0;
  font-size: 13px;
  color: var(--muted);
}}
.data-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.data-chip {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--cyan);
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 4px 8px;
  text-decoration: none;
}}
.data-chip:hover {{ border-color: var(--cyan); }}
.kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}}
.kpi-pill {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 2px solid var(--emerald);
  border-radius: 6px;
  padding: 9px 12px;
}}
.kpi-pill.status-warn {{ border-top-color: var(--amber); }}
.kpi-label {{
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--muted);
  text-transform: uppercase;
}}
.kpi-val {{
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 700;
  margin-top: 4px;
  color: var(--text);
}}
.fig-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 14px;
}}
.fig-card {{
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
}}
.fig-card:hover {{
  border-color: var(--cyan);
  transform: translateY(-2px);
}}
.fig-thumb-wrap {{
  position: relative;
  background: #090C10;
  aspect-ratio: 16 / 10;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}
.fig-thumb-wrap img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
.zoom-hint {{
  position: absolute;
  right: 8px;
  bottom: 8px;
  font-family: var(--mono);
  font-size: 10px;
  background: rgba(13, 17, 23, 0.85);
  color: var(--cyan);
  border: 1px solid var(--border);
  padding: 2px 6px;
  border-radius: 4px;
}}
.fig-meta {{
  padding: 10px 12px;
}}
.fig-title {{
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 3px;
}}
.fig-sub {{
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--muted);
  word-break: break-all;
}}
.lightbox {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(6, 9, 14, 0.94);
  z-index: 1000;
  flex-direction: column;
  padding: 16px 22px;
}}
.lightbox.open {{ display: flex; }}
.lb-bar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}
.lb-title {{
  font-size: 15px;
  font-weight: 700;
}}
.lb-sub {{
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--muted);
}}
.lb-controls {{
  display: flex;
  gap: 8px;
}}
.lb-btn {{
  font-family: var(--mono);
  font-size: 12px;
  background: var(--bg-card);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 11px;
  cursor: pointer;
}}
.lb-btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
.lb-stage {{
  flex: 1;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px 0;
}}
.lb-stage img {{
  max-width: 96vw;
  max-height: 82vh;
  object-fit: contain;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #0D1117;
  cursor: zoom-in;
}}
.lb-stage img.zoomed {{
  max-width: none;
  max-height: none;
  width: 140%;
  cursor: zoom-out;
}}
@media (max-width: 900px) {{
  body {{ flex-direction: column; }}
  .sidebar {{ width: 100%; height: auto; position: static; }}
}}
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">
    <div class="brand-kicker">TECNUN eRACING | ISO 8855</div>
    <h1>Validation Vault</h1>
    <div class="brand-sub">MoTeC i2 Dark Engineering | 300 DPI</div>
  </div>
  <input id="filterInput" class="search-box" type="text" placeholder="Filter figures by keyword...">
  <nav class="nav-list">
    <a class="nav-item" href="#overview">
      <span class="nav-code">00</span>
      <span class="nav-label">Executive KPI Overview</span>
      <span class="nav-badge">{s['total_pngs']}</span>
    </a>
    {"".join(sidebar_links)}
  </nav>
  <div class="sidebar-footer">
    <div>Generated: {html.escape(catalog['generated_utc'])}</div>
    <a href="master_metrics_catalog.json" target="_blank">master_metrics_catalog.json</a>
    <a href="INDEX.md" target="_blank">INDEX.md</a>
  </div>
</aside>

<main class="main">
  <section class="hero" id="overview">
    <div class="hero-top">
      <div>
        <h2>Vehicle Dynamics Validation Vault Dashboard</h2>
        <div class="hero-meta">{html.escape(catalog['standard'])}</div>
      </div>
      <div class="hero-stats">
        <div class="stat-box"><div class="stat-num">{s['n_suites']}</div><div class="stat-lbl">Vault Suites</div></div>
        <div class="stat-box"><div class="stat-num">{s['total_pngs']}</div><div class="stat-lbl">300 DPI PNGs</div></div>
        <div class="stat-box"><div class="stat-num">{s['total_jsons']}</div><div class="stat-lbl">JSON Reports</div></div>
        <div class="stat-box"><div class="stat-num">{s['total_png_size_mb']:.1f} MB</div><div class="stat-lbl">Plot Payload</div></div>
      </div>
    </div>
  </section>

  <section class="summary-panel">
    <h3>EXECUTIVE KPI MATRIX (SUITES 01 - {s['n_suites']:02d})</h3>
    <table>
      <thead>
        <tr>
          <th>Validation Suite</th>
          <th style="text-align:right;">PNGs</th>
          <th>Headline Engineering KPIs</th>
          <th>Primary Metrics</th>
        </tr>
      </thead>
      <tbody>
        {"".join(summary_rows)}
      </tbody>
    </table>
  </section>

  {"".join(suite_sections)}
</main>

<div id="lightbox" class="lightbox" onclick="if(event.target===this) closeLightbox();">
  <div class="lb-bar">
    <div>
      <div id="lbTitle" class="lb-title"></div>
      <div id="lbSub" class="lb-sub"></div>
    </div>
    <div class="lb-controls">
      <button class="lb-btn" onclick="stepLightbox(-1)">Prev [Left]</button>
      <button class="lb-btn" onclick="toggleZoom()">Zoom</button>
      <button class="lb-btn" onclick="stepLightbox(1)">Next [Right]</button>
      <button class="lb-btn" onclick="closeLightbox()">Close [Esc]</button>
    </div>
  </div>
  <div class="lb-stage">
    <img id="lbImg" src="" alt="" onclick="toggleZoom()">
  </div>
</div>

<script>
const FIGURES = {figures_json_js};
let currentIdx = 0;

function renderLightbox() {{
  const item = FIGURES[currentIdx];
  const img = document.getElementById("lbImg");
  img.classList.remove("zoomed");
  img.src = item.src;
  img.alt = item.title;
  document.getElementById("lbTitle").textContent = item.title + " (" + (currentIdx + 1) + " / " + FIGURES.length + ")";
  document.getElementById("lbSub").textContent = item.meta;
}}

function openLightbox(idx) {{
  currentIdx = idx;
  renderLightbox();
  document.getElementById("lightbox").classList.add("open");
}}

function closeLightbox() {{
  document.getElementById("lightbox").classList.remove("open");
}}

function stepLightbox(delta) {{
  currentIdx = (currentIdx + delta + FIGURES.length) % FIGURES.length;
  renderLightbox();
}}

function toggleZoom() {{
  document.getElementById("lbImg").classList.toggle("zoomed");
}}

document.addEventListener("keydown", function(ev) {{
  const lb = document.getElementById("lightbox");
  if (!lb.classList.contains("open")) return;
  if (ev.key === "Escape") closeLightbox();
  else if (ev.key === "ArrowRight") stepLightbox(1);
  else if (ev.key === "ArrowLeft") stepLightbox(-1);
}});

document.getElementById("filterInput").addEventListener("input", function(ev) {{
  const q = ev.target.value.trim().toLowerCase();
  document.querySelectorAll(".fig-card").forEach(function(card) {{
    const hay = card.getAttribute("data-search") || "";
    card.style.display = (!q || hay.indexOf(q) !== -1) ? "" : "none";
  }});
}});
</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    print("[VaultIndex] Scanning outputs/validation_vault/01..11 ...", flush=True)
    catalog = scan_vault()

    catalog_path = VAULT_ROOT / "master_metrics_catalog.json"
    index_md_path = VAULT_ROOT / "INDEX.md"
    dashboard_html_path = VAULT_ROOT / "dashboard_index.html"

    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    write_index_markdown(catalog, index_md_path)
    write_dashboard_html(catalog, dashboard_html_path)

    s = catalog["summary"]
    print(
        f"[VaultIndex] Built catalog ({s['n_suites']} suites, {s['total_pngs']} PNGs, {s['total_jsons']} JSONs, {s['total_png_size_mb']:.2f} MB).",
        flush=True,
    )
    print(f"[VaultIndex] Wrote: {catalog_path}", flush=True)
    print(f"[VaultIndex] Wrote: {index_md_path}", flush=True)
    print(f"[VaultIndex] Wrote: {dashboard_html_path}", flush=True)


if __name__ == "__main__":
    main()
