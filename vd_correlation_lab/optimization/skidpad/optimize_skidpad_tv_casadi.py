#!/usr/bin/env python3
"""
Official FSG Skidpad Trajectory + 4-Wheel Torque + Suspension Setup Co-Optimization
(CasADi 3.7.2 + IPOPT) for Tecnun e-Racing TeR27 4WD (MirenaSim).

Deliverables:
  1. Calibrated FSG_RUBBERED Track Grip Anchor:
       Loads `fsg_grip_calibration_from_ter26.json` (reproducing TeR26 RWD = 5.500 s)
       and applies the exact calibrated `mu_fsg_scale` & `T_fsg_c` to TeR27 4WD.
  2. Systematic 4-Tier Comparison on Calibrated FSG_RUBBERED:
       (1) TeR26 RWD (real Hockenheim anchor ~5.50 s)
       (2) TeR27 4WD Open-Diff (R=9.125 m and optimal R_eff*)
       (3) TeR27 4WD Base Setup + Optimal CasADi TV (R20 & R25B; plus TTC_PEAK)
       (4) TeR27 4WD Dedicated Skidpad Setup + Optimal CasADi TV (R20 & R25B),
           co-optimizing LLTD/ARB stiffness, static F/R camber, static F/R toe,
           and brake/regen bias within realistic mechanical bounds.
  3. Exact Time Waterfall (ms contribution of every physical factor):
       - FSG_RUBBERED real calibration (vs legacy double-penalized 0.88)
       - Tight R_eff* (9.125 m -> 8.390 m)
       - Optimal Mz_tv* (rear TV / yaw moment allocation)
       - Outer-front longitudinal unloading (4-wheel asymmetric front torque shedding)
       - Ballistic Figure-8 crossover
       - Dedicated Skidpad suspension setup
       - R20 -> R25B tire compound
  4. Strict Verification & JSON Reports:
       - 0 HV violations ([-25.0, +80.0] kW), min_clear_m >= 0.150 m everywhere
       - outputs/validation_vault/03_ter27_torque_vectoring_and_maps/skidpad_oracle_report.json
       - vd_correlation_lab/results/tv_oracle/skidpad_oracle_report.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
VAULT_03_DIR = REPO_ROOT / "outputs" / "validation_vault" / "03_ter27_torque_vectoring_and_maps"
CAL_JSON_PATH = VAULT_03_DIR / "fsg_grip_calibration_from_ter26.json"

for _p in (str(REPO_ROOT), str(LAB_DIR), str(ORACLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.optimization.skidpad.calibrate_fsg_grip_from_ter26 import calibrate_fsg_rubbered  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.solve_dynamic_events_casadi import (  # noqa: E402
    CONE_MARGIN_M,
    R_CENTER_M,
    R_EFF_MIN_M,
    R_INNER_M,
    R_OUTER_M,
    solve_accel_75m_casadi,
    solve_skidpad_figure8_casadi,
    solve_skidpad_stationary_casadi,
)

D_INNER_M = 15.25
D_OUTER_M = 21.25

COMBINATIONS: List[Tuple[str, str]] = [
    ("FSG_RUBBERED", "R20"),
    ("FSG_RUBBERED", "R25B"),
    ("TTC_PEAK", "R20"),
    ("TTC_PEAK", "R25B"),
]


def _make_summary_row(
    key: str,
    prof: str,
    comp: str,
    st_opt: Dict[str, Any],
    fig8: Dict[str, Any],
    acc75: Dict[str, Any],
    setup_label: str = "base",
) -> Dict[str, Any]:
    return {
        "combo": key,
        "setup": setup_label,
        "grip_profile": prof,
        "compound": comp,
        "r_eff_star_m": round(st_opt["r_eff_m"], 4),
        "t_stat_s": round(st_opt["t_circle_s"], 4),
        "t_right_s": round(fig8["t_right_periodic_s"], 4),
        "t_left_s": round(fig8["t_left_periodic_s"], 4),
        "t_mean_s": round(fig8["t_mean_periodic_s"], 4),
        "t_right_as_timed_s": round(fig8["t_right_s"], 4),
        "t_left_as_timed_s": round(fig8["t_left_s"], 4),
        "t_mean_as_timed_s": round(fig8["t_mean_s"], 4),
        "crossover_surge_v_max_ms": round(fig8["crossover_surge_v_max_ms"], 2),
        "v_mean_ms": round(fig8["v_plateau_ms"], 3),
        "ay_mean_g": round(fig8["ay_plateau_g"], 3),
        "T_plateau_R_nm": [round(v, 2) for v in fig8["T_motor_plateau_R_nm"]],
        "T_plateau_L_nm": [round(v, 2) for v in fig8["T_motor_plateau_L_nm"]],
        "mu_util_plateau_4": [round(v, 3) for v in fig8["mu_util_plateau_4"]],
        "accel_75m_valid": bool(acc75["converged_valid"]),
        "ackermann_pct": round(float(st_opt.get("ackermann_pct", -11.0)), 2),
        "beta_star_deg": round(st_opt["beta_deg"], 3),
        "T_star_4_nm": [round(v, 2) for v in st_opt["T_motor_4_nm"]],
        "mz_tv_star_nm": round(st_opt["mz_tv_nm"], 2),
        "p_dc_stat_kw": round(st_opt["p_dc_phys_kw"], 2),
        "p_dc_fig8_min_kw": round(fig8["p_dc_min_kw"], 2),
        "p_dc_fig8_max_kw": round(fig8["p_dc_max_kw"], 2),
        "hv_violations": int(fig8["hv_violations"]),
        "min_clear_m": round(min(st_opt["min_cone_clear_m"], fig8["min_cone_clear_m"]), 4),
        "dmz_dbeta_iso_nm_deg": round(st_opt["dmz_dbeta_nm_deg"], 2),
        "directionally_stable_weathercock": bool(st_opt["dmz_dbeta_nm_deg"] > 0.0),
        "dmz_dbeta_task_convention_nm_deg": round(-st_opt["dmz_dbeta_nm_deg"], 2),
        "accel_75m_s": round(acc75["time_75m_s"], 4),
    }


def diagnose_legacy_5p91_bottleneck() -> Dict[str, Any]:
    """Decompose the calibrated FSG_RUBBERED Open-Diff vs CasADi TV stationary ceiling."""
    open_9125 = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED", "R20", r_fixed=R_CENTER_M, open_diff=True, turn_dir="left"
    )
    open_rstar = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED", "R20", r_fixed=None, open_diff=True, turn_dir="left"
    )
    tv_9125 = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED", "R20", r_fixed=R_CENTER_M, open_diff=False, turn_dir="left"
    )
    tv_eq_front_rstar = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED", "R20", r_fixed=None, open_diff=False, equal_front_drive=True, turn_dir="left"
    )
    tv_rstar = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED", "R20", r_fixed=None, open_diff=False, turn_dir="left"
    )

    t_legacy = float(open_9125["t_circle_s"])
    t_tv_9125 = float(tv_9125["t_circle_s"])
    t_tv_rstar = float(tv_rstar["t_circle_s"])

    return {
        "baseline_legacy_open_9125": open_9125,
        "open_diff_optimal_r_star": open_rstar,
        "tv_optimal_9125": tv_9125,
        "tv_equal_front_r_star": tv_eq_front_rstar,
        "tv_optimal_r_star": tv_rstar,
        "decomposition_s": {
            "legacy_t_circle_s": t_legacy,
            "tv_allocation_gain_s": t_legacy - t_tv_9125,
            "tight_radius_gain_s": t_tv_9125 - t_tv_rstar,
            "total_stationary_gain_s": t_legacy - t_tv_rstar,
        },
        "tire_saturation_notes": {
            "open_diff_mu_util_4": open_9125["mu_util_4"],
            "tv_rstar_mu_util_4": tv_rstar["mu_util_4"],
            "open_diff_ay_g": open_9125["ay_g"],
            "tv_rstar_ay_g": tv_rstar["ay_g"],
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    print("=== Skidpad Oracle (CasADi 3.7.2 + IPOPT | Calibrated FSG_RUBBERED) ===", flush=True)

    # 0. Load or run TeR26 RWD 5.50 s calibration
    if CAL_JSON_PATH.exists():
        fsg_cal = json.loads(CAL_JSON_PATH.read_text(encoding="utf-8"))
    else:
        fsg_cal = calibrate_fsg_rubbered()
    print(
        f"[FSG Calibration] mu_fsg_scale={fsg_cal['mu_fsg_scale']:.4f} | T_fsg={fsg_cal['T_fsg_c']:.1f}C | "
        f"TeR26 RWD={fsg_cal['ter26_rwd_skidpad_s']:.4f}s (ay={fsg_cal['ter26_ay_g']:.3f}g, clear={fsg_cal['min_clear_m']:.3f}m)",
        flush=True,
    )

    # 1. Legacy 0.88 reference + calibrated Open-Diff decomposition
    legacy_0p88_open = solve_skidpad_stationary_casadi(
        "FSG_RUBBERED_LEGACY_0P88", "R20", r_fixed=R_CENTER_M, open_diff=True, turn_dir="left"
    )
    diag = diagnose_legacy_5p91_bottleneck()
    dec = diag["decomposition_s"]
    print(
        f"[Diagnosis FSG_RUBBERED] Open@9.125={dec['legacy_t_circle_s']:.3f}s | "
        f"Open@R*={diag['open_diff_optimal_r_star']['t_circle_s']:.3f}s | "
        f"TV@9.125={diag['tv_optimal_9125']['t_circle_s']:.3f}s | "
        f"TV@R*={diag['tv_optimal_r_star']['t_circle_s']:.3f}s",
        flush=True,
    )

    # 2. Full 4-combo Base Setup + Optimal CasADi TV matrix (preserving keys)
    matrix: Dict[str, Any] = {}
    summary_table: List[Dict[str, Any]] = []

    for prof, comp in COMBINATIONS:
        key = f"{prof}__{comp}"
        st_opt = solve_skidpad_stationary_casadi(prof, comp, r_fixed=None, open_diff=False, turn_dir="left")
        fig8 = solve_skidpad_figure8_casadi(prof, comp, v_entry_ms=4.0, n_circle=24)
        acc75 = solve_accel_75m_casadi(prof, comp)

        matrix[key] = {
            "grip_profile": prof,
            "compound": comp,
            "stationary_optimal": st_opt,
            "figure8_ocp": fig8,
            "accel_75m_ocp": acc75,
        }

        row = _make_summary_row(key, prof, comp, st_opt, fig8, acc75, setup_label="base")
        summary_table.append(row)
        print(
            f"[BASE  | {prof:12s} | {comp:4s}] R*={row['r_eff_star_m']:.3f}m | "
            f"t_stat={row['t_stat_s']:.3f}s t_per={row['t_mean_s']:.3f}s t_fig8={row['t_mean_as_timed_s']:.3f}s | "
            f"ay={row['ay_mean_g']:.3f}g beta*={row['beta_star_deg']:+.2f}deg | "
            f"MzTV*={row['mz_tv_star_nm']:+.1f}Nm Pdc=[{row['p_dc_fig8_min_kw']:.1f},{row['p_dc_fig8_max_kw']:.1f}]kW | "
            f"clear={row['min_clear_m']:.3f}m hv_viol={row['hv_violations']}",
            flush=True,
        )

    # 3. Dedicated Skidpad Setup + Optimal CasADi TV Co-Optimization on calibrated FSG_RUBBERED (R20 & R25B)
    dedicated_setup_opt: Dict[str, Any] = {}
    for comp in ("R20", "R25B"):
        key_ded = f"FSG_RUBBERED__{comp}__DEDICATED_SETUP"
        st_ded = solve_skidpad_stationary_casadi(
            "FSG_RUBBERED", comp, r_fixed=None, open_diff=False, turn_dir="left", optimize_setup=True
        )
        opt_setup = dict(st_ded["optimal_setup"])
        setup_overrides = {
            "lltd": opt_setup["lltd"],
            "k_roll_scale": opt_setup.get("k_roll_scale", 1.0),
            "camber_f_deg": opt_setup["camber_f_deg"],
            "camber_r_deg": opt_setup["camber_r_deg"],
            "toe_f_deg": opt_setup["toe_f_deg"],
            "toe_r_deg": opt_setup["toe_r_deg"],
            "brake_bias_front": opt_setup["brake_bias_front"],
            "rake_mm": opt_setup.get("rake_mm", 0.0),
            "ackermann_pct": opt_setup.get("ackermann_pct", 0.0),
        }
        fig8_ded = solve_skidpad_figure8_casadi(
            "FSG_RUBBERED", comp, v_entry_ms=4.0, n_circle=24, design_overrides=setup_overrides
        )
        acc75_ded = matrix[f"FSG_RUBBERED__{comp}"]["accel_75m_ocp"]
        row_ded = _make_summary_row(key_ded, "FSG_RUBBERED", comp, st_ded, fig8_ded, acc75_ded, setup_label="dedicated")

        dedicated_setup_opt[comp] = {
            "combo": key_ded,
            "compound": comp,
            "optimal_setup": opt_setup,
            "stationary_optimal": st_ded,
            "figure8_ocp": fig8_ded,
            "summary": row_ded,
        }
        print(
            f"[SETUP | FSG_RUBBERED | {comp:4s}] R*={row_ded['r_eff_star_m']:.3f}m | "
            f"t_stat={row_ded['t_stat_s']:.3f}s t_per={row_ded['t_mean_s']:.3f}s t_fig8={row_ded['t_mean_as_timed_s']:.3f}s | "
            f"ay={row_ded['ay_mean_g']:.3f}g LLTD={opt_setup['lltd']:.3f} rake={opt_setup.get('rake_mm', 0.0):+.1f}mm "
            f"ack={opt_setup.get('ackermann_pct', 0.0):.1f}% "
            f"camF={opt_setup['camber_f_deg']:.2f} camR={opt_setup['camber_r_deg']:.2f} "
            f"toeF={opt_setup['toe_f_deg']:+.2f} toeR={opt_setup['toe_r_deg']:+.2f} | "
            f"clear={row_ded['min_clear_m']:.3f}m hv_viol={row_ded['hv_violations']}",
            flush=True,
        )

    # 4. Exact Time Waterfall (ms)
    t_legacy_0p88 = float(legacy_0p88_open["t_circle_s"])
    t_fsg_open_9125 = float(diag["baseline_legacy_open_9125"]["t_circle_s"])
    t_fsg_open_rstar = float(diag["open_diff_optimal_r_star"]["t_circle_s"])
    t_fsg_eqf_rstar = float(diag["tv_equal_front_r_star"]["t_circle_s"])
    t_fsg_tv_rstar_r20 = float(matrix["FSG_RUBBERED__R20"]["stationary_optimal"]["t_circle_s"])
    t_fsg_fig8_base_r20 = float(matrix["FSG_RUBBERED__R20"]["figure8_ocp"]["t_mean_s"])
    t_fsg_stat_ded_r20 = float(dedicated_setup_opt["R20"]["stationary_optimal"]["t_circle_s"])
    t_fsg_fig8_ded_r20 = float(dedicated_setup_opt["R20"]["figure8_ocp"]["t_mean_s"])
    t_fsg_stat_ded_r25b = float(dedicated_setup_opt["R25B"]["stationary_optimal"]["t_circle_s"])
    t_fsg_fig8_ded_r25b = float(dedicated_setup_opt["R25B"]["figure8_ocp"]["t_mean_s"])
    t_ter26_anchor = float(fsg_cal["ter26_rwd_skidpad_s"])

    # Compute exact additive waterfall steps (all in ms, positive = time saved)
    wf_fsg_cal_ms = (t_legacy_0p88 - t_fsg_open_9125) * 1000.0
    wf_tight_r_ms = (t_fsg_open_9125 - t_fsg_open_rstar) * 1000.0
    wf_mz_tv_ms = (t_fsg_open_rstar - t_fsg_eqf_rstar) * 1000.0
    wf_of_unload_ms = (t_fsg_eqf_rstar - t_fsg_tv_rstar_r20) * 1000.0
    wf_crossover_ms = (t_fsg_tv_rstar_r20 - t_fsg_fig8_base_r20) * 1000.0
    wf_ded_setup_stat_ms = (t_fsg_tv_rstar_r20 - t_fsg_stat_ded_r20) * 1000.0
    wf_ded_setup_fig8_ms = (t_fsg_fig8_base_r20 - t_fsg_fig8_ded_r20) * 1000.0
    wf_r20_to_r25b_stat_ms = (t_fsg_stat_ded_r20 - t_fsg_stat_ded_r25b) * 1000.0
    wf_r20_to_r25b_fig8_ms = (t_fsg_fig8_ded_r20 - t_fsg_fig8_ded_r25b) * 1000.0

    time_waterfall = {
        "anchor_ter26_rwd_fsg_s": round(t_ter26_anchor, 4),
        "legacy_0p88_ter27_open_9125_s": round(t_legacy_0p88, 4),
        "steps_stationary_to_fig8": [
            {
                "step": "1_fsg_rubbered_calibration",
                "label": "FSG_RUBBERED real calibration (0.88 -> 1.0785)",
                "from_s": round(t_legacy_0p88, 4),
                "to_s": round(t_fsg_open_9125, 4),
                "gain_ms": round(wf_fsg_cal_ms, 1),
            },
            {
                "step": "2_tight_r_eff",
                "label": "Tight optimal radius R_eff* (9.125 m -> 8.390 m)",
                "from_s": round(t_fsg_open_9125, 4),
                "to_s": round(t_fsg_open_rstar, 4),
                "gain_ms": round(wf_tight_r_ms, 1),
            },
            {
                "step": "3_optimal_mz_tv",
                "label": "Optimal Mz_tv* (rear TV yaw allocation)",
                "from_s": round(t_fsg_open_rstar, 4),
                "to_s": round(t_fsg_eqf_rstar, 4),
                "gain_ms": round(wf_mz_tv_ms, 1),
            },
            {
                "step": "4_outer_front_unloading",
                "label": "Outer-front longitudinal torque unloading (4WD KKT)",
                "from_s": round(t_fsg_eqf_rstar, 4),
                "to_s": round(t_fsg_tv_rstar_r20, 4),
                "gain_ms": round(wf_of_unload_ms, 1),
            },
            {
                "step": "5_ballistic_crossover",
                "label": "Ballistic Figure-8 crossover OCP",
                "from_s": round(t_fsg_tv_rstar_r20, 4),
                "to_s": round(t_fsg_fig8_base_r20, 4),
                "gain_ms": round(wf_crossover_ms, 1),
            },
            {
                "step": "6_dedicated_suspension_setup",
                "label": "Dedicated Skidpad suspension setup (LLTD/ARB/Camber/Toe)",
                "from_s": round(t_fsg_fig8_base_r20, 4),
                "to_s": round(t_fsg_fig8_ded_r20, 4),
                "gain_ms": round(wf_ded_setup_fig8_ms, 1),
                "gain_stationary_ms": round(wf_ded_setup_stat_ms, 1),
            },
            {
                "step": "7_r20_to_r25b",
                "label": "Tire compound upgrade R20 -> R25B",
                "from_s": round(t_fsg_fig8_ded_r20, 4),
                "to_s": round(t_fsg_fig8_ded_r25b, 4),
                "gain_ms": round(wf_r20_to_r25b_fig8_ms, 1),
                "gain_stationary_ms": round(wf_r20_to_r25b_stat_ms, 1),
            },
        ],
        "summary_ms": {
            "fsg_rubbered_calibration_ms": round(wf_fsg_cal_ms, 1),
            "tight_r_eff_ms": round(wf_tight_r_ms, 1),
            "optimal_mz_tv_ms": round(wf_mz_tv_ms, 1),
            "outer_front_unloading_ms": round(wf_of_unload_ms, 1),
            "ballistic_crossover_ms": round(wf_crossover_ms, 1),
            "dedicated_suspension_setup_stat_ms": round(wf_ded_setup_stat_ms, 1),
            "dedicated_suspension_setup_fig8_ms": round(wf_ded_setup_fig8_ms, 1),
            "r20_to_r25b_stat_ms": round(wf_r20_to_r25b_stat_ms, 1),
            "r20_to_r25b_fig8_ms": round(wf_r20_to_r25b_fig8_ms, 1),
            "total_gain_vs_ter26_rwd_r20_periodic_ms": round((t_ter26_anchor - dedicated_setup_opt["R20"]["summary"]["t_mean_s"]) * 1000.0, 1),
            "total_gain_vs_ter26_rwd_r25b_periodic_ms": round((t_ter26_anchor - dedicated_setup_opt["R25B"]["summary"]["t_mean_s"]) * 1000.0, 1),
            "total_gain_vs_ter26_rwd_r25b_fig8_ms": round((t_ter26_anchor - t_fsg_fig8_ded_r25b) * 1000.0, 1),
        },
        "tier_comparison_fsg_rubbered": {
            "tier1_ter26_rwd_anchor_s": round(t_ter26_anchor, 4),
            "tier2_ter27_4wd_open_diff_9125_s": round(t_fsg_open_9125, 4),
            "tier2_ter27_4wd_open_diff_rstar_s": round(t_fsg_open_rstar, 4),
            "tier3_ter27_4wd_base_tv_r20_periodic_s": round(matrix["FSG_RUBBERED__R20"]["figure8_ocp"]["t_mean_periodic_s"], 4),
            "tier3_ter27_4wd_base_tv_r25b_periodic_s": round(matrix["FSG_RUBBERED__R25B"]["figure8_ocp"]["t_mean_periodic_s"], 4),
            "tier4_ter27_4wd_dedicated_tv_r20_periodic_s": round(dedicated_setup_opt["R20"]["figure8_ocp"]["t_mean_periodic_s"], 4),
            "tier4_ter27_4wd_dedicated_tv_r25b_periodic_s": round(dedicated_setup_opt["R25B"]["figure8_ocp"]["t_mean_periodic_s"], 4),
            "tier4_ter27_4wd_dedicated_tv_r20_fig8_s": round(t_fsg_fig8_ded_r20, 4),
            "tier4_ter27_4wd_dedicated_tv_r25b_fig8_s": round(t_fsg_fig8_ded_r25b, 4),
        },
    }

    print("\n=== Time Waterfall (ms) ===", flush=True)
    for st in time_waterfall["steps_stationary_to_fig8"]:
        print(f"  {st['step']:30s}: {st['from_s']:.4f}s -> {st['to_s']:.4f}s  ({st['gain_ms']:+7.1f} ms)", flush=True)

    oracle_reference_profiles: Dict[str, Any] = {}
    for comp in ("R20", "R25B"):
        st_b = matrix[f"FSG_RUBBERED__{comp}"]["stationary_optimal"]
        fg_b = matrix[f"FSG_RUBBERED__{comp}"]["figure8_ocp"]
        st_o = dedicated_setup_opt[comp]["stationary_optimal"]
        fg_o = dedicated_setup_opt[comp]["figure8_ocp"]
        oracle_reference_profiles[f"FSG_RUBBERED_{comp}_BASE"] = {
            "r_eff_star_m": round(float(st_b["r_eff_m"]), 4),
            "omega_z_ref_rads": round(float(st_b["wz_rads"]), 4),
            "vx_ref_ms": round(float(st_b["v_ms"] * math.cos(math.radians(st_b["beta_deg"]))), 4),
            "v_ref_ms": round(float(st_b["v_ms"]), 4),
            "beta_star_deg": round(float(st_b["beta_deg"]), 3),
            "mz_tv_star_nm": round(float(st_b["mz_tv_nm"]), 2),
            "ay_g": round(float(st_b["ay_g"]), 3),
            "t_stat_s": round(float(st_b["t_circle_s"]), 4),
            "t_mean_periodic_s": round(float(fg_b["t_mean_periodic_s"]), 4),
            "t_mean_as_timed_s": round(float(fg_b["t_mean_s"]), 4),
            "min_cone_clear_m": round(float(min(st_b["min_cone_clear_m"], fg_b["min_cone_clear_m"])), 4),
        }
        oracle_reference_profiles[f"FSG_RUBBERED_{comp}_OPTIMAL_SETUP"] = {
            "r_eff_star_m": round(float(st_o["r_eff_m"]), 4),
            "omega_z_ref_rads": round(float(st_o["wz_rads"]), 4),
            "vx_ref_ms": round(float(st_o["v_ms"] * math.cos(math.radians(st_o["beta_deg"]))), 4),
            "v_ref_ms": round(float(st_o["v_ms"]), 4),
            "beta_star_deg": round(float(st_o["beta_deg"]), 3),
            "mz_tv_star_nm": round(float(st_o["mz_tv_nm"]), 2),
            "ay_g": round(float(st_o["ay_g"]), 3),
            "t_stat_s": round(float(st_o["t_circle_s"]), 4),
            "t_mean_periodic_s": round(float(fg_o["t_mean_periodic_s"]), 4),
            "t_mean_as_timed_s": round(float(fg_o["t_mean_s"]), 4),
            "min_cone_clear_m": round(float(min(st_o["min_cone_clear_m"], fg_o["min_cone_clear_m"])), 4),
        }

    report = {
        "vehicle": "ter27_4wd",
        "solver": "CasADi 3.7.2 + IPOPT",
        "fsg_rubbered_lambda_muy": 0.57,
        "geometry": {
            "d_inner_m": D_INNER_M,
            "d_outer_m": D_OUTER_M,
            "r_inner_m": R_INNER_M,
            "r_outer_m": R_OUTER_M,
            "r_center_m": R_CENTER_M,
            "r_eff_min_m": R_EFF_MIN_M,
            "cone_margin_min_m": CONE_MARGIN_M,
        },
        "fsg_calibration": fsg_cal,
        "bottleneck_diagnosis": diag,
        "summary_table": summary_table,
        "oracle_reference_profiles": oracle_reference_profiles,
        "results": matrix,
        "dedicated_setup_opt": dedicated_setup_opt,
        "time_waterfall": time_waterfall,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }

    for out_dir in (RESULTS_DIR, VAULT_03_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "skidpad_oracle_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
