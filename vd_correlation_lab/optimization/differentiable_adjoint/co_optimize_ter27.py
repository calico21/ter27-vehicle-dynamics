#!/usr/bin/env python3
"""
End-to-End Gradient-Based Joint Mechanical + Control Co-Optimization for TeR27 (4WD).

Uses L-BFGS-B driven by exact discrete reverse-mode adjoint gradients from
DiscreteAdjointEngine (compiled C library libter_diff_14dof.so + Diff4WDController)
across all three Formula Student dynamic events simultaneously:
  1. Skidpad (FSG circle, min_clear >= 0.150 m, 0 cones)
  2. Acceleration (75 m launch, <= 80.0 kW HV DC)
  3. Autocross (multi-corner braking hairpin + S-chicane + exit, [-25, +80] kW, BSPD EV 2.3)

Strictly fixes vehicle mass, inertia, total downforce/drag, and tire grip
(COLD_TEST_OCT = 0.47, FSG_RUBBERED = 0.57 untouched), optimizing only real
trackside mechanical setup parameters and 4WD AL-QP / aero-pitch control gains.

Saves complete structured report to:
  outputs/differentiable_adjoint_report.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    DEFAULT_SO_PATH,
    FSG_RUBBERED,
    PARAM_IDX,
)
from vd_correlation_lab.optimization.differentiable_adjoint.diff_4wd_controller import (
    CTRL_PARAM_BOUNDS,
    CTRL_PARAM_NAMES,
)
from vd_correlation_lab.optimization.differentiable_adjoint.discrete_adjoint import (
    MIN_CONE_CLEAR_M,
    DiscreteAdjointEngine,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = REPO_ROOT / "outputs" / "differentiable_adjoint_report.json"

# Physically tunable TeR27 trackside mechanical setup parameters (grip/mass/ClA fixed!)
OPT_VEH_BOUNDS: Dict[str, Tuple[float, float]] = {
    "k_spring_f": (26000.0, 50000.0),
    "k_spring_r": (34000.0, 65000.0),
    "k_arb_f": (1500.0, 12000.0),
    "k_arb_r": (1500.0, 14000.0),
    "c_damp_f_scale": (0.65, 1.45),
    "c_damp_r_scale": (0.65, 1.45),
    "camber_f_rad": (math.radians(-2.5), math.radians(-0.5)),
    "camber_r_rad": (math.radians(-2.2), math.radians(-0.5)),
    "ackermann_factor": (-0.25, 0.40),
    "aero_balance_front": (0.42, 0.54),
}

OPT_CTRL_KEYS: List[str] = [
    "k_us_deg_g",
    "k_ff",
    "yaw_agility_gain",
    "ki_yaw",
    "kd_yaw",
    "k_beta_scale",
    "k_beta_dot_scale",
    "skidpad_mz_ff_gain",
    "fz_shed_gain",
    "alqp_w_mz",
    "alqp_w_fx",
    "alqp_w_reg",
    "aero_pitch_brake_gain",
    "aero_pitch_accel_gain",
]


def _build_opt_mapping(
    eng: DiscreteAdjointEngine,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    specs: List[Dict[str, Any]] = []
    z0_list: List[float] = []
    lo_list: List[float] = []
    hi_list: List[float] = []

    for name, (lo, hi) in OPT_VEH_BOUNDS.items():
        idx = PARAM_IDX[name]
        val = float(eng.theta_veh_nom[idx])
        specs.append({"name": name, "group": "vehicle", "idx": idx, "lo": lo, "hi": hi})
        z0_list.append(val)
        lo_list.append(lo)
        hi_list.append(hi)

    for name in OPT_CTRL_KEYS:
        idx = CTRL_PARAM_NAMES.index(name)
        lo, hi = CTRL_PARAM_BOUNDS[name]
        val = float(eng.theta_ctrl_nom[idx])
        specs.append({"name": name, "group": "control", "idx": idx, "lo": lo, "hi": hi})
        z0_list.append(val)
        lo_list.append(lo)
        hi_list.append(hi)

    return (
        specs,
        np.asarray(z0_list, dtype=np.float64),
        np.asarray(lo_list, dtype=np.float64),
        np.asarray(hi_list, dtype=np.float64),
    )


def _unpack_normalized(
    u_norm: np.ndarray,
    specs: List[Dict[str, Any]],
    lo: np.ndarray,
    hi: np.ndarray,
    th_v_base: np.ndarray,
    th_c_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    z_phys = lo + np.clip(u_norm, 0.0, 1.0) * (hi - lo)
    th_v = th_v_base.copy()
    th_c = th_c_base.copy()
    for i, sp in enumerate(specs):
        if sp["group"] == "vehicle":
            th_v[sp["idx"]] = z_phys[i]
        else:
            th_c[sp["idx"]] = z_phys[i]
    return th_v, th_c


def _summarize_event(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_time_s": round(float(res["event_time_s"]), 5),
        "cost": round(float(res["cost"]), 5),
        "stage_penalty": round(float(res["stage_penalty"]), 6),
        "p_dc_min_kw": round(float(res["p_dc_min_kw"]), 3),
        "p_dc_max_kw": round(float(res["p_dc_max_kw"]), 3),
        "hv_violations": int(res["hv_violations"]),
        "bspd_violations": int(res["bspd_violations"]),
        "min_clear_m": round(float(res["min_clear_m"]), 4) if res["min_clear_m"] is not None else None,
        "cones_hit": int(res["cones_hit"]),
        "max_dmz_dbeta_nm_rad": round(float(res["max_dmz_dbeta_nm_rad"]), 2),
        "yaw_stable": bool(res["yaw_stable"]),
        "min_rh_f_mm": round(float(res["min_rh_f_mm"]), 2),
        "max_ay_g": round(float(res["max_ay_g"]), 3),
        "max_vx_ms": round(float(res["max_vx_ms"]), 3),
    }


def run_ter27_joint_co_optimization(
    maxiter: int = 18,
    report_path: Path = DEFAULT_REPORT_PATH,
    grip_profile: str = "FSG_RUBBERED",
    tire_compound: str = "R25B",
    suspension_mode: str = "3d_double_wishbone_lut",
) -> Dict[str, Any]:
    """
    Execute full verification, sensitivity analysis, and L-BFGS-B adjoint
    joint co-optimization for TeR27 4WD, saving structured JSON report.
    """
    t_start = time.perf_counter()
    print("=== TeR27 4WD End-to-End Differentiable Adjoint Engine ===", flush=True)
    eng = DiscreteAdjointEngine(
        grip_profile=grip_profile,
        tire_compound=tire_compound,
        suspension_mode=suspension_mode,
        dt=0.005,
    )

    # 1. Exact Adjoint vs Central FD Verification (< 1e-6) across all 3 events
    print("[1/4] Verifying discrete reverse-mode adjoint vs central finite differences...", flush=True)
    fd_checks: Dict[str, Any] = {}
    worst_abs_err = 0.0
    worst_rel_err = 0.0
    all_fd_passed = True
    for ev in ("skidpad", "acceleration", "autocross"):
        chk = eng.verify_adjoint_vs_fd(event=ev, n_steps=80)
        fd_checks[ev] = chk
        worst_abs_err = max(worst_abs_err, float(chk["max_abs_err"]))
        worst_rel_err = max(worst_rel_err, float(chk["max_rel_err"]))
        all_fd_passed = all_fd_passed and bool(chk["passed_1e_6"])
        print(
            f"  FD [{ev:12s}] max_abs_err={chk['max_abs_err']:.3e} | "
            f"max_rel_err={chk['max_rel_err']:.3e} | passed={chk['passed_1e_6']}",
            flush=True,
        )

    # 2. Full 45-Parameter Sensitivity Rankings at Nominal TeR27 Baseline
    print("[2/4] Computing full-horizon adjoint sensitivity rankings (45 params x 3 events)...", flush=True)
    sens_baseline = eng.compute_sensitivity_rankings()
    for ev in ("skidpad", "acceleration", "autocross"):
        ev_d = sens_baseline["events"][ev]
        top3 = ", ".join(
            f"{r['param']}({r['dT_ms_per_char_delta']:+.1f}ms)"
            for r in ev_d["rankings"][:3]
        )
        print(
            f"  SENS [{ev:12s}] t={ev_d['event_time_s']:.4f}s (wall={ev_d['adjoint_wall_s']:.2f}s) | top3: {top3}",
            flush=True,
        )

    # 3. Joint Mechanical + Control Co-Optimization via L-BFGS-B + Reverse-Mode Adjoint
    print(f"[3/4] Running L-BFGS-B joint co-optimization (maxiter={maxiter}, 24 params)...", flush=True)
    specs, z0, lo, hi = _build_opt_mapping(eng)
    u0 = np.clip((z0 - lo) / (hi - lo), 0.0, 1.0)
    span = hi - lo

    event_weights = {"skidpad": 1.0, "acceleration": 1.0, "autocross": 1.0}
    history: List[Dict[str, Any]] = []
    eval_cache: Dict[str, Any] = {"best_cost": 1e18, "best_u": u0.copy(), "n_evals": 0}

    def _eval_joint(u_norm: np.ndarray) -> Tuple[float, np.ndarray]:
        th_v, th_c = _unpack_normalized(u_norm, specs, lo, hi, eng.theta_veh_nom, eng.theta_ctrl_nom)
        j_tot = 0.0
        g_v_tot = np.zeros_like(th_v)
        g_c_tot = np.zeros_like(th_c)
        ev_times: Dict[str, float] = {}
        ev_res: Dict[str, Dict[str, Any]] = {}

        for ev, w_ev in event_weights.items():
            res = eng.rollout_and_adjoint(ev, theta_veh=th_v, theta_ctrl=th_c, compute_grad=True)
            j_tot += w_ev * float(res["cost"])
            g_v_tot += w_ev * res["grad_theta_veh"]
            g_c_tot += w_ev * res["grad_theta_ctrl"]
            ev_times[ev] = float(res["event_time_s"])
            ev_res[ev] = res

        # Chain rule to normalized decision variables u_norm in [0, 1]
        g_u = np.zeros(len(specs), dtype=np.float64)
        for i, sp in enumerate(specs):
            if sp["group"] == "vehicle":
                g_u[i] = g_v_tot[sp["idx"]] * span[i]
            else:
                g_u[i] = g_c_tot[sp["idx"]] * span[i]

        eval_cache["n_evals"] += 1
        feasible = all(
            res["hv_violations"] == 0
            and res["bspd_violations"] == 0
            and res["yaw_stable"]
            and (res["min_clear_m"] is None or res["min_clear_m"] >= MIN_CONE_CLEAR_M)
            for res in ev_res.values()
        )
        if feasible and j_tot < eval_cache["best_cost"]:
            eval_cache["best_cost"] = float(j_tot)
            eval_cache["best_u"] = u_norm.copy()

        history.append({
            "eval": int(eval_cache["n_evals"]),
            "joint_cost": round(float(j_tot), 5),
            "grad_norm": round(float(np.linalg.norm(g_u)), 5),
            "skidpad_s": round(ev_times["skidpad"], 5),
            "acceleration_s": round(ev_times["acceleration"], 5),
            "autocross_s": round(ev_times["autocross"], 5),
            "feasible": bool(feasible),
        })
        print(
            f"  eval {eval_cache['n_evals']:02d} | J={j_tot:.5f} | |g|={np.linalg.norm(g_u):.3e} | "
            f"skid={ev_times['skidpad']:.4f}s acc={ev_times['acceleration']:.4f}s "
            f"acx={ev_times['autocross']:.4f}s | feas={feasible}",
            flush=True,
        )
        return float(j_tot), g_u

    opt_res = minimize(
        _eval_joint,
        u0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(specs),
        options={"maxiter": int(maxiter), "ftol": 1e-7, "gtol": 1e-6},
    )

    u_star = eval_cache["best_u"]
    th_v_opt, th_c_opt = _unpack_normalized(u_star, specs, lo, hi, eng.theta_veh_nom, eng.theta_ctrl_nom)

    # 4. Certify Baseline vs Optimized Full-Horizon Rollouts
    print("[4/4] Certifying baseline vs optimized TeR27 performance and constraints...", flush=True)
    baseline_metrics: Dict[str, Any] = {}
    optimized_metrics: Dict[str, Any] = {}
    for ev in ("skidpad", "acceleration", "autocross"):
        r_base = eng.rollout_and_adjoint(ev, theta_veh=eng.theta_veh_nom, theta_ctrl=eng.theta_ctrl_nom, compute_grad=False)
        r_opt = eng.rollout_and_adjoint(ev, theta_veh=th_v_opt, theta_ctrl=th_c_opt, compute_grad=False)
        baseline_metrics[ev] = _summarize_event(r_base)
        optimized_metrics[ev] = _summarize_event(r_opt)

    improvements = {
        "skidpad_gain_ms": round((baseline_metrics["skidpad"]["event_time_s"] - optimized_metrics["skidpad"]["event_time_s"]) * 1000.0, 2),
        "acceleration_gain_ms": round((baseline_metrics["acceleration"]["event_time_s"] - optimized_metrics["acceleration"]["event_time_s"]) * 1000.0, 2),
        "autocross_gain_ms": round((baseline_metrics["autocross"]["event_time_s"] - optimized_metrics["autocross"]["event_time_s"]) * 1000.0, 2),
    }
    improvements["total_combined_gain_ms"] = round(
        improvements["skidpad_gain_ms"]
        + improvements["acceleration_gain_ms"]
        + improvements["autocross_gain_ms"],
        2,
    )

    opt_param_table: List[Dict[str, Any]] = []
    for i, sp in enumerate(specs):
        val_before = float(z0[i])
        val_after = float(lo[i] + u_star[i] * span[i])
        opt_param_table.append({
            "param": sp["name"],
            "group": sp["group"],
            "baseline": round(val_before, 6),
            "optimized": round(val_after, 6),
            "delta": round(val_after - val_before, 6),
            "lower_bound": round(float(sp["lo"]), 6),
            "upper_bound": round(float(sp["hi"]), 6),
        })

    hv_ok = all(
        optimized_metrics[ev]["hv_violations"] == 0
        and optimized_metrics[ev]["p_dc_min_kw"] >= -25.0
        and optimized_metrics[ev]["p_dc_max_kw"] <= 80.0
        for ev in ("skidpad", "acceleration", "autocross")
    )
    bspd_ok = all(optimized_metrics[ev]["bspd_violations"] == 0 for ev in ("skidpad", "acceleration", "autocross"))
    skidpad_ok = (
        optimized_metrics["skidpad"]["min_clear_m"] >= MIN_CONE_CLEAR_M
        and optimized_metrics["skidpad"]["cones_hit"] == 0
    )
    yaw_ok = all(optimized_metrics[ev]["yaw_stable"] for ev in ("skidpad", "acceleration", "autocross"))

    wall_total_s = time.perf_counter() - t_start

    report: Dict[str, Any] = {
        "metadata": {
            "vehicle_id": "ter27_4wd",
            "model_id": "differentiable_casadi_14dof",
            "suspension_mode": suspension_mode,
            "tire_compound": tire_compound,
            "grip_profile": grip_profile,
            "cold_test_oct_lambda_muy": COLD_TEST_OCT,
            "fsg_rubbered_lambda_muy": FSG_RUBBERED,
            "compiled_library": str(DEFAULT_SO_PATH.relative_to(REPO_ROOT)),
            "dt_s": eng.dt,
            "optimizer": "L-BFGS-B (exact reverse-mode discrete adjoint)",
            "n_tuned_params": len(specs),
            "n_sensitivity_params": len(sens_baseline["combined_ranking"]),
            "total_wall_s": round(wall_total_s, 2),
        },
        "adjoint_fd_verification": {
            "worst_max_abs_err": worst_abs_err,
            "worst_max_rel_err": worst_rel_err,
            "all_passed_1e_6": bool(all_fd_passed),
            "per_event": fd_checks,
        },
        "sensitivity_rankings": sens_baseline,
        "co_optimization": {
            "scipy_success": bool(opt_res.success),
            "scipy_nit": int(opt_res.nit),
            "scipy_nfev": int(eval_cache["n_evals"]),
            "scipy_message": str(opt_res.message),
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": optimized_metrics,
            "improvements_ms": improvements,
            "optimal_parameters": opt_param_table,
            "convergence_history": history,
            "constraints_certified": {
                "hv_dc_strict_minus25_to_80kw": bool(hv_ok),
                "bspd_ev23_zero_violations": bool(bspd_ok),
                "skidpad_min_clear_ge_0p150m_zero_cones": bool(skidpad_ok),
                "yaw_stability_dmz_dbeta_lt_0": bool(yaw_ok),
                "all_constraints_passed": bool(hv_ok and bspd_ok and skidpad_ok and yaw_ok),
            },
        },
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n=== Co-Optimization Summary ===", flush=True)
    print(
        f"  Skidpad      : {baseline_metrics['skidpad']['event_time_s']:.4f}s -> "
        f"{optimized_metrics['skidpad']['event_time_s']:.4f}s ({improvements['skidpad_gain_ms']:+.2f} ms) | "
        f"clear={optimized_metrics['skidpad']['min_clear_m']:.3f}m | cones={optimized_metrics['skidpad']['cones_hit']}",
        flush=True,
    )
    print(
        f"  Acceleration : {baseline_metrics['acceleration']['event_time_s']:.4f}s -> "
        f"{optimized_metrics['acceleration']['event_time_s']:.4f}s ({improvements['acceleration_gain_ms']:+.2f} ms) | "
        f"Pdc_max={optimized_metrics['acceleration']['p_dc_max_kw']:.2f}kW",
        flush=True,
    )
    print(
        f"  Autocross    : {baseline_metrics['autocross']['event_time_s']:.4f}s -> "
        f"{optimized_metrics['autocross']['event_time_s']:.4f}s ({improvements['autocross_gain_ms']:+.2f} ms) | "
        f"Pdc=[{optimized_metrics['autocross']['p_dc_min_kw']:.2f},{optimized_metrics['autocross']['p_dc_max_kw']:.2f}]kW",
        flush=True,
    )
    print(
        f"  Combined Gain: {improvements['total_combined_gain_ms']:+.2f} ms | "
        f"All Constraints Certified: {report['co_optimization']['constraints_certified']['all_constraints_passed']} | "
        f"Saved: {report_path}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_ter27_joint_co_optimization(maxiter=18)
