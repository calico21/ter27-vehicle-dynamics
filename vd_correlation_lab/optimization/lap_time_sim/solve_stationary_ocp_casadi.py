#!/usr/bin/env python3
"""
CasADi 3.7.2 + IPOPT Cross-Validation of the Stationary Physical Oracle NLP
for Tecnun e-Racing TeR27 4WD (MirenaSim).

Maximizes steady-state cornering lateral acceleration a_y*(R) = V^2 / R
over free variables (V, delta, beta, kappa_FL, kappa_FR, kappa_RL, kappa_RR)
using exact symbolic Jacobians and Hessians (IPOPT + MUMPS), and cross-validates
all 4 radii (R = 5.0, 9.125, 15.0, 25.0 m) under BOTH FSG_REAL_TRACK and
TTC_LAB_PEAK against the SciPy SLSQP report:
  vd_correlation_lab/results/tv_oracle/stationary_oracle_report.json

Also verifies every solution against the exact non-smoothed ter27_physics model
to guarantee 0 HV violations in [-25.0, +80.0] kW.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
sys.path.insert(0, str(ORACLE_DIR))

from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import TeR27CasADiVehicle, build_casadi_vehicle  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import TeR27PhysicsModel  # noqa: E402

RADII_M = [5.0, 9.125, 15.0, 25.0]
PROFILES = ["FSG_REAL_TRACK", "TTC_LAB_PEAK"]


def build_stationary_solver(
    veh: TeR27CasADiVehicle,
    open_diff: bool = False,
) -> Tuple[ca.Function, Dict[str, Any]]:
    """
    Build a parameterized CasADi+IPOPT NLP solver (parameter p = [radius_m])
    with exact Hessian and MUMPS linear solver.
    """
    p = veh.params
    mg = p.mass * p.gravity
    mgl = mg * p.wheelbase
    p_max_kw = (p.p_dc_max_w - 50.0) * 1e-3  # 50 W safety margin -> strictly <= 80.0 kW
    p_min_kw = (p.p_dc_min_w + 50.0) * 1e-3  # 50 W safety margin -> strictly >= -25.0 kW

    # Decision variables (scaled to O(1) for crisp IPOPT conditioning)
    # x_sc = [V/10, delta/0.3, beta/0.1, kFL/0.1, kFR/0.1, kRL/0.1, kRR/0.1]
    scale = np.array([10.0, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
    x_sc = ca.SX.sym("x_sc", 7)
    x = x_sc * ca.DM(scale)

    radius_sym = ca.SX.sym("radius_m")

    V = x[0]
    delta = x[1]
    beta = x[2]
    kappa_4 = x[3:7]

    vx = V * ca.cos(beta)
    vy = V * ca.sin(beta)
    r = V / radius_sym
    ax_ss = -r * vy
    ay_ss = r * vx

    # At steady-state equilibrium Fx_tot/m == ax_ss and Fy_tot/m == ay_ss,
    # so passing ax_lt = ax_ss, ay_lt = ay_ss is exact with zero extra variables.
    ev = veh.eval_forces_from_kappa(vx, vy, r, delta, kappa_4, 0.0, ax_ss, ay_ss)

    eq_fx = (ev["Fx_tot"] - p.mass * ax_ss) / mg
    eq_fy = (ev["Fy_tot"] - p.mass * ay_ss) / mg
    eq_mz = ev["Mz_tot"] / mgl

    t_mot = ev["T_motor_4"]
    t_max = ev["T_max_4"]
    t_min = ev["T_min_4"]
    p_ecu_kw = ev["P_dc_ecu_kw"]
    p_drv_kw = ev["P_dc_drv_kw"]

    g_list = [
        eq_fx,
        eq_fy,
        eq_mz,
        (t_max - t_mot) / 100.0,       # >= 0 (4)
        (t_mot - t_min) / 100.0,       # >= 0 (4)
        (p_max_kw - p_ecu_kw) / 10.0,  # >= 0
        (p_max_kw - p_drv_kw) / 10.0,  # >= 0
        (p_ecu_kw - p_min_kw) / 10.0,  # >= 0
        (p_drv_kw - p_min_kw) / 10.0,  # >= 0
    ]
    lbg = [0.0, 0.0, 0.0] + [0.0] * 12
    ubg = [0.0, 0.0, 0.0] + [ca.inf] * 12

    if open_diff:
        sf = p.front_drive_ratio
        sr = 1.0 - sf
        g_list.extend(
            [
                (t_mot[0] - t_mot[1]) / 100.0,
                (t_mot[2] - t_mot[3]) / 100.0,
                (sr * (t_mot[0] + t_mot[1]) - sf * (t_mot[2] + t_mot[3])) / 100.0,
                t_mot / 100.0,
            ]
        )
        lbg.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0, ca.inf, ca.inf, ca.inf, ca.inf])

    g = ca.vertcat(*g_list)
    f = -V / 10.0

    nlp = {"x": x_sc, "p": radius_sym, "f": f, "g": g}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-9,
        "ipopt.acceptable_tol": 1e-7,
        "ipopt.max_iter": 300,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("stat_oracle", "ipopt", nlp, opts)

    # Evaluation helper for extracting all physical diagnostics at x_sc*
    fn_diag = ca.Function(
        "stat_diag",
        [x_sc, radius_sym],
        [
            vx,
            vy,
            r,
            ev["Fx_tot"] - p.mass * ax_ss,
            ev["Fy_tot"] - p.mass * ay_ss,
            ev["Mz_tot"],
            ev["T_motor_4"],
            ev["T_max_4"],
            ev["T_min_4"],
            ev["Fz_4"],
            ev["Fx_w_4"],
            ev["Fy_w_4"],
            ev["alpha_4"],
            ev["delta_w_4"],
            ev["Fx_f_sin_delta"],
            ev["Mz_tv_long"],
            ev["P_dc_ecu_kw"],
            ev["P_dc_drv_kw"],
            ev["phi"],
        ],
        ["x_sc", "radius_m"],
        [
            "vx",
            "vy",
            "r",
            "eq_fx",
            "eq_fy",
            "eq_mz",
            "T_motor_4",
            "T_max_4",
            "T_min_4",
            "Fz_4",
            "Fx_w_4",
            "Fy_w_4",
            "alpha_4",
            "delta_w_4",
            "Fx_f_sin_delta",
            "Mz_tv_long",
            "P_dc_ecu_kw",
            "P_dc_drv_kw",
            "phi",
        ],
    )

    lbx_raw = np.array([3.0, 0.01, -0.22, -0.14, -0.14, -0.14, -0.14], dtype=float)
    ubx_raw = np.array([38.0, p.max_steer_angle, 0.14, 0.14, 0.14, 0.14, 0.14], dtype=float)

    meta = {
        "scale": scale,
        "lbx": (lbx_raw / scale).tolist(),
        "ubx": (ubx_raw / scale).tolist(),
        "lbg": lbg,
        "ubg": ubg,
        "lbx_raw": lbx_raw,
        "ubx_raw": ubx_raw,
        "fn_diag": fn_diag,
    }
    return solver, meta


def solve_radius_casadi(
    veh: TeR27CasADiVehicle,
    exact_model: TeR27PhysicsModel,
    solver: ca.Function,
    meta: Dict[str, Any],
    radius_m: float,
    open_diff: bool = False,
) -> Dict[str, Any]:
    """Multi-start IPOPT solve for a single radius."""
    p = veh.params
    mg = p.mass * p.gravity
    mgl = mg * p.wheelbase
    scale = meta["scale"]

    ay_guess_g = 1.10 if veh.grip_profile == "FSG_REAL_TRACK" else 1.40
    v_guess = math.sqrt(ay_guess_g * p.gravity * radius_m)
    delta_kin = min(math.atan(p.wheelbase / radius_m) * 1.15, p.max_steer_angle * 0.95)
    beta_kin = -math.atan(p.b_cog / radius_m)

    if open_diff:
        starts = [
            np.array([v_guess * 0.90, delta_kin, 0.02, 0.008, 0.004, 0.012, 0.006]),
            np.array([v_guess * 0.80, min(delta_kin * 1.15, p.max_steer_angle * 0.98), -0.02, 0.01, 0.005, 0.015, 0.008]),
            np.array([v_guess * 1.05, min(delta_kin * 1.05, p.max_steer_angle * 0.96), beta_kin, 0.012, 0.006, 0.018, 0.009]),
            np.array([v_guess * 0.95, min(delta_kin * 0.90, p.max_steer_angle * 0.92), 0.04, 0.005, 0.003, 0.008, 0.004]),
            np.array([v_guess * 0.85, min(delta_kin * 1.10, p.max_steer_angle * 0.92), 0.08, 0.006, 0.003, 0.010, 0.005]),
        ]
    else:
        starts = [
            np.array([v_guess * 0.95, delta_kin, 0.02, 0.015, 0.035, -0.005, -0.015]),
            np.array([v_guess * 1.08, min(delta_kin * 1.15, p.max_steer_angle * 0.98), beta_kin, 0.01, 0.04, -0.01, 0.02]),
            np.array([v_guess * 0.88, min(delta_kin * 1.00, p.max_steer_angle * 0.95), -0.03, 0.01, 0.02, 0.01, 0.03]),
        ]

    best: Dict[str, Any] | None = None
    best_v = -1.0
    t_all0 = time.perf_counter()

    for idx, x0_raw in enumerate(starts):
        x0_c = np.clip(x0_raw, meta["lbx_raw"], meta["ubx_raw"])
        t0 = time.perf_counter()
        sol = solver(
            x0=(x0_c / scale).tolist(),
            p=radius_m,
            lbx=meta["lbx"],
            ubx=meta["ubx"],
            lbg=meta["lbg"],
            ubg=meta["ubg"],
        )
        dt = time.perf_counter() - t0
        stats = solver.stats()
        x_opt = np.asarray(sol["x"]).ravel() * scale
        diag = meta["fn_diag"](x_sc=np.asarray(sol["x"]).ravel(), radius_m=radius_m)

        eq_fx = float(diag["eq_fx"])
        eq_fy = float(diag["eq_fy"])
        eq_mz = float(diag["eq_mz"])
        eq_err = max(abs(eq_fx) / mg, abs(eq_fy) / mg, abs(eq_mz) / mgl)
        t_mot = np.asarray(diag["T_motor_4"]).ravel()
        t_max = np.asarray(diag["T_max_4"]).ravel()
        t_min = np.asarray(diag["T_min_4"]).ravel()
        t_viol = max(float(np.max(t_mot - t_max)), float(np.max(t_min - t_mot)))
        p_ecu_kw = float(diag["P_dc_ecu_kw"])
        p_drv_kw = float(diag["P_dc_drv_kw"])
        hv_ok = (max(p_ecu_kw, p_drv_kw) <= 80.0) and (min(p_ecu_kw, p_drv_kw) >= -25.0)

        iters = stats.get("iterations", {})
        inf_pr = float(iters["inf_pr"][-1]) if "inf_pr" in iters and len(iters["inf_pr"]) > 0 else float("nan")
        inf_du = float(iters["inf_du"][-1]) if "inf_du" in iters and len(iters["inf_du"]) > 0 else float("nan")

        if eq_err < 1e-4 and t_viol < 0.5 and hv_ok:
            if float(x_opt[0]) > best_v:
                best_v = float(x_opt[0])
                best = {
                    "x_opt": x_opt,
                    "diag": diag,
                    "start_idx": idx,
                    "best_start_solve_time_s": dt,
                    "ipopt_status": str(stats.get("return_status", "unknown")),
                    "ipopt_success": bool(stats.get("success", False)),
                    "ipopt_iter_count": int(stats.get("iter_count", -1)),
                    "ipopt_inf_pr": inf_pr,
                    "ipopt_inf_du": inf_du,
                }

    total_multistart_time_s = time.perf_counter() - t_all0
    if best is None:
        raise RuntimeError(
            f"CasADi+IPOPT failed to converge feasible stationary solution for "
            f"{veh.grip_profile} R={radius_m}m (open={open_diff})"
        )

    best["multistart_solve_time_s"] = total_multistart_time_s

    # Cross-check exact non-smoothed ter27_physics model at x_opt
    x_opt = best["x_opt"]
    V, delta, beta = float(x_opt[0]), float(x_opt[1]), float(x_opt[2])
    kappa = np.asarray(x_opt[3:7], dtype=float)
    vx = V * math.cos(beta)
    vy = V * math.sin(beta)
    r = V / radius_m
    ax_ss = -r * vy
    ay_ss = r * vx
    exact_res = exact_model.evaluate_state_from_kappa(vx, vy, r, delta, kappa, ax_ss, ay_ss, n_load_iter=4)
    best["exact_check"] = {
        "eq_fx_n": float(exact_res["sum_fx"] - p.mass * ax_ss),
        "eq_fy_n": float(exact_res["sum_fy"] - p.mass * ay_ss),
        "eq_mz_nm": float(exact_res["sum_mz"]),
        "P_dc_ecu_kw": float(exact_res["p_dc_ecu_w"]) * 1e-3,
        "P_dc_drv_kw": float(exact_res["p_dc_drv_w"]) * 1e-3,
        "hv_violation_exact": bool(
            max(float(exact_res["p_dc_ecu_w"]), float(exact_res["p_dc_drv_w"])) > p.p_dc_max_w
            or min(float(exact_res["p_dc_ecu_w"]), float(exact_res["p_dc_drv_w"])) < p.p_dc_min_w
        ),
    }
    return best


def build_xval_record(
    veh: TeR27CasADiVehicle,
    radius_m: float,
    res_opt: Dict[str, Any],
    res_open: Dict[str, Any],
    scipy_rec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    p = veh.params
    g = p.gravity
    x_opt = res_opt["x_opt"]
    d = res_opt["diag"]
    x_open = res_open["x_opt"]
    d_open = res_open["diag"]

    V = float(x_opt[0])
    vx = float(d["vx"])
    vy = float(d["vy"])
    r = float(d["r"])
    ay_opt = (V * V) / radius_m
    ay_open = (float(x_open[0]) ** 2) / radius_m
    delta_rad = float(x_opt[1])
    beta_rad = float(x_opt[2])
    alpha_4 = np.asarray(d["alpha_4"]).ravel()
    t_mot = [float(v) for v in np.asarray(d["T_motor_4"]).ravel()]
    fz_4 = [float(v) for v in np.asarray(d["Fz_4"]).ravel()]
    fx_w = [float(v) for v in np.asarray(d["Fx_w_4"]).ravel()]
    fy_w = [float(v) for v in np.asarray(d["Fy_w_4"]).ravel()]
    delta_w = np.asarray(d["delta_w_4"]).ravel()
    fy_sin_d = float(d["Fx_f_sin_delta"])
    mz_tv = float(d["Mz_tv_long"])
    mz_from_fx_sin_delta = float(p.a_cog * fy_sin_d)
    mz_from_fx_lr_diff = float(np.sum(-veh.y_w * (np.asarray(fx_w) * np.cos(delta_w))))

    p_ecu_kw = float(d["P_dc_ecu_kw"])
    p_drv_kw = float(d["P_dc_drv_kw"])
    hv_viol_casadi = bool(max(p_ecu_kw, p_drv_kw) > 80.0 or min(p_ecu_kw, p_drv_kw) < -25.0)
    hv_viol_exact = bool(res_opt["exact_check"]["hv_violation_exact"])

    rec: Dict[str, Any] = {
        "profile": veh.grip_profile,
        "radius_m": radius_m,
        "v_total_mps": V,
        "v_x_mps": vx,
        "v_y_mps": vy,
        "r_rad_s": r,
        "a_y_mps2": ay_opt,
        "a_y_g": ay_opt / g,
        "delta_rad": delta_rad,
        "delta_deg": math.degrees(delta_rad),
        "beta_rad": beta_rad,
        "beta_deg": math.degrees(beta_rad),
        "alpha_f_deg": math.degrees(0.5 * float(alpha_4[0] + alpha_4[1])),
        "alpha_r_deg": math.degrees(0.5 * float(alpha_4[2] + alpha_4[3])),
        "kappa_wheels": [float(v) for v in x_opt[3:7]],
        "T_wheel_nm": t_mot,
        "Fz_wheel_n": fz_4,
        "Fx_wheel_n": fx_w,
        "Fy_wheel_n": fy_w,
        "M_z_TV_nm": mz_tv,
        "M_z_TV_lr_diff_nm": mz_from_fx_lr_diff,
        "M_z_TV_fx_sin_delta_nm": mz_from_fx_sin_delta,
        "F_xf_sin_delta_n": fy_sin_d,
        "a_y_from_fx_sin_delta_g": (fy_sin_d / p.mass) / g,
        "P_dc_kw": p_ecu_kw,
        "P_dc_drivetrain_kw": p_drv_kw,
        "hv_violation": hv_viol_casadi or hv_viol_exact,
        "open_diff_ref": {
            "v_x_mps": float(d_open["vx"]),
            "a_y_mps2": ay_open,
            "a_y_g": ay_open / g,
            "delta_deg": math.degrees(float(x_open[1])),
            "beta_deg": math.degrees(float(x_open[2])),
            "T_wheel_nm": [float(v) for v in np.asarray(d_open["T_motor_4"]).ravel()],
            "P_dc_kw": float(d_open["P_dc_ecu_kw"]),
            "ipopt_iter_count": res_open["ipopt_iter_count"],
            "solve_time_s": res_open["multistart_solve_time_s"],
        },
        "tv_gain_vs_open_g": (ay_opt - ay_open) / g,
        "tv_gain_pct": 100.0 * (ay_opt / max(ay_open, 1e-6) - 1.0),
        "ipopt": {
            "status": res_opt["ipopt_status"],
            "success": res_opt["ipopt_success"],
            "iter_count": res_opt["ipopt_iter_count"],
            "inf_pr": res_opt["ipopt_inf_pr"],
            "inf_du": res_opt["ipopt_inf_du"],
            "best_start_solve_time_s": res_opt["best_start_solve_time_s"],
            "multistart_solve_time_s": res_opt["multistart_solve_time_s"],
        },
        "residuals_casadi": {
            "eq_fx_n": float(d["eq_fx"]),
            "eq_fy_n": float(d["eq_fy"]),
            "eq_mz_nm": float(d["eq_mz"]),
        },
        "exact_physics_check": res_opt["exact_check"],
    }

    if scipy_rec is not None:
        t_sp = np.asarray(scipy_rec["T_wheel_nm"], dtype=float)
        t_ca = np.asarray(t_mot, dtype=float)
        rec["delta_vs_scipy"] = {
            "d_vx_mps": vx - float(scipy_rec["v_x_mps"]),
            "d_ay_g": (ay_opt / g) - float(scipy_rec["a_y_g"]),
            "d_ay_pct": 100.0 * ((ay_opt / g) / float(scipy_rec["a_y_g"]) - 1.0),
            "d_delta_deg": math.degrees(delta_rad) - float(scipy_rec["delta_deg"]),
            "d_beta_deg": math.degrees(beta_rad) - float(scipy_rec["beta_deg"]),
            "d_Mz_TV_nm": mz_tv - float(scipy_rec["M_z_TV_nm"]),
            "d_F_xf_sin_delta_n": fy_sin_d - float(scipy_rec["F_xf_sin_delta_n"]),
            "d_P_dc_kw": p_ecu_kw - float(scipy_rec["P_dc_kw"]),
            "max_abs_d_T_wheel_nm": float(np.max(np.abs(t_ca - t_sp))),
            "d_T_wheel_nm": [float(v) for v in (t_ca - t_sp)],
            "d_open_ay_g": (ay_open / g) - float(scipy_rec["open_diff_ref"]["a_y_g"]),
        }

    return rec


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scipy_path = RESULTS_DIR / "stationary_oracle_report.json"
    scipy_report = json.loads(scipy_path.read_text(encoding="utf-8")) if scipy_path.exists() else None

    report: Dict[str, Any] = {
        "vehicle_id": "ter27_4wd",
        "solver": "CasADi 3.7.2 + IPOPT (exact Hessian, MUMPS)",
        "radii_m": RADII_M,
        "profiles": {},
        "hv_violations_total": 0,
        "summary": {},
    }

    print("=== TeR27 4WD Stationary Physical Oracle (CasADi 3.7.2 + IPOPT) ===", flush=True)
    hv_viols = 0
    all_d_ay_g: List[float] = []
    all_d_ay_pct: List[float] = []
    total_solve_s = 0.0
    total_best_start_s = 0.0

    for prof in PROFILES:
        t_build0 = time.perf_counter()
        veh = build_casadi_vehicle(grip_profile=prof)
        exact_model = TeR27PhysicsModel(profile=prof)
        solver_opt, meta_opt = build_stationary_solver(veh, open_diff=False)
        solver_open, meta_open = build_stationary_solver(veh, open_diff=True)
        build_time_s = time.perf_counter() - t_build0

        scipy_list = scipy_report["profiles"][prof] if scipy_report is not None else [None] * len(RADII_M)
        prof_list: List[Dict[str, Any]] = []
        print(
            f"\n[Profile: {prof}] build={build_time_s:.3f}s | LMUX={veh.lmux:.4f}, LMUY={veh.lmuy:.4f}",
            flush=True,
        )

        for r_m, sp_rec in zip(RADII_M, scipy_list):
            res_opt = solve_radius_casadi(veh, exact_model, solver_opt, meta_opt, r_m, open_diff=False)
            res_open = solve_radius_casadi(veh, exact_model, solver_open, meta_open, r_m, open_diff=True)
            rec = build_xval_record(veh, r_m, res_opt, res_open, sp_rec)
            rec["solver_build_time_s"] = build_time_s
            if rec["hv_violation"]:
                hv_viols += 1
            total_solve_s += rec["ipopt"]["multistart_solve_time_s"]
            total_best_start_s += rec["ipopt"]["best_start_solve_time_s"]
            if "delta_vs_scipy" in rec:
                all_d_ay_g.append(abs(rec["delta_vs_scipy"]["d_ay_g"]))
                all_d_ay_pct.append(abs(rec["delta_vs_scipy"]["d_ay_pct"]))

            d_sp = rec.get("delta_vs_scipy", {})
            print(
                f"  R={r_m:6.3f}m | vx={rec['v_x_mps']:6.3f} m/s ({rec['a_y_g']:.4f}g, d_ay={d_sp.get('d_ay_g', 0.0):+.5f}g) | "
                f"delta={rec['delta_deg']:5.2f}deg beta={rec['beta_deg']:+5.2f}deg | "
                f"Mz_TV={rec['M_z_TV_nm']:+6.1f}Nm Fx_f*sin(d)={rec['F_xf_sin_delta_n']:+6.1f}N | "
                f"P_dc={rec['P_dc_kw']:5.2f}kW | iters={rec['ipopt']['iter_count']:2d} "
                f"t={rec['ipopt']['multistart_solve_time_s']*1000:5.1f}ms (1-start={rec['ipopt']['best_start_solve_time_s']*1000:4.1f}ms) | "
                f"pr={rec['ipopt']['inf_pr']:.1e} du={rec['ipopt']['inf_du']:.1e}",
                flush=True,
            )
            prof_list.append(rec)

        report["profiles"][prof] = prof_list

    report["hv_violations_total"] = hv_viols
    report["summary"] = {
        "total_multistart_solve_time_s": total_solve_s,
        "total_best_start_solve_time_s": total_best_start_s,
        "mean_multistart_solve_time_s": total_solve_s / (len(PROFILES) * len(RADII_M)),
        "max_abs_d_ay_g_vs_scipy": max(all_d_ay_g) if all_d_ay_g else None,
        "max_abs_d_ay_pct_vs_scipy": max(all_d_ay_pct) if all_d_ay_pct else None,
    }

    out_path = RESULTS_DIR / "stationary_oracle_casadi_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\nSaved -> {out_path} | HV violations: {hv_viols} | "
        f"Total solve: {total_solve_s:.3f}s (1-start sum: {total_best_start_s:.3f}s) | "
        f"Max |d_ay| vs SciPy: {report['summary']['max_abs_d_ay_g_vs_scipy']:.6f} g "
        f"({report['summary']['max_abs_d_ay_pct_vs_scipy']:.4f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
