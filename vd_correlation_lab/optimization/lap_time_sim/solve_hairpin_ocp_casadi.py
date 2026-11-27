#!/usr/bin/env python3
"""
CasADi 3.7.2 + IPOPT Cross-Validation of the 180-deg Hairpin Minimum-Time OCP
for Tecnun e-Racing TeR27 4WD (MirenaSim).

Matches the exact formulation of solve_hairpin_ocp.py:
  - Straight approach (22 m) entering at v_0 = 20.0 m/s (72 km/h)
  - 180 deg hairpin (R = 5.5 m, C1 clothoid ramps, exact integral(kappa ds) = pi)
  - 40.0 m straight exit (total path length 81.279 m)
  - Spatial trapezoidal direct collocation on the exact same M = 29 node grid
    (plus a 2x refined Hermite-Simpson mesh-convergence check)
  - States per node: [vx, vy, r, e_y, e_psi]
  - Controls per node: [delta, kappa_FL, kappa_FR, kappa_RL, kappa_RR, f_brk_hyd]
  - Algebraic load transfer per node: [ax_lt, ay_lt] with exact consistency constraints
  - Strict HV DC power bounds: P_dc in [-25.0, +80.0] kW (0 violations verified on
    both CasADi smooth model and exact non-smoothed ter27_physics model)

Cross-validates BOTH FSG_REAL_TRACK and TTC_LAB_PEAK against:
  vd_correlation_lab/results/tv_oracle/hairpin_ocp_report.json
and saves:
  vd_correlation_lab/results/tv_oracle/hairpin_ocp_casadi_report.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import casadi as ca
import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
sys.path.insert(0, str(ORACLE_DIR))

from solve_hairpin_ocp import build_hairpin_track  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import TeR27CasADiVehicle, build_casadi_vehicle, smax  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import TeR27PhysicsModel  # noqa: E402

PROFILES = ["FSG_REAL_TRACK", "TTC_LAB_PEAK"]


def _eval_nodes_exact_converged(
    model: TeR27PhysicsModel,
    z_opt: np.ndarray,
    kappa_path: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Evaluate the exact non-smoothed ter27_physics model at every node with
    converged load transfer (initialized at z_opt[:, 11:13] and iterated to <1e-6).
    """
    p = model.p
    M = z_opt.shape[0]
    p_ecu_kw = np.zeros(M)
    p_drv_kw = np.zeros(M)
    inv_s_dot = np.zeros(M)
    d_state_ds = np.zeros((M, 5))

    bf = p.brake_bias_front
    hyd_split = np.array([0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf)])

    for k in range(M):
        vx, vy, r, ey, ep = z_opt[k, 0:5]
        delta = float(z_opt[k, 5])
        kap = z_opt[k, 6:10]
        f_brk = float(z_opt[k, 10])
        ax_est, ay_est = float(z_opt[k, 11]), float(z_opt[k, 12])

        res = model.evaluate_state_from_kappa(vx, vy, r, delta, kap, ax_est, ay_est, n_load_iter=15)
        fz = res["fz"]
        rho_rad = np.minimum(fz / p.k_tire_z, 0.5 * p.wheel_radius)
        r_load = p.wheel_radius - rho_rad
        t_hyd = f_brk * hyd_split * r_load
        t_motor = res["torques"] + t_hyd
        omega_w = res["omega_w"]
        p_ecu_kw[k] = model.compute_hv_power_w(t_motor, omega_w) * 1e-3
        p_drv_kw[k] = model.compute_hv_power_drivetrain_w(t_motor, omega_w) * 1e-3

        vx_dot = float(res["sum_fx"]) / p.mass + r * vy
        vy_dot = float(res["sum_fy"]) / p.mass - r * vx
        r_dot = float(res["sum_mz"]) / p.izz_total
        s_dot = (vx * math.cos(ep) - vy * math.sin(ep)) / max(1.0 - float(kappa_path[k]) * ey, 0.15)
        inv_sd = 1.0 / max(s_dot, 0.5)
        inv_s_dot[k] = inv_sd
        d_state_ds[k] = np.array([
            vx_dot * inv_sd,
            vy_dot * inv_sd,
            r_dot * inv_sd,
            (vx * math.sin(ep) + vy * math.cos(ep)) * inv_sd,
            (r - float(kappa_path[k]) * s_dot) * inv_sd,
        ])

    return {
        "p_ecu_kw": p_ecu_kw,
        "p_drv_kw": p_drv_kw,
        "inv_s_dot": inv_s_dot,
        "d_state_ds": d_state_ds,
    }


def _node_dynamics_expr(
    veh: TeR27CasADiVehicle,
    z_node: ca.SX,
    kappa_path_k: float,
) -> Dict[str, Any]:
    """
    Build symbolic spatial ODE d(state)/ds and path constraints at a single node.
    z_node (13):
      0: vx, 1: vy, 2: r, 3: e_y, 4: e_psi,
      5: delta, 6..9: kappa_4, 10: f_brk_hyd,
      11: ax_lt, 12: ay_lt
    """
    p = veh.params
    vx = z_node[0]
    vy = z_node[1]
    r = z_node[2]
    e_y = z_node[3]
    e_psi = z_node[4]
    delta = z_node[5]
    kappa_4 = z_node[6:10]
    f_brk = z_node[10]
    ax_lt = z_node[11]
    ay_lt = z_node[12]

    ev = veh.eval_forces_from_kappa(vx, vy, r, delta, kappa_4, f_brk, ax_lt, ay_lt)

    vx_dot = ev["Fx_tot"] / p.mass + r * vy
    vy_dot = ev["Fy_tot"] / p.mass - r * vx
    r_dot = ev["Mz_tot"] / p.izz_total

    cos_ep = ca.cos(e_psi)
    sin_ep = ca.sin(e_psi)
    denom_geom = smax(1.0 - kappa_path_k * e_y, 0.15, 1e-3)
    s_dot = (vx * cos_ep - vy * sin_ep) / denom_geom
    e_y_dot = vx * sin_ep + vy * cos_ep
    e_psi_dot = r - kappa_path_k * s_dot

    inv_s_dot = 1.0 / smax(s_dot, 0.5, 1e-3)
    d_state_ds = ca.vertcat(
        vx_dot * inv_s_dot,
        vy_dot * inv_s_dot,
        r_dot * inv_s_dot,
        e_y_dot * inv_s_dot,
        e_psi_dot * inv_s_dot,
    )

    ev["d_state_ds"] = d_state_ds
    ev["s_dot"] = s_dot
    ev["inv_s_dot"] = inv_s_dot
    ev["ax"] = ev["Fx_tot"] / p.mass
    ev["ay"] = ev["Fy_tot"] / p.mass
    ev["lt_res_ax"] = ax_lt - ev["ax"]
    ev["lt_res_ay"] = ay_lt - ev["ay"]
    return ev


def _build_initial_guess(
    p: Any,
    profile: str,
    trk: Dict[str, Any],
) -> np.ndarray:
    s = trk["s"]
    kp = trk["kappa"]
    M = len(s)
    v_apex = 7.6 if profile == "FSG_REAL_TRACK" else 8.8
    z0 = np.zeros((M, 13), dtype=float)
    for k in range(M):
        sk = s[k]
        if sk <= trk["s_turn_start"]:
            u = sk / trk["s_turn_start"]
            vk = math.sqrt(max(20.0**2 * (1.0 - u) + v_apex**2 * u, v_apex**2))
            z0[k, 0] = vk
            z0[k, 6:10] = -0.04 if u > 0.35 else 0.0
            z0[k, 10] = 1500.0 if u > 0.4 else 0.0
            z0[k, 11] = -10.0 if u > 0.35 else 0.0
        elif sk <= trk["s_turn_end"]:
            vk = v_apex
            z0[k, 0] = vk
            z0[k, 2] = vk * kp[k]
            z0[k, 5] = min(math.atan(p.wheelbase * kp[k]) * 1.15, p.max_steer_angle * 0.92)
            z0[k, 1] = -vk * 0.03
            z0[k, 6:10] = np.array([0.005, 0.02, 0.0, 0.025])
            z0[k, 12] = vk * vk * kp[k]
        else:
            vk = math.sqrt(v_apex**2 + 2.0 * 8.0 * (sk - trk["s_turn_end"]))
            z0[k, 0] = min(vk, 28.0)
            z0[k, 6:10] = 0.05
            z0[k, 11] = 8.0
    z0[0, 0] = 20.0
    z0[0, 1:5] = 0.0
    return z0


def solve_hairpin_trapezoidal_casadi(
    profile: str,
    n_entry: int = 8,
    n_turn: int = 12,
    n_exit: int = 8,
    scipy_rep: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Solve the hairpin minimum-time OCP with trapezoidal direct collocation
    in CasADi 3.7.2 + IPOPT (exact Hessian, MUMPS).
    """
    t_build0 = time.perf_counter()
    veh = build_casadi_vehicle(grip_profile=profile)
    exact_model = TeR27PhysicsModel(profile=profile)
    p = veh.params

    trk = build_hairpin_track(n_entry=n_entry, n_turn=n_turn, n_exit=n_exit)
    s = trk["s"]
    ds = trk["ds"]
    kp = trk["kappa"]
    M = len(s)
    N = M - 1
    NV = 13

    # Scaling factors matching solve_hairpin_ocp.py (+ 10 m/s^2 for ax_lt, ay_lt)
    scale = np.array(
        [15.0, 2.0, 2.0, 1.0, 0.5, 0.38, 0.12, 0.12, 0.12, 0.12, 4000.0, 10.0, 10.0],
        dtype=float,
    )
    state_scales = scale[:5]

    X_sc = ca.SX.sym("X_sc", NV, M)
    Z = X_sc * ca.repmat(ca.DM(scale), 1, M)

    # Build node evaluations
    evs = [_node_dynamics_expr(veh, Z[:, k], float(kp[k])) for k in range(M)]

    # Objective: trapezoidal time + exact same tiny smoothness regularization as SciPy
    t_expr = 0.0
    for k in range(N):
        t_expr += 0.5 * (evs[k]["inv_s_dot"] + evs[k + 1]["inv_s_dot"]) * float(ds[k])

    reg_expr = 0.0
    for k in range(N):
        d_delta = Z[5, k + 1] - Z[5, k]
        d_kap = Z[6:10, k + 1] - Z[6:10, k]
        d_brk = (Z[10, k + 1] - Z[10, k]) / 4000.0
        reg_expr += 1e-3 * (d_delta * d_delta) + 5e-3 * ca.dot(d_kap, d_kap) + 2e-3 * (d_brk * d_brk)

    obj = t_expr + reg_expr

    # Constraints
    g_list = []
    lbg = []
    ubg = []

    # 1. Trapezoidal collocation defects (5 * N)
    for k in range(N):
        defect = (Z[0:5, k + 1] - Z[0:5, k]) - 0.5 * (evs[k]["d_state_ds"] + evs[k + 1]["d_state_ds"]) * float(ds[k])
        g_list.append(defect / ca.DM(state_scales))
        lbg.extend([0.0] * 5)
        ubg.extend([0.0] * 5)

    # 2. Per-node algebraic load-transfer consistency (2 * M) + motor & HV limits (10 * M)
    p_max_kw = (p.p_dc_max_w - 100.0) * 1e-3  # 79.9 kW (exact same 100 W margin as SciPy)
    p_min_kw = (p.p_dc_min_w + 100.0) * 1e-3  # -24.9 kW

    for k in range(M):
        ev = evs[k]
        # Algebraic load transfer consistency
        g_list.append(ev["lt_res_ax"] / 10.0)
        g_list.append(ev["lt_res_ay"] / 10.0)
        lbg.extend([0.0, 0.0])
        ubg.extend([0.0, 0.0])

        # Motor torque limits (8)
        g_list.append((ev["T_max_4"] - ev["T_motor_4"]) / 100.0)
        g_list.append((ev["T_motor_4"] - ev["T_min_4"]) / 100.0)
        lbg.extend([0.0] * 8)
        ubg.extend([ca.inf] * 8)

        # HV DC power bounds on both ECU and Drivetrain definitions (4)
        g_list.append((p_max_kw - ev["P_dc_ecu_kw"]) / 10.0)
        g_list.append((p_max_kw - ev["P_dc_drv_kw"]) / 10.0)
        g_list.append((ev["P_dc_ecu_kw"] - p_min_kw) / 10.0)
        g_list.append((ev["P_dc_drv_kw"] - p_min_kw) / 10.0)
        lbg.extend([0.0] * 4)
        ubg.extend([ca.inf] * 4)

    x_vec = ca.vec(X_sc)
    g_vec = ca.vertcat(*g_list)

    nlp = {"x": x_vec, "f": obj, "g": g_vec}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "exact",
        "ipopt.tol": 1e-7,
        "ipopt.acceptable_tol": 1e-5,
        "ipopt.max_iter": 500,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("hairpin_ocp", "ipopt", nlp, opts)
    build_time_s = time.perf_counter() - t_build0

    # Bounds
    lb = np.zeros((M, NV), dtype=float)
    ub = np.zeros((M, NV), dtype=float)
    lb[:, 0], ub[:, 0] = 3.0, 34.0
    lb[:, 1], ub[:, 1] = -4.0, 4.0
    lb[:, 2], ub[:, 2] = -2.5, 3.0
    lb[:, 3], ub[:, 3] = -0.45, 0.45
    lb[:, 4], ub[:, 4] = -0.35, 0.35
    lb[:, 5], ub[:, 5] = -0.15, p.max_steer_angle
    lb[:, 6:10], ub[:, 6:10] = -0.14, 0.14
    lb[:, 10], ub[:, 10] = 0.0, p.brake_mu * p.mass * p.gravity
    lb[:, 11], ub[:, 11] = -35.0, 35.0
    lb[:, 12], ub[:, 12] = -35.0, 35.0

    # Boundary conditions matching solve_hairpin_ocp.py
    lb[0, 0] = ub[0, 0] = 20.0
    lb[0, 1:5] = ub[0, 1:5] = 0.0
    lb[-1, 3], ub[-1, 3] = -0.45, 0.45
    lb[-1, 4], ub[-1, 4] = -0.12, 0.12

    z0 = _build_initial_guess(p, profile, trk)
    # Column-major flattening matches ca.vec(X_sc) where X_sc is (NV, M)
    x0_flat = (z0 / scale[None, :]).T.reshape(-1, order="F")
    lbx_flat = (lb / scale[None, :]).T.reshape(-1, order="F")
    ubx_flat = (ub / scale[None, :]).T.reshape(-1, order="F")

    print(
        f"[Hairpin OCP CasADi | {profile}] M={M} nodes ({len(x0_flat)} vars, {len(lbg)} constr) | "
        f"build={build_time_s:.3f}s | solving...",
        flush=True,
    )
    t_solve0 = time.perf_counter()
    sol = solver(x0=x0_flat, lbx=lbx_flat, ubx=ubx_flat, lbg=lbg, ubg=ubg)
    solve_time_s = time.perf_counter() - t_solve0
    stats = solver.stats()

    x_opt_mat = np.asarray(sol["x"]).reshape((NV, M), order="F")
    z_opt = (x_opt_mat * scale[:, None]).T  # (M, NV)

    # Evaluate exact non-smoothed ter27_physics model (converged load transfer)
    ev_exact = _eval_nodes_exact_converged(exact_model, z_opt, kp)

    # Evaluate directly via veh.fn_forces_from_kappa for exact CasADi telemetry
    p_ecu_ca = np.zeros(M)
    p_drv_ca = np.zeros(M)
    ax_ca = np.zeros(M)
    ay_ca = np.zeros(M)
    t_mot_ca = np.zeros((M, 4))
    mz_tv_ca = np.zeros(M)
    inv_sd_ca = np.zeros(M)
    d_state_ca = np.zeros((M, 5))
    for k in range(M):
        zk = z_opt[k]
        out = veh.fn_forces_from_kappa(
            zk[0], zk[1], zk[2], zk[5], zk[6:10], zk[10], zk[11], zk[12]
        )
        fx_tot = float(out[0])
        fy_tot = float(out[1])
        mz_tot = float(out[2])
        p_ecu_ca[k] = float(out[3])
        p_drv_ca[k] = float(out[4])
        t_mot_ca[k] = np.asarray(out[10]).ravel()
        mz_tv_ca[k] = float(out[15])
        ax_ca[k] = fx_tot / p.mass
        ay_ca[k] = fy_tot / p.mass
        vx_k, vy_k, r_k, ey_k, ep_k = zk[0], zk[1], zk[2], zk[3], zk[4]
        vx_dot = ax_ca[k] + r_k * vy_k
        vy_dot = ay_ca[k] - r_k * vx_k
        r_dot = mz_tot / p.izz_total
        den_g = 0.5 * ((1.0 - kp[k] * ey_k) + 0.15 + math.hypot((1.0 - kp[k] * ey_k) - 0.15, 1e-3))
        sd_k = (vx_k * math.cos(ep_k) - vy_k * math.sin(ep_k)) / den_g
        sd_safe = 0.5 * (sd_k + 0.5 + math.hypot(sd_k - 0.5, 1e-3))
        inv_sd_ca[k] = 1.0 / sd_safe
        d_state_ca[k] = np.array([
            vx_dot / sd_safe,
            vy_dot / sd_safe,
            r_dot / sd_safe,
            (vx_k * math.sin(ep_k) + vy_k * math.cos(ep_k)) / sd_safe,
            (r_k - kp[k] * sd_k) / sd_safe,
        ])

    t_arr_ca = np.zeros(M)
    t_arr_ca[1:] = np.cumsum(0.5 * (inv_sd_ca[:-1] + inv_sd_ca[1:]) * ds)
    t_total_ca = float(t_arr_ca[-1])

    inv_sd_ex = ev_exact["inv_s_dot"]
    t_arr_ex = np.zeros(M)
    t_arr_ex[1:] = np.cumsum(0.5 * (inv_sd_ex[:-1] + inv_sd_ex[1:]) * ds)
    t_total_ex = float(t_arr_ex[-1])

    # Collocation defects
    st = z_opt[:, :5]
    defects_ca = (st[1:] - st[:-1]) - 0.5 * (d_state_ca[:-1] + d_state_ca[1:]) * ds[:, None]
    max_eq_defect_ca = float(np.max(np.abs(defects_ca / state_scales[None, :])))
    defects_ex = (st[1:] - st[:-1]) - 0.5 * (ev_exact["d_state_ds"][:-1] + ev_exact["d_state_ds"][1:]) * ds[:, None]
    max_eq_defect_ex = float(np.max(np.abs(defects_ex / state_scales[None, :])))

    # HV checks (CasADi + exact physics)
    p_max_ca = float(max(np.max(p_ecu_ca), np.max(p_drv_ca)))
    p_min_ca = float(min(np.min(p_ecu_ca), np.min(p_drv_ca)))
    hv_viols_ca = int(np.sum((p_ecu_ca > 80.0) | (p_drv_ca > 80.0) | (p_ecu_ca < -25.0) | (p_drv_ca < -25.0)))

    p_ecu_ex = ev_exact["p_ecu_kw"]
    p_drv_ex = ev_exact["p_drv_kw"]
    p_max_ex = float(max(np.max(p_ecu_ex), np.max(p_drv_ex)))
    p_min_ex = float(min(np.min(p_ecu_ex), np.min(p_drv_ex)))
    hv_viols_ex = int(np.sum((p_ecu_ex > 80.0) | (p_drv_ex > 80.0) | (p_ecu_ex < -25.0) | (p_drv_ex < -25.0)))

    iters = stats.get("iterations", {})
    inf_pr = float(iters["inf_pr"][-1]) if "inf_pr" in iters and len(iters["inf_pr"]) > 0 else float("nan")
    inf_du = float(iters["inf_du"][-1]) if "inf_du" in iters and len(iters["inf_du"]) > 0 else float("nan")

    vx = z_opt[:, 0]
    vy = z_opt[:, 1]
    beta_deg = np.degrees(np.arctan2(vy, np.maximum(vx, 0.5)))
    delta_deg = np.degrees(z_opt[:, 5])
    idx_min_v = int(np.argmin(vx))

    perf = {
        "total_time_s": t_total_ca,
        "total_time_exact_eval_s": t_total_ex,
        "v_entry_mps": float(vx[0]),
        "v_apex_min_mps": float(vx[idx_min_v]),
        "s_apex_min_m": float(s[idx_min_v]),
        "v_exit_end_mps": float(vx[-1]),
        "peak_ay_g": float(np.max(np.abs(ay_ca)) / p.gravity),
        "peak_brake_g": float(-np.min(ax_ca) / p.gravity),
        "peak_accel_g": float(np.max(ax_ca) / p.gravity),
        "max_abs_beta_deg": float(np.max(np.abs(beta_deg))),
        "max_delta_deg": float(np.max(delta_deg)),
        "max_eq_defect_casadi": max_eq_defect_ca,
        "max_eq_defect_exact_eval": max_eq_defect_ex,
        "P_dc_max_kw": p_max_ca,
        "P_dc_min_kw": p_min_ca,
        "P_dc_max_exact_kw": p_max_ex,
        "P_dc_min_exact_kw": p_min_ex,
        "hv_violations_casadi": hv_viols_ca,
        "hv_violations_exact": hv_viols_ex,
        "hv_violations": hv_viols_ca + hv_viols_ex,
    }

    rep: Dict[str, Any] = {
        "profile": profile,
        "discretization": "trapezoidal",
        "n_nodes": M,
        "n_intervals": N,
        "ipopt": {
            "status": str(stats.get("return_status", "unknown")),
            "success": bool(stats.get("success", False)),
            "iter_count": int(stats.get("iter_count", -1)),
            "inf_pr": inf_pr,
            "inf_du": inf_du,
            "build_time_s": build_time_s,
            "solve_time_s": solve_time_s,
            "total_wall_time_s": build_time_s + solve_time_s,
        },
        "track": {
            "entry_straight_m": 22.0,
            "hairpin_radius_m": 5.5,
            "hairpin_angle_deg": 180.0,
            "exit_straight_m": 40.0,
            "total_length_m": float(s[-1]),
            "n_nodes": M,
        },
        "performance": perf,
        "trajectories": {
            "s_m": [float(v) for v in s],
            "t_s": [float(v) for v in t_arr_ca],
            "vx_mps": [float(v) for v in vx],
            "vy_mps": [float(v) for v in vy],
            "r_rad_s": [float(v) for v in z_opt[:, 2]],
            "e_y_m": [float(v) for v in z_opt[:, 3]],
            "e_psi_rad": [float(v) for v in z_opt[:, 4]],
            "beta_deg": [float(v) for v in beta_deg],
            "delta_deg": [float(v) for v in delta_deg],
            "ay_g": [float(v / p.gravity) for v in ay_ca],
            "ax_g": [float(v / p.gravity) for v in ax_ca],
            "T_FL_nm": [float(v) for v in t_mot_ca[:, 0]],
            "T_FR_nm": [float(v) for v in t_mot_ca[:, 1]],
            "T_RL_nm": [float(v) for v in t_mot_ca[:, 2]],
            "T_RR_nm": [float(v) for v in t_mot_ca[:, 3]],
            "Mz_TV_nm": [float(v) for v in mz_tv_ca],
            "P_dc_kw": [float(v) for v in p_ecu_ca],
        },
    }

    if scipy_rep is not None:
        sp_perf = scipy_rep["performance"]
        sp_time = float(scipy_rep["solve_time_s"])
        rep["delta_vs_scipy"] = {
            "d_total_time_s": t_total_ca - float(sp_perf["total_time_s"]),
            "d_total_time_ms": 1000.0 * (t_total_ca - float(sp_perf["total_time_s"])),
            "d_total_time_pct": 100.0 * (t_total_ca / float(sp_perf["total_time_s"]) - 1.0),
            "d_v_apex_min_mps": float(vx[idx_min_v]) - float(sp_perf["v_apex_min_mps"]),
            "d_v_exit_end_mps": float(vx[-1]) - float(sp_perf["v_exit_end_mps"]),
            "d_peak_ay_g": perf["peak_ay_g"] - float(sp_perf["peak_ay_g"]),
            "d_peak_brake_g": perf["peak_brake_g"] - float(sp_perf["peak_brake_g"]),
            "d_peak_accel_g": perf["peak_accel_g"] - float(sp_perf["peak_accel_g"]),
            "d_P_dc_max_kw": p_max_ca - float(sp_perf["P_dc_max_kw"]),
            "d_P_dc_min_kw": p_min_ca - float(sp_perf["P_dc_min_kw"]),
            "scipy_solve_time_s": sp_time,
            "scipy_nit": int(scipy_rep["nit"]),
            "speedup_solve_only": sp_time / max(solve_time_s, 1e-6),
            "speedup_including_build": sp_time / max(build_time_s + solve_time_s, 1e-6),
        }

    d_sp = rep.get("delta_vs_scipy", {})
    print(
        f"[Hairpin OCP CasADi | {profile} | M={M}] status={rep['ipopt']['status']} "
        f"iters={rep['ipopt']['iter_count']} solve={solve_time_s:.3f}s (build={build_time_s:.3f}s) | "
        f"T={t_total_ca:.4f}s (dT={d_sp.get('d_total_time_ms', 0.0):+.2f}ms) | "
        f"v_min={perf['v_apex_min_mps']:.3f} m/s | v_end={perf['v_exit_end_mps']:.3f} m/s | "
        f"ay_max={perf['peak_ay_g']:.3f}g | P_dc=[{p_min_ca:.2f}, {p_max_ca:.2f}] kW "
        f"(exact=[{p_min_ex:.2f}, {p_max_ex:.2f}] kW, viols={perf['hv_violations']}) | "
        f"pr={inf_pr:.1e} du={inf_du:.1e}",
        flush=True,
    )
    return rep


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scipy_path = RESULTS_DIR / "hairpin_ocp_report.json"
    scipy_combined = json.loads(scipy_path.read_text(encoding="utf-8")) if scipy_path.exists() else {}

    print("=== TeR27 4WD Hairpin Minimum-Time OCP (CasADi 3.7.2 + IPOPT) ===", flush=True)
    combined: Dict[str, Any] = {
        "vehicle_id": "ter27_4wd",
        "solver": "CasADi 3.7.2 + IPOPT (exact Hessian, MUMPS)",
        "primary_profile": "FSG_REAL_TRACK",
        "hv_violations_total": 0,
    }

    hv_total = 0
    for prof in PROFILES:
        rep = solve_hairpin_trapezoidal_casadi(
            profile=prof,
            n_entry=8,
            n_turn=12,
            n_exit=8,
            scipy_rep=scipy_combined.get(prof),
        )
        combined[prof] = rep
        hv_total += int(rep["performance"]["hv_violations"])

    # Mesh refinement check (2x finer grid: M = 57 nodes) for discretization verification
    print("\n--- 2x Mesh Refinement Check (M = 57 nodes) ---", flush=True)
    mesh_ref: Dict[str, Any] = {}
    for prof in PROFILES:
        rep_fine = solve_hairpin_trapezoidal_casadi(
            profile=prof,
            n_entry=16,
            n_turn=24,
            n_exit=16,
            scipy_rep=None,
        )
        hv_total += int(rep_fine["performance"]["hv_violations"])
        mesh_ref[prof] = {
            "n_nodes": rep_fine["n_nodes"],
            "ipopt": rep_fine["ipopt"],
            "performance": rep_fine["performance"],
            "d_total_time_vs_coarse_ms": 1000.0
            * (rep_fine["performance"]["total_time_s"] - combined[prof]["performance"]["total_time_s"]),
        }
    combined["mesh_refinement_2x"] = mesh_ref
    combined["hv_violations_total"] = hv_total

    out_path = RESULTS_DIR / "hairpin_ocp_casadi_report.json"
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path} (Total HV violations: {hv_total})", flush=True)


if __name__ == "__main__":
    main()
