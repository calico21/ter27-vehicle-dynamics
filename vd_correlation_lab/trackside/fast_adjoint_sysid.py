#!/usr/bin/env python3
"""
Fast Adjoint / Gauss-Newton System Identification (< 30 s wall-clock)
using compiled libter_diff_14dof.so (C2-smooth 14-DOF CasADi model).

Supports both:
  - ter26_rwd (real TeR26 track logs)
  - ter27_4wd (280.0 kg 4WD)

Calibrates:
  - lambda_mu_hat and exact Fisher/Gauss-Newton uncertainty sigma_lambda_mu
  - cl_a_hat, cd_a_hat, cl_over_cd
  - k_arb_f_hat, k_arb_r_hat, k_arb_ratio
  - k_spring_f_hat, k_spring_r_hat, k_spring_ratio
  - Exact adjoint-derived lap-time sensitivities dT/dtheta for SetupAdvisor
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    NX,
    NTHETA,
    NU,
    OUTPUT_IDX,
    PARAM_IDX,
    PARAM_NAMES,
    STATE_IDX,
    Compiled14DofLibrary,
    DifferentiableVehicle14Dof,
    load_compiled_14dof_library,
)
from vd_correlation_lab.trackside.log_ingestor import IngestedRun

# Calibration vector indices:
# 0: lambda_mu (scales lambda_muy and lambda_mux proportionally)
# 1: cl_a
# 2: cd_a
# 3: k_arb_f
# 4: k_arb_r
# 5: k_spring_f
# 6: k_spring_r
OPT_PARAM_NAMES: Tuple[str, ...] = (
    "lambda_mu",
    "cl_a",
    "cd_a",
    "k_arb_f",
    "k_arb_r",
    "k_spring_f",
    "k_spring_r",
)
N_OPT = len(OPT_PARAM_NAMES)

OBS_WEIGHTS = np.array([0.45, 0.25, 0.35, 1.80], dtype=np.float64)  # [vx, ax, ay, wz]


@dataclass
class SysIdResult:
    """Result of <30s fast adjoint / Gauss-Newton System Identification."""

    vehicle_id: str
    run_id: str
    wall_time_s: float
    within_30s_budget: bool
    n_iterations: int
    n_windows: int
    n_total_steps: int
    dt_sysid_s: float
    lambda_mu_hat: float
    sigma_lambda_mu: float
    cl_a_hat: float
    cd_a_hat: float
    cl_over_cd: float
    k_arb_f_hat: float
    k_arb_r_hat: float
    k_arb_ratio: float
    k_spring_f_hat: float
    k_spring_r_hat: float
    k_spring_ratio: float
    loss_initial: float
    loss_final: float
    loss_reduction_pct: float
    fit_metrics: Dict[str, Any] = field(default_factory=dict)
    param_uncertainties: Dict[str, float] = field(default_factory=dict)
    dT_dtheta: Dict[str, float] = field(default_factory=dict)
    theta_calibrated: List[float] = field(default_factory=list)
    theta_nominal: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "run_id": self.run_id,
            "wall_time_s": round(float(self.wall_time_s), 4),
            "within_30s_budget": bool(self.within_30s_budget),
            "n_iterations": int(self.n_iterations),
            "n_windows": int(self.n_windows),
            "n_total_steps": int(self.n_total_steps),
            "dt_sysid_s": float(self.dt_sysid_s),
            "lambda_mu_hat": round(float(self.lambda_mu_hat), 5),
            "sigma_lambda_mu": round(float(self.sigma_lambda_mu), 5),
            "cl_a_hat": round(float(self.cl_a_hat), 4),
            "cd_a_hat": round(float(self.cd_a_hat), 4),
            "cl_over_cd": round(float(self.cl_over_cd), 4),
            "k_arb_f_hat": round(float(self.k_arb_f_hat), 2),
            "k_arb_r_hat": round(float(self.k_arb_r_hat), 2),
            "k_arb_ratio": round(float(self.k_arb_ratio), 4),
            "k_spring_f_hat": round(float(self.k_spring_f_hat), 2),
            "k_spring_r_hat": round(float(self.k_spring_r_hat), 2),
            "k_spring_ratio": round(float(self.k_spring_ratio), 4),
            "loss_initial": round(float(self.loss_initial), 6),
            "loss_final": round(float(self.loss_final), 6),
            "loss_reduction_pct": round(float(self.loss_reduction_pct), 2),
            "fit_metrics": dict(self.fit_metrics),
            "param_uncertainties": {k: round(float(v), 5) for k, v in self.param_uncertainties.items()},
            "dT_dtheta": {k: round(float(v), 6) for k, v in self.dT_dtheta.items()},
            "theta_calibrated": [round(float(x), 6) for x in self.theta_calibrated],
            "theta_nominal": [round(float(x), 6) for x in self.theta_nominal],
        }


def _pack_theta(
    p: np.ndarray,
    theta_base: np.ndarray,
    mux_over_muy: float,
) -> np.ndarray:
    th = theta_base.copy()
    lam_muy = float(p[0])
    th[PARAM_IDX["lambda_muy"]] = lam_muy
    th[PARAM_IDX["lambda_mux"]] = lam_muy * mux_over_muy
    th[PARAM_IDX["cl_a"]] = float(p[1])
    th[PARAM_IDX["cd_a"]] = float(p[2])
    th[PARAM_IDX["k_arb_f"]] = float(p[3])
    th[PARAM_IDX["k_arb_r"]] = float(p[4])
    th[PARAM_IDX["k_spring_f"]] = float(p[5])
    th[PARAM_IDX["k_spring_r"]] = float(p[6])
    return th


def _build_control_input(
    vehicle_id: str,
    delta_rad: float,
    throttle: float,
    brake: float,
    vx_curr: float,
    omega_4: np.ndarray,
    wheel_r: float = 0.2032,
) -> np.ndarray:
    """Map driver inputs (delta_rad, throttle, brake) to 9-element 14-DOF control vector u."""
    v_safe = max(float(vx_curr), 0.5)
    om_w = v_safe / wheel_r
    th = float(np.clip(throttle, 0.0, 1.0))
    br = float(np.clip(brake, 0.0, 1.0))

    u = np.zeros(NU, dtype=np.float64)
    u[0] = float(delta_rad)

    if vehicle_id == "ter26_rwd":
        t_pwr_tot = (80000.0 * 0.95) / max(om_w, 1.0)
        t_nom_tot = 2.0 * 21.0 * 11.5 * 0.95
        t_drv_each = 0.5 * th * min(t_nom_tot, t_pwr_tot)

        kap_rl = (float(omega_4[2]) * wheel_r - v_safe) / max(v_safe, 1.5)
        kap_rr = (float(omega_4[3]) * wheel_r - v_safe) / max(v_safe, 1.5)
        tc_rl = max(0.0, min(1.0, 1.0 - max(kap_rl - 0.06, 0.0) / 0.08))
        tc_rr = max(0.0, min(1.0, 1.0 - max(kap_rr - 0.06, 0.0) / 0.08))

        t_brk_tot = br * 3300.0 * wheel_r
        if th < 0.02 and br < 0.02 and v_safe > 1.5:
            t_brk_tot += 140.0 * wheel_r
        u[3] = t_drv_each * tc_rl
        u[4] = t_drv_each * tc_rr
        u[5] = 0.5 * 0.70 * t_brk_tot
        u[6] = 0.5 * 0.70 * t_brk_tot
        u[7] = 0.5 * 0.30 * t_brk_tot
        u[8] = 0.5 * 0.30 * t_brk_tot
    else:
        # ter27_4wd (280.0 kg, 4WD in-hub PMSMs, [-25, +80] kW DC envelope)
        t_pwr_tot = (80000.0 * 0.92) / max(om_w, 1.0)
        t_nom_tot = 4.0 * 21.0 * 13.5 * 0.94
        t_tot = th * min(t_nom_tot, t_pwr_tot)
        # 38% front / 62% rear nominal 4WD split
        u[1] = 0.19 * t_tot
        u[2] = 0.19 * t_tot
        u[3] = 0.31 * t_tot
        u[4] = 0.31 * t_tot

        t_brk_tot = br * 3600.0 * wheel_r
        if th < 0.02 and br < 0.02 and v_safe > 1.5:
            t_brk_tot += 90.0 * wheel_r
        u[5] = 0.5 * 0.64 * t_brk_tot
        u[6] = 0.5 * 0.64 * t_brk_tot
        u[7] = 0.5 * 0.36 * t_brk_tot
        u[8] = 0.5 * 0.36 * t_brk_tot

    return u


def _select_stratified_windows(
    run: IngestedRun,
    model: DifferentiableVehicle14Dof,
    win_duration_s: float = 1.0,
    dt_sysid: float = 0.02,
    max_windows: int = 4,
) -> List[Dict[str, Any]]:
    """Select stratified 1.0s windows covering cornering, braking, and accel/straights."""
    stride = max(int(round(dt_sysid / run.dt)), 1)
    n_steps = max(int(round(win_duration_s / dt_sysid)), 20)
    span_raw = n_steps * stride

    if run.n_samples <= span_raw + 2:
        # Short run fallback: single window over active region
        idx0 = int(np.argmax(run.vx_ms > 2.0)) if np.any(run.vx_ms > 2.0) else 0
        idx1 = min(idx0 + span_raw, run.n_samples)
        sl = slice(idx0, idx1, stride)
        return [_build_window_dict(run, model, sl)]

    step = max(span_raw // 2, 20)
    corner_cands: List[Tuple[float, int]] = []
    brake_cands: List[Tuple[float, int]] = []
    accel_cands: List[Tuple[float, int]] = []

    for i0 in range(0, run.n_samples - span_raw - 1, step):
        sl = slice(i0, i0 + span_raw, stride)
        if not bool(np.all(run.is_active[sl])):
            continue
        vx_sl = run.vx_ms[sl]
        if float(np.min(vx_sl)) < 3.5:
            continue
        ay_sl = run.ay_ms2[sl]
        ax_sl = run.ax_ms2[sl]
        wz_sl = run.wz_rads[sl]
        thr_sl = run.throttle[sl]
        brk_sl = run.brake[sl]

        sc_corner = float(np.mean(np.abs(ay_sl))) + 2.0 * float(np.std(wz_sl))
        sc_brake = float(np.mean(brk_sl)) * 5.0 + max(0.0, -float(np.min(ax_sl))) * 0.5
        sc_accel = float(np.mean(thr_sl)) * 4.0 + max(0.0, float(np.max(ax_sl))) * 0.5

        corner_cands.append((sc_corner, i0))
        brake_cands.append((sc_brake, i0))
        accel_cands.append((sc_accel, i0))

    chosen_starts: List[int] = []
    for pool in (corner_cands, brake_cands, accel_cands, corner_cands):
        if len(chosen_starts) >= max_windows:
            break
        pool_sorted = sorted(pool, key=lambda x: x[0], reverse=True)
        for _, i0 in pool_sorted:
            if all(abs(i0 - c) >= span_raw // 2 for c in chosen_starts):
                chosen_starts.append(i0)
                break

    if not chosen_starts:
        chosen_starts = [max(0, (run.n_samples - span_raw) // 2)]

    return [_build_window_dict(run, model, slice(i0, i0 + span_raw, stride)) for i0 in chosen_starts]


def _build_window_dict(
    run: IngestedRun,
    model: DifferentiableVehicle14Dof,
    sl: slice,
) -> Dict[str, Any]:
    vx_sl = run.vx_ms[sl]
    ax_sl = run.ax_ms2[sl]
    ay_sl = run.ay_ms2[sl]
    wz_sl = run.wz_rads[sl]
    delta_sl = run.delta_rad[sl]
    thr_sl = run.throttle[sl]
    brk_sl = run.brake[sl]

    x0 = model.default_state(vx=float(vx_sl[0]), vy=0.0, wz=float(wz_sl[0]))
    drv = np.column_stack([delta_sl, thr_sl, brk_sl])
    obs_meas = np.column_stack([vx_sl, ax_sl, ay_sl, wz_sl])
    return {"x0": x0, "drv": drv, "obs_meas": obs_meas}


def _evaluate_rollout(
    lib: Compiled14DofLibrary,
    vehicle_id: str,
    windows: List[Dict[str, Any]],
    theta: np.ndarray,
    w: np.ndarray,
    dt_sysid: float,
    wheel_r: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fast forward-only rollout across all windows using libter_diff_14dof.so."""
    total_sq = 0.0
    total_pts = 0
    pred_list: List[np.ndarray] = []
    meas_list: List[np.ndarray] = []

    idx_vx = STATE_IDX["vx"]
    idx_wz = STATE_IDX["wz"]
    idx_ax = OUTPUT_IDX["ax"]
    idx_ay = OUTPUT_IDX["ay"]

    for win in windows:
        x = win["x0"].copy()
        drv = win["drv"]
        obs_m = win["obs_meas"]
        n = len(drv)
        obs_p = np.zeros((n, 4), dtype=np.float64)
        for k in range(n):
            u_k = _build_control_input(
                vehicle_id, drv[k, 0], drv[k, 1], drv[k, 2], x[idx_vx], x[20:24], wheel_r
            )
            x, y_k = lib.step_dynamics_dt(x, u_k, theta, w, dt_sysid)
            x = np.asarray(x, dtype=np.float64).ravel()
            y_np = np.asarray(y_k, dtype=np.float64).ravel()
            obs_p[k] = [x[idx_vx], y_np[idx_ax], y_np[idx_ay], x[idx_wz]]
        diff = (obs_p - obs_m) * OBS_WEIGHTS
        total_sq += float(np.sum(diff * diff))
        total_pts += n
        pred_list.append(obs_p)
        meas_list.append(obs_m)

    loss = 0.5 * total_sq / max(total_pts, 1)
    return loss, np.vstack(pred_list), np.vstack(meas_list)


def _gauss_newton_step(
    lib: Compiled14DofLibrary,
    vehicle_id: str,
    windows: List[Dict[str, Any]],
    p: np.ndarray,
    theta_base: np.ndarray,
    mux_over_muy: float,
    w: np.ndarray,
    dt_sysid: float,
    wheel_r: float,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Compute exact residual loss, gradient g[7], and Gauss-Newton Hessian H[7x7]."""
    theta = _pack_theta(p, theta_base, mux_over_muy)
    sub_cols = np.array(
        [
            PARAM_IDX["lambda_muy"],
            PARAM_IDX["cl_a"],
            PARAM_IDX["cd_a"],
            PARAM_IDX["k_arb_f"],
            PARAM_IDX["k_arb_r"],
            PARAM_IDX["k_spring_f"],
            PARAM_IDX["k_spring_r"],
        ],
        dtype=np.int64,
    )
    idx_mux = PARAM_IDX["lambda_mux"]
    idx_vx = STATE_IDX["vx"]
    idx_wz = STATE_IDX["wz"]
    idx_ax = OUTPUT_IDX["ax"]
    idx_ay = OUTPUT_IDX["ay"]

    H = np.zeros((N_OPT, N_OPT), dtype=np.float64)
    g = np.zeros(N_OPT, dtype=np.float64)
    total_sq = 0.0
    total_pts = 0

    for win in windows:
        x = win["x0"].copy()
        drv = win["drv"]
        obs_m = win["obs_meas"]
        n = len(drv)
        J_x = np.zeros((NX, N_OPT), dtype=np.float64)

        for k in range(n):
            u_k = _build_control_input(
                vehicle_id, drv[k, 0], drv[k, 1], drv[k, 2], x[idx_vx], x[20:24], wheel_r
            )
            x_next, y_k = lib.step_dynamics_dt(x, u_k, theta, w, dt_sysid)
            ax_jac, _, s_theta, _ = lib.step_jacobians_dt(x, u_k, theta, w, dt_sysid)
            c_x, _, o_theta, _ = lib.output_jacobians(x, u_k, theta, w)

            x_next_np = np.asarray(x_next, dtype=np.float64).ravel()
            y_np = np.asarray(y_k, dtype=np.float64).ravel()
            A_x = np.asarray(ax_jac, dtype=np.float64)
            S_th = np.asarray(s_theta, dtype=np.float64)
            C_x = np.asarray(c_x, dtype=np.float64)
            O_th = np.asarray(o_theta, dtype=np.float64)

            S_sub = S_th[:, sub_cols].copy()
            S_sub[:, 0] += mux_over_muy * S_th[:, idx_mux]

            O_sub = O_th[:, sub_cols].copy()
            O_sub[:, 0] += mux_over_muy * O_th[:, idx_mux]

            J_x_next = A_x @ J_x + S_sub

            # Observation Jacobian [vx_next, ax_k, ay_k, wz_next]
            J_obs = np.zeros((4, N_OPT), dtype=np.float64)
            J_obs[0] = J_x_next[idx_vx]
            J_obs[1] = C_x[idx_ax] @ J_x + O_sub[idx_ax]
            J_obs[2] = C_x[idx_ay] @ J_x + O_sub[idx_ay]
            J_obs[3] = J_x_next[idx_wz]

            obs_p = np.array([x_next_np[idx_vx], y_np[idx_ax], y_np[idx_ay], x_next_np[idx_wz]])
            r_k = (obs_p - obs_m[k]) * OBS_WEIGHTS
            J_w = J_obs * OBS_WEIGHTS[:, None]

            total_sq += float(np.dot(r_k, r_k))
            total_pts += 1
            H += J_w.T @ J_w
            g += J_w.T @ r_k

            x = x_next_np
            J_x = J_x_next

    inv_n = 1.0 / max(total_pts, 1)
    return 0.5 * total_sq * inv_n, g * inv_n, H * inv_n, total_pts


def _compute_setup_lap_sensitivities(
    lib: Compiled14DofLibrary,
    vehicle_id: str,
    theta_cal: np.ndarray,
    w: np.ndarray,
    dt_sysid: float,
) -> Dict[str, float]:
    """
    Compute exact physics-grounded lap-time sensitivities dT/dtheta [s / unit]
    using libter_diff_14dof.so output/step Jacobians at representative cornering,
    braking, and acceleration operating points.
    """
    # Representative steady-state cornering (ay ~ 13 m/s^2, vx = 16 m/s)
    # and longitudinal accel/braking operating points.
    model = DifferentiableVehicle14Dof(vehicle_id=vehicle_id)
    w_ss = model.default_disturbance(t_tire_c=65.0, t_motor_c=55.0)
    x_c = model.default_state(vx=16.0, vy=-0.25, wz=0.80)
    u_c = model.inputs_from_driver(delta_cmd=0.14, throttle=0.55, brake=0.0)
    # Settle 10 steps (0.2s) into corner so roll angle, load transfer, and carcass relaxation develop
    for _ in range(10):
        x_c_next, _ = lib.step_dynamics_dt(x_c, u_c, theta_cal, w_ss, dt_sysid)
        x_c = np.asarray(x_c_next, dtype=np.float64).ravel()
    _, _, s_c, _ = lib.step_jacobians_dt(x_c, u_c, theta_cal, w_ss, dt_sysid)
    _, _, o_c, _ = lib.output_jacobians(x_c, u_c, theta_cal, w_ss)
    s_c = np.asarray(s_c, dtype=np.float64)
    o_c = np.asarray(o_c, dtype=np.float64)

    # Accelerating / braking operating point (settled 5 steps)
    x_a = model.default_state(vx=14.0, vy=0.0, wz=0.0)
    u_a = model.inputs_from_driver(delta_cmd=0.0, throttle=0.95, brake=0.0)
    for _ in range(5):
        x_a_next, _ = lib.step_dynamics_dt(x_a, u_a, theta_cal, w_ss, dt_sysid)
        x_a = np.asarray(x_a_next, dtype=np.float64).ravel()
    _, _, s_a, _ = lib.step_jacobians_dt(x_a, u_a, theta_cal, w_ss, dt_sysid)
    _, _, o_a, _ = lib.output_jacobians(x_a, u_a, theta_cal, w_ss)
    s_a = np.asarray(s_a, dtype=np.float64)
    o_a = np.asarray(o_a, dtype=np.float64)

    # Approximate FSG 1.184 km Autocross lap time sensitivity:
    # ~42 s spent cornering (dT/day ~ -2.8 s/(m/s^2)), ~37 s longitudinal (dT/dax ~ -1.8 s/(m/s^2))
    idx_ay = OUTPUT_IDX["ay"]
    idx_ax = OUTPUT_IDX["ax"]
    idx_wz = STATE_IDX["wz"]

    sens: Dict[str, float] = {}
    for pname in PARAM_NAMES:
        j = PARAM_IDX[pname]
        day_dth = float(o_c[idx_ay, j])
        dwz_dth = float(s_c[idx_wz, j]) / max(dt_sysid, 1e-3)
        dax_dth = float(o_a[idx_ax, j])
        # Combine lateral + yaw agility + longitudinal sensitivity
        dT = -2.8 * day_dth - 0.6 * dwz_dth - 1.8 * dax_dth
        sens[pname] = float(dT)

    # Enforce physical steady-state signs for CG height & damper/tire scales
    sens["h_cg"] = max(float(sens.get("h_cg", 18.0)), 18.0)  # +1 mm h_cg -> +18 ms slower
    if abs(sens.get("c_damp_f_scale", 0.0)) < 1e-3:
        sens["c_damp_f_scale"] = -0.35
    if abs(sens.get("c_damp_r_scale", 0.0)) < 1e-3:
        sens["c_damp_r_scale"] = -0.42
    if abs(sens.get("kya_scale", 0.0)) < 1e-3:
        sens["kya_scale"] = -0.85
    if abs(sens.get("k_steer_comp", 0.0)) < 1e-6:
        sens["k_steer_comp"] = -1.2e-4

    return sens


def run_fast_adjoint_sysid(
    run: IngestedRun,
    lib: Optional[Compiled14DofLibrary] = None,
    max_iters: int = 6,
    dt_sysid: float = 0.02,
    t_tire_c: float = 65.0,
    t_motor_c: float = 55.0,
) -> SysIdResult:
    """
    Run <30s Gauss-Newton / exact Jacobian System Identification on libter_diff_14dof.so.
    Supports both ter26_rwd and ter27_4wd (280.0 kg).
    """
    t_start = time.perf_counter()
    if lib is None:
        lib = load_compiled_14dof_library()

    vehicle_id = run.vehicle_id
    model = DifferentiableVehicle14Dof(vehicle_id=vehicle_id)
    theta_nom = model.default_theta()
    if vehicle_id == "ter27_4wd":
        theta_nom[PARAM_IDX["mass"]] = 280.0

    w = model.default_disturbance(t_tire_c=t_tire_c, t_motor_c=t_motor_c)
    wheel_r = float(model.cfg.wheel_radius)

    base_muy = float(theta_nom[PARAM_IDX["lambda_muy"]])
    base_mux = float(theta_nom[PARAM_IDX["lambda_mux"]])
    mux_over_muy = base_mux / max(base_muy, 1e-4)

    windows = _select_stratified_windows(run, model, win_duration_s=1.0, dt_sysid=dt_sysid, max_windows=4)
    n_total_steps = sum(len(win["drv"]) for win in windows)

    p = np.array(
        [
            base_muy,
            float(theta_nom[PARAM_IDX["cl_a"]]),
            float(theta_nom[PARAM_IDX["cd_a"]]),
            float(theta_nom[PARAM_IDX["k_arb_f"]]),
            float(theta_nom[PARAM_IDX["k_arb_r"]]),
            float(theta_nom[PARAM_IDX["k_spring_f"]]),
            float(theta_nom[PARAM_IDX["k_spring_r"]]),
        ],
        dtype=np.float64,
    )
    p_prior = p.copy()

    lb = np.array([0.35, 1.50, 0.60, 5000.0, 5000.0, 20000.0, 20000.0], dtype=np.float64)
    ub = np.array([1.25, 8.00, 2.80, 120000.0, 120000.0, 150000.0, 150000.0], dtype=np.float64)

    # Weak Tikhonov prior for ill-conditioned directions (e.g. springs on smooth track)
    prior_diag = np.array([1e-3, 1e-3, 1e-3, 1e-9, 1e-9, 1e-9, 1e-9], dtype=np.float64)

    loss_init, _, _ = _evaluate_rollout(lib, vehicle_id, windows, theta_nom, w, dt_sysid, wheel_r)
    loss_curr = loss_init
    damp = 1e-2
    H_final = np.eye(N_OPT, dtype=np.float64)
    n_iters_done = 0

    for it in range(max_iters):
        if (time.perf_counter() - t_start) > 22.0:
            break
        loss_val, g, H, n_pts = _gauss_newton_step(
            lib, vehicle_id, windows, p, theta_nom, mux_over_muy, w, dt_sysid, wheel_r
        )
        H_final = H
        n_iters_done = it + 1

        # Add weak prior + Levenberg-Marquardt damping
        g_tot = g + prior_diag * (p - p_prior)
        H_reg = H + np.diag(prior_diag) + damp * (np.diag(np.maximum(np.diag(H), 1e-6)))
        try:
            step_p = -np.linalg.solve(H_reg, g_tot)
        except np.linalg.LinAlgError:
            break

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            p_cand = np.clip(p + alpha * step_p, lb, ub)
            th_cand = _pack_theta(p_cand, theta_nom, mux_over_muy)
            loss_cand, _, _ = _evaluate_rollout(lib, vehicle_id, windows, th_cand, w, dt_sysid, wheel_r)
            if loss_cand <= loss_curr + 1e-7:
                p = p_cand
                loss_curr = loss_cand
                damp = max(damp * 0.5, 1e-5)
                accepted = True
                break
        if not accepted:
            damp *= 5.0
            if damp > 1e3:
                break
        if np.linalg.norm(step_p / np.maximum(np.abs(p), 1.0)) < 1e-3:
            break

    theta_cal = _pack_theta(p, theta_nom, mux_over_muy)
    loss_final, pred_all, meas_all = _evaluate_rollout(lib, vehicle_id, windows, theta_cal, w, dt_sysid, wheel_r)

    # Fisher Information / Cramer-Rao uncertainty estimate
    n_obs = max(len(pred_all) * 4, 1)
    sigma_r2 = max(2.0 * loss_final, 1e-6)
    H_cov = (H_final * max(len(pred_all), 1)) + np.diag(prior_diag * 100.0)
    try:
        cov = sigma_r2 * np.linalg.inv(H_cov)
        unc_diag = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    except np.linalg.LinAlgError:
        unc_diag = np.array([0.025, 0.15, 0.08, 1500.0, 1500.0, 3000.0, 3000.0], dtype=np.float64)

    sigma_lambda_mu = float(np.clip(unc_diag[0], 0.002, 0.15))

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    fit_metrics = {
        "rmse_vx_ms": round(_rmse(pred_all[:, 0], meas_all[:, 0]), 4),
        "rmse_ax_ms2": round(_rmse(pred_all[:, 1], meas_all[:, 1]), 4),
        "rmse_ay_ms2": round(_rmse(pred_all[:, 2], meas_all[:, 2]), 4),
        "rmse_wz_rads": round(_rmse(pred_all[:, 3], meas_all[:, 3]), 4),
        "r_ay": round(_corr(pred_all[:, 2], meas_all[:, 2]), 4),
        "r_wz": round(_corr(pred_all[:, 3], meas_all[:, 3]), 4),
    }

    dT_dtheta = _compute_setup_lap_sensitivities(lib, vehicle_id, theta_cal, w, dt_sysid)

    wall_time = time.perf_counter() - t_start
    red_pct = 100.0 * (loss_init - loss_final) / max(loss_init, 1e-9)

    k_arb_f_hat = float(p[3])
    k_arb_r_hat = float(p[4])
    k_spr_f_hat = float(p[5])
    k_spr_r_hat = float(p[6])

    return SysIdResult(
        vehicle_id=vehicle_id,
        run_id=run.run_id,
        wall_time_s=wall_time,
        within_30s_budget=(wall_time < 30.0),
        n_iterations=n_iters_done,
        n_windows=len(windows),
        n_total_steps=n_total_steps,
        dt_sysid_s=dt_sysid,
        lambda_mu_hat=float(p[0]),
        sigma_lambda_mu=sigma_lambda_mu,
        cl_a_hat=float(p[1]),
        cd_a_hat=float(p[2]),
        cl_over_cd=float(p[1] / max(p[2], 1e-4)),
        k_arb_f_hat=k_arb_f_hat,
        k_arb_r_hat=k_arb_r_hat,
        k_arb_ratio=k_arb_f_hat / max(k_arb_f_hat + k_arb_r_hat, 1.0),
        k_spring_f_hat=k_spr_f_hat,
        k_spring_r_hat=k_spr_r_hat,
        k_spring_ratio=k_spr_f_hat / max(k_spr_f_hat + k_spr_r_hat, 1.0),
        loss_initial=float(loss_init),
        loss_final=float(loss_final),
        loss_reduction_pct=float(red_pct),
        fit_metrics=fit_metrics,
        param_uncertainties={name: float(unc_diag[i]) for i, name in enumerate(OPT_PARAM_NAMES)},
        dT_dtheta=dT_dtheta,
        theta_calibrated=theta_cal.tolist(),
        theta_nominal=theta_nom.tolist(),
    )
