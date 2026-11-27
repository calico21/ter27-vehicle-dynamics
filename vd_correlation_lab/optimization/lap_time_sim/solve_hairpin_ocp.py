#!/usr/bin/env python3
"""
Direct-Collocation Minimum-Time Hairpin OCP for Tecnun e-Racing TeR27 4WD (MirenaSim).

Scenario:
  - Straight approach (22 m) entering at v_0 = 20.0 m/s (72 km/h)
  - 180 deg hairpin (R = 5.5 m, arc length pi * 5.5 = 17.279 m with short clothoid ramps
    preserving exact 180 deg total heading change)
  - 40.0 m straight exit (total path length ~ 79.3 m)

Method:
  - Spatial trapezoidal direct collocation in arc-length s over N intervals (N+1 nodes)
  - States per node: [vx, vy, r, e_y, e_psi]
  - Controls per node: [delta, kappa_FL, kappa_FR, kappa_RL, kappa_RR, f_brk_hyd]
  - Exact vectorized MF 6.2 combined-slip tire model, lateral+longitudinal load transfer,
    aero downforce/drag, motor torque/speed limits, and strict HV bounds [-25.0, +80.0] kW.
  - Block-colored sparse finite-difference Jacobians for fast SLSQP convergence.

Outputs:
  - vd_correlation_lab/results/tv_oracle/hairpin_ocp_report.json
  - vd_correlation_lab/results/tv_oracle/hairpin_ocp_trajectories.png
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
sys.path.insert(0, str(ORACLE_DIR))

from vd_correlation_lab.models.vehicle_14dof.ter27_physics import TeR27PhysicsModel  # noqa: E402


def build_hairpin_track(
    l_entry: float = 22.0,
    radius_m: float = 5.5,
    l_exit: float = 40.0,
    l_clothoid: float = 2.0,
    n_entry: int = 8,
    n_turn: int = 12,
    n_exit: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Constructs a smooth C1 curvature profile kappa_path(s) with:
      - straight entry of length l_entry
      - 180 deg left turn of minimum radius radius_m (exact integral(kappa ds) = pi)
      - straight exit of length l_exit
    """
    k_max = 1.0 / radius_m
    # For a trapezoidal curvature pulse of peak k_max and ramp length l_clothoid:
    # area = k_max * (l_const + l_clothoid) = pi  => l_const = pi / k_max - l_clothoid
    l_arc_equiv = math.pi * radius_m
    l_const = max(l_arc_equiv - l_clothoid, 1.0)
    l_turn_total = l_const + 2.0 * l_clothoid

    s_entry = np.linspace(0.0, l_entry, n_entry + 1)
    s_turn = np.linspace(l_entry, l_entry + l_turn_total, n_turn + 1)[1:]
    s_exit = np.linspace(l_entry + l_turn_total, l_entry + l_turn_total + l_exit, n_exit + 1)[1:]
    s = np.concatenate([s_entry, s_turn, s_exit])

    kappa = np.zeros_like(s)
    s1 = l_entry
    s2 = l_entry + l_clothoid
    s3 = l_entry + l_clothoid + l_const
    s4 = l_entry + l_turn_total

    for i, si in enumerate(s):
        if s1 < si < s2:
            u = (si - s1) / l_clothoid
            kappa[i] = k_max * (3.0 * u * u - 2.0 * u * u * u)
        elif s2 <= si <= s3:
            kappa[i] = k_max
        elif s3 < si < s4:
            u = (s4 - si) / l_clothoid
            kappa[i] = k_max * (3.0 * u * u - 2.0 * u * u * u)

    # Normalize curvature slightly so discrete trapezoidal integral equals pi exactly
    psi_int = float(np.trapezoid(kappa, s))
    if psi_int > 1e-6:
        kappa *= math.pi / psi_int

    # Integrate centerline (X, Y, psi) for plotting
    psi = np.zeros_like(s)
    x_c = np.zeros_like(s)
    y_c = np.zeros_like(s)
    for i in range(len(s) - 1):
        ds = s[i + 1] - s[i]
        psi[i + 1] = psi[i] + 0.5 * (kappa[i] + kappa[i + 1]) * ds
        x_c[i + 1] = x_c[i] + 0.5 * (math.cos(psi[i]) + math.cos(psi[i + 1])) * ds
        y_c[i + 1] = y_c[i] + 0.5 * (math.sin(psi[i]) + math.sin(psi[i + 1])) * ds

    return {
        "s": s,
        "ds": np.diff(s),
        "kappa": kappa,
        "psi": psi,
        "x_c": x_c,
        "y_c": y_c,
        "s_turn_start": s1,
        "s_apex": 0.5 * (s1 + s4),
        "s_turn_end": s4,
        "s_total": float(s[-1]),
    }


def eval_nodes_vectorized(
    model: TeR27PhysicsModel,
    z_mat: np.ndarray,
    kappa_path: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized evaluation of vehicle dynamics and constraints across all M nodes at once.
    z_mat shape: (M, 11) where columns are:
      [0: vx, 1: vy, 2: r, 3: e_y, 4: e_psi, 5: delta, 6: kFL, 7: kFR, 8: kRL, 9: kRR, 10: f_brk]
    """
    p = model.p
    M = z_mat.shape[0]
    vx = z_mat[:, 0]
    vy = z_mat[:, 1]
    r = z_mat[:, 2]
    e_y = z_mat[:, 3]
    e_psi = z_mat[:, 4]
    delta = z_mat[:, 5]
    kap = z_mat[:, 6:10]  # (M, 4)
    f_brk = z_mat[:, 10]  # hydraulic brake force demand (N), >= 0

    # Ackermann wheel steer angles (vectorized)
    abs_d = np.abs(delta)
    sgn_d = np.where(delta >= 0.0, 1.0, -1.0)
    ack = float(np.clip(p.ackermann_factor, -1.0, 1.0))
    tan_d = np.maximum(np.tan(abs_d), 1e-6)
    r_turn = p.wheelbase / tan_d
    half_t = 0.5 * p.track_front
    is_inner_fl = delta > 0.0
    r_fl = np.where(is_inner_fl, np.maximum(r_turn - half_t, 0.25 * p.wheelbase), r_turn + half_t)
    r_fr = np.where(is_inner_fl, r_turn + half_t, np.maximum(r_turn - half_t, 0.25 * p.wheelbase))
    d_fl = np.where(abs_d < 1e-6, delta, sgn_d * (abs_d + ack * (np.arctan2(p.wheelbase, r_fl) - abs_d)))
    d_fr = np.where(abs_d < 1e-6, delta, sgn_d * (abs_d + ack * (np.arctan2(p.wheelbase, r_fr) - abs_d)))
    delta_w = np.stack([d_fl, d_fr, np.zeros(M), np.zeros(M)], axis=1)  # (M, 4)

    cos_dw = np.cos(delta_w)
    sin_dw = np.sin(delta_w)

    # Wheel hub velocities & slip angles
    x_w = model.x_w[None, :]  # (1, 4)
    y_w = model.y_w[None, :]  # (1, 4)
    v_hub_x = vx[:, None] - r[:, None] * y_w
    v_hub_y = vy[:, None] + r[:, None] * x_w
    v_wx = v_hub_x * cos_dw + v_hub_y * sin_dw
    v_wy = -v_hub_x * sin_dw + v_hub_y * cos_dw
    u_den = np.maximum(np.abs(v_wx), p.slip_reg_speed)
    alpha = -np.arctan2(v_wy, u_den)

    # Aerodynamics
    speed = np.sqrt(vx * vx + vy * vy + 1e-8)
    q = 0.5 * p.air_density * speed * speed
    f_drag_x = -q * p.cd_a * (vx / speed)
    f_drag_y = -q * p.cd_a * (vy / speed)
    f_down_f = q * p.cl_a * p.aero_balance_front * model.env_aero_cl_scale
    f_down_r = q * p.cl_a * (1.0 - p.aero_balance_front) * model.env_aero_cl_scale

    # Fixed-point load transfer iteration (3 iterations for exact self-consistency)
    mg = p.mass * p.gravity
    fz_f_stat = mg * (p.b_cog / p.wheelbase)
    fz_r_stat = mg * (p.a_cog / p.wheelbase)
    m_s = p.sprung_mass
    m_u_axle = 0.5 * p.unsprung_mass_total
    k_roll_tot = p.k_roll_f + p.k_roll_r
    roll_denom = max(k_roll_tot - m_s * p.gravity * p.h_roll_arm, 1.0)
    g_stat = np.array(
        [p.static_camber_f_rad, -p.static_camber_f_rad, p.static_camber_r_rad, -p.static_camber_r_rad]
    )[None, :]
    g_steer = np.zeros((M, 4))
    g_steer[:, 0] = p.caster_rad * sin_dw[:, 0] + p.kpi_rad * (1.0 - cos_dw[:, 0])
    g_steer[:, 1] = p.caster_rad * sin_dw[:, 1] - p.kpi_rad * (1.0 - cos_dw[:, 1])
    is_right_2d = np.broadcast_to(model.IS_RIGHT[None, :], (M, 4))

    ax_cur = np.zeros(M)
    ay_cur = vx * r
    fz = np.zeros((M, 4))
    fx_w = np.zeros((M, 4))
    fy_w = np.zeros((M, 4))
    mz_tire = np.zeros((M, 4))
    fx_c = np.zeros((M, 4))
    fy_c = np.zeros((M, 4))

    for _ in range(3):
        dfz_long = (p.mass * ax_cur * p.h_cog) / p.wheelbase
        fz_f_tot = np.maximum(fz_f_stat - dfz_long + f_down_f, 10.0)
        fz_r_tot = np.maximum(fz_r_stat + dfz_long + f_down_r, 10.0)
        phi = (m_s * ay_cur * p.h_roll_arm) / roll_denom
        dfz_lat_f = (
            (p.k_roll_f * phi) / p.track_front
            + (m_s * (p.b_cog / p.wheelbase) * ay_cur * p.h_rc_f) / p.track_front
            + (m_u_axle * ay_cur * p.wheel_radius) / p.track_front
        )
        dfz_lat_r = (
            (p.k_roll_r * phi) / p.track_rear
            + (m_s * (p.a_cog / p.wheelbase) * ay_cur * p.h_rc_r) / p.track_rear
            + (m_u_axle * ay_cur * p.wheel_radius) / p.track_rear
        )
        dfz_lat_f = np.clip(dfz_lat_f, -0.49 * fz_f_tot, 0.49 * fz_f_tot)
        dfz_lat_r = np.clip(dfz_lat_r, -0.49 * fz_r_tot, 0.49 * fz_r_tot)
        fz = np.stack(
            [
                0.5 * fz_f_tot - dfz_lat_f,
                0.5 * fz_f_tot + dfz_lat_f,
                0.5 * fz_r_tot - dfz_lat_r,
                0.5 * fz_r_tot + dfz_lat_r,
            ],
            axis=1,
        )
        gamma = g_stat - phi[:, None] + g_steer
        fx_w, fy_w, mz_tire, _ = model.tire.compute_combined_forces_vec(
            kap, alpha, fz, gamma, model.lmux, model.lmuy, is_right_2d, p.p_infl_kpa
        )
        fx_c = fx_w * cos_dw - fy_w * sin_dw
        fy_c = fx_w * sin_dw + fy_w * cos_dw
        ax_new = (np.sum(fx_c, axis=1) + f_drag_x) / p.mass
        ay_new = (np.sum(fy_c, axis=1) + f_drag_y) / p.mass
        ax_cur = 0.5 * ax_cur + 0.5 * ax_new
        ay_cur = 0.5 * ay_cur + 0.5 * ay_new

    sum_fx = np.sum(fx_c, axis=1) + f_drag_x
    sum_fy = np.sum(fy_c, axis=1) + f_drag_y
    sum_mz = np.sum(x_w * fy_c - y_w * fx_c + mz_tire, axis=1)

    # Rigid-body time derivatives (ISO 8855)
    vx_dot = sum_fx / p.mass + r * vy
    vy_dot = sum_fy / p.mass - r * vx
    r_dot = sum_mz / p.izz_total

    # Curvilinear path kinematics
    cos_ep = np.cos(e_psi)
    sin_ep = np.sin(e_psi)
    s_dot = (vx * cos_ep - vy * sin_ep) / np.maximum(1.0 - kappa_path * e_y, 0.15)
    e_y_dot = vx * sin_ep + vy * cos_ep
    e_psi_dot = r - kappa_path * s_dot

    inv_s_dot = 1.0 / np.maximum(s_dot, 0.5)
    # Spatial state derivatives d(state)/ds
    d_state_ds = np.stack(
        [vx_dot * inv_s_dot, vy_dot * inv_s_dot, r_dot * inv_s_dot, e_y_dot * inv_s_dot, e_psi_dot * inv_s_dot],
        axis=1,
    )

    # Wheel torques, hydraulic brake split, motor bounds, and HV power
    rho_rad = np.minimum(fz / p.k_tire_z, 0.5 * p.wheel_radius)
    r_load = p.wheel_radius - rho_rad
    r_eff = p.wheel_radius - rho_rad / 3.0
    f_rr = p.rolling_resistance_coeff * fz * np.tanh(v_wx / 0.5)
    t_total = (fx_w + f_rr) * r_load

    bf = p.brake_bias_front
    hyd_split = np.array([0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf)])[None, :]
    t_hyd = f_brk[:, None] * hyd_split * r_load  # >= 0
    t_motor = t_total + t_hyd  # T_total = T_motor - T_hyd

    omega_w = (v_wx + kap * u_den) / np.maximum(r_eff, 0.1)

    # Vectorized motor torque limits
    v_scale = np.clip(p.v_oc_nom / p.v_nom, 0.5, 1.25)
    omega_base_m = p.base_speed_rpm * (2.0 * math.pi / 60.0) * v_scale
    omega_max_m = p.motor_max_rpm * (2.0 * math.pi / 60.0)
    omega_taper_start = omega_max_m * p.motor_taper_ratio
    p_motor_peak = p.motor_peak_torque * omega_base_m
    om_m = np.abs(omega_w) * p.gear_ratio
    t_env = np.where(om_m > omega_base_m, np.minimum(p.motor_peak_torque, p_motor_peak / np.maximum(om_m, 1e-3)), p.motor_peak_torque)
    spd_scale = np.where(
        om_m <= omega_taper_start,
        1.0,
        np.where(om_m >= omega_max_m, 0.0, 1.0 - (om_m - omega_taper_start) / max(omega_max_m - omega_taper_start, 1e-5)),
    )
    t_max = t_env * p.gear_ratio * spd_scale
    t_min = -np.minimum(p.t_regen_max_wheel, t_max)

    p_mech = t_motor * omega_w
    p_elec_ecu = np.sum(np.where(p_mech >= 0.0, p_mech / p.eta_drive, p_mech * p.eta_regen), axis=1)
    p_elec_drv = np.sum(np.where(p_mech >= 0.0, p_mech / p.eta_drive, p_mech * p.eta_drive), axis=1)
    p_drive_gross = np.sum(np.where(p_mech > 0.0, p_mech / p.eta_drive, 0.0), axis=1)
    p_regen_credit = np.sum(np.where(p_mech < 0.0, -p_mech * p.eta_regen, 0.0), axis=1)

    mz_tv_long = np.sum(-y_w * (fx_w * cos_dw) + x_w * (fx_w * sin_dw), axis=1)

    return {
        "d_state_ds": d_state_ds,
        "s_dot": s_dot,
        "inv_s_dot": inv_s_dot,
        "ax": sum_fx / p.mass,
        "ay": sum_fy / p.mass,
        "fz": fz,
        "fx_w": fx_w,
        "fy_w": fy_w,
        "alpha": alpha,
        "delta_w": delta_w,
        "t_motor": t_motor,
        "t_hyd": t_hyd,
        "t_max": t_max,
        "t_min": t_min,
        "omega_w": omega_w,
        "p_elec_ecu": p_elec_ecu,
        "p_elec_drv": p_elec_drv,
        "p_drive_gross": p_drive_gross,
        "p_regen_credit": p_regen_credit,
        "mz_tv_long": mz_tv_long,
    }


def solve_hairpin_ocp(profile: str = "FSG_REAL_TRACK") -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    t0 = time.perf_counter()
    model = TeR27PhysicsModel(profile=profile)
    p = model.p
    trk = build_hairpin_track()
    s = trk["s"]
    ds = trk["ds"]
    kp = trk["kappa"]
    M = len(s)  # number of nodes
    N = M - 1   # number of intervals
    NV = 11     # variables per node

    # Scaling factors for well-conditioned NLP
    # [vx, vy, r, e_y, e_psi, delta, kFL, kFR, kRL, kRR, f_brk]
    scale = np.array([15.0, 2.0, 2.0, 1.0, 0.5, 0.38, 0.12, 0.12, 0.12, 0.12, 4000.0], dtype=float)
    inv_scale = 1.0 / scale

    def unpack(x_sc: np.ndarray) -> np.ndarray:
        return x_sc.reshape((M, NV)) * scale[None, :]

    def pack(z_mat: np.ndarray) -> np.ndarray:
        return (z_mat * inv_scale[None, :]).reshape(-1)

    # Initial guess: smooth speed profile braking from 20 m/s to apex speed and accelerating out
    v_apex = 7.6 if profile == "FSG_REAL_TRACK" else 8.8
    z0 = np.zeros((M, NV), dtype=float)
    for k in range(M):
        sk = s[k]
        if sk <= trk["s_turn_start"]:
            u = sk / trk["s_turn_start"]
            vk = math.sqrt(max(20.0**2 * (1.0 - u) + v_apex**2 * u, v_apex**2))
            z0[k, 0] = vk
            z0[k, 6:10] = -0.04 if u > 0.35 else 0.0
            z0[k, 10] = 1500.0 if u > 0.4 else 0.0
        elif sk <= trk["s_turn_end"]:
            vk = v_apex
            z0[k, 0] = vk
            z0[k, 2] = vk * kp[k]
            z0[k, 5] = min(math.atan(p.wheelbase * kp[k]) * 1.15, p.max_steer_angle * 0.92)
            z0[k, 1] = -vk * 0.03
            z0[k, 6:10] = np.array([0.005, 0.02, 0.0, 0.025])
        else:
            u = (sk - trk["s_turn_end"]) / (trk["s_total"] - trk["s_turn_end"])
            vk = math.sqrt(v_apex**2 + 2.0 * 8.0 * (sk - trk["s_turn_end"]))
            z0[k, 0] = min(vk, 28.0)
            z0[k, 6:10] = 0.05
    z0[0, 0] = 20.0
    z0[0, 1:5] = 0.0

    # Bounds
    lb = np.zeros((M, NV), dtype=float)
    ub = np.zeros((M, NV), dtype=float)
    lb[:, 0], ub[:, 0] = 3.0, 34.0        # vx
    lb[:, 1], ub[:, 1] = -4.0, 4.0        # vy
    lb[:, 2], ub[:, 2] = -2.5, 3.0        # r
    lb[:, 3], ub[:, 3] = -0.45, 0.45      # e_y (tight corridor around R=5.5m hairpin)
    lb[:, 4], ub[:, 4] = -0.35, 0.35      # e_psi
    lb[:, 5], ub[:, 5] = -0.15, p.max_steer_angle  # delta
    lb[:, 6:10], ub[:, 6:10] = -0.14, 0.14         # kappa_i
    lb[:, 10], ub[:, 10] = 0.0, p.brake_mu * p.mass * p.gravity  # f_brk_hyd >= 0

    # Initial boundary condition at s=0: straight entry at 20.0 m/s
    lb[0, 0] = ub[0, 0] = 20.0
    lb[0, 1] = ub[0, 1] = 0.0
    lb[0, 2] = ub[0, 2] = 0.0
    lb[0, 3] = ub[0, 3] = 0.0
    lb[0, 4] = ub[0, 4] = 0.0
    # Final exit heading aligned with straight
    lb[-1, 3], ub[-1, 3] = -0.45, 0.45
    lb[-1, 4], ub[-1, 4] = -0.12, 0.12

    bounds_sc = list(zip(pack(lb), pack(ub)))

    p_max_w = p.p_dc_max_w - 100.0  # 100 W safety margin -> strictly <= 80.0 kW
    p_min_w = p.p_dc_min_w + 100.0  # strictly >= -25.0 kW
    state_scales = scale[:5]

    # Objective + exact gradient via block-colored FD
    def obj_and_grad(x_sc: np.ndarray) -> Tuple[float, np.ndarray]:
        z = unpack(x_sc)
        ev = eval_nodes_vectorized(model, z, kp)
        inv_sd = ev["inv_s_dot"]
        t_tot = float(np.sum(0.5 * (inv_sd[:-1] + inv_sd[1:]) * ds))
        # Tiny smooth control-rate regularization for clean, chatter-free trajectories
        d_delta = np.diff(z[:, 5])
        d_kap = np.diff(z[:, 6:10], axis=0)
        d_brk = np.diff(z[:, 10]) / 4000.0
        reg = 1e-3 * float(np.sum(d_delta * d_delta)) + 5e-3 * float(np.sum(d_kap * d_kap)) + 2e-3 * float(np.sum(d_brk * d_brk))
        f0 = t_tot + reg

        # Gradient of f0 wrt x_sc: node k only affects inv_sd[k] (directly) + control diff terms
        grad = np.zeros_like(x_sc)
        eps = 1e-6
        w_trap = np.zeros(M)
        w_trap[:-1] += 0.5 * ds
        w_trap[1:] += 0.5 * ds
        for j in range(NV):
            zp = z.copy()
            zp[:, j] += eps * scale[j]
            evp = eval_nodes_vectorized(model, zp, kp)
            d_inv_sd = (evp["inv_s_dot"] - inv_sd) / eps
            grad[j::NV] = w_trap * d_inv_sd

        # Analytical gradient of regularization terms
        g_reg = np.zeros((M, NV))
        g_reg[:-1, 5] -= 2e-3 * d_delta * scale[5]
        g_reg[1:, 5] += 2e-3 * d_delta * scale[5]
        g_reg[:-1, 6:10] -= 1e-2 * d_kap * scale[6:10][None, :]
        g_reg[1:, 6:10] += 1e-2 * d_kap * scale[6:10][None, :]
        g_reg[:-1, 10] -= 4e-3 * d_brk * (scale[10] / 4000.0)
        g_reg[1:, 10] += 4e-3 * d_brk * (scale[10] / 4000.0)
        grad += g_reg.reshape(-1)
        return f0, grad

    # Equality constraints: trapezoidal collocation defects (5 * N)
    def eq_fun(x_sc: np.ndarray) -> np.ndarray:
        z = unpack(x_sc)
        ev = eval_nodes_vectorized(model, z, kp)
        st = z[:, :5]
        f_ds = ev["d_state_ds"]
        defects = (st[1:] - st[:-1]) - 0.5 * (f_ds[:-1] + f_ds[1:]) * ds[:, None]
        return (defects / state_scales[None, :]).reshape(-1)

    def eq_jac(x_sc: np.ndarray) -> np.ndarray:
        z = unpack(x_sc)
        ev0 = eval_nodes_vectorized(model, z, kp)
        f0 = ev0["d_state_ds"]
        J = np.zeros((5 * N, M * NV), dtype=float)
        eps = 1e-6
        # Perturb all nodes simultaneously for each variable j (only 11 vectorized calls!)
        for j in range(NV):
            zp = z.copy()
            zp[:, j] += eps * scale[j]
            evp = eval_nodes_vectorized(model, zp, kp)
            df_dx = (evp["d_state_ds"] - f0) / eps  # (M, 5): derivative of node k's f_ds wrt node k's var j
            # Defect k depends on node k and node k+1
            for s_idx in range(5):
                row_idx = np.arange(N) * 5 + s_idx
                # wrt node k (col = k*NV + j)
                col_k = np.arange(N) * NV + j
                val_k = -0.5 * ds * df_dx[:-1, s_idx] / state_scales[s_idx]
                if j == s_idx:
                    val_k = val_k - 1.0
                J[row_idx, col_k] = val_k
                # wrt node k+1 (col = (k+1)*NV + j)
                col_kp1 = np.arange(1, M) * NV + j
                val_kp1 = -0.5 * ds * df_dx[1:, s_idx] / state_scales[s_idx]
                if j == s_idx:
                    val_kp1 = val_kp1 + 1.0
                J[row_idx, col_kp1] = val_kp1
        return J

    # Inequality constraints per node (10 * M):
    #   0..3:  (t_max - t_motor) / 100 >= 0
    #   4..7:  (t_motor - t_min) / 100 >= 0
    #   8:     (p_max_w - p_elec_ecu) / 10000 >= 0  (ecu uses eta_regen=0.88 < 0.90, so strictly tighter for +80 kW!)
    #   9:     (p_elec_drv - p_min_w) / 10000 >= 0  (drv uses eta_regen=0.90 > 0.88, so strictly tighter for -25 kW!)
    NC = 10

    def _eval_ineq_mat(ev: Dict[str, np.ndarray]) -> np.ndarray:
        c = np.zeros((M, NC), dtype=float)
        c[:, 0:4] = (ev["t_max"] - ev["t_motor"]) / 100.0
        c[:, 4:8] = (ev["t_motor"] - ev["t_min"]) / 100.0
        c[:, 8] = (p_max_w - ev["p_elec_ecu"]) / 10000.0
        c[:, 9] = (ev["p_elec_drv"] - p_min_w) / 10000.0
        return c

    def ineq_fun(x_sc: np.ndarray) -> np.ndarray:
        z = unpack(x_sc)
        ev = eval_nodes_vectorized(model, z, kp)
        return _eval_ineq_mat(ev).reshape(-1)

    def ineq_jac(x_sc: np.ndarray) -> np.ndarray:
        z = unpack(x_sc)
        ev0 = eval_nodes_vectorized(model, z, kp)
        c0 = _eval_ineq_mat(ev0)
        J = np.zeros((NC * M, M * NV), dtype=float)
        eps = 1e-6
        # Node k's path constraints depend ONLY on node k! Only 11 vectorized calls!
        nodes = np.arange(M)
        for j in range(NV):
            zp = z.copy()
            zp[:, j] += eps * scale[j]
            evp = eval_nodes_vectorized(model, zp, kp)
            dc_dx = (_eval_ineq_mat(evp) - c0) / eps  # (M, NC)
            col = nodes * NV + j
            for ci in range(NC):
                J[nodes * NC + ci, col] = dc_dx[:, ci]
        return J

    x0 = pack(z0)
    print(f"[Hairpin OCP | {profile}] Starting SLSQP ({M} nodes, {len(x0)} vars, {5*N} eq, {NC*M} ineq)...", flush=True)
    sol = minimize(
        obj_and_grad,
        x0,
        method="SLSQP",
        jac=True,
        bounds=bounds_sc,
        constraints=[
            {"type": "eq", "fun": eq_fun, "jac": eq_jac},
            {"type": "ineq", "fun": ineq_fun, "jac": ineq_jac},
        ],
        options={"maxiter": 300, "ftol": 1e-6, "disp": False},
    )
    elapsed_s = time.perf_counter() - t0
    z_opt = unpack(sol.x)
    ev_opt = eval_nodes_vectorized(model, z_opt, kp)

    # Time vector
    inv_sd = ev_opt["inv_s_dot"]
    t_arr = np.zeros(M)
    t_arr[1:] = np.cumsum(0.5 * (inv_sd[:-1] + inv_sd[1:]) * ds)
    t_total = float(t_arr[-1])

    # Residuals & HV verification
    eq_res = float(np.max(np.abs(eq_fun(sol.x))))
    p_ecu_kw = ev_opt["p_elec_ecu"] / 1000.0
    p_drv_kw = ev_opt["p_elec_drv"] / 1000.0
    p_max_kw = float(max(np.max(p_ecu_kw), np.max(p_drv_kw)))
    p_min_kw = float(min(np.min(p_ecu_kw), np.min(p_drv_kw)))
    hv_viols = int(np.sum((p_ecu_kw > 80.0) | (p_drv_kw > 80.0) | (p_ecu_kw < -25.0) | (p_drv_kw < -25.0)))

    # Quantitative extraction of the 3 required physical mechanisms
    vx = z_opt[:, 0]
    vy = z_opt[:, 1]
    beta_deg = np.degrees(np.arctan2(vy, np.maximum(vx, 0.5)))
    delta_deg = np.degrees(z_opt[:, 5])
    t_mot = ev_opt["t_motor"]  # (M, 4): [FL, FR, RL, RR]; left turn => FL,RL inner; FR,RR outer
    t_hyd = ev_opt["t_hyd"]
    mz_tv = ev_opt["mz_tv_long"]
    p_drive_gross_kw = ev_opt["p_drive_gross"] / 1000.0
    p_regen_credit_kw = ev_opt["p_regen_credit"] / 1000.0

    # Phase masks
    mask_entry = (s >= trk["s_turn_start"] - 10.0) & (s <= trk["s_apex"])
    mask_apex = (s >= trk["s_apex"] - 4.0) & (s <= trk["s_apex"] + 4.0)
    mask_exit_early = (s > trk["s_apex"]) & (s <= trk["s_turn_end"] + 10.0)
    mask_exit_full = s > trk["s_turn_end"] + 10.0

    # (a) Asymmetric inner regen braking at entry/apex
    # Inner wheels in left turn: FL (0) and RL (2); Outer wheels: FR (1) and RR (3)
    t_in_regen = np.maximum(-t_mot[:, 0], 0.0) + np.maximum(-t_mot[:, 2], 0.0)
    t_out_regen = np.maximum(-t_mot[:, 1], 0.0) + np.maximum(-t_mot[:, 3], 0.0)
    t_asym_regen = np.maximum(t_in_regen - t_out_regen, 0.0)
    p_in_regen_kw = (
        np.maximum(-t_mot[:, 0] * ev_opt["omega_w"][:, 0], 0.0)
        + np.maximum(-t_mot[:, 2] * ev_opt["omega_w"][:, 2], 0.0)
    ) * p.eta_regen / 1000.0

    # (b) Inner -> outer energy coupling near 80.0 kW
    # Defined where gross drive power > 70 kW (or net P_dc > 70 kW) AND inner wheels regen simultaneously
    coupling_kw = np.minimum(np.maximum(p_drive_gross_kw - 80.0, 0.0), p_regen_credit_kw)
    near_80_mask = (p_ecu_kw >= 75.0) | (p_drive_gross_kw >= 78.0)
    max_gross_drive_kw = float(np.max(p_drive_gross_kw))
    max_coupling_over_80_kw = float(np.max(coupling_kw))
    max_simultaneous_regen_while_driving_kw = float(
        np.max(np.where(p_drive_gross_kw > 15.0, p_regen_credit_kw, 0.0))
    )
    mean_regen_near_80_kw = float(np.mean(p_regen_credit_kw[near_80_mask])) if np.any(near_80_mask) else 0.0

    # (c) F/R and L/R torque distribution transition opening the wheel out of the apex
    def _dist_stats(mask: np.ndarray) -> Dict[str, float]:
        tm = t_mot[mask]
        t_f = np.sum(tm[:, 0:2], axis=1)
        t_r = np.sum(tm[:, 2:4], axis=1)
        t_l = tm[:, 0] + tm[:, 2]
        t_ri = tm[:, 1] + tm[:, 3]
        t_abs_sum = np.maximum(np.abs(t_f) + np.abs(t_r), 1e-3)
        t_lr_sum = np.maximum(np.abs(t_l) + np.abs(t_ri), 1e-3)
        return {
            "mean_T_FL_nm": float(np.mean(tm[:, 0])),
            "mean_T_FR_nm": float(np.mean(tm[:, 1])),
            "mean_T_RL_nm": float(np.mean(tm[:, 2])),
            "mean_T_RR_nm": float(np.mean(tm[:, 3])),
            "mean_front_share_pct": float(100.0 * np.mean(t_f / t_abs_sum)),
            "mean_rear_share_pct": float(100.0 * np.mean(t_r / t_abs_sum)),
            "mean_outer_minus_inner_nm": float(np.mean(t_ri - t_l)),
            "mean_outer_share_pct": float(100.0 * np.mean(t_ri / t_lr_sum)),
            "mean_delta_deg": float(np.mean(delta_deg[mask])),
            "mean_P_dc_kw": float(np.mean(p_ecu_kw[mask])),
        }

    idx_min_v = int(np.argmin(vx))

    report = {
        "profile": profile,
        "success": bool(sol.success),
        "status_message": str(sol.message),
        "nit": int(sol.nit),
        "solve_time_s": elapsed_s,
        "track": {
            "entry_straight_m": 22.0,
            "hairpin_radius_m": 5.5,
            "hairpin_angle_deg": 180.0,
            "exit_straight_m": 40.0,
            "total_length_m": float(s[-1]),
            "n_nodes": M,
        },
        "performance": {
            "total_time_s": t_total,
            "v_entry_mps": float(vx[0]),
            "v_apex_min_mps": float(vx[idx_min_v]),
            "s_apex_min_m": float(s[idx_min_v]),
            "v_exit_end_mps": float(vx[-1]),
            "peak_ay_g": float(np.max(np.abs(ev_opt["ay"])) / p.gravity),
            "peak_brake_g": float(-np.min(ev_opt["ax"]) / p.gravity),
            "peak_accel_g": float(np.max(ev_opt["ax"]) / p.gravity),
            "max_abs_beta_deg": float(np.max(np.abs(beta_deg))),
            "max_delta_deg": float(np.max(delta_deg)),
            "max_eq_defect": eq_res,
            "P_dc_max_kw": p_max_kw,
            "P_dc_min_kw": p_min_kw,
            "hv_violations": hv_viols,
        },
        "mechanism_a_asymmetric_inner_regen": {
            "peak_asym_inner_regen_nm": float(np.max(t_asym_regen[mask_entry])),
            "mean_asym_inner_regen_entry_apex_nm": float(np.mean(t_asym_regen[mask_entry])),
            "peak_inner_regen_power_kw": float(np.max(p_in_regen_kw[mask_entry])),
            "peak_mz_tv_entry_apex_nm": float(np.max(mz_tv[mask_entry])),
            "peak_hyd_brake_force_n": float(np.max(z_opt[:, 10])),
        },
        "mechanism_b_inner_to_outer_energy_coupling": {
            "max_gross_drive_power_kw": max_gross_drive_kw,
            "max_coupling_above_80kw_kw": max_coupling_over_80_kw,
            "max_simultaneous_regen_while_driving_kw": max_simultaneous_regen_while_driving_kw,
            "mean_regen_credit_near_80kw_kw": mean_regen_near_80_kw,
            "time_at_80kw_limit_s": float(np.sum(np.where(p_ecu_kw[:-1] >= 78.0, 0.5 * (inv_sd[:-1] + inv_sd[1:]) * ds, 0.0))),
        },
        "mechanism_c_torque_distribution_transition": {
            "entry_braking_turnin": _dist_stats(mask_entry),
            "apex_window": _dist_stats(mask_apex),
            "early_exit_unwinding": _dist_stats(mask_exit_early),
            "full_power_straight_exit": _dist_stats(mask_exit_full),
        },
        "trajectories": {
            "s_m": [float(v) for v in s],
            "t_s": [float(v) for v in t_arr],
            "vx_mps": [float(v) for v in vx],
            "vy_mps": [float(v) for v in vy],
            "r_rad_s": [float(v) for v in z_opt[:, 2]],
            "beta_deg": [float(v) for v in beta_deg],
            "delta_deg": [float(v) for v in delta_deg],
            "ay_g": [float(v / p.gravity) for v in ev_opt["ay"]],
            "ax_g": [float(v / p.gravity) for v in ev_opt["ax"]],
            "T_FL_nm": [float(v) for v in t_mot[:, 0]],
            "T_FR_nm": [float(v) for v in t_mot[:, 1]],
            "T_RL_nm": [float(v) for v in t_mot[:, 2]],
            "T_RR_nm": [float(v) for v in t_mot[:, 3]],
            "T_hyd_sum_nm": [float(v) for v in np.sum(t_hyd, axis=1)],
            "Mz_TV_nm": [float(v) for v in mz_tv],
            "P_dc_kw": [float(v) for v in p_ecu_kw],
            "P_drive_gross_kw": [float(v) for v in p_drive_gross_kw],
            "P_regen_credit_kw": [float(v) for v in p_regen_credit_kw],
        },
    }

    print(
        f"[Hairpin OCP | {profile}] Done in {elapsed_s:.2f}s (nit={sol.nit}, ok={sol.success}) | "
        f"T={t_total:.3f}s | v_min={vx[idx_min_v]:.2f} m/s | v_end={vx[-1]:.2f} m/s | "
        f"ay_max={report['performance']['peak_ay_g']:.2f}g | P_dc=[{p_min_kw:.2f}, {p_max_kw:.2f}] kW (viols={hv_viols}) | "
        f"eq_err={eq_res:.2e}",
        flush=True,
    )
    return report, {"s": s, "t": t_arr, "trk": trk, "z": z_opt, "ev": ev_opt}


def plot_hairpin_trajectories(report: Dict[str, object], out_path: Path) -> None:
    tr = report["trajectories"]
    t = np.asarray(tr["t_s"])
    s = np.asarray(tr["s_m"])

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 11.5), sharex=True)
    plt.rcParams.update({"font.size": 9.5})

    # 1. Speed, Ay, Ax
    ax0 = axes[0]
    ax0.plot(t, tr["vx_mps"], color="#1f77b4", lw=2.0, label="vx [m/s]")
    ax0.set_ylabel("Speed [m/s]", color="#1f77b4")
    ax0.grid(True, alpha=0.3)
    ax0_r = ax0.twinx()
    ax0_r.plot(t, tr["ay_g"], color="#d62728", lw=1.6, linestyle="-", label="ay [g]")
    ax0_r.plot(t, tr["ax_g"], color="#2ca02c", lw=1.4, linestyle="--", label="ax [g]")
    ax0_r.set_ylabel("Accel [g]")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0r, labels0r = ax0_r.get_legend_handles_labels()
    ax0.legend(lines0 + lines0r, labels0 + labels0r, loc="upper right", ncol=3, fontsize=8.5)
    perf = report["performance"]
    ax0.set_title(
        f"TeR27 4WD Hairpin Minimum-Time OCP ({report['profile']}) - "
        f"T = {perf['total_time_s']:.3f} s, v_min = {perf['v_apex_min_mps']:.2f} m/s, "
        f"ay_max = {perf['peak_ay_g']:.2f} g, HV viols = {perf['hv_violations']}"
    )

    # 2. Wheel Motor Torques
    ax1 = axes[1]
    ax1.plot(t, tr["T_FL_nm"], color="#1f77b4", lw=1.6, linestyle="--", label="T_FL (inner F)")
    ax1.plot(t, tr["T_FR_nm"], color="#1f77b4", lw=2.0, linestyle="-", label="T_FR (outer F)")
    ax1.plot(t, tr["T_RL_nm"], color="#d62728", lw=1.6, linestyle="--", label="T_RL (inner R)")
    ax1.plot(t, tr["T_RR_nm"], color="#d62728", lw=2.0, linestyle="-", label="T_RR (outer R)")
    ax1.axhline(0.0, color="#333333", lw=0.8)
    ax1.set_ylabel("Motor Torque [Nm]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=4, fontsize=8.5)

    # 3. Steering angle, sideslip, TV yaw moment
    ax2 = axes[2]
    ax2.plot(t, tr["delta_deg"], color="#9467bd", lw=1.9, label="delta [deg]")
    ax2.plot(t, tr["beta_deg"], color="#ff7f0e", lw=1.7, label="beta [deg]")
    ax2.axhline(0.0, color="#333333", lw=0.7)
    ax2.set_ylabel("Angles [deg]")
    ax2.grid(True, alpha=0.3)
    ax2_r = ax2.twinx()
    ax2_r.plot(t, tr["Mz_TV_nm"], color="#17becf", lw=1.5, linestyle="-.", label="Mz_TV [Nm]")
    ax2_r.set_ylabel("Mz_TV [Nm]", color="#17becf")
    l2, lb2 = ax2.get_legend_handles_labels()
    l2r, lb2r = ax2_r.get_legend_handles_labels()
    ax2.legend(l2 + l2r, lb2 + lb2r, loc="upper right", ncol=3, fontsize=8.5)

    # 4. HV DC Power & Inner->Outer Energy Coupling
    ax3 = axes[3]
    ax3.plot(t, tr["P_dc_kw"], color="#111111", lw=2.1, label="P_dc net [kW]")
    ax3.plot(t, tr["P_drive_gross_kw"], color="#2ca02c", lw=1.4, linestyle="--", label="P_drive gross [kW]")
    ax3.plot(t, -np.asarray(tr["P_regen_credit_kw"]), color="#1f77b4", lw=1.4, linestyle=":", label="-P_regen credit [kW]")
    ax3.axhline(80.0, color="#d62728", lw=1.2, linestyle="--", label="+80 kW HV limit")
    ax3.axhline(-25.0, color="#d62728", lw=1.2, linestyle=":", label="-25 kW regen limit")
    ax3.axhline(0.0, color="#555555", lw=0.7)
    ax3.set_ylabel("HV Power [kW]")
    ax3.set_xlabel("Time [s]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right", ncol=3, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_real, _ = solve_hairpin_ocp(profile="FSG_REAL_TRACK")
    rep_lab, _ = solve_hairpin_ocp(profile="TTC_LAB_PEAK")

    combined = {
        "vehicle_id": "ter27_4wd",
        "primary_profile": "FSG_REAL_TRACK",
        "FSG_REAL_TRACK": rep_real,
        "TTC_LAB_PEAK": rep_lab,
    }
    out_json = RESULTS_DIR / "hairpin_ocp_report.json"
    out_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    out_png = RESULTS_DIR / "hairpin_ocp_trajectories.png"
    plot_hairpin_trajectories(rep_real, out_png)

    print(f"Saved -> {out_json}", flush=True)
    print(f"Saved -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
