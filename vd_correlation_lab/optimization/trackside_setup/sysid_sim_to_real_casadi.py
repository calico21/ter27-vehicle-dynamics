#!/usr/bin/env python3
"""
Sim-to-Real Parameter Identification (Nonlinear Least Squares / MHE) in CasADi + IPOPT.

Identifies three key physical parameters of the TeR27 4WD directly from 200 Hz CSV
telemetry (compatible with SimUI TelemetryCsvLogger.gd format):
  1. mu_track   : track grip multiplier relative to nominal FSG_REAL_TRACK (scales lmux, lmuy)
  2. aero_scale : aerodynamic effectiveness multiplier (scales both CL*A and CD*A)
  3. I_zz       : total vehicle yaw moment of inertia (kg*m^2)

Formulation (Multiple-Shooting MHE / NLS in CasADi + IPOPT):
  - Decision variables:
      theta = [mu_scale, aero_scale, izz_scale]  (3 global physical parameters)
      wz_hat[0..M-1]                             (noise-free yaw rate state trajectory)
  - Objective (weighted Nonlinear Least Squares):
      min  sum_k [ ((Fx_k(theta)/m - ax_meas_k)/s_ax)^2
                 + ((Fy_k(theta)/m - ay_meas_k)/s_ay)^2
                 + ((wz_hat_k - wz_meas_k)/s_wz)^2 ]
  - Dynamics equality constraints (exact trapezoidal integration, zero noisy differentiation):
      wz_hat[k+1] - wz_hat[k] - 0.5 * dt_k * (Mz_k(theta) + Mz_{k+1}(theta)) / I_zz == 0

Validation:
  - Generates realistic 200 Hz dynamic telemetry (RK4 plant integration across straight
    acceleration/braking, high-speed sweepers, and dynamic slalom/chicane yaw transients)
    under 3 distinct parameter perturbation scenarios, injects realistic MEMS IMU Gaussian
    noise (ax, ay, wz), wheel encoder quantization + velocity noise, steering encoder
    quantization, and motor torque measurement noise, writes to TelemetryCsvLogger.gd CSV
    format, reloads via the CSV parser, and verifies parameter recovery error < 2.5% across
    all 3 parameters and all cases.

Outputs:
  - vd_correlation_lab/results/casadi_suite/sysid_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np
from scipy.signal import savgol_filter

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
OUT_DIR = LAB_DIR / "results" / "casadi_suite"

if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))

from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    TeR27CasADiVehicle,
    build_casadi_vehicle,
    sabs,
    sclip,
    smax,
    smin,
    spos,
    ssign,
)

# Exact CSV header from src/mirena_sim/Scenes/Sim/UI/TelemetryCsvLogger.gd
SIMUI_CSV_COLUMNS = [
    "t", "lap", "s_m", "vx", "vy", "wz", "ax", "ay",
    "beta_deg", "beta_dot_dps", "delta_raw", "delta_deg",
    "thr", "brk", "T_FL", "T_FR", "T_RL", "T_RR",
    "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR",
    "P_dc_kw", "Mz_TV_nm", "tv_mode", "tv_regime",
]


def build_parametric_sysid_function(veh: TeR27CasADiVehicle) -> ca.Function:
    """
    Build a smooth C^2 CasADi SX function evaluating (Fx_tot, Fy_tot, Mz_tot) as a function
    of both vehicle state/inputs AND the symbolic parameters (mu_scale, aero_scale).
    """
    p = veh.params
    t = veh.tire
    s = veh.smoothing

    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_T4 = ca.SX.sym("T_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    s_mu = ca.SX.sym("mu_scale")
    s_aero = ca.SX.sym("aero_scale")

    # Parametric tire MF 6.2 evaluator (scaling lmux, lmuy by s_mu)
    def eval_tire_param(kappa: Any, alpha: Any, fz: Any, gamma: Any, side: Any) -> Tuple[Any, Any, Any]:
        p_infl = p.p_infl_kpa
        fz_pos = spos(fz, s["eps_fz_n"])
        dfz_raw = (fz_pos - t.fnomin) / t.fnomin
        dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"]) if math.isfinite(t.dfz_max) else dfz_raw
        dpi = (p_infl - t.pinom) / t.pinom
        abs_gamma = sabs(gamma, s["eps_gamma_rad"])
        gamma_sq = gamma * gamma

        # Pure lateral Fy0
        cy = t.PCY1
        camber_f_y = smax(1.0 - t.PDY3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_y = (
            (t.PDY1 + t.PDY2 * dfz)
            * (1.0 + t.PPY3 * dpi + t.PPY4 * dpi * dpi)
            * camber_f_y
            * (veh.lmuy * s_mu)
        )
        dy = mu_y * fz_pos
        shy = side * (t.PHY1 + t.PHY2 * dfz) + t.PHY3 * gamma
        svy = side * fz_pos * (t.PVY1 + t.PVY2 * dfz) + fz_pos * (t.PVY3 + t.PVY4 * dfz) * gamma
        alpha_y = alpha + shy
        kya = (
            t.PKY1
            * t.fnomin
            * (1.0 + t.PPY1 * dpi)
            * (1.0 - t.PKY3 * abs_gamma)
            * ca.sin(2.0 * ca.atan(fz_pos / (t.PKY2 * t.fnomin * (1.0 + t.PPY2 * dpi) + 1e-4)))
        )
        by = kya / (cy * dy + 1e-4)
        ey_raw = (t.PEY1 + t.PEY2 * dfz) * (
            1.0 + t.PEY5 * gamma_sq - (t.PEY3 + t.PEY4 * gamma) * ssign(alpha_y, s["eps_sign_slip"])
        )
        ey = sclip(ey_raw, -10.0, 1.0, s["eps_e_clip"])
        y = by * alpha_y
        fy0 = dy * ca.sin(cy * ca.atan(y - ey * (y - ca.atan(y)))) + svy

        # Pure longitudinal Fx0
        cx = t.PCX1
        camber_f_x = smax(1.0 - t.PDX3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_x = (
            (t.PDX1 + t.PDX2 * dfz)
            * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
            * camber_f_x
            * (veh.lmux * s_mu)
        )
        dx = mu_x * fz_pos
        kx = kappa + t.PHX1 + t.PHX2 * dfz
        svx = fz_pos * (t.PVX1 + t.PVX2 * dfz)
        kxk = (
            fz_pos
            * (t.PKX1 + t.PKX2 * dfz)
            * ca.exp(t.PKX3 * dfz)
            * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
        )
        bx = kxk / (cx * dx + 1e-4)
        ex_raw = (t.PEX1 + t.PEX2 * dfz + t.PEX3 * dfz * dfz) * (
            1.0 - t.PEX4 * ssign(kx, s["eps_sign_slip"])
        )
        ex = sclip(ex_raw, -10.0, 1.0, s["eps_e_clip"])
        x = bx * kx
        fx0 = dx * ca.sin(cx * ca.atan(x - ex * (x - ca.atan(x)))) + svx

        # Combined slip
        bxa = (t.RBX1 + t.RBX3 * gamma_sq) * ca.cos(ca.atan(t.RBX2 * kappa))
        exa = sclip(t.REX1 + t.REX2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        gxa_den = smax(veh._shape_cos(bxa, t.RCX1, exa, t.RHX1), 1e-4, 1e-5)
        gxa = smax(veh._shape_cos(bxa, t.RCX1, exa, alpha + t.RHX1) / gxa_den, 0.0, s["eps_g_floor"])

        byk = (t.RBY1 + t.RBY4 * gamma_sq) * ca.cos(ca.atan(t.RBY2 * (alpha - t.RBY3)))
        eyk = sclip(t.REY1 + t.REY2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        shyk = t.RHY1 + t.RHY2 * dfz
        gyk_den = smax(veh._shape_cos(byk, t.RCY1, eyk, shyk), 1e-4, 1e-5)
        gyk = smax(veh._shape_cos(byk, t.RCY1, eyk, kappa + shyk) / gyk_den, 0.0, s["eps_g_floor"])

        dvyk = (
            mu_y
            * fz_pos
            * (t.RVY1 + t.RVY2 * dfz + t.RVY3 * gamma)
            * ca.cos(ca.atan(t.RVY4 * alpha))
        )
        svyk = dvyk * ca.sin(t.RVY5 * ca.atan(t.RVY6 * kappa))

        fx = gxa * fx0
        fy = gyk * fy0 + svyk

        # Self-aligning torque Mz
        shf = shy + svy / (kya + 1e-3)
        sht = t.QHZ1 + t.QHZ2 * dfz + (t.QHZ3 + t.QHZ4 * dfz) * gamma
        alpha_t = alpha + sht
        alpha_r = alpha + shf
        kx_eq = (kxk / (kya + 1e-3)) * kappa
        alpha_t_mag = ca.sqrt(alpha_t * alpha_t + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_r_mag = ca.sqrt(alpha_r * alpha_r + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_t_signed = ssign(alpha_t, s["eps_sign_slip"]) * alpha_t_mag

        dt_t = (
            (fz_pos / t.fnomin)
            * t.r0
            * (t.QDZ1 + t.QDZ2 * dfz)
            * (1.0 + t.PPZ1 * dpi)
            * (1.0 + t.QDZ3 * abs_gamma + t.QDZ4 * gamma_sq)
        )
        ct = t.QCZ1
        bt = (t.QBZ1 + t.QBZ2 * dfz + t.QBZ3 * dfz * dfz) * (
            1.0 + t.QBZ4 * gamma + t.QBZ5 * abs_gamma
        )
        t_arg = bt * alpha_t_mag
        et_raw = (t.QEZ1 + t.QEZ2 * dfz + t.QEZ3 * dfz * dfz) * (
            1.0 + (t.QEZ4 + t.QEZ5 * gamma) * (2.0 / math.pi) * ca.atan(bt * ct * alpha_t_signed)
        )
        et = sclip(et_raw, -10.0, 1.0, s["eps_e_clip"])
        trail = dt_t * ca.cos(ct * ca.atan(t_arg - et * (t_arg - ca.atan(t_arg)))) * ca.cos(alpha)
        mz_prime = -trail * (fy - svyk)

        dr = (
            fz_pos
            * t.r0
            * (side * (t.QDZ6 + t.QDZ7 * dfz) + (t.QDZ8 + t.QDZ9 * dfz) * (1.0 + t.PPZ2 * dpi) * gamma)
            * ca.cos(alpha)
        )
        br = t.QBZ9 + t.QBZ10 * by * cy
        mzr = dr * ca.cos(ca.atan(br * alpha_r_mag))
        s_arm = t.r0 * (t.SSZ1 + t.SSZ2 * (fy / t.fnomin) + (t.SSZ3 + t.SSZ4 * dfz) * gamma)
        mz = mz_prime + mzr + s_arm * fx
        return fx, fy, mz

    # Parametric aero forces (scaling both cd_a and cl_a by s_aero)
    eps_v = s["eps_speed_ms"]
    v_sq = s_vx * s_vx + s_vy * s_vy
    speed = ca.sqrt(v_sq + eps_v * eps_v)
    q = 0.5 * p.air_density * v_sq
    f_drag_x = -0.5 * p.air_density * (p.cd_a * s_aero) * speed * s_vx
    f_drag_y = -0.5 * p.air_density * (p.cd_a * s_aero) * speed * s_vy
    f_down_f = q * (p.cl_a * s_aero) * p.aero_balance_front * veh.env_aero_cl_scale
    f_down_r = q * (p.cl_a * s_aero) * (1.0 - p.aero_balance_front) * veh.env_aero_cl_scale

    # Normal forces with parametric downforce
    eps_fz = s["eps_fz_n"]
    mg = p.mass * p.gravity
    fz_f_stat = mg * (p.b_cog / p.wheelbase)
    fz_r_stat = mg * (p.a_cog / p.wheelbase)
    dfz_long = (p.mass * s_ax * p.h_cog) / p.wheelbase
    fz_f_tot = spos(fz_f_stat - dfz_long + f_down_f, eps_fz)
    fz_r_tot = spos(fz_r_stat + dfz_long + f_down_r, eps_fz)

    m_s = p.sprung_mass
    m_u_axle = 0.5 * p.unsprung_mass_total
    k_roll_tot = p.k_roll_f + p.k_roll_r
    roll_denom = max(k_roll_tot - m_s * p.gravity * p.h_roll_arm, 1.0)
    phi = (m_s * s_ay * p.h_roll_arm) / roll_denom

    dfz_lat_f_raw = (
        (p.k_roll_f * phi) / p.track_front
        + (m_s * (p.b_cog / p.wheelbase) * s_ay * p.h_rc_f) / p.track_front
        + (m_u_axle * s_ay * p.wheel_radius) / p.track_front
    )
    dfz_lat_r_raw = (
        (p.k_roll_r * phi) / p.track_rear
        + (m_s * (p.a_cog / p.wheelbase) * s_ay * p.h_rc_r) / p.track_rear
        + (m_u_axle * s_ay * p.wheel_radius) / p.track_rear
    )
    dfz_lat_f = sclip(dfz_lat_f_raw, -0.5 * fz_f_tot, 0.5 * fz_f_tot, eps_fz)
    dfz_lat_r = sclip(dfz_lat_r_raw, -0.5 * fz_r_tot, 0.5 * fz_r_tot, eps_fz)
    fz = ca.vertcat(
        spos(0.5 * fz_f_tot - dfz_lat_f, eps_fz),
        spos(0.5 * fz_f_tot + dfz_lat_f, eps_fz),
        spos(0.5 * fz_r_tot - dfz_lat_r, eps_fz),
        spos(0.5 * fz_r_tot + dfz_lat_r, eps_fz),
    )

    delta_w = veh.ackermann_steer_smooth(s_del)
    cos_d = ca.cos(delta_w)
    sin_d = ca.sin(delta_w)
    v_wx, _, _, alpha = veh._wheel_kinematics(s_vx, s_vy, s_wz, delta_w)
    gamma = veh.cambers_smooth(delta_w, phi)

    # Smooth inversion of T_4 -> kappa_4 under parametric mu_scale
    rho_rad = smin(fz / p.k_tire_z, 0.5 * p.wheel_radius, s["eps_r_eff"])
    r_load = p.wheel_radius - rho_rad
    f_rr = p.rolling_resistance_coeff * fz * ca.tanh(v_wx / 0.5)
    hyd_split = ca.DM(veh.hyd_split)
    T_hyd_4 = s_fbrk * hyd_split * r_load
    fx_target = (s_T4 - T_hyd_4) / r_load - f_rr

    side = ca.DM(veh.side_sign)
    fz_pos = spos(fz, s["eps_fz_n"])
    dfz_raw = (fz_pos - t.fnomin) / t.fnomin
    dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"]) if math.isfinite(t.dfz_max) else dfz_raw
    dpi = (p.p_infl_kpa - t.pinom) / t.pinom
    camber_f_x = smax(1.0 - t.PDX3 * gamma * gamma, 0.2, s["eps_camber_f"])
    mu_x = (
        (t.PDX1 + t.PDX2 * dfz)
        * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
        * camber_f_x
        * (veh.lmux * s_mu)
    )
    dx = mu_x * fz_pos
    kxk = (
        fz_pos
        * (t.PKX1 + t.PKX2 * dfz)
        * ca.exp(t.PKX3 * dfz)
        * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
    )
    bx = kxk / (t.PCX1 * dx + 1e-4)
    kappa_pk = sclip(1.65 / smax(bx, 5.0, 0.5), 0.04, 0.12, 1e-3)

    fx_pos_lim, _, _ = eval_tire_param(kappa_pk, alpha, fz, gamma, side)
    fx_neg_lim, _, _ = eval_tire_param(-kappa_pk, alpha, fz, gamma, side)
    fx_hi = smax(fx_pos_lim, fx_neg_lim, 1.0)
    fx_lo = smin(fx_pos_lim, fx_neg_lim, 1.0)
    tgt = sclip(fx_target, 0.998 * fx_lo, 0.998 * fx_hi, 0.5)

    fx_0p, _, _ = eval_tire_param(1e-3 * ca.DM.ones(4), alpha, fz, gamma, side)
    fx_0m, _, _ = eval_tire_param(-1e-3 * ca.DM.ones(4), alpha, fz, gamma, side)
    k_slope0 = smax((fx_0p - fx_0m) / 2e-3, 50.0, 5.0)
    k_cur = sclip(tgt / k_slope0, -0.85 * kappa_pk, 0.85 * kappa_pk, 1e-3)

    h_fd = 1e-5
    for _ in range(6):
        fx_c, _, _ = eval_tire_param(k_cur, alpha, fz, gamma, side)
        fx_p, _, _ = eval_tire_param(k_cur + h_fd, alpha, fz, gamma, side)
        dfx_dk = (fx_p - fx_c) / h_fd
        slope_safe = smax(dfx_dk, 0.05 * k_slope0, 5.0)
        step = sclip((fx_c - tgt) / slope_safe, -0.04, 0.04, 1e-3)
        k_cur = sclip(k_cur - step, -kappa_pk, kappa_pk, 1e-4)

    fx_w, fy_w, mz_tire = eval_tire_param(k_cur, alpha, fz, gamma, side)
    fx_c = fx_w * cos_d - fy_w * sin_d
    fy_c = fx_w * sin_d + fy_w * cos_d

    x_w = ca.DM(veh.x_w)
    y_w = ca.DM(veh.y_w)
    Fx_tot = ca.sum1(fx_c) + f_drag_x
    Fy_tot = ca.sum1(fy_c) + f_drag_y
    Mz_tot = ca.sum1(x_w * fy_c - y_w * fx_c + mz_tire)

    return ca.Function(
        "ter27_sysid_forces",
        [s_vx, s_vy, s_wz, s_del, s_T4, s_fbrk, s_ax, s_ay, s_mu, s_aero],
        [Fx_tot, Fy_tot, Mz_tot, fz],
        ["vx", "vy", "wz", "delta", "T_4", "F_brk_hyd", "ax_lt", "ay_lt", "mu_scale", "aero_scale"],
        ["Fx_tot", "Fy_tot", "Mz_tot", "Fz_4"],
    )


def generate_synthetic_200hz_telemetry_csv(
    csv_path: Path,
    veh: TeR27CasADiVehicle,
    fn_sysid: ca.Function,
    mu_true: float,
    aero_true: float,
    izz_scale_true: float,
    duration_s: float = 20.0,
    fs_hz: float = 200.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate realistic 200 Hz telemetry in exact TelemetryCsvLogger.gd format by RK4
    integrating the true perturbed nonlinear TeR27 4WD dynamics across a rich excitation
    profile (high-speed straight + braking, high-speed sweepers, dynamic slalom with TV)
    and injecting realistic MEMS IMU noise + encoder quantization.
    """
    rng = np.random.default_rng(seed)
    p = veh.params
    mass = float(p.mass)
    izz_true = float(p.izz_total) * izz_scale_true
    dt = 1.0 / fs_hz
    n_steps = int(round(duration_s * fs_hz))

    # Realistic sensor noise parameters
    sigma_ax = 0.15       # m/s^2 (MEMS accelerometer @ 200 Hz)
    sigma_ay = 0.15       # m/s^2
    sigma_wz = 0.008      # rad/s (~0.46 deg/s MEMS gyro @ 200 Hz)
    sigma_vx = 0.03       # m/s
    sigma_vy = 0.05       # m/s
    vx_quant = 0.02       # m/s wheel encoder quantization
    delta_quant_deg = 0.05  # deg steering encoder quantization
    sigma_T = 0.5         # Nm inverter torque feedback noise

    # State: [vx, vy, wz], plus algebraic acceleration memory for load transfer
    vx, vy, wz = 14.0, 0.0, 0.0
    ax_prev, ay_prev = 0.0, 0.0
    s_m = 0.0

    def driver_inputs(t_s: float, vx_cur: float, vy_cur: float, wz_cur: float) -> Tuple[float, np.ndarray, float, float, float]:
        """Smooth, physically realizable multi-regime excitation (no spins, BSPD-compliant, HV <= 80 kW)."""
        beta_cur = math.atan2(vy_cur, max(vx_cur, 2.0))
        def _env(tau_local: float, dur: float, ramp: float = 0.4) -> float:
            if tau_local <= 0.0 or tau_local >= dur:
                return 0.0
            e_in = min(1.0, tau_local / ramp)
            e_out = min(1.0, (dur - tau_local) / ramp)
            return 0.5 * (1.0 - math.cos(math.pi * min(e_in, e_out)))

        if t_s < 4.5:
            # Segment 1: Full-power straight acceleration (14 -> 29 m/s) -> excites aero drag & downforce
            delta = float(np.clip(-0.20 * wz_cur + 0.50 * beta_cur, -0.15, 0.15))
            t_drv = min(160.0, 68000.0 / max(4.0 * vx_cur / p.wheel_radius, 1.0)) * _env(t_s, 4.5, 0.3)
            T_4 = np.array([0.85 * t_drv, 0.85 * t_drv, 1.15 * t_drv, 1.15 * t_drv])
            f_brk = 0.0
            thr, brk = 0.9, 0.0
        elif t_s < 7.0:
            # Segment 2: High-speed coast / light regen -> isolates aero drag CD*A cleanly
            delta = float(np.clip(-0.20 * wz_cur + 0.50 * beta_cur, -0.15, 0.15))
            T_4 = np.array([-8.0, -8.0, -8.0, -8.0])
            f_brk = 0.0
            thr, brk = 0.0, 0.04
        elif t_s < 12.5:
            # Segment 3: Smooth high-speed S-sweeper (21-26 m/s, ~1.0g) -> excites aero CL*A + mu_track
            tau = t_s - 7.0
            e = _env(tau, 5.5, 0.5)
            ay_ref = 9.5 * e * math.sin(2.0 * math.pi * 0.25 * tau)
            wz_ref = ay_ref / max(vx_cur, 8.0)
            delta_ff = 1.15 * p.wheelbase * wz_ref / max(vx_cur, 8.0)
            delta = float(np.clip(delta_ff - 0.20 * (wz_cur - wz_ref) + 0.55 * beta_cur, -0.16, 0.16))
            t_base = min(75.0, 48000.0 / max(4.0 * vx_cur / p.wheel_radius, 1.0))
            dT_tv = 20.0 * e * math.sin(2.0 * math.pi * 0.25 * tau)
            T_4 = np.array([
                t_base - 0.5 * dT_tv,
                t_base + 0.5 * dT_tv,
                t_base - 0.8 * dT_tv,
                t_base + 0.8 * dT_tv,
            ])
            f_brk = 0.0
            thr, brk = 0.55, 0.0
        elif t_s < 14.0:
            # Segment 4: Smooth braking into medium-speed slalom (BSPD compliant: T_4 <= 0)
            tau = t_s - 12.5
            e = _env(tau, 1.5, 0.25)
            delta = float(np.clip(-0.20 * wz_cur + 0.50 * beta_cur, -0.15, 0.15))
            T_4 = np.array([-55.0, -55.0, -55.0, -55.0]) * e
            f_brk = 1400.0 * e if vx_cur > 14.5 else 0.0
            thr, brk = 0.0, 0.35 * e if f_brk > 0.0 else 0.15 * e
        else:
            # Segment 5: Smooth high-bandwidth slalom + TV yaw-moment chirp (0.6-1.3 Hz) -> excites I_zz & mu_track
            tau = t_s - 14.0
            e = _env(tau, 6.0, 0.4)
            phase = 2.0 * math.pi * (0.55 * tau + 0.06 * tau * tau)
            ay_ref = 9.0 * e * math.sin(phase)
            wz_ref = ay_ref / max(vx_cur, 8.0)
            delta_ff = 1.25 * p.wheelbase * wz_ref / max(vx_cur, 8.0)
            delta = float(np.clip(delta_ff - 0.15 * (wz_cur - wz_ref) + 0.50 * beta_cur, -0.22, 0.22))
            t_base = 45.0 if vx_cur < 16.5 else 20.0
            # Strong smooth asymmetric TV torque chirp directly exciting yaw acceleration Mz = I_zz * dwz/dt
            dT_tv = 95.0 * e * math.sin(2.0 * math.pi * (0.85 * tau + 0.08 * tau * tau))
            T_4 = np.array([
                t_base - 0.8 * dT_tv,
                t_base + 0.8 * dT_tv,
                t_base - 1.0 * dT_tv,
                t_base + 1.0 * dT_tv,
            ])
            f_brk = 0.0
            thr, brk = 0.40, 0.0
        return delta, T_4, f_brk, thr, brk

    def rhs(st: np.ndarray, delta: float, T_4: np.ndarray, f_brk: float, ax_lt: float, ay_lt: float) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        vx_s, vy_s, wz_s = float(st[0]), float(st[1]), float(st[2])
        res = fn_sysid(vx_s, vy_s, wz_s, delta, T_4.tolist(), f_brk, ax_lt, ay_lt, mu_true, aero_true)
        fx = float(res[0])
        fy = float(res[1])
        mz = float(res[2])
        fz = np.array(res[3].full()).ravel()
        dvx = fx / mass + vy_s * wz_s
        dvy = fy / mass - vx_s * wz_s
        dwz = mz / izz_true
        return np.array([dvx, dvy, dwz]), fx / mass, fy / mass, mz, fz

    rows: List[List[Any]] = []
    prev_beta_deg = 0.0

    for k in range(n_steps):
        t_s = k * dt
        delta, T_4, f_brk, thr, brk = driver_inputs(t_s, vx, vy, wz)
        st = np.array([vx, vy, wz], dtype=np.float64)

        # Fixed-point load transfer convergence (3 iterations) for exact physical acceleration
        ax_lt, ay_lt = ax_prev, ay_prev
        for _ in range(3):
            _, ax_new, ay_new, _, _ = rhs(st, delta, T_4, f_brk, ax_lt, ay_lt)
            ax_lt, ay_lt = ax_new, ay_new
        k1, ax_true, ay_true, mz_true, fz_true = rhs(st, delta, T_4, f_brk, ax_lt, ay_lt)

        # Log noisy sample in TelemetryCsvLogger.gd schema
        vx_m = round((vx + rng.normal(0.0, sigma_vx)) / vx_quant) * vx_quant
        vy_m = vy + rng.normal(0.0, sigma_vy)
        wz_m = wz + rng.normal(0.0, sigma_wz)
        ax_m = ax_true + rng.normal(0.0, sigma_ax)
        ay_m = ay_true + rng.normal(0.0, sigma_ay)
        delta_deg_m = round((math.degrees(delta)) / delta_quant_deg) * delta_quant_deg
        T_4_m = T_4 + rng.normal(0.0, sigma_T, size=4)
        beta_deg_m = math.degrees(math.atan2(vy_m, max(vx_m, 1.0)))
        beta_dot_dps = (beta_deg_m - prev_beta_deg) / dt if k > 0 else 0.0
        prev_beta_deg = beta_deg_m

        # Approximate HV power and TV yaw moment for CSV completeness
        omega_approx = vx_m / p.wheel_radius
        p_dc_kw = float(np.sum(np.where(T_4_m >= 0, T_4_m * omega_approx / p.eta_drive, T_4_m * omega_approx * p.eta_regen))) * 1e-3
        mz_tv_nm = float(0.5 * p.track_front * ((T_4_m[1] - T_4_m[0]) + (T_4_m[3] - T_4_m[2])) / p.wheel_radius)

        # Encode hydraulic brake pedal fraction matching f_brk (max_hyd_force = 4000 N -> brk = f_brk / 4000)
        brk_logged = float(f_brk / 4000.0) if f_brk > 0.0 else 0.0

        rows.append([
            f"{t_s:.4f}", 1, f"{s_m:.2f}",
            f"{vx_m:.4f}", f"{vy_m:.4f}", f"{wz_m:.5f}",
            f"{ax_m:.4f}", f"{ay_m:.4f}",
            f"{beta_deg_m:.3f}", f"{beta_dot_dps:.3f}",
            f"{math.radians(delta_deg_m):.5f}", f"{delta_deg_m:.3f}",
            f"{thr:.3f}", f"{brk_logged:.4f}",
            f"{T_4_m[0]:.2f}", f"{T_4_m[1]:.2f}", f"{T_4_m[2]:.2f}", f"{T_4_m[3]:.2f}",
            f"{fz_true[0]:.1f}", f"{fz_true[1]:.1f}", f"{fz_true[2]:.1f}", f"{fz_true[3]:.1f}",
            f"{p_dc_kw:.2f}", f"{mz_tv_nm:.2f}", "AUTOCROSS", "SYSID_EXCITE",
        ])

        # RK4 step
        k2, _, _, _, _ = rhs(st + 0.5 * dt * k1, delta, T_4, f_brk, ax_lt, ay_lt)
        k3, _, _, _, _ = rhs(st + 0.5 * dt * k2, delta, T_4, f_brk, ax_lt, ay_lt)
        k4, _, _, _, _ = rhs(st + dt * k3, delta, T_4, f_brk, ax_lt, ay_lt)
        st_next = st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        vx = max(float(st_next[0]), 4.0)
        vy = float(st_next[1])
        wz = float(st_next[2])
        ax_prev, ay_prev = ax_true, ay_true
        s_m += vx * dt

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SIMUI_CSV_COLUMNS)
        writer.writerows(rows)

    return {
        "n_samples": n_steps,
        "fs_hz": fs_hz,
        "duration_s": duration_s,
        "noise_spec": {
            "sigma_ax_mps2": sigma_ax,
            "sigma_ay_mps2": sigma_ay,
            "sigma_wz_rads": sigma_wz,
            "sigma_vx_mps": sigma_vx,
            "sigma_vy_mps": sigma_vy,
            "vx_quant_mps": vx_quant,
            "delta_quant_deg": delta_quant_deg,
            "sigma_T_nm": sigma_T,
        },
    }


def load_simui_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load a 200 Hz CSV file in SimUI TelemetryCsvLogger.gd format."""
    data: Dict[str, List[float]] = {
        k: [] for k in ["t", "vx", "vy", "wz", "ax", "ay", "delta_deg", "brk", "T_FL", "T_FR", "T_RL", "T_RR"]
    }
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                data[k].append(float(row[k]))
    arr = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    arr["delta"] = np.deg2rad(arr["delta_deg"])
    arr["T_4"] = np.vstack([arr["T_FL"], arr["T_FR"], arr["T_RL"], arr["T_RR"]])
    arr["F_brk_hyd"] = np.maximum(arr["brk"], 0.0) * 4000.0
    return arr


def identify_parameters_casadi_mhe(
    csv_path: Path,
    veh: TeR27CasADiVehicle,
    fn_sysid: ca.Function,
    decimate: int = 5,
    print_level: int = 0,
) -> Dict[str, Any]:
    """
    Estimate (mu_track_scale, aero_scale, I_zz) from 200 Hz CSV telemetry using
    Multiple-Shooting MHE / Nonlinear Least Squares in CasADi + IPOPT.
    """
    t0 = time.perf_counter()
    raw = load_simui_csv(csv_path)
    n_raw = len(raw["t"])

    # Zero-phase Savitzky-Golay pre-smoothing of load-transfer inputs (ax_lt, ay_lt) and vx
    win = 7 if n_raw >= 15 else max(5, (n_raw // 2) * 2 - 1)
    vx_f = savgol_filter(raw["vx"], win, 2)
    vy_f = savgol_filter(raw["vy"], win, 2)
    ax_lt_f = savgol_filter(raw["ax"], win, 2)
    ay_lt_f = savgol_filter(raw["ay"], win, 2)

    # Decimate 200 Hz -> 50 Hz (dt = 0.020 s) via local block averaging for low-variance residuals
    # while keeping full 50 Hz bandwidth (> 12x vehicle yaw bandwidth)
    m = n_raw // decimate
    sl = slice(0, m * decimate)

    def _bin_avg(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a[sl].reshape(m, decimate).mean(axis=1)
        return a[:, sl].reshape(a.shape[0], m, decimate).mean(axis=2)

    t_b = _bin_avg(raw["t"])
    vx_b = _bin_avg(vx_f)
    vy_meas = _bin_avg(raw["vy"])
    wz_meas = _bin_avg(raw["wz"])
    ax_meas = _bin_avg(raw["ax"])
    ay_meas = _bin_avg(raw["ay"])
    ax_lt_b = _bin_avg(ax_lt_f)
    ay_lt_b = _bin_avg(ay_lt_f)
    delta_b = _bin_avg(raw["delta"])
    T4_b = _bin_avg(raw["T_4"])
    fbrk_b = _bin_avg(raw["F_brk_hyd"])
    dt_vec = np.diff(t_b)  # length m - 1

    p = veh.params
    mass = float(p.mass)
    izz_nom = float(p.izz_total)

    # Build vectorized CasADi evaluation over all m knots
    fn_map = fn_sysid.map(m)

    # Decision variables (2-state Multiple-Shooting MHE):
    #   theta  = [mu_scale, aero_scale, izz_scale]  (3)
    #   vy_hat = noise-free lateral velocity state  (m)
    #   wz_hat = noise-free yaw-rate state          (m)
    s_mu = ca.MX.sym("mu_scale")
    s_aero = ca.MX.sym("aero_scale")
    s_izz_sc = ca.MX.sym("izz_scale")
    vy_hat = ca.MX.sym("vy_hat", 1, m)
    wz_hat = ca.MX.sym("wz_hat", 1, m)

    mu_row = ca.repmat(s_mu, 1, m)
    aero_row = ca.repmat(s_aero, 1, m)
    vx_dm = ca.DM(vx_b).T

    Fx_hat, Fy_hat, Mz_hat, _ = fn_map(
        vx_dm,
        vy_hat,
        wz_hat,
        ca.DM(delta_b).T,
        ca.DM(T4_b),
        ca.DM(fbrk_b).T,
        ca.DM(ax_lt_b).T,
        ca.DM(ay_lt_b).T,
        mu_row,
        aero_row,
    )

    ax_hat = Fx_hat / mass
    ay_hat = Fy_hat / mass

    # Measurement residuals (normalized by sensor std-dev)
    res_ax = (ax_hat - ca.DM(ax_meas).T) / 0.15
    res_ay = (ay_hat - ca.DM(ay_meas).T) / 0.15
    res_wz = (wz_hat - ca.DM(wz_meas).T) / 0.008
    res_vy = (vy_hat - ca.DM(vy_meas).T) / 0.05

    # Weighted Nonlinear Least Squares objective (normalized by m)
    J_ax = ca.sum2(res_ax * res_ax) / float(m)
    J_ay = ca.sum2(res_ay * res_ay) / float(m)
    J_wz = ca.sum2(res_wz * res_wz) / float(m)
    J_vy = 0.25 * ca.sum2(res_vy * res_vy) / float(m)
    obj = J_ax + J_ay + J_wz + J_vy

    # Exact trapezoidal lateral & yaw dynamics constraints:
    #   vy_hat[k+1] - vy_hat[k] - 0.5 * dt_k * (dvy_k + dvy_{k+1}) == 0
    #   wz_hat[k+1] - wz_hat[k] - 0.5 * dt_k * (Mz_k + Mz_{k+1}) / I_zz == 0
    dvy_hat = ay_hat - vx_dm * wz_hat
    izz_sym = izz_nom * s_izz_sc
    dt_dm = ca.DM(dt_vec).T
    vy_defects = vy_hat[0, 1:] - vy_hat[0, :-1] - 0.5 * dt_dm * (dvy_hat[0, :-1] + dvy_hat[0, 1:])
    wz_defects = (
        wz_hat[0, 1:] - wz_hat[0, :-1]
        - 0.5 * dt_dm * (Mz_hat[0, :-1] + Mz_hat[0, 1:]) / izz_sym
    )

    x_vars = ca.vertcat(s_mu, s_aero, s_izz_sc, vy_hat.T, wz_hat.T)
    lbx = np.concatenate([[0.45, 0.30, 0.50], np.full(m, -6.0), np.full(m, -4.0)])
    ubx = np.concatenate([[1.55, 2.00, 1.80], np.full(m, 6.0), np.full(m, 4.0)])
    x0 = np.concatenate([[1.0, 1.0, 1.0], vy_meas, wz_meas])

    g_vec = ca.vertcat(vy_defects.T, wz_defects.T)
    lbg = np.zeros(2 * (m - 1), dtype=np.float64)
    ubg = np.zeros(2 * (m - 1), dtype=np.float64)

    nlp = {"x": x_vars, "f": obj, "g": g_vec}
    opts = {
        "print_time": False,
        "ipopt.print_level": int(print_level),
        "ipopt.linear_solver": "mumps",
        "ipopt.max_iter": 200,
        "ipopt.tol": 1e-8,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    }
    solver = ca.nlpsol("ter27_sysid_mhe", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_opt = np.array(sol["x"].full()).ravel()

    mu_est = float(x_opt[0])
    aero_est = float(x_opt[1])
    izz_sc_est = float(x_opt[2])
    izz_est = izz_nom * izz_sc_est
    vy_est = x_opt[3 : 3 + m]
    wz_est = x_opt[3 + m :]

    # Compute final RMSE residuals on the binned series
    res_eval = fn_map(
        vx_b[None, :],
        vy_est[None, :],
        wz_est[None, :],
        delta_b[None, :],
        T4_b,
        fbrk_b[None, :],
        ax_lt_b[None, :],
        ay_lt_b[None, :],
        np.full((1, m), mu_est),
        np.full((1, m), aero_est),
    )
    ax_fit = np.array(res_eval[0].full()).ravel() / mass
    ay_fit = np.array(res_eval[1].full()).ravel() / mass

    rmse_ax = float(np.sqrt(np.mean((ax_fit - ax_meas) ** 2)))
    rmse_ay = float(np.sqrt(np.mean((ay_fit - ay_meas) ** 2)))
    rmse_wz = float(np.sqrt(np.mean((wz_est - wz_meas) ** 2)))

    return {
        "status": str(stats.get("return_status", "Unknown")),
        "iterations": int(stats.get("iter_count", -1)),
        "solve_time_s": round(time.perf_counter() - t0, 3),
        "n_raw_samples_200hz": n_raw,
        "n_mhe_knots": m,
        "estimated": {
            "mu_track_scale": round(mu_est, 5),
            "mu_track_eff": round(veh.env_mu_scale * mu_est, 5),
            "aero_scale": round(aero_est, 5),
            "cl_a_eff": round(p.cl_a * veh.env_aero_cl_scale * aero_est, 4),
            "cd_a_eff": round(p.cd_a * aero_est, 4),
            "I_zz_scale": round(izz_sc_est, 5),
            "I_zz_kgm2": round(izz_est, 4),
        },
        "residuals_rmse": {
            "ax_mps2": round(rmse_ax, 4),
            "ay_mps2": round(rmse_ay, 4),
            "wz_rads": round(rmse_wz, 5),
        },
    }


def run_validation_suite() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[SYSID] Building parametric CasADi vehicle model (FSG_REAL_TRACK)...", flush=True)
    veh = build_casadi_vehicle(grip_profile="FSG_REAL_TRACK")
    fn_sysid = build_parametric_sysid_function(veh)
    izz_nom = float(veh.params.izz_total)

    cases = [
        {
            "case_id": "cold_track_high_aero_heavy_yaw",
            "mu_scale_true": 0.88,
            "aero_scale_true": 1.12,
            "I_zz_scale_true": 1.10,
            "seed": 101,
        },
        {
            "case_id": "rubbered_in_low_aero_light_yaw",
            "mu_scale_true": 1.08,
            "aero_scale_true": 0.85,
            "I_zz_scale_true": 0.92,
            "seed": 202,
        },
        {
            "case_id": "damp_asphalt_nominal_aero_high_yaw",
            "mu_scale_true": 0.78,
            "aero_scale_true": 1.00,
            "I_zz_scale_true": 1.15,
            "seed": 303,
        },
    ]

    case_results: List[Dict[str, Any]] = []
    max_err_pct_all = 0.0

    with tempfile.TemporaryDirectory(prefix="ter27_sysid_") as tmp:
        tmp_dir = Path(tmp)
        for c in cases:
            cid = c["case_id"]
            csv_file = tmp_dir / f"{cid}_200hz.csv"
            print(
                f"[SYSID] Generating 200 Hz noisy telemetry for '{cid}' "
                f"(mu={c['mu_scale_true']:.2f}, aero={c['aero_scale_true']:.2f}, I_zz={c['I_zz_scale_true']:.2f}x)...",
                flush=True,
            )
            gen_meta = generate_synthetic_200hz_telemetry_csv(
                csv_path=csv_file,
                veh=veh,
                fn_sysid=fn_sysid,
                mu_true=c["mu_scale_true"],
                aero_true=c["aero_scale_true"],
                izz_scale_true=c["I_zz_scale_true"],
                duration_s=20.0,
                fs_hz=200.0,
                seed=c["seed"],
            )
            est = identify_parameters_casadi_mhe(csv_file, veh, fn_sysid, decimate=4, print_level=0)

            err_mu = abs(est["estimated"]["mu_track_scale"] - c["mu_scale_true"]) / c["mu_scale_true"] * 100.0
            err_aero = abs(est["estimated"]["aero_scale"] - c["aero_scale_true"]) / c["aero_scale_true"] * 100.0
            izz_true = izz_nom * c["I_zz_scale_true"]
            err_izz = abs(est["estimated"]["I_zz_kgm2"] - izz_true) / izz_true * 100.0
            case_max_err = max(err_mu, err_aero, err_izz)
            max_err_pct_all = max(max_err_pct_all, case_max_err)
            case_pass = case_max_err < 2.5

            print(
                f"[SYSID] '{cid}' ({est['status']}, {est['iterations']} iters, {est['solve_time_s']:.2f}s): "
                f"mu={est['estimated']['mu_track_scale']:.4f} (true {c['mu_scale_true']:.2f}, err={err_mu:.2f}%) | "
                f"aero={est['estimated']['aero_scale']:.4f} (true {c['aero_scale_true']:.2f}, err={err_aero:.2f}%) | "
                f"I_zz={est['estimated']['I_zz_kgm2']:.2f} (true {izz_true:.2f}, err={err_izz:.2f}%) | "
                f"PASS={case_pass}",
                flush=True,
            )

            case_results.append({
                "case_id": cid,
                "telemetry_meta": gen_meta,
                "ground_truth": {
                    "mu_track_scale": c["mu_scale_true"],
                    "aero_scale": c["aero_scale_true"],
                    "I_zz_scale": c["I_zz_scale_true"],
                    "I_zz_kgm2": round(izz_true, 4),
                },
                "identification": est,
                "error_pct": {
                    "mu_track": round(err_mu, 3),
                    "aero_scale": round(err_aero, 3),
                    "I_zz": round(err_izz, 3),
                    "max_pct": round(case_max_err, 3),
                },
                "pass_lt_2p5_pct": bool(case_pass),
            })

    overall_pass = all(r["pass_lt_2p5_pct"] for r in case_results)
    report: Dict[str, Any] = {
        "grip_profile_nominal": "FSG_REAL_TRACK",
        "nominal_params": {
            "env_mu_scale": veh.env_mu_scale,
            "cl_a_eff_nominal": round(veh.params.cl_a * veh.env_aero_cl_scale, 4),
            "cd_a_nominal": veh.params.cd_a,
            "I_zz_nominal_kgm2": izz_nom,
            "mass_kg": veh.params.mass,
        },
        "csv_format": "SimUI TelemetryCsvLogger.gd (200 Hz)",
        "estimator": "CasADi + IPOPT Multiple-Shooting MHE / NLS (ax, ay, wz residuals + exact trapezoidal yaw dynamics)",
        "requirement_max_error_pct": 2.5,
        "max_error_pct_across_all_cases": round(max_err_pct_all, 3),
        "overall_pass": bool(overall_pass),
        "cases": case_results,
        "total_runtime_s": round(time.perf_counter() - t0, 3),
    }

    rep_path = OUT_DIR / "sysid_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[SYSID] SUITE DONE: overall_pass={overall_pass} | max_err={max_err_pct_all:.3f}% (< 2.5%) | "
        f"saved {rep_path}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CasADi+IPOPT Sim-to-Real SysID for TeR27 4WD")
    parser.add_argument("--csv", type=str, default="", help="Optional path to a 200 Hz SimUI CSV telemetry file")
    parser.add_argument("--grip-profile", type=str, default="FSG_REAL_TRACK")
    args = parser.parse_args()

    if args.csv:
        veh = build_casadi_vehicle(grip_profile=args.grip_profile)
        fn_sysid = build_parametric_sysid_function(veh)
        res = identify_parameters_casadi_mhe(Path(args.csv), veh, fn_sysid, decimate=5, print_level=0)
        print(json.dumps(res, indent=2), flush=True)
    else:
        run_validation_suite()


if __name__ == "__main__":
    main()
