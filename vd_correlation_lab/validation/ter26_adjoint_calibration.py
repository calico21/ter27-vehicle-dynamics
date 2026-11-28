#!/usr/bin/env python3
"""
Empirical TeR26 (RWD) Adjoint System Identification & C2 Smoothing Audit
for the Differentiable 14-DOF CasADi Vehicle Model.

Capabilities:
1. Calibrated C2 smoothing audit (sabs, sclip, bump-stop transition, carcass
   relaxation & tire MF6.2 epsilons) using local symbolic DifferentiableVehicle14Dof
   instances (never touching outputs/casadi_cache/libter_diff_14dof.so), verifying
   smoothing error < 1e-4 and AD vs central FD gradient error < 1e-6.
2. Exact discrete reverse-mode adjoint gradients dL_telemetry/dphi using CasADi
   native reverse(1) VJPs across model parameters (aero ClA/CdA, Izz, springs,
   ARBs, steering compliance, tire relaxation sigma_y/sigma_x), actuator parameters
   (drivetrain efficiency, hydraulic brake gain, brake bias, coasting drag), and
   3D IMU mounting misalignment (yaw/pitch/roll) + static biases (ax_bias, ay_bias).
   Grip multipliers (lambda_muy, lambda_mux, COLD_TEST_OCT=0.47, FSG_RUBBERED=0.57)
   are strictly locked (zero grip inflation).
3. Strict train/validation split across the 5 real TeR26 SocketCAN track runs:
   - Train (3 runs):      ter26_track_1, ter26_track_3, ter26_track_4
   - Held-out Val (2 runs): ter26_track_2, ter26_track_5
4. Residual closure ablation (Baseline -> +Drivetrain/Brakes -> +Aero/Susp/Tires ->
   +IMU Misalignment/Bias) and parity vs TeR26HighFidelityCasADi reference model.
5. Structured 100% pure-ASCII JSON export with before/after time-series windows
   for 300 DPI plotting to:
   outputs/differentiable_deep_validation/ter26_adjoint_calibration_report.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.analysis.benchmarks.metrics_engine import compute_signal_metrics
from vd_correlation_lab.models.differentiable_casadi import (
    COLD_TEST_OCT,
    FSG_RUBBERED,
    DifferentiableVehicle14Dof,
    NU,
    NX,
    NTHETA,
    NY,
    OUTPUT_IDX,
    PARAM_IDX,
    STATE_IDX,
)
from vd_correlation_lab.models.differentiable_casadi.smooth_ops import (
    sabs,
    sclip,
    smax,
    smin,
    smooth_bumpstop,
    smooth_digressive_damper,
    smooth_piecewise_linear_1d,
    spos,
)
from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (
    TeR26HighFidelityCasADi,
    theta_dict_to_vec,
)
from vd_correlation_lab.validation.differentiable_ter26_correlation import (
    SYSID_PARAMS_JSON,
    WHEEL_RADIUS_M,
    _sanitize_json,
    load_real_ter26_track_runs,
)

DEFAULT_REPORT_PATH = (
    REPO_ROOT / "outputs" / "differentiable_deep_validation" / "ter26_adjoint_calibration_report.json"
)

TRAIN_RUN_IDS = ("ter26_track_1", "ter26_track_3", "ter26_track_4")
VAL_RUN_IDS = ("ter26_track_2", "ter26_track_5")

# Parameter layout for the 19-parameter extended calibration vector phi:
#   0..9:   Model theta parameters (aero, chassis, suspension, compliance, relaxation)
#   10..13: Actuator parameters (drivetrain gain, brake gain, brake bias, coast drag)
#   14..18: IMU observation parameters (yaw, pitch, roll misalignment, ax_bias, ay_bias)
THETA_OPT_NAMES: Tuple[str, ...] = (
    "cl_a",
    "cd_a",
    "i_zz",
    "k_spring_f",
    "k_spring_r",
    "k_arb_f",
    "k_arb_r",
    "k_steer_comp",
    "sigma_y",
    "sigma_x",
)
ACT_OPT_NAMES: Tuple[str, ...] = (
    "eta_drive",
    "k_brake_gain",
    "brake_bias_f",
    "f_coast_drag",
)
IMU_OPT_NAMES: Tuple[str, ...] = (
    "imu_yaw_rad",
    "imu_pitch_rad",
    "imu_roll_rad",
    "ax_bias",
    "ay_bias",
)
ALL_OPT_NAMES: Tuple[str, ...] = THETA_OPT_NAMES + ACT_OPT_NAMES + IMU_OPT_NAMES
N_THETA_OPT = len(THETA_OPT_NAMES)  # 10
N_ACT_OPT = len(ACT_OPT_NAMES)      # 4
N_IMU_OPT = len(IMU_OPT_NAMES)      # 5
N_PHI = len(ALL_OPT_NAMES)          # 19


def _default_act_params() -> np.ndarray:
    return np.array([1.0, 3300.0, 0.70, 140.0], dtype=np.float64)


def _default_imu_params() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def build_smooth_rwd_ctrl_function(wheel_r: float = WHEEL_RADIUS_M) -> ca.Function:
    """
    Build a C2-smooth CasADi SX function mapping:
      (x[32], drv[3] = [delta_rad, throttle, brake], phi_act[4]) -> u[9]
    so reverse(1) provides exact actuator & traction-control feedback VJPs.
    """
    x = ca.SX.sym("x", NX)
    drv = ca.SX.sym("drv", 3)
    phi_act = ca.SX.sym("phi_act", N_ACT_OPT)

    delta_rad = drv[0]
    thr = sclip(drv[1], 0.0, 1.0, 1e-4)
    brk = sclip(drv[2], 0.0, 1.0, 1e-4)

    eta_drive = phi_act[0]
    k_brk_gain = phi_act[1]
    brk_bias_f = sclip(phi_act[2], 0.45, 0.85, 1e-4)
    f_coast = spos(phi_act[3], 1e-2)

    vx = x[STATE_IDX["vx"]]
    om_rl = x[STATE_IDX["omega_RL"]]
    om_rr = x[STATE_IDX["omega_RR"]]

    v_safe = smax(vx, 0.5, 1e-3)
    om_w = v_safe / wheel_r
    t_pwr_tot = (80000.0 * 0.95 * eta_drive) / smax(om_w, 1.0, 1e-3)
    t_nom_tot = 2.0 * 21.0 * 11.5 * 0.95 * eta_drive
    t_drv_each = 0.5 * thr * smin(t_nom_tot, t_pwr_tot, 1.0)

    v_slip_den = smax(v_safe, 1.5, 1e-3)
    kap_rl = (om_rl * wheel_r - v_safe) / v_slip_den
    kap_rr = (om_rr * wheel_r - v_safe) / v_slip_den
    tc_rl = sclip(1.0 - spos(kap_rl - 0.06, 1e-3) / 0.08, 0.0, 1.0, 1e-3)
    tc_rr = sclip(1.0 - spos(kap_rr - 0.06, 1e-3) / 0.08, 0.0, 1.0, 1e-3)

    # Smooth lift-off coasting gate matching (thr < 0.02 and brk < 0.02 and v_safe > 1.5)
    g_thr = 0.5 * (1.0 - ca.tanh((thr - 0.02) / 0.01))
    g_brk = 0.5 * (1.0 - ca.tanh((brk - 0.02) / 0.01))
    g_spd = 0.5 * (1.0 + ca.tanh((v_safe - 1.5) / 0.25))
    coast_gate = g_thr * g_brk * g_spd

    t_brk_tot = (brk * k_brk_gain + coast_gate * f_coast) * wheel_r
    t_brk_f = 0.5 * brk_bias_f * t_brk_tot
    t_brk_r = 0.5 * (1.0 - brk_bias_f) * t_brk_tot

    u = ca.vertcat(
        delta_rad,
        0.0,
        0.0,
        t_drv_each * tc_rl,
        t_drv_each * tc_rr,
        t_brk_f,
        t_brk_f,
        t_brk_r,
        t_brk_r,
    )
    return ca.Function("smooth_rwd_ctrl", [x, drv, phi_act], [u], ["x", "drv", "phi_act"], ["u"])


def build_imu_obs_function(gravity: float = 9.81) -> ca.Function:
    """
    Build a C2-smooth CasADi SX observation function mapping:
      (x_next[32], y_step[36], phi_imu[5]) -> obs[6]
      obs = [wz, ay_imu, ax_imu, vx, wheel_rpm_rl, wheel_rpm_rr]

    Applies 3D IMU mounting misalignment (yaw, pitch, roll) + static bias (ax_bias, ay_bias)
    and body roll/pitch gravity projection consistent with ISO 8855.
    """
    x = ca.SX.sym("x", NX)
    y = ca.SX.sym("y", NY)
    phi_imu = ca.SX.sym("phi_imu", N_IMU_OPT)

    psi_m = phi_imu[0]    # IMU mounting yaw misalignment [rad]
    theta_m = phi_imu[1]  # IMU mounting pitch misalignment [rad]
    phi_m = phi_imu[2]    # IMU mounting roll misalignment [rad]
    ax_b0 = phi_imu[3]    # Static ax bias [m/s^2]
    ay_b0 = phi_imu[4]    # Static ay bias [m/s^2]

    ax_body = y[OUTPUT_IDX["ax"]]
    ay_body = y[OUTPUT_IDX["ay"]]
    az_imu = y[OUTPUT_IDX["az"]]  # ~ +g in level driving

    # Small-angle 3D mounting rotation R_imu(psi_m, theta_m, phi_m) on specific force
    cy, sy = ca.cos(psi_m), ca.sin(psi_m)
    cp, sp = ca.cos(theta_m), ca.sin(theta_m)
    cr, sr = ca.cos(phi_m), ca.sin(phi_m)

    # First apply pitch & roll mounting tilt (gravity cross-coupling from az_imu)
    ax_tilt = cp * ax_body + sp * az_imu
    ay_tilt = cr * ay_body - sr * az_imu
    # Then apply residual yaw mounting misalignment in horizontal plane
    ax_obs = cy * ax_tilt - sy * ay_tilt + ax_b0
    ay_obs = sy * ax_tilt + cy * ay_tilt + ay_b0

    rpm_scale = 60.0 / (2.0 * math.pi)
    obs = ca.vertcat(
        x[STATE_IDX["wz"]],
        ay_obs,
        ax_obs,
        x[STATE_IDX["vx"]],
        x[STATE_IDX["omega_RL"]] * rpm_scale,
        x[STATE_IDX["omega_RR"]] * rpm_scale,
    )
    return ca.Function("imu_obs_map", [x, y, phi_imu], [obs], ["x", "y", "phi_imu"], ["obs"])


# ---------------------------------------------------------------------------
# 1. Calibrated C2 Smoothing Audit (Local Symbolic Instances Only)
# ---------------------------------------------------------------------------

def run_c2_smoothing_audit(
    model_nominal: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
) -> Dict[str, Any]:
    """
    Audit C2 smoothing primitives and full 14-DOF symbolic model smoothing
    sensitivity (without overwriting outputs/casadi_cache/libter_diff_14dof.so):
      - Verify primitive smoothing error < 1e-4 at calibrated epsilons.
      - Verify 14-DOF state/output smoothing sensitivity across epsilon tiers.
      - Verify exact reverse-mode AD vs central FD gradient error < 1e-6.
    """
    t0 = time.perf_counter()
    xs = np.linspace(-2.0, 2.0, 4001)

    # A. Primitive smoothing error audit
    prim_audit: Dict[str, Any] = {}
    for eps in (1e-3, 1e-4, 1e-5):
        err_sabs = float(np.max(np.abs(sabs(xs, eps=eps) - np.abs(xs))))
        err_sclip = float(np.max(np.abs(sclip(xs, -0.5, 0.5, eps=eps) - np.clip(xs, -0.5, 0.5))))
        err_spos = float(np.max(np.abs(spos(xs, eps=eps) - np.maximum(xs, 0.0))))
        prim_audit[f"eps_{eps:.0e}"] = {
            "sabs_max_err": err_sabs,
            "sclip_max_err": err_sclip,
            "spos_max_err": err_spos,
            "all_lt_1e4": bool(max(err_sabs, err_sclip, err_spos) < 1e-4),
        }

    # Bump-stop transition audit (TeR26 front bumpstop: start=0.025m, k1=120k, k2=2M)
    dz_bs = np.linspace(-0.02, 0.045, 2001)
    ex_exact = np.maximum(dz_bs - 0.025, 0.0)
    f_bs_exact = 120000.0 * ex_exact + 2000000.0 * (ex_exact ** 2)
    f_bs_scale = max(float(np.max(f_bs_exact)), 1.0)
    bs_sweep: Dict[str, Any] = {}
    for eps_bs in (1e-4, 1e-5, 1e-6):
        f_bs_sm = smooth_bumpstop(dz_bs, 0.025, 120000.0, 2000000.0, eps=eps_bs)
        max_abs_n = float(np.max(np.abs(f_bs_sm - f_bs_exact)))
        # Relative error normalized to peak bumpstop force
        rel_err = max_abs_n / f_bs_scale
        # Also verify displacement smoothing error itself is < 1e-4 m
        dz_err_m = float(np.max(np.abs(spos(dz_bs - 0.025, eps=eps_bs) - ex_exact)))
        bs_sweep[f"eps_{eps_bs:.0e}"] = {
            "disp_max_err_m": dz_err_m,
            "force_max_abs_err_n": max_abs_n,
            "force_rel_err": rel_err,
            "disp_lt_1e4": bool(dz_err_m < 1e-4),
        }

    # Digressive damper transition audit
    vd = np.linspace(-0.3, 0.3, 2001)
    f_dmp_exact = np.where(
        vd >= 0.0,
        np.where(vd <= 0.05, 2000.0 * vd, 2000.0 * 0.05 + 1000.0 * (vd - 0.05)),
        np.where(vd >= -0.05, 3000.0 * vd, -3000.0 * 0.05 + 1500.0 * (vd + 0.05)),
    )
    f_dmp_sm = smooth_digressive_damper(vd, 0.05, 2000.0, 1000.0, 3000.0, 1500.0, eps_v=1e-4)
    dmp_rel_err = float(np.max(np.abs(f_dmp_sm - f_dmp_exact))) / max(float(np.max(np.abs(f_dmp_exact))), 1.0)

    # B. Local symbolic 14-DOF epsilon sensitivity sweep (no .so modification)
    # Compare calibrated tire smoothing vs 0.1x tightened smoothing on a 1.0s dynamic slalom
    m_tight = DifferentiableVehicle14Dof(
        vehicle_id="ter26_rwd",
        suspension_mode="3d_double_wishbone_lut",
        tire_compound="R20",
        use_pinn=True,
        dt=model_nominal.dt,
        rk4_substeps=model_nominal.rk4_substeps,
    )
    # Tighten tire smoothing epsilons by 10x on local symbolic instance and rebuild graph
    for k_sm in list(m_tight.tire.smoothing.keys()):
        m_tight.tire.smoothing[k_sm] = float(m_tight.tire.smoothing[k_sm]) * 0.1
    m_tight._build_casadi_functions()

    m_coarse = DifferentiableVehicle14Dof(
        vehicle_id="ter26_rwd",
        suspension_mode="3d_double_wishbone_lut",
        tire_compound="R20",
        use_pinn=True,
        dt=model_nominal.dt,
        rk4_substeps=model_nominal.rk4_substeps,
    )
    for k_sm in list(m_coarse.tire.smoothing.keys()):
        m_coarse.tire.smoothing[k_sm] = float(m_coarse.tire.smoothing[k_sm]) * 10.0
    m_coarse._build_casadi_functions()

    th = model_nominal.default_theta()
    w = model_nominal.default_disturbance(t_tire_c=55.0, t_motor_c=55.0)
    x_nom = model_nominal.default_state(vx=15.0, vy=0.0, wz=0.0)
    x_tgh = x_nom.copy()
    x_crs = x_nom.copy()
    n_sw = 100  # 1.0 s at 100 Hz
    wz_nom_arr, wz_tgh_arr, wz_crs_arr = [], [], []
    ay_nom_arr, ay_tgh_arr, ay_crs_arr = [], [], []
    ax_nom_arr, ax_tgh_arr, ax_crs_arr = [], [], []

    for k in range(n_sw):
        tk = k * model_nominal.dt
        delta_k = 0.085 * math.sin(2.0 * math.pi * 0.85 * tk)
        thr_k = 0.35 if tk < 0.55 else 0.0
        brk_k = 0.0 if tk < 0.60 else 0.40
        u_k = model_nominal.inputs_from_driver(delta_cmd=delta_k, throttle=thr_k, brake=brk_k)
        x_nom, y_nom = model_nominal.step(x_nom, u_k, th, w)
        x_tgh, y_tgh = m_tight.step(x_tgh, u_k, th, w)
        x_crs, y_crs = m_coarse.step(x_crs, u_k, th, w)
        wz_nom_arr.append(x_nom[STATE_IDX["wz"]])
        wz_tgh_arr.append(x_tgh[STATE_IDX["wz"]])
        wz_crs_arr.append(x_crs[STATE_IDX["wz"]])
        ay_nom_arr.append(y_nom[OUTPUT_IDX["ay"]])
        ay_tgh_arr.append(y_tgh[OUTPUT_IDX["ay"]])
        ay_crs_arr.append(y_crs[OUTPUT_IDX["ay"]])
        ax_nom_arr.append(y_nom[OUTPUT_IDX["ax"]])
        ax_tgh_arr.append(y_tgh[OUTPUT_IDX["ax"]])
        ax_crs_arr.append(y_crs[OUTPUT_IDX["ax"]])

    wz_nom_np, wz_tgh_np, wz_crs_np = np.array(wz_nom_arr), np.array(wz_tgh_arr), np.array(wz_crs_arr)
    ay_nom_np, ay_tgh_np, ay_crs_np = np.array(ay_nom_arr), np.array(ay_tgh_arr), np.array(ay_crs_arr)
    ax_nom_np, ax_tgh_np, ax_crs_np = np.array(ax_nom_arr), np.array(ax_tgh_arr), np.array(ax_crs_arr)

    cal_vs_tight_wz_rel = float(np.max(np.abs(wz_nom_np - wz_tgh_np)) / max(np.max(np.abs(wz_tgh_np)), 1e-6))
    cal_vs_tight_ay_rel = float(np.max(np.abs(ay_nom_np - ay_tgh_np)) / max(np.max(np.abs(ay_tgh_np)), 1e-6))
    cal_vs_tight_ax_rel = float(np.max(np.abs(ax_nom_np - ax_tgh_np)) / max(np.max(np.abs(ax_tgh_np)), 1e-6))
    coarse_vs_tight_ay_rel = float(np.max(np.abs(ay_crs_np - ay_tgh_np)) / max(np.max(np.abs(ay_tgh_np)), 1e-6))

    # C. Full 19-parameter Reverse-Mode Adjoint vs Central FD verification
    x0_chk = model_nominal.default_state(vx=14.0, vy=0.1, wz=0.15)
    n_chk = 30
    drv_seq = np.zeros((n_chk, 3), dtype=np.float64)
    for k in range(n_chk):
        tk = k * model_nominal.dt
        drv_seq[k, 0] = 0.08 * math.sin(2.0 * math.pi * 0.9 * tk) + 0.02
        drv_seq[k, 1] = 0.45 if k < 15 else 0.0
        drv_seq[k, 2] = 0.0 if k < 15 else 0.35

    phi_nom = pack_phi(model_nominal.default_theta(), _default_act_params(), _default_imu_params())
    # Generate target observations at perturbed phi_target so gradients are non-zero everywhere
    phi_tgt = phi_nom.copy()
    phi_tgt[0] *= 0.85   # cl_a
    phi_tgt[1] *= 1.15   # cd_a
    phi_tgt[2] *= 1.12   # i_zz
    phi_tgt[8] = 0.16    # sigma_y
    phi_tgt[10] = 1.08   # eta_drive
    phi_tgt[11] = 2800.0 # k_brake_gain
    phi_tgt[14] = 0.02   # imu_yaw
    phi_tgt[15] = 0.015  # imu_pitch
    phi_tgt[17] = 0.20   # ax_bias
    phi_tgt[18] = -0.15  # ay_bias

    w_eval = model_nominal.default_disturbance(t_tire_c=55.0, t_motor_c=55.0)
    obs_tgt = _rollout_window_obs(model_nominal, fn_ctrl, fn_obs, x0_chk, drv_seq, phi_tgt, w_eval)

    # Evaluate adjoint at an intermediate point with non-zero angles/biases
    phi_eval = phi_nom.copy()
    phi_eval[8] = 0.10
    phi_eval[14] = 0.01
    phi_eval[15] = 0.01
    phi_eval[16] = -0.01
    phi_eval[17] = 0.05
    phi_eval[18] = -0.05

    fn_step_rev = model_nominal.fn_step.reverse(1)
    fn_ctrl_rev = fn_ctrl.reverse(1)
    fn_obs_rev = fn_obs.reverse(1)
    obs_weights = np.array([1.0 / 0.4**2, 1.0 / 5.0**2, 1.0 / 2.5**2, 1.0 / 3.0**2, 1.0 / 150.0**2, 1.0 / 150.0**2])

    loss_ad, grad_ad = _window_loss_and_adjoint(
        model_nominal, fn_ctrl, fn_obs, fn_step_rev, fn_ctrl_rev, fn_obs_rev,
        x0_chk, drv_seq, obs_tgt, phi_eval, w_eval, obs_weights, compute_grad=True,
    )

    per_param_fd: Dict[str, Any] = {}
    max_rel_err = 0.0
    for j, name in enumerate(ALL_OPT_NAMES):
        eps_fd = max(abs(float(phi_eval[j])) * 1e-5, 1e-6)
        pp = phi_eval.copy()
        pm = phi_eval.copy()
        pp[j] += eps_fd
        pm[j] -= eps_fd
        lp, _ = _window_loss_and_adjoint(
            model_nominal, fn_ctrl, fn_obs, fn_step_rev, fn_ctrl_rev, fn_obs_rev,
            x0_chk, drv_seq, obs_tgt, pp, w_eval, obs_weights, compute_grad=False,
        )
        lm, _ = _window_loss_and_adjoint(
            model_nominal, fn_ctrl, fn_obs, fn_step_rev, fn_ctrl_rev, fn_obs_rev,
            x0_chk, drv_seq, obs_tgt, pm, w_eval, obs_weights, compute_grad=False,
        )
        g_fd = (lp - lm) / (2.0 * eps_fd)
        g_ad = float(grad_ad[j])
        denom = max(abs(g_fd), abs(g_ad), 1e-10)
        rel_err = abs(g_ad - g_fd) / denom
        max_rel_err = max(max_rel_err, rel_err)
        per_param_fd[name] = {
            "ad_grad": g_ad,
            "fd_grad": float(g_fd),
            "abs_err": abs(g_ad - g_fd),
            "rel_err": float(rel_err),
            "passed_lt_1e6": bool(rel_err < 1e-6),
        }

    return {
        "primitives": prim_audit,
        "bumpstop_transition": bs_sweep,
        "digressive_damper_rel_err": dmp_rel_err,
        "model_14dof_smoothing_sensitivity": {
            "calibrated_vs_0p1x_tight_wz_rel_err": cal_vs_tight_wz_rel,
            "calibrated_vs_0p1x_tight_ay_rel_err": cal_vs_tight_ay_rel,
            "calibrated_vs_0p1x_tight_ax_rel_err": cal_vs_tight_ax_rel,
            "coarse_10x_vs_0p1x_tight_ay_rel_err": coarse_vs_tight_ay_rel,
            "smoothing_error_lt_1e4": bool(
                max(cal_vs_tight_wz_rel, cal_vs_tight_ay_rel, cal_vs_tight_ax_rel) < 1e-4
            ),
            "calibrated_smoothing_constants": dict(model_nominal.tire.smoothing),
        },
        "adjoint_vs_central_fd_19params": {
            "n_steps": n_chk,
            "eval_loss": float(loss_ad),
            "max_rel_err": float(max_rel_err),
            "all_passed_lt_1e6": bool(max_rel_err < 1e-6),
            "per_parameter": per_param_fd,
        },
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 2. Parameter Packing / Unpacking & Exact Reverse-Mode Adjoint Rollout
# ---------------------------------------------------------------------------

def pack_phi(theta_base: np.ndarray, act: np.ndarray, imu: np.ndarray) -> np.ndarray:
    th_sub = np.array([float(theta_base[PARAM_IDX[k]]) for k in THETA_OPT_NAMES], dtype=np.float64)
    return np.concatenate([th_sub, np.asarray(act, dtype=np.float64), np.asarray(imu, dtype=np.float64)])


def unpack_phi(phi: np.ndarray, theta_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    th = theta_base.copy()
    for j, name in enumerate(THETA_OPT_NAMES):
        th[PARAM_IDX[name]] = float(phi[j])
    act = np.asarray(phi[N_THETA_OPT : N_THETA_OPT + N_ACT_OPT], dtype=np.float64)
    imu = np.asarray(phi[N_THETA_OPT + N_ACT_OPT :], dtype=np.float64)
    return th, act, imu


def _rollout_window_obs(
    model: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
    x0: np.ndarray,
    drv_seq: np.ndarray,
    phi: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    th, act, imu = unpack_phi(phi, model.default_theta())
    n = len(drv_seq)
    obs_out = np.zeros((n, 6), dtype=np.float64)
    x = x0.copy()
    for k in range(n):
        u_k = np.asarray(fn_ctrl(x, drv_seq[k], act), dtype=np.float64).ravel()
        x_next, y_k = model.step(x, u_k, th, w)
        obs_k = np.asarray(fn_obs(x_next, y_k, imu), dtype=np.float64).ravel()
        obs_out[k] = obs_k
        x = x_next
    return obs_out


def _window_loss_and_adjoint(
    model: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
    fn_step_rev: ca.Function,
    fn_ctrl_rev: ca.Function,
    fn_obs_rev: ca.Function,
    x0: np.ndarray,
    drv_seq: np.ndarray,
    obs_target: np.ndarray,
    phi: np.ndarray,
    w: np.ndarray,
    obs_weights: np.ndarray,
    compute_grad: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Forward simulation + exact reverse-mode CasADi VJP adjoint pass over one window.
    Returns (loss, grad_phi[19]).
    """
    theta_base = model.default_theta()
    th, act, imu = unpack_phi(phi, theta_base)
    n = len(drv_seq)

    xs = np.zeros((n + 1, NX), dtype=np.float64)
    us = np.zeros((n, NU), dtype=np.float64)
    ys = np.zeros((n, NY), dtype=np.float64)
    obss = np.zeros((n, 6), dtype=np.float64)

    x = x0.copy()
    xs[0] = x
    loss = 0.0
    inv_n = 1.0 / float(n)

    for k in range(n):
        u_k = np.asarray(fn_ctrl(x, drv_seq[k], act), dtype=np.float64).ravel()
        x_next, y_k = model.step(x, u_k, th, w)
        obs_k = np.asarray(fn_obs(x_next, y_k, imu), dtype=np.float64).ravel()
        us[k] = u_k
        xs[k + 1] = x_next
        ys[k] = y_k
        obss[k] = obs_k
        diff = obs_k - obs_target[k]
        loss += 0.5 * float(np.dot(obs_weights, diff * diff)) * inv_n
        x = x_next

    if not compute_grad:
        return float(loss), np.zeros(N_PHI, dtype=np.float64)

    lam_x = np.zeros(NX, dtype=np.float64)
    grad_th_full = np.zeros(NTHETA, dtype=np.float64)
    grad_act = np.zeros(N_ACT_OPT, dtype=np.float64)
    grad_imu = np.zeros(N_IMU_OPT, dtype=np.float64)

    for k in range(n - 1, -1, -1):
        dL_dobs = obs_weights * (obss[k] - obs_target[k]) * inv_n
        adj_x_obs, adj_y_obs, adj_imu_k = fn_obs_rev(xs[k + 1], ys[k], imu, obss[k], dL_dobs)
        lam_x_tot = lam_x + np.asarray(adj_x_obs, dtype=np.float64).ravel()
        adj_y_np = np.asarray(adj_y_obs, dtype=np.float64).ravel()
        grad_imu += np.asarray(adj_imu_k, dtype=np.float64).ravel()

        adj_x_st, adj_u_st, adj_th_k, _ = fn_step_rev(
            xs[k], us[k], th, w, xs[k + 1], ys[k], lam_x_tot, adj_y_np
        )
        adj_u_np = np.asarray(adj_u_st, dtype=np.float64).ravel()
        grad_th_full += np.asarray(adj_th_k, dtype=np.float64).ravel()

        adj_x_ct, _, adj_act_k = fn_ctrl_rev(xs[k], drv_seq[k], act, us[k], adj_u_np)
        grad_act += np.asarray(adj_act_k, dtype=np.float64).ravel()
        lam_x = np.asarray(adj_x_st, dtype=np.float64).ravel() + np.asarray(adj_x_ct, dtype=np.float64).ravel()

    grad_th_sub = np.array([grad_th_full[PARAM_IDX[name]] for name in THETA_OPT_NAMES], dtype=np.float64)
    grad_phi = np.concatenate([grad_th_sub, grad_act, grad_imu])
    return float(loss), grad_phi


# ---------------------------------------------------------------------------
# 3. Training Window Extraction (Strictly from Train Runs Only)
# ---------------------------------------------------------------------------

def select_training_windows(
    train_runs: List[Dict[str, Any]],
    model: DifferentiableVehicle14Dof,
    win_len: int = 200,
    n_per_run: int = 6,
) -> List[Dict[str, Any]]:
    """
    Select stratified representative 2.0s windows from the 3 training runs ONLY:
    2 high-lateral cornering/slalom, 2 longitudinal accel/brake/coast, and
    2 representative mixed-driving windows matching the full-run operating distribution.
    """
    windows: List[Dict[str, Any]] = []
    for run in train_runs:
        n = len(run["t_s"])
        active = run["is_active"]
        lat_cands: List[Tuple[float, int]] = []
        long_cands: List[Tuple[float, int]] = []
        mix_cands: List[Tuple[float, int]] = []
        step = max(win_len // 2, 50)
        for i0 in range(0, n - win_len - 1, step):
            sl = slice(i0, i0 + win_len)
            if not bool(np.all(active[sl])):
                continue
            vx_sl = run["vx_ms"][sl]
            if float(np.min(vx_sl)) < 4.0:
                continue
            ay_sl = run["ay_ms2"][sl]
            ax_sl = run["ax_ms2"][sl]
            wz_sl = run["wz_rads"][sl]
            thr_sl = run["throttle"][sl]
            brk_sl = run["brake"][sl]

            lat_sc = float(np.std(ay_sl)) / 5.0 + float(np.std(wz_sl)) / 0.4
            long_sc = (
                float(np.std(ax_sl)) / 2.5
                + 1.0 * float(np.mean(thr_sl > 0.15))
                + 1.5 * float(np.mean(brk_sl > 0.10))
                + 1.0 * float(np.mean((thr_sl < 0.05) & (brk_sl < 0.02)))
            )
            # Balanced mixed score favors windows with near-zero mean ax (matched accel+brake+coast)
            mix_sc = 0.7 * lat_sc + 0.7 * long_sc - 0.5 * abs(float(np.mean(ax_sl)))
            lat_cands.append((lat_sc, i0))
            long_cands.append((long_sc - 0.3 * float(np.mean(np.abs(wz_sl))), i0))
            mix_cands.append((mix_sc, i0))

        lat_cands.sort(key=lambda item: item[0], reverse=True)
        long_cands.sort(key=lambda item: item[0], reverse=True)
        mix_cands.sort(key=lambda item: item[0], reverse=True)

        picked_starts: List[int] = []
        quotas = [("lateral", lat_cands, 2), ("longitudinal", long_cands, 2), ("mixed", mix_cands, max(n_per_run - 4, 2))]
        for kind, cands, quota in quotas:
            taken = 0
            for score, i0 in cands:
                if any(abs(i0 - s) < win_len for s in picked_starts):
                    continue
                picked_starts.append(i0)
                sl = slice(i0, i0 + win_len)
                x0 = model.default_state(
                    vx=max(float(run["vx_ms"][i0]), 1.0),
                    vy=0.0,
                    wz=float(run["wz_rads"][i0]),
                )
                x0[STATE_IDX["omega_RL"]] = float(run["wheel_rpm_rl"][i0]) * (2.0 * math.pi / 60.0)
                x0[STATE_IDX["omega_RR"]] = float(run["wheel_rpm_rr"][i0]) * (2.0 * math.pi / 60.0)
                drv_seq = np.column_stack([
                    run["delta_rad"][sl],
                    run["throttle"][sl],
                    run["brake"][sl],
                ])
                obs_tgt = np.column_stack([
                    run["wz_rads"][sl],
                    run["ay_ms2"][sl],
                    run["ax_ms2"][sl],
                    run["vx_ms"][sl],
                    run["wheel_rpm_rl"][sl],
                    run["wheel_rpm_rr"][sl],
                ])
                windows.append({
                    "run_id": run["run_id"],
                    "i0": int(i0),
                    "t_start_s": round(float(run["t_s"][i0]), 2),
                    "t_end_s": round(float(run["t_s"][i0 + win_len - 1]), 2),
                    "kind": kind,
                    "score": round(score, 3),
                    "x0": x0,
                    "drv_seq": drv_seq,
                    "obs_tgt": obs_tgt,
                })
                taken += 1
                if taken >= quota:
                    break
    return windows


# ---------------------------------------------------------------------------
# 4. Exact Reverse-Mode Adjoint L-BFGS-B Calibration on Real Track Windows
# ---------------------------------------------------------------------------

def run_adjoint_lbfgsb_calibration(
    model: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
    train_windows: List[Dict[str, Any]],
    maxiter: int = 45,
) -> Dict[str, Any]:
    """
    Identify the 19 physical/actuator/IMU parameters on real TeR26 training windows
    using ONLY exact reverse-mode CasADi adjoint gradients + L-BFGS-B.
    """
    t0 = time.perf_counter()
    theta_nom = model.default_theta()
    phi_nom = pack_phi(theta_nom, _default_act_params(), _default_imu_params())

    # Physical bounds (strictly grounded; lambda_muy and lambda_mux NOT in phi)
    ks_f = float(theta_nom[PARAM_IDX["k_spring_f"]])
    ks_r = float(theta_nom[PARAM_IDX["k_spring_r"]])
    ka_f = float(theta_nom[PARAM_IDX["k_arb_f"]])
    ka_r = float(theta_nom[PARAM_IDX["k_arb_r"]])

    raw_bounds: List[Tuple[float, float]] = [
        (2.20, 5.50),              # cl_a [m^2] (real track aero package range)
        (0.55, 2.00),              # cd_a [m^2]
        (45.0, 115.0),             # i_zz [kg*m^2]
        (0.65 * ks_f, 1.50 * ks_f),# k_spring_f [N/m]
        (0.65 * ks_r, 1.50 * ks_r),# k_spring_r [N/m]
        (0.45 * ka_f, 2.00 * ka_f),# k_arb_f [N/m]
        (0.45 * ka_r, 2.00 * ka_r),# k_arb_r [N/m]
        (1800.0, 25000.0),         # k_steer_comp [Nm/rad]
        (0.04, 0.55),              # sigma_y [m]
        (0.04, 0.45),              # sigma_x [m]
        (0.80, 1.12),              # eta_drive [-] (physical torque-constant/efficiency envelope)
        (1800.0, 4000.0),          # k_brake_gain [N]
        (0.52, 0.80),              # brake_bias_f [-]
        (0.0, 240.0),              # f_coast_drag [N]
        (-0.10, 0.10),             # imu_yaw_rad [rad] (+/- 5.7 deg)
        (-0.05, 0.05),             # imu_pitch_rad [rad] (+/- 2.9 deg)
        (-0.05, 0.05),             # imu_roll_rad [rad] (+/- 2.9 deg)
        (-0.45, 0.45),             # ax_bias [m/s^2]
        (-0.45, 0.45),             # ay_bias [m/s^2]
    ]

    # Characteristic scales for normalized decision variable z = (phi - phi_nom) / scale
    scales = np.array([
        1.5,    # cl_a
        0.5,    # cd_a
        20.0,   # i_zz
        0.3 * ks_f,
        0.3 * ks_r,
        0.4 * ka_f,
        0.4 * ka_r,
        5000.0, # k_steer_comp
        0.15,   # sigma_y
        0.10,   # sigma_x
        0.10,   # eta_drive
        600.0,  # k_brake_gain
        0.10,   # brake_bias_f
        80.0,   # f_coast_drag
        0.04,   # imu_yaw_rad
        0.02,   # imu_pitch_rad
        0.02,   # imu_roll_rad
        0.20,   # ax_bias
        0.20,   # ay_bias
    ], dtype=np.float64)

    z_bounds = [
        ((lo - float(phi_nom[j])) / float(scales[j]), (hi - float(phi_nom[j])) / float(scales[j]))
        for j, (lo, hi) in enumerate(raw_bounds)
    ]

    # Mild Tikhonov regularization around nominal physical prior (prevents overfitting & pitch/ax_bias collinearity)
    reg_weights = np.array([
        2e-3, 2e-3, 2e-3, 3e-3, 3e-3, 3e-3, 3e-3, 2e-3, 1e-3, 2e-3,
        4e-3, 2e-3, 3e-3, 1e-3,
        6e-3, 1.5e-2, 1.5e-2, 1.5e-2, 1.5e-2,
    ], dtype=np.float64)

    # Channel weights: balanced across ALL 6 channels (wz, ay, ax, vx, wheel_rpm_rl/rr)
    obs_weights = np.array([
        1.40 / (0.35 ** 2),    # wz [rad/s]
        1.20 / (4.50 ** 2),    # ay [m/s^2]
        1.40 / (2.00 ** 2),    # ax [m/s^2]
        1.20 / (1.80 ** 2),    # vx [m/s]
        0.90 / (85.0 ** 2),    # wheel_rpm_rl
        0.90 / (85.0 ** 2),    # wheel_rpm_rr
    ], dtype=np.float64)

    fn_step_rev = model.fn_step.reverse(1)
    fn_ctrl_rev = fn_ctrl.reverse(1)
    fn_obs_rev = fn_obs.reverse(1)
    w_eval = model.default_disturbance(t_tire_c=55.0, t_motor_c=55.0)
    n_win = len(train_windows)
    inv_nw = 1.0 / float(n_win)

    history: List[Dict[str, Any]] = []

    def _obj_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        phi = phi_nom + z * scales
        tot_loss = 0.0
        tot_grad_phi = np.zeros(N_PHI, dtype=np.float64)
        for win in train_windows:
            lw, gw = _window_loss_and_adjoint(
                model, fn_ctrl, fn_obs, fn_step_rev, fn_ctrl_rev, fn_obs_rev,
                win["x0"], win["drv_seq"], win["obs_tgt"], phi, w_eval, obs_weights, compute_grad=True,
            )
            tot_loss += lw * inv_nw
            tot_grad_phi += gw * inv_nw

        reg_loss = 0.5 * float(np.sum(reg_weights * (z ** 2)))
        total = tot_loss + reg_loss
        grad_z = tot_grad_phi * scales + reg_weights * z

        history.append({
            "eval": len(history),
            "data_loss": float(tot_loss),
            "reg_loss": float(reg_loss),
            "total_loss": float(total),
            "grad_norm": float(np.linalg.norm(grad_z)),
        })
        if len(history) == 1 or len(history) % 5 == 0:
            print(
                f"  [Adjoint L-BFGS-B] eval={len(history):3d} | data_loss={tot_loss:.5f} | "
                f"total={total:.5f} | |g|={np.linalg.norm(grad_z):.3e}",
                flush=True,
            )
        return float(total), grad_z

    z0 = np.zeros(N_PHI, dtype=np.float64)
    res = minimize(
        _obj_and_grad,
        z0,
        method="L-BFGS-B",
        jac=True,
        bounds=z_bounds,
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )

    phi_opt = phi_nom + res.x * scales
    param_summary: Dict[str, Any] = {}
    for j, name in enumerate(ALL_OPT_NAMES):
        nom_v = float(phi_nom[j])
        opt_v = float(phi_opt[j])
        lo, hi = raw_bounds[j]
        delta_pct = (100.0 * (opt_v - nom_v) / abs(nom_v)) if abs(nom_v) > 1e-9 else None
        param_summary[name] = {
            "nominal": nom_v,
            "calibrated": opt_v,
            "delta": opt_v - nom_v,
            "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
            "bounds": [float(lo), float(hi)],
            "at_bound": bool(abs(opt_v - lo) < 1e-6 or abs(opt_v - hi) < 1e-6),
        }

    return {
        "method": "L-BFGS-B with exact reverse-mode CasADi VJP adjoint gradients (fn.reverse(1))",
        "n_train_windows": n_win,
        "window_len_samples": int(len(train_windows[0]["drv_seq"])),
        "train_run_ids": list(TRAIN_RUN_IDS),
        "val_run_ids": list(VAL_RUN_IDS),
        "grip_locked": {
            "lambda_muy": float(theta_nom[PARAM_IDX["lambda_muy"]]),
            "lambda_mux": float(theta_nom[PARAM_IDX["lambda_mux"]]),
            "COLD_TEST_OCT": float(COLD_TEST_OCT),
            "FSG_RUBBERED": float(FSG_RUBBERED),
        },
        "converged": bool(res.success),
        "message": str(res.message),
        "n_iterations": int(res.nit),
        "n_evaluations": int(res.nfev),
        "initial_data_loss": float(history[0]["data_loss"]),
        "final_data_loss": float(history[-1]["data_loss"]),
        "loss_reduction_pct": round(
            100.0 * (1.0 - history[-1]["data_loss"] / max(history[0]["data_loss"], 1e-12)), 3
        ),
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "phi_nominal": phi_nom.tolist(),
        "phi_calibrated": phi_opt.tolist(),
        "parameters": param_summary,
        "convergence_history": history,
    }


# ---------------------------------------------------------------------------
# 5. Full 5-Run Evaluation, Ablation Waterfall & Ref14 Parity
# ---------------------------------------------------------------------------

def simulate_calibrated_on_run(
    model: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
    run: Dict[str, Any],
    phi: np.ndarray,
    horizon_reset_s: float = 10.0,
    vx_observer_tau_s: float = 2.5,
) -> Dict[str, np.ndarray]:
    """Run DifferentiableVehicle14Dof with calibrated (theta, act, imu) on a full track run."""
    t_s = run["t_s"]
    vx_r = run["vx_ms"]
    wz_r = run["wz_rads"]
    delta_r = run["delta_rad"]
    thr_r = run["throttle"]
    brk_r = run["brake"]
    n = len(t_s)
    dt = float(model.dt)
    wheel_r = float(model.cfg.wheel_radius)
    th, act, imu = unpack_phi(phi, model.default_theta())
    w = model.default_disturbance(t_tire_c=55.0, t_motor_c=55.0)

    vx_s = np.zeros(n, dtype=np.float64)
    ax_s = np.zeros(n, dtype=np.float64)
    ay_s = np.zeros(n, dtype=np.float64)
    wz_s = np.zeros(n, dtype=np.float64)
    rl_s = np.zeros(n, dtype=np.float64)
    rr_s = np.zeros(n, dtype=np.float64)

    x = model.default_state(vx=max(float(vx_r[0]), 0.5), vy=0.0, wz=float(wz_r[0]))
    last_reset = float(t_s[0])
    alpha_h = (1.0 - math.exp(-dt / vx_observer_tau_s)) if vx_observer_tau_s > 0.0 else 0.0

    for k in range(n):
        if vx_r[k] < 2.0 or (horizon_reset_s > 0.0 and (float(t_s[k]) - last_reset) >= horizon_reset_s):
            x = model.default_state(vx=max(float(vx_r[k]), 0.5), vy=0.0, wz=float(wz_r[k]))
            last_reset = float(t_s[k])

        drv_k = np.array([float(delta_r[k]), float(thr_r[k]), float(brk_r[k])], dtype=np.float64)
        u_k = np.asarray(fn_ctrl(x, drv_k, act), dtype=np.float64).ravel()
        x_next, y_next = model.step(x, u_k, th, w)
        obs_k = np.asarray(fn_obs(x_next, y_next, imu), dtype=np.float64).ravel()

        wz_s[k] = obs_k[0]
        ay_s[k] = obs_k[1]
        ax_s[k] = obs_k[2]
        vx_s[k] = obs_k[3]
        rl_s[k] = obs_k[4]
        rr_s[k] = obs_k[5]

        x = x_next
        if alpha_h > 0.0 and (k + 1 < n):
            v_t = max(float(vx_r[k + 1]), 0.5)
            dv = (v_t - x[STATE_IDX["vx"]]) * alpha_h
            x[STATE_IDX["vx"]] += dv
            x[20:24] += dv / wheel_r

    return {
        "vx_sim": vx_s,
        "ax_sim": ax_s,
        "ay_sim": ay_s,
        "wz_sim": wz_s,
        "wheel_rpm_rl_sim": rl_s,
        "wheel_rpm_rr_sim": rr_s,
    }


def _compute_channel_metrics(
    real_dict: Dict[str, np.ndarray],
    sim_dict: Dict[str, np.ndarray],
    dt: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    return {
        "wz_rads": compute_signal_metrics(real_dict["wz_rads"], sim_dict["wz_sim"], dt=dt),
        "ay_ms2": compute_signal_metrics(real_dict["ay_ms2"], sim_dict["ay_sim"], dt=dt),
        "ax_ms2": compute_signal_metrics(real_dict["ax_ms2"], sim_dict["ax_sim"], dt=dt),
        "vx_ms": compute_signal_metrics(real_dict["vx_ms"], sim_dict["vx_sim"], dt=dt),
        "wheel_rpm_rl": compute_signal_metrics(real_dict["wheel_rpm_rl"], sim_dict["wheel_rpm_rl_sim"], dt=dt),
        "wheel_rpm_rr": compute_signal_metrics(real_dict["wheel_rpm_rr"], sim_dict["wheel_rpm_rr_sim"], dt=dt),
    }


def _evaluate_phi_across_runs(
    model: DifferentiableVehicle14Dof,
    fn_ctrl: ca.Function,
    fn_obs: ca.Function,
    runs: List[Dict[str, Any]],
    phi: np.ndarray,
    dt: float = 0.01,
    keep_sims: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """Evaluate a parameter vector phi across all 5 runs, returning pooled, train, val, and per-run metrics."""
    keys_r = ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads", "wheel_rpm_rl", "wheel_rpm_rr")
    keys_s = ("vx_sim", "ax_sim", "ay_sim", "wz_sim", "wheel_rpm_rl_sim", "wheel_rpm_rr_sim")

    pools = {
        "all": ({k: [] for k in keys_r}, {k: [] for k in keys_s}),
        "train": ({k: [] for k in keys_r}, {k: [] for k in keys_s}),
        "val": ({k: [] for k in keys_r}, {k: [] for k in keys_s}),
    }
    per_run: Dict[str, Any] = {}
    sims_out: Dict[str, Dict[str, np.ndarray]] = {}

    for run in runs:
        rid = run["run_id"]
        split = "train" if rid in TRAIN_RUN_IDS else "val"
        sim = simulate_calibrated_on_run(model, fn_ctrl, fn_obs, run, phi, horizon_reset_s=10.0, vx_observer_tau_s=2.5)
        ma = run["is_active"] & (np.abs(sim["vx_sim"] - run["vx_ms"]) < 6.0)
        r_m = {k: run[k][ma] for k in keys_r}
        s_m = {k: sim[k][ma] for k in keys_s}

        per_run[rid] = {
            "split": split,
            "n_eval_samples": int(np.sum(ma)),
            "metrics": _compute_channel_metrics(r_m, s_m, dt=dt),
        }
        for grp in ("all", split):
            for k in keys_r:
                pools[grp][0][k].append(r_m[k])
            for k in keys_s:
                pools[grp][1][k].append(s_m[k])
        if keep_sims:
            sims_out[rid] = {**sim, "eval_mask": ma}

    summary = {
        "pooled_all_5_runs": _compute_channel_metrics(
            {k: np.concatenate(v) for k, v in pools["all"][0].items()},
            {k: np.concatenate(v) for k, v in pools["all"][1].items()},
            dt=dt,
        ),
        "pooled_train_3_runs": _compute_channel_metrics(
            {k: np.concatenate(v) for k, v in pools["train"][0].items()},
            {k: np.concatenate(v) for k, v in pools["train"][1].items()},
            dt=dt,
        ),
        "pooled_val_2_runs": _compute_channel_metrics(
            {k: np.concatenate(v) for k, v in pools["val"][0].items()},
            {k: np.concatenate(v) for k, v in pools["val"][1].items()},
            dt=dt,
        ),
        "per_run": per_run,
    }
    return summary, sims_out


def run_full_ter26_adjoint_calibration(
    report_path: Path = DEFAULT_REPORT_PATH,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Execute complete TeR26 C2 smoothing audit + real-track adjoint SysID + ablation + parity."""
    t_start = time.perf_counter()
    print("[TeR26-Adjoint-SysID] Loading real TeR26 SocketCAN track runs...", flush=True)
    runs, data_source = load_real_ter26_track_runs(dt=dt)
    train_runs = [r for r in runs if r["run_id"] in TRAIN_RUN_IDS]
    val_runs = [r for r in runs if r["run_id"] in VAL_RUN_IDS]
    print(
        f"[TeR26-Adjoint-SysID] Loaded {len(runs)} runs: train={[r['run_id'] for r in train_runs]}, "
        f"val={[r['run_id'] for r in val_runs]}",
        flush=True,
    )

    # Local symbolic DifferentiableVehicle14Dof (rk4_substeps=2 => 5 ms RK4 step, never touches .so)
    model = DifferentiableVehicle14Dof(
        vehicle_id="ter26_rwd",
        suspension_mode="3d_double_wishbone_lut",
        tire_compound="R20",
        use_pinn=True,
        dt=dt,
        rk4_substeps=2,
    )
    fn_ctrl = build_smooth_rwd_ctrl_function(wheel_r=float(model.cfg.wheel_radius))
    fn_obs = build_imu_obs_function(gravity=float(model.cfg.gravity))

    # 1. Calibrated C2 smoothing & AD-vs-FD audit
    print("[TeR26-Adjoint-SysID] Running C2 smoothing audit & 19-param AD vs central FD check...", flush=True)
    c2_audit = run_c2_smoothing_audit(model, fn_ctrl, fn_obs)
    print(
        f"  -> Smoothing err < 1e-4: {c2_audit['model_14dof_smoothing_sensitivity']['smoothing_error_lt_1e4']} | "
        f"AD vs FD max_rel_err={c2_audit['adjoint_vs_central_fd_19params']['max_rel_err']:.2e} "
        f"(all < 1e-6: {c2_audit['adjoint_vs_central_fd_19params']['all_passed_lt_1e6']})",
        flush=True,
    )

    # 2. Extract stratified training windows strictly from the 3 training runs
    train_windows = select_training_windows(train_runs, model, win_len=200, n_per_run=6)
    print(f"[TeR26-Adjoint-SysID] Selected {len(train_windows)} stratified training windows (2.0s each) from 3 train runs", flush=True)

    # 3. Run exact reverse-mode adjoint L-BFGS-B calibration
    print("[TeR26-Adjoint-SysID] Running exact reverse-mode adjoint L-BFGS-B on real track runs...", flush=True)
    sysid_res = run_adjoint_lbfgsb_calibration(model, fn_ctrl, fn_obs, train_windows, maxiter=45)
    print(
        f"  -> Converged={sysid_res['converged']} in {sysid_res['n_iterations']} iters "
        f"({sysid_res['wall_time_s']}s), train data loss -{sysid_res['loss_reduction_pct']:.2f}%",
        flush=True,
    )

    phi_nom = np.asarray(sysid_res["phi_nominal"], dtype=np.float64)
    phi_cal = np.asarray(sysid_res["phi_calibrated"], dtype=np.float64)

    # Construct ablation waterfall parameter vectors:
    #   Stage 0: Baseline (phi_nom)
    #   Stage 1: + Powertrain & Hydraulic Brakes only
    phi_stage1 = phi_nom.copy()
    phi_stage1[N_THETA_OPT : N_THETA_OPT + N_ACT_OPT] = phi_cal[N_THETA_OPT : N_THETA_OPT + N_ACT_OPT]
    #   Stage 2: + Powertrain/Brakes + Aero/Susp/Compliance/Relaxation
    phi_stage2 = phi_stage1.copy()
    phi_stage2[:N_THETA_OPT] = phi_cal[:N_THETA_OPT]
    #   Stage 3: Full Calibrated (+ IMU mounting misalignment & static biases)
    phi_stage3 = phi_cal.copy()

    # 4. Evaluate all 4 stages across ALL 5 runs (100 Hz)
    print("[TeR26-Adjoint-SysID] Evaluating Stage 0 (Baseline) across all 5 runs...", flush=True)
    eval_before, sims_before = _evaluate_phi_across_runs(model, fn_ctrl, fn_obs, runs, phi_nom, dt=dt, keep_sims=True)
    print("[TeR26-Adjoint-SysID] Evaluating Stage 1 (+Powertrain/Brakes) across all 5 runs...", flush=True)
    eval_stage1, _ = _evaluate_phi_across_runs(model, fn_ctrl, fn_obs, runs, phi_stage1, dt=dt, keep_sims=False)
    print("[TeR26-Adjoint-SysID] Evaluating Stage 2 (+Aero/Susp/Tires) across all 5 runs...", flush=True)
    eval_stage2, _ = _evaluate_phi_across_runs(model, fn_ctrl, fn_obs, runs, phi_stage2, dt=dt, keep_sims=False)
    print("[TeR26-Adjoint-SysID] Evaluating Stage 3 (Full Adjoint-Calibrated) across all 5 runs...", flush=True)
    eval_after, sims_after = _evaluate_phi_across_runs(model, fn_ctrl, fn_obs, runs, phi_stage3, dt=dt, keep_sims=True)

    # 5. Evaluate TeR26HighFidelityCasADi reference model across all 5 runs for parity
    print("[TeR26-Adjoint-SysID] Evaluating Ref14 (TeR26HighFidelityCasADi) for parity...", flush=True)
    model_ref = TeR26HighFidelityCasADi(use_3d_kinematics=True, use_thermal_coupling=True)
    theta_ref_sysid = json.loads(SYSID_PARAMS_JSON.read_text(encoding="utf-8"))["identified_params"]
    th_ref_vec = theta_dict_to_vec(theta_ref_sysid)

    keys_r = ("vx_ms", "ax_ms2", "ay_ms2", "wz_rads", "wheel_rpm_rl", "wheel_rpm_rr")
    keys_s = ("vx_sim", "ax_sim", "ay_sim", "wz_sim", "wheel_rpm_rl_sim", "wheel_rpm_rr_sim")
    pool_real = {k: [] for k in keys_r}
    pool_ref = {k: [] for k in keys_s}
    pool_cal = {k: [] for k in keys_s}
    sims_ref: Dict[str, Dict[str, np.ndarray]] = {}

    for run in runs:
        rid = run["run_id"]
        ax_cmd = model_ref.map_pedals_to_ax_cmd_array(
            run["vx_ms"], run["throttle"], run["brake"], run["trq_limit_nm"]
        )
        sim_r = model_ref.simulate_trajectory(
            t_s=run["t_s"],
            delta_rad=run["delta_rad"],
            ax_cmd_drive=ax_cmd,
            vx_meas=run["vx_ms"],
            wz_meas=run["wz_rads"],
            theta=th_ref_vec,
            horizon_observer_tau_s=2.5,
        )
        sim_r["wheel_rpm_rl_sim"] = sim_r["wheel_rpm_sim"][:, 2]
        sim_r["wheel_rpm_rr_sim"] = sim_r["wheel_rpm_sim"][:, 3]
        sims_ref[rid] = sim_r

        ma = sims_after[rid]["eval_mask"]
        for k in keys_r:
            pool_real[k].append(run[k][ma])
        for k in keys_s:
            pool_ref[k].append(sim_r[k][ma])
            pool_cal[k].append(sims_after[rid][k][ma])

    cat_real = {k: np.concatenate(v) for k, v in pool_real.items()}
    cat_ref = {k: np.concatenate(v) for k, v in pool_ref.items()}
    cat_cal = {k: np.concatenate(v) for k, v in pool_cal.items()}
    ref14_vs_real = _compute_channel_metrics(cat_real, cat_ref, dt=dt)
    cal_diff14_vs_ref14 = {
        "wz_rads": compute_signal_metrics(cat_ref["wz_sim"], cat_cal["wz_sim"], dt=dt),
        "ay_ms2": compute_signal_metrics(cat_ref["ay_sim"], cat_cal["ay_sim"], dt=dt),
        "ax_ms2": compute_signal_metrics(cat_ref["ax_sim"], cat_cal["ax_sim"], dt=dt),
        "vx_ms": compute_signal_metrics(cat_ref["vx_sim"], cat_cal["vx_sim"], dt=dt),
    }

    # 6. Build Before vs After delta table (All 5 runs, Train 3, Held-out Val 2)
    def _delta_table(m_before: Dict[str, Any], m_after: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for ch in ("wz_rads", "ay_ms2", "ax_ms2", "vx_ms", "wheel_rpm_rl", "wheel_rpm_rr"):
            b, a = m_before[ch], m_after[ch]
            rmse_red_pct = 100.0 * (b["rmse"] - a["rmse"]) / max(b["rmse"], 1e-12)
            out[ch] = {
                "before_r": round(b["r"], 5),
                "after_r": round(a["r"], 5),
                "delta_r": round(a["r"] - b["r"], 5),
                "before_r2": round(b["r2"], 5),
                "after_r2": round(a["r2"], 5),
                "before_rmse": round(b["rmse"], 5),
                "after_rmse": round(a["rmse"], 5),
                "rmse_reduction_pct": round(rmse_red_pct, 2),
                "before_bias": round(b["bias"], 5),
                "after_bias": round(a["bias"], 5),
            }
        return out

    # 7. Extract compact 25s before/after time-series windows (one train run, one held-out val run)
    #    at 50 Hz (1250 samples) for 300 DPI plotting by @mirenasim-parity-qa
    plot_windows: Dict[str, Any] = {}
    for target_split, target_rid in (("held_out_val", "ter26_track_5"), ("train", "ter26_track_3")):
        run_obj = next(r for r in runs if r["run_id"] == target_rid)
        ma = sims_after[target_rid]["eval_mask"]
        idx_act = np.where(ma)[0]
        win_n = min(2500, len(idx_act))
        best_sl = None
        best_sc = -1.0
        for i0 in range(0, max(1, len(idx_act) - win_n), 400):
            sl = idx_act[i0 : i0 + win_n]
            if np.max(np.diff(sl)) > 5:
                continue
            sc = float(np.std(run_obj["ay_ms2"][sl])) + 2.0 * float(np.std(run_obj["ax_ms2"][sl]))
            if sc > best_sc:
                best_sc = sc
                best_sl = sl
        if best_sl is not None:
            sl_p = best_sl[::2]  # 50 Hz
            t_rel = run_obj["t_s"][sl_p] - run_obj["t_s"][sl_p[0]]
            plot_windows[target_split] = {
                "run_id": target_rid,
                "split": target_split,
                "sample_rate_hz": 50.0,
                "t_s": np.round(t_rel, 3),
                "delta_deg": np.round(np.degrees(run_obj["delta_rad"][sl_p]), 3),
                "wz_real_rads": np.round(run_obj["wz_rads"][sl_p], 4),
                "wz_before_rads": np.round(sims_before[target_rid]["wz_sim"][sl_p], 4),
                "wz_after_rads": np.round(sims_after[target_rid]["wz_sim"][sl_p], 4),
                "wz_ref14_rads": np.round(sims_ref[target_rid]["wz_sim"][sl_p], 4),
                "ay_real_ms2": np.round(run_obj["ay_ms2"][sl_p], 3),
                "ay_before_ms2": np.round(sims_before[target_rid]["ay_sim"][sl_p], 3),
                "ay_after_ms2": np.round(sims_after[target_rid]["ay_sim"][sl_p], 3),
                "ay_ref14_ms2": np.round(sims_ref[target_rid]["ay_sim"][sl_p], 3),
                "ax_real_ms2": np.round(run_obj["ax_ms2"][sl_p], 3),
                "ax_before_ms2": np.round(sims_before[target_rid]["ax_sim"][sl_p], 3),
                "ax_after_ms2": np.round(sims_after[target_rid]["ax_sim"][sl_p], 3),
                "vx_real_ms": np.round(run_obj["vx_ms"][sl_p], 3),
                "vx_before_ms": np.round(sims_before[target_rid]["vx_sim"][sl_p], 3),
                "vx_after_ms": np.round(sims_after[target_rid]["vx_sim"][sl_p], 3),
                "wheel_rpm_rl_real": np.round(run_obj["wheel_rpm_rl"][sl_p], 1),
                "wheel_rpm_rl_before": np.round(sims_before[target_rid]["wheel_rpm_rl_sim"][sl_p], 1),
                "wheel_rpm_rl_after": np.round(sims_after[target_rid]["wheel_rpm_rl_sim"][sl_p], 1),
                "wheel_rpm_rr_real": np.round(run_obj["wheel_rpm_rr"][sl_p], 1),
                "wheel_rpm_rr_before": np.round(sims_before[target_rid]["wheel_rpm_rr_sim"][sl_p], 1),
                "wheel_rpm_rr_after": np.round(sims_after[target_rid]["wheel_rpm_rr_sim"][sl_p], 1),
            }

    report = {
        "metadata": {
            "vehicle_id": "ter26_rwd",
            "model_id": "differentiable_casadi_14dof",
            "suspension_mode": "3d_double_wishbone_lut",
            "tire_compound": "R20",
            "use_pinn": True,
            "data_source": data_source,
            "n_runs": len(runs),
            "train_runs": list(TRAIN_RUN_IDS),
            "held_out_val_runs": list(VAL_RUN_IDS),
            "sample_rate_hz": round(1.0 / dt, 1),
            "wall_time_s": round(time.perf_counter() - t_start, 2),
        },
        "c2_smoothing_and_gradient_audit": c2_audit,
        "adjoint_sysid_calibration": sysid_res,
        "before_vs_after_summary": {
            "all_5_runs": _delta_table(eval_before["pooled_all_5_runs"], eval_after["pooled_all_5_runs"]),
            "train_3_runs": _delta_table(eval_before["pooled_train_3_runs"], eval_after["pooled_train_3_runs"]),
            "held_out_val_2_runs": _delta_table(eval_before["pooled_val_2_runs"], eval_after["pooled_val_2_runs"]),
        },
        "residual_ablation_waterfall_all_5_runs": {
            "stage0_baseline": eval_before["pooled_all_5_runs"],
            "stage1_plus_powertrain_brakes": eval_stage1["pooled_all_5_runs"],
            "stage2_plus_aero_susp_tires": eval_stage2["pooled_all_5_runs"],
            "stage3_full_adjoint_calibrated": eval_after["pooled_all_5_runs"],
        },
        "before_metrics": eval_before,
        "after_metrics": eval_after,
        "ref14_parity": {
            "ref14_vs_real_all_5_runs": ref14_vs_real,
            "calibrated_diff14_vs_ref14_all_5_runs": cal_diff14_vs_ref14,
        },
        "plot_timeseries_windows": plot_windows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    clean_report = _sanitize_json(report)
    report_path.write_text(json.dumps(clean_report, indent=2, ensure_ascii=True), encoding="ascii")
    print(f"[TeR26-Adjoint-SysID] Saved report to {report_path}", flush=True)
    return clean_report


def main() -> None:
    rep = run_full_ter26_adjoint_calibration(report_path=DEFAULT_REPORT_PATH)
    print("\n=== TeR26 BEFORE vs AFTER (ALL 5 REAL TRACK RUNS) ===", flush=True)
    for ch, d in rep["before_vs_after_summary"]["all_5_runs"].items():
        print(
            f"  {ch:14s} | R: {d['before_r']:.4f} -> {d['after_r']:.4f} | "
            f"R2: {d['before_r2']:.4f} -> {d['after_r2']:.4f} | "
            f"RMSE: {d['before_rmse']:.4f} -> {d['after_rmse']:.4f} ({d['rmse_reduction_pct']:+.2f}%) | "
            f"bias: {d['before_bias']:+.4f} -> {d['after_bias']:+.4f}",
            flush=True,
        )
    print("\n=== HELD-OUT VALIDATION (2 UNSEEN RUNS: ter26_track_2, ter26_track_5) ===", flush=True)
    for ch, d in rep["before_vs_after_summary"]["held_out_val_2_runs"].items():
        print(
            f"  {ch:14s} | R: {d['before_r']:.4f} -> {d['after_r']:.4f} | "
            f"RMSE: {d['before_rmse']:.4f} -> {d['after_rmse']:.4f} ({d['rmse_reduction_pct']:+.2f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
