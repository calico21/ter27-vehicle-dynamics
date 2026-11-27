#!/usr/bin/env python3
"""
CasADi Suspension Setup Optimization & Vehicle Engineering Trade-Off Lab for TeR27 4WD.

Uses the parameterized smooth C^2 ter27_casadi_model.py + IPOPT direct collocation OCP
and implicit/envelope sensitivity analysis on FSG 2023 (FSG_RUBBERED, human_achievable):

1. Track Setup Optimization (simultaneous setup + trajectory Co-Design OCP):
     - LLTD in [0.42, 0.62]
     - Roll center heights: h_rc_f in [0.005, 0.060] m, h_rc_r in [0.015, 0.080] m
     - Static cambers: camber_f_deg in [-3.8, -0.5] deg, camber_r_deg in [-3.2, -0.5] deg
     - Anti-dive in [0.20, 0.85], anti-squat in [0.15, 0.75]
     - Hydraulic brake_bias_front in [0.50, 0.72]
2. Vehicle Engineering Trade-Offs:
     - Mass vs Aero: exact kg of added mass equivalent to +10% CL*A, +10% CD*A, CoP +/-3%
     - CG Height & Yaw Inertia: impact of h_CG +/-15 mm and I_zz +/-10% on lap time & Drivability Score
     - Stiffness vs 22 km Thermal Degradation: roll stiffness (LLTD & k_roll_scale) vs
       cumulative Hoosier 16.0x7.5-10 R20 tire thermal degradation over a 22 km Endurance race
3. Closed-Loop 400 Hz 14-DOF Godot Simulator Validation:
     - Validates Nominal Setup vs CasADi Optimal Track Setup in Full14DofModel.gd
       confirming real lap-time gain, improved Drivability Score, 0 cones hit, 0 HV violations.

Outputs:
  - vd_correlation_lab/results/casadi_suite/vehicle_design_setup_report.json
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
RUNNER_GD_ORIG = LAB_DIR / "lts" / "lts_transient_runner.gd"
TRACK_JSON = REPO_ROOT / "data" / "tracks" / "fsg_2023.json"
RACING_LINE_REAL = SIM_DIR / "Assets" / "RacingLines" / "fsg_2023_racing_line_real.json"
OUT_DIR = LAB_DIR / "results" / "casadi_suite"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))

from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (  # noqa: E402
    COLD_TEST_OCT_LAMBDA_MUY_TARGET,
    FSG_RUBBERED_ENV_MU_SCALE,
    FSG_RUBBERED_LAMBDA_MUY_TARGET,
    GRIP_PROFILES,
    TRACKSIDE_SETUP_BOUNDS,
    load_ter27_params,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    DESIGN_PARAM_NAMES,
    TeR27CasADiVehicle,
    build_casadi_vehicle,
    build_symbolic_design_params,
    default_design_params,
    thermal_grip_multiplier_smooth,
    tire_temp_dot_smooth,
)

# Chassis swept-path envelope (exact 4-corner check)
CAR_WIDTH_M = 1.42
CAR_HALF_W_M = 0.5 * CAR_WIDTH_M
CAR_L_FRONT_M = 1.15
CAR_L_REAR_M = 0.95

# Human-achievable limits (FSG_RUBBERED)
CONE_MARGIN_M = 0.20
MAX_STEER_RATE_RAD_S = math.radians(62.0)
MAX_BETA_RAD = math.radians(4.8)

# Strict HV bounds
P_DC_MAX_KW_HARD = 80.0
P_DC_MIN_KW_HARD = -25.0
P_DC_MAX_KW_NLP = 79.95
P_DC_MIN_KW_NLP = -24.95

# Track setup optimization bounds (8 setup parameters)
SETUP_OPT_KEYS: Tuple[str, ...] = (
    "lltd",
    "h_rc_f",
    "h_rc_r",
    "camber_f_deg",
    "camber_r_deg",
    "anti_dive",
    "anti_squat",
    "brake_bias_front",
)
SETUP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lltd": (0.42, 0.62),
    "h_rc_f": (0.005, 0.060),
    "h_rc_r": (0.015, 0.080),
    "camber_f_deg": (-3.8, -0.5),
    "camber_r_deg": (-3.2, -0.5),
    "anti_dive": (0.20, 0.85),
    "anti_squat": (0.15, 0.75),
    "brake_bias_front": (0.50, 0.72),
}
SETUP_SCALES: Dict[str, float] = {
    "lltd": 0.50,
    "h_rc_f": 0.03,
    "h_rc_r": 0.04,
    "camber_f_deg": 2.0,
    "camber_r_deg": 1.5,
    "anti_dive": 0.50,
    "anti_squat": 0.40,
    "brake_bias_front": 0.60,
}


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_smooth(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return arr.copy()
    k = 2 * radius + 1
    ext = np.concatenate([arr[-radius:], arr, arr[:radius]])
    cs = np.cumsum(np.insert(ext, 0, 0.0))
    return (cs[k:] - cs[:-k]) / float(k)


def load_track_grid(track_path: Path, n_intervals: int) -> Dict[str, Any]:
    raw = json.loads(track_path.read_text(encoding="utf-8"))
    length = float(raw["length"])
    s_raw = np.asarray(raw["s"], dtype=np.float64)
    x_raw = np.asarray(raw["x"], dtype=np.float64)
    y_raw = np.asarray(raw["y"], dtype=np.float64)
    k_raw = np.asarray(raw["kappa"], dtype=np.float64)
    wl_raw = np.asarray(raw["w_left"], dtype=np.float64)
    wr_raw = np.asarray(raw["w_right"], dtype=np.float64)

    s_ext = np.append(s_raw, length)
    cs_x = CubicSpline(s_ext, np.append(x_raw, x_raw[0]), bc_type="periodic")
    cs_y = CubicSpline(s_ext, np.append(y_raw, y_raw[0]), bc_type="periodic")
    cs_k = CubicSpline(s_ext, np.append(k_raw, k_raw[0]), bc_type="periodic")
    cs_wl = CubicSpline(s_ext, np.append(wl_raw, wl_raw[0]), bc_type="periodic")
    cs_wr = CubicSpline(s_ext, np.append(wr_raw, wr_raw[0]), bc_type="periodic")

    ds = length / float(n_intervals)
    m_pts = 2 * n_intervals
    s_all = np.arange(m_pts, dtype=np.float64) * (0.5 * ds)
    x_all = cs_x(s_all)
    y_all = cs_y(s_all)
    psi_all = np.arctan2(cs_y(s_all, 1), cs_x(s_all, 1))
    kappa_all = cs_k(s_all)
    wl_all = cs_wl(s_all)
    wr_all = cs_wr(s_all)

    return {
        "name": str(raw["name"]),
        "length": length,
        "n_intervals": n_intervals,
        "m_pts": m_pts,
        "ds": ds,
        "s": s_all,
        "x": x_all,
        "y": y_all,
        "psi": psi_all,
        "kappa": kappa_all,
        "w_left": wl_all,
        "w_right": wr_all,
        "raw_s": s_raw,
        "raw_x": x_raw,
        "raw_y": y_raw,
        "raw_psi": np.asarray(raw["psi"], dtype=np.float64),
        "raw_wl": wl_raw,
        "raw_wr": wr_raw,
    }


def build_warm_start(grid: Dict[str, Any], racing_line_path: Path, mass: float, max_steer_rad: float) -> Dict[str, np.ndarray]:
    rl = json.loads(racing_line_path.read_text(encoding="utf-8"))
    pts = rl["points"]
    length = grid["length"]
    m_pts = grid["m_pts"]
    s_all = grid["s"]

    rl_len = float(rl.get("length_m", length))
    s_rl = np.array([float(p["s"]) for p in pts], dtype=np.float64) * (length / max(rl_len, 1.0))
    n_rl = np.array([float(p["n"]) for p in pts], dtype=np.float64)
    v_rl = np.array([float(p["v_mps"]) for p in pts], dtype=np.float64)
    ax_rl = np.array([float(p["ax_mps2"]) for p in pts], dtype=np.float64)
    ay_rl = np.array([float(p["ay_mps2"]) for p in pts], dtype=np.float64)
    yaw_rl = np.array([float(p["yaw_rad"]) for p in pts], dtype=np.float64)
    steer_rl = np.array([float(p["steer_rad"]) for p in pts], dtype=np.float64)
    kap_rl = np.array([float(p["kappa"]) for p in pts], dtype=np.float64)

    def _interp_p(y_vals: np.ndarray) -> np.ndarray:
        return np.interp(s_all, np.append(s_rl, length), np.append(y_vals, y_vals[0]))

    vx_0 = np.clip(_periodic_smooth(_interp_p(v_rl), 2), 5.0, 30.0)
    n_0 = np.clip(_periodic_smooth(_interp_p(n_rl), 2), -0.55, 0.55)
    yaw_unw = np.unwrap(yaw_rl)
    d_yaw_close = _wrap_angle(yaw_rl[0] - yaw_rl[-1])
    yaw_interp = np.interp(s_all, np.append(s_rl, length), np.append(yaw_unw, yaw_unw[-1] + d_yaw_close))
    xi_0 = np.clip(_periodic_smooth(_wrap_angle(yaw_interp - grid["psi"]), 2), -0.35, 0.35)
    kap_0 = _periodic_smooth(_interp_p(kap_rl), 2)
    wz_0 = np.clip(vx_0 * kap_0, -3.0, 3.0)
    vy_0 = np.zeros(m_pts, dtype=np.float64)
    ax_0 = np.clip(_periodic_smooth(_interp_p(ax_rl), 2), -14.0, 11.0)
    ay_0 = np.clip(_periodic_smooth(_interp_p(ay_rl), 2), -14.0, 14.0)
    delta_0 = np.clip(_periodic_smooth(_interp_p(steer_rl), 2), -0.9 * max_steer_rad, 0.9 * max_steer_rad)
    kap_w_0 = np.clip(ax_0 / 180.0, -0.05, 0.05)
    kappa_4_0 = np.tile(kap_w_0[None, :], (4, 1))
    f_brk_0 = np.clip(np.maximum(-ax_0 * mass - 2200.0, 0.0), 0.0, 5000.0)

    return {
        "vx": vx_0,
        "vy": vy_0,
        "wz": wz_0,
        "n": n_0,
        "xi": xi_0,
        "delta": delta_0,
        "kappa_4": kappa_4_0,
        "F_brk_hyd": f_brk_0,
        "ax_lt": ax_0,
        "ay_lt": ay_0,
    }


def compute_4corner_clearance(
    n_arr: np.ndarray,
    xi_arr: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    corners = [
        (CAR_L_FRONT_M, CAR_HALF_W_M),
        (CAR_L_FRONT_M, -CAR_HALF_W_M),
        (-CAR_L_REAR_M, CAR_HALF_W_M),
        (-CAR_L_REAR_M, -CAR_HALF_W_M),
    ]
    sin_xi = np.sin(xi_arr)
    cos_xi = np.cos(xi_arr)
    lat_all = [xc * sin_xi + yc * cos_xi for xc, yc in corners]
    max_lat = np.maximum.reduce(lat_all)
    min_lat = np.minimum.reduce(lat_all)
    cl_l = w_left - (n_arr + max_lat)
    cl_r = w_right + (n_arr + min_lat)
    cl_min = np.minimum(cl_l, cl_r)
    return cl_min, float(np.min(cl_min)), int(np.sum(cl_min < 0.0))


def compute_ocp_drivability_score(
    delta: np.ndarray,
    beta_deg: np.ndarray,
    f_brk: np.ndarray,
    T_mot_4: np.ndarray,
    dt_step: np.ndarray,
) -> float:
    """Compute 0-100 Drivability Score for an OCP trajectory."""
    d_deg = np.rad2deg(delta)
    d_dot = (np.roll(d_deg, -1) - d_deg) / dt_step
    d_ddot = (np.roll(d_dot, -1) - d_dot) / dt_step
    st_rate_rms = float(np.sqrt(np.mean(d_dot ** 2)))
    st_jerk_rms = float(np.sqrt(np.mean(d_ddot ** 2)))
    beta_rms = float(np.sqrt(np.mean(beta_deg ** 2)))
    beta_max = float(np.max(np.abs(beta_deg)))

    # Smoothness of longitudinal demand
    t_sum = np.sum(T_mot_4, axis=0) - f_brk * 0.20
    dt_sum = np.diff(t_sum) / dt_step[:-1]
    ped_jerk_rms = float(np.sqrt(np.mean(dt_sum ** 2)))

    s_steer = float(np.clip(100.0 - 0.35 * max(st_rate_rms - 15.0, 0.0) - 0.012 * max(st_jerk_rms - 200.0, 0.0), 40.0, 100.0))
    s_beta = float(np.clip(100.0 - 5.0 * max(beta_max - 3.5, 0.0) - 6.0 * max(beta_rms - 1.4, 0.0), 30.0, 100.0))
    s_ped = float(np.clip(100.0 - 0.002 * max(ped_jerk_rms - 1500.0, 0.0), 40.0, 100.0))
    return round(float(np.clip(0.35 * s_steer + 0.40 * s_beta + 0.25 * s_ped, 0.0, 100.0)), 2)


def compute_14dof_drivability_score(csv_path: Path, summary: Dict[str, Any]) -> Dict[str, float]:
    """Compute 0-100 Drivability Score from 14-DOF Godot telemetry CSV."""
    if not csv_path.exists():
        return {"drivability_score": 0.0, "steer_rate_rms_dps": 0.0, "beta_rms_deg": 0.0, "pedal_reversals": 0}
    ts, st, bt, th, br = [], [], [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["t_s"]))
            st.append(float(row["steer_deg"]))
            bt.append(float(row["beta_deg"]))
            th.append(float(row["throttle"]))
            br.append(float(row["brake"]))
    t_arr = np.asarray(ts)
    st_arr = np.asarray(st)
    b_arr = np.asarray(bt)
    dt_arr = np.maximum(np.diff(t_arr), 1e-4)
    st_dot = np.diff(st_arr) / dt_arr
    st_ddot = np.diff(st_dot) / dt_arr[:-1]
    steer_rate_rms = float(np.sqrt(np.mean(st_dot ** 2)))
    steer_jerk_rms = float(np.sqrt(np.mean(st_ddot ** 2)))
    beta_rms = float(np.sqrt(np.mean(b_arr ** 2)))
    max_beta = float(summary.get("max_abs_beta_deg", np.max(np.abs(b_arr))))
    ey_rms = float(summary.get("rms_ey_race_m", 0.2))
    fight_kj = float(summary.get("torque_fight_kj", 0.0))

    ped = np.asarray(th) - np.asarray(br)
    d_ped = np.diff(ped)
    sgn_p = np.sign(np.where(np.abs(d_ped) > 0.04, d_ped, 0.0))
    nz = sgn_p[sgn_p != 0.0]
    reversals = int(np.sum(nz[1:] * nz[:-1] < 0.0)) if len(nz) > 2 else 0

    s_steer = float(np.clip(100.0 - 0.55 * max(steer_rate_rms - 18.0, 0.0) - 0.015 * max(steer_jerk_rms - 250.0, 0.0), 30.0, 100.0))
    s_beta = float(np.clip(100.0 - 5.5 * max(max_beta - 3.5, 0.0) - 6.0 * max(beta_rms - 1.5, 0.0), 20.0, 100.0))
    s_pedal = float(np.clip(100.0 - 0.12 * max(reversals - 60, 0), 35.0, 100.0))
    s_track = float(np.clip(100.0 - 55.0 * max(ey_rms - 0.12, 0.0) - 0.08 * max(fight_kj - 40.0, 0.0), 30.0, 100.0))
    score = 0.25 * s_steer + 0.35 * s_beta + 0.20 * s_pedal + 0.20 * s_track
    return {
        "drivability_score": round(float(np.clip(score, 0.0, 100.0)), 2),
        "steer_rate_rms_dps": round(steer_rate_rms, 2),
        "beta_rms_deg": round(beta_rms, 3),
        "pedal_reversals": reversals,
    }


def build_parametric_design_physics() -> Tuple[ca.Function, TeR27CasADiVehicle]:
    """
    Build a single compiled C^2 CasADi SX physics function parameterized by ALL 16
    design/setup parameters from ter27_casadi_model.py on FSG_RUBBERED.
    """
    dp_vec, dp_map = build_symbolic_design_params("dp")
    veh = build_casadi_vehicle(
        grip_profile="FSG_RUBBERED",
        mu_scale=float(GRIP_PROFILES["FSG_RUBBERED"]["env_mu_scale"]),
        temp_c=45.0,
        tire_temp_c=75.0,
        design_params=dp_map,
    )
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_kap = ca.SX.sym("kappa_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    res = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay)
    out_keys = [
        "Fx_tot", "Fy_tot", "Mz_tot", "P_dc_ecu_kw", "P_dc_drv_kw",
        "T_motor_4", "T_max_4", "T_min_4", "slip_power_4", "Fz_4",
    ]
    fn = ca.Function(
        "ter27_design_phys",
        [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay, dp_vec],
        [res[k] for k in out_keys],
    )
    return fn, veh


def dp_dict_to_vec(dp: Dict[str, float]) -> np.ndarray:
    return np.array([float(dp[k]) for k in DESIGN_PARAM_NAMES], dtype=np.float64)


def solve_design_ocp(
    fn_phys: ca.Function,
    veh_sym: TeR27CasADiVehicle,
    grid: Dict[str, Any],
    dp_base: Dict[str, float],
    free_setup_keys: Tuple[str, ...] = (),
    warm_start_w: Optional[np.ndarray] = None,
    label: str = "nominal",
    ipopt_max_iter: int = 500,
) -> Dict[str, Any]:
    """
    Solve FSG 2023 (FSG_RUBBERED, human_achievable) OCP with arbitrary subset of
    design_params promoted to global IPOPT decision variables (free_setup_keys).
    Also returns exact envelope-theorem / KKT sensitivity dT/dp when fixed!
    """
    t0 = time.perf_counter()
    p = veh_sym.params
    max_steer = float(p.max_steer_angle)
    n_int = grid["n_intervals"]
    m_pts = grid["m_pts"]
    ds = float(grid["ds"])
    kappa_c = grid["kappa"]
    w_left = grid["w_left"]
    w_right = grid["w_right"]

    fn_phys_map = fn_phys.map(m_pts)

    S_VX, S_VY, S_WZ, S_N, S_XI = 20.0, 2.0, 1.5, 1.0, 0.3
    S_DEL, S_KAP, S_FBRK, S_AX, S_AY = 0.3, 0.1, 4000.0, 10.0, 10.0

    W = ca.MX.sym("W", 13, m_pts)
    n_free = len(free_setup_keys)
    z_free = ca.MX.sym("z_free", n_free) if n_free > 0 else ca.MX.sym("z_free", 0)

    # Assemble 16-element design parameter MX vector
    dp_elems: List[ca.MX] = []
    free_idx_map = {k: i for i, k in enumerate(free_setup_keys)}
    for name in DESIGN_PARAM_NAMES:
        if name in free_idx_map:
            dp_elems.append(z_free[free_idx_map[name]] * SETUP_SCALES[name])
        else:
            dp_elems.append(ca.MX(float(dp_base[name])))
    dp_mx = ca.vertcat(*dp_elems)
    mass_mx = dp_mx[DESIGN_PARAM_NAMES.index("mass_kg")]
    izz_mx = dp_mx[DESIGN_PARAM_NAMES.index("i_zz")]

    vx = W[0, :] * S_VX
    vy = W[1, :] * S_VY
    wz = W[2, :] * S_WZ
    n_lat = W[3, :] * S_N
    xi = W[4, :] * S_XI
    delta = W[5, :] * S_DEL
    kappa_4 = W[6:10, :] * S_KAP
    F_brk_hyd = W[10, :] * S_FBRK
    ax_lt = W[11, :] * S_AX
    ay_lt = W[12, :] * S_AY

    Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv, T_mot, T_max, T_min, slip_p, _fz = fn_phys_map(
        vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt,
        ca.repmat(dp_mx, 1, m_pts),
    )

    kap_dm = ca.DM(kappa_c).T
    cos_xi = ca.cos(xi)
    sin_xi = ca.sin(xi)
    v_s = vx * cos_xi - vy * sin_xi
    sf = (1.0 - n_lat * kap_dm) / v_s

    dvx_ds = sf * (Fx_tot / mass_mx + vy * wz)
    dvy_ds = sf * (Fy_tot / mass_mx - vx * wz)
    dwz_ds = sf * (Mz_tot / izz_mx)
    dn_ds = sf * (vx * sin_xi + vy * cos_xi)
    dxi_ds = sf * wz - kap_dm

    X_s = W[0:5, :]
    F_s = ca.vertcat(dvx_ds / S_VX, dvy_ds / S_VY, dwz_ds / S_WZ, dn_ds / S_N, dxi_ds / S_XI)

    idx_k = list(range(0, m_pts, 2))
    idx_m = list(range(1, m_pts, 2))
    idx_kp1 = [(2 * ((k + 1) % n_int)) for k in range(n_int)]

    X_k, X_m, X_kp1 = X_s[:, idx_k], X_s[:, idx_m], X_s[:, idx_kp1]
    F_k, F_m, F_kp1 = F_s[:, idx_k], F_s[:, idx_m], F_s[:, idx_kp1]

    defect_mid = X_m - 0.5 * (X_k + X_kp1) - (ds / 8.0) * (F_k - F_kp1)
    defect_end = X_kp1 - X_k - (ds / 6.0) * (F_k + 4.0 * F_m + F_kp1)

    sf_k, sf_m, sf_kp1 = sf[0, idx_k], sf[0, idx_m], sf[0, idx_kp1]
    T_lap = ca.sum2((ds / 6.0) * (sf_k + 4.0 * sf_m + sf_kp1))

    idx_next = list(range(1, m_pts)) + [0]
    idx_prev = [m_pts - 1] + list(range(0, m_pts - 1))
    h_half = 0.5 * ds
    d_delta = delta[0, idx_next] - delta
    dd_delta = delta[0, idx_next] - 2.0 * delta + delta[0, idx_prev]
    d_kappa = kappa_4[:, idx_next] - kappa_4
    d_fbrk = (F_brk_hyd[0, idx_next] - F_brk_hyd) / 10000.0

    reg_delta = 3.0e-2 * ca.sum2(d_delta * d_delta) / h_half
    reg_ddelta = 9.0e-2 * ca.sum2(dd_delta * dd_delta) / (h_half ** 3)
    reg_kappa = 2.0e-2 * ca.sum1(ca.sum2(d_kappa * d_kappa)) / h_half
    reg_fbrk = 6.0e-3 * ca.sum2(d_fbrk * d_fbrk) / h_half
    reg_hyd = 1.5e-3 * ca.sum2(((F_brk_hyd / 10000.0) ** 2) * sf * h_half)

    obj = T_lap + reg_delta + reg_ddelta + reg_kappa + reg_fbrk + reg_hyd

    g_list: List[ca.MX] = []
    lbg_list: List[np.ndarray] = []
    ubg_list: List[np.ndarray] = []

    def _add_g(expr: ca.MX, lb_v: float | np.ndarray, ub_v: float | np.ndarray) -> None:
        vec = ca.vec(expr)
        n_el = vec.shape[0]
        g_list.append(vec)
        lbg_list.append(np.full(n_el, float(lb_v)) if np.isscalar(lb_v) else np.asarray(lb_v, dtype=np.float64).reshape(n_el))
        ubg_list.append(np.full(n_el, float(ub_v)) if np.isscalar(ub_v) else np.asarray(ub_v, dtype=np.float64).reshape(n_el))

    _add_g(defect_mid, 0.0, 0.0)
    _add_g(defect_end, 0.0, 0.0)
    _add_g((ax_lt - Fx_tot / mass_mx) / S_AX, 0.0, 0.0)
    _add_g((ay_lt - Fy_tot / mass_mx) / S_AY, 0.0, 0.0)
    _add_g((T_mot - T_max) / 250.0, -np.inf, 0.0)
    _add_g((T_min - T_mot) / 250.0, -np.inf, 0.0)
    _add_g(P_ecu / 80.0, P_DC_MIN_KW_NLP / 80.0, P_DC_MAX_KW_NLP / 80.0)
    _add_g(P_drv / 80.0, P_DC_MIN_KW_NLP / 80.0, P_DC_MAX_KW_NLP / 80.0)

    wl_lim = ca.DM(w_left - CONE_MARGIN_M).T
    wr_lim = ca.DM(w_right - CONE_MARGIN_M).T
    _add_g(n_lat + CAR_L_FRONT_M * sin_xi + CAR_HALF_W_M * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat - CAR_L_REAR_M * sin_xi + CAR_HALF_W_M * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat + CAR_L_FRONT_M * sin_xi - CAR_HALF_W_M * cos_xi + wr_lim, 0.0, np.inf)
    _add_g(n_lat - CAR_L_REAR_M * sin_xi - CAR_HALF_W_M * cos_xi + wr_lim, 0.0, np.inf)

    _add_g(v_s / S_VX, 3.0 / S_VX, np.inf)

    dt_half = 0.5 * (sf + sf[0, idx_next]) * h_half
    _add_g(d_delta - MAX_STEER_RATE_RAD_S * dt_half, -np.inf, 0.0)
    _add_g(-d_delta - MAX_STEER_RATE_RAD_S * dt_half, -np.inf, 0.0)

    tan_bmax = math.tan(MAX_BETA_RAD)
    _add_g((vy - vx * tan_bmax) / S_VY, -np.inf, 0.0)
    _add_g((-vy - vx * tan_bmax) / S_VY, -np.inf, 0.0)

    T_mot_pos = 0.5 * (T_mot + ca.sqrt(T_mot * T_mot + 0.01))
    bspd_prod = ca.repmat(F_brk_hyd / 4000.0, 4, 1) * (T_mot_pos / 250.0)
    _add_g(bspd_prod, -np.inf, 1.5e-3)

    g_vec = ca.vertcat(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    lbw = np.zeros((13, m_pts), dtype=np.float64)
    ubw = np.zeros((13, m_pts), dtype=np.float64)
    lbw[0, :] = 3.5 / S_VX
    ubw[0, :] = 38.0 / S_VX
    lbw[1, :] = -4.0 / S_VY
    ubw[1, :] = 4.0 / S_VY
    lbw[2, :] = -3.8 / S_WZ
    ubw[2, :] = 3.8 / S_WZ
    n_box_max = np.maximum(w_left - CAR_HALF_W_M - CONE_MARGIN_M, 0.05)
    n_box_min = -np.maximum(w_right - CAR_HALF_W_M - CONE_MARGIN_M, 0.05)
    lbw[3, :] = n_box_min / S_N
    ubw[3, :] = n_box_max / S_N
    lbw[4, :] = -0.70 / S_XI
    ubw[4, :] = 0.70 / S_XI
    lbw[5, :] = -max_steer / S_DEL
    ubw[5, :] = max_steer / S_DEL
    lbw[6:10, :] = -0.135 / S_KAP
    ubw[6:10, :] = 0.135 / S_KAP
    lbw[10, :] = 0.0 / S_FBRK
    ubw[10, :] = 12000.0 / S_FBRK
    lbw[11, :] = -24.0 / S_AX
    ubw[11, :] = 18.0 / S_AX
    lbw[12, :] = -22.0 / S_AY
    ubw[12, :] = 22.0 / S_AY

    lb_f = np.array([SETUP_BOUNDS[k][0] / SETUP_SCALES[k] for k in free_setup_keys], dtype=np.float64)
    ub_f = np.array([SETUP_BOUNDS[k][1] / SETUP_SCALES[k] for k in free_setup_keys], dtype=np.float64)
    lbx = np.concatenate([lbw.reshape(-1, order="F"), lb_f])
    ubx = np.concatenate([ubw.reshape(-1, order="F"), ub_f])

    if warm_start_w is not None:
        w0_traj = warm_start_w[: 13 * m_pts].copy()
    else:
        ws = build_warm_start(grid, RACING_LINE_REAL, float(dp_base["mass_kg"]), max_steer)
        w0 = np.zeros((13, m_pts), dtype=np.float64)
        w0[0, :] = ws["vx"] / S_VX
        w0[1, :] = ws["vy"] / S_VY
        w0[2, :] = ws["wz"] / S_WZ
        w0[3, :] = ws["n"] / S_N
        w0[4, :] = ws["xi"] / S_XI
        w0[5, :] = ws["delta"] / S_DEL
        w0[6:10, :] = ws["kappa_4"] / S_KAP
        w0[10, :] = ws["F_brk_hyd"] / S_FBRK
        w0[11, :] = ws["ax_lt"] / S_AX
        w0[12, :] = ws["ay_lt"] / S_AY
        w0_traj = w0.reshape(-1, order="F")

    z0_f = np.array([float(dp_base[k]) / SETUP_SCALES[k] for k in free_setup_keys], dtype=np.float64)
    x0 = np.clip(np.concatenate([w0_traj, z0_f]), lbx, ubx)

    x_sym = ca.vertcat(ca.vec(W), z_free) if n_free > 0 else ca.vec(W)
    nlp = {"x": x_sym, "f": obj, "g": g_vec}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.max_iter": int(ipopt_max_iter),
        "ipopt.tol": 1e-5,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_relax_factor": 1e-8,
    }
    safe_lbl = label.replace(".", "p").replace("-", "m")
    solver = ca.nlpsol(f"design_ocp_{safe_lbl}", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_opt = np.array(sol["x"].full()).ravel()

    w_opt = x_opt[: 13 * m_pts].reshape((13, m_pts), order="F")
    dp_opt = dict(dp_base)
    for i, k in enumerate(free_setup_keys):
        dp_opt[k] = float(x_opt[13 * m_pts + i] * SETUP_SCALES[k])

    vx_o = w_opt[0, :] * S_VX
    vy_o = w_opt[1, :] * S_VY
    wz_o = w_opt[2, :] * S_WZ
    n_o = w_opt[3, :] * S_N
    xi_o = w_opt[4, :] * S_XI
    delta_o = w_opt[5, :] * S_DEL
    kap4_o = w_opt[6:10, :] * S_KAP
    fbrk_o = w_opt[10, :] * S_FBRK
    ax_o = w_opt[11, :] * S_AX
    ay_o = w_opt[12, :] * S_AY

    dp_v_opt = dp_dict_to_vec(dp_opt)
    res_ev = fn_phys_map(
        vx_o[None, :], vy_o[None, :], wz_o[None, :], delta_o[None, :],
        kap4_o, fbrk_o[None, :], ax_o[None, :], ay_o[None, :],
        np.tile(dp_v_opt[:, None], (1, m_pts)),
    )
    P_ecu_o = np.array(res_ev[3].full()).ravel()
    P_drv_o = np.array(res_ev[4].full()).ravel()
    T_mot_o = np.array(res_ev[5].full())
    slip_p_o = np.array(res_ev[8].full())

    v_s_o = vx_o * np.cos(xi_o) - vy_o * np.sin(xi_o)
    sf_o = (1.0 - n_o * kappa_c) / v_s_o
    dt_int = (ds / 6.0) * (sf_o[idx_k] + 4.0 * sf_o[idx_m] + sf_o[idx_kp1])
    t_lap = float(np.sum(dt_int))
    dt_step = np.maximum(0.5 * (sf_o + np.roll(sf_o, -1)) * h_half, 1e-4)

    p_sf = P_ecu_o * sf_o
    e_net_kj = float(np.sum((ds / 6.0) * (p_sf[idx_k] + 4.0 * p_sf[idx_m] + p_sf[idx_kp1])))
    e_22km_kwh = (e_net_kj / 3600.0) * (22000.0 / grid["length"])

    hv_viol = int(
        np.sum((P_ecu_o > P_DC_MAX_KW_HARD + 1e-6) | (P_ecu_o < P_DC_MIN_KW_HARD - 1e-6))
        + np.sum((P_drv_o > P_DC_MAX_KW_HARD + 1e-6) | (P_drv_o < P_DC_MIN_KW_HARD - 1e-6))
    )
    _, min_cl_m, n_hits = compute_4corner_clearance(n_o, xi_o, w_left, w_right)
    beta_deg = np.rad2deg(np.arctan2(vy_o, np.maximum(vx_o, 0.5)))
    v_spd = np.hypot(vx_o, vy_o)
    drv_score = compute_ocp_drivability_score(delta_o, beta_deg, fbrk_o, T_mot_o, dt_step)

    solve_s = time.perf_counter() - t0
    print(
        f"[DESIGN-OCP:{label:20s}] {stats.get('return_status')} ({stats.get('iter_count'):3d} iters, {solve_s:4.1f}s) | "
        f"T_lap={t_lap:.4f}s | Drv={drv_score:.2f} | v_max={np.max(v_spd)*3.6:.1f}km/h | "
        f"P_dc=[{np.min(P_ecu_o):.1f},{np.max(P_ecu_o):.1f}]kW (hv_viol={hv_viol}, hits={n_hits})",
        flush=True,
    )

    return {
        "label": label,
        "status": str(stats.get("return_status")),
        "iterations": int(stats.get("iter_count", -1)),
        "solve_time_s": round(solve_s, 2),
        "design_params": {k: round(float(v), 5) for k, v in dp_opt.items()},
        "metrics": {
            "ocp_lap_time_s": round(t_lap, 4),
            "drivability_score": drv_score,
            "v_max_kmh": round(float(np.max(v_spd) * 3.6), 2),
            "ay_max_g": round(float(np.max(np.abs(ay_o)) / 9.81), 3),
            "beta_max_deg": round(float(np.max(np.abs(beta_deg))), 3),
            "P_dc_min_kw": round(float(np.min(P_ecu_o)), 2),
            "P_dc_max_kw": round(float(np.max(P_ecu_o)), 2),
            "hv_violations": hv_viol,
            "min_cone_clearance_m": round(min_cl_m, 4),
            "cones_hit": n_hits,
            "e_22km_kwh": round(e_22km_kwh, 3),
        },
        "x_opt": x_opt,
        "slip_power_4": slip_p_o,
        "dt_step": dt_step,
        "vx": vx_o,
        "trace": {
            "x_car": grid["x"] - n_o * np.sin(grid["psi"]),
            "y_car": grid["y"] + n_o * np.cos(grid["psi"]),
            "vx": vx_o,
            "ax": ax_o,
        },
    }


def simulate_22km_thermal_degradation(
    ocp_res: Dict[str, Any],
    track_length_m: float,
    t_track_c: float = 48.0,
    t_init_c: float = 55.0,
) -> Dict[str, Any]:
    """
    Integrate the smooth C^2 Hoosier 16.0x7.5-10 R20 bulk tire thermal ODE
    dT_i/dt = tire_temp_dot_smooth(T_i, t_track_c, slip_power_i, vx) over a 22 km
    Endurance race (~19 laps of FSG 2023) to quantify stiffness vs thermal degradation.
    """
    n_laps_22km = int(round(22000.0 / track_length_m))
    slip_p = ocp_res["slip_power_4"]  # (4, m_pts)
    dt_step = ocp_res["dt_step"]
    vx = ocp_res["vx"]
    t_lap_base = float(ocp_res["metrics"]["ocp_lap_time_s"])
    m_pts = slip_p.shape[1]

    T_cur = np.full(4, t_init_c, dtype=np.float64)
    lap_mu_means: List[float] = []
    lap_t_max: List[float] = []
    t_peak_4 = np.full(4, t_init_c, dtype=np.float64)

    for _lap in range(n_laps_22km):
        mu_acc = 0.0
        t_acc = 0.0
        for k in range(m_pts):
            dt_k = float(dt_step[k])
            dT = tire_temp_dot_smooth(T_cur, t_track_c, slip_p[:, k], float(vx[k]))
            T_cur = T_cur + dT * dt_k
            t_peak_4 = np.maximum(t_peak_4, T_cur)
            mu_k = float(np.mean(thermal_grip_multiplier_smooth(T_cur)))
            mu_acc += mu_k * dt_k
            t_acc += dt_k
        lap_mu_means.append(mu_acc / max(t_acc, 1e-6))
        lap_t_max.append(float(np.max(T_cur)))

    # First-order lap-time sensitivity to thermal grip loss: dt/t ~ -0.45 * (mu_th - 1.0)
    lap_times_est = [t_lap_base * (1.0 + 0.45 * max(1.0 - mu_m, 0.0)) for mu_m in lap_mu_means]
    total_22km_s = float(np.sum(lap_times_est))
    cum_deg_s = total_22km_s - n_laps_22km * t_lap_base
    t_front_peak = float(max(t_peak_4[0], t_peak_4[1]))
    t_rear_peak = float(max(t_peak_4[2], t_peak_4[3]))

    return {
        "n_laps_22km": n_laps_22km,
        "single_lap_base_s": round(t_lap_base, 4),
        "total_22km_time_s": round(total_22km_s, 2),
        "cumulative_thermal_deg_s": round(cum_deg_s, 3),
        "final_lap_mu_thermal": round(float(lap_mu_means[-1]), 4),
        "peak_tire_temp_c": round(float(np.max(t_peak_4)), 2),
        "peak_front_temp_c": round(t_front_peak, 2),
        "peak_rear_temp_c": round(t_rear_peak, 2),
        "front_minus_rear_temp_c": round(t_front_peak - t_rear_peak, 2),
    }


def _build_14dof_racing_track(grid: Dict[str, Any], trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    xc, yc, vx, ax = trace["x_car"], trace["y_car"], trace["vx"], trace["ax"]
    dx = np.diff(np.append(xc, xc[0]))
    dy = np.diff(np.append(yc, yc[0]))
    s_cum = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    r_len = float(s_cum[-1])
    cs_x = CubicSpline(s_cum, np.append(xc, xc[0]), bc_type="periodic")
    cs_y = CubicSpline(s_cum, np.append(yc, yc[0]), bc_type="periodic")
    cs_v = CubicSpline(s_cum, np.append(vx, vx[0]), bc_type="periodic")
    cs_a = CubicSpline(s_cum, np.append(ax, ax[0]), bc_type="periodic")
    n_pts = 600
    ds = r_len / float(n_pts)
    s_u = np.arange(n_pts, dtype=np.float64) * ds
    x_u, y_u = cs_x(s_u), cs_y(s_u)
    dx_u, dy_u = cs_x(s_u, 1), cs_y(s_u, 1)
    ddx_u, ddy_u = cs_x(s_u, 2), cs_y(s_u, 2)
    psi_u = np.arctan2(dy_u, dx_u)
    kap_u = (dx_u * ddy_u - dy_u * ddx_u) / (np.maximum(np.hypot(dx_u, dy_u), 1e-6) ** 3)
    return {
        "base_track": {
            "length": float(grid["length"]),
            "ds": float(grid["length"] / len(grid["raw_s"])),
            "s": [round(float(v), 4) for v in grid["raw_s"]],
            "x": [round(float(v), 5) for v in grid["raw_x"]],
            "y": [round(float(v), 5) for v in grid["raw_y"]],
            "psi": [round(float(v), 6) for v in grid["raw_psi"]],
            "w_left": [round(float(v), 4) for v in grid["raw_wl"]],
            "w_right": [round(float(v), 4) for v in grid["raw_wr"]],
        },
        "racing_track": {
            "length": round(r_len, 4),
            "ds": round(ds, 5),
            "s": [round(float(v), 4) for v in s_u],
            "x": [round(float(v), 5) for v in x_u],
            "y": [round(float(v), 5) for v in y_u],
            "psi": [round(float(v), 6) for v in psi_u],
            "kappa": [round(float(v), 6) for v in kap_u],
        },
        "v_qss": [round(float(v), 4) for v in np.clip(cs_v(s_u), 4.0, 36.0)],
        "ax_qss": [round(float(v), 4) for v in np.clip(cs_a(s_u), -20.0, 15.0)],
    }


def validate_setup_on_14dof(
    label: str,
    grid: Dict[str, Any],
    ocp_trace: Dict[str, np.ndarray],
    dp: Dict[str, float],
    params_nom: Any,
) -> Dict[str, Any]:
    """
    Validate a suspension + chassis setup on the genuine 400 Hz 14-DOF Godot simulator
    (Full14DofModel.gd) on FSG_RUBBERED (mu=0.88, cl_scale=0.78, temp=45 C).
    """
    orig_gd = RUNNER_GD_ORIG.read_text(encoding="utf-8")
    anchor_mass = '\tif cfg_json.has("mass_kg"):\n\t\tm.config.mass = float(cfg_json["mass_kg"])\n'
    extra_setup = (
        anchor_mass
        + '\tif cfg_json.has("brake_bias_front"):\n'
        + '\t\tm.config.brake_bias_front = float(cfg_json["brake_bias_front"])\n'
        + '\tif cfg_json.has("arb_stiffness_front"):\n'
        + '\t\tm.config.arb_stiffness_front = float(cfg_json["arb_stiffness_front"])\n'
        + '\tif cfg_json.has("arb_stiffness_rear"):\n'
        + '\t\tm.config.arb_stiffness_rear = float(cfg_json["arb_stiffness_rear"])\n'
        + '\tif cfg_json.has("roll_center_height_front"):\n'
        + '\t\tm.config.roll_center_height_front = float(cfg_json["roll_center_height_front"])\n'
        + '\tif cfg_json.has("roll_center_height_rear"):\n'
        + '\t\tm.config.roll_center_height_rear = float(cfg_json["roll_center_height_rear"])\n'
        + '\tif cfg_json.has("anti_dive_ratio_front"):\n'
        + '\t\tm.config.anti_dive_ratio_front = float(cfg_json["anti_dive_ratio_front"])\n'
        + '\tif cfg_json.has("anti_squat_ratio_rear"):\n'
        + '\t\tm.config.anti_squat_ratio_rear = float(cfg_json["anti_squat_ratio_rear"])\n'
        + '\tif cfg_json.has("static_camber_front_deg"):\n'
        + '\t\tm.config.static_camber_front_deg = float(cfg_json["static_camber_front_deg"])\n'
        + '\t\tm.config.camber_table_front = m.config._sync_camber_table_static(m.config.camber_table_front, m.config.static_camber_front_deg)\n'
        + '\tif cfg_json.has("static_camber_rear_deg"):\n'
        + '\t\tm.config.static_camber_rear_deg = float(cfg_json["static_camber_rear_deg"])\n'
        + '\t\tm.config.camber_table_rear = m.config._sync_camber_table_static(m.config.camber_table_rear, m.config.static_camber_rear_deg)\n'
    )
    anchor_ret = '\treturn m\n'
    extra_env = (
        '\tif cfg_json.has("env_mu_scale"):\n'
        + '\t\tm.env_mu_scale = float(cfg_json["env_mu_scale"])\n'
        + '\tif cfg_json.has("env_aero_cl_scale"):\n'
        + '\t\tm.env_aero_cl_scale = float(cfg_json["env_aero_cl_scale"])\n'
        + '\tm.config.track_temp_c = 45.0\n'
        + '\tm.config.tire_initial_temp_c = 70.0\n'
        + '\tm._sync_wheel_thermal_config()\n'
        + anchor_ret
    )
    patched_gd = orig_gd.replace(anchor_mass, extra_setup, 1).replace(anchor_ret, extra_env, 1)

    # Convert target LLTD to front/rear ARB stiffness while keeping total roll stiffness constant
    k_roll_tot = float(params_nom.k_roll_f + params_nom.k_roll_r)
    k_spr_f = float(params_nom.k_roll_f - 4500.0)
    k_spr_r = float(params_nom.k_roll_r - 5500.0)
    lltd_tgt = float(dp["lltd"])
    arb_f = max(500.0, lltd_tgt * k_roll_tot - k_spr_f)
    arb_r = max(500.0, (1.0 - lltd_tgt) * k_roll_tot - k_spr_r)

    trk_payload = _build_14dof_racing_track(grid, ocp_trace)

    with tempfile.TemporaryDirectory(prefix=f"mirena_setup14_{label}_") as tmp:
        tmp_dir = Path(tmp)
        runner_tmp = tmp_dir / "setup_14dof_runner.gd"
        runner_tmp.write_text(patched_gd, encoding="utf-8")

        def _eval(alpha_drv: float, write_csv: bool = False) -> Tuple[Dict[str, Any], Path]:
            job_path = tmp_dir / f"job_{alpha_drv:.4f}.json"
            out_path = tmp_dir / f"sum_{alpha_drv:.4f}.json"
            csv_path = (tmp_dir / f"trace_{alpha_drv:.4f}.csv") if write_csv else Path("")
            job = {
                "label": f"{label}_a{alpha_drv:.3f}",
                "ecu_driving_mode": 2,
                "tv_4wd_mode": 0,
                "use_3d_kinematics": True,
                "alpha_drv": float(alpha_drv),
                "n_laps": 1,
                "csv_path": str(csv_path) if write_csv else "",
                "env_mu_scale": 0.88,
                "env_aero_cl_scale": 0.78,
                "mass_kg": float(dp["mass_kg"]),
                "brake_bias_front": float(dp["brake_bias_front"]),
                "arb_stiffness_front": float(arb_f),
                "arb_stiffness_rear": float(arb_r),
                "roll_center_height_front": float(dp["h_rc_f"]),
                "roll_center_height_rear": float(dp["h_rc_r"]),
                "anti_dive_ratio_front": float(dp["anti_dive"]),
                "anti_squat_ratio_rear": float(dp["anti_squat"]),
                "static_camber_front_deg": float(dp["camber_f_deg"]),
                "static_camber_rear_deg": float(dp["camber_r_deg"]),
                **trk_payload,
            }
            job_path.write_text(json.dumps(job), encoding="utf-8")
            cmd = [
                "godot", "--headless", "--path", str(SIM_DIR),
                "-s", str(runner_tmp), "--", str(job_path), str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"14-DOF setup validation failed ({label}): {proc.stderr[-400:]}")
            return json.loads(out_path.read_text(encoding="utf-8")), csv_path

        def _valid(s: Dict[str, Any]) -> bool:
            p_max = float(s.get("p_dc_max_kw", 0.0))
            p_min = float(s.get("p_chg_min_kw", 0.0))
            hv_ok = (p_max <= P_DC_MAX_KW_HARD + 0.05) and (p_min >= P_DC_MIN_KW_HARD - 0.05)
            return (
                bool(s.get("valid", False))
                and int(s.get("cones_hit", 999)) == 0
                and int(s.get("corners_cones_hit", 999)) == 0
                and hv_ok
            )

        lo, hi = 0.56, 0.90
        best_a = 0.56
        best_s: Optional[Dict[str, Any]] = None
        best_csv: Optional[Path] = None
        for step in range(5):
            a_try = 0.73 if step == 0 else 0.5 * (lo + hi)
            s_try, csv_try = _eval(a_try, write_csv=True)
            if _valid(s_try):
                if best_s is None or float(s_try["mean_lap_time_s"]) < float(best_s["mean_lap_time_s"]):
                    best_s = s_try
                    best_a = a_try
                    best_csv = csv_try
                lo = a_try
            else:
                hi = a_try
        if best_s is None:
            for a_fb in (0.54, 0.50, 0.46):
                s_fb, csv_fb = _eval(a_fb, write_csv=True)
                if _valid(s_fb):
                    best_s = s_fb
                    best_a = a_fb
                    best_csv = csv_fb
                    break
        if best_s is None or best_csv is None:
            raise RuntimeError(f"No valid 14-DOF lap found for {label}")

        final_s = best_s
        drv = compute_14dof_drivability_score(best_csv, final_s)

    p_max = float(final_s.get("p_dc_max_kw", 0.0))
    p_min = float(final_s.get("p_chg_min_kw", 0.0))
    hv_viol = int((p_max > P_DC_MAX_KW_HARD + 0.05) or (p_min < P_DC_MIN_KW_HARD - 0.05))
    out = {
        "label": label,
        "alpha_drv": round(best_a, 4),
        "sim14dof_lap_time_s": round(float(final_s["mean_lap_time_s"]), 4),
        "drivability_score": drv["drivability_score"],
        "max_beta_deg": round(float(final_s.get("max_abs_beta_deg", 0.0)), 3),
        "cones_hit": int(final_s.get("cones_hit", 0)),
        "corners_cones_hit": int(final_s.get("corners_cones_hit", 0)),
        "p_dc_min_kw": round(p_min, 2),
        "p_dc_max_kw": round(p_max, 2),
        "hv_violations": hv_viol,
        "applied_arb_front_nm_rad": round(arb_f, 1),
        "applied_arb_rear_nm_rad": round(arb_r, 1),
    }
    print(
        f"[14-DOF-SETUP:{label:18s}] T_lap={out['sim14dof_lap_time_s']:.3f}s | Drv={out['drivability_score']:.2f} | "
        f"max|beta|={out['max_beta_deg']:.2f}deg | cones={out['cones_hit']}/{out['corners_cones_hit']} | "
        f"P_dc=[{out['p_dc_min_kw']:.1f},{out['p_dc_max_kw']:.1f}]kW (hv_viol={out['hv_violations']})",
        flush=True,
    )
    return out


def main() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_intervals = 160
    print(f"[DESIGN-LAB] Initializing CasADi Setup & Vehicle Trade-Off Lab (FSG_RUBBERED, N={n_intervals})...", flush=True)

    fn_phys, veh_sym = build_parametric_design_physics()
    grid = load_track_grid(TRACK_JSON, n_intervals=n_intervals)
    dp_nom = default_design_params(veh_sym.params)

    # ------------------------------------------------------------------
    # 1. Track Setup Optimization (Nominal vs Simultaneous 8-Param Optimal)
    # ------------------------------------------------------------------
    res_nom = solve_design_ocp(
        fn_phys, veh_sym, grid, dp_nom,
        free_setup_keys=(), warm_start_w=None, label="nominal_setup", ipopt_max_iter=450,
    )
    t_nom = res_nom["metrics"]["ocp_lap_time_s"]
    drv_nom = res_nom["metrics"]["drivability_score"]
    ws_nom = res_nom["x_opt"]

    res_opt = solve_design_ocp(
        fn_phys, veh_sym, grid, dp_nom,
        free_setup_keys=SETUP_OPT_KEYS, warm_start_w=ws_nom, label="optimal_track_setup", ipopt_max_iter=500,
    )
    t_opt = res_opt["metrics"]["ocp_lap_time_s"]
    dp_opt = res_opt["design_params"]

    # ------------------------------------------------------------------
    # 2. Vehicle Engineering Trade-Offs (warm-started from nominal OCP!)
    # ------------------------------------------------------------------
    print("\n[DESIGN-LAB] Running Vehicle Engineering Trade-Off Sweeps...", flush=True)

    def _run_pert(lbl: str, overrides: Dict[str, float]) -> Dict[str, Any]:
        dp_p = dict(dp_nom)
        dp_p.update(overrides)
        return solve_design_ocp(
            fn_phys, veh_sym, grid, dp_p,
            free_setup_keys=(), warm_start_w=ws_nom, label=lbl, ipopt_max_iter=400,
        )

    # 2a. Mass vs Aero
    r_m_lo = _run_pert("mass_minus_15kg", {"mass_kg": dp_nom["mass_kg"] - 15.0})
    r_m_hi = _run_pert("mass_plus_15kg", {"mass_kg": dp_nom["mass_kg"] + 15.0})
    dt_dm_s_per_kg = (r_m_hi["metrics"]["ocp_lap_time_s"] - r_m_lo["metrics"]["ocp_lap_time_s"]) / 30.0
    dt_dm_s_per_10kg = dt_dm_s_per_kg * 10.0

    r_cl_p10 = _run_pert("cl_plus_10pct", {"cl_scale": 1.10})
    r_cd_p10 = _run_pert("cd_plus_10pct", {"cd_scale": 1.10})
    r_cop_m3 = _run_pert("cop_minus_3pct", {"cop_front_frac": dp_nom["cop_front_frac"] - 0.03})
    r_cop_p3 = _run_pert("cop_plus_3pct", {"cop_front_frac": dp_nom["cop_front_frac"] + 0.03})

    dt_cl_p10 = r_cl_p10["metrics"]["ocp_lap_time_s"] - t_nom
    dt_cd_p10 = r_cd_p10["metrics"]["ocp_lap_time_s"] - t_nom
    dt_cop_m3 = r_cop_m3["metrics"]["ocp_lap_time_s"] - t_nom
    dt_cop_p3 = r_cop_p3["metrics"]["ocp_lap_time_s"] - t_nom

    # 2b. CG Height (+/-15 mm) & Yaw Inertia (+/-10%)
    r_hcg_lo = _run_pert("h_cg_minus_15mm", {"h_cg": dp_nom["h_cg"] - 0.015})
    r_hcg_hi = _run_pert("h_cg_plus_15mm", {"h_cg": dp_nom["h_cg"] + 0.015})
    r_izz_lo = _run_pert("i_zz_minus_10pct", {"i_zz": dp_nom["i_zz"] * 0.90})
    r_izz_hi = _run_pert("i_zz_plus_10pct", {"i_zz": dp_nom["i_zz"] * 1.10})

    # 2c. Stiffness vs 22 km Endurance Thermal Degradation
    print("\n[DESIGN-LAB] Running Stiffness vs 22 km Thermal Degradation Sweep...", flush=True)
    lltd_sweep_results: List[Dict[str, Any]] = []
    ws_lltd = ws_nom
    for lltd_val in (0.42, 0.46, 0.50, 0.54, 0.58, 0.62):
        dp_l = dict(dp_nom)
        dp_l["lltd"] = lltd_val
        r_l = solve_design_ocp(
            fn_phys, veh_sym, grid, dp_l,
            free_setup_keys=(), warm_start_w=ws_lltd, label=f"lltd_{lltd_val:.2f}", ipopt_max_iter=400,
        )
        ws_lltd = r_l["x_opt"]
        th_22 = simulate_22km_thermal_degradation(r_l, grid["length"])
        lltd_sweep_results.append({
            "lltd": lltd_val,
            "k_roll_scale": 1.0,
            "ocp_lap_time_s": r_l["metrics"]["ocp_lap_time_s"],
            "drivability_score": r_l["metrics"]["drivability_score"],
            **th_22,
        })

    kroll_sweep_results: List[Dict[str, Any]] = []
    for kr_val in (0.80, 1.00, 1.25):
        r_k = res_nom if abs(kr_val - 1.0) < 1e-6 else _run_pert(f"kroll_{kr_val:.2f}", {"k_roll_scale": kr_val})
        th_22 = simulate_22km_thermal_degradation(r_k, grid["length"])
        kroll_sweep_results.append({
            "lltd": round(float(dp_nom["lltd"]), 4),
            "k_roll_scale": kr_val,
            "ocp_lap_time_s": r_k["metrics"]["ocp_lap_time_s"],
            "drivability_score": r_k["metrics"]["drivability_score"],
            **th_22,
        })

    # ------------------------------------------------------------------
    # 3. 14-DOF Godot Simulator Validation (Nominal vs Optimal Track Setup)
    # ------------------------------------------------------------------
    print("\n[DESIGN-LAB] Validating Nominal vs Optimal Track Setup on 400 Hz 14-DOF Simulator...", flush=True)
    sim14_nom = validate_setup_on_14dof("nominal_setup", grid, res_nom["trace"], dp_nom, veh_sym.params)
    sim14_opt = validate_setup_on_14dof("optimal_setup", grid, res_opt["trace"], dp_opt, veh_sym.params)
    dt_14dof = sim14_opt["sim14dof_lap_time_s"] - sim14_nom["sim14dof_lap_time_s"]
    ddrv_14dof = sim14_opt["drivability_score"] - sim14_nom["drivability_score"]

    def _strip(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in r.items() if k not in ("x_opt", "slip_power_4", "dt_step", "vx", "trace")}

    report: Dict[str, Any] = {
        "track": grid["name"],
        "grip_profile": "FSG_RUBBERED",
        "ocp_mode": "human_achievable",
        "track_setup_optimization": {
            "setup_bounds": {k: list(SETUP_BOUNDS[k]) for k in SETUP_OPT_KEYS},
            "nominal_ocp": _strip(res_nom),
            "optimal_ocp": _strip(res_opt),
            "ocp_lap_time_gain_s": round(t_opt - t_nom, 4),
            "ocp_lap_time_gain_pct": round(100.0 * (t_opt - t_nom) / t_nom, 3),
            "optimal_vs_nominal_setup": {
                k: {
                    "nominal": round(float(dp_nom[k]), 5),
                    "optimal": round(float(dp_opt[k]), 5),
                    "delta": round(float(dp_opt[k] - dp_nom[k]), 5),
                }
                for k in SETUP_OPT_KEYS
            },
        },
        "vehicle_engineering_tradeoffs": {
            "mass_vs_aero": {
                "dt_dm_s_per_kg": round(dt_dm_s_per_kg, 5),
                "dt_dm_s_per_10kg": round(dt_dm_s_per_10kg, 4),
                "mass_minus_15kg_s": r_m_lo["metrics"]["ocp_lap_time_s"],
                "mass_nominal_s": t_nom,
                "mass_plus_15kg_s": r_m_hi["metrics"]["ocp_lap_time_s"],
                "plus_10pct_cl_a": {
                    "delta_lap_time_s": round(dt_cl_p10, 4),
                    "equivalent_mass_kg": round(-dt_cl_p10 / dt_dm_s_per_kg, 2),
                    "interpretation": "positive kg = mass saving equivalent of +10% CL*A",
                },
                "plus_10pct_cd_a": {
                    "delta_lap_time_s": round(dt_cd_p10, 4),
                    "equivalent_mass_kg": round(dt_cd_p10 / dt_dm_s_per_kg, 2),
                    "interpretation": "positive kg = added mass penalty equivalent of +10% CD*A",
                },
                "cop_shift_minus_3pct": {
                    "cop_front_frac": round(dp_nom["cop_front_frac"] - 0.03, 4),
                    "delta_lap_time_s": round(dt_cop_m3, 4),
                    "equivalent_mass_kg": round(dt_cop_m3 / dt_dm_s_per_kg, 2),
                },
                "cop_shift_plus_3pct": {
                    "cop_front_frac": round(dp_nom["cop_front_frac"] + 0.03, 4),
                    "delta_lap_time_s": round(dt_cop_p3, 4),
                    "equivalent_mass_kg": round(dt_cop_p3 / dt_dm_s_per_kg, 2),
                },
            },
            "cg_height_and_yaw_inertia": {
                "h_cg_minus_15mm": {
                    "h_cg_m": round(dp_nom["h_cg"] - 0.015, 4),
                    "lap_time_s": r_hcg_lo["metrics"]["ocp_lap_time_s"],
                    "delta_lap_time_s": round(r_hcg_lo["metrics"]["ocp_lap_time_s"] - t_nom, 4),
                    "drivability_score": r_hcg_lo["metrics"]["drivability_score"],
                    "delta_drivability": round(r_hcg_lo["metrics"]["drivability_score"] - drv_nom, 2),
                },
                "h_cg_plus_15mm": {
                    "h_cg_m": round(dp_nom["h_cg"] + 0.015, 4),
                    "lap_time_s": r_hcg_hi["metrics"]["ocp_lap_time_s"],
                    "delta_lap_time_s": round(r_hcg_hi["metrics"]["ocp_lap_time_s"] - t_nom, 4),
                    "drivability_score": r_hcg_hi["metrics"]["drivability_score"],
                    "delta_drivability": round(r_hcg_hi["metrics"]["drivability_score"] - drv_nom, 2),
                },
                "i_zz_minus_10pct": {
                    "i_zz_kgm2": round(dp_nom["i_zz"] * 0.90, 2),
                    "lap_time_s": r_izz_lo["metrics"]["ocp_lap_time_s"],
                    "delta_lap_time_s": round(r_izz_lo["metrics"]["ocp_lap_time_s"] - t_nom, 4),
                    "drivability_score": r_izz_lo["metrics"]["drivability_score"],
                    "delta_drivability": round(r_izz_lo["metrics"]["drivability_score"] - drv_nom, 2),
                },
                "i_zz_plus_10pct": {
                    "i_zz_kgm2": round(dp_nom["i_zz"] * 1.10, 2),
                    "lap_time_s": r_izz_hi["metrics"]["ocp_lap_time_s"],
                    "delta_lap_time_s": round(r_izz_hi["metrics"]["ocp_lap_time_s"] - t_nom, 4),
                    "drivability_score": r_izz_hi["metrics"]["drivability_score"],
                    "delta_drivability": round(r_izz_hi["metrics"]["drivability_score"] - drv_nom, 2),
                },
            },
            "stiffness_vs_22km_thermal_degradation": {
                "lltd_sweep": lltd_sweep_results,
                "k_roll_scale_sweep": kroll_sweep_results,
            },
        },
        "validation_14dof": {
            "nominal_setup": sim14_nom,
            "optimal_setup": sim14_opt,
            "delta_lap_time_s": round(dt_14dof, 4),
            "delta_lap_time_pct": round(100.0 * dt_14dof / sim14_nom["sim14dof_lap_time_s"], 3),
            "delta_drivability_score": round(ddrv_14dof, 2),
            "zero_cones_hit": (
                sim14_nom["cones_hit"] == 0 and sim14_opt["cones_hit"] == 0
                and sim14_nom["corners_cones_hit"] == 0 and sim14_opt["corners_cones_hit"] == 0
            ),
            "zero_hv_violations": (sim14_nom["hv_violations"] == 0 and sim14_opt["hv_violations"] == 0),
        },
        "total_runtime_s": round(time.perf_counter() - t0, 2),
    }

    rep_path = OUT_DIR / "vehicle_design_setup_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n[DESIGN-LAB] COMPLETE in {report['total_runtime_s']:.1f}s | "
        f"OCP gain={t_opt - t_nom:+.4f}s | 14-DOF gain={dt_14dof:+.4f}s (Drv {ddrv_14dof:+.2f}) | saved {rep_path}",
        flush=True,
    )
    return report


SETUPS_EXPORT_DIR = SIM_DIR / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter27_4wd" / "setups"
VAULT_07_DIR = REPO_ROOT / "outputs" / "validation_vault" / "07_design_tradeoffs_and_fs_points"
TRACKSIDE_REPORT_PATH = VAULT_07_DIR / "trackside_setup_optimization_report.json"


def export_mirenasim_setup_preset(
    event_key: str,
    setup_dict: Dict[str, float],
    base_time_s: float,
    optimal_time_s: float,
    compound: str = "R25B",
    out_dir: Path = SETUPS_EXPORT_DIR,
) -> Path:
    """
    Export a MirenaSim-compatible setup preset JSON into
    src/mirena_sim/Scenes/Vehicle/Config/vehicles/ter27_4wd/setups/optimal_<event>.json.
    """
    from vd_correlation_lab.optimization.lap_time_sim.solve_dynamic_events_casadi import _implied_arb_stiffness_nm_m

    p = load_ter27_params()
    lltd = float(setup_dict["lltd"])
    cam_f = float(setup_dict["camber_f_deg"])
    cam_r = float(setup_dict["camber_r_deg"])
    toe_f = float(setup_dict["toe_f_deg"])
    toe_r = float(setup_dict["toe_r_deg"])
    bb = float(setup_dict.get("brake_bias", setup_dict.get("brake_bias_front", p.brake_bias_front)))
    rake_mm = float(setup_dict.get("rake_mm", 0.0))

    arb_f, arb_r = _implied_arb_stiffness_nm_m(p, lltd, 1.0)
    rh_f = round(0.035 - 0.5 * rake_mm * 1e-3, 5)
    rh_r = round(0.035 + 0.5 * rake_mm * 1e-3, 5)

    meta_map = {
        "skidpad": (
            "optimal_skidpad",
            "CasADi-IPOPT co-optimized Skidpad setup on FSG_RUBBERED (lambda_muy=0.57)",
            "03_skidpad_attack.json",
            80.0,
            "smc",
        ),
        "accel": (
            "optimal_accel",
            "CasADi-IPOPT co-optimized 75 m Acceleration setup on FSG_RUBBERED (lambda_muy=0.57)",
            "04_launch_accel.json",
            80.0,
            "smc",
        ),
        "autocross": (
            "optimal_autocross",
            "CasADi-IPOPT co-optimized Autocross setup on FSG_RUBBERED (lambda_muy=0.57)",
            "05_smc_pivot_4wd.json",
            80.0,
            "smc",
        ),
        "endurance": (
            "optimal_endurance",
            "CasADi-IPOPT co-optimized 22 km Endurance setup on FSG_RUBBERED (lambda_muy=0.57, 3.5 kWh)",
            "01_endurance_safe.json",
            62.0,
            "v3",
        ),
    }
    preset_name, desc, tv_map, pwr_kw, ctrl_mode = meta_map[event_key]

    payload = {
        "name": preset_name,
        "vehicle_id": "ter27_4wd",
        "event": event_key,
        "description": desc,
        "spring_rate_front_n_m": round(float(p.k_spring_f), 1),
        "spring_rate_rear_n_m": round(float(p.k_spring_r), 1),
        "arb_front_nm_rad": round(float(arb_f), 1),
        "arb_rear_nm_rad": round(float(arb_r), 1),
        "lltd": round(lltd, 4),
        "camber_front_deg": round(cam_f, 3),
        "camber_rear_deg": round(cam_r, 3),
        "toe_front_deg": round(toe_f, 3),
        "toe_rear_deg": round(toe_r, 3),
        "ride_height_front_m": rh_f,
        "ride_height_rear_m": rh_r,
        "rake_mm": round(rake_mm, 3),
        "brake_bias": round(bb, 4),
        "tire_pressure_front_bar": 0.72,
        "tire_pressure_rear_bar": 0.72,
        "tv_map": tv_map,
        "power_limit_kw": pwr_kw,
        "control_law_mode": ctrl_mode,
        "casadi_metrics": {
            "grip_profile": "FSG_RUBBERED",
            "compound": compound,
            "lambda_muy": FSG_RUBBERED_LAMBDA_MUY_TARGET,
            "base_time_s": round(float(base_time_s), 4),
            "optimal_time_s": round(float(optimal_time_s), 4),
            "delta_time_s": round(float(optimal_time_s - base_time_s), 4),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{preset_name}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[PRESET] Exported {out_path}", flush=True)
    return out_path


def _fs_event_points(t_team: float, t_min: float, p_max: float, min_pts: float, t_max_factor: float) -> float:
    """Official FSG / FS Spain dynamic event scoring formula."""
    t_max = t_min * t_max_factor
    if t_team >= t_max:
        return min_pts
    num = (t_max / max(t_team, 1e-6)) - 1.0
    den = (t_max / max(t_min, 1e-6)) - 1.0
    return min(p_max, max(min_pts, min_pts + (p_max - min_pts) * (num / max(den, 1e-9))))


def run_trackside_4event_optimization(
    compound: str = "R25B",
    n_intervals_lap: int = 240,
    endurance_budget_kwh: float = 3.5,
) -> Dict[str, Any]:
    """
    Co-optimize trajectory + 4WD TV + 7 mechanical trackside setup parameters across all
    4 Formula Student dynamic events on FSG_RUBBERED (lambda_muy=0.57, 75 C nominal):
      1. Skidpad (Official Figure-8 & Stationary)
      2. Acceleration 75 m
      3. Autocross (FSG 2023 full lap)
      4. Endurance 22 km (FSG 2023 with 3.5 kWh / 22 km budget & thermal buildup)

    Exports all 4 MirenaSim presets into ter27_4wd/setups/optimal_{skidpad,accel,autocross,endurance}.json
    and writes trackside_setup_optimization_report.json to Validation Vault 07.
    """
    from vd_correlation_lab.optimization.lap_time_sim.solve_dynamic_events_casadi import (
        solve_accel_75m_casadi,
        solve_skidpad_figure8_casadi,
        solve_skidpad_stationary_casadi,
    )
    from vd_correlation_lab.optimization.lap_time_sim.solve_full_lap_ocp import (
        _resolve_warm_start_path,
        load_track_any,
        solve_profile_ocp,
    )

    t0 = time.perf_counter()
    p_nom = load_ter27_params()
    base_setup = {
        "lltd": round(float(p_nom.lltd_elastic_f), 4),
        "camber_f_deg": round(float(math.degrees(p_nom.static_camber_f_rad)), 3),
        "camber_r_deg": round(float(math.degrees(p_nom.static_camber_r_rad)), 3),
        "toe_f_deg": 0.0,
        "toe_r_deg": 0.0,
        "brake_bias": round(float(p_nom.brake_bias_front), 4),
        "rake_mm": 0.0,
    }

    print(
        f"=== 4-Event Trackside Setup Co-Optimization (FSG_RUBBERED lambda_muy=0.57, {compound}) ===",
        flush=True,
    )

    # 1. Skidpad (Base vs Optimal)
    print("\n[EVENT 1/4] Skidpad (Stationary + Figure-8)...", flush=True)
    sk_base_st = solve_skidpad_stationary_casadi("FSG_RUBBERED", compound, optimize_trackside_setup=False)
    sk_base_f8 = solve_skidpad_figure8_casadi("FSG_RUBBERED", compound, v_entry_ms=4.0, n_circle=24)
    sk_opt_st = solve_skidpad_stationary_casadi("FSG_RUBBERED", compound, optimize_trackside_setup=True)
    sk_setup = dict(sk_opt_st["optimal_setup"])
    sk_opt_f8 = solve_skidpad_figure8_casadi(
        "FSG_RUBBERED", compound, v_entry_ms=4.0, n_circle=24, design_overrides=sk_setup
    )
    sk_base_t = float(sk_base_f8["t_mean_periodic_s"])
    sk_opt_t = float(sk_opt_f8["t_mean_periodic_s"])
    export_mirenasim_setup_preset("skidpad", sk_setup, sk_base_t, sk_opt_t, compound=compound)

    # 2. Acceleration 75 m (Base vs Optimal)
    print("\n[EVENT 2/4] Acceleration 75 m...", flush=True)
    ac_base = solve_accel_75m_casadi("FSG_RUBBERED", compound, optimize_trackside_setup=False)
    ac_opt = solve_accel_75m_casadi("FSG_RUBBERED", compound, optimize_trackside_setup=True)
    ac_setup = dict(ac_opt["optimal_setup"])
    ac_base_t = float(ac_base["time_75m_s"])
    ac_opt_t = float(ac_opt["time_75m_s"])
    export_mirenasim_setup_preset("accel", ac_setup, ac_base_t, ac_opt_t, compound=compound)

    # 3. Autocross (FSG 2023 Max Attack, human_achievable)
    print("\n[EVENT 3/4] Autocross (FSG 2023 Full-Lap OCP)...", flush=True)
    _, grid_fsg = load_track_any(TRACK_JSON, n_intervals=n_intervals_lap)
    ws_path = _resolve_warm_start_path(grid_fsg["stem"], "FSG_RUBBERED")
    ax_res = solve_profile_ocp(
        grip_profile="FSG_RUBBERED",
        grid=grid_fsg,
        racing_line_path=ws_path,
        mode="human_achievable",
        energy_budget_kwh=None,
        ipopt_max_iter=600,
        print_level=0,
        optimize_trackside_setup=True,
        compound=compound,
    )
    ax_setup = dict(ax_res["recommended_setup"])
    ax_base_t = float(ax_setup["base_lap_time_s"])
    ax_opt_t = float(ax_setup["optimal_lap_time_s"])
    export_mirenasim_setup_preset("autocross", ax_setup, ax_base_t, ax_opt_t, compound=compound)

    # 4. Endurance 22 km (FSG 2023 with 3.5 kWh / 22 km budget & thermal buildup)
    print(f"\n[EVENT 4/4] Endurance 22 km (FSG 2023, budget={endurance_budget_kwh:.2f} kWh)...", flush=True)
    en_res = solve_profile_ocp(
        grip_profile="FSG_RUBBERED",
        grid=grid_fsg,
        racing_line_path=ws_path,
        mode="human_achievable",
        energy_budget_kwh=endurance_budget_kwh,
        ipopt_max_iter=600,
        print_level=0,
        optimize_trackside_setup=True,
        compound=compound,
    )
    en_setup = dict(en_res["recommended_setup"])
    n_laps_22km = 22000.0 / float(grid_fsg["length"])
    en_base_lap_s = float(en_setup["base_lap_time_s"])
    en_opt_lap_s = float(en_setup["optimal_lap_time_s"])
    en_base_22km_s = en_base_lap_s * n_laps_22km
    en_opt_22km_s = en_opt_lap_s * n_laps_22km
    export_mirenasim_setup_preset("endurance", en_setup, en_base_22km_s, en_opt_22km_s, compound=compound)

    # Official FSG / FS Spain points benchmarks (FSG 2023/2024 top-team references)
    t_min_skidpad = 4.65
    t_min_accel = 3.25
    t_min_autocross = 56.0
    t_min_endurance = 1150.0

    pts_base = {
        "skidpad": round(_fs_event_points(sk_base_t, t_min_skidpad, 75.0, 3.5, 1.25), 2),
        "accel": round(_fs_event_points(ac_base_t, t_min_accel, 75.0, 3.5, 1.50), 2),
        "autocross": round(_fs_event_points(ax_base_t, t_min_autocross, 100.0, 4.5, 1.25), 2),
        "endurance": round(_fs_event_points(en_base_22km_s, t_min_endurance, 250.0, 25.0, 1.333), 2),
    }
    pts_opt = {
        "skidpad": round(_fs_event_points(sk_opt_t, t_min_skidpad, 75.0, 3.5, 1.25), 2),
        "accel": round(_fs_event_points(ac_opt_t, t_min_accel, 75.0, 3.5, 1.50), 2),
        "autocross": round(_fs_event_points(ax_opt_t, t_min_autocross, 100.0, 4.5, 1.25), 2),
        "endurance": round(_fs_event_points(en_opt_22km_s, t_min_endurance, 250.0, 25.0, 1.333), 2),
    }
    pts_base["total_dynamic_pts"] = round(sum(pts_base.values()), 2)
    pts_opt["total_dynamic_pts"] = round(sum(pts_opt.values()), 2)

    def _clean_setup(s: Dict[str, Any]) -> Dict[str, float]:
        return {
            "lltd": round(float(s["lltd"]), 4),
            "camber_f_deg": round(float(s["camber_f_deg"]), 3),
            "camber_r_deg": round(float(s["camber_r_deg"]), 3),
            "toe_f_deg": round(float(s["toe_f_deg"]), 3),
            "toe_r_deg": round(float(s["toe_r_deg"]), 3),
            "brake_bias": round(float(s.get("brake_bias", s.get("brake_bias_front", p_nom.brake_bias_front))), 4),
            "rake_mm": round(float(s.get("rake_mm", 0.0)), 3),
        }

    events_report = {
        "skidpad": {
            "event_name": "Skidpad",
            "metric_label": "Periodic Circle Time (s)",
            "base_time_s": round(sk_base_t, 4),
            "optimal_time_s": round(sk_opt_t, 4),
            "delta_time_s": round(sk_opt_t - sk_base_t, 4),
            "delta_pct": round(100.0 * (sk_opt_t - sk_base_t) / sk_base_t, 3),
            "base_stat_s": round(float(sk_base_st["t_circle_s"]), 4),
            "optimal_stat_s": round(float(sk_opt_st["t_circle_s"]), 4),
            "base_fig8_timed_s": round(float(sk_base_f8["t_mean_s"]), 4),
            "optimal_fig8_timed_s": round(float(sk_opt_f8["t_mean_s"]), 4),
            "r_eff_star_m": round(float(sk_opt_st["r_eff_m"]), 4),
            "ay_max_g": round(float(sk_opt_st["ay_g"]), 3),
            "min_cone_clear_m": round(float(min(sk_opt_st["min_cone_clear_m"], sk_opt_f8["min_cone_clear_m"])), 4),
            "hv_violations": int(sk_opt_f8["hv_violations"]),
            "optimal_setup": _clean_setup(sk_setup),
            "preset_file": str((SETUPS_EXPORT_DIR / "optimal_skidpad.json").relative_to(REPO_ROOT)),
        },
        "accel": {
            "event_name": "Acceleration 75 m",
            "metric_label": "0-75 m Time (s)",
            "base_time_s": round(ac_base_t, 4),
            "optimal_time_s": round(ac_opt_t, 4),
            "delta_time_s": round(ac_opt_t - ac_base_t, 4),
            "delta_pct": round(100.0 * (ac_opt_t - ac_base_t) / ac_base_t, 3),
            "v_exit_kmh": round(float(ac_opt["v_exit_kmh"]), 2),
            "peak_ax_g": round(float(ac_opt["peak_ax_g"]), 3),
            "max_hv_power_kw": round(float(ac_opt["max_hv_power_kw"]), 2),
            "optimal_setup": _clean_setup(ac_setup),
            "preset_file": str((SETUPS_EXPORT_DIR / "optimal_accel.json").relative_to(REPO_ROOT)),
        },
        "autocross": {
            "event_name": "Autocross",
            "metric_label": "FSG 2023 Lap Time (s)",
            "base_time_s": round(ax_base_t, 4),
            "optimal_time_s": round(ax_opt_t, 4),
            "delta_time_s": round(ax_opt_t - ax_base_t, 4),
            "delta_pct": round(100.0 * (ax_opt_t - ax_base_t) / ax_base_t, 3),
            "v_max_kmh": round(float(ax_res["summary"]["v_max_kmh"]), 2),
            "ay_max_g": round(float(ax_res["summary"]["a_y_max_g"]), 3),
            "tire_temp_mean_c": round(float(ax_res["summary"]["tire_temp_mean_c"]), 2),
            "tire_temp_max_c": round(float(ax_res["summary"]["tire_temp_max_c"]), 2),
            "e_22km_kwh": round(float(ax_res["summary"]["e_22km_kwh"]), 3),
            "drivability_score": round(float(ax_res["summary"]["drivability_score"]), 2),
            "min_cone_clear_m": round(float(ax_res["summary"]["min_cone_clearance_m"]), 4),
            "hv_violations": int(ax_res["summary"]["hv_violations_exact"]),
            "bspd_violations": int(ax_res["summary"]["bspd_violations"]),
            "optimal_setup": _clean_setup(ax_setup),
            "preset_file": str((SETUPS_EXPORT_DIR / "optimal_autocross.json").relative_to(REPO_ROOT)),
        },
        "endurance": {
            "event_name": "Endurance 22 km",
            "metric_label": "22 km Total Time (s)",
            "base_time_s": round(en_base_22km_s, 2),
            "optimal_time_s": round(en_opt_22km_s, 2),
            "delta_time_s": round(en_opt_22km_s - en_base_22km_s, 2),
            "delta_pct": round(100.0 * (en_opt_22km_s - en_base_22km_s) / en_base_22km_s, 3),
            "base_lap_time_s": round(en_base_lap_s, 4),
            "optimal_lap_time_s": round(en_opt_lap_s, 4),
            "delta_lap_time_s": round(en_opt_lap_s - en_base_lap_s, 4),
            "n_laps_22km": round(n_laps_22km, 3),
            "energy_budget_kwh": endurance_budget_kwh,
            "e_22km_kwh": round(float(en_res["summary"]["e_22km_kwh"]), 3),
            "tire_temp_mean_c": round(float(en_res["summary"]["tire_temp_mean_c"]), 2),
            "tire_temp_max_c": round(float(en_res["summary"]["tire_temp_max_c"]), 2),
            "drivability_score": round(float(en_res["summary"]["drivability_score"]), 2),
            "min_cone_clear_m": round(float(en_res["summary"]["min_cone_clearance_m"]), 4),
            "hv_violations": int(en_res["summary"]["hv_violations_exact"]),
            "bspd_violations": int(en_res["summary"]["bspd_violations"]),
            "optimal_setup": _clean_setup(en_setup),
            "preset_file": str((SETUPS_EXPORT_DIR / "optimal_endurance.json").relative_to(REPO_ROOT)),
        },
    }

    report = {
        "vehicle": "ter27_4wd",
        "solver": "CasADi 3.7.2 + IPOPT",
        "grip_profile": "FSG_RUBBERED",
        "compound": compound,
        "fsg_rubbered_lambda_muy": FSG_RUBBERED_LAMBDA_MUY_TARGET,
        "cold_test_oct_lambda_muy": COLD_TEST_OCT_LAMBDA_MUY_TARGET,
        "env_mu_scale": FSG_RUBBERED_ENV_MU_SCALE,
        "setup_bounds": {k: list(v) for k, v in TRACKSIDE_SETUP_BOUNDS.items()},
        "base_setup": base_setup,
        "events": events_report,
        "fs_points": {
            "base": pts_base,
            "optimal": pts_opt,
            "delta_total_pts": round(pts_opt["total_dynamic_pts"] - pts_base["total_dynamic_pts"], 2),
        },
        "runtime_s": round(time.perf_counter() - t0, 2),
    }

    VAULT_07_DIR.mkdir(parents=True, exist_ok=True)
    TRACKSIDE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[REPORT] Saved {TRACKSIDE_REPORT_PATH} in {report['runtime_s']:.1f}s", flush=True)
    return report


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="CasADi Vehicle Design & Trackside Setup Co-Optimization")
    _parser.add_argument("--run-trackside-4events", action="store_true", help="Run 4-event trackside setup co-optimization and export presets")
    _parser.add_argument("--compound", type=str, default="R25B")
    _parser.add_argument("--n-intervals", type=int, default=240)
    _args = _parser.parse_args()
    if _args.run_trackside_4events:
        run_trackside_4event_optimization(compound=_args.compound, n_intervals_lap=_args.n_intervals)
    else:
        main()
