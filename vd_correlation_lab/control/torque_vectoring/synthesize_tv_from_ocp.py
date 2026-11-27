#!/usr/bin/env python3
"""
CasADi Robust Torque Vectoring Synthesis for TeR27 4WD (MirenaSim).

1. Solves clean human_achievable CasADi + IPOPT full-lap OCPs on FSG 2023 across
   3 grip/temperature conditions using the C^2 thermal-frictional model:
     - Cold asphalt:           mu_scale=0.72, temp_c=18 C
     - Optimal FSG_RUBBERED:   mu_scale=0.88, temp_c=45 C
     - Hot/degraded Endurance: mu_scale=0.82, temp_c=65 C (elevated tire bulk temp)
   Extracts target yaw-moment law M_z,TV*(vx, delta, ay, ax) and wheel torque split T*.
2. Simultaneously optimizes with IPOPT across all 3 conditions the real ECU gains
   and gain-scheduled tables (centripetal_fx_gain, delta_alpha_eq_gain, load weights
   w_load, understeer gradient k_us_deg_g, hydraulic brake_bias + -25 kW regen entry
   bias, and LQR gain schedule k_r/k_beta/k_beta_dot), maximizing speed and Drivability
   Score while guaranteeing directional stability dM_z/dbeta < 0 across all 3 conditions.
3. Evaluates on the genuine 400 Hz 14-DOF Godot simulator (Full14DofModel.gd) the 3x3
   matrix: Open-Diff vs Current TV vs CasADi Robust Synthesized TV across all 3
   grip/temperature conditions (0 cones hit, 0 HV violations).
4. Exports the winning calibration to:
     - vd_correlation_lab/results/casadi_suite/exported_tv_params.h
     - vd_correlation_lab/results/casadi_suite/tv_synthesis_report.json
"""

from __future__ import annotations

import concurrent.futures
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
TV_CFG_JSON = SIM_DIR / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter27_4wd" / "tv_4wd_config.json"
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
    steady_tire_temp_smooth,
    thermal_grip_multiplier_smooth,
)

# Chassis swept-path envelope (exact 4-corner check)
CAR_WIDTH_M = 1.42
CAR_HALF_W_M = 0.5 * CAR_WIDTH_M
CAR_L_FRONT_M = 1.15
CAR_L_REAR_M = 0.95

# Clean human_achievable limits
CONE_MARGIN_M = 0.22
MAX_STEER_RATE_RAD_S = math.radians(62.0)
MAX_BETA_RAD = math.radians(4.8)

# Strict HV bounds
P_DC_MAX_KW_HARD = 80.0
P_DC_MIN_KW_HARD = -25.0
P_DC_MAX_KW_NLP = 79.95
P_DC_MIN_KW_NLP = -24.95

# 3 operating conditions for robust multi-condition synthesis
CONDITIONS: List[Dict[str, Any]] = [
    {
        "id": "cold_asphalt",
        "label": "Cold Asphalt (mu=0.72, 18 C)",
        "grip_profile": "COLD_LOW_GRIP",
        "mu_scale": 0.72,
        "cl_scale_env": 0.64,
        "temp_c": 18.0,
        "tire_temp_c": 32.0,
        "sim_tire_init_c": 22.0,
        "weight": 0.28,
    },
    {
        "id": "optimal_fsg_rubbered",
        "label": "Optimal FSG_RUBBERED (mu=0.88, 45 C)",
        "grip_profile": "FSG_RUBBERED",
        "mu_scale": 0.88,
        "cl_scale_env": 0.78,
        "temp_c": 45.0,
        "tire_temp_c": 77.5,
        "sim_tire_init_c": 70.0,
        "weight": 0.44,
    },
    {
        "id": "hot_degraded_endurance",
        "label": "Hot/Degraded Endurance (mu=0.82, 65 C)",
        "grip_profile": "CUSTOM",
        "mu_scale": 0.82,
        "cl_scale_env": 0.78,
        "temp_c": 65.0,
        "tire_temp_c": 104.0,
        "sim_tire_init_c": 98.0,
        "weight": 0.28,
    },
]

VX_BREAKPOINTS_MS = np.array([4.0, 10.0, 16.0, 24.0, 32.0], dtype=np.float64)


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


def compute_drivability_score_from_csv(csv_path: Path, summary: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute 0-100 Drivability Score from 400 Hz (200 Hz logged) 14-DOF telemetry CSV.
    Combines steering smoothness, sideslip control, pedal cleanliness, and low torque fighting.
    """
    if not csv_path.exists():
        return {"drivability_score": 0.0, "steer_rate_rms_dps": 0.0, "pedal_reversals": 0, "beta_rms_deg": 0.0}
    ts: List[float] = []
    steer_deg: List[float] = []
    beta_deg: List[float] = []
    thr: List[float] = []
    brk: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["t_s"]))
            steer_deg.append(float(row["steer_deg"]))
            beta_deg.append(float(row["beta_deg"]))
            thr.append(float(row["throttle"]))
            brk.append(float(row["brake"]))
    if len(ts) < 20:
        return {"drivability_score": 0.0, "steer_rate_rms_dps": 0.0, "pedal_reversals": 0, "beta_rms_deg": 0.0}

    t_arr = np.asarray(ts, dtype=np.float64)
    st_arr = np.asarray(steer_deg, dtype=np.float64)
    b_arr = np.asarray(beta_deg, dtype=np.float64)
    th_arr = np.asarray(thr, dtype=np.float64)
    br_arr = np.asarray(brk, dtype=np.float64)

    dt_arr = np.maximum(np.diff(t_arr), 1e-4)
    st_dot = np.diff(st_arr) / dt_arr
    st_ddot = np.diff(st_dot) / dt_arr[:-1]
    steer_rate_rms = float(np.sqrt(np.mean(st_dot ** 2)))
    steer_jerk_rms = float(np.sqrt(np.mean(st_ddot ** 2)))

    beta_rms = float(np.sqrt(np.mean(b_arr ** 2)))
    max_beta = float(summary.get("max_abs_beta_deg", np.max(np.abs(b_arr))))
    ey_rms = float(summary.get("rms_ey_race_m", 0.2))
    fight_kj = float(summary.get("torque_fight_kj", 0.0))

    # Count distinct brake-to-throttle-to-brake micro-reversals (chatter)
    pedal_net = th_arr - br_arr
    d_ped = np.diff(pedal_net)
    sgn_ped = np.sign(np.where(np.abs(d_ped) > 0.04, d_ped, 0.0))
    nz = sgn_ped[sgn_ped != 0.0]
    reversals = int(np.sum(nz[1:] * nz[:-1] < 0.0)) if len(nz) > 2 else 0

    # Clean, bounded 0-100 composite score
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


def solve_condition_oracle_ocp(
    cond: Dict[str, Any],
    grid: Dict[str, Any],
    warm_start_w: Optional[np.ndarray] = None,
    ipopt_max_iter: int = 450,
) -> Dict[str, Any]:
    """
    Solve clean human_achievable CasADi + IPOPT full-lap OCP for a grip/temp condition
    and extract the exact oracle target M_z,TV* and wheel torque split T*.
    """
    t0 = time.perf_counter()
    cid = cond["id"]
    veh = build_casadi_vehicle(
        grip_profile=cond["grip_profile"],
        mu_scale=float(cond["mu_scale"]),
        temp_c=float(cond["temp_c"]),
        tire_temp_c=float(cond["tire_temp_c"]),
    )
    p = veh.params
    mass = float(p.mass)
    izz = float(p.izz_total)
    max_steer = float(p.max_steer_angle)
    n_int = grid["n_intervals"]
    m_pts = grid["m_pts"]
    ds = float(grid["ds"])
    kappa_c = grid["kappa"]
    w_left = grid["w_left"]
    w_right = grid["w_right"]

    # Build fast mapped physics function including Mz_tv_long, alpha_4, Fz_4, Fx_f_sin_delta
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_kap = ca.SX.sym("kappa_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    res_s = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay)
    out_keys = [
        "Fx_tot", "Fy_tot", "Mz_tot", "P_dc_ecu_kw", "P_dc_drv_kw",
        "T_motor_4", "T_max_4", "T_min_4", "Mz_tv_long", "alpha_4",
        "Fz_4", "Fx_f_sin_delta", "slip_power_4", "mu_thermal_4",
    ]
    fn_phys = ca.Function(
        f"tv_oracle_phys_{cid}",
        [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay],
        [res_s[k] for k in out_keys],
    )
    fn_phys_map = fn_phys.map(m_pts)

    S_VX, S_VY, S_WZ, S_N, S_XI = 20.0, 2.0, 1.5, 1.0, 0.3
    S_DEL, S_KAP, S_FBRK, S_AX, S_AY = 0.3, 0.1, 4000.0, 10.0, 10.0

    W = ca.MX.sym("W", 13, m_pts)
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

    (
        Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv,
        T_mot, T_max, T_min, Mz_tv, alpha_4,
        Fz_4, Fx_f_sin_d, slip_p_4, _mu_th,
    ) = fn_phys_map(vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt)

    kap_dm = ca.DM(kappa_c).T
    cos_xi = ca.cos(xi)
    sin_xi = ca.sin(xi)
    v_s = vx * cos_xi - vy * sin_xi
    sf = (1.0 - n_lat * kap_dm) / v_s

    dvx_ds = sf * (Fx_tot / mass + vy * wz)
    dvy_ds = sf * (Fy_tot / mass - vx * wz)
    dwz_ds = sf * (Mz_tot / izz)
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

    # Clean human_achievable control smoothness
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
    _add_g((ax_lt - Fx_tot / mass) / S_AX, 0.0, 0.0)
    _add_g((ay_lt - Fy_tot / mass) / S_AY, 0.0, 0.0)
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

    # Strict FSAE EV 2.3 BSPD complementarity
    T_mot_pos = 0.5 * (T_mot + ca.sqrt(T_mot * T_mot + 0.01))
    bspd_prod = ca.repmat(F_brk_hyd / 4000.0, 4, 1) * (T_mot_pos / 250.0)
    _add_g(bspd_prod, -np.inf, 1.5e-3)

    g_vec = ca.vertcat(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    lbw = np.zeros((13, m_pts), dtype=np.float64)
    ubw = np.zeros((13, m_pts), dtype=np.float64)
    lbw[0, :] = 3.5 / S_VX
    ubw[0, :] = 36.0 / S_VX
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

    lbx = lbw.reshape(-1, order="F")
    ubx = ubw.reshape(-1, order="F")

    if warm_start_w is not None:
        x0 = np.clip(warm_start_w.copy(), lbx, ubx)
    else:
        ws = build_warm_start(grid, RACING_LINE_REAL, mass, max_steer)
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
        x0 = np.clip(w0.reshape(-1, order="F"), lbx, ubx)

    nlp = {"x": ca.vec(W), "f": obj, "g": g_vec}
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
    solver = ca.nlpsol(f"tv_oracle_{cid}", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    x_opt = np.array(sol["x"].full()).ravel()
    w_opt = x_opt.reshape((13, m_pts), order="F")

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

    out_eval = fn_phys_map(
        vx_o[None, :], vy_o[None, :], wz_o[None, :], delta_o[None, :],
        kap4_o, fbrk_o[None, :], ax_o[None, :], ay_o[None, :],
    )
    P_ecu_o = np.array(out_eval[3].full()).ravel()
    P_drv_o = np.array(out_eval[4].full()).ravel()
    T_mot_o = np.array(out_eval[5].full())
    Mz_tv_o = np.array(out_eval[8].full()).ravel()
    alpha4_o = np.array(out_eval[9].full())
    Fz4_o = np.array(out_eval[10].full())
    Fx_f_sin_o = np.array(out_eval[11].full()).ravel()
    slip_p_o = np.array(out_eval[12].full())

    v_s_o = vx_o * np.cos(xi_o) - vy_o * np.sin(xi_o)
    sf_o = (1.0 - n_o * kappa_c) / v_s_o
    dt_int = (ds / 6.0) * (sf_o[idx_k] + 4.0 * sf_o[idx_m] + sf_o[idx_kp1])
    t_lap = float(np.sum(dt_int))
    dt_step = np.maximum(0.5 * (sf_o + np.roll(sf_o, -1)) * h_half, 1e-4)

    beta_o = np.arctan2(vy_o, np.maximum(vx_o, 0.5))
    beta_dot_o = (np.roll(beta_o, -1) - np.roll(beta_o, 1)) / (2.0 * dt_step)
    delta_dot_o = (np.roll(delta_o, -1) - np.roll(delta_o, 1)) / (2.0 * dt_step)
    wz_dot_o = (np.roll(wz_o, -1) - np.roll(wz_o, 1)) / (2.0 * dt_step)

    hv_viol = int(
        np.sum((P_ecu_o > P_DC_MAX_KW_HARD + 1e-6) | (P_ecu_o < P_DC_MIN_KW_HARD - 1e-6))
        + np.sum((P_drv_o > P_DC_MAX_KW_HARD + 1e-6) | (P_drv_o < P_DC_MIN_KW_HARD - 1e-6))
    )
    _, min_cl_m, n_hits = compute_4corner_clearance(n_o, xi_o, w_left, w_right)
    t_ss_mean = float(np.mean(steady_tire_temp_smooth(cond["temp_c"], slip_p_o, vx_o[None, :])))
    mu_th_val = float(thermal_grip_multiplier_smooth(cond["tire_temp_c"]))

    solve_s = time.perf_counter() - t0
    print(
        f"[TV-ORACLE:{cid}] {stats.get('return_status')} ({stats.get('iter_count')} iters, {solve_s:.1f}s) | "
        f"T_lap={t_lap:.3f}s | mu={cond['mu_scale']:.2f}*{mu_th_val:.3f}, T_tire={cond['tire_temp_c']:.1f}C "
        f"(T_ss={t_ss_mean:.1f}C) | Mz_tv_rms={np.sqrt(np.mean(Mz_tv_o**2)):.1f}Nm | "
        f"P_dc=[{np.min(P_ecu_o):.2f},{np.max(P_ecu_o):.2f}]kW (hv_viol={hv_viol}, hits={n_hits})",
        flush=True,
    )

    return {
        "condition": cond,
        "veh": veh,
        "status": str(stats.get("return_status")),
        "iterations": int(stats.get("iter_count", -1)),
        "solve_time_s": round(solve_s, 2),
        "ocp_lap_time_s": round(t_lap, 4),
        "mu_thermal": round(mu_th_val, 4),
        "t_tire_ss_mean_c": round(t_ss_mean, 2),
        "hv_violations": hv_viol,
        "cones_hit": n_hits,
        "min_cone_clearance_m": round(min_cl_m, 4),
        "x_opt": x_opt,
        "data": {
            "vx": vx_o,
            "vy": vy_o,
            "wz": wz_o,
            "wz_dot": wz_dot_o,
            "delta": delta_o,
            "delta_dot": delta_dot_o,
            "beta": beta_o,
            "beta_dot": beta_dot_o,
            "ax": ax_o,
            "ay": ay_o,
            "alpha_f": 0.5 * (alpha4_o[0, :] + alpha4_o[1, :]),
            "alpha_r": 0.5 * (alpha4_o[2, :] + alpha4_o[3, :]),
            "Fz_4": Fz4_o,
            "T_motor_4": T_mot_o,
            "F_brk_hyd": fbrk_o,
            "Mz_tv_star": Mz_tv_o,
            "Fx_f_sin_delta": Fx_f_sin_o,
            "P_dc_kw": P_ecu_o,
        },
        "trace": {
            "x_car": grid["x"] - n_o * np.sin(grid["psi"]),
            "y_car": grid["y"] + n_o * np.cos(grid["psi"]),
            "vx": vx_o,
            "ax": ax_o,
        },
    }


def compute_passive_yaw_stiffness_table(veh: TeR27CasADiVehicle) -> np.ndarray:
    """
    Compute passive tire directional stability derivative N_beta_passive = -dMz_ISO/dbeta_ISO
    (Nm/rad, < 0 is restoring/stable in Milliken/YMD convention) at each vx breakpoint
    using exact CasADi automatic differentiation.
    """
    p = veh.params
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_kap = ca.SX.sym("kap", 4)
    s_ax = ca.SX.sym("ax")
    s_ay = ca.SX.sym("ay")
    res = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, 0.0, s_ax, s_ay)
    # In ISO 8855, dMz_ISO/dbeta_ISO = (dMz_ISO/dvy) * vx.
    # Milliken directional stability derivative dM_z/dbeta = -dMz_ISO/dbeta_ISO (< 0 = stable).
    n_beta_sym = -ca.jacobian(res["Mz_tot"], s_vy) * s_vx
    fn_nb = ca.Function("fn_nb", [s_vx, s_vy, s_wz, s_del, s_kap, s_ax, s_ay], [n_beta_sym])

    out = np.zeros(len(VX_BREAKPOINTS_MS), dtype=np.float64)
    mu_th = float(thermal_grip_multiplier_smooth(veh.tire_temp_c)) if veh.tire_temp_c is not None else 1.0
    mu_eff = float(veh.env_mu_scale) * mu_th
    for i, vx_v in enumerate(VX_BREAKPOINTS_MS):
        # Evaluate at realistic steady cornering (R >= 7.0 m, 0.70*mu_eff*g)
        ay_v = min(0.70 * 9.81 * mu_eff, (vx_v * vx_v) / 7.0)
        wz_v = ay_v / max(vx_v, 3.0)
        del_v = min(p.wheelbase * wz_v / max(vx_v, 3.0) + 0.012, 0.30)
        vy_v = p.b_cog * wz_v - 0.022 * vx_v
        val = float(fn_nb(vx_v, vy_v, wz_v, del_v, np.full(4, 0.01), 0.0, ay_v))
        out[i] = val
    return out


def _pwl_interp_casadi(vx_arr: np.ndarray, bp: np.ndarray, y_sym: ca.SX) -> ca.SX:
    """Exact piecewise-linear interpolation of 5-breakpoint table y_sym at numpy array vx_arr."""
    n = len(vx_arr)
    m = len(bp)
    W_mat = np.zeros((n, m), dtype=np.float64)
    for k in range(n):
        x = float(vx_arr[k])
        if x <= bp[0]:
            W_mat[k, 0] = 1.0
        elif x >= bp[-1]:
            W_mat[k, -1] = 1.0
        else:
            j = int(np.searchsorted(bp, x) - 1)
            j = max(0, min(j, m - 2))
            u = (x - bp[j]) / (bp[j + 1] - bp[j])
            W_mat[k, j] = 1.0 - u
            W_mat[k, j + 1] = u
    return ca.mtimes(ca.DM(W_mat), y_sym)


def synthesize_robust_tv_ipopt(oracles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simultaneously optimize real TeR27 ECU gains & gain-scheduled tables across all 3
    grip/temperature conditions with CasADi + IPOPT while guaranteeing dM_z/dbeta < 0.
    """
    t0 = time.perf_counter()
    p = oracles[0]["veh"].params
    wb = float(p.wheelbase)
    izz = float(p.izz_total)
    tf = float(p.track_front)
    tr = float(p.track_rear)
    r_eff = float(p.wheel_radius)

    # Load current tv_4wd_config.json as prior/baseline
    cur_cfg = json.loads(TV_CFG_JSON.read_text(encoding="utf-8"))
    cur_ac = cur_cfg["modes"]["autocross"]
    cur_lqr = cur_cfg["lqr_gain_schedule"]

    # Passive stability derivatives across all 3 conditions (5 vx breakpoints each)
    nb_passive = {o["condition"]["id"]: compute_passive_yaw_stiffness_table(o["veh"]) for o in oracles}

    # Decision variables (25 real ECU parameters)
    cent_gain = ca.SX.sym("centripetal_fx_gain")
    da_gain = ca.SX.sym("delta_alpha_eq_gain")
    w_load_f = ca.SX.sym("w_load_f")
    w_load_r = ca.SX.sym("w_load_r")
    k_us_hi = ca.SX.sym("k_us_deg_g")
    k_us_lo = ca.SX.sym("k_us_low_deg_g")
    brake_bias = ca.SX.sym("brake_bias_front")
    tb_yaw_gain = ca.SX.sym("trail_brake_yaw_gain")
    ti_pitch_nm = ca.SX.sym("turnin_pitch_gain_nm")
    hp_pivot = ca.SX.sym("hairpin_pivot_gain")
    yaw_agil = ca.SX.sym("yaw_agility_gain")
    yaw_ff = ca.SX.sym("yaw_accel_ff_gain")
    d_lead = ca.SX.sym("delta_dot_lead_gain")
    slalom_damp = ca.SX.sym("slalom_beta_damp")
    exit_blend = ca.SX.sym("exit_traction_blend")
    k_r_tbl = ca.SX.sym("k_r", 5)
    k_b_tbl = ca.SX.sym("k_beta", 5)
    k_bd_tbl = ca.SX.sym("k_beta_dot", 5)

    x_sym = ca.vertcat(
        cent_gain, da_gain, w_load_f, w_load_r, k_us_hi, k_us_lo,
        brake_bias, tb_yaw_gain, ti_pitch_nm, hp_pivot, yaw_agil,
        yaw_ff, d_lead, slalom_damp, exit_blend,
        k_r_tbl, k_b_tbl, k_bd_tbl,
    )

    obj = 0.0
    per_cond_mz_err: Dict[str, ca.SX] = {}
    per_cond_tq_err: Dict[str, ca.SX] = {}

    for o in oracles:
        cid = o["condition"]["id"]
        w_c = float(o["condition"]["weight"])
        d = o["data"]
        m_pts = len(d["vx"])

        vx = d["vx"]
        wz = d["wz"]
        delta = d["delta"]
        delta_dot = d["delta_dot"]
        beta = d["beta"]
        beta_dot = d["beta_dot"]
        ax = d["ax"]
        ay = d["ay"]
        alpha_f = d["alpha_f"]
        alpha_r = d["alpha_r"]
        Fz_4 = d["Fz_4"]
        T_star = d["T_motor_4"]
        f_brk = d["F_brk_hyd"]
        Mz_star = d["Mz_tv_star"]
        Fx_f_sin = d["Fx_f_sin_delta"]

        # Blend factors matching EcuTorqueVectoring4Wd.gd
        u_kus = np.clip((vx - 10.5) / 9.0, 0.0, 1.0)
        u_hi_spd = np.clip((vx - 9.5) / 2.0, 0.0, 1.0)
        u_hi_spd = u_hi_spd * u_hi_spd * (3.0 - 2.0 * u_hi_spd)
        u_lo_spd = 1.0 - u_hi_spd
        u_hp = u_lo_spd * np.clip((np.abs(delta) - math.radians(8.0)) / math.radians(5.0), 0.0, 1.0)
        is_brk = np.clip((-ax - 3.5) / 4.0, 0.0, 1.0)
        is_tb = is_brk * np.clip((np.abs(delta) - 0.06) / 0.12, 0.0, 1.0)
        da_rel = np.maximum(np.abs(alpha_f) - np.abs(alpha_r) - 0.004, 0.0)
        sgn_d = np.tanh(delta / 0.02)

        # Gain-scheduled LQR gains evaluated at trajectory speeds
        kr_v = _pwl_interp_casadi(vx, VX_BREAKPOINTS_MS, k_r_tbl)
        kb_v = _pwl_interp_casadi(vx, VX_BREAKPOINTS_MS, k_b_tbl)
        kbd_v = _pwl_interp_casadi(vx, VX_BREAKPOINTS_MS, k_bd_tbl)

        # Understeer gradient k_us(vx) in rad/(m/s^2)
        kus_lo_rad = k_us_lo * (math.pi / (180.0 * 9.81))
        kus_hi_rad = k_us_hi * (math.pi / (180.0 * 9.81))
        kus_v = (1.0 - ca.DM(u_kus)) * kus_lo_rad + ca.DM(u_kus) * kus_hi_rad

        # Target yaw rate r_ref(vx, delta; k_us, hp_pivot)
        vx_dm = ca.DM(vx)
        del_dm = ca.DM(delta)
        r_std = (vx_dm * del_dm) / (wb + kus_v * vx_dm * vx_dm)
        r_hp = (vx_dm * ca.DM(np.tan(np.clip(delta, -0.50, 0.50)))) / wb * hp_pivot
        r_ref = (1.0 - ca.DM(u_hp)) * r_std + ca.DM(u_hp) * r_hp

        # Yaw moment control law Mz_ECU (matching EcuTorqueVectoring4Wd.gd structure)
        r_gain = vx_dm / (wb + kus_v * vx_dm * vx_dm)
        mz_ff = 0.35 * yaw_ff * yaw_agil * izz * r_gain * ca.DM(delta_dot) + ca.DM(u_hi_spd) * d_lead * ca.DM(delta_dot)
        mz_fb = (
            kr_v * yaw_agil * (r_ref - ca.DM(wz))
            + kb_v * ca.DM(beta)
            + kbd_v * ca.DM(beta_dot)
            + da_gain * ca.DM(da_rel * sgn_d * u_hi_spd)
            + ca.DM(is_tb * sgn_d) * tb_yaw_gain * 115.0
            - ca.DM(u_hi_spd * np.maximum(np.abs(wz) - np.abs(vx * np.tan(delta) / wb), 0.0) * np.tanh(wz / 0.1)) * slalom_damp * 0.25
        )
        mz_ecu = mz_ff + mz_fb

        # Wheel torque allocation surrogate (load-weighted + centripetal + brake_bias + regen entry)
        T_sum_star = np.sum(T_star, axis=0)  # total motor torque along trajectory
        fz_f_tot = np.maximum(Fz_4[0, :] + Fz_4[1, :], 200.0)
        fz_r_tot = np.maximum(Fz_4[2, :] + Fz_4[3, :], 200.0)
        w_f_eff = w_load_f * fz_f_tot
        w_r_eff = w_load_r * fz_r_tot
        f_share = w_f_eff / (w_f_eff + w_r_eff)

        # Centripetal front torque pull & regen entry pitch bias
        cent_pull = cent_gain * ca.DM(np.sin(np.abs(delta)) * np.clip(T_sum_star / 200.0, 0.0, 1.0))
        entry_pitch = -ca.DM(is_brk) * ti_pitch_nm * 0.35
        exit_rear = -ca.DM(np.clip(ax / 8.0, 0.0, 1.0)) * (exit_blend - 0.5) * 60.0
        T_f_tot = f_share * ca.DM(T_sum_star) + cent_pull + entry_pitch + exit_rear
        T_r_tot = ca.DM(T_sum_star) - T_f_tot

        # Left/right split from mz_ecu (BMz_FL/FR ~ +/- tf/(2*r), RL/RR ~ +/- tr/(2*r))
        mz_share_f = w_f_eff / (w_f_eff + w_r_eff)
        dT_f = (mz_share_f * mz_ecu * r_eff) / tf
        dT_r = ((1.0 - mz_share_f) * mz_ecu * r_eff) / tr
        T_ecu_0 = 0.5 * T_f_tot - dT_f
        T_ecu_1 = 0.5 * T_f_tot + dT_f
        T_ecu_2 = 0.5 * T_r_tot - dT_r
        T_ecu_3 = 0.5 * T_r_tot + dT_r

        # Hydraulic brake bias match on heavy braking zones
        brk_mask = (f_brk > 50.0).astype(np.float64)
        fz_front_frac_brk = fz_f_tot / np.maximum(fz_f_tot + fz_r_tot, 400.0)
        bb_err = ca.sum1(ca.DM(brk_mask) * ((brake_bias - ca.DM(fz_front_frac_brk)) ** 2)) / max(float(np.sum(brk_mask)), 1.0)

        # Speed-focused & cornering-weighted Mz tracking + wheel torque tracking + smoothness
        corner_w = 1.0 + 1.5 * np.clip(np.abs(ay) / 9.81, 0.0, 1.8)
        mz_res = mz_ecu - ca.DM(Mz_star)
        mz_mse = ca.sum1(ca.DM(corner_w) * (mz_res ** 2)) / float(m_pts)

        tq_mse = (
            ca.sum1((T_ecu_0 - ca.DM(T_star[0, :])) ** 2)
            + ca.sum1((T_ecu_1 - ca.DM(T_star[1, :])) ** 2)
            + ca.sum1((T_ecu_2 - ca.DM(T_star[2, :])) ** 2)
            + ca.sum1((T_ecu_3 - ca.DM(T_star[3, :])) ** 2)
        ) / float(4 * m_pts)

        # Smoothness (drivability) penalty on spatial Mz chatter dMz/ds
        dmz = mz_ecu[1:] - mz_ecu[:-1]
        mz_chatter = ca.sum1(dmz ** 2) / float(m_pts)

        per_cond_mz_err[cid] = ca.sqrt(mz_mse)
        per_cond_tq_err[cid] = ca.sqrt(tq_mse)

        obj += w_c * (
            (mz_mse / (120.0 ** 2))
            + 0.45 * (tq_mse / (80.0 ** 2))
            + 0.18 * (mz_chatter / (60.0 ** 2))
            + 2.5 * bb_err
        )

    # Mild physical prior regularization toward proven safe baseline gains
    prior_reg = (
        0.04 * ((cent_gain - 18.0) / 15.0) ** 2
        + 0.04 * ((da_gain - 2600.0) / 1000.0) ** 2
        + 0.06 * ((w_load_f - 1.0) / 0.25) ** 2
        + 0.06 * ((w_load_r - 1.0) / 0.25) ** 2
        + 0.04 * ((k_us_hi - 0.70) / 0.35) ** 2
        + 0.04 * ((k_us_lo - 0.14) / 0.15) ** 2
        + 0.05 * ((brake_bias - 0.62) / 0.08) ** 2
        + 0.05 * ((tb_yaw_gain - 1.05) / 0.40) ** 2
        + 0.04 * ((ti_pitch_nm - 280.0) / 80.0) ** 2
        + 0.05 * ((hp_pivot - 1.08) / 0.20) ** 2
        + 0.05 * ((yaw_agil - 1.05) / 0.20) ** 2
        + 0.04 * ((yaw_ff - 1.00) / 0.25) ** 2
        + 0.04 * ((d_lead - 35.0) / 20.0) ** 2
        + 0.04 * ((slalom_damp - 360.0) / 150.0) ** 2
        + 0.04 * ((exit_blend - 0.75) / 0.20) ** 2
        + 0.04 * ca.sum1(((k_r_tbl - ca.DM(cur_lqr["k_r"])) / 300.0) ** 2)
        + 0.04 * ca.sum1(((k_b_tbl - ca.DM(cur_lqr["k_beta"])) / 250.0) ** 2)
        + 0.04 * ca.sum1(((k_bd_tbl - ca.DM(cur_lqr["k_beta_dot"])) / 60.0) ** 2)
    )
    obj += prior_reg

    # Constraints:
    # 1. Load weights normalized: 0.5*(w_load_f + w_load_r) == 1.0
    # 2. Monotonicity of gain schedules k_r, k_beta, k_beta_dot with vx
    # 3. Hard directional stability across ALL 3 conditions & ALL 5 speeds:
    #    dM_z/dbeta = N_beta_passive(vx, c) - k_beta(vx) <= -200.0 Nm/rad < 0
    g_list: List[ca.SX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    g_list.append(0.5 * (w_load_f + w_load_r) - 1.0)
    lbg.append(0.0)
    ubg.append(0.0)

    for tbl in (k_r_tbl, k_b_tbl, k_bd_tbl):
        for i in range(4):
            g_list.append(tbl[i + 1] - tbl[i])
            lbg.append(10.0)
            ubg.append(500.0)

    # Understeer gradient monotonicity: k_us_hi >= k_us_lo
    g_list.append(k_us_hi - k_us_lo)
    lbg.append(0.05)
    ubg.append(1.0)

    # Hard directional stability dM_z/dbeta <= -200 Nm/rad across all 3 conditions
    for cid, nb_arr in nb_passive.items():
        for i in range(5):
            # Milliken stability derivative: N_beta_cl = N_beta_passive - k_beta(vx) < 0
            n_beta_cl = float(nb_arr[i]) - k_b_tbl[i]
            g_list.append(n_beta_cl)
            lbg.append(-1e6)
            ubg.append(-200.0)

    g_vec = ca.vertcat(*g_list)

    lbx = np.array([
        5.0, 600.0, 0.70, 0.70, 0.25, 0.06,
        0.54, 0.45, 160.0, 0.90, 0.90,
        0.75, 15.0, 220.0, 0.50,
        280.0, 450.0, 700.0, 900.0, 1050.0,
        140.0, 280.0, 450.0, 600.0, 740.0,
        35.0, 70.0, 100.0, 130.0, 160.0,
    ], dtype=np.float64)
    ubx = np.array([
        42.0, 3600.0, 1.30, 1.30, 1.05, 0.35,
        0.68, 1.80, 380.0, 1.35, 1.30,
        1.35, 65.0, 650.0, 0.95,
        550.0, 800.0, 1150.0, 1400.0, 1600.0,
        350.0, 560.0, 780.0, 980.0, 1150.0,
        90.0, 140.0, 190.0, 230.0, 270.0,
    ], dtype=np.float64)
    x0 = np.array([
        float(cur_ac.get("centripetal_fx_gain", 15.0)),
        float(cur_ac.get("delta_alpha_eq_gain", 2620.0)),
        1.0, 1.0,
        float(cur_ac.get("k_us_deg_g", 0.85)),
        float(cur_ac.get("k_us_low_deg_g", 0.12)),
        0.58,
        float(cur_ac.get("trail_brake_yaw_gain", 1.0)),
        float(cur_ac.get("turnin_pitch_gain_nm", 280.0)),
        float(cur_ac.get("hairpin_pivot_gain", 1.08)),
        float(cur_ac.get("yaw_agility_gain", 1.05)),
        float(cur_ac.get("yaw_accel_ff_gain", 1.0)),
        float(cur_ac.get("delta_dot_lead_gain", 35.0)),
        float(cur_ac.get("slalom_yaw_damp_gain", 340.0)),
        float(cur_ac.get("exit_traction_blend", 0.75)),
        *[float(v) for v in cur_lqr["k_r"]],
        *[float(v) for v in cur_lqr["k_beta"]],
        *[float(v) for v in cur_lqr["k_beta_dot"]],
    ], dtype=np.float64)
    x0 = np.clip(x0, lbx, ubx)

    nlp = {"x": x_sym, "f": obj, "g": g_vec}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.linear_solver": "mumps",
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-7,
    }
    solver = ca.nlpsol("tv_robust_synth", "ipopt", nlp, opts)
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=np.array(lbg), ubg=np.array(ubg))
    stats = solver.stats()
    xo = np.array(sol["x"].full()).ravel()

    # Evaluate tracking RMSEs at baseline vs optimized
    fn_err = ca.Function(
        "tv_err_eval",
        [x_sym],
        [per_cond_mz_err[c["id"]] for c in CONDITIONS] + [per_cond_tq_err[c["id"]] for c in CONDITIONS],
    )
    err_base = [float(v) for v in fn_err(x0)]
    err_opt = [float(v) for v in fn_err(xo)]

    kr_opt = [round(float(v), 2) for v in xo[15:20]]
    kb_opt = [round(float(v), 2) for v in xo[20:25]]
    kbd_opt = [round(float(v), 2) for v in xo[25:30]]

    # Directional stability verification table (dM_z/dbeta < 0 across all 3 conditions)
    stab_audit: Dict[str, Any] = {}
    all_stable = True
    for c in CONDITIONS:
        cid = c["id"]
        nb_p = nb_passive[cid]
        nb_cl = nb_p - np.array(kb_opt)
        max_nb = float(np.max(nb_cl))
        if max_nb >= 0.0:
            all_stable = False
        stab_audit[cid] = {
            "vx_breakpoints_ms": VX_BREAKPOINTS_MS.tolist(),
            "dMz_dbeta_passive_nm_per_rad": [round(float(v), 1) for v in nb_p],
            "dMz_dbeta_closed_loop_nm_per_rad": [round(float(v), 1) for v in nb_cl],
            "max_dMz_dbeta_nm_per_rad": round(max_nb, 1),
            "strictly_negative": bool(max_nb < 0.0),
        }

    synthesized = {
        "centripetal_fx_gain": round(float(xo[0]), 2),
        "delta_alpha_eq_gain": round(float(xo[1]), 1),
        "w_load": [round(float(xo[2]), 4), round(float(xo[2]), 4), round(float(xo[3]), 4), round(float(xo[3]), 4)],
        "k_us_deg_g": round(float(xo[4]), 4),
        "k_us_low_deg_g": round(float(xo[5]), 4),
        "brake_bias_front": round(float(xo[6]), 4),
        "trail_brake_yaw_gain": round(float(xo[7]), 3),
        "turnin_pitch_gain_nm": round(float(xo[8]), 1),
        "hairpin_pivot_gain": round(float(xo[9]), 3),
        "yaw_agility_gain": round(float(xo[10]), 3),
        "yaw_accel_ff_gain": round(float(xo[11]), 3),
        "delta_dot_lead_gain": round(float(xo[12]), 2),
        "slalom_beta_damp": round(float(xo[13]), 1),
        "exit_traction_blend": round(float(xo[14]), 3),
        "lqr_gain_schedule": {
            "vx_breakpoints_ms": VX_BREAKPOINTS_MS.tolist(),
            "k_r": kr_opt,
            "k_beta": kb_opt,
            "k_beta_dot": kbd_opt,
        },
        # Fixed safety-compliant parameters matching STM32 header
        "inner_regen_max_nm": round(float(cur_ac.get("hairpin_regen_max_nm", 95.0)), 1),
        "launch_rear_bias_pct": round(float(cur_ac.get("launch_rear_bias_pct", 70.0)), 1),
        "tc_slip_target": round(float(cur_ac.get("kappa_opt", 0.13)), 4),
        "friction_ellipse_margin": round(float(cur_ac.get("friction_ellipse_margin", 0.94)), 3),
        "skidpad_neutral_trim": 0.0,
        "beta_soft_deg": round(float(cur_ac.get("beta_soft_deg", 4.5)), 2),
        "power_limit_kw": 80.0,
        "regen_limit_kw": -25.0,
        "lift_regen_nm": 12.0,
        "arb_front_balance_pct": 45.0,
        "throttle_exp": round(float(cur_ac.get("throttle_corner_exp", 1.15)), 2),
    }

    tracking_summary: Dict[str, Any] = {}
    for idx, c in enumerate(CONDITIONS):
        cid = c["id"]
        tracking_summary[cid] = {
            "mz_rmse_baseline_nm": round(err_base[idx], 2),
            "mz_rmse_synthesized_nm": round(err_opt[idx], 2),
            "mz_rmse_reduction_pct": round(100.0 * (err_base[idx] - err_opt[idx]) / max(err_base[idx], 1e-6), 2),
            "torque_rmse_baseline_nm": round(err_base[3 + idx], 2),
            "torque_rmse_synthesized_nm": round(err_opt[3 + idx], 2),
        }

    print(
        f"[TV-SYNTH] IPOPT {stats.get('return_status')} ({stats.get('iter_count')} iters, {time.perf_counter()-t0:.2f}s) | "
        f"cent={synthesized['centripetal_fx_gain']}, da_eq={synthesized['delta_alpha_eq_gain']}, "
        f"k_us={synthesized['k_us_deg_g']}/{synthesized['k_us_low_deg_g']} deg/g, "
        f"bb={synthesized['brake_bias_front']:.4f}, tb_yaw={synthesized['trail_brake_yaw_gain']}, "
        f"w_load={synthesized['w_load']} | all_stable(dMz/dbeta<0)={all_stable}",
        flush=True,
    )

    return {
        "ipopt_status": str(stats.get("return_status")),
        "iterations": int(stats.get("iter_count", -1)),
        "synthesized_params": synthesized,
        "oracle_tracking_fit": tracking_summary,
        "directional_stability_audit": stab_audit,
        "all_conditions_directionally_stable": all_stable,
    }


def _build_14dof_racing_track(grid: Dict[str, Any], trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    xc = trace["x_car"]
    yc = trace["y_car"]
    vx = trace["vx"]
    ax = trace["ax"]
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
    x_u = cs_x(s_u)
    y_u = cs_y(s_u)
    dx_u = cs_x(s_u, 1)
    dy_u = cs_y(s_u, 1)
    ddx_u = cs_x(s_u, 2)
    ddy_u = cs_y(s_u, 2)
    psi_u = np.arctan2(dy_u, dx_u)
    spd_u = np.maximum(np.hypot(dx_u, dy_u), 1e-6)
    kap_u = (dx_u * ddy_u - dy_u * ddx_u) / (spd_u ** 3)
    v_u = np.clip(cs_v(s_u), 4.0, 36.0)
    a_u = np.clip(cs_a(s_u), -20.0, 15.0)

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
        "v_qss": [round(float(v), 4) for v in v_u],
        "ax_qss": [round(float(v), 4) for v in a_u],
    }


def _make_patched_runner_gd() -> str:
    """Patch lts_transient_runner.gd in memory to support custom env_mu_scale, temp, brake_bias, and LQR/w_load."""
    orig = RUNNER_GD_ORIG.read_text(encoding="utf-8")
    anchor_mass = '\tif cfg_json.has("mass_kg"):\n\t\tm.config.mass = float(cfg_json["mass_kg"])\n'
    extra_cfg = (
        anchor_mass
        + '\tif cfg_json.has("brake_bias_front"):\n'
        + '\t\tm.config.brake_bias_front = float(cfg_json["brake_bias_front"])\n'
        + '\tif cfg_json.has("track_temp_c"):\n'
        + '\t\tm.config.track_temp_c = float(cfg_json["track_temp_c"])\n'
        + '\tif cfg_json.has("tire_initial_temp_c"):\n'
        + '\t\tm.config.tire_initial_temp_c = float(cfg_json["tire_initial_temp_c"])\n'
    )
    anchor_ecu = '\t\tm.ecu_tv_4wd._mc_mode = -1\n\t\tm.ecu_tv_4wd.reset()\n'
    extra_ecu = (
        '\t\tif cfg_json.has("w_wheels") and cfg_json["w_wheels"] is Array:\n'
        + '\t\t\tvar ww: Array = cfg_json["w_wheels"]\n'
        + '\t\t\tfor wi in range(mini(4, ww.size())):\n'
        + '\t\t\t\tm.ecu_tv_4wd.w_wheels[wi] = float(ww[wi])\n'
        + '\t\tif cfg_json.has("lqr_gain_schedule") and cfg_json["lqr_gain_schedule"] is Dictionary:\n'
        + '\t\t\tvar lq: Dictionary = cfg_json["lqr_gain_schedule"]\n'
        + '\t\t\tif lq.has("k_r"): m.ecu_tv_4wd._copy_float_arr(lq["k_r"], m.ecu_tv_4wd.kr_tbl)\n'
        + '\t\t\tif lq.has("k_beta"): m.ecu_tv_4wd._copy_float_arr(lq["k_beta"], m.ecu_tv_4wd.kb_tbl)\n'
        + '\t\t\tif lq.has("k_beta_dot"): m.ecu_tv_4wd._copy_float_arr(lq["k_beta_dot"], m.ecu_tv_4wd.kbd_tbl)\n'
        + anchor_ecu
    )
    anchor_ret = '\treturn m\n'
    extra_env = (
        '\tif cfg_json.has("env_mu_scale"):\n'
        + '\t\tm.env_mu_scale = float(cfg_json["env_mu_scale"])\n'
        + '\tif cfg_json.has("env_aero_cl_scale"):\n'
        + '\t\tm.env_aero_cl_scale = float(cfg_json["env_aero_cl_scale"])\n'
        + '\tif cfg_json.has("tire_initial_temp_c") or cfg_json.has("track_temp_c"):\n'
        + '\t\tm._sync_wheel_thermal_config()\n'
        + anchor_ret
    )
    if anchor_mass not in orig or anchor_ecu not in orig or anchor_ret not in orig:
        raise RuntimeError("Could not patch lts_transient_runner.gd anchors")
    patched = orig.replace(anchor_mass, extra_cfg, 1)
    patched = patched.replace(anchor_ecu, extra_ecu, 1)
    patched = patched.replace(anchor_ret, extra_env, 1)
    patched = patched.replace(
        "\tvar max_time_s: float = float(n_laps) * 120.0 + 25.0\n",
        "\tvar max_time_s: float = float(n_laps) * 240.0 + 25.0\n",
        1,
    )
    return patched


def eval_14dof_matrix_cell(
    variant: str,
    cond: Dict[str, Any],
    trk_payload: Dict[str, Any],
    synth_params: Dict[str, Any],
    patched_gd: str,
) -> Dict[str, Any]:
    """
    Evaluate one (variant, condition) cell on the 400 Hz 14-DOF Godot simulator.
    Bisects alpha_drv to find the fastest strictly valid lap (0 cones hit, 0 HV violations).
    """
    cid = cond["id"]
    cell_lbl = f"{variant}_{cid}"

    with tempfile.TemporaryDirectory(prefix=f"mirena_tv14_{cell_lbl}_") as tmp:
        tmp_dir = Path(tmp)
        runner_tmp = tmp_dir / "tv_14dof_runner.gd"
        runner_tmp.write_text(patched_gd, encoding="utf-8")

        def _run_alpha(alpha_drv: float, write_csv: bool = False) -> Tuple[Dict[str, Any], Path]:
            job_path = tmp_dir / f"job_{alpha_drv:.4f}.json"
            out_path = tmp_dir / f"sum_{alpha_drv:.4f}.json"
            csv_path = (tmp_dir / f"trace_{alpha_drv:.4f}.csv") if write_csv else Path("")
            job: Dict[str, Any] = {
                "label": f"{cell_lbl}_a{alpha_drv:.3f}",
                "ecu_driving_mode": 0 if variant == "open_diff" else 2,
                "tv_4wd_mode": 0,
                "use_3d_kinematics": True,
                "alpha_drv": float(alpha_drv),
                "n_laps": 1,
                "csv_path": str(csv_path) if write_csv else "",
                "env_mu_scale": float(cond["mu_scale"]),
                "env_aero_cl_scale": float(cond["cl_scale_env"]),
                "track_temp_c": float(cond["temp_c"]),
                "tire_initial_temp_c": float(cond["sim_tire_init_c"]),
                **trk_payload,
            }
            if variant == "casadi_robust_tv":
                job["brake_bias_front"] = float(synth_params["brake_bias_front"])
                job["w_wheels"] = list(synth_params["w_load"])
                job["lqr_gain_schedule"] = dict(synth_params["lqr_gain_schedule"])
                job["mode_overrides"] = {
                    "autocross": {
                        "centripetal_fx_gain": float(synth_params["centripetal_fx_gain"]),
                        "delta_alpha_eq_gain": float(synth_params["delta_alpha_eq_gain"]),
                        "k_us_deg_g": float(synth_params["k_us_deg_g"]),
                        "k_us_low_deg_g": float(synth_params["k_us_low_deg_g"]),
                        "trail_brake_yaw_gain": float(synth_params["trail_brake_yaw_gain"]),
                        "trail_brake_pro_turn_gain": float(synth_params["trail_brake_yaw_gain"]),
                        "turnin_pitch_gain_nm": float(synth_params["turnin_pitch_gain_nm"]),
                        "hairpin_pivot_gain": float(synth_params["hairpin_pivot_gain"]),
                        "yaw_agility_gain": float(synth_params["yaw_agility_gain"]),
                        "yaw_accel_ff_gain": float(synth_params["yaw_accel_ff_gain"]),
                        "delta_dot_lead_gain": float(synth_params["delta_dot_lead_gain"]),
                        "slalom_yaw_damp_gain": float(synth_params["slalom_beta_damp"]),
                        "exit_traction_blend": float(synth_params["exit_traction_blend"]),
                    }
                }

            job_path.write_text(json.dumps(job), encoding="utf-8")
            cmd = [
                "godot", "--headless", "--path", str(SIM_DIR),
                "-s", str(runner_tmp), "--", str(job_path), str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"14-DOF failed for {cell_lbl}: {proc.stderr[-400:]}")
            return json.loads(out_path.read_text(encoding="utf-8")), csv_path

        def _is_valid(s: Dict[str, Any]) -> bool:
            p_max = float(s.get("p_dc_max_kw", 0.0))
            p_min = float(s.get("p_chg_min_kw", 0.0))
            hv_ok = (p_max <= P_DC_MAX_KW_HARD + 0.05) and (p_min >= P_DC_MIN_KW_HARD - 0.05)
            return (
                bool(s.get("valid", False))
                and int(s.get("cones_hit", 999)) == 0
                and int(s.get("corners_cones_hit", 999)) == 0
                and hv_ok
            )

        # Bisection over alpha_drv in [0.55, 0.99] (5 evals, writing CSV directly)
        lo, hi = 0.55, 0.99
        best_alpha = 0.55
        best_sum: Optional[Dict[str, Any]] = None
        best_csv: Optional[Path] = None

        for step in range(5):
            a_try = 0.80 if step == 0 else 0.5 * (lo + hi)
            s_try, csv_try = _run_alpha(a_try, write_csv=True)
            if _is_valid(s_try):
                if best_sum is None or float(s_try["mean_lap_time_s"]) < float(best_sum["mean_lap_time_s"]):
                    best_sum = s_try
                    best_alpha = a_try
                    best_csv = csv_try
                lo = a_try
            else:
                hi = a_try

        if best_sum is None:
            for a_fb in (0.50, 0.45, 0.40):
                s_fb, csv_fb = _run_alpha(a_fb, write_csv=True)
                if _is_valid(s_fb):
                    best_sum = s_fb
                    best_alpha = a_fb
                    best_csv = csv_fb
                    break

        if best_sum is None or best_csv is None:
            raise RuntimeError(f"No strictly valid 14-DOF lap found for {cell_lbl}")

        final_sum = best_sum
        drv_metrics = compute_drivability_score_from_csv(best_csv, final_sum)

    p_max = float(final_sum.get("p_dc_max_kw", 0.0))
    p_min = float(final_sum.get("p_chg_min_kw", 0.0))
    hv_viol = int((p_max > P_DC_MAX_KW_HARD + 0.05) or (p_min < P_DC_MIN_KW_HARD - 0.05))
    res = {
        "variant": variant,
        "condition": cid,
        "alpha_drv": round(best_alpha, 4),
        "lap_time_s": round(float(final_sum["mean_lap_time_s"]), 4),
        "drivability_score": drv_metrics["drivability_score"],
        "max_beta_deg": round(float(final_sum.get("max_abs_beta_deg", 0.0)), 3),
        "beta_rms_deg": drv_metrics["beta_rms_deg"],
        "steer_rate_rms_dps": drv_metrics["steer_rate_rms_dps"],
        "pedal_reversals": drv_metrics["pedal_reversals"],
        "cones_hit": int(final_sum.get("cones_hit", 0)),
        "corners_cones_hit": int(final_sum.get("corners_cones_hit", 0)),
        "p_dc_min_kw": round(p_min, 2),
        "p_dc_max_kw": round(p_max, 2),
        "hv_violations": hv_viol,
        "torque_fight_kj": round(float(final_sum.get("torque_fight_kj", 0.0)), 2),
    }
    print(
        f"[14-DOF-TV:{cell_lbl:38s}] T_lap={res['lap_time_s']:.3f}s | Drv={res['drivability_score']:.2f} | "
        f"max|beta|={res['max_beta_deg']:.2f}deg | cones={res['cones_hit']}/{res['corners_cones_hit']} | "
        f"P_dc=[{res['p_dc_min_kw']:.1f},{res['p_dc_max_kw']:.1f}]kW (hv_viol={res['hv_violations']})",
        flush=True,
    )
    return res


def export_stm32_header(synth_params: Dict[str, Any], out_path: Path) -> None:
    """
    Export winning TV calibration matching the exact STM32 header format of
    TracksideTuningController.gd / test_dil_features.gd.
    """
    ordered_keys = [
        ("hairpin_pivot_gain", float(synth_params["hairpin_pivot_gain"])),
        ("inner_regen_max_nm", float(synth_params["inner_regen_max_nm"])),
        ("trail_brake_yaw_gain", float(synth_params["trail_brake_yaw_gain"])),
        ("exit_traction_blend", float(synth_params["exit_traction_blend"])),
        ("launch_rear_bias_pct", float(synth_params["launch_rear_bias_pct"])),
        ("tc_slip_target", float(synth_params["tc_slip_target"])),
        ("friction_ellipse_margin", float(synth_params["friction_ellipse_margin"])),
        ("skidpad_neutral_trim", float(synth_params["skidpad_neutral_trim"])),
        ("yaw_agility_gain", float(synth_params["yaw_agility_gain"])),
        ("yaw_accel_ff_gain", float(synth_params["yaw_accel_ff_gain"])),
        ("centripetal_fx_gain", float(synth_params["centripetal_fx_gain"])),
        ("delta_alpha_eq_gain", float(synth_params["delta_alpha_eq_gain"])),
        ("delta_dot_lead_gain", float(synth_params["delta_dot_lead_gain"])),
        ("slalom_beta_damp", float(synth_params["slalom_beta_damp"])),
        ("beta_soft_deg", float(synth_params["beta_soft_deg"])),
        ("power_limit_kw", float(synth_params["power_limit_kw"])),
        ("regen_limit_kw", float(synth_params["regen_limit_kw"])),
        ("lift_regen_nm", float(synth_params["lift_regen_nm"])),
        ("brake_bias_front_pct", float(synth_params["brake_bias_front"]) * 100.0),
        ("arb_front_balance_pct", float(synth_params["arb_front_balance_pct"])),
        ("throttle_exp", float(synth_params["throttle_exp"])),
    ]
    lines = [
        "#ifndef TV_EXPORTED_PARAMS_H",
        "#define TV_EXPORTED_PARAMS_H",
        "",
        "/* Exported from MirenaSim CasADi Robust TV Synthesis | Active Map: Autocross | Preset: casadi_robust_3cond */",
    ]
    for key, val in ordered_keys:
        lines.append(f"#define TV_PARAM_{key.upper()} ({val:.4f}f)")

    lqr = synth_params["lqr_gain_schedule"]
    w_load = synth_params["w_load"]
    lines.extend([
        "",
        "/* CasADi Robust Multi-Condition Extended Calibration (Cold 18C / FSG_RUBBERED 45C / Hot 65C) */",
        f"#define TV_PARAM_K_US_DEG_G ({float(synth_params['k_us_deg_g']):.4f}f)",
        f"#define TV_PARAM_K_US_LOW_DEG_G ({float(synth_params['k_us_low_deg_g']):.4f}f)",
        f"#define TV_PARAM_TURNIN_PITCH_GAIN_NM ({float(synth_params['turnin_pitch_gain_nm']):.4f}f)",
        f"#define TV_PARAM_W_LOAD_FL ({float(w_load[0]):.4f}f)",
        f"#define TV_PARAM_W_LOAD_FR ({float(w_load[1]):.4f}f)",
        f"#define TV_PARAM_W_LOAD_RL ({float(w_load[2]):.4f}f)",
        f"#define TV_PARAM_W_LOAD_RR ({float(w_load[3]):.4f}f)",
        "static const float TV_LQR_VX_BP_MS[5] = { " + ", ".join(f"{v:.2f}f" for v in lqr["vx_breakpoints_ms"]) + " };",
        "static const float TV_LQR_K_R[5] = { " + ", ".join(f"{v:.2f}f" for v in lqr["k_r"]) + " };",
        "static const float TV_LQR_K_BETA[5] = { " + ", ".join(f"{v:.2f}f" for v in lqr["k_beta"]) + " };",
        "static const float TV_LQR_K_BETA_DOT[5] = { " + ", ".join(f"{v:.2f}f" for v in lqr["k_beta_dot"]) + " };",
        "",
        "#endif /* TV_EXPORTED_PARAMS_H */",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_intervals = 160
    print(f"[TV-SYNTH] Starting CasADi Robust TV Synthesis across 3 conditions (N={n_intervals})...", flush=True)
    grid = load_track_grid(TRACK_JSON, n_intervals=n_intervals)

    # 1. Solve clean CasADi OCP trajectories across all 3 grip/temperature conditions
    oracles: List[Dict[str, Any]] = []
    ws_prev: Optional[np.ndarray] = None
    # Solve optimal first to warm-start cold and hot
    order = [1, 0, 2]
    oracle_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx in order:
        res_o = solve_condition_oracle_ocp(CONDITIONS[idx], grid, warm_start_w=ws_prev)
        oracle_by_idx[idx] = res_o
        if idx == 1:
            ws_prev = res_o["x_opt"]
    oracles = [oracle_by_idx[0], oracle_by_idx[1], oracle_by_idx[2]]

    # 2. Simultaneous multi-condition IPOPT synthesis of ECU gains + tables with dM_z/dbeta < 0
    synth = synthesize_robust_tv_ipopt(oracles)
    synth_params = synth["synthesized_params"]

    # 3. Export STM32 header
    hdr_path = OUT_DIR / "exported_tv_params.h"
    export_stm32_header(synth_params, hdr_path)
    print(f"[TV-SYNTH] Exported STM32 header -> {hdr_path}", flush=True)

    # 4. Evaluate 3x3 matrix on 400 Hz 14-DOF Godot simulator (parallelized across 5 workers)
    print("\n[TV-SYNTH] Evaluating 3x3 matrix on 400 Hz 14-DOF Godot simulator...", flush=True)
    patched_gd = _make_patched_runner_gd()
    # Use clean human-trackable reference line (RACING_LINE_REAL) for fair 3x3 comparison
    rl_ref = json.loads(RACING_LINE_REAL.read_text(encoding="utf-8"))
    pts_ref = rl_ref["points"]
    ref_trace = {
        "x_car": np.array([float(pt["x"]) for pt in pts_ref], dtype=np.float64),
        "y_car": np.array([float(pt["y"]) for pt in pts_ref], dtype=np.float64),
        "vx": np.array([float(pt["v_mps"]) for pt in pts_ref], dtype=np.float64),
        "ax": np.array([float(pt["ax_mps2"]) for pt in pts_ref], dtype=np.float64),
    }
    trk_ref = _build_14dof_racing_track(grid, ref_trace)
    trk_by_cid = {c["id"]: trk_ref for c in CONDITIONS}
    variants = ["open_diff", "current_tv", "casadi_robust_tv"]
    tasks = [(var, c) for c in CONDITIONS for var in variants]

    matrix_results: Dict[str, Dict[str, Any]] = {c["id"]: {} for c in CONDITIONS}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {
            ex.submit(eval_14dof_matrix_cell, var, c, trk_by_cid[c["id"]], synth_params, patched_gd): (var, c["id"])
            for var, c in tasks
        }
        for fut in concurrent.futures.as_completed(futs):
            var, cid = futs[fut]
            matrix_results[cid][var] = fut.result()

    # Compute deltas & aggregate checks
    all_zero_cones = True
    all_zero_hv = True
    comparison_summary: Dict[str, Any] = {}
    for c in CONDITIONS:
        cid = c["id"]
        od = matrix_results[cid]["open_diff"]
        ct = matrix_results[cid]["current_tv"]
        rt = matrix_results[cid]["casadi_robust_tv"]
        for cell in (od, ct, rt):
            if cell["cones_hit"] != 0 or cell["corners_cones_hit"] != 0:
                all_zero_cones = False
            if cell["hv_violations"] != 0:
                all_zero_hv = False
        comparison_summary[cid] = {
            "delta_vs_open_diff_s": round(rt["lap_time_s"] - od["lap_time_s"], 4),
            "delta_vs_current_tv_s": round(rt["lap_time_s"] - ct["lap_time_s"], 4),
            "drivability_delta_vs_open_diff": round(rt["drivability_score"] - od["drivability_score"], 2),
            "drivability_delta_vs_current_tv": round(rt["drivability_score"] - ct["drivability_score"], 2),
        }

    report: Dict[str, Any] = {
        "track": grid["name"],
        "conditions": [
            {
                "id": o["condition"]["id"],
                "label": o["condition"]["label"],
                "mu_scale": o["condition"]["mu_scale"],
                "temp_c": o["condition"]["temp_c"],
                "tire_temp_c": o["condition"]["tire_temp_c"],
                "mu_thermal": o["mu_thermal"],
                "ocp_lap_time_s": o["ocp_lap_time_s"],
                "ocp_hv_violations": o["hv_violations"],
                "ocp_cones_hit": o["cones_hit"],
                "ocp_min_cone_clearance_m": o["min_cone_clearance_m"],
            }
            for o in oracles
        ],
        "synthesis": synth,
        "validation_14dof_matrix": matrix_results,
        "comparison_summary": comparison_summary,
        "checks": {
            "all_conditions_directionally_stable": synth["all_conditions_directionally_stable"],
            "all_14dof_zero_cones_hit": all_zero_cones,
            "all_14dof_zero_hv_violations": all_zero_hv,
            "stm32_header_exported": hdr_path.exists(),
        },
        "exported_header_path": str(hdr_path.relative_to(REPO_ROOT)),
        "total_runtime_s": round(time.perf_counter() - t0, 2),
    }

    rep_path = OUT_DIR / "tv_synthesis_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[TV-SYNTH] COMPLETE in {report['total_runtime_s']:.1f}s | saved {rep_path}", flush=True)
    return report


if __name__ == "__main__":
    main()
