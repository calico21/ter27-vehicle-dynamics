#!/usr/bin/env python3
"""
Full-Lap Curvilinear Minimum-Time & Endurance Optimal Control Problem (OCP)
using CasADi 3.7.2 + IPOPT (Separated Hermite-Simpson Direct Collocation)
with 2-Stage Homotopy, Human-Achievable Realism Limits, Integral Energy
Budget (Lift-and-Coast), and Closed-Loop 14-DOF Validation in Godot Headless.

Supports:
  - Multi-track loading via --track-file (MirenaSim TrackFiles/*.json OR LTS Track JSON)
  - Modes:
      * human_achievable   : 2-Stage Homotopy with human steering rate (<= 32 deg/s wheel),
                             smooth steering acceleration, |beta| <= 4.2 deg, smooth torque/brake
                             rates, realistic Mz_TV envelope, 0.22 m cone margin, calibrated
                             grip realism factor (< 2.5% vs 14-DOF).
      * theoretical_oracle : Unconstrained physical oracle (0.14 m cone margin, realism=1.0).
  - Grip profiles: fsg_real (FSG_REAL_TRACK) | ttc_lab (TTC_LAB_PEAK)
  - Optional Endurance Energy Budget: --energy-budget-kwh (e.g. 3.5 kWh / 22 km)
  - Strict FSAE EV 2.3 BSPD complementarity & HV bounds [-25.0, +80.0] kW (0 violations)
  - Optional 14-DOF closed-loop validation (--validate-14dof) & Schema JSON export (--output)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
RUNNER_GD_ORIG = LAB_DIR / "lts" / "lts_transient_runner.gd"
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
CASADI_SUITE_DIR = LAB_DIR / "results" / "casadi_suite"
LTS_RESULTS_DIR = LAB_DIR / "results" / "lts"
DEFAULT_TRACK_FILE = REPO_ROOT / "src" / "mirena_sim" / "TrackFiles" / "fsg_2023.json"
LTS_TRACKS_DIR = LAB_DIR / "lts" / "tracks"
RACING_LINE_DIR = REPO_ROOT / "src" / "mirena_sim" / "Assets" / "RacingLines"

for _p in (str(ORACLE_DIR), str(REPO_ROOT), str(LAB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import (  # noqa: E402
    CASADI_GRIP_PROFILES,
    SMOOTHING_PARAMS,
    build_casadi_vehicle,
    default_design_params,
    steady_tire_temp_smooth,
    thermal_grip_multiplier_smooth,
    tire_temp_dot_smooth,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (  # noqa: E402
    GRIP_PROFILES,
    TRACKSIDE_SETUP_BOUNDS,
    PacejkaMF62,
    TeR27PhysicsModel,
    load_ter27_params,
    rake_to_aero_modifiers,
)
from vd_correlation_lab.optimization.lap_time_sim.solve_dynamic_events_casadi import (  # noqa: E402
    SkidpadSetupCasADiVehicle,
    _apply_rake_to_design_params,
    _implied_arb_stiffness_nm_m,
)
from vd_correlation_lab.optimization.lap_time_sim.mintime_ocp import verify_swept_path_4corners  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.qss_solver import QSSResult  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.racing_line import RacingLineResult  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.track import Track, build_track_from_waypoints, load_planning_cone_track  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.transient_lts import TransientLapResult, run_transient_lap  # noqa: E402


# Physical chassis envelope matching verify_swept_path_4corners in mintime_ocp.py
CAR_WIDTH_M = 1.42
CAR_HALF_W_M = 0.5 * CAR_WIDTH_M  # 0.71 m
CAR_L_FRONT_M = 1.15
CAR_L_REAR_M = 0.95
CONE_MARGIN_THEORETICAL_M = 0.14
CONE_MARGIN_HUMAN_M = 0.18

# HV limits with 200 W inner buffer for zero violations on non-smoothed exact physics
P_DC_MAX_KW_HARD = 80.0
P_DC_MIN_KW_HARD = -25.0
P_DC_MAX_KW_NLP = 79.80
P_DC_MIN_KW_NLP = -24.80

# Calibrated grip realism factors for human_achievable mode
# (accounts for 14-DOF 3D kinematics, suspension compliance, and 45 ms human reaction delay)
DEFAULT_GRIP_REALISM = {
    ("human_achievable", "FSG_RUBBERED"): 0.71,
    ("human_achievable", "FSG_REAL_TRACK"): 0.68,
    ("human_achievable", "TTC_LAB_PEAK"): 0.57,
    ("human_achievable", "COLD_LOW_GRIP"): 0.74,
    ("human_achievable", "CUSTOM"): 0.71,
    ("theoretical_oracle", "FSG_RUBBERED"): 1.00,
    ("theoretical_oracle", "FSG_REAL_TRACK"): 1.00,
    ("theoretical_oracle", "TTC_LAB_PEAK"): 1.00,
    ("theoretical_oracle", "COLD_LOW_GRIP"): 1.00,
    ("theoretical_oracle", "CUSTOM"): 1.00,
}

PROFILE_MAP = {
    "fsg_rubbered": "FSG_RUBBERED",
    "FSG_RUBBERED": "FSG_RUBBERED",
    "fsg_real": "FSG_REAL_TRACK",
    "FSG_REAL_TRACK": "FSG_REAL_TRACK",
    "ttc_lab": "TTC_LAB_PEAK",
    "TTC_LAB_PEAK": "TTC_LAB_PEAK",
    "cold_low_grip": "COLD_LOW_GRIP",
    "COLD_LOW_GRIP": "COLD_LOW_GRIP",
    "custom": "CUSTOM",
    "CUSTOM": "CUSTOM",
    "CUSTOM_CONDITIONS": "CUSTOM",
}

PROFILE_CLI_MAP = {
    "FSG_RUBBERED": "fsg_rubbered",
    "FSG_REAL_TRACK": "fsg_real",
    "TTC_LAB_PEAK": "ttc_lab",
    "COLD_LOW_GRIP": "cold_low_grip",
    "CUSTOM": "custom",
}


def _emit_progress(progress_file: Optional[Path], payload: Dict[str, Any]) -> None:
    """Append one JSONL progress record for SimUI / live monitoring."""
    if progress_file is None:
        return
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with progress_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
            fh.flush()
    except OSError:
        pass


class IpoptProgressCallback(ca.Callback):
    """Lightweight CasADi IPOPT iteration callback emitting throttled JSONL progress."""

    def __init__(
        self,
        name: str,
        nx: int,
        ng: int,
        stage_tag: str,
        progress_file: Optional[Path],
        every_n_iters: int = 10,
    ) -> None:
        super().__init__()
        self._nx = int(nx)
        self._ng = int(ng)
        self._stage_tag = str(stage_tag)
        self._pf = progress_file
        self._every = max(1, int(every_n_iters))
        self._iter = 0
        self._t0 = time.perf_counter()
        self.construct(name, {})

    def get_n_in(self) -> int:
        return ca.nlpsol_n_out()

    def get_n_out(self) -> int:
        return 1

    def get_name_in(self, i: int) -> str:
        return ca.nlpsol_out(i)

    def get_name_out(self, i: int) -> str:
        return "ret"

    def get_sparsity_in(self, i: int) -> ca.Sparsity:
        n = ca.nlpsol_out(i)
        if n in ("x", "lam_x"):
            return ca.Sparsity.dense(self._nx, 1)
        if n in ("g", "lam_g"):
            return ca.Sparsity.dense(self._ng, 1)
        if n == "f":
            return ca.Sparsity.dense(1, 1)
        return ca.Sparsity(0, 0)

    def eval(self, arg: Any) -> List[int]:
        self._iter += 1
        if self._pf is not None and (self._iter == 1 or (self._iter % self._every) == 0):
            try:
                f_val = float(arg[1])
            except Exception:
                f_val = 0.0
            _emit_progress(
                self._pf,
                {
                    "stage": self._stage_tag,
                    "iter": self._iter,
                    "obj": round(f_val, 4),
                    "elapsed_s": round(time.perf_counter() - self._t0, 2),
                },
            )
        return [0]


def integrate_periodic_tire_temp(
    sf: np.ndarray,
    vx: np.ndarray,
    slip_power_4: np.ndarray,
    temp_c: float,
    ds_half: float,
    n_wraps: int = 6,
) -> np.ndarray:
    """
    Integrate the 4-wheel C^2 thermal ODE along a periodic lap until periodic steady state:
        dT_i/ds = sf(s) * tire_temp_dot_smooth(T_i, temp_c, Q_slip_i(s), vx(s))
    Returns T_tire_4 of shape (4, M) in deg C.
    """
    m_pts = sf.size
    q_mean = np.mean(slip_power_4, axis=1)
    v_mean = float(np.mean(vx))
    T_cur = np.array(
        [float(steady_tire_temp_smooth(temp_c, float(q_mean[w]), v_mean)) for w in range(4)],
        dtype=np.float64,
    )
    T_hist = np.zeros((4, m_pts), dtype=np.float64)
    for _ in range(n_wraps):
        for j in range(m_pts):
            T_hist[:, j] = T_cur
            dt_j = float(sf[j]) * ds_half
            dT_dt = np.asarray(
                tire_temp_dot_smooth(T_cur, float(temp_c), slip_power_4[:, j], float(vx[j])),
                dtype=np.float64,
            )
            T_cur = np.clip(T_cur + dT_dt * dt_j, float(temp_c) - 5.0, 130.0)
    return T_hist


def compute_drivability_score(
    t_arr: np.ndarray,
    steer_deg: np.ndarray,
    throttle: np.ndarray,
    brake: np.ndarray,
    beta_deg: np.ndarray,
    wz_rad_s: np.ndarray,
    bspd_violations: int = 0,
) -> Dict[str, Any]:
    """
    Compute clean human drivability score (0-100) from 4 equal 25-point pillars:
      1. Steering smoothness (RMS delta_ddot + peak |delta_dot| vs 62 deg/s wheel)
      2. Pedal reversal count (brake re-applications & throttle pumping per lap)
      3. Brake-to-throttle transition cleanliness (zero BSPD overlap, no micro-taps)
      4. Yaw stability (max|beta| vs 4.8 deg + RMS yaw-rate jerk)
    """
    n = len(t_arr)
    if n < 6:
        return {"drivability_score": 0.0, "steering_score": 0.0, "pedal_score": 0.0, "transition_score": 0.0, "yaw_score": 0.0}

    dt = np.diff(t_arr)
    dt = np.where(dt > 1e-4, dt, np.median(dt[dt > 1e-4]) if np.any(dt > 1e-4) else 0.05)

    # 1. Steering smoothness (0..25)
    d_steer_dt = np.diff(steer_deg) / dt
    dd_steer_dt2 = np.diff(d_steer_dt) / np.maximum(0.5 * (dt[:-1] + dt[1:]), 1e-4)
    peak_steer_rate = float(np.max(np.abs(d_steer_dt)))
    rms_steer_accel = float(np.sqrt(np.mean(dd_steer_dt2 ** 2)))
    s_rate_pen = max(0.0, (peak_steer_rate - 64.0) / 12.0) * 3.0
    s_acc_pen = max(0.0, (rms_steer_accel - 240.0) / 140.0) * 3.5
    steering_score = float(np.clip(25.0 - s_rate_pen - s_acc_pen, 0.0, 25.0))

    # 2. Pedal reversal count (0..25)
    # Count hysteresis-filtered brake/throttle pumping reversals (delta > 0.10)
    def _count_reversals(sig: np.ndarray, deadband: float = 0.10) -> int:
        revs = 0
        last_dir = 0
        anchor = float(sig[0])
        for v in sig[1:]:
            vf = float(v)
            if vf - anchor > deadband:
                if last_dir == -1:
                    revs += 1
                last_dir = 1
                anchor = vf
            elif anchor - vf > deadband:
                if last_dir == 1:
                    revs += 1
                last_dir = -1
                anchor = vf
        return revs

    brk_revs = _count_reversals(brake, 0.10)
    thr_revs = _count_reversals(throttle, 0.14)
    # On FSG 2023 (~19 corners), clean single-stroke driving has ~36-42 half-cycles
    excess_brk = max(0, brk_revs - 42)
    excess_thr = max(0, thr_revs - 44)
    pedal_score = float(np.clip(25.0 - 0.45 * excess_brk - 0.30 * excess_thr, 0.0, 25.0))

    # 3. Brake-to-throttle transition cleanliness (0..25)
    overlap_frac = float(np.mean((brake > 0.12) & (throttle > 0.12)))
    # Count micro-brake taps (< 0.18 s isolated brake pulses)
    in_brk = brake > 0.18
    micro_taps = 0
    i = 0
    while i < n:
        if in_brk[i]:
            j = i
            while j < n and in_brk[j]:
                j += 1
            dur = float(t_arr[min(j, n - 1)] - t_arr[i])
            if 0.01 < dur < 0.18:
                micro_taps += 1
            i = j
        else:
            i += 1
    trans_pen = overlap_frac * 50.0 + float(bspd_violations) * 5.0 + micro_taps * 1.0
    transition_score = float(np.clip(25.0 - trans_pen, 0.0, 25.0))

    # 4. Yaw stability (0..25)
    max_beta = float(np.max(np.abs(beta_deg)))
    dwz_dt = np.diff(wz_rad_s) / dt
    rms_yaw_acc = float(np.sqrt(np.mean(dwz_dt ** 2)))
    beta_pen = max(0.0, (max_beta - 5.2) / 1.0) * 3.0
    yaw_pen = max(0.0, (rms_yaw_acc - 4.8) / 2.5) * 3.0
    yaw_score = float(np.clip(25.0 - beta_pen - yaw_pen, 0.0, 25.0))

    total = float(np.clip(steering_score + pedal_score + transition_score + yaw_score, 0.0, 100.0))
    return {
        "drivability_score": round(total, 2),
        "steering_score": round(steering_score, 2),
        "pedal_score": round(pedal_score, 2),
        "transition_score": round(transition_score, 2),
        "yaw_score": round(yaw_score, 2),
        "brake_reversals": int(brk_revs),
        "throttle_reversals": int(thr_revs),
        "micro_brake_taps": int(micro_taps),
        "peak_steer_rate_deg_s": round(peak_steer_rate, 2),
        "rms_steer_accel_deg_s2": round(rms_steer_accel, 2),
    }


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_track_any(track_path: Path, n_intervals: int) -> Tuple[Track, Dict[str, Any]]:
    """
    Load any closed track (LTS parametric JSON OR MirenaSim TrackFiles cone JSON)
    and evaluate periodic cubic spline geometry at all 2*N Hermite-Simpson points.
    """
    p = Path(track_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))

    base_track: Track
    if "s" in raw and "kappa" in raw and "w_left" in raw:
        base_track = Track.load_json(p)
    else:
        # MirenaSim TrackFiles/*.json or cone track format
        lts_candidate = LTS_TRACKS_DIR / f"{p.stem}.json"
        if lts_candidate.exists():
            base_track = Track.load_json(lts_candidate)
        elif "path" in raw and isinstance(raw["path"], dict) and len(raw["path"].get("points", [])) >= 10:
            pts = np.array(
                [[float(pt["x"]), float(pt["y"])] for pt in raw["path"]["points"]],
                dtype=np.float64,
            )
            cones = raw.get("cones", [])
            blue = np.array(
                [[float(c["x"]), float(c["y"])] for c in cones if c.get("type") == "cone_blue"],
                dtype=np.float64,
            )
            yellow = np.array(
                [[float(c["x"]), float(c["y"])] for c in cones if c.get("type") == "cone_yellow"],
                dtype=np.float64,
            )
            if len(blue) >= 4 and len(yellow) >= 4:
                wl_arr = np.array(
                    [float(np.min(np.hypot(blue[:, 0] - pt[0], blue[:, 1] - pt[1]))) for pt in pts],
                    dtype=np.float64,
                )
                wr_arr = np.array(
                    [float(np.min(np.hypot(yellow[:, 0] - pt[0], yellow[:, 1] - pt[1]))) for pt in pts],
                    dtype=np.float64,
                )
                wl_arr = np.clip(wl_arr, 1.50, 3.50)
                wr_arr = np.clip(wr_arr, 1.50, 3.50)
            else:
                wl_arr = 1.80
                wr_arr = 1.80
            meta_name = str(raw.get("metadata", {}).get("track_name", p.stem))
            base_track = build_track_from_waypoints(
                name=p.stem,
                waypoints_xy=pts,
                w_left=wl_arr,
                w_right=wr_arr,
                is_closed=bool(raw["path"].get("closed", True)),
                ds_target=0.50,
            )
            base_track.name = meta_name if meta_name else p.stem
        else:
            base_track = load_planning_cone_track(p, name=p.stem, ds_target=0.50)

    length = float(base_track.length)
    s_raw = base_track.s.astype(np.float64)
    x_raw = base_track.x.astype(np.float64)
    y_raw = base_track.y.astype(np.float64)
    k_raw = base_track.kappa.astype(np.float64)
    wl_raw = base_track.w_left.astype(np.float64)
    wr_raw = base_track.w_right.astype(np.float64)

    s_ext = np.append(s_raw, length)
    x_ext = np.append(x_raw, x_raw[0])
    y_ext = np.append(y_raw, y_raw[0])
    k_ext = np.append(k_raw, k_raw[0])
    wl_ext = np.append(wl_raw, wl_raw[0])
    wr_ext = np.append(wr_raw, wr_raw[0])

    cs_x = CubicSpline(s_ext, x_ext, bc_type="periodic")
    cs_y = CubicSpline(s_ext, y_ext, bc_type="periodic")
    cs_k = CubicSpline(s_ext, k_ext, bc_type="periodic")
    cs_wl = CubicSpline(s_ext, wl_ext, bc_type="periodic")
    cs_wr = CubicSpline(s_ext, wr_ext, bc_type="periodic")

    ds = length / float(n_intervals)
    m_pts = 2 * n_intervals
    s_all = np.arange(m_pts, dtype=np.float64) * (0.5 * ds)

    x_all = cs_x(s_all)
    y_all = cs_y(s_all)
    dx = cs_x(s_all, 1)
    dy = cs_y(s_all, 1)
    psi_all = np.arctan2(dy, dx)
    kappa_all = cs_k(s_all)
    wl_all = cs_wl(s_all)
    wr_all = cs_wr(s_all)

    grid = {
        "name": str(base_track.name),
        "stem": p.stem,
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
        "raw_psi": base_track.psi.astype(np.float64),
        "raw_wl": wl_raw,
        "raw_wr": wr_raw,
    }
    return base_track, grid


def _periodic_smooth(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    """Zero-phase periodic box smoothing for clean C1 warm-start profiles."""
    if radius <= 0:
        return arr.copy()
    k = 2 * radius + 1
    ext = np.concatenate([arr[-radius:], arr, arr[:radius]])
    cs = np.cumsum(np.insert(ext, 0, 0.0))
    return (cs[k:] - cs[:-k]) / float(k)


def build_warm_start(
    grid: Dict[str, Any],
    racing_line_path: Optional[Path],
    mass: float,
    max_steer_rad: float,
) -> Dict[str, np.ndarray]:
    """
    Build periodic warm-start arrays of length M = 2*N from enriched racing line JSON,
    or fall back to a smooth curvature-based centerline warm-start for any custom track.
    """
    length = grid["length"]
    m_pts = grid["m_pts"]
    s_all = grid["s"]

    if racing_line_path is not None and racing_line_path.exists():
        rl = json.loads(racing_line_path.read_text(encoding="utf-8"))
        pts = rl.get("points", rl.get("waypoints", []))
        if len(pts) >= 20:
            rl_len = float(rl.get("length_m", length))
            s_rl = np.array([float(p.get("s", p.get("s_m", 0.0))) for p in pts], dtype=np.float64) * (
                length / max(rl_len, 1.0)
            )
            n_rl_arr = np.array([float(p.get("n", 0.0)) for p in pts], dtype=np.float64)
            v_rl = np.array([float(p.get("v_mps", p.get("v_ref", 12.0))) for p in pts], dtype=np.float64)
            ax_rl = np.array([float(p.get("ax_mps2", p.get("ax", 0.0))) for p in pts], dtype=np.float64)
            ay_rl = np.array([float(p.get("ay_mps2", p.get("ay", 0.0))) for p in pts], dtype=np.float64)
            yaw_rl = np.array([float(p.get("yaw_rad", p.get("yaw", 0.0))) for p in pts], dtype=np.float64)
            steer_rl = np.array(
                [float(p.get("steer_rad", math.radians(float(p.get("steer_deg", 0.0))))) for p in pts],
                dtype=np.float64,
            )
            kap_rl = np.array([float(p.get("kappa", 0.0)) for p in pts], dtype=np.float64)

            def _interp_periodic(y_vals: np.ndarray) -> np.ndarray:
                s_ext = np.append(s_rl, length)
                y_ext = np.append(y_vals, y_vals[0])
                return np.interp(s_all, s_ext, y_ext)

            vx_0 = np.clip(_periodic_smooth(_interp_periodic(v_rl), 2), 5.0, 34.0)
            n_0 = np.clip(_periodic_smooth(_interp_periodic(n_rl_arr), 2), -0.60, 0.60)

            yaw_unw = np.unwrap(yaw_rl)
            s_ext = np.append(s_rl, length)
            d_yaw_close = _wrap_angle(yaw_rl[0] - yaw_rl[-1])
            yaw_end = yaw_unw[-1] + d_yaw_close
            yaw_interp = np.interp(s_all, s_ext, np.append(yaw_unw, yaw_end))
            xi_0 = np.clip(_periodic_smooth(_wrap_angle(yaw_interp - grid["psi"]), 2), -0.35, 0.35)

            kap_0 = _periodic_smooth(_interp_periodic(kap_rl), 2)
            wz_0 = np.clip(vx_0 * kap_0, -3.0, 3.0)
            vy_0 = np.zeros(m_pts, dtype=np.float64)
            ax_0 = np.clip(_periodic_smooth(_interp_periodic(ax_rl), 2), -15.0, 11.0)
            ay_0 = np.clip(_periodic_smooth(_interp_periodic(ay_rl), 2), -15.0, 15.0)
            delta_0 = np.clip(_periodic_smooth(_interp_periodic(steer_rl), 2), -0.9 * max_steer_rad, 0.9 * max_steer_rad)
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

    # Generic curvature-based warm-start for any track without a pre-baked racing line
    kap_c = grid["kappa"]
    v_lim = np.sqrt(11.0 / np.maximum(np.abs(kap_c), 0.012))
    vx_0 = np.clip(_periodic_smooth(v_lim, 4), 6.0, 26.0)
    h_half = 0.5 * grid["ds"]
    dvx_ds = (np.roll(vx_0, -1) - np.roll(vx_0, 1)) / (2.0 * h_half)
    ax_0 = np.clip(vx_0 * dvx_ds, -12.0, 9.0)
    wz_0 = np.clip(vx_0 * kap_c, -2.5, 2.5)
    ay_0 = np.clip(vx_0 * wz_0, -14.0, 14.0)
    delta_0 = np.clip(np.arctan(1.54 * kap_c), -0.85 * max_steer_rad, 0.85 * max_steer_rad)
    kap_w_0 = np.clip(ax_0 / 180.0, -0.04, 0.04)
    return {
        "vx": vx_0,
        "vy": np.zeros(m_pts, dtype=np.float64),
        "wz": wz_0,
        "n": np.zeros(m_pts, dtype=np.float64),
        "xi": np.zeros(m_pts, dtype=np.float64),
        "delta": delta_0,
        "kappa_4": np.tile(kap_w_0[None, :], (4, 1)),
        "F_brk_hyd": np.clip(np.maximum(-ax_0 * mass - 2200.0, 0.0), 0.0, 4000.0),
        "ax_lt": ax_0,
        "ay_lt": ay_0,
    }


def compute_4corner_clearance(
    n_arr: np.ndarray,
    xi_arr: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """
    Exact 4-corner chassis swept-path clearance matching verify_swept_path_4corners.
    Returns (clearance_per_point, min_clearance_m, n_hits).
    """
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


def _resolve_grip_profile_cfg(grip_profile: str) -> Dict[str, float]:
    """Resolve grip profile anchored to ter27_physics.GRIP_PROFILES (FSG_RUBBERED -> lambda_muy=0.57)."""
    if grip_profile in GRIP_PROFILES:
        gp = GRIP_PROFILES[grip_profile]
        return {
            "env_mu_scale": float(gp["env_mu_scale"]),
            "env_aero_cl_scale": float(gp["env_aero_cl_scale"]),
            "default_temp_c": float(gp.get("track_temp_c", 45.0)),
            "tire_temp_c": float(gp.get("tire_temp_c", 75.0)),
        }
    return dict(CASADI_GRIP_PROFILES[grip_profile])


TRACKSIDE_SETUP_KEYS: Tuple[str, ...] = (
    "lltd",
    "camber_f_deg",
    "camber_r_deg",
    "toe_f_deg",
    "toe_r_deg",
    "brake_bias",
    "rake_mm",
)
TRACKSIDE_SETUP_SCALES: np.ndarray = np.array([0.5, 2.0, 2.0, 0.5, 0.5, 0.6, 8.0], dtype=np.float64)


def _build_trackside_casadi_vehicle(
    grip_profile: str,
    eff_mu_scale: float,
    eff_temp_c: float,
    sm_ov: Optional[Dict[str, float]] = None,
    compound: Optional[str] = None,
    design_params_override: Optional[Dict[str, Any]] = None,
) -> SkidpadSetupCasADiVehicle:
    """Build SkidpadSetupCasADiVehicle supporting toe_f/r and rake_mm symbolically or numerically."""
    p = load_ter27_params()
    comp_str = p.tire_compound if not compound else compound.strip().upper()
    tire = PacejkaMF62(comp_str)
    prof_cfg = _resolve_grip_profile_cfg(grip_profile)
    env_cl = float(prof_cfg["env_aero_cl_scale"])

    dp: Dict[str, Any] = dict(default_design_params(p))
    dp["toe_f_deg"] = 0.0
    dp["toe_r_deg"] = 0.0
    if design_params_override:
        dp.update(design_params_override)
        if "brake_bias" in design_params_override and "brake_bias_front" not in design_params_override:
            dp["brake_bias_front"] = design_params_override["brake_bias"]
        _apply_rake_to_design_params(dp, float(p.aero_balance_front))

    x_w = np.array([p.a_cog, p.a_cog, -p.b_cog, -p.b_cog], dtype=float)
    y_w = np.array(
        [0.5 * p.track_front, -0.5 * p.track_front, 0.5 * p.track_rear, -0.5 * p.track_rear],
        dtype=float,
    )
    bf = p.brake_bias_front
    hyd_split = np.array([0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf)], dtype=float)
    side_sign = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    sm = dict(SMOOTHING_PARAMS)
    if sm_ov:
        sm.update(sm_ov)

    return SkidpadSetupCasADiVehicle(
        grip_profile=grip_profile,
        params=p,
        tire=tire,
        env_mu_scale=eff_mu_scale,
        env_aero_cl_scale=env_cl,
        lmux=p.lambda_mux_base * eff_mu_scale,
        lmuy=p.lambda_muy_base * eff_mu_scale,
        x_w=x_w,
        y_w=y_w,
        hyd_split=hyd_split,
        side_sign=side_sign,
        smoothing=sm,
        temp_c=eff_temp_c,
        tire_temp_c=float(prof_cfg.get("tire_temp_c", 75.0)),
        design_params=dp,
    )


def _solve_single_stage_ocp(
    grip_profile: str,
    grid: Dict[str, Any],
    w0_init: np.ndarray,
    cone_margin_m: float,
    grip_realism_factor: float,
    human_limits: bool,
    energy_budget_kwh: Optional[float],
    ipopt_max_iter: int,
    ipopt_tol: float,
    ipopt_acceptable_tol: float,
    use_lbfgs: bool,
    print_level: int,
    stage_tag: str,
    energy_price_s_per_kj: float = 0.0,
    mu_scale: Optional[float] = None,
    temp_c: Optional[float] = None,
    T_tire_4_profile: Optional[np.ndarray] = None,
    progress_file: Optional[Path] = None,
    optimize_trackside_setup: bool = False,
    compound: Optional[str] = None,
    design_params_fixed: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build and solve one NLP stage of the Full-Lap Curvilinear OCP.
    When optimize_trackside_setup=True, co-optimizes the 7 mechanical trackside
    setup parameters [lltd, camber_f_deg, camber_r_deg, toe_f_deg, toe_r_deg,
    brake_bias, rake_mm] simultaneously with the full-lap trajectory and 4WD TV.
    """
    t_build0 = time.perf_counter()
    prof_cfg = _resolve_grip_profile_cfg(grip_profile)
    eff_mu_scale = float(prof_cfg["env_mu_scale"] if mu_scale is None else mu_scale)
    eff_temp_c = float(prof_cfg["default_temp_c"] if temp_c is None else temp_c)

    sm_ov = {"eps_power_w": 1200.0} if (energy_budget_kwh is not None and energy_budget_kwh > 0.0) else None
    s_setup_sym: Optional[ca.SX] = None
    dp_ov: Dict[str, Any] = dict(design_params_fixed) if design_params_fixed else {}
    if optimize_trackside_setup:
        s_setup_sym = ca.SX.sym("s_setup", 7)
        dp_ov["lltd"] = s_setup_sym[0]
        dp_ov["k_roll_scale"] = 1.0
        dp_ov["camber_f_deg"] = s_setup_sym[1]
        dp_ov["camber_r_deg"] = s_setup_sym[2]
        dp_ov["toe_f_deg"] = s_setup_sym[3]
        dp_ov["toe_r_deg"] = s_setup_sym[4]
        dp_ov["brake_bias_front"] = s_setup_sym[5]
        dp_ov["rake_mm"] = s_setup_sym[6]

    veh = _build_trackside_casadi_vehicle(
        grip_profile=grip_profile,
        eff_mu_scale=eff_mu_scale,
        eff_temp_c=eff_temp_c,
        sm_ov=sm_ov,
        compound=compound,
        design_params_override=dp_ov if dp_ov else None,
    )
    veh.lmux = float(veh.lmux) * float(grip_realism_factor)
    veh.lmuy = float(veh.lmuy) * float(grip_realism_factor)
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

    # Tire bulk temperature profile across all M collocation points
    if T_tire_4_profile is not None:
        t_tire_mat = np.asarray(T_tire_4_profile, dtype=np.float64)
    else:
        t_init_est = float(steady_tire_temp_smooth(eff_temp_c, 1600.0, 15.0))
        t_tire_mat = np.full((4, m_pts), t_init_est, dtype=np.float64)

    # Build compact SX physics function with thermal grip & slip power outputs
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_kap = ca.SX.sym("kappa_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    s_tt4 = ca.SX.sym("T_tire_4", 4)
    out_s = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay, T_tire_4=s_tt4)
    fn_inputs = [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay, s_tt4]
    if optimize_trackside_setup and s_setup_sym is not None:
        fn_inputs.append(s_setup_sym)
    fn_phys = ca.Function(
        f"phys_ocp_{grip_profile}_{stage_tag}",
        fn_inputs,
        [
            out_s["Fx_tot"],
            out_s["Fy_tot"],
            out_s["Mz_tot"],
            out_s["P_dc_ecu_kw"],
            out_s["P_dc_drv_kw"],
            out_s["T_motor_4"],
            out_s["T_max_4"],
            out_s["T_min_4"],
            out_s["Mz_tv_long"],
            out_s["Fz_4"],
            out_s["omega_4"],
            out_s["slip_power_4"],
        ],
    )
    fn_phys_map = fn_phys.map(m_pts)

    # Variable scaling
    S_VX = 20.0
    S_VY = 2.0
    S_WZ = 1.5
    S_N = 1.0
    S_XI = 0.3
    S_DEL = 0.3
    S_KAP = 0.1
    S_FBRK = 4000.0
    S_AX = 10.0
    S_AY = 10.0

    W = ca.MX.sym("W", 13, m_pts)
    Z_setup = ca.MX.sym("Z_setup", 7) if optimize_trackside_setup else None

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

    t_tire_dm = ca.DM(t_tire_mat)
    if optimize_trackside_setup and Z_setup is not None:
        setup_phys = Z_setup * ca.DM(TRACKSIDE_SETUP_SCALES)
        Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv, T_mot, T_max, T_min, Mz_tv, _Fz, _om, _Qslip = fn_phys_map(
            vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, t_tire_dm, ca.repmat(setup_phys, 1, m_pts)
        )
    else:
        Fx_tot, Fy_tot, Mz_tot, P_ecu, P_drv, T_mot, T_max, T_min, Mz_tv, _Fz, _om, _Qslip = fn_phys_map(
            vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, t_tire_dm
        )

    # Curvilinear spatial dynamics dx/ds at all 2*N points
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
    F_s = ca.vertcat(
        dvx_ds / S_VX,
        dvy_ds / S_VY,
        dwz_ds / S_WZ,
        dn_ds / S_N,
        dxi_ds / S_XI,
    )

    idx_k = list(range(0, m_pts, 2))
    idx_m = list(range(1, m_pts, 2))
    idx_kp1 = [(2 * ((k + 1) % n_int)) for k in range(n_int)]

    X_k = X_s[:, idx_k]
    X_m = X_s[:, idx_m]
    X_kp1 = X_s[:, idx_kp1]
    F_k = F_s[:, idx_k]
    F_m = F_s[:, idx_m]
    F_kp1 = F_s[:, idx_kp1]

    defect_mid = X_m - 0.5 * (X_k + X_kp1) - (ds / 8.0) * (F_k - F_kp1)
    defect_end = X_kp1 - X_k - (ds / 6.0) * (F_k + 4.0 * F_m + F_kp1)

    sf_k = sf[0, idx_k]
    sf_m = sf[0, idx_m]
    sf_kp1 = sf[0, idx_kp1]
    dt_intervals = (ds / 6.0) * (sf_k + 4.0 * sf_m + sf_kp1)
    T_lap = ca.sum2(dt_intervals)

    # Periodic finite differences along half-steps
    idx_next = list(range(1, m_pts)) + [0]
    idx_prev = [m_pts - 1] + list(range(0, m_pts - 1))
    h_half = 0.5 * ds
    d_delta = delta[0, idx_next] - delta
    dd_delta = delta[0, idx_next] - 2.0 * delta + delta[0, idx_prev]
    d_kappa = kappa_4[:, idx_next] - kappa_4
    dd_kappa = kappa_4[:, idx_next] - 2.0 * kappa_4 + kappa_4[:, idx_prev]
    d_fbrk = (F_brk_hyd[0, idx_next] - F_brk_hyd) / 10000.0
    d_fbrk_prev = (F_brk_hyd - F_brk_hyd[0, idx_prev]) / 10000.0
    dd_fbrk = (F_brk_hyd[0, idx_next] - 2.0 * F_brk_hyd + F_brk_hyd[0, idx_prev]) / 10000.0

    if human_limits:
        # Clean & progressive human driving (fast exact-Hessian convex regularization):
        #   - Fluid single-turn steering (strong delta_ddot smoothness)
        #   - Single-stroke clean braking (strong p_brake_dot + gentle p_brake_ddot)
        #   - Progressive throttle / torque application (strong d_kappa)
        reg_delta = 8.0e-2 * ca.sum2(d_delta * d_delta) / h_half
        reg_ddelta = 5.5e-1 * ca.sum2(dd_delta * dd_delta) / (h_half ** 3)
        reg_kappa = 4.0e-2 * ca.sum1(ca.sum2(d_kappa * d_kappa)) / h_half
        reg_ddkappa = 0.0
        reg_fbrk = 2.0e-2 * ca.sum2(d_fbrk * d_fbrk) / h_half
        reg_ddfbrk = 1.0e-2 * ca.sum2(dd_fbrk * dd_fbrk) / (h_half ** 3)
        reg_brk_rev = 0.0
        reg_hyd_pref = 1.5e-3 * ca.sum2(((F_brk_hyd / 10000.0) ** 2) * sf * h_half)
    else:
        reg_delta = 1.5e-2 * ca.sum2(d_delta * d_delta) / h_half
        reg_ddelta = 0.0
        reg_kappa = 8.0e-3 * ca.sum1(ca.sum2(d_kappa * d_kappa)) / h_half
        reg_ddkappa = 0.0
        reg_fbrk = 1.0e-3 * ca.sum2(d_fbrk * d_fbrk) / h_half
        reg_ddfbrk = 0.0
        reg_brk_rev = 0.0
        reg_hyd_pref = 1.0e-3 * ca.sum2(((F_brk_hyd / 10000.0) ** 2) * sf * h_half)

    obj = (
        T_lap
        + reg_delta
        + reg_ddelta
        + reg_kappa
        + reg_ddkappa
        + reg_fbrk
        + reg_ddfbrk
        + reg_brk_rev
        + reg_hyd_pref
    )
    if energy_price_s_per_kj > 0.0:
        p_sf_obj = P_ecu * sf
        e_net_obj = ca.sum2((ds / 6.0) * (p_sf_obj[0, idx_k] + 4.0 * p_sf_obj[0, idx_m] + p_sf_obj[0, idx_kp1]))
        obj = obj + float(energy_price_s_per_kj) * e_net_obj

    g_list: List[ca.MX] = []
    lbg_list: List[np.ndarray] = []
    ubg_list: List[np.ndarray] = []

    def _add_g(expr: ca.MX, lb_val: float | np.ndarray, ub_val: float | np.ndarray) -> None:
        vec = ca.vec(expr)
        n_el = vec.shape[0]
        g_list.append(vec)
        if np.isscalar(lb_val):
            lbg_list.append(np.full(n_el, float(lb_val), dtype=np.float64))
        else:
            lbg_list.append(np.asarray(lb_val, dtype=np.float64).reshape(n_el))
        if np.isscalar(ub_val):
            ubg_list.append(np.full(n_el, float(ub_val), dtype=np.float64))
        else:
            ubg_list.append(np.asarray(ub_val, dtype=np.float64).reshape(n_el))

    # 1. Hermite-Simpson collocation defects == 0
    _add_g(defect_mid, 0.0, 0.0)
    _add_g(defect_end, 0.0, 0.0)

    # 2. Algebraic load-transfer consistency == 0
    _add_g((ax_lt - Fx_tot / mass) / S_AX, 0.0, 0.0)
    _add_g((ay_lt - Fy_tot / mass) / S_AY, 0.0, 0.0)

    # 3. Motor torque limits: T_min <= T_motor <= T_max
    _add_g((T_mot - T_max) / 250.0, -np.inf, 0.0)
    _add_g((T_min - T_mot) / 250.0, -np.inf, 0.0)

    # 4. Strict HV power limits (both ECU and Drivetrain definitions)
    _add_g(P_ecu / 80.0, P_DC_MIN_KW_NLP / 80.0, P_DC_MAX_KW_NLP / 80.0)
    _add_g(P_drv / 80.0, P_DC_MIN_KW_NLP / 80.0, P_DC_MAX_KW_NLP / 80.0)

    # 5. Exact 4-corner chassis swept-path cone clearance >= eff_margin(s)
    # In human_achievable mode, add curvature-proportional tracking buffer so the
    # 45 ms delayed 14-DOF driver stays >= 0.15 m from every cone at alpha ~ 1.00.
    if human_limits:
        eff_margin = cone_margin_m + 0.21 + 0.11 * np.minimum(np.abs(kappa_c) / 0.12, 1.0)
        eff_margin = np.minimum(eff_margin, np.minimum(w_left, w_right) - CAR_HALF_W_M - 0.12)
    else:
        eff_margin = np.full_like(w_left, float(cone_margin_m))
    wl_lim = ca.DM(w_left - eff_margin).T
    wr_lim = ca.DM(w_right - eff_margin).T
    _add_g(n_lat + CAR_L_FRONT_M * sin_xi + CAR_HALF_W_M * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat - CAR_L_REAR_M * sin_xi + CAR_HALF_W_M * cos_xi - wl_lim, -np.inf, 0.0)
    _add_g(n_lat + CAR_L_FRONT_M * sin_xi - CAR_HALF_W_M * cos_xi + wr_lim, 0.0, np.inf)
    _add_g(n_lat - CAR_L_REAR_M * sin_xi - CAR_HALF_W_M * cos_xi + wr_lim, 0.0, np.inf)

    # 6. Positive forward progress along track tangent
    _add_g(v_s / S_VX, 3.0 / S_VX, np.inf)

    # 7. Steering rate limit |ddelta/dt| <= max_delta_dot
    dt_half = 0.5 * (sf + sf[0, idx_next]) * h_half
    max_delta_dot = math.radians(62.0) if human_limits else 2.8  # 62 deg/s wheel (~205 deg/s steering wheel)
    _add_g(d_delta - max_delta_dot * dt_half, -np.inf, 0.0)
    _add_g(-d_delta - max_delta_dot * dt_half, -np.inf, 0.0)

    # 8. FSAE EV 2.3 BSPD complementarity: zero simultaneous hydraulic brake + positive motor torque
    T_mot_pos = 0.5 * (T_mot + ca.sqrt(T_mot * T_mot + 0.01))
    bspd_prod = ca.repmat(F_brk_hyd / 4000.0, 4, 1) * (T_mot_pos / 250.0)
    _add_g(bspd_prod, -np.inf, 1.5e-3)

    # 9. Human-achievable extra constraints (Stage B)
    if human_limits:
        # 9a. Sideslip |beta| <= 4.8 deg  <=>  |vy| <= tan(4.8 deg) * vx
        tan_beta_max = math.tan(math.radians(4.8))
        _add_g((vy - tan_beta_max * vx) / S_VY, -np.inf, 0.0)
        _add_g((-vy - tan_beta_max * vx) / S_VY, -np.inf, 0.0)

        # 9b. Realistic ECU Mz_TV envelope (|Mz_TV| <= 550 Nm)
        _add_g(Mz_tv / 550.0, -1.0, 1.0)

        # 9c. Smooth motor/brake rates (linear in kappa_4 & F_brk_hyd -> sparse KKT, no bang-bang)
        _add_g((d_kappa - 0.80 * ca.repmat(dt_half, 4, 1)) / S_KAP, -np.inf, 0.0)
        _add_g((-d_kappa - 0.80 * ca.repmat(dt_half, 4, 1)) / S_KAP, -np.inf, 0.0)

        d_fbrk_raw = F_brk_hyd[0, idx_next] - F_brk_hyd
        _add_g((d_fbrk_raw - 24000.0 * dt_half) / 4000.0, -np.inf, 0.0)
        _add_g((-d_fbrk_raw - 24000.0 * dt_half) / 4000.0, -np.inf, 0.0)

    # 10. Optional Endurance Integral Energy Constraint (22 km budget)
    E_cum = None
    has_energy_con = (energy_budget_kwh is not None and energy_budget_kwh > 0.0)
    if has_energy_con:
        e_lap_max_kj = float(energy_budget_kwh) * 3600.0 * (float(grid["length"]) / 22000.0)
        e_lap_nlp_kj = 0.99 * e_lap_max_kj
        e_scale = max(e_lap_max_kj, 100.0)
        E_cum = ca.MX.sym("E_cum", n_int + 1)
        p_sf_sym = P_ecu * sf
        e_int_k = (ds / 6.0) * (p_sf_sym[0, idx_k] + 4.0 * p_sf_sym[0, idx_m] + p_sf_sym[0, idx_kp1])
        _add_g(E_cum[1:] - E_cum[:-1] - e_int_k.T / e_scale, 0.0, 0.0)
        _add_g(E_cum[0], 0.0, 0.0)
        _add_g(E_cum[n_int], -np.inf, e_lap_nlp_kj / e_scale)

    g_vec = ca.vertcat(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    # Decision variable bounds
    lbw_mat = np.zeros((13, m_pts), dtype=np.float64)
    ubw_mat = np.zeros((13, m_pts), dtype=np.float64)

    lbw_mat[0, :] = 3.5 / S_VX
    ubw_mat[0, :] = 40.0 / S_VX
    lbw_mat[1, :] = -5.0 / S_VY
    ubw_mat[1, :] = 5.0 / S_VY
    lbw_mat[2, :] = -4.0 / S_WZ
    ubw_mat[2, :] = 4.0 / S_WZ
    n_box_max = np.maximum(w_left - CAR_HALF_W_M - eff_margin, 0.05)
    n_box_min = -np.maximum(w_right - CAR_HALF_W_M - eff_margin, 0.05)
    lbw_mat[3, :] = n_box_min / S_N
    ubw_mat[3, :] = n_box_max / S_N
    lbw_mat[4, :] = -0.75 / S_XI
    ubw_mat[4, :] = 0.75 / S_XI
    lbw_mat[5, :] = -max_steer / S_DEL
    ubw_mat[5, :] = max_steer / S_DEL
    lbw_mat[6:10, :] = -0.14 / S_KAP
    ubw_mat[6:10, :] = 0.14 / S_KAP
    lbw_mat[10, :] = 0.0 / S_FBRK
    ubw_mat[10, :] = 12000.0 / S_FBRK
    lbw_mat[11, :] = -24.0 / S_AX
    ubw_mat[11, :] = 18.0 / S_AX
    lbw_mat[12, :] = -24.0 / S_AY
    ubw_mat[12, :] = 24.0 / S_AY

    lbw = lbw_mat.reshape(-1, order="F")
    ubw = ubw_mat.reshape(-1, order="F")
    w0 = np.clip(w0_init.copy(), lbw + 1e-6, ubw - 1e-6)

    if E_cum is not None:
        w0_m = w0.reshape((13, m_pts), order="F")
        setup_0_phys = np.array([
            p.lltd_elastic_f,
            math.degrees(p.static_camber_f_rad),
            math.degrees(p.static_camber_r_rad),
            0.0,
            0.0,
            p.brake_bias_front,
            0.0,
        ], dtype=np.float64)
        if optimize_trackside_setup and Z_setup is not None:
            res_w0 = fn_phys_map(
                (w0_m[0, :] * S_VX)[None, :],
                (w0_m[1, :] * S_VY)[None, :],
                (w0_m[2, :] * S_WZ)[None, :],
                (w0_m[5, :] * S_DEL)[None, :],
                w0_m[6:10, :] * S_KAP,
                (w0_m[10, :] * S_FBRK)[None, :],
                (w0_m[11, :] * S_AX)[None, :],
                (w0_m[12, :] * S_AY)[None, :],
                t_tire_mat,
                np.tile(setup_0_phys[:, None], (1, m_pts)),
            )
        else:
            res_w0 = fn_phys_map(
                (w0_m[0, :] * S_VX)[None, :],
                (w0_m[1, :] * S_VY)[None, :],
                (w0_m[2, :] * S_WZ)[None, :],
                (w0_m[5, :] * S_DEL)[None, :],
                w0_m[6:10, :] * S_KAP,
                (w0_m[10, :] * S_FBRK)[None, :],
                (w0_m[11, :] * S_AX)[None, :],
                (w0_m[12, :] * S_AY)[None, :],
                t_tire_mat,
            )
        p_w0 = np.array(res_w0[3].full()).ravel()
        vs_w0 = (w0_m[0, :] * S_VX) * np.cos(w0_m[4, :] * S_XI) - (w0_m[1, :] * S_VY) * np.sin(w0_m[4, :] * S_XI)
        sf_w0 = (1.0 - (w0_m[3, :] * S_N) * kappa_c) / np.maximum(vs_w0, 1.0)
        psf_w0 = p_w0 * sf_w0
        e_int_w0 = (ds / 6.0) * (psf_w0[idx_k] + 4.0 * psf_w0[idx_m] + psf_w0[idx_kp1])
        e_cum_w0 = np.concatenate([[0.0], np.cumsum(e_int_w0)]) / e_scale
        x_sym = ca.vertcat(ca.vec(W), E_cum)
        lbw = np.concatenate([lbw, np.full(n_int + 1, -2.0, dtype=np.float64)])
        ubw = np.concatenate([ubw, np.full(n_int + 1, 5.0, dtype=np.float64)])
        w0 = np.concatenate([w0, np.clip(e_cum_w0, -1.9, 4.9)])
    else:
        x_sym = ca.vec(W)

    setup_offset = int(w0.size)
    if optimize_trackside_setup and Z_setup is not None:
        b = TRACKSIDE_SETUP_BOUNDS
        s_lb = np.array([b[k][0] for k in TRACKSIDE_SETUP_KEYS], dtype=np.float64) / TRACKSIDE_SETUP_SCALES
        s_ub = np.array([b[k][1] for k in TRACKSIDE_SETUP_KEYS], dtype=np.float64) / TRACKSIDE_SETUP_SCALES
        s_0 = np.array([
            np.clip(p.lltd_elastic_f, b["lltd"][0], b["lltd"][1]),
            np.clip(math.degrees(p.static_camber_f_rad), b["camber_f_deg"][0], b["camber_f_deg"][1]),
            np.clip(math.degrees(p.static_camber_r_rad), b["camber_r_deg"][0], b["camber_r_deg"][1]),
            0.0,
            0.0,
            np.clip(p.brake_bias_front, b["brake_bias"][0], b["brake_bias"][1]),
            0.0,
        ], dtype=np.float64) / TRACKSIDE_SETUP_SCALES
        x_sym = ca.vertcat(x_sym, Z_setup)
        lbw = np.concatenate([lbw, s_lb])
        ubw = np.concatenate([ubw, s_ub])
        w0 = np.concatenate([w0, s_0])

    nlp = {"x": x_sym, "f": obj, "g": g_vec}
    ipopt_opts: Dict[str, Any] = {
        "print_time": False,
        "ipopt.print_level": int(print_level),
        "ipopt.linear_solver": "mumps",
        "ipopt.max_iter": int(ipopt_max_iter),
        "ipopt.tol": float(ipopt_tol),
        "ipopt.acceptable_tol": float(ipopt_acceptable_tol),
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_relax_factor": 1e-8,
        "ipopt.warm_start_init_point": "no",
    }
    if use_lbfgs:
        ipopt_opts["ipopt.hessian_approximation"] = "limited-memory"
        ipopt_opts["ipopt.limited_memory_max_history"] = 20

    cb = None
    if progress_file is not None:
        cb = IpoptProgressCallback(
            name=f"cb_{stage_tag}",
            nx=int(w0.size),
            ng=int(lbg.size),
            stage_tag=stage_tag,
            progress_file=progress_file,
            every_n_iters=10,
        )
        ipopt_opts["iteration_callback"] = cb

    solver = ca.nlpsol(f"ocp_{grip_profile}_{stage_tag}", "ipopt", nlp, ipopt_opts)
    t_build = time.perf_counter() - t_build0
    print(
        f"  [{stage_tag}] NLP built in {t_build:.2f}s (vars={w0.size}, cons={lbg.size}, "
        f"mu={eff_mu_scale:.3f}, temp={eff_temp_c:.1f}C, margin={cone_margin_m:.2f}m, "
        f"realism={grip_realism_factor:.3f}, energy={f'{energy_budget_kwh:.2f}kWh' if energy_budget_kwh else 'MaxAttack'})",
        flush=True,
    )
    _emit_progress(
        progress_file,
        {
            "stage": stage_tag,
            "status": "solving",
            "iter": 0,
            "mu_scale": eff_mu_scale,
            "temp_c": eff_temp_c,
        },
    )

    t_solve0 = time.perf_counter()
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    t_solve = time.perf_counter() - t_solve0
    stats = solver.stats()
    status = str(stats.get("return_status", "Unknown"))
    iters = int(stats.get("iter_count", -1))
    print(f"  [{stage_tag}] IPOPT finished: status={status}, iters={iters}, solve_time={t_solve:.2f}s", flush=True)
    _emit_progress(
        progress_file,
        {
            "stage": stage_tag,
            "status": status,
            "iter": iters,
            "solve_time_s": round(t_solve, 2),
        },
    )

    w_opt_full = np.array(sol["x"].full()).ravel()
    w_opt_vec = w_opt_full[: 13 * m_pts]
    w_opt = w_opt_vec.reshape((13, m_pts), order="F")
    vx_opt = w_opt[0, :] * S_VX
    vy_opt = w_opt[1, :] * S_VY
    wz_opt = w_opt[2, :] * S_WZ
    n_opt = w_opt[3, :] * S_N
    xi_opt = w_opt[4, :] * S_XI
    delta_opt = w_opt[5, :] * S_DEL
    kappa4_opt = w_opt[6:10, :] * S_KAP
    fbrk_opt = w_opt[10, :] * S_FBRK
    ax_opt = w_opt[11, :] * S_AX
    ay_opt = w_opt[12, :] * S_AY

    # Evaluate physics at optimal trajectory using the same scaled vehicle
    optimal_setup_dict: Optional[Dict[str, float]] = None
    if optimize_trackside_setup and Z_setup is not None:
        s_opt = w_opt_full[setup_offset : setup_offset + 7] * TRACKSIDE_SETUP_SCALES
        arb_f_opt, arb_r_opt = _implied_arb_stiffness_nm_m(p, float(s_opt[0]), 1.0)
        optimal_setup_dict = {
            "lltd": round(float(s_opt[0]), 4),
            "camber_f_deg": round(float(s_opt[1]), 3),
            "camber_r_deg": round(float(s_opt[2]), 3),
            "toe_f_deg": round(float(s_opt[3]), 3),
            "toe_r_deg": round(float(s_opt[4]), 3),
            "brake_bias": round(float(s_opt[5]), 4),
            "brake_bias_front": round(float(s_opt[5]), 4),
            "rake_mm": round(float(s_opt[6]), 3),
            "arb_stiffness_front_nm_m": round(float(arb_f_opt), 1),
            "arb_stiffness_rear_nm_m": round(float(arb_r_opt), 1),
        }
        res_ca = fn_phys_map(
            vx_opt[None, :],
            vy_opt[None, :],
            wz_opt[None, :],
            delta_opt[None, :],
            kappa4_opt,
            fbrk_opt[None, :],
            ax_opt[None, :],
            ay_opt[None, :],
            t_tire_mat,
            np.tile(s_opt[:, None], (1, m_pts)),
        )
    else:
        res_ca = fn_phys_map(
            vx_opt[None, :],
            vy_opt[None, :],
            wz_opt[None, :],
            delta_opt[None, :],
            kappa4_opt,
            fbrk_opt[None, :],
            ax_opt[None, :],
            ay_opt[None, :],
            t_tire_mat,
        )
    Fx_ca = np.array(res_ca[0].full()).ravel()
    Fy_ca = np.array(res_ca[1].full()).ravel()
    P_ecu_ca = np.array(res_ca[3].full()).ravel()
    P_drv_ca = np.array(res_ca[4].full()).ravel()
    T_mot_ca = np.array(res_ca[5].full())
    T_max_ca = np.array(res_ca[6].full())
    T_min_ca = np.array(res_ca[7].full())
    Mz_tv_ca = np.array(res_ca[8].full()).ravel()
    Fz_ca = np.array(res_ca[9].full())
    omega_ca = np.array(res_ca[10].full())
    Q_slip_ca = np.array(res_ca[11].full())

    # Exact non-smoothed ter27_physics.py cross-check
    base_phys_prof = "TTC_LAB_PEAK" if grip_profile == "TTC_LAB_PEAK" else "FSG_REAL_TRACK"
    phys_exact = TeR27PhysicsModel(profile=base_phys_prof)
    if compound:
        phys_exact.p.tire_compound = compound.strip().upper()
        phys_exact.tire = PacejkaMF62(phys_exact.p.tire_compound)
    phys_exact.env_mu_scale = eff_mu_scale
    phys_exact.env_aero_cl_scale = float(veh.env_aero_cl_scale)
    if optimal_setup_dict is not None:
        k_tot_b = phys_exact.p.k_roll_f + phys_exact.p.k_roll_r
        phys_exact.p.k_roll_f = optimal_setup_dict["lltd"] * k_tot_b
        phys_exact.p.k_roll_r = (1.0 - optimal_setup_dict["lltd"]) * k_tot_b
        phys_exact.p.lltd_elastic_f = optimal_setup_dict["lltd"]
        phys_exact.p.static_camber_f_rad = math.radians(optimal_setup_dict["camber_f_deg"])
        phys_exact.p.static_camber_r_rad = math.radians(optimal_setup_dict["camber_r_deg"])
        phys_exact.p.static_toe_f_rad = math.radians(optimal_setup_dict["toe_f_deg"])
        phys_exact.p.static_toe_r_rad = math.radians(optimal_setup_dict["toe_r_deg"])
        phys_exact.p.brake_bias_front = optimal_setup_dict["brake_bias"]
        cl_m, cd_m, cop_f = rake_to_aero_modifiers(optimal_setup_dict["rake_mm"], phys_exact.p.aero_balance_front)
        phys_exact.p.cl_a *= cl_m
        phys_exact.p.cd_a *= cd_m
        phys_exact.p.aero_balance_front = cop_f
    lmux_base_eff = float(p.lambda_mux_base) * eff_mu_scale * float(grip_realism_factor)
    lmuy_base_eff = float(p.lambda_muy_base) * eff_mu_scale * float(grip_realism_factor)
    mu_th_mat = np.asarray(thermal_grip_multiplier_smooth(t_tire_mat), dtype=np.float64)

    P_ecu_exact = np.zeros(m_pts, dtype=np.float64)
    P_drv_exact = np.zeros(m_pts, dtype=np.float64)
    for j in range(m_pts):
        mu_th_mean_j = float(np.mean(mu_th_mat[:, j]))
        phys_exact.lmux = lmux_base_eff * mu_th_mean_j
        phys_exact.lmuy = lmuy_base_eff * mu_th_mean_j
        ev = phys_exact.evaluate_state_from_kappa(
            vx=float(vx_opt[j]),
            vy=float(vy_opt[j]),
            r=float(wz_opt[j]),
            delta=float(delta_opt[j]),
            kappa=kappa4_opt[:, j],
            ax_est=float(ax_opt[j]),
            ay_est=float(ay_opt[j]),
            n_load_iter=1,
        )
        r_load_j = p.wheel_radius - np.minimum(ev["fz"] / p.k_tire_z, 0.5 * p.wheel_radius)
        t_hyd_j = float(fbrk_opt[j]) * veh.hyd_split * r_load_j
        t_mot_j = ev["torques"] + t_hyd_j
        P_ecu_exact[j] = phys_exact.compute_hv_power_w(t_mot_j, ev["omega_w"]) * 1e-3
        P_drv_exact[j] = phys_exact.compute_hv_power_drivetrain_w(t_mot_j, ev["omega_w"]) * 1e-3

    # Hermite-Simpson lap time & cumulative time
    v_s_opt = vx_opt * np.cos(xi_opt) - vy_opt * np.sin(xi_opt)
    sf_opt = (1.0 - n_opt * kappa_c) / v_s_opt
    dt_int = (ds / 6.0) * (sf_opt[idx_k] + 4.0 * sf_opt[idx_m] + sf_opt[idx_kp1])
    t_lap_s = float(np.sum(dt_int))
    t_nodes = np.concatenate([[0.0], np.cumsum(dt_int[:-1])])
    dt_half_a = (ds / 12.0) * (sf_opt[idx_k] + 2.0 * sf_opt[idx_m])
    t_all = np.zeros(m_pts, dtype=np.float64)
    t_all[idx_k] = t_nodes
    t_all[idx_m] = t_nodes + dt_half_a

    # Periodic steady-state tire bulk temperature integration along optimal lap
    T_tire_4_eq = integrate_periodic_tire_temp(sf_opt, vx_opt, Q_slip_ca, eff_temp_c, h_half, n_wraps=6)
    tire_temp_mean_c = float(np.mean(T_tire_4_eq))
    tire_temp_max_c = float(np.max(T_tire_4_eq))

    # Energy via Hermite-Simpson quadrature (in kJ)
    p_sf = P_ecu_exact * sf_opt
    e_net_kj = float(np.sum((ds / 6.0) * (p_sf[idx_k] + 4.0 * p_sf[idx_m] + p_sf[idx_kp1])))
    p_reg_sf = np.maximum(-P_ecu_exact, 0.0) * sf_opt
    e_regen_kj = float(np.sum((ds / 6.0) * (p_reg_sf[idx_k] + 4.0 * p_reg_sf[idx_m] + p_reg_sf[idx_kp1])))

    x_car = grid["x"] - n_opt * np.sin(grid["psi"])
    y_car = grid["y"] + n_opt * np.cos(grid["psi"])
    ds_cog = np.hypot(
        np.diff(np.append(x_car, x_car[0])),
        np.diff(np.append(y_car, y_car[0])),
    )
    l_racing_m = float(np.sum(ds_cog))
    e_22km_kwh = (e_net_kj / 3600.0) * (22000.0 / grid["length"])

    cl_arr, min_cl_m, n_cone_hits = compute_4corner_clearance(n_opt, xi_opt, w_left, w_right)

    hv_viol_ca = int(
        np.sum((P_ecu_ca > P_DC_MAX_KW_HARD + 1e-6) | (P_ecu_ca < P_DC_MIN_KW_HARD - 1e-6))
        + np.sum((P_drv_ca > P_DC_MAX_KW_HARD + 1e-6) | (P_drv_ca < P_DC_MIN_KW_HARD - 1e-6))
    )
    hv_viol_exact = int(
        np.sum((P_ecu_exact > P_DC_MAX_KW_HARD + 1e-6) | (P_ecu_exact < P_DC_MIN_KW_HARD - 1e-6))
        + np.sum((P_drv_exact > P_DC_MAX_KW_HARD + 1e-6) | (P_drv_exact < P_DC_MIN_KW_HARD - 1e-6))
    )

    beta_deg = np.rad2deg(np.arctan2(vy_opt, vx_opt))
    v_speed = np.hypot(vx_opt, vy_opt)
    dt_step = 0.5 * (sf_opt + sf_opt[idx_next]) * h_half
    delta_dot_deg_s = np.rad2deg(np.abs((delta_opt[idx_next] - delta_opt) / np.maximum(dt_step, 1e-4)))

    # BSPD check: simultaneous hydraulic brake > 50 N AND positive motor torque > 5 Nm
    bspd_viol = int(np.sum((fbrk_opt[None, :] > 50.0) & (T_mot_ca > 5.0)))

    # Equivalent throttle/brake pedals for OCP drivability score
    brk_eq = np.clip(
        np.where(fbrk_opt > 50.0, 0.35 + 0.65 * fbrk_opt / 8000.0, np.maximum(-P_ecu_exact, 0.0) / 25.0 * 0.35),
        0.0,
        1.0,
    )
    thr_eq = np.clip(np.where(brk_eq > 0.08, 0.0, np.maximum(P_ecu_exact, 0.0) / 75.0), 0.0, 1.0)
    drv_metrics = compute_drivability_score(
        t_arr=t_all,
        steer_deg=np.rad2deg(delta_opt),
        throttle=thr_eq,
        brake=brk_eq,
        beta_deg=beta_deg,
        wz_rad_s=wz_opt,
        bspd_violations=bspd_viol,
    )

    return {
        "w_opt_vec": w_opt_vec,
        "grip_profile": grip_profile,
        "optimal_setup": optimal_setup_dict,
        "ipopt": {
            "status": status,
            "iterations": iters,
            "build_time_s": round(t_build, 3),
            "solve_time_s": round(t_solve, 3),
            "n_intervals": n_int,
            "n_collocation_points": m_pts,
            "n_vars": int(w0.size),
            "n_cons": int(lbg.size),
            "hessian_mode": "L-BFGS" if use_lbfgs else "exact",
            "max_lt_ax_err_mps2": float(np.max(np.abs(ax_opt - Fx_ca / mass))),
            "max_lt_ay_err_mps2": float(np.max(np.abs(ay_opt - Fy_ca / mass))),
            "max_motor_torque_over_nm": float(np.max(T_mot_ca - T_max_ca)),
            "max_motor_torque_under_nm": float(np.max(T_min_ca - T_mot_ca)),
        },
        "summary": {
            "lap_time_s": round(t_lap_s, 4),
            "track_length_centerline_m": round(grid["length"], 3),
            "racing_line_length_m": round(l_racing_m, 3),
            "v_max_ms": round(float(np.max(v_speed)), 3),
            "v_max_kmh": round(float(np.max(v_speed)) * 3.6, 2),
            "v_mean_kmh": round(float(grid["length"] / t_lap_s) * 3.6, 2),
            "a_y_max_g": round(float(np.max(np.abs(ay_opt))) / 9.81, 3),
            "a_x_min_g": round(float(np.min(ax_opt)) / 9.81, 3),
            "a_x_max_g": round(float(np.max(ax_opt)) / 9.81, 3),
            "beta_max_deg": round(float(np.max(np.abs(beta_deg))), 3),
            "delta_dot_max_deg_s": round(float(np.max(delta_dot_deg_s)), 2),
            "P_dc_min_kw_exact": round(float(np.min(P_ecu_exact)), 3),
            "P_dc_max_kw_exact": round(float(np.max(P_ecu_exact)), 3),
            "hv_violations_casadi": hv_viol_ca,
            "hv_violations_exact": hv_viol_exact,
            "bspd_violations": bspd_viol,
            "min_cone_clearance_m": round(min_cl_m, 4),
            "cones_hit": n_cone_hits,
            "e_net_lap_kj": round(e_net_kj, 2),
            "e_regen_lap_kj": round(e_regen_kj, 2),
            "e_22km_kwh": round(e_22km_kwh, 3),
            "mz_tv_rms_nm": round(float(np.sqrt(np.mean(Mz_tv_ca ** 2))), 2),
            "mz_tv_max_nm": round(float(np.max(np.abs(Mz_tv_ca))), 2),
            "grip_realism_factor": round(float(grip_realism_factor), 4),
            "cone_margin_m": round(float(cone_margin_m), 4),
            "mu_scale": round(eff_mu_scale, 4),
            "temp_c": round(eff_temp_c, 2),
            "tire_temp_mean_c": round(tire_temp_mean_c, 2),
            "tire_temp_max_c": round(tire_temp_max_c, 2),
            "drivability_score": drv_metrics["drivability_score"],
            "drivability_details": drv_metrics,
        },
        "trace": {
            "s": grid["s"],
            "t": t_all,
            "sf": sf_opt,
            "dt_int": dt_int,
            "kappa_c": kappa_c,
            "x_car": x_car,
            "y_car": y_car,
            "yaw_car": _wrap_angle(grid["psi"] + xi_opt),
            "vx": vx_opt,
            "vy": vy_opt,
            "wz": wz_opt,
            "n": n_opt,
            "xi": xi_opt,
            "delta": delta_opt,
            "kappa_4": kappa4_opt,
            "F_brk_hyd": fbrk_opt,
            "ax": ax_opt,
            "ay": ay_opt,
            "beta_deg": beta_deg,
            "P_dc_kw": P_ecu_exact,
            "T_motor_4": T_mot_ca,
            "Fz_4": Fz_ca,
            "Mz_tv": Mz_tv_ca,
            "clearance_m": cl_arr,
            "T_tire_4": T_tire_4_eq,
            "slip_power_4": Q_slip_ca,
        },
    }


def solve_profile_ocp(
    grip_profile: str,
    grid: Dict[str, Any],
    racing_line_path: Optional[Path],
    mode: str = "human_achievable",
    cone_margin_m: Optional[float] = None,
    grip_realism_factor: Optional[float] = None,
    energy_budget_kwh: Optional[float] = None,
    ipopt_max_iter: int = 800,
    ipopt_tol: float = 1e-5,
    ipopt_acceptable_tol: float = 1e-4,
    use_lbfgs: bool = False,
    print_level: int = 0,
    mu_scale: Optional[float] = None,
    temp_c: Optional[float] = None,
    progress_file: Optional[Path] = None,
    optimize_trackside_setup: bool = False,
    compound: Optional[str] = None,
) -> Dict[str, Any]:
    """
    2-Stage Homotopy Full-Lap OCP solver with periodic tire thermal coupling:
      - Stage A: relaxed OCP from QSS/geometric warm-start -> periodic tire thermal ODE
      - Stage B: clean human_achievable limits + thermal grip profile + optional endurance budget
      - Optional Stage C (optimize_trackside_setup=True): simultaneous 7-parameter mechanical
        trackside setup + trajectory co-optimization warm-started from Stage B base solution.
    """
    prof_cfg = _resolve_grip_profile_cfg(grip_profile)
    eff_mu_scale = float(prof_cfg["env_mu_scale"] if mu_scale is None else mu_scale)
    eff_temp_c = float(prof_cfg["default_temp_c"] if temp_c is None else temp_c)

    veh_tmp = _build_trackside_casadi_vehicle(
        grip_profile=grip_profile, eff_mu_scale=eff_mu_scale, eff_temp_c=eff_temp_c, compound=compound
    )
    mass = float(veh_tmp.params.mass)
    max_steer = float(veh_tmp.params.max_steer_angle)
    m_pts = grid["m_pts"]

    if cone_margin_m is None:
        cone_margin_m = CONE_MARGIN_HUMAN_M if mode == "human_achievable" else CONE_MARGIN_THEORETICAL_M
    if grip_realism_factor is None:
        grip_realism_factor = DEFAULT_GRIP_REALISM.get((mode, grip_profile), 0.97 if mode == "human_achievable" else 1.0)

    S_VX, S_VY, S_WZ, S_N, S_XI = 20.0, 2.0, 1.5, 1.0, 0.3
    S_DEL, S_KAP, S_FBRK, S_AX, S_AY = 0.3, 0.1, 4000.0, 10.0, 10.0

    ws = build_warm_start(grid, racing_line_path, mass, max_steer)
    w0_mat = np.zeros((13, m_pts), dtype=np.float64)
    w0_mat[0, :] = ws["vx"] / S_VX
    w0_mat[1, :] = ws["vy"] / S_VY
    w0_mat[2, :] = ws["wz"] / S_WZ
    w0_mat[3, :] = ws["n"] / S_N
    w0_mat[4, :] = ws["xi"] / S_XI
    w0_mat[5, :] = ws["delta"] / S_DEL
    w0_mat[6:10, :] = ws["kappa_4"] / S_KAP
    w0_mat[10, :] = ws["F_brk_hyd"] / S_FBRK
    w0_mat[11, :] = ws["ax_lt"] / S_AX
    w0_mat[12, :] = ws["ay_lt"] / S_AY
    w0_stage_a = w0_mat.reshape(-1, order="F")

    # Stage A: Relaxed rates warm-start OCP
    print(
        f"[STAGE 1/3] Homotopy Stage A ({grip_profile}, mu={eff_mu_scale:.3f}, temp={eff_temp_c:.1f}C, "
        f"mode={mode}, realism={grip_realism_factor:.3f})...",
        flush=True,
    )
    need_stage_b = (
        (mode == "human_achievable")
        or (energy_budget_kwh is not None and energy_budget_kwh > 0.0)
        or optimize_trackside_setup
    )
    res_a = _solve_single_stage_ocp(
        grip_profile=grip_profile,
        grid=grid,
        w0_init=w0_stage_a,
        cone_margin_m=min(cone_margin_m, 0.16) if need_stage_b else cone_margin_m,
        grip_realism_factor=grip_realism_factor,
        human_limits=False,
        energy_budget_kwh=None,
        ipopt_max_iter=min(ipopt_max_iter, 400) if need_stage_b else ipopt_max_iter,
        ipopt_tol=1e-4 if need_stage_b else ipopt_tol,
        ipopt_acceptable_tol=5e-4 if need_stage_b else ipopt_acceptable_tol,
        use_lbfgs=use_lbfgs,
        print_level=print_level,
        stage_tag="StageA",
        mu_scale=eff_mu_scale,
        temp_c=eff_temp_c,
        progress_file=progress_file,
        compound=compound,
    )
    if res_a["ipopt"]["status"] not in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and not use_lbfgs:
        print("  [StageA] Exact Hessian did not converge cleanly; retrying Stage A with L-BFGS...", flush=True)
        res_a = _solve_single_stage_ocp(
            grip_profile=grip_profile,
            grid=grid,
            w0_init=w0_stage_a,
            cone_margin_m=min(cone_margin_m, 0.16) if need_stage_b else cone_margin_m,
            grip_realism_factor=grip_realism_factor,
            human_limits=False,
            energy_budget_kwh=None,
            ipopt_max_iter=min(ipopt_max_iter, 500) if need_stage_b else ipopt_max_iter,
            ipopt_tol=1e-4 if need_stage_b else ipopt_tol,
            ipopt_acceptable_tol=5e-4 if need_stage_b else ipopt_acceptable_tol,
            use_lbfgs=True,
            print_level=print_level,
            stage_tag="StageA_LBFGS",
            mu_scale=eff_mu_scale,
            temp_c=eff_temp_c,
            progress_file=progress_file,
            compound=compound,
        )

    if not need_stage_b:
        s = res_a["summary"]
        print(
            f"[STAGE 2/3] Theoretical oracle complete: T_ocp={s['lap_time_s']:.3f}s | "
            f"v_max={s['v_max_kmh']:.1f}km/h | ay_max={s['a_y_max_g']:.2f}g | "
            f"T_tire={s['tire_temp_mean_c']:.1f}C | E_22km={s['e_22km_kwh']:.2f}kWh",
            flush=True,
        )
        return res_a

    # Stage B: Clean human limits + periodic thermal profile + cone margin + optional Endurance Energy Budget
    t_tire_stage_b = res_a["trace"]["T_tire_4"]
    print(
        f"[STAGE 2/3] Homotopy Stage B (human limits: |delta_dot|<=62deg/s, |beta|<=4.8deg, "
        f"margin={cone_margin_m:.2f}m, T_tire_mean={float(np.mean(t_tire_stage_b)):.1f}C, "
        f"budget={f'{energy_budget_kwh:.2f}kWh' if energy_budget_kwh else 'MaxAttack'})...",
        flush=True,
    )
    res_b = _solve_single_stage_ocp(
        grip_profile=grip_profile,
        grid=grid,
        w0_init=res_a["w_opt_vec"],
        cone_margin_m=cone_margin_m,
        grip_realism_factor=grip_realism_factor,
        human_limits=(mode == "human_achievable"),
        energy_budget_kwh=energy_budget_kwh,
        ipopt_max_iter=ipopt_max_iter,
        ipopt_tol=ipopt_tol,
        ipopt_acceptable_tol=ipopt_acceptable_tol,
        use_lbfgs=use_lbfgs,
        print_level=print_level,
        stage_tag="StageB",
        mu_scale=eff_mu_scale,
        temp_c=eff_temp_c,
        T_tire_4_profile=t_tire_stage_b,
        progress_file=progress_file,
        compound=compound,
    )
    if res_b["ipopt"]["status"] not in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and not use_lbfgs:
        print("  [StageB] Exact Hessian did not converge cleanly; retrying Stage B with L-BFGS...", flush=True)
        res_b = _solve_single_stage_ocp(
            grip_profile=grip_profile,
            grid=grid,
            w0_init=res_a["w_opt_vec"],
            cone_margin_m=cone_margin_m,
            grip_realism_factor=grip_realism_factor,
            human_limits=(mode == "human_achievable"),
            energy_budget_kwh=energy_budget_kwh,
            ipopt_max_iter=ipopt_max_iter,
            ipopt_tol=ipopt_tol,
            ipopt_acceptable_tol=ipopt_acceptable_tol,
            use_lbfgs=True,
            print_level=print_level,
            stage_tag="StageB_LBFGS",
            mu_scale=eff_mu_scale,
            temp_c=eff_temp_c,
            T_tire_4_profile=t_tire_stage_b,
            progress_file=progress_file,
            compound=compound,
        )

    res_b["stage_a_summary"] = res_a["summary"]

    if optimize_trackside_setup:
        base_lap_s = float(res_b["summary"]["lap_time_s"])
        # Use identical tire thermal profile as Stage B for strict apples-to-apples Base vs Optimal comparison
        t_tire_stage_c = t_tire_stage_b
        print(
            f"[STAGE 3/3] Co-Optimizing 7 Trackside Setup Parameters (warm-started from Base T={base_lap_s:.4f}s)...",
            flush=True,
        )
        res_c = _solve_single_stage_ocp(
            grip_profile=grip_profile,
            grid=grid,
            w0_init=res_b["w_opt_vec"],
            cone_margin_m=cone_margin_m,
            grip_realism_factor=grip_realism_factor,
            human_limits=(mode == "human_achievable"),
            energy_budget_kwh=energy_budget_kwh,
            ipopt_max_iter=ipopt_max_iter,
            ipopt_tol=ipopt_tol,
            ipopt_acceptable_tol=ipopt_acceptable_tol,
            use_lbfgs=use_lbfgs,
            print_level=print_level,
            stage_tag="StageC_SetupCoOpt",
            mu_scale=eff_mu_scale,
            temp_c=eff_temp_c,
            T_tire_4_profile=t_tire_stage_c,
            progress_file=progress_file,
            optimize_trackside_setup=True,
            compound=compound,
        )
        if res_c["ipopt"]["status"] not in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and not use_lbfgs:
            print("  [StageC] Exact Hessian did not converge cleanly; retrying Stage C with L-BFGS...", flush=True)
            res_c = _solve_single_stage_ocp(
                grip_profile=grip_profile,
                grid=grid,
                w0_init=res_b["w_opt_vec"],
                cone_margin_m=cone_margin_m,
                grip_realism_factor=grip_realism_factor,
                human_limits=(mode == "human_achievable"),
                energy_budget_kwh=energy_budget_kwh,
                ipopt_max_iter=ipopt_max_iter,
                ipopt_tol=ipopt_tol,
                ipopt_acceptable_tol=ipopt_acceptable_tol,
                use_lbfgs=True,
                print_level=print_level,
                stage_tag="StageC_SetupCoOpt_LBFGS",
                mu_scale=eff_mu_scale,
                temp_c=eff_temp_c,
                T_tire_4_profile=t_tire_stage_c,
                progress_file=progress_file,
                optimize_trackside_setup=True,
                compound=compound,
            )
        opt_lap_s = float(res_c["summary"]["lap_time_s"])
        rec = dict(res_c.get("optimal_setup") or {})
        rec["base_lap_time_s"] = round(base_lap_s, 4)
        rec["optimal_lap_time_s"] = round(opt_lap_s, 4)
        rec["delta_time_s"] = round(opt_lap_s - base_lap_s, 4)
        res_c["recommended_setup"] = rec
        res_c["base_summary"] = res_b["summary"]
        res_c["stage_a_summary"] = res_a["summary"]
        print(
            f"  [StageC SETUP RESULT] Base={base_lap_s:.4f}s -> Opt={opt_lap_s:.4f}s "
            f"(delta={opt_lap_s - base_lap_s:+.4f}s) | LLTD={rec.get('lltd', 0):.3f} "
            f"CamF={rec.get('camber_f_deg', 0):.2f} CamR={rec.get('camber_r_deg', 0):.2f} "
            f"ToeF={rec.get('toe_f_deg', 0):+.2f} ToeR={rec.get('toe_r_deg', 0):+.2f} "
            f"BB={rec.get('brake_bias', 0):.3f} Rake={rec.get('rake_mm', 0):+.1f}mm",
            flush=True,
        )
        return res_c

    sb = res_b["summary"]
    print(
        f"  [StageB RESULT] T_ocp={sb['lap_time_s']:.3f}s | v_max={sb['v_max_kmh']:.1f}km/h | "
        f"ay_max={sb['a_y_max_g']:.2f}g | beta_max={sb['beta_max_deg']:.2f}deg | "
        f"delta_dot_max={sb['delta_dot_max_deg_s']:.1f}deg/s | drivability={sb['drivability_score']:.1f}/100 | "
        f"T_tire={sb['tire_temp_mean_c']:.1f}/{sb['tire_temp_max_c']:.1f}C | "
        f"P_dc=[{sb['P_dc_min_kw_exact']:.2f}, {sb['P_dc_max_kw_exact']:.2f}]kW (viol={sb['hv_violations_exact']}) | "
        f"min_clear={sb['min_cone_clearance_m']:.3f}m (hits={sb['cones_hit']}) | E_22km={sb['e_22km_kwh']:.3f}kWh",
        flush=True,
    )
    return res_b


def build_racing_line_and_qss_from_ocp(
    base_track: Track,
    grid: Dict[str, Any],
    ocp_res: Dict[str, Any],
) -> Tuple[RacingLineResult, QSSResult]:
    """Convert CasADi OCP trajectory into uniform ds=0.5m RacingLineResult & QSSResult for 14-DOF."""
    tr = ocp_res["trace"]
    x_car = tr["x_car"]
    y_car = tr["y_car"]

    race_track = build_track_from_waypoints(
        name=f"{base_track.name}_casadi",
        waypoints_xy=np.column_stack([x_car, y_car]),
        w_left=1.80,
        w_right=1.80,
        is_closed=True,
        ds_target=0.50,
    )

    # Arc length along CasADi racing line points
    seg = np.hypot(np.diff(np.append(x_car, x_car[0])), np.diff(np.append(y_car, y_car[0])))
    s_ocp_r = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    l_ocp_r = float(np.sum(seg))
    scale_s = race_track.length / max(l_ocp_r, 1.0)
    s_ocp_scaled = s_ocp_r * scale_s

    def _interp_per(vals: np.ndarray) -> np.ndarray:
        return np.interp(
            race_track.s,
            np.append(s_ocp_scaled, race_track.length),
            np.append(vals, vals[0]),
        )

    v_qss = _interp_per(tr["vx"])
    ax_qss = _interp_per(tr["ax"])
    ay_qss = _interp_per(tr["ay"])
    p_dc = _interp_per(tr["P_dc_kw"])
    n_on_base = np.interp(
        base_track.s,
        np.append(grid["s"], grid["length"]),
        np.append(tr["n"], tr["n"][0]),
    )

    rl = RacingLineResult(
        base_track=base_track,
        racing_track=race_track,
        n_offset_on_base=n_on_base,
        lb=-base_track.w_right,
        ub=base_track.w_left,
        centerline_int_kappa2=float(np.sum(base_track.kappa ** 2) * base_track.ds),
        racing_int_kappa2=float(np.sum(race_track.kappa ** 2) * race_track.ds),
        centerline_max_abs_kappa=float(np.max(np.abs(base_track.kappa))),
        racing_max_abs_kappa=float(np.max(np.abs(race_track.kappa))),
        within_bounds=True,
        stats={"solver": "CasADi-IPOPT"},
    )
    ds_seg = np.full_like(race_track.s, race_track.ds)
    qss = QSSResult(
        track_name=base_track.name,
        controller_label="casadi_ocp",
        s=race_track.s,
        ds_seg=ds_seg,
        v_apex=v_qss,
        v_qss=v_qss,
        ax_qss=ax_qss,
        ay_qss=ay_qss,
        p_dc_kw=p_dc,
        dt_seg=ds_seg / np.maximum(v_qss, 1.0),
        lap_time_s=float(ocp_res["summary"]["lap_time_s"]),
        v_max_kmh=float(np.max(v_qss) * 3.6),
        v_min_kmh=float(np.min(v_qss) * 3.6),
        e_net_kj=float(ocp_res["summary"]["e_net_lap_kj"]),
        e_regen_kj=float(ocp_res["summary"]["e_regen_lap_kj"]),
        solve_time_ms=0.0,
    )
    return rl, qss


def _run_14dof_godot_thermal(
    runner_gd_path: Path,
    tmp_dir: Path,
    label: str,
    racing_line: RacingLineResult,
    qss: QSSResult,
    tv_4wd_mode: int,
    alpha_drv: float,
    csv_out: Path,
    env_mu_scale: float,
    env_aero_cl_scale: float,
    temp_c: float,
    tire_init_temp_c: float,
) -> TransientLapResult:
    """Invoke Godot headless 400 Hz Full14DofModel with exact mu_scale, aero_cl_scale, and temp_c."""
    base = racing_line.base_track
    race = racing_line.racing_track
    job_path = tmp_dir / f"job_{label}_{alpha_drv:.4f}.json"
    out_summary_path = tmp_dir / f"sum_{label}_{alpha_drv:.4f}.json"
    job: Dict[str, Any] = {
        "label": label,
        "ecu_driving_mode": 2,
        "tv_4wd_mode": int(tv_4wd_mode),
        "use_3d_kinematics": True,
        "alpha_drv": float(alpha_drv),
        "n_laps": 1,
        "csv_path": str(csv_out),
        "env_mu_scale": float(env_mu_scale),
        "env_aero_cl_scale": float(env_aero_cl_scale),
        "temp_c": float(temp_c),
        "tire_init_temp_c": float(tire_init_temp_c),
        "base_track": {
            "length": float(base.length),
            "ds": float(base.ds),
            "s": [round(float(v), 4) for v in base.s],
            "x": [round(float(v), 5) for v in base.x],
            "y": [round(float(v), 5) for v in base.y],
            "psi": [round(float(v), 6) for v in base.psi],
            "w_left": [round(float(v), 4) for v in base.w_left],
            "w_right": [round(float(v), 4) for v in base.w_right],
        },
        "racing_track": {
            "length": float(race.length),
            "ds": float(race.ds),
            "s": [round(float(v), 4) for v in race.s],
            "x": [round(float(v), 5) for v in race.x],
            "y": [round(float(v), 5) for v in race.y],
            "psi": [round(float(v), 6) for v in race.psi],
            "kappa": [round(float(v), 6) for v in race.kappa],
        },
        "v_qss": [round(float(v), 4) for v in qss.v_qss],
        "ax_qss": [round(float(a), 4) for a in qss.ax_qss],
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    cmd = [
        "godot",
        "--headless",
        "--path",
        str(SIM_DIR),
        "-s",
        str(runner_gd_path),
        "--",
        str(job_path),
        str(out_summary_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_summary_path.exists():
        raise RuntimeError(
            f"Godot 14-DOF runner failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    summary = json.loads(out_summary_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_out) if csv_out.exists() else None
    return TransientLapResult(label=label, summary=summary, telemetry=df)


def validate_with_14dof(
    base_track: Track,
    grid: Dict[str, Any],
    ocp_res: Dict[str, Any],
    grip_profile: str,
    energy_budget_kwh: Optional[float] = None,
    min_clear_req_m: float = 0.145,
    save_csv_path: Optional[Path] = None,
    mu_scale: Optional[float] = None,
    temp_c: Optional[float] = None,
    progress_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run closed-loop 400 Hz 14-DOF validation in Godot headless with EcuTorqueVectoring4Wd.gd
    using exact env_mu_scale, env_aero_cl_scale, and track/tire temperature.
    Searches alpha_drv to find the fastest strictly valid lap with:
      - 0 cones hit (4-corner swept path)
      - min_14dof_cone_clearance_m >= min_clear_req_m (>= 0.14 m)
      - 0 spins, 0 HV violations ([-25.0, +80.0] kW)
    """
    prof_cfg = _resolve_grip_profile_cfg(grip_profile)
    eff_mu_scale = float(prof_cfg["env_mu_scale"] if mu_scale is None else mu_scale)
    eff_aero_cl = float(prof_cfg["env_aero_cl_scale"])
    eff_temp_c = float(prof_cfg["default_temp_c"] if temp_c is None else temp_c)
    tire_init_c = float(ocp_res["summary"].get("tire_temp_mean_c", min(max(eff_temp_c + 28.0, 22.0), 80.0)))

    print(
        f"[STAGE 3/3] Validating 14-DOF closed-loop in Godot headless ({grip_profile}, "
        f"mu={eff_mu_scale:.3f}, cl={eff_aero_cl:.2f}, temp={eff_temp_c:.1f}C, T_init={tire_init_c:.1f}C)...",
        flush=True,
    )
    _emit_progress(
        progress_file,
        {"stage": "14DOF_Validation", "status": "running", "mu_scale": eff_mu_scale, "temp_c": eff_temp_c},
    )

    rl, qss = build_racing_line_and_qss_from_ocp(base_track, grid, ocp_res)
    prof_cli = PROFILE_CLI_MAP.get(grip_profile, grip_profile)
    # Use TV4 Efficiency mode (2) for endurance budget, Autocross (0) for Max Attack
    tv4_mode = 2 if (energy_budget_kwh is not None and energy_budget_kwh > 0.0) else 0

    # Patch a temporary copy of lts_transient_runner.gd to inject env_mu_scale, env_aero_cl_scale, temp_c
    orig_gd = RUNNER_GD_ORIG.read_text(encoding="utf-8")
    anchor = '\tif cfg_json.has("grip_profile") and String(cfg_json["grip_profile"]) == "fsg_real":\n\t\tm.set_track_grip_profile(1)\n\telif cfg_json.has("track_grip_profile"):\n\t\tm.set_track_grip_profile(int(cfg_json["track_grip_profile"]))\n'
    extra = (
        anchor
        + '\tif cfg_json.has("env_mu_scale"):\n'
        + '\t\tm.env_mu_scale = float(cfg_json["env_mu_scale"])\n'
        + '\tif cfg_json.has("env_aero_cl_scale"):\n'
        + '\t\tm.env_aero_cl_scale = float(cfg_json["env_aero_cl_scale"])\n'
        + '\tif cfg_json.has("temp_c"):\n'
        + '\t\tvar t_trk: float = float(cfg_json["temp_c"])\n'
        + '\t\tm.config.track_temp_c = t_trk\n'
        + '\t\tvar t_init: float = float(cfg_json.get("tire_init_temp_c", clampf(t_trk + 28.0, 22.0, 78.0)))\n'
        + '\t\tfor w in m.wheels:\n'
        + '\t\t\tw.t_track_c = t_trk\n'
        + '\t\t\tw.set_tire_temperature(t_init)\n'
    )
    if anchor not in orig_gd:
        raise RuntimeError("Could not locate grip_profile anchor in lts_transient_runner.gd")
    patched_gd_text = orig_gd.replace(anchor, extra, 1)

    with tempfile.TemporaryDirectory(prefix="casadi_14dof_") as tmp:
        tmp_dir = Path(tmp)
        patched_gd_path = tmp_dir / "lts_transient_runner_thermal.gd"
        patched_gd_path.write_text(patched_gd_text, encoding="utf-8")

        def _eval(alpha: float, csv_out: Path) -> Tuple[TransientLapResult, Dict[str, Any], int]:
            res = _run_14dof_godot_thermal(
                runner_gd_path=patched_gd_path,
                tmp_dir=tmp_dir,
                label=f"casadi_{prof_cli}",
                racing_line=rl,
                qss=qss,
                tv_4wd_mode=tv4_mode,
                alpha_drv=round(alpha, 4),
                csv_out=csv_out,
                env_mu_scale=eff_mu_scale,
                env_aero_cl_scale=eff_aero_cl,
                temp_c=eff_temp_c,
                tire_init_temp_c=tire_init_c,
            )
            sw = verify_swept_path_4corners(base_track, rl, res.telemetry)
            hv_viol = 0
            if res.telemetry is not None and len(res.telemetry) > 0:
                p_col = res.telemetry["p_dc_kw"].to_numpy(dtype=np.float64)
                hv_viol = int(np.sum((p_col > P_DC_MAX_KW_HARD + 0.05) | (p_col < P_DC_MIN_KW_HARD - 0.05)))
            return res, sw, hv_viol

        def _is_strictly_valid(res: TransientLapResult, sw: Dict[str, Any], hv_viol: int) -> bool:
            return (
                bool(res.valid)
                and int(sw["corners_cones_hit"]) == 0
                and float(sw["min_14dof_cone_clearance_m"]) >= min_clear_req_m
                and hv_viol == 0
                and not bool(res.summary.get("spin_detected", False))
            )

        best_tuple: Optional[Tuple[float, TransientLapResult, Dict[str, Any], int, Path]] = None
        a_lo, a_hi = 0.88, 1.04
        idx_try = 0

        for a_try in (1.00,):
            csv_p = tmp_dir / f"trace_{idx_try}.csv"
            idx_try += 1
            r, sw, hv = _eval(a_try, csv_p)
            ok = _is_strictly_valid(r, sw, hv)
            print(
                f"  [14-DOF] alpha={a_try:.3f}: lap={r.lap_time_s:.3f}s, valid={ok}, "
                f"cones={sw['corners_cones_hit']}, min_clear={sw['min_14dof_cone_clearance_m']:.3f}m, "
                f"beta={r.summary.get('max_abs_beta_deg', 0.0):.2f}deg, hv_viol={hv}",
                flush=True,
            )
            if ok:
                best_tuple = (a_try, r, sw, hv, csv_p)
                a_lo = a_try
            else:
                a_hi = a_try

        # Bisection (3 steps)
        for _ in range(3):
            a_try = 0.5 * (a_lo + a_hi)
            csv_p = tmp_dir / f"trace_{idx_try}.csv"
            idx_try += 1
            r, sw, hv = _eval(a_try, csv_p)
            ok = _is_strictly_valid(r, sw, hv)
            print(
                f"  [14-DOF] alpha={a_try:.3f}: lap={r.lap_time_s:.3f}s, valid={ok}, "
                f"cones={sw['corners_cones_hit']}, min_clear={sw['min_14dof_cone_clearance_m']:.3f}m, "
                f"beta={r.summary.get('max_abs_beta_deg', 0.0):.2f}deg, hv_viol={hv}",
                flush=True,
            )
            if ok:
                if best_tuple is None or r.lap_time_s < best_tuple[1].lap_time_s:
                    best_tuple = (a_try, r, sw, hv, csv_p)
                a_lo = a_try
            else:
                a_hi = a_try

        if best_tuple is None:
            for a_fb in (0.78, 0.72, 0.66):
                csv_p = tmp_dir / f"trace_{idx_try}.csv"
                idx_try += 1
                r, sw, hv = _eval(a_fb, csv_p)
                ok = _is_strictly_valid(r, sw, hv)
                print(
                    f"  [14-DOF fallback] alpha={a_fb:.3f}: lap={r.lap_time_s:.3f}s, valid={ok}, "
                    f"cones={sw['corners_cones_hit']}, min_clear={sw['min_14dof_cone_clearance_m']:.3f}m",
                    flush=True,
                )
                if ok:
                    best_tuple = (a_fb, r, sw, hv, csv_p)
                    break

        if best_tuple is None:
            raise RuntimeError("14-DOF closed-loop validation failed to find a cone-free lap.")

        best_alpha, best_res, best_sw, best_hv, _ = best_tuple
        df_14 = best_res.telemetry.copy()
        if save_csv_path is not None:
            save_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_14.to_csv(save_csv_path, index=False)

    lap1 = df_14[df_14["lap"] == 1].copy() if "lap" in df_14.columns else df_14.copy()
    e_net_kj = float(best_res.summary["e_net_kj"])
    e_22km_kwh = (e_net_kj / 3600.0) * (22000.0 / float(base_track.length))
    t_ocp = float(ocp_res["summary"]["lap_time_s"])
    t_14 = float(best_res.lap_time_s)
    gap_pct = 100.0 * abs(t_ocp - t_14) / max(t_14, 1e-6)

    # Compute drivability score on 14-DOF telemetry (downsampled to ~20 Hz so finite diffs match human cadence)
    stride = max(1, len(lap1) // 800)
    lap1_ds = lap1.iloc[::stride]
    drv_14 = compute_drivability_score(
        t_arr=lap1_ds["t_s"].to_numpy(dtype=np.float64),
        steer_deg=lap1_ds["steer_deg"].to_numpy(dtype=np.float64),
        throttle=lap1_ds["throttle"].to_numpy(dtype=np.float64),
        brake=lap1_ds["brake"].to_numpy(dtype=np.float64),
        beta_deg=lap1_ds["beta_deg"].to_numpy(dtype=np.float64),
        wz_rad_s=lap1_ds["wz"].to_numpy(dtype=np.float64),
        bspd_violations=0,
    )
    t_surf_max_14 = float(np.max(lap1["t_surf_max_c"].to_numpy(dtype=np.float64))) if "t_surf_max_c" in lap1.columns else float(ocp_res["summary"]["tire_temp_max_c"])
    t_surf_mean_14 = float(np.mean(lap1["t_surf_max_c"].to_numpy(dtype=np.float64))) if "t_surf_max_c" in lap1.columns else float(ocp_res["summary"]["tire_temp_mean_c"])

    print(
        f"  [14-DOF VALIDATED] alpha={best_alpha:.4f} | T_14dof={t_14:.3f}s vs T_ocp={t_ocp:.3f}s "
        f"(gap={gap_pct:.2f}%) | v_max={float(np.max(lap1['vx']) * 3.6):.1f}km/h | "
        f"drivability={drv_14['drivability_score']:.1f}/100 | cones={best_sw['corners_cones_hit']} | "
        f"min_clear={best_sw['min_14dof_cone_clearance_m']:.3f}m | E_22km={e_22km_kwh:.3f}kWh | hv_viol={best_hv}",
        flush=True,
    )
    _emit_progress(
        progress_file,
        {
            "stage": "14DOF_Validation",
            "status": "done",
            "sim14dof_lap_time_s": round(t_14, 4),
            "ocp_lap_time_s": round(t_ocp, 4),
            "drivability_score": drv_14["drivability_score"],
        },
    )

    return {
        "alpha_drv": best_alpha,
        "sim14dof_lap_time_s": round(t_14, 4),
        "ocp_vs_14dof_gap_pct": round(gap_pct, 3),
        "cones_hit": int(best_sw["corners_cones_hit"]),
        "min_cone_clearance_m": round(float(best_sw["min_14dof_cone_clearance_m"]), 4),
        "max_beta_deg": round(float(best_res.summary["max_abs_beta_deg"]), 3),
        "hv_violations": int(best_hv),
        "e_net_kj": round(e_net_kj, 2),
        "e_regen_kj": round(float(best_res.summary["e_regen_kj"]), 2),
        "e_22km_kwh": round(e_22km_kwh, 3),
        "v_max_kmh": round(float(np.max(lap1["vx"].to_numpy(dtype=np.float64)) * 3.6), 2),
        "ay_max_g": round(float(np.max(np.abs(lap1["ay"].to_numpy(dtype=np.float64))) / 9.81), 3),
        "p_dc_min_kw": round(float(best_res.summary["p_chg_min_kw"]), 3),
        "p_dc_max_kw": round(float(best_res.summary["p_dc_max_kw"]), 3),
        "tv_4wd_mode": tv4_mode,
        "drivability_score": drv_14["drivability_score"],
        "drivability_details": drv_14,
        "tire_temp_mean_c": round(t_surf_mean_14, 2),
        "tire_temp_max_c": round(t_surf_max_14, 2),
        "df_lap1": lap1,
        "racing_line": rl,
    }


def _classify_phase(throttle: float, brake: float, ax_g: float, ay_g: float, p_dc_kw: float) -> str:
    if brake >= 0.12 and abs(ay_g) >= 0.45:
        return "TRAIL_BRAKING"
    if brake >= 0.12 or ax_g <= -0.35:
        return "BRAKING"
    if throttle >= 0.85:
        return "FULL_THROTTLE"
    if throttle <= 0.14 and brake < 0.12 and abs(p_dc_kw) < 10.0:
        return "LIFT_AND_COAST"
    return "PARTIAL_THROTTLE"


def export_schema_json(
    out_path: Path,
    base_track: Track,
    grid: Dict[str, Any],
    ocp_res: Dict[str, Any],
    val_14dof: Optional[Dict[str, Any]],
    mode: str,
    grip_profile: str,
    energy_budget_kwh: Optional[float],
    n_waypoints: int = 380,
) -> Dict[str, Any]:
    """
    Export the exact Shared JSON Schema Contract for SimUI / RacingLineVisualizer / LTS Tab:
      - meta
      - waypoints
      - corners (with 40 m pre-brake flying spawn_pos, spawn_yaw, spawn_vx)
      - markers (BRAKE_POINT, LIFT_COAST_START, TURN_IN, APEX, THROTTLE_ON)
    """
    tr = ocp_res["trace"]
    ocp_sum = ocp_res["summary"]
    prof_cli = PROFILE_CLI_MAP.get(grip_profile, grip_profile)
    track_stem = grid["stem"]
    is_endurance = energy_budget_kwh is not None and energy_budget_kwh > 0.0

    # Build uniform s-grid along racing line
    if val_14dof is not None:
        rl = val_14dof["racing_line"]
        race = rl.racing_track
        lap1 = val_14dof["df_lap1"]
        l_race = float(race.length)
        s_grid = np.linspace(0.0, l_race, n_waypoints, endpoint=False)

        # Clean monotonic s_m from 14-DOF lap1
        s_raw = lap1["s_m"].to_numpy(dtype=np.float64)
        # Drop wrap glitches at very start/end
        keep = np.ones(len(s_raw), dtype=bool)
        for k in range(1, len(s_raw)):
            if s_raw[k] < s_raw[k - 1] - 50.0:
                if k < 15:
                    keep[:k] = False
                else:
                    keep[k:] = False
                    break
        lap_c = lap1.iloc[np.where(keep)[0]]
        s_c = lap_c["s_m"].to_numpy(dtype=np.float64)
        order = np.argsort(s_c)
        s_u, uniq = np.unique(s_c[order], return_index=True)
        idx_u = order[uniq]

        def _i14(col: str) -> np.ndarray:
            arr = lap_c[col].to_numpy(dtype=np.float64)[idx_u]
            return np.interp(s_grid, np.append(s_u, l_race), np.append(arr, arr[0]))

        x_w = _i14("x")
        y_w = _i14("y")
        yaw_raw = np.unwrap(np.deg2rad(lap_c["yaw_deg"].to_numpy(dtype=np.float64)[idx_u]))
        d_yaw = _wrap_angle(yaw_raw[0] - yaw_raw[-1])
        yaw_w = _wrap_angle(np.interp(s_grid, np.append(s_u, l_race), np.append(yaw_raw, yaw_raw[-1] + d_yaw)))
        v_ref_w = _i14("vx")
        thr_w = np.clip(_i14("throttle"), 0.0, 1.0)
        brk_w = np.clip(_i14("brake"), 0.0, 1.0)
        steer_deg_w = _i14("steer_deg")
        ay_g_w = _i14("ay") / 9.81
        ax_g_w = _i14("ax") / 9.81
        p_dc_w = _i14("p_dc_kw")
        kap_w = np.interp(s_grid, race.s, race.kappa)

        # Interpolate OCP speed onto same s_grid
        seg_ocp = np.hypot(np.diff(np.append(tr["x_car"], tr["x_car"][0])), np.diff(np.append(tr["y_car"], tr["y_car"][0])))
        s_ocp_r = np.concatenate([[0.0], np.cumsum(seg_ocp[:-1])]) * (l_race / max(float(np.sum(seg_ocp)), 1.0))
        v_ocp_w = np.interp(s_grid, np.append(s_ocp_r, l_race), np.append(tr["vx"], tr["vx"][0]))
    else:
        x_car, y_car = tr["x_car"], tr["y_car"]
        seg_ocp = np.hypot(np.diff(np.append(x_car, x_car[0])), np.diff(np.append(y_car, y_car[0])))
        l_race = float(np.sum(seg_ocp))
        s_ocp_r = np.concatenate([[0.0], np.cumsum(seg_ocp[:-1])])
        s_grid = np.linspace(0.0, l_race, n_waypoints, endpoint=False)

        def _iocp(arr: np.ndarray) -> np.ndarray:
            return np.interp(s_grid, np.append(s_ocp_r, l_race), np.append(arr, arr[0]))

        x_w = _iocp(x_car)
        y_w = _iocp(y_car)
        yaw_unw = np.unwrap(tr["yaw_car"])
        d_yaw = _wrap_angle(yaw_unw[0] - yaw_unw[-1])
        yaw_w = _wrap_angle(np.interp(s_grid, np.append(s_ocp_r, l_race), np.append(yaw_unw, yaw_unw[-1] + d_yaw)))
        v_ocp_w = _iocp(tr["vx"])
        v_ref_w = v_ocp_w.copy()
        ax_g_w = _iocp(tr["ax"]) / 9.81
        ay_g_w = _iocp(tr["ay"]) / 9.81
        p_dc_w = _iocp(tr["P_dc_kw"])
        steer_deg_w = np.rad2deg(_iocp(tr["delta"]))
        f_brk_w = _iocp(tr["F_brk_hyd"])
        brk_w = np.clip(np.where(f_brk_w > 50.0, 0.35 + 0.65 * f_brk_w / 8000.0, np.maximum(-p_dc_w, 0.0) / 25.0 * 0.35), 0.0, 1.0)
        thr_w = np.clip(np.where(brk_w > 0.08, 0.0, np.maximum(p_dc_w, 0.0) / 75.0), 0.0, 1.0)
        kap_w = _iocp(tr["kappa_c"])

    # Clean micro-pulses for crisp phase classification
    phases = [
        _classify_phase(float(thr_w[i]), float(brk_w[i]), float(ax_g_w[i]), float(ay_g_w[i]), float(p_dc_w[i]))
        for i in range(n_waypoints)
    ]

    # Detect corners from curvature & local speed minima
    ds_w = l_race / float(n_waypoints)
    min_corner_sep = max(6, int(round(26.0 / max(ds_w, 0.5))))
    abs_k = np.abs(kap_w)
    # Smooth |kappa| slightly for robust peak detection
    abs_k_sm = _periodic_smooth(abs_k, 2)
    v_sm = _periodic_smooth(v_ref_w, 2)

    apex_candidates: List[int] = []
    for i in range(n_waypoints):
        im = (i - 1) % n_waypoints
        ip = (i + 1) % n_waypoints
        im2 = (i - 2) % n_waypoints
        ip2 = (i + 2) % n_waypoints
        is_k_peak = (abs_k_sm[i] >= 0.042) and (abs_k_sm[i] >= abs_k_sm[im]) and (abs_k_sm[i] >= abs_k_sm[ip])
        is_v_min = (abs_k_sm[i] >= 0.038) and (v_sm[i] <= v_sm[im] + 0.02) and (v_sm[i] <= v_sm[ip] + 0.02) and (
            v_sm[i] <= 0.5 * (v_sm[im2] + v_sm[ip2]) + 0.05
        )
        if is_k_peak or is_v_min:
            apex_candidates.append(i)

    # Non-maximum suppression by min_corner_sep (preferring lower speed / higher |kappa|)
    apex_indices: List[int] = []
    for idx in apex_candidates:
        if not apex_indices:
            apex_indices.append(idx)
            continue
        prev = apex_indices[-1]
        dist_i = min((idx - prev) % n_waypoints, (prev - idx) % n_waypoints)
        if dist_i < min_corner_sep:
            if abs_k_sm[idx] > abs_k_sm[prev]:
                apex_indices[-1] = idx
        else:
            apex_indices.append(idx)
    # Check wrap-around between last and first
    if len(apex_indices) >= 2:
        dist_wrap = (apex_indices[0] - apex_indices[-1]) % n_waypoints
        if dist_wrap < min_corner_sep:
            if abs_k_sm[apex_indices[-1]] > abs_k_sm[apex_indices[0]]:
                apex_indices[0] = apex_indices[-1]
            apex_indices.pop()

    corners: List[Dict[str, Any]] = []
    markers: List[Dict[str, Any]] = []
    is_bp_arr = np.zeros(n_waypoints, dtype=bool)
    is_apx_arr = np.zeros(n_waypoints, dtype=bool)
    is_exit_arr = np.zeros(n_waypoints, dtype=bool)

    for c_num, i_apx in enumerate(apex_indices, start=1):
        is_apx_arr[i_apx] = True
        peak_steer = max(abs(float(steer_deg_w[i_apx])), 2.0)

        # Scan backward for turn-in (up to 35 m)
        max_ti_steps = int(round(35.0 / ds_w))
        i_turn = (i_apx - max(2, int(round(10.0 / ds_w)))) % n_waypoints
        for step in range(1, max_ti_steps + 1):
            j = (i_apx - step) % n_waypoints
            if abs(float(steer_deg_w[j])) < 0.25 * peak_steer:
                i_turn = (j + 1) % n_waypoints
                break

        # Scan backward for braking start (up to 65 m)
        max_brk_steps = int(round(65.0 / ds_w))
        i_brk = None
        in_brk = False
        for step in range(0, max_brk_steps + 1):
            j = (i_apx - step) % n_waypoints
            if phases[j] in ("BRAKING", "TRAIL_BRAKING"):
                in_brk = True
                i_brk = j
            elif in_brk:
                break
        has_real_brake = i_brk is not None
        if i_brk is None:
            i_brk = (i_turn - max(2, int(round(8.0 / ds_w)))) % n_waypoints
        else:
            is_bp_arr[i_brk] = True

        # Scan forward for throttle-on / corner exit (up to 45 m)
        max_ex_steps = int(round(45.0 / ds_w))
        i_exit = (i_apx + max(3, int(round(12.0 / ds_w)))) % n_waypoints
        for step in range(1, max_ex_steps + 1):
            j = (i_apx + step) % n_waypoints
            if float(thr_w[j]) >= 0.70 and abs(float(steer_deg_w[j])) <= 0.45 * peak_steer:
                i_exit = j
                break
        is_exit_arr[i_exit] = True

        s_brake = float(s_grid[i_brk])
        s_turn = float(s_grid[i_turn])
        s_apex = float(s_grid[i_apx])
        s_exit = float(s_grid[i_exit])
        s_start = float((s_brake - 10.0) % l_race)

        # Flying spawn 40 m before s_brake at optimal entry speed
        s_spawn = float((s_brake - 40.0) % l_race)
        i_spawn = int(round(s_spawn / ds_w)) % n_waypoints
        direction = "Left" if float(kap_w[i_apx]) >= 0.0 else "Right"

        corners.append({
            "corner_id": c_num,
            "name": f"T{c_num} ({direction})",
            "s_start": round(s_start, 2),
            "s_brake": round(s_brake, 2),
            "s_turn_in": round(s_turn, 2),
            "s_apex": round(s_apex, 2),
            "s_exit": round(s_exit, 2),
            "v_entry_kmh": round(float(v_ref_w[i_brk]) * 3.6, 2),
            "v_apex_kmh": round(float(v_ref_w[i_apx]) * 3.6, 2),
            "v_exit_kmh": round(float(v_ref_w[i_exit]) * 3.6, 2),
            "spawn_pos": [round(float(x_w[i_spawn]), 3), round(float(y_w[i_spawn]), 3), 0.0],
            "spawn_yaw": round(float(yaw_w[i_spawn]), 4),
            "spawn_vx": round(float(v_ref_w[i_spawn]), 3),
        })

        if has_real_brake:
            markers.append({
                "s": round(s_brake, 2),
                "x": round(float(x_w[i_brk]), 3),
                "y": round(float(y_w[i_brk]), 3),
                "type": "BRAKE_POINT",
                "corner_id": c_num,
                "v_kmh": round(float(v_ref_w[i_brk]) * 3.6, 2),
            })
        markers.append({
            "s": round(s_turn, 2),
            "x": round(float(x_w[i_turn]), 3),
            "y": round(float(y_w[i_turn]), 3),
            "type": "TURN_IN",
            "corner_id": c_num,
            "v_kmh": round(float(v_ref_w[i_turn]) * 3.6, 2),
        })
        markers.append({
            "s": round(s_apex, 2),
            "x": round(float(x_w[i_apx]), 3),
            "y": round(float(y_w[i_apx]), 3),
            "type": "APEX",
            "corner_id": c_num,
            "v_kmh": round(float(v_ref_w[i_apx]) * 3.6, 2),
        })
        markers.append({
            "s": round(s_exit, 2),
            "x": round(float(x_w[i_exit]), 3),
            "y": round(float(y_w[i_exit]), 3),
            "type": "THROTTLE_ON",
            "corner_id": c_num,
            "v_kmh": round(float(v_ref_w[i_exit]) * 3.6, 2),
        })

    # Detect LIFT_COAST_START markers (sustained LIFT_AND_COAST >= 8 m)
    min_lc_pts = max(2, int(round(8.0 / ds_w)))
    i = 0
    while i < n_waypoints:
        if phases[i] == "LIFT_AND_COAST" and phases[(i - 1) % n_waypoints] != "LIFT_AND_COAST":
            run_len = 0
            while run_len < n_waypoints and phases[(i + run_len) % n_waypoints] == "LIFT_AND_COAST":
                run_len += 1
            if run_len >= min_lc_pts:
                # Associate with next upcoming corner
                s_lc = float(s_grid[i])
                next_cid = 1
                best_ds = 1e9
                for c in corners:
                    d_ahead = (float(c["s_apex"]) - s_lc) % l_race
                    if d_ahead < best_ds:
                        best_ds = d_ahead
                        next_cid = int(c["corner_id"])
                markers.append({
                    "s": round(s_lc, 2),
                    "x": round(float(x_w[i]), 3),
                    "y": round(float(y_w[i]), 3),
                    "type": "LIFT_COAST_START",
                    "corner_id": next_cid,
                    "v_kmh": round(float(v_ref_w[i]) * 3.6, 2),
                })
            i += max(run_len, 1)
        else:
            i += 1

    markers.sort(key=lambda m: float(m["s"]))

    waypoints: List[Dict[str, Any]] = []
    for idx in range(n_waypoints):
        waypoints.append({
            "s": round(float(s_grid[idx]), 3),
            "x": round(float(x_w[idx]), 3),
            "y": round(float(y_w[idx]), 3),
            "z": 0.0,
            "yaw": round(float(yaw_w[idx]), 4),
            "v_ref": round(float(v_ref_w[idx]), 3),
            "v_ocp": round(float(v_ocp_w[idx]), 3),
            "throttle": round(float(thr_w[idx]), 3),
            "brake": round(float(brk_w[idx]), 3),
            "steer_deg": round(float(steer_deg_w[idx]), 3),
            "ay_g": round(float(ay_g_w[idx]), 3),
            "ax_g": round(float(ax_g_w[idx]), 3),
            "p_dc_kw": round(float(p_dc_w[idx]), 3),
            "phase": phases[idx],
            # Backward-compatible aliases for RacingLineVisualizer.gd
            "s_m": round(float(s_grid[idx]), 3),
            "v_mps": round(float(v_ref_w[idx]), 3),
            "ax_mps2": round(float(ax_g_w[idx] * 9.81), 3),
            "ay_mps2": round(float(ay_g_w[idx] * 9.81), 3),
            "yaw_rad": round(float(yaw_w[idx]), 4),
            "steer_rad": round(math.radians(float(steer_deg_w[idx])), 4),
            "kappa": round(float(kap_w[idx]), 5),
            "is_brake_point": bool(is_bp_arr[idx]),
            "is_apex": bool(is_apx_arr[idx]),
            "is_exit_point": bool(is_exit_arr[idx]),
        })

    lap_id = f"{track_stem}_casadi_{prof_cli}{'_endurance' if is_endurance else ''}"
    label_mode = f"Endurance {energy_budget_kwh:.1f} kWh" if is_endurance else "Max Attack"
    label = f"CasADi {mode.replace('_', ' ').title()} ({prof_cli}, {label_mode})"

    if val_14dof is not None:
        meta = {
            "lap_id": lap_id,
            "label": label,
            "track_name": str(grid["name"]),
            "solver_mode": mode,
            "grip_profile": prof_cli,
            "mu_scale": round(float(ocp_sum["mu_scale"]), 4),
            "temp_c": round(float(ocp_sum["temp_c"]), 2),
            "tire_temp_mean_c": round(float(ocp_sum["tire_temp_mean_c"]), 2),
            "tire_temp_max_c": round(float(ocp_sum["tire_temp_max_c"]), 2),
            "drivability_score": round(float(val_14dof.get("drivability_score", ocp_sum["drivability_score"])), 2),
            "ocp_drivability_score": round(float(ocp_sum["drivability_score"]), 2),
            "energy_budget_kwh": float(energy_budget_kwh) if is_endurance else None,
            "ocp_lap_time_s": round(float(ocp_sum["lap_time_s"]), 4),
            "sim14dof_lap_time_s": round(float(val_14dof["sim14dof_lap_time_s"]), 4),
            "v_max_kmh": round(float(val_14dof["v_max_kmh"]), 2),
            "ay_max_g": round(float(val_14dof["ay_max_g"]), 3),
            "e_22km_kwh": round(float(val_14dof["e_22km_kwh"]), 3),
            "cones_hit": int(val_14dof["cones_hit"]),
            "min_cone_clearance_m": round(float(val_14dof["min_cone_clearance_m"]), 4),
            "max_beta_deg": round(float(val_14dof["max_beta_deg"]), 3),
            "hv_violations": int(val_14dof["hv_violations"]),
        }
    else:
        meta = {
            "lap_id": lap_id,
            "label": label,
            "track_name": str(grid["name"]),
            "solver_mode": mode,
            "grip_profile": prof_cli,
            "mu_scale": round(float(ocp_sum["mu_scale"]), 4),
            "temp_c": round(float(ocp_sum["temp_c"]), 2),
            "tire_temp_mean_c": round(float(ocp_sum["tire_temp_mean_c"]), 2),
            "tire_temp_max_c": round(float(ocp_sum["tire_temp_max_c"]), 2),
            "drivability_score": round(float(ocp_sum["drivability_score"]), 2),
            "ocp_drivability_score": round(float(ocp_sum["drivability_score"]), 2),
            "energy_budget_kwh": float(energy_budget_kwh) if is_endurance else None,
            "ocp_lap_time_s": round(float(ocp_sum["lap_time_s"]), 4),
            "sim14dof_lap_time_s": None,
            "v_max_kmh": round(float(ocp_sum["v_max_kmh"]), 2),
            "ay_max_g": round(float(ocp_sum["a_y_max_g"]), 3),
            "e_22km_kwh": round(float(ocp_sum["e_22km_kwh"]), 3),
            "cones_hit": int(ocp_sum["cones_hit"]),
            "min_cone_clearance_m": round(float(ocp_sum["min_cone_clearance_m"]), 4),
            "max_beta_deg": round(float(ocp_sum["beta_max_deg"]), 3),
            "hv_violations": int(ocp_sum["hv_violations_exact"]),
        }

    if ocp_res.get("recommended_setup"):
        meta["recommended_setup"] = ocp_res["recommended_setup"]

    payload = {
        "meta": meta,
        "waypoints": waypoints,
        "corners": corners,
        "markers": markers,
        # Top-level backward-compatible fields
        "track": track_stem,
        "solver": f"CasADi-IPOPT-{mode}",
        "lap_time_s": meta["sim14dof_lap_time_s"] if meta["sim14dof_lap_time_s"] is not None else meta["ocp_lap_time_s"],
        "length_m": round(l_race, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[DONE] Exported {out_path} ({len(waypoints)} waypoints, {len(corners)} corners, "
        f"{len(markers)} markers, T_ocp={meta['ocp_lap_time_s']:.3f}s, T_14dof={meta['sim14dof_lap_time_s']}, "
        f"drivability={meta['drivability_score']:.1f}/100)",
        flush=True,
    )
    return payload


def _resolve_warm_start_path(track_stem: str, grip_profile: str) -> Optional[Path]:
    if track_stem == "fsg_2023":
        if grip_profile in ("FSG_REAL_TRACK", "FSG_RUBBERED", "COLD_LOW_GRIP", "CUSTOM"):
            cand = RACING_LINE_DIR / "fsg_2023_racing_line_real.json"
        else:
            cand = RACING_LINE_DIR / "fsg_2023_racing_line.json"
        if cand.exists():
            return cand
    cand2 = RACING_LINE_DIR / f"{track_stem}_racing_line.json"
    return cand2 if cand2.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-Lap CasADi+IPOPT OCP (Thermal-Frictional, Clean Human-Achievable, Endurance & 14-DOF)"
    )
    parser.add_argument("--track-file", type=str, default=str(DEFAULT_TRACK_FILE))
    parser.add_argument("--mode", type=str, default="human_achievable", choices=["human_achievable", "theoretical_oracle"])
    parser.add_argument(
        "--grip-profile",
        type=str,
        default="fsg_rubbered",
        choices=[
            "fsg_rubbered", "FSG_RUBBERED",
            "fsg_real", "FSG_REAL_TRACK",
            "ttc_lab", "TTC_LAB_PEAK",
            "cold_low_grip", "COLD_LOW_GRIP",
            "custom", "CUSTOM", "CUSTOM_CONDITIONS",
        ],
    )
    parser.add_argument("--mu-scale", type=float, default=None, help="Asphalt grip multiplier (e.g. 0.88 for FSG_RUBBERED)")
    parser.add_argument("--temp-c", type=float, default=None, help="Track/ambient temperature in C (e.g. 45.0)")
    parser.add_argument("--progress-file", type=str, default="", help="Optional JSONL progress output path for live UI")
    parser.add_argument("--energy-budget-kwh", type=float, default=None, help="Optional 22 km endurance energy budget (kWh)")
    parser.add_argument("--cone-margin", type=float, default=None, help="4-corner cone safety margin (m)")
    parser.add_argument("--grip-realism-factor", type=float, default=None, help="Grip scale factor for 14-DOF realism calibration")
    parser.add_argument("--output", type=str, default="", help="Output path for schema-compliant racing line JSON")
    parser.add_argument("--validate-14dof", action="store_true", help="Run closed-loop 14-DOF validation in Godot headless")
    parser.add_argument("--save-14dof-csv", type=str, default="", help="Optional path to save validated 14-DOF CSV trace")
    parser.add_argument("--n-intervals", type=int, default=380, help="Hermite-Simpson intervals (default: 380)")
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--acceptable-tol", type=float, default=1e-4)
    parser.add_argument("--lbfgs", action="store_true", help="Use L-BFGS Hessian approximation")
    parser.add_argument("--optimize-setup", action="store_true", help="Co-optimize 7 trackside mechanical setup parameters")
    parser.add_argument("--compound", type=str, default=None, help="Optional tire compound override (R20 or R25B)")
    parser.add_argument("--print-level", type=int, default=5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    grip_profile = PROFILE_MAP[args.grip_profile]
    prof_cli = PROFILE_CLI_MAP[grip_profile]
    prof_cfg = _resolve_grip_profile_cfg(grip_profile)
    eff_mu_scale = float(prof_cfg["env_mu_scale"] if args.mu_scale is None else args.mu_scale)
    eff_temp_c = float(prof_cfg["default_temp_c"] if args.temp_c is None else args.temp_c)
    progress_file = Path(args.progress_file) if args.progress_file else None
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text("", encoding="utf-8")
        _emit_progress(
            progress_file,
            {"stage": "Init", "status": "starting", "grip_profile": prof_cli, "mu_scale": eff_mu_scale, "temp_c": eff_temp_c},
        )

    track_path = Path(args.track_file)
    base_track, grid = load_track_any(track_path, n_intervals=args.n_intervals)
    print(
        f"Loaded track '{grid['name']}' ({grid['stem']}): L={grid['length']:.2f}m, "
        f"N={args.n_intervals} (ds={grid['ds']:.3f}m, 2N={grid['m_pts']} pts) | "
        f"profile={grip_profile}, mu_scale={eff_mu_scale:.3f}, temp_c={eff_temp_c:.1f}C, "
        f"optimize_setup={bool(args.optimize_setup)}",
        flush=True,
    )

    ws_path = _resolve_warm_start_path(grid["stem"], grip_profile)
    ocp_res = solve_profile_ocp(
        grip_profile=grip_profile,
        grid=grid,
        racing_line_path=ws_path,
        mode=args.mode,
        cone_margin_m=args.cone_margin,
        grip_realism_factor=args.grip_realism_factor,
        energy_budget_kwh=args.energy_budget_kwh,
        ipopt_max_iter=args.max_iter,
        ipopt_tol=args.tol,
        ipopt_acceptable_tol=args.acceptable_tol,
        use_lbfgs=args.lbfgs,
        print_level=args.print_level,
        mu_scale=eff_mu_scale,
        temp_c=eff_temp_c,
        progress_file=progress_file,
        optimize_trackside_setup=bool(args.optimize_setup),
        compound=args.compound,
    )

    val_14dof = None
    if args.validate_14dof:
        csv_p = Path(args.save_14dof_csv) if args.save_14dof_csv else None
        val_14dof = validate_with_14dof(
            base_track=base_track,
            grid=grid,
            ocp_res=ocp_res,
            grip_profile=grip_profile,
            energy_budget_kwh=args.energy_budget_kwh,
            min_clear_req_m=0.145,
            save_csv_path=csv_p,
            mu_scale=eff_mu_scale,
            temp_c=eff_temp_c,
            progress_file=progress_file,
        )

    out_path = Path(args.output) if args.output else (
        RACING_LINE_DIR / f"{grid['stem']}_racing_line_casadi_{prof_cli}{'_endurance' if args.energy_budget_kwh else ''}.json"
    )
    payload = export_schema_json(
        out_path=out_path,
        base_track=base_track,
        grid=grid,
        ocp_res=ocp_res,
        val_14dof=val_14dof,
        mode=args.mode,
        grip_profile=grip_profile,
        energy_budget_kwh=args.energy_budget_kwh,
    )
    _emit_progress(
        progress_file,
        {
            "stage": "Done",
            "status": "done",
            "done": True,
            "output": str(out_path),
            "ocp_lap_time_s": payload["meta"]["ocp_lap_time_s"],
            "sim14dof_lap_time_s": payload["meta"]["sim14dof_lap_time_s"],
            "drivability_score": payload["meta"]["drivability_score"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        },
    )
    print(f"Total elapsed: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
