#!/usr/bin/env python3
"""
Milliken Yaw Moment Diagram (YMD) generator for Tecnun e-Racing TeR27 4WD (MirenaSim).

Sweeps (delta, beta) at 30, 55, and 80 km/h under FSG_REAL_TRACK (default) and plots
M_z [Nm] vs a_y [g] for:
  - Open-Diff (symmetric L/R, 35/65 F/R)
  - TV 4WD Baseline (current EcuTorqueVectoring4Wd law from tv_4wd_config.json)
  - Physical Oracle (optimal 4-wheel torque allocation envelope + trim limit)
  - Optional: TV 4WD Optimized (via --include-final for 4-way final comparison)

Computes:
  - a_y_max [g] (at trim M_z = 0)
  - Control reserve M_z [Nm] at the limit (available yaw authority from max steer / TV at limit)
  - Stability margin dM_z/dbeta [Nm/deg] (at Ay=0g, Ay=0.7*Ay_max, and near limit;
    note ISO 8855 sign convention: +beta = nose-right / +vy -> restoring yaw moment is +Mz => dMz/dbeta > 0 is stable;
    we also report the SAE-sign stability margin -dMz/dbeta_ISO where < 0 is stable for clarity).

Outputs in vd_correlation_lab/results/tv_oracle/:
  - ymd_30kmh.png
  - ymd_55kmh.png
  - ymd_80kmh.png
  - ymd_metrics.json
  - ymd_baseline_cache.json (snapshot of Open-Diff, TV Baseline, Oracle for --include-final)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results" / "tv_oracle"
TV_CFG_PATH = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter27_4wd" / "tv_4wd_config.json"

sys.path.insert(0, str(ORACLE_DIR))
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import TeR27PhysicsModel  # noqa: E402

SPEEDS_KMH = [30.0, 55.0, 80.0]
BETA_GRID_DEG = np.array([-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0], dtype=float)
DELTA_GRID_DEG = np.array(
    [-20.0, -16.0, -12.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0],
    dtype=float,
)


def _interp_1d(x: float, xp: List[float], fp: List[float]) -> float:
    return float(np.interp(x, xp, fp))


class TV4WdSteadyLaw:
    """Steady-state evaluation of EcuTorqueVectoring4Wd.gd control law from tv_4wd_config.json."""

    def __init__(self, cfg_path: Path = TV_CFG_PATH, mode_name: str = "autocross") -> None:
        d = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.d = d
        self.lq = d["lqr_gain_schedule"]
        self.m = d["modes"][mode_name]
        self.el = d["electrical"]
        self.ph = d["physical"]

    def compute_mz_req(self, vx: float, vy: float, r: float, delta: float, mu_est: float = 1.25) -> float:
        L = float(self.ph["wheelbase_m"])
        vx_eff = max(abs(vx), 1.5)
        # Speed-scheduled understeer gradient K_us
        v_lo = float(self.m.get("k_us_vx_low_ms", 10.5))
        v_hi = float(self.m.get("k_us_vx_high_ms", 19.5))
        u_v = float(np.clip((vx_eff - v_lo) / max(v_hi - v_lo, 1.0), 0.0, 1.0))
        k_us_lo = math.radians(float(self.m.get("k_us_low_deg_g", 0.15))) / 9.81
        k_us_hi = math.radians(float(self.m.get("k_us_deg_g", 0.85))) / 9.81
        k_us = k_us_lo + u_v * (k_us_hi - k_us_lo)

        r_ref = (vx_eff / (L + k_us * vx_eff * vx_eff)) * math.tan(delta)
        r_lim = float(self.m.get("r_lim_scale", 0.90)) * (mu_est * 9.81) / max(vx_eff, 2.0)
        r_ref = float(np.clip(r_ref, -r_lim, r_lim))

        k_r = _interp_1d(vx_eff, self.lq["vx_breakpoints_ms"], self.lq["k_r"])
        k_beta = _interp_1d(vx_eff, self.lq["vx_breakpoints_ms"], self.lq["k_beta"])
        beta = math.atan2(vy, vx_eff)

        # Feedforward + LQR feedback + beta soft wall
        k_ff = float(self.m.get("k_ff", 0.50))
        k_ff_mult = float(self.m.get("k_ff_low_mult", 1.50))
        k_ff_eff = k_ff * (k_ff_mult * (1.0 - u_v) + u_v)
        mz_ff = k_ff_eff * k_r * (r_ref * 0.45)

        e_r = r_ref - r
        mz_fb = k_r * e_r + k_beta * beta

        # Soft beta stabilization wall
        beta_soft_deg = float(self.m.get("beta_soft_deg", 4.5))
        beta_hi_deg = float(self.m.get("beta_soft_high_spd_deg", 2.4))
        b_lim = math.radians(beta_soft_deg + u_v * (beta_hi_deg - beta_soft_deg))
        b_gain = float(self.m.get("beta_soft_damp_gain", 1150.0))
        if abs(beta) > b_lim:
            excess = abs(beta) - b_lim
            mz_fb += math.copysign(b_gain * excess, beta)

        # Candidate B (kept): Delta_alpha = |alpha_f| - |alpha_r| regulation
        da_gain = float(self.m.get("delta_alpha_eq_gain", 0.0))
        if da_gain > 1.0 and vx_eff > 9.0 and abs(delta) > math.radians(2.0):
            lf = L * (1.0 - float(self.ph.get("front_weight_dist", 0.45)))
            lr = L * float(self.ph.get("front_weight_dist", 0.45))
            alpha_f = delta - math.atan2(vy + lf * r, vx_eff)
            alpha_r = -math.atan2(vy - lr * r, vx_eff)
            delta_alpha_rel = abs(alpha_f) - abs(alpha_r)
            da_b_ok = float(np.clip((math.radians(7.0) - abs(beta)) / math.radians(2.0), 0.0, 1.0))
            if delta_alpha_rel > 0.004 and da_b_ok > 0.01:
                mz_dalpha = float(np.clip(da_gain * (delta_alpha_rel - 0.004) * math.copysign(1.0, delta), -340.0, 340.0)) * da_b_ok
                mz_fb += mz_dalpha
            elif delta_alpha_rel < -0.012 and abs(beta) > math.radians(2.0):
                mz_dalpha = float(np.clip(1200.0 * (delta_alpha_rel + 0.012) * math.copysign(1.0, r if abs(r) > 1e-4 else delta), -240.0, 240.0))
                mz_fb += mz_dalpha

        agility = float(self.m.get("yaw_agility_gain", 1.0))
        mz_req = (mz_ff + mz_fb) * agility
        return float(np.clip(mz_req, -1400.0, 1400.0))

    def compute_rear_bias(self, vx: float, vy: float, r: float, delta: float) -> float:
        base_rear = 0.62
        cent_gain = float(self.m.get("centripetal_fx_gain", 0.0))
        if cent_gain <= 0.1 or abs(delta) <= 0.10:
            return base_rear
        vx_eff = max(abs(vx), 1.5)
        L = float(self.ph["wheelbase_m"])
        lf = L * (1.0 - float(self.ph.get("front_weight_dist", 0.45)))
        lr = L * float(self.ph.get("front_weight_dist", 0.45))
        alpha_f = delta - math.atan2(vy + lf * r, vx_eff)
        alpha_r = -math.atan2(vy - lr * r, vx_eff)
        delta_alpha_rel = abs(alpha_f) - abs(alpha_r)
        beta = math.atan2(vy, vx_eff)
        if delta_alpha_rel > 0.0 and abs(beta) < math.radians(5.0):
            b_ok = float(np.clip((math.radians(5.0) - abs(beta)) / math.radians(1.4), 0.0, 1.0))
            front_shift = float(np.clip(0.015 * cent_gain * math.sin(abs(delta)) * b_ok, 0.0, 0.18))
            return float(np.clip(base_rear - front_shift, 0.44, base_rear))
        return base_rear


def _allocate_torques_for_fx_mz(
    model: TeR27PhysicsModel,
    fx_req: float,
    mz_req: float,
    fz: np.ndarray,
    delta_w: np.ndarray,
    omega_w: np.ndarray,
    v_wx: np.ndarray,
    allow_regen: bool = True,
    rear_bias_base: float = 0.65,
) -> np.ndarray:
    """Fast closed-form + projection allocator enforcing motor limits and [-25, +80] kW HV bounds."""
    p = model.p
    t_max, t_min = model.wheel_motor_limits(omega_w)
    if not allow_regen:
        t_min = np.zeros(4)

    rho_rad = np.minimum(fz / p.k_tire_z, 0.5 * p.wheel_radius)
    r_load = p.wheel_radius - rho_rad
    f_rr = p.rolling_resistance_coeff * fz * np.tanh(v_wx / 0.5)

    # Friction-circle longitudinal force cap per wheel (conservative 0.85 * mu_x * Fz)
    fx_cap = 0.90 * model.lmux * model.tire.PDX1 * fz
    t_ub = np.minimum(t_max, (fx_cap + f_rr) * r_load)
    t_lb = np.maximum(t_min, (-fx_cap + f_rr) * r_load)

    # Base F/R distribution
    sf = 1.0 - rear_bias_base
    sr = rear_bias_base
    fx_base = np.array([0.5 * sf, 0.5 * sf, 0.5 * sr, 0.5 * sr]) * fx_req

    # Differential L/R force for Mz_req (left turn: +Mz needs FR/RR > FL/RL, i.e., -y_w * Fx > 0)
    # Also weight L/R differential by axle load fraction
    fz_sum = max(float(np.sum(fz)), 10.0)
    w_f = float(fz[0] + fz[1]) / fz_sum
    w_r = float(fz[2] + fz[3]) / fz_sum
    dfx_f = (mz_req * w_f) / max(p.track_front, 0.5)
    dfx_r = (mz_req * w_r) / max(p.track_rear, 0.5)
    fx_cmd = fx_base + np.array([-dfx_f, +dfx_f, -dfx_r, +dfx_r])
    t_cmd = np.clip((fx_cmd + f_rr) * r_load, t_lb, t_ub)

    # Strict HV power projection [-25.0, +80.0] kW
    p_ecu = model.compute_hv_power_w(t_cmd, omega_w)
    p_drv = model.compute_hv_power_drivetrain_w(t_cmd, omega_w)
    p_hi = max(p_ecu, p_drv)
    if p_hi > p.p_dc_max_w:
        scale_p = (p.p_dc_max_w - 50.0) / max(p_hi, 1.0)
        t_cmd = np.where(t_cmd * omega_w > 0.0, t_cmd * scale_p, t_cmd)
    p_lo = min(model.compute_hv_power_w(t_cmd, omega_w), model.compute_hv_power_drivetrain_w(t_cmd, omega_w))
    if p_lo < p.p_dc_min_w:
        scale_r = (p.p_dc_min_w + 50.0) / min(p_lo, -1.0)
        t_cmd = np.where(t_cmd * omega_w < 0.0, t_cmd * scale_r, t_cmd)
    return t_cmd


def eval_tethered_point(
    model: TeR27PhysicsModel,
    v_mps: float,
    beta_deg: float,
    delta_deg: float,
    controller: str,
    tv_law: Optional[TV4WdSteadyLaw] = None,
    r_seed: float = 0.0,
) -> Dict[str, float]:
    """
    Quasi-steady-state tethered YMD evaluation matching godot_ymd_probe.gd convention:
    - Fixed speed V, sideslip beta, steer delta
    - Iterate yaw rate r = ay / V and longitudinal equilibrium sum_Fx = -m * r * vy
    """
    p = model.p
    beta = math.radians(beta_deg)
    delta = math.radians(delta_deg)
    vx = v_mps * math.cos(beta)
    vy = v_mps * math.sin(beta)
    delta_w = model.ackermann_steer(delta)
    f_drag_x, _, _, _ = model.aero_forces(vx, vy)

    r_cur = r_seed
    ay_cur = v_mps * r_cur
    ax_cur = -r_cur * vy
    res: Dict[str, object] = {}

    for _ in range(18):
        v_wx, _, alpha = model.wheel_velocities_and_alphas(vx, vy, r_cur, delta_w)
        fz, phi = model.compute_normal_forces(ax_cur, ay_cur, vx, vy)
        gamma = model.compute_cambers(delta_w, phi)
        rho_rad = np.minimum(fz / p.k_tire_z, 0.5 * p.wheel_radius)
        r_load = p.wheel_radius - rho_rad
        r_eff = p.wheel_radius - rho_rad / 3.0
        omega_w = np.maximum(v_wx, 0.5) / np.maximum(r_eff, 0.1)
        f_rr = p.rolling_resistance_coeff * fz * np.tanh(v_wx / 0.5)

        fx_req = -p.mass * r_cur * vy - f_drag_x
        if controller == "OPEN_DIFF":
            t_cmd = _allocate_torques_for_fx_mz(
                model, fx_req, 0.0, fz, delta_w, omega_w, v_wx, allow_regen=False, rear_bias_base=1.0 - p.front_drive_ratio
            )
            mz_req = 0.0
        else:
            assert tv_law is not None
            mz_req = tv_law.compute_mz_req(vx, vy, r_cur, delta, mu_est=1.25 * model.env_mu_scale)
            r_bias = tv_law.compute_rear_bias(vx, vy, r_cur, delta)
            t_cmd = _allocate_torques_for_fx_mz(
                model, fx_req, mz_req, fz, delta_w, omega_w, v_wx, allow_regen=True, rear_bias_base=r_bias
            )

        fx_tgt = t_cmd / np.maximum(r_load, 0.1) - f_rr
        kappa, _, _, _ = model.solve_kappa_for_fx_target(fx_tgt, alpha, fz, gamma)
        res = model.evaluate_state_from_kappa(vx, vy, r_cur, delta, kappa, ax_cur, ay_cur, n_load_iter=2)
        ay_new = float(res["ay"])
        r_tgt = float(np.clip(ay_new / max(v_mps, 2.0), -3.0, 3.0))
        if abs(r_tgt - r_cur) < 1e-4 and abs(ay_new - ay_cur) < 1e-3:
            r_cur = r_tgt
            ay_cur = ay_new
            break
        r_cur = 0.6 * r_cur + 0.4 * r_tgt
        ay_cur = ay_new
        ax_cur = -r_cur * vy

    return {
        "beta_deg": float(beta_deg),
        "delta_deg": float(delta_deg),
        "ay_g": float(res["ay"]) / p.gravity,
        "mz_nm": float(res["sum_mz"]),
        "mz_tv_nm": float(res["mz_tv_long"]),
        "r_rad_s": float(r_cur),
        "p_dc_kw": float(res["p_dc_ecu_w"]) / 1000.0,
    }


def eval_oracle_envelope_point(
    model: TeR27PhysicsModel,
    v_mps: float,
    beta_deg: float,
    delta_deg: float,
    objective: str = "max_mz",
    r_seed: float = 0.0,
) -> Dict[str, float]:
    """
    Computes the Physical Oracle reachable (ay, Mz) at fixed (V, beta, delta) by optimizing
    free 4-wheel kappa[4] within motor + MF 6.2 + [-25, +80] kW HV bounds while maintaining
    longitudinal equilibrium sum_Fx = -m * r * vy.
    objective in {'max_mz', 'min_mz', 'trim_max_ay'}.
    """
    p = model.p
    beta = math.radians(beta_deg)
    delta = math.radians(delta_deg)
    vx = v_mps * math.cos(beta)
    vy = v_mps * math.sin(beta)
    r_cur = r_seed
    mg = p.mass * p.gravity

    def _ev(kap: np.ndarray, r_val: float) -> Dict[str, object]:
        return model.evaluate_state_from_kappa(vx, vy, r_val, delta, kap, -r_val * vy, r_val * vx, n_load_iter=2)

    # 2 outer fixed-point updates of r = ay / V
    kap_opt = np.zeros(4)
    res_best = _ev(kap_opt, r_cur)
    for _ in range(2):
        r_fixed = r_cur
        ax_req = -r_fixed * vy

        def obj_fn(k: np.ndarray) -> float:
            ev = _ev(k, r_fixed)
            if objective == "max_mz":
                return -float(ev["sum_mz"]) / 500.0
            if objective == "min_mz":
                return float(ev["sum_mz"]) / 500.0
            # trim_max_ay: maximize ay with Mz=0
            return -float(ev["ay"]) / p.gravity

        cons = [
            {"type": "eq", "fun": lambda k: (float(_ev(k, r_fixed)["sum_fx"]) - p.mass * ax_req) / mg},
            {"type": "ineq", "fun": lambda k: (p.p_dc_max_w - 50.0 - float(_ev(k, r_fixed)["p_dc_ecu_w"])) / 10000.0},
            {"type": "ineq", "fun": lambda k: (p.p_dc_max_w - 50.0 - float(_ev(k, r_fixed)["p_dc_drv_w"])) / 10000.0},
            {"type": "ineq", "fun": lambda k: (float(_ev(k, r_fixed)["p_dc_ecu_w"]) - (p.p_dc_min_w + 50.0)) / 10000.0},
            {"type": "ineq", "fun": lambda k: (_ev(k, r_fixed)["t_max"] - _ev(k, r_fixed)["torques"]) / 100.0},
            {"type": "ineq", "fun": lambda k: (_ev(k, r_fixed)["torques"] - _ev(k, r_fixed)["t_min"]) / 100.0},
        ]
        if objective == "trim_max_ay":
            cons.append({"type": "eq", "fun": lambda k: float(_ev(k, r_fixed)["sum_mz"]) / (mg * p.wheelbase)})

        k0 = np.array([-0.04, 0.05, -0.04, 0.05]) if objective == "max_mz" else (
            np.array([0.05, -0.04, 0.05, -0.04]) if objective == "min_mz" else np.array([0.01, 0.03, -0.01, 0.02])
        )
        sol = minimize(
            obj_fn,
            k0,
            method="SLSQP",
            bounds=[(-0.13, 0.13)] * 4,
            constraints=cons,
            options={"maxiter": 60, "ftol": 1e-4, "eps": 1e-4},
        )
        kap_opt = sol.x
        res_best = _ev(kap_opt, r_fixed)
        r_cur = float(np.clip(float(res_best["ay"]) / max(v_mps, 2.0), -3.0, 3.0))

    return {
        "beta_deg": float(beta_deg),
        "delta_deg": float(delta_deg),
        "ay_g": float(res_best["ay"]) / p.gravity,
        "mz_nm": float(res_best["sum_mz"]),
        "mz_tv_nm": float(res_best["mz_tv_long"]),
        "r_rad_s": float(r_cur),
        "p_dc_kw": float(res_best["p_dc_ecu_w"]) / 1000.0,
    }


def sweep_controller(
    model: TeR27PhysicsModel,
    v_kmh: float,
    controller: str,
    tv_law: Optional[TV4WdSteadyLaw] = None,
) -> Dict[str, object]:
    v_mps = v_kmh / 3.6
    nb, nd = len(BETA_GRID_DEG), len(DELTA_GRID_DEG)
    ay_g = np.zeros((nb, nd))
    mz_nm = np.zeros((nb, nd))
    p_dc = np.zeros((nb, nd))

    for i, b in enumerate(BETA_GRID_DEG):
        for j, d in enumerate(DELTA_GRID_DEG):
            # Seed yaw rate from kinematic Ackermann rate matching steer direction to avoid branch-flipping
            r_kin = (v_mps / model.p.wheelbase) * math.tan(math.radians(float(d)))
            r_seed = float(np.clip(r_kin, -2.2, 2.2))
            pt = eval_tethered_point(model, v_mps, float(b), float(d), controller, tv_law, r_seed)
            ay_g[i, j] = pt["ay_g"]
            mz_nm[i, j] = pt["mz_nm"]
            p_dc[i, j] = pt["p_dc_kw"]

    metrics = extract_ymd_metrics(ay_g, mz_nm, BETA_GRID_DEG, DELTA_GRID_DEG)
    metrics["max_p_dc_kw"] = float(np.max(p_dc))
    metrics["min_p_dc_kw"] = float(np.min(p_dc))
    metrics["hv_violations"] = int(np.sum((p_dc > 80.0) | (p_dc < -25.0)))
    return {
        "controller": controller,
        "v_kmh": v_kmh,
        "ay_g": ay_g.tolist(),
        "mz_nm": mz_nm.tolist(),
        "metrics": metrics,
    }


def sweep_oracle(model: TeR27PhysicsModel, v_kmh: float) -> Dict[str, object]:
    """
    Computes the Physical Oracle YMD envelope (upper +Mz and lower -Mz controllable boundaries)
    and exact Oracle trim max Ay at speed v_kmh.
    """
    v_mps = v_kmh / 3.6
    # Evaluate boundary along beta = [-6, -4, -2, 0, +2, +4, +6] at extreme & intermediate deltas
    b_sub = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    d_sub = np.array([-16.0, -10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 10.0, 16.0])
    nb, nd = len(b_sub), len(d_sub)
    ay_hi = np.zeros((nb, nd))
    mz_hi = np.zeros((nb, nd))
    ay_lo = np.zeros((nb, nd))
    mz_lo = np.zeros((nb, nd))

    for i, b in enumerate(b_sub):
        r_s1 = 0.0
        r_s2 = 0.0
        for j, d in enumerate(d_sub):
            p_hi = eval_oracle_envelope_point(model, v_mps, float(b), float(d), "max_mz", r_s1)
            p_lo = eval_oracle_envelope_point(model, v_mps, float(b), float(d), "min_mz", r_s2)
            ay_hi[i, j], mz_hi[i, j] = p_hi["ay_g"], p_hi["mz_nm"]
            ay_lo[i, j], mz_lo[i, j] = p_lo["ay_g"], p_lo["mz_nm"]
            r_s1, r_s2 = p_hi["r_rad_s"], p_lo["r_rad_s"]

    # Also compute exact Oracle trim max Ay (Mz = 0) at this speed via stationary NLP!
    p = model.p
    mg = p.mass * p.gravity
    mgl = mg * p.wheelbase
    best_ay_trim = 0.0
    best_ctrl_reserve = 0.0
    for d_init in [0.12, 0.22, 0.34]:
        for b_init in [-0.04, 0.0, 0.04]:
            # Variables: [r, delta, beta, kappa[4]] at fixed V = v_mps
            def _ev_v(x: np.ndarray) -> Dict[str, object]:
                r_v, d_v, b_v = float(x[0]), float(x[1]), float(x[2])
                vx_v = v_mps * math.cos(b_v)
                vy_v = v_mps * math.sin(b_v)
                return model.evaluate_state_from_kappa(vx_v, vy_v, r_v, d_v, x[3:7], -r_v * vy_v, r_v * vx_v, n_load_iter=3)

            sol = minimize(
                lambda x: -float(x[0]),
                np.array([1.2 * p.gravity / v_mps, min(d_init, p.max_steer_angle * 0.95), b_init, 0.01, 0.03, -0.01, 0.02]),
                method="SLSQP",
                bounds=[(0.1, 3.5), (0.01, p.max_steer_angle), (-0.18, 0.12)] + [(-0.13, 0.13)] * 4,
                constraints=[
                    {"type": "eq", "fun": lambda x: (float(_ev_v(x)["sum_fx"]) + p.mass * float(x[0]) * v_mps * math.sin(float(x[2]))) / mg},
                    {"type": "eq", "fun": lambda x: (float(_ev_v(x)["sum_fy"]) - p.mass * float(x[0]) * v_mps * math.cos(float(x[2]))) / mg},
                    {"type": "eq", "fun": lambda x: float(_ev_v(x)["sum_mz"]) / mgl},
                    {"type": "ineq", "fun": lambda x: (p.p_dc_max_w - 50.0 - float(_ev_v(x)["p_dc_ecu_w"])) / 10000.0},
                    {"type": "ineq", "fun": lambda x: (p.p_dc_max_w - 50.0 - float(_ev_v(x)["p_dc_drv_w"])) / 10000.0},
                    {"type": "ineq", "fun": lambda x: (float(_ev_v(x)["p_dc_ecu_w"]) - (p.p_dc_min_w + 50.0)) / 10000.0},
                    {"type": "ineq", "fun": lambda x: (_ev_v(x)["t_max"] - _ev_v(x)["torques"]) / 100.0},
                    {"type": "ineq", "fun": lambda x: (_ev_v(x)["torques"] - _ev_v(x)["t_min"]) / 100.0},
                ],
                options={"maxiter": 150, "ftol": 1e-6},
            )
            ev_s = _ev_v(sol.x)
            eq_err = max(
                abs(float(ev_s["sum_fx"]) + p.mass * float(sol.x[0]) * v_mps * math.sin(float(sol.x[2]))) / mg,
                abs(float(ev_s["sum_fy"]) - p.mass * float(sol.x[0]) * v_mps * math.cos(float(sol.x[2]))) / mg,
                abs(float(ev_s["sum_mz"])) / mgl,
            )
            if eq_err < 2e-3 and float(ev_s["ay"]) / p.gravity > best_ay_trim:
                best_ay_trim = float(ev_s["ay"]) / p.gravity
                # Control reserve at the limit: max |Mz| achievable at 90% of Ay_max
                p_hi_lim = eval_oracle_envelope_point(
                    model, v_mps, math.degrees(float(sol.x[2])), math.degrees(float(sol.x[1])), "max_mz", float(sol.x[0])
                )
                p_lo_lim = eval_oracle_envelope_point(
                    model, v_mps, math.degrees(float(sol.x[2])), math.degrees(float(sol.x[1])), "min_mz", float(sol.x[0])
                )
                best_ctrl_reserve = 0.5 * (abs(p_hi_lim["mz_nm"]) + abs(p_lo_lim["mz_nm"]))

    # Stability margin of Oracle upper/lower envelope at beta=0
    i_b0 = int(np.argmin(np.abs(b_sub)))
    j_d0 = int(np.argmin(np.abs(d_sub)))
    mz_mid = 0.5 * (mz_hi + mz_lo)
    dmz_dbeta_iso = float((mz_mid[i_b0 + 1, j_d0] - mz_mid[i_b0 - 1, j_d0]) / (b_sub[i_b0 + 1] - b_sub[i_b0 - 1]))

    return {
        "controller": "ORACLE",
        "v_kmh": v_kmh,
        "beta_sub_deg": b_sub.tolist(),
        "delta_sub_deg": d_sub.tolist(),
        "ay_hi_g": ay_hi.tolist(),
        "mz_hi_nm": mz_hi.tolist(),
        "ay_lo_g": ay_lo.tolist(),
        "mz_lo_nm": mz_lo.tolist(),
        "metrics": {
            "a_y_max_trim_g": best_ay_trim,
            "control_reserve_mz_at_limit_nm": best_ctrl_reserve,
            "stability_dmz_dbeta_iso_nm_per_deg": dmz_dbeta_iso,
            "stability_dmz_dbeta_sae_nm_per_deg": -dmz_dbeta_iso,
            "hv_violations": 0,
        },
    }


def extract_ymd_metrics(
    ay_g: np.ndarray,
    mz_nm: np.ndarray,
    betas: np.ndarray,
    deltas: np.ndarray,
) -> Dict[str, float]:
    # Trim crossings (Mz = 0) along delta for each beta row AND along beta for each delta column
    trims = []
    for i, b in enumerate(betas):
        for j in range(len(deltas) - 1):
            m0, m1 = mz_nm[i, j], mz_nm[i, j + 1]
            if m0 == 0.0 or m0 * m1 < 0.0:
                u = 0.0 if abs(m1 - m0) < 1e-12 else float(np.clip(-m0 / (m1 - m0), 0.0, 1.0))
                trims.append(
                    {
                        "ay_g": float(ay_g[i, j] + u * (ay_g[i, j + 1] - ay_g[i, j])),
                        "delta_deg": float(deltas[j] + u * (deltas[j + 1] - deltas[j])),
                        "beta_deg": float(b),
                    }
                )
    for j, d in enumerate(deltas):
        for i in range(len(betas) - 1):
            m0, m1 = mz_nm[i, j], mz_nm[i + 1, j]
            if m0 == 0.0 or m0 * m1 < 0.0:
                u = 0.0 if abs(m1 - m0) < 1e-12 else float(np.clip(-m0 / (m1 - m0), 0.0, 1.0))
                trims.append(
                    {
                        "ay_g": float(ay_g[i, j] + u * (ay_g[i + 1, j] - ay_g[i, j])),
                        "delta_deg": float(d),
                        "beta_deg": float(betas[i] + u * (betas[i + 1] - betas[i])),
                    }
                )
    pos_trims = [t for t in trims if t["ay_g"] >= 0.0]
    if pos_trims:
        best = max(pos_trims, key=lambda t: t["ay_g"])
        ay_max = best["ay_g"]
    else:
        ay_max = float(np.max(ay_g))

    # Control reserve Mz at the limit: max |Mz| achievable across deltas at Ay >= 0.85 * ay_max
    near_lim_mask = ay_g >= 0.85 * ay_max
    ctrl_reserve_lim = float(np.max(np.abs(mz_nm[near_lim_mask]))) if np.any(near_lim_mask) else 0.0

    # Stability margin dMz/dbeta at Ay=0g (delta=0, beta=0) and at 0.80 * ay_max
    i_b0 = int(np.argmin(np.abs(betas)))
    j_d0 = int(np.argmin(np.abs(deltas)))
    dmz_dbeta_0g_iso = float((mz_nm[i_b0 + 1, j_d0] - mz_nm[i_b0 - 1, j_d0]) / (betas[i_b0 + 1] - betas[i_b0 - 1]))

    # Interpolate Mz at ay_target = 0.80 * ay_max for each beta row
    ay_tgt = 0.80 * ay_max
    bs_t, mzs_t = [], []
    for i, b in enumerate(betas):
        for j in range(len(deltas) - 1):
            a0, a1 = ay_g[i, j], ay_g[i, j + 1]
            if (a0 - ay_tgt) * (a1 - ay_tgt) <= 0.0 and abs(a1 - a0) > 1e-6:
                u = (ay_tgt - a0) / (a1 - a0)
                bs_t.append(float(b))
                mzs_t.append(float(mz_nm[i, j] + u * (mz_nm[i, j + 1] - mz_nm[i, j])))
                break
    if len(bs_t) >= 2:
        idx_c = int(np.argmin(np.abs(np.array(bs_t))))
        i0, i1 = max(0, idx_c - 1), min(len(bs_t) - 1, idx_c + 1)
        dmz_dbeta_lim_iso = float((mzs_t[i1] - mzs_t[i0]) / (bs_t[i1] - bs_t[i0])) if abs(bs_t[i1] - bs_t[i0]) > 1e-6 else dmz_dbeta_0g_iso
    else:
        dmz_dbeta_lim_iso = dmz_dbeta_0g_iso

    return {
        "a_y_max_trim_g": float(ay_max),
        "control_reserve_mz_at_limit_nm": float(ctrl_reserve_lim),
        "stability_dmz_dbeta_0g_iso_nm_per_deg": float(dmz_dbeta_0g_iso),
        "stability_dmz_dbeta_lim_iso_nm_per_deg": float(dmz_dbeta_lim_iso),
        "stability_dmz_dbeta_lim_sae_nm_per_deg": float(-dmz_dbeta_lim_iso),
    }


def plot_single_speed_ymd(
    v_kmh: float,
    open_rec: Dict[str, object],
    tv_base_rec: Dict[str, object],
    oracle_rec: Dict[str, object],
    tv_opt_rec: Optional[Dict[str, object]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plt.rcParams.update({"font.size": 10.0})

    # Plot Oracle reachable envelope (convex-like upper/lower boundary)
    ay_hi = np.asarray(oracle_rec["ay_hi_g"])
    mz_hi = np.asarray(oracle_rec["mz_hi_nm"])
    ay_lo = np.asarray(oracle_rec["ay_lo_g"])
    mz_lo = np.asarray(oracle_rec["mz_lo_nm"])
    pts_x = np.concatenate([ay_hi.ravel(), ay_lo.ravel()])
    pts_y = np.concatenate([mz_hi.ravel(), mz_lo.ravel()])
    # Bin by Ay to draw clean upper & lower Oracle controllability boundary
    bins = np.linspace(float(np.min(pts_x)), float(np.max(pts_x)), 26)
    bx, by_max, by_min = [], [], []
    for k in range(len(bins) - 1):
        m = (pts_x >= bins[k]) & (pts_x <= bins[k + 1])
        if np.any(m):
            bx.append(0.5 * (bins[k] + bins[k + 1]))
            by_max.append(float(np.max(pts_y[m])))
            by_min.append(float(np.min(pts_y[m])))
    if bx:
        ax.fill_between(bx, by_min, by_max, color="#2ca02c", alpha=0.10, label=f"Oracle Reachable Envelope (Ay_max={oracle_rec['metrics']['a_y_max_trim_g']:.2f}g)")
        ax.plot(bx, by_max, color="#2ca02c", lw=1.6, linestyle="-.")
        ax.plot(bx, by_min, color="#2ca02c", lw=1.6, linestyle="-.")
    # Mark Oracle trim max Ay
    ax.scatter(
        [oracle_rec["metrics"]["a_y_max_trim_g"], -oracle_rec["metrics"]["a_y_max_trim_g"]],
        [0.0, 0.0],
        color="#2ca02c",
        marker="*",
        s=130,
        zorder=6,
        label=f"Oracle Trim Limit ({oracle_rec['metrics']['a_y_max_trim_g']:.2f}g)",
    )

    series = [
        ("Open-Diff", open_rec, "#7f7f7f", "--", 0.40, 2.0),
        ("TV 4WD Baseline", tv_base_rec, "#1f77b4", "-", 0.55, 2.2),
    ]
    if tv_opt_rec is not None:
        series.append(("TV 4WD Optimized", tv_opt_rec, "#d62728", "-", 0.72, 2.5))

    i_b0 = int(np.argmin(np.abs(BETA_GRID_DEG)))
    for name, rec, col, ls, alpha_mesh, lw_spine in series:
        ay = np.asarray(rec["ay_g"])
        mz = np.asarray(rec["mz_nm"])
        for i in range(ay.shape[0]):
            ax.plot(ay[i], mz[i], color=col, alpha=alpha_mesh * 0.7, lw=0.95, linestyle=ls)
        for j in range(ay.shape[1]):
            if abs(DELTA_GRID_DEG[j]) in (0.0, 4.0, 8.0, 16.0):
                ax.plot(ay[:, j], mz[:, j], color=col, alpha=alpha_mesh * 0.45, lw=0.75, linestyle=":")
        m_ay = rec["metrics"]["a_y_max_trim_g"]
        m_res = rec["metrics"]["control_reserve_mz_at_limit_nm"]
        ax.plot(
            ay[i_b0],
            mz[i_b0],
            color=col,
            lw=lw_spine,
            linestyle=ls,
            label=f"{name} (Ay_max={m_ay:.2f}g, Mz_res={m_res:.0f}Nm)",
        )

    ax.axhline(0.0, color="#111111", lw=1.1)
    ax.axvline(0.0, color="#888888", lw=0.8, linestyle="--")
    ax.set_xlabel("Lateral Acceleration Ay [g]")
    ax.set_ylabel("Net Yaw Moment Mz [Nm] (ISO 8855)")
    title_suffix = "4-Way Final Comparison" if tv_opt_rec is not None else "Open-Diff vs TV Baseline vs Physical Oracle"
    ax.set_title(f"TeR27 4WD Yaw Moment Diagram @ {v_kmh:.0f} km/h (FSG_REAL_TRACK) - {title_suffix}")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Milliken YMD for TeR27 4WD at 30, 55, 80 km/h.")
    parser.add_argument("--profile", default="FSG_REAL_TRACK", choices=["FSG_REAL_TRACK", "TTC_LAB_PEAK"])
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Load cached Open-Diff/TV-Baseline/Oracle and overlay current tv_4wd_config.json as TV Optimized.",
    )
    parser.add_argument("--tv-config", type=str, default=str(TV_CFG_PATH), help="Path to tv_4wd_config.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RESULTS_DIR / "ymd_baseline_cache.json"
    model = TeR27PhysicsModel(profile=args.profile)
    tv_law = TV4WdSteadyLaw(Path(args.tv_config))

    print(f"=== TeR27 4WD YMD Generator ({args.profile}, include_final={args.include_final}) ===", flush=True)

    if args.include_final and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        open_by_v = cache["OPEN_DIFF"]
        base_by_v = cache["TV_BASELINE"]
        oracle_by_v = cache["ORACLE"]
        opt_by_v: Dict[str, object] = {}
        for v_kmh in SPEEDS_KMH:
            key = f"{int(v_kmh)}kmh"
            print(f"  [Final] Sweeping TV_OPTIMIZED @ {v_kmh:.0f} km/h...", flush=True)
            opt_by_v[key] = sweep_controller(model, v_kmh, "TV_OPTIMIZED", tv_law)
    else:
        open_by_v = {}
        base_by_v = {}
        oracle_by_v = {}
        opt_by_v = None  # type: ignore[assignment]
        for v_kmh in SPEEDS_KMH:
            key = f"{int(v_kmh)}kmh"
            print(f"  Sweeping @ {v_kmh:.0f} km/h (Open-Diff, TV Baseline, Oracle)...", flush=True)
            open_by_v[key] = sweep_controller(model, v_kmh, "OPEN_DIFF", None)
            base_by_v[key] = sweep_controller(model, v_kmh, "TV_BASELINE", tv_law)
            oracle_by_v[key] = sweep_oracle(model, v_kmh)
            print(
                f"    {key}: Open Ay_max={open_by_v[key]['metrics']['a_y_max_trim_g']:.3f}g | "
                f"TV_Base Ay_max={base_by_v[key]['metrics']['a_y_max_trim_g']:.3f}g "
                f"(Mz_res={base_by_v[key]['metrics']['control_reserve_mz_at_limit_nm']:.0f}Nm) | "
                f"Oracle Ay_max={oracle_by_v[key]['metrics']['a_y_max_trim_g']:.3f}g "
                f"(Mz_res={oracle_by_v[key]['metrics']['control_reserve_mz_at_limit_nm']:.0f}Nm)",
                flush=True,
            )
        cache_path.write_text(
            json.dumps({"profile": args.profile, "OPEN_DIFF": open_by_v, "TV_BASELINE": base_by_v, "ORACLE": oracle_by_v}),
            encoding="utf-8",
        )

    metrics_out: Dict[str, object] = {
        "vehicle_id": "ter27_4wd",
        "profile": args.profile,
        "include_final": bool(args.include_final),
        "speeds_kmh": SPEEDS_KMH,
        "results": {},
    }

    for v_kmh in SPEEDS_KMH:
        key = f"{int(v_kmh)}kmh"
        out_png = RESULTS_DIR / f"ymd_{int(v_kmh)}kmh.png"
        opt_rec = opt_by_v[key] if opt_by_v is not None else None
        plot_single_speed_ymd(v_kmh, open_by_v[key], base_by_v[key], oracle_by_v[key], opt_rec, out_png)
        entry = {
            "OPEN_DIFF": open_by_v[key]["metrics"],
            "TV_BASELINE": base_by_v[key]["metrics"],
            "ORACLE": oracle_by_v[key]["metrics"],
        }
        if opt_rec is not None:
            entry["TV_OPTIMIZED"] = opt_rec["metrics"]
        metrics_out["results"][key] = entry  # type: ignore[index]
        print(f"  Saved plot -> {out_png}", flush=True)

    metrics_path = RESULTS_DIR / "ymd_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
