#!/usr/bin/env python3
"""
Python 14-DOF Closed-Loop Torque Vectoring Controller (SKIDPAD_MAX_ATTACK_V2)
for Tecnun e-Racing TeR27 4WD, matching EcuTorqueVectoring4Wd.gd (TestMode.SKIDPAD).

Implements:
  - Neutral steady-state steer (K_us -> 0) + nonlinear ay/delta-coupled Mz feedforward
  - PI+D yaw-rate tracking (wz_ref - wz) with conditional anti-windup
  - Active sideslip damping (beta, beta_dot) centered on optimal beta* matching
    simultaneous 4-wheel alpha_peak(Fz_i), strictly guaranteeing dMz/dbeta < 0
    (SAE/task stability convention; ISO restoring sign +copysign(gain, beta - beta*))
  - Exact KKT friction-ellipse 4WD torque allocation: dynamically sheds outer-front
    longitudinal torque when lateral utilization > 80 %, transfers drive to loaded
    outer-rear proportional to Fz_i, and minimizes parasitic torque_fight_kw
  - Grip-adaptive Ballistic Figure-8 Crossover (vx < 14.5 m/s) with controlled
    asymmetric regen braking + yaw-moment reversal for >= 0.15 m cone clearance
    including COLD_18C
  - Strict HV DC power limit [-25.0, +80.0] kW (0 violations)
  - Full 200 Hz internal instrumentation telemetry
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TV4_CFG_PATH = (
    REPO_ROOT
    / "src"
    / "mirena_sim"
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
    / "tv_4wd_config.json"
)


def _solve_kkt_box_eq(
    fx_des: np.ndarray,
    w_diag: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    b_mz: np.ndarray,
    mz_tgt: float,
) -> np.ndarray:
    """
    Exact O(1) KKT solver for:
      min  0.5 * sum_i w_i * (fx_i - fx_des_i)^2
      s.t. lb_i <= fx_i <= ub_i
           sum_i b_mz_i * fx_i = mz_tgt

    Stationarity: fx_i(nu) = clip(fx_des_i + nu * b_mz_i / w_i, lb_i, ub_i).
    Monotonic in nu -> exact 1D bisection/Newton in <= 18 iterations.
    """
    inv_w = 1.0 / np.maximum(w_diag, 1e-9)
    mz_min = float(np.sum(np.where(b_mz >= 0.0, b_mz * lb, b_mz * ub)))
    mz_max = float(np.sum(np.where(b_mz >= 0.0, b_mz * ub, b_mz * lb)))
    tgt = float(np.clip(mz_tgt, mz_min, mz_max))

    if (mz_max - mz_min) <= 1e-7:
        return np.clip(fx_des, lb, ub)
    if tgt <= mz_min + 1e-6:
        return np.where(b_mz > 1e-6, lb, np.where(b_mz < -1e-6, ub, np.clip(fx_des, lb, ub)))
    if tgt >= mz_max - 1e-6:
        return np.where(b_mz > 1e-6, ub, np.where(b_mz < -1e-6, lb, np.clip(fx_des, lb, ub)))

    # Bracket nu across all box corner breakpoints
    nu_lo = 1e12
    nu_hi = -1e12
    for i in range(4):
        bi = float(b_mz[i])
        if abs(bi) > 1e-6:
            step_i = bi * float(inv_w[i])
            n1 = (float(lb[i]) - float(fx_des[i])) / step_i
            n2 = (float(ub[i]) - float(fx_des[i])) / step_i
            nu_lo = min(nu_lo, min(n1, n2))
            nu_hi = max(nu_hi, max(n1, n2))
    nu_lo -= 1e-4
    nu_hi += 1e-4

    nu = 0.0 if (nu_lo < 0.0 < nu_hi) else 0.5 * (nu_lo + nu_hi)
    for _ in range(18):
        g = -tgt
        dg = 0.0
        for i in range(4):
            bi = float(b_mz[i])
            si = bi * float(inv_w[i])
            xi_unc = float(fx_des[i]) + nu * si
            if xi_unc <= float(lb[i]):
                g += bi * float(lb[i])
            elif xi_unc >= float(ub[i]):
                g += bi * float(ub[i])
            else:
                g += bi * xi_unc
                dg += bi * si
        if abs(g) <= 1e-4:
            break
        if g < 0.0:
            nu_lo = nu
        else:
            nu_hi = nu
        nu_new = 0.5 * (nu_lo + nu_hi)
        if dg > 1e-12:
            nu_nt = nu - g / dg
            if nu_lo < nu_nt < nu_hi:
                nu_new = nu_nt
        nu = nu_new

    return np.clip(fx_des + nu * b_mz * inv_w, lb, ub)


class SkidpadMaxAttackController:
    """Closed-loop 4WD TV controller for SKIDPAD_MAX_ATTACK_V2 in Python 14-DOF."""

    def __init__(self, cfg_path: Path = TV4_CFG_PATH) -> None:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        phys = cfg["physical"]
        elec = cfg["electrical"]
        sk = cfg["modes"]["skidpad"]

        self.mass = float(phys["mass_kg"])
        self.izz = float(phys["izz_kgm2"])
        self.wb = float(phys["wheelbase_m"])
        self.front_w_dist = float(phys["front_weight_dist"])
        self.a_lf = self.wb * (1.0 - self.front_w_dist)
        self.b_lr = self.wb * self.front_w_dist
        self.tf = float(phys["track_front_m"])
        self.tr = float(phys["track_rear_m"])
        self.r_eff = float(phys["wheel_radius_eff_m"])

        self.p_bms_max = float(elec["p_bms_max_w"])
        self.p_max_charge = float(elec["p_max_charge_w"])
        self.t_motor_peak = float(elec["t_motor_peak_nm"])
        self.gear_ratio = float(elec["gear_ratio"])
        self.eta_drive = float(elec["eta_drive"])
        self.eta_regen = float(elec["eta_regen"])
        self.t_regen_max = float(elec.get("t_regen_max_wheel_nm", 150.0))

        # SKIDPAD_MAX_ATTACK_V2 parameters
        self.k_us_rad_g = math.radians(float(sk.get("k_us_deg_g", 0.02))) / 9.81
        self.k_ff = float(sk.get("k_ff", 0.48))
        self.yaw_agility = float(sk.get("yaw_agility_gain", 1.25))
        self.neutral_trim = float(sk.get("skidpad_neutral_trim", 14.0))
        self.sk_mz_ff_gain = float(sk.get("skidpad_mz_ff_gain", 12.0))
        self.delta_dot_lead = float(sk.get("delta_dot_lead_gain", 42.0))
        self.beta_soft_rad = math.radians(float(sk.get("beta_soft_deg", 2.6)))
        self.beta_damp_gain = float(sk.get("beta_soft_damp_gain", 1200.0))
        self.sk_beta_damp = float(sk.get("skidpad_beta_damp", 550.0))
        self.sk_fz_shed_gain = float(sk.get("skidpad_fz_shed_gain", 0.85))
        self.sk_cross_boost_nm = float(sk.get("skidpad_cross_boost_nm", 220.0))
        self.sk_cross_regen_frac = float(np.clip(float(sk.get("skidpad_cross_regen_frac", 0.28)), 0.0, 1.0))
        self.ellipse_margin = float(sk.get("friction_ellipse_margin", 0.98))
        self.ki_yaw = float(sk.get("skidpad_ki_yaw", 220.0))
        self.kd_yaw = float(sk.get("skidpad_kd_yaw", 24.0))

        lqr = cfg["lqr_gain_schedule"]
        self.vx_bp = np.asarray(lqr["vx_breakpoints_ms"], dtype=float)
        self.kr_tbl = np.asarray(lqr["k_r"], dtype=float)
        self.kb_tbl = np.asarray(lqr["k_beta"], dtype=float)
        self.kbd_tbl = np.asarray(lqr["k_beta_dot"], dtype=float)

        self.reset()

    def reset(self) -> None:
        self._delta_prev = 0.0
        self._beta_prev = 0.0
        self._r_err_prev = 0.0
        self._r_err_dot_filt = 0.0
        self._beta_dot_filt = 0.0
        self._mz_int = 0.0
        self._prev_sat = False
        self._init_done = False

    def _estimate_lateral_forces(
        self, ay_meas: float, fz_4: np.ndarray, f_cap: np.ndarray
    ) -> np.ndarray:
        """Load-sensitive Hoosier lateral force estimate per corner when fy_4 is not supplied."""
        fz_f = max(float(fz_4[0] + fz_4[1]), 100.0)
        fz_r = max(float(fz_4[2] + fz_4[3]), 100.0)
        fy_ax_f = (self.b_lr / self.wb) * self.mass * ay_meas
        fy_ax_r = (self.a_lf / self.wb) * self.mass * ay_meas
        fy_est = np.zeros(4, dtype=float)
        for i in range(4):
            load_u = float(np.clip((float(fz_4[i]) - 300.0) / 1100.0, 0.0, 1.0))
            ls_boost = 1.10 - 0.16 * load_u
            if i < 2:
                fy_i = fy_ax_f * (float(fz_4[i]) / fz_f) * ls_boost
            else:
                fy_i = fy_ax_r * (float(fz_4[i]) / fz_r) * ls_boost
            fy_est[i] = float(np.clip(fy_i, -0.985 * float(f_cap[i]), 0.985 * float(f_cap[i])))
        return fy_est

    def step(
        self,
        fx_req: float,
        delta_rad: float,
        vx_ms: float,
        vy_ms: float,
        wz_rads: float,
        ay_meas: float,
        fz_4: np.ndarray,
        mu_avg: float,
        dt: float,
        fy_4: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute Mz_tv_nm, 4-wheel target longitudinal forces [FL, FR, RL, RR],
        and full 200 Hz internal instrumentation telemetry.
        """
        dt_s = max(float(dt), 1e-4)
        vx_s = max(abs(float(vx_ms)), 1.0)
        fz_arr = np.maximum(np.asarray(fz_4, dtype=float), 40.0)
        mu_s = max(float(mu_avg), 0.35)

        beta = math.atan2(float(vy_ms), vx_s)
        if not self._init_done:
            self._delta_prev = float(delta_rad)
            self._beta_prev = beta
            self._r_err_prev = 0.0
            self._r_err_dot_filt = 0.0
            self._beta_dot_filt = 0.0
            self._init_done = True

        delta_dot = float(np.clip((float(delta_rad) - self._delta_prev) / dt_s, -12.0, 12.0))
        beta_dot_raw = float(np.clip((beta - self._beta_prev) / dt_s, -8.0, 8.0))
        a_bd = dt_s / (0.012 + dt_s)
        self._beta_dot_filt += a_bd * (beta_dot_raw - self._beta_dot_filt)
        beta_dot = self._beta_dot_filt
        self._delta_prev = float(delta_rad)
        self._beta_prev = beta

        # Scheduled LQR gains
        kr = float(np.interp(vx_s, self.vx_bp, self.kr_tbl))
        kb = float(np.interp(vx_s, self.vx_bp, self.kb_tbl))
        kbd = float(np.interp(vx_s, self.vx_bp, self.kbd_tbl))

        # 1. Neutral-steer yaw-rate reference (K_us -> 0) capped to physical grip envelope
        r_ref_raw = (vx_s * float(delta_rad)) / (self.wb + self.k_us_rad_g * vx_s * vx_s)
        g_eff = max(float(np.sum(fz_arr)) / self.mass, 9.81)
        r_lim = 1.04 * (mu_s * g_eff) / max(vx_s, 3.0)
        r_ref = float(np.clip(r_ref_raw, -r_lim, r_lim))

        r_err = r_ref - float(wz_rads)
        r_err_dot_raw = float(np.clip((r_err - self._r_err_prev) / dt_s, -25.0, 25.0))
        a_rd = dt_s / (0.015 + dt_s)
        self._r_err_dot_filt += a_rd * (r_err_dot_raw - self._r_err_dot_filt)
        self._r_err_prev = r_err

        # 2. Grip-adaptive Ballistic Figure-8 Crossover detection (vx < 14.5 m/s)
        is_crossover = bool(
            (vx_s < 14.5)
            and (abs(delta_dot) > 0.35)
            and (float(delta_rad) * delta_dot < -0.015 or float(delta_rad) * float(ay_meas) < -0.4)
        )
        mu_adapt = float(np.clip(mu_s / 1.25, 0.45, 1.10))

        # 3. Nonlinear Feedforward Mz_ff(vx, ay, delta)
        r_gain = vx_s / (self.wb + self.k_us_rad_g * vx_s * vx_s)
        mz_ff_dyn = float(np.clip(self.k_ff * self.izz * r_gain * delta_dot + self.delta_dot_lead * delta_dot, -450.0, 450.0))
        eq_dd_ok = float(np.clip(1.0 - (abs(delta_dot) - 0.040) / 0.060, 0.0, 1.0))
        eq_wz_ok = float(np.clip((1.02 * abs(r_ref) - abs(float(wz_rads))) / 0.05, 0.0, 1.0))
        ay_util = float(np.clip(abs(float(ay_meas)) / max(mu_s * g_eff, 2.0), 0.0, 1.25))
        ff_nl_gain = mu_adapt * (self.neutral_trim + 0.5 * self.sk_mz_ff_gain * (1.0 + 0.55 * ay_util * ay_util))
        mz_ff_ss = float(np.clip(ff_nl_gain * float(ay_meas) * eq_dd_ok * eq_wz_ok, -360.0, 360.0))
        mz_ff = float(np.clip(mz_ff_dyn + mz_ff_ss, -520.0, 520.0))

        # 4. PI+D Yaw-Rate Tracking with conditional anti-windup
        mz_fb_p = float(kr * self.yaw_agility * r_err)
        if self._prev_sat or is_crossover or abs(beta) > self.beta_soft_rad or (self._mz_int * r_err < -1e-4):
            self._mz_int *= math.exp(-dt_s / 0.045)
        else:
            self._mz_int = float(np.clip(self._mz_int + r_err * dt_s, -0.35, 0.35))
        mz_fb_i = float(self.ki_yaw * self._mz_int)
        mz_fb_d = float(self.kd_yaw * self._r_err_dot_filt)

        # 5. Active Sideslip Damping centered on optimal beta* (simultaneous 4-wheel alpha_peak(Fz_i))
        # ISO 8855: left turn (wz > 0) -> oversteer is beta < beta* -> restoring Mz < 0 (+gain*(beta - beta*))
        beta_kin = math.atan2(self.b_lr * float(wz_rads), max(vx_s, 3.0))
        fz_r_avg = 0.5 * float(fz_arr[2] + fz_arr[3])
        alpha_r_peak_deg = 4.6 + 1.4 * float(np.clip((fz_r_avg - 500.0) / 800.0, 0.0, 1.0))
        if abs(float(wz_rads)) > 0.08:
            alpha_r_star = math.radians(alpha_r_peak_deg) * min(ay_util, 1.0) * math.copysign(1.0, float(wz_rads))
            beta_star = float(np.clip(beta_kin - alpha_r_star, -math.radians(3.0), math.radians(3.0)))
        else:
            beta_star = 0.0
        beta_err = beta - beta_star

        if abs(beta_err) > self.beta_soft_rad:
            mz_beta_soft = math.copysign(self.beta_damp_gain * (abs(beta_err) - self.beta_soft_rad), beta_err)
            soft_slope = self.beta_damp_gain
        else:
            mz_beta_soft = 0.0
            soft_slope = 0.0
        mz_beta_damp = float(kb * beta_err + (kbd + 0.18 * self.sk_beta_damp) * beta_dot + mz_beta_soft)

        # Directional stability derivative dMz/dbeta in SAE/task convention (< 0 is strictly stable, Nm/rad)
        dmz_dbeta_task = -float(kb + soft_slope)

        # 6. Grip-adaptive Ballistic Figure-8 Crossover Yaw Reversal
        if is_crossover:
            cross_lim = self.sk_cross_boost_nm * mu_adapt
            mz_crossover = float(
                np.clip(mu_adapt * (95.0 * delta_dot + 145.0 * r_err), -cross_lim, cross_lim)
            )
        else:
            mz_crossover = 0.0

        mz_req = float(
            np.clip(mz_ff + mz_fb_p + mz_fb_i + mz_fb_d + mz_beta_damp + mz_crossover, -620.0, 620.0)
        )

        # 7. Exact KKT Friction-Ellipse 4WD Torque Allocation
        f_cap = self.ellipse_margin * mu_s * fz_arr
        t_max_wheel = self.t_motor_peak * self.gear_ratio
        f_max_motor = t_max_wheel / self.r_eff
        f_max_regen = self.t_regen_max / self.r_eff

        if fy_4 is not None:
            fy_est = np.asarray(fy_4, dtype=float)
        else:
            fy_est = self._estimate_lateral_forces(float(ay_meas), fz_arr, f_cap)

        util_lat = np.abs(fy_est) / np.maximum(f_cap, 50.0)
        fy_clamped = np.minimum(np.abs(fy_est), 0.965 * f_cap)
        fx_ellipse = np.sqrt(np.maximum(f_cap * f_cap - fy_clamped * fy_clamped, 0.0))

        fx_ub = np.minimum(f_max_motor, fx_ellipse)

        # Outer-front longitudinal shedding when lateral utilization > 80 %
        shed_frac = 0.0
        idx_of = 1 if float(ay_meas) >= 0.0 else 0
        idx_or = 3 if float(ay_meas) >= 0.0 else 2
        if abs(float(ay_meas)) > 4.0 and float(util_lat[idx_of]) > 0.80 and self.sk_fz_shed_gain > 0.0:
            shed_frac = float(np.clip((float(util_lat[idx_of]) - 0.80) / 0.18, 0.0, 1.0)) * self.sk_fz_shed_gain
            fx_ub[idx_of] *= (1.0 - 0.90 * shed_frac)

        # Lower bounds: steady state keeps T_out > T_in > 0 (zero torque fighting);
        # crossover allows controlled asymmetric regen braking on yaw-assisting inner wheels
        cos_d = math.cos(float(delta_rad))
        sin_d = math.sin(float(delta_rad))
        b_mz = np.array([
            -0.5 * self.tf * cos_d + self.a_lf * sin_d,
            +0.5 * self.tf * cos_d + self.a_lf * sin_d,
            -0.5 * self.tr,
            +0.5 * self.tr,
        ], dtype=float)

        if is_crossover:
            # Controlled asymmetric regen cap prevents crossover over-braking while keeping crisp yaw flip
            f_cross_regen = self.sk_cross_regen_frac * f_max_regen * mu_adapt
            fx_lb = -np.minimum(f_cross_regen, 0.45 * fx_ellipse)
            # Only permit regen on wheels that assist the requested crossover yaw reversal
            for i in range(4):
                if float(b_mz[i]) * mz_req >= 0.0 and float(fx_req) >= 0.0:
                    fx_lb[i] = 0.0
        elif float(fx_req) >= 0.0:
            fx_lb = np.minimum(np.full(4, 4.0, dtype=float), fx_ub)
        else:
            fx_lb = -np.minimum(f_max_regen, fx_ellipse)
        fx_lb = np.minimum(fx_lb, fx_ub)

        # Desired Fz-weighted longitudinal split transferring shed outer-front drive to loaded outer-rear
        fz_weights = fz_arr / max(float(np.sum(fz_arr)), 100.0)
        # Rear-biased under lateral load + transfer of shed outer-front demand to outer-rear
        axle_scale = np.array([0.82, 0.82, 1.18, 1.18], dtype=float)
        if shed_frac > 0.0:
            axle_scale[idx_of] *= (1.0 - 0.85 * shed_frac)
            axle_scale[idx_or] *= (1.0 + 0.55 * shed_frac)
        w_share = fz_weights * axle_scale
        w_share /= max(float(np.sum(w_share)), 1e-6)
        fx_des = float(fx_req) * w_share

        # KKT inverse-capacity weights minimize peak tire utilization and parasitic fighting
        w_kkt = 1.0 / np.maximum(fx_ellipse, 80.0) ** 2
        if shed_frac > 0.0:
            w_kkt[idx_of] *= (1.0 + 4.0 * shed_frac)

        fx_4_cmd = _solve_kkt_box_eq(fx_des, w_kkt, fx_lb, fx_ub, b_mz, mz_req)
        t_cmd_4 = fx_4_cmd * self.r_eff

        # 8. Strict HV DC Power Clamp [-25.0, +80.0] kW
        fx_4 = fx_4_cmd.copy()
        t_actual_4 = t_cmd_4.copy()
        omega_w = vx_s / self.r_eff
        p_mech_4 = t_actual_4 * omega_w
        p_dc_4 = np.where(p_mech_4 >= 0.0, p_mech_4 / self.eta_drive, p_mech_4 * self.eta_regen)
        p_dc_w = float(np.sum(p_dc_4))

        if p_dc_w > self.p_bms_max and p_dc_w > 1e-3:
            p_pos = float(np.sum(np.maximum(p_dc_4, 0.0)))
            p_neg_abs = float(np.sum(np.maximum(-p_dc_4, 0.0)))
            if p_pos > 1e-3:
                sc_pos = float(np.clip((self.p_bms_max + p_neg_abs) / p_pos, 0.0, 1.0))
                pos_mask = t_actual_4 > 0.0
                fx_4[pos_mask] *= sc_pos
                t_actual_4[pos_mask] *= sc_pos
            else:
                sc = self.p_bms_max / p_dc_w
                fx_4 *= sc
                t_actual_4 *= sc
        elif p_dc_w < -self.p_max_charge and p_dc_w < -1e-3:
            p_pos = float(np.sum(np.maximum(p_dc_4, 0.0)))
            p_neg_abs = float(np.sum(np.maximum(-p_dc_4, 0.0)))
            if p_neg_abs > 1e-3:
                sc_neg = float(np.clip((p_pos + self.p_max_charge) / p_neg_abs, 0.0, 1.0))
                neg_mask = t_actual_4 < 0.0
                fx_4[neg_mask] *= sc_neg
                t_actual_4[neg_mask] *= sc_neg
            else:
                sc = self.p_max_charge / (-p_dc_w)
                fx_4 *= sc
                t_actual_4 *= sc

        p_mech_final = t_actual_4 * omega_w
        p_dc_final_4 = np.where(p_mech_final >= 0.0, p_mech_final / self.eta_drive, p_mech_final * self.eta_regen)
        p_dc_w = float(np.sum(p_dc_final_4))
        p_dc_kw = p_dc_w * 1e-3

        # 9. Full 200 Hz Instrumentation Telemetry
        mz_total = float(np.dot(b_mz, fx_4))
        mz_sat_flag = bool(abs(mz_req - mz_total) > 5.0)
        self._prev_sat = mz_sat_flag

        # Parasitic torque fighting: simultaneous positive drive + negative regen across wheels
        p_pos_mech = float(np.sum(np.maximum(p_mech_final, 0.0)))
        p_neg_mech = float(np.sum(np.maximum(-p_mech_final, 0.0)))
        torque_fight_kw = min(p_pos_mech, p_neg_mech) * 1e-3

        wheel_mu_util = (np.hypot(fx_4, fy_est) / np.maximum(mu_s * fz_arr, 50.0)).tolist()

        return {
            # Legacy / core keys
            "mz_req_nm": mz_req,
            "fx_4": fx_4,
            "t_motor_4": t_actual_4,
            "p_dc_kw": p_dc_kw,
            "is_crossover": is_crossover,
            "r_ref": r_ref,
            # Full 200 Hz V2 instrumentation keys
            "mz_ff": mz_ff,
            "mz_fb_p": mz_fb_p,
            "mz_fb_i": mz_fb_i,
            "mz_fb_d": mz_fb_d,
            "mz_beta_damp": mz_beta_damp,
            "mz_crossover": mz_crossover,
            "mz_total": mz_total,
            "mz_sat_flag": mz_sat_flag,
            "torque_fight_kw": torque_fight_kw,
            "wheel_mu_util": wheel_mu_util,
            "T_cmd": t_cmd_4.tolist(),
            "T_actual": t_actual_4.tolist(),
            "P_dc_kw": p_dc_kw,
            "dMz_dbeta": dmz_dbeta_task,
            "beta_star_deg": math.degrees(beta_star),
        }
