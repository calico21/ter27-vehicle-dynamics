#!/usr/bin/env python3
"""
Online Grip Estimator (lambda_mu_hat) and Tire Relaxation Length Precompensator
(sigma_alpha, sigma_kappa) for Tecnun e-Racing TeR27 4WD.

Deterministic, float32-compatible, zero-heap-allocation runtime modules designed
for 200 Hz STM32F405 execution (subsampled 50-100 Hz RLS grip updates, 200 Hz
tire relaxation lead-lag precompensation) and 1-to-1 parity with C99 / Godot 4.

Standards preserved:
  - COLD_TEST_OCT nominal grip reference: lambda_mu = 0.47 (mu_scale_hat = 1.000)
  - FSG_RUBBERED grip target:             lambda_mu = 0.57 (mu_scale_hat = 1.212766)
  - COLD_18C / COLD_LOW_GRIP target:      lambda_mu = 0.3384 (mu_scale_hat = 0.720)
  - Directional stability dMz/dbeta < 0 untouched (precompensator acts on FF only)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shared Physical & Control Constants (float32-exact, pure ASCII)
# ---------------------------------------------------------------------------
LAMBDA_MU_COLD_TEST_OCT: float = 0.47
LAMBDA_MU_FSG_RUBBERED: float = 0.57
LAMBDA_MU_COLD_18C: float = 0.3384  # 0.47 * 0.72 (COLD_LOW_GRIP / COLD_18C)
LAMBDA_MU_MIN: float = 0.32
LAMBDA_MU_MAX: float = 0.72

HOOSIER_SIGMA_ALPHA_M: float = 0.24  # Lateral carcass relaxation length [m]
HOOSIER_SIGMA_KAPPA_M: float = 0.14  # Longitudinal carcass relaxation length [m]
RELAX_VX_MIN_MS: float = 1.5         # Speed floor for tau_rel = sigma / max(vx, v_min)
RELAX_FADE_START_MS: float = 8.0     # Smooth fade-out start speed [m/s]
RELAX_FADE_END_MS: float = 12.0      # Smooth fade-out end speed [m/s]
LAUNCH_DIST_FADE_START_M: float = 12.0
LAUNCH_DIST_FADE_END_M: float = 15.0


def _f32(val: float) -> float:
    """Round scalar to IEEE-754 binary32 precision for deterministic STM32 parity."""
    return float(np.float32(val))


def _clamp_f32(val: float, lo: float, hi: float) -> float:
    v = _f32(val)
    if v < lo:
        return _f32(lo)
    if v > hi:
        return _f32(hi)
    return v


def _smoothstep_falloff_f32(x: float, x_lo: float, x_hi: float) -> float:
    """Smooth cubic falloff: 1.0 for x <= x_lo, 0.0 for x >= x_hi."""
    if x <= x_lo:
        return _f32(1.0)
    if x >= x_hi:
        return _f32(0.0)
    u = _f32((x_hi - x) / max(x_hi - x_lo, 1e-6))
    return _f32(u * u * (_f32(3.0) - _f32(2.0) * u))


class OnlineGripEstimator:
    """
    Deterministic float32 Scalar-RLS / Scalar-EKF Online Tire-Road Grip Estimator.

    Estimates live effective tire friction scaling factor lambda_mu_hat in
    [0.32, 0.72] by comparing Smith-predictor-compensated vehicle accelerations
    (ax_comp, ay_comp) against nominal MF62+PINN predicted vehicle forces
    (fx_nom_n, fy_nom_n) / mass_kg evaluated at lambda_mu_nom = 0.47.

    Measurement model (directional projection along nominal force vector):
        a_nom_x = fx_nom_n / mass_kg
        a_nom_y = fy_nom_n / mass_kg
        a_nom   = hypot(a_nom_x, a_nom_y)
        phi     = a_nom / lambda_mu_nom
        y       = (ax_comp * a_nom_x + ay_comp * a_nom_y) / a_nom
        y       = lambda_mu * phi + noise

    Using vector projection instead of scalar magnitude eliminates Ricean noise
    bias and handles pure longitudinal, pure lateral, and combined-slip regimes
    identically with 0 heap allocations in step().
    """

    __slots__ = (
        "mass_kg",
        "lambda_mu_nom",
        "lambda_mu_init",
        "lambda_mu_min",
        "lambda_mu_max",
        "accel_gate_ms2",
        "nom_accel_gate_ms2",
        "rls_forgetting",
        "r_meas",
        "q_proc",
        "p_init",
        "p_min",
        "p_max",
        "innov_clamp_ms2",
        "rate_limit_per_s",
        "subsample_div",
        "kappa_base",
        "kappa_fz_slope",
        "alpha_base_deg",
        "alpha_fz_slope_deg",
        "lambda_mu_hat",
        "mu_scale_hat",
        "p_cov",
        "last_innov_ms2",
        "gate_active",
        "update_count",
        "_sub_count",
        "_acc_ax",
        "_acc_ay",
        "_acc_fx_nom",
        "_acc_fy_nom",
        "_acc_dt",
    )

    def __init__(
        self,
        mass_kg: float = 290.0,
        lambda_mu_init: float = LAMBDA_MU_COLD_TEST_OCT,
        lambda_mu_nom: float = LAMBDA_MU_COLD_TEST_OCT,
        lambda_mu_min: float = LAMBDA_MU_MIN,
        lambda_mu_max: float = LAMBDA_MU_MAX,
        accel_gate_ms2: float = 3.5,
        nom_accel_gate_ms2: float = 2.0,
        rls_forgetting: float = 0.945,
        r_meas: float = 1.20,
        q_proc: float = 1.5e-4,
        p_init: float = 8.0e-4,
        p_min: float = 1.0e-5,
        p_max: float = 4.0e-3,
        innov_clamp_ms2: float = 4.5,
        rate_limit_per_s: float = 0.65,
        subsample_div: int = 2,
        kappa_base: float = 0.095,
        kappa_fz_slope: float = 0.028,
        alpha_base_deg: float = 5.8,
        alpha_fz_slope_deg: float = 1.5,
    ) -> None:
        self.mass_kg: float = _f32(max(mass_kg, 50.0))
        self.lambda_mu_nom: float = _f32(max(lambda_mu_nom, 0.10))
        self.lambda_mu_min: float = _f32(lambda_mu_min)
        self.lambda_mu_max: float = _f32(lambda_mu_max)
        self.lambda_mu_init: float = _clamp_f32(lambda_mu_init, self.lambda_mu_min, self.lambda_mu_max)
        self.accel_gate_ms2: float = _f32(max(accel_gate_ms2, 0.5))
        self.nom_accel_gate_ms2: float = _f32(max(nom_accel_gate_ms2, 0.5))
        self.rls_forgetting: float = _clamp_f32(rls_forgetting, 0.80, 0.999)
        self.r_meas: float = _f32(max(r_meas, 1e-3))
        self.q_proc: float = _f32(max(q_proc, 0.0))
        self.p_init: float = _f32(max(p_init, 1e-6))
        self.p_min: float = _f32(max(p_min, 1e-7))
        self.p_max: float = _f32(max(p_max, self.p_min))
        self.innov_clamp_ms2: float = _f32(max(innov_clamp_ms2, 0.5))
        self.rate_limit_per_s: float = _f32(max(rate_limit_per_s, 0.05))
        self.subsample_div: int = max(1, int(subsample_div))
        self.kappa_base: float = _f32(kappa_base)
        self.kappa_fz_slope: float = _f32(kappa_fz_slope)
        self.alpha_base_deg: float = _f32(alpha_base_deg)
        self.alpha_fz_slope_deg: float = _f32(alpha_fz_slope_deg)

        self.lambda_mu_hat: float = self.lambda_mu_init
        self.mu_scale_hat: float = _f32(self.lambda_mu_hat / self.lambda_mu_nom)
        self.p_cov: float = self.p_init
        self.last_innov_ms2: float = _f32(0.0)
        self.gate_active: bool = False
        self.update_count: int = 0
        self._sub_count: int = 0
        self._acc_ax: float = _f32(0.0)
        self._acc_ay: float = _f32(0.0)
        self._acc_fx_nom: float = _f32(0.0)
        self._acc_fy_nom: float = _f32(0.0)
        self._acc_dt: float = _f32(0.0)

    def reset(self, lambda_mu_init: Optional[float] = None) -> None:
        """Reset estimator state to initial grip belief with zero allocation."""
        init_val = self.lambda_mu_init if lambda_mu_init is None else lambda_mu_init
        self.lambda_mu_hat = _clamp_f32(init_val, self.lambda_mu_min, self.lambda_mu_max)
        self.mu_scale_hat = _f32(self.lambda_mu_hat / self.lambda_mu_nom)
        self.p_cov = self.p_init
        self.last_innov_ms2 = _f32(0.0)
        self.gate_active = False
        self.update_count = 0
        self._sub_count = 0
        self._acc_ax = _f32(0.0)
        self._acc_ay = _f32(0.0)
        self._acc_fx_nom = _f32(0.0)
        self._acc_fy_nom = _f32(0.0)
        self._acc_dt = _f32(0.0)

    def compute_scaled_target_slips(self, fz_n: float = 711.225) -> Tuple[float, float]:
        """
        Compute grip- and load-scaled target slips (kappa_star, alpha_star_rad)
        for AL-QP friction ellipses. Default fz_n is static TeR27 quarter load.
        """
        fz = _f32(max(fz_n, 50.0))
        mu_sc = self.mu_scale_hat

        u_fz_k = _clamp_f32((fz - _f32(300.0)) / _f32(1000.0), 0.0, 1.2)
        kappa_nom = _f32(self.kappa_base + self.kappa_fz_slope * u_fz_k)
        kappa_star = _clamp_f32(
            kappa_nom * (_f32(0.80) + _f32(0.20) * mu_sc),
            0.065,
            0.145,
        )

        u_fz_a = _clamp_f32((fz - _f32(500.0)) / _f32(800.0), 0.0, 1.0)
        alpha_nom_deg = _f32(self.alpha_base_deg + self.alpha_fz_slope_deg * u_fz_a)
        alpha_deg = _clamp_f32(
            alpha_nom_deg * (_f32(0.78) + _f32(0.22) * mu_sc),
            4.0,
            9.0,
        )
        alpha_rad = _f32(alpha_deg * _f32(math.pi / 180.0))
        return kappa_star, alpha_rad

    def step_scalar(
        self,
        ax_comp_ms2: float,
        ay_comp_ms2: float,
        fx_nom_n: float,
        fy_nom_n: float,
        dt_s: float = 0.005,
        fz_ref_n: float = 711.225,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Zero-allocation float32 step returning:
            (lambda_mu_hat, mu_scale_hat, kappa_star, alpha_star_rad, gate_active)

        Accumulates inputs over `subsample_div` ticks (default 2 -> 100 Hz at
        200 Hz controller rate) and executes one scalar RLS/EKF update.
        """
        self._acc_ax = _f32(self._acc_ax + _f32(ax_comp_ms2))
        self._acc_ay = _f32(self._acc_ay + _f32(ay_comp_ms2))
        self._acc_fx_nom = _f32(self._acc_fx_nom + _f32(fx_nom_n))
        self._acc_fy_nom = _f32(self._acc_fy_nom + _f32(fy_nom_n))
        self._acc_dt = _f32(self._acc_dt + _f32(max(dt_s, 1e-4)))
        self._sub_count += 1

        if self._sub_count >= self.subsample_div:
            inv_n = _f32(1.0 / float(self._sub_count))
            ax_bar = _f32(self._acc_ax * inv_n)
            ay_bar = _f32(self._acc_ay * inv_n)
            fx_bar = _f32(self._acc_fx_nom * inv_n)
            fy_bar = _f32(self._acc_fy_nom * inv_n)
            dt_upd = self._acc_dt

            self._sub_count = 0
            self._acc_ax = _f32(0.0)
            self._acc_ay = _f32(0.0)
            self._acc_fx_nom = _f32(0.0)
            self._acc_fy_nom = _f32(0.0)
            self._acc_dt = _f32(0.0)

            inv_m = _f32(1.0 / self.mass_kg)
            ax_nom = _f32(fx_bar * inv_m)
            ay_nom = _f32(fy_bar * inv_m)

            a_tot = _f32(math.sqrt(_f32(ax_bar * ax_bar + ay_bar * ay_bar)))
            a_nom = _f32(math.sqrt(_f32(ax_nom * ax_nom + ay_nom * ay_nom)))

            if a_tot > self.accel_gate_ms2 and a_nom > self.nom_accel_gate_ms2:
                self.gate_active = True
                phi = _f32(a_nom / self.lambda_mu_nom)
                y_proj = _f32(_f32(ax_bar * ax_nom + ay_bar * ay_nom) / a_nom)

                # Reject inconsistent sign (e.g. severe transient / bump reversal)
                if y_proj > _f32(0.5 * self.accel_gate_ms2):
                    innov_raw = _f32(y_proj - _f32(self.lambda_mu_hat * phi))
                    innov = _clamp_f32(innov_raw, -self.innov_clamp_ms2, self.innov_clamp_ms2)
                    self.last_innov_ms2 = innov

                    p_pred = _clamp_f32(
                        _f32(self.p_cov / self.rls_forgetting) + _f32(self.q_proc * dt_upd),
                        self.p_min,
                        self.p_max,
                    )
                    s_var = _f32(_f32(_f32(phi * phi) * p_pred) + self.r_meas)
                    k_gain = _f32(_f32(p_pred * phi) / max(s_var, _f32(1e-6)))

                    max_step = _f32(self.rate_limit_per_s * dt_upd)
                    d_lam = _clamp_f32(_f32(k_gain * innov), -max_step, max_step)

                    self.lambda_mu_hat = _clamp_f32(
                        _f32(self.lambda_mu_hat + d_lam),
                        self.lambda_mu_min,
                        self.lambda_mu_max,
                    )
                    self.mu_scale_hat = _f32(self.lambda_mu_hat / self.lambda_mu_nom)
                    self.p_cov = _clamp_f32(
                        _f32(_f32(1.0) - _f32(k_gain * phi)) * p_pred,
                        self.p_min,
                        self.p_max,
                    )
                    self.update_count += 1
                else:
                    self.gate_active = False
                    self.last_innov_ms2 = _f32(0.0)
            else:
                self.gate_active = False
                self.last_innov_ms2 = _f32(0.0)

        kappa_star, alpha_star_rad = self.compute_scaled_target_slips(fz_ref_n)
        return self.lambda_mu_hat, self.mu_scale_hat, kappa_star, alpha_star_rad, self.gate_active

    def step(
        self,
        ax_comp_ms2: float,
        ay_comp_ms2: float,
        fx_nom_n: float,
        fy_nom_n: float,
        dt_s: float = 0.005,
        fz_ref_n: float = 711.225,
    ) -> Dict[str, float]:
        """Dictionary-returning convenience wrapper around step_scalar()."""
        lam, scale, kap, alp, active = self.step_scalar(
            ax_comp_ms2=ax_comp_ms2,
            ay_comp_ms2=ay_comp_ms2,
            fx_nom_n=fx_nom_n,
            fy_nom_n=fy_nom_n,
            dt_s=dt_s,
            fz_ref_n=fz_ref_n,
        )
        return {
            "lambda_mu_hat": lam,
            "mu_scale_hat": scale,
            "kappa_star": kap,
            "alpha_star_rad": alp,
            "alpha_star_deg": _f32(alp * _f32(180.0 / math.pi)),
            "gate_active": active,
            "innovation_ms2": self.last_innov_ms2,
            "p_cov": self.p_cov,
        }


class TireRelaxationCompensator:
    """
    Deterministic float32 1st-Order Speed-Dependent Tire Relaxation Length
    Precompensator for Hoosier 16.0x7.5-10 (sigma_alpha=0.24 m, sigma_kappa=0.14 m).

    Inverts low-speed carcass phase lag tau_rel(vx) = sigma / max(|vx|, v_min)
    using a proper noise-bounded lead-lag filter:
        C_lead(s) = 1 + w(vx) * max(gamma * tau_rel - tau_f, 0) * s / (1 + tau_f * s)

    Features:
      - Smooth cubic fade-out in [8.0, 12.0] m/s (0.0 above 12.0 m/s).
      - Bounded high-frequency gain <= max_hf_gain (default 2.5x) so sensor noise
        is never amplified.
      - Acts strictly on feedforward yaw moment (hairpin exits) and launch
        traction force / slip target (first 15 m of Acceleration), preserving
        dMz/dbeta < 0 directional stability.
      - Zero heap allocations in step_scalar().
    """

    __slots__ = (
        "sigma_alpha_m",
        "sigma_kappa_m",
        "v_min_ms",
        "v_fade_start_ms",
        "v_fade_end_ms",
        "dist_fade_start_m",
        "dist_fade_end_m",
        "lead_gamma_alpha",
        "lead_gamma_kappa",
        "tau_f_alpha_min_s",
        "tau_f_kappa_min_s",
        "max_hf_gain",
        "mz_lead_clamp_nm",
        "fx_lead_clamp_n",
        "kappa_lead_clamp",
        "_mz_lpf",
        "_fx_lpf",
        "_kappa_lpf",
        "_initialized",
        "last_tau_alpha_s",
        "last_tau_kappa_s",
        "last_speed_weight",
        "last_dist_weight",
        "last_mz_lead_nm",
        "last_fx_lead_n",
        "last_kappa_lead",
    )

    def __init__(
        self,
        sigma_alpha_m: float = HOOSIER_SIGMA_ALPHA_M,
        sigma_kappa_m: float = HOOSIER_SIGMA_KAPPA_M,
        v_min_ms: float = RELAX_VX_MIN_MS,
        v_fade_start_ms: float = RELAX_FADE_START_MS,
        v_fade_end_ms: float = RELAX_FADE_END_MS,
        dist_fade_start_m: float = LAUNCH_DIST_FADE_START_M,
        dist_fade_end_m: float = LAUNCH_DIST_FADE_END_M,
        lead_gamma_alpha: float = 0.90,
        lead_gamma_kappa: float = 0.90,
        tau_f_alpha_min_s: float = 0.020,
        tau_f_kappa_min_s: float = 0.015,
        max_hf_gain: float = 2.5,
        mz_lead_clamp_nm: float = 120.0,
        fx_lead_clamp_n: float = 350.0,
        kappa_lead_clamp: float = 0.025,
    ) -> None:
        self.sigma_alpha_m: float = _f32(max(sigma_alpha_m, 0.0))
        self.sigma_kappa_m: float = _f32(max(sigma_kappa_m, 0.0))
        self.v_min_ms: float = _f32(max(v_min_ms, 0.5))
        self.v_fade_start_ms: float = _f32(v_fade_start_ms)
        self.v_fade_end_ms: float = _f32(max(v_fade_end_ms, v_fade_start_ms + 0.5))
        self.dist_fade_start_m: float = _f32(dist_fade_start_m)
        self.dist_fade_end_m: float = _f32(max(dist_fade_end_m, dist_fade_start_m + 0.5))
        self.lead_gamma_alpha: float = _clamp_f32(lead_gamma_alpha, 0.0, 1.0)
        self.lead_gamma_kappa: float = _clamp_f32(lead_gamma_kappa, 0.0, 1.0)
        self.tau_f_alpha_min_s: float = _f32(max(tau_f_alpha_min_s, 0.005))
        self.tau_f_kappa_min_s: float = _f32(max(tau_f_kappa_min_s, 0.005))
        self.max_hf_gain: float = _f32(max(max_hf_gain, 1.05))
        self.mz_lead_clamp_nm: float = _f32(max(mz_lead_clamp_nm, 0.0))
        self.fx_lead_clamp_n: float = _f32(max(fx_lead_clamp_n, 0.0))
        self.kappa_lead_clamp: float = _f32(max(kappa_lead_clamp, 0.0))

        self._mz_lpf: float = _f32(0.0)
        self._fx_lpf: float = _f32(0.0)
        self._kappa_lpf: float = _f32(0.0)
        self._initialized: bool = False
        self.last_tau_alpha_s: float = _f32(0.0)
        self.last_tau_kappa_s: float = _f32(0.0)
        self.last_speed_weight: float = _f32(1.0)
        self.last_dist_weight: float = _f32(1.0)
        self.last_mz_lead_nm: float = _f32(0.0)
        self.last_fx_lead_n: float = _f32(0.0)
        self.last_kappa_lead: float = _f32(0.0)

    def reset(self) -> None:
        """Reset filter states with zero heap allocation."""
        self._mz_lpf = _f32(0.0)
        self._fx_lpf = _f32(0.0)
        self._kappa_lpf = _f32(0.0)
        self._initialized = False
        self.last_tau_alpha_s = _f32(0.0)
        self.last_tau_kappa_s = _f32(0.0)
        self.last_speed_weight = _f32(1.0)
        self.last_dist_weight = _f32(1.0)
        self.last_mz_lead_nm = _f32(0.0)
        self.last_fx_lead_n = _f32(0.0)
        self.last_kappa_lead = _f32(0.0)

    def compute_relaxation_times(self, vx_ms: float) -> Tuple[float, float, float]:
        """Return (tau_alpha_s, tau_kappa_s, speed_weight) at speed vx_ms."""
        v_abs = _f32(abs(vx_ms))
        v_eff = _f32(max(v_abs, self.v_min_ms))
        tau_a = _f32(self.sigma_alpha_m / v_eff)
        tau_k = _f32(self.sigma_kappa_m / v_eff)
        w_spd = _smoothstep_falloff_f32(v_abs, self.v_fade_start_ms, self.v_fade_end_ms)
        return tau_a, tau_k, w_spd

    def step_scalar(
        self,
        vx_ms: float,
        mz_ff_nm: float,
        fx_launch_n: float = 0.0,
        kappa_target: float = 0.0,
        dt_s: float = 0.005,
        dist_m: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Zero-allocation float32 step returning:
            (mz_ff_comp_nm, fx_launch_comp_n, kappa_target_comp)
        """
        dt = _f32(max(dt_s, 1e-4))
        mz_in = _f32(mz_ff_nm)
        fx_in = _f32(fx_launch_n)
        kap_in = _f32(kappa_target)

        tau_a, tau_k, w_spd = self.compute_relaxation_times(vx_ms)
        w_dist = (
            _f32(1.0)
            if dist_m is None
            else _smoothstep_falloff_f32(_f32(max(dist_m, 0.0)), self.dist_fade_start_m, self.dist_fade_end_m)
        )
        w_launch = _f32(w_spd * w_dist)

        self.last_tau_alpha_s = tau_a
        self.last_tau_kappa_s = tau_k
        self.last_speed_weight = w_spd
        self.last_dist_weight = w_dist

        if not self._initialized:
            self._mz_lpf = mz_in
            self._fx_lpf = fx_in
            self._kappa_lpf = kap_in
            self._initialized = True
            self.last_mz_lead_nm = _f32(0.0)
            self.last_fx_lead_n = _f32(0.0)
            self.last_kappa_lead = _f32(0.0)
            return mz_in, fx_in, kap_in

        # Lateral / yaw feedforward lead-lag channel (hairpin exits, 0-12 m/s)
        tau_lead_a = _f32(self.lead_gamma_alpha * tau_a)
        tau_f_a = _f32(max(self.tau_f_alpha_min_s, _f32(tau_lead_a / self.max_hf_gain)))
        inv_denom_a = _f32(1.0 / _f32(tau_f_a + dt))
        mz_dot_f = _f32(_f32(mz_in - self._mz_lpf) * inv_denom_a)
        self._mz_lpf = _f32(self._mz_lpf + _f32(dt * mz_dot_f))
        gain_a = _f32(max(_f32(tau_lead_a - tau_f_a), _f32(0.0)))
        mz_lead = _clamp_f32(
            _f32(_f32(w_spd * gain_a) * mz_dot_f),
            -self.mz_lead_clamp_nm,
            self.mz_lead_clamp_nm,
        )
        mz_out = _f32(mz_in + mz_lead)

        # Longitudinal launch traction & slip target lead-lag channel (0-15 m, 0-12 m/s)
        tau_lead_k = _f32(self.lead_gamma_kappa * tau_k)
        tau_f_k = _f32(max(self.tau_f_kappa_min_s, _f32(tau_lead_k / self.max_hf_gain)))
        inv_denom_k = _f32(1.0 / _f32(tau_f_k + dt))

        fx_dot_f = _f32(_f32(fx_in - self._fx_lpf) * inv_denom_k)
        self._fx_lpf = _f32(self._fx_lpf + _f32(dt * fx_dot_f))
        gain_k = _f32(max(_f32(tau_lead_k - tau_f_k), _f32(0.0)))
        fx_lead = _clamp_f32(
            _f32(_f32(w_launch * gain_k) * fx_dot_f),
            -self.fx_lead_clamp_n,
            self.fx_lead_clamp_n,
        )
        fx_out = _f32(fx_in + fx_lead)

        kap_dot_f = _f32(_f32(kap_in - self._kappa_lpf) * inv_denom_k)
        self._kappa_lpf = _f32(self._kappa_lpf + _f32(dt * kap_dot_f))
        kap_lead = _clamp_f32(
            _f32(_f32(w_launch * gain_k) * kap_dot_f),
            -self.kappa_lead_clamp,
            self.kappa_lead_clamp,
        )
        kap_out = _f32(kap_in + kap_lead)

        self.last_mz_lead_nm = mz_lead
        self.last_fx_lead_n = fx_lead
        self.last_kappa_lead = kap_lead
        return mz_out, fx_out, kap_out

    def step(
        self,
        vx_ms: float,
        mz_ff_nm: float,
        fx_launch_n: float = 0.0,
        kappa_target: float = 0.0,
        dt_s: float = 0.005,
        dist_m: Optional[float] = None,
    ) -> Dict[str, float]:
        """Dictionary-returning convenience wrapper around step_scalar()."""
        mz_c, fx_c, kap_c = self.step_scalar(
            vx_ms=vx_ms,
            mz_ff_nm=mz_ff_nm,
            fx_launch_n=fx_launch_n,
            kappa_target=kappa_target,
            dt_s=dt_s,
            dist_m=dist_m,
        )
        return {
            "mz_ff_comp_nm": mz_c,
            "fx_launch_comp_n": fx_c,
            "kappa_target_comp": kap_c,
            "mz_lead_nm": self.last_mz_lead_nm,
            "fx_lead_n": self.last_fx_lead_n,
            "kappa_lead": self.last_kappa_lead,
            "tau_alpha_s": self.last_tau_alpha_s,
            "tau_kappa_s": self.last_tau_kappa_s,
            "speed_weight": self.last_speed_weight,
            "dist_weight": self.last_dist_weight,
        }
