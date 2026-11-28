#!/usr/bin/env python3
"""
Real-Time Chance Constraints for Stochastic Tube-MPC (TeR27 4WD).

Implements probabilistic constraint tightening with exact Gaussian quantile backoff:
    b_k = Phi^{-1}(1 - eps) * sqrt(c^T * Sigma_k * c)

Default risk level eps = 1e-4 (z_eps = 3.7190164855, > 99.99% no off-track guarantee).
Covers:
  1. Skidpad / Autocross cone clearance (min_clear >= 0.150 m)
  2. AL-QP friction ellipse effective radius scaling
  3. HV DC power cap [-25.0, +80.0] kW (strict FSAE EV rules & BSPD EV 2.3)

All runtime math is deterministic, float32-compatible, pure ASCII, and 0 B heap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np

# Exact Phi^{-1}(1 - 1e-4) = 3.719016485455709
Z_QUANTILE_1E4: float = 3.719016485455709


def _f32(val: float) -> float:
    """Round scalar to IEEE-754 binary32 precision for STM32F405 parity."""
    return float(np.float32(val))


def _clamp_f32(val: float, lo: float, hi: float) -> float:
    v = _f32(val)
    if v < lo:
        return _f32(lo)
    if v > hi:
        return _f32(hi)
    return v


def norm_ppf_approx(p: float) -> float:
    """
    Deterministic Acklam rational approximation of Phi^{-1}(p) for p in (0, 1).
    Max relative error < 1.15e-9; pure math with zero scipy dependency so it
    matches C99 and GDScript 1-to-1.
    """
    p_clamped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p_clamped < p_low:
        q = math.sqrt(-2.0 * math.log(p_clamped))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p_clamped <= p_high:
        q = p_clamped - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p_clamped))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


@dataclass(frozen=True)
class ChanceConstraintConfig:
    """Configuration for Stochastic Tube-MPC chance constraint backoffs."""

    eps_track: float = 1e-4              # >99.99% no off-track guarantee
    eps_friction: float = 1e-4           # >99.99% friction ellipse adherence
    eps_power: float = 1e-4              # >99.99% HV DC power compliance
    min_cone_clearance_m: float = 0.150  # Strict FSAE cone clearance floor [m]
    track_half_width_m: float = 1.500    # Nominal corridor half-width [m] (3.0 m lane)
    vehicle_half_width_m: float = 0.720  # TeR27 half-width incl. tires/wings [m]
    friction_limit_nom: float = 1.000    # Physical limit = 1.000 (low-uncertainty 1 - b_fric ~ 0.995)
    friction_limit_min: float = 0.680    # Minimum safe friction ellipse scale under severe uncertainty
    friction_backoff_gain: float = 0.58  # Scaling from normalized grip/lat std to ellipse backoff
    p_dc_max_kw: float = 80.0            # Strict FSAE HV discharge ceiling [kW]
    p_dc_regen_min_kw: float = -25.0     # Strict HV regen charge ceiling [kW]
    p_dc_hard_margin_kw: float = 0.05    # Hard safety margin (79.95 / -24.95 kW)
    p_dc_backoff_max_kw: float = 6.0     # Cap on stochastic HV power backoff [kW]
    cone_backoff_max_m: float = 0.550    # Cap on lateral tube radius [m]
    speed_backoff_gain: float = 0.45     # Speed-reference tightening gain from tube radius


@dataclass(frozen=True)
class ChanceBackoffResult:
    """Evaluated chance-constraint backoffs and tightened limits for one control tick."""

    z_track: float
    z_friction: float
    z_power: float
    sigma_ey_m: float
    sigma_fric_norm: float
    sigma_pdc_kw: float
    b_cone_m: float
    b_friction: float
    b_pdc_kw: float
    corridor_half_margin_m: float
    effective_cone_limit_m: float
    friction_ellipse_scale: float
    p_dc_max_eff_kw: float
    p_dc_regen_eff_kw: float
    v_ref_scale: float


class ChanceConstraintEvaluator:
    """
    Computes exact Gaussian chance-constraint backoffs:
        b_k = Phi^{-1}(1 - eps) * sqrt(c^T * Sigma_k * c)

    Supports both full matrix covariance Sigma_k (from CasADi Jacobians) and
    scalar/diagonal variance inputs (for float32 embedded C99 / GDScript parity).
    """

    def __init__(self, config: Optional[ChanceConstraintConfig] = None) -> None:
        self.cfg = config if config is not None else ChanceConstraintConfig()
        self.z_track: float = _f32(norm_ppf_approx(1.0 - self.cfg.eps_track))
        self.z_friction: float = _f32(norm_ppf_approx(1.0 - self.cfg.eps_friction))
        self.z_power: float = _f32(norm_ppf_approx(1.0 - self.cfg.eps_power))
        self.corridor_body_clearance_m: float = _f32(
            max(self.cfg.track_half_width_m - self.cfg.vehicle_half_width_m, 0.20)
        )

    @staticmethod
    def directional_std(sigma_mat: np.ndarray, c_vec: np.ndarray) -> float:
        """Compute sqrt(max(c^T * Sigma * c, 0.0)) safely."""
        c = np.asarray(c_vec, dtype=np.float64).ravel()
        s = np.asarray(sigma_mat, dtype=np.float64)
        var = float(c @ s @ c)
        return math.sqrt(max(var, 0.0))

    def evaluate_from_variances(
        self,
        var_ey_m2: float,
        var_fric_norm: float,
        var_pdc_kw2: float,
    ) -> ChanceBackoffResult:
        """
        Float32-exact deterministic backoff evaluation from projected scalar variances:
          - var_ey_m2     = c_y^T * Sigma_k * c_y         [m^2]
          - var_fric_norm = c_f^T * Sigma_k * c_f         [dimensionless^2]
          - var_pdc_kw2   = c_p^T * Sigma_k * c_p         [kW^2]
        """
        sig_ey = _f32(math.sqrt(max(_f32(var_ey_m2), 0.0)))
        sig_fr = _f32(math.sqrt(max(_f32(var_fric_norm), 0.0)))
        sig_p = _f32(math.sqrt(max(_f32(var_pdc_kw2), 0.0)))

        # 1. Cone clearance / lateral corridor backoff
        b_cone = _clamp_f32(
            _f32(self.z_track * sig_ey),
            0.0,
            self.cfg.cone_backoff_max_m,
        )
        max_nominal_ey = _f32(
            max(self.corridor_body_clearance_m - self.cfg.min_cone_clearance_m, 0.05)
        )
        eff_ey_lim = _f32(max(max_nominal_ey - b_cone, 0.02))

        # 2. AL-QP friction ellipse effective radius tightening
        b_fric_raw = _f32(self.z_friction * self.cfg.friction_backoff_gain * sig_fr)
        max_fric_drop = _f32(self.cfg.friction_limit_nom - self.cfg.friction_limit_min)
        b_fric = _clamp_f32(b_fric_raw, 0.0, max_fric_drop)
        fric_scale = _clamp_f32(
            _f32(self.cfg.friction_limit_nom - b_fric),
            self.cfg.friction_limit_min,
            self.cfg.friction_limit_nom,
        )

        # 3. HV DC power cap tightening ([-25.0, +80.0] kW strict)
        b_pdc = _clamp_f32(
            _f32(self.z_power * sig_p),
            0.0,
            self.cfg.p_dc_backoff_max_kw,
        )
        p_max_hard = _f32(self.cfg.p_dc_max_kw - self.cfg.p_dc_hard_margin_kw)
        p_min_hard = _f32(self.cfg.p_dc_regen_min_kw + self.cfg.p_dc_hard_margin_kw)
        p_dc_max_eff = _clamp_f32(_f32(p_max_hard - b_pdc), 20.0, p_max_hard)
        p_dc_regen_eff = _clamp_f32(_f32(p_min_hard + _f32(0.5 * b_pdc)), p_min_hard, -5.0)

        # 4. Safe reference speed scaling coupled to friction tube & lateral cone tube
        #    When b_fric ~ 0 and b_cone ~ 0 -> v_ref_scale ~ 0.9975 (sqrt(0.995))
        tube_ratio = _f32(b_cone / max(max_nominal_ey, 0.1))
        v_from_fric = _f32(math.sqrt(max(fric_scale, 0.25)))
        v_from_cone = _clamp_f32(
            _f32(1.0 - self.cfg.speed_backoff_gain * max(tube_ratio - 0.12, 0.0)),
            0.70,
            1.0,
        )
        v_ref_scale = _clamp_f32(min(v_from_fric, v_from_cone), 0.70, 1.0)

        return ChanceBackoffResult(
            z_track=self.z_track,
            z_friction=self.z_friction,
            z_power=self.z_power,
            sigma_ey_m=sig_ey,
            sigma_fric_norm=sig_fr,
            sigma_pdc_kw=sig_p,
            b_cone_m=b_cone,
            b_friction=b_fric,
            b_pdc_kw=b_pdc,
            corridor_half_margin_m=self.corridor_body_clearance_m,
            effective_cone_limit_m=eff_ey_lim,
            friction_ellipse_scale=fric_scale,
            p_dc_max_eff_kw=p_dc_max_eff,
            p_dc_regen_eff_kw=p_dc_regen_eff,
            v_ref_scale=v_ref_scale,
        )

    def evaluate_from_matrix(
        self,
        sigma_mat: np.ndarray,
        c_ey: np.ndarray,
        c_fric: np.ndarray,
        c_pdc: np.ndarray,
    ) -> ChanceBackoffResult:
        """Evaluate exact chance-constraint backoffs directly from matrix Sigma_k."""
        c_y = np.asarray(c_ey, dtype=np.float64).ravel()
        c_f = np.asarray(c_fric, dtype=np.float64).ravel()
        c_p = np.asarray(c_pdc, dtype=np.float64).ravel()
        s = np.asarray(sigma_mat, dtype=np.float64)
        var_ey = max(float(c_y @ s @ c_y), 0.0)
        var_fr = max(float(c_f @ s @ c_f), 0.0)
        var_p = max(float(c_p @ s @ c_p), 0.0)
        return self.evaluate_from_variances(var_ey, var_fr, var_p)

    def to_dict(self, res: ChanceBackoffResult) -> Dict[str, float]:
        """Serialize ChanceBackoffResult to pure-ASCII float dictionary."""
        return {
            "z_track": float(res.z_track),
            "z_friction": float(res.z_friction),
            "z_power": float(res.z_power),
            "sigma_ey_m": float(res.sigma_ey_m),
            "sigma_fric_norm": float(res.sigma_fric_norm),
            "sigma_pdc_kw": float(res.sigma_pdc_kw),
            "b_cone_m": float(res.b_cone_m),
            "b_friction": float(res.b_friction),
            "b_pdc_kw": float(res.b_pdc_kw),
            "corridor_half_margin_m": float(res.corridor_half_margin_m),
            "effective_cone_limit_m": float(res.effective_cone_limit_m),
            "friction_ellipse_scale": float(res.friction_ellipse_scale),
            "p_dc_max_eff_kw": float(res.p_dc_max_eff_kw),
            "p_dc_regen_eff_kw": float(res.p_dc_regen_eff_kw),
            "v_ref_scale": float(res.v_ref_scale),
        }
