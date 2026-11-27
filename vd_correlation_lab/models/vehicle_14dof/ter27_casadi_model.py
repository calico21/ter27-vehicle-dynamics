#!/usr/bin/env python3
"""
Smooth C^2 symbolic CasADi 3.7.2 model of Tecnun e-Racing TeR27 4WD (MirenaSim).

Matches vd_correlation_lab/oracle/ter27_physics.py for both FSG_REAL_TRACK and
TTC_LAB_PEAK grip profiles with < 0.1% error while replacing all non-differentiable
operations (clip, abs, max, min, piecewise if/else) with smooth C^2 expressions
suitable for exact IPOPT Jacobians and Hessians.

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
Wheel order: [0: FL, 1: FR, 2: RL, 3: RR].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import numpy as np

try:
    from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
        GRIP_PROFILES,
        PacejkaMF62,
        TeR27Params,
        TeR27PhysicsModel,
        load_ter27_params,
    )
except ImportError:
    from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
        GRIP_PROFILES,
        PacejkaMF62,
        TeR27Params,
        TeR27PhysicsModel,
        load_ter27_params,
    )


# Extended grip profiles (preserving exact GRIP_PROFILES for FSG_REAL_TRACK & TTC_LAB_PEAK)
CASADI_GRIP_PROFILES: Dict[str, Dict[str, float]] = {
    "TTC_LAB_PEAK": {
        "env_mu_scale": float(GRIP_PROFILES["TTC_LAB_PEAK"]["env_mu_scale"]),
        "env_aero_cl_scale": float(GRIP_PROFILES["TTC_LAB_PEAK"]["env_aero_cl_scale"]),
        "default_temp_c": 45.0,
    },
    "FSG_REAL_TRACK": {
        "env_mu_scale": float(GRIP_PROFILES["FSG_REAL_TRACK"]["env_mu_scale"]),
        "env_aero_cl_scale": float(GRIP_PROFILES["FSG_REAL_TRACK"]["env_aero_cl_scale"]),
        "default_temp_c": 40.0,
    },
    "FSG_RUBBERED": {
        "env_mu_scale": 0.88,
        "env_aero_cl_scale": 0.90,
        "default_temp_c": 45.0,
    },
    "COLD_LOW_GRIP": {
        "env_mu_scale": 0.72,
        "env_aero_cl_scale": 0.64,
        "default_temp_c": 18.0,
    },
    "CUSTOM": {
        "env_mu_scale": 0.88,
        "env_aero_cl_scale": 0.78,
        "default_temp_c": 45.0,
    },
}

DESIGN_PARAM_NAMES: Tuple[str, ...] = (
    "lltd",
    "k_roll_scale",
    "h_rc_f",
    "h_rc_r",
    "camber_f_deg",
    "camber_r_deg",
    "anti_dive",
    "anti_squat",
    "brake_bias_front",
    "gear_ratio",
    "cl_scale",
    "cd_scale",
    "cop_front_frac",
    "h_cg",
    "i_zz",
    "mass_kg",
)

# Hoosier 16.0x7.5-10 R20 lumped thermal parameters (per tire)
TIRE_THERMAL_PARAMS: Dict[str, float] = {
    "t_opt_low_c": 70.0,
    "t_opt_high_c": 85.0,
    "t_opt_center_c": 77.5,
    "c_th_j_per_k": 4200.0,     # Effective rubber + carcass heat capacity (J/K)
    "eta_slip_heat": 0.65,      # Fraction of contact slip power entering tire bulk (-)
    "h_cond_w_per_k": 18.0,     # Conductive + zero-speed cooling coefficient (W/K)
    "h_conv_w_per_k_ms": 1.6,   # Convective air cooling per m/s of speed (W/(K*(m/s)))
}


def _is_casadi(x: Any) -> bool:
    return isinstance(x, (ca.SX, ca.MX, ca.DM))


# ---------------------------------------------------------------------------
# Smooth C^2 (actually C^infinity) primitives - overflow-safe for SX and MX
# ---------------------------------------------------------------------------

def sabs(x: Any, eps: float = 1e-4) -> Any:
    """Smooth |x| = sqrt(x^2 + eps^2)."""
    if _is_casadi(x):
        return ca.sqrt(x * x + eps * eps)
    return np.sqrt(np.asarray(x, dtype=float) ** 2 + eps * eps) if isinstance(x, np.ndarray) else math.sqrt(float(x) * float(x) + eps * eps)


def smax(a: Any, b: Any, eps: float = 1e-4) -> Any:
    """Smooth max(a, b) = 0.5 * (a + b + sqrt((a - b)^2 + eps^2))."""
    d = a - b
    if _is_casadi(a) or _is_casadi(b):
        return 0.5 * (a + b + ca.sqrt(d * d + eps * eps))
    if isinstance(d, np.ndarray):
        return 0.5 * (a + b + np.sqrt(d * d + eps * eps))
    return 0.5 * (float(a) + float(b) + math.sqrt(float(d) * float(d) + eps * eps))


def smin(a: Any, b: Any, eps: float = 1e-4) -> Any:
    """Smooth min(a, b) = 0.5 * (a + b - sqrt((a - b)^2 + eps^2))."""
    d = a - b
    if _is_casadi(a) or _is_casadi(b):
        return 0.5 * (a + b - ca.sqrt(d * d + eps * eps))
    if isinstance(d, np.ndarray):
        return 0.5 * (a + b - np.sqrt(d * d + eps * eps))
    return 0.5 * (float(a) + float(b) - math.sqrt(float(d) * float(d) + eps * eps))


def spos(x: Any, eps: float = 1e-4) -> Any:
    """Smooth max(x, 0) = 0.5 * (x + sqrt(x^2 + eps^2)) > 0."""
    if _is_casadi(x):
        return 0.5 * (x + ca.sqrt(x * x + eps * eps))
    if isinstance(x, np.ndarray):
        return 0.5 * (x + np.sqrt(x * x + eps * eps))
    xf = float(x)
    return 0.5 * (xf + math.sqrt(xf * xf + eps * eps))


def sneg(x: Any, eps: float = 1e-4) -> Any:
    """Smooth min(x, 0) = 0.5 * (x - sqrt(x^2 + eps^2)) < 0."""
    if _is_casadi(x):
        return 0.5 * (x - ca.sqrt(x * x + eps * eps))
    if isinstance(x, np.ndarray):
        return 0.5 * (x - np.sqrt(x * x + eps * eps))
    xf = float(x)
    return 0.5 * (xf - math.sqrt(xf * xf + eps * eps))


def sclip(x: Any, lo: Any, hi: Any, eps: float = 1e-4) -> Any:
    """Smooth clamp to [lo, hi]."""
    return smin(smax(x, lo, eps), hi, eps)


def ssign(x: Any, eps: float = 1e-3) -> Any:
    """Smooth sign(x) = tanh(x / eps)."""
    if _is_casadi(x):
        return ca.tanh(x / eps)
    if isinstance(x, np.ndarray):
        return np.tanh(x / eps)
    return math.tanh(float(x) / eps)


# ---------------------------------------------------------------------------
# Smooth C^2 Thermal-Frictional Model for Hoosier 16.0x7.5-10 R20
# ---------------------------------------------------------------------------

def thermal_grip_multiplier_smooth(T_tire_c: Any) -> Any:
    """
    Smooth C^infinity thermal-frictional grip multiplier for Hoosier 16.0x7.5-10 R20.

    Characteristics:
      - Optimal window: 70-85 C -> ~1.000 (peak 1.000 at 77.5 C)
      - Smooth cold drop below 45 C (~0.94 at 45 C, ~0.81 at 30 C, ~0.74 at 18 C)
      - Smooth overheat degradation above 95 C (~0.97 at 95 C, ~0.89 at 105 C, ~0.79 at 115 C)

    Works seamlessly with float, numpy.ndarray, ca.SX, ca.MX, or ca.DM.
    """
    if _is_casadi(T_tire_c):
        tanh_cold = ca.tanh((T_tire_c - 34.0) / 16.0)
        tanh_hot = ca.tanh((T_tire_c - 108.0) / 12.0)
    elif isinstance(T_tire_c, np.ndarray):
        tanh_cold = np.tanh((T_tire_c - 34.0) / 16.0)
        tanh_hot = np.tanh((T_tire_c - 108.0) / 12.0)
    else:
        t_val = float(T_tire_c)
        tanh_cold = math.tanh((t_val - 34.0) / 16.0)
        tanh_hot = math.tanh((t_val - 108.0) / 12.0)

    cold_loss = 0.30 * 0.5 * (1.0 - tanh_cold)
    hot_loss = 0.28 * 0.5 * (1.0 + tanh_hot)
    # Normalize so peak at T=77.5 C is exactly 1.000
    mu_raw = (1.0 - cold_loss - hot_loss) / 0.997015
    return sclip(mu_raw, 0.55, 1.005, 1e-3)


def slip_power_smooth(
    fx_w: Any,
    fy_w: Any,
    v_wx: Any,
    v_wy: Any,
    kappa: Any,
    eps_w: float = 10.0,
) -> Any:
    """
    Smooth C^2 contact-patch slip power dissipation per tire (W):
        Q_slip = |Fx * vx * kappa| + |Fy * vy_slip|
    """
    p_long = fx_w * v_wx * kappa
    p_lat = fy_w * v_wy
    return sabs(p_long, eps_w) + sabs(p_lat, eps_w)


def tire_temp_dot_smooth(
    T_tire_c: Any,
    T_track_c: Any,
    slip_power_w: Any,
    vx: Any = 15.0,
) -> Any:
    """
    Smooth C^2 tire bulk temperature rate dT_tire/dt (C/s) for Hoosier 16.0x7.5-10 R20.

    Args:
        T_tire_c: Current tire bulk temperature (C), scalar or 4-vector.
        T_track_c: Track/asphalt temperature (C).
        slip_power_w: Per-tire slip power dissipation Q_slip (W >= 0).
        vx: Vehicle/wheel longitudinal speed (m/s) for convective cooling (default 15.0 m/s).
    """
    tp = TIRE_THERMAL_PARAMS
    v_eff = sabs(vx, 1.0)
    h_cool = tp["h_cond_w_per_k"] + tp["h_conv_w_per_k_ms"] * v_eff
    q_in = tp["eta_slip_heat"] * spos(slip_power_w, 5.0)
    q_out = h_cool * (T_tire_c - T_track_c)
    return (q_in - q_out) / tp["c_th_j_per_k"]


def steady_tire_temp_smooth(
    T_track_c: Any,
    slip_power_w: Any,
    vx: Any = 15.0,
) -> Any:
    """
    Smooth C^2 steady-state tire bulk temperature T_tire_ss (C) where dT_tire/dt = 0.
    """
    tp = TIRE_THERMAL_PARAMS
    v_eff = sabs(vx, 1.0)
    h_cool = tp["h_cond_w_per_k"] + tp["h_conv_w_per_k_ms"] * v_eff
    q_in = tp["eta_slip_heat"] * spos(slip_power_w, 5.0)
    return T_track_c + q_in / h_cool


def default_design_params(params: Optional[TeR27Params] = None) -> Dict[str, float]:
    """Return exact nominal TeR27 setup/design parameter dict."""
    p = load_ter27_params() if params is None else params
    return {
        "lltd": float(p.lltd_elastic_f),
        "k_roll_scale": 1.0,
        "h_rc_f": float(p.h_rc_f),
        "h_rc_r": float(p.h_rc_r),
        "camber_f_deg": float(math.degrees(p.static_camber_f_rad)),
        "camber_r_deg": float(math.degrees(p.static_camber_r_rad)),
        "anti_dive": float(p.anti_dive_f),
        "anti_squat": float(p.anti_squat_r),
        "brake_bias_front": float(p.brake_bias_front),
        "gear_ratio": float(p.gear_ratio),
        "cl_scale": 1.0,
        "cd_scale": 1.0,
        "cop_front_frac": float(p.aero_balance_front),
        "h_cg": float(p.h_cog),
        "i_zz": float(p.izz_total),
        "mass_kg": float(p.mass),
    }


def build_symbolic_design_params(prefix: str = "dp") -> Tuple[ca.SX, Dict[str, ca.SX]]:
    """
    Build a CasADi SX vector and name->SX mapping for all 16 design/setup parameters
    so downstream IPOPT / implicit differentiation scripts can differentiate w.r.t. setup.
    """
    vec = ca.SX.sym(prefix, len(DESIGN_PARAM_NAMES))
    mapping = {name: vec[i] for i, name in enumerate(DESIGN_PARAM_NAMES)}
    return vec, mapping


# ---------------------------------------------------------------------------
# Smoothing constants (documented for downstream OCP / NLP users)
# ---------------------------------------------------------------------------

SMOOTHING_PARAMS: Dict[str, float] = {
    "eps_fz_n": 0.5,          # Normal load floor & lateral transfer clamp (N)
    "eps_dfz": 1e-3,          # Normalized load dfz upper clamp (-)
    "eps_gamma_rad": 1e-4,    # Camber |gamma| smoothing (rad)
    "eps_camber_f": 1e-3,     # Camber factor floor 0.2 (-)
    "eps_e_clip": 1e-3,       # MF 6.2 curvature factor E clip [-10, 1] (-)
    "eps_g_floor": 1e-4,      # Combined-slip weighting Gxa/Gyk floor (-)
    "eps_sign_slip": 1e-3,    # Smooth sign(alpha_y), sign(kx) (-)
    "eps_alpha_eq_sq": 1e-10, # Equivalent slip sqrt regularization (rad^2)
    "eps_speed_ms": 1e-3,     # Speed & wheel longitudinal velocity smoothing (m/s)
    "eps_ack_denom": 1e-4,    # Ackermann turning radius floor (m)
    "eps_r_eff": 1e-4,        # Radial deflection & effective radius floor (m)
    "eps_omega_rads": 1e-2,   # Motor speed |omega| smoothing (rad/s)
    "eps_torque_nm": 0.02,    # Motor torque envelope knee smoothing (Nm)
    "eps_taper": 1e-3,        # Motor speed taper [0, 1] clip (-)
    "eps_regen_nm": 0.05,     # Regen torque floor smoothing (Nm)
    "eps_power_w": 15.0,      # HV drive/regen efficiency transition (W)
    "newton_iters_t2k": 8,    # Unrolled damped Newton steps for T_4 -> kappa_4
}


# ---------------------------------------------------------------------------
# CasADi Vehicle Model
# ---------------------------------------------------------------------------

@dataclass
class TeR27CasADiVehicle:
    """
    Smooth C^2 CasADi representation of the TeR27 4WD vehicle.

    Provides both:
      - Symbolic methods (eval_*) that work directly with ca.SX or ca.MX
      - Pre-compiled casadi.Function objects (fn_*) built on SX graphs

    Also supports dict-style indexing (veh["key"]) for convenience.
    """

    grip_profile: str
    params: TeR27Params
    tire: PacejkaMF62
    env_mu_scale: Any
    env_aero_cl_scale: Any
    lmux: Any
    lmuy: Any
    x_w: np.ndarray
    y_w: np.ndarray
    hyd_split: Any
    side_sign: np.ndarray  # [+1, -1, +1, -1] for [FL, FR, RL, RR]
    smoothing: Dict[str, float]
    temp_c: Any = 45.0
    tire_temp_c: Optional[Any] = None
    design_params: Optional[Dict[str, Any]] = None

    # Compiled CasADi SX functions (None if built with free symbolic design_params)
    fn_tire_mf62: Optional[ca.Function] = None
    fn_motor_limits: Optional[ca.Function] = None
    fn_hv_power_kw: Optional[ca.Function] = None
    fn_normal_forces: Optional[ca.Function] = None
    fn_forces_from_kappa: Optional[ca.Function] = None
    fn_forces_and_power: Optional[ca.Function] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def _dp(self, key: str, default: Any) -> Any:
        if self.design_params is not None and key in self.design_params:
            return self.design_params[key]
        return default

    @property
    def mass_kg(self) -> Any:
        return self._dp("mass_kg", self.params.mass)

    @property
    def izz_total(self) -> Any:
        return self._dp("i_zz", self.params.izz_total)

    @property
    def mu_scale(self) -> Any:
        return self.env_mu_scale

    # ------------------------------------------------------------------
    # Symbolic building blocks (work with ca.SX, ca.MX, or float)
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_cos(b: Any, c: Any, e: Any, x: Any) -> Any:
        bx = b * x
        return ca.cos(c * ca.atan(bx - e * (bx - ca.atan(bx))))

    def eval_tire_mf62(
        self,
        kappa: Any,
        alpha: Any,
        fz: Any,
        gamma: Any,
        side: Any,
        p_kpa: Optional[float] = None,
        mu_thermal: Any = 1.0,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Smooth C^2 Pacejka MF 6.2 combined-slip tire model (single wheel or 4-vector).
        side: +1.0 for left wheels (FL, RL), -1.0 for right wheels (FR, RR).
        mu_thermal: optional thermal grip multiplier (scalar or 4-vector, default 1.0).
        Returns (Fx, Fy, Mz, mu_y_eff).
        """
        t = self.tire
        s = self.smoothing
        p_infl = self.params.p_infl_kpa if p_kpa is None else float(p_kpa)
        lmux_eff = self.lmux * mu_thermal
        lmuy_eff = self.lmuy * mu_thermal

        fz_pos = spos(fz, s["eps_fz_n"])
        dfz_raw = (fz_pos - t.fnomin) / t.fnomin
        if math.isfinite(t.dfz_max):
            dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"])
        else:
            dfz = dfz_raw
        dpi = (p_infl - t.pinom) / t.pinom
        abs_gamma = sabs(gamma, s["eps_gamma_rad"])
        gamma_sq = gamma * gamma

        # --- Pure lateral Fy0 ---
        cy = t.PCY1
        camber_f_y = smax(1.0 - t.PDY3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_y = (
            (t.PDY1 + t.PDY2 * dfz)
            * (1.0 + t.PPY3 * dpi + t.PPY4 * dpi * dpi)
            * camber_f_y
            * lmuy_eff
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

        # --- Pure longitudinal Fx0 ---
        cx = t.PCX1
        camber_f_x = smax(1.0 - t.PDX3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_x = (
            (t.PDX1 + t.PDX2 * dfz)
            * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
            * camber_f_x
            * lmux_eff
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

        # --- Combined slip weighting ---
        bxa = (t.RBX1 + t.RBX3 * gamma_sq) * ca.cos(ca.atan(t.RBX2 * kappa))
        exa = sclip(t.REX1 + t.REX2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        gxa_den = smax(self._shape_cos(bxa, t.RCX1, exa, t.RHX1), 1e-4, 1e-5)
        gxa = smax(self._shape_cos(bxa, t.RCX1, exa, alpha + t.RHX1) / gxa_den, 0.0, s["eps_g_floor"])

        byk = (t.RBY1 + t.RBY4 * gamma_sq) * ca.cos(ca.atan(t.RBY2 * (alpha - t.RBY3)))
        eyk = sclip(t.REY1 + t.REY2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        shyk = t.RHY1 + t.RHY2 * dfz
        gyk_den = smax(self._shape_cos(byk, t.RCY1, eyk, shyk), 1e-4, 1e-5)
        gyk = smax(self._shape_cos(byk, t.RCY1, eyk, kappa + shyk) / gyk_den, 0.0, s["eps_g_floor"])

        dvyk = (
            mu_y
            * fz_pos
            * (t.RVY1 + t.RVY2 * dfz + t.RVY3 * gamma)
            * ca.cos(ca.atan(t.RVY4 * alpha))
        )
        svyk = dvyk * ca.sin(t.RVY5 * ca.atan(t.RVY6 * kappa))

        fx = gxa * fx0
        fy = gyk * fy0 + svyk

        # --- Self-aligning torque Mz ---
        shf = shy + svy / (kya + 1e-3)
        sht = t.QHZ1 + t.QHZ2 * dfz + (t.QHZ3 + t.QHZ4 * dfz) * gamma
        alpha_t = alpha + sht
        alpha_r = alpha + shf
        kx_eq = (kxk / (kya + 1e-3)) * kappa
        # Note: cos(c*atan(odd(x))) is even in x, so using positive sqrt magnitude is exact
        # and avoids any discontinuity at alpha_t = 0, kappa != 0. For QEZ4/QEZ5 we apply ssign.
        alpha_t_mag = ca.sqrt(alpha_t * alpha_t + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_r_mag = ca.sqrt(alpha_r * alpha_r + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_t_signed = ssign(alpha_t, s["eps_sign_slip"]) * alpha_t_mag

        dt = (
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
        trail = dt * ca.cos(ct * ca.atan(t_arg - et * (t_arg - ca.atan(t_arg)))) * ca.cos(alpha)
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

        return fx, fy, mz, mu_y

    def ackermann_steer_smooth(self, delta: Any) -> Any:
        """
        Exact C^infinity Ackermann wheel steer angles [FL, FR, RL, RR] (rad).
        Algebraically equivalent to ter27_physics.ackermann_steer without any branch at delta=0.
        """
        p = self.params
        ack = float(np.clip(p.ackermann_factor, -1.0, 1.0))
        if abs(ack) < 1e-9:
            return ca.vertcat(delta, delta, 0.0, 0.0)
        L = p.wheelbase
        half_t = 0.5 * p.track_front
        tan_d = ca.tan(delta)
        abs_tan_d = sabs(tan_d, 1e-6)
        min_den = 0.25 * L * abs_tan_d
        eps = self.smoothing["eps_ack_denom"]
        den_fl = smax(L - half_t * tan_d, min_den, eps)
        den_fr = smax(L + half_t * tan_d, min_den, eps)
        d_fl_ack = ca.atan2(L * tan_d, den_fl)
        d_fr_ack = ca.atan2(L * tan_d, den_fr)
        d_fl = delta + ack * (d_fl_ack - delta)
        d_fr = delta + ack * (d_fr_ack - delta)
        return ca.vertcat(d_fl, d_fr, 0.0, 0.0)

    def aero_forces_smooth(self, vx: Any, vy: Any, ax: Any = 0.0) -> Tuple[Any, Any, Any, Any]:
        """Smooth C^2 aerodynamic forces (F_drag_x, F_drag_y, F_down_f, F_down_r) in ISO 8855."""
        p = self.params
        eps_v = self.smoothing["eps_speed_ms"]
        cl_scale = self._dp("cl_scale", 1.0)
        cd_scale = self._dp("cd_scale", 1.0)
        cop_f = self._dp("cop_front_frac", p.aero_balance_front)
        anti_dive = self._dp("anti_dive", p.anti_dive_f)
        anti_squat = self._dp("anti_squat", p.anti_squat_r)

        v_sq = vx * vx + vy * vy
        speed = ca.sqrt(v_sq + eps_v * eps_v)
        q = 0.5 * p.air_density * v_sq
        f_drag_x = -0.5 * p.air_density * (p.cd_a * cd_scale) * speed * vx
        f_drag_y = -0.5 * p.air_density * (p.cd_a * cd_scale) * speed * vy
        f_down_tot = q * (p.cl_a * cl_scale) * self.env_aero_cl_scale

        # Pitch sensitivity from anti-dive / anti-squat deviation relative to nominal (0 at nominal!)
        d_ad = anti_dive - p.anti_dive_f
        d_as = anti_squat - p.anti_squat_r
        ax_brk_g = -sneg(ax, 0.25) / p.gravity
        ax_acc_g = spos(ax, 0.25) / p.gravity
        f_down_f = f_down_tot * cop_f * (1.0 + 0.045 * d_ad * ax_brk_g - 0.015 * d_ad * d_ad)
        f_down_r = f_down_tot * (1.0 - cop_f) * (1.0 + 0.035 * d_as * ax_acc_g - 0.012 * d_as * d_as)
        return f_drag_x, f_drag_y, f_down_f, f_down_r

    def normal_forces_smooth(self, ax: Any, ay: Any, vx: Any, vy: Any) -> Tuple[Any, Any]:
        """
        Smooth C^2 4-wheel normal forces Fz [FL, FR, RL, RR] (N) and roll angle phi (rad)
        given algebraic accelerations (ax, ay) and velocities (vx, vy).
        """
        p = self.params
        eps_fz = self.smoothing["eps_fz_n"]
        _, _, f_down_f, f_down_r = self.aero_forces_smooth(vx, vy, ax)

        mass = self._dp("mass_kg", p.mass)
        h_cg = self._dp("h_cg", p.h_cog)
        h_rc_f = self._dp("h_rc_f", p.h_rc_f)
        h_rc_r = self._dp("h_rc_r", p.h_rc_r)
        lltd = self._dp("lltd", p.lltd_elastic_f)
        k_roll_scale = self._dp("k_roll_scale", 1.0)

        mg = mass * p.gravity
        fz_f_stat = mg * (p.b_cog / p.wheelbase)
        fz_r_stat = mg * (p.a_cog / p.wheelbase)

        dfz_long = (mass * ax * h_cg) / p.wheelbase
        fz_f_tot = spos(fz_f_stat - dfz_long + f_down_f, eps_fz)
        fz_r_tot = spos(fz_r_stat + dfz_long + f_down_r, eps_fz)

        m_s = mass - p.unsprung_mass_total
        m_u_axle = 0.5 * p.unsprung_mass_total
        k_roll_tot = (p.k_roll_f + p.k_roll_r) * k_roll_scale
        k_roll_f = lltd * k_roll_tot
        k_roll_r = (1.0 - lltd) * k_roll_tot

        h_s = p.h_sprung_cog + (h_cg - p.h_cog)
        h_rc_cg = h_rc_f + (h_rc_r - h_rc_f) * (p.a_cog / p.wheelbase)
        h_roll_arm = smax(h_s - h_rc_cg, 0.0, 1e-5)

        roll_denom = smax(k_roll_tot - m_s * p.gravity * h_roll_arm, 1.0, 1e-3)
        phi = (m_s * ay * h_roll_arm) / roll_denom

        dfz_lat_f_raw = (
            (k_roll_f * phi) / p.track_front
            + (m_s * (p.b_cog / p.wheelbase) * ay * h_rc_f) / p.track_front
            + (m_u_axle * ay * p.wheel_radius) / p.track_front
        )
        dfz_lat_r_raw = (
            (k_roll_r * phi) / p.track_rear
            + (m_s * (p.a_cog / p.wheelbase) * ay * h_rc_r) / p.track_rear
            + (m_u_axle * ay * p.wheel_radius) / p.track_rear
        )

        dfz_lat_f = sclip(dfz_lat_f_raw, -0.5 * fz_f_tot, 0.5 * fz_f_tot, eps_fz)
        dfz_lat_r = sclip(dfz_lat_r_raw, -0.5 * fz_r_tot, 0.5 * fz_r_tot, eps_fz)

        fz = ca.vertcat(
            spos(0.5 * fz_f_tot - dfz_lat_f, eps_fz),
            spos(0.5 * fz_f_tot + dfz_lat_f, eps_fz),
            spos(0.5 * fz_r_tot - dfz_lat_r, eps_fz),
            spos(0.5 * fz_r_tot + dfz_lat_r, eps_fz),
        )
        return fz, phi

    def cambers_smooth(self, delta_w: Any, phi: Any, ax: Any = 0.0) -> Any:
        """Smooth C^2 ground camber gamma [FL, FR, RL, RR] (rad) in ISO 8855."""
        p = self.params
        cam_f_deg = self._dp("camber_f_deg", math.degrees(p.static_camber_f_rad))
        cam_r_deg = self._dp("camber_r_deg", math.degrees(p.static_camber_r_rad))
        cam_f_rad = cam_f_deg * (math.pi / 180.0)
        cam_r_rad = cam_r_deg * (math.pi / 180.0)

        # Pitch-induced camber delta from anti-dive / anti-squat deviation (0 at nominal!)
        d_ad = self._dp("anti_dive", p.anti_dive_f) - p.anti_dive_f
        d_as = self._dp("anti_squat", p.anti_squat_r) - p.anti_squat_r
        ax_brk_g = -sneg(ax, 0.25) / p.gravity
        ax_acc_g = spos(ax, 0.25) / p.gravity
        d_cam_f = -0.006 * d_ad * ax_brk_g
        d_cam_r = -0.005 * d_as * ax_acc_g

        g0 = (cam_f_rad + d_cam_f) - phi + p.caster_rad * ca.sin(delta_w[0]) + p.kpi_rad * (1.0 - ca.cos(delta_w[0]))
        g1 = -(cam_f_rad + d_cam_f) - phi + p.caster_rad * ca.sin(delta_w[1]) - p.kpi_rad * (1.0 - ca.cos(delta_w[1]))
        g2 = (cam_r_rad + d_cam_r) - phi
        g3 = -(cam_r_rad + d_cam_r) - phi
        return ca.vertcat(g0, g1, g2, g3)

    def motor_torque_limit_smooth(
        self, omega_w: Any, v_dc: Optional[float] = None
    ) -> Tuple[Any, Any]:
        """
        Smooth C^2 per-wheel drive torque upper bound T_max >= 0 and regen lower bound T_min <= 0 (Nm).
        Works on scalar or 4-vector omega_w (rad/s).
        """
        p = self.params
        s = self.smoothing
        gear_ratio = self._dp("gear_ratio", p.gear_ratio)
        v_eff = p.v_oc_nom if (v_dc is None or v_dc <= 0.0) else float(v_dc)
        v_scale = float(np.clip(v_eff / p.v_nom, 0.5, 1.25))
        omega_base_m = p.base_speed_rpm * (2.0 * math.pi / 60.0) * v_scale
        omega_max_m = p.motor_max_rpm * (2.0 * math.pi / 60.0)
        omega_taper_start = omega_max_m * p.motor_taper_ratio
        p_motor_peak = p.motor_peak_torque * omega_base_m

        om_m = sabs(omega_w, s["eps_omega_rads"]) * gear_ratio
        om_m_safe = smax(om_m, 1.0, 0.5)
        t_env = smin(p.motor_peak_torque, p_motor_peak / om_m_safe, s["eps_torque_nm"])
        taper_raw = 1.0 - (om_m - omega_taper_start) / max(omega_max_m - omega_taper_start, 1e-5)
        speed_scale = sclip(taper_raw, 0.0, 1.0, s["eps_taper"])
        t_max = t_env * gear_ratio * speed_scale
        t_min = -smin(p.t_regen_max_wheel, t_max, s["eps_regen_nm"])
        return t_max, t_min

    def eval_hv_power_kw(self, T_4: Any, omega_4: Any) -> Tuple[Any, Any]:
        """
        Smooth C^2 HV DC electrical power (kW).
        Returns (P_dc_ecu_kw, P_dc_drv_kw):
          - P_dc_ecu_kw uses eta_drive=0.90 (drive) and eta_regen=0.88 (regen), matching ECU
          - P_dc_drv_kw uses eta_drive=0.90 both ways, matching DrivetrainModel
        Strictly convex in mechanical power P_mech_i = T_i * omega_i.
        """
        p = self.params
        eps_p = self.smoothing["eps_power_w"]
        p_mech = T_4 * omega_4
        p_pos = spos(p_mech, eps_p)
        p_neg = sneg(p_mech, eps_p)
        p_ecu_w = ca.sum1(p_pos / p.eta_drive + p_neg * p.eta_regen)
        p_drv_w = ca.sum1(p_pos / p.eta_drive + p_neg * p.eta_drive)
        return p_ecu_w * 1e-3, p_drv_w * 1e-3

    def _wheel_kinematics(self, vx: Any, vy: Any, wz: Any, delta_w: Any) -> Tuple[Any, Any, Any, Any]:
        """Wheel longitudinal/lateral velocities, regularized denominator, and slip angles."""
        p = self.params
        eps_v = self.smoothing["eps_speed_ms"]
        x_w = ca.DM(self.x_w)
        y_w = ca.DM(self.y_w)
        cos_d = ca.cos(delta_w)
        sin_d = ca.sin(delta_w)
        v_hub_x = vx - wz * y_w
        v_hub_y = vy + wz * x_w
        v_wx = v_hub_x * cos_d + v_hub_y * sin_d
        v_wy = -v_hub_x * sin_d + v_hub_y * cos_d
        u_den = smax(sabs(v_wx, eps_v), p.slip_reg_speed, eps_v)
        alpha = -ca.atan(v_wy / u_den)
        return v_wx, v_wy, u_den, alpha

    def _hyd_split_vec(self) -> Any:
        bf = self._dp("brake_bias_front", self.params.brake_bias_front)
        if _is_casadi(bf):
            return ca.vertcat(0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf))
        return ca.DM([0.5 * float(bf), 0.5 * float(bf), 0.5 * (1.0 - float(bf)), 0.5 * (1.0 - float(bf))])

    def eval_forces_from_kappa(
        self,
        vx: Any,
        vy: Any,
        wz: Any,
        delta: Any,
        kappa_4: Any,
        F_brk_hyd: Any,
        ax_lt: Any,
        ay_lt: Any,
        T_tire_4: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Primary C^2 vehicle evaluation parameterized by wheel slip ratios kappa_4.
        Recommended for IPOPT NLPs and direct-collocation OCPs: zero Pacejka inversion,
        minimal expression graph, and exact sparse Hessians.

        Load transfer uses algebraic variables (ax_lt, ay_lt); in an NLP/OCP add the
        consistency constraints:
            ax_lt - Fx_tot / mass == 0
            ay_lt - Fy_tot / mass == 0

        Optional T_tire_4 (scalar or 4-vector tire bulk temp in C):
          - If provided (or self.tire_temp_c is set), applies smooth C^2 thermal grip
            multiplier thermal_grip_multiplier_smooth(T_tire_4) to MF6.2 forces.
          - Always computes slip_power_4 (W) and dT_tire_dt_4 (C/s) for thermal ODEs.
        """
        p = self.params
        s = self.smoothing
        delta_w = self.ackermann_steer_smooth(delta)
        cos_d = ca.cos(delta_w)
        sin_d = ca.sin(delta_w)

        v_wx, v_wy, u_den, alpha = self._wheel_kinematics(vx, vy, wz, delta_w)
        f_drag_x, f_drag_y, f_down_f, f_down_r = self.aero_forces_smooth(vx, vy, ax_lt)
        fz, phi = self.normal_forces_smooth(ax_lt, ay_lt, vx, vy)
        gamma = self.cambers_smooth(delta_w, phi, ax_lt)

        t_eff = T_tire_4 if T_tire_4 is not None else self.tire_temp_c
        if t_eff is not None:
            mu_thermal = thermal_grip_multiplier_smooth(t_eff)
        else:
            mu_thermal = 1.0

        side = ca.DM(self.side_sign)
        fx_w, fy_w, mz_tire, _ = self.eval_tire_mf62(kappa_4, alpha, fz, gamma, side, mu_thermal=mu_thermal)

        fx_c = fx_w * cos_d - fy_w * sin_d
        fy_c = fx_w * sin_d + fy_w * cos_d

        x_w = ca.DM(self.x_w)
        y_w = ca.DM(self.y_w)

        Fx_tot = ca.sum1(fx_c) + f_drag_x
        Fy_tot = ca.sum1(fy_c) + f_drag_y
        Mz_tot = ca.sum1(x_w * fy_c - y_w * fx_c + mz_tire)

        # Loaded & effective rolling radii
        rho_rad = smin(fz / p.k_tire_z, 0.5 * p.wheel_radius, s["eps_r_eff"])
        r_load = p.wheel_radius - rho_rad
        r_eff = p.wheel_radius - rho_rad / 3.0

        f_rr = p.rolling_resistance_coeff * fz * ca.tanh(v_wx / 0.5)
        T_total_4 = (fx_w + f_rr) * r_load
        hyd_split = self._hyd_split_vec()
        T_hyd_4 = F_brk_hyd * hyd_split * r_load
        T_motor_4 = T_total_4 + T_hyd_4

        omega_4 = (v_wx + kappa_4 * u_den) / smax(r_eff, 0.1, s["eps_r_eff"])
        T_max_4, T_min_4 = self.motor_torque_limit_smooth(omega_4)
        P_dc_ecu_kw, P_dc_drv_kw = self.eval_hv_power_kw(T_motor_4, omega_4)

        Fx_f_sin_delta = fx_w[0] * sin_d[0] + fx_w[1] * sin_d[1]
        Mz_tv_long = ca.sum1(-y_w * (fx_w * cos_d) + x_w * (fx_w * sin_d))

        # Tire thermal dissipation & bulk temperature rate
        slip_power_4 = slip_power_smooth(fx_w, fy_w, v_wx, v_wy, kappa_4)
        t_ref_for_ode = t_eff if t_eff is not None else (self.temp_c + 30.0)
        dT_tire_dt_4 = tire_temp_dot_smooth(t_ref_for_ode, self.temp_c, slip_power_4, vx)

        return {
            "Fx_tot": Fx_tot,
            "Fy_tot": Fy_tot,
            "Mz_tot": Mz_tot,
            "P_dc_kw": P_dc_ecu_kw,
            "P_dc_ecu_kw": P_dc_ecu_kw,
            "P_dc_drv_kw": P_dc_drv_kw,
            "Fz_4": fz,
            "Fx_w_4": fx_w,
            "Fy_w_4": fy_w,
            "Mz_tire_4": mz_tire,
            "alpha_4": alpha,
            "kappa_4": kappa_4,
            "T_motor_4": T_motor_4,
            "T_total_4": T_total_4,
            "omega_4": omega_4,
            "T_max_4": T_max_4,
            "T_min_4": T_min_4,
            "Fx_f_sin_delta": Fx_f_sin_delta,
            "Mz_tv_long": Mz_tv_long,
            "phi": phi,
            "delta_w_4": delta_w,
            "f_drag_x": f_drag_x,
            "f_drag_y": f_drag_y,
            "f_down_f": f_down_f,
            "f_down_r": f_down_r,
            "slip_power_4": slip_power_4,
            "dT_tire_dt_4": dT_tire_dt_4,
            "mu_thermal_4": mu_thermal,
        }

    def _invert_fx_to_kappa_smooth(
        self,
        fx_target: Any,
        alpha: Any,
        fz: Any,
        gamma: Any,
        side: Any,
    ) -> Any:
        """
        Smooth C^2 unrolled damped Newton inversion of MF 6.2 Fx(kappa, alpha, fz, gamma) = fx_target
        on the stable branch |kappa| <= 0.135.
        """
        t = self.tire
        s = self.smoothing
        p_infl = self.params.p_infl_kpa

        fz_pos = spos(fz, s["eps_fz_n"])
        dfz_raw = (fz_pos - t.fnomin) / t.fnomin
        dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"]) if math.isfinite(t.dfz_max) else dfz_raw
        dpi = (p_infl - t.pinom) / t.pinom

        # Linear longitudinal stiffness kxk & exact stable-branch peak kappa_peak
        cx = t.PCX1
        camber_f_x = smax(1.0 - t.PDX3 * gamma * gamma, 0.2, s["eps_camber_f"])
        mu_x = (
            (t.PDX1 + t.PDX2 * dfz)
            * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
            * camber_f_x
            * self.lmux
        )
        dx = mu_x * fz_pos
        kxk = (
            fz_pos
            * (t.PKX1 + t.PKX2 * dfz)
            * ca.exp(t.PKX3 * dfz)
            * (1.0 + t.PPX1 * dpi + t.PPX2 * dpi * dpi)
        )
        bx = kxk / (cx * dx + 1e-4)
        # Peak of sin(cx * atan(x - ex*(x - atan(x)))) for cx~1.5, ex~-0.18 is at x_pk ~ 1.70
        kappa_pk = sclip(1.65 / smax(bx, 5.0, 0.5), 0.04, 0.12, 1e-3)

        # Peak attainable Fx on stable branch [-kappa_pk, +kappa_pk]
        fx_pos_lim, _, _, _ = self.eval_tire_mf62(kappa_pk, alpha, fz, gamma, side)
        fx_neg_lim, _, _, _ = self.eval_tire_mf62(-kappa_pk, alpha, fz, gamma, side)
        fx_hi = smax(fx_pos_lim, fx_neg_lim, 1.0)
        fx_lo = smin(fx_pos_lim, fx_neg_lim, 1.0)
        # Smooth clamp strictly inside stable branch
        tgt = sclip(fx_target, 0.998 * fx_lo, 0.998 * fx_hi, 0.5)

        # Initial guess from linear stiffness (scaled by pure-slip combined factor at kappa=0)
        fx_0p, _, _, _ = self.eval_tire_mf62(1e-3 * ca.DM.ones(4), alpha, fz, gamma, side)
        fx_0m, _, _, _ = self.eval_tire_mf62(-1e-3 * ca.DM.ones(4), alpha, fz, gamma, side)
        k_slope0 = smax((fx_0p - fx_0m) / 2e-3, 50.0, 5.0)
        k_cur = sclip(tgt / k_slope0, -0.85 * kappa_pk, 0.85 * kappa_pk, 1e-3)

        # Unrolled damped Newton on monotone stable branch [-kappa_pk, +kappa_pk]
        h_fd = 1e-5
        n_iter = int(s["newton_iters_t2k"])
        for _ in range(n_iter):
            fx_c, _, _, _ = self.eval_tire_mf62(k_cur, alpha, fz, gamma, side)
            fx_p, _, _, _ = self.eval_tire_mf62(k_cur + h_fd, alpha, fz, gamma, side)
            dfx_dk = (fx_p - fx_c) / h_fd
            slope_safe = smax(dfx_dk, 0.05 * k_slope0, 5.0)
            step = sclip((fx_c - tgt) / slope_safe, -0.04, 0.04, 1e-3)
            k_cur = sclip(k_cur - step, -kappa_pk, kappa_pk, 1e-4)
        return k_cur

    def eval_forces_and_power(
        self,
        vx: Any,
        vy: Any,
        wz: Any,
        delta: Any,
        T_4: Any,
        F_brk_hyd: Any,
        ax_lt: Any,
        ay_lt: Any,
        T_tire_4: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Smooth C^2 vehicle evaluation parameterized by 4 motor torques T_4 (Nm) and
        hydraulic brake force F_brk_hyd (N >= 0).

        Inverts MF 6.2 smoothly on the stable slip branch to obtain kappa_4, then
        evaluates all forces, yaw moment, normal loads, and HV power.

        Note: For large direct-collocation OCPs (e.g. 380-node full lap), using
        eval_forces_from_kappa() with kappa_4 as control/algebraic variables is
        significantly faster and has a much smaller Hessian graph.
        """
        p = self.params
        s = self.smoothing
        delta_w = self.ackermann_steer_smooth(delta)
        v_wx, _, _, alpha = self._wheel_kinematics(vx, vy, wz, delta_w)
        fz, phi = self.normal_forces_smooth(ax_lt, ay_lt, vx, vy)
        gamma = self.cambers_smooth(delta_w, phi, ax_lt)

        rho_rad = smin(fz / p.k_tire_z, 0.5 * p.wheel_radius, s["eps_r_eff"])
        r_load = p.wheel_radius - rho_rad
        f_rr = p.rolling_resistance_coeff * fz * ca.tanh(v_wx / 0.5)
        hyd_split = self._hyd_split_vec()
        T_hyd_4 = F_brk_hyd * hyd_split * r_load
        fx_target = (T_4 - T_hyd_4) / r_load - f_rr

        side = ca.DM(self.side_sign)
        kappa_4 = self._invert_fx_to_kappa_smooth(fx_target, alpha, fz, gamma, side)

        out = self.eval_forces_from_kappa(vx, vy, wz, delta, kappa_4, F_brk_hyd, ax_lt, ay_lt, T_tire_4=T_tire_4)
        # Recompute HV power with exact input T_4 so P_dc_kw reflects commanded motor torques directly
        P_dc_ecu_kw, P_dc_drv_kw = self.eval_hv_power_kw(T_4, out["omega_4"])
        out["P_dc_kw"] = P_dc_ecu_kw
        out["P_dc_ecu_kw"] = P_dc_ecu_kw
        out["P_dc_drv_kw"] = P_dc_drv_kw
        return out


def build_casadi_vehicle(
    grip_profile: str = "FSG_REAL_TRACK",
    smoothing_overrides: Optional[Dict[str, float]] = None,
    mu_scale: Optional[Any] = None,
    temp_c: Optional[Any] = None,
    tire_temp_c: Optional[Any] = None,
    design_params: Optional[Dict[str, Any]] = None,
) -> TeR27CasADiVehicle:
    """
    Factory returning a compiled smooth C^2 CasADi model of the TeR27 4WD.

    Args:
        grip_profile: "FSG_RUBBERED", "FSG_REAL_TRACK", "TTC_LAB_PEAK", "COLD_LOW_GRIP", or "CUSTOM".
        smoothing_overrides: Optional dict overriding entries in SMOOTHING_PARAMS.
        mu_scale: Optional asphalt grip scale override (float or symbolic SX/MX).
        temp_c: Optional track/asphalt temperature in C (float or symbolic SX/MX).
        tire_temp_c: Optional fixed or symbolic tire bulk temp in C (None = thermal mult 1.0).
        design_params: Optional dict of float or symbolic overrides for setup/design parameters:
            lltd, k_roll_scale, h_rc_f, h_rc_r, camber_f_deg, camber_r_deg, anti_dive,
            anti_squat, brake_bias_front, gear_ratio, cl_scale, cd_scale, cop_front_frac,
            h_cg, i_zz, mass_kg.
    """
    prof_key = str(grip_profile)
    alias_map = {
        "fsg_rubbered": "FSG_RUBBERED",
        "fsg_real": "FSG_REAL_TRACK",
        "ttc_lab": "TTC_LAB_PEAK",
        "cold_low_grip": "COLD_LOW_GRIP",
        "custom": "CUSTOM",
    }
    prof_key = alias_map.get(prof_key.lower(), prof_key)
    if prof_key not in CASADI_GRIP_PROFILES:
        raise ValueError(
            f"Unknown grip_profile '{grip_profile}'. Expected one of {list(CASADI_GRIP_PROFILES)}"
        )

    p = load_ter27_params()
    tire = PacejkaMF62(p.tire_compound)
    prof_cfg = CASADI_GRIP_PROFILES[prof_key]
    env_mu = prof_cfg["env_mu_scale"] if mu_scale is None else mu_scale
    env_cl = prof_cfg["env_aero_cl_scale"]
    t_track = prof_cfg["default_temp_c"] if temp_c is None else temp_c
    lmux = p.lambda_mux_base * env_mu
    lmuy = p.lambda_muy_base * env_mu

    dp_merged: Dict[str, Any] = default_design_params(p)
    if design_params is not None:
        unknown = set(design_params.keys()) - set(DESIGN_PARAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown design_params keys: {sorted(unknown)}. Allowed: {DESIGN_PARAM_NAMES}")
        dp_merged.update(design_params)

    x_w = np.array([p.a_cog, p.a_cog, -p.b_cog, -p.b_cog], dtype=float)
    y_w = np.array(
        [0.5 * p.track_front, -0.5 * p.track_front, 0.5 * p.track_rear, -0.5 * p.track_rear],
        dtype=float,
    )
    bf = dp_merged["brake_bias_front"]
    if _is_casadi(bf):
        hyd_split: Any = ca.vertcat(0.5 * bf, 0.5 * bf, 0.5 * (1.0 - bf), 0.5 * (1.0 - bf))
    else:
        hyd_split = np.array([0.5 * float(bf), 0.5 * float(bf), 0.5 * (1.0 - float(bf)), 0.5 * (1.0 - float(bf))], dtype=float)
    side_sign = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)

    smoothing = dict(SMOOTHING_PARAMS)
    if smoothing_overrides:
        smoothing.update(smoothing_overrides)

    has_sym = (
        _is_casadi(env_mu)
        or _is_casadi(t_track)
        or (tire_temp_c is not None and _is_casadi(tire_temp_c))
        or any(_is_casadi(v) for v in dp_merged.values())
    )

    veh = TeR27CasADiVehicle(
        grip_profile=prof_key,
        params=p,
        tire=tire,
        env_mu_scale=env_mu,
        env_aero_cl_scale=env_cl,
        lmux=lmux,
        lmuy=lmuy,
        x_w=x_w,
        y_w=y_w,
        hyd_split=hyd_split,
        side_sign=side_sign,
        smoothing=smoothing,
        temp_c=t_track,
        tire_temp_c=tire_temp_c,
        design_params=dp_merged,
    )

    if has_sym:
        # When symbolic design_params / mu_scale / temp_c are provided, callers use
        # veh.eval_* directly inside their own SX/MX NLP or ca.Function graphs.
        return veh

    # 1. fn_tire_mf62
    s_kap = ca.SX.sym("kappa", 4)
    s_alp = ca.SX.sym("alpha", 4)
    s_fz = ca.SX.sym("fz", 4)
    s_gam = ca.SX.sym("gamma", 4)
    s_side = ca.SX.sym("side", 4)
    fx_t, fy_t, mz_t, muy_t = veh.eval_tire_mf62(s_kap, s_alp, s_fz, s_gam, s_side)
    veh.fn_tire_mf62 = ca.Function(
        f"ter27_tire_mf62_{prof_key}",
        [s_kap, s_alp, s_fz, s_gam, s_side],
        [fx_t, fy_t, mz_t, muy_t],
        ["kappa", "alpha", "fz", "gamma", "side"],
        ["fx", "fy", "mz", "mu_y_eff"],
    )

    # 2. fn_motor_limits
    s_om = ca.SX.sym("omega_4", 4)
    t_max, t_min = veh.motor_torque_limit_smooth(s_om)
    veh.fn_motor_limits = ca.Function(
        f"ter27_motor_limits_{prof_key}",
        [s_om],
        [t_max, t_min],
        ["omega_4"],
        ["T_max_4", "T_min_4"],
    )

    # 3. fn_hv_power_kw
    s_T4 = ca.SX.sym("T_4", 4)
    p_ecu_kw, p_drv_kw = veh.eval_hv_power_kw(s_T4, s_om)
    veh.fn_hv_power_kw = ca.Function(
        f"ter27_hv_power_kw_{prof_key}",
        [s_T4, s_om],
        [p_ecu_kw, p_drv_kw],
        ["T_4", "omega_4"],
        ["P_dc_ecu_kw", "P_dc_drv_kw"],
    )

    # 4. fn_normal_forces
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")
    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    fz_n, phi_n = veh.normal_forces_smooth(s_ax, s_ay, s_vx, s_vy)
    veh.fn_normal_forces = ca.Function(
        f"ter27_normal_forces_{prof_key}",
        [s_ax, s_ay, s_vx, s_vy],
        [fz_n, phi_n],
        ["ax_lt", "ay_lt", "vx", "vy"],
        ["Fz_4", "phi"],
    )

    # 5. fn_forces_from_kappa
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_fbrk = ca.SX.sym("F_brk_hyd")
    out_k = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay)
    out_k_names = [
        "Fx_tot",
        "Fy_tot",
        "Mz_tot",
        "P_dc_kw",
        "P_dc_drv_kw",
        "Fz_4",
        "Fx_w_4",
        "Fy_w_4",
        "Mz_tire_4",
        "alpha_4",
        "T_motor_4",
        "omega_4",
        "T_max_4",
        "T_min_4",
        "Fx_f_sin_delta",
        "Mz_tv_long",
        "phi",
    ]
    veh.fn_forces_from_kappa = ca.Function(
        f"ter27_forces_from_kappa_{prof_key}",
        [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay],
        [out_k[k] for k in out_k_names],
        ["vx", "vy", "wz", "delta", "kappa_4", "F_brk_hyd", "ax_lt", "ay_lt"],
        out_k_names,
    )

    # 6. fn_forces_and_power (parameterized by T_4)
    out_t = veh.eval_forces_and_power(s_vx, s_vy, s_wz, s_del, s_T4, s_fbrk, s_ax, s_ay)
    out_t_names = [
        "Fx_tot",
        "Fy_tot",
        "Mz_tot",
        "P_dc_kw",
        "P_dc_drv_kw",
        "Fz_4",
        "Fx_w_4",
        "Fy_w_4",
        "alpha_4",
        "kappa_4",
        "Fx_f_sin_delta",
        "omega_4",
        "T_max_4",
        "T_min_4",
        "Mz_tv_long",
    ]
    veh.fn_forces_and_power = ca.Function(
        f"ter27_forces_and_power_{prof_key}",
        [s_vx, s_vy, s_wz, s_del, s_T4, s_fbrk, s_ax, s_ay],
        [out_t[k] for k in out_t_names],
        ["vx", "vy", "wz", "delta", "T_4", "F_brk_hyd", "ax_lt", "ay_lt"],
        out_t_names,
    )

    return veh


# ---------------------------------------------------------------------------
# Validation Suite against ter27_physics.py + C^2 Jacobian/Hessian Checks
# ---------------------------------------------------------------------------

def _generate_operating_points() -> list[Tuple[str, float, float, float, float, np.ndarray, float]]:
    """
    Generates 60 diverse physical operating points:
      (regime, vx, vy, wz, delta, kappa_4, F_brk_hyd)
    covering hairpins, skidpads, high-speed aero corners, heavy regen+hyd braking,
    80 kW launch, straight-line accel/decel, right/left turns, and mixed slip.
    """
    pts: list[Tuple[str, float, float, float, float, np.ndarray, float]] = []
    rng = np.random.default_rng(2026)

    # 1. Low-speed hairpins (R ~ 5-6 m, left & right, asymmetric TV regen/drive)
    for i in range(10):
        sgn = 1.0 if i % 2 == 0 else -1.0
        vx = float(rng.uniform(7.0, 10.5))
        vy = float(-sgn * rng.uniform(0.1, 0.8))
        wz = float(sgn * vx / rng.uniform(5.0, 6.5))
        delta = float(sgn * rng.uniform(0.22, 0.40))
        kap = np.array([
            rng.uniform(-0.06, 0.04) * sgn,
            rng.uniform(0.01, 0.08) * sgn,
            rng.uniform(-0.08, 0.05) * sgn,
            rng.uniform(0.02, 0.09) * sgn,
        ])
        pts.append(("hairpin", vx, vy, wz, delta, kap, 0.0))

    # 2. Skidpad steady cornering (R = 9.125 m, left & right)
    for i in range(10):
        sgn = 1.0 if i % 2 == 0 else -1.0
        vx = float(rng.uniform(11.0, 14.5))
        vy = float(-sgn * rng.uniform(0.15, 0.9))
        wz = float(sgn * vx / 9.125)
        delta = float(sgn * rng.uniform(0.14, 0.25))
        kap = rng.uniform(-0.04, 0.06, size=4)
        pts.append(("skidpad", vx, vy, wz, delta, kap, 0.0))

    # 3. High-speed aero corners (R = 20-35 m, v = 22-32 m/s)
    for i in range(10):
        sgn = 1.0 if i % 2 == 0 else -1.0
        vx = float(rng.uniform(22.0, 32.0))
        vy = float(-sgn * rng.uniform(0.1, 1.1))
        wz = float(sgn * vx / rng.uniform(20.0, 35.0))
        delta = float(sgn * rng.uniform(0.04, 0.12))
        kap = rng.uniform(-0.03, 0.07, size=4)
        pts.append(("high_speed_aero", vx, vy, wz, delta, kap, 0.0))

    # 4. Heavy regen + hydraulic braking (straight & trail braking)
    for i in range(10):
        sgn = 1.0 if i % 2 == 0 else -1.0
        vx = float(rng.uniform(14.0, 30.0))
        vy = float(-sgn * rng.uniform(0.0, 0.5))
        wz = float(sgn * rng.uniform(0.0, 0.8))
        delta = float(sgn * rng.uniform(0.0, 0.15))
        kap = rng.uniform(-0.10, -0.02, size=4)
        f_brk = float(rng.uniform(500.0, 4500.0))
        pts.append(("heavy_braking", vx, vy, wz, delta, kap, f_brk))

    # 5. Full-throttle ~80 kW launch & straight-line accel
    for i in range(10):
        vx = float(rng.uniform(5.0, 28.0))
        vy = float(rng.uniform(-0.1, 0.1))
        wz = float(rng.uniform(-0.05, 0.05))
        delta = float(rng.uniform(-0.03, 0.03))
        kap = rng.uniform(0.03, 0.10, size=4)
        pts.append(("full_throttle", vx, vy, wz, delta, kap, 0.0))

    # 6. Mixed extreme slip / zero-steer / edge transitions
    for i in range(10):
        sgn = 1.0 if i % 2 == 0 else -1.0
        vx = float(rng.uniform(4.0, 26.0))
        vy = float(sgn * rng.uniform(-1.0, 1.0))
        wz = float(sgn * rng.uniform(0.0, 1.6))
        delta = 0.0 if i < 2 else float(sgn * rng.uniform(0.01, 0.35))
        kap = np.zeros(4) if i == 0 else rng.uniform(-0.09, 0.09, size=4)
        f_brk = 0.0 if i % 2 == 0 else float(rng.uniform(100.0, 2000.0))
        pts.append(("mixed_slip", vx, vy, wz, delta, kap, f_brk))

    return pts


def validate_against_numpy(verbose: bool = True) -> Dict[str, Any]:
    """
    Validates ter27_casadi_model.py against ter27_physics.py across 60 diverse
    operating points in BOTH FSG_REAL_TRACK and TTC_LAB_PEAK, and checks C^2
    Jacobians and Hessians for 0 NaNs / 0 Infs.
    """
    pts = _generate_operating_points()
    summary: Dict[str, Any] = {}

    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        ref = TeR27PhysicsModel(prof)
        veh = build_casadi_vehicle(prof)

        # Build Jacobian & Hessian test functions for fn_forces_from_kappa
        x_sym = ca.SX.sym("x", 11)  # [vx, vy, wz, delta, k0..k3, F_brk, ax_lt, ay_lt]
        res_sym = veh.eval_forces_from_kappa(
            x_sym[0], x_sym[1], x_sym[2], x_sym[3], x_sym[4:8], x_sym[8], x_sym[9], x_sym[10]
        )
        y_vec = ca.vertcat(
            res_sym["Fx_tot"],
            res_sym["Fy_tot"],
            res_sym["Mz_tot"],
            res_sym["P_dc_kw"],
            res_sym["Fz_4"],
            res_sym["T_max_4"],
            res_sym["T_min_4"],
        )
        jac_fn = ca.Function("jac_test", [x_sym], [ca.jacobian(y_vec, x_sym)])
        scalar_obj = (
            1e-3 * res_sym["Fx_tot"]
            + 1e-3 * res_sym["Fy_tot"]
            + 1e-3 * res_sym["Mz_tot"]
            + res_sym["P_dc_kw"]
            + 1e-3 * ca.sum1(res_sym["Fz_4"])
        )
        hess_mat, _ = ca.hessian(scalar_obj, x_sym)
        hess_fn = ca.Function("hess_test", [x_sym], [hess_mat])

        # Also build Jacobian & Hessian test for fn_forces_and_power (T_4 interface)
        xt_sym = ca.SX.sym("xt", 11)  # [vx, vy, wz, delta, T0..T3, F_brk, ax_lt, ay_lt]
        rest_sym = veh.eval_forces_and_power(
            xt_sym[0], xt_sym[1], xt_sym[2], xt_sym[3], xt_sym[4:8], xt_sym[8], xt_sym[9], xt_sym[10]
        )
        yt_vec = ca.vertcat(rest_sym["Fx_tot"], rest_sym["Fy_tot"], rest_sym["Mz_tot"], rest_sym["P_dc_kw"])
        jact_fn = ca.Function("jact_test", [xt_sym], [ca.jacobian(yt_vec, xt_sym)])
        obj_t = 1e-3 * rest_sym["Fx_tot"] + 1e-3 * rest_sym["Fy_tot"] + 1e-3 * rest_sym["Mz_tot"] + rest_sym["P_dc_kw"]
        hesst_mat, _ = ca.hessian(obj_t, xt_sym)
        hesst_fn = ca.Function("hesst_test", [xt_sym], [hesst_mat])

        err_k: Dict[str, list[float]] = {k: [] for k in ("Fx_tot", "Fy_tot", "Mz_tot", "Fz_4", "P_dc_kw", "T_lim")}
        err_t: Dict[str, list[float]] = {k: [] for k in ("Fx_tot", "Fy_tot", "Mz_tot", "Fz_4", "P_dc_kw")}
        nan_inf_count = 0
        max_hess_norm = 0.0

        # Characteristic scales for relative error denominator when a quantity is near 0
        # (avoids dividing 0.01 N by 0.0 N on straight-line Fy = 0, while remaining strict)
        scales = {"Fx_tot": 500.0, "Fy_tot": 500.0, "Mz_tot": 200.0, "Fz_4": 100.0, "P_dc_kw": 2.0, "T_lim": 20.0}

        for _, vx, vy, wz, delta, kap, f_brk in pts:
            # Reference evaluation (exact ter27_physics.py)
            ax_est = 0.0
            ay_est = vx * wz
            r_ref = ref.evaluate_state_from_kappa(vx, vy, wz, delta, kap, ax_est, ay_est, n_load_iter=6)
            ax_lt = float(r_ref["ax"])
            ay_lt = float(r_ref["ay"])
            # One clean single-pass reference at the exact converged (ax_lt, ay_lt) so both use identical (ax_lt, ay_lt)
            delta_w_ref = ref.ackermann_steer(delta)
            v_wx_ref, _, alpha_ref = ref.wheel_velocities_and_alphas(vx, vy, wz, delta_w_ref)
            f_dx_ref, f_dy_ref, _, _ = ref.aero_forces(vx, vy)
            fz_ref, phi_ref = ref.compute_normal_forces(ax_lt, ay_lt, vx, vy)
            gam_ref = ref.compute_cambers(delta_w_ref, phi_ref)
            fxw_ref, fyw_ref, mzt_ref, _ = ref.tire.compute_combined_forces_vec(
                kap, alpha_ref, fz_ref, gam_ref, ref.lmux, ref.lmuy, ref.IS_RIGHT, ref.p.p_infl_kpa
            )
            cos_d = np.cos(delta_w_ref)
            sin_d = np.sin(delta_w_ref)
            fxc_ref = fxw_ref * cos_d - fyw_ref * sin_d
            fyc_ref = fxw_ref * sin_d + fyw_ref * cos_d
            Fx_ref = float(np.sum(fxc_ref) + f_dx_ref)
            Fy_ref = float(np.sum(fyc_ref) + f_dy_ref)
            Mz_ref = float(np.sum(ref.x_w * fyc_ref - ref.y_w * fxc_ref + mzt_ref))
            rho_ref = np.minimum(fz_ref / ref.p.k_tire_z, 0.5 * ref.p.wheel_radius)
            r_load_ref = ref.p.wheel_radius - rho_ref
            r_eff_ref = ref.p.wheel_radius - rho_ref / 3.0
            frr_ref = ref.p.rolling_resistance_coeff * fz_ref * np.tanh(v_wx_ref / 0.5)
            T_tot_ref = (fxw_ref + frr_ref) * r_load_ref
            T_hyd_ref = f_brk * veh.hyd_split * r_load_ref
            T_mot_ref = T_tot_ref + T_hyd_ref
            u_den_ref = np.maximum(np.abs(v_wx_ref), ref.p.slip_reg_speed)
            om_ref = (v_wx_ref + kap * u_den_ref) / np.maximum(r_eff_ref, 0.1)
            P_ecu_ref = ref.compute_hv_power_w(T_mot_ref, om_ref) * 1e-3
            T_max_ref, T_min_ref = ref.wheel_motor_limits(om_ref)

            # 1) CasADi eval_forces_from_kappa
            out_k = veh.fn_forces_from_kappa(vx, vy, wz, delta, kap, f_brk, ax_lt, ay_lt)
            Fx_c = float(out_k[0])
            Fy_c = float(out_k[1])
            Mz_c = float(out_k[2])
            P_c = float(out_k[3])
            Fz_c = np.asarray(out_k[5], dtype=float).ravel()
            Tmax_c = np.asarray(out_k[12], dtype=float).ravel()
            Tmin_c = np.asarray(out_k[13], dtype=float).ravel()

            err_k["Fx_tot"].append(abs(Fx_c - Fx_ref) / max(abs(Fx_ref), scales["Fx_tot"]))
            err_k["Fy_tot"].append(abs(Fy_c - Fy_ref) / max(abs(Fy_ref), scales["Fy_tot"]))
            err_k["Mz_tot"].append(abs(Mz_c - Mz_ref) / max(abs(Mz_ref), scales["Mz_tot"]))
            err_k["Fz_4"].append(float(np.max(np.abs(Fz_c - fz_ref) / np.maximum(np.abs(fz_ref), scales["Fz_4"]))))
            err_k["P_dc_kw"].append(abs(P_c - P_ecu_ref) / max(abs(P_ecu_ref), scales["P_dc_kw"]))
            err_k["T_lim"].append(
                float(
                    max(
                        np.max(np.abs(Tmax_c - T_max_ref) / np.maximum(np.abs(T_max_ref), scales["T_lim"])),
                        np.max(np.abs(Tmin_c - T_min_ref) / np.maximum(np.abs(T_min_ref), scales["T_lim"])),
                    )
                )
            )

            # 2) CasADi eval_forces_and_power (T_4 interface, tested on bijective stable slip branch |kappa| <= 0.05)
            kap_st = np.clip(kap, -0.05, 0.05)
            fxw_st, fyw_st, mzt_st, _ = ref.tire.compute_combined_forces_vec(
                kap_st, alpha_ref, fz_ref, gam_ref, ref.lmux, ref.lmuy, ref.IS_RIGHT, ref.p.p_infl_kpa
            )
            fxc_st = fxw_st * cos_d - fyw_st * sin_d
            fyc_st = fxw_st * sin_d + fyw_st * cos_d
            Fx_st_ref = float(np.sum(fxc_st) + f_dx_ref)
            Fy_st_ref = float(np.sum(fyc_st) + f_dy_ref)
            Mz_st_ref = float(np.sum(ref.x_w * fyc_st - ref.y_w * fxc_st + mzt_st))
            T_mot_st = (fxw_st + frr_ref) * r_load_ref + T_hyd_ref
            om_st = (v_wx_ref + kap_st * u_den_ref) / np.maximum(r_eff_ref, 0.1)
            P_ecu_st_ref = ref.compute_hv_power_w(T_mot_st, om_st) * 1e-3

            out_t = veh.fn_forces_and_power(vx, vy, wz, delta, T_mot_st, f_brk, ax_lt, ay_lt)
            Fx_ct = float(out_t[0])
            Fy_ct = float(out_t[1])
            Mz_ct = float(out_t[2])
            P_ct = float(out_t[3])
            Fz_ct = np.asarray(out_t[5], dtype=float).ravel()

            err_t["Fx_tot"].append(abs(Fx_ct - Fx_st_ref) / max(abs(Fx_st_ref), scales["Fx_tot"]))
            err_t["Fy_tot"].append(abs(Fy_ct - Fy_st_ref) / max(abs(Fy_st_ref), scales["Fy_tot"]))
            err_t["Mz_tot"].append(abs(Mz_ct - Mz_st_ref) / max(abs(Mz_st_ref), scales["Mz_tot"]))
            err_t["Fz_4"].append(float(np.max(np.abs(Fz_ct - fz_ref) / np.maximum(np.abs(fz_ref), scales["Fz_4"]))))
            err_t["P_dc_kw"].append(abs(P_ct - P_ecu_st_ref) / max(abs(P_ecu_st_ref), scales["P_dc_kw"]))

            # 3) Jacobian & Hessian checks (both interfaces)
            xv = np.concatenate([[vx, vy, wz, delta], kap, [f_brk, ax_lt, ay_lt]])
            J = np.asarray(jac_fn(xv), dtype=float)
            H = np.asarray(hess_fn(xv), dtype=float)
            xtv = np.concatenate([[vx, vy, wz, delta], T_mot_st, [f_brk, ax_lt, ay_lt]])
            Jt = np.asarray(jact_fn(xtv), dtype=float)
            Ht = np.asarray(hesst_fn(xtv), dtype=float)
            for mat in (J, H, Jt, Ht):
                if not np.all(np.isfinite(mat)):
                    nan_inf_count += 1
            max_hess_norm = max(max_hess_norm, float(np.max(np.abs(H))))

        prof_res = {
            "n_points": len(pts),
            "kappa_interface_max_rel_err_pct": {k: 100.0 * max(v) for k, v in err_k.items()},
            "torque_interface_max_rel_err_pct": {k: 100.0 * max(v) for k, v in err_t.items()},
            "nan_inf_count": nan_inf_count,
            "max_hess_entry": max_hess_norm,
        }
        summary[prof] = prof_res

        if verbose:
            print(f"\n=== Validation: {prof} ({len(pts)} operating points) ===", flush=True)
            print("  [kappa interface] max relative error (%):", flush=True)
            for k, val in prof_res["kappa_interface_max_rel_err_pct"].items():
                print(f"    {k:10s}: {val:.6f}%", flush=True)
            print("  [T_4 interface]   max relative error (%):", flush=True)
            for k, val in prof_res["torque_interface_max_rel_err_pct"].items():
                print(f"    {k:10s}: {val:.6f}%", flush=True)
            print(f"  Jacobian/Hessian NaN/Inf count: {nan_inf_count}", flush=True)
            print(f"  Max |Hessian| entry (kappa ifc): {max_hess_norm:.3e}", flush=True)

        # Strict assertions
        for k in ("Fx_tot", "Fy_tot", "Mz_tot", "Fz_4", "P_dc_kw"):
            assert prof_res["kappa_interface_max_rel_err_pct"][k] < 0.1, (
                f"{prof} kappa interface {k} error {prof_res['kappa_interface_max_rel_err_pct'][k]:.4f}% >= 0.1%"
            )
            assert prof_res["torque_interface_max_rel_err_pct"][k] < 0.1, (
                f"{prof} T_4 interface {k} error {prof_res['torque_interface_max_rel_err_pct'][k]:.4f}% >= 0.1%"
            )
        assert nan_inf_count == 0, f"{prof} produced {nan_inf_count} NaN/Inf Jacobian/Hessian matrices!"

    if verbose:
        print("\nALL VALIDATION CHECKS PASSED (< 0.1% error, 0 NaNs/Infs in J & H).", flush=True)
    return summary


if __name__ == "__main__":
    validate_against_numpy(verbose=True)
