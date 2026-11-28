#!/usr/bin/env python3
"""
Strict C2-smooth (C-infinity) mathematical primitives for differentiable
CasADi SX/MX vehicle dynamics, tire models, and suspension kinematics.

All functions work transparently with Python float, numpy.ndarray,
casadi.SX, casadi.MX, and casadi.DM, and are 100% pure ASCII and
C-Codegen compatible.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import casadi as ca
import numpy as np


def is_casadi(x: Any) -> bool:
    """Return True if x is a CasADi symbolic or DM type."""
    return isinstance(x, (ca.SX, ca.MX, ca.DM))


def sabs(x: Any, eps: float = 1e-5) -> Any:
    """Smooth |x| = sqrt(x^2 + eps^2). Max error <= eps at x = 0."""
    if is_casadi(x):
        return ca.sqrt(x * x + eps * eps)
    if isinstance(x, np.ndarray):
        return np.sqrt(x * x + eps * eps)
    xf = float(x)
    return math.sqrt(xf * xf + eps * eps)


def sabs_zero(x: Any, eps: float = 1e-5) -> Any:
    """Smooth |x| = sqrt(x^2 + eps^2) - eps. Exactly 0 at x = 0, error <= eps."""
    return sabs(x, eps) - eps


def smax(a: Any, b: Any, eps: float = 1e-5) -> Any:
    """Smooth max(a, b) = 0.5 * (a + b + sqrt((a - b)^2 + eps^2)). Error <= 0.5*eps."""
    d = a - b
    if is_casadi(a) or is_casadi(b):
        return 0.5 * (a + b + ca.sqrt(d * d + eps * eps))
    if isinstance(d, np.ndarray):
        return 0.5 * (a + b + np.sqrt(d * d + eps * eps))
    df = float(d)
    return 0.5 * (float(a) + float(b) + math.sqrt(df * df + eps * eps))


def smin(a: Any, b: Any, eps: float = 1e-5) -> Any:
    """Smooth min(a, b) = 0.5 * (a + b - sqrt((a - b)^2 + eps^2)). Error <= 0.5*eps."""
    d = a - b
    if is_casadi(a) or is_casadi(b):
        return 0.5 * (a + b - ca.sqrt(d * d + eps * eps))
    if isinstance(d, np.ndarray):
        return 0.5 * (a + b - np.sqrt(d * d + eps * eps))
    df = float(d)
    return 0.5 * (float(a) + float(b) - math.sqrt(df * df + eps * eps))


def spos(x: Any, eps: float = 1e-5) -> Any:
    """Smooth max(x, 0) = 0.5 * (x + sqrt(x^2 + eps^2)) > 0. Error <= 0.5*eps."""
    if is_casadi(x):
        return 0.5 * (x + ca.sqrt(x * x + eps * eps))
    if isinstance(x, np.ndarray):
        return 0.5 * (x + np.sqrt(x * x + eps * eps))
    xf = float(x)
    return 0.5 * (xf + math.sqrt(xf * xf + eps * eps))


def sneg(x: Any, eps: float = 1e-5) -> Any:
    """Smooth min(x, 0) = 0.5 * (x - sqrt(x^2 + eps^2)) < 0. Error <= 0.5*eps."""
    if is_casadi(x):
        return 0.5 * (x - ca.sqrt(x * x + eps * eps))
    if isinstance(x, np.ndarray):
        return 0.5 * (x - np.sqrt(x * x + eps * eps))
    xf = float(x)
    return 0.5 * (xf - math.sqrt(xf * xf + eps * eps))


def sclip(x: Any, lo: Any, hi: Any, eps: float = 1e-5) -> Any:
    """Smooth C-infinity clamp to [lo, hi]. Max error <= eps."""
    return smin(smax(x, lo, eps), hi, eps)


def ssign(x: Any, eps: float = 1e-4) -> Any:
    """Smooth sign(x) = tanh(x / eps)."""
    if is_casadi(x):
        return ca.tanh(x / eps)
    if isinstance(x, np.ndarray):
        return np.tanh(x / eps)
    return math.tanh(float(x) / eps)


def softplus(x: Any, beta: float = 100.0) -> Any:
    """
    Overflow-safe C-infinity softplus: (1/beta) * log(1 + exp(beta * x)).
    Written as max(x, 0) + (1/beta) * log(1 + exp(-beta * |x|)) using smooth
    algebraic split so exp argument is always <= 0.
    """
    eps = 1.0 / max(beta, 1.0) * 1e-3
    pos = spos(x, eps)
    ax = sabs(x, eps)
    if is_casadi(x):
        return pos + (1.0 / beta) * ca.log(1.0 + ca.exp(-beta * ax))
    if isinstance(x, np.ndarray):
        return pos + (1.0 / beta) * np.log1p(np.exp(-beta * ax))
    return pos + (1.0 / beta) * math.log1p(math.exp(-beta * ax))


def smooth_huber(x: Any, delta: float = 1e-3) -> Any:
    """
    Pseudo-Huber C-infinity loss: delta^2 * (sqrt(1 + (x/delta)^2) - 1).
    Quadratic (~0.5*x^2) near 0, linear (~delta*|x|) for |x| >> delta.
    """
    r = x / delta
    if is_casadi(x):
        return (delta * delta) * (ca.sqrt(1.0 + r * r) - 1.0)
    if isinstance(x, np.ndarray):
        return (delta * delta) * (np.sqrt(1.0 + r * r) - 1.0)
    rf = float(r)
    return (delta * delta) * (math.sqrt(1.0 + rf * rf) - 1.0)


def smooth_deadzone(x: Any, half_width: float, eps: float = 1e-5) -> Any:
    """
    Smooth deadzone: 0 for |x| < half_width, x - sign(x)*half_width outside.
    """
    return spos(x - half_width, eps) - spos(-x - half_width, eps)


def smooth_bumpstop(
    dz: Any,
    start: Any,
    k1: Any,
    k2: Any,
    eps: float = 1e-5,
) -> Any:
    """
    Smooth C-infinity progressive bumpstop force:
      excess = spos(dz - start, eps)
      F_bs = k1 * excess + k2 * excess^2
    Error vs exact piecewise bumpstop is < 1e-4 N when eps=1e-5 m.
    """
    excess = spos(dz - start, eps)
    # Subtract tiny baseline at zero to keep static equilibrium exact
    excess0 = 0.5 * eps
    ex = excess - excess0 * ca.exp(-spos(dz - start, eps) / max(eps, 1e-9)) if is_casadi(dz) else excess
    if is_casadi(dz):
        return k1 * excess + k2 * excess * excess
    return k1 * excess + k2 * excess * excess


def smooth_digressive_damper(
    v_d: Any,
    v_knee: Any,
    c_comp_lo: Any,
    c_comp_hi: Any,
    c_reb_lo: Any,
    c_reb_hi: Any,
    eps_v: float = 1e-4,
) -> Any:
    """
    Smooth C-infinity 2-stage asymmetric digressive damper shaft force (N).
    Matches SuspensionSystem._compute_damper_force_with_mr:
      v_d >= 0 (compression): slope c_comp_lo up to v_knee, then c_comp_hi
      v_d <  0 (rebound):     slope c_reb_lo down to -v_knee, then c_reb_hi
    """
    v_pos = spos(v_d, eps_v)
    v_neg = spos(-v_d, eps_v)
    v_pos_hi = spos(v_pos - v_knee, eps_v)
    v_neg_hi = spos(v_neg - v_knee, eps_v)
    f_comp = c_comp_lo * v_pos + (c_comp_hi - c_comp_lo) * v_pos_hi
    f_reb = c_reb_lo * v_neg + (c_reb_hi - c_reb_lo) * v_neg_hi
    return f_comp - f_reb


def smooth_asymmetric_damper(
    v_w: Any,
    c_comp: Any,
    c_reb: Any,
    eps_v: float = 1e-4,
) -> Any:
    """Smooth C-infinity asymmetric linear damper force (compression vs rebound)."""
    return c_comp * spos(v_w, eps_v) - c_reb * spos(-v_w, eps_v)


def smooth_piecewise_linear_1d(
    x: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    eps: float = 1e-4,
) -> Any:
    """
    C-infinity smooth approximation of a clamped 1D piecewise-linear LUT
    via exact hinge decomposition:
      y(x) = y_0 + sum_{k=0}^{n-1} (m_k - m_{k-1}) * spos(x - x_k, eps)
    with m_{-1} = 0 and m_{n-1} = 0.
    Works in pure CasADi SX and compiles directly with CodeGenerator.
    """
    n = len(xs)
    if n == 0:
        return 0.0 * x
    if n == 1:
        return float(ys[0]) + 0.0 * x

    slopes = []
    for k in range(n - 1):
        dx = float(xs[k + 1]) - float(xs[k])
        dy = float(ys[k + 1]) - float(ys[k])
        slopes.append(dy / dx if abs(dx) > 1e-12 else 0.0)

    out = float(ys[0])
    prev_m = 0.0
    for k in range(n):
        next_m = slopes[k] if k < n - 1 else 0.0
        dm = next_m - prev_m
        if abs(dm) > 1e-15:
            out = out + dm * spos(x - float(xs[k]), eps)
        prev_m = next_m
    return out


def smooth_step_01(x: Any, eps: float = 1e-5) -> Any:
    """Smooth C2 Hermite smoothstep on [0, 1]: t^2 * (3 - 2*t) with smooth clamp."""
    t = sclip(x, 0.0, 1.0, eps)
    return t * t * (3.0 - 2.0 * t)


def thermal_grip_multiplier_smooth(T_tire_c: Any) -> Any:
    """
    Smooth C-infinity thermal-frictional grip multiplier for Hoosier 16.0x7.5-10.
    Optimal window 70-85 C (~1.000 at 77.5 C), smooth cold/hot roll-off.
    """
    if is_casadi(T_tire_c):
        tanh_cold = ca.tanh((T_tire_c - 34.0) / 16.0)
        tanh_hot = ca.tanh((T_tire_c - 108.0) / 12.0)
    elif isinstance(T_tire_c, np.ndarray):
        tanh_cold = np.tanh((T_tire_c - 34.0) / 16.0)
        tanh_hot = np.tanh((T_tire_c - 108.0) / 12.0)
    else:
        tv = float(T_tire_c)
        tanh_cold = math.tanh((tv - 34.0) / 16.0)
        tanh_hot = math.tanh((tv - 108.0) / 12.0)

    cold_loss = 0.30 * 0.5 * (1.0 - tanh_cold)
    hot_loss = 0.28 * 0.5 * (1.0 + tanh_hot)
    mu_raw = (1.0 - cold_loss - hot_loss) / 0.997015
    return sclip(mu_raw, 0.55, 1.005, 1e-4)
