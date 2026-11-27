"""Service for parametric Hoosier 16.0x7.5-10 Pacejka MF 6.2 tire evaluation."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from api.models.tires import (
    FrictionEllipsePoint,
    TireCurvePoint,
    TireEvaluateRequest,
    TireEvaluateResponse,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
    GRIP_PROFILES,
    PacejkaMF62,
    load_ter27_params,
)

_TIRE_CACHE: Dict[str, PacejkaMF62] = {}


def _get_tire(compound: str) -> PacejkaMF62:
    key = compound.upper()
    if key not in _TIRE_CACHE:
        _TIRE_CACHE[key] = PacejkaMF62(key)
    return _TIRE_CACHE[key]


def _thermal_mu_factor(temp_c: float, compound: str) -> float:
    """Smooth Gaussian tire thermal grip window matching MirenaSim WheelDynamics."""
    t_opt = 75.0 if compound.upper() == "R25B" else 65.0
    sigma_cold = 32.0
    sigma_hot = 28.0
    dt = temp_c - t_opt
    sigma = sigma_cold if dt < 0.0 else sigma_hot
    factor = 0.72 + 0.28 * math.exp(-0.5 * (dt / sigma) ** 2)
    return float(max(0.65, min(1.02, factor)))


def evaluate_tire_curves(req: TireEvaluateRequest) -> TireEvaluateResponse:
    params = load_ter27_params()
    tire = _get_tire(req.compound)
    profile_cfg = GRIP_PROFILES[req.grip_profile]
    env_mu_scale = float(profile_cfg["env_mu_scale"])

    therm_factor = _thermal_mu_factor(req.temp_c, req.compound)
    t_nom = float(
        profile_cfg.get("tire_temp_c", 75.0 if req.grip_profile == "FSG_RUBBERED" else 55.0)
    )
    therm_nom = _thermal_mu_factor(t_nom, req.compound)
    rel_therm = therm_factor / max(therm_nom, 1e-6)

    lmux_eff = params.lambda_mux_base * env_mu_scale * rel_therm
    lmuy_eff = params.lambda_muy_base * env_mu_scale * rel_therm

    n = req.n_points
    gamma_rad = math.radians(req.camber_deg)

    alpha_deg = np.linspace(-18.0, 18.0, n)
    alpha_rad = np.radians(alpha_deg)
    zeros = np.zeros_like(alpha_rad)
    fz_arr = np.full_like(alpha_rad, req.fz_n)
    gamma_arr = np.full_like(alpha_rad, gamma_rad)
    is_right = np.zeros(n, dtype=bool)

    _, fy_pure, mz_pure, _ = tire.compute_combined_forces_vec(
        kappa=zeros,
        alpha=alpha_rad,
        fz=fz_arr,
        gamma=gamma_arr,
        lmux=lmux_eff,
        lmuy=lmuy_eff,
        is_right=is_right,
        p_kpa=req.pressure_kpa,
    )

    kappa_sweep = np.linspace(-0.30, 0.30, n)
    fx_pure, _, _, _ = tire.compute_combined_forces_vec(
        kappa=kappa_sweep,
        alpha=zeros,
        fz=fz_arr,
        gamma=gamma_arr,
        lmux=lmux_eff,
        lmuy=lmuy_eff,
        is_right=is_right,
        p_kpa=req.pressure_kpa,
    )

    peak_fy = float(np.max(np.abs(fy_pure)))
    peak_fx = float(np.max(np.abs(fx_pure)))
    idx_alpha_peak = int(np.argmax(np.abs(fy_pure)))
    idx_kappa_peak = int(np.argmax(np.abs(fx_pure)))
    alpha_peak = abs(float(alpha_rad[idx_alpha_peak]))
    kappa_peak = abs(float(kappa_sweep[idx_kappa_peak]))

    n_ell = 73
    angles_deg = np.linspace(0.0, 360.0, n_ell)
    angles_rad = np.radians(angles_deg)
    kappa_ell = kappa_peak * np.cos(angles_rad)
    alpha_ell = alpha_peak * np.sin(angles_rad)
    fx_ell, fy_ell, _, _ = tire.compute_combined_forces_vec(
        kappa=kappa_ell,
        alpha=alpha_ell,
        fz=np.full(n_ell, req.fz_n),
        gamma=np.full(n_ell, gamma_rad),
        lmux=lmux_eff,
        lmuy=lmuy_eff,
        is_right=np.zeros(n_ell, dtype=bool),
        p_kpa=req.pressure_kpa,
    )

    return TireEvaluateResponse(
        compound=req.compound,
        grip_profile=req.grip_profile,
        fz_n=round(req.fz_n, 2),
        camber_deg=round(req.camber_deg, 3),
        temp_c=round(req.temp_c, 2),
        pressure_kpa=round(req.pressure_kpa, 2),
        lambda_mux_eff=round(lmux_eff, 5),
        lambda_muy_eff=round(lmuy_eff, 5),
        thermal_mu_factor=round(therm_factor, 4),
        peak_fx_n=round(peak_fx, 2),
        peak_fy_n=round(peak_fy, 2),
        mu_x_eff=round(peak_fx / max(req.fz_n, 1.0), 4),
        mu_y_eff=round(peak_fy / max(req.fz_n, 1.0), 4),
        fy_vs_alpha=[
            TireCurvePoint(x=round(float(a), 3), y=round(float(f), 2))
            for a, f in zip(alpha_deg, fy_pure)
        ],
        fx_vs_kappa=[
            TireCurvePoint(x=round(float(k), 4), y=round(float(f), 2))
            for k, f in zip(kappa_sweep, fx_pure)
        ],
        mz_vs_alpha=[
            TireCurvePoint(x=round(float(a), 3), y=round(float(m), 3))
            for a, m in zip(alpha_deg, mz_pure)
        ],
        friction_ellipse=[
            FrictionEllipsePoint(
                angle_deg=round(float(ang), 1),
                fx_n=round(float(fx_v), 2),
                fy_n=round(float(fy_v), 2),
            )
            for ang, fx_v, fy_v in zip(angles_deg, fx_ell, fy_ell)
        ],
    )
