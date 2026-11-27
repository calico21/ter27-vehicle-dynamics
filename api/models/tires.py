"""Pydantic v2 models for parametric Pacejka MF 6.2 tire evaluation."""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

TireCompoundName = Literal["R20", "R25B"]
GripProfileName = Literal["COLD_TEST_OCT", "FSG_RUBBERED", "FSG_REAL_TRACK", "TTC_LAB_PEAK"]


class TireEvaluateRequest(BaseModel):
    compound: TireCompoundName = Field("R25B", description="Hoosier 16.0x7.5-10 compound")
    grip_profile: GripProfileName = Field("FSG_RUBBERED", description="Track grip profile")
    fz_n: float = Field(750.0, ge=50.0, le=3500.0, description="Normal load Fz (N)")
    camber_deg: float = Field(-1.5, ge=-6.0, le=6.0, description="Camber angle (deg)")
    temp_c: float = Field(75.0, ge=5.0, le=120.0, description="Tire surface temperature (C)")
    pressure_kpa: float = Field(83.0, ge=55.0, le=130.0, description="Inflation pressure (kPa)")
    n_points: int = Field(61, ge=15, le=201, description="Sweep resolution")


class TireCurvePoint(BaseModel):
    x: float = Field(..., description="Sweep abscissa (alpha_deg or kappa)")
    y: float = Field(..., description="Evaluated force or moment (N or Nm)")


class FrictionEllipsePoint(BaseModel):
    angle_deg: float
    fx_n: float
    fy_n: float


class TireEvaluateResponse(BaseModel):
    compound: TireCompoundName
    grip_profile: GripProfileName
    fz_n: float
    camber_deg: float
    temp_c: float
    pressure_kpa: float
    lambda_mux_eff: float
    lambda_muy_eff: float
    thermal_mu_factor: float
    peak_fx_n: float
    peak_fy_n: float
    mu_x_eff: float
    mu_y_eff: float
    fy_vs_alpha: List[TireCurvePoint]
    fx_vs_kappa: List[TireCurvePoint]
    mz_vs_alpha: List[TireCurvePoint]
    friction_ellipse: List[FrictionEllipsePoint]
