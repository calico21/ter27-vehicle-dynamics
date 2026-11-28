"""Pydantic v2 models for trackside setups matrix and live MirenaSim UDP apply."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from api.models.telemetry import SensorProfileName


class SetupValues(BaseModel):
    name: str
    event: str
    description: str
    spring_rate_front_n_m: float
    spring_rate_rear_n_m: float
    arb_front_nm_rad: float
    arb_rear_nm_rad: float
    lltd: float
    camber_front_deg: float
    camber_rear_deg: float
    toe_front_deg: float
    toe_rear_deg: float
    ride_height_front_m: float
    ride_height_rear_m: float
    rake_mm: float
    brake_bias: float
    tire_pressure_front_bar: float
    tire_pressure_rear_bar: float
    tv_map: str
    power_limit_kw: float
    control_law_mode: str


class SetupPresetDiff(BaseModel):
    preset: SetupValues
    deltas_vs_base: Dict[str, float]
    casadi_metrics: Dict[str, Any] = Field(default_factory=dict)


class EventOptimizationGain(BaseModel):
    event: str
    event_name: str
    metric_label: str
    base_time_s: float
    optimal_time_s: float
    delta_time_s: float
    delta_pct: float
    preset_file: str


class SetupsMatrixResponse(BaseModel):
    base_setup: SetupValues
    presets: List[SetupPresetDiff]
    optimization_gains: List[EventOptimizationGain]
    fs_points_base: Dict[str, float]
    fs_points_optimal: Dict[str, float]
    delta_total_pts: float


class ApplySetupRequest(BaseModel):
    preset: Optional[str] = Field(None, description="Preset name, e.g. optimal_skidpad")
    setup: Optional[Dict[str, Any]] = Field(None, description="Optional full setup dictionary override")
    sensor_profile: Optional[SensorProfileName] = Field(None, description="Optional sensor profile switch")
    vehicle_model: Optional[str] = Field(
        None,
        description="Optional vehicle model switch (full_14dof, fast_3dof, differentiable_casadi_14dof)",
    )
    suspension_mode: Optional[str] = Field(
        None,
        description="Optional orthogonal suspension mode (analytic_1d_lut, 3d_double_wishbone_lut, decoupled_heave_roll, standard_pushrod)",
    )


class ApplySetupResponse(BaseModel):
    acknowledged: bool
    udp_dispatched: bool
    mirena_sim_online: bool
    target_host: str
    target_port: int
    command_payload: Dict[str, Any]
    message: str
