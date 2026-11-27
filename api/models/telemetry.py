"""Pydantic v2 models for health, real-time 60 Hz telemetry, and recorded sessions."""

from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

TelemetrySource = Literal["godot_udp", "dil_session", "synthetic_fallback"]
SensorProfileName = Literal["SENSOR_IDEAL", "SENSOR_TRACK_REALISM"]
SessionKindName = Literal["dil_recording", "casadi_reference", "godot_parity"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = Field(..., description="API semantic version")
    udp_connected: bool = Field(..., description="True when live Godot UDP frames are being received")
    telemetry_source: TelemetrySource = Field(..., description="Active telemetry source for WS /ws/telemetry")
    active_ws_clients: int = Field(..., ge=0, description="Number of connected WebSocket clients")
    godot_udp_port: int = Field(9876, description="UDP port receiving MirenaSim telemetry")
    godot_cmd_port: int = Field(9877, description="UDP port sending commands to MirenaSim")
    grip_profiles: Dict[str, float] = Field(..., description="Standardized lambda_muy grip targets")


class VehicleState14Dof(BaseModel):
    vx: float = Field(..., description="Longitudinal velocity in ISO 8855 (m/s)")
    vy: float = Field(..., description="Lateral velocity in ISO 8855 (m/s)")
    vz: float = Field(..., description="Vertical velocity in ISO 8855 (m/s)")
    ax_g: float = Field(..., description="Longitudinal acceleration (g)")
    ay_g: float = Field(..., description="Lateral acceleration (g)")
    yaw_rate_rad_s: float = Field(..., description="Yaw rate omega_z (rad/s)")
    beta_deg: float = Field(..., description="Vehicle sideslip angle beta (deg)")
    steer_deg: float = Field(..., description="Front road-wheel steer angle (deg)")
    throttle: float = Field(..., ge=0.0, le=1.0, description="Normalized throttle command [0, 1]")
    brake: float = Field(..., ge=0.0, le=1.0, description="Normalized brake command [0, 1]")
    x_m: float = Field(..., description="Global X position in ISO 8855 (m)")
    y_m: float = Field(..., description="Global Y position in ISO 8855 (m)")
    z_m: float = Field(..., description="Global Z position in ISO 8855 (m)")
    heading_rad: float = Field(..., description="Global yaw heading (rad)")


class WheelTelemetry(BaseModel):
    fx_n: float = Field(..., description="Longitudinal tire force (N)")
    fy_n: float = Field(..., description="Lateral tire force (N)")
    fz_n: float = Field(..., description="Normal contact load (N)")
    slip_angle_deg: float = Field(..., description="Tire slip angle alpha (deg)")
    slip_ratio: float = Field(..., description="Longitudinal slip ratio kappa")
    tire_util: float = Field(..., ge=0.0, description="Combined friction circle utilization")
    camber_deg: float = Field(..., description="Effective inclination angle (deg)")
    wheel_travel_m: float = Field(..., description="Suspension compression travel (m)")
    motor_torque_nm: float = Field(..., description="Net wheel motor torque (Nm)")
    temp_c: float = Field(..., description="Tire surface temperature (C)")


class WheelsTelemetry(BaseModel):
    fl: WheelTelemetry
    fr: WheelTelemetry
    rl: WheelTelemetry
    rr: WheelTelemetry


class HvPowerTelemetry(BaseModel):
    p_dc_kw: float = Field(..., description="HV DC tractive discharge power (kW)")
    p_regen_kw: float = Field(..., description="HV DC regenerative charge power (kW)")
    soc_pct: float = Field(..., ge=0.0, le=100.0, description="Battery state of charge (%)")
    bspd_active: bool = Field(..., description="FSAE EV 2.3 BSPD plausibility flag")
    tv_mz_nm: float = Field(..., description="Torque vectoring yaw moment command (Nm)")


class EkfStateTelemetry(BaseModel):
    ekf_vx: float = Field(..., description="EKF estimated longitudinal velocity (m/s)")
    ekf_vy: float = Field(..., description="EKF estimated lateral velocity (m/s)")
    ekf_yaw_rate: float = Field(..., description="EKF estimated yaw rate (rad/s)")
    ekf_beta_deg: float = Field(..., description="EKF estimated sideslip angle (deg)")
    sensor_latency_ms: float = Field(..., ge=0.0, description="CAN transport latency (ms)")


class DilDeltaTelemetry(BaseModel):
    delta_time_s: float = Field(..., description="Live time delta vs CasADi reference (s)")
    delta_vx_mps: float = Field(..., description="Speed delta vs CasADi reference (m/s)")
    delta_ay_g: float = Field(..., description="Lateral accel delta vs CasADi reference (g)")


class TelemetryFrame(BaseModel):
    timestamp_s: float = Field(..., description="Simulation or stream timestamp (s)")
    source: TelemetrySource = Field(..., description="Telemetry origin")
    vehicle_id: str = Field(..., description="Active vehicle identifier")
    grip_profile: str = Field(..., description="Active track grip profile")
    sensor_profile: SensorProfileName = Field(..., description="Active sensor realism profile")
    state: VehicleState14Dof
    wheels: WheelsTelemetry
    hv: HvPowerTelemetry
    ekf: EkfStateTelemetry
    dil_delta: DilDeltaTelemetry


class SessionSummary(BaseModel):
    session_id: str
    title: str
    kind: SessionKindName
    track_or_event: str
    grip_profile: str
    n_samples: int
    duration_s: float
    source_path: str


class SessionsListResponse(BaseModel):
    sessions: List[SessionSummary]
    total_count: int


class TrajectorySample(BaseModel):
    t_s: float
    s_m: float
    x_m: float
    y_m: float
    vx_mps: float
    ay_g: float
    ax_g: float
    beta_deg: float
    steer_deg: float
    mz_tv_nm: float = 0.0
    delta_t_s: float = 0.0


class SessionABComparisonResponse(BaseModel):
    session: SessionSummary
    driver_trace: List[TrajectorySample]
    casadi_reference_trace: List[TrajectorySample]
    kpis: Dict[str, float]
