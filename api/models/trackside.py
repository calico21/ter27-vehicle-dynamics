"""Pydantic v2 models for Trackside Race Engineering & What-If Simulation endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TracksideAnalyzeRequest(BaseModel):
    log_path: Optional[str] = Field(
        default=None,
        description="Path to CAN/CSV/ASC/BLF/Parquet telemetry log file (relative to repo root or absolute).",
    )
    trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline telemetry trace dictionary (vx_ms, ax_ms2, ay_ms2, wz_rads, steer_deg, throttle, brake).",
    )
    vehicle_id: Literal["ter27_4wd", "ter26_rwd"] = "ter27_4wd"
    event: Literal["skidpad", "accel", "autocross", "endurance"] = "autocross"
    fs_hz: float = 100.0
    apply_tire_temp_remap: Optional[bool] = None
    max_iters: int = Field(default=6, ge=1, le=20)


class PitBoxAdjustmentSchema(BaseModel):
    rank: int
    lever_id: str
    category: str
    action_label: str
    delta_steps: float
    step_unit: str
    delta_value_si: float
    theta_param: str
    predicted_delta_ms: float
    time_to_adjust_min: float
    is_pit_box_legal: bool = True
    rationale: str = ""


class SysIdSummarySchema(BaseModel):
    vehicle_id: str
    run_id: str
    wall_time_s: float
    within_30s_budget: bool
    n_iterations: int
    n_windows: int
    n_total_steps: int
    dt_sysid_s: float
    lambda_mu_hat: float
    sigma_lambda_mu: float
    cl_a_hat: float
    cd_a_hat: float
    cl_over_cd: float
    k_arb_f_hat: float
    k_arb_r_hat: float
    k_arb_ratio: float
    k_spring_f_hat: float
    k_spring_r_hat: float
    k_spring_ratio: float
    loss_initial: float
    loss_final: float
    loss_reduction_pct: float
    fit_metrics: Dict[str, Any] = Field(default_factory=dict)
    param_uncertainties: Dict[str, float] = Field(default_factory=dict)
    dT_dtheta: Dict[str, float] = Field(default_factory=dict)


class TracksideAnalyzeResponse(BaseModel):
    vehicle_id: str
    event: str
    run_id: str
    total_pipeline_time_s: float
    within_30s_budget: bool
    within_15min_budget: bool
    total_pit_time_min: float
    combined_delta_ms: float
    ingested_summary: Dict[str, Any]
    sysid: SysIdSummarySchema
    top_3_adjustments: List[PitBoxAdjustmentSchema]
    fsae_ev_invariants: Dict[str, Any]
    excluded_hardware_swaps: List[Dict[str, Any]] = Field(default_factory=list)


class WhatIfSimulateRequest(BaseModel):
    event: Literal["skidpad", "accel", "autocross", "endurance"] = "autocross"
    mass_kg: float = Field(default=280.0, ge=250.0, le=340.0)
    grip_lambda_mu: float = Field(default=0.57, ge=0.30, le=1.20)
    tire_temp_c: float = Field(default=75.0, ge=15.0, le=120.0)
    aero_cla_m2: float = Field(default=4.68, ge=0.0, le=7.0)
    suspension_mode: Literal[
        "analytic_1d_lut",
        "3d_double_wishbone_lut",
        "decoupled_heave_roll",
        "standard_pushrod",
    ] = "3d_double_wishbone_lut"
    model_id: Literal[
        "full_14dof",
        "fast_3dof",
        "differentiable_casadi_14dof",
    ] = "differentiable_casadi_14dof"
    tube_mpc_enabled: bool = True
    fault_mode: Literal[
        "nominal",
        "sensor_delay_noise",
        "grip_drop",
        "3wd_limp_braking",
        "3wd_limp_slalom",
    ] = "nominal"
    use_cached_matrix: bool = True


class WhatIfSimulateResponse(BaseModel):
    vehicle_id: str = "ter27_4wd"
    event: str
    source: Literal["cached_matrix", "casadi_14dof_adjoint_eval"]
    request_params: WhatIfSimulateRequest
    predicted_time_s: float
    baseline_280kg_time_s: float
    delta_vs_baseline_ms: float
    max_ay_g: float
    max_ax_g: float
    max_vx_mps: float
    p_dc_min_kw: float
    p_dc_max_kw: float
    min_clear_m: float
    cones_hit: int
    bspd_violations: int
    dmz_dbeta_nm_rad: float
    yaw_stable: bool
    invariants_passed: bool
    matched_scenario_id: Optional[str] = None
