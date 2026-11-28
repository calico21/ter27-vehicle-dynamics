"""Pydantic v2 models for vehicle physical parameters and suspension kinematics."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VehicleModelSpec(BaseModel):
    id: str
    name: str
    dof: int
    recommended_hz: int
    differentiable: bool
    supports_codegen: bool
    description: str


class SuspensionModeSpec(BaseModel):
    id: str
    name: str
    description: str


class GripProfileSpec(BaseModel):
    name: str
    env_mu_scale: float
    env_aero_cl_scale: float
    lambda_muy_target: Optional[float] = None
    track_temp_c: Optional[float] = None
    tire_temp_c: Optional[float] = None


class CornerHardpoints3D(BaseModel):
    lca_front: List[float]
    lca_rear: List[float]
    lbj: List[float]
    uca_front: List[float]
    uca_rear: List[float]
    ubj: List[float]
    pushrod_wishbone: List[float]
    pushrod_rocker: List[float]
    tie_outer: List[float]
    tie_inner: List[float]
    wheel_center: List[float]
    contact_patch: List[float]
    damper_chassis: List[float]
    damper_rocker: List[float]
    rocker_axis_p1: List[float]
    rocker_pivot: List[float]
    arb_link_rocker: List[float]
    arb_arm_tip: List[float]
    arb_chassis_piv: List[float]
    spring_stiffness_n_m: float


class SuspensionOverview(BaseModel):
    use_3d_kinematics: bool
    frame: str
    spring_stiffness_front_n_m: float
    spring_stiffness_rear_n_m: float
    motion_ratio_front: float
    motion_ratio_rear: float
    arb_stiffness_front_nm_rad: float
    arb_stiffness_rear_nm_rad: float
    lltd_elastic_front: float
    static_camber_front_deg: float
    static_camber_rear_deg: float
    caster_deg: float
    kpi_deg: float
    anti_dive_front: float
    anti_squat_rear: float
    front_left_hardpoints: CornerHardpoints3D
    rear_left_hardpoints: CornerHardpoints3D


class VehiclePhysicalParams(BaseModel):
    mass_kg: float
    sprung_mass_kg: float
    unsprung_mass_total_kg: float
    wheelbase_m: float
    track_front_m: float
    track_rear_m: float
    cg_height_m: float
    front_weight_dist: float
    ixx_kgm2: float
    iyy_kgm2: float
    izz_total_kgm2: float
    cl_a_m2: float
    cd_a_m2: float
    aero_balance_front: float
    wheel_radius_m: float
    tire_compound: str
    p_dc_max_kw: float
    p_dc_regen_max_kw: float
    motor_peak_torque_nm: float
    gear_ratio: float
    battery_capacity_kwh: float
    battery_nominal_v: float


class ValidationStatusSummary(BaseModel):
    vault_suites: int
    total_figures: int
    cold_test_oct_lambda_muy: float
    fsg_rubbered_lambda_muy: float
    skidpad_reference_time_s: Optional[float] = None
    autocross_reference_time_s: Optional[float] = None
    endurance_22km_time_s: Optional[float] = None
    status: str


class VehicleOverviewResponse(BaseModel):
    vehicle_id: str
    display_name: str
    drive_type: str
    params: VehiclePhysicalParams
    suspension: SuspensionOverview
    grip_profiles: List[GripProfileSpec]
    setup_bounds: Dict[str, List[float]]
    validation_status: ValidationStatusSummary
    available_vehicle_models: List[VehicleModelSpec] = Field(default_factory=list)
    available_suspension_modes: List[SuspensionModeSpec] = Field(default_factory=list)
    active_vehicle_model: str = "full_14dof"
    active_suspension_mode: str = "3d_double_wishbone_lut"


class ModelSuspensionCombination(BaseModel):
    combination_id: str
    vehicle_model_id: str
    vehicle_model_name: str
    suspension_mode_id: str
    suspension_mode_name: str
    dof: int
    recommended_hz: int
    differentiable: bool
    supports_codegen: bool


class ModelsCatalogConstraints(BaseModel):
    p_dc_min_kw: float = -25.0
    p_dc_max_kw: float = 80.0
    bspd_ev23_max_kw: float = 5.0
    min_clear_m: float = 0.150
    yaw_stability_dmz_dbeta_max: float = 0.0
    cold_test_oct_lambda_muy: float = 0.47
    fsg_rubbered_lambda_muy: float = 0.57
    ter27_nominal_mass_kg: float = 280.0


class ModelsCatalogResponse(BaseModel):
    vehicle_id: str = "ter27_4wd"
    total_combinations: int
    vehicle_models: List[VehicleModelSpec]
    suspension_modes: List[SuspensionModeSpec]
    combinations: List[ModelSuspensionCombination]
    parameters: VehiclePhysicalParams
    constraints: ModelsCatalogConstraints
    setup_bounds: Dict[str, List[float]]
    grip_profiles: List[GripProfileSpec]
