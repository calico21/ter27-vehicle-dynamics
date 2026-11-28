"""Service for vehicle overview, trackside setups matrix, and live UDP setup apply."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from api.core.config import (
    SETUP_REPORT_PATH,
    SETUPS_DIR,
    UDP_COMMAND_HOST,
    UDP_COMMAND_PORT,
    VAULT_CATALOG_PATH,
    VEHICLE_CONFIG_DIR,
)
from api.models.setups import (
    ApplySetupRequest,
    ApplySetupResponse,
    EventOptimizationGain,
    SetupPresetDiff,
    SetupsMatrixResponse,
    SetupValues,
)
from api.models.vehicle import (
    CornerHardpoints3D,
    GripProfileSpec,
    ModelSuspensionCombination,
    ModelsCatalogConstraints,
    ModelsCatalogResponse,
    SuspensionModeSpec,
    SuspensionOverview,
    ValidationStatusSummary,
    VehicleModelSpec,
    VehicleOverviewResponse,
    VehiclePhysicalParams,
)
from api.services.udp_bridge import telemetry_bridge
from vd_correlation_lab.models.registry import (
    list_suspension_modes,
    list_vehicle_models,
    normalize_model_id,
    normalize_suspension_mode,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
    COLD_TEST_OCT_LAMBDA_MUY_TARGET,
    FSG_RUBBERED_LAMBDA_MUY_TARGET,
    GRIP_PROFILES,
    TRACKSIDE_SETUP_BOUNDS,
    load_ter27_params,
)

PRESET_FILENAMES = [
    "optimal_skidpad.json",
    "optimal_accel.json",
    "optimal_autocross.json",
    "optimal_endurance.json",
]

NUMERIC_DIFF_FIELDS = [
    "spring_rate_front_n_m",
    "spring_rate_rear_n_m",
    "arb_front_nm_rad",
    "arb_rear_nm_rad",
    "lltd",
    "camber_front_deg",
    "camber_rear_deg",
    "toe_front_deg",
    "toe_rear_deg",
    "ride_height_front_m",
    "ride_height_rear_m",
    "rake_mm",
    "brake_bias",
    "power_limit_kw",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_grip_profile_targets() -> Dict[str, float]:
    return {
        "COLD_TEST_OCT": COLD_TEST_OCT_LAMBDA_MUY_TARGET,
        "FSG_RUBBERED": FSG_RUBBERED_LAMBDA_MUY_TARGET,
    }


def build_vehicle_overview() -> VehicleOverviewResponse:
    p = load_ter27_params(VEHICLE_CONFIG_DIR)
    veh = _read_json(VEHICLE_CONFIG_DIR / "vehicle.json")
    pt = _read_json(VEHICLE_CONFIG_DIR / "powertrain.json")
    susp = _read_json(VEHICLE_CONFIG_DIR / "suspension.json")
    dwk = susp.get("double_wishbone_kinematics", {})

    catalog = _read_json(VAULT_CATALOG_PATH)
    setup_report = _read_json(SETUP_REPORT_PATH)
    events = setup_report.get("events", {})

    grip_specs: List[GripProfileSpec] = []
    for name, spec in GRIP_PROFILES.items():
        grip_specs.append(
            GripProfileSpec(
                name=name,
                env_mu_scale=float(spec["env_mu_scale"]),
                env_aero_cl_scale=float(spec["env_aero_cl_scale"]),
                lambda_muy_target=spec.get("lambda_muy_target"),
                track_temp_c=spec.get("track_temp_c"),
                tire_temp_c=spec.get("tire_temp_c"),
            )
        )

    bounds_out: Dict[str, List[float]] = {
        k: [float(v[0]), float(v[1])] for k, v in TRACKSIDE_SETUP_BOUNDS.items()
    }

    summary = catalog.get("summary", {})
    val_status = ValidationStatusSummary(
        vault_suites=int(summary.get("n_suites", 10)),
        total_figures=int(summary.get("total_pngs", 67)),
        cold_test_oct_lambda_muy=COLD_TEST_OCT_LAMBDA_MUY_TARGET,
        fsg_rubbered_lambda_muy=FSG_RUBBERED_LAMBDA_MUY_TARGET,
        skidpad_reference_time_s=events.get("skidpad", {}).get("optimal_time_s"),
        autocross_reference_time_s=events.get("autocross", {}).get("optimal_time_s"),
        endurance_22km_time_s=events.get("endurance", {}).get("optimal_time_s"),
        status="validated",
    )

    return VehicleOverviewResponse(
        vehicle_id=str(veh.get("vehicle_id", "ter27_4wd")),
        display_name=str(veh.get("display_name", "TeR27 4WD")),
        drive_type=str(pt.get("drive_type", "4WD")),
        params=VehiclePhysicalParams(
            mass_kg=p.mass,
            sprung_mass_kg=p.sprung_mass,
            unsprung_mass_total_kg=p.unsprung_mass_total,
            wheelbase_m=p.wheelbase,
            track_front_m=p.track_front,
            track_rear_m=p.track_rear,
            cg_height_m=p.h_cog,
            front_weight_dist=p.front_weight_dist,
            ixx_kgm2=p.ixx,
            iyy_kgm2=p.iyy,
            izz_total_kgm2=round(p.izz_total, 4),
            cl_a_m2=p.cl_a,
            cd_a_m2=p.cd_a,
            aero_balance_front=p.aero_balance_front,
            wheel_radius_m=p.wheel_radius,
            tire_compound=p.tire_compound,
            p_dc_max_kw=round(p.p_dc_max_w * 0.001, 3),
            p_dc_regen_max_kw=round(abs(p.p_dc_min_w) * 0.001, 3),
            motor_peak_torque_nm=p.motor_peak_torque,
            gear_ratio=p.gear_ratio,
            battery_capacity_kwh=float(pt.get("battery_capacity_kwh", 7.2)),
            battery_nominal_v=p.v_nom,
        ),
        suspension=SuspensionOverview(
            use_3d_kinematics=bool(dwk.get("use_3d_kinematics", True)),
            frame=str(dwk.get("frame", "iso_ros_left_from_front_axle_ground_m")),
            spring_stiffness_front_n_m=p.k_spring_f,
            spring_stiffness_rear_n_m=p.k_spring_r,
            motion_ratio_front=p.mr_f,
            motion_ratio_rear=p.mr_r,
            arb_stiffness_front_nm_rad=p.k_arb_f,
            arb_stiffness_rear_nm_rad=p.k_arb_r,
            lltd_elastic_front=round(p.lltd_elastic_f, 4),
            static_camber_front_deg=round(math.degrees(p.static_camber_f_rad), 3),
            static_camber_rear_deg=round(math.degrees(p.static_camber_r_rad), 3),
            caster_deg=round(math.degrees(p.caster_rad), 3),
            kpi_deg=round(math.degrees(p.kpi_rad), 3),
            anti_dive_front=p.anti_dive_f,
            anti_squat_rear=p.anti_squat_r,
            front_left_hardpoints=CornerHardpoints3D.model_validate(dwk["front_left"]),
            rear_left_hardpoints=CornerHardpoints3D.model_validate(dwk["rear_left"]),
        ),
        grip_profiles=grip_specs,
        setup_bounds=bounds_out,
        validation_status=val_status,
        available_vehicle_models=[
            VehicleModelSpec.model_validate(m) for m in list_vehicle_models()
        ],
        available_suspension_modes=[
            SuspensionModeSpec.model_validate(s) for s in list_suspension_modes()
        ],
        active_vehicle_model="full_14dof",
        active_suspension_mode=(
            "3d_double_wishbone_lut"
            if bool(dwk.get("use_3d_kinematics", True))
            else "analytic_1d_lut"
        ),
    )


def _parse_setup(raw: Dict[str, Any], default_lltd: float) -> SetupValues:
    rh_f = float(raw.get("ride_height_front_m", 0.035))
    rh_r = float(raw.get("ride_height_rear_m", 0.035))
    rake_mm = float(raw.get("rake_mm", (rh_r - rh_f) * 1000.0))
    return SetupValues(
        name=str(raw.get("name", "unknown")),
        event=str(raw.get("event", "base")),
        description=str(raw.get("description", "")),
        spring_rate_front_n_m=float(raw.get("spring_rate_front_n_m", 35000.0)),
        spring_rate_rear_n_m=float(raw.get("spring_rate_rear_n_m", 48000.0)),
        arb_front_nm_rad=float(raw.get("arb_front_nm_rad", 4500.0)),
        arb_rear_nm_rad=float(raw.get("arb_rear_nm_rad", 5500.0)),
        lltd=round(float(raw.get("lltd", default_lltd)), 4),
        camber_front_deg=float(raw.get("camber_front_deg", -2.0)),
        camber_rear_deg=float(raw.get("camber_rear_deg", -1.5)),
        toe_front_deg=float(raw.get("toe_front_deg", 0.0)),
        toe_rear_deg=float(raw.get("toe_rear_deg", 0.0)),
        ride_height_front_m=rh_f,
        ride_height_rear_m=rh_r,
        rake_mm=round(rake_mm, 3),
        brake_bias=float(raw.get("brake_bias", 0.58)),
        tire_pressure_front_bar=float(raw.get("tire_pressure_front_bar", 0.72)),
        tire_pressure_rear_bar=float(raw.get("tire_pressure_rear_bar", 0.72)),
        tv_map=str(raw.get("tv_map", "05_smc_pivot_4wd.json")),
        power_limit_kw=float(raw.get("power_limit_kw", 80.0)),
        control_law_mode=str(raw.get("control_law_mode", "smc")),
    )


def build_setups_matrix() -> SetupsMatrixResponse:
    params = load_ter27_params(VEHICLE_CONFIG_DIR)
    base_raw = json.loads((SETUPS_DIR / "default.json").read_text(encoding="utf-8"))
    base_setup = _parse_setup(base_raw, params.lltd_elastic_f)
    base_dict = base_setup.model_dump()

    presets: List[SetupPresetDiff] = []
    for fname in PRESET_FILENAMES:
        fpath: Path = SETUPS_DIR / fname
        if not fpath.exists():
            continue
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        parsed = _parse_setup(raw, params.lltd_elastic_f)
        p_dict = parsed.model_dump()
        deltas: Dict[str, float] = {}
        for field in NUMERIC_DIFF_FIELDS:
            deltas[field] = round(float(p_dict[field]) - float(base_dict[field]), 5)
        presets.append(
            SetupPresetDiff(
                preset=parsed,
                deltas_vs_base=deltas,
                casadi_metrics=dict(raw.get("casadi_metrics", {})),
            )
        )

    gains: List[EventOptimizationGain] = []
    fs_base: Dict[str, float] = {}
    fs_opt: Dict[str, float] = {}
    delta_pts = 0.0
    if SETUP_REPORT_PATH.exists():
        rep = json.loads(SETUP_REPORT_PATH.read_text(encoding="utf-8"))
        for ev_key, ev_data in rep.get("events", {}).items():
            gains.append(
                EventOptimizationGain(
                    event=str(ev_key),
                    event_name=str(ev_data.get("event_name", ev_key)),
                    metric_label=str(ev_data.get("metric_label", "Time (s)")),
                    base_time_s=float(ev_data.get("base_time_s", 0.0)),
                    optimal_time_s=float(ev_data.get("optimal_time_s", 0.0)),
                    delta_time_s=float(ev_data.get("delta_time_s", 0.0)),
                    delta_pct=float(ev_data.get("delta_pct", 0.0)),
                    preset_file=str(ev_data.get("preset_file", "")),
                )
            )
        fs_pts = rep.get("fs_points", {})
        fs_base = {k: float(v) for k, v in fs_pts.get("base", {}).items()}
        fs_opt = {k: float(v) for k, v in fs_pts.get("optimal", {}).items()}
        delta_pts = float(fs_pts.get("delta_total_pts", 0.0))

    return SetupsMatrixResponse(
        base_setup=base_setup,
        presets=presets,
        optimization_gains=gains,
        fs_points_base=fs_base,
        fs_points_optimal=fs_opt,
        delta_total_pts=delta_pts,
    )


def apply_trackside_setup(req: ApplySetupRequest) -> ApplySetupResponse:
    cmd_payload: Dict[str, Any] = {"cmd": "apply_setup"}
    if req.preset:
        cmd_payload["preset"] = req.preset
    if req.setup:
        cmd_payload["setup"] = req.setup
    if req.vehicle_model:
        cmd_payload["vehicle_model"] = normalize_model_id(req.vehicle_model)
    if req.suspension_mode:
        cmd_payload["suspension_mode"] = normalize_suspension_mode(req.suspension_mode)
    if not req.preset and not req.setup:
        cmd_payload["preset"] = "default"

    ok, msg = telemetry_bridge.send_godot_command(cmd_payload)
    if req.sensor_profile:
        telemetry_bridge.send_godot_command(
            {"cmd": "set_sensor_profile", "profile": req.sensor_profile}
        )

    online = telemetry_bridge.is_udp_connected
    if not online:
        msg = f"{msg} (MirenaSim offline: command acknowledged and queued as best-effort UDP)"

    return ApplySetupResponse(
        acknowledged=True,
        udp_dispatched=ok,
        mirena_sim_online=online,
        target_host=UDP_COMMAND_HOST,
        target_port=UDP_COMMAND_PORT,
        command_payload=cmd_payload,
        message=msg,
    )


def build_models_catalog() -> ModelsCatalogResponse:
    """Return full catalog of 3 vehicle models x 4 suspension modes, parameters, and constraints."""
    overview = build_vehicle_overview()
    combinations: List[ModelSuspensionCombination] = []
    for vm in overview.available_vehicle_models:
        for sm in overview.available_suspension_modes:
            combinations.append(
                ModelSuspensionCombination(
                    combination_id=f"{vm.id}__{sm.id}",
                    vehicle_model_id=vm.id,
                    vehicle_model_name=vm.name,
                    suspension_mode_id=sm.id,
                    suspension_mode_name=sm.name,
                    dof=vm.dof,
                    recommended_hz=vm.recommended_hz,
                    differentiable=vm.differentiable,
                    supports_codegen=vm.supports_codegen,
                )
            )
    return ModelsCatalogResponse(
        vehicle_id=overview.vehicle_id,
        total_combinations=len(combinations),
        vehicle_models=overview.available_vehicle_models,
        suspension_modes=overview.available_suspension_modes,
        combinations=combinations,
        parameters=overview.params,
        constraints=ModelsCatalogConstraints(),
        setup_bounds=overview.setup_bounds,
        grip_profiles=overview.grip_profiles,
    )
