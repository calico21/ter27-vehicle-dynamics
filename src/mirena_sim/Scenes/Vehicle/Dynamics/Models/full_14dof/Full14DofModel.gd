class_name VehicleDynamicsModel
extends RefCounted

## vehicle dynamics model (10-dof core) for mirena dynamics
## fully decoupled coordinator for physical & mathematical calculations
## internal coordinate convention: iso 8855 / ros rep-103
## +x: forward, +y: left, +z: up

const VehicleConfig = preload("../../../Config/VehicleConfig.gd")
const VehicleInputs = preload("../../../Inputs/VehicleInputs.gd")
const SetupProfileScript = preload("../../../Config/SetupProfile.gd")
const WheelDynamics = preload("../../Tires/WheelDynamics.gd")
const DrivetrainModel = preload("../../Drivetrain/DrivetrainModel.gd")
const TorqueVectoring = preload("../../Drivetrain/TorqueVectoring.gd")
const EcuTorqueVectoringV3 = preload("../../Drivetrain/EcuTorqueVectoringV3.gd")
const EcuTorqueVectoring4Wd = preload("../../Drivetrain/EcuTorqueVectoring4Wd.gd")
const SuspensionModel = preload("../../Suspension/SuspensionModel.gd")
const SuspensionCorner = preload("../../Suspension/SuspensionCorner.gd")
const SuspensionSystem = preload("../../Suspension/SuspensionSystem.gd")
const AeroModel = preload("../../Aerodynamics/AeroModel.gd")
const ChassisDynamics = preload("../../Chassis/ChassisDynamics.gd")
const BrakeModel = preload("../../Brakes/BrakeModel.gd")
const RoadSurfaceScript = preload("RoadSurface.gd")
const SensorRealismFilterScript = preload("../../Telemetry/SensorRealismFilter.gd")

enum TrackGripProfile { TTC_LAB_PEAK = 0, FSG_REAL_TRACK = 1, FSG_RUBBERED = 2 }

var track_grip_profile: int = TrackGripProfile.TTC_LAB_PEAK
var env_mu_scale: float = 1.0
var env_aero_cl_scale: float = 1.0

var config: VehicleConfig
var wheels: Array[WheelDynamics] = []
var suspension: SuspensionSystem
var torque_vectoring: TorqueVectoring
var ecu_tv_v3: EcuTorqueVectoringV3
var ecu_tv_4wd: EcuTorqueVectoring4Wd
var sensor_realism: SensorRealismFilterScript
var hv_state: DrivetrainModel.HVPowertrainState
var brake_state: BrakeModel.BrakeState
var road_surface: RoadSurfaceScript
var _prev_ax_ros: float = 0.0
var chassis_state: ChassisDynamics.State6DOF
var inputs: VehicleInputs
var _last_input_tv_map_id: int = 2

# output net quantities (in iso/ros chassis frame around cog)
var net_force_chassis_ros: Vector3 = Vector3.ZERO
var net_torque_chassis_ros: Vector3 = Vector3.ZERO

# output net quantities (converted to godot frame for rigidbody3d)
var net_force_chassis_godot: Vector3 = Vector3.ZERO
var net_torque_chassis_godot: Vector3 = Vector3.ZERO

# chassis states (in iso/ros chassis frame)
var linear_velocity_ros: Vector3 = Vector3.ZERO
var angular_velocity_ros: Vector3 = Vector3.ZERO
var _prev_fy_front: float = 0.0
var _prev_fy_rear: float = 0.0
var _prev_fx_front: float = 0.0
var _prev_fx_rear: float = 0.0
var _prev_fx_corners: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _prev_fy_corners: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _prev_fz_corners: Array[float] = [640.0, 640.0, 780.0, 780.0]
var _prev_ay_ros: float = 0.0
var _prev_az_ros: float = 9.81

# preallocated scratch buffers (zero heap allocations inside step_forces)
var _dz_scratch: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _dz_dot_scratch: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _dz_zero_scratch: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _g_scratch: Array[bool] = [true, true, true, true]
var _w_steers_scratch: Array[float] = [0.0, 0.0, 0.0, 0.0]

# salida fisica de force feedback (ffb) en columna de direccion
var steering_feedback_torque_nm: float = 0.0
var steering_feedback_normalized: float = 0.0
var steering_torque_nm: float = 0.0
var kp_moment_fl_nm: float = 0.0
var kp_moment_fr_nm: float = 0.0
var _prev_steer_angle_ffb: float = 0.0
var _ffb_norm_filt: float = 0.0

# descomposicion por rueda para visualizacion 3d DiL (Nm positivos en rueda)
var wheel_drive_torque_nm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_regen_brake_nm: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_mech_brake_nm: Array[float] = [0.0, 0.0, 0.0, 0.0]

func _init(cfg: VehicleConfig = null) -> void:
	if cfg != null:
		config = cfg
	else:
		config = VehicleConfig.new()
	if config.tire_compound != "" and TireModel.get_compound() != config.tire_compound:
		TireModel.set_compound(config.tire_compound)
	
	inputs = VehicleInputs.new()
	inputs.drive_mode = config.drive_mode
	_last_input_tv_map_id = inputs.tv_map_id
	suspension = SuspensionSystem.new()
	suspension.sync_from_config(config)
	torque_vectoring = TorqueVectoring.new()
	ecu_tv_v3 = EcuTorqueVectoringV3.new()
	ecu_tv_v3.configure_from_vehicle_config(config)
	ecu_tv_4wd = EcuTorqueVectoring4Wd.new()
	ecu_tv_4wd.configure_from_vehicle_config(config)
	sensor_realism = SensorRealismFilterScript.new()
	sensor_realism.load_from_tv_config_json(SensorRealismFilterScript.DEFAULT_TV_CFG_PATH, true)
	_sync_sensor_realism_vehicle_params()
	hv_state = DrivetrainModel.HVPowertrainState.new(config)
	brake_state = BrakeModel.BrakeState.new(config.disc_ambient_temp_c)
	chassis_state = ChassisDynamics.State6DOF.new(config.h_cog)
	road_surface = RoadSurfaceScript.new()
	_init_wheels()

func _init_wheels() -> void:
	wheels.clear()
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.FL, "FL", false))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.FR, "FR", true))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.RL, "RL", false))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.RR, "RR", true))
	_sync_wheel_thermal_config()

func _sync_wheel_thermal_config() -> void:
	if config == null:
		return
	var init_t: float = config.tire_initial_temp_c
	var track_t: float = config.track_temp_c
	var p_front_kpa: float = config.tire_pressure_front_bar * 100.0
	var p_rear_kpa: float = config.tire_pressure_rear_bar * 100.0
	for i in range(wheels.size()):
		var w = wheels[i]
		w.set_tire_temperature(init_t)
		w.t_track_c = track_t
		w.inflation_pressure_kpa = p_front_kpa if i < 2 else p_rear_kpa


## recarga en caliente la config completa del coche (e.g. "ter26_rwd", "ter27_4wd") and optional set-up profile
func load_vehicle(vehicle_id: String, setup_name: String = "default") -> bool:
	var new_cfg := VehicleConfig.new()
	if not new_cfg.load_vehicle_configs(vehicle_id):
		return false
	config = new_cfg
	if config.tire_compound != "" and TireModel.get_compound() != config.tire_compound:
		TireModel.set_compound(config.tire_compound)
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
	_sync_sensor_realism_vehicle_params()
	if ecu_tv_v3 != null:
		ecu_tv_v3.configure_from_vehicle_config(config)
		var v_maps := ecu_tv_v3.list_available_tv_maps()
		if not v_maps.is_empty():
			ecu_tv_v3.load_tv_map_from_file(String(v_maps[0]["path"]))
	reset()
	if setup_name.strip_edges() != "":
		apply_setup(setup_name)
	return true


func set_track_grip_profile(profile_id: int) -> void:
	if profile_id == TrackGripProfile.FSG_REAL_TRACK:
		track_grip_profile = TrackGripProfile.FSG_REAL_TRACK
		env_mu_scale = 0.86
		env_aero_cl_scale = 0.64
	elif profile_id == TrackGripProfile.FSG_RUBBERED:
		# FSG_RUBBERED: lambda_muy = 0.57 (R25B @ 75 C, proportional R20); COLD_TEST_OCT stays 0.47
		track_grip_profile = TrackGripProfile.FSG_RUBBERED
		env_mu_scale = 0.57 / 0.47
		env_aero_cl_scale = 0.78
		for w in wheels:
			w.t_track_c = 45.0
			w.set_tire_temperature(75.0)
	else:
		track_grip_profile = TrackGripProfile.TTC_LAB_PEAK
		env_mu_scale = 1.00
		env_aero_cl_scale = 1.00


func _sync_sensor_realism_vehicle_params() -> void:
	if sensor_realism == null or config == null:
		return
	var wb_safe: float = maxf(config.wheelbase, 0.1)
	var fwd_dist: float = clampf(config.front_weight_distribution, 0.2, 0.8)
	sensor_realism.configure({
		"mass_kg": config.mass,
		"wheelbase_m": wb_safe,
		"front_weight_dist": fwd_dist,
		"track_front_m": maxf(config.track_width_f, 0.5),
		"track_rear_m": maxf(config.track_width_r, 0.5),
		"h_cog_m": maxf(config.h_cog, 0.05),
		"wheel_radius_m": maxf(config.wheel_radius, 0.05),
		"gravity_ms2": maxf(config.gravity, 1.0)
	})


func set_sensor_realism_mode(mode_int: int) -> void:
	if sensor_realism != null:
		sensor_realism.set_mode(mode_int)


func set_sensor_can_latency_ms(ms: float) -> void:
	if sensor_realism != null:
		sensor_realism.set_can_latency_ms(ms)


func set_sensor_imu_noise_scale(scale: float) -> void:
	if sensor_realism != null:
		sensor_realism.set_imu_noise_scale(scale)


func reset_sensor_realism(seed: int = -1) -> void:
	if sensor_realism != null:
		sensor_realism.reset(seed)


func get_sensor_realism_diagnostics() -> Dictionary:
	if sensor_realism == null:
		return {
			"mode": 0,
			"mode_name": "SENSOR_IDEAL",
			"can_accel_latency_ms": 0.0,
			"can_steer_latency_ms": 0.0,
			"imu_noise_scale": 0.0,
			"beta_hat": 0.0,
			"fz_hat_4": _prev_fz_corners.duplicate()
		}
	return sensor_realism.get_diagnostics()


## carga y aplica un setupprofile for the active vehicle
func apply_setup(setup_name: String) -> bool:
	if config == null:
		return false
	var prof := SetupProfileScript.load_for_vehicle(config.vehicle_id, setup_name)
	if prof == null:
		return false
	return apply_setup_profile(prof)


## aplica un setupprofile to config, susp, tires, power limit, and ecu/tv
func apply_setup_profile(prof: SetupProfileScript) -> bool:
	if prof == null or config == null:
		return false
	prof.apply_to_config(config)
	if suspension != null:
		suspension.sync_from_config(config)
	for i in range(wheels.size()):
		wheels[i].inflation_pressure_kpa = (config.tire_pressure_front_bar if i < 2 else config.tire_pressure_rear_bar) * 100.0
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
	_sync_sensor_realism_vehicle_params()
	if ecu_tv_v3 != null:
		ecu_tv_v3.configure_from_vehicle_config(config)
		if prof.tv_map.strip_edges() != "":
			ecu_tv_v3.load_tv_map_by_filename(prof.tv_map)
		if prof.control_law_mode.strip_edges() != "":
			var law := prof.control_law_mode.strip_edges().to_lower()
			ecu_tv_v3.control_law_mode = (
				EcuTorqueVectoringV3.ControlLawMode.V4_SMC_HYBRID
				if (law == "smc" or law == "v4" or law == "v4_smc_hybrid")
				else EcuTorqueVectoringV3.ControlLawMode.V3_PID_ALQP
			)
		if not prof.tv_overrides.is_empty() and ecu_tv_v3.has_method("set_live_params"):
			ecu_tv_v3.set_live_params(prof.tv_overrides)
	inputs.drive_mode = config.drive_mode
	return true

## saca angulo de direccion del (iso +z ccw left) with configurable ackermann geometry
static func compute_ackermann_steer(steer_angle: float, is_left: bool, config: VehicleConfig) -> float:
	return SuspensionSystem.compute_ackermann_steer(steer_angle, is_left, config)

## devuelve pos de rueda respecto al cdg in iso/ros frame:
func get_wheel_pos_ros(index: int) -> Vector3:
	return ChassisDynamics.get_wheel_pos_ros(index, config)

## saca pares motor 4wd y freno mecanico per wheel
func compute_drivetrain_torques(
	in_inputs: VehicleInputs,
	u_speed: float,
	dt: float = 0.0,
	advance_tv: bool = false
) -> Dictionary:
	var wheel_omegas: Array[float] = [
		wheels[0].omega,
		wheels[1].omega,
		wheels[2].omega,
		wheels[3].omega
	]
	var v_dc_eff: float = hv_state.v_dc if hv_state != null else -1.0
	return DrivetrainModel.compute_torques(in_inputs, u_speed, config, wheel_omegas, torque_vectoring, dt, advance_tv, v_dc_eff)

## computes drag y downforce aero forces in iso/ros frame
func compute_aerodynamics(v_ros: Vector3, z_front: float = -1.0, z_rear: float = -1.0) -> Dictionary:
	return AeroModel.compute_forces_ros(v_ros, config, z_front, z_rear)

## transf de carga analitica for unit testing and offline simulation:
func compute_analytic_normal_forces(ax_ros: float, ay_ros: float, speed: float) -> Array[float]:
	return SuspensionModel.compute_analytic_normal_forces(ax_ros, ay_ros, speed, config)

## wrapper compatible pa step_forces() pa llamadas externas y tests
func step(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	ground_contacts: Array = [],
	dt: float = 0.0025
) -> Dictionary:
	return step_forces(in_inputs, local_v_ros, local_w_ros, ground_contacts, dt, true)

## ruta comun de fuerzas, susp, deslizamiento, neuma y aero usada x ambos
## mirenacardynamics (godot rigidbody3d) and step_standalone (14-dof headless)
func step_forces(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	ground_contacts: Array = [],
	dt: float = 0.0025,
	advance_tv: bool = true
) -> Dictionary:
	linear_velocity_ros = local_v_ros
	angular_velocity_ros = local_w_ros
	inputs = in_inputs
	
	var u = local_v_ros.x
	var v = local_v_ros.y
	var r = local_w_ros.z
	
	# 1. transmision, tv 200 hz, bus hv y lag de inversores
	if ecu_tv_v3 != null and in_inputs.tv_map_id != _last_input_tv_map_id:
		if in_inputs.tv_map_id != ecu_tv_v3.active_map_id:
			ecu_tv_v3.set_map(in_inputs.tv_map_id)
		_last_input_tv_map_id = ecu_tv_v3.active_map_id

	var use_3d: bool = suspension != null and suspension.is_3d_active(config)
	var dz_fl_hint: float = float(ground_contacts[0].get("delta_z", 0.0)) if (ground_contacts.size() >= 2 and ground_contacts[0] is Dictionary) else 0.0
	var dz_fr_hint: float = float(ground_contacts[1].get("delta_z", 0.0)) if (ground_contacts.size() >= 2 and ground_contacts[1] is Dictionary) else 0.0
	if use_3d:
		_w_steers_scratch[0] = suspension.get_ackermann_steer_3d(in_inputs.steer_angle, dz_fl_hint, true, config)
		_w_steers_scratch[1] = suspension.get_ackermann_steer_3d(in_inputs.steer_angle, dz_fr_hint, false, config)
	else:
		_w_steers_scratch[0] = compute_ackermann_steer(in_inputs.steer_angle, true, config)
		_w_steers_scratch[1] = compute_ackermann_steer(in_inputs.steer_angle, false, config)
	_w_steers_scratch[2] = 0.0
	_w_steers_scratch[3] = 0.0

	var w_omegas: Array[float] = [wheels[0].omega, wheels[1].omega, wheels[2].omega, wheels[3].omega]
	var sr_active: bool = sensor_realism != null and sensor_realism.mode == SensorRealismFilterScript.SensorMode.SENSOR_TRACK_REALISM
	var sr_steer: float = in_inputs.steer_angle
	var sr_u: float = u
	var sr_v: float = v
	var sr_r: float = r
	var sr_ax: float = _prev_ax_ros
	var sr_ay: float = _prev_ay_ros
	var sr_omegas: Array[float] = w_omegas
	var sr_beta_hat: float = atan2(v, maxf(absf(u), 0.5)) if absf(u) > 0.1 else 0.0
	var sr_fz_hat_4: Array[float] = _prev_fz_corners.duplicate()
	if sr_active:
		var sr_out: Dictionary = (
			sensor_realism.step(_prev_ax_ros, _prev_ay_ros, r, u, v, in_inputs.steer_angle, _prev_fz_corners, w_omegas, dt)
			if advance_tv or sensor_realism.get_last_output().is_empty()
			else sensor_realism.get_last_output()
		)
		sr_ax = float(sr_out.get("ax", _prev_ax_ros))
		sr_ay = float(sr_out.get("ay", _prev_ay_ros))
		sr_r = float(sr_out.get("wz", r))
		sr_u = float(sr_out.get("vx", u))
		sr_v = float(sr_out.get("vy", v))
		sr_steer = float(sr_out.get("delta", in_inputs.steer_angle))
		var om_raw = sr_out.get("omega_4", w_omegas)
		if om_raw is Array and om_raw.size() >= 4:
			sr_omegas = [float(om_raw[0]), float(om_raw[1]), float(om_raw[2]), float(om_raw[3])]
		sr_beta_hat = float(sr_out.get("beta_hat", sr_beta_hat))
		var fz_raw = sr_out.get("fz_hat_4", sr_fz_hat_4)
		if fz_raw is Array and fz_raw.size() >= 4:
			sr_fz_hat_4 = [float(fz_raw[0]), float(fz_raw[1]), float(fz_raw[2]), float(fz_raw[3])]
	var drive_trqs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var brake_trqs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var ecu_supports_mode: bool = (in_inputs.drive_mode == VehicleConfig.DriveMode.FOUR_WD or in_inputs.drive_mode == VehicleConfig.DriveMode.REAR_WD)
	if in_inputs.ecu_driving_mode == 2 and in_inputs.drive_mode == VehicleConfig.DriveMode.FOUR_WD and not in_inputs.override_wheel_torques and ecu_tv_4wd != null:
		ecu_tv_4wd.active_mode = in_inputs.tv_4wd_mode
		var r_whl4: float = maxf(config.wheel_radius, 1e-4)
		var max_whl_trq4: float = config.motor_peak_torque * config.gear_ratio
		var max_pwr_f4: float = (config.power_limit_watts * config.drivetrain_efficiency) / maxf(absf(u), 1.0)
		var fx_drv4: float = in_inputs.throttle * minf(4.0 * max_whl_trq4 / r_whl4, max_pwr_f4)
		if in_inputs.brake > 0.05 and in_inputs.throttle < 0.05:
			var p_chg: float = maxf(config.p_max_charge_w, 1000.0)
			if hv_state != null and hv_state.soc > 0.985:
				p_chg *= clampf((1.0 - hv_state.soc) / 0.015, 0.0, 1.0)
			var p_chg_mech: float = p_chg / maxf(ecu_tv_4wd.eta_regen, 0.5)
			var f_reg_cap: float = minf(4.0 * ecu_tv_4wd.t_regen_max / r_whl4, p_chg_mech / maxf(absf(u), 2.0))
			if ecu_tv_4wd.mech_brake_start > 0.05:
				var reg_frac: float = clampf(in_inputs.brake / ecu_tv_4wd.mech_brake_start, 0.0, 1.0)
				fx_drv4 = -reg_frac * f_reg_cap if absf(u) > 1.0 else 0.0
			else:
				var f_brk_dem: float = in_inputs.brake * config.brake_mu * config.mass * config.gravity
				fx_drv4 = -minf(f_brk_dem, f_reg_cap) if absf(u) > 2.0 else 0.0
		if sr_active and advance_tv:
			for i in range(4):
				ecu_tv_4wd.fz_est[i] = sr_fz_hat_4[i]
		drive_trqs = (
			ecu_tv_4wd.step(fx_drv4, sr_steer, sr_u, sr_r, sr_ax, sr_ay, _prev_az_ros, sr_omegas, in_inputs.brake, dt, _w_steers_scratch)
			if advance_tv else ecu_tv_4wd.last_t_cmd.duplicate()
		)
	elif in_inputs.ecu_driving_mode == 1 and ecu_supports_mode and not in_inputs.override_wheel_torques:
		if ecu_tv_v3 != null:
			var expected_layout: int = (
				EcuTorqueVectoringV3.DriveLayout.RWD_2WD
				if in_inputs.drive_mode == VehicleConfig.DriveMode.REAR_WD
				else EcuTorqueVectoringV3.DriveLayout.AWD_4WD
			)
			if ecu_tv_v3.drive_layout != expected_layout:
				ecu_tv_v3.drive_layout = expected_layout
				if expected_layout == EcuTorqueVectoringV3.DriveLayout.RWD_2WD:
					ecu_tv_v3.front_split_active = 0.0
					ecu_tv_v3.rear_split_active = 1.0
		var r_whl: float = maxf(config.wheel_radius, 1e-4)
		var max_whl_trq: float = config.motor_peak_torque * config.gear_ratio
		var spd_clamped: float = maxf(absf(u), 1.0)
		var max_pwr_force: float = (config.power_limit_watts * config.drivetrain_efficiency) / spd_clamped
		var n_motors: float = 2.0 if in_inputs.drive_mode == VehicleConfig.DriveMode.REAR_WD else 4.0
		var max_fx_4wd: float = minf(n_motors * max_whl_trq / r_whl, max_pwr_force)
		var fx_driver: float = in_inputs.throttle * max_fx_4wd
		var tv_trqs: Array[float] = (
			ecu_tv_v3.step(fx_driver, sr_steer, sr_u, sr_v, sr_r, sr_ay, sr_ax, sr_omegas, in_inputs.brake, dt, _w_steers_scratch)
			if advance_tv else ecu_tv_v3.last_t_cmd.duplicate()
		)
		var lc_armed: bool = ecu_tv_v3.launch_state == EcuTorqueVectoringV3.LaunchState.LC_ARMED
		if in_inputs.throttle > 0.0 or in_inputs.brake < 0.05 or lc_armed:
			var eff: float = maxf(config.drivetrain_efficiency, 1e-3)
			var v_nom: float = maxf(config.battery_nominal_voltage_v, 1.0)
			var v_dc_cur: float = hv_state.v_dc if (hv_state != null and hv_state.v_dc > 0.0) else v_nom
			var v_scale: float = clampf(v_dc_cur / v_nom, 0.5, 1.25)
			var omega_base_motor: float = maxf(config.base_speed_rpm, 0.0) * (TAU / 60.0) * v_scale
			var p_motor_peak: float = config.motor_peak_torque * omega_base_motor
			for i in range(4):
				if tv_trqs[i] > 0.0:
					var omega_motor: float = absf(w_omegas[i]) * maxf(config.gear_ratio, 1e-3)
					var t_motor_env: float = config.motor_peak_torque
					if omega_base_motor > 0.0 and omega_motor > omega_base_motor:
						t_motor_env = minf(config.motor_peak_torque, p_motor_peak / omega_motor)
					var speed_scale: float = TorqueVectoring.compute_motor_speed_scale(w_omegas[i], true, config)
					tv_trqs[i] = minf(tv_trqs[i] * speed_scale, t_motor_env * config.gear_ratio * speed_scale)
			var p_dc_limit: float = maxf(config.max_dc_power_w, 0.0)
			if p_dc_limit > 0.0:
				var p_elec: float = 0.0
				for i in range(4):
					p_elec += maxf(0.0, tv_trqs[i] * absf(w_omegas[i])) / eff
				if p_elec > p_dc_limit:
					var p_scale: float = p_dc_limit / p_elec
					for i in range(4):
						tv_trqs[i] *= p_scale
			drive_trqs = tv_trqs
	else:
		var torques = compute_drivetrain_torques(in_inputs, u, dt, advance_tv)
		drive_trqs = torques["drive"]
		brake_trqs = torques["brake"]

	if hv_state != null:
		if dt > 0.0 and not advance_tv and not in_inputs.override_wheel_torques:
			drive_trqs = hv_state.inverter_torques.duplicate()
		else:
			drive_trqs = DrivetrainModel.step_hv_and_inverters(
				hv_state,
				drive_trqs,
				w_omegas,
				dt,
				config,
				advance_tv,
				in_inputs.override_wheel_torques
			)
	
	# 2. saca fz normales y alineacion de susp (sin contar doble la transf de carga)
	var normal_forces: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var toe_offsets: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var corner_dz: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var has_corner_travel: bool = false
	var chassis_roll_rad: float = chassis_state.roll
	if ground_contacts.size() >= 4:
		if ground_contacts[0] is Dictionary and ground_contacts[0].has("chassis_roll"):
			chassis_roll_rad = float(ground_contacts[0].get("chassis_roll", chassis_roll_rad))
		has_corner_travel = (ground_contacts[0] is Dictionary) and ground_contacts[0].has("delta_z")
		if has_corner_travel:
			var road_active := road_surface != null and road_surface.is_active()
			var r_wb_road: Basis = ChassisDynamics.basis_body_to_world(chassis_state.roll, chassis_state.pitch, chassis_state.yaw) if road_active else Basis.IDENTITY
			for i in range(4):
				var c: Dictionary = ground_contacts[i]
				_g_scratch[i] = bool(c.get("grounded", true))
				_dz_scratch[i] = float(c.get("delta_z", 0.0))
				_dz_dot_scratch[i] = float(c.get("delta_z_dot", 0.0))
				if road_active and _g_scratch[i]:
					var wx_r: float = 0.0
					var wy_r: float = 0.0
					var vx_w_r: float = 0.0
					var vy_w_r: float = 0.0
					if c.has("world_pos_ros"):
						var wp: Vector2 = c["world_pos_ros"]
						var wv: Vector2 = c.get("world_vel_ros", Vector2.ZERO)
						wx_r = wp.x; wy_r = wp.y
						vx_w_r = wv.x; vy_w_r = wv.y
					else:
						var rw_b := ChassisDynamics.get_wheel_pos_ros(i, config)
						var pw_ros := chassis_state.position + r_wb_road * rw_b
						var vw_ros := r_wb_road * (local_v_ros + local_w_ros.cross(rw_b))
						wx_r = pw_ros.x; wy_r = pw_ros.y
						vx_w_r = vw_ros.x; vy_w_r = vw_ros.y
					var rz: Vector2 = road_surface.sample_world(wx_r, wy_r, vx_w_r, vy_w_r)
					_dz_scratch[i] += rz.x
					_dz_dot_scratch[i] += rz.y
				corner_dz[i] = _dz_scratch[i]
			suspension.evaluate_from_corners(_dz_scratch, _dz_dot_scratch, _g_scratch, config, in_inputs.steer_angle)
			if use_3d:
				suspension.apply_non_elastic_lateral_load_transfer_3d(_prev_fy_corners, _prev_ay_ros, config)
				suspension.apply_non_elastic_longitudinal_load_transfer_3d(_prev_fx_corners, config)
			else:
				suspension.apply_non_elastic_lateral_load_transfer(_prev_fy_front, _prev_fy_rear, _prev_ay_ros, config)
				suspension.apply_non_elastic_longitudinal_load_transfer(_prev_fx_front, _prev_fx_rear, config)
			var corners: Array[SuspensionCorner] = suspension.corners
			for i in range(4):
				var sc: SuspensionCorner = corners[i]
				wheels[i].is_grounded = sc.is_grounded
				toe_offsets[i] = sc.toe_rad
				normal_forces[i] = sc.fz
		else:
			if use_3d:
				suspension.evaluate_kinematics_only(_dz_zero_scratch, config, in_inputs.steer_angle)
			for i in range(4):
				var c = ground_contacts[i]
				wheels[i].is_grounded = c.get("grounded", true)
				normal_forces[i] = c.get("fz", 0.0)
				toe_offsets[i] = suspension.corners[i].toe_rad if use_3d else SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	else:
		var ax_approx = (drive_trqs[0] + drive_trqs[1] + drive_trqs[2] + drive_trqs[3]) / (config.mass * config.wheel_radius)
		normal_forces = compute_analytic_normal_forces(ax_approx, u * r, local_v_ros.length())
		if use_3d:
			suspension.evaluate_kinematics_only(_dz_zero_scratch, config, in_inputs.steer_angle)
		for i in range(4):
			wheels[i].is_grounded = true
			toe_offsets[i] = suspension.corners[i].toe_rad if use_3d else SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	
	# 2b. frenada mixta hidraulica + regen & termica de disco 1 nodo
	var brake_res: Dictionary = BrakeModel.compute_blended_braking(
		in_inputs,
		config,
		brake_state,
		w_omegas,
		normal_forces,
		u,
		dt if advance_tv else 0.0,
		hv_state
	)
	if brake_res.has("total_torques"):
		var blended_trqs: Array = brake_res["total_torques"]
		var bias_f := clampf(config.brake_bias_front, 0.1, 0.9)
		if in_inputs.ecu_driving_mode == 2 and ecu_tv_4wd != null and ecu_tv_4wd.mech_brake_start > 0.05:
			var m_st := clampf(ecu_tv_4wd.mech_brake_start, 0.05, 0.95)
			var hyd_frac := clampf((in_inputs.brake - m_st) / maxf(1.0 - m_st, 0.05), 0.0, 1.0)
			var low_spd_hyd := clampf((2.0 - absf(u)) / 1.2, 0.0, 1.0)
			var hyd_scale := maxf(hyd_frac, low_spd_hyd * in_inputs.brake) if in_inputs.brake > 0.05 else 0.0
			var f_hyd_tot := hyd_scale * config.brake_mu * config.mass * config.gravity
			for i in range(4):
				var b_i := bias_f if i < 2 else (1.0 - bias_f)
				brake_trqs[i] = f_hyd_tot * 0.5 * b_i * config.wheel_radius
		else:
			var sum_regen_tv4 := 0.0
			if in_inputs.ecu_driving_mode == 2 and in_inputs.brake > 0.05:
				for j in range(4):
					sum_regen_tv4 += -minf(0.0, drive_trqs[j])
			for i in range(4):
				brake_trqs[i] = float(blended_trqs[i])
				if sum_regen_tv4 > 0.0:
					var cut_i := 0.5 * sum_regen_tv4 * (bias_f if i < 2 else (1.0 - bias_f))
					brake_trqs[i] = maxf(0.0, brake_trqs[i] - cut_i)

	# 3. cinematica, deslizamientos, fuerzas y giro x rueda
	for i in range(4):
		var dt_i: float = float(drive_trqs[i])
		wheel_drive_torque_nm[i] = maxf(dt_i, 0.0)
		wheel_regen_brake_nm[i] = maxf(-dt_i, 0.0)
		wheel_mech_brake_nm[i] = maxf(float(brake_trqs[i]), 0.0)
	var steer_angle = in_inputs.steer_angle
	var total_wheel_force_ros = Vector3.ZERO
	var total_wheel_torque_ros = Vector3.ZERO
	var wheel_forces_ros: Array[Vector3] = []
	var wheel_moments_ros: Array[Vector3] = []
	var wheel_steer_angles: Array[float] = []
	
	for i in range(4):
		var w = wheels[i]
		var r_pos: Vector3 = get_wheel_pos_ros(i)
		if ground_contacts.size() >= 4 and ground_contacts[i] is Dictionary:
			r_pos = ground_contacts[i].get("r_pos_ros", r_pos)
			if ground_contacts[i].has("surface_mu"):
				w.surface_mu_scale = float(ground_contacts[i]["surface_mu"]) * env_mu_scale
			else:
				w.surface_mu_scale = env_mu_scale
		else:
			w.surface_mu_scale = env_mu_scale
		if use_3d:
			var sc_cp: SuspensionCorner = suspension.corners[i]
			r_pos.x += sc_cp.cp_offset_ros.x
			r_pos.y += sc_cp.cp_offset_ros.y
		var v_hub = local_v_ros + local_w_ros.cross(r_pos)
		w.v_hub_chassis = v_hub
		
		var delta_w: float = 0.0
		if use_3d:
			var sc_st: SuspensionCorner = suspension.corners[i]
			var delta_comp_3d: float = SuspensionSystem.compute_steering_compliance(w.M_z, w.F_y, config, sc_st.mechanical_trail_m) if i < 2 else 0.0
			delta_w = sc_st.wheel_steer_rad + delta_comp_3d
			w.camber_angle = sc_st.camber_rad - chassis_roll_rad
		else:
			var base_steer: float = 0.0
			if i == VehicleConfig.WheelIndex.FL:
				base_steer = compute_ackermann_steer(steer_angle, true, config)
			elif i == VehicleConfig.WheelIndex.FR:
				base_steer = compute_ackermann_steer(steer_angle, false, config)
			var delta_comp: float = SuspensionSystem.compute_steering_compliance(w.M_z, w.F_y, config) if i < 2 else 0.0
			delta_w = base_steer + toe_offsets[i] + delta_comp
			w.camber_angle = SuspensionSystem.compute_ground_camber_iso(
				corner_dz[i], delta_w, chassis_roll_rad, i < 2, w.is_right_side, config
			)
		wheel_steer_angles.append(delta_w)
		
		w.update_radial_deflection(normal_forces[i], i < 2, config)
		w.compute_slip(v_hub.x, v_hub.y, delta_w, config.slip_regularization_speed, w.effective_radius, dt)
		w.compute_forces(normal_forces[i], config)
		w.integrate_wheel_spin(drive_trqs[i], brake_trqs[i], dt, config)
		if advance_tv and dt > 0.0:
			w.step_thermal(dt, config)
		
		var cos_d = cos(delta_w)
		var sin_d = sin(delta_w)
		var F_chassis_x = w.F_x * cos_d - w.F_y * sin_d
		var F_chassis_y = w.F_x * sin_d + w.F_y * cos_d
		var F_chassis_z = w.F_z
		var F_wheel_chassis = Vector3(F_chassis_x, F_chassis_y, F_chassis_z)
		var M_tire_chassis = Vector3(w.M_x * cos_d, w.M_x * sin_d, w.M_z)
		
		wheel_forces_ros.append(F_wheel_chassis)
		wheel_moments_ros.append(M_tire_chassis)
		total_wheel_force_ros += F_wheel_chassis
		var tau_wheel = r_pos.cross(F_wheel_chassis) + M_tire_chassis
		total_wheel_torque_ros += tau_wheel
	
	# 3b. reaccion geometrica anti-dive/squat solo si no hay corner travel (con travel ya va en r_pos x f_z)
	if not has_corner_travel:
		var fx_corners: Array = [
			wheel_forces_ros[0].x,
			wheel_forces_ros[1].x,
			wheel_forces_ros[2].x,
			wheel_forces_ros[3].x
		]
		var anti_pitch: Dictionary = SuspensionSystem.compute_anti_pitch_forces(fx_corners, config)
		total_wheel_torque_ros.y += float(anti_pitch["pitch_torque_ros"])

	# 3c. par fisico de force feedback en columna (sat + caster trail + scrub radius + damping + soft-lock)
	_update_steering_feedback(in_inputs.steer_angle, dt)

	# 4. aerodynamics (ride-height sensitive ground effect & front-wing stall)
	var z_front_dyn: float = -1.0
	var z_rear_dyn: float = -1.0
	if has_corner_travel:
		z_front_dyn = maxf(0.0, config.static_ride_height_front - 0.5 * (corner_dz[0] + corner_dz[1]))
		z_rear_dyn = maxf(0.0, config.static_ride_height_rear - 0.5 * (corner_dz[2] + corner_dz[3]))
	var aero = compute_aerodynamics(local_v_ros, z_front_dyn, z_rear_dyn)
	var f_aero_ros: Vector3 = aero["force"]
	var tau_aero_ros: Vector3 = aero["torque"]
	if env_aero_cl_scale != 1.0:
		f_aero_ros.z *= env_aero_cl_scale
		tau_aero_ros.y *= env_aero_cl_scale
	
	# 5. net force & torque on chassis (ros frame)
	net_force_chassis_ros = total_wheel_force_ros + f_aero_ros
	net_torque_chassis_ros = total_wheel_torque_ros + tau_aero_ros
	for i in range(4):
		var w_prev: WheelDynamics = wheels[i]
		var roll_spd: float = maxf(absf(w_prev.v_longitudinal), maxf(absf(w_prev.v_lateral), absf(w_prev.omega * config.wheel_radius)))
		var roll_fade: float = clampf((roll_spd - 0.02) / 0.18, 0.0, 1.0)
		_prev_fx_corners[i] = wheel_forces_ros[i].x
		_prev_fy_corners[i] = wheel_forces_ros[i].y * roll_fade
		_prev_fz_corners[i] = normal_forces[i]
	_prev_fy_front = _prev_fy_corners[0] + _prev_fy_corners[1]
	_prev_fy_rear = _prev_fy_corners[2] + _prev_fy_corners[3]
	_prev_fx_front = _prev_fx_corners[0] + _prev_fx_corners[1]
	_prev_fx_rear = _prev_fx_corners[2] + _prev_fx_corners[3]
	_prev_ax_ros = net_force_chassis_ros.x / maxf(config.mass, 1e-3)
	_prev_ay_ros = net_force_chassis_ros.y / maxf(config.mass, 1e-3)
	_prev_az_ros = config.gravity + (net_force_chassis_ros.z / maxf(config.mass, 1e-3))
	
	# convert net quantities to godot frame:
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)

	var tire_temps: Array[float] = [wheels[0].t_surf_c, wheels[1].t_surf_c, wheels[2].t_surf_c, wheels[3].t_surf_c]
	# telemetria mz_req/wz_ref segun ecu activa (v3 o 4wd)
	var mz_req_out: float = ecu_tv_v3.last_mz_req if (ecu_tv_v3 != null and in_inputs.ecu_driving_mode == 1) else 0.0
	var wz_ref_out: float = ecu_tv_v3.last_wz_ref if ecu_tv_v3 != null else 0.0
	if in_inputs.ecu_driving_mode == 2 and ecu_tv_4wd != null:
		mz_req_out = ecu_tv_4wd.last_mz_req
		wz_ref_out = ecu_tv_4wd.last_wz_ref

	return {
		"net_force_ros": net_force_chassis_ros,
		"net_torque_ros": net_torque_chassis_ros,
		"net_force_godot": net_force_chassis_godot,
		"net_torque_godot": net_torque_chassis_godot,
		"wheel_forces_ros": wheel_forces_ros,
		"wheel_moments_ros": wheel_moments_ros,
		"aero_force_ros": f_aero_ros,
		"aero_torque_ros": tau_aero_ros,
		"drive_torques": drive_trqs,
		"brake_torques": brake_trqs,
		"wheel_drive_torque_nm": wheel_drive_torque_nm,
		"wheel_regen_brake_nm": wheel_regen_brake_nm,
		"wheel_mech_brake_nm": wheel_mech_brake_nm,
		"steering_torque_nm": steering_torque_nm,
		"kp_moment_fl_nm": kp_moment_fl_nm,
		"kp_moment_fr_nm": kp_moment_fr_nm,
		"normal_forces": normal_forces,
		"wheel_steer_angles": wheel_steer_angles,
		"tire_temps": tire_temps,
		"hv_state": hv_state,
		"v_dc": hv_state.v_dc if hv_state != null else 588.0,
		"i_dc": hv_state.i_dc if hv_state != null else 0.0,
		"soc": hv_state.soc if hv_state != null else 0.95,
		"power_kw": hv_state.power_kw if hv_state != null else 0.0,
		"energy_kwh": hv_state.energy_kwh if hv_state != null else 0.0,
		"map_id": ecu_tv_v3.active_map_id if ecu_tv_v3 != null else 0,
		"mz_req": mz_req_out,
		"wz_ref": wz_ref_out,
		"z_front": z_front_dyn,
		"z_rear": z_rear_dyn,
		"aero_front_stalled": bool(aero.get("front_stalled", false)),
		"disc_temps": brake_state.disc_temp.duplicate() if brake_state != null else [40.0, 40.0, 40.0, 40.0],
		"regen_power_kw": (brake_state.p_regen_w / 1000.0) if brake_state != null else 0.0,
		"surface_mu": [wheels[0].surface_mu_scale, wheels[1].surface_mu_scale, wheels[2].surface_mu_scale, wheels[3].surface_mu_scale],
		"sensor_realism_mode": sensor_realism.mode if sensor_realism != null else 0,
		"sensor_beta_hat": sr_beta_hat,
		"sensor_fz_hat_4": sr_fz_hat_4,
		"sensor_ax_meas": sr_ax,
		"sensor_ay_meas": sr_ay,
		"sensor_wz_meas": sr_r,
		"sensor_can_latency_ms": sensor_realism.can_accel_latency_ms if (sensor_realism != null and sr_active) else 0.0,
		"sensor_imu_noise_scale": sensor_realism.imu_noise_scale if (sensor_realism != null and sr_active) else 0.0
	}

## unified 14-dof standalone step (6-dof rigid-body chassis via rk2 heun + 4 wheels + 4 susp corners)
## shares the exact same step_forces() evaluation path used by mirenacardynamics.gd
func step_standalone(in_inputs: VehicleInputs, dt: float, ground_height: float = 0.0) -> Dictionary:
	if dt <= 0.0:
		var contacts_idle: Array = ChassisDynamics.compute_ground_contacts_6dof(chassis_state, config, ground_height)
		var res_idle: Dictionary = step_forces(in_inputs, chassis_state.linear_velocity, chassis_state.angular_velocity, contacts_idle, 0.0)
		return {
			"state_6dof": chassis_state,
			"position_ros": chassis_state.position,
			"euler_ros": Vector3(chassis_state.roll, chassis_state.pitch, chassis_state.yaw),
			"linear_velocity_ros": chassis_state.linear_velocity,
			"angular_velocity_ros": chassis_state.angular_velocity,
			"normal_forces": res_idle["normal_forces"],
			"net_force_ros": res_idle["net_force_ros"],
			"net_torque_ros": res_idle["net_torque_ros"],
			"forces": res_idle,
			"ground_contacts": contacts_idle
		}

	var s0: ChassisDynamics.State6DOF = chassis_state.clone()
	var w0: Array[Vector4] = [
		wheels[0].get_integrator_state(),
		wheels[1].get_integrator_state(),
		wheels[2].get_integrator_state(),
		wheels[3].get_integrator_state()
	]

	# rk2 stage 1 (evaluate at t_n; advance 200 hz tv accumulator once per physical step)
	var contacts_1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, config, ground_height)
	var res_1: Dictionary = step_forces(in_inputs, s0.linear_velocity, s0.angular_velocity, contacts_1, dt, true)
	var k1: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s0, res_1["net_force_ros"], res_1["net_torque_ros"], config)
	var s_pred: ChassisDynamics.State6DOF = ChassisDynamics.advance_euler(s0, k1, dt)

	# rk2 stage 2 (evaluate at predictor t_n + dt; reuse latched tv command)
	var contacts_2: Array = ChassisDynamics.compute_ground_contacts_6dof(s_pred, config, ground_height)
	var res_2: Dictionary = step_forces(in_inputs, s_pred.linear_velocity, s_pred.angular_velocity, contacts_2, dt, false)
	var k2: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s_pred, res_2["net_force_ros"], res_2["net_torque_ros"], config)

	# heun update for 6-dof chassis and 4 wheels
	chassis_state = ChassisDynamics.advance_heun(s0, k1, k2, dt)
	var all_wheels_stopped: bool = true
	for i in range(4):
		var w2_step: Vector4 = wheels[i].get_integrator_state()
		var w_heun: Vector4 = 0.5 * (w0[i] + w2_step)
		# mantiene el cero limpio al frenar hasta parar
		if is_zero_approx(w2_step.x) and in_inputs.brake > 0.0 and in_inputs.throttle == 0.0:
			w_heun.x = 0.0
		if absf(w_heun.x) > 1e-3:
			all_wheels_stopped = false
		wheels[i].set_integrator_state(w_heun)

	# evita rebote o creep de carcasa en parado
	var fz_arr: Array = res_2["normal_forces"]
	var max_fz: float = maxf(maxf(float(fz_arr[0]), float(fz_arr[1])), maxf(float(fz_arr[2]), float(fz_arr[3])))
	var v_planar: float = Vector2(chassis_state.linear_velocity.x, chassis_state.linear_velocity.y).length()
	var stopped_hold: bool = (max_fz > 1.0) and (in_inputs.throttle < 0.01) and (v_planar < 0.30) and (in_inputs.brake > 0.02 or absf(_prev_ax_ros) < 0.45)
	if stopped_hold:
		chassis_state.linear_velocity.x = 0.0
		chassis_state.linear_velocity.y = 0.0
		chassis_state.angular_velocity.z = 0.0
		if hv_state != null:
			hv_state.inverter_torques = [0.0, 0.0, 0.0, 0.0]
		for i in range(4):
			wheels[i].omega = 0.0
			wheels[i].kappa = 0.0
			wheels[i].alpha = 0.0
			wheels[i].kappa_transient = 0.0
			wheels[i].alpha_transient = 0.0
	elif in_inputs.brake > 0.0 and in_inputs.throttle == 0.0:
		if s0.linear_velocity.x >= 0.0 and chassis_state.linear_velocity.x <= 0.0:
			chassis_state.linear_velocity.x = 0.0
			if all_wheels_stopped:
				for i in range(4):
					wheels[i].kappa = 0.0
					wheels[i].kappa_transient = 0.0

	linear_velocity_ros = chassis_state.linear_velocity
	angular_velocity_ros = chassis_state.angular_velocity
	net_force_chassis_ros = 0.5 * (res_1["net_force_ros"] + res_2["net_force_ros"])
	net_torque_chassis_ros = 0.5 * (res_1["net_torque_ros"] + res_2["net_torque_ros"])
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)

	return {
		"state_6dof": chassis_state,
		"position_ros": chassis_state.position,
		"euler_ros": Vector3(chassis_state.roll, chassis_state.pitch, chassis_state.yaw),
		"linear_velocity_ros": chassis_state.linear_velocity,
		"angular_velocity_ros": chassis_state.angular_velocity,
		"normal_forces": res_2["normal_forces"],
		"net_force_ros": net_force_chassis_ros,
		"net_torque_ros": net_torque_chassis_ros,
		"forces": res_2,
		"ground_contacts": contacts_2
	}

func reset() -> void:
	for w in wheels:
		w.reset()
	_sync_wheel_thermal_config()
	suspension.reset()
	suspension.sync_from_config(config)
	torque_vectoring.reset()
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
	_sync_sensor_realism_vehicle_params()
	if ecu_tv_v3 != null:
		ecu_tv_v3.configure_from_vehicle_config(config)
		ecu_tv_v3.reset()
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
		ecu_tv_4wd.reset()
	if hv_state != null:
		hv_state.reset(config)
	if brake_state != null:
		brake_state = BrakeModel.BrakeState.new(config.disc_ambient_temp_c)
	chassis_state.reset(config.h_cog)
	inputs.reset()
	inputs.drive_mode = config.drive_mode
	_last_input_tv_map_id = inputs.tv_map_id
	net_force_chassis_ros = Vector3.ZERO
	net_torque_chassis_ros = Vector3.ZERO
	net_force_chassis_godot = Vector3.ZERO
	net_torque_chassis_godot = Vector3.ZERO
	linear_velocity_ros = Vector3.ZERO
	angular_velocity_ros = Vector3.ZERO
	_prev_fy_front = 0.0
	_prev_fy_rear = 0.0
	_prev_fx_front = 0.0
	_prev_fx_rear = 0.0
	for i in range(4):
		_prev_fx_corners[i] = 0.0
		_prev_fy_corners[i] = 0.0
		_prev_fz_corners[i] = 640.0 if i < 2 else 780.0
	if sensor_realism != null:
		sensor_realism.reset(-1)
	_prev_az_ros = config.gravity
	_prev_ax_ros = 0.0
	_prev_ay_ros = 0.0
	steering_feedback_torque_nm = 0.0
	steering_feedback_normalized = 0.0
	steering_torque_nm = 0.0
	kp_moment_fl_nm = 0.0
	kp_moment_fr_nm = 0.0
	_prev_steer_angle_ffb = 0.0
	_ffb_norm_filt = 0.0
	for i in range(4):
		wheel_drive_torque_nm[i] = 0.0
		wheel_regen_brake_nm[i] = 0.0
		wheel_mech_brake_nm[i] = 0.0

## calcula par ffb en columna (nm) y salida normalizada [-1, 1] con lpf ligero
func _update_steering_feedback(steer_rad: float, dt: float) -> void:
	if wheels.size() < 2:
		return
	var t_mech_fl: float = config.mechanical_trail_front * cos(steer_rad)
	var t_mech_fr: float = t_mech_fl
	var r_scrub_fl: float = config.scrub_radius_front
	var r_scrub_fr: float = r_scrub_fl
	if suspension != null and suspension.is_3d_active(config):
		var c_fl: SuspensionCorner = suspension.corners[0]
		var c_fr: SuspensionCorner = suspension.corners[1]
		t_mech_fl = c_fl.mechanical_trail_m
		t_mech_fr = c_fr.mechanical_trail_m
		r_scrub_fl = c_fl.scrub_radius_m
		r_scrub_fr = c_fr.scrub_radius_m
	kp_moment_fl_nm = wheels[0].M_z - wheels[0].F_y * t_mech_fl - wheels[0].F_x * r_scrub_fl
	kp_moment_fr_nm = wheels[1].M_z - wheels[1].F_y * t_mech_fr + wheels[1].F_x * r_scrub_fr
	var sr: float = maxf(config.steering_ratio, 0.1)
	var sw_rad: float = steer_rad * sr
	var sw_dot: float = ((steer_rad - _prev_steer_angle_ffb) * sr / dt) if dt > 1e-6 else 0.0
	_prev_steer_angle_ffb = steer_rad
	var tau_damp: float = -config.ffb_damping_nms_rad * sw_dot
	var sw_max_rad: float = deg_to_rad(maxf(config.fs_steer_wheel_max_deg, 10.0))
	var tau_softlock: float = 0.0
	if absf(sw_rad) > sw_max_rad:
		tau_softlock = -config.ffb_soft_lock_stiffness_nm_rad * (absf(sw_rad) - sw_max_rad) * signf(sw_rad)
	steering_feedback_torque_nm = ((kp_moment_fl_nm + kp_moment_fr_nm) / sr) + tau_damp + tau_softlock
	steering_torque_nm = steering_feedback_torque_nm
	var raw_norm: float = clampf(steering_feedback_torque_nm / maxf(config.ffb_max_wheel_torque_nm, 0.5), -1.0, 1.0)
	var alpha_ffb: float = 1.0 - exp(-dt / 0.015) if dt > 0.0 else 1.0
	_ffb_norm_filt = lerpf(_ffb_norm_filt, raw_norm, alpha_ffb)
	steering_feedback_normalized = clampf(_ffb_norm_filt, -1.0, 1.0)
