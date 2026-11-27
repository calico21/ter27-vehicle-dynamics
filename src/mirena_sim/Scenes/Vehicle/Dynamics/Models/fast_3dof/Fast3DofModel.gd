class_name Fast3DofVehicleModel
extends RefCounted

## solver vehicular simplificado de 7 gdl efectivos (chasis plano 3-dof + 4 ruedas)
## pensado pa driverless a 200 hz (dt = 0.005 s)
## lee exactamente los mismos json de config/vehicles/<id>/ q vehicledynamicsmodel
## convención interna: iso 8855 / ros rep-103 (+x adelante, +y izq, +z arriba)

const VehicleConfig = preload("../../../Config/VehicleConfig.gd")
const VehicleInputs = preload("../../../Inputs/VehicleInputs.gd")
const SetupProfileScript = preload("../../../Config/SetupProfile.gd")
const WheelDynamics = preload("../../Tires/WheelDynamics.gd")
const TireModel = preload("../../Tires/TireModel.gd")
const DrivetrainModel = preload("../../Drivetrain/DrivetrainModel.gd")
const TorqueVectoring = preload("../../Drivetrain/TorqueVectoring.gd")
const EcuTorqueVectoringV3 = preload("../../Drivetrain/EcuTorqueVectoringV3.gd")
const EcuTorqueVectoring4Wd = preload("../../Drivetrain/EcuTorqueVectoring4Wd.gd")
const SuspensionSystem = preload("../../Suspension/SuspensionSystem.gd")
const AeroModel = preload("../../Aerodynamics/AeroModel.gd")
const ChassisDynamics = preload("../../Chassis/ChassisDynamics.gd")
const BrakeModel = preload("../../Brakes/BrakeModel.gd")

const TAU_WT_S: float = 0.035  # filtro 1er orden de transferencia de carga (35 ms)

var config: VehicleConfig
var wheels: Array[WheelDynamics] = []
var suspension: SuspensionSystem
var torque_vectoring: TorqueVectoring
var ecu_tv_v3: EcuTorqueVectoringV3
var ecu_tv_4wd: EcuTorqueVectoring4Wd
var hv_state: DrivetrainModel.HVPowertrainState
var brake_state: BrakeModel.BrakeState
var chassis_state: ChassisDynamics.State6DOF
var inputs: VehicleInputs

var _last_input_tv_map_id: int = 2
var _ax_filt: float = 0.0
var _ay_filt: float = 0.0
var _raw_ax_prev: float = 0.0
var _raw_ay_prev: float = 0.0
var _prev_ax_ros: float = 0.0
var _prev_ay_ros: float = 0.0

var net_force_chassis_ros: Vector3 = Vector3.ZERO
var net_torque_chassis_ros: Vector3 = Vector3.ZERO
var net_force_chassis_godot: Vector3 = Vector3.ZERO
var net_torque_chassis_godot: Vector3 = Vector3.ZERO
var linear_velocity_ros: Vector3 = Vector3.ZERO
var angular_velocity_ros: Vector3 = Vector3.ZERO

# paridad de interfaz ffb con full_14dof
var steering_feedback_torque_nm: float = 0.0
var steering_feedback_normalized: float = 0.0
var _prev_steer_angle_ffb: float = 0.0
var _ffb_norm_filt: float = 0.0


func _init(cfg: VehicleConfig = null) -> void:
	config = cfg if cfg != null else VehicleConfig.new()
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
	hv_state = DrivetrainModel.HVPowertrainState.new(config)
	brake_state = BrakeModel.BrakeState.new(config.disc_ambient_temp_c)
	chassis_state = ChassisDynamics.State6DOF.new(config.h_cog)
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
	var p_f: float = config.tire_pressure_front_bar * 100.0
	var p_r: float = config.tire_pressure_rear_bar * 100.0
	for i in range(wheels.size()):
		var w := wheels[i]
		w.set_tire_temperature(config.tire_initial_temp_c)
		w.t_track_c = config.track_temp_c
		w.inflation_pressure_kpa = p_f if i < 2 else p_r


func load_vehicle(vehicle_id: String, setup_name: String = "default") -> bool:
	var new_cfg := VehicleConfig.new()
	if not new_cfg.load_vehicle_configs(vehicle_id):
		return false
	config = new_cfg
	if config.tire_compound != "" and TireModel.get_compound() != config.tire_compound:
		TireModel.set_compound(config.tire_compound)
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
	if ecu_tv_v3 != null:
		ecu_tv_v3.configure_from_vehicle_config(config)
		var v_maps := ecu_tv_v3.list_available_tv_maps()
		if not v_maps.is_empty():
			ecu_tv_v3.load_tv_map_from_file(String(v_maps[0]["path"]))
	reset()
	if setup_name.strip_edges() != "":
		apply_setup(setup_name)
	return true


func apply_setup(setup_name: String) -> bool:
	if config == null:
		return false
	var prof := SetupProfileScript.load_for_vehicle(config.vehicle_id, setup_name)
	if prof == null:
		return false
	return apply_setup_profile(prof)


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


func get_wheel_pos_ros(index: int) -> Vector3:
	return ChassisDynamics.get_wheel_pos_ros(index, config)


func compute_lltd_front() -> float:
	var kw_f: float = config.spring_stiffness_front * config.motion_ratio_front * config.motion_ratio_front
	var kw_r: float = config.spring_stiffness_rear * config.motion_ratio_rear * config.motion_ratio_rear
	var tf: float = maxf(config.track_width_f, 0.5)
	var tr: float = maxf(config.track_width_r, 0.5)
	var k_phi_f: float = 0.5 * kw_f * tf * tf + config.arb_stiffness_front
	var k_phi_r: float = 0.5 * kw_r * tr * tr + config.arb_stiffness_rear
	return clampf(k_phi_f / maxf(k_phi_f + k_phi_r, 1.0), 0.1, 0.9)


func compute_normal_forces_quasistatic(ax_f: float, ay_f: float, vx: float) -> Array[float]:
	var m: float = maxf(config.mass, 100.0)
	var g: float = config.gravity
	var wb: float = maxf(config.wheelbase, 0.5)
	var tf: float = maxf(config.track_width_f, 0.5)
	var tr: float = maxf(config.track_width_r, 0.5)

	var f_down: float = 0.5 * config.air_density * config.cl_a * vx * absf(vx)
	var fz_stat_f: float = 0.5 * m * g * config.front_weight_distribution
	var fz_stat_r: float = 0.5 * m * g * config.weight_dist_r
	var fz_aero_f: float = 0.5 * f_down * config.aero_balance_front
	var fz_aero_r: float = 0.5 * f_down * (1.0 - config.aero_balance_front)

	var dfz_long: float = (m * ax_f * config.h_cog) / wb
	var lltd_f: float = compute_lltd_front()
	var a_lf: float = config.a_cog
	var b_lr: float = config.b_cog
	var h_rc_cg: float = config.roll_center_height_cg
	var h_arm: float = maxf(config.h_sprung_cog - h_rc_cg, 0.05)

	var dfz_lat_f: float = (m * ay_f / tf) * ((b_lr / wb) * config.roll_center_height_front + lltd_f * h_arm)
	var dfz_lat_r: float = (m * ay_f / tr) * ((a_lf / wb) * config.roll_center_height_rear + (1.0 - lltd_f) * h_arm)

	var fz_fl: float = maxf(fz_stat_f + fz_aero_f - 0.5 * dfz_long - dfz_lat_f, 0.0)
	var fz_fr: float = maxf(fz_stat_f + fz_aero_f - 0.5 * dfz_long + dfz_lat_f, 0.0)
	var fz_rl: float = maxf(fz_stat_r + fz_aero_r + 0.5 * dfz_long - dfz_lat_r, 0.0)
	var fz_rr: float = maxf(fz_stat_r + fz_aero_r + 0.5 * dfz_long + dfz_lat_r, 0.0)
	return [fz_fl, fz_fr, fz_rl, fz_rr]


func _compute_planar_forces(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	dt: float,
	advance_tv: bool,
	ground_contacts: Array = [],
	integrate_wheels: bool = true
) -> Dictionary:
	linear_velocity_ros = local_v_ros
	angular_velocity_ros = local_w_ros
	inputs = in_inputs

	var u: float = local_v_ros.x
	var v: float = local_v_ros.y
	var r: float = local_w_ros.z

	# transferencia de carga cuasi-estática con filtro de 35 ms
	if advance_tv and dt > 0.0:
		var alpha_wt: float = 1.0 - exp(-dt / TAU_WT_S)
		_ax_filt = lerpf(_ax_filt, _raw_ax_prev, alpha_wt)
		_ay_filt = lerpf(_ay_filt, _raw_ay_prev, alpha_wt)

	var normal_forces: Array[float] = compute_normal_forces_quasistatic(_ax_filt, _ay_filt, u)
	# altura dinamica x efecto suelo pa paridad fisica con full_14dof (sin alterar compute_normal_forces_quasistatic)
	var f_down_nom: float = 0.5 * config.air_density * config.cl_a * u * absf(u)
	var fz_aero_f_nom: float = 0.5 * f_down_nom * config.aero_balance_front
	var fz_aero_r_nom: float = 0.5 * f_down_nom * (1.0 - config.aero_balance_front)
	var dfz_long_est: float = (maxf(config.mass, 100.0) * _ax_filt * config.h_cog) / maxf(config.wheelbase, 0.5)
	var anti_fac: float = config.anti_squat_ratio_rear if _ax_filt >= 0.0 else config.anti_dive_ratio_front
	var kw_f_3: float = config.spring_stiffness_front * config.motion_ratio_front * config.motion_ratio_front
	var kw_r_3: float = config.spring_stiffness_rear * config.motion_ratio_rear * config.motion_ratio_rear
	var dz_f_est: float = (fz_aero_f_nom - 0.5 * (1.0 - anti_fac) * dfz_long_est) / maxf(kw_f_3, 1000.0)
	var dz_r_est: float = (fz_aero_r_nom + 0.5 * (1.0 - anti_fac) * dfz_long_est) / maxf(kw_r_3, 1000.0)
	var z_f_dyn: float = maxf(0.0, config.static_ride_height_front - dz_f_est)
	var z_r_dyn: float = maxf(0.0, config.static_ride_height_rear - dz_r_est)
	var aero_res: Dictionary = AeroModel.compute_forces_ros(local_v_ros, config, z_f_dyn, z_r_dyn)
	var dfz_af_ge: float = 0.5 * float(aero_res["downforce_front"]) - fz_aero_f_nom
	var dfz_ar_ge: float = 0.5 * float(aero_res["downforce_rear"]) - fz_aero_r_nom
	normal_forces[0] = maxf(normal_forces[0] + dfz_af_ge, 0.0)
	normal_forces[1] = maxf(normal_forces[1] + dfz_af_ge, 0.0)
	normal_forces[2] = maxf(normal_forces[2] + dfz_ar_ge, 0.0)
	normal_forces[3] = maxf(normal_forces[3] + dfz_ar_ge, 0.0)
	var total_fz: float = 0.0
	for i in range(4):
		total_fz += normal_forces[i]
		if suspension != null and i < suspension.corners.size():
			suspension.corners[i].fz = normal_forces[i]
			suspension.corners[i].is_grounded = normal_forces[i] > 1.0

	if ecu_tv_v3 != null and in_inputs.tv_map_id != _last_input_tv_map_id:
		if in_inputs.tv_map_id != ecu_tv_v3.active_map_id:
			ecu_tv_v3.set_map(in_inputs.tv_map_id)
		_last_input_tv_map_id = ecu_tv_v3.active_map_id

	var w_omegas: Array[float] = [wheels[0].omega, wheels[1].omega, wheels[2].omega, wheels[3].omega]
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
			# demanda regen coherente con brakemodel pa evitar doble escalado de p_chg
			var p_chg: float = maxf(config.p_max_charge_w, 20000.0)
			var f_brk_dem: float = in_inputs.brake * config.brake_mu * config.mass * config.gravity
			fx_drv4 = -minf(f_brk_dem, minf(4.0 * ecu_tv_4wd.t_regen_max / r_whl4, p_chg / maxf(absf(u), 2.0))) if absf(u) > 2.0 else 0.0
		var w_st4: Array[float] = [
			SuspensionSystem.compute_ackermann_steer(in_inputs.steer_angle, true, config),
			SuspensionSystem.compute_ackermann_steer(in_inputs.steer_angle, false, config),
			0.0, 0.0
		]
		drive_trqs = (
			ecu_tv_4wd.step(fx_drv4, in_inputs.steer_angle, u, r, _prev_ax_ros, _prev_ay_ros, config.gravity, w_omegas, in_inputs.brake, dt, w_st4)
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
		var w_steers: Array[float] = [
			SuspensionSystem.compute_ackermann_steer(in_inputs.steer_angle, true, config),
			SuspensionSystem.compute_ackermann_steer(in_inputs.steer_angle, false, config),
			0.0, 0.0
		]
		var tv_trqs: Array[float] = (
			ecu_tv_v3.step(fx_driver, in_inputs.steer_angle, u, v, r, _prev_ay_ros, _prev_ax_ros, w_omegas, in_inputs.brake, dt, w_steers)
			if advance_tv else ecu_tv_v3.last_t_cmd.duplicate()
		)
		var lc_armed: bool = ecu_tv_v3 != null and ecu_tv_v3.launch_state == EcuTorqueVectoringV3.LaunchState.LC_ARMED
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
		var v_dc_eff: float = hv_state.v_dc if hv_state != null else -1.0
		var dt_res: Dictionary = DrivetrainModel.compute_torques(
			in_inputs, u, config, w_omegas, torque_vectoring, dt, advance_tv, v_dc_eff
		)
		drive_trqs = dt_res["drive"]
		brake_trqs = dt_res["brake"]

	var brake_res: Dictionary = BrakeModel.compute_blended_braking(
		in_inputs, config, brake_state, w_omegas, normal_forces, u, dt if advance_tv else 0.0, hv_state
	)
	if brake_res.has("total_torques"):
		var blended_trqs: Array = brake_res["total_torques"]
		var sum_regen_tv4 := 0.0
		if in_inputs.ecu_driving_mode == 2 and in_inputs.brake > 0.05:
			for j in range(4):
				sum_regen_tv4 += -minf(0.0, drive_trqs[j])
		var bias_f := clampf(config.brake_bias_front, 0.1, 0.9)
		for i in range(4):
			brake_trqs[i] = float(blended_trqs[i])
			if sum_regen_tv4 > 0.0:
				var cut_i := 0.5 * sum_regen_tv4 * (bias_f if i < 2 else (1.0 - bias_f))
				brake_trqs[i] = maxf(0.0, brake_trqs[i] - cut_i)

	var eff_drive_trqs: Array = drive_trqs
	if hv_state != null:
		if dt > 0.0 and not advance_tv and not in_inputs.override_wheel_torques:
			eff_drive_trqs = hv_state.inverter_torques.duplicate()
		else:
			eff_drive_trqs = DrivetrainModel.step_hv_and_inverters(
				hv_state, drive_trqs, w_omegas, dt, config, advance_tv, in_inputs.override_wheel_torques
			)

	var net_fx: float = 0.0
	var net_fy: float = 0.0
	var net_mz: float = 0.0
	var wheel_forces_out: Array = []
	var wheel_forces_ros: Array[Vector3] = []
	var wheel_moments_ros: Array[Vector3] = []
	var wheel_steer_angles: Array[float] = []

	for i in range(4):
		var is_front: bool = (i < 2)
		var is_left: bool = (i == 0 or i == 2)
		var r_pos: Vector3 = get_wheel_pos_ros(i)
		var w: WheelDynamics = wheels[i]
		var base_steer: float = SuspensionSystem.compute_ackermann_steer(in_inputs.steer_angle, is_left, config) if is_front else 0.0
		var toe_off: float = SuspensionSystem.compute_corner_toe_iso(0.0, is_front, w.is_right_side, config)
		var delta_comp: float = SuspensionSystem.compute_steering_compliance(w.M_z, w.F_y, config) if is_front else 0.0
		var steer_rad: float = base_steer + toe_off + delta_comp
		var v_hub_x: float = u - r * r_pos.y
		var v_hub_y: float = v + r * r_pos.x
		w.is_grounded = normal_forces[i] > 1.0
		w.surface_mu_scale = float(ground_contacts[i].get("surface_mu", 1.0)) if (i < ground_contacts.size() and ground_contacts[i] is Dictionary) else 1.0
		w.camber_angle = SuspensionSystem.compute_ground_camber_iso(0.0, steer_rad, chassis_state.roll, is_front, w.is_right_side, config)
		w.update_radial_deflection(normal_forces[i], is_front, config)
		w.compute_slip(v_hub_x, v_hub_y, steer_rad, maxf(config.slip_regularization_speed, 0.35), w.effective_radius, dt)
		w.compute_forces(normal_forces[i], config)

		if integrate_wheels and dt > 0.0:
			var n_spin: int = maxi(1, int(ceili(dt / 0.0025)))
			var spin_dt: float = dt / float(n_spin)
			for _sp in range(n_spin):
				w.integrate_wheel_spin(float(eff_drive_trqs[i]), float(brake_trqs[i]), spin_dt, config)

		var cos_s: float = cos(steer_rad)
		var sin_s: float = sin(steer_rad)
		var fx_c: float = w.F_x * cos_s - w.F_y * sin_s
		var fy_c: float = w.F_x * sin_s + w.F_y * cos_s
		net_fx += fx_c
		net_fy += fy_c
		net_mz += (r_pos.x * fy_c - r_pos.y * fx_c) + w.M_z
		wheel_forces_out.append({"fx": fx_c, "fy": fy_c, "fz": normal_forces[i], "mz": w.M_z})
		wheel_forces_ros.append(Vector3(fx_c, fy_c, normal_forces[i]))
		wheel_moments_ros.append(Vector3(w.M_x, 0.0, w.M_z))
		wheel_steer_angles.append(steer_rad)

	var f_aero: Vector3 = aero_res["force"]
	net_fx += f_aero.x
	net_fy += f_aero.y

	# ffb fisico en columna pa paridad con full_14dof
	if wheels.size() >= 2:
		var tau_sat: float = wheels[0].M_z + wheels[1].M_z
		var tau_trail: float = -(wheels[0].F_y + wheels[1].F_y) * config.mechanical_trail_front * cos(in_inputs.steer_angle)
		var tau_scrub: float = (wheels[1].F_x - wheels[0].F_x) * config.scrub_radius_front
		var sr: float = maxf(config.steering_ratio, 0.1)
		var sw_rad: float = in_inputs.steer_angle * sr
		var sw_dot: float = ((in_inputs.steer_angle - _prev_steer_angle_ffb) * sr / dt) if dt > 1e-6 else 0.0
		_prev_steer_angle_ffb = in_inputs.steer_angle
		var tau_damp: float = -config.ffb_damping_nms_rad * sw_dot
		var sw_max_rad: float = deg_to_rad(maxf(config.fs_steer_wheel_max_deg, 10.0))
		var tau_softlock: float = 0.0
		if absf(sw_rad) > sw_max_rad:
			tau_softlock = -config.ffb_soft_lock_stiffness_nm_rad * (absf(sw_rad) - sw_max_rad) * signf(sw_rad)
		steering_feedback_torque_nm = ((tau_sat + tau_trail + tau_scrub) / sr) + tau_damp + tau_softlock
		var raw_norm: float = clampf(steering_feedback_torque_nm / maxf(config.ffb_max_wheel_torque_nm, 0.5), -1.0, 1.0)
		var alpha_ffb: float = 1.0 - exp(-dt / 0.015) if dt > 0.0 else 1.0
		_ffb_norm_filt = lerpf(_ffb_norm_filt, raw_norm, alpha_ffb)
		steering_feedback_normalized = clampf(_ffb_norm_filt, -1.0, 1.0)

	var mass: float = maxf(config.mass, 100.0)
	_raw_ax_prev = net_fx / mass
	_raw_ay_prev = net_fy / mass
	_prev_ax_ros = _raw_ax_prev
	_prev_ay_ros = _raw_ay_prev

	# salida visual estética de balanceo/cabeceo (sin retroalimentación a fuerzas)
	var kw_f: float = config.spring_stiffness_front * config.motion_ratio_front * config.motion_ratio_front
	var kw_r: float = config.spring_stiffness_rear * config.motion_ratio_rear * config.motion_ratio_rear
	var k_phi_tot: float = 0.5 * kw_f * config.track_width_f * config.track_width_f + 0.5 * kw_r * config.track_width_r * config.track_width_r + config.arb_stiffness_front + config.arb_stiffness_rear
	var h_arm_v: float = maxf(config.h_sprung_cog - 0.5 * (config.roll_center_height_front + config.roll_center_height_rear), 0.05)
	chassis_state.roll = clampf(mass * _ay_filt * h_arm_v / maxf(k_phi_tot, 1000.0), deg_to_rad(-4.5), deg_to_rad(4.5))
	var k_theta_tot: float = (kw_f + kw_r) * config.wheelbase * config.wheelbase * 0.25
	chassis_state.pitch = clampf(mass * _ax_filt * config.h_cog / maxf(k_theta_tot, 2000.0), deg_to_rad(-3.5), deg_to_rad(3.5))

	net_force_chassis_ros = Vector3(net_fx, net_fy, total_fz)
	net_torque_chassis_ros = Vector3(0.0, 0.0, net_mz)
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)

	return {
		"net_force_ros": net_force_chassis_ros,
		"net_torque_ros": net_torque_chassis_ros,
		"normal_forces": normal_forces,
		"wheel_forces": wheel_forces_out,
		"wheel_forces_ros": wheel_forces_ros,
		"wheel_moments_ros": wheel_moments_ros,
		"wheel_steer_angles": wheel_steer_angles,
		"aero_force_ros": f_aero,
		"aero_torque_ros": aero_res.get("torque", Vector3.ZERO),
		"aero": aero_res,
		"drive_torques": eff_drive_trqs,
		"brake_torques": brake_trqs,
	}


func step(in_inputs: VehicleInputs, local_v_ros: Vector3, local_w_ros: Vector3, ground_contacts: Array = [], dt: float = 0.005) -> Dictionary:
	return step_forces(in_inputs, local_v_ros, local_w_ros, ground_contacts, dt, true)


## paso acoplado pa rigidbody3d en sim.tscn: fuerzas planares 3-dof + soporte vertical plano
func step_forces(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	ground_contacts: Array = [],
	dt: float = 0.005,
	advance_tv: bool = true
) -> Dictionary:
	var res := _compute_planar_forces(in_inputs, local_v_ros, local_w_ros, dt, advance_tv, ground_contacts)
	# mantener estabilización vertical min si corre sobre rigidbody3d con raycasts
	if ground_contacts.size() >= 4:
		var fz_tot: float = 0.0
		var mx_tot: float = 0.0
		var my_tot: float = 0.0
		for i in range(4):
			var gc: Dictionary = ground_contacts[i]
			if bool(gc.get("grounded", false)):
				var fz_i: float = float(res["normal_forces"][i])
				var dz: float = float(gc.get("delta_z", 0.0))
				var dz_d: float = float(gc.get("delta_z_dot", 0.0))
				var fz_stab: float = maxf(0.0, fz_i + 18000.0 * dz - 2400.0 * dz_d)
				var rp: Vector3 = gc.get("r_pos_ros", get_wheel_pos_ros(i))
				fz_tot += fz_stab
				mx_tot += rp.y * fz_stab
				my_tot += -rp.x * fz_stab
		net_force_chassis_ros.z = fz_tot
		net_torque_chassis_ros.x = mx_tot
		net_torque_chassis_ros.y = my_tot
		net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
		net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)
		res["net_force_ros"] = net_force_chassis_ros
		res["net_torque_ros"] = net_torque_chassis_ros
	return res


## integrador plano 7-gdl (3-dof chasis + 4 ruedas) con rk2 (heun) a 200 hz
func step_standalone(in_inputs: VehicleInputs, dt: float = 0.005, _road_z: float = 0.0) -> Dictionary:
	if dt <= 0.0:
		var res0 := _compute_planar_forces(in_inputs, chassis_state.linear_velocity, chassis_state.angular_velocity, 0.0, true)
		return {
			"state_6dof": chassis_state,
			"position_ros": chassis_state.position,
			"euler_ros": Vector3(chassis_state.roll, chassis_state.pitch, chassis_state.yaw),
			"linear_velocity_ros": chassis_state.linear_velocity,
			"angular_velocity_ros": chassis_state.angular_velocity,
			"normal_forces": res0["normal_forces"],
			"net_force_ros": res0["net_force_ros"],
			"net_torque_ros": res0["net_torque_ros"],
			"forces": res0,
		}

	var mass: float = maxf(config.mass, 100.0)
	var izz: float = maxf(config.get_total_izz(), 20.0)

	# estado inicial t0
	var x0: float = chassis_state.position.x
	var y0: float = chassis_state.position.y
	var psi0: float = chassis_state.yaw
	var vx0: float = chassis_state.linear_velocity.x
	var vy0: float = chassis_state.linear_velocity.y
	var wz0: float = chassis_state.angular_velocity.z
	var w0: Array[Vector4] = []
	for i in range(4):
		w0.append(wheels[i].get_integrator_state())

	# etapa 1 (euler predictor)
	var res1 := _compute_planar_forces(in_inputs, Vector3(vx0, vy0, 0.0), Vector3(0.0, 0.0, wz0), dt, true)
	var f1: Vector3 = res1["net_force_ros"]
	var t1: Vector3 = res1["net_torque_ros"]
	var ax1: float = f1.x / mass + wz0 * vy0
	var ay1: float = f1.y / mass - wz0 * vx0
	var dwz1: float = t1.z / izz
	var vx1_world: float = vx0 * cos(psi0) - vy0 * sin(psi0)
	var vy1_world: float = vx0 * sin(psi0) + vy0 * cos(psi0)

	var x_p: float = x0 + vx1_world * dt
	var y_p: float = y0 + vy1_world * dt
	var psi_p: float = psi0 + wz0 * dt
	var vx_p: float = vx0 + ax1 * dt
	var vy_p: float = vy0 + ay1 * dt
	var wz_p: float = wz0 + dwz1 * dt
	var w1: Array[Vector4] = []
	for i in range(4):
		w1.append(wheels[i].get_integrator_state())

	# etapa 2 (corrector heun)
	var res2 := _compute_planar_forces(in_inputs, Vector3(vx_p, vy_p, 0.0), Vector3(0.0, 0.0, wz_p), dt, false, [], true)
	var f2: Vector3 = res2["net_force_ros"]
	var t2: Vector3 = res2["net_torque_ros"]
	var ax2: float = f2.x / mass + wz_p * vy_p
	var ay2: float = f2.y / mass - wz_p * vx_p
	var dwz2: float = t2.z / izz
	var vx2_world: float = vx_p * cos(psi_p) - vy_p * sin(psi_p)
	var vy2_world: float = vx_p * sin(psi_p) + vy_p * cos(psi_p)

	chassis_state.position.x = x0 + 0.5 * (vx1_world + vx2_world) * dt
	chassis_state.position.y = y0 + 0.5 * (vy1_world + vy2_world) * dt
	chassis_state.yaw = psi0 + 0.5 * (wz0 + wz_p) * dt
	chassis_state.linear_velocity = Vector3(vx0 + 0.5 * (ax1 + ax2) * dt, vy0 + 0.5 * (ay1 + ay2) * dt, 0.0)
	chassis_state.angular_velocity = Vector3(0.0, 0.0, wz0 + 0.5 * (dwz1 + dwz2) * dt)

	for i in range(4):
		var w_after2: Vector4 = wheels[i].get_integrator_state()
		var w2_step: Vector4 = w0[i] + (w_after2 - w1[i])
		var w_heun: Vector4 = 0.5 * (w1[i] + w2_step)
		wheels[i].set_integrator_state(w_heun)

	# anclaje limpio en parado (sin creep ni rebote)
	var v_planar: float = Vector2(chassis_state.linear_velocity.x, chassis_state.linear_velocity.y).length()
	if in_inputs.throttle < 0.01 and v_planar < 0.30 and (in_inputs.brake > 0.02 or absf(_prev_ax_ros) < 0.45):
		chassis_state.linear_velocity = Vector3.ZERO
		chassis_state.angular_velocity = Vector3.ZERO
		if hv_state != null:
			hv_state.inverter_torques = [0.0, 0.0, 0.0, 0.0]
		for w in wheels:
			w.omega = 0.0
			w.kappa = 0.0
			w.alpha = 0.0
			w.kappa_transient = 0.0
			w.alpha_transient = 0.0

	linear_velocity_ros = chassis_state.linear_velocity
	angular_velocity_ros = chassis_state.angular_velocity
	net_force_chassis_ros = 0.5 * (f1 + f2)
	net_torque_chassis_ros = 0.5 * (t1 + t2)
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)

	return {
		"state_6dof": chassis_state,
		"position_ros": chassis_state.position,
		"euler_ros": Vector3(chassis_state.roll, chassis_state.pitch, chassis_state.yaw),
		"linear_velocity_ros": chassis_state.linear_velocity,
		"angular_velocity_ros": chassis_state.angular_velocity,
		"normal_forces": res2["normal_forces"],
		"net_force_ros": net_force_chassis_ros,
		"net_torque_ros": net_torque_chassis_ros,
		"forces": res2,
	}


func reset() -> void:
	for w in wheels:
		w.reset()
	_sync_wheel_thermal_config()
	if suspension != null:
		suspension.sync_from_config(config)
	if torque_vectoring != null:
		torque_vectoring.reset()
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
	if ecu_tv_v3 != null:
		ecu_tv_v3.configure_from_vehicle_config(config)
		ecu_tv_v3.reset()
	if ecu_tv_4wd != null:
		ecu_tv_4wd.configure_from_vehicle_config(config)
		ecu_tv_4wd.reset()
	if hv_state != null:
		hv_state.reset(config)
	if brake_state != null:
		brake_state.reset(config.disc_ambient_temp_c)
	if chassis_state != null:
		chassis_state.reset(config.h_cog)
	_ax_filt = 0.0
	_ay_filt = 0.0
	_raw_ax_prev = 0.0
	_raw_ay_prev = 0.0
	_prev_ax_ros = 0.0
	_prev_ay_ros = 0.0
	if inputs != null:
		inputs.reset()
		inputs.drive_mode = config.drive_mode
