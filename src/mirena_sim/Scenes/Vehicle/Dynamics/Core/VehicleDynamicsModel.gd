class_name VehicleDynamicsModel
extends RefCounted

## Vehicle Dynamics Model (10-DOF Core) for Mirena Dynamics
## Fully decoupled coordinator for physical & mathematical calculations.
##
## Internal coordinate convention: ISO 8855 / ROS REP-103
##   +X: Forward, +Y: Left, +Z: Up

const VehicleConfig = preload("VehicleConfig.gd")
const VehicleInputs = preload("VehicleInputs.gd")
const WheelDynamics = preload("../Tires/WheelDynamics.gd")
const DrivetrainModel = preload("../Drivetrain/DrivetrainModel.gd")
const TorqueVectoring = preload("../Drivetrain/TorqueVectoring.gd")
const EcuTorqueVectoringV3 = preload("../Drivetrain/EcuTorqueVectoringV3.gd")
const SuspensionModel = preload("../Suspension/SuspensionModel.gd")
const SuspensionCorner = preload("../Suspension/SuspensionCorner.gd")
const SuspensionSystem = preload("../Suspension/SuspensionSystem.gd")
const AeroModel = preload("../Aerodynamics/AeroModel.gd")
const ChassisDynamics = preload("../Chassis/ChassisDynamics.gd")
const BrakeModel = preload("../Brakes/BrakeModel.gd")

var config: VehicleConfig
var wheels: Array[WheelDynamics] = []
var suspension: SuspensionSystem
var torque_vectoring: TorqueVectoring
var ecu_tv_v3: EcuTorqueVectoringV3
var hv_state: DrivetrainModel.HVPowertrainState
var brake_state: BrakeModel.BrakeState
var _prev_ax_ros: float = 0.0
var chassis_state: ChassisDynamics.State6DOF
var inputs: VehicleInputs
var _last_input_tv_map_id: int = 2

# --- Output Net Quantities (in ISO/ROS chassis frame around CoG) ---
var net_force_chassis_ros: Vector3 = Vector3.ZERO
var net_torque_chassis_ros: Vector3 = Vector3.ZERO

# --- Output Net Quantities (converted to Godot frame for RigidBody3D) ---
var net_force_chassis_godot: Vector3 = Vector3.ZERO
var net_torque_chassis_godot: Vector3 = Vector3.ZERO

# --- Chassis States (in ISO/ROS chassis frame) ---
var linear_velocity_ros: Vector3 = Vector3.ZERO
var angular_velocity_ros: Vector3 = Vector3.ZERO
var _prev_fy_front: float = 0.0
var _prev_fy_rear: float = 0.0
var _prev_ay_ros: float = 0.0

func _init(cfg: VehicleConfig = null) -> void:
	if cfg != null:
		config = cfg
	else:
		config = VehicleConfig.new()
	
	inputs = VehicleInputs.new()
	_last_input_tv_map_id = inputs.tv_map_id
	suspension = SuspensionSystem.new()
	torque_vectoring = TorqueVectoring.new()
	ecu_tv_v3 = EcuTorqueVectoringV3.new()
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
	var init_t: float = config.tire_initial_temp_c
	var track_t: float = config.track_temp_c
	for w in wheels:
		w.set_tire_temperature(init_t)
		w.t_track_c = track_t

## Computes front wheel steer angle (ISO +Z CCW left) with configurable Ackermann geometry.
static func compute_ackermann_steer(steer_angle: float, is_left: bool, config: VehicleConfig) -> float:
	return SuspensionSystem.compute_ackermann_steer(steer_angle, is_left, config)

## Returns wheel position relative to CoG in ISO/ROS frame:
func get_wheel_pos_ros(index: int) -> Vector3:
	return ChassisDynamics.get_wheel_pos_ros(index, config)

## Computes 4WD motor drive torques and friction brake torques per wheel.
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

## Computes aerodynamic drag and downforce forces in ISO/ROS frame.
func compute_aerodynamics(v_ros: Vector3, z_front: float = -1.0, z_rear: float = -1.0) -> Dictionary:
	return AeroModel.compute_forces_ros(v_ros, config, z_front, z_rear)

## Analytic dynamic load transfer for unit testing and offline simulation:
func compute_analytic_normal_forces(ax_ros: float, ay_ros: float, speed: float) -> Array[float]:
	return SuspensionModel.compute_analytic_normal_forces(ax_ros, ay_ros, speed, config)

## Backward-compatible wrapper around step_forces() for external callers & tests.
func step(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	ground_contacts: Array = [],
	dt: float = 0.0025
) -> Dictionary:
	return step_forces(in_inputs, local_v_ros, local_w_ros, ground_contacts, dt, true)

## Shared force, suspension, slip, tire & aero evaluation path used by both
## MirenaCarDynamics (Godot RigidBody3D) and step_standalone (14-DOF headless).
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
	
	# 1. Drivetrain, 200 Hz Torque Vectoring, HV Bus & Inverter Lag
	if ecu_tv_v3 != null and in_inputs.tv_map_id != _last_input_tv_map_id:
		if in_inputs.tv_map_id != ecu_tv_v3.active_map_id:
			ecu_tv_v3.set_map(in_inputs.tv_map_id)
		_last_input_tv_map_id = ecu_tv_v3.active_map_id

	var w_omegas: Array[float] = [wheels[0].omega, wheels[1].omega, wheels[2].omega, wheels[3].omega]
	var drive_trqs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var brake_trqs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if in_inputs.ecu_driving_mode == 1 and in_inputs.drive_mode == VehicleConfig.DriveMode.FOUR_WD and not in_inputs.override_wheel_torques:
		var r_whl: float = maxf(config.wheel_radius, 1e-4)
		var max_whl_trq: float = config.motor_peak_torque * config.gear_ratio
		var spd_clamped: float = maxf(absf(u), 1.0)
		var max_pwr_force: float = (config.power_limit_watts * config.drivetrain_efficiency) / spd_clamped
		var max_fx_4wd: float = minf(4.0 * max_whl_trq / r_whl, max_pwr_force)
		var fx_driver: float = in_inputs.throttle * max_fx_4wd
		var w_steers: Array[float] = [
			compute_ackermann_steer(in_inputs.steer_angle, true, config),
			compute_ackermann_steer(in_inputs.steer_angle, false, config),
			0.0,
			0.0
		]
		var tv_trqs: Array[float] = (
			ecu_tv_v3.step(fx_driver, in_inputs.steer_angle, u, v, r, _prev_ay_ros, _prev_ax_ros, w_omegas, in_inputs.brake, dt, w_steers)
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
	
	# 2. Determine Normal Forces Fz and Suspension Alignment (zero load-transfer double counting)
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
			var dz_arr: Array[float] = [0.0, 0.0, 0.0, 0.0]
			var dz_dot_arr: Array[float] = [0.0, 0.0, 0.0, 0.0]
			var g_arr: Array[bool] = [false, false, false, false]
			for i in range(4):
				var c: Dictionary = ground_contacts[i]
				g_arr[i] = bool(c.get("grounded", true))
				dz_arr[i] = float(c.get("delta_z", 0.0))
				dz_dot_arr[i] = float(c.get("delta_z_dot", 0.0))
				corner_dz[i] = dz_arr[i]
			suspension.evaluate_from_corners(dz_arr, dz_dot_arr, g_arr, config)
			suspension.apply_non_elastic_lateral_load_transfer(_prev_fy_front, _prev_fy_rear, _prev_ay_ros, config)
			var corners: Array[SuspensionCorner] = suspension.corners
			for i in range(4):
				var sc: SuspensionCorner = corners[i]
				wheels[i].is_grounded = sc.is_grounded
				toe_offsets[i] = sc.toe_rad
				normal_forces[i] = sc.fz
		else:
			for i in range(4):
				var c = ground_contacts[i]
				wheels[i].is_grounded = c.get("grounded", true)
				normal_forces[i] = c.get("fz", 0.0)
				toe_offsets[i] = SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	else:
		var ax_approx = (drive_trqs[0] + drive_trqs[1] + drive_trqs[2] + drive_trqs[3]) / (config.mass * config.wheel_radius)
		normal_forces = compute_analytic_normal_forces(ax_approx, u * r, local_v_ros.length())
		for i in range(4):
			wheels[i].is_grounded = true
			toe_offsets[i] = SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	
	# 2b. Blended Hydraulic + Regenerative Braking & 1-Node Disc Thermal Update
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
		for i in range(4):
			brake_trqs[i] = float(blended_trqs[i])

	# 3. Kinematics, Slips, Forces and Wheel Spin per Wheel
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
				w.surface_mu_scale = float(ground_contacts[i]["surface_mu"])
			else:
				w.surface_mu_scale = 1.0
		else:
			w.surface_mu_scale = 1.0
		var v_hub = local_v_ros + local_w_ros.cross(r_pos)
		w.v_hub_chassis = v_hub
		
		var base_steer: float = 0.0
		if i == VehicleConfig.WheelIndex.FL:
			base_steer = compute_ackermann_steer(steer_angle, true, config)
		elif i == VehicleConfig.WheelIndex.FR:
			base_steer = compute_ackermann_steer(steer_angle, false, config)
		var delta_comp: float = SuspensionSystem.compute_steering_compliance(w.M_z, w.F_y, config) if i < 2 else 0.0
		var delta_w = base_steer + toe_offsets[i] + delta_comp
		wheel_steer_angles.append(delta_w)
		
		w.camber_angle = SuspensionSystem.compute_ground_camber_iso(
			corner_dz[i], delta_w, chassis_roll_rad, i < 2, w.is_right_side, config
		)
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
	
	# 3b. 3D Wishbone Anti-Dive / Anti-Squat geometric pitch reaction (preserves sum(Fz))
	var fx_corners: Array = [
		wheel_forces_ros[0].x,
		wheel_forces_ros[1].x,
		wheel_forces_ros[2].x,
		wheel_forces_ros[3].x
	]
	var anti_pitch: Dictionary = SuspensionSystem.compute_anti_pitch_forces(fx_corners, config)
	total_wheel_torque_ros.y += float(anti_pitch["pitch_torque_ros"])

	# 4. Aerodynamics (ride-height sensitive ground effect & front-wing stall)
	var z_front_dyn: float = -1.0
	var z_rear_dyn: float = -1.0
	if has_corner_travel:
		z_front_dyn = maxf(0.0, config.static_ride_height_front - 0.5 * (corner_dz[0] + corner_dz[1]))
		z_rear_dyn = maxf(0.0, config.static_ride_height_rear - 0.5 * (corner_dz[2] + corner_dz[3]))
	var aero = compute_aerodynamics(local_v_ros, z_front_dyn, z_rear_dyn)
	var f_aero_ros: Vector3 = aero["force"]
	var tau_aero_ros: Vector3 = aero["torque"]
	
	# 5. Net Force & Torque on Chassis (ROS Frame)
	net_force_chassis_ros = total_wheel_force_ros + f_aero_ros
	net_torque_chassis_ros = total_wheel_torque_ros + tau_aero_ros
	_prev_fy_front = wheel_forces_ros[0].y + wheel_forces_ros[1].y
	_prev_fy_rear = wheel_forces_ros[2].y + wheel_forces_ros[3].y
	_prev_ax_ros = net_force_chassis_ros.x / maxf(config.mass, 1e-3)
	_prev_ay_ros = net_force_chassis_ros.y / maxf(config.mass, 1e-3)
	
	# Convert net quantities to Godot frame:
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)

	var tire_temps: Array[float] = [wheels[0].t_surf_c, wheels[1].t_surf_c, wheels[2].t_surf_c, wheels[3].t_surf_c]
	
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
		"mz_req": ecu_tv_v3.last_mz_req if (ecu_tv_v3 != null and in_inputs.ecu_driving_mode == 1) else 0.0,
		"wz_ref": ecu_tv_v3.last_wz_ref if ecu_tv_v3 != null else 0.0,
		"z_front": z_front_dyn,
		"z_rear": z_rear_dyn,
		"aero_front_stalled": bool(aero.get("front_stalled", false)),
		"disc_temps": brake_state.disc_temp.duplicate() if brake_state != null else [40.0, 40.0, 40.0, 40.0],
		"regen_power_kw": (brake_state.p_regen_w / 1000.0) if brake_state != null else 0.0,
		"surface_mu": [wheels[0].surface_mu_scale, wheels[1].surface_mu_scale, wheels[2].surface_mu_scale, wheels[3].surface_mu_scale]
	}

## Unified 14-DOF standalone step (6-DOF Rigid-Body Chassis via RK2 Heun + 4 Wheels + 4 Suspension Corners).
## Shares the exact same step_forces() evaluation path used by MirenaCarDynamics.gd.
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

	# RK2 Stage 1 (evaluate at t_n; advance 200 Hz TV accumulator once per physical step)
	var contacts_1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, config, ground_height)
	var res_1: Dictionary = step_forces(in_inputs, s0.linear_velocity, s0.angular_velocity, contacts_1, dt, true)
	var k1: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s0, res_1["net_force_ros"], res_1["net_torque_ros"], config)
	var s_pred: ChassisDynamics.State6DOF = ChassisDynamics.advance_euler(s0, k1, dt)

	# RK2 Stage 2 (evaluate at predictor t_n + dt; reuse latched TV command)
	var contacts_2: Array = ChassisDynamics.compute_ground_contacts_6dof(s_pred, config, ground_height)
	var res_2: Dictionary = step_forces(in_inputs, s_pred.linear_velocity, s_pred.angular_velocity, contacts_2, dt, false)
	var k2: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s_pred, res_2["net_force_ros"], res_2["net_torque_ros"], config)

	# Heun update for 6-DOF chassis and 4 wheels
	chassis_state = ChassisDynamics.advance_heun(s0, k1, k2, dt)
	var all_wheels_stopped: bool = true
	for i in range(4):
		var w2_step: Vector4 = wheels[i].get_integrator_state()
		var w_heun: Vector4 = 0.5 * (w0[i] + w2_step)
		# Preserve crisp zero-crossing brake hold at standstill
		if is_zero_approx(w2_step.x) and in_inputs.brake > 0.0 and in_inputs.throttle == 0.0:
			w_heun.x = 0.0
		if absf(w_heun.x) > 1e-3:
			all_wheels_stopped = false
		wheels[i].set_integrator_state(w_heun)

	# Prevent transient tire carcass deflection from rebounding the car backward past zero under brakes
	if in_inputs.brake > 0.0 and in_inputs.throttle == 0.0:
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
	torque_vectoring.reset()
	if ecu_tv_v3 != null:
		ecu_tv_v3.reset()
	if hv_state != null:
		hv_state.reset(config)
	chassis_state.reset(config.h_cog)
	inputs.reset()
	_last_input_tv_map_id = inputs.tv_map_id
	net_force_chassis_ros = Vector3.ZERO
	net_torque_chassis_ros = Vector3.ZERO
	net_force_chassis_godot = Vector3.ZERO
	net_torque_chassis_godot = Vector3.ZERO
	linear_velocity_ros = Vector3.ZERO
	angular_velocity_ros = Vector3.ZERO
	_prev_fy_front = 0.0
	_prev_fy_rear = 0.0
	_prev_ax_ros = 0.0
	_prev_ay_ros = 0.0
