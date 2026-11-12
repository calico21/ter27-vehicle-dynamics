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
const SuspensionModel = preload("../Suspension/SuspensionModel.gd")
const SuspensionCorner = preload("../Suspension/SuspensionCorner.gd")
const SuspensionSystem = preload("../Suspension/SuspensionSystem.gd")
const AeroModel = preload("../Aerodynamics/AeroModel.gd")
const ChassisDynamics = preload("../Chassis/ChassisDynamics.gd")

var config: VehicleConfig
var wheels: Array[WheelDynamics] = []
var suspension: SuspensionSystem
var chassis_state: ChassisDynamics.State6DOF
var inputs: VehicleInputs

# --- Output Net Quantities (in ISO/ROS chassis frame around CoG) ---
var net_force_chassis_ros: Vector3 = Vector3.ZERO
var net_torque_chassis_ros: Vector3 = Vector3.ZERO

# --- Output Net Quantities (converted to Godot frame for RigidBody3D) ---
var net_force_chassis_godot: Vector3 = Vector3.ZERO
var net_torque_chassis_godot: Vector3 = Vector3.ZERO

# --- Chassis States (in ISO/ROS chassis frame) ---
var linear_velocity_ros: Vector3 = Vector3.ZERO
var angular_velocity_ros: Vector3 = Vector3.ZERO

func _init(cfg: VehicleConfig = null) -> void:
	if cfg != null:
		config = cfg
	else:
		config = VehicleConfig.new()
	
	inputs = VehicleInputs.new()
	suspension = SuspensionSystem.new()
	chassis_state = ChassisDynamics.State6DOF.new(config.h_cog)
	_init_wheels()

func _init_wheels() -> void:
	wheels.clear()
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.FL, "FL", false))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.FR, "FR", true))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.RL, "RL", false))
	wheels.append(WheelDynamics.new(VehicleConfig.WheelIndex.RR, "RR", true))

## Returns wheel position relative to CoG in ISO/ROS frame:
func get_wheel_pos_ros(index: int) -> Vector3:
	return ChassisDynamics.get_wheel_pos_ros(index, config)

## Computes 4WD motor drive torques and friction brake torques per wheel.
func compute_drivetrain_torques(in_inputs: VehicleInputs, u_speed: float) -> Dictionary:
	return DrivetrainModel.compute_torques(in_inputs, u_speed, config)

## Computes aerodynamic drag and downforce forces in ISO/ROS frame.
func compute_aerodynamics(v_ros: Vector3) -> Dictionary:
	return AeroModel.compute_forces_ros(v_ros, config)

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
	return step_forces(in_inputs, local_v_ros, local_w_ros, ground_contacts, dt)

## Shared force, suspension, slip, tire & aero evaluation path used by both
## MirenaCarDynamics (Godot RigidBody3D) and step_standalone (14-DOF headless).
func step_forces(
	in_inputs: VehicleInputs,
	local_v_ros: Vector3,
	local_w_ros: Vector3,
	ground_contacts: Array = [],
	dt: float = 0.0025
) -> Dictionary:
	linear_velocity_ros = local_v_ros
	angular_velocity_ros = local_w_ros
	inputs = in_inputs
	
	var u = local_v_ros.x
	var v = local_v_ros.y
	var r = local_w_ros.z
	
	# 1. Drivetrain & Brakes
	var torques = compute_drivetrain_torques(in_inputs, u)
	var drive_trqs: Array[float] = torques["drive"]
	var brake_trqs: Array[float] = torques["brake"]
	
	# 2. Determine Normal Forces Fz and Suspension Alignment (zero load-transfer double counting)
	var normal_forces: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var toe_offsets: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if ground_contacts.size() >= 4:
		var has_corner_travel: bool = (ground_contacts[0] is Dictionary) and ground_contacts[0].has("delta_z")
		if has_corner_travel:
			var dz_arr: Array[float] = [0.0, 0.0, 0.0, 0.0]
			var dz_dot_arr: Array[float] = [0.0, 0.0, 0.0, 0.0]
			var g_arr: Array[bool] = [false, false, false, false]
			for i in range(4):
				var c: Dictionary = ground_contacts[i]
				g_arr[i] = bool(c.get("grounded", true))
				dz_arr[i] = float(c.get("delta_z", 0.0))
				dz_dot_arr[i] = float(c.get("delta_z_dot", 0.0))
			var corners: Array[SuspensionCorner] = suspension.evaluate_from_corners(dz_arr, dz_dot_arr, g_arr, config)
			for i in range(4):
				var sc: SuspensionCorner = corners[i]
				wheels[i].is_grounded = sc.is_grounded
				wheels[i].camber_angle = sc.camber_rad
				toe_offsets[i] = sc.toe_rad
				normal_forces[i] = sc.fz
		else:
			for i in range(4):
				var c = ground_contacts[i]
				wheels[i].is_grounded = c.get("grounded", true)
				normal_forces[i] = c.get("fz", 0.0)
				wheels[i].camber_angle = SuspensionSystem.compute_corner_camber_iso(0.0, i < 2, wheels[i].is_right_side, config)
				toe_offsets[i] = SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	else:
		var ax_approx = (drive_trqs[0] + drive_trqs[1] + drive_trqs[2] + drive_trqs[3]) / (config.mass * config.wheel_radius)
		normal_forces = compute_analytic_normal_forces(ax_approx, u * r, local_v_ros.length())
		for i in range(4):
			wheels[i].is_grounded = true
			wheels[i].camber_angle = SuspensionSystem.compute_corner_camber_iso(0.0, i < 2, wheels[i].is_right_side, config)
			toe_offsets[i] = SuspensionSystem.compute_corner_toe_iso(0.0, i < 2, wheels[i].is_right_side, config)
	
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
		var v_hub = local_v_ros + local_w_ros.cross(r_pos)
		w.v_hub_chassis = v_hub
		
		var base_steer = steer_angle if (i == VehicleConfig.WheelIndex.FL or i == VehicleConfig.WheelIndex.FR) else 0.0
		var delta_w = base_steer + toe_offsets[i]
		wheel_steer_angles.append(delta_w)
		
		w.compute_slip(v_hub.x, v_hub.y, delta_w, config.slip_regularization_speed, config.wheel_radius, dt)
		w.compute_forces(normal_forces[i], config)
		w.integrate_wheel_spin(drive_trqs[i], brake_trqs[i], dt, config)
		
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
	
	# 4. Aerodynamics
	var aero = compute_aerodynamics(local_v_ros)
	var f_aero_ros: Vector3 = aero["force"]
	var tau_aero_ros: Vector3 = aero["torque"]
	
	# 5. Net Force & Torque on Chassis (ROS Frame)
	net_force_chassis_ros = total_wheel_force_ros + f_aero_ros
	net_torque_chassis_ros = total_wheel_torque_ros + tau_aero_ros
	
	# Convert net quantities to Godot frame:
	net_force_chassis_godot = VehicleConfig.ros_to_godot_vec(net_force_chassis_ros)
	net_torque_chassis_godot = VehicleConfig.ros_to_godot_vec(net_torque_chassis_ros)
	
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
		"wheel_steer_angles": wheel_steer_angles
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

	# RK2 Stage 1 (evaluate at t_n)
	var contacts_1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, config, ground_height)
	var res_1: Dictionary = step_forces(in_inputs, s0.linear_velocity, s0.angular_velocity, contacts_1, dt)
	var k1: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s0, res_1["net_force_ros"], res_1["net_torque_ros"], config)
	var s_pred: ChassisDynamics.State6DOF = ChassisDynamics.advance_euler(s0, k1, dt)

	# RK2 Stage 2 (evaluate at predictor t_n + dt; wheels advance from w1 -> w0 + dt*(k1_w + k2_w))
	var contacts_2: Array = ChassisDynamics.compute_ground_contacts_6dof(s_pred, config, ground_height)
	var res_2: Dictionary = step_forces(in_inputs, s_pred.linear_velocity, s_pred.angular_velocity, contacts_2, dt)
	var k2: ChassisDynamics.Derivatives6DOF = ChassisDynamics.compute_derivatives(s_pred, res_2["net_force_ros"], res_2["net_torque_ros"], config)

	# Heun update for 6-DOF chassis and 4 wheels
	chassis_state = ChassisDynamics.advance_heun(s0, k1, k2, dt)
	for i in range(4):
		var w2_step: Vector4 = wheels[i].get_integrator_state()
		var w_heun: Vector4 = 0.5 * (w0[i] + w2_step)
		# Preserve crisp zero-crossing brake hold at standstill
		if is_zero_approx(w2_step.x) and in_inputs.brake > 0.0 and in_inputs.throttle == 0.0:
			w_heun.x = 0.0
		wheels[i].set_integrator_state(w_heun)

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
	suspension.reset()
	chassis_state.reset(config.h_cog)
	inputs.reset()
	net_force_chassis_ros = Vector3.ZERO
	net_torque_chassis_ros = Vector3.ZERO
	net_force_chassis_godot = Vector3.ZERO
	net_torque_chassis_godot = Vector3.ZERO
	linear_velocity_ros = Vector3.ZERO
	angular_velocity_ros = Vector3.ZERO
