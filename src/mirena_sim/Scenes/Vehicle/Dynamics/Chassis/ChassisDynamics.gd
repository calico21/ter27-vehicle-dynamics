class_name ChassisDynamics
extends RefCounted

## Chassis Geometry, Mass, Coordinate Transforms & 6-DOF Rigid-Body Dynamics
## Frame convention: ISO 8855 / ROS REP-103 (+X Forward, +Y Left, +Z Up)
## Euler convention: Z-Y-X (yaw psi around +Z, pitch theta around +Y', roll phi around +X'')

const VehicleConfig = preload("../Core/VehicleConfig.gd")

class State6DOF extends RefCounted:
	## Inertial CoG position (X, Y, Z) in meters (+Z up from ground = 0).
	var position: Vector3 = Vector3(0.0, 0.0, 0.33)
	## Roll (phi, + right down), Pitch (theta, + nose down), Yaw (psi, + CCW left) in radians.
	var roll: float = 0.0
	var pitch: float = 0.0
	var yaw: float = 0.0
	## Linear velocity (u, v, w) expressed in body ISO 8855 frame (m/s).
	var linear_velocity: Vector3 = Vector3.ZERO
	## Angular velocity (p, q, r) expressed in body ISO 8855 frame (rad/s).
	var angular_velocity: Vector3 = Vector3.ZERO

	func _init(h_cog: float = 0.33) -> void:
		position = Vector3(0.0, 0.0, h_cog)

	func reset(h_cog: float = 0.33) -> void:
		position = Vector3(0.0, 0.0, h_cog)
		roll = 0.0
		pitch = 0.0
		yaw = 0.0
		linear_velocity = Vector3.ZERO
		angular_velocity = Vector3.ZERO

	func clone() -> State6DOF:
		var s = State6DOF.new(position.z)
		s.position = position
		s.roll = roll
		s.pitch = pitch
		s.yaw = yaw
		s.linear_velocity = linear_velocity
		s.angular_velocity = angular_velocity
		return s

class Derivatives6DOF extends RefCounted:
	var pos_dot: Vector3 = Vector3.ZERO
	var euler_dot: Vector3 = Vector3.ZERO
	var vel_dot: Vector3 = Vector3.ZERO
	var omega_dot: Vector3 = Vector3.ZERO

## Convert vector from Godot (+X Right, +Y Up, -Z Forward) to ROS (+X Forward, +Y Left, +Z Up).
static func godot_to_ros_vec(v: Vector3) -> Vector3:
	return Vector3(-v.z, -v.x, v.y)

## Convert vector from ROS (+X Forward, +Y Left, +Z Up) to Godot (+X Right, +Y Up, -Z Forward).
static func ros_to_godot_vec(v: Vector3) -> Vector3:
	return Vector3(-v.y, v.z, -v.x)

## Returns static normal load (N) on front and rear axles at rest on level ground.
static func get_static_axle_loads(config: VehicleConfig) -> Dictionary:
	var total_weight = config.mass * config.gravity
	var fz_front_axle = total_weight * (config.b_cog / config.wheelbase)
	var fz_rear_axle = total_weight * (config.a_cog / config.wheelbase)
	return {
		"front_axle": fz_front_axle,
		"rear_axle": fz_rear_axle,
		"wheel_fl": fz_front_axle * 0.5,
		"wheel_fr": fz_front_axle * 0.5,
		"wheel_rl": fz_rear_axle * 0.5,
		"wheel_rr": fz_rear_axle * 0.5
	}

## Returns wheel position relative to CoG in ISO/ROS frame:
## FL: (+a, +w/2, -h), FR: (+a, -w/2, -h), RL: (-b, +w/2, -h), RR: (-b, -w/2, -h)
static func get_wheel_pos_ros(index: int, config: VehicleConfig) -> Vector3:
	var a = config.a_cog
	var b = config.b_cog
	var wf_half = config.track_width_f * 0.5
	var wr_half = config.track_width_r * 0.5
	var h = -config.h_cog
	
	match index:
		VehicleConfig.WheelIndex.FL:
			return Vector3(a, wf_half, h)
		VehicleConfig.WheelIndex.FR:
			return Vector3(a, -wf_half, h)
		VehicleConfig.WheelIndex.RL:
			return Vector3(-b, wr_half, h)
		VehicleConfig.WheelIndex.RR:
			return Vector3(-b, -wr_half, h)
		_:
			return Vector3.ZERO

## Body-to-inertial rotation matrix R_wb = R_z(yaw) * R_y(pitch) * R_x(roll).
static func basis_body_to_world(roll: float, pitch: float, yaw: float) -> Basis:
	var cr: float = cos(roll)
	var sr: float = sin(roll)
	var cp: float = cos(pitch)
	var sp: float = sin(pitch)
	var cy: float = cos(yaw)
	var sy: float = sin(yaw)

	var col_x = Vector3(cy * cp, sy * cp, -sp)
	var col_y = Vector3(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr)
	var col_z = Vector3(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)
	return Basis(col_x, col_y, col_z)

## Gravity force (N) expressed in ISO 8855 body frame: R_wb^T * (0, 0, -m*g).
static func compute_gravity_body(roll: float, pitch: float, config: VehicleConfig) -> Vector3:
	var mg: float = config.mass * config.gravity
	var cp: float = cos(pitch)
	return Vector3(
		mg * sin(pitch),
		-mg * cp * sin(roll),
		-mg * cp * cos(roll)
	)

## Evaluates Newton-Euler 6-DOF rigid-body state derivatives in ISO 8855 body frame.
static func compute_derivatives(
	state: State6DOF,
	net_force_body: Vector3,
	net_torque_body: Vector3,
	config: VehicleConfig
) -> Derivatives6DOF:
	var d = Derivatives6DOF.new()
	var m: float = maxf(config.mass, 1e-3)
	var v: Vector3 = state.linear_velocity
	var w: Vector3 = state.angular_velocity

	# Translational Newton equation in rotating body frame:
	# m * (v_dot + w x v) = F_total_body + F_grav_body
	var f_grav_b: Vector3 = compute_gravity_body(state.roll, state.pitch, config)
	d.vel_dot = (net_force_body + f_grav_b) / m - w.cross(v)

	# Rotational Euler equation for principal-axis inertia diag(Ixx, Iyy, Izz):
	# I * w_dot + w x (I * w) = M_total_body
	var ix: float = maxf(config.ixx, 1e-3)
	var iy: float = maxf(config.iyy, 1e-3)
	var iz: float = maxf(config.izz, 1e-3)
	var i_w = Vector3(ix * w.x, iy * w.y, iz * w.z)
	var rhs_rot: Vector3 = net_torque_body - w.cross(i_w)
	d.omega_dot = Vector3(rhs_rot.x / ix, rhs_rot.y / iy, rhs_rot.z / iz)

	# Inertial position kinematics
	var r_wb: Basis = basis_body_to_world(state.roll, state.pitch, state.yaw)
	d.pos_dot = r_wb * v

	# Z-Y-X Euler angle kinematics
	var theta: float = clampf(state.pitch, -1.55, 1.55)
	var sr: float = sin(state.roll)
	var cr: float = cos(state.roll)
	var cp: float = cos(theta)
	var tp: float = tan(theta)
	var q_sr_r_cr: float = w.y * sr + w.z * cr
	d.euler_dot = Vector3(
		w.x + q_sr_r_cr * tp,
		w.y * cr - w.z * sr,
		q_sr_r_cr / cp
	)
	return d

## Advances 6-DOF state by dt using Explicit Euler (also serves as Heun predictor step).
static func advance_euler(state: State6DOF, deriv: Derivatives6DOF, dt: float) -> State6DOF:
	var nxt: State6DOF = state.clone()
	nxt.position += deriv.pos_dot * dt
	nxt.roll += deriv.euler_dot.x * dt
	nxt.pitch += deriv.euler_dot.y * dt
	nxt.yaw = wrapf(nxt.yaw + deriv.euler_dot.z * dt, -PI, PI)
	nxt.linear_velocity += deriv.vel_dot * dt
	nxt.angular_velocity += deriv.omega_dot * dt
	return nxt

## Advances 6-DOF state by dt using 2nd-order Runge-Kutta (Heun / trapezoidal).
static func advance_heun(state: State6DOF, k1: Derivatives6DOF, k2: Derivatives6DOF, dt: float) -> State6DOF:
	var half_dt: float = 0.5 * dt
	var nxt: State6DOF = state.clone()
	nxt.position += (k1.pos_dot + k2.pos_dot) * half_dt
	nxt.roll += (k1.euler_dot.x + k2.euler_dot.x) * half_dt
	nxt.pitch += (k1.euler_dot.y + k2.euler_dot.y) * half_dt
	nxt.yaw = wrapf(nxt.yaw + (k1.euler_dot.z + k2.euler_dot.z) * half_dt, -PI, PI)
	nxt.linear_velocity += (k1.vel_dot + k2.vel_dot) * half_dt
	nxt.angular_velocity += (k1.omega_dot + k2.omega_dot) * half_dt
	return nxt

## Computes 4-corner vertical contact kinematics on a flat ground plane at z = ground_height
## from the current 6-DOF CoG state and body-frame wheel coordinates.
static func compute_ground_contacts_6dof(
	state: State6DOF,
	config: VehicleConfig,
	ground_height: float = 0.0
) -> Array:
	var contacts: Array = []
	var r_wb: Basis = basis_body_to_world(state.roll, state.pitch, state.yaw)
	var max_droop: float = maxf(0.0, config.ray_cast_length - config.ray_rest_length)

	for i in range(4):
		var r_wheel_body: Vector3 = get_wheel_pos_ros(i, config)
		var p_nom_world: Vector3 = state.position + r_wb * r_wheel_body
		var v_corner_body: Vector3 = state.linear_velocity + state.angular_velocity.cross(r_wheel_body)
		var v_corner_world: Vector3 = r_wb * v_corner_body

		var dz: float = ground_height - p_nom_world.z
		var dz_dot: float = -v_corner_world.z
		var is_g: bool = (dz >= -max_droop)
		var r_contact_body: Vector3 = r_wheel_body + Vector3(0.0, 0.0, dz) if is_g else r_wheel_body

		contacts.append({
			"grounded": is_g,
			"delta_z": dz,
			"delta_z_dot": dz_dot,
			"r_pos_ros": r_contact_body,
			"chassis_roll": state.roll
		})

	return contacts
