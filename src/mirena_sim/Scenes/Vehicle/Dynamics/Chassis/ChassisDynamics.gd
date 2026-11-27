class_name ChassisDynamics
extends RefCounted

## chassis geometry, mass, coordinate transforms & 6-dof rigid-body dynamics
## frame convention: iso 8855 / ros rep-103 (+x forward, +y left, +z up)
## euler convention: z-y-x (yaw psi around +z, pitch theta around +y', roll phi around +x'')

const VehicleConfig = preload("../../Config/VehicleConfig.gd")

class State6DOF extends RefCounted:
	## inertial cog position (x, y, z) in meters (+z up from ground = 0)
	var position: Vector3 = Vector3(0.0, 0.0, 0.33)
	## roll (phi, + right down), pitch (theta, + nose down), yaw (psi, + ccw left) in radians
	var roll: float = 0.0
	var pitch: float = 0.0
	var yaw: float = 0.0
	## linear velocity (u, v, w) expressed in body iso 8855 frame (m/s)
	var linear_velocity: Vector3 = Vector3.ZERO
	## angular velocity (p, q, r) expressed in body iso 8855 frame (rad/s)
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

## convert vector from godot (+x right, +y up, -z forward) to ros (+x forward, +y left, +z up)
static func godot_to_ros_vec(v: Vector3) -> Vector3:
	return Vector3(-v.z, -v.x, v.y)

## convert vector from ros (+x forward, +y left, +z up) to godot (+x right, +y up, -z forward)
static func ros_to_godot_vec(v: Vector3) -> Vector3:
	return Vector3(-v.y, v.z, -v.x)

## returns static normal load (n) on front and rear axles at rest on level ground
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

## devuelve pos de rueda respecto al cdg in iso/ros frame:
## fl: (+a, +w/2, -h), fr: (+a, -w/2, -h), rl: (-b, +w/2, -h), rr: (-b, -w/2, -h)
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

## body-to-inertial rotation matrix r_wb = r_z(yaw) * r_y(pitch) * r_x(roll)
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

## gravity force (n) expressed in iso 8855 body frame: r_wb^t * (0, 0, -m*g)
static func compute_gravity_body(roll: float, pitch: float, config: VehicleConfig) -> Vector3:
	var mg: float = config.mass * config.gravity
	var cp: float = cos(pitch)
	return Vector3(
		mg * sin(pitch),
		-mg * cp * sin(roll),
		-mg * cp * cos(roll)
	)

## evaluates newton-euler 6-dof rigid-body state derivatives in iso 8855 body frame
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

	# translational newton equation in rotating body frame:
	# m * (v_dot + w x v) = f_total_body + f_grav_body
	var f_grav_b: Vector3 = compute_gravity_body(state.roll, state.pitch, config)
	d.vel_dot = (net_force_body + f_grav_b) / m - w.cross(v)

	# rotational euler equation with sprung/total ixx, iyy, izz and ixz coupling:
	# i * w_dot + w x (i_diag * w) = m_total_body
	var ix: float = maxf(config.ixx, 1e-3)
	var iy: float = maxf(config.iyy, 1e-3)
	var iz: float = maxf(config.get_total_izz(), 1e-3)
	var ixz: float = config.ixz
	var i_w = Vector3(ix * w.x, iy * w.y, iz * w.z)
	var rhs_rot: Vector3 = net_torque_body - w.cross(i_w)
	var det_xz: float = ix * iz - ixz * ixz
	if det_xz > 1e-3:
		d.omega_dot = Vector3(
			(iz * rhs_rot.x + ixz * rhs_rot.z) / det_xz,
			rhs_rot.y / iy,
			(ixz * rhs_rot.x + ix * rhs_rot.z) / det_xz
		)
	else:
		d.omega_dot = Vector3(rhs_rot.x / ix, rhs_rot.y / iy, rhs_rot.z / iz)

	# inertial position kinematics
	var r_wb: Basis = basis_body_to_world(state.roll, state.pitch, state.yaw)
	d.pos_dot = r_wb * v

	# z-y-x euler angle kinematics
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

## advances 6-dof state by dt using explicit euler (also serves as heun predictor step)
static func advance_euler(state: State6DOF, deriv: Derivatives6DOF, dt: float) -> State6DOF:
	var nxt: State6DOF = state.clone()
	nxt.position += deriv.pos_dot * dt
	nxt.roll += deriv.euler_dot.x * dt
	nxt.pitch += deriv.euler_dot.y * dt
	nxt.yaw = wrapf(nxt.yaw + deriv.euler_dot.z * dt, -PI, PI)
	nxt.linear_velocity += deriv.vel_dot * dt
	nxt.angular_velocity += deriv.omega_dot * dt
	return nxt

## advances 6-dof state by dt using 2nd-order runge-kutta (heun / trapezoidal)
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

## computes 4-corner vertical contact kinematics on a flat ground plane at z = ground_height
## from the current 6-dof cog state and body-frame wheel coordinates
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
