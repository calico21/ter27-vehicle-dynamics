class_name AeroModel
extends RefCounted

## Aerodynamics Model for Mirena Dynamics (Block D)
## Computes separate drag and front/rear axle downforce components in ISO 8855 (+X fwd, +Y left, +Z up),
## generating true aerodynamic pitching moment M_y = a_cog * F_down_front - b_cog * F_down_rear
## around CoG so front/rear suspensions load directly without double counting.

const VehicleConfig = preload("../Core/VehicleConfig.gd")

## Computes aerodynamic forces and pitching moment in ISO/ROS frame (+X fwd, +Y left, +Z up).
## Returns Dictionary {"force": Vector3, "torque": Vector3, "drag": Vector3, "downforce_front": float, "downforce_rear": float}
static func compute_forces_ros(v_ros: Vector3, config: VehicleConfig) -> Dictionary:
	var speed = v_ros.length()
	if speed < 0.1:
		return {
			"force": Vector3.ZERO,
			"torque": Vector3.ZERO,
			"drag": Vector3.ZERO,
			"downforce_front": 0.0,
			"downforce_rear": 0.0
		}
	
	var q: float = 0.5 * config.air_density * speed * speed
	
	# Drag opposes velocity vector
	var f_drag: Vector3 = -q * config.cd_a * (v_ros / speed)
	
	# Separate front and rear axle downforce magnitudes (N, positive downward)
	var f_down_front: float = q * config.get_cl_a_front()
	var f_down_rear: float = q * config.get_cl_a_rear()
	var f_downforce_vec: Vector3 = Vector3(0.0, 0.0, -(f_down_front + f_down_rear))
	
	# Pitching moment around CoG from front (+a_cog), rear (-b_cog) downforce and drag at cp_height:
	# (r x F)_y = r_z * F_x - r_x * F_z = (cp_height - h_cog) * F_drag_x + a_cog * F_down_front - b_cog * F_down_rear
	var pitch_moment_y: float = config.a_cog * f_down_front - config.b_cog * f_down_rear + (config.cp_height - config.h_cog) * f_drag.x
	var torque_ros: Vector3 = Vector3(0.0, pitch_moment_y, 0.0)
	
	return {
		"force": f_drag + f_downforce_vec,
		"torque": torque_ros,
		"drag": f_drag,
		"downforce_front": f_down_front,
		"downforce_rear": f_down_rear
	}

## Computes aerodynamic forces in Godot frame (+X right, +Y up, -Z fwd).
## Returns Dictionary {"drag": Vector3, "downforce": Vector3}
static func compute_forces_godot(linear_velocity: Vector3, chassis_up: Vector3, config: VehicleConfig) -> Dictionary:
	var speed = linear_velocity.length()
	if speed < 0.1:
		return {"drag": Vector3.ZERO, "downforce": Vector3.ZERO}
	
	var q = 0.5 * config.air_density * speed * speed
	var f_drag = -q * config.cd_a * (linear_velocity / speed)
	var f_downforce = -q * (config.get_cl_a_front() + config.get_cl_a_rear()) * chassis_up
	
	return {"drag": f_drag, "downforce": f_downforce}
