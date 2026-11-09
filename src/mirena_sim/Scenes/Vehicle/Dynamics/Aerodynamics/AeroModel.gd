class_name AeroModel
extends RefCounted

## Aerodynamics Model for Mirena Dynamics
## Computes drag and downforce forces for the chassis.

const VehicleConfig = preload("../Core/VehicleConfig.gd")

## Computes aerodynamic forces in ISO/ROS frame (+X fwd, +Y left, +Z up).
## Returns Dictionary {"force": Vector3, "torque": Vector3}
static func compute_forces_ros(v_ros: Vector3, config: VehicleConfig) -> Dictionary:
	var speed = v_ros.length()
	if speed < 0.1:
		return {"force": Vector3.ZERO, "torque": Vector3.ZERO}
	
	var q = 0.5 * config.air_density * speed * speed
	
	# Drag opposes velocity vector
	var f_drag = -q * config.cd_a * (v_ros / speed)
	
	# Downforce acts along -Z in ROS frame
	var f_downforce = Vector3(0.0, 0.0, -q * config.cl_a)
	
	return {"force": f_drag + f_downforce, "torque": Vector3.ZERO}

## Computes aerodynamic forces in Godot frame (+X right, +Y up, -Z fwd).
## Returns Dictionary {"drag": Vector3, "downforce": Vector3}
static func compute_forces_godot(linear_velocity: Vector3, chassis_up: Vector3, config: VehicleConfig) -> Dictionary:
	var speed = linear_velocity.length()
	if speed < 0.1:
		return {"drag": Vector3.ZERO, "downforce": Vector3.ZERO}
	
	var q = 0.5 * config.air_density * speed * speed
	var f_drag = -q * config.cd_a * (linear_velocity / speed)
	var f_downforce = -q * config.cl_a * chassis_up
	
	return {"drag": f_drag, "downforce": f_downforce}
