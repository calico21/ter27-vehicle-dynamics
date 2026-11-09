class_name ChassisDynamics
extends RefCounted

## Chassis Geometry, Mass & Coordinate Dynamics for Mirena Dynamics

const VehicleConfig = preload("../Core/VehicleConfig.gd")

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
