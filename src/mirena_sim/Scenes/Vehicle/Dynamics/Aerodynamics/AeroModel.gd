class_name AeroModel
extends RefCounted

## Aerodynamics Model for Mirena Dynamics (Block D)
## Computes separate drag and front/rear axle downforce components in ISO 8855 (+X fwd, +Y left, +Z up),
## generating true aerodynamic pitching moment M_y = a_cog * F_down_front - b_cog * F_down_rear
## around CoG so front/rear suspensions load directly without double counting,
## with front/rear ground-effect and front-wing stall ride-height sensitivity.

const VehicleConfig = preload("../Core/VehicleConfig.gd")

## Computes ride-height aerodynamic modulation factors for front/rear downforce and drag.
## Negative ride height assumes static design ride height (1.0 nominal factors).
static func compute_ride_height_modulation(z_front: float, z_rear: float, config: VehicleConfig) -> Dictionary:
	if z_front < 0.0 or z_rear < 0.0:
		return {
			"scale_front": 1.0,
			"scale_rear": 1.0,
			"drag_scale": 1.0,
			"front_stalled": false
		}

	var z_f0: float = maxf(config.static_ride_height_front, 0.005)
	var z_r0: float = maxf(config.static_ride_height_rear, 0.005)
	var z_stall: float = clampf(config.aero_stall_ride_height_front, 0.002, z_f0 - 0.002)

	# 1. Front wing ground effect & stall below aero_stall_ride_height_front (18 mm nominal)
	var scale_front: float = 1.0
	var front_stalled: bool = false
	var stall_factor: float = 0.0
	if z_front < z_stall:
		front_stalled = true
		var stall_span: float = maxf(z_stall - 0.008, 1e-4)
		stall_factor = clampf((z_stall - z_front) / stall_span, 0.0, 1.0)
		var smooth_stall: float = stall_factor * stall_factor * (3.0 - 2.0 * stall_factor)
		scale_front = lerpf(1.18, 0.45, smooth_stall)
	elif z_front <= z_f0:
		var t_ge_f: float = clampf((z_f0 - z_front) / maxf(z_f0 - z_stall, 1e-4), 0.0, 1.0)
		scale_front = 1.0 + 0.18 * sin(0.5 * PI * t_ge_f)
	else:
		var t_up_f: float = clampf((z_front - z_f0) / maxf(z_f0, 0.01), 0.0, 1.0)
		scale_front = 1.0 - 0.20 * t_up_f

	# 2. Rear diffuser ground effect + rake modulation (exact 1.0 at static ride height)
	var scale_rear_base: float = 1.0
	if z_rear <= z_r0:
		var z_r_peak: float = maxf(0.015, 0.4 * z_r0)
		var t_ge_r: float = clampf((z_r0 - z_rear) / maxf(z_r0 - z_r_peak, 1e-4), 0.0, 1.0)
		scale_rear_base = 1.0 + 0.22 * sin(0.5 * PI * t_ge_r)
	else:
		var t_up_r: float = clampf((z_rear - z_r0) / maxf(z_r0, 0.01), 0.0, 1.0)
		scale_rear_base = 1.0 - 0.20 * t_up_r

	var delta_rake: float = (z_rear - z_front) - (z_r0 - z_f0)
	var rake_mod: float = clampf(1.5 * delta_rake, -0.05, 0.06)
	var scale_rear: float = clampf(scale_rear_base + rake_mod, 0.70, 1.30)

	# 3. Drag scaling (induced drag from ground-effect suction + separation drag during front stall)
	var avg_lift_scale: float = 0.5 * (scale_front + scale_rear)
	var drag_scale: float = clampf(1.0 + 0.08 * (avg_lift_scale - 1.0) + 0.18 * stall_factor, 0.94, 1.30)

	return {
		"scale_front": scale_front,
		"scale_rear": scale_rear,
		"drag_scale": drag_scale,
		"front_stalled": front_stalled
	}

## Computes aerodynamic forces and pitching moment in ISO/ROS frame (+X fwd, +Y left, +Z up).
## Optional z_front/z_rear ride heights (m) modulate downforce & drag; negative values assume static height.
static func compute_forces_ros(
	v_ros: Vector3,
	config: VehicleConfig,
	z_front: float = -1.0,
	z_rear: float = -1.0
) -> Dictionary:
	var mod: Dictionary = compute_ride_height_modulation(z_front, z_rear, config)
	var scale_f: float = float(mod["scale_front"])
	var scale_r: float = float(mod["scale_rear"])
	var drag_scale: float = float(mod["drag_scale"])
	var cl_a_front_eff: float = config.get_cl_a_front() * scale_f
	var cl_a_rear_eff: float = config.get_cl_a_rear() * scale_r
	var cd_a_eff: float = config.cd_a * drag_scale

	var speed = v_ros.length()
	if speed < 0.1:
		return {
			"force": Vector3.ZERO,
			"torque": Vector3.ZERO,
			"drag": Vector3.ZERO,
			"downforce_front": 0.0,
			"downforce_rear": 0.0,
			"front_stalled": bool(mod["front_stalled"]),
			"cl_a_front_eff": cl_a_front_eff,
			"cl_a_rear_eff": cl_a_rear_eff,
			"cd_a_eff": cd_a_eff
		}
	
	var q: float = 0.5 * config.air_density * speed * speed
	
	# Drag opposes velocity vector
	var f_drag: Vector3 = -q * cd_a_eff * (v_ros / speed)
	
	# Separate front and rear axle downforce magnitudes (N, positive downward)
	var f_down_front: float = q * cl_a_front_eff
	var f_down_rear: float = q * cl_a_rear_eff
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
		"downforce_rear": f_down_rear,
		"front_stalled": bool(mod["front_stalled"]),
		"cl_a_front_eff": cl_a_front_eff,
		"cl_a_rear_eff": cl_a_rear_eff,
		"cd_a_eff": cd_a_eff
	}

## Computes aerodynamic forces in Godot frame (+X right, +Y up, -Z fwd).
## Returns Dictionary {"drag": Vector3, "downforce": Vector3}
static func compute_forces_godot(
	linear_velocity: Vector3,
	chassis_up: Vector3,
	config: VehicleConfig,
	z_front: float = -1.0,
	z_rear: float = -1.0
) -> Dictionary:
	var speed = linear_velocity.length()
	if speed < 0.1:
		return {"drag": Vector3.ZERO, "downforce": Vector3.ZERO}
	
	var mod: Dictionary = compute_ride_height_modulation(z_front, z_rear, config)
	var cl_a_eff: float = config.get_cl_a_front() * float(mod["scale_front"]) + config.get_cl_a_rear() * float(mod["scale_rear"])
	var cd_a_eff: float = config.cd_a * float(mod["drag_scale"])
	var q = 0.5 * config.air_density * speed * speed
	var f_drag = -q * cd_a_eff * (linear_velocity / speed)
	var f_downforce = -q * cl_a_eff * chassis_up
	
	return {"drag": f_drag, "downforce": f_downforce}
