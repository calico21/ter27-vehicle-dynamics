class_name SuspensionModel
extends RefCounted

## Suspension & Load Transfer Model for Mirena Dynamics
## Computes spring/damper forces and dynamic axle load transfers.

const VehicleConfig = preload("../Core/VehicleConfig.gd")

## Computes spring and damping force for a single suspension corner.
static func compute_corner_force(compression: float, comp_velocity: float, is_front: bool, config: VehicleConfig) -> float:
	var k_spring = config.spring_stiffness_front if is_front else config.spring_stiffness_rear
	var c_damper = 0.0
	
	if comp_velocity >= 0.0:
		c_damper = config.damping_compression_front if is_front else config.damping_compression_rear
	else:
		c_damper = config.damping_relaxation_front if is_front else config.damping_relaxation_rear
	
	var f_spring = k_spring * compression
	var f_damper = c_damper * comp_velocity
	return maxf(0.0, f_spring + f_damper)

## Analytic dynamic load transfer for unit testing and offline 10-DOF simulation:
## Computes dynamic Fz per wheel [FL, FR, RL, RR] from static weight, ax, ay and aero downforce.
static func compute_analytic_normal_forces(ax_ros: float, ay_ros: float, speed: float, config: VehicleConfig) -> Array[float]:
	var m = config.mass
	var g = config.gravity
	var L = config.wheelbase
	var a = config.a_cog
	var b = config.b_cog
	var h = config.h_cog
	var wf = config.track_width_f
	var wr = config.track_width_r
	
	# Static weight
	var fz_f_static = (m * g * b) / L
	var fz_r_static = (m * g * a) / L
	
	# Longitudinal load transfer: dFz = m * ax * h / L
	var dFz_long = (m * ax_ros * h) / L
	var fz_f_long = maxf(0.0, fz_f_static - dFz_long)
	var fz_r_long = maxf(0.0, fz_r_static + dFz_long)
	
	# Lateral load transfer
	var dFz_lat_f = (m * ay_ros * h * (b / L)) / wf
	var dFz_lat_r = (m * ay_ros * h * (a / L)) / wr
	
	# Aero downforce
	var aero_fz = 0.5 * config.air_density * speed * speed * config.cl_a
	var aero_fz_each = aero_fz * 0.25
	
	var fz_fl = maxf(0.0, fz_f_long * 0.5 - dFz_lat_f + aero_fz_each)
	var fz_fr = maxf(0.0, fz_f_long * 0.5 + dFz_lat_f + aero_fz_each)
	var fz_rl = maxf(0.0, fz_r_long * 0.5 - dFz_lat_r + aero_fz_each)
	var fz_rr = maxf(0.0, fz_r_long * 0.5 + dFz_lat_r + aero_fz_each)
	
	return [fz_fl, fz_fr, fz_rl, fz_rr]
