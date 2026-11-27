class_name SuspensionModel
extends RefCounted

## susp & load transfer model for mirena dynamics
## computes spring/damper forces and dynamic axle load transfers

const VehicleConfig = preload("../../Config/VehicleConfig.gd")
const SuspensionSystem = preload("SuspensionSystem.gd")
const AeroModel = preload("../Aerodynamics/AeroModel.gd")

## legacy single-corner helper; delegates wheel-referred stiffness/damping (mr^2) and bumpstops
## to suspensionsystem with optional static preload
static func compute_corner_force(
	compression: float,
	comp_velocity: float,
	is_front: bool,
	config: VehicleConfig,
	static_preload: float = 0.0
) -> float:
	var k_wheel = SuspensionSystem.get_wheel_spring_stiffness(is_front, config)
	var c_wheel = SuspensionSystem.get_wheel_damping(is_front, comp_velocity, config)
	var f_spring = k_wheel * compression
	var f_damper = c_wheel * comp_velocity
	var f_bump = SuspensionSystem.compute_bumpstop_force(compression, is_front, config)
	return maxf(0.0, static_preload + f_spring + f_damper + f_bump)

## transf de carga analitica for unit testing and offline 10-dof simulation:
## computes dynamic fz per wheel [fl, fr, rl, rr] from static weight, ax, ay and aero downforce
static func compute_analytic_normal_forces(
	ax_ros: float,
	ay_ros: float,
	speed: float,
	config: VehicleConfig,
	z_front: float = -1.0,
	z_rear: float = -1.0
) -> Array[float]:
	var m = config.mass
	var g = config.gravity
	var L = config.wheelbase
	var a = config.a_cog
	var b = config.b_cog
	var h = config.h_cog
	var wf = config.track_width_f
	var wr = config.track_width_r
	
	# static weight
	var fz_f_static = (m * g * b) / L
	var fz_r_static = (m * g * a) / L
	
	# longitudinal load transfer: dfz = m * ax * h / l
	var dFz_long = (m * ax_ros * h) / L
	var fz_f_long = maxf(0.0, fz_f_static - dFz_long)
	var fz_r_long = maxf(0.0, fz_r_static + dFz_long)
	
	# lateral load transfer
	var dFz_lat_f = (m * ay_ros * h * (b / L)) / wf
	var dFz_lat_r = (m * ay_ros * h * (a / L)) / wr
	
	# aero downforce split by axle balance (with optional ride-height modulation)
	var mod: Dictionary = AeroModel.compute_ride_height_modulation(z_front, z_rear, config)
	var q = 0.5 * config.air_density * speed * speed
	var aero_fz_f_each = q * config.get_cl_a_front() * float(mod["scale_front"]) * 0.5
	var aero_fz_r_each = q * config.get_cl_a_rear() * float(mod["scale_rear"]) * 0.5
	
	var fz_fl = maxf(0.0, fz_f_long * 0.5 - dFz_lat_f + aero_fz_f_each)
	var fz_fr = maxf(0.0, fz_f_long * 0.5 + dFz_lat_f + aero_fz_f_each)
	var fz_rl = maxf(0.0, fz_r_long * 0.5 - dFz_lat_r + aero_fz_r_each)
	var fz_rr = maxf(0.0, fz_r_long * 0.5 + dFz_lat_r + aero_fz_r_each)
	
	return [fz_fl, fz_fr, fz_rl, fz_rr]
