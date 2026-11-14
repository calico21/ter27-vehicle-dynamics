class_name WheelDynamics
extends RefCounted

## Individual Wheel Dynamics & Contact Model for Mirena Dynamics
## Represents a single wheel with slip kinematics, transient carcass relaxation,
## hybrid tire contact forces (MF 6.2 + Neural Residual), and rotational spin.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const TireModel = preload("TireModel.gd")
const Pacejka62 = preload("MF62/Pacejka62.gd")

const THERM_INIT_C: float = 55.0
const THERM_TRACK_DEFAULT_C: float = 28.0
const THERM_ETA_FRICTION: float = 0.65
const THERM_C_ZONE_J_K: float = 420.0
const THERM_C_CORE_J_K: float = 4200.0
const THERM_G_CORE_ZONE_W_K: float = 9.0
const THERM_G_LAT_ZONE_W_K: float = 5.0
const THERM_G_TRACK_NOM_W_K: float = 18.0
const THERM_G_CORE_AIR_W_K: float = 2.0

var wheel_id: int = 0
var name: String = "Wheel"
var is_right_side: bool = false

# --- Wheel Kinematics State ---
## Rotational speed around wheel spindle axis (rad/s).
var omega: float = 0.0

## Accumulated rotation angle (rad).
var rotation_angle: float = 0.0

## Steer angle around suspension vertical axis (rad).
var steer_angle: float = 0.0

## Hub linear velocity expressed in chassis frame (m/s).
var v_hub_chassis: Vector3 = Vector3.ZERO

## Contact velocity expressed along wheel heading and lateral directions (m/s).
var v_longitudinal: float = 0.0
var v_lateral: float = 0.0

# --- Slip Quantities ---
## Steady-state kinematic longitudinal slip ratio: kappa = (omega * R - vx) / max(|vx|, eps)
var kappa: float = 0.0

## Steady-state kinematic lateral slip angle: alpha = -atan2(vy, |vx|) (rad)
var alpha: float = 0.0

## Transient (carcass-deflected) longitudinal slip ratio
var kappa_transient: float = 0.0

## Transient (carcass-deflected) lateral slip angle (rad)
var alpha_transient: float = 0.0

## Wheel camber / inclination angle (rad)
var camber_angle: float = 0.0

## Tire inflation pressure (kPa)
var inflation_pressure_kpa: float = 83.0

# --- Contact Forces in Wheel Frame ---
## Normal contact load perpendicular to road surface (N).
var F_z: float = 0.0

## Longitudinal tractive / braking force in wheel heading direction (N).
var F_x: float = 0.0

## Lateral cornering force perpendicular to wheel heading direction (N).
var F_y: float = 0.0

## Self-aligning torque around vertical axis (Nm).
var M_z: float = 0.0

## Overturning moment around longitudinal contact axis (Nm).
var M_x: float = 0.0

## Rolling resistance moment around wheel spindle (Nm).
var M_y: float = 0.0

# --- Diagnostics ---
var delta_Fx: float = 0.0
var delta_Fy: float = 0.0
var delta_Mz: float = 0.0

# --- Suspension & Tire Radial Deflection State ---
var is_grounded: bool = true
var suspension_compression: float = 0.0 # m
var suspension_velocity: float = 0.0    # m/s
var radial_deflection: float = 0.0      # rho_i = max(0, Fz / kz) (m)
var loaded_radius: float = 0.2032       # R_l = R0 - rho_i (m)
var effective_radius: float = 0.2032    # R_e = R0 - rho_i / 3 (m)
var ride_rate: float = 0.0              # k_ride = (k_wheel * kz) / (k_wheel + kz) (N/m)

# --- 3-Zone Tire Thermal State (Surface Tread + Core Carcass) ---
var t_surf_c: float = THERM_INIT_C
var t_core_c: float = THERM_INIT_C
var t_zone_inner_c: float = THERM_INIT_C
var t_zone_mid_c: float = THERM_INIT_C
var t_zone_outer_c: float = THERM_INIT_C
var t_track_c: float = THERM_TRACK_DEFAULT_C
var thermal_mu_factor: float = 1.0
var surface_mu_scale: float = 1.0
var slip_power_w: float = 0.0

static func _cfg_float(config: VehicleConfig, prop: String, default_val: float) -> float:
	if config == null:
		return default_val
	var v: Variant = config.get(prop)
	return float(v) if (typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT) else default_val

func _init(id: int = 0, wheel_name: String = "Wheel", right_side: bool = false) -> void:
	wheel_id = id
	name = wheel_name
	is_right_side = right_side

## Computes slip kinematics and transient carcass relaxation.
func compute_slip(
	vx_chassis: float,
	vy_chassis: float,
	steer_rad: float,
	v_eps: float,
	wheel_radius: float,
	dt: float = 0.0025
) -> void:
	steer_angle = steer_rad
	var cos_s = cos(steer_angle)
	var sin_s = sin(steer_angle)
	
	# Transform velocity from chassis frame into wheel frame
	# (+X_w steered forward, +Y_w steered left)
	v_longitudinal = vx_chassis * cos_s + vy_chassis * sin_s
	v_lateral = -vx_chassis * sin_s + vy_chassis * cos_s
	
	var u_denom = maxf(absf(v_longitudinal), v_eps)
	
	# Longitudinal slip ratio: kappa = (omega * R - vx) / max(|vx|, eps)
	kappa = (omega * wheel_radius - v_longitudinal) / u_denom
	
	# Lateral slip angle: alpha = -atan2(vy, |vx|)
	alpha = -atan2(v_lateral, u_denom)
	
	# First-order sub-stepped transient carcass relaxation integration (Pacejka62)
	if TireModel.enable_transient_relaxation and dt > 0.0:
		# Allow longitudinal slip to relax during standstill launch when wheel spins while v_x == 0
		var v_kappa_rel: float = v_longitudinal if absf(v_longitudinal) >= absf(omega * wheel_radius) else (omega * wheel_radius)
		var eff_slip: Vector2 = Pacejka62.integrate_transient_slip_substep(
			alpha_transient,
			alpha,
			kappa_transient,
			kappa,
			v_longitudinal,
			dt,
			F_z,
			v_kappa_rel
		)
		alpha_transient = eff_slip.x
		kappa_transient = eff_slip.y
	else:
		alpha_transient = alpha
		kappa_transient = kappa

## Updates radial tire deflection rho_i, loaded radius R_l, effective rolling radius R_e,
## and series ride rate k_ride from normal load F_z and tire vertical stiffness k_z.
func update_radial_deflection(normal_force: float, is_front: bool, config: VehicleConfig) -> void:
	var r0: float = config.wheel_radius if config != null else 0.2032
	if config == null:
		radial_deflection = 0.0
		loaded_radius = r0
		effective_radius = r0
		ride_rate = 0.0
		return
	var kz: float = maxf(config.tire_vertical_stiffness, 1.0)
	var fz_pos: float = maxf(0.0, normal_force) if is_grounded else 0.0
	radial_deflection = minf(fz_pos / kz, 0.5 * r0)
	loaded_radius = r0 - radial_deflection
	effective_radius = r0 - radial_deflection / 3.0
	var k_spring: float = config.spring_stiffness_front if is_front else config.spring_stiffness_rear
	var mr: float = config.motion_ratio_front if is_front else config.motion_ratio_rear
	var k_wheel: float = k_spring * mr * mr
	ride_rate = (k_wheel * kz) / maxf(k_wheel + kz, 1.0)

## Computes tire forces using the high-performance hybrid tire model (MF 6.2 + Neural Residual).
func compute_forces(normal_force: float, config: VehicleConfig) -> void:
	F_z = maxf(0.0, normal_force)
	var is_front: bool = (wheel_id < 2)
	update_radial_deflection(F_z, is_front, config)
	
	if not is_grounded or F_z <= 1e-3:
		F_x = 0.0
		F_y = 0.0
		M_z = 0.0
		M_x = 0.0
		M_y = 0.0
		delta_Fx = 0.0
		delta_Fy = 0.0
		delta_Mz = 0.0
		return
	
	var k_eval = kappa_transient if TireModel.enable_transient_relaxation else kappa
	var a_eval = alpha_transient if TireModel.enable_transient_relaxation else alpha
	
	var tire_res = TireModel.compute_forces(
		k_eval,
		a_eval,
		F_z,
		config,
		camber_angle,
		inflation_pressure_kpa,
		v_longitudinal,
		TireModel.enable_neural_correction,
		is_right_side,
		thermal_mu_factor,
		t_surf_c,
		surface_mu_scale
	)
	
	F_x = tire_res["Fx"]
	F_y = tire_res["Fy"]
	M_z = tire_res["Mz"]
	M_x = tire_res.get("Mx", 0.0)
	M_y = tire_res.get("My", 0.0)
	delta_Fx = tire_res.get("delta_Fx", 0.0)
	delta_Fy = tire_res.get("delta_Fy", 0.0)
	delta_Mz = tire_res.get("delta_Mz", 0.0)

## Integrates the rotational equation of motion for this wheel:
## I_w * d(omega)/dt = T_drive - T_brake * sgn(omega) - F_x * R_w - F_rr * R_w
## Guarantees clean zero-crossing hold without chatter or spurious reversal when T_drive == 0.
func integrate_wheel_spin(T_drive: float, T_brake: float, dt: float, config: VehicleConfig) -> void:
	if dt <= 0.0:
		return
	
	var r = loaded_radius if loaded_radius > 0.0 else config.wheel_radius
	var I_w = config.get_effective_wheel_inertia()
	var no_drive: bool = absf(T_drive) <= 1e-6
	
	# Rolling resistance torque opposing rotation
	var F_rr = config.rolling_resistance_coeff * F_z
	var sign_w = 1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0)
	var T_rr = F_rr * r * sign_w
	
	# Net tyre reaction torque on wheel
	var T_tyre = F_x * r
	
	if absf(omega) > 0.1:
		var T_brk_applied = T_brake * sign_w
		var net_torque = T_drive - T_brk_applied - T_tyre - T_rr
		var d_omega = (net_torque / I_w) * dt
		
		# Clean zero-crossing prevention when braking without drive torque
		if no_drive and T_brake > 0.0 and ((sign_w > 0.0 and (omega + d_omega) <= 0.0) or (sign_w < 0.0 and (omega + d_omega) >= 0.0)):
			omega = 0.0
		else:
			omega += d_omega
	else:
		# Standstill / low-speed caliper hold: prevent residual-torque chatter or reverse creep
		var hold_capacity: float = T_brake + F_rr * r
		var disturbing_torque: float = absf(T_drive - T_tyre)
		if no_drive and T_brake > 0.0 and (hold_capacity >= disturbing_torque or omega * (omega + ((T_drive - T_brake * sign_w - T_tyre - T_rr) / I_w) * dt) <= 0.0):
			omega = 0.0
		else:
			var T_brk_applied = T_brake * sign_w
			var net_torque = T_drive - T_brk_applied - T_tyre - T_rr
			var next_omega = omega + (net_torque / I_w) * dt
			if no_drive and T_brake > 0.0 and sign_w != 0.0 and (sign_w * next_omega) <= 0.0:
				omega = 0.0
			else:
				omega = next_omega
	
	# Update accumulated roll angle
	rotation_angle += omega * dt

## Returns wheel rotational speed in RPM.
func get_rpm() -> float:
	return (omega * 60.0) / (2.0 * PI)

## Snapshots dynamic integrator state for multi-stage ODE integration (RK2).
func get_integrator_state() -> Vector4:
	return Vector4(omega, rotation_angle, alpha_transient, kappa_transient)

## Restores dynamic integrator state from a Vector4 snapshot.
func set_integrator_state(s: Vector4) -> void:
	omega = s.x
	rotation_angle = s.y
	alpha_transient = s.z
	kappa_transient = s.w

## Sets tread surface and core temperatures (C) and refreshes thermal grip multiplier.
func set_tire_temperature(surf_c: float, core_c: float = -999.0) -> void:
	t_surf_c = surf_c
	t_zone_inner_c = surf_c
	t_zone_mid_c = surf_c
	t_zone_outer_c = surf_c
	t_core_c = surf_c if core_c < -900.0 else core_c
	thermal_mu_factor = compute_thermal_grip_factor(t_surf_c)

## Computes tire grip multiplier from tread surface temperature (C) for Hoosier R20/R25B.
static func compute_thermal_grip_factor(temp_c: float) -> float:
	if temp_c <= 25.0:
		return clampf(0.88 + (temp_c - 25.0) * 0.0020, 0.80, 0.88)
	if temp_c < 45.0:
		var u: float = (temp_c - 25.0) / 20.0
		var s: float = u * u * (3.0 - 2.0 * u)
		return lerpf(0.88, 0.98, s)
	if temp_c < 55.0:
		var u2: float = (temp_c - 45.0) / 10.0
		return lerpf(0.98, 1.00, u2)
	if temp_c <= 80.0:
		return 1.00
	if temp_c <= 95.0:
		return lerpf(1.00, 0.95, (temp_c - 80.0) / 15.0)
	return maxf(0.75, 0.95 - 0.006 * (temp_c - 95.0))

## Advances 3-zone tread surface + core carcass thermal model over dt (s).
func step_thermal(dt: float, config: VehicleConfig = null) -> void:
	if config != null:
		t_track_c = _cfg_float(config, "track_temp_c", t_track_c)
	var r_eff: float = effective_radius if effective_radius > 0.0 else (config.wheel_radius if config != null else 0.2032)
	var v_sx: float = omega * r_eff - v_longitudinal
	var v_sy: float = v_lateral
	var in_contact: bool = is_grounded and F_z > 1.0
	slip_power_w = (absf(F_x * v_sx) + absf(F_y * v_sy)) if in_contact else 0.0
	if dt <= 0.0:
		thermal_mu_factor = compute_thermal_grip_factor(t_surf_c)
		return

	# Camber shifts contact heating & track conduction toward inner or outer shoulder:
	# left wheel: negative camber (camber_angle < 0) loads inner; right wheel: positive camber loads inner.
	var camber_inward: float = camber_angle if is_right_side else -camber_angle
	var shift: float = clampf(camber_inward / deg_to_rad(5.0), -1.0, 1.0) * 0.22
	var w_in: float = (1.0 / 3.0) + shift
	var w_out: float = (1.0 / 3.0) - shift
	var w_mid: float = 1.0 - w_in - w_out

	var t_air_c: float = maxf(t_track_c - 4.0, 15.0)
	var contact_scale: float = clampf(F_z / 800.0, 0.25, 1.8) if in_contact else 0.0
	var g_track_total: float = THERM_G_TRACK_NOM_W_K * contact_scale
	var g_air_zone: float = (3.0 + 0.55 * absf(v_longitudinal)) / 3.0
	var q_fric_total: float = THERM_ETA_FRICTION * slip_power_w

	var n_sub: int = maxi(1, int(ceil(dt / 0.05)))
	var h: float = dt / float(n_sub)
	for _s in range(n_sub):
		var q_core_in: float = THERM_G_CORE_ZONE_W_K * (t_core_c - t_zone_inner_c)
		var q_core_mid: float = THERM_G_CORE_ZONE_W_K * (t_core_c - t_zone_mid_c)
		var q_core_out: float = THERM_G_CORE_ZONE_W_K * (t_core_c - t_zone_outer_c)

		var q_lat_im: float = THERM_G_LAT_ZONE_W_K * (t_zone_mid_c - t_zone_inner_c)
		var q_lat_mo: float = THERM_G_LAT_ZONE_W_K * (t_zone_outer_c - t_zone_mid_c)

		var q_in: float = q_fric_total * w_in + (g_track_total * w_in) * (t_track_c - t_zone_inner_c) \
			+ g_air_zone * (t_air_c - t_zone_inner_c) + q_core_in + q_lat_im
		var q_mid: float = q_fric_total * w_mid + (g_track_total * w_mid) * (t_track_c - t_zone_mid_c) \
			+ g_air_zone * (t_air_c - t_zone_mid_c) + q_core_mid - q_lat_im + q_lat_mo
		var q_out: float = q_fric_total * w_out + (g_track_total * w_out) * (t_track_c - t_zone_outer_c) \
			+ g_air_zone * (t_air_c - t_zone_outer_c) + q_core_out - q_lat_mo

		var q_core: float = -(q_core_in + q_core_mid + q_core_out) + THERM_G_CORE_AIR_W_K * (t_air_c - t_core_c)

		t_zone_inner_c = clampf(t_zone_inner_c + (q_in / THERM_C_ZONE_J_K) * h, -40.0, 220.0)
		t_zone_mid_c = clampf(t_zone_mid_c + (q_mid / THERM_C_ZONE_J_K) * h, -40.0, 220.0)
		t_zone_outer_c = clampf(t_zone_outer_c + (q_out / THERM_C_ZONE_J_K) * h, -40.0, 220.0)
		t_core_c = clampf(t_core_c + (q_core / THERM_C_CORE_J_K) * h, -40.0, 220.0)

	t_surf_c = (t_zone_inner_c + t_zone_mid_c + t_zone_outer_c) / 3.0
	thermal_mu_factor = compute_thermal_grip_factor(t_surf_c)

## Resets dynamic states to zero.
func reset() -> void:
	omega = 0.0
	rotation_angle = 0.0
	steer_angle = 0.0
	v_hub_chassis = Vector3.ZERO
	v_longitudinal = 0.0
	v_lateral = 0.0
	kappa = 0.0
	alpha = 0.0
	kappa_transient = 0.0
	alpha_transient = 0.0
	camber_angle = 0.0
	inflation_pressure_kpa = 83.0
	F_z = 0.0
	F_x = 0.0
	F_y = 0.0
	M_z = 0.0
	M_x = 0.0
	M_y = 0.0
	delta_Fx = 0.0
	delta_Fy = 0.0
	delta_Mz = 0.0
	is_grounded = true
	suspension_compression = 0.0
	suspension_velocity = 0.0
	radial_deflection = 0.0
	loaded_radius = 0.2032
	effective_radius = 0.2032
	ride_rate = 0.0
	t_surf_c = THERM_INIT_C
	t_core_c = THERM_INIT_C
	t_zone_inner_c = THERM_INIT_C
	t_zone_mid_c = THERM_INIT_C
	t_zone_outer_c = THERM_INIT_C
	t_track_c = THERM_TRACK_DEFAULT_C
	thermal_mu_factor = 1.0
	surface_mu_scale = 1.0
	slip_power_w = 0.0
