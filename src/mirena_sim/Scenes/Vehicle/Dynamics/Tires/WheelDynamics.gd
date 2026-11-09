class_name WheelDynamics
extends RefCounted

## Individual Wheel Dynamics & Contact Model for Mirena Dynamics
## Represents a single wheel with slip kinematics, transient carcass relaxation,
## hybrid tire contact forces (MF 6.2 + Neural Residual), and rotational spin.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const TireModel = preload("TireModel.gd")
const TransientRelaxation = preload("Transient/TransientRelaxation.gd")

var wheel_id: int = 0
var name: String = "Wheel"

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

# --- Suspension State ---
var is_grounded: bool = true
var suspension_compression: float = 0.0 # m
var suspension_velocity: float = 0.0    # m/s

func _init(id: int = 0, wheel_name: String = "Wheel") -> void:
	wheel_id = id
	name = wheel_name

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
	
	# First-order transient carcass relaxation integration
	if TireModel.enable_transient_relaxation and dt > 0.0:
		alpha_transient = TransientRelaxation.integrate_alpha_relaxation(alpha_transient, alpha, v_longitudinal, dt, F_z)
		kappa_transient = TransientRelaxation.integrate_kappa_relaxation(kappa_transient, kappa, v_longitudinal, dt, F_z)
	else:
		alpha_transient = alpha
		kappa_transient = kappa

## Computes tire forces using the high-performance hybrid tire model (MF 6.2 + Neural Residual).
func compute_forces(normal_force: float, config: VehicleConfig) -> void:
	F_z = maxf(0.0, normal_force)
	
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
		TireModel.enable_neural_correction
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
func integrate_wheel_spin(T_drive: float, T_brake: float, dt: float, config: VehicleConfig) -> void:
	if dt <= 0.0:
		return
	
	var r = config.wheel_radius
	var I_w = maxf(config.wheel_inertia, 1e-4)
	
	# Rolling resistance torque opposing rotation
	var F_rr = config.rolling_resistance_coeff * F_z
	var T_rr = F_rr * r * (1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0))
	
	# Net tyre reaction torque on wheel
	var T_tyre = F_x * r
	
	# Handle braking torque (opposes motion)
	var T_brk_applied = 0.0
	var sign_w = 1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0)
	
	if absf(omega) > 0.1:
		T_brk_applied = T_brake * sign_w
		var net_torque = T_drive - T_brk_applied - T_tyre - T_rr
		var d_omega = (net_torque / I_w) * dt
		
		# Zero-crossing prevention when braking heavily to standstill
		if sign_w > 0.0 and (omega + d_omega) < 0.0 and T_drive == 0.0:
			omega = 0.0
		elif sign_w < 0.0 and (omega + d_omega) > 0.0 and T_drive == 0.0:
			omega = 0.0
		else:
			omega += d_omega
	else:
		# Low-speed / static friction logic to prevent chatter
		var hold_torque = absf(T_drive - T_tyre)
		if T_brake >= hold_torque and T_drive == 0.0:
			omega = 0.0
		else:
			T_brk_applied = T_brake * sign_w
			var net_torque = T_drive - T_brk_applied - T_tyre - T_rr
			omega += (net_torque / I_w) * dt
	
	# Update accumulated roll angle
	rotation_angle += omega * dt

## Returns wheel rotational speed in RPM.
func get_rpm() -> float:
	return (omega * 60.0) / (2.0 * PI)

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
