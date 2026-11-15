class_name BrakeModel
extends RefCounted

## Brakes Model for Mirena Dynamics
## Computes per-wheel mechanical friction brake torques, EBS distribution,
## regenerative brake blending, and 1-node AP Racing RQ3 disc thermal model.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")

const DEFAULT_BRAKE_MU: float = 1.30
const DEFAULT_BRAKE_BIAS_FRONT: float = 0.58
const DEFAULT_REGEN_BLENDING_ENABLED: bool = true
const DEFAULT_P_MAX_CHARGE_W: float = 25000.0
const DEFAULT_REGEN_MIN_SPEED_MPS: float = 1.5
const DEFAULT_DISC_AMBIENT_TEMP_C: float = 40.0
const DEFAULT_DISC_MAX_TEMP_C: float = 650.0
const DEFAULT_DISC_THERMAL_MASS_J_K: float = 1350.0
const DEFAULT_DISC_CONV_BASE_W_K: float = 12.0
const DEFAULT_DISC_CONV_SPEED_COEFF: float = 1.8


## Per-wheel thermal and blended braking state [FL, FR, RL, RR].
class BrakeState extends RefCounted:
	var disc_temp: Array[float] = [40.0, 40.0, 40.0, 40.0]
	var hydraulic_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var regen_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var total_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var pad_mu_eff: Array[float] = [1.30, 1.30, 1.30, 1.30]
	var p_regen_w: float = 0.0

	func _init(ambient_c: float = 40.0) -> void:
		reset(ambient_c)

	## Resets disc temperatures, torques, and pad friction to resting defaults.
	func reset(ambient_c: float = 40.0) -> void:
		for i in range(4):
			disc_temp[i] = ambient_c
			hydraulic_torques[i] = 0.0
			regen_torques[i] = 0.0
			total_torques[i] = 0.0
			pad_mu_eff[i] = DEFAULT_BRAKE_MU
		p_regen_w = 0.0


static func _cfg_float(config: Variant, prop: String, default_val: float) -> float:
	if config == null:
		return default_val
	if config is Dictionary:
		return float(config.get(prop, default_val))
	var v: Variant = config.get(prop)
	if typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT:
		return float(v)
	return default_val


static func _cfg_bool(config: Variant, prop: String, default_val: bool) -> bool:
	if config == null:
		return default_val
	if config is Dictionary:
		return bool(config.get(prop, default_val))
	var v: Variant = config.get(prop)
	if typeof(v) == TYPE_BOOL:
		return bool(v)
	return default_val


## Computes per-wheel friction brake torques [FL, FR, RL, RR] in Nm.
static func compute_brake_torques(inputs: VehicleInputs, config: VehicleConfig) -> Array[float]:
	var brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	if inputs.override_wheel_torques:
		for i in range(4):
			brake_torques[i] = inputs.wheel_brake_torques[i]
		return brake_torques
	
	var brake_cmd = inputs.brake
	var ebs_cmd = inputs.ebs
	if brake_cmd > 0.0 or ebs_cmd > 0.0:
		var r = config.wheel_radius
		var total_brake_force = config.brake_mu * config.mass * config.gravity
		var f_front = total_brake_force * config.brake_bias_front
		var f_rear = total_brake_force * (1.0 - config.brake_bias_front)
		
		var ebs_trq_each = (total_brake_force * 0.25 * r) * ebs_cmd
		
		brake_torques[VehicleConfig.WheelIndex.FL] = (f_front * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.FR] = (f_front * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.RL] = (f_rear * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.RR] = (f_rear * 0.5 * brake_cmd) * r + ebs_trq_each
	
	return brake_torques


## AP Racing RQ3 pad friction multiplier vs disc temperature (deg C).
## Optimal window [120, 500] C -> 1.0, cold (40 C) -> 0.94, fade >520 C -> ~0.78 at 650 C.
static func compute_pad_mu_scale(temp_c: float) -> float:
	if temp_c <= 40.0:
		return 0.94
	if temp_c < 120.0:
		return lerpf(0.94, 1.0, smoothstep(40.0, 120.0, temp_c))
	if temp_c <= 520.0:
		return 1.0
	return lerpf(1.0, 0.78, smoothstep(520.0, 650.0, temp_c))


## Computes motor speed taper factor [0, 1] for regenerative torque capability.
static func _compute_motor_taper_scale(wheel_omega: float, config: Variant) -> float:
	var max_rpm: float = _cfg_float(config, "motor_max_rpm", 22000.0)
	if max_rpm <= 0.0:
		return 1.0
	var gear_ratio: float = maxf(_cfg_float(config, "gear_ratio", 12.5), 1e-3)
	var taper_ratio: float = clampf(_cfg_float(config, "motor_taper_ratio", 0.85), 0.5, 0.999)
	var omega_max: float = max_rpm * (TAU / 60.0)
	var omega_taper_start: float = omega_max * taper_ratio
	var omega_motor: float = absf(wheel_omega) * gear_ratio
	if omega_motor <= omega_taper_start:
		return 1.0
	if omega_motor >= omega_max - 1e-6:
		return 0.0
	return 1.0 - (omega_motor - omega_taper_start) / maxf(omega_max - omega_taper_start, 1e-5)


## Reads HV accumulator SOC safely when hv_state is attached.
static func _compute_soc_regen_scale(hv_state: Variant) -> float:
	if hv_state == null:
		return 1.0
	var soc_val: Variant = hv_state.get("soc", null) if hv_state is Dictionary else hv_state.get("soc")
	if typeof(soc_val) != TYPE_FLOAT and typeof(soc_val) != TYPE_INT:
		return 1.0
	var soc: float = float(soc_val)
	if soc < 0.995:
		return 1.0
	# Smoothly derate regen once SOC >= 0.995 to prevent cell overcharge.
	return clampf((0.998 - soc) / 0.008, 0.0, 1.0)


## Computes series-blended regenerative + hydraulic braking torques and updates 1-node disc thermal state.
static func compute_blended_braking(
	inputs: Variant,
	config: Variant,
	brake_state: BrakeState,
	wheel_omegas: Array,
	wheel_fz: Array,
	v_x: float,
	dt: float,
	hv_state: Variant = null
) -> Dictionary:
	var t_amb: float = _cfg_float(config, "disc_ambient_temp_c", DEFAULT_DISC_AMBIENT_TEMP_C)
	var t_max: float = maxf(_cfg_float(config, "disc_max_temp_c", DEFAULT_DISC_MAX_TEMP_C), t_amb)
	var state: BrakeState = brake_state if brake_state != null else BrakeState.new(t_amb)

	var brake_mu: float = maxf(_cfg_float(config, "brake_mu", DEFAULT_BRAKE_MU), 0.0)
	var bias_front: float = clampf(_cfg_float(config, "brake_bias_front", DEFAULT_BRAKE_BIAS_FRONT), 0.0, 1.0)
	var regen_enabled: bool = _cfg_bool(config, "regen_blending_enabled", DEFAULT_REGEN_BLENDING_ENABLED)
	var p_max_charge_w: float = maxf(_cfg_float(config, "p_max_charge_w", DEFAULT_P_MAX_CHARGE_W), 0.0)
	var regen_min_speed: float = maxf(_cfg_float(config, "regen_min_speed_mps", DEFAULT_REGEN_MIN_SPEED_MPS), 0.0)
	var c_disc: float = maxf(_cfg_float(config, "disc_thermal_mass_j_k", DEFAULT_DISC_THERMAL_MASS_J_K), 1.0)
	var h_base: float = maxf(_cfg_float(config, "disc_conv_base_w_k", DEFAULT_DISC_CONV_BASE_W_K), 0.0)
	var h_v: float = maxf(_cfg_float(config, "disc_conv_speed_coeff", DEFAULT_DISC_CONV_SPEED_COEFF), 0.0)

	var mass: float = maxf(_cfg_float(config, "mass", 290.0), 1.0)
	var gravity: float = maxf(_cfg_float(config, "gravity", 9.81), 0.1)
	var wheel_radius: float = maxf(_cfg_float(config, "wheel_radius", 0.2032), 1e-4)
	var motor_peak_trq: float = maxf(_cfg_float(config, "motor_peak_torque", 25.0), 0.0)
	var gear_ratio: float = maxf(_cfg_float(config, "gear_ratio", 12.5), 1e-3)

	var total_req: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var override_trq: bool = false
	if inputs != null:
		var ov: Variant = inputs.get("override_wheel_torques", false) if inputs is Dictionary else inputs.get("override_wheel_torques")
		override_trq = bool(ov) if ov != null else false

	if override_trq:
		var ov_arr: Variant = inputs.get("wheel_brake_torques", []) if inputs is Dictionary else inputs.get("wheel_brake_torques")
		if ov_arr is Array:
			for i in range(mini(4, ov_arr.size())):
				total_req[i] = maxf(0.0, float(ov_arr[i]))
	elif inputs != null:
		var b_raw: Variant = inputs.get("brake", 0.0) if inputs is Dictionary else inputs.get("brake")
		var e_raw: Variant = inputs.get("ebs", 0.0) if inputs is Dictionary else inputs.get("ebs")
		var brake_cmd: float = clampf(float(b_raw) if b_raw != null else 0.0, 0.0, 1.0)
		var ebs_cmd: float = clampf(float(e_raw) if e_raw != null else 0.0, 0.0, 1.0)
		if brake_cmd > 0.0 or ebs_cmd > 0.0:
			var total_brake_force: float = brake_mu * mass * gravity
			var f_front: float = total_brake_force * bias_front
			var f_rear: float = total_brake_force * (1.0 - bias_front)
			var ebs_trq_each: float = (total_brake_force * 0.25 * wheel_radius) * ebs_cmd
			total_req[VehicleConfig.WheelIndex.FL] = (f_front * 0.5 * brake_cmd) * wheel_radius + ebs_trq_each
			total_req[VehicleConfig.WheelIndex.FR] = (f_front * 0.5 * brake_cmd) * wheel_radius + ebs_trq_each
			total_req[VehicleConfig.WheelIndex.RL] = (f_rear * 0.5 * brake_cmd) * wheel_radius + ebs_trq_each
			total_req[VehicleConfig.WheelIndex.RR] = (f_rear * 0.5 * brake_cmd) * wheel_radius + ebs_trq_each

	var omegas_abs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var regen_trq: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var soc_scale: float = _compute_soc_regen_scale(hv_state)
	var p_charge_limit: float = p_max_charge_w * soc_scale
	var allow_regen: bool = regen_enabled and absf(v_x) >= regen_min_speed and p_charge_limit > 0.0 and not override_trq

	var dm: int = int(config.get("drive_mode", VehicleConfig.DriveMode.FOUR_WD)) if (config is Dictionary) else int(config.drive_mode)
	for i in range(4):
		var w_i: float = absf(float(wheel_omegas[i])) if i < wheel_omegas.size() else (absf(v_x) / wheel_radius)
		omegas_abs[i] = w_i
		var axle_has_motor: bool = true
		if dm == VehicleConfig.DriveMode.REAR_WD and i < 2:
			axle_has_motor = false
		elif dm == VehicleConfig.DriveMode.FRONT_WD and i >= 2:
			axle_has_motor = false
		if allow_regen and axle_has_motor and total_req[i] > 0.0 and w_i > 1e-4:
			var speed_scale: float = _compute_motor_taper_scale(w_i, config)
			var trq_motor_cap: float = motor_peak_trq * gear_ratio * speed_scale * soc_scale
			var fz_i: float = maxf(float(wheel_fz[i]), 0.0) if i < wheel_fz.size() else (mass * gravity * 0.25)
			var trq_tire_cap: float = fz_i * 1.35 * wheel_radius
			regen_trq[i] = minf(total_req[i], minf(trq_motor_cap, trq_tire_cap))

	var p_regen_w: float = 0.0
	for i in range(4):
		p_regen_w += regen_trq[i] * omegas_abs[i]
	if p_regen_w > p_charge_limit and p_regen_w > 1e-6:
		var scale_p: float = p_charge_limit / p_regen_w
		for i in range(4):
			regen_trq[i] *= scale_p
		p_regen_w *= scale_p

	for i in range(4):
		var temp_i: float = state.disc_temp[i]
		var mu_scale: float = compute_pad_mu_scale(temp_i)
		var hydraulic_req: float = maxf(0.0, total_req[i] - regen_trq[i])
		var hydraulic_actual: float = hydraulic_req * mu_scale
		var total_actual: float = regen_trq[i] + hydraulic_actual

		if dt > 0.0:
			var q_in: float = hydraulic_actual * omegas_abs[i]
			var q_cool: float = (h_base + h_v * absf(v_x)) * (temp_i - t_amb)
			state.disc_temp[i] = clampf(temp_i + ((q_in - q_cool) * dt) / c_disc, t_amb, t_max)

		state.regen_torques[i] = regen_trq[i]
		state.hydraulic_torques[i] = hydraulic_actual
		state.total_torques[i] = total_actual
		state.pad_mu_eff[i] = brake_mu * mu_scale

	state.p_regen_w = p_regen_w

	return {
		"total_torques": state.total_torques.duplicate(),
		"hydraulic_torques": state.hydraulic_torques.duplicate(),
		"regen_torques": state.regen_torques.duplicate(),
		"p_regen_w": state.p_regen_w,
		"disc_temps": state.disc_temp.duplicate(),
		"pad_mu_eff": state.pad_mu_eff.duplicate()
	}
