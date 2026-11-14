class_name DrivetrainModel
extends RefCounted

## Drivetrain & Powertrain Model for Mirena Dynamics (Block D)
## Computes per-wheel motor drive torques [FL, FR, RL, RR] with:
## - Formula Student 80 kW global power ceiling
## - Motor angular speed limiter (linear taper to 0 tractive torque at motor_max_rpm)
## - Decoupled 200 Hz Torque Vectoring layer (MODE_OPEN / MODE_DYC)

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")
const BrakeModel = preload("../Brakes/BrakeModel.gd")
const TorqueVectoring = preload("TorqueVectoring.gd")

## Stateful HV accumulator and first-order inverter torque dynamics.
class HVPowertrainState extends RefCounted:
	var soc: float = 0.95
	var v_oc: float = 594.0
	var v_dc: float = 594.0
	var i_dc: float = 0.0
	var power_kw: float = 0.0
	var energy_kwh: float = 0.0
	var r_int_ohm: float = 0.14
	var inverter_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

	func _init(config: VehicleConfig = null) -> void:
		reset(config)

	func reset(config: VehicleConfig = null) -> void:
		var p: Dictionary = DrivetrainModel._pack_params(config)
		soc = clampf(float(p["soc_init"]), 0.0, 1.0)
		r_int_ohm = float(p["r_int"])
		v_oc = DrivetrainModel.compute_ocv(soc, float(p["v_full"]), float(p["v_empty"]))
		v_dc = v_oc
		i_dc = 0.0
		power_kw = 0.0
		energy_kwh = 0.0
		inverter_torques = [0.0, 0.0, 0.0, 0.0]


static func _cfg_float(config: VehicleConfig, prop: String, default_val: float) -> float:
	if config == null:
		return default_val
	var v: Variant = config.get(prop)
	return float(v) if (typeof(v) == TYPE_FLOAT or typeof(v) == TYPE_INT) else default_val


static func _pack_params(config: VehicleConfig) -> Dictionary:
	var v_full: float = _cfg_float(config, "battery_ocv_full_v", 600.0)
	return {
		"soc_init": clampf(_cfg_float(config, "battery_initial_soc", 0.95), 0.0, 1.0),
		"r_int": maxf(_cfg_float(config, "battery_internal_resistance_ohm", 0.14), 1e-4),
		"v_nom": maxf(_cfg_float(config, "battery_nominal_voltage_v", 540.0), 1.0),
		"v_full": v_full,
		"v_empty": maxf(v_full - 160.0, 0.70 * v_full),
		"cap_kwh": maxf(_cfg_float(config, "battery_capacity_kwh", 7.2), 0.1)
	}


## Realistic Li-Ion HV pack open-circuit voltage curve vs state of charge [0, 1].
static func compute_ocv(soc: float, v_full: float = 600.0, v_empty: float = 440.0) -> float:
	var s: float = clampf(soc, 0.0, 1.0)
	var span: float = maxf(v_full - v_empty, 0.0)
	var knee: float = 0.18 * span * (1.0 - exp(-10.0 * s)) / (1.0 - exp(-10.0))
	return v_empty + knee + 0.82 * span * s


## Exact quadratic DC bus solver for P_elec = (V_oc - I_dc * R_int) * I_dc.
static func solve_dc_bus(p_elec_w: float, v_oc: float, r_int: float) -> Dictionary:
	var voc: float = maxf(v_oc, 1.0)
	var r: float = maxf(r_int, 0.0)
	if r <= 1e-6:
		return {"v_dc": voc, "i_dc": p_elec_w / voc, "v_oc": voc}
	var disc_min: float = (0.25 * voc) * (0.25 * voc)
	var disc: float = maxf(voc * voc - 4.0 * r * p_elec_w, disc_min)
	var i_dc: float = (voc - sqrt(disc)) / (2.0 * r)
	var v_dc: float = voc - i_dc * r
	return {"v_dc": v_dc, "i_dc": i_dc, "v_oc": voc}


## Exact first-order exponential inverter torque response filter.
static func step_inverter_filter(
	target_torques: Array[float],
	prev_torques: Array[float],
	dt: float,
	tau_inv_s: float = 0.008
) -> Array[float]:
	var out: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if tau_inv_s <= 1e-6:
		for i in range(mini(4, target_torques.size())):
			out[i] = float(target_torques[i])
		return out
	if dt <= 0.0:
		for i in range(4):
			out[i] = float(prev_torques[i]) if i < prev_torques.size() else (float(target_torques[i]) if i < target_torques.size() else 0.0)
		return out
	var alpha: float = 1.0 - exp(-dt / maxf(tau_inv_s, 1e-5))
	for i in range(4):
		var p: float = float(prev_torques[i]) if i < prev_torques.size() else 0.0
		var t: float = float(target_torques[i]) if i < target_torques.size() else 0.0
		out[i] = p + alpha * (t - p)
	return out


## Advances inverter torque lag, DC electrical power, bus sag, and SOC/energy integration.
## Set advance_state = false in RK2 predictor stages to avoid double-consuming battery energy.
static func step_hv_and_inverters(
	hv_state: HVPowertrainState,
	target_torques: Array[float],
	wheel_omegas: Array,
	dt: float,
	config: VehicleConfig,
	advance_state: bool = true,
	bypass_inverter_lag: bool = false
) -> Array[float]:
	if hv_state == null:
		return target_torques.duplicate()

	var eff: float = clampf(_cfg_float(config, "drivetrain_efficiency", 0.90), 0.1, 1.0)
	var p_dc_limit: float = maxf(_cfg_float(config, "max_dc_power_w", _cfg_float(config, "power_limit_watts", 80000.0)), 0.0)
	var tau_inv: float = maxf(_cfg_float(config, "inverter_tau_s", 0.008), 0.0)
	var pack: Dictionary = _pack_params(config)
	var r_int: float = maxf(_cfg_float(config, "battery_internal_resistance_ohm", hv_state.r_int_ohm), 1e-4)
	var v_full: float = float(pack["v_full"])
	var v_empty: float = float(pack["v_empty"])
	var cap_kwh: float = float(pack["cap_kwh"])

	var filtered: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if dt > 0.0 and not bypass_inverter_lag:
		filtered = step_inverter_filter(target_torques, hv_state.inverter_torques, dt, tau_inv)
	else:
		for i in range(mini(4, target_torques.size())):
			filtered[i] = float(target_torques[i])

	var p_elec_drive: float = 0.0
	var p_elec_regen: float = 0.0
	for i in range(4):
		var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else 0.0
		var p_mech_i: float = filtered[i] * w_i
		if p_mech_i >= 0.0:
			p_elec_drive += p_mech_i / eff
		else:
			p_elec_regen += p_mech_i * eff

	if p_dc_limit > 0.0 and not bypass_inverter_lag and (p_elec_drive + p_elec_regen) > p_dc_limit and p_elec_drive > 1e-6:
		var scale_p: float = clampf((p_dc_limit - p_elec_regen) / p_elec_drive, 0.0, 1.0)
		for i in range(4):
			var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else 0.0
			if filtered[i] * w_i > 0.0:
				filtered[i] *= scale_p
		p_elec_drive *= scale_p

	var p_elec_w: float = p_elec_drive + p_elec_regen
	var v_oc_now: float = compute_ocv(hv_state.soc, v_full, v_empty)
	var bus: Dictionary = solve_dc_bus(p_elec_w, v_oc_now, r_int)

	if advance_state:
		hv_state.inverter_torques = filtered.duplicate()
		hv_state.r_int_ohm = r_int
		hv_state.v_dc = float(bus["v_dc"])
		hv_state.i_dc = float(bus["i_dc"])
		hv_state.power_kw = p_elec_w * 0.001
		if dt > 0.0:
			var d_kwh: float = (p_elec_w * dt) / 3600000.0
			hv_state.energy_kwh += d_kwh
			hv_state.soc = clampf(hv_state.soc - d_kwh / cap_kwh, 0.0, 1.0)
		hv_state.v_oc = compute_ocv(hv_state.soc, v_full, v_empty)

	return filtered


## Computes base symmetric drive torques for all 4 wheels in Nm.
static func compute_drive_torques(
	inputs: VehicleInputs,
	speed: float,
	config: VehicleConfig,
	wheel_omegas: Array = [],
	v_dc_override: float = -1.0
) -> Array[float]:
	var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	if inputs.override_wheel_torques:
		for i in range(4):
			drive_torques[i] = inputs.wheel_motor_torques[i]
		return drive_torques
	
	var throttle = inputs.throttle
	if throttle > 0.0:
		var r = maxf(config.wheel_radius, 1e-4)
		var G = config.gear_ratio
		var eff = config.drivetrain_efficiency
		
		# Peak wheel drive torque per motor: T_wheel = T_motor * G * eff
		var max_wheel_trq = config.motor_peak_torque * G * eff
		
		# Electric power envelope: P_max * eff / speed
		var speed_clamped = maxf(absf(speed), 1.0)
		var max_power_force = (config.power_limit_watts * eff) / speed_clamped
		var max_power_trq_total = max_power_force * r
		
		var total_max_trq = minf(4.0 * max_wheel_trq, max_power_trq_total)
		var requested_total_trq = throttle * total_max_trq
		
		var s_front = config.front_drive_torque_ratio
		var s_rear = 1.0 - s_front
		
		match inputs.drive_mode:
			VehicleConfig.DriveMode.FOUR_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * s_rear * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * s_rear * 0.5
			VehicleConfig.DriveMode.REAR_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = 0.0
				drive_torques[VehicleConfig.WheelIndex.FR] = 0.0
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * 0.5
			VehicleConfig.DriveMode.FRONT_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = 0.0
				drive_torques[VehicleConfig.WheelIndex.RR] = 0.0
			_:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * s_rear * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * s_rear * 0.5

		# Apply per-wheel field-weakening envelope + motor angular speed limiter
		# Base speed scales proportionally with actual DC bus voltage V_dc / V_nom
		var fallback_omega: float = absf(speed) / r
		var pack: Dictionary = _pack_params(config)
		var v_nom: float = float(pack["v_nom"])
		var v_dc_eff: float = v_dc_override
		if v_dc_eff <= 0.0:
			var v_oc_est: float = compute_ocv(float(pack["soc_init"]), float(pack["v_full"]), float(pack["v_empty"]))
			var p_est_w: float = 0.0
			for i in range(4):
				var w_i_est: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
				p_est_w += maxf(0.0, drive_torques[i] * absf(w_i_est)) / maxf(eff, 1e-3)
			var p_dc_lim_est: float = maxf(config.max_dc_power_w, 0.0)
			if p_dc_lim_est > 0.0:
				p_est_w = minf(p_est_w, p_dc_lim_est)
			v_dc_eff = float(solve_dc_bus(p_est_w, v_oc_est, float(pack["r_int"]))["v_dc"])

		var v_scale: float = clampf(v_dc_eff / v_nom, 0.5, 1.25)
		var omega_base_motor: float = maxf(config.base_speed_rpm, 0.0) * (TAU / 60.0) * v_scale
		var p_motor_peak: float = config.motor_peak_torque * omega_base_motor
		for i in range(4):
			var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
			if drive_torques[i] > 0.0:
				var omega_motor: float = absf(w_i) * maxf(G, 1e-3)
				var t_motor_env: float = config.motor_peak_torque
				if omega_base_motor > 0.0 and omega_motor > omega_base_motor:
					t_motor_env = minf(config.motor_peak_torque, p_motor_peak / omega_motor)
				var speed_scale: float = TorqueVectoring.compute_motor_speed_scale(w_i, true, config)
				var max_trq_i: float = t_motor_env * G * eff * speed_scale
				drive_torques[i] = minf(drive_torques[i], max_trq_i)

		# Enforce strict 80 kW DC bus electrical power ceiling: sum(max(0, T_i * w_i) / eff) <= P_max_dc
		var p_dc_limit: float = maxf(config.max_dc_power_w, 0.0)
		if p_dc_limit > 0.0:
			var p_elec_total: float = 0.0
			for i in range(4):
				var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
				p_elec_total += maxf(0.0, drive_torques[i] * absf(w_i)) / maxf(eff, 1e-3)
			if p_elec_total > p_dc_limit:
				var p_scale: float = p_dc_limit / p_elec_total
				for i in range(4):
					drive_torques[i] *= p_scale
	
	return drive_torques

## Combined convenience method: returns {"drive": Array[float], "brake": Array[float]}.
## Optionally routes drive torques through a stateful 200 Hz TorqueVectoring controller.
static func compute_torques(
	inputs: VehicleInputs,
	speed: float,
	config: VehicleConfig,
	wheel_omegas: Array = [],
	tv: TorqueVectoring = null,
	dt: float = 0.0,
	advance_tv: bool = true,
	v_dc_override: float = -1.0
) -> Dictionary:
	var drive = compute_drive_torques(inputs, speed, config, wheel_omegas, v_dc_override)
	if tv != null:
		drive = tv.step_allocation(drive, inputs, wheel_omegas, speed, dt, config, advance_tv)
	var brake = BrakeModel.compute_brake_torques(inputs, config)
	return {"drive": drive, "brake": brake}
