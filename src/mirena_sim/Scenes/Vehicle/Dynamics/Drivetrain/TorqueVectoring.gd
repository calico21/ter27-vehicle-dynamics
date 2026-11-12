class_name TorqueVectoring
extends RefCounted

## Decoupled 200 Hz Torque Vectoring Controller / Allocator (Block D)
##
## Runs behind an internal time accumulator (default 200 Hz, dt_tv = 0.005 s)
## decoupled from Godot's 400 Hz physics tick.
##
## Modes:
## - MODE_OPEN: symmetric 50/50 L-R distribution (respecting front/rear axle ratio).
## - MODE_DYC:  Direct Yaw Control allocating differential torque dT = +/- (dMz * Rw / track)
##              within individual motor limits [T_min, T_max] and the 80 kW power ceiling,
##              preserving coherent total longitudinal torque sum(T_i).

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")

enum TVMode {
	MODE_OPEN = 0,
	MODE_DYC = 1
}

var _accumulator: float = 0.0
var _latched_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var update_count: int = 0

func reset() -> void:
	_accumulator = 0.0
	_latched_torques = [0.0, 0.0, 0.0, 0.0]
	update_count = 0

func get_latched_torques() -> Array[float]:
	return _latched_torques.duplicate()

## Updates Torque Vectoring at target rate (default 200 Hz) via internal accumulator.
## Pass advance_clock = false on RK2 sub-stages so the ECU clock advances once per physical step.
func step_allocation(
	base_torques: Array[float],
	inputs: VehicleInputs,
	wheel_omegas: Array,
	speed: float,
	dt: float,
	config: VehicleConfig,
	advance_clock: bool = true
) -> Array[float]:
	if inputs.override_wheel_torques:
		_latched_torques = base_torques.duplicate()
		return _latched_torques.duplicate()

	if dt <= 0.0:
		return compute_allocation(base_torques, inputs, wheel_omegas, speed, config)

	var rate_hz: float = maxf(config.tv_rate_hz, 1.0)
	var tv_dt: float = 1.0 / rate_hz

	if advance_clock:
		_accumulator += dt
		while _accumulator >= tv_dt - 1e-9:
			_accumulator -= tv_dt
			_latched_torques = compute_allocation(base_torques, inputs, wheel_omegas, speed, config)
			update_count += 1

	# For standard open drive modes before first TV clock tick, track base torques cleanly
	if update_count == 0 and inputs.tv_mode == TVMode.MODE_OPEN and inputs.drive_mode != VehicleConfig.DriveMode.TORQUE_VECTORING:
		return base_torques.duplicate()

	return _latched_torques.duplicate()

## Computes motor angular speed derating factor [0.0, 1.0] for tractive drive torque.
## Linearly tapers from 1.0 at motor_taper_ratio * omega_max down to 0.0 at omega_max.
static func compute_motor_speed_scale(wheel_omega: float, is_tractive: bool, config: VehicleConfig) -> float:
	if not is_tractive:
		return 1.0
	var max_rpm: float = config.motor_max_rpm
	if max_rpm <= 0.0:
		return 1.0
	var omega_max: float = max_rpm * (TAU / 60.0)
	var taper_ratio: float = clampf(config.motor_taper_ratio, 0.5, 0.999)
	var omega_taper_start: float = omega_max * taper_ratio
	var omega_motor: float = absf(wheel_omega) * maxf(config.gear_ratio, 1e-3)
	if omega_motor <= omega_taper_start:
		return 1.0
	if omega_motor >= omega_max:
		return 0.0
	return 1.0 - (omega_motor - omega_taper_start) / maxf(omega_max - omega_taper_start, 1e-5)

## Pure stateless allocation function computing per-wheel torques [FL, FR, RL, RR] in Nm.
static func compute_allocation(
	base_torques: Array[float],
	inputs: VehicleInputs,
	wheel_omegas: Array,
	speed: float,
	config: VehicleConfig
) -> Array[float]:
	var out: Array[float] = [
		float(base_torques[0]),
		float(base_torques[1]),
		float(base_torques[2]),
		float(base_torques[3])
	]

	if inputs.tv_mode != TVMode.MODE_DYC or absf(inputs.tv_yaw_moment_req) < 1e-6:
		return out

	var r: float = maxf(config.wheel_radius, 1e-4)
	var G: float = config.gear_ratio
	var eff: float = config.drivetrain_efficiency
	var t_peak: float = config.motor_peak_torque * G * eff
	var t_regen: float = (config.regen_peak_torque * G * eff) if config.tv_allow_regen else 0.0

	var t_max: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_min: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var w_abs: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var min_w: float = absf(speed) / r

	for i in range(4):
		var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else min_w
		w_abs[i] = maxf(absf(w_i), min_w)
		t_max[i] = t_peak * compute_motor_speed_scale(w_abs[i], true, config)
		t_min[i] = -t_regen

	var sf: float = config.front_drive_torque_ratio
	var sr: float = 1.0 - sf
	if inputs.drive_mode == VehicleConfig.DriveMode.REAR_WD:
		sf = 0.0
		sr = 1.0
	elif inputs.drive_mode == VehicleConfig.DriveMode.FRONT_WD:
		sf = 1.0
		sr = 0.0

	# ISO 8855: +Mz turns LEFT (CCW around +Z) => increase RIGHT torque (+), decrease LEFT torque (-)
	var dt_f_req: float = (inputs.tv_yaw_moment_req * sf * r) / maxf(config.track_width_f, 0.1)
	var dt_r_req: float = (inputs.tv_yaw_moment_req * sr * r) / maxf(config.track_width_r, 0.1)

	# Clamp symmetrically on each axle to obey individual motor limits while conserving sum(T_i)
	var fl0: float = clampf(out[0], t_min[0], t_max[0])
	var fr0: float = clampf(out[1], t_min[1], t_max[1])
	var dtf_lo: float = maxf(fl0 - t_max[0], t_min[1] - fr0)
	var dtf_hi: float = minf(fl0 - t_min[0], t_max[1] - fr0)
	var dt_f: float = clampf(dt_f_req, dtf_lo, dtf_hi)
	out[0] = fl0 - dt_f
	out[1] = fr0 + dt_f

	var rl0: float = clampf(out[2], t_min[2], t_max[2])
	var rr0: float = clampf(out[3], t_min[3], t_max[3])
	var dtr_lo: float = maxf(rl0 - t_max[2], t_min[3] - rr0)
	var dtr_hi: float = minf(rl0 - t_min[2], t_max[3] - rr0)
	var dt_r: float = clampf(dt_r_req, dtr_lo, dtr_hi)
	out[2] = rl0 - dt_r
	out[3] = rr0 + dt_r

	# Enforce global 80 kW power limit across all 4 wheels
	var p_limit_mech: float = config.power_limit_watts * eff
	var p_mech: float = 0.0
	for i in range(4):
		if out[i] > 0.0:
			p_mech += out[i] * maxf(w_abs[i], 1.0 / r)

	if p_mech > p_limit_mech and p_mech > 1e-6:
		var scale_p: float = p_limit_mech / p_mech
		for i in range(4):
			if out[i] > 0.0:
				out[i] *= scale_p

	return out
