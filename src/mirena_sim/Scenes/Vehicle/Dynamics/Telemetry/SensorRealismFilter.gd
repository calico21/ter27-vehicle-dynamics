class_name SensorRealismFilter
extends RefCounted

## Sensor Realism & CAN Latency Profile (SENSOR_IDEAL vs. SENSOR_TRACK_REALISM).
## Exact numerical parity with Python vd_correlation_lab/core/sensor_realism_profile.py.
##
## Features in SENSOR_TRACK_REALISM:
## - Differentiated CAN transport latency ring-buffers:
##   * Accelerometer (ax, ay): default 125.0 ms (TeR26 CAN audit)
##   * Steering & bus (delta, wz, omega_4): default 20.0 ms
## - Discrete 200 Hz zero-order sample-and-hold and ADC/encoder quantization
##   (0.05 rad/s wheel speed encoder quantization).
## - Deterministic RandomNumberGenerator noise + residual uncancelled IMU bias.
## - Kinematic-inertial EKF estimating beta_hat and load-transfer observer
##   estimating fz_hat_4 from delayed/noisy sensor signals.

const DEFAULT_TV_CFG_PATH := "res://Scenes/Vehicle/Config/vehicles/ter27_4wd/tv_4wd_config.json"

enum SensorMode {
	SENSOR_IDEAL = 0,
	SENSOR_TRACK_REALISM = 1
}

var mode: int = SensorMode.SENSOR_IDEAL
var seed_value: int = 20260427

# Sampling & CAN transport latency
var sample_rate_hz: float = 200.0
var can_accel_latency_ms: float = 125.0
var can_steer_latency_ms: float = 20.0

# Noise & residual uncancelled bias
var imu_noise_scale: float = 1.0
var ax_noise_std_ms2: float = 0.18
var ay_noise_std_ms2: float = 0.18
var wz_noise_std_rads: float = 0.012
var delta_noise_std_rad: float = 0.0015
var omega_noise_std_rads: float = 0.08
var ax_bias_ms2: float = 0.08
var ay_bias_ms2: float = -0.06
var wz_bias_rads: float = 0.003

# ADC & wheel encoder quantization
var accel_quant_ms2: float = 0.01
var gyro_quant_rads: float = 0.001
var steer_quant_rad: float = 0.0005
var wheel_omega_quant_rads: float = 0.05

# Vehicle physical parameters for EKF & Fz observer
var mass_kg: float = 290.0
var wheelbase_m: float = 1.54
var front_weight_dist: float = 0.45
var track_front_m: float = 1.23
var track_rear_m: float = 1.23
var h_cog_m: float = 0.29
var lltd_front: float = 0.455
var cl_a_m2: float = 6.0
var aero_balance_front: float = 0.48
var wheel_radius_m: float = 0.2032
var tau_wt_s: float = 0.035
var gravity_ms2: float = 9.81
var rho_air_kgm3: float = 1.225

# EKF tuning & bounds
var ekf_q_vx: float = 0.15
var ekf_q_vy: float = 0.35
var ekf_r_vx: float = 0.05
var ekf_r_vy_base: float = 0.25
var ekf_r_vy_ay_gain: float = 0.04
var beta_clamp_rad: float = 0.5235987756
var fz_max_n: float = 4000.0

# Internal state
var _rng: RandomNumberGenerator = RandomNumberGenerator.new()
var _initialized: bool = false
var _sample_accum_s: float = 0.0
var _tick_count: int = 0

var _accel_delay_ticks: int = 25
var _steer_delay_ticks: int = 4
var _buf_ax: PackedFloat64Array = PackedFloat64Array()
var _buf_ay: PackedFloat64Array = PackedFloat64Array()
var _buf_wz: PackedFloat64Array = PackedFloat64Array()
var _buf_delta: PackedFloat64Array = PackedFloat64Array()
var _buf_om0: PackedFloat64Array = PackedFloat64Array()
var _buf_om1: PackedFloat64Array = PackedFloat64Array()
var _buf_om2: PackedFloat64Array = PackedFloat64Array()
var _buf_om3: PackedFloat64Array = PackedFloat64Array()
var _head_accel: int = 0
var _head_steer: int = 0

var _ekf_vx: float = 0.0
var _ekf_vy: float = 0.0
var _ekf_p_vx: float = 0.2
var _ekf_p_vy: float = 0.2
var _ax_wt: float = 0.0
var _ay_wt: float = 0.0
var _beta_hat: float = 0.0
var _fz_hat_4: Array[float] = [640.0, 640.0, 780.0, 780.0]

var _last_output: Dictionary = {}


func _init() -> void:
	reset(seed_value)


func set_mode(mode_int: int) -> void:
	var clamped: int = SensorMode.SENSOR_TRACK_REALISM if mode_int == SensorMode.SENSOR_TRACK_REALISM else SensorMode.SENSOR_IDEAL
	if clamped != mode:
		mode = clamped
		reset(-1)


func set_can_latency_ms(ms: float) -> void:
	var new_val: float = maxf(ms, 0.0)
	if absf(new_val - can_accel_latency_ms) > 1e-6:
		can_accel_latency_ms = new_val
		reset(-1)


func set_imu_noise_scale(scale: float) -> void:
	imu_noise_scale = maxf(scale, 0.0)


func configure(cfg: Dictionary) -> void:
	if cfg.is_empty():
		return
	var rebuild: bool = false
	if cfg.has("sample_rate_hz"):
		var sr := maxf(float(cfg["sample_rate_hz"]), 10.0)
		if absf(sr - sample_rate_hz) > 1e-6:
			sample_rate_hz = sr
			rebuild = true
	if cfg.has("can_accel_latency_ms"):
		var lat := maxf(float(cfg["can_accel_latency_ms"]), 0.0)
		if absf(lat - can_accel_latency_ms) > 1e-6:
			can_accel_latency_ms = lat
			rebuild = true
	elif cfg.has("can_latency_ms"):
		var lat2 := maxf(float(cfg["can_latency_ms"]), 0.0)
		if absf(lat2 - can_accel_latency_ms) > 1e-6:
			can_accel_latency_ms = lat2
			rebuild = true
	if cfg.has("can_steer_latency_ms"):
		var slat := maxf(float(cfg["can_steer_latency_ms"]), 0.0)
		if absf(slat - can_steer_latency_ms) > 1e-6:
			can_steer_latency_ms = slat
			rebuild = true
	if cfg.has("imu_noise_scale"):
		imu_noise_scale = maxf(float(cfg["imu_noise_scale"]), 0.0)
	if cfg.has("ax_noise_std_ms2"):
		ax_noise_std_ms2 = float(cfg["ax_noise_std_ms2"])
	if cfg.has("ay_noise_std_ms2"):
		ay_noise_std_ms2 = float(cfg["ay_noise_std_ms2"])
	if cfg.has("wz_noise_std_rads"):
		wz_noise_std_rads = float(cfg["wz_noise_std_rads"])
	if cfg.has("delta_noise_std_rad"):
		delta_noise_std_rad = float(cfg["delta_noise_std_rad"])
	if cfg.has("omega_noise_std_rads"):
		omega_noise_std_rads = float(cfg["omega_noise_std_rads"])
	if cfg.has("ax_bias_ms2"):
		ax_bias_ms2 = float(cfg["ax_bias_ms2"])
	elif cfg.has("ax_bias"):
		ax_bias_ms2 = float(cfg["ax_bias"])
	if cfg.has("ay_bias_ms2"):
		ay_bias_ms2 = float(cfg["ay_bias_ms2"])
	elif cfg.has("ay_bias"):
		ay_bias_ms2 = float(cfg["ay_bias"])
	if cfg.has("wz_bias_rads"):
		wz_bias_rads = float(cfg["wz_bias_rads"])
	elif cfg.has("wz_bias"):
		wz_bias_rads = float(cfg["wz_bias"])
	if cfg.has("accel_quant_ms2"):
		accel_quant_ms2 = float(cfg["accel_quant_ms2"])
	if cfg.has("gyro_quant_rads"):
		gyro_quant_rads = float(cfg["gyro_quant_rads"])
	if cfg.has("steer_quant_rad"):
		steer_quant_rad = float(cfg["steer_quant_rad"])
	if cfg.has("wheel_omega_quant_rads"):
		wheel_omega_quant_rads = float(cfg["wheel_omega_quant_rads"])
	if cfg.has("mass_kg"):
		mass_kg = float(cfg["mass_kg"])
	if cfg.has("wheelbase_m"):
		wheelbase_m = float(cfg["wheelbase_m"])
	if cfg.has("front_weight_dist"):
		front_weight_dist = float(cfg["front_weight_dist"])
	if cfg.has("track_front_m"):
		track_front_m = float(cfg["track_front_m"])
	if cfg.has("track_rear_m"):
		track_rear_m = float(cfg["track_rear_m"])
	if cfg.has("h_cog_m"):
		h_cog_m = float(cfg["h_cog_m"])
	if cfg.has("lltd_front"):
		lltd_front = float(cfg["lltd_front"])
	if cfg.has("cl_a_m2"):
		cl_a_m2 = float(cfg["cl_a_m2"])
	if cfg.has("aero_balance_front"):
		aero_balance_front = float(cfg["aero_balance_front"])
	if cfg.has("wheel_radius_m"):
		wheel_radius_m = float(cfg["wheel_radius_m"])
	elif cfg.has("wheel_radius_eff_m"):
		wheel_radius_m = float(cfg["wheel_radius_eff_m"])
	if cfg.has("tau_wt_s"):
		tau_wt_s = float(cfg["tau_wt_s"])
	if cfg.has("gravity_ms2"):
		gravity_ms2 = float(cfg["gravity_ms2"])
	if cfg.has("rho_air_kgm3"):
		rho_air_kgm3 = float(cfg["rho_air_kgm3"])
	if cfg.has("ekf_q_vx"):
		ekf_q_vx = float(cfg["ekf_q_vx"])
	if cfg.has("ekf_q_vy"):
		ekf_q_vy = float(cfg["ekf_q_vy"])
	if cfg.has("ekf_r_vx"):
		ekf_r_vx = float(cfg["ekf_r_vx"])
	if cfg.has("ekf_r_vy_base"):
		ekf_r_vy_base = float(cfg["ekf_r_vy_base"])
	if cfg.has("ekf_r_vy_ay_gain"):
		ekf_r_vy_ay_gain = float(cfg["ekf_r_vy_ay_gain"])
	if cfg.has("beta_clamp_rad"):
		beta_clamp_rad = float(cfg["beta_clamp_rad"])
	if cfg.has("fz_max_n"):
		fz_max_n = float(cfg["fz_max_n"])
	if cfg.has("seed"):
		seed_value = int(cfg["seed"])
		rebuild = true
	if cfg.has("mode"):
		var m_val = cfg["mode"]
		if m_val is String:
			var m_str := String(m_val).strip_edges().to_upper()
			mode = SensorMode.SENSOR_TRACK_REALISM if (m_str == "SENSOR_TRACK_REALISM" or m_str == "1") else SensorMode.SENSOR_IDEAL
		else:
			mode = SensorMode.SENSOR_TRACK_REALISM if int(m_val) == 1 else SensorMode.SENSOR_IDEAL
		rebuild = true
	if rebuild:
		reset(seed_value)


func load_from_tv_config_json(path: String = DEFAULT_TV_CFG_PATH, keep_mode_ideal: bool = true) -> bool:
	if not FileAccess.file_exists(path):
		return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return false
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if not (parsed is Dictionary):
		return false
	var d: Dictionary = parsed
	if d.has("physical") and d["physical"] is Dictionary:
		configure(d["physical"])
	if d.has("sensor_realism") and d["sensor_realism"] is Dictionary:
		var sr: Dictionary = d["sensor_realism"].duplicate()
		if keep_mode_ideal:
			sr.erase("mode")
			sr.erase("default_mode")
		configure(sr)
	return true


func _compute_delay_ticks(latency_ms: float) -> int:
	return maxi(0, int(floor((maxf(latency_ms, 0.0) / 1000.0) * sample_rate_hz + 0.5)))


func reset(new_seed: int = -1) -> void:
	if new_seed >= 0:
		seed_value = new_seed
	_rng.seed = seed_value
	_initialized = false
	_sample_accum_s = 0.0
	_tick_count = 0

	_accel_delay_ticks = _compute_delay_ticks(can_accel_latency_ms)
	_steer_delay_ticks = _compute_delay_ticks(can_steer_latency_ms)
	var a_len: int = _accel_delay_ticks + 1
	var s_len: int = _steer_delay_ticks + 1
	_buf_ax.resize(a_len)
	_buf_ay.resize(a_len)
	_buf_ax.fill(0.0)
	_buf_ay.fill(0.0)
	_buf_wz.resize(s_len)
	_buf_delta.resize(s_len)
	_buf_om0.resize(s_len)
	_buf_om1.resize(s_len)
	_buf_om2.resize(s_len)
	_buf_om3.resize(s_len)
	_buf_wz.fill(0.0)
	_buf_delta.fill(0.0)
	_buf_om0.fill(0.0)
	_buf_om1.fill(0.0)
	_buf_om2.fill(0.0)
	_buf_om3.fill(0.0)
	_head_accel = 0
	_head_steer = 0

	_ekf_vx = 0.0
	_ekf_vy = 0.0
	_ekf_p_vx = 0.2
	_ekf_p_vy = 0.2
	_ax_wt = 0.0
	_ay_wt = 0.0
	_beta_hat = 0.0
	var fz_f: float = 0.5 * mass_kg * gravity_ms2 * front_weight_dist
	var fz_r: float = 0.5 * mass_kg * gravity_ms2 * (1.0 - front_weight_dist)
	_fz_hat_4 = [fz_f, fz_f, fz_r, fz_r]
	_last_output = {}


static func _quantize(val: float, q_step: float) -> float:
	if q_step <= 0.0:
		return val
	return floor(val / q_step + 0.5) * q_step


static func _median4(v0: float, v1: float, v2: float, v3: float) -> float:
	var a := v0
	var b := v1
	var c := v2
	var d := v3
	var tmp: float
	if a > b:
		tmp = a; a = b; b = tmp
	if c > d:
		tmp = c; c = d; d = tmp
	if a > c:
		tmp = a; a = c; c = tmp
	if b > d:
		tmp = b; b = d; d = tmp
	if b > c:
		tmp = b; b = c; c = tmp
	return 0.5 * (b + c)


func _step_ekf_and_fz(
	ax_m: float,
	ay_m: float,
	wz_m: float,
	delta_m: float,
	omega_m: Array[float],
	dt_tick: float,
	is_init: bool
) -> void:
	var vx_wheel: float = _median4(
		omega_m[0] * wheel_radius_m,
		omega_m[1] * wheel_radius_m,
		omega_m[2] * wheel_radius_m,
		omega_m[3] * wheel_radius_m
	)

	if is_init:
		_ekf_vx = vx_wheel
		_ekf_vy = 0.0
		_ekf_p_vx = 0.2
		_ekf_p_vy = 0.2
		_ax_wt = ax_m
		_ay_wt = ay_m
	else:
		var vx_pred: float = _ekf_vx + (ax_m + wz_m * _ekf_vy) * dt_tick
		var vy_pred: float = _ekf_vy + (ay_m - wz_m * _ekf_vx) * dt_tick
		_ekf_p_vx += ekf_q_vx * dt_tick
		_ekf_p_vy += ekf_q_vy * dt_tick

		var k_vx: float = _ekf_p_vx / (_ekf_p_vx + ekf_r_vx)
		_ekf_vx = vx_pred + k_vx * (vx_wheel - vx_pred)
		_ekf_p_vx = (1.0 - k_vx) * _ekf_p_vx

		var vy_kin: float = 0.0
		if absf(_ekf_vx) > 0.5:
			vy_kin = _ekf_vx * front_weight_dist * tan(clampf(delta_m, -0.6, 0.6))
		var r_vy: float = ekf_r_vy_base + ekf_r_vy_ay_gain * ay_m * ay_m
		var k_vy: float = _ekf_p_vy / (_ekf_p_vy + r_vy)
		_ekf_vy = vy_pred + k_vy * (vy_kin - vy_pred)
		_ekf_p_vy = (1.0 - k_vy) * _ekf_p_vy
		if absf(_ekf_vx) < 0.5:
			_ekf_vy *= 0.8

	if absf(_ekf_vx) > 0.2:
		_beta_hat = clampf(atan2(_ekf_vy, maxf(absf(_ekf_vx), 0.5)), -beta_clamp_rad, beta_clamp_rad)
	else:
		_beta_hat = 0.0

	var alpha_wt: float = 1.0 - exp(-dt_tick / maxf(tau_wt_s, 1e-4))
	_ax_wt += alpha_wt * (ax_m - _ax_wt)
	_ay_wt += alpha_wt * (ay_m - _ay_wt)

	var f_aero: float = 0.5 * rho_air_kgm3 * cl_a_m2 * _ekf_vx * _ekf_vx
	var fz_f_base: float = mass_kg * gravity_ms2 * front_weight_dist + f_aero * aero_balance_front
	var fz_r_base: float = mass_kg * gravity_ms2 * (1.0 - front_weight_dist) + f_aero * (1.0 - aero_balance_front)
	var dfz_long: float = mass_kg * _ax_wt * h_cog_m / maxf(wheelbase_m, 0.1)
	var fz_f: float = fz_f_base - dfz_long
	var fz_r: float = fz_r_base + dfz_long
	var dfz_lat_f: float = lltd_front * mass_kg * _ay_wt * h_cog_m / maxf(track_front_m, 0.1)
	var dfz_lat_r: float = (1.0 - lltd_front) * mass_kg * _ay_wt * h_cog_m / maxf(track_rear_m, 0.1)

	_fz_hat_4[0] = clampf(0.5 * fz_f - dfz_lat_f, 0.0, fz_max_n)
	_fz_hat_4[1] = clampf(0.5 * fz_f + dfz_lat_f, 0.0, fz_max_n)
	_fz_hat_4[2] = clampf(0.5 * fz_r - dfz_lat_r, 0.0, fz_max_n)
	_fz_hat_4[3] = clampf(0.5 * fz_r + dfz_lat_r, 0.0, fz_max_n)


func _run_200hz_tick(
	ax: float,
	ay: float,
	wz: float,
	delta: float,
	omega_4: Array[float],
	dt_tick: float
) -> Dictionary:
	var is_init: bool = not _initialized
	if is_init:
		_buf_ax.fill(ax)
		_buf_ay.fill(ay)
		_buf_wz.fill(wz)
		_buf_delta.fill(delta)
		_buf_om0.fill(omega_4[0])
		_buf_om1.fill(omega_4[1])
		_buf_om2.fill(omega_4[2])
		_buf_om3.fill(omega_4[3])
		_head_accel = 0
		_head_steer = 0
		_initialized = true

	var a_len: int = _buf_ax.size()
	_buf_ax[_head_accel] = ax
	_buf_ay[_head_accel] = ay
	var next_a: int = (_head_accel + 1) % a_len
	var ax_d: float = _buf_ax[next_a]
	var ay_d: float = _buf_ay[next_a]
	_head_accel = next_a

	var s_len: int = _buf_delta.size()
	_buf_wz[_head_steer] = wz
	_buf_delta[_head_steer] = delta
	_buf_om0[_head_steer] = omega_4[0]
	_buf_om1[_head_steer] = omega_4[1]
	_buf_om2[_head_steer] = omega_4[2]
	_buf_om3[_head_steer] = omega_4[3]
	var next_s: int = (_head_steer + 1) % s_len
	var wz_d: float = _buf_wz[next_s]
	var delta_d: float = _buf_delta[next_s]
	var om0_d: float = _buf_om0[next_s]
	var om1_d: float = _buf_om1[next_s]
	var om2_d: float = _buf_om2[next_s]
	var om3_d: float = _buf_om3[next_s]
	_head_steer = next_s

	var n_scale: float = imu_noise_scale
	var ax_n: float = ax_d + ax_bias_ms2 + (ax_noise_std_ms2 * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var ay_n: float = ay_d + ay_bias_ms2 + (ay_noise_std_ms2 * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var wz_n: float = wz_d + wz_bias_rads + (wz_noise_std_rads * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var delta_n: float = delta_d + (delta_noise_std_rad * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var om0_n: float = om0_d + (omega_noise_std_rads * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var om1_n: float = om1_d + (omega_noise_std_rads * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var om2_n: float = om2_d + (omega_noise_std_rads * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)
	var om3_n: float = om3_d + (omega_noise_std_rads * n_scale * _rng.randfn(0.0, 1.0) if n_scale > 0.0 else 0.0)

	var ax_m: float = _quantize(ax_n, accel_quant_ms2)
	var ay_m: float = _quantize(ay_n, accel_quant_ms2)
	var wz_m: float = _quantize(wz_n, gyro_quant_rads)
	var delta_m: float = _quantize(delta_n, steer_quant_rad)
	var omega_m: Array[float] = [
		_quantize(om0_n, wheel_omega_quant_rads),
		_quantize(om1_n, wheel_omega_quant_rads),
		_quantize(om2_n, wheel_omega_quant_rads),
		_quantize(om3_n, wheel_omega_quant_rads)
	]

	_step_ekf_and_fz(ax_m, ay_m, wz_m, delta_m, omega_m, dt_tick, is_init)
	_tick_count += 1

	return {
		"ax": ax_m,
		"ay": ay_m,
		"wz": wz_m,
		"vx": _ekf_vx,
		"vy": _ekf_vy,
		"delta": delta_m,
		"fz_4": _fz_hat_4.duplicate(),
		"omega_4": omega_m,
		"beta_hat": _beta_hat,
		"fz_hat_4": _fz_hat_4.duplicate(),
		"mode": mode,
		"mode_name": "SENSOR_TRACK_REALISM",
		"can_latency_ms": can_accel_latency_ms,
		"can_steer_latency_ms": can_steer_latency_ms,
		"imu_noise_scale": imu_noise_scale
	}


func step(
	ax: float,
	ay: float,
	wz: float,
	vx: float,
	vy: float,
	delta: float,
	fz_4: Array[float],
	omega_4: Array[float],
	dt: float
) -> Dictionary:
	if mode == SensorMode.SENSOR_IDEAL:
		var beta_gt: float = atan2(vy, maxf(absf(vx), 0.5)) if absf(vx) > 0.1 else 0.0
		_ekf_vx = vx
		_ekf_vy = vy
		_beta_hat = beta_gt
		for i in range(4):
			_fz_hat_4[i] = fz_4[i]
		_last_output = {
			"ax": ax,
			"ay": ay,
			"wz": wz,
			"vx": vx,
			"vy": vy,
			"delta": delta,
			"fz_4": fz_4.duplicate(),
			"omega_4": omega_4.duplicate(),
			"beta_hat": beta_gt,
			"fz_hat_4": fz_4.duplicate(),
			"mode": SensorMode.SENSOR_IDEAL,
			"mode_name": "SENSOR_IDEAL",
			"can_latency_ms": 0.0,
			"can_steer_latency_ms": 0.0,
			"imu_noise_scale": 0.0
		}
		return _last_output

	var sample_dt: float = 1.0 / maxf(sample_rate_hz, 10.0)
	if not _initialized:
		_last_output = _run_200hz_tick(ax, ay, wz, delta, omega_4, sample_dt)
		_sample_accum_s = 0.0
		return _last_output

	if dt > 0.0:
		_sample_accum_s += dt
		while _sample_accum_s >= sample_dt - 1e-9:
			_sample_accum_s -= sample_dt
			if _sample_accum_s < 0.0:
				_sample_accum_s = 0.0
			_last_output = _run_200hz_tick(ax, ay, wz, delta, omega_4, sample_dt)

	return _last_output


func get_last_output() -> Dictionary:
	return _last_output


func get_diagnostics() -> Dictionary:
	return {
		"mode": mode,
		"mode_name": "SENSOR_TRACK_REALISM" if mode == SensorMode.SENSOR_TRACK_REALISM else "SENSOR_IDEAL",
		"seed": seed_value,
		"sample_rate_hz": sample_rate_hz,
		"can_accel_latency_ms": can_accel_latency_ms,
		"can_steer_latency_ms": can_steer_latency_ms,
		"accel_delay_ticks": _accel_delay_ticks,
		"steer_delay_ticks": _steer_delay_ticks,
		"imu_noise_scale": imu_noise_scale,
		"ekf_vx": _ekf_vx,
		"ekf_vy": _ekf_vy,
		"beta_hat": _beta_hat,
		"fz_hat_4": _fz_hat_4.duplicate(),
		"tick_count": _tick_count
	}
