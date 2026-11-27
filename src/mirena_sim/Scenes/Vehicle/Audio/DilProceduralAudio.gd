class_name DilProceduralAudio
extends Node

## Real-time DiL procedural audio synthesizer supporting 3 selectable profiles:
##   0: FS Electric (TeR27 4WD) — 4 in-hub planetary gear + inverter harmonics + regen tone
##   1: F1 V10 (19000 RPM)      — 7-speed sequential, 5th firing order, 45ms cut upshift, blip, overrun crackle
##   2: F1 V12 (16500 RPM)      — 6-speed sequential, 6th firing order, intake resonance, metallic overrun
## Common stereo tire & aero layer: road roll, vx^2 wind rush, and 4-wheel slip squeal/scrub.
## Zero external WAV dependencies; safe in headless runs.

const PROFILE_FS_ELECTRIC: int = 0
const PROFILE_F1_V10: int = 1
const PROFILE_F1_V12: int = 2

const MIX_RATE: float = 48000.0
const BUFFER_LENGTH_S: float = 0.08
const MAX_CHUNK_FRAMES: int = 1536

const V10_UPSHIFT_SPEEDS_MPS: Array[float] = [9.44, 13.61, 17.78, 21.67, 25.28, 28.61, 32.50]
const V10_MIN_RPM: float = 6000.0
const V10_MAX_RPM: float = 19000.0
const V10_UPSHIFT_RPM: float = 18600.0
const V10_DOWNSHIFT_RPM: float = 11800.0

const V12_UPSHIFT_SPEEDS_MPS: Array[float] = [10.56, 15.00, 19.44, 23.89, 28.06, 32.50]
const V12_MIN_RPM: float = 5500.0
const V12_MAX_RPM: float = 16500.0
const V12_UPSHIFT_RPM: float = 16100.0
const V12_DOWNSHIFT_RPM: float = 10200.0

const TEST_REV_DURATION_S: float = 1.20
const IGNITION_CUT_DURATION_S: float = 0.045
const DOWNSHIFT_BLIP_DURATION_S: float = 0.090

var enabled: bool = false
var volume: float = 0.75
var sound_profile: int = PROFILE_FS_ELECTRIC

var master_db: float = -2.5
var engine_db: float = 0.0
var tire_db: float = -2.0

var _player: AudioStreamPlayer = null
var _generator: AudioStreamGenerator = null
var _is_headless: bool = false

# Smoothed telemetry / synthesis targets
var _target_vx_mps: float = 0.0
var _target_throttle: float = 0.0
var _target_brake: float = 0.0
var _target_p_dc_kw: float = 0.0
var _target_tire_left: float = 0.0
var _target_tire_right: float = 0.0
var _target_roll_gain: float = 0.0
var _target_wind_gain: float = 0.0
var _target_squeal_freq_hz: float = 1050.0

var _wheel_omega: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var _wheel_torque_abs: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])

# Per-wheel phase accumulators for FS Electric (4 motors + 4 inverters)
var _ev_motor_phase: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var _ev_inv_phase: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var _ev_sub_phase: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var _ev_smooth_freq: PackedFloat32Array = PackedFloat32Array([65.0, 65.0, 65.0, 65.0])
var _ev_smooth_inv_freq: PackedFloat32Array = PackedFloat32Array([1900.0, 1900.0, 1900.0, 1900.0])
var _ev_smooth_gain: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var _ev_smooth_inv_gain: PackedFloat32Array = PackedFloat32Array([0.0, 0.0, 0.0, 0.0])

# ICE (F1 V10 / V12) gearbox & oscillator states
var _ice_gear: int = 1
var _ice_rpm: float = 6000.0
var _ice_target_rpm: float = 6000.0
var _ice_smooth_rpm: float = 6000.0
var _ice_smooth_load: float = 0.0
var _ice_ign_cut_timer: float = 0.0
var _ice_blip_timer: float = 0.0
var _ice_crackle_timer: float = 0.0
var _ice_crackle_env: float = 0.0
var _ice_prev_throttle: float = 0.0
var _ice_shift_pop_env: float = 0.0

var _ice_phase_half: float = 0.0
var _ice_phase_fire: float = 0.0
var _ice_phase_2x: float = 0.0
var _ice_phase_3x: float = 0.0
var _ice_phase_15x: float = 0.0
var _ice_phase_intake: float = 0.0

# Tire & Aero DSP filter states
var _rng_state: int = 135792468
var _lp_roll_l: float = 0.0
var _lp_roll_r: float = 0.0
var _lp_wind: float = 0.0
var _hp_wind: float = 0.0
var _lp_squeal_l: float = 0.0
var _lp_squeal_r: float = 0.0
var _squeal_phase: float = 0.0
var _smooth_roll_gain: float = 0.0
var _smooth_wind_gain: float = 0.0
var _smooth_tire_l: float = 0.0
var _smooth_tire_r: float = 0.0
var _smooth_squeal_freq: float = 1050.0

# Test rev sweep state
var _test_rev_active: bool = false
var _test_rev_elapsed: float = 0.0
var _stepped_this_frame: bool = false
var _last_step_msec: int = 0

# Public diagnostic fields (compatible with legacy & telemetry checks)
var last_motor_freq_hz: float = 180.0
var last_inv_freq_hz: float = 360.0
var last_motor_gain: float = 0.0
var last_inv_gain: float = 0.0
var last_front_scrub_gain: float = 0.0
var last_rear_scrub_gain: float = 0.0
var last_ice_rpm: float = 6000.0
var last_ice_gear: int = 1


func _ready() -> void:
	_is_headless = (DisplayServer.get_name() == "headless") \
		or OS.has_feature("headless") \
		or (AudioServer.get_driver_name() == "Dummy")
	master_db = _linear_to_db_clamped(volume)
	if _is_headless:
		return
	_setup_audio_player()
	if enabled:
		_start_player_safe()


func _exit_tree() -> void:
	_stop_player_safe()


func _process(delta: float) -> void:
	if not _test_rev_active:
		return
	var now_msec := Time.get_ticks_msec()
	# If step_audio() already ran this frame, let step_audio drive the buffer;
	# otherwise (e.g. game paused in menu) advance the test rev here.
	if _stepped_this_frame and (now_msec - _last_step_msec) < 35:
		_stepped_this_frame = false
		return
	_stepped_this_frame = false
	_advance_test_rev(clampf(delta, 0.001, 0.05))
	_pump_audio_buffer()


func _setup_audio_player() -> void:
	if is_instance_valid(_player):
		return
	var safe_mix_rate := maxf(MIX_RATE, 8000.0)
	_generator = AudioStreamGenerator.new()
	_generator.mix_rate = safe_mix_rate
	_generator.buffer_length = BUFFER_LENGTH_S
	_player = AudioStreamPlayer.new()
	_player.name = "DilAudioPlayer"
	_player.stream = _generator
	_player.bus = "Master"
	add_child(_player)


func _reset_dsp_state() -> void:
	for i in range(4):
		_ev_motor_phase[i] = 0.0
		_ev_inv_phase[i] = 0.0
		_ev_sub_phase[i] = 0.0
		_ev_smooth_gain[i] = 0.0
		_ev_smooth_inv_gain[i] = 0.0
	_ice_phase_half = 0.0
	_ice_phase_fire = 0.0
	_ice_phase_2x = 0.0
	_ice_phase_3x = 0.0
	_ice_phase_15x = 0.0
	_ice_phase_intake = 0.0
	_ice_ign_cut_timer = 0.0
	_ice_blip_timer = 0.0
	_ice_crackle_timer = 0.0
	_ice_crackle_env = 0.0
	_ice_shift_pop_env = 0.0
	_lp_roll_l = 0.0
	_lp_roll_r = 0.0
	_lp_wind = 0.0
	_hp_wind = 0.0
	_lp_squeal_l = 0.0
	_lp_squeal_r = 0.0
	_squeal_phase = 0.0
	_smooth_roll_gain = 0.0
	_smooth_wind_gain = 0.0
	_smooth_tire_l = 0.0
	_smooth_tire_r = 0.0


func _start_player_safe() -> void:
	if _is_headless or not is_inside_tree():
		return
	if not is_instance_valid(_player):
		_setup_audio_player()
	if not is_instance_valid(_player) or not _player.is_inside_tree():
		return
	if _generator == null:
		_generator = AudioStreamGenerator.new()
		_generator.mix_rate = maxf(MIX_RATE, 8000.0)
		_generator.buffer_length = BUFFER_LENGTH_S
	if _player.stream != _generator:
		_player.stream = _generator
	if not _player.playing:
		_player.play()


func _stop_player_safe() -> void:
	if is_instance_valid(_player) and _player.is_inside_tree() and _player.playing:
		_player.stop()
	_reset_dsp_state()


func set_enabled(on: bool) -> void:
	enabled = on
	if _is_headless:
		if not enabled and not _test_rev_active:
			_reset_dsp_state()
		return
	if enabled or _test_rev_active:
		_start_player_safe()
	else:
		_stop_player_safe()


func set_volume(vol_0_1: float) -> void:
	volume = clampf(vol_0_1 if is_finite(vol_0_1) else 0.75, 0.0, 1.0)
	master_db = _linear_to_db_clamped(volume)


func set_sound_profile(profile_id: int) -> void:
	var clamped_id := clampi(profile_id, PROFILE_FS_ELECTRIC, PROFILE_F1_V12)
	if clamped_id == sound_profile:
		return
	sound_profile = clamped_id
	_ice_gear = 1
	_ice_ign_cut_timer = 0.0
	_ice_blip_timer = 0.0
	_ice_crackle_timer = 0.0
	_ice_crackle_env = 0.0
	_ice_shift_pop_env = 0.0
	if sound_profile == PROFILE_F1_V10:
		_ice_rpm = V10_MIN_RPM
		_ice_target_rpm = V10_MIN_RPM
		_ice_smooth_rpm = V10_MIN_RPM
	elif sound_profile == PROFILE_F1_V12:
		_ice_rpm = V12_MIN_RPM
		_ice_target_rpm = V12_MIN_RPM
		_ice_smooth_rpm = V12_MIN_RPM
	last_ice_rpm = _ice_rpm
	last_ice_gear = _ice_gear


func get_sound_profile() -> int:
	return sound_profile


func set_volumes(p_master_db: float, p_engine_db: float, p_tire_db: float) -> void:
	master_db = clampf(p_master_db if is_finite(p_master_db) else -2.5, -60.0, 6.0)
	engine_db = clampf(p_engine_db if is_finite(p_engine_db) else 0.0, -60.0, 6.0)
	tire_db = clampf(p_tire_db if is_finite(p_tire_db) else -2.0, -60.0, 6.0)
	volume = clampf(_db_to_linear_safe(master_db), 0.0, 1.0)


func get_volumes() -> Dictionary:
	return {
		"master_db": master_db,
		"engine_db": engine_db,
		"tire_db": tire_db,
	}


func trigger_test_rev() -> void:
	_test_rev_active = true
	_test_rev_elapsed = 0.0
	_ice_gear = 1
	_ice_ign_cut_timer = 0.0
	_ice_blip_timer = 0.0
	_ice_crackle_timer = 0.0
	_ice_crackle_env = 0.0
	_ice_prev_throttle = 0.0
	if sound_profile == PROFILE_F1_V10:
		_ice_rpm = V10_MIN_RPM
		_ice_smooth_rpm = V10_MIN_RPM
	elif sound_profile == PROFILE_F1_V12:
		_ice_rpm = V12_MIN_RPM
		_ice_smooth_rpm = V12_MIN_RPM
	if not _is_headless:
		_start_player_safe()


static func _linear_to_db_clamped(lin: float) -> float:
	if lin <= 0.001:
		return -60.0
	return clampf(20.0 * log(lin) / log(10.0), -60.0, 6.0)


static func _db_to_linear_safe(db_val: float) -> float:
	if not is_finite(db_val) or db_val <= -59.5:
		return 0.0
	return pow(10.0, clampf(db_val, -60.0, 6.0) / 20.0)


## Pure math calculation for motor pitch, inverter harmonic gain, and front/rear tire scrub gains.
## Preserved for backward compatibility with unit tests and external callers.
static func compute_synth_params(
	omega_avg_rad_s: float,
	p_dc_kw: float,
	sat_f: float,
	sat_r: float,
	beta_dyn_rad: float
) -> Dictionary:
	var om := maxf(absf(omega_avg_rad_s), 0.0) if is_finite(omega_avg_rad_s) else 0.0
	var p_kw := p_dc_kw if is_finite(p_dc_kw) else 0.0
	var sf := clampf(sat_f if is_finite(sat_f) else 0.0, 0.0, 1.5)
	var sr := clampf(sat_r if is_finite(sat_r) else 0.0, 0.0, 1.5)
	var bd := absf(beta_dyn_rad) if is_finite(beta_dyn_rad) else 0.0

	var speed_frac := clampf(om / 156.0, 0.0, 1.0)
	var motor_freq := lerpf(160.0, 1450.0, speed_frac)
	var load_frac := clampf(absf(p_kw) / 80.0, 0.0, 1.0)
	var motor_gain := lerpf(0.08, 0.30, load_frac) * clampf(om / 10.0, 0.12, 1.0)
	var is_regen := p_kw < -1.0
	var inv_gain := motor_gain * (0.72 if is_regen else 0.48) * (0.25 + 0.75 * load_frac)

	var front_scrub := clampf((sf - 0.68) / 0.32, 0.0, 1.0) * 0.25 * clampf(om / 14.0, 0.0, 1.0)
	var beta_boost := 1.0 + clampf(bd / 0.14, 0.0, 1.2)
	var rear_scrub := clampf((sr - 0.68) / 0.32, 0.0, 1.0) * 0.30 * beta_boost * clampf(om / 14.0, 0.0, 1.0)

	return {
		"motor_freq_hz": motor_freq,
		"inv_freq_hz": motor_freq * (1.85 if is_regen else 2.0),
		"motor_gain": motor_gain,
		"inv_gain": inv_gain,
		"front_scrub_gain": front_scrub,
		"rear_scrub_gain": rear_scrub,
		"is_regen": is_regen,
	}


## Synthesizes one mono sample (compatibility helper, clamped in [-0.98, 0.98]).
func synthesize_next_sample(
	m_freq_hz: float,
	inv_freq_hz: float,
	m_gain: float,
	i_gain: float,
	f_gain: float,
	r_gain: float,
	mix_rate_hz: float = MIX_RATE
) -> float:
	var sr := maxf(mix_rate_hz if is_finite(mix_rate_hz) else MIX_RATE, 8000.0)
	var max_f := 0.42 * sr
	var inv_sr := 1.0 / sr

	var mf := clampf(m_freq_hz if is_finite(m_freq_hz) else 180.0, 40.0, max_f)
	var inf := clampf(inv_freq_hz if is_finite(inv_freq_hz) else 360.0, 40.0, max_f)
	var sf_sub := clampf(0.5 * mf, 40.0, max_f)
	var mg := m_gain if is_finite(m_gain) else 0.0
	var ig := i_gain if is_finite(i_gain) else 0.0
	var fg := f_gain if is_finite(f_gain) else 0.0
	var rg := r_gain if is_finite(r_gain) else 0.0

	_ev_motor_phase[0] = fmod(_ev_motor_phase[0] + mf * inv_sr, 1.0)
	_ev_inv_phase[0] = fmod(_ev_inv_phase[0] + inf * inv_sr, 1.0)
	_ev_sub_phase[0] = fmod(_ev_sub_phase[0] + sf_sub * inv_sr, 1.0)

	var sig_motor := sin(TAU * _ev_motor_phase[0]) * mg \
		+ sin(TAU * _ev_inv_phase[0]) * ig \
		+ sin(TAU * _ev_sub_phase[0]) * (0.28 * mg)

	var white := _next_white_noise()
	_lp_roll_l = clampf(lerpf(_lp_roll_l, white, 0.18), -4.0, 4.0)
	var sig_front := (_lp_roll_l - lerpf(_lp_roll_l, 0.0, 0.05)) * fg
	_lp_squeal_r = clampf(lerpf(_lp_squeal_r, white, 0.36), -4.0, 4.0)
	var sig_rear := (white - _lp_squeal_r) * rg

	var s := tanh((sig_motor + sig_front + sig_rear) * 1.45)
	if not is_finite(s):
		s = 0.0
	return clampf(s, -0.98, 0.98)


func _next_white_noise() -> float:
	_rng_state = (_rng_state * 1664525 + 1013904223) & 0x7fffffff
	return (float(_rng_state) / 1073741824.0) - 1.0


func _advance_test_rev(dt: float) -> void:
	_test_rev_elapsed += dt
	if _test_rev_elapsed >= TEST_REV_DURATION_S:
		_test_rev_active = false
		_target_throttle = 0.0
		_target_p_dc_kw = 0.0
		_target_vx_mps = 0.0
		for i in range(4):
			_wheel_omega[i] = 0.0
			_wheel_torque_abs[i] = 0.0
		if not enabled and not _is_headless:
			_stop_player_safe()
		return

	var u := _test_rev_elapsed / TEST_REV_DURATION_S
	var vx := 0.0
	var thr := 0.0
	var p_kw := 0.0
	if u < 0.72:
		var ramp := clampf(u / 0.72, 0.0, 1.0)
		vx = lerpf(4.0, 29.5, ramp * ramp * (3.0 - 2.0 * ramp))
		thr = 1.0
		p_kw = lerpf(28.0, 80.0, clampf(ramp * 1.6, 0.0, 1.0))
	else:
		var decel := clampf((u - 0.72) / 0.28, 0.0, 1.0)
		vx = lerpf(29.5, 11.0, decel)
		thr = 0.0
		p_kw = lerpf(-25.0, -8.0, decel)

	var om := vx / 0.205
	for i in range(4):
		_wheel_omega[i] = om
		_wheel_torque_abs[i] = 180.0 if thr > 0.1 else 95.0
	_target_vx_mps = vx
	_target_throttle = thr
	_target_brake = 0.0 if thr > 0.1 else 0.45
	_target_p_dc_kw = p_kw
	_target_tire_left = 0.03 * clampf(vx / 30.0, 0.0, 1.0)
	_target_tire_right = _target_tire_left
	_target_roll_gain = 0.08 * clampf(vx / 32.0, 0.0, 1.0)
	var v_n := clampf(vx / 32.0, 0.0, 1.0)
	_target_wind_gain = 0.12 * v_n * v_n
	_update_ice_gearbox(dt)


func step_audio(
	omega_avg_rad_s: float,
	p_dc_kw: float,
	sat_f: float,
	sat_r: float,
	beta_dyn_rad: float
) -> void:
	_stepped_this_frame = true
	_last_step_msec = Time.get_ticks_msec()

	var dt := 1.0 / 60.0
	if _test_rev_active:
		_advance_test_rev(dt)
		_pump_audio_buffer()
		return

	var prm := compute_synth_params(omega_avg_rad_s, p_dc_kw, sat_f, sat_r, beta_dyn_rad)
	last_motor_freq_hz = float(prm["motor_freq_hz"])
	last_inv_freq_hz = float(prm["inv_freq_hz"])
	last_motor_gain = float(prm["motor_gain"])
	last_inv_gain = float(prm["inv_gain"])
	last_front_scrub_gain = float(prm["front_scrub_gain"])
	last_rear_scrub_gain = float(prm["rear_scrub_gain"])

	var om_avg := maxf(absf(omega_avg_rad_s), 0.0) if is_finite(omega_avg_rad_s) else 0.0
	var p_kw := p_dc_kw if is_finite(p_dc_kw) else 0.0
	_target_p_dc_kw = p_kw
	_target_vx_mps = om_avg * 0.205
	_target_throttle = clampf(p_kw / 80.0, 0.0, 1.0) if p_kw > 0.5 else 0.0
	_target_brake = clampf(-p_kw / 25.0, 0.0, 1.0) if p_kw < -0.5 else 0.0

	var default_trq := clampf(absf(p_kw) * 3.0, 0.0, 260.0)
	for i in range(4):
		_wheel_omega[i] = om_avg
		_wheel_torque_abs[i] = default_trq

	var slip_l := 0.0
	var slip_r := 0.0
	var slip_peak := 0.0

	# Enrich with real 4-wheel & driver input state from MirenaCarDynamics parent when present
	var parent_node := get_parent()
	if parent_node != null:
		if "gas" in parent_node:
			_target_throttle = clampf(float(parent_node.get("gas")), 0.0, 1.0)
		if "brake" in parent_node:
			_target_brake = clampf(float(parent_node.get("brake")), 0.0, 1.0)
		var mdl = parent_node.get("model") if ("model" in parent_node) else null
		if mdl != null:
			if "linear_velocity_ros" in mdl:
				var lv: Vector3 = mdl.linear_velocity_ros
				if is_finite(lv.x):
					_target_vx_mps = maxf(absf(lv.x), 0.0)
			if "wheels" in mdl and mdl.wheels.size() >= 4:
				var d_trq = mdl.get("wheel_drive_torque_nm") if ("wheel_drive_torque_nm" in mdl) else null
				var r_trq = mdl.get("wheel_regen_brake_nm") if ("wheel_regen_brake_nm" in mdl) else null
				for i in range(4):
					var w = mdl.wheels[i]
					if w != null:
						_wheel_omega[i] = absf(float(w.omega)) if is_finite(w.omega) else om_avg
						var td := float(d_trq[i]) if (d_trq is Array and d_trq.size() > i) else 0.0
						var tr := float(r_trq[i]) if (r_trq is Array and r_trq.size() > i) else 0.0
						_wheel_torque_abs[i] = clampf(absf(td - tr), 0.0, 320.0) if (d_trq != null or r_trq != null) else default_trq
						var alpha_abs := absf(float(w.alpha)) if ("alpha" in w and is_finite(w.alpha)) else 0.0
						var kappa_abs := absf(float(w.kappa)) if ("kappa" in w and is_finite(w.kappa)) else 0.0
						var slip_i := clampf((alpha_abs - 0.06) / 0.16, 0.0, 1.0) * 0.75 + clampf((kappa_abs - 0.07) / 0.18, 0.0, 1.0) * 0.65
						slip_peak = maxf(slip_peak, slip_i)
						if (i % 2) == 0:
							slip_l += slip_i * 0.5
						else:
							slip_r += slip_i * 0.5

	var base_scrub_l := (last_front_scrub_gain + last_rear_scrub_gain) * 0.5
	var base_scrub_r := base_scrub_l
	var speed_gate := clampf(_target_vx_mps / 2.5, 0.0, 1.0)
	_target_tire_left = clampf((base_scrub_l + slip_l * 0.34) * speed_gate, 0.0, 0.75)
	_target_tire_right = clampf((base_scrub_r + slip_r * 0.34) * speed_gate, 0.0, 0.75)
	_target_squeal_freq_hz = lerpf(880.0, 1460.0, clampf(slip_peak, 0.0, 1.0))

	var v_norm := clampf(_target_vx_mps / 32.0, 0.0, 1.0)
	_target_roll_gain = 0.11 * v_norm * clampf(_target_vx_mps / 1.5, 0.0, 1.0)
	_target_wind_gain = 0.16 * v_norm * v_norm

	_update_ice_gearbox(dt)
	_pump_audio_buffer()


func _update_ice_gearbox(dt: float) -> void:
	if sound_profile == PROFILE_FS_ELECTRIC:
		return

	var is_v10 := (sound_profile == PROFILE_F1_V10)
	var min_rpm := V10_MIN_RPM if is_v10 else V12_MIN_RPM
	var max_rpm := V10_MAX_RPM if is_v10 else V12_MAX_RPM
	var up_rpm := V10_UPSHIFT_RPM if is_v10 else V12_UPSHIFT_RPM
	var dn_rpm := V10_DOWNSHIFT_RPM if is_v10 else V12_DOWNSHIFT_RPM
	var speeds := V10_UPSHIFT_SPEEDS_MPS if is_v10 else V12_UPSHIFT_SPEEDS_MPS
	var max_gear := speeds.size()

	_ice_gear = clampi(_ice_gear, 1, max_gear)
	var top_v := speeds[_ice_gear - 1]
	var rpm_from_v := min_rpm + (_target_vx_mps / maxf(top_v, 1.0)) * (max_rpm - min_rpm)
	var idle_flare := _target_throttle * 2600.0 * clampf(1.0 - _target_vx_mps / 4.0, 0.0, 1.0)
	var raw_rpm := clampf(maxf(rpm_from_v, min_rpm + idle_flare), min_rpm, max_rpm)

	# Sequential upshift / downshift with hysteresis
	if raw_rpm >= up_rpm and _ice_gear < max_gear and _target_throttle > 0.15:
		_ice_gear += 1
		_ice_ign_cut_timer = IGNITION_CUT_DURATION_S
		_ice_shift_pop_env = 1.0
		top_v = speeds[_ice_gear - 1]
		raw_rpm = clampf(min_rpm + (_target_vx_mps / maxf(top_v, 1.0)) * (max_rpm - min_rpm), min_rpm, max_rpm)
	elif raw_rpm <= dn_rpm and _ice_gear > 1:
		_ice_gear -= 1
		_ice_blip_timer = DOWNSHIFT_BLIP_DURATION_S
		top_v = speeds[_ice_gear - 1]
		raw_rpm = clampf(min_rpm + (_target_vx_mps / maxf(top_v, 1.0)) * (max_rpm - min_rpm), min_rpm, max_rpm)

	# Overrun crackle trigger on high-RPM throttle lift-off
	if _ice_rpm > (max_rpm * 0.60) and _ice_prev_throttle > 0.35 and _target_throttle < 0.15:
		_ice_crackle_timer = 0.45
	_ice_prev_throttle = _target_throttle

	if _ice_ign_cut_timer > 0.0:
		_ice_ign_cut_timer = maxf(_ice_ign_cut_timer - dt, 0.0)
	if _ice_blip_timer > 0.0:
		_ice_blip_timer = maxf(_ice_blip_timer - dt, 0.0)
		raw_rpm = clampf(raw_rpm + 1900.0 * (_ice_blip_timer / DOWNSHIFT_BLIP_DURATION_S), min_rpm, max_rpm)
	if _ice_crackle_timer > 0.0:
		_ice_crackle_timer = maxf(_ice_crackle_timer - dt, 0.0)

	_ice_target_rpm = raw_rpm
	_ice_rpm = lerpf(_ice_rpm, _ice_target_rpm, 0.42)
	last_ice_rpm = _ice_rpm
	last_ice_gear = _ice_gear


func _pump_audio_buffer() -> void:
	if _is_headless or (not enabled and not _test_rev_active) or not is_inside_tree():
		return
	if not is_instance_valid(_player) or not _player.is_inside_tree() or not _player.playing:
		return
	if not _player.has_stream_playback():
		return

	var pb := _player.get_stream_playback() as AudioStreamGeneratorPlayback
	if pb == null:
		return

	var to_fill := mini(pb.get_frames_available(), MAX_CHUNK_FRAMES)
	if to_fill <= 0:
		return

	var sr := maxf(_generator.mix_rate if _generator != null else MIX_RATE, 8000.0)
	var inv_sr := 1.0 / sr
	var max_f := 0.44 * sr

	var master_lin := _db_to_linear_safe(master_db)
	var eng_lin := _db_to_linear_safe(engine_db) * master_lin
	var tire_lin := _db_to_linear_safe(tire_db) * master_lin

	# Precompute EV targets per wheel before sample loop
	var ev_tgt_f := PackedFloat32Array([65.0, 65.0, 65.0, 65.0])
	var ev_tgt_inv_f := PackedFloat32Array([1900.0, 1900.0, 1900.0, 1900.0])
	var ev_tgt_g := PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
	var ev_tgt_inv_g := PackedFloat32Array([0.0, 0.0, 0.0, 0.0])

	var is_regen := _target_p_dc_kw < -1.0
	var p_frac := clampf(absf(_target_p_dc_kw) / 80.0, 0.0, 1.0)
	var regen_frac := clampf(-_target_p_dc_kw / 25.0, 0.0, 1.0) if is_regen else 0.0

	if sound_profile == PROFILE_FS_ELECTRIC:
		for i in range(4):
			var om := _wheel_omega[i]
			var trq_frac := clampf(_wheel_torque_abs[i] / 240.0, 0.0, 1.0)
			var load_i := maxf(trq_frac, p_frac * 0.65)
			var spd_frac := clampf(om / 156.0, 0.0, 1.0)
			var f_m := clampf(lerpf(62.0, 1380.0, spd_frac), 40.0, max_f)
			var f_inv := clampf((f_m * 1.85 + 750.0 * regen_frac) if is_regen else (1650.0 + f_m * 1.45), 80.0, max_f)
			var spd_gate := clampf(om / 8.0, 0.08, 1.0)
			ev_tgt_f[i] = f_m
			ev_tgt_inv_f[i] = f_inv
			ev_tgt_g[i] = lerpf(0.05, 0.24, load_i) * spd_gate
			ev_tgt_inv_g[i] = lerpf(0.02, 0.22 if is_regen else 0.19, maxf(load_i, regen_frac)) * spd_gate

	var ice_load_target := clampf(maxf(_target_throttle, 1.0 if _ice_blip_timer > 0.0 else 0.0), 0.0, 1.0)
	var is_v10 := (sound_profile == PROFILE_F1_V10)
	var fire_order := 5.0 if is_v10 else 6.0
	var crackle_active := _ice_crackle_timer > 0.0
	var ign_cut_mult := 0.14 if _ice_ign_cut_timer > 0.0 else 1.0

	for _k in range(to_fill):
		var eng_l := 0.0
		var eng_r := 0.0

		if sound_profile == PROFILE_FS_ELECTRIC:
			for i in range(4):
				_ev_smooth_freq[i] = lerpf(_ev_smooth_freq[i], ev_tgt_f[i], 0.008)
				_ev_smooth_inv_freq[i] = lerpf(_ev_smooth_inv_freq[i], ev_tgt_inv_f[i], 0.008)
				_ev_smooth_gain[i] = lerpf(_ev_smooth_gain[i], ev_tgt_g[i], 0.012)
				_ev_smooth_inv_gain[i] = lerpf(_ev_smooth_inv_gain[i], ev_tgt_inv_g[i], 0.012)

				var fm := _ev_smooth_freq[i]
				var fi := _ev_smooth_inv_freq[i]
				_ev_motor_phase[i] = fmod(_ev_motor_phase[i] + fm * inv_sr, 1.0)
				_ev_inv_phase[i] = fmod(_ev_inv_phase[i] + fi * inv_sr, 1.0)
				_ev_sub_phase[i] = fmod(_ev_sub_phase[i] + clampf(0.5 * fm, 30.0, max_f) * inv_sr, 1.0)

				var w_sig := sin(TAU * _ev_motor_phase[i]) * _ev_smooth_gain[i] \
					+ sin(TAU * 2.0 * _ev_motor_phase[i]) * (_ev_smooth_gain[i] * 0.32) \
					+ sin(TAU * _ev_inv_phase[i]) * _ev_smooth_inv_gain[i] \
					+ sin(TAU * _ev_sub_phase[i]) * (_ev_smooth_gain[i] * 0.25)

				if (i % 2) == 0:
					eng_l += w_sig * 0.65
					eng_r += w_sig * 0.35
				else:
					eng_l += w_sig * 0.35
					eng_r += w_sig * 0.65
		else:
			_ice_smooth_rpm = lerpf(_ice_smooth_rpm, _ice_target_rpm, 0.010)
			_ice_smooth_load = lerpf(_ice_smooth_load, ice_load_target, 0.015)

			var f_fire := clampf(fire_order * _ice_smooth_rpm / 60.0, 60.0, max_f * 0.30)
			var f_half := clampf(0.5 * f_fire, 35.0, max_f)
			var f_2x := clampf(2.0 * f_fire, 60.0, max_f)
			var f_3x := clampf(3.0 * f_fire, 60.0, max_f)
			var f_15x := clampf(1.5 * f_fire, 60.0, max_f)
			var f_intake := clampf(0.25 * f_fire + 55.0, 60.0, 420.0)

			_ice_phase_half = fmod(_ice_phase_half + f_half * inv_sr, 1.0)
			_ice_phase_fire = fmod(_ice_phase_fire + f_fire * inv_sr, 1.0)
			_ice_phase_2x = fmod(_ice_phase_2x + f_2x * inv_sr, 1.0)
			_ice_phase_3x = fmod(_ice_phase_3x + f_3x * inv_sr, 1.0)
			_ice_phase_15x = fmod(_ice_phase_15x + f_15x * inv_sr, 1.0)
			_ice_phase_intake = fmod(_ice_phase_intake + f_intake * inv_sr, 1.0)

			var ld := _ice_smooth_load
			var raw_comb := 0.0
			if is_v10:
				# F1 V10: strong 5x + 2.5x + 10x + 15x + 7.5x bark, raucous rasp on throttle
				raw_comb = sin(TAU * _ice_phase_half) * lerpf(0.32, 0.24, ld) \
					+ sin(TAU * _ice_phase_fire) * lerpf(0.30, 0.42, ld) \
					+ sin(TAU * _ice_phase_15x) * lerpf(0.06, 0.22, ld) \
					+ sin(TAU * _ice_phase_2x) * lerpf(0.08, 0.28, ld) \
					+ sin(TAU * _ice_phase_3x) * lerpf(0.04, 0.18, ld)
				raw_comb = tanh(raw_comb * lerpf(1.1, 2.35, ld))
			else:
				# F1 V12: musical 6x + 3x + 12x + 18x wail + deep intake resonance under load
				var intake_res := sin(TAU * _ice_phase_intake) * (0.28 * ld)
				raw_comb = sin(TAU * _ice_phase_half) * lerpf(0.24, 0.20, ld) \
					+ sin(TAU * _ice_phase_fire) * lerpf(0.32, 0.44, ld) \
					+ sin(TAU * _ice_phase_2x) * lerpf(0.10, 0.30, ld) \
					+ sin(TAU * _ice_phase_3x) * lerpf(0.06, 0.22, ld) \
					+ intake_res
				raw_comb = tanh(raw_comb * lerpf(1.0, 1.95, ld))

			# Overrun crackle & upshift pop
			var pop_sig := 0.0
			if crackle_active:
				var wn := _next_white_noise()
				if absf(wn) > 0.9965:
					_ice_crackle_env = maxf(_ice_crackle_env, 0.75)
			if _ice_crackle_env > 0.002:
				pop_sig += _next_white_noise() * _ice_crackle_env * (0.42 if is_v10 else 0.34)
				_ice_crackle_env *= 0.9935
			if _ice_shift_pop_env > 0.002:
				pop_sig += _next_white_noise() * _ice_shift_pop_env * 0.48
				_ice_shift_pop_env *= 0.991

			var ice_mono := (raw_comb * ign_cut_mult * 0.48 + pop_sig)
			eng_l = ice_mono
			eng_r = ice_mono

		# Common stereo Tire & Aero layer
		_smooth_roll_gain = lerpf(_smooth_roll_gain, _target_roll_gain, 0.008)
		_smooth_wind_gain = lerpf(_smooth_wind_gain, _target_wind_gain, 0.008)
		_smooth_tire_l = lerpf(_smooth_tire_l, _target_tire_left, 0.012)
		_smooth_tire_r = lerpf(_smooth_tire_r, _target_tire_right, 0.012)
		_smooth_squeal_freq = lerpf(_smooth_squeal_freq, _target_squeal_freq_hz, 0.008)

		var white_l := _next_white_noise()
		var white_r := _next_white_noise()
		_lp_roll_l = clampf(lerpf(_lp_roll_l, white_l, 0.05), -2.0, 2.0)
		_lp_roll_r = clampf(lerpf(_lp_roll_r, white_r, 0.05), -2.0, 2.0)

		_lp_wind = clampf(lerpf(_lp_wind, white_l, 0.24), -2.0, 2.0)
		_hp_wind = clampf(lerpf(_hp_wind, _lp_wind, 0.04), -2.0, 2.0)
		var wind_sig := (_lp_wind - _hp_wind) * _smooth_wind_gain

		_squeal_phase = fmod(_squeal_phase + _smooth_squeal_freq * inv_sr, 1.0)
		var squeal_tone := sin(TAU * _squeal_phase) * 0.38
		_lp_squeal_l = clampf(lerpf(_lp_squeal_l, white_l, 0.28), -2.0, 2.0)
		_lp_squeal_r = clampf(lerpf(_lp_squeal_r, white_r, 0.28), -2.0, 2.0)
		var scrub_l := ((white_l - _lp_squeal_l) * 0.62 + squeal_tone) * _smooth_tire_l
		var scrub_r := ((white_r - _lp_squeal_r) * 0.62 + squeal_tone) * _smooth_tire_r

		var tire_l := _lp_roll_l * _smooth_roll_gain + wind_sig + scrub_l
		var tire_r := _lp_roll_r * _smooth_roll_gain + wind_sig + scrub_r

		var mix_l := tanh((eng_l * eng_lin + tire_l * tire_lin) * 1.40)
		var mix_r := tanh((eng_r * eng_lin + tire_r * tire_lin) * 1.40)
		if not is_finite(mix_l):
			mix_l = 0.0
		if not is_finite(mix_r):
			mix_r = 0.0
		pb.push_frame(Vector2(clampf(mix_l, -0.98, 0.98), clampf(mix_r, -0.98, 0.98)))
