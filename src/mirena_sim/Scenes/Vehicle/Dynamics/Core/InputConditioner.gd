class_name InputConditioner
extends RefCounted

## Input conditioning layer for MirenaSim (Gamepad / Wheel Direct / Keyboard).
## Operates purely on driver commands before VehicleInputs; zero physics modifications.

enum InputProfile { AUTO = 0, GAMEPAD = 1, WHEEL_DIRECT = 2, KEYBOARD = 3 }

const PROFILE_NAMES: Array[String] = ["Auto", "Gamepad", "Wheel (Direct)", "Keyboard"]

# Gamepad steering & brake parameters (calibrated from TeR27 Milliken YMD)
const GP_DEADZONE: float = 0.06
const GP_THROTTLE_DEADZONE: float = 0.04
const GP_BRAKE_DEADZONE: float = 0.04
const GP_CURVE_GAMMA: float = 1.6
const GP_CURVE_MIX: float = 0.40            # 0.60*|x| + 0.40*|x|^1.6
const GP_RATE_IN_RAD_S: float = 4.2        # >= 3.8 rad/s -> < 0.10 s full lock at low speed
const GP_RATE_RETURN_RAD_S: float = 6.2    # >= 5.5 rad/s fast countersteer / return
const GP_SPEED_FULL_MPS: float = 8.333     # 30 km/h: 100% authority
const GP_SPEED_CAP_MPS: float = 11.111     # 40 km/h: begin speed-sensitive cap
const GP_AY_CAP_MPS2: float = 17.50        # ~1.78 g lateral authority envelope
const GP_STEER_MARGIN_RAD: float = 0.14835 # +8.5 deg countersteer / mid-corner margin
const GP_STEER_MIN_RAD: float = 0.131      # >= 7.5 deg minimum authority at high speed

# Keyboard steering parameters
const KB_RATE_IN_RAD_S: float = 2.8
const KB_RATE_RETURN_RAD_S: float = 5.5

var profile: int = InputProfile.AUTO
var active_resolved_profile: int = InputProfile.KEYBOARD
var steer_angle_state: float = 0.0
var raw_steer_norm: float = 0.0

var last_steer_raw: float = 0.0
var last_steer_out: float = 0.0
var last_thr_raw: float = 0.0
var last_thr_out: float = 0.0
var last_brk_raw: float = 0.0
var last_brk_out: float = 0.0

# Runtime-tunable input conditioning parameters (bound by Controls UI)
var steer_deadzone: float = GP_DEADZONE
var steer_gamma: float = GP_CURVE_GAMMA
var steer_speed_sensitivity: float = 1.0
var throttle_gamma: float = 1.0
var brake_gamma: float = 1.0
var rumble_intensity: float = 0.7


func reset() -> void:
	steer_angle_state = 0.0
	raw_steer_norm = 0.0
	last_steer_raw = 0.0
	last_steer_out = 0.0
	last_thr_raw = 0.0
	last_thr_out = 0.0
	last_brk_raw = 0.0
	last_brk_out = 0.0


func get_live_input_monitor() -> Dictionary:
	return {
		"steer_raw": last_steer_raw,
		"steer_out": last_steer_out,
		"thr_raw": last_thr_raw,
		"thr_out": last_thr_out,
		"brk_raw": last_brk_raw,
		"brk_out": last_brk_out,
	}


func apply_input_preset(preset_id: String) -> Dictionary:
	var key := preset_id.strip_edges().to_upper()
	match key:
		"KEYBOARD_SHARP":
			profile = InputProfile.KEYBOARD
			active_resolved_profile = InputProfile.KEYBOARD
			steer_deadzone = 0.0
			steer_gamma = 1.0
			steer_speed_sensitivity = 0.85
			throttle_gamma = 1.0
			brake_gamma = 1.0
		"GAMEPAD_PRECISION":
			profile = InputProfile.GAMEPAD
			active_resolved_profile = InputProfile.GAMEPAD
			steer_deadzone = GP_DEADZONE
			steer_gamma = GP_CURVE_GAMMA
			steer_speed_sensitivity = 1.0
			throttle_gamma = 1.0
			brake_gamma = 1.0
		"WHEEL_DIRECT":
			profile = InputProfile.WHEEL_DIRECT
			active_resolved_profile = InputProfile.WHEEL_DIRECT
			steer_deadzone = 0.0
			steer_gamma = 1.0
			steer_speed_sensitivity = 0.0
			throttle_gamma = 1.0
			brake_gamma = 1.0
	return get_tuning_params()


func set_profile(p: int) -> void:
	profile = clampi(p, 0, PROFILE_NAMES.size() - 1)


func get_profile_name() -> String:
	return PROFILE_NAMES[clampi(profile, 0, PROFILE_NAMES.size() - 1)]


func get_tuning_params() -> Dictionary:
	return {
		"steer_deadzone": steer_deadzone,
		"steer_gamma": steer_gamma,
		"steer_speed_sensitivity": steer_speed_sensitivity,
		"throttle_gamma": throttle_gamma,
		"brake_gamma": brake_gamma,
		"rumble_intensity": rumble_intensity,
	}


func set_tuning_params(params: Dictionary) -> void:
	if params.has("steer_deadzone"):
		steer_deadzone = clampf(float(params["steer_deadzone"]), 0.0, 0.30)
	if params.has("steer_gamma"):
		steer_gamma = clampf(float(params["steer_gamma"]), 1.0, 3.5)
	if params.has("steer_speed_sensitivity"):
		steer_speed_sensitivity = clampf(float(params["steer_speed_sensitivity"]), 0.0, 1.5)
	if params.has("throttle_gamma"):
		throttle_gamma = clampf(float(params["throttle_gamma"]), 0.5, 2.5)
	if params.has("brake_gamma"):
		brake_gamma = clampf(float(params["brake_gamma"]), 0.5, 2.5)
	if params.has("rumble_intensity"):
		rumble_intensity = clampf(float(params["rumble_intensity"]), 0.0, 1.0)


static func apply_deadzone_renorm(x: float, dz: float) -> float:
	var ax := absf(x)
	if ax <= dz:
		return 0.0
	var norm := clampf((ax - dz) / maxf(1.0 - dz, 1e-4), 0.0, 1.0)
	return signf(x) * norm


static func shape_gamepad_steer_norm(raw_x: float) -> float:
	var u := apply_deadzone_renorm(clampf(raw_x, -1.0, 1.0), GP_DEADZONE)
	var au := absf(u)
	var shaped := (1.0 - GP_CURVE_MIX) * au + GP_CURVE_MIX * pow(au, GP_CURVE_GAMMA)
	return signf(u) * clampf(shaped, 0.0, 1.0)


static func shape_gamepad_throttle(raw_throttle: float) -> float:
	var u := apply_deadzone_renorm(clampf(raw_throttle, 0.0, 1.0), GP_THROTTLE_DEADZONE)
	return clampf(1.15 * u - 0.15 * u * u, 0.0, 1.0)


static func shape_gamepad_brake(raw_brake: float) -> float:
	return apply_deadzone_renorm(clampf(raw_brake, 0.0, 1.0), GP_BRAKE_DEADZONE)


static func shape_gamepad_brake_80_20(raw_brake: float) -> float:
	return shape_gamepad_brake(raw_brake)


static func compute_gamepad_steer_cap_rad(speed_mps: float, max_steer_rad: float, wheelbase_m: float = 1.54) -> float:
	var v := absf(speed_mps)
	if v <= GP_SPEED_FULL_MPS:
		return max_steer_rad
	var cap_high := clampf(
		atan((wheelbase_m * GP_AY_CAP_MPS2) / maxf(v * v, 1.0)) + GP_STEER_MARGIN_RAD,
		minf(GP_STEER_MIN_RAD, max_steer_rad),
		max_steer_rad
	)
	if v >= GP_SPEED_CAP_MPS:
		return cap_high
	var alpha := (v - GP_SPEED_FULL_MPS) / maxf(GP_SPEED_CAP_MPS - GP_SPEED_FULL_MPS, 1e-3)
	return lerpf(max_steer_rad, cap_high, clampf(alpha, 0.0, 1.0))


func resolve_profile(has_joypad: bool, is_wheel_device: bool, raw_joy_activity: bool, raw_kb_activity: bool) -> int:
	if profile != InputProfile.AUTO:
		active_resolved_profile = profile
		return active_resolved_profile
	if is_wheel_device:
		active_resolved_profile = InputProfile.WHEEL_DIRECT
	elif has_joypad and (raw_joy_activity or not raw_kb_activity):
		active_resolved_profile = InputProfile.GAMEPAD
	else:
		active_resolved_profile = InputProfile.KEYBOARD
	return active_resolved_profile


func step(
	raw_steer: float,
	raw_throttle: float,
	raw_brake: float,
	speed_mps: float,
	wz_rad_s: float,
	beta_rad: float,
	dt: float,
	max_steer_rad: float,
	wheelbase_m: float = 1.54,
	resolved_override: int = -1
) -> Dictionary:
	var eff_profile := resolved_override if resolved_override >= 0 else active_resolved_profile
	var in_s := clampf(raw_steer, -1.0, 1.0)
	var in_t := clampf(raw_throttle, 0.0, 1.0)
	var in_b := clampf(raw_brake, 0.0, 1.0)
	raw_steer_norm = in_s
	last_steer_raw = in_s
	last_thr_raw = in_t
	last_brk_raw = in_b

	if eff_profile == InputProfile.WHEEL_DIRECT:
		# Strictly 1:1 linear passthrough, zero deadzone, zero speed scaling
		steer_angle_state = in_s * max_steer_rad
		last_steer_out = in_s
		last_thr_out = in_t
		last_brk_out = in_b
		return {
			"steer_angle": steer_angle_state,
			"steer_norm": in_s,
			"raw_steer_norm": raw_steer_norm,
			"gas": in_t,
			"brake": in_b,
			"profile": eff_profile,
		}

	var base_cap_rad := compute_gamepad_steer_cap_rad(speed_mps, max_steer_rad, wheelbase_m)
	var s_sens := clampf(steer_speed_sensitivity, 0.0, 1.5)
	var cap_rad := clampf(lerpf(max_steer_rad, base_cap_rad, s_sens), minf(GP_STEER_MIN_RAD, max_steer_rad), max_steer_rad)
	var target_rad := 0.0
	var rate_in := GP_RATE_IN_RAD_S
	var rate_ret := GP_RATE_RETURN_RAD_S
	var out_gas := in_t
	var out_brake := in_b

	if eff_profile == InputProfile.GAMEPAD:
		var u_s := apply_deadzone_renorm(clampf(in_s, -1.0, 1.0), clampf(steer_deadzone, 0.0, 0.30))
		var au_s := absf(u_s)
		var g_s := clampf(steer_gamma, 1.0, 3.5)
		var shaped_s := (1.0 - GP_CURVE_MIX) * au_s + GP_CURVE_MIX * pow(au_s, g_s)
		target_rad = signf(u_s) * clampf(shaped_s, 0.0, 1.0) * cap_rad
		var base_gas := shape_gamepad_throttle(in_t)
		var tg := clampf(throttle_gamma, 0.5, 2.5)
		out_gas = pow(base_gas, tg) if absf(tg - 1.0) > 1e-4 else base_gas
		var base_brk := shape_gamepad_brake(in_b)
		var bg := clampf(brake_gamma, 0.5, 2.5)
		out_brake = pow(base_brk, bg) if absf(bg - 1.0) > 1e-4 else base_brk
	else:
		# KEYBOARD
		target_rad = in_s * cap_rad
		rate_in = KB_RATE_IN_RAD_S
		rate_ret = KB_RATE_RETURN_RAD_S

	# Immediate fast rate for return-to-center or countersteering against wz / beta
	if out_gas > 0.70 and out_brake < 0.20:
		out_brake = 0.0
	var is_returning := (absf(target_rad) < absf(steer_angle_state)) or (target_rad * steer_angle_state < 0.0)
	var is_countersteer := (in_s * wz_rad_s < -0.05) or (in_s * beta_rad > 0.02 and absf(wz_rad_s) > 0.15)
	var active_rate := rate_ret if (is_returning or is_countersteer) else rate_in

	var max_step := active_rate * maxf(dt, 0.0)
	steer_angle_state = move_toward(steer_angle_state, target_rad, max_step)
	steer_angle_state = clampf(steer_angle_state, -max_steer_rad, max_steer_rad)

	var out_norm := steer_angle_state / maxf(max_steer_rad, 1e-4)
	last_steer_out = out_norm
	last_thr_out = out_gas
	last_brk_out = out_brake
	return {
		"steer_angle": steer_angle_state,
		"steer_norm": out_norm,
		"raw_steer_norm": raw_steer_norm,
		"gas": out_gas,
		"brake": out_brake,
		"profile": eff_profile,
	}
