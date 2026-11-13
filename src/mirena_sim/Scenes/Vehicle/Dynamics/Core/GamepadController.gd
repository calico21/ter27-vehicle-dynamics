class_name GamepadController
extends RefCounted

## Assetto Corsa-style analog gamepad input processor for Formula Student (PS4 DualShock 4).
## Handles steering deadzone, gamma, speed sensitivity, rate limiting, exponential filtering,
## non-linear throttle/brake pedal curves, trigger axis normalization, and dual-motor haptics.

const DEFAULT_CONFIG_PATH := "res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json"
const V_REF_MPS: float = 20.0
const RETURN_SPEED_MULTIPLIER: float = 1.35
const TRIGGER_DEADZONE: float = 0.04

var steering_deadzone: float = 0.06
var steering_gamma: float = 2.2
var steering_speed: float = 1.8
var steering_filter: float = 0.12
var speed_sensitivity: float = 0.65
var throttle_gamma: float = 1.6
var brake_gamma: float = 1.4
var rumble_effects: float = 0.7

var _steer_rate_limited: float = 0.0
var _steer_filtered: float = 0.0
var _trigger_bipolar_l: bool = false
var _trigger_bipolar_r: bool = false
var _prev_dz_dot: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _has_prev_dz_dot: bool = false

func _init(config_path: String = "") -> void:
	if not config_path.is_empty():
		load_from_json(config_path)

func load_from_json(file_path: String = DEFAULT_CONFIG_PATH) -> bool:
	if not FileAccess.file_exists(file_path):
		return false
	var f := FileAccess.open(file_path, FileAccess.READ)
	if f == null:
		return false
	var json := JSON.new()
	if json.parse(f.get_as_text()) != OK:
		return false
	if not (json.data is Dictionary):
		return false
	var d: Dictionary = json.data
	steering_deadzone = clampf(float(d.get("steering_deadzone", steering_deadzone)), 0.0, 0.30)
	steering_gamma = clampf(float(d.get("steering_gamma", steering_gamma)), 1.0, 4.0)
	steering_speed = clampf(float(d.get("steering_speed", steering_speed)), 0.5, 5.0)
	steering_filter = clampf(float(d.get("steering_filter", steering_filter)), 0.0, 0.5)
	speed_sensitivity = clampf(float(d.get("speed_sensitivity", speed_sensitivity)), 0.0, 1.0)
	throttle_gamma = clampf(float(d.get("throttle_gamma", throttle_gamma)), 1.0, 3.0)
	brake_gamma = clampf(float(d.get("brake_gamma", brake_gamma)), 1.0, 3.0)
	rumble_effects = clampf(float(d.get("rumble_effects", rumble_effects)), 0.0, 1.0)
	return true

func reset() -> void:
	_steer_rate_limited = 0.0
	_steer_filtered = 0.0
	_prev_dz_dot = [0.0, 0.0, 0.0, 0.0]
	_has_prev_dz_dot = false

## Normalizes a gamepad trigger axis (L2 / R2) to [0.0, 1.0].
## Supports both unipolar [0.0, 1.0] drivers and bipolar [-1.0, 1.0] drivers (rest = -1.0).
func normalize_trigger(raw_axis: float, is_left_trigger: bool, deadzone: float = TRIGGER_DEADZONE) -> float:
	if raw_axis < -0.05:
		if is_left_trigger:
			_trigger_bipolar_l = true
		else:
			_trigger_bipolar_r = true

	var is_bipolar: bool = _trigger_bipolar_l if is_left_trigger else _trigger_bipolar_r
	var u: float = clampf((raw_axis + 1.0) * 0.5, 0.0, 1.0) if is_bipolar else clampf(raw_axis, 0.0, 1.0)
	if deadzone > 0.0 and deadzone < 1.0:
		if u <= deadzone:
			return 0.0
		return clampf((u - deadzone) / (1.0 - deadzone), 0.0, 1.0)
	return u

## Applies symmetric deadzone and rescales [deadzone, 1.0] -> [0.0, 1.0] without jump.
static func apply_deadzone(raw: float, deadzone: float) -> float:
	var dz: float = clampf(deadzone, 0.0, 0.99)
	var mag: float = absf(raw)
	if mag <= dz:
		return 0.0
	return signf(raw) * clampf((mag - dz) / (1.0 - dz), 0.0, 1.0)

## Applies signed power curve: sign(u) * |u|^gamma.
static func apply_gamma(u: float, gamma: float) -> float:
	var clamped: float = clampf(u, -1.0, 1.0)
	if is_zero_approx(clamped):
		return 0.0
	return signf(clamped) * pow(absf(clamped), maxf(gamma, 0.01))

## Speed-sensitive steering attenuation factor: 1 / (1 + k * (|vx| / 20.0)^2).
func compute_speed_factor(vx_mps: float) -> float:
	var ratio: float = absf(vx_mps) / V_REF_MPS
	return 1.0 / (1.0 + maxf(speed_sensitivity, 0.0) * ratio * ratio)

## Full Assetto Corsa steering chain:
## deadzone -> gamma -> speed_sensitivity -> steering_speed (35% faster return) -> steering_filter.
func step_steering(raw_steer: float, vx_mps: float, dt: float) -> float:
	var u_dz: float = apply_deadzone(raw_steer, steering_deadzone)
	var u_gamma: float = apply_gamma(u_dz, steering_gamma)
	var target: float = clampf(u_gamma * compute_speed_factor(vx_mps), -1.0, 1.0)

	if dt <= 0.0:
		_steer_rate_limited = target
		_steer_filtered = target
		return _steer_filtered

	var returning: bool = absf(target) < absf(_steer_rate_limited) or (target * _steer_rate_limited < 0.0)
	var rate: float = steering_speed * (RETURN_SPEED_MULTIPLIER if returning else 1.0)
	_steer_rate_limited = move_toward(_steer_rate_limited, target, rate * dt)

	if steering_filter <= 0.0:
		_steer_filtered = _steer_rate_limited
	else:
		var alpha: float = 1.0 - exp(-dt / steering_filter)
		_steer_filtered = lerpf(_steer_filtered, _steer_rate_limited, alpha)

	_steer_filtered = clampf(_steer_filtered, -1.0, 1.0)
	return _steer_filtered

## Non-linear throttle shaping: u^throttle_gamma in [0.0, 1.0].
func shape_throttle(raw_throttle: float) -> float:
	var u: float = clampf(raw_throttle, 0.0, 1.0)
	return pow(u, maxf(throttle_gamma, 0.01))

## Non-linear brake shaping: u^brake_gamma in [0.0, 1.0].
func shape_brake(raw_brake: float) -> float:
	var u: float = clampf(raw_brake, 0.0, 1.0)
	return pow(u, maxf(brake_gamma, 0.01))

## Computes dual-motor rumble intensities Vector2(weak_high_freq, strong_low_freq) in [0.0, 1.0].
## - Weak motor: longitudinal slip |kappa| (lockup/wheelspin) + front understeer |alpha| saturation.
## - Strong motor: vertical impacts / curbs (|z_ddot| from suspension travel rate derivative).
func compute_rumble(wheels: Array, suspension_corners: Array, speed_mps: float, dt: float) -> Vector2:
	if rumble_effects <= 0.0:
		_has_prev_dz_dot = false
		return Vector2.ZERO

	var max_kappa: float = 0.0
	var max_front_alpha: float = 0.0
	var max_z_ddot: float = 0.0

	if speed_mps > 1.0:
		for i in range(mini(wheels.size(), 4)):
			var w = wheels[i]
			if w.is_grounded:
				max_kappa = maxf(max_kappa, absf(w.kappa))
				if i < 2:
					max_front_alpha = maxf(max_front_alpha, absf(w.alpha))

	if dt > 1e-4 and suspension_corners.size() >= 4:
		for i in range(4):
			var sc = suspension_corners[i]
			var dz_dot: float = float(sc.delta_z_dot) if sc.is_grounded else 0.0
			if _has_prev_dz_dot and sc.is_grounded:
				var z_ddot: float = absf(dz_dot - _prev_dz_dot[i]) / dt
				max_z_ddot = maxf(max_z_ddot, z_ddot)
			_prev_dz_dot[i] = dz_dot
		_has_prev_dz_dot = true
	else:
		_has_prev_dz_dot = false

	var slip_rumble: float = clampf((max_kappa - 0.10) / 0.35, 0.0, 1.0)
	var understeer_rumble: float = clampf((max_front_alpha - 0.14) / 0.22, 0.0, 1.0) * 0.75
	var weak: float = clampf(maxf(slip_rumble, understeer_rumble) * rumble_effects, 0.0, 1.0)

	# Vertical curb/bump impacts above ~6 m/s^2 suspension vertical acceleration
	var impact_rumble: float = clampf((max_z_ddot - 6.0) / 25.0, 0.0, 1.0)
	var strong: float = clampf(impact_rumble * rumble_effects, 0.0, 1.0)

	return Vector2(weak, strong)
