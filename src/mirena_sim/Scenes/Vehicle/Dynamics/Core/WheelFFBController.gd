class_name WheelFFBController
extends RefCounted

## Direct Drive / SimRacing Steering Wheel Input & Physical Force Feedback Controller.
##
## Features:
## - Auto-detection of Direct Drive / SimRacing wheel bases via Input.get_joy_name(joy_id)
##   with manual override (`AUTO`, `GAMEPAD`, `WHEEL`).
## - Pure linear 1:1 steering mapping in `WHEEL` mode (+/-110 deg physical steering lock,
##   zero speed sensitivity attenuation, gamma = 1.0, zero artificial filtering).
## - Physical steering column torque calculation from front tire self-aligning moments (M_z),
##   lateral forces (F_y * mechanical_trail), and longitudinal split (dF_x * scrub_radius),
##   divided by steering rack ratio i_rack, plus optional soft column damper & safe saturation.

enum InputDeviceMode {
	AUTO = 0,
	GAMEPAD = 1,
	WHEEL = 2
}

const WHEEL_MAX_DEG: float = 110.0
const WHEEL_MAX_RAD: float = WHEEL_MAX_DEG * PI / 180.0

const WHEEL_KEYWORDS: Array[String] = [
	"fanatec",
	"simucube",
	"moza",
	"simagic",
	"logitech",
	"g29",
	"g920",
	"g923",
	"thrustmaster",
	"t300",
	"vrs",
	"asetek",
	"cammus",
	"wheel",
	"dd"
]

var input_device_override: int = InputDeviceMode.AUTO
var active_device_mode: int = InputDeviceMode.GAMEPAD

# Steering shaping parameters (locked to linear 1:1 when active_device_mode == WHEEL)
var speed_sensitivity: float = 0.25
var steering_gamma: float = 2.2
var steering_deadzone: float = 0.06
var steering_filter: float = 0.12

# Physical steering column & FFB tuning
var rack_ratio: float = 4.8
var mechanical_trail: float = 0.013
var scrub_radius: float = 0.018
var ffb_max_torque_nm: float = 15.0
var ffb_gain: float = 1.0
var column_damping_nm_s: float = 0.05

# Live steering & FFB telemetry
var wheel_angle_deg: float = 0.0
var wheel_angle_rad: float = 0.0
var ffb_torque_nm: float = 0.0
var ffb_normalized: float = 0.0

var _prev_wheel_angle_rad: float = 0.0
var _has_prev_wheel_angle: bool = false

func reset() -> void:
	wheel_angle_deg = 0.0
	wheel_angle_rad = 0.0
	ffb_torque_nm = 0.0
	ffb_normalized = 0.0
	_prev_wheel_angle_rad = 0.0
	_has_prev_wheel_angle = false

static func is_wheel_name(joy_name: String) -> bool:
	var lower: String = joy_name.to_lower()
	if lower.is_empty():
		return false
	for kw in WHEEL_KEYWORDS:
		if lower.contains(kw):
			return true
	return false

## Resolves effective input device mode from an explicit device name and `input_device_override`.
func detect_input_device_from_name(joy_name: String) -> int:
	if input_device_override == InputDeviceMode.GAMEPAD:
		active_device_mode = InputDeviceMode.GAMEPAD
	elif input_device_override == InputDeviceMode.WHEEL:
		active_device_mode = InputDeviceMode.WHEEL
	else:
		active_device_mode = InputDeviceMode.WHEEL if is_wheel_name(joy_name) else InputDeviceMode.GAMEPAD

	if active_device_mode == InputDeviceMode.WHEEL:
		speed_sensitivity = 0.0
		steering_gamma = 1.0
		steering_deadzone = 0.0
		steering_filter = 0.0
	else:
		speed_sensitivity = 0.25
		steering_gamma = 2.2
		steering_deadzone = 0.06
		steering_filter = 0.12

	return active_device_mode

## Inspects `Input.get_joy_name(joy_id)` (or override) and configures active device mode.
func detect_input_device(joy_id: int = 0) -> int:
	var joy_name: String = Input.get_joy_name(joy_id)
	return detect_input_device_from_name(joy_name)

## Synchronizes a GamepadController instance for linear wheel mode when WHEEL is active.
func configure_gamepad_controller(gp: GamepadController) -> void:
	if gp == null:
		return
	if active_device_mode == InputDeviceMode.WHEEL:
		gp.speed_sensitivity = 0.0
		gp.steering_gamma = 1.0
		gp.steering_deadzone = 0.0
		gp.steering_filter = 0.0

## Converts normalized steering axis [-1, 1] <-> physical wheel angle [-110 deg, +110 deg].
static func normalized_to_wheel_deg(norm_axis: float) -> float:
	return clampf(norm_axis, -1.0, 1.0) * WHEEL_MAX_DEG

static func wheel_deg_to_normalized(angle_deg: float) -> float:
	return clampf(angle_deg / WHEEL_MAX_DEG, -1.0, 1.0)

static func normalized_to_wheel_rad(norm_axis: float) -> float:
	return clampf(norm_axis, -1.0, 1.0) * WHEEL_MAX_RAD

static func wheel_rad_to_normalized(angle_rad: float) -> float:
	return clampf(angle_rad / WHEEL_MAX_RAD, -1.0, 1.0)

## Pure linear 1:1 steering mapping for +/-110 deg wheel bases (no gamma, no speed cut, no LPF).
func map_wheel_axis(raw_axis: float) -> float:
	var clamped: float = clampf(raw_axis, -1.0, 1.0)
	wheel_angle_deg = clamped * WHEEL_MAX_DEG
	wheel_angle_rad = clamped * WHEEL_MAX_RAD
	return clamped

## Processes steering axis according to active mode (`WHEEL` = linear +/-110 deg, `GAMEPAD` = optional GamepadController).
func step_steering(raw_axis: float, vx_mps: float = 0.0, dt: float = 0.0, fallback_gamepad: GamepadController = null) -> float:
	if active_device_mode == InputDeviceMode.WHEEL or input_device_override == InputDeviceMode.WHEEL:
		speed_sensitivity = 0.0
		steering_gamma = 1.0
		return map_wheel_axis(raw_axis)

	if fallback_gamepad != null:
		return fallback_gamepad.step_steering(raw_axis, vx_mps, dt)
	return clampf(raw_axis, -1.0, 1.0)

static func _read_wheel_field(w, primary_key: String, alt_key: String) -> float:
	if w == null:
		return 0.0
	if w is Dictionary:
		if w.has(primary_key):
			return float(w[primary_key])
		if w.has(alt_key):
			return float(w[alt_key])
		return 0.0
	if w is Object:
		var val = w.get(primary_key)
		if val != null:
			return float(val)
		var alt_val = w.get(alt_key)
		if alt_val != null:
			return float(alt_val)
	return 0.0

## Computes physical steering column torque (Nm):
##   tau_steer = ((M_z_FL + M_z_FR) + (F_y_FL + F_y_FR) * t_mech + (F_x_FL - F_x_FR) * r_scrub) / i_rack
func compute_steering_torque_nm(
	wheels: Array,
	p_rack_ratio: float = 4.8,
	p_mechanical_trail: float = 0.013,
	p_scrub_radius: float = 0.018
) -> float:
	if wheels.size() < 2:
		ffb_torque_nm = 0.0
		ffb_normalized = 0.0
		return 0.0

	var w_fl = wheels[0]
	var w_fr = wheels[1]
	var mz_fl: float = _read_wheel_field(w_fl, "M_z", "mz")
	var mz_fr: float = _read_wheel_field(w_fr, "M_z", "mz")
	var fy_fl: float = _read_wheel_field(w_fl, "F_y", "fy")
	var fy_fr: float = _read_wheel_field(w_fr, "F_y", "fy")
	var fx_fl: float = _read_wheel_field(w_fl, "F_x", "fx")
	var fx_fr: float = _read_wheel_field(w_fr, "F_x", "fx")

	var inv_rack: float = 1.0 / maxf(p_rack_ratio, 1e-3)
	var kingpin_moment_nm: float = (
		(mz_fl + mz_fr)
		+ (fy_fl + fy_fr) * p_mechanical_trail
		+ (fx_fl - fx_fr) * p_scrub_radius
	)
	ffb_torque_nm = inv_rack * kingpin_moment_nm
	ffb_normalized = clampf(ffb_torque_nm / maxf(ffb_max_torque_nm, 1.0), -1.0, 1.0)
	return ffb_torque_nm

## Computes normalized signed FFB torque command in [-1.0, 1.0] with optional soft column damper
## and dispatches safe haptic feedback if the target joypad is connected.
func apply_haptic_feedback(joy_id: int, wheels: Array, speed_mps: float, dt: float = 0.0) -> void:
	compute_steering_torque_nm(wheels, rack_ratio, mechanical_trail, scrub_radius)

	var damper_nm: float = 0.0
	if dt > 1e-5 and column_damping_nm_s > 0.0:
		var steer_vel: float = ((wheel_angle_rad - _prev_wheel_angle_rad) / dt) if _has_prev_wheel_angle else 0.0
		damper_nm = -column_damping_nm_s * steer_vel
	_prev_wheel_angle_rad = wheel_angle_rad
	_has_prev_wheel_angle = true

	var net_torque_nm: float = (ffb_torque_nm + damper_nm) * maxf(ffb_gain, 0.0)
	ffb_normalized = clampf(net_torque_nm / maxf(ffb_max_torque_nm, 1.0), -1.0, 1.0)

	if joy_id >= 0 and Input.get_connected_joypads().has(joy_id):
		var mag: float = absf(ffb_normalized)
		if mag > 0.01 and speed_mps >= 0.0:
			var low_freq: float = clampf(mag * 0.85, 0.0, 1.0)
			var high_freq: float = clampf(mag * 0.45, 0.0, 1.0)
			Input.start_joy_vibration(joy_id, high_freq, low_freq, 0.05)
		else:
			Input.stop_joy_vibration(joy_id)
