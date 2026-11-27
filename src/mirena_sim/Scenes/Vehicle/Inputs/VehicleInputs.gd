class_name VehicleInputs
extends RefCounted

const VehicleConfig = preload("../Config/VehicleConfig.gd")

## vehicle inputs & driver/autonomy commands for mirena dynamics
## represents the sanitized, normalized demands sent to the dynamics model

## normalized throttle demand [0.0, 1.0]
var throttle: float = 0.0

## normalized friction braking demand [0.0, 1.0]
var brake: float = 0.0

## normalized manual steering command [-1.0, 1.0] (left: +1.0, right: -1.0)
var steer_normalized: float = 0.0

## physical front road wheel steer angle in radians (+left, -right in ros convention)
var steer_angle: float = 0.0

## emergency brake system trigger [0.0, 1.0]
var ebs: float = 0.0

## operating drive mode (4wd, rwd, fwd, torque_vectoring)
var drive_mode: int = VehicleConfig.DriveMode.FOUR_WD

## torque vectoring sub-mode (0: mode_open symmetric 50/50, 1: mode_dyc differential yaw moment)
var tv_mode: int = 0
## ecu driving mode (0: lineal_4wd, 1: ecu_tv_v3_4wd, 2: ecu_tv_4wd_cascade)
var ecu_driving_mode: int = 0

## modo de prueba tv 4wd (0: autocross, 1: skidpad, 2: efficiency, 3: acceleration, 4: tire_warmer)
var tv_4wd_mode: int = 0

## ecu torque vectoring v3 live map profile (0: open_4wd, 1: endurance_safe, 2: autocross_neutral, 3: skidpad_attack)
var tv_map_id: int = 2

## requested yaw moment delta_mz (nm, iso 8855 +z ccw left) for torquevectoring mode_dyc
var tv_yaw_moment_req: float = 0.0

## individual wheel torque commands (nm at the wheel hub)
## indices: [0: fl, 1: fr, 2: rl, 3: rr]
var wheel_motor_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

## when true, the model applies `wheel_motor_torques` and `wheel_brake_torques`
## directly without going through the default axle distribution
var override_wheel_torques: bool = false

## formatos habituales de ejes de pedales en godot
enum PedalAxisMode {
	NORMAL = 0,   # [0, 1]
	INVERTED = 1, # [1, 0]
	BIPOLAR = 2,  # [-1, 1] (1 reposo -> -1 a fondo)
}

## perfil pa volante y pedales fisicos (direct drive / logitech / thrustmaster)
var wheel_input_enabled: bool = false
var wheel_calibration: Dictionary = {
	"wheel_physical_deg": 900.0,
	"fs_steer_wheel_max_deg": 115.0,
	"deadzone": 0.0,
	"gamma": 1.0,
	"throttle_mode": PedalAxisMode.NORMAL,
	"brake_mode": PedalAxisMode.NORMAL,
	"throttle_rest": 0.0,
	"brake_rest": 0.0,
	"pedal_deadzone": 0.01,
}

## unified longitudinal & steering mapping:
## gas >= 0 => throttle = clamp(gas, 0..1), brake = clamp(manual_brake, 0..1)
## gas < 0 => throttle = 0, brake = clamp(max(-gas, manual_brake), 0..1)
func set_unified_control(gas: float, steer_rad: float, manual_brake: float = 0.0, max_steer: float = 0.35) -> void:
	var brk_manual: float = clampf(manual_brake, 0.0, 1.0)
	if gas >= 0.0:
		throttle = clampf(gas, 0.0, 1.0)
		brake = brk_manual
	else:
		throttle = 0.0
		brake = clampf(maxf(-gas, brk_manual), 0.0, 1.0)
	
	steer_angle = clampf(steer_rad, -max_steer, max_steer)
	steer_normalized = steer_angle / maxf(max_steer, 1e-5)

## sets inputs from canonical ros `carcontrol` message fields:
## msg.gas: [-1.0, 1.0] where > 0 is throttle and < 0 is braking
## msg.steer_angle: front steer angle in radians
func set_from_ros_control(gas: float, steer_rad: float, max_steer: float = 0.35) -> void:
	set_unified_control(gas, steer_rad, 0.0, max_steer)

## sets inputs from manual gamepad/keyboard inputs
func set_from_manual(throttle_in: float, brake_in: float, steer_in: float, max_steer: float = 0.35) -> void:
	throttle = clampf(throttle_in, 0.0, 1.0)
	brake = clampf(brake_in, 0.0, 1.0)
	steer_normalized = clampf(steer_in, -1.0, 1.0)
	steer_angle = steer_normalized * max_steer

## normaliza eje de pedal segun formato (normal, invertido o bipolar) con zona muerta min
static func normalize_pedal_axis(raw_val: float, mode: int, rest_val: float = 0.0, deadzone: float = 0.01) -> float:
	var norm := 0.0
	match mode:
		PedalAxisMode.INVERTED:
			var top := rest_val if rest_val > 0.5 else 1.0
			norm = (top - raw_val) / maxf(top, 1e-4)
		PedalAxisMode.BIPOLAR:
			var top_b := rest_val if rest_val > 0.2 else 1.0
			norm = (top_b - raw_val) / maxf(top_b + 1.0, 1e-4)
		_:
			var base := rest_val if (rest_val >= 0.0 and rest_val < 0.5) else 0.0
			norm = (raw_val - base) / maxf(1.0 - base, 1e-4)
	norm = clampf(norm, 0.0, 1.0)
	var dz := clampf(deadzone, 0.0, 0.2)
	if norm <= dz:
		return 0.0
	return clampf((norm - dz) / maxf(1.0 - dz, 1e-4), 0.0, 1.0)

## captura reposo actual de pedales y fija zona muerta minima de ruido (0.01)
func calibrate_wheel_pedals(raw_throttle: float, raw_brake: float, throttle_mode: int = -1, brake_mode: int = -1) -> void:
	if throttle_mode >= 0:
		wheel_calibration["throttle_mode"] = throttle_mode
	if brake_mode >= 0:
		wheel_calibration["brake_mode"] = brake_mode
	wheel_calibration["throttle_rest"] = raw_throttle
	wheel_calibration["brake_rest"] = raw_brake
	wheel_calibration["pedal_deadzone"] = 0.01
	wheel_calibration["deadzone"] = 0.0
	wheel_calibration["gamma"] = 1.0

## mapea volante fisico 1:1 lineal (sin deadzone ni gamma) + soft-lock en tope fs
func set_from_physical_wheel(raw_steer_norm: float, raw_throttle: float, raw_brake: float, max_steer_rad: float = 0.38) -> void:
	wheel_input_enabled = true
	wheel_calibration["deadzone"] = 0.0
	wheel_calibration["gamma"] = 1.0
	var phys_deg := maxf(float(wheel_calibration.get("wheel_physical_deg", 900.0)), 30.0)
	var fs_max_deg := maxf(float(wheel_calibration.get("fs_steer_wheel_max_deg", 115.0)), 10.0)
	# direccion lineal 1:1 dentro de +/- fs_max_deg con soft-lock suave en el ultimo 8 %
	var sw_deg := clampf(raw_steer_norm, -1.0, 1.0) * (0.5 * phys_deg)
	var ratio := sw_deg / fs_max_deg
	var abs_r := absf(ratio)
	var out_norm := ratio
	if abs_r > 0.92:
		var excess := abs_r - 0.92
		out_norm = signf(ratio) * clampf(0.92 + 0.08 * tanh(excess / 0.08), 0.0, 1.0)
	steer_normalized = clampf(out_norm, -1.0, 1.0)
	steer_angle = steer_normalized * max_steer_rad
	var dz := float(wheel_calibration.get("pedal_deadzone", 0.01))
	throttle = normalize_pedal_axis(raw_throttle, int(wheel_calibration.get("throttle_mode", PedalAxisMode.NORMAL)), float(wheel_calibration.get("throttle_rest", 0.0)), dz)
	brake = normalize_pedal_axis(raw_brake, int(wheel_calibration.get("brake_mode", PedalAxisMode.NORMAL)), float(wheel_calibration.get("brake_rest", 0.0)), dz)

## sets independent 4wd motor torques (e.g. from a custom torque vectoring controller)
func set_torque_vectoring_demands(motor_trqs: Array[float], brake_trqs: Array[float] = []) -> void:
	override_wheel_torques = true
	drive_mode = VehicleConfig.DriveMode.TORQUE_VECTORING
	for i in range(min(4, motor_trqs.size())):
		wheel_motor_torques[i] = motor_trqs[i]
	if brake_trqs.size() >= 4:
		for i in range(4):
			wheel_brake_torques[i] = brake_trqs[i]

## resets all inputs to resting state
func reset() -> void:
	throttle = 0.0
	brake = 0.0
	steer_normalized = 0.0
	steer_angle = 0.0
	ebs = 0.0
	tv_mode = 0
	ecu_driving_mode = 0
	tv_map_id = 2
	tv_yaw_moment_req = 0.0
	override_wheel_torques = false
	wheel_motor_torques = [0.0, 0.0, 0.0, 0.0]
	wheel_brake_torques = [0.0, 0.0, 0.0, 0.0]
