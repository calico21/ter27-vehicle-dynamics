class_name VehicleInputs
extends RefCounted

const VehicleConfig = preload("VehicleConfig.gd")

## Vehicle Inputs & Driver/Autonomy Commands for Mirena Dynamics
## Represents the sanitized, normalized demands sent to the dynamics model.

## Normalized throttle demand [0.0, 1.0].
var throttle: float = 0.0

## Normalized friction braking demand [0.0, 1.0].
var brake: float = 0.0

## Normalized manual steering command [-1.0, 1.0] (Left: +1.0, Right: -1.0).
var steer_normalized: float = 0.0

## Physical front road wheel steer angle in radians (+left, -right in ROS convention).
var steer_angle: float = 0.0

## Emergency Brake System trigger [0.0, 1.0].
var ebs: float = 0.0

## Operating drive mode (4WD, RWD, FWD, TORQUE_VECTORING).
var drive_mode: int = VehicleConfig.DriveMode.FOUR_WD

## Torque Vectoring sub-mode (0: MODE_OPEN symmetric 50/50, 1: MODE_DYC differential yaw moment).
var tv_mode: int = 0
## ECU Driving Mode (0: LINEAL_4WD, 1: ECU_TV_V3_4WD).
var ecu_driving_mode: int = 0

## Requested yaw moment Delta_Mz (Nm, ISO 8855 +Z CCW left) for TorqueVectoring MODE_DYC.
var tv_yaw_moment_req: float = 0.0

## Individual wheel torque commands (Nm at the wheel hub).
## Indices: [0: FL, 1: FR, 2: RL, 3: RR].
var wheel_motor_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

## When true, the model applies `wheel_motor_torques` and `wheel_brake_torques`
## directly without going through the default axle distribution.
var override_wheel_torques: bool = false

## Unified longitudinal & steering mapping:
## - gas >= 0 => throttle = clamp(gas, 0..1), brake = clamp(manual_brake, 0..1)
## - gas < 0  => throttle = 0, brake = clamp(max(-gas, manual_brake), 0..1)
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

## Sets inputs from canonical ROS `CarControl` message fields:
## msg.gas: [-1.0, 1.0] where > 0 is throttle and < 0 is braking.
## msg.steer_angle: front steer angle in radians.
func set_from_ros_control(gas: float, steer_rad: float, max_steer: float = 0.35) -> void:
	set_unified_control(gas, steer_rad, 0.0, max_steer)

## Sets inputs from manual gamepad/keyboard inputs.
func set_from_manual(throttle_in: float, brake_in: float, steer_in: float, max_steer: float = 0.35) -> void:
	throttle = clampf(throttle_in, 0.0, 1.0)
	brake = clampf(brake_in, 0.0, 1.0)
	steer_normalized = clampf(steer_in, -1.0, 1.0)
	steer_angle = steer_normalized * max_steer

## Sets independent 4WD motor torques (e.g. from a custom Torque Vectoring controller).
func set_torque_vectoring_demands(motor_trqs: Array[float], brake_trqs: Array[float] = []) -> void:
	override_wheel_torques = true
	drive_mode = VehicleConfig.DriveMode.TORQUE_VECTORING
	for i in range(min(4, motor_trqs.size())):
		wheel_motor_torques[i] = motor_trqs[i]
	if brake_trqs.size() >= 4:
		for i in range(4):
			wheel_brake_torques[i] = brake_trqs[i]

## Resets all inputs to resting state.
func reset() -> void:
	throttle = 0.0
	brake = 0.0
	steer_normalized = 0.0
	steer_angle = 0.0
	ebs = 0.0
	tv_mode = 0
	ecu_driving_mode = 0
	tv_yaw_moment_req = 0.0
	override_wheel_torques = false
	wheel_motor_torques = [0.0, 0.0, 0.0, 0.0]
	wheel_brake_torques = [0.0, 0.0, 0.0, 0.0]
