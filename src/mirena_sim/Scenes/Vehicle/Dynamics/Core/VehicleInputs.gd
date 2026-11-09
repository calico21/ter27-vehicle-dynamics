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

## Individual wheel torque commands (Nm at the wheel hub).
## Indices: [0: FL, 1: FR, 2: RL, 3: RR].
var wheel_motor_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var wheel_brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

## When true, the model applies `wheel_motor_torques` and `wheel_brake_torques`
## directly without going through the default axle distribution.
var override_wheel_torques: bool = false

## Sets inputs from canonical ROS `CarControl` message fields:
## msg.gas: [-1.0, 1.0] where > 0 is throttle and < 0 is braking.
## msg.steer_angle: front steer angle in radians.
func set_from_ros_control(gas: float, steer_rad: float, max_steer: float = 0.35) -> void:
	if gas >= 0.0:
		throttle = clampf(gas, 0.0, 1.0)
		brake = 0.0
	else:
		throttle = 0.0
		brake = clampf(-gas, 0.0, 1.0)
	
	steer_angle = clampf(steer_rad, -max_steer, max_steer)
	steer_normalized = steer_angle / maxf(max_steer, 1e-5)

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
	override_wheel_torques = false
	wheel_motor_torques = [0.0, 0.0, 0.0, 0.0]
	wheel_brake_torques = [0.0, 0.0, 0.0, 0.0]
