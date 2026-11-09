class_name BrakeModel
extends RefCounted

## Brakes Model for Mirena Dynamics
## Computes per-wheel mechanical friction brake torques and EBS distribution.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")

## Computes per-wheel friction brake torques [FL, FR, RL, RR] in Nm.
static func compute_brake_torques(inputs: VehicleInputs, config: VehicleConfig) -> Array[float]:
	var brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	if inputs.override_wheel_torques:
		for i in range(4):
			brake_torques[i] = inputs.wheel_brake_torques[i]
		return brake_torques
	
	var brake_cmd = inputs.brake
	var ebs_cmd = inputs.ebs
	if brake_cmd > 0.0 or ebs_cmd > 0.0:
		var r = config.wheel_radius
		var total_brake_force = config.brake_mu * config.mass * config.gravity
		var f_front = total_brake_force * config.brake_bias_front
		var f_rear = total_brake_force * (1.0 - config.brake_bias_front)
		
		var ebs_trq_each = (total_brake_force * 0.25 * r) * ebs_cmd
		
		brake_torques[VehicleConfig.WheelIndex.FL] = (f_front * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.FR] = (f_front * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.RL] = (f_rear * 0.5 * brake_cmd) * r + ebs_trq_each
		brake_torques[VehicleConfig.WheelIndex.RR] = (f_rear * 0.5 * brake_cmd) * r + ebs_trq_each
	
	return brake_torques
