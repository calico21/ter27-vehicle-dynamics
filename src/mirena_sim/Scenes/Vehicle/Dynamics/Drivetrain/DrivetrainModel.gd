class_name DrivetrainModel
extends RefCounted

## Drivetrain & Powertrain Model for Mirena Dynamics
## Computes per-wheel motor drive torques [FL, FR, RL, RR] with 80 kW power limiter.

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")
const BrakeModel = preload("../Brakes/BrakeModel.gd")

## Computes drive torques for all 4 wheels in Nm.
static func compute_drive_torques(inputs: VehicleInputs, speed: float, config: VehicleConfig) -> Array[float]:
	var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	if inputs.override_wheel_torques:
		for i in range(4):
			drive_torques[i] = inputs.wheel_motor_torques[i]
		return drive_torques
	
	var throttle = inputs.throttle
	if throttle > 0.0:
		var r = config.wheel_radius
		var G = config.gear_ratio
		var eff = config.drivetrain_efficiency
		
		# Peak wheel drive torque per motor: T_wheel = T_motor * G * eff
		var max_wheel_trq = config.motor_peak_torque * G * eff
		
		# Electric power envelope: P_max * eff / speed
		var speed_clamped = maxf(absf(speed), 1.0)
		var max_power_force = (config.power_limit_watts * eff) / speed_clamped
		var max_power_trq_total = max_power_force * r
		
		var total_max_trq = minf(4.0 * max_wheel_trq, max_power_trq_total)
		var requested_total_trq = throttle * total_max_trq
		
		var s_front = config.front_drive_torque_ratio
		var s_rear = 1.0 - s_front
		
		match inputs.drive_mode:
			VehicleConfig.DriveMode.FOUR_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * s_rear * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * s_rear * 0.5
			VehicleConfig.DriveMode.REAR_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = 0.0
				drive_torques[VehicleConfig.WheelIndex.FR] = 0.0
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * 0.5
			VehicleConfig.DriveMode.FRONT_WD:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = 0.0
				drive_torques[VehicleConfig.WheelIndex.RR] = 0.0
			_:
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * 0.25
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * 0.25
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * 0.25
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * 0.25
	
	return drive_torques

## Combined convenience method: returns {"drive": Array[float], "brake": Array[float]}
## Preserves complete backward compatibility with previous interface.
static func compute_torques(inputs: VehicleInputs, speed: float, config: VehicleConfig) -> Dictionary:
	var drive = compute_drive_torques(inputs, speed, config)
	var brake = BrakeModel.compute_brake_torques(inputs, config)
	return {"drive": drive, "brake": brake}
