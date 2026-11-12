class_name DrivetrainModel
extends RefCounted

## Drivetrain & Powertrain Model for Mirena Dynamics (Block D)
## Computes per-wheel motor drive torques [FL, FR, RL, RR] with:
## - Formula Student 80 kW global power ceiling
## - Motor angular speed limiter (linear taper to 0 tractive torque at motor_max_rpm)
## - Decoupled 200 Hz Torque Vectoring layer (MODE_OPEN / MODE_DYC)

const VehicleConfig = preload("../Core/VehicleConfig.gd")
const VehicleInputs = preload("../Core/VehicleInputs.gd")
const BrakeModel = preload("../Brakes/BrakeModel.gd")
const TorqueVectoring = preload("TorqueVectoring.gd")

## Computes base symmetric drive torques for all 4 wheels in Nm.
static func compute_drive_torques(
	inputs: VehicleInputs,
	speed: float,
	config: VehicleConfig,
	wheel_omegas: Array = []
) -> Array[float]:
	var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	if inputs.override_wheel_torques:
		for i in range(4):
			drive_torques[i] = inputs.wheel_motor_torques[i]
		return drive_torques
	
	var throttle = inputs.throttle
	if throttle > 0.0:
		var r = maxf(config.wheel_radius, 1e-4)
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
				drive_torques[VehicleConfig.WheelIndex.FL] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.FR] = requested_total_trq * s_front * 0.5
				drive_torques[VehicleConfig.WheelIndex.RL] = requested_total_trq * s_rear * 0.5
				drive_torques[VehicleConfig.WheelIndex.RR] = requested_total_trq * s_rear * 0.5

		# Apply per-wheel motor angular speed limiter (taper near cutoff, 0 beyond motor_max_rpm)
		var fallback_omega: float = absf(speed) / r
		for i in range(4):
			var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
			if drive_torques[i] > 0.0:
				var speed_scale: float = TorqueVectoring.compute_motor_speed_scale(w_i, true, config)
				drive_torques[i] = minf(drive_torques[i], max_wheel_trq * speed_scale)
	
	return drive_torques

## Combined convenience method: returns {"drive": Array[float], "brake": Array[float]}.
## Optionally routes drive torques through a stateful 200 Hz TorqueVectoring controller.
static func compute_torques(
	inputs: VehicleInputs,
	speed: float,
	config: VehicleConfig,
	wheel_omegas: Array = [],
	tv: TorqueVectoring = null,
	dt: float = 0.0,
	advance_tv: bool = true
) -> Dictionary:
	var drive = compute_drive_torques(inputs, speed, config, wheel_omegas)
	if tv != null:
		drive = tv.step_allocation(drive, inputs, wheel_omegas, speed, dt, config, advance_tv)
	var brake = BrakeModel.compute_brake_torques(inputs, config)
	return {"drive": drive, "brake": brake}
