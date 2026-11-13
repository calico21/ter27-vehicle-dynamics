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

		# Apply per-wheel field-weakening envelope + motor angular speed limiter
		var fallback_omega: float = absf(speed) / r
		var omega_base_motor: float = maxf(config.base_speed_rpm, 0.0) * (TAU / 60.0)
		var p_motor_peak: float = config.motor_peak_torque * omega_base_motor
		for i in range(4):
			var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
			if drive_torques[i] > 0.0:
				var omega_motor: float = absf(w_i) * maxf(G, 1e-3)
				var t_motor_env: float = config.motor_peak_torque
				if omega_base_motor > 0.0 and omega_motor > omega_base_motor:
					t_motor_env = minf(config.motor_peak_torque, p_motor_peak / omega_motor)
				var speed_scale: float = TorqueVectoring.compute_motor_speed_scale(w_i, true, config)
				var max_trq_i: float = t_motor_env * G * eff * speed_scale
				drive_torques[i] = minf(drive_torques[i], max_trq_i)

		# Enforce strict 80 kW DC bus electrical power ceiling: sum(max(0, T_i * w_i) / eff) <= P_max_dc
		var p_dc_limit: float = maxf(config.max_dc_power_w, 0.0)
		if p_dc_limit > 0.0:
			var p_elec_total: float = 0.0
			for i in range(4):
				var w_i: float = float(wheel_omegas[i]) if i < wheel_omegas.size() else fallback_omega
				p_elec_total += maxf(0.0, drive_torques[i] * absf(w_i)) / maxf(eff, 1e-3)
			if p_elec_total > p_dc_limit:
				var p_scale: float = p_dc_limit / p_elec_total
				for i in range(4):
					drive_torques[i] *= p_scale
	
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
