extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	var config = VehicleConfigScript.new()
	var model = VehicleDynamicsModelScript.new(config)
	var inputs = VehicleInputsScript.new()
	
	# Test 1: Symmetric 4WD torque distribution at standstill (u = 0, throttle = 1.0)
	inputs.throttle = 1.0
	inputs.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	config.front_drive_torque_ratio = 0.50
	
	var torques = model.compute_drivetrain_torques(inputs, 0.0)
	var drive_trqs: Array[float] = torques["drive"]
	
	if absf(drive_trqs[0] - drive_trqs[1]) < 1e-4 and absf(drive_trqs[2] - drive_trqs[3]) < 1e-4 and absf(drive_trqs[0] - drive_trqs[2]) < 1e-4:
		passed += 1
	else:
		failed += 1
		errors.append("4WD symmetric torque distribution failed: %s" % str(drive_trqs))
	
	# Test 2: Power envelope capping at high speed (u = 30 m/s)
	var torques_high_speed = model.compute_drivetrain_torques(inputs, 30.0)
	var trq_sum = 0.0
	for t in torques_high_speed["drive"]:
		trq_sum += t
	
	var expected_trq_sum = (config.power_limit_watts * config.drivetrain_efficiency / 30.0) * config.wheel_radius
	if absf(trq_sum - expected_trq_sum) < 0.5:
		passed += 1
	else:
		failed += 1
		errors.append("Power envelope cap failed: got sum=%.2f Nm, expected=%.2f Nm" % [trq_sum, expected_trq_sum])
	
	# Test 3: Friction brake front/rear bias (bias_front = 0.60)
	inputs.throttle = 0.0
	inputs.brake = 1.0
	var b_torques = model.compute_drivetrain_torques(inputs, 10.0)
	var brake_trqs: Array[float] = b_torques["brake"]
	
	var front_brake_sum = brake_trqs[0] + brake_trqs[1]
	var rear_brake_sum = brake_trqs[2] + brake_trqs[3]
	var total_brake = front_brake_sum + rear_brake_sum
	
	var front_ratio = front_brake_sum / maxf(total_brake, 1e-4)
	if absf(front_ratio - config.brake_bias_front) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Brake bias failed: front ratio is %.4f, expected %.4f" % [front_ratio, config.brake_bias_front])
	
	# Test 4: Torque override for Torque Vectoring
	inputs.reset()
	inputs.override_wheel_torques = true
	var custom_drive: Array[float] = [120.0, 140.0, 220.0, 260.0]
	var custom_brake: Array[float] = [10.0, 15.0, 20.0, 25.0]
	inputs.wheel_motor_torques = custom_drive
	inputs.wheel_brake_torques = custom_brake
	
	var tv_torques = model.compute_drivetrain_torques(inputs, 15.0)
	if tv_torques["drive"] == inputs.wheel_motor_torques and tv_torques["brake"] == inputs.wheel_brake_torques:
		passed += 1
	else:
		failed += 1
		errors.append("Torque Vectoring direct override failed to pass through torques accurately")

	# Test 5: Effective reflected wheel inertia I_eff = I_wheel + G^2 * I_rotor
	var exp_i_eff: float = config.wheel_assembly_inertia + config.gear_ratio * config.gear_ratio * config.motor_rotor_inertia
	if absf(config.get_effective_wheel_inertia() - exp_i_eff) < 1e-6 and absf(exp_i_eff - 0.5075) < 1e-4:
		passed += 1
	else:
		failed += 1
		errors.append("Effective wheel inertia mismatch: got %.5f, expected %.5f" % [config.get_effective_wheel_inertia(), exp_i_eff])

	# Test 6: Field weakening & strict 80 kW DC bus electrical power ceiling
	inputs.reset()
	inputs.throttle = 1.0
	inputs.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	var w_fw: float = (12000.0 * TAU / 60.0) / config.gear_ratio
	var trq_fw = model.DrivetrainModel.compute_drive_torques(inputs, 20.0, config, [w_fw, w_fw, w_fw, w_fw])
	var p_elec_dc: float = 0.0
	for t in trq_fw:
		p_elec_dc += maxf(0.0, t * w_fw) / config.drivetrain_efficiency
	if p_elec_dc <= config.max_dc_power_w + 1e-2 and trq_fw[0] > 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("80 kW DC bus ceiling violated: P_elec=%.2f W (limit %.2f W)" % [p_elec_dc, config.max_dc_power_w])
	
	return {"name": "test_drivetrain_4wd", "passed": passed, "failed": failed, "errors": errors}
