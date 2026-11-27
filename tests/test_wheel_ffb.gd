extends RefCounted

const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")

static func run() -> Dictionary:
	var st := {"name": "test_wheel_ffb", "passed": 0, "failed": 0, "errors": []}

	# 1. normalizacion de pedales (invertido y bipolar) + linealidad 1:1 de direccion
	var inp := VehicleInputsScript.new()
	inp.calibrate_wheel_pedals(1.0, 1.0, VehicleInputsScript.PedalAxisMode.INVERTED, VehicleInputsScript.PedalAxisMode.BIPOLAR)
	inp.set_from_physical_wheel(0.0, 1.0, 1.0, 0.38)
	var rest_ok := (inp.throttle == 0.0 and inp.brake == 0.0 and absf(inp.steer_angle) < 1e-6)
	inp.set_from_physical_wheel(0.10, 0.0, -1.0, 0.38)
	# 900 deg * 0.5 * 0.10 = 45 deg -> 45 / 115 = 0.391304 (1:1 lineal sin deadzone ni gamma)
	var exp_norm := (0.10 * 450.0) / 115.0
	var lin_ok := (absf(inp.throttle - 1.0) < 1e-4 and absf(inp.brake - 1.0) < 1e-4 and absf(inp.steer_normalized - exp_norm) < 1e-4)
	if rest_ok and lin_ok:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Wheel input normalization/linearity failed: rest_ok=%s thr=%.3f brk=%.3f str=%.4f" % [rest_ok, inp.throttle, inp.brake, inp.steer_normalized])

	# 2. signo fisico correcto de steering_feedback_torque_nm (se opone al giro en apoyo lateral)
	var m := Full14DofScript.new()
	m.load_vehicle("ter27_4wd", "default")
	m.chassis_state.linear_velocity = Vector3(15.0, 0.0, 0.0)
	for w in m.wheels:
		w.omega = 15.0 / maxf(m.config.wheel_radius, 0.1)
	var inp_turn := VehicleInputsScript.new()
	inp_turn.drive_mode = m.config.drive_mode
	inp_turn.throttle = 0.25
	inp_turn.steer_angle = deg_to_rad(4.0)  # giro izq (+), ffb debe tirar a der (-)
	for _k in range(40):
		m.step_standalone(inp_turn, 0.0025)
	if m.steering_feedback_torque_nm < -0.5 and m.steering_feedback_normalized < -0.05 and absf(m.steering_feedback_normalized) <= 1.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("FFB opposing sign failed: trq=%.3f norm=%.3f" % [m.steering_feedback_torque_nm, m.steering_feedback_normalized])

	# 3. caida del par de autoalineamiento (sat) al acercarse a saturacion ~1.5 g
	var sat_mid := absf(m.wheels[0].M_z + m.wheels[1].M_z)
	var inp_sat := VehicleInputsScript.new()
	inp_sat.drive_mode = m.config.drive_mode
	inp_sat.throttle = 0.20
	inp_sat.steer_angle = deg_to_rad(14.0)  # angulo alto pa saturar eje del
	for _k in range(50):
		m.step_standalone(inp_sat, 0.0025)
	var sat_high := absf(m.wheels[0].M_z + m.wheels[1].M_z)
	if sat_high < sat_mid:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("SAT drop near saturation failed: sat_mid=%.2f sat_high=%.2f" % [sat_mid, sat_high])

	# 4. par de scrub radius al aplicar par diferencial entre fl y fr
	var m_sc := Full14DofScript.new()
	m_sc.load_vehicle("ter27_4wd", "default")
	m_sc.chassis_state.linear_velocity = Vector3(14.0, 0.0, 0.0)
	for w in m_sc.wheels:
		w.omega = 14.0 / maxf(m_sc.config.wheel_radius, 0.1)
	var inp_sc := VehicleInputsScript.new()
	inp_sc.override_wheel_torques = true
	inp_sc.steer_angle = 0.0
	inp_sc.wheel_motor_torques.assign([-140.0, 140.0, 0.0, 0.0])  # fx_fr > fx_fl -> par positivo (+z)
	for _k in range(10):
		m_sc.step_standalone(inp_sc, 0.0025)
	if m_sc.steering_feedback_torque_nm > 0.5:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Scrub radius differential torque failed: trq=%.3f" % m_sc.steering_feedback_torque_nm)

	return st
