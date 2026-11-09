extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleDynamicsModelScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const VehicleInputsScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	var config = VehicleConfigScript.new()
	var model = VehicleDynamicsModelScript.new(config)
	var inputs = VehicleInputsScript.new()
	
	# Test 1: Pure longitudinal translation (u = 10 m/s, v = 0, r = 0)
	var v_trans = Vector3(10.0, 0.0, 0.0)
	var w_zero = Vector3.ZERO
	model.step(inputs, v_trans, w_zero, [], 0.0025)
	
	var all_wheels_match = true
	for i in range(4):
		var w = model.wheels[i]
		if absf(w.v_longitudinal - 10.0) > 1e-3 or absf(w.v_lateral) > 1e-3:
			all_wheels_match = false
			errors.append("Wheel %s kinematics failed on pure translation: vx=%.3f, vy=%.3f" % [w.name, w.v_longitudinal, w.v_lateral])
	if all_wheels_match:
		passed += 1
	else:
		failed += 1
	
	# Test 2: Pure yaw rotation (u = 0, v = 0, r = 1.0 rad/s)
	model.reset()
	var v_zero = Vector3.ZERO
	var w_yaw = Vector3(0.0, 0.0, 1.0)
	model.step(inputs, v_zero, w_yaw, [], 0.0025)
	
	var w_fl = model.wheels[0]
	var w_fr = model.wheels[1]
	var w_rl = model.wheels[2]
	var w_rr = model.wheels[3]
	
	var wf_half = config.track_width_f * 0.5
	var wr_half = config.track_width_r * 0.5
	var a = config.a_cog
	var b = config.b_cog
	
	var yaw_ok = true
	if absf(w_fl.v_longitudinal - (-wf_half)) > 1e-3 or absf(w_fl.v_lateral - a) > 1e-3:
		yaw_ok = false
		errors.append("FL yaw kinematics failed: got vx=%.3f, vy=%.3f, expected (%.3f, %.3f)" % [w_fl.v_longitudinal, w_fl.v_lateral, -wf_half, a])
	if absf(w_fr.v_longitudinal - wf_half) > 1e-3 or absf(w_fr.v_lateral - a) > 1e-3:
		yaw_ok = false
		errors.append("FR yaw kinematics failed: got vx=%.3f, vy=%.3f, expected (%.3f, %.3f)" % [w_fr.v_longitudinal, w_fr.v_lateral, wf_half, a])
	if absf(w_rl.v_longitudinal - (-wr_half)) > 1e-3 or absf(w_rl.v_lateral - (-b)) > 1e-3:
		yaw_ok = false
		errors.append("RL yaw kinematics failed: got vx=%.3f, vy=%.3f, expected (%.3f, %.3f)" % [w_rl.v_longitudinal, w_rl.v_lateral, -wr_half, -b])
	if absf(w_rr.v_longitudinal - wr_half) > 1e-3 or absf(w_rr.v_lateral - (-b)) > 1e-3:
		yaw_ok = false
		errors.append("RR yaw kinematics failed: got vx=%.3f, vy=%.3f, expected (%.3f, %.3f)" % [w_rr.v_longitudinal, w_rr.v_lateral, wr_half, -b])
	
	if yaw_ok:
		passed += 1
	else:
		failed += 1
	
	# Test 3: Steered front wheels (delta = 0.2 rad, u = 10 m/s)
	model.reset()
	inputs.steer_angle = 0.2
	model.step(inputs, Vector3(10.0, 0.0, 0.0), Vector3.ZERO, [], 0.0025)
	
	var expected_vx_steer = 10.0 * cos(0.2)
	var expected_vy_steer = -10.0 * sin(0.2)
	
	if absf(w_fl.v_longitudinal - expected_vx_steer) < 1e-3 and absf(w_fl.v_lateral - expected_vy_steer) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Steered FL kinematics failed: got (%.3f, %.3f), expected (%.3f, %.3f)" % [
			w_fl.v_longitudinal, w_fl.v_lateral, expected_vx_steer, expected_vy_steer
		])
	
	return {"name": "test_wheel_kinematics", "passed": passed, "failed": failed, "errors": errors}
