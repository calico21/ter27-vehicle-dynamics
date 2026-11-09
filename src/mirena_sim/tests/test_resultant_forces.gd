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
	
	# Test 1: Straight line acceleration equilibrium
	inputs.throttle = 0.8
	inputs.steer_angle = 0.0
	
	var step_res = model.step(inputs, Vector3(5.0, 0.0, 0.0), Vector3.ZERO, [], 0.0025)
	var f_net_ros: Vector3 = step_res["net_force_ros"]
	var tau_net_ros: Vector3 = step_res["net_torque_ros"]
	
	if absf(f_net_ros.y) < 1e-3 and absf(tau_net_ros.z) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Straight line symmetry violated: net_Fy=%.4f N, net_tau_z=%.4f Nm" % [f_net_ros.y, tau_net_ros.z])
	
	# Test 2: Steer step response direction
	model.reset()
	for w in model.wheels:
		w.omega = 50.0 # free rolling at 10 m/s (u / R = 10 / 0.2)
	inputs.throttle = 0.0
	inputs.steer_angle = 0.15
	
	var steer_res = model.step(inputs, Vector3(10.0, 0.0, 0.0), Vector3.ZERO, [], 0.0025)
	var f_steer_ros: Vector3 = steer_res["net_force_ros"]
	var tau_steer_ros: Vector3 = steer_res["net_torque_ros"]
	
	if f_steer_ros.y > 0.0 and tau_steer_ros.z > 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Steer step direction failed: net_Fy=%.2f N (must be > 0), net_tau_z=%.2f Nm (must be > 0)" % [
			f_steer_ros.y, tau_steer_ros.z
		])
	
	# Test 3: Consistency between wheel contact sum and net chassis force
	var wheel_force_sum = Vector3.ZERO
	for i in range(4):
		var w = model.wheels[i]
		var delta_w = inputs.steer_angle if (i == 0 or i == 1) else 0.0
		var fx_c = w.F_x * cos(delta_w) - w.F_y * sin(delta_w)
		var fy_c = w.F_x * sin(delta_w) + w.F_y * cos(delta_w)
		wheel_force_sum += Vector3(fx_c, fy_c, w.F_z)
	
	var aero = model.compute_aerodynamics(Vector3(10.0, 0.0, 0.0))
	var expected_net_force = wheel_force_sum + aero["force"]
	
	if f_steer_ros.distance_to(expected_net_force) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Resultant force sum mismatch: model returned %s, wheel sum was %s" % [f_steer_ros, expected_net_force])
	
	return {"name": "test_resultant_forces", "passed": passed, "failed": failed, "errors": errors}
