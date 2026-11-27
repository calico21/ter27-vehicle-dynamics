extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	var config = VehicleConfigScript.new()
	var model = VehicleDynamicsModelScript.new(config)
	
	var total_weight = config.mass * config.gravity
	
	# Test 1: Static load distribution at rest (ax = 0, ay = 0, speed = 0)
	var static_loads = model.compute_analytic_normal_forces(0.0, 0.0, 0.0)
	var sum_static = static_loads[0] + static_loads[1] + static_loads[2] + static_loads[3]
	
	if absf(sum_static - total_weight) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Static load sum failed: got %.2f N, expected %.2f N" % [sum_static, total_weight])
	
	var front_static = static_loads[0] + static_loads[1]
	var rear_static = static_loads[2] + static_loads[3]
	var expected_front = total_weight * (config.b_cog / config.wheelbase)
	var expected_rear = total_weight * (config.a_cog / config.wheelbase)
	
	if absf(front_static - expected_front) < 1e-2 and absf(rear_static - expected_rear) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("Axle static distribution failed: front=%.2f (exp %.2f), rear=%.2f (exp %.2f)" % [
			front_static, expected_front, rear_static, expected_rear
		])
	
	# Test 2: Longitudinal acceleration load transfer (ax = +3.0 m/s^2)
	var accel_loads = model.compute_analytic_normal_forces(3.0, 0.0, 0.0)
	var front_accel = accel_loads[0] + accel_loads[1]
	var rear_accel = accel_loads[2] + accel_loads[3]
	var sum_accel = front_accel + rear_accel
	
	var dFz_expected = (config.mass * 3.0 * config.h_cog) / config.wheelbase
	if absf(sum_accel - total_weight) < 1e-3 and (front_accel < front_static) and (rear_accel > rear_static):
		if absf((rear_accel - rear_static) - dFz_expected) < 1e-2:
			passed += 1
		else:
			failed += 1
			errors.append("Longitudinal transfer delta failed: got %.2f N, expected %.2f N" % [rear_accel - rear_static, dFz_expected])
	else:
		failed += 1
		errors.append("Longitudinal load transfer direction/sum failed: front=%.2f, rear=%.2f" % [front_accel, rear_accel])
	
	# Test 3: Lateral acceleration load transfer (ay = +4.0 m/s^2, left turn)
	var cornering_loads = model.compute_analytic_normal_forces(0.0, 4.0, 0.0)
	var left_wheels = cornering_loads[0] + cornering_loads[2]
	var right_wheels = cornering_loads[1] + cornering_loads[3]
	
	if right_wheels > left_wheels and absf((left_wheels + right_wheels) - total_weight) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Lateral load transfer failed: left=%.2f, right=%.2f (right must be > left)" % [left_wheels, right_wheels])
	
	return {"name": "test_load_transfer", "passed": passed, "failed": failed, "errors": errors}
