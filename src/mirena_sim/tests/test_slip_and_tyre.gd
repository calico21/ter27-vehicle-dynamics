extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const WheelDynamicsScript = preload("../Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	var config = VehicleConfigScript.new()
	var wheel = WheelDynamicsScript.new(0, "FL")
	
	var F_z_nominal = 750.0
	
	# Test 1: Linear regime for small longitudinal slip (kappa = 0.01)
	wheel.reset()
	wheel.omega = (2.0 * 1.01) / config.wheel_radius
	for step in range(50):
		wheel.compute_slip(2.0, 0.0, 0.0, config.slip_regularization_speed, config.wheel_radius, 0.0025)
		wheel.compute_forces(F_z_nominal, config)
	
	if wheel.F_x > 250.0 and wheel.F_x < 450.0:
		passed += 1
	else:
		failed += 1
		errors.append("Linear slip Fx failed: got Fx=%.2f N (expected 250 to 450 N)" % wheel.F_x)
	
	# Test 2: Friction saturation under massive combined demand
	wheel.reset()
	wheel.omega = 30.0
	wheel.compute_slip(3.0, 1.0, 0.0, config.slip_regularization_speed, config.wheel_radius)
	wheel.compute_forces(F_z_nominal, config)
	
	var total_force = sqrt(pow(wheel.F_x, 2) + pow(wheel.F_y, 2))
	var max_allowed = 2.65 * F_z_nominal # peak friction limit
	
	if total_force <= max_allowed:
		passed += 1
	else:
		failed += 1
		errors.append("Friction saturation violated: total_force = %.2f N (max allowed = %.2f N)" % [total_force, max_allowed])
	
	# Test 3: Zero normal load yields zero forces
	wheel.reset()
	wheel.omega = 20.0
	wheel.compute_slip(5.0, 1.0, 0.0, config.slip_regularization_speed, config.wheel_radius)
	wheel.compute_forces(0.0, config)
	
	if wheel.F_x == 0.0 and wheel.F_y == 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Zero normal load test failed: Fx=%.2f, Fy=%.2f (must be 0.0)" % [wheel.F_x, wheel.F_y])
	
	# Test 4: Lateral force restores opposite to lateral slip
	wheel.reset()
	wheel.omega = 10.0
	wheel.compute_slip(2.0, 0.5, 0.0, config.slip_regularization_speed, config.wheel_radius)
	wheel.compute_forces(F_z_nominal, config)
	
	if wheel.F_y < 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Lateral restoring force direction failed: vy > 0 produced Fy=%.2f (must be negative)" % wheel.F_y)

	# Test 5: Canonical MF 6.2 lambda_muy scales peak Fy without distorting linear Kya
	var TireModelScript = preload("../Scenes/Vehicle/Dynamics/Tires/TireModel.gd")
	var mf = TireModelScript.get_mf62()
	mf.LMUY = 1.0
	var comb_full = mf.compute_combined_forces(0.0, deg_to_rad(8.0), F_z_nominal, 0.0, 83.0, false)
	mf.LMUY = 0.80
	var comb_scaled = mf.compute_combined_forces(0.0, deg_to_rad(8.0), F_z_nominal, 0.0, 83.0, false)
	mf.LMUY = 1.0
	var kya_diff = absf(float(comb_full["Kya"]) - float(comb_scaled["Kya"]))
	var fy_ratio = absf(float(comb_scaled["Fy"])) / maxf(absf(float(comb_full["Fy"])), 1e-3)
	if kya_diff < 1e-3 and fy_ratio > 0.70 and fy_ratio < 0.90:
		passed += 1
	else:
		failed += 1
		errors.append("Canonical LMUY scaling failed: kya_diff=%.4f, fy_ratio=%.4f" % [kya_diff, fy_ratio])
	
	return {"name": "test_slip_and_tyre", "passed": passed, "failed": failed, "errors": errors}
