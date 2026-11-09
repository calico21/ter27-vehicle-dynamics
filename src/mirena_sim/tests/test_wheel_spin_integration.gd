extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const WheelDynamicsScript = preload("../Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	var config = VehicleConfigScript.new()
	var wheel = WheelDynamicsScript.new(0, "FL")
	
	# Test 1: Airborne wheel spin-up under constant drive torque
	wheel.reset()
	wheel.is_grounded = false
	wheel.F_z = 0.0
	var dt = 0.01
	var T_drive = 120.0
	wheel.integrate_wheel_spin(T_drive, 0.0, dt, config)
	
	var expected_d_omega = (T_drive / config.wheel_inertia) * dt # 120 / 0.6 * 0.01 = 2.0 rad/s
	if absf(wheel.omega - expected_d_omega) < 1e-4:
		passed += 1
	else:
		failed += 1
		errors.append("Wheel free acceleration failed: got omega=%.4f, expected=%.4f" % [wheel.omega, expected_d_omega])
	
	# Test 2: Braking to standstill without reverse rotation chatter
	wheel.omega = 5.0
	wheel.integrate_wheel_spin(0.0, 600.0, dt, config)
	if absf(wheel.omega) < 1e-5:
		passed += 1
	else:
		failed += 1
		errors.append("Braking standstill zero-crossing clamp failed: omega is %.4f (expected 0.0)" % wheel.omega)
	
	# Test 3: Standstill brake holding
	wheel.omega = 0.0
	wheel.integrate_wheel_spin(0.0, 400.0, dt, config)
	if absf(wheel.omega) < 1e-5:
		passed += 1
	else:
		failed += 1
		errors.append("Standstill brake holding failed: wheel started moving with brake only (omega=%.4f)" % wheel.omega)
	
	# Test 4: Accumulated roll angle integration
	wheel.reset()
	wheel.omega = 10.0
	wheel.integrate_wheel_spin(0.0, 0.0, 0.05, config)
	var expected_theta = 10.0 * 0.05
	if absf(wheel.rotation_angle - expected_theta) < 1e-4:
		passed += 1
	else:
		failed += 1
		errors.append("Rotation angle integration failed: got theta=%.4f, expected=%.4f" % [wheel.rotation_angle, expected_theta])
	
	return {"name": "test_wheel_spin_integration", "passed": passed, "failed": failed, "errors": errors}
