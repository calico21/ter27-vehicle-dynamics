extends RefCounted

## Automated Test Suite for Pacejka MF 6.2 + Neural Residual + Transient Relaxation
## Tests physical validity, calibration fidelity, neural bounds, and transient dynamics.

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const WheelDynamicsScript = preload("../Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")
const TireModelScript = preload("../Scenes/Vehicle/Dynamics/Tires/TireModel.gd")
const Pacejka62Script = preload("../Scenes/Vehicle/Dynamics/Tires/MF62/Pacejka62.gd")
const NeuralCorrectionScript = preload("../Scenes/Vehicle/Dynamics/Tires/NeuralCorrection/NeuralTireCorrection.gd")
const TransientRelaxationScript = preload("../Scenes/Vehicle/Dynamics/Tires/Transient/TransientRelaxation.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors: Array[String] = []
	
	var config = VehicleConfigScript.new()
	var mf = Pacejka62Script.new()
	var nn = NeuralCorrectionScript.new()
	var wheel = WheelDynamicsScript.new(0, "FL")
	
	var F_z_nom = 800.0 # N
	
	# --- Test 1: MF 6.2 Pure Lateral Force Peak and Shape ---
	var fy_peak = 0.0
	var alpha_peak_deg = 0.0
	for deg in range(0, 16):
		var a_rad = deg_to_rad(float(deg))
		var fy = mf.compute_fy0(a_rad, F_z_nom, 0.0, 83.0)
		if fy > fy_peak:
			fy_peak = fy
			alpha_peak_deg = float(deg)
	
	var mu_peak = fy_peak / F_z_nom
	# Hoosier 16x7.5-10 R25B reaches mu ~ 2.4 - 2.6 at 800 N
	if mu_peak >= 2.2 and mu_peak <= 2.8 and alpha_peak_deg >= 7.0 and alpha_peak_deg <= 11.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Lateral Peak check failed: peak_mu=%.2f (expected 2.2-2.8), alpha_peak=%.1f deg (expected 7-11)" % [mu_peak, alpha_peak_deg])
	
	# --- Test 2: Self-Aligning Torque Mz Peak and Zero-Crossing ---
	# Mz starts positive, peaks around 3-5 deg, then drops toward zero/negative past peak lateral slip
	var mz_small = mf.compute_mz(deg_to_rad(3.0), F_z_nom, mf.compute_fy0(deg_to_rad(3.0), F_z_nom), 0.0, 83.0)
	var mz_high = mf.compute_mz(deg_to_rad(12.0), F_z_nom, mf.compute_fy0(deg_to_rad(12.0), F_z_nom), 0.0, 83.0)
	
	if mz_small > 25.0 and mz_high < mz_small:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Mz alignment curve check failed: mz(3 deg)=%.2f Nm, mz(12 deg)=%.2f Nm" % [mz_small, mz_high])
	
	# --- Test 3: Camber Sensitivity (IA = 4 deg vs 0 deg) ---
	var fy_c0 = mf.compute_fy0(deg_to_rad(4.0), F_z_nom, 0.0, 83.0)
	var fy_c4 = mf.compute_fy0(deg_to_rad(4.0), F_z_nom, deg_to_rad(4.0), 83.0)
	# Positive camber adds camber thrust
	if fy_c4 != fy_c0 and absf(fy_c4 - fy_c0) > 10.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Camber sensitivity check failed: fy_c0=%.1f N, fy_c4=%.1f N" % [fy_c0, fy_c4])
	
	# --- Test 4: Pressure Dependence ---
	var fy_p55 = mf.compute_fy0(deg_to_rad(8.0), F_z_nom, 0.0, 55.0) # 8 psi
	var fy_p83 = mf.compute_fy0(deg_to_rad(8.0), F_z_nom, 0.0, 83.0) # 12 psi
	if fy_p55 != fy_p83 and absf(fy_p55 - fy_p83) > 15.0:
		passed += 1
	else:
		failed += 1
		errors.append("MF6.2 Pressure sensitivity check failed: fy(55 kPa)=%.1f N, fy(83 kPa)=%.1f N" % [fy_p55, fy_p83])
	
	# --- Test 5: Neural Residual Physics Gates (Zero-Slip & Low-Load) ---
	var zero_corr = nn.predict_correction(0.0, F_z_nom, 0.0, 83.0)
	var air_corr = nn.predict_correction(deg_to_rad(5.0), 0.0, 0.0, 83.0)
	if absf(zero_corr["delta_Fy"]) < 1e-3 and absf(air_corr["delta_Fy"]) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Neural physics gating failed: zero_slip_Fy=%.4f, airborne_Fy=%.4f (must be 0.0)" % [zero_corr["delta_Fy"], air_corr["delta_Fy"]])
	
	# --- Test 6: Neural Residual Saturation Clamp Limit ---
	# Even at extreme input, correction must never exceed safety limits (+-200 N, +-10 Nm)
	var extreme_corr = nn.predict_correction(deg_to_rad(25.0), 2000.0, deg_to_rad(10.0), 120.0)
	if absf(extreme_corr["delta_Fy"]) <= 200.01 and absf(extreme_corr["delta_Mz"]) <= 10.01:
		passed += 1
	else:
		failed += 1
		errors.append("Neural saturation limit exceeded: delta_Fy=%.2f, delta_Mz=%.2f" % [extreme_corr["delta_Fy"], extreme_corr["delta_Mz"]])
	
	# --- Test 7: Transient Carcass Relaxation Dynamics ---
	# Apply step steer input from 0 to 0.1 rad at 10 m/s
	var alpha_target = 0.10
	var alpha_trans = 0.0
	var dt = 0.0025 # 400 Hz
	
	# Integrate 10 steps (0.025 s -> 0.25 m traveled at 10 m/s ~ 1 relaxation length)
	for step in range(10):
		alpha_trans = TransientRelaxationScript.integrate_alpha_relaxation(alpha_trans, alpha_target, 10.0, dt, F_z_nom)
	
	# After ~1 relaxation length, transient angle should reach ~63% of target (0.05 to 0.08)
	if alpha_trans > 0.04 and alpha_trans < 0.095:
		passed += 1
	else:
		failed += 1
		errors.append("Transient relaxation dynamic response failed: alpha_trans=%.4f (expected 0.04 to 0.095)" % alpha_trans)
	
	# --- Test 8: End-to-End Hybrid Wheel Integration ---
	wheel.reset()
	wheel.omega = 50.0
	wheel.compute_slip(10.0, -1.0, 0.0, config.slip_regularization_speed, config.wheel_radius, dt)
	wheel.compute_forces(F_z_nom, config)
	
	if wheel.F_y > 500.0 and absf(wheel.M_z) > 1.0 and wheel.is_grounded:
		passed += 1
	else:
		failed += 1
		errors.append("Hybrid wheel force integration failed: Fy=%.2f N, Mz=%.2f Nm" % [wheel.F_y, wheel.M_z])
	
	return {
		"name": "test_tire_mf62_suite",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
