class_name TestTransientDynamics
extends SceneTree

## Dedicated Verification Test Suite for Transient Carcass Relaxation Dynamics
## Tests step steer lag, longitudinal slip lag, standstill damping, sign reversals, and dt sensitivity.

const TransientRelaxationScript = preload("res://Scenes/Vehicle/Dynamics/Tires/Transient/TransientRelaxation.gd")

func _init() -> void:
	var res = run()
	if res["failed"] == 0:
		quit(0)
	else:
		quit(1)

static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array[String] = []
	
	var F_z = 800.0
	var sigma_y = TransientRelaxationScript.get_lateral_relaxation_length(F_z)
	var sigma_x = TransientRelaxationScript.get_longitudinal_relaxation_length(F_z)
	
	# --- Check 1: Theoretical Relaxation Length Values ---
	if absf(sigma_y - 0.180) < 1e-3 and absf(sigma_x - 0.100) < 1e-3:
		passed += 1
	else:
		failed += 1
		errors.append("Nominal relaxation lengths incorrect: sigma_y=%.3f, sigma_x=%.3f" % [sigma_y, sigma_x])
		
	# --- Check 2: Step Steer Response at 10 m/s ---
	# Target step: alpha_target = 0.10 rad.
	# At V = 10 m/s, time constant tau = sigma_y / V = 0.180 / 10 = 0.018 s.
	# At t = tau = 0.018 s, alpha should reach (1 - 1/e) * 0.10 = 0.0632 rad.
	var alpha_cur = 0.0
	var alpha_target = 0.10
	var dt = 0.001 # 1 ms steps
	var v_x = 10.0
	
	for i in range(18): # 18 ms = exactly 1 tau
		alpha_cur = TransientRelaxationScript.integrate_alpha_relaxation(alpha_cur, alpha_target, v_x, dt, F_z)
		
	var expected_1tau = alpha_target * (1.0 - exp(-1.0)) # ~0.0632 rad
	if absf(alpha_cur - expected_1tau) < 0.002:
		passed += 1
	else:
		failed += 1
		errors.append("Step steer 1-tau lag failed: reached=%.4f, expected=%.4f" % [alpha_cur, expected_1tau])
		
	# After 3 tau (54 ms), alpha should reach > 95% of target (0.095 rad)
	for i in range(36): # 54 ms total
		alpha_cur = TransientRelaxationScript.integrate_alpha_relaxation(alpha_cur, alpha_target, v_x, dt, F_z)
		
	if alpha_cur >= 0.094:
		passed += 1
	else:
		failed += 1
		errors.append("Step steer 3-tau saturation failed: reached=%.4f (expected >= 0.094)" % alpha_cur)
		
	# --- Check 3: Standstill Dissipation at Zero Speed (Vx = 0) ---
	# Vehicle is stopped with residual deflection alpha = 0.05, target = 0.0 (straight steering)
	var alpha_stopped = 0.05
	for i in range(500): # 0.5 seconds at standstill (dt = 0.001)
		alpha_stopped = TransientRelaxationScript.integrate_alpha_relaxation(alpha_stopped, 0.0, 0.0, dt, F_z)
		
	# Deflection should smoothly damp toward zero without chattering
	if absf(alpha_stopped) < 0.005:
		passed += 1
	else:
		failed += 1
		errors.append("Standstill residual damping failed: alpha_stopped=%.4f (must be < 0.005)" % alpha_stopped)
		
	# --- Check 4: Sign Reversal Stability ---
	# Car transitions from +5 m/s to -5 m/s
	var alpha_rev = 0.05
	alpha_rev = TransientRelaxationScript.integrate_alpha_relaxation(alpha_rev, 0.08, 5.0, 0.005, F_z)
	alpha_rev = TransientRelaxationScript.integrate_alpha_relaxation(alpha_rev, 0.08, 0.0, 0.005, F_z)
	alpha_rev = TransientRelaxationScript.integrate_alpha_relaxation(alpha_rev, 0.08, -5.0, 0.005, F_z)
	if not is_nan(alpha_rev) and not is_inf(alpha_rev) and alpha_rev > 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Velocity reversal test failed: alpha_rev=" + str(alpha_rev))
		
	# --- Check 5: Time Step Sensitivity (200 Hz vs 1000 Hz) ---
	# Compare 50 ms simulation of step response with dt=5ms (200 Hz) vs dt=1ms (1000 Hz)
	var a_200hz = 0.0
	for i in range(10): # 10 * 5ms = 50ms
		a_200hz = TransientRelaxationScript.integrate_alpha_relaxation(a_200hz, 0.10, 10.0, 0.005, F_z)
		
	var a_1000hz = 0.0
	for i in range(50): # 50 * 1ms = 50ms
		a_1000hz = TransientRelaxationScript.integrate_alpha_relaxation(a_1000hz, 0.10, 10.0, 0.001, F_z)
		
	var dt_rel_diff = absf(a_200hz - a_1000hz) / a_1000hz
	if dt_rel_diff < 0.01: # Less than 1% difference due to exact exponential formulation!
		passed += 1
	else:
		failed += 1
		errors.append("Time step sensitivity check failed: rel_diff=%.4f (must be < 0.01)" % dt_rel_diff)
		
	return {
		"name": "test_transient_dynamics",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
