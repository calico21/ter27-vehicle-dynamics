extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputsScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const DrivetrainModelScript = preload("../Scenes/Vehicle/Dynamics/Drivetrain/DrivetrainModel.gd")
const TorqueVectoringScript = preload("../Scenes/Vehicle/Dynamics/Drivetrain/TorqueVectoring.gd")
const AeroModelScript = preload("../Scenes/Vehicle/Dynamics/Aerodynamics/AeroModel.gd")

static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var cfg = VehicleConfigScript.new()

	# ---------------------------------------------------------
	# 1. Motor angular speed limiter (taper + cutoff at motor_max_rpm)
	# ---------------------------------------------------------
	var omega_wheel_low: float = (8000.0 * TAU / 60.0) / cfg.gear_ratio
	var mid_rpm: float = 0.5 * (cfg.motor_taper_ratio + 1.0) * cfg.motor_max_rpm
	var omega_wheel_mid: float = (mid_rpm * TAU / 60.0) / cfg.gear_ratio
	var omega_wheel_max: float = (cfg.motor_max_rpm * TAU / 60.0) / cfg.gear_ratio

	var sc_low: float = TorqueVectoringScript.compute_motor_speed_scale(omega_wheel_low, true, cfg)
	var sc_mid: float = TorqueVectoringScript.compute_motor_speed_scale(omega_wheel_mid, true, cfg)
	var sc_cut: float = TorqueVectoringScript.compute_motor_speed_scale(omega_wheel_max + 5.0, true, cfg)
	var sc_regen: float = TorqueVectoringScript.compute_motor_speed_scale(omega_wheel_max + 5.0, false, cfg)

	var inp_drv = VehicleInputsScript.new()
	inp_drv.throttle = 1.0
	var trq_at_cut: Array[float] = DrivetrainModelScript.compute_drive_torques(
		inp_drv, 20.0, cfg, [omega_wheel_max, omega_wheel_max, omega_wheel_max, omega_wheel_max]
	)
	if absf(sc_low - 1.0) < 1e-6 and absf(sc_mid - 0.5) < 1e-2 and sc_cut == 0.0 and sc_regen == 1.0 \
			and trq_at_cut[0] == 0.0 and trq_at_cut[1] == 0.0 and trq_at_cut[2] == 0.0 and trq_at_cut[3] == 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Motor speed limiter failed: sc=(%.3f, %.3f, %.3f, %.3f), trq_cut=%s" % [sc_low, sc_mid, sc_cut, sc_regen, str(trq_at_cut)])

	# ---------------------------------------------------------
	# 2. Unified longitudinal input & continuous gas<0 braking without wheel reversal
	# ---------------------------------------------------------
	var inp_u = VehicleInputsScript.new()
	inp_u.set_unified_control(-0.6, 0.1, 0.0, cfg.max_steer_angle)
	var ok_neg: bool = (inp_u.throttle == 0.0 and absf(inp_u.brake - 0.6) < 1e-6)
	inp_u.set_unified_control(0.4, 0.0, 0.8, cfg.max_steer_angle)
	var ok_man: bool = (absf(inp_u.throttle - 0.4) < 1e-6 and absf(inp_u.brake - 0.8) < 1e-6)
	if ok_neg and ok_man:
		passed += 1
	else:
		failed += 1
		errors.append("Unified control mapping failed")

	var model_brk = VehicleDynamicsModelScript.new(cfg)
	var u0_brk: float = 10.0
	model_brk.chassis_state.linear_velocity = Vector3(u0_brk, 0.0, 0.0)
	for w in model_brk.wheels:
		w.omega = u0_brk / cfg.wheel_radius
	var inp_brk = VehicleInputsScript.new()
	inp_brk.set_from_ros_control(-0.8, 0.0, cfg.max_steer_angle)

	var min_omega_seen: float = 999.0
	for _k in range(800): # 2.0 s @ 400 Hz
		model_brk.step_standalone(inp_brk, 0.0025, 0.0)
		for w in model_brk.wheels:
			min_omega_seen = minf(min_omega_seen, w.omega)

	var max_final_omega: float = 0.0
	for w in model_brk.wheels:
		max_final_omega = maxf(max_final_omega, absf(w.omega))

	if min_omega_seen >= -1e-4 and max_final_omega < 1e-2 and absf(model_brk.chassis_state.linear_velocity.x) < 0.1:
		passed += 1
	else:
		failed += 1
		errors.append("Continuous gas<0 braking failed: min_omega=%.5f, final_max_omega=%.4f, u=%.3f" % [
			min_omega_seen, max_final_omega, model_brk.chassis_state.linear_velocity.x
		])

	# ---------------------------------------------------------
	# 3. 200 Hz Torque Vectoring (accumulator + MODE_OPEN + MODE_DYC bounds)
	# ---------------------------------------------------------
	var model_tv = VehicleDynamicsModelScript.new(cfg)
	model_tv.chassis_state.linear_velocity = Vector3(12.0, 0.0, 0.0)
	for w in model_tv.wheels:
		w.omega = 12.0 / cfg.wheel_radius

	var inp_tv = VehicleInputsScript.new()
	inp_tv.drive_mode = VehicleConfigScript.DriveMode.TORQUE_VECTORING
	inp_tv.throttle = 0.7
	inp_tv.tv_mode = TorqueVectoringScript.TVMode.MODE_OPEN

	for _k in range(40): # 0.10 s @ 400 Hz -> 20 TV updates @ 200 Hz
		model_tv.step_standalone(inp_tv, 0.0025, 0.0)

	var latched_open: Array[float] = model_tv.torque_vectoring.get_latched_torques()
	var open_sym: bool = absf(latched_open[0] - latched_open[1]) < 1e-4 and absf(latched_open[2] - latched_open[3]) < 1e-4 and latched_open[0] > 10.0
	if model_tv.torque_vectoring.update_count == 20 and open_sym:
		passed += 1
	else:
		failed += 1
		errors.append("200 Hz TV clock/OPEN check failed: count=%d (exp 20), open=%s" % [model_tv.torque_vectoring.update_count, str(latched_open)])

	# Test MODE_DYC: +Mz left turn shifts torque RIGHT > LEFT while obeying per-motor & 80 kW power limit
	inp_tv.tv_mode = TorqueVectoringScript.TVMode.MODE_DYC
	inp_tv.tv_yaw_moment_req = 600.0
	var base_dyc: Array[float] = DrivetrainModelScript.compute_drive_torques(inp_tv, 18.0, cfg)
	var w_dyc = [18.0 / cfg.wheel_radius, 18.0 / cfg.wheel_radius, 18.0 / cfg.wheel_radius, 18.0 / cfg.wheel_radius]
	var alloc_mid: Array[float] = TorqueVectoringScript.compute_allocation(base_dyc, inp_tv, w_dyc, 18.0, cfg)
	var base_sum: float = base_dyc[0] + base_dyc[1] + base_dyc[2] + base_dyc[3]
	var mid_sum: float = alloc_mid[0] + alloc_mid[1] + alloc_mid[2] + alloc_mid[3]

	# Extreme request saturating motor + 80 kW envelope
	inp_tv.throttle = 1.0
	inp_tv.tv_yaw_moment_req = 5000.0
	var w_fast: float = 35.0 / cfg.wheel_radius
	var w_fast_arr = [w_fast, w_fast, w_fast, w_fast]
	var base_sat: Array[float] = DrivetrainModelScript.compute_drive_torques(inp_tv, 35.0, cfg, w_fast_arr)
	var alloc_sat: Array[float] = TorqueVectoringScript.compute_allocation(base_sat, inp_tv, w_fast_arr, 35.0, cfg)
	var p_sat: float = (alloc_sat[0] + alloc_sat[1] + alloc_sat[2] + alloc_sat[3]) * w_fast
	var t_motor_max_wheel: float = cfg.motor_peak_torque * cfg.gear_ratio * cfg.drivetrain_efficiency
	var p_limit_mech: float = cfg.power_limit_watts * cfg.drivetrain_efficiency
	var bounds_ok: bool = (alloc_sat[1] <= t_motor_max_wheel + 1e-3) and (alloc_sat[3] <= t_motor_max_wheel + 1e-3) \
			and (alloc_sat[0] >= 0.0) and (alloc_sat[2] >= 0.0) and (p_sat <= p_limit_mech + 10.0)

	if alloc_mid[1] > alloc_mid[0] and alloc_mid[3] > alloc_mid[2] \
			and absf(mid_sum - base_sum) < 1e-2 and bounds_ok:
		passed += 1
	else:
		failed += 1
		errors.append("DYC allocation failed: mid=%s (sum %.2f vs %.2f), sat=%s, P=%.1f/%.1f" % [
			str(alloc_mid), mid_sum, base_sum, str(alloc_sat), p_sat, p_limit_mech
		])

	# ---------------------------------------------------------
	# 4. Axle-split Aerodynamics (pitch moment + non-double-counted Fz loading)
	# ---------------------------------------------------------
	var cfg_a_front = VehicleConfigScript.new()
	cfg_a_front.aero_balance_front = 0.60
	var aero_f: Dictionary = AeroModelScript.compute_forces_ros(Vector3(30.0, 0.0, 0.0), cfg_a_front)

	var cfg_a_rear = VehicleConfigScript.new()
	cfg_a_rear.aero_balance_front = 0.30
	var aero_r: Dictionary = AeroModelScript.compute_forces_ros(Vector3(30.0, 0.0, 0.0), cfg_a_rear)

	var exp_pitch_f: float = cfg_a_front.a_cog * float(aero_f["downforce_front"]) - cfg_a_front.b_cog * float(aero_f["downforce_rear"]) + (cfg_a_front.cp_height - cfg_a_front.h_cog) * float(aero_f["drag"].x)
	if absf(float(aero_f["torque"].y) - exp_pitch_f) < 1e-3 and float(aero_f["torque"].y) > 10.0 and float(aero_r["torque"].y) < -10.0:
		passed += 1
	else:
		failed += 1
		errors.append("Aero pitching moment failed: M_f=%.2f (exp %.2f), M_r=%.2f" % [float(aero_f["torque"].y), exp_pitch_f, float(aero_r["torque"].y)])

	# Closed-loop 14-DOF check: front-biased aero transfers vertical load onto front suspension cleanly without double counting
	var m_front_aero = VehicleDynamicsModelScript.new(cfg_a_front)
	var m_rear_aero = VehicleDynamicsModelScript.new(cfg_a_rear)
	m_front_aero.chassis_state.linear_velocity = Vector3(25.0, 0.0, 0.0)
	m_rear_aero.chassis_state.linear_velocity = Vector3(25.0, 0.0, 0.0)
	for i in range(4):
		m_front_aero.wheels[i].omega = 25.0 / cfg_a_front.wheel_radius
		m_rear_aero.wheels[i].omega = 25.0 / cfg_a_rear.wheel_radius
	var inp_coast = VehicleInputsScript.new()
	var res_af: Dictionary = {}
	var res_ar: Dictionary = {}
	for _k in range(120):
		res_af = m_front_aero.step_standalone(inp_coast, 0.0025, 0.0)
		res_ar = m_rear_aero.step_standalone(inp_coast, 0.0025, 0.0)

	var fz_af: Array = res_af["normal_forces"]
	var fz_ar: Array = res_ar["normal_forces"]
	var front_load_af: float = float(fz_af[0]) + float(fz_af[1])
	var front_load_ar: float = float(fz_ar[0]) + float(fz_ar[1])
	var total_af: float = front_load_af + float(fz_af[2]) + float(fz_af[3])
	var aero_25: Dictionary = AeroModelScript.compute_forces_ros(m_front_aero.chassis_state.linear_velocity, cfg_a_front)
	var expected_total_fz: float = cfg_a_front.mass * cfg_a_front.gravity + float(aero_25["downforce_front"]) + float(aero_25["downforce_rear"])

	if front_load_af > front_load_ar + 150.0 and absf(total_af - expected_total_fz) < 35.0:
		passed += 1
	else:
		failed += 1
		errors.append("Aero 14-DOF axle load check failed: front_af=%.1f vs front_ar=%.1f, sum=%.1f (exp %.1f)" % [
			front_load_af, front_load_ar, total_af, expected_total_fz
		])

	return {
		"name": "Block D Subsystems (Powertrain/Brakes/TV/Aero)",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
