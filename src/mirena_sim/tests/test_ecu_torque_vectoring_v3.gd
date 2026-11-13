extends RefCounted

const VehicleConfig = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModel = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const EcuTV3 = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoringV3.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []

	# 1. gp_math parity
	var sp_0: float = EcuTV3.gp_softplus(0.0)
	var sp_hi: float = EcuTV3.gp_softplus(25.0)
	var sig_0: float = EcuTV3.gp_sigmoid(0.0)
	var kp_mid: float = EcuTV3.gp_bilinear_interp_4x4(EcuTV3.KP_MAP, 0.5, 0.5)
	if absf(sp_0 - log(2.0)) < 1e-5 and is_equal_approx(sp_hi, 25.0) and is_equal_approx(sig_0, 0.5) and kp_mid > 200.0 and kp_mid < 300.0:
		passed += 1
	else:
		failed += 1
		errors.append("gp_math parity failed: sp0=%.4f sp25=%.2f sig0=%.2f kp_mid=%.2f" % [sp_0, sp_hi, sig_0, kp_mid])

	# 2. gp_ekf 2-state [vy, bw] predict + kinematic_ss + gps update
	var ecu := EcuTV3.new()
	for _i in range(20):
		ecu.gp_ekf_predict(0.08, 0.0, 6.0, 0.5, 12.0, 0.005)
		ecu.gp_ekf_update_kinematic_ss(0.0, 6.0, 0.5, 12.0)
	if is_finite(ecu.ekf_x[0]) and absf(ecu.ekf_beta_est) <= 0.523 and ecu.ekf_vy_std > 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("gp_ekf failed to produce valid bounded state")

	# 3. 4WD closed-form KKT solver: sum(T_i) == T_target and redistribution under saturation
	var t_warm: Array[float] = [120.0, 180.0, 130.0, 170.0]
	var t_prev: Array[float] = [100.0, 100.0, 100.0, 100.0]
	var t_lb: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var t_ub: Array[float] = [20.0, 300.0, 300.0, 300.0] # FL heavily unloaded/saturated
	var fx_req: float = 500.0 / EcuTV3.GP_R_WHEEL
	var qp_res: Dictionary = EcuTV3.gp_qp_solve_4wd_closedform(t_warm, t_prev, fx_req, t_lb, t_ub)
	var t_sol: Array[float] = qp_res["torques"]
	var t_sum: float = t_sol[0] + t_sol[1] + t_sol[2] + t_sol[3]
	var within_bounds: bool = true
	for i in range(4):
		if t_sol[i] < t_lb[i] - 1e-4 or t_sol[i] > t_ub[i] + 1e-4:
			within_bounds = false
	if absf(t_sum - 500.0) < 1e-3 and is_equal_approx(t_sol[0], 20.0) and within_bounds and t_sol[1] > 120.0:
		passed += 1
	else:
		failed += 1
		errors.append("4WD KKT solver failed redistribution: t_sol=%s sum=%.3f" % [t_sol, t_sum])

	# 4. 4WD TC RLS activation under excessive slip on all 4 wheels
	var ecu_tc := EcuTV3.new()
	var vx_tc: float = 10.0
	var omega_slip: Array[float] = [
		(vx_tc * 1.45) / EcuTV3.GP_R_WHEEL,
		(vx_tc * 1.45) / EcuTV3.GP_R_WHEEL,
		(vx_tc * 1.45) / EcuTV3.GP_R_WHEEL,
		(vx_tc * 1.45) / EcuTV3.GP_R_WHEEL
	]
	var fz_tc: Array[float] = [700.0, 700.0, 750.0, 750.0]
	var t_tc: Array[float] = [200.0, 200.0, 200.0, 200.0]
	for _k in range(20):
		t_tc = [200.0, 200.0, 200.0, 200.0]
		ecu_tc.gp_tc_step_4wd(t_tc, omega_slip, vx_tc, 0.0, 0.0, fz_tc, 0.005)
	var all_cut: bool = t_tc[0] < 195.0 and t_tc[1] < 195.0 and t_tc[2] < 195.0 and t_tc[3] < 195.0
	if all_cut:
		passed += 1
	else:
		failed += 1
		errors.append("4WD TC RLS did not cut torque on all 4 spinning wheels: %s" % [t_tc])

	# 5. Tight cornering (R = 9.125 m) under throttle:
	# 5a. Verify outer > inner torque vectoring during turn-in
	var cfg := VehicleConfig.new()
	var m_vec := VehicleDynamicsModel.new(cfg)
	var v_turn := 10.5
	var steer_turn := 0.16 # ~9.2 deg left turn
	m_vec.chassis_state.linear_velocity = Vector3(v_turn, 0.0, 0.0)
	m_vec.chassis_state.angular_velocity = Vector3(0.0, 0.0, 0.8)
	for i in range(4):
		m_vec.wheels[i].omega = v_turn / cfg.wheel_radius
	var in_vec := VehicleInputs.new()
	in_vec.set_from_manual(0.75, 0.0, steer_turn, cfg.max_steer_angle)
	in_vec.ecu_driving_mode = 1
	var last_ecu_trqs: Array = [0.0, 0.0, 0.0, 0.0]
	for _step in range(80):
		var res_vec := m_vec.step_standalone(in_vec, 0.005)
		last_ecu_trqs = res_vec["forces"]["drive_torques"]
	var vectorized: bool = (last_ecu_trqs[1] > last_ecu_trqs[0] + 1.0) and (last_ecu_trqs[3] > last_ecu_trqs[2] + 1.0)
	if vectorized:
		passed += 1
	else:
		failed += 1
		errors.append("ECU_TV_V3_4WD failed outer>inner vectoring: trqs=%s" % [last_ecu_trqs])

	# 5b. Verify oversteer / sideslip reduction vs LINEAL_4WD in R = 9.125 m corner with initial slip
	var m_lin := VehicleDynamicsModel.new(cfg)
	var m_ecu := VehicleDynamicsModel.new(cfg)
	for m in [m_lin, m_ecu]:
		m.chassis_state.linear_velocity = Vector3(11.0, -0.6, 0.0)
		m.chassis_state.angular_velocity = Vector3(0.0, 0.0, 1.55)
		for i in range(4):
			m.wheels[i].omega = 11.0 / cfg.wheel_radius

	var in_lin := VehicleInputs.new()
	in_lin.set_from_manual(0.75, 0.0, 0.15, cfg.max_steer_angle)
	in_lin.ecu_driving_mode = 0

	var in_ecu := VehicleInputs.new()
	in_ecu.set_from_manual(0.75, 0.0, 0.15, cfg.max_steer_angle)
	in_ecu.ecu_driving_mode = 1

	var peak_beta_lin := 0.0
	var peak_beta_ecu := 0.0
	for _step in range(160): # 0.8 s @ 200 Hz
		m_lin.step_standalone(in_lin, 0.005)
		m_ecu.step_standalone(in_ecu, 0.005)
		var beta_lin := absf(atan2(m_lin.chassis_state.linear_velocity.y, maxf(m_lin.chassis_state.linear_velocity.x, 1.0)))
		var beta_ecu := absf(atan2(m_ecu.chassis_state.linear_velocity.y, maxf(m_ecu.chassis_state.linear_velocity.x, 1.0)))
		peak_beta_lin = maxf(peak_beta_lin, beta_lin)
		peak_beta_ecu = maxf(peak_beta_ecu, beta_ecu)

	if peak_beta_ecu < peak_beta_lin:
		passed += 1
	else:
		failed += 1
		errors.append("ECU_TV_V3_4WD did not reduce sideslip vs LINEAL_4WD: beta_lin=%.4f beta_ecu=%.4f" % [peak_beta_lin, peak_beta_ecu])

	# 6. 80 kW DC / 72 kW mech global power budget with >18 kW per rear wheel under acceleration
	var m_pwr := VehicleDynamicsModel.new(cfg)
	var v_pwr := 20.0
	m_pwr.chassis_state.linear_velocity = Vector3(v_pwr, 0.0, 0.0)
	for i in range(4):
		m_pwr.wheels[i].omega = v_pwr / cfg.wheel_radius
	var in_pwr := VehicleInputs.new()
	in_pwr.set_from_manual(1.0, 0.0, 0.0, cfg.max_steer_angle)
	in_pwr.ecu_driving_mode = 1
	var pwr_trqs: Array = [0.0, 0.0, 0.0, 0.0]
	for _s in range(40):
		var r_pwr := m_pwr.step_standalone(in_pwr, 0.005)
		pwr_trqs = r_pwr["forces"]["drive_torques"]
	var p_rear_wheel: float = pwr_trqs[2] * absf(m_pwr.wheels[2].omega)
	var p_total_mech: float = 0.0
	for i in range(4):
		p_total_mech += pwr_trqs[i] * absf(m_pwr.wheels[i].omega)
	if p_rear_wheel > 18000.0 and p_total_mech > 68000.0 and p_total_mech <= 72500.0:
		passed += 1
	else:
		failed += 1
		errors.append("4WD power budget check failed: P_rl=%.1f W P_total_mech=%.1f W" % [p_rear_wheel, p_total_mech])

	# 7. Straight-line 0-75 m acceleration & 80 kW aerodynamic top speed equilibrium (ECU_TV_V3_4WD)
	var m_acc := VehicleDynamicsModel.new(cfg)
	var in_acc := VehicleInputs.new()
	in_acc.set_from_manual(1.0, 0.0, 0.0, cfg.max_steer_angle)
	in_acc.ecu_driving_mode = 1
	var dist_x: float = 0.0
	var t_75m: float = -1.0
	var t_sim: float = 0.0
	var v_max: float = 0.0
	for _step in range(3000): # 15.0 s @ 200 Hz
		m_acc.step_standalone(in_acc, 0.005)
		t_sim += 0.005
		var vx_now: float = m_acc.chassis_state.linear_velocity.x
		dist_x += vx_now * 0.005
		v_max = maxf(v_max, vx_now)
		if t_75m < 0.0 and dist_x >= 75.0:
			t_75m = t_sim
	if t_75m > 0.0 and t_75m < 3.95 and v_max >= 30.0 and v_max <= 38.0:
		passed += 1
	else:
		failed += 1
		errors.append("ECU_TV_V3_4WD 0-75m / top speed failed: t_75m=%.2f s v_max=%.2f m/s (%.1f km/h)" % [t_75m, v_max, v_max * 3.6])

	return {
		"name": "test_ecu_torque_vectoring_v3",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
