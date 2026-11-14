extends RefCounted

const VehicleConfig = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModel = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const EcuTV3 = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoringV3.gd")
const TvSilBridge = preload("res://addons/rclgd/tv_sil_bridge.gd")

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
	m_vec.chassis_state.angular_velocity = Vector3(0.0, 0.0, 0.0)
	for i in range(4):
		m_vec.wheels[i].omega = v_turn / cfg.wheel_radius
	var in_vec := VehicleInputs.new()
	in_vec.set_from_manual(0.55, 0.0, steer_turn, cfg.max_steer_angle)
	in_vec.ecu_driving_mode = 1
	var last_ecu_trqs: Array = [0.0, 0.0, 0.0, 0.0]
	for _step in range(22):
		var res_vec := m_vec.step_standalone(in_vec, 0.005)
		last_ecu_trqs = res_vec["forces"]["drive_torques"]
	var vectorized: bool = (last_ecu_trqs[1] > last_ecu_trqs[0] + 0.2) and (last_ecu_trqs[3] > last_ecu_trqs[2] + 0.2)
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

	# 8. Accel -> hard brake -> accel: mu_surface must not degrade and second launch equals first
	var m_ab := VehicleDynamicsModel.new(cfg)
	var in_go := VehicleInputs.new()
	in_go.set_from_manual(1.0, 0.0, 0.0, cfg.max_steer_angle)
	in_go.ecu_driving_mode = 1
	var in_stop := VehicleInputs.new()
	in_stop.set_from_manual(0.0, 1.0, 0.0, cfg.max_steer_angle)
	in_stop.ecu_driving_mode = 1
	var launch_times: Array[float] = []
	var min_mu: float = 99.0
	for phase in range(2):
		var t_launch: float = -1.0
		var tt: float = 0.0
		for _s2 in range(1600): # up to 8 s
			m_ab.step_standalone(in_go, 0.005)
			tt += 0.005
			for mu_i in m_ab.ecu_tv_v3.tc_mu_surface:
				min_mu = minf(min_mu, mu_i)
			if m_ab.chassis_state.linear_velocity.x >= 20.0:
				t_launch = tt
				break
		launch_times.append(t_launch)
		for _s3 in range(1600): # brake to standstill
			m_ab.step_standalone(in_stop, 0.005)
			for mu_i in m_ab.ecu_tv_v3.tc_mu_surface:
				min_mu = minf(min_mu, mu_i)
			if m_ab.chassis_state.linear_velocity.x < 0.05:
				break
	var rel_diff: float = absf(launch_times[1] - launch_times[0]) / maxf(launch_times[0], 1e-3)
	if launch_times[0] > 0.0 and launch_times[1] > 0.0 and min_mu >= 1.60 and rel_diff < 0.03:
		passed += 1
	else:
		failed += 1
		errors.append("Accel-brake-accel degradation: t1=%.3f s t2=%.3f s diff=%.2f%% min_mu=%.3f" % [launch_times[0], launch_times[1], rel_diff * 100.0, min_mu])

	# 9. 4 live TV maps + VehicleInputs default tv_map_id + cycle_map() + custom params
	var vi_def := VehicleInputs.new()
	var ecu_maps := EcuTV3.new()
	var default_ok: bool = (
		vi_def.tv_map_id == 2
		and ecu_maps.active_map_id == EcuTV3.TVMap.AUTOCROSS_NEUTRAL
		and ecu_maps.get_map_name() == "AUTOCROSS_NEUTRAL"
		and is_equal_approx(ecu_maps.k_us_active, 0.0018)
		and is_equal_approx(ecu_maps.tv_max_mz_active, 750.0)
		and is_equal_approx(ecu_maps.tc_kp_active, 22.5)
		and is_equal_approx(ecu_maps.min_fx_reserve_active, 0.06)
	)
	ecu_maps.set_map(EcuTV3.TVMap.OPEN_4WD)
	var m0_ok: bool = (
		ecu_maps.active_map_id == 0
		and ecu_maps.get_map_name() == "OPEN_4WD"
		and is_equal_approx(ecu_maps.front_split_active, 0.35)
		and is_equal_approx(ecu_maps.rear_split_active, 0.65)
		and is_equal_approx(ecu_maps.tv_max_mz_active, 0.0)
	)
	ecu_maps.set_map(EcuTV3.TVMap.ENDURANCE_SAFE)
	var m1_ok: bool = (
		ecu_maps.active_map_id == EcuTV3.TVMap.ENDURANCE_SAFE
		and is_equal_approx(ecu_maps.k_us_active, 0.0030)
		and is_equal_approx(ecu_maps.tv_max_mz_active, 550.0)
		and is_equal_approx(ecu_maps.tc_kp_active, 28.0)
		and is_equal_approx(ecu_maps.min_fx_reserve_active, 0.08)
	)
	ecu_maps.set_map(EcuTV3.TVMap.SKIDPAD_ATTACK)
	var m3_ok: bool = (
		ecu_maps.active_map_id == EcuTV3.TVMap.SKIDPAD_ATTACK
		and is_equal_approx(ecu_maps.k_us_active, 0.0008)
		and is_equal_approx(ecu_maps.tv_max_mz_active, 850.0)
		and is_equal_approx(ecu_maps.tc_kp_active, 20.0)
		and is_equal_approx(ecu_maps.min_fx_reserve_active, 0.05)
	)
	ecu_maps.set_custom_parameters(0.019, 600.0, 22.5, 0.08)
	var custom_ok: bool = (
		ecu_maps.get_map_name() == "CUSTOM"
		and is_equal_approx(ecu_maps.k_us_active, 0.019)
		and is_equal_approx(ecu_maps.tv_max_mz_active, 600.0)
		and is_equal_approx(ecu_maps.tc_kp_active, 22.5)
		and is_equal_approx(ecu_maps.min_fx_reserve_active, 0.08)
	)
	if default_ok and m0_ok and m1_ok and m3_ok and custom_ok:
		passed += 1
	else:
		failed += 1
		errors.append("4-map TV profile check failed: def=%s m0=%s m1=%s m3=%s custom=%s" % [default_ok, m0_ok, m1_ok, m3_ok, custom_ok])

	# 10. Map 0 (OPEN_4WD): 35/65 split, Mz == 0, 72 kW cap, 22000 RPM motor overspeed cutoff
	var ecu_open := EcuTV3.new()
	ecu_open.set_map(EcuTV3.TVMap.OPEN_4WD)
	var omega_open: Array[float] = [50.0, 50.0, 50.0, 50.0]
	var t_open: Array[float] = ecu_open.step(2000.0, 0.12, 10.0, 0.0, 0.25, 2.0, 1.0, omega_open, 0.0, 0.005)
	var open_sum: float = t_open[0] + t_open[1] + t_open[2] + t_open[3]
	var open_front_ratio: float = (t_open[0] + t_open[1]) / maxf(open_sum, 1e-6)
	var split_ok: bool = (
		is_zero_approx(ecu_open.last_mz_req)
		and is_equal_approx(t_open[0], t_open[1])
		and is_equal_approx(t_open[2], t_open[3])
		and absf(open_front_ratio - 0.35) < 1e-4
		and open_sum > 100.0
	)
	var ecu_pwr := EcuTV3.new()
	ecu_pwr.set_map(EcuTV3.TVMap.OPEN_4WD)
	var omega_pwr: Array[float] = [100.0, 100.0, 100.0, 100.0]
	var t_pwr: Array[float] = ecu_pwr.step(12000.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, omega_pwr, 0.0, 0.005)
	var p_total: float = (t_pwr[0] + t_pwr[1] + t_pwr[2] + t_pwr[3]) * 100.0
	var ecu_rpm := EcuTV3.new()
	ecu_rpm.set_map(EcuTV3.TVMap.OPEN_4WD)
	var omega_overspeed: Array[float] = [200.0, 200.0, 200.0, 200.0] # 200 * 12.5 * 60 / (2*PI) = 23873 RPM > 22000
	var t_overspeed: Array[float] = ecu_rpm.step(3000.0, 0.0, 40.64, 0.0, 0.0, 0.0, 0.0, omega_overspeed, 0.0, 0.005)
	var overspeed_zero: bool = (
		t_overspeed[0] <= 1e-4 and t_overspeed[1] <= 1e-4
		and t_overspeed[2] <= 1e-4 and t_overspeed[3] <= 1e-4
	)
	if split_ok and p_total <= EcuTV3.GP_P_MAX_TOTAL_MECH + 1.0 and overspeed_zero:
		passed += 1
	else:
		failed += 1
		errors.append("OPEN_4WD bounds failed: split_ok=%s ratio=%.3f p_total=%.1f overspeed=%s" % [split_ok, open_front_ratio, p_total, t_overspeed])

	# 11. CAN latency ring buffer (12.0 ms) & 4096 PPR encoder quantization
	var ecu_can := EcuTV3.new()
	ecu_can.set_map(EcuTV3.TVMap.OPEN_4WD)
	ecu_can.set_hardware_realism(true, 12.0, 4096)
	var omega_can: Array[float] = [50.0, 50.0, 50.0, 50.0]
	var cmd_5ms: Array[float] = ecu_can.step(1500.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, omega_can, 0.0, 0.005)
	var cmd_10ms: Array[float] = ecu_can.step(1500.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, omega_can, 0.0, 0.005)
	var cmd_15ms: Array[float] = ecu_can.step(1500.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, omega_can, 0.0, 0.005)
	var cmd_20ms: Array[float] = ecu_can.step(1500.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, omega_can, 0.0, 0.005)
	var delay_ok: bool = (
		cmd_5ms[2] < 1e-4
		and cmd_10ms[2] < 1e-4
		and cmd_15ms[2] < 1e-4
		and cmd_20ms[2] > 50.0
	)
	var quantum_rad_s: float = (2.0 * PI) / (4096.0 * 4.0 * 0.005)
	var q_a: float = EcuTV3.quantize_wheel_omega(50.001, 0.005, 4096)
	var q_b: float = EcuTV3.quantize_wheel_omega(50.002, 0.005, 4096)
	var q_hi: float = EcuTV3.quantize_wheel_omega(50.10, 0.005, 4096)
	var quant_ok: bool = (
		is_equal_approx(q_a, q_b)
		and q_hi > q_a
		and absf(q_a - 50.001) <= 0.5 * quantum_rad_s + 1e-5
	)
	if delay_ok and quant_ok:
		passed += 1
	else:
		failed += 1
		errors.append("Hardware realism CAN/Encoder check failed: cmds=[%.2f, %.2f, %.2f, %.2f] q=[%.4f, %.4f, %.4f]" % [cmd_5ms[2], cmd_10ms[2], cmd_15ms[2], cmd_20ms[2], q_a, q_b, q_hi])

	# 12. TvSilBridge status & safe GDScript fallback when native GDExtension is absent
	var ecu_sil := EcuTV3.new()
	var class_present: bool = ClassDB.class_exists("GpTorqueVectoringSIL")
	var switched_native: bool = ecu_sil.set_use_native_sil(true)
	var sil_status: Dictionary = ecu_sil.get_sil_status()
	var sil_ok: bool = (
		sil_status.has("backend_name")
		and sil_status.has("native_available")
		and sil_status.has("message")
		and not String(sil_status["message"]).is_empty()
		and (class_present or (not switched_native and not ecu_sil.use_native_sil and sil_status["backend_name"] == "GDSCRIPT" and not TvSilBridge.is_native_sil_available()))
	)
	if sil_ok:
		passed += 1
	else:
		failed += 1
		errors.append("TvSilBridge fallback invalid: %s" % [sil_status])

	# 13. Live DiL Tuning: set_live_param() preserves EKF state, changes torque output, and throttle_mz_cut attenuates corner-exit Mz
	var ecu_live := EcuTV3.new()
	ecu_live.set_map(EcuTV3.TVMap.AUTOCROSS_NEUTRAL)
	var omega_live: Array[float] = [60.0, 60.0, 60.0, 60.0]
	for _i in range(20):
		ecu_live.step(2600.0, 0.14, 12.2, 0.4, 0.45, 7.0, 2.0, omega_live, 0.0, 0.005)
	var ekf_vy_before: float = ecu_live.ekf_x[0]
	ecu_live.set_live_param("live_throttle_mz_cut", 0.0)
	ecu_live.set_live_param("live_max_mz", 1000.0)
	var trq_no_cut: Array[float] = ecu_live.step(2600.0, 0.14, 12.2, 0.4, 0.45, 7.0, 2.0, omega_live, 0.0, 0.005)
	var mz_no_cut: float = ecu_live.last_mz_req
	var ekf_vy_after: float = ecu_live.ekf_x[0]
	ecu_live.set_live_param("live_throttle_mz_cut", 0.80)
	var trq_with_cut: Array[float] = ecu_live.step(2600.0, 0.14, 12.2, 0.4, 0.45, 7.0, 2.0, omega_live, 0.0, 0.005)
	var mz_with_cut: float = ecu_live.last_mz_req
	var c_hdr: String = ecu_live.export_c_header_snippet()
	var live_ok: bool = (
		absf(ekf_vy_before) > 1e-4
		and absf(ekf_vy_after) > 1e-4
		and mz_no_cut > 0.0
		and mz_with_cut < mz_no_cut
		and absf((trq_with_cut[3] - trq_with_cut[2]) - (trq_no_cut[3] - trq_no_cut[2])) > 0.1
		and c_hdr.contains("#define GP_THROTTLE_MZ_CUT")
	)
	if live_ok:
		passed += 1
	else:
		failed += 1
		errors.append("Live DiL tuning check failed: vy=[%.4f, %.4f] mz=[%.1f -> %.1f]" % [ekf_vy_before, ekf_vy_after, mz_no_cut, mz_with_cut])

	# 14. Dynamic JSON TV map listing, loading, saving, cycling & deletion protection
	var ecu_json := EcuTV3.new()
	var maps_list := ecu_json.list_available_tv_maps()
	var has_five := maps_list.size() >= 5
	var loaded_skid := ecu_json.load_tv_map_from_file("res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/03_skidpad_attack.json")
	var skid_ok := loaded_skid and is_equal_approx(ecu_json.live_max_mz, 850.0) and ecu_json.active_map_filename == "03_skidpad_attack"
	ecu_json.set_live_param("live_k_us", 0.0133)
	var saved_path := ecu_json.save_current_tv_map_to_file("Unit Test Custom Map", "Temp test")
	var ecu_json2 := EcuTV3.new()
	var reloaded_custom := saved_path != "" and ecu_json2.load_tv_map_from_file(saved_path) and is_equal_approx(ecu_json2.live_k_us, 0.0133)
	var protected_ok := not ecu_json.delete_tv_map_file("res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/00_open_4wd.json")
	var deleted_ok := saved_path != "" and ecu_json.delete_tv_map_file(saved_path)
	var cyc0 := ecu_json.cycle_map()
	var cyc1 := ecu_json.cycle_map()
	if has_five and skid_ok and reloaded_custom and protected_ok and deleted_ok and cyc0 != cyc1:
		passed += 1
	else:
		failed += 1
		errors.append("JSON TV map manager failed: count=%d skid=%s reload=%s prot=%s del=%s cyc=[%d,%d]" % [
			maps_list.size(), str(skid_ok), str(reloaded_custom), str(protected_ok), str(deleted_ok), cyc0, cyc1
		])

	# 15. 8-Stage X-Ray Telemetry completeness & NaN-free updates
	var ecu_xray := EcuTV3.new()
	ecu_xray.set_map(EcuTV3.TVMap.AUTOCROSS_NEUTRAL)
	var w_xray: Array[float] = [60.0, 60.0, 60.0, 60.0]
	for _s in range(10):
		ecu_xray.step(2000.0, 0.16, 14.0, 0.2, 0.65, 9.0, 2.5, w_xray, 0.0, 0.005)
	var xray_finite: bool = (
		is_finite(ecu_xray.last_vy_ss)
		and is_finite(ecu_xray.last_ff_mz)
		and is_finite(ecu_xray.last_p_mz)
		and is_finite(ecu_xray.last_i_mz)
		and is_finite(ecu_xray.last_beta_mz)
		and is_finite(ecu_xray.last_mz_raw)
		and is_finite(ecu_xray.last_mz_req)
		and ecu_xray.last_kp > 0.0
		and ecu_xray.last_fz_est.size() == 4
		and ecu_xray.last_fy_est.size() == 4
		and ecu_xray.last_t_nominal.size() == 4
		and ecu_xray.last_t_qp.size() == 4
		and ecu_xray.last_tc_cut_nm.size() == 4
		and ecu_xray.last_fz_est[0] > 100.0
	)
	if xray_finite and absf(ecu_xray.last_mz_raw) > 1.0:
		passed += 1
	else:
		failed += 1
		errors.append("8-stage X-Ray telemetry failed: finite=%s kp=%.1f mz_raw=%.1f fz0=%.1f" % [
			str(xray_finite), ecu_xray.last_kp, ecu_xray.last_mz_raw, ecu_xray.last_fz_est[0]
		])

	return {
		"name": "test_ecu_torque_vectoring_v3",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
