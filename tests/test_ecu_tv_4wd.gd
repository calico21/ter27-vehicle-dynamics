extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const EcuTv4WdScript = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoring4Wd.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const Fast3DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/fast_3dof/Fast3DofModel.gd")


static func run() -> Dictionary:
	var st := {"name": "test_ecu_tv_4wd", "passed": 0, "failed": 0, "errors": []}
	var cfg := VehicleConfigScript.new()
	cfg.load_vehicle_configs("ter27_4wd")

	# 1. determinismo estricto: dos ejecuciones identicas dan exactamente los mismos pares
	var e1 := EcuTv4WdScript.new()
	var e2 := EcuTv4WdScript.new()
	e1.configure_from_vehicle_config(cfg)
	e2.configure_from_vehicle_config(cfg)
	var om: Array[float] = [60.0, 62.0, 61.0, 63.0]
	var wst: Array[float] = [0.12, 0.10, 0.0, 0.0]
	var t1: Array[float] = []
	var t2: Array[float] = []
	for _k in range(15):
		t1 = e1.step(2800.0, 0.11, 12.5, 0.75, 3.0, 8.0, 9.81, om, 0.0, 0.005, wst)
		t2 = e2.step(2800.0, 0.11, 12.5, 0.75, 3.0, 8.0, 9.81, om, 0.0, 0.005, wst)
	var det_ok := true
	for i in range(4):
		if not is_equal_approx(t1[i], t2[i]):
			det_ok = false
	if det_ok:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Determinism failed: %s vs %s" % [str(t1), str(t2)])

	# 2. cumplimiento estricto del limite de 80 kw dc a alta velocidad
	var ep := EcuTv4WdScript.new()
	ep.configure_from_vehicle_config(cfg)
	var om_hi: Array[float] = [130.0, 130.0, 130.0, 130.0]  # ~26.4 m/s
	var tp: Array[float] = []
	for _k in range(20):
		tp = ep.step(6000.0, 0.04, 26.4, 0.2, 4.0, 3.0, 9.81, om_hi, 0.0, 0.005, [0.04, 0.04, 0.0, 0.0])
	if ep.last_power_w <= 80000.0 + 5.0 and tp.size() == 4:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("80kW DC ceiling exceeded: %.1f W" % ep.last_power_w)

	# 3. respeto de la elipse de friccion acoplada (fx^2 + fy^2 <= (mu*fz)^2)
	var ee := EcuTv4WdScript.new()
	ee.configure_from_vehicle_config(cfg)
	var te: Array[float] = []
	for _k in range(25):
		te = ee.step(5000.0, 0.22, 16.0, 1.1, 1.0, 14.0, 9.81, [78.0, 80.0, 78.0, 80.0], 0.0, 0.005, [0.22, 0.19, 0.0, 0.0])
	var ell_ok := true
	for i in range(4):
		var fx_i := te[i] / ee.r_eff
		var cap := ee.mu_hat[i] * ee.fz_est[i]
		var comb := sqrt(fx_i * fx_i + ee.fy_est[i] * ee.fy_est[i])
		if comb > cap + 5.0:
			ell_ok = false
	if ell_ok:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Friction ellipse violated")

	# 4. modos de prueba: efficiency (sin pares antagonicos) y tire_warmer (pares opuestos del/tras)
	var em := EcuTv4WdScript.new()
	em.configure_from_vehicle_config(cfg)
	em.active_mode = EcuTv4WdScript.TestMode.EFFICIENCY
	var t_eff := em.step(1200.0, 0.08, 12.0, 0.4, 1.0, 4.0, 9.81, [59.0, 59.0, 59.0, 59.0], 0.0, 0.005, [0.08, 0.08, 0.0, 0.0])
	var no_antag := (t_eff[0] >= -1e-4 and t_eff[1] >= -1e-4 and t_eff[2] >= -1e-4 and t_eff[3] >= -1e-4)
	em.active_mode = EcuTv4WdScript.TestMode.TIRE_WARMER
	var t_warm: Array[float] = []
	for _kw in range(20):
		t_warm = em.step(200.0, 0.0, 10.0, 0.0, 0.0, 0.0, 9.81, [49.0, 49.0, 49.0, 49.0], 0.0, 0.005, [0.0, 0.0, 0.0, 0.0])
	var warm_ok := (t_warm[0] < -50.0 and t_warm[2] > 50.0)
	if no_antag and warm_ok:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Mode checks failed: eff=%s warm=%s" % [str(t_eff), str(t_warm)])

	# 5. guardias de seguridad: curb-strike y limp-home
	var eg := EcuTv4WdScript.new()
	eg.configure_from_vehicle_config(cfg)
	eg.step(2000.0, 0.05, 14.0, 0.2, 2.0, 2.0, 9.81, [69.0, 69.0, 69.0, 69.0], 0.0, 0.005, [0.05, 0.05, 0.0, 0.0])
	# simular golpe en piano (salto az + aceleracion angular brusca)
	eg.step(2000.0, 0.05, 14.0, 0.2, 2.0, 2.0, 22.0, [75.0, 69.0, 69.0, 69.0], 0.0, 0.005, [0.05, 0.05, 0.0, 0.0])
	var curb_ok := eg._curb_timer_s > 0.0
	eg.fault_steer_sensor = true
	eg.step(2000.0, 0.20, 14.0, 0.5, 2.0, 5.0, 9.81, [69.0, 69.0, 69.0, 69.0], 0.0, 0.005, [0.20, 0.20, 0.0, 0.0])
	var limp_ok := eg.limp_home_active and is_zero_approx(eg.last_mz_req)
	if curb_ok and limp_ok:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Safety guards failed: curb=%s limp=%s" % [curb_ok, limp_ok])

	# 6. integracion en full_14dof y fast_3dof con ecu_driving_mode = 2
	var m14 := Full14DofScript.new()
	m14.load_vehicle("ter27_4wd", "default")
	var m3 := Fast3DofScript.new()
	m3.load_vehicle("ter27_4wd", "default")
	var inp := VehicleInputsScript.new()
	inp.drive_mode = m14.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.throttle = 0.75
	inp.steer_angle = deg_to_rad(5.0)
	for _k in range(100):
		m14.step_standalone(inp, 0.0025)
		m3.step_standalone(inp, 0.005)
	if m14.chassis_state.linear_velocity.x > 2.0 and m3.chassis_state.linear_velocity.x > 2.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Model integration failed")

	# 7. pulso longitudinal asimetrico (<40 ms): regen delante (anti-dive 56.9%) vs detras (41.2%)
	var m_f := Full14DofScript.new()
	m_f.load_vehicle("ter27_4wd", "default")
	var m_r := Full14DofScript.new()
	m_r.load_vehicle("ter27_4wd", "default")
	for m_pulse in [m_f, m_r]:
		m_pulse.chassis_state.linear_velocity = Vector3(18.0, 0.0, 0.0)
		for w in m_pulse.wheels:
			w.omega = 18.0 / maxf(m_pulse.config.wheel_radius, 0.1)
	var inp_f := VehicleInputsScript.new()
	inp_f.override_wheel_torques = true
	inp_f.wheel_motor_torques.assign([-120.0, -120.0, 0.0, 0.0])
	var inp_r := VehicleInputsScript.new()
	inp_r.override_wheel_torques = true
	inp_r.wheel_motor_torques.assign([0.0, 0.0, -120.0, -120.0])
	var res_f: Dictionary = {}
	var res_r: Dictionary = {}
	for _p in range(6):  # 6 * 2.5 ms = 15 ms (< 40 ms)
		res_f = m_f.step_standalone(inp_f, 0.0025)
		res_r = m_r.step_standalone(inp_r, 0.0025)
	var fz_f_arr: Array = res_f.get("normal_forces", [0, 0, 0, 0])
	var fz_r_arr: Array = res_r.get("normal_forces", [0, 0, 0, 0])
	var fz_front_f: float = float(fz_f_arr[0]) + float(fz_f_arr[1])
	var fz_front_r: float = float(fz_r_arr[0]) + float(fz_r_arr[1])
	var pitch_f: float = m_f.chassis_state.pitch
	var pitch_r: float = m_r.chassis_state.pitch
	if fz_front_f > fz_front_r + 10.0 and pitch_f < pitch_r:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Anti-dive/anti-squat pulse (<40ms) failed: fz_front_f=%.1f fz_front_r=%.1f pitch_f=%.5f pitch_r=%.5f" % [fz_front_f, fz_front_r, pitch_f, pitch_r])

	# 8. limp-home automatico ante fallo de 1 sensor de rueda (rl=0 rad/s en curva a 15 m/s)
	var m_lh := Full14DofScript.new()
	m_lh.load_vehicle("ter27_4wd", "default")
	m_lh.chassis_state.linear_velocity = Vector3(15.0, 0.0, 0.0)
	for w in m_lh.wheels:
		w.omega = 15.0 / maxf(m_lh.config.wheel_radius, 0.1)
	var inp_lh := VehicleInputsScript.new()
	inp_lh.drive_mode = m_lh.config.drive_mode
	inp_lh.override_wheel_torques = true
	inp_lh.throttle = 0.35
	inp_lh.steer_angle = deg_to_rad(5.5)
	for k_lh in range(160):
		var w_sens: Array[float] = [m_lh.wheels[0].omega, m_lh.wheels[1].omega, m_lh.wheels[2].omega, m_lh.wheels[3].omega]
		if k_lh >= 40:
			w_sens[2] = 0.0  # fallo de sensor rl a 0 (rueda fisica sigue girando)
		var u_x := m_lh.chassis_state.linear_velocity.x
		var wz_c := m_lh.chassis_state.angular_velocity.z
		var ax_c := m_lh.net_force_chassis_ros.x / m_lh.config.mass
		var ay_c := m_lh.net_force_chassis_ros.y / m_lh.config.mass
		var tq_lh := m_lh.ecu_tv_4wd.step(1200.0, inp_lh.steer_angle, u_x, wz_c, ax_c, ay_c, m_lh.config.gravity, w_sens, 0.0, 0.005, [inp_lh.steer_angle, inp_lh.steer_angle, 0.0, 0.0])
		inp_lh.wheel_motor_torques.assign(tq_lh)
		m_lh.step_standalone(inp_lh, 0.005)
	var beta_lh := absf(rad_to_deg(atan2(m_lh.chassis_state.linear_velocity.y, maxf(m_lh.chassis_state.linear_velocity.x, 1.0))))
	var ekf_err_lh := absf(m_lh.ecu_tv_4wd.ekf_vx - m_lh.chassis_state.linear_velocity.x)
	if m_lh.ecu_tv_4wd.limp_home_active and ekf_err_lh < 2.0 and beta_lh < 12.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Single-wheel Limp-Home failed: active=%s ekf_err=%.2f beta=%.2f" % [m_lh.ecu_tv_4wd.limp_home_active, ekf_err_lh, beta_lh])

	# 9. jerarquia mz > fx en horquilla (7 m/s, 16 deg, gas 100%): coherencia de signo 100%
	var m_hq := Full14DofScript.new()
	m_hq.load_vehicle("ter27_4wd", "default")
	m_hq.chassis_state.linear_velocity = Vector3(7.0, 0.0, 0.0)
	for w in m_hq.wheels:
		w.omega = 7.0 / maxf(m_hq.config.wheel_radius, 0.1)
	var inp_hq := VehicleInputsScript.new()
	inp_hq.drive_mode = m_hq.config.drive_mode
	inp_hq.ecu_driving_mode = 2
	inp_hq.tv_4wd_mode = EcuTv4WdScript.TestMode.AUTOCROSS
	inp_hq.steer_angle = deg_to_rad(16.0)
	inp_hq.throttle = 1.0
	var hq_ok := true
	var hq_cnt := 0
	for _k in range(40):
		m_hq.step_standalone(inp_hq, 0.005)
		var mz_req_hq: float = m_hq.ecu_tv_4wd.last_mz_req
		if absf(mz_req_hq) > 25.0:
			var mz_ach_hq := 0.0
			for i in range(4):
				mz_ach_hq += m_hq.ecu_tv_4wd._qp_bmz[i] * m_hq.ecu_tv_4wd.last_t_cmd[i]
			hq_cnt += 1
			if signf(mz_ach_hq) != signf(mz_req_hq):
				hq_ok = false
	if hq_ok and hq_cnt > 0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Hierarchy Mz>Fx failed in hairpin: ok=%s samples=%d" % [hq_ok, hq_cnt])

	# 10. cero ringing en mz_ff ante escalon de 8 deg a 15 m/s
	var e_ff := EcuTv4WdScript.new()
	e_ff.configure_from_vehicle_config(cfg)
	var om_15: Array[float] = [15.0 / e_ff.r_eff, 15.0 / e_ff.r_eff, 15.0 / e_ff.r_eff, 15.0 / e_ff.r_eff]
	for _k in range(10):
		e_ff.step(400.0, 0.0, 15.0, 0.0, 0.0, 0.0, 9.81, om_15, 0.0, 0.005, [0.0, 0.0, 0.0, 0.0])
	var min_mz_ff := 1e9
	var max_mz_ff := -1e9
	var st_8 := deg_to_rad(8.0)
	for _k in range(60):
		e_ff.step(400.0, st_8, 15.0, 0.0, 0.0, 0.0, 9.81, om_15, 0.0, 0.005, [st_8, st_8, 0.0, 0.0])
		min_mz_ff = minf(min_mz_ff, e_ff.last_mz_ff)
		max_mz_ff = maxf(max_mz_ff, e_ff.last_mz_ff)
	if min_mz_ff >= -1.0 and max_mz_ff > 20.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Mz_ff ringing check failed: min=%.2f max=%.2f" % [min_mz_ff, max_mz_ff])

	# 11. deteccion y aislamiento automatico de rueda fallida (rl=0) en <= 2 ciclos (10 ms)
	var e_flt := EcuTv4WdScript.new()
	e_flt.configure_from_vehicle_config(cfg)
	var st_4 := deg_to_rad(4.0)
	for _k in range(20):
		e_flt.step(800.0, st_4, 15.0, 0.5, 0.0, 7.5, 9.81, om_15, 0.0, 0.005, [st_4, st_4, 0.0, 0.0])
	var om_bad: Array[float] = [om_15[0], om_15[1], 0.0, om_15[3]]
	for _k in range(2):
		e_flt.step(800.0, st_4, 15.0, 0.5, 0.0, 7.5, 9.81, om_bad, 0.0, 0.005, [st_4, st_4, 0.0, 0.0])
	if e_flt._fault_auto[2] and e_flt.limp_home_active and is_zero_approx(e_flt.last_mz_req) and e_flt._w_eff[2] > 50.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Fast fault isolation (<=10ms) failed: auto=%s lh=%s mz=%.2f w_eff=%.2f" % [
			e_flt._fault_auto[2], e_flt.limp_home_active, e_flt.last_mz_req, e_flt._w_eff[2]
		])

	# 12. pedalera 35/65 regen-first (sin hidraulico <=35%, techo carga -25 kw, hidraulico >35%)
	var e_p35 := EcuTv4WdScript.new()
	e_p35.configure_from_vehicle_config(cfg)
	var om_20: Array[float] = [20.0 / e_p35.r_eff, 20.0 / e_p35.r_eff, 20.0 / e_p35.r_eff, 20.0 / e_p35.r_eff]
	for _k in range(10):
		e_p35.step(-2500.0, 0.0, 20.0, 0.0, -4.0, 0.0, 9.81, om_20, 0.25, 0.005, [0.0, 0.0, 0.0, 0.0])
	var no_hyd_25 := is_zero_approx(e_p35.t_hyd_est[0]) and is_zero_approx(e_p35.t_hyd_est[2])
	var chg_ok := e_p35.last_power_w >= -(e_p35.p_max_charge_w + 50.0)
	e_p35.step(-2500.0, 0.0, 20.0, 0.0, -8.0, 0.0, 9.81, om_20, 0.65, 0.005, [0.0, 0.0, 0.0, 0.0])
	var hyd_65 := e_p35.t_hyd_est[0] > 50.0
	if no_hyd_25 and chg_ok and hyd_65:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("35/65 regen-first check failed: no_hyd_25=%s chg_ok=%s (%.1fW) hyd_65=%s" % [no_hyd_25, chg_ok, e_p35.last_power_w, hyd_65])

	# 13. thermal derating predictivo en qp: motor caliente (rl=100 c) cede par al eje frio
	var e_th := EcuTv4WdScript.new()
	e_th.configure_from_vehicle_config(cfg)
	e_th.set_motor_temperatures(45.0, 45.0, 100.0, 45.0)
	var t_th: Array[float] = []
	for _k in range(15):
		t_th = e_th.step(2200.0, 0.0, 12.0, 0.0, 4.0, 0.0, 9.81, [59.0, 59.0, 59.0, 59.0], 0.0, 0.005, [0.0, 0.0, 0.0, 0.0])
	if t_th[2] < t_th[3] - 10.0 and t_th[0] > t_th[2]:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Thermal derating QP shift failed: %s" % str(t_th))

	return st
