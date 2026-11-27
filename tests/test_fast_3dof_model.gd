extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const Fast3DofVehicleModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/fast_3dof/Fast3DofModel.gd")


static func run() -> Dictionary:
	var st := {"name": "test_fast_3dof_model", "passed": 0, "failed": 0, "errors": []}

	# 1. conservación de carga vertical total: sum(fz_i) == m*g + f_down
	var m_fast := Fast3DofVehicleModelScript.new()
	m_fast.load_vehicle("ter26_rwd", "default")
	var mg: float = m_fast.config.mass * m_fast.config.gravity
	var fz_static: Array[float] = m_fast.compute_normal_forces_quasistatic(0.0, 0.0, 0.0)
	var sum_stat: float = fz_static[0] + fz_static[1] + fz_static[2] + fz_static[3]
	if absf(sum_stat - mg) < 1e-2:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Static Fz sum mismatch: %.3f vs %.3f" % [sum_stat, mg])

	var vx_test: float = 20.0
	var f_down_exp: float = 0.5 * m_fast.config.air_density * m_fast.config.cl_a * vx_test * vx_test
	var fz_dyn: Array[float] = m_fast.compute_normal_forces_quasistatic(3.0, 8.0, vx_test)
	var sum_dyn: float = fz_dyn[0] + fz_dyn[1] + fz_dyn[2] + fz_dyn[3]
	if absf(sum_dyn - (mg + f_down_exp)) < 1e-1:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Dynamic Fz sum mismatch: %.3f vs %.3f" % [sum_dyn, mg + f_down_exp])

	# 2. paridad en acel recta (fast_3dof 200 hz vs full_14dof 1000 hz) < 3 %
	var m14 := VehicleDynamicsModelScript.new()
	m14.load_vehicle("ter26_rwd", "default")
	var m3 := Fast3DofVehicleModelScript.new()
	m3.load_vehicle("ter26_rwd", "default")

	var inp14 := VehicleInputsScript.new()
	inp14.drive_mode = m14.config.drive_mode
	inp14.throttle = 0.85
	var inp3 := VehicleInputsScript.new()
	inp3.drive_mode = m3.config.drive_mode
	inp3.throttle = 0.85

	for _k in range(3000):  # 3.0 s a 1000 hz
		m14.step_standalone(inp14, 0.001)
	for _k in range(600):   # 3.0 s a 200 hz
		m3.step_standalone(inp3, 0.005)

	var vx14: float = m14.chassis_state.linear_velocity.x
	var vx3: float = m3.chassis_state.linear_velocity.x
	var rel_vx: float = absf(vx3 - vx14) / maxf(vx14, 1.0)
	if rel_vx < 0.03:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Straight accel parity failed: vx3=%.3f vs vx14=%.3f (rel=%.2f%%)" % [vx3, vx14, rel_vx * 100.0])

	# 3. paridad en estado estacionario (skidpad) < 3 %
	m14.reset()
	m3.reset()
	m14.chassis_state.linear_velocity = Vector3(10.0, 0.0, 0.0)
	m3.chassis_state.linear_velocity = Vector3(10.0, 0.0, 0.0)
	for w in m14.wheels:
		w.omega = 10.0 / m14.config.wheel_radius
	for w in m3.wheels:
		w.omega = 10.0 / m3.config.wheel_radius
	inp14.throttle = 0.28
	inp14.steer_angle = deg_to_rad(5.0)
	inp3.throttle = 0.28
	inp3.steer_angle = deg_to_rad(5.0)

	for _k in range(2500):  # 2.5 s a 1000 hz
		m14.step_standalone(inp14, 0.001)
	for _k in range(500):   # 2.5 s a 200 hz
		m3.step_standalone(inp3, 0.005)

	var wz14: float = m14.chassis_state.angular_velocity.z
	var wz3: float = m3.chassis_state.angular_velocity.z
	var rel_wz: float = absf(wz3 - wz14) / maxf(absf(wz14), 0.1)
	if rel_wz < 0.03:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Skidpad wz parity failed: wz3=%.4f vs wz14=%.4f (rel=%.2f%%)" % [wz3, wz14, rel_wz * 100.0])

	# 4. benchmark de vel: 10 s simulados (fast_3dof 200 hz vs full_14dof 1000 hz)
	m14.reset()
	m3.reset()
	var t0_14 := Time.get_ticks_usec()
	for _k in range(10000):
		m14.step_standalone(inp14, 0.001)
	var dt_14_ms := float(Time.get_ticks_usec() - t0_14) / 1000.0

	var t0_3 := Time.get_ticks_usec()
	for _k in range(2000):
		m3.step_standalone(inp3, 0.005)
	var dt_3_ms := float(Time.get_ticks_usec() - t0_3) / 1000.0

	var speedup := dt_14_ms / maxf(dt_3_ms, 0.01)
	if speedup > 2.0:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("Benchmark speedup too low: %.2fx (14dof=%.1f ms, 3dof=%.1f ms)" % [speedup, dt_14_ms, dt_3_ms])

	# 5. paridad 4wd + torque vectoring (ter27_4wd: acel + curva) < 4 %
	var m14_4wd := VehicleDynamicsModelScript.new()
	m14_4wd.load_vehicle("ter27_4wd", "default")
	var m3_4wd := Fast3DofVehicleModelScript.new()
	m3_4wd.load_vehicle("ter27_4wd", "default")
	var i14_4wd := VehicleInputsScript.new()
	i14_4wd.drive_mode = m14_4wd.config.drive_mode
	i14_4wd.ecu_driving_mode = 1
	i14_4wd.tv_mode = 2
	var i3_4wd := VehicleInputsScript.new()
	i3_4wd.drive_mode = m3_4wd.config.drive_mode
	i3_4wd.ecu_driving_mode = 1
	i3_4wd.tv_mode = 2

	for step in range(250):  # 2.5 s: 1.5s recta + 1.0s entrada en curva
		var t_s := float(step) * 0.01
		var thr := 0.70 if t_s < 1.5 else 0.45
		var str_a := deg_to_rad(5.0 * sin((t_s - 1.5) * 1.6)) if t_s > 1.5 else 0.0
		i14_4wd.throttle = thr
		i14_4wd.steer_angle = str_a
		i3_4wd.throttle = thr
		i3_4wd.steer_angle = str_a
		for _s in range(4):
			m14_4wd.step_standalone(i14_4wd, 0.0025)
		for _s in range(2):
			m3_4wd.step_standalone(i3_4wd, 0.005)

	var vx14_4 := m14_4wd.chassis_state.linear_velocity.x
	var vx3_4 := m3_4wd.chassis_state.linear_velocity.x
	var ay14_4 := m14_4wd.net_force_chassis_ros.y / m14_4wd.config.mass
	var ay3_4 := m3_4wd.net_force_chassis_ros.y / m3_4wd.config.mass
	var wz14_4 := m14_4wd.chassis_state.angular_velocity.z
	var wz3_4 := m3_4wd.chassis_state.angular_velocity.z
	var err_vx4 := absf(vx3_4 - vx14_4) / maxf(vx14_4, 1.0)
	var err_ay4 := absf(ay3_4 - ay14_4) / maxf(absf(ay14_4), 1.0)
	var err_wz4 := absf(wz3_4 - wz14_4) / maxf(absf(wz14_4), 0.1)
	var max_tq_err := 0.0
	for wi in range(4):
		var tq14 := float(m14_4wd.hv_state.inverter_torques[wi])
		var tq3 := float(m3_4wd.hv_state.inverter_torques[wi])
		max_tq_err = maxf(max_tq_err, absf(tq3 - tq14) / maxf(absf(tq14), 50.0))

	if err_vx4 < 0.04 and err_ay4 < 0.08 and err_wz4 < 0.08 and max_tq_err < 0.06:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append("4WD+TV parity failed: err_vx=%.2f%% err_ay=%.2f%% err_wz=%.2f%% max_tq_err=%.2f%%" % [
			err_vx4 * 100.0, err_ay4 * 100.0, err_wz4 * 100.0, max_tq_err * 100.0
		])

	return st
