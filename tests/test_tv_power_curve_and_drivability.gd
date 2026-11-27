extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const EcuTv4WdScript = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoring4Wd.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const SimScene = preload("res://Scenes/Sim/Sim.tscn")
const InputConditionerScript = preload("res://Scenes/Vehicle/Dynamics/Core/InputConditioner.gd")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func run() -> Dictionary:
	var st := {"name": "test_tv_power_curve_and_drivability", "passed": 0, "failed": 0, "errors": []}

	var cfg := VehicleConfigScript.new()
	_check(cfg.load_vehicle_configs("ter27_4wd"), "Loaded ter27_4wd configs", st)

	var ecu := EcuTv4WdScript.new()
	ecu.configure_from_vehicle_config(cfg)

	# 1. P_max(vx) in EFFICIENCY at 5, 14, 18.5, 25 m/s
	ecu.active_mode = EcuTv4WdScript.TestMode.EFFICIENCY
	ecu._refresh_mode_cache(ecu._get_mode_dict())
	var eff_cfg: Dictionary = ecu._get_mode_dict()
	var p_lo_kw := float(eff_cfg.get("p_max_low_kw", 68.0))
	var p_hi_kw := float(eff_cfg.get("p_max_high_kw", 36.0))
	var v_lo := float(eff_cfg.get("p_max_v_low_mps", 14.0))
	var v_hi := float(eff_cfg.get("p_max_v_high_mps", 23.0))
	var v_mid := 0.5 * (v_lo + v_hi)

	var c_5 := ecu.get_effective_power_cap_w(5.0) * 0.001
	var c_14 := ecu.get_effective_power_cap_w(v_lo) * 0.001
	var c_mid := ecu.get_effective_power_cap_w(v_mid) * 0.001
	var c_25 := ecu.get_effective_power_cap_w(25.0) * 0.001
	_check(absf(c_5 - p_lo_kw) < 1e-3, "EFFICIENCY P_max(5 m/s) == %.2f kW (got %.2f)" % [p_lo_kw, c_5], st)
	_check(absf(c_14 - p_lo_kw) < 1e-3, "EFFICIENCY P_max(v_low) == %.2f kW (got %.2f)" % [p_lo_kw, c_14], st)
	_check(absf(c_mid - 0.5 * (p_lo_kw + p_hi_kw)) < 1e-3, "EFFICIENCY P_max(mid) == %.2f kW (got %.2f)" % [0.5 * (p_lo_kw + p_hi_kw), c_mid], st)
	_check(absf(c_25 - p_hi_kw) < 1e-3, "EFFICIENCY P_max(25 m/s) == %.2f kW (got %.2f)" % [p_hi_kw, c_25], st)
	_check(c_5 <= 80.0 and c_14 <= 80.0 and c_mid <= 80.0 and c_25 <= 80.0, "EFFICIENCY P_max(vx) <= 80.0 kW everywhere", st)

	# AUTOCROSS keeps 80 kW across all speeds including standstill/low speed
	ecu.active_mode = EcuTv4WdScript.TestMode.AUTOCROSS
	ecu._refresh_mode_cache(ecu._get_mode_dict())
	_check(absf(ecu.get_effective_power_cap_w(1.0) * 0.001 - 80.0) < 1e-3 and absf(ecu.get_effective_power_cap_w(25.0) * 0.001 - 80.0) < 1e-3, "AUTOCROSS keeps 80.0 kW cap from low speed to high speed", st)
	_check(absf(float(ecu._get_mode_dict().get("beta_soft_deg", 0.0)) - 4.5) < 1e-3, "AUTOCROSS beta_soft_deg calibrated to 4.5 deg", st)

	# 2. Cornering Throttle Shaping: 1:1 at delta=0, progressive at |delta|=0.30 rad
	var t_straight := ecu.shape_corner_throttle(0.50, 0.0)
	var t_corner := ecu.shape_corner_throttle(0.50, 0.30)
	var t_full := ecu.shape_corner_throttle(1.00, 0.30)
	_check(absf(t_straight - 0.50) < 1e-5, "Cornering throttle is 1:1 at delta=0 (got %.4f)" % t_straight, st)
	_check(t_corner > 0.0 and t_corner < 0.50, "Cornering throttle progressive < 0.50 at |delta|=0.30 (got %.4f)" % t_corner, st)
	_check(absf(t_full - 1.00) < 1e-5, "Cornering throttle reaches 1.00 at WOT (got %.4f)" % t_full, st)

	# 3. Soft Beta-Barrier: zero below beta_soft_deg, restoring above
	var b_soft_rad := deg_to_rad(float(ecu._get_mode_dict().get("beta_soft_deg", 3.2)))
	var mz_low := ecu.compute_soft_beta_barrier_nm(0.5 * b_soft_rad, 0.0)
	var mz_hi_neg := ecu.compute_soft_beta_barrier_nm(-(b_soft_rad + deg_to_rad(1.5)), -0.3)
	var mz_hi_pos := ecu.compute_soft_beta_barrier_nm((b_soft_rad + deg_to_rad(1.5)), 0.3)
	_check(absf(mz_low) < 1e-6, "Soft Beta-Barrier == 0 below beta_soft_deg", st)
	_check(mz_hi_neg < -1.0 and mz_hi_pos > 1.0, "Soft Beta-Barrier applies smooth restoring yaw moment above beta_soft", st)

	# 3b. Soft Beta-Barrier evaluates beta_dyn = beta - atan2(b * wz, max(vx, 3)): R=5.2 m steady turn gives 0
	var vx_hp := 8.0
	var wz_hp := vx_hp / 5.2
	var b_kin_hp := atan2(ecu.b_lr * wz_hp, vx_hp)  # ~7.6 deg kinematic sideslip
	var mz_kin_only := ecu.compute_soft_beta_barrier_nm(b_kin_hp, 0.0, wz_hp, vx_hp)
	var mz_dyn_over := ecu.compute_soft_beta_barrier_nm(b_kin_hp - (b_soft_rad + deg_to_rad(1.5)), -0.3, wz_hp, vx_hp)
	_check(b_kin_hp > deg_to_rad(7.0) and absf(mz_kin_only) < 1e-6, "Soft Beta-Barrier ignores pure kinematic sideslip in R=5.2m hairpin", st)
	_check(mz_dyn_over < -1.0, "Soft Beta-Barrier fires only on true dynamic slip above beta_soft", st)

	# 3c. Progressive 35/65 braking with linear trigger & -25 kW charge cap
	_check(absf(ecu.mech_brake_start - 0.35) < 1e-5, "mech_brake_start configured to 0.35 (35/65 split)", st)
	var om_brk: Array[float] = [20.0 / ecu.r_eff, 20.0 / ecu.r_eff, 20.0 / ecu.r_eff, 20.0 / ecu.r_eff]
	ecu.step(-2000.0, 0.0, 20.0, 0.0, -4.0, 0.0, 9.81, om_brk, 0.25, 0.005, [0.0, 0.0, 0.0, 0.0])
	var no_hyd_lo := is_zero_approx(ecu.t_hyd_est[0]) and (ecu.last_power_w >= -(ecu.p_max_charge_w + 50.0))
	ecu.step(-2500.0, 0.0, 20.0, 0.0, -7.0, 0.0, 9.81, om_brk, 0.50, 0.005, [0.0, 0.0, 0.0, 0.0])
	var hyd_mid := ecu.t_hyd_est[0]
	ecu.step(-2500.0, 0.0, 20.0, 0.0, -14.0, 0.0, 9.81, om_brk, 1.00, 0.005, [0.0, 0.0, 0.0, 0.0])
	var hyd_full := ecu.t_hyd_est[0]
	_check(no_hyd_lo and hyd_mid > 20.0 and hyd_full > 3.5 * hyd_mid, "35/65 brake scales linearly above 0.35 after pure regen", st)

	# 4. Lift-off Regen: smooth with throttle=0, brake=0, vx=15 m/s; cut off if |beta|>3.5 deg; >= -25.0 kW
	var model := Full14DofScript.new(cfg)
	model.chassis_state.linear_velocity = Vector3(15.0, 0.0, 0.0)
	for w in model.wheels:
		w.omega = 15.0 / cfg.wheel_radius
	model.ecu_tv_4wd.ekf_vx = 15.0
	model.ecu_tv_4wd._ekf_init = true

	var inp := VehicleInputsScript.new()
	inp.drive_mode = cfg.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 2  # EFFICIENCY
	inp.throttle = 0.0
	inp.brake = 0.0
	inp.steer_angle = 0.0

	for _k in range(40):
		model.step_standalone(inp, 0.0025, 0.0)
	var p_lift_kw := model.ecu_tv_4wd.last_power_w * 0.001
	var fx_filt_norm := model.ecu_tv_4wd._lift_regen_fx_filt
	_check(fx_filt_norm < -5.0 and p_lift_kw < -0.5 and p_lift_kw >= -25.05, "Lift-off regen smooth & >= -25 kW (P=%.2f kW)" % p_lift_kw, st)

	# High beta (> 3.5 deg): lift-off regen cuts cleanly to 0
	model.chassis_state.linear_velocity = Vector3(15.0, 15.0 * tan(deg_to_rad(5.0)), 0.0)
	model.ecu_tv_4wd.ekf_vx = 15.0
	model.ecu_tv_4wd.ekf_vy = 15.0 * tan(deg_to_rad(5.0))
	model.step_standalone(inp, 0.0025, 0.0)
	_check(absf(model.ecu_tv_4wd._lift_regen_fx_filt) < 1e-6, "Lift-off regen cuts to 0 when |beta| > 3.5 deg", st)

	return st


static func _reset_and_settle_car(tree: SceneTree, car: MirenaCar, pose: Transform3D, vx_init: float = 0.0) -> void:
	car.pilot = MirenaCar.PilotMode.ROS
	car.set_control(0.0, 0.0, 0.0)
	car.model.reset()
	if car.input_conditioner != null:
		car.input_conditioner.reset()
	car.linear_velocity = Vector3.ZERO
	car.angular_velocity = Vector3.ZERO
	car.global_transform = pose
	for _i in range(40):
		await tree.physics_frame
	if vx_init > 0.0:
		var settled_y: float = car.global_position.y
		var launch_tf := Transform3D(pose.basis, Vector3(pose.origin.x, settled_y, pose.origin.z))
		car.global_transform = launch_tf
		car.linear_velocity = -launch_tf.basis.z * vx_init
		car.angular_velocity = Vector3.ZERO
		car.model.reset()
		car.model.chassis_state.linear_velocity = Vector3(vx_init, 0.0, 0.0)
		for w in car.model.wheels:
			w.omega = vx_init / maxf(car.config.wheel_radius, 0.1)
		if car.model.ecu_tv_4wd != null:
			car.model.ecu_tv_4wd.ekf_vx = vx_init
			car.model.ecu_tv_4wd.ekf_vy = 0.0
			car.model.ecu_tv_4wd._ekf_init = true
		await tree.physics_frame
	else:
		car.linear_velocity = Vector3.ZERO
		car.angular_velocity = Vector3.ZERO
		car.model.reset()
		if car.input_conditioner != null:
			car.input_conditioner.reset()


static func _compute_mz_tv_nm(car: MirenaCar, trqs: Array) -> float:
	if trqs.size() < 4:
		return 0.0
	var r_eff: float = maxf(car.config.wheel_radius, 0.1)
	var tf_half: float = 0.5 * car.config.track_width_f
	var tr_half: float = 0.5 * car.config.track_width_r
	var t_fl := float(trqs[0])
	var t_fr := float(trqs[1])
	var t_rl := float(trqs[2])
	var t_rr := float(trqs[3])
	return ((t_fr - t_fl) * tf_half + (t_rr - t_rl) * tr_half) / r_eff


static func run_async(tree: SceneTree) -> Dictionary:
	var st := run()

	var sim := SimScene.instantiate()
	tree.root.add_child(sim)
	await tree.process_frame
	await tree.process_frame

	var car := sim.get_node_or_null("Car") as MirenaCar
	var trk := sim.get_node_or_null("Track") as Track
	_check(car != null and trk != null, "Instantiated real Sim.tscn in SceneTree", st)
	if car == null or trk == null:
		if is_instance_valid(sim):
			sim.queue_free()
		return st

	_check(car.ecu_driving_mode == 2 and car.tv_4wd_mode == 0, "Sim.tscn Car defaults to ECU_TV_4WD_CASCADE AUTOCROSS", st)
	var dt_phys: float = 1.0 / float(maxi(Engine.physics_ticks_per_second, 60))

	# Test 1 -- Live Standstill Launch (real Sim.tscn + acceleration.json + RigidBody3D, KEYBOARD & GAMEPAD)
	trk.load_track("res://TrackFiles/acceleration.json")
	await tree.process_frame
	await tree.process_frame
	var accel_pose := Transform3D(trk.origin.basis, Vector3(trk.origin.origin.x, 0.05, trk.origin.origin.z))

	for prof_id in [InputConditionerScript.InputProfile.KEYBOARD, InputConditionerScript.InputProfile.GAMEPAD]:
		await _reset_and_settle_car(tree, car, accel_pose, 0.0)
		car.pilot = MirenaCar.PilotMode.MANUAL
		car.set_input_profile(prof_id)
		Input.action_press("manual_gas_pos", 1.0)

		var p_dc_max := -1e9
		var p_dc_min := 1e9
		var t_50 := -1.0
		var t_100 := -1.0
		var hv_bad := 0
		var n_ticks := int(round(3.15 / dt_phys))
		for k in range(n_ticks):
			await tree.physics_frame
			var t_s: float = float(k + 1) * dt_phys
			var inv_b: Basis = car.global_transform.basis.inverse()
			var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
			var kmh: float = lv_ros.x * 3.6
			var p_kw: float = car.tv_elec_power_kw
			p_dc_max = maxf(p_dc_max, p_kw)
			p_dc_min = minf(p_dc_min, p_kw)
			if p_kw > 80.05 or p_kw < -25.05:
				hv_bad += 1
			if t_50 < 0.0 and kmh >= 50.0:
				t_50 = t_s
			if t_100 < 0.0 and kmh >= 100.0:
				t_100 = t_s
				if t_s >= 3.0:
					break
		Input.action_release("manual_gas_pos")
		var prof_label := "KEYBOARD" if prof_id == InputConditionerScript.InputProfile.KEYBOARD else "GAMEPAD"
		print("  [TEST1_LAUNCH] %s: 0-50=%.3fs 0-100=%.3fs P_dc_max=%.2fkW P_dc_min=%.2fkW hv_bad=%d" % [prof_label, t_50, t_100, p_dc_max, p_dc_min, hv_bad])
		_check(t_50 > 0.0 and t_50 <= 1.25, "Test 1 %s 0-50 km/h=%.3f s <= 1.25 s" % [prof_label, t_50], st)
		_check(t_100 > 0.0 and t_100 <= 2.95, "Test 1 %s 0-100 km/h=%.3f s <= 2.95 s" % [prof_label, t_100], st)
		_check(p_dc_max >= 75.0 and p_dc_max <= 80.05 and hv_bad == 0, "Test 1 %s P_dc_max=%.2f kW in [75, 80] kW (hv_bad=%d)" % [prof_label, p_dc_max, hv_bad], st)

	# Cleanup Test 1 and run Test 1B: Re-Launch after braking to standstill in AUTOCROSS
	car.set_sim_drive_mode(1)
	await _reset_and_settle_car(tree, car, accel_pose, 0.0)
	car.pilot = MirenaCar.PilotMode.ROS
	var rl_hv_bad := 0
	# Phase A: accelerate to 15 m/s
	car.set_control(1.0, 0.0, 0.0)
	for _k in range(int(round(2.5 / dt_phys))):
		await tree.physics_frame
		var p_kw: float = car.tv_elec_power_kw
		if p_kw > 80.05 or p_kw < -25.05:
			rl_hv_bad += 1
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		if lv_ros.x >= 15.0:
			break
	# Phase B: brake to vx < 0.2 m/s
	car.set_control(0.0, 0.0, 1.0)
	var vx_stopped := 99.0
	for _k in range(int(round(3.0 / dt_phys))):
		await tree.physics_frame
		var p_kw: float = car.tv_elec_power_kw
		if p_kw > 80.05 or p_kw < -25.05:
			rl_hv_bad += 1
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		vx_stopped = lv_ros.x
		if vx_stopped < 0.2:
			break
	# Phase C: release brake & full throttle -> 0-50 km/h <= 1.15s, P_dc_max >= 78.0 kW
	car.set_control(1.0, 0.0, 0.0)
	var rl_t50 := -1.0
	var rl_pmax := -1e9
	var rl_pmin := 1e9
	for k in range(int(round(2.0 / dt_phys))):
		await tree.physics_frame
		var t_s: float = float(k + 1) * dt_phys
		var p_kw: float = car.tv_elec_power_kw
		rl_pmax = maxf(rl_pmax, p_kw)
		rl_pmin = minf(rl_pmin, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			rl_hv_bad += 1
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		if rl_t50 < 0.0 and lv_ros.x * 3.6 >= 50.0:
			rl_t50 = t_s
	print("  [TEST1B_RELAUNCH] vx_stop=%.3fm/s 0-50=%.3fs P_dc_max=%.2fkW P_dc_min=%.2fkW hv_bad=%d" % [
		vx_stopped, rl_t50, rl_pmax, rl_pmin, rl_hv_bad
	])
	_check(vx_stopped < 0.2 and rl_t50 > 0.0 and rl_t50 <= 1.15,
		"Test 1B Re-Launch after standstill: vx_stop=%.3f<0.2, 0-50=%.3fs <= 1.15s" % [vx_stopped, rl_t50], st)
	_check(rl_pmax >= 78.0 and rl_pmax <= 80.05 and rl_hv_bad == 0,
		"Test 1B Re-Launch P_dc_max=%.2f kW >= 78.0 kW (hv_bad=%d)" % [rl_pmax, rl_hv_bad], st)

	# Clear cones for open-asphalt 10-Re-Launch, Hairpin, Trail-Brake, Skidpad, and Slalom/Chicane maneuvers on real Ground StaticBody3D
	trk.clear_track()
	await tree.process_frame
	var open_pose := Transform3D(Basis.IDENTITY, Vector3(0.0, 0.05, 0.0))

	# Test 1C -- Strict 10-Consecutive Re-Launch Repeatability (AUTOCROSS & ACCELERATION @ 400 Hz Sim.tscn)
	var sim_auto_1c := tree.root.get_node_or_null("Sim")
	if sim_auto_1c != null:
		sim_auto_1c.set("car", car)
		sim_auto_1c.set("track", trk)
	var overlap_ticks: int = maxi(int(round(0.030 / dt_phys)), 1)
	var steer_abs_rad: float = 0.60 * car.config.max_steer_angle
	for rl_mode_spec in [
		{"mode": 1, "name": "AUTOCROSS"},
		{"mode": 4, "name": "ACCELERATION"},
	]:
		var rl_m_id: int = int(rl_mode_spec["mode"])
		var rl_m_name: String = String(rl_mode_spec["name"])
		car.set_sim_drive_mode(rl_m_id)
		await _reset_and_settle_car(tree, car, open_pose, 0.0)
		car.pilot = MirenaCar.PilotMode.ROS

		var mode_pass_cnt := 0
		var mode_hv_bad := 0
		var mode_worst_t50 := 0.0
		var mode_min_pmax := 1e9
		var mode_max_pmax := -1e9
		var mode_min_pmin := 1e9

		for cyc in range(10):
			var cyc_type := ""
			var cyc_hv_bad := 0
			var vx_pre := 0.0

			if cyc <= 7:
				# Ensure car is at speed (>= 14 m/s) before braking into stop/rolling-stop
				var inv_b0: Basis = car.global_transform.basis.inverse()
				var lv0: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b0 * car.linear_velocity)
				if lv0.x < 14.0:
					car.set_control(1.0, 0.0, 0.0)
					for _a in range(int(round(2.0 / dt_phys))):
						await tree.physics_frame
						var p_a: float = car.tv_elec_power_kw
						mode_max_pmax = maxf(mode_max_pmax, p_a)
						mode_min_pmin = minf(mode_min_pmin, p_a)
						if p_a > 80.005 or p_a < -25.005:
							cyc_hv_bad += 1
						var lv_a: Vector3 = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity)
						if lv_a.x >= 14.0:
							break

				if cyc <= 2:
					cyc_type = "FULL_STOP"
					car.set_control(0.0, 0.0, 1.0)
					for _b in range(int(round(4.5 / dt_phys))):
						await tree.physics_frame
						var p_b: float = car.tv_elec_power_kw
						mode_max_pmax = maxf(mode_max_pmax, p_b)
						mode_min_pmin = minf(mode_min_pmin, p_b)
						if p_b > 80.005 or p_b < -25.005:
							cyc_hv_bad += 1
						var lv_b: Vector3 = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity)
						vx_pre = lv_b.x
						if vx_pre <= 0.05:
							break
					for _s in range(10):
						await tree.physics_frame
						var p_s: float = car.tv_elec_power_kw
						if p_s > 80.005 or p_s < -25.005:
							cyc_hv_bad += 1
						vx_pre = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity).x
				elif cyc <= 4:
					cyc_type = "ABS_STEER_STOP"
					car.set_control(0.0, steer_abs_rad, 1.0)
					for _b in range(int(round(4.5 / dt_phys))):
						await tree.physics_frame
						var p_b: float = car.tv_elec_power_kw
						mode_max_pmax = maxf(mode_max_pmax, p_b)
						mode_min_pmin = minf(mode_min_pmin, p_b)
						if p_b > 80.005 or p_b < -25.005:
							cyc_hv_bad += 1
						var lv_b: Vector3 = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity)
						vx_pre = lv_b.x
						if vx_pre <= 0.10:
							break
				else:
					cyc_type = "ROLLING_OVERLAP"
					car.set_control(0.0, 0.0, 1.0)
					for _b in range(int(round(4.5 / dt_phys))):
						await tree.physics_frame
						var p_b: float = car.tv_elec_power_kw
						mode_max_pmax = maxf(mode_max_pmax, p_b)
						mode_min_pmin = minf(mode_min_pmin, p_b)
						if p_b > 80.005 or p_b < -25.005:
							cyc_hv_bad += 1
						var lv_b: Vector3 = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity)
						vx_pre = lv_b.x
						if vx_pre <= 0.85:
							break
					# 30 ms brake-throttle overlap
					car.set_control(1.0, 0.0, 1.0)
					for _ov in range(overlap_ticks):
						await tree.physics_frame
						var p_ov: float = car.tv_elec_power_kw
						mode_max_pmax = maxf(mode_max_pmax, p_ov)
						mode_min_pmin = minf(mode_min_pmin, p_ov)
						if p_ov > 80.005 or p_ov < -25.005:
							cyc_hv_bad += 1
						vx_pre = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity).x
			else:
				cyc_type = "LAUNCH_PAD_RESPAWN"
				car.set_control(0.0, 0.0, 0.0)
				if sim_auto_1c != null and sim_auto_1c.has_method("spawn_test_zone"):
					sim_auto_1c.spawn_test_zone("launch", car)
				else:
					await _reset_and_settle_car(tree, car, open_pose, 0.0)
				car.pilot = MirenaCar.PilotMode.ROS
				car.set_control(0.0, 0.0, 0.0)
				for _w in range(60):
					await tree.physics_frame
				vx_pre = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity).x

			# Launch phase: straighten wheel (0.0 rad), release brake, full throttle
			car.set_control(1.0, 0.0, 0.0)
			var cyc_t50 := -1.0
			var cyc_pmax := -1e9
			var cyc_pmin := 1e9
			for k in range(int(round(1.80 / dt_phys))):
				await tree.physics_frame
				var t_s: float = float(k + 1) * dt_phys
				var p_kw: float = car.tv_elec_power_kw
				cyc_pmax = maxf(cyc_pmax, p_kw)
				cyc_pmin = minf(cyc_pmin, p_kw)
				mode_max_pmax = maxf(mode_max_pmax, p_kw)
				mode_min_pmin = minf(mode_min_pmin, p_kw)
				if p_kw > 80.005 or p_kw < -25.005:
					cyc_hv_bad += 1
				var lv_l: Vector3 = VehicleConfigScript.godot_to_ros_vec(car.global_transform.basis.inverse() * car.linear_velocity)
				if cyc_t50 < 0.0 and lv_l.x * 3.6 >= 50.0:
					cyc_t50 = t_s

			mode_hv_bad += cyc_hv_bad
			mode_worst_t50 = maxf(mode_worst_t50, cyc_t50 if cyc_t50 > 0.0 else 99.0)
			mode_min_pmax = minf(mode_min_pmax, cyc_pmax)
			var t50_lim := 1.15
			if rl_m_name == "AUTOCROSS":
				if cyc_type == "ABS_STEER_STOP":
					t50_lim = 1.35
				elif cyc_type == "LAUNCH_PAD_RESPAWN":
					t50_lim = 1.25
			var cyc_ok: bool = (cyc_t50 > 0.0 and cyc_t50 <= t50_lim and cyc_pmax >= 78.0 and cyc_pmax <= 80.005 and cyc_hv_bad == 0)
			if cyc_ok:
				mode_pass_cnt += 1
			print("  [TEST1C_%s_%02d] %-18s vx_pre=%.3fm/s 0-50=%.3fs P_dc=[%.2f,%.3f]kW hv_bad=%d ok=%s" % [
				rl_m_name, cyc + 1, cyc_type, vx_pre, cyc_t50, cyc_pmin, cyc_pmax, cyc_hv_bad, str(cyc_ok)
			])

		print("  [TEST1C_SUMMARY_%s] passed=%d/10 worst_0_50=%.3fs min_P_dc_max=%.2fkW max_P_dc=%.3fkW min_P_dc=%.2fkW hv_bad=%d" % [
			rl_m_name, mode_pass_cnt, mode_worst_t50, mode_min_pmax, mode_max_pmax, mode_min_pmin, mode_hv_bad
		])
		_check(mode_pass_cnt == 10 and mode_hv_bad == 0,
			"Test 1C %s 10/10 Re-Launch repeatability: %d/10 passed, worst 0-50=%.3fs, min P_dc_max=%.2fkW >= 78.0, hv_bad=%d" % [
				rl_m_name, mode_pass_cnt, mode_worst_t50, mode_min_pmax, mode_hv_bad
			], st)
	car.set_sim_drive_mode(1)  # Restore AUTOCROSS

	# Test 2 -- Aggressive Hairpin Pivot (vx = 6.5 m/s, delta_norm_raw = 0.80 Left/Right)
	var steer_hp_rad: float = 0.80 * car.config.max_steer_angle
	var mz_exit_left := 0.0
	var mz_exit_right := 0.0
	var mz_entry_left := 0.0
	var mz_entry_right := 0.0
	for sgn in [1.0, -1.0]:
		var dir_label := "LEFT" if sgn > 0.0 else "RIGHT"

		# Phase 2A: Entry (thr = 0.15 -> T_in <= -45 Nm, T_out >= +60 Nm, |Mz_TV| >= 300 Nm)
		await _reset_and_settle_car(tree, car, open_pose, 6.5)
		car.pilot = MirenaCar.PilotMode.ROS
		car.set_control(0.15, float(sgn) * steer_hp_rad, 0.0)
		var hv_entry_bad := 0
		var int_min_entry := 1e9
		var out_max_entry := -1e9
		var mz_entry_peak := 0.0
		var mz_entry_last := 0.0
		var trqs_entry_last: Array = [0.0, 0.0, 0.0, 0.0]
		var n_entry := int(round(0.35 / dt_phys))
		for _k in range(n_entry):
			await tree.physics_frame
			var p_kw: float = car.tv_elec_power_kw
			if p_kw > 80.05 or p_kw < -25.05:
				hv_entry_bad += 1
			var trqs: Array = car.model.ecu_tv_4wd.last_t_cmd
			trqs_entry_last = trqs.duplicate()
			mz_entry_last = _compute_mz_tv_nm(car, trqs)
			if absf(mz_entry_last) > absf(mz_entry_peak):
				mz_entry_peak = mz_entry_last
			if sgn > 0.0:
				int_min_entry = minf(int_min_entry, minf(float(trqs[0]), float(trqs[2])))
				out_max_entry = maxf(out_max_entry, maxf(float(trqs[1]), float(trqs[3])))
			else:
				int_min_entry = minf(int_min_entry, minf(float(trqs[1]), float(trqs[3])))
				out_max_entry = maxf(out_max_entry, maxf(float(trqs[0]), float(trqs[2])))
		if sgn > 0.0:
			mz_entry_left = mz_entry_last
		else:
			mz_entry_right = mz_entry_last
		var pro_turn_entry: bool = (mz_entry_last > 0.0) if sgn > 0.0 else (mz_entry_last < 0.0)
		var entry_ok: bool = (int_min_entry <= -45.0) and (out_max_entry >= 60.0) and (absf(mz_entry_peak) >= 300.0) and pro_turn_entry and (hv_entry_bad == 0)
		print("  [TEST2_HAIRPIN_ENTRY] %s: T=[%.1f,%.1f,%.1f,%.1f] int_min=%.1fNm out_max=%.1fNm Mz_TV=%.1fNm (peak=%.1f) hv_bad=%d" % [
			dir_label, float(trqs_entry_last[0]), float(trqs_entry_last[1]), float(trqs_entry_last[2]), float(trqs_entry_last[3]),
			int_min_entry, out_max_entry, mz_entry_last, mz_entry_peak, hv_entry_bad
		])
		_check(entry_ok, "Test 2 %s hairpin entry: T_in=%.1f<=-45, T_out=%.1f>=60, |Mz_TV|=%.1f>=300 (hv_bad=%d)" % [dir_label, int_min_entry, out_max_entry, absf(mz_entry_peak), hv_entry_bad], st)

		# Phase 2B: Exit (thr = 0.85 -> T_out >> T_in >= 0, pro-turn Mz_TV)
		await _reset_and_settle_car(tree, car, open_pose, 6.5)
		car.pilot = MirenaCar.PilotMode.ROS
		car.set_control(0.85, float(sgn) * steer_hp_rad, 0.0)
		var hv_exit_bad := 0
		var mz_exit_sum := 0.0
		var t_sum: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var n_win := 0
		var n_exit := int(round(0.40 / dt_phys))
		var k_win_start := int(round(0.15 / dt_phys))
		for k in range(n_exit):
			await tree.physics_frame
			var p_kw: float = car.tv_elec_power_kw
			if p_kw > 80.05 or p_kw < -25.05:
				hv_exit_bad += 1
			if k >= k_win_start:
				var trqs: Array = car.model.ecu_tv_4wd.last_t_cmd
				for wi in range(4):
					t_sum[wi] += float(trqs[wi])
				mz_exit_sum += _compute_mz_tv_nm(car, trqs)
				n_win += 1
		var inv_w: float = 1.0 / float(maxi(n_win, 1))
		var t_fl: float = t_sum[0] * inv_w
		var t_fr: float = t_sum[1] * inv_w
		var t_rl: float = t_sum[2] * inv_w
		var t_rr: float = t_sum[3] * inv_w
		var mz_exit_avg := mz_exit_sum * inv_w
		if sgn > 0.0:
			mz_exit_left = mz_exit_avg
		else:
			mz_exit_right = mz_exit_avg
		print("  [TEST2_HAIRPIN_EXIT]  %s: T_avg=[%.1f,%.1f,%.1f,%.1f] Mz_TV_avg=%.1fNm hv_bad=%d" % [
			dir_label, t_fl, t_fr, t_rl, t_rr, mz_exit_avg, hv_exit_bad
		])
		if sgn > 0.0:
			_check((t_fr + t_rr) > (t_fl + t_rl) + 50.0 and t_rr > t_rl + 50.0 and minf(t_fl, t_rl) >= -0.5 and mz_exit_avg > 0.0 and hv_exit_bad == 0,
				"Test 2 LEFT hairpin exit: T_out>>T_in>=0 (FL=%.1f FR=%.1f RL=%.1f RR=%.1f), Mz_TV=%.1f>0 (hv_bad=%d)" % [t_fl, t_fr, t_rl, t_rr, mz_exit_avg, hv_exit_bad], st)
		else:
			_check((t_fl + t_rl) > (t_fr + t_rr) + 50.0 and t_rl > t_rr + 50.0 and minf(t_fr, t_rr) >= -0.5 and mz_exit_avg < 0.0 and hv_exit_bad == 0,
				"Test 2 RIGHT hairpin exit: T_out>>T_in>=0 (FL=%.1f FR=%.1f RL=%.1f RR=%.1f), Mz_TV=%.1f<0 (hv_bad=%d)" % [t_fl, t_fr, t_rl, t_rr, mz_exit_avg, hv_exit_bad], st)

	var sym_entry_err: float = absf(mz_entry_left + mz_entry_right)
	var sym_exit_err: float = absf(mz_exit_left + mz_exit_right)
	_check(sym_entry_err < 15.0 and sym_exit_err < 25.0,
		"Test 2 L/R hairpin symmetry: |Mz_L+Mz_R| entry=%.1f Nm, exit=%.1f Nm" % [sym_entry_err, sym_exit_err], st)

	# Phase 2C: hairpin_pivot_gain at max (2.00) -> |Mz_TV| >= 420 Nm
	car.apply_live_tuning({"hairpin_pivot_gain": 2.0})
	await _reset_and_settle_car(tree, car, open_pose, 6.5)
	car.pilot = MirenaCar.PilotMode.ROS
	car.set_control(0.15, steer_hp_rad, 0.0)
	var mz_max_gain_peak := 0.0
	var hv_max_gain_bad := 0
	for _k in range(int(round(0.35 / dt_phys))):
		await tree.physics_frame
		var p_kw: float = car.tv_elec_power_kw
		if p_kw > 80.05 or p_kw < -25.05:
			hv_max_gain_bad += 1
		var mz_g := _compute_mz_tv_nm(car, car.model.ecu_tv_4wd.last_t_cmd)
		if absf(mz_g) > absf(mz_max_gain_peak):
			mz_max_gain_peak = mz_g
	car.reset_map_live_tuning()
	print("  [TEST2C_MAX_GAIN] hairpin_pivot_gain=2.0: |Mz_TV|_peak=%.1fNm hv_bad=%d" % [absf(mz_max_gain_peak), hv_max_gain_bad])
	_check(absf(mz_max_gain_peak) >= 420.0 and hv_max_gain_bad == 0,
		"Test 2C hairpin_pivot_gain=2.0: |Mz_TV|=%.1f Nm >= 420 Nm (hv_bad=%d)" % [absf(mz_max_gain_peak), hv_max_gain_bad], st)

	# Test 3 -- Trail-Braking on entry (vx = 18 m/s, brake = 0.65, steer = 0.50)
	await _reset_and_settle_car(tree, car, open_pose, 18.0)
	car.pilot = MirenaCar.PilotMode.ROS
	var tb_hv_bad := 0
	var tb_p_min := 1e9
	var tb_t_sum: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var tb_n := 0
	var tb_rear_lock_streak := 0
	var tb_rear_lock_max := 0
	var tb_diag_last := {}
	var r_eff_tb: float = maxf(car.config.wheel_radius, 0.1)
	var n_tb := int(round(0.25 / dt_phys))
	var k_tb_start := 2
	var k_tb_end := 14
	for k in range(n_tb):
		car.set_control(0.0, 0.50 * car.config.max_steer_angle, 0.65)
		await tree.physics_frame
		var p_kw: float = car.tv_elec_power_kw
		tb_p_min = minf(tb_p_min, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			tb_hv_bad += 1
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		var om_rl: float = car.model.wheels[2].omega
		var om_rr: float = car.model.wheels[3].omega
		if lv_ros.x > 3.0 and minf(om_rl, om_rr) * r_eff_tb < 0.25 * lv_ros.x:
			tb_rear_lock_streak += 1
			tb_rear_lock_max = maxi(tb_rear_lock_max, tb_rear_lock_streak)
		else:
			tb_rear_lock_streak = 0
		if k >= k_tb_start and k < k_tb_end:
			var trqs: Array = car.model.ecu_tv_4wd.last_t_cmd
			for wi in range(4):
				tb_t_sum[wi] += float(trqs[wi])
			tb_diag_last = car.get_tv_live_diagnostics()
			tb_n += 1
	var tb_inv := 1.0 / float(maxi(tb_n, 1))
	var tb_fl := tb_t_sum[0] * tb_inv
	var tb_fr := tb_t_sum[1] * tb_inv
	var tb_rl := tb_t_sum[2] * tb_inv
	var tb_rr := tb_t_sum[3] * tb_inv
	var tb_in_avg := 0.5 * (tb_fl + tb_rl)
	var tb_out_avg := 0.5 * (tb_fr + tb_rr)
	print("  [TEST3_TRAIL_BRAKE] T_avg=[%.1f,%.1f,%.1f,%.1f] T_in=%.1f T_out=%.1f rear_lock_ticks=%d P_min=%.2fkW regime=%s beta=%.2fdeg hv_bad=%d" % [
		tb_fl, tb_fr, tb_rl, tb_rr, tb_in_avg, tb_out_avg, tb_rear_lock_max, tb_p_min,
		String(tb_diag_last.get("active_regime", "")), float(tb_diag_last.get("beta_deg", 0.0)), tb_hv_bad
	])
	_check(tb_in_avg < tb_out_avg - 2.0 and tb_out_avg <= 0.5 and tb_rear_lock_max < 15 and tb_p_min >= -25.05 and tb_hv_bad == 0,
		"Test 3 Trail-Braking: T_in=%.1f < T_out=%.1f <= 0, rear_lock_ticks=%d < 15, P_min=%.2fkW >= -25 (hv_bad=%d)" % [tb_in_avg, tb_out_avg, tb_rear_lock_max, tb_p_min, tb_hv_bad], st)

	# Test 4 -- SKIDPAD (R = 9.125 m in Sim.tscn) in FSG_REAL_TRACK and TTC_LAB_PEAK
	car.set_sim_drive_mode(3)  # SKIDPAD
	for gp_case in [
		{"id": 1, "name": "FSG_REAL_TRACK", "vx0": 11.5, "min_ay_g": 1.22},
		{"id": 0, "name": "TTC_LAB_PEAK", "vx0": 12.5, "min_ay_g": 1.40},
	]:
		var gp_id: int = int(gp_case["id"])
		var gp_name: String = String(gp_case["name"])
		var vx0: float = float(gp_case["vx0"])
		var min_ay_g: float = float(gp_case["min_ay_g"])
		car.set_track_grip_profile(gp_id)
		await _reset_and_settle_car(tree, car, open_pose, vx0)
		car.pilot = MirenaCar.PilotMode.ROS
		var sk_hv_bad := 0
		var sk_ay_max := 0.0
		var sk_ay_sum := 0.0
		var sk_t_sum: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sk_n := 0
		var sk_regime_last := ""
		var sk_regime_hits := 0
		var n_sk := int(round(0.80 / dt_phys))
		var k_sk_win := int(round(0.20 / dt_phys))
		var steer_sk_rad: float = 0.52 * car.config.max_steer_angle
		for k in range(n_sk):
			car.set_control(0.72, steer_sk_rad, 0.0)
			await tree.physics_frame
			var p_kw: float = car.tv_elec_power_kw
			if p_kw > 80.05 or p_kw < -25.05:
				sk_hv_bad += 1
			if k >= k_sk_win:
				var inv_b: Basis = car.global_transform.basis.inverse()
				var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
				var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
				var ay_g: float = absf(lv_ros.x * lw_ros.z) / 9.81
				sk_ay_max = maxf(sk_ay_max, ay_g)
				sk_ay_sum += ay_g
				var trqs: Array = car.model.ecu_tv_4wd.last_t_cmd
				for wi in range(4):
					sk_t_sum[wi] += float(trqs[wi])
				var diag: Dictionary = car.get_tv_live_diagnostics()
				sk_regime_last = String(diag.get("active_regime", ""))
				if sk_regime_last == "SKIDPAD_STEADY":
					sk_regime_hits += 1
				sk_n += 1
		var sk_inv := 1.0 / float(maxi(sk_n, 1))
		var sk_ay_avg := sk_ay_sum * sk_inv
		var sk_fl := sk_t_sum[0] * sk_inv
		var sk_fr := sk_t_sum[1] * sk_inv
		var sk_rl := sk_t_sum[2] * sk_inv
		var sk_rr := sk_t_sum[3] * sk_inv
		var sk_in_avg := 0.5 * (sk_fl + sk_rl)
		var sk_out_avg := 0.5 * (sk_fr + sk_rr)
		print("  [TEST4_SKIDPAD_%s] ay_max=%.3fg ay_avg=%.3fg T_avg=[%.1f,%.1f,%.1f,%.1f] regime=%s (%d/%d) hv_bad=%d" % [
			gp_name, sk_ay_max, sk_ay_avg, sk_fl, sk_fr, sk_rl, sk_rr, sk_regime_last, sk_regime_hits, sk_n, sk_hv_bad
		])
		_check(sk_ay_max >= min_ay_g and sk_out_avg > sk_in_avg and sk_in_avg > 0.0 and sk_regime_hits > 0 and sk_hv_bad == 0,
			"Test 4 SKIDPAD %s: ay_max=%.3fg>=%.2fg, T_out=%.1f>T_in=%.1f>0, regime=%s (hv_bad=%d)" % [gp_name, sk_ay_max, min_ay_g, sk_out_avg, sk_in_avg, sk_regime_last, sk_hv_bad], st)
	car.set_track_grip_profile(0)
	car.set_sim_drive_mode(1)  # Restore AUTOCROSS

	# Test 5 -- Slalom 55 km/h & Fast Chicane (max|beta| < 3.2 deg, stabilization < 0.30 s, 0 spins)
	await _reset_and_settle_car(tree, car, open_pose, 15.3)
	car.pilot = MirenaCar.PilotMode.ROS
	var max_abs_beta_deg := 0.0
	var max_abs_wz := 0.0
	var slalom_spun := false
	var slalom_hv_bad := 0
	var slalom_p_min := 1e9
	var slalom_p_max := -1e9
	var n_slalom := int(round(4.0 / dt_phys))
	for k in range(n_slalom):
		var t_s: float = float(k) * dt_phys
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
		var thr_hold: float = clampf(0.18 + 0.15 * (15.3 - lv_ros.x), 0.0, 0.50)
		var steer_cmd: float = 0.6 * car.config.max_steer_angle * sin(TAU * 0.8 * t_s)
		car.set_control(thr_hold, steer_cmd, 0.0)
		await tree.physics_frame
		inv_b = car.global_transform.basis.inverse()
		lv_ros = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		lw_ros = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
		var beta_deg: float = rad_to_deg(atan2(lv_ros.y, maxf(absf(lv_ros.x), 0.5)))
		var abs_wz: float = absf(lw_ros.z)
		max_abs_beta_deg = maxf(max_abs_beta_deg, absf(beta_deg))
		max_abs_wz = maxf(max_abs_wz, abs_wz)
		if absf(beta_deg) > 25.0 or abs_wz > 4.0 or lv_ros.x < 3.0:
			slalom_spun = true
		var p_kw: float = car.tv_elec_power_kw
		slalom_p_min = minf(slalom_p_min, p_kw)
		slalom_p_max = maxf(slalom_p_max, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			slalom_hv_bad += 1

	# Straight release phase (1.0 s)
	var rec_time_s := -1.0
	var settled_streak := 0
	var n_rel := int(round(1.0 / dt_phys))
	for k in range(n_rel):
		car.set_control(0.15, 0.0, 0.0)
		await tree.physics_frame
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
		var beta_deg: float = rad_to_deg(atan2(lv_ros.y, maxf(absf(lv_ros.x), 0.5)))
		var abs_wz: float = absf(lw_ros.z)
		max_abs_beta_deg = maxf(max_abs_beta_deg, absf(beta_deg))
		max_abs_wz = maxf(max_abs_wz, abs_wz)
		if absf(beta_deg) > 25.0 or abs_wz > 4.0 or lv_ros.x < 3.0:
			slalom_spun = true
		var p_kw: float = car.tv_elec_power_kw
		slalom_p_min = minf(slalom_p_min, p_kw)
		slalom_p_max = maxf(slalom_p_max, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			slalom_hv_bad += 1
		if abs_wz < 0.15 and absf(beta_deg) < 1.5:
			settled_streak += 1
			if rec_time_s < 0.0 and settled_streak >= 8:
				rec_time_s = float(k + 1 - 7) * dt_phys
		else:
			settled_streak = 0

	print("  [TEST5_SLALOM] max|beta|=%.2fdeg max|wz|=%.3frad/s spun=%s rec_t=%.3fs P=[%.2f,%.2f]kW hv_bad=%d" % [
		max_abs_beta_deg, max_abs_wz, str(slalom_spun), rec_time_s, slalom_p_min, slalom_p_max, slalom_hv_bad
	])
	_check(max_abs_beta_deg < 3.2 and max_abs_wz < 2.5 and not slalom_spun,
		"Test 5 Slalom 55km/h stable: max|beta|=%.2f deg < 3.2, max|wz|=%.3f < 2.5, spun=%s" % [max_abs_beta_deg, max_abs_wz, str(slalom_spun)], st)
	_check(rec_time_s >= 0.0 and rec_time_s < 0.30 and slalom_hv_bad == 0,
		"Test 5 Slalom straight recovery=%.3f s < 0.30 s, hv_bad=%d" % [rec_time_s, slalom_hv_bad], st)

	# Test 5B -- Fast Chicane (vx = 18.0 m/s, quick L->R reversal + straight release)
	await _reset_and_settle_car(tree, car, open_pose, 18.0)
	car.pilot = MirenaCar.PilotMode.ROS
	var chi_max_beta_deg := 0.0
	var chi_max_wz := 0.0
	var chi_spun := false
	var chi_hv_bad := 0
	var chi_rec_s := -1.0
	var chi_settled := 0
	var n_chi_leg := int(round(0.60 / dt_phys))
	for k in range(2 * n_chi_leg):
		var steer_chi: float = (0.50 if k < n_chi_leg else -0.50) * car.config.max_steer_angle
		car.set_control(0.25, steer_chi, 0.0)
		await tree.physics_frame
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
		var beta_deg: float = rad_to_deg(atan2(lv_ros.y, maxf(absf(lv_ros.x), 0.5)))
		chi_max_beta_deg = maxf(chi_max_beta_deg, absf(beta_deg))
		chi_max_wz = maxf(chi_max_wz, absf(lw_ros.z))
		if absf(beta_deg) > 25.0 or absf(lw_ros.z) > 4.0 or lv_ros.x < 3.0:
			chi_spun = true
		var p_kw: float = car.tv_elec_power_kw
		if p_kw > 80.05 or p_kw < -25.05:
			chi_hv_bad += 1
	for k in range(int(round(1.0 / dt_phys))):
		car.set_control(0.20, 0.0, 0.0)
		await tree.physics_frame
		var inv_b: Basis = car.global_transform.basis.inverse()
		var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
		var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
		var beta_deg: float = rad_to_deg(atan2(lv_ros.y, maxf(absf(lv_ros.x), 0.5)))
		chi_max_beta_deg = maxf(chi_max_beta_deg, absf(beta_deg))
		chi_max_wz = maxf(chi_max_wz, absf(lw_ros.z))
		if absf(beta_deg) > 25.0 or absf(lw_ros.z) > 4.0 or lv_ros.x < 3.0:
			chi_spun = true
		var p_kw: float = car.tv_elec_power_kw
		if p_kw > 80.05 or p_kw < -25.05:
			chi_hv_bad += 1
		if absf(lw_ros.z) < 0.15 and absf(beta_deg) < 1.5:
			chi_settled += 1
			if chi_rec_s < 0.0 and chi_settled >= 8:
				chi_rec_s = float(k + 1 - 7) * dt_phys
		else:
			chi_settled = 0
	print("  [TEST5B_CHICANE] max|beta|=%.2fdeg max|wz|=%.3frad/s spun=%s rec_t=%.3fs hv_bad=%d" % [
		chi_max_beta_deg, chi_max_wz, str(chi_spun), chi_rec_s, chi_hv_bad
	])
	_check(chi_max_beta_deg < 3.2 and not chi_spun and chi_rec_s >= 0.0 and chi_rec_s < 0.30 and chi_hv_bad == 0,
		"Test 5B Fast Chicane: max|beta|=%.2f<3.2 deg, rec_t=%.3f<0.30s, spun=%s (hv_bad=%d)" % [chi_max_beta_deg, chi_rec_s, str(chi_spun), chi_hv_bad], st)

	# Test 4 -- All 5 TV Maps @ 400 Hz Sim.tscn (AUTOCROSS, SKIDPAD, ENDURANCE/EFFICIENCY, ACCELERATION, TIRE_WARMER)
	var map_specs := [
		{"mode": 1, "name": "AUTOCROSS"},
		{"mode": 3, "name": "SKIDPAD"},
		{"mode": 2, "name": "EFFICIENCY"},
		{"mode": 4, "name": "ACCELERATION"},
		{"mode": 5, "name": "TIRE_WARMER"},
	]
	for ms in map_specs:
		var m_id: int = int(ms["mode"])
		var m_name: String = String(ms["name"])
		car.set_sim_drive_mode(m_id)

		# 4a. Launch 0-50 / 0-100 & HV check
		await _reset_and_settle_car(tree, car, open_pose, 0.0)
		car.pilot = MirenaCar.PilotMode.MANUAL
		car.set_input_profile(InputConditionerScript.InputProfile.KEYBOARD)
		Input.action_press("manual_gas_pos", 1.0)
		var m_t50 := -1.0
		var m_t100 := -1.0
		var m_hv_bad := 0
		var m_pmax := -1e9
		var m_pmin := 1e9
		var n_m_launch := int(round(3.15 / dt_phys))
		for k in range(n_m_launch):
			await tree.physics_frame
			var t_s: float = float(k + 1) * dt_phys
			var inv_b: Basis = car.global_transform.basis.inverse()
			var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
			var kmh: float = lv_ros.x * 3.6
			var p_kw: float = car.tv_elec_power_kw
			m_pmax = maxf(m_pmax, p_kw)
			m_pmin = minf(m_pmin, p_kw)
			if p_kw > 80.05 or p_kw < -25.05:
				m_hv_bad += 1
			if m_t50 < 0.0 and kmh >= 50.0:
				m_t50 = t_s
			if m_t100 < 0.0 and kmh >= 100.0:
				m_t100 = t_s
				break
		Input.action_release("manual_gas_pos")

		# 4b. Hairpin entry (LEFT, vx=6.5 m/s, steer=16 deg, thr=0.0)
		await _reset_and_settle_car(tree, car, open_pose, 6.5)
		car.pilot = MirenaCar.PilotMode.ROS
		car.set_control(0.0, steer_hp_rad, 0.0)
		var m_int_min := 1e9
		var m_out_max := -1e9
		for _k in range(int(round(0.30 / dt_phys))):
			await tree.physics_frame
			var p_kw: float = car.tv_elec_power_kw
			if p_kw > 80.05 or p_kw < -25.05:
				m_hv_bad += 1
			var trqs: Array = car.model.ecu_tv_4wd.last_t_cmd
			m_int_min = minf(m_int_min, minf(float(trqs[0]), float(trqs[2])))
			m_out_max = maxf(m_out_max, maxf(float(trqs[1]), float(trqs[3])))

		# 4c. Slalom (2.5 s @ 15.3 m/s)
		await _reset_and_settle_car(tree, car, open_pose, 15.3)
		car.pilot = MirenaCar.PilotMode.ROS
		var m_beta_max := 0.0
		var m_spun := false
		for k in range(int(round(2.5 / dt_phys))):
			var t_s: float = float(k) * dt_phys
			var inv_b: Basis = car.global_transform.basis.inverse()
			var lv_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
			var thr_hold: float = clampf(0.18 + 0.15 * (15.3 - lv_ros.x), 0.0, 0.50)
			var steer_cmd: float = 0.6 * car.config.max_steer_angle * sin(TAU * 0.8 * t_s)
			car.set_control(thr_hold, steer_cmd, 0.0)
			await tree.physics_frame
			inv_b = car.global_transform.basis.inverse()
			lv_ros = VehicleConfigScript.godot_to_ros_vec(inv_b * car.linear_velocity)
			var lw_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_b * car.angular_velocity)
			var beta_deg: float = absf(rad_to_deg(atan2(lv_ros.y, maxf(absf(lv_ros.x), 0.5))))
			m_beta_max = maxf(m_beta_max, beta_deg)
			if beta_deg > 25.0 or absf(lw_ros.z) > 4.0 or lv_ros.x < 3.0:
				m_spun = true
			var p_kw: float = car.tv_elec_power_kw
			if p_kw > 80.05 or p_kw < -25.05:
				m_hv_bad += 1

		print("  [TEST4_MAP_%s] 0-50=%.3fs 0-100=%.3fs hp_int_min=%.1fNm hp_out_max=%.1fNm slalom_max|beta|=%.2fdeg P=[%.2f,%.2f]kW hv_bad=%d" % [
			m_name, m_t50, m_t100, m_int_min, m_out_max, m_beta_max, m_pmin, m_pmax, m_hv_bad
		])
		_check(m_hv_bad == 0, "Test 4 %s: 0 HV violations (hv_bad=%d)" % [m_name, m_hv_bad], st)
		if m_name in ["AUTOCROSS", "SKIDPAD", "ACCELERATION"]:
			_check(m_t50 > 0.0 and m_t50 <= 1.25, "Test 4 %s: 0-50=%.3fs <= 1.25s" % [m_name, m_t50], st)
			_check(m_t100 > 0.0 and m_t100 <= 2.95, "Test 4 %s: 0-100=%.3fs <= 2.95s" % [m_name, m_t100], st)
			_check(m_int_min <= 0.0 and m_out_max > 0.0, "Test 4 %s: hairpin inner regen <= 0 (%.1f) + outer + (%.1f)" % [m_name, m_int_min, m_out_max], st)
			_check(m_beta_max < 3.2 and not m_spun, "Test 4 %s: slalom max|beta|=%.2f deg < 3.2 (spun=%s)" % [m_name, m_beta_max, str(m_spun)], st)
		elif m_name == "EFFICIENCY":
			_check(m_t50 > 0.0 and m_t50 <= 1.25, "Test 4 %s: 0-50=%.3fs <= 1.25s" % [m_name, m_t50], st)
			_check(m_int_min <= 0.0, "Test 4 %s: hairpin inner regen <= 0 (%.1f)" % [m_name, m_int_min], st)
			_check(not m_spun, "Test 4 %s: slalom stable max|beta|=%.2f deg (spun=%s)" % [m_name, m_beta_max, str(m_spun)], st)
		elif m_name == "TIRE_WARMER":
			_check(m_int_min <= 0.0 and m_out_max > 0.0, "Test 4 %s: hairpin inner regen <= 0 (%.1f) + outer + (%.1f)" % [m_name, m_int_min, m_out_max], st)
			_check(m_beta_max < 3.2 and not m_spun, "Test 4 %s: slalom max|beta|=%.2f deg < 3.2 (spun=%s)" % [m_name, m_beta_max, str(m_spun)], st)

	# Cleanup Sim.tscn
	var sim_auto := tree.root.get_node_or_null("Sim")
	if sim_auto != null:
		sim_auto.set("car", null)
		sim_auto.set("track", null)
	sim.queue_free()
	await tree.process_frame
	return st
