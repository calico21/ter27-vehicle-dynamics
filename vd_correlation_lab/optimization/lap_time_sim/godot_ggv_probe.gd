extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _configure_car(cfg_json: Dictionary) -> Full14Dof:
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	if cfg_json.has("use_3d_kinematics"):
		m.config.use_3d_kinematics = bool(cfg_json["use_3d_kinematics"])
	if cfg_json.has("mass_kg"):
		m.config.mass = float(cfg_json["mass_kg"])
	if cfg_json.has("cl_a_scale"):
		m.config._has_explicit_axle_cl = false
		m.config.cl_a *= float(cfg_json["cl_a_scale"])
	elif cfg_json.has("cl_a"):
		m.config._has_explicit_axle_cl = false
		m.config.cl_a = float(cfg_json["cl_a"])
	m.config.update_derived_geometry()
	m.suspension.sync_from_config(m.config)
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.configure_from_vehicle_config(m.config)
		if cfg_json.has("mech_brake_start"):
			m.ecu_tv_4wd.mech_brake_start = float(cfg_json["mech_brake_start"])
		if cfg_json.has("p_eff_max_w") and m.ecu_tv_4wd.modes_cfg.has("efficiency"):
			m.ecu_tv_4wd.modes_cfg["efficiency"]["p_eff_max_w"] = float(cfg_json["p_eff_max_w"])
		if cfg_json.has("mode_overrides") and cfg_json["mode_overrides"] is Dictionary:
			var mo: Dictionary = cfg_json["mode_overrides"]
			for mkey in mo.keys():
				if m.ecu_tv_4wd.modes_cfg.has(mkey) and mo[mkey] is Dictionary:
					for pk in (mo[mkey] as Dictionary).keys():
						m.ecu_tv_4wd.modes_cfg[mkey][pk] = mo[mkey][pk]
		m.ecu_tv_4wd._mc_mode = -1
		m.ecu_tv_4wd.reset()
	if m.ecu_tv_v3 != null:
		m.ecu_tv_v3.configure_from_vehicle_config(m.config)
		m.ecu_tv_v3.reset()
	m.chassis_state.reset(m.config.h_cog)
	if cfg_json.has("grip_profile") and String(cfg_json["grip_profile"]) == "fsg_real":
		m.set_track_grip_profile(1)
	elif cfg_json.has("track_grip_profile"):
		m.set_track_grip_profile(int(cfg_json["track_grip_profile"]))
	return m


func _init_moving_car(m: Full14Dof, v0: float) -> void:
	m.reset()
	m.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	var rw: float = maxf(m.config.wheel_radius, 0.05)
	for w in m.wheels:
		w.omega = v0 / rw
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.ekf_vx = v0
		m.ecu_tv_4wd.ekf_vy = 0.0
		m.ecu_tv_4wd._ekf_init = true


func _probe_ay_max(m: Full14Dof, v_tgt: float, ecu_mode: int, tv4_mode: int) -> Dictionary:
	_init_moving_car(m, v_tgt)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0

	var dt: float = 0.0025
	# Phase 1: settle aero ride height for 0.25 s straight
	for _k in range(100):
		var err_v0: float = v_tgt - m.chassis_state.linear_velocity.x
		inp.throttle = clampf(0.18 + 0.35 * err_v0, 0.0, 0.8)
		inp.brake = 0.0
		inp.steer_angle = 0.0
		m.step_standalone(inp, dt, 0.0)

	# Phase 2: slow steer sweep up to max_steer_angle while holding speed ~ v_tgt
	var n_steps: int = 1400  # 3.5 s sweep
	var steer_max: float = minf(m.config.max_steer_angle, 0.38)
	var ay_best: float = 0.0
	var steer_best: float = 0.0
	var beta_at_best: float = 0.0
	var p_at_best: float = 0.0
	var ay_win_sum: float = 0.0
	var p_win_sum: float = 0.0
	var win_cnt: int = 0
	var ay_prev_win: float = -1.0

	for k in range(n_steps):
		var frac: float = clampf(float(k) / float(n_steps - 1), 0.0, 1.0)
		var st_cmd: float = steer_max * frac
		var lv: Vector3 = m.chassis_state.linear_velocity
		var v_err: float = v_tgt - lv.x
		inp.steer_angle = st_cmd
		if v_err >= -0.15:
			inp.throttle = clampf(0.25 + 1.35 * v_err, 0.0, 1.0)
			inp.brake = 0.0
		else:
			inp.throttle = 0.0
			inp.brake = clampf(-0.6 * v_err, 0.0, 0.85)

		m.step_standalone(inp, dt, 0.0)
		var ay_cur: float = m.net_force_chassis_ros.y / maxf(m.config.mass, 1.0)
		var beta_deg: float = absf(rad_to_deg(atan2(m.chassis_state.linear_velocity.y, maxf(m.chassis_state.linear_velocity.x, 0.5))))
		ay_win_sum += ay_cur
		p_win_sum += (m.hv_state.power_kw if m.hv_state != null else 0.0)
		win_cnt += 1

		if win_cnt == 16:  # 40 ms averaging window
			var ay_avg: float = ay_win_sum / float(win_cnt)
			var p_avg: float = p_win_sum / float(win_cnt)
			var dv_ok: bool = absf(m.chassis_state.linear_velocity.x - v_tgt) < maxf(1.2, 0.08 * v_tgt)
			var beta_lim: float = 5.5 if v_tgt < 8.0 else 4.9
			if beta_deg < beta_lim and dv_ok and ay_avg > ay_best:
				ay_best = ay_avg
				steer_best = st_cmd
				beta_at_best = beta_deg
				p_at_best = p_avg
			# Stop sweep cleanly if saturated and falling off with high slip
			if ay_prev_win > 0.0 and beta_deg > 7.5 and ay_avg < 0.90 * ay_best:
				break
			ay_prev_win = ay_avg
			ay_win_sum = 0.0
			p_win_sum = 0.0
			win_cnt = 0

	return {
		"v_ms": v_tgt,
		"ay_max": ay_best,
		"steer_at_max": steer_best,
		"beta_deg": beta_at_best,
		"p_dc_kw": p_at_best,
	}


func _probe_longitudinal(
	m: Full14Dof,
	v_tgt: float,
	steer_hold: float,
	ecu_mode: int,
	tv4_mode: int,
	is_braking: bool
) -> Dictionary:
	# Slightly lead initial speed so measurement around steady transfer crosses v_tgt
	var v_start: float = clampf(v_tgt + (0.6 if is_braking else -0.5), 4.0, 42.0)
	_init_moving_car(m, v_start)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0

	var dt: float = 0.0025
	# Settle cornering attitude + aero + weight transfer for 0.45 s if steered, 0.16 s if straight
	var n_settle: int = 180 if absf(steer_hold) > 1e-4 else 64
	for k in range(n_settle):
		var err_v: float = v_start - m.chassis_state.linear_velocity.x
		inp.steer_angle = steer_hold * clampf(float(k + 1) / 40.0, 0.0, 1.0)
		inp.throttle = clampf(0.20 + 0.50 * err_v, 0.0, 1.0)
		inp.brake = 0.0
		m.step_standalone(inp, dt, 0.0)

	# Apply full traction or threshold braking for 0.16 s and average stable plateau [0.06..0.14 s]
	var ax_sum: float = 0.0
	var ay_sum: float = 0.0
	var p_sum: float = 0.0
	var cnt: int = 0
	var max_beta: float = 0.0
	var b_start: float = 0.97 if ecu_mode == 2 else 0.90
	if absf(steer_hold) > 0.05:
		b_start = 0.92 if ecu_mode == 2 else 0.80
	var brk_cmd: float = b_start

	for k in range(64):  # 0.16 s
		inp.steer_angle = steer_hold
		if not is_braking:
			inp.throttle = 1.0
			inp.brake = 0.0
		else:
			inp.throttle = 0.0
			# Average loaded-axle braking slip (weighted by outer wheels) to avoid clamping on 1 unloaded inside corner
			var k_sum: float = 0.0
			for i_w in range(4):
				k_sum += absf(m.wheels[i_w].kappa)
			var k_avg: float = 0.25 * k_sum
			if k_avg > 0.16:
				brk_cmd -= 0.015
			elif k_avg < 0.10:
				brk_cmd += 0.015
			var b_min: float = 0.72 if ecu_mode == 2 else 0.30
			brk_cmd = clampf(brk_cmd, b_min, 1.0)
			inp.brake = brk_cmd

		m.step_standalone(inp, dt, 0.0)
		var lv: Vector3 = m.chassis_state.linear_velocity
		var beta_deg: float = absf(rad_to_deg(atan2(lv.y, maxf(lv.x, 0.5))))
		if k >= 24 and k < 56:  # 0.06..0.14 s window (after inverter tau=8ms & weight transfer)
			ax_sum += m.net_force_chassis_ros.x / maxf(m.config.mass, 1.0)
			ay_sum += m.net_force_chassis_ros.y / maxf(m.config.mass, 1.0)
			p_sum += (m.hv_state.power_kw if m.hv_state != null else 0.0)
			cnt += 1
			max_beta = maxf(max_beta, beta_deg)

	return {
		"ax": ax_sum / maxf(float(cnt), 1.0),
		"ay": ay_sum / maxf(float(cnt), 1.0),
		"p_dc_kw": p_sum / maxf(float(cnt), 1.0),
		"max_beta_deg": max_beta,
	}


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		push_error("Usage: godot -s godot_ggv_probe.gd -- <cfg.json> <out.json>")
		quit(1)
		return

	var cfg_text := FileAccess.get_file_as_string(args[0])
	var cfg_json: Dictionary = JSON.parse_string(cfg_text)
	var out_path: String = args[1]

	var ecu_mode: int = int(cfg_json.get("ecu_driving_mode", 2))
	var tv4_mode: int = int(cfg_json.get("tv_4wd_mode", 0))
	var speeds: Array = cfg_json.get("speeds_ms", [6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0])
	var ay_fracs: Array = cfg_json.get("ay_fracs", [0.0, 0.35, 0.65, 0.85, 0.96])

	var m := _configure_car(cfg_json)
	var rows: Array = []

	for v_val in speeds:
		var v: float = float(v_val)
		var ay_res := _probe_ay_max(m, v, ecu_mode, tv4_mode)
		var st_max: float = float(ay_res["steer_at_max"])
		var ax_max_list: Array = []
		var ax_min_list: Array = []
		var p_acc_list: Array = []
		var p_brk_list: Array = []
		var ay_acc_list: Array = []
		var ay_brk_list: Array = []

		for f_val in ay_fracs:
			var frac: float = float(f_val)
			var st_hold: float = st_max * frac
			var acc_r := _probe_longitudinal(m, v, st_hold, ecu_mode, tv4_mode, false)
			var brk_r := _probe_longitudinal(m, v, st_hold, ecu_mode, tv4_mode, true)
			ax_max_list.append(float(acc_r["ax"]))
			ax_min_list.append(float(brk_r["ax"]))
			p_acc_list.append(float(acc_r["p_dc_kw"]))
			p_brk_list.append(float(brk_r["p_dc_kw"]))
			ay_acc_list.append(float(acc_r["ay"]))
			ay_brk_list.append(float(brk_r["ay"]))

		rows.append({
			"v_ms": v,
			"ay_max": float(ay_res["ay_max"]),
			"steer_at_max": st_max,
			"beta_at_ay_max_deg": float(ay_res["beta_deg"]),
			"p_ay_max_kw": float(ay_res["p_dc_kw"]),
			"ay_fracs": ay_fracs,
			"ax_max": ax_max_list,
			"ax_min": ax_min_list,
			"p_acc_kw": p_acc_list,
			"p_brk_kw": p_brk_list,
			"ay_acc": ay_acc_list,
			"ay_brk": ay_brk_list,
		})

	var out_dict := {
		"config": cfg_json,
		"mass_kg": m.config.mass,
		"cl_a": m.config.cl_a,
		"cd_a": m.config.cd_a,
		"use_3d_kinematics": m.suspension.is_3d_active(m.config),
		"speeds_ms": speeds,
		"ay_fracs": ay_fracs,
		"rows": rows,
	}
	var f_out := FileAccess.open(out_path, FileAccess.WRITE)
	f_out.store_string(JSON.stringify(out_dict, "  "))
	f_out.close()
	print("GGV probe completed -> ", out_path)
	quit(0)
