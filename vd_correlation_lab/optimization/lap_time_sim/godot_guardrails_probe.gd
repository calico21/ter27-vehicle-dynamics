extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _configure_car(cfg_json: Dictionary) -> Full14Dof:
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	m.config.update_derived_geometry()
	m.suspension.sync_from_config(m.config)
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.configure_from_vehicle_config(m.config)
		if cfg_json.has("mode_overrides") and cfg_json["mode_overrides"] is Dictionary:
			var mo: Dictionary = cfg_json["mode_overrides"]
			for mkey in mo.keys():
				if m.ecu_tv_4wd.modes_cfg.has(mkey) and mo[mkey] is Dictionary:
					for pk in (mo[mkey] as Dictionary).keys():
						m.ecu_tv_4wd.modes_cfg[mkey][pk] = mo[mkey][pk]
		m.ecu_tv_4wd._mc_mode = -1
		m.ecu_tv_4wd.reset()
	m.chassis_state.reset(m.config.h_cog)
	return m


func _run_sim(out_path: String, cfg_json: Dictionary, v0: float, dur: float, ecu_mode: int, tv4_mode: int, ctrl_fn: Callable) -> void:
	var m := _configure_car(cfg_json)
	m.reset()
	m.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	var rw: float = maxf(m.config.wheel_radius, 0.05)
	for w in m.wheels:
		w.omega = v0 / rw
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.ekf_vx = v0
		m.ecu_tv_4wd.ekf_vy = 0.0
		m.ecu_tv_4wd._ekf_init = true

	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0

	var dt: float = 0.0025
	var n_steps: int = int(ceil(dur / dt))
	var f := FileAccess.open(out_path, FileAccess.WRITE)
	f.store_line("t_s,vx,vy,wz,ax,ay,beta_deg,steer_deg,p_dc_kw")

	for step_i in range(n_steps):
		var t: float = float(step_i) * dt
		ctrl_fn.call(t, m, inp)
		var res := m.step_standalone(inp, dt, 0.0)
		if step_i % 2 == 0:
			var vx: float = m.chassis_state.linear_velocity.x
			var vy: float = m.chassis_state.linear_velocity.y
			var wz: float = m.chassis_state.angular_velocity.z
			var nf: Vector3 = res["net_force_ros"]
			var ax: float = nf.x / m.config.mass
			var ay: float = nf.y / m.config.mass
			var beta_deg: float = rad_to_deg(atan2(vy, maxf(absf(vx), 0.5)))
			var p_kw: float = (m.ecu_tv_4wd.last_power_w * 0.001) if m.ecu_tv_4wd != null else 0.0
			f.store_line("%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f" % [
				t, vx, vy, wz, ax, ay, beta_deg, rad_to_deg(inp.steer_angle), p_kw
			])
	f.close()


func _init() -> void:
	var cfg_path := ""
	var out_dir := ""
	var tasks_str := "skidpad,dlc20,bode"
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--cfg="):
			cfg_path = arg.substr(6)
		elif arg.begins_with("--out_dir="):
			out_dir = arg.substr(10)
		elif arg.begins_with("--tasks="):
			tasks_str = arg.substr(8)

	var cfg_json: Dictionary = {}
	if cfg_path != "" and FileAccess.file_exists(cfg_path):
		var txt := FileAccess.get_file_as_string(cfg_path)
		var parsed = JSON.parse_string(txt)
		if parsed is Dictionary:
			cfg_json = parsed

	var tasks := tasks_str.split(",")
	if tasks.has("skidpad"):
		_run_sim(out_dir + "/skidpad.csv", cfg_json, 10.0, 14.0, 2, 1, func(t: float, m: Full14Dof, inp: VehicleInputs) -> void:
			var sgn := -1.0 if t < 6.5 else 1.0
			var v_tgt := 11.2
			var err_v := v_tgt - m.chassis_state.linear_velocity.x
			inp.throttle = clampf(0.28 + 0.25 * err_v, 0.0, 0.85)
			inp.brake = 0.0
			var wz_tgt := sgn * m.chassis_state.linear_velocity.x / 9.125
			inp.steer_angle = clampf(sgn * deg_to_rad(9.6) + 0.18 * (wz_tgt - m.chassis_state.angular_velocity.z), -0.35, 0.35)
		)
	if tasks.has("dlc20"):
		_run_sim(out_dir + "/dlc20.csv", cfg_json, 20.0, 4.5, 2, 0, func(t: float, _m: Full14Dof, inp: VehicleInputs) -> void:
			inp.throttle = 0.22
			inp.brake = 0.0
			var d := 0.0
			if t >= 0.4 and t < 1.1:
				d = deg_to_rad(6.5) * sin(PI * (t - 0.4) / 0.7)
			elif t >= 1.1 and t < 2.1:
				d = -deg_to_rad(8.5) * sin(PI * (t - 1.1) / 1.0)
			elif t >= 2.1 and t < 2.9:
				d = deg_to_rad(5.5) * sin(PI * (t - 2.1) / 0.8)
			inp.steer_angle = d
		)
	if tasks.has("bode"):
		_run_sim(out_dir + "/bode.csv", cfg_json, 16.0, 8.0, 2, 0, func(t: float, _m: Full14Dof, inp: VehicleInputs) -> void:
			inp.throttle = 0.25
			inp.brake = 0.0
			var ph := TAU * (0.2 * t + 0.5 * (3.3 / 8.0) * t * t)
			inp.steer_angle = deg_to_rad(5.0) * sin(ph)
		)
	quit()
