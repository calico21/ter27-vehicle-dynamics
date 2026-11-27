extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _wrap_angle(a: float) -> float:
	var w := fmod(a + PI, TAU)
	if w < 0.0:
		w += TAU
	return w - PI


func _to_packed(src: Array) -> PackedFloat64Array:
	var out := PackedFloat64Array()
	out.resize(src.size())
	for i in range(src.size()):
		out[i] = float(src[i])
	return out


func _interp_periodic(arr: PackedFloat64Array, s_m: float, ds: float, track_len: float) -> float:
	var n: int = arr.size()
	if n <= 0:
		return 0.0
	var f_idx: float = fposmod(s_m, track_len) / maxf(ds, 1e-4)
	var i0: int = int(floor(f_idx)) % n
	var i1: int = (i0 + 1) % n
	var frac: float = f_idx - floor(f_idx)
	return lerpf(arr[i0], arr[i1], frac)


func _find_closest_idx(px: float, py: float, xs: PackedFloat64Array, ys: PackedFloat64Array, prev_idx: int) -> int:
	var n: int = xs.size()
	var best_i: int = prev_idx
	var best_d2: float = 1.0e30
	if prev_idx < 0:
		for i in range(n):
			var dx: float = px - xs[i]
			var dy: float = py - ys[i]
			var d2: float = dx * dx + dy * dy
			if d2 < best_d2:
				best_d2 = d2
				best_i = i
		return best_i
	for off in range(-4, 17):
		var i: int = (prev_idx + off + n) % n
		var dx: float = px - xs[i]
		var dy: float = py - ys[i]
		var d2: float = dx * dx + dy * dy
		if d2 < best_d2:
			best_d2 = d2
			best_i = i
	return best_i


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


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		push_error("Usage: godot -s lts_transient_runner.gd -- <job.json> <summary_out.json>")
		quit(1)
		return

	var job: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	var summary_path: String = args[1]
	var csv_path: String = String(job.get("csv_path", ""))

	var base_trk: Dictionary = job["base_track"]
	var race_trk: Dictionary = job["racing_track"]
	var v_qss_arr := _to_packed(job["v_qss"])
	var ax_qss_arr := _to_packed(job["ax_qss"]) if job.has("ax_qss") else PackedFloat64Array()
	var alpha_drv: float = float(job.get("alpha_drv", 0.96))
	var n_laps: int = maxi(1, int(job.get("n_laps", 1)))
	var ecu_mode: int = int(job.get("ecu_driving_mode", 2))
	var tv4_mode: int = int(job.get("tv_4wd_mode", 0))
	var approach_m: float = float(job.get("approach_m", 32.0))

	var bx := _to_packed(base_trk["x"])
	var by := _to_packed(base_trk["y"])
	var bpsi := _to_packed(base_trk["psi"])
	var bs := _to_packed(base_trk["s"])
	var bwl := _to_packed(base_trk["w_left"])
	var bwr := _to_packed(base_trk["w_right"])
	var base_len: float = float(base_trk["length"])

	var rx := _to_packed(race_trk["x"])
	var ry := _to_packed(race_trk["y"])
	var rpsi := _to_packed(race_trk["psi"])
	var rkap := _to_packed(race_trk["kappa"])
	var rs := _to_packed(race_trk["s"])
	var race_len: float = float(race_trk["length"])
	var race_ds: float = float(race_trk["ds"])
	var n_race: int = rx.size()
	if ax_qss_arr.size() != n_race:
		ax_qss_arr.resize(n_race)

	var m := _configure_car(job)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0

	var start_idx: int = (n_race - clampi(int(round(approach_m / maxf(race_ds, 0.1))), 10, n_race / 3)) % n_race
	var v_spawn: float = clampf(alpha_drv * v_qss_arr[start_idx], 6.0, 35.0)
	m.chassis_state.position = Vector3(rx[start_idx], ry[start_idx], m.config.h_cog)
	m.chassis_state.yaw = rpsi[start_idx]
	m.chassis_state.linear_velocity = Vector3(v_spawn, 0.0, 0.0)
	m.chassis_state.angular_velocity = Vector3(0.0, 0.0, v_spawn * rkap[start_idx])
	var rw: float = maxf(m.config.wheel_radius, 0.05)
	for w in m.wheels:
		w.omega = v_spawn / rw
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.ekf_vx = v_spawn
		m.ecu_tv_4wd.ekf_vy = 0.0
		m.ecu_tv_4wd._ekf_init = true

	var csv_f: FileAccess = null
	if csv_path != "":
		csv_f = FileAccess.open(csv_path, FileAccess.WRITE)
		csv_f.store_line("t_s,lap,s_m,x,y,yaw_deg,vx,vy,ax,ay,wz,beta_deg,steer_deg,throttle,brake,p_dc_kw,e_net_kj,e_reg_kj,ey_race_m,ey_base_m,fx0,fx1,fx2,fx3,fy0,fy1,fy2,fy3,fz0,fz1,fz2,fz3,dz0,dz1,dz2,dz3,fspr0,fspr1,fspr2,fspr3,fdmp0,fdmp1,fdmp2,fdmp3,farb0,farb1,farb2,farb3,t_surf_max_c,disc_max_c,soc,v_dc,ekf_vx,mu_hat_min,rls0,mz_int,curb_act,limp_act")

	var dt: float = 0.0025
	var max_time_s: float = float(n_laps) * 120.0 + 25.0
	var max_steps: int = int(round(max_time_s / dt))

	var idx_r: int = start_idx
	var idx_b: int = _find_closest_idx(m.chassis_state.position.x, m.chassis_state.position.y, bx, by, -1)
	var prev_s_r: float = rs[idx_r]
	var lap_count: int = 0
	var lap_cross_t: Array = []
	var lap_start_e_net: Array = []
	var lap_start_e_reg: Array = []

	var t_sim: float = 0.0
	var steer_prev: float = atan(m.config.wheelbase * rkap[start_idx])
	var steer_filt: float = steer_prev
	const STEER_DELAY_STEPS: int = 18  # ~45 ms human reaction delay at 400 Hz (dt=0.0025s)
	var steer_delay_buf: PackedFloat64Array = PackedFloat64Array()
	steer_delay_buf.resize(STEER_DELAY_STEPS)
	for _di in range(STEER_DELAY_STEPS):
		steer_delay_buf[_di] = steer_prev
	var steer_delay_idx: int = 0
	var throttle_prev: float = 0.0
	var brake_prev: float = 0.0
	var v_err_prev: float = 0.0
	var ey_int: float = 0.0
	var brk_slip_scale: float = 1.0
	var e_net_kj: float = 0.0
	var e_reg_kj: float = 0.0

	var max_abs_ey_base: float = 0.0
	var max_abs_ey_race: float = 0.0
	var ey_race_sq_sum: float = 0.0
	var max_abs_beta_deg: float = 0.0
	var off_track_ticks: int = 0
	var cones_hit: int = 0
	var corners_off_track_ticks: int = 0
	var corners_cones_hit: int = 0
	var prev_off_cone_bin: int = -999999
	var prev_corner_cone_bin: int = -999999
	var spin_detected: bool = false
	var nan_detected: bool = false
	var p_dc_max_kw: float = 0.0
	var p_chg_min_kw: float = 0.0
	var torque_fight_kj: float = 0.0
	var half_car_w: float = 0.5 * m.config.track_width_f
	var mbs: float = m.ecu_tv_4wd.mech_brake_start if m.ecu_tv_4wd != null else 0.80
	var t_reg_max: float = m.ecu_tv_4wd.t_regen_max if m.ecu_tv_4wd != null else 150.0
	var p_chg_max: float = m.ecu_tv_4wd.p_max_charge_w if m.ecu_tv_4wd != null else 25000.0
	var eta_reg: float = m.ecu_tv_4wd.eta_regen if m.ecu_tv_4wd != null else 0.88

	var timed_samples: int = 0
	var fz_peak := [0.0, 0.0, 0.0, 0.0]
	var fx_peak := [0.0, 0.0, 0.0, 0.0]
	var fy_peak := [0.0, 0.0, 0.0, 0.0]
	var dz_peak := [0.0, 0.0, 0.0, 0.0]
	var farb_peak := [0.0, 0.0, 0.0, 0.0]
	var fz_sq := [0.0, 0.0, 0.0, 0.0]
	var fx_sq := [0.0, 0.0, 0.0, 0.0]
	var fy_sq := [0.0, 0.0, 0.0, 0.0]
	var dz_sq := [0.0, 0.0, 0.0, 0.0]
	var farb_sq := [0.0, 0.0, 0.0, 0.0]
	var per_lap_stats: Array = []
	var cur_lap_max_beta: float = 0.0
	var cur_lap_max_ey: float = 0.0
	var cur_lap_off: int = 0
	var res: Dictionary = {}

	for step_k in range(max_steps):
		var px: float = m.chassis_state.position.x
		var py: float = m.chassis_state.position.y
		var yaw: float = m.chassis_state.yaw
		var lv: Vector3 = m.chassis_state.linear_velocity
		var av: Vector3 = m.chassis_state.angular_velocity
		var vx: float = lv.x
		var vy: float = lv.y
		var v_abs: float = sqrt(vx * vx + vy * vy)

		if is_nan(px) or is_nan(py) or is_nan(vx) or is_nan(yaw):
			nan_detected = true
			break

		idx_r = _find_closest_idx(px, py, rx, ry, idx_r)
		idx_b = _find_closest_idx(px, py, bx, by, idx_b)

		var tx_r: float = cos(rpsi[idx_r])
		var ty_r: float = sin(rpsi[idx_r])
		var dx_r: float = px - rx[idx_r]
		var dy_r: float = py - ry[idx_r]
		var s_cur_r: float = fposmod(rs[idx_r] + dx_r * tx_r + dy_r * ty_r, race_len)
		var ey_race: float = -dx_r * ty_r + dy_r * tx_r

		var tx_b: float = cos(bpsi[idx_b])
		var ty_b: float = sin(bpsi[idx_b])
		var dx_b: float = px - bx[idx_b]
		var dy_b: float = py - by[idx_b]
		var s_cur_b: float = fposmod(bs[idx_b] + dx_b * tx_b + dy_b * ty_b, base_len)
		var ey_base: float = -dx_b * ty_b + dy_b * tx_b

		if step_k > 4 and prev_s_r > 0.75 * race_len and s_cur_r < 0.25 * race_len:
			var frac_c: float = (race_len - prev_s_r) / maxf((race_len - prev_s_r) + s_cur_r, 1e-4)
			var t_cross: float = t_sim - (1.0 - clampf(frac_c, 0.0, 1.0)) * dt
			lap_cross_t.append(t_cross)
			lap_start_e_net.append(e_net_kj)
			lap_start_e_reg.append(e_reg_kj)
			if lap_count >= 1:
				var prev_t: float = float(lap_cross_t[lap_count - 1])
				var prev_en: float = float(lap_start_e_net[lap_count - 1])
				var prev_er: float = float(lap_start_e_reg[lap_count - 1])
				var min_mu_h := 10.0
				var max_mu_h := 0.0
				var t_s_max := 0.0
				var d_t_max := 0.0
				for j in range(4):
					t_s_max = maxf(t_s_max, m.wheels[j].t_surf_c)
					if m.brake_state != null:
						d_t_max = maxf(d_t_max, float(m.brake_state.disc_temp[j]))
					if m.ecu_tv_4wd != null:
						min_mu_h = minf(min_mu_h, m.ecu_tv_4wd.mu_hat[j])
						max_mu_h = maxf(max_mu_h, m.ecu_tv_4wd.mu_hat[j])
				per_lap_stats.append({
					"lap": lap_count,
					"lap_time_s": t_cross - prev_t,
					"e_net_kj": e_net_kj - prev_en,
					"e_regen_kj": e_reg_kj - prev_er,
					"max_beta_deg": cur_lap_max_beta,
					"max_abs_ey_base_m": cur_lap_max_ey,
					"off_track_ticks": cur_lap_off,
					"t_surf_max_c": t_s_max,
					"disc_max_c": d_t_max,
					"soc": m.hv_state.soc if m.hv_state != null else 0.95,
					"v_dc": m.hv_state.v_dc if m.hv_state != null else 540.0,
					"ekf_vx_err_ms": (m.ecu_tv_4wd.ekf_vx - vx) if m.ecu_tv_4wd != null else 0.0,
					"mu_hat_min": min_mu_h if m.ecu_tv_4wd != null else 1.5,
					"mu_hat_max": max_mu_h if m.ecu_tv_4wd != null else 1.5,
					"mz_int_nm": m.ecu_tv_4wd._mz_int if m.ecu_tv_4wd != null else 0.0,
					"rls_theta_0": m.ecu_tv_4wd.rls_theta[0] if m.ecu_tv_4wd != null else 0.0,
				})
				cur_lap_max_beta = 0.0
				cur_lap_max_ey = 0.0
				cur_lap_off = 0
			lap_count += 1
			if lap_count > n_laps:
				break
		prev_s_r = s_cur_r

		var beta_signed_deg: float = rad_to_deg(atan2(vy, maxf(vx, 0.5)))
		var beta_deg: float = absf(beta_signed_deg)
		if step_k > 40 and (beta_deg > 28.0 or absf(ey_race) > 4.5):
			spin_detected = true
			break
		if lap_count >= 1:
			max_abs_ey_race = maxf(max_abs_ey_race, absf(ey_race))
			ey_race_sq_sum += ey_race * ey_race
			max_abs_ey_base = maxf(max_abs_ey_base, absf(ey_base))
			max_abs_beta_deg = maxf(max_abs_beta_deg, beta_deg)
			cur_lap_max_beta = maxf(cur_lap_max_beta, beta_deg)
			cur_lap_max_ey = maxf(cur_lap_max_ey, absf(ey_base))
			var lim_l: float = bwl[idx_b] - half_car_w
			var lim_r: float = bwr[idx_b] - half_car_w
			if ey_base > lim_l or ey_base < -lim_r:
				off_track_ticks += 1
				cur_lap_off += 1
				var cone_bin: int = int(floor(s_cur_b / 4.0))
				if cone_bin != prev_off_cone_bin:
					cones_hit += 1
					prev_off_cone_bin = cone_bin
			# 4-corner swept path check (w_car = 1.42 m -> half_w = 0.71 m, L_f = 1.15 m, L_r = 0.95 m)
			var xi_b: float = _wrap_angle(yaw - bpsi[idx_b])
			var sin_xi: float = sin(xi_b)
			var cos_xi: float = cos(xi_b)
			var max_lat_c: float = 0.71 * absf(cos_xi) + maxf(1.15 * sin_xi, -0.95 * sin_xi)
			var min_lat_c: float = -0.71 * absf(cos_xi) + minf(1.15 * sin_xi, -0.95 * sin_xi)
			if ey_base + max_lat_c > bwl[idx_b] or ey_base + min_lat_c < -bwr[idx_b]:
				corners_off_track_ticks += 1
				var c_bin: int = int(floor(s_cur_b / 4.0))
				if c_bin != prev_corner_cone_bin:
					corners_cones_hit += 1
					prev_corner_cone_bin = c_bin
			if beta_deg > 22.0:
				spin_detected = true
				break

		# --- Virtual Driver: Humanized Lateral Control (45 ms compensated preview + 2.6 rad/s rate limit) ---
		var d_ff: float = clampf(0.110 * v_abs, 0.70, 2.5)
		var i_ff: int = (idx_r + int(round(d_ff / race_ds))) % n_race
		var k_ff: float = rkap[i_ff]
		var k_us_drv: float = 0.0036
		var delta_ff: float = atan(m.config.wheelbase * k_ff + k_us_drv * v_abs * v_abs * k_ff)

		var d_look: float = clampf(1.2 + 0.13 * v_abs, 1.4, 5.0)
		var i_look: int = (idx_r + int(round(d_look / race_ds))) % n_race
		var alpha_look: float = _wrap_angle(atan2(ry[i_look] - py, rx[i_look] - px) - yaw)
		var e_psi: float = _wrap_angle(rpsi[idx_r] - yaw)
		ey_int = clampf(ey_int + ey_race * dt, -0.22, 0.22)

		var wz_err: float = av.z - v_abs * rkap[idx_r]
		var delta_raw: float = (
			delta_ff
			+ 0.42 * alpha_look
			- atan2(4.4 * ey_race, maxf(v_abs, 2.5))
			- 0.25 * ey_int
			+ 0.50 * e_psi
			- 0.035 * wz_err
		)
		# 45 ms human reaction delay ring buffer + 2.6 rad/s steering wheel rate limit
		var delta_delayed: float = steer_delay_buf[steer_delay_idx]
		steer_delay_buf[steer_delay_idx] = delta_raw
		steer_delay_idx = (steer_delay_idx + 1) % STEER_DELAY_STEPS
		var max_st: float = m.config.max_steer_angle
		var max_dsteer: float = 2.60 * dt
		var delta_cmd: float = clampf(steer_prev + clampf(delta_delayed - steer_prev, -max_dsteer, max_dsteer), -max_st, max_st)
		steer_prev = delta_cmd
		inp.steer_angle = delta_cmd

		# --- Virtual Driver: Aggressive Limit Longitudinal Control (Threshold + Trail-Braking + Early WOT) ---
		var s_loc: float = s_cur_r + clampf(0.055 * v_abs, 0.25, 1.35)
		var v_ref_now: float = alpha_drv * _interp_periodic(v_qss_arr, s_loc, race_ds, race_len)
		var ax_ff: float = alpha_drv * alpha_drv * _interp_periodic(ax_qss_arr, s_loc, race_ds, race_len)
		var v_err: float = v_ref_now - v_abs
		var ax_track: float = ax_ff + 4.2 * v_err

		var d_scan: float = maxf(6.0, (v_abs * v_abs) / 26.0 + 0.12 * v_abs)
		var n_scan: int = int(ceil(d_scan / race_ds))
		var ax_prev_min: float = 999.0
		var j_step: int = 2
		for j in range(2, n_scan + 1, j_step):
			var d_p: float = float(j) * race_ds
			var i_p: int = (idx_r + j) % n_race
			var v_lim_p: float = alpha_drv * v_qss_arr[i_p]
			if v_lim_p < v_abs - 0.30:
				var d_eff: float = maxf(d_p - 0.020 * v_abs, 0.45)
				var ax_req: float = (v_lim_p * v_lim_p - v_abs * v_abs) / (2.0 * d_eff)
				if ax_req < -10.5 and ax_req < ax_prev_min:
					ax_prev_min = ax_req

		var ax_cmd: float = minf(ax_track, ax_prev_min)
		# Use dynamic sideslip beta_dyn (subtracting kinematic Ackermann beta_kin) so hairpins never falsely cut WOT
		var beta_tot_rad: float = atan2(vy, maxf(vx, 0.5))
		var beta_kin_rad: float = atan2(m.config.b_cog * av.z, maxf(vx, 3.0))
		var beta_dyn_abs: float = absf(beta_tot_rad - beta_kin_rad)
		if ax_cmd > 0.0 and beta_dyn_abs > deg_to_rad(6.5):
			ax_cmd *= clampf(1.0 - (beta_dyn_abs - deg_to_rad(6.5)) / deg_to_rad(5.5), 0.20, 1.0)

		var thr_tgt: float = 0.0
		var brk_tgt: float = 0.0
		if ax_cmd >= -0.05:
			brk_slip_scale = minf(1.0, brk_slip_scale + 4.0 * dt)
			var f_aero_roll: float = 0.5 * m.config.air_density * m.config.cd_a * v_abs * v_abs + 45.0
			var max_fx_drv: float = minf(
				4.0 * m.config.motor_peak_torque * m.config.gear_ratio / rw,
				(m.config.power_limit_watts * m.config.drivetrain_efficiency) / maxf(v_abs, 1.0)
			)
			var f_req_drv: float = m.config.mass * maxf(ax_cmd, 0.0) + f_aero_roll
			thr_tgt = clampf(f_req_drv / maxf(max_fx_drv, 200.0), 0.0, 1.0)
		else:
			var a_dec: float = -ax_cmd
			var k_abs_avg: float = 0.25 * (absf(m.wheels[0].kappa) + absf(m.wheels[1].kappa) + absf(m.wheels[2].kappa) + absf(m.wheels[3].kappa))
			if k_abs_avg > 0.145:
				brk_slip_scale = maxf(0.65, brk_slip_scale - 2.2 * dt)
			else:
				brk_slip_scale = minf(1.0, brk_slip_scale + 3.0 * dt)
			if ecu_mode == 2 and mbs > 0.05:
				var f_brk_cmd: float = m.config.mass * a_dec
				var f_reg_cap: float = minf(4.0 * t_reg_max / rw, (p_chg_max / maxf(eta_reg, 0.5)) / maxf(v_abs, 2.0))
				var f_hyd_max: float = maxf(m.config.brake_mu * m.config.mass * m.config.gravity, 200.0)
				if f_brk_cmd <= f_reg_cap:
					brk_tgt = clampf(mbs * (f_brk_cmd / maxf(f_reg_cap, 50.0)), 0.02, mbs) * brk_slip_scale
				else:
					var hyd_frac: float = clampf((f_brk_cmd - f_reg_cap) / f_hyd_max, 0.0, 1.0)
					brk_tgt = mbs + (1.0 - mbs) * (hyd_frac * brk_slip_scale)
			else:
				brk_tgt = clampf(a_dec / maxf(m.config.brake_mu * m.config.gravity, 8.0), 0.02, 1.0) * brk_slip_scale

		if brk_tgt > 0.02:
			thr_tgt = 0.0
		inp.throttle = clampf(throttle_prev + clampf(thr_tgt - throttle_prev, -18.0 * dt, 18.0 * dt), 0.0, 1.0)
		inp.brake = clampf(brake_prev + clampf(brk_tgt - brake_prev, -24.0 * dt, 24.0 * dt), 0.0, 1.0)
		throttle_prev = inp.throttle
		brake_prev = inp.brake

		res = m.step_standalone(inp, dt, 0.0)
		t_sim += dt

		var p_kw: float = m.hv_state.power_kw if m.hv_state != null else 0.0
		if lap_count >= 1:
			e_net_kj += p_kw * dt
			if p_kw < 0.0:
				e_reg_kj += (-p_kw) * dt
			p_dc_max_kw = maxf(p_dc_max_kw, p_kw)
			p_chg_min_kw = minf(p_chg_min_kw, p_kw)
			var tq_i: Array = m.hv_state.inverter_torques if m.hv_state != null else [0.0, 0.0, 0.0, 0.0]
			var has_pos := false
			var has_neg := false
			var abs_t_neg := 0.0
			for j in range(4):
				var tj: float = float(tq_i[j])
				if tj > 2.0:
					has_pos = true
				elif tj < -2.0:
					has_neg = true
					abs_t_neg += absf(tj) * absf(m.wheels[j].omega)
			if has_pos and has_neg:
				torque_fight_kj += 0.001 * abs_t_neg * dt

			var fz_a: Array = res.get("normal_forces", [0.0, 0.0, 0.0, 0.0])
			var wf_a: Array = res.get("forces", {}).get("wheel_forces_ros", [])
			timed_samples += 1
			for j in range(4):
				var fzj: float = float(fz_a[j])
				var fxj: float = float(Vector3(wf_a[j]).x) if wf_a.size() == 4 else m.wheels[j].F_x
				var fyj: float = float(Vector3(wf_a[j]).y) if wf_a.size() == 4 else m.wheels[j].F_y
				var sc: Variant = m.suspension.corners[j]
				var dzj: float = sc.delta_z
				var arbj: float = sc.arb_force
				fz_peak[j] = maxf(fz_peak[j], absf(fzj))
				fx_peak[j] = maxf(fx_peak[j], absf(fxj))
				fy_peak[j] = maxf(fy_peak[j], absf(fyj))
				dz_peak[j] = maxf(dz_peak[j], absf(dzj))
				farb_peak[j] = maxf(farb_peak[j], absf(arbj))
				fz_sq[j] += fzj * fzj
				fx_sq[j] += fxj * fxj
				fy_sq[j] += fyj * fyj
				dz_sq[j] += dzj * dzj
				farb_sq[j] += arbj * arbj

		if csv_f != null and (step_k % 2 == 0) and lap_count >= 1:
			var fz_a: Array = res.get("normal_forces", [0.0, 0.0, 0.0, 0.0])
			var wf_a: Array = res.get("forces", {}).get("wheel_forces_ros", [])
			var ax_m: float = m.net_force_chassis_ros.x / maxf(m.config.mass, 1.0)
			var ay_m: float = m.net_force_chassis_ros.y / maxf(m.config.mass, 1.0)
			var ts_max := 0.0
			var dt_max := 0.0
			for j in range(4):
				ts_max = maxf(ts_max, m.wheels[j].t_surf_c)
				if m.brake_state != null:
					dt_max = maxf(dt_max, float(m.brake_state.disc_temp[j]))
			var min_mh := 1.5
			if m.ecu_tv_4wd != null:
				for j in range(4):
					min_mh = minf(min_mh, m.ecu_tv_4wd.mu_hat[j])
			var curb_i := 1 if (m.ecu_tv_4wd != null and m.ecu_tv_4wd._curb_timer_s > 0.0) else 0
			var limp_i := 1 if (m.ecu_tv_4wd != null and m.ecu_tv_4wd.limp_home_active) else 0
			var c0 = m.suspension.corners[0]
			var c1 = m.suspension.corners[1]
			var c2 = m.suspension.corners[2]
			var c3 = m.suspension.corners[3]
			var fx_o := [Vector3(wf_a[0]).x, Vector3(wf_a[1]).x, Vector3(wf_a[2]).x, Vector3(wf_a[3]).x] if wf_a.size() == 4 else [m.wheels[0].F_x, m.wheels[1].F_x, m.wheels[2].F_x, m.wheels[3].F_x]
			var fy_o := [Vector3(wf_a[0]).y, Vector3(wf_a[1]).y, Vector3(wf_a[2]).y, Vector3(wf_a[3]).y] if wf_a.size() == 4 else [m.wheels[0].F_y, m.wheels[1].F_y, m.wheels[2].F_y, m.wheels[3].F_y]
			csv_f.store_line("%.4f,%d,%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.4f,%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.5f,%.5f,%.5f,%.5f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.5f,%.2f,%.3f,%.3f,%.0f,%.2f,%d,%d" % [
				t_sim - float(lap_cross_t[0]), lap_count, s_cur_r, px, py, rad_to_deg(yaw), vx, vy, ax_m, ay_m, av.z, beta_signed_deg,
				rad_to_deg(inp.steer_angle), inp.throttle, inp.brake, p_kw, e_net_kj, e_reg_kj, ey_race, ey_base,
				fx_o[0], fx_o[1], fx_o[2], fx_o[3], fy_o[0], fy_o[1], fy_o[2], fy_o[3],
				float(fz_a[0]), float(fz_a[1]), float(fz_a[2]), float(fz_a[3]),
				c0.delta_z, c1.delta_z, c2.delta_z, c3.delta_z,
				c0.spring_force, c1.spring_force, c2.spring_force, c3.spring_force,
				c0.damper_force, c1.damper_force, c2.damper_force, c3.damper_force,
				c0.arb_force, c1.arb_force, c2.arb_force, c3.arb_force,
				ts_max, dt_max, (m.hv_state.soc if m.hv_state != null else 0.95), (m.hv_state.v_dc if m.hv_state != null else 540.0),
				(m.ecu_tv_4wd.ekf_vx if m.ecu_tv_4wd != null else vx), min_mh,
				(m.ecu_tv_4wd.rls_theta[0] if m.ecu_tv_4wd != null else 0.0),
				(m.ecu_tv_4wd._mz_int if m.ecu_tv_4wd != null else 0.0), curb_i, limp_i
			])

	if csv_f != null:
		csv_f.close()

	var total_lap_t: float = 0.0
	if lap_cross_t.size() >= n_laps + 1:
		total_lap_t = float(lap_cross_t[n_laps]) - float(lap_cross_t[0])
	var inv_n: float = 1.0 / maxf(float(timed_samples), 1.0)
	var fz_rms := []; var fx_rms := []; var fy_rms := []; var dz_rms := []; var farb_rms := []
	for j in range(4):
		fz_rms.append(sqrt(float(fz_sq[j]) * inv_n))
		fx_rms.append(sqrt(float(fx_sq[j]) * inv_n))
		fy_rms.append(sqrt(float(fy_sq[j]) * inv_n))
		dz_rms.append(sqrt(float(dz_sq[j]) * inv_n))
		farb_rms.append(sqrt(float(farb_sq[j]) * inv_n))

	var completed: bool = (lap_cross_t.size() >= n_laps + 1) and (not spin_detected) and (not nan_detected)
	var valid_lap: bool = completed and (off_track_ticks == 0) and (cones_hit == 0) and (max_abs_beta_deg < 15.0)
	var out_summary := {
		"valid": valid_lap,
		"completed": completed,
		"alpha_drv": alpha_drv,
		"n_laps": n_laps,
		"lap_time_total_s": total_lap_t,
		"mean_lap_time_s": total_lap_t / float(n_laps) if completed else 999.0,
		"per_lap": per_lap_stats,
		"max_abs_ey_base_m": max_abs_ey_base,
		"max_abs_ey_race_m": max_abs_ey_race,
		"rms_ey_race_m": sqrt(ey_race_sq_sum * inv_n),
		"max_abs_beta_deg": max_abs_beta_deg,
		"off_track_ticks": off_track_ticks,
		"cones_hit": cones_hit,
		"corners_off_track_ticks": corners_off_track_ticks,
		"corners_cones_hit": corners_cones_hit,
		"spin_detected": spin_detected,
		"nan_detected": nan_detected,
		"e_net_kj": e_net_kj,
		"e_regen_kj": e_reg_kj,
		"torque_fight_kj": torque_fight_kj,
		"p_dc_max_kw": p_dc_max_kw,
		"p_chg_min_kw": p_chg_min_kw,
		"suspension_loads": {
			"fz_peak_n": fz_peak, "fz_rms_n": fz_rms,
			"fx_peak_n": fx_peak, "fx_rms_n": fx_rms,
			"fy_peak_n": fy_peak, "fy_rms_n": fy_rms,
			"dz_peak_m": dz_peak, "dz_rms_m": dz_rms,
			"farb_peak_n": farb_peak, "farb_rms_n": farb_rms,
		}
	}
	var fs := FileAccess.open(summary_path, FileAccess.WRITE)
	fs.store_string(JSON.stringify(out_summary, "  "))
	fs.close()
	quit(0)
