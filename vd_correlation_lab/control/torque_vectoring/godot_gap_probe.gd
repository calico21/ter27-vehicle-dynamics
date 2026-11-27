extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const ChassisDynamics = preload("res://Scenes/Vehicle/Dynamics/Chassis/ChassisDynamics.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const EcuTv4Wd = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoring4Wd.gd")

var _out_path: String = "/tmp/gap_probe_out.json"
var _cfg_path: String = ""
var _ablation_mask: int = 0
var _ecu_timings_us: Array[float] = []
var _hv_viol_hi: int = 0
var _hv_viol_lo: int = 0
var _p_dc_max_kw: float = -1e9
var _p_dc_min_kw: float = 1e9


static func _wrap_pi(a: float) -> float:
	var x := fmod(a + PI, TAU)
	if x < 0.0:
		x += TAU
	return x - PI


func _new_car(v0: float, grip_profile: int, tv4_mode: int) -> Dictionary:
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	m.set_track_grip_profile(grip_profile)
	if _cfg_path != "" and _cfg_path != "default":
		m.ecu_tv_4wd.load_config_json(_cfg_path)
		m.ecu_tv_4wd.configure_from_vehicle_config(m.config)
		m.ecu_tv_4wd.reset()
	m.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	for w in m.wheels:
		w.omega = v0 / maxf(m.config.wheel_radius, 0.1)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0
	return {"m": m, "inp": inp}


func _step_car(m: Full14Dof, inp: VehicleInputs, dt: float, record_timing: bool = true) -> Dictionary:
	var s0 := m.chassis_state.clone()
	var w0: Array[Vector4] = [
		m.wheels[0].get_integrator_state(),
		m.wheels[1].get_integrator_state(),
		m.wheels[2].get_integrator_state(),
		m.wheels[3].get_integrator_state()
	]
	var c1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, m.config, 0.0)
	var t0 := Time.get_ticks_usec()
	var r1: Dictionary = m.step_forces(inp, s0.linear_velocity, s0.angular_velocity, c1, dt, true)
	var t1 := Time.get_ticks_usec()
	if record_timing:
		_ecu_timings_us.append(float(t1 - t0))
	var k1 := ChassisDynamics.compute_derivatives(s0, r1["net_force_ros"], r1["net_torque_ros"], m.config)
	var s_pred := ChassisDynamics.advance_euler(s0, k1, dt)

	var c2: Array = ChassisDynamics.compute_ground_contacts_6dof(s_pred, m.config, 0.0)
	var r2: Dictionary = m.step_forces(inp, s_pred.linear_velocity, s_pred.angular_velocity, c2, dt, false)
	var k2 := ChassisDynamics.compute_derivatives(s_pred, r2["net_force_ros"], r2["net_torque_ros"], m.config)
	m.chassis_state = ChassisDynamics.advance_heun(s0, k1, k2, dt)
	for i in range(4):
		var w2 := m.wheels[i].get_integrator_state()
		var wh := 0.5 * (w0[i] + w2)
		if is_zero_approx(w2.x) and inp.brake > 0.0 and inp.throttle == 0.0:
			wh.x = 0.0
		m.wheels[i].set_integrator_state(wh)

	var p_kw := m.hv_state.power_kw
	_p_dc_max_kw = maxf(_p_dc_max_kw, p_kw)
	_p_dc_min_kw = minf(_p_dc_min_kw, p_kw)
	if p_kw > 80.05:
		_hv_viol_hi += 1
	if p_kw < -25.05:
		_hv_viol_lo += 1
	return r2


func _run_stationary_corner(radius: float, grip_profile: int, tv4_mode: int) -> Dictionary:
	# Quasi-static circular ramp around center (0, R) starting at (0, 0), yaw = 0 (CCW left turn)
	var mu_nom := 1.25 if grip_profile == Full14Dof.TrackGripProfile.FSG_REAL_TRACK else 1.50
	var v_frac := 0.65 if radius < 12.0 else 0.78
	var v_start := clampf(v_frac * sqrt(mu_nom * 9.81 * radius), 4.8, 16.5)
	var car := _new_car(v_start, grip_profile, tv4_mode)
	var m: Full14Dof = car["m"]
	var inp: VehicleInputs = car["inp"]
	# Initialize on-circle yaw rate so the vehicle settles cleanly without initial departure
	m.chassis_state.angular_velocity = Vector3(0.0, 0.0, v_start / radius)

	var dt_ctrl := 0.005
	var wb := m.config.wheelbase
	var max_steer := m.config.max_steer_angle
	var delta_ff := atan(wb / radius)
	var delta_cur := delta_ff
	inp.steer_angle = delta_cur
	var dur := 18.0
	var n_steps := int(round(dur / dt_ctrl))

	var prev_e_r := 0.0
	var best_vx := 0.0
	var best_ay := 0.0
	var best_snap := {}

	# Rolling 0.40s window (80 ticks) for steady-state verification
	var win_len := 80
	var vx_ring: Array[float] = []
	var ay_ring: Array[float] = []
	var er_ring: Array[float] = []
	var b_ring: Array[float] = []
	var ax_ring: Array[float] = []

	for k in range(n_steps):
		var t := float(k) * dt_ctrl
		var pos := m.chassis_state.position
		var lv := m.chassis_state.linear_velocity
		var yaw := m.chassis_state.yaw

		var dx := pos.x
		var dy := pos.y - radius
		var r_cur := sqrt(dx * dx + dy * dy)
		var e_r := r_cur - radius
		var e_r_dot := (e_r - prev_e_r) / dt_ctrl if k > 0 else 0.0
		prev_e_r = e_r

		var phi := atan2(dy, dx)
		var psi_tan := phi + 0.5 * PI
		var e_psi := _wrap_pi(psi_tan - yaw)

		# Smooth Stanley-like circular tracker with rate limit so driver never slams past tire peak
		var vx_s := maxf(lv.x, 3.0)
		var k_psi := 1.20
		var k_cte := 1.65
		var delta_raw := clampf(delta_ff + k_psi * e_psi + atan2(k_cte * e_r + 0.25 * e_r_dot, vx_s), -max_steer, max_steer)
		var max_d_step := 1.4 * dt_ctrl
		delta_cur = clampf(delta_raw, delta_cur - max_d_step, delta_cur + max_d_step)
		inp.steer_angle = delta_cur

		# Warmup 1.6s at v_start, then gentle quasi-static ramp +0.45 m/s^2
		var v_tgt := v_start + (0.45 * maxf(t - 1.6, 0.0))
		var v_err := v_tgt - lv.x
		inp.throttle = clampf(0.18 + 0.40 * v_err, 0.0, 1.0)
		inp.brake = clampf(-0.25 * v_err, 0.0, 0.30) if v_err < -0.8 else 0.0

		var res := {}
		for _s in range(2):
			res = _step_car(m, inp, dt_ctrl * 0.5, true)

		var vx := m.chassis_state.linear_velocity.x
		var vy := m.chassis_state.linear_velocity.y
		var wz := m.chassis_state.angular_velocity.z
		var beta_deg := rad_to_deg(atan2(vy, maxf(vx, 0.5)))
		var v_tot2 := vx * vx + vy * vy
		var ay_geom := v_tot2 / maxf(r_cur, 1.0)
		var ax_cur := m.net_force_chassis_ros.x / maxf(m.config.mass, 100.0)

		vx_ring.append(vx)
		ay_ring.append(ay_geom)
		er_ring.append(absf(e_r))
		b_ring.append(absf(beta_deg))
		ax_ring.append(absf(ax_cur))
		if vx_ring.size() > win_len:
			vx_ring.pop_front()
			ay_ring.pop_front()
			er_ring.pop_front()
			b_ring.pop_front()
			ax_ring.pop_front()

		if t > 2.2 and vx_ring.size() == win_len:
			var max_er := 0.0
			var max_b := 0.0
			var max_ax := 0.0
			var sum_vx := 0.0
			var sum_ay := 0.0
			for j in range(win_len):
				max_er = maxf(max_er, er_ring[j])
				max_b = maxf(max_b, b_ring[j])
				max_ax = maxf(max_ax, ax_ring[j])
				sum_vx += vx_ring[j]
				sum_ay += ay_ring[j]
			var avg_vx := sum_vx / float(win_len)
			var avg_ay := sum_ay / float(win_len)
			var r_tol := clampf(0.045 * radius, 0.30, 0.70)
			if max_er < r_tol and max_b < 11.5 and max_ax < 1.25 and avg_ay > best_ay:
				best_vx = avg_vx
				best_ay = avg_ay
				var tq := m.hv_state.inverter_torques
				var fz_a: Array = res.get("normal_forces", [0.0, 0.0, 0.0, 0.0])
				var mz_tv := 0.0
				for i in range(4):
					mz_tv += m.ecu_tv_4wd._qp_bmz[i] * float(tq[i])
				best_snap = {
					"vx_ms": avg_vx,
					"ay_ms2": avg_ay,
					"ay_g": avg_ay / 9.81,
					"delta_deg": rad_to_deg(inp.steer_angle),
					"beta_deg": beta_deg,
					"wz_rads": wz,
					"alpha_f_deg": 0.5 * rad_to_deg(m.wheels[0].alpha + m.wheels[1].alpha),
					"alpha_r_deg": 0.5 * rad_to_deg(m.wheels[2].alpha + m.wheels[3].alpha),
					"T_cmd": [float(tq[0]), float(tq[1]), float(tq[2]), float(tq[3])],
					"Fz": [float(fz_a[0]), float(fz_a[1]), float(fz_a[2]), float(fz_a[3])],
					"Fx": [m.wheels[0].F_x, m.wheels[1].F_x, m.wheels[2].F_x, m.wheels[3].F_x],
					"Fy": [m.wheels[0].F_y, m.wheels[1].F_y, m.wheels[2].F_y, m.wheels[3].F_y],
					"kappa": [m.wheels[0].kappa, m.wheels[1].kappa, m.wheels[2].kappa, m.wheels[3].kappa],
					"mz_tv_nm": mz_tv,
					"mz_req_nm": m.ecu_tv_4wd.last_mz_req,
					"p_dc_kw": m.hv_state.power_kw,
					"r_err_m": max_er,
					"throttle": inp.throttle,
				}

		# Early stop if vehicle departed circle or spun
		if t > 2.5 and (absf(e_r) > maxf(1.5, 0.14 * radius) or absf(beta_deg) > 25.0 or vx < 2.0):
			break

	return best_snap


func _run_hairpin_180(grip_profile: int) -> Dictionary:
	# 180 deg hairpin matching Oracle OCP: 22m entry straight from x=-22 at v0=20 m/s -> 180 deg CCW R=5.5m arc -> 40m exit straight
	var R := 5.5
	var L_entry := 22.0
	var L_arc := PI * R
	var L_exit := 40.0
	var s_total := L_entry + L_arc + L_exit

	var car := _new_car(20.0, grip_profile, EcuTv4Wd.TestMode.AUTOCROSS)
	var m: Full14Dof = car["m"]
	var inp: VehicleInputs = car["inp"]
	m.chassis_state.position = Vector3(-L_entry, 0.0, m.config.h_cog)
	m.chassis_state.yaw = 0.0

	var mu_nom := 1.25 if grip_profile == Full14Dof.TrackGripProfile.FSG_REAL_TRACK else 1.50
	var v_apex_tgt := sqrt(0.92 * mu_nom * 9.81 * R)
	var a_brk_plan := 0.85 * mu_nom * 9.81
	var wb := m.config.wheelbase
	var max_steer := m.config.max_steer_angle
	var dt_ctrl := 0.005
	var max_dur := 12.0
	var n_steps := int(round(max_dur / dt_ctrl))

	var t_finish := -1.0
	var t_turn_start := -1.0
	var t_turn_end := -1.0
	var v_min := 1e9
	var max_cte := 0.0
	var max_beta := 0.0
	var p_max := -1e9
	var p_min := 1e9
	var hv_bad := 0
	var spun := false
	var s_prev := 0.0
	var prev_cte := 0.0
	var sum_t_in_apex := 0.0
	var sum_t_out_apex := 0.0
	var n_apex := 0

	for k in range(n_steps):
		var t := float(k) * dt_ctrl
		var pos := m.chassis_state.position
		var lv := m.chassis_state.linear_velocity
		var av := m.chassis_state.angular_velocity
		var yaw := m.chassis_state.yaw
		var vx := lv.x

		# Project car onto centerline (entry straight, semicircle, exit straight)
		var s_cur := s_prev
		var cte := 0.0
		var psi_ref := 0.0
		var curv_ref := 0.0

		if s_prev < L_entry - 0.3 and pos.x <= 0.0 and pos.y < R:
			s_cur = clampf(pos.x + L_entry, 0.0, L_entry)
			cte = -pos.y
			psi_ref = 0.0
			curv_ref = 0.0
		elif s_prev < L_entry + L_arc - 0.3 and (pos.x > -0.3 or pos.y < 2.0 * R - 0.5):
			var dx := pos.x
			var dy := pos.y - R
			var phi := atan2(dy, dx)  # -PI/2 at entry (0,0), 0 at apex (R,R), +PI/2 at exit (0,2R)
			var arc_ang := clampf(phi + 0.5 * PI, 0.0, PI)
			s_cur = L_entry + R * arc_ang
			var r_cur := sqrt(dx * dx + dy * dy)
			cte = r_cur - R
			psi_ref = phi + 0.5 * PI
			curv_ref = 1.0 / R
		else:
			s_cur = L_entry + L_arc + clampf(-pos.x, 0.0, L_exit + 5.0)
			cte = pos.y - 2.0 * R
			psi_ref = PI
			curv_ref = 0.0

		s_prev = maxf(s_prev, s_cur)
		if t_turn_start < 0.0 and s_cur >= L_entry:
			t_turn_start = t
		if t_turn_end < 0.0 and s_cur >= L_entry + L_arc:
			t_turn_end = t

		# Lookahead curvature blending for smooth turn-in and exit unwind
		var s_la := s_cur + clampf(0.14 * vx, 0.9, 2.2)
		var curv_la := (1.0 / R) if (s_la >= L_entry and s_la <= L_entry + L_arc) else 0.0
		var curv_eff := 0.60 * curv_ref + 0.40 * curv_la

		var e_psi := _wrap_pi(psi_ref - yaw)
		var cte_dot := (cte - prev_cte) / dt_ctrl if k > 0 else 0.0
		prev_cte = cte
		var delta_ff := atan(wb * curv_eff)
		var vx_s := maxf(vx, 3.0)
		var delta_cmd := clampf(delta_ff + 1.45 * e_psi + atan2(2.2 * cte + 0.35 * cte_dot, vx_s), -max_steer, max_steer)
		inp.steer_angle = delta_cmd

		# Longitudinal speed profile: strong threshold braking into turn-in, trail-brake to apex, full throttle exit
		var v_ref := 35.0
		if s_cur < L_entry + 0.35 * L_arc:
			var dist_to_turn := maxf((L_entry - 1.5) - s_cur, 0.0)
			v_ref = sqrt(v_apex_tgt * v_apex_tgt + 2.0 * a_brk_plan * dist_to_turn)
		elif s_cur < L_entry + 0.70 * L_arc:
			# Gently modulate speed in mid-corner if running wide
			var cte_cut := clampf(1.0 - 0.45 * maxf(cte - 0.10, 0.0), 0.65, 1.0)
			v_ref = v_apex_tgt * 1.02 * cte_cut
		else:
			v_ref = 35.0

		var v_err := v_ref - vx
		if v_err < -0.10:
			inp.throttle = 0.0
			inp.brake = clampf(-0.95 * v_err, 0.0, 1.0)
		else:
			inp.brake = 0.0
			if s_cur >= L_entry + 0.70 * L_arc and absf(cte) < 0.8:
				inp.throttle = 1.0
			else:
				inp.throttle = clampf(0.25 + 0.55 * v_err, 0.0, 1.0)

		for _s in range(2):
			_step_car(m, inp, dt_ctrl * 0.5, true)

		var beta_deg := rad_to_deg(atan2(m.chassis_state.linear_velocity.y, maxf(m.chassis_state.linear_velocity.x, 0.5)))
		v_min = minf(v_min, m.chassis_state.linear_velocity.x)
		max_cte = maxf(max_cte, absf(cte))
		max_beta = maxf(max_beta, absf(beta_deg))
		var p_kw := m.hv_state.power_kw
		p_max = maxf(p_max, p_kw)
		p_min = minf(p_min, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			hv_bad += 1
		if absf(beta_deg) > 25.0 or absf(av.z) > 4.5:
			spun = true

		if s_cur >= L_entry + 0.25 * L_arc and s_cur <= L_entry + 0.75 * L_arc:
			var tq := m.hv_state.inverter_torques
			sum_t_in_apex += 0.5 * (float(tq[0]) + float(tq[2]))
			sum_t_out_apex += 0.5 * (float(tq[1]) + float(tq[3]))
			n_apex += 1

		if s_cur >= s_total:
			t_finish = t
			break

	return {
		"completed": t_finish > 0.0 and not spun,
		"t_total_s": t_finish if t_finish > 0.0 else max_dur,
		"t_turn_180_s": (t_turn_end - t_turn_start) if (t_turn_end > 0.0 and t_turn_start > 0.0) else -1.0,
		"t_exit_40m_s": (t_finish - t_turn_end) if (t_finish > 0.0 and t_turn_end > 0.0) else -1.0,
		"v_min_apex_ms": v_min,
		"max_cte_m": max_cte,
		"max_beta_deg": max_beta,
		"spun": spun,
		"p_dc_max_kw": p_max,
		"p_dc_min_kw": p_min,
		"hv_bad": hv_bad,
		"apex_t_in_avg_nm": (sum_t_in_apex / float(n_apex)) if n_apex > 0 else 0.0,
		"apex_t_out_avg_nm": (sum_t_out_apex / float(n_apex)) if n_apex > 0 else 0.0,
	}


func _run_slalom_55() -> Dictionary:
	# Exact match to Test 5 in test_tv_power_curve_and_drivability.gd: 15.3 m/s (55 km/h), 0.8 Hz sine steer
	var car := _new_car(15.3, Full14Dof.TrackGripProfile.TTC_LAB_PEAK, EcuTv4Wd.TestMode.AUTOCROSS)
	var m: Full14Dof = car["m"]
	var inp: VehicleInputs = car["inp"]
	var dt_ctrl := 0.005
	var max_beta := 0.0
	var max_wz := 0.0
	var spun := false
	var hv_bad := 0
	var p_max := -1e9
	var p_min := 1e9

	var n_sl := int(round(4.0 / dt_ctrl))
	for k in range(n_sl):
		var t := float(k) * dt_ctrl
		var vx := m.chassis_state.linear_velocity.x
		inp.throttle = clampf(0.18 + 0.15 * (15.3 - vx), 0.0, 0.50)
		inp.brake = 0.0
		inp.steer_angle = 0.6 * m.config.max_steer_angle * sin(TAU * 0.8 * t)
		for _s in range(2):
			_step_car(m, inp, dt_ctrl * 0.5, true)
		var b_deg := absf(rad_to_deg(atan2(m.chassis_state.linear_velocity.y, maxf(absf(m.chassis_state.linear_velocity.x), 0.5))))
		var wz := absf(m.chassis_state.angular_velocity.z)
		max_beta = maxf(max_beta, b_deg)
		max_wz = maxf(max_wz, wz)
		if b_deg > 25.0 or wz > 4.0 or m.chassis_state.linear_velocity.x < 3.0:
			spun = true
		var p_kw := m.hv_state.power_kw
		p_max = maxf(p_max, p_kw)
		p_min = minf(p_min, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			hv_bad += 1

	var rec_time_s := -1.0
	var streak := 0
	var n_rel := int(round(1.0 / dt_ctrl))
	for k in range(n_rel):
		inp.throttle = 0.15
		inp.brake = 0.0
		inp.steer_angle = 0.0
		for _s in range(2):
			_step_car(m, inp, dt_ctrl * 0.5, true)
		var b_deg := absf(rad_to_deg(atan2(m.chassis_state.linear_velocity.y, maxf(absf(m.chassis_state.linear_velocity.x), 0.5))))
		var wz := absf(m.chassis_state.angular_velocity.z)
		max_beta = maxf(max_beta, b_deg)
		max_wz = maxf(max_wz, wz)
		if wz < 0.15 and b_deg < 1.5:
			streak += 1
			if rec_time_s < 0.0 and streak >= 8:
				rec_time_s = float(k + 1 - 7) * dt_ctrl
		else:
			streak = 0

	return {
		"max_beta_deg": max_beta,
		"max_wz_rads": max_wz,
		"spun": spun,
		"rec_time_s": rec_time_s,
		"p_dc_max_kw": p_max,
		"p_dc_min_kw": p_min,
		"hv_bad": hv_bad,
	}


func _run_5map_hv_audit() -> Dictionary:
	var maps := [
		["AUTOCROSS", EcuTv4Wd.TestMode.AUTOCROSS],
		["SKIDPAD", EcuTv4Wd.TestMode.SKIDPAD],
		["EFFICIENCY", EcuTv4Wd.TestMode.EFFICIENCY],
		["ACCELERATION", EcuTv4Wd.TestMode.ACCELERATION],
		["TIRE_WARMER", EcuTv4Wd.TestMode.TIRE_WARMER],
	]
	var dt_ctrl := 0.005
	var out := {}
	for mp in maps:
		var m_name: String = mp[0]
		var m_id: int = mp[1]
		var car := _new_car(14.0, Full14Dof.TrackGripProfile.TTC_LAB_PEAK, m_id)
		var m: Full14Dof = car["m"]
		var inp: VehicleInputs = car["inp"]
		var p_hi := -1e9
		var p_lo := 1e9
		var bad := 0
		for k in range(400): # 2.0s: 0.7s WOT straight, 0.7s WOT corner, 0.6s heavy brake + steer
			var t := float(k) * dt_ctrl
			if t < 0.7:
				inp.throttle = 1.0; inp.brake = 0.0; inp.steer_angle = 0.0
			elif t < 1.4:
				inp.throttle = 1.0; inp.brake = 0.0; inp.steer_angle = deg_to_rad(10.0)
			else:
				inp.throttle = 0.0; inp.brake = 0.85; inp.steer_angle = deg_to_rad(8.0)
			for _s in range(2):
				_step_car(m, inp, dt_ctrl * 0.5, false)
			var p_kw := m.hv_state.power_kw
			p_hi = maxf(p_hi, p_kw)
			p_lo = minf(p_lo, p_kw)
			if p_kw > 80.05 or p_kw < -25.05:
				bad += 1
		out[m_name] = {"p_max_kw": p_hi, "p_min_kw": p_lo, "hv_bad": bad}
	return out


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() >= 1:
		_out_path = args[0]
	if args.size() >= 2:
		_cfg_path = args[1]
	if args.size() >= 3:
		_ablation_mask = int(args[2])

	var radii := [5.0, 9.125, 15.0, 25.0]
	var stationary := {"FSG_REAL_TRACK": {}, "TTC_LAB_PEAK": {}}

	for gp in [
		["FSG_REAL_TRACK", Full14Dof.TrackGripProfile.FSG_REAL_TRACK],
		["TTC_LAB_PEAK", Full14Dof.TrackGripProfile.TTC_LAB_PEAK],
	]:
		var gp_name: String = gp[0]
		var gp_id: int = gp[1]
		for r in radii:
			var key := "R_%.3f" % r
			stationary[gp_name][key] = _run_stationary_corner(r, gp_id, EcuTv4Wd.TestMode.AUTOCROSS)
		stationary[gp_name]["R_9.125_SKIDPAD"] = _run_stationary_corner(9.125, gp_id, EcuTv4Wd.TestMode.SKIDPAD)

	var hairpin := {
		"FSG_REAL_TRACK": _run_hairpin_180(Full14Dof.TrackGripProfile.FSG_REAL_TRACK),
		"TTC_LAB_PEAK": _run_hairpin_180(Full14Dof.TrackGripProfile.TTC_LAB_PEAK),
	}
	var slalom := _run_slalom_55()
	var hv_maps := _run_5map_hv_audit()

	var sorted_us := _ecu_timings_us.duplicate()
	sorted_us.sort()
	var n_t := sorted_us.size()
	var sum_us := 0.0
	for u in sorted_us:
		sum_us += u
	var mean_ms := (sum_us / maxf(float(n_t), 1.0)) * 0.001
	var p99_ms: float = (sorted_us[int(0.99 * float(n_t - 1))] if n_t > 0 else 0.0) * 0.001
	var max_ms: float = (sorted_us[n_t - 1] if n_t > 0 else 0.0) * 0.001

	var report := {
		"stationary": stationary,
		"hairpin_180": hairpin,
		"slalom_55kmh": slalom,
		"hv_5maps": hv_maps,
		"solver_audit": {
			"step_time_mean_ms": mean_ms,
			"step_time_p99_ms": p99_ms,
			"step_time_max_ms": max_ms,
			"hv_viol_hi_80kw": _hv_viol_hi,
			"hv_viol_lo_25kw": _hv_viol_lo,
			"p_dc_max_kw": _p_dc_max_kw,
			"p_dc_min_kw": _p_dc_min_kw,
			"n_steps": n_t,
		},
	}

	var f := FileAccess.open(_out_path, FileAccess.WRITE)
	f.store_string(JSON.stringify(report, "\t"))
	f.close()
	quit(0)
