extends SceneTree

# Headless Godot 4 14-DOF 3D Double Wishbone Skidpad Parity Trace Exporter
# Drives TeR27 4WD in SKIDPAD_MAX_ATTACK_V2 through official FSG Figure-8
# (Right circle -> Crossover -> Left circle) and exports 50 Hz telemetry JSON
# for Plot 12 (12_python_vs_godot_skidpad_parity.png).
#
# Run:
#   godot --headless --path src/mirena_sim -s ../../vd_correlation_lab/benchmarks/export_godot_skidpad_parity.gd \
#     -- --out=../../outputs/validation_vault/09_ter27_skidpad_deep_dive/godot_skidpad_parity_trace.json

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const ChassisDynamics = preload("res://Scenes/Vehicle/Dynamics/Chassis/ChassisDynamics.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")

const R_CENTER: float = 9.125
const R_INNER: float = 7.625
const R_OUTER: float = 10.625
const HALF_TRACK: float = 0.615
const G_ACCEL: float = 9.81

var _out_path: String = "outputs/validation_vault/09_ter27_skidpad_deep_dive/godot_skidpad_parity_trace.json"
var _grip_name: String = "FSG_RUBBERED"
var _r_target: float = 8.40
var _v_target: float = 10.85


static func _wrap_pi(a: float) -> float:
	var x := fmod(a + PI, TAU)
	if x < 0.0:
		x += TAU
	return x - PI


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for a in args:
		if a.begins_with("--out="):
			_out_path = a.substr(6)
		elif a.begins_with("--grip="):
			_grip_name = a.substr(7).to_upper()
		elif a.begins_with("--r_target="):
			_r_target = float(a.substr(11))
		elif a.begins_with("--v_target="):
			_v_target = float(a.substr(11))


func _resolve_grip_profile(m: Full14Dof, name_upper: String) -> int:
	# Support FSG_RUBBERED if added to TrackGripProfile enum, otherwise fallback safely
	if name_upper == "TTC_LAB_PEAK" or name_upper == "TTC_PEAK":
		return 0
	if name_upper == "FSG_RUBBERED":
		if m.has_method("set_track_grip_profile_by_name"):
			m.call("set_track_grip_profile_by_name", "FSG_RUBBERED")
			return -1
		if Full14Dof.TrackGripProfile.has("FSG_RUBBERED"):
			return int(Full14Dof.TrackGripProfile["FSG_RUBBERED"])
		return 1
	return 1


func _step_heun(m: Full14Dof, inp: VehicleInputs, dt: float) -> Dictionary:
	var s0 := m.chassis_state.clone()
	var w0: Array[Vector4] = [
		m.wheels[0].get_integrator_state(),
		m.wheels[1].get_integrator_state(),
		m.wheels[2].get_integrator_state(),
		m.wheels[3].get_integrator_state()
	]
	var c1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, m.config, 0.0)
	var r1: Dictionary = m.step_forces(inp, s0.linear_velocity, s0.angular_velocity, c1, dt, true)
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
	return r2


func _corner_cone_clearance(x_cg: float, y_cg: float, yaw: float, cx: float, cy: float, a_cog: float, b_cog: float) -> float:
	var c := cos(yaw)
	var s := sin(yaw)
	var corners := [
		Vector2(a_cog, HALF_TRACK),
		Vector2(a_cog, -HALF_TRACK),
		Vector2(-b_cog, HALF_TRACK),
		Vector2(-b_cog, -HALF_TRACK)
	]
	var min_clr := 99.0
	for pt in corners:
		var wx: float = x_cg + pt.x * c - pt.y * s
		var wy: float = y_cg + pt.x * s + pt.y * c
		var d_act := Vector2(wx - cx, wy - cy).length()
		var clr_act := minf(d_act - R_INNER, R_OUTER - d_act)
		if absf(wy) < 2.2 and absf(wx) < 5.0:
			var dr := Vector2(wx, wy + R_CENTER).length()
			var dl := Vector2(wx, wy - R_CENTER).length()
			var clr_r := minf(dr - R_INNER, R_OUTER - dr)
			var clr_l := minf(dl - R_INNER, R_OUTER - dl)
			clr_act = maxf(clr_r, clr_l)
		min_clr = minf(min_clr, clr_act)
	return min_clr


func _init() -> void:
	_parse_args()
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	var grip_id := _resolve_grip_profile(m, _grip_name)
	if grip_id >= 0:
		m.set_track_grip_profile(grip_id)

	# Ensure 3D Double Wishbone kinematics is active
	var use_3d := false
	if m.suspension != null:
		if "use_3d_kinematics" in m.suspension:
			m.suspension.use_3d_kinematics = true
			use_3d = bool(m.suspension.use_3d_kinematics)

	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.active_mode = m.ecu_tv_4wd.TestMode.SKIDPAD

	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 3  # SKIDPAD / MAX_ATTACK
	inp.tv_mode = 0

	# Start on Right circle at top tangent (0, -R_CENTER + r_target), heading +X (psi = 0, turning CW / right)
	var v0 := _v_target * 0.92
	m.chassis_state.position = Vector3(0.0, -R_CENTER + _r_target, m.chassis_state.position.z)
	m.chassis_state.yaw = 0.0
	m.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	m.chassis_state.angular_velocity = Vector3(0.0, 0.0, -v0 / _r_target)
	for w in m.wheels:
		w.omega = v0 / maxf(m.config.wheel_radius, 0.1)

	var wb := m.config.wheelbase
	var a_cog := wb * (1.0 - m.config.front_weight_distribution)
	var b_cog := wb * m.config.front_weight_distribution
	var max_steer := m.config.max_steer_angle
	var delta_cur := -atan(wb / _r_target)

	var dt_ctrl := 0.005
	var max_steps := int(round(32.0 / dt_ctrl))
	var phase := 0  # 0 = Right circle (CW), 1 = Left circle (CCW)
	var theta_accum := 0.0
	var prev_ang := atan2(m.chassis_state.position.y - (-R_CENTER), m.chassis_state.position.x)
	var t_right_start := -1.0
	var t_right_end := -1.0
	var t_left_start := -1.0
	var t_left_end := -1.0
	var prev_e_r := 0.0
	var v_int := 0.0

	var hv_violations := 0
	var p_dc_max_kw := -1e9
	var p_dc_min_kw := 1e9
	var min_clear_m := 99.0

	var hist_t: Array[float] = []
	var hist_x: Array[float] = []
	var hist_y: Array[float] = []
	var hist_vx: Array[float] = []
	var hist_vy: Array[float] = []
	var hist_ay_g: Array[float] = []
	var hist_ax_g: Array[float] = []
	var hist_wz: Array[float] = []
	var hist_beta_deg: Array[float] = []
	var hist_delta_deg: Array[float] = []
	var hist_mz_tv: Array[float] = []
	var hist_p_dc_kw: Array[float] = []
	var hist_t_mot: Array = []
	var hist_fz: Array = []
	var hist_camber_deg: Array = []
	var hist_clear_m: Array[float] = []
	var hist_phase: Array[int] = []

	for step_i in range(max_steps):
		var t_s := float(step_i) * dt_ctrl
		var cx := 0.0
		var cy := -R_CENTER if phase == 0 else R_CENTER
		var turn_sgn := -1.0 if phase == 0 else 1.0

		var pos := m.chassis_state.position
		var lv := m.chassis_state.linear_velocity
		var av := m.chassis_state.angular_velocity
		var yaw := m.chassis_state.yaw

		var dx := pos.x - cx
		var dy := pos.y - cy
		var r_cur := maxf(sqrt(dx * dx + dy * dy), 1.0)
		var e_r := r_cur - _r_target
		var e_r_dot := (e_r - prev_e_r) / dt_ctrl if step_i > 0 else 0.0
		prev_e_r = e_r

		var ang_c := atan2(dy, dx)
		var psi_tan := ang_c + turn_sgn * 0.5 * PI
		var e_psi := _wrap_pi(psi_tan - yaw)

		var vx_s := maxf(lv.x, 2.5)
		var delta_ff := turn_sgn * atan(wb / _r_target)
		var delta_lim := minf(max_steer, 0.235)
		var delta_raw := clampf(
			delta_ff + 1.30 * e_psi + turn_sgn * atan2(1.85 * e_r + 0.35 * e_r_dot, vx_s),
			-delta_lim,
			delta_lim
		)
		var max_dd := 3.8 * dt_ctrl
		delta_cur = clampf(delta_raw, delta_cur - max_dd, delta_cur + max_dd)
		inp.steer_angle = delta_cur

		# Speed controller with radial-error grip protection so front tires never wash out
		var v_cmd := _v_target
		if e_r > 0.12:
			v_cmd = _v_target * clampf(1.0 - 0.22 * (e_r - 0.12), 0.78, 1.0)
		var v_err := v_cmd - lv.x
		v_int = clampf(v_int + v_err * dt_ctrl, -1.5, 1.5)
		var thr := clampf(0.28 + 0.42 * v_err + 0.18 * v_int, 0.0, 1.0)
		var brk := clampf(-0.32 * v_err, 0.0, 0.40) if v_err < -0.35 else 0.0
		inp.throttle = thr
		inp.brake = brk

		var step_res: Dictionary = {}
		for _sub in range(2):
			step_res = _step_heun(m, inp, dt_ctrl * 0.5)

		var p_kw := float(m.hv_state.power_kw)
		p_dc_max_kw = maxf(p_dc_max_kw, p_kw)
		p_dc_min_kw = minf(p_dc_min_kw, p_kw)
		if p_kw > 80.05 or p_kw < -25.05:
			hv_violations += 1

		var clr_now := _corner_cone_clearance(m.chassis_state.position.x, m.chassis_state.position.y, m.chassis_state.yaw, cx, cy, a_cog, b_cog)
		min_clear_m = minf(min_clear_m, clr_now)

		# Record at 50 Hz (every 4th 200 Hz tick)
		if step_i % 4 == 0:
			var vx_n := m.chassis_state.linear_velocity.x
			var vy_n := m.chassis_state.linear_velocity.y
			var wz_n := m.chassis_state.angular_velocity.z
			var ax_n := m.net_force_chassis_ros.x / maxf(m.config.mass, 100.0)
			var ay_n := m.net_force_chassis_ros.y / maxf(m.config.mass, 100.0)
			var mz_tv_n := float(m.ecu_tv_4wd.last_mz_req) if m.ecu_tv_4wd != null else 0.0
			var fz_step: Array = step_res.get("normal_forces", [0.0, 0.0, 0.0, 0.0])
			var t_arr: Array[float] = []
			var fz_arr: Array[float] = []
			var cam_arr: Array[float] = []
			for wi in range(4):
				var t_nm := 0.0
				if m.ecu_tv_4wd != null and wi < m.ecu_tv_4wd.last_t_cmd.size():
					t_nm = float(m.ecu_tv_4wd.last_t_cmd[wi])
				elif "wheel_drive_torque_nm" in m and wi < m.wheel_drive_torque_nm.size():
					t_nm = float(m.wheel_drive_torque_nm[wi])
				t_arr.append(snappedf(t_nm, 0.01))
				fz_arr.append(snappedf(float(fz_step[wi]) if wi < fz_step.size() else 0.0, 0.1))
				var cam_d := 0.0
				if m.suspension != null and wi < m.suspension.corners.size():
					cam_d = rad_to_deg(float(m.suspension.corners[wi].camber_rad))
				cam_arr.append(snappedf(cam_d, 0.001))

			hist_t.append(snappedf(t_s, 0.001))
			hist_x.append(snappedf(m.chassis_state.position.x, 0.001))
			hist_y.append(snappedf(m.chassis_state.position.y, 0.001))
			hist_vx.append(snappedf(vx_n, 0.001))
			hist_vy.append(snappedf(vy_n, 0.001))
			hist_ay_g.append(snappedf(ay_n / G_ACCEL, 0.001))
			hist_ax_g.append(snappedf(ax_n / G_ACCEL, 0.001))
			hist_wz.append(snappedf(wz_n, 0.001))
			hist_beta_deg.append(snappedf(rad_to_deg(atan2(vy_n, maxf(absf(vx_n), 1.0))), 0.01))
			hist_delta_deg.append(snappedf(rad_to_deg(delta_cur), 0.01))
			hist_mz_tv.append(snappedf(mz_tv_n, 0.1))
			hist_p_dc_kw.append(snappedf(p_kw, 0.01))
			hist_t_mot.append(t_arr)
			hist_fz.append(fz_arr)
			hist_camber_deg.append(cam_arr)
			hist_clear_m.append(snappedf(clr_now, 0.001))
			hist_phase.append(phase)

		var ang_now := atan2(m.chassis_state.position.y - cy, m.chassis_state.position.x - cx)
		var dang := _wrap_pi(ang_now - prev_ang)
		theta_accum += absf(dang)
		prev_ang = ang_now

		if phase == 0:
			if t_right_start < 0.0 and theta_accum >= 0.98 * TAU:
				t_right_start = t_s
			if t_right_end < 0.0 and theta_accum >= 1.98 * TAU:
				t_right_end = t_s
				phase = 1
				theta_accum = 0.0
				prev_ang = atan2(m.chassis_state.position.y - R_CENTER, m.chassis_state.position.x)
		else:
			if t_left_start < 0.0 and theta_accum >= 1.0 * TAU:
				t_left_start = t_s
			if t_left_end < 0.0 and theta_accum >= 2.0 * TAU:
				t_left_end = t_s
				break

	var t_right := (t_right_end - t_right_start) if (t_right_start >= 0.0 and t_right_end >= 0.0) else 9.99
	var t_left := (t_left_end - t_left_start) if (t_left_start >= 0.0 and t_left_end >= 0.0) else 9.99
	var t_mean := 0.5 * (t_right + t_left)

	var vx_sum := 0.0
	var ay_abs_max := 0.0
	var beta_abs_max := 0.0
	for k in range(hist_vx.size()):
		vx_sum += hist_vx[k]
		ay_abs_max = maxf(ay_abs_max, absf(hist_ay_g[k]))
		beta_abs_max = maxf(beta_abs_max, absf(hist_beta_deg[k]))
	var v_mean := vx_sum / maxf(float(hist_vx.size()), 1.0)

	var payload := {
		"engine": "Godot 4 Full14DofModel (3D Double Wishbone)",
		"grip_profile": _grip_name,
		"use_3d_kinematics": use_3d,
		"r_target_m": _r_target,
		"v_target_ms": _v_target,
		"t_right_s": snappedf(t_right, 0.001),
		"t_left_s": snappedf(t_left, 0.001),
		"t_mean_s": snappedf(t_mean, 0.001),
		"v_mean_ms": snappedf(v_mean, 0.001),
		"ay_peak_g": snappedf(ay_abs_max, 0.001),
		"beta_max_abs_deg": snappedf(beta_abs_max, 0.01),
		"p_dc_min_kw": snappedf(p_dc_min_kw, 0.01),
		"p_dc_max_kw": snappedf(p_dc_max_kw, 0.01),
		"hv_violations": hv_violations,
		"min_cone_clear_m": snappedf(min_clear_m, 0.001),
		"n_samples": hist_t.size(),
		"hist": {
			"t": hist_t,
			"X": hist_x,
			"Y": hist_y,
			"vx": hist_vx,
			"vy": hist_vy,
			"ay_g": hist_ay_g,
			"ax_g": hist_ax_g,
			"wz": hist_wz,
			"beta_deg": hist_beta_deg,
			"delta_deg": hist_delta_deg,
			"mz_tv": hist_mz_tv,
			"p_dc_kw": hist_p_dc_kw,
			"t_mot": hist_t_mot,
			"fz": hist_fz,
			"camber_deg": hist_camber_deg,
			"min_clear_m": hist_clear_m,
			"phase": hist_phase
		}
	}

	var dir_path := _out_path.get_base_dir()
	if dir_path != "":
		DirAccess.make_dir_recursive_absolute(dir_path)
	var f := FileAccess.open(_out_path, FileAccess.WRITE)
	if f == null:
		push_error("Failed to open output file: " + _out_path)
		quit(1)
		return
	f.store_string(JSON.stringify(payload, "  "))
	f.close()
	print("[GODOT_PARITY] t_R=%.3fs t_L=%.3fs t_mean=%.3fs v_mean=%.2fm/s ay_peak=%.2fg clear=%.3fm hv=%d 3d=%s -> %s" % [
		t_right, t_left, t_mean, v_mean, ay_abs_max, min_clear_m, hv_violations, str(use_3d), _out_path
	])
	quit(0)
