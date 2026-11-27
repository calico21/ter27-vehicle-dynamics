extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")

const G: float = 9.81
const BETA_GRID_DEG: Array = [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
const DELTA_GRID_DEG: Array = [
	-18.0, -14.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0,
	0.0,
	1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0
]
const SPEEDS_KMH: Array = [40.0, 65.0, 90.0]


func _init() -> void:
	var out_path := ProjectSettings.globalize_path("res://").path_join("../../vd_correlation_lab/results/ymd/ymd_raw.json").simplify_path()
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--out="):
			out_path = arg.substr(6)

	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	m.config.update_derived_geometry()
	m.suspension.sync_from_config(m.config)
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.configure_from_vehicle_config(m.config)
		m.ecu_tv_4wd.reset()

	var payload: Dictionary = {
		"vehicle_id": m.config.vehicle_id,
		"mass_kg": m.config.mass,
		"wheelbase_m": m.config.wheelbase,
		"a_cog_m": m.config.a_cog,
		"b_cog_m": m.config.b_cog,
		"cl_a": m.config.cl_a,
		"cd_a": m.config.cd_a,
		"max_steer_deg": rad_to_deg(m.config.max_steer_angle),
		"use_3d_kinematics": m.config.use_3d_kinematics,
		"beta_grid_deg": BETA_GRID_DEG,
		"delta_grid_deg": DELTA_GRID_DEG,
		"speeds_kmh": SPEEDS_KMH,
		"sweeps": [],
		"low_speed_ackermann": _run_low_speed_check(m),
	}

	var controllers := [
		{"name": "OPEN_4WD", "ecu_mode": 0, "tv4_mode": 0},
		{"name": "TV4_AUTOCROSS", "ecu_mode": 2, "tv4_mode": 0},
	]

	for ctrl in controllers:
		for v_kmh in SPEEDS_KMH:
			var pts: Array = _sweep_speed(m, float(v_kmh), int(ctrl["ecu_mode"]), int(ctrl["tv4_mode"]))
			(payload["sweeps"] as Array).append({
				"controller": ctrl["name"],
				"v_kmh": float(v_kmh),
				"v_mps": float(v_kmh) / 3.6,
				"points": pts,
			})

	var f := FileAccess.open(out_path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(payload))
		f.close()
		print("YMD probe saved -> %s (%d sweeps)" % [out_path, (payload["sweeps"] as Array).size()])
	else:
		push_error("Failed to write YMD output: " + out_path)
	quit()


func _init_car_at_speed(m: Full14Dof, v_mps: float, beta_rad: float) -> void:
	m.reset()
	var vx: float = v_mps * cos(beta_rad)
	var vy: float = v_mps * sin(beta_rad)
	m.chassis_state.linear_velocity = Vector3(vx, vy, 0.0)
	var rw: float = maxf(m.config.wheel_radius, 0.05)
	for w in m.wheels:
		w.omega = vx / rw
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.ekf_vx = vx
		m.ecu_tv_4wd.ekf_vy = vy
		m.ecu_tv_4wd._ekf_init = true


func _eval_tethered_point(
	m: Full14Dof,
	v_mps: float,
	beta_deg: float,
	delta_deg: float,
	ecu_mode: int,
	tv4_mode: int,
	r_init: float = 0.0
) -> Dictionary:
	var beta_rad: float = deg_to_rad(beta_deg)
	var delta_rad: float = deg_to_rad(delta_deg)
	var vx: float = v_mps * cos(beta_rad)
	var vy: float = v_mps * sin(beta_rad)

	_init_car_at_speed(m, v_mps, beta_rad)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	inp.tv_4wd_mode = tv4_mode
	inp.tv_mode = 0
	inp.steer_angle = delta_rad
	inp.brake = 0.0

	var dt: float = 0.0025
	var r_cur: float = r_init
	var mass_kg: float = maxf(m.config.mass, 100.0)
	var inv_mass: float = 1.0 / mass_kg
	var ay_cur: float = 0.0
	var mz_cur: float = 0.0
	var last_res: Dictionary = {}

	# Quasi-steady-state tethered iteration: settle suspension 3D, aero heave, tire slip & yaw rate r = ay / V
	for k in range(140):
		m.chassis_state.linear_velocity.x = vx
		m.chassis_state.linear_velocity.y = vy
		m.chassis_state.angular_velocity.z = r_cur
		if m.ecu_tv_4wd != null:
			m.ecu_tv_4wd.ekf_vx = vx
			m.ecu_tv_4wd.ekf_vy = vy

		# Maintain constant longitudinal speed against drag/induced drag without braking
		var fx_req: float = -mass_kg * r_cur * vy
		var fx_prev: float = float(last_res.get("net_force_ros", Vector3.ZERO).x) if not last_res.is_empty() else 0.0
		inp.throttle = clampf(0.14 + 0.0012 * (fx_req - fx_prev), 0.0, 0.70)

		last_res = m.step_standalone(inp, dt, 0.0)
		var net_f: Vector3 = last_res["net_force_ros"]
		var net_t: Vector3 = last_res["net_torque_ros"]
		# Centripetal equilibrium: ay_total = Fx_net/m? No, in body frame ay = Fy/m + vx*r_cur?
		# Actually Fy_tire/m = ay_inertial = v_dot_y + vx * r -> at steady beta, v_dot_y = 0 => ay_inertial = net_f.y / m + vx * r_cur?
		# Wait: step_standalone returns net_force_ros (sum of tire+aero forces), so ay_inertial = net_f.y / mass_kg!
		var ay_meas: float = net_f.y * inv_mass
		var r_tgt: float = clampf(ay_meas / maxf(v_mps, 2.0), -2.5, 2.5)
		r_cur = lerpf(r_cur, r_tgt, 0.22)
		if k >= 70 and absf(ay_meas - ay_cur) < 0.015 and absf(net_t.z - mz_cur) < 1.5:
			ay_cur = ay_meas
			mz_cur = net_t.z
			break
		ay_cur = ay_meas
		mz_cur = net_t.z

	var fz_arr: Array = last_res.get("normal_forces", [0.0, 0.0, 0.0, 0.0])
	var mu_est: float = 1.50
	var fy_f: float = absf(float(m.wheels[0].F_y)) + absf(float(m.wheels[1].F_y))
	var fy_r: float = absf(float(m.wheels[2].F_y)) + absf(float(m.wheels[3].F_y))
	var fz_f: float = maxf(float(fz_arr[0]) + float(fz_arr[1]), 10.0)
	var fz_r: float = maxf(float(fz_arr[2]) + float(fz_arr[3]), 10.0)
	var fy_arr := [float(m.wheels[0].F_y), float(m.wheels[1].F_y), float(m.wheels[2].F_y), float(m.wheels[3].F_y)]
	var alpha_deg: Array = []
	for wi in range(4):
		alpha_deg.append(rad_to_deg(float(m.wheels[wi].alpha)))

	var mz_tv: float = 0.0
	if ecu_mode == 2 and m.ecu_tv_4wd != null:
		mz_tv = float(m.ecu_tv_4wd.last_mz_req)

	var cn: float = mz_cur / (mass_kg * G * maxf(m.config.wheelbase, 1.0))
	var roll_deg: float = rad_to_deg(float(m.chassis_state.roll))
	return {
		"beta_deg": beta_deg,
		"delta_deg": delta_deg,
		"ay_mps2": ay_cur,
		"ay_g": ay_cur / G,
		"mz_nm": mz_cur,
		"cn": cn,
		"r_rad_s": r_cur,
		"roll_deg": roll_deg,
		"fz": [float(fz_arr[0]), float(fz_arr[1]), float(fz_arr[2]), float(fz_arr[3])],
		"fy": fy_arr,
		"alpha_deg": alpha_deg,
		"sat_front": fy_f / (mu_est * fz_f),
		"sat_rear": fy_r / (mu_est * fz_r),
		"mz_tv_nm": mz_tv,
	}


func _sweep_speed(m: Full14Dof, v_kmh: float, ecu_mode: int, tv4_mode: int) -> Array:
	var v_mps: float = v_kmh / 3.6
	var pts: Array = []
	for b_deg in BETA_GRID_DEG:
		var r_seed: float = 0.0
		for d_deg in DELTA_GRID_DEG:
			var pt := _eval_tethered_point(m, v_mps, float(b_deg), float(d_deg), ecu_mode, tv4_mode, r_seed)
			r_seed = float(pt["r_rad_s"])
			pts.append(pt)
	return pts


func _run_low_speed_check(m: Full14Dof) -> Array:
	var v_mps: float = 25.0 / 3.6
	var out: Array = []
	for radius_m in [6.0, 8.0]:
		var ay_tgt: float = (v_mps * v_mps) / float(radius_m)
		var best_delta := 0.0
		var best_err := 1e9
		var best_pt: Dictionary = {}
		for d_i in range(2, 44):
			var d_deg: float = float(d_i) * 0.5
			var pt := _eval_tethered_point(m, v_mps, 0.0, d_deg, 2, 0, ay_tgt / v_mps)
			var err := absf(float(pt["ay_mps2"]) - ay_tgt)
			if err < best_err:
				best_err = err
				best_delta = d_deg
				best_pt = pt
		var ack_kin_deg: float = rad_to_deg(atan(m.config.wheelbase / float(radius_m)))
		out.append({
			"radius_m": float(radius_m),
			"v_kmh": 25.0,
			"ay_tgt_g": ay_tgt / G,
			"delta_trim_deg": best_delta,
			"ackermann_kin_deg": ack_kin_deg,
			"understeer_delta_deg": best_delta - ack_kin_deg,
			"sat_front": float(best_pt.get("sat_front", 0.0)),
			"sat_rear": float(best_pt.get("sat_rear", 0.0)),
		})
	return out
