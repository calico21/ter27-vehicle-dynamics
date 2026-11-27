extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const DoubleWishboneScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/DoubleWishboneKinematics.gd")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func run() -> Dictionary:
	var st := {"name": "test_double_wishbone_kinematics", "passed": 0, "failed": 0, "errors": []}

	var cfg := VehicleConfigScript.new()
	cfg.load_vehicle_configs("ter27_4wd")
	_check(cfg.use_3d_kinematics and not cfg.double_wishbone_kinematics.is_empty(), "ter27_4wd enables 3D double-wishbone kinematics", st)

	var kin := DoubleWishboneScript.new()
	_check(kin.configure(cfg.double_wishbone_kinematics, cfg.wheel_radius, cfg.wheelbase, cfg.h_cog), "DoubleWishboneKinematics.configure succeeds", st)
	_check(kin.build_luts(), "DoubleWishboneKinematics.build_luts succeeds", st)

	# ---------------------------------------------------------
	# 1. Exact static Excel verification (delta_z = 0, steer = 0)
	# ---------------------------------------------------------
	var sf: Dictionary = kin.compute_static_metrics(true)
	var sr: Dictionary = kin.compute_static_metrics(false)

	_check(absf(float(sf["kpi_deg"]) - 10.00) < 0.01, "Front KPI = %.4f deg (exp 10.00)" % float(sf["kpi_deg"]), st)
	_check(absf(float(sf["caster_deg"]) - 5.00) < 0.01, "Front Caster = %.4f deg (exp 5.00)" % float(sf["caster_deg"]), st)
	_check(absf(float(sf["scrub_radius_m"]) * 1000.0 - 10.00) < 0.02, "Front Scrub = %.3f mm (exp 10.00)" % (float(sf["scrub_radius_m"]) * 1000.0), st)
	_check(absf(float(sf["mechanical_trail_m"]) * 1000.0 - 13.00) < 0.02, "Front Trail = %.3f mm (exp 13.00)" % (float(sf["mechanical_trail_m"]) * 1000.0), st)
	_check(absf(float(sf["fvsa_ic_y"]) - (-2.5975)) < 0.002 and absf(float(sf["fvsa_ic_z"]) - 0.1023) < 0.001, "Front FVSA IC = (%.4f, %.4f)" % [float(sf["fvsa_ic_y"]), float(sf["fvsa_ic_z"])], st)
	_check(absf(float(sf["roll_center_z"]) * 1000.0 - 19.59) < 0.05, "Front RC height = %.3f mm (exp 19.59)" % (float(sf["roll_center_z"]) * 1000.0), st)
	_check(absf(float(sf["svsa_ic_x"]) - (-1.6086)) < 0.002 and absf(float(sf["svsa_ic_z"]) - 0.2235) < 0.001 and absf(float(sf["svsa_slope"]) - 0.1390) < 0.001 and absf(float(sf["anti_pitch_100pct"]) * 100.0 - 73.79) < 0.1, "Front SVSA IC & 73.79%% anti-dive match Excel", st)
	_check(absf(float(sf["rocker_arm_pushrod_m"]) * 1000.0 - 42.01) < 0.03 and absf(float(sf["rocker_arm_damper_m"]) * 1000.0 - 62.99) < 0.03 and absf(float(sf["spring_to_pushrod_force_ratio"]) - 0.6669) < 0.0005, "Front rocker arms (42.01/62.99 mm) & force ratio (0.6669) match Excel", st)

	_check(absf(float(sr["kpi_deg"]) - 18.74) < 0.01, "Rear KPI = %.4f deg (exp 18.74)" % float(sr["kpi_deg"]), st)
	_check(absf(float(sr["caster_deg"]) - 0.00) < 0.01, "Rear Caster = %.4f deg (exp 0.00)" % float(sr["caster_deg"]), st)
	_check(absf(float(sr["scrub_radius_m"]) * 1000.0) < 0.02 and absf(float(sr["mechanical_trail_m"]) * 1000.0) < 0.02, "Rear Scrub & Trail = 0.00 mm", st)
	_check(absf(float(sr["roll_center_z"]) * 1000.0 - 40.03) < 0.05, "Rear RC height = %.3f mm (exp 40.03)" % (float(sr["roll_center_z"]) * 1000.0), st)
	_check(absf(float(sr["svsa_ic_x"]) - (-0.3949)) < 0.002 and absf(float(sr["svsa_ic_z"]) - 0.1322) < 0.001 and absf(float(sr["svsa_slope"]) - 0.1154) < 0.001 and absf(float(sr["anti_pitch_100pct"]) * 100.0 - 61.29) < 0.1, "Rear SVSA IC & 61.29%% anti-squat match Excel", st)
	_check(absf(float(sr["rocker_arm_pushrod_m"]) * 1000.0 - 34.60) < 0.03 and absf(float(sr["rocker_arm_damper_m"]) * 1000.0 - 60.00) < 0.03 and absf(float(sr["spring_to_pushrod_force_ratio"]) - 0.5766) < 0.0005, "Rear rocker arms (34.60/60.00 mm) & force ratio (0.5766) match Excel", st)

	# ---------------------------------------------------------
	# 2. Rigid link invariants (< 1e-5 m) across bump/droop & steer
	# ---------------------------------------------------------
	var max_link_err: float = 0.0
	var max_pr_uca_dist_err: float = 0.0
	var pr_to_uca_f0: float = (kin.front_left.pushrod_wishbone - kin.front_left.uca_front).length()
	var pr_to_uca_r0: float = (kin.front_left.pushrod_wishbone - kin.front_left.uca_rear).length()
	var pr_to_ubj0: float = (kin.front_left.pushrod_wishbone - kin.front_left.ubj).length()

	for dz in [-0.045, -0.030, -0.015, 0.0, 0.015, 0.030, 0.045]:
		for rack in [-0.025, -0.012, 0.0, 0.012, 0.025]:
			var pf: Dictionary = kin.solve_pose_3d(dz, rack, true, false)
			max_link_err = maxf(max_link_err, maxf(float(pf["err_kp"]), float(pf["err_tie"])))
			max_link_err = maxf(max_link_err, maxf(float(pf["err_pr"]), float(pf["err_arb"])))
			max_link_err = maxf(max_link_err, maxf(float(pf["err_lca"]), float(pf["err_uca"])))
			var pr_wb: Vector3 = pf["pushrod_wishbone"]
			var ubj_p: Vector3 = pf["ubj"]
			max_pr_uca_dist_err = maxf(max_pr_uca_dist_err, absf((pr_wb - kin.front_left.uca_front).length() - pr_to_uca_f0))
			max_pr_uca_dist_err = maxf(max_pr_uca_dist_err, absf((pr_wb - kin.front_left.uca_rear).length() - pr_to_uca_r0))
			max_pr_uca_dist_err = maxf(max_pr_uca_dist_err, absf((pr_wb - ubj_p).length() - pr_to_ubj0))
		var pr: Dictionary = kin.solve_pose_3d(dz, 0.0, false, false)
		max_link_err = maxf(max_link_err, maxf(float(pr["err_kp"]), float(pr["err_tie"])))
		max_link_err = maxf(max_link_err, maxf(float(pr["err_pr"]), float(pr["err_arb"])))
		max_link_err = maxf(max_link_err, maxf(float(pr["err_lca"]), float(pr["err_uca"])))

	_check(max_link_err < 1.0e-5, "3D link rigidity invariant across full travel & steer: max_err=%s m (< 1e-5 m)" % String.num_scientific(max_link_err), st)
	_check(max_pr_uca_dist_err < 1.0e-5, "Pushrod wishbone clevis rigidly locked to UCA: max_err=%s m (< 1e-5 m)" % String.num_scientific(max_pr_uca_dist_err), st)

	# ---------------------------------------------------------
	# 3. LUT vs exact 3D solver parity on non-grid points
	# ---------------------------------------------------------
	var buf := PackedFloat64Array()
	buf.resize(DoubleWishboneScript.STRIDE)
	var max_cam_deg_err: float = 0.0
	var max_toe_deg_err: float = 0.0
	var max_mr_pct_err: float = 0.0
	var max_cp_mm_err: float = 0.0
	var max_rc_mm_err: float = 0.0

	for dz_q in [-0.0313, -0.0142, 0.0063, 0.0237, 0.0381]:
		for st_q in [-0.337, -0.112, 0.064, 0.289]:
			kin.sample_front_into(dz_q, st_q, false, buf, 0)
			var rack_q: float = kin.get_rack_disp_for_steer(st_q)
			var ref_f: Dictionary = kin.solve_pose_with_derivatives(dz_q, rack_q, true, false)
			max_cam_deg_err = maxf(max_cam_deg_err, absf(rad_to_deg(buf[DoubleWishboneScript.CH_CAMBER] - float(ref_f["camber_rad"]))))
			max_toe_deg_err = maxf(max_toe_deg_err, absf(rad_to_deg(buf[DoubleWishboneScript.CH_WHEEL_STEER] - float(ref_f["wheel_steer_rad"]))))
			var mr_ref: float = float(ref_f["motion_ratio"])
			max_mr_pct_err = maxf(max_mr_pct_err, absf(buf[DoubleWishboneScript.CH_MR] - mr_ref) / maxf(absf(mr_ref), 1e-3) * 100.0)
			var cp_ref: Vector2 = ref_f["cp_offset_ros"]
			max_cp_mm_err = maxf(max_cp_mm_err, Vector2(buf[DoubleWishboneScript.CH_CP_DX], buf[DoubleWishboneScript.CH_CP_DY]).distance_to(cp_ref) * 1000.0)
			max_rc_mm_err = maxf(max_rc_mm_err, absf(buf[DoubleWishboneScript.CH_RC_Z] - float(ref_f["roll_center_z"])) * 1000.0)

		kin.sample_rear_into(dz_q, false, buf, 0)
		var ref_r: Dictionary = kin.solve_pose_with_derivatives(dz_q, 0.0, false, false)
		max_cam_deg_err = maxf(max_cam_deg_err, absf(rad_to_deg(buf[DoubleWishboneScript.CH_CAMBER] - float(ref_r["camber_rad"]))))
		max_toe_deg_err = maxf(max_toe_deg_err, absf(rad_to_deg(buf[DoubleWishboneScript.CH_WHEEL_STEER] - float(ref_r["wheel_steer_rad"]))))
		var mr_ref_r: float = float(ref_r["motion_ratio"])
		max_mr_pct_err = maxf(max_mr_pct_err, absf(buf[DoubleWishboneScript.CH_MR] - mr_ref_r) / maxf(absf(mr_ref_r), 1e-3) * 100.0)
		var cp_ref_r: Vector2 = ref_r["cp_offset_ros"]
		max_cp_mm_err = maxf(max_cp_mm_err, Vector2(buf[DoubleWishboneScript.CH_CP_DX], buf[DoubleWishboneScript.CH_CP_DY]).distance_to(cp_ref_r) * 1000.0)
		max_rc_mm_err = maxf(max_rc_mm_err, absf(buf[DoubleWishboneScript.CH_RC_Z] - float(ref_r["roll_center_z"])) * 1000.0)

	_check(max_cam_deg_err < 0.02, "LUT camber parity < 0.02 deg (max=%.5f deg)" % max_cam_deg_err, st)
	_check(max_toe_deg_err < 0.02, "LUT steer/toe parity < 0.02 deg (max=%.5f deg)" % max_toe_deg_err, st)
	_check(max_mr_pct_err < 0.5, "LUT MR parity < 0.5%% (max=%.4f%%)" % max_mr_pct_err, st)
	_check(max_cp_mm_err < 1.0, "LUT CP migration parity < 1 mm (max=%.4f mm)" % max_cp_mm_err, st)
	_check(max_rc_mm_err < 1.0, "LUT RC migration parity < 1 mm (max=%.4f mm)" % max_rc_mm_err, st)

	# ---------------------------------------------------------
	# 4. L/R symmetry in bump, steer antisymmetry, geometric Kw, ARB, Ackermann
	# ---------------------------------------------------------
	var p_fl_bump: Dictionary = kin.solve_pose_3d(0.022, 0.0, true, false)
	var p_fr_bump: Dictionary = kin.solve_pose_3d(0.022, 0.0, true, true)
	_check(
		absf(float(p_fl_bump["camber_rad"]) + float(p_fr_bump["camber_rad"])) < 1.0e-6 \
		and absf(float(p_fl_bump["toe_rad"]) + float(p_fr_bump["toe_rad"])) < 1.0e-6 \
		and absf(float(p_fl_bump["damper_disp_m"]) - float(p_fr_bump["damper_disp_m"])) < 1.0e-6 \
		and absf((p_fl_bump["cp_offset_ros"] as Vector2).y + (p_fr_bump["cp_offset_ros"] as Vector2).y) < 1.0e-6,
		"L/R pure bump symmetry verified on independent mirrored 3D geometry",
		st
	)

	var rack_25: float = kin.get_rack_disp_for_steer(0.25)
	var p_fl_st: Dictionary = kin.solve_pose_3d(0.0, rack_25, true, false)
	var p_fr_st: Dictionary = kin.solve_pose_3d(0.0, rack_25, true, true)
	var st_fl: float = float(p_fl_st["wheel_steer_rad"])
	var st_fr: float = float(p_fr_st["wheel_steer_rad"])
	var mean_st: float = 0.5 * (st_fl + st_fr)
	_check(absf(mean_st - 0.25) < 1.0e-3 and st_fr > st_fl, "Rack calibration matches mean steer (%.4f vs 0.25) with real tie-rod Anti-Ackermann (FR=%.4f > FL=%.4f)" % [mean_st, st_fr, st_fl], st)

	# Steer antisymmetry via LUT
	var b_l := PackedFloat64Array(); b_l.resize(DoubleWishboneScript.STRIDE)
	var b_r := PackedFloat64Array(); b_r.resize(DoubleWishboneScript.STRIDE)
	kin.sample_front_into(0.015, 0.22, false, b_l, 0)
	kin.sample_front_into(0.015, -0.22, true, b_r, 0)
	_check(absf(b_l[DoubleWishboneScript.CH_WHEEL_STEER] + b_r[DoubleWishboneScript.CH_WHEEL_STEER]) < 1.0e-6 and absf(b_l[DoubleWishboneScript.CH_CAMBER] + b_r[DoubleWishboneScript.CH_CAMBER]) < 1.0e-6, "Steer L/R antisymmetry verified", st)

	# Geometric tangent stiffness Kw = Ks * MR^2 + Fs * dMR/dz & pure heave ARB = 0
	var m_stiff := Full14DofScript.new()
	m_stiff.load_vehicle("ter27_4wd", "default")
	var h_k: float = 2.0e-4
	var c_p := m_stiff.suspension.evaluate_from_corners([0.02 + h_k, 0.02 + h_k, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m_stiff.config, 0.0)
	var f_p: float = c_p[0].spring_force
	var c_m := m_stiff.suspension.evaluate_from_corners([0.02 - h_k, 0.02 - h_k, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m_stiff.config, 0.0)
	var f_m: float = c_m[0].spring_force
	var c_0 := m_stiff.suspension.evaluate_from_corners([0.02, 0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m_stiff.config, 0.0)
	var kw_num: float = (f_p - f_m) / (2.0 * h_k)
	var kw_ana: float = c_0[0].wheel_spring_stiffness
	_check(absf(kw_num - kw_ana) / maxf(kw_ana, 1.0) < 0.01 and absf(c_0[0].arb_force) < 1e-6, "Geometric stiffness Kw = Ks*MR^2 + Fs*dMR/dz (num=%.1f vs ana=%.1f) & heave ARB=0" % [kw_num, kw_ana], st)

	# ---------------------------------------------------------
	# 5. Zero double counting of load transfer in 14-DOF steady state
	# ---------------------------------------------------------
	var cfg_lt := VehicleConfigScript.new()
	cfg_lt.load_vehicle_configs("ter27_4wd")
	cfg_lt.cd_a = 0.0
	cfg_lt.cl_a = 0.0
	var m_long := Full14DofScript.new(cfg_lt)
	m_long.chassis_state.linear_velocity = Vector3(12.0, 0.0, 0.0)
	for w in m_long.wheels:
		w.omega = 12.0 / cfg_lt.wheel_radius
	var inp_long := VehicleInputsScript.new()
	inp_long.override_wheel_torques = true
	inp_long.wheel_motor_torques.assign([100.0, 100.0, 150.0, 150.0])
	var res_long: Dictionary = {}
	for _k in range(1200): # 3.0 s @ 400 Hz -> pitch/heave settled
		res_long = m_long.step_standalone(inp_long, 0.0025, 0.0)
	var fz_l: Array = res_long["normal_forces"]
	var f_res_l: Dictionary = res_long["forces"]
	var stat_loads: Dictionary = cfg_lt.get_static_axle_loads()
	var rear_frac: float = float(stat_loads["rear_axle"]) / (float(stat_loads["front_axle"]) + float(stat_loads["rear_axle"]))
	var sum_fz_l: float = float(fz_l[0]) + float(fz_l[1]) + float(fz_l[2]) + float(fz_l[3])
	var dfz_rear_meas: float = (float(fz_l[2]) + float(fz_l[3])) - rear_frac * sum_fz_l
	var ax_ss: float = Vector3(f_res_l["net_force_ros"]).x / cfg_lt.mass
	var dfz_long_exp: float = (cfg_lt.mass * ax_ss * m_long.chassis_state.position.z) / cfg_lt.wheelbase
	_check(absf(dfz_rear_meas - dfz_long_exp) / maxf(absf(dfz_long_exp), 1.0) < 0.015, "Zero double counting longitudinal LT: meas=%.2f N vs m*ax*h/L=%.2f N" % [dfz_rear_meas, dfz_long_exp], st)

	var m_lat := Full14DofScript.new(cfg_lt)
	m_lat.chassis_state.linear_velocity = Vector3(12.0, 0.0, 0.0)
	for w in m_lat.wheels:
		w.omega = 12.0 / cfg_lt.wheel_radius
	var inp_lat := VehicleInputsScript.new()
	inp_lat.drive_mode = cfg_lt.drive_mode
	inp_lat.throttle = 0.25
	inp_lat.steer_angle = 0.14
	var res_lat: Dictionary = {}
	for _k in range(1600): # 4.0 s @ 400 Hz -> steady cornering
		res_lat = m_lat.step_standalone(inp_lat, 0.0025, 0.0)
	var fz_lat: Array = res_lat["normal_forces"]
	var f_res_lat: Dictionary = res_lat["forces"]
	var mx_sum: float = 0.0
	for mw in f_res_lat["wheel_moments_ros"]:
		mx_sum += Vector3(mw).x
	var dfz_lat_meas: float = 0.5 * ((float(fz_lat[1]) + float(fz_lat[3])) - (float(fz_lat[0]) + float(fz_lat[2]))) - mx_sum / cfg_lt.track_width_f
	var ay_ss: float = Vector3(f_res_lat["net_force_ros"]).y / cfg_lt.mass
	var dfz_lat_exp: float = (cfg_lt.mass * ay_ss * m_lat.chassis_state.position.z) / cfg_lt.track_width_f
	var tau_x_ss: float = Vector3(f_res_lat["net_torque_ros"]).x
	_check(absf(dfz_lat_meas - dfz_lat_exp) / maxf(absf(dfz_lat_exp), 1.0) < 0.02 and absf(tau_x_ss) < 2.0, "Zero double counting lateral LT: meas_corr=%.2f N vs m*ay*h/t=%.2f N (tau_x=%.3f Nm)" % [dfz_lat_meas, dfz_lat_exp, tau_x_ss], st)

	# ---------------------------------------------------------
	# 6. TeR26 RWD 3D Double Wishbone & Parametric A/B toggle
	# ---------------------------------------------------------
	var m_rwd := Full14DofScript.new()
	m_rwd.load_vehicle("ter26_rwd", "default")
	_check(m_rwd.config.use_3d_kinematics and m_rwd.suspension.is_3d_active(m_rwd.config), "ter26_rwd enables 3D double-wishbone kinematics by default", st)
	m_rwd.config.use_3d_kinematics = false
	m_rwd.suspension.sync_from_config(m_rwd.config)
	_check(not m_rwd.suspension.is_3d_active(m_rwd.config), "ter26_rwd A/B toggle use_3d_kinematics = false switches cleanly to parametric", st)

	m_stiff.config.use_3d_kinematics = false
	m_stiff.suspension.sync_from_config(m_stiff.config)
	_check(not m_stiff.suspension.is_3d_active(m_stiff.config), "A/B toggle use_3d_kinematics = false cleanly disables 3D mode", st)

	return st
