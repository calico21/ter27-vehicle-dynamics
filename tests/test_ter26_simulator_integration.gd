extends RefCounted

const SimScene = preload("res://Scenes/Sim/Sim.tscn")
const MirenaCarScene = preload("res://Scenes/Vehicle/mirena_car.tscn")
const MirenaCarDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const SuspVisScript = preload("res://Scenes/Vehicle/Visuals/SuspensionVisualizer3D.gd")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const DoubleWishboneScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/DoubleWishboneKinematics.gd")
const TracksideTuningControllerScript = preload("res://Scenes/Sim/UI/TracksideTuningController.gd")

const SPEC_PATH := "res://Assets/Vehicles/TeR26/ter26_validated_spec.json"
const RIGID_ROD_KEYS := ["uca_f", "uca_r", "lca_f", "lca_r", "tie", "pushrod"]


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func _is_ascii(s: String) -> bool:
	for i in range(s.length()):
		if s.unicode_at(i) > 127:
			return false
	return true


static func _check_ui_ascii_and_focus(node: Node, preset_edit: Control, ascii_info: Dictionary, focus_fail: Array) -> void:
	if node is Control:
		var ctrl := node as Control
		if ctrl != preset_edit and (ctrl is BaseButton or ctrl is Range or ctrl is TabContainer or ctrl is TabBar or ctrl is ScrollContainer):
			if ctrl.focus_mode != Control.FOCUS_NONE:
				focus_fail.append(String(ctrl.name))
	if node.has_method("get_text"):
		var t := str(node.call("get_text"))
		if not _is_ascii(t):
			ascii_info["ok"] = false
			ascii_info["offender"] = "%s: %s" % [node.name, t]
	if node is OptionButton:
		var ob := node as OptionButton
		for ii in range(ob.item_count):
			var it := ob.get_item_text(ii)
			if not _is_ascii(it):
				ascii_info["ok"] = false
				ascii_info["offender"] = "%s[%d]: %s" % [ob.name, ii, it]
	for ch in node.get_children():
		_check_ui_ascii_and_focus(ch, preset_edit, ascii_info, focus_fail)


static func _vec3_from_arr(a: Variant) -> Vector3:
	if a is Vector3:
		return a
	if a is Array and (a as Array).size() >= 3:
		return Vector3(float(a[0]), float(a[1]), float(a[2]))
	return Vector3.ZERO


static func _run_godot_ab_case(use_3d: bool, vx0: float, delta_rad: float, throttle: float, duration_s: float) -> Dictionary:
	var m := Full14DofScript.new()
	m.load_vehicle("ter26_rwd", "default")
	m.config.use_3d_kinematics = use_3d
	m.config.tire_initial_temp_c = 55.0
	m.reset()
	m.suspension.sync_from_config(m.config)
	for w in m.wheels:
		w.omega = vx0 / m.config.wheel_radius
	m.chassis_state.linear_velocity = Vector3(vx0, 0.0, 0.0)

	var inp := VehicleInputsScript.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 0
	inp.throttle = throttle
	inp.brake = 0.0
	inp.steer_angle = delta_rad

	var dt := 0.0025
	var steps := int(round(duration_s / dt))
	var max_front_t := 0.0
	var ay_peak := 0.0
	var wz_peak := 0.0
	var last_res: Dictionary = {}
	for _k in range(steps):
		last_res = m.step_standalone(inp, dt, 0.0)
		var net_f: Vector3 = last_res["forces"]["net_force_ros"]
		ay_peak = maxf(ay_peak, absf(net_f.y / m.config.mass))
		wz_peak = maxf(wz_peak, absf(m.chassis_state.angular_velocity.z))
		var t_cmd: Array = last_res.get("torque_cmd", [0.0, 0.0, 0.0, 0.0])
		if t_cmd.size() >= 2:
			max_front_t = maxf(max_front_t, maxf(absf(float(t_cmd[0])), absf(float(t_cmd[1]))))

	var net_f_end: Vector3 = last_res["forces"]["net_force_ros"]
	var vx_end := m.chassis_state.linear_velocity.x
	var vy_end := m.chassis_state.linear_velocity.y
	var wz_end := m.chassis_state.angular_velocity.z
	var ksum := DoubleWishboneScript.compute_telemetry_summary(m.suspension.corners, delta_rad, m.config, m.suspension.kinematics_3d)
	return {
		"vx_end": vx_end,
		"ay_end": net_f_end.y / m.config.mass,
		"wz_end": wz_end,
		"ay_peak": ay_peak,
		"wz_peak": wz_peak,
		"beta_end_deg": rad_to_deg(atan2(vy_end, maxf(vx_end, 0.5))),
		"max_front_t": max_front_t,
		"camber_fl_deg": float(ksum["camber_fl_deg"]),
		"camber_fr_deg": float(ksum["camber_fr_deg"]),
		"toe_fl_deg": float(ksum["toe_fl_deg"]),
		"toe_fr_deg": float(ksum["toe_fr_deg"]),
		"ackermann_pct": float(ksum["ackermann_pct"]),
		"mr_front": float(ksum["mr_front"]),
		"mr_rear": float(ksum["mr_rear"]),
		"lltd_pct": float(ksum["lltd_pct"]),
	}


static func run_async(tree: SceneTree) -> Dictionary:
	var st := {"name": "test_ter26_simulator_integration", "passed": 0, "failed": 0, "errors": []}

	# ------------------------------------------------------------------
	# 1. Validated Spec JSON & TER21 FINAL.xlsx Hardpoints Parity
	# ------------------------------------------------------------------
	var spec_exists := FileAccess.file_exists(SPEC_PATH)
	var spec: Dictionary = {}
	if spec_exists:
		var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(SPEC_PATH))
		if parsed is Dictionary:
			spec = parsed
	_check(spec_exists and not spec.is_empty(), "ter26_validated_spec.json exists and parses", st)

	var cfg := VehicleConfigScript.new()
	_check(cfg.load_vehicle_configs("ter26_rwd"), "VehicleConfig loads ter26_rwd", st)

	if not spec.is_empty():
		var meta: Dictionary = spec.get("metadata", {})
		var dc: Dictionary = meta.get("domain_closure_verification", {})
		var sysid: Dictionary = spec.get("sysid_params", {})
		var ssum: Dictionary = spec.get("suspension_summary", {})
		var hpts: Dictionary = spec.get("hardpoints_17", {})

		_check(
			String(meta.get("vehicle_id", "")) == "ter26_rwd"
			and bool(dc.get("closure_pass_lt_1e6_mm", false))
			and float(dc.get("max_residual_overall_mm", 999.0)) < 1.0e-6,
			"ter26_validated_spec metadata & 3D closure pass (<1e-6 mm)",
			st
		)

		var params_ok := (
			is_equal_approx(cfg.mass, float(sysid.get("mass_kg", 0.0)))
			and is_equal_approx(cfg.wheelbase, float(sysid.get("wheelbase_m", 0.0)))
			and is_equal_approx(cfg.front_weight_distribution, float(sysid.get("front_weight_distribution", 0.0)))
			and is_equal_approx(cfg.h_cog, float(sysid.get("h_cg_m", 0.0)))
			and absf(cfg.get_total_izz() - float(sysid.get("i_zz_total_kg_m2", 0.0))) < 0.05
			and absf(cfg.cl_a - float(sysid.get("cl_a_m2", 0.0))) < 1.0e-3
			and absf(cfg.cd_a - float(sysid.get("cd_a_m2", 0.0))) < 1.0e-3
			and absf(cfg.lambda_mux - float(sysid.get("lambda_mux_effective", 0.0))) < 1.0e-4
			and absf(cfg.lambda_muy - float(sysid.get("lambda_muy_effective", 0.0))) < 1.0e-4
			and absf(cfg.lambda_kya - float(sysid.get("kya_scale", 0.0))) < 1.0e-4
			and absf(cfg.lambda_camber - float(sysid.get("camber_scale", 0.0))) < 1.0e-4
			and absf(cfg.tire_relaxation_sigma_y - float(sysid.get("sigma_y_m", 0.0))) < 1.0e-4
			and absf(cfg.steering_ratio - float(sysid.get("i_steer", 0.0))) < 1.0e-3
			and absf(cfg.brake_bias_front - float(sysid.get("brake_bias_front", 0.0))) < 1.0e-4
			and cfg.drive_mode == VehicleConfigScript.DriveMode.REAR_WD
		)
		_check(params_ok, "VehicleConfig ter26_rwd matches ter26_validated_spec.json SysID parameters", st)

		var fl_spec: Dictionary = hpts.get("front_left", {}).get("iso_ros_axle_ground_m", {})
		var rl_spec: Dictionary = hpts.get("rear_left", {}).get("iso_ros_axle_ground_m", {})
		var msim_kin: Dictionary = spec.get("mirena_sim_double_wishbone_kinematics", {})
		var fl_msim: Dictionary = msim_kin.get("front_left", {})
		var rl_msim: Dictionary = msim_kin.get("rear_left", {})
		var fl_cfg: Dictionary = cfg.double_wishbone_kinematics.get("front_left", {})
		var rl_cfg: Dictionary = cfg.double_wishbone_kinematics.get("rear_left", {})
		var excel_map := {
			"lca_front": "CHAS_LowFor", "lca_rear": "CHAS_LowAft", "lbj": "UPRI_LowPnt",
			"uca_front": "CHAS_UppFor", "uca_rear": "CHAS_UppAft", "ubj": "UPRI_UppPnt",
			"pushrod_wishbone": "NSMA_PPAttPnt", "pushrod_rocker": "ROCK_RodPnt",
			"tie_outer": "UPRI_TiePnt", "tie_inner": "CHAS_TiePnt",
			"damper_chassis": "CHAS_AttPnt", "damper_rocker": "ROCK_CoiPnt",
			"rocker_axis_p1": "CHAS_RocAxi", "rocker_pivot": "CHAS_RocPiv",
			"arb_link_rocker": "NSMA_UBarAttPnt", "arb_arm_tip": "UBAR_AttPnt",
			"arb_chassis_piv": "CHAS_PivPnt",
		}
		var max_hp_err := 0.0
		for hp_k in DoubleWishboneScript.LINK_NAMES:
			var p_f_cfg := _vec3_from_arr(fl_cfg.get(hp_k, Vector3.ZERO))
			var p_f_msim := _vec3_from_arr(fl_msim.get(hp_k, Vector3.ZERO))
			var p_r_cfg := _vec3_from_arr(rl_cfg.get(hp_k, Vector3.ZERO))
			var p_r_msim := _vec3_from_arr(rl_msim.get(hp_k, Vector3.ZERO))
			max_hp_err = maxf(max_hp_err, p_f_cfg.distance_to(p_f_msim))
			max_hp_err = maxf(max_hp_err, p_r_cfg.distance_to(p_r_msim))
			if excel_map.has(hp_k):
				var ex_k: String = excel_map[hp_k]
				max_hp_err = maxf(max_hp_err, p_f_cfg.distance_to(_vec3_from_arr(fl_spec.get(ex_k, Vector3.ZERO))))
				var p_r_ax := _vec3_from_arr(rl_spec.get(ex_k, Vector3.ZERO)) + Vector3(-cfg.wheelbase, 0.0, 0.0)
				max_hp_err = maxf(max_hp_err, p_r_cfg.distance_to(p_r_ax))
		_check(
			max_hp_err < 1.0e-5
			and bool(fl_cfg.get("pushrod_on_lower", false))
			and not bool(rl_cfg.get("pushrod_on_lower", true)),
			"TER21 FINAL.xlsx 17 hardpoints & pushrod couplings match spec (<1e-5 m, max=%.6f m)" % max_hp_err,
			st
		)

		var kin := DoubleWishboneScript.new()
		_check(kin.configure(cfg.double_wishbone_kinematics, cfg.wheel_radius, cfg.wheelbase, cfg.h_cog) and kin.build_luts(),
			"TeR26 DoubleWishboneKinematics configure & LUT build succeed", st)
		var sf: Dictionary = kin.compute_static_metrics(true)
		var sr: Dictionary = kin.compute_static_metrics(false)
		_check(
			absf(float(sf.get("roll_center_z", 0.0)) * 1000.0 - float(ssum.get("h_rc_front_mm", 0.0))) < 3.5
			and absf(float(sr.get("roll_center_z", 0.0)) * 1000.0 - float(ssum.get("h_rc_rear_mm", 0.0))) < 1.0
			and absf(float(sf.get("caster_deg", 0.0)) - float(ssum.get("caster_front_deg", 0.0))) < 0.02
			and absf(float(sf.get("kpi_deg", 0.0)) - float(ssum.get("kpi_front_deg", 0.0))) < 0.02
			and absf(kin.mr_0_front - float(ssum.get("mr_spring_front", 0.0))) < 1.0e-3
			and absf(kin.mr_0_rear - float(ssum.get("mr_spring_rear", 0.0))) < 1.0e-3,
			"TeR26 static metrics match TER21 FINAL.xlsx summary (RC_f=%.2fmm, RC_r=%.2fmm, MR=%.4f/%.4f)" % [
				float(sf.get("roll_center_z", 0.0)) * 1000.0,
				float(sr.get("roll_center_z", 0.0)) * 1000.0,
				kin.mr_0_front,
				kin.mr_0_rear,
			],
			st
		)

		# ------------------------------------------------------------------
		# 2. TeR26 3D Kinematic Loop Closure across full travel & steer
		# ------------------------------------------------------------------
		var max_loop_err := 0.0
		for dz in [-0.04, -0.02, 0.0, 0.02, 0.04]:
			for rack in [-0.045, -0.022, 0.0, 0.022, 0.045]:
				var pf: Dictionary = kin.solve_pose_3d(dz, rack, true, false)
				max_loop_err = maxf(max_loop_err, float(pf["err_kp"]))
				max_loop_err = maxf(max_loop_err, float(pf["err_tie"]))
				max_loop_err = maxf(max_loop_err, float(pf["err_pr"]))
				max_loop_err = maxf(max_loop_err, float(pf["err_arb"]))
				max_loop_err = maxf(max_loop_err, float(pf["err_lca"]))
				max_loop_err = maxf(max_loop_err, float(pf["err_uca"]))
			var pr: Dictionary = kin.solve_pose_3d(dz, 0.0, false, false)
			max_loop_err = maxf(max_loop_err, float(pr["err_kp"]))
			max_loop_err = maxf(max_loop_err, float(pr["err_tie"]))
			max_loop_err = maxf(max_loop_err, float(pr["err_pr"]))
			max_loop_err = maxf(max_loop_err, float(pr["err_arb"]))
			max_loop_err = maxf(max_loop_err, float(pr["err_lca"]))
			max_loop_err = maxf(max_loop_err, float(pr["err_uca"]))
		_check(max_loop_err < 1.0e-5, "TeR26 3D kinematic loop closure max_err=%s m (<1e-5 m)" % String.num_scientific(max_loop_err), st)

	# ------------------------------------------------------------------
	# 3. X-Ray Rendered Rigid Link Lengths & Pushrod Clevis Rigidity
	# ------------------------------------------------------------------
	var car := MirenaCarScene.instantiate()
	var model := Full14DofScript.new()
	model.load_vehicle("ter26_rwd", "default")
	car.config = model.config
	car.model = model
	car.inputs = VehicleInputsScript.new()
	car.inputs.drive_mode = model.config.drive_mode
	car.center_of_mass = car._compute_center_of_mass(model.config)
	car._sync_geometry_from_config()
	car._setup_raycasts_and_wheels()

	var vis := SuspVisScript.new()
	car.add_child(vis)
	car.susp_visualizer = vis
	vis._ready()
	vis.bind_car(car)

	var car_mesh := car.get_node_or_null("CAR_MESH") as MeshInstance3D
	var orig_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null
	_check(car.set_suspension_visualizer_enabled(true) and vis.visible and vis.is_xray_enabled(),
		"TeR26 X-Ray / 3D Suspension turns ON cleanly", st)

	# Reference static pose & rod lengths
	for ci in range(4):
		model.suspension.corners[ci].delta_z = 0.0
	car.inputs.steer_angle = 0.0
	var ref_poses: Array[Dictionary] = vis.get_rendered_poses_cg_ros()
	_check(ref_poses.size() == 4, "X-Ray get_rendered_poses_cg_ros returns 4 corners on TeR26", st)

	var max_wc_err := 0.0
	var wheel_names := ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	for i in range(4):
		var wn := car.get_node_or_null(wheel_names[i]) as Node3D
		if wn != null and i < ref_poses.size():
			var wc_g: Vector3 = ref_poses[i].get("wheel_center_godot", Vector3.ZERO)
			max_wc_err = maxf(max_wc_err, wc_g.distance_to(wn.position))
	_check(max_wc_err < 0.02, "TeR26 3D wheel_center matches wheel nodes (<0.02 m, max=%.5f m)" % max_wc_err, st)

	var max_rod_err := 0.0
	var max_pr_clevis_err := 0.0
	var ref_pr_f_lca_f := (ref_poses[0]["pushrod_wishbone"] as Vector3).distance_to(ref_poses[0]["lca_front"] as Vector3)
	var ref_pr_f_lbj := (ref_poses[0]["pushrod_wishbone"] as Vector3).distance_to(ref_poses[0]["lbj"] as Vector3)
	var ref_pr_r_uca_f := (ref_poses[2]["pushrod_wishbone"] as Vector3).distance_to(ref_poses[2]["uca_front"] as Vector3)
	var ref_pr_r_ubj := (ref_poses[2]["pushrod_wishbone"] as Vector3).distance_to(ref_poses[2]["ubj"] as Vector3)

	for use_3d_mode in [true, false]:
		car.set_kinematics_3d_enabled(use_3d_mode)
		for dz in [-0.04, -0.02, 0.0, 0.02, 0.04]:
			for st_ang in [-0.38, 0.0, 0.38]:
				for ci in range(4):
					model.suspension.corners[ci].delta_z = dz
				car.inputs.steer_angle = st_ang
				var poses: Array[Dictionary] = vis.get_rendered_poses_cg_ros()
				if poses.size() != 4:
					max_rod_err = 999.0
					continue
				for ci in range(4):
					var ref_rods: Dictionary = ref_poses[ci].get("rendered_rod_lengths", {})
					var cur_rods: Dictionary = poses[ci].get("rendered_rod_lengths", {})
					for rk in RIGID_ROD_KEYS:
						max_rod_err = maxf(max_rod_err, absf(float(cur_rods.get(rk, 0.0)) - float(ref_rods.get(rk, 0.0))))
				# Front pushrod clevis locked to LCA; Rear pushrod clevis locked to UCA
				var pf0: Dictionary = poses[0]
				max_pr_clevis_err = maxf(max_pr_clevis_err, absf((pf0["pushrod_wishbone"] as Vector3).distance_to(pf0["lca_front"] as Vector3) - ref_pr_f_lca_f))
				max_pr_clevis_err = maxf(max_pr_clevis_err, absf((pf0["pushrod_wishbone"] as Vector3).distance_to(pf0["lbj"] as Vector3) - ref_pr_f_lbj))
				var pr2: Dictionary = poses[2]
				max_pr_clevis_err = maxf(max_pr_clevis_err, absf((pr2["pushrod_wishbone"] as Vector3).distance_to(pr2["uca_front"] as Vector3) - ref_pr_r_uca_f))
				max_pr_clevis_err = maxf(max_pr_clevis_err, absf((pr2["pushrod_wishbone"] as Vector3).distance_to(pr2["ubj"] as Vector3) - ref_pr_r_ubj))

	_check(max_rod_err < 1.0e-4, "TeR26 X-Ray 0 deformed arm lengths across z_w=[-40,+40]mm & steer=[-0.38,+0.38]rad (max=%.6f m)" % max_rod_err, st)
	_check(max_pr_clevis_err < 1.0e-4, "TeR26 front pushrod locked to LCA & rear pushrod locked to UCA (max=%.6f m)" % max_pr_clevis_err, st)

	car.set_suspension_visualizer_enabled(false)
	var restored_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null
	_check(not vis.visible and not vis.is_xray_enabled() and restored_mat == orig_mat,
		"TeR26 X-Ray OFF cleanly restores CAR_MESH material", st)
	car.free()

	# ------------------------------------------------------------------
	# 4. Live Sim.tscn + SimUI Hot-Switching (Setup/TV & Vehicle tabs)
	# ------------------------------------------------------------------
	var sim_inst := SimScene.instantiate()
	tree.root.add_child(sim_inst)
	await tree.process_frame
	await tree.process_frame

	var sim_car := sim_inst.get_node_or_null("Car")
	var sim_ui := sim_inst.get_node_or_null("SimUI")
	var main_menu := sim_ui.get_node_or_null("MainMenu") if sim_ui != null else null
	var tv_ctrl := main_menu.get("_tv_tuning_ctrl") as TracksideTuningControllerScript if main_menu != null else null

	_check(sim_car != null and sim_ui != null and main_menu != null and tv_ctrl != null,
		"Live Sim.tscn exposes Car, SimUI, MainMenu, and TracksideTuningController", st)

	if sim_car != null and sim_ui != null and main_menu != null and tv_ctrl != null:
		# No floating HUDs on car or SimUI
		var car_has_canvas := false
		for ch in sim_car.get_children():
			if ch is CanvasLayer:
				car_has_canvas = true
		var extra_hud := false
		for ui_ch in sim_ui.get_children():
			if ui_ch.name != "MainMenu" and ui_ch.name != "DilTimingBar" and ui_ch.name != "Stats" and ui_ch.name != "ControlsLabel" and ui_ch is Control and (ui_ch as Control).visible:
				extra_hud = true
		_check(not car_has_canvas and not extra_hud, "Zero floating HUDs or extra windows outside SimUI", st)

		# FOCUS_NONE & 100% ASCII across entire SimUI
		var preset_edit := tv_ctrl.get("_preset_name_edit") as Control
		var ascii_info := {"ok": true, "offender": ""}
		var focus_fail: Array = []
		_check_ui_ascii_and_focus(sim_ui, preset_edit, ascii_info, focus_fail)
		_check(bool(ascii_info["ok"]), "SimUI 100%% ASCII / zero emojis (offender=%s)" % String(ascii_info["offender"]), st)
		_check(focus_fail.is_empty(), "All SimUI interactive controls have FOCUS_NONE (fails=%s)" % str(focus_fail), st)

		# F8 shortcut & View checkbox untouched
		var chk_arrows := main_menu.get("_chk_arrows") as CheckButton
		var f8_ev := InputEventKey.new()
		f8_ev.pressed = true
		f8_ev.keycode = KEY_F8
		sim_car._input(f8_ev)
		var f8_toggled_on := bool(sim_car.wheel_arrows_enabled)
		sim_car._input(f8_ev)
		var f8_toggled_off := not bool(sim_car.wheel_arrows_enabled)
		_check(chk_arrows != null and chk_arrows.text.contains("[F8]") and f8_toggled_on and f8_toggled_off,
			"F8 wheel-force arrows shortcut and label untouched", st)

		# Setup / TV controls & synchronization with Vehicle & View tabs
		var veh_opt_tv := tv_ctrl.get("_veh_opt_tv") as OptionButton
		var kin_opt_tv := tv_ctrl.get("_kin_opt_tv") as OptionButton
		var chk_xray_tv := tv_ctrl.get("_chk_xray_susp_tv") as CheckButton
		var kin_lbl := tv_ctrl.get("_kin_lbl") as Label
		var veh_opt := main_menu.get("_veh_opt") as OptionButton
		var kin_opt := main_menu.get("_kin_opt") as OptionButton
		var chk_susp_vis := main_menu.get("_chk_susp_vis") as CheckButton

		var labels_ok := (
			veh_opt_tv != null and veh_opt_tv.item_count == 2
			and veh_opt_tv.get_item_text(0) == "TeR27 4WD"
			and veh_opt_tv.get_item_text(1) == "TeR26 Validado (RWD)"
			and kin_opt_tv != null and kin_opt_tv.item_count == 2
			and kin_opt_tv.get_item_text(0) == "Doble Horquilla 3D"
			and kin_opt_tv.get_item_text(1) == "Parametrico"
			and chk_xray_tv != null
			and kin_lbl != null
			and veh_opt != null and veh_opt.get_item_text(1) == "TeR26 Validado (RWD)"
			and kin_opt != null and kin_opt.get_item_text(0) == "Doble Horquilla 3D" and kin_opt.get_item_text(1) == "Parametrico"
		)
		_check(labels_ok, "Setup / TV and Vehicle tab labels & controls match spec cleanly", st)

		# Hot-switch TeR27 -> TeR26 via Setup / TV
		if veh_opt_tv != null:
			veh_opt_tv.select(1)
			tv_ctrl._on_tv_vehicle_selected(1)
		await tree.process_frame
		_check(
			sim_car.config.vehicle_id == "ter26_rwd"
			and sim_car.drive_mode == VehicleConfigScript.DriveMode.REAR_WD
			and sim_car.is_kinematics_3d_enabled()
			and veh_opt != null and veh_opt.selected == 1
			and kin_opt != null and kin_opt.selected == 0
			and kin_lbl != null and kin_lbl.text.contains("3D-DW") and kin_lbl.text.contains("LLTD:") and kin_lbl.text.contains("Ack:") and kin_lbl.text.contains("MR F/R:"),
			"Setup/TV switch to TeR26 Validado (RWD) syncs Vehicle tab & live kinematics telemetry (%s)" % (kin_lbl.text.replace("\n", " | ") if kin_lbl != null else ""),
			st
		)

		# Switch Suspension Model -> Parametrico via Setup / TV & toggle X-Ray ON
		if kin_opt_tv != null:
			kin_opt_tv.select(1)
			tv_ctrl._on_tv_kinematics_selected(1)
		if chk_xray_tv != null:
			chk_xray_tv.button_pressed = true
			tv_ctrl._on_tv_xray_susp_toggled(true)
		await tree.process_frame
		_check(
			not sim_car.is_kinematics_3d_enabled()
			and kin_opt != null and kin_opt.selected == 1
			and sim_car.susp_visualizer.visible
			and sim_car.susp_visualizer.is_xray_enabled()
			and chk_susp_vis != null and chk_susp_vis.button_pressed and not chk_susp_vis.disabled
			and kin_lbl != null and kin_lbl.text.contains("PARAM"),
			"Setup/TV switch to Parametrico + X-Ray ON syncs Vehicle & View tabs cleanly",
			st
		)

		# Switch Suspension Model back to Doble Horquilla 3D via Vehicle tab
		if kin_opt != null:
			kin_opt.select(0)
			main_menu._on_kin_selected(0)
		await tree.process_frame
		_check(
			sim_car.is_kinematics_3d_enabled()
			and kin_opt_tv != null and kin_opt_tv.selected == 0
			and sim_car.susp_visualizer.visible
			and kin_lbl != null and kin_lbl.text.contains("3D-DW"),
			"Vehicle tab switch back to Doble Horquilla 3D syncs Setup/TV while keeping X-Ray active",
			st
		)

		# Turn X-Ray OFF & switch back to TeR27 4WD via Setup / TV
		if chk_xray_tv != null:
			chk_xray_tv.button_pressed = false
			tv_ctrl._on_tv_xray_susp_toggled(false)
		if veh_opt_tv != null:
			veh_opt_tv.select(0)
			tv_ctrl._on_tv_vehicle_selected(0)
		await tree.process_frame
		_check(
			sim_car.config.vehicle_id == "ter27_4wd"
			and sim_car.sim_drive_mode == MirenaCarDynamicsScript.SimDriveMode.AUTOCROSS
			and not sim_car.susp_visualizer.visible
			and not sim_car.susp_visualizer.is_xray_enabled()
			and veh_opt != null and veh_opt.selected == 0,
			"Setup/TV hot-switch back to TeR27 4WD restores AUTOCROSS & cleans X-Ray",
			st
		)

	sim_inst.queue_free()
	await tree.process_frame

	# ------------------------------------------------------------------
	# 5. TeR26 Physics Parity vs Python Validated Model + Godot A/B (3D vs Parametric)
	# ------------------------------------------------------------------
	var step_3d := _run_godot_ab_case(true, 15.0, 0.06, 0.25, 2.0)
	var step_par := _run_godot_ab_case(false, 15.0, 0.06, 0.25, 2.0)
	var skid_3d := _run_godot_ab_case(true, 10.5, 0.12, 0.20, 3.0)
	var skid_par := _run_godot_ab_case(false, 10.5, 0.12, 0.20, 3.0)

	# Asymmetric bump-steer A/B probe
	var m_b3 := Full14DofScript.new(); m_b3.load_vehicle("ter26_rwd", "default"); m_b3.config.use_3d_kinematics = true; m_b3.suspension.sync_from_config(m_b3.config)
	m_b3.suspension.evaluate_from_corners([0.025, -0.020, 0.010, -0.010], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m_b3.config, 0.14)
	var bump_3d := DoubleWishboneScript.compute_telemetry_summary(m_b3.suspension.corners, 0.14, m_b3.config, m_b3.suspension.kinematics_3d)

	var m_bp := Full14DofScript.new(); m_bp.load_vehicle("ter26_rwd", "default"); m_bp.config.use_3d_kinematics = false; m_bp.suspension.sync_from_config(m_bp.config)
	m_bp.suspension.evaluate_from_corners([0.025, -0.020, 0.010, -0.010], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m_bp.config, 0.14)
	var bump_par := DoubleWishboneScript.compute_telemetry_summary(m_bp.suspension.corners, 0.14, m_bp.config, m_bp.suspension.kinematics_3d)

	# Python TeR26HighFidelityCasADi reference targets (verified in compare_ter26_godot_ab.py):
	# Step-Steer (15 m/s, 0.06 rad, 2.0 s): ay_end=9.694 m/s^2, wz_end=0.594 rad/s
	# Skidpad    (10.5 m/s, 0.12 rad, 3.0 s): ay_end=9.802 m/s^2, wz_end=0.818 rad/s
	var step_ay_err_pct := absf(float(step_3d["ay_end"]) - 9.6941) / 9.6941 * 100.0
	var step_wz_err_pct := absf(float(step_3d["wz_end"]) - 0.5939) / 0.5939 * 100.0
	var skid_ay_err_pct := absf(float(skid_3d["ay_end"]) - 9.8023) / 9.8023 * 100.0
	var skid_wz_err_pct := absf(float(skid_3d["wz_end"]) - 0.8184) / 0.8184 * 100.0

	_check(
		is_zero_approx(float(step_3d["max_front_t"])) and is_zero_approx(float(skid_3d["max_front_t"])),
		"TeR26 remains strictly RWD (T_FL = T_FR = 0.0 Nm) during step-steer & skidpad",
		st
	)
	_check(
		step_ay_err_pct < 10.0 and step_wz_err_pct < 10.0 and skid_ay_err_pct < 10.0 and skid_wz_err_pct < 10.0
		and float(step_3d["beta_end_deg"]) < 0.0 and float(skid_3d["beta_end_deg"]) < 0.0,
		"TeR26 Godot 14-DOF parity vs Python validated model (<10%%: step ay=%.2f%% wz=%.2f%%, skid ay=%.2f%% wz=%.2f%%)" % [
			step_ay_err_pct, step_wz_err_pct, skid_ay_err_pct, skid_wz_err_pct
		],
		st
	)

	var d_skid_ay := float(skid_3d["ay_end"]) - float(skid_par["ay_end"])
	var d_skid_cam_fl := float(skid_3d["camber_fl_deg"]) - float(skid_par["camber_fl_deg"])
	var d_skid_lltd := float(skid_3d["lltd_pct"]) - float(skid_par["lltd_pct"])
	var d_bump_cam_fl := float(bump_3d["camber_fl_deg"]) - float(bump_par["camber_fl_deg"])
	var d_bump_ack := float(bump_3d["ackermann_pct"]) - float(bump_par["ackermann_pct"])
	var d_bump_lltd := float(bump_3d["lltd_pct"]) - float(bump_par["lltd_pct"])

	_check(
		absf(d_skid_cam_fl) > 0.25 and absf(d_skid_lltd) > 0.05
		and absf(d_bump_cam_fl) > 0.30 and absf(d_bump_ack) > 1.5 and absf(d_bump_lltd) > 0.10,
		"TeR26 A/B 3D vs Parametric captures real camber/Ackermann/LLTD deltas (skid dCamFL=%+.3fdeg dLLTD=%+.3f%% dAy=%+.3fm/s2 | bump dCamFL=%+.3fdeg dAck=%+.2f%% dLLTD=%+.3f%%)" % [
			d_skid_cam_fl, d_skid_lltd, d_skid_ay, d_bump_cam_fl, d_bump_ack, d_bump_lltd
		],
		st
	)

	print("[TeR26 A/B] Step-Steer 3D: ay=%.3f m/s2 wz=%.3f rad/s camFL/FR=%.2f/%.2f deg LLTD=%.2f%% | Param: ay=%.3f wz=%.3f camFL/FR=%.2f/%.2f LLTD=%.2f%%" % [
		float(step_3d["ay_end"]), float(step_3d["wz_end"]), float(step_3d["camber_fl_deg"]), float(step_3d["camber_fr_deg"]), float(step_3d["lltd_pct"]),
		float(step_par["ay_end"]), float(step_par["wz_end"]), float(step_par["camber_fl_deg"]), float(step_par["camber_fr_deg"]), float(step_par["lltd_pct"]),
	])
	print("[TeR26 A/B] Skidpad    3D: ay=%.3f m/s2 wz=%.3f rad/s camFL/FR=%.2f/%.2f deg LLTD=%.2f%% | Param: ay=%.3f wz=%.3f camFL/FR=%.2f/%.2f LLTD=%.2f%%" % [
		float(skid_3d["ay_end"]), float(skid_3d["wz_end"]), float(skid_3d["camber_fl_deg"]), float(skid_3d["camber_fr_deg"]), float(skid_3d["lltd_pct"]),
		float(skid_par["ay_end"]), float(skid_par["wz_end"]), float(skid_par["camber_fl_deg"]), float(skid_par["camber_fr_deg"]), float(skid_par["lltd_pct"]),
	])
	print("[TeR26 A/B] Bump-Steer 3D: camFL/FR=%.2f/%.2f deg Ack=%.2f%% MR=%.3f/%.3f LLTD=%.2f%% | Param: camFL/FR=%.2f/%.2f Ack=%.2f%% MR=%.3f/%.3f LLTD=%.2f%%" % [
		float(bump_3d["camber_fl_deg"]), float(bump_3d["camber_fr_deg"]), float(bump_3d["ackermann_pct"]), float(bump_3d["mr_front"]), float(bump_3d["mr_rear"]), float(bump_3d["lltd_pct"]),
		float(bump_par["camber_fl_deg"]), float(bump_par["camber_fr_deg"]), float(bump_par["ackermann_pct"]), float(bump_par["mr_front"]), float(bump_par["mr_rear"]), float(bump_par["lltd_pct"]),
	])

	return st
