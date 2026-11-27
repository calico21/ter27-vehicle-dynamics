extends RefCounted

const MirenaCarScene = preload("res://Scenes/Vehicle/mirena_car.tscn")
const SimUIScene = preload("res://Scenes/Sim/UI/SimUI.tscn")
const SimScene = preload("res://Scenes/Sim/Sim.tscn")
const MirenaCarDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const SuspVisScript = preload("res://Scenes/Vehicle/Visuals/SuspensionVisualizer3D.gd")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const DoubleWishboneScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/DoubleWishboneKinematics.gd")

const POSE_VECTOR_KEYS := [
	"lca_front", "lca_rear", "lbj",
	"uca_front", "uca_rear", "ubj",
	"tie_inner", "tie_outer",
	"pushrod_wishbone", "pushrod_rocker",
	"rocker_pivot", "damper_rocker", "damper_chassis",
	"arb_link_rocker", "arb_arm_tip", "arb_chassis_piv",
	"wheel_center", "contact_patch",
]


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func _is_vec3_finite(v: Vector3) -> bool:
	return is_finite(v.x) and is_finite(v.y) and is_finite(v.z)


static func _link_lengths(p: Dictionary) -> Dictionary:
	return {
		"lca_f": (p["lbj"] as Vector3).distance_to(p["lca_front"] as Vector3),
		"lca_r": (p["lbj"] as Vector3).distance_to(p["lca_rear"] as Vector3),
		"uca_f": (p["ubj"] as Vector3).distance_to(p["uca_front"] as Vector3),
		"uca_r": (p["ubj"] as Vector3).distance_to(p["uca_rear"] as Vector3),
		"tie": (p["tie_outer"] as Vector3).distance_to(p["tie_inner"] as Vector3),
		"pushrod": (p["pushrod_rocker"] as Vector3).distance_to(p["pushrod_wishbone"] as Vector3),
	}


static func run() -> Dictionary:
	var st := {"name": "test_suspension_visualizer_and_modes", "passed": 0, "failed": 0, "errors": []}

	# ------------------------------------------------------------------
	# 1. DoubleWishboneKinematics.compute_visual_pose_3d() checks
	# ------------------------------------------------------------------
	var cfg := VehicleConfigScript.new()
	cfg.load_vehicle_configs("ter27_4wd")
	var kin := DoubleWishboneScript.new()
	_check(kin.configure(cfg.double_wishbone_kinematics, cfg.wheel_radius, cfg.wheelbase, cfg.h_cog), "Kinematics configured", st)
	_check(kin.build_luts(), "Kinematics LUTs built", st)

	for axle_front in [true, false]:
		var ref_left := kin.compute_visual_pose_3d(axle_front, true, 0.0, 0.0, cfg.a_cog)
		var ref_right := kin.compute_visual_pose_3d(axle_front, false, 0.0, 0.0, cfg.a_cog)
		_check(not ref_left.is_empty() and not ref_right.is_empty(), "Pose non-empty for axle_front=%s" % str(axle_front), st)

		# Y-symmetry between left & right at steer=0
		var sym_ok := true
		for k in POSE_VECTOR_KEYS:
			var pl: Vector3 = ref_left[k]
			var pr: Vector3 = ref_right[k]
			if absf(pl.x - pr.x) > 1e-4 or absf(pl.y + pr.y) > 1e-4 or absf(pl.z - pr.z) > 1e-4:
				sym_ok = false
				break
		_check(sym_ok, "Left/Right Y-symmetry holds for axle_front=%s" % str(axle_front), st)

		var ref_len := _link_lengths(ref_left)
		var all_finite := true
		var rigid_ok := true
		var max_err := 0.0
		var steers := [-0.38, 0.0, 0.38] if axle_front else [0.0]
		for dz in [-0.04, -0.02, 0.0, 0.02, 0.04]:
			for delta_s in steers:
				var pose := kin.compute_visual_pose_3d(axle_front, true, dz, delta_s, cfg.a_cog)
				for k in POSE_VECTOR_KEYS:
					if not _is_vec3_finite(pose.get(k, Vector3(NAN, NAN, NAN))):
						all_finite = false
				if not is_finite(float(pose.get("camber_rad", NAN))) or not is_finite(float(pose.get("wheel_steer_rad", NAN))):
					all_finite = false
				var cur_len := _link_lengths(pose)
				for lk in ref_len.keys():
					var err := absf(float(cur_len[lk]) - float(ref_len[lk]))
					max_err = maxf(max_err, err)
					if err > 1e-4:
						rigid_ok = false
		_check(all_finite, "All 18 visual pose points finite across travel/steer (front=%s)" % str(axle_front), st)
		_check(rigid_ok, "Rigid link lengths preserved (<1e-4 m, max=%.6f m, front=%s)" % [max_err, str(axle_front)], st)

	# ------------------------------------------------------------------
	# 2. Default AUTOCROSS mode, and 6-mode cycling + 20 physics steps
	# ------------------------------------------------------------------
	var car := MirenaCarScene.instantiate()
	var model := Full14DofScript.new(cfg)
	car.config = cfg
	car.model = model

	_check(car.sim_drive_mode == MirenaCarDynamicsScript.SimDriveMode.AUTOCROSS, "ter27_4wd starts in AUTOCROSS mode", st)
	_check(car.ecu_driving_mode == 2 and car.tv_4wd_mode == 0, "AUTOCROSS maps to ecu_driving_mode=2, tv_4wd_mode=0", st)
	_check(car.get_sim_drive_mode_name() == "AUTOCROSS", "Mode label is AUTOCROSS", st)

	var expected_names := ["OPEN 4WD", "AUTOCROSS", "ENDURANCE", "SKIDPAD", "ACCELERATION", "TIRE WARMER"]
	var expected_tv := [0, 0, 2, 1, 3, 4]
	var expected_ecu := [0, 2, 2, 2, 2, 2]
	var modes_ok := true
	var physics_ok := true
	var inp := VehicleInputsScript.new()
	inp.drive_mode = cfg.drive_mode
	inp.throttle = 0.7
	inp.brake = 0.0
	inp.steer_angle = 0.15

	for m_idx in range(6):
		car.set_sim_drive_mode(m_idx)
		if car.sim_drive_mode != m_idx or car.get_sim_drive_mode_name() != expected_names[m_idx]:
			modes_ok = false
		if car.tv_4wd_mode != expected_tv[m_idx] or car.ecu_driving_mode != expected_ecu[m_idx]:
			modes_ok = false

		model.reset()
		model.chassis_state.linear_velocity = Vector3(18.0, 0.0, 0.0)
		for w in model.wheels:
			w.omega = 18.0 / cfg.wheel_radius
		inp.ecu_driving_mode = car.ecu_driving_mode
		inp.tv_4wd_mode = car.tv_4wd_mode

		for _step in range(20):
			model.step_standalone(inp, 0.0025, 0.0)
			var lv_s: Vector3 = model.chassis_state.linear_velocity
			var av_s: Vector3 = model.chassis_state.angular_velocity
			if not _is_vec3_finite(lv_s) or not _is_vec3_finite(av_s):
				physics_ok = false
			if model.ecu_tv_4wd != null:
				var p_kw: float = float(model.ecu_tv_4wd.last_power_w) * 0.001
				if not is_finite(p_kw) or p_kw < -25.1 or p_kw > 80.1:
					physics_ok = false

	_check(modes_ok, "All 6 SimDriveModes map cleanly without state errors", st)
	_check(physics_ok, "20 physics steps per mode finite with HV power in [-25.1, +80.1] kW", st)

	var wrapped := int(car.cycle_sim_drive_mode())
	_check(wrapped == 0 and car.get_sim_drive_mode_name() == "OPEN 4WD", "cycle_sim_drive_mode wraps 5 -> 0 (OPEN 4WD)", st)

	# ------------------------------------------------------------------
	# 3. SuspensionVisualizer3D, X-Ray toggle & ter26_rwd fallback
	# ------------------------------------------------------------------
	var vis := SuspVisScript.new()
	car.add_child(vis)
	car.susp_visualizer = vis
	vis._ready()
	vis.bind_car(car)

	_check(vis.is_supported(), "SuspensionVisualizer3D supported on ter27_4wd", st)
	_check(not vis.visible, "SuspensionVisualizer3D starts hidden (zero idle overhead)", st)

	var car_mesh := car.get_node_or_null("CAR_MESH") as MeshInstance3D
	var orig_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null

	var vis_on := bool(car.toggle_suspension_visualizer())
	_check(vis_on and vis.visible and vis.is_xray_enabled(), "Toggle ON shows 3D suspension and enables X-Ray", st)
	var surf_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null
	if surf_mat is StandardMaterial3D:
		var xmat := surf_mat as StandardMaterial3D
		_check(xmat.albedo_color.a < 0.25, "CAR_MESH alpha lowered for X-Ray (a=%.2f)" % xmat.albedo_color.a, st)
	else:
		_check(false, "CAR_MESH surface_override_material(0) should be transparent StandardMaterial3D in X-Ray", st)

	vis.update_from_car()
	var snap_on: Dictionary = car.get_hud_snapshot()
	var snap_corners: Array = snap_on.get("corners", [])
	_check(snap_corners.size() == 4 and is_finite(float(snap_corners[0].get("fz_n", NAN))), "get_hud_snapshot returns 4 valid corner metrics", st)
	var vis_off := bool(car.toggle_suspension_visualizer())
	_check(not vis_off and not vis.visible and not vis.is_xray_enabled(), "Toggle OFF hides 3D suspension and disables X-Ray", st)
	var restored_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null
	_check(car_mesh != null and restored_mat == orig_mat, "CAR_MESH original surface_override_material restored on X-Ray OFF", st)

	# 3D Double Wishbone & X-Ray support on ter26_rwd
	var cfg_rwd := VehicleConfigScript.new()
	var switched := cfg_rwd.load_vehicle_configs("ter26_rwd")
	_check(switched, "Loaded ter26_rwd config", st)
	car.config = cfg_rwd
	car.model = Full14DofScript.new(cfg_rwd)
	car.center_of_mass = car._compute_center_of_mass(cfg_rwd)
	_check(cfg_rwd.use_3d_kinematics and car.is_kinematics_3d_supported() and car.is_kinematics_3d_enabled(), "ter26_rwd enables 3D Double Wishbone kinematics by default", st)
	_check(vis.is_supported(), "SuspensionVisualizer3D reports supported on ter26_rwd", st)
	var rwd_toggle_on := bool(car.toggle_suspension_visualizer())
	_check(rwd_toggle_on and vis.visible and vis.is_xray_enabled(), "Toggle ON on ter26_rwd shows 3D suspension and enables X-Ray", st)
	var rwd_poses: Array = vis.get_rendered_poses_cg_ros()
	_check(rwd_poses.size() == 4 and bool(rwd_poses[0].get("is_3d_kinematics", false)), "ter26_rwd rendered poses report 4 corners in 3D mode", st)
	var rwd_toggle_off := bool(car.toggle_suspension_visualizer())
	_check(not rwd_toggle_off and not vis.visible and not vis.is_xray_enabled(), "Toggle OFF on ter26_rwd hides 3D suspension and restores CAR_MESH", st)
	var restored_rwd_mat: Material = car_mesh.get_surface_override_material(0) if (car_mesh != null and car_mesh.mesh != null and car_mesh.mesh.get_surface_count() > 0) else null
	_check(car_mesh != null and restored_rwd_mat == orig_mat, "CAR_MESH material restored after ter26_rwd X-Ray OFF", st)

	# ------------------------------------------------------------------
	# 4. No floating top HUD, Sim.tscn ter27_4wd default, wheel_center alignment, & SimUI tabs
	# ------------------------------------------------------------------
	var has_hud_canvas := false
	for ch in car.get_children():
		if ch is CanvasLayer:
			has_hud_canvas = true
	_check(not has_hud_canvas, "No floating CanvasLayer HUD attached to MirenaCar", st)

	# Check wheel_center alignment (< 0.02 m) on ter27_4wd
	car.config = cfg
	car.model = model
	car.center_of_mass = Vector3(0.0, cfg.h_cog, -0.117)
	car._sync_geometry_from_config()
	car._setup_raycasts_and_wheels()
	var wheel_names := ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	var max_wc_err := 0.0
	for i in range(4):
		var wn := car.get_node_or_null(wheel_names[i]) as Node3D
		var pose_i := kin.compute_visual_pose_3d(i < 2, (i % 2 == 0), 0.0, 0.0, cfg.a_cog)
		var wc_g: Vector3 = car.center_of_mass + VehicleConfigScript.ros_to_godot_vec(pose_i["wheel_center"])
		if wn != null:
			max_wc_err = maxf(max_wc_err, wc_g.distance_to(wn.position))
	_check(max_wc_err < 0.02, "3D wheel_center aligns with MirenaCar wheel nodes (<0.02m, max=%.5fm)" % max_wc_err, st)

	# Check Sim.tscn spawns Car with drive_mode=0 (FOUR_WD -> ter27_4wd)
	var sim_root := SimScene.instantiate()
	var sim_car := sim_root.get_node_or_null("Car")
	_check(sim_car != null and int(sim_car.get("drive_mode")) == 0, "Sim.tscn Car defaults to drive_mode=0 (ter27_4wd)", st)

	# Check Sim.tscn ground covers >= 800 x 800 m (>= 350 m half-width in X and Z) and cameras have far >= 1200 m
	var ground_mesh_node := sim_root.get_node_or_null("Track/Ground/GroundMesh") as MeshInstance3D
	var ground_plane := ground_mesh_node.mesh as PlaneMesh if ground_mesh_node != null else null
	_check(ground_plane != null and ground_plane.size.x >= 800.0 and ground_plane.size.y >= 800.0,
		"Sim.tscn GroundMesh PlaneMesh is >= 800x800 m", st)
	var ground_col_node := sim_root.get_node_or_null("Track/Ground/CollisionShape3D") as CollisionShape3D
	var col_covers := false
	if ground_col_node != null and ground_col_node.shape is ConvexPolygonShape3D:
		var pts := (ground_col_node.shape as ConvexPolygonShape3D).points
		var min_x := INF
		var max_x := -INF
		var min_z := INF
		var max_z := -INF
		for p in pts:
			min_x = minf(min_x, p.x)
			max_x = maxf(max_x, p.x)
			min_z = minf(min_z, p.z)
			max_z = maxf(max_z, p.z)
		col_covers = (min_x <= -350.0 and max_x >= 350.0 and min_z <= -350.0 and max_z >= 350.0)
	elif ground_col_node != null and ground_col_node.shape is BoxShape3D:
		var bsize := (ground_col_node.shape as BoxShape3D).size
		col_covers = (bsize.x >= 700.0 and bsize.z >= 700.0)
	_check(col_covers, "Sim.tscn Ground CollisionShape3D covers >= 350 m half-width in X and Z", st)
	var freecam := sim_root.get_node_or_null("Freecam3D") as Camera3D
	var fpcam := sim_car.get_node_or_null("FPCam") as Camera3D if sim_car != null else null
	var tpcam := sim_car.get_node_or_null("TPCam") as Camera3D if sim_car != null else null
	_check(freecam != null and freecam.far >= 1200.0 and fpcam != null and fpcam.far >= 1200.0 and tpcam != null and tpcam.far >= 1200.0,
		"All scene cameras have far >= 1200.0 m", st)

	# Check TrackPreset OptionButton in SimUI.tscn
	var sim_ui := SimUIScene.instantiate()
	var main_menu := sim_ui.get_node_or_null("MainMenu")
	var tabs := sim_ui.get_node_or_null("MainMenu/MarginContainer/VBoxContainer/MenuTabs") as TabContainer
	# Children _ready() runs before parent _ready() in SceneTree
	var track_tab := sim_ui.get_node_or_null("MainMenu/MarginContainer/VBoxContainer/MenuTabs/TRACK")
	if track_tab != null and track_tab.has_method("_ready"):
		track_tab._ready()
	if main_menu != null and main_menu.has_method("_ready"):
		main_menu._ready()
	var track_preset_opt := track_tab.find_child("TrackPreset", true, false) as OptionButton if track_tab != null else null
	_check(track_preset_opt != null and track_preset_opt.focus_mode == Control.FOCUS_NONE,
		"TrackPreset OptionButton exists and has FOCUS_NONE", st)
	var found_fsg := false
	var found_skidpad := false
	var found_accel := false
	var all_presets_loadable := true
	var sim_track := sim_root.get_node_or_null("Track")
	if sim_track != null and sim_car != null and sim_track.is_connected("track_loaded", Callable(sim_car, "reset_position")):
		sim_track.disconnect("track_loaded", Callable(sim_car, "reset_position"))
	if track_preset_opt != null:
		for pi in range(track_preset_opt.item_count):
			var p_lbl := track_preset_opt.get_item_text(pi)
			var p_meta := str(track_preset_opt.get_item_metadata(pi))
			if p_meta.is_empty():
				continue
			if not FileAccess.file_exists(p_meta):
				all_presets_loadable = false
			if p_lbl.contains("FSG 2023"):
				found_fsg = true
			elif p_lbl.contains("Skidpad"):
				found_skidpad = true
			elif p_lbl.contains("Acceleration"):
				found_accel = true
			if p_lbl.contains("FSG 2023") or p_lbl.contains("Skidpad") or p_lbl.contains("Acceleration"):
				if sim_track != null and sim_track.has_method("load_track"):
					sim_track.load_track(p_meta)
					if sim_track.get_node("Gates").get_child_count() <= 0:
						all_presets_loadable = false
	_check(found_fsg and found_skidpad and found_accel,
		"TrackPreset lists FSG 2023, Skidpad, and Acceleration", st)
	_check(all_presets_loadable, "All TrackPreset paths exist and load cleanly", st)
	sim_root.free()
	var tab_titles := []
	if tabs != null:
		for ti in range(tabs.get_tab_count()):
			tab_titles.append(tabs.get_tab_title(ti))
	var expected_tabs := ["Vehicle", "Setup / TV", "Track", "View", "Audio", "Controls", "Telemetry", "AS / Sys", "LTS"]
	_check(tab_titles == expected_tabs, "Options Menu has exact 9 tabs %s (got %s)" % [str(expected_tabs), str(tab_titles)], st)

	var footer_reset := sim_ui.get_node_or_null("MainMenu/MarginContainer/VBoxContainer/ResetCarButton") as Button
	_check(footer_reset != null and footer_reset.text == "Reset Car Position", "Fixed footer contains Reset Car Position button", st)

	var all_focus_none := true
	if main_menu != null:
		var stack: Array[Node] = [main_menu]
		while not stack.is_empty():
			var n: Node = stack.pop_back()
			if n is BaseButton or n is Range or n is TabContainer or n is TabBar or n is ScrollContainer or n is LineEdit:
				if (n as Control).focus_mode != Control.FOCUS_NONE:
					all_focus_none = false
			for ch in n.get_children():
				stack.append(ch)
	_check(all_focus_none, "All interactive Options Menu controls have FOCUS_NONE", st)

	var mode_opt = main_menu.get("_mode_opt") if main_menu != null else null
	_check(mode_opt != null and mode_opt.item_count == 6, "Vehicle tab Drive Mode OptionButton has all 6 official modes", st)
	sim_ui.free()

	# ------------------------------------------------------------------
	# 5. Hot-switching: TeR27 <-> TeR26, 14-DOF <-> 3-DOF, 3D <-> Parametric, SetupProfile
	# ------------------------------------------------------------------
	var sw_rwd := bool(car.switch_vehicle("ter26_rwd"))
	_check(sw_rwd and car.config.vehicle_id == "ter26_rwd" and car.is_kinematics_3d_supported() and car.is_kinematics_3d_enabled() and vis.is_supported(), "Hot-switch to ter26_rwd enables 3D kinematics & visualizer cleanly", st)
	var vis_rwd_on := bool(car.set_suspension_visualizer_enabled(true))
	var k_rwd_param := bool(car.set_kinematics_3d_enabled(false))
	var poses_rwd_param: Array = vis.get_rendered_poses_cg_ros()
	_check(vis_rwd_on and k_rwd_param and not car.is_kinematics_3d_enabled() and vis.is_supported() and vis.visible and poses_rwd_param.size() == 4 and not bool(poses_rwd_param[0].get("is_3d_kinematics", true)), "ter26_rwd Parametric mode keeps 3D visualizer coherent without physics 3D", st)
	var k_rwd_3d := bool(car.set_kinematics_3d_enabled(true))
	var poses_rwd_3d: Array = vis.get_rendered_poses_cg_ros()
	_check(k_rwd_3d and car.is_kinematics_3d_enabled() and vis.visible and poses_rwd_3d.size() == 4 and bool(poses_rwd_3d[0].get("is_3d_kinematics", false)), "ter26_rwd switch back to 3D Double Wishbone restores 3D kinematics & coherent visualizer", st)
	car.set_suspension_visualizer_enabled(false)

	var sw_4wd := bool(car.switch_vehicle("ter27_4wd"))
	_check(sw_4wd and car.config.vehicle_id == "ter27_4wd" and car.sim_drive_mode == MirenaCarDynamicsScript.SimDriveMode.AUTOCROSS and car.is_kinematics_3d_supported(), "Hot-switch back to ter27_4wd restores AUTOCROSS & 3D kinematics", st)

	# 14-DOF <-> 3-DOF
	var sw_3dof := bool(car.set_physics_fidelity("fast_3dof"))
	_check(sw_3dof and car.physics_fidelity == "fast_3dof" and not car.is_kinematics_3d_supported(), "Hot-switch to fast_3dof succeeds and disables 3D kinematics", st)
	var sw_14dof := bool(car.set_physics_fidelity("full_14dof"))
	_check(sw_14dof and car.physics_fidelity == "full_14dof" and car.is_kinematics_3d_supported(), "Hot-switch back to full_14dof succeeds and restores 3D kinematics", st)

	# 3D Double Wishbone <-> Parametric A/B
	var k_param := bool(car.set_kinematics_3d_enabled(false))
	_check(k_param and not car.is_kinematics_3d_enabled() and vis.is_supported(), "A/B switch to Parametric disables 3D physics kinematics while keeping visualizer supported", st)
	var k_3d := bool(car.set_kinematics_3d_enabled(true))
	_check(k_3d and car.is_kinematics_3d_enabled() and vis.is_supported(), "A/B switch back to 3D Double Wishbone re-enables 3D kinematics & visualizer", st)

	# SetupProfile listing
	var setups: Array = car.list_setup_profiles()
	_check(setups.size() >= 1, "list_setup_profiles returns >= 1 profile for ter27_4wd (got %d)" % setups.size(), st)

	car.free()
	return st
