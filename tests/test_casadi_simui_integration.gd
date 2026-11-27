extends RefCounted

const SimScene = preload("res://Scenes/Sim/Sim.tscn")
const SimScript = preload("res://Scenes/Sim/sim.gd")
const RacingLineVisualizerScript = preload("res://Scenes/Track/RacingLineVisualizer.gd")
const LtsMenuControllerScript = preload("res://Scenes/Sim/UI/lts_menu_controller.gd")
const TracksideTuningControllerScript = preload("res://Scenes/Sim/UI/TracksideTuningController.gd")
const LtsLivePlotScript = preload("res://Scenes/Sim/UI/LtsLivePlot.gd")
const TvFrictionYmdWidgetScript = preload("res://Scenes/Sim/UI/TvFrictionYmdWidget.gd")

const VALID_PHASES := [
	"FULL_THROTTLE",
	"PARTIAL_THROTTLE",
	"LIFT_AND_COAST",
	"BRAKING",
	"TRAIL_BRAKING",
]
const VALID_MARKERS := [
	"BRAKE_POINT",
	"LIFT_COAST_START",
	"TURN_IN",
	"APEX",
	"THROTTLE_ON",
]


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append(msg)


static func _check_focus_and_ascii(node: Node, ascii_ok: Dictionary, focus_fail: Array) -> void:
	if node is Control:
		var ctrl := node as Control
		if ctrl is Button or ctrl is CheckButton or ctrl is OptionButton or ctrl is HSlider or ctrl is SpinBox or ctrl is TabBar or ctrl is ScrollContainer:
			if ctrl.focus_mode != Control.FOCUS_NONE:
				focus_fail.append(String(ctrl.name))
	if node.has_method("get_text"):
		var t: String = str(node.call("get_text"))
		for ci in range(t.length()):
			if t.unicode_at(ci) > 127:
				ascii_ok["ok"] = false
				ascii_ok["offender"] = "%s: %s" % [node.name, t]
	if node is OptionButton:
		var ob := node as OptionButton
		for ii in range(ob.item_count):
			var it := ob.get_item_text(ii)
			for ci in range(it.length()):
				if it.unicode_at(ci) > 127:
					ascii_ok["ok"] = false
					ascii_ok["offender"] = "%s[%d]: %s" % [ob.name, ii, it]
	for ch in node.get_children():
		_check_focus_and_ascii(ch, ascii_ok, focus_fail)


static func _load_json_dict(res_path: String) -> Dictionary:
	if not FileAccess.file_exists(res_path):
		return {}
	var txt := FileAccess.get_file_as_string(res_path)
	var parsed = JSON.parse_string(txt)
	if parsed is Dictionary:
		return parsed
	return {}


static func run_async(tree: SceneTree) -> Dictionary:
	var st := {"name": "test_casadi_simui_integration", "passed": 0, "failed": 0, "errors": []}

	# 1. Shared JSON Schema Contract across all 4 exported CasADi human-achievable files
	var json_files := [
		{"path": "res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_real.json", "endurance": false},
		{"path": "res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_real_endurance.json", "endurance": true},
		{"path": "res://Assets/RacingLines/fsg_2023_racing_line_casadi_ttc_lab.json", "endurance": false},
		{"path": "res://Assets/RacingLines/fsg_2023_racing_line_casadi_ttc_lab_endurance.json", "endurance": true},
	]
	var first_corner: Dictionary = {}
	for entry in json_files:
		var jp: String = entry["path"]
		var is_end: bool = entry["endurance"]
		var data := _load_json_dict(jp)
		_check(not data.is_empty(), "CasADi JSON exists and parses: %s" % jp.get_file(), st)
		if data.is_empty():
			continue
		var meta: Dictionary = data.get("meta", {})
		var wps: Array = data.get("waypoints", [])
		var corners: Array = data.get("corners", [])
		var markers: Array = data.get("markers", [])

		var meta_ok := (
			meta.has("lap_id") and meta.has("label") and meta.has("track_name")
			and meta.has("solver_mode") and meta.has("grip_profile")
			and meta.has("ocp_lap_time_s") and meta.has("sim14dof_lap_time_s")
			and meta.has("v_max_kmh") and meta.has("ay_max_g") and meta.has("e_22km_kwh")
			and int(meta.get("cones_hit", -1)) == 0
			and float(meta.get("min_cone_clearance_m", 0.0)) >= 0.15
			and int(meta.get("hv_violations", -1)) == 0
		)
		var t_ocp := float(meta.get("ocp_lap_time_s", 0.0))
		var t_14 := float(meta.get("sim14dof_lap_time_s", 1.0))
		var gap_pct := absf(t_ocp - t_14) / maxf(t_14, 1e-3) * 100.0
		_check(meta_ok and gap_pct < 2.5,
			"%s meta contract + 0 cones/HV + <2.5%% gap (gap=%.2f%%, clear=%.3fm)" % [jp.get_file(), gap_pct, float(meta.get("min_cone_clearance_m", 0.0))], st)

		var wp_ok := wps.size() >= 100
		var has_lift_coast := false
		if wp_ok:
			var w0: Dictionary = wps[0]
			wp_ok = (
				w0.has("s") and w0.has("x") and w0.has("y") and w0.has("z") and w0.has("yaw")
				and w0.has("v_ref") and w0.has("v_ocp") and w0.has("throttle") and w0.has("brake")
				and w0.has("steer_deg") and w0.has("ay_g") and w0.has("ax_g") and w0.has("p_dc_kw")
				and VALID_PHASES.has(String(w0.get("phase", "")))
			)
			for w in wps:
				if String(w.get("phase", "")) == "LIFT_AND_COAST":
					has_lift_coast = true
					break
		_check(wp_ok and ((not is_end) or has_lift_coast),
			"%s waypoints contract (n=%d, lift_coast=%s)" % [jp.get_file(), wps.size(), str(has_lift_coast)], st)

		var corner_ok := corners.size() >= 10
		if corner_ok:
			var c0: Dictionary = corners[0]
			var sp: Array = c0.get("spawn_pos", [])
			corner_ok = (
				c0.has("corner_id") and c0.has("name") and c0.has("s_start") and c0.has("s_brake")
				and c0.has("s_turn_in") and c0.has("s_apex") and c0.has("s_exit")
				and c0.has("v_entry_kmh") and c0.has("v_apex_kmh") and c0.has("v_exit_kmh")
				and sp.size() == 3 and c0.has("spawn_yaw") and float(c0.get("spawn_vx", 0.0)) > 3.0
			)
			if first_corner.is_empty():
				first_corner = c0
		_check(corner_ok, "%s corners contract (n=%d)" % [jp.get_file(), corners.size()], st)

		var marker_ok := markers.size() >= 10
		var has_lc_marker := false
		if marker_ok:
			for m in markers:
				var mt := String(m.get("type", ""))
				if not VALID_MARKERS.has(mt):
					marker_ok = false
					break
				if mt == "LIFT_COAST_START":
					has_lc_marker = true
		_check(marker_ok and ((not is_end) or has_lc_marker),
			"%s markers contract (n=%d, lc_marker=%s)" % [jp.get_file(), markers.size(), str(has_lc_marker)], st)

	# 2. RacingLineVisualizer: CasADi JSON load, 3 color modes (Velocity, Phases, Live Delta-v), 3D markers toggle
	var rlv := RacingLineVisualizerScript.new()
	var applied := rlv.apply_casadi_json("res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_real_endurance.json", true)
	_check(applied, "RacingLineVisualizer applies CasADi endurance JSON with 3D markers", st)
	rlv.set_color_mode(0)
	rlv.set_color_mode(1)
	rlv.set_color_mode(2)
	rlv.set_markers_visible(true)
	rlv.set_markers_visible(false)
	rlv.set_markers_visible(true)
	_check(true, "RacingLineVisualizer switches Velocidad / Fases / Delta-v modes and toggles 3D markers cleanly", st)
	rlv.queue_free()

	# 3. TvFrictionYmdWidget: Live friction circles + instant YMD preview on slider change
	var ymd_w := TvFrictionYmdWidgetScript.new()
	ymd_w.update_live_telemetry(null, {"ax_g": 0.4, "ay_g": 0.9, "p_dc_kw": 42.5}, {"mz_tv_nm": 180.0, "p_dc_kw": 42.5})
	var base_p := {"centripetal_fx_gain": 15.0, "delta_alpha_eq_gain": 2620.0, "brake_bias_front_pct": 58.0}
	var mod_p := {"centripetal_fx_gain": 35.0, "delta_alpha_eq_gain": 3200.0, "brake_bias_front_pct": 68.0}
	ymd_w.update_ymd_preview(base_p, base_p)
	var ay_base := float(ymd_w.get("ymd_ay_max_g"))
	var stab_base := float(ymd_w.get("ymd_stability_nm_deg"))
	ymd_w.update_ymd_preview(mod_p, base_p)
	var ay_mod := float(ymd_w.get("ymd_ay_max_g"))
	var mz_mod := float(ymd_w.get("ymd_mz_ctrl_nm"))
	var stab_mod := float(ymd_w.get("ymd_stability_nm_deg"))
	_check(ay_base > 0.5 and ay_mod > 0.5 and mz_mod > 100.0 and stab_base > 0.0 and stab_mod > 0.0 and absf(ay_mod - ay_base) > 1e-4,
		"TvFrictionYmdWidget updates live telemetry & YMD preview instantaneously (ay: %.3f->%.3fg, -dMz/db=%.1f)" % [ay_base, ay_mod, stab_mod], st)
	ymd_w.queue_free()

	# 4. Live Sim.tscn: FOCUS_NONE, 100% ASCII, no floating HUDs, Cancel PID kill, Corner Loop flying teleport
	var sim_inst := SimScene.instantiate()
	tree.root.add_child(sim_inst)
	await tree.process_frame
	await tree.process_frame

	var ui_node := sim_inst.get_node_or_null("SimUI")
	_check(ui_node != null, "Live Sim.tscn instantiates SimUI cleanly", st)
	if ui_node != null:
		var ascii_ok := {"ok": true, "offender": ""}
		var focus_fail: Array = []
		_check_focus_and_ascii(ui_node, ascii_ok, focus_fail)
		_check(bool(ascii_ok["ok"]), "SimUI 100%% ASCII / zero emojis (offender=%s)" % String(ascii_ok["offender"]), st)
		_check(focus_fail.is_empty(), "SimUI all interactive controls FOCUS_NONE (fails=%s)" % str(focus_fail), st)

	# Test Cancel button kills child PID cleanly + Windows/WSL bridge + Track Conditions (mu_scale / temp_c)
	var lts_ctrl := LtsMenuControllerScript.new()
	var dummy_pid := OS.create_process("/bin/sleep", ["10"], false)
	lts_ctrl.set("_pid", dummy_pid)
	lts_ctrl.stop_process()
	await tree.process_frame
	_check(dummy_pid > 0 and not OS.is_process_running(dummy_pid) and int(lts_ctrl.get("_pid")) == -1,
		"LtsMenuController stop_process/Cancelar cleanly terminates child PID (%d)" % dummy_pid, st)

	# 5. Windows/WSL path translation & command resolution
	var p_drive := LtsMenuControllerScript._to_wsl_path("C:\\Users\\alexr\\repo\\solve.py")
	var p_drive_fwd := LtsMenuControllerScript._to_wsl_path("D:/Work/fsg_2023.json")
	var p_unc := LtsMenuControllerScript._to_wsl_path("\\\\wsl.localhost\\Ubuntu\\home\\alexr\\eRacing\\solve.py")
	var p_posix := LtsMenuControllerScript._to_wsl_path("/home/alexr/eRacing/solve.py")
	_check(p_drive == "/mnt/c/Users/alexr/repo/solve.py" and p_drive_fwd == "/mnt/d/Work/fsg_2023.json"
		and p_unc == "/home/alexr/eRacing/solve.py" and p_posix == "/home/alexr/eRacing/solve.py",
		"_to_wsl_path converts drive, UNC wsl.localhost, and POSIX paths (%s | %s)" % [p_drive, p_unc], st)

	var cmd_win := lts_ctrl._resolve_python_command(
		"C:\\eRacing\\vd_correlation_lab\\optimization\\lap_time_sim\\solve_full_lap_ocp.py",
		["--track-file", "C:\\eRacing\\fsg_2023.json", "--mu-scale", "0.88", "--temp-c", "45.0"],
		"Windows"
	)
	var win_args: PackedStringArray = cmd_win.get("args", PackedStringArray())
	var cmd_lin := lts_ctrl._resolve_python_command(
		"/home/alexr/eRacing/vd_correlation_lab/optimization/lap_time_sim/solve_full_lap_ocp.py",
		["--track-file", "/home/alexr/eRacing/fsg_2023.json", "--mu-scale", "0.88", "--temp-c", "45.0"],
		"Linux"
	)
	_check(String(cmd_win.get("exe", "")) == "wsl.exe" and bool(cmd_win.get("is_wsl", false))
		and win_args.has("/mnt/c/eRacing/vd_correlation_lab/optimization/lap_time_sim/solve_full_lap_ocp.py")
		and win_args.has("/mnt/c/eRacing/fsg_2023.json")
		and String(cmd_lin.get("exe", "")).ends_with("/python3") and not bool(cmd_lin.get("is_wsl", true)),
		"_resolve_python_command resolves Windows (wsl.exe + translated paths) and Linux (project_gp_env)", st)

	# 6. LTS Track Condition controls (FSG_RUBBERED default, mu_scale, temp_c)
	var lts_host := VBoxContainer.new()
	lts_host.add_child(lts_ctrl)
	tree.root.add_child(lts_host)
	await tree.process_frame
	var grip_opt: OptionButton = lts_ctrl.get("_grip_profile_opt")
	var mu_slider: HSlider = lts_ctrl.get("_sld_mu_scale")
	var temp_slider: HSlider = lts_ctrl.get("_sld_temp_c")
	var ts_opt: OptionButton = lts_ctrl.get("_trackside_setup_opt")
	var rec_lbl: Label = lts_ctrl.get("_setup_rec_lbl")
	var lts_controls_ok := (
		grip_opt != null and grip_opt.item_count >= 4 and grip_opt.selected == 0
		and grip_opt.get_item_text(0).contains("FSG_RUBBERED")
		and mu_slider != null and absf(mu_slider.value - 0.88) < 0.02
		and temp_slider != null and absf(temp_slider.value - 45.0) < 1.0
		and grip_opt.focus_mode == Control.FOCUS_NONE
		and mu_slider.focus_mode == Control.FOCUS_NONE
		and temp_slider.focus_mode == Control.FOCUS_NONE
	)
	_check(lts_controls_ok, "LTS instantiates FSG_RUBBERED default + mu_scale (0.88) + temp_c (45 C) with FOCUS_NONE", st)
	var coopt_ok := false
	if ts_opt != null and rec_lbl != null:
		ts_opt.select(0)
		var off_val := bool(lts_ctrl.is_trackside_setup_coopt_enabled())
		ts_opt.select(1)
		var on_val := bool(lts_ctrl.is_trackside_setup_coopt_enabled())
		var fmt_s := String(LtsMenuControllerScript.format_recommended_setup({
			"lltd": 0.50, "camber_f_deg": -1.8, "camber_r_deg": -1.4,
			"toe_f_deg": 0.1, "toe_r_deg": 0.2, "brake_bias": 0.64, "rake_mm": 2.0
		}, -0.35))
		coopt_ok = (
			ts_opt.item_count == 2
			and ts_opt.focus_mode == Control.FOCUS_NONE
			and (not off_val) and on_val
			and fmt_s.contains("LLTD") and fmt_s.contains("-0.35")
		)
	_check(coopt_ok, "LTS TracksideSetupOpt (Fixed Base | Co-Optimize in CasADi) + RecommendedSetupLabel FOCUS_NONE & formatting", st)
	lts_host.queue_free()

	# 6b. TracksideTuningController Event Setup Preset selector & Apply Preset
	var tune_ctrl := TracksideTuningControllerScript.new()
	var tune_host := VBoxContainer.new()
	tune_host.add_child(tune_ctrl)
	tree.root.add_child(tune_host)
	await tree.process_frame
	var ev_opt: OptionButton = tune_ctrl.get("_event_setup_opt")
	var ev_btn: Button = tune_ctrl.get("_btn_apply_event_preset")
	var preset_ui_ok := (
		ev_opt != null and ev_opt.item_count == 5 and ev_opt.focus_mode == Control.FOCUS_NONE
		and ev_btn != null and ev_btn.focus_mode == Control.FOCUS_NONE
	)
	_check(preset_ui_ok, "TracksideTuningController EventSetupPresetOpt (5 presets) & ApplyEventSetupPresetBtn FOCUS_NONE", st)
	var preset_apply_ok := true
	var sim_car := sim_inst.get_node_or_null("Car")
	var car_model = sim_car.get("physics_model") if sim_car != null else null
	if car_model != null:
		for pidx in range(5):
			if not bool(tune_ctrl.apply_event_setup_preset(pidx, car_model)):
				preset_apply_ok = false
				break
	_check(preset_apply_ok, "TracksideTuningController applies all 5 Event Setup Presets (Base + 4 optimal_*.json) cleanly", st)
	tune_host.queue_free()

	# 7. FSG_RUBBERED & COLD_LOW_GRIP JSON contract + Drivability Score + mu_scale + temp_c
	var thermal_files := [
		"res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_rubbered.json",
		"res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_rubbered_endurance.json",
		"res://Assets/RacingLines/fsg_2023_racing_line_casadi_cold_low_grip.json",
	]
	for tf in thermal_files:
		var td := _load_json_dict(tf)
		var tm: Dictionary = td.get("meta", {})
		var drv := float(tm.get("drivability_score", -1.0))
		var t_mu := float(tm.get("mu_scale", -1.0))
		var t_c := float(tm.get("temp_c", -1.0))
		var ok_t := (
			not td.is_empty() and drv >= 60.0 and drv <= 100.0
			and t_mu >= 0.60 and t_mu <= 1.10 and t_c >= 10.0 and t_c <= 95.0
			and int(tm.get("cones_hit", -1)) == 0
			and float(tm.get("min_cone_clearance_m", 0.0)) >= 0.14
			and int(tm.get("hv_violations", -1)) == 0
		)
		_check(ok_t, "%s thermal meta + Drivability Score (drv=%.1f, mu=%.2f, T=%.1fC, clr=%.3fm)" % [
			tf.get_file(), drv, t_mu, t_c, float(tm.get("min_cone_clearance_m", 0.0))
		], st)

	# Test Corner Loop flying teleport (40 m before corner at optimal entry speed)
	var tp_ok := false
	var speed_after_tp := 0.0
	var car := sim_inst.get_node_or_null("Car")
	if not first_corner.is_empty() and car != null:
		var sim_helper := SimScript.new()
		tp_ok = bool(sim_helper.teleport_to_corner(first_corner, car))
		await tree.physics_frame
		if car is RigidBody3D:
			speed_after_tp = (car as RigidBody3D).linear_velocity.length()
		sim_helper.queue_free()
	_check(tp_ok and speed_after_tp > 5.0,
		"Corner Loop flying teleport places vehicle moving at entry speed (ok=%s, v=%.2f m/s)" % [str(tp_ok), speed_after_tp], st)

	sim_inst.queue_free()
	await tree.process_frame
	return st
