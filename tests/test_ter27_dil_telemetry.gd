extends RefCounted

const SensorRealismFilterScript = preload("res://Scenes/Vehicle/Dynamics/Telemetry/SensorRealismFilter.gd")
const Full14DofModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const RacingLineVisualizerScript = preload("res://Scenes/Track/RacingLineVisualizer.gd")
const DilSessionManagerScript = preload("res://Scenes/Sim/DilSessionManager.gd")
const TracksideTuningControllerScript = preload("res://Scenes/Sim/UI/TracksideTuningController.gd")
const SimUIScene = preload("res://Scenes/Sim/UI/SimUI.tscn")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append(msg)


static func _check_controls_focus_and_ascii(node: Node, ascii_ok: Dictionary, focus_fail: Array, allowed_line_edit: LineEdit = null) -> void:
	if node is Control:
		var ctrl := node as Control
		if ctrl is Button or ctrl is CheckButton or ctrl is OptionButton or ctrl is HSlider or ctrl is SpinBox or ctrl is TabBar or ctrl is ScrollContainer or ctrl is LineEdit:
			if ctrl != allowed_line_edit and ctrl.focus_mode != Control.FOCUS_NONE:
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
		_check_controls_focus_and_ascii(ch, ascii_ok, focus_fail, allowed_line_edit)


static func run() -> Dictionary:
	var st := {"name": "test_ter27_dil_telemetry", "passed": 0, "failed": 0, "errors": []}

	# 1. SensorRealismFilter: SENSOR_IDEAL exact pass-through vs SENSOR_TRACK_REALISM 125ms CAN delay + EKF
	var filt := SensorRealismFilterScript.new()
	filt.set_mode(SensorRealismFilterScript.SensorMode.SENSOR_IDEAL)
	var fz_in: Array[float] = [620.0, 580.0, 690.0, 640.0]
	var om_in: Array[float] = [85.0, 86.0, 84.5, 85.5]
	var ideal_out: Dictionary = filt.step(3.25, -7.80, 0.65, 17.5, -0.45, 0.12, fz_in, om_in, 0.005)
	var ideal_ok := (
		absf(float(ideal_out.get("ax", 0.0)) - 3.25) < 1e-6
		and absf(float(ideal_out.get("ay", 0.0)) - (-7.80)) < 1e-6
		and absf(float(ideal_out.get("wz", 0.0)) - 0.65) < 1e-6
		and absf(float(ideal_out.get("delta", 0.0)) - 0.12) < 1e-6
	)
	_check(ideal_ok, "SensorRealismFilter SENSOR_IDEAL is exact pass-through", st)

	var filt_a := SensorRealismFilterScript.new()
	var filt_b := SensorRealismFilterScript.new()
	filt_a.set_mode(SensorRealismFilterScript.SensorMode.SENSOR_TRACK_REALISM)
	filt_b.set_mode(SensorRealismFilterScript.SensorMode.SENSOR_TRACK_REALISM)
	filt_a.set_can_latency_ms(125.0)
	filt_b.set_can_latency_ms(125.0)
	filt_a.set_imu_noise_scale(1.0)
	filt_b.set_imu_noise_scale(1.0)
	filt_a.reset(20260427)
	filt_b.reset(20260427)

	var ax_k15 := 0.0
	var ax_k42 := 0.0
	var determ_ok := true
	var ekf_finite_ok := true
	for k in range(60):
		var ax_step := 8.0 if k >= 10 else 0.0
		var oa: Dictionary = filt_a.step(ax_step, 4.0, 0.30, 18.0, -0.35, 0.08, fz_in, om_in, 0.005)
		var ob: Dictionary = filt_b.step(ax_step, 4.0, 0.30, 18.0, -0.35, 0.08, fz_in, om_in, 0.005)
		if absf(float(oa.get("ax", 0.0)) - float(ob.get("ax", 0.0))) > 1e-6:
			determ_ok = false
		if k == 15:
			ax_k15 = float(oa.get("ax", 0.0))
		if k == 42:
			ax_k42 = float(oa.get("ax", 0.0))
		var bh := float(oa.get("beta_hat", NAN))
		var fzh: Array = oa.get("fz_hat_4", [])
		if not is_finite(bh) or absf(bh) > 0.6 or fzh.size() != 4:
			ekf_finite_ok = false
		else:
			for v_fz in fzh:
				if not is_finite(float(v_fz)) or float(v_fz) <= 0.0:
					ekf_finite_ok = false
	_check(determ_ok and ax_k15 < 2.0 and ax_k42 > 6.5 and ekf_finite_ok,
		"SensorRealismFilter TRACK_REALISM 125ms CAN lag (k15=%.2f, k42=%.2f), deterministic seed, finite EKF" % [ax_k15, ax_k42], st)

	# 2. Full14DofModel SensorRealism integration (default IDEAL, toggleable to TRACK_REALISM)
	var mdl := Full14DofModelScript.new()
	mdl.load_vehicle("ter27_4wd", "default")
	var diag0: Dictionary = mdl.get_sensor_realism_diagnostics()
	var def_ideal := (int(diag0.get("mode", -1)) == SensorRealismFilterScript.SensorMode.SENSOR_IDEAL)
	mdl.set_sensor_realism_mode(SensorRealismFilterScript.SensorMode.SENSOR_TRACK_REALISM)
	mdl.set_sensor_can_latency_ms(125.0)
	mdl.set_sensor_imu_noise_scale(1.0)
	mdl.reset_sensor_realism(77)
	mdl.chassis_state.linear_velocity = Vector3(16.0, 0.0, 0.0)
	for w in mdl.wheels:
		w.omega = 16.0 / maxf(mdl.config.wheel_radius, 0.1)
	var inp := VehicleInputsScript.new()
	inp.drive_mode = mdl.config.drive_mode
	inp.throttle = 0.35
	inp.steer_angle = deg_to_rad(4.0)
	var step_ok := true
	for _i in range(40):
		var out_m := mdl.step_standalone(inp, 0.005)
		var forces_d: Dictionary = out_m.get("forces", {}) if out_m.get("forces") is Dictionary else {}
		var s_bh := float(forces_d.get("sensor_beta_hat", out_m.get("sensor_beta_hat", 0.0)))
		if not is_finite(mdl.chassis_state.linear_velocity.x) or not is_finite(s_bh):
			step_ok = false
			break
	var diag1: Dictionary = mdl.get_sensor_realism_diagnostics()
	_check(def_ideal and int(diag1.get("mode", -1)) == 1 and step_ok,
		"Full14DofModel defaults to SENSOR_IDEAL and runs stably in SENSOR_TRACK_REALISM", st)

	# 3. RacingLineVisualizer SIM_RACING_3D progressive +Y braking ramp & relative-speed alert modulation
	var rlv := RacingLineVisualizerScript.new()
	var loaded_casadi := rlv.apply_casadi_json("res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_rubbered.json", true)
	rlv.set_display_mode(RacingLineVisualizerScript.LineDisplayMode.SIM_RACING_3D)
	rlv.set_brake_elevation_scale(0.60)
	rlv.set_speed_alert_gain(1.2)
	# Probe at a braking point (find first braking waypoint from loaded samples)
	var brk_s := 50.0
	var brk_v_ref := 18.0
	var pts: Array = rlv.get("_last_points")
	for pt in pts:
		if float(pt.get("brake", 0.0)) >= 0.25 or float(pt.get("ax", 0.0)) <= -4.0:
			brk_s = float(pt.get("s", 50.0))
			brk_v_ref = maxf(float(pt.get("v_mps", pt.get("v", 18.0))), 8.0)
			break
	rlv.update_live_driver_state(brk_s, brk_v_ref, 0.80, brk_v_ref)
	var m_ref: Dictionary = rlv.get_braking_ramp_metrics()
	rlv.update_live_driver_state(brk_s, brk_v_ref + 5.0, 0.80, brk_v_ref)
	var m_over: Dictionary = rlv.get_braking_ramp_metrics()
	rlv.set_brake_elevation_scale(0.0)
	rlv.update_live_driver_state(brk_s, brk_v_ref, 0.80, brk_v_ref)
	var m_flat: Dictionary = rlv.get_braking_ramp_metrics()
	var peak_ref := float(m_ref.get("peak_y_elevation_m", 0.0))
	var peak_over := float(m_over.get("peak_y_elevation_m", 0.0))
	var peak_flat := float(m_flat.get("peak_y_elevation_m", 1.0))
	var alert_ref := float(m_ref.get("alert_factor", 0.0))
	var alert_over := float(m_over.get("alert_factor", 0.0))
	_check(loaded_casadi and peak_ref > 0.10 and peak_over >= peak_ref and alert_over >= alert_ref and peak_flat < 0.01 and int(m_over.get("nan_count", 1)) == 0,
		"RacingLineVisualizer SIM_RACING_3D progressive +Y ramp (ref=%.3fm, over=%.3fm, flat=%.3fm, nan=%d)" % [peak_ref, peak_over, peak_flat, int(m_over.get("nan_count", -1))], st)
	rlv.queue_free()

	# 4. DilSessionManager live deltas & JSON+CSV export into outputs/dil_sessions/
	var sess := DilSessionManagerScript.new()
	var ref_ok := sess.load_casadi_reference_json("res://Assets/RacingLines/fsg_2023_racing_line_casadi_fsg_rubbered.json")
	sess.set_reference_enabled(true)
	var curve := Curve3D.new()
	for i in range(41):
		var th := TAU * float(i) / 40.0
		curve.add_point(Vector3(20.0 * (1.0 - cos(th)), 0.0, -20.0 * sin(th)))
	sess.configure_track(Transform3D.IDENTITY, curve.get_baked_length(), curve, [])
	for step_i in range(25):
		var s_cur := float(step_i) * 0.8
		var p_c := curve.sample_baked(fmod(s_cur, curve.get_baked_length()))
		sess.step_session(0.05, p_c, Basis.IDENTITY, 16.0, {
			"vx_ms": 16.0,
			"ax_mps2": -2.0,
			"ay_mps2": 6.5,
			"wz_rads": 0.45,
			"brake": 0.25,
			"throttle": 0.10,
			"max_grip_util": 0.78,
		})
	var live_t: Dictionary = sess.get_live_dil_telemetry()
	var live_keys_ok := (
		live_t.has("delta_t_s") and live_t.has("delta_vx_ms") and live_t.has("e_lat_m")
		and live_t.has("delta_wz_rads") and live_t.has("brake_prog") and live_t.has("brake_margin_m")
		and live_t.has("max_grip_util") and live_t.has("phase_deltas_s")
	)
	var exp_res: Dictionary = sess.export_dil_session("ci_test_dil")
	var jp := String(exp_res.get("json_path", ""))
	var cp := String(exp_res.get("csv_path", ""))
	var exp_ok := (
		bool(exp_res.get("ok", false))
		and not jp.is_empty() and FileAccess.file_exists(jp)
		and not cp.is_empty() and FileAccess.file_exists(cp)
	)
	if FileAccess.file_exists(jp):
		DirAccess.remove_absolute(jp)
	if FileAccess.file_exists(cp):
		DirAccess.remove_absolute(cp)
	_check(ref_ok and live_keys_ok and exp_ok,
		"DilSessionManager live CasADi deltas and JSON+CSV export into outputs/dil_sessions/ verified", st)
	sess.queue_free()

	# 5. Expanded TracksideTuningController UI controls + FOCUS_NONE + pure ASCII + readout sync
	var sim_ui := SimUIScene.instantiate()
	var main_menu := sim_ui.get_node_or_null("MainMenu")
	if main_menu != null and main_menu.has_method("_ready"):
		main_menu._ready()
	var tv_ctrl := main_menu.get("_tv_tuning_ctrl") as TracksideTuningControllerScript if main_menu != null else null
	if tv_ctrl != null and tv_ctrl.has_method("_ready"):
		tv_ctrl._ready()
	var preset_edit := tv_ctrl.get("_preset_name_edit") as LineEdit if tv_ctrl != null else null
	var ascii_ok := {"ok": true, "offender": ""}
	var focus_fail: Array = []
	if tv_ctrl != null:
		_check_controls_focus_and_ascii(tv_ctrl, ascii_ok, focus_fail, preset_edit)
	var ctrl_names := [
		"SensorRealismModeOpt",
		"SensorCanLatencySlider",
		"SensorImuNoiseSlider",
		"SensorSeedSlider",
		"ResetEkfFiltersBtn",
		"CasadiReferenceOpt",
		"RacingLine3DModeOpt",
		"Brake3DHeightSlider",
		"PreBrakeAlertGainSlider",
		"ExportDilSessionBtn",
		"DilLiveDiagLabel",
		"DilSectorDeltaLabel",
		"DilTimeGainHintLabel",
	]
	var all_found := (tv_ctrl != null)
	if tv_ctrl != null:
		for cname in ctrl_names:
			if tv_ctrl.find_child(cname, true, false) == null:
				all_found = false
				break
	_check(all_found and bool(ascii_ok["ok"]) and focus_fail.is_empty(),
		"TracksideTuningController expanded Sensors/3D-Line/DiL controls present, FOCUS_NONE (%s), pure ASCII (%s)" % [str(focus_fail), String(ascii_ok["offender"])], st)
	sim_ui.queue_free()

	return st
