extends RefCounted

const DilSessionManagerScript = preload("res://Scenes/Sim/DilSessionManager.gd")
const RacingLineVisualizerScript = preload("res://Scenes/Track/RacingLineVisualizer.gd")
const DilProceduralAudioScript = preload("res://Scenes/Vehicle/Audio/DilProceduralAudio.gd")
const WheelForcesVisualizer3DScript = preload("res://Scenes/Vehicle/Visuals/WheelForcesVisualizer3D.gd")
const RoadSurfaceScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/RoadSurface.gd")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const LtsLivePlotScript = preload("res://Scenes/Sim/UI/LtsLivePlot.gd")
const LtsMenuControllerScript = preload("res://Scenes/Sim/UI/lts_menu_controller.gd")
const TracksideTuningControllerScript = preload("res://Scenes/Sim/UI/TracksideTuningController.gd")
const RewindBufferScript = preload("res://Scenes/Sim/UI/RewindBuffer.gd")
const TelemetryCsvLoggerScript = preload("res://Scenes/Sim/UI/TelemetryCsvLogger.gd")
const SimUIScene = preload("res://Scenes/Sim/UI/SimUI.tscn")
const SimScene = preload("res://Scenes/Sim/Sim.tscn")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] += 1
	else:
		st["failed"] += 1
		st["errors"].append(msg)


static func _check_focus_and_ascii(node: Node, ascii_ok: Dictionary, focus_fail: Array, allowed_line_edit: LineEdit = null) -> void:
	if node is Control:
		var ctrl := node as Control
		if ctrl is Button or ctrl is CheckButton or ctrl is OptionButton or ctrl is HSlider or ctrl is SpinBox or ctrl is TabBar or ctrl is ScrollContainer or ctrl is LineEdit:
			if ctrl == allowed_line_edit:
				pass
			elif ctrl.focus_mode != Control.FOCUS_NONE:
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
		_check_focus_and_ascii(ch, ascii_ok, focus_fail, allowed_line_edit)


static func run() -> Dictionary:
	var st := {"name": "test_dil_features", "passed": 0, "failed": 0, "errors": []}

	# 1. DilSessionManager: 3 sectors + lap cross + DOO +2.0s + ghost traces (best & LTS optimal)
	var sess := DilSessionManagerScript.new()
	var curve := Curve3D.new()
	var r_loop := 18.0
	var n_pts := 64
	for i in range(n_pts + 1):
		var th := TAU * float(i) / float(n_pts)
		curve.add_point(Vector3(r_loop * (1.0 - cos(th)), 0.0, -r_loop * sin(th)))
	var fake_cone := Node3D.new()
	fake_cone.position = Vector3(5.0, 0.0, -10.0)
	var lap_len := curve.get_baked_length()
	sess.configure_track(Transform3D.IDENTITY, lap_len, curve, [{"pos": fake_cone.position, "node": fake_cone}])
	var lts_loaded := sess.load_lts_trace_from_json("res://Assets/RacingLines/fsg_2023_racing_line.json")
	_check(lts_loaded, "DilSessionManager loads fsg_2023_racing_line.json", st)
	sess.set_ghost_mode(2)
	var dt := 0.05
	var v_loop := 12.0
	# Step from -1.2 m to (lap_len + 1.8 m) so we cross line_signed=0 at lap start AND lap finish
	var s_cursor := -1.2
	while s_cursor <= lap_len + 1.8:
		var s_wrapped := fmod(s_cursor + lap_len, lap_len)
		var p_c := curve.sample_baked(s_wrapped)
		var p_next := curve.sample_baked(fmod(s_wrapped + 0.5, lap_len))
		var fwd := (p_next - p_c).normalized()
		var yaw := atan2(-fwd.x, -fwd.z)
		var b := Basis(Vector3.UP, yaw)
		sess.step_session(dt, p_c, b, v_loop)
		s_cursor += v_loop * dt
	var sum_1 := sess.get_summary()
	_check(float(sum_1["last_lap_s"]) > 5.0 and float(sum_1["best_lap_s"]) > 5.0 and bool(sum_1["has_best_ghost"]) and bool(sum_1["has_lts_ghost"]),
		"DilSessionManager completes lap with valid sector/lap timing and best/LTS ghosts (last=%.2fs)" % float(sum_1["last_lap_s"]), st)
	# Test Ghost Car orientation alignment with velocity vector along curve (dot > 0.95)
	sess._ensure_ghost_node()
	var ghost_root := sess.get("_ghost_root") as Node3D
	var ghost_mesh := sess.get("_ghost_mesh") as MeshInstance3D
	var ghost_dot_ok := false
	if ghost_root != null and ghost_mesh != null:
		sess.set_ghost_mode(2)
		var p_g0 := curve.sample_baked(5.0)
		var p_g1 := curve.sample_baked(5.5)
		var vel_dir := (p_g1 - p_g0).normalized()
		var yaw_g := atan2(-vel_dir.x, -vel_dir.z)
		ghost_root.position = p_g0
		ghost_root.transform.basis = Basis.from_euler(Vector3(0.0, yaw_g, 0.0), EulerOrder.EULER_ORDER_YXZ)
		var root_fwd := -ghost_root.transform.basis.z.normalized()
		var mesh_fwd := (ghost_root.transform.basis * ghost_mesh.transform.basis).z.normalized()
		ghost_dot_ok = (root_fwd.dot(vel_dir) > 0.95 and mesh_fwd.dot(vel_dir) > 0.95 and absf(ghost_mesh.rotation.y - PI) < 1e-4)
	_check(ghost_dot_ok, "GhostCar forward aligns with velocity vector along curve (dot > 0.95, mesh yaw=PI)", st)
	# Test DOO penalty trigger and debounce
	sess.step_session(0.05, fake_cone.position, Basis.IDENTITY, 10.0)
	sess.step_session(0.05, fake_cone.position + Vector3(0.05, 0.0, 0.0), Basis.IDENTITY, 10.0)
	var sum_doo := sess.get_summary()
	_check(int(sum_doo["cones_hit"]) == 1 and absf(float(sum_doo["penalty_s"]) - 2.0) < 1e-4,
		"DOO cone hit registers +2.0s once with debounce (hits=%d pen=%.1f)" % [int(sum_doo["cones_hit"]), float(sum_doo["penalty_s"])], st)
	sess.queue_free()
	fake_cone.queue_free()

	# 2. RacingLineVisualizer: loads FSG 2023 line, classifies continuous gradient, builds ribbon + border + markers without NaN
	var rlv := RacingLineVisualizerScript.new()
	var samples := rlv.load_racing_line_json("res://Assets/RacingLines/fsg_2023_racing_line.json")
	var ribbon_mesh := RacingLineVisualizerScript.build_ribbon_array_mesh(samples, true)
	var border_mesh := RacingLineVisualizerScript.build_border_array_mesh(samples, true)
	var markers_mesh := RacingLineVisualizerScript.build_markers_array_mesh(samples, true)
	var c_gas := RacingLineVisualizerScript.classify_zone_color(0.95, 0.0, 5.0)
	var c_coast := RacingLineVisualizerScript.classify_zone_color(0.20, 0.0, -0.2)
	var c_micropulse := RacingLineVisualizerScript.classify_zone_color(0.10, 0.10, -2.0)
	var c_trail := RacingLineVisualizerScript.classify_zone_color(0.0, 0.18, -3.5)
	var c_brk := RacingLineVisualizerScript.classify_zone_color(0.0, 0.70, -10.0)
	var no_nan_verts := true
	for m_chk in [ribbon_mesh, border_mesh, markers_mesh]:
		if m_chk != null and m_chk is ArrayMesh:
			var am := m_chk as ArrayMesh
			if am.get_surface_count() > 0:
				var arrs: Array = am.surface_get_arrays(0)
				var vs: PackedVector3Array = arrs[Mesh.ARRAY_VERTEX]
				for v3 in vs:
					if not is_finite(v3.x) or not is_finite(v3.y) or not is_finite(v3.z):
						no_nan_verts = false
						break
	_check(samples.size() >= 200 and ribbon_mesh != null and ribbon_mesh.get_surface_count() == 1 and border_mesh != null and border_mesh.get_surface_count() == 1 and markers_mesh != null and markers_mesh.get_surface_count() == 1 and no_nan_verts and c_gas.g > c_gas.r and c_coast.b > c_coast.r and c_coast.g > 0.6 and c_micropulse == c_coast and c_trail.r > 0.9 and c_trail.g > 0.4 and c_trail.b < 0.2 and c_brk.r > 0.9 and c_brk.g < 0.2,
		"RacingLineVisualizer loads %d samples, 4-state palette (no micro-pulse red), ribbon+border+markers finite ArrayMeshes" % samples.size(), st)

	# 2b. RacingLineVisualizer: clean loading of fsg_2023_racing_line_real.json (360 pts), no false brake micro-pulses, finite meshes
	var samples_real := rlv.load_racing_line_json("res://Assets/RacingLines/fsg_2023_racing_line_real.json")
	var ribbon_real := RacingLineVisualizerScript.build_ribbon_array_mesh(samples_real, true)
	var border_real := RacingLineVisualizerScript.build_border_array_mesh(samples_real, true)
	var markers_real := RacingLineVisualizerScript.build_markers_array_mesh(samples_real, true)
	var no_nan_real := true
	var no_false_brake_real := true
	for sr in samples_real:
		var b_val := float(sr.get("brake", 0.0))
		var ax_val := float(sr.get("ax", 0.0))
		if b_val > 0.0 and b_val < 0.14 and ax_val > -3.5:
			no_false_brake_real = false
			break
	for m_chk_r in [ribbon_real, border_real, markers_real]:
		if m_chk_r != null and m_chk_r is ArrayMesh:
			var am_r := m_chk_r as ArrayMesh
			if am_r.get_surface_count() > 0:
				var arrs_r: Array = am_r.surface_get_arrays(0)
				var vs_r: PackedVector3Array = arrs_r[Mesh.ARRAY_VERTEX]
				for v3_r in vs_r:
					if not is_finite(v3_r.x) or not is_finite(v3_r.y) or not is_finite(v3_r.z):
						no_nan_real = false
						break
	_check(samples_real.size() == 360 and ribbon_real != null and ribbon_real.get_surface_count() == 1 and border_real != null and border_real.get_surface_count() == 1 and markers_real != null and markers_real.get_surface_count() == 1 and no_nan_real and no_false_brake_real,
		"RacingLineVisualizer cleanly loads fsg_2023_racing_line_real.json (%d samples, no false brake pulses) with finite ArrayMeshes" % samples_real.size(), st)

	# 2c. RacingLineVisualizer: explicit grown custom_aabb and anti-culling settings across full FSG 2023 circuit
	rlv.update_from_samples(samples, true)
	rlv.update_bump_markers([{
		"x_ros": 200.0,
		"y_ros": 95.0,
		"tx": 1.0,
		"ty": 0.0,
		"lat_off": 0.0,
		"half_w": 1.5,
		"len_m": 0.8,
		"amp_m": 0.03,
	}])
	var furthest_pt := Vector3.ZERO
	var max_dist_sq := -1.0
	for s_pt in samples:
		var sx := float(s_pt.get("x_godot", 0.0))
		var sz := float(s_pt.get("z_godot", 0.0))
		var d2 := sx * sx + sz * sz
		if d2 > max_dist_sq:
			max_dist_sq = d2
			furthest_pt = Vector3(sx, 0.045, sz)
	var mid_sample: Dictionary = samples[samples.size() / 2]
	var mid_pt := Vector3(float(mid_sample.get("x_godot", 0.0)), 0.045, float(mid_sample.get("z_godot", 0.0)))
	var bump_pt := Vector3(-95.0, 0.032, -200.0)
	var aabb_cull_ok := (max_dist_sq > 150.0 * 150.0)
	for line_mesh_name in ["RacingLineBorderMesh", "RacingLineMesh", "RacingLineMarkersMesh"]:
		var mi := rlv.get_node_or_null(line_mesh_name) as MeshInstance3D
		if mi == null or mi.extra_cull_margin < 50.0 or not mi.ignore_occlusion_culling:
			aabb_cull_ok = false
			break
		var am := mi.mesh as ArrayMesh
		if am == null or not am.custom_aabb.has_volume():
			aabb_cull_ok = false
			break
		if not am.custom_aabb.has_point(furthest_pt) or not am.custom_aabb.has_point(mid_pt):
			aabb_cull_ok = false
			break
	var bump_mi := rlv.get_node_or_null("RoadBumpsMesh") as MeshInstance3D
	if bump_mi == null or bump_mi.extra_cull_margin < 50.0 or not bump_mi.ignore_occlusion_culling:
		aabb_cull_ok = false
	else:
		var bump_am := bump_mi.mesh as ArrayMesh
		if bump_am == null or not bump_am.custom_aabb.has_volume() or not bump_am.custom_aabb.has_point(bump_pt):
			aabb_cull_ok = false
	_check(aabb_cull_ok,
		"RacingLineVisualizer custom_aabb contains FSG 2023 far end (furthest=%s, s_mid=%s) and MeshInstance3D anti-culling active" % [str(furthest_pt), str(mid_pt)], st)
	rlv.queue_free()

	# 3. DilProceduralAudio: parameter extraction in drive vs regen vs oversteer vs understeer
	var p_drv := DilProceduralAudioScript.compute_synth_params(120.0, 42.0, 0.5, 0.5, 0.02)
	var p_reg := DilProceduralAudioScript.compute_synth_params(110.0, -18.0, 0.85, 0.65, 0.01)
	var p_ov := DilProceduralAudioScript.compute_synth_params(95.0, 20.0, 0.6, 0.96, deg_to_rad(6.0))
	var p_un := DilProceduralAudioScript.compute_synth_params(95.0, 20.0, 0.96, 0.6, deg_to_rad(0.5))
	_check(bool(p_reg["is_regen"]) and not bool(p_drv["is_regen"]) and float(p_ov["rear_scrub_gain"]) > float(p_ov["front_scrub_gain"]) and float(p_un["front_scrub_gain"]) > float(p_un["rear_scrub_gain"]),
		"DilProceduralAudio differentiates drive/regen and front/rear tire scrub gains", st)

	# 3b. DilProceduralAudio: repeated enable/disable lifecycle + extreme DSP inputs (vx=0, vx=40, P=+/-80kW, sat=1.5, NaN/INF)
	var aud := DilProceduralAudioScript.new()
	var aud_samples_ok := true
	aud.set_enabled(false)
	aud.set_volume(0.85)
	aud.step_audio(0.0, 0.0, 0.0, 0.0, 0.0)
	aud.set_enabled(true)
	var extreme_cases := [
		[0.0, 0.0, 0.0, 0.0, 0.0],
		[156.0, 80.0, 1.5, 1.5, 0.5],
		[156.0, -80.0, 1.5, 1.5, -0.5],
		[NAN, INF, NAN, -INF, NAN],
	]
	for ec in extreme_cases:
		aud.step_audio(float(ec[0]), float(ec[1]), float(ec[2]), float(ec[3]), float(ec[4]))
		var ep := DilProceduralAudioScript.compute_synth_params(float(ec[0]), float(ec[1]), float(ec[2]), float(ec[3]), float(ec[4]))
		for _f in range(256):
			var smp := aud.synthesize_next_sample(
				float(ep["motor_freq_hz"]),
				float(ep["inv_freq_hz"]),
				float(ep["motor_gain"]) * aud.volume,
				float(ep["inv_gain"]) * aud.volume,
				float(ep["front_scrub_gain"]) * aud.volume,
				float(ep["rear_scrub_gain"]) * aud.volume,
				22050.0
			)
			if not is_finite(smp) or absf(smp) > 0.9501:
				aud_samples_ok = false
				break
	aud.set_enabled(false)
	aud.set_enabled(true)
	aud.step_audio(120.0, 60.0, 1.2, 1.1, 0.08)
	for _f2 in range(128):
		var smp2 := aud.synthesize_next_sample(aud.last_motor_freq_hz, aud.last_inv_freq_hz, aud.last_motor_gain, aud.last_inv_gain, aud.last_front_scrub_gain, aud.last_rear_scrub_gain, 8000.0)
		if not is_finite(smp2) or absf(smp2) > 0.9501:
			aud_samples_ok = false
			break
	aud.set_enabled(false)
	aud.queue_free()
	_check(aud_samples_ok, "DilProceduralAudio toggle cycle (false->true->false->true) and extreme DSP produce finite bounded samples", st)

	# 4. WheelForcesVisualizer3D: Fz arrow scaling & stacked drive/regen/mech corner bars
	var h_fz_zero := WheelForcesVisualizer3DScript.compute_fz_arrow_height(0.0)
	var h_fz_2200 := WheelForcesVisualizer3DScript.compute_fz_arrow_height(2200.0)
	var lay_drv := WheelForcesVisualizer3DScript.compute_corner_bar_layout(250.0, 0.0, 0.0, 312.5, 0.42)
	var lay_mix := WheelForcesVisualizer3DScript.compute_corner_bar_layout(125.0, 93.75, 187.5, 312.5, 0.42)
	_check(absf(h_fz_zero) < 1e-6 and absf(h_fz_2200 - 1.10) < 1e-4 and float(lay_drv["drive_h"]) > 0.30 and float(lay_drv["drive_cy"]) > 0.0 and float(lay_mix["regen_h"]) > 0.10 and float(lay_mix["mech_h"]) > 0.15 and float(lay_mix["mech_cy"]) < float(lay_mix["regen_cy"]),
		"WheelForcesVisualizer3D Fz and stacked drive/regen/mech bar layout verified", st)

	# 5. RoadSurface: FLAT exact 0, ISO_A < ISO_B < ISO_C RMS, C^1 35mm bump, Full14DofModel stability + Fz peak
	var rs := RoadSurfaceScript.new()
	var rms_by_mode := {}
	for mode_id in [RoadSurfaceScript.SurfaceMode.FLAT, RoadSurfaceScript.SurfaceMode.ISO_A, RoadSurfaceScript.SurfaceMode.ISO_B, RoadSurfaceScript.SurfaceMode.ISO_C]:
		rs.configure(mode_id, 1.0, RoadSurfaceScript.BumpMode.OFF, 0.0, 0.8, 35.0)
		var sum_sq := 0.0
		var n_samp := 400
		for si in range(n_samp):
			var xs := float(si) * 0.25
			var s_v := rs.sample_world(xs, 1.3, 18.0, 0.0)
			sum_sq += s_v.x * s_v.x
		rms_by_mode[mode_id] = sqrt(sum_sq / float(n_samp))
	var rms_flat: float = float(rms_by_mode[RoadSurfaceScript.SurfaceMode.FLAT])
	var rms_a: float = float(rms_by_mode[RoadSurfaceScript.SurfaceMode.ISO_A])
	var rms_b: float = float(rms_by_mode[RoadSurfaceScript.SurfaceMode.ISO_B])
	var rms_c: float = float(rms_by_mode[RoadSurfaceScript.SurfaceMode.ISO_C])
	rs.configure(RoadSurfaceScript.SurfaceMode.FLAT, 1.0, RoadSurfaceScript.BumpMode.SPEED_BUMPS, 35.0, 0.80, 35.0)
	var b0 := rs.sample_world(14.2, 0.0, 15.0, 0.0)
	var b_peak := rs.sample_world(15.0, 0.0, 15.0, 0.0)
	var b_end := rs.sample_world(15.8, 0.0, 15.0, 0.0)
	_check(rms_flat == 0.0 and rms_a > 0.0005 and rms_a < rms_b and rms_b < rms_c and absf(b0.x) < 1e-4 and absf(b_peak.x - 0.035) < 1e-4 and absf(b_end.x) < 1e-4,
		"RoadSurface RMS ordering (0 < %.4f < %.4f < %.4f m) and 35mm C^1 bump peak=%.4f m" % [rms_a, rms_b, rms_c, b_peak.x], st)

	# Quantitative 14-DOF Full14DofModel test over 35mm bump + ISO_C 40mm stability
	var m_bump := Full14DofScript.new()
	m_bump.load_vehicle("ter27_4wd", "default")
	m_bump.road_surface.configure(RoadSurfaceScript.SurfaceMode.FLAT, 1.0, RoadSurfaceScript.BumpMode.SPEED_BUMPS, 35.0, 0.80, 35.0)
	m_bump.chassis_state.position = Vector3(14.0, 0.0, m_bump.config.h_cog)
	m_bump.chassis_state.linear_velocity = Vector3(15.0, 0.0, 0.0)
	for w in m_bump.wheels:
		w.omega = 15.0 / maxf(m_bump.config.wheel_radius, 0.1)
	var inp_b := VehicleInputsScript.new()
	inp_b.drive_mode = m_bump.config.drive_mode
	inp_b.throttle = 0.05
	var fz_peak_fl := 0.0
	var fz_peak_rl := 0.0
	var bump_stable := true
	for _k in range(180):
		var out_b := m_bump.step_standalone(inp_b, 0.0025)
		var n_arr: Array = out_b["normal_forces"]
		fz_peak_fl = maxf(fz_peak_fl, float(n_arr[0]))
		fz_peak_rl = maxf(fz_peak_rl, float(n_arr[2]))
		if is_nan(m_bump.chassis_state.position.z) or absf(m_bump.chassis_state.roll) > 0.5:
			bump_stable = false
			break
	m_bump.road_surface.configure(RoadSurfaceScript.SurfaceMode.ISO_C, 1.0, RoadSurfaceScript.BumpMode.ASYM_POTHOLES, 40.0, 0.80, 25.0)
	for _k in range(200):
		m_bump.step_standalone(inp_b, 0.0025)
		if is_nan(m_bump.chassis_state.position.z) or absf(m_bump.chassis_state.position.z) > 2.0 or absf(m_bump.chassis_state.roll) > 0.5:
			bump_stable = false
			break
	_check(bump_stable and fz_peak_fl > 800.0 and fz_peak_rl > 800.0,
		"Full14DofModel stable on 35mm bump (FL peak=%.1fN, RL peak=%.1fN) + ISO_C 40mm potholes" % [fz_peak_fl, fz_peak_rl], st)

	# 6. Physical steering_torque_nm profile (straight / ~0.5g / ~1.2g / saturated understeer)
	var m_str := Full14DofScript.new()
	m_str.load_vehicle("ter27_4wd", "default")
	var trq_vals := {}
	for tag in [["straight", 0.0], ["mid_05g", 2.0], ["high_12g", 5.5], ["sat", 16.0]]:
		m_str.reset()
		m_str.chassis_state.linear_velocity = Vector3(18.0, 0.0, 0.0)
		for w in m_str.wheels:
			w.omega = 18.0 / maxf(m_str.config.wheel_radius, 0.1)
		var inp_s := VehicleInputsScript.new()
		inp_s.drive_mode = m_str.config.drive_mode
		inp_s.throttle = 0.20
		inp_s.steer_angle = deg_to_rad(float(tag[1]))
		var out_s: Dictionary = {}
		for _k in range(80):
			out_s = m_str.step_standalone(inp_s, 0.0025)
		var ay_g := absf(float(out_s["net_force_ros"].y) / maxf(m_str.config.mass, 1.0)) / 9.81
		trq_vals[String(tag[0])] = {"trq": m_str.steering_torque_nm, "ay": ay_g}
	var t_straight: float = float(trq_vals["straight"]["trq"])
	var t_05g: float = float(trq_vals["mid_05g"]["trq"])
	var t_12g: float = float(trq_vals["high_12g"]["trq"])
	var t_sat: float = float(trq_vals["sat"]["trq"])
	_check(absf(t_straight) < 0.05 and absf(t_05g) > 1.0 and absf(t_12g) > absf(t_05g) and t_05g < 0.0 and t_12g < 0.0,
		"Physical steering_torque_nm straight=%.3f 05g=%.3f 12g=%.3f sat=%.3f Nm" % [t_straight, t_05g, t_12g, t_sat], st)

	# 7. LtsLivePlot: JSONL progress ingestion matching optimize_tv_maps.py schema
	var plot := LtsLivePlotScript.new()
	plot.reset_plot(LtsLivePlotScript.PlotMode.CONVERGENCE)
	var l1 := "{\"stage\":\"autocross\",\"trial\":1,\"total_trials\":4,\"feasible\":true,\"is_best\":true,\"lap_s\":74.33,\"e_22km_kwh\":6.42,\"j_fsg\":35470.0,\"params\":{\"k_ff_low_mult\":1.35}}"
	var l2 := "{\"stage\":\"autocross\",\"trial\":2,\"total_trials\":4,\"feasible\":false,\"is_best\":false,\"lap_s\":74.90}"
	var l3 := "{\"stage\":\"autocross\",\"trial\":3,\"total_trials\":4,\"feasible\":true,\"is_best\":true,\"lap_s\":73.95,\"e_22km_kwh\":6.31,\"j_fsg\":34510.0,\"params\":{\"k_ff_low_mult\":1.42}}"
	var l4 := "{\"done\":true,\"summary_path\":\"/tmp/sum.json\",\"best_trace_csv\":\"/tmp/trace.csv\"}"
	for ln in [l1, l2, l3, l4]:
		plot.ingest_jsonl_line(ln)
	_check(plot.is_done and absf(plot.best_lap_s - 73.95) < 1e-4 and plot.feasible_count == 2 and plot.rejected_count == 1 and plot.best_trace_csv == "/tmp/trace.csv",
		"LtsLivePlot JSONL parsing updates best_lap=%.2fs, counts=%d/%d, done=%s" % [plot.best_lap_s, plot.feasible_count, plot.rejected_count, str(plot.is_done)], st)
	plot.queue_free()

	# 7b. LtsMenuController: repo_root, script paths & python3 interpreter resolution
	var lts_root := LtsMenuControllerScript.resolve_repo_root()
	var lts_py := LtsMenuControllerScript.resolve_python_executable(lts_root)
	var bench_py := lts_root.path_join("vd_correlation_lab/optimization/lap_time_sim/run_fsg_2023_benchmark.py")
	var opt_py := lts_root.path_join("vd_correlation_lab/control/torque_vectoring/optimize_tv_maps.py")
	_check(not lts_py.is_empty() and FileAccess.file_exists(bench_py) and FileAccess.file_exists(opt_py),
		"LtsMenuController resolves python (%s) and LTS scripts in %s" % [lts_py, lts_root], st)

	# 8. SimUI 9 tabs + FOCUS_NONE (except preset LineEdit) + zero non-ASCII/emojis + DilTimingBar + F8 + Audio 3 profiles + Setup/TV + Presets + Diff + STM32 + CSV (50/100/200 Hz)
	var sim_ui := SimUIScene.instantiate()
	var main_menu := sim_ui.get_node_or_null("MainMenu")
	var tabs := sim_ui.get_node_or_null("MainMenu/MarginContainer/VBoxContainer/MenuTabs") as TabContainer
	if tabs != null:
		for ti in range(tabs.get_tab_count()):
			var tc := tabs.get_tab_control(ti)
			if tc != null and tc.has_method("_ready"):
				tc._ready()
	if main_menu != null and main_menu.has_method("_ready"):
		main_menu._ready()
	var tv_ctrl := main_menu.get("_tv_tuning_ctrl") as TracksideTuningControllerScript if main_menu != null else null
	if tv_ctrl != null and tv_ctrl.has_method("_ready"):
		tv_ctrl._ready()
	var preset_edit := tv_ctrl.get("_preset_name_edit") as LineEdit if tv_ctrl != null else null
	var tab_names: Array[String] = []
	if tabs != null:
		for ti in range(tabs.get_tab_count()):
			tab_names.append(tabs.get_tab_title(ti))
	if main_menu != null and main_menu.has_method("_release_any_focus"):
		main_menu._release_any_focus()
	var dil_bar := sim_ui.get_node_or_null("DilTimingBar")
	var ascii_ok := {"ok": true, "offender": ""}
	var focus_fail: Array = []
	_check_focus_and_ascii(sim_ui, ascii_ok, focus_fail, preset_edit)
	var exp_tabs := ["Vehicle", "Setup / TV", "Track", "View", "Audio", "Controls", "Telemetry", "AS / Sys", "LTS"]
	var no_extra_hud := true
	for ui_ch in sim_ui.get_children():
		if ui_ch.name != "MainMenu" and ui_ch.name != "DilTimingBar" and ui_ch.name != "Stats" and ui_ch.name != "ControlsLabel" and ui_ch is Control and (ui_ch as Control).visible:
			no_extra_hud = false
	_check(dil_bar != null and no_extra_hud and tab_names == exp_tabs and bool(ascii_ok["ok"]) and focus_fail.is_empty(),
		"Test 6 UI Integrity: 9 tabs (%s), DilTimingBar only HUD (%s), zero emojis (%s), FOCUS_NONE (%s)" % [str(tab_names), str(no_extra_hud), str(ascii_ok["offender"]), str(focus_fail)], st)

	# 8b. F8 untouched + Audio 3 profiles + Setup/TV live tuning + Diff Viewer + Named Presets + STM32 export
	var chk_arrows := main_menu.get("_chk_arrows") as CheckButton if main_menu != null else null
	var aud_prof_opt := main_menu.get("_audio_profile_opt") as OptionButton if main_menu != null else null
	var f8_ok := (chk_arrows != null and chk_arrows.text.contains("[F8]"))
	var aud_3_ok := (aud_prof_opt != null and aud_prof_opt.item_count == 3 and aud_prof_opt.get_item_text(0) == "FS Electric" and aud_prof_opt.get_item_text(1) == "F1 V10" and aud_prof_opt.get_item_text(2) == "F1 V12")
	var tv_map_opt := tv_ctrl.get("_map_opt") as OptionButton if tv_ctrl != null else null
	var tv_sliders := tv_ctrl.get("_sliders") as Dictionary if tv_ctrl != null else {}
	var tv_ok := (tv_ctrl != null and tv_map_opt != null and tv_map_opt.item_count == 5 and tv_sliders.size() >= 10)
	var fem_slider := tv_sliders.get("friction_ellipse_margin") as HSlider if tv_ctrl != null else null
	var yff_slider := tv_sliders.get("yaw_accel_ff_gain") as HSlider if tv_ctrl != null else null
	var diag_lbl := tv_ctrl.get("_diag_lbl") as Label if tv_ctrl != null else null
	var new_tv_ui_ok := (
		fem_slider != null and fem_slider.focus_mode == Control.FOCUS_NONE
		and yff_slider != null and yff_slider.focus_mode == Control.FOCUS_NONE
		and diag_lbl != null and diag_lbl.text.contains("Launch State:")
	)
	_check(f8_ok and aud_3_ok and tv_ok and new_tv_ui_ok,
		"Test 6 F8 label (%s), Audio 3 profiles (%s), Setup/TV 5-map live tuning (%s), friction_ellipse_margin/yaw_accel_ff_gain/Launch State (%s)" % [str(f8_ok), str(aud_3_ok), str(tv_ok), str(new_tv_ui_ok)], st)

	# Preset LineEdit releases focus on Enter & Escape (when in SceneTree)
	var line_edit_focus_ok := false
	if preset_edit != null and tv_ctrl != null:
		var esc_ev := InputEventKey.new()
		esc_ev.pressed = true
		esc_ev.keycode = KEY_ESCAPE
		tv_ctrl._on_preset_name_gui_input(esc_ev)
		preset_edit.text_submitted.emit("tmp_check")
		line_edit_focus_ok = (preset_edit.focus_mode == Control.FOCUS_CLICK and not preset_edit.has_focus())
	_check(line_edit_focus_ok, "Test 6 Preset LineEdit configured FOCUS_CLICK and releases focus on Enter/Escape", st)

	# Diff Viewer + Named Presets (save / load / delete)
	var diff_ok := false
	var preset_ok := false
	if tv_ctrl != null:
		var diff_lbl := tv_ctrl.get("_diff_lbl") as Label
		var hp_slider := tv_sliders.get("hairpin_pivot_gain") as HSlider
		if diff_lbl != null and hp_slider != null:
			hp_slider.value = 1.65
			tv_ctrl._on_param_slider_changed("hairpin_pivot_gain", 1.65)
			diff_ok = diff_lbl.text.contains("Hairpin Pivot") and diff_lbl.text.contains("1.65")
		if fem_slider != null and yff_slider != null and diff_lbl != null:
			var val_labels := tv_ctrl.get("_val_labels") as Dictionary
			fem_slider.value = 0.85
			tv_ctrl._on_param_slider_changed("friction_ellipse_margin", 0.85)
			yff_slider.value = 1.35
			tv_ctrl._on_param_slider_changed("yaw_accel_ff_gain", 1.35)
			var fem_lbl := val_labels.get("friction_ellipse_margin") as Label
			var yff_lbl := val_labels.get("yaw_accel_ff_gain") as Label
			var new_diff_ok := (
				fem_lbl != null and fem_lbl.text == "85%*"
				and yff_lbl != null and yff_lbl.text == "1.35x*"
				and diff_lbl.text.contains("Friction Ellipse Margin")
				and diff_lbl.text.contains("Yaw Accel FF Gain")
			)
			diff_ok = diff_ok and new_diff_ok
			fem_slider.value = 0.94
			tv_ctrl._on_param_slider_changed("friction_ellipse_margin", 0.94)
			yff_slider.value = 1.00
			tv_ctrl._on_param_slider_changed("yaw_accel_ff_gain", 1.00)
		if preset_edit != null:
			preset_edit.text = "test_ci_preset"
			tv_ctrl._on_save_preset()
			var p_opt := tv_ctrl.get("_preset_opt") as OptionButton
			var found_idx := -1
			if p_opt != null:
				for pi in range(p_opt.item_count):
					if p_opt.get_item_text(pi) == "test_ci_preset":
						found_idx = pi
						break
			if found_idx >= 0:
				p_opt.select(found_idx)
				if hp_slider != null:
					hp_slider.value = 1.00
				tv_ctrl._on_param_slider_changed("hairpin_pivot_gain", 1.00)
				tv_ctrl._on_load_preset()
				var loaded_val := float((tv_ctrl.get("_current") as Dictionary).get("hairpin_pivot_gain", 0.0))
				tv_ctrl._on_delete_preset()
				var still_there := false
				for pi2 in range(p_opt.item_count):
					if p_opt.get_item_text(pi2) == "test_ci_preset":
						still_there = true
				preset_ok = (absf(loaded_val - 1.65) < 0.05 and not still_there)
	_check(diff_ok, "Test 6 Diff Viewer displays modified TV parameter vs base map", st)
	_check(preset_ok, "Test 6 Named Presets save, load, and delete round-trip cleanly", st)

	# Audio 3 profiles DSP synthesis check
	var aud3 := DilProceduralAudioScript.new()
	var aud3_ok := true
	for pid in [DilProceduralAudioScript.PROFILE_FS_ELECTRIC, DilProceduralAudioScript.PROFILE_F1_V10, DilProceduralAudioScript.PROFILE_F1_V12]:
		aud3.set_sound_profile(pid)
		if aud3.get_sound_profile() != pid:
			aud3_ok = false
		aud3.step_audio(85.0, 45.0, 0.4, 0.5, 0.02)
		if not is_finite(aud3.last_motor_freq_hz):
			aud3_ok = false
	aud3.queue_free()
	_check(aud3_ok, "DilProceduralAudio supports all 3 profiles (FS Electric, F1 V10, F1 V12) cleanly", st)

	# STM32 C header export check
	var stm_ok := false
	if tv_ctrl != null:
		tv_ctrl._on_export_stm32()
		var abs_stm := ProjectSettings.globalize_path(TracksideTuningControllerScript.STM32_EXPORT_PATH)
		if FileAccess.file_exists(TracksideTuningControllerScript.STM32_EXPORT_PATH):
			var hdr_txt := FileAccess.get_file_as_string(TracksideTuningControllerScript.STM32_EXPORT_PATH)
			stm_ok = (
				hdr_txt.contains("#ifndef TV_EXPORTED_PARAMS_H")
				and hdr_txt.contains("#define TV_PARAM_POWER_LIMIT_KW")
				and hdr_txt.contains("#define TV_PARAM_FRICTION_ELLIPSE_MARGIN")
				and hdr_txt.contains("#define TV_PARAM_YAW_ACCEL_FF_GAIN")
				and hdr_txt.contains("#define TV_PARAM_CENTRIPETAL_FX_GAIN")
				and hdr_txt.contains("#define TV_PARAM_DELTA_ALPHA_EQ_GAIN")
			)
			DirAccess.remove_absolute(abs_stm)
	_check(stm_ok, "Test 6 Setup/TV STM32 C header export writes valid TV_EXPORTED_PARAMS_H", st)

	# TelemetryCsvLogger 50 / 100 / 200 Hz start/log/stop check
	var csv_ok := true
	var csv_counts := {}
	for hz in [50, 100, 200]:
		var csv_log := TelemetryCsvLoggerScript.new()
		var applied_hz := csv_log.set_sample_rate_hz(hz)
		var csv_path := csv_log.start_recording("test_track", "Autocross")
		for si in range(20):
			csv_log.log_step(0.0025, {"kmh": 72.0, "p_dc_kw": 40.0, "t_cmd": [50.0, 50.0, 70.0, 70.0]}, null, 1, 12.5, "Autocross")
		csv_log.stop_recording()
		csv_counts[hz] = csv_log.sample_count
		var ok_hz: bool = (applied_hz == hz and not csv_path.is_empty() and csv_log.sample_count >= 1 and FileAccess.file_exists(csv_path))
		if ok_hz:
			var csv_txt := FileAccess.get_file_as_string(csv_path)
			ok_hz = csv_txt.contains("t,lap,s_m,vx,vy,wz,ax,ay,beta_deg") and csv_txt.contains("Autocross")
			DirAccess.remove_absolute(ProjectSettings.globalize_path(csv_path))
		if not ok_hz:
			csv_ok = false
	_check(csv_ok and int(csv_counts[200]) > int(csv_counts[100]) and int(csv_counts[100]) >= int(csv_counts[50]),
		"Test 6 TelemetryCsvLogger records valid CSV at 50/100/200 Hz (counts=%s)" % str(csv_counts), st)

	sim_ui.queue_free()

	# 9. Full14DofModel TrackGripProfile (TTC_LAB_PEAK vs FSG_REAL_TRACK)
	var m_gp := Full14DofScript.new()
	m_gp.load_vehicle("ter27_4wd", "default")
	var gp_def_ok := (m_gp.track_grip_profile == Full14DofScript.TrackGripProfile.TTC_LAB_PEAK and absf(m_gp.env_mu_scale - 1.0) < 1e-6 and absf(m_gp.env_aero_cl_scale - 1.0) < 1e-6)
	m_gp.chassis_state.linear_velocity = Vector3(22.0, 0.0, 0.0)
	for w in m_gp.wheels:
		w.omega = 22.0 / maxf(m_gp.config.wheel_radius, 0.1)
	var inp_gp := VehicleInputsScript.new()
	inp_gp.drive_mode = m_gp.config.drive_mode
	inp_gp.throttle = 0.25
	inp_gp.steer_angle = deg_to_rad(5.0)
	var out_lab := {}
	for _k in range(70):
		out_lab = m_gp.step_standalone(inp_gp, 0.0025)
	var fz_aero_lab := absf(float(out_lab["forces"]["aero_force_ros"].z))
	var ay_lab := absf(float(out_lab["net_force_ros"].y) / maxf(m_gp.config.mass, 1.0)) / 9.81

	m_gp.reset()
	m_gp.set_track_grip_profile(Full14DofScript.TrackGripProfile.FSG_REAL_TRACK)
	m_gp.chassis_state.linear_velocity = Vector3(22.0, 0.0, 0.0)
	for w in m_gp.wheels:
		w.omega = 22.0 / maxf(m_gp.config.wheel_radius, 0.1)
	var out_real := {}
	for _k in range(70):
		out_real = m_gp.step_standalone(inp_gp, 0.0025)
	var fz_aero_real := absf(float(out_real["forces"]["aero_force_ros"].z))
	var ay_real := absf(float(out_real["net_force_ros"].y) / maxf(m_gp.config.mass, 1.0)) / 9.81
	var aero_ratio := fz_aero_real / maxf(fz_aero_lab, 1.0)
	_check(gp_def_ok and absf(m_gp.env_mu_scale - 0.86) < 1e-5 and absf(m_gp.env_aero_cl_scale - 0.64) < 1e-5 and ay_real < ay_lab - 0.05 and absf(aero_ratio - 0.64) < 0.06,
		"TrackGripProfile FSG_REAL_TRACK scales aero=%.3f and reduces lateral g (%.2fg -> %.2fg)" % [aero_ratio, ay_lab, ay_real], st)

	print("  [PROBE_DiL] RMS_FLAT=%.6f RMS_ISO_A=%.6f RMS_ISO_B=%.6f RMS_ISO_C=%.6f | BUMP35_Fz_FL=%.1fN BUMP35_Fz_RL=%.1fN" % [rms_flat, rms_a, rms_b, rms_c, fz_peak_fl, fz_peak_rl])
	print("  [PROBE_DiL] STEER_TRQ straight=%.3fNm 05g(ay=%.2fg)=%.3fNm 12g(ay=%.2fg)=%.3fNm sat(ay=%.2fg)=%.3fNm" % [
		t_straight, float(trq_vals["mid_05g"]["ay"]), t_05g,
		float(trq_vals["high_12g"]["ay"]), t_12g,
		float(trq_vals["sat"]["ay"]), t_sat
	])
	print("  [PROBE_DiL] BAR_LAYOUT mix(drv=125Nm,reg=93.75Nm,mech=187.5Nm) -> drv_h=%.4f drv_cy=%.4f reg_h=%.4f reg_cy=%.4f mech_h=%.4f mech_cy=%.4f" % [
		float(lay_mix["drive_h"]), float(lay_mix["drive_cy"]),
		float(lay_mix["regen_h"]), float(lay_mix["regen_cy"]),
		float(lay_mix["mech_h"]), float(lay_mix["mech_cy"])
	])

	return st


class _RaycastProbe extends Node:
	var points: Array = []
	var base_y: float = 0.045
	var exclude_rids: Array[RID] = []
	var done: bool = false
	var hit_count: int = 0
	var buried_count: int = 0
	var nan_count: int = 0
	var min_clearance_m: float = 1e9
	var max_clearance_m: float = -1e9

	func _physics_process(_delta: float) -> void:
		if done:
			return
		var w3d := get_viewport().find_world_3d()
		if w3d == null or w3d.direct_space_state == null:
			return
		var space := w3d.direct_space_state
		for pt in points:
			var sx := float(pt.get("x_godot", NAN))
			var sz := float(pt.get("z_godot", NAN))
			if not is_finite(sx) or not is_finite(sz):
				nan_count += 1
				continue
			var q := PhysicsRayQueryParameters3D.create(
				Vector3(sx, 5.0, sz),
				Vector3(sx, -5.0, sz),
				32769,
				exclude_rids
			)
			var hit: Dictionary = space.intersect_ray(q)
			if hit.is_empty():
				buried_count += 1
				continue
			hit_count += 1
			var hit_pos: Vector3 = hit["position"]
			var clearance: float = base_y - hit_pos.y
			min_clearance_m = minf(min_clearance_m, clearance)
			max_clearance_m = maxf(max_clearance_m, clearance)
			if clearance <= 0.0:
				buried_count += 1
		done = true
		set_physics_process(false)


static func run_async(tree: SceneTree) -> Dictionary:
	var st := run()

	# Test 4 -- Live Sim.tscn FSG 2023 360-Point 3D Racing Line Visibility
	var sim := SimScene.instantiate()
	tree.root.add_child(sim)
	await tree.process_frame
	await tree.process_frame

	var trk := sim.get_node_or_null("Track") as Track
	var car := sim.get_node_or_null("Car") as MirenaCar
	_check(trk != null and car != null, "Test 4: Instantiated real Sim.tscn for FSG 2023 visibility check", st)

	# RewindBuffer 3s rewind check inside live SceneTree
	var rw_buf := RewindBufferScript.new()
	var rw_car := RigidBody3D.new()
	tree.root.add_child(rw_car)
	for rwi in range(80):
		rw_car.global_position = Vector3(float(rwi) * 0.5, 0.3, 0.0)
		rw_car.linear_velocity = Vector3(10.0, 0.0, 0.0)
		rw_buf.record_step(0.051, rw_car)
	var pos_before_rw := rw_car.global_position.x
	var rw_ok := rw_buf.rewind_seconds(3.0, rw_car) and rw_car.global_position.x < pos_before_rw - 20.0
	rw_car.queue_free()
	_check(rw_ok, "RewindBuffer 3.0s rewind restores earlier car state cleanly", st)
	if trk == null or car == null:
		if is_instance_valid(sim):
			sim.queue_free()
		return st

	# Test 6B: Split-Mu Test Toggle + Multi-Pin (3 slots) in live Sim.tscn
	var sim_auto_bind := tree.root.get_node_or_null("Sim")
	if sim_auto_bind != null:
		sim_auto_bind.set("car", car)
		sim_auto_bind.set("track", trk)
	var split_mu_ok := false
	var sim_ui_live := sim.find_child("MainMenu", true, false)
	if sim_ui_live != null:
		var chk_split := sim_ui_live.get("_chk_split_mu") as CheckButton
		if chk_split != null:
			chk_split.button_pressed = true
			await tree.process_frame
			var enabled_ok := bool(car.split_mu_test_enabled)
			chk_split.button_pressed = false
			await tree.process_frame
			var disabled_ok := not bool(car.split_mu_test_enabled)
			split_mu_ok = enabled_ok and disabled_ok
	_check(split_mu_ok, "Test 6B Split-Mu Test Toggle updates car.split_mu_test_enabled cleanly", st)

	var multi_pin_ok := true
	var sim_auto_pin := tree.root.get_node_or_null("Sim")
	if sim_auto_pin != null and sim_auto_pin.has_method("set_practice_pin"):
		for slot_i in range(3):
			car.global_position = Vector3(float(slot_i + 1) * 5.0, 0.25, -float(slot_i) * 3.0)
			var set_ok := bool(sim_auto_pin.set_practice_pin(slot_i, car))
			var has_ok := bool(sim_auto_pin.has_practice_pin(slot_i))
			var info: Dictionary = sim_auto_pin.get_practice_pin_info(slot_i)
			if not set_ok or not has_ok or info.is_empty():
				multi_pin_ok = false
		car.global_position = Vector3.ZERO
		for slot_i in range(3):
			var tp_ok := bool(sim_auto_pin.teleport_to_pin(slot_i, car))
			var exp_x := float(slot_i + 1) * 5.0
			if not tp_ok or absf(car.global_position.x - exp_x) > 0.2:
				multi_pin_ok = false
			sim_auto_pin.clear_practice_pin(slot_i)
	else:
		multi_pin_ok = false
	_check(multi_pin_ok, "Test 6B Multi-Pin 3 slots (set, info, teleport, clear) verified in live Sim.tscn", st)

	trk.load_track("res://TrackFiles/fsg_2023.json")
	car.set_racing_line_enabled(true)
	await tree.process_frame
	await tree.process_frame
	await tree.physics_frame

	# 4a. GroundMesh material has refraction_enabled == false (opaque pass)
	var gm := trk.get_node_or_null("Ground/GroundMesh") as MeshInstance3D
	var gm_mat: Material = null
	if gm != null:
		gm_mat = gm.material_override if gm.material_override != null else (gm.mesh.surface_get_material(0) if gm.mesh != null and gm.mesh.get_surface_count() > 0 else null)
	var gm_std := gm_mat as StandardMaterial3D
	var ground_opaque_ok := (gm_std != null and not gm_std.refraction_enabled and gm_std.transparency == BaseMaterial3D.TRANSPARENCY_DISABLED)
	_check(ground_opaque_ok, "Test 4: GroundMesh material is opaque (refraction_enabled=%s, transparency=%d)" % [
		str(gm_std.refraction_enabled) if gm_std != null else "null",
		int(gm_std.transparency) if gm_std != null else -1
	], st)

	# 4b. RacingLineVisualizer layers have CULL_DISABLED and render_priority > 0
	var rlv := trk.racing_line_vis
	var layer_names := ["RoadBumpsMesh", "RacingLineBorderMesh", "RacingLineMesh", "RacingLineMarkersMesh"]
	var layer_prios: Array[int] = []
	var layers_ok := (rlv != null and rlv.visible)
	if layers_ok:
		for lname in layer_names:
			var mi := rlv.get_node_or_null(lname) as MeshInstance3D
			var lmat := (mi.material_override as StandardMaterial3D) if mi != null else null
			if mi == null or lmat == null or lmat.cull_mode != BaseMaterial3D.CULL_DISABLED or lmat.render_priority <= 0:
				layers_ok = false
				break
			layer_prios.append(lmat.render_priority)
	var prio_ascending := (layer_prios.size() == 4 and layer_prios[0] < layer_prios[1] and layer_prios[1] < layer_prios[2] and layer_prios[2] < layer_prios[3])
	_check(layers_ok and prio_ascending, "Test 4: RacingLineVisualizer 4 layers have CULL_DISABLED and ascending render_priority > 0 (%s)" % str(layer_prios), st)

	# 4c. All 360 racing-line points above Ground collider via vertical raycast (0 buried, 0 NaNs, 0 transparent sort flip)
	var rl_pts: Array = rlv._last_points if rlv != null else []
	if rl_pts.size() != 360 and rlv != null:
		rl_pts = rlv.load_racing_line_json("res://Assets/RacingLines/fsg_2023_racing_line.json")
	var probe := _RaycastProbe.new()
	probe.points = rl_pts
	probe.base_y = (rlv.global_position.y if rlv != null else 0.0) + RacingLineVisualizerScript.RIBBON_Y_M
	probe.exclude_rids = [car.get_rid()]
	sim.add_child(probe)
	for _w in range(10):
		await tree.physics_frame
		if probe.done:
			break

	# Transparent sort flip check: GroundMesh is opaque (refraction_enabled=false, transparency=DISABLED) and ribbon prio > 0
	var ribbon_mi := rlv.get_node_or_null("RacingLineMesh") as MeshInstance3D if rlv != null else null
	var ribbon_mat := (ribbon_mi.material_override as StandardMaterial3D) if ribbon_mi != null else null
	var ground_in_transparent_queue := (gm_std == null or gm_std.refraction_enabled or gm_std.transparency != BaseMaterial3D.TRANSPARENCY_DISABLED)
	var sort_flip_hidden := 0
	if ground_in_transparent_queue:
		var gm_prio: int = gm_std.render_priority if gm_std != null else 0
		var rl_prio: int = ribbon_mat.render_priority if ribbon_mat != null else 0
		if gm_prio >= rl_prio:
			var gm_center: Vector3 = gm.global_transform * gm.get_aabb().get_center() if gm != null else Vector3.ZERO
			var rl_center: Vector3 = ribbon_mi.global_transform * ribbon_mi.get_aabb().get_center() if ribbon_mi != null else Vector3.ZERO
			for pt in rl_pts:
				var cam_p := Vector3(float(pt.get("x_godot", 0.0)), 2.5, float(pt.get("z_godot", 0.0)))
				if cam_p.distance_squared_to(gm_center) < cam_p.distance_squared_to(rl_center):
					sort_flip_hidden += 1

	print("  [TEST4_RACING_LINE] pts=%d hits=%d buried=%d nans=%d clearance=[%.4f,%.4f]m sort_flip_hidden=%d ground_refract=%s prios=%s" % [
		rl_pts.size(), probe.hit_count, probe.buried_count, probe.nan_count,
		probe.min_clearance_m, probe.max_clearance_m, sort_flip_hidden,
		str(gm_std.refraction_enabled) if gm_std != null else "null", str(layer_prios)
	])
	_check(probe.done and rl_pts.size() == 360 and probe.hit_count == 360 and probe.buried_count == 0 and probe.nan_count == 0 and probe.min_clearance_m > 0.0 and sort_flip_hidden == 0,
		"Test 4: 360/360 FSG 2023 racing line points above Ground collider (hits=%d buried=%d nans=%d min_clr=%.4fm sort_flip=%d)" % [
			probe.hit_count, probe.buried_count, probe.nan_count, probe.min_clearance_m, sort_flip_hidden
		], st)

	var sim_auto := tree.root.get_node_or_null("Sim")
	if sim_auto != null:
		sim_auto.set("car", null)
		sim_auto.set("track", null)
	sim.queue_free()
	await tree.process_frame
	return st
