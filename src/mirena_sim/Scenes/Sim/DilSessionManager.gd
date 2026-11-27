class_name DilSessionManager
extends Node

## Driver-in-the-Loop session manager:
## - Official FS lap & 3-sector timer (S1 0-33%, S2 33-66%, S3 66-100%)
## - Official DOO cone penalty (+2.0 s/cone) with rectangular footprint & per-lap debounce
## - 20 Hz lap recorder & translucent Ghost Car (Off / Best Lap / LTS Optimal)

enum GhostMode {
	OFF = 0,
	BEST_LAP = 1,
	LTS_OPTIMAL = 2,
}

const PENALTY_PER_CONE_S: float = 2.0
const CAR_HALF_LENGTH_M: float = 1.425
const CAR_HALF_WIDTH_M: float = 0.720
const CONE_RADIUS_M: float = 0.250
const SAMPLE_INTERVAL_S: float = 0.05 # 20 Hz

var ghost_mode: int = GhostMode.OFF

# Lap & sector timers
var lap_active: bool = false
var current_lap_s: float = 0.0
var last_lap_s: float = 0.0
var best_lap_s: float = 0.0
var sector_times: Array[float] = [0.0, 0.0, 0.0]
var _cur_sector_idx: int = 0
var _sector_start_t: float = 0.0
var live_delta_s: float = 0.0

# Cone penalty state (DOO)
var cones_hit: int = 0
var penalty_s: float = 0.0
var _hit_cone_ids: Dictionary = {} # key -> true
var _cone_cache: Array = [] # [{ "id": int, "pos": Vector3, "node": Node3D }]

# Track timing reference
var _start_origin: Transform3D = Transform3D.IDENTITY
var _track_length_m: float = 100.0
var _track_curve: Curve3D = null
var _s_cur_m: float = 0.0
var _s_max_frac: float = 0.0
var _prev_line_signed: float = -1.0
var _has_prev_pos: bool = false
var _lap_armed: bool = false

# Recordings [ {s, t, x, y, z, yaw, roll, pitch} ]
var _cur_trace: Array = []
var _best_trace: Array = []
var _lts_trace: Array = []
var _sample_accum_s: float = 0.0

# Ghost visual node
var _ghost_root: Node3D = null
var _ghost_mesh: MeshInstance3D = null
var ghost_opacity: float = 0.34
var pace_rabbit_offset_s: float = 0.0
var ghost_visible_override: bool = true
var current_tv_map_name: String = "AUTOCROSS"
var current_preset_name: String = "default"
var lap_history: Array[Dictionary] = []

# Live DiL CasADi Delta Engine & Session Recorder
const MAX_DIL_SESSION_SAMPLES: int = 36000
const BRAKE_APPROACH_WINDOW_M: float = 45.0

var reference_enabled: bool = true
var _casadi_ref_path: String = ""
var _casadi_ref_meta: Dictionary = {}
var _casadi_ref_points: Array[Dictionary] = []
var _casadi_ref_corners: Array[Dictionary] = []
var _casadi_braking_zones: Array[Dictionary] = []
var _casadi_ref_length_m: float = 0.0
var _casadi_sector_ref_s: Array[float] = [0.0, 0.0, 0.0]

var _bound_car: Node = null
var _bound_racing_line_vis: Node = null

var _prev_car_yaw: float = 0.0
var _prev_speed_mps: float = 0.0
var _has_prev_kinematics: bool = false
var _last_delta_t_s: float = 0.0
var _sector_deltas_s: Array[float] = [0.0, 0.0, 0.0]
var _phase_deltas_s: Dictionary = {
	"braking": 0.0,
	"entry": 0.0,
	"apex": 0.0,
	"exit": 0.0,
}
var _corner_deltas: Dictionary = {}
var _live_dil_telemetry: Dictionary = {}
var _dil_session_samples: Array[Dictionary] = []
var _dil_sample_accum_s: float = 0.0
var _session_elapsed_s: float = 0.0


func _ready() -> void:
	_ensure_ghost_node()


func _ensure_ghost_node() -> void:
	if _ghost_root != null:
		return
	_ghost_root = Node3D.new()
	_ghost_root.name = "GhostCar"
	_ghost_root.visible = false
	_ghost_mesh = MeshInstance3D.new()
	_ghost_mesh.name = "GhostMesh"
	_ghost_mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.render_priority = 15
	mat.albedo_color = Color(0.28, 0.90, 1.00, 0.34)
	mat.emission_enabled = true
	mat.emission = Color(0.20, 0.85, 1.00)
	mat.emission_energy_multiplier = 0.75
	_ghost_mesh.material_override = mat
	if ResourceLoader.exists("res://Assets/Models/Car/TeR.res"):
		_ghost_mesh.mesh = load("res://Assets/Models/Car/TeR.res")
		# TeR.res nose points +Z in local mesh space; match CAR_MESH in mirena_car.tscn
		_ghost_mesh.rotation = Vector3(0.0, PI, 0.0)
		_ghost_mesh.position = Vector3(0.0, 0.422, 0.6)
	else:
		var box := BoxMesh.new()
		box.size = Vector3(1.38, 0.55, 2.75)
		_ghost_mesh.mesh = box
		_ghost_mesh.position = Vector3(0.0, 0.422, 0.0)
	_ghost_root.add_child(_ghost_mesh)
	add_child(_ghost_root)


func set_ghost_mode(mode: int) -> void:
	ghost_mode = clampi(mode, GhostMode.OFF, GhostMode.LTS_OPTIMAL)
	if _ghost_root != null:
		_ghost_root.visible = (ghost_visible_override and ghost_mode != GhostMode.OFF and not _get_active_ghost_trace().is_empty())


func set_ghost_source(source: Variant) -> void:
	if source is int:
		set_ghost_mode(int(source))
		return
	var s := str(source).strip_edges().to_upper()
	match s:
		"LTS_OPTIMAL", "LTS", "LTS OPTIMAL":
			set_ghost_mode(GhostMode.LTS_OPTIMAL)
		"BEST_SESSION_LAP", "BEST_LAP", "BEST SESSION LAP", "BEST":
			set_ghost_mode(GhostMode.BEST_LAP)
		_:
			set_ghost_mode(GhostMode.OFF)


func get_ghost_source() -> String:
	match ghost_mode:
		GhostMode.BEST_LAP:
			return "BEST_SESSION_LAP"
		GhostMode.LTS_OPTIMAL:
			return "LTS_OPTIMAL"
		_:
			return "OFF"


func set_active_context(tv_map: String, preset_name: String) -> void:
	if not tv_map.strip_edges().is_empty():
		current_tv_map_name = tv_map.strip_edges()
	if not preset_name.strip_edges().is_empty():
		current_preset_name = preset_name.strip_edges()


func set_ghost_opacity(alpha: float) -> void:
	ghost_opacity = clampf(alpha, 0.05, 1.0)
	if _ghost_mesh != null and _ghost_mesh.material_override is StandardMaterial3D:
		var mat := _ghost_mesh.material_override as StandardMaterial3D
		mat.albedo_color.a = ghost_opacity


func set_pace_rabbit_offset(offset_s: float) -> void:
	pace_rabbit_offset_s = clampf(offset_s, 0.0, 5.0)


func set_ghost_visible(vis: bool) -> void:
	ghost_visible_override = vis
	if _ghost_root != null:
		_ghost_root.visible = (ghost_visible_override and ghost_mode != GhostMode.OFF and not _get_active_ghost_trace().is_empty())


func get_lap_history() -> Array[Dictionary]:
	return lap_history.duplicate(true)


func reset_cones_only() -> void:
	cones_hit = 0
	penalty_s = 0.0
	_hit_cone_ids.clear()
	for c_v in _cone_cache:
		if c_v is Dictionary:
			var c_node: Variant = c_v.get("node", null)
			if c_node != null and is_instance_valid(c_node) and c_node is Node3D:
				(c_node as Node3D).rotation = Vector3.ZERO


func configure_track_from_node(track_node: Node) -> void:
	if track_node == null:
		return
	var org: Transform3D = track_node.get("origin") if "origin" in track_node else Transform3D.IDENTITY
	var crv: Curve3D = track_node.get("track_curve") if "track_curve" in track_node else null
	var t_len := crv.get_baked_length() if (crv != null and crv.get_baked_length() > 1.0) else 200.0
	var cones_arr: Array = []
	if track_node.has_node("Gates"):
		var gates := track_node.get_node("Gates")
		var cid := 0
		for ch in gates.get_children():
			if ch is Node3D and (ch.is_in_group("Cones") or ch.get_class() == "Cone" or ("type" in ch)):
				var c3 := ch as Node3D
				cones_arr.append({"id": cid, "pos": c3.global_position if c3.is_inside_tree() else c3.position, "node": ch})
				cid += 1
	configure_track(org, t_len, crv, cones_arr)


func configure_track(origin_tf: Transform3D, track_len_m: float, curve: Curve3D = null, cones: Array = []) -> void:
	_start_origin = origin_tf
	_track_length_m = maxf(track_len_m, 10.0)
	_track_curve = curve
	_cone_cache.clear()
	for idx in range(cones.size()):
		var c_v: Variant = cones[idx]
		if c_v is Dictionary:
			_cone_cache.append({
				"id": int(c_v.get("id", idx)),
				"pos": Vector3(c_v.get("pos", Vector3.ZERO)),
				"node": c_v.get("node", null),
			})
		elif c_v is Vector3:
			_cone_cache.append({"id": idx, "pos": c_v, "node": null})
	reset_session()


func load_lts_trace_from_json(json_path: String) -> bool:
	return load_casadi_reference_json(json_path)


func load_casadi_reference_json(json_path: String) -> bool:
	_lts_trace.clear()
	_casadi_ref_points.clear()
	_casadi_ref_corners.clear()
	_casadi_braking_zones.clear()
	_casadi_ref_meta.clear()
	_casadi_ref_path = ""
	_casadi_ref_length_m = 0.0
	_casadi_sector_ref_s = [0.0, 0.0, 0.0]
	if json_path.is_empty() or not FileAccess.file_exists(json_path):
		return false
	var f := FileAccess.open(json_path, FileAccess.READ)
	if f == null:
		return false
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY:
		return false
	var d: Dictionary = parsed
	var raw_pts: Variant = d.get("waypoints", d.get("points", []))
	if not (raw_pts is Array) or (raw_pts as Array).is_empty():
		return false
	_casadi_ref_path = json_path
	if d.get("meta") is Dictionary:
		_casadi_ref_meta = (d.get("meta") as Dictionary).duplicate(true)
	if d.get("corners") is Array:
		for c_v in (d.get("corners") as Array):
			if c_v is Dictionary:
				_casadi_ref_corners.append((c_v as Dictionary).duplicate(true))

	var pts: Array = raw_pts
	var t_accum := 0.0
	var prev_s := 0.0
	for i in range(pts.size()):
		if not (pts[i] is Dictionary):
			continue
		var p: Dictionary = pts[i]
		var s_i := float(p.get("s", p.get("s_m", float(i) * 4.5)))
		var v_i := maxf(float(p.get("v_ref", p.get("v_mps", 12.0))), 1.5)
		if i > 0:
			t_accum += maxf(s_i - prev_s, 0.0) / v_i
		prev_s = s_i
		var x_ros := float(p.get("x", 0.0))
		var y_ros := float(p.get("y", 0.0))
		var xg := -y_ros
		var zg := -x_ros
		var yaw_rad := float(p.get("yaw", p.get("yaw_rad", 0.0)))
		var ax_val := float(p.get("ax_mps2", p.get("ax", float(p.get("ax_g", 0.0)) * 9.81)))
		var ay_val := float(p.get("ay_mps2", p.get("ay", float(p.get("ay_g", 0.0)) * 9.81)))
		var kappa_val := float(p.get("kappa", 0.0))
		var wz_val := float(p.get("wz_ref", p.get("wz_rads", kappa_val * v_i)))
		var delta_val := float(p.get("steer_rad", deg_to_rad(float(p.get("steer_deg", 0.0)))))
		var thr_val := float(p.get("throttle", 0.0))
		var brk_val := float(p.get("brake", 0.0))
		var grip_val := float(p.get("grip_util", clampf(sqrt(ax_val * ax_val + ay_val * ay_val) / 14.5, 0.0, 1.5)))
		var phase_val := String(p.get("phase", ""))
		var wt_raw: Variant = p.get("wheel_torques_nm", [])
		var wt_arr: Array[float] = [0.0, 0.0, 0.0, 0.0]
		if wt_raw is Array and (wt_raw as Array).size() >= 4:
			for wi in range(4):
				wt_arr[wi] = float((wt_raw as Array)[wi])
		else:
			var total_fx := 260.0 * ax_val
			var per_wheel_nm := (total_fx * 0.205) * 0.25
			wt_arr = [per_wheel_nm, per_wheel_nm, per_wheel_nm, per_wheel_nm]

		_lts_trace.append({
			"s": s_i,
			"t": t_accum,
			"x": xg,
			"y": 0.0,
			"z": zg,
			"yaw": yaw_rad,
			"roll": 0.0,
			"pitch": 0.0,
		})
		_casadi_ref_points.append({
			"s": s_i,
			"t": t_accum,
			"x_godot": xg,
			"z_godot": zg,
			"yaw": yaw_rad,
			"vx_ref": v_i,
			"ax_ref": ax_val,
			"ay_ref": ay_val,
			"wz_ref": wz_val,
			"delta_ref": delta_val,
			"kappa": kappa_val,
			"throttle_ref": thr_val,
			"brake_ref": brk_val,
			"grip_util_ref": grip_val,
			"phase": phase_val,
			"wheel_torques_nm": wt_arr,
		})

	if _casadi_ref_points.is_empty():
		return false

	_casadi_ref_length_m = maxf(float(d.get("length_m", prev_s)), maxf(prev_s, 10.0))
	var t_s1 := _interp_time_by_s(_lts_trace, _casadi_ref_length_m * 0.3333)
	var t_s2 := _interp_time_by_s(_lts_trace, _casadi_ref_length_m * 0.6667)
	var t_end := float(_lts_trace[_lts_trace.size() - 1]["t"])
	_casadi_sector_ref_s = [t_s1, maxf(t_s2 - t_s1, 0.0), maxf(t_end - t_s2, 0.0)]
	_build_casadi_braking_zones()
	return true


func _build_casadi_braking_zones() -> void:
	_casadi_braking_zones.clear()
	if not _casadi_ref_corners.is_empty():
		for c in _casadi_ref_corners:
			var s_brk := float(c.get("s_brake", -1.0))
			var s_ti := float(c.get("s_turn_in", c.get("s_apex", -1.0)))
			var s_apx := float(c.get("s_apex", s_ti))
			var s_ex := float(c.get("s_exit", s_apx + 15.0))
			if s_brk >= 0.0 and s_apx > s_brk + 0.5:
				_casadi_braking_zones.append({
					"id": int(c.get("corner_id", _casadi_braking_zones.size() + 1)),
					"name": String(c.get("name", "C%d" % (_casadi_braking_zones.size() + 1))),
					"s_start": s_brk,
					"s_turn_in": s_ti if s_ti > s_brk else s_apx,
					"s_apex": s_apx,
					"s_end": maxf(s_ti, s_apx),
					"s_exit": maxf(s_ex, s_apx + 5.0),
					"v_entry_mps": float(c.get("v_entry_kmh", 60.0)) / 3.6,
					"v_apex_mps": float(c.get("v_apex_kmh", 35.0)) / 3.6,
				})
	if _casadi_braking_zones.is_empty() and not _casadi_ref_points.is_empty():
		var in_z := false
		var z_start_s := 0.0
		var z_entry_v := 0.0
		var z_min_v := 999.0
		var z_id := 0
		for i in range(_casadi_ref_points.size()):
			var pt: Dictionary = _casadi_ref_points[i]
			var brk := float(pt.get("brake_ref", 0.0))
			var ph := String(pt.get("phase", "")).to_upper()
			var is_b := (brk >= 0.15 or ph == "BRAKING" or ph == "TRAIL_BRAKING")
			var s_i := float(pt.get("s", 0.0))
			var v_i := float(pt.get("vx_ref", 12.0))
			if is_b and not in_z:
				in_z = true
				z_start_s = s_i
				z_entry_v = v_i
				z_min_v = v_i
			elif is_b and in_z:
				z_min_v = minf(z_min_v, v_i)
			elif not is_b and in_z:
				in_z = false
				if s_i > z_start_s + 1.0:
					z_id += 1
					_casadi_braking_zones.append({
						"id": z_id,
						"name": "BZ%d" % z_id,
						"s_start": z_start_s,
						"s_turn_in": s_i,
						"s_apex": s_i,
						"s_end": s_i,
						"s_exit": s_i + 15.0,
						"v_entry_mps": z_entry_v,
						"v_apex_mps": z_min_v,
					})
		if in_z and not _casadi_ref_points.is_empty():
			var s_last := float(_casadi_ref_points[_casadi_ref_points.size() - 1].get("s", z_start_s))
			if s_last > z_start_s + 1.0:
				z_id += 1
				_casadi_braking_zones.append({
					"id": z_id,
					"name": "BZ%d" % z_id,
					"s_start": z_start_s,
					"s_turn_in": s_last,
					"s_apex": s_last,
					"s_end": s_last,
					"s_exit": s_last + 15.0,
					"v_entry_mps": z_entry_v,
					"v_apex_mps": z_min_v,
				})


func load_lts_trace_from_csv(csv_path: String) -> bool:
	_lts_trace.clear()
	if csv_path.is_empty() or not FileAccess.file_exists(csv_path):
		return false
	var f := FileAccess.open(csv_path, FileAccess.READ)
	if f == null:
		return false
	var header := f.get_csv_line()
	if header.is_empty():
		return false
	var col_idx: Dictionary = {}
	for i in range(header.size()):
		col_idx[header[i].strip_edges()] = i
	if not (col_idx.has("t_s") and col_idx.has("s_m") and col_idx.has("x") and col_idx.has("y")):
		return false
	var t0 := -1.0
	var stride := 0
	while not f.eof_reached():
		var row := f.get_csv_line()
		if row.size() < header.size():
			continue
		if col_idx.has("lap") and int(row[col_idx["lap"]]) != 1:
			continue
		stride += 1
		if (stride % 5) != 1:
			continue
		var t_val := float(row[col_idx["t_s"]])
		if t0 < 0.0:
			t0 = t_val
		var yaw_deg := float(row[col_idx["yaw_deg"]]) if col_idx.has("yaw_deg") else 0.0
		_lts_trace.append({
			"s": float(row[col_idx["s_m"]]),
			"t": t_val - t0,
			"x": -float(row[col_idx["y"]]),
			"y": 0.0,
			"z": -float(row[col_idx["x"]]),
			"yaw": deg_to_rad(yaw_deg),
			"roll": 0.0,
			"pitch": 0.0,
		})
	return not _lts_trace.is_empty()


func reset_session(keep_best: bool = true) -> void:
	lap_active = false
	current_lap_s = 0.0
	_cur_sector_idx = 0
	_sector_start_t = 0.0
	sector_times = [0.0, 0.0, 0.0]
	live_delta_s = 0.0
	cones_hit = 0
	penalty_s = 0.0
	_hit_cone_ids.clear()
	_s_cur_m = 0.0
	_s_max_frac = 0.0
	_prev_line_signed = -1.0
	_has_prev_pos = false
	_lap_armed = false
	_cur_trace.clear()
	_sample_accum_s = 0.0
	_has_prev_kinematics = false
	_last_delta_t_s = 0.0
	_sector_deltas_s = [0.0, 0.0, 0.0]
	_phase_deltas_s = {"braking": 0.0, "entry": 0.0, "apex": 0.0, "exit": 0.0}
	_corner_deltas.clear()
	_dil_sample_accum_s = 0.0
	_session_elapsed_s = 0.0
	if not keep_best:
		best_lap_s = 0.0
		last_lap_s = 0.0
		_best_trace.clear()
		lap_history.clear()
		_dil_session_samples.clear()
	if _ghost_root != null:
		_ghost_root.visible = false


func _get_active_ghost_trace() -> Array:
	if ghost_mode == GhostMode.BEST_LAP:
		return _best_trace
	if ghost_mode == GhostMode.LTS_OPTIMAL:
		return _lts_trace
	return []


func step_session(dt: float, car_pos: Vector3, car_basis: Basis, speed_mps: float, car_state: Dictionary = {}) -> void:
	if dt <= 0.0:
		return
	_session_elapsed_s += dt

	# Forward vector in Godot car frame is -Z
	var fwd_line := -_start_origin.basis.z
	fwd_line.y = 0.0
	fwd_line = fwd_line.normalized() if fwd_line.length() > 1e-4 else Vector3(0.0, 0.0, -1.0)
	var lat_line := Vector3(-fwd_line.z, 0.0, fwd_line.x)

	var rel := car_pos - _start_origin.origin
	rel.y = 0.0
	var line_signed := rel.dot(fwd_line)
	var lat_dist := absf(rel.dot(lat_line))

	if _track_curve != null and _track_curve.get_baked_length() > 1.0:
		_s_cur_m = _track_curve.get_closest_offset(car_pos)
	else:
		_s_cur_m = maxf(_s_cur_m + maxf(speed_mps, 0.0) * dt, 0.0)
		if _s_cur_m > _track_length_m:
			_s_cur_m = fmod(_s_cur_m, _track_length_m)

	var s_frac := clampf(_s_cur_m / maxf(_track_length_m, 1.0), 0.0, 1.0)
	if lap_active:
		current_lap_s += dt
		_s_max_frac = maxf(_s_max_frac, s_frac)
		# Check sector splits
		if _cur_sector_idx == 0 and s_frac >= 0.3333:
			sector_times[0] = current_lap_s - _sector_start_t
			_sector_start_t = current_lap_s
			_cur_sector_idx = 1
		elif _cur_sector_idx == 1 and s_frac >= 0.6667:
			sector_times[1] = current_lap_s - _sector_start_t
			_sector_start_t = current_lap_s
			_cur_sector_idx = 2

	# Detect start / finish crossing in forward direction near start gate
	if _has_prev_pos and lat_dist < 8.0 and _prev_line_signed <= 0.0 and line_signed > 0.0 and speed_mps > 0.5:
		if lap_active and _s_max_frac >= 0.60:
			sector_times[2] = current_lap_s - _sector_start_t
			var raw_lap := current_lap_s
			var lap_cones := cones_hit
			var lap_pen := penalty_s
			var total_lap_with_pen := raw_lap + lap_pen
			last_lap_s = total_lap_with_pen
			lap_history.append({
				"lap": lap_history.size() + 1,
				"s1_s": sector_times[0],
				"s2_s": sector_times[1],
				"s3_s": sector_times[2],
				"sectors": sector_times.duplicate(),
				"raw_s": raw_lap,
				"cones": lap_cones,
				"penalty_s": lap_pen,
				"official_s": total_lap_with_pen,
				"tv_map": current_tv_map_name,
				"preset_name": current_preset_name,
			})
			if best_lap_s <= 0.0 or total_lap_with_pen < best_lap_s:
				best_lap_s = total_lap_with_pen
				_best_trace = _cur_trace.duplicate(true)
			# Start new lap cleanly
			current_lap_s = 0.0
			_cur_sector_idx = 0
			_sector_start_t = 0.0
			cones_hit = 0
			penalty_s = 0.0
			_hit_cone_ids.clear()
			_s_max_frac = 0.0
			_cur_trace.clear()
			_sample_accum_s = 0.0
		elif not lap_active:
			lap_active = true
			current_lap_s = 0.0
			_cur_sector_idx = 0
			_sector_start_t = 0.0
			_s_max_frac = 0.0
			_cur_trace.clear()
			_sample_accum_s = 0.0

	_prev_line_signed = line_signed
	_has_prev_pos = true

	# Cone collision check (DOO +2.0 s) against car rectangular footprint
	if not _cone_cache.is_empty():
		var car_fwd := -car_basis.z
		car_fwd.y = 0.0
		car_fwd = car_fwd.normalized() if car_fwd.length() > 1e-4 else Vector3(0.0, 0.0, -1.0)
		var car_right := Vector3(-car_fwd.z, 0.0, car_fwd.x)
		var half_l := CAR_HALF_LENGTH_M + CONE_RADIUS_M
		var half_w := CAR_HALF_WIDTH_M + CONE_RADIUS_M
		for c_v in _cone_cache:
			var c: Dictionary = c_v
			var cid := int(c["id"])
			if _hit_cone_ids.has(cid):
				continue
			var cp: Vector3 = c["pos"]
			var d_vec := cp - car_pos
			d_vec.y = 0.0
			if absf(d_vec.dot(car_fwd)) <= half_l and absf(d_vec.dot(car_right)) <= half_w:
				_hit_cone_ids[cid] = true
				cones_hit += 1
				penalty_s = float(cones_hit) * PENALTY_PER_CONE_S
				var c_node: Variant = c.get("node", null)
				if c_node != null and is_instance_valid(c_node) and c_node is Node3D:
					(c_node as Node3D).rotation.z = 1.15

	# Record current lap at 20 Hz
	if lap_active:
		_sample_accum_s += dt
		if _cur_trace.is_empty() or _sample_accum_s >= SAMPLE_INTERVAL_S:
			_sample_accum_s = 0.0
			var eul := car_basis.get_euler(EulerOrder.EULER_ORDER_YXZ)
			_cur_trace.append({
				"s": _s_cur_m,
				"t": current_lap_s,
				"x": car_pos.x,
				"y": car_pos.y,
				"z": car_pos.z,
				"yaw": eul.y,
				"roll": eul.z,
				"pitch": eul.x,
			})

	# Ghost interpolation & live_delta_s
	var active_trace := _get_active_ghost_trace()
	if not active_trace.is_empty():
		var t_ghost_at_s := _interp_time_by_s(active_trace, _s_cur_m)
		live_delta_s = (current_lap_s + penalty_s) - t_ghost_at_s if lap_active else 0.0
		if _ghost_root != null:
			_ghost_root.visible = (ghost_visible_override and ghost_mode != GhostMode.OFF)
			if _ghost_root.visible:
				var ghost_t := (current_lap_s + pace_rabbit_offset_s) if lap_active else pace_rabbit_offset_s
				var pose := _interp_pose_by_t(active_trace, ghost_t)
				_ghost_root.global_position = pose["pos"]
				var b := Basis.from_euler(Vector3(float(pose["pitch"]), float(pose["yaw"]), float(pose["roll"])), EulerOrder.EULER_ORDER_YXZ)
				_ghost_root.global_transform.basis = b
	else:
		live_delta_s = 0.0
		if _ghost_root != null:
			_ghost_root.visible = false

	_update_live_dil_delta(dt, car_pos, car_basis, speed_mps, car_state)


static func _interp_time_by_s(trace: Array, s_target: float) -> float:
	if trace.is_empty():
		return 0.0
	if s_target <= float(trace[0]["s"]):
		return float(trace[0]["t"])
	var n := trace.size()
	for i in range(1, n):
		var s0 := float(trace[i - 1]["s"])
		var s1 := float(trace[i]["s"])
		if s_target <= s1:
			var u := (s_target - s0) / maxf(s1 - s0, 1e-4)
			return lerpf(float(trace[i - 1]["t"]), float(trace[i]["t"]), clampf(u, 0.0, 1.0))
	return float(trace[n - 1]["t"])


static func _interp_pose_by_t(trace: Array, t_target: float) -> Dictionary:
	if trace.is_empty():
		return {"pos": Vector3.ZERO, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
	var n := trace.size()
	var t_max := maxf(float(trace[n - 1]["t"]), 1e-4)
	var t_wrap := fposmod(t_target, t_max)
	var idx1 := n - 1
	for i in range(1, n):
		if t_wrap <= float(trace[i]["t"]):
			idx1 = i
			break
	var idx0 := maxi(0, idx1 - 1)
	var d0: Dictionary = trace[idx0]
	var d1: Dictionary = trace[idx1]
	var t0 := float(d0["t"])
	var t1 := float(d1["t"])
	var u := clampf((t_wrap - t0) / maxf(t1 - t0, 1e-4), 0.0, 1.0)
	var p0 := Vector3(float(d0["x"]), float(d0["y"]), float(d0["z"]))
	var p1 := Vector3(float(d1["x"]), float(d1["y"]), float(d1["z"]))
	return {
		"pos": p0.lerp(p1, u),
		"yaw": lerp_angle(float(d0["yaw"]), float(d1["yaw"]), u),
		"roll": lerpf(float(d0["roll"]), float(d1["roll"]), u),
		"pitch": lerpf(float(d0["pitch"]), float(d1["pitch"]), u),
	}


func export_session_summary() -> String:
	var log_dir := "user://telemetry_logs"
	var abs_dir := ProjectSettings.globalize_path(log_dir)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var dt_dict := Time.get_datetime_dict_from_system()
	var stamp := "%04d%02d%02d_%02d%02d%02d" % [
		int(dt_dict.get("year", 2026)),
		int(dt_dict.get("month", 1)),
		int(dt_dict.get("day", 1)),
		int(dt_dict.get("hour", 0)),
		int(dt_dict.get("minute", 0)),
		int(dt_dict.get("second", 0)),
	]
	var out_path := "%s/session_%s.json" % [log_dir, stamp]
	var payload := {
		"timestamp": stamp,
		"track_length_m": _track_length_m,
		"best_lap_s": best_lap_s,
		"last_lap_s": last_lap_s,
		"tv_map": current_tv_map_name,
		"preset_name": current_preset_name,
		"ghost_source": get_ghost_source(),
		"laps": lap_history.duplicate(true),
	}
	var f := FileAccess.open(out_path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(JSON.stringify(payload, "  "))
	f.flush()
	f.close()
	return out_path


func get_summary() -> Dictionary:
	return {
		"lap_active": lap_active,
		"current_lap_s": current_lap_s,
		"last_lap_s": last_lap_s,
		"best_lap_s": best_lap_s,
		"live_delta_s": live_delta_s,
		"sector_times": sector_times.duplicate(),
		"cones_hit": cones_hit,
		"penalty_s": penalty_s,
		"ghost_mode": ghost_mode,
		"ghost_source": get_ghost_source(),
		"tv_map": current_tv_map_name,
		"preset_name": current_preset_name,
		"has_best_ghost": not _best_trace.is_empty(),
		"has_lts_ghost": not _lts_trace.is_empty(),
		"reference_enabled": reference_enabled,
		"has_casadi_reference": not _casadi_ref_points.is_empty(),
	}


func bind_vehicle(car_node: Node) -> void:
	_bound_car = car_node


func bind_racing_line_visualizer(vis_node: Node) -> void:
	_bound_racing_line_vis = vis_node


func set_reference_enabled(enabled: bool) -> void:
	reference_enabled = enabled


func is_reference_enabled() -> bool:
	return reference_enabled


func get_corner_deltas() -> Dictionary:
	return _corner_deltas.duplicate(true)


func get_live_dil_telemetry() -> Dictionary:
	if _live_dil_telemetry.is_empty():
		return {
			"s_m": _s_cur_m,
			"t_lap_s": current_lap_s,
			"t_ref_s": 0.0,
			"delta_t_s": 0.0,
			"vx_ms": 0.0,
			"vx_ref_ms": 0.0,
			"delta_vx_ms": 0.0,
			"ax_mps2": 0.0,
			"ax_ref_mps2": 0.0,
			"ay_mps2": 0.0,
			"ay_ref_mps2": 0.0,
			"wz_rads": 0.0,
			"wz_ref_rads": 0.0,
			"delta_wz_rads": 0.0,
			"delta_rad": 0.0,
			"delta_ref_rad": 0.0,
			"e_lat_m": 0.0,
			"brake_in": 0.0,
			"throttle_in": 0.0,
			"brake_ref": 0.0,
			"throttle_ref": 0.0,
			"brake_margin_m": -1.0,
			"brake_v_rel_ms": 0.0,
			"brake_prog": 0.0,
			"in_braking_zone": false,
			"approaching_braking_zone": false,
			"max_grip_util": 0.0,
			"phase": "STRAIGHT",
			"sector_deltas_s": _sector_deltas_s.duplicate(),
			"phase_deltas_s": _phase_deltas_s.duplicate(true),
			"corner_deltas": _corner_deltas.duplicate(true),
			"wheel_torques_ref_nm": [0.0, 0.0, 0.0, 0.0],
			"reference_enabled": reference_enabled,
			"has_reference": not _casadi_ref_points.is_empty(),
			"reference_label": String(_casadi_ref_meta.get("label", "")),
			"sample_count": _dil_session_samples.size(),
		}
	return _live_dil_telemetry.duplicate(true)


func _sample_casadi_reference_at_s(s_query: float) -> Dictionary:
	var n := _casadi_ref_points.size()
	if n == 0:
		return {}
	if n == 1:
		return _casadi_ref_points[0]
	var s_max := float(_casadi_ref_points[n - 1]["s"])
	var s_w := clampf(fposmod(s_query, maxf(s_max, 1.0)), 0.0, s_max) if s_max > 1.0 else 0.0
	var lo := 0
	var hi := n - 1
	while lo + 1 < hi:
		var mid := (lo + hi) >> 1
		if float(_casadi_ref_points[mid]["s"]) <= s_w:
			lo = mid
		else:
			hi = mid
	var p0: Dictionary = _casadi_ref_points[lo]
	var p1: Dictionary = _casadi_ref_points[hi]
	var s0 := float(p0["s"])
	var s1 := float(p1["s"])
	var u := clampf((s_w - s0) / maxf(s1 - s0, 1e-4), 0.0, 1.0)
	var wt0: Array = p0.get("wheel_torques_nm", [0.0, 0.0, 0.0, 0.0])
	var wt1: Array = p1.get("wheel_torques_nm", [0.0, 0.0, 0.0, 0.0])
	var wt_interp: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for wi in range(4):
		var a := float(wt0[wi]) if wi < wt0.size() else 0.0
		var b := float(wt1[wi]) if wi < wt1.size() else 0.0
		wt_interp[wi] = lerpf(a, b, u)
	var dx := float(p1["x_godot"]) - float(p0["x_godot"])
	var dz := float(p1["z_godot"]) - float(p0["z_godot"])
	var t_len := sqrt(dx * dx + dz * dz)
	var tx := dx / t_len if t_len > 1e-5 else 0.0
	var tz := dz / t_len if t_len > 1e-5 else -1.0
	return {
		"s": s_w,
		"t": lerpf(float(p0["t"]), float(p1["t"]), u),
		"x_godot": lerpf(float(p0["x_godot"]), float(p1["x_godot"]), u),
		"z_godot": lerpf(float(p0["z_godot"]), float(p1["z_godot"]), u),
		"tx": tx,
		"tz": tz,
		"yaw": lerp_angle(float(p0["yaw"]), float(p1["yaw"]), u),
		"vx_ref": lerpf(float(p0["vx_ref"]), float(p1["vx_ref"]), u),
		"ax_ref": lerpf(float(p0["ax_ref"]), float(p1["ax_ref"]), u),
		"ay_ref": lerpf(float(p0["ay_ref"]), float(p1["ay_ref"]), u),
		"wz_ref": lerpf(float(p0["wz_ref"]), float(p1["wz_ref"]), u),
		"delta_ref": lerpf(float(p0["delta_ref"]), float(p1["delta_ref"]), u),
		"kappa": lerpf(float(p0["kappa"]), float(p1["kappa"]), u),
		"throttle_ref": lerpf(float(p0["throttle_ref"]), float(p1["throttle_ref"]), u),
		"brake_ref": lerpf(float(p0["brake_ref"]), float(p1["brake_ref"]), u),
		"grip_util_ref": lerpf(float(p0["grip_util_ref"]), float(p1["grip_util_ref"]), u),
		"phase": String(p0.get("phase", "")),
		"wheel_torques_nm": wt_interp,
	}


func _extract_live_car_kinematics(dt: float, car_basis: Basis, speed_mps: float, car_state: Dictionary) -> Dictionary:
	var eul := car_basis.get_euler(EulerOrder.EULER_ORDER_YXZ)
	var yaw_now := eul.y
	var vx_ms := float(car_state.get("vx_ms", speed_mps))
	var wz_rads := 0.0
	var ax_mps2 := 0.0
	var ay_mps2 := 0.0
	var delta_rad := float(car_state.get("steer_rad", 0.0))
	var brake_in := float(car_state.get("brake", 0.0))
	var throttle_in := float(car_state.get("throttle", 0.0))
	var grip_util := float(car_state.get("grip_util", -1.0))

	if _has_prev_kinematics and dt > 1e-5:
		wz_rads = wrapf(yaw_now - _prev_car_yaw, -PI, PI) / dt
		ax_mps2 = (vx_ms - _prev_speed_mps) / dt
		ay_mps2 = vx_ms * wz_rads
	_prev_car_yaw = yaw_now
	_prev_speed_mps = vx_ms
	_has_prev_kinematics = true

	if car_state.has("wz_rads"):
		wz_rads = float(car_state["wz_rads"])
	if car_state.has("ax_mps2"):
		ax_mps2 = float(car_state["ax_mps2"])
	if car_state.has("ay_mps2"):
		ay_mps2 = float(car_state["ay_mps2"])

	if car_state.is_empty() and _bound_car != null and is_instance_valid(_bound_car):
		if "brake" in _bound_car:
			brake_in = clampf(float(_bound_car.get("brake")), 0.0, 1.0)
		if "gas" in _bound_car:
			throttle_in = clampf(float(_bound_car.get("gas")), 0.0, 1.0)
		if "steering" in _bound_car:
			delta_rad = float(_bound_car.get("steering"))
		if "model" in _bound_car and _bound_car.get("model") != null:
			var mdl: Variant = _bound_car.get("model")
			if "linear_velocity_ros" in mdl:
				var lv: Vector3 = mdl.get("linear_velocity_ros")
				if is_finite(lv.x):
					vx_ms = lv.x
			if "angular_velocity_ros" in mdl:
				var av: Vector3 = mdl.get("angular_velocity_ros")
				if is_finite(av.z):
					wz_rads = av.z
			if "net_force_chassis_ros" in mdl and "config" in _bound_car and _bound_car.get("config") != null:
				var nf: Vector3 = mdl.get("net_force_chassis_ros")
				var m_kg := maxf(float(_bound_car.get("config").get("mass")), 1.0)
				if is_finite(nf.x) and is_finite(nf.y):
					ax_mps2 = nf.x / m_kg
					ay_mps2 = nf.y / m_kg
		if "_tire_sat_pct_scratch" in _bound_car:
			var sat_arr: Variant = _bound_car.get("_tire_sat_pct_scratch")
			if sat_arr is Array or sat_arr is PackedFloat32Array or sat_arr is PackedFloat64Array:
				var m_sat := 0.0
				for sv in sat_arr:
					m_sat = maxf(m_sat, float(sv) * 0.01)
				grip_util = m_sat

	if grip_util < 0.0:
		var mu_ref := float(_casadi_ref_meta.get("mu_scale", 0.88)) * 1.45
		grip_util = clampf(sqrt(ax_mps2 * ax_mps2 + ay_mps2 * ay_mps2) / maxf(mu_ref * 9.81, 1.0), 0.0, 1.8)

	return {
		"vx_ms": vx_ms if is_finite(vx_ms) else 0.0,
		"wz_rads": wz_rads if is_finite(wz_rads) else 0.0,
		"ax_mps2": ax_mps2 if is_finite(ax_mps2) else 0.0,
		"ay_mps2": ay_mps2 if is_finite(ay_mps2) else 0.0,
		"delta_rad": delta_rad if is_finite(delta_rad) else 0.0,
		"brake_in": clampf(brake_in if is_finite(brake_in) else 0.0, 0.0, 1.0),
		"throttle_in": clampf(throttle_in if is_finite(throttle_in) else 0.0, 0.0, 1.0),
		"max_grip_util": grip_util if is_finite(grip_util) else 0.0,
	}


func _update_live_dil_delta(dt: float, car_pos: Vector3, car_basis: Basis, speed_mps: float, car_state: Dictionary) -> void:
	var kin := _extract_live_car_kinematics(dt, car_basis, speed_mps, car_state)
	var vx_ms: float = kin["vx_ms"]
	var wz_rads: float = kin["wz_rads"]
	var ax_mps2: float = kin["ax_mps2"]
	var ay_mps2: float = kin["ay_mps2"]
	var delta_rad: float = kin["delta_rad"]
	var brake_in: float = kin["brake_in"]
	var throttle_in: float = kin["throttle_in"]
	var max_grip_util: float = kin["max_grip_util"]

	var has_ref := reference_enabled and not _casadi_ref_points.is_empty()
	var vx_ref := vx_ms
	var ax_ref := 0.0
	var ay_ref := 0.0
	var wz_ref := 0.0
	var delta_ref := 0.0
	var brake_ref := 0.0
	var throttle_ref := 0.0
	var t_ref := 0.0
	var e_lat_m := 0.0
	var delta_t_s := 0.0
	var brake_margin_m := -1.0
	var brake_prog := 0.0
	var in_braking_zone := false
	var approaching_braking_zone := false
	var phase_label := "STRAIGHT"
	var active_corner_name := ""
	var wt_ref: Array = [0.0, 0.0, 0.0, 0.0]

	if has_ref:
		var ref := _sample_casadi_reference_at_s(_s_cur_m)
		vx_ref = float(ref.get("vx_ref", vx_ms))
		ax_ref = float(ref.get("ax_ref", 0.0))
		ay_ref = float(ref.get("ay_ref", 0.0))
		wz_ref = float(ref.get("wz_ref", 0.0))
		delta_ref = float(ref.get("delta_ref", 0.0))
		brake_ref = float(ref.get("brake_ref", 0.0))
		throttle_ref = float(ref.get("throttle_ref", 0.0))
		t_ref = float(ref.get("t", 0.0))
		wt_ref = ref.get("wheel_torques_nm", [0.0, 0.0, 0.0, 0.0])

		var rx := float(ref.get("x_godot", car_pos.x))
		var rz := float(ref.get("z_godot", car_pos.z))
		var tx := float(ref.get("tx", 0.0))
		var tz := float(ref.get("tz", -1.0))
		var nx := -tz
		var nz := tx
		e_lat_m = (car_pos.x - rx) * nx + (car_pos.z - rz) * nz

		delta_t_s = ((current_lap_s + penalty_s) - t_ref) if lap_active else 0.0
		var dt_step := delta_t_s - _last_delta_t_s
		if not lap_active or absf(dt_step) > 2.0:
			dt_step = 0.0
		_last_delta_t_s = delta_t_s

		# Sector deltas vs CasADi reference
		for si in range(3):
			if sector_times[si] > 0.0 and _casadi_sector_ref_s[si] > 0.0:
				_sector_deltas_s[si] = sector_times[si] - _casadi_sector_ref_s[si]

		# Braking zone proximity, progression, and corner phase attribution
		var track_len := maxf(_casadi_ref_length_m, maxf(_track_length_m, 10.0))
		var best_ahead := 1e9
		for bz in _casadi_braking_zones:
			var s_st := float(bz["s_start"])
			var s_ti := float(bz["s_turn_in"])
			var s_apx := float(bz["s_apex"])
			var s_en := float(bz["s_end"])
			var s_ex := float(bz["s_exit"])
			var c_name := String(bz["name"])
			if _s_cur_m >= s_st and _s_cur_m <= s_en:
				in_braking_zone = true
				brake_margin_m = 0.0
				brake_prog = clampf((_s_cur_m - s_st) / maxf(s_en - s_st, 0.5), 0.0, 1.0)
				active_corner_name = c_name
				if _s_cur_m < s_ti - 2.0:
					phase_label = "BRAKING"
				else:
					phase_label = "ENTRY"
			elif _s_cur_m > s_en and _s_cur_m <= s_apx + 4.0:
				active_corner_name = c_name
				phase_label = "APEX"
			elif _s_cur_m > s_apx + 4.0 and _s_cur_m <= s_ex:
				active_corner_name = c_name
				phase_label = "EXIT"

			var d_ahead := fposmod(s_st - _s_cur_m, track_len)
			if d_ahead < best_ahead:
				best_ahead = d_ahead

		if not in_braking_zone and best_ahead < 1e8:
			brake_margin_m = best_ahead
			approaching_braking_zone = (best_ahead <= BRAKE_APPROACH_WINDOW_M)

		if phase_label == "STRAIGHT":
			var r_ph := String(ref.get("phase", "")).to_upper()
			if r_ph == "BRAKING" or brake_ref >= 0.35:
				phase_label = "BRAKING"
			elif r_ph == "TRAIL_BRAKING" or brake_ref >= 0.15:
				phase_label = "ENTRY"
			elif absf(float(ref.get("kappa", 0.0))) > 0.055:
				phase_label = "APEX"
			elif throttle_ref >= 0.65:
				phase_label = "EXIT"

		# Accumulate per-phase and per-corner deltas
		if lap_active and absf(dt_step) < 1.0:
			var ph_key := phase_label.to_lower()
			if _phase_deltas_s.has(ph_key):
				_phase_deltas_s[ph_key] = float(_phase_deltas_s[ph_key]) + dt_step
			if not active_corner_name.is_empty():
				if not _corner_deltas.has(active_corner_name):
					_corner_deltas[active_corner_name] = {
						"braking_dt_s": 0.0,
						"entry_dt_s": 0.0,
						"apex_dt_s": 0.0,
						"exit_dt_s": 0.0,
						"total_dt_s": 0.0,
						"min_vx_delta_ms": 999.0,
						"entry_v_rel_ms": vx_ms - vx_ref,
					}
				var cd: Dictionary = _corner_deltas[active_corner_name]
				var sub_key := "%s_dt_s" % ph_key
				if cd.has(sub_key):
					cd[sub_key] = float(cd[sub_key]) + dt_step
				cd["total_dt_s"] = float(cd["total_dt_s"]) + dt_step
				cd["min_vx_delta_ms"] = minf(float(cd["min_vx_delta_ms"]), vx_ms - vx_ref)

	var delta_vx_ms := vx_ms - vx_ref
	var delta_wz_rads := wz_rads - wz_ref

	_live_dil_telemetry = {
		"s_m": _s_cur_m,
		"t_lap_s": current_lap_s,
		"t_ref_s": t_ref,
		"delta_t_s": delta_t_s,
		"vx_ms": vx_ms,
		"vx_ref_ms": vx_ref,
		"delta_vx_ms": delta_vx_ms,
		"ax_mps2": ax_mps2,
		"ax_ref_mps2": ax_ref,
		"ay_mps2": ay_mps2,
		"ay_ref_mps2": ay_ref,
		"wz_rads": wz_rads,
		"wz_ref_rads": wz_ref,
		"delta_wz_rads": delta_wz_rads,
		"delta_rad": delta_rad,
		"delta_ref_rad": delta_ref,
		"e_lat_m": e_lat_m,
		"brake_in": brake_in,
		"throttle_in": throttle_in,
		"brake_ref": brake_ref,
		"throttle_ref": throttle_ref,
		"brake_margin_m": brake_margin_m,
		"brake_v_rel_ms": delta_vx_ms,
		"brake_prog": brake_prog,
		"in_braking_zone": in_braking_zone,
		"approaching_braking_zone": approaching_braking_zone,
		"max_grip_util": max_grip_util,
		"phase": phase_label,
		"sector_deltas_s": _sector_deltas_s.duplicate(),
		"phase_deltas_s": _phase_deltas_s.duplicate(true),
		"corner_deltas": _corner_deltas.duplicate(true),
		"wheel_torques_ref_nm": wt_ref,
		"reference_enabled": reference_enabled,
		"has_reference": has_ref,
		"reference_label": String(_casadi_ref_meta.get("label", "")),
		"sample_count": _dil_session_samples.size(),
	}

	# Feed live state to bound RacingLineVisualizer
	if _bound_racing_line_vis != null and is_instance_valid(_bound_racing_line_vis):
		if _bound_racing_line_vis.has_method("update_live_driver_state"):
			_bound_racing_line_vis.update_live_driver_state(_s_cur_m, vx_ms, brake_in, vx_ref)

	# Record 20 Hz DiL session telemetry for export
	if lap_active or speed_mps > 0.5:
		_dil_sample_accum_s += dt
		if _dil_session_samples.is_empty() or _dil_sample_accum_s >= SAMPLE_INTERVAL_S:
			_dil_sample_accum_s = 0.0
			if _dil_session_samples.size() < MAX_DIL_SESSION_SAMPLES:
				_dil_session_samples.append({
					"t_s": _session_elapsed_s,
					"lap": lap_history.size() + 1,
					"s_m": _s_cur_m,
					"delta_t_s": delta_t_s,
					"vx_ms": vx_ms,
					"vx_ref_ms": vx_ref,
					"delta_vx_ms": delta_vx_ms,
					"e_lat_m": e_lat_m,
					"wz_rads": wz_rads,
					"wz_ref_rads": wz_ref,
					"delta_wz_rads": delta_wz_rads,
					"ax_mps2": ax_mps2,
					"ax_ref_mps2": ax_ref,
					"ay_mps2": ay_mps2,
					"ay_ref_mps2": ay_ref,
					"delta_rad": delta_rad,
					"delta_ref_rad": delta_ref,
					"brake_in": brake_in,
					"brake_ref": brake_ref,
					"brake_margin_m": brake_margin_m,
					"brake_v_rel_ms": delta_vx_ms,
					"brake_prog": brake_prog,
					"max_grip_util": max_grip_util,
					"phase": phase_label,
				})


static func _resolve_dil_export_dir() -> String:
	var res_abs := ProjectSettings.globalize_path("res://")
	var repo_candidate := res_abs.path_join("../..").simplify_path()
	if DirAccess.dir_exists_absolute(repo_candidate.path_join("src")):
		var out_dir := repo_candidate.path_join("outputs/dil_sessions")
		if DirAccess.make_dir_recursive_absolute(out_dir) == OK:
			return out_dir
	var fallback := ProjectSettings.globalize_path("user://dil_sessions")
	DirAccess.make_dir_recursive_absolute(fallback)
	return fallback


func export_dil_session(tag: String = "") -> Dictionary:
	var out_dir := _resolve_dil_export_dir()
	var dt_dict := Time.get_datetime_dict_from_system()
	var stamp := "%04d%02d%02d_%02d%02d%02d" % [
		int(dt_dict.get("year", 2026)),
		int(dt_dict.get("month", 1)),
		int(dt_dict.get("day", 1)),
		int(dt_dict.get("hour", 0)),
		int(dt_dict.get("minute", 0)),
		int(dt_dict.get("second", 0)),
	]
	var safe_tag := tag.strip_edges().replace(" ", "_").replace("/", "_")
	var stem := "dil_session_%s_%s" % [stamp, safe_tag] if not safe_tag.is_empty() else "dil_session_%s" % stamp
	var json_path := out_dir.path_join("%s.json" % stem)
	var csv_path := out_dir.path_join("%s.csv" % stem)

	var payload := {
		"timestamp": stamp,
		"tag": safe_tag,
		"track_length_m": _track_length_m,
		"tv_map": current_tv_map_name,
		"preset_name": current_preset_name,
		"reference_path": _casadi_ref_path,
		"reference_meta": _casadi_ref_meta.duplicate(true),
		"best_lap_s": best_lap_s,
		"last_lap_s": last_lap_s,
		"sector_deltas_s": _sector_deltas_s.duplicate(),
		"phase_deltas_s": _phase_deltas_s.duplicate(true),
		"corner_deltas": _corner_deltas.duplicate(true),
		"laps": lap_history.duplicate(true),
		"live_snapshot": get_live_dil_telemetry(),
		"sample_count": _dil_session_samples.size(),
		"samples": _dil_session_samples,
	}

	var json_ok := false
	var fj := FileAccess.open(json_path, FileAccess.WRITE)
	if fj != null:
		fj.store_string(JSON.stringify(payload, "  "))
		fj.flush()
		fj.close()
		json_ok = true

	var csv_ok := false
	var fc := FileAccess.open(csv_path, FileAccess.WRITE)
	if fc != null:
		fc.store_line("t_s,lap,s_m,delta_t_s,vx_ms,vx_ref_ms,delta_vx_ms,e_lat_m,wz_rads,wz_ref_rads,delta_wz_rads,ax_mps2,ax_ref_mps2,ay_mps2,ay_ref_mps2,delta_rad,delta_ref_rad,brake_in,brake_ref,brake_margin_m,brake_v_rel_ms,brake_prog,max_grip_util,phase")
		for smp in _dil_session_samples:
			fc.store_line("%.4f,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%.3f,%.4f,%.4f,%.4f,%s" % [
				float(smp.get("t_s", 0.0)),
				int(smp.get("lap", 1)),
				float(smp.get("s_m", 0.0)),
				float(smp.get("delta_t_s", 0.0)),
				float(smp.get("vx_ms", 0.0)),
				float(smp.get("vx_ref_ms", 0.0)),
				float(smp.get("delta_vx_ms", 0.0)),
				float(smp.get("e_lat_m", 0.0)),
				float(smp.get("wz_rads", 0.0)),
				float(smp.get("wz_ref_rads", 0.0)),
				float(smp.get("delta_wz_rads", 0.0)),
				float(smp.get("ax_mps2", 0.0)),
				float(smp.get("ax_ref_mps2", 0.0)),
				float(smp.get("ay_mps2", 0.0)),
				float(smp.get("ay_ref_mps2", 0.0)),
				float(smp.get("delta_rad", 0.0)),
				float(smp.get("delta_ref_rad", 0.0)),
				float(smp.get("brake_in", 0.0)),
				float(smp.get("brake_ref", 0.0)),
				float(smp.get("brake_margin_m", -1.0)),
				float(smp.get("brake_v_rel_ms", 0.0)),
				float(smp.get("brake_prog", 0.0)),
				float(smp.get("max_grip_util", 0.0)),
				String(smp.get("phase", "STRAIGHT")),
			])
		fc.flush()
		fc.close()
		csv_ok = true

	return {
		"ok": json_ok and csv_ok,
		"json_path": json_path if json_ok else "",
		"csv_path": csv_path if csv_ok else "",
		"dir": out_dir,
		"sample_count": _dil_session_samples.size(),
		"lap_count": lap_history.size(),
	}
