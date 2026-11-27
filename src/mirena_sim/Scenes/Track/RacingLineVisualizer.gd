class_name RacingLineVisualizer
extends Node3D

## Premium 3D motorsport racing line visualizer:
##   - Main ribbon (0.42 m wide @ y = 0.045 m) with continuous high-contrast input color gradient:
##       #00E676 (WOT) -> #AEEA00 / #FFD600 (modulation) -> #FF6D00 (trail-braking) -> #FF1744 (threshold braking)
##   - Dark anthracite contrast border (0.54 m wide @ y = 0.040 m, alpha 0.55) for crisp daylight readability
##   - Directional chevrons every ~8 m (denser in braking zones), red transverse Brake Markers (1.6 m x 0.25 m),
##     and cyan/white Apex Diamonds on the inside edge of each optimal apex (toggleable via show_markers).
##   - Discrete Road Bump / Apex Curb markers. Zero dynamic per-frame allocations.

const RIBBON_HALF_WIDTH_M: float = 0.21   # 0.42 m main ribbon
const BORDER_HALF_WIDTH_M: float = 0.27   # 0.54 m dark contrast border
const BORDER_Y_M: float = 0.040
const RIBBON_Y_M: float = 0.045
const CHEVRON_Y_M: float = 0.050
const MARKER_Y_M: float = 0.053
const AABB_SAFETY_MARGIN_M: float = 15.0
const MESH_EXTRA_CULL_MARGIN_M: float = 50.0

const COL_BORDER := Color(0.05, 0.06, 0.09, 0.55)
const COL_WOT_GREEN := Color(0.000, 0.902, 0.463, 0.90)      # #00E676 - Acceleration (throttle >= 0.65)
const COL_PARTIAL_LIME := Color(0.682, 0.918, 0.000, 0.90)   # #AEEA00 - Partial throttle
const COL_COAST_CYAN := Color(0.161, 0.714, 0.965, 0.88)     # #29B6F6 - Coasting / lift-and-coast
const COL_TRAIL_AMBER := Color(1.000, 0.569, 0.000, 0.92)    # #FF9100 - Trail-braking (0.15 <= brake < 0.35)
const COL_BRAKE_RED := Color(1.000, 0.090, 0.267, 0.95)      # #FF1744 - Hard braking (brake >= 0.35)
const COL_APEX_CYAN := Color(0.000, 0.898, 1.000, 0.95)      # #00E5FF - Apex diamond
const COL_LIFT_BLUE := Color(0.240, 0.780, 1.000, 0.95)      # #3DC7FF - Lift-and-coast start marker
const COL_EXIT_GREEN := Color(0.000, 0.902, 0.463, 0.96)     # #00E676 - Exit to full gas marker
const COL_CHEVRON_STD := Color(0.96, 0.98, 1.00, 0.72)
const COL_CHEVRON_BRK := Color(1.00, 0.94, 0.82, 0.90)

enum ColorMode {
	VELOCITY = 0,
	DRIVING_PHASES = 1,
	LIVE_DELTA_V = 2,
}

enum LineDisplayMode {
	OFF = 0,
	STANDARD = 1,
	SIM_RACING_3D = 2,
}

var show_markers: bool = true
var color_mode: int = ColorMode.DRIVING_PHASES
var display_mode: int = LineDisplayMode.STANDARD
var brake_elevation_scale: float = 0.45
var speed_alert_gain: float = 1.0

const RENDER_PRIO_BUMPS: int = 10
const RENDER_PRIO_BORDER: int = 11
const RENDER_PRIO_RIBBON: int = 12
const RENDER_PRIO_MARKERS: int = 13

var _border_mesh_inst: MeshInstance3D = null
var _line_mesh_inst: MeshInstance3D = null
var _markers_mesh_inst: MeshInstance3D = null
var _bump_mesh_inst: MeshInstance3D = null
var _unshaded_mat: StandardMaterial3D = null
var _bump_mat: StandardMaterial3D = null
var _border_mat: StandardMaterial3D = null
var _line_mat: StandardMaterial3D = null
var _markers_mat: StandardMaterial3D = null
var _last_points: Array = []
var _last_is_closed: bool = true
var last_casadi_meta: Dictionary = {}
var last_casadi_corners: Array = []
var last_casadi_markers: Array = []
var _live_driver_speeds: Dictionary = {}
var _live_delta_dirty: bool = false
var _live_delta_accum: float = 0.0

# Sim-Racing 3D dynamic ribbon state (reuses single ArrayMesh with zero leaks)
var _dynamic_ribbon_mesh: ArrayMesh = null
var _braking_zones: Array[Dictionary] = []
var _sample_zone_id: PackedInt32Array = PackedInt32Array()
var _sample_zone_u: PackedFloat32Array = PackedFloat32Array()
var _sample_s_m: PackedFloat32Array = PackedFloat32Array()
var _total_track_s: float = 100.0
var _live_s_m: float = 0.0
var _live_vx_ms: float = 0.0
var _live_brake_in: float = 0.0
var _live_vx_ref_ms: float = 0.0
var _live_v_rel_ms: float = 0.0
var _live_active_zone_id: int = -1
var _live_brake_prog: float = 0.0
var _live_alert_factor: float = 0.0
var _live_dist_to_brake_m: float = -1.0
var _live_in_brake_zone: bool = false
var _pulse_phase_rad: float = 0.0
var _peak_y_elevation_m: float = 0.0
var _max_vertex_y_m: float = RIBBON_Y_M
var _nan_vertex_count: int = 0


func _ready() -> void:
	_ensure_nodes()
	visible = false


static func _make_overlay_material(prio: int) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.render_priority = prio
	mat.albedo_color = Color(1.0, 1.0, 1.0, 1.0)
	return mat


func _ensure_nodes() -> void:
	if _bump_mat == null:
		_bump_mat = _make_overlay_material(RENDER_PRIO_BUMPS)
	if _border_mat == null:
		_border_mat = _make_overlay_material(RENDER_PRIO_BORDER)
	if _line_mat == null:
		_line_mat = _make_overlay_material(RENDER_PRIO_RIBBON)
	if _markers_mat == null:
		_markers_mat = _make_overlay_material(RENDER_PRIO_MARKERS)
	_unshaded_mat = _line_mat

	if _border_mesh_inst == null:
		_border_mesh_inst = _make_mesh_child("RacingLineBorderMesh", _border_mat)
	if _line_mesh_inst == null:
		_line_mesh_inst = _make_mesh_child("RacingLineMesh", _line_mat)
	if _markers_mesh_inst == null:
		_markers_mesh_inst = _make_mesh_child("RacingLineMarkersMesh", _markers_mat)
	if _bump_mesh_inst == null:
		_bump_mesh_inst = _make_mesh_child("RoadBumpsMesh", _bump_mat)

	_border_mesh_inst.material_override = _border_mat
	_line_mesh_inst.material_override = _line_mat
	_markers_mesh_inst.material_override = _markers_mat
	_bump_mesh_inst.material_override = _bump_mat

	_configure_mesh_instance(_border_mesh_inst)
	_configure_mesh_instance(_line_mesh_inst)
	_configure_mesh_instance(_markers_mesh_inst)
	_configure_mesh_instance(_bump_mesh_inst)
	_markers_mesh_inst.visible = show_markers


static func _configure_mesh_instance(mi: MeshInstance3D) -> void:
	if mi == null:
		return
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mi.extra_cull_margin = MESH_EXTRA_CULL_MARGIN_M
	mi.ignore_occlusion_culling = true


static func _compute_vertices_aabb(verts: PackedVector3Array, grow_margin: float = AABB_SAFETY_MARGIN_M) -> AABB:
	var has_pt := false
	var aabb := AABB()
	for v in verts:
		if not is_finite(v.x) or not is_finite(v.y) or not is_finite(v.z):
			continue
		if not has_pt:
			aabb = AABB(v, Vector3.ZERO)
			has_pt = true
		else:
			aabb = aabb.expand(v)
	if not has_pt:
		return AABB()
	return aabb.grow(grow_margin)


static func _assign_mesh_custom_aabb(mesh: ArrayMesh, verts: PackedVector3Array) -> void:
	if mesh == null or verts.is_empty():
		return
	var computed_aabb := _compute_vertices_aabb(verts, AABB_SAFETY_MARGIN_M)
	if computed_aabb.has_volume():
		mesh.custom_aabb = computed_aabb


func _make_mesh_child(node_name: String, mat: StandardMaterial3D = null) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.name = node_name
	mi.material_override = mat if mat != null else _unshaded_mat
	_configure_mesh_instance(mi)
	add_child(mi)
	return mi


func set_markers_visible(on: bool) -> void:
	show_markers = on
	_ensure_nodes()
	if _markers_mesh_inst != null:
		_markers_mesh_inst.visible = show_markers


func set_color_mode(mode: int) -> void:
	var clamped := clampi(mode, ColorMode.VELOCITY, ColorMode.LIVE_DELTA_V)
	if color_mode == clamped and _line_mesh_inst != null and _line_mesh_inst.mesh != null:
		return
	color_mode = clamped
	if not _last_points.is_empty():
		_ensure_nodes()
		_assign_mesh_to_instance(
			_line_mesh_inst,
			build_ribbon_array_mesh(_last_points, _last_is_closed, color_mode, _live_driver_speeds)
		)


func get_color_mode() -> int:
	return color_mode


func set_display_mode(mode: int) -> void:
	var clamped := clampi(mode, LineDisplayMode.OFF, LineDisplayMode.SIM_RACING_3D)
	display_mode = clamped
	_ensure_nodes()
	if display_mode == LineDisplayMode.OFF:
		visible = false
		_peak_y_elevation_m = 0.0
		_max_vertex_y_m = RIBBON_Y_M
		return
	visible = true
	if not _last_points.is_empty():
		if display_mode == LineDisplayMode.SIM_RACING_3D:
			_rebuild_sim_racing_3d_ribbon()
		else:
			_peak_y_elevation_m = 0.0
			_max_vertex_y_m = RIBBON_Y_M
			_nan_vertex_count = 0
			_assign_mesh_to_instance(
				_line_mesh_inst,
				build_ribbon_array_mesh(_last_points, _last_is_closed, color_mode, _live_driver_speeds)
			)


func get_display_mode() -> int:
	return display_mode


func set_brake_elevation_scale(scale: float) -> void:
	brake_elevation_scale = clampf(scale if is_finite(scale) else 0.45, 0.0, 1.5)
	if display_mode == LineDisplayMode.SIM_RACING_3D and not _last_points.is_empty():
		_ensure_nodes()
		_rebuild_sim_racing_3d_ribbon()


func get_brake_elevation_scale() -> float:
	return brake_elevation_scale


func set_speed_alert_gain(gain: float) -> void:
	speed_alert_gain = clampf(gain if is_finite(gain) else 1.0, 0.0, 3.0)
	if display_mode == LineDisplayMode.SIM_RACING_3D and not _last_points.is_empty():
		_ensure_nodes()
		_rebuild_sim_racing_3d_ribbon()


func get_speed_alert_gain() -> float:
	return speed_alert_gain


func update_live_driver_state(s_m: float, vx_ms: float, brake_in: float, vx_ref_ms: float) -> void:
	var s_safe := s_m if is_finite(s_m) else 0.0
	var vx_safe := vx_ms if is_finite(vx_ms) else 0.0
	var brk_safe := clampf(brake_in if is_finite(brake_in) else 0.0, 0.0, 1.0)
	var vref_safe := vx_ref_ms if is_finite(vx_ref_ms) else vx_safe

	_live_s_m = maxf(s_safe, 0.0)
	_live_vx_ms = vx_safe
	_live_brake_in = brk_safe
	_live_vx_ref_ms = vref_safe
	_live_v_rel_ms = vx_safe - vref_safe
	_pulse_phase_rad = fmod(_pulse_phase_rad + 0.18, TAU)

	_live_in_brake_zone = false
	_live_active_zone_id = -1
	_live_brake_prog = 0.0
	_live_dist_to_brake_m = -1.0
	_live_alert_factor = 0.0

	if not _braking_zones.is_empty():
		var track_len := maxf(_total_track_s, 10.0)
		var s_wrap := fposmod(_live_s_m, track_len)
		var best_ahead := 1e9
		var upcoming_id := -1
		for bz in _braking_zones:
			var zid := int(bz["id"])
			var s_st := float(bz["s_start"])
			var s_en := float(bz["s_end"])
			if s_wrap >= s_st and s_wrap <= s_en:
				_live_in_brake_zone = true
				_live_active_zone_id = zid
				_live_dist_to_brake_m = 0.0
				_live_brake_prog = clampf((s_wrap - s_st) / maxf(s_en - s_st, 0.5), 0.0, 1.0)
				break
			var d_ahead := fposmod(s_st - s_wrap, track_len)
			if d_ahead < best_ahead:
				best_ahead = d_ahead
				upcoming_id = zid

		if not _live_in_brake_zone and upcoming_id >= 0:
			_live_dist_to_brake_m = best_ahead
			if best_ahead <= 45.0:
				_live_active_zone_id = upcoming_id

		var v_rel_norm := clampf((_live_v_rel_ms / 5.0) * speed_alert_gain, -0.6, 1.5)
		if _live_in_brake_zone:
			_live_alert_factor = clampf(0.55 + 0.50 * v_rel_norm + 0.35 * _live_brake_in, 0.0, 1.5)
		elif _live_dist_to_brake_m >= 0.0 and _live_dist_to_brake_m <= 45.0:
			var prox := clampf(1.0 - _live_dist_to_brake_m / 45.0, 0.0, 1.0)
			_live_alert_factor = clampf(prox * (0.45 + 0.55 * v_rel_norm), 0.0, 1.5)

	if display_mode == LineDisplayMode.SIM_RACING_3D and not _last_points.is_empty():
		_ensure_nodes()
		_rebuild_sim_racing_3d_ribbon()


func get_braking_ramp_metrics() -> Dictionary:
	return {
		"display_mode": display_mode,
		"brake_elevation_scale": brake_elevation_scale,
		"speed_alert_gain": speed_alert_gain,
		"peak_y_elevation_m": _peak_y_elevation_m,
		"max_vertex_y_m": _max_vertex_y_m,
		"active_zone_id": _live_active_zone_id,
		"active_zone_progress": _live_brake_prog,
		"alert_factor": _live_alert_factor,
		"v_rel_ms": _live_v_rel_ms,
		"dist_to_brake_m": _live_dist_to_brake_m,
		"in_braking_zone": _live_in_brake_zone,
		"braking_zone_count": _braking_zones.size(),
		"nan_count": _nan_vertex_count,
	}


func _precompute_braking_zones(samples: Array, is_closed: bool) -> void:
	_braking_zones.clear()
	var n := samples.size()
	_sample_zone_id.resize(n)
	_sample_zone_u.resize(n)
	_sample_s_m.resize(n)
	if n == 0:
		_total_track_s = 100.0
		return

	var s_accum := 0.0
	for i in range(n):
		var d: Dictionary = samples[i]
		var s_val := float(d.get("s_m", -1.0))
		if i > 0:
			var dp: Dictionary = samples[i - 1]
			var dx := float(d.get("x_godot", 0.0)) - float(dp.get("x_godot", 0.0))
			var dz := float(d.get("z_godot", 0.0)) - float(dp.get("z_godot", 0.0))
			if is_finite(dx) and is_finite(dz):
				s_accum += sqrt(dx * dx + dz * dz)
		_sample_s_m[i] = s_val if (s_val >= 0.0 and is_finite(s_val) and (i == 0 or s_val > _sample_s_m[i - 1])) else s_accum
		_sample_zone_id[i] = -1
		_sample_zone_u[i] = 0.0
	_total_track_s = maxf(_sample_s_m[n - 1], 10.0)
	if is_closed and n >= 2:
		var d_end := samples[n - 1] as Dictionary
		var d_beg := samples[0] as Dictionary
		var cdx := float(d_beg.get("x_godot", 0.0)) - float(d_end.get("x_godot", 0.0))
		var cdz := float(d_beg.get("z_godot", 0.0)) - float(d_end.get("z_godot", 0.0))
		if is_finite(cdx) and is_finite(cdz):
			_total_track_s += sqrt(cdx * cdx + cdz * cdz)

	var in_zone := false
	var z_start_idx := 0
	var zid := 0
	for i in range(n):
		var d: Dictionary = samples[i]
		var brk := float(d.get("brake", 0.0))
		var ph := String(d.get("phase", "")).strip_edges().to_upper()
		var is_brk := (brk >= 0.15 or ph == "BRAKING" or ph == "TRAIL_BRAKING")
		if is_brk and not in_zone:
			in_zone = true
			z_start_idx = i
		elif not is_brk and in_zone:
			in_zone = false
			var z_end_idx := i - 1
			if z_end_idx >= z_start_idx:
				_register_braking_zone(zid, z_start_idx, z_end_idx)
				zid += 1
	if in_zone and n - 1 >= z_start_idx:
		_register_braking_zone(zid, z_start_idx, n - 1)


func _register_braking_zone(zid: int, i_start: int, i_end: int) -> void:
	var s_st := _sample_s_m[i_start]
	var s_en := _sample_s_m[i_end]
	var z_len := maxf(s_en - s_st, 0.5)
	_braking_zones.append({
		"id": zid,
		"start_idx": i_start,
		"end_idx": i_end,
		"s_start": s_st,
		"s_end": s_en,
		"length_m": z_len,
	})
	var span_idx := maxi(i_end - i_start, 1)
	for k in range(i_start, i_end + 1):
		_sample_zone_id[k] = zid
		var u_val := (_sample_s_m[k] - s_st) / z_len if (s_en > s_st + 0.25) else float(k - i_start) / float(span_idx)
		_sample_zone_u[k] = clampf(u_val, 0.0, 1.0)


func _classify_sim_racing_3d_color(d_cur: Dictionary, zone_id: int, zone_u: float, alert_local: float) -> Color:
	var thr := clampf(float(d_cur.get("throttle", 0.0)), 0.0, 1.0)
	var brk := clampf(float(d_cur.get("brake", 0.0)), 0.0, 1.0)
	var kap := absf(float(d_cur.get("kappa", 0.0)))
	var ph := String(d_cur.get("phase", "")).strip_edges().to_upper()

	var col_emerald := Color(0.00, 0.92, 0.54, 0.93)
	var col_cyan := Color(0.08, 0.84, 0.94, 0.91)
	var col_amber := Color(1.00, 0.62, 0.06, 0.94)
	var col_red := Color(1.00, 0.06, 0.18, 0.98)

	if zone_id >= 0 or brk >= 0.15 or ph == "BRAKING" or ph == "TRAIL_BRAKING":
		var red_t := clampf((brk - 0.12) / 0.35, 0.25, 1.0)
		if ph == "BRAKING":
			red_t = maxf(red_t, 0.85)
		var base_c := col_amber.lerp(col_red, red_t)
		if zone_id >= 0 and zone_id == _live_active_zone_id:
			var pulse := 0.5 + 0.5 * sin(_pulse_phase_rad - zone_u * 4.0)
			if _live_v_rel_ms >= 0.0:
				var hot := Color(1.00, 0.18, 0.08, 1.00)
				base_c = base_c.lerp(hot, clampf(alert_local * (0.35 + 0.35 * pulse), 0.0, 0.85))
			else:
				var calm := Color(0.84, 0.14, 0.22, 0.90)
				base_c = base_c.lerp(calm, clampf(-_live_v_rel_ms / 8.0, 0.0, 0.45))
		return base_c

	if kap >= 0.055 or ph == "PARTIAL_THROTTLE":
		var apex_t := clampf((kap - 0.04) / 0.08, 0.0, 1.0)
		return col_cyan.lerp(col_amber, maxf(apex_t, 0.45 if ph == "PARTIAL_THROTTLE" else 0.0))

	if thr >= 0.55 or ph == "FULL_THROTTLE":
		return col_cyan.lerp(col_emerald, clampf((thr - 0.45) / 0.50, 0.35, 1.0))

	return col_cyan


func _rebuild_sim_racing_3d_ribbon() -> void:
	var n := _last_points.size()
	if n < 2:
		_peak_y_elevation_m = 0.0
		_max_vertex_y_m = RIBBON_Y_M
		_nan_vertex_count = 0
		return
	if _dynamic_ribbon_mesh == null:
		_dynamic_ribbon_mesh = ArrayMesh.new()
	else:
		_dynamic_ribbon_mesh.clear_surfaces()

	var verts := PackedVector3Array()
	var cols := PackedColorArray()
	var norms := PackedVector3Array()
	var indices := PackedInt32Array()

	var peak_elev := 0.0
	var max_y := RIBBON_Y_M
	var nan_cnt := 0
	var elev_y_per_pt := PackedFloat32Array()
	elev_y_per_pt.resize(n)
	var px_arr := PackedFloat32Array()
	var pz_arr := PackedFloat32Array()
	var nx_arr := PackedFloat32Array()
	var nz_arr := PackedFloat32Array()
	var hw_arr := PackedFloat32Array()
	px_arr.resize(n); pz_arr.resize(n); nx_arr.resize(n); nz_arr.resize(n); hw_arr.resize(n)

	var track_len := maxf(_total_track_s, 10.0)
	var active_zone_s_start := -1.0
	if _live_active_zone_id >= 0 and _live_active_zone_id < _braking_zones.size():
		active_zone_s_start = float(_braking_zones[_live_active_zone_id]["s_start"])

	for i in range(n):
		var d_cur: Dictionary = _last_points[i]
		var px := float(d_cur.get("x_godot", 0.0))
		var pz := float(d_cur.get("z_godot", 0.0))
		if not is_finite(px) or not is_finite(pz):
			px = 0.0
			pz = 0.0
		var i_prev := (i - 1 + n) % n if _last_is_closed else maxi(0, i - 1)
		var i_next := (i + 1) % n if _last_is_closed else mini(n - 1, i + 1)
		var d_prev: Dictionary = _last_points[i_prev]
		var d_next: Dictionary = _last_points[i_next]
		var tx := float(d_next.get("x_godot", px)) - float(d_prev.get("x_godot", px))
		var tz := float(d_next.get("z_godot", pz)) - float(d_prev.get("z_godot", pz))
		var t_len := sqrt(tx * tx + tz * tz)
		if t_len < 1e-5:
			tx = 0.0
			tz = -1.0
		else:
			tx /= t_len
			tz /= t_len
		var nx := -tz
		var nz := tx

		var zid := _sample_zone_id[i] if i < _sample_zone_id.size() else -1
		var zu := _sample_zone_u[i] if i < _sample_zone_u.size() else 0.0
		var s_i := _sample_s_m[i] if i < _sample_s_m.size() else 0.0
		var brk := clampf(float(d_cur.get("brake", 0.0)), 0.0, 1.0)

		var alert_local := 0.0
		if zid >= 0 and zid == _live_active_zone_id:
			alert_local = _live_alert_factor
		elif active_zone_s_start >= 0.0:
			var dist_before_zone := fposmod(active_zone_s_start - s_i, track_len)
			if dist_before_zone > 0.0 and dist_before_zone <= 22.0:
				alert_local = _live_alert_factor * clampf(1.0 - dist_before_zone / 22.0, 0.0, 1.0)

		var elev_y := 0.0
		if zid >= 0 and brake_elevation_scale > 0.0:
			# Smooth 3D ramp rising progressively into turn-in and tapering cleanly at exit
			var ramp_shape := sin(PI * pow(clampf(zu, 0.0, 1.0), 0.72))
			var brk_intensity := clampf(brk, 0.40, 1.0)
			if zid == _live_active_zone_id:
				var drive_mod := clampf(
					0.35 + 0.45 * _live_brake_in + 0.35 * _live_brake_prog + 0.25 * maxf(_live_alert_factor, 0.0),
					0.30,
					1.35
				)
				elev_y = clampf(brake_elevation_scale * drive_mod * ramp_shape, 0.0, 1.5)
			else:
				elev_y = clampf(brake_elevation_scale * 0.28 * brk_intensity * ramp_shape, 0.0, 1.5)

		var half_w := RIBBON_HALF_WIDTH_M * (1.0 + 0.32 * clampf(alert_local, 0.0, 1.2))
		var y_v := RIBBON_Y_M + elev_y
		if not is_finite(y_v):
			y_v = RIBBON_Y_M
			elev_y = 0.0
			nan_cnt += 1

		peak_elev = maxf(peak_elev, elev_y)
		max_y = maxf(max_y, y_v)
		elev_y_per_pt[i] = elev_y
		px_arr[i] = px; pz_arr[i] = pz; nx_arr[i] = nx; nz_arr[i] = nz; hw_arr[i] = half_w

		var col := _classify_sim_racing_3d_color(d_cur, zid, zu, alert_local)
		if alert_local > 0.05 and zid < 0 and _live_v_rel_ms > 0.5:
			col = col.lerp(COL_TRAIL_AMBER, clampf(alert_local * 0.45, 0.0, 0.55))

		verts.append(Vector3(px - nx * half_w, y_v, pz - nz * half_w))
		verts.append(Vector3(px + nx * half_w, y_v, pz + nz * half_w))
		cols.append(col)
		cols.append(col)
		norms.append(Vector3.UP)
		norms.append(Vector3.UP)

	var seg_end := n if _last_is_closed else (n - 1)
	for i in range(seg_end):
		var j := (i + 1) % n
		var i0 := 2 * i
		var i1 := 2 * i + 1
		var j0 := 2 * j
		var j1 := 2 * j + 1
		indices.append(i0); indices.append(j0); indices.append(i1)
		indices.append(i1); indices.append(j0); indices.append(j1)

	# Translucent vertical side-curtains under elevated red braking ramps for clear 3D depth perception
	var wall_col_top := Color(1.00, 0.08, 0.20, 0.52)
	var wall_col_bot := Color(1.00, 0.08, 0.20, 0.14)
	for i in range(seg_end):
		var j := (i + 1) % n
		if elev_y_per_pt[i] <= 0.015 and elev_y_per_pt[j] <= 0.015:
			continue
		for side_sign in [-1.0, 1.0]:
			var p0_top := Vector3(px_arr[i] + nx_arr[i] * (side_sign * hw_arr[i]), RIBBON_Y_M + elev_y_per_pt[i], pz_arr[i] + nz_arr[i] * (side_sign * hw_arr[i]))
			var p1_top := Vector3(px_arr[j] + nx_arr[j] * (side_sign * hw_arr[j]), RIBBON_Y_M + elev_y_per_pt[j], pz_arr[j] + nz_arr[j] * (side_sign * hw_arr[j]))
			var p1_bot := Vector3(p1_top.x, RIBBON_Y_M, p1_top.z)
			var p0_bot := Vector3(p0_top.x, RIBBON_Y_M, p0_top.z)
			var base := verts.size()
			verts.append(p0_top); verts.append(p1_top); verts.append(p1_bot); verts.append(p0_bot)
			cols.append(wall_col_top); cols.append(wall_col_top); cols.append(wall_col_bot); cols.append(wall_col_bot)
			var w_norm := Vector3(nx_arr[i] * side_sign, 0.0, nz_arr[i] * side_sign)
			norms.append(w_norm); norms.append(w_norm); norms.append(w_norm); norms.append(w_norm)
			indices.append(base); indices.append(base + 1); indices.append(base + 2)
			indices.append(base); indices.append(base + 2); indices.append(base + 3)

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = norms
	arrays[Mesh.ARRAY_COLOR] = cols
	arrays[Mesh.ARRAY_INDEX] = indices
	_dynamic_ribbon_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_assign_mesh_custom_aabb(_dynamic_ribbon_mesh, verts)
	_assign_mesh_to_instance(_line_mesh_inst, _dynamic_ribbon_mesh)
	_peak_y_elevation_m = peak_elev
	_max_vertex_y_m = max_y
	_nan_vertex_count = nan_cnt


func clear_live_driver_speeds() -> void:
	_live_driver_speeds.clear()
	_live_delta_dirty = true


func update_live_driver_speed(s_m: float, v_driver_mps: float, dt: float = 0.05) -> void:
	if _last_points.is_empty() or not is_finite(s_m) or not is_finite(v_driver_mps):
		return
	var best_idx := -1
	var best_ds := 1e9
	for i in range(_last_points.size()):
		var pt: Dictionary = _last_points[i]
		var ds := absf(float(pt.get("s_m", float(i) * 2.5)) - s_m)
		if ds < best_ds:
			best_ds = ds
			best_idx = i
	if best_idx >= 0 and best_ds <= 8.0:
		var prev_v: float = float(_live_driver_speeds.get(best_idx, -999.0))
		if absf(prev_v - v_driver_mps) > 0.15:
			_live_driver_speeds[best_idx] = v_driver_mps
			_live_delta_dirty = true
	_live_delta_accum += maxf(dt, 0.0)
	if color_mode == ColorMode.LIVE_DELTA_V and _live_delta_dirty and _live_delta_accum >= 0.12:
		_live_delta_accum = 0.0
		_live_delta_dirty = false
		_ensure_nodes()
		_assign_mesh_to_instance(
			_line_mesh_inst,
			build_ribbon_array_mesh(_last_points, _last_is_closed, color_mode, _live_driver_speeds)
		)


## 4-state motorsport palette without flicker:
##   - Hard Braking (brake >= 0.35): Intense Red (#FF1744)
##   - Trail-Braking (0.15 <= brake < 0.35): Amber / Orange (#FF9100)
##   - Coasting / Maintenance (throttle < 0.65 and brake < 0.15): Cyan / Light Blue (#29B6F6)
##   - Acceleration (throttle >= 0.65): Bright Green (#00E676)
static func classify_zone_color(throttle: float, brake: float, _ax_mps2: float = 0.0) -> Color:
	var thr := clampf(throttle if is_finite(throttle) else 0.0, 0.0, 1.0)
	var brk := clampf(brake if is_finite(brake) else 0.0, 0.0, 1.0)

	if brk >= 0.35:
		return COL_BRAKE_RED
	if brk >= 0.15:
		return COL_TRAIL_AMBER
	if thr >= 0.65:
		return COL_WOT_GREEN
	return COL_COAST_CYAN


static func classify_phase_color(phase: String, throttle: float = 0.0, brake: float = 0.0, ax_mps2: float = 0.0) -> Color:
	match phase.strip_edges().to_upper():
		"FULL_THROTTLE":
			return COL_WOT_GREEN
		"PARTIAL_THROTTLE":
			return COL_PARTIAL_LIME
		"LIFT_AND_COAST":
			return COL_COAST_CYAN
		"TRAIL_BRAKING":
			return COL_TRAIL_AMBER
		"BRAKING":
			return COL_BRAKE_RED
		_:
			return classify_zone_color(throttle, brake, ax_mps2)


static func classify_velocity_color(v_mps: float, v_min: float, v_max: float) -> Color:
	var span := maxf(v_max - v_min, 1.0)
	var u := clampf((v_mps - v_min) / span, 0.0, 1.0)
	if u < 0.33:
		return Color(0.12, 0.42, 0.96, 0.92).lerp(Color(0.08, 0.86, 0.88, 0.92), u / 0.33)
	elif u < 0.66:
		return Color(0.08, 0.86, 0.88, 0.92).lerp(Color(0.18, 0.92, 0.42, 0.92), (u - 0.33) / 0.33)
	else:
		return Color(0.18, 0.92, 0.42, 0.92).lerp(Color(1.00, 0.82, 0.14, 0.94), (u - 0.66) / 0.34)


static func classify_delta_v_color(delta_v_mps: float) -> Color:
	if not is_finite(delta_v_mps):
		return Color(0.55, 0.62, 0.72, 0.80)
	var u := clampf(delta_v_mps / 3.5, -1.0, 1.0)
	if u >= 0.0:
		return Color(0.65, 0.76, 0.82, 0.86).lerp(Color(0.04, 0.92, 0.44, 0.96), u)
	return Color(0.65, 0.76, 0.82, 0.86).lerp(Color(0.98, 0.14, 0.24, 0.96), -u)


static func _build_strip_mesh(
	samples: Array,
	half_width: float,
	y_height: float,
	is_closed: bool,
	use_zone_colors: bool,
	fixed_color: Color = COL_BORDER,
	ribbon_color_mode: int = ColorMode.DRIVING_PHASES,
	live_driver_speeds: Dictionary = {}
) -> ArrayMesh:
	var n := samples.size()
	var mesh := ArrayMesh.new()
	if n < 2:
		return mesh

	var v_min := INF
	var v_max := -INF
	if use_zone_colors and ribbon_color_mode == ColorMode.VELOCITY:
		for s_v in samples:
			var sv := float((s_v as Dictionary).get("v_ref", (s_v as Dictionary).get("v_mps", 12.0)))
			if is_finite(sv):
				v_min = minf(v_min, sv)
				v_max = maxf(v_max, sv)
		if not is_finite(v_min) or v_max <= v_min + 0.2:
			v_min = 6.0
			v_max = 32.0

	var verts := PackedVector3Array()
	var cols := PackedColorArray()
	var norms := PackedVector3Array()
	var indices := PackedInt32Array()

	for i in range(n):
		var d_cur: Dictionary = samples[i]
		var px := float(d_cur.get("x_godot", 0.0))
		var pz := float(d_cur.get("z_godot", 0.0))
		if not is_finite(px) or not is_finite(pz):
			px = 0.0
			pz = 0.0
		var i_prev := (i - 1 + n) % n if is_closed else maxi(0, i - 1)
		var i_next := (i + 1) % n if is_closed else mini(n - 1, i + 1)
		var d_prev: Dictionary = samples[i_prev]
		var d_next: Dictionary = samples[i_next]
		var tx := float(d_next.get("x_godot", px)) - float(d_prev.get("x_godot", px))
		var tz := float(d_next.get("z_godot", pz)) - float(d_prev.get("z_godot", pz))
		var t_len := sqrt(tx * tx + tz * tz)
		if t_len < 1e-5:
			tx = 0.0
			tz = -1.0
		else:
			tx /= t_len
			tz /= t_len
		var nx := -tz
		var nz := tx
		var col := fixed_color
		if use_zone_colors:
			if ribbon_color_mode == ColorMode.VELOCITY:
				var v_cur := float(d_cur.get("v_ref", d_cur.get("v_mps", 12.0)))
				col = classify_velocity_color(v_cur, v_min, v_max)
			elif ribbon_color_mode == ColorMode.LIVE_DELTA_V:
				var v_ref := float(d_cur.get("v_ref", d_cur.get("v_mps", 12.0)))
				if live_driver_speeds.has(i):
					col = classify_delta_v_color(float(live_driver_speeds[i]) - v_ref)
				else:
					col = Color(0.50, 0.58, 0.68, 0.78)
			else:
				var phase_str := String(d_cur.get("phase", ""))
				if not phase_str.is_empty():
					col = classify_phase_color(
						phase_str,
						float(d_cur.get("throttle", 0.0)),
						float(d_cur.get("brake", 0.0)),
						float(d_cur.get("ax", d_cur.get("ax_mps2", 0.0)))
					)
				else:
					col = classify_zone_color(
						float(d_cur.get("throttle", 0.0)),
						float(d_cur.get("brake", 0.0)),
						float(d_cur.get("ax", d_cur.get("ax_mps2", 0.0)))
					)
		verts.append(Vector3(px - nx * half_width, y_height, pz - nz * half_width))
		verts.append(Vector3(px + nx * half_width, y_height, pz + nz * half_width))
		cols.append(col)
		cols.append(col)
		norms.append(Vector3.UP)
		norms.append(Vector3.UP)

	var seg_end := n if is_closed else (n - 1)
	for i in range(seg_end):
		var j := (i + 1) % n
		var i0 := 2 * i
		var i1 := 2 * i + 1
		var j0 := 2 * j
		var j1 := 2 * j + 1
		indices.append(i0); indices.append(j0); indices.append(i1)
		indices.append(i1); indices.append(j0); indices.append(j1)

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = norms
	arrays[Mesh.ARRAY_COLOR] = cols
	arrays[Mesh.ARRAY_INDEX] = indices
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_assign_mesh_custom_aabb(mesh, verts)
	return mesh


static func build_ribbon_array_mesh(
	samples: Array,
	is_closed: bool = true,
	ribbon_color_mode: int = ColorMode.DRIVING_PHASES,
	live_driver_speeds: Dictionary = {}
) -> ArrayMesh:
	return _build_strip_mesh(samples, RIBBON_HALF_WIDTH_M, RIBBON_Y_M, is_closed, true, COL_BORDER, ribbon_color_mode, live_driver_speeds)


static func build_border_array_mesh(samples: Array, is_closed: bool = true) -> ArrayMesh:
	return _build_strip_mesh(samples, BORDER_HALF_WIDTH_M, BORDER_Y_M, is_closed, false, COL_BORDER)


static func _append_quad(
	verts: PackedVector3Array,
	cols: PackedColorArray,
	norms: PackedVector3Array,
	indices: PackedInt32Array,
	p0: Vector3,
	p1: Vector3,
	p2: Vector3,
	p3: Vector3,
	col: Color
) -> void:
	var base := verts.size()
	verts.append(p0); verts.append(p1); verts.append(p2); verts.append(p3)
	for _k in range(4):
		cols.append(col)
		norms.append(Vector3.UP)
	indices.append(base); indices.append(base + 1); indices.append(base + 2)
	indices.append(base); indices.append(base + 2); indices.append(base + 3)


static func build_markers_array_mesh(samples: Array, is_closed: bool = true, explicit_markers: Array = []) -> ArrayMesh:
	var n := samples.size()
	var mesh := ArrayMesh.new()
	if n < 3:
		return mesh

	var verts := PackedVector3Array()
	var cols := PackedColorArray()
	var norms := PackedVector3Array()
	var indices := PackedInt32Array()

	var dist_accum := 0.0
	var next_chevron_s := 4.0
	var has_explicit := not explicit_markers.is_empty()

	for i in range(n):
		var d_cur: Dictionary = samples[i]
		var px := float(d_cur.get("x_godot", 0.0))
		var pz := float(d_cur.get("z_godot", 0.0))
		if not is_finite(px) or not is_finite(pz):
			continue
		var i_prev := (i - 1 + n) % n if is_closed else maxi(0, i - 1)
		var i_next := (i + 1) % n if is_closed else mini(n - 1, i + 1)
		var d_prev: Dictionary = samples[i_prev]
		var d_next: Dictionary = samples[i_next]
		var tx := float(d_next.get("x_godot", px)) - float(d_prev.get("x_godot", px))
		var tz := float(d_next.get("z_godot", pz)) - float(d_prev.get("z_godot", pz))
		var t_len := sqrt(tx * tx + tz * tz)
		if t_len < 1e-5:
			continue
		tx /= t_len
		tz /= t_len
		# Left-right normal in XZ plane
		var nx := -tz
		var nz := tx

		if i > 0:
			var dx_step := px - float(d_prev.get("x_godot", px))
			var dz_step := pz - float(d_prev.get("z_godot", pz))
			dist_accum += sqrt(dx_step * dx_step + dz_step * dz_step)

		var brk := float(d_cur.get("brake", 0.0))
		var is_braking := (brk >= 0.15)
		var chev_spacing := 4.8 if is_braking else 8.0

		# 1. Directional chevron (>>>) pointing along travel tangent (tx, tz)
		if dist_accum >= next_chevron_s:
			next_chevron_s = dist_accum + chev_spacing
			var c_col := COL_CHEVRON_BRK if is_braking else COL_CHEVRON_STD
			var tip_x := px + tx * 0.20
			var tip_z := pz + tz * 0.20
			var back_x := px - tx * 0.14
			var back_z := pz - tz * 0.14
			var wing := 0.155
			var thick := 0.055
			# Left arm of chevron
			_append_quad(
				verts, cols, norms, indices,
				Vector3(tip_x, CHEVRON_Y_M, tip_z),
				Vector3(tip_x - tx * thick, CHEVRON_Y_M, tip_z - tz * thick),
				Vector3(back_x - tx * thick - nx * wing, CHEVRON_Y_M, back_z - tz * thick - nz * wing),
				Vector3(back_x - nx * wing, CHEVRON_Y_M, back_z - nz * wing),
				c_col
			)
			# Right arm of chevron
			_append_quad(
				verts, cols, norms, indices,
				Vector3(tip_x, CHEVRON_Y_M, tip_z),
				Vector3(back_x + nx * wing, CHEVRON_Y_M, back_z + nz * wing),
				Vector3(back_x - tx * thick + nx * wing, CHEVRON_Y_M, back_z - tz * thick + nz * wing),
				Vector3(tip_x - tx * thick, CHEVRON_Y_M, tip_z - tz * thick),
				c_col
			)

		if not has_explicit:
			# 2. Brake Marker: crisp transverse red bar (1.6 m wide x 0.25 m long)
			if bool(d_cur.get("is_brake_point", false)):
				var hw := 0.80
				var hl := 0.125
				_append_quad(
					verts, cols, norms, indices,
					Vector3(px - tx * hl - nx * hw, MARKER_Y_M, pz - tz * hl - nz * hw),
					Vector3(px - tx * hl + nx * hw, MARKER_Y_M, pz - tz * hl + nz * hw),
					Vector3(px + tx * hl + nx * hw, MARKER_Y_M, pz + tz * hl + nz * hw),
					Vector3(px + tx * hl - nx * hw, MARKER_Y_M, pz + tz * hl - nz * hw),
					COL_BRAKE_RED
				)

			# 3. Apex Diamond on inside edge of ribbon (bright cyan outer + white core)
			if bool(d_cur.get("is_apex", false)):
				var kap := float(d_cur.get("kappa", 0.0))
				var side_sign := -1.0 if kap >= 0.0 else 1.0
				var cx := px + nx * (side_sign * (RIBBON_HALF_WIDTH_M + 0.14))
				var cz := pz + nz * (side_sign * (RIBBON_HALF_WIDTH_M + 0.14))
				var r_out := 0.24
				_append_quad(
					verts, cols, norms, indices,
					Vector3(cx + tx * r_out, MARKER_Y_M + 0.002, cz + tz * r_out),
					Vector3(cx + nx * r_out, MARKER_Y_M + 0.002, cz + nz * r_out),
					Vector3(cx - tx * r_out, MARKER_Y_M + 0.002, cz - tz * r_out),
					Vector3(cx - nx * r_out, MARKER_Y_M + 0.002, cz - nz * r_out),
					COL_APEX_CYAN
				)
				var r_in := 0.11
				_append_quad(
					verts, cols, norms, indices,
					Vector3(cx + tx * r_in, MARKER_Y_M + 0.004, cz + tz * r_in),
					Vector3(cx + nx * r_in, MARKER_Y_M + 0.004, cz + nz * r_in),
					Vector3(cx - tx * r_in, MARKER_Y_M + 0.004, cz - tz * r_in),
					Vector3(cx - nx * r_in, MARKER_Y_M + 0.004, cz - nz * r_in),
					Color(1.0, 1.0, 1.0, 0.98)
				)

			# 4. Exit-to-Full-Gas Marker: crisp green transverse bar (1.2 m wide x 0.20 m long)
			var is_exit := bool(d_cur.get("is_exit_point", false))
			if not is_exit and i > 0:
				var thr_now := float(d_cur.get("throttle", 0.0))
				var thr_prv := float(d_prev.get("throttle", 0.0))
				is_exit = (thr_now >= 0.65 and thr_prv < 0.65 and brk < 0.15)
			if is_exit:
				var hw_e := 0.60
				var hl_e := 0.10
				_append_quad(
					verts, cols, norms, indices,
					Vector3(px - tx * hl_e - nx * hw_e, MARKER_Y_M + 0.001, pz - tz * hl_e - nz * hw_e),
					Vector3(px - tx * hl_e + nx * hw_e, MARKER_Y_M + 0.001, pz - tz * hl_e + nz * hw_e),
					Vector3(px + tx * hl_e + nx * hw_e, MARKER_Y_M + 0.001, pz + tz * hl_e + nz * hw_e),
					Vector3(px + tx * hl_e - nx * hw_e, MARKER_Y_M + 0.001, pz + tz * hl_e - nz * hw_e),
					COL_EXIT_GREEN
				)

	if has_explicit:
		for m_v in explicit_markers:
			if not (m_v is Dictionary):
				continue
			var m: Dictionary = m_v
			var m_type := String(m.get("type", "")).strip_edges().to_upper()
			var mx := -float(m.get("y", 0.0))
			var mz := -float(m.get("x", 0.0))
			if not is_finite(mx) or not is_finite(mz):
				continue
			# Find nearest sample tangent
			var best_i := 0
			var best_d2 := 1e18
			for si in range(n):
				var sp: Dictionary = samples[si]
				var dx := float(sp.get("x_godot", 0.0)) - mx
				var dz := float(sp.get("z_godot", 0.0)) - mz
				var d2 := dx * dx + dz * dz
				if d2 < best_d2:
					best_d2 = d2
					best_i = si
			var sp_prev: Dictionary = samples[(best_i - 1 + n) % n if is_closed else maxi(0, best_i - 1)]
			var sp_next: Dictionary = samples[(best_i + 1) % n if is_closed else mini(n - 1, best_i + 1)]
			var mtx := float(sp_next.get("x_godot", mx)) - float(sp_prev.get("x_godot", mx))
			var mtz := float(sp_next.get("z_godot", mz)) - float(sp_prev.get("z_godot", mz))
			var mt_len := sqrt(mtx * mtx + mtz * mtz)
			if mt_len < 1e-5:
				mtx = 0.0; mtz = -1.0
			else:
				mtx /= mt_len; mtz /= mt_len
			var mnx := -mtz
			var mnz := mtx

			if m_type == "BRAKE_POINT":
				var hw := 0.85
				var hl := 0.14
				_append_quad(
					verts, cols, norms, indices,
					Vector3(mx - mtx * hl - mnx * hw, MARKER_Y_M + 0.002, mz - mtz * hl - mnz * hw),
					Vector3(mx - mtx * hl + mnx * hw, MARKER_Y_M + 0.002, mz - mtz * hl + mnz * hw),
					Vector3(mx + mtx * hl + mnx * hw, MARKER_Y_M + 0.002, mz + mtz * hl + mnz * hw),
					Vector3(mx + mtx * hl - mnx * hw, MARKER_Y_M + 0.002, mz + mtz * hl - mnz * hw),
					COL_BRAKE_RED
				)
			elif m_type == "LIFT_COAST_START":
				var hw := 0.78
				var hl := 0.12
				_append_quad(
					verts, cols, norms, indices,
					Vector3(mx - mtx * hl - mnx * hw, MARKER_Y_M + 0.002, mz - mtz * hl - mnz * hw),
					Vector3(mx - mtx * hl + mnx * hw, MARKER_Y_M + 0.002, mz - mtz * hl + mnz * hw),
					Vector3(mx + mtx * hl + mnx * hw, MARKER_Y_M + 0.002, mz + mtz * hl + mnz * hw),
					Vector3(mx + mtx * hl - mnx * hw, MARKER_Y_M + 0.002, mz + mtz * hl - mnz * hw),
					COL_LIFT_BLUE
				)
			elif m_type == "APEX":
				var r_out := 0.28
				_append_quad(
					verts, cols, norms, indices,
					Vector3(mx + mtx * r_out, MARKER_Y_M + 0.003, mz + mtz * r_out),
					Vector3(mx + mnx * r_out, MARKER_Y_M + 0.003, mz + mnz * r_out),
					Vector3(mx - mtx * r_out, MARKER_Y_M + 0.003, mz - mtz * r_out),
					Vector3(mx - mnx * r_out, MARKER_Y_M + 0.003, mz - mnz * r_out),
					COL_APEX_CYAN
				)
				var r_in := 0.13
				_append_quad(
					verts, cols, norms, indices,
					Vector3(mx + mtx * r_in, MARKER_Y_M + 0.005, mz + mtz * r_in),
					Vector3(mx + mnx * r_in, MARKER_Y_M + 0.005, mz + mnz * r_in),
					Vector3(mx - mtx * r_in, MARKER_Y_M + 0.005, mz - mtz * r_in),
					Vector3(mx - mnx * r_in, MARKER_Y_M + 0.005, mz - mnz * r_in),
					Color(1.0, 1.0, 1.0, 0.98)
				)
			elif m_type == "THROTTLE_ON":
				var hw_e := 0.65
				var hl_e := 0.10
				_append_quad(
					verts, cols, norms, indices,
					Vector3(mx - mtx * hl_e - mnx * hw_e, MARKER_Y_M + 0.002, mz - mtz * hl_e - mnz * hw_e),
					Vector3(mx - mtx * hl_e + mnx * hw_e, MARKER_Y_M + 0.002, mz - mtz * hl_e + mnz * hw_e),
					Vector3(mx + mtx * hl_e + mnx * hw_e, MARKER_Y_M + 0.002, mz + mtz * hl_e + mnz * hw_e),
					Vector3(mx + mtx * hl_e - mnx * hw_e, MARKER_Y_M + 0.002, mz + mtz * hl_e - mnz * hw_e),
					COL_EXIT_GREEN
				)

	if not verts.is_empty():
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = verts
		arrays[Mesh.ARRAY_NORMAL] = norms
		arrays[Mesh.ARRAY_COLOR] = cols
		arrays[Mesh.ARRAY_INDEX] = indices
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_assign_mesh_custom_aabb(mesh, verts)
	return mesh


func load_casadi_line_json(json_path: String) -> Dictionary:
	if json_path.is_empty() or not FileAccess.file_exists(json_path):
		return {}
	var f := FileAccess.open(json_path, FileAccess.READ)
	if f == null:
		return {}
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY:
		return {}
	var d: Dictionary = parsed
	var raw_pts: Variant = d.get("waypoints", d.get("points", []))
	if not (raw_pts is Array) or (raw_pts as Array).is_empty():
		return {}
	var samples := _convert_raw_points(raw_pts as Array)
	return {
		"meta": d.get("meta", {}) if d.get("meta") is Dictionary else {},
		"waypoints": raw_pts,
		"corners": d.get("corners", []) if d.get("corners") is Array else [],
		"markers": d.get("markers", []) if d.get("markers") is Array else [],
		"samples": samples,
	}


static func _convert_raw_points(raw_pts: Array) -> Array:
	var out: Array = []
	for p_v in raw_pts:
		if p_v is Dictionary:
			var p: Dictionary = p_v
			var xg := 0.0
			var zg := 0.0
			if p.has("z") and not p.has("y"):
				xg = float(p.get("x", 0.0))
				zg = float(p.get("z", 0.0))
			else:
				var x_ros := float(p.get("x", 0.0))
				var y_ros := float(p.get("y", 0.0))
				xg = -y_ros
				zg = -x_ros
			var v_val := float(p.get("v_ref", p.get("v_mps", 12.0)))
			var ax_val := float(p.get("ax_mps2", p.get("ax", float(p.get("ax_g", 0.0)) * 9.81)))
			var ay_val := float(p.get("ay_mps2", p.get("ay", float(p.get("ay_g", 0.0)) * 9.81)))
			out.append({
				"x_godot": xg,
				"z_godot": zg,
				"throttle": float(p.get("throttle", 0.5)),
				"brake": float(p.get("brake", 0.0)),
				"ax": ax_val,
				"ay": ay_val,
				"v_mps": v_val,
				"v_ref": v_val,
				"v_ocp": float(p.get("v_ocp", v_val)),
				"s_m": float(p.get("s", p.get("s_m", 0.0))),
				"kappa": float(p.get("kappa", 0.0)),
				"grip_util": float(p.get("grip_util", 0.0)),
				"phase": String(p.get("phase", "")),
				"p_dc_kw": float(p.get("p_dc_kw", 0.0)),
				"is_brake_point": bool(p.get("is_brake_point", false)),
				"is_apex": bool(p.get("is_apex", false)),
				"is_exit_point": bool(p.get("is_exit_point", false)),
			})
	return out


func load_racing_line_json(json_path: String) -> Array:
	var data := load_casadi_line_json(json_path)
	if data.is_empty():
		return []
	last_casadi_meta = data.get("meta", {})
	last_casadi_corners = data.get("corners", [])
	last_casadi_markers = data.get("markers", [])
	return data.get("samples", [])


func apply_casadi_json(json_path: String, is_closed: bool = true) -> bool:
	var data := load_casadi_line_json(json_path)
	if data.is_empty():
		return false
	last_casadi_meta = data.get("meta", {})
	last_casadi_corners = data.get("corners", [])
	last_casadi_markers = data.get("markers", [])
	var samples: Array = data.get("samples", [])
	if samples.is_empty():
		return false
	_ensure_nodes()
	_live_driver_speeds.clear()
	_apply_samples_meshes(samples, is_closed)
	return true


func build_from_track(track_node: Node, track_path: String = "") -> bool:
	_ensure_nodes()
	var samples: Array = []
	var is_closed := true

	var stem := track_path.get_file().get_basename().to_lower() if track_path != "" else ""
	if stem != "":
		var prebaked := "res://Assets/RacingLines/%s_racing_line.json" % stem
		if FileAccess.file_exists(prebaked):
			samples = load_racing_line_json(prebaked)

	if samples.is_empty() and track_node != null and "track_curve" in track_node:
		var curve: Curve3D = track_node.get("track_curve")
		if curve != null and curve.get_baked_length() > 1.0:
			is_closed = curve.closed
			var total_l := curve.get_baked_length()
			var n_pts := clampi(int(total_l / 2.2), 20, 480)
			var ds := total_l / float(n_pts)
			var pts_pos: Array[Vector3] = []
			for i in range(n_pts):
				pts_pos.append(curve.sample_baked(float(i) * ds, false))
			# Compute centerline signed curvature & apply smooth outer-inner-outer offset
			var k_raw: Array[float] = []
			k_raw.resize(n_pts)
			var track_w := float(track_node.get("track_width")) if ("track_width" in track_node) else 3.0
			var n_max_g := maxf(0.5 * track_w - 0.85, 0.35)
			for i in range(n_pts):
				var im := (i - 1 + n_pts) % n_pts if is_closed else maxi(0, i - 1)
				var ip := (i + 1) % n_pts if is_closed else mini(n_pts - 1, i + 1)
				var d1 := pts_pos[i] - pts_pos[im]
				var d2 := pts_pos[ip] - pts_pos[i]
				var a1 := atan2(d1.z, d1.x)
				var a2 := atan2(d2.z, d2.x)
				k_raw[i] = wrapf(a2 - a1, -PI, PI) / maxf(ds, 0.25)
			# Smooth outer-inner-outer lateral shift: hug inside on corners, swing wide on entry/exit
			var n_lat: Array[float] = []
			n_lat.resize(n_pts)
			for i in range(n_pts):
				var k_here := k_raw[i]
				var k_ahead := k_raw[(i + 5) % n_pts if is_closed else mini(n_pts - 1, i + 5)]
				var k_behind := k_raw[(i - 5 + n_pts) % n_pts if is_closed else maxi(0, i - 5)]
				var shift := clampf(k_here * 10.5 - 0.35 * (k_ahead + k_behind) * 5.0, -1.0, 1.0) * n_max_g
				n_lat[i] = shift
			for _sm in range(3):
				var tmp := n_lat.duplicate()
				for i in range(n_pts):
					var im := (i - 1 + n_pts) % n_pts if is_closed else maxi(0, i - 1)
					var ip := (i + 1) % n_pts if is_closed else mini(n_pts - 1, i + 1)
					n_lat[i] = clampf(0.25 * tmp[im] + 0.50 * tmp[i] + 0.25 * tmp[ip], -n_max_g, n_max_g)
			var opt_pts: Array[Vector3] = []
			for i in range(n_pts):
				var im := (i - 1 + n_pts) % n_pts if is_closed else maxi(0, i - 1)
				var ip := (i + 1) % n_pts if is_closed else mini(n_pts - 1, i + 1)
				var t_v := (pts_pos[ip] - pts_pos[im])
				t_v.y = 0.0
				t_v = t_v.normalized() if t_v.length() > 1e-4 else Vector3(0.0, 0.0, -1.0)
				var n_v := Vector3(-t_v.z, 0.0, t_v.x)
				opt_pts.append(pts_pos[i] + n_v * n_lat[i])

			var kappas: Array[float] = []
			kappas.resize(n_pts)
			var v_est: Array[float] = []
			v_est.resize(n_pts)
			for i in range(n_pts):
				var im := (i - 1 + n_pts) % n_pts if is_closed else maxi(0, i - 1)
				var ip := (i + 1) % n_pts if is_closed else mini(n_pts - 1, i + 1)
				var d1 := opt_pts[i] - opt_pts[im]
				var d2 := opt_pts[ip] - opt_pts[i]
				var dpsi := wrapf(atan2(d2.z, d2.x) - atan2(d1.z, d1.x), -PI, PI)
				var kap := dpsi / maxf(ds, 0.25)
				kappas[i] = kap
				v_est[i] = clampf(sqrt(16.0 / maxf(absf(kap), 0.01)), 6.0, 30.0)
			for _pass in range(2):
				for idx_f in range(n_pts):
					var nxt := (idx_f + 1) % n_pts if is_closed else mini(n_pts - 1, idx_f + 1)
					v_est[nxt] = minf(v_est[nxt], sqrt(v_est[idx_f] * v_est[idx_f] + 2.0 * 9.5 * ds))
				for idx_r in range(n_pts - 1, -1, -1):
					var nxt := (idx_r + 1) % n_pts if is_closed else mini(n_pts - 1, idx_r + 1)
					v_est[idx_r] = minf(v_est[idx_r], sqrt(v_est[nxt] * v_est[nxt] + 2.0 * 14.5 * ds))
			var prev_brk := false
			for i in range(n_pts):
				var nxt := (i + 1) % n_pts if is_closed else mini(n_pts - 1, i + 1)
				var prv := (i - 1 + n_pts) % n_pts if is_closed else maxi(0, i - 1)
				var ax_e := (v_est[nxt] * v_est[nxt] - v_est[i] * v_est[i]) / maxf(2.0 * ds, 0.2)
				var thr_e := 0.95 if ax_e > 1.5 else (0.50 if ax_e > -0.4 else 0.0)
				var brk_e := clampf(-ax_e / 14.0, 0.0, 1.0) if ax_e < -1.5 else 0.0
				var is_bp := (brk_e >= 0.30 and not prev_brk)
				prev_brk = (brk_e >= 0.22)
				var is_apx := (absf(kappas[i]) > 0.075 and v_est[i] <= v_est[prv] + 0.02 and v_est[i] <= v_est[nxt] + 0.02)
				samples.append({
					"x_godot": opt_pts[i].x,
					"z_godot": opt_pts[i].z,
					"throttle": thr_e,
					"brake": brk_e,
					"ax": ax_e,
					"v_mps": v_est[i],
					"kappa": kappas[i],
					"is_brake_point": is_bp,
					"is_apex": is_apx,
				})

	if samples.is_empty():
		_clear_line_meshes()
		_last_points.clear()
		return false

	_apply_samples_meshes(samples, is_closed)
	return true


static func _assign_mesh_to_instance(mi: MeshInstance3D, mesh: ArrayMesh) -> void:
	if mi == null:
		return
	mi.mesh = mesh
	mi.custom_aabb = mesh.custom_aabb if (mesh != null and mesh.custom_aabb.has_volume()) else AABB()
	_configure_mesh_instance(mi)


func _clear_line_meshes() -> void:
	_assign_mesh_to_instance(_border_mesh_inst, null)
	_assign_mesh_to_instance(_line_mesh_inst, null)
	_assign_mesh_to_instance(_markers_mesh_inst, null)


func _apply_samples_meshes(samples: Array, is_closed: bool) -> void:
	_last_points = samples
	_last_is_closed = is_closed
	_precompute_braking_zones(samples, is_closed)
	_assign_mesh_to_instance(_border_mesh_inst, build_border_array_mesh(samples, is_closed))
	if display_mode == LineDisplayMode.SIM_RACING_3D:
		_rebuild_sim_racing_3d_ribbon()
	else:
		_peak_y_elevation_m = 0.0
		_max_vertex_y_m = RIBBON_Y_M
		_nan_vertex_count = 0
		_assign_mesh_to_instance(_line_mesh_inst, build_ribbon_array_mesh(samples, is_closed, color_mode, _live_driver_speeds))
	_assign_mesh_to_instance(_markers_mesh_inst, build_markers_array_mesh(samples, is_closed, last_casadi_markers))
	_markers_mesh_inst.visible = show_markers


func update_from_samples(samples: Array, is_closed: bool = true) -> void:
	_ensure_nodes()
	if samples.is_empty():
		_clear_line_meshes()
		_last_points.clear()
		return
	_apply_samples_meshes(samples, is_closed)


func update_bump_markers(markers: Array) -> void:
	_ensure_nodes()
	if markers.is_empty():
		_assign_mesh_to_instance(_bump_mesh_inst, null)
		return
	var mesh := ArrayMesh.new()
	var verts := PackedVector3Array()
	var cols := PackedColorArray()
	var norms := PackedVector3Array()
	var indices := PackedInt32Array()
	for m_v in markers:
		if not (m_v is Dictionary):
			continue
		var m: Dictionary = m_v
		var xr := float(m.get("x_ros", 0.0))
		var yr := float(m.get("y_ros", 0.0))
		var tx := float(m.get("tx", 1.0))
		var ty := float(m.get("ty", 0.0))
		var lat_off := float(m.get("lat_off", 0.0))
		var hw := minf(float(m.get("half_w", 1.6)), 1.8)
		var hl := clampf(float(m.get("len_m", 0.8)) * 0.45, 0.18, 1.2)
		var cx_r := xr - ty * lat_off
		var cy_r := yr + tx * lat_off
		var cx := -cy_r
		var cz := -cx_r
		var gx_t := -ty
		var gz_t := -tx
		var gx_n := -tx
		var gz_n := ty
		var b_col := Color(0.98, 0.58, 0.12, 0.72) if float(m.get("amp_m", 0.02)) >= 0.028 else Color(0.95, 0.82, 0.22, 0.65)
		_append_quad(
			verts, cols, norms, indices,
			Vector3(cx - gx_t * hl - gx_n * hw, 0.032, cz - gz_t * hl - gz_n * hw),
			Vector3(cx - gx_t * hl + gx_n * hw, 0.032, cz - gz_t * hl + gz_n * hw),
			Vector3(cx + gx_t * hl + gx_n * hw, 0.032, cz + gz_t * hl + gz_n * hw),
			Vector3(cx + gx_t * hl - gx_n * hw, 0.032, cz + gz_t * hl - gz_n * hw),
			b_col
		)

	if not verts.is_empty():
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = verts
		arrays[Mesh.ARRAY_NORMAL] = norms
		arrays[Mesh.ARRAY_COLOR] = cols
		arrays[Mesh.ARRAY_INDEX] = indices
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		_assign_mesh_custom_aabb(mesh, verts)
		_assign_mesh_to_instance(_bump_mesh_inst, mesh)
	else:
		_assign_mesh_to_instance(_bump_mesh_inst, null)
