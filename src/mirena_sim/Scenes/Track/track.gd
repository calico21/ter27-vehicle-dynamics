extends Node3D
class_name Track

# Properties
## D 8.1.1: the minimum track width is 3 m. The generator holds the centreline
## to the turning and length rules; this is the one it cannot, because it is a
## property of how the cones are laid out either side of that centreline.
@export var track_width : float = 3.0
## Distance between gates along the track, and so between cones down one side.
## The rules cap that at 5 m.
@export var track_spacing : float = 4.0
## How far behind the start line the car lines up, as the rules place it.
const START_OFFSET := 6.0

# ROS 2 Frame: X Forward (+Z), Y Left (+X), Z Up (+Y)
var track_curve : Curve3D = Curve3D.new() 
@onready var track_path : Path3D = $TrackPath

# Internal Refs
static var _gate_scene = preload("res://Scenes/Track/Gate/gate.tscn")
static var _cone_scene = preload("res://Scenes/Track/Cone/cone.tscn")
const RacingLineVisScript = preload("res://Scenes/Track/RacingLineVisualizer.gd")
const DilSessionMgrScript = preload("res://Scenes/Sim/DilSessionManager.gd")

const SKIDPAD_RADIUS_M: float = 9.125
const SKIDPAD_RING_HALF_WIDTH_M: float = 0.18
const SKIDPAD_RING_Y_M: float = 0.042
const SKIDPAD_RING_SEGMENTS: int = 96
const TRAIL_BRAKE_SPEED_MPS: float = 75.0 / 3.6

var racing_line_vis: RacingLineVisScript = null
var dil_session: DilSessionMgrScript = null
var last_loaded_track_path: String = ""

var _skidpad_ring_mesh: MeshInstance3D = null
var _skidpad_ring_visible: bool = false
var _skidpad_center: Vector3 = Vector3(110.0, 0.0, 0.0)
var _track_markers_visible: bool = true

# Skidpad 9.125m auto-timing state
var last_skidpad_lap_s: float = 0.0
var best_skidpad_lap_s: float = 0.0
var avg_ay_g: float = 0.0
var avg_beta_deg: float = 0.0
var _skidpad_has_prev_angle: bool = false
var _skidpad_prev_angle: float = 0.0
var _skidpad_accum_angle: float = 0.0
var _skidpad_lap_time_s: float = 0.0
var _skidpad_ay_sum: float = 0.0
var _skidpad_beta_sum: float = 0.0
var _skidpad_dt_sum: float = 0.0
var _skidpad_live_ay_g: float = 0.0
var _skidpad_live_beta_deg: float = 0.0

# Track Generator
var track_gen = TrackGenerator.new()
signal track_loaded

# Surface Grip Map (optional per-track variable friction zones in Godot world XZ)
var grip_zones: Array = []
var default_surface_mu: float = 1.0

# ROS node to manage tracks
var _node: Object
var _track_pub : Object
var _map_pub : Object
var _map_timer : Object
var _track_timer : Object

func _new_ros_msg(msg_class: String) -> Object:
	var script: Script = load("res://addons/rclgd/gen/%s.gd" % msg_class)
	return script.new() if script else null

# Standard ROS 2 Origin (X: Forward, Y: Left, PSI: Counter-Clockwise Yaw)
var origin : Transform3D 

# Parameter handling
func _on_ros_parameter_changed(param_name: String, value: Variant):
	if param_name == "track":
		if value == "random":
			create_track()
		else:
			if FileAccess.file_exists(value):
				load_track(value)

func _ensure_dil_children() -> void:
	if racing_line_vis == null:
		racing_line_vis = RacingLineVisScript.new()
		racing_line_vis.name = "RacingLineVisualizer"
		add_child(racing_line_vis)
	if dil_session == null:
		dil_session = DilSessionMgrScript.new()
		dil_session.name = "DilSessionManager"
		add_child(dil_session)


func _on_track_geometry_updated() -> void:
	_ensure_dil_children()
	if racing_line_vis != null:
		racing_line_vis.build_from_track(self, last_loaded_track_path)
	if dil_session != null:
		dil_session.configure_track_from_node(self)
		var stem := last_loaded_track_path.get_file().get_basename().to_lower() if last_loaded_track_path != "" else ""
		if stem != "":
			var rl_json := "res://Assets/RacingLines/%s_racing_line.json" % stem
			if FileAccess.file_exists(rl_json):
				dil_session.load_lts_trace_from_json(rl_json)
	_update_skidpad_center()
	_rebuild_skidpad_ring_mesh()
	var sim_node: Node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and "car" in sim_node and sim_node.get("car") != null:
		var c: Node = sim_node.get("car")
		if c.has_method("sync_road_surface_from_track"):
			c.sync_road_surface_from_track(self)


func _ensure_ground_opaque() -> void:
	var gm := get_node_or_null("Ground/GroundMesh") as MeshInstance3D
	if gm == null or gm.mesh == null or gm.mesh.get_surface_count() <= 0:
		return
	var mat := gm.mesh.surface_get_material(0)
	if mat is StandardMaterial3D and (mat as StandardMaterial3D).refraction_enabled:
		var opaque_mat := (mat as StandardMaterial3D).duplicate() as StandardMaterial3D
		opaque_mat.refraction_enabled = false
		gm.material_override = opaque_mat


func _ready() -> void:
	_ensure_ground_opaque()
	_ensure_dil_children()
	if not ClassDB.class_exists("RosNode"):
		return
	_node = ClassDB.instantiate("RosNode")
	_node.init("track_manager")
	# Track name parameter and callback
	_node.declare_parameter("track", "")
	_node.parameter_changed.connect(_on_ros_parameter_changed)
	# Publishers
	_track_pub = _node.create_publisher("~/track","mirena_common/msg/Track")
	_map_pub = _node.create_publisher("~/full_map","mirena_common/msg/EntityList")
	# Timers for publishers
	_map_timer = _node.create_timer(1,_publish_full_map)
	_track_timer = _node.create_timer(1,_publish_track)
	
	# Check parameter
	var value = _node.get_parameter("track")
	if value == "random":
		create_track()
	else:
		if FileAccess.file_exists(value):
			load_track(value)

# ROS publishers
func _publish_full_map():
	var msg = _new_ros_msg("RosMirenaCommonEntityList")
	msg.header.frame_id = "debug_map"
	msg.header.stamp = _node.now()
	msg.entities = get_tree().get_nodes_in_group("Cones").map(func(c): return _to_ent(c,true))
	_map_pub.publish(msg)

func _publish_track():
	var sim_node: Node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node == null or not sim_node.get("track"): return
	var msg = _new_ros_msg("RosMirenaCommonTrack")
	msg.header.frame_id = "debug_map"
	msg.header.stamp = _node.now()
	msg.is_closed = track_curve.closed
	msg.gates = get_gate_positions().map(func(t: Transform3D):
		var gate = _new_ros_msg("RosMirenaCommonGate")
		gate.x = -t.origin.z  # Godot -Z Forward -> ROS +X
		gate.y = -t.origin.x  # Godot -X Left    -> ROS +Y
		gate.psi = -t.basis.get_euler(EulerOrder.EULER_ORDER_YXZ).x
		return gate
	)
	_track_pub.publish(msg)
	
func _to_ent(cone: Node3D, global : bool = false) -> Object:
	var ent = _new_ros_msg("RosMirenaCommonEntity")
	var pos = cone.global_position if global else to_local(cone.global_position)
	ent.type = cone.get_type_as_string()
	
	# ROS Swizzle: Forward=Z, Left=-X, Up=Y
	ent.position.x = -pos.z
	ent.position.y = -pos.x
	ent.position.z = pos.y
	return ent
	
func create_track():
	clear_track()
	last_loaded_track_path = ""
	parse_grip_map({})
	# Update curve and path
	track_curve = track_gen.generate()
	track_path.curve = track_curve
	var stats: Dictionary = track_gen.last_stats
	print("Track: %.0f m lap, tightest turn %.1f m radius, longest straight %.0f m (%d attempts%s)" % [
		stats.get("lap_length", 0.0), stats.get("min_turn_radius", 0.0),
		stats.get("longest_straight", 0.0), stats.get("attempts", 0),
		", fallback" if stats.get("fallback", false) else ""])
	
	var length = track_curve.get_baked_length()
	var num_gates = int(length / track_spacing)

	for i in range(0, num_gates):
		var d = (i * track_spacing)
		var gate = _gate_scene.instantiate() as Gate
		$Gates.add_child(gate) 
		
		gate.gate_width = track_width
		
		# --- INITIAL GATE SETUP ---
		if i == 0:
			gate.gate_type = Gate.GateType.EVENT
			gate.add_to_group("InitialGate")
		else:
			gate.gate_type = Gate.GateType.STANDARD
		
		var trans = track_curve.sample_baked_with_rotation(d, true)
		gate.global_transform = trans
		gate.rotate_object_local(Vector3.UP, PI)

	# --- Shifted Origin Logic ---
	var p0 = track_curve.get_point_position(0)
	var p1 = track_curve.get_point_position(1)
	var dir = (p0 - p1).normalized()

	var shifted_p0 = p0 + (dir * START_OFFSET)

	origin.origin = shifted_p0
	origin.basis = Basis.looking_at(dir, Vector3.UP, true)

	_on_track_geometry_updated()
	track_loaded.emit()

func export_track(path: String) -> bool:
	var to_start := origin.affine_inverse()

	var cones := []
	for cone in get_tree().get_nodes_in_group("Cones"):
		var p = to_start * cone.global_position
		cones.append({
			"type": cone.get_type_as_string(),
			"x": snappedf(-p.z, 0.001),
			"y": snappedf(-p.x, 0.001)})

	var points := []
	var length := track_curve.get_baked_length()
	var walked := 0.0
	while walked < length:
		var p = to_start * track_curve.sample_baked(walked)
		points.append({"x": snappedf(-p.z, 0.001), "y": snappedf(-p.x, 0.001)})
		walked += track_spacing

	var data := {
		"metadata": {
			"track_name": path.get_file().get_basename(),
			"description": "Generated by mirena_sim",
			"date": Time.get_date_string_from_system()},
		"setup": {"car_start_pose": {"x": 0.0, "y": 0.0, "psi": 0.0}},
		"cones": cones,
		"path": {"closed": track_curve.closed, "points": points}}

	var file = FileAccess.open(path, FileAccess.WRITE)
	if not file:
		push_error("Track: cannot write %s" % path)
		return false
	file.store_string(JSON.stringify(data, "  "))
	return true

func load_track(path : String):
	clear_track()
	last_loaded_track_path = path
	var file = FileAccess.open(path, FileAccess.READ)
	if not file: return
	
	var json = JSON.new()
	json.parse(file.get_as_text())
	var data = json.data
	if typeof(data) == TYPE_DICTIONARY:
		parse_grip_map(data)
	else:
		parse_grip_map({})
	
	# Load Cones
	if data.has("cones"):
		for cone_data in data["cones"]:
			var cone = _cone_scene.instantiate() as Cone
			cone.position = Vector3(-cone_data["y"], 0.0, -cone_data["x"])
			
			match cone_data["type"]:
				"cone_blue": cone.type = Cone.ConeColor.BLUE
				"cone_yellow": cone.type = Cone.ConeColor.YELLOW
				"cone_orange": cone.type = Cone.ConeColor.ORANGE
				"cone_big_orange": cone.type = Cone.ConeColor.BIG_ORANGE
			
			cone.add_to_group("Cones")
			cone.rotation.y = randf_range(0, PI/2)
			$Gates.add_child(cone)

	if data.has('path'):
		track_curve.clear_points()
		for p in data['path']['points']:
			track_curve.add_point(Vector3(-p['y'], 0.0, -p['x']))
		track_curve.closed = data['path']['closed']
		
		# --- SPAWN INITIAL GATE FOR LOADED TRACKS ---
		if track_curve.get_baked_length() > 0:
			var initial_gate = _gate_scene.instantiate() as Gate
			$Gates.add_child(initial_gate)
			initial_gate.gate_width = track_width
			initial_gate.gate_type = Gate.GateType.EVENT
			initial_gate.add_to_group("InitialGate")
			
			var trans = track_curve.sample_baked_with_rotation(0.0, true)
			initial_gate.global_transform = trans
			initial_gate.rotate_object_local(Vector3.UP, PI)

	origin.origin = Vector3(-data.setup.car_start_pose.y, 0.0, -data.setup.car_start_pose.x)
	origin.basis = Basis(Vector3.UP, data.setup.car_start_pose.psi)
	_on_track_geometry_updated()
	track_loaded.emit()

func clear_track():
	grip_zones.clear()
	default_surface_mu = 1.0
	for n in $Gates.get_children(): n.queue_free()

## Parses optional "surface_grip" or "grip_zones" definitions from a loaded track JSON Dictionary.
func parse_grip_map(track_data: Dictionary) -> void:
	grip_zones.clear()
	default_surface_mu = 1.0
	if track_data.is_empty():
		return

	var raw_zones: Array = []
	if track_data.has("surface_grip"):
		var sg: Variant = track_data["surface_grip"]
		if sg is Dictionary:
			default_surface_mu = clampf(float(sg.get("default_mu_scale", sg.get("default_surface_mu", 1.0))), 0.05, 2.0)
			var z_candidate: Variant = sg.get("zones", sg.get("grip_zones", []))
			if z_candidate is Array:
				raw_zones = z_candidate
		elif sg is Array:
			raw_zones = sg
			default_surface_mu = clampf(float(track_data.get("default_mu_scale", 1.0)), 0.05, 2.0)
	elif track_data.has("grip_zones"):
		default_surface_mu = clampf(float(track_data.get("default_mu_scale", track_data.get("default_surface_mu", 1.0))), 0.05, 2.0)
		var gz: Variant = track_data["grip_zones"]
		if gz is Array:
			raw_zones = gz

	for z in raw_zones:
		if z is Dictionary:
			grip_zones.append(z.duplicate(true))

## Returns surface friction multiplier at world position (Godot XZ plane).
func get_surface_mu_at(world_pos: Vector3) -> float:
	return eval_grip_zones(world_pos, grip_zones, default_surface_mu)

## Evaluates surface friction multiplier for a list of grip zones with smoothstep feathering.
static func eval_grip_zones(world_pos: Vector3, zones: Array, default_mu: float = 1.0) -> float:
	var mu: float = clampf(default_mu, 0.05, 2.0)
	if zones.is_empty():
		return mu

	for zone_var in zones:
		if not (zone_var is Dictionary):
			continue
		var z: Dictionary = zone_var
		var shape_type: String = str(z.get("type", z.get("shape", ""))).to_lower()
		var zone_mu: float = clampf(float(z.get("mu_scale", mu)), 0.05, 2.0)
		var feather_m: float = maxf(float(z.get("feather_m", 1.0)), 0.0)

		var d_out: float = INF
		match shape_type:
			"circle":
				var cx: float = float(z.get("cx", 0.0))
				var cz: float = float(z.get("cz", 0.0))
				var radius: float = maxf(float(z.get("radius", 0.0)), 0.0)
				var dist: float = Vector2(world_pos.x - cx, world_pos.z - cz).length()
				d_out = dist - radius
			"annulus":
				var cx_a: float = float(z.get("cx", 0.0))
				var cz_a: float = float(z.get("cz", 0.0))
				var r_in: float = maxf(float(z.get("inner_radius", 0.0)), 0.0)
				var r_out: float = maxf(float(z.get("outer_radius", r_in)), r_in)
				var dist_a: float = Vector2(world_pos.x - cx_a, world_pos.z - cz_a).length()
				d_out = maxf(r_in - dist_a, dist_a - r_out)
			"rect":
				var x0: float = float(z.get("min_x", 0.0))
				var x1: float = float(z.get("max_x", 0.0))
				var z0: float = float(z.get("min_z", 0.0))
				var z1: float = float(z.get("max_z", 0.0))
				var min_x: float = minf(x0, x1)
				var max_x: float = maxf(x0, x1)
				var min_z: float = minf(z0, z1)
				var max_z: float = maxf(z0, z1)
				var dx: float = maxf(min_x - world_pos.x, maxf(0.0, world_pos.x - max_x))
				var dz: float = maxf(min_z - world_pos.z, maxf(0.0, world_pos.z - max_z))
				d_out = sqrt(dx * dx + dz * dz)
			_:
				continue

		var weight: float = 0.0
		if d_out <= 0.0:
			weight = 1.0
		elif feather_m > 1e-6 and d_out < feather_m:
			weight = 1.0 - smoothstep(0.0, feather_m, d_out)

		if weight > 0.0:
			mu = lerpf(mu, zone_mu, weight)

	return clampf(mu, 0.05, 2.0)

func get_gate_positions() -> Array[Transform3D]:
	var length = track_curve.get_baked_length()
	var num_points = int(length / track_spacing)
	var transforms: Array[Transform3D] = []
	
	for i in range(0, num_points):
		var d = (i * track_spacing)
		var pos = track_curve.sample_baked(d)
		
		var next_d = fmod(d + 0.1, length)
		var next_pos = track_curve.sample_baked(next_d)
		var dir = (next_pos - pos).normalized()
		
		var gate_xform = Transform3D()
		gate_xform.origin = pos
		gate_xform.basis = Basis.looking_at(dir, Vector3.UP, true)
		
		transforms.append(gate_xform)
		
	return transforms


func reset_cones_only() -> void:
	if dil_session != null and dil_session.has_method("reset_cones_only"):
		dil_session.reset_cones_only()
	if has_node("Gates"):
		for ch in get_node("Gates").get_children():
			if ch is Node3D:
				(ch as Node3D).rotation = Vector3.ZERO
			if ch is RigidBody3D:
				var rb := ch as RigidBody3D
				rb.linear_velocity = Vector3.ZERO
				rb.angular_velocity = Vector3.ZERO
	if is_inside_tree():
		for cone_n in get_tree().get_nodes_in_group("Cones"):
			if cone_n is Node3D:
				(cone_n as Node3D).rotation = Vector3.ZERO
			if cone_n is RigidBody3D:
				var rb2 := cone_n as RigidBody3D
				rb2.linear_velocity = Vector3.ZERO
				rb2.angular_velocity = Vector3.ZERO


func _physics_process(delta: float) -> void:
	if not is_inside_tree():
		return
	var sim_node: Node = get_node_or_null("/root/Sim")
	if sim_node == null or not ("car" in sim_node) or sim_node.get("car") == null:
		return
	var c: Node = sim_node.get("car")
	if not (c is Node3D):
		return
	var car_pos := (c as Node3D).global_position
	var dist_xz := Vector2(car_pos.x - _skidpad_center.x, car_pos.z - _skidpad_center.z).length()
	if dist_xz <= SKIDPAD_RADIUS_M + 8.0 or _skidpad_ring_visible:
		var spd := (c as RigidBody3D).linear_velocity.length() if (c is RigidBody3D) else 0.0
		var ay_g := float(c.get("_last_ay_g")) if ("_last_ay_g" in c) else 0.0
		var beta_deg := 0.0
		if "model" in c and c.model != null and ("linear_velocity_ros" in c.model):
			var lv: Vector3 = c.model.linear_velocity_ros
			if absf(lv.x) > 0.5:
				beta_deg = rad_to_deg(atan2(lv.y, absf(lv.x)))
		step_skidpad_timing(delta, car_pos, spd, ay_g, beta_deg)


func set_skidpad_ring_visible(vis: bool) -> void:
	_skidpad_ring_visible = vis
	if vis and _skidpad_ring_mesh == null:
		_update_skidpad_center()
		_rebuild_skidpad_ring_mesh()
	if _skidpad_ring_mesh != null:
		_skidpad_ring_mesh.visible = vis


func is_skidpad_ring_visible() -> bool:
	return _skidpad_ring_visible


func set_track_markers_visible(vis: bool) -> void:
	_track_markers_visible = vis
	if racing_line_vis != null and racing_line_vis.has_method("set_markers_visible"):
		racing_line_vis.set_markers_visible(vis)


func is_track_markers_visible() -> bool:
	return _track_markers_visible


func reset_skidpad_stats() -> void:
	last_skidpad_lap_s = 0.0
	best_skidpad_lap_s = 0.0
	avg_ay_g = 0.0
	avg_beta_deg = 0.0
	_skidpad_has_prev_angle = false
	_skidpad_prev_angle = 0.0
	_skidpad_accum_angle = 0.0
	_skidpad_lap_time_s = 0.0
	_skidpad_ay_sum = 0.0
	_skidpad_beta_sum = 0.0
	_skidpad_dt_sum = 0.0
	_skidpad_live_ay_g = 0.0
	_skidpad_live_beta_deg = 0.0


func get_skidpad_stats() -> Dictionary:
	return {
		"last_skidpad_lap_s": last_skidpad_lap_s,
		"best_skidpad_lap_s": best_skidpad_lap_s,
		"avg_ay_g": avg_ay_g,
		"avg_beta_deg": avg_beta_deg,
		"current_lap_s": _skidpad_lap_time_s,
		"progress_deg": clampf(rad_to_deg(absf(_skidpad_accum_angle)), 0.0, 360.0),
		"live_ay_g": _skidpad_live_ay_g,
		"live_beta_deg": _skidpad_live_beta_deg,
		"center": _skidpad_center,
		"radius_m": SKIDPAD_RADIUS_M,
	}


func step_skidpad_timing(dt: float, car_pos: Vector3, speed_mps: float, ay_g: float, beta_deg: float) -> void:
	if dt <= 0.0:
		return
	var dx := car_pos.x - _skidpad_center.x
	var dz := car_pos.z - _skidpad_center.z
	var r := sqrt(dx * dx + dz * dz)
	_skidpad_live_ay_g = ay_g
	_skidpad_live_beta_deg = beta_deg
	if r < 4.0 or r > 16.0 or speed_mps < 1.5:
		_skidpad_has_prev_angle = false
		return
	var theta := atan2(dz, dx)
	if not _skidpad_has_prev_angle:
		_skidpad_has_prev_angle = true
		_skidpad_prev_angle = theta
		return
	var d_theta := wrapf(theta - _skidpad_prev_angle, -PI, PI)
	_skidpad_prev_angle = theta
	_skidpad_accum_angle += d_theta
	_skidpad_lap_time_s += dt
	_skidpad_ay_sum += absf(ay_g) * dt
	_skidpad_beta_sum += absf(beta_deg) * dt
	_skidpad_dt_sum += dt

	if absf(_skidpad_accum_angle) >= TAU:
		last_skidpad_lap_s = _skidpad_lap_time_s
		if best_skidpad_lap_s <= 0.0 or last_skidpad_lap_s < best_skidpad_lap_s:
			best_skidpad_lap_s = last_skidpad_lap_s
		var denom := maxf(_skidpad_dt_sum, 1e-4)
		avg_ay_g = _skidpad_ay_sum / denom
		avg_beta_deg = _skidpad_beta_sum / denom
		_skidpad_accum_angle = fmod(_skidpad_accum_angle, TAU)
		_skidpad_lap_time_s = 0.0
		_skidpad_ay_sum = 0.0
		_skidpad_beta_sum = 0.0
		_skidpad_dt_sum = 0.0


func _update_skidpad_center() -> void:
	var min_x := INF
	var max_x := -INF
	var min_z := INF
	var max_z := -INF
	var has_pts := false
	if track_curve != null and track_curve.get_baked_length() > 1.0:
		var total_len := track_curve.get_baked_length()
		var s := 0.0
		while s < total_len:
			var p := track_curve.sample_baked(s)
			min_x = minf(min_x, p.x)
			max_x = maxf(max_x, p.x)
			min_z = minf(min_z, p.z)
			max_z = maxf(max_z, p.z)
			has_pts = true
			s += 4.0
	if has_pts:
		var mid_z := 0.5 * (min_z + max_z)
		var cand_x := clampf(max_x + 22.0, -340.0, 340.0)
		var cand_z := clampf(mid_z, -340.0, 340.0)
		_skidpad_center = Vector3(cand_x, 0.0, cand_z)
	else:
		_skidpad_center = Vector3(110.0, 0.0, 0.0)


func _rebuild_skidpad_ring_mesh() -> void:
	if _skidpad_ring_mesh == null:
		_skidpad_ring_mesh = MeshInstance3D.new()
		_skidpad_ring_mesh.name = "Skidpad9125RingMesh"
		_skidpad_ring_mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		_skidpad_ring_mesh.extra_cull_margin = 50.0
		_skidpad_ring_mesh.ignore_occlusion_culling = true
		var mat := StandardMaterial3D.new()
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
		mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		mat.render_priority = 11
		mat.albedo_color = Color(0.00, 0.898, 1.00, 1.0)
		_skidpad_ring_mesh.material_override = mat
		_skidpad_ring_mesh.visible = _skidpad_ring_visible
		add_child(_skidpad_ring_mesh)

	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var r_in := SKIDPAD_RADIUS_M - SKIDPAD_RING_HALF_WIDTH_M
	var r_out := SKIDPAD_RADIUS_M + SKIDPAD_RING_HALF_WIDTH_M
	for i in range(SKIDPAD_RING_SEGMENTS):
		var a0 := TAU * float(i) / float(SKIDPAD_RING_SEGMENTS)
		var a1 := TAU * float(i + 1) / float(SKIDPAD_RING_SEGMENTS)
		var c0 := cos(a0)
		var s0 := sin(a0)
		var c1 := cos(a1)
		var s1 := sin(a1)
		var v_in0 := Vector3(r_in * c0, SKIDPAD_RING_Y_M, r_in * s0)
		var v_out0 := Vector3(r_out * c0, SKIDPAD_RING_Y_M, r_out * s0)
		var v_in1 := Vector3(r_in * c1, SKIDPAD_RING_Y_M, r_in * s1)
		var v_out1 := Vector3(r_out * c1, SKIDPAD_RING_Y_M, r_out * s1)
		st.add_vertex(v_in0)
		st.add_vertex(v_out0)
		st.add_vertex(v_out1)
		st.add_vertex(v_in0)
		st.add_vertex(v_out1)
		st.add_vertex(v_in1)
	_skidpad_ring_mesh.mesh = st.commit()
	_skidpad_ring_mesh.position = _skidpad_center


func _sample_curve_curvature_at(s_m: float, total_len: float, delta_m: float = 2.0) -> float:
	var p0 := track_curve.sample_baked(fposmod(s_m - delta_m, total_len))
	var p1 := track_curve.sample_baked(fposmod(s_m, total_len))
	var p2 := track_curve.sample_baked(fposmod(s_m + delta_m, total_len))
	var d1 := (p1 - p0)
	var d2 := (p2 - p1)
	d1.y = 0.0
	d2.y = 0.0
	if d1.length() < 1e-4 or d2.length() < 1e-4:
		return 0.0
	return absf(d1.normalized().signed_angle_to(d2.normalized(), Vector3.UP))


func _transform_at_curve_s(s_m: float, total_len: float) -> Transform3D:
	var s_wrap := fposmod(s_m, total_len)
	var pos := track_curve.sample_baked(s_wrap)
	var next_pos := track_curve.sample_baked(fposmod(s_wrap + 0.5, total_len))
	var fwd := next_pos - pos
	fwd.y = 0.0
	if fwd.length() < 1e-4:
		return origin
	var xf := Transform3D()
	xf.origin = Vector3(pos.x, origin.origin.y, pos.z)
	xf.basis = Basis.looking_at(fwd.normalized(), Vector3.UP)
	return xf


func get_test_zone_spawn(zone_name: String) -> Dictionary:
	var clean := zone_name.strip_edges().to_lower().replace("-", "_").replace(" ", "_")
	if clean == "launch_pad" or clean == "launch":
		return {
			"zone": "launch",
			"transform": origin,
			"initial_speed_mps": 0.0,
		}
	if clean == "skidpad" or clean == "skidpad_9125m" or clean == "skidpad_9.125m" or clean == "skidpad_9.125m_test":
		_update_skidpad_center()
		_rebuild_skidpad_ring_mesh()
		var spawn_pos := _skidpad_center + Vector3(SKIDPAD_RADIUS_M, origin.origin.y, 0.0)
		var tangent_fwd := Vector3(0.0, 0.0, -1.0)
		var xf_skid := Transform3D()
		xf_skid.origin = spawn_pos
		xf_skid.basis = Basis.looking_at(tangent_fwd, Vector3.UP)
		return {
			"zone": "skidpad",
			"transform": xf_skid,
			"initial_speed_mps": 0.0,
			"skidpad_center": _skidpad_center,
			"radius_m": SKIDPAD_RADIUS_M,
		}

	if track_curve == null or track_curve.get_baked_length() < 10.0:
		return {
			"zone": clean,
			"transform": origin,
			"initial_speed_mps": TRAIL_BRAKE_SPEED_MPS if clean.begins_with("trail_brake") else 0.0,
		}

	var total_len := track_curve.get_baked_length()
	if clean == "hairpin" or clean == "hairpin_test":
		var best_curv := -1.0
		var best_s := total_len * 0.25
		var s := 0.0
		while s < total_len:
			var ang := _sample_curve_curvature_at(s, total_len, 2.0)
			if ang > best_curv:
				best_curv = ang
				best_s = s
			s += 2.0
		return {
			"zone": "hairpin",
			"transform": _transform_at_curve_s(best_s - 25.0, total_len),
			"initial_speed_mps": 0.0,
		}

	if clean == "slalom" or clean == "slalom_test":
		# Find fast / alternating-curvature slalom section entry
		var best_score := -1.0
		var best_s_sl := total_len * 0.5
		var s2 := 0.0
		while s2 < total_len:
			var c_prev := _sample_curve_curvature_at(s2 - 4.0, total_len, 2.0)
			var c_mid := _sample_curve_curvature_at(s2, total_len, 2.0)
			var c_next := _sample_curve_curvature_at(s2 + 6.0, total_len, 2.0)
			# Prefer gentle entry (small c_prev/c_mid) leading into brisk S-bends (c_next)
			var score := c_next - 0.5 * c_prev - 0.25 * c_mid
			if score > best_score:
				best_score = score
				best_s_sl = s2
			s2 += 3.0
		return {
			"zone": "slalom",
			"transform": _transform_at_curve_s(best_s_sl, total_len),
			"initial_speed_mps": 0.0,
		}

	if clean == "trail_brake" or clean == "trail_brake_entry" or clean == "trail_brake_entry_test":
		# Find heavy braking corner preceded by a fast/straight 45m approach
		var best_tb_score := -1.0
		var best_tb_s := total_len * 0.35
		var s3 := 0.0
		while s3 < total_len:
			var corner_curv := _sample_curve_curvature_at(s3, total_len, 2.5)
			var approach_curv := _sample_curve_curvature_at(s3 - 25.0, total_len, 4.0) + _sample_curve_curvature_at(s3 - 40.0, total_len, 4.0)
			var tb_score := corner_curv / (1.0 + 3.0 * approach_curv)
			if tb_score > best_tb_score:
				best_tb_score = tb_score
				best_tb_s = s3
			s3 += 2.0
		return {
			"zone": "trail_brake",
			"transform": _transform_at_curve_s(best_tb_s - 45.0, total_len),
			"initial_speed_mps": TRAIL_BRAKE_SPEED_MPS,
		}

	return {
		"zone": clean,
		"transform": origin,
		"initial_speed_mps": 0.0,
	}


func get_quick_spawn_transform(preset_name: String) -> Transform3D:
	var info := get_test_zone_spawn(preset_name)
	return info.get("transform", origin)
