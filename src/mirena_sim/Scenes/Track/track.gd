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

func _ready() -> void:
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
	var sim_node: Node = get_node_or_null("/root/Sim")
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
