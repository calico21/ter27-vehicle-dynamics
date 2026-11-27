extends Node

# -------------------------------------------------
# Global objects
# -------------------------------------------------

var _stats: Dictionary = {
	"cones_fallen": 0
}


const TelemetryUdpStreamerScript = preload("res://Scenes/Sim/TelemetryUdpStreamer.gd")

var car : MirenaCar
var track : Track
var arguments : Dictionary
var _wired_dil_session : Node = null
var telemetry_udp_streamer : Node = null


# -------------------------------------------------
# Runtime
# -------------------------------------------------


func _ready() -> void:	
	self._parse_arguments()
	_attach_telemetry_udp_streamer()


func _attach_telemetry_udp_streamer() -> void:
	if telemetry_udp_streamer != null:
		return
	telemetry_udp_streamer = TelemetryUdpStreamerScript.new()
	telemetry_udp_streamer.name = "TelemetryUdpStreamer"
	telemetry_udp_streamer.bind_sim(self)
	add_child(telemetry_udp_streamer)


func _process(_delta: float) -> void:
	if track != null and car != null and "dil_session" in track and track.dil_session != null:
		if _wired_dil_session != track.dil_session:
			wire_dil_pipeline()


func wire_dil_pipeline() -> void:
	if track == null or not ("dil_session" in track) or track.dil_session == null:
		return
	var dil: Node = track.dil_session
	if car != null and is_instance_valid(car) and dil.has_method("bind_vehicle"):
		dil.bind_vehicle(car)
	if "racing_line_vis" in track and track.racing_line_vis != null and dil.has_method("bind_racing_line_visualizer"):
		dil.bind_racing_line_visualizer(track.racing_line_vis)
	_wired_dil_session = dil

	
func _input(event):
	# Camera Toggle
	if event.is_action_pressed("alternate_camera"):
		# The manager handles the internal index math (modulo, bounds checking)
		cycle_camera()
		
		# Optional: Print the name of the node for debugging
		var active = get_active_camera()
		if active:
			print("Switched to camera: ", active.name)

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		print("Ctrl+C or Close detected. Shutting down...")
		if telemetry_udp_streamer != null:
			telemetry_udp_streamer.close_sockets()
		get_tree().quit()

# -------------------------------------------------
# Argument Parser
# -------------------------------------------------

func _parse_arguments() -> void:
	var arguments = {}
	#Process user messages
	for argument in OS.get_cmdline_args():
		if argument.contains("="):
			var key_value = argument.split("=")
			arguments[key_value[0].trim_prefix("--")] = key_value[1]
		else:
			# Options without an argument will be present in the dictionary,
			# with the value set to an empty string.
			arguments[argument.trim_prefix("--")] = ""
	self.arguments = arguments
	print(arguments)

	#else:
		
	# Hndle custom track load
	#if arguments.has("track"):
	#	SIM.get_env().get_track_manager().loadTrack(arguments["track"])
	#if arguments.has("follow"):
	#	SIM.get_vehicle().follow_path($Track.path)

	
# -------------------------------------------------
# Camera management
# -------------------------------------------------

# Dictionary to store cameras by a unique String ID
var _cameras: Array[Camera3D] = []
var _active_index: int = -1

## Just pass the camera. No ID needed.
func register_camera(camera: Camera3D):
	if not _cameras.has(camera):
		_cameras.append(camera)
		
		# AUTOMATION: When the camera is deleted, it cleans itself up
		camera.tree_exiting.connect(func(): unregister_camera(camera))
		
		if _cameras.size() == 1:
			switch_to_index(0)

func unregister_camera(camera: Camera3D):
	var idx = _cameras.find(camera)
	if idx != -1:
		_cameras.remove_at(idx)
		# If we removed the active camera, snap to the previous valid one
		if _active_index == idx:
			_active_index = posmod(_active_index - 1, max(1, _cameras.size()))
			_update_camera_states()

## Cycle to the next camera in the order they were added
func cycle_camera():
	if _cameras.is_empty(): return
	var next_idx = (_active_index + 1) % _cameras.size()
	switch_to_index(next_idx)

func switch_to_index(idx: int):
	if idx < 0 or idx >= _cameras.size(): return
	_active_index = idx
	_update_camera_states()

func _update_camera_states():
	for i in _cameras.size():
		if is_instance_valid(_cameras[i]):
			_cameras[i].current = (i == _active_index)

## Helper to get the currently active camera object
func get_active_camera() -> Camera3D:
	if _active_index != -1 and _active_index < _cameras.size():
		return _cameras[_active_index]
	return null


func set_active_camera_fov(fov_deg: float) -> void:
	var clamped := clampf(fov_deg, 55.0, 95.0)
	for cam in _cameras:
		if is_instance_valid(cam):
			cam.fov = clamped


func get_active_camera_fov() -> float:
	var cam := get_active_camera()
	if cam != null:
		return cam.fov
	return 75.0


func select_camera_by_label(label: String) -> bool:
	var target := label.strip_edges().to_lower()
	for i in range(_cameras.size()):
		if is_instance_valid(_cameras[i]):
			var nm := String(_cameras[i].name).to_lower()
			if nm == target or nm.contains(target):
				switch_to_index(i)
				return true
	return false


func set_time_scale(scale: float) -> void:
	Engine.time_scale = clampf(scale, 0.1, 1.0)


# -------------------------------------------------
# Multi-Pin Practice System (3 slots: 0, 1, 2)
# -------------------------------------------------

const PRACTICE_PIN_SLOTS: int = 3
var _practice_pins: Array[Dictionary] = [{}, {}, {}]


func has_practice_pin(slot: int) -> bool:
	if slot < 0 or slot >= PRACTICE_PIN_SLOTS:
		return false
	return not _practice_pins[slot].is_empty()


func get_practice_pin_info(slot: int) -> Dictionary:
	if slot < 0 or slot >= PRACTICE_PIN_SLOTS:
		return {}
	return _practice_pins[slot].duplicate(true)


func clear_practice_pin(slot: int) -> void:
	if slot >= 0 and slot < PRACTICE_PIN_SLOTS:
		_practice_pins[slot] = {}


func set_practice_pin(slot: int, target_car: Node = null) -> bool:
	if slot < 0 or slot >= PRACTICE_PIN_SLOTS:
		return false
	var c := target_car if target_car != null else car
	if c == null or not is_instance_valid(c) or not (c is Node3D):
		return false
	var xf := (c as Node3D).global_transform
	var lin_vel := Vector3.ZERO
	var ang_vel := Vector3.ZERO
	if c is RigidBody3D:
		var rb := c as RigidBody3D
		lin_vel = rb.linear_velocity
		ang_vel = rb.angular_velocity
	var local_v := xf.basis.inverse() * lin_vel
	var local_w := xf.basis.inverse() * ang_vel
	var vx := -local_v.z
	var vy := -local_v.x
	var wz := local_w.y
	var w_omegas: Array[float] = [0.0, 0.0, 0.0, 0.0]
	if "model" in c and c.model != null and "wheels" in c.model and c.model.wheels is Array:
		for i in range(mini(4, c.model.wheels.size())):
			if c.model.wheels[i] != null and "omega" in c.model.wheels[i]:
				w_omegas[i] = float(c.model.wheels[i].omega)

	_practice_pins[slot] = {
		"transform": xf,
		"lin_vel": lin_vel,
		"ang_vel": ang_vel,
		"vx": vx,
		"vy": vy,
		"wz": wz,
		"wheel_omegas": w_omegas,
	}
	return true


func teleport_to_pin(slot: int, target_car: Node = null) -> bool:
	if not has_practice_pin(slot):
		return false
	var c := target_car if target_car != null else car
	var pin := _practice_pins[slot]
	return teleport_car_with_state(
		c,
		pin["transform"],
		pin["lin_vel"],
		pin["ang_vel"],
		pin.get("wheel_omegas", [])
	)


func teleport_car_with_state(
	target_car: Node,
	xf: Transform3D,
	lin_vel: Vector3 = Vector3.ZERO,
	ang_vel: Vector3 = Vector3.ZERO,
	wheel_omegas: Array = []
) -> bool:
	if target_car == null or not is_instance_valid(target_car) or not (target_car is Node3D):
		return false

	if "gas" in target_car:
		target_car.gas = 0.0
	if "steering" in target_car:
		target_car.steering = 0.0
	if "brake" in target_car:
		target_car.brake = 0.0
	if "input_conditioner" in target_car and target_car.input_conditioner != null and target_car.input_conditioner.has_method("reset"):
		target_car.input_conditioner.reset()

	if "model" in target_car and target_car.model != null and target_car.model.has_method("reset"):
		target_car.model.reset()

	if target_car is RigidBody3D:
		var rb := target_car as RigidBody3D
		rb.global_transform = xf
		rb.linear_velocity = lin_vel
		rb.angular_velocity = ang_vel
		rb.sleeping = false
	else:
		(target_car as Node3D).global_transform = xf

	if "origin" in target_car:
		target_car.origin = xf

	if "model" in target_car and target_car.model != null:
		var local_v := xf.basis.inverse() * lin_vel
		var local_w := xf.basis.inverse() * ang_vel
		var v_ros := Vector3(-local_v.z, -local_v.x, local_v.y)
		var w_ros := Vector3(-local_w.z, -local_w.x, local_w.y)
		if "linear_velocity_ros" in target_car.model:
			target_car.model.linear_velocity_ros = v_ros
		if "angular_velocity_ros" in target_car.model:
			target_car.model.angular_velocity_ros = w_ros
		if "chassis_state" in target_car.model and target_car.model.chassis_state != null:
			target_car.model.chassis_state.linear_velocity = v_ros
			target_car.model.chassis_state.angular_velocity = w_ros
		if "wheels" in target_car.model and target_car.model.wheels is Array:
			var r_eff := 0.205
			if "config" in target_car and target_car.config != null and "tire_radius" in target_car.config:
				r_eff = maxf(float(target_car.config.tire_radius), 0.1)
			for i in range(mini(4, target_car.model.wheels.size())):
				if target_car.model.wheels[i] != null and "omega" in target_car.model.wheels[i]:
					if i < wheel_omegas.size():
						target_car.model.wheels[i].omega = float(wheel_omegas[i])
					else:
						target_car.model.wheels[i].omega = v_ros.x / r_eff

	return true


func spawn_test_zone(zone_name: String, target_car: Node = null) -> Dictionary:
	var c := target_car if target_car != null else car
	if track == null or not track.has_method("get_test_zone_spawn"):
		return {}
	var info: Dictionary = track.get_test_zone_spawn(zone_name)
	if info.is_empty():
		return {}
	var xf: Transform3D = info.get("transform", Transform3D.IDENTITY)
	var v0: float = float(info.get("initial_speed_mps", 0.0))
	var fwd := -xf.basis.z
	fwd.y = 0.0
	fwd = fwd.normalized() if fwd.length() > 1e-4 else Vector3(0.0, 0.0, -1.0)
	var lin_vel := fwd * v0
	teleport_car_with_state(c, xf, lin_vel, Vector3.ZERO)
	if str(info.get("zone", "")) == "skidpad":
		if track.has_method("set_skidpad_ring_visible"):
			track.set_skidpad_ring_visible(true)
		if track.has_method("reset_skidpad_stats"):
			track.reset_skidpad_stats()
	return info


func teleport_to_corner(corner: Dictionary, target_car: Node = null) -> bool:
	var c := target_car if target_car != null else car
	if c == null or corner.is_empty():
		return false
	var sp: Variant = corner.get("spawn_pos", [0.0, 0.0, 0.0])
	var x_ros := 0.0
	var y_ros := 0.0
	var z_ros := 0.0
	if sp is Array and (sp as Array).size() >= 2:
		x_ros = float(sp[0])
		y_ros = float(sp[1])
		z_ros = float(sp[2]) if (sp as Array).size() >= 3 else 0.0
	elif sp is Vector3:
		x_ros = (sp as Vector3).x
		y_ros = (sp as Vector3).y
		z_ros = (sp as Vector3).z
	var spawn_yaw := float(corner.get("spawn_yaw", 0.0))
	var spawn_vx := clampf(float(corner.get("spawn_vx", float(corner.get("v_entry_kmh", 50.0)) / 3.6)), 0.0, 45.0)
	var pos_godot := Vector3(-y_ros, maxf(z_ros, 0.28), -x_ros)
	var basis_godot := Basis(Vector3.UP, spawn_yaw)
	var xf := Transform3D(basis_godot, pos_godot)
	var fwd := -basis_godot.z
	fwd.y = 0.0
	fwd = fwd.normalized() if fwd.length() > 1e-4 else Vector3(0.0, 0.0, -1.0)
	return teleport_car_with_state(c, xf, fwd * spawn_vx, Vector3.ZERO)


func sync_track_conditions_from_meta(meta: Dictionary, target_car: Node = null) -> bool:
	var c := target_car if target_car != null else car
	if c == null or meta.is_empty():
		return false
	var synced := false
	if meta.has("mu_scale"):
		var mu_val := clampf(float(meta["mu_scale"]), 0.60, 1.15)
		if "model" in c and c.model != null:
			if c.model.has_method("set_track_grip_profile"):
				var gp := String(meta.get("grip_profile", "")).strip_edges().to_lower()
				if gp == "ttc_lab" or gp == "ttc_lab_peak":
					c.model.set_track_grip_profile(0)
				else:
					c.model.set_track_grip_profile(1)
			if "env_mu_scale" in c.model:
				c.model.env_mu_scale = 1.0
		if c.has_method("apply_track_conditions"):
			c.apply_track_conditions({"mu_scale": mu_val})
		elif c.has_method("set_track_surface_mu"):
			c.set_track_surface_mu(mu_val)
		synced = true
	if meta.has("tire_temp_mean_c") or meta.has("temp_c"):
		var t_val := clampf(float(meta.get("tire_temp_mean_c", meta.get("temp_c", 45.0))), 10.0, 110.0)
		if "model" in c and c.model != null and "wheels" in c.model and c.model.wheels is Array:
			for w in c.model.wheels:
				if w != null and w.has_method("set_tire_temperature"):
					w.set_tire_temperature(t_val, t_val)
		synced = true
	return synced


func load_ghost_from_casadi_json(json_path: String, pace_offset_s: float = 0.0) -> bool:
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
	var pts: Array = raw_pts
	var trace: Array = []
	var t_accum := 0.0
	var prev_s := 0.0
	for i in range(pts.size()):
		if not (pts[i] is Dictionary):
			continue
		var p: Dictionary = pts[i]
		var s_i := float(p.get("s", p.get("s_m", float(i) * 2.5)))
		var v_i := maxf(float(p.get("v_ref", p.get("v_mps", 12.0))), 1.5)
		if i > 0:
			t_accum += maxf(s_i - prev_s, 0.0) / v_i
		prev_s = s_i
		var x_ros := float(p.get("x", 0.0))
		var y_ros := float(p.get("y", 0.0))
		var yaw_rad := float(p.get("yaw", p.get("yaw_rad", 0.0)))
		trace.append({
			"s": s_i,
			"t": t_accum,
			"x": -y_ros,
			"y": 0.0,
			"z": -x_ros,
			"yaw": yaw_rad,
			"roll": 0.0,
			"pitch": 0.0,
		})
	if trace.is_empty():
		return false
	if track != null:
		if "dil_session" in track and track.dil_session != null:
			if track.dil_session.has_method("load_casadi_reference_json"):
				track.dil_session.load_casadi_reference_json(json_path)
			else:
				track.dil_session._lts_trace = trace
			if track.dil_session.has_method("set_pace_rabbit_offset"):
				track.dil_session.set_pace_rabbit_offset(pace_offset_s)
			if track.dil_session.has_method("set_ghost_visible"):
				track.dil_session.set_ghost_visible(true)
			if track.dil_session.has_method("set_ghost_mode"):
				track.dil_session.set_ghost_mode(2)
		if "racing_line_vis" in track and track.racing_line_vis != null and track.racing_line_vis.has_method("apply_casadi_json"):
			track.racing_line_vis.apply_casadi_json(json_path, true)
			track.racing_line_vis.visible = true
		wire_dil_pipeline()
	if car != null and car.has_method("set_racing_line_enabled"):
		car.set_racing_line_enabled(true)
	if d.get("meta") is Dictionary:
		sync_track_conditions_from_meta(d.get("meta") as Dictionary)
	return true
