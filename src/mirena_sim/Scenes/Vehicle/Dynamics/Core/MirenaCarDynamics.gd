extends RigidBody3D
class_name MirenaCar

## Mirena Dynamics — Vehicle Dynamics Model using RigidBody3D
## Formula Student Driverless Monoplaza (Mirena)
##
## Replaces native VehicleBody3D with custom 10-DOF dynamics:
## - 4 RayCast3D ground contact sensors for suspension.
## - Custom tire model with combined slip and friction ellipse (TireModel.gd).
## - Configurable 4WD / RWD powertrain with 80 kW limit (DrivetrainModel.gd).
## - Chassis integration via _integrate_forces (no double integration).

const VehicleConfigScript = preload("VehicleConfig.gd")
const VehicleInputsScript = preload("VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("VehicleDynamicsModel.gd")
const TireModelScript = preload("../Tires/TireModel.gd")
const DrivetrainModelScript = preload("../Drivetrain/DrivetrainModel.gd")
const BrakeModelScript = preload("../Brakes/BrakeModel.gd")
const SuspensionModelScript = preload("../Suspension/SuspensionModel.gd")
const AeroModelScript = preload("../Aerodynamics/AeroModel.gd")
const VehicleTelemetryScript = preload("../Telemetry/VehicleTelemetry.gd")

signal cone_hit(count: int)
enum PilotMode { NO_PILOT, MANUAL, ROS, TRACK_RAIL }

# --- Physical Configuration ---
var config: VehicleConfigScript
var model: VehicleDynamicsModelScript
var inputs: VehicleInputsScript
var telemetry: VehicleTelemetryScript

@export var pilot: PilotMode = PilotMode.NO_PILOT
@export var drive_mode: VehicleConfigScript.DriveMode = VehicleConfigScript.DriveMode.FOUR_WD
@export var rail_speed: float = 10.0
@export var rail_look_ahead: float = 3.0
var _rail_progress: float = 0.0
var path: Path3D

# --- Suspension Mount Geometry ---
# Hardpoints for raycasts (in car local space, +X right, +Y up, -Z forward)
var ray_mount_positions: Array[Vector3] = [
	Vector3(-0.587, 0.28, -0.993), # FL
	Vector3( 0.587, 0.28, -0.993), # FR
	Vector3(-0.587, 0.28,  0.600), # RL
	Vector3( 0.587, 0.28,  0.600)  # RR
]

var ray_rest_lengths: Array[float] = [0.32, 0.32, 0.32, 0.32] # 4 cm static travel allowance
var ray_cast_length: float = 0.45

var raycasts: Array[RayCast3D] = []
var wheel_mesh_nodes: Array[Node3D] = []

# --- ROS Configuration & Variables ---
@export var frame_id: String = "car/cog"
@export var wheel_speed_rate: float = 100.0
@export var wheel_speed_lpf_tau: float = 0.05
@export var wheel_speed_quantise_rpm: float = 1.0

var _node = null
var _state_pub = null
var _as_info_pub = null
var _perception_pub = null
var _slam_pub = null
var _inferred_control_pub = null
var _wheel_speeds_pub = null
var _lap_status_pub = null
var _debug_tim = null

var _control_sub = null
var _mission_info_sub = null
var _as_status_sub = null

var _ros_gas: float = 0.0
var _ros_steer: float = 0.0
var _wheel_decim_n: int = 0
var _wheel_rpm_acc: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _wheel_rpm_lpf: Array[float] = [0.0, 0.0, 0.0, 0.0]

# --- Internal States ---
var origin: Transform3D = Transform3D.IDENTITY
var gas: float = 0.0
var steering: float = 0.0
var brake: float = 0.0
var _steer_smoothed: float = 0.0

var slam_cones: Array = []
var hit_cones: Array = []
var _as_info = null
var _mission_info = null

var initial_gate_crossings: int = 0
var lap_start_ticks: float = 0.0

var car_state = null
var wheel_speeds = null
var lap_status_msg = null

# Previous suspension compressions for damping
var _prev_compressions: Array[float] = [0.0, 0.0, 0.0, 0.0]

func _ready() -> void:
	# Configure RigidBody3D properties
	mass = 300.0
	center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	center_of_mass = Vector3(0.0, 0.33, -0.117)
	
	contact_monitor = true
	max_contacts_reported = 10
	body_entered.connect(_on_body_entered)
	add_to_group("vehicle")
	add_to_group("Car")
	
	# Instantiate Dynamics Subsystems
	config = VehicleConfigScript.new()
	config.drive_mode = drive_mode
	model = VehicleDynamicsModelScript.new(config)
	inputs = VehicleInputsScript.new()
	telemetry = VehicleTelemetryScript.new()
	
	_setup_raycasts_and_wheels()
	cone_collision_set(true)
	_connect_to_initial_gate()
	_init_ros()
	
	# Register Cameras
	var sim_node = get_node_or_null("/root/Sim")
	if sim_node:
		if has_node("TPCam"):
			sim_node.register_camera($TPCam)
		if has_node("FPCam"):
			sim_node.register_camera($FPCam)

func _setup_raycasts_and_wheels() -> void:
	raycasts.clear()
	wheel_mesh_nodes.clear()
	
	var wheel_names = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	var ray_names = ["FL_RAY", "FR_RAY", "RL_RAY", "RR_RAY"]
	
	for i in range(4):
		# Look for or create RayCast3D
		var ray: RayCast3D = null
		if has_node(ray_names[i]):
			ray = get_node(ray_names[i]) as RayCast3D
		else:
			ray = RayCast3D.new()
			ray.name = ray_names[i]
			add_child(ray)
			ray.position = ray_mount_positions[i]
			ray.target_position = Vector3(0.0, -ray_cast_length, 0.0)
			ray.collision_mask = 1 # Ground / Track surface
			ray.add_exception(self)
			ray.enabled = true
		raycasts.append(ray)
		
		# Look for wheel mesh node
		var w_node: Node3D = null
		if has_node(wheel_names[i]):
			w_node = get_node(wheel_names[i]) as Node3D
		wheel_mesh_nodes.append(w_node)

func _init_ros() -> void:
	if not ClassDB.class_exists("RosNode"):
		print("MirenaCar: RosNode GDExtension not loaded, running in standalone mode.")
		_as_info = {"status": 0, "mission": 0}
		_mission_info = {"status": 0}
		return
	
	car_state = RosMirenaCommonCar.new()
	wheel_speeds = RosMirenaCommonWheelSpeeds.new()
	lap_status_msg = RosMirenaCommonLapStatus.new()
	_as_info = RosMirenaCommonAsInfo.new()
	_as_info.status = 0
	_mission_info = RosMirenaCommonMissionInfo.new()
	_mission_info.status = 0
	
	_node = RosNode.new()
	_node.init("mirena_car", "")
	
	_as_info_pub          = _node.create_publisher("/as_status",        "mirena_common/msg/AsInfo")
	_state_pub            = _node.create_publisher("/debug/state/car", "mirena_common/msg/Car")
	_perception_pub       = _node.create_publisher("/debug/perception",  "mirena_common/msg/EntityList")
	_slam_pub             = _node.create_publisher("/debug/slam",        "mirena_common/msg/EntityList")
	_inferred_control_pub = _node.create_publisher("/inferred_control",  "mirena_common/msg/CarControl")
	_lap_status_pub       = _node.create_publisher("/lap_status",        "mirena_common/msg/LapStatus")
	
	var wheel_qos := RosQoS.new()
	wheel_qos.history = RosQoS.KEEP_LAST
	wheel_qos.depth = 5
	wheel_qos.reliability = RosQoS.BEST_EFFORT
	wheel_qos.durability = RosQoS.VOLATILE
	_wheel_speeds_pub = _node.create_publisher("/sensors/wheel_speeds", "mirena_common/msg/WheelSpeeds", wheel_qos)
	
	_debug_tim = _node.create_timer(0.1, _debug_publish)
	
	_control_sub = _node.create_subscription("/control", "mirena_common/msg/CarControl", _on_control)
	_mission_info_sub = _node.create_subscription("/system/mission_status", "mirena_common/msg/MissionInfo", _on_mission_info)
	_as_status_sub = _node.create_subscription("/as_status", "mirena_common/msg/AsInfo", _on_as_status)

# --- Physics Process (Piloting & Telemetry) ---
func _physics_process(delta: float) -> void:
	match pilot:
		PilotMode.NO_PILOT:
			_process_no_pilot()
		PilotMode.MANUAL:
			_process_manual_pilot(delta)
		PilotMode.ROS:
			_process_ros_pilot()
		PilotMode.TRACK_RAIL:
			_process_track_rail(delta)
	
	if global_position.y < -1.0:
		reset_position()
	
	_update_wheel_speeds_ros()
	_publish_car_state()

# --- Physics Integration (Forces & Moments) ---
## Native Godot RigidBody3D force integration callback.
## Executed every physics tick (400 Hz) directly inside physics solver step.
func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:
	var dt = state.step
	if dt <= 0.0:
		return
	
	# 1. Update inputs
	inputs.drive_mode = drive_mode
	inputs.set_from_ros_control(self.gas, self.steering, config.max_steer_angle)
	if self.brake > 0.0:
		inputs.brake = maxf(inputs.brake, clampf(self.brake / 20.0, 0.0, 1.0))
	
	# 2. Drivetrain torques
	var local_v_godot = state.transform.basis.inverse() * state.linear_velocity
	var u_speed = -local_v_godot.z
	var torques = DrivetrainModelScript.compute_torques(inputs, u_speed, config)
	var drive_trqs: Array[float] = torques["drive"]
	var brake_trqs: Array[float] = torques["brake"]
	
	# 3. Raycast Suspension & Contact Dynamics
	var current_compressions: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var normal_forces: Array[float] = [0.0, 0.0, 0.0, 0.0]
	
	# First pass: calculate suspension compression and spring forces
	for i in range(4):
		var ray = raycasts[i]
		ray.force_raycast_update()
		
		if ray.is_colliding():
			var hit_pt = ray.get_collision_point()
			var hit_dist = ray.global_position.distance_to(hit_pt)
			var comp = maxf(0.0, ray_rest_lengths[i] - hit_dist)
			current_compressions[i] = comp
			
			var k_spring = config.spring_stiffness_front if (i == 0 or i == 1) else config.spring_stiffness_rear
			var c_damper = config.damping_compression_front if (i == 0 or i == 1) else config.damping_compression_rear
			
			# Damping velocity
			var comp_vel = (comp - _prev_compressions[i]) / dt
			_prev_compressions[i] = comp
			
			var f_spring = k_spring * comp
			var f_damper = c_damper * comp_vel
			normal_forces[i] = maxf(0.0, f_spring + f_damper)
		else:
			_prev_compressions[i] = 0.0
			normal_forces[i] = 0.0
	
	# Anti-roll bar coupling
	var arb_f = config.arb_stiffness_front * (current_compressions[0] - current_compressions[1])
	var arb_r = config.arb_stiffness_rear * (current_compressions[2] - current_compressions[3])
	normal_forces[0] = maxf(0.0, normal_forces[0] - arb_f)
	normal_forces[1] = maxf(0.0, normal_forces[1] + arb_f)
	normal_forces[2] = maxf(0.0, normal_forces[2] - arb_r)
	normal_forces[3] = maxf(0.0, normal_forces[3] + arb_r)
	
	# 4. Wheel Kinematics, Tire Forces and Force Application
	var steer_rad = inputs.steer_angle
	
	for i in range(4):
		var ray = raycasts[i]
		var w = model.wheels[i]
		var delta_w = steer_rad if (i == 0 or i == 1) else 0.0
		w.is_grounded = ray.is_colliding()
		
		if ray.is_colliding():
			var hit_pt = ray.get_collision_point()
			var hit_norm = ray.get_collision_normal()
			var r_contact_cg = hit_pt - state.transform.origin
			
			# Contact patch velocity in world
			var v_contact_world = state.linear_velocity + state.angular_velocity.cross(r_contact_cg)
			
			# Steered wheel heading unit vectors in Godot frame
			# (+X right, +Y up, -Z forward)
			var fwd_local = Vector3(sin(delta_w), 0.0, -cos(delta_w))
			var lat_local = Vector3(cos(delta_w), 0.0, sin(delta_w)) # Rightward
			
			var fwd_world = state.transform.basis * fwd_local
			var lat_world = state.transform.basis * lat_local
			
			var vx_w = v_contact_world.dot(fwd_world)
			var vy_w = -v_contact_world.dot(lat_world) # In ROS: +vy is left
			
			w.compute_slip(vx_w, vy_w, delta_w, config.slip_regularization_speed, config.wheel_radius, dt)
			w.compute_forces(normal_forces[i], config)
			w.integrate_wheel_spin(drive_trqs[i], brake_trqs[i], dt, config)
			
			# Total force on chassis at contact point:
			# Fx along wheel heading, Fy along left (so -Fy along lat_world), Fz along ground normal
			var F_wheel_world = w.F_x * fwd_world - w.F_y * lat_world + normal_forces[i] * hit_norm
			
			# Apply force at contact point relative to CoG
			state.apply_force(F_wheel_world, r_contact_cg)
		else:
			w.compute_forces(0.0, config)
			w.integrate_wheel_spin(drive_trqs[i], brake_trqs[i], dt, config)
		
		# 5. Visual Wheel Mesh Transform Update
		_update_wheel_visual(i, ray, delta_w, w.rotation_angle)
	
	# 6. Aerodynamic Drag & Downforce
	var speed = state.linear_velocity.length()
	if speed > 0.1:
		var q = 0.5 * config.air_density * speed * speed
		# Drag opposes velocity
		state.apply_central_force(-q * config.cd_a * state.linear_velocity / speed)
		# Downforce acts downward along vehicle local Y
		state.apply_central_force(-q * config.cl_a * state.transform.basis.y)
	
	# 7. Low-speed brake hold to prevent chatter / creeping when stopped
	if inputs.brake > 0.1 and inputs.throttle == 0.0:
		if speed < 0.2:
			state.linear_velocity = Vector3.ZERO
			state.angular_velocity = Vector3.ZERO

func _update_wheel_visual(index: int, ray: RayCast3D, delta_w: float, roll_angle: float) -> void:
	var w_node = wheel_mesh_nodes[index]
	if not w_node:
		return
	
	# Suspension vertical travel
	if ray.is_colliding():
		var hit_pt = ray.get_collision_point()
		var hit_dist = ray.global_position.distance_to(hit_pt)
		w_node.position.y = ray_mount_positions[index].y - hit_dist + config.wheel_radius
	else:
		w_node.position.y = ray_mount_positions[index].y - ray_cast_length + config.wheel_radius
	
	# Steering yaw rotation for front wheels
	if index == 0 or index == 1:
		w_node.rotation.y = delta_w

# --- Pilot Logic ---
func _process_no_pilot() -> void:
	self.gas = 0.0
	self.steering = 0.0
	self.brake = 20.0

func _process_manual_pilot(delta: float) -> void:
	var steer_input = Input.get_action_strength("manual_steer_l") - Input.get_action_strength("manual_steer_r")
	_steer_smoothed = _smooth_steer(_steer_smoothed, steer_input, delta, 2.0)
	
	self.gas = Input.get_action_strength("manual_gas_pos") - Input.get_action_strength("manual_gas_neg")
	self.steering = _steer_smoothed * config.max_steer_angle
	self.brake = Input.get_action_strength("EBS") * 20.0

func _process_ros_pilot() -> void:
	self.gas = _ros_gas
	self.steering = clampf(_ros_steer, -config.max_steer_angle, config.max_steer_angle)

## Helper for programmatic and test driving
func set_control(gas_val: float, steer_val: float, brake_val: float = 0.0) -> void:
	self.gas = gas_val
	self.steering = steer_val
	self.brake = brake_val
	_ros_gas = gas_val
	_ros_steer = steer_val

func _process_track_rail(delta: float) -> void:
	if not path or not path.curve:
		return
	
	_rail_progress += rail_speed * delta
	var local_tf = path.curve.sample_baked_with_rotation(_rail_progress, true)
	var target_tf = path.global_transform * local_tf
	
	var motion = target_tf.origin - global_position
	var collision = move_and_collide(motion)
	if collision:
		var remainder = collision.get_remainder().slide(collision.get_normal())
		move_and_collide(remainder)
	
	var look_ahead_p = _rail_progress + rail_look_ahead
	var look_target = path.curve.sample_baked_with_rotation(look_ahead_p, true)
	var look_global = path.global_transform * look_target
	
	global_transform.basis = global_transform.basis.slerp(look_global.basis.orthonormalized(), 5.0 * delta).orthonormalized()

func _smooth_steer(current: float, target: float, delta: float, speed: float) -> float:
	var diff = target - current
	var t = clampf(absf(diff), 0.0, 1.0)
	var ease_val = t * t * (3.0 - 2.0 * t)
	current += signf(diff) * ease_val * speed * delta
	if signf(target - current) != signf(diff):
		current = target
	return clampf(current, -1.0, 1.0)

# --- ROS Callbacks and Publishing ---
func _on_control(msg) -> void:
	_ros_gas = msg.gas
	_ros_steer = msg.steer_angle

func _on_mission_info(msg) -> void:
	_mission_info = msg

func _on_as_status(msg) -> void:
	_as_info.status = msg.status
	_as_info.mission = msg.mission

func _update_wheel_speeds_ros() -> void:
	if not _node or not _wheel_speeds_pub:
		return
	for i in range(4):
		_wheel_rpm_acc[i] += model.wheels[i].get_rpm()
	_wheel_decim_n += 1
	
	var window = maxi(1, int(round(Engine.physics_ticks_per_second / wheel_speed_rate)))
	if _wheel_decim_n < window:
		return
	
	var dt = float(_wheel_decim_n) / float(Engine.physics_ticks_per_second)
	var alpha = 1.0 - exp(-dt / maxf(wheel_speed_lpf_tau, 1e-6))
	
	for i in range(4):
		var raw_rpm = _wheel_rpm_acc[i] / float(_wheel_decim_n)
		_wheel_rpm_acc[i] = 0.0
		_wheel_rpm_lpf[i] = lerpf(_wheel_rpm_lpf[i], raw_rpm, alpha)
	_wheel_decim_n = 0
	
	wheel_speeds.header.stamp = _node.now()
	wheel_speeds.header.frame_id = frame_id
	wheel_speeds.rl = _quantise_rpm(_wheel_rpm_lpf[2])
	wheel_speeds.rr = _quantise_rpm(_wheel_rpm_lpf[3])
	_wheel_speeds_pub.publish(wheel_speeds)

func _quantise_rpm(rpm: float) -> float:
	if wheel_speed_quantise_rpm <= 0.0:
		return rpm
	return roundf(rpm / wheel_speed_quantise_rpm) * wheel_speed_quantise_rpm

func _publish_car_state() -> void:
	if not _node or not _state_pub:
		return
	
	var now = _node.now()
	car_state.header.stamp = now
	car_state.header.frame_id = "odom"
	car_state.child_frame_id = frame_id
	
	var odom_tf = origin.inverse() * global_transform
	car_state.x = -odom_tf.origin.z
	car_state.y = -odom_tf.origin.x
	car_state.psi = odom_tf.basis.get_euler().y
	
	var local_vel = global_basis.inverse() * linear_velocity
	car_state.u = -local_vel.z
	car_state.v = -local_vel.x
	car_state.omega = angular_velocity.y
	
	var cov = []
	cov.resize(36)
	for i in range(36):
		cov[i] = 0.0
	car_state.covariance = cov
	_state_pub.publish(car_state)

func _debug_publish() -> void:
	_publish_perception()
	_publish_slam()
	_publish_control()

func _publish_perception() -> void:
	var cones = get_cones_in_sight(12.0)
	for cone in cones:
		if not slam_cones.has(cone):
			slam_cones.append(cone)
	
	var msg = RosMirenaCommonEntityList.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = frame_id
	msg.entities = cones.map(func(c): return _to_ent(c))
	_perception_pub.publish(msg)

func _publish_slam() -> void:
	var msg = RosMirenaCommonEntityList.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = "map"
	slam_cones = slam_cones.filter(func(c): return is_instance_valid(c))
	msg.entities = slam_cones.map(func(c): return _to_ent(c, true))
	_slam_pub.publish(msg)

func _publish_control() -> void:
	var msg = RosMirenaCommonCarControl.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = "map"
	msg.gas = self.gas
	msg.steer_angle = self.steering
	_inferred_control_pub.publish(msg)

func _to_ent(cone: Node3D, global: bool = false) -> RosMirenaCommonEntity:
	var ent = RosMirenaCommonEntity.new()
	var pos = cone.global_position if global else to_local(cone.global_position)
	ent.type = cone.get_type_as_string()
	ent.position.x = -pos.z
	ent.position.y = -pos.x
	ent.position.z = pos.y
	return ent

# --- Gate Crossing & Cone Timing ---
func _on_body_entered(body: Node) -> void:
	if body.is_in_group("Cones"):
		hit_cones.append(body)
		cone_hit.emit(hit_cones.size())
		if lap_start_ticks == 0.0:
			return
		var elapsed_ms = Time.get_ticks_msec() - lap_start_ticks
		var lap_time = elapsed_ms / 1000.0
		_publish_lap_status(hit_cones.size(), lap_time)

var _gate_retry_count: int = 0
func _connect_to_initial_gate() -> void:
	var gate = get_tree().get_first_node_in_group("InitialGate")
	if gate:
		gate.vehicle_passed.connect(_on_initial_gate_passed)
	elif _gate_retry_count < 10:
		_gate_retry_count += 1
		call_deferred("_connect_to_initial_gate")

func _on_initial_gate_passed() -> void:
	initial_gate_crossings += 1
	if initial_gate_crossings == 1:
		lap_start_ticks = Time.get_ticks_msec()
	elif initial_gate_crossings >= 2:
		var elapsed_ms = Time.get_ticks_msec() - lap_start_ticks
		var lap_time = elapsed_ms / 1000.0
		_publish_lap_status(hit_cones.size(), lap_time)
		initial_gate_crossings = 1
		lap_start_ticks = Time.get_ticks_msec()

func _publish_lap_status(cones_hit: int, lap_time: float) -> void:
	if not _lap_status_pub:
		return
	lap_status_msg.header.stamp = _node.now()
	lap_status_msg.header.frame_id = "map"
	lap_status_msg.cones_hit = cones_hit
	lap_status_msg.lap_time = lap_time
	_lap_status_pub.publish(lap_status_msg)

# --- Interface & Utility Methods ---
func set_origin(target_transform: Transform3D, reset_vel: bool = false) -> void:
	origin = target_transform
	if reset_vel:
		linear_velocity = Vector3.ZERO
		angular_velocity = Vector3.ZERO
	await get_tree().process_frame
	set_deferred("global_transform", origin)

func reset_position() -> void:
	var sim_node = get_node_or_null("/root/Sim")
	if sim_node and sim_node.get("track"):
		set_origin(sim_node.track.origin, true)
	self.gas = 0.0
	self.steering = 0.0
	self.brake = 0.0
	_rail_progress = 0.0
	slam_cones.clear()
	hit_cones.clear()
	initial_gate_crossings = 0
	lap_start_ticks = 0.0
	cone_hit.emit(0)
	
	model.reset()
	for i in range(4):
		_wheel_rpm_acc[i] = 0.0
		_wheel_rpm_lpf[i] = 0.0
		_prev_compressions[i] = 0.0
	_wheel_decim_n = 0
	
	if has_node("INS") and get_node("INS").has_method("reset"):
		get_node("INS").reset()

func cone_collision_set(enable: bool) -> void:
	collision_layer = (collision_layer & ~2) | (2 * int(enable))
	collision_mask = (collision_mask & ~2) | (2 * int(enable))

func get_cones_in_sight(max_dist: float = 10.0) -> Array:
	var visible_cones: Array = []
	if not has_node("Camera") or not get_node("Camera").has_method("is_position_in_frustum"):
		# Fallback to TPCam or distance check
		for cone in get_tree().get_nodes_in_group("Cones"):
			if global_position.distance_to(cone.global_position) < max_dist:
				visible_cones.append(cone)
		return visible_cones
	
	var camera_node = get_node("Camera")
	var cam = camera_node.get("_camera")
	if not cam:
		return []
	for cone in get_tree().get_nodes_in_group("Cones"):
		if global_position.distance_to(cone.global_position) < max_dist:
			if cam.is_position_in_frustum(cone.global_position):
				visible_cones.append(cone)
	return visible_cones
