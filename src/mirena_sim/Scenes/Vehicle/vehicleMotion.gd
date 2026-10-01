extends VehicleBody3D
class_name MirenaCar

signal cone_hit(count: int)
enum PilotMode { NO_PILOT, MANUAL, ROS, TRACK_RAIL }

# --- Configuration ---
const POWER_LIM = 80000 # W
const MOTOR_PEAK_TRQ = 100 # Nm
const GEAR_RATIO = 5 # 1:5
const WHEEL_RADIUS = 0.23 # m
const BRAKE_F = 20
const MAX_STEER = deg_to_rad(30)
## Forward speed over which regen is at full strength. Below it regen fades out
## and the friction brakes take over, so a braking command can only ever remove
## speed: /control carries a [-1, 1] throttle where -1 is full braking, never
## reverse, and the car must not drive backwards on one.
const REGEN_FADE_SPEED = 0.5 # m/s

# --- Pilot Settings ---
@export var pilot: PilotMode = PilotMode.NO_PILOT
@export var rail_speed: float = 10.0
@export var rail_look_ahead: float = 3.0
var _rail_progress: float = 0.0
var path : Path3D

# --- ROS Variables ---
@export var frame_id: String = "car/cog"
## Rate the rear wheel speeds go out at, in Hz.
@export var wheel_speed_rate: float = 100.0
@export var wheel_speed_lpf_tau: float = 0.05 # Low-pass filter time constant (s)
var _node: RosNode
# Publishers
var _state_pub: RosPublisher
var _as_info_pub : RosPublisher
var _perception_pub: RosPublisher
var _slam_pub: RosPublisher
var _inferred_control_pub: RosPublisher
var _wheel_speeds_pub: RosPublisher
var _lap_status_pub: RosPublisher
var _debug_tim : RosTimer
var _wheel_speeds_tim : RosTimer
var _wheel_rpm_lpf: Array = [0.0, 0.0] # [rl, rr] filtered RPM
# Subscribers
var _control_sub: RosSubscriber
var _mission_info_sub: RosSubscriber
var _as_status_sub: RosSubscriber
var _ros_gas: float = 0.0
var _ros_steer: float = 0.0

# --- Internal State ---
var origin : Transform3D = Transform3D.IDENTITY
var gas: float = 0.0
var _steer_smoothed: float = 0.0
var slam_cones: Array = [] # Seen cones
var hit_cones: Array = [] # Unique struck cones
var _as_info : RosMirenaCommonAsInfo
var _mission_info

var initial_gate_crossings: int = 0
var lap_start_ticks: float = 0.0

@onready var car_state : RosMirenaCommonCar = RosMirenaCommonCar.new()
@onready var wheel_speeds : RosMirenaCommonWheelSpeeds = RosMirenaCommonWheelSpeeds.new()
@onready var lap_status_msg : RosMirenaCommonLapStatus = RosMirenaCommonLapStatus.new()

func _ready():
	# Enable collision contact reporting
	contact_monitor = true
	max_contacts_reported = 4
	body_entered.connect(_on_body_entered)

	# Enable cone collisions by default
	cone_collision_set(true)

	# Add to vehicle group so gates can detect us
	add_to_group("vehicle")

	# Connect to InitialGate's vehicle_passed signal for lap timing
	_connect_to_initial_gate()

	## ROS — no namespace: all topic names below are absolute
	_node = RosNode.new()
	_node.init("mirena_car", "")
	
	## Publishers (absolute topic names from topic contract)
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
	_wheel_speeds_pub = _node.create_publisher(
		"/sensors/wheel_speeds", "mirena_common/msg/WheelSpeeds", wheel_qos)

	# State setup
	_as_info = RosMirenaCommonAsInfo.new()
	_as_info.status = 0
	_mission_info = RosMirenaCommonMissionInfo.new()
	_mission_info.status = 0
	
	## Publisher timers
	_debug_tim = _node.create_timer(0.1, _debug_publish)
	_wheel_speeds_tim = _node.create_timer(1.0 / wheel_speed_rate, _publish_wheel_speeds)
	
	# Subscribers
	_control_sub = _node.create_subscription("/control", "mirena_common/msg/CarControl", _on_control)
	_mission_info_sub = _node.create_subscription("/system/mission_status", "mirena_common/msg/MissionInfo", _on_mission_info)
	_as_status_sub = _node.create_subscription("/as_status", "mirena_common/msg/AsInfo", _on_as_status)

	# Camera Registration
	Sim.register_camera($TPCam)
	Sim.register_camera($FPCam)

func _on_body_entered(body: Node) -> void:
	if body.is_in_group("Cones") and not hit_cones.has(body):
		hit_cones.append(body)
		cone_hit.emit(hit_cones.size())
		var elapsed_ms: float = Time.get_ticks_msec() - lap_start_ticks
		var lap_time: float = elapsed_ms / 1000.0
		var total_cones_hit: int = hit_cones.size()
		_publish_lap_status(total_cones_hit, lap_time)
		

func _connect_to_initial_gate():
	var gate = get_tree().get_first_node_in_group("InitialGate")
	if gate:
		gate.vehicle_passed.connect(_on_initial_gate_passed)
	else:
		# Gate not ready yet, try again next frame
		call_deferred("_connect_to_initial_gate")

func _on_initial_gate_passed():
	initial_gate_crossings += 1

	if initial_gate_crossings == 1:
		# First crossing: start timing
		lap_start_ticks = Time.get_ticks_msec()
		print("Lap started!")

	elif initial_gate_crossings >= 2:
		# Finishing crossing: calculate lap time and publish
		var elapsed_ms: float = Time.get_ticks_msec() - lap_start_ticks
		var lap_time: float = elapsed_ms / 1000.0
		var total_cones_hit: int = hit_cones.size()

		_publish_lap_status(total_cones_hit, lap_time)
		print("Lap finished! Time: %.3f s | Cones hit: %d" % [lap_time, total_cones_hit])

		# Reset to 1 and update start ticks to time subsequent continuous laps
		initial_gate_crossings = 1
		lap_start_ticks = Time.get_ticks_msec()

func _publish_lap_status(cones_hit: int, lap_time: float) -> void:
	if _lap_status_pub == null:
		return

	lap_status_msg.header.stamp = _node.now()
	lap_status_msg.header.frame_id = "map"
	lap_status_msg.cones_hit = cones_hit
	lap_status_msg.lap_time = lap_time

	_lap_status_pub.publish(lap_status_msg)

func _on_control(msg):
	_ros_gas = msg.gas
	_ros_steer = msg.steer_angle
	
func _on_mission_info(msg):
	_mission_info = msg
	
func _physics_process(delta: float) -> void:
	# Process driving commands
	match pilot:
		PilotMode.NO_PILOT:
			_process_no_pilot()
			_apply_vehicle_physics(delta)
		PilotMode.MANUAL:
			_process_manual_pilot(delta)
			_apply_vehicle_physics(delta)
		PilotMode.ROS:
			_process_ros_pilot()
			_apply_vehicle_physics(delta)
		PilotMode.TRACK_RAIL:
			_process_track_rail(delta)
	
	if global_position.y < -1:
		reset_position()

	_publish_car_state()
		
# Publish Debug Info
func _debug_publish():
	_publish_perception()
	_publish_slam()
	_publish_control()

func _on_as_status(msg):
	_as_info.status = msg.status
	_as_info.mission = msg.mission

func _publish_car_state():
	var now = _node.now()
	car_state.header.stamp = now
	car_state.header.frame_id = "odom"
	car_state.child_frame_id = frame_id  # "car/cog"
	
	var odom_transform : Transform3D = origin.inverse() * global_transform
	# 1. Pose and Dynamics
	car_state.x = -odom_transform.origin.z
	car_state.y = -odom_transform.origin.x
	car_state.psi = odom_transform.basis.get_euler().y
	
	var local_vel = _body_velocity()
	car_state.u = -local_vel.z
	car_state.v = -local_vel.x
	car_state.omega = angular_velocity.y
	
	# 2. Covariance Setup
	var cov = []
	cov.resize(36)
	for i in range(36):
		cov[i] = 0.0
		
	cov[0]  = 0
	cov[7]  = 0
	cov[14] = 0
	cov[21] = 0
	cov[28] = 0
	cov[35] = 0
	
	car_state.covariance = cov
	_state_pub.publish(car_state)

func _publish_wheel_speeds():
	wheel_speeds.header.stamp = _node.now()
	wheel_speeds.header.frame_id = frame_id  # "car/cog"
	# Low-pass filter: Godot vehicle physics produces jittery RPM values.
	# Proper exponential moving average: alpha = 1 - exp(-dt/tau).
	var dt := 1.0 / wheel_speed_rate
	var alpha := 1.0 - exp(-dt / wheel_speed_lpf_tau)
	var raw_rl: float = -$RL_WHEEL.get_rpm()
	#var raw_rr: float = -$RR_WHEEL.get_rpm()
	_wheel_rpm_lpf[0] = lerpf(_wheel_rpm_lpf[0], raw_rl, alpha)
	#_wheel_rpm_lpf[1] = lerpf(_wheel_rpm_lpf[1], raw_rr, alpha)
	wheel_speeds.rl = _wheel_rpm_lpf[0]
	wheel_speeds.rr = _wheel_rpm_lpf[1]
	_wheel_speeds_pub.publish(wheel_speeds)

func _publish_perception():
	var cones = get_cones_in_sight(12.0)
	for cone in cones:
		if not slam_cones.has(cone): slam_cones.append(cone)
	
	var msg = RosMirenaCommonEntityList.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = frame_id  # "car/cog"
	msg.entities = cones.map(func(c): return _to_ent(c))
	_perception_pub.publish(msg)

func _publish_slam():
	var msg = RosMirenaCommonEntityList.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = "map"
	slam_cones = slam_cones.filter(func(c): return is_instance_valid(c))
	msg.entities = slam_cones.map(func(c): return _to_ent(c, true))
	_slam_pub.publish(msg)

func _publish_control():
	var msg = RosMirenaCommonCarControl.new()
	msg.header.stamp = _node.now()
	msg.header.frame_id = "map"
	msg.gas = self.gas
	msg.steer_angle = self.steering

	_inferred_control_pub.publish(msg)

# Conversion helper
func _to_ent(cone: Node3D, global : bool = false) -> RosMirenaCommonEntity:
	var ent = RosMirenaCommonEntity.new()
	var pos = cone.global_position if global else to_local(cone.global_position)
	ent.type = cone.get_type_as_string()
	
	# ROS Swizzle: Forward=Z, Left=-X, Up=Y
	ent.position.x = -pos.z
	ent.position.y = -pos.x
	ent.position.z = pos.y
	return ent

# --- Pilot Logic ---

func _process_no_pilot() -> void:
	self.gas = 0.0
	self.steering = 0.0
	self.brake = BRAKE_F

func _process_manual_pilot(delta: float) -> void:
	var steer_input = Input.get_action_strength("manual_steer_l") - Input.get_action_strength("manual_steer_r")
	_steer_smoothed = _smooth_steer(_steer_smoothed, steer_input, delta, 2.0)
	
	self.gas = Input.get_action_strength("manual_gas_pos") - Input.get_action_strength("manual_gas_neg")
	self.steering = _steer_smoothed * MAX_STEER
	self.brake = Input.get_action_strength("EBS") * BRAKE_F

func _process_ros_pilot() -> void:
	self.gas = _ros_gas
	self.steering = _ros_steer 
	self.brake = 0.0

func _process_track_rail(delta: float) -> void:
	if not path or not path.curve: 
		return

	_rail_progress += rail_speed * delta
	var local_transform = path.curve.sample_baked_with_rotation(_rail_progress, true)
	var target_global_transform = path.global_transform * local_transform
	
	var motion = target_global_transform.origin - global_position
	var collision = move_and_collide(motion)
	if collision:
		var remainder = collision.get_remainder().slide(collision.get_normal())
		move_and_collide(remainder)

	var look_ahead_p = _rail_progress + rail_look_ahead
	var look_target_local = path.curve.sample_baked_with_rotation(look_ahead_p, true)
	var look_target_global = path.global_transform * look_target_local
	
	global_transform.basis = global_transform.basis.slerp(
		look_target_global.basis.orthonormalized(), 
		5.0 * delta
	).orthonormalized()

# --- Physics & Low Level Control ---

func _body_velocity() -> Vector3:
	return global_basis.inverse() * linear_velocity

func _apply_vehicle_physics(_delta: float) -> void:
	var u = -_body_velocity().z
	var max_fx_motor = MOTOR_PEAK_TRQ * GEAR_RATIO / WHEEL_RADIUS

	var throttle = max(gas, 0.0)
	var braking = max(-gas, 0.0)
	var regen_scale = clamp(u / REGEN_FADE_SPEED, 0.0, 1.0)
	var fx = (throttle - braking * regen_scale) * max_fx_motor

	$RL_WHEEL.engine_force = fx / 2.0
	#$RR_WHEEL.engine_force = fx / 2.0

	var brake_force = min(brake + braking * (1.0 - regen_scale) * BRAKE_F, BRAKE_F)
	$RL_WHEEL.brake = brake_force
	$RR_WHEEL.brake = brake_force

func _smooth_steer(current: float, target: float, delta: float, speed: float) -> float:
	var diff = target - current
	var t = clamp(abs(diff), 0.0, 1.0)
	var _ease = t * t * (3.0 - 2.0 * t) 
	current += sign(diff) * _ease * speed * delta
	if sign(target - current) != sign(diff): current = target
	return clamp(current, -1.0, 1.0)

# --- Interface & Utility ---

func set_origin(transform, reset_vel: bool = false) -> void:
	origin = transform
	if reset_vel:
		linear_velocity = Vector3.ZERO
		angular_velocity = Vector3.ZERO
	await get_tree().process_frame
	set_deferred("global_transform", origin)

func reset_position() -> void:
	set_origin(Sim.track.origin, true)
	self.gas = 0
	self.steering = 0
	self.brake = 0
	_rail_progress = 0
	slam_cones.clear()
	
	# Reset gate crossing and cone timing tracking state
	hit_cones.clear()
	initial_gate_crossings = 0
	lap_start_ticks = 0.0
	cone_hit.emit(0)

func cone_collision_set(enable: bool) -> void:
	self.collision_layer = (self.collision_layer & ~2) | (2 * int(enable))
	self.collision_mask = (self.collision_mask & ~2) | (2 * int(enable))
	
func get_cones_in_sight(max_dist: float = 10.0) -> Array:
	var visible_cones: Array = []
	var camera = $Camera._camera
	if not camera: return []
	for cone in get_tree().get_nodes_in_group("Cones"):
		if global_position.distance_to(cone.global_position) < max_dist:
			if camera.is_position_in_frustum(cone.global_position):
				visible_cones.append(cone)
	return visible_cones
