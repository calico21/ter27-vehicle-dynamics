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
const GamepadControllerScript = preload("GamepadController.gd")
const TireModelScript = preload("../Tires/TireModel.gd")
const DrivetrainModelScript = preload("../Drivetrain/DrivetrainModel.gd")
const BrakeModelScript = preload("../Brakes/BrakeModel.gd")
const SuspensionModelScript = preload("../Suspension/SuspensionModel.gd")
const AeroModelScript = preload("../Aerodynamics/AeroModel.gd")
const ChassisDynamicsScript = preload("../Chassis/ChassisDynamics.gd")
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
@export var tv_mode: int = 0
@export_enum("LINEAL_4WD", "ECU_TV_V3_4WD") var ecu_driving_mode: int = 1
var tv_yaw_moment_req: float = 0.0
@export var manual_steer_rate: float = 8.0
@export var manual_steer_return_rate: float = 12.0

@export_group("Assetto Corsa Gamepad Settings")
@export_range(0.0, 0.30, 0.01) var steering_deadzone: float = 0.06
@export_range(1.0, 4.0, 0.05) var steering_gamma: float = 2.2
@export_range(0.5, 5.0, 0.05) var steering_speed: float = 1.8
@export_range(0.0, 0.5, 0.01) var steering_filter: float = 0.12
@export_range(0.0, 1.0, 0.01) var speed_sensitivity: float = 0.65
@export_range(1.0, 3.0, 0.05) var throttle_gamma: float = 1.6
@export_range(1.0, 3.0, 0.05) var brake_gamma: float = 1.4
@export_range(0.0, 1.0, 0.05) var rumble_effects: float = 0.7

var gamepad: GamepadControllerScript = GamepadControllerScript.new()
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
var _wheel_base_yaws: Array[float] = [PI, PI, PI, PI]

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
var _ros_brake: float = 0.0
var _wheel_decim_n: int = 0
var _wheel_rpm_acc: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _wheel_rpm_lpf: Array[float] = [0.0, 0.0, 0.0, 0.0]

# --- Internal States ---
var origin: Transform3D = Transform3D.IDENTITY
var gas: float = 0.0
var steering: float = 0.0
var brake: float = 0.0
var _steer_smoothed: float = 0.0
var _joy_vibrating: bool = false

# --- ECU TV v3 Live Telemetry (visible in Godot Remote Inspector) ---
var tv_vy_est: float = 0.0
var tv_beta_est_deg: float = 0.0
var tv_wz_ref: float = 0.0
var tv_mz_req: float = 0.0
var tv_mu_est: Array[float] = [1.5, 1.5, 1.5, 1.5]
var tv_kappa_opt: Array[float] = [0.12, 0.12, 0.12, 0.12]
var tv_torque_4wd: Array[float] = [0.0, 0.0, 0.0, 0.0]

var slam_cones: Array = []
var hit_cones: Array = []
var _as_info = null
var _mission_info = null

var initial_gate_crossings: int = 0
var lap_start_ticks: float = 0.0

var car_state = null
var wheel_speeds = null
var lap_status_msg = null

# Previous suspension compressions (signed delta_z) & contact state for damping
var _prev_compressions: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _was_grounded: Array[bool] = [false, false, false, false]

func _ready() -> void:
	contact_monitor = true
	max_contacts_reported = 10
	body_entered.connect(_on_body_entered)
	add_to_group("vehicle")
	add_to_group("Car")
	
	# Instantiate Dynamics Subsystems (single source of truth from Core/Config/*.json)
	config = VehicleConfigScript.new()
	config.drive_mode = drive_mode
	TireModelScript.enable_neural_correction = false
	model = VehicleDynamicsModelScript.new(config)
	inputs = VehicleInputsScript.new()
	telemetry = VehicleTelemetryScript.new()
	if gamepad.load_from_json():
		steering_deadzone = gamepad.steering_deadzone
		steering_gamma = gamepad.steering_gamma
		steering_speed = gamepad.steering_speed
		steering_filter = gamepad.steering_filter
		speed_sensitivity = gamepad.speed_sensitivity
		throttle_gamma = gamepad.throttle_gamma
		brake_gamma = gamepad.brake_gamma
		rumble_effects = gamepad.rumble_effects
	
	# Configure RigidBody3D properties from VehicleConfig
	mass = config.mass
	inertia = Vector3(config.iyy, config.izz, config.ixx)
	center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	center_of_mass = Vector3(0.0, config.h_cog, -0.117)
	
	_sync_geometry_from_config()
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

func _sync_geometry_from_config() -> void:
	ray_rest_lengths = [
		config.ray_rest_length,
		config.ray_rest_length,
		config.ray_rest_length,
		config.ray_rest_length
	]
	ray_cast_length = config.ray_cast_length
	ray_mount_positions.clear()
	for i in range(4):
		var r_ros: Vector3 = ChassisDynamicsScript.get_wheel_pos_ros(i, config)
		# Convert ISO/ROS horizontal offsets from CoG (+X fwd, +Y left) to Godot local (+X right, -Z fwd)
		var x_godot: float = center_of_mass.x - r_ros.y
		var z_godot: float = center_of_mass.z - r_ros.x
		ray_mount_positions.append(Vector3(x_godot, config.ray_rest_length, z_godot))

func _setup_raycasts_and_wheels() -> void:
	raycasts.clear()
	wheel_mesh_nodes.clear()
	
	var wheel_names = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	var ray_names = ["FL_RAY", "FR_RAY", "RL_RAY", "RR_RAY"]
	
	for i in range(4):
		# Look for or create RayCast3D and align dynamically to VehicleConfig geometry
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
		
		# Look for wheel mesh node and align horizontally without altering .tscn
		var w_node: Node3D = null
		if has_node(wheel_names[i]):
			w_node = get_node(wheel_names[i]) as Node3D
			_wheel_base_yaws[i] = w_node.rotation.y
			w_node.position.x = ray_mount_positions[i].x
			w_node.position.z = ray_mount_positions[i].z
			w_node.position.y = ray_mount_positions[i].y - ray_rest_lengths[i] + config.wheel_radius
		wheel_mesh_nodes.append(w_node)

static func _new_ros_msg(msg_name: String) -> Object:
	var s = load("res://addons/rclgd/gen/%s.gd" % msg_name)
	return s.new() if s != null else null

func _init_ros() -> void:
	if not Engine.has_singleton("rclgd") or not ClassDB.class_exists("RosNode"):
		print("MirenaCar: RosNode GDExtension not loaded, running in standalone Manual mode.")
		_as_info = {"status": 0, "mission": 0}
		_mission_info = {"status": 0}
		pilot = PilotMode.MANUAL
		return
	
	car_state = _new_ros_msg("RosMirenaCommonCar")
	wheel_speeds = _new_ros_msg("RosMirenaCommonWheelSpeeds")
	lap_status_msg = _new_ros_msg("RosMirenaCommonLapStatus")
	_as_info = _new_ros_msg("RosMirenaCommonAsInfo")
	if _as_info:
		_as_info.set("status", 0)
	_mission_info = _new_ros_msg("RosMirenaCommonMissionInfo")
	if _mission_info:
		_mission_info.set("status", 0)
	
	_node = ClassDB.instantiate("RosNode")
	_node.call("init", "mirena_car", "")
	
	_as_info_pub          = _node.call("create_publisher", "/as_status",        "mirena_common/msg/AsInfo")
	_state_pub            = _node.call("create_publisher", "/debug/state/car", "mirena_common/msg/Car")
	_perception_pub       = _node.call("create_publisher", "/debug/perception",  "mirena_common/msg/EntityList")
	_slam_pub             = _node.call("create_publisher", "/debug/slam",        "mirena_common/msg/EntityList")
	_inferred_control_pub = _node.call("create_publisher", "/inferred_control",  "mirena_common/msg/CarControl")
	_lap_status_pub       = _node.call("create_publisher", "/lap_status",        "mirena_common/msg/LapStatus")
	
	var wheel_qos: Object = ClassDB.instantiate("RosQoS")
	if wheel_qos:
		wheel_qos.set("history", ClassDB.class_get_integer_constant("RosQoS", "KEEP_LAST"))
		wheel_qos.set("depth", 5)
		wheel_qos.set("reliability", ClassDB.class_get_integer_constant("RosQoS", "BEST_EFFORT"))
		wheel_qos.set("durability", ClassDB.class_get_integer_constant("RosQoS", "VOLATILE"))
	_wheel_speeds_pub = _node.call("create_publisher", "/sensors/wheel_speeds", "mirena_common/msg/WheelSpeeds", wheel_qos)
	
	_debug_tim = _node.call("create_timer", 0.1, _debug_publish)
	
	_control_sub = _node.call("create_subscription", "/control", "mirena_common/msg/CarControl", _on_control)
	_mission_info_sub = _node.call("create_subscription", "/system/mission_status", "mirena_common/msg/MissionInfo", _on_mission_info)
	_as_status_sub = _node.call("create_subscription", "/as_status", "mirena_common/msg/AsInfo", _on_as_status)

# --- Physics Process (Piloting & Telemetry) ---
func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F8:
		ecu_driving_mode = 1 - ecu_driving_mode
		var mode_name: String = "ECU_TV_V3_4WD" if ecu_driving_mode == 1 else "LINEAL_4WD"
		print("[ECU Mode]: %s" % mode_name)

func _physics_process(delta: float) -> void:
	if pilot != PilotMode.MANUAL and _joy_vibrating:
		_stop_all_joy_vibration()
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
## Executed every physics tick using actual physics step delta (dt = state.step).
func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:
	var dt = state.step
	if dt <= 0.0:
		return
	
	# 1. Update inputs (unified longitudinal mapping + normalized [0, 1] brake + TV command)
	inputs.drive_mode = drive_mode
	inputs.tv_mode = tv_mode
	inputs.ecu_driving_mode = ecu_driving_mode
	inputs.tv_yaw_moment_req = tv_yaw_moment_req
	inputs.set_unified_control(self.gas, self.steering, self.brake, config.max_steer_angle)
	
	# 2. Local chassis velocities in ISO/ROS frame (+X fwd, +Y left, +Z up)
	var inv_basis: Basis = state.transform.basis.inverse()
	var local_v_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_basis * state.linear_velocity)
	var local_w_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_basis * state.angular_velocity)
	
	# 3. Raycast Suspension & Contact Geometry (signed delta_z: >0 jounce, 0 design static, <0 droop)
	var delta_z: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var delta_z_dot: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var hit_pts: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO, Vector3.ZERO, Vector3.ZERO]
	var hit_norms: Array[Vector3] = [Vector3.UP, Vector3.UP, Vector3.UP, Vector3.UP]
	var grounded: Array[bool] = [false, false, false, false]
	
	for i in range(4):
		var ray = raycasts[i]
		ray.force_raycast_update()
		
		if ray.is_colliding():
			grounded[i] = true
			var hit_pt = ray.get_collision_point()
			hit_pts[i] = hit_pt
			hit_norms[i] = ray.get_collision_normal()
			var hit_dist = ray.global_position.distance_to(hit_pt)
			var dz = ray_rest_lengths[i] - hit_dist
			delta_z[i] = dz
			delta_z_dot[i] = (dz - _prev_compressions[i]) / dt if _was_grounded[i] else 0.0
			_prev_compressions[i] = dz
			_was_grounded[i] = true
		else:
			_prev_compressions[i] = ray_rest_lengths[i] - ray_cast_length
			_was_grounded[i] = false
			delta_z[i] = _prev_compressions[i]
			delta_z_dot[i] = 0.0
	
	# 4. Build ground contacts in ISO/ROS frame and delegate suspension + tire physics to VehicleDynamicsModel
	var ground_contacts: Array = []
	for i in range(4):
		var r_pos_ros: Vector3 = model.get_wheel_pos_ros(i)
		if grounded[i]:
			var r_from_com_world: Vector3 = hit_pts[i] - (state.transform.origin + state.center_of_mass)
			r_pos_ros = VehicleConfigScript.godot_to_ros_vec(inv_basis * r_from_com_world)
		ground_contacts.append({
			"grounded": grounded[i],
			"delta_z": delta_z[i],
			"delta_z_dot": delta_z_dot[i],
			"r_pos_ros": r_pos_ros
		})
	
	var step_res: Dictionary = model.step_forces(inputs, local_v_ros, local_w_ros, ground_contacts, dt)
	if model.ecu_tv_v3 != null:
		tv_vy_est = model.ecu_tv_v3.ekf_x[0]
		tv_beta_est_deg = rad_to_deg(model.ecu_tv_v3.ekf_beta_est)
		tv_wz_ref = model.ecu_tv_v3.last_wz_ref
		tv_mz_req = model.ecu_tv_v3.last_mz_req
		tv_mu_est = model.ecu_tv_v3.tc_mu_surface.duplicate()
		tv_kappa_opt = model.ecu_tv_v3.tc_kappa_opt.duplicate()
		tv_torque_4wd = step_res["drive_torques"].duplicate()
	var normal_forces: Array = step_res["normal_forces"]
	var wheel_forces_ros: Array = step_res["wheel_forces_ros"]
	var wheel_moments_ros: Array = step_res["wheel_moments_ros"]
	var wheel_steer_angles: Array = step_res["wheel_steer_angles"]
	
	# 5. Apply wheel tire forces, ground normal forces, and tire moments (Mz, Mx) to RigidBody3D
	for i in range(4):
		var ray = raycasts[i]
		var w = model.wheels[i]
		var delta_w: float = float(wheel_steer_angles[i])
		
		if grounded[i]:
			var f_ros: Vector3 = wheel_forces_ros[i]
			var f_planar_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(Vector3(f_ros.x, f_ros.y, 0.0))
			var f_wheel_world: Vector3 = state.transform.basis * f_planar_godot_local + float(normal_forces[i]) * hit_norms[i]
			var r_from_origin_world: Vector3 = hit_pts[i] - state.transform.origin
			state.apply_force(f_wheel_world, r_from_origin_world)
			
			var m_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(wheel_moments_ros[i])
			state.apply_torque(state.transform.basis * m_godot_local)
		
		# Visual Wheel Mesh Transform Update
		_update_wheel_visual(i, ray, delta_w, w.rotation_angle)
	
	# 6. Aerodynamic Drag & Downforce
	var f_aero_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(step_res["aero_force_ros"])
	var tau_aero_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(step_res["aero_torque_ros"])
	state.apply_central_force(state.transform.basis * f_aero_godot_local)
	if not tau_aero_godot_local.is_zero_approx():
		state.apply_torque(state.transform.basis * tau_aero_godot_local)
	
	# 7. Low-speed brake hold to prevent chatter / creeping when stopped
	if inputs.brake > 0.1 and inputs.throttle == 0.0:
		if state.linear_velocity.length() < 0.2:
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
	
	# Steering yaw rotation for front wheels (+delta_w = CCW / left turn)
	if index == 0 or index == 1:
		w_node.rotation.y = _wheel_base_yaws[index] + delta_w

# --- Pilot Logic ---
func _process_no_pilot() -> void:
	self.gas = 0.0
	self.steering = 0.0
	self.brake = 1.0

func _sync_gamepad_settings() -> void:
	gamepad.steering_deadzone = steering_deadzone
	gamepad.steering_gamma = steering_gamma
	gamepad.steering_speed = steering_speed
	gamepad.steering_filter = steering_filter
	gamepad.speed_sensitivity = speed_sensitivity
	gamepad.throttle_gamma = throttle_gamma
	gamepad.brake_gamma = brake_gamma
	gamepad.rumble_effects = rumble_effects

func _stop_all_joy_vibration() -> void:
	for joy_id in Input.get_connected_joypads():
		Input.stop_joy_vibration(joy_id)
	_joy_vibrating = false

func _notification(what: int) -> void:
	if what == NOTIFICATION_PAUSED or what == NOTIFICATION_EXIT_TREE or what == NOTIFICATION_APPLICATION_FOCUS_OUT:
		if _joy_vibrating:
			_stop_all_joy_vibration()

func _process_manual_pilot(delta: float) -> void:
	_sync_gamepad_settings()
	var joypads: Array[int] = Input.get_connected_joypads()
	var has_joypad: bool = joypads.size() > 0
	var kb_steer: float = Input.get_action_strength("manual_steer_l") - Input.get_action_strength("manual_steer_r")
	var kb_throttle: float = clampf(Input.get_action_strength("manual_gas_pos"), 0.0, 1.0)
	var kb_brake: float = clampf(Input.get_action_strength("manual_gas_neg"), 0.0, 1.0)
	var ebs_in: float = clampf(Input.get_action_strength("EBS"), 0.0, 1.0)

	if has_joypad:
		var joy_id: int = joypads[0]
		var raw_x: float = -Input.get_joy_axis(joy_id, JOY_AXIS_LEFT_X)
		var vx_mps: float = absf((global_transform.basis.inverse() * linear_velocity).z)
		if absf(raw_x) > gamepad.steering_deadzone or is_zero_approx(kb_steer):
			_steer_smoothed = gamepad.step_steering(raw_x, vx_mps, delta)
		else:
			var steer_rate = manual_steer_return_rate if (absf(kb_steer) < absf(_steer_smoothed) or kb_steer * _steer_smoothed < 0.0) else manual_steer_rate
			_steer_smoothed = _smooth_steer(_steer_smoothed, kb_steer, delta, steer_rate)

		var joy_throttle: float = gamepad.normalize_trigger(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_RIGHT), false)
		var joy_brake: float = gamepad.normalize_trigger(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_LEFT), true)
		self.gas = gamepad.shape_throttle(maxf(kb_throttle, joy_throttle))
		self.brake = maxf(gamepad.shape_brake(maxf(kb_brake, joy_brake)), ebs_in)

		if model != null:
			var rumble: Vector2 = gamepad.compute_rumble(model.wheels, model.suspension.corners, linear_velocity.length(), delta)
			if rumble.x > 0.01 or rumble.y > 0.01:
				Input.start_joy_vibration(joy_id, rumble.x, rumble.y, 0.05)
				_joy_vibrating = true
			elif _joy_vibrating:
				_stop_all_joy_vibration()
	else:
		var steer_rate = manual_steer_return_rate if (absf(kb_steer) < absf(_steer_smoothed) or kb_steer * _steer_smoothed < 0.0) else manual_steer_rate
		_steer_smoothed = _smooth_steer(_steer_smoothed, kb_steer, delta, steer_rate)
		self.gas = kb_throttle
		self.brake = maxf(kb_brake, ebs_in)
		if _joy_vibrating:
			_stop_all_joy_vibration()

	self.steering = _steer_smoothed * config.max_steer_angle

func _process_ros_pilot() -> void:
	self.gas = _ros_gas
	self.steering = clampf(_ros_steer, -config.max_steer_angle, config.max_steer_angle)
	self.brake = _ros_brake

## Helper for programmatic and test driving
func set_control(gas_val: float, steer_val: float, brake_val: float = 0.0) -> void:
	self.gas = gas_val
	self.steering = steer_val
	self.brake = clampf(brake_val, 0.0, 1.0)
	_ros_gas = gas_val
	_ros_steer = steer_val
	_ros_brake = self.brake

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
	if delta <= 0.0:
		return current
	return clampf(move_toward(current, clampf(target, -1.0, 1.0), speed * delta), -1.0, 1.0)

# --- ROS Callbacks and Publishing ---
func _on_control(msg) -> void:
	_ros_gas = msg.gas
	_ros_steer = msg.steer_angle
	_ros_brake = 0.0

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
	if not _node or not _perception_pub:
		return
	var cones = get_cones_in_sight(12.0)
	for cone in cones:
		if not slam_cones.has(cone):
			slam_cones.append(cone)
	
	var msg: Object = _new_ros_msg("RosMirenaCommonEntityList")
	if not msg:
		return
	msg.get("header").set("stamp", _node.call("now"))
	msg.get("header").set("frame_id", frame_id)
	msg.set("entities", cones.map(func(c): return _to_ent(c)))
	_perception_pub.call("publish", msg)

func _publish_slam() -> void:
	if not _node or not _slam_pub:
		return
	var msg: Object = _new_ros_msg("RosMirenaCommonEntityList")
	if not msg:
		return
	msg.get("header").set("stamp", _node.call("now"))
	msg.get("header").set("frame_id", "map")
	slam_cones = slam_cones.filter(func(c): return is_instance_valid(c))
	msg.set("entities", slam_cones.map(func(c): return _to_ent(c, true)))
	_slam_pub.call("publish", msg)

func _publish_control() -> void:
	if not _node or not _inferred_control_pub:
		return
	var msg: Object = _new_ros_msg("RosMirenaCommonCarControl")
	if not msg:
		return
	msg.get("header").set("stamp", _node.call("now"))
	msg.get("header").set("frame_id", "map")
	msg.set("gas", self.gas)
	msg.set("steer_angle", self.steering)
	_inferred_control_pub.call("publish", msg)

func _to_ent(cone: Node3D, global: bool = false) -> Object:
	var ent: Object = _new_ros_msg("RosMirenaCommonEntity")
	if not ent:
		return null
	var pos = cone.global_position if global else to_local(cone.global_position)
	ent.set("type", cone.get_type_as_string())
	var ent_pos = ent.get("position")
	ent_pos.set("x", -pos.z)
	ent_pos.set("y", -pos.x)
	ent_pos.set("z", pos.y)
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
	_steer_smoothed = 0.0
	gamepad.reset()
	if _joy_vibrating:
		_stop_all_joy_vibration()
	_ros_gas = 0.0
	_ros_steer = 0.0
	_ros_brake = 0.0
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
		_was_grounded[i] = false
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
