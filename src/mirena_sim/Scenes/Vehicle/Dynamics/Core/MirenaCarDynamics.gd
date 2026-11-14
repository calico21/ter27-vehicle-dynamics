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
const WheelFFBControllerScript = preload("WheelFFBController.gd")
const TireModelScript = preload("../Tires/TireModel.gd")
const DrivetrainModelScript = preload("../Drivetrain/DrivetrainModel.gd")
const BrakeModelScript = preload("../Brakes/BrakeModel.gd")
const SuspensionModelScript = preload("../Suspension/SuspensionModel.gd")
const AeroModelScript = preload("../Aerodynamics/AeroModel.gd")
const ChassisDynamicsScript = preload("../Chassis/ChassisDynamics.gd")
const VehicleTelemetryScript = preload("../Telemetry/VehicleTelemetry.gd")
const TelemetryRecorderScript = preload("../Telemetry/TelemetryRecorder.gd")
const TvLiveTuningOverlayScript = preload("../../../UI/TvLiveTuningOverlay.gd")
const TelemetryInspectorOverlayScript = preload("../../../UI/TelemetryInspectorOverlay.gd")

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
@export_range(1.0, 4.0, 0.05) var steering_gamma: float = 2.0
@export_range(0.5, 5.0, 0.05) var steering_speed: float = 1.8
@export_range(0.0, 0.5, 0.01) var steering_filter: float = 0.12
@export_range(0.0, 1.0, 0.01) var speed_sensitivity: float = 0.25
@export_range(12.0, 35.0, 0.5) var v_ref_mps: float = 28.0
@export_range(1.0, 3.0, 0.05) var throttle_gamma: float = 1.25
@export_range(1.0, 3.0, 0.05) var brake_gamma: float = 1.4
@export_range(0.0, 1.0, 0.05) var rumble_effects: float = 0.7

var gamepad: GamepadControllerScript = GamepadControllerScript.new()
var wheel_ffb: WheelFFBControllerScript = WheelFFBControllerScript.new()
var telemetry_recorder: TelemetryRecorderScript = TelemetryRecorderScript.new()
@export var input_device_mode: WheelFFBControllerScript.InputDeviceMode = WheelFFBControllerScript.InputDeviceMode.AUTO
@export var tv_hardware_realism: bool = false
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

# --- 3D Wheel Force & Saturation Arrows ---
var wheel_arrows_enabled: bool = false
var _wheel_arrow_roots: Array[Node3D] = []
var _wheel_arrow_shafts: Array[Node3D] = []
var _wheel_arrow_tips: Array[Node3D] = []
var _wheel_arrow_mats: Array[StandardMaterial3D] = []

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
var _raw_steer_in: float = 0.0
var _raw_throttle_in: float = 0.0
var _raw_brake_in: float = 0.0
var _joy_vibrating: bool = false

# --- ECU TV v3 Live Telemetry (visible in Godot Remote Inspector) ---
var tv_vy_est: float = 0.0
var tv_beta_est_deg: float = 0.0
var tv_wz_ref: float = 0.0
var tv_mz_req: float = 0.0
var tv_mu_est: Array[float] = [1.5, 1.5, 1.5, 1.5]
var tv_kappa_opt: Array[float] = [0.12, 0.12, 0.12, 0.12]
var tv_torque_4wd: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tv_elec_power_kw: float = 0.0
var tv_tc_active: bool = false
var tv_map_id: int = 2
var tv_map_name: String = "AUTOCROSS_NEUTRAL"
var ffb_torque_nm: float = 0.0

var _hud_layer: CanvasLayer = null
var _hud_label: Label = null
var _tv_overlay: TvLiveTuningOverlayScript = null
var _telemetry_inspector: TelemetryInspectorOverlayScript = null
var _last_step_res: Dictionary = {}

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
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(tv_hardware_realism)
		tv_map_id = model.ecu_tv_v3.active_map_id
		tv_map_name = model.ecu_tv_v3.get_map_name()
		inputs.tv_map_id = tv_map_id
	wheel_ffb.input_device_override = input_device_mode
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
	inertia = Vector3(config.iyy, config.get_total_izz(), config.ixx)
	center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	center_of_mass = Vector3(0.0, config.h_cog, -0.117)
	
	_sync_geometry_from_config()
	_setup_raycasts_and_wheels()
	cone_collision_set(true)
	_connect_to_initial_gate()
	_init_ros()
	_setup_minimal_hud()
	
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
	_setup_wheel_force_arrows()

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
func _get_active_track_name() -> String:
	var sim_node = get_node_or_null("/root/Sim")
	if sim_node:
		var trk = sim_node.get("track")
		if trk != null:
			var t_name = trk.get("track_name")
			if t_name != null and String(t_name).strip_edges() != "":
				return String(t_name)
			if trk is Node:
				return trk.name
	return "manual"

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_F7:
			ecu_driving_mode = 1 - ecu_driving_mode
			var mode_name: String = "ECU_TV_V3_4WD" if ecu_driving_mode == 1 else "LINEAL_4WD"
			print("[ECU Mode]: %s" % mode_name)
			_update_minimal_hud()
		elif event.keycode == KEY_F6:
			var overlay_typing := (
				(_tv_overlay != null and _tv_overlay.has_method("is_text_input_focused") and _tv_overlay.is_text_input_focused())
				or (_telemetry_inspector != null and _telemetry_inspector.has_method("is_text_input_focused") and _telemetry_inspector.is_text_input_focused())
			)
			if overlay_typing:
				return
			if model != null and model.ecu_tv_v3 != null:
				tv_map_id = model.ecu_tv_v3.cycle_map()
				tv_map_name = model.ecu_tv_v3.active_map_filename if model.ecu_tv_v3.active_map_filename != "" else model.ecu_tv_v3.get_map_name()
				if inputs != null:
					inputs.tv_map_id = tv_map_id
				print("[TV Map]: %d (%s)" % [tv_map_id, tv_map_name])
				if _tv_overlay != null:
					_tv_overlay.sync_from_ecu()
				_update_minimal_hud()
		elif event.keycode == KEY_F8:
			toggle_wheel_arrows()
		elif event.keycode == KEY_F10:
			if _tv_overlay != null:
				_tv_overlay.toggle_visibility()
		elif event.keycode == KEY_F11:
			if _telemetry_inspector != null:
				_telemetry_inspector.toggle_visibility()
		elif event.keycode == KEY_F12:
			if _telemetry_inspector != null and _telemetry_inspector.visible:
				_telemetry_inspector.toggle_freeze()
		elif event.keycode == KEY_F9:
			if telemetry_recorder != null:
				var rec_map_name := tv_map_name
				if model != null and model.ecu_tv_v3 != null and model.ecu_tv_v3.active_map_filename != "":
					rec_map_name = model.ecu_tv_v3.active_map_filename
				var rec_on: bool = telemetry_recorder.toggle_recording(_get_active_track_name(), rec_map_name)
				if rec_on:
					print("[Telemetry REC ON]: %s" % telemetry_recorder.get_last_file_path())
				else:
					print("[Telemetry REC OFF]: %d samples saved -> %s" % [
						telemetry_recorder.get_sample_count(),
						telemetry_recorder.get_last_file_path()
					])
				_update_minimal_hud()

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
	_update_minimal_hud()

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
	inputs.tv_map_id = tv_map_id
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
	var track_node: Node = null
	var sim_root: Node = get_node_or_null("/root/Sim")
	if sim_root != null and "track" in sim_root and sim_root.get("track") != null:
		track_node = sim_root.get("track")
	elif get_parent() != null and get_parent().has_node("Track"):
		track_node = get_parent().get_node("Track")
	var ground_contacts: Array = []
	for i in range(4):
		var r_pos_ros: Vector3 = model.get_wheel_pos_ros(i)
		var surface_mu_i: float = 1.0
		if grounded[i]:
			var r_from_com_world: Vector3 = hit_pts[i] - (state.transform.origin + state.center_of_mass)
			r_pos_ros = VehicleConfigScript.godot_to_ros_vec(inv_basis * r_from_com_world)
			if track_node != null and track_node.has_method("get_surface_mu_at"):
				surface_mu_i = float(track_node.call("get_surface_mu_at", hit_pts[i]))
		ground_contacts.append({
			"grounded": grounded[i],
			"delta_z": delta_z[i],
			"delta_z_dot": delta_z_dot[i],
			"r_pos_ros": r_pos_ros,
			"surface_mu": surface_mu_i
		})
	
	var step_res: Dictionary = model.step_forces(inputs, local_v_ros, local_w_ros, ground_contacts, dt)
	_last_step_res = step_res
	if model.ecu_tv_v3 != null:
		tv_vy_est = model.ecu_tv_v3.ekf_x[0]
		tv_beta_est_deg = rad_to_deg(model.ecu_tv_v3.ekf_beta_est)
		tv_wz_ref = model.ecu_tv_v3.last_wz_ref
		tv_mz_req = model.ecu_tv_v3.last_mz_req
		tv_mu_est = model.ecu_tv_v3.tc_mu_surface.duplicate()
		tv_kappa_opt = model.ecu_tv_v3.tc_kappa_opt.duplicate()
		tv_torque_4wd = step_res["drive_torques"].duplicate()
		tv_tc_active = model.ecu_tv_v3.tc_active and ecu_driving_mode == 1
		tv_map_id = model.ecu_tv_v3.active_map_id
		tv_map_name = model.ecu_tv_v3.get_map_name()
	if model.hv_state != null:
		tv_elec_power_kw = model.hv_state.power_kw
	else:
		var eff: float = maxf(config.drivetrain_efficiency, 1e-3)
		var p_elec_w: float = 0.0
		var d_trqs: Array = step_res["drive_torques"]
		for i in range(4):
			p_elec_w += maxf(0.0, float(d_trqs[i]) * absf(model.wheels[i].omega)) / eff
		tv_elec_power_kw = p_elec_w * 0.001

	if wheel_ffb != null:
		ffb_torque_nm = wheel_ffb.compute_steering_torque_nm(
			model.wheels, wheel_ffb.rack_ratio, wheel_ffb.mechanical_trail, wheel_ffb.scrub_radius
		)
	var inv_mass: float = 1.0 / maxf(config.mass, 1e-3)
	var accel_ros: Vector3 = model.net_force_chassis_ros * inv_mass
	if telemetry != null:
		telemetry.update_from_model(model, accel_ros, inputs.steer_angle, {"dt": dt, "forces": step_res})
	if telemetry_recorder != null and telemetry_recorder.is_recording:
		telemetry_recorder.record_step(dt, model, inputs, step_res, {
			"ax": accel_ros.x,
			"ay": accel_ros.y,
			"ffb_torque_nm": ffb_torque_nm
		})
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

	if wheel_arrows_enabled:
		_update_wheel_force_arrows(wheel_forces_ros, normal_forces, grounded)
	
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

func _setup_wheel_force_arrows() -> void:
	for n in _wheel_arrow_roots:
		if is_instance_valid(n):
			n.queue_free()
	_wheel_arrow_roots.clear()
	_wheel_arrow_shafts.clear()
	_wheel_arrow_tips.clear()
	_wheel_arrow_mats.clear()

	var shaft_mesh := CylinderMesh.new()
	shaft_mesh.top_radius = 0.024
	shaft_mesh.bottom_radius = 0.024
	shaft_mesh.height = 1.0
	shaft_mesh.radial_segments = 10

	var tip_mesh := CylinderMesh.new()
	tip_mesh.top_radius = 0.0
	tip_mesh.bottom_radius = 0.062
	tip_mesh.height = 0.18
	tip_mesh.radial_segments = 12

	for i in range(4):
		var root := Node3D.new()
		root.name = "WheelForceArrow_%d" % i
		root.visible = wheel_arrows_enabled
		if i < ray_mount_positions.size():
			root.position = Vector3(ray_mount_positions[i].x, 0.06, ray_mount_positions[i].z)
		add_child(root)

		var mat := StandardMaterial3D.new()
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.albedo_color = Color(0.0627, 0.7255, 0.5059)
		mat.emission_enabled = true
		mat.emission = mat.albedo_color
		mat.emission_energy_multiplier = 1.35

		var shaft_pivot := Node3D.new()
		root.add_child(shaft_pivot)
		var shaft_mi := MeshInstance3D.new()
		shaft_mi.mesh = shaft_mesh
		shaft_mi.material_override = mat
		shaft_mi.rotation.x = -PI * 0.5
		shaft_mi.position = Vector3(0.0, 0.0, -0.5)
		shaft_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		shaft_pivot.add_child(shaft_mi)

		var tip_mi := MeshInstance3D.new()
		tip_mi.mesh = tip_mesh
		tip_mi.material_override = mat
		tip_mi.rotation.x = -PI * 0.5
		tip_mi.position = Vector3(0.0, 0.0, -1.09)
		tip_mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		root.add_child(tip_mi)

		_wheel_arrow_roots.append(root)
		_wheel_arrow_shafts.append(shaft_pivot)
		_wheel_arrow_tips.append(tip_mi)
		_wheel_arrow_mats.append(mat)

func set_wheel_arrows_enabled(enabled: bool) -> void:
	wheel_arrows_enabled = enabled
	if enabled and _wheel_arrow_roots.is_empty():
		_setup_wheel_force_arrows()
	for root in _wheel_arrow_roots:
		if is_instance_valid(root):
			root.visible = enabled
	if _telemetry_inspector != null and _telemetry_inspector.has_method("sync_wheel_arrows_button"):
		_telemetry_inspector.sync_wheel_arrows_button(wheel_arrows_enabled)

func toggle_wheel_arrows() -> bool:
	set_wheel_arrows_enabled(not wheel_arrows_enabled)
	return wheel_arrows_enabled

static func compute_wheel_saturation(fx: float, fy: float, fz: float, mu_eff: float) -> float:
	var f_res := sqrt(fx * fx + fy * fy)
	var f_cap := maxf(mu_eff * fz, 1.0)
	return clampf(f_res / f_cap, 0.0, 1.2)

static func saturation_to_color(eta: float) -> Color:
	var col_green := Color(0.0627, 0.7255, 0.5059) # #10B981
	var col_amber := Color(0.9608, 0.6196, 0.0431) # #F59E0B
	var col_red := Color(0.9373, 0.2667, 0.2667)   # #EF4444
	if eta <= 0.55:
		return col_green
	if eta <= 0.80:
		return col_green.lerp(col_amber, (eta - 0.55) / 0.25)
	if eta < 0.98:
		return col_amber.lerp(col_red, (eta - 0.80) / 0.18)
	return col_red

func _update_wheel_force_arrows(wheel_forces_ros: Array, normal_forces: Array, grounded: Array) -> void:
	if not wheel_arrows_enabled:
		return
	if _wheel_arrow_roots.size() < 4:
		_setup_wheel_force_arrows()
	for i in range(4):
		var root := _wheel_arrow_roots[i]
		if not is_instance_valid(root):
			continue
		var is_g: bool = bool(grounded[i]) if i < grounded.size() else true
		if not is_g or i >= wheel_forces_ros.size():
			root.visible = false
			continue
		var f_ros: Vector3 = wheel_forces_ros[i]
		var f_planar_godot: Vector3 = VehicleConfigScript.ros_to_godot_vec(Vector3(f_ros.x, f_ros.y, 0.0))
		var f_res := sqrt(f_ros.x * f_ros.x + f_ros.y * f_ros.y)
		if f_res < 15.0:
			root.visible = false
			continue

		root.visible = true
		var y_pos := 0.06
		if i < wheel_mesh_nodes.size() and wheel_mesh_nodes[i] != null:
			y_pos = wheel_mesh_nodes[i].position.y - config.wheel_radius + 0.06
		if i < ray_mount_positions.size():
			root.position = Vector3(ray_mount_positions[i].x, y_pos, ray_mount_positions[i].z)
		root.rotation.y = atan2(-f_planar_godot.x, -f_planar_godot.z)

		var total_len := clampf(f_res / 1800.0, 0.18, 2.2)
		var shaft_len := maxf(0.05, total_len - 0.18)
		_wheel_arrow_shafts[i].scale = Vector3(1.0, 1.0, shaft_len)
		_wheel_arrow_tips[i].position = Vector3(0.0, 0.0, -(shaft_len + 0.09))

		var fz: float = float(normal_forces[i]) if i < normal_forces.size() else 0.0
		var mu_eff: float = 1.5 * (float(model.wheels[i].surface_mu_scale) if (model != null and i < model.wheels.size()) else 1.0)
		var eta := compute_wheel_saturation(f_ros.x, f_ros.y, fz, mu_eff)
		var col := saturation_to_color(eta)
		var mat := _wheel_arrow_mats[i]
		mat.albedo_color = col
		mat.emission = col

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
	gamepad.v_ref_mps = v_ref_mps
	gamepad.throttle_gamma = throttle_gamma
	gamepad.brake_gamma = brake_gamma
	gamepad.rumble_effects = rumble_effects

func get_live_gamepad_params() -> Dictionary:
	return gamepad.get_live_gamepad_params()

func set_live_gamepad_param(key: String, val: float) -> void:
	gamepad.set_live_gamepad_param(key, val)
	match key:
		"steering_deadzone":
			steering_deadzone = gamepad.steering_deadzone
		"steering_gamma":
			steering_gamma = gamepad.steering_gamma
		"steering_speed":
			steering_speed = gamepad.steering_speed
		"steering_filter":
			steering_filter = gamepad.steering_filter
		"speed_sensitivity":
			speed_sensitivity = gamepad.speed_sensitivity
		"v_ref_mps":
			v_ref_mps = gamepad.v_ref_mps
		"throttle_gamma":
			throttle_gamma = gamepad.throttle_gamma
		"brake_gamma":
			brake_gamma = gamepad.brake_gamma
		"rumble_effects":
			rumble_effects = gamepad.rumble_effects

func save_gamepad_config_to_json(path: String = "res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json") -> bool:
	return gamepad.save_gamepad_config_to_json(path)

func get_setup_params() -> Dictionary:
	if config == null:
		return {}
	return config.get_setup_params()

func apply_live_setup_param(key: String, val: Variant) -> void:
	if key == "pinn_tire_enabled":
		TireModelScript.enable_neural_correction = bool(val)
		return
	if config == null:
		return
	config.apply_setup_params({key: val})
	_sync_setup_to_model()

func apply_setup_preset(params: Dictionary) -> void:
	if config == null:
		return
	config.apply_setup_params(params)
	_sync_setup_to_model()

func _sync_setup_to_model() -> void:
	if config == null:
		return
	if model != null and model.suspension != null:
		model.suspension.sync_from_config(config)
	mass = config.mass
	inertia = Vector3(config.iyy, config.get_total_izz(), config.ixx)

func _collect_imo_temps() -> Array:
	var out: Array = []
	if model == null:
		return out
	for i in range(mini(4, model.wheels.size())):
		var w = model.wheels[i]
		out.append([w.t_zone_inner_c, w.t_zone_mid_c, w.t_zone_outer_c])
	return out

func _collect_disc_temps() -> Array[float]:
	var amb: float = config.disc_ambient_temp_c if config != null else 40.0
	var out: Array[float] = [amb, amb, amb, amb]
	for k in ["disc_temps_c", "disc_temps", "disc_temp", "brake_disc_temps"]:
		if _last_step_res.has(k) and _last_step_res[k] is Array:
			var arr: Array = _last_step_res[k]
			for i in range(mini(4, arr.size())):
				out[i] = float(arr[i])
			return out
	if model != null:
		var b_state = model.get("brake_state")
		if b_state != null:
			var bs_temps = b_state.get("disc_temp")
			if bs_temps is Array and bs_temps.size() >= 4:
				for i in range(4):
					out[i] = float(bs_temps[i])
				return out
		var m_discs = model.get("brake_disc_temps")
		if m_discs is Array and m_discs.size() >= 4:
			for i in range(4):
				out[i] = float(m_discs[i])
			return out
		for i in range(mini(4, model.wheels.size())):
			var w = model.wheels[i]
			var dt_val = w.get("disc_temp_c")
			if dt_val == null:
				dt_val = w.get("brake_disc_temp_c")
			if dt_val != null:
				out[i] = float(dt_val)
	if telemetry != null and telemetry.get("disc_temp") is Array:
		var t_discs: Array = telemetry.get("disc_temp")
		for i in range(mini(4, t_discs.size())):
			out[i] = float(t_discs[i])
	return out

func _get_active_surface_mu() -> float:
	if _last_step_res.has("surface_mu"):
		var s_mu = _last_step_res["surface_mu"]
		if s_mu is Array and s_mu.size() > 0:
			var sum_mu: float = 0.0
			for v in s_mu:
				sum_mu += float(v)
			return sum_mu / float(s_mu.size())
		return float(s_mu)
	if model != null:
		var m_mu = model.get("surface_mu")
		if m_mu != null:
			return float(m_mu)
		if model.wheels.size() >= 4:
			var sum_w: float = 0.0
			var has_w_mu := false
			for w in model.wheels:
				var w_mu = w.get("surface_mu_scale")
				if w_mu == null:
					w_mu = w.get("surface_mu")
				if w_mu != null:
					has_w_mu = true
					sum_w += float(w_mu)
			if has_w_mu:
				return sum_w / float(model.wheels.size())
	if model != null and model.ecu_tv_v3 != null and tv_mu_est.size() >= 4:
		return 0.25 * (tv_mu_est[0] + tv_mu_est[1] + tv_mu_est[2] + tv_mu_est[3])
	if config != null:
		return config.surface_grip_scale
	return 1.0

func _stop_all_joy_vibration() -> void:
	for joy_id in Input.get_connected_joypads():
		Input.stop_joy_vibration(joy_id)
	_joy_vibrating = false

func _notification(what: int) -> void:
	if what == NOTIFICATION_PAUSED or what == NOTIFICATION_EXIT_TREE or what == NOTIFICATION_APPLICATION_FOCUS_OUT:
		if _joy_vibrating:
			_stop_all_joy_vibration()
		if what == NOTIFICATION_EXIT_TREE and telemetry_recorder != null and telemetry_recorder.is_recording:
			telemetry_recorder.stop_recording()

func _process_manual_pilot(delta: float) -> void:
	_sync_gamepad_settings()
	if wheel_ffb != null:
		wheel_ffb.input_device_override = input_device_mode
	var joypads: Array[int] = Input.get_connected_joypads()
	var has_joypad: bool = joypads.size() > 0
	var kb_steer: float = Input.get_action_strength("manual_steer_l") - Input.get_action_strength("manual_steer_r")
	var kb_throttle: float = clampf(Input.get_action_strength("manual_gas_pos"), 0.0, 1.0)
	var kb_brake: float = clampf(Input.get_action_strength("manual_gas_neg"), 0.0, 1.0)
	var ebs_in: float = clampf(Input.get_action_strength("EBS"), 0.0, 1.0)

	if has_joypad:
		var joy_id: int = joypads[0]
		var device_mode: int = WheelFFBControllerScript.InputDeviceMode.GAMEPAD
		if wheel_ffb != null:
			device_mode = wheel_ffb.detect_input_device(joy_id)
			wheel_ffb.configure_gamepad_controller(gamepad)
		var is_wheel: bool = (device_mode == WheelFFBControllerScript.InputDeviceMode.WHEEL)
		var raw_x: float = -Input.get_joy_axis(joy_id, JOY_AXIS_LEFT_X)
		var vx_mps: float = absf((global_transform.basis.inverse() * linear_velocity).z)
		if is_wheel:
			if absf(raw_x) > 0.002 or is_zero_approx(kb_steer):
				_steer_smoothed = wheel_ffb.step_steering(raw_x, vx_mps, delta, gamepad)
			else:
				var steer_rate = manual_steer_return_rate if (absf(kb_steer) < absf(_steer_smoothed) or kb_steer * _steer_smoothed < 0.0) else manual_steer_rate
				_steer_smoothed = _smooth_steer(_steer_smoothed, kb_steer, delta, steer_rate)
		elif absf(raw_x) > gamepad.steering_deadzone or is_zero_approx(kb_steer):
			_steer_smoothed = gamepad.step_steering(raw_x, vx_mps, delta)
		else:
			var steer_rate = manual_steer_return_rate if (absf(kb_steer) < absf(_steer_smoothed) or kb_steer * _steer_smoothed < 0.0) else manual_steer_rate
			_steer_smoothed = _smooth_steer(_steer_smoothed, kb_steer, delta, steer_rate)

		var joy_throttle: float = gamepad.normalize_trigger(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_RIGHT), false)
		var joy_brake: float = gamepad.normalize_trigger(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_LEFT), true)
		_raw_steer_in = raw_x if absf(raw_x) > 0.002 or is_zero_approx(kb_steer) else kb_steer
		_raw_throttle_in = maxf(kb_throttle, joy_throttle)
		_raw_brake_in = maxf(maxf(kb_brake, joy_brake), ebs_in)
		if is_wheel:
			self.gas = clampf(_raw_throttle_in, 0.0, 1.0)
			self.brake = clampf(_raw_brake_in, 0.0, 1.0)
		else:
			self.gas = gamepad.shape_throttle(_raw_throttle_in)
			self.brake = maxf(gamepad.shape_brake(maxf(kb_brake, joy_brake)), ebs_in)

		if model != null:
			if is_wheel and wheel_ffb != null:
				wheel_ffb.apply_haptic_feedback(joy_id, model.wheels, linear_velocity.length(), delta)
				ffb_torque_nm = wheel_ffb.ffb_torque_nm
				_joy_vibrating = absf(wheel_ffb.ffb_normalized) > 0.01
			else:
				var rumble: Vector2 = gamepad.compute_rumble(model.wheels, model.suspension.corners, linear_velocity.length(), delta)
				if rumble.x > 0.01 or rumble.y > 0.01:
					Input.start_joy_vibration(joy_id, rumble.x, rumble.y, 0.05)
					_joy_vibrating = true
				elif _joy_vibrating:
					_stop_all_joy_vibration()
	else:
		if wheel_ffb != null:
			wheel_ffb.detect_input_device_from_name("")
		var steer_rate = manual_steer_return_rate if (absf(kb_steer) < absf(_steer_smoothed) or kb_steer * _steer_smoothed < 0.0) else manual_steer_rate
		_steer_smoothed = _smooth_steer(_steer_smoothed, kb_steer, delta, steer_rate)
		_raw_steer_in = kb_steer
		_raw_throttle_in = kb_throttle
		_raw_brake_in = maxf(kb_brake, ebs_in)
		self.gas = kb_throttle
		self.brake = _raw_brake_in
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

func _setup_minimal_hud() -> void:
	if DisplayServer.get_name() == "headless" or OS.has_feature("headless"):
		return
	_hud_layer = CanvasLayer.new()
	_hud_layer.layer = 10
	var panel := PanelContainer.new()
	panel.position = Vector2(16, 16)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.05, 0.06, 0.08, 0.72)
	style.set_corner_radius_all(6)
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)
	_hud_label = Label.new()
	_hud_label.add_theme_font_size_override("font_size", 15)
	_hud_label.add_theme_color_override("font_color", Color(0.95, 0.97, 1.0))
	panel.add_child(_hud_label)
	_hud_layer.add_child(panel)
	add_child(_hud_layer)
	_tv_overlay = TvLiveTuningOverlayScript.new()
	_tv_overlay.bind_car(model.ecu_tv_v3 if model != null else null, gamepad, self)
	add_child(_tv_overlay)
	_telemetry_inspector = TelemetryInspectorOverlayScript.new()
	if _telemetry_inspector.has_method("bind_car"):
		_telemetry_inspector.bind_car(self)
	add_child(_telemetry_inspector)
	_update_minimal_hud()

func _update_minimal_hud() -> void:
	if _telemetry_inspector != null and _telemetry_inspector.visible:
		_telemetry_inspector.update_from_car(self, 1.0 / 60.0)
	if _tv_overlay != null and _tv_overlay.visible and model != null:
		_tv_overlay.update_telemetry(tv_wz_ref, model.angular_velocity_ros.z, tv_beta_est_deg, tv_mz_req, tv_tc_active)
		_tv_overlay.update_input_telemetry(_raw_steer_in, _steer_smoothed, _raw_throttle_in, self.gas, _raw_brake_in, self.brake)
		if _tv_overlay.has_method("update_chassis_telemetry"):
			_tv_overlay.update_chassis_telemetry(_collect_imo_temps(), _collect_disc_temps(), _get_active_surface_mu())
	if _hud_label == null:
		return
	var kmh: float = linear_velocity.length() * 3.6
	var mode_str: String = "ECU TV v3 4WD [F7]" if ecu_driving_mode == 1 else "LINEAL 4WD [F7]"
	var total_maps: int = model.ecu_tv_v3._available_tv_maps.size() if (model != null and model.ecu_tv_v3 != null and not model.ecu_tv_v3._available_tv_maps.is_empty()) else 4
	var map_str: String = "  |  TV Map [%d/%d]: %s [F6]" % [tv_map_id + 1, total_maps, tv_map_name] if ecu_driving_mode == 1 else ""
	var tc_str: String = "  |  TC ACTIVE" if tv_tc_active else ""
	var rec_ind: String = telemetry_recorder.get_hud_indicator() if telemetry_recorder != null else ""
	var rec_str: String = rec_ind if rec_ind != "" else "  |  [F9] REC"
	var hv_str: String = ""
	if model != null and model.hv_state != null:
		hv_str = "   |   %3.0f V (%.0f%% SoC)" % [model.hv_state.v_dc, model.hv_state.soc * 100.0]
	var dev_str: String = "  |  WHEEL 1:1" if (wheel_ffb != null and wheel_ffb.active_device_mode == WheelFFBControllerScript.InputDeviceMode.WHEEL) else ""
	_hud_label.text = "%5.1f km/h   |   %4.1f / 80.0 kW%s\n%s%s%s%s%s  |  [F10] Live Tuning  |  [F11] Telemetry X-Ray" % [
		kmh, tv_elec_power_kw, hv_str, mode_str, map_str, tc_str, dev_str, rec_str
	]

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
	_last_step_res.clear()
	if wheel_ffb != null:
		wheel_ffb.reset()
	if telemetry_recorder != null and telemetry_recorder.is_recording:
		telemetry_recorder.stop_recording()
	ffb_torque_nm = 0.0
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
