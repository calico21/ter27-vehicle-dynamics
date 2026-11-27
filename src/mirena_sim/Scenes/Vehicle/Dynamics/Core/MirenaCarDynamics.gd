extends RigidBody3D
class_name MirenaCar

## mirena dynamics - vehicle dynamics model using rigidbody3d
## formula student driverless monoplaza (mirena)
## replaces native vehiclebody3d with custom 10-dof dynamics:
## 4 raycast3d ground contact sensors for susp
## custom tire model with combined slip and friction ellipse (tiremodel.gd)
## configurable 4wd / rwd powertrain with 80 kw limit (drivetrainmodel.gd)
## chassis integration via _integrate_forces (no double integration)

const VehicleConfigScript = preload("../../Config/VehicleConfig.gd")
const VehicleInputsScript = preload("../../Inputs/VehicleInputs.gd")
const SetupProfileScript = preload("../../Config/SetupProfile.gd")
const VehicleModelRegistryScript = preload("VehicleModelRegistry.gd")
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
const SuspensionVisualizer3DScript = preload("../../Visuals/SuspensionVisualizer3D.gd")
const WheelForcesVisualizer3DScript = preload("../../Visuals/WheelForcesVisualizer3D.gd")
const DilProceduralAudioScript = preload("../../Audio/DilProceduralAudio.gd")
const InputConditionerScript = preload("InputConditioner.gd")
const DoubleWishboneKinematicsScript = preload("../Suspension/DoubleWishboneKinematics.gd")

signal cone_hit(count: int)
enum PilotMode { NO_PILOT, MANUAL, ROS, TRACK_RAIL }
enum SimDriveMode { OPEN_4WD = 0, AUTOCROSS = 1, ENDURANCE = 2, SKIDPAD = 3, ACCELERATION = 4, TIRE_WARMER = 5 }
const SIM_MODE_NAMES: Array[String] = ["OPEN 4WD", "AUTOCROSS", "ENDURANCE", "SKIDPAD", "ACCELERATION", "TIRE WARMER"]

# config fisica
var config: VehicleConfigScript
var model = null
var inputs: VehicleInputsScript
var telemetry: VehicleTelemetryScript

@export_enum("full_14dof", "fast_3dof") var physics_fidelity: String = "full_14dof"
@export var pilot: PilotMode = PilotMode.NO_PILOT
@export var drive_mode: VehicleConfigScript.DriveMode = VehicleConfigScript.DriveMode.FOUR_WD
@export var tv_mode: int = 0
@export_enum("LINEAL_4WD", "ECU_TV_V3_4WD", "ECU_TV_4WD_CASCADE") var ecu_driving_mode: int = 2
@export var tv_4wd_mode: int = 0
@export var sim_drive_mode: int = SimDriveMode.AUTOCROSS
var tv_yaw_moment_req: float = 0.0

@export_group("Input Conditioning & Haptics")
@export_range(0.0, 0.30, 0.01) var steering_deadzone: float = 0.06
@export_range(1.0, 4.0, 0.05) var steering_gamma: float = 1.6
@export_range(0.0, 1.5, 0.01) var speed_sensitivity: float = 1.0
@export_range(0.5, 2.5, 0.05) var throttle_gamma: float = 1.0
@export_range(0.5, 2.5, 0.05) var brake_gamma: float = 1.0
@export_range(0.0, 1.0, 0.05) var rumble_effects: float = 0.7

var wheel_ffb = null
var telemetry_recorder: TelemetryRecorderScript = TelemetryRecorderScript.new()
@export_enum("AUTO", "GAMEPAD", "WHEEL") var input_device_mode: int = 0
@export var tv_hardware_realism: bool = false
@export var rail_speed: float = 10.0
@export var rail_look_ahead: float = 3.0
var _rail_progress: float = 0.0
var path: Path3D

var _base_cl_a: float = 6.0
var _base_cd_a: float = 1.5
var _track_aero_scale: float = 1.0
var _track_temp_preset: String = "WARM_55C"

# susp mount geometry
# hardpoints for raycasts (in car local space, +x right, +y up, -z forward)
var ray_mount_positions: Array[Vector3] = [
	Vector3(-0.587, 0.28, -0.993), # fl
	Vector3( 0.587, 0.28, -0.993), # fr
	Vector3(-0.587, 0.28,  0.600), # rl
	Vector3( 0.587, 0.28,  0.600)  # rr
]

var ray_rest_lengths: Array[float] = [0.32, 0.32, 0.32, 0.32] # 4 cm static travel allowance
var ray_cast_length: float = 0.45

var raycasts: Array[RayCast3D] = []
var wheel_mesh_nodes: Array[Node3D] = []
var _wheel_base_yaws: Array[float] = [PI, PI, PI, PI]

# 3d wheel force & saturation arrows
var wheel_arrows_enabled: bool = false
var _wheel_arrow_roots: Array[Node3D] = []
var _wheel_arrow_shafts: Array[Node3D] = []
var _wheel_arrow_tips: Array[Node3D] = []
var _wheel_arrow_mats: Array[StandardMaterial3D] = []

# ros configuration & variables
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

# internal states
var origin: Transform3D = Transform3D.IDENTITY
var gas: float = 0.0
var steering: float = 0.0
var brake: float = 0.0
var _steer_smoothed: float = 0.0
var _filtered_steer_angle: float = 0.0
var _raw_steer_in: float = 0.0
var _raw_throttle_in: float = 0.0
var _raw_brake_in: float = 0.0
var _joy_vibrating: bool = false

# ecu tv v3 live telemetry (visible in godot remote inspector)
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

var susp_visualizer: SuspensionVisualizer3DScript = null
var wheel_forces_vis: WheelForcesVisualizer3DScript = null
var dil_audio: DilProceduralAudioScript = null
var _susp_cam: Camera3D = null
var _tv_overlay: TvLiveTuningOverlayScript = null
var _telemetry_inspector: TelemetryInspectorOverlayScript = null
var _last_step_res: Dictionary = {}
var track_surface_mu_scale: float = 1.0
var split_mu_test_enabled: bool = false
var raw_steer_norm: float = 0.0
var _session_time_s: float = 0.0
var _session_energy_kj: float = 0.0
var _last_ax_g: float = 0.0
var _last_ay_g: float = 0.0
var input_conditioner: InputConditionerScript = InputConditionerScript.new()
var ffb_haptics_enabled: bool = true
var racing_line_enabled: bool = false
var _racing_line_markers_enabled: bool = true
var _tire_sat_pct_scratch: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _tire_temps_scratch: Array[float] = [40.0, 40.0, 40.0, 40.0]
var _last_bump_rumble: float = 0.0

var slam_cones: Array = []
var hit_cones: Array = []
var _as_info = null
var _mission_info = null

var initial_gate_crossings: int = 0
var lap_start_ticks: float = 0.0

var car_state = null
var wheel_speeds = null
var lap_status_msg = null

# previous susp compressions (signed delta_z) & contact state for damping
var _prev_compressions: Array[float] = [0.0, 0.0, 0.0, 0.0]
var _was_grounded: Array[bool] = [false, false, false, false]

func _ready() -> void:
	contact_monitor = true
	max_contacts_reported = 10
	body_entered.connect(_on_body_entered)
	add_to_group("vehicle")
	add_to_group("Car")
	
	# permitir forzar fidelidad desde variable de entorno o cli (--fidelity=fast_3dof)
	var env_fid := OS.get_environment("MIRENA_PHYSICS_FIDELITY").strip_edges().to_lower()
	if env_fid == "fast_3dof" or env_fid == "full_14dof":
		physics_fidelity = env_fid
	var cli_args := OS.get_cmdline_args() + OS.get_cmdline_user_args()
	for idx_arg in range(cli_args.size()):
		var a := String(cli_args[idx_arg])
		if a.begins_with("--fidelity="):
			var v := a.substr(11).strip_edges().to_lower()
			if v == "fast_3dof" or v == "full_14dof":
				physics_fidelity = v
		elif a == "--fidelity" and idx_arg + 1 < cli_args.size():
			var v2 := String(cli_args[idx_arg + 1]).strip_edges().to_lower()
			if v2 == "fast_3dof" or v2 == "full_14dof":
				physics_fidelity = v2

	# instancia subsistemas de dinamica (fuente unica desde config/vehicles/<vehicle_id>/)
	config = VehicleConfigScript.new()
	config.load_for_drive_mode(drive_mode)
	TireModelScript.enable_neural_correction = false
	model = VehicleModelRegistryScript.create_model(physics_fidelity, config)
	inputs = VehicleInputsScript.new()
	telemetry = VehicleTelemetryScript.new()
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(tv_hardware_realism)
		tv_map_id = model.ecu_tv_v3.active_map_id
		tv_map_name = model.ecu_tv_v3.get_map_name()
		inputs.tv_map_id = tv_map_id
	if wheel_ffb != null:
		wheel_ffb.input_device_override = input_device_mode
	_base_cl_a = config.cl_a
	_base_cd_a = config.cd_a
	_sync_input_conditioner_exports()
	
	# configura rigidbody3d desde vehicleconfig
	mass = config.mass
	inertia = Vector3(config.iyy, config.get_total_izz(), config.ixx)
	center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	center_of_mass = _compute_center_of_mass(config)
	linear_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	linear_damp = 0.0
	angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	angular_damp = 0.0
	
	_sync_geometry_from_config()
	_setup_raycasts_and_wheels()
	cone_collision_set(true)
	_connect_to_initial_gate()
	_init_ros()

	susp_visualizer = SuspensionVisualizer3DScript.new()
	susp_visualizer.name = "SuspensionVisualizer3D"
	susp_visualizer.bind_car(self)
	add_child(susp_visualizer)

	wheel_forces_vis = WheelForcesVisualizer3DScript.new()
	wheel_forces_vis.name = "WheelForcesVisualizer3D"
	wheel_forces_vis.bind_car(self)
	add_child(wheel_forces_vis)

	dil_audio = DilProceduralAudioScript.new()
	dil_audio.name = "DilProceduralAudio"
	add_child(dil_audio)

	_susp_cam = Camera3D.new()
	_susp_cam.name = "SUSPENSION CAM"
	var cam_pos := Vector3(1.38, 0.48, -1.78)
	var cam_tgt := Vector3(0.56, 0.22, -0.96)
	_susp_cam.transform = Transform3D(Basis.looking_at((cam_tgt - cam_pos).normalized(), Vector3.UP), cam_pos)
	add_child(_susp_cam)

	if config.vehicle_id == "ter27_4wd":
		set_sim_drive_mode(SimDriveMode.AUTOCROSS)
	else:
		set_sim_drive_mode(SimDriveMode.OPEN_4WD)

	_setup_minimal_hud()

	# register cameras
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node:
		if has_node("TPCam"):
			sim_node.register_camera($TPCam)
		if has_node("FPCam"):
			sim_node.register_camera($FPCam)
		if _susp_cam != null:
			sim_node.register_camera(_susp_cam)

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
		# convert iso/ros horizontal offsets from cog (+x fwd, +y left) to godot local (+x right, -z fwd)
		var x_godot: float = center_of_mass.x - r_ros.y
		var z_godot: float = center_of_mass.z - r_ros.x
		ray_mount_positions.append(Vector3(x_godot, config.ray_rest_length, z_godot))

func _setup_raycasts_and_wheels() -> void:
	raycasts.clear()
	wheel_mesh_nodes.clear()
	
	var wheel_names = ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	var ray_names = ["FL_RAY", "FR_RAY", "RL_RAY", "RR_RAY"]
	
	for i in range(4):
		# look for or create raycast3d and align dynamically to vehicleconfig geometry
		var ray: RayCast3D = null
		if has_node(ray_names[i]):
			ray = get_node(ray_names[i]) as RayCast3D
		else:
			ray = RayCast3D.new()
			ray.name = ray_names[i]
			add_child(ray)
		ray.position = ray_mount_positions[i]
		ray.target_position = Vector3(0.0, -ray_cast_length, 0.0)
		ray.collision_mask = 1 # ground / track surface
		ray.add_exception(self)
		ray.enabled = true
		raycasts.append(ray)
		
		# look for wheel mesh node and align horizontally without altering .tscn
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

# physics process (piloting & telemetry)
func _get_active_track_name() -> String:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node:
		var trk = sim_node.get("track")
		if trk != null:
			var t_name = trk.get("track_name")
			if t_name != null and String(t_name).strip_edges() != "":
				return String(t_name)
			if trk is Node:
				return trk.name
	return "manual"

func set_sim_drive_mode(mode_id: int) -> void:
	var clamped_id: int = clampi(mode_id, 0, SIM_MODE_NAMES.size() - 1)
	sim_drive_mode = clamped_id
	match clamped_id:
		SimDriveMode.OPEN_4WD:
			ecu_driving_mode = 0
			tv_4wd_mode = 0
		SimDriveMode.AUTOCROSS:
			ecu_driving_mode = 2
			tv_4wd_mode = 0
		SimDriveMode.ENDURANCE:
			ecu_driving_mode = 2
			tv_4wd_mode = 2
		SimDriveMode.SKIDPAD:
			ecu_driving_mode = 2
			tv_4wd_mode = 1
		SimDriveMode.ACCELERATION:
			ecu_driving_mode = 2
			tv_4wd_mode = 3
		SimDriveMode.TIRE_WARMER:
			ecu_driving_mode = 2
			tv_4wd_mode = 4
	if inputs != null:
		inputs.ecu_driving_mode = ecu_driving_mode
		inputs.tv_4wd_mode = tv_4wd_mode
	if model != null and model.ecu_tv_4wd != null:
		model.ecu_tv_4wd.active_mode = tv_4wd_mode
		model.ecu_tv_4wd._refresh_mode_cache(model.ecu_tv_4wd._get_mode_dict())
	if ecu_driving_mode == 2:
		tv_map_id = tv_4wd_mode
		tv_map_name = get_sim_drive_mode_name()
	_update_minimal_hud()


func cycle_sim_drive_mode() -> int:
	var next_id: int = (sim_drive_mode + 1) % SIM_MODE_NAMES.size()
	set_sim_drive_mode(next_id)
	print("[Drive Mode]: %s (ecu=%d tv4=%d)" % [get_sim_drive_mode_name(), ecu_driving_mode, tv_4wd_mode])
	return sim_drive_mode


func get_sim_drive_mode_name() -> String:
	return SIM_MODE_NAMES[clampi(sim_drive_mode, 0, SIM_MODE_NAMES.size() - 1)]


func toggle_suspension_visualizer() -> bool:
	if susp_visualizer == null:
		return false
	var on: bool = susp_visualizer.toggle_visualizer()
	_update_minimal_hud()
	return on


func cycle_camera_view() -> void:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and sim_node.has_method("cycle_camera"):
		sim_node.cycle_camera()


func _is_overlay_typing() -> bool:
	return (
		(_tv_overlay != null and _tv_overlay.has_method("is_text_input_focused") and _tv_overlay.is_text_input_focused())
		or (_telemetry_inspector != null and _telemetry_inspector.has_method("is_text_input_focused") and _telemetry_inspector.is_text_input_focused())
	)


func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_M:
			if not _is_overlay_typing():
				cycle_sim_drive_mode()
		elif event.keycode == KEY_V:
			if not _is_overlay_typing():
				toggle_suspension_visualizer()
		elif event.keycode == KEY_C:
			if not _is_overlay_typing():
				cycle_camera_view()
		elif event.keycode == KEY_F8:
			toggle_wheel_arrows()
		elif event.keycode == KEY_L:
			if not _is_overlay_typing():
				launch_control_arm()
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

# physics integration (forces & moments)
## native godot rigidbody3d force integration callback
## executed every physics tick using actual physics step delta (dt = state.step)
func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:
	var dt = state.step
	if dt <= 0.0:
		return
	
	# 1. update inputs (lag de 1er orden del actuador de direccion 25 ms; bypass en WHEEL_DIRECT 1:1)
	var is_wheel_direct: bool = (
		pilot == PilotMode.MANUAL
		and input_conditioner != null
		and input_conditioner.active_resolved_profile == InputConditionerScript.InputProfile.WHEEL_DIRECT
	)
	if is_wheel_direct:
		_filtered_steer_angle = self.steering
	else:
		var steer_alpha: float = 1.0 - exp(-dt / 0.025)
		_filtered_steer_angle = lerpf(_filtered_steer_angle, self.steering, steer_alpha)
	inputs.drive_mode = drive_mode
	inputs.tv_mode = tv_mode
	inputs.ecu_driving_mode = ecu_driving_mode
	inputs.tv_4wd_mode = tv_4wd_mode
	inputs.tv_map_id = tv_map_id
	inputs.tv_yaw_moment_req = tv_yaw_moment_req
	inputs.set_unified_control(self.gas, _filtered_steer_angle, self.brake, config.max_steer_angle)
	
	# 2. vels locales de chasis en ejes iso/ros (+x fwd, +y left, +z up)
	var inv_basis: Basis = state.transform.basis.inverse()
	var local_v_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_basis * state.linear_velocity)
	var local_w_ros: Vector3 = VehicleConfigScript.godot_to_ros_vec(inv_basis * state.angular_velocity)
	
	# 3. susp x raycast y geometria de contacto (signed delta_z: >0 jounce, 0 design static, <0 droop)
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
	
	# 4. monta contactos con el suelo in iso/ros frame and delegate susp + tire physics to vehicledynamicsmodel
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
			surface_mu_i *= track_surface_mu_scale
			if split_mu_test_enabled:
				var is_left_wheel: bool = (i == 0 or i == 2)
				surface_mu_i *= (0.65 if is_left_wheel else 1.00)
		ground_contacts.append({
			"grounded": grounded[i],
			"delta_z": delta_z[i],
			"delta_z_dot": delta_z_dot[i],
			"r_pos_ros": r_pos_ros,
			"surface_mu": surface_mu_i,
			"world_pos_ros": Vector2(-hit_pts[i].z, -hit_pts[i].x),
			"world_vel_ros": Vector2(-state.linear_velocity.z, -state.linear_velocity.x)
		})
	
	if model != null and model.ecu_tv_4wd != null:
		var max_s: float = maxf(config.max_steer_angle, 1e-4) if config != null else 0.45
		var d_norm: float = raw_steer_norm if (pilot == PilotMode.MANUAL and input_conditioner != null) else clampf(self.steering / max_s, -1.0, 1.0)
		model.ecu_tv_4wd.set_raw_steer_norm(d_norm)
	var step_res: Dictionary = model.step_forces(inputs, local_v_ros, local_w_ros, ground_contacts, dt)
	_last_step_res = step_res
	if ecu_driving_mode == 2 and model.ecu_tv_4wd != null:
		var e4 = model.ecu_tv_4wd
		tv_vy_est = e4.ekf_vy
		tv_beta_est_deg = rad_to_deg(atan2(e4.ekf_vy, maxf(e4.ekf_vx, 0.5))) if e4.ekf_vx > 1.5 else 0.0
		tv_wz_ref = e4.last_wz_ref
		tv_mz_req = e4.last_mz_req
		tv_mu_est = e4.mu_hat.duplicate()
		tv_kappa_opt = [e4._mc_kappa_lim, e4._mc_kappa_lim, e4._mc_kappa_lim, e4._mc_kappa_lim]
		tv_torque_4wd = step_res["drive_torques"].duplicate()
		tv_tc_active = e4.tc_active
		tv_map_id = tv_4wd_mode
		tv_map_name = get_sim_drive_mode_name()
	elif model.ecu_tv_v3 != null:
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

	if "steering_torque_nm" in model:
		ffb_torque_nm = float(model.steering_torque_nm)
	elif wheel_ffb != null:
		ffb_torque_nm = wheel_ffb.compute_steering_torque_nm(
			model.wheels, wheel_ffb.rack_ratio, wheel_ffb.mechanical_trail, wheel_ffb.scrub_radius
		)

	# DiL telemetry scratch, 3D Fz/torque visuals, procedural audio & DiL lap session
	var w_mu_ref: float = maxf(config.tyre_mu_y if config != null else 1.5, 0.5)
	var om_sum: float = 0.0
	var max_dz_dot_abs: float = 0.0
	for i in range(4):
		var wi = model.wheels[i]
		var mu_eff_i: float = maxf(w_mu_ref * wi.surface_mu_scale, 0.2)
		var eta_i: float = compute_wheel_saturation(wi.F_x, wi.F_y, wi.F_z, mu_eff_i)
		_tire_sat_pct_scratch[i] = clampf(eta_i * 100.0, 0.0, 120.0)
		_tire_temps_scratch[i] = wi.t_surf_c
		om_sum += absf(wi.omega)
		max_dz_dot_abs = maxf(max_dz_dot_abs, absf(delta_z_dot[i]))
	_last_bump_rumble = clampf((max_dz_dot_abs - 0.18) / 0.65, 0.0, 1.0)

	if wheel_forces_vis != null and (wheel_forces_vis.show_fz_arrows or wheel_forces_vis.show_torque_bars) and ("wheel_drive_torque_nm" in model):
		var t_whl_max: float = maxf(config.motor_peak_torque * config.gear_ratio, 100.0)
		wheel_forces_vis.update_visuals(
			step_res["normal_forces"],
			model.wheel_drive_torque_nm,
			model.wheel_regen_brake_nm,
			model.wheel_mech_brake_nm,
			t_whl_max
		)

	if dil_audio != null and dil_audio.enabled:
		var sat_f_m: float = 0.005 * (_tire_sat_pct_scratch[0] + _tire_sat_pct_scratch[1])
		var sat_r_m: float = 0.005 * (_tire_sat_pct_scratch[2] + _tire_sat_pct_scratch[3])
		var lv_a: Vector3 = model.linear_velocity_ros
		var b_cg_m: float = config.b_cog if config != null else 0.693
		var b_kin_a := atan2(b_cg_m * model.angular_velocity_ros.z, maxf(absf(lv_a.x), 3.0))
		var beta_dyn_m: float = atan2(lv_a.y, maxf(absf(lv_a.x), 0.5)) - b_kin_a
		dil_audio.step_audio(0.25 * om_sum, tv_elec_power_kw, sat_f_m, sat_r_m, beta_dyn_m)

	if track_node != null and ("dil_session" in track_node) and track_node.dil_session != null:
		track_node.dil_session.step_session(dt, state.transform.origin, state.transform.basis, state.linear_velocity.length())
	var inv_mass: float = 1.0 / maxf(config.mass, 1e-3)
	var accel_ros: Vector3 = model.net_force_chassis_ros * inv_mass
	_last_ax_g = accel_ros.x / 9.81
	_last_ay_g = accel_ros.y / 9.81
	_session_time_s += dt
	_session_energy_kj += tv_elec_power_kw * dt
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
	
	# 5. aplica fuerzas de rueda y normales (Godot apply_force posicion relativa al origen, resta center_of_mass internamente)
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
		
		# visual wheel mesh transform update
		_update_wheel_visual(i, ray, delta_w, w.rotation_angle)

	if wheel_arrows_enabled:
		_update_wheel_force_arrows(wheel_forces_ros, normal_forces, grounded)
	
	# 6. aerodynamic drag & downforce
	var f_aero_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(step_res["aero_force_ros"])
	var tau_aero_godot_local: Vector3 = VehicleConfigScript.ros_to_godot_vec(step_res["aero_torque_ros"])
	state.apply_central_force(state.transform.basis * f_aero_godot_local)
	if not tau_aero_godot_local.is_zero_approx():
		state.apply_torque(state.transform.basis * tau_aero_godot_local)
	
	# 7. bloqueo de freno a baja vel pa q no haya creep en parado
	if inputs.brake > 0.1 and inputs.throttle == 0.0:
		if state.linear_velocity.length() < 0.2:
			state.linear_velocity = Vector3.ZERO
			state.angular_velocity = Vector3.ZERO

func _update_wheel_visual(index: int, ray: RayCast3D, delta_w: float, roll_angle: float) -> void:
	var w_node = wheel_mesh_nodes[index]
	if not w_node:
		return
	
	# susp vertical travel
	if ray.is_colliding():
		var hit_pt = ray.get_collision_point()
		var hit_dist = ray.global_position.distance_to(hit_pt)
		w_node.position.y = ray_mount_positions[index].y - hit_dist + config.wheel_radius
	else:
		w_node.position.y = ray_mount_positions[index].y - ray_cast_length + config.wheel_radius
	
	# steering yaw rotation for front wheels (+delta_w = ccw / left turn)
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
	var col_green := Color(0.0627, 0.7255, 0.5059) # 10b981
	var col_amber := Color(0.9608, 0.6196, 0.0431) # f59e0b
	var col_red := Color(0.9373, 0.2667, 0.2667)   # ef4444
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

# pilot logic
func _process_no_pilot() -> void:
	self.gas = 0.0
	self.steering = 0.0
	self.brake = 1.0

func _sync_input_conditioner_exports() -> void:
	if input_conditioner == null:
		return
	input_conditioner.steer_deadzone = clampf(steering_deadzone, 0.0, 0.30)
	input_conditioner.steer_gamma = clampf(steering_gamma, 1.0, 3.5)
	input_conditioner.steer_speed_sensitivity = clampf(speed_sensitivity, 0.0, 1.5)
	input_conditioner.throttle_gamma = clampf(throttle_gamma, 0.5, 2.5)
	input_conditioner.brake_gamma = clampf(brake_gamma, 0.5, 2.5)
	input_conditioner.rumble_intensity = clampf(rumble_effects, 0.0, 1.0)


func get_live_gamepad_params() -> Dictionary:
	if input_conditioner == null:
		return {}
	return {
		"steering_deadzone": input_conditioner.steer_deadzone,
		"steering_gamma": input_conditioner.steer_gamma,
		"speed_sensitivity": input_conditioner.steer_speed_sensitivity,
		"throttle_gamma": input_conditioner.throttle_gamma,
		"brake_gamma": input_conditioner.brake_gamma,
		"rumble_effects": input_conditioner.rumble_intensity,
	}


func set_live_gamepad_param(key: String, val: float) -> void:
	if input_conditioner == null:
		return
	match key:
		"steering_deadzone", "steer_deadzone":
			steering_deadzone = clampf(val, 0.0, 0.30)
			input_conditioner.steer_deadzone = steering_deadzone
		"steering_gamma", "steer_gamma":
			steering_gamma = clampf(val, 1.0, 3.5)
			input_conditioner.steer_gamma = steering_gamma
		"speed_sensitivity", "steer_speed_sensitivity":
			speed_sensitivity = clampf(val, 0.0, 1.5)
			input_conditioner.steer_speed_sensitivity = speed_sensitivity
		"throttle_gamma":
			throttle_gamma = clampf(val, 0.5, 2.5)
			input_conditioner.throttle_gamma = throttle_gamma
		"brake_gamma":
			brake_gamma = clampf(val, 0.5, 2.5)
			input_conditioner.brake_gamma = brake_gamma
		"rumble_effects", "rumble_intensity":
			rumble_effects = clampf(val, 0.0, 1.0)
			input_conditioner.rumble_intensity = rumble_effects


func save_gamepad_config_to_json(_path: String = "user://gamepad.json") -> bool:
	return false


func get_default_live_tuning() -> Dictionary:
	var ecu_def: Dictionary = model.ecu_tv_4wd.get_default_live_tuning() if (model != null and model.ecu_tv_4wd != null) else {}
	var b_bias_pct: float = clampf(config.brake_bias_front * 100.0, 40.0, 75.0) if config != null else 58.0
	var arb_f: float = config.arb_stiffness_front if config != null else 4500.0
	var arb_r: float = config.arb_stiffness_rear if config != null else 5500.0
	var arb_tot: float = maxf(arb_f + arb_r, 1.0)
	var arb_bal_pct: float = clampf((arb_f / arb_tot) * 100.0, 20.0, 80.0)
	var lift_nm: float = clampf(config.lift_off_regen_nm, 0.0, 100.0) if config != null else 0.0
	return {
		"hairpin_pivot_gain": float(ecu_def.get("hairpin_pivot_gain", 1.0)),
		"inner_regen_max_nm": float(ecu_def.get("inner_regen_max_nm", 95.0)),
		"yaw_agility_gain": float(ecu_def.get("yaw_agility_gain", 1.0)),
		"slalom_beta_damp": float(ecu_def.get("slalom_beta_damp", 320.0)),
		"beta_soft_deg": float(ecu_def.get("beta_soft_deg", 4.2)),
		"launch_rear_bias_pct": float(ecu_def.get("launch_rear_bias_pct", 70.0)),
		"tc_slip_target": float(ecu_def.get("tc_slip_target", 0.13)),
		"power_limit_kw": clampf(float(ecu_def.get("power_limit_kw", 80.0)), 10.0, 80.0),
		"regen_limit_kw": clampf(float(ecu_def.get("regen_limit_kw", -25.0)), -25.0, 0.0),
		"lift_regen_nm": lift_nm,
		"brake_bias_front_pct": b_bias_pct,
		"arb_front_balance_pct": arb_bal_pct,
		"throttle_exp": float(ecu_def.get("throttle_exp", 1.15)),
		"trail_brake_yaw_gain": float(ecu_def.get("trail_brake_yaw_gain", 1.0)),
		"skidpad_neutral_trim": float(ecu_def.get("skidpad_neutral_trim", 0.0)),
		"delta_dot_lead_gain": float(ecu_def.get("delta_dot_lead_gain", 35.0)),
		"exit_traction_blend": float(ecu_def.get("exit_traction_blend", 0.75)),
		"friction_ellipse_margin": float(ecu_def.get("friction_ellipse_margin", 0.94)),
		"yaw_accel_ff_gain": float(ecu_def.get("yaw_accel_ff_gain", 1.0)),
		"centripetal_fx_gain": float(ecu_def.get("centripetal_fx_gain", 15.0)),
		"delta_alpha_eq_gain": float(ecu_def.get("delta_alpha_eq_gain", 2620.0)),
	}


func get_live_tuning() -> Dictionary:
	var ecu_cur: Dictionary = model.ecu_tv_4wd.get_live_tuning() if (model != null and model.ecu_tv_4wd != null) else {}
	var b_bias_pct: float = clampf(config.brake_bias_front * 100.0, 40.0, 75.0) if config != null else 58.0
	var arb_f: float = config.arb_stiffness_front if config != null else 4500.0
	var arb_r: float = config.arb_stiffness_rear if config != null else 5500.0
	var arb_tot: float = maxf(arb_f + arb_r, 1.0)
	var arb_bal_pct: float = clampf((arb_f / arb_tot) * 100.0, 20.0, 80.0)
	var lift_nm: float = clampf(config.lift_off_regen_nm, 0.0, 100.0) if config != null else 0.0
	return {
		"hairpin_pivot_gain": float(ecu_cur.get("hairpin_pivot_gain", 1.0)),
		"inner_regen_max_nm": float(ecu_cur.get("inner_regen_max_nm", 95.0)),
		"yaw_agility_gain": float(ecu_cur.get("yaw_agility_gain", 1.0)),
		"slalom_beta_damp": float(ecu_cur.get("slalom_beta_damp", 320.0)),
		"beta_soft_deg": float(ecu_cur.get("beta_soft_deg", 4.2)),
		"launch_rear_bias_pct": float(ecu_cur.get("launch_rear_bias_pct", 70.0)),
		"tc_slip_target": float(ecu_cur.get("tc_slip_target", 0.13)),
		"power_limit_kw": clampf(float(ecu_cur.get("power_limit_kw", 80.0)), 10.0, 80.0),
		"regen_limit_kw": clampf(float(ecu_cur.get("regen_limit_kw", -25.0)), -25.0, 0.0),
		"lift_regen_nm": lift_nm,
		"brake_bias_front_pct": b_bias_pct,
		"arb_front_balance_pct": arb_bal_pct,
		"throttle_exp": float(ecu_cur.get("throttle_exp", 1.15)),
		"trail_brake_yaw_gain": float(ecu_cur.get("trail_brake_yaw_gain", 1.0)),
		"skidpad_neutral_trim": float(ecu_cur.get("skidpad_neutral_trim", 0.0)),
		"delta_dot_lead_gain": float(ecu_cur.get("delta_dot_lead_gain", 35.0)),
		"exit_traction_blend": float(ecu_cur.get("exit_traction_blend", 0.75)),
		"friction_ellipse_margin": float(ecu_cur.get("friction_ellipse_margin", 0.94)),
		"yaw_accel_ff_gain": float(ecu_cur.get("yaw_accel_ff_gain", 1.0)),
		"centripetal_fx_gain": float(ecu_cur.get("centripetal_fx_gain", 15.0)),
		"delta_alpha_eq_gain": float(ecu_cur.get("delta_alpha_eq_gain", 2620.0)),
	}


func get_tv_live_diagnostics() -> Dictionary:
	if model != null and model.ecu_tv_4wd != null:
		return model.ecu_tv_4wd.get_tv_live_diagnostics()
	return {
		"active_regime": "BALANCED",
		"mz_tv_nm": 0.0,
		"t_out_in_ratio": 1.0,
		"pivot_blend": 0.0,
		"beta_deg": tv_beta_est_deg,
		"beta_dot_dps": 0.0,
		"balance_state": "NEUTRAL",
		"p_dc_kw": tv_elec_power_kw,
		"mu_hat_avg": 1.50,
		"launch_state": "IDLE",
	}


func apply_live_tuning(params: Dictionary) -> Dictionary:
	if model != null and model.ecu_tv_4wd != null:
		model.ecu_tv_4wd.apply_live_tuning(params)
	if config != null:
		if params.has("power_limit_kw"):
			var p_w := clampf(float(params["power_limit_kw"]), 10.0, 80.0) * 1000.0
			config.max_dc_power_w = p_w
			config.power_limit_watts = p_w
		if params.has("regen_limit_kw"):
			var r_raw := float(params["regen_limit_kw"])
			var r_kw := clampf(r_raw if r_raw <= 0.0 else -r_raw, -25.0, 0.0)
			config.p_max_charge_w = absf(r_kw) * 1000.0
		if params.has("lift_regen_nm"):
			config.lift_off_regen_nm = clampf(float(params["lift_regen_nm"]), 0.0, 100.0)
			if model != null and model.ecu_tv_4wd != null:
				var max_whl := maxf(4.0 * model.ecu_tv_4wd.t_regen_max, 100.0)
				model.ecu_tv_4wd._mc_lift_regen_frac = clampf((4.0 * config.lift_off_regen_nm) / max_whl, 0.0, 0.50)
		if params.has("brake_bias_front_pct"):
			config.brake_bias_front = clampf(float(params["brake_bias_front_pct"]) * 0.01, 0.40, 0.75)
			if model != null and model.ecu_tv_4wd != null:
				model.ecu_tv_4wd.brake_bias_f_cfg = config.brake_bias_front
		if params.has("arb_front_balance_pct"):
			var bal := clampf(float(params["arb_front_balance_pct"]) * 0.01, 0.20, 0.80)
			var arb_tot := maxf(config.arb_stiffness_front + config.arb_stiffness_rear, 1000.0)
			config.arb_stiffness_front = arb_tot * bal
			config.arb_stiffness_rear = arb_tot * (1.0 - bal)
			_sync_setup_to_model()
			if model != null and model.ecu_tv_4wd != null:
				var kw_f := config.spring_stiffness_front * config.motion_ratio_front * config.motion_ratio_front
				var kw_r := config.spring_stiffness_rear * config.motion_ratio_rear * config.motion_ratio_rear
				var k_phi_f := 0.5 * kw_f * config.track_width_f * config.track_width_f + config.arb_stiffness_front
				var k_phi_r := 0.5 * kw_r * config.track_width_r * config.track_width_r + config.arb_stiffness_rear
				model.ecu_tv_4wd.lltd_f = clampf(k_phi_f / maxf(k_phi_f + k_phi_r, 1.0), 0.15, 0.85)
	return get_live_tuning()


func reset_map_live_tuning() -> Dictionary:
	if model != null and model.ecu_tv_4wd != null:
		model.ecu_tv_4wd.reset_live_tuning()
	if config != null:
		config.max_dc_power_w = 80000.0
		config.power_limit_watts = 80000.0
		config.p_max_charge_w = 25000.0
		config.lift_off_regen_nm = 0.0
		if config.vehicle_id == "ter26_rwd":
			config.brake_bias_front = 0.70
			config.arb_stiffness_front = 8079.2801
			config.arb_stiffness_rear = 16226.9132
		else:
			config.brake_bias_front = 0.58
			config.arb_stiffness_front = 4500.0
			config.arb_stiffness_rear = 5500.0
		_sync_setup_to_model()
		if model != null and model.ecu_tv_4wd != null:
			model.ecu_tv_4wd.configure_from_vehicle_config(config)
			model.ecu_tv_4wd._refresh_mode_cache(model.ecu_tv_4wd._get_mode_dict())
	return get_live_tuning()


func apply_track_conditions(cond: Dictionary) -> void:
	if cond.has("mu_scale"):
		track_surface_mu_scale = clampf(float(cond["mu_scale"]), 0.60, 1.15)
	if cond.has("split_mu_test_enabled"):
		split_mu_test_enabled = bool(cond["split_mu_test_enabled"])
	if cond.has("tire_temp_preset"):
		var preset := String(cond["tire_temp_preset"]).strip_edges().to_upper()
		var t_c := 55.0
		if preset == "COLD_25C":
			t_c = 25.0
			_track_temp_preset = "COLD_25C"
		elif preset == "OPT_70C":
			t_c = 70.0
			_track_temp_preset = "OPT_70C"
		else:
			t_c = 55.0
			_track_temp_preset = "WARM_55C"
		if model != null and ("wheels" in model):
			for w in model.wheels:
				w.t_surf_c = t_c
				w.t_zone_inner_c = t_c
				w.t_zone_mid_c = t_c
				w.t_zone_outer_c = t_c
				if "t_carcass_c" in w:
					w.t_carcass_c = t_c
	if cond.has("aero_scale") and config != null:
		_track_aero_scale = clampf(float(cond["aero_scale"]), 0.50, 1.15)
		config.cl_a = _base_cl_a * _track_aero_scale
		config.cd_a = _base_cd_a * _track_aero_scale
		config._has_explicit_axle_cl = false
		if model != null and model.ecu_tv_4wd != null:
			model.ecu_tv_4wd.cl_a = config.cl_a

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

static func _compute_center_of_mass(cfg: VehicleConfigScript) -> Vector3:
	if cfg == null:
		return Vector3(0.0, 0.290, -0.117)
	if cfg.vehicle_id == "ter27_4wd":
		return Vector3(0.0, cfg.h_cog, -0.117)
	var com_z: float = -(0.5 - cfg.front_weight_distribution) * cfg.wheelbase
	return Vector3(0.0, cfg.h_cog, com_z)


func _sync_setup_to_model() -> void:
	if config == null:
		return
	if model != null and model.suspension != null:
		model.suspension.sync_from_config(config)
	if model != null:
		for i in range(mini(4, model.wheels.size())):
			model.wheels[i].inflation_pressure_kpa = (config.tire_pressure_front_bar if i < 2 else config.tire_pressure_rear_bar) * 100.0
			model.wheels[i].sigma_alpha_override = config.tire_relaxation_sigma_y
	mass = config.mass
	inertia = Vector3(config.iyy, config.get_total_izz(), config.ixx)


func _refresh_suspension_visualizer() -> void:
	if susp_visualizer == null:
		return
	if not susp_visualizer.is_supported():
		susp_visualizer.set_visualizer_active(false)
	elif susp_visualizer.visible:
		susp_visualizer.update_from_car()


## cambia el solver físico en caliente entre "full_14dof" y "fast_3dof"
func set_physics_fidelity(fidelity: String) -> bool:
	var clean := fidelity.strip_edges().to_lower()
	if clean != "full_14dof" and clean != "fast_3dof":
		return false
	if clean == physics_fidelity and model != null:
		return true
	physics_fidelity = clean
	var active_vid := config.vehicle_id if config != null else "ter27_4wd"
	var active_setup := config.active_setup_name if config != null else "default"
	model = VehicleModelRegistryScript.create_model(physics_fidelity, config)
	model.load_vehicle(active_vid, active_setup)
	config = model.config
	_base_cl_a = config.cl_a
	_base_cd_a = config.cd_a
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(tv_hardware_realism)
	_sync_setup_to_model()
	_refresh_suspension_visualizer()
	if config.vehicle_id == "ter27_4wd":
		set_sim_drive_mode(SimDriveMode.AUTOCROSS)
	else:
		set_sim_drive_mode(SimDriveMode.OPEN_4WD)
	return true


## cambia en caliente el perfil de coche (e.g. "ter26_rwd" vs "ter27_4wd") and optional set-up name
func switch_vehicle(vehicle_id: String, setup_name: String = "default") -> bool:
	if model == null or not model.load_vehicle(vehicle_id, setup_name):
		return false
	config = model.config
	_base_cl_a = config.cl_a
	_base_cd_a = config.cd_a
	drive_mode = config.drive_mode
	if inputs != null:
		inputs.drive_mode = config.drive_mode
	mass = config.mass
	inertia = Vector3(config.iyy, config.get_total_izz(), config.ixx)
	center_of_mass = _compute_center_of_mass(config)
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	_sync_geometry_from_config()
	var wheel_names := ["FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL"]
	for i in range(4):
		if i < raycasts.size() and raycasts[i] != null:
			raycasts[i].position = ray_mount_positions[i]
			raycasts[i].target_position = Vector3(0.0, -ray_cast_length, 0.0)
		if has_node(wheel_names[i]):
			var wn := get_node(wheel_names[i]) as Node3D
			wn.position.x = ray_mount_positions[i].x
			wn.position.z = ray_mount_positions[i].z
			wn.position.y = ray_mount_positions[i].y - ray_rest_lengths[i] + config.wheel_radius
		_prev_compressions[i] = 0.0
		_was_grounded[i] = false
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(tv_hardware_realism)
		tv_map_id = model.ecu_tv_v3.active_map_id
		tv_map_name = model.ecu_tv_v3.get_map_name()
		if inputs != null:
			inputs.tv_map_id = tv_map_id
	_sync_setup_to_model()
	_refresh_suspension_visualizer()
	if config.vehicle_id == "ter27_4wd":
		set_sim_drive_mode(SimDriveMode.AUTOCROSS)
	else:
		set_sim_drive_mode(SimDriveMode.OPEN_4WD)
	return true


## applies a named setupprofile json for the currently active vehicle
func apply_vehicle_setup(setup_name: String) -> bool:
	if model == null or not model.apply_setup(setup_name):
		return false
	config = model.config
	_base_cl_a = config.cl_a
	_base_cd_a = config.cd_a
	drive_mode = config.drive_mode
	if inputs != null:
		inputs.drive_mode = config.drive_mode
	if model.ecu_tv_v3 != null:
		tv_map_id = model.ecu_tv_v3.active_map_id
		tv_map_name = model.ecu_tv_v3.get_map_name()
	_sync_setup_to_model()
	_refresh_suspension_visualizer()
	if config.vehicle_id == "ter27_4wd":
		set_sim_drive_mode(SimDriveMode.AUTOCROSS)
	else:
		set_sim_drive_mode(SimDriveMode.OPEN_4WD)
	return true


## saves the active vehicle's parameters into a modular setupprofile json
func save_vehicle_setup(setup_name: String) -> String:
	if config == null:
		return ""
	var tv_file := ""
	var law_mode := "smc"
	if model != null and model.ecu_tv_v3 != null:
		tv_file = model.ecu_tv_v3.active_map_path.get_file()
		law_mode = "smc" if model.ecu_tv_v3.control_law_mode == model.ecu_tv_v3.ControlLawMode.V4_SMC_HYBRID else "v3"
	var prof = SetupProfileScript.from_vehicle_config(config, setup_name, tv_file, law_mode)
	var saved_path: String = str(prof.save_for_vehicle(config.vehicle_id, setup_name))
	if saved_path != "":
		config.active_setup_name = prof.name
	return saved_path

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

func set_input_profile(p: int) -> void:
	if input_conditioner != null:
		input_conditioner.set_profile(p)


func get_input_profile() -> int:
	return input_conditioner.profile if input_conditioner != null else 0


func _process_manual_pilot(delta: float) -> void:
	if wheel_ffb != null:
		wheel_ffb.input_device_override = input_device_mode
	var joypads: Array[int] = Input.get_connected_joypads()
	var has_joypad: bool = joypads.size() > 0
	var kb_steer: float = Input.get_action_strength("manual_steer_l") - Input.get_action_strength("manual_steer_r")
	var kb_throttle: float = clampf(Input.get_action_strength("manual_gas_pos"), 0.0, 1.0)
	var kb_brake: float = clampf(Input.get_action_strength("manual_gas_neg"), 0.0, 1.0)
	var ebs_in: float = clampf(Input.get_action_strength("EBS"), 0.0, 1.0)

	var raw_steer := kb_steer
	var raw_throttle := kb_throttle
	var raw_brake := kb_brake
	var is_wheel := false
	var joy_id := -1
	var raw_joy_act := false
	var raw_kb_act := (absf(kb_steer) > 0.01 or kb_throttle > 0.01 or kb_brake > 0.01)

	if has_joypad:
		joy_id = joypads[0]
		if wheel_ffb != null:
			is_wheel = (wheel_ffb.detect_input_device(joy_id) == 2)
		var raw_x: float = -Input.get_joy_axis(joy_id, JOY_AXIS_LEFT_X)
		var joy_throttle: float = clampf(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_RIGHT), 0.0, 1.0)
		var joy_brake: float = clampf(Input.get_joy_axis(joy_id, JOY_AXIS_TRIGGER_LEFT), 0.0, 1.0)
		raw_joy_act = (absf(raw_x) > 0.06 or joy_throttle > 0.02 or joy_brake > 0.02)
		if raw_joy_act and not raw_kb_act:
			raw_steer = raw_x
			raw_throttle = joy_throttle
			raw_brake = joy_brake
		elif raw_joy_act:
			raw_steer = raw_x if absf(raw_x) > absf(kb_steer) else kb_steer
			raw_throttle = maxf(kb_throttle, joy_throttle)
			raw_brake = maxf(kb_brake, joy_brake)
	elif wheel_ffb != null:
		wheel_ffb.detect_input_device_from_name("")

	if (raw_throttle > 0.05 and raw_brake < 0.05) or (raw_throttle > 0.70 and raw_brake < 0.20):
		raw_brake = 0.0

	_raw_steer_in = raw_steer
	_raw_throttle_in = raw_throttle
	_raw_brake_in = maxf(raw_brake, ebs_in)

	var vx_mps: float = linear_velocity.length()
	var wz_cur: float = model.angular_velocity_ros.z if model != null else 0.0
	var beta_cur: float = 0.0
	if model != null and model.chassis_state != null:
		var lv_c: Vector3 = model.chassis_state.linear_velocity
		beta_cur = atan2(lv_c.y, maxf(absf(lv_c.x), 0.5))

	input_conditioner.resolve_profile(has_joypad, is_wheel, raw_joy_act, raw_kb_act)
	var cond := input_conditioner.step(
		raw_steer, raw_throttle, raw_brake,
		vx_mps, wz_cur, beta_cur, delta,
		config.max_steer_angle, config.wheelbase
	)
	_steer_smoothed = float(cond["steer_norm"])
	raw_steer_norm = float(cond.get("raw_steer_norm", raw_steer))
	self.steering = float(cond["steer_angle"])
	self.gas = float(cond["gas"])
	self.brake = maxf(float(cond["brake"]), ebs_in)

	if ffb_haptics_enabled and has_joypad and model != null and DisplayServer.get_name() != "headless":
		if is_wheel and wheel_ffb != null:
			wheel_ffb.apply_haptic_feedback(joy_id, model.wheels, linear_velocity.length(), delta)
			ffb_torque_nm = float(model.get("steering_torque_nm")) if ("steering_torque_nm" in model) else wheel_ffb.ffb_torque_nm
			_joy_vibrating = absf(ffb_torque_nm) > 0.05
		else:
			var r_scale: float = clampf(input_conditioner.rumble_intensity if input_conditioner != null else rumble_effects, 0.0, 1.0)
			var sat_f_h: float = 0.005 * (_tire_sat_pct_scratch[0] + _tire_sat_pct_scratch[1])
			var sat_r_h: float = 0.005 * (_tire_sat_pct_scratch[2] + _tire_sat_pct_scratch[3])
			var curb_hit_r: float = 0.45 if (model.ecu_tv_4wd != null and model.ecu_tv_4wd._curb_timer_s > 0.0) else 0.0
			var tc_pulse: float = 0.25 if tv_tc_active else 0.0
			var abs_pulse: float = 0.35 if (self.brake > 0.05 and model.ecu_tv_4wd != null and (model.ecu_tv_4wd._qp_w_abs[0] > 0.0 or model.ecu_tv_4wd._qp_w_abs[1] > 0.0 or model.ecu_tv_4wd._qp_w_abs[2] > 0.0 or model.ecu_tv_4wd._qp_w_abs[3] > 0.0)) else 0.0
			var weak_rumble := clampf(maxf((sat_f_h - 0.72) / 0.28, maxf(tc_pulse, abs_pulse)), 0.0, 1.0) * r_scale
			var strong_rumble := clampf(maxf((sat_r_h - 0.72) / 0.28, maxf(_last_bump_rumble, curb_hit_r)), 0.0, 1.0) * r_scale
			if weak_rumble > 0.02 or strong_rumble > 0.02:
				Input.start_joy_vibration(joy_id, weak_rumble, strong_rumble, 0.05)
				_joy_vibrating = true
			elif _joy_vibrating:
				_stop_all_joy_vibration()
	elif _joy_vibrating:
		_stop_all_joy_vibration()

func _process_ros_pilot() -> void:
	self.gas = _ros_gas
	self.steering = clampf(_ros_steer, -config.max_steer_angle, config.max_steer_angle)
	self.brake = _ros_brake
	raw_steer_norm = clampf(self.steering / maxf(config.max_steer_angle, 1e-4), -1.0, 1.0)

## helper for programmatic and test driving
func set_control(gas_val: float, steer_val: float, brake_val: float = 0.0) -> void:
	self.gas = gas_val
	self.steering = steer_val
	self.brake = clampf(brake_val, 0.0, 1.0)
	raw_steer_norm = clampf(steer_val / maxf(config.max_steer_angle, 1e-4), -1.0, 1.0)
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

# ros callbacks and publishing
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

# gate crossing & cone timing
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
	_tv_overlay = TvLiveTuningOverlayScript.new()
	_tv_overlay.bind_car(model.ecu_tv_v3 if model != null else null, null, self)
	add_child(_tv_overlay)
	_telemetry_inspector = TelemetryInspectorOverlayScript.new()
	if _telemetry_inspector.has_method("bind_car"):
		_telemetry_inspector.bind_car(self)
	add_child(_telemetry_inspector)


func _update_minimal_hud() -> void:
	if _telemetry_inspector != null and _telemetry_inspector.visible:
		_telemetry_inspector.update_from_car(self, 1.0 / 60.0)
	if _tv_overlay != null and _tv_overlay.visible and model != null:
		_tv_overlay.update_telemetry(tv_wz_ref, model.angular_velocity_ros.z, tv_beta_est_deg, tv_mz_req, tv_tc_active)
		_tv_overlay.update_input_telemetry(_raw_steer_in, _steer_smoothed, _raw_throttle_in, self.gas, _raw_brake_in, self.brake)
		if _tv_overlay.has_method("update_chassis_telemetry"):
			_tv_overlay.update_chassis_telemetry(_collect_imo_temps(), _collect_disc_temps(), _get_active_surface_mu())


func get_hud_snapshot() -> Dictionary:
	var kmh: float = linear_velocity.length() * 3.6
	var beta_deg: float = 0.0
	if model != null and model.chassis_state != null:
		var lv: Vector3 = model.chassis_state.linear_velocity
		beta_deg = rad_to_deg(atan2(lv.y, maxf(absf(lv.x), 0.5)))

	var mu_min_val: float = 1.50
	var tc_on: bool = tv_tc_active
	var abs_on: bool = false
	var curb_on: bool = false
	var limp_on: bool = false
	var mz_sat_pct: float = 0.0
	var mbs: float = 0.80
	if model != null and model.ecu_tv_4wd != null and ecu_driving_mode == 2:
		var e4 = model.ecu_tv_4wd
		tc_on = e4.tc_active
		limp_on = e4.limp_home_active
		curb_on = e4._curb_timer_s > 0.0
		mz_sat_pct = clampf(e4.mz_sat_ratio * 100.0, 0.0, 100.0)
		mbs = clampf(e4.mech_brake_start, 0.05, 0.95)
		mu_min_val = minf(minf(e4.mu_hat[0], e4.mu_hat[1]), minf(e4.mu_hat[2], e4.mu_hat[3]))
		abs_on = self.brake > 0.05 and (e4._qp_w_abs[0] > 0.0 or e4._qp_w_abs[1] > 0.0 or e4._qp_w_abs[2] > 0.0 or e4._qp_w_abs[3] > 0.0)

	var v_dc_v: float = model.hv_state.v_dc if (model != null and model.hv_state != null) else 540.0
	var soc_pct: float = (model.hv_state.soc * 100.0) if (model != null and model.hv_state != null) else 95.0
	var rec_ind: String = telemetry_recorder.get_hud_indicator() if telemetry_recorder != null else ""
	var corners: Array = susp_visualizer.get_corner_metrics() if (susp_visualizer != null and susp_visualizer.has_method("get_corner_metrics")) else []
	var kin_summary: Dictionary = {}
	if model != null and ("suspension" in model) and model.suspension != null and config != null:
		var st_ang: float = inputs.steer_angle if inputs != null else 0.0
		var k3d = model.suspension.kinematics_3d if ("kinematics_3d" in model.suspension) else null
		kin_summary = DoubleWishboneKinematicsScript.compute_telemetry_summary(model.suspension.corners, st_ang, config, k3d)
		if corners.is_empty():
			var c_names := ["FL", "FR", "RL", "RR"]
			for ci in range(mini(4, model.suspension.corners.size())):
				var sc = model.suspension.corners[ci]
				corners.append({
					"name": c_names[ci],
					"dz_mm": float(sc.delta_z) * 1000.0,
					"camber_deg": rad_to_deg(float(sc.camber_rad)),
					"fz_n": float(sc.fz),
					"f_arb_n": float(sc.arb_force),
				})
	var max_disc_c: float = 25.0
	for dt_c in _collect_disc_temps():
		max_disc_c = maxf(max_disc_c, float(dt_c))
	var max_tire_c: float = 25.0
	for tt_arr in _collect_imo_temps():
		if tt_arr is Array:
			for tt_v in tt_arr:
				max_tire_c = maxf(max_tire_c, float(tt_v))
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	var trk_node: Node = sim_node.get("track") if (sim_node != null and "track" in sim_node) else (get_parent().get_node_or_null("Track") if get_parent() != null else null)
	var dil_sum: Dictionary = trk_node.dil_session.get_summary() if (trk_node != null and "dil_session" in trk_node and trk_node.dil_session != null) else {}
	var road_cfg: Dictionary = {}
	if model != null and ("road_surface" in model) and model.road_surface != null:
		var rs = model.road_surface
		road_cfg = {
			"surface_mode": rs.surface_mode,
			"roughness_scale": rs.roughness_scale,
			"bump_mode": rs.bump_mode,
			"bump_height_mm": rs.bump_height_m * 1000.0,
			"bump_spacing_m": rs.bump_spacing_m,
		}
	return {
		"vehicle_id": config.vehicle_id if config != null else "ter27_4wd",
		"physics_fidelity": physics_fidelity,
		"kinematics_3d_supported": is_kinematics_3d_supported(),
		"kinematics_3d_enabled": is_kinematics_3d_enabled(),
		"active_setup_name": config.active_setup_name if config != null else "default",
		"input_profile": input_conditioner.profile if input_conditioner != null else 0,
		"mode_id": sim_drive_mode,
		"mode_name": get_sim_drive_mode_name(),
		"kmh": kmh,
		"beta_deg": beta_deg,
		"ax_g": _last_ax_g,
		"ay_g": _last_ay_g,
		"mu_min": mu_min_val,
		"p_dc_kw": tv_elec_power_kw,
		"v_dc_v": v_dc_v,
		"soc_pct": soc_pct,
		"brake": clampf(self.brake, 0.0, 1.0),
		"mech_brake_start": mbs,
		"max_disc_c": max_disc_c,
		"max_tire_c": max_tire_c,
		"tc": tc_on,
		"abs": abs_on,
		"curb": curb_on,
		"limp": limp_on,
		"mz_sat_pct": mz_sat_pct,
		"rec_tag": rec_ind,
		"rec_active": telemetry_recorder != null and telemetry_recorder.is_recording,
		"tv_tune_open": _tv_overlay != null and _tv_overlay.visible,
		"scope_open": _telemetry_inspector != null and _telemetry_inspector.visible,
		"wheel_arrows": wheel_arrows_enabled,
		"fz_arrows": wheel_forces_vis != null and wheel_forces_vis.show_fz_arrows,
		"torque_bars": wheel_forces_vis != null and wheel_forces_vis.show_torque_bars,
		"racing_line": racing_line_enabled,
		"racing_line_markers": _racing_line_markers_enabled,
		"ghost_mode": int(dil_sum.get("ghost_mode", 0)),
		"dil_audio": dil_audio != null and dil_audio.enabled,
		"dil_audio_vol": dil_audio.volume if dil_audio != null else 0.55,
		"haptics_enabled": ffb_haptics_enabled,
		"steering_torque_nm": ffb_torque_nm,
		"tire_sat_pct": _tire_sat_pct_scratch,
		"tire_temps_c": _tire_temps_scratch,
		"road_surface_cfg": road_cfg,
		"track_grip_profile": int(model.track_grip_profile) if (model != null and ("track_grip_profile" in model)) else 0,
		"dil_summary": dil_sum,
		"track_mu": track_surface_mu_scale,
		"session_time_s": _session_time_s,
		"session_energy_kj": _session_energy_kj,
		"susp_supported": susp_visualizer != null and susp_visualizer.is_supported(),
		"susp_visible": susp_visualizer != null and susp_visualizer.visible,
		"xray_body_only": susp_visualizer != null and susp_visualizer.is_xray_body_only(),
		"always_on_top": susp_visualizer != null and susp_visualizer.is_always_on_top(),
		"corners": corners,
		"kinematics_summary": kin_summary,
	}


func _get_track_node() -> Node:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and "track" in sim_node and sim_node.get("track") != null:
		return sim_node.get("track")
	return get_parent().get_node_or_null("Track") if get_parent() != null else null


func set_fz_arrows_enabled(enabled: bool) -> void:
	if wheel_forces_vis != null:
		wheel_forces_vis.set_fz_arrows_enabled(enabled)


func set_torque_bars_enabled(enabled: bool) -> void:
	if wheel_forces_vis != null:
		wheel_forces_vis.set_torque_bars_enabled(enabled)


func set_racing_line_enabled(enabled: bool) -> void:
	racing_line_enabled = enabled
	var trk := _get_track_node()
	if trk != null and ("racing_line_vis" in trk) and trk.racing_line_vis != null:
		trk.racing_line_vis.visible = enabled
		trk.racing_line_vis.set_markers_visible(_racing_line_markers_enabled)


func set_racing_line_markers_enabled(enabled: bool) -> void:
	_racing_line_markers_enabled = enabled
	var trk := _get_track_node()
	if trk != null and ("racing_line_vis" in trk) and trk.racing_line_vis != null:
		trk.racing_line_vis.set_markers_visible(enabled)


func set_ghost_mode(mode: int) -> void:
	var trk := _get_track_node()
	if trk != null and ("dil_session" in trk) and trk.dil_session != null:
		trk.dil_session.set_ghost_mode(mode)


func set_dil_audio_enabled(enabled: bool) -> void:
	if is_instance_valid(dil_audio):
		dil_audio.set_enabled(enabled)


func set_dil_audio_volume(vol_0_1: float) -> void:
	if is_instance_valid(dil_audio):
		dil_audio.set_volume(vol_0_1)


func set_track_grip_profile(profile_id: int) -> void:
	if model != null and model.has_method("set_track_grip_profile"):
		model.set_track_grip_profile(profile_id)
	var trk := _get_track_node()
	if trk != null and ("last_loaded_track_path" in trk):
		var t_path := String(trk.last_loaded_track_path).to_lower()
		if "fsg_2023" in t_path:
			var rl_path := "res://Assets/RacingLines/fsg_2023_racing_line_real.json" if profile_id == 1 else "res://Assets/RacingLines/fsg_2023_racing_line.json"
			if FileAccess.file_exists(rl_path):
				if ("racing_line_vis" in trk) and trk.racing_line_vis != null:
					var samples: Array = trk.racing_line_vis.load_racing_line_json(rl_path)
					if not samples.is_empty():
						trk.racing_line_vis.update_from_samples(samples, true)
				if ("dil_session" in trk) and trk.dil_session != null:
					trk.dil_session.load_lts_trace_from_json(rl_path)


func set_haptics_enabled(enabled: bool) -> void:
	ffb_haptics_enabled = enabled
	if not enabled and _joy_vibrating:
		_stop_all_joy_vibration()


func sync_road_surface_from_track(trk: Node = null) -> void:
	var t := trk if trk != null else _get_track_node()
	if model == null or not ("road_surface" in model) or model.road_surface == null or t == null:
		return
	var rs = model.road_surface
	if "track_curve" in t and t.track_curve != null and t.track_curve.get_baked_length() > 1.0:
		var crv: Curve3D = t.track_curve
		var total_l := crv.get_baked_length()
		var n_pts := clampi(int(total_l / 3.0), 16, 400)
		var px := PackedFloat64Array(); px.resize(n_pts)
		var py := PackedFloat64Array(); py.resize(n_pts)
		for i in range(n_pts):
			var p3 := crv.sample_baked(float(i) * total_l / float(n_pts), false)
			px[i] = -p3.z
			py[i] = -p3.x
		rs.set_track_polyline_ros(px, py)
	else:
		rs.clear_track_polyline()
	if "racing_line_vis" in t and t.racing_line_vis != null:
		t.racing_line_vis.update_bump_markers(rs.get_bump_markers())


func set_road_surface_config(
	surface_mode: int,
	roughness_scale: float,
	bump_mode: int,
	bump_height_mm: float,
	bump_spacing_m: float
) -> void:
	if model == null or not ("road_surface" in model) or model.road_surface == null:
		return
	model.road_surface.configure(surface_mode, roughness_scale, bump_mode, bump_height_mm, 0.80, bump_spacing_m)
	var trk := _get_track_node()
	if trk != null and ("racing_line_vis" in trk) and trk.racing_line_vis != null:
		trk.racing_line_vis.update_bump_markers(model.road_surface.get_bump_markers())


func is_kinematics_3d_supported() -> bool:
	return (
		config != null
		and not config.double_wishbone_kinematics.is_empty()
		and physics_fidelity == "full_14dof"
	)


func is_kinematics_3d_enabled() -> bool:
	return (
		config != null
		and config.use_3d_kinematics
		and model != null
		and ("suspension" in model)
		and model.suspension != null
		and model.suspension.use_3d_kinematics
	)


func set_kinematics_3d_enabled(enabled: bool) -> bool:
	if config == null or model == null or not ("suspension" in model) or model.suspension == null:
		return false
	if enabled and not is_kinematics_3d_supported():
		return false
	config.use_3d_kinematics = enabled
	model.suspension.sync_from_config(config)
	_refresh_suspension_visualizer()
	return model.suspension.use_3d_kinematics == enabled


func list_setup_profiles() -> Array[Dictionary]:
	if config == null:
		return []
	return SetupProfileScript.list_setups_for_vehicle(config.vehicle_id)


func set_track_surface_mu(mu_val: float) -> void:
	track_surface_mu_scale = clampf(mu_val, 0.20, 1.50)


func toggle_recording_shortcut() -> bool:
	if telemetry_recorder == null:
		return false
	var rec_map_name := tv_map_name
	if model != null and model.ecu_tv_v3 != null and model.ecu_tv_v3.active_map_filename != "":
		rec_map_name = model.ecu_tv_v3.active_map_filename
	var rec_on: bool = telemetry_recorder.toggle_recording(_get_active_track_name(), rec_map_name)
	return rec_on


func toggle_tv_tune_overlay() -> bool:
	if _tv_overlay == null:
		return false
	_tv_overlay.toggle_visibility()
	return _tv_overlay.visible


func toggle_telemetry_scope() -> bool:
	if _telemetry_inspector == null:
		return false
	_telemetry_inspector.toggle_visibility()
	return _telemetry_inspector.visible


func get_camera_names() -> Array[String]:
	var names: Array[String] = []
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and "_cameras" in sim_node:
		for cam in sim_node._cameras:
			if is_instance_valid(cam):
				names.append(String(cam.name))
	return names


func get_active_camera_index() -> int:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and "_active_index" in sim_node:
		return int(sim_node._active_index)
	return -1


func select_camera_index(idx: int) -> void:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and sim_node.has_method("switch_to_index"):
		sim_node.switch_to_index(idx)


func focus_suspension_cam() -> void:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node != null and "_cameras" in sim_node and _susp_cam != null:
		var idx: int = sim_node._cameras.find(_susp_cam)
		if idx != -1 and sim_node.has_method("switch_to_index"):
			sim_node.switch_to_index(idx)


func set_suspension_visualizer_enabled(enabled: bool) -> bool:
	if susp_visualizer == null:
		return false
	return susp_visualizer.set_visualizer_active(enabled)


func set_suspension_xray_only(enabled: bool) -> void:
	if susp_visualizer != null:
		susp_visualizer.set_xray_body_only(enabled)


func set_suspension_always_on_top(enabled: bool) -> void:
	if susp_visualizer != null:
		susp_visualizer.set_always_on_top(enabled)

func launch_control_arm() -> bool:
	if model == null or model.ecu_tv_v3 == null:
		return false
	var vx_cur: float = model.chassis_state.linear_velocity.x if model.chassis_state != null else 0.0
	var armed: bool = model.ecu_tv_v3.request_launch_control_arm(vx_cur, brake)
	if armed:
		print("[Launch Control]: ARMED (preload 8 Nm) — Mantén 100% Gas y Suelta Freno")
	else:
		print("[Launch Control]: Requiere coche detenido (vx < 0.5 m/s) y freno pisado (> 25%)")
	_update_minimal_hud()
	return armed

# interface & utility methods
func set_origin(target_transform: Transform3D, reset_vel: bool = false) -> void:
	origin = target_transform
	if reset_vel:
		linear_velocity = Vector3.ZERO
		angular_velocity = Vector3.ZERO
	await get_tree().process_frame
	if reset_vel:
		linear_velocity = Vector3.ZERO
		angular_velocity = Vector3.ZERO
	set_deferred("global_transform", origin)

func reset_position() -> void:
	var sim_node = get_node_or_null("/root/Sim") if is_inside_tree() else null
	if sim_node and sim_node.get("track"):
		var trk_r: Node = sim_node.get("track")
		set_origin(trk_r.origin, true)
		if "dil_session" in trk_r and trk_r.dil_session != null:
			trk_r.dil_session.reset_session(true)
	self.gas = 0.0
	self.steering = 0.0
	self.brake = 0.0
	_steer_smoothed = 0.0
	_filtered_steer_angle = 0.0
	if input_conditioner != null:
		input_conditioner.reset()
	if model != null:
		model.reset()
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
	_session_time_s = 0.0
	_session_energy_kj = 0.0
	_last_ax_g = 0.0
	_last_ay_g = 0.0
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
		# fallback to tpcam or distance check
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
