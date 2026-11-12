class_name VehicleConfig
extends Resource

## Vehicle Configuration & Physical Parameters for Mirena Dynamics
## Loads physical parameters from modular JSON files in Core/Config/
## and provides typed properties with safe defaults for simulation subsystems.

enum DriveMode {
	FOUR_WD,
	REAR_WD,
	FRONT_WD,
	TORQUE_VECTORING
}

enum WheelIndex {
	FL = 0,
	FR = 1,
	RL = 2,
	RR = 3
}

# --- Chassis & Mass Geometry ---
@export var mass: float = 300.0
@export var gravity: float = 9.81
@export var wheelbase: float = 1.530
@export var track_width_f: float = 1.200
@export var track_width_r: float = 1.180
@export var h_cog: float = 0.33
@export var weight_dist_r: float = 0.55

@export var ixx: float = 60.0
@export var iyy: float = 110.0
@export var izz: float = 130.0

var a_cog: float = 0.8262
var b_cog: float = 0.7038

# --- Suspension & Raycast ---
@export var spring_stiffness_front: float = 45000.0
@export var spring_stiffness_rear: float = 40000.0
@export var damping_compression_front: float = 6700.0
@export var damping_relaxation_front: float = 2700.0
@export var damping_compression_rear: float = 6700.0
@export var damping_relaxation_rear: float = 2700.0
@export var arb_stiffness_front: float = 2000.0
@export var arb_stiffness_rear: float = 1000.0
@export var ray_rest_length: float = 0.32
@export var ray_cast_length: float = 0.45

# --- Powertrain & Drivetrain ---
@export var drive_mode: DriveMode = DriveMode.FOUR_WD
@export var power_limit_watts: float = 80000.0
@export var motor_peak_torque: float = 100.0
@export var gear_ratio: float = 5.0
@export var drivetrain_efficiency: float = 0.92
@export var front_drive_torque_ratio: float = 0.50
@export var max_steer_angle: float = 0.35

# --- Brakes ---
@export var brake_mu: float = 1.30
@export var brake_bias_front: float = 0.60

# --- Tires ---
@export var wheel_radius: float = 0.20
@export var wheel_inertia: float = 0.35
@export var tyre_mu_x: float = 1.6
@export var tyre_mu_y: float = 1.5
@export var slip_stiffness: float = 80000.0
@export var cornering_stiffness: float = 45000.0
@export var rolling_resistance_coeff: float = 0.015
@export var slip_regularization_speed: float = 0.5

# --- Aerodynamics ---
@export var air_density: float = 1.225
@export var cd_a: float = 2.75
@export var cl_a: float = 4.554

func _init() -> void:
	load_configs()
	update_derived_geometry()

## Loads configuration values from modular JSON files.
func load_configs(custom_config_dir: String = "") -> void:
	var base_dirs = [
		custom_config_dir,
		"res://Scenes/Vehicle/Dynamics/Core/Config/",
		"src/mirena_sim/Scenes/Vehicle/Dynamics/Core/Config/",
		"Scenes/Vehicle/Dynamics/Core/Config/",
		"Core/Config/"
	]
	
	var found_dir = ""
	for d in base_dirs:
		if d != "" and (DirAccess.dir_exists_absolute(d) or FileAccess.file_exists(d + "chassis.json")):
			found_dir = d.rstrip("/") + "/"
			break
	
	if found_dir == "":
		update_derived_geometry()
		return
	
	_load_chassis_json(found_dir + "chassis.json")
	_load_suspension_json(found_dir + "suspension.json")
	_load_tires_json(found_dir + "tires.json")
	_load_drivetrain_json(found_dir + "drivetrain.json")
	_load_brakes_json(found_dir + "brakes.json")
	_load_aero_json(found_dir + "aero.json")
	
	update_derived_geometry()

func _read_json_file(file_path: String) -> Dictionary:
	if not FileAccess.file_exists(file_path):
		return {}
	var f = FileAccess.open(file_path, FileAccess.READ)
	if not f:
		return {}
	var text = f.get_as_text()
	var res = JSON.parse_string(text)
	if res is Dictionary:
		return res
	return {}

func _load_chassis_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	mass = float(d.get("mass", mass))
	gravity = float(d.get("gravity", gravity))
	wheelbase = float(d.get("wheelbase", wheelbase))
	track_width_f = float(d.get("track_width_front", track_width_f))
	track_width_r = float(d.get("track_width_rear", track_width_r))
	h_cog = float(d.get("cg_height", h_cog))
	weight_dist_r = float(d.get("weight_distribution_rear", weight_dist_r))
	
	var inertia = d.get("inertia", {})
	if inertia is Dictionary:
		ixx = float(inertia.get("ixx", ixx))
		iyy = float(inertia.get("iyy", iyy))
		izz = float(inertia.get("izz", izz))

func _load_suspension_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	spring_stiffness_front = float(d.get("spring_stiffness_front", spring_stiffness_front))
	spring_stiffness_rear = float(d.get("spring_stiffness_rear", spring_stiffness_rear))
	damping_compression_front = float(d.get("damping_compression_front", damping_compression_front))
	damping_relaxation_front = float(d.get("damping_relaxation_front", damping_relaxation_front))
	damping_compression_rear = float(d.get("damping_compression_rear", damping_compression_rear))
	damping_relaxation_rear = float(d.get("damping_relaxation_rear", damping_relaxation_rear))
	arb_stiffness_front = float(d.get("arb_stiffness_front", arb_stiffness_front))
	arb_stiffness_rear = float(d.get("arb_stiffness_rear", arb_stiffness_rear))
	ray_rest_length = float(d.get("ray_rest_length", ray_rest_length))
	ray_cast_length = float(d.get("ray_cast_length", ray_cast_length))

func _load_tires_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	wheel_radius = float(d.get("wheel_radius", wheel_radius))
	wheel_inertia = float(d.get("wheel_inertia", wheel_inertia))
	tyre_mu_x = float(d.get("tyre_mu_x", tyre_mu_x))
	tyre_mu_y = float(d.get("tyre_mu_y", tyre_mu_y))
	slip_stiffness = float(d.get("slip_stiffness", slip_stiffness))
	cornering_stiffness = float(d.get("cornering_stiffness", cornering_stiffness))
	rolling_resistance_coeff = float(d.get("rolling_resistance_coeff", rolling_resistance_coeff))
	slip_regularization_speed = float(d.get("slip_regularization_speed", slip_regularization_speed))

func _load_drivetrain_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	power_limit_watts = float(d.get("power_limit_watts", power_limit_watts))
	motor_peak_torque = float(d.get("motor_peak_torque", motor_peak_torque))
	gear_ratio = float(d.get("gear_ratio", gear_ratio))
	drivetrain_efficiency = float(d.get("drivetrain_efficiency", drivetrain_efficiency))
	front_drive_torque_ratio = float(d.get("front_drive_torque_ratio", front_drive_torque_ratio))
	max_steer_angle = float(d.get("max_steer_angle", max_steer_angle))
	
	var dm_str = str(d.get("drive_mode", "FOUR_WD"))
	match dm_str:
		"REAR_WD": drive_mode = DriveMode.REAR_WD
		"FRONT_WD": drive_mode = DriveMode.FRONT_WD
		"TORQUE_VECTORING": drive_mode = DriveMode.TORQUE_VECTORING
		_: drive_mode = DriveMode.FOUR_WD

func _load_brakes_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	brake_mu = float(d.get("brake_mu", brake_mu))
	brake_bias_front = float(d.get("brake_bias_front", brake_bias_front))

func _load_aero_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	air_density = float(d.get("air_density", air_density))
	cd_a = float(d.get("cd_a", cd_a))
	cl_a = float(d.get("cl_a", cl_a))

## Recalculates longitudinal CoG distances based on wheelbase and weight distribution.
func update_derived_geometry() -> void:
	a_cog = wheelbase * weight_dist_r
	b_cog = wheelbase * (1.0 - weight_dist_r)

## Convert vector from Godot (+X Right, +Y Up, -Z Forward) to ROS (+X Forward, +Y Left, +Z Up).
static func godot_to_ros_vec(v: Vector3) -> Vector3:
	return Vector3(-v.z, -v.x, v.y)

## Convert vector from ROS (+X Forward, +Y Left, +Z Up) to Godot (+X Right, +Y Up, -Z Forward).
static func ros_to_godot_vec(v: Vector3) -> Vector3:
	return Vector3(-v.y, v.z, -v.x)

## Returns static normal load (N) on front and rear axles at rest on level ground.
func get_static_axle_loads() -> Dictionary:
	var total_weight = mass * gravity
	var fz_front_axle = total_weight * (b_cog / wheelbase)
	var fz_rear_axle = total_weight * (a_cog / wheelbase)
	return {
		"front_axle": fz_front_axle,
		"rear_axle": fz_rear_axle,
		"wheel_fl": fz_front_axle * 0.5,
		"wheel_fr": fz_front_axle * 0.5,
		"wheel_rl": fz_rear_axle * 0.5,
		"wheel_rr": fz_rear_axle * 0.5
	}
