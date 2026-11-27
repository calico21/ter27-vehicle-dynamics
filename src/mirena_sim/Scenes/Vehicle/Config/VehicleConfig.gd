class_name VehicleConfig
extends Resource

## vehicle configuration & physical parameters for mirena dynamics
## loads physical parameters from modular json files in config/vehicles/ter27_4wd/
## and provides typed properties with safe defaults for simulation subsystems

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

# vehicle identity & set-up
@export var vehicle_id: String = "ter27_4wd"
@export var display_name: String = "TeR27 4WD"
@export var active_setup_name: String = "default"
var active_config_dir: String = ""

# chassis & mass geometry
@export var mass: float = 290.0
@export var sprung_mass: float = 236.0
@export var unsprung_mass: float = 13.5
@export var unsprung_mass_total: float = 54.0
@export var gravity: float = 9.81
@export var wheelbase: float = 1.540
@export var track_width_f: float = 1.230
@export var track_width_r: float = 1.230
@export var h_cog: float = 0.290
@export var h_sprung_cog: float = 0.30986
@export var roll_center_height_front: float = 0.01959
@export var roll_center_height_rear: float = 0.04003
@export var front_weight_distribution: float = 0.45
@export var weight_dist_r: float = 0.55
@export var ackermann_factor: float = -0.11

@export var ixx: float = 15.4
@export var iyy: float = 84.2
@export var izz: float = 83.1
@export var ixz: float = -6.4

var a_cog: float = 0.847
var b_cog: float = 0.693
var roll_center_height_cg: float = 0.030832
var h_roll_arm: float = 0.279028

# susp & raycast
# ter27 design values: spring stiffness & constant motion ratios (mr = dx_spring / dz_wheel = 1 / 1.13)
@export var spring_stiffness_front: float = 35000.0
@export var spring_stiffness_rear: float = 48000.0
@export var motion_ratio_front: float = 0.884956
@export var motion_ratio_rear: float = 0.884956
# estimated / configurable damper, arb, bumpstop & geometry parameters
@export var damping_compression_front: float = 2000.0
@export var damping_relaxation_front: float = 3000.0
@export var damping_compression_rear: float = 2000.0
@export var damping_relaxation_rear: float = 3000.0
@export var damper_knee_velocity: float = 0.05
@export var damper_comp_low: float = 2600.0
@export var damper_reb_low: float = 3800.0
@export var damper_comp_high: float = 1100.0
@export var damper_reb_high: float = 1600.0
var use_digressive_damper: bool = true
@export var arb_stiffness_front: float = 4500.0
@export var arb_stiffness_rear: float = 5500.0
@export var anti_dive_ratio_front: float = 0.569
@export var anti_squat_ratio_rear: float = 0.412
@export var jacking_coeff_front: float = 0.0323
@export var jacking_coeff_rear: float = 0.0651
@export var mechanical_trail_front: float = 0.013
@export var scrub_radius_front: float = 0.012
var _syncing_compliance: bool = false
@export var steering_compliance_stiffness_nm_rad: float = 9500.0:
	set(v):
		steering_compliance_stiffness_nm_rad = maxf(1.0, v)
		if not _syncing_compliance:
			_syncing_compliance = true
			steering_compliance_coeff_rad_nm = 1.0 / steering_compliance_stiffness_nm_rad
			_syncing_compliance = false
@export var steering_compliance_coeff_rad_nm: float = 1.0 / 9500.0:
	set(v):
		steering_compliance_coeff_rad_nm = maxf(0.0, v)
		if not _syncing_compliance:
			_syncing_compliance = true
			if steering_compliance_coeff_rad_nm > 1e-9:
				steering_compliance_stiffness_nm_rad = 1.0 / steering_compliance_coeff_rad_nm
			_syncing_compliance = false
@export var bumpstop_start_front: float = 0.025
@export var bumpstop_start_rear: float = 0.025
@export var bumpstop_stiffness_front: float = 120000.0
@export var bumpstop_stiffness_rear: float = 150000.0
@export var bumpstop_quadratic_front: float = 2000000.0
@export var bumpstop_quadratic_rear: float = 2500000.0
@export var ray_rest_length: float = 0.32
@export var ray_cast_length: float = 0.45
@export var caster_angle_deg: float = 5.0
@export var kpi_angle_deg: float = 10.0
@export var static_camber_front_deg: float = -2.0
@export var static_camber_rear_deg: float = -1.5
@export var toe_front_deg: float = -0.15
@export var toe_rear_deg: float = 0.20

# optimum kinematics 1d tables stored internally as [[delta_z_m, angle_rad], ...]
var camber_table_front: Array = []
var camber_table_rear: Array = []
var toe_table_front: Array = []
var toe_table_rear: Array = []

# 3d double-wishbone kinematics hardpoints & activation toggle (backward compatible default: false)
@export var use_3d_kinematics: bool = false
var double_wishbone_kinematics: Dictionary = {}

# powertrain & drivetrain
@export var drive_mode: DriveMode = DriveMode.FOUR_WD
@export var power_limit_watts: float = 80000.0
@export var max_dc_power_w: float = 80000.0
@export var motor_peak_torque: float = 25.0
@export var motor_max_rpm: float = 22000.0
@export var base_speed_rpm: float = 6500.0
@export var motor_taper_ratio: float = 0.85
@export var gear_ratio: float = 12.5
@export var drivetrain_efficiency: float = 0.90
@export var wheel_assembly_inertia: float = 0.32
@export var motor_rotor_inertia: float = 0.0012
@export var front_drive_torque_ratio: float = 0.35
@export var max_steer_angle: float = 0.38
@export var steering_ratio: float = 1.0
@export var fs_steer_wheel_max_deg: float = 115.0
@export var ffb_damping_nms_rad: float = 0.35
@export var ffb_max_wheel_torque_nm: float = 8.0
@export var ffb_soft_lock_stiffness_nm_rad: float = 28.0
@export var tv_rate_hz: float = 200.0
@export var tv_allow_regen: bool = false
@export var regen_peak_torque: float = 0.0
@export var regen_enable: int = 0
@export var regen_max_trq: float = 0.0
@export var regen_thr_speed: float = 5.0
@export var regen_thr_rpm: float = 100.0
@export var regen_trq_slope: float = 1.0
@export var regen_mode: int = 0
@export var regen_max_positive_trq_thr: float = 0.05
@export var lift_off_regen_nm: float = 0.0
@export var battery_internal_resistance_ohm: float = 0.14
@export var battery_initial_soc: float = 0.95
@export var battery_nominal_voltage_v: float = 540.0
@export var battery_ocv_full_v: float = 600.0
@export var battery_capacity_kwh: float = 7.2
@export var inverter_tau_s: float = 0.008

# brakes
@export var brake_mu: float = 1.30
@export var brake_bias_front: float = 0.58
@export var regen_blending_enabled: bool = true
@export var p_max_charge_w: float = 25000.0
@export var regen_min_speed_mps: float = 1.5
@export var disc_ambient_temp_c: float = 40.0
@export var disc_max_temp_c: float = 650.0
@export var disc_thermal_mass_j_k: float = 1350.0
@export var disc_conv_base_w_k: float = 12.0
@export var disc_conv_speed_coeff: float = 1.8

# tires
@export var wheel_radius: float = 0.2032
@export var wheel_inertia: float = 0.35
@export var tyre_mu_x: float = 1.6
@export var tyre_mu_y: float = 1.5
@export var slip_stiffness: float = 80000.0
@export var cornering_stiffness: float = 45000.0
@export var rolling_resistance_coeff: float = 0.015
@export var slip_regularization_speed: float = 0.5
@export var tire_vertical_stiffness: float = 115000.0
@export var track_temp_c: float = 28.0
@export var tire_initial_temp_c: float = 55.0
@export var tire_compound: String = "R20"
@export var tire_pressure_front_bar: float = 0.72
@export var tire_pressure_rear_bar: float = 0.72
## canonical mf 6.2 friction scaling factors (calspan ttc belt -> real asphalt track grip)
@export var lambda_mux: float = 0.95
@export var lambda_muy: float = 0.90
@export var lambda_kya: float = 1.0
@export var lambda_camber: float = 1.0
@export var tire_relaxation_sigma_y: float = -1.0
## legacy fallback alias for surface_grip_scale when lambda_mu* keys are absent
@export var surface_grip_scale: float = 0.78

# aerodynamics
@export var air_density: float = 1.225
@export var cd_a: float = 1.5
@export var cl_a: float = 6.0
@export var aero_balance_front: float = 0.48
@export var cp_height: float = 0.144
@export var static_ride_height_front: float = 0.035
@export var static_ride_height_rear: float = 0.035
@export var aero_stall_ride_height_front: float = 0.018
var _has_explicit_axle_cl: bool = false
var _cl_a_front_explicit: float = 0.0
var _cl_a_rear_explicit: float = 0.0

func _init() -> void:
	load_configs()
	update_derived_geometry()

func _has_vehicle_files(dir_path: String) -> bool:
	if dir_path == "":
		return false
	var d := dir_path.rstrip("/") + "/"
	return FileAccess.file_exists(d + "vehicle.json") or FileAccess.file_exists(d + "chassis.json")


static func _vehicles_root_candidates() -> Array[String]:
	return [
		"res://Scenes/Vehicle/Config/vehicles/",
		"src/mirena_sim/Scenes/Vehicle/Config/vehicles/",
		"Scenes/Vehicle/Config/vehicles/",
		"Config/vehicles/",
	]


## lists all available vehicle profiles under config/vehicles/<vehicle_id>/
static func list_available_vehicles() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var seen: Dictionary = {}
	for root in _vehicles_root_candidates():
		if not DirAccess.dir_exists_absolute(root):
			continue
		var d := DirAccess.open(root)
		if d == null:
			continue
		d.list_dir_begin()
		var sub := d.get_next()
		while sub != "":
			if d.current_is_dir() and not sub.begins_with(".") and not seen.has(sub):
				var vdir := root + sub + "/"
				var meta_path := vdir + "vehicle.json"
				if not FileAccess.file_exists(meta_path):
					meta_path = vdir + "chassis.json"
				if FileAccess.file_exists(meta_path):
					seen[sub] = true
					var disp := sub
					var desc := ""
					var f := FileAccess.open(meta_path, FileAccess.READ)
					if f != null:
						var parsed = JSON.parse_string(f.get_as_text())
						f.close()
						if parsed is Dictionary:
							disp = String(parsed.get("display_name", sub))
							desc = String(parsed.get("description", ""))
					out.append({
						"vehicle_id": sub,
						"display_name": disp,
						"description": desc,
						"dir": vdir,
					})
			sub = d.get_next()
		d.list_dir_end()
		if not out.is_empty():
			break
	out.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return String(a["vehicle_id"]) < String(b["vehicle_id"])
	)
	return out


static func vehicle_id_for_drive_mode(mode: DriveMode) -> String:
	if mode == DriveMode.REAR_WD:
		return "ter26_rwd"
	return "ter27_4wd"


func load_for_drive_mode(mode: DriveMode) -> bool:
	var ok := load_vehicle_configs(vehicle_id_for_drive_mode(mode))
	drive_mode = mode
	return ok


func load_vehicle_configs(target_vehicle_id: String) -> bool:
	var clean_id := target_vehicle_id.strip_edges()
	if clean_id == "":
		return false
	for root in _vehicles_root_candidates():
		var vdir := root + clean_id + "/"
		if _has_vehicle_files(vdir):
			vehicle_id = clean_id
			load_configs(vdir)
			return true
	return false


## loads configuration values from modular json files
func load_configs(custom_config_dir: String = "") -> void:
	var base_dirs = [
		custom_config_dir,
		"res://Scenes/Vehicle/Config/vehicles/ter27_4wd/",
		"src/mirena_sim/Scenes/Vehicle/Config/vehicles/ter27_4wd/",
		"Scenes/Vehicle/Config/vehicles/ter27_4wd/",
		"Config/vehicles/ter27_4wd/"
	]
	
	var found_dir: String = ""
	for d in base_dirs:
		var ds: String = str(d)
		if _has_vehicle_files(ds):
			found_dir = ds.rstrip("/") + "/"
			break
	
	if found_dir == "":
		update_derived_geometry()
		return
	
	active_config_dir = found_dir
	var chassis_file: String = found_dir + "vehicle.json"
	if not FileAccess.file_exists(chassis_file):
		chassis_file = found_dir + "chassis.json"
	var drivetrain_file: String = found_dir + "powertrain.json"
	if not FileAccess.file_exists(drivetrain_file):
		drivetrain_file = found_dir + "drivetrain.json"
	_load_chassis_json(chassis_file)
	_load_suspension_json(found_dir + "suspension.json")
	_load_tires_json(found_dir + "tires.json")
	_load_drivetrain_json(drivetrain_file)
	_load_brakes_json(found_dir + "brakes.json")
	_load_aero_json(found_dir + "aero.json")
	
	update_derived_geometry()


## devuelve dir de tv_maps for the active vehicle (falling back to shared config/vehicles/ter27_4wd/tv_maps/)
func get_active_tv_maps_dir() -> String:
	if active_config_dir != "" and DirAccess.dir_exists_absolute(active_config_dir + "tv_maps"):
		return active_config_dir + "tv_maps/"
	if vehicle_id != "":
		for root in _vehicles_root_candidates():
			var p := root + vehicle_id + "/tv_maps/"
			if DirAccess.dir_exists_absolute(p):
				return p
	for fallback in [
		"res://Scenes/Vehicle/Config/legacy_v3_tv_maps/",
		"src/mirena_sim/Scenes/Vehicle/Config/legacy_v3_tv_maps/",
		"Scenes/Vehicle/Config/legacy_v3_tv_maps/",
	]:
		if DirAccess.dir_exists_absolute(fallback):
			return fallback
	return "res://Scenes/Vehicle/Config/legacy_v3_tv_maps/"

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
	if d.has("vehicle_id"):
		vehicle_id = String(d.get("vehicle_id", vehicle_id))
	if d.has("display_name"):
		display_name = String(d.get("display_name", display_name))
	mass = float(d.get("mass", mass))
	sprung_mass = float(d.get("sprung_mass", sprung_mass))
	if d.has("unsprung_mass"):
		unsprung_mass = maxf(0.0, float(d.get("unsprung_mass", unsprung_mass)))
		unsprung_mass_total = 4.0 * unsprung_mass
	elif d.has("unsprung_mass_total"):
		unsprung_mass_total = maxf(0.0, float(d.get("unsprung_mass_total", unsprung_mass_total)))
		unsprung_mass = unsprung_mass_total * 0.25
	gravity = float(d.get("gravity", gravity))
	wheelbase = float(d.get("wheelbase", wheelbase))
	if d.has("track_width_front"):
		track_width_f = float(d.get("track_width_front", track_width_f))
	elif d.has("track_front"):
		track_width_f = float(d.get("track_front", track_width_f))
	if d.has("track_width_rear"):
		track_width_r = float(d.get("track_width_rear", track_width_r))
	elif d.has("track_rear"):
		track_width_r = float(d.get("track_rear", track_width_r))
	h_cog = float(d.get("cg_height", h_cog))
	h_sprung_cog = float(d.get("sprung_cg_height", h_sprung_cog))
	roll_center_height_front = float(d.get("roll_center_height_front", roll_center_height_front))
	roll_center_height_rear = float(d.get("roll_center_height_rear", roll_center_height_rear))
	if d.has("front_weight_distribution"):
		front_weight_distribution = clampf(float(d.get("front_weight_distribution", front_weight_distribution)), 0.0, 1.0)
		weight_dist_r = 1.0 - front_weight_distribution
	else:
		weight_dist_r = clampf(float(d.get("weight_distribution_rear", weight_dist_r)), 0.0, 1.0)
		front_weight_distribution = 1.0 - weight_dist_r
	ackermann_factor = clampf(float(d.get("ackermann_factor", ackermann_factor)), -1.0, 1.0)
	if d.has("max_steer_angle"):
		max_steer_angle = float(d.get("max_steer_angle", max_steer_angle))
	if d.has("steering_ratio"):
		steering_ratio = maxf(0.1, float(d.get("steering_ratio", steering_ratio)))
	if d.has("fs_steer_wheel_max_deg"):
		fs_steer_wheel_max_deg = maxf(10.0, float(d.get("fs_steer_wheel_max_deg", fs_steer_wheel_max_deg)))
	if d.has("ffb_damping_nms_rad"):
		ffb_damping_nms_rad = maxf(0.0, float(d.get("ffb_damping_nms_rad", ffb_damping_nms_rad)))
	if d.has("ffb_max_wheel_torque_nm"):
		ffb_max_wheel_torque_nm = maxf(0.5, float(d.get("ffb_max_wheel_torque_nm", ffb_max_wheel_torque_nm)))
	if d.has("ffb_soft_lock_stiffness_nm_rad"):
		ffb_soft_lock_stiffness_nm_rad = maxf(0.0, float(d.get("ffb_soft_lock_stiffness_nm_rad", ffb_soft_lock_stiffness_nm_rad)))
	
	var inertia = d.get("inertia", {})
	if inertia is Dictionary:
		ixx = float(inertia.get("ixx", ixx))
		iyy = float(inertia.get("iyy", iyy))
		izz = float(inertia.get("izz", izz))
		ixz = float(inertia.get("ixz", ixz))

func _load_suspension_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	spring_stiffness_front = float(d.get("spring_stiffness_front", spring_stiffness_front))
	spring_stiffness_rear = float(d.get("spring_stiffness_rear", spring_stiffness_rear))
	motion_ratio_front = float(d.get("motion_ratio_front", motion_ratio_front))
	motion_ratio_rear = float(d.get("motion_ratio_rear", motion_ratio_rear))
	damping_compression_front = float(d.get("damping_compression_front", damping_compression_front))
	damping_relaxation_front = float(d.get("damping_relaxation_front", damping_relaxation_front))
	damping_compression_rear = float(d.get("damping_compression_rear", damping_compression_rear))
	damping_relaxation_rear = float(d.get("damping_relaxation_rear", damping_relaxation_rear))
	if d.has("damper_knee_velocity") and d.has("damper_comp_low") and d.has("damper_reb_low") and d.has("damper_comp_high") and d.has("damper_reb_high"):
		use_digressive_damper = true
		damper_knee_velocity = float(d.get("damper_knee_velocity", damper_knee_velocity))
		damper_comp_low = float(d.get("damper_comp_low", damper_comp_low))
		damper_reb_low = float(d.get("damper_reb_low", damper_reb_low))
		damper_comp_high = float(d.get("damper_comp_high", damper_comp_high))
		damper_reb_high = float(d.get("damper_reb_high", damper_reb_high))
	else:
		use_digressive_damper = false
	arb_stiffness_front = float(d.get("arb_stiffness_front", arb_stiffness_front))
	arb_stiffness_rear = float(d.get("arb_stiffness_rear", arb_stiffness_rear))
	anti_dive_ratio_front = float(d.get("anti_dive_ratio_front", anti_dive_ratio_front))
	anti_squat_ratio_rear = float(d.get("anti_squat_ratio_rear", anti_squat_ratio_rear))
	jacking_coeff_front = float(d.get("jacking_coeff_front", jacking_coeff_front))
	jacking_coeff_rear = float(d.get("jacking_coeff_rear", jacking_coeff_rear))
	mechanical_trail_front = float(d.get("mechanical_trail_front", mechanical_trail_front))
	scrub_radius_front = float(d.get("scrub_radius_front", scrub_radius_front))
	if d.has("steering_compliance_stiffness_nm_rad"):
		steering_compliance_stiffness_nm_rad = maxf(1.0, float(d.get("steering_compliance_stiffness_nm_rad", steering_compliance_stiffness_nm_rad)))
	elif d.has("steering_compliance_coeff_rad_nm"):
		steering_compliance_coeff_rad_nm = maxf(0.0, float(d.get("steering_compliance_coeff_rad_nm", steering_compliance_coeff_rad_nm)))
	bumpstop_start_front = float(d.get("bumpstop_start_front", bumpstop_start_front))
	bumpstop_start_rear = float(d.get("bumpstop_start_rear", bumpstop_start_rear))
	bumpstop_stiffness_front = float(d.get("bumpstop_stiffness_front", bumpstop_stiffness_front))
	bumpstop_stiffness_rear = float(d.get("bumpstop_stiffness_rear", bumpstop_stiffness_rear))
	bumpstop_quadratic_front = float(d.get("bumpstop_quadratic_front", bumpstop_quadratic_front))
	bumpstop_quadratic_rear = float(d.get("bumpstop_quadratic_rear", bumpstop_quadratic_rear))
	ray_rest_length = float(d.get("ray_rest_length", ray_rest_length))
	ray_cast_length = float(d.get("ray_cast_length", ray_cast_length))
	caster_angle_deg = float(d.get("caster_angle_deg", caster_angle_deg))
	kpi_angle_deg = float(d.get("kpi_angle_deg", kpi_angle_deg))
	var has_static_f: bool = d.has("static_camber_front_deg")
	var has_static_r: bool = d.has("static_camber_rear_deg")
	static_camber_front_deg = float(d.get("static_camber_front_deg", static_camber_front_deg))
	static_camber_rear_deg = float(d.get("static_camber_rear_deg", static_camber_rear_deg))
	toe_front_deg = float(d.get("toe_front_deg", 0.0)) if d.has("toe_front_deg") else 0.0
	toe_rear_deg = float(d.get("toe_rear_deg", 0.0)) if d.has("toe_rear_deg") else 0.0
	camber_table_front = _load_camber_table(d, "camber_table_front_deg", "camber_gain_front_deg_per_10mm", static_camber_front_deg, has_static_f)
	camber_table_rear = _load_camber_table(d, "camber_table_rear_deg", "camber_gain_rear_deg_per_10mm", static_camber_rear_deg, has_static_r)
	toe_table_front = _parse_kinematic_table_deg(d.get("toe_table_front_deg", []))
	toe_table_rear = _parse_kinematic_table_deg(d.get("toe_table_rear_deg", []))

	use_3d_kinematics = false
	double_wishbone_kinematics = {}
	if d.has("double_wishbone_kinematics") and d["double_wishbone_kinematics"] is Dictionary:
		double_wishbone_kinematics = (d["double_wishbone_kinematics"] as Dictionary).duplicate(true)
		use_3d_kinematics = bool(double_wishbone_kinematics.get("use_3d_kinematics", false))
	if d.has("use_3d_kinematics"):
		use_3d_kinematics = bool(d.get("use_3d_kinematics", use_3d_kinematics))

static func _load_camber_table(d: Dictionary, table_key: String, gain_key: String, static_deg: float, has_static: bool) -> Array:
	var table: Array = _parse_kinematic_table_deg(d.get(table_key, []))
	if table.size() > 0:
		if has_static:
			var base_zero_rad: float = _lookup_table_rad(table, 0.0)
			var offset_rad: float = deg_to_rad(static_deg) - base_zero_rad
			if absf(offset_rad) > 1e-7:
				for pt in table:
					pt[1] = float(pt[1]) + offset_rad
		return table
	if d.has(gain_key):
		var gain_per_10mm: float = float(d.get(gain_key, 0.0))
		for dz_mm in [-30, -20, -10, 0, 10, 20, 30]:
			var dz_m: float = float(dz_mm) * 0.001
			var ang_deg: float = static_deg + gain_per_10mm * (float(dz_mm) / 10.0)
			table.append([dz_m, deg_to_rad(ang_deg)])
		return table
	if has_static and not d.has(table_key):
		return [[0.0, deg_to_rad(static_deg)]]
	return table

static func _lookup_table_rad(table: Array, delta_z: float) -> float:
	var n: int = table.size()
	if n == 0:
		return 0.0
	if n == 1 or delta_z <= float(table[0][0]):
		return float(table[0][1])
	if delta_z >= float(table[n - 1][0]):
		return float(table[n - 1][1])
	for i in range(n - 1):
		var xa: float = float(table[i][0])
		var xb: float = float(table[i + 1][0])
		if delta_z >= xa and delta_z <= xb:
			var dx: float = xb - xa
			if absf(dx) < 1e-9:
				return float(table[i][1])
			return lerpf(float(table[i][1]), float(table[i + 1][1]), (delta_z - xa) / dx)
	return float(table[n - 1][1])

static func _parse_kinematic_table_deg(raw_table: Variant) -> Array:
	var out: Array = []
	if not (raw_table is Array):
		return out
	for pt in raw_table:
		if pt is Array and pt.size() >= 2:
			out.append([float(pt[0]), deg_to_rad(float(pt[1]))])
	out.sort_custom(func(a, b): return float(a[0]) < float(b[0]))
	return out

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
	tire_vertical_stiffness = maxf(1.0, float(d.get("tire_vertical_stiffness", tire_vertical_stiffness)))
	track_temp_c = float(d.get("track_temp_c", track_temp_c))
	tire_initial_temp_c = float(d.get("tire_initial_temp_c", tire_initial_temp_c))
	tire_compound = String(d.get("tire_compound", tire_compound))
	tire_pressure_front_bar = clampf(float(d.get("tire_pressure_front_bar", tire_pressure_front_bar)), 0.3, 2.5)
	tire_pressure_rear_bar = clampf(float(d.get("tire_pressure_rear_bar", tire_pressure_rear_bar)), 0.3, 2.5)
	if d.has("surface_grip_scale"):
		surface_grip_scale = clampf(float(d.get("surface_grip_scale", surface_grip_scale)), 0.1, 1.5)
	lambda_mux = clampf(float(d.get("lambda_mux", surface_grip_scale)), 0.1, 1.5)
	lambda_muy = clampf(float(d.get("lambda_muy", surface_grip_scale)), 0.1, 1.5)
	lambda_kya = clampf(float(d.get("lambda_kya", 1.0)), 0.1, 3.0)
	lambda_camber = clampf(float(d.get("lambda_camber", 1.0)), 0.1, 3.0)
	tire_relaxation_sigma_y = float(d.get("tire_relaxation_sigma_y", -1.0))

func _load_drivetrain_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	power_limit_watts = float(d.get("power_limit_watts", power_limit_watts))
	max_dc_power_w = float(d.get("max_dc_power_w", power_limit_watts))
	power_limit_watts = max_dc_power_w
	motor_peak_torque = float(d.get("motor_peak_torque", motor_peak_torque))
	motor_max_rpm = float(d.get("motor_max_rpm", motor_max_rpm))
	base_speed_rpm = float(d.get("base_speed_rpm", base_speed_rpm))
	motor_taper_ratio = float(d.get("motor_taper_ratio", motor_taper_ratio))
	gear_ratio = float(d.get("gear_ratio", gear_ratio))
	drivetrain_efficiency = float(d.get("drivetrain_efficiency", drivetrain_efficiency))
	wheel_assembly_inertia = float(d.get("wheel_assembly_inertia", wheel_assembly_inertia))
	motor_rotor_inertia = float(d.get("motor_rotor_inertia", motor_rotor_inertia))
	if d.has("front_torque_split"):
		front_drive_torque_ratio = clampf(float(d.get("front_torque_split", front_drive_torque_ratio)), 0.0, 1.0)
	else:
		front_drive_torque_ratio = clampf(float(d.get("front_drive_torque_ratio", front_drive_torque_ratio)), 0.0, 1.0)
	max_steer_angle = float(d.get("max_steer_angle", max_steer_angle))
	tv_rate_hz = float(d.get("tv_rate_hz", tv_rate_hz))
	tv_allow_regen = bool(d.get("tv_allow_regen", tv_allow_regen))
	regen_peak_torque = float(d.get("regen_peak_torque", regen_peak_torque))
	regen_enable = int(d.get("regen_enable", regen_enable))
	regen_max_trq = float(d.get("regen_max_trq", regen_max_trq))
	regen_thr_speed = float(d.get("regen_thr_speed", regen_thr_speed))
	regen_thr_rpm = float(d.get("regen_thr_rpm", regen_thr_rpm))
	regen_trq_slope = float(d.get("regen_trq_slope", regen_trq_slope))
	regen_mode = int(d.get("regen_mode", regen_mode))
	regen_max_positive_trq_thr = float(d.get("regen_max_positive_trq_thr", regen_max_positive_trq_thr))
	lift_off_regen_nm = maxf(0.0, float(d.get("lift_off_regen_nm", lift_off_regen_nm)))
	battery_internal_resistance_ohm = maxf(0.0, float(d.get("battery_internal_resistance_ohm", battery_internal_resistance_ohm)))
	battery_initial_soc = clampf(float(d.get("battery_initial_soc", battery_initial_soc)), 0.0, 1.0)
	battery_nominal_voltage_v = maxf(1.0, float(d.get("battery_nominal_voltage_v", battery_nominal_voltage_v)))
	battery_ocv_full_v = maxf(1.0, float(d.get("battery_ocv_full_v", battery_ocv_full_v)))
	battery_capacity_kwh = maxf(0.01, float(d.get("battery_capacity_kwh", battery_capacity_kwh)))
	inverter_tau_s = maxf(0.0, float(d.get("inverter_tau_s", inverter_tau_s)))
	
	var dm_str := ""
	if d.has("drive_type"):
		var dt := String(d.get("drive_type", "")).strip_edges().to_upper()
		if dt == "RWD" or dt == "REAR_WD" or dt == "2WD":
			dm_str = "REAR_WD"
		elif dt == "FWD" or dt == "FRONT_WD":
			dm_str = "FRONT_WD"
		elif dt == "4WD" or dt == "AWD" or dt == "FOUR_WD":
			dm_str = "FOUR_WD"
	if dm_str == "":
		dm_str = str(d.get("drive_mode", "FOUR_WD"))
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
	brake_bias_front = clampf(float(d.get("brake_bias_front", brake_bias_front)), 0.0, 1.0)
	regen_blending_enabled = bool(d.get("regen_blending_enabled", regen_blending_enabled))
	p_max_charge_w = maxf(0.0, float(d.get("p_max_charge_w", p_max_charge_w)))
	regen_min_speed_mps = maxf(0.0, float(d.get("regen_min_speed_mps", regen_min_speed_mps)))
	disc_ambient_temp_c = float(d.get("disc_ambient_temp_c", disc_ambient_temp_c))
	disc_max_temp_c = maxf(disc_ambient_temp_c + 10.0, float(d.get("disc_max_temp_c", disc_max_temp_c)))
	disc_thermal_mass_j_k = maxf(1.0, float(d.get("disc_thermal_mass_j_k", disc_thermal_mass_j_k)))
	disc_conv_base_w_k = maxf(0.0, float(d.get("disc_conv_base_w_k", disc_conv_base_w_k)))
	disc_conv_speed_coeff = maxf(0.0, float(d.get("disc_conv_speed_coeff", disc_conv_speed_coeff)))

func _load_aero_json(file_path: String) -> void:
	var d = _read_json_file(file_path)
	if d.is_empty():
		return
	air_density = float(d.get("air_density", air_density))
	cd_a = float(d.get("cd_a", cd_a))
	cl_a = float(d.get("cl_a", cl_a))
	aero_balance_front = clampf(float(d.get("aero_balance_front", aero_balance_front)), 0.0, 1.0)
	cp_height = float(d.get("cp_height", cp_height))
	static_ride_height_front = maxf(0.001, float(d.get("static_ride_height_front", static_ride_height_front)))
	static_ride_height_rear = maxf(0.001, float(d.get("static_ride_height_rear", static_ride_height_rear)))
	if d.has("stall_ride_height_front"):
		aero_stall_ride_height_front = maxf(0.001, float(d.get("stall_ride_height_front", aero_stall_ride_height_front)))
	elif d.has("aero_stall_ride_height_front"):
		aero_stall_ride_height_front = maxf(0.001, float(d.get("aero_stall_ride_height_front", aero_stall_ride_height_front)))
	if d.has("cl_a_front") and d.has("cl_a_rear"):
		_has_explicit_axle_cl = true
		_cl_a_front_explicit = float(d.get("cl_a_front", 0.0))
		_cl_a_rear_explicit = float(d.get("cl_a_rear", 0.0))
		cl_a = _cl_a_front_explicit + _cl_a_rear_explicit
		if cl_a > 1e-6:
			aero_balance_front = _cl_a_front_explicit / cl_a
	else:
		_has_explicit_axle_cl = false

## front-axle downforce area coefficient c_l,f * a
func get_cl_a_front() -> float:
	if _has_explicit_axle_cl:
		return _cl_a_front_explicit
	return cl_a * clampf(aero_balance_front, 0.0, 1.0)

## rear-axle downforce area coefficient c_l,r * a
func get_cl_a_rear() -> float:
	if _has_explicit_axle_cl:
		return _cl_a_rear_explicit
	return cl_a * (1.0 - clampf(aero_balance_front, 0.0, 1.0))

## returns total vehicle yaw inertia (kg*m^2): sprung izz + unsprung steiner corners + 4 * wheel_assembly_inertia
func get_total_izz() -> float:
	var m_u: float = maxf(unsprung_mass, 0.0)
	var half_wf: float = 0.5 * track_width_f
	var half_wr: float = 0.5 * track_width_r
	var steiner_f: float = 2.0 * m_u * (a_cog * a_cog + half_wf * half_wf)
	var steiner_r: float = 2.0 * m_u * (b_cog * b_cog + half_wr * half_wr)
	return maxf(izz + steiner_f + steiner_r + 4.0 * maxf(wheel_assembly_inertia, 0.0), 1e-3)

## returns effective reflected wheel rotational inertia: i_eff = i_wheel + g^2 * i_rotor (kg*m^2)
func get_effective_wheel_inertia() -> float:
	return maxf(wheel_assembly_inertia + gear_ratio * gear_ratio * motor_rotor_inertia, 1e-4)

## recalculates longitudinal cog distances, roll-center geometry, and effective wheel inertia
func update_derived_geometry() -> void:
	front_weight_distribution = 1.0 - weight_dist_r
	a_cog = wheelbase * weight_dist_r
	b_cog = wheelbase * (1.0 - weight_dist_r)
	roll_center_height_cg = (1.0 - weight_dist_r) * roll_center_height_front + weight_dist_r * roll_center_height_rear
	h_roll_arm = maxf(0.0, h_sprung_cog - roll_center_height_cg)
	wheel_inertia = get_effective_wheel_inertia()

## convert vector from godot (+x right, +y up, -z forward) to ros (+x forward, +y left, +z up)
static func godot_to_ros_vec(v: Vector3) -> Vector3:
	return Vector3(-v.z, -v.x, v.y)

## convert vector from ros (+x forward, +y left, +z up) to godot (+x right, +y up, -z forward)
static func ros_to_godot_vec(v: Vector3) -> Vector3:
	return Vector3(-v.y, v.z, -v.x)

## returns static normal load (n) on front and rear axles at rest on level ground
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

static func _sync_camber_table_static(table: Array, static_deg: float) -> Array:
	var target_rad: float = deg_to_rad(static_deg)
	if table.is_empty():
		return [[0.0, target_rad]]
	var base_zero_rad: float = _lookup_table_rad(table, 0.0)
	var offset_rad: float = target_rad - base_zero_rad
	if absf(offset_rad) > 1e-7:
		for pt in table:
			pt[1] = float(pt[1]) + offset_rad
	return table

static func _sanitize_preset_stem(raw_name: String) -> String:
	var s := raw_name.strip_edges().to_lower()
	if s.ends_with(".json"):
		s = s.substr(0, s.length() - 5)
	var out := ""
	for i in range(s.length()):
		var ch := s[i]
		if (ch >= "a" and ch <= "z") or (ch >= "0" and ch <= "9"):
			out += ch
		elif ch == " " or ch == "-" or ch == "_":
			if not out.ends_with("_"):
				out += "_"
	return out.trim_prefix("_").trim_suffix("_")

## returns the directory containing built-in mechanical setup preset json files
static func get_setup_presets_dir() -> String:
	var candidates: Array[String] = [
		"res://Scenes/Vehicle/Config/vehicles/ter27_4wd/setup_presets/",
		"src/mirena_sim/Scenes/Vehicle/Config/vehicles/ter27_4wd/setup_presets/",
		"Scenes/Vehicle/Config/vehicles/ter27_4wd/setup_presets/",
		"Config/vehicles/ter27_4wd/setup_presets/"
	]
	for d in candidates:
		var abs_d: String = ProjectSettings.globalize_path(d) if d.begins_with("res://") else d
		if DirAccess.dir_exists_absolute(d) or DirAccess.dir_exists_absolute(abs_d):
			return d
	return "res://Scenes/Vehicle/Config/vehicles/ter27_4wd/setup_presets/"

## scans setup_presets/ and user://setup_presets/ for .json presets sorted alphabetically
static func list_setup_presets() -> Array[Dictionary]:
	var entries: Array[Dictionary] = []
	var seen_paths: Dictionary = {}
	var dirs: Array[String] = [get_setup_presets_dir(), "user://setup_presets/"]
	for dir_path in dirs:
		var norm_dir: String = dir_path.rstrip("/") + "/"
		var d := DirAccess.open(norm_dir)
		if d == null:
			continue
		d.list_dir_begin()
		var fname: String = d.get_next()
		while fname != "":
			if not d.current_is_dir() and fname.to_lower().ends_with(".json"):
				var full_path: String = norm_dir + fname
				if not seen_paths.has(full_path):
					seen_paths[full_path] = true
					var meta: Dictionary = load_setup_preset(full_path)
					var base_name: String = fname.get_basename()
					var params: Dictionary = meta.get("params", {}) if (meta.get("params") is Dictionary) else {}
					entries.append({
						"name": String(meta.get("name", base_name)),
						"description": String(meta.get("description", "")),
						"path": full_path,
						"filename": base_name,
						"params": params
					})
			fname = d.get_next()
		d.list_dir_end()
	entries.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return String(a["path"]).get_file() < String(b["path"]).get_file()
	)
	return entries

## returns the 10 key mechanical setup parameters plus regen/ride-height options
func get_setup_params() -> Dictionary:
	return {
		"spring_stiffness_front": spring_stiffness_front,
		"spring_stiffness_rear": spring_stiffness_rear,
		"arb_stiffness_front": arb_stiffness_front,
		"arb_stiffness_rear": arb_stiffness_rear,
		"static_camber_front_deg": static_camber_front_deg,
		"static_camber_rear_deg": static_camber_rear_deg,
		"toe_front_deg": toe_front_deg,
		"toe_rear_deg": toe_rear_deg,
		"brake_bias_front": brake_bias_front,
		"aero_balance_front": aero_balance_front,
		"regen_blending_enabled": regen_blending_enabled,
		"static_ride_height_front": static_ride_height_front,
		"static_ride_height_rear": static_ride_height_rear
	}

## applies mechanical setup parameters in-place and updates derived geometry
func apply_setup_params(params: Dictionary) -> void:
	var p: Dictionary = params
	if params.has("params") and params["params"] is Dictionary:
		p = params["params"]
	if p.has("spring_stiffness_front"):
		spring_stiffness_front = maxf(1000.0, float(p["spring_stiffness_front"]))
	if p.has("spring_stiffness_rear"):
		spring_stiffness_rear = maxf(1000.0, float(p["spring_stiffness_rear"]))
	if p.has("arb_stiffness_front"):
		arb_stiffness_front = maxf(0.0, float(p["arb_stiffness_front"]))
	if p.has("arb_stiffness_rear"):
		arb_stiffness_rear = maxf(0.0, float(p["arb_stiffness_rear"]))
	if p.has("static_camber_front_deg"):
		static_camber_front_deg = clampf(float(p["static_camber_front_deg"]), -10.0, 5.0)
		camber_table_front = _sync_camber_table_static(camber_table_front, static_camber_front_deg)
	if p.has("static_camber_rear_deg"):
		static_camber_rear_deg = clampf(float(p["static_camber_rear_deg"]), -10.0, 5.0)
		camber_table_rear = _sync_camber_table_static(camber_table_rear, static_camber_rear_deg)
	if p.has("toe_front_deg"):
		toe_front_deg = clampf(float(p["toe_front_deg"]), -5.0, 5.0)
	if p.has("toe_rear_deg"):
		toe_rear_deg = clampf(float(p["toe_rear_deg"]), -5.0, 5.0)
	if p.has("brake_bias_front"):
		brake_bias_front = clampf(float(p["brake_bias_front"]), 0.0, 1.0)
	if p.has("aero_balance_front"):
		aero_balance_front = clampf(float(p["aero_balance_front"]), 0.0, 1.0)
		_has_explicit_axle_cl = false
	if p.has("regen_blending_enabled"):
		regen_blending_enabled = bool(p["regen_blending_enabled"])
	if p.has("static_ride_height_front"):
		static_ride_height_front = maxf(0.001, float(p["static_ride_height_front"]))
	if p.has("static_ride_height_rear"):
		static_ride_height_rear = maxf(0.001, float(p["static_ride_height_rear"]))
	if p.has("use_3d_kinematics"):
		use_3d_kinematics = bool(p["use_3d_kinematics"])
	update_derived_geometry()

## loads and parses a setup preset json file
static func load_setup_preset(path: String) -> Dictionary:
	if path == "" or not FileAccess.file_exists(path):
		return {}
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if parsed is Dictionary:
		return parsed
	return {}

## saves current mechanical setup parameters to setup_presets/ (or user://setup_presets/ fallback)
func save_setup_preset(preset_name: String, description: String = "", file_stem: String = "") -> String:
	var stem: String = _sanitize_preset_stem(file_stem if file_stem.strip_edges() != "" else preset_name)
	if stem == "":
		stem = "custom_setup_%d" % int(Time.get_unix_time_from_system())
	var display_name: String = preset_name.strip_edges()
	if display_name == "":
		display_name = stem
	var payload: Dictionary = {
		"name": display_name,
		"description": description,
		"author": "TeR27 Vehicle Dynamics",
		"params": get_setup_params()
	}
	var json_text: String = JSON.stringify(payload, "\t") + "\n"
	var res_dir: String = get_setup_presets_dir().rstrip("/") + "/"
	var abs_res_dir: String = ProjectSettings.globalize_path(res_dir) if res_dir.begins_with("res://") else res_dir
	DirAccess.make_dir_recursive_absolute(abs_res_dir)
	var target_path: String = res_dir + stem + ".json"
	var f := FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		DirAccess.make_dir_recursive_absolute("user://setup_presets")
		target_path = "user://setup_presets/" + stem + ".json"
		f = FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(json_text)
	f.close()
	return target_path
