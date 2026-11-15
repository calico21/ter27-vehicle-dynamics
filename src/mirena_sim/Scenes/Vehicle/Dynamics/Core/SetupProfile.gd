class_name SetupProfile
extends Resource

## Modular per-vehicle set-up profile supporting mechanical, brake, tire pressure,
## and electronics (TV map, power limit, control law) parameters.

@export var name: String = "default"
@export var vehicle_id: String = "ter27_4wd"
@export var description: String = ""

# --- Suspension ---
@export var spring_rate_front_n_m: float = 35000.0
@export var spring_rate_rear_n_m: float = 48000.0
@export var arb_front_nm_rad: float = 4500.0
@export var arb_rear_nm_rad: float = 5500.0
@export var camber_front_deg: float = -2.0
@export var camber_rear_deg: float = -1.5
@export var toe_front_deg: float = 0.0
@export var toe_rear_deg: float = 0.0
@export var ride_height_front_m: float = 0.035
@export var ride_height_rear_m: float = 0.035

# --- Brakes & Tires ---
@export var brake_bias: float = 0.58
@export var tire_pressure_front_bar: float = 0.72
@export var tire_pressure_rear_bar: float = 0.72

# --- Electronics & Control ---
@export var tv_map: String = ""
@export var power_limit_kw: float = 80.0
@export var control_law_mode: String = "smc"
@export var aero_balance_front: float = -1.0 # < 0 means preserve vehicle aero default
@export var regen_blending_enabled: Variant = null
@export var tv_overrides: Dictionary = {}


static func _vehicle_setups_dir_candidates(v_id: String) -> Array[String]:
	return [
		"res://Scenes/Vehicle/Dynamics/Core/Config/vehicles/%s/setups/" % v_id,
		"src/mirena_sim/Scenes/Vehicle/Dynamics/Core/Config/vehicles/%s/setups/" % v_id,
		"Scenes/Vehicle/Dynamics/Core/Config/vehicles/%s/setups/" % v_id,
		"Core/Config/vehicles/%s/setups/" % v_id,
		"user://setups/%s/" % v_id,
	]


static func _sanitize_stem(raw_name: String) -> String:
	var s: String = raw_name.strip_edges().to_lower()
	if s.ends_with(".json"):
		s = s.substr(0, s.length() - 5)
	var out := ""
	for i in range(s.length()):
		var ch := s[i]
		var code := s.unicode_at(i)
		var is_alnum := (code >= 48 and code <= 57) or (code >= 97 and code <= 122)
		if is_alnum or ch == "_" or ch == "-":
			out += ch
		elif ch == " " or ch == ".":
			out += "_"
	return out


## Lists available setups for a vehicle (merging built-in and user://setups/<vehicle_id>/).
static func list_setups_for_vehicle(v_id: String) -> Array[Dictionary]:
	var by_stem: Dictionary = {}
	for dir_path in _vehicle_setups_dir_candidates(v_id):
		if not DirAccess.dir_exists_absolute(dir_path):
			continue
		var d := DirAccess.open(dir_path)
		if d == null:
			continue
		d.list_dir_begin()
		var fname := d.get_next()
		while fname != "":
			if not d.current_is_dir() and fname.ends_with(".json"):
				var full_path: String = dir_path.rstrip("/") + "/" + fname
				var f := FileAccess.open(full_path, FileAccess.READ)
				if f != null:
					var parsed = JSON.parse_string(f.get_as_text())
					f.close()
					if parsed is Dictionary:
						var stem: String = fname.get_basename()
						by_stem[stem] = {
							"name": String(parsed.get("name", stem)),
							"stem": stem,
							"description": String(parsed.get("description", "")),
							"path": full_path,
							"data": parsed,
						}
			fname = d.get_next()
		d.list_dir_end()
	var result: Array[Dictionary] = []
	for k in by_stem.keys():
		result.append(by_stem[k])
	result.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		var sa := String(a["stem"])
		var sb := String(b["stem"])
		if sa == "default" and sb != "default":
			return true
		if sb == "default" and sa != "default":
			return false
		return sa < sb
	)
	return result


const VehicleConfigScript = preload("VehicleConfig.gd")


static func load_for_vehicle(v_id: String, setup_name: String = "default") -> Resource:
	var stem: String = _sanitize_stem(setup_name)
	if stem == "":
		stem = "default"
	var setups := list_setups_for_vehicle(v_id)
	# Prioritize user override / matching stem or name
	for entry in setups:
		if String(entry["stem"]) == stem or _sanitize_stem(String(entry["name"])) == stem:
			var prof = load("res://Scenes/Vehicle/Dynamics/Core/SetupProfile.gd").new()
			prof.from_dict(entry["data"])
			if prof.vehicle_id == "":
				prof.vehicle_id = v_id
			if prof.name == "":
				prof.name = stem
			return prof
	return null


static func from_vehicle_config(config: VehicleConfigScript, setup_name: String = "custom", tv_map_filename: String = "", control_law: String = "smc") -> Resource:
	var prof = load("res://Scenes/Vehicle/Dynamics/Core/SetupProfile.gd").new()
	prof.name = setup_name
	prof.vehicle_id = config.vehicle_id if config != null else "ter27_4wd"
	if config != null:
		prof.spring_rate_front_n_m = config.spring_stiffness_front
		prof.spring_rate_rear_n_m = config.spring_stiffness_rear
		prof.arb_front_nm_rad = config.arb_stiffness_front
		prof.arb_rear_nm_rad = config.arb_stiffness_rear
		prof.camber_front_deg = config.static_camber_front_deg
		prof.camber_rear_deg = config.static_camber_rear_deg
		prof.toe_front_deg = config.toe_front_deg
		prof.toe_rear_deg = config.toe_rear_deg
		prof.ride_height_front_m = config.static_ride_height_front
		prof.ride_height_rear_m = config.static_ride_height_rear
		prof.brake_bias = config.brake_bias_front
		prof.tire_pressure_front_bar = config.tire_pressure_front_bar
		prof.tire_pressure_rear_bar = config.tire_pressure_rear_bar
		prof.power_limit_kw = config.power_limit_watts / 1000.0
		prof.aero_balance_front = config.aero_balance_front
		prof.regen_blending_enabled = config.regen_blending_enabled
	prof.tv_map = tv_map_filename
	prof.control_law_mode = control_law
	return prof


func from_dict(d: Dictionary) -> void:
	name = String(d.get("name", name))
	vehicle_id = String(d.get("vehicle_id", vehicle_id))
	description = String(d.get("description", description))
	spring_rate_front_n_m = float(d.get("spring_rate_front_n_m", d.get("spring_stiffness_front", spring_rate_front_n_m)))
	spring_rate_rear_n_m = float(d.get("spring_rate_rear_n_m", d.get("spring_stiffness_rear", spring_rate_rear_n_m)))
	arb_front_nm_rad = float(d.get("arb_front_nm_rad", d.get("arb_stiffness_front", arb_front_nm_rad)))
	arb_rear_nm_rad = float(d.get("arb_rear_nm_rad", d.get("arb_stiffness_rear", arb_rear_nm_rad)))
	camber_front_deg = float(d.get("camber_front_deg", d.get("static_camber_front_deg", camber_front_deg)))
	camber_rear_deg = float(d.get("camber_rear_deg", d.get("static_camber_rear_deg", camber_rear_deg)))
	toe_front_deg = float(d.get("toe_front_deg", toe_front_deg))
	toe_rear_deg = float(d.get("toe_rear_deg", toe_rear_deg))
	ride_height_front_m = float(d.get("ride_height_front_m", d.get("static_ride_height_front", ride_height_front_m)))
	ride_height_rear_m = float(d.get("ride_height_rear_m", d.get("static_ride_height_rear", ride_height_rear_m)))
	brake_bias = float(d.get("brake_bias", d.get("brake_bias_front", brake_bias)))
	tire_pressure_front_bar = float(d.get("tire_pressure_front_bar", tire_pressure_front_bar))
	tire_pressure_rear_bar = float(d.get("tire_pressure_rear_bar", tire_pressure_rear_bar))
	tv_map = String(d.get("tv_map", tv_map))
	power_limit_kw = float(d.get("power_limit_kw", power_limit_kw))
	control_law_mode = String(d.get("control_law_mode", control_law_mode))
	if d.has("aero_balance_front"):
		aero_balance_front = float(d.get("aero_balance_front", aero_balance_front))
	if d.has("regen_blending_enabled"):
		regen_blending_enabled = bool(d.get("regen_blending_enabled", true))
	if d.get("tv_overrides") is Dictionary:
		tv_overrides = (d["tv_overrides"] as Dictionary).duplicate(true)


func to_dict() -> Dictionary:
	var out := {
		"name": name,
		"vehicle_id": vehicle_id,
		"description": description,
		"spring_rate_front_n_m": spring_rate_front_n_m,
		"spring_rate_rear_n_m": spring_rate_rear_n_m,
		"arb_front_nm_rad": arb_front_nm_rad,
		"arb_rear_nm_rad": arb_rear_nm_rad,
		"camber_front_deg": camber_front_deg,
		"camber_rear_deg": camber_rear_deg,
		"toe_front_deg": toe_front_deg,
		"toe_rear_deg": toe_rear_deg,
		"ride_height_front_m": ride_height_front_m,
		"ride_height_rear_m": ride_height_rear_m,
		"brake_bias": brake_bias,
		"tire_pressure_front_bar": tire_pressure_front_bar,
		"tire_pressure_rear_bar": tire_pressure_rear_bar,
		"tv_map": tv_map,
		"power_limit_kw": power_limit_kw,
		"control_law_mode": control_law_mode,
	}
	if aero_balance_front >= 0.0:
		out["aero_balance_front"] = aero_balance_front
	if regen_blending_enabled != null:
		out["regen_blending_enabled"] = bool(regen_blending_enabled)
	if not tv_overrides.is_empty():
		out["tv_overrides"] = tv_overrides.duplicate(true)
	return out


func apply_to_config(config: VehicleConfig) -> void:
	if config == null:
		return
	var mech := {
		"spring_stiffness_front": spring_rate_front_n_m,
		"spring_stiffness_rear": spring_rate_rear_n_m,
		"arb_stiffness_front": arb_front_nm_rad,
		"arb_stiffness_rear": arb_rear_nm_rad,
		"static_camber_front_deg": camber_front_deg,
		"static_camber_rear_deg": camber_rear_deg,
		"toe_front_deg": toe_front_deg,
		"toe_rear_deg": toe_rear_deg,
		"brake_bias_front": brake_bias,
		"static_ride_height_front": ride_height_front_m,
		"static_ride_height_rear": ride_height_rear_m,
	}
	if aero_balance_front >= 0.0:
		mech["aero_balance_front"] = aero_balance_front
	if regen_blending_enabled != null:
		mech["regen_blending_enabled"] = bool(regen_blending_enabled)
	config.apply_setup_params(mech)
	config.tire_pressure_front_bar = clampf(tire_pressure_front_bar, 0.3, 2.5)
	config.tire_pressure_rear_bar = clampf(tire_pressure_rear_bar, 0.3, 2.5)
	if power_limit_kw > 0.1:
		var p_w: float = power_limit_kw * 1000.0
		config.power_limit_watts = p_w
		config.max_dc_power_w = p_w
	config.active_setup_name = name


func save_for_vehicle(v_id: String, setup_name: String = "") -> String:
	var target_vid: String = v_id if v_id != "" else vehicle_id
	vehicle_id = target_vid
	if setup_name.strip_edges() != "":
		name = setup_name.strip_edges()
	var stem := _sanitize_stem(name)
	if stem == "":
		stem = "custom_%d" % int(Time.get_unix_time_from_system())
	var json_text := JSON.stringify(to_dict(), "\t") + "\n"
	var res_dir := "res://Scenes/Vehicle/Dynamics/Core/Config/vehicles/%s/setups/" % target_vid
	var abs_res_dir := ProjectSettings.globalize_path(res_dir)
	DirAccess.make_dir_recursive_absolute(abs_res_dir)
	var target_path := res_dir + stem + ".json"
	var f := FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		var user_dir := "user://setups/%s" % target_vid
		DirAccess.make_dir_recursive_absolute(user_dir)
		target_path = user_dir + "/" + stem + ".json"
		f = FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(json_text)
	f.close()
	return target_path
