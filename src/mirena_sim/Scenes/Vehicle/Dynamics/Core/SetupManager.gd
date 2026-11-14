class_name SetupManager
extends RefCounted

## Unified DiL Mechanical + Electronic Setup Manager for TeR27.
## Loads/saves JSON presets in res://.../Config/setups/ and user://setups/,
## applies changes cleanly to VehicleConfig & EcuTorqueVectoringV3, and computes
## parameter diffs against the loaded base preset.

const RES_SETUPS_DIR: String = "res://Scenes/Vehicle/Dynamics/Core/Config/setups/"
const USER_SETUPS_DIR: String = "user://setups/"

const PARAM_SPECS: Array = [
	{"key": "spring_rate_front_n_m", "label": "Muelle Del (N/mm)", "scale": 0.001, "unit": "N/mm", "tol": 100.0},
	{"key": "spring_rate_rear_n_m", "label": "Muelle Tras (N/mm)", "scale": 0.001, "unit": "N/mm", "tol": 100.0},
	{"key": "arb_stiffness_front_nm_rad", "label": "ARB Del (Nm/rad)", "scale": 1.0, "unit": "Nm/rad", "tol": 10.0},
	{"key": "arb_stiffness_rear_nm_rad", "label": "ARB Tras (Nm/rad)", "scale": 1.0, "unit": "Nm/rad", "tol": 10.0},
	{"key": "static_camber_front_deg", "label": "Caida Del (deg)", "scale": 1.0, "unit": "deg", "tol": 0.02},
	{"key": "static_camber_rear_deg", "label": "Caida Tras (deg)", "scale": 1.0, "unit": "deg", "tol": 0.02},
	{"key": "brake_bias_front", "label": "Brake Bias Del (%)", "scale": 100.0, "unit": "%", "tol": 0.002},
	{"key": "ride_height_front_m", "label": "Altura Del (mm)", "scale": 1000.0, "unit": "mm", "tol": 0.0005},
	{"key": "ride_height_rear_m", "label": "Altura Tras (mm)", "scale": 1000.0, "unit": "mm", "tol": 0.0005},
	{"key": "tv_map_filename", "label": "Mapa TV", "scale": 1.0, "unit": "", "tol": 0.0},
	{"key": "control_law_mode", "label": "Ley Control", "scale": 1.0, "unit": "", "tol": 0.1},
	{"key": "grip_cap", "label": "Grip Cap", "scale": 1.0, "unit": "", "tol": 0.005},
	{"key": "smc_k_gain", "label": "SMC K_gain", "scale": 1.0, "unit": "Nm", "tol": 5.0},
	{"key": "smc_lambda_beta", "label": "SMC lambda_beta", "scale": 1.0, "unit": "", "tol": 0.05},
	{"key": "smc_k_ffd", "label": "SMC K_ffd", "scale": 1.0, "unit": "", "tol": 1.0},
	{"key": "smc_tc_kappa_base", "label": "TC kappa_base", "scale": 1.0, "unit": "", "tol": 0.001},
	{"key": "eco_power_cap_kw", "label": "Tope Potencia (kW)", "scale": 1.0, "unit": "kW", "tol": 0.5}
]

var base_preset_name: String = "Default"
var base_preset_path: String = ""
var base_snapshot: Dictionary = {}

static func capture_from_car(config, ecu) -> Dictionary:
	var out: Dictionary = {}
	if config != null:
		out["spring_rate_front_n_m"] = float(config.spring_stiffness_front)
		out["spring_rate_rear_n_m"] = float(config.spring_stiffness_rear)
		out["arb_stiffness_front_nm_rad"] = float(config.arb_stiffness_front)
		out["arb_stiffness_rear_nm_rad"] = float(config.arb_stiffness_rear)
		out["static_camber_front_deg"] = float(config.static_camber_front_deg)
		out["static_camber_rear_deg"] = float(config.static_camber_rear_deg)
		out["brake_bias_front"] = float(config.brake_bias_front)
		out["ride_height_front_m"] = float(config.static_ride_height_front)
		out["ride_height_rear_m"] = float(config.static_ride_height_rear)
		out["max_dc_power_kw"] = float(config.max_dc_power_w) * 0.001
	if ecu != null:
		out["tv_map_filename"] = String(ecu.active_map_filename)
		out["event_mode"] = String(ecu.active_event_mode)
		out["control_law_mode"] = int(ecu.control_law_mode)
		out["grip_cap"] = float(ecu.live_grip_cap)
		out["smc_k_gain"] = float(ecu.live_smc_k_gain)
		out["smc_lambda_beta"] = float(ecu.live_smc_lambda_beta)
		out["smc_k_ffd"] = float(ecu.live_smc_k_ffd)
		out["smc_tc_kappa_base"] = float(ecu.live_smc_tc_kappa_base)
		out["eco_power_cap_kw"] = float(ecu.live_eco_power_cap_kw)
	return out

func apply_to_car(setup: Dictionary, config, ecu, reset_cb: Callable = Callable()) -> bool:
	if setup.is_empty():
		return false
	var mech: Dictionary = setup.get("mechanical", setup)
	var elec: Dictionary = setup.get("electronic", setup)

	if config != null:
		var live_p: Dictionary = {}
		if mech.has("spring_rate_front_n_m"):
			live_p["spring_stiffness_front"] = clampf(float(mech["spring_rate_front_n_m"]), 15000.0, 120000.0)
		if mech.has("spring_rate_rear_n_m"):
			live_p["spring_stiffness_rear"] = clampf(float(mech["spring_rate_rear_n_m"]), 15000.0, 120000.0)
		if mech.has("arb_stiffness_front_nm_rad"):
			live_p["arb_stiffness_front"] = clampf(float(mech["arb_stiffness_front_nm_rad"]), 0.0, 25000.0)
		if mech.has("arb_stiffness_rear_nm_rad"):
			live_p["arb_stiffness_rear"] = clampf(float(mech["arb_stiffness_rear_nm_rad"]), 0.0, 25000.0)
		if mech.has("static_camber_front_deg"):
			live_p["static_camber_front_deg"] = clampf(float(mech["static_camber_front_deg"]), -5.0, 1.0)
		if mech.has("static_camber_rear_deg"):
			live_p["static_camber_rear_deg"] = clampf(float(mech["static_camber_rear_deg"]), -5.0, 1.0)
		if mech.has("brake_bias_front"):
			live_p["brake_bias_front"] = clampf(float(mech["brake_bias_front"]), 0.40, 0.85)
		if mech.has("ride_height_front_m"):
			live_p["static_ride_height_front"] = clampf(float(mech["ride_height_front_m"]), 0.020, 0.080)
		if mech.has("ride_height_rear_m"):
			live_p["static_ride_height_rear"] = clampf(float(mech["ride_height_rear_m"]), 0.020, 0.080)
		if config.has_method("apply_setup_params"):
			config.apply_setup_params(live_p)
		if config.has_method("update_derived_geometry"):
			config.update_derived_geometry()

	if ecu != null:
		if elec.has("tv_map_filename") and String(elec["tv_map_filename"]) != "":
			var map_fn: String = String(elec["tv_map_filename"])
			var map_path: String = "res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/%s.json" % map_fn
			if FileAccess.file_exists(map_path) and ecu.has_method("load_tv_map_from_file"):
				ecu.load_tv_map_from_file(map_path)
		if elec.has("control_law_mode"):
			ecu.control_law_mode = int(elec["control_law_mode"])
		if elec.has("grip_cap"):
			ecu.set_live_param("grip_cap", float(elec["grip_cap"]))
		if elec.has("smc_k_gain"):
			ecu.set_live_param("smc_k_gain", float(elec["smc_k_gain"]))
		if elec.has("smc_lambda_beta"):
			ecu.set_live_param("smc_lambda_beta", float(elec["smc_lambda_beta"]))
		if elec.has("smc_k_ffd"):
			ecu.set_live_param("smc_k_ffd", float(elec["smc_k_ffd"]))
		if elec.has("smc_tc_kappa_base"):
			ecu.set_live_param("smc_tc_kappa_base", float(elec["smc_tc_kappa_base"]))
		if elec.has("eco_power_cap_kw"):
			ecu.live_eco_power_cap_kw = clampf(float(elec["eco_power_cap_kw"]), 0.0, 120.0)

	if reset_cb.is_valid():
		reset_cb.call()
	return true

func list_presets() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var seen: Dictionary = {}
	for dir_path in [RES_SETUPS_DIR, USER_SETUPS_DIR]:
		var d := DirAccess.open(dir_path)
		if d == null:
			continue
		d.list_dir_begin()
		var fname := d.get_next()
		while fname != "":
			if not d.current_is_dir() and fname.ends_with(".json"):
				var full_path: String = dir_path + fname
				var meta: Dictionary = _read_json(full_path)
				var base_fn: String = fname.get_basename()
				var entry := {
					"name": String(meta.get("name", base_fn)),
					"filename": base_fn,
					"path": full_path,
					"description": String(meta.get("description", ""))
				}
				seen[base_fn] = entry
			fname = d.get_next()
		d.list_dir_end()
	var keys := seen.keys()
	keys.sort()
	for k in keys:
		out.append(seen[k])
	return out

func load_preset(file_path: String, config, ecu, reset_cb: Callable = Callable()) -> bool:
	var data: Dictionary = _read_json(file_path)
	if data.is_empty():
		return false
	base_preset_name = String(data.get("name", file_path.get_file().get_basename()))
	base_preset_path = file_path
	var ok: bool = apply_to_car(data, config, ecu, reset_cb)
	if ok:
		base_snapshot = capture_from_car(config, ecu)
	return ok

func set_base_from_car(config, ecu, preset_label: String = "Baseline Actual") -> void:
	base_preset_name = preset_label
	base_snapshot = capture_from_car(config, ecu)

func save_preset(preset_name: String, config, ecu, description: String = "") -> String:
	var clean: String = preset_name.strip_edges().to_lower().replace(" ", "_")
	if clean == "":
		clean = "custom_setup"
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(USER_SETUPS_DIR))
	var target_path: String = USER_SETUPS_DIR + clean + ".json"
	var snap: Dictionary = capture_from_car(config, ecu)
	var payload := {
		"name": preset_name.strip_edges() if preset_name.strip_edges() != "" else clean,
		"description": description,
		"mechanical": {
			"spring_rate_front_n_m": snap.get("spring_rate_front_n_m", 45000.0),
			"spring_rate_rear_n_m": snap.get("spring_rate_rear_n_m", 45000.0),
			"arb_stiffness_front_nm_rad": snap.get("arb_stiffness_front_nm_rad", 4000.0),
			"arb_stiffness_rear_nm_rad": snap.get("arb_stiffness_rear_nm_rad", 4000.0),
			"static_camber_front_deg": snap.get("static_camber_front_deg", -1.8),
			"static_camber_rear_deg": snap.get("static_camber_rear_deg", -1.2),
			"brake_bias_front": snap.get("brake_bias_front", 0.58),
			"ride_height_front_m": snap.get("ride_height_front_m", 0.032),
			"ride_height_rear_m": snap.get("ride_height_rear_m", 0.035)
		},
		"electronic": {
			"tv_map_filename": snap.get("tv_map_filename", "02_autocross_neutral"),
			"control_law_mode": snap.get("control_law_mode", 1),
			"grip_cap": snap.get("grip_cap", 0.94),
			"smc_k_gain": snap.get("smc_k_gain", 920.0),
			"smc_lambda_beta": snap.get("smc_lambda_beta", 6.8),
			"smc_k_ffd": snap.get("smc_k_ffd", 95.0),
			"smc_tc_kappa_base": snap.get("smc_tc_kappa_base", 0.052),
			"eco_power_cap_kw": snap.get("eco_power_cap_kw", 0.0)
		}
	}
	var f := FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(JSON.stringify(payload, "\t") + "\n")
	f.close()
	base_preset_name = String(payload["name"])
	base_preset_path = target_path
	base_snapshot = snap
	return target_path

func compute_diffs_vs_base(current_snap: Dictionary) -> Array[Dictionary]:
	var diffs: Array[Dictionary] = []
	if base_snapshot.is_empty():
		return diffs
	for spec in PARAM_SPECS:
		var key: String = String(spec["key"])
		if not current_snap.has(key) or not base_snapshot.has(key):
			continue
		var cur_v = current_snap[key]
		var base_v = base_snapshot[key]
		if cur_v is String or base_v is String:
			if String(cur_v) != String(base_v):
				diffs.append({
					"key": key,
					"label": String(spec["label"]),
					"base": String(base_v),
					"current": String(cur_v),
					"unit": String(spec["unit"])
				})
		else:
			var tol: float = float(spec["tol"])
			var sc: float = float(spec["scale"])
			if absf(float(cur_v) - float(base_v)) > tol:
				diffs.append({
					"key": key,
					"label": String(spec["label"]),
					"base": "%.2f" % (float(base_v) * sc),
					"current": "%.2f" % (float(cur_v) * sc),
					"unit": String(spec["unit"])
				})
	return diffs

static func _read_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	return parsed if (parsed is Dictionary) else {}
