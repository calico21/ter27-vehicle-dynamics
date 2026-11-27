class_name TelemetryRecorder
extends RefCounted

## exportable csv telemetry recorder (100-200 hz, motec i2 & pandas compatible)
## logs physical vehicle states, torque vectoring commands, tire temperatures,
## and hv powertrain channels to `user://telemetry/<track>_<map_name>_<timestamp>.csv`

const VehicleDynamicsModel = preload("../Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("../../Inputs/VehicleInputs.gd")
const VehicleTelemetry = preload("VehicleTelemetry.gd")

const TELEMETRY_DIR: String = "user://telemetry/"
const FLUSH_INTERVAL_SAMPLES: int = 50
const CSV_HEADER: String = "time,dist_m,vx,vy,wz,wz_ref,beta_deg,ax,ay,steer_deg,gas,brake,map_id,mz_req,T_fl,T_fr,T_rl,T_rr,kappa_fl,kappa_fr,kappa_rl,kappa_rr,tire_temp_fl,tire_temp_fr,tire_temp_rl,tire_temp_rr,v_dc,i_dc,power_kw,energy_kwh,fz_fl,fz_fr,fz_rl,fz_rr,brake_bias_front,regen_blending_enabled,disc_temp_fl,disc_temp_fr,disc_temp_rl,disc_temp_rr,surface_mu_fl,surface_mu_fr,surface_mu_rl,surface_mu_rr,regen_power_kw"

## target logging frequency in hz (typically 100.0 - 200.0 hz)
var sample_rate_hz: float = 200.0:
	set(value):
		sample_rate_hz = clampf(value, 10.0, 400.0)

var is_recording: bool = false
var sim_time: float = 0.0
var dist_m: float = 0.0

var _sample_acc: float = 0.0
var _sample_count: int = 0
var _last_file_path: String = ""
var _file: FileAccess = null
var _energy_kwh_acc: float = 0.0

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _file != null:
		_file.flush()
		_file.close()
		_file = null

static func _sanitize_name(raw: String, fallback: String = "unknown") -> String:
	var out: String = ""
	var prev_us: bool = false
	for idx in range(raw.length()):
		var ch: String = raw[idx]
		var code: int = ch.unicode_at(0)
		var is_alnum: bool = (
			(code >= 48 and code <= 57) or
			(code >= 65 and code <= 90) or
			(code >= 97 and code <= 122) or
			ch == "-" or ch == "_"
		)
		if is_alnum:
			out += ch
			prev_us = false
		elif not prev_us and out.length() > 0:
			out += "_"
			prev_us = true
	out = out.strip_edges().trim_prefix("_").trim_suffix("_")
	return out if out.length() > 0 else fallback

static func _make_timestamp_tag() -> String:
	var dt_dict: Dictionary = Time.get_datetime_dict_from_system(false)
	return "%04d%02d%02d_%02d%02d%02d" % [
		int(dt_dict.get("year", 2026)),
		int(dt_dict.get("month", 1)),
		int(dt_dict.get("day", 1)),
		int(dt_dict.get("hour", 0)),
		int(dt_dict.get("minute", 0)),
		int(dt_dict.get("second", 0))
	]

## starts a new csv telemetry recording session and returns the output file path
func start_recording(track_name: String = "manual", map_name: String = "map2", custom_path: String = "") -> String:
	if is_recording:
		stop_recording()

	var target_path: String = custom_path.strip_edges()
	if target_path == "":
		var err_dir: Error = DirAccess.make_dir_recursive_absolute(TELEMETRY_DIR)
		if err_dir != OK:
			push_warning("TelemetryRecorder: make_dir_recursive_absolute('%s') returned %d" % [TELEMETRY_DIR, err_dir])
		var safe_track: String = _sanitize_name(track_name, "manual")
		var safe_map: String = _sanitize_name(map_name, "map2")
		var ts: String = _make_timestamp_tag()
		target_path = "%s%s_%s_%s.csv" % [TELEMETRY_DIR, safe_track, safe_map, ts]
	else:
		var base_dir: String = target_path.get_base_dir()
		if base_dir != "":
			DirAccess.make_dir_recursive_absolute(base_dir)

	var f: FileAccess = FileAccess.open(target_path, FileAccess.WRITE)
	if f == null:
		push_error("TelemetryRecorder: failed to open '%s' (err=%d)" % [target_path, FileAccess.get_open_error()])
		is_recording = false
		return ""

	_file = f
	_file.store_line(CSV_HEADER)
	_file.flush()

	is_recording = true
	sim_time = 0.0
	dist_m = 0.0
	_sample_acc = 0.0
	_sample_count = 0
	_energy_kwh_acc = 0.0
	_last_file_path = target_path
	return _last_file_path

## flushes and closes the active csv recording, returning the recorded file path
func stop_recording() -> String:
	if _file != null:
		_file.flush()
		_file.close()
		_file = null
	is_recording = false
	return _last_file_path

## toggles active telemetry recording on/off; returns the new recording state
func toggle_recording(track_name: String = "manual", map_name: String = "map2") -> bool:
	if is_recording:
		stop_recording()
		return false
	var path: String = start_recording(track_name, map_name)
	return is_recording and path != ""

## advances distance, simulation time, energy integration, and writes samples at 1 / sample_rate_hz
func record_step(
	dt: float,
	model,
	inputs: VehicleInputs,
	step_res: Dictionary = {},
	extra: Dictionary = {}
) -> void:
	if not is_recording or _file == null or model == null or dt <= 0.0:
		return

	var vx: float = model.linear_velocity_ros.x
	var vy: float = model.linear_velocity_ros.y
	var wz: float = model.angular_velocity_ros.z

	sim_time += dt
	dist_m += sqrt(vx * vx + vy * vy) * dt
	_sample_acc += dt

	# extract torques and integrate hv energy at full physics tick resolution
	var forces_data: Dictionary = step_res.get("forces", step_res) if step_res.get("forces") is Dictionary else step_res
	var d_trqs: Array = forces_data.get("drive_torques", [0.0, 0.0, 0.0, 0.0])
	if extra.has("drive_torques") and extra["drive_torques"] is Array:
		d_trqs = extra["drive_torques"]

	var hv_merged: Dictionary = step_res.duplicate()
	for k in extra.keys():
		hv_merged[k] = extra[k]
	var hv: Dictionary = VehicleTelemetry.extract_hv_state(model, d_trqs, dt, hv_merged, _energy_kwh_acc)
	_energy_kwh_acc = float(hv["energy_kwh"])

	var sample_period: float = 1.0 / clampf(sample_rate_hz, 10.0, 400.0)
	if _sample_acc + 1e-9 < sample_period:
		return
	while _sample_acc + 1e-9 >= sample_period:
		_sample_acc = maxf(0.0, _sample_acc - sample_period)

	var steer_rad: float = inputs.steer_angle if inputs != null else 0.0
	var steer_deg: float = rad_to_deg(steer_rad)
	var gas_val: float = inputs.throttle if inputs != null else 0.0
	var brake_val: float = inputs.brake if inputs != null else 0.0
	var beta_deg: float = VehicleTelemetry.compute_beta_deg(vx, vy)

	var mass_kg: float = maxf(model.config.mass, 1e-3) if model.config != null else 290.0
	var ax: float = float(extra.get("ax", model.net_force_chassis_ros.x / mass_kg))
	var ay: float = float(extra.get("ay", model.net_force_chassis_ros.y / mass_kg))

	var tv_info: Dictionary = VehicleTelemetry.extract_tv_state(model, steer_rad, hv_merged)
	var wz_ref: float = float(tv_info["wz_ref"])
	var map_id: int = int(tv_info["map_id"])
	var mz_req: float = float(tv_info["mz_req"])

	var temps: Array[float] = VehicleTelemetry.extract_tire_temps(model, hv_merged)
	var bs_info: Dictionary = VehicleTelemetry.extract_brake_and_surface_state(model, hv_merged)
	var disc_temps: Array = bs_info["disc_temp"]
	var surf_mus: Array = bs_info["surface_mu"]
	var bias_front: float = float(bs_info["brake_bias_front"])
	var regen_en_int: int = 1 if bool(bs_info["regen_blending_enabled"]) else 0
	var regen_kw: float = float(bs_info["regen_power_kw"])

	var t_vals: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var k_vals: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var fz_vals: Array[float] = [0.0, 0.0, 0.0, 0.0]
	var nf_arr: Array = forces_data.get("normal_forces", step_res.get("normal_forces", []))

	for i in range(4):
		t_vals[i] = float(d_trqs[i]) if i < d_trqs.size() else 0.0
		if i < model.wheels.size():
			k_vals[i] = model.wheels[i].kappa
			fz_vals[i] = float(nf_arr[i]) if i < nf_arr.size() else model.wheels[i].F_z

	var row: String = (
		"%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.3f," % [
			sim_time, dist_m, vx, vy, wz, wz_ref, beta_deg, ax, ay,
			steer_deg, gas_val, brake_val, map_id, mz_req
		]
	) + (
		"%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.5f," % [
			t_vals[0], t_vals[1], t_vals[2], t_vals[3],
			k_vals[0], k_vals[1], k_vals[2], k_vals[3]
		]
	) + (
		"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.5f,%.1f,%.1f,%.1f,%.1f," % [
			temps[0], temps[1], temps[2], temps[3],
			float(hv["v_dc"]), float(hv["i_dc"]), float(hv["power_kw"]), float(hv["energy_kwh"]),
			fz_vals[0], fz_vals[1], fz_vals[2], fz_vals[3]
		]
	) + (
		"%.4f,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.3f" % [
			bias_front, regen_en_int,
			float(disc_temps[0]), float(disc_temps[1]), float(disc_temps[2]), float(disc_temps[3]),
			float(surf_mus[0]), float(surf_mus[1]), float(surf_mus[2]), float(surf_mus[3]),
			regen_kw
		]
	)

	_file.store_line(row)
	_sample_count += 1
	if _sample_count % FLUSH_INTERVAL_SAMPLES == 0:
		_file.flush()

## returns hud recording indicator string
func get_hud_indicator() -> String:
	return "  |  [REC]" if is_recording else ""

## returns path of the current or most recent telemetry csv file
func get_last_file_path() -> String:
	return _last_file_path

## returns number of csv data rows written in the current or last session
func get_sample_count() -> int:
	return _sample_count
