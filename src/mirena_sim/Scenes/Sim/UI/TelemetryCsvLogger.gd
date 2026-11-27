class_name TelemetryCsvLogger
extends RefCounted

## Session CSV telemetry logger writing to user://telemetry_logs/run_*.csv at 50 / 100 / 200 Hz.

const LOG_DIR := "user://telemetry_logs"
const VALID_RATES_HZ: Array[int] = [50, 100, 200]

var sample_rate_hz: int = 100
var is_recording: bool = false
var auto_record_laps: bool = false
var last_file_path: String = ""
var sample_count: int = 0

var _file: FileAccess = null
var _t_elapsed_s: float = 0.0
var _accum_s: float = 0.0
var _sample_interval_s: float = 0.01
var _prev_beta_deg: float = 0.0
var _has_prev_beta: bool = false
var _last_auto_lap: int = -1


func set_sample_rate_hz(hz: int) -> int:
	var best_hz := 100
	var best_diff := 999999
	for candidate in VALID_RATES_HZ:
		var diff := absi(hz - candidate)
		if diff < best_diff:
			best_diff = diff
			best_hz = candidate
	sample_rate_hz = best_hz
	_sample_interval_s = 1.0 / float(sample_rate_hz)
	return sample_rate_hz


func get_sample_rate_hz() -> int:
	return sample_rate_hz


func set_auto_record_laps(enabled: bool) -> void:
	auto_record_laps = enabled


func open_logs_folder() -> String:
	var abs_dir := ProjectSettings.globalize_path(LOG_DIR)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	if DisplayServer.get_name() != "headless":
		OS.shell_open(abs_dir)
	return abs_dir


func start_recording(track_name: String = "", tv_map: String = "") -> String:
	stop_recording()
	var abs_dir := ProjectSettings.globalize_path(LOG_DIR)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	var dt_dict := Time.get_datetime_dict_from_system()
	var stamp := "%04d%02d%02d_%02d%02d%02d" % [
		int(dt_dict.get("year", 2026)),
		int(dt_dict.get("month", 1)),
		int(dt_dict.get("day", 1)),
		int(dt_dict.get("hour", 0)),
		int(dt_dict.get("minute", 0)),
		int(dt_dict.get("second", 0)),
	]
	var safe_trk := track_name.strip_edges().replace(" ", "_").to_lower()
	if safe_trk.is_empty():
		safe_trk = "track"
	last_file_path = "%s/run_%s_%s.csv" % [LOG_DIR, safe_trk, stamp]
	_file = FileAccess.open(last_file_path, FileAccess.WRITE)
	if _file == null:
		is_recording = false
		return ""
	_file.store_line("t,lap,s_m,vx,vy,wz,ax,ay,beta_deg,beta_dot_dps,delta_raw,delta_deg,thr,brk,T_FL,T_FR,T_RL,T_RR,Fz_FL,Fz_FR,Fz_RL,Fz_RR,P_dc_kw,Mz_TV_nm,tv_mode,tv_regime")
	is_recording = true
	sample_count = 0
	_t_elapsed_s = 0.0
	_sample_interval_s = 1.0 / float(maxi(sample_rate_hz, 1))
	_accum_s = _sample_interval_s
	_prev_beta_deg = 0.0
	_has_prev_beta = false
	return last_file_path


func stop_recording() -> String:
	if _file != null:
		_file.flush()
		_file.close()
		_file = null
	is_recording = false
	_last_auto_lap = -1
	return last_file_path


func toggle_recording(track_name: String = "", tv_map: String = "") -> bool:
	if is_recording:
		stop_recording()
	else:
		start_recording(track_name, tv_map)
	return is_recording


func on_lap_update(lap_active: bool, lap_idx: int, track_name: String = "", tv_map: String = "") -> void:
	if not auto_record_laps:
		return
	if lap_active and not is_recording:
		_last_auto_lap = lap_idx
		start_recording(track_name, tv_map)
	elif lap_active and is_recording and _last_auto_lap != -1 and lap_idx != _last_auto_lap:
		_last_auto_lap = lap_idx
		start_recording(track_name, tv_map)
	elif not lap_active and is_recording and _last_auto_lap != -1:
		stop_recording()


func log_step(dt: float, snap: Dictionary, car: Node, lap_idx: int = 1, s_m: float = 0.0, tv_map: String = "") -> void:
	if not is_recording or _file == null:
		return
	var dt_safe := maxf(dt, 0.0)
	_t_elapsed_s += dt_safe
	_accum_s += dt_safe
	if _accum_s < _sample_interval_s:
		return
	var step_dt := _accum_s
	_accum_s = 0.0

	var t_cmd: Array = snap.get("t_cmd", [0.0, 0.0, 0.0, 0.0]) if snap.get("t_cmd") is Array else [0.0, 0.0, 0.0, 0.0]
	var fl := float(t_cmd[0]) if t_cmd.size() > 0 else 0.0
	var fr := float(t_cmd[1]) if t_cmd.size() > 1 else 0.0
	var rl := float(t_cmd[2]) if t_cmd.size() > 2 else 0.0
	var rr := float(t_cmd[3]) if t_cmd.size() > 3 else 0.0

	var fz := [0.0, 0.0, 0.0, 0.0]
	var vx := float(snap.get("kmh", 0.0)) / 3.6
	var vy := 0.0
	var wz := float(car.angular_velocity.y) if (car != null and car is RigidBody3D) else 0.0
	if car != null and ("model" in car) and car.model != null:
		if "linear_velocity_ros" in car.model:
			var lv: Vector3 = car.model.linear_velocity_ros
			vx = lv.x
			vy = lv.y
		if "angular_velocity_ros" in car.model:
			wz = float(car.model.angular_velocity_ros.z)
		if "wheels" in car.model and car.model.wheels is Array:
			for i in range(mini(4, car.model.wheels.size())):
				if car.model.wheels[i] != null and "F_z" in car.model.wheels[i]:
					fz[i] = float(car.model.wheels[i].F_z)

	var tv_diag: Dictionary = {}
	if car != null:
		if car.has_method("get_tv_live_diagnostics"):
			var d_var: Variant = car.get_tv_live_diagnostics()
			if d_var is Dictionary:
				tv_diag = d_var
		elif ("tv_controller" in car) and car.tv_controller != null and car.tv_controller.has_method("get_tv_live_diagnostics"):
			var d_var2: Variant = car.tv_controller.get_tv_live_diagnostics()
			if d_var2 is Dictionary:
				tv_diag = d_var2

	var steer_rad := float(car.steering) if (car != null and "steering" in car) else 0.0
	var delta_deg := float(tv_diag.get("delta_deg", rad_to_deg(steer_rad)))
	var delta_raw := float(tv_diag.get("delta_norm_raw", car.raw_steer_norm if (car != null and "raw_steer_norm" in car) else steer_rad))
	if not tv_diag.has("delta_norm_raw") and not (car != null and "raw_steer_norm" in car) and car != null and ("input_conditioner" in car) and car.input_conditioner != null and car.input_conditioner.has_method("get_live_input_monitor"):
		var mon: Variant = car.input_conditioner.get_live_input_monitor()
		if mon is Dictionary:
			delta_raw = float((mon as Dictionary).get("steer_raw", steer_rad))

	var thr := float(car.gas) if (car != null and "gas" in car) else 0.0
	var brk := float(snap.get("brake", 0.0))
	var beta_deg := float(snap.get("beta_deg", 0.0))
	var beta_dot_dps := 0.0
	if tv_diag.has("beta_dot_dps"):
		beta_dot_dps = float(tv_diag["beta_dot_dps"])
	elif _has_prev_beta and step_dt > 1e-4:
		beta_dot_dps = (beta_deg - _prev_beta_deg) / step_dt
	_prev_beta_deg = beta_deg
	_has_prev_beta = true

	var ax := float(snap.get("ax_g", 0.0)) * 9.81
	var ay := float(snap.get("ay_g", 0.0)) * 9.81
	var tv_mode := str(snap.get("mode_name", tv_map))
	if tv_mode.is_empty():
		tv_mode = tv_map if not tv_map.is_empty() else "AUTOCROSS"
	var tv_regime := str(tv_diag.get("active_regime", tv_diag.get("regime", tv_diag.get("tv_regime", "UNKNOWN"))))

	_file.store_line("%.4f,%d,%.2f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.2f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%s,%s" % [
		_t_elapsed_s,
		lap_idx,
		s_m,
		vx,
		vy,
		wz,
		ax,
		ay,
		beta_deg,
		beta_dot_dps,
		delta_raw,
		delta_deg,
		thr,
		brk,
		fl, fr, rl, rr,
		fz[0], fz[1], fz[2], fz[3],
		float(snap.get("p_dc_kw", 0.0)),
		float(snap.get("mz_tv_nm", 0.0)),
		tv_mode.replace(",", "_"),
		tv_regime.replace(",", "_"),
	])
	sample_count += 1
	if (sample_count % 50) == 0:
		_file.flush()
