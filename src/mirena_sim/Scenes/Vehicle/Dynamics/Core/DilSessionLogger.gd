class_name DilSessionLogger
extends RefCounted

## Automated Driver-in-the-Loop (DiL) Session, Lap & A/B KPI Logger for TeR27.
## Records 50 Hz CSV telemetry per lap/run, detects finish-line & 75 m Acceleration
## crossings automatically, computes KPIs, exports summary JSON + Markdown report,
## and provides live time-delta (Delta-t) against a pinned Baseline reference lap.

const SAMPLE_DT: float = 0.02 # 50 Hz
const CSV_HEADER: String = "t,s_dist,vx,ay,ax,wz_ref,wz_real,beta_deg,delta_deg,throttle,brake,Mz_req,T_cmd_FL,T_cmd_FR,T_cmd_RL,T_cmd_RR,TC_cut_FL,TC_cut_FR,TC_cut_RL,TC_cut_RR,Fz_FL,Fz_FR,Fz_RL,Fz_RR,kappa_FL,kappa_FR,kappa_RL,kappa_RR,alpha_FL,alpha_FR,alpha_RL,alpha_RR,T_tire_FL,T_tire_FR,T_tire_RL,T_tire_RR,T_disc_FL,T_disc_FR,T_disc_RL,T_disc_RR,P_elec_kw"

var session_id: String = ""
var session_dir: String = ""
var event_type: String = "circuit" # "circuit", "acceleration", "skidpad"
var min_lap_dist_m: float = 30.0

var start_line_pos: Vector3 = Vector3.ZERO
var start_line_fwd: Vector3 = Vector3(1.0, 0.0, 0.0)
var _prev_signed_dist: float = 0.0
var _has_prev_pos: bool = false
var _sample_acc: float = 0.0

var current_lap_index: int = 1
var current_lap_time_s: float = 0.0
var current_lap_dist_m: float = 0.0
var best_lap_time_s: float = 0.0
var _accel_running: bool = false

var _rows: PackedStringArray = PackedStringArray()
var _dist_trace: PackedFloat64Array = PackedFloat64Array()
var _time_trace: PackedFloat64Array = PackedFloat64Array()

# Live KPI accumulators for current lap
var _max_ay_g: float = 0.0
var _max_ax_g: float = 0.0
var _sum_wz_err_sq: float = 0.0
var _peak_beta_deg: float = 0.0
var _sum_beta_sq: float = 0.0
var _sum_sat_pct: float = 0.0
var _energy_j: float = 0.0
var _max_tire_temp_c: float = 0.0
var _max_disc_temp_c: float = 0.0
var _kpi_count: int = 0
var _prev_pos: Vector3 = Vector3.ZERO

var completed_laps: Array[Dictionary] = []
var baseline_lap: Dictionary = {}
var pilot_notes: String = ""

func _init(custom_session_dir: String = "") -> void:
	start_new_session(custom_session_dir)

func start_new_session(custom_dir: String = "") -> void:
	var dt := Time.get_datetime_dict_from_system()
	session_id = "%04d-%02d-%02d_%02d-%02d" % [
		int(dt.get("year", 2026)), int(dt.get("month", 1)), int(dt.get("day", 1)),
		int(dt.get("hour", 0)), int(dt.get("minute", 0))
	]
	session_dir = custom_dir if custom_dir != "" else ("user://dil_sessions/%s/" % session_id)
	if not session_dir.ends_with("/"):
		session_dir += "/"
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(session_dir))
	completed_laps.clear()
	baseline_lap.clear()
	current_lap_index = 1
	best_lap_time_s = 0.0
	_reset_lap_accumulators()

func configure_track(mode_name: String, line_pos: Vector3, line_fwd: Vector3, min_dist_m: float = 30.0) -> void:
	event_type = mode_name.to_lower()
	start_line_pos = line_pos
	start_line_fwd = line_fwd.normalized() if line_fwd.length_squared() > 1e-6 else Vector3(1.0, 0.0, 0.0)
	min_lap_dist_m = min_dist_m
	_has_prev_pos = false
	_accel_running = false

func _reset_lap_accumulators() -> void:
	current_lap_time_s = 0.0
	current_lap_dist_m = 0.0
	_sample_acc = 0.0
	_rows = PackedStringArray()
	_dist_trace = PackedFloat64Array()
	_time_trace = PackedFloat64Array()
	_max_ay_g = 0.0
	_max_ax_g = 0.0
	_sum_wz_err_sq = 0.0
	_peak_beta_deg = 0.0
	_sum_beta_sq = 0.0
	_sum_sat_pct = 0.0
	_energy_j = 0.0
	_max_tire_temp_c = 0.0
	_max_disc_temp_c = 0.0
	_kpi_count = 0

## Steps the logger at physics rate; records 50 Hz samples and returns completed lap dict (or {}).
func step_sample(
	dt: float,
	pos: Vector3,
	vel_world: Vector3,
	sample: Dictionary,
	setup_snapshot: Dictionary
) -> Dictionary:
	if dt <= 0.0:
		return {}
	var vx: float = float(sample.get("vx", 0.0))
	if not _has_prev_pos:
		_prev_pos = pos
		_prev_signed_dist = (pos - start_line_pos).dot(start_line_fwd)
		_has_prev_pos = true

	var ds: float = pos.distance_to(_prev_pos)
	_prev_pos = pos

	if event_type == "acceleration":
		if not _accel_running and absf(vx) > 0.8:
			_accel_running = true
			_reset_lap_accumulators()
		if _accel_running:
			current_lap_time_s += dt
			current_lap_dist_m += ds
	else:
		current_lap_time_s += dt
		current_lap_dist_m += ds

	# Accumulate KPIs
	var ay_ms2: float = float(sample.get("ay", 0.0))
	var ax_ms2: float = float(sample.get("ax", 0.0))
	var wz_ref: float = float(sample.get("wz_ref", 0.0))
	var wz_real: float = float(sample.get("wz_real", 0.0))
	var beta_deg: float = float(sample.get("beta_deg", 0.0))
	var p_elec_kw: float = float(sample.get("p_elec_kw", 0.0))
	var sat_pct: float = float(sample.get("avg_sat_pct", 0.0))
	var t_tires: Array = sample.get("t_tire", [55.0, 55.0, 55.0, 55.0])
	var t_discs: Array = sample.get("t_disc", [40.0, 40.0, 40.0, 40.0])

	if event_type != "acceleration" or _accel_running:
		_max_ay_g = maxf(_max_ay_g, absf(ay_ms2) / 9.81)
		_max_ax_g = maxf(_max_ax_g, absf(ax_ms2) / 9.81)
		var wz_e: float = wz_ref - wz_real
		_sum_wz_err_sq += wz_e * wz_e
		_peak_beta_deg = maxf(_peak_beta_deg, absf(beta_deg))
		_sum_beta_sq += beta_deg * beta_deg
		_sum_sat_pct += sat_pct
		_energy_j += maxf(0.0, p_elec_kw) * 1000.0 * dt
		for i in range(4):
			if i < t_tires.size():
				_max_tire_temp_c = maxf(_max_tire_temp_c, float(t_tires[i]))
			if i < t_discs.size():
				_max_disc_temp_c = maxf(_max_disc_temp_c, float(t_discs[i]))
		_kpi_count += 1

		_sample_acc += dt
		if _sample_acc >= SAMPLE_DT - 1e-5:
			_sample_acc -= SAMPLE_DT
			_append_csv_row(sample)

	# Lap / run completion check
	var signed_dist: float = (pos - start_line_pos).dot(start_line_fwd)
	var crossed_finish: bool = false
	if event_type == "acceleration":
		if _accel_running and current_lap_dist_m >= 75.0:
			crossed_finish = true
			_accel_running = false
	else:
		var fwd_speed: float = vel_world.dot(start_line_fwd)
		if _prev_signed_dist < 0.0 and signed_dist >= 0.0 and current_lap_dist_m >= min_lap_dist_m and fwd_speed > 0.5:
			crossed_finish = true
	_prev_signed_dist = signed_dist

	if crossed_finish:
		return finalize_lap(setup_snapshot)
	return {}

func _append_csv_row(s: Dictionary) -> void:
	var t_cmd: Array = s.get("t_cmd", [0.0, 0.0, 0.0, 0.0])
	var tc_cut_raw = s.get("tc_cut", [0.0, 0.0, 0.0, 0.0])
	var tc_cut: Array = tc_cut_raw if (tc_cut_raw is Array and tc_cut_raw.size() >= 4) else [float(tc_cut_raw), float(tc_cut_raw), float(tc_cut_raw), float(tc_cut_raw)]
	var fz: Array = s.get("fz", [0.0, 0.0, 0.0, 0.0])
	var kap: Array = s.get("kappa", [0.0, 0.0, 0.0, 0.0])
	var alp: Array = s.get("alpha", [0.0, 0.0, 0.0, 0.0])
	var tt: Array = s.get("t_tire", [55.0, 55.0, 55.0, 55.0])
	var td: Array = s.get("t_disc", [40.0, 40.0, 40.0, 40.0])
	_dist_trace.append(current_lap_dist_m)
	_time_trace.append(current_lap_time_s)
	var row := "%.3f,%.2f,%.3f,%.3f,%.3f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f" % [
		current_lap_time_s, current_lap_dist_m,
		float(s.get("vx", 0.0)), float(s.get("ay", 0.0)), float(s.get("ax", 0.0)),
		float(s.get("wz_ref", 0.0)), float(s.get("wz_real", 0.0)),
		float(s.get("beta_deg", 0.0)), float(s.get("delta_deg", 0.0)),
		float(s.get("throttle", 0.0)), float(s.get("brake", 0.0)), float(s.get("mz_req", 0.0)),
		float(t_cmd[0]), float(t_cmd[1]), float(t_cmd[2]), float(t_cmd[3]),
		float(tc_cut[0]), float(tc_cut[1]), float(tc_cut[2]), float(tc_cut[3]),
		float(fz[0]), float(fz[1]), float(fz[2]), float(fz[3]),
		float(kap[0]), float(kap[1]), float(kap[2]), float(kap[3]),
		float(alp[0]), float(alp[1]), float(alp[2]), float(alp[3]),
		float(tt[0]), float(tt[1]), float(tt[2]), float(tt[3]),
		float(td[0]), float(td[1]), float(td[2]), float(td[3]),
		float(s.get("p_elec_kw", 0.0))
	]
	_rows.append(row)

func finalize_lap(setup_snapshot: Dictionary, notes: String = "") -> Dictionary:
	var n: float = maxf(float(_kpi_count), 1.0)
	var dist_km: float = maxf(current_lap_dist_m / 1000.0, 1e-4)
	var energy_kwh: float = _energy_j / 3600000.0
	var csv_name: String = "telemetry_lap_%02d.csv" % current_lap_index
	var csv_path: String = session_dir.path_join(csv_name)

	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(session_dir))
	var f_csv := FileAccess.open(csv_path, FileAccess.WRITE)
	if f_csv != null:
		f_csv.store_line(CSV_HEADER)
		for r in _rows:
			f_csv.store_line(r)
		f_csv.close()

	var lap_kpis := {
		"lap_index": current_lap_index,
		"event_type": event_type,
		"lap_time_s": snappedf(current_lap_time_s, 0.001),
		"dist_m": snappedf(current_lap_dist_m, 0.1),
		"max_ay_g": snappedf(_max_ay_g, 0.01),
		"max_ax_g": snappedf(_max_ax_g, 0.01),
		"wz_rmse": snappedf(sqrt(_sum_wz_err_sq / n), 0.0001),
		"peak_beta_deg": snappedf(_peak_beta_deg, 0.01),
		"rms_beta_deg": snappedf(sqrt(_sum_beta_sq / n), 0.01),
		"avg_tire_sat_pct": snappedf(_sum_sat_pct / n, 0.1),
		"energy_kwh_km": snappedf(energy_kwh / dist_km, 0.001),
		"max_tire_temp_c": snappedf(_max_tire_temp_c, 0.1),
		"max_disc_temp_c": snappedf(_max_disc_temp_c, 0.1),
		"csv_file": csv_name,
		"notes": notes if notes != "" else pilot_notes,
		"setup_snapshot": setup_snapshot.duplicate(true),
		"dist_trace": Array(_dist_trace),
		"time_trace": Array(_time_trace)
	}
	if best_lap_time_s <= 0.0 or current_lap_time_s < best_lap_time_s:
		best_lap_time_s = current_lap_time_s
	completed_laps.append(lap_kpis)
	current_lap_index += 1
	export_session_reports()
	_reset_lap_accumulators()
	return lap_kpis

func set_baseline_lap(lap_idx: int = -1) -> bool:
	if completed_laps.is_empty():
		return false
	if lap_idx < 0:
		baseline_lap = completed_laps[completed_laps.size() - 1]
		return true
	for lap in completed_laps:
		if int(lap.get("lap_index", -1)) == lap_idx:
			baseline_lap = lap
			return true
	return false

func get_live_delta_s(s_dist: float, cur_time_s: float) -> float:
	if baseline_lap.is_empty():
		return 0.0
	var d_arr: Array = baseline_lap.get("dist_trace", [])
	var t_arr: Array = baseline_lap.get("time_trace", [])
	if d_arr.size() < 2 or t_arr.size() != d_arr.size():
		return 0.0
	if s_dist <= float(d_arr[0]):
		return cur_time_s - float(t_arr[0])
	var last_i: int = d_arr.size() - 1
	if s_dist >= float(d_arr[last_i]):
		return cur_time_s - float(t_arr[last_i])
	for i in range(1, d_arr.size()):
		var d1: float = float(d_arr[i])
		if d1 >= s_dist:
			var d0: float = float(d_arr[i - 1])
			var frac: float = (s_dist - d0) / maxf(d1 - d0, 1e-4)
			var ref_t: float = lerpf(float(t_arr[i - 1]), float(t_arr[i]), frac)
			return cur_time_s - ref_t
	return 0.0

func export_session_reports() -> void:
	var summary_laps: Array = []
	for lap in completed_laps:
		var clean_lap: Dictionary = lap.duplicate(true)
		clean_lap.erase("dist_trace")
		clean_lap.erase("time_trace")
		summary_laps.append(clean_lap)

	var summary := {
		"session_id": session_id,
		"event_type": event_type,
		"best_lap_time_s": snappedf(best_lap_time_s, 0.001),
		"baseline_lap_index": int(baseline_lap.get("lap_index", -1)),
		"laps": summary_laps
	}
	var f_json := FileAccess.open(session_dir.path_join("session_summary.json"), FileAccess.WRITE)
	if f_json != null:
		f_json.store_string(JSON.stringify(summary, "\t") + "\n")
		f_json.close()

	var md := "# TeR27 DiL Session Report — %s\n\n" % session_id
	md += "- **Event:** %s\n- **Best Lap:** %.3f s\n\n" % [event_type, best_lap_time_s]
	md += "| Lap | Time (s) | Max ay (g) | Max ax (g) | wz RMSE | Pk Beta | RMS Beta | Sat % | kWh/km | T_tire | T_disc | TV Map |\n"
	md += "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
	for lap in summary_laps:
		var snap: Dictionary = lap.get("setup_snapshot", {})
		md += "| %d | %.3f | %.2f | %.2f | %.4f | %.2f° | %.2f° | %.1f%% | %.3f | %.1f°C | %.1f°C | %s |\n" % [
			int(lap["lap_index"]), float(lap["lap_time_s"]),
			float(lap["max_ay_g"]), float(lap["max_ax_g"]),
			float(lap["wz_rmse"]), float(lap["peak_beta_deg"]), float(lap["rms_beta_deg"]),
			float(lap["avg_tire_sat_pct"]), float(lap["energy_kwh_km"]),
			float(lap["max_tire_temp_c"]), float(lap["max_disc_temp_c"]),
			String(snap.get("tv_map_filename", "n/a"))
		]
	var f_md := FileAccess.open(session_dir.path_join("session_report.md"), FileAccess.WRITE)
	if f_md != null:
		f_md.store_string(md)
		f_md.close()
