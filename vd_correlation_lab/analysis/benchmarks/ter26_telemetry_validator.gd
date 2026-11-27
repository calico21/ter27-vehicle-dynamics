extends SceneTree
class_name TeR26TelemetryValidator

## TeR26 Open-Loop Telemetry Replay & Correlation Validator
## Usage (CLI):
##   godot --headless --path src/mirena_sim -s res://tools/ter26_telemetry_validator.gd -- --input <log.csv> [--setup default] [--out_csv <cmp.csv>] [--out_json <metrics.json>]

const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _init() -> void:
	# Only execute CLI flow when invoked directly as main script
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		var full_args := OS.get_cmdline_args()
		var idx := full_args.find("--")
		if idx >= 0:
			args = full_args.slice(idx + 1)
	var opts := _parse_cli_args(args)
	if opts.has("batch"):
		var batch_ok := _run_batch_file(String(opts["batch"]))
		quit(0 if batch_ok else 1)
		return
	if not opts.has("input"):
		quit(0)
		return

	var input_path: String = String(opts["input"])
	var setup_name: String = String(opts.get("setup", "default"))
	var out_csv: String = String(opts.get("out_csv", input_path.get_basename() + "_comparison.csv"))
	var out_json: String = String(opts.get("out_json", ""))
	var override_json: String = String(opts.get("override_json", ""))
	var ecu_mode: int = int(opts.get("ecu_mode", 0))
	var trq_limit_nm: float = float(opts.get("trq_limit", 0.0))
	var torque_source: String = String(opts.get("torque_source", "pedal")).to_lower()

	var overrides: Dictionary = {}
	if override_json != "" and FileAccess.file_exists(override_json):
		var jf := FileAccess.open(override_json, FileAccess.READ)
		if jf != null:
			var parsed = JSON.parse_string(jf.get_as_text())
			jf.close()
			if parsed is Dictionary:
				overrides = parsed

	var result := run_validation_from_file(input_path, "ter26_rwd", setup_name, out_csv, out_json, overrides, ecu_mode, trq_limit_nm, torque_source)
	if not bool(result.get("ok", false)):
		push_error("TeR26TelemetryValidator failed: " + String(result.get("error", "unknown error")))
		quit(1)
		return

	var rmse: Dictionary = result.get("rmse", {})
	print("============================================================")
	print("  TeR26 OPEN-LOOP TELEMETRY VALIDATION SUMMARY")
	print("  Samples: %d | Vehicle: ter26_rwd | Setup: %s" % [int(result.get("sample_count", 0)), setup_name])
	print("  RMSE vx: %.4f m/s | RMSE ay: %.4f m/s^2 | RMSE wz: %.4f rad/s" % [
		float(rmse.get("vx_ms", 0.0)),
		float(rmse.get("ay_ms2", 0.0)),
		float(rmse.get("wz_rads", 0.0)),
	])
	print("  Comparison CSV: %s" % out_csv)
	print("============================================================")
	quit(0)


static func _parse_cli_args(args: PackedStringArray) -> Dictionary:
	var out: Dictionary = {}
	var i := 0
	while i < args.size():
		var a := args[i]
		if a.begins_with("--") and a.contains("="):
			var eq := a.find("=")
			out[a.substr(2, eq - 2)] = a.substr(eq + 1)
			i += 1
		elif a.begins_with("--") and i + 1 < args.size():
			out[a.substr(2)] = args[i + 1]
			i += 2
		else:
			i += 1
	return out


## Parses a TeR26 track log CSV into an Array of Dictionaries keyed by column name.
static func load_csv_log(csv_path: String) -> Array[Dictionary]:
	var rows: Array[Dictionary] = []
	if not FileAccess.file_exists(csv_path):
		return rows
	var f := FileAccess.open(csv_path, FileAccess.READ)
	if f == null:
		return rows
	var header_line := f.get_line().strip_edges()
	if header_line == "":
		f.close()
		return rows
	var headers := header_line.split(",")
	for idx in range(headers.size()):
		headers[idx] = headers[idx].strip_edges().to_lower()
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line == "" or line.begins_with("#"):
			continue
		var parts := line.split(",")
		if parts.size() < 4:
			continue
		var row: Dictionary = {}
		for j in range(mini(headers.size(), parts.size())):
			var key := headers[j]
			row[key] = float(parts[j].strip_edges())
		rows.append(row)
	f.close()
	return rows


## Applies a Dictionary of config overrides in memory (for parallel headless calibration).
static func _apply_config_overrides(model: VehicleDynamicsModel, overrides: Dictionary) -> void:
	if overrides.is_empty() or model == null or model.config == null:
		return
	var cfg := model.config
	for k in overrides.keys():
		var key := String(k)
		if key == "front_weight_distribution":
			cfg.front_weight_distribution = clampf(float(overrides[k]), 0.1, 0.9)
			cfg.weight_dist_r = 1.0 - cfg.front_weight_distribution
		elif key == "regen_enable" or key == "regen_mode":
			cfg.set(key, int(overrides[k]))
		else:
			cfg.set(key, float(overrides[k]))
	cfg.update_derived_geometry()
	if model.suspension != null:
		model.suspension.sync_from_config(cfg)
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.configure_from_vehicle_config(cfg)
	model.reset()


## Runs open-loop replay on a parsed TeR26 CSV dataset and returns paired rows + RMSE metrics.
static func run_open_loop_replay(
	rows: Array[Dictionary],
	vehicle_id: String = "ter26_rwd",
	setup_name: String = "default",
	overrides: Dictionary = {},
	ecu_mode: int = 1,
	trq_limit_nm: float = 0.0,
	torque_source: String = "pedal"
) -> Dictionary:
	if rows.size() < 2:
		return {"ok": false, "error": "CSV must contain at least 2 telemetry rows"}

	var model := VehicleDynamicsModelScript.new()
	if not model.load_vehicle(vehicle_id, setup_name):
		return {"ok": false, "error": "Failed to load vehicle profile '%s'" % vehicle_id}
	_apply_config_overrides(model, overrides)
	if ecu_mode == 0 and trq_limit_nm > 0.0:
		model.config.motor_peak_torque = minf(model.config.motor_peak_torque, trq_limit_nm * 0.5)

	var v0: float = float(rows[0].get("vx_init", rows[0].get("vx_ms", 0.0)))
	var wz0: float = float(rows[0].get("wz_init", rows[0].get("wz_rads", 0.0)))
	model.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	model.chassis_state.angular_velocity = Vector3(0.0, 0.0, wz0)
	var w0_rad_s: float = v0 / maxf(model.config.wheel_radius, 0.05)
	for i in range(4):
		model.wheels[i].omega = w0_rad_s
		if model.ecu_tv_v3 != null:
			model.ecu_tv_v3.tc_omega_prev_ema[i] = w0_rad_s

	var paired_rows: Array[Dictionary] = []
	var sum_sq_vx: float = 0.0
	var sum_sq_ay: float = 0.0
	var sum_sq_wz: float = 0.0
	var sum_sq_rpm: float = 0.0
	var n_samples: int = 0
	var n_rpm: int = 0

	var inp := VehicleInputsScript.new()
	inp.drive_mode = model.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	var use_inv_trq: bool = (torque_source == "inverter")
	inp.override_wheel_torques = use_inv_trq
	if int(rows[0].get("regen_enable", 0)) == 1 and not overrides.has("regen_enable"):
		model.config.regen_enable = 1
		model.config.regen_max_trq = float(rows[0].get("regen_max_trq", model.config.regen_max_trq))
		model.config.regen_thr_speed = float(rows[0].get("regen_thr_speed", model.config.regen_thr_speed))
		model.config.regen_thr_rpm = float(rows[0].get("regen_thr_rpm", model.config.regen_thr_rpm))
		model.config.regen_trq_slope = float(rows[0].get("regen_trq_slope", model.config.regen_trq_slope))
		model.config.regen_mode = int(rows[0].get("regen_mode", model.config.regen_mode))
		model.config.regen_max_positive_trq_thr = float(rows[0].get("regen_max_positive_trq_thr", model.config.regen_max_positive_trq_thr)) / 100.0

	# Brief static/rolling settle on sample 0 so suspension compression initializes cleanly
	inp.steer_angle = clampf(deg_to_rad(float(rows[0].get("steer_deg", 0.0))), -model.config.max_steer_angle, model.config.max_steer_angle)
	inp.throttle = clampf(float(rows[0].get("throttle", 0.0)), 0.0, 1.0)
	inp.brake = clampf(float(rows[0].get("brake", 0.0)), 0.0, 1.0)

	for idx in range(rows.size()):
		var cur: Dictionary = rows[idx]
		var t_cur: float = float(cur.get("time_s", float(idx) * 0.01))
		var step_res: Dictionary = {}
		if idx == 0:
			step_res = model.step_standalone(inp, 0.0)
		else:
			var prev: Dictionary = rows[idx - 1]
			var t_prev: float = float(prev.get("time_s", float(idx - 1) * 0.01))
			var dt_step: float = maxf(t_cur - t_prev, 1e-4)
			var n_sub: int = maxi(1, int(ceili(dt_step / 0.0025)))
			var sub_dt: float = dt_step / float(n_sub)
			inp.steer_angle = clampf(deg_to_rad(float(prev.get("steer_deg", 0.0))), -model.config.max_steer_angle, model.config.max_steer_angle)
			inp.throttle = clampf(float(prev.get("throttle", 0.0)), 0.0, 1.0)
			inp.brake = clampf(float(prev.get("brake", 0.0)), 0.0, 1.0)
			if use_inv_trq:
				var trq_l: float = float(prev.get("real_trq_rl", prev.get("inv_left_dem", 0.0)))
				var trq_r: float = float(prev.get("real_trq_rr", prev.get("inv_right_dem", 0.0)))
				var scale_w: float = model.config.gear_ratio * model.config.drivetrain_efficiency
				inp.wheel_motor_torques = [0.0, 0.0, trq_l * scale_w, trq_r * scale_w]
			for _s in range(n_sub):
				step_res = model.step_standalone(inp, sub_dt)

		var forces: Dictionary = step_res.get("forces", {})
		var drive_trqs: Array = forces.get("drive_torques", [0.0, 0.0, 0.0, 0.0])
		var normal_forces: Array = forces.get("normal_forces", step_res.get("normal_forces", [0.0, 0.0, 0.0, 0.0]))
		var sim_vx: float = model.chassis_state.linear_velocity.x
		var sim_vy: float = model.chassis_state.linear_velocity.y
		var sim_ax: float = model.net_force_chassis_ros.x / maxf(model.config.mass, 1.0)
		var sim_ay: float = model.net_force_chassis_ros.y / maxf(model.config.mass, 1.0)
		var sim_wz: float = model.chassis_state.angular_velocity.z
		var sim_beta_deg: float = rad_to_deg(atan2(sim_vy, maxf(absf(sim_vx), 0.5))) if absf(sim_vx) > 0.25 else 0.0
		var sim_roll_deg: float = rad_to_deg(model.chassis_state.roll)
		var sim_pitch_deg: float = rad_to_deg(model.chassis_state.pitch)
		var real_vx: float = float(cur.get("vx_ms", 0.0))
		var real_ay: float = float(cur.get("ay_ms2", 0.0))
		var real_wz: float = float(cur.get("wz_rads", 0.0))

		sum_sq_vx += (sim_vx - real_vx) * (sim_vx - real_vx)
		sum_sq_ay += (sim_ay - real_ay) * (sim_ay - real_ay)
		sum_sq_wz += (sim_wz - real_wz) * (sim_wz - real_wz)
		n_samples += 1

		var sim_rpms: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_susps: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_kappas: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_alphas_deg: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_fzs: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_tws: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_t_tire: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_t_in: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_t_mid: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_t_out: Array[float] = [0.0, 0.0, 0.0, 0.0]
		for wi in range(4):
			var w = model.wheels[wi]
			sim_rpms[wi] = w.omega * 60.0 / TAU
			sim_kappas[wi] = w.kappa
			sim_alphas_deg[wi] = rad_to_deg(w.alpha)
			sim_fzs[wi] = float(normal_forces[wi]) if wi < normal_forces.size() else w.F_z
			sim_tws[wi] = float(drive_trqs[wi]) if wi < drive_trqs.size() else 0.0
			sim_t_tire[wi] = w.t_surf_c
			sim_t_in[wi] = w.t_zone_inner_c
			sim_t_mid[wi] = w.t_zone_mid_c
			sim_t_out[wi] = w.t_zone_outer_c
			if model.suspension != null and wi < model.suspension.corners.size():
				sim_susps[wi] = model.suspension.corners[wi].delta_z * 1000.0

		if cur.has("wheel_rpm_rl") and cur.has("wheel_rpm_rr"):
			var dr_rl: float = sim_rpms[2] - float(cur.get("wheel_rpm_rl", sim_rpms[2]))
			var dr_rr: float = sim_rpms[3] - float(cur.get("wheel_rpm_rr", sim_rpms[3]))
			sum_sq_rpm += 0.5 * (dr_rl * dr_rl + dr_rr * dr_rr)
			n_rpm += 1

		paired_rows.append({
			"time_s": t_cur,
			"steer_deg": float(cur.get("steer_deg", 0.0)),
			"throttle": float(cur.get("throttle", 0.0)),
			"brake": float(cur.get("brake", 0.0)),
			"real_vx_ms": real_vx,
			"sim_vx_ms": sim_vx,
			"err_vx_ms": sim_vx - real_vx,
			"sim_vy_ms": sim_vy,
			"sim_ax_ms2": sim_ax,
			"real_ay_ms2": real_ay,
			"sim_ay_ms2": sim_ay,
			"err_ay_ms2": sim_ay - real_ay,
			"real_wz_rads": real_wz,
			"sim_wz_rads": sim_wz,
			"err_wz_rads": sim_wz - real_wz,
			"sim_beta_deg": sim_beta_deg,
			"sim_roll_deg": sim_roll_deg,
			"sim_pitch_deg": sim_pitch_deg,
			"sim_pos_x_m": model.chassis_state.position.x,
			"sim_pos_y_m": model.chassis_state.position.y,
			"sim_yaw_rad": model.chassis_state.yaw,
			"real_wheel_rpm_fl": float(cur.get("wheel_rpm_fl", 0.0)),
			"sim_wheel_rpm_fl": sim_rpms[0],
			"real_wheel_rpm_fr": float(cur.get("wheel_rpm_fr", 0.0)),
			"sim_wheel_rpm_fr": sim_rpms[1],
			"real_wheel_rpm_rl": float(cur.get("wheel_rpm_rl", 0.0)),
			"sim_wheel_rpm_rl": sim_rpms[2],
			"real_wheel_rpm_rr": float(cur.get("wheel_rpm_rr", 0.0)),
			"sim_wheel_rpm_rr": sim_rpms[3],
			"sim_kappa_fl": sim_kappas[0],
			"sim_kappa_fr": sim_kappas[1],
			"sim_kappa_rl": sim_kappas[2],
			"sim_kappa_rr": sim_kappas[3],
			"sim_alpha_deg_fl": sim_alphas_deg[0],
			"sim_alpha_deg_fr": sim_alphas_deg[1],
			"sim_alpha_deg_rl": sim_alphas_deg[2],
			"sim_alpha_deg_rr": sim_alphas_deg[3],
			"sim_fz_fl": sim_fzs[0],
			"sim_fz_fr": sim_fzs[1],
			"sim_fz_rl": sim_fzs[2],
			"sim_fz_rr": sim_fzs[3],
			"real_susp_mm_fl": float(cur.get("susp_mm_fl", 0.0)),
			"sim_susp_mm_fl": sim_susps[0],
			"real_susp_mm_fr": float(cur.get("susp_mm_fr", 0.0)),
			"sim_susp_mm_fr": sim_susps[1],
			"real_susp_mm_rl": float(cur.get("susp_mm_rl", 0.0)),
			"sim_susp_mm_rl": sim_susps[2],
			"real_susp_mm_rr": float(cur.get("susp_mm_rr", 0.0)),
			"sim_susp_mm_rr": sim_susps[3],
			"sim_t_wheel_fl": sim_tws[0],
			"sim_t_wheel_fr": sim_tws[1],
			"sim_t_wheel_rl": sim_tws[2],
			"sim_t_wheel_rr": sim_tws[3],
			"sim_mz_tv": float(forces.get("mz_req", 0.0)),
			"sim_wz_ref_rads": float(forces.get("wz_ref", 0.0)),
			"sim_power_kw": float(forces.get("power_kw", 0.0)),
			"sim_t_tire_fl": sim_t_tire[0],
			"sim_t_tire_fr": sim_t_tire[1],
			"sim_t_tire_rl": sim_t_tire[2],
			"sim_t_tire_rr": sim_t_tire[3],
			"sim_t_tire_inner_fl": sim_t_in[0],
			"sim_t_tire_mid_fl": sim_t_mid[0],
			"sim_t_tire_outer_fl": sim_t_out[0],
			"sim_t_tire_inner_fr": sim_t_in[1],
			"sim_t_tire_mid_fr": sim_t_mid[1],
			"sim_t_tire_outer_fr": sim_t_out[1],
			"sim_t_tire_inner_rl": sim_t_in[2],
			"sim_t_tire_mid_rl": sim_t_mid[2],
			"sim_t_tire_outer_rl": sim_t_out[2],
			"sim_t_tire_inner_rr": sim_t_in[3],
			"sim_t_tire_mid_rr": sim_t_mid[3],
			"sim_t_tire_outer_rr": sim_t_out[3],
		})

	var inv_n: float = 1.0 / float(maxi(n_samples, 1))
	var rmse := {
		"vx_ms": sqrt(sum_sq_vx * inv_n),
		"ay_ms2": sqrt(sum_sq_ay * inv_n),
		"wz_rads": sqrt(sum_sq_wz * inv_n),
		"wheel_rpm_rear": sqrt(sum_sq_rpm / float(n_rpm)) if n_rpm > 0 else 0.0,
	}
	return {
		"ok": true,
		"vehicle_id": vehicle_id,
		"setup_name": setup_name,
		"sample_count": n_samples,
		"rmse": rmse,
		"rows": paired_rows,
	}


## Writes the side-by-side real vs simulated telemetry comparison CSV.
static func export_comparison_csv(paired_rows: Array[Dictionary], out_csv_path: String) -> bool:
	var cols: Array[String] = [
		"time_s", "steer_deg", "throttle", "brake",
		"real_vx_ms", "sim_vx_ms", "err_vx_ms",
		"sim_vy_ms", "sim_ax_ms2",
		"real_ay_ms2", "sim_ay_ms2", "err_ay_ms2",
		"real_wz_rads", "sim_wz_rads", "err_wz_rads",
		"sim_beta_deg", "sim_roll_deg", "sim_pitch_deg",
		"sim_pos_x_m", "sim_pos_y_m", "sim_yaw_rad",
		"real_wheel_rpm_fl", "sim_wheel_rpm_fl",
		"real_wheel_rpm_fr", "sim_wheel_rpm_fr",
		"real_wheel_rpm_rl", "sim_wheel_rpm_rl",
		"real_wheel_rpm_rr", "sim_wheel_rpm_rr",
		"sim_kappa_fl", "sim_kappa_fr", "sim_kappa_rl", "sim_kappa_rr",
		"sim_alpha_deg_fl", "sim_alpha_deg_fr", "sim_alpha_deg_rl", "sim_alpha_deg_rr",
		"sim_fz_fl", "sim_fz_fr", "sim_fz_rl", "sim_fz_rr",
		"real_susp_mm_fl", "sim_susp_mm_fl",
		"real_susp_mm_fr", "sim_susp_mm_fr",
		"real_susp_mm_rl", "sim_susp_mm_rl",
		"real_susp_mm_rr", "sim_susp_mm_rr",
		"sim_t_wheel_fl", "sim_t_wheel_fr", "sim_t_wheel_rl", "sim_t_wheel_rr",
		"sim_mz_tv", "sim_wz_ref_rads", "sim_power_kw",
		"sim_t_tire_fl", "sim_t_tire_fr", "sim_t_tire_rl", "sim_t_tire_rr",
		"sim_t_tire_inner_fl", "sim_t_tire_mid_fl", "sim_t_tire_outer_fl",
		"sim_t_tire_inner_fr", "sim_t_tire_mid_fr", "sim_t_tire_outer_fr",
		"sim_t_tire_inner_rl", "sim_t_tire_mid_rl", "sim_t_tire_outer_rl",
		"sim_t_tire_inner_rr", "sim_t_tire_mid_rr", "sim_t_tire_outer_rr",
	]
	var f := FileAccess.open(out_csv_path, FileAccess.WRITE)
	if f == null:
		return false
	f.store_line(",".join(PackedStringArray(cols)))
	for r in paired_rows:
		var vals: PackedStringArray = []
		for c in cols:
			vals.append("%.6f" % float(r.get(c, 0.0)))
		f.store_line(",".join(vals))
	f.close()
	return true


## Convenience end-to-end helper: loads CSV, replays open-loop, writes comparison CSV and optional JSON metrics.
static func run_validation_from_file(
	input_csv: String,
	vehicle_id: String = "ter26_rwd",
	setup_name: String = "default",
	out_csv: String = "",
	out_json: String = "",
	overrides: Dictionary = {},
	ecu_mode: int = 1,
	trq_limit_nm: float = 0.0,
	torque_source: String = "pedal"
) -> Dictionary:
	var rows := load_csv_log(input_csv)
	if rows.size() < 2:
		return {"ok": false, "error": "Unable to load >= 2 CSV rows from " + input_csv}
	var res := run_open_loop_replay(rows, vehicle_id, setup_name, overrides, ecu_mode, trq_limit_nm, torque_source)
	if not bool(res.get("ok", false)):
		return res
	if out_csv != "":
		res["csv_exported"] = export_comparison_csv(res["rows"], out_csv)
		res["out_csv"] = out_csv
	if out_json != "":
		var summary := {
			"vehicle_id": vehicle_id,
			"setup_name": setup_name,
			"sample_count": res["sample_count"],
			"rmse": res["rmse"],
		}
		var jf := FileAccess.open(out_json, FileAccess.WRITE)
		if jf != null:
			jf.store_string(JSON.stringify(summary, "\t") + "\n")
			jf.close()
			res["json_exported"] = true
	return res


## Batch mode: instantiates VehicleDynamicsModel once and evaluates all candidate param dicts in memory.
static func _run_batch_file(spec_path: String) -> bool:
	if not FileAccess.file_exists(spec_path):
		push_error("Batch spec not found: " + spec_path)
		return false
	var sf := FileAccess.open(spec_path, FileAccess.READ)
	if sf == null:
		return false
	var parsed = JSON.parse_string(sf.get_as_text())
	sf.close()
	if not (parsed is Dictionary):
		push_error("Invalid batch JSON: " + spec_path)
		return false
	var spec: Dictionary = parsed
	var vehicle_id: String = String(spec.get("vehicle_id", "ter26_rwd"))
	var setup_name: String = String(spec.get("setup_name", "default"))
	var ecu_mode: int = int(spec.get("ecu_mode", 0))
	var out_json: String = String(spec.get("out_json", spec_path.get_basename() + "_results.json"))
	var max_sub_dt: float = clampf(float(spec.get("sub_dt", 0.004)), 0.001, 0.005)

	var win_entries: Array = spec.get("windows", [])
	if win_entries.is_empty() and spec.has("input_csv"):
		win_entries = [String(spec["input_csv"])]
	var loaded_windows: Array = []
	for wp in win_entries:
		var wp_path: String = String(wp["csv"]) if (wp is Dictionary) else String(wp)
		var w_ecu: int = int(wp.get("ecu_mode", ecu_mode)) if (wp is Dictionary) else ecu_mode
		var w_trq: float = float(wp.get("trq_limit_nm", spec.get("trq_limit_nm", 0.0))) if (wp is Dictionary) else float(spec.get("trq_limit_nm", 0.0))
		var r := load_csv_log(wp_path)
		if r.size() >= 2:
			loaded_windows.append({"rows": r, "ecu_mode": w_ecu, "trq_limit_nm": w_trq})
	if loaded_windows.is_empty():
		push_error("No valid CSV windows in batch spec")
		return false

	var model := VehicleDynamicsModelScript.new()
	if not model.load_vehicle(vehicle_id, setup_name):
		push_error("Failed to load vehicle: " + vehicle_id)
		return false

	var candidates: Array = spec.get("candidates", [])
	var out_results: Array = []
	var inp := VehicleInputsScript.new()
	inp.drive_mode = model.config.drive_mode
	inp.ecu_driving_mode = ecu_mode
	var torque_source: String = String(spec.get("torque_source", "pedal")).to_lower()
	inp.override_wheel_torques = (torque_source == "inverter")

	for cand in candidates:
		if not (cand is Dictionary):
			continue
		var cid = cand.get("id", 0)
		var params: Dictionary = cand.get("params", {})
		_apply_config_overrides(model, params)
		var metrics := _eval_candidate_fast(model, inp, loaded_windows, max_sub_dt)
		metrics["id"] = cid
		out_results.append(metrics)

	var out_payload := {"ok": true, "results": out_results}
	var of := FileAccess.open(out_json, FileAccess.WRITE)
	if of == null:
		return false
	of.store_string(JSON.stringify(out_payload))
	of.close()
	return true


static func _eval_candidate_fast(
	model: VehicleDynamicsModel,
	inp: VehicleInputs,
	loaded_windows: Array,
	max_sub_dt: float = 0.004
) -> Dictionary:
	var sum_sq_vx := 0.0
	var sum_sq_ax := 0.0
	var sum_sq_ay := 0.0
	var sum_sq_wz := 0.0
	var sum_sq_rpm := 0.0
	var sum_sq_susp := 0.0
	var sum_sq_roll := 0.0
	var sum_sq_pitch := 0.0
	var n_samples := 0
	var n_ax := 0
	var n_rpm := 0
	var n_susp := 0
	var n_rp := 0

	# Online linear regression accumulators for K_us (understeer gradient deg/g)
	var kus_n := 0
	var kus_sx := 0.0
	var kus_sy := 0.0
	var kus_sxx := 0.0
	var kus_sxy := 0.0

	var max_steer := model.config.max_steer_angle
	var mass := maxf(model.config.mass, 1.0)
	var wb := maxf(model.config.wheelbase, 0.5)
	var r_whl := maxf(model.config.wheel_radius, 0.05)

	var base_motor_peak: float = model.config.motor_peak_torque
	for win_obj in loaded_windows:
		var rows: Array = win_obj["rows"]
		var w_ecu: int = int(win_obj.get("ecu_mode", 0))
		var w_trq: float = float(win_obj.get("trq_limit_nm", 0.0))
		inp.ecu_driving_mode = w_ecu
		if w_ecu == 0 and w_trq > 0.0:
			model.config.motor_peak_torque = minf(base_motor_peak, w_trq * 0.5)
		else:
			model.config.motor_peak_torque = base_motor_peak
		model.reset()
		var v0 := float(rows[0].get("vx_init", rows[0].get("vx_ms", 0.0)))
		var wz0 := float(rows[0].get("wz_init", rows[0].get("wz_rads", 0.0)))
		model.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
		model.chassis_state.angular_velocity = Vector3(0.0, 0.0, wz0)
		var w0 := v0 / r_whl
		for i in range(4):
			model.wheels[i].omega = w0
			if model.ecu_tv_v3 != null:
				model.ecu_tv_v3.tc_omega_prev_ema[i] = w0

		inp.steer_angle = clampf(deg_to_rad(float(rows[0].get("steer_deg", 0.0))), -max_steer, max_steer)
		inp.throttle = clampf(float(rows[0].get("throttle", 0.0)), 0.0, 1.0)
		inp.brake = clampf(float(rows[0].get("brake", 0.0)), 0.0, 1.0)

		for idx in range(rows.size()):
			var cur: Dictionary = rows[idx]
			var t_cur := float(cur.get("time_s", float(idx) * 0.02))
			if idx == 0:
				model.step_standalone(inp, 0.0)
			else:
				var prev: Dictionary = rows[idx - 1]
				var t_prev := float(prev.get("time_s", float(idx - 1) * 0.02))
				var dt_step := maxf(t_cur - t_prev, 1e-4)
				var n_sub := maxi(1, int(ceili(dt_step / max_sub_dt)))
				var sub_dt := dt_step / float(n_sub)
				inp.steer_angle = clampf(deg_to_rad(float(prev.get("steer_deg", 0.0))), -max_steer, max_steer)
				inp.throttle = clampf(float(prev.get("throttle", 0.0)), 0.0, 1.0)
				inp.brake = clampf(float(prev.get("brake", 0.0)), 0.0, 1.0)
				for _s in range(n_sub):
					model.step_standalone(inp, sub_dt)

			var svx := model.chassis_state.linear_velocity.x
			var sax := model.net_force_chassis_ros.x / mass
			var say := model.net_force_chassis_ros.y / mass
			var swz := model.chassis_state.angular_velocity.z

			var dvx := svx - float(cur.get("vx_ms", 0.0))
			var day := say - float(cur.get("ay_ms2", 0.0))
			var dwz := swz - float(cur.get("wz_rads", 0.0))
			sum_sq_vx += dvx * dvx
			sum_sq_ay += day * day
			sum_sq_wz += dwz * dwz
			n_samples += 1

			if cur.has("ax_ms2"):
				var dax := sax - float(cur["ax_ms2"])
				sum_sq_ax += dax * dax
				n_ax += 1

			if cur.has("wheel_rpm_rl") and cur.has("wheel_rpm_rr"):
				var s_rpm := 0.5 * (model.wheels[2].omega + model.wheels[3].omega) * 60.0 / TAU
				var r_rpm := 0.5 * (float(cur["wheel_rpm_rl"]) + float(cur["wheel_rpm_rr"]))
				var drpm := s_rpm - r_rpm
				sum_sq_rpm += drpm * drpm
				n_rpm += 1

			if cur.has("has_susp") and float(cur["has_susp"]) > 0.5 and model.suspension != null:
				var s_fl := model.suspension.corners[0].delta_z * 1000.0 - float(cur.get("susp_mm_fl", 0.0))
				var s_fr := model.suspension.corners[1].delta_z * 1000.0 - float(cur.get("susp_mm_fr", 0.0))
				var s_rl := model.suspension.corners[2].delta_z * 1000.0 - float(cur.get("susp_mm_rl", 0.0))
				var s_rr := model.suspension.corners[3].delta_z * 1000.0 - float(cur.get("susp_mm_rr", 0.0))
				sum_sq_susp += 0.25 * (s_fl * s_fl + s_fr * s_fr + s_rl * s_rl + s_rr * s_rr)
				n_susp += 1

			if cur.has("roll_deg") and cur.has("pitch_deg"):
				var dr := rad_to_deg(model.chassis_state.roll) - float(cur["roll_deg"])
				var dp := rad_to_deg(model.chassis_state.pitch) - float(cur["pitch_deg"])
				sum_sq_roll += dr * dr
				sum_sq_pitch += dp * dp
				n_rp += 1

			if svx > 4.0:
				var ay_g := say / 9.80665
				var us := float(cur.get("steer_deg", 0.0)) - rad_to_deg(wb * swz / svx)
				kus_n += 1
				kus_sx += ay_g
				kus_sy += us
				kus_sxx += ay_g * ay_g
				kus_sxy += ay_g * us

	model.config.motor_peak_torque = base_motor_peak
	var kus_sim := 0.0
	var kus_valid := false
	if kus_n >= 25:
		var denom := float(kus_n) * kus_sxx - kus_sx * kus_sx
		if denom > 1e-6:
			kus_sim = (float(kus_n) * kus_sxy - kus_sx * kus_sy) / denom
			kus_valid = true

	var inv_n := 1.0 / float(maxi(n_samples, 1))
	return {
		"sample_count": n_samples,
		"rmse": {
			"vx_ms": sqrt(sum_sq_vx * inv_n),
			"ax_ms2": sqrt(sum_sq_ax / float(n_ax)) if n_ax > 0 else 0.0,
			"ay_ms2": sqrt(sum_sq_ay * inv_n),
			"wz_rads": sqrt(sum_sq_wz * inv_n),
			"rpm_rear": sqrt(sum_sq_rpm / float(n_rpm)) if n_rpm > 0 else 0.0,
			"susp_mm": sqrt(sum_sq_susp / float(n_susp)) if n_susp > 0 else 0.0,
			"roll_deg": sqrt(sum_sq_roll / float(n_rp)) if n_rp > 0 else 0.0,
			"pitch_deg": sqrt(sum_sq_pitch / float(n_rp)) if n_rp > 0 else 0.0,
		},
		"kus_sim": kus_sim,
		"kus_valid": kus_valid,
	}
