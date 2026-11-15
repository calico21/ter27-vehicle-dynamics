extends SceneTree
class_name TeR26TelemetryValidator

## TeR26 Open-Loop Telemetry Replay & Correlation Validator
## Usage (CLI):
##   godot --headless --path src/mirena_sim -s res://tools/ter26_telemetry_validator.gd -- --input <log.csv> [--setup default] [--out_csv <cmp.csv>] [--out_json <metrics.json>]

const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")


func _init() -> void:
	# Only execute CLI flow when invoked directly as main script
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		var full_args := OS.get_cmdline_args()
		var idx := full_args.find("--")
		if idx >= 0:
			args = full_args.slice(idx + 1)
	var opts := _parse_cli_args(args)
	if not opts.has("input"):
		quit(0)
		return

	var input_path: String = String(opts["input"])
	var setup_name: String = String(opts.get("setup", "default"))
	var out_csv: String = String(opts.get("out_csv", input_path.get_basename() + "_comparison.csv"))
	var out_json: String = String(opts.get("out_json", ""))

	var result := run_validation_from_file(input_path, "ter26_rwd", setup_name, out_csv, out_json)
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
		if a.begins_with("--") and i + 1 < args.size():
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


## Runs open-loop replay on a parsed TeR26 CSV dataset and returns paired rows + RMSE metrics.
static func run_open_loop_replay(
	rows: Array[Dictionary],
	vehicle_id: String = "ter26_rwd",
	setup_name: String = "default"
) -> Dictionary:
	if rows.size() < 2:
		return {"ok": false, "error": "CSV must contain at least 2 telemetry rows"}

	var model := VehicleDynamicsModelScript.new()
	if not model.load_vehicle(vehicle_id, setup_name):
		return {"ok": false, "error": "Failed to load vehicle profile '%s'" % vehicle_id}

	var v0: float = float(rows[0].get("vx_ms", 0.0))
	model.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	model.chassis_state.angular_velocity = Vector3(0.0, 0.0, float(rows[0].get("wz_rads", 0.0)))
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
	inp.ecu_driving_mode = 1

	# Brief static/rolling settle on sample 0 so suspension compression initializes cleanly
	inp.steer_angle = clampf(deg_to_rad(float(rows[0].get("steer_deg", 0.0))), -model.config.max_steer_angle, model.config.max_steer_angle)
	inp.throttle = clampf(float(rows[0].get("throttle", 0.0)), 0.0, 1.0)
	inp.brake = clampf(float(rows[0].get("brake", 0.0)), 0.0, 1.0)

	for idx in range(rows.size()):
		var cur: Dictionary = rows[idx]
		var t_cur: float = float(cur.get("time_s", float(idx) * 0.01))
		if idx > 0:
			var prev: Dictionary = rows[idx - 1]
			var t_prev: float = float(prev.get("time_s", float(idx - 1) * 0.01))
			var dt_step: float = maxf(t_cur - t_prev, 1e-4)
			var n_sub: int = maxi(1, int(ceili(dt_step / 0.0025)))
			var sub_dt: float = dt_step / float(n_sub)
			inp.steer_angle = clampf(deg_to_rad(float(prev.get("steer_deg", 0.0))), -model.config.max_steer_angle, model.config.max_steer_angle)
			inp.throttle = clampf(float(prev.get("throttle", 0.0)), 0.0, 1.0)
			inp.brake = clampf(float(prev.get("brake", 0.0)), 0.0, 1.0)
			for _s in range(n_sub):
				model.step_standalone(inp, sub_dt)

		var sim_vx: float = model.chassis_state.linear_velocity.x
		var sim_ay: float = model.net_force_chassis_ros.y / maxf(model.config.mass, 1.0)
		var sim_wz: float = model.chassis_state.angular_velocity.z
		var real_vx: float = float(cur.get("vx_ms", 0.0))
		var real_ay: float = float(cur.get("ay_ms2", 0.0))
		var real_wz: float = float(cur.get("wz_rads", 0.0))

		sum_sq_vx += (sim_vx - real_vx) * (sim_vx - real_vx)
		sum_sq_ay += (sim_ay - real_ay) * (sim_ay - real_ay)
		sum_sq_wz += (sim_wz - real_wz) * (sim_wz - real_wz)
		n_samples += 1

		var sim_rpms: Array[float] = [0.0, 0.0, 0.0, 0.0]
		var sim_susps: Array[float] = [0.0, 0.0, 0.0, 0.0]
		for wi in range(4):
			sim_rpms[wi] = model.wheels[wi].omega * 60.0 / TAU
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
			"real_ay_ms2": real_ay,
			"sim_ay_ms2": sim_ay,
			"err_ay_ms2": sim_ay - real_ay,
			"real_wz_rads": real_wz,
			"sim_wz_rads": sim_wz,
			"err_wz_rads": sim_wz - real_wz,
			"real_wheel_rpm_fl": float(cur.get("wheel_rpm_fl", 0.0)),
			"sim_wheel_rpm_fl": sim_rpms[0],
			"real_wheel_rpm_fr": float(cur.get("wheel_rpm_fr", 0.0)),
			"sim_wheel_rpm_fr": sim_rpms[1],
			"real_wheel_rpm_rl": float(cur.get("wheel_rpm_rl", 0.0)),
			"sim_wheel_rpm_rl": sim_rpms[2],
			"real_wheel_rpm_rr": float(cur.get("wheel_rpm_rr", 0.0)),
			"sim_wheel_rpm_rr": sim_rpms[3],
			"real_susp_mm_fl": float(cur.get("susp_mm_fl", 0.0)),
			"sim_susp_mm_fl": sim_susps[0],
			"real_susp_mm_fr": float(cur.get("susp_mm_fr", 0.0)),
			"sim_susp_mm_fr": sim_susps[1],
			"real_susp_mm_rl": float(cur.get("susp_mm_rl", 0.0)),
			"sim_susp_mm_rl": sim_susps[2],
			"real_susp_mm_rr": float(cur.get("susp_mm_rr", 0.0)),
			"sim_susp_mm_rr": sim_susps[3],
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
		"real_ay_ms2", "sim_ay_ms2", "err_ay_ms2",
		"real_wz_rads", "sim_wz_rads", "err_wz_rads",
		"real_wheel_rpm_fl", "sim_wheel_rpm_fl",
		"real_wheel_rpm_fr", "sim_wheel_rpm_fr",
		"real_wheel_rpm_rl", "sim_wheel_rpm_rl",
		"real_wheel_rpm_rr", "sim_wheel_rpm_rr",
		"real_susp_mm_fl", "sim_susp_mm_fl",
		"real_susp_mm_fr", "sim_susp_mm_fr",
		"real_susp_mm_rl", "sim_susp_mm_rl",
		"real_susp_mm_rr", "sim_susp_mm_rr",
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
	out_json: String = ""
) -> Dictionary:
	var rows := load_csv_log(input_csv)
	if rows.size() < 2:
		return {"ok": false, "error": "Unable to load >= 2 CSV rows from " + input_csv}
	var res := run_open_loop_replay(rows, vehicle_id, setup_name)
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
