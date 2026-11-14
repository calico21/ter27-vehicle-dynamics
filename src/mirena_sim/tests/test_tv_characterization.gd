extends SceneTree

## Automated TV Characterization Bench, Closed-Loop Track Driver & A/B Test Suite.
## Evaluates all 4 TV maps (OPEN_4WD, ENDURANCE_SAFE, AUTOCROSS_NEUTRAL, SKIDPAD_ATTACK)
## across all TrackFiles/*.json circuits and ISO maneuvers (Step Steer, Slalom, Braking-in-Turn).
## Supports headless parameter optimization CLI:
##   godot --headless --path src/mirena_sim -s res://tests/test_tv_characterization.gd -- \
##     --eval-params k_us=0.015 max_mz=500.0 tc_kp=23.113 fx_res=0.08 --tracks=skidpad,acceleration

const VehicleConfigScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const EcuTV3Script = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoringV3.gd")
const VehicleTelemetryScript = preload("res://Scenes/Vehicle/Dynamics/Telemetry/VehicleTelemetry.gd")
const TelemetryRecorderScript = preload("res://Scenes/Vehicle/Dynamics/Telemetry/TelemetryRecorder.gd")

const TRACKS_DIR: String = "res://TrackFiles/"
const SIM_DT: float = 0.0025  # 400 Hz physics step (matches project.godot)
const ACCEL_TARGET_DIST_M: float = 75.0
const SRR_GAP_DEG: float = 2.0

const MAP_IDS: Array[int] = [0, 1, 2, 3]
const MAP_NAMES: Dictionary = {
	0: "OPEN_4WD",
	1: "ENDURANCE_SAFE",
	2: "AUTOCROSS_NEUTRAL",
	3: "SKIDPAD_ATTACK"
}

func _init() -> void:
	var cli: Dictionary = _parse_cli_args()
	if bool(cli.get("eval_params", false)):
		var eval_out: Dictionary = _run_eval_params(cli)
		print("TV_OPT_RESULT:" + JSON.stringify(eval_out))
		quit(0 if bool(eval_out.get("ok", false)) else 1)
		return

	var res: Dictionary = run(true)
	var failed: int = int(res.get("failed", 1))
	if failed == 0:
		print(">> TV CHARACTERIZATION SUITE PASSED! <<\n")
		quit(0)
	else:
		for err in res.get("errors", []):
			print("  -> ERROR: %s" % str(err))
		print(">> TV CHARACTERIZATION SUITE FAILED! <<\n")
		quit(1)

static func _parse_cli_args() -> Dictionary:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	if args.is_empty():
		args = OS.get_cmdline_args()

	var out: Dictionary = {
		"eval_params": false,
		"params": {},
		"tracks": [],
		"map_id": 2
	}
	for raw in args:
		var tok: String = str(raw).strip_edges()
		if tok == "--eval-params":
			out["eval_params"] = true
			continue
		if tok.begins_with("--eval-params="):
			out["eval_params"] = true
			tok = tok.substr("--eval-params=".length())
		if tok.begins_with("--tracks="):
			var t_str: String = tok.substr("--tracks=".length())
			var t_arr: Array[String] = []
			for part in t_str.split(",", false):
				var cleaned: String = part.strip_edges().to_lower()
				if cleaned != "":
					t_arr.append(cleaned)
			out["tracks"] = t_arr
			continue
		if tok.begins_with("--map="):
			out["map_id"] = int(tok.substr("--map=".length()))
			continue

		# Parse key=val or comma-separated key=val pairs (strip leading -- if present)
		var clean_tok: String = tok.trim_prefix("--")
		for sub in clean_tok.split(",", false):
			if sub.contains("="):
				var kv: PackedStringArray = sub.split("=", true, 1)
				var k: String = kv[0].strip_edges().to_lower()
				var v: String = kv[1].strip_edges()
				match k:
					"k_us", "gp_k_us":
						out["params"]["k_us"] = float(v)
					"max_mz", "gp_tv_max_mz":
						out["params"]["max_mz"] = float(v)
					"tc_kp", "gp_tc_kp":
						out["params"]["tc_kp"] = float(v)
					"fx_res", "min_fx_reserve":
						out["params"]["fx_res"] = float(v)
					"tracks":
						var t_arr2: Array[String] = []
						for part2 in v.split("+", false):
							if part2.strip_edges() != "":
								t_arr2.append(part2.strip_edges().to_lower())
						out["tracks"] = t_arr2
					"map", "map_id":
						out["map_id"] = int(v)
	return out

static func _run_eval_params(cli: Dictionary) -> Dictionary:
	var cfg: VehicleConfigScript = VehicleConfigScript.new()
	var all_tracks: Array[Dictionary] = _discover_tracks()
	var requested: Array = cli.get("tracks", [])
	var selected_tracks: Array[Dictionary] = []

	if requested.is_empty():
		selected_tracks = all_tracks
	else:
		for trk in all_tracks:
			var tid: String = String(trk["id"]).to_lower()
			for req in requested:
				if tid == String(req) or tid.contains(String(req)):
					selected_tracks.append(trk)
					break

	var params: Dictionary = cli.get("params", {})
	var base_map: int = int(cli.get("map_id", 2))
	var track_results: Dictionary = {}
	var ok_all: bool = not selected_tracks.is_empty()

	for trk in selected_tracks:
		var m_res: Dictionary = _run_track_benchmark(cfg, trk, base_map, params)
		track_results[String(trk["id"])] = m_res
		if not bool(m_res.get("completed", false)) or not bool(m_res.get("finite_ok", false)):
			ok_all = false

	var agg: Dictionary = _aggregate_metrics(track_results)
	return {
		"ok": ok_all,
		"params": {
			"k_us": float(params.get("k_us", 0.015)),
			"max_mz": float(params.get("max_mz", 500.0)),
			"tc_kp": float(params.get("tc_kp", 23.113)),
			"fx_res": float(params.get("fx_res", 0.08))
		},
		"metrics": agg,
		"tracks": track_results
	}

static func run(verbose: bool = false) -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var cfg: VehicleConfigScript = VehicleConfigScript.new()
	var tracks: Array[Dictionary] = _discover_tracks()

	if verbose:
		print("\n==================================================================================")
		print("   MIRENA DYNAMICS (TeR27) — AUTOMATED TV CHARACTERIZATION & A/B BENCHMARK SUITE")
		print("==================================================================================")
		print("Discovered TrackFiles (%d): %s\n" % [
			tracks.size(),
			", ".join(tracks.map(func(t): return String(t["file"])))
		])

	# Verify TrackFiles discovery includes skidpad.json & acceleration.json
	var track_ids: Array[String] = []
	for t in tracks:
		track_ids.append(String(t["id"]).to_lower())
	if track_ids.has("skidpad") and track_ids.has("acceleration"):
		passed += 1
	else:
		failed += 1
		errors.append("Missing expected TrackFiles (skidpad/acceleration): found %s" % str(track_ids))

	var track_results_by_map: Dictionary = {}  # track_id -> {map_id -> metrics}
	var iso_results_by_map: Dictionary = {}    # iso_name -> {map_id -> metrics}

	# 1. Run every discovered circuit across all 4 TV maps
	for trk in tracks:
		var tid: String = String(trk["id"])
		track_results_by_map[tid] = {}
		for map_id in MAP_IDS:
			var res: Dictionary = _run_track_benchmark(cfg, trk, map_id, {})
			track_results_by_map[tid][map_id] = res

		if verbose:
			_print_comparison_table("TRACK: %s (%s)" % [String(trk["name"]), String(trk["file"])], track_results_by_map[tid])

	# 2. Run ISO maneuvers across all 4 TV maps
	var iso_specs: Array[String] = ["step_steer", "slalom", "braking_in_turn"]
	for iso_name in iso_specs:
		iso_results_by_map[iso_name] = {}
		for map_id in MAP_IDS:
			var res_iso: Dictionary = _run_iso_maneuver(cfg, iso_name, map_id, {})
			iso_results_by_map[iso_name][map_id] = res_iso

		if verbose:
			_print_comparison_table("ISO MANEUVER: %s" % iso_name.to_upper(), iso_results_by_map[iso_name])

	# 3. Physical & A/B Assertions
	# A) All maps complete all tracks and ISO maneuvers with finite states
	var all_finite_and_done: bool = true
	for tid in track_results_by_map.keys():
		for map_id in MAP_IDS:
			var m: Dictionary = track_results_by_map[tid][map_id]
			if not bool(m.get("finite_ok", false)) or not bool(m.get("completed", false)):
				all_finite_and_done = false
				errors.append("Track '%s' map %d failed: finite=%s completed=%s reason=%s" % [
					tid, map_id, str(m.get("finite_ok")), str(m.get("completed")), str(m.get("abort_reason", ""))
				])
	for iso_name in iso_results_by_map.keys():
		for map_id in MAP_IDS:
			var mi: Dictionary = iso_results_by_map[iso_name][map_id]
			if not bool(mi.get("finite_ok", false)) or not bool(mi.get("completed", false)):
				all_finite_and_done = false
				errors.append("ISO '%s' map %d failed: finite=%s completed=%s" % [
					iso_name, map_id, str(mi.get("finite_ok")), str(mi.get("completed"))
				])
	if all_finite_and_done:
		passed += 1
	else:
		failed += 1

	# B) Acceleration 75 m time realistic (< 5.0 s) on all maps
	if track_results_by_map.has("acceleration"):
		var accel_ok: bool = true
		for map_id in MAP_IDS:
			var t_acc: float = float(track_results_by_map["acceleration"][map_id]["sector_time_s"])
			if t_acc <= 1.5 or t_acc >= 5.0:
				accel_ok = false
				errors.append("Acceleration map %d unrealistic time: %.3f s" % [map_id, t_acc])
		if accel_ok:
			passed += 1
		else:
			failed += 1

	# C) Skidpad full figure-8 completed & realistic time on all maps
	if track_results_by_map.has("skidpad"):
		var sk_ok: bool = true
		for map_id in MAP_IDS:
			var sk: Dictionary = track_results_by_map["skidpad"][map_id]
			var t_sk: float = float(sk["sector_time_s"])
			if not bool(sk["completed"]) or t_sk < 14.0 or t_sk > 45.0:
				sk_ok = false
				errors.append("Skidpad map %d invalid completion: completed=%s t=%.3f s" % [
					map_id, str(sk["completed"]), t_sk
				])
		if sk_ok:
			passed += 1
		else:
			failed += 1

	# D) OPEN_4WD (map 0) applies mz_req == 0.0 everywhere
	var open_mz_ok: bool = true
	for tid in track_results_by_map.keys():
		var max_mz0: float = float(track_results_by_map[tid][0].get("max_abs_mz_req", 999.0))
		if max_mz0 > 1e-4:
			open_mz_ok = false
			errors.append("OPEN_4WD (map 0) produced non-zero Mz on '%s': max|mz_req|=%.3f" % [tid, max_mz0])
	for iso_name in iso_results_by_map.keys():
		var max_mzi: float = float(iso_results_by_map[iso_name][0].get("max_abs_mz_req", 999.0))
		if max_mzi > 1e-4:
			open_mz_ok = false
			errors.append("OPEN_4WD (map 0) produced non-zero Mz on ISO '%s': max|mz_req|=%.3f" % [iso_name, max_mzi])
	if open_mz_ok:
		passed += 1
	else:
		failed += 1

	# E) DYC maps improve yaw-tracking (wz_rmse_rads) or sideslip stability (peak_beta_deg) vs OPEN_4WD
	var dyc_improves_ok: bool = true
	var any_dyc_checked: bool = false
	for map_id in [1, 2, 3]:
		var improved_here: bool = false
		if track_results_by_map.has("skidpad"):
			any_dyc_checked = true
			var base_sk: Dictionary = track_results_by_map["skidpad"][0]
			var map_sk: Dictionary = track_results_by_map["skidpad"][map_id]
			if float(map_sk["wz_rmse_rads"]) < float(base_sk["wz_rmse_rads"]) + 1e-4 \
					or float(map_sk["peak_beta_deg"]) < float(base_sk["peak_beta_deg"]) + 1e-3:
				improved_here = true
		for iso_name in ["step_steer", "slalom"]:
			if iso_results_by_map.has(iso_name):
				any_dyc_checked = true
				var base_i: Dictionary = iso_results_by_map[iso_name][0]
				var map_i: Dictionary = iso_results_by_map[iso_name][map_id]
				if float(map_i["wz_rmse_rads"]) < float(base_i["wz_rmse_rads"]) \
						or float(map_i["peak_beta_deg"]) < float(base_i["peak_beta_deg"]):
					improved_here = true
		if any_dyc_checked and not improved_here:
			dyc_improves_ok = false
			errors.append("DYC map %d (%s) did not improve wz_rmse or peak_beta vs OPEN_4WD" % [map_id, MAP_NAMES[map_id]])
	if dyc_improves_ok:
		passed += 1
	else:
		failed += 1

	# F) TelemetryRecorder & VehicleTelemetry CSV export smoke verification
	var tel_ok: bool = _verify_telemetry_export(cfg, errors)
	if tel_ok:
		passed += 1
	else:
		failed += 1

	# G) V3 AL-QP (03_skidpad_attack) vs V4 SMC Hybrid (05_smc_pivot_4wd) A/B Benchmark
	var v4_ok: bool = _run_v3_vs_v4_benchmark(cfg, tracks, verbose, errors)
	if v4_ok:
		passed += 1
	else:
		failed += 1

	# H) V4 SMC Hybrid specific assertions: Step-Steer Entry (65 km/h no gas) & Full-Throttle Cornering TC
	var v4_spec_ok: bool = _verify_v4_smc_entry_and_tc(cfg, verbose, errors)
	if v4_spec_ok:
		passed += 1
	else:
		failed += 1

	if verbose:
		print("----------------------------------------------------------------------------------")
		print("TV Characterization Summary: %d/%d assertions PASSED." % [passed, passed + failed])
		print("==================================================================================\n")

	return {
		"name": "test_tv_characterization",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}

# ── Track Discovery & Polyline Geometry ───────────────────────────────────────

static func _discover_tracks() -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var dir: DirAccess = DirAccess.open(TRACKS_DIR)
	if dir == null:
		return out

	var files: Array[String] = []
	dir.list_dir_begin()
	var fname: String = dir.get_next()
	while fname != "":
		if not dir.current_is_dir() and fname.to_lower().ends_with(".json"):
			files.append(fname)
		fname = dir.get_next()
	dir.list_dir_end()
	files.sort()

	for f in files:
		var parsed: Dictionary = _load_track_json(TRACKS_DIR + f, f)
		if not parsed.is_empty():
			out.append(parsed)
	return out

static func _load_track_json(path: String, file_name: String) -> Dictionary:
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary):
		return {}
	var data: Dictionary = parsed
	var track_id: String = file_name.get_basename().to_lower()
	var meta: Dictionary = data.get("metadata", {}) if data.get("metadata") is Dictionary else {}
	var track_name: String = String(meta.get("track_name", track_id))

	var setup: Dictionary = data.get("setup", {}) if data.get("setup") is Dictionary else {}
	var pose_d: Dictionary = setup.get("car_start_pose", {}) if setup.get("car_start_pose") is Dictionary else {}
	var start_pose: Vector3 = Vector3(
		float(pose_d.get("x", 0.0)),
		float(pose_d.get("y", 0.0)),
		float(pose_d.get("psi", 0.0))
	)

	var raw_pts: Array[Vector2] = []
	var is_closed: bool = false
	if data.has("path") and data["path"] is Dictionary:
		var p_dict: Dictionary = data["path"]
		is_closed = bool(p_dict.get("closed", false))
		var pts = p_dict.get("points", [])
		if pts is Array:
			for pt in pts:
				if pt is Dictionary:
					raw_pts.append(Vector2(float(pt.get("x", 0.0)), float(pt.get("y", 0.0))))

	if raw_pts.size() < 2:
		if track_id.contains("skidpad"):
			raw_pts = _build_skidpad_figure8_from_cones(data.get("cones", []), start_pose)
			is_closed = false
		else:
			var heading := Vector2(cos(start_pose.z), sin(start_pose.z))
			for step_i in range(25):
				raw_pts.append(Vector2(start_pose.x, start_pose.y) + heading * float(step_i) * 4.0)

	var resampled: Array[Vector2] = _resample_polyline(raw_pts, 1.0, is_closed)
	var geom: Dictionary = _compute_polyline_geometry(resampled, is_closed)

	return {
		"id": track_id,
		"file": file_name,
		"name": track_name,
		"start_pose": start_pose,
		"is_closed": is_closed,
		"points": resampled,
		"s_arr": geom["s_arr"],
		"kappa_arr": geom["kappa_arr"],
		"total_length": float(geom["total_length"]),
		"is_accel": track_id.contains("accel"),
		"is_skidpad": track_id.contains("skidpad")
	}

static func _build_skidpad_figure8_from_cones(cones: Variant, start_pose: Vector3) -> Array[Vector2]:
	var center_r := Vector2(9.25, 0.0)
	var center_l := Vector2(-9.25, 0.0)
	var radius: float = 9.125

	if cones is Array and cones.size() > 0:
		var sum_r := Vector2.ZERO
		var cnt_r: int = 0
		var sum_l := Vector2.ZERO
		var cnt_l: int = 0
		for c in cones:
			if c is Dictionary:
				var cx: float = float(c.get("x", 0.0))
				var cy: float = float(c.get("y", 0.0))
				var ctype: String = String(c.get("type", ""))
				if ctype == "cone_blue" or ctype == "cone_yellow":
					if cx > 2.0:
						sum_r += Vector2(cx, cy)
						cnt_r += 1
					elif cx < -2.0:
						sum_l += Vector2(cx, cy)
						cnt_l += 1
		if cnt_r > 4:
			center_r = sum_r / float(cnt_r)
		if cnt_l > 4:
			center_l = sum_l / float(cnt_l)
		var r_accum: float = 0.0
		var r_cnt: int = 0
		for c in cones:
			if c is Dictionary:
				var ctype: String = String(c.get("type", ""))
				if ctype == "cone_blue" or ctype == "cone_yellow":
					var p := Vector2(float(c.get("x", 0.0)), float(c.get("y", 0.0)))
					if p.x > 2.0:
						r_accum += p.distance_to(center_r)
						r_cnt += 1
					elif p.x < -2.0:
						r_accum += p.distance_to(center_l)
						r_cnt += 1
		if r_cnt > 4:
			radius = r_accum / float(r_cnt)

	var pts: Array[Vector2] = []
	var y_start: float = minf(start_pose.y - 0.6, -15.0)
	var entry_steps: int = int(round(absf(y_start)))
	for i in range(entry_steps):
		pts.append(Vector2(0.0, y_start + float(i)))

	var n_circle: int = 58
	# Two clockwise laps around right center (entry from (0,0) at angle PI, moving toward +Y)
	for lap in range(2):
		for k in range(n_circle):
			var ang: float = PI - TAU * (float(k) / float(n_circle))
			pts.append(center_r + Vector2(cos(ang), sin(ang)) * radius)

	# Two counter-clockwise laps around left center (entry from (0,0) at angle 0, moving toward +Y)
	for lap in range(2):
		for k in range(n_circle):
			var ang_l: float = TAU * (float(k) / float(n_circle))
			pts.append(center_l + Vector2(cos(ang_l), sin(ang_l)) * radius)

	for i in range(21):
		pts.append(Vector2(0.0, float(i)))
	return pts

static func _resample_polyline(raw: Array[Vector2], ds: float, is_closed: bool) -> Array[Vector2]:
	if raw.size() < 2:
		return raw
	var s_raw: Array[float] = [0.0]
	for i in range(1, raw.size()):
		s_raw.append(s_raw[i - 1] + raw[i - 1].distance_to(raw[i]))
	if is_closed:
		var close_d: float = raw[-1].distance_to(raw[0])
		if close_d > 1e-3:
			raw = raw.duplicate()
			raw.append(raw[0])
			s_raw.append(s_raw[-1] + close_d)

	var total_len: float = s_raw[-1]
	if total_len < ds:
		return raw

	var out: Array[Vector2] = []
	var n_samples: int = maxi(2, int(floor(total_len / ds)))
	var seg_idx: int = 0
	for k in range(n_samples + 1):
		var target_s: float = minf(float(k) * ds, total_len)
		while seg_idx < s_raw.size() - 2 and s_raw[seg_idx + 1] < target_s:
			seg_idx += 1
		var seg_len: float = maxf(s_raw[seg_idx + 1] - s_raw[seg_idx], 1e-6)
		var alpha: float = clampf((target_s - s_raw[seg_idx]) / seg_len, 0.0, 1.0)
		out.append(raw[seg_idx].lerp(raw[seg_idx + 1], alpha))
	if out[-1].distance_to(raw[-1]) > 0.25 and not is_closed:
		out.append(raw[-1])
	return out

static func _compute_polyline_geometry(pts: Array[Vector2], is_closed: bool) -> Dictionary:
	var n: int = pts.size()
	var s_arr: Array[float] = []
	var kappa_arr: Array[float] = []
	s_arr.resize(n)
	kappa_arr.resize(n)

	if n == 0:
		return {"s_arr": s_arr, "kappa_arr": kappa_arr, "total_length": 0.0}

	s_arr[0] = 0.0
	for i in range(1, n):
		s_arr[i] = s_arr[i - 1] + pts[i - 1].distance_to(pts[i])

	var span: int = 4  # ~4 m chord half-span suppresses noisy waypoint jitter while preserving exact circle curvature
	for i in range(n):
		var i0: int = i - span
		var i2: int = i + span
		if is_closed:
			i0 = posmod(i0, n)
			i2 = posmod(i2, n)
		else:
			i0 = maxi(0, i0)
			i2 = mini(n - 1, i2)
		if i0 == i or i2 == i or i0 == i2:
			kappa_arr[i] = 0.0
			continue
		var p0: Vector2 = pts[i0]
		var p1: Vector2 = pts[i]
		var p2: Vector2 = pts[i2]
		var area2: float = absf((p1 - p0).cross(p2 - p0))
		var d01: float = p0.distance_to(p1)
		var d12: float = p1.distance_to(p2)
		var d20: float = p2.distance_to(p0)
		var denom: float = maxf(d01 * d12 * d20, 1e-6)
		kappa_arr[i] = (2.0 * area2) / denom

	return {"s_arr": s_arr, "kappa_arr": kappa_arr, "total_length": s_arr[-1]}

# ── TV Profile & Safe Custom Parameter Application ────────────────────────────

static func _apply_tv_profile(
	model: VehicleDynamicsModelScript,
	inputs: VehicleInputsScript,
	map_id: int,
	custom_params: Dictionary = {}
) -> void:
	inputs.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	inputs.tv_mode = 0
	inputs.tv_yaw_moment_req = 0.0

	var ecu = model.ecu_tv_v3
	var has_custom: bool = not custom_params.is_empty()

	if map_id == 0 and not has_custom:
		# OPEN_4WD baseline: symmetric open-differential 4WD (Mz_req = 0.0)
		inputs.ecu_driving_mode = 0
		if ecu != null and ecu.has_method("set_map"):
			ecu.call("set_map", 0)
		if ecu != null:
			ecu.last_mz_req = 0.0
		return

	inputs.ecu_driving_mode = 1
	if ecu != null:
		if custom_params.has("map_file") and ecu.has_method("load_tv_map_from_file"):
			ecu.call("load_tv_map_from_file", String(custom_params["map_file"]))
			return
		if ecu.has_method("set_map"):
			ecu.call("set_map", map_id)
		if has_custom and ecu.has_method("set_custom_parameters"):
			var arg_count: int = 4
			for m in ecu.get_method_list():
				if String(m.get("name", "")) == "set_custom_parameters":
					arg_count = (m.get("args", []) as Array).size()
					break
			if arg_count == 1:
				ecu.call("set_custom_parameters", custom_params)
			else:
				ecu.call(
					"set_custom_parameters",
					float(custom_params.get("k_us", 0.015)),
					float(custom_params.get("max_mz", 500.0)),
					float(custom_params.get("tc_kp", 23.113)),
					float(custom_params.get("fx_res", 0.08))
				)

static func _compute_wz_ref_for_eval(
	model: VehicleDynamicsModelScript,
	steer_rad: float,
	map_id: int
) -> float:
	var ecu = model.ecu_tv_v3
	if map_id != 0 and ecu != null and "last_wz_ref" in ecu and model.inputs.ecu_driving_mode == 1:
		return float(ecu.last_wz_ref)
	var vx_safe: float = maxf(absf(model.chassis_state.linear_velocity.x), 0.5)
	var delta_dz: float = 0.0 if absf(steer_rad) < 0.0087 else steer_rad
	var wb: float = model.config.wheelbase if model.config != null else 1.540
	var k_us_val: float = 0.015
	if ecu != null and "k_us_active" in ecu and map_id != 0:
		k_us_val = float(ecu.get("k_us_active"))
	return (vx_safe * delta_dz) / (wb + k_us_val * vx_safe * vx_safe)

# ── Closed-Loop Track Driver (Pure Pursuit + Curvature Speed Control) ────────

static func _run_track_benchmark(
	cfg: VehicleConfigScript,
	trk: Dictionary,
	map_id: int,
	custom_params: Dictionary = {}
) -> Dictionary:
	var model: VehicleDynamicsModelScript = VehicleDynamicsModelScript.new(cfg)
	var inputs: VehicleInputsScript = VehicleInputsScript.new()

	var start_pose: Vector3 = trk["start_pose"]
	model.chassis_state.position = Vector3(start_pose.x, start_pose.y, cfg.h_cog)
	model.chassis_state.yaw = start_pose.z
	model.chassis_state.linear_velocity = Vector3.ZERO
	model.chassis_state.angular_velocity = Vector3.ZERO

	_apply_tv_profile(model, inputs, map_id, custom_params)

	var pts: Array[Vector2] = trk["points"]
	var s_arr: Array[float] = trk["s_arr"]
	var kappa_arr: Array[float] = trk["kappa_arr"]
	var total_len: float = float(trk["total_length"])
	var is_closed: bool = bool(trk["is_closed"])
	var is_accel: bool = bool(trk["is_accel"])
	var n_pts: int = pts.size()

	# Initialize progress index near car start pose
	var pos_start_2d := Vector2(start_pose.x, start_pose.y)
	var start_heading := Vector2(cos(start_pose.z), sin(start_pose.z))
	var progress_idx: int = 0
	var best_d0: float = 1e18
	for i in range(mini(15, n_pts)):
		var d0: float = pos_start_2d.distance_to(pts[i])
		if d0 < best_d0:
			best_d0 = d0
			progress_idx = i

	var max_time_s: float = 10.0 if is_accel else 65.0
	var max_steps: int = int(ceil(max_time_s / SIM_DT))
	var ay_max: float = 14.3
	var ax_brake_plan: float = 11.0
	var v_max_straight: float = 38.0

	var prev_steer: float = 0.0
	var int_err_v: float = 0.0
	var lap_walked_m: float = 0.0
	var prev_s: float = s_arr[progress_idx]

	var metrics_acc: Dictionary = _init_metrics_accumulator(model)
	var completed: bool = false
	var finite_ok: bool = true
	var abort_reason: String = ""
	var t_elapsed: float = 0.0
	var dist_traveled: float = 0.0

	for _step in range(max_steps):
		var pos_cog := Vector2(model.chassis_state.position.x, model.chassis_state.position.y)
		var psi: float = model.chassis_state.yaw
		var heading := Vector2(cos(psi), sin(psi))
		var pos_rear := pos_cog - cfg.b_cog * heading

		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		var v_mps: float = sqrt(vx * vx + vy * vy)

		if not is_finite(pos_cog.x) or not is_finite(pos_cog.y) or not is_finite(vx) or not is_finite(wz):
			finite_ok = false
			abort_reason = "NaN/Inf state"
			break

		# Sequential forward-window progress search (disambiguates Figure-8 crossings)
		var best_idx: int = progress_idx
		var best_dist: float = 1e18
		var search_win: int = 18
		for offset in range(-2, search_win + 1):
			var idx: int = progress_idx + offset
			if is_closed:
				idx = posmod(idx, n_pts)
			elif idx < 0 or idx >= n_pts:
				continue
			var d_p: float = pos_cog.distance_to(pts[idx])
			if d_p < best_dist:
				best_dist = d_p
				best_idx = idx
		progress_idx = best_idx

		var ds_prog: float = s_arr[progress_idx] - prev_s
		if is_closed and ds_prog < -0.5 * total_len:
			ds_prog += total_len
		if ds_prog > 0.0:
			lap_walked_m += ds_prog
		prev_s = s_arr[progress_idx]

		# Check completion conditions
		if is_accel:
			var forward_m: float = (pos_cog - pos_start_2d).dot(start_heading)
			if forward_m >= ACCEL_TARGET_DIST_M:
				completed = true
				break
		elif is_closed:
			if lap_walked_m >= total_len - 1.0:
				completed = true
				break
		else:
			if progress_idx >= n_pts - 2 or s_arr[progress_idx] >= total_len - 1.5:
				completed = true
				break

		# Off-track safety bailout
		if best_dist > 10.0 and t_elapsed > 1.0:
			abort_reason = "Off-track deviation (%.2f m)" % best_dist
			break

		# Pure Pursuit Lookahead point
		var l_d: float = 6.5 if is_accel else clampf(2.4 + 0.14 * v_mps, 2.4, 6.5)
		var look_idx: int = progress_idx
		var target_pt: Vector2 = pts[n_pts - 1]
		if is_closed:
			var walked_la: float = 0.0
			var cur_i: int = progress_idx
			for _k in range(n_pts):
				var nxt_i: int = (cur_i + 1) % n_pts
				var seg_d: float = pts[cur_i].distance_to(pts[nxt_i])
				if walked_la + seg_d >= l_d:
					var f_seg: float = (l_d - walked_la) / maxf(seg_d, 1e-6)
					target_pt = pts[cur_i].lerp(pts[nxt_i], f_seg)
					look_idx = nxt_i
					break
				walked_la += seg_d
				cur_i = nxt_i
		else:
			var s_goal: float = minf(s_arr[progress_idx] + l_d, total_len)
			var cur_i2: int = progress_idx
			while cur_i2 < n_pts - 2 and s_arr[cur_i2 + 1] < s_goal:
				cur_i2 += 1
			if cur_i2 >= n_pts - 1:
				target_pt = pts[n_pts - 1]
			else:
				var seg_s: float = maxf(s_arr[cur_i2 + 1] - s_arr[cur_i2], 1e-6)
				var f_s: float = clampf((s_goal - s_arr[cur_i2]) / seg_s, 0.0, 1.0)
				target_pt = pts[cur_i2].lerp(pts[cur_i2 + 1], f_s)
			look_idx = cur_i2

		# Pure Pursuit steering law in vehicle body frame
		var rel: Vector2 = target_pt - pos_rear
		var local_x: float = heading.x * rel.x + heading.y * rel.y
		var local_y: float = -heading.y * rel.x + heading.x * rel.y
		var d_actual: float = maxf(rel.length(), 0.25)
		var alpha_pp: float = atan2(local_y, maxf(local_x, 0.1))
		var delta_raw: float = atan2(2.0 * cfg.wheelbase * sin(alpha_pp), d_actual)
		var delta_clamped: float = clampf(delta_raw, -cfg.max_steer_angle, cfg.max_steer_angle)
		var steer_cmd: float = move_toward(prev_steer, delta_clamped, 3.2 * SIM_DT)
		prev_steer = steer_cmd

		# Curvature-limited target speed lookahead
		var v_target: float = v_max_straight
		if not is_accel:
			var scan_m: float = 28.0
			var d_scan: float = 0.0
			var idx_scan: int = progress_idx
			for _j in range(35):
				var kap: float = absf(kappa_arr[idx_scan])
				var v_curv: float = sqrt(ay_max / maxf(kap, 1e-4))
				var v_allow: float = sqrt(v_curv * v_curv + 2.0 * ax_brake_plan * d_scan)
				v_target = minf(v_target, v_allow)
				var nxt_scan: int = (idx_scan + 1) % n_pts if is_closed else idx_scan + 1
				if not is_closed and nxt_scan >= n_pts:
					break
				d_scan += pts[idx_scan].distance_to(pts[nxt_scan])
				idx_scan = nxt_scan
				if d_scan > scan_m:
					break
			v_target = clampf(v_target, 3.5, v_max_straight)

		# Longitudinal PI speed control
		var throttle_cmd: float = 1.0
		var brake_cmd: float = 0.0
		if not is_accel:
			var err_v: float = v_target - vx
			int_err_v = clampf(int_err_v + err_v * SIM_DT, -3.0, 3.0)
			if err_v >= -0.35:
				throttle_cmd = clampf(0.18 + 0.50 * err_v + 0.20 * int_err_v, 0.0, 1.0)
				brake_cmd = 0.0
			else:
				throttle_cmd = 0.0
				brake_cmd = clampf((-err_v - 0.35) * 0.45, 0.0, 0.85)

		inputs.throttle = throttle_cmd
		inputs.brake = brake_cmd
		inputs.steer_angle = steer_cmd
		inputs.steer_normalized = steer_cmd / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		t_elapsed += SIM_DT
		dist_traveled += v_mps * SIM_DT

		var wz_ref: float = _compute_wz_ref_for_eval(model, steer_cmd, map_id)
		_accumulate_metrics(metrics_acc, model, inputs, step_res, wz_ref, SIM_DT)

	return _finalize_metrics(metrics_acc, t_elapsed, dist_traveled, completed, finite_ok, abort_reason)

# ── ISO Maneuvers ─────────────────────────────────────────────────────────────

static func _run_iso_maneuver(
	cfg: VehicleConfigScript,
	iso_name: String,
	map_id: int,
	custom_params: Dictionary = {}
) -> Dictionary:
	var model: VehicleDynamicsModelScript = VehicleDynamicsModelScript.new(cfg)
	var inputs: VehicleInputsScript = VehicleInputsScript.new()
	_apply_tv_profile(model, inputs, map_id, custom_params)

	var u0: float = 16.0
	var duration_s: float = 1.5
	match iso_name:
		"step_steer":
			u0 = 16.0
			duration_s = 1.5
		"slalom":
			u0 = 15.0
			duration_s = 4.0
		"braking_in_turn":
			u0 = 20.0
			duration_s = 1.2

	model.chassis_state.position = Vector3(0.0, 0.0, cfg.h_cog)
	model.chassis_state.linear_velocity = Vector3(u0, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u0 / cfg.wheel_radius

	var n_steps: int = int(round(duration_s / SIM_DT))
	var int_v: float = 0.0
	var t_elapsed: float = 0.0
	var dist_traveled: float = 0.0
	var finite_ok: bool = true
	var metrics_acc: Dictionary = _init_metrics_accumulator(model)

	for k in range(n_steps):
		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		if not is_finite(vx) or not is_finite(vy) or not is_finite(wz):
			finite_ok = false
			break

		var steer_cmd: float = 0.0
		var throttle_cmd: float = 0.0
		var brake_cmd: float = 0.0

		match iso_name:
			"step_steer":
				steer_cmd = 0.10
				var e_v: float = u0 - vx
				int_v = clampf(int_v + e_v * SIM_DT, -2.0, 2.0)
				throttle_cmd = clampf(0.28 + 0.35 * e_v + 0.25 * int_v, 0.05, 0.90)
			"slalom":
				steer_cmd = 0.08 * sin(TAU * 0.65 * t_elapsed)
				var e_vs: float = u0 - vx
				int_v = clampf(int_v + e_vs * SIM_DT, -2.0, 2.0)
				throttle_cmd = clampf(0.25 + 0.35 * e_vs + 0.25 * int_v, 0.05, 0.90)
			"braking_in_turn":
				steer_cmd = 0.11
				throttle_cmd = 0.0
				brake_cmd = 0.55

		inputs.throttle = throttle_cmd
		inputs.brake = brake_cmd
		inputs.steer_angle = clampf(steer_cmd, -cfg.max_steer_angle, cfg.max_steer_angle)
		inputs.steer_normalized = inputs.steer_angle / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		t_elapsed += SIM_DT
		dist_traveled += sqrt(vx * vx + vy * vy) * SIM_DT

		var wz_ref: float = _compute_wz_ref_for_eval(model, inputs.steer_angle, map_id)
		_accumulate_metrics(metrics_acc, model, inputs, step_res, wz_ref, SIM_DT)

	return _finalize_metrics(metrics_acc, t_elapsed, dist_traveled, finite_ok, finite_ok, "")

# ── Metrics Computation (RMSE, Peak Beta, Stability Ratio, Energy, SRR, Temp) ──

static func _init_metrics_accumulator(model: VehicleDynamicsModelScript) -> Dictionary:
	var init_temps: Array[float] = VehicleTelemetryScript.extract_tire_temps(model)
	return {
		"n": 0,
		"wz_err_sq_sum": 0.0,
		"peak_beta_deg": 0.0,
		"alpha_f_sq_sum": 0.0,
		"alpha_r_sq_sum": 0.0,
		"alpha_samples": 0,
		"energy_kwh": 0.0,
		"max_abs_mz_req": 0.0,
		"max_kappa": 0.0,
		"slew_sq_sum": 0.0,
		"slew_count": 0,
		"prev_trqs": [0.0, 0.0, 0.0, 0.0],
		"has_prev_trqs": false,
		"srr_dir": 0,
		"srr_ext": 0.0,
		"srr_initialized": false,
		"srr_count": 0,
		"init_wheel_temps": init_temps,
		"proxy_temps": init_temps.duplicate(),
		"last_model": model
	}

static func _accumulate_metrics(
	acc: Dictionary,
	model: VehicleDynamicsModelScript,
	inputs: VehicleInputsScript,
	step_res: Dictionary,
	wz_ref: float,
	dt: float
) -> void:
	var vx: float = model.chassis_state.linear_velocity.x
	var vy: float = model.chassis_state.linear_velocity.y
	var wz: float = model.chassis_state.angular_velocity.z
	var speed: float = sqrt(vx * vx + vy * vy)

	var wz_err: float = wz_ref - wz
	acc["wz_err_sq_sum"] = float(acc["wz_err_sq_sum"]) + wz_err * wz_err
	acc["n"] = int(acc["n"]) + 1

	if speed > 1.0:
		var beta_deg: float = absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
		acc["peak_beta_deg"] = maxf(float(acc["peak_beta_deg"]), beta_deg)

	# Rear stability ratio inputs: RMS rear slip vs RMS front slip during active cornering
	if speed > 2.0 and (absf(inputs.steer_angle) > 0.01 or absf(wz) > 0.05):
		var af: float = 0.5 * (absf(model.wheels[0].alpha) + absf(model.wheels[1].alpha))
		var ar: float = 0.5 * (absf(model.wheels[2].alpha) + absf(model.wheels[3].alpha))
		acc["alpha_f_sq_sum"] = float(acc["alpha_f_sq_sum"]) + af * af
		acc["alpha_r_sq_sum"] = float(acc["alpha_r_sq_sum"]) + ar * ar
		acc["alpha_samples"] = int(acc["alpha_samples"]) + 1

	# HV energy integration
	var forces_d: Dictionary = step_res.get("forces", step_res) if step_res.get("forces") is Dictionary else step_res
	var d_trqs: Array = forces_d.get("drive_torques", [0.0, 0.0, 0.0, 0.0])
	var hv: Dictionary = VehicleTelemetryScript.extract_hv_state(model, d_trqs, dt, step_res, float(acc["energy_kwh"]))
	acc["energy_kwh"] = float(hv["energy_kwh"])

	# Track max |mz_req| (to assert OPEN_4WD == 0)
	var tv_info: Dictionary = VehicleTelemetryScript.extract_tv_state(model, inputs.steer_angle, step_res)
	acc["max_abs_mz_req"] = maxf(float(acc["max_abs_mz_req"]), absf(float(tv_info["mz_req"])))

	for wi in range(mini(4, model.wheels.size())):
		acc["max_kappa"] = maxf(float(acc["max_kappa"]), absf(float(model.wheels[wi].kappa)))
	if bool(acc["has_prev_trqs"]) and dt > 1e-6:
		var pt: Array = acc["prev_trqs"]
		var s2 := 0.0
		for wi in range(4):
			var d_rate := (float(d_trqs[wi]) - float(pt[wi])) / dt
			s2 += d_rate * d_rate
		acc["slew_sq_sum"] = float(acc["slew_sq_sum"]) + s2 * 0.25
		acc["slew_count"] = int(acc["slew_count"]) + 1
	acc["prev_trqs"] = [float(d_trqs[0]), float(d_trqs[1]), float(d_trqs[2]), float(d_trqs[3])]
	acc["has_prev_trqs"] = true

	# SAE J2944 hysteresis-gap Steering Reversal Rate (> 2 deg)
	var steer_deg: float = rad_to_deg(inputs.steer_angle)
	if not bool(acc["srr_initialized"]):
		acc["srr_initialized"] = true
		acc["srr_ext"] = steer_deg
		acc["srr_dir"] = 0
	else:
		var dir: int = int(acc["srr_dir"])
		var ext: float = float(acc["srr_ext"])
		if dir == 0:
			if steer_deg - ext >= SRR_GAP_DEG:
				acc["srr_dir"] = 1
				acc["srr_ext"] = steer_deg
			elif ext - steer_deg >= SRR_GAP_DEG:
				acc["srr_dir"] = -1
				acc["srr_ext"] = steer_deg
		elif dir == 1:
			if steer_deg > ext:
				acc["srr_ext"] = steer_deg
			elif ext - steer_deg >= SRR_GAP_DEG:
				acc["srr_count"] = int(acc["srr_count"]) + 1
				acc["srr_dir"] = -1
				acc["srr_ext"] = steer_deg
		else:
			if steer_deg < ext:
				acc["srr_ext"] = steer_deg
			elif steer_deg - ext >= SRR_GAP_DEG:
				acc["srr_count"] = int(acc["srr_count"]) + 1
				acc["srr_dir"] = 1
				acc["srr_ext"] = steer_deg

	# Read-only lumped thermal tracker fallback (used only if WheelDynamics.step_thermal isn't stepped by VehicleDynamicsModel)
	var proxy: Array[float] = acc["proxy_temps"]
	var r_nom: float = model.config.wheel_radius if model.config != null else 0.2032
	for i in range(4):
		var w = model.wheels[i]
		var v_sx: float = w.omega * (w.effective_radius if w.effective_radius > 0.0 else r_nom) - w.v_longitudinal
		var v_sy: float = w.v_lateral
		var p_slip: float = (absf(w.F_x * v_sx) + absf(w.F_y * v_sy)) if (w.is_grounded and w.F_z > 1.0) else 0.0
		var q_net: float = 0.22 * p_slip - 22.0 * (proxy[i] - 25.0)
		proxy[i] = clampf(proxy[i] + (q_net / 1200.0) * dt, 15.0, 150.0)

static func _finalize_metrics(
	acc: Dictionary,
	t_elapsed: float,
	dist_traveled: float,
	completed: bool,
	finite_ok: bool,
	abort_reason: String
) -> Dictionary:
	var n: int = maxi(int(acc["n"]), 1)
	var wz_rmse: float = sqrt(float(acc["wz_err_sq_sum"]) / float(n))
	var alpha_n: int = int(acc["alpha_samples"])
	var rear_stab: float = 0.0
	if alpha_n > 0:
		var rms_f: float = sqrt(float(acc["alpha_f_sq_sum"]) / float(alpha_n))
		var rms_r: float = sqrt(float(acc["alpha_r_sq_sum"]) / float(alpha_n))
		rear_stab = rms_r / maxf(rms_f, 1e-4)

	var energy_kwh: float = float(acc["energy_kwh"])
	var dist_km: float = maxf(dist_traveled / 1000.0, 1e-6)
	var eff_kwh_km: float = energy_kwh / dist_km
	var srr_per_min: float = float(acc["srr_count"]) / maxf(t_elapsed / 60.0, 1e-6)

	var model: VehicleDynamicsModelScript = acc["last_model"]
	var final_temps: Array[float] = VehicleTelemetryScript.extract_tire_temps(model)
	var init_temps: Array[float] = acc["init_wheel_temps"]
	var thermal_live: bool = false
	for i in range(4):
		var w = model.wheels[i]
		if ("slip_power_w" in w and float(w.get("slip_power_w")) > 1e-3) or absf(final_temps[i] - init_temps[i]) > 1e-4:
			thermal_live = true
			break
	var used_temps: Array[float] = final_temps if thermal_live else acc["proxy_temps"]
	var mean_temp_c: float = 0.25 * (used_temps[0] + used_temps[1] + used_temps[2] + used_temps[3])

	var slew_rms := sqrt(float(acc["slew_sq_sum"]) / float(maxi(int(acc["slew_count"]), 1)))

	return {
		"sector_time_s": t_elapsed,
		"dist_m": dist_traveled,
		"wz_rmse_rads": wz_rmse,
		"peak_beta_deg": float(acc["peak_beta_deg"]),
		"rear_stability_ratio": rear_stab,
		"max_kappa": float(acc["max_kappa"]),
		"slew_rms": slew_rms,
		"energy_kwh": energy_kwh,
		"efficiency_kwh_km": eff_kwh_km,
		"srr_per_min": srr_per_min,
		"final_tire_temp_c": mean_temp_c,
		"max_abs_mz_req": float(acc["max_abs_mz_req"]),
		"completed": completed,
		"finite_ok": finite_ok,
		"abort_reason": abort_reason
	}

static func _aggregate_metrics(track_results: Dictionary) -> Dictionary:
	if track_results.is_empty():
		return {}
	var sum_time: float = 0.0
	var sum_dist: float = 0.0
	var sum_energy: float = 0.0
	var sum_wz_sq_t: float = 0.0
	var peak_beta: float = 0.0
	var max_rear_stab: float = 0.0
	var sum_srr_t: float = 0.0
	var sum_temp_t: float = 0.0

	for tid in track_results.keys():
		var m: Dictionary = track_results[tid]
		var t_s: float = maxf(float(m.get("sector_time_s", 0.0)), 1e-6)
		sum_time += t_s
		sum_dist += float(m.get("dist_m", 0.0))
		sum_energy += float(m.get("energy_kwh", 0.0))
		var wr: float = float(m.get("wz_rmse_rads", 0.0))
		sum_wz_sq_t += wr * wr * t_s
		peak_beta = maxf(peak_beta, float(m.get("peak_beta_deg", 0.0)))
		max_rear_stab = maxf(max_rear_stab, float(m.get("rear_stability_ratio", 0.0)))
		sum_srr_t += float(m.get("srr_per_min", 0.0)) * t_s
		sum_temp_t += float(m.get("final_tire_temp_c", 25.0)) * t_s

	return {
		"sector_time_s": sum_time,
		"dist_m": sum_dist,
		"wz_rmse_rads": sqrt(sum_wz_sq_t / maxf(sum_time, 1e-6)),
		"peak_beta_deg": peak_beta,
		"rear_stability_ratio": max_rear_stab,
		"energy_kwh": sum_energy,
		"efficiency_kwh_km": sum_energy / maxf(sum_dist / 1000.0, 1e-6),
		"srr_per_min": sum_srr_t / maxf(sum_time, 1e-6),
		"final_tire_temp_c": sum_temp_t / maxf(sum_time, 1e-6)
	}

# ── Reporting & Telemetry Smoke Verification ──────────────────────────────────

static func _print_comparison_table(title: String, by_map: Dictionary) -> void:
	print("[%s]" % title)
	print("  %-18s | %7s | %8s | %8s | %7s | %9s | %8s | %7s | %7s | %s" % [
		"Map", "Time(s)", "wzRMSE", "PkBeta", "RearStb", "Energy", "kWh/km", "SRR/min", "T_surf", "Status"
	])
	print("  " + "-".repeat(106))
	for map_id in MAP_IDS:
		var m: Dictionary = by_map.get(map_id, {})
		var st: String = "OK" if (bool(m.get("completed", false)) and bool(m.get("finite_ok", false))) else "FAIL"
		print("  %-18s | %7.3f | %8.4f | %7.2f° | %7.3f | %7.4fkWh | %8.3f | %7.1f | %6.1f°C | %s" % [
			String(MAP_NAMES.get(map_id, "MAP_%d" % map_id)),
			float(m.get("sector_time_s", 0.0)),
			float(m.get("wz_rmse_rads", 0.0)),
			float(m.get("peak_beta_deg", 0.0)),
			float(m.get("rear_stability_ratio", 0.0)),
			float(m.get("energy_kwh", 0.0)),
			float(m.get("efficiency_kwh_km", 0.0)),
			float(m.get("srr_per_min", 0.0)),
			float(m.get("final_tire_temp_c", 25.0)),
			st
		])
	print("")

static func _verify_telemetry_export(cfg: VehicleConfigScript, errors: Array) -> bool:
	var model: VehicleDynamicsModelScript = VehicleDynamicsModelScript.new(cfg)
	var inputs: VehicleInputsScript = VehicleInputsScript.new()
	_apply_tv_profile(model, inputs, 2, {})
	inputs.throttle = 0.6
	inputs.steer_angle = 0.08

	var rec: TelemetryRecorderScript = TelemetryRecorderScript.new()
	rec.sample_rate_hz = 200.0
	var csv_path: String = "user://telemetry/_test_tv_char_smoke.csv"
	var opened_path: String = rec.start_recording("smoke_track", "autocross_neutral", csv_path)
	if opened_path == "" or rec.get_hud_indicator() != "  |  [REC]":
		errors.append("TelemetryRecorder failed to start or HUD indicator mismatch")
		return false

	var tel: VehicleTelemetryScript = VehicleTelemetryScript.new()
	for _k in range(40): # 0.1 s @ 400 Hz => 20 samples @ 200 Hz
		var res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		rec.record_step(SIM_DT, model, inputs, res)
		var acc_ros: Vector3 = model.net_force_chassis_ros / maxf(cfg.mass, 1e-3)
		var step_info: Dictionary = res.duplicate()
		step_info["dt"] = SIM_DT
		tel.update_from_model(model, acc_ros, inputs.steer_angle, step_info)

	var stopped_path: String = rec.stop_recording()
	if rec.get_hud_indicator() != "" or rec.get_sample_count() != 20:
		errors.append("TelemetryRecorder sample count/HUD stop mismatch: count=%d" % rec.get_sample_count())
		return false

	var f: FileAccess = FileAccess.open(stopped_path, FileAccess.READ)
	if f == null:
		errors.append("TelemetryRecorder CSV unreadable at '%s'" % stopped_path)
		return false
	var header: String = f.get_line()
	var row1: String = f.get_line()
	f.close()
	DirAccess.remove_absolute(ProjectSettings.globalize_path(stopped_path))

	var expected_cols: int = TelemetryRecorderScript.CSV_HEADER.split(",").size()
	if header != TelemetryRecorderScript.CSV_HEADER or row1.split(",").size() != expected_cols:
		errors.append("TelemetryRecorder CSV header/row column count mismatch")
		return false

	var d: Dictionary = tel.to_dict()
	for required_key in ["wz_ref", "beta_deg", "map_id", "mz_req", "tire_temp_fl", "tire_temp_fr", "tire_temp_rl", "tire_temp_rr", "v_dc", "i_dc", "power_kw", "energy_kwh"]:
		if not d.has(required_key):
			errors.append("VehicleTelemetry.to_dict() missing key '%s'" % required_key)
			return false
	return true

static func _run_v3_vs_v4_benchmark(
	cfg: VehicleConfigScript,
	tracks: Array[Dictionary],
	verbose: bool,
	errors: Array
) -> bool:
	var skidpad_trk: Dictionary = {}
	for t in tracks:
		if String(t["id"]).to_lower() == "skidpad":
			skidpad_trk = t
			break
	if skidpad_trk.is_empty():
		errors.append("V3 vs V4 benchmark: skidpad track not found")
		return false

	var v3_path := "res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/03_skidpad_attack.json"
	var v4_path := "res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/05_smc_pivot_4wd.json"

	var sk_v3 := _run_track_benchmark(cfg, skidpad_trk, 3, {"map_file": v3_path})
	var sk_v4 := _run_track_benchmark(cfg, skidpad_trk, 3, {"map_file": v4_path})
	var sl_v3 := _run_iso_maneuver(cfg, "slalom", 3, {"map_file": v3_path})
	var sl_v4 := _run_iso_maneuver(cfg, "slalom", 3, {"map_file": v4_path})

	if verbose:
		print("[V3 AL-QP vs V4 SMC HYBRID — A/B BENCHMARK]")
		print("  %-26s | %7s | %8s | %8s | %8s | %9s | %s" % [
			"Scenario / Map", "Time(s)", "wzRMSE", "PkBeta", "MaxKappa", "SlewRMS", "Status"
		])
		print("  " + "-".repeat(92))
		var rows := [
			["Skidpad | 03_skidpad_attack", sk_v3],
			["Skidpad | 05_smc_pivot_4wd ", sk_v4],
			["Slalom  | 03_skidpad_attack", sl_v3],
			["Slalom  | 05_smc_pivot_4wd ", sl_v4]
		]
		for row in rows:
			var m: Dictionary = row[1]
			var st := "OK" if (bool(m.get("completed", false)) and bool(m.get("finite_ok", false))) else "FAIL"
			print("  %-26s | %7.3f | %8.4f | %7.2f° | %8.4f | %9.1f | %s" % [
				row[0],
				float(m.get("sector_time_s", 0.0)),
				float(m.get("wz_rmse_rads", 0.0)),
				float(m.get("peak_beta_deg", 0.0)),
				float(m.get("max_kappa", 0.0)),
				float(m.get("slew_rms", 0.0)),
				st
			])
		print("")

	var ok := true
	for pair in [["skidpad_v4", sk_v4], ["slalom_v4", sl_v4]]:
		var m: Dictionary = pair[1]
		if not bool(m.get("completed", false)) or not bool(m.get("finite_ok", false)):
			ok = false
			errors.append("V4 SMC failed on %s: completed=%s finite=%s reason=%s" % [
				pair[0], str(m.get("completed")), str(m.get("finite_ok")), str(m.get("abort_reason", ""))
			])
		if float(m.get("peak_beta_deg", 99.0)) > 15.0:
			ok = false
			errors.append("V4 SMC excessive sideslip on %s: peak_beta=%.2f deg" % [pair[0], float(m.get("peak_beta_deg", 0.0))])
	return ok

static func _verify_v4_smc_entry_and_tc(
	cfg: VehicleConfigScript,
	verbose: bool,
	errors: Array
) -> bool:
	var v4_path := "res://Scenes/Vehicle/Dynamics/Core/Config/tv_maps/05_smc_pivot_4wd.json"
	var ok := true

	# 1. Step-Steer Entry at 65 km/h (18.06 m/s) off-throttle
	var m_entry := VehicleDynamicsModelScript.new(cfg)
	var inp_entry := VehicleInputsScript.new()
	_apply_tv_profile(m_entry, inp_entry, 3, {"map_file": v4_path})
	var v_init := 65.0 / 3.6
	m_entry.chassis_state.linear_velocity = Vector3(v_init, 0.0, 0.0)
	for wi in range(4):
		m_entry.wheels[wi].omega = v_init / cfg.wheel_radius
	inp_entry.throttle = 0.0
	inp_entry.brake = 0.0

	var wz_sq_sum := 0.0
	var wz_n := 0
	var peak_beta_entry := 0.0
	var target_delta := 0.10
	var total_steps := int(round(1.5 / SIM_DT))
	for step_i in range(total_steps):
		var t := float(step_i) * SIM_DT
		inp_entry.steer_angle = target_delta * clampf(t / 0.15, 0.0, 1.0)
		var res: Dictionary = m_entry.step_standalone(inp_entry, SIM_DT, 0.0)
		var vx := m_entry.chassis_state.linear_velocity.x
		var vy := m_entry.chassis_state.linear_velocity.y
		var wz := m_entry.chassis_state.angular_velocity.z
		var beta_deg := absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
		peak_beta_entry = maxf(peak_beta_entry, beta_deg)
		var tv_d: Dictionary = VehicleTelemetryScript.extract_tv_state(m_entry, inp_entry.steer_angle, res)
		if t >= 0.25:
			var w_err := float(tv_d["wz_ref"]) - wz
			wz_sq_sum += w_err * w_err
			wz_n += 1
	var wz_rmse_entry := sqrt(wz_sq_sum / float(maxi(wz_n, 1)))
	if wz_rmse_entry >= 0.08 or peak_beta_entry >= 1.8:
		ok = false
		errors.append("V4 Step-Steer Entry 65 km/h failed: wz_rmse=%.4f (<0.08), peak_beta=%.2f deg (<1.8)" % [wz_rmse_entry, peak_beta_entry])

	# 2. Full-Throttle Cornering TC (100% gas in steady corner)
	var m_tc := VehicleDynamicsModelScript.new(cfg)
	var inp_tc := VehicleInputsScript.new()
	_apply_tv_profile(m_tc, inp_tc, 3, {"map_file": v4_path})
	var v_tc := 14.0
	m_tc.chassis_state.linear_velocity = Vector3(v_tc, 0.0, 0.0)
	for wi in range(4):
		m_tc.wheels[wi].omega = v_tc / cfg.wheel_radius
	inp_tc.steer_angle = 0.12
	inp_tc.throttle = 0.15
	for _s in range(int(round(0.4 / SIM_DT))):
		m_tc.step_standalone(inp_tc, SIM_DT, 0.0)

	inp_tc.throttle = 1.0
	var max_tc_cut := 0.0
	var max_kappa_mid := 0.0
	for _s in range(int(round(1.0 / SIM_DT))):
		m_tc.step_standalone(inp_tc, SIM_DT, 0.0)
		for wi in range(4):
			max_tc_cut = maxf(max_tc_cut, float(m_tc.ecu_tv_v3.last_tc_cut_nm[wi]))
			max_kappa_mid = maxf(max_kappa_mid, absf(float(m_tc.wheels[wi].kappa)))
	if max_tc_cut <= 30.0 or max_kappa_mid >= 0.055:
		ok = false
		errors.append("V4 Full-Throttle Cornering TC failed: max_tc_cut=%.1f Nm (>30), max_kappa=%.4f (<0.055)" % [max_tc_cut, max_kappa_mid])

	if verbose:
		print("[V4 SMC SPECIFIC VERIFICATION]")
		print("  Step-Steer Entry (65 km/h, 0%% gas) : wz_rmse=%.4f rad/s | peak_beta=%.2f deg | %s" % [
			wz_rmse_entry, peak_beta_entry, "PASS" if (wz_rmse_entry < 0.08 and peak_beta_entry < 1.8) else "FAIL"
		])
		print("  Full-Throttle Cornering TC (100%%)  : max_tc_cut=%.1f Nm | max_kappa=%.4f | %s\n" % [
			max_tc_cut, max_kappa_mid, "PASS" if (max_tc_cut > 30.0 and max_kappa_mid < 0.055) else "FAIL"
		])
	return ok
