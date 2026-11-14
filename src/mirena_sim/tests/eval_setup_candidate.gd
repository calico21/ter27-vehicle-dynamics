extends SceneTree

## Headless 14-DOF Setup Candidate Evaluator for Mirena Dynamics (TeR27).
## Runs 3 deterministic closed-loop maneuvers at 240 Hz (dt = 1/240 s) with
## EcuTorqueVectoringV3 active (ecu_driving_mode = 1):
##   1. Steady-state Skidpad (R = 9.125 m, ~8.5 s)
##   2. Fast Slalom (65 km/h = 18.06 m/s, 0.55 Hz, ~5.5 s)
##   3. Combined Trail Braking in Turn (90 -> 35 km/h, 25.0 -> 9.72 m/s)
##
## Usage:
##   godot --headless --path src/mirena_sim -s res://tests/eval_setup_candidate.gd -- \
##     --setup=/path/to/params.json --spring_f=38000 --brake_bias=0.68

const VehicleConfigScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const VehicleTelemetryScript = preload("res://Scenes/Vehicle/Dynamics/Telemetry/VehicleTelemetry.gd")

const SIM_DT: float = 1.0 / 240.0
const SKIDPAD_RADIUS_M: float = 9.125
const DEFAULT_TV_MAP_ID: int = 2

const PARAM_KEYS: Array[String] = [
	"spring_stiffness_front",
	"spring_stiffness_rear",
	"arb_stiffness_front",
	"arb_stiffness_rear",
	"static_camber_front_deg",
	"static_camber_rear_deg",
	"toe_front_deg",
	"toe_rear_deg",
	"brake_bias_front",
	"aero_balance_front"
]

const CLI_ALIAS_TO_PROP: Dictionary = {
	"spring_f": "spring_stiffness_front",
	"spring_stiffness_front": "spring_stiffness_front",
	"spring_r": "spring_stiffness_rear",
	"spring_stiffness_rear": "spring_stiffness_rear",
	"arb_f": "arb_stiffness_front",
	"arb_stiffness_front": "arb_stiffness_front",
	"arb_r": "arb_stiffness_rear",
	"arb_stiffness_rear": "arb_stiffness_rear",
	"camber_f": "static_camber_front_deg",
	"static_camber_front_deg": "static_camber_front_deg",
	"camber_r": "static_camber_rear_deg",
	"static_camber_rear_deg": "static_camber_rear_deg",
	"toe_f": "toe_front_deg",
	"toe_front_deg": "toe_front_deg",
	"toe_r": "toe_rear_deg",
	"toe_rear_deg": "toe_rear_deg",
	"brake_bias": "brake_bias_front",
	"brake_bias_front": "brake_bias_front",
	"aero_balance": "aero_balance_front",
	"aero_balance_front": "aero_balance_front"
}


func _init() -> void:
	var t0_usec: int = Time.get_ticks_usec()
	var cli: Dictionary = _parse_cli_args()
	var res: Dictionary = evaluate_candidate(cli, t0_usec)
	if bool(cli.get("verbose", false)):
		print("[eval_setup_candidate] Elapsed: %.3f s | feasible=%s | penalty=%.2f" % [
			float(res.get("eval_time_s", 0.0)),
			str(res.get("feasible", false)),
			float(res.get("penalty", 0.0))
		])
	print("SETUP_OPT_RESULT:" + JSON.stringify(res))
	quit(0 if bool(res.get("ok", false)) else 1)


static func _parse_cli_args() -> Dictionary:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	if args.is_empty():
		args = OS.get_cmdline_args()

	var out: Dictionary = {
		"setup_path": "",
		"overrides": {},
		"map_id": DEFAULT_TV_MAP_ID,
		"verbose": false
	}

	var i: int = 0
	while i < args.size():
		var raw: String = str(args[i]).strip_edges()
		if raw == "--verbose" or raw == "-v":
			out["verbose"] = true
			i += 1
			continue
		if raw == "--setup" or raw == "--params_json" or raw == "--params-json":
			if i + 1 < args.size():
				out["setup_path"] = str(args[i + 1]).strip_edges()
				i += 2
				continue
		if raw.begins_with("--setup="):
			out["setup_path"] = raw.substr("--setup=".length()).strip_edges()
			i += 1
			continue
		if raw.begins_with("--params_json="):
			out["setup_path"] = raw.substr("--params_json=".length()).strip_edges()
			i += 1
			continue
		if raw.begins_with("--params-json="):
			out["setup_path"] = raw.substr("--params-json=".length()).strip_edges()
			i += 1
			continue
		if raw.begins_with("--map="):
			out["map_id"] = int(raw.substr("--map=".length()))
			i += 1
			continue

		var clean_tok: String = raw.trim_prefix("--")
		if clean_tok.contains("="):
			var kv: PackedStringArray = clean_tok.split("=", true, 1)
			var k: String = kv[0].strip_edges().to_lower()
			var v: String = kv[1].strip_edges()
			if k == "map" or k == "map_id":
				out["map_id"] = int(v)
			elif CLI_ALIAS_TO_PROP.has(k):
				var prop: String = String(CLI_ALIAS_TO_PROP[k])
				out["overrides"][prop] = float(v)
		i += 1

	return out


static func _load_setup_json(path: String) -> Dictionary:
	if path == "":
		return {}
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	if not (parsed is Dictionary):
		return {}
	var d: Dictionary = parsed
	if d.has("params") and d["params"] is Dictionary:
		d = d["params"]
	var out: Dictionary = {}
	for raw_k in d.keys():
		var k_norm: String = String(raw_k).strip_edges().to_lower()
		if CLI_ALIAS_TO_PROP.has(k_norm):
			var prop: String = String(CLI_ALIAS_TO_PROP[k_norm])
			out[prop] = float(d[raw_k])
	return out


static func _sync_camber_table_to_static(table: Array, static_deg: float) -> void:
	var target_rad: float = deg_to_rad(static_deg)
	if table.is_empty():
		table.append([0.0, target_rad])
		return
	var base0: float = VehicleConfigScript._lookup_table_rad(table, 0.0)
	var offset: float = target_rad - base0
	if absf(offset) > 1e-6:
		for pt in table:
			if pt is Array and pt.size() >= 2:
				pt[1] = float(pt[1]) + offset


static func _build_configured_vehicle(cli: Dictionary) -> Dictionary:
	var cfg: VehicleConfigScript = VehicleConfigScript.new()
	var merged: Dictionary = {}
	var json_params: Dictionary = _load_setup_json(String(cli.get("setup_path", "")))
	for k in json_params.keys():
		merged[k] = json_params[k]
	var cli_overrides: Dictionary = cli.get("overrides", {})
	for k in cli_overrides.keys():
		merged[k] = cli_overrides[k]

	# Apply requested setup properties onto VehicleConfig safely
	for prop in merged.keys():
		var val: float = float(merged[prop])
		if prop == "aero_balance_front" and ("_has_explicit_axle_cl" in cfg):
			cfg.set("_has_explicit_axle_cl", false)
		if prop in cfg:
			cfg.set(prop, val)

	cfg.update_derived_geometry()

	# Ensure camber lookup tables match updated static_camber_*_deg if not shifted by update_derived_geometry
	_sync_camber_table_to_static(cfg.camber_table_front, cfg.static_camber_front_deg)
	_sync_camber_table_to_static(cfg.camber_table_rear, cfg.static_camber_rear_deg)

	# Build parameter echo dictionary
	var echo: Dictionary = {}
	for prop in PARAM_KEYS:
		if prop in cfg:
			echo[prop] = float(cfg.get(prop))
		elif merged.has(prop):
			echo[prop] = float(merged[prop])
		else:
			echo[prop] = 0.0

	return {"config": cfg, "params": echo}


static func _init_model_and_inputs(cfg: VehicleConfigScript, map_id: int, u0: float) -> Dictionary:
	var model: VehicleDynamicsModelScript = VehicleDynamicsModelScript.new(cfg)
	model.chassis_state.position = Vector3(0.0, 0.0, cfg.h_cog)
	model.chassis_state.linear_velocity = Vector3(u0, 0.0, 0.0)
	model.chassis_state.angular_velocity = Vector3.ZERO
	for w in model.wheels:
		w.omega = u0 / maxf(cfg.wheel_radius, 1e-4)

	var inputs: VehicleInputsScript = VehicleInputsScript.new()
	inputs.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	inputs.tv_mode = 0
	inputs.ecu_driving_mode = 1
	inputs.tv_map_id = map_id
	if model.ecu_tv_v3 != null and model.ecu_tv_v3.has_method("set_map"):
		model.ecu_tv_v3.call("set_map", map_id)

	return {"model": model, "inputs": inputs}


static func _is_state_finite(model: VehicleDynamicsModelScript) -> bool:
	var p: Vector3 = model.chassis_state.position
	var v: Vector3 = model.chassis_state.linear_velocity
	var w: Vector3 = model.chassis_state.angular_velocity
	return is_finite(p.x) and is_finite(p.y) and is_finite(p.z) \
		and is_finite(v.x) and is_finite(v.y) and is_finite(v.z) \
		and is_finite(w.x) and is_finite(w.y) and is_finite(w.z) \
		and is_finite(model.chassis_state.roll) and is_finite(model.chassis_state.pitch)


static func _min_array_float(arr: Array) -> float:
	var m: float = 1e18
	for v in arr:
		m = minf(m, float(v))
	return m


# ── Maneuver 1: Steady-State Skidpad (R = 9.125 m, 8.5 s @ 240 Hz) ───────────
static func _run_skidpad(cfg: VehicleConfigScript, map_id: int) -> Dictionary:
	var u0: float = 9.5
	var bundle: Dictionary = _init_model_and_inputs(cfg, map_id, u0)
	var model: VehicleDynamicsModelScript = bundle["model"]
	var inputs: VehicleInputsScript = bundle["inputs"]

	var center := Vector2(0.0, SKIDPAD_RADIUS_M)
	var ff_steer: float = atan(cfg.wheelbase / SKIDPAD_RADIUS_M)
	var prev_steer: float = ff_steer
	inputs.steer_angle = ff_steer
	inputs.steer_normalized = ff_steer / maxf(cfg.max_steer_angle, 1e-5)

	var total_steps: int = int(round(8.5 / SIM_DT))       # 2040 steps
	var settle_steps: int = int(round(0.25 / SIM_DT))
	var ss_start_step: int = int(round(4.5 / SIM_DT))     # final 4.0 s steady-state window

	var v_target: float = u0
	var v_target_max: float = 15.5
	var int_v: float = 0.0

	var finite_ok: bool = true
	var min_fz_n: float = 1e9
	var max_roll_deg: float = 0.0
	var peak_beta_deg: float = 0.0

	var ss_n: int = 0
	var ss_lat_g_sum: float = 0.0
	var ss_wz_err_sq_sum: float = 0.0
	var ss_orbit_omega_sum: float = 0.0
	var ss_r_err_abs_sum: float = 0.0
	var ss_dist_m: float = 0.0
	var total_dist_m: float = 0.0

	for k in range(total_steps):
		if not _is_state_finite(model):
			finite_ok = false
			break

		var pos_2d := Vector2(model.chassis_state.position.x, model.chassis_state.position.y)
		var yaw: float = model.chassis_state.yaw
		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		var speed: float = sqrt(vx * vx + vy * vy)

		var radial_vec: Vector2 = pos_2d - center
		var r_act: float = maxf(radial_vec.length(), 1.0)
		var radial_err: float = r_act - SKIDPAD_RADIUS_M
		var tangent := Vector2(-radial_vec.y, radial_vec.x).normalized()
		var heading := Vector2(cos(yaw), sin(yaw))
		var psi_err: float = atan2(heading.cross(tangent), heading.dot(tangent))

		var r_des: float = maxf(vx, 1.0) / SKIDPAD_RADIUS_M
		var e_r: float = r_des - wz
		var steer_raw: float = ff_steer + 0.85 * psi_err + 0.14 * radial_err + 0.18 * e_r
		var steer_clamped: float = clampf(steer_raw, -cfg.max_steer_angle, cfg.max_steer_angle)
		var steer_cmd: float = move_toward(prev_steer, steer_clamped, 3.5 * SIM_DT)
		prev_steer = steer_cmd

		var beta_deg_abs: float = absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
		if absf(radial_err) < 0.35 and beta_deg_abs < 7.5 and absf(e_r) < 0.28:
			v_target = minf(v_target_max, v_target + 1.10 * SIM_DT)
		else:
			var backoff: float = 0.65 + maxf(absf(radial_err) - 0.35, 0.0) * 2.5 + maxf(beta_deg_abs - 7.5, 0.0) * 0.45
			v_target = maxf(8.0, v_target - backoff * SIM_DT)

		var e_v: float = v_target - vx
		int_v = clampf(int_v + e_v * SIM_DT, -2.0, 2.0)
		if e_v >= -0.45:
			inputs.throttle = clampf(0.22 + 0.55 * e_v + 0.35 * int_v, 0.0, 1.0)
			inputs.brake = 0.0
		else:
			inputs.throttle = 0.0
			inputs.brake = clampf((-e_v - 0.45) * 0.45, 0.0, 0.6)

		inputs.steer_angle = steer_cmd
		inputs.steer_normalized = steer_cmd / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		total_dist_m += speed * SIM_DT

		if absf(radial_err) > 4.5 and k > settle_steps:
			finite_ok = false
			break

		if k >= settle_steps:
			var fz_min_step: float = _min_array_float(step_res["normal_forces"])
			min_fz_n = minf(min_fz_n, fz_min_step)
			max_roll_deg = maxf(max_roll_deg, absf(rad_to_deg(model.chassis_state.roll)))
			peak_beta_deg = maxf(peak_beta_deg, beta_deg_abs)

		if k >= ss_start_step:
			var v_world := Vector2(vx * cos(yaw) - vy * sin(yaw), vx * sin(yaw) + vy * cos(yaw))
			var v_tan: float = maxf(v_world.dot(tangent), 0.1)
			var orbit_omega: float = v_tan / r_act
			var wz_ref: float = model.ecu_tv_v3.last_wz_ref if model.ecu_tv_v3 != null else r_des
			var wz_err: float = wz_ref - wz

			ss_lat_g_sum += absf(vx * wz) / cfg.gravity
			ss_wz_err_sq_sum += wz_err * wz_err
			ss_orbit_omega_sum += orbit_omega
			ss_r_err_abs_sum += absf(radial_err)
			ss_dist_m += speed * SIM_DT
			ss_n += 1

	var completed: bool = finite_ok and ss_n > 20
	var mean_orbit_omega: float = (ss_orbit_omega_sum / float(maxi(ss_n, 1))) if ss_n > 0 else 0.1
	var mean_r_err: float = (ss_r_err_abs_sum / float(maxi(ss_n, 1))) if ss_n > 0 else 9.0
	var lap_time_s: float = (TAU / maxf(mean_orbit_omega, 0.1)) if completed else 99.0
	if mean_r_err > 1.2:
		lap_time_s += (mean_r_err - 1.2) * 2.0

	var energy_kwh: float = model.hv_state.energy_kwh if model.hv_state != null else 0.0
	return {
		"ok": completed,
		"skidpad_lap_time_s": lap_time_s,
		"skidpad_lat_g": (ss_lat_g_sum / float(maxi(ss_n, 1))) if ss_n > 0 else 0.0,
		"skidpad_wz_rmse": sqrt(ss_wz_err_sq_sum / float(maxi(ss_n, 1))) if ss_n > 0 else 9.9,
		"skidpad_peak_beta_deg": peak_beta_deg,
		"skidpad_min_fz_n": min_fz_n if min_fz_n < 1e8 else 0.0,
		"skidpad_max_roll_deg": max_roll_deg,
		"skidpad_mean_r_err_m": mean_r_err,
		"dist_m": total_dist_m,
		"energy_kwh": energy_kwh
	}


# ── Maneuver 2: Fast Slalom @ 65 km/h (18.06 m/s, 0.55 Hz, 5.5 s @ 240 Hz) ────
static func _run_slalom(cfg: VehicleConfigScript, map_id: int) -> Dictionary:
	var u0: float = 65.0 / 3.6  # 18.0556 m/s
	var bundle: Dictionary = _init_model_and_inputs(cfg, map_id, u0)
	var model: VehicleDynamicsModelScript = bundle["model"]
	var inputs: VehicleInputsScript = bundle["inputs"]

	var total_steps: int = int(round(5.5 / SIM_DT))       # 1320 steps
	var eval_start_step: int = int(round(0.40 / SIM_DT))  # after soft ramp-in
	var freq_hz: float = 0.55
	var wz_amp: float = 0.82

	var prev_steer: float = 0.0
	var int_v: float = 0.0
	var int_wz: float = 0.0

	var finite_ok: bool = true
	var min_fz_n: float = 1e9
	var max_roll_deg: float = 0.0
	var peak_beta_deg: float = 0.0
	var wz_err_sq_sum: float = 0.0
	var eval_n: int = 0
	var total_dist_m: float = 0.0

	var wz_ref_hist: Array[float] = []
	var wz_act_hist: Array[float] = []

	for k in range(total_steps):
		if not _is_state_finite(model):
			finite_ok = false
			break

		var t_now: float = float(k) * SIM_DT
		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		var speed: float = sqrt(vx * vx + vy * vy)

		var ramp: float = clampf(t_now / 0.35, 0.0, 1.0)
		var wz_target: float = ramp * wz_amp * sin(TAU * freq_hz * t_now)
		var delta_ff: float = (cfg.wheelbase / maxf(vx, 2.0)) * wz_target
		var e_wz: float = wz_target - wz
		int_wz = clampf(int_wz + e_wz * SIM_DT, -0.4, 0.4)
		var steer_raw: float = delta_ff + 0.16 * e_wz + 0.35 * int_wz
		var steer_clamped: float = clampf(steer_raw, -cfg.max_steer_angle, cfg.max_steer_angle)
		var steer_cmd: float = move_toward(prev_steer, steer_clamped, 4.5 * SIM_DT)
		prev_steer = steer_cmd

		var e_v: float = u0 - vx
		int_v = clampf(int_v + e_v * SIM_DT, -2.0, 2.0)
		inputs.throttle = clampf(0.24 + 0.45 * e_v + 0.25 * int_v, 0.0, 1.0)
		inputs.brake = 0.0
		inputs.steer_angle = steer_cmd
		inputs.steer_normalized = steer_cmd / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		total_dist_m += speed * SIM_DT

		if k >= eval_start_step:
			var fz_min_step: float = _min_array_float(step_res["normal_forces"])
			min_fz_n = minf(min_fz_n, fz_min_step)
			max_roll_deg = maxf(max_roll_deg, absf(rad_to_deg(model.chassis_state.roll)))
			var beta_deg_abs: float = absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
			peak_beta_deg = maxf(peak_beta_deg, beta_deg_abs)
			wz_err_sq_sum += e_wz * e_wz
			wz_ref_hist.append(wz_target)
			wz_act_hist.append(wz)
			eval_n += 1

	# Cross-correlation yaw response phase lag (agility metric in seconds)
	var max_lag_samples: int = mini(60, maxi(wz_ref_hist.size() / 4, 1))
	var best_lag: int = 0
	var best_corr: float = -1e18
	for lag in range(max_lag_samples):
		var corr: float = 0.0
		for idx in range(wz_ref_hist.size() - lag):
			corr += wz_ref_hist[idx] * wz_act_hist[idx + lag]
		if corr > best_corr:
			best_corr = corr
			best_lag = lag
	var agility_s: float = float(best_lag) * SIM_DT

	var completed: bool = finite_ok and eval_n > 20
	var energy_kwh: float = model.hv_state.energy_kwh if model.hv_state != null else 0.0
	return {
		"ok": completed,
		"slalom_wz_rmse_rads": sqrt(wz_err_sq_sum / float(maxi(eval_n, 1))) if eval_n > 0 else 9.9,
		"slalom_peak_beta_deg": peak_beta_deg,
		"slalom_min_fz_n": min_fz_n if min_fz_n < 1e8 else 0.0,
		"slalom_max_roll_deg": max_roll_deg,
		"slalom_agility_s": agility_s,
		"dist_m": total_dist_m,
		"energy_kwh": energy_kwh
	}


# ── Maneuver 3: Combined Trail Braking in Turn (90 -> 35 km/h @ 240 Hz) ───────
static func _run_trail_braking(cfg: VehicleConfigScript, map_id: int) -> Dictionary:
	var u0: float = 25.0                 # 90 km/h
	var v_end: float = 35.0 / 3.6        # 9.7222 m/s
	var bundle: Dictionary = _init_model_and_inputs(cfg, map_id, u0)
	var model: VehicleDynamicsModelScript = bundle["model"]
	var inputs: VehicleInputsScript = bundle["inputs"]

	var max_steps: int = int(round(4.0 / SIM_DT))
	var turn_in_s: float = 0.20

	var prev_steer: float = 0.0
	var finite_ok: bool = true
	var reached_v_end: bool = false

	var min_fz_n: float = 1e9
	var max_roll_deg: float = 0.0
	var peak_beta_deg: float = 0.0
	var rear_lock_margin: float = 1.0
	var brake_lockup_peak: float = 0.0
	var brake_time_s: float = 0.0
	var brake_dist_m: float = 0.0
	var total_dist_m: float = 0.0

	for k in range(max_steps):
		if not _is_state_finite(model):
			finite_ok = false
			break

		var t_now: float = float(k) * SIM_DT
		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		var speed: float = sqrt(vx * vx + vy * vy)

		if t_now >= turn_in_s and vx <= v_end:
			reached_v_end = true
			break

		var trail_frac: float = clampf((u0 - vx) / maxf(u0 - v_end, 1.0), 0.0, 1.0)
		var ramp_turn: float = clampf(t_now / turn_in_s, 0.0, 1.0)
		var kappa_ref: float = ramp_turn * lerpf(1.0 / 52.0, 1.0 / 26.0, trail_frac)
		var wz_target: float = vx * kappa_ref

		var delta_ff: float = atan(cfg.wheelbase * kappa_ref)
		var e_wz: float = wz_target - wz
		var steer_raw: float = delta_ff + 0.16 * e_wz
		var steer_cmd: float = move_toward(prev_steer, clampf(steer_raw, -cfg.max_steer_angle, cfg.max_steer_angle), 4.0 * SIM_DT)
		prev_steer = steer_cmd

		var brake_cmd: float = 0.0
		if t_now < turn_in_s:
			brake_cmd = lerpf(0.0, 1.00, t_now / turn_in_s)
		else:
			brake_cmd = lerpf(1.00, 0.50, trail_frac)

		inputs.throttle = 0.0
		inputs.brake = brake_cmd
		inputs.steer_angle = steer_cmd
		inputs.steer_normalized = steer_cmd / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		total_dist_m += speed * SIM_DT

		if t_now >= turn_in_s:
			brake_time_s += SIM_DT
			brake_dist_m += speed * SIM_DT

			var fz_min_step: float = _min_array_float(step_res["normal_forces"])
			min_fz_n = minf(min_fz_n, fz_min_step)
			max_roll_deg = maxf(max_roll_deg, absf(rad_to_deg(model.chassis_state.roll)))
			var beta_deg_abs: float = absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
			peak_beta_deg = maxf(peak_beta_deg, beta_deg_abs)

			var front_slip: float = maxf(0.0, -minf(model.wheels[0].kappa, model.wheels[1].kappa))
			var rear_slip: float = maxf(0.0, -minf(model.wheels[2].kappa, model.wheels[3].kappa))
			rear_lock_margin = minf(rear_lock_margin, front_slip - rear_slip)
			for wi in range(4):
				brake_lockup_peak = maxf(brake_lockup_peak, absf(model.wheels[wi].kappa))

	var completed: bool = finite_ok and reached_v_end
	if not reached_v_end:
		brake_time_s = 9.9
		brake_dist_m = 99.0

	var energy_kwh: float = model.hv_state.energy_kwh if model.hv_state != null else 0.0
	return {
		"ok": completed,
		"trail_brake_time_s": brake_time_s,
		"trail_brake_dist_m": brake_dist_m,
		"trail_peak_beta_deg": peak_beta_deg,
		"trail_min_fz_n": min_fz_n if min_fz_n < 1e8 else 0.0,
		"trail_max_roll_deg": max_roll_deg,
		"rear_lock_margin": rear_lock_margin,
		"brake_lockup_peak": brake_lockup_peak,
		"dist_m": total_dist_m,
		"energy_kwh": energy_kwh
	}


static func _run_high_speed_corner(cfg: VehicleConfigScript, map_id: int) -> Dictionary:
	const HIGH_SPEED_RADIUS_M: float = 32.0
	const HIGH_SPEED_MPS: float = 26.389 # 95 km/h
	var bundle: Dictionary = _init_model_and_inputs(cfg, map_id, HIGH_SPEED_MPS)
	var model: VehicleDynamicsModelScript = bundle["model"]
	var inputs: VehicleInputsScript = bundle["inputs"]

	var total_s: float = 4.5
	var settle_s: float = 1.5
	var n_steps: int = int(round(total_s / SIM_DT))

	var prev_steer: float = 0.0
	var finite_ok: bool = true

	var wz_err_sq_sum: float = 0.0
	var meas_count: int = 0
	var min_fz_n: float = 1e9
	var max_roll_deg: float = 0.0
	var max_beta_deg: float = 0.0
	var dist_m: float = 0.0

	var kappa_ref: float = 1.0 / HIGH_SPEED_RADIUS_M
	var delta_ff: float = atan(cfg.wheelbase * kappa_ref)

	for k in range(n_steps):
		if not _is_state_finite(model):
			finite_ok = false
			break

		var t_now: float = float(k) * SIM_DT
		var vx: float = model.chassis_state.linear_velocity.x
		var vy: float = model.chassis_state.linear_velocity.y
		var wz: float = model.chassis_state.angular_velocity.z
		var speed: float = sqrt(vx * vx + vy * vy)

		var ramp: float = clampf(t_now / 0.6, 0.0, 1.0)
		var wz_target: float = ramp * HIGH_SPEED_MPS * kappa_ref
		var e_wz: float = wz_target - wz
		var steer_raw: float = ramp * delta_ff + 0.18 * e_wz
		var steer_cmd: float = move_toward(prev_steer, clampf(steer_raw, -cfg.max_steer_angle, cfg.max_steer_angle), 3.5 * SIM_DT)
		prev_steer = steer_cmd

		var v_err: float = HIGH_SPEED_MPS - speed
		var throttle_cmd: float = clampf(0.28 + 0.35 * v_err, 0.0, 1.0)
		inputs.throttle = throttle_cmd
		inputs.brake = 0.0
		inputs.steer_angle = steer_cmd
		inputs.steer_normalized = steer_cmd / maxf(cfg.max_steer_angle, 1e-5)

		var step_res: Dictionary = model.step_standalone(inputs, SIM_DT, 0.0)
		dist_m += speed * SIM_DT

		if t_now >= settle_s:
			meas_count += 1
			wz_err_sq_sum += e_wz * e_wz
			var fz_min_step: float = _min_array_float(step_res["normal_forces"])
			min_fz_n = minf(min_fz_n, fz_min_step)
			max_roll_deg = maxf(max_roll_deg, absf(rad_to_deg(model.chassis_state.roll)))
			var beta_deg_abs: float = absf(VehicleTelemetryScript.compute_beta_deg(vx, vy))
			max_beta_deg = maxf(max_beta_deg, beta_deg_abs)

	var wz_rmse: float = sqrt(wz_err_sq_sum / float(maxi(meas_count, 1))) if finite_ok else 9.9
	var energy_kwh: float = model.hv_state.energy_kwh if model.hv_state != null else 0.0
	return {
		"ok": finite_ok and meas_count > 0,
		"high_speed_beta_max_deg": max_beta_deg,
		"high_speed_wz_rmse": wz_rmse,
		"high_speed_min_fz_n": min_fz_n if min_fz_n < 1e8 else 0.0,
		"high_speed_max_roll_deg": max_roll_deg,
		"dist_m": dist_m,
		"energy_kwh": energy_kwh
	}


static func evaluate_candidate(cli: Dictionary, t0_usec: int) -> Dictionary:
	var built: Dictionary = _build_configured_vehicle(cli)
	var cfg: VehicleConfigScript = built["config"]
	var params_echo: Dictionary = built["params"]
	var map_id: int = int(cli.get("map_id", DEFAULT_TV_MAP_ID))

	var m1: Dictionary = _run_skidpad(cfg, map_id)
	var m2: Dictionary = _run_slalom(cfg, map_id)
	var m3: Dictionary = _run_trail_braking(cfg, map_id)
	var m4: Dictionary = _run_high_speed_corner(cfg, map_id)

	var ok: bool = bool(m1["ok"]) and bool(m2["ok"]) and bool(m3["ok"]) and bool(m4["ok"])
	var min_fz_n: float = minf(
		minf(float(m1["skidpad_min_fz_n"]), float(m2["slalom_min_fz_n"])),
		minf(float(m3["trail_min_fz_n"]), float(m4["high_speed_min_fz_n"]))
	)
	var max_roll_deg: float = maxf(
		maxf(float(m1["skidpad_max_roll_deg"]), float(m2["slalom_max_roll_deg"])),
		maxf(float(m3["trail_max_roll_deg"]), float(m4["high_speed_max_roll_deg"]))
	)
	var peak_beta_deg: float = maxf(
		maxf(float(m1["skidpad_peak_beta_deg"]), float(m2["slalom_peak_beta_deg"])),
		maxf(float(m3["trail_peak_beta_deg"]), float(m4["high_speed_beta_max_deg"]))
	)
	var high_speed_beta_max_deg: float = float(m4["high_speed_beta_max_deg"])
	var high_speed_wz_rmse: float = float(m4["high_speed_wz_rmse"])
	var brake_lockup_peak: float = float(m3["brake_lockup_peak"])

	var total_energy_kwh: float = float(m1["energy_kwh"]) + float(m2["energy_kwh"]) + float(m3["energy_kwh"]) + float(m4["energy_kwh"])
	var total_dist_km: float = maxf((float(m1["dist_m"]) + float(m2["dist_m"]) + float(m3["dist_m"]) + float(m4["dist_m"])) / 1000.0, 1e-6)
	var efficiency_kwh_km: float = total_energy_kwh / total_dist_km

	var penalty: float = 0.0
	if not ok:
		penalty += 10000.0
	if min_fz_n < 110.0:
		penalty += 1000.0 + 30.0 * (110.0 - min_fz_n)
	if max_roll_deg > 1.6:
		penalty += 1000.0 + 400.0 * (max_roll_deg - 1.6)
	if peak_beta_deg > 4.5:
		penalty += 1000.0 + 250.0 * (peak_beta_deg - 4.5)
	if high_speed_beta_max_deg > 3.5:
		penalty += 1000.0 + 300.0 * (high_speed_beta_max_deg - 3.5)
	if brake_lockup_peak > 0.16:
		penalty += 1000.0 + 2500.0 * (brake_lockup_peak - 0.16)

	var feasible: bool = (
		ok
		and min_fz_n >= 110.0
		and max_roll_deg <= 1.6
		and peak_beta_deg <= 4.5
		and high_speed_beta_max_deg <= 3.5
		and brake_lockup_peak <= 0.16
	)
	var eval_time_s: float = float(Time.get_ticks_usec() - t0_usec) * 1e-6

	return {
		"ok": ok,
		"feasible": feasible,
		"penalty": penalty,
		"eval_time_s": eval_time_s,
		"min_fz_n": min_fz_n,
		"max_roll_deg": max_roll_deg,
		"peak_beta_deg": peak_beta_deg,
		"high_speed_beta_max_deg": high_speed_beta_max_deg,
		"high_speed_wz_rmse": high_speed_wz_rmse,
		"brake_lockup_peak": brake_lockup_peak,
		"skidpad_lap_time_s": float(m1["skidpad_lap_time_s"]),
		"skidpad_lat_g": float(m1["skidpad_lat_g"]),
		"skidpad_wz_rmse": float(m1["skidpad_wz_rmse"]),
		"skidpad_peak_beta_deg": float(m1["skidpad_peak_beta_deg"]),
		"skidpad_min_fz_n": float(m1["skidpad_min_fz_n"]),
		"skidpad_max_roll_deg": float(m1["skidpad_max_roll_deg"]),
		"slalom_wz_rmse_rads": float(m2["slalom_wz_rmse_rads"]),
		"slalom_peak_beta_deg": float(m2["slalom_peak_beta_deg"]),
		"slalom_min_fz_n": float(m2["slalom_min_fz_n"]),
		"slalom_max_roll_deg": float(m2["slalom_max_roll_deg"]),
		"slalom_agility_s": float(m2["slalom_agility_s"]),
		"trail_brake_time_s": float(m3["trail_brake_time_s"]),
		"trail_brake_dist_m": float(m3["trail_brake_dist_m"]),
		"trail_peak_beta_deg": float(m3["trail_peak_beta_deg"]),
		"trail_min_fz_n": float(m3["trail_min_fz_n"]),
		"trail_max_roll_deg": float(m3["trail_max_roll_deg"]),
		"rear_lock_margin": float(m3["rear_lock_margin"]),
		"efficiency_kwh_km": efficiency_kwh_km,
		"params": params_echo
	}
