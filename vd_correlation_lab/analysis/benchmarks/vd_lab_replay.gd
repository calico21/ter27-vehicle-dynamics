extends SceneTree
class_name VdLabReplay

## herramienta headless multi-vehículo pa vd_correlation_lab
## ejecuta replay en lazo abierto / horizonte 10s o genera maniobras sintéticas de referencia
## uso: godot --headless --path src/mirena_sim -s res://tools/vd_lab_replay.gd -- --batch <batch.json>

const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const Fast3DofVehicleModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/fast_3dof/Fast3DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		var full_args := OS.get_cmdline_args()
		var idx := full_args.find("--")
		if idx >= 0:
			args = full_args.slice(idx + 1)
	var opts := _parse_cli_args(args)
	if opts.has("batch"):
		var ok := _run_batch(String(opts["batch"]))
		quit(0 if ok else 1)
		return
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


static func _load_json(path: String) -> Variant:
	if not FileAccess.file_exists(path):
		return null
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return null
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	return parsed


static func _load_csv(csv_path: String) -> Array[Dictionary]:
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
	for i in range(headers.size()):
		headers[i] = headers[i].strip_edges().to_lower()
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line == "" or line.begins_with("#"):
			continue
		var parts := line.split(",")
		if parts.size() < 4:
			continue
		var row: Dictionary = {}
		for j in range(mini(headers.size(), parts.size())):
			row[headers[j]] = float(parts[j].strip_edges())
		rows.append(row)
	f.close()
	return rows


static func _sync_state_from_row(model, row: Dictionary) -> void:
	var r0 := maxf(model.config.wheel_radius, 0.1)
	var vx0 := maxf(float(row.get("vx_ms", 0.0)), 0.0)
	var wz0 := float(row.get("wz_rads", 0.0))
	model.chassis_state.linear_velocity = Vector3(vx0, 0.0, 0.0)
	model.chassis_state.angular_velocity = Vector3(0.0, 0.0, wz0)
	if row.has("wheel_rpm_fl") and row.has("wheel_rpm_fr") and row.has("wheel_rpm_rl") and row.has("wheel_rpm_rr"):
		model.wheels[0].omega = float(row["wheel_rpm_fl"]) * TAU / 60.0
		model.wheels[1].omega = float(row["wheel_rpm_fr"]) * TAU / 60.0
		model.wheels[2].omega = float(row["wheel_rpm_rl"]) * TAU / 60.0
		model.wheels[3].omega = float(row["wheel_rpm_rr"]) * TAU / 60.0
	else:
		var w0 := vx0 / r0
		for w in model.wheels:
			w.omega = w0
	for w in model.wheels:
		w.kappa = 0.0
		w.alpha = 0.0
		w.kappa_transient = 0.0
		w.alpha_transient = 0.0


static func _apply_overrides(model, overrides: Dictionary) -> void:
	if overrides.is_empty() or model == null or model.config == null:
		return
	var cfg = model.config
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


static func _run_replay_job(job: Dictionary) -> bool:
	var vehicle_id := String(job.get("vehicle_id", "ter26_rwd"))
	var setup_name := String(job.get("setup", "default"))
	var input_csv := String(job.get("input_csv", ""))
	var out_csv := String(job.get("out_csv", ""))
	var horizon_s := float(job.get("horizon_s", 0.0))
	var ecu_mode := int(job.get("ecu_mode", 0))
	var tv_map_id := int(job.get("tv_map_id", 2))
	var overrides: Dictionary = job.get("overrides", {}) if job.get("overrides") is Dictionary else {}

	var rows := _load_csv(input_csv)
	if rows.size() < 2:
		push_error("VdLabReplay: CSV vacío o insuficiente: " + input_csv)
		return false

	var fidelity := String(job.get("fidelity", "full_14dof")).to_lower()
	var model = Fast3DofVehicleModelScript.new() if fidelity == "fast_3dof" else VehicleDynamicsModelScript.new()
	if not model.load_vehicle(vehicle_id, setup_name):
		push_error("VdLabReplay: no se pudo cargar vehículo " + vehicle_id)
		return false
	if not overrides.is_empty():
		_apply_overrides(model, overrides)

	var mass := maxf(model.config.mass, 100.0)
	var max_steer := deg_to_rad(maxf(model.config.max_steer_angle, 20.0))
	var base_motor_peak: float = float(model.config.motor_peak_torque)
	var inp := VehicleInputsScript.new()
	inp.drive_mode = model.config.drive_mode
	inp.tv_map_id = tv_map_id
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(false)

	var out_dir := out_csv.get_base_dir()
	if out_dir != "":
		DirAccess.make_dir_recursive_absolute(out_dir)
	var f_out := FileAccess.open(out_csv, FileAccess.WRITE)
	if f_out == null:
		push_error("VdLabReplay: no se pudo abrir salida " + out_csv)
		return false

	f_out.store_line(",".join([
		"t_s", "vx_sim", "vy_sim", "ax_sim", "ay_sim", "wz_sim",
		"roll_deg_sim", "pitch_deg_sim",
		"wheel_rpm_fl_sim", "wheel_rpm_fr_sim", "wheel_rpm_rl_sim", "wheel_rpm_rr_sim",
		"kappa_fl", "kappa_fr", "kappa_rl", "kappa_rr",
		"alpha_fl_deg", "alpha_fr_deg", "alpha_rl_deg", "alpha_rr_deg",
		"fz_fl", "fz_fr", "fz_rl", "fz_rr",
		"fx_fl", "fx_fr", "fx_rl", "fx_rr",
		"fy_fl", "fy_fr", "fy_rl", "fy_rr",
		"susp_mm_fl", "susp_mm_fr", "susp_mm_rl", "susp_mm_rr",
		"damper_vel_fl", "damper_vel_fr", "damper_vel_rl", "damper_vel_rr",
		"f_downforce_n",
		"torque_fl", "torque_fr", "torque_rl", "torque_rr",
		"tv_mz_nm", "tv_wz_ref_rads", "tc_active"
	]))

	model.reset()
	_sync_state_from_row(model, rows[0])
	var filtered_steer := clampf(deg_to_rad(float(rows[0].get("steer_deg", 0.0))), -max_steer, max_steer)
	var steer_tau_s := float(job.get("steer_tau_s", 0.025))
	var standstill_elapsed := 0.0
	var max_sub_dt := 0.005 if fidelity == "fast_3dof" else 0.004

	for i in range(rows.size()):
		var cur := rows[i]
		var t_cur := float(cur.get("t_s", float(i) * 0.01))
		if i > 0:
			var prev := rows[i - 1]
			var t_prev := float(prev.get("t_s", float(i - 1) * 0.01))
			var dt_step := clampf(t_cur - t_prev, 0.001, 0.1)

			if prev.has("trq_limit_nm") and float(prev["trq_limit_nm"]) > 10.0:
				model.config.motor_peak_torque = base_motor_peak * clampf(float(prev["trq_limit_nm"]) / 120.0, 0.2, 1.5)
			else:
				model.config.motor_peak_torque = base_motor_peak

			var target_steer := clampf(deg_to_rad(float(prev.get("steer_deg", 0.0))), -max_steer, max_steer)
			if steer_tau_s > 1e-4:
				filtered_steer = lerpf(filtered_steer, target_steer, 1.0 - exp(-dt_step / steer_tau_s))
			else:
				filtered_steer = target_steer
			inp.steer_angle = filtered_steer
			inp.throttle = clampf(float(prev.get("throttle", 0.0)), 0.0, 1.0)
			inp.brake = clampf(float(prev.get("brake", 0.0)), 0.0, 1.0)
			inp.tv_mode = ecu_mode

			var n_sub := maxi(1, int(ceili(dt_step / max_sub_dt)))
			var sub_dt := dt_step / float(n_sub)
			for _s in range(n_sub):
				model.step_standalone(inp, sub_dt)

			var real_vx_cur := float(cur.get("vx_ms", 0.0))
			if real_vx_cur < 0.35 and inp.throttle < 0.02:
				standstill_elapsed += dt_step
				if standstill_elapsed >= 0.40:
					model.chassis_state.linear_velocity = Vector3(real_vx_cur, 0.0, 0.0)
					model.chassis_state.angular_velocity = Vector3.ZERO
					for wi in range(4):
						var re_i := maxf(model.wheels[wi].effective_radius, 0.05)
						model.wheels[wi].omega = real_vx_cur / re_i
						model.wheels[wi].kappa = 0.0
						model.wheels[wi].kappa_transient = 0.0
			else:
				standstill_elapsed = 0.0

			if horizon_s > 0.0:
				var tau_h := maxf(horizon_s * 0.25, 1.5)
				var alpha_h := 1.0 - exp(-dt_step / tau_h)
				var dv_corr: float = (real_vx_cur - float(model.chassis_state.linear_velocity.x)) * alpha_h
				model.chassis_state.linear_velocity.x += dv_corr
				for wi in range(4):
					var re_w := maxf(model.wheels[wi].effective_radius, 0.05)
					model.wheels[wi].omega += dv_corr / re_w

		var svx: float = float(model.chassis_state.linear_velocity.x)
		var svy: float = float(model.chassis_state.linear_velocity.y)
		var sax: float = float(model.net_force_chassis_ros.x) / mass
		var say: float = float(model.net_force_chassis_ros.y) / mass
		var swz: float = float(model.chassis_state.angular_velocity.z)
		var roll_deg := rad_to_deg(model.chassis_state.roll)
		var pitch_deg := rad_to_deg(model.chassis_state.pitch)
		var rpm_k := 60.0 / TAU
		var downforce: float = 0.5 * float(model.config.air_density) * float(model.config.cl_a) * svx * absf(svx)

		var tq := [0.0, 0.0, 0.0, 0.0]
		if model.hv_state != null and model.hv_state.inverter_torques.size() >= 4:
			for k in range(4):
				tq[k] = float(model.hv_state.inverter_torques[k])

		var mz_req := 0.0
		var wz_ref := 0.0
		var tc_act := 0
		if model.ecu_tv_v3 != null:
			mz_req = model.ecu_tv_v3.last_mz_req
			wz_ref = model.ecu_tv_v3.last_wz_ref
			tc_act = 1 if model.ecu_tv_v3.tc_active else 0

		var vals := [
			"%.4f" % t_cur, "%.5f" % svx, "%.5f" % svy, "%.5f" % sax, "%.5f" % say, "%.5f" % swz,
			"%.4f" % roll_deg, "%.4f" % pitch_deg,
			"%.3f" % (model.wheels[0].omega * rpm_k), "%.3f" % (model.wheels[1].omega * rpm_k),
			"%.3f" % (model.wheels[2].omega * rpm_k), "%.3f" % (model.wheels[3].omega * rpm_k),
			"%.5f" % model.wheels[0].kappa, "%.5f" % model.wheels[1].kappa,
			"%.5f" % model.wheels[2].kappa, "%.5f" % model.wheels[3].kappa,
			"%.4f" % rad_to_deg(model.wheels[0].alpha), "%.4f" % rad_to_deg(model.wheels[1].alpha),
			"%.4f" % rad_to_deg(model.wheels[2].alpha), "%.4f" % rad_to_deg(model.wheels[3].alpha),
			"%.2f" % model.wheels[0].F_z, "%.2f" % model.wheels[1].F_z,
			"%.2f" % model.wheels[2].F_z, "%.2f" % model.wheels[3].F_z,
			"%.2f" % model.wheels[0].F_x, "%.2f" % model.wheels[1].F_x,
			"%.2f" % model.wheels[2].F_x, "%.2f" % model.wheels[3].F_x,
			"%.2f" % model.wheels[0].F_y, "%.2f" % model.wheels[1].F_y,
			"%.2f" % model.wheels[2].F_y, "%.2f" % model.wheels[3].F_y,
			"%.4f" % (model.suspension.corners[0].delta_z * 1000.0),
			"%.4f" % (model.suspension.corners[1].delta_z * 1000.0),
			"%.4f" % (model.suspension.corners[2].delta_z * 1000.0),
			"%.4f" % (model.suspension.corners[3].delta_z * 1000.0),
			"%.5f" % model.suspension.corners[0].delta_z_dot,
			"%.5f" % model.suspension.corners[1].delta_z_dot,
			"%.5f" % model.suspension.corners[2].delta_z_dot,
			"%.5f" % model.suspension.corners[3].delta_z_dot,
			"%.2f" % downforce,
			"%.3f" % tq[0], "%.3f" % tq[1], "%.3f" % tq[2], "%.3f" % tq[3],
			"%.3f" % mz_req, "%.5f" % wz_ref, "%d" % tc_act
		]
		f_out.store_line(",".join(vals))

	f_out.close()
	model.config.motor_peak_torque = base_motor_peak
	return true


static func _generate_synthetic_log(job: Dictionary) -> bool:
	var vehicle_id := String(job.get("vehicle_id", "ter27_4wd"))
	var out_csv := String(job.get("out_csv", ""))
	var duration_s := float(job.get("duration_s", 40.0))
	var dt := 0.01
	var n_steps := int(round(duration_s / dt))

	var fidelity := String(job.get("fidelity", "full_14dof")).to_lower()
	var model = Fast3DofVehicleModelScript.new() if fidelity == "fast_3dof" else VehicleDynamicsModelScript.new()
	if not model.load_vehicle(vehicle_id, "default"):
		return false

	var mass := maxf(model.config.mass, 100.0)
	var max_steer := deg_to_rad(maxf(model.config.max_steer_angle, 20.0))
	var inp := VehicleInputsScript.new()
	inp.drive_mode = model.config.drive_mode
	inp.tv_map_id = int(job.get("tv_map_id", 2))
	inp.tv_mode = int(job.get("ecu_mode", 2))
	if model.ecu_tv_v3 != null:
		model.ecu_tv_v3.set_hardware_realism(false)

	var out_dir := out_csv.get_base_dir()
	if out_dir != "":
		DirAccess.make_dir_recursive_absolute(out_dir)
	var f := FileAccess.open(out_csv, FileAccess.WRITE)
	if f == null:
		return false

	f.store_line("t_s,vx_ms,ax_ms2,ay_ms2,wz_rads,steer_deg,throttle,brake,trq_limit_nm,wheel_rpm_fl,wheel_rpm_fr,wheel_rpm_rl,wheel_rpm_rr")
	model.reset()
	var rng := RandomNumberGenerator.new()
	rng.seed = 27042026

	for i in range(n_steps):
		var t := float(i) * dt
		# perfil de maniobra realista formula student (dentro de la envolvente ~1.5 g)
		var thr := 0.0
		var brk := 0.0
		var str_deg := 0.0
		if t < 2.2:
			thr = clampf((t - 0.2) / 0.5, 0.0, 0.80)
		elif t < 4.2:
			brk = 0.45
		elif t < 14.0:
			thr = 0.36 + 0.08 * sin(t * 1.1)
			str_deg = 5.5 * sin((t - 4.2) * 1.45)
		elif t < 22.0:
			thr = 0.42
			str_deg = 6.2 * clampf((t - 14.0) / 1.0, 0.0, 1.0)
		elif t < 25.0:
			brk = clampf((t - 22.0) / 0.4, 0.0, 0.65)
			str_deg = 6.2 * clampf(1.0 - (t - 22.0) / 2.0, 0.0, 1.0)
		elif t < 34.0:
			thr = 0.45
			str_deg = -5.5 * sin((t - 25.0) * 0.9)
		else:
			brk = clampf((t - 34.0) / 0.5, 0.0, 0.70)

		inp.steer_angle = clampf(deg_to_rad(str_deg), -max_steer, max_steer)
		inp.throttle = thr
		inp.brake = brk

		for _s in range(5):
			model.step_standalone(inp, dt / 5.0)

		var vx := maxf(model.chassis_state.linear_velocity.x, 0.0)
		var ax: float = float(model.net_force_chassis_ros.x) / mass + rng.randfn(0.0, 0.08)
		var ay: float = float(model.net_force_chassis_ros.y) / mass + rng.randfn(0.0, 0.10)
		var wz: float = float(model.chassis_state.angular_velocity.z) + rng.randfn(0.0, 0.012)
		var rpm_k := 60.0 / TAU
		f.store_line(",".join([
			"%.4f" % t, "%.5f" % vx, "%.5f" % ax, "%.5f" % ay, "%.5f" % wz,
			"%.4f" % str_deg, "%.4f" % thr, "%.4f" % brk, "120.0",
			"%.3f" % (model.wheels[0].omega * rpm_k), "%.3f" % (model.wheels[1].omega * rpm_k),
			"%.3f" % (model.wheels[2].omega * rpm_k), "%.3f" % (model.wheels[3].omega * rpm_k),
		]))

	f.close()
	return true


func _run_batch(batch_path: String) -> bool:
	var data = _load_json(batch_path)
	if not (data is Array):
		push_error("VdLabReplay: batch JSON inválido: " + batch_path)
		return false
	for item in data:
		if not (item is Dictionary):
			continue
		var kind := String(item.get("kind", "replay"))
		var ok := false
		if kind == "synthetic":
			ok = _generate_synthetic_log(item)
		else:
			ok = _run_replay_job(item)
		if not ok:
			return false
	return true
