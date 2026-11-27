extends SceneTree

const VehicleConfig = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const VehicleDynamicsModel = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const BrakeModel = preload("res://Scenes/Vehicle/Dynamics/Brakes/BrakeModel.gd")
const TrackScript = preload("res://Scenes/Track/track.gd")
const VehicleTelemetry = preload("res://Scenes/Vehicle/Dynamics/Telemetry/VehicleTelemetry.gd")
const TelemetryRecorder = preload("res://Scenes/Vehicle/Dynamics/Telemetry/TelemetryRecorder.gd")


func _init() -> void:
	var res: Dictionary = run()
	var passed: int = int(res.get("passed", 0))
	var failed: int = int(res.get("failed", 0))
	var errors: Array = res.get("errors", [])
	print("\n[%s] Passed: %d | Failed: %d" % [res.get("name", "test_brake_blending_and_grip_map"), passed, failed])
	for err in errors:
		print("  -> ERROR: %s" % err)
	quit(0 if failed == 0 else 1)


static func _load_json_dict(path: String) -> Dictionary:
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var parsed: Variant = JSON.parse_string(f.get_as_text())
	f.close()
	return parsed if parsed is Dictionary else {}


static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	# 1. brakes.json parameters & compute_brake_torques backward compatibility
	var brakes_json: Dictionary = _load_json_dict("res://Scenes/Vehicle/Config/vehicles/ter27_4wd/brakes.json")
	var required_brake_keys: Array[String] = [
		"brake_bias_front", "brake_mu", "regen_blending_enabled", "p_max_charge_w",
		"regen_min_speed_mps", "disc_ambient_temp_c", "disc_max_temp_c",
		"disc_thermal_mass_j_k", "disc_conv_base_w_k", "disc_conv_speed_coeff"
	]
	var missing_keys: Array[String] = []
	for k in required_brake_keys:
		if not brakes_json.has(k):
			missing_keys.append(k)
	var cfg: VehicleConfig = VehicleConfig.new()
	if missing_keys.is_empty() and is_equal_approx(cfg.brake_bias_front, 0.58) and is_equal_approx(cfg.brake_mu, 1.30):
		passed += 1
	else:
		failed += 1
		errors.append("brakes.json config mismatch: missing=%s bias=%.4f mu=%.2f" % [
			str(missing_keys), cfg.brake_bias_front, cfg.brake_mu
		])

	var inp: VehicleInputs = VehicleInputs.new()
	inp.set_from_manual(0.0, 1.0, 0.0, cfg.max_steer_angle)
	var legacy_trq: Array[float] = BrakeModel.compute_brake_torques(inp, cfg)
	var leg_f: float = legacy_trq[0] + legacy_trq[1]
	var leg_r: float = legacy_trq[2] + legacy_trq[3]
	if absf((leg_f / maxf(leg_f + leg_r, 1e-5)) - 0.58) < 1e-4:
		passed += 1
	else:
		failed += 1
		errors.append("Legacy compute_brake_torques bias mismatch: F=%.2f R=%.2f" % [leg_f, leg_r])

	# 2. AP Racing RQ3 pad_mu_scale curve
	var mu_40: float = BrakeModel.compute_pad_mu_scale(40.0)
	var mu_120: float = BrakeModel.compute_pad_mu_scale(120.0)
	var mu_350: float = BrakeModel.compute_pad_mu_scale(350.0)
	var mu_500: float = BrakeModel.compute_pad_mu_scale(500.0)
	var mu_585: float = BrakeModel.compute_pad_mu_scale(585.0)
	var mu_650: float = BrakeModel.compute_pad_mu_scale(650.0)
	if (
		is_equal_approx(mu_40, 0.94)
		and is_equal_approx(mu_120, 1.0)
		and is_equal_approx(mu_350, 1.0)
		and is_equal_approx(mu_500, 1.0)
		and mu_585 < 1.0 and mu_585 > 0.78
		and is_equal_approx(mu_650, 0.78)
	):
		passed += 1
	else:
		failed += 1
		errors.append("compute_pad_mu_scale curve invalid: [40:%.3f, 120:%.3f, 350:%.3f, 500:%.3f, 585:%.3f, 650:%.3f]" % [
			mu_40, mu_120, mu_350, mu_500, mu_585, mu_650
		])

	# 3. Brake blending respects p_max_charge_w = 25 kW, min speed cutoff, and unloaded wheel Fz limit
	var vx: float = 25.0
	var omega_25: float = vx / cfg.wheel_radius
	var omegas: Array = [omega_25, omega_25, omega_25, omega_25]
	var fz_nom: Array = [900.0, 900.0, 700.0, 700.0]
	var bs_regen: BrakeModel.BrakeState = BrakeModel.BrakeState.new()
	var res_hi: Dictionary = BrakeModel.compute_blended_braking(inp, cfg, bs_regen, omegas, fz_nom, vx, 0.01, {"soc": 0.80})
	var p_reg: float = float(res_hi["p_regen_w"])
	var reg_trqs: Array = res_hi["regen_torques"]
	var hyd_trqs: Array = res_hi["hydraulic_torques"]
	if p_reg > 1000.0 and p_reg <= 25000.0 + 1e-2 and reg_trqs[0] > 0.0 and hyd_trqs[0] >= 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Charge power ceiling violated: p_regen_w=%.2f W reg=%s hyd=%s" % [p_reg, str(reg_trqs), str(hyd_trqs)])

	# Low speed cutoff (< 1.5 m/s) & unloaded wheel check
	var bs_low: BrakeModel.BrakeState = BrakeModel.BrakeState.new()
	var res_low: Dictionary = BrakeModel.compute_blended_braking(inp, cfg, bs_low, [4.0, 4.0, 4.0, 4.0], fz_nom, 1.0, 0.01)
	var bs_fz0: BrakeModel.BrakeState = BrakeModel.BrakeState.new()
	var res_fz0: Dictionary = BrakeModel.compute_blended_braking(inp, cfg, bs_fz0, omegas, [0.0, 900.0, 700.0, 700.0], vx, 0.01)
	if is_zero_approx(float(res_low["p_regen_w"])) and is_zero_approx(float(res_low["regen_torques"][0])) and is_zero_approx(float(res_fz0["regen_torques"][0])):
		passed += 1
	else:
		failed += 1
		errors.append("Min speed or Fz=0 regen cutoff failed: p_low=%.3f reg_fz0=%.3f" % [
			float(res_low["p_regen_w"]), float(res_fz0["regen_torques"][0])
		])

	# High SOC (>= 0.995) derates regen to prevent overcharge
	var bs_soc_ok: BrakeModel.BrakeState = BrakeModel.BrakeState.new()
	var bs_soc_hi: BrakeModel.BrakeState = BrakeModel.BrakeState.new()
	var r_soc_ok: Dictionary = BrakeModel.compute_blended_braking(inp, cfg, bs_soc_ok, omegas, fz_nom, vx, 0.01, {"soc": 0.80})
	var r_soc_hi: Dictionary = BrakeModel.compute_blended_braking(inp, cfg, bs_soc_hi, omegas, fz_nom, vx, 0.01, {"soc": 0.996})
	if float(r_soc_hi["p_regen_w"]) < float(r_soc_ok["p_regen_w"]) * 0.5:
		passed += 1
	else:
		failed += 1
		errors.append("SOC >= 0.995 did not reduce regen: ok=%.1f W hi=%.1f W" % [
			float(r_soc_ok["p_regen_w"]), float(r_soc_hi["p_regen_w"])
		])

	# 4. Regen reduces hydraulic torque and disc temperature rise vs pure hydraulic
	var cfg_noregen: Dictionary = {
		"brake_mu": 1.30, "brake_bias_front": 0.58, "regen_blending_enabled": false,
		"p_max_charge_w": 25000.0, "regen_min_speed_mps": 1.5,
		"disc_ambient_temp_c": 40.0, "disc_max_temp_c": 650.0,
		"disc_thermal_mass_j_k": 1350.0, "disc_conv_base_w_k": 12.0, "disc_conv_speed_coeff": 1.8,
		"mass": cfg.mass, "gravity": cfg.gravity, "wheel_radius": cfg.wheel_radius,
		"motor_peak_torque": cfg.motor_peak_torque, "gear_ratio": cfg.gear_ratio
	}
	var bs_blend: BrakeModel.BrakeState = BrakeModel.BrakeState.new(40.0)
	var bs_hyd: BrakeModel.BrakeState = BrakeModel.BrakeState.new(40.0)
	var inp_mod: VehicleInputs = VehicleInputs.new()
	inp_mod.set_from_manual(0.0, 0.65, 0.0, cfg.max_steer_angle)
	var last_blend: Dictionary = {}
	var last_hyd: Dictionary = {}
	for _s in range(100): # 1.0 s @ 100 Hz
		last_blend = BrakeModel.compute_blended_braking(inp_mod, cfg, bs_blend, omegas, fz_nom, vx, 0.01)
		last_hyd = BrakeModel.compute_blended_braking(inp_mod, cfg_noregen, bs_hyd, omegas, fz_nom, vx, 0.01)

	if (
		float(last_blend["hydraulic_torques"][0]) < float(last_hyd["hydraulic_torques"][0])
		and bs_blend.disc_temp[0] > 40.0
		and bs_blend.disc_temp[0] < bs_hyd.disc_temp[0]
	):
		passed += 1
	else:
		failed += 1
		errors.append("Regen thermal/hydraulic reduction failed: T_blend=%.2f T_hyd=%.2f hyd_b=%.1f hyd_h=%.1f" % [
			bs_blend.disc_temp[0], bs_hyd.disc_temp[0],
			float(last_blend["hydraulic_torques"][0]), float(last_hyd["hydraulic_torques"][0])
		])

	# 5. Disc heating upper clamp (650 C) and convective cooling (faster at speed, >= 40 C)
	var bs_hot: BrakeModel.BrakeState = BrakeModel.BrakeState.new(645.0)
	BrakeModel.compute_blended_braking(inp, cfg_noregen, bs_hot, [250.0, 250.0, 250.0, 250.0], fz_nom, 5.0, 2.0)
	var clamped_top: bool = is_equal_approx(bs_hot.disc_temp[0], 650.0)

	var bs_cool_fast: BrakeModel.BrakeState = BrakeModel.BrakeState.new(400.0)
	var bs_cool_still: BrakeModel.BrakeState = BrakeModel.BrakeState.new(400.0)
	var inp_coast: VehicleInputs = VehicleInputs.new()
	for _c in range(50): # 0.5 s
		BrakeModel.compute_blended_braking(inp_coast, cfg, bs_cool_fast, omegas, fz_nom, 25.0, 0.01)
		BrakeModel.compute_blended_braking(inp_coast, cfg, bs_cool_still, [0.0, 0.0, 0.0, 0.0], fz_nom, 0.0, 0.01)
	var cool_ok: bool = (
		bs_cool_fast.disc_temp[0] >= 40.0
		and bs_cool_fast.disc_temp[0] < 400.0
		and bs_cool_fast.disc_temp[0] < bs_cool_still.disc_temp[0]
	)
	if clamped_top and cool_ok:
		passed += 1
	else:
		failed += 1
		errors.append("Thermal clamp/cooling failed: top=%.2f fast_cool=%.2f still_cool=%.2f" % [
			bs_hot.disc_temp[0], bs_cool_fast.disc_temp[0], bs_cool_still.disc_temp[0]
		])

	# 6. Track.eval_grip_zones primitives (circle, annulus, rect, smoothstep feathering)
	var test_zones: Array = [
		{"type": "circle", "cx": 0.0, "cz": 0.0, "radius": 5.0, "mu_scale": 1.0, "feather_m": 1.0},
		{"type": "rect", "min_x": 10.0, "max_x": 14.0, "min_z": -2.0, "max_z": 2.0, "mu_scale": 1.05, "feather_m": 1.0},
		{"type": "annulus", "cx": -20.0, "cz": 0.0, "inner_radius": 4.0, "outer_radius": 7.0, "mu_scale": 0.96, "feather_m": 1.0}
	]
	var mu_circle_in: float = TrackScript.eval_grip_zones(Vector3(0.0, 0.0, 0.0), test_zones, 0.86)
	var mu_circle_edge: float = TrackScript.eval_grip_zones(Vector3(5.5, 0.0, 0.0), test_zones, 0.86)
	var mu_outside: float = TrackScript.eval_grip_zones(Vector3(8.0, 0.0, 8.0), test_zones, 0.86)
	var mu_rect_in: float = TrackScript.eval_grip_zones(Vector3(12.0, 0.0, 0.0), test_zones, 0.86)
	var mu_ann_in: float = TrackScript.eval_grip_zones(Vector3(-14.5, 0.0, 0.0), test_zones, 0.86)
	var mu_ann_hole: float = TrackScript.eval_grip_zones(Vector3(-20.0, 0.0, 0.0), test_zones, 0.86)
	if (
		is_equal_approx(mu_circle_in, 1.0)
		and mu_circle_edge > 0.86 and mu_circle_edge < 1.0
		and is_equal_approx(mu_outside, 0.86)
		and is_equal_approx(mu_rect_in, 1.05)
		and is_equal_approx(mu_ann_in, 0.96)
		and is_equal_approx(mu_ann_hole, 0.86)
	):
		passed += 1
	else:
		failed += 1
		errors.append("eval_grip_zones primitive mismatch: c_in=%.3f c_edge=%.3f out=%.3f rect=%.3f ann_in=%.3f ann_hole=%.3f" % [
			mu_circle_in, mu_circle_edge, mu_outside, mu_rect_in, mu_ann_in, mu_ann_hole
		])

	# 7. skidpad.json & acceleration.json surface_grip parsing and spatial evaluation
	var trk: TrackScript = TrackScript.new()
	var skid_data: Dictionary = _load_json_dict("res://TrackFiles/skidpad.json")
	trk.parse_grip_map(skid_data)
	var skid_lane_circle: float = trk.get_surface_mu_at(Vector3(0.0, 0.0, 18.35))
	var skid_lane_center: float = trk.get_surface_mu_at(Vector3(0.0, 0.0, 0.0))
	var skid_dust_inner: float = trk.get_surface_mu_at(Vector3(0.0, 0.0, 9.25))
	var skid_dust_outer: float = trk.get_surface_mu_at(Vector3(0.0, 0.0, 24.0))

	var accel_data: Dictionary = _load_json_dict("res://TrackFiles/acceleration.json")
	trk.parse_grip_map(accel_data)
	var accel_lane: float = trk.get_surface_mu_at(Vector3(0.0, 0.0, 0.0))
	var accel_dust: float = trk.get_surface_mu_at(Vector3(5.0, 0.0, 0.0))
	trk.free()

	if (
		is_equal_approx(skid_lane_circle, 1.0)
		and is_equal_approx(skid_lane_center, 1.0)
		and skid_dust_inner >= 0.84 and skid_dust_inner <= 0.88
		and skid_dust_outer >= 0.84 and skid_dust_outer <= 0.88
		and is_equal_approx(accel_lane, 1.0)
		and accel_dust >= 0.84 and accel_dust <= 0.88
	):
		passed += 1
	else:
		failed += 1
		errors.append("Track JSON grip map mismatch: skid=[%.3f, %.3f, %.3f, %.3f] accel=[%.3f, %.3f]" % [
			skid_lane_circle, skid_lane_center, skid_dust_inner, skid_dust_outer, accel_lane, accel_dust
		])

	# 8. VehicleTelemetry & TelemetryRecorder new brake/grip fields + CSV export
	var model: VehicleDynamicsModel = VehicleDynamicsModel.new(cfg)
	var tel: VehicleTelemetry = VehicleTelemetry.new()
	var step_extra: Dictionary = {
		"dt": 0.005,
		"forces": {
			"drive_torques": [0.0, 0.0, 0.0, 0.0],
			"brake_torques": last_blend["total_torques"],
			"disc_temps": bs_blend.disc_temp,
			"p_regen_w": bs_blend.p_regen_w,
			"surface_mu": [1.0, 1.0, 0.86, 0.86],
			"brake_bias_front": 0.58,
			"regen_blending_enabled": true
		}
	}
	tel.update_from_model(model, Vector3(-5.0, 0.0, 0.0), 0.0, step_extra)
	var d_tel: Dictionary = tel.to_dict()
	var tel_ok: bool = (
		is_equal_approx(float(d_tel.get("brake_bias_front", 0.0)), 0.58)
		and bool(d_tel.get("regen_blending_enabled", false)) == true
		and is_equal_approx(float(d_tel.get("disc_temp_fl", 0.0)), bs_blend.disc_temp[0])
		and is_equal_approx(float(d_tel.get("surface_mu_rl", 0.0)), 0.86)
		and is_equal_approx(float(d_tel.get("regen_power_kw", -1.0)), bs_blend.p_regen_w * 0.001)
	)
	tel.reset()
	var reset_ok: bool = (
		is_equal_approx(tel.disc_temp[0], 40.0)
		and is_equal_approx(tel.surface_mu[2], 1.0)
		and is_zero_approx(tel.regen_power_kw)
	)

	var rec: TelemetryRecorder = TelemetryRecorder.new()
	rec.sample_rate_hz = 200.0
	var csv_path: String = "user://telemetry/_test_brake_blending_grip.csv"
	var opened: String = rec.start_recording("skidpad", "test", csv_path)
	rec.record_step(0.005, model, inp_mod, step_extra["forces"])
	var closed: String = rec.stop_recording()
	var csv_ok: bool = false
	if opened != "" and closed != "" and rec.get_sample_count() == 1:
		var f: FileAccess = FileAccess.open(closed, FileAccess.READ)
		if f != null:
			var hdr: String = f.get_line()
			var row1: String = f.get_line()
			f.close()
			DirAccess.remove_absolute(ProjectSettings.globalize_path(closed))
			var hdr_cols: PackedStringArray = hdr.split(",")
			var row_cols: PackedStringArray = row1.split(",")
			csv_ok = (
				hdr == TelemetryRecorder.CSV_HEADER
				and hdr_cols.size() == row_cols.size()
				and hdr.find("brake_bias_front") != -1
				and hdr.find("regen_blending_enabled") != -1
				and hdr.find("disc_temp_fl") != -1
				and hdr.find("surface_mu_rr") != -1
				and hdr.find("regen_power_kw") != -1
			)

	if tel_ok and reset_ok and csv_ok:
		passed += 1
	else:
		failed += 1
		errors.append("Telemetry/CSV contract mismatch: tel_ok=%s reset_ok=%s csv_ok=%s" % [
			str(tel_ok), str(reset_ok), str(csv_ok)
		])

	return {
		"name": "test_brake_blending_and_grip_map",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
