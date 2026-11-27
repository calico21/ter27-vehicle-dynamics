extends RefCounted

const InputConditionerScript = preload("res://Scenes/Vehicle/Dynamics/Core/InputConditioner.gd")
const MirenaCarScene = preload("res://Scenes/Vehicle/mirena_car.tscn")
const SimUIScene = preload("res://Scenes/Sim/UI/SimUI.tscn")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func run() -> Dictionary:
	var st := {"name": "test_input_conditioning", "passed": 0, "failed": 0, "errors": []}
	var max_steer_rad := 0.380  # ~21.77 deg TeR27
	var wb := 1.540

	# 1. WHEEL_DIRECT strictly 1:1 linear at any speed
	var cond := InputConditionerScript.new()
	cond.set_profile(InputConditionerScript.InputProfile.WHEEL_DIRECT)
	var wheel_ok := true
	for v in [0.0, 5.5, 19.44, 30.0]:
		for x in [-1.0, -0.5, -0.02, 0.0, 0.02, 0.5, 1.0]:
			var res: Dictionary = cond.step(x, 0.73, 0.42, v, 0.5, 0.05, 0.0025, max_steer_rad, wb, InputConditionerScript.InputProfile.WHEEL_DIRECT)
			if absf(float(res["steer_angle"]) - x * max_steer_rad) > 1e-6:
				wheel_ok = false
			if absf(float(res["gas"]) - 0.73) > 1e-6 or absf(float(res["brake"]) - 0.42) > 1e-6:
				wheel_ok = false
	_check(wheel_ok, "WHEEL_DIRECT is strictly 1:1 linear across all speeds", st)

	# 2. GAMEPAD reaches >= 95% of steer_max in < 0.15 s at 20 km/h (5.56 m/s)
	cond.reset()
	cond.set_profile(InputConditionerScript.InputProfile.GAMEPAD)
	var out_20 := 0.0
	for _k in range(58):  # 58 * 0.0025 = 0.145 s < 0.15 s
		var r20: Dictionary = cond.step(1.0, 0.0, 0.0, 20.0 / 3.6, 0.0, 0.0, 0.0025, max_steer_rad, wb, InputConditionerScript.InputProfile.GAMEPAD)
		out_20 = float(r20["steer_angle"])
	_check(out_20 >= 0.95 * max_steer_rad, "GAMEPAD reaches >=95%% of steer_max in <0.15s at 20 km/h (got %.4f / %.4f rad)" % [out_20, max_steer_rad], st)

	# 3. GAMEPAD at 70 km/h caps step below 21.8 deg while keeping >= 1.30 * delta_lim_70kmh (~6.0 deg) & fast countersteer
	cond.reset()
	var cap_70: float = InputConditionerScript.compute_gamepad_steer_cap_rad(70.0 / 3.6, max_steer_rad, wb)
	var delta_lim_65 := deg_to_rad(6.0)
	_check(cap_70 < max_steer_rad * 0.75 and cap_70 >= 1.30 * delta_lim_65, "GAMEPAD 70 km/h cap=%.2f deg (<16.3 deg, >=1.30x 6.0 deg)" % rad_to_deg(cap_70), st)

	var out_70 := 0.0
	for _k in range(80):  # 0.20 s full stick step at 70 km/h
		var r70: Dictionary = cond.step(1.0, 0.0, 0.0, 70.0 / 3.6, 0.5, 0.02, 0.0025, max_steer_rad, wb, InputConditionerScript.InputProfile.GAMEPAD)
		out_70 = float(r70["steer_angle"])
	_check(out_70 <= cap_70 + 1e-4 and out_70 >= 1.30 * delta_lim_65, "GAMEPAD 70 km/h full stick settles at %.2f deg without 21.8 deg spike" % rad_to_deg(out_70), st)

	# Instant countersteer when stick * wz < 0
	var pre_cs := out_70
	var dt_cs := 0.020
	var r_cs: Dictionary = cond.step(-1.0, 0.0, 0.0, 70.0 / 3.6, 1.2, 0.08, dt_cs, max_steer_rad, wb, InputConditionerScript.InputProfile.GAMEPAD)
	var delta_cs := pre_cs - float(r_cs["steer_angle"])
	var rate_cs := delta_cs / dt_cs
	_check(rate_cs >= 5.5 - 1e-3, "GAMEPAD countersteer rate >= 5.5 rad/s (got %.2f rad/s)" % rate_cs, st)

	# 4. Brake strictly linear 1:1 after 0.04 deadzone & no NaNs
	var b0 := InputConditionerScript.shape_gamepad_brake(0.0)
	var b_dz := InputConditionerScript.shape_gamepad_brake(0.04)
	var b35 := InputConditionerScript.shape_gamepad_brake(0.35)
	var b675 := InputConditionerScript.shape_gamepad_brake(0.675)
	var b100 := InputConditionerScript.shape_gamepad_brake(1.00)
	var brk_ok := (
		is_finite(b0) and is_finite(b_dz) and is_finite(b35) and is_finite(b675) and is_finite(b100)
		and absf(b0) < 1e-6 and absf(b_dz) < 1e-6
		and absf(b35 - ((0.35 - 0.04) / 0.96)) < 1e-5
		and absf(b675 - 0.5 * (b35 + b100)) < 1e-5
		and absf(b100 - 1.0) < 1e-5
	)
	_check(brk_ok, "GAMEPAD linear 1:1 brake shaping after 0.04 deadzone preserves exact linearity", st)

	# 4b. Throttle immediate convex shaping (1.15*u - 0.15*u^2) after 0.04 deadzone & no NaNs
	var t0 := InputConditionerScript.shape_gamepad_throttle(0.0)
	var t_dz := InputConditionerScript.shape_gamepad_throttle(0.04)
	var t52 := InputConditionerScript.shape_gamepad_throttle(0.52)  # u = 0.50 -> 0.5375
	var t100 := InputConditionerScript.shape_gamepad_throttle(1.00)
	var r_gp_th: Dictionary = cond.step(0.0, 0.52, 0.0, 10.0, 0.0, 0.0, 0.0025, max_steer_rad, wb, InputConditionerScript.InputProfile.GAMEPAD)
	var thr_ok := (
		is_finite(t0) and is_finite(t_dz) and is_finite(t52) and is_finite(t100)
		and absf(t0) < 1e-6 and absf(t_dz) < 1e-6
		and absf(t52 - 0.5375) < 1e-5
		and t52 > 0.50
		and absf(t100 - 1.0) < 1e-5
		and absf(float(r_gp_th["gas"]) - t52) < 1e-5
	)
	_check(thr_ok, "GAMEPAD convex throttle shaping (1.15u - 0.15u^2) after 0.04 deadzone", st)

	# 5. SimUI Input Profile OptionButton has FOCUS_NONE & updates MirenaCarDynamics
	var sim_ui := SimUIScene.instantiate()
	var main_menu := sim_ui.get_node_or_null("MainMenu")
	if main_menu != null and main_menu.has_method("_ready"):
		main_menu._ready()
	var ip_opt := main_menu.get("_input_profile_opt") as OptionButton
	_check(ip_opt != null and ip_opt.item_count == 4 and ip_opt.focus_mode == Control.FOCUS_NONE, "SimUI Input Profile OptionButton exists (4 items) with FOCUS_NONE", st)
	sim_ui.free()

	var car := MirenaCarScene.instantiate()
	car.set_input_profile(InputConditionerScript.InputProfile.WHEEL_DIRECT)
	_check(car.get_input_profile() == InputConditionerScript.InputProfile.WHEEL_DIRECT, "MirenaCarDynamics set_input_profile updates profile", st)
	car.free()

	return st
