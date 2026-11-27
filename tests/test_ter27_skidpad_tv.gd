extends RefCounted

const MirenaCarScene = preload("res://Scenes/Vehicle/mirena_car.tscn")
const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


static func _check(cond: bool, msg: String, st: Dictionary) -> void:
	if cond:
		st["passed"] = int(st["passed"]) + 1
	else:
		st["failed"] = int(st["failed"]) + 1
		(st["errors"] as Array).append(msg)


static func run() -> Dictionary:
	var st := {"name": "test_ter27_skidpad_tv", "passed": 0, "failed": 0, "errors": []}
	var cfg := VehicleConfigScript.new()
	cfg.load_vehicle_configs("ter27_4wd")
	var model := Full14DofScript.new(cfg)
	model.set_track_grip_profile(2)  # FSG_RUBBERED (lambda_muy = 0.57 @ 75 C)
	_check(absf(model.env_mu_scale - (0.57 / 0.47)) < 1e-4, "FSG_RUBBERED env_mu_scale matches 0.57/0.47 (got %.5f)" % model.env_mu_scale, st)

	var inp := VehicleInputsScript.new()
	inp.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = model.ecu_tv_4wd.TestMode.SKIDPAD if model.ecu_tv_4wd != null else 1
	inp.throttle = 0.55
	inp.brake = 0.0
	inp.steer_angle = 0.24

	# 1. Steady-state SKIDPAD_MAX_ATTACK_V2 stability, ay, beta, & HV compliance in 14-DOF (FSG_RUBBERED)
	model.reset()
	if model.ecu_tv_4wd != null:
		model.ecu_tv_4wd.active_mode = model.ecu_tv_4wd.TestMode.SKIDPAD
	model.chassis_state.linear_velocity = Vector3(12.0, 0.0, 0.0)
	for w in model.wheels:
		w.omega = 12.0 / cfg.wheel_radius

	var hv_bad := 0
	var nan_bad := 0
	var max_ay_g := 0.0
	var max_beta_deg := 0.0
	var ss_outer_gt_inner := 0
	for i in range(480):
		model.step_standalone(inp, 0.0025, 0.0)
		var p_kw := float(model.ecu_tv_4wd.last_power_w) * 0.001
		if p_kw > 80.05 or p_kw < -25.05:
			hv_bad += 1
		var vx_raw := model.chassis_state.linear_velocity.x
		var vy := model.chassis_state.linear_velocity.y
		var wz := model.chassis_state.angular_velocity.z
		if is_nan(vx_raw) or is_nan(vy) or is_nan(wz):
			nan_bad += 1
			continue
		var vx := maxf(absf(vx_raw), 1.0)
		max_ay_g = maxf(max_ay_g, absf(vx * wz) / 9.81)
		max_beta_deg = maxf(max_beta_deg, absf(rad_to_deg(atan2(vy, vx))))
		if i >= 360 and model.ecu_tv_4wd != null and model.ecu_tv_4wd.last_t_cmd.size() >= 4:
			var t_rl := float(model.ecu_tv_4wd.last_t_cmd[2])
			var t_rr := float(model.ecu_tv_4wd.last_t_cmd[3])
			if t_rr > t_rl and t_rl >= 0.0:
				ss_outer_gt_inner += 1

	_check(nan_bad == 0, "SKIDPAD_MAX_ATTACK_V2 14-DOF 0 NaN states (bad=%d)" % nan_bad, st)
	_check(hv_bad == 0, "SKIDPAD_MAX_ATTACK_V2 14-DOF 0 HV violations (bad=%d)" % hv_bad, st)
	_check(max_ay_g >= 1.25 and max_beta_deg < 6.5, "SKIDPAD_MAX_ATTACK_V2 FSG_RUBBERED sustains high ay=%.3fg with controlled beta=%.2f deg" % [max_ay_g, max_beta_deg], st)
	_check(ss_outer_gt_inner > 0, "SKIDPAD_MAX_ATTACK_V2 steady-state outer rear torque > inner rear torque >= 0 (hits=%d)" % ss_outer_gt_inner, st)

	# 2. Figure-8 Crossover reversal check at Skidpad speed
	model.chassis_state.linear_velocity = Vector3(10.5, 0.0, 0.0)
	for w in model.wheels:
		w.omega = 10.5 / cfg.wheel_radius
	inp.throttle = 0.45
	inp.steer_angle = -0.24
	var cr_hv_bad := 0
	var cr_max_beta_deg := 0.0
	for _j in range(480):
		model.step_standalone(inp, 0.0025, 0.0)
		var p_kw := float(model.ecu_tv_4wd.last_power_w) * 0.001
		if p_kw > 80.05 or p_kw < -25.05:
			cr_hv_bad += 1
		var vx := maxf(absf(model.chassis_state.linear_velocity.x), 1.0)
		var vy := model.chassis_state.linear_velocity.y
		cr_max_beta_deg = maxf(cr_max_beta_deg, absf(rad_to_deg(atan2(vy, vx))))
	var wz_after := model.chassis_state.angular_velocity.z
	_check(cr_hv_bad == 0 and wz_after < -0.5, "Figure-8 Crossover reverses yaw cleanly (wz=%.3f rad/s, hv_bad=%d)" % [wz_after, cr_hv_bad], st)
	_check(cr_max_beta_deg < 10.0, "Figure-8 Crossover keeps sideslip controlled (max|beta|=%.2f deg)" % cr_max_beta_deg, st)

	return st
