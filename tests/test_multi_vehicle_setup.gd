extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const SetupProfileScript = preload("res://Scenes/Vehicle/Config/SetupProfile.gd")
const BrakeModelScript = preload("res://Scenes/Vehicle/Dynamics/Brakes/BrakeModel.gd")
const EcuTvScript = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoringV3.gd")
const ValidatorScript = preload("../vd_correlation_lab/analysis/benchmarks/ter26_telemetry_validator.gd")


static func _check(cond: bool, msg: String, state: Dictionary) -> void:
	if cond:
		state["passed"] = int(state["passed"]) + 1
	else:
		state["failed"] = int(state["failed"]) + 1
		(state["errors"] as Array).append(msg)


static func run() -> Dictionary:
	var st := {"name": "TestMultiVehicleSetup", "passed": 0, "failed": 0, "errors": []}

	# 1. vehicle discovery
	var vehicles := VehicleConfigScript.list_available_vehicles()
	var ids: Array[String] = []
	for v in vehicles:
		ids.append(String(v.get("vehicle_id", "")))
	_check(ids.has("ter26_rwd"), "Discovery includes ter26_rwd", st)
	_check(ids.has("ter27_4wd"), "Discovery includes ter27_4wd", st)

	# 2. hot-swap between ter26_rwd and ter27_4wd + setups
	var model := VehicleDynamicsModelScript.new()
	_check(model.load_vehicle("ter26_rwd", "default"), "load_vehicle ter26_rwd default succeeds", st)
	_check(is_equal_approx(model.config.mass, 300.0), "ter26_rwd mass == 300 kg", st)
	_check(model.config.drive_mode == VehicleConfigScript.DriveMode.REAR_WD, "ter26_rwd drive_mode == REAR_WD", st)
	_check(model.ecu_tv_v3.drive_layout == EcuTvScript.DriveLayout.RWD_2WD, "ter26_rwd ECU drive_layout == RWD_2WD", st)
	_check(is_equal_approx(model.config.tire_pressure_front_bar, 0.72), "ter26_rwd default pressure == 0.72 bar", st)
	_check(is_equal_approx(model.wheels[0].inflation_pressure_kpa, 72.0), "FL wheel pressure synced to 72 kPa", st)

	var rwd_maps := model.ecu_tv_v3.list_available_tv_maps()
	var rwd_map_files: Array[String] = []
	for m in rwd_maps:
		rwd_map_files.append(String(m.get("filename", "")))
	_check(rwd_map_files.has("01_ter26_race_rwd"), "ter26_rwd exposes 01_ter26_race_rwd TV map", st)

	_check(model.apply_setup("autocross_wet"), "Apply autocross_wet setup on ter26_rwd", st)
	_check(is_equal_approx(model.config.spring_stiffness_front, 28000.0), "autocross_wet front spring == 28000 N/m", st)
	_check(is_equal_approx(model.config.power_limit_watts, 65000.0), "autocross_wet power limit == 65 kW", st)
	_check(is_equal_approx(model.wheels[2].inflation_pressure_kpa, 85.0), "autocross_wet RL pressure == 85 kPa", st)
	_check(model.ecu_tv_v3.control_law_mode == EcuTvScript.ControlLawMode.V3_PID_ALQP, "autocross_wet selects V3 control law", st)

	# save & reload custom setupprofile in user://
	var custom_prof := SetupProfileScript.from_vehicle_config(model.config, "test_custom_rwd", "01_ter26_race_rwd.json", "smc")
	custom_prof.brake_bias = 0.65
	var user_dir := "user://setups/ter26_rwd"
	DirAccess.make_dir_recursive_absolute(user_dir)
	var custom_path := user_dir + "/test_custom_rwd.json"
	var cf := FileAccess.open(custom_path, FileAccess.WRITE)
	if cf != null:
		cf.store_string(JSON.stringify(custom_prof.to_dict(), "\t") + "\n")
		cf.close()
	var reloaded := SetupProfileScript.load_for_vehicle("ter26_rwd", "test_custom_rwd")
	_check(reloaded != null and is_equal_approx(reloaded.brake_bias, 0.65), "SetupProfile round-trip preserves custom brake_bias", st)
	if FileAccess.file_exists(custom_path):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(custom_path))

	_check(model.load_vehicle("ter27_4wd", "endurance"), "Hot-swap back to ter27_4wd endurance", st)
	_check(is_equal_approx(model.config.mass, 290.0), "ter27_4wd mass == 290 kg", st)
	_check(model.config.drive_mode == VehicleConfigScript.DriveMode.FOUR_WD, "ter27_4wd drive_mode == FOUR_WD", st)
	_check(model.ecu_tv_v3.drive_layout == EcuTvScript.DriveLayout.AWD_4WD, "ter27_4wd ECU drive_layout == AWD_4WD", st)
	_check(is_equal_approx(model.config.power_limit_watts, 56000.0), "ter27_4wd endurance power limit == 56 kW", st)

	# 3. strict rwd physics proof (acceleration, cornering tv mz, launch control, and regen braking)
	model.load_vehicle("ter26_rwd", "default")
	var v_init := 12.0
	model.chassis_state.linear_velocity = Vector3(v_init, 0.0, 0.0)
	for i in range(4):
		model.wheels[i].omega = v_init / model.config.wheel_radius
		model.ecu_tv_v3.tc_omega_prev_ema[i] = model.wheels[i].omega

	var inp := VehicleInputsScript.new()
	inp.drive_mode = model.config.drive_mode
	inp.ecu_driving_mode = 1
	inp.throttle = 1.0
	inp.brake = 0.0
	inp.steer_angle = 0.16 # left turn -> demands mz > 0 -> rr > rl

	var max_front_drive := 0.0
	var rear_diff_seen := 0.0
	for _k in range(60): # 0.15 s at 400 hz
		model.step_standalone(inp, 0.0025)
		var tc: Array[float] = model.ecu_tv_v3.last_t_cmd
		max_front_drive = maxf(max_front_drive, maxf(absf(tc[0]), absf(tc[1])))
		rear_diff_seen = maxf(rear_diff_seen, absf(tc[3] - tc[2]))

	_check(is_zero_approx(max_front_drive), "ter26_rwd strictly outputs T_FL = T_FR = 0.0 under full drive + TV", st)
	_check(model.ecu_tv_v3.last_t_cmd[2] > 20.0 or model.ecu_tv_v3.last_t_cmd[3] > 20.0, "ter26_rwd rear axle carries positive drive torque", st)
	_check(rear_diff_seen > 15.0, "ter26_rwd generates differential Mz across RL/RR (diff=%.1f Nm)" % rear_diff_seen, st)

	# launch control on ter26_rwd must also keep front motors at 0.0
	model.reset()
	_check(model.ecu_tv_v3.request_launch_control_arm(0.0, 1.0), "ter26_rwd arms Launch Control cleanly", st)
	inp.ecu_driving_mode = 1
	inp.drive_mode = model.config.drive_mode
	inp.throttle = 0.0
	inp.brake = 1.0
	inp.steer_angle = 0.0
	model.step_standalone(inp, 0.005)
	_check(
		is_zero_approx(model.ecu_tv_v3.last_t_cmd[0]) and is_zero_approx(model.ecu_tv_v3.last_t_cmd[1]) and model.ecu_tv_v3.last_t_cmd[2] > 1.0,
		"ter26_rwd LC_ARMED preload keeps front motors at 0.0 and preloads rear",
		st
	)
	inp.throttle = 1.0
	inp.brake = 0.0
	for _k in range(15):
		model.step_standalone(inp, 0.005)
	_check(
		model.ecu_tv_v3.launch_state == EcuTvScript.LaunchState.LC_ACTIVE and is_zero_approx(model.ecu_tv_v3.last_t_cmd[0]) and is_zero_approx(model.ecu_tv_v3.last_t_cmd[1]) and model.ecu_tv_v3.last_t_cmd[2] > 50.0,
		"ter26_rwd LC_ACTIVE launch keeps front motors strictly at 0.0 Nm (state=%s, t_cmd=%s)" % [model.ecu_tv_v3.get_launch_state_name(), str(model.ecu_tv_v3.last_t_cmd)],
		st
	)

	# regenerative braking on ter26_rwd: fl/fr must receive 0.0 regen while rl/rr regen > 0
	var b_state := BrakeModelScript.BrakeState.new(model.config.disc_ambient_temp_c)
	var b_inp := VehicleInputsScript.new()
	b_inp.drive_mode = VehicleConfigScript.DriveMode.REAR_WD
	b_inp.brake = 0.55
	var w_omegas: Array[float] = [90.0, 90.0, 90.0, 90.0]
	var w_fz: Array[float] = [650.0, 650.0, 600.0, 600.0]
	var b_res := BrakeModelScript.compute_blended_braking(b_inp, model.config, b_state, w_omegas, w_fz, 18.0, 0.005, model.hv_state)
	var regen: Array = b_res.get("regen_torques", [0.0, 0.0, 0.0, 0.0])
	_check(is_zero_approx(float(regen[0])) and is_zero_approx(float(regen[1])), "ter26_rwd front axle has zero regen torque", st)
	_check(float(regen[2]) > 10.0 and float(regen[3]) > 10.0, "ter26_rwd rear axle performs blended regenerative braking", st)

	# 4. ter26 open-loop telemetry replay & rmse validator
	var ref_model := VehicleDynamicsModelScript.new()
	ref_model.load_vehicle("ter26_rwd", "default")
	var v0 := 14.0
	ref_model.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	for i in range(4):
		ref_model.wheels[i].omega = v0 / ref_model.config.wheel_radius
		ref_model.ecu_tv_v3.tc_omega_prev_ema[i] = ref_model.wheels[i].omega
	var ref_inp := VehicleInputsScript.new()
	ref_inp.drive_mode = ref_model.config.drive_mode
	ref_inp.ecu_driving_mode = 1

	var synth_csv_path := "user://test_ter26_synth_log.csv"
	var out_csv_path := "user://test_ter26_cmp_out.csv"
	var out_json_path := "user://test_ter26_metrics.json"
	var sf := FileAccess.open(synth_csv_path, FileAccess.WRITE)
	if sf != null:
		sf.store_line("time_s,steer_deg,throttle,brake,vx_ms,ay_ms2,wz_rads,wheel_rpm_fl,wheel_rpm_fr,wheel_rpm_rl,wheel_rpm_rr")
		for k in range(51): # 0.50 s at 100 hz
			var t_s := float(k) * 0.01
			var steer_deg := 6.0 * sin(t_s * 3.0)
			var throttle := 0.55 if k < 40 else 0.0
			var brake := 0.0 if k < 40 else 0.40
			if k > 0:
				ref_inp.steer_angle = deg_to_rad(6.0 * sin((t_s - 0.01) * 3.0))
				ref_inp.throttle = 0.55 if (k - 1) < 40 else 0.0
				ref_inp.brake = 0.0 if (k - 1) < 40 else 0.40
				for _s in range(4):
					ref_model.step_standalone(ref_inp, 0.0025)
			var vx_r := ref_model.chassis_state.linear_velocity.x
			var ay_r := ref_model.net_force_chassis_ros.y / ref_model.config.mass
			var wz_r := ref_model.chassis_state.angular_velocity.z
			sf.store_line("%.4f,%.4f,%.3f,%.3f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.2f" % [
				t_s, steer_deg, throttle, brake, vx_r, ay_r, wz_r,
				ref_model.wheels[0].omega * 60.0 / TAU,
				ref_model.wheels[1].omega * 60.0 / TAU,
				ref_model.wheels[2].omega * 60.0 / TAU,
				ref_model.wheels[3].omega * 60.0 / TAU,
			])
		sf.close()

	var val_res := ValidatorScript.run_validation_from_file(synth_csv_path, "ter26_rwd", "default", out_csv_path, out_json_path)
	_check(bool(val_res.get("ok", false)), "TeR26TelemetryValidator succeeds on synthetic log", st)
	_check(int(val_res.get("sample_count", 0)) == 51, "Validator processes all 51 CSV samples", st)
	_check(FileAccess.file_exists(out_csv_path), "Comparison CSV file exported", st)
	_check(FileAccess.file_exists(out_json_path), "RMSE summary JSON exported", st)
	var rmse: Dictionary = val_res.get("rmse", {})
	_check(float(rmse.get("vx_ms", 999.0)) < 0.15, "Self-consistency vx RMSE < 0.15 m/s (got %.4f)" % float(rmse.get("vx_ms", 999.0)), st)
	_check(float(rmse.get("wz_rads", 999.0)) < 0.05, "Self-consistency wz RMSE < 0.05 rad/s (got %.4f)" % float(rmse.get("wz_rads", 999.0)), st)

	for tmp in [synth_csv_path, out_csv_path, out_json_path]:
		if FileAccess.file_exists(tmp):
			DirAccess.remove_absolute(ProjectSettings.globalize_path(tmp))

	return st
