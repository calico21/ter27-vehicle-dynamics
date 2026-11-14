extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const EcuTvV3Script = preload("res://Scenes/Vehicle/Dynamics/Drivetrain/EcuTorqueVectoringV3.gd")
const SetupManagerScript = preload("res://Scenes/Vehicle/Dynamics/Core/SetupManager.gd")
const DilSessionLoggerScript = preload("res://Scenes/Vehicle/Dynamics/Core/DilSessionLogger.gd")
const MirenaCarScript = preload("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd")

static func _check(cond: bool, msg: String, res: Dictionary) -> void:
	if cond:
		res["passed"] += 1
	else:
		res["failed"] += 1
		res["errors"].append(msg)

static func run() -> Dictionary:
	var res := {
		"name": "DiL Suite (SetupManager + Logger + 3D Wheels)",
		"passed": 0,
		"failed": 0,
		"errors": []
	}

	# 1. SetupManager: load & apply unified preset + diff detection
	var cfg := VehicleConfigScript.new()
	var ecu := EcuTvV3Script.new()
	var mgr := SetupManagerScript.new()
	var presets: Array[Dictionary] = mgr.list_presets()
	_check(presets.size() >= 4, "Expected at least 4 base unified presets, got %d" % presets.size(), res)

	var base_path := "res://Scenes/Vehicle/Dynamics/Core/Config/setups/01_baseline_autocross.json"
	var reset_state := {"called": false}
	var loaded_ok := mgr.load_preset(base_path, cfg, ecu, func(): reset_state["called"] = true)
	_check(loaded_ok, "SetupManager.load_preset should succeed for 01_baseline_autocross.json", res)
	_check(bool(reset_state["called"]), "SetupManager.load_preset should invoke reset callback when provided", res)
	_check(is_equal_approx(cfg.spring_stiffness_front, 45000.0), "Expected spring_stiffness_front=45000 N/m (45 N/mm), got %.1f" % cfg.spring_stiffness_front, res)
	_check(is_equal_approx(cfg.brake_bias_front, 0.58), "Expected brake_bias_front=0.58, got %.3f" % cfg.brake_bias_front, res)

	var base_dict := SetupManagerScript.capture_from_car(cfg, ecu)
	var diffs_clean := mgr.compute_diffs_vs_base(base_dict)
	_check(diffs_clean.is_empty(), "No diffs expected right after applying base preset", res)

	cfg.spring_stiffness_front = 52000.0
	ecu.live_smc_k_gain = 1050.0
	var mod_dict := SetupManagerScript.capture_from_car(cfg, ecu)
	var diffs_mod := mgr.compute_diffs_vs_base(mod_dict)
	_check(diffs_mod.size() >= 2, "Expected >=2 diffs after modifying k_f and live_max_mz, got %d" % diffs_mod.size(), res)

	# 2. DilSessionLogger: 75m Acceleration detection & export
	var logger := DilSessionLoggerScript.new()
	logger.session_dir = "user://dil_sessions_test/accel"
	logger.configure_track("Acceleration", Vector3.ZERO, Vector3(0.0, 0.0, -1.0), 10.0)
	var snap := SetupManagerScript.capture_from_car(cfg, ecu)
	var pos := Vector3.ZERO
	var dt := 0.02
	var accel_done := false
	for step in range(250):
		var vx := 18.0
		pos.z -= vx * dt
		var sample := {
			"vx": vx,
			"ay": 0.2,
			"ax": 6.5,
			"wz_ref": 0.0,
			"wz_real": 0.01,
			"beta_deg": 0.4,
			"delta_deg": 0.0,
			"throttle": 1.0,
			"brake": 0.0,
			"mz_req": 0.0,
			"t_cmd": [20.0, 20.0, 35.0, 35.0],
			"tc_cut": 0.0,
			"fz": [500.0, 500.0, 900.0, 900.0],
			"kappa": [0.08, 0.08, 0.10, 0.10],
			"alpha": [0.0, 0.0, 0.0, 0.0],
			"t_tire": [55.0, 55.0, 62.0, 62.0],
			"t_disc": [80.0, 80.0, 75.0, 75.0],
			"p_elec_kw": 45.0,
			"avg_sat_pct": 68.0,
		}
		logger.step_sample(dt, pos, Vector3(0.0, 0.0, -vx), sample, snap)
		if logger.completed_laps.size() >= 1:
			accel_done = true
			break
	_check(accel_done, "DilSessionLogger should auto-complete 75m Acceleration run", res)
	_check(logger.completed_laps.size() == 1, "Expected 1 completed Acceleration run", res)
	if logger.completed_laps.size() >= 1:
		var kpi: Dictionary = logger.completed_laps[0]
		_check(float(kpi.get("lap_time_s", 0.0)) > 3.5 and float(kpi.get("lap_time_s", 0.0)) < 5.0, "Acceleration 75m time at 18m/s should be ~4.16s, got %.3f" % float(kpi.get("lap_time_s", 0.0)), res)
		_check(FileAccess.file_exists(logger.session_dir.path_join(String(kpi.get("csv_file", "")))), "Expected telemetry CSV file on disk", res)
	_check(FileAccess.file_exists(logger.session_dir.path_join("session_summary.json")), "Expected session_summary.json on disk", res)
	_check(FileAccess.file_exists(logger.session_dir.path_join("session_report.md")), "Expected session_report.md on disk", res)

	# 3. DilSessionLogger: Circuit finish-line lap crossing + Baseline delta
	var circ := DilSessionLoggerScript.new()
	circ.session_dir = "user://dil_sessions_test/circuit"
	circ.configure_track("Autocross", Vector3.ZERO, Vector3(0.0, 0.0, -1.0), 30.0)
	var radius := 12.0
	var omega_lap := 1.25 # rad/s -> ~5.02s lap, circumference ~75.4m > 30m
	var lap_count := 0
	for k in range(550):
		var th := float(k) * dt * omega_lap
		# Start at (0,0,0), circle in XZ plane centered at (radius, 0, 0), heading -Z at th=0
		var px := radius * (1.0 - cos(th))
		var pz := -radius * sin(th)
		var vx_w := radius * omega_lap * sin(th)
		var vz_w := -radius * omega_lap * cos(th)
		var sample_c := {
			"vx": 15.0,
			"ay": 18.75,
			"ax": 0.5,
			"wz_ref": omega_lap,
			"wz_real": omega_lap - 0.05,
			"beta_deg": 2.1,
			"delta_deg": 8.0,
			"throttle": 0.75,
			"brake": 0.0,
			"mz_req": 180.0,
			"t_cmd": [15.0, 22.0, 18.0, 28.0],
			"tc_cut": 0.0,
			"fz": [550.0, 850.0, 600.0, 950.0],
			"kappa": [0.03, 0.04, 0.05, 0.06],
			"alpha": [3.5, 4.0, 3.8, 4.2],
			"t_tire": [64.0, 68.0, 66.0, 72.0],
			"t_disc": [120.0, 125.0, 110.0, 115.0],
			"p_elec_kw": 36.0,
			"avg_sat_pct": 82.0,
		}
		circ.step_sample(dt, Vector3(px, 0.0, pz), Vector3(vx_w, 0.0, vz_w), sample_c, snap)
		lap_count = circ.completed_laps.size()
		if lap_count >= 2:
			break
	_check(lap_count >= 1, "Circuit finish-line crossing should record >= 1 lap (got %d)" % lap_count, res)
	_check(circ.set_baseline_lap(1), "set_baseline_lap(1) should succeed after lap completion", res)
	var delta_val := circ.get_live_delta_s(37.7, 2.30)
	_check(delta_val < 0.0, "Expected negative delta (faster than baseline) at mid-lap, got %.3f" % delta_val, res)

	# 4. 3D Wheel Saturation & Hoosier R20 Dimensions
	var eta := MirenaCarScript.compute_wheel_saturation(1116.0, 1488.0, 1000.0, 1.86)
	_check(is_equal_approx(eta, 1.0), "Saturation sqrt(1116^2+1488^2)/(1.86*1000) should equal 1.0, got %.4f" % eta, res)
	_check(is_equal_approx(MirenaCarScript.HOOSIER_R20_RADIUS_M, 0.2032), "Hoosier R20 radius should be 0.2032 m", res)
	_check(is_equal_approx(MirenaCarScript.HOOSIER_R20_WIDTH_M, 0.203), "Hoosier R20 width should be 0.203 m", res)

	return res
