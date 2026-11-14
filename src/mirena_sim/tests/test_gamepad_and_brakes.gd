extends RefCounted

const VehicleConfig = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModel = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const GamepadController = preload("res://Scenes/Vehicle/Dynamics/Core/GamepadController.gd")
const WheelFFBController = preload("res://Scenes/Vehicle/Dynamics/Core/WheelFFBController.gd")
const WheelDynamics = preload("res://Scenes/Vehicle/Dynamics/Tires/WheelDynamics.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []

	# 1. Config JSON loading
	var gp := GamepadController.new()
	if gp.load_from_json("res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json"):
		passed += 1
	else:
		failed += 1
		errors.append("Failed to load gamepad.json")

	if is_equal_approx(gp.steering_deadzone, 0.06) and is_equal_approx(gp.steering_gamma, 2.0) and is_equal_approx(gp.speed_sensitivity, 0.25):
		passed += 1
	else:
		failed += 1
		errors.append("Unexpected default parameters in gamepad.json")

	# 2. Hydraulic braking vs coasting on 4 wheels
	var cfg := VehicleConfig.new()
	var model_coast := VehicleDynamicsModel.new(cfg)
	var model_brake := VehicleDynamicsModel.new(cfg)

	var v_init := 20.0
	model_coast.chassis_state.linear_velocity = Vector3(v_init, 0.0, 0.0)
	model_brake.chassis_state.linear_velocity = Vector3(v_init, 0.0, 0.0)
	for i in range(4):
		model_coast.wheels[i].omega = v_init / cfg.wheel_radius
		model_brake.wheels[i].omega = v_init / cfg.wheel_radius

	var in_coast := VehicleInputs.new()
	in_coast.set_from_manual(0.0, 0.0, 0.0, cfg.max_steer_angle)
	var in_brake := VehicleInputs.new()
	in_brake.set_from_manual(0.0, 0.8, 0.0, cfg.max_steer_angle)

	var trq_check: Dictionary = model_brake.compute_drivetrain_torques(in_brake, v_init)
	var b_trqs: Array = trq_check["brake"]
	var dt := 0.005
	for step_i in range(40): # 0.2 s
		model_coast.step_standalone(in_coast, dt)
		model_brake.step_standalone(in_brake, dt)
	var all_wheels_braked: bool = b_trqs[0] > 10.0 and b_trqs[1] > 10.0 and b_trqs[2] > 10.0 and b_trqs[3] > 10.0
	if all_wheels_braked and b_trqs[0] > b_trqs[2]:
		passed += 1
	else:
		failed += 1
		errors.append("Hydraulic brake torques not applied on all 4 wheels with front bias: %s" % [b_trqs])

	var vx_coast := model_coast.chassis_state.linear_velocity.x
	var vx_brake := model_brake.chassis_state.linear_velocity.x
	if vx_brake < vx_coast - 1.0 and vx_brake < v_init - 1.5:
		passed += 1
	else:
		failed += 1
		errors.append("Hydraulic braking did not decelerate vehicle significantly: coast=%.2f brake=%.2f" % [vx_coast, vx_brake])

	# 3. Independent throttle + brake (simultaneous inputs)
	var in_both := VehicleInputs.new()
	in_both.set_unified_control(0.6, 0.0, 0.5, cfg.max_steer_angle)
	if is_equal_approx(in_both.throttle, 0.6) and is_equal_approx(in_both.brake, 0.5):
		passed += 1
	else:
		failed += 1
		errors.append("Throttle and brake should be independent when gas >= 0: th=%.2f br=%.2f" % [in_both.throttle, in_both.brake])

	# 4. Trigger normalization (bipolar [-1,1] and unipolar [0,1])
	var gp_trig := GamepadController.new()
	var rest_bipolar := gp_trig.normalize_trigger(-1.0, true)
	var half_bipolar := gp_trig.normalize_trigger(0.0, true)
	var full_bipolar := gp_trig.normalize_trigger(1.0, true)
	var rest_uni := gp_trig.normalize_trigger(0.0, false)
	var full_uni := gp_trig.normalize_trigger(1.0, false)
	if is_zero_approx(rest_bipolar) and half_bipolar > 0.45 and half_bipolar < 0.55 and is_equal_approx(full_bipolar, 1.0) and is_zero_approx(rest_uni) and is_equal_approx(full_uni, 1.0):
		passed += 1
	else:
		failed += 1
		errors.append("Trigger normalization failed: bi=[%.2f, %.2f, %.2f] uni=[%.2f, %.2f]" % [rest_bipolar, half_bipolar, full_bipolar, rest_uni, full_uni])

	# 5. Steering chain: deadzone continuity, bounds [-1, 1], speed sensitivity attenuation
	var dz_zero := GamepadController.apply_deadzone(0.05, 0.06)
	var dz_edge := GamepadController.apply_deadzone(0.0601, 0.06)
	if is_zero_approx(dz_zero) and dz_edge >= 0.0 and dz_edge < 0.002:
		passed += 1
	else:
		failed += 1
		errors.append("Steering deadzone not continuous at boundary: zero=%.4f edge=%.4f" % [dz_zero, dz_edge])

	var gp_low := GamepadController.new()
	gp_low.load_from_json("res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json")
	var gp_high := GamepadController.new()
	gp_high.load_from_json("res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json")

	var out_low := 0.0
	var out_high := 0.0
	var bounded := true
	for _i in range(200):
		out_low = gp_low.step_steering(1.0, 5.0, 0.01)
		out_high = gp_high.step_steering(1.0, 28.0, 0.01)
		if absf(out_low) > 1.0 + 1e-5 or absf(out_high) > 1.0 + 1e-5:
			bounded = false

	if bounded and out_low > 0.85 and out_high > 0.70 and out_high < out_low * 0.90:
		passed += 1
	else:
		failed += 1
		errors.append("Steering speed sensitivity/bounds check failed: low=%.3f high=%.3f bounded=%s" % [out_low, out_high, bounded])

	# 6. Throttle & brake gamma monotonicity & endpoints
	var gp_pedals := GamepadController.new()
	gp_pedals.load_from_json("res://Scenes/Vehicle/Dynamics/Core/Config/gamepad.json")
	var th_0 := gp_pedals.shape_throttle(0.0)
	var th_half := gp_pedals.shape_throttle(0.5)
	var th_1 := gp_pedals.shape_throttle(1.0)
	var br_0 := gp_pedals.shape_brake(0.0)
	var br_half := gp_pedals.shape_brake(0.5)
	var br_1 := gp_pedals.shape_brake(1.0)
	if is_zero_approx(th_0) and is_equal_approx(th_1, 1.0) and th_0 < th_half and th_half < 0.5 and th_half < th_1 \
	and is_zero_approx(br_0) and is_equal_approx(br_1, 1.0) and br_0 < br_half and br_half < 0.5 and br_half < br_1:
		passed += 1
	else:
		failed += 1
		errors.append("Pedal gamma shaping invalid: th=[%.3f, %.3f, %.3f] br=[%.3f, %.3f, %.3f]" % [th_0, th_half, th_1, br_0, br_half, br_1])

	# 7. WheelFFBController auto-detection, linear +/-110 deg WHEEL mode & speed invariance
	var ffb := WheelFFBController.new()
	var mode_dd: int = ffb.detect_input_device_from_name("Fanatec Podium DD1")
	var dd_linear_ok: bool = (
		mode_dd == WheelFFBController.InputDeviceMode.WHEEL
		and is_zero_approx(ffb.speed_sensitivity)
		and is_equal_approx(ffb.steering_gamma, 1.0)
		and is_zero_approx(ffb.steering_deadzone)
		and is_zero_approx(ffb.steering_filter)
		and is_equal_approx(WheelFFBController.WHEEL_MAX_DEG, 110.0)
		and is_equal_approx(WheelFFBController.WHEEL_MAX_RAD, deg_to_rad(110.0))
	)
	var out_slow: float = ffb.step_steering(0.5, 5.0, 0.01)
	var deg_half: float = ffb.wheel_angle_deg
	var out_fast: float = ffb.step_steering(0.5, 35.0, 0.01)
	var out_full_neg: float = ffb.step_steering(-1.0, 25.0, 0.01)
	var deg_full_neg: float = ffb.wheel_angle_deg
	var mode_gp: int = ffb.detect_input_device_from_name("Xbox 360 Controller")
	ffb.input_device_override = WheelFFBController.InputDeviceMode.WHEEL
	var mode_override: int = ffb.detect_input_device_from_name("Generic Controller")
	if dd_linear_ok and is_equal_approx(out_slow, 0.5) and is_equal_approx(out_fast, 0.5) \
	and is_equal_approx(deg_half, 55.0) and is_equal_approx(out_full_neg, -1.0) \
	and is_equal_approx(deg_full_neg, -110.0) and mode_gp == WheelFFBController.InputDeviceMode.GAMEPAD \
	and mode_override == WheelFFBController.InputDeviceMode.WHEEL:
		passed += 1
	else:
		failed += 1
		errors.append("WheelFFBController WHEEL mode invalid: dd=%s slow=%.3f fast=%.3f deg=%.1f modes=%d/%d" % [dd_linear_ok, out_slow, out_fast, deg_half, mode_gp, mode_override])

	# 8. WheelFFBController physical steering column FFB torque & signed normalized haptic output
	var w_fl := WheelDynamics.new(true)
	var w_fr := WheelDynamics.new(false)
	w_fl.M_z = 12.0
	w_fr.M_z = 8.0
	w_fl.F_y = 900.0
	w_fr.F_y = 700.0
	w_fl.F_x = 600.0
	w_fr.F_x = 200.0
	# Expected: ((12 + 8) + (900 + 700)*0.013 + (600 - 200)*0.018) / 4.8 = (20 + 20.8 + 7.2) / 4.8 = 10.0 Nm
	ffb.reset()
	var tau_pos: float = ffb.compute_steering_torque_nm([w_fl, w_fr])
	ffb.apply_haptic_feedback(0, [w_fl, w_fr], 15.0, 0.0)
	var norm_pos: float = ffb.ffb_normalized
	var pos_ok: bool = (
		absf(tau_pos - 10.0) < 1e-4
		and absf(ffb.ffb_torque_nm - 10.0) < 1e-4
		and is_equal_approx(norm_pos, 10.0 / ffb.ffb_max_torque_nm)
	)
	w_fl.M_z = -12.0
	w_fr.M_z = -8.0
	w_fl.F_y = -900.0
	w_fr.F_y = -700.0
	w_fl.F_x = 200.0
	w_fr.F_x = 600.0
	var tau_neg: float = ffb.compute_steering_torque_nm([w_fl, w_fr])
	ffb.apply_haptic_feedback(0, [w_fl, w_fr], 15.0, 0.0)
	var neg_ok: bool = (
		absf(tau_neg + 10.0) < 1e-4
		and ffb.ffb_normalized < -0.5
		and ffb.ffb_normalized >= -1.0
	)
	if pos_ok and neg_ok:
		passed += 1
	else:
		failed += 1
		errors.append("WheelFFBController FFB torque failed: tau_pos=%.3f norm_pos=%.3f tau_neg=%.3f norm_neg=%.3f" % [tau_pos, norm_pos, tau_neg, ffb.ffb_normalized])

	# 9. Live gamepad param tuning & JSON save/reload
	var gp_live := GamepadController.new()
	gp_live.set_live_gamepad_param("steering_gamma", 1.5)
	gp_live.set_live_gamepad_param("v_ref_mps", 20.0)
	var live_path := "user://test_gamepad_live.json"
	var saved_ok := gp_live.save_gamepad_config_to_json(live_path)
	var gp_reloaded := GamepadController.new()
	var loaded_ok := gp_reloaded.load_from_json(live_path)
	var p_reloaded := gp_reloaded.get_live_gamepad_params()
	if saved_ok and loaded_ok and is_equal_approx(float(p_reloaded.get("steering_gamma", 0.0)), 1.5) and is_equal_approx(float(p_reloaded.get("v_ref_mps", 0.0)), 20.0):
		passed += 1
	else:
		failed += 1
		errors.append("Gamepad live param save/reload failed: saved=%s loaded=%s gamma=%s v_ref=%s" % [
			str(saved_ok), str(loaded_ok), str(p_reloaded.get("steering_gamma", -1.0)), str(p_reloaded.get("v_ref_mps", -1.0))
		])

	return {
		"name": "test_gamepad_and_brakes",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
