extends RefCounted

const TelemetryInspectorOverlay = preload("res://Scenes/UI/TelemetryInspectorOverlay.gd")
const MirenaCarDynamics = preload("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd")
const VehicleConfig = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModel = preload("res://Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")


class MockCarNode extends Node:
	var model: VehicleDynamicsModel = null
	var _last_step_res: Dictionary = {}


static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var overlay := TelemetryInspectorOverlay.new()
	overlay._ready()

	# 1. Hidden by default, toggles cleanly with F11/F12 semantics
	if not overlay.visible and not overlay.frozen:
		passed += 1
	else:
		failed += 1
		errors.append("TelemetryInspectorOverlay should start hidden and unfrozen")

	overlay.toggle_visibility()
	if overlay.visible:
		passed += 1
	else:
		failed += 1
		errors.append("toggle_visibility() did not show overlay")

	# 2. Sample updates from VehicleDynamicsModel without NaNs and populates ring buffer
	var cfg := VehicleConfig.new()
	var mdl := VehicleDynamicsModel.new(cfg)
	var inp := VehicleInputs.new()
	inp.set_from_manual(0.7, 0.0, 0.14, cfg.max_steer_angle)
	inp.ecu_driving_mode = 1
	mdl.chassis_state.linear_velocity = Vector3(14.0, 0.0, 0.0)
	for i in range(4):
		mdl.wheels[i].omega = 14.0 / cfg.wheel_radius

	var car := MockCarNode.new()
	car.model = mdl
	for _k in range(4):
		car._last_step_res = mdl.step_standalone(inp, 0.005, 0.0)
		overlay.update_from_car(car, 0.02)

	var samples := overlay.get_ordered_samples()
	if samples.size() >= 3 and is_finite(float(samples[-1].get("wz_real", NAN))) and is_finite(float(samples[-1].get("T_cmd_FL", NAN))):
		passed += 1
	else:
		failed += 1
		errors.append("Ring buffer failed to record valid samples (size=%d)" % samples.size())

	# 3. Freeze stops sampling
	var count_before := samples.size()
	overlay.toggle_freeze()
	overlay.update_from_car(car, 0.02)
	var count_after := overlay.get_ordered_samples().size()
	if overlay.frozen and count_after == count_before:
		passed += 1
	else:
		failed += 1
		errors.append("Freeze did not pause sampling: %d -> %d" % [count_before, count_after])

	# 4. CustomChartWindow: creation, move, resize (min 220x120), arbitrary channels
	var cw := overlay.create_chart_window(["wz_ref", "Fz_FL"], Vector2(120, 90), Vector2(340, 210))
	cw.move_window(Vector2(250, 140))
	cw.resize_window(Vector2(100, 80)) # should clamp to min 220x120
	var clamped_ok := cw.size.x >= 220.0 and cw.size.y >= 120.0
	cw.resize_window(Vector2(480, 290))
	var resized_ok := absf(cw.size.x - 480.0) < 0.5 and absf(cw.size.y - 290.0) < 0.5
	cw.toggle_channel("T_cmd_FL", true)
	cw.toggle_channel("wz_ref", false)
	var chs := cw.get_selected_channels()
	if clamped_ok and resized_ok and cw.position == Vector2(250, 140) and chs.has("Fz_FL") and chs.has("T_cmd_FL") and not chs.has("wz_ref"):
		passed += 1
	else:
		failed += 1
		errors.append("CustomChartWindow move/resize/channels failed: pos=%s size=%s ch=%s" % [str(cw.position), str(cw.size), str(chs)])

	# 5. Layout save & load round-trip
	var tmp_layout := "user://test_telemetry_layout.json"
	var saved_ok := overlay.save_layout(tmp_layout)
	overlay.close_chart_window(cw)
	var loaded_ok := overlay.load_layout(tmp_layout)
	var restored := overlay.get_chart_windows()
	if FileAccess.file_exists(tmp_layout):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(tmp_layout))
	if saved_ok and loaded_ok and restored.size() == 1 and restored[0].get_selected_channels().has("T_cmd_FL"):
		passed += 1
	else:
		failed += 1
		errors.append("Layout save/load round-trip failed (saved=%s loaded=%s count=%d)" % [str(saved_ok), str(loaded_ok), restored.size()])

	# 6. Saturation eta_i math & color gradient in MirenaCarDynamics
	var eta_low := MirenaCarDynamics.compute_wheel_saturation(300.0, 200.0, 1000.0, 1.5)
	var eta_hi := MirenaCarDynamics.compute_wheel_saturation(2000.0, 2000.0, 1000.0, 1.5)
	var c_green := MirenaCarDynamics.saturation_to_color(0.40)
	var c_amber := MirenaCarDynamics.saturation_to_color(0.80)
	var c_red := MirenaCarDynamics.saturation_to_color(1.05)
	if eta_low < 0.30 and absf(eta_hi - 1.2) < 1e-4 and c_green.g > c_green.r and c_amber.r > 0.9 and c_red.r > 0.9 and c_red.g < 0.35:
		passed += 1
	else:
		failed += 1
		errors.append("Saturation math/color gradient invalid: eta_low=%.3f eta_hi=%.3f" % [eta_low, eta_hi])

	# 7. 3D Wheel Force Arrows update & zero-cost hide in MirenaCarDynamics
	var dyn := MirenaCarDynamics.new()
	dyn.config = cfg
	dyn.model = mdl
	dyn._sync_geometry_from_config()
	dyn._setup_wheel_force_arrows()
	dyn.set_wheel_arrows_enabled(true)
	dyn._update_wheel_force_arrows(
		[Vector3(450.0, 650.0, 0.0), Vector3(420.0, 610.0, 0.0), Vector3(600.0, 720.0, 0.0), Vector3(580.0, 690.0, 0.0)],
		[750.0, 720.0, 820.0, 800.0],
		[true, true, true, true]
	)
	var arrows_ok := dyn._wheel_arrow_roots.size() == 4 and dyn._wheel_arrow_roots[0].visible
	dyn.set_wheel_arrows_enabled(false)
	var hidden_ok := not dyn._wheel_arrow_roots[0].visible
	if arrows_ok and hidden_ok:
		passed += 1
	else:
		failed += 1
		errors.append("3D wheel force arrows failed: arrows_ok=%s hidden_ok=%s" % [str(arrows_ok), str(hidden_ok)])

	dyn.free()
	overlay.free()
	car.free()

	return {
		"name": "test_telemetry_inspector",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
