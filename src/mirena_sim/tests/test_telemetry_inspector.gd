extends RefCounted

const TelemetryInspectorOverlay = preload("res://Scenes/UI/TelemetryInspectorOverlay.gd")
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
	if samples.size() >= 3 and is_finite(float(samples[-1].get("wz_real", NAN))) and is_finite(float(samples[-1].get("t_fl", NAN))):
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

	overlay.free()
	car.free()

	return {
		"name": "test_telemetry_inspector",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
