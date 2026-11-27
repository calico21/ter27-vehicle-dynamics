extends RefCounted

const MirenaCarDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd")
const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")

static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	# 1. Preserve global physics tick rate in project.godot (400 Hz)
	var phys_hz: int = int(ProjectSettings.get_setting("physics/common/physics_ticks_per_second", 0))
	if phys_hz == 400:
		passed += 1
	else:
		failed += 1
		errors.append("project.godot physics_ticks_per_second altered: %d (expected 400)" % phys_hz)

	# 2. Verify ROS topic, message type, QoS, and frame contracts in MirenaCarDynamics.gd
	var f = FileAccess.open("res://Scenes/Vehicle/Dynamics/Core/MirenaCarDynamics.gd", FileAccess.READ)
	var src: String = f.get_as_text() if f != null else ""
	var required_contracts: Array[String] = [
		'"create_publisher", "/as_status",        "mirena_common/msg/AsInfo"',
		'"create_publisher", "/debug/state/car", "mirena_common/msg/Car"',
		'"create_publisher", "/debug/perception",  "mirena_common/msg/EntityList"',
		'"create_publisher", "/debug/slam",        "mirena_common/msg/EntityList"',
		'"create_publisher", "/inferred_control",  "mirena_common/msg/CarControl"',
		'"create_publisher", "/lap_status",        "mirena_common/msg/LapStatus"',
		'"create_publisher", "/sensors/wheel_speeds", "mirena_common/msg/WheelSpeeds", wheel_qos',
		'"create_subscription", "/control", "mirena_common/msg/CarControl", _on_control',
		'"create_subscription", "/system/mission_status", "mirena_common/msg/MissionInfo", _on_mission_info',
		'"create_subscription", "/as_status", "mirena_common/msg/AsInfo", _on_as_status',
		'wheel_qos.set("history", ClassDB.class_get_integer_constant("RosQoS", "KEEP_LAST"))',
		'wheel_qos.set("depth", 5)',
		'wheel_qos.set("reliability", ClassDB.class_get_integer_constant("RosQoS", "BEST_EFFORT"))',
		'wheel_qos.set("durability", ClassDB.class_get_integer_constant("RosQoS", "VOLATILE"))',
		'car_state.header.frame_id = "odom"',
		'car_state.child_frame_id = frame_id',
		'lap_status_msg.header.frame_id = "map"'
	]
	var missing: Array[String] = []
	for pattern in required_contracts:
		if src.find(pattern) == -1:
			missing.append(pattern)

	if missing.is_empty():
		passed += 1
	else:
		failed += 1
		errors.append("Missing ROS 2 contract lines in MirenaCarDynamics.gd: %s" % str(missing))

	# 3. Verify default ROS sensor properties & decimation (100 Hz from 400 Hz tick => window = 4)
	var car = MirenaCarDynamicsScript.new()
	car.config = VehicleConfigScript.new()
	car.inputs = VehicleInputsScript.new()
	var decimation_win: int = maxi(1, int(round(float(phys_hz) / car.wheel_speed_rate)))
	if car.frame_id == "car/cog" and absf(car.wheel_speed_rate - 100.0) < 1e-6 and decimation_win == 4:
		passed += 1
	else:
		failed += 1
		errors.append("MirenaCar ROS sensor config regression: frame_id=%s, rate=%.1f, decimation=%d" % [
			car.frame_id, car.wheel_speed_rate, decimation_win
		])

	# 4. Verify live RosNode initialization (if rclgd extension active) & callback preservation
	car._init_ros()
	var ros_ok: bool = true
	if ClassDB.class_exists("RosNode"):
		ros_ok = (car._node != null and car._control_sub != null and car._wheel_speeds_pub != null and car._lap_status_pub != null and car._state_pub != null)
	if ros_ok:
		passed += 1
	else:
		failed += 1
		errors.append("MirenaCar._init_ros() failed to bind expected RosNode pubs/subs")

	# 5. Verify /control callback -> VehicleInputs longitudinal & steering contract (+gas accel, -gas brake, +steer left)
	car._on_control({"gas": 0.65, "steer_angle": 0.2})
	car._process_ros_pilot()
	car.inputs.set_unified_control(car.gas, car.steering, car.brake, car.config.max_steer_angle)
	var ok_fwd: bool = absf(car.inputs.throttle - 0.65) < 1e-6 and car.inputs.brake == 0.0 and absf(car.inputs.steer_angle - 0.2) < 1e-6

	car._on_control({"gas": -0.75, "steer_angle": -0.15})
	car._process_ros_pilot()
	car.inputs.set_unified_control(car.gas, car.steering, car.brake, car.config.max_steer_angle)
	var ok_rev: bool = car.inputs.throttle == 0.0 and absf(car.inputs.brake - 0.75) < 1e-6 and absf(car.inputs.steer_angle + 0.15) < 1e-6

	if ok_fwd and ok_rev:
		passed += 1
	else:
		failed += 1
		errors.append("ROS /control callback mapping regression")

	# Clean up RosNode if instantiated
	if car._node != null and car._node is Node:
		car._node.queue_free()
	car.free()

	return {
		"name": "test_ros_contracts",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
