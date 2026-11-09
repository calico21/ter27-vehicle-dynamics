extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")

static func run() -> Dictionary:
	var passed = 0
	var failed = 0
	var errors = []
	
	# Test 1: Forward direction (+X ROS is -Z Godot)
	var godot_fwd = Vector3(0.0, 0.0, -10.0)
	var ros_fwd = VehicleConfigScript.godot_to_ros_vec(godot_fwd)
	if ros_fwd.is_equal_approx(Vector3(10.0, 0.0, 0.0)):
		passed += 1
	else:
		failed += 1
		errors.append("godot_to_ros_vec forward failed: expected (10, 0, 0), got %s" % ros_fwd)
	
	# Test 2: Left direction (+Y ROS is -X Godot)
	var godot_left = Vector3(-5.0, 0.0, 0.0)
	var ros_left = VehicleConfigScript.godot_to_ros_vec(godot_left)
	if ros_left.is_equal_approx(Vector3(0.0, 5.0, 0.0)):
		passed += 1
	else:
		failed += 1
		errors.append("godot_to_ros_vec left failed: expected (0, 5, 0), got %s" % ros_left)
	
	# Test 3: Up direction (+Z ROS is +Y Godot)
	var godot_up = Vector3(0.0, 2.5, 0.0)
	var ros_up = VehicleConfigScript.godot_to_ros_vec(godot_up)
	if ros_up.is_equal_approx(Vector3(0.0, 0.0, 2.5)):
		passed += 1
	else:
		failed += 1
		errors.append("godot_to_ros_vec up failed: expected (0, 0, 2.5), got %s" % ros_up)
	
	# Test 4: Inverse roundtrip (ROS -> Godot -> ROS)
	var ros_test = Vector3(12.34, -5.67, 8.90)
	var godot_conv = VehicleConfigScript.ros_to_godot_vec(ros_test)
	var ros_roundtrip = VehicleConfigScript.godot_to_ros_vec(godot_conv)
	if ros_test.distance_to(ros_roundtrip) < 1e-5:
		passed += 1
	else:
		failed += 1
		errors.append("Roundtrip ROS->Godot->ROS failed: original %s, roundtrip %s" % [ros_test, ros_roundtrip])
	
	# Test 5: Inverse roundtrip (Godot -> ROS -> Godot)
	var godot_test = Vector3(3.14, 2.71, -1.41)
	var ros_conv = VehicleConfigScript.godot_to_ros_vec(godot_test)
	var godot_roundtrip = VehicleConfigScript.ros_to_godot_vec(ros_conv)
	if godot_test.distance_to(godot_roundtrip) < 1e-5:
		passed += 1
	else:
		failed += 1
		errors.append("Roundtrip Godot->ROS->Godot failed: original %s, roundtrip %s" % [godot_test, godot_roundtrip])
	
	return {"name": "test_coordinates", "passed": passed, "failed": failed, "errors": errors}
