extends SceneTree

const TestCoordinates = preload("test_coordinates.gd")
const TestWheelKinematics = preload("test_wheel_kinematics.gd")
const TestDrivetrain4WD = preload("test_drivetrain_4wd.gd")
const TestWheelSpinIntegration = preload("test_wheel_spin_integration.gd")
const TestSlipAndTyre = preload("test_slip_and_tyre.gd")
const TestLoadTransfer = preload("test_load_transfer.gd")
const TestSuspensionSystem = preload("test_suspension_system.gd")
const TestChassis14DOF = preload("test_chassis_14dof.gd")
const TestBlockDSubsystems = preload("test_block_d_subsystems.gd")
const TestResultantForces = preload("test_resultant_forces.gd")
const TestTireMF62 = preload("test_tire_mf62_suite.gd")
const TestTireParity = preload("test_tire_parity.gd")
const TestTransientDynamics = preload("test_transient_dynamics.gd")
const TestRosContracts = preload("test_ros_contracts.gd")
const TestGamepadAndBrakes = preload("test_gamepad_and_brakes.gd")
const TestEcuTorqueVectoringV3 = preload("test_ecu_torque_vectoring_v3.gd")
const TestBrakeBlendingAndGripMap = preload("test_brake_blending_and_grip_map.gd")
const TestTelemetryInspector = preload("test_telemetry_inspector.gd")
const AcceptanceManeuversSuite = preload("acceptance_maneuvers_suite.gd")
const TestTVCharacterization = preload("test_tv_characterization.gd")
const TestDilSessionLogger = preload("test_dil_session_logger.gd")

func _init() -> void:
	print("\n========================================================")
	print("       MIRENA DYNAMICS — AUTOMATED TEST SUITE")
	print("========================================================\n")
	
	var suites = [
		TestCoordinates,
		TestWheelKinematics,
		TestDrivetrain4WD,
		TestWheelSpinIntegration,
		TestSlipAndTyre,
		TestTireMF62,
		TestTireParity,
		TestTransientDynamics,
		TestLoadTransfer,
		TestSuspensionSystem,
		TestChassis14DOF,
		TestBlockDSubsystems,
		TestResultantForces,
		TestRosContracts,
		TestGamepadAndBrakes,
		TestEcuTorqueVectoringV3,
		TestBrakeBlendingAndGripMap,
		TestTelemetryInspector,
		AcceptanceManeuversSuite,
		TestTVCharacterization,
		TestDilSessionLogger
	]
	
	var total_passed = 0
	var total_failed = 0
	var total_assertions = 0
	var start_time = Time.get_ticks_msec()
	
	for suite in suites:
		var res = suite.run()
		var s_name: String = res.get("name", "UnknownSuite")
		var s_pass: int = res.get("passed", 0)
		var s_fail: int = res.get("failed", 0)
		var s_errs: Array = res.get("errors", [])
		
		total_passed += s_pass
		total_failed += s_fail
		total_assertions += (s_pass + s_fail)
		
		if s_fail == 0:
			print("  [PASS] %-30s (%d/%d checks passed)" % [s_name, s_pass, s_pass])
		else:
			print("  [FAIL] %-30s (%d passed, %d FAILED)" % [s_name, s_pass, s_fail])
			for err in s_errs:
				print("         -> ERROR: %s" % err)
	
	var elapsed_ms = Time.get_ticks_msec() - start_time
	print("\n--------------------------------------------------------")
	print("Results: %d total assertions across %d test suites." % [total_assertions, suites.size()])
	print("Passed: %d | Failed: %d | Elapsed: %d ms" % [total_passed, total_failed, elapsed_ms])
	print("========================================================\n")
	
	if total_failed == 0:
		print(">> ALL DYNAMICS TESTS PASSED SUCCESSFULLY! <<\n")
		quit(0)
	else:
		print(">> SOME TESTS FAILED! <<\n")
		quit(1)
