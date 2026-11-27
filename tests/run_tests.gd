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
const TestMultiVehicleSetup = preload("test_multi_vehicle_setup.gd")
const TestFast3DofModel = preload("test_fast_3dof_model.gd")
const TestEcuTv4Wd = preload("test_ecu_tv_4wd.gd")
const TestWheelFfb = preload("test_wheel_ffb.gd")
const TestDoubleWishboneKinematics = preload("test_double_wishbone_kinematics.gd")
const TestSuspensionVisualizerAndModes = preload("test_suspension_visualizer_and_modes.gd")
const TestInputConditioning = preload("test_input_conditioning.gd")
const TestTvPowerCurveAndDrivability = preload("test_tv_power_curve_and_drivability.gd")
const TestDilFeatures = preload("test_dil_features.gd")
const TestCasadiSimUiIntegration = preload("test_casadi_simui_integration.gd")
const TestTer26SimulatorIntegration = preload("test_ter26_simulator_integration.gd")
const TestTer27SkidpadTv = preload("test_ter27_skidpad_tv.gd")
const TestTer27DilTelemetry = preload("test_ter27_dil_telemetry.gd")

func _suite_has_run_async(suite: Script) -> bool:
	for m in suite.get_script_method_list():
		if String(m.get("name", "")) == "run_async":
			return true
	return false


func _initialize() -> void:
	_run_all_suites()


func _run_all_suites() -> void:
	print("\n========================================================")
	print("       MIRENA DYNAMICS -- AUTOMATED TEST SUITE")
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
		TestMultiVehicleSetup,
		AcceptanceManeuversSuite,
		TestTVCharacterization,
		TestFast3DofModel,
		TestEcuTv4Wd,
		TestWheelFfb,
		TestDoubleWishboneKinematics,
		TestSuspensionVisualizerAndModes,
		TestInputConditioning,
		TestTvPowerCurveAndDrivability,
		TestDilFeatures,
		TestCasadiSimUiIntegration,
		TestTer26SimulatorIntegration,
		TestTer27SkidpadTv,
		TestTer27DilTelemetry
	]
	
	var total_passed = 0
	var total_failed = 0
	var total_assertions = 0
	var start_time = Time.get_ticks_msec()
	
	for suite in suites:
		var res: Dictionary
		if _suite_has_run_async(suite):
			res = await suite.run_async(self)
		else:
			res = suite.run()
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
