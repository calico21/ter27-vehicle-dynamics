extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputsScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const SuspensionCornerScript = preload("../Scenes/Vehicle/Dynamics/Suspension/SuspensionCorner.gd")
const SuspensionSystemScript = preload("../Scenes/Vehicle/Dynamics/Suspension/SuspensionSystem.gd")

static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var config = VehicleConfigScript.new()
	var susp = SuspensionSystemScript.new()

	# 1. Static preload equilibrium at delta_z = 0 (TeR27: 45% front / 55% rear)
	var dz_zero = [0.0, 0.0, 0.0, 0.0]
	var dzd_zero = [0.0, 0.0, 0.0, 0.0]
	var g_all = [true, true, true, true]
	var corners: Array[SuspensionCornerScript] = susp.evaluate_from_corners(dz_zero, dzd_zero, g_all, config)

	var total_weight: float = config.mass * config.gravity
	var f0_fl: float = corners[0].fz
	var f0_rl: float = corners[2].fz
	var f_front: float = corners[0].fz + corners[1].fz
	var f_rear: float = corners[2].fz + corners[3].fz
	var f_sum: float = f_front + f_rear

	if absf(f_sum - total_weight) < 1e-3 and absf(f_front - 0.45 * total_weight) < 1e-2 and absf(f_rear - 0.55 * total_weight) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("Static preload equilibrium failed: total=%.2f (exp %.2f), front=%.2f, rear=%.2f" % [f_sum, total_weight, f_front, f_rear])

	# Zero parasitic bumpstop or ARB forces at static design height
	if absf(corners[0].bumpstop_force) < 1e-6 and absf(corners[0].arb_force) < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("Non-zero bumpstop or ARB force at static design height")

	# 2. Motion Ratio conversion: k_wheel = k_spring * MR^2 (TeR27 design values: MR = 1 / 1.13)
	var kw_f: float = SuspensionSystemScript.get_wheel_spring_stiffness(true, config)
	var kw_r: float = SuspensionSystemScript.get_wheel_spring_stiffness(false, config)
	var exp_kw_f: float = 35000.0 * 0.884956 * 0.884956
	var exp_kw_r: float = 48000.0 * 0.884956 * 0.884956
	if absf(kw_f - exp_kw_f) < 1e-2 and absf(kw_r - exp_kw_r) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("Motion ratio wheel rate mismatch: kw_f=%.2f (exp %.2f), kw_r=%.2f (exp %.2f)" % [kw_f, exp_kw_f, kw_r, exp_kw_r])

	# Symmetric jounce +0.01 m increases corner force by k_wheel * 0.01
	var jounce_corners: Array[SuspensionCornerScript] = susp.evaluate_from_corners([0.01, 0.01, 0.01, 0.01], dzd_zero, g_all, config)
	var df_fl: float = jounce_corners[0].fz - f0_fl
	var df_rl: float = jounce_corners[2].fz - f0_rl
	if absf(df_fl - kw_f * 0.01) < 1e-2 and absf(df_rl - kw_r * 0.01) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("Symmetric jounce delta_F mismatch: dFL=%.2f, dRL=%.2f" % [df_fl, df_rl])

	# 3. ARB pure roll moment: F_ARB_L + F_ARB_R = 0, zero heave change (within grounded travel)
	var arb_corners: Array[SuspensionCornerScript] = susp.evaluate_from_corners([0.005, -0.005, 0.004, -0.004], dzd_zero, g_all, config)
	var sum_arb_f: float = arb_corners[0].arb_force + arb_corners[1].arb_force
	var sum_arb_r: float = arb_corners[2].arb_force + arb_corners[3].arb_force
	var arb_heave_f: float = arb_corners[0].fz + arb_corners[1].fz
	var arb_heave_r: float = arb_corners[2].fz + arb_corners[3].fz
	var exp_arb_fl: float = config.arb_stiffness_front * 0.010
	var exp_arb_rl: float = config.arb_stiffness_rear * 0.008

	if absf(sum_arb_f) < 1e-6 and absf(sum_arb_r) < 1e-6 and arb_corners[0].arb_force > 0.0 and arb_corners[1].arb_force < 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("ARB pure roll moment invariant failed: sum_f=%.4f, sum_r=%.4f" % [sum_arb_f, sum_arb_r])

	if absf(arb_corners[0].arb_force - exp_arb_fl) < 1e-3 and absf(arb_corners[2].arb_force - exp_arb_rl) < 1e-3 \
			and absf(arb_heave_f - f_front) < 1e-2 and absf(arb_heave_r - f_rear) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("ARB magnitude / zero-heave invariant failed: arb_f=%.2f (exp %.2f), heave_f=%.2f (exp %.2f)" % [arb_corners[0].arb_force, exp_arb_fl, arb_heave_f, f_front])

	# 4. Asymmetric damping & progressive bumpstops + droop clamp (>= 0)
	var c_comp_f: float = SuspensionSystemScript.get_wheel_damping(true, 0.1, config)
	var c_reb_f: float = SuspensionSystemScript.get_wheel_damping(true, -0.1, config)
	var f_bump_below: float = SuspensionSystemScript.compute_bumpstop_force(config.bumpstop_start_front, true, config)
	var f_bump_above: float = SuspensionSystemScript.compute_bumpstop_force(config.bumpstop_start_front + 0.01, true, config)
	var droop_corners: Array[SuspensionCornerScript] = susp.evaluate_from_corners([-0.10, -0.10, -0.10, -0.10], dzd_zero, g_all, config)

	if c_reb_f > c_comp_f and f_bump_below == 0.0 and f_bump_above > 1000.0 and droop_corners[0].fz == 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Asymmetric damping/bumpstop/droop failed: c_comp=%.1f c_reb=%.1f bump_below=%.1f bump_above=%.1f droop_fz=%.1f" % [c_comp_f, c_reb_f, f_bump_below, f_bump_above, droop_corners[0].fz])

	# 5. Airborne corner unloads to zero and suppresses axle ARB force
	var air_corners: Array[SuspensionCornerScript] = susp.evaluate_from_corners([0.02, -0.05, 0.0, 0.0], dzd_zero, [true, false, true, true], config)
	if air_corners[1].fz == 0.0 and air_corners[0].arb_force == 0.0 and air_corners[1].arb_force == 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Airborne wheel ARB suppression failed: fr_fz=%.2f, fl_arb=%.2f" % [air_corners[1].fz, air_corners[0].arb_force])

	# 6. 1D piecewise-linear kinematic interpolation + L/R sign conventions + end-to-end model integration
	var cfg_kin = VehicleConfigScript.new()
	cfg_kin.ackermann_factor = 0.0
	cfg_kin.camber_table_front = [[-0.02, deg_to_rad(-1.0)], [0.00, deg_to_rad(-2.0)], [0.02, deg_to_rad(-3.0)]]
	cfg_kin.toe_table_front = [[-0.02, deg_to_rad(0.0)], [0.00, deg_to_rad(0.5)], [0.02, deg_to_rad(1.0)]]

	var interp_camber: float = SuspensionSystemScript.lookup_1d_table(cfg_kin.camber_table_front, 0.01)
	var gamma_fl: float = SuspensionSystemScript.compute_corner_camber_iso(0.01, true, false, cfg_kin)
	var gamma_fr: float = SuspensionSystemScript.compute_corner_camber_iso(0.01, true, true, cfg_kin)
	var tau_fl: float = SuspensionSystemScript.compute_corner_toe_iso(0.01, true, false, cfg_kin)
	var tau_fr: float = SuspensionSystemScript.compute_corner_toe_iso(0.01, true, true, cfg_kin)

	if absf(interp_camber - deg_to_rad(-2.5)) < 1e-6 \
			and absf(gamma_fl - deg_to_rad(-2.5)) < 1e-6 and absf(gamma_fr - deg_to_rad(2.5)) < 1e-6 \
			and absf(tau_fl - deg_to_rad(-0.75)) < 1e-6 and absf(tau_fr - deg_to_rad(0.75)) < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("1D kinematic interpolation / L-R signs failed: gamma=(%.4f, %.4f), tau=(%.4f, %.4f)" % [gamma_fl, gamma_fr, tau_fl, tau_fr])

	# End-to-end check inside VehicleDynamicsModel.step()
	var model = VehicleDynamicsModelScript.new(cfg_kin)
	var inp = VehicleInputsScript.new()
	inp.steer_angle = 0.1
	var contacts = [
		{"grounded": true, "delta_z": 0.01, "delta_z_dot": 0.0},
		{"grounded": true, "delta_z": 0.01, "delta_z_dot": 0.0},
		{"grounded": true, "delta_z": 0.0, "delta_z_dot": 0.0},
		{"grounded": true, "delta_z": 0.0, "delta_z_dot": 0.0}
	]
	var step_res = model.step(inp, Vector3(15.0, 0.0, 0.0), Vector3.ZERO, contacts, 0.0025)
	var steers: Array = step_res["wheel_steer_angles"]
	if absf(model.wheels[0].camber_angle - gamma_fl) < 1e-6 \
			and absf(model.wheels[1].camber_angle - gamma_fr) < 1e-6 \
			and absf(float(steers[0]) - (0.1 + tau_fl)) < 1e-6 \
			and absf(float(steers[1]) - (0.1 + tau_fr)) < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("VehicleDynamicsModel kinematics propagation failed")

	return {
		"name": "SuspensionSystem (Block B)",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
