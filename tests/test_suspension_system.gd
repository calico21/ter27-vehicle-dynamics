extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const SuspensionCornerScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/SuspensionCorner.gd")
const SuspensionSystemScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/SuspensionSystem.gd")
const AeroModelScript = preload("res://Scenes/Vehicle/Dynamics/Aerodynamics/AeroModel.gd")
const ChassisDynamicsScript = preload("res://Scenes/Vehicle/Dynamics/Chassis/ChassisDynamics.gd")

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
	cfg_kin.use_3d_kinematics = false
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
	var exp_ground_fl: float = SuspensionSystemScript.compute_ground_camber_iso(0.01, float(steers[0]), 0.0, true, false, cfg_kin)
	var exp_ground_fr: float = SuspensionSystemScript.compute_ground_camber_iso(0.01, float(steers[1]), 0.0, true, true, cfg_kin)
	if absf(model.wheels[0].camber_angle - exp_ground_fl) < 1e-6 \
			and absf(model.wheels[1].camber_angle - exp_ground_fr) < 1e-6 \
			and absf(float(steers[0]) - (0.1 + tau_fl)) < 1e-6 \
			and absf(float(steers[1]) - (0.1 + tau_fr)) < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("VehicleDynamicsModel kinematics propagation failed")

	# 7. TeR27 Roll Center (h_rc,cg) & 21.7% sprung roll moment arm reduction
	var cfg_rc = VehicleConfigScript.new()
	var h_roll: float = SuspensionSystemScript.get_roll_moment_arm(cfg_rc)
	var roll_torque_ratio: float = (cfg_rc.sprung_mass * h_roll) / (cfg_rc.mass * cfg_rc.h_cog)
	var reduction_pct: float = (1.0 - roll_torque_ratio) * 100.0
	if absf(cfg_rc.roll_center_height_cg - 0.030832) < 1e-4 and absf(reduction_pct - 21.7) < 0.2:
		passed += 1
	else:
		failed += 1
		errors.append("Roll center / 21.7%% reduction failed: h_rc_cg=%.5f, reduction=%.2f%%" % [cfg_rc.roll_center_height_cg, reduction_pct])

	# 8. Caster (5 deg) & KPI (10 deg) ground camber coupling under steer & roll
	var g_left_turn_fr: float = SuspensionSystemScript.compute_ground_camber_iso(0.0, 0.2, 0.02, true, true, cfg_rc)
	var g_zero_fr: float = SuspensionSystemScript.compute_ground_camber_iso(0.0, 0.0, 0.0, true, true, cfg_rc)
	if g_left_turn_fr > g_zero_fr - 0.02:
		passed += 1
	else:
		failed += 1
		errors.append("Caster/KPI ground camber coupling failed: g_steer=%.4f, g_zero=%.4f" % [g_left_turn_fr, g_zero_fr])

	# 9. Tire vertical stiffness kz, radial deflection rho, loaded radius Rl, and series ride rate
	var w_test = model.wheels[0]
	w_test.is_grounded = true
	w_test.update_radial_deflection(1150.0, true, cfg_rc)
	var exp_rho: float = 1150.0 / cfg_rc.tire_vertical_stiffness
	var kw_front: float = SuspensionSystemScript.get_wheel_spring_stiffness(true, cfg_rc)
	var exp_k_ride: float = (kw_front * cfg_rc.tire_vertical_stiffness) / (kw_front + cfg_rc.tire_vertical_stiffness)
	if absf(w_test.radial_deflection - exp_rho) < 1e-6 \
			and absf(w_test.loaded_radius - (cfg_rc.wheel_radius - exp_rho)) < 1e-6 \
			and absf(w_test.ride_rate - exp_k_ride) < 1e-2:
		passed += 1
	else:
		failed += 1
		errors.append("Radial deflection / ride rate failed: rho=%.6f (exp %.6f), k_ride=%.2f (exp %.2f)" % [w_test.radial_deflection, exp_rho, w_test.ride_rate, exp_k_ride])

	# 10. 3D M^-1 Anti-Dive (56.9%) pitch reduction under braking while preserving sum(Fz) = m*g
	var anti_res: Dictionary = SuspensionSystemScript.compute_anti_pitch_forces([-2000.0, -2000.0, -500.0, -500.0], cfg_rc)
	var anti_forces: Array = anti_res["corner_forces"]
	var anti_pitch_tau: float = float(anti_res["pitch_torque_ros"])
	# Braking (Fx < 0) produces positive front anti-dive force and negative (nose-up) pitch torque
	if absf(float(anti_forces[0]) - 0.569 * 2000.0) < 1e-3 and anti_pitch_tau < 0.0:
		passed += 1
	else:
		failed += 1
		errors.append("Anti-dive M^-1 check failed: dfz0=%.2f, tau_y=%.2f" % [float(anti_forces[0]), anti_pitch_tau])

	# 11. Steering compliance: delta_comp < 0 on outer/front wheel under positive Fy & negative Mz
	var delta_comp: float = SuspensionSystemScript.compute_steering_compliance(-50.0, 2000.0, cfg_rc)
	var exp_comp: float = (-50.0 - 2000.0 * cfg_rc.mechanical_trail_front) / cfg_rc.steering_compliance_stiffness_nm_rad
	if delta_comp < 0.0 and absf(delta_comp - exp_comp) < 1e-7:
		passed += 1
	else:
		failed += 1
		errors.append("Steering compliance check failed: delta_comp=%.6f (exp %.6f)" % [delta_comp, exp_comp])

	# 12. Digressive 2-stage damper continuity at v_damper = 0.05 m/s and reduced high-speed slope
	var mr_f: float = cfg_rc.motion_ratio_front
	var dzd_knee: float = cfg_rc.damper_knee_velocity / mr_f
	var f_below: float = SuspensionSystemScript.compute_damper_force_wheel(true, dzd_knee - 1e-6, cfg_rc)
	var f_above: float = SuspensionSystemScript.compute_damper_force_wheel(true, dzd_knee + 1e-6, cfg_rc)
	var f_high: float = SuspensionSystemScript.compute_damper_force_wheel(true, dzd_knee * 3.0, cfg_rc)
	var slope_low: float = SuspensionSystemScript.get_wheel_damping(true, dzd_knee * 0.5, cfg_rc)
	var slope_high: float = SuspensionSystemScript.get_wheel_damping(true, dzd_knee * 2.0, cfg_rc)
	if absf(f_above - f_below) < 0.05 and slope_low > slope_high and f_high > f_above:
		passed += 1
	else:
		failed += 1
		errors.append("Digressive damper continuity/slope failed: f_below=%.2f, f_above=%.2f, slope_low=%.1f, slope_high=%.1f" % [f_below, f_above, slope_low, slope_high])

	# 13. TeR27 Anti-Ackermann (-11%) geometry: outer wheel steers more than inner wheel
	var steer_cmd: float = 0.20
	var steer_inner_fl: float = SuspensionSystemScript.compute_ackermann_steer(steer_cmd, true, cfg_rc)
	var steer_outer_fr: float = SuspensionSystemScript.compute_ackermann_steer(steer_cmd, false, cfg_rc)
	var cfg_pro = VehicleConfigScript.new()
	cfg_pro.ackermann_factor = 0.60
	var steer_pro_fl: float = SuspensionSystemScript.compute_ackermann_steer(steer_cmd, true, cfg_pro)
	var steer_pro_fr: float = SuspensionSystemScript.compute_ackermann_steer(steer_cmd, false, cfg_pro)
	if absf(cfg_rc.ackermann_factor + 0.11) < 1e-6 \
			and steer_inner_fl < steer_cmd and steer_outer_fr > steer_cmd \
			and steer_pro_fl > steer_cmd and steer_pro_fr < steer_cmd:
		passed += 1
	else:
		failed += 1
		errors.append("Anti-Ackermann (-11%%) check failed: anti=(%.4f, %.4f), pro=(%.4f, %.4f)" % [
			steer_inner_fl, steer_outer_fr, steer_pro_fl, steer_pro_fr
		])

	# 14. TeR27 tabulated camber gain (-0.181 deg/10mm front, -0.288 deg/10mm rear) & bump steer
	var c_f0: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.camber_table_front, 0.0))
	var c_f10: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.camber_table_front, 0.01))
	var c_r0: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.camber_table_rear, 0.0))
	var c_r10: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.camber_table_rear, 0.01))
	var toe_f25: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.toe_table_front, 0.025))
	var toe_r25: float = rad_to_deg(SuspensionSystemScript.lookup_1d_table(cfg_rc.toe_table_rear, 0.025))
	if absf(c_f0 - (-2.0)) < 1e-4 and absf((c_f10 - c_f0) - (-0.181)) < 1e-3 \
			and absf(c_r0 - (-1.5)) < 1e-4 and absf((c_r10 - c_r0) - (-0.288)) < 1e-3 \
			and absf(toe_f25 - (-0.018)) < 1e-4 and absf(toe_r25) < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("TeR27 camber/bump-steer tables mismatch: cf=(%.3f, d=%.3f), cr=(%.3f, d=%.3f), toe=(%.4f, %.4f)" % [
			c_f0, c_f10 - c_f0, c_r0, c_r10 - c_r0, toe_f25, toe_r25
		])

	# 15. Ride-height-sensitive aerodynamics (static=1.0, ground effect, front stall < 18mm, high ride loss)
	var mod_static: Dictionary = AeroModelScript.compute_ride_height_modulation(0.035, 0.035, cfg_rc)
	var mod_ge: Dictionary = AeroModelScript.compute_ride_height_modulation(0.022, 0.022, cfg_rc)
	var mod_stall: Dictionary = AeroModelScript.compute_ride_height_modulation(0.010, 0.035, cfg_rc)
	var mod_high: Dictionary = AeroModelScript.compute_ride_height_modulation(0.065, 0.065, cfg_rc)
	var aero_ge: Dictionary = AeroModelScript.compute_forces_ros(Vector3(30.0, 0.0, 0.0), cfg_rc, 0.022, 0.022)
	var aero_stall: Dictionary = AeroModelScript.compute_forces_ros(Vector3(30.0, 0.0, 0.0), cfg_rc, 0.010, 0.035)
	if absf(float(mod_static["scale_front"]) - 1.0) < 1e-6 and absf(float(mod_static["scale_rear"]) - 1.0) < 1e-6 \
			and absf(float(mod_static["drag_scale"]) - 1.0) < 1e-6 and not bool(mod_static["front_stalled"]) \
			and float(mod_ge["scale_front"]) > 1.10 and float(mod_ge["scale_rear"]) > 1.10 \
			and bool(mod_stall["front_stalled"]) and float(mod_stall["scale_front"]) < 0.70 \
			and float(mod_stall["drag_scale"]) > 1.05 and float(mod_high["scale_front"]) < 0.90 \
			and float(aero_stall["downforce_front"]) < float(aero_ge["downforce_front"]) \
			and bool(aero_stall["front_stalled"]):
		passed += 1
	else:
		failed += 1
		errors.append("Ride-height aero modulation failed: static=%s ge=%s stall=%s high=%s" % [
			str(mod_static), str(mod_ge), str(mod_stall), str(mod_high)
		])

	# 16. Total Izz with Steiner unsprung corners (~137.14 kg*m^2) & Ixz roll-yaw coupling
	var total_izz: float = cfg_rc.get_total_izz()
	var s_coupl = ChassisDynamicsScript.State6DOF.new(cfg_rc.h_cog)
	var d_coupl = ChassisDynamicsScript.compute_derivatives(s_coupl, Vector3.ZERO, Vector3(100.0, 0.0, 0.0), cfg_rc)
	var s_free = ChassisDynamicsScript.State6DOF.new(2.0)
	s_free.angular_velocity = Vector3(0.0, 0.0, 2.0)
	var d_free = ChassisDynamicsScript.compute_derivatives(s_free, Vector3.ZERO, Vector3.ZERO, cfg_rc)
	if absf(total_izz - 137.141) < 0.2 and absf(d_coupl.omega_dot.z) > 0.1 and d_coupl.omega_dot.z < 0.0 \
			and d_free.omega_dot.length() < 1e-6:
		passed += 1
	else:
		failed += 1
		errors.append("Total Izz / Ixz coupling failed: Izz=%.3f, wz_dot=%.5f, free_wdot=%s" % [
			total_izz, d_coupl.omega_dot.z, str(d_free.omega_dot)
		])

	return {
		"name": "SuspensionSystem (Block B)",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
