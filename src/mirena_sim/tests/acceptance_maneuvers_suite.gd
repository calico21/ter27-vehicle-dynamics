extends RefCounted

const VehicleConfigScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleConfig.gd")
const VehicleInputsScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("../Scenes/Vehicle/Dynamics/Core/VehicleDynamicsModel.gd")
const TorqueVectoringScript = preload("../Scenes/Vehicle/Dynamics/Drivetrain/TorqueVectoring.gd")

static func run(verbose: bool = false) -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var cfg = VehicleConfigScript.new()
	var g: float = cfg.gravity
	var total_weight: float = cfg.mass * g

	if verbose:
		print("\n=================================================================")
		print("   MIRENA DYNAMICS (TeR27) — 14-DOF ACCEPTANCE MANEUVERS SUITE   ")
		print("=================================================================\n")

	# -----------------------------------------------------------------
	# MANEUVER 1: Static Equilibrium (2.0 s @ 200 Hz, dt = 0.005 s)
	# -----------------------------------------------------------------
	var model = VehicleDynamicsModelScript.new(cfg)
	model.chassis_state.position.z = cfg.h_cog + 0.01 # start 1 cm perturbed, settle to delta_z=0
	var inp = VehicleInputsScript.new()
	var res_eq: Dictionary = {}
	for _k in range(400): # 2.0 s @ dt = 0.005 s
		res_eq = model.step_standalone(inp, 0.005, 0.0)

	var fz_eq: Array = res_eq["normal_forces"]
	var sum_fz_eq: float = float(fz_eq[0]) + float(fz_eq[1]) + float(fz_eq[2]) + float(fz_eq[3])
	var rel_err_pct: float = absf(sum_fz_eq - total_weight) / total_weight * 100.0
	var max_dz_eq: float = 0.0
	for c in model.suspension.corners:
		max_dz_eq = maxf(max_dz_eq, absf(c.delta_z))
	var lat_drift: float = absf(model.chassis_state.linear_velocity.y)
	var yaw_drift: float = absf(model.chassis_state.yaw)

	var m1_ok: bool = (rel_err_pct < 0.1 and max_dz_eq < 5e-4 and lat_drift < 1e-3 and yaw_drift < 1e-4)
	if m1_ok:
		passed += 1
	else:
		failed += 1
		errors.append("M1 Static Equilibrium failed: rel_err=%.4f%%, max_dz=%.5f, vy=%.5f, yaw=%.5f" % [rel_err_pct, max_dz_eq, lat_drift, yaw_drift])

	if verbose:
		print("[M1] Static Equilibrium (2.0 s @ 200 Hz)")
		print("     sum(Fz) = %.2f N (expected %.2f N, err = %.4f%%)" % [sum_fz_eq, total_weight, rel_err_pct])
		print("     max |delta_z_i| = %.5f m | v_y = %.6f m/s | yaw = %.6f rad" % [max_dz_eq, lat_drift, yaw_drift])
		print("     Front/Rear split = %.1f%% / %.1f%%\n" % [
			(float(fz_eq[0]) + float(fz_eq[1])) / sum_fz_eq * 100.0,
			(float(fz_eq[2]) + float(fz_eq[3])) / sum_fz_eq * 100.0
		])

	# -----------------------------------------------------------------
	# MANEUVER 2: Straight Acceleration (0 -> 25 m/s & 75 m) + Max Brake (gas = -1.0)
	# -----------------------------------------------------------------
	model.reset()
	var dt: float = 0.0025
	inp.reset()
	inp.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	inp.set_from_ros_control(1.0, 0.0, cfg.max_steer_angle)

	var t_sim: float = 0.0
	var t_to_25: float = -1.0
	var t_75m: float = -1.0
	var peak_ax_g: float = 0.0
	var dfz_long_acc_at_peak: float = 0.0
	var dfz_analytic_acc_at_peak: float = 0.0
	var prev_u: float = 0.0

	var static_front: float = float(cfg.get_static_axle_loads()["front_axle"])
	var static_rear: float = float(cfg.get_static_axle_loads()["rear_axle"])

	for k in range(2000): # up to 5.0 s
		var res_acc: Dictionary = model.step_standalone(inp, dt, 0.0)
		t_sim += dt
		var u_now: float = model.chassis_state.linear_velocity.x
		var ax_now: float = (u_now - prev_u) / dt
		prev_u = u_now
		var ax_g: float = ax_now / g
		if k > 10 and ax_g > peak_ax_g:
			peak_ax_g = ax_g
			var fz_a: Array = res_acc["normal_forces"]
			dfz_long_acc_at_peak = (float(fz_a[2]) + float(fz_a[3])) - static_rear
			dfz_analytic_acc_at_peak = cfg.mass * ax_now * cfg.h_cog / cfg.wheelbase

		if t_to_25 < 0.0 and u_now >= 25.0:
			t_to_25 = t_sim
		if t_75m < 0.0 and model.chassis_state.position.x >= 75.0:
			t_75m = t_sim
		if t_to_25 > 0.0 and t_75m > 0.0:
			break

	# Now full braking from >= 25 m/s via gas = -1.0
	var u_brk_start: float = model.chassis_state.linear_velocity.x
	var x_brk_start: float = model.chassis_state.position.x
	inp.set_from_ros_control(-1.0, 0.0, cfg.max_steer_angle)

	var t_brk: float = 0.0
	var prev_u_b: float = u_brk_start
	var peak_decel_g: float = 0.0
	var min_u_b: float = u_brk_start
	var peak_dfz_brk: float = 0.0
	var t_stopped: float = -1.0

	for k in range(2000): # up to 5.0 s braking
		var res_b: Dictionary = model.step_standalone(inp, dt, 0.0)
		t_brk += dt
		var u_b: float = model.chassis_state.linear_velocity.x
		min_u_b = minf(min_u_b, u_b)
		var ax_b: float = (u_b - prev_u_b) / dt
		prev_u_b = u_b
		if k > 5 and (-ax_b / g) > peak_decel_g:
			peak_decel_g = -ax_b / g
		var fz_b: Array = res_b["normal_forces"]
		peak_dfz_brk = maxf(peak_dfz_brk, (float(fz_b[0]) + float(fz_b[1])) - static_front)
		if t_stopped < 0.0 and absf(u_b) < 0.05:
			t_stopped = t_brk

	var brk_dist: float = model.chassis_state.position.x - x_brk_start
	var avg_decel_g: float = (u_brk_start / maxf(t_stopped, 1e-4)) / g
	var u_final_brk: float = model.chassis_state.linear_velocity.x
	var max_w_final: float = 0.0
	for w in model.wheels:
		max_w_final = maxf(max_w_final, absf(w.omega))

	var m2_ok: bool = (t_to_25 > 0.5 and t_75m > 1.5 and t_stopped > 0.3 and peak_decel_g > 1.0 \
			and absf(u_final_brk) < 1e-3 and min_u_b >= -1e-3 and max_w_final < 1e-2 \
			and dfz_long_acc_at_peak > 300.0 and peak_dfz_brk > 300.0)
	if m2_ok:
		passed += 1
	else:
		failed += 1
		errors.append("M2 Accel/Brake failed: t25=%.2fs t75=%.2fs t_stop=%.2fs u_fin=%.4f min_u=%.4f" % [
			t_to_25, t_75m, t_stopped, u_final_brk, min_u_b
		])

	if verbose:
		print("[M2] Straight Accel (0 -> 25 m/s / 75 m) & Max Braking (gas = -1.0)")
		print("     0->25 m/s: %.2f s | 0-75 m: %.2f s | Peak accel: %.2f g" % [t_to_25, t_75m, peak_ax_g])
		print("     Accel peak dFz_rear = +%.1f N (quasi-static m*a_x*h/L = %.1f N)" % [dfz_long_acc_at_peak, dfz_analytic_acc_at_peak])
		print("     Brake from %.2f m/s: dist = %.2f m, t_stop = %.2f s, avg/peak decel = %.2f g / %.2f g" % [
			u_brk_start, brk_dist, t_stopped, avg_decel_g, peak_decel_g
		])
		print("     Brake peak dFz_front = +%.1f N | final v_x = %.4f m/s | max |omega_w| = %.4f rad/s\n" % [
			peak_dfz_brk, u_final_brk, max_w_final
		])

	# -----------------------------------------------------------------
	# MANEUVER 3: Skidpad Steady-State (R = 9.125 m and R = 18.25 m) — OPEN vs DYC
	# -----------------------------------------------------------------
	var sp_9_open: Dictionary = _run_skidpad(cfg, 9.125, 11.0, TorqueVectoringScript.TVMode.MODE_OPEN)
	var sp_9_dyc: Dictionary = _run_skidpad(cfg, 9.125, 11.0, TorqueVectoringScript.TVMode.MODE_DYC)
	var sp_18_open: Dictionary = _run_skidpad(cfg, 18.25, 15.5, TorqueVectoringScript.TVMode.MODE_OPEN)
	var sp_18_dyc: Dictionary = _run_skidpad(cfg, 18.25, 15.5, TorqueVectoringScript.TVMode.MODE_DYC)

	var m3_ok: bool = (sp_9_open["ay_g"] > 1.0 and sp_9_dyc["ay_g"] > 1.0 \
			and sp_18_open["ay_g"] > 1.0 and sp_18_dyc["ay_g"] > 1.0 \
			and sp_9_open["roll_deg"] > 0.1 and sp_9_dyc["roll_deg"] > 0.1 \
			and sp_9_open["fz_right"] > sp_9_open["fz_left"] \
			and sp_18_dyc["fz_right"] > sp_18_dyc["fz_left"])
	if m3_ok:
		passed += 1
	else:
		failed += 1
		errors.append("M3 Skidpad failed: ay_g=[%.2f, %.2f, %.2f, %.2f]" % [
			sp_9_open["ay_g"], sp_9_dyc["ay_g"], sp_18_open["ay_g"], sp_18_dyc["ay_g"]
		])

	if verbose:
		print("[M3] Steady-State Skidpad (R = 9.125 m @ 11 m/s & R = 18.25 m @ 15.5 m/s)")
		for label_res in [
			["R= 9.125m OPEN", sp_9_open],
			["R= 9.125m DYC ", sp_9_dyc],
			["R=18.250m OPEN", sp_18_open],
			["R=18.250m DYC ", sp_18_dyc]
		]:
			var tag: String = label_res[0]
			var s: Dictionary = label_res[1]
			print("     %s: u=%.2f m/s | ay=%.2f g | r_err=%.3f rad/s | steer=%.2f deg | roll=%.2f deg | beta=%.2f deg | Fz_L/R=%.0f/%.0f N" % [
				tag, s["u"], s["ay_g"], s["r_err"], rad_to_deg(s["steer"]), s["roll_deg"], s["beta_deg"], s["fz_left"], s["fz_right"]
			])
		print("")

	# -----------------------------------------------------------------
	# MANEUVER 4: Step Steer & Slalom (sign coherence & low phase lag)
	# -----------------------------------------------------------------
	# 4A. Step Steer from u = 16 m/s with delta = +0.08 rad
	model.reset()
	var u_step: float = 16.0
	model.chassis_state.linear_velocity = Vector3(u_step, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u_step / cfg.wheel_radius
	inp.reset()
	inp.throttle = 0.25

	var r_at_50ms: float = 0.0
	var ay_at_50ms: float = 0.0
	var peak_r_step: float = 0.0
	var peak_ay_step_g: float = 0.0
	var r_signed_ok: bool = true

	for k in range(480): # 1.2 s (0.2 s baseline + 1.0 s step)
		if k >= 80:
			inp.steer_angle = 0.08
		model.step_standalone(inp, dt, 0.0)
		var r_now: float = model.chassis_state.angular_velocity.z
		var ay_now: float = model.chassis_state.linear_velocity.x * r_now
		if k == 100: # 50 ms after step onset
			r_at_50ms = r_now
			ay_at_50ms = ay_now
		if k >= 85 and (r_now <= 0.0 or ay_now <= 0.0):
			r_signed_ok = false
		if k >= 80:
			peak_r_step = maxf(peak_r_step, r_now)
			peak_ay_step_g = maxf(peak_ay_step_g, ay_now / g)

	# 4B. Slalom: delta(t) = 0.06 * sin(2*pi * 0.6 * t) at u = 15 m/s
	model.reset()
	var u_slalom: float = 15.0
	model.chassis_state.linear_velocity = Vector3(u_slalom, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u_slalom / cfg.wheel_radius
	inp.reset()
	inp.throttle = 0.2

	var d_hist: Array[float] = []
	var r_hist: Array[float] = []
	var peak_ay_slalom_g: float = 0.0
	var min_r_slalom: float = 0.0
	var max_r_slalom: float = 0.0

	for k in range(1400): # 3.5 s
		var t_sl: float = float(k) * dt
		var d_cmd: float = 0.06 * sin(TAU * 0.6 * t_sl)
		inp.steer_angle = d_cmd
		model.step_standalone(inp, dt, 0.0)
		var r_sl: float = model.chassis_state.angular_velocity.z
		d_hist.append(d_cmd)
		r_hist.append(r_sl)
		if k >= 200:
			peak_ay_slalom_g = maxf(peak_ay_slalom_g, absf(model.chassis_state.linear_velocity.x * r_sl) / g)
			min_r_slalom = minf(min_r_slalom, r_sl)
			max_r_slalom = maxf(max_r_slalom, r_sl)

	# Cross-correlate steer & yaw rate over last 2.5 s to measure phase lag
	var best_lag_samples: int = 0
	var best_corr: float = -1e18
	var start_idx: int = 200
	for lag in range(0, 100): # 0 .. 250 ms
		var corr: float = 0.0
		for i in range(start_idx, d_hist.size() - lag):
			corr += d_hist[i] * r_hist[i + lag]
		if corr > best_corr:
			best_corr = corr
			best_lag_samples = lag
	var phase_lag_ms: float = float(best_lag_samples) * dt * 1000.0

	var m4_ok: bool = (r_signed_ok and r_at_50ms > 0.1 and ay_at_50ms > 1.0 and peak_r_step > 0.5 \
			and max_r_slalom > 0.3 and min_r_slalom < -0.3 and phase_lag_ms <= 200.0 and peak_ay_slalom_g > 0.8)
	if m4_ok:
		passed += 1
	else:
		failed += 1
		errors.append("M4 StepSteer/Slalom failed: r50=%.3f, phase_lag=%.1f ms, slalom_r=[%.2f, %.2f]" % [
			r_at_50ms, phase_lag_ms, min_r_slalom, max_r_slalom
		])

	if verbose:
		print("[M4] Step Steer (+0.08 rad @ 16 m/s) & Slalom (0.6 Hz @ 15 m/s)")
		print("     Step Steer @ +50 ms: r = %.3f rad/s, ay = %.2f m/s^2 | Peak r = %.3f rad/s, Peak ay = %.2f g" % [
			r_at_50ms, ay_at_50ms, peak_r_step, peak_ay_step_g
		])
		print("     Slalom: r in [%.2f, %.2f] rad/s | Peak |ay| = %.2f g | Steer-to-Yaw phase lag = %.1f ms\n" % [
			min_r_slalom, max_r_slalom, peak_ay_slalom_g, phase_lag_ms
		])

	# -----------------------------------------------------------------
	# MANEUVER 5: High-Speed Max-Steer Rollover Stability (22 m/s)
	# -----------------------------------------------------------------
	model.reset()
	var u_lim: float = 22.0
	model.chassis_state.linear_velocity = Vector3(u_lim, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u_lim / cfg.wheel_radius
	inp.reset()
	inp.drive_mode = VehicleConfigScript.DriveMode.FOUR_WD
	inp.throttle = 0.20

	var max_roll_lim_deg: float = 0.0
	var max_pitch_lim_deg: float = 0.0
	var peak_ay_lim_g: float = 0.0
	var min_grounded_lim: int = 4
	var min_z_lim: float = cfg.h_cog

	for k in range(800): # 2.0 s @ 400 Hz
		if k >= 40:
			inp.steer_angle = cfg.max_steer_angle
		var res_lim: Dictionary = model.step_standalone(inp, dt, 0.0)
		max_roll_lim_deg = maxf(max_roll_lim_deg, absf(rad_to_deg(model.chassis_state.roll)))
		max_pitch_lim_deg = maxf(max_pitch_lim_deg, absf(rad_to_deg(model.chassis_state.pitch)))
		min_z_lim = minf(min_z_lim, model.chassis_state.position.z)
		var ay_lim: float = absf(model.chassis_state.linear_velocity.x * model.chassis_state.angular_velocity.z) / g
		peak_ay_lim_g = maxf(peak_ay_lim_g, ay_lim)
		var fz_lim: Array = res_lim["normal_forces"]
		var grounded_now: int = 0
		for corner_fz in fz_lim:
			if float(corner_fz) > 1.0:
				grounded_now += 1
		min_grounded_lim = mini(min_grounded_lim, grounded_now)

	var ssf: float = (0.5 * (cfg.track_width_f + cfg.track_width_r)) / (2.0 * cfg.h_cog)
	var m5_ok: bool = (ssf > 2.05 and max_roll_lim_deg < 15.0 and max_pitch_lim_deg < 10.0 \
			and peak_ay_lim_g > 1.3 and min_grounded_lim >= 4 and min_z_lim > 0.15)
	if m5_ok:
		passed += 1
	else:
		failed += 1
		errors.append("M5 Rollover Stability failed: SSF=%.3f, max_roll=%.2f deg, max_pitch=%.2f deg, min_grounded=%d/4, peak_ay=%.2f g" % [
			ssf, max_roll_lim_deg, max_pitch_lim_deg, min_grounded_lim, peak_ay_lim_g
		])

	if verbose:
		print("[M5] Limit Step-Steer Rollover Stability (delta = %.2f rad / %.1f deg @ 22 m/s)" % [
			cfg.max_steer_angle, rad_to_deg(cfg.max_steer_angle)
		])
		print("     SSF = %.3f g | max|roll| = %.2f deg | max|pitch| = %.2f deg | Peak |ay| = %.2f g | Min grounded = %d/4\n" % [
			ssf, max_roll_lim_deg, max_pitch_lim_deg, peak_ay_lim_g, min_grounded_lim
		])
		print("-----------------------------------------------------------------")
		print("Acceptance Summary: %d/5 maneuvers PASSED." % passed)
		print("=================================================================\n")

	return {
		"name": "Acceptance Maneuvers (Block E)",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}

static func _run_skidpad(cfg: VehicleConfigScript, radius: float, v_target: float, tv_mode: int) -> Dictionary:
	var model = VehicleDynamicsModelScript.new(cfg)
	var dt: float = 0.0025
	model.chassis_state.linear_velocity = Vector3(v_target, 0.0, 0.0)
	for w in model.wheels:
		w.omega = v_target / cfg.wheel_radius

	var inp = VehicleInputsScript.new()
	inp.drive_mode = VehicleConfigScript.DriveMode.TORQUE_VECTORING
	inp.tv_mode = tv_mode

	var int_v: float = 0.0
	var r_ref: float = v_target / radius
	var ff_steer: float = atan(cfg.wheelbase / radius)

	var acc_u: float = 0.0
	var acc_ay_g: float = 0.0
	var acc_r_err: float = 0.0
	var acc_steer: float = 0.0
	var acc_roll: float = 0.0
	var acc_beta: float = 0.0
	var acc_fz_l: float = 0.0
	var acc_fz_r: float = 0.0
	var avg_n: int = 0

	for k in range(1600): # 4.0 s total, average final 1.0 s (ticks 1200..1600)
		var u_now: float = model.chassis_state.linear_velocity.x
		var v_now: float = model.chassis_state.linear_velocity.y
		var r_now: float = model.chassis_state.angular_velocity.z

		# Speed PI hold
		var e_v: float = v_target - u_now
		int_v = clampf(int_v + e_v * dt, -2.0, 2.0)
		var throttle_cmd: float = clampf(0.18 + 0.35 * e_v + 0.4 * int_v, 0.05, 0.9)
		inp.throttle = throttle_cmd

		# Curvature-tracking yaw-rate feedback
		var r_des: float = maxf(u_now, 1.0) / radius
		var e_r: float = r_des - r_now
		inp.steer_angle = clampf(ff_steer + 0.28 * e_r, -cfg.max_steer_angle, cfg.max_steer_angle)

		# TV DYC assists yaw tracking toward r_des when tv_mode == MODE_DYC
		if tv_mode == TorqueVectoringScript.TVMode.MODE_DYC:
			inp.tv_yaw_moment_req = clampf(150.0 + 700.0 * e_r, -800.0, 1200.0)
		else:
			inp.tv_yaw_moment_req = 0.0

		var res: Dictionary = model.step_standalone(inp, dt, 0.0)
		if k >= 1200:
			var fz: Array = res["normal_forces"]
			acc_u += u_now
			acc_ay_g += (u_now * r_now) / cfg.gravity
			acc_r_err += absf(r_ref - r_now)
			acc_steer += inp.steer_angle
			acc_roll += rad_to_deg(model.chassis_state.roll)
			acc_beta += rad_to_deg(atan2(v_now, maxf(u_now, 0.5)))
			acc_fz_l += float(fz[0]) + float(fz[2])
			acc_fz_r += float(fz[1]) + float(fz[3])
			avg_n += 1

	var inv_n: float = 1.0 / float(max(avg_n, 1))
	return {
		"u": acc_u * inv_n,
		"ay_g": acc_ay_g * inv_n,
		"r_err": acc_r_err * inv_n,
		"steer": acc_steer * inv_n,
		"roll_deg": acc_roll * inv_n,
		"beta_deg": acc_beta * inv_n,
		"fz_left": acc_fz_l * inv_n,
		"fz_right": acc_fz_r * inv_n
	}
