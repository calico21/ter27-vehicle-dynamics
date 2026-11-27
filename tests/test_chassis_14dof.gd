extends RefCounted

const VehicleConfigScript = preload("res://Scenes/Vehicle/Config/VehicleConfig.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const VehicleDynamicsModelScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")

static func run() -> Dictionary:
	var passed: int = 0
	var failed: int = 0
	var errors: Array = []

	var dt: float = 0.0025
	var cfg = VehicleConfigScript.new()
	var total_weight: float = cfg.mass * cfg.gravity

	# ---------------------------------------------------------
	# 1. Static equilibrium from delta_z = 0 & drop convergence
	# ---------------------------------------------------------
	var model = VehicleDynamicsModelScript.new(cfg)
	var inp = VehicleInputsScript.new()
	var res_eq: Dictionary = {}
	for _k in range(100):
		res_eq = model.step_standalone(inp, dt, 0.0)

	var fz_eq: Array = res_eq["normal_forces"]
	var sum_fz_eq: float = float(fz_eq[0]) + float(fz_eq[1]) + float(fz_eq[2]) + float(fz_eq[3])
	var s_eq = model.chassis_state
	if absf(s_eq.position.z - cfg.h_cog) < 1e-4 \
			and absf(s_eq.linear_velocity.z) < 1e-4 \
			and s_eq.angular_velocity.length() < 1e-5 \
			and absf(sum_fz_eq - total_weight) < 0.5:
		passed += 1
	else:
		failed += 1
		errors.append("Static equilibrium failed: Z=%.5f (exp %.3f), w=%.5f, |w_rot|=%.5f, sumFz=%.2f" % [
			s_eq.position.z, cfg.h_cog, s_eq.linear_velocity.z, s_eq.angular_velocity.length(), sum_fz_eq
		])

	# Drop-settle test from +0.02 m above design height
	model.reset()
	model.chassis_state.position.z = cfg.h_cog + 0.02
	var res_settle: Dictionary = {}
	for _k in range(400):
		res_settle = model.step_standalone(inp, dt, 0.0)
	var fz_set: Array = res_settle["normal_forces"]
	var sum_fz_set: float = float(fz_set[0]) + float(fz_set[1]) + float(fz_set[2]) + float(fz_set[3])
	var s_set = model.chassis_state
	if absf(s_set.position.z - cfg.h_cog) < 1e-3 \
			and absf(s_set.linear_velocity.z) < 1e-3 \
			and absf(s_set.angular_velocity.x) < 1e-3 \
			and absf(s_set.angular_velocity.y) < 1e-3 \
			and absf(s_set.angular_velocity.z) < 1e-4 \
			and absf(sum_fz_set - total_weight) < 1.0:
		passed += 1
	else:
		failed += 1
		errors.append("Drop settling failed: Z=%.5f, w=%.5f, q=%.5f, sumFz=%.2f" % [
			s_set.position.z, s_set.linear_velocity.z, s_set.angular_velocity.y, sum_fz_set
		])

	# ---------------------------------------------------------
	# 2. Free-fall kinematics & angular momentum conservation (vacuum)
	# ---------------------------------------------------------
	var cfg_ff = VehicleConfigScript.new()
	cfg_ff.cd_a = 0.0
	cfg_ff.cl_a = 0.0
	var model_ff = VehicleDynamicsModelScript.new(cfg_ff)
	var z0: float = 2.0
	model_ff.chassis_state.position.z = z0
	model_ff.chassis_state.angular_velocity = Vector3(0.0, 0.0, 2.0)
	var steps_ff: int = 80
	var t_ff: float = float(steps_ff) * dt
	var res_ff: Dictionary = {}
	for _k in range(steps_ff):
		res_ff = model_ff.step_standalone(inp, dt, 0.0)

	var s_ff = model_ff.chassis_state
	var exp_w: float = -cfg_ff.gravity * t_ff
	var exp_z: float = z0 - 0.5 * cfg_ff.gravity * t_ff * t_ff
	var fz_ff: Array = res_ff["normal_forces"]
	var max_fz_ff: float = maxf(maxf(float(fz_ff[0]), float(fz_ff[1])), maxf(float(fz_ff[2]), float(fz_ff[3])))

	if absf(s_ff.linear_velocity.z - exp_w) < 1e-4 \
			and absf(s_ff.position.z - exp_z) < 1e-4 \
			and max_fz_ff == 0.0 \
			and absf(s_ff.angular_velocity.x) < 1e-6 \
			and absf(s_ff.angular_velocity.y) < 1e-6 \
			and absf(s_ff.angular_velocity.z - 2.0) < 1e-5:
		passed += 1
	else:
		failed += 1
		errors.append("Free-fall check failed: Z=%.4f (exp %.4f), w=%.4f (exp %.4f), maxFz=%.2f, w_rot=%s" % [
			s_ff.position.z, exp_z, s_ff.linear_velocity.z, exp_w, max_fz_ff, s_ff.angular_velocity
		])

	# ---------------------------------------------------------
	# 3. Emergent longitudinal load transfer under acceleration
	# ---------------------------------------------------------
	model.reset()
	var u0_acc: float = 5.0
	model.chassis_state.linear_velocity = Vector3(u0_acc, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u0_acc / cfg.wheel_radius
	inp.reset()
	inp.throttle = 0.8

	var res_acc: Dictionary = {}
	for _k in range(120):
		res_acc = model.step_standalone(inp, dt, 0.0)

	var fz_acc: Array = res_acc["normal_forces"]
	var front_acc: float = float(fz_acc[0]) + float(fz_acc[1])
	var rear_acc: float = float(fz_acc[2]) + float(fz_acc[3])
	var static_loads: Dictionary = cfg.get_static_axle_loads()
	var front_stat: float = float(static_loads["front_axle"])
	var rear_stat: float = float(static_loads["rear_axle"])

	if model.chassis_state.linear_velocity.x > u0_acc + 1.0 \
			and model.chassis_state.pitch < -1e-4 \
			and rear_acc > rear_stat + 100.0 \
			and front_acc < front_stat - 100.0:
		passed += 1
	else:
		failed += 1
		errors.append("Longitudinal load transfer failed: u=%.2f, pitch=%.5f, front=%.1f (stat %.1f), rear=%.1f (stat %.1f)" % [
			model.chassis_state.linear_velocity.x, model.chassis_state.pitch, front_acc, front_stat, rear_acc, rear_stat
		])

	# ---------------------------------------------------------
	# 4. Emergent lateral load transfer & left turn under positive steer
	# ---------------------------------------------------------
	model.reset()
	var u0_lat: float = 15.0
	model.chassis_state.linear_velocity = Vector3(u0_lat, 0.0, 0.0)
	for w in model.wheels:
		w.omega = u0_lat / cfg.wheel_radius
	inp.reset()
	inp.throttle = 0.15
	inp.steer_angle = 0.15 # positive ISO steer = left turn

	var res_lat: Dictionary = {}
	for _k in range(160):
		res_lat = model.step_standalone(inp, dt, 0.0)

	var fz_lat: Array = res_lat["normal_forces"]
	var s_lat = model.chassis_state
	if s_lat.angular_velocity.z > 0.1 \
			and s_lat.yaw > 0.02 \
			and s_lat.roll > 1e-4 \
			and float(fz_lat[1]) > float(fz_lat[0]) + 50.0 \
			and float(fz_lat[3]) > float(fz_lat[2]) + 50.0:
		passed += 1
	else:
		failed += 1
		errors.append("Lateral load transfer failed: r=%.3f, yaw=%.3f, roll=%.5f, Fz=[%.1f, %.1f, %.1f, %.1f]" % [
			s_lat.angular_velocity.z, s_lat.yaw, s_lat.roll,
			float(fz_lat[0]), float(fz_lat[1]), float(fz_lat[2]), float(fz_lat[3])
		])

	return {
		"name": "Chassis 14-DOF Integrator (Block C)",
		"passed": passed,
		"failed": failed,
		"errors": errors
	}
