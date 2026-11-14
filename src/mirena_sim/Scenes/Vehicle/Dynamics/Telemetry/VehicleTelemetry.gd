class_name VehicleTelemetry
extends RefCounted

## Vehicle Telemetry & Logging Interface for Mirena Dynamics
## Formats physical vehicle states for ROS bags and validation logs.

const VehicleDynamicsModel = preload("../Core/VehicleDynamicsModel.gd")

var timestamp: float = 0.0

# Kinematics (ROS Frame: +X Fwd, +Y Left, +Z Up)
var vx: float = 0.0
var vy: float = 0.0
var vz: float = 0.0
var ax: float = 0.0
var ay: float = 0.0
var az: float = 0.0
var yaw_rate: float = 0.0
var pitch_rate: float = 0.0
var roll_rate: float = 0.0
var steer_angle: float = 0.0

# Wheel speeds (rad/s)
var wheel_omega: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Powertrain & Brakes (Nm)
var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Tire forces (N)
var tire_fx: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fy: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fz: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Slip quantities
var tire_kappa: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_alpha: Array[float] = [0.0, 0.0, 0.0, 0.0]

# Torque Vectoring & Sideslip
var wz_ref: float = 0.0
var beta_deg: float = 0.0
var map_id: int = 0
var mz_req: float = 0.0

# Tire surface temperatures (deg C) [FL, FR, RL, RR]
var tire_temp: Array[float] = [25.0, 25.0, 25.0, 25.0]

# HV Powertrain Telemetry
var v_dc: float = 588.0
var i_dc: float = 0.0
var power_kw: float = 0.0
var energy_kwh: float = 0.0

## Resets all telemetry fields to nominal resting defaults.
func reset() -> void:
	timestamp = 0.0
	vx = 0.0
	vy = 0.0
	vz = 0.0
	ax = 0.0
	ay = 0.0
	az = 0.0
	yaw_rate = 0.0
	pitch_rate = 0.0
	roll_rate = 0.0
	steer_angle = 0.0
	wz_ref = 0.0
	beta_deg = 0.0
	map_id = 0
	mz_req = 0.0
	v_dc = 588.0
	i_dc = 0.0
	power_kw = 0.0
	energy_kwh = 0.0
	for i in range(4):
		wheel_omega[i] = 0.0
		drive_torques[i] = 0.0
		brake_torques[i] = 0.0
		tire_fx[i] = 0.0
		tire_fy[i] = 0.0
		tire_fz[i] = 0.0
		tire_kappa[i] = 0.0
		tire_alpha[i] = 0.0
		tire_temp[i] = 25.0

## Computes chassis sideslip angle in degrees from body velocities (ISO +X fwd, +Y left).
static func compute_beta_deg(vx_mps: float, vy_mps: float) -> float:
	if sqrt(vx_mps * vx_mps + vy_mps * vy_mps) < 0.2:
		return 0.0
	return rad_to_deg(atan2(vy_mps, maxf(absf(vx_mps), 0.5)))

## Safely reads tire surface temperatures (deg C) whether WheelDynamics.t_surf_c is present or not.
static func extract_tire_temps(model: VehicleDynamicsModel, extra: Dictionary = {}) -> Array[float]:
	var out: Array[float] = [25.0, 25.0, 25.0, 25.0]
	var ext_arr = extra.get("tire_temp", extra.get("tire_temps", null))
	if ext_arr is Array and ext_arr.size() >= 4:
		for i in range(4):
			out[i] = float(ext_arr[i])
		return out
	if model == null:
		return out
	for i in range(min(4, model.wheels.size())):
		var w = model.wheels[i]
		if w == null:
			continue
		if "t_surf_c" in w:
			out[i] = float(w.get("t_surf_c"))
		elif "tire_temp_c" in w:
			out[i] = float(w.get("tire_temp_c"))
		elif "thermal_state" in w:
			var th = w.get("thermal_state")
			if th != null and "t_surf_c" in th:
				out[i] = float(th.get("t_surf_c"))
	return out

## Safely extracts Torque Vectoring state (map_id, wz_ref, mz_req) from ECU, inputs, or extra.
static func extract_tv_state(model: VehicleDynamicsModel, steer_rad: float, extra: Dictionary = {}) -> Dictionary:
	var forces_data: Dictionary = extra.get("forces", {}) if extra.get("forces") is Dictionary else {}
	var map_val: int = 0
	var wz_ref_val: float = 0.0
	var mz_req_val: float = 0.0

	var ecu = model.ecu_tv_v3 if model != null else null
	var inp = model.inputs if model != null else null

	# Resolve map_id
	if extra.has("map_id"):
		map_val = int(extra["map_id"])
	elif forces_data.has("map_id"):
		map_val = int(forces_data["map_id"])
	elif ecu != null and ecu.has_method("get_map"):
		map_val = int(ecu.call("get_map"))
	elif ecu != null and "active_map_id" in ecu:
		map_val = int(ecu.get("active_map_id"))
	elif ecu != null and "active_map" in ecu:
		map_val = int(ecu.get("active_map"))
	elif ecu != null and "current_map" in ecu:
		map_val = int(ecu.get("current_map"))
	elif ecu != null and "map_id" in ecu:
		map_val = int(ecu.get("map_id"))
	elif ecu != null and "tv_map" in ecu:
		map_val = int(ecu.get("tv_map"))
	elif inp != null and "tv_map" in inp:
		map_val = int(inp.get("tv_map"))
	elif inp != null and "map_id" in inp:
		map_val = int(inp.get("map_id"))
	elif inp != null:
		map_val = int(inp.ecu_driving_mode)

	var ecu_active: bool = (inp != null and inp.ecu_driving_mode == 1)

	# Resolve mz_req
	if extra.has("mz_req"):
		mz_req_val = float(extra["mz_req"])
	elif forces_data.has("mz_req"):
		mz_req_val = float(forces_data["mz_req"])
	elif ecu_active and ecu != null and "last_mz_req" in ecu:
		mz_req_val = float(ecu.get("last_mz_req"))
	elif inp != null:
		mz_req_val = float(inp.tv_yaw_moment_req)

	# Resolve wz_ref
	if extra.has("wz_ref"):
		wz_ref_val = float(extra["wz_ref"])
	elif forces_data.has("wz_ref"):
		wz_ref_val = float(forces_data["wz_ref"])
	elif ecu_active and ecu != null and "last_wz_ref" in ecu:
		wz_ref_val = float(ecu.get("last_wz_ref"))
	elif model != null:
		var vx_safe: float = maxf(absf(model.linear_velocity_ros.x), 0.5)
		var wb: float = model.config.wheelbase if model.config != null else 1.540
		var dz_steer: float = 0.0 if absf(steer_rad) < 0.0087 else steer_rad
		wz_ref_val = (vx_safe * dz_steer) / (wb + 0.015 * vx_safe * vx_safe)

	return {
		"map_id": map_val,
		"wz_ref": wz_ref_val,
		"mz_req": mz_req_val,
	}

## Safely extracts HV powertrain state (v_dc, i_dc, power_kw, energy_kwh) with analytical fallback.
static func extract_hv_state(
	model: VehicleDynamicsModel,
	drive_torques: Array,
	dt: float = 0.0,
	extra: Dictionary = {},
	prev_energy_kwh: float = 0.0
) -> Dictionary:
	var forces_data: Dictionary = extra.get("forces", {}) if extra.get("forces") is Dictionary else {}
	var hv_obj: Variant = null

	for key in ["hv_state", "hv_powertrain", "powertrain_state", "hv_powertrain_state"]:
		if extra.has(key) and extra[key] != null:
			hv_obj = extra[key]
			break
		if forces_data.has(key) and forces_data[key] != null:
			hv_obj = forces_data[key]
			break
		if model != null and key in model:
			var candidate = model.get(key)
			if candidate != null:
				hv_obj = candidate
				break

	if hv_obj != null:
		var v_val: float = 588.0
		var i_val: float = 0.0
		var p_val: float = 0.0
		var e_val: float = prev_energy_kwh
		if hv_obj is Dictionary:
			v_val = float(hv_obj.get("v_dc", v_val))
			i_val = float(hv_obj.get("i_dc", i_val))
			p_val = float(hv_obj.get("power_kw", p_val))
			e_val = float(hv_obj.get("energy_kwh", e_val))
		else:
			if "v_dc" in hv_obj:
				v_val = float(hv_obj.get("v_dc"))
			if "i_dc" in hv_obj:
				i_val = float(hv_obj.get("i_dc"))
			if "power_kw" in hv_obj:
				p_val = float(hv_obj.get("power_kw"))
			if "energy_kwh" in hv_obj:
				e_val = float(hv_obj.get("energy_kwh"))
		return {"v_dc": v_val, "i_dc": i_val, "power_kw": p_val, "energy_kwh": e_val, "authoritative": true}

	if extra.has("v_dc") or extra.has("power_kw") or forces_data.has("v_dc") or forces_data.has("power_kw"):
		var src: Dictionary = extra if (extra.has("v_dc") or extra.has("power_kw")) else forces_data
		var p_kw_direct: float = float(src.get("power_kw", 0.0))
		var v_dc_direct: float = float(src.get("v_dc", 588.0))
		var i_dc_direct: float = float(src.get("i_dc", (p_kw_direct * 1000.0) / maxf(v_dc_direct, 1.0)))
		var e_kwh_direct: float = float(src.get("energy_kwh", prev_energy_kwh + (p_kw_direct * maxf(dt, 0.0) / 3600.0)))
		return {"v_dc": v_dc_direct, "i_dc": i_dc_direct, "power_kw": p_kw_direct, "energy_kwh": e_kwh_direct, "authoritative": true}

	# Analytical fallback when HVPowertrainState is not attached
	var eff: float = 0.90
	if model != null and model.config != null:
		eff = maxf(model.config.drivetrain_efficiency, 1e-3)
	var p_elec_w: float = 0.0
	if model != null:
		for i in range(min(4, model.wheels.size())):
			var trq_i: float = float(drive_torques[i]) if i < drive_torques.size() else 0.0
			var w_i: float = absf(model.wheels[i].omega)
			if trq_i > 0.0:
				p_elec_w += (trq_i * w_i) / eff
	var p_kw_fb: float = p_elec_w * 0.001
	var v_oc: float = 588.0
	var r_int: float = 0.12
	var disc: float = maxf(v_oc * v_oc - 4.0 * r_int * p_elec_w, 100.0)
	var v_dc_fb: float = 0.5 * (v_oc + sqrt(disc))
	var i_dc_fb: float = p_elec_w / maxf(v_dc_fb, 1.0)
	var e_kwh_fb: float = prev_energy_kwh + (p_kw_fb * maxf(dt, 0.0) / 3600.0)
	return {"v_dc": v_dc_fb, "i_dc": i_dc_fb, "power_kw": p_kw_fb, "energy_kwh": e_kwh_fb, "authoritative": false}

## Updates telemetry fields directly from the simulated vehicle model.
func update_from_model(model: VehicleDynamicsModel, linear_accel_ros: Vector3, steer_rad: float, last_step_data: Dictionary = {}) -> void:
	vx = model.linear_velocity_ros.x
	vy = model.linear_velocity_ros.y
	vz = model.linear_velocity_ros.z
	
	ax = linear_accel_ros.x
	ay = linear_accel_ros.y
	az = linear_accel_ros.z
	
	roll_rate = model.angular_velocity_ros.x
	pitch_rate = model.angular_velocity_ros.y
	yaw_rate = model.angular_velocity_ros.z
	
	steer_angle = steer_rad
	beta_deg = compute_beta_deg(vx, vy)
	
	var forces_data: Dictionary = last_step_data.get("forces", last_step_data) if last_step_data.get("forces") is Dictionary else last_step_data
	var d_trqs: Array = forces_data.get("drive_torques", last_step_data.get("drive_torques", [0.0, 0.0, 0.0, 0.0]))
	var b_trqs: Array = forces_data.get("brake_torques", last_step_data.get("brake_torques", [0.0, 0.0, 0.0, 0.0]))
	
	for i in range(4):
		var w = model.wheels[i]
		wheel_omega[i] = w.omega
		drive_torques[i] = d_trqs[i] if d_trqs.size() > i else 0.0
		brake_torques[i] = b_trqs[i] if b_trqs.size() > i else 0.0
		tire_fx[i] = w.F_x
		tire_fy[i] = w.F_y
		tire_fz[i] = w.F_z
		tire_kappa[i] = w.kappa
		tire_alpha[i] = w.alpha

	var tv_info: Dictionary = extract_tv_state(model, steer_rad, last_step_data)
	wz_ref = float(tv_info["wz_ref"])
	map_id = int(tv_info["map_id"])
	mz_req = float(tv_info["mz_req"])

	tire_temp = extract_tire_temps(model, last_step_data)

	var dt_val: float = float(last_step_data.get("dt", 0.0))
	var hv_info: Dictionary = extract_hv_state(model, d_trqs, dt_val, last_step_data, energy_kwh)
	v_dc = float(hv_info["v_dc"])
	i_dc = float(hv_info["i_dc"])
	power_kw = float(hv_info["power_kw"])
	energy_kwh = float(hv_info["energy_kwh"])

## Formats telemetry as a Dictionary matching CSV columns for validation scripts.
func to_dict() -> Dictionary:
	return {
		"timestamp": timestamp,
		"vx": vx, "vy": vy, "vz": vz,
		"ax": ax, "ay": ay, "az": az,
		"yaw_rate": yaw_rate, "pitch_rate": pitch_rate, "roll_rate": roll_rate,
		"steer_angle": steer_angle,
		"wz_ref": wz_ref, "beta_deg": beta_deg,
		"map_id": map_id, "mz_req": mz_req,
		"omega_fl": wheel_omega[0], "omega_fr": wheel_omega[1],
		"omega_rl": wheel_omega[2], "omega_rr": wheel_omega[3],
		"drive_trq_fl": drive_torques[0], "drive_trq_fr": drive_torques[1],
		"drive_trq_rl": drive_torques[2], "drive_trq_rr": drive_torques[3],
		"brake_trq_fl": brake_torques[0], "brake_trq_fr": brake_torques[1],
		"brake_trq_rl": brake_torques[2], "brake_trq_rr": brake_torques[3],
		"fx_fl": tire_fx[0], "fx_fr": tire_fx[1], "fx_rl": tire_fx[2], "fx_rr": tire_fx[3],
		"fy_fl": tire_fy[0], "fy_fr": tire_fy[1], "fy_rl": tire_fy[2], "fy_rr": tire_fy[3],
		"fz_fl": tire_fz[0], "fz_fr": tire_fz[1], "fz_rl": tire_fz[2], "fz_rr": tire_fz[3],
		"kappa_fl": tire_kappa[0], "kappa_fr": tire_kappa[1],
		"kappa_rl": tire_kappa[2], "kappa_rr": tire_kappa[3],
		"alpha_fl": tire_alpha[0], "alpha_fr": tire_alpha[1],
		"alpha_rl": tire_alpha[2], "alpha_rr": tire_alpha[3],
		"tire_temp_fl": tire_temp[0], "tire_temp_fr": tire_temp[1],
		"tire_temp_rl": tire_temp[2], "tire_temp_rr": tire_temp[3],
		"v_dc": v_dc, "i_dc": i_dc, "power_kw": power_kw, "energy_kwh": energy_kwh
	}
