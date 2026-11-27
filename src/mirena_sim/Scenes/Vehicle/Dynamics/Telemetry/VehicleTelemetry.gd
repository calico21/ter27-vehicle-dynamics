class_name VehicleTelemetry
extends RefCounted

## vehicle telemetry & logging interface for mirena dynamics
## formats physical vehicle states for ros bags and validation logs

const VehicleDynamicsModel = preload("../Models/full_14dof/Full14DofModel.gd")

var timestamp: float = 0.0

# kinematics (ros frame: +x fwd, +y left, +z up)
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

# wheel speeds (rad/s)
var wheel_omega: Array[float] = [0.0, 0.0, 0.0, 0.0]

# powertrain & brakes (nm)
var drive_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]
var brake_torques: Array[float] = [0.0, 0.0, 0.0, 0.0]

# tire forces (n)
var tire_fx: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fy: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_fz: Array[float] = [0.0, 0.0, 0.0, 0.0]

# slip quantities
var tire_kappa: Array[float] = [0.0, 0.0, 0.0, 0.0]
var tire_alpha: Array[float] = [0.0, 0.0, 0.0, 0.0]

# torque vectoring & sideslip
var wz_ref: float = 0.0
var beta_deg: float = 0.0
var map_id: int = 0
var mz_req: float = 0.0

# tire surface temperatures (deg c) [fl, fr, rl, rr]
var tire_temp: Array[float] = [25.0, 25.0, 25.0, 25.0]

# hv powertrain telemetry
var v_dc: float = 588.0
var i_dc: float = 0.0
var power_kw: float = 0.0
var energy_kwh: float = 0.0

# brake blending, disc thermal & surface grip telemetry
var brake_bias_front: float = 0.58
var regen_blending_enabled: bool = true
var disc_temp: Array[float] = [40.0, 40.0, 40.0, 40.0]
var surface_mu: Array[float] = [1.0, 1.0, 1.0, 1.0]
var regen_power_kw: float = 0.0

## resets all telemetry fields to nominal resting defaults
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
	brake_bias_front = 0.58
	regen_blending_enabled = true
	regen_power_kw = 0.0
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
		disc_temp[i] = 40.0
		surface_mu[i] = 1.0

## computes chassis sideslip angle in degrees from body velocities (iso +x fwd, +y left)
static func compute_beta_deg(vx_mps: float, vy_mps: float) -> float:
	if sqrt(vx_mps * vx_mps + vy_mps * vy_mps) < 0.2:
		return 0.0
	return rad_to_deg(atan2(vy_mps, maxf(absf(vx_mps), 0.5)))

## safely reads tire surface temperatures (deg c) whether wheeldynamics.t_surf_c is present or not
static func extract_tire_temps(model, extra: Dictionary = {}) -> Array[float]:
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

## safely extracts torque vectoring state (map_id, wz_ref, mz_req) from ecu, inputs, or extra
static func extract_tv_state(model, steer_rad: float, extra: Dictionary = {}) -> Dictionary:
	var forces_data: Dictionary = extra.get("forces", {}) if extra.get("forces") is Dictionary else {}
	var map_val: int = 0
	var wz_ref_val: float = 0.0
	var mz_req_val: float = 0.0

	var ecu = model.ecu_tv_v3 if model != null else null
	var inp = model.inputs if model != null else null

	# resolve map_id
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

	# resolve mz_req
	if extra.has("mz_req"):
		mz_req_val = float(extra["mz_req"])
	elif forces_data.has("mz_req"):
		mz_req_val = float(forces_data["mz_req"])
	elif ecu_active and ecu != null and "last_mz_req" in ecu:
		mz_req_val = float(ecu.get("last_mz_req"))
	elif inp != null:
		mz_req_val = float(inp.tv_yaw_moment_req)

	# resolve wz_ref
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

## safely extracts hv powertrain state (v_dc, i_dc, power_kw, energy_kwh) with analytical fallback
static func extract_hv_state(
	model,
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

	# analytical fallback when hvpowertrainstate is not attached
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

## safely extracts brake blending, disc temperatures, and per-wheel surface grip from model or step_result
static func extract_brake_and_surface_state(model, extra: Dictionary = {}) -> Dictionary:
	var forces_data: Dictionary = extra.get("forces", {}) if extra.get("forces") is Dictionary else {}
	var bias_val: float = 0.58
	if extra.has("brake_bias_front"):
		bias_val = float(extra["brake_bias_front"])
	elif forces_data.has("brake_bias_front"):
		bias_val = float(forces_data["brake_bias_front"])
	elif model != null and model.config != null:
		bias_val = float(model.config.brake_bias_front)

	var regen_en: bool = true
	if extra.has("regen_blending_enabled"):
		regen_en = bool(extra["regen_blending_enabled"])
	elif forces_data.has("regen_blending_enabled"):
		regen_en = bool(forces_data["regen_blending_enabled"])
	elif model != null and model.config != null:
		var cfg_regen: Variant = model.config.get("regen_blending_enabled")
		if typeof(cfg_regen) == TYPE_BOOL:
			regen_en = bool(cfg_regen)

	var brake_state_obj: Variant = extra.get("brake_state", forces_data.get("brake_state", null))
	if brake_state_obj == null and model != null and "brake_state" in model:
		brake_state_obj = model.get("brake_state")

	var disc_temps_out: Array[float] = [40.0, 40.0, 40.0, 40.0]
	var dt_arr: Variant = extra.get("disc_temps", extra.get("disc_temp", forces_data.get("disc_temps", forces_data.get("disc_temp", null))))
	if dt_arr is Array and dt_arr.size() >= 4:
		for i in range(4):
			disc_temps_out[i] = float(dt_arr[i])
	elif brake_state_obj != null:
		var bs_temps: Variant = brake_state_obj.get("disc_temp", null) if brake_state_obj is Dictionary else brake_state_obj.get("disc_temp")
		if bs_temps is Array and bs_temps.size() >= 4:
			for i in range(4):
				disc_temps_out[i] = float(bs_temps[i])

	var regen_kw: float = 0.0
	if extra.has("regen_power_kw"):
		regen_kw = float(extra["regen_power_kw"])
	elif forces_data.has("regen_power_kw"):
		regen_kw = float(forces_data["regen_power_kw"])
	elif extra.has("p_regen_w"):
		regen_kw = float(extra["p_regen_w"]) * 0.001
	elif forces_data.has("p_regen_w"):
		regen_kw = float(forces_data["p_regen_w"]) * 0.001
	elif brake_state_obj != null:
		var p_reg_raw: Variant = brake_state_obj.get("p_regen_w", null) if brake_state_obj is Dictionary else brake_state_obj.get("p_regen_w")
		if typeof(p_reg_raw) == TYPE_FLOAT or typeof(p_reg_raw) == TYPE_INT:
			regen_kw = float(p_reg_raw) * 0.001

	var surf_mu_out: Array[float] = [1.0, 1.0, 1.0, 1.0]
	var sm_arr: Variant = extra.get("surface_mu", extra.get("surface_mus", forces_data.get("surface_mu", forces_data.get("surface_mus", null))))
	if sm_arr is Array and sm_arr.size() >= 4:
		for i in range(4):
			surf_mu_out[i] = float(sm_arr[i])
	elif model != null:
		for i in range(mini(4, model.wheels.size())):
			var w = model.wheels[i]
			if w != null and "surface_mu" in w:
				surf_mu_out[i] = float(w.get("surface_mu"))

	return {
		"brake_bias_front": bias_val,
		"regen_blending_enabled": regen_en,
		"disc_temp": disc_temps_out,
		"surface_mu": surf_mu_out,
		"regen_power_kw": regen_kw,
	}

## updates telemetry fields directly from the simulated vehicle model
func update_from_model(model, linear_accel_ros: Vector3, steer_rad: float, last_step_data: Dictionary = {}) -> void:
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

	var bs_info: Dictionary = extract_brake_and_surface_state(model, last_step_data)
	brake_bias_front = float(bs_info["brake_bias_front"])
	regen_blending_enabled = bool(bs_info["regen_blending_enabled"])
	disc_temp = bs_info["disc_temp"]
	surface_mu = bs_info["surface_mu"]
	regen_power_kw = float(bs_info["regen_power_kw"])

## formats telemetry as a dictionary matching csv columns for validation scripts
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
		"v_dc": v_dc, "i_dc": i_dc, "power_kw": power_kw, "energy_kwh": energy_kwh,
		"brake_bias_front": brake_bias_front,
		"regen_blending_enabled": regen_blending_enabled,
		"disc_temp_fl": disc_temp[0], "disc_temp_fr": disc_temp[1],
		"disc_temp_rl": disc_temp[2], "disc_temp_rr": disc_temp[3],
		"surface_mu_fl": surface_mu[0], "surface_mu_fr": surface_mu[1],
		"surface_mu_rl": surface_mu[2], "surface_mu_rr": surface_mu[3],
		"regen_power_kw": regen_power_kw
	}
