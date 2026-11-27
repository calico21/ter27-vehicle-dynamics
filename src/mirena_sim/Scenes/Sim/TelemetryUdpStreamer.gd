class_name TelemetryUdpStreamer
extends Node

## Non-blocking 60 Hz UDP telemetry streamer and command listener for the TeR27 telemetry backend.
## - Sends JSON telemetry frames to 127.0.0.1:9876
## - Listens for JSON control commands on 127.0.0.1:9877
## - Zero FPS impact, silent when no peer is listening, safe in headless mode.

const SetupProfileScript = preload("res://Scenes/Vehicle/Config/SetupProfile.gd")

const TELEMETRY_HOST: String = "127.0.0.1"
const TELEMETRY_PORT: int = 9876
const COMMAND_HOST: String = "127.0.0.1"
const COMMAND_PORT: int = 9877
const SEND_HZ: float = 60.0
const SEND_INTERVAL_S: float = 1.0 / SEND_HZ
const MAX_CMD_PACKETS_PER_FRAME: int = 8

var enabled: bool = true
var enable_in_headless: bool = false

var _sim_node: Node = null
var _tx_peer: PacketPeerUDP = null
var _rx_peer: PacketPeerUDP = null
var _sockets_open: bool = false
var _send_accum_s: float = 0.0
var _sim_time_s: float = 0.0


func bind_sim(sim_ref: Node) -> void:
	_sim_node = sim_ref


func _ready() -> void:
	if _sim_node == null:
		_sim_node = get_parent()
	if not enabled:
		return
	if _is_headless() and not enable_in_headless:
		return
	_open_sockets()


func _process(delta: float) -> void:
	if not enabled:
		return
	_sim_time_s += maxf(delta, 0.0)
	if not _sockets_open:
		return
	_poll_commands()
	_send_accum_s += maxf(delta, 0.0)
	if _send_accum_s >= SEND_INTERVAL_S:
		_send_accum_s = fmod(_send_accum_s, SEND_INTERVAL_S)
		_send_telemetry_frame()


func _exit_tree() -> void:
	close_sockets()


func _is_headless() -> bool:
	return DisplayServer.get_name().to_lower() == "headless"


func _open_sockets() -> void:
	if _sockets_open:
		return
	_tx_peer = PacketPeerUDP.new()
	var tx_err: int = _tx_peer.set_dest_address(TELEMETRY_HOST, TELEMETRY_PORT)
	if tx_err != OK:
		_tx_peer.close()
		_tx_peer = null

	_rx_peer = PacketPeerUDP.new()
	var rx_err: int = _rx_peer.bind(COMMAND_PORT, COMMAND_HOST)
	if rx_err != OK:
		_rx_peer.close()
		_rx_peer = null

	_sockets_open = (_tx_peer != null or _rx_peer != null)


func close_sockets() -> void:
	if _tx_peer != null:
		_tx_peer.close()
		_tx_peer = null
	if _rx_peer != null:
		_rx_peer.close()
		_rx_peer = null
	_sockets_open = false


func build_telemetry_payload() -> Dictionary:
	var car: Node = _get_car()
	var track: Node = _get_track()
	var model: Variant = car.model if (car != null and "model" in car) else null
	var cfg: Variant = model.config if (model != null and "config" in model) else null

	var vehicle_id: String = "ter27_4wd"
	if cfg != null and "vehicle_id" in cfg and str(cfg.vehicle_id) != "":
		vehicle_id = str(cfg.vehicle_id)

	var grip_profile: String = _resolve_grip_profile(model)
	var sensor_diag: Dictionary = {}
	if model != null and model.has_method("get_sensor_realism_diagnostics"):
		sensor_diag = model.get_sensor_realism_diagnostics()
	var sensor_profile: String = str(sensor_diag.get("mode_name", "SENSOR_IDEAL"))

	var state_dict: Dictionary = _build_state_14dof(car, model)
	var wheels_dict: Dictionary = _build_wheels_dict(model)
	var hv_dict: Dictionary = _build_hv_dict(model)
	var ekf_dict: Dictionary = _build_ekf_dict(model, sensor_diag, state_dict)
	var dil_delta_dict: Dictionary = _build_dil_delta_dict(track, state_dict)

	return {
		"timestamp_s": snappedf(_sim_time_s, 0.0001),
		"source": "godot_udp",
		"vehicle_id": vehicle_id,
		"grip_profile": grip_profile,
		"sensor_profile": sensor_profile,
		"state": state_dict,
		"wheels": wheels_dict,
		"hv": hv_dict,
		"ekf": ekf_dict,
		"dil_delta": dil_delta_dict,
	}


func _send_telemetry_frame() -> void:
	if _tx_peer == null:
		return
	var car: Node = _get_car()
	if car == null or not is_instance_valid(car):
		return
	var payload: Dictionary = build_telemetry_payload()
	var json_str: String = JSON.stringify(payload)
	# Ignore return code so absence of a UDP listener never logs warnings or stalls.
	_tx_peer.put_packet(json_str.to_utf8_buffer())


func _poll_commands() -> void:
	if _rx_peer == null:
		return
	var processed: int = 0
	while _rx_peer.get_available_packet_count() > 0 and processed < MAX_CMD_PACKETS_PER_FRAME:
		processed += 1
		var pkt: PackedByteArray = _rx_peer.get_packet()
		if pkt.is_empty():
			continue
		var raw_text: String = pkt.get_string_from_utf8()
		var parsed: Variant = JSON.parse_string(raw_text)
		if typeof(parsed) == TYPE_DICTIONARY:
			handle_command(parsed as Dictionary)


func handle_command(cmd_dict: Dictionary) -> bool:
	var cmd: String = str(cmd_dict.get("cmd", "")).strip_edges().to_lower()
	if cmd == "apply_setup":
		return _cmd_apply_setup(cmd_dict)
	elif cmd == "set_sensor_profile":
		return _cmd_set_sensor_profile(cmd_dict)
	elif cmd == "replay_seek":
		return _cmd_replay_seek(cmd_dict)
	return false


func _cmd_apply_setup(cmd_dict: Dictionary) -> bool:
	var car: Node = _get_car()
	if car == null or not ("model" in car) or car.model == null:
		return false
	var model: Variant = car.model
	if cmd_dict.has("preset") and str(cmd_dict.get("preset", "")).strip_edges() != "":
		var preset_name: String = str(cmd_dict.get("preset", "")).strip_edges()
		if model.has_method("apply_setup"):
			return bool(model.apply_setup(preset_name))
	var setup_payload: Variant = cmd_dict.get("setup", cmd_dict)
	if setup_payload is Dictionary and model.has_method("apply_setup_profile"):
		var prof := SetupProfileScript.new()
		var d: Dictionary = setup_payload as Dictionary
		if d.has("spring_rate_front_n_m"):
			prof.spring_rate_front_n_m = float(d["spring_rate_front_n_m"])
		if d.has("spring_rate_rear_n_m"):
			prof.spring_rate_rear_n_m = float(d["spring_rate_rear_n_m"])
		if d.has("arb_front_nm_rad"):
			prof.arb_front_nm_rad = float(d["arb_front_nm_rad"])
		if d.has("arb_rear_nm_rad"):
			prof.arb_rear_nm_rad = float(d["arb_rear_nm_rad"])
		if d.has("camber_front_deg"):
			prof.camber_front_deg = float(d["camber_front_deg"])
		if d.has("camber_rear_deg"):
			prof.camber_rear_deg = float(d["camber_rear_deg"])
		if d.has("toe_front_deg"):
			prof.toe_front_deg = float(d["toe_front_deg"])
		if d.has("toe_rear_deg"):
			prof.toe_rear_deg = float(d["toe_rear_deg"])
		if d.has("ride_height_front_m"):
			prof.ride_height_front_m = float(d["ride_height_front_m"])
		if d.has("ride_height_rear_m"):
			prof.ride_height_rear_m = float(d["ride_height_rear_m"])
		if d.has("brake_bias"):
			prof.brake_bias = float(d["brake_bias"])
		if d.has("tire_pressure_front_bar"):
			prof.tire_pressure_front_bar = float(d["tire_pressure_front_bar"])
		if d.has("tire_pressure_rear_bar"):
			prof.tire_pressure_rear_bar = float(d["tire_pressure_rear_bar"])
		if d.has("power_limit_kw"):
			prof.power_limit_kw = float(d["power_limit_kw"])
		if d.has("tv_map"):
			prof.tv_map = str(d["tv_map"])
		if d.has("control_law_mode"):
			prof.control_law_mode = str(d["control_law_mode"])
		return bool(model.apply_setup_profile(prof))
	return false


func _cmd_set_sensor_profile(cmd_dict: Dictionary) -> bool:
	var car: Node = _get_car()
	if car == null or not ("model" in car) or car.model == null:
		return false
	var model: Variant = car.model
	if not model.has_method("set_sensor_realism_mode"):
		return false
	var prof_str: String = str(cmd_dict.get("profile", "SENSOR_IDEAL")).strip_edges().to_upper()
	var mode_int: int = 1 if (prof_str == "SENSOR_TRACK_REALISM" or prof_str == "1") else 0
	model.set_sensor_realism_mode(mode_int)
	return true


func _cmd_replay_seek(cmd_dict: Dictionary) -> bool:
	var t_s: float = maxf(float(cmd_dict.get("t_s", 0.0)), 0.0)
	var track: Node = _get_track()
	if track != null and "dil_session" in track and track.dil_session != null:
		var dil: Node = track.dil_session
		if dil.has_method("seek_replay"):
			dil.seek_replay(t_s)
			return true
		if "current_lap_s" in dil:
			dil.current_lap_s = t_s
			return true
	return false


func _get_car() -> Node:
	if _sim_node != null and "car" in _sim_node:
		return _sim_node.car
	return null


func _get_track() -> Node:
	if _sim_node != null and "track" in _sim_node:
		return _sim_node.track
	return null


func _resolve_grip_profile(model: Variant) -> String:
	if model == null or not ("track_grip_profile" in model):
		return "FSG_RUBBERED"
	var idx: int = int(model.track_grip_profile)
	if idx == 2:
		return "FSG_RUBBERED"
	elif idx == 1:
		return "FSG_REAL_TRACK"
	return "TTC_LAB_PEAK"


func _build_state_14dof(car: Node, model: Variant) -> Dictionary:
	var vx: float = 0.0
	var vy: float = 0.0
	var vz: float = 0.0
	var ax_g: float = 0.0
	var ay_g: float = 0.0
	var yaw_rate: float = 0.0
	var steer_deg: float = 0.0
	var throttle: float = 0.0
	var brake: float = 0.0
	var x_m: float = 0.0
	var y_m: float = 0.0
	var z_m: float = 0.0
	var heading_rad: float = 0.0

	if model != null:
		if "linear_velocity_ros" in model:
			var v: Vector3 = model.linear_velocity_ros
			vx = v.x
			vy = v.y
			vz = v.z
		if "angular_velocity_ros" in model:
			yaw_rate = float(model.angular_velocity_ros.z)
		if "_prev_ax_ros" in model:
			ax_g = float(model._prev_ax_ros) / 9.81
		if "_prev_ay_ros" in model:
			ay_g = float(model._prev_ay_ros) / 9.81
		if "inputs" in model and model.inputs != null:
			if "steering_angle" in model.inputs:
				steer_deg = rad_to_deg(float(model.inputs.steering_angle))
			if "throttle" in model.inputs:
				throttle = float(model.inputs.throttle)
			if "brake" in model.inputs:
				brake = float(model.inputs.brake)

	if car != null and is_instance_valid(car) and car is Node3D:
		var xf: Transform3D = (car as Node3D).global_transform
		# Godot -> ISO 8855 (+X fwd = -Z_godot, +Y left = -X_godot, +Z up = +Y_godot)
		x_m = -xf.origin.z
		y_m = -xf.origin.x
		z_m = xf.origin.y
		heading_rad = xf.basis.get_euler().y
		if "gas" in car and throttle == 0.0:
			throttle = float(car.gas)
		if "brake" in car and brake == 0.0:
			brake = float(car.brake)
		if "steering" in car and steer_deg == 0.0:
			steer_deg = rad_to_deg(float(car.steering))

	var beta_deg: float = rad_to_deg(atan2(vy, maxf(absf(vx), 0.5)))

	return {
		"vx": vx,
		"vy": vy,
		"vz": vz,
		"ax_g": ax_g,
		"ay_g": ay_g,
		"yaw_rate_rad_s": yaw_rate,
		"beta_deg": beta_deg,
		"steer_deg": steer_deg,
		"throttle": throttle,
		"brake": brake,
		"x_m": x_m,
		"y_m": y_m,
		"z_m": z_m,
		"heading_rad": heading_rad,
	}


func _build_wheels_dict(model: Variant) -> Dictionary:
	var names: Array[String] = ["fl", "fr", "rl", "rr"]
	var out: Dictionary = {}
	var wheels: Array = []
	var drive_t: Array = []
	var regen_t: Array = []
	if model != null:
		if "wheels" in model and model.wheels is Array:
			wheels = model.wheels
		if "wheel_drive_torque_nm" in model and model.wheel_drive_torque_nm is Array:
			drive_t = model.wheel_drive_torque_nm
		if "wheel_regen_brake_nm" in model and model.wheel_regen_brake_nm is Array:
			regen_t = model.wheel_regen_brake_nm

	for i in range(4):
		var key: String = names[i]
		var fx_n: float = 0.0
		var fy_n: float = 0.0
		var fz_n: float = 0.0
		var alpha_deg: float = 0.0
		var kappa_val: float = 0.0
		var camber_deg: float = 0.0
		var travel_m: float = 0.0
		var temp_c: float = 55.0
		var motor_t_nm: float = 0.0
		if i < wheels.size() and wheels[i] != null:
			var w: Variant = wheels[i]
			fx_n = float(w.F_x) if "F_x" in w else 0.0
			fy_n = float(w.F_y) if "F_y" in w else 0.0
			fz_n = float(w.F_z) if "F_z" in w else 0.0
			alpha_deg = rad_to_deg(float(w.alpha)) if "alpha" in w else 0.0
			kappa_val = float(w.kappa) if "kappa" in w else 0.0
			camber_deg = rad_to_deg(float(w.camber_angle)) if "camber_angle" in w else 0.0
			travel_m = float(w.suspension_compression) if "suspension_compression" in w else 0.0
			temp_c = float(w.t_surf_c) if "t_surf_c" in w else 55.0
		if i < drive_t.size():
			motor_t_nm += float(drive_t[i])
		if i < regen_t.size():
			motor_t_nm -= float(regen_t[i])
		var f_comb: float = sqrt(fx_n * fx_n + fy_n * fy_n)
		var tire_util: float = clampf(f_comb / maxf(fz_n * 1.65, 1.0), 0.0, 1.5)
		out[key] = {
			"fx_n": fx_n,
			"fy_n": fy_n,
			"fz_n": fz_n,
			"slip_angle_deg": alpha_deg,
			"slip_ratio": kappa_val,
			"tire_util": tire_util,
			"camber_deg": camber_deg,
			"wheel_travel_m": travel_m,
			"motor_torque_nm": motor_t_nm,
			"temp_c": temp_c,
		}
	return out


func _build_hv_dict(model: Variant) -> Dictionary:
	var p_dc_kw: float = 0.0
	var p_regen_kw: float = 0.0
	var soc_pct: float = 95.0
	var bspd_active: bool = false
	var tv_mz_nm: float = 0.0

	if model != null:
		if "hv_state" in model and model.hv_state != null:
			var hv: Variant = model.hv_state
			if "soc" in hv:
				soc_pct = clampf(float(hv.soc) * 100.0, 0.0, 100.0)
			if "last_dc_power_w" in hv:
				var pw_kw: float = float(hv.last_dc_power_w) * 0.001
				p_dc_kw = maxf(pw_kw, 0.0)
				p_regen_kw = maxf(-pw_kw, 0.0)
		if "ecu_tv_4wd" in model and model.ecu_tv_4wd != null:
			var ecu: Variant = model.ecu_tv_4wd
			if "last_power_w" in ecu and p_dc_kw == 0.0 and p_regen_kw == 0.0:
				var pw2: float = float(ecu.last_power_w) * 0.001
				p_dc_kw = maxf(pw2, 0.0)
				p_regen_kw = maxf(-pw2, 0.0)
			if "last_mz_cmd_nm" in ecu:
				tv_mz_nm = float(ecu.last_mz_cmd_nm)
			elif "last_mz_nm" in ecu:
				tv_mz_nm = float(ecu.last_mz_nm)
		if "inputs" in model and model.inputs != null:
			var brk: float = float(model.inputs.brake) if "brake" in model.inputs else 0.0
			bspd_active = (brk > 0.15 and p_dc_kw > 5.0)

	return {
		"p_dc_kw": p_dc_kw,
		"p_regen_kw": p_regen_kw,
		"soc_pct": soc_pct,
		"bspd_active": bspd_active,
		"tv_mz_nm": tv_mz_nm,
	}


func _build_ekf_dict(model: Variant, sensor_diag: Dictionary, state_dict: Dictionary) -> Dictionary:
	var ekf_vx: float = float(state_dict.get("vx", 0.0))
	var ekf_vy: float = float(state_dict.get("vy", 0.0))
	var ekf_wz: float = float(state_dict.get("yaw_rate_rad_s", 0.0))
	var ekf_beta_deg: float = float(state_dict.get("beta_deg", 0.0))
	var latency_ms: float = float(sensor_diag.get("can_accel_latency_ms", 0.0))

	if model != null and "sensor_realism" in model and model.sensor_realism != null:
		var sr: Variant = model.sensor_realism
		if "_ekf_vx" in sr:
			ekf_vx = float(sr._ekf_vx)
		if "_ekf_vy" in sr:
			ekf_vy = float(sr._ekf_vy)
		if "_beta_hat" in sr:
			ekf_beta_deg = rad_to_deg(float(sr._beta_hat))
		if "mode" in sr and int(sr.mode) == 0:
			latency_ms = 0.0

	return {
		"ekf_vx": ekf_vx,
		"ekf_vy": ekf_vy,
		"ekf_yaw_rate": ekf_wz,
		"ekf_beta_deg": ekf_beta_deg,
		"sensor_latency_ms": latency_ms,
	}


func _build_dil_delta_dict(track: Node, state_dict: Dictionary) -> Dictionary:
	var delta_time_s: float = 0.0
	var delta_vx_mps: float = 0.0
	var delta_ay_g: float = 0.0

	if track != null and "dil_session" in track and track.dil_session != null:
		var dil: Node = track.dil_session
		if dil.has_method("get_live_dil_telemetry"):
			var snap: Dictionary = dil.get_live_dil_telemetry()
			delta_time_s = float(snap.get("delta_t_s", 0.0))
			delta_vx_mps = float(snap.get("delta_vx_ms", 0.0))
			if snap.has("ay_ref_mps2"):
				var ay_ref_g: float = float(snap.get("ay_ref_mps2", 0.0)) / 9.81
				delta_ay_g = float(state_dict.get("ay_g", 0.0)) - ay_ref_g

	return {
		"delta_time_s": delta_time_s,
		"delta_vx_mps": delta_vx_mps,
		"delta_ay_g": delta_ay_g,
	}
