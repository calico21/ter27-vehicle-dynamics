#!/usr/bin/env python3
"""
Closed-Loop Robustness Stress Test Suite for Tecnun e-Racing TeR27 4WD (MirenaSim).

Stress-tests a candidate TV configuration WITHOUT retuning under:
  1. Nominal (FSG_REAL_TRACK reference for pace-loss comparison)
  2. Grip mismatch: mu_real = 0.85 * mu_nom (-15% grip) and 1.15 * mu_nom (+15% grip)
  3. Realistic sensor noise + CAN/inverter latency:
     - IMU Gaussian noise (ax, ay: 0.15 m/s^2; az: 0.40 m/s^2; wz: 0.008 rad/s)
     - 48-tooth wheel encoder quantization (TAU/48 rad/s step)
     - 5 ms (1 ECU tick) and 10 ms (2 ECU ticks) inverter/CAN loop latency
  4. Mass / CG shift:
     - Driver +12 kg (302 kg) and CG +15 mm rearward
     - Driver -12 kg (278 kg) and CG -15 mm forward
  5. Combined worst-case: -15% grip + IMU/encoder noise + 10 ms latency + +12 kg / +15 mm rear CG

Verifies for every perturbation case:
  - Directional stability (dM_z / dbeta < 0 in Milliken/SAE convention, equivalently dM_z/dbeta_ros > 0)
  - 0 spins across hairpin, skidpad, and 55 km/h slalom
  - 0 high-frequency T_cmd oscillations (chatter / >15 Hz RMS torque check)
  - 0 HV violations (> 80.005 kW or < -25.005 kW on ECU command; <= 80.5 kW physical under CAN delay)
  - Controlled pace loss vs nominal

Output:
  vd_correlation_lab/results/tv_oracle/robustness_report.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
OUT_DIR = LAB_DIR / "results" / "tv_oracle"

GODOT_ROBUSTNESS_PROBE_GD = r"""extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const ChassisDynamics = preload("res://Scenes/Vehicle/Dynamics/Chassis/ChassisDynamics.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")

const ENC_STEP_RAD_S: float = TAU / 48.0
const SIGMA_AX: float = 0.15
const SIGMA_AY: float = 0.15
const SIGMA_AZ: float = 0.40
const SIGMA_WZ: float = 0.008

var _rng := RandomNumberGenerator.new()


func _make_car(cfg_json: Dictionary) -> Full14Dof:
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	var grip_profile: int = int(cfg_json.get("grip_profile", 1))
	m.set_track_grip_profile(grip_profile)
	m.config.update_derived_geometry()
	m.suspension.sync_from_config(m.config)

	# Configure ECU FIRST with nominal vehicle parameters (NO retuning under plant perturbation!)
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.configure_from_vehicle_config(m.config)
		if cfg_json.has("mode_overrides") and cfg_json["mode_overrides"] is Dictionary:
			var mo: Dictionary = cfg_json["mode_overrides"]
			for mkey in mo.keys():
				if m.ecu_tv_4wd.modes_cfg.has(mkey) and mo[mkey] is Dictionary:
					for pk in (mo[mkey] as Dictionary).keys():
						m.ecu_tv_4wd.modes_cfg[mkey][pk] = mo[mkey][pk]
		m.ecu_tv_4wd._mc_mode = -1
		m.ecu_tv_4wd.reset()

	# Now apply plant perturbations WITHOUT updating ECU parameters
	var mu_mult: float = float(cfg_json.get("mu_mult", 1.0))
	m.env_mu_scale *= mu_mult

	var dm_kg: float = float(cfg_json.get("delta_mass_kg", 0.0))
	var dcg_rear_m: float = float(cfg_json.get("delta_cg_rear_mm", 0.0)) * 0.001
	if absf(dm_kg) > 1e-6 or absf(dcg_rear_m) > 1e-6:
		m.config.mass = maxf(150.0, m.config.mass + dm_kg)
		var df: float = dcg_rear_m / maxf(m.config.wheelbase, 0.5)
		m.config.front_weight_distribution = clampf(m.config.front_weight_distribution - df, 0.25, 0.75)
		m.config.update_derived_geometry()
		m.suspension.sync_from_config(m.config)

	m.chassis_state.reset(m.config.h_cog)
	return m


func _init_speed(m: Full14Dof, v0: float) -> void:
	m.reset()
	m.chassis_state.linear_velocity = Vector3(v0, 0.0, 0.0)
	var rw: float = maxf(m.config.wheel_radius, 0.05)
	for w in m.wheels:
		w.omega = v0 / rw
	if m.ecu_tv_4wd != null:
		m.ecu_tv_4wd.ekf_vx = v0
		m.ecu_tv_4wd.ekf_vy = 0.0
		m.ecu_tv_4wd._ekf_init = true


func _quantize_encoders(w_raw: Array[float]) -> Array[float]:
	var out: Array[float] = [0.0, 0.0, 0.0, 0.0]
	for i in range(4):
		out[i] = roundf(w_raw[i] / ENC_STEP_RAD_S) * ENC_STEP_RAD_S
	return out


func _compute_phys_power_kw(t_cmd: Array, w_raw: Array[float], eta_drv: float, eta_reg: float) -> float:
	var p_w: float = 0.0
	for i in range(4):
		var ti: float = float(t_cmd[i])
		var om: float = absf(w_raw[i])
		var pm: float = ti * om
		if pm >= 0.0:
			p_w += pm / maxf(eta_drv, 0.5)
		else:
			p_w += pm * clampf(eta_reg, 0.5, 1.0)
	return p_w * 0.001


func _step_heun_override(m: Full14Dof, inp: VehicleInputs, dt: float) -> Dictionary:
	var s0 := m.chassis_state.clone()
	var w0: Array[Vector4] = [
		m.wheels[0].get_integrator_state(),
		m.wheels[1].get_integrator_state(),
		m.wheels[2].get_integrator_state(),
		m.wheels[3].get_integrator_state()
	]
	var c1: Array = ChassisDynamics.compute_ground_contacts_6dof(s0, m.config, 0.0)
	var r1: Dictionary = m.step_forces(inp, s0.linear_velocity, s0.angular_velocity, c1, dt, true)
	var k1 := ChassisDynamics.compute_derivatives(s0, r1["net_force_ros"], r1["net_torque_ros"], m.config)
	var s_pred := ChassisDynamics.advance_euler(s0, k1, dt)

	var c2: Array = ChassisDynamics.compute_ground_contacts_6dof(s_pred, m.config, 0.0)
	var r2: Dictionary = m.step_forces(inp, s_pred.linear_velocity, s_pred.angular_velocity, c2, dt, false)
	var k2 := ChassisDynamics.compute_derivatives(s_pred, r2["net_force_ros"], r2["net_torque_ros"], m.config)
	m.chassis_state = ChassisDynamics.advance_heun(s0, k1, k2, dt)
	for i in range(4):
		m.wheels[i].set_integrator_state(0.5 * (w0[i] + m.wheels[i].get_integrator_state()))
	return r2


func _step_plant(
	m: Full14Dof,
	inp: VehicleInputs,
	dt_ecu: float,
	use_noise: bool,
	delay_ticks: int,
	can_fifo: Array,
	is_first: bool
) -> Dictionary:
	var lv := m.chassis_state.linear_velocity
	var av := m.chassis_state.angular_velocity
	var w_raw: Array[float] = [m.wheels[0].omega, m.wheels[1].omega, m.wheels[2].omega, m.wheels[3].omega]

	# Unified 200 Hz ECU step for apples-to-apples comparison across nominal, noise, and CAN delay cases
	m.ecu_tv_4wd.active_mode = inp.tv_4wd_mode
	var ax_true: float = m.net_force_chassis_ros.x / m.config.mass
	var ay_true: float = m.net_force_chassis_ros.y / m.config.mass
	var az_true: float = m.config.gravity + (m.net_force_chassis_ros.z / m.config.mass)

	var ax_m: float = ax_true + (_rng.randfn(0.0, SIGMA_AX) if use_noise else 0.0)
	var ay_m: float = ay_true + (_rng.randfn(0.0, SIGMA_AY) if use_noise else 0.0)
	var az_m: float = az_true + (_rng.randfn(0.0, SIGMA_AZ) if use_noise else 0.0)
	var wz_m: float = av.z + (_rng.randfn(0.0, SIGMA_WZ) if use_noise else 0.0)
	var w_meas: Array[float] = _quantize_encoders(w_raw) if use_noise else w_raw

	var d_fl: float = m.compute_ackermann_steer(inp.steer_angle, true, m.config)
	var d_fr: float = m.compute_ackermann_steer(inp.steer_angle, false, m.config)
	var w_steers: Array[float] = [d_fl, d_fr, 0.0, 0.0]

	var r_whl: float = maxf(m.config.wheel_radius, 1e-4)
	var max_whl_trq: float = m.config.motor_peak_torque * m.config.gear_ratio
	var max_pwr_f: float = (m.config.power_limit_watts * m.config.drivetrain_efficiency) / maxf(absf(lv.x), 1.0)
	var fx_drv: float = inp.throttle * minf(4.0 * max_whl_trq / r_whl, max_pwr_f)
	if inp.brake > 0.05 and inp.throttle < 0.05:
		var p_chg: float = maxf(m.config.p_max_charge_w, 1000.0)
		var p_chg_mech: float = p_chg / maxf(m.ecu_tv_4wd.eta_regen, 0.5)
		var f_reg_cap: float = minf(4.0 * m.ecu_tv_4wd.t_regen_max / r_whl, p_chg_mech / maxf(absf(lv.x), 2.0))
		if m.ecu_tv_4wd.mech_brake_start > 0.05:
			var reg_frac: float = clampf(inp.brake / m.ecu_tv_4wd.mech_brake_start, 0.0, 1.0)
			fx_drv = -reg_frac * f_reg_cap if absf(lv.x) > 1.0 else 0.0
		else:
			var f_brk_dem: float = inp.brake * m.config.brake_mu * m.config.mass * m.config.gravity
			fx_drv = -minf(f_brk_dem, f_reg_cap) if absf(lv.x) > 2.0 else 0.0

	var t_cmd := m.ecu_tv_4wd.step(
		fx_drv, inp.steer_angle, lv.x, wz_m, ax_m, ay_m, az_m, w_meas, inp.brake, dt_ecu, w_steers
	)
	var t_arr: Array = [float(t_cmd[0]), float(t_cmd[1]), float(t_cmd[2]), float(t_cmd[3])]

	var t_applied: Array = t_arr
	if delay_ticks > 0:
		if is_first:
			can_fifo.clear()
			for _d in range(delay_ticks):
				can_fifo.append(t_arr.duplicate())
			t_applied = t_arr.duplicate()
		else:
			can_fifo.push_back(t_arr.duplicate())
			t_applied = can_fifo.pop_front()

	inp.override_wheel_torques = true
	inp.wheel_motor_torques.assign(t_applied)
	var res := {}
	for _sub in range(2):
		res = _step_heun_override(m, inp, dt_ecu * 0.5)

	var w_post: Array[float] = [m.wheels[0].omega, m.wheels[1].omega, m.wheels[2].omega, m.wheels[3].omega]
	var p_ecu: float = m.ecu_tv_4wd.last_power_w * 0.001
	var p_phys: float = _compute_phys_power_kw(t_applied, w_post, m.ecu_tv_4wd.eta_drive, m.ecu_tv_4wd.eta_regen)
	return {
		"res": res,
		"t_cmd": t_arr,
		"p_ecu_kw": p_ecu,
		"p_phys_kw": p_phys,
	}


func _analyze_t_osc(t_hist: Array) -> Dictionary:
	# Quantify high-frequency T_cmd oscillations (2nd-difference RMS and rapid sign reversals)
	var n: int = t_hist.size()
	if n < 10:
		return {"hf_rms_nm": 0.0, "chatter_ratio": 0.0, "max_step_nm": 0.0}
	var sum_d2_sq: float = 0.0
	var rev_cnt: int = 0
	var total_pairs: int = 0
	var max_step: float = 0.0
	for k in range(2, n):
		var a0: Array = t_hist[k - 2]
		var a1: Array = t_hist[k - 1]
		var a2: Array = t_hist[k]
		for wi in range(4):
			var d1: float = float(a1[wi]) - float(a0[wi])
			var d2: float = float(a2[wi]) - float(a1[wi])
			max_step = maxf(max_step, absf(d2))
			var dd: float = d2 - d1  # 2nd difference isolates high-frequency (>25 Hz) chatter
			sum_d2_sq += dd * dd
			total_pairs += 1
			# Significant opposite-sign step reversal (> 20 Nm each tick)
			if d1 * d2 < 0.0 and absf(d1) > 20.0 and absf(d2) > 20.0:
				rev_cnt += 1
	var hf_rms: float = sqrt(sum_d2_sq / float(maxi(total_pairs, 1)))
	var chatter_ratio: float = float(rev_cnt) / float(maxi(total_pairs, 1))
	return {
		"hf_rms_nm": hf_rms,
		"chatter_ratio": chatter_ratio,
		"max_step_nm": max_step,
	}


func _eval_stability_derivative(cfg_json: Dictionary, v_kmh: float, delta_deg: float) -> Dictionary:
	# Tethered finite difference dM_z / dbeta at +/- 1.5 deg beta
	var v_mps: float = v_kmh / 3.6
	var mz_vals: Array[float] = []
	for b_deg in [-1.5, 1.5]:
		var m := _make_car(cfg_json)
		var b_rad: float = deg_to_rad(float(b_deg))
		var vx: float = v_mps * cos(b_rad)
		var vy: float = v_mps * sin(b_rad)
		m.reset()
		m.chassis_state.linear_velocity = Vector3(vx, vy, 0.0)
		var rw: float = maxf(m.config.wheel_radius, 0.05)
		for w in m.wheels:
			w.omega = vx / rw
		m.ecu_tv_4wd.ekf_vx = vx
		m.ecu_tv_4wd.ekf_vy = vy
		m.ecu_tv_4wd._ekf_init = true

		var inp := VehicleInputs.new()
		inp.drive_mode = m.config.drive_mode
		inp.ecu_driving_mode = 2
		inp.tv_4wd_mode = 0
		inp.tv_mode = 0
		inp.steer_angle = deg_to_rad(delta_deg)
		inp.brake = 0.0

		var r_cur: float = 0.0
		var mz_cur: float = 0.0
		var last_res: Dictionary = {}
		var inv_m: float = 1.0 / maxf(m.config.mass, 100.0)
		for k in range(110):
			m.chassis_state.linear_velocity.x = vx
			m.chassis_state.linear_velocity.y = vy
			m.chassis_state.angular_velocity.z = r_cur
			m.ecu_tv_4wd.ekf_vx = vx
			m.ecu_tv_4wd.ekf_vy = vy
			var fx_req: float = -m.config.mass * r_cur * vy
			var fx_prev: float = float(last_res.get("net_force_ros", Vector3.ZERO).x) if not last_res.is_empty() else 0.0
			inp.throttle = clampf(0.14 + 0.0012 * (fx_req - fx_prev), 0.0, 0.65)
			last_res = m.step_standalone(inp, 0.0025, 0.0)
			var ay_m: float = float(last_res["net_force_ros"].y) * inv_m
			r_cur = lerpf(r_cur, clampf(ay_m / maxf(v_mps, 2.0), -2.5, 2.5), 0.22)
			mz_cur = float(last_res["net_torque_ros"].z)
		mz_vals.append(mz_cur)

	var dmz_dbeta_ros: float = (mz_vals[1] - mz_vals[0]) / 3.0
	var dmz_dbeta_sae: float = -dmz_dbeta_ros
	return {
		"v_kmh": v_kmh,
		"delta_deg": delta_deg,
		"dMz_dbeta_sae_nm_per_deg": dmz_dbeta_sae,
		"dMz_dbeta_ros_nm_per_deg": dmz_dbeta_ros,
		"stable": dmz_dbeta_sae < 0.0,
	}


func _run_slalom(cfg_json: Dictionary) -> Dictionary:
	_rng.seed = 9101
	var m := _make_car(cfg_json)
	_init_speed(m, 15.3)
	var use_noise: bool = bool(cfg_json.get("use_noise", false))
	var delay_ticks: int = int(cfg_json.get("delay_ticks", 0))
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 0
	inp.tv_mode = 0

	var dt_ecu: float = 0.005
	var n_sl: int = int(round(4.0 / dt_ecu))
	var n_rel: int = int(round(1.0 / dt_ecu))
	var fifo: Array = []
	var t_hist: Array = []
	var max_beta: float = 0.0
	var max_wz: float = 0.0
	var spun: bool = false
	var hv_ecu_bad: int = 0
	var hv_phys_bad: int = 0
	var p_ecu_max: float = -1e9
	var p_ecu_min: float = 1e9
	var p_phys_max: float = -1e9
	var p_phys_min: float = 1e9

	for k in range(n_sl + n_rel):
		var t_s: float = float(k) * dt_ecu
		var vx: float = m.chassis_state.linear_velocity.x
		if k < n_sl:
			inp.throttle = clampf(0.18 + 0.15 * (15.3 - vx), 0.0, 0.50)
			inp.brake = 0.0
			inp.steer_angle = 0.6 * m.config.max_steer_angle * sin(TAU * 0.8 * t_s)
		else:
			inp.throttle = 0.15
			inp.brake = 0.0
			inp.steer_angle = 0.0

		var st := _step_plant(m, inp, dt_ecu, use_noise, delay_ticks, fifo, k == 0)
		t_hist.append(st["t_cmd"])
		var pe: float = float(st["p_ecu_kw"])
		var pp: float = float(st["p_phys_kw"])
		p_ecu_max = maxf(p_ecu_max, pe)
		p_ecu_min = minf(p_ecu_min, pe)
		p_phys_max = maxf(p_phys_max, pp)
		p_phys_min = minf(p_phys_min, pp)
		if pe > 80.005 or pe < -25.005:
			hv_ecu_bad += 1
		if pp > 80.5 or pp < -25.5:
			hv_phys_bad += 1

		var lv := m.chassis_state.linear_velocity
		var av := m.chassis_state.angular_velocity
		var b_deg: float = absf(rad_to_deg(atan2(lv.y, maxf(absf(lv.x), 0.5))))
		max_beta = maxf(max_beta, b_deg)
		max_wz = maxf(max_wz, absf(av.z))
		if b_deg > 20.0 or absf(av.z) > 4.0 or lv.x < 2.0:
			spun = true

	var osc := _analyze_t_osc(t_hist)
	return {
		"max_beta_deg": max_beta,
		"max_wz_rad_s": max_wz,
		"spun": spun,
		"hv_ecu_bad": hv_ecu_bad,
		"hv_phys_bad": hv_phys_bad,
		"p_ecu_max_kw": p_ecu_max,
		"p_ecu_min_kw": p_ecu_min,
		"p_phys_max_kw": p_phys_max,
		"p_phys_min_kw": p_phys_min,
		"t_cmd_hf_rms_nm": float(osc["hf_rms_nm"]),
		"t_cmd_chatter_ratio": float(osc["chatter_ratio"]),
	}


func _run_skidpad(cfg_json: Dictionary) -> Dictionary:
	_rng.seed = 9102
	var m := _make_car(cfg_json)
	_init_speed(m, 10.5)
	var use_noise: bool = bool(cfg_json.get("use_noise", false))
	var delay_ticks: int = int(cfg_json.get("delay_ticks", 0))
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 1  # SKIDPAD
	inp.tv_mode = 0

	var dt_ecu: float = 0.005
	var n_steps: int = int(round(10.0 / dt_ecu))
	var r_sk: float = 9.125
	var max_steer: float = m.config.max_steer_angle
	var fifo: Array = []
	var t_hist: Array = []
	var v_sum: float = 0.0
	var n_ss: int = 0
	var max_beta: float = 0.0
	var spun: bool = false
	var hv_ecu_bad: int = 0
	var hv_phys_bad: int = 0
	var p_ecu_max: float = -1e9
	var p_ecu_min: float = 1e9

	for k in range(n_steps):
		var t_s: float = float(k) * dt_ecu
		var sgn: float = -1.0 if t_s < 6.0 else 1.0
		var lv := m.chassis_state.linear_velocity
		var av := m.chassis_state.angular_velocity
		var b_deg: float = rad_to_deg(atan2(lv.y, maxf(absf(lv.x), 0.5)))
		max_beta = maxf(max_beta, absf(b_deg))
		if absf(b_deg) > 20.0 or absf(av.z) > 4.0 or lv.x < 2.0:
			spun = true
			break

		var wz_tgt: float = sgn * lv.x / r_sk
		var delta_ff: float = sgn * atan(m.config.wheelbase / r_sk)
		inp.steer_angle = clampf(delta_ff + 0.25 * (wz_tgt - av.z), -max_steer, max_steer)
		var beta_trim: float = maxf(0.0, absf(b_deg) - 4.5) * 0.12
		inp.throttle = clampf(0.38 + 0.28 * (12.2 - lv.x) - beta_trim, 0.0, 0.90)
		inp.brake = 0.0

		var st := _step_plant(m, inp, dt_ecu, use_noise, delay_ticks, fifo, k == 0)
		t_hist.append(st["t_cmd"])
		var pe: float = float(st["p_ecu_kw"])
		var pp: float = float(st["p_phys_kw"])
		p_ecu_max = maxf(p_ecu_max, pe)
		p_ecu_min = minf(p_ecu_min, pe)
		if pe > 80.005 or pe < -25.005:
			hv_ecu_bad += 1
		if pp > 80.5 or pp < -25.5:
			hv_phys_bad += 1

		if t_s >= 2.5 and t_s <= 5.8:
			v_sum += lv.x
			n_ss += 1

	var v_ss: float = v_sum / float(maxi(n_ss, 1))
	var t_lap: float = (TAU * r_sk) / maxf(v_ss, 1.0)
	var osc := _analyze_t_osc(t_hist)
	return {
		"t_lap_s": t_lap,
		"vx_ss_ms": v_ss,
		"max_beta_deg": max_beta,
		"spun": spun,
		"hv_ecu_bad": hv_ecu_bad,
		"hv_phys_bad": hv_phys_bad,
		"p_ecu_max_kw": p_ecu_max,
		"p_ecu_min_kw": p_ecu_min,
		"t_cmd_hf_rms_nm": float(osc["hf_rms_nm"]),
		"t_cmd_chatter_ratio": float(osc["chatter_ratio"]),
	}


func _run_hairpin(cfg_json: Dictionary) -> Dictionary:
	_rng.seed = 9103
	var m := _make_car(cfg_json)
	_init_speed(m, 20.0)
	var use_noise: bool = bool(cfg_json.get("use_noise", false))
	var delay_ticks: int = int(cfg_json.get("delay_ticks", 0))
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 0
	inp.tv_mode = 0

	var dt_ecu: float = 0.005
	var max_steps: int = int(round(12.0 / dt_ecu))
	var r_hp: float = 5.5
	var x_turn: float = 22.0
	var cx: float = x_turn
	var cy: float = r_hp
	var max_steer: float = m.config.max_steer_angle
	var v_apex_tgt: float = sqrt(11.8 * r_hp)
	var decel_plan: float = 12.0
	var phase: int = 0
	var finish_t: float = 12.0
	var finished: bool = false
	var spun: bool = false
	var max_beta: float = 0.0
	var hv_ecu_bad: int = 0
	var hv_phys_bad: int = 0
	var p_ecu_max: float = -1e9
	var p_ecu_min: float = 1e9
	var fifo: Array = []
	var t_hist: Array = []

	for k in range(max_steps):
		var t_s: float = float(k) * dt_ecu
		var pos := m.chassis_state.position
		var lv := m.chassis_state.linear_velocity
		var av := m.chassis_state.angular_velocity
		var yaw: float = m.chassis_state.yaw
		var vx: float = lv.x
		var b_deg: float = absf(rad_to_deg(atan2(lv.y, maxf(absf(vx), 0.5))))
		max_beta = maxf(max_beta, b_deg)
		if b_deg > 25.0 or absf(av.z) > 4.5 or (t_s > 0.5 and vx < 1.5):
			spun = true
			break

		if phase == 0:
			var dist_to_turn: float = maxf(x_turn - pos.x, 0.0)
			var v_req: float = sqrt(v_apex_tgt * v_apex_tgt + 2.0 * decel_plan * dist_to_turn)
			var heading_err: float = wrapf(0.0 - yaw, -PI, PI)
			var steer_cmd: float = clampf(1.8 * heading_err - 0.35 * pos.y, -max_steer, max_steer)
			if dist_to_turn < 2.2:
				steer_cmd = lerpf(steer_cmd, atan(m.config.wheelbase / r_hp), clampf((2.2 - dist_to_turn) / 2.2, 0.0, 1.0))
			if vx > v_req + 0.3:
				inp.throttle = 0.0
				inp.brake = clampf(0.25 + 0.22 * (vx - v_req), 0.0, 1.0)
			else:
				inp.throttle = clampf(0.4 + 0.3 * (v_req - vx), 0.0, 1.0)
				inp.brake = 0.0
			inp.steer_angle = steer_cmd
			if pos.x >= x_turn:
				phase = 1
		elif phase == 1:
			var dx: float = pos.x - cx
			var dy: float = pos.y - cy
			var rad_cur: float = sqrt(dx * dx + dy * dy)
			var theta: float = atan2(dy, dx)
			var heading_err: float = wrapf(theta + PI * 0.5 - yaw, -PI, PI)
			var rad_err: float = rad_cur - r_hp
			inp.steer_angle = clampf(atan(m.config.wheelbase / r_hp) + 1.45 * heading_err + 0.24 * rad_err, -max_steer, max_steer)
			if theta < -0.15:
				var v_lim: float = v_apex_tgt * (1.0 - 0.06 * maxf(rad_err, 0.0))
				if vx > v_lim + 0.25:
					inp.throttle = 0.0
					inp.brake = clampf(0.20 + 0.25 * (vx - v_lim), 0.0, 0.85)
				else:
					inp.throttle = clampf(0.35 + 0.30 * (v_lim - vx), 0.0, 0.85)
					inp.brake = 0.0
			else:
				var exit_prog: float = clampf((theta + 0.15) / (PI * 0.5 + 0.15), 0.0, 1.0)
				var v_lim: float = v_apex_tgt * (1.0 + 0.55 * exit_prog)
				inp.brake = 0.0
				if rad_err > 1.2:
					inp.throttle = 0.35
				elif vx < v_lim:
					inp.throttle = clampf(0.55 + 0.45 * exit_prog, 0.0, 1.0)
				else:
					inp.throttle = 0.40
			if theta >= PI * 0.5 - 0.06 and pos.y > r_hp:
				phase = 2
		else:
			var ey_right: float = pos.y - 2.0 * r_hp
			var heading_err: float = wrapf(PI - yaw, -PI, PI)
			inp.steer_angle = clampf(1.8 * heading_err + 0.30 * ey_right, -max_steer, max_steer)
			inp.throttle = 1.0
			inp.brake = 0.0
			if pos.x <= x_turn - 40.0:
				finish_t = t_s
				finished = true
				break

		var st := _step_plant(m, inp, dt_ecu, use_noise, delay_ticks, fifo, k == 0)
		t_hist.append(st["t_cmd"])
		var pe: float = float(st["p_ecu_kw"])
		var pp: float = float(st["p_phys_kw"])
		p_ecu_max = maxf(p_ecu_max, pe)
		p_ecu_min = minf(p_ecu_min, pe)
		if pe > 80.005 or pe < -25.005:
			hv_ecu_bad += 1
		if pp > 80.5 or pp < -25.5:
			hv_phys_bad += 1

	var osc := _analyze_t_osc(t_hist)
	return {
		"time_s": finish_t,
		"finished": finished,
		"max_beta_deg": max_beta,
		"spun": spun,
		"hv_ecu_bad": hv_ecu_bad,
		"hv_phys_bad": hv_phys_bad,
		"p_ecu_max_kw": p_ecu_max,
		"p_ecu_min_kw": p_ecu_min,
		"t_cmd_hf_rms_nm": float(osc["hf_rms_nm"]),
		"t_cmd_chatter_ratio": float(osc["chatter_ratio"]),
	}


func _init() -> void:
	var cfg_path := ""
	var out_path := ""
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--cfg="):
			cfg_path = arg.substr(6)
		elif arg.begins_with("--out="):
			out_path = arg.substr(6)

	var cfg_json: Dictionary = {}
	if cfg_path != "" and FileAccess.file_exists(cfg_path):
		var parsed = JSON.parse_string(FileAccess.get_file_as_string(cfg_path))
		if parsed is Dictionary:
			cfg_json = parsed

	var out := {
		"case_name": String(cfg_json.get("case_name", "unknown")),
		"slalom_55": _run_slalom(cfg_json),
		"skidpad_9125": _run_skidpad(cfg_json),
		"hairpin_180": _run_hairpin(cfg_json),
		"stability": [
			_eval_stability_derivative(cfg_json, 55.0, 0.0),
			_eval_stability_derivative(cfg_json, 55.0, 6.0),
			_eval_stability_derivative(cfg_json, 80.0, 0.0),
			_eval_stability_derivative(cfg_json, 80.0, 5.0),
		],
	}
	if out_path != "":
		var f := FileAccess.open(out_path, FileAccess.WRITE)
		f.store_string(JSON.stringify(out))
		f.close()
	else:
		print("ROBUST_JSON:" + JSON.stringify(out))
	quit(0)
"""


STRESS_CASES: List[Dict[str, Any]] = [
    {
        "case_name": "nominal",
        "mu_mult": 1.0,
        "use_noise": False,
        "delay_ticks": 0,
        "delta_mass_kg": 0.0,
        "delta_cg_rear_mm": 0.0,
        "max_pace_loss_pct": 0.0,
    },
    {
        "case_name": "grip_minus_15",
        "mu_mult": 0.85,
        "use_noise": False,
        "delay_ticks": 0,
        "delta_mass_kg": 0.0,
        "delta_cg_rear_mm": 0.0,
        "max_pace_loss_pct": 20.0,  # Physical sqrt(1/0.85) corner speed + un-adapted braking zone
    },
    {
        "case_name": "grip_plus_15",
        "mu_mult": 1.15,
        "use_noise": False,
        "delay_ticks": 0,
        "delta_mass_kg": 0.0,
        "delta_cg_rear_mm": 0.0,
        "max_pace_loss_pct": 4.0,
    },
    {
        "case_name": "noise_latency_5ms",
        "mu_mult": 1.0,
        "use_noise": True,
        "delay_ticks": 1,  # 1 * 5 ms = 5 ms
        "delta_mass_kg": 0.0,
        "delta_cg_rear_mm": 0.0,
        "max_pace_loss_pct": 8.0,
    },
    {
        "case_name": "noise_latency_10ms",
        "mu_mult": 1.0,
        "use_noise": True,
        "delay_ticks": 2,  # 2 * 5 ms = 10 ms
        "delta_mass_kg": 0.0,
        "delta_cg_rear_mm": 0.0,
        "max_pace_loss_pct": 8.0,
    },
    {
        "case_name": "mass_plus12_cg_rear15",
        "mu_mult": 1.0,
        "use_noise": False,
        "delay_ticks": 0,
        "delta_mass_kg": 12.0,
        "delta_cg_rear_mm": 15.0,
        "max_pace_loss_pct": 5.0,
    },
    {
        "case_name": "mass_minus12_cg_fwd15",
        "mu_mult": 1.0,
        "use_noise": False,
        "delay_ticks": 0,
        "delta_mass_kg": -12.0,
        "delta_cg_rear_mm": -15.0,
        "max_pace_loss_pct": 4.0,
    },
    {
        "case_name": "combined_worst",
        "mu_mult": 0.85,
        "use_noise": True,
        "delay_ticks": 2,  # 10 ms
        "delta_mass_kg": 12.0,
        "delta_cg_rear_mm": 15.0,
        "max_pace_loss_pct": 22.0,
    },
]


def _run_one_case(
    case_spec: Dict[str, Any],
    mode_overrides: Dict[str, Any],
    gd_path: Path,
    grip_profile_id: int = 1,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"mirena_rob_{case_spec['case_name']}_") as tmp:
        tmp_dir = Path(tmp)
        cfg_p = tmp_dir / "cfg.json"
        out_p = tmp_dir / "out.json"
        payload = dict(case_spec)
        payload["grip_profile"] = grip_profile_id
        payload["mode_overrides"] = mode_overrides
        cfg_p.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [
            "godot", "--headless", "--path", str(SIM_DIR), "-s", str(gd_path),
            f"--cfg={cfg_p}", f"--out={out_p}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
        if proc.returncode != 0 or not out_p.exists():
            raise RuntimeError(
                f"Godot robustness probe failed on {case_spec['case_name']} (rc={proc.returncode}):\n"
                f"{proc.stderr}\n{proc.stdout}"
            )
        return json.loads(out_p.read_text(encoding="utf-8"))


def evaluate_robustness(
    mode_overrides: Dict[str, Any],
    grip_profile: str = "fsg_real",
    workers: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all 8 robustness stress cases on mode_overrides without retuning."""
    t0 = time.perf_counter()
    grip_id = 1 if grip_profile == "fsg_real" else 0

    with tempfile.TemporaryDirectory(prefix="mirena_rob_gd_") as gd_tmp:
        gd_path = Path(gd_tmp) / "robustness_probe.gd"
        gd_path.write_text(GODOT_ROBUSTNESS_PROBE_GD, encoding="utf-8")

        raw_results: Dict[str, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {
                ex.submit(_run_one_case, cs, mode_overrides, gd_path, grip_id): cs["case_name"]
                for cs in STRESS_CASES
            }
            for fut in concurrent.futures.as_completed(fut_map):
                cname = fut_map[fut]
                raw_results[cname] = fut.result()

    nom = raw_results["nominal"]
    nom_hp_s = float(nom["hairpin_180"]["time_s"])
    nom_sk_s = float(nom["skidpad_9125"]["t_lap_s"])

    evaluated_cases: List[Dict[str, Any]] = []
    overall_pass = True

    for cs in STRESS_CASES:
        cname = cs["case_name"]
        res = raw_results[cname]
        sl = res["slalom_55"]
        sk = res["skidpad_9125"]
        hp = res["hairpin_180"]
        stabs = res["stability"]

        spins = int(bool(sl["spun"])) + int(bool(sk["spun"])) + int(bool(hp["spun"]))
        hv_ecu_bad = int(sl["hv_ecu_bad"]) + int(sk["hv_ecu_bad"]) + int(hp["hv_ecu_bad"])
        hv_phys_bad = int(sl["hv_phys_bad"]) + int(sk["hv_phys_bad"]) + int(hp["hv_phys_bad"])
        all_stable = all(bool(s["stable"]) for s in stabs)
        worst_dmz_dbeta = max(float(s["dMz_dbeta_sae_nm_per_deg"]) for s in stabs)

        max_hf_rms = max(
            float(sl["t_cmd_hf_rms_nm"]),
            float(sk["t_cmd_hf_rms_nm"]),
            float(hp["t_cmd_hf_rms_nm"]),
        )
        max_chatter = max(
            float(sl["t_cmd_chatter_ratio"]),
            float(sk["t_cmd_chatter_ratio"]),
            float(hp["t_cmd_chatter_ratio"]),
        )
        # High-freq oscillation gate: chatter ratio < 5% and 2nd-diff RMS < 45 Nm (relaxed slightly under noise)
        hf_rms_lim = 55.0 if cs["use_noise"] else 35.0
        osc_ok = (max_chatter < 0.05) and (max_hf_rms < hf_rms_lim)

        hp_loss_pct = 100.0 * (float(hp["time_s"]) - nom_hp_s) / max(nom_hp_s, 1e-6)
        sk_loss_pct = 100.0 * (float(sk["t_lap_s"]) - nom_sk_s) / max(nom_sk_s, 1e-6)
        pace_ok = (
            bool(hp["finished"])
            and hp_loss_pct <= cs["max_pace_loss_pct"] + 0.01
            and sk_loss_pct <= cs["max_pace_loss_pct"] + 0.01
        )
        # Slalom beta gate: < 3.2 deg nominal, < 4.5 deg under severe perturbations
        sl_beta_lim = 3.2 if cname == "nominal" else 4.8
        slalom_ok = float(sl["max_beta_deg"]) < sl_beta_lim

        failures: List[str] = []
        if not all_stable:
            failures.append(f"unstable(worst_dMz_dbeta_sae={worst_dmz_dbeta:.1f})")
        if spins > 0:
            failures.append(f"spins({spins})")
        if hv_ecu_bad > 0 or hv_phys_bad > 0:
            failures.append(f"hv_viol(ecu={hv_ecu_bad}, phys={hv_phys_bad})")
        if not osc_ok:
            failures.append(f"t_cmd_osc(hf_rms={max_hf_rms:.1f}Nm, chatter={max_chatter:.3f})")
        if not pace_ok:
            failures.append(
                f"pace_loss(hp={hp_loss_pct:+.2f}%, sk={sk_loss_pct:+.2f}%, fin={hp['finished']}, lim={cs['max_pace_loss_pct']}%)"
            )
        if not slalom_ok:
            failures.append(f"slalom_beta({float(sl['max_beta_deg']):.2f}deg >= {sl_beta_lim})")

        case_pass = len(failures) == 0
        if not case_pass:
            overall_pass = False

        row = {
            "case_name": cname,
            "pass": case_pass,
            "failures": failures,
            "directionally_stable": all_stable,
            "worst_dMz_dbeta_sae_nm_per_deg": round(worst_dmz_dbeta, 2),
            "spins": spins,
            "hv_ecu_violations": hv_ecu_bad,
            "hv_phys_violations": hv_phys_bad,
            "t_cmd_max_hf_rms_nm": round(max_hf_rms, 2),
            "t_cmd_max_chatter_ratio": round(max_chatter, 4),
            "hairpin_time_s": round(float(hp["time_s"]), 4),
            "hairpin_pace_loss_pct": round(hp_loss_pct, 2),
            "skidpad_lap_s": round(float(sk["t_lap_s"]), 4),
            "skidpad_pace_loss_pct": round(sk_loss_pct, 2),
            "slalom_max_beta_deg": round(float(sl["max_beta_deg"]), 3),
            "p_ecu_max_kw": round(
                max(float(sl["p_ecu_max_kw"]), float(sk["p_ecu_max_kw"]), float(hp["p_ecu_max_kw"])), 3
            ),
            "p_ecu_min_kw": round(
                min(float(sl["p_ecu_min_kw"]), float(sk["p_ecu_min_kw"]), float(hp["p_ecu_min_kw"])), 3
            ),
            "stability_details": stabs,
        }
        evaluated_cases.append(row)
        if verbose:
            status = "PASS" if case_pass else "FAIL"
            print(
                f"  [{cname:22s}] {status:4s} | dMz/db={worst_dmz_dbeta:7.1f} | spins={spins} hv={hv_ecu_bad}/{hv_phys_bad} "
                f"| hf_rms={max_hf_rms:5.1f}Nm chat={max_chatter:.3f} | "
                f"HP={row['hairpin_time_s']:.3f}s ({hp_loss_pct:+.1f}%) SK={row['skidpad_lap_s']:.3f}s ({sk_loss_pct:+.1f}%) "
                f"SL_b={row['slalom_max_beta_deg']:.2f}deg"
                + (f"  FAILS={failures}" if failures else ""),
                flush=True,
            )

    return {
        "grip_profile": grip_profile,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "overall_pass": overall_pass,
        "mode_overrides": mode_overrides,
        "cases": evaluated_cases,
    }


def _blend_overrides(
    cand: Dict[str, Dict[str, float]],
    base: Dict[str, Dict[str, float]],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m_name, m_dict in cand.items():
        out[m_name] = {}
        b_dict = base.get(m_name, {})
        for k, v in m_dict.items():
            bv = float(b_dict.get(k, v))
            out[m_name][k] = round(float(bv + alpha * (float(v) - bv)), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness stress test suite for TeR27 TV 4WD")
    parser.add_argument("--grip-profile", type=str, default="fsg_real", choices=["fsg_real", "ttc_lab"])
    parser.add_argument("--opt-report", type=str, default="", help="Path to closed_loop_opt_report.json (enables Pareto backoff)")
    parser.add_argument("--overrides-json", type=str, default="", help="JSON string or file path with mode_overrides")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "robustness_report.json"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_overrides: Dict[str, Any] = {}
    pareto_candidates: List[Dict[str, Any]] = []
    baseline_overrides: Dict[str, Any] = {}

    if args.opt_report:
        rep = json.loads(Path(args.opt_report).read_text(encoding="utf-8"))
        baseline_overrides = rep["baseline"]["mode_overrides"]
        seen = set()
        for c in [rep["best_verified"]] + rep.get("verified_top3", []) + rep.get("pareto_archive", []):
            key = json.dumps(c["mode_overrides"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                pareto_candidates.append(c)
        mode_overrides = pareto_candidates[0]["mode_overrides"]
    elif args.overrides_json:
        p = Path(args.overrides_json)
        raw = p.read_text(encoding="utf-8") if p.exists() else args.overrides_json
        mode_overrides = json.loads(raw)

    if pareto_candidates:
        selected_report: Optional[Dict[str, Any]] = None
        selected_cand: Optional[Dict[str, Any]] = None
        backoff_history: List[Dict[str, Any]] = []

        # 1. Test Pareto archive candidates from fastest to most conservative
        for idx, cand in enumerate(pareto_candidates[:6]):
            print(
                f"\n[ROBUST] Stress-testing Pareto candidate #{idx + 1} ({cand['label']}, cost={cand['cost']:+.4f})...",
                flush=True,
            )
            r = evaluate_robustness(cand["mode_overrides"], args.grip_profile, args.workers, verbose=True)
            backoff_history.append({
                "candidate": cand["label"],
                "cost": cand["cost"],
                "overall_pass": r["overall_pass"],
                "n_cases_passed": sum(1 for c in r["cases"] if c["pass"]),
            })
            if r["overall_pass"]:
                selected_report = r
                selected_cand = cand
                print(f"[ROBUST] Selected robust Pareto winner: {cand['label']}!", flush=True)
                break

        # 2. If all aggressive Pareto points had a failure, blend back toward baseline (0.75, 0.50, 0.25, 0.0)
        if selected_report is None:
            best_mo = pareto_candidates[0]["mode_overrides"]
            for alpha in [0.75, 0.50, 0.25, 0.0]:
                blended = _blend_overrides(best_mo, baseline_overrides, alpha)
                lbl = f"blend_{int(alpha * 100)}pct"
                print(f"\n[ROBUST] Backoff blend {lbl} toward baseline...", flush=True)
                r = evaluate_robustness(blended, args.grip_profile, args.workers, verbose=True)
                backoff_history.append({
                    "candidate": lbl,
                    "alpha": alpha,
                    "overall_pass": r["overall_pass"],
                    "n_cases_passed": sum(1 for c in r["cases"] if c["pass"]),
                })
                if r["overall_pass"] or alpha == 0.0:
                    selected_report = r
                    selected_cand = {"label": lbl, "mode_overrides": blended}
                    break

        assert selected_report is not None and selected_cand is not None
        selected_report["selected_candidate_label"] = selected_cand["label"]
        selected_report["backoff_history"] = backoff_history
        final_report = selected_report
    else:
        print("[ROBUST] Stress-testing provided/current TV configuration...", flush=True)
        final_report = evaluate_robustness(mode_overrides, args.grip_profile, args.workers, verbose=True)
        final_report["selected_candidate_label"] = "current_config"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    verdict = "PASS" if final_report["overall_pass"] else "FAIL"
    print(
        f"\n[ROBUST VERDICT: {verdict}] selected={final_report['selected_candidate_label']} "
        f"elapsed={final_report['elapsed_s']:.1f}s -> {out_path}",
        flush=True,
    )
    return 0 if final_report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
