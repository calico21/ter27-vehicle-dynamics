#!/usr/bin/env python3
"""
Closed-Loop Multi-Scenario 14-DOF TV Optimizer for Tecnun e-Racing TeR27 4WD (MirenaSim).

Optimizes the compact winning TV parameter set on the multi-scenario 14-DOF closed-loop cost
under FSG_REAL_TRACK grip profile:
  - 180 deg hairpin time (R = 5.5 m, 20 m/s entry, 40 m exit straight)
  - 9.125 m Skidpad speed / lap time
  - FSG 2023 14-DOF closed-loop lap time (with 4-corner swept-path cone check)

Hard constraints (infinite penalty if violated):
  - 0 HV violations (> 80.005 kW or < -25.005 kW)
  - max|beta| < 3.2 deg in 55 km/h slalom (+ 0 spins, straight recovery < 0.30 s)
  - 0-50 km/h <= 1.13 s in repeated launches (standstill + brake-to-stop re-launch)
  - 0 cones hit on FSG 2023

Algorithm:
  - Deterministic Latin Hypercube exploration + evolutionary (mu, lambda) CMA-like adaptive search
    + local Nelder-Mead / Powell refinement (using scipy + numpy only).
  - Automatically detects any new mode parameters added by ablation (comparing against
    vd_correlation_lab/results/tv_oracle/tv_4wd_config_baseline.json) or accepts --extra-params.
  - Saves full Pareto archive and best/robust candidates to:
    vd_correlation_lab/results/tv_oracle/closed_loop_opt_report.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

ORACLE_DIR = Path(__file__).resolve().parent
LAB_DIR = ORACLE_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
TV4_CFG_PATH = (
    REPO_ROOT
    / "src"
    / "mirena_sim"
    / "Scenes"
    / "Vehicle"
    / "Config"
    / "vehicles"
    / "ter27_4wd"
    / "tv_4wd_config.json"
)
OUT_DIR = LAB_DIR / "results" / "tv_oracle"
BASELINE_CFG_PATH = OUT_DIR / "tv_4wd_config_baseline.json"
FSG_2023_TRACK_PATH = REPO_ROOT / "data" / "tracks" / "fsg_2023.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from vd_correlation_lab.optimization.lap_time_sim import (  # noqa: E402
    Track,
    extract_or_load_ggv,
    optimize_mintime_racing_line,
    run_transient_lap,
    search_fastest_valid_lap,
    solve_qss,
    verify_swept_path_4corners,
)
from vd_correlation_lab.optimization.lap_time_sim.tracks.build_fsg_2023 import main as build_fsg_main  # noqa: E402

SEED = 42

# Hard constraint thresholds
HV_MAX_KW = 80.005
HV_MIN_KW = -25.005
SLALOM_MAX_BETA_DEG = 3.20
SLALOM_MAX_REC_S = 0.30
LAUNCH_MAX_T50_S = 1.13

# Headless Godot multi-maneuver probe:
# Runs 180 deg hairpin (R=5.5 m, 20 m/s entry, 40 m exit), Skidpad (R=9.125 m),
# 55 km/h Slalom + recovery, and Repeated Launch (2 consecutive launches) in one invocation.
GODOT_MANEUVER_PROBE_GD = r"""extends SceneTree

const Full14Dof = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputs = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")


func _make_car(cfg_json: Dictionary, grip_profile: int) -> Full14Dof:
	var m := Full14Dof.new()
	m.load_vehicle("ter27_4wd", "default")
	m.set_track_grip_profile(grip_profile)
	m.config.update_derived_geometry()
	m.suspension.sync_from_config(m.config)
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


func _check_hv(m: Full14Dof) -> Dictionary:
	var p_ecu: float = (m.ecu_tv_4wd.last_power_w * 0.001) if m.ecu_tv_4wd != null else 0.0
	var p_hv: float = m.hv_state.power_kw if m.hv_state != null else p_ecu
	var p_max: float = maxf(p_ecu, p_hv)
	var p_min: float = minf(p_ecu, p_hv)
	var bad: int = 1 if (p_max > 80.005 or p_min < -25.005) else 0
	return {"p_max": p_max, "p_min": p_min, "bad": bad}


func _run_hairpin_180(cfg_json: Dictionary, grip_profile: int) -> Dictionary:
	# 180 deg hairpin: entry at 20 m/s, 22 m straight braking zone -> R=5.5 m semicircle -> 40 m exit straight
	var m := _make_car(cfg_json, grip_profile)
	_init_speed(m, 20.0)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 0  # AUTOCROSS
	inp.tv_mode = 0

	var dt: float = 0.0025
	var max_steps: int = int(round(12.0 / dt))
	var r_hp: float = 5.5
	var x_turn: float = 22.0
	var cx: float = x_turn
	var cy: float = r_hp
	var max_steer: float = m.config.max_steer_angle
	var phase: int = 0  # 0=entry straight, 1=180 deg arc, 2=exit straight
	var finish_t: float = 12.0
	var finished: bool = false
	var spun: bool = false
	var hv_bad: int = 0
	var p_max: float = -1e9
	var p_min: float = 1e9
	var max_beta_deg: float = 0.0
	var min_vx_apex: float = 99.0
	var max_lat_err: float = 0.0

	# Estimate physical corner speed for R=5.5 m under current grip
	var ay_cap: float = 11.8 if grip_profile == 1 else 14.5
	var v_apex_tgt: float = sqrt(ay_cap * r_hp)
	var decel_plan: float = 12.0 if grip_profile == 1 else 15.0

	for k in range(max_steps):
		var t: float = float(k) * dt
		var pos: Vector3 = m.chassis_state.position
		var lv: Vector3 = m.chassis_state.linear_velocity
		var av: Vector3 = m.chassis_state.angular_velocity
		var yaw: float = m.chassis_state.yaw
		var vx: float = lv.x
		var beta_deg: float = rad_to_deg(atan2(lv.y, maxf(absf(vx), 0.5)))
		max_beta_deg = maxf(max_beta_deg, absf(beta_deg))
		if absf(beta_deg) > 25.0 or absf(av.z) > 4.5 or (t > 0.5 and vx < 1.5):
			spun = true
			break

		if phase == 0:
			var dist_to_turn: float = maxf(x_turn - pos.x, 0.0)
			var v_req: float = sqrt(v_apex_tgt * v_apex_tgt + 2.0 * decel_plan * dist_to_turn)
			var ey: float = pos.y
			max_lat_err = maxf(max_lat_err, absf(ey))
			var heading_err: float = wrapf(0.0 - yaw, -PI, PI)
			var steer_cmd: float = clampf(1.8 * heading_err - 0.35 * ey, -max_steer, max_steer)
			if dist_to_turn < 2.0:
				var blend: float = clampf((2.0 - dist_to_turn) / 2.0, 0.0, 1.0)
				steer_cmd = lerpf(steer_cmd, atan(m.config.wheelbase / r_hp), blend)
			if vx > v_req + 0.25:
				inp.throttle = 0.0
				inp.brake = clampf(0.25 + 0.28 * (vx - v_req), 0.0, 1.0)
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
			var theta: float = atan2(dy, dx)  # -PI/2 -> 0 -> +PI/2
			var psi_tan: float = theta + PI * 0.5
			var heading_err: float = wrapf(psi_tan - yaw, -PI, PI)
			var rad_err: float = rad_cur - r_hp
			max_lat_err = maxf(max_lat_err, absf(rad_err))
			min_vx_apex = minf(min_vx_apex, vx)
			var delta_ff: float = atan(m.config.wheelbase / r_hp)
			var steer_cmd: float = clampf(delta_ff + 1.65 * heading_err + 0.30 * rad_err, -max_steer, max_steer)
			inp.steer_angle = steer_cmd
			if theta < -0.10:
				var v_lim: float = v_apex_tgt * (1.0 - 0.08 * maxf(rad_err, 0.0))
				if vx > v_lim + 0.20:
					inp.throttle = 0.0
					inp.brake = clampf(0.20 + 0.30 * (vx - v_lim), 0.0, 0.85)
				else:
					inp.throttle = clampf(0.35 + 0.30 * (v_lim - vx), 0.0, 0.85)
					inp.brake = 0.0
			else:
				var exit_prog: float = clampf((theta + 0.10) / (PI * 0.5 + 0.10), 0.0, 1.0)
				var v_lim: float = v_apex_tgt * (1.0 + 0.65 * exit_prog)
				inp.brake = 0.0
				if rad_err > 1.2:
					inp.throttle = 0.30
				elif vx < v_lim:
					inp.throttle = clampf(0.55 + 0.45 * exit_prog, 0.0, 1.0)
				else:
					inp.throttle = 0.40
			if theta >= PI * 0.5 - 0.05 and pos.y > r_hp:
				phase = 2
		else:
			# Exit straight along y = 2*R going in -X direction (yaw = PI)
			# Car left is -Y, so pos.y > 2*R means car is to the RIGHT of path -> steer LEFT (+)
			var ey_right: float = pos.y - 2.0 * r_hp
			max_lat_err = maxf(max_lat_err, absf(ey_right))
			var heading_err: float = wrapf(PI - yaw, -PI, PI)
			inp.steer_angle = clampf(1.8 * heading_err + 0.30 * ey_right, -max_steer, max_steer)
			inp.throttle = 1.0
			inp.brake = 0.0
			if pos.x <= x_turn - 40.0:
				finish_t = t
				finished = true
				break

		m.step_standalone(inp, dt, 0.0)
		var hv := _check_hv(m)
		p_max = maxf(p_max, float(hv["p_max"]))
		p_min = minf(p_min, float(hv["p_min"]))
		hv_bad += int(hv["bad"])

	return {
		"time_s": finish_t,
		"finished": finished,
		"spun": spun,
		"max_beta_deg": max_beta_deg,
		"min_vx_apex_ms": min_vx_apex,
		"max_lat_err_m": max_lat_err,
		"hv_bad": hv_bad,
		"p_max_kw": p_max,
		"p_min_kw": p_min,
	}


func _run_skidpad(cfg_json: Dictionary, grip_profile: int) -> Dictionary:
	# R = 9.125 m steady-state + figure-8 transition skidpad
	var m := _make_car(cfg_json, grip_profile)
	var v0: float = 10.5 if grip_profile == 1 else 11.5
	_init_speed(m, v0)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 1  # SKIDPAD
	inp.tv_mode = 0

	var dt: float = 0.0025
	var dur: float = 11.0
	var n_steps: int = int(round(dur / dt))
	var r_sk: float = 9.125
	var max_steer: float = m.config.max_steer_angle
	var v_sum: float = 0.0
	var ay_sum: float = 0.0
	var n_ss: int = 0
	var max_beta_deg: float = 0.0
	var trans_wz_max: float = 0.0
	var trans_wz_end: float = 0.0
	var spun: bool = false
	var hv_bad: int = 0
	var p_max: float = -1e9
	var p_min: float = 1e9

	# Target speed slightly above grip limit so the car finds its natural maximum cornering speed
	var v_tgt: float = 12.2 if grip_profile == 1 else 13.4

	for k in range(n_steps):
		var t: float = float(k) * dt
		var sgn: float = -1.0 if t < 6.5 else 1.0
		var lv: Vector3 = m.chassis_state.linear_velocity
		var av: Vector3 = m.chassis_state.angular_velocity
		var vx: float = lv.x
		var wz: float = av.z
		var beta_deg: float = rad_to_deg(atan2(lv.y, maxf(absf(vx), 0.5)))
		max_beta_deg = maxf(max_beta_deg, absf(beta_deg))
		if absf(beta_deg) > 20.0 or absf(wz) > 4.0 or vx < 3.0:
			spun = true
			break

		var wz_tgt: float = sgn * vx / r_sk
		var delta_ff: float = sgn * atan(m.config.wheelbase / r_sk)
		var steer_cmd: float = clampf(delta_ff + 0.25 * (wz_tgt - wz), -max_steer, max_steer)
		var err_v: float = v_tgt - vx
		var beta_trim: float = maxf(0.0, absf(beta_deg) - 4.5) * 0.12
		inp.throttle = clampf(0.38 + 0.28 * err_v - beta_trim, 0.0, 0.90)
		inp.brake = 0.0
		inp.steer_angle = steer_cmd

		var res := m.step_standalone(inp, dt, 0.0)
		var hv := _check_hv(m)
		p_max = maxf(p_max, float(hv["p_max"]))
		p_min = minf(p_min, float(hv["p_min"]))
		hv_bad += int(hv["bad"])

		if t >= 2.5 and t <= 6.2:
			v_sum += vx
			ay_sum += absf(float(res["net_force_ros"].y) / m.config.mass)
			n_ss += 1
		if t >= 6.5 and t <= 8.5:
			trans_wz_max = maxf(trans_wz_max, absf(wz))
			trans_wz_end = absf(wz)

	var v_ss: float = v_sum / float(maxi(n_ss, 1))
	var ay_ss_g: float = (ay_sum / float(maxi(n_ss, 1))) / 9.81
	var t_lap: float = (TAU * r_sk) / maxf(v_ss, 1.0)
	var ov: float = maxf(0.0, trans_wz_max - trans_wz_end)
	return {
		"t_lap_s": t_lap,
		"vx_ss_ms": v_ss,
		"ay_ss_g": ay_ss_g,
		"max_beta_deg": max_beta_deg,
		"trans_wz_overshoot": ov,
		"spun": spun,
		"hv_bad": hv_bad,
		"p_max_kw": p_max,
		"p_min_kw": p_min,
	}


func _run_slalom_55(cfg_json: Dictionary, grip_profile: int) -> Dictionary:
	# Exact 55 km/h (15.3 m/s) slalom matching Test 5 in test_tv_power_curve_and_drivability.gd
	var m := _make_car(cfg_json, grip_profile)
	_init_speed(m, 15.3)
	var inp := VehicleInputs.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 2
	inp.tv_4wd_mode = 0  # AUTOCROSS
	inp.tv_mode = 0

	var dt: float = 0.0025
	var n_slalom: int = int(round(4.0 / dt))
	var n_rel: int = int(round(1.0 / dt))
	var max_beta_deg: float = 0.0
	var max_wz: float = 0.0
	var spun: bool = false
	var hv_bad: int = 0
	var p_max: float = -1e9
	var p_min: float = 1e9

	for k in range(n_slalom):
		var t_s: float = float(k) * dt
		var vx: float = m.chassis_state.linear_velocity.x
		inp.throttle = clampf(0.18 + 0.15 * (15.3 - vx), 0.0, 0.50)
		inp.brake = 0.0
		inp.steer_angle = 0.6 * m.config.max_steer_angle * sin(TAU * 0.8 * t_s)
		m.step_standalone(inp, dt, 0.0)
		var lv: Vector3 = m.chassis_state.linear_velocity
		var av: Vector3 = m.chassis_state.angular_velocity
		var beta_deg: float = absf(rad_to_deg(atan2(lv.y, maxf(absf(lv.x), 0.5))))
		max_beta_deg = maxf(max_beta_deg, beta_deg)
		max_wz = maxf(max_wz, absf(av.z))
		if beta_deg > 25.0 or absf(av.z) > 4.0 or lv.x < 3.0:
			spun = true
		var hv := _check_hv(m)
		p_max = maxf(p_max, float(hv["p_max"]))
		p_min = minf(p_min, float(hv["p_min"]))
		hv_bad += int(hv["bad"])

	var rec_time_s: float = -1.0
	var settled_streak: int = 0
	for k in range(n_rel):
		inp.throttle = 0.15
		inp.brake = 0.0
		inp.steer_angle = 0.0
		m.step_standalone(inp, dt, 0.0)
		var lv: Vector3 = m.chassis_state.linear_velocity
		var av: Vector3 = m.chassis_state.angular_velocity
		var beta_deg: float = absf(rad_to_deg(atan2(lv.y, maxf(absf(lv.x), 0.5))))
		max_beta_deg = maxf(max_beta_deg, beta_deg)
		max_wz = maxf(max_wz, absf(av.z))
		if beta_deg > 25.0 or absf(av.z) > 4.0 or lv.x < 3.0:
			spun = true
		var hv := _check_hv(m)
		p_max = maxf(p_max, float(hv["p_max"]))
		p_min = minf(p_min, float(hv["p_min"]))
		hv_bad += int(hv["bad"])
		if absf(av.z) < 0.15 and beta_deg < 1.5:
			settled_streak += 1
			if rec_time_s < 0.0 and settled_streak >= 8:
				rec_time_s = float(k + 1 - 7) * dt
		else:
			settled_streak = 0

	return {
		"max_beta_deg": max_beta_deg,
		"max_wz_rad_s": max_wz,
		"rec_time_s": rec_time_s,
		"spun": spun,
		"hv_bad": hv_bad,
		"p_max_kw": p_max,
		"p_min_kw": p_min,
	}


func _run_repeated_launches(cfg_json: Dictionary, grip_profile: int) -> Dictionary:
	# 2 consecutive launches: standstill launch -> brake to full stop -> re-launch (AUTOCROSS & ACCELERATION)
	var worst_t50: float = 0.0
	var t50_list: Array = []
	var hv_bad: int = 0
	var p_max: float = -1e9
	var p_min: float = 1e9
	var dt: float = 0.0025

	for tv_mode in [0, 3]:  # AUTOCROSS, ACCELERATION
		var m := _make_car(cfg_json, grip_profile)
		_init_speed(m, 0.0)
		var inp := VehicleInputs.new()
		inp.drive_mode = m.config.drive_mode
		inp.ecu_driving_mode = 2
		inp.tv_4wd_mode = tv_mode
		inp.tv_mode = 0

		for cyc in range(2):
			if cyc > 0:
				# Brake to standstill
				inp.throttle = 0.0
				inp.brake = 1.0
				inp.steer_angle = 0.0
				for _b in range(int(round(3.5 / dt))):
					m.step_standalone(inp, dt, 0.0)
					var hv_b := _check_hv(m)
					p_max = maxf(p_max, float(hv_b["p_max"]))
					p_min = minf(p_min, float(hv_b["p_min"]))
					hv_bad += int(hv_b["bad"])
					if m.chassis_state.linear_velocity.x <= 0.05:
						break
				for _s in range(12):
					m.step_standalone(inp, dt, 0.0)

			# Full-throttle launch
			inp.throttle = 1.0
			inp.brake = 0.0
			inp.steer_angle = 0.0
			var t50: float = 9.99
			for k in range(int(round(2.0 / dt))):
				m.step_standalone(inp, dt, 0.0)
				var hv := _check_hv(m)
				p_max = maxf(p_max, float(hv["p_max"]))
				p_min = minf(p_min, float(hv["p_min"]))
				hv_bad += int(hv["bad"])
				if t50 > 9.0 and m.chassis_state.linear_velocity.x * 3.6 >= 50.0:
					t50 = float(k + 1) * dt
					break
			t50_list.append(t50)
			worst_t50 = maxf(worst_t50, t50)

	return {
		"worst_t50_s": worst_t50,
		"t50_list": t50_list,
		"hv_bad": hv_bad,
		"p_max_kw": p_max,
		"p_min_kw": p_min,
	}


func _init() -> void:
	var cfg_path := ""
	var out_path := ""
	var grip_profile: int = 1  # FSG_REAL_TRACK
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--cfg="):
			cfg_path = arg.substr(6)
		elif arg.begins_with("--out="):
			out_path = arg.substr(6)
		elif arg.begins_with("--grip="):
			grip_profile = int(arg.substr(7))

	var cfg_json: Dictionary = {}
	if cfg_path != "" and FileAccess.file_exists(cfg_path):
		var parsed = JSON.parse_string(FileAccess.get_file_as_string(cfg_path))
		if parsed is Dictionary:
			cfg_json = parsed

	var out := {
		"grip_profile": grip_profile,
		"hairpin_180": _run_hairpin_180(cfg_json, grip_profile),
		"skidpad_9125": _run_skidpad(cfg_json, grip_profile),
		"slalom_55": _run_slalom_55(cfg_json, 0),  # Standard Test 5 slalom on nominal TTC_LAB_PEAK
		"launches": _run_repeated_launches(cfg_json, 0),  # Standard Test 1C repeated launches on nominal TTC_LAB_PEAK
	}
	if out_path != "":
		var f := FileAccess.open(out_path, FileAccess.WRITE)
		f.store_string(JSON.stringify(out))
		f.close()
	else:
		print("PROBE_JSON:" + JSON.stringify(out))
	quit(0)
"""


def _default_param_spec(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the compact optimization parameter vector from tv_4wd_config.json.
    Automatically includes core autocross/skidpad parameters plus any new scalar parameters
    introduced by @ablation (compared against tv_4wd_config_baseline.json).
    """
    auto_cfg = base_cfg["modes"]["autocross"]
    skid_cfg = base_cfg["modes"]["skidpad"]

    spec: List[Dict[str, Any]] = [
        {"mode": "autocross", "key": "centripetal_fx_gain", "lo": 0.0, "hi": 45.0},
        {"mode": "autocross", "key": "delta_alpha_eq_gain", "lo": 800.0, "hi": 3600.0},
        {"mode": "autocross", "key": "k_us_low_deg_g", "lo": 0.04, "hi": 0.30},
        {"mode": "autocross", "key": "k_us_deg_g", "lo": 0.45, "hi": 1.05},
        {"mode": "autocross", "key": "k_ff", "lo": 0.30, "hi": 0.80},
        {"mode": "autocross", "key": "k_ff_low_mult", "lo": 1.10, "hi": 1.80},
        {"mode": "autocross", "key": "hairpin_pivot_gain", "lo": 0.90, "hi": 1.55},
        {"mode": "autocross", "key": "dynamic_rear_bias_gain", "lo": 0.25, "hi": 0.70},
        {"mode": "autocross", "key": "slalom_yaw_damp_gain", "lo": 260.0, "hi": 520.0},
        {"mode": "autocross", "key": "hairpin_regen_max_nm", "lo": 70.0, "hi": 140.0},
        {"mode": "skidpad", "key": "k_us_deg_g", "lo": 0.08, "hi": 0.38},
        {"mode": "skidpad", "key": "alpha_eq_gain", "lo": 200.0, "hi": 650.0},
    ]

    # Auto-detect any new numeric parameters added by @ablation in autocross
    if BASELINE_CFG_PATH.exists():
        orig_cfg = json.loads(BASELINE_CFG_PATH.read_text(encoding="utf-8"))
        orig_auto = orig_cfg.get("modes", {}).get("autocross", {})
        existing_keys = {(s["mode"], s["key"]) for s in spec}
        for k, v in auto_cfg.items():
            if k not in orig_auto and isinstance(v, (int, float)) and not isinstance(v, bool):
                if ("autocross", k) not in existing_keys:
                    fv = float(v)
                    if abs(fv) > 1e-6:
                        lo = max(0.0, fv * 0.4) if fv > 0 else fv * 1.6
                        hi = fv * 1.8 if fv > 0 else min(0.0, fv * 0.4)
                    else:
                        lo, hi = 0.0, 1.0
                    spec.append({"mode": "autocross", "key": k, "lo": lo, "hi": hi})

    # Populate nominal values and ensure bounds cover nominal
    valid_spec: List[Dict[str, Any]] = []
    for item in spec:
        m_dict = base_cfg["modes"].get(item["mode"], {})
        if item["key"] not in m_dict:
            continue
        nom = float(m_dict[item["key"]])
        lo = min(float(item["lo"]), nom)
        hi = max(float(item["hi"]), nom)
        valid_spec.append({
            "mode": item["mode"],
            "key": item["key"],
            "nom": nom,
            "lo": lo,
            "hi": hi,
        })
    return valid_spec


def _vec_to_overrides(vec: np.ndarray, spec: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    mo: Dict[str, Dict[str, float]] = {}
    for val, item in zip(vec, spec):
        m_name = item["mode"]
        if m_name not in mo:
            mo[m_name] = {}
        mo[m_name][item["key"]] = round(float(np.clip(val, item["lo"], item["hi"])), 4)
    # Enforce k_us_deg_g >= k_us_low_deg_g if both present
    if "autocross" in mo and "k_us_deg_g" in mo["autocross"] and "k_us_low_deg_g" in mo["autocross"]:
        if mo["autocross"]["k_us_deg_g"] < mo["autocross"]["k_us_low_deg_g"]:
            mo["autocross"]["k_us_deg_g"] = mo["autocross"]["k_us_low_deg_g"]
    return mo


def _run_maneuver_probe(
    mode_overrides: Dict[str, Any],
    gd_path: Path,
    grip_profile_id: int = 1,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mirena_opt_probe_") as tmp:
        tmp_dir = Path(tmp)
        cfg_p = tmp_dir / "cfg.json"
        out_p = tmp_dir / "out.json"
        cfg_p.write_text(json.dumps({"mode_overrides": mode_overrides}), encoding="utf-8")
        cmd = [
            "godot", "--headless", "--path", str(SIM_DIR), "-s", str(gd_path),
            f"--cfg={cfg_p}", f"--out={out_p}", f"--grip={grip_profile_id}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
        if proc.returncode != 0 or not out_p.exists():
            raise RuntimeError(
                f"Godot maneuver probe failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
            )
        return json.loads(out_p.read_text(encoding="utf-8"))


class ClosedLoopEvaluator:
    """Thread-safe multi-scenario 14-DOF evaluator with hard constraints."""

    def __init__(
        self,
        spec: List[Dict[str, Any]],
        trk_fsg: Track,
        rl_fsg: Any,
        qss_fsg: Any,
        gd_path: Path,
        grip_profile: str = "fsg_real",
        weights: Tuple[float, float, float] = (1.5, 2.0, 1.0),
    ) -> None:
        self.spec = spec
        self.trk_fsg = trk_fsg
        self.rl_fsg = rl_fsg
        self.qss_fsg = qss_fsg
        self.gd_path = gd_path
        self.grip_profile = grip_profile
        self.grip_id = 1 if grip_profile == "fsg_real" else 0
        self.w_hp, self.w_sk, self.w_fsg = weights
        self.ref_hp_s: Optional[float] = None
        self.ref_sk_s: Optional[float] = None
        self.ref_fsg_s: Optional[float] = None
        self.base_alpha_drv: float = 0.96

    def evaluate(
        self,
        vec: np.ndarray,
        label: str,
        full_alpha_search: bool = False,
        check_cones: bool = True,
    ) -> Dict[str, Any]:
        mo = _vec_to_overrides(vec, self.spec)
        t0 = time.perf_counter()

        # 1. Fast Godot multi-maneuver probe: hairpin 180 + skidpad + slalom 55 + repeated launches
        probe = _run_maneuver_probe(mo, self.gd_path, grip_profile_id=self.grip_id)
        hp = probe["hairpin_180"]
        sk = probe["skidpad_9125"]
        sl = probe["slalom_55"]
        ln = probe["launches"]

        hv_bad_probe = int(hp["hv_bad"]) + int(sk["hv_bad"]) + int(sl["hv_bad"]) + int(ln["hv_bad"])
        p_max_probe = max(float(hp["p_max_kw"]), float(sk["p_max_kw"]), float(sl["p_max_kw"]), float(ln["p_max_kw"]))
        p_min_probe = min(float(hp["p_min_kw"]), float(sk["p_min_kw"]), float(sl["p_min_kw"]), float(ln["p_min_kw"]))

        violations: List[str] = []
        if hv_bad_probe > 0 or p_max_probe > HV_MAX_KW or p_min_probe < HV_MIN_KW:
            violations.append(f"hv_probe(bad={hv_bad_probe}, P=[{p_min_probe:.2f},{p_max_probe:.2f}])")
        if bool(sl["spun"]) or float(sl["max_beta_deg"]) >= SLALOM_MAX_BETA_DEG:
            violations.append(f"slalom_beta({float(sl['max_beta_deg']):.3f}deg >= {SLALOM_MAX_BETA_DEG})")
        if float(sl["rec_time_s"]) < 0.0 or float(sl["rec_time_s"]) >= SLALOM_MAX_REC_S:
            violations.append(f"slalom_rec({float(sl['rec_time_s']):.3f}s >= {SLALOM_MAX_REC_S})")
        if float(ln["worst_t50_s"]) > LAUNCH_MAX_T50_S:
            violations.append(f"launch_t50({float(ln['worst_t50_s']):.3f}s > {LAUNCH_MAX_T50_S})")
        if bool(hp["spun"]) or not bool(hp["finished"]):
            violations.append(f"hairpin_fail(fin={hp['finished']}, spun={hp['spun']})")
        if bool(sk["spun"]):
            violations.append("skidpad_spun")

        # Early reject before expensive FSG 2023 lap if any fast hard constraint failed (unless baseline)
        if violations and label != "baseline":
            return {
                "label": label,
                "mode_overrides": mo,
                "feasible": False,
                "cost": float("inf"),
                "violations": violations,
                "metrics": {
                    "hairpin_time_s": round(float(hp["time_s"]), 4),
                    "skidpad_lap_s": round(float(sk["t_lap_s"]), 4),
                    "skidpad_vx_ss_ms": round(float(sk["vx_ss_ms"]), 3),
                    "skidpad_ay_ss_g": round(float(sk["ay_ss_g"]), 3),
                    "slalom_max_beta_deg": round(float(sl["max_beta_deg"]), 3),
                    "slalom_rec_s": round(float(sl["rec_time_s"]), 3),
                    "launch_worst_t50_s": round(float(ln["worst_t50_s"]), 3),
                    "fsg_lap_s": 999.0,
                    "fsg_valid": False,
                    "cones_hit": -1,
                    "p_max_kw": round(p_max_probe, 3),
                    "p_min_kw": round(p_min_probe, 3),
                },
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }

        # 2. FSG 2023 14-DOF closed-loop lap (FSG_REAL_TRACK)
        with tempfile.TemporaryDirectory(prefix="mirena_fsg_opt_") as tmp_fsg:
            csv_p = Path(tmp_fsg) / "fsg_trace.csv" if check_cones else None
            if full_alpha_search:
                fsg_res, _ = search_fastest_valid_lap(
                    label=f"opt_{label}",
                    racing_line=self.rl_fsg,
                    qss=self.qss_fsg,
                    ecu_driving_mode=2,
                    tv_4wd_mode=0,
                    n_laps=1,
                    use_3d_kinematics=True,
                    mode_overrides=mo,
                    save_csv_path=csv_p,
                    grip_profile=self.grip_profile,
                )
            else:
                fsg_res = run_transient_lap(
                    label=f"opt_{label}",
                    racing_line=self.rl_fsg,
                    qss=self.qss_fsg,
                    ecu_driving_mode=2,
                    tv_4wd_mode=0,
                    alpha_drv=self.base_alpha_drv,
                    n_laps=1,
                    use_3d_kinematics=True,
                    mode_overrides=mo,
                    save_csv_path=csv_p,
                    grip_profile=self.grip_profile,
                )

            cones_hit = 0
            min_clear_m = 0.0
            if check_cones and fsg_res.telemetry is not None:
                sw = verify_swept_path_4corners(self.trk_fsg, self.rl_fsg, fsg_res.telemetry)
                cones_hit = int(sw["corners_cones_hit"])
                min_clear_m = float(sw["min_14dof_cone_clearance_m"])

        p_fsg_max = float(fsg_res.summary.get("p_dc_max_kw", 0.0))
        p_fsg_min = float(fsg_res.summary.get("p_chg_min_kw", 0.0))
        p_max_all = max(p_max_probe, p_fsg_max)
        p_min_all = min(p_min_probe, p_fsg_min)

        if p_max_all > HV_MAX_KW or p_min_all < HV_MIN_KW:
            violations.append(f"hv_fsg(P=[{p_min_all:.2f},{p_max_all:.2f}])")
        if not bool(fsg_res.valid):
            violations.append(
                f"fsg_invalid(off={fsg_res.summary.get('off_track_ticks', -1)}, beta={fsg_res.summary.get('max_abs_beta_deg', -1)})"
            )
        if cones_hit > 0:
            violations.append(f"cones_hit({cones_hit})")

        hp_s = float(hp["time_s"])
        sk_s = float(sk["t_lap_s"])
        fsg_s = float(fsg_res.lap_time_s)

        if self.ref_hp_s is None:
            self.ref_hp_s = hp_s
            self.ref_sk_s = sk_s
            self.ref_fsg_s = fsg_s
            self.base_alpha_drv = float(fsg_res.summary.get("alpha_drv", 0.96))

        feasible = len(violations) == 0
        # Weighted normalized multi-scenario time cost (lower is better)
        cost = (
            self.w_hp * (hp_s - self.ref_hp_s)
            + self.w_sk * (sk_s - self.ref_sk_s)
            + self.w_fsg * (fsg_s - self.ref_fsg_s)
        ) if feasible else float("inf")

        return {
            "label": label,
            "mode_overrides": mo,
            "feasible": feasible,
            "cost": round(float(cost), 5) if math.isfinite(cost) else float("inf"),
            "violations": violations,
            "metrics": {
                "hairpin_time_s": round(hp_s, 4),
                "hairpin_min_vx_ms": round(float(hp["min_vx_apex_ms"]), 3),
                "hairpin_max_beta_deg": round(float(hp["max_beta_deg"]), 3),
                "skidpad_lap_s": round(sk_s, 4),
                "skidpad_vx_ss_ms": round(float(sk["vx_ss_ms"]), 3),
                "skidpad_ay_ss_g": round(float(sk["ay_ss_g"]), 3),
                "skidpad_max_beta_deg": round(float(sk["max_beta_deg"]), 3),
                "slalom_max_beta_deg": round(float(sl["max_beta_deg"]), 3),
                "slalom_rec_s": round(float(sl["rec_time_s"]), 3),
                "launch_worst_t50_s": round(float(ln["worst_t50_s"]), 3),
                "fsg_lap_s": round(fsg_s, 4),
                "fsg_valid": bool(fsg_res.valid),
                "fsg_alpha_drv": round(float(fsg_res.summary.get("alpha_drv", 0.98)), 4),
                "fsg_max_beta_deg": round(float(fsg_res.summary.get("max_abs_beta_deg", 0.0)), 3),
                "cones_hit": cones_hit,
                "min_cone_clear_m": round(min_clear_m, 3),
                "p_max_kw": round(p_max_all, 3),
                "p_min_kw": round(p_min_all, 3),
            },
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }


def _vec_from_overrides(mo: Dict[str, Dict[str, float]], spec: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([float(mo[s["mode"]][s["key"]]) for s in spec], dtype=np.float64)


def _format_candidate_log(cand: Dict[str, Any]) -> str:
    m = cand["metrics"]
    return (
        f"  [{cand['label']:16s}] feas={str(cand['feasible']):5s} cost={cand['cost']:+.4f} | "
        f"HP={m['hairpin_time_s']:.3f}s SK={m['skidpad_lap_s']:.3f}s ({m['skidpad_vx_ss_ms']:.2f}m/s) "
        f"FSG={m['fsg_lap_s']:.3f}s | slalom_b={m['slalom_max_beta_deg']:.2f}deg "
        f"t50={m['launch_worst_t50_s']:.3f}s cones={m['cones_hit']} "
        f"P=[{m['p_min_kw']:.1f},{m['p_max_kw']:.1f}]kW ({cand['elapsed_s']:.1f}s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop 14-DOF TV optimizer (FSG_REAL_TRACK)")
    parser.add_argument("--grip-profile", type=str, default="fsg_real", choices=["fsg_real", "ttc_lab"])
    parser.add_argument("--n-lhs", type=int, default=16, help="Latin Hypercube initial samples")
    parser.add_argument("--n-es-gens", type=int, default=4, help="Evolutionary CMA-like generations")
    parser.add_argument("--pop-size", type=int, default=8, help="Offspring per ES generation")
    parser.add_argument("--n-powell", type=int, default=12, help="Local Nelder-Mead/Powell refinement evals")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Godot worker threads")
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "closed_loop_opt_report.json"))
    args = parser.parse_args()

    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FSG_2023_TRACK_PATH.exists():
        build_fsg_main()

    print(f"[OPT] Loading FSG 2023 track + GGV ({args.grip_profile})...", flush=True)
    trk_fsg = Track.load_json(FSG_2023_TRACK_PATH)
    ggv_auto = extract_or_load_ggv(
        label="tv4_auto",
        ecu_driving_mode=2,
        tv_4wd_mode=0,
        use_3d_kinematics=True,
        grip_profile=args.grip_profile,
    )
    rl_fsg = optimize_mintime_racing_line(trk_fsg, ggv=ggv_auto)
    qss_fsg = solve_qss(rl_fsg.racing_track, ggv_auto)

    base_cfg = json.loads(TV4_CFG_PATH.read_text(encoding="utf-8"))
    spec = _default_param_spec(base_cfg)
    dim = len(spec)
    lo_vec = np.array([s["lo"] for s in spec], dtype=np.float64)
    hi_vec = np.array([s["hi"] for s in spec], dtype=np.float64)
    nom_vec = np.array([s["nom"] for s in spec], dtype=np.float64)
    span_vec = np.maximum(hi_vec - lo_vec, 1e-6)

    print(f"[OPT] Optimizing {dim} parameters: {[s['mode'] + '.' + s['key'] for s in spec]}", flush=True)

    with tempfile.TemporaryDirectory(prefix="mirena_opt_gd_") as gd_tmp:
        gd_path = Path(gd_tmp) / "maneuver_probe.gd"
        gd_path.write_text(GODOT_MANEUVER_PROBE_GD, encoding="utf-8")

        evaluator = ClosedLoopEvaluator(
            spec=spec,
            trk_fsg=trk_fsg,
            rl_fsg=rl_fsg,
            qss_fsg=qss_fsg,
            gd_path=gd_path,
            grip_profile=args.grip_profile,
        )

        # 0. Baseline evaluation (with full alpha search)
        print("[OPT] Stage 0: Evaluating baseline...", flush=True)
        baseline = evaluator.evaluate(nom_vec, "baseline", full_alpha_search=True, check_cones=True)
        print(_format_candidate_log(baseline), flush=True)
        history: List[Dict[str, Any]] = [baseline]

        # 1. Stage 1: Latin Hypercube exploration (parallel)
        print(f"[OPT] Stage 1: Latin Hypercube ({args.n_lhs} samples, {args.workers} workers)...", flush=True)
        sampler = LatinHypercube(d=dim, seed=SEED)
        u_lhs = sampler.random(n=args.n_lhs)
        lhs_vecs = [lo_vec + u_lhs[i] * span_vec for i in range(args.n_lhs)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(evaluator.evaluate, lhs_vecs[i], f"lhs_{i:02d}", False, True)
                for i in range(args.n_lhs)
            ]
            for fut in concurrent.futures.as_completed(futs):
                cand = fut.result()
                history.append(cand)
                print(_format_candidate_log(cand), flush=True)

        # 2. Stage 2: Adaptive (mu, lambda) CMA-like Evolutionary Search in normalized [0, 1]^d space
        print(
            f"[OPT] Stage 2: Adaptive ES ({args.n_es_gens} gens x {args.pop_size} pop, {args.workers} workers)...",
            flush=True,
        )
        rng = np.random.default_rng(SEED + 100)
        feas_hist = [h for h in history if h["feasible"]]
        if feas_hist:
            feas_hist.sort(key=lambda c: c["cost"])
            mean_u = (_vec_from_overrides(feas_hist[0]["mode_overrides"], spec) - lo_vec) / span_vec
        else:
            mean_u = (nom_vec - lo_vec) / span_vec

        sigma = 0.16
        cov_diag = np.ones(dim, dtype=np.float64)

        for gen in range(args.n_es_gens):
            offspring_u: List[np.ndarray] = []
            for _ in range(args.pop_size):
                step = rng.normal(0.0, sigma * np.sqrt(cov_diag), size=dim)
                u_cand = np.clip(mean_u + step, 0.0, 1.0)
                offspring_u.append(u_cand)

            gen_results: List[Tuple[np.ndarray, Dict[str, Any]]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                fut_map = {
                    ex.submit(
                        evaluator.evaluate,
                        lo_vec + offspring_u[i] * span_vec,
                        f"es_g{gen}_i{i:02d}",
                        False,
                        True,
                    ): offspring_u[i]
                    for i in range(args.pop_size)
                }
                for fut in concurrent.futures.as_completed(fut_map):
                    cand = fut.result()
                    history.append(cand)
                    gen_results.append((fut_map[fut], cand))
                    print(_format_candidate_log(cand), flush=True)

            feas_gen = [(u, c) for (u, c) in gen_results if c["feasible"]]
            if feas_gen:
                feas_gen.sort(key=lambda item: item[1]["cost"])
                mu_k = max(2, min(4, len(feas_gen)))
                elites = feas_gen[:mu_k]
                w_rank = np.array([math.log(mu_k + 0.5) - math.log(r + 1.0) for r in range(len(elites))])
                w_rank /= np.sum(w_rank)
                new_mean_u = np.zeros(dim, dtype=np.float64)
                for w_i, (u_e, _) in zip(w_rank, elites):
                    new_mean_u += w_i * u_e
                # Update diagonal covariance + step-size adaptation
                diff_sq = np.zeros(dim, dtype=np.float64)
                for w_i, (u_e, _) in zip(w_rank, elites):
                    diff_sq += w_i * ((u_e - mean_u) / max(sigma, 1e-4)) ** 2
                cov_diag = np.clip(0.70 * cov_diag + 0.30 * diff_sq, 0.15, 2.5)
                mean_u = np.clip(new_mean_u, 0.0, 1.0)
                sigma = max(0.04, sigma * 0.82)
            else:
                # Contract toward best feasible point if generation had no feasible points
                all_feas = sorted([h for h in history if h["feasible"]], key=lambda c: c["cost"])
                if all_feas:
                    mean_u = (_vec_from_overrides(all_feas[0]["mode_overrides"], spec) - lo_vec) / span_vec
                sigma = max(0.03, sigma * 0.65)

        # 3. Stage 3: Local Nelder-Mead refinement around best feasible candidate
        all_feas = sorted([h for h in history if h["feasible"]], key=lambda c: c["cost"])
        if all_feas and args.n_powell > 0:
            print(f"[OPT] Stage 3: Local Nelder-Mead refinement ({args.n_powell} max evals)...", flush=True)
            x0_u = (_vec_from_overrides(all_feas[0]["mode_overrides"], spec) - lo_vec) / span_vec
            nm_counter = {"n": 0}

            def _nm_obj(u_x: np.ndarray) -> float:
                u_c = np.clip(u_x, 0.0, 1.0)
                idx_n = nm_counter["n"]
                nm_counter["n"] += 1
                cand = evaluator.evaluate(lo_vec + u_c * span_vec, f"nm_{idx_n:02d}", False, True)
                history.append(cand)
                print(_format_candidate_log(cand), flush=True)
                return float(cand["cost"]) if cand["feasible"] else 1e6

            init_simplex = np.tile(x0_u, (dim + 1, 1))
            for d_i in range(dim):
                delta_u = 0.05 if x0_u[d_i] < 0.90 else -0.05
                init_simplex[d_i + 1, d_i] = np.clip(x0_u[d_i] + delta_u, 0.0, 1.0)

            minimize(
                _nm_obj,
                x0_u,
                method="Nelder-Mead",
                options={"initial_simplex": init_simplex, "maxfev": args.n_powell, "xatol": 1e-3, "fatol": 1e-3},
            )

        # 4. Stage 4: Full alpha_drv verification of Top-3 unique feasible candidates
        all_feas = sorted([h for h in history if h["feasible"]], key=lambda c: c["cost"])
        verified_top: List[Dict[str, Any]] = []
        seen_keys = set()
        for cand in all_feas:
            key = json.dumps(cand["mode_overrides"], sort_keys=True)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if len(verified_top) >= 3:
                break
            v_vec = _vec_from_overrides(cand["mode_overrides"], spec)
            print(f"[OPT] Stage 4: Full-alpha verification of {cand['label']}...", flush=True)
            ver = evaluator.evaluate(v_vec, f"verified_{len(verified_top)}", full_alpha_search=True, check_cones=True)
            print(_format_candidate_log(ver), flush=True)
            if ver["feasible"]:
                verified_top.append(ver)

        verified_top.sort(key=lambda c: c["cost"])
        best = verified_top[0] if verified_top else baseline

        # Build Pareto archive sorted by cost (for automatic robustness backoff if needed)
        pareto_archive = sorted(
            [h for h in history if h["feasible"]],
            key=lambda c: c["cost"],
        )[:15]

        report = {
            "seed": SEED,
            "grip_profile": args.grip_profile,
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "param_spec": spec,
            "hard_constraints": {
                "hv_max_kw": HV_MAX_KW,
                "hv_min_kw": HV_MIN_KW,
                "slalom_max_beta_deg": SLALOM_MAX_BETA_DEG,
                "slalom_max_rec_s": SLALOM_MAX_REC_S,
                "launch_max_t50_s": LAUNCH_MAX_T50_S,
                "cones_hit": 0,
            },
            "baseline": baseline,
            "best_verified": best,
            "verified_top3": verified_top,
            "pareto_archive": pareto_archive,
            "n_total_evals": len(history),
            "n_feasible_evals": sum(1 for h in history if h["feasible"]),
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"\n[OPT DONE] elapsed={report['elapsed_s']:.1f}s | evals={report['n_total_evals']} "
            f"(feasible={report['n_feasible_evals']}) | best={best['label']} cost={best['cost']:+.4f} -> {out_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
