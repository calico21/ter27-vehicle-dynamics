#!/usr/bin/env python3
"""
Compare Python validated TeR26 CasADi model against Godot 14-DOF TeR26
(Doble Horquilla 3D vs Parametric A/B comparison).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

CORR_DIR = Path(__file__).resolve().parent
LAB_DIR = CORR_DIR.parents[1]
REPO_ROOT = LAB_DIR.parent
ORACLE_DIR = LAB_DIR / "oracle"
RESULTS_DIR = LAB_DIR / "results" / "ter26_extended_correlation"
SIM_PROJECT_DIR = REPO_ROOT / "src" / "mirena_sim"

for _p in (str(REPO_ROOT), str(ORACLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vd_correlation_lab.config.paths import GODOT_BIN  # noqa: E402
from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import TeR26HighFidelityCasADi, load_ter26_params  # noqa: E402


def run_python_maneuver(
    vx0: float,
    delta_rad: float,
    throttle: float,
    duration_s: float,
    use_3d: bool,
    sysid_theta: Dict[str, float],
    vx_trace: np.ndarray | None = None,
) -> Dict[str, Any]:
    dt = 0.005
    n = int(round(duration_s / dt)) + 1
    t_s = np.linspace(0.0, duration_s, n)
    delta_arr = np.full(n, delta_rad, dtype=np.float64)
    params = load_ter26_params(use_cad_3d=True)
    mdl = TeR26HighFidelityCasADi(params=params, use_3d_kinematics=use_3d, use_thermal_coupling=False)
    vx_meas = vx_trace if vx_trace is not None else np.full(n, vx0, dtype=np.float64)
    ax_arr = np.array(
        [mdl.map_pedals_to_ax_cmd(float(vx_meas[k]), throttle, 0.0, 120.0) for k in range(n)],
        dtype=np.float64,
    )
    wz_meas = np.zeros(n, dtype=np.float64)
    t4 = np.full((n, 4), 55.0, dtype=np.float64)

    out = mdl.simulate_trajectory(
        t_s=t_s,
        delta_rad=delta_arr,
        ax_cmd_drive=ax_arr,
        vx_meas=vx_meas,
        wz_meas=wz_meas,
        t_tire_4=t4,
        theta=sysid_theta,
        use_measured_vx=(vx_trace is not None),
    )
    return {
        "vx_end": float(out["vx_sim"][-1]),
        "ay_end": float(out["ay_sim"][-1]),
        "wz_end": float(out["wz_sim"][-1]),
        "ay_peak": float(np.max(np.abs(out["ay_sim"]))),
        "wz_peak": float(np.max(np.abs(out["wz_sim"]))),
        "beta_end_deg": float(math.degrees(math.atan2(out["vy_sim"][-1], max(out["vx_sim"][-1], 0.5)))),
    }


GODOT_AB_RUNNER_GD = r"""extends SceneTree

const Full14DofScript = preload("res://Scenes/Vehicle/Dynamics/Models/full_14dof/Full14DofModel.gd")
const VehicleInputsScript = preload("res://Scenes/Vehicle/Inputs/VehicleInputs.gd")
const DoubleWishboneScript = preload("res://Scenes/Vehicle/Dynamics/Suspension/DoubleWishboneKinematics.gd")

func _run_case(use_3d: bool, vx0: float, delta_rad: float, throttle: float, duration_s: float) -> Dictionary:
	var m := Full14DofScript.new()
	m.load_vehicle("ter26_rwd", "default")
	m.config.use_3d_kinematics = use_3d
	m.config.tire_initial_temp_c = 55.0
	m.reset()
	m.suspension.sync_from_config(m.config)
	for w in m.wheels:
		w.omega = vx0 / m.config.wheel_radius
	m.chassis_state.linear_velocity = Vector3(vx0, 0.0, 0.0)

	var inp := VehicleInputsScript.new()
	inp.drive_mode = m.config.drive_mode
	inp.ecu_driving_mode = 0
	inp.throttle = throttle
	inp.brake = 0.0
	inp.steer_angle = delta_rad

	var dt := 0.0025
	var steps := int(round(duration_s / dt))
	var ay_peak := 0.0
	var wz_peak := 0.0
	var vx_trace: Array[float] = [vx0]
	var last_res := {}
	for k in range(steps):
		last_res = m.step_standalone(inp, dt, 0.0)
		var net_f: Vector3 = last_res["forces"]["net_force_ros"]
		var ay_k := net_f.y / m.config.mass
		var wz_k := m.chassis_state.angular_velocity.z
		ay_peak = maxf(ay_peak, absf(ay_k))
		wz_peak = maxf(wz_peak, absf(wz_k))
		if (k + 1) % 2 == 0:
			vx_trace.append(m.chassis_state.linear_velocity.x)

	var net_f_end: Vector3 = last_res["forces"]["net_force_ros"]
	var ay_end := net_f_end.y / m.config.mass
	var vx_end := m.chassis_state.linear_velocity.x
	var vy_end := m.chassis_state.linear_velocity.y
	var wz_end := m.chassis_state.angular_velocity.z
	var ksum := DoubleWishboneScript.compute_telemetry_summary(m.suspension.corners, delta_rad, m.config, m.suspension.kinematics_3d)
	return {
		"vx_end": vx_end,
		"ay_end": ay_end,
		"wz_end": wz_end,
		"ay_peak": ay_peak,
		"wz_peak": wz_peak,
		"beta_end_deg": rad_to_deg(atan2(vy_end, maxf(vx_end, 0.5))),
		"camber_fl_deg": float(ksum["camber_fl_deg"]),
		"camber_fr_deg": float(ksum["camber_fr_deg"]),
		"toe_fl_deg": float(ksum["toe_fl_deg"]),
		"toe_fr_deg": float(ksum["toe_fr_deg"]),
		"ackermann_pct": float(ksum["ackermann_pct"]),
		"mr_front": float(ksum["mr_front"]),
		"mr_rear": float(ksum["mr_rear"]),
		"lltd_pct": float(ksum["lltd_pct"]),
		"vx_trace": vx_trace,
	}

func _initialize() -> void:
	var bump_3d := {}
	var bump_par := {}
	for use_3d in [true, false]:
		var m := Full14DofScript.new()
		m.load_vehicle("ter26_rwd", "default")
		m.config.use_3d_kinematics = use_3d
		m.suspension.sync_from_config(m.config)
		m.suspension.evaluate_from_corners([0.025, -0.020, 0.010, -0.010], [0.0, 0.0, 0.0, 0.0], [true, true, true, true], m.config, 0.14)
		var ks := DoubleWishboneScript.compute_telemetry_summary(m.suspension.corners, 0.14, m.config, m.suspension.kinematics_3d)
		if use_3d:
			bump_3d = ks
		else:
			bump_par = ks

	var payload := {
		"step_steer_3d": _run_case(true, 15.0, 0.06, 0.25, 2.0),
		"step_steer_param": _run_case(false, 15.0, 0.06, 0.25, 2.0),
		"skidpad_3d": _run_case(true, 10.5, 0.12, 0.20, 3.0),
		"skidpad_param": _run_case(false, 10.5, 0.12, 0.20, 3.0),
		"bump_steer_3d": bump_3d,
		"bump_steer_param": bump_par,
	}
	print("JSON_RESULT:" + JSON.stringify(payload))
	quit(0)
"""


def run_godot_ab() -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".gd", delete=False, dir="/tmp") as tf:
        tf.write(GODOT_AB_RUNNER_GD)
        script_path = tf.name
    try:
        proc = subprocess.run(
            [str(GODOT_BIN), "--headless", "--path", str(SIM_PROJECT_DIR), "-s", script_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("JSON_RESULT:"):
                return json.loads(line[len("JSON_RESULT:") :])
        raise RuntimeError(f"JSON_RESULT not found in Godot stdout:\n{proc.stdout}\n{proc.stderr}")
    finally:
        Path(script_path).unlink(missing_ok=True)


def main() -> None:
    sysid_data = json.loads((RESULTS_DIR / "ter26_sysid_params.json").read_text(encoding="utf-8"))
    sysid_theta = {k: float(v) for k, v in sysid_data["identified_params"].items()}

    print("=== Running Godot 14-DOF TeR26 A/B (3D Double Wishbone vs Parametric) ===", flush=True)
    gd = run_godot_ab()

    print("=== Running Python TeR26HighFidelityCasADi (SysID params) ===", flush=True)
    vx_step = np.array(gd["step_steer_3d"].pop("vx_trace"), dtype=np.float64)
    gd["step_steer_param"].pop("vx_trace", None)
    vx_skid = np.array(gd["skidpad_3d"].pop("vx_trace"), dtype=np.float64)
    gd["skidpad_param"].pop("vx_trace", None)

    py_step_3d = run_python_maneuver(15.0, 0.06, 0.25, 2.0, True, sysid_theta, vx_trace=vx_step)
    py_skid_3d = run_python_maneuver(10.5, 0.12, 0.20, 3.0, True, sysid_theta, vx_trace=vx_skid)

    report = {
        "python_step_steer_3d": py_step_3d,
        "python_skidpad_3d": py_skid_3d,
        "godot": gd,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
